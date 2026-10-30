"""Run the small batch command through native DeepEval and local exports."""

import csv
import json

import pytest

from meeting_eval import cli, create_metric


def input_folders(tmp_path):
    transcripts, summaries = tmp_path / "transcripts", tmp_path / "summaries"
    transcripts.mkdir()
    summaries.mkdir()
    return transcripts, summaries


def use_offline_judge(monkeypatch, judge):
    metrics = []

    def factory(**kwargs):
        kwargs["model"] = judge
        metric = create_metric(**kwargs)
        metrics.append(metric)
        return metric

    monkeypatch.setattr(cli, "create_metric", factory)
    return metrics


def test_each_meeting_gets_fresh_native_questions(tmp_path, monkeypatch, judge):
    transcripts, summaries = input_folders(tmp_path)
    for meeting_id, marker in (("alpha", "Alpha"), ("beta", "Beta")):
        (transcripts / f"{meeting_id}.txt").write_text(f"Source{marker}: original discussion.")
        (summaries / f"{meeting_id}.md").write_text(f"Summary{marker}: meeting account.")
    metrics = use_offline_judge(monkeypatch, judge)

    results = cli.evaluate_batch(transcripts, summaries, n=2)

    assert [row["meeting_id"] for row in results] == ["alpha", "beta"]
    assert all(row["status"] == "completed" for row in results)
    assert len(metrics) == 2 and metrics[0] is not metrics[1]
    assert all(metric.async_mode is False for metric in metrics)
    assert "SourceAlpha" in results[0]["assessment_questions"][0]
    assert "SourceBeta" in results[1]["assessment_questions"][0]


def test_batch_preserves_successes_and_reports_input_and_judge_errors(tmp_path, monkeypatch, judge):
    transcripts, summaries = input_folders(tmp_path)
    for name, source in (("alpha", "SourceAlpha"), ("beta", "SourceFailure"), ("blank", " \n")):
        (transcripts / f"{name}.txt").write_text(source)
        (summaries / f"{name}.md").write_text("SummaryAlpha")
    (transcripts / "missing_summary.txt").write_text("SourceAlpha")
    (summaries / "missing_transcript.md").write_text("SummaryAlpha")
    (transcripts / "duplicate.txt").write_text("SourceAlpha")
    (transcripts / "duplicate.md").write_text("SourceAlpha")
    (summaries / "duplicate.md").write_text("SummaryAlpha")
    (transcripts / "unreadable.txt").write_bytes(b"\xff\xfe\xff")
    (summaries / "unreadable.md").write_text("SummaryAlpha")
    (summaries / "ignored.csv").write_text("Unsupported extension")
    judge.fail_source = "SourceFailure"
    use_offline_judge(monkeypatch, judge)

    results = {row["meeting_id"]: row for row in cli.evaluate_batch(transcripts, summaries, n=2)}

    assert set(results) == {"alpha", "beta", "blank", "duplicate", "missing_summary", "missing_transcript", "unreadable"}
    assert results["alpha"]["status"] == "completed"
    for name, row in results.items():
        if name != "alpha":
            assert row["status"] == "error" and row["error"]
            assert row["score"] is row["score_breakdown"] is row["reason"] is None
            assert row["alignment_verdicts"] == row["coverage_verdicts"] == []


def test_cli_exports_native_details_and_does_not_gate_summary_quality(tmp_path, monkeypatch, judge):
    transcripts, summaries = input_folders(tmp_path)
    (transcripts / "alpha.txt").write_text("\ufeffSourceAlpha\r\nOriginal discussion.", encoding="utf-8")
    (summaries / "alpha.md").write_text("SummaryAlpha: meeting account.")
    context = tmp_path / "business.md"
    context.write_text("AudienceMarker: advisor preparation.")
    output = tmp_path / "output"
    use_offline_judge(monkeypatch, judge)

    status = cli.main([
        "--transcripts", str(transcripts), "--summaries", str(summaries),
        "--output", str(output), "--business-context", str(context), "--n", "2",
    ])

    assert status == 0
    run_dir, = output.iterdir()
    payload = json.loads((run_dir / "results.json").read_text())
    row, = payload["results"]
    assert row["score"] == pytest.approx(1 / 3)
    assert row["score_breakdown"] == {"Alignment": pytest.approx(1 / 3), "Coverage": 0.5}
    assert row["alignment_verdicts"][1]["verdict"] == "no"
    assert row["coverage_verdicts"][1]["summary_verdict"] == "no"
    assert payload["settings"]["n"] == 2
    assert payload["settings"]["business_context"] == context.read_text()
    with (run_dir / "scores.csv").open(newline="") as file:
        exported, = csv.DictReader(file)
    assert float(exported["alignment"]) == pytest.approx(1 / 3)
    assert float(exported["coverage"]) == 0.5
    assert (run_dir / "report.html").is_file()
    truth_prompt = next(prompt for stage, prompt in judge.prompts if stage == "Truths")
    assert "SourceAlpha\nOriginal discussion." in truth_prompt
    assert "\ufeff" not in truth_prompt and "\r" not in truth_prompt


def test_cli_returns_failure_for_unmatched_inputs(tmp_path):
    transcripts, summaries = input_folders(tmp_path)
    (transcripts / "missing.txt").write_text("SourceAlpha")
    assert cli.main(["--transcripts", str(transcripts), "--summaries", str(summaries),
                     "--output", str(tmp_path / "output")]) == 1


def test_empty_or_missing_input_folders_fail_before_scoring(tmp_path):
    transcripts, summaries = input_folders(tmp_path)
    with pytest.raises(ValueError):
        cli.evaluate_batch(transcripts, summaries)
    with pytest.raises(ValueError):
        cli.evaluate_batch(tmp_path / "absent", summaries)


def test_cli_rejects_invalid_question_count_and_missing_folders(tmp_path):
    transcripts, summaries = input_folders(tmp_path)
    for extra in (["--n", "0"], []):
        with pytest.raises(SystemExit) as exc:
            cli.main(["--transcripts", str(transcripts), "--summaries", str(summaries), *extra])
        assert exc.value.code == 2
