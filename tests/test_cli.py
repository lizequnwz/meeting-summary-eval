"""Exercise frozen questions, durable batches, and offline report commands."""

import asyncio
from collections import Counter
import json

import pytest

from meeting_eval import cli, evaluation, metrics, questions
from meeting_eval.evaluation import load_evaluation


def input_folders(tmp_path):
    transcripts, summaries = tmp_path / "transcripts", tmp_path / "summaries"
    transcripts.mkdir()
    summaries.mkdir()
    return transcripts, summaries


def add_meeting(transcripts, summaries, meeting_id="alpha", marker="Alpha"):
    (transcripts / f"{meeting_id}.txt").write_text(f"Source{marker}: original discussion.")
    (summaries / f"{meeting_id}.md").write_text(f"Summary{marker}: meeting account.")


def offline_judge(monkeypatch, judge):
    monkeypatch.setattr(questions, "resolve_model", lambda model=None: judge)
    monkeypatch.setattr(metrics, "resolve_model", lambda model=None: judge)


def arguments(tmp_path, transcripts, summaries):
    return ["evaluate", "--transcripts", str(transcripts), "--summaries", str(summaries),
            "--questions", str(tmp_path / "questions"), "--output", str(tmp_path / "output"), "--n", "2"]


def run_files(tmp_path):
    return sorted((tmp_path / "output").glob("*/results.json"))


def test_each_meeting_is_frozen_and_reused_across_summary_versions(tmp_path, monkeypatch, judge):
    transcripts, summaries = input_folders(tmp_path)
    for meeting_id, marker in (("alpha", "Alpha"), ("beta", "Beta")):
        add_meeting(transcripts, summaries, meeting_id, marker)
    offline_judge(monkeypatch, judge)
    args = arguments(tmp_path, transcripts, summaries)

    assert cli.main(args) == 0
    first = load_evaluation(run_files(tmp_path)[0])
    assert Counter(stage for stage, _ in judge.prompts)["Questions"] == 2
    assert all(row["question_generation_performed"] for row in first["results"])
    assert "SourceAlpha" in first["results"][0]["question_set"]["questions"][0]
    assert "SourceBeta" in first["results"][1]["question_set"]["questions"][0]
    (summaries / "alpha.md").write_text("SummaryAlpha: revised meeting account.")

    assert cli.main(args) == 0
    second = load_evaluation(run_files(tmp_path)[1])
    assert Counter(stage for stage, _ in judge.prompts)["Questions"] == 2
    assert all(not row["question_generation_performed"] for row in second["results"])
    assert [row["question_set"] for row in first["results"]] == [row["question_set"] for row in second["results"]]
    assert first["results"][0]["inputs"]["summary"]["sha256"] != second["results"][0]["inputs"]["summary"]["sha256"]


def test_batch_preserves_successes_and_input_and_judge_errors(tmp_path, monkeypatch, judge):
    transcripts, summaries = input_folders(tmp_path)
    for meeting_id, marker in (("alpha", "Alpha"), ("beta", "Failure"), ("blank", "Alpha")):
        add_meeting(transcripts, summaries, meeting_id, marker)
    (transcripts / "blank.txt").write_text(" \n")
    (transcripts / "missing_summary.txt").write_text("SourceAlpha")
    (summaries / "missing_transcript.md").write_text("SummaryAlpha")
    (transcripts / "duplicate.txt").write_text("SourceAlpha")
    (transcripts / "duplicate.md").write_text("SourceAlpha")
    (summaries / "duplicate.md").write_text("SummaryAlpha")
    (transcripts / "unreadable.txt").write_bytes(b"\xff\xfe\xff")
    (summaries / "unreadable.md").write_text("SummaryAlpha")
    (summaries / "ignored.csv").write_text("Unsupported extension")
    judge.fail_source = "SourceFailure"
    offline_judge(monkeypatch, judge)

    assert cli.main(arguments(tmp_path, transcripts, summaries)) == 1

    path, = run_files(tmp_path)
    run = load_evaluation(path)
    rows = {row["meeting_id"]: row for row in run["results"]}
    assert set(rows) == {"alpha", "beta", "blank", "duplicate", "missing_summary", "missing_transcript", "unreadable"}
    assert rows["alpha"]["status"] == "completed"
    for name, row in rows.items():
        if name != "alpha":
            assert row["status"] == "error" and row["error"]
            assert row["scores"] is row["counts"] is row["reason"] is None
            assert row["alignment"] == row["coverage"] == []
    assert (path.parent / "report.xlsx").is_file()
    assert (path.parent / "report.html").is_file()


def test_cli_exports_native_details_and_does_not_gate_quality(tmp_path, monkeypatch, judge):
    transcripts, summaries = input_folders(tmp_path)
    add_meeting(transcripts, summaries)
    (transcripts / "alpha.txt").write_text("\ufeffSourceAlpha\r\nOriginal discussion.", encoding="utf-8")
    context = tmp_path / "business.md"
    context.write_text("AudienceMarker: advisor preparation.")
    offline_judge(monkeypatch, judge)

    assert cli.main([*arguments(tmp_path, transcripts, summaries), "--business-context", str(context)]) == 0

    path, = run_files(tmp_path)
    row, = load_evaluation(path)["results"]
    assert row["scores"] == {"alignment": pytest.approx(1 / 3), "coverage": 0.5, "overall": pytest.approx(1 / 3)}
    assert row["counts"] == {"alignment_supported": 1, "alignment_total": 3,
                             "coverage_covered": 1, "coverage_eligible": 2, "coverage_total": 2}
    assert row["alignment"][1]["verdict"] == "no"
    assert row["coverage"][1]["outcome"] == "missing"
    assert row["question_set"]["business_context"] == context.read_text()
    truth_prompt = next(prompt for stage, prompt in judge.prompts if stage == "Truths")
    assert "SourceAlpha\nOriginal discussion." in truth_prompt
    assert "\ufeff" not in truth_prompt and "\r" not in truth_prompt


def test_stale_questions_require_explicit_regeneration_before_scoring(tmp_path, monkeypatch, judge):
    transcripts, summaries = input_folders(tmp_path)
    add_meeting(transcripts, summaries)
    offline_judge(monkeypatch, judge)
    args = arguments(tmp_path, transcripts, summaries)
    assert cli.main(args) == 0
    old = json.loads((tmp_path / "questions" / "alpha.json").read_text())
    previous_calls = len(judge.prompts)
    (transcripts / "alpha.txt").write_text("SourceAlpha: changed original discussion.")

    assert cli.main(args) == 1
    assert len(judge.prompts) == previous_calls
    assert json.loads((tmp_path / "questions" / "alpha.json").read_text()) == old
    assert cli.main([*args, "--regenerate-questions"]) == 0
    new = json.loads((tmp_path / "questions" / "alpha.json").read_text())
    assert new["question_set_id"] != old["question_set_id"]
    assert load_evaluation(run_files(tmp_path)[0])["results"][0]["question_set"] == old


def test_invalid_output_fails_before_any_model_calls(tmp_path, monkeypatch, judge):
    transcripts, summaries = input_folders(tmp_path)
    add_meeting(transcripts, summaries)
    offline_judge(monkeypatch, judge)
    (tmp_path / "output").write_text("This is a file.")
    with pytest.raises(SystemExit) as exc:
        cli.main(arguments(tmp_path, transcripts, summaries))
    assert exc.value.code == 2
    assert not judge.prompts


def test_report_command_needs_only_saved_json(tmp_path, monkeypatch, judge):
    transcripts, summaries = input_folders(tmp_path)
    add_meeting(transcripts, summaries)
    offline_judge(monkeypatch, judge)
    assert cli.main(arguments(tmp_path, transcripts, summaries)) == 0
    path, = run_files(tmp_path)
    original = path.read_bytes()
    for directory in (transcripts, summaries, tmp_path / "questions"):
        for file in directory.iterdir():
            file.unlink()
        directory.rmdir()

    def forbidden(*args, **kwargs):
        raise AssertionError("Report command must not make model calls")

    monkeypatch.setattr(metrics, "resolve_model", forbidden)
    monkeypatch.setattr(questions, "resolve_model", forbidden)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cli.main(["report", str(path), "--output", str(tmp_path / "export")]) == 0
    assert (tmp_path / "export" / "report.xlsx").exists()
    assert path.read_bytes() == original


@pytest.mark.parametrize("report_fails", [False, True])
def test_interruption_checkpoints_successes_and_marks_remaining_unprocessed(tmp_path, monkeypatch, judge, report_fails):
    transcripts, summaries = input_folders(tmp_path)
    for meeting_id, marker in (("alpha", "Alpha"), ("beta", "Beta")):
        add_meeting(transcripts, summaries, meeting_id, marker)
    offline_judge(monkeypatch, judge)
    original_evaluate = evaluation.evaluate_meeting

    async def interrupt_second(meeting_id, *args, **kwargs):
        if meeting_id == "beta":
            # Prove the first result was saved before entering the second evaluation.
            saved = load_evaluation(run_files(tmp_path)[0])
            assert saved["results"][0]["status"] == "completed"
            raise KeyboardInterrupt
        return await original_evaluate(meeting_id, *args, **kwargs)

    monkeypatch.setattr(cli, "evaluate_meeting", interrupt_second)
    if report_fails:
        def failed_report(*args, **kwargs):
            raise OSError("Report destination failed during interruption")

        monkeypatch.setattr(cli, "export_reports", failed_report)
    assert cli.main(arguments(tmp_path, transcripts, summaries)) == 130
    path, = run_files(tmp_path)
    run = load_evaluation(path)
    assert run["status"] == "interrupted"
    assert [row["status"] for row in run["results"]] == ["completed", "unprocessed"]
    assert run["results"][1]["question_set"] is not None
    assert (path.parent / "report.xlsx").exists() is not report_fails


def test_report_failure_preserves_canonical_evaluation(tmp_path, monkeypatch, judge):
    transcripts, summaries = input_folders(tmp_path)
    add_meeting(transcripts, summaries)
    offline_judge(monkeypatch, judge)

    def failed(*args, **kwargs):
        raise OSError("Report destination failed")

    monkeypatch.setattr(cli, "export_reports", failed)
    with pytest.raises(SystemExit) as exc:
        cli.main(arguments(tmp_path, transcripts, summaries))
    assert exc.value.code == 2
    path, = run_files(tmp_path)
    assert load_evaluation(path)["results"][0]["status"] == "completed"


def test_unmatched_inputs_do_not_prepare_questions(tmp_path, monkeypatch, judge):
    transcripts, summaries = input_folders(tmp_path)
    (transcripts / "missing.txt").write_text("SourceAlpha")
    offline_judge(monkeypatch, judge)
    assert cli.main(arguments(tmp_path, transcripts, summaries)) == 1
    assert not judge.prompts


def test_empty_or_missing_input_folders_fail_before_scoring(tmp_path):
    transcripts, summaries = input_folders(tmp_path)
    for left in (transcripts, tmp_path / "absent"):
        with pytest.raises(ValueError):
            asyncio.run(cli.evaluate_batch(left, summaries, output_dir=tmp_path / "output"))


def test_cli_rejects_invalid_count_and_removed_command_shape(tmp_path):
    transcripts, summaries = input_folders(tmp_path)
    for argv in ([*arguments(tmp_path, transcripts, summaries), "--n", "0"],
                 ["--transcripts", str(transcripts), "--summaries", str(summaries)]):
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 2
