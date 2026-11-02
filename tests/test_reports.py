"""Reports present one saved evaluation without reading inputs or judging again."""

import asyncio
from copy import deepcopy
from contextlib import closing
from html.parser import HTMLParser
import json

from openpyxl import load_workbook
import pytest

from meeting_eval.questions import get_or_prepare_questions
from meeting_eval import reports
from meeting_eval.storage import text_hash


@pytest.fixture
def saved_run(tmp_path, judge):
    transcript, summary = "SourceAlpha: original discussion.", "SummaryAlpha: supplied summary."
    question_set, _ = asyncio.run(get_or_prepare_questions(
        "meeting", transcript, questions_dir=tmp_path / "questions", model=judge, n=2,
    ))
    question_set["generation"]["cost"] = 0.125
    row = {
        "meeting_id": "meeting", "status": "completed",
        "inputs": {
            "transcript": {"path": "absent/transcript.txt", "sha256": text_hash(transcript)},
            "summary": {"path": "absent/summary.md", "sha256": text_hash(summary)},
        },
        "question_set": question_set, "question_generation_performed": False,
        "scores": {"alignment": 1 / 3, "coverage": 0.5, "overall": 1 / 3},
        "counts": {
            "alignment_supported": 1, "alignment_total": 3, "coverage_covered": 1,
            "coverage_eligible": 2, "coverage_total": 2,
        },
        "alignment": [
            {"claim_id": "C001", "claim": "First claim", "verdict": "yes", "reason": None},
            {"claim_id": "C002", "claim": "Second claim", "verdict": "no", "reason": "A contradiction."},
            {"claim_id": "C003", "claim": "Third claim", "verdict": "borderline", "reason": "Unsupported detail."},
        ],
        "coverage": [
            {"question_id": "Q001", "question": question_set["questions"][0], "transcript_answer": "yes",
             "summary_answer": "yes", "in_denominator": True, "outcome": "covered"},
            {"question_id": "Q002", "question": question_set["questions"][1], "transcript_answer": "yes",
             "summary_answer": "no", "in_denominator": True, "outcome": "missing"},
        ],
        "truths": ["The transcript records three facts."],
        "reason": "An important risk is missing.", "error": None,
        "evaluation": {
            "model": "offline-scripted-judge", "endpoint": None, "started_at": "2026-11-02T14:00:00+00:00",
            "duration_seconds": 1.25, "cost": None,
        },
    }
    run = {
        "schema_version": 1, "created_at": "2026-11-02T14:00:00+00:00",
        "updated_at": "2026-11-02T14:00:02+00:00", "status": "completed",
        "settings": {"model": "offline-scripted-judge", "deepeval_version": "4.3.2", "app_version": "0.4.0"},
        "results": [row],
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(run), encoding="utf-8")
    return path, run


def save(path, run):
    path.write_text(json.dumps(run), encoding="utf-8")


def values(sheet, row=2):
    return dict(zip((cell.value for cell in sheet[1]), (cell.value for cell in sheet[row]), strict=True))


def test_exports_four_sheets_and_matching_native_scores(saved_run):
    path, run = saved_run
    original = path.read_bytes()

    directory = reports.export_reports(path)

    assert directory == path.parent
    assert (directory / "report.html").is_file()
    assert not (directory / "scores.csv").exists()
    assert path.read_bytes() == original
    with closing(load_workbook(directory / "report.xlsx")) as workbook:
        assert workbook.sheetnames == ["Scores", "Coverage", "Alignment", "Run details"]
        scores = values(workbook["Scores"])
        assert scores["Alignment"] == pytest.approx(1 / 3)
        assert scores["Coverage"] == 0.5
        assert scores["Overall"] == pytest.approx(1 / 3)
        assert scores["Overall calculation"] == "min(0.333333, 0.5) = 0.333333"
        assert scores["Supported claims"] == 1 and scores["All claims"] == 3
        assert scores["Covered eligible questions"] == 1 and scores["Eligible questions"] == 2
        assert scores["All questions"] == 2
        assert scores["Evaluation cost (USD)"] is None
        assert scores["Question generation cost this run (USD)"] is None
        assert scores["Questions generated this run"] is False
        assert values(workbook["Coverage"], 3)["Outcome"] == "missing"
        assert values(workbook["Alignment"], 3)["Native reason"] == "A contradiction."
        details = list(workbook["Run details"].values)
        assert ("meeting", "question_set.generation.cost", 0.125) in details
        assert ("meeting", "inputs.transcript.sha256", run["results"][0]["inputs"]["transcript"]["sha256"]) in details
        assert ("meeting", "Transcript statement 1", "The transcript records three facts.") in details
        for sheet in workbook:
            assert sheet.freeze_panes is not None
            assert sheet.auto_filter.ref is not None
    markup = (directory / "report.html").read_text()
    assert "Alignment: 1/3 = 0.33. Coverage: 1/2 = 0.50." in markup
    assert "Overall: min(0.33, 0.50) = 0.33." in markup


def test_missing_questions_and_unsupported_claims_appear_first(saved_run):
    path, run = saved_run
    directory = reports.export_reports(path)
    markup = (directory / "report.html").read_text()
    questions = run["results"][0]["question_set"]["questions"]
    assert markup.index(questions[1]) < markup.index(questions[0])
    assert markup.index("Second claim") < markup.index("First claim")
    assert markup.index("Third claim") < markup.index("First claim")
    assert "All coverage questions (2)" in markup
    assert "All summary claims (3)" in markup
    assert "Native details and provenance" in markup


def test_errors_and_unprocessed_meetings_have_blank_scores(saved_run):
    path, run = saved_run
    for meeting_id, status, error in (("broken", "error", "Judge request failed."), ("later", "unprocessed", None)):
        row = deepcopy(run["results"][0])
        row.update(meeting_id=meeting_id, status=status, question_set=None, scores=None, counts=None,
                   alignment=[], coverage=[], truths=[], reason=None, error=error, evaluation=None)
        run["results"].append(row)
    run["status"] = "interrupted"
    save(path, run)

    directory = reports.export_reports(path)

    with closing(load_workbook(directory / "report.xlsx")) as workbook:
        for index in (3, 4):
            row = values(workbook["Scores"], index)
            assert all(row[field] is None for field in ("Alignment", "Coverage", "Overall", "Reason", "All claims"))
        assert values(workbook["Scores"], 3)["Error"] == "Judge request failed."
        assert workbook["Coverage"].max_row == 3
    markup = (directory / "report.html").read_text()
    assert "Run status: interrupted" in markup
    assert "1 completed · 1 errors · 1 unprocessed" in markup
    assert "Judge request failed." in markup
    assert "This meeting has not been evaluated." in markup


def test_zero_denominators_and_excluded_questions_are_explicit(saved_run):
    path, run = saved_run
    row = run["results"][0]
    for item in row["coverage"]:
        item.update(transcript_answer="no", in_denominator=False, outcome="excluded")
    row["alignment"] = []
    row["counts"].update(alignment_supported=0, alignment_total=0, coverage_covered=0, coverage_eligible=0)
    row["scores"] = {"alignment": 0.0, "coverage": 0.0, "overall": 0.0}
    save(path, run)

    directory = reports.export_reports(path)

    markup = (directory / "report.html").read_text()
    assert "0 (no source-answerable questions)" in markup
    assert "0 (no extracted claims)" in markup
    assert "excluded" in markup
    with closing(load_workbook(directory / "report.xlsx")) as workbook:
        scores = values(workbook["Scores"])
        assert scores["Coverage"] == 0
        assert scores["Eligible questions"] == 0
        assert values(workbook["Coverage"])["In denominator"] == "no"
        assert values(workbook["Coverage"])["Outcome"] == "excluded"


def test_current_question_generation_cost_is_separate_from_history(saved_run):
    path, run = saved_run
    run["results"][0]["question_generation_performed"] = True
    run["results"][0]["evaluation"]["cost"] = 0.03
    save(path, run)

    directory = reports.export_reports(path)

    with closing(load_workbook(directory / "report.xlsx")) as workbook:
        row = values(workbook["Scores"])
        assert row["Evaluation cost (USD)"] == 0.03
        assert row["Question generation cost this run (USD)"] == 0.125


@pytest.mark.parametrize("text", ["=1+1", "+SUM(1)", "-2+3", "@SUM(1)", "https://example.com", "\tmessage", "\rmessage", "\nmessage"])
def test_excel_external_text_stays_literal_without_changing_saved_json(saved_run, text):
    path, run = saved_run
    row = run["results"][0]
    row["reason"] = row["alignment"][0]["claim"] = row["alignment"][1]["reason"] = text
    save(path, run)
    original = path.read_bytes()

    directory = reports.export_reports(path)

    with closing(load_workbook(directory / "report.xlsx", data_only=False)) as workbook:
        for sheet_name, coordinate in (("Scores", "K2"), ("Alignment", "C2"), ("Alignment", "E3")):
            cell = workbook[sheet_name][coordinate]
            assert cell.value == text.replace("\r\n", "\n").replace("\r", "\n")
            assert cell.data_type == "s"
            assert cell.hyperlink is None
    assert path.read_bytes() == original


def test_html_escapes_source_judge_and_metadata_text(saved_run):
    path, run = saved_run
    attack = '<script>alert("x")</script>'
    row = run["results"][0]
    row["reason"] = row["alignment"][0]["claim"] = row["alignment"][1]["reason"] = attack
    row["truths"] = [attack]
    row["inputs"]["transcript"]["path"] = attack
    run["settings"]["note"] = attack
    save(path, run)

    directory = reports.export_reports(path)

    markup = (directory / "report.html").read_text()
    assert "<script>" not in markup
    assert "&lt;script&gt;" in markup

    class Tags(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tags = []

        def handle_starttag(self, tag, attrs):
            self.tags.append(tag)

    parser = Tags()
    parser.feed(markup)
    assert "script" not in parser.tags
    assert "table" in parser.tags
    assert "details" in parser.tags


def test_report_regeneration_is_offline_and_does_not_read_sources(saved_run, monkeypatch):
    path, _ = saved_run
    from meeting_eval import metrics

    def forbidden(*args, **kwargs):
        raise AssertionError("Report generation must not resolve a judge")

    monkeypatch.setattr(metrics, "resolve_model", forbidden)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    directory = reports.export_reports(path, path.parent / "exports")
    assert directory == path.parent / "exports"
    assert (directory / "report.xlsx").is_file()
    assert (directory / "report.html").is_file()
    (directory / "report.html").write_text("old report")
    reports.export_reports(path, directory)
    assert "old report" not in (directory / "report.html").read_text()


@pytest.mark.parametrize("content", ["{", '{"schema_version": 2}', '{"results": []}'])
def test_invalid_saved_evaluation_fails_readably_before_outputs(tmp_path, content):
    source = tmp_path / "results.json"
    source.write_text(content)
    with pytest.raises(ValueError):
        reports.export_reports(source)
    assert not (tmp_path / "report.xlsx").exists()
    assert not (tmp_path / "report.html").exists()


def test_export_failure_preserves_both_previous_reports(saved_run, monkeypatch):
    path, _ = saved_run
    excel, markup = path.parent / "report.xlsx", path.parent / "report.html"
    excel.write_bytes(b"previous workbook")
    markup.write_text("previous report")

    def failed(*args, **kwargs):
        raise ValueError("HTML export failed")

    monkeypatch.setattr(reports, "_html", failed)
    with pytest.raises(ValueError, match="HTML export failed"):
        reports.export_reports(path)
    assert excel.read_bytes() == b"previous workbook"
    assert markup.read_text() == "previous report"
    assert not list(path.parent.glob(".report-*"))


@pytest.mark.parametrize("text", ["x" * 32768, "illegal\x00text"])
def test_excel_text_limits_do_not_silently_change_native_details(saved_run, text):
    path, run = saved_run
    run["results"][0]["alignment"][0]["claim"] = text
    save(path, run)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Excel text"):
        reports.export_reports(path)
    assert path.read_bytes() == original
    assert not (path.parent / "report.xlsx").exists()
    assert not (path.parent / "report.html").exists()
