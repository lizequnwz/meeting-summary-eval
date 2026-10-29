import csv
import json

import pytest

from meeting_eval.models import BatchResult, Evidence, Finding, MeetingResult, MetricResult
from meeting_eval.reports import aggregate_results, score_rows, write_reports, write_csv


def test_aggregate_excludes_missing_and_failed_scores():
    results = [
        MeetingResult(meeting_id="zero", metrics={"faithfulness": MetricResult(name="faithfulness", score=0)}),
        MeetingResult(meeting_id="good", metrics={"faithfulness": MetricResult(name="faithfulness", score=1)}),
        MeetingResult(meeting_id="missing", metrics={"faithfulness": MetricResult(name="faithfulness", status="error")}),
        MeetingResult(meeting_id="not_run"),
    ]
    stats = aggregate_results(results)["metrics"]["faithfulness"]
    assert stats["valid_count"] == 2
    assert stats["meeting_denominator"] == 4
    assert stats["mean"] == 0.5
    assert stats["median"] == 0.5
    assert stats["status_counts"] == {"completed": 2, "error": 1, "not_run": 1}
    rows = score_rows(results)
    assert rows[0]["faithfulness_score"] == 0
    assert rows[2]["faithfulness_score"] is None
    assert rows[3]["faithfulness_score"] is None


def test_issue_denominators_only_include_completed_diagnostics():
    finding = Finding(kind="unsupported_claim", severity="high", description="Example")
    structure = Finding(kind="structure", severity="low", description="Missing heading")
    results = [
        MeetingResult(meeting_id="flagged", metadata={"diagnostics_status": "completed"}, findings=[finding, finding]),
        MeetingResult(meeting_id="clear", metadata={"diagnostics_status": "completed"}),
        MeetingResult(meeting_id="failed", metadata={"diagnostics_status": "error"}),
        MeetingResult(meeting_id="unknown", findings=[structure], metrics={"structure": MetricResult(name="structure", score=0.5)}),
    ]
    stats = aggregate_results(results)
    assert stats["diagnostics_completed_count"] == 2
    assert stats["issue_categories"]["unsupported_claim"] == {
        "finding_count": 2,
        "meeting_count": 1,
        "assessed_meeting_count": 1,
        "meeting_denominator": 2,
        "meeting_fraction": 0.5,
        "denominator_type": "completed_diagnostics",
    }
    assert stats["issue_categories"]["structure"]["meeting_denominator"] == 1


def test_unknown_diagnostic_coverage_is_not_assumed_successful():
    result = MeetingResult(meeting_id="unknown", findings=[Finding(kind="uncertainty", severity="low", description="Unclear speaker")])
    stats = aggregate_results([result])
    assert stats["diagnostics_completed_count"] == 0
    assert stats["issue_categories"]["uncertainty"]["meeting_fraction"] is None


def test_reports_escape_untrusted_html_and_csv_formulas(tmp_path):
    payload = '<script>alert("x")</script>'
    result = MeetingResult(
        meeting_id='=HYPERLINK("https://example.invalid")',
        metrics={"faithfulness": MetricResult(name="faithfulness", score=0.6, reason=payload, details={"comment": payload})},
        findings=[Finding(kind="contradiction", severity="high", description=payload, section=payload,
                          evidence=[Evidence(source="transcript", quote=payload, line_start=2, line_end=3, verified=True)],
                          suggested_prompt_change=payload)],
        errors=["  +cmd"],
        review_reasons=["@SUM(1)"],
        review_required=True,
        metadata={"diagnostics_status": "completed"},
    )
    batch = BatchResult(run_id="example", results=[result])
    output = write_reports(batch, tmp_path)
    report = (output / "report.html").read_text()
    assert payload not in report
    assert "&lt;script&gt;" in report
    assert "line 2–3" in report
    assert "Quote verified" in report
    assert "0.60" in report
    assert "Prompt improvement" in report
    with (output / "scores.csv").open(newline="") as file:
        row = next(csv.DictReader(file))
    assert row["meeting_id"].startswith("'=")
    assert row["errors"].startswith("'  +")
    assert row["review_reasons"].startswith("'@")
    detailed = json.loads((output / "results.json").read_text())
    assert detailed["results"][0]["metrics"]["faithfulness"]["reason"] == payload
    assert detailed["aggregate"]["meeting_count"] == 1
    assert detailed["output_dir"] == str(output)


def test_run_directories_never_overwrite_an_existing_run(tmp_path):
    batch = BatchResult(run_id="same", results=[])
    first = write_reports(batch, tmp_path)
    second = write_reports(batch, tmp_path)
    assert first != second
    assert (first / "results.json").exists()
    assert (second / "results.json").exists()


@pytest.mark.parametrize("run_id", ["../../outside", "/absolute", ".", ""])
def test_run_directory_cannot_escape_output_root(tmp_path, run_id):
    output = write_reports(BatchResult(run_id=run_id, results=[]), tmp_path)
    assert output.parent == tmp_path.resolve()
    assert (output / "report.html").exists()


def test_empty_batch_has_valid_reports_without_zero_scores(tmp_path):
    batch = BatchResult(run_id="empty", results=[])
    output = write_reports(batch, tmp_path)
    aggregate = aggregate_results([])
    assert aggregate["metrics"] == {}
    assert aggregate["meeting_count"] == 0
    assert "No metrics were run" in (output / "report.html").read_text()
    with (output / "scores.csv").open(newline="") as file:
        assert list(csv.DictReader(file)) == []


def test_report_counts_partial_diagnostic_failures_as_evaluation_errors(tmp_path):
    result = MeetingResult(meeting_id="partial", execution_status="partial", review_required=True,
        metrics={"faithfulness": MetricResult(name="faithfulness", score=1)},
        errors=["diagnostics: timeout"], metadata={"diagnostics_status": "error"})
    run = write_reports(BatchResult(run_id="partial", results=[result]), tmp_path)
    report = (run / "report.html").read_text()
    assert '<strong>1</strong><span>Invalid inputs or evaluation errors</span>' in report


def test_analysis_csv_exports_use_formula_protection_and_all_columns(tmp_path):
    path = write_csv([{"meeting_id": "=formula", "description": "+unsafe"},
                      {"meeting_id": "safe", "extra": "@formula"}], tmp_path / "table.csv")
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["meeting_id"] == "'=formula"
    assert rows[0]["description"] == "'+unsafe"
    assert rows[1]["extra"] == "'@formula"


def test_unknown_token_usage_does_not_become_zero():
    results = [MeetingResult(meeting_id="one", metadata={"usage": {
        "request_count": 2, "successful_requests": 1, "failed_requests": 1,
        "requests_with_token_usage": 1, "input_tokens": None, "output_tokens": None, "total_tokens": None}})]
    usage = aggregate_results(results)["usage"]
    assert usage["request_count"] == 2
    assert usage["total_tokens"] is None
