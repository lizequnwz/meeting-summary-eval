import csv
import json
from html.parser import HTMLParser

import pytest

from meeting_eval.reports import write_reports


def result(meeting_id="meeting", *, failed=False):
    return {
        "meeting_id": meeting_id,
        "status": "error" if failed else "completed",
        "score": None if failed else 0.5,
        "score_breakdown": None if failed else {"Alignment": 1.0, "Coverage": 0.5},
        "reason": None if failed else "An important risk is missing.",
        "error": "Judge request failed." if failed else None,
        "assessment_questions": [] if failed else ["Is duration risk discussed?"],
        "truths": [] if failed else ["Duration carries price risk."],
        "claims": [] if failed else ["Duration has price risk."],
        "alignment_verdicts": [] if failed else [{"verdict": "yes", "reason": None}],
        "coverage_verdicts": [] if failed else [
            {"question": "Is duration risk discussed?", "original_verdict": "yes", "summary_verdict": "no"}
        ],
    }


def test_exports_native_details_and_empty_failed_scores(tmp_path):
    results = [result(), result("broken", failed=True)]
    settings = {"model": "offline", "deepeval_version": "4.3.2", "n": 5, "business_context": "Advisor conversations"}
    directory = write_reports(results, tmp_path, settings)
    assert {path.name for path in directory.iterdir()} == {"results.json", "scores.csv", "report.html"}
    saved = json.loads((directory / "results.json").read_text())
    assert saved["results"] == results
    assert saved["settings"] == settings
    assert saved["created_at"].endswith("+00:00")
    with (directory / "scores.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0]) == ["meeting_id", "status", "alignment", "coverage", "score", "reason", "error"]
    assert rows[0]["alignment"] == "1.0"
    assert rows[0]["coverage"] == "0.5"
    assert rows[1]["status"] == "error"
    assert all(rows[1][field] == "" for field in ("alignment", "coverage", "score", "reason"))
    html = (directory / "report.html").read_text()
    for text in ("1 completed", "1 errors", "An important risk is missing.", "Judge request failed.",
                 "Is duration risk discussed?", "Extracted summary claims", "Alignment verdicts", "Coverage verdicts"):
        assert text in html


def test_html_escapes_every_untrusted_field(tmp_path):
    attack = '<script>alert("x")</script>'
    entry = result(attack)
    entry.update(reason=attack, error=attack, assessment_questions=[attack], claims=[attack], truths=[attack])
    entry["alignment_verdicts"][0]["reason"] = attack
    entry["coverage_verdicts"][0]["question"] = attack
    directory = write_reports([entry], tmp_path, {"business_context": attack})
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
    assert json.loads((directory / "results.json").read_text())["results"][0]["reason"] == attack


@pytest.mark.parametrize("formula", ["=1+1", "+SUM(1)", "-2+3", "@SUM(1)", "  =1", "\tmessage", "\rmessage", "\nmessage"])
def test_csv_formula_protection_does_not_modify_json(tmp_path, formula):
    entry = result(formula)
    entry.update(reason=formula, error=formula)
    directory = write_reports([entry], tmp_path, {})
    with (directory / "scores.csv").open(newline="") as file:
        row = next(csv.DictReader(file))
    assert all(row[field] == "'" + formula for field in ("meeting_id", "reason", "error"))
    assert json.loads((directory / "results.json").read_text())["results"][0] == entry


def test_new_runs_never_overwrite_previous_outputs(tmp_path):
    first = write_reports([result("first")], tmp_path, {})
    second = write_reports([result("second")], tmp_path, {})
    assert first != second
    assert json.loads((first / "results.json").read_text())["results"][0]["meeting_id"] == "first"
    assert json.loads((second / "results.json").read_text())["results"][0]["meeting_id"] == "second"
