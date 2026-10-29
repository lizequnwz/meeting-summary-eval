"""Exercise the repeat-trial workflow with controlled, offline evaluator results."""

import importlib.util
import json
from pathlib import Path

import pytest

from meeting_eval.models import MeetingResult, MetricResult

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calibrate.py"
_SPEC = importlib.util.spec_from_file_location("calibrate", _SCRIPT)
calibrate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(calibrate)


def test_authored_manifest_has_pending_reviews_and_distinct_boundary_cases():
    manifest = calibrate.load_manifest(calibrate.DEFAULT_MANIFEST)
    assert len(manifest["cases"]) >= 15
    assert all(case["human_review"]["status"] == "pending" for case in manifest["cases"])
    assert {"negation", "units", "attribution", "outreach", "provided_context", "qa", "injection", "correction"} <= {
        case["category"] for case in manifest["cases"]
    }


@pytest.mark.parametrize("change", ["duplicate", "pending_label", "empty_context", "missing_rationale"])
def test_manifest_rejects_malformed_cases(tmp_path, change):
    manifest = calibrate.load_manifest(calibrate.DEFAULT_MANIFEST)
    case = manifest["cases"][0]
    if change == "duplicate":
        manifest["cases"].append(case)
    elif change == "pending_label":
        case["human_review"]["review_required"] = False
    elif change == "empty_context":
        case["provided_context"] = "  "
    else:
        case["design_expectation"]["rationale"] = ""
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        calibrate.load_manifest(path)


def test_default_command_never_invokes_judge(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline validation must not invoke a judge")
    monkeypatch.setattr(calibrate, "run_trials", forbidden)
    assert calibrate.main([]) == 0
    assert "Offline validation only" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        calibrate.main(["--run", "--model", "offline-model"])
    assert exc.value.code == 2


def test_trial_summary_excludes_failed_metric_scores():
    manifest = calibrate.load_manifest(calibrate.DEFAULT_MANIFEST)
    case = manifest["cases"][0]
    results = [
        MeetingResult(meeting_id="valid", metrics={"coverage": MetricResult(name="coverage", score=0.8)},
                      metadata={"calibration_case_id": case["id"]}),
        MeetingResult(meeting_id="failed", execution_status="partial", review_required=True,
                      metrics={"coverage": MetricResult(name="coverage", status="error", score=0.1)},
                      metadata={"calibration_case_id": case["id"]}),
    ]
    row = calibrate.summarize_trials(results, [case], requested_trials=2)[0]
    assert row["trials"] == 2
    assert row["completed_trials"] == 1
    assert row["review_flagged_trials"] == 1
    assert row["coverage_valid_trials"] == 1
    assert row["coverage_mean"] == 0.8
    assert row["human_review_status"] == "pending"


def test_run_trials_preserves_context_failures_and_blank_human_annotations(tmp_path, monkeypatch):
    import csv
    import meeting_eval

    calls = []
    closed = []
    class OfflineEvaluator:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            closed.append(True)
        def evaluate(self, transcript, summary, meeting_id, *, provided_context=None):
            calls.append((meeting_id, provided_context))
            result = MeetingResult(meeting_id=meeting_id,
                metadata={"diagnostics_status": "completed", "model": "offline-model",
                          "judge_config": {"endpoint": "https://example.invalid/v1"},
                          "versions": {"deepeval": "controlled"}, "evaluator_fingerprint": "controlled-policy",
                          "metrics": ["coverage"], "transcript_sha256": "per-meeting-source"},
                metrics={"coverage": MetricResult(name="coverage", score=0.8)})
            if meeting_id.endswith("_2"):
                result.execution_status = "partial"
                result.metrics["coverage"] = MetricResult(name="coverage", status="error")
                result.errors = ["controlled timeout"]
                result.review_required = True
            return result
    monkeypatch.setattr(meeting_eval, "MeetingSummaryEvaluator", OfflineEvaluator)
    manifest = calibrate.load_manifest(calibrate.DEFAULT_MANIFEST)
    manifest["cases"] = [next(case for case in manifest["cases"] if case["id"] == "provided_context_supported")]
    output = calibrate.run_trials(manifest, calibrate.DEFAULT_MANIFEST, model="offline-model", trials=2, output_dir=tmp_path)
    assert len(calls) == 2 and calls[0][1] == manifest["cases"][0]["provided_context"]
    assert closed == [True]
    payload = json.loads((output / "results.json").read_text())
    assert payload["results"][1]["execution_status"] == "partial"
    assert payload["metadata"]["manifest_sha256"]
    assert payload["metadata"]["judge_config"] == {"endpoint": "https://example.invalid/v1"}
    assert payload["metadata"]["evaluator_fingerprint"] == "controlled-policy"
    assert payload["metadata"]["metrics"] == ["coverage"]
    assert "transcript_sha256" not in payload["metadata"]
    row = json.loads((output / "trial_summary.json").read_text())[0]
    assert row["coverage_valid_trials"] == 1
    with (output / "human_annotations.csv").open(newline="") as file:
        annotations = list(csv.DictReader(file))
    assert annotations[0]["review_status"] == "pending"
    assert annotations[0]["reviewer"] == annotations[0]["review_required"] == ""
    blind = json.loads((output / "review_cases.json").read_text())[0]
    assert set(blind) == {"id", "transcript", "provided_context", "summary"}
    assert "design_expectation" not in blind


def test_global_judge_failure_stops_requests_and_records_unrun_trials(tmp_path, monkeypatch):
    import meeting_eval

    calls = []
    class MisconfiguredEvaluator:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def evaluate(self, transcript, summary, meeting_id, **kwargs):
            calls.append(meeting_id)
            return MeetingResult(meeting_id=meeting_id, execution_status="error", review_required=True,
                errors=["controlled model configuration failure"],
                metadata={"judge_configuration_error": True, "diagnostics_status": "not_run"})
    monkeypatch.setattr(meeting_eval, "MeetingSummaryEvaluator", MisconfiguredEvaluator)
    manifest = calibrate.load_manifest(calibrate.DEFAULT_MANIFEST)
    manifest["cases"] = manifest["cases"][:2]
    output = calibrate.run_trials(manifest, calibrate.DEFAULT_MANIFEST, model="offline-model", trials=2, output_dir=tmp_path)
    assert len(calls) == 1
    payload = json.loads((output / "results.json").read_text())
    assert len(payload["results"]) == 1
    assert payload["metadata"]["requested_evaluations"] == 4
    assert payload["metadata"]["attempted_evaluations"] == 1
    assert payload["metadata"]["stopped_after_configuration_failure"] is True
    summary = json.loads((output / "trial_summary.json").read_text())
    assert summary[0]["trials"] == 1 and summary[0]["unrun_trials"] == 1
    assert summary[1]["trials"] == 0 and summary[1]["unrun_trials"] == 2
    assert summary[1]["requested_trials"] == 2
