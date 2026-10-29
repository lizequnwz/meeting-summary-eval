import json

import pytest

from meeting_eval.comparison import compare_batches, main
from meeting_eval.models import BatchResult, MeetingResult, MetricResult


def batch(run_id, score):
    metadata = {"evaluator_fingerprint": "policy", "model": "judge", "judge_config": {
        "endpoint": "https://example.invalid/v1", "custom_parameters_present": False},
        "versions": {"deepeval": "4.3.2"}, "metrics": ["coverage"], "business_context_sha256": None}
    meeting = MeetingResult(meeting_id="one", metrics={"coverage": MetricResult(name="coverage", score=score)},
        metadata={**metadata, "transcript_sha256": "source", "context_sha256": None,
                  "diagnostics_status": "completed"})
    return BatchResult(run_id=run_id, results=[meeting], metadata=metadata)


def test_paired_comparison_preserves_missing_scores_and_excludes_incomplete_pairs():
    baseline, candidate = batch("baseline", 0.5), batch("candidate", 0.8)
    comparison = compare_batches(baseline, candidate)
    assert comparison["metric_deltas"]["coverage"]["mean_delta"] == pytest.approx(0.3)
    candidate.results[0].execution_status = "partial"
    candidate.results[0].metrics["coverage"] = MetricResult(name="coverage", status="error")
    candidate.results[0].metadata["diagnostics_status"] = "error"
    comparison = compare_batches(baseline, candidate)
    assert comparison["rows"][0]["coverage_delta"] is None
    assert comparison["rows"][0]["candidate_high_severity_findings"] is None
    assert comparison["metric_deltas"]["coverage"]["paired_count"] == 0
    assert comparison["metric_deltas"]["coverage"]["mean_delta"] is None


@pytest.mark.parametrize("key", ["evaluator_fingerprint", "model", "judge_config", "versions", "metrics", "business_context_sha256"])
def test_different_judge_or_policy_cannot_appear_comparable(key):
    baseline, candidate = batch("baseline", 0.5), batch("candidate", 0.8)
    candidate.metadata[key] = "changed"
    with pytest.raises(ValueError, match="different"):
        compare_batches(baseline, candidate)


@pytest.mark.parametrize("key", ["transcript_sha256", "context_sha256"])
def test_source_changes_cannot_appear_comparable(key):
    baseline, candidate = batch("baseline", 0.5), batch("candidate", 0.8)
    candidate.results[0].metadata[key] = "changed"
    with pytest.raises(ValueError, match="Source inputs changed"):
        compare_batches(baseline, candidate)


def test_unrecorded_parameters_and_missing_provenance_are_rejected():
    baseline, candidate = batch("baseline", 0.5), batch("candidate", 0.8)
    baseline.metadata["judge_config"]["custom_parameters_present"] = True
    candidate.metadata["judge_config"]["custom_parameters_present"] = True
    with pytest.raises(ValueError, match="Unrecorded"):
        compare_batches(baseline, candidate)
    baseline, candidate = batch("baseline", 0.5), batch("candidate", 0.8)
    del candidate.results[0].metadata["context_sha256"]
    with pytest.raises(ValueError, match="Source identity"):
        compare_batches(baseline, candidate)


def test_comparison_command_writes_new_files_and_refuses_overwrite(tmp_path):
    for name, score in (("baseline", 0.5), ("candidate", 0.8)):
        (tmp_path / f"{name}.json").write_text(batch(name, score).model_dump_json())
    argv = [str(tmp_path / "baseline.json"), str(tmp_path / "candidate.json"), "--output", str(tmp_path / "comparison")]
    assert main(argv) == 0
    assert json.loads((tmp_path / "comparison" / "comparison.json").read_text())["complete_pair_count"] == 1
    assert (tmp_path / "comparison" / "comparison.csv").exists()
    with pytest.raises(SystemExit):
        main(argv)


def test_changed_provider_model_snapshot_is_not_hidden_by_same_requested_alias():
    baseline, candidate = batch("baseline", 0.5), batch("candidate", 0.8)
    baseline.results[0].metadata["usage"] = {"response_models": ["snapshot-one"]}
    candidate.results[0].metadata["usage"] = {"response_models": ["snapshot-two"]}
    with pytest.raises(ValueError, match="response models changed"):
        compare_batches(baseline, candidate)


@pytest.mark.parametrize("scope", ["run", "meeting"])
def test_missing_business_context_identity_cannot_appear_comparable(scope):
    baseline, candidate = batch("baseline", 0.5), batch("candidate", 0.8)
    metadata = candidate.metadata if scope == "run" else candidate.results[0].metadata
    del metadata["business_context_sha256"]
    with pytest.raises(ValueError, match="business_context_sha256"):
        compare_batches(baseline, candidate)


def test_meeting_business_context_must_match_the_recorded_run():
    baseline, candidate = batch("baseline", 0.5), batch("candidate", 0.8)
    candidate.results[0].metadata["business_context_sha256"] = "changed-context"
    with pytest.raises(ValueError, match="business_context_sha256"):
        compare_batches(baseline, candidate)
