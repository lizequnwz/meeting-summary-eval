"""Paired comparisons of saved runs with the same evaluation conditions."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .models import BatchResult
from .reports import _valid_score, write_csv


def compare_batches(baseline: BatchResult, candidate: BatchResult) -> dict:
    """Compare changed summaries while refusing changed judges or source inputs.

    Only pairs with complete evaluations enter aggregate deltas. Incomplete
    pairs remain visible with their available scores and execution states.
    """
    for batch in (baseline, candidate):
        if not batch.results or len({r.meeting_id for r in batch.results}) != len(batch.results):
            raise ValueError("Comparison requires nonempty runs with unique meeting IDs.")
    for key in ("evaluator_fingerprint", "model", "judge_config", "versions", "metrics"):
        if key not in baseline.metadata or key not in candidate.metadata:
            raise ValueError(f"Comparison requires recorded {key} in both runs.")
        if baseline.metadata[key] != candidate.metadata[key]:
            raise ValueError(f"Runs have different {key}; establish a new baseline.")
    config = baseline.metadata["judge_config"]
    if not isinstance(config, dict):
        raise ValueError("Comparison requires a recorded judge configuration.")
    if config.get("custom_parameters_present"):
        raise ValueError("Unrecorded custom judge settings prevent a trustworthy comparison.")
    left = {result.meeting_id: result for result in baseline.results}
    right = {result.meeting_id: result for result in candidate.results}
    if left.keys() != right.keys():
        raise ValueError("Comparison requires the same meeting IDs in both runs.")

    rows, findings = [], {}
    for meeting_id in sorted(left):
        before, after = left[meeting_id], right[meeting_id]
        resolved_before = before.metadata.get("usage", {}).get("response_models", [])
        resolved_after = after.metadata.get("usage", {}).get("response_models", [])
        if resolved_before and resolved_after and sorted(resolved_before) != sorted(resolved_after):
            raise ValueError(f"Resolved response models changed for meeting {meeting_id}.")
        for key in ("transcript_sha256", "context_sha256"):
            if key not in before.metadata or key not in after.metadata:
                raise ValueError(f"Source identity {key} is missing for meeting {meeting_id}.")
            if before.metadata[key] != after.metadata[key]:
                raise ValueError(f"Source inputs changed for meeting {meeting_id}.")
        for result in (before, after):
            for key in ("evaluator_fingerprint", "model", "judge_config", "versions", "metrics"):
                if result.metadata.get(key) != baseline.metadata[key]:
                    raise ValueError(f"Meeting {meeting_id} does not match the run's {key}.")
        row = {
            "meeting_id": meeting_id,
            "baseline_status": before.execution_status,
            "candidate_status": after.execution_status,
            "complete_pair": before.execution_status == after.execution_status == "completed",
            "baseline_review_required": before.review_required,
            "candidate_review_required": after.review_required,
            "response_model_identity_available": bool(resolved_before and resolved_after),
        }
        for label, result in (("baseline", before), ("candidate", after)):
            row[f"{label}_high_severity_findings"] = (
                sum(f.severity == "high" for f in result.findings)
                if result.metadata.get("diagnostics_status") == "completed" else None
            )
            for name in ("request_count", "total_tokens"):
                row[f"{label}_{name}"] = result.metadata.get("usage", {}).get(name)
            row[f"{label}_elapsed_seconds"] = result.metadata.get("elapsed_seconds")
        for name in baseline.metadata["metrics"]:
            first, second = _valid_score(before.metrics.get(name)), _valid_score(after.metrics.get(name))
            row[f"baseline_{name}"] = first
            row[f"candidate_{name}"] = second
            row[f"{name}_delta"] = second - first if first is not None and second is not None else None
        rows.append(row)
        findings[meeting_id] = {
            label: [finding.model_dump(mode="json") for finding in result.findings if finding.severity == "high"]
            for label, result in (("baseline", before), ("candidate", after))
        }
    deltas = {}
    for name in baseline.metadata["metrics"]:
        values = [row[f"{name}_delta"] for row in rows if row["complete_pair"] and row[f"{name}_delta"] is not None]
        deltas[name] = {"paired_count": len(values), "mean_delta": statistics.mean(values) if values else None,
                        "median_delta": statistics.median(values) if values else None}
    return {
        "baseline_run_id": baseline.run_id, "candidate_run_id": candidate.run_id,
        "evaluator_fingerprint": baseline.metadata["evaluator_fingerprint"],
        "meeting_count": len(rows), "complete_pair_count": sum(row["complete_pair"] for row in rows),
        "metric_deltas": deltas, "rows": rows, "high_severity_findings": findings,
        "interpretation": "Positive score deltas favor the candidate. Findings remain judge assessments. Repeat trials and human review are needed to distinguish improvements from judge variation.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare two runs with identical judges, policy, and meeting sources.")
    parser.add_argument("baseline", type=Path, help="Baseline results.json")
    parser.add_argument("candidate", type=Path, help="Candidate results.json")
    parser.add_argument("--output", required=True, type=Path, help="New comparison directory")
    args = parser.parse_args(argv)
    try:
        baseline = BatchResult.model_validate_json(args.baseline.read_text(encoding="utf-8"))
        candidate = BatchResult.model_validate_json(args.candidate.read_text(encoding="utf-8"))
        comparison = compare_batches(baseline, candidate)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "comparison.json").write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        write_csv(comparison["rows"], args.output / "comparison.csv")
    except (OSError, ValueError) as exc:
        # Pydantic validation errors can include input text; do not echo them.
        message = str(exc) if type(exc) is ValueError else type(exc).__name__
        parser.exit(2, f"Comparison could not be created: {message}\n")
    print(f"Compared {comparison['complete_pair_count']} complete pairs of {comparison['meeting_count']} meetings.")
    print(f"Comparison: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
