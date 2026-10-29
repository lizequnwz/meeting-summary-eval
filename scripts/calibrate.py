"""Validate fictional fixtures offline or explicitly run repeated judge trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "calibration" / "fixtures.json"


def load_manifest(path: Path) -> dict:
    """Validate the authored case set without importing a model client."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or not isinstance(manifest.get("cases"), list):
        raise ValueError("Expected fixture schema_version 1 and a cases list.")
    if not manifest["cases"]:
        raise ValueError("The manifest must contain at least one case.")
    for field in ("suite_id", "methodology_version"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"Manifest {field} must contain text.")
    ids = set()
    for case in manifest["cases"]:
        if not isinstance(case, dict):
            raise ValueError("Every case must be an object.")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", case_id):
            raise ValueError("Case IDs must be nonempty lowercase letters, digits, underscores, or hyphens.")
        if case_id in ids:
            raise ValueError(f"Duplicate case ID: {case_id}")
        ids.add(case_id)
        for field in ("category", "transcript", "summary"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"{case_id}: {field} must contain text.")
        context = case.get("provided_context")
        if context is not None and (not isinstance(context, str) or not context.strip()):
            raise ValueError(f"{case_id}: provided_context must be null or nonempty text.")
        expectation = case.get("design_expectation", {})
        if not isinstance(expectation, dict) or not isinstance(expectation.get("rationale"), str) or not expectation["rationale"].strip():
            raise ValueError(f"{case_id}: a design rationale is required.")
        if expectation.get("expected_review") is not None and type(expectation["expected_review"]) is not bool:
            raise ValueError(f"{case_id}: expected_review must be boolean or null.")
        review = case.get("human_review", {})
        if not isinstance(review, dict) or review.get("status") not in {"pending", "reviewed", "adjudicated", "uncertain"}:
            raise ValueError(f"{case_id}: human review status is required.")
        if review["status"] in {"reviewed", "adjudicated"}:
            if not review.get("reviewer") or type(review.get("review_required")) is not bool or type(review.get("material_issue_present")) is not bool:
                raise ValueError(f"{case_id}: reviewed labels require a reviewer and boolean review/material-issue labels.")
        if review["status"] == "pending" and any(review.get(field) is not None for field in ("reviewer", "review_required", "material_issue_present")):
            raise ValueError(f"{case_id}: pending human review cannot contain completed labels.")
    return manifest


def summarize_trials(results: list, cases: list[dict], *, requested_trials: int) -> list[dict]:
    """Report observed variation without interpreting design expectations as truth."""
    rows = []
    metric_names = sorted({name for result in results for name in result.metrics})
    for case in cases:
        trials = [result for result in results if result.metadata.get("calibration_case_id") == case["id"]]
        row = {
            "case_id": case["id"],
            "category": case["category"],
            "requested_trials": requested_trials,
            "trials": len(trials),
            "unrun_trials": max(0, requested_trials - len(trials)),
            "completed_trials": sum(result.execution_status == "completed" for result in trials),
            "review_flagged_trials": sum(result.review_required for result in trials),
            "human_review_status": case["human_review"]["status"],
            "design_expected_review": case["design_expectation"].get("expected_review"),
            "design_rationale": case["design_expectation"]["rationale"],
        }
        for name in metric_names:
            scores = [metric.score for result in trials
                      if (metric := result.metrics.get(name)) is not None
                      and metric.status == "completed" and metric.score is not None and math.isfinite(metric.score)]
            row[f"{name}_valid_trials"] = len(scores)
            row[f"{name}_mean"] = statistics.mean(scores) if scores else None
            row[f"{name}_minimum"] = min(scores) if scores else None
            row[f"{name}_maximum"] = max(scores) if scores else None
        rows.append(row)
    return rows


def run_trials(manifest: dict, manifest_path: Path, *, model: str, trials: int, output_dir: Path) -> Path:
    """Make explicitly requested model calls and preserve unscored failures."""
    from meeting_eval import MeetingSummaryEvaluator
    from meeting_eval.models import BatchResult
    from meeting_eval.reports import write_csv, write_reports
    from meeting_eval.policy import METHODOLOGY_VERSION

    if manifest["methodology_version"] != METHODOLOGY_VERSION:
        raise ValueError("Fixture and installed evaluator policy versions differ; review and update the manifest before running.")
    if trials < 1:
        raise ValueError("trials must be positive.")
    results = []
    configuration_failed = False
    with MeetingSummaryEvaluator(model=model, max_concurrency=1) as evaluator:
        for case in manifest["cases"]:
            for trial in range(1, trials + 1):
                meeting_id = f"{case['id']}__trial_{trial}"
                result = evaluator.evaluate(case["transcript"], case["summary"], meeting_id=meeting_id,
                                            provided_context=case.get("provided_context"))
                configuration_failed = bool(result.metadata.get("judge_configuration_error"))
                result.metadata.update({"calibration_case_id": case["id"], "calibration_trial": trial})
                results.append(result)
                print(f"{meeting_id}: {result.execution_status}, review_required={result.review_required}", flush=True)
                if configuration_failed:
                    print("Stopped further trials after judge configuration failure; requested but unrun trials remain visible in the summary.", flush=True)
                    break
            if configuration_failed:
                break
    provenance_keys = (
        "model", "judge_config", "versions", "evaluator_fingerprint", "metrics",
        "methodology_version",
        "code_revision", "code_dirty", "sales_policy", "scoring", "max_concurrency",
    )
    provenance = {key: results[0].metadata[key] for key in provenance_keys if key in results[0].metadata}
    batch = BatchResult(
        run_id="calibration-" + uuid4().hex[:12],
        results=results,
        metadata={
            **provenance,
            "suite_id": manifest["suite_id"],
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "selected_case_ids": [case["id"] for case in manifest["cases"]],
            "trials_per_case": trials,
            "requested_evaluations": len(manifest["cases"]) * trials,
            "attempted_evaluations": len(results),
            "stopped_after_configuration_failure": configuration_failed,
            "methodology_version": manifest["methodology_version"],
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "calibration_status": "human_review_required",
            "note": "Design expectations are authored checks, not independent human labels; repetitions are separate input evaluations.",
        },
    )
    run_dir = write_reports(batch, output_dir)
    (run_dir / "fixtures.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    review_cases = [{field: case.get(field) for field in ("id", "transcript", "provided_context", "summary")}
                    for case in manifest["cases"]]
    (run_dir / "review_cases.json").write_text(json.dumps(review_cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = summarize_trials(results, manifest["cases"], requested_trials=trials)
    (run_dir / "trial_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(summary, run_dir / "trial_summary.csv")
    annotations = [{"case_id": case["id"], "reviewer": "", "review_status": "pending",
                    "review_required": "", "material_issue_present": "", "notes": ""}
                   for case in manifest["cases"]]
    write_csv(annotations, run_dir / "human_annotations.csv")
    return run_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case", action="append", dest="case_ids", help="Select a case ID; repeat to select several.")
    parser.add_argument("--run", action="store_true", help="Make real, billable judge calls; default is offline validation only.")
    parser.add_argument("--model", help="Explicit judge model for a real run.")
    parser.add_argument("--trials", type=int, help="Explicit number of trials per case for a real run.")
    parser.add_argument("--output", type=Path, default=Path("results/calibration"))
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.case_ids:
            unknown = set(args.case_ids) - {case["id"] for case in manifest["cases"]}
            if unknown:
                raise ValueError("Unknown case IDs: " + ", ".join(sorted(unknown)))
            manifest["cases"] = [case for case in manifest["cases"] if case["id"] in args.case_ids]
        print(f"Validated {len(manifest['cases'])} fictional cases; human review: "
              + ", ".join(f"{count} {status}" for status, count in sorted(Counter(case['human_review']['status'] for case in manifest['cases']).items())))
        if not args.run:
            if args.model is not None or args.trials is not None:
                parser.error("--model and --trials require --run; no model calls were made.")
            print("Offline validation only. Choose --run --model MODEL --trials N to make real judge requests.")
            return 0
        if not args.model or args.trials is None or args.trials < 1:
            parser.error("--run requires an explicit --model and positive --trials.")
        print(f"Running {len(manifest['cases'])} cases × {args.trials} trials with {args.model}; real judge calls are billable.", flush=True)
        run_dir = run_trials(manifest, args.manifest, model=args.model, trials=args.trials, output_dir=args.output)
        print(f"Saved: {run_dir}")
        payload = json.loads((run_dir / "results.json").read_text())
        return int(any(result["execution_status"] != "completed" for result in payload["results"]))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Calibration input/output error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
