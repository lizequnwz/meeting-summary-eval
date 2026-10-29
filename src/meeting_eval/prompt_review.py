"""Inspect and export the effective evaluation prompts without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .diagnostics import diagnostic_prompt
from .judge import JUDGE_SYSTEM_INSTRUCTION
from .inputs import normalize_text
from .metrics import (
    CLARITY_RUBRIC, CLARITY_STEPS, COVERAGE_RUBRIC, COVERAGE_STEPS,
    MeetingFaithfulnessTemplate,
)
from .policy import METHODOLOGY_VERSION, POLICY
from .provenance import evaluator_fingerprint


def _rubric_text(steps, rubric) -> str:
    instructions = "\n\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))
    anchors = "\n".join(f"- {item.score_range[0]}–{item.score_range[1]}: {item.expected_outcome}" for item in rubric)
    return instructions + "\n\nScoring rubric before 0–1 normalization:\n\n" + anchors


def review_prompts(business_context: str | None = None) -> dict[str, str]:
    """Return actual effective instructions using visible placeholders for input data."""
    if business_context is not None:
        if not isinstance(business_context, str) or not normalize_text(business_context).strip():
            raise ValueError("Business context must contain text, or be omitted.")
        business_context = normalize_text(business_context)
    prompts = {
        "policy": POLICY,
        "judge-system": JUDGE_SYSTEM_INSTRUCTION,
        "faithfulness-source": MeetingFaithfulnessTemplate.generate_truths(retrieval_context="[LABELED TRANSCRIPT AND OPTIONAL CONTEXT]"),
        "faithfulness-claims": MeetingFaithfulnessTemplate.generate_claims(actual_output="[SUMMARY]"),
        "faithfulness-verdicts": MeetingFaithfulnessTemplate.generate_verdicts(
            claims=["[EXTRACTED CLAIM]"], retrieval_context="[EXTRACTED SOURCE STATEMENTS]"),
        "faithfulness-reason": MeetingFaithfulnessTemplate.generate_reason(
            score="[COMPUTED SCORE]", contradictions=["[NEGATIVE OR BORDERLINE VERDICT REASON]"]),
        "coverage": _rubric_text(COVERAGE_STEPS, COVERAGE_RUBRIC),
        "clarity": _rubric_text(CLARITY_STEPS, CLARITY_RUBRIC),
        "diagnostics": diagnostic_prompt("[TRANSCRIPT]", "[SUMMARY]", "[OPTIONAL PROVIDED CONTEXT]", business_context),
    }
    if business_context is not None:
        prompts["business-context"] = business_context
    return prompts


def export_prompts(output_dir: str | Path, business_context: str | None = None) -> Path:
    """Write a new review bundle; exported copies do not change runtime prompts."""
    output = Path(output_dir).expanduser().resolve()
    prompts = review_prompts(business_context)
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for name, prompt in prompts.items():
        text = f"# {name}\n\n" + prompt.rstrip() + "\n"
        (output / f"{name}.md").write_text(text, encoding="utf-8")
        entries.append({"name": name, "file": f"{name}.md", "sha256": hashlib.sha256(text.encode()).hexdigest()})
    manifest = {
        "methodology_version": METHODOLOGY_VERSION,
        "evaluator_fingerprint": evaluator_fingerprint(),
        "business_context_sha256": hashlib.sha256(prompts["business-context"].encode()).hexdigest() if "business-context" in prompts else None,
        "prompts": entries,
        "notes": [
            "Review copies only. Runtime reads the Python evaluation prompt definitions.",
            "Input data are placeholders. No transcript, credentials, or model responses were loaded.",
            "coverage.md and clarity.md show configured steps/rubrics; DeepEval adds its scoring/schema wrapper.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Review or export current prompts without making model calls.")
    choices = (*review_prompts(), "business-context")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--show", choices=choices, help="Print one current prompt")
    action.add_argument("--output", type=Path, help="Export all prompts to a new directory")
    parser.add_argument("--business-context", type=Path, help="Optional audience and purpose context to include")
    args = parser.parse_args(argv)
    try:
        business_context = args.business_context.read_text(encoding="utf-8-sig") if args.business_context else None
        prompts = review_prompts(business_context)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Could not load business context: {type(exc).__name__}. Choose a readable, nonempty context file.\n")
    if args.show:
        if args.show not in prompts:
            parser.error("--show business-context requires --business-context FILE.")
        print(prompts[args.show])
    elif args.output:
        try:
            print(f"Prompt review bundle: {export_prompts(args.output, business_context)}")
        except OSError as exc:
            parser.exit(2, f"Could not export prompts: {type(exc).__name__}. Choose a new writable directory.\n")
    else:
        print(f"Evaluation methodology {METHODOLOGY_VERSION}")
        print("Available prompts:\n" + "\n".join(f"  {name}" for name in prompts))
        print("Use --show NAME to read a prompt, or --output NEW_DIRECTORY to export a review bundle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
