"""Evaluate paired folders with native DeepEval metrics and save local reports."""

import argparse
from importlib.metadata import version
import os
from pathlib import Path

from deepeval.test_case import LLMTestCase

from .inputs import discover_pairs, normalize_text
from .metrics import create_metric
from .reports import write_reports


def evaluate_batch(transcript_dir, summary_dir, *, model=None, business_context=None, n=5):
    """Measure sorted pairs sequentially, preserving an error row for failed meetings."""
    pairs = discover_pairs(transcript_dir, summary_dir)
    if not pairs:
        raise ValueError("No .txt or .md input files were found.")
    results = []
    for pair in pairs:
        result = {
            "meeting_id": pair.meeting_id,
            "status": "error",
            "score": None,
            "score_breakdown": None,
            "reason": None,
            "error": pair.error,
            "assessment_questions": [],
            "truths": [],
            "claims": [],
            "alignment_verdicts": [],
            "coverage_verdicts": [],
        }
        results.append(result)
        if pair.error:
            continue
        try:
            transcript = normalize_text(pair.transcript_path.read_text(encoding="utf-8-sig"))
            summary = normalize_text(pair.summary_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            continue
        if not transcript.strip() or not summary.strip():
            result["error"] = "Transcript and summary must both contain text."
            continue
        try:
            metric = create_metric(model=model, business_context=business_context, n=n, async_mode=False)
            metric.measure(LLMTestCase(input=transcript, actual_output=summary))
            result.update(
                status="completed",
                score=metric.score,
                score_breakdown=metric.score_breakdown,
                reason=metric.reason,
                assessment_questions=metric.assessment_questions,
                truths=metric.truths,
                claims=metric.claims,
                alignment_verdicts=[v.model_dump(mode="json") for v in metric.alignment_verdicts],
                coverage_verdicts=[v.model_dump(mode="json") for v in metric.coverage_verdicts],
            )
        except Exception as exc:
            # Provider exceptions may include submitted sources or authentication headers.
            result["error"] = (
                f"{type(exc).__name__}: evaluation failed; check the model, endpoint, credentials, and connection."
            )
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate transcript/summary pairs with DeepEval.")
    parser.add_argument("--transcripts", required=True, type=Path)
    parser.add_argument("--summaries", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--business-context", type=Path, help="Audience and purpose; not factual evidence")
    parser.add_argument("--n", type=int, default=5, help="Coverage questions per meeting (default: 5)")
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n must be a positive integer")
    try:
        business_context = args.business_context.read_text(encoding="utf-8-sig") if args.business_context else None
        if business_context is not None and not business_context.strip():
            raise ValueError("Business context must contain text.")
        model = args.model or os.getenv("OPENAI_MODEL", "gpt-6-luna")
        results = evaluate_batch(args.transcripts, args.summaries, model=model,
                                 business_context=business_context, n=args.n)
        output = write_reports(results, args.output, {
            "model": model, "deepeval_version": version("deepeval"),
            "n": args.n, "business_context": business_context,
        })
    except (ValueError, OSError, UnicodeError) as exc:
        parser.exit(2, f"Input/output error: {exc}\n")
    for result in results:
        if result["status"] == "completed":
            breakdown = result["score_breakdown"]
            print(f"{result['meeting_id']}: alignment={breakdown['Alignment']:.2f}, "
                  f"coverage={breakdown['Coverage']:.2f}, score={result['score']:.2f}")
        else:
            print(f"{result['meeting_id']}: {result['error']}")
    print(f"Reports: {output}")
    return 1 if any(result["status"] == "error" for result in results) else 0
