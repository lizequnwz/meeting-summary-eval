"""Lightweight command-line entry point."""

import argparse
from pathlib import Path

from .evaluator import MeetingSummaryEvaluator


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate transcript/summary pairs with DeepEval.")
    parser.add_argument("--transcripts", required=True, type=Path)
    parser.add_argument("--summaries", required=True, type=Path)
    context = parser.add_mutually_exclusive_group()
    context.add_argument("--context", type=Path, help="UTF-8 context supplied to every meeting")
    context.add_argument("--contexts", type=Path, help="Optional context files matched by meeting stem")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--metrics", nargs="+", choices=["faithfulness", "coverage", "structure", "clarity"], default=["faithfulness", "coverage"])
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        with MeetingSummaryEvaluator(model=args.model, max_concurrency=args.max_concurrency,
                                     metrics=args.metrics, env_file=args.env_file, timeout=args.timeout,
                                     max_retries=args.max_retries) as evaluator:
            supplied_context = args.context.read_text(encoding="utf-8-sig") if args.context else None
            batch = evaluator.evaluate_batch(args.transcripts, args.summaries, args.output,
                                             context_dir=args.contexts, provided_context=supplied_context)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Input/output error: {exc}\n")
    counts = {status: sum(r.execution_status == status for r in batch.results)
              for status in ("completed", "partial", "invalid", "error")}
    print("Meetings: " + ", ".join(f"{count} {name}" for name, count in counts.items()))
    print(f"Human review flagged: {sum(r.review_required for r in batch.results)}")
    for result in batch.results:
        scores = ", ".join(f"{name}={metric.score:.2f}" if metric.score is not None else f"{name}=unavailable"
                           for name, metric in result.metrics.items())
        print(f"{result.meeting_id}: {scores or result.execution_status}")
    print(f"Reports: {batch.output_dir}")
    # Judge quality findings do not make CLI execution fail; infrastructure/input failures do.
    return 1 if any(r.execution_status != "completed" for r in batch.results) else 0
