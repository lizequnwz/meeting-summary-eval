"""Evaluate paired folders or rebuild reports from an existing saved evaluation."""

import argparse
import asyncio
import os
from pathlib import Path
import tempfile

from .evaluation import empty_result, evaluate_meeting, load_evaluation, new_run, save_evaluation
from .inputs import discover_pairs, normalize_text
from .questions import get_or_prepare_questions
from .reports import export_reports
from .storage import create_run_directory, text_hash


async def evaluate_batch(
    transcript_dir, summary_dir, *, model=None, business_context=None,
    n=None, questions_dir="results/questions", regenerate_questions=False, output_dir="results",
) -> Path:
    """Checkpoint each meeting; question preparation is independent of summary versions."""
    if n is not None and (type(n) is not int or n < 1):
        raise ValueError("n must be a positive integer")
    if business_context is not None:
        business_context = normalize_text(business_context)
        if not business_context.strip():
            raise ValueError("Business context must contain text.")
    pairs = discover_pairs(transcript_dir, summary_dir)
    if not pairs:
        raise ValueError("No .txt or .md input files were found.")
    run_dir = create_run_directory(output_dir)
    questions_dir = Path(questions_dir)
    questions_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=questions_dir):
        pass
    run = new_run({"business_context": business_context})
    run["results"] = [empty_result(pair.meeting_id, pair.transcript_path, pair.summary_path) for pair in pairs]
    results_json = save_evaluation(run, run_dir / "results.json")
    try:
        for index, pair in enumerate(pairs):
            row = run["results"][index]
            if pair.error:
                row.update(status="error", error=pair.error)
            else:
                try:
                    transcript = normalize_text(pair.transcript_path.read_text(encoding="utf-8-sig"))
                    summary = normalize_text(pair.summary_path.read_text(encoding="utf-8-sig"))
                    row["inputs"]["transcript"]["sha256"] = text_hash(transcript)
                    row["inputs"]["summary"]["sha256"] = text_hash(summary)
                    if not transcript.strip() or not summary.strip():
                        raise ValueError("Transcript and summary must both contain text.")
                    question_set, generated = await get_or_prepare_questions(
                        pair.meeting_id, transcript, business_context=business_context,
                        questions_dir=questions_dir, model=model, n=n,
                        regenerate=regenerate_questions,
                    )
                    row.update(question_set=question_set, question_generation_performed=generated)
                    save_evaluation(run, results_json)
                    run["results"][index] = await evaluate_meeting(
                        pair.meeting_id, transcript, summary, question_set=question_set, model=model,
                        transcript_path=pair.transcript_path, summary_path=pair.summary_path,
                        question_generation_performed=generated,
                    )
                except (ValueError, OSError, UnicodeError) as exc:
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            save_evaluation(run, results_json)
            _print_result(run["results"][index])
    except (KeyboardInterrupt, asyncio.CancelledError):
        run["status"] = "interrupted"
        save_evaluation(run, results_json)
        print(f"Partial results: {results_json}")
        try:
            export_reports(results_json)
        except (ValueError, OSError, UnicodeError) as exc:
            print(f"Report export failed; rebuild from the saved JSON: {exc}")
        raise
    run["status"] = "completed"
    save_evaluation(run, results_json)
    export_reports(results_json)
    return results_json


def _print_result(result):
    if result["status"] == "completed":
        scores = result["scores"]
        print(f"{result['meeting_id']}: alignment={scores['alignment']:.2f}, "
              f"coverage={scores['coverage']:.2f}, score={scores['overall']:.2f}")
    else:
        print(f"{result['meeting_id']}: {result['error'] or result['status']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate meeting summaries with frozen coverage questions.")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate folders, preparing missing questions once")
    evaluate.add_argument("--transcripts", required=True, type=Path)
    evaluate.add_argument("--summaries", required=True, type=Path)
    evaluate.add_argument("--output", type=Path, default=Path("results"))
    evaluate.add_argument("--questions", type=Path, default=Path("results/questions"))
    evaluate.add_argument("--model", default=None)
    evaluate.add_argument("--business-context", type=Path, help="Audience and purpose; not factual evidence")
    evaluate.add_argument("--n", type=int, default=None, help="Questions for a new set (default: 5); saved count otherwise")
    evaluate.add_argument("--regenerate-questions", action="store_true", help="Explicitly replace saved question sets")
    report = commands.add_parser("report", help="Rebuild Excel and HTML from saved JSON without model calls")
    report.add_argument("results_json", type=Path)
    report.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "evaluate" and args.n is not None and args.n < 1:
        evaluate.error("--n must be a positive integer")
    try:
        if args.command == "report":
            output = export_reports(args.results_json, args.output)
            print(f"Reports: {output}")
            return 0
        business_context = args.business_context.read_text(encoding="utf-8-sig") if args.business_context else None
        results_json = asyncio.run(evaluate_batch(
            args.transcripts, args.summaries, model=args.model or os.getenv("OPENAI_MODEL", "gpt-6-luna"),
            business_context=business_context, n=args.n, questions_dir=args.questions,
            regenerate_questions=args.regenerate_questions, output_dir=args.output,
        ))
        print(f"Saved evaluation: {results_json}\nReports: {results_json.parent}")
        return 1 if any(row["status"] == "error" for row in load_evaluation(results_json)["results"]) else 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, UnicodeError) as exc:
        parser.exit(2, f"Input/output error: {exc}\n")
