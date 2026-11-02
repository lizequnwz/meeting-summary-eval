"""Evaluate once, validate native judgments, and persist inspectable run records."""

from copy import deepcopy
from datetime import datetime
from importlib.metadata import version
import math
from pathlib import Path
from time import perf_counter

from deepeval.test_case import LLMTestCase

from .inputs import normalize_text
from .metrics import create_metric
from .questions import validate_question_set
from .storage import atomic_write_json, model_endpoint, read_json, safe_endpoint, text_hash, utc_now


class InvalidJudgment(ValueError):
    """A structurally incomplete or inconsistent native evaluation."""


def empty_result(meeting_id, transcript_path=None, summary_path=None) -> dict:
    def source(path):
        return {"path": str(Path(path).resolve()) if path is not None else None, "sha256": None}

    return {
        "meeting_id": meeting_id,
        "status": "unprocessed",
        "inputs": {"transcript": source(transcript_path), "summary": source(summary_path)},
        "question_set": None,
        "question_generation_performed": False,
        "scores": None,
        "counts": None,
        "alignment": [],
        "coverage": [],
        "truths": [],
        "reason": None,
        "error": None,
        "evaluation": None,
    }


def new_run(settings=None) -> dict:
    now = utc_now()
    return {
        "schema_version": 1,
        "created_at": now,
        "updated_at": now,
        "status": "running",
        "settings": {
            "deepeval_version": version("deepeval"),
            "app_version": version("meeting-summary-eval"),
            **(settings or {}),
        },
        "results": [],
    }


def _counts(alignment, coverage):
    return {
        "alignment_supported": sum(row["verdict"] == "yes" for row in alignment),
        "alignment_total": len(alignment),
        "coverage_covered": sum(row["outcome"] == "covered" for row in coverage),
        "coverage_eligible": sum(row["in_denominator"] for row in coverage),
        "coverage_total": len(coverage),
    }


def _check_scores(scores, counts):
    if not isinstance(scores, dict) or set(scores) != {"alignment", "coverage", "overall"}:
        raise InvalidJudgment("Scores must contain alignment, coverage, and overall.")
    for value in scores.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise InvalidJudgment("Scores must be finite numbers between zero and one.")
    alignment = counts["alignment_supported"] / counts["alignment_total"] if counts["alignment_total"] else 0
    coverage = counts["coverage_covered"] / counts["coverage_eligible"] if counts["coverage_eligible"] else 0
    expected = {"alignment": alignment, "coverage": coverage, "overall": min(alignment, coverage)}
    if any(not math.isclose(scores[key], value, rel_tol=1e-9, abs_tol=1e-9) for key, value in expected.items()):
        raise InvalidJudgment("Native scores do not match their complete verdicts.")


def _native_details(metric, questions):
    if metric.assessment_questions != questions:
        raise InvalidJudgment("The evaluated questions differ from the frozen set.")
    if len(metric.claims) != len(metric.alignment_verdicts):
        raise InvalidJudgment("Each summary claim must have exactly one alignment verdict.")
    if len(metric.coverage_verdicts) != len(questions):
        raise InvalidJudgment("Each frozen question must have exactly one coverage verdict.")
    alignment = []
    for index, (claim, verdict) in enumerate(zip(metric.claims, metric.alignment_verdicts), 1):
        value = getattr(verdict.verdict, "value", verdict.verdict)
        if value not in {"yes", "no", "borderline"}:
            raise InvalidJudgment("Unknown alignment verdict.")
        alignment.append({"claim_id": f"C{index:03}", "claim": claim,
                          "verdict": value, "reason": verdict.reason})
    coverage = []
    for index, (question, verdict) in enumerate(zip(questions, metric.coverage_verdicts), 1):
        if verdict.question != question:
            raise InvalidJudgment("Coverage verdicts must match frozen question order.")
        original, summary = verdict.original_verdict.strip().lower(), verdict.summary_verdict.strip().lower()
        if original not in {"yes", "no"} or summary not in {"yes", "no"}:
            raise InvalidJudgment("Coverage answerability must be yes or no.")
        outcome = "excluded" if original == "no" else "covered" if summary == "yes" else "missing"
        coverage.append({"question_id": f"Q{index:03}", "question": question,
                         "transcript_answer": original, "summary_answer": summary,
                         "in_denominator": original == "yes", "outcome": outcome})
    counts = _counts(alignment, coverage)
    scores = {"alignment": metric.score_breakdown["Alignment"],
              "coverage": metric.score_breakdown["Coverage"], "overall": metric.score}
    _check_scores(scores, counts)
    return alignment, coverage, counts, scores


async def evaluate_meeting(
    meeting_id, transcript, summary, *, question_set, model=None,
    transcript_path=None, summary_path=None, question_generation_performed=False,
) -> dict:
    """Return a completed or error record without generating coverage questions."""
    result = empty_result(meeting_id, transcript_path, summary_path)
    transcript, summary = normalize_text(transcript), normalize_text(summary)
    result["inputs"]["transcript"]["sha256"] = text_hash(transcript)
    result["inputs"]["summary"]["sha256"] = text_hash(summary)
    result["question_generation_performed"] = question_generation_performed
    try:
        validate_question_set(question_set)
        if question_set["meeting_id"] != meeting_id or question_set["transcript_sha256"] != text_hash(transcript):
            raise ValueError("Question set does not match this meeting's transcript.")
        result["question_set"] = deepcopy(question_set)
        if not transcript.strip() or not summary.strip():
            raise ValueError("Transcript and summary must both contain text.")
    except ValueError as exc:
        result.update(status="error", error=str(exc))
        return result

    started_at, started = utc_now(), perf_counter()
    metric = None
    try:
        metric = create_metric(model=model, assessment_questions=question_set["questions"])
        await metric.a_measure(LLMTestCase(input=transcript, actual_output=summary), _show_indicator=False)
        alignment, coverage, counts, scores = _native_details(metric, question_set["questions"])
        result.update(status="completed", alignment=alignment, coverage=coverage, counts=counts,
                      scores=scores, truths=list(metric.truths), reason=metric.reason)
    except InvalidJudgment as exc:
        result.update(status="error", error=f"Incomplete evaluation: {exc}")
    except Exception as exc:
        # Provider errors can contain credentials or submitted source text.
        result.update(status="error", error=(
            f"{type(exc).__name__}: evaluation failed; check the model, endpoint, credentials, and connection."
        ))
    finally:
        metadata = {
            "model": model if isinstance(model, str) else None,
            "endpoint": None,
            "started_at": started_at,
            "duration_seconds": perf_counter() - started,
            "cost": None,
        }
        try:
            if metric is not None:
                metadata["model"] = metric.evaluation_model
                metadata["endpoint"] = model_endpoint(metric.model)
        except Exception:
            # Optional provenance must not mask a judge failure or expose its message.
            pass
        result["evaluation"] = metadata
    return result


def validate_evaluation(run):
    """Reject incompatible or inconsistent saved records before export."""
    try:
        if not isinstance(run, dict) or type(run["schema_version"]) is not int or run["schema_version"] != 1:
            raise ValueError("Unsupported evaluation schema.")
        if run["status"] not in {"running", "completed", "interrupted"}:
            raise ValueError("Invalid run status.")
        if not isinstance(run["settings"], dict) or not isinstance(run["results"], list):
            raise ValueError("Invalid run settings or results.")
        if not all(isinstance(run[key], str) for key in ("created_at", "updated_at")):
            raise ValueError("Missing run timestamps.")
        if any(datetime.fromisoformat(run[key]).tzinfo is None for key in ("created_at", "updated_at")):
            raise ValueError("Run timestamps require a timezone.")
        if not all(isinstance(run["settings"][key], str) for key in ("deepeval_version", "app_version")):
            raise ValueError("Missing package versions.")
        ids = set()
        for row in run["results"]:
            if not isinstance(row, dict) or set(row) != set(empty_result("meeting")):
                raise ValueError("Invalid meeting record fields.")
            meeting_id = row["meeting_id"]
            if not isinstance(meeting_id, str) or not meeting_id or meeting_id in ids:
                raise ValueError("Meeting IDs must be nonempty and unique.")
            ids.add(meeting_id)
            if row["status"] not in {"unprocessed", "error", "completed"}:
                raise ValueError("Invalid meeting status.")
            if (not isinstance(row["alignment"], list) or not isinstance(row["coverage"], list)
                    or not isinstance(row["truths"], list) or not all(isinstance(item, str) for item in row["truths"])
                    or any(row[key] is not None and not isinstance(row[key], str) for key in ("reason", "error"))):
                raise ValueError("Invalid meeting details or explanation.")
            if row["evaluation"] is not None:
                metadata = row["evaluation"]
                if not isinstance(metadata, dict) or set(metadata) != {"model", "endpoint", "started_at", "duration_seconds", "cost"}:
                    raise ValueError("Invalid evaluation provenance.")
                if metadata["model"] is not None and not isinstance(metadata["model"], str):
                    raise ValueError("Invalid judge model.")
                if metadata["endpoint"] is not None and (
                    not isinstance(metadata["endpoint"], str) or safe_endpoint(metadata["endpoint"]) != metadata["endpoint"]
                ):
                    raise ValueError("Invalid judge endpoint.")
                if not isinstance(metadata["started_at"], str) or datetime.fromisoformat(metadata["started_at"]).tzinfo is None:
                    raise ValueError("Invalid evaluation timestamp.")
                for key in ("duration_seconds", "cost"):
                    value = metadata[key]
                    if key == "cost" and value is None:
                        continue
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError("Invalid evaluation duration or cost.")
            for source in ("transcript", "summary"):
                item = row["inputs"][source]
                if item["path"] is not None and not isinstance(item["path"], str):
                    raise ValueError("Invalid input path.")
                if item["sha256"] is not None and (
                    not isinstance(item["sha256"], str) or len(item["sha256"]) != 64
                    or any(character not in "0123456789abcdef" for character in item["sha256"])
                ):
                    raise ValueError("Invalid input hash.")
            if not isinstance(row["question_generation_performed"], bool):
                raise ValueError("Invalid question preparation status.")
            if row["question_set"] is not None:
                validate_question_set(row["question_set"])
                if row["question_set"]["meeting_id"] != meeting_id:
                    raise ValueError("Question-set meeting ID does not match.")
            if row["status"] != "completed":
                if row["scores"] is not None or row["counts"] is not None or row["alignment"] or row["coverage"]:
                    raise ValueError("Incomplete meetings cannot carry scores or verdicts.")
                if row["status"] == "error" and not row["error"]:
                    raise ValueError("Error meetings require an explanation.")
                if run["status"] == "completed" and row["status"] == "unprocessed":
                    raise ValueError("Completed runs cannot contain unprocessed meetings.")
                continue
            question_set = row["question_set"]
            if question_set is None or question_set["transcript_sha256"] != row["inputs"]["transcript"]["sha256"]:
                raise ValueError("Completed meeting lacks matching question provenance.")
            if row["inputs"]["summary"]["sha256"] is None or row["evaluation"] is None:
                raise ValueError("Completed meeting lacks input or judge provenance.")
            if len(row["coverage"]) != len(question_set["questions"]):
                raise ValueError("Coverage verdict count differs from frozen questions.")
            for index, detail in enumerate(row["alignment"], 1):
                if (detail["claim_id"] != f"C{index:03}" or not isinstance(detail["claim"], str)
                        or detail["verdict"] not in {"yes", "no", "borderline"}
                        or (detail["reason"] is not None and not isinstance(detail["reason"], str))):
                    raise ValueError("Invalid alignment detail.")
            for index, detail in enumerate(row["coverage"]):
                original, summary = detail["transcript_answer"], detail["summary_answer"]
                expected = "excluded" if original == "no" else "covered" if summary == "yes" else "missing"
                if (detail["question_id"] != f"Q{index + 1:03}"
                        or detail["question"] != question_set["questions"][index]
                        or original not in {"yes", "no"} or summary not in {"yes", "no"}
                        or type(detail["in_denominator"]) is not bool
                        or detail["in_denominator"] != (original == "yes")
                        or detail["outcome"] != expected):
                    raise ValueError("Invalid coverage detail.")
            counts = _counts(row["alignment"], row["coverage"])
            if not isinstance(row["counts"], dict) or row["counts"] != counts or any(type(value) is not int for value in row["counts"].values()):
                raise ValueError("Saved fractions do not match verdicts.")
            _check_scores(row["scores"], counts)
            if row["error"] is not None:
                raise ValueError("Completed meetings cannot have execution errors.")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid evaluation record structure.") from exc


def save_evaluation(run: dict, path: str | Path) -> Path:
    run["updated_at"] = utc_now()
    validate_evaluation(run)
    return atomic_write_json(path, run)


def load_evaluation(path: str | Path) -> dict:
    run = read_json(path)
    validate_evaluation(run)
    return run
