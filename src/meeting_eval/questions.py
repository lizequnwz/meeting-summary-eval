"""Generate transcript-specific coverage questions once, then reuse them explicitly."""

from datetime import datetime
import json
import math
import numbers
from pathlib import Path
import re
import tempfile
from time import perf_counter

from deepeval.metrics.summarization import SummarizationTemplate
from deepeval.metrics.summarization.schema import Questions

from .inputs import normalize_text
from .metrics import resolve_model, validate_questions
from .storage import (
    atomic_write_json, cost_value, model_endpoint, read_json, safe_endpoint, text_hash, utc_now,
)


_KEYS = {
    "schema_version", "meeting_id", "question_set_id", "transcript_sha256",
    "business_context", "business_context_sha256", "n", "prompt_sha256", "questions",
    "generated_at", "generation",
}
_GENERATION_KEYS = {"model", "endpoint", "duration_seconds", "cost"}
_CONTEXT_INSTRUCTIONS = (
    "Select the most consequential transcript information for the audience "
    "and purpose below. Use the business context only to prioritize topics, "
    "never as factual evidence. Ignore topics absent from the transcript, "
    "small talk, repetition, headings, and presentation format. Prioritize material "
    "economic and market developments, figures and time horizons, investment and "
    "portfolio perspectives, risks, caveats, conditions and disagreements, narrowly "
    "supported attribution, explicitly discussed actions, product discussions, and "
    "substantive questions with their answers or unanswered status. Select only "
    "categories present in this transcript; this is not a mandatory checklist. "
    "Distinguish actions discussed in the meeting from generated outreach ideas, "
    "which are not transcript facts. Questions must be standalone questions about "
    "specific transcript facts, not instructions to compare a summary with a source. Questions "
    "must concern specific transcript information, preserving material "
    "figures, time horizons, attribution, uncertainty, and qualifications.\n\n"
)


def _prompt(transcript: str, n: int, business_context: str | None) -> str:
    native_prompt = SummarizationTemplate.generate_questions(text=transcript, n=n)
    if business_context is None:
        return native_prompt
    return (
        _CONTEXT_INSTRUCTIONS
        + f"Business context (audience and purpose, not meeting facts):\n{business_context}\n\n"
        + native_prompt
    )


def _prompt_sha256() -> str:
    # Render both paths so changes to native or contextual instructions are detected.
    return text_hash(json.dumps({
        "native": _prompt("[TRANSCRIPT]", 5, None),
        "contextual": _prompt("[TRANSCRIPT]", 5, "[BUSINESS CONTEXT]"),
    }, sort_keys=True, ensure_ascii=False))


def _question_set_id(question_set: dict) -> str:
    content = {key: question_set[key] for key in (
        "transcript_sha256", "business_context_sha256", "prompt_sha256", "n", "questions",
    )}
    return text_hash(json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def _positive_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_number(value) -> bool:
    return (
        isinstance(value, numbers.Real) and not isinstance(value, bool)
        and math.isfinite(value) and value >= 0
    )


def _hash(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _valid_meeting_id(meeting_id) -> bool:
    return (
        isinstance(meeting_id, str) and bool(meeting_id.strip())
        and meeting_id not in {".", ".."}
        and not any(character in meeting_id for character in ("/", "\\", "\x00"))
    )


def validate_question_set(question_set: dict) -> None:
    """Validate a saved artifact or historical evaluation snapshot without a judge."""
    try:
        if not isinstance(question_set, dict) or set(question_set) != _KEYS:
            raise ValueError
        if type(question_set["schema_version"]) is not int or question_set["schema_version"] != 1:
            raise ValueError
        if not _valid_meeting_id(question_set["meeting_id"]):
            raise ValueError
        if not all(_hash(question_set[key]) for key in ("transcript_sha256", "prompt_sha256", "question_set_id")):
            raise ValueError
        context = question_set["business_context"]
        if context is None:
            if question_set["business_context_sha256"] is not None:
                raise ValueError
        elif (
            not isinstance(context, str) or not context.strip()
            or normalize_text(context) != context
            or question_set["business_context_sha256"] != text_hash(context)
        ):
            raise ValueError
        questions = validate_questions(question_set["questions"])
        if not _positive_count(question_set["n"]) or question_set["n"] != len(questions):
            raise ValueError
        if _question_set_id(question_set) != question_set["question_set_id"]:
            raise ValueError
        if not isinstance(question_set["generated_at"], str):
            raise ValueError
        generated_at = datetime.fromisoformat(question_set["generated_at"])
        if generated_at.tzinfo is None:
            raise ValueError
        generation = question_set["generation"]
        if not isinstance(generation, dict) or set(generation) != _GENERATION_KEYS:
            raise ValueError
        if not isinstance(generation["model"], str) or not generation["model"].strip():
            raise ValueError
        endpoint = generation["endpoint"]
        if endpoint is not None and (not isinstance(endpoint, str) or safe_endpoint(endpoint) != endpoint):
            raise ValueError
        if not _nonnegative_number(generation["duration_seconds"]):
            raise ValueError
        if generation["cost"] is not None and not _nonnegative_number(generation["cost"]):
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError("Saved question set is invalid; regenerate it explicitly.") from None


async def get_or_prepare_questions(
    meeting_id,
    transcript,
    *,
    business_context=None,
    questions_dir=Path("results/questions"),
    model=None,
    n=None,
    regenerate=False,
) -> tuple[dict, bool]:
    """Reuse matching questions, requiring explicit replacement of stale artifacts."""
    if not _valid_meeting_id(meeting_id):
        raise ValueError("meeting_id must be a nonempty filename component.")
    if not isinstance(transcript, str):
        raise ValueError("Transcript must contain text.")
    transcript = normalize_text(transcript)
    if not transcript.strip():
        raise ValueError("Transcript must contain text.")
    if business_context is not None:
        if not isinstance(business_context, str):
            raise ValueError("Business context must contain text, or be omitted.")
        business_context = normalize_text(business_context)
        if not business_context.strip():
            raise ValueError("Business context must contain text, or be omitted.")
    if n is not None and not _positive_count(n):
        raise ValueError("n must be a positive integer, or be omitted.")

    path = Path(questions_dir) / f"{meeting_id}.json"
    if path.exists() and not path.is_file():
        raise ValueError("Question destination must be a file, not a directory or special file.")
    existing = None
    if path.exists():
        try:
            existing = read_json(path)
            validate_question_set(existing)
            if existing["meeting_id"] != meeting_id:
                raise ValueError
        except (OSError, ValueError, TypeError):
            existing = None
            if not regenerate:
                raise ValueError("Saved question set is invalid; regenerate it explicitly.") from None

    transcript_sha256 = text_hash(transcript)
    context_sha256 = text_hash(business_context) if business_context is not None else None
    prompt_sha256 = _prompt_sha256()
    if existing is not None and not regenerate:
        changed = (
            existing["transcript_sha256"] != transcript_sha256
            or existing["business_context_sha256"] != context_sha256
            or existing["prompt_sha256"] != prompt_sha256
            or (n is not None and existing["n"] != n)
        )
        if changed:
            raise ValueError(
                "Saved questions do not match the transcript, business context, prompt, or count; "
                "regenerate them explicitly."
            )
        return existing, False

    count = n if n is not None else existing["n"] if existing is not None else 5
    # Check the actual destination before a billable call, without replacing any
    # existing artifact. Atomic persistence uses a temporary file in this directory.
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}-") as probe:
        probe.write("Question destination write check.\n")
        probe.flush()
    started = perf_counter()
    try:
        judge = resolve_model(model)
        response = await judge.a_generate(_prompt(transcript, count, business_context), schema=Questions)
        cost = None
        if isinstance(response, tuple) and len(response) == 2:
            response, raw_cost = response
            cost = cost_value(raw_cost)
        if isinstance(response, str):
            response = Questions.model_validate_json(response)
        elif not isinstance(response, Questions):
            response = Questions.model_validate(response)
        generation = {
            "model": judge.get_model_name(),
            "endpoint": model_endpoint(judge),
            "duration_seconds": perf_counter() - started,
            "cost": cost,
        }
    except Exception:
        raise ValueError(
            "Question generation failed; check the model, endpoint, credentials, and connection."
        ) from None

    questions = validate_questions(response.questions)
    if len(questions) != count:
        raise ValueError(f"Question generation must return exactly {count} distinct questions.")
    question_set = {
        "schema_version": 1,
        "meeting_id": meeting_id,
        "question_set_id": "",
        "transcript_sha256": transcript_sha256,
        "business_context": business_context,
        "business_context_sha256": context_sha256,
        "n": count,
        "prompt_sha256": prompt_sha256,
        "questions": questions,
        "generated_at": utc_now(),
        "generation": generation,
    }
    question_set["question_set_id"] = _question_set_id(question_set)
    validate_question_set(question_set)
    atomic_write_json(path, question_set)
    return question_set, True
