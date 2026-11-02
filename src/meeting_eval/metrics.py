"""Configure native DeepEval scoring with an explicit frozen question set."""

import os

from deepeval.metrics import SummarizationMetric
from deepeval.models import DeepEvalBaseLLM, OpenAIModel


def resolve_model(model: str | DeepEvalBaseLLM | None = None) -> DeepEvalBaseLLM:
    """Use the same judge configuration for preparation and evaluation."""
    if model is None or isinstance(model, str):
        return OpenAIModel(
            model=model or os.getenv("OPENAI_MODEL", "gpt-6-luna"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    if not isinstance(model, DeepEvalBaseLLM):
        raise TypeError("model must be a model name or DeepEvalBaseLLM instance")
    return model


def validate_questions(questions: list[str]) -> list[str]:
    """Return a copy; an empty list would re-enable native question generation."""
    if not isinstance(questions, list) or not questions:
        raise ValueError("Assessment questions must be a nonempty list.")
    if any(not isinstance(question, str) or not question.strip() for question in questions):
        raise ValueError("Assessment questions must contain nonblank strings.")
    if len({question.strip().casefold() for question in questions}) != len(questions):
        raise ValueError("Assessment questions must be distinct.")
    return list(questions)


def create_metric(
    model: str | DeepEvalBaseLLM | None = None,
    *,
    assessment_questions: list[str],
    async_mode: bool = True,
) -> SummarizationMetric:
    """Return a fresh score-only metric that never generates its own questions."""
    questions = validate_questions(assessment_questions)
    return SummarizationMetric(
        model=resolve_model(model),
        assessment_questions=questions,
        n=len(questions),
        async_mode=async_mode,
        threshold=None,
        eval_mode="llm",
        include_reason=True,
    )
