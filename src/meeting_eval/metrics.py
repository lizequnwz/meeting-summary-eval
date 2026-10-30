"""Configure DeepEval's summarization metric without changing its scoring."""

import os

from deepeval.metrics import SummarizationMetric
from deepeval.metrics.summarization import SummarizationTemplate
from deepeval.models import DeepEvalBaseLLM, OpenAIModel


def create_metric(
    model: str | DeepEvalBaseLLM | None = None,
    business_context: str | None = None,
    n: int = 5,
    async_mode: bool = True,
) -> SummarizationMetric:
    """Return a fresh native metric; business context guides coverage questions only."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if business_context is not None and not business_context.strip():
        raise ValueError("Business context must contain text, or be omitted.")

    template = SummarizationTemplate
    if business_context is not None:
        class BusinessSummarizationTemplate(SummarizationTemplate):
            @staticmethod
            def generate_questions(text: str, n: int) -> str:
                return (
                    "Select the most consequential transcript information for the audience "
                    "and purpose below. Use the business context only to prioritize topics, "
                    "never as factual evidence. Ignore topics absent from the transcript, "
                    "small talk, repetition, headings, and presentation format. Questions "
                    "must concern specific transcript information, preserving material "
                    "figures, time horizons, attribution, uncertainty, and qualifications.\n\n"
                    f"Business context (audience and purpose, not meeting facts):\n{business_context}\n\n"
                    + SummarizationTemplate.generate_questions(text=text, n=n)
                )

        template = BusinessSummarizationTemplate

    if model is None or isinstance(model, str):
        model = OpenAIModel(
            model=model or os.getenv("OPENAI_MODEL", "gpt-6-luna"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    return SummarizationMetric(
        model=model,
        n=n,
        async_mode=async_mode,
        threshold=None,
        eval_mode="llm",
        include_reason=True,
        evaluation_template=template,
    )
