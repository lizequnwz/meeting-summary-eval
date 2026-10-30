"""Exercise the installed summarization metric, not a second scoring algorithm."""

import asyncio
from collections import Counter

import pytest
from deepeval.metrics import SummarizationMetric
from deepeval.metrics.summarization import SummarizationTemplate
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase

from meeting_eval import create_metric


@pytest.mark.parametrize("verdicts,alignment,overall", [
    (["yes", "no", "borderline"], 1 / 3, 1 / 3),
    (["yes", "yes", "yes"], 1, 0.5),
])
def test_native_metric_computes_both_scores_and_the_minimum(judge, verdicts, alignment, overall):
    judge.verdicts = verdicts
    metric = create_metric(model=judge, n=2)
    case = LLMTestCase(input="SourceAlpha: the original transcript.", actual_output="SummaryAlpha: the supplied summary.")

    score = asyncio.run(metric.a_measure(case, _show_indicator=False))

    assert isinstance(metric, SummarizationMetric)
    assert metric.model is judge
    assert metric.score_breakdown == {"Alignment": pytest.approx(alignment), "Coverage": 0.5}
    assert score == pytest.approx(overall)
    assert metric.reason == "Inspect the native claim and coverage verdicts."
    assert len(metric.claims) == len(metric.alignment_verdicts) == 3
    assert [v.summary_verdict for v in metric.coverage_verdicts] == ["yes", "no"]
    assert metric.threshold is None
    assert metric.eval_mode == "llm"
    assert Counter(stage for stage, _ in judge.prompts) == {
        "Truths": 1, "Claims": 1, "Questions": 1, "Answers": 2,
        "Verdicts": 1, "SummarizationScoreReason": 1,
    }


def test_business_context_only_changes_question_generation(judge):
    context = "AudienceMarker: institutional sales readers preparing advisor conversations."
    metric = create_metric(model=judge, business_context=context, n=2)
    case = LLMTestCase(input="SourceAlpha: the original transcript.", actual_output="SummaryAlpha: the supplied summary.")
    asyncio.run(metric.a_measure(case, _show_indicator=False))

    assert all((context in prompt) == (stage == "Questions") for stage, prompt in judge.prompts)
    assert metric.evaluation_template.generate_alignment_verdicts is SummarizationTemplate.generate_alignment_verdicts
    assert metric.evaluation_template.generate_answers is SummarizationTemplate.generate_answers
    assert "SourceAlpha" in next(prompt for stage, prompt in judge.prompts if stage == "Truths")
    assert "SummaryAlpha" in next(prompt for stage, prompt in judge.prompts if stage == "Claims")


def test_context_is_local_to_each_metric_and_omission_uses_stock_template(judge):
    first = create_metric(model=judge, business_context="Audience one")
    second = create_metric(model=judge, business_context="Audience two")
    first_prompt = first.evaluation_template.generate_questions(text="SourceAlpha", n=5)
    second_prompt = second.evaluation_template.generate_questions(text="SourceBeta", n=5)

    assert "Audience one" in first_prompt and "Audience two" not in first_prompt
    assert "Audience two" in second_prompt and "Audience one" not in second_prompt
    assert create_metric(model=judge).evaluation_template is SummarizationTemplate
    assert create_metric(model=judge).n == 5


def test_native_openai_model_uses_existing_environment_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "changed-offline-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-6-luna")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://offline.invalid/v1")
    metric = create_metric()
    explicit = create_metric(model="gpt-4o")

    assert isinstance(metric.model, OpenAIModel)
    assert metric.model.get_model_name() == "gpt-6-luna"
    assert metric.model.api_key.get_secret_value() == "changed-offline-test-key"
    assert metric.model.base_url == "https://offline.invalid/v1"
    assert explicit.model.get_model_name() == "gpt-4o"


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_invalid_question_count_is_rejected(judge, n):
    with pytest.raises(ValueError):
        create_metric(model=judge, n=n)


def test_blank_business_context_is_rejected(judge):
    with pytest.raises(ValueError):
        create_metric(model=judge, business_context=" \n ")
