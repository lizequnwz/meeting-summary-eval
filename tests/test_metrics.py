"""Exercise installed native scoring with explicitly frozen questions."""

import asyncio
from collections import Counter

import pytest
from deepeval.metrics import SummarizationMetric
from deepeval.metrics.summarization import SummarizationTemplate
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase

from meeting_eval import create_metric
from meeting_eval.metrics import resolve_model


QUESTIONS = ["Does SourceAlpha explain the outlook?", "Does SourceAlpha identify a risk?"]


@pytest.mark.parametrize("verdicts,alignment,overall", [
    (["yes", "no", "borderline"], 1 / 3, 1 / 3),
    (["yes", "yes", "yes"], 1, 0.5),
])
def test_native_metric_scores_frozen_questions_without_generating(judge, verdicts, alignment, overall):
    judge.verdicts = verdicts
    metric = create_metric(model=judge, assessment_questions=QUESTIONS)
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
    assert metric.evaluation_template is SummarizationTemplate
    assert metric.assessment_questions == QUESTIONS
    assert Counter(stage for stage, _ in judge.prompts) == {
        "Truths": 1, "Claims": 1, "Answers": 2,
        "Verdicts": 1, "SummarizationScoreReason": 1,
    }


def test_questions_are_copied_for_each_fresh_metric(judge):
    supplied = list(QUESTIONS)
    first = create_metric(model=judge, assessment_questions=supplied)
    second = create_metric(model=judge, assessment_questions=supplied)
    supplied.append("An additional question?")
    first.assessment_questions.append("Only the first metric?")

    assert second.assessment_questions == QUESTIONS


def test_native_openai_model_uses_environment_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "changed-offline-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-6-luna")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://offline.invalid/v1")
    metric = create_metric(assessment_questions=QUESTIONS)
    explicit = create_metric(model="gpt-4o", assessment_questions=QUESTIONS)

    assert isinstance(metric.model, OpenAIModel)
    assert metric.model.get_model_name() == "gpt-6-luna"
    assert metric.model.api_key.get_secret_value() == "changed-offline-test-key"
    assert metric.model.base_url == "https://offline.invalid/v1"
    assert explicit.model.get_model_name() == "gpt-4o"


@pytest.mark.parametrize("questions", [[], [" \n "], ["Same?", " same? "], [42], None, "Question?"])
def test_invalid_frozen_questions_fail_before_model_creation(monkeypatch, questions):
    monkeypatch.setattr("meeting_eval.metrics.resolve_model", lambda *args: pytest.fail("Model must not be created"))
    with pytest.raises(ValueError, match="Assessment questions"):
        create_metric(assessment_questions=questions)


def test_questions_are_required():
    with pytest.raises(TypeError):
        create_metric()


def test_invalid_model_type_is_rejected():
    with pytest.raises(TypeError, match="model must"):
        resolve_model(object())
