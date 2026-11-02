"""Exercise installed native scoring with explicitly frozen questions."""

import asyncio
from collections import Counter

import pytest
from deepeval.metrics import SummarizationMetric
from deepeval.metrics.summarization import SummarizationTemplate
from deepeval.test_case import LLMTestCase
from langchain_openai import ChatOpenAI

from meeting_eval import create_metric
from meeting_eval.judge import LangChainJudge
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
    assert isinstance(metric.model, LangChainJudge)
    assert metric.model.chat_model is judge
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


def test_default_langchain_openai_model_uses_environment_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "changed-offline-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-6-luna")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://offline.invalid/v1")
    metric = create_metric(assessment_questions=QUESTIONS)
    explicit = create_metric(model="gpt-4o", assessment_questions=QUESTIONS)

    assert isinstance(metric.model.chat_model, ChatOpenAI)
    assert metric.model.get_model_name() == "gpt-6-luna"
    assert metric.model.chat_model.openai_api_key.get_secret_value() == "changed-offline-test-key"
    assert metric.model.base_url == "https://offline.invalid/v1"
    assert metric.model.chat_model.request_timeout == 120
    assert metric.model.chat_model.max_retries == 2
    assert explicit.model.get_model_name() == "gpt-4o"


def test_explicit_langchain_instance_is_used_without_reconfiguration():
    supplied = ChatOpenAI(model="my-deployment", api_key="provided-key",
                          base_url="https://explicit.invalid/v1", timeout=45, max_retries=0)
    metric = create_metric(model=supplied, assessment_questions=QUESTIONS)
    assert metric.model.chat_model is supplied
    assert metric.model.get_model_name() == "my-deployment"
    assert metric.model.chat_model.request_timeout == 45
    assert metric.model.chat_model.max_retries == 0


def test_api_base_environment_alias_and_default_endpoint(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_BASE", "https://alias.invalid/v1")
    assert resolve_model().base_url == "https://alias.invalid/v1"
    monkeypatch.delenv("OPENAI_API_BASE")
    assert resolve_model().base_url == "https://api.openai.com/v1"


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


def test_deepeval_adapter_instances_are_not_a_public_model_input(judge):
    with pytest.raises(TypeError, match="LangChain BaseChatModel"):
        resolve_model(LangChainJudge(judge))


def test_native_synchronous_scoring_uses_langchain_schema_responses(judge):
    metric = create_metric(model=judge, assessment_questions=QUESTIONS, async_mode=False)
    score = metric.measure(LLMTestCase(input="SourceAlpha", actual_output="SummaryAlpha"), _show_indicator=False)
    assert score == pytest.approx(1 / 3)
    assert metric.evaluation_cost is None
    assert not any(stage == "Questions" for stage, _ in judge.prompts)
