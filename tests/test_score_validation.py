import pytest
from deepeval.models import DeepEvalBaseLLM

from meeting_eval import MeetingSummaryEvaluator
from test_evaluator import FakeJudge


class ScriptedJudge(FakeJudge, DeepEvalBaseLLM):
    def __init__(self, verdicts, truths=None, claims=None):
        self.verdicts = verdicts
        self.truths = ["Rates rose."] if truths is None else truths
        self.claims = ["Rates fell.", "A fund is guaranteed."] if claims is None else claims

    def load_model(self):
        return self

    def get_model_name(self):
        return "offline-judge"

    def generate(self, prompt, schema=None):
        responses = {
            "Truths": {"truths": self.truths},
            "Claims": {"claims": self.claims},
            "Verdicts": {"verdicts": self.verdicts},
            "FaithfulnessScoreReason": {"reason": "Controlled response."},
        }
        return schema.model_validate(responses[schema.__name__])

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)


@pytest.mark.parametrize("count", [0, 1, 3])
async def test_real_deepeval_incomplete_or_extra_verdicts_cannot_produce_clean_one(count):
    async with MeetingSummaryEvaluator(metrics=("faithfulness",), env_file=None) as evaluator:
        evaluator._judge = ScriptedJudge([{"verdict": "yes", "reason": None}] * count)
        result = await evaluator.aevaluate("Speaker: Rates rose.", "Rates fell. A fund is guaranteed.")
    metric = result.metrics["faithfulness"]
    assert metric.status == "error" and metric.score is None
    assert "one valid verdict" in metric.reason
    assert len(metric.details["claims"]) == 2
    assert len(metric.details["verdicts"]) == count
    assert result.execution_status == "partial" and result.review_required


async def test_real_deepeval_empty_extracted_source_is_an_evaluation_error():
    async with MeetingSummaryEvaluator(metrics=("faithfulness",), env_file=None) as evaluator:
        evaluator._judge = ScriptedJudge([{"verdict": "borderline", "reason": "No source."}] * 2, truths=[])
        result = await evaluator.aevaluate("Speaker: Rates rose.", "Rates rose.")
    metric = result.metrics["faithfulness"]
    assert metric.score is None and metric.status == "error"
    assert metric.details["truths"] == []
    assert "No source statements" in metric.reason
    assert result.errors and result.review_required


async def test_complete_real_metric_preserves_extracted_source_and_computed_ratio():
    async with MeetingSummaryEvaluator(metrics=("faithfulness",), env_file=None) as evaluator:
        evaluator._judge = ScriptedJudge([
            {"verdict": "yes", "reason": None},
            {"verdict": "borderline", "reason": "Not supported."},
        ])
        result = await evaluator.aevaluate("Speaker: Rates rose.", "Two claims.")
    assert result.metrics["faithfulness"].score == 0.5
    assert result.metrics["faithfulness"].details["truths"] == ["Rates rose."]
    assert result.execution_status == "completed" and result.review_required


@pytest.mark.parametrize("field", ["truths", "claims"])
async def test_blank_extraction_entries_do_not_count_as_assessable_evidence(field):
    async with MeetingSummaryEvaluator(metrics=("faithfulness",), env_file=None) as evaluator:
        evaluator._judge = ScriptedJudge([{"verdict": "yes", "reason": None}], **{field: [" "]})
        result = await evaluator.aevaluate("Speaker: Rates rose.", "A summary.")
    assert result.metrics["faithfulness"].status == "error"
    assert result.metrics["faithfulness"].score is None
    assert "nonblank" in result.metrics["faithfulness"].reason
