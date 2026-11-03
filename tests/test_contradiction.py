"""Verify claim-level contradiction scoring through the real structured judge adapter."""

import asyncio
from copy import deepcopy

import pytest
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.metrics import BaseMetric
from deepeval.metrics.utils import copy_metrics
from deepeval.test_case import LLMTestCase
from pydantic import Field

from conftest import ScriptedJudge
from meeting_eval import SummaryContradictionMetric
from meeting_eval.judge import LangChainJudge


TRANSCRIPT = (
    "Transcript opening marker. Dana: We may launch in June. "
    "Dana: The deadline is Thursday. Transcript closing marker."
)
SUMMARY = (
    "Summary opening marker. Dana said the team may launch in June. "
    "The deadline is Friday. The budget is $9 million. Summary closing marker."
)
CLAIMS = [
    {
        "text": "Dana said the team may launch in June.",
        "summary_quote": "Dana said the team may launch in June.",
    },
    {"text": "The deadline is Friday.", "summary_quote": "The deadline is Friday."},
    {"text": "The budget is $9 million.", "summary_quote": "The budget is $9 million."},
]
VERDICTS = [
    {
        "claim_id": "C001",
        "verdict": "supported",
        "reason": "The source preserves the same speaker, tentative launch and month.",
        "transcript_quotes": ["Dana: We may launch in June."],
    },
    {
        "claim_id": "C002",
        "verdict": "contradicted",
        "reason": "The transcript specifies Thursday rather than Friday.",
        "transcript_quotes": ["Dana: The deadline is Thursday."],
    },
    {
        "claim_id": "C003",
        "verdict": "unsupported",
        "reason": "The transcript provides no budget amount.",
        "transcript_quotes": [],
    },
]


class ContradictionJudge(ScriptedJudge):
    extraction: list[dict] = Field(default_factory=lambda: deepcopy(CLAIMS))
    judgments: list[dict] = Field(default_factory=lambda: deepcopy(VERDICTS))
    extraction_valid: object = True
    omit_extraction_valid: bool = False
    failing_stage: str | None = None

    def respond(self, prompt, schema):
        self.prompts.append((schema.__name__, prompt))
        if schema.__name__ == self.failing_stage:
            raise RuntimeError("Private provider details and credentials")
        judgments = {"verdicts": self.judgments}
        if not self.omit_extraction_valid:
            judgments["extraction_valid"] = self.extraction_valid
        data = {
            "ClaimExtraction": {"claims": self.extraction},
            "ClaimJudgments": judgments,
        }
        return schema.model_validate(data[schema.__name__])


def case_for_metric():
    return LLMTestCase(input=TRANSCRIPT, actual_output=SUMMARY)


def measure(metric, invocation="async", case=None):
    case = case if case is not None else case_for_metric()
    if invocation == "async":
        return asyncio.run(metric.a_measure(case, _show_indicator=False))
    return metric.measure(case, _show_indicator=False)


@pytest.mark.parametrize("invocation", ["async", "sync", "sync_with_async_mode"])
def test_two_stage_metric_uses_complete_context_and_returns_inspectable_findings(invocation):
    judge = ContradictionJudge()
    metric = SummaryContradictionMetric(model=judge, async_mode=invocation != "sync")

    assert measure(metric, invocation) == 0

    assert isinstance(metric, BaseMetric)
    assert isinstance(metric.model, LangChainJudge)
    assert metric.model.chat_model is judge
    assert metric.threshold == 1
    assert metric.is_successful() is False
    assert metric.contradiction_rate == pytest.approx(1 / 3)
    assert metric.unsupported_rate == pytest.approx(1 / 3)
    assert metric.counts == {"total": 3, "supported": 1, "contradicted": 1, "unsupported": 1}
    assert [claim.model_dump() for claim in metric.claims] == CLAIMS
    assert [verdict.model_dump() for verdict in metric.verdicts] == VERDICTS
    assert "unsupported" in metric.reason.lower()
    assert metric.error is None
    assert metric.evaluation_cost is None

    assert [stage for stage, _ in judge.prompts] == ["ClaimExtraction", "ClaimJudgments"]
    extraction_prompt, judgments_prompt = [prompt for _, prompt in judge.prompts]
    assert SUMMARY in extraction_prompt
    assert "Transcript opening marker" not in extraction_prompt
    assert TRANSCRIPT in judgments_prompt
    assert SUMMARY in judgments_prompt
    assert all(verdict["claim_id"] in judgments_prompt for verdict in VERDICTS)


@pytest.mark.parametrize("verdict", ["supported", "unsupported"])
def test_no_contradictions_passes_without_hiding_unsupported_claims(verdict):
    judge = ContradictionJudge(
        extraction=[deepcopy(CLAIMS[0])],
        judgments=[{
            "claim_id": "C001",
            "verdict": verdict,
            "reason": "Inspect this claim's source support.",
            "transcript_quotes": [] if verdict == "unsupported" else ["We may launch in June."],
        }],
    )
    metric = SummaryContradictionMetric(model=judge)

    assert measure(metric) == 1
    assert metric.is_successful() is True
    assert metric.contradiction_rate == 0
    assert metric.unsupported_rate == (1 if verdict == "unsupported" else 0)
    assert metric.counts[verdict] == 1
    assert metric.verdicts[0].verdict == verdict


def test_batched_verdicts_are_matched_by_identifier_not_response_order():
    judge = ContradictionJudge(judgments=list(reversed(deepcopy(VERDICTS))))
    metric = SummaryContradictionMetric(model=judge)

    assert measure(metric) == 0
    assert [verdict.claim_id for verdict in metric.verdicts] == ["C001", "C002", "C003"]
    assert [verdict.verdict for verdict in metric.verdicts] == ["supported", "contradicted", "unsupported"]


def test_deepeval_can_clone_the_metric_with_its_configured_judge():
    judge = ContradictionJudge()
    original = SummaryContradictionMetric(model=judge, async_mode=False)
    assert measure(original, "sync") == 0

    clone, = copy_metrics([original])

    assert clone is not original
    assert clone.model.chat_model is judge
    assert clone.async_mode is False
    assert clone.score is None
    assert clone.claims == []
    assert clone.verdicts == []
    assert measure(clone, "sync") == 0
    assert clone.contradiction_rate == pytest.approx(1 / 3)


@pytest.mark.parametrize("run_async", [False, True])
def test_native_deepeval_evaluate_records_the_custom_metric(tmp_path, monkeypatch, run_async):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPEVAL_LOCAL_STORE", "json")
    # Exercise native execution and result storage while explicitly disabling uploads.
    monkeypatch.setattr("deepeval.test_run.test_run.is_confident", lambda: False)
    judge = ContradictionJudge(
        extraction=[deepcopy(CLAIMS[0])],
        judgments=[deepcopy(VERDICTS[0])],
    )
    metric = SummaryContradictionMetric(model=judge, async_mode=run_async)
    case = LLMTestCase(input=TRANSCRIPT, actual_output=CLAIMS[0]["summary_quote"])

    result = evaluate(
        test_cases=[case],
        metrics=[metric],
        async_config=AsyncConfig(run_async=run_async),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        display_config=DisplayConfig(
            show_indicator=False,
            print_results=False,
            inspect_after_run=False,
            results_folder=str(tmp_path / "runs"),
        ),
    )

    test_result, = result.test_results
    metric_data, = test_result.metrics_data
    assert test_result.success is True
    assert test_result.input == TRANSCRIPT
    assert test_result.actual_output == CLAIMS[0]["summary_quote"]
    assert metric_data.name == "Summary Contradiction"
    assert metric_data.score == 1
    assert metric_data.success is True
    assert metric_data.threshold == 1
    assert metric_data.strict_mode is True
    assert metric_data.evaluation_model == judge.model_name
    assert metric_data.error is None
    assert metric_data.reason
    assert result.confident_link is None
    assert [stage for stage, _ in judge.prompts] == ["ClaimExtraction", "ClaimJudgments"]


@pytest.mark.parametrize("bad_ids", [[], ["C001"], ["C001", "C001", "C003"], ["C001", "C002", "C999"]])
def test_missing_duplicate_or_unexpected_claim_verdicts_cannot_produce_scores(bad_ids):
    judgments = [
        {**deepcopy(VERDICTS[0]), "claim_id": claim_id}
        for claim_id in bad_ids
    ]
    metric = SummaryContradictionMetric(model=ContradictionJudge(judgments=judgments))

    with pytest.raises(ValueError):
        measure(metric)

    assert_no_completed_judgment(metric)


@pytest.mark.parametrize("claim", [
    {"text": "A rewritten assertion.", "summary_quote": "This quotation was never in the summary."},
    {"text": "", "summary_quote": "The deadline is Friday."},
    {"text": "The deadline is Friday.", "summary_quote": "   "},
])
def test_invalid_extracted_claims_fail_before_judgment(claim):
    judge = ContradictionJudge(extraction=[claim])
    metric = SummaryContradictionMetric(model=judge)

    with pytest.raises(ValueError):
        measure(metric)

    assert [stage for stage, _ in judge.prompts] == ["ClaimExtraction"]
    assert_no_completed_judgment(metric)


def test_empty_extraction_never_passes_a_nonempty_summary():
    judge = ContradictionJudge(extraction=[])
    metric = SummaryContradictionMetric(model=judge)

    with pytest.raises(ValueError):
        measure(metric)

    assert len(judge.prompts) == 1
    assert_no_completed_judgment(metric)


@pytest.mark.parametrize("invocation", ["async", "sync"])
def test_incomplete_or_changed_extraction_cannot_pass_as_unsupported(invocation):
    judge = ContradictionJudge(extraction_valid=False)
    metric = SummaryContradictionMetric(model=judge, async_mode=invocation != "sync")

    with pytest.raises(ValueError):
        measure(metric, invocation)

    assert len(judge.prompts) == 2
    assert_no_completed_judgment(metric)


@pytest.mark.parametrize("settings", [
    {"omit_extraction_valid": True},
    {"extraction_valid": None},
    {"extraction_valid": 1},
    {"extraction_valid": "true"},
])
def test_extraction_integrity_flag_must_be_an_explicit_boolean(settings):
    metric = SummaryContradictionMetric(model=ContradictionJudge(**settings))

    with pytest.raises(ValueError):
        measure(metric)

    assert_no_completed_judgment(metric)


@pytest.mark.parametrize("bad_evidence", [
    {"verdict": "supported", "transcript_quotes": []},
    {"verdict": "contradicted", "transcript_quotes": []},
    {"verdict": "supported", "transcript_quotes": ["A fabricated transcript quotation."]},
    {"verdict": "unsupported", "transcript_quotes": ["A fabricated transcript quotation."]},
    {"verdict": "contradicted", "transcript_quotes": ["   "]},
])
def test_incomplete_or_fabricated_evidence_never_produces_a_score(bad_evidence):
    judgments = deepcopy(VERDICTS)
    judgments[0].update(bad_evidence)
    metric = SummaryContradictionMetric(model=ContradictionJudge(judgments=judgments))

    with pytest.raises(ValueError):
        measure(metric)

    assert_no_completed_judgment(metric)


@pytest.mark.parametrize("mutation", [{"reason": " "}, {"verdict": "yes"}])
def test_malformed_judgments_do_not_become_content_failures(mutation):
    judgments = deepcopy(VERDICTS)
    judgments[0].update(mutation)
    metric = SummaryContradictionMetric(model=ContradictionJudge(judgments=judgments))

    with pytest.raises(ValueError):
        measure(metric)

    assert_no_completed_judgment(metric)


@pytest.mark.parametrize("field", ["input", "actual_output"])
@pytest.mark.parametrize("blank", ["", " \n "])
def test_blank_inputs_fail_without_any_judge_call(field, blank):
    judge = ContradictionJudge()
    metric = SummaryContradictionMetric(model=judge)
    values = {"input": TRANSCRIPT, "actual_output": SUMMARY, field: blank}

    with pytest.raises(ValueError):
        measure(metric, case=LLMTestCase(**values))

    assert judge.prompts == []
    assert_no_completed_judgment(metric)


@pytest.mark.parametrize("case", [
    {"input": TRANSCRIPT, "actual_output": SUMMARY},
    LLMTestCase(input=TRANSCRIPT, actual_output=SUMMARY, multimodal=True),
])
def test_unsupported_case_types_fail_without_any_judge_call(case):
    judge = ContradictionJudge()
    metric = SummaryContradictionMetric(model=judge)

    with pytest.raises((TypeError, ValueError)):
        measure(metric, case=case)

    assert judge.prompts == []
    assert_no_completed_judgment(metric)


@pytest.mark.parametrize("failing_stage", ["ClaimExtraction", "ClaimJudgments"])
@pytest.mark.parametrize("invocation", ["async", "sync"])
def test_provider_failures_do_not_expose_private_details_or_leave_old_scores(failing_stage, invocation):
    judge = ContradictionJudge(
        extraction=[deepcopy(CLAIMS[0])],
        judgments=[deepcopy(VERDICTS[0])],
    )
    metric = SummaryContradictionMetric(model=judge, async_mode=invocation != "sync")
    assert measure(metric, invocation) == 1
    assert metric.is_successful() is True
    judge.failing_stage = failing_stage

    with pytest.raises(Exception) as error:
        measure(metric, invocation)

    assert "Private" not in str(error.value)
    assert "Private" not in metric.error
    assert_no_completed_judgment(metric)

    judge.failing_stage = None
    assert measure(metric, invocation) == 1
    assert metric.error is None
    assert metric.is_successful() is True
    assert metric.counts == {"total": 1, "supported": 1, "contradicted": 0, "unsupported": 0}


def assert_no_completed_judgment(metric):
    assert metric.score is None
    assert metric.reason is None
    assert metric.contradiction_rate is None
    assert metric.unsupported_rate is None
    assert not metric.counts
    assert metric.claims == []
    assert metric.verdicts == []
    assert metric.error
    assert metric.is_successful() is False
