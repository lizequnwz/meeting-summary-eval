import asyncio
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest

import meeting_eval.evaluator as engine
from meeting_eval import MeetingSummaryEvaluator
from meeting_eval.models import Diagnostics, Evidence, Finding


class FakeJudge:
    model = "offline-judge"

    @contextmanager
    def usage_scope(self):
        yield {"request_count": 0, "successful_requests": 0, "failed_requests": 0,
               "requests_with_token_usage": 0, "input_tokens": None,
               "output_tokens": None, "total_tokens": None, "elapsed_seconds": 0.0,
               "response_models": [], "stages": {}}

    def stage(self, name):
        return nullcontext()

    async def aclose(self):
        pass

    async def structured(self, prompt, schema):
        return Diagnostics(findings=[])


class FakeMetric:
    def __init__(self, name, fail=False):
        self.name, self.fail = name, fail
        self.score, self.reason = 1.0, "Supported test assessment"
        self.truths = ["A source-supported claim"]
        self.claims = ["A source-supported claim"]
        self.verdicts = [{"verdict": "yes", "reason": None}]

    async def a_measure(self, case):
        if self.fail:
            raise TimeoutError("Sensitive source text should never appear in the error")
        await asyncio.sleep(0)


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.setattr(engine, "LangChainJudge", lambda *args, **kwargs: FakeJudge())
    monkeypatch.setattr(engine, "create_metrics", lambda judge, names: {n: FakeMetric(n) for n in names})
    return MeetingSummaryEvaluator(env_file=None)


def test_sync_reuse_and_input_normalization(evaluator):
    first = evaluator.evaluate("\ufeffSpeaker: a claim.\r\n", "A claim.", "one")
    second = evaluator.evaluate("Speaker: another claim.", "Another claim.", "two")
    assert first.execution_status == second.execution_status == "completed"
    assert first.metrics["faithfulness"].score == 1.0
    assert not first.review_required
    assert first.metadata["diagnostics_status"] == "completed"
    assert len(first.metadata["transcript_sha256"]) == 64


@pytest.mark.parametrize("transcript,summary", [("", "x"), ("x", " "), (None, "x")])
def test_invalid_inputs_do_not_initialize_judge(monkeypatch, transcript, summary):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid inputs must not make a judge request")
    monkeypatch.setattr(engine, "LangChainJudge", unexpected)
    result = MeetingSummaryEvaluator(env_file=None).evaluate(transcript, summary)
    assert result.execution_status == "invalid"
    assert result.review_required


async def test_sync_wrapper_explains_active_loop(evaluator):
    with pytest.raises(RuntimeError, match="await evaluator"):
        evaluator.evaluate("source", "summary")


async def test_failed_metric_keeps_successes_and_hides_source(evaluator, monkeypatch):
    monkeypatch.setattr(engine, "create_metrics", lambda judge, names: {
        n: FakeMetric(n, fail=n == "coverage") for n in names
    })
    result = await evaluator.aevaluate("source", "summary")
    assert result.execution_status == "partial"
    assert result.metrics["faithfulness"].score == 1.0
    assert result.metrics["coverage"].score is None
    assert result.metrics["coverage"].status == "error"
    assert "Sensitive source" not in str(result.errors)
    assert result.review_required


async def test_diagnostics_failure_never_means_clean_review(evaluator):
    class Broken(FakeJudge):
        async def structured(self, *args):
            raise ValueError("Malformed response")
    evaluator._judge = Broken()
    result = await evaluator.aevaluate("source", "summary")
    assert result.execution_status == "partial"
    assert result.metadata["diagnostics_status"] == "error"
    assert result.review_required


async def test_high_scores_do_not_hide_material_findings(evaluator):
    class Critical(FakeJudge):
        async def structured(self, *args):
            return Diagnostics(findings=[Finding(kind="certainty_inflation", severity="high",
                description="A tentative source became a certainty.", evidence=[
                    Evidence(source="transcript", quote="It may rise.", verified=True, line_start=999),
                    Evidence(source="summary", quote="It will rise.")])])
    evaluator._judge = Critical()
    result = await evaluator.aevaluate("Speaker:\nIt may rise.", "It will rise.")
    assert result.execution_status == "completed"
    assert result.review_required
    assert result.findings[0].evidence[0].line_start == 2
    assert all(e.verified for e in result.findings[0].evidence)


async def test_no_claims_is_unscorable(evaluator, monkeypatch):
    def metrics(judge, names):
        metric = FakeMetric("faithfulness")
        metric.claims = []
        return {"faithfulness": metric}
    monkeypatch.setattr(engine, "create_metrics", metrics)
    evaluator.metric_names = ("faithfulness",)
    result = await evaluator.aevaluate("source", "The meeting ended.")
    assert result.metrics["faithfulness"].status == "not_applicable"
    assert result.metrics["faithfulness"].score is None
    assert result.review_required


async def test_metric_negative_verdict_requires_review_even_if_diagnostics_clean(evaluator, monkeypatch):
    def metrics(judge, names):
        metric = FakeMetric("faithfulness")
        metric.verdicts = [{"verdict": "borderline", "reason": "Support unclear"}]
        metric.score = 0.0
        return {"faithfulness": metric}
    monkeypatch.setattr(engine, "create_metrics", metrics)
    result = await evaluator.aevaluate("source", "summary")
    assert result.review_required
    assert result.findings == []
    assert any("Faithfulness" in reason for reason in result.review_reasons)


async def test_batch_pairing_stable_order_and_reports(evaluator, tmp_path):
    transcripts, summaries = tmp_path / "transcripts", tmp_path / "summaries"
    transcripts.mkdir()
    summaries.mkdir()
    for meeting in ("003", "001", "002"):
        (transcripts / f"{meeting}.txt").write_text("source", encoding="utf8")
        (summaries / f"{meeting}.md").write_text("summary", encoding="utf8")
    (summaries / "orphan.txt").write_text("unmatched", encoding="utf8")
    (transcripts / "002.md").write_text("duplicate", encoding="utf8")
    batch = await evaluator.aevaluate_batch(transcripts, summaries, tmp_path / "results")
    assert [r.meeting_id for r in batch.results] == ["001", "002", "003", "orphan"]
    assert [r.execution_status for r in batch.results] == ["completed", "invalid", "completed", "invalid"]
    assert {p.name for p in Path(batch.output_dir).iterdir()} >= {"results.json", "scores.csv", "report.html"}


async def test_batch_meeting_concurrency_is_bounded(evaluator, tmp_path, monkeypatch):
    active = peak = 0
    original = evaluator.aevaluate
    async def measured(*args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        result = await original(*args, **kwargs)
        active -= 1
        return result
    monkeypatch.setattr(evaluator, "aevaluate", measured)
    evaluator.max_concurrency = 2
    for directory in ("t", "s"):
        (tmp_path / directory).mkdir()
        for i in range(7):
            (tmp_path / directory / f"{i}.txt").write_text("source", encoding="utf8")
    batch = await evaluator.aevaluate_batch(tmp_path / "t", tmp_path / "s")
    assert len(batch.results) == 7
    assert peak == 2


async def test_global_configuration_failure_stops_queued_meetings(evaluator, tmp_path, monkeypatch):
    count = 0
    class BadMetric(FakeMetric):
        async def a_measure(self, case):
            nonlocal count
            count += 1
            raise engine.JudgeConfigurationError("unsupported model")
    monkeypatch.setattr(engine, "create_metrics", lambda judge, names: {n: BadMetric(n) for n in names})
    evaluator.max_concurrency = 1
    for directory in ("t", "s"):
        (tmp_path / directory).mkdir()
        for i in range(3):
            (tmp_path / directory / f"{i}.txt").write_text("source", encoding="utf8")
    batch = await evaluator.aevaluate_batch(tmp_path / "t", tmp_path / "s")
    assert count == 1
    assert all(r.execution_status == "error" for r in batch.results)
    assert all(r.review_required for r in batch.results)


def test_validation():
    for kwargs in ({"max_concurrency": 0}, {"metrics": []}, {"metrics": ["unknown"]}, {"metrics": ["structure"]}, {"max_retries": -1}):
        with pytest.raises(ValueError):
            MeetingSummaryEvaluator(env_file=None, **kwargs)


async def test_unexpected_pair_failure_does_not_discard_batch(evaluator, tmp_path, monkeypatch):
    original = evaluator.aevaluate
    async def flaky(transcript, summary, meeting_id, **kwargs):
        if meeting_id == "bad":
            raise RuntimeError("unexpected bug")
        return await original(transcript, summary, meeting_id, **kwargs)
    monkeypatch.setattr(evaluator, "aevaluate", flaky)
    for directory in ("t", "s"):
        (tmp_path / directory).mkdir()
        for meeting in ("bad", "good"):
            (tmp_path / directory / f"{meeting}.txt").write_text("source", encoding="utf8")
    batch = await evaluator.aevaluate_batch(tmp_path / "t", tmp_path / "s")
    assert [r.execution_status for r in batch.results] == ["error", "completed"]


async def test_async_methods_reject_closed_evaluator(evaluator, tmp_path):
    await evaluator.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await evaluator.aevaluate("source", "summary")
    with pytest.raises(RuntimeError, match="closed"):
        await evaluator.aevaluate_batch(tmp_path, tmp_path)


async def test_response_format_configuration_error_stops_entire_batch(evaluator, tmp_path, monkeypatch):
    import httpx2
    from openai import BadRequestError

    calls = []
    class Unsupported(FakeMetric):
        async def a_measure(self, case):
            calls.append(self.name)
            raise BadRequestError("Sensitive source must stay out of reports", response=httpx2.Response(
                400, request=httpx2.Request("POST", "https://example.invalid")),
                body={"type": "invalid_request_error", "param": "response_format", "code": None})
    monkeypatch.setattr(engine, "create_metrics", lambda judge, names: {n: Unsupported(n) for n in names})
    evaluator.max_concurrency = 1
    for directory in ("t", "s"):
        (tmp_path / directory).mkdir()
        for index in range(3):
            (tmp_path / directory / f"{index}.txt").write_text("Source")
    batch = await evaluator.aevaluate_batch(tmp_path / "t", tmp_path / "s")
    assert calls == ["faithfulness"]
    assert all(result.execution_status == "error" for result in batch.results)
    assert "Sensitive source" not in batch.model_dump_json()


def test_context_limit_error_remains_meeting_specific():
    import httpx2
    from openai import BadRequestError

    error = BadRequestError("Context too long", response=httpx2.Response(
        400, request=httpx2.Request("POST", "https://example.invalid")),
        body={"type": "invalid_request_error", "param": "messages", "code": "context_length_exceeded"})
    assert not engine._global_error(error)


async def test_async_context_manager_closes_evaluator(evaluator):
    async with evaluator:
        result = await evaluator.aevaluate("Source", "Summary")
        assert result.execution_status == "completed"
    with pytest.raises(RuntimeError, match="closed"):
        await evaluator.aevaluate("Source", "Summary")


@pytest.mark.parametrize("summary", [
    "The launch is paused pending review. Rae owns the follow-up by Thursday.",
    "- Rae will follow up by Thursday.\n- Launch is paused pending review.",
    "# Team update\nThe launch is paused pending review.\n\n## Next step\nRae will follow up by Thursday.",
])
async def test_summary_format_does_not_create_a_metric_or_review_finding(evaluator, summary):
    transcript = "Rae: The launch is paused pending review. I will follow up by Thursday."
    result = await evaluator.aevaluate(transcript, summary)
    assert result.execution_status == "completed"
    assert set(result.metrics) == {"faithfulness", "coverage"}
    assert result.findings == []
    assert not result.review_required
    assert result.metadata["business_context_sha256"] is None
