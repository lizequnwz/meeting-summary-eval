"""Exercise the complete evaluation and comparison pipeline without network calls."""

import asyncio
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import httpx2
import pytest
from langchain_openai import ChatOpenAI
from openai import DefaultAsyncHttpxClient, DefaultHttpxClient

from meeting_eval import BatchResult, MeetingSummaryEvaluator
from meeting_eval.comparison import compare_batches


async def test_offline_provider_to_reports_and_paired_comparison(tmp_path):
    transcripts, contexts = tmp_path / "transcripts", tmp_path / "contexts"
    baseline_summaries, candidate_summaries = tmp_path / "baseline", tmp_path / "candidate"
    for directory in (transcripts, contexts, baseline_summaries, candidate_summaries):
        directory.mkdir()

    identifiers = {"alpha": "PipelineAlpha", "beta": "PipelineBeta"}
    context_quotes = {name: f"{identifier}: RIC may mean Regional Investment Consultant."
                      for name, identifier in identifiers.items()}
    for meeting_id, identifier in identifiers.items():
        (transcripts / f"{meeting_id}.txt").write_text(f"{identifier}: Rates are unchanged.\n", encoding="utf-8")
        (contexts / f"{meeting_id}.txt").write_text(
            "Supplementary terminology context.\n" + context_quotes[meeting_id] + "\n", encoding="utf-8")
        summary = f"""## Executive Briefing
- {identifier} reported unchanged rates.
## Economy & Market Updates
- Rates are unchanged.
## Advisor Outreach & Sales Actions
- Suggested outreach: Explore whether unchanged rates affect advisor conversations.
## Market & Fidelity Perspectives
- The supplied context says RIC may mean Regional Investment Consultant for {identifier}.
## Questions and Answers
No substantive Q&A occurred.
## Search Tags
Interest Rates, Advisor Conversations
"""
        (baseline_summaries / f"{meeting_id}.md").write_text(summary, encoding="utf-8")
        (candidate_summaries / f"{meeting_id}.md").write_text(
            summary.replace("reported unchanged rates", "reported that rates remain unchanged"), encoding="utf-8")

    active = peak = 0
    schema_calls = Counter()
    routed_calls = Counter()

    async def respond(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0)  # Permit the real metric/batch coroutines to overlap.
            payload = json.loads(request.content)
            assert payload["response_format"]["type"] == "json_schema"
            schema_name = payload["response_format"]["json_schema"]["name"]
            assert payload["response_format"]["json_schema"]["strict"] is True
            assert payload["messages"][0]["role"] == "system"
            prompt = payload["messages"][-1]["content"]
            schema_calls[schema_name] += 1
            matched = [name for name, identifier in identifiers.items() if identifier in prompt]
            if schema_name == "FaithfulnessScoreReason":
                # The reason call sees only score/verdict reasons, not raw inputs.
                assert not matched
                meeting_id, identifier = None, None
                input_tokens, output_tokens = 3, 2
            else:
                assert len(matched) == 1, "A provider request mixed or lost meeting identity"
                meeting_id = matched[0]
                identifier = identifiers[meeting_id]
                routed_calls[meeting_id] += 1
                input_tokens, output_tokens = (11, 3) if meeting_id == "alpha" else (21, 4)
            if schema_name in {"Truths", "ReasonScore", "Diagnostics"}:
                assert context_quotes[meeting_id] in prompt
            if schema_name == "Truths":
                result = {"truths": [f"transcript: {identifier}: Rates are unchanged.",
                                      f"context: {context_quotes[meeting_id]}"]}
            elif schema_name == "Claims":
                result = {"claims": [f"{identifier} reported unchanged rates.",
                                      f"The context says {context_quotes[meeting_id]}"]}
            elif schema_name == "Verdicts":
                result = {"verdicts": [{"verdict": "yes", "reason": None},
                                       {"verdict": "yes", "reason": None}]}
            elif schema_name == "FaithfulnessScoreReason":
                result = {"reason": "The extracted claims received no negative verdicts."}
            elif schema_name == "ReasonScore":
                result = {"score": 8, "reason": "The core rate observation is represented."}
            elif schema_name == "Diagnostics":
                result = {"qa_present": False, "findings": [{
                    "kind": "uncertainty", "severity": "medium",
                    "description": "The supplied role expansion is explicitly tentative.",
                    "section": "Market & Fidelity Perspectives", "suggested_prompt_change": None,
                    "evidence": [{"source": "context", "quote": context_quotes[meeting_id],
                                  "line_start": None, "line_end": None, "verified": False}],
                }]}
            else:
                pytest.fail(f"Unexpected provider response schema: {schema_name}")
            return httpx2.Response(200, json={
                "id": "offline-pipeline-response", "object": "chat.completion", "created": 0,
                "model": "resolved-offline-pipeline-model",
                "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                          "total_tokens": input_tokens + output_tokens},
                "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(result)},
                             "finish_reason": "stop"}],
            })
        finally:
            active -= 1

    async_client = DefaultAsyncHttpxClient(transport=httpx2.MockTransport(respond))
    sync_client = DefaultHttpxClient(transport=httpx2.MockTransport(
        lambda request: pytest.fail("The asynchronous pipeline attempted a synchronous provider call")))
    model = ChatOpenAI(model="offline-pipeline-model", api_key="offline-key",
                       base_url="https://pipeline.invalid/v1", timeout=10, max_retries=0,
                       http_async_client=async_client, http_client=sync_client)
    try:
        async with MeetingSummaryEvaluator(model=model, env_file=None, max_concurrency=2) as evaluator:
            baseline = await evaluator.aevaluate_batch(
                transcripts, baseline_summaries, tmp_path / "reports", context_dir=contexts)
            candidate = await evaluator.aevaluate_batch(
                transcripts, candidate_summaries, tmp_path / "reports", context_dir=contexts)
        assert not async_client.is_closed and not sync_client.is_closed
    finally:
        await async_client.aclose()
        sync_client.close()

    assert active == 0 and peak == 2
    assert schema_calls == Counter({name: 4 for name in (
        "Truths", "Claims", "Verdicts", "FaithfulnessScoreReason", "ReasonScore", "Diagnostics")})
    assert routed_calls == {"alpha": 10, "beta": 10}

    saved_batches = []
    for batch in (baseline, candidate):
        assert len(batch.results) == 2
        for result in batch.results:
            assert result.execution_status == "completed", result.errors
            assert result.metrics["faithfulness"].score == 1
            assert result.metrics["coverage"].score == pytest.approx(0.8)
            usage = result.metadata["usage"]
            assert usage["request_count"] == usage["successful_requests"] == usage["requests_with_token_usage"] == 6
            assert usage["failed_requests"] == 0
            assert usage["total_tokens"] == (75 if result.meeting_id == "alpha" else 130)
            assert usage["response_models"] == ["resolved-offline-pipeline-model"]
            assert {stage: item["request_count"] for stage, item in usage["stages"].items()} == {
                "faithfulness": 4, "coverage": 1, "diagnostics": 1}
            evidence = result.findings[0].evidence[0]
            assert evidence.source == "context" and evidence.quote == context_quotes[result.meeting_id]
            assert evidence.verified and evidence.line_start == evidence.line_end == 2
            assert result.metadata["context_sha256"] == hashlib.sha256(
                (contexts / f"{result.meeting_id}.txt").read_bytes()).hexdigest()
            assert result.metadata["context_path"] == str((contexts / f"{result.meeting_id}.txt").resolve())
            assert result.review_required
        run_dir = Path(batch.output_dir)
        saved = BatchResult.model_validate_json((run_dir / "results.json").read_text(encoding="utf-8"))
        saved_batches.append(saved)
        with (run_dir / "scores.csv").open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        assert [row["meeting_id"] for row in rows] == ["alpha", "beta"]
        assert all(float(row["coverage_score"]) == 0.8 for row in rows)
        report = (run_dir / "report.html").read_text(encoding="utf-8")
        assert "resolved-offline-pipeline-model" in report
        assert "evaluator_fingerprint" in report
        assert all(quote in report for quote in context_quotes.values())

    comparison = compare_batches(*saved_batches)
    assert comparison["meeting_count"] == comparison["complete_pair_count"] == 2
    for metric in ("faithfulness", "coverage"):
        assert comparison["metric_deltas"][metric] == {"paired_count": 2, "mean_delta": 0, "median_delta": 0}
    for before, after in zip(baseline.results, candidate.results, strict=True):
        assert before.metadata["summary_sha256"] != after.metadata["summary_sha256"]
        assert before.metadata["transcript_sha256"] == after.metadata["transcript_sha256"]
        assert before.metadata["context_sha256"] == after.metadata["context_sha256"]
