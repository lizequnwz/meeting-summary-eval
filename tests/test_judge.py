"""Exercise the adapter contract and its real request-concurrency bound offline."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from meeting_eval.judge import LangChainJudge


def test_ambient_tracing_is_disabled_for_judge_calls(monkeypatch):
    from langsmith.utils import tracing_is_enabled
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    chat = chat_model()
    observed = []
    def sync_invoke(messages):
        observed.append(tracing_is_enabled())
        return AIMessage(content="answer")
    async def async_invoke(messages):
        observed.append(tracing_is_enabled())
        return AIMessage(content="answer")
    chat.invoke = sync_invoke
    chat.ainvoke = async_invoke
    judge = LangChainJudge(chat)
    judge.generate("prompt")
    asyncio.run(judge.a_generate("prompt"))
    assert observed == [False, False]


class ExampleResponse(BaseModel):
    answer: str


def chat_model():
    chat = MagicMock(spec=ChatOpenAI)
    chat.model_name = "offline-judge"
    chat.invoke.return_value = AIMessage(content="plain answer")
    chat.ainvoke = AsyncMock(return_value=AIMessage(content="plain answer"))
    structured = {"raw": AIMessage(content="typed"), "parsed": ExampleResponse(answer="typed"), "parsing_error": None}
    chat.with_structured_output.return_value.invoke.return_value = structured
    chat.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value=structured
    )
    return chat


def test_sync_and_async_schema_contract():
    chat = chat_model()
    judge = LangChainJudge(chat)
    assert judge.load_model() is chat
    assert judge.get_model_name() == "offline-judge"
    assert judge.generate("prompt") == "plain answer"
    assert judge.generate("prompt", ExampleResponse) == ExampleResponse(answer="typed")
    assert asyncio.run(judge.a_generate("prompt")) == "plain answer"
    assert asyncio.run(judge.structured("prompt", ExampleResponse)) == ExampleResponse(answer="typed")
    chat.with_structured_output.assert_called_with(
        ExampleResponse, method="json_schema", strict=True, include_raw=True
    )
    messages = chat.invoke.call_args.args[0]
    assert messages[0].type == "system"
    assert "untrusted data" in messages[0].content
    assert messages[1].content == "prompt"


def test_requests_are_bounded_and_adapter_can_reuse_sequential_event_loops():
    chat = chat_model()
    active = 0
    peak = 0

    async def invoke(messages):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.001)
            return AIMessage(content=messages[-1].content)
        finally:
            active -= 1

    chat.ainvoke = invoke
    judge = LangChainJudge(chat, max_concurrency=2)

    async def run():
        return await asyncio.gather(*(judge.a_generate(str(i)) for i in range(8)))

    assert asyncio.run(run()) == [str(i) for i in range(8)]
    assert asyncio.run(run()) == [str(i) for i in range(8)]
    assert peak == 2
    assert active == 0


def test_structured_diagnostics_share_metric_request_limit():
    chat = chat_model()
    active = 0
    peak = 0

    async def invoke(messages):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.001)
        active -= 1
        return AIMessage(content=messages[-1].content)

    async def structured(messages):
        await invoke(messages)
        return {"raw": AIMessage(content="typed"), "parsed": ExampleResponse(answer=messages[-1].content), "parsing_error": None}

    chat.ainvoke = invoke
    chat.with_structured_output.return_value.ainvoke = structured
    judge = LangChainJudge(chat, max_concurrency=1)

    async def run():
        await asyncio.gather(
            judge.a_generate("metric"),
            judge.structured("diagnostic", ExampleResponse),
        )

    asyncio.run(run())
    assert peak == 1


@pytest.mark.parametrize("kwargs", [
    {"max_concurrency": 0}, {"timeout": 0}, {"max_retries": -1},
])
def test_invalid_limits_fail_before_creating_client(kwargs):
    with pytest.raises(ValueError):
        LangChainJudge(chat_model(), **kwargs)


def test_only_owned_model_clients_are_closed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-key")
    owned = LangChainJudge("offline-model")
    assert owned.model.root_async_client._client is owned.model.http_async_client
    assert owned.model.root_client._client is owned.model.http_client
    asyncio.run(owned.aclose())
    assert owned.model.http_async_client.is_closed
    assert owned.model.http_client.is_closed

    external = chat_model()
    external.root_async_client = MagicMock()
    external.root_client = MagicMock()
    external.root_async_client.close = AsyncMock()
    judge = LangChainJudge(external)
    asyncio.run(judge.aclose())
    external.root_async_client.close.assert_not_called()
    external.root_client.close.assert_not_called()


def test_real_langchain_client_can_repeat_requests_on_one_runner():
    # The transport models an async HTTP pool that belongs to the first loop it
    # uses, without opening sockets or calling the model endpoint.
    import json
    import httpx2
    from openai import DefaultAsyncHttpxClient

    first_loop = None

    async def respond(request):
        nonlocal first_loop
        loop = asyncio.get_running_loop()
        if first_loop is None:
            first_loop = loop
        assert loop is first_loop, "An async HTTP pool crossed event loops"
        payload = json.loads(request.content)
        messages = payload["messages"]
        assert messages[0]["role"] == "system"
        return httpx2.Response(200, json={
            "id": "offline-completion", "object": "chat.completion", "created": 0,
            "model": "resolved-offline-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            "choices": [{"index": 0, "message": {"role": "assistant", "content":
                '{"answer": "typed"}' if "response_format" in payload else "answer"}, "finish_reason": "stop"}],
        })

    client = DefaultAsyncHttpxClient(transport=httpx2.MockTransport(respond))
    model = ChatOpenAI(model="offline-model", api_key="offline-key", http_async_client=client)
    judge = LangChainJudge(model)

    async def call(prompt, schema=None):
        with judge.usage_scope() as usage:
            result = await judge.a_generate(prompt, schema)
        return result, usage

    with asyncio.Runner() as runner:
        first, first_usage = runner.run(call("first"))
        second, second_usage = runner.run(call("second"))
        third, third_usage = runner.run(call("third", ExampleResponse))
        assert first == second == "answer"
        assert third == ExampleResponse(answer="typed")
        for usage in (first_usage, second_usage, third_usage):
            assert usage["request_count"] == 1
            assert usage["total_tokens"] == 12
            assert usage["response_models"] == ["resolved-offline-model"]
        runner.run(client.aclose())


def test_evaluator_reuses_async_client_loop_across_sync_calls():
    import json
    import httpx2
    from openai import DefaultAsyncHttpxClient
    from meeting_eval import MeetingSummaryEvaluator

    first_loop = None
    count = 0

    async def respond(request):
        nonlocal first_loop, count
        loop = asyncio.get_running_loop()
        first_loop = first_loop or loop
        assert loop is first_loop, "Evaluator reused its HTTP pool on another loop"
        count += 1
        assert json.loads(request.content)["response_format"]["type"] == "json_schema"
        return httpx2.Response(200, json={
            "id": "offline-completion", "object": "chat.completion", "created": 0,
            "model": "offline-model",
            "choices": [{"index": 0, "message": {
                "role": "assistant", "content": '{"qa_present": false, "findings": []}'
            }, "finish_reason": "stop"}],
        })

    client = DefaultAsyncHttpxClient(transport=httpx2.MockTransport(respond))
    model = ChatOpenAI(model="offline-model", api_key="offline-key", http_async_client=client)
    with MeetingSummaryEvaluator(model=model, metrics=("structure",), env_file=None) as evaluator:
        first = evaluator.evaluate("Presenter: Rates are unchanged.", "A summary.")
        second = evaluator.evaluate("Presenter: Rates are unchanged.", "Another summary.")
        assert first.execution_status == "completed"
        assert second.execution_status == "completed"
        assert count == 2
    assert not client.is_closed  # A supplied model retains ownership of its pool.
    with pytest.raises(RuntimeError, match="closed"):
        evaluator.evaluate("Transcript", "Summary")
    asyncio.run(client.aclose())


def usage_message(model="resolved-offline-model", input_tokens=11, output_tokens=3):
    return AIMessage(content="private source text must not be retained",
                     usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens,
                                     "total_tokens": input_tokens + output_tokens},
                     response_metadata={"model_name": model, "private": "secret"})


def test_usage_scope_tracks_structured_and_plain_calls_without_retaining_content():
    import json

    chat = chat_model()
    chat.invoke.return_value = usage_message()
    chat.with_structured_output.return_value.invoke.return_value = {
        "raw": usage_message(), "parsed": ExampleResponse(answer="private parsed result"), "parsing_error": None,
    }
    judge = LangChainJudge(chat)
    with judge.usage_scope() as usage:
        with judge.stage("faithfulness"):
            judge.generate("private prompt")
        with judge.stage("diagnostics"):
            judge.generate("private prompt", ExampleResponse)
    assert usage["request_count"] == usage["successful_requests"] == 2
    assert usage["failed_requests"] == 0
    assert usage["input_tokens"] == 22
    assert usage["output_tokens"] == 6
    assert usage["total_tokens"] == 28
    assert usage["response_models"] == ["resolved-offline-model"]
    assert usage["stages"]["faithfulness"]["request_count"] == 1
    assert usage["stages"]["diagnostics"]["total_tokens"] == 14
    assert usage["elapsed_seconds"] >= 0
    assert "private" not in json.dumps(usage)
    assert "secret" not in json.dumps(usage)


async def test_concurrent_meeting_scopes_are_isolated_and_child_calls_share_counters():
    chat = chat_model()

    async def invoke(messages):
        await asyncio.sleep(0)
        return usage_message(model=messages[-1].content)

    chat.ainvoke = invoke
    judge = LangChainJudge(chat, max_concurrency=2)

    async def meeting(name, count):
        with judge.usage_scope() as usage:
            with judge.stage("faithfulness"):
                await asyncio.gather(*(judge.a_generate(name) for _ in range(count)))
        return usage

    first, second = await asyncio.gather(meeting("one", 2), meeting("two", 3))
    assert first["request_count"] == 2 and second["request_count"] == 3
    assert first["response_models"] == ["one"]
    assert second["response_models"] == ["two"]
    assert first["input_tokens"] == 22 and second["input_tokens"] == 33
    await judge.a_generate("outside-scope")
    assert first["request_count"] == 2 and second["request_count"] == 3


def test_unknown_usage_and_failures_are_not_reported_as_zero_token_cost():
    chat = chat_model()
    chat.invoke.side_effect = [usage_message(), TimeoutError("private request body"), AIMessage(content="answer")]
    judge = LangChainJudge(chat)
    with judge.usage_scope() as usage:
        judge.generate("one")
        with pytest.raises(TimeoutError):
            judge.generate("two")
        judge.generate("three")
    assert usage["request_count"] == 3
    assert usage["successful_requests"] == 2
    assert usage["failed_requests"] == 1
    assert usage["requests_with_token_usage"] == 1
    assert usage["input_tokens"] is usage["output_tokens"] is usage["total_tokens"] is None
    with judge.usage_scope() as empty:
        pass
    assert empty["request_count"] == 0 and empty["total_tokens"] is None


@pytest.mark.parametrize("response", [
    {"parsed": None, "parsing_error": None},
    {"parsed": ExampleResponse(answer="answer"), "parsing_error": ValueError("private malformed output")},
    {"parsed": {"answer": "unvalidated"}, "parsing_error": None},
])
def test_invalid_structured_responses_fail_and_preserve_available_usage(response):
    chat = chat_model()
    chat.with_structured_output.return_value.invoke.return_value = {"raw": usage_message(), **response}
    judge = LangChainJudge(chat)
    with judge.usage_scope() as usage:
        with pytest.raises(ValueError, match="invalid structured"):
            judge.generate("prompt", ExampleResponse)
    assert usage["failed_requests"] == 1
    assert usage["successful_requests"] == 0
    assert usage["total_tokens"] == 14


async def test_async_refusal_is_a_failed_invocation():
    chat = chat_model()
    raw = usage_message()
    raw.additional_kwargs["refusal"] = "private refusal"
    chat.with_structured_output.return_value.ainvoke.return_value = {
        "raw": raw, "parsed": None, "parsing_error": None,
    }
    judge = LangChainJudge(chat)
    with judge.usage_scope() as usage:
        with pytest.raises(ValueError, match="refusal"):
            await judge.structured("prompt", ExampleResponse)
    assert usage["failed_requests"] == 1 and usage["total_tokens"] == 14
