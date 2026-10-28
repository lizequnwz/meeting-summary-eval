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
    chat.with_structured_output.return_value.invoke.return_value = ExampleResponse(answer="typed")
    chat.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value=ExampleResponse(answer="typed")
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
        ExampleResponse, method="json_schema", strict=True
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
        return ExampleResponse(answer=messages[-1].content)

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
        messages = json.loads(request.content)["messages"]
        assert messages[0]["role"] == "system"
        return httpx2.Response(200, json={
            "id": "offline-completion", "object": "chat.completion", "created": 0,
            "model": "offline-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}],
        })

    client = DefaultAsyncHttpxClient(transport=httpx2.MockTransport(respond))
    model = ChatOpenAI(model="offline-model", api_key="offline-key", http_async_client=client)
    judge = LangChainJudge(model)
    with asyncio.Runner() as runner:
        assert runner.run(judge.a_generate("first")) == "answer"
        assert runner.run(judge.a_generate("second")) == "answer"
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
