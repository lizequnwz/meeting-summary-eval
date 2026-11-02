"""Exercise the LangChain bridge and actual ChatOpenAI requests without networking."""

import asyncio
import json

import httpx
import pytest
from deepeval.metrics.summarization.schema import Questions
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from conftest import ScriptedJudge
from meeting_eval.judge import LangChainJudge
from meeting_eval.questions import get_or_prepare_questions


def completion(message):
    return {
        "id": "offline-completion",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


@pytest.fixture
def transport_model():
    clients = []

    def build(handler):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append((client, async_client))
        return ChatOpenAI(model="gpt-4o", api_key="offline-transport-key",
                          base_url="https://judge.invalid/custom/v1", max_retries=0,
                          http_client=client, http_async_client=async_client)

    yield build
    for client, async_client in clients:
        client.close()
        asyncio.run(async_client.aclose())


@pytest.mark.parametrize("asynchronous", [False, True])
def test_actual_chatopenai_preserves_prompt_schema_key_and_endpoint(transport_model, asynchronous):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=completion({"content": json.dumps({"questions": ["Was inflation discussed?"]})}))

    adapter = LangChainJudge(transport_model(handler))
    if asynchronous:
        result = asyncio.run(adapter.a_generate("The exact native prompt.", schema=Questions))
    else:
        result = adapter.generate("The exact native prompt.", schema=Questions)

    assert isinstance(result, Questions)
    assert result.questions == ["Was inflation discussed?"]
    assert len(requests) == 1
    request = requests[0]
    body = json.loads(request.content)
    assert str(request.url) == "https://judge.invalid/custom/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer offline-transport-key"
    assert body["model"] == "gpt-4o"
    assert body["messages"] == [{"content": "The exact native prompt.", "role": "user"}]
    assert body["response_format"]["type"] == "json_schema"
    schema = body["response_format"]["json_schema"]
    assert schema["name"] == "Questions"
    assert schema["schema"]["properties"]["questions"]["items"]["type"] == "string"
    assert adapter.get_model_name() == "gpt-4o"
    assert adapter.base_url == "https://judge.invalid/custom/v1"


@pytest.mark.parametrize("message", [
    {"content": '{"questions":'},
    {"content": json.dumps({"questions": [42]})},
    {"content": None, "refusal": "Private provider refusal"},
])
def test_actual_chatopenai_invalid_output_is_sanitized_and_never_saved(tmp_path, transport_model, message):
    model = transport_model(lambda request: httpx.Response(200, json=completion(message)))
    with pytest.raises(ValueError, match="Question generation failed") as error:
        asyncio.run(get_or_prepare_questions("meeting", "Source text", model=model, n=1, questions_dir=tmp_path))
    assert "Private provider refusal" not in str(error.value)
    assert list(tmp_path.iterdir()) == []


def test_actual_chatopenai_authentication_failure_is_sanitized(tmp_path, transport_model):
    def handler(request):
        return httpx.Response(401, json={"error": {"message": "Private source and credential", "type": "authentication_error"}})

    model = transport_model(handler)
    with pytest.raises(ValueError, match="Question generation failed") as error:
        asyncio.run(get_or_prepare_questions("meeting", "Source text", model=model, n=1, questions_dir=tmp_path))
    assert "Private source" not in str(error.value)
    assert list(tmp_path.iterdir()) == []


class EnvelopeModel(ScriptedJudge):
    envelope: object = None

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        assert include_raw is True
        return RunnableLambda(lambda prompt: self.envelope)


class OtherSchema(BaseModel):
    value: str


@pytest.mark.parametrize("envelope", [
    {"raw": AIMessage(content=""), "parsed": None, "parsing_error": None},
    {"raw": AIMessage(content=""), "parsed": OtherSchema(value="x"), "parsing_error": None},
    {"raw": AIMessage(content=""), "parsed": Questions(questions=["Question?"]), "parsing_error": ValueError("Private parse details")},
    {"raw": AIMessage(content="", additional_kwargs={"refusal": "Private refusal"}), "parsed": Questions(questions=["Question?"]), "parsing_error": None},
    {"raw": AIMessage(content=[{"type": "refusal", "refusal": "Private refusal"}]), "parsed": Questions(questions=["Question?"]), "parsing_error": None},
])
def test_custom_langchain_model_cannot_return_invalid_or_refused_schema(envelope):
    adapter = LangChainJudge(EnvelopeModel(envelope=envelope))
    with pytest.raises(ValueError, match="Judge") as error:
        adapter.generate("Prompt", schema=Questions)
    assert "Private" not in str(error.value)


def test_judge_requires_schema_before_invocation(judge):
    adapter = LangChainJudge(judge)
    with pytest.raises(ValueError, match="Pydantic response schema"):
        adapter.generate("Prompt")
    assert judge.prompts == []


def test_metadata_uses_known_strings_without_serializing_configuration(judge):
    judge.model_name = ""
    judge.model = "custom-model"
    judge.base_url = "https://custom.invalid/v1"
    adapter = LangChainJudge(judge)
    assert adapter.get_model_name() == "custom-model"
    assert adapter.base_url == "https://custom.invalid/v1"
    assert adapter.load_model() is judge

    judge.model = ""
    judge.deployment_name = "custom-deployment"
    assert adapter.get_model_name() == "custom-deployment"
    judge.deployment_name = ""
    assert adapter.get_model_name() == "ScriptedJudge"


def test_non_langchain_objects_are_rejected():
    with pytest.raises(TypeError, match="LangChain BaseChatModel"):
        LangChainJudge(object())
