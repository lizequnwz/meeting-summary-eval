"""Offline judge responses exercise DeepEval without contacting a provider."""

import os
import socket

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"
os.environ["DEEPEVAL_DISABLE_LEGACY_KEYFILE"] = "1"
os.environ["OPENAI_API_KEY"] = "offline-test-key"
os.environ["LANGSMITH_TRACING"] = "false"

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Tests must not make network connections")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")


class ScriptedJudge(BaseChatModel):
    """Exercise LangChain's real structured-output runnable and parser offline."""

    model_config = ConfigDict(extra="allow")
    model_name: str = "offline-scripted-judge"
    prompts: list = Field(default_factory=list)
    verdicts: list[str] = Field(default_factory=lambda: ["yes", "no", "borderline"])
    summary_answers: list[str] = Field(default_factory=lambda: ["yes", "no"])
    fail_source: str | None = None

    @property
    def _llm_type(self):
        return "offline-scripted-judge"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.bind(schema=tools[0])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        schema = kwargs["schema"]
        prompt = "\n".join(str(message.content) for message in messages)
        parsed = self.respond(prompt, schema)
        raw = AIMessage(
            content="",
            tool_calls=[{"id": "offline-call", "name": schema.__name__, "args": parsed.model_dump(mode="json")}],
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        return ChatResult(generations=[ChatGeneration(message=raw)])

    def respond(self, prompt, schema):
        stage = schema.__name__
        self.prompts.append((stage, prompt))
        if stage == "Truths" and self.fail_source and self.fail_source in prompt:
            raise RuntimeError("Scripted judge failure")
        source = "SourceBeta" if "SourceBeta" in prompt else "SourceAlpha"
        data = {
            "Truths": {"truths": ["The transcript records three facts."]},
            "Claims": {"claims": ["First claim", "Second claim", "Third claim"]},
            "Questions": {"questions": [f"Does {source} explain the outlook?", f"Does {source} identify a risk?"]},
            "Answers": {"answers": self.summary_answers if "SummaryAlpha" in prompt or "SummaryBeta" in prompt else ["yes", "yes"]},
            "Verdicts": {"verdicts": [
                {"verdict": value, "reason": None if value == "yes" else f"The claim is {value}."}
                for value in self.verdicts
            ]},
            "SummarizationScoreReason": {"reason": "Inspect the native claim and coverage verdicts."},
        }
        return schema.model_validate(data[stage])

@pytest.fixture
def judge():
    return ScriptedJudge()
