"""Offline judge responses exercise DeepEval without contacting a provider."""

import os
import socket

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"
os.environ["DEEPEVAL_DISABLE_LEGACY_KEYFILE"] = "1"
os.environ["OPENAI_API_KEY"] = "offline-test-key"

import pytest
from deepeval.models import DeepEvalBaseLLM


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Tests must not make network connections")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")


class ScriptedJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.prompts = []
        self.verdicts = ["yes", "no", "borderline"]
        self.summary_answers = ["yes", "no"]
        self.fail_source = None

    def load_model(self):
        return self

    def get_model_name(self):
        return "offline-scripted-judge"

    def generate(self, prompt, schema=None):
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

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)


@pytest.fixture
def judge():
    return ScriptedJudge()
