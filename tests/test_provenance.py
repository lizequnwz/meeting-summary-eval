"""Provenance includes reproducible settings without credential-bearing fields."""

import json
from pathlib import Path

from langchain_openai import ChatOpenAI

from meeting_eval.provenance import evaluator_fingerprint, runtime_metadata


def test_string_model_metadata_sanitizes_endpoint_and_is_json_serializable(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://user:secret@endpoint.invalid:8443/v1/?api_key=secret#secret")
    metadata = runtime_metadata("offline-model", timeout=15, max_retries=1)
    assert metadata["model"] == "offline-model"
    assert metadata["judge_config"]["endpoint"] == "https://endpoint.invalid:8443/v1"
    assert metadata["judge_config"]["timeout"] == 15
    assert metadata["judge_config"]["max_retries"] == 1
    assert set(metadata["versions"]) >= {"python", "openai", "langchain-core", "pydantic"}
    assert len(metadata["evaluator_fingerprint"]) == 64
    assert "secret" not in json.dumps(metadata)


def test_supplied_model_uses_effective_settings_without_dumping_arbitrary_data():
    model = ChatOpenAI(model="offline-model", api_key="private-api-key",
                       base_url="https://user:private-password@models.invalid/v1?token=private-query",
                       timeout=7, max_retries=1, temperature=0.3, top_p=0.9, max_tokens=321,
                       reasoning={"effort": "low", "summary": "auto"},
                       default_headers={"private-header": "private-header-value"},
                       model_kwargs={"user": "private-user"},
                       extra_body={"temperature": 0.5, "private-extra": "private-extra-value"})
    metadata = runtime_metadata(model, timeout=120, max_retries=20)
    settings = metadata["judge_config"]
    assert settings["timeout"] == 7 and settings["max_retries"] == 1
    assert settings["endpoint"] == "https://models.invalid/v1"
    assert settings["generation_parameters"]["temperature"] == 0.3
    assert settings["generation_parameters"]["reasoning"] == {"effort": "low", "summary": "auto"}
    assert settings["generation_parameters"]["max_completion_tokens"] == 321
    assert settings["extra_body_generation_parameters"] == {"temperature": 0.5}
    assert settings["custom_parameters_present"] is True
    assert "private" not in json.dumps(metadata)
    model.root_client.close()
    import asyncio
    asyncio.run(model.root_async_client.close())


def test_allowlisted_custom_settings_are_fully_recorded():
    model = ChatOpenAI(model="offline", api_key="offline", extra_body={"temperature": 0.7})
    metadata = runtime_metadata(model)
    assert metadata["judge_config"]["extra_body_generation_parameters"] == {"temperature": 0.7}
    assert metadata["judge_config"]["custom_parameters_present"] is False
    model.root_client.close()
    import asyncio
    asyncio.run(model.root_async_client.close())


def test_fingerprint_is_stable_and_reloads_changed_evaluation_sources(monkeypatch):
    first = evaluator_fingerprint()
    assert evaluator_fingerprint() == first
    read_bytes = Path.read_bytes

    def changed(path):
        content = read_bytes(path)
        return content + b"\n# changed metric policy" if path.name == "metrics.py" else content

    monkeypatch.setattr(Path, "read_bytes", changed)
    assert evaluator_fingerprint() != first


def test_fingerprint_tracks_policy_resource_and_schema(monkeypatch):
    import meeting_eval.policy as policy
    from meeting_eval.models import Diagnostics

    first = evaluator_fingerprint()
    monkeypatch.setattr(policy, "policy_fingerprint", lambda: "changed-policy")
    second = evaluator_fingerprint()
    assert second != first
    original = Diagnostics.model_json_schema()
    monkeypatch.setattr(Diagnostics, "model_json_schema", lambda: {**original, "description": "changed schema"})
    assert evaluator_fingerprint() != second
