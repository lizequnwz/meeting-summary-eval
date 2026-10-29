"""Safe judge configuration and reproducible evaluator identity."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from langchain_openai import ChatOpenAI


def _endpoint(value) -> str | None:
    """Keep endpoint identity without authentication, query, or fragment values."""
    if value is None:
        return None
    try:
        parsed = urlsplit(str(value))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return None
        hostname = parsed.hostname
        if ":" in hostname:
            hostname = f"[{hostname}]"
        authority = hostname + (f":{parsed.port}" if parsed.port is not None else "")
        return urlunsplit((parsed.scheme, authority, parsed.path.rstrip("/"), "", ""))
    except ValueError:
        return None


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _timeout(value):
    if value is None or _number(value) is not None:
        return value
    if isinstance(value, tuple):
        return [_number(item) for item in value]
    # Both installed OpenAI HTTP clients expose these timeout attributes.
    return {name: _number(getattr(value, name, None)) for name in ("connect", "read", "write", "pool")}


def _generation_parameters(parameters: dict) -> dict:
    safe = {}
    for name in ("temperature", "top_p", "seed", "max_tokens", "max_completion_tokens",
                 "max_output_tokens", "presence_penalty", "frequency_penalty", "top_logprobs", "n"):
        if name in parameters and _number(parameters[name]) is not None:
            safe[name] = parameters[name]
    for name in ("stream", "logprobs", "store"):
        if isinstance(parameters.get(name), bool):
            safe[name] = parameters[name]
    enums = {
        "reasoning_effort": {"none", "minimal", "low", "medium", "high", "xhigh", "max"},
        "verbosity": {"low", "medium", "high"},
        "service_tier": {"auto", "default", "flex", "priority", "scale"},
        "truncation": {"auto", "disabled"},
    }
    for name, allowed in enums.items():
        value = parameters.get(name)
        if isinstance(value, str) and value in allowed:
            safe[name] = value
    reasoning = parameters.get("reasoning")
    if isinstance(reasoning, dict):
        safe_reasoning = {}
        if isinstance(reasoning.get("effort"), str) and reasoning["effort"] in enums["reasoning_effort"]:
            safe_reasoning["effort"] = reasoning["effort"]
        if isinstance(reasoning.get("summary"), str) and reasoning["summary"] in {"auto", "concise", "detailed"}:
            safe_reasoning["summary"] = reasoning["summary"]
        if safe_reasoning:
            safe["reasoning"] = safe_reasoning
    return safe


def _code_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=2, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
                                capture_output=True, text=True, timeout=2, check=True).stdout
        return {"code_revision": revision, "code_dirty": bool(status)}
    except (OSError, subprocess.SubprocessError):
        return {"code_revision": None, "code_dirty": None}


def evaluator_fingerprint() -> str:
    """Hash implementation, effective policy resources, and diagnostic schemas.

    Read on every call so local prompt edits cannot reuse a stale cached identity.
    Imports stay lazy to avoid cycles during model/evaluator initialization.
    """
    from .models import Diagnostics
    from .policy import policy_fingerprint
    from deepeval.metrics.faithfulness.schema import Claims, FaithfulnessScoreReason, Truths, Verdicts
    from deepeval.metrics.g_eval.schema import ReasonScore
    from deepeval.templates.resolver import iter_base_template_methods

    digest = hashlib.sha256()
    package = Path(__file__).parent
    for name in ("policy.py", "metrics.py", "diagnostics.py", "judge.py", "models.py", "evaluator.py"):
        path = package / name
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    digest.update(policy_fingerprint().encode("ascii"))
    # DeepEval 4.3 builds template classes dynamically from bundled Jinja text.
    for template_class in ("FaithfulnessMetric", "GEval"):
        templates = iter_base_template_methods("metrics", template_class)
        digest.update(json.dumps(templates, separators=(",", ":")).encode("utf-8"))
    for schema in (Diagnostics, Claims, Truths, Verdicts, FaithfulnessScoreReason, ReasonScore):
        digest.update(json.dumps(schema.model_json_schema(), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def runtime_metadata(model: str | ChatOpenAI, timeout: float = 120, max_retries: int = 2) -> dict:
    """Return allowlisted settings, never a model dump or arbitrary headers/body.

    For a supplied model, its existing request settings take precedence over
    evaluator constructor defaults. Once initialized, pass ``judge.model`` to
    capture resolved SDK settings and endpoint identity.
    """
    versions = {"python": platform.python_version()}
    for name in ("meeting-summary-eval", "deepeval", "langchain-openai", "langchain-core", "openai", "pydantic"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "source"
    if isinstance(model, str):
        config = {
            "endpoint": _endpoint(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")),
            "timeout": _timeout(timeout), "max_retries": max_retries,
            "generation_parameters": {}, "custom_parameters_present": False,
            "use_responses_api": None,
        }
        model_name = model
    else:
        # The pinned LangChain adapter resolves model_kwargs and token aliases
        # here; filter that mapping instead of serializing the model or client.
        parameters = model._default_params
        client = model.root_async_client
        extra_body = parameters.get("extra_body")
        generation = _generation_parameters(parameters)
        extra_generation = _generation_parameters(extra_body) if isinstance(extra_body, dict) else {}
        unrecorded = bool(set(parameters) - set(generation) - {"model", "extra_body"})
        if isinstance(parameters.get("reasoning"), dict):
            unrecorded |= parameters["reasoning"] != generation.get("reasoning", {})
        if extra_body:
            unrecorded |= not isinstance(extra_body, dict) or extra_body != extra_generation
        config = {
            "endpoint": _endpoint(getattr(client, "base_url", None) or model.openai_api_base),
            "timeout": _timeout(getattr(client, "timeout", model.request_timeout)),
            "max_retries": getattr(client, "max_retries", model.max_retries),
            "generation_parameters": generation,
            "custom_parameters_present": unrecorded,
            "use_responses_api": model.use_responses_api,
        }
        if isinstance(extra_body, dict):
            config["extra_body_generation_parameters"] = extra_generation
        model_name = model.model_name
    config["structured_output"] = {"method": "json_schema", "strict": True}
    return {"model": model_name, "judge_config": config, "versions": versions,
            **_code_identity(), "evaluator_fingerprint": evaluator_fingerprint()}
