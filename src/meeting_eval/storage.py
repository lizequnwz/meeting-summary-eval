"""Small filesystem and provenance helpers shared by the evaluation workflow."""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit, urlunsplit

from .inputs import normalize_text


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def safe_endpoint(url) -> str | None:
    """Keep endpoint provenance without credentials, query parameters, or fragments."""
    if not url:
        return None
    try:
        parsed = urlsplit(str(url))
        if not parsed.scheme or not parsed.hostname:
            return None
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        if parsed.port is not None:
            host += f":{parsed.port}"
    except ValueError:
        return None
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def model_endpoint(model) -> str | None:
    endpoint = getattr(model, "base_url", None)
    if endpoint is None and type(model).__name__ == "OpenAIModel":
        endpoint = "https://api.openai.com/v1"
    return safe_endpoint(endpoint)


def cost_value(cost) -> float | None:
    """DeepEval can represent unknown pricing as float(0) with value=None."""
    value = getattr(cost, "value", cost)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def read_json(path: str | Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc


def atomic_write_json(path: str | Path, payload) -> Path:
    path = Path(path)
    content = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}-", delete=False) as file:
            temporary = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def create_run_directory(output_dir: str | Path = "results") -> Path:
    """Create a unique writable destination before spending on model calls."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    prefix = "run-" + utc_now().replace(":", "-") + "-"
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))
