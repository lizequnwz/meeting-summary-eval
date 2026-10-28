"""Load only the requested configuration; never persist credentials."""

import os
from pathlib import Path

from dotenv import load_dotenv


def prepare_environment(env_file: str | Path | None = ".env") -> None:
    # Set these before importing DeepEval. Standalone metrics need no run store.
    os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
    os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"
    os.environ["DEEPEVAL_DISABLE_LEGACY_KEYFILE"] = "1"
    os.environ["DEEPEVAL_FILE_SYSTEM"] = "READ_ONLY"
    if env_file is not None:
        load_dotenv(Path(env_file), override=False)


prepare_environment(None)
