"""Native DeepEval scoring for meeting summaries."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load local settings before DeepEval initializes its native models.
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"
os.environ["DEEPEVAL_DISABLE_LEGACY_KEYFILE"] = "1"
load_dotenv(Path(".env"), override=False)

from .metrics import create_metric

__all__ = ["create_metric"]
