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
from .questions import get_or_prepare_questions
from .evaluation import evaluate_meeting, load_evaluation, new_run, save_evaluation
from .reports import export_reports
from .storage import create_run_directory

__all__ = ["create_metric", "get_or_prepare_questions", "evaluate_meeting", "new_run",
           "save_evaluation", "load_evaluation", "export_reports", "create_run_directory"]
