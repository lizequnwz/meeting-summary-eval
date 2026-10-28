"""Evaluate meeting summaries against their complete source transcripts."""

from . import config as _config
from .evaluator import JudgeConfigurationError, MeetingSummaryEvaluator
from .models import BatchResult, Evidence, Finding, MeetingResult, MetricResult

__all__ = ["MeetingSummaryEvaluator", "JudgeConfigurationError", "BatchResult", "MeetingResult", "MetricResult", "Finding", "Evidence"]
