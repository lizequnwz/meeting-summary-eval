"""Single-pair and bounded batch evaluation using public DeepEval metrics."""

import asyncio
import hashlib
import importlib.metadata
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_openai import ChatOpenAI
from openai import AuthenticationError, BadRequestError, NotFoundError, PermissionDeniedError

from .config import prepare_environment
from .diagnostics import BUSINESS_INSTRUCTIONS, check_structure, diagnostic_prompt, evidence_complete, verify_evidence
from .inputs import discover_pairs, normalize_text
from .judge import LangChainJudge
from .metrics import create_metrics
from .models import BatchResult, Diagnostics, MeetingResult, MetricResult


class JudgeConfigurationError(RuntimeError):
    """The configured judge cannot be used; changing meeting inputs will not help."""


def _global_error(exc: Exception) -> bool:
    if isinstance(exc, (JudgeConfigurationError, AuthenticationError, PermissionDeniedError, NotFoundError)):
        return True
    code = getattr(exc, "code", None)
    return code in {"model_not_found", "invalid_api_key", "insufficient_quota", "unsupported_parameter", "unsupported_value"}


def _error_message(exc: Exception) -> str:
    # SDK exception bodies can contain submitted data. Keep only a safe type/code.
    code = getattr(exc, "code", None)
    if _global_error(exc):
        return f"{type(exc).__name__}{f' ({code})' if code else ''}: check judge credentials, model access and endpoint support."
    if isinstance(exc, BadRequestError):
        return f"BadRequestError{f' ({code})' if code else ''}: check request context length and structured-output support."
    return f"{type(exc).__name__}: judge evaluation failed; check connectivity, timeout and structured response validity."


def _plain(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


class MeetingSummaryEvaluator:
    def __init__(self, model: str | ChatOpenAI | None = None, max_concurrency: int = 4,
                 metrics: tuple[str, ...] | list[str] = ("faithfulness", "coverage"),
                 env_file: str | Path | None = ".env", timeout: float = 120, max_retries: int = 2):
        prepare_environment(env_file)
        if not isinstance(max_concurrency, int) or isinstance(max_concurrency, bool) or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        if timeout <= 0 or max_retries < 0:
            raise ValueError("timeout must be positive and max_retries nonnegative")
        if not metrics or set(metrics) - {"faithfulness", "coverage", "structure", "clarity"}:
            raise ValueError("metrics must select faithfulness, coverage, structure or clarity")
        if len(metrics) != len(set(metrics)):
            raise ValueError("metrics must not contain duplicates")
        self.model = model if model is not None else os.getenv("OPENAI_MODEL", "gpt-6-luna")
        self.max_concurrency = max_concurrency
        self.metric_names = tuple(metrics)
        self.timeout = timeout
        self.max_retries = max_retries
        self._judge = None
        self._runner = None
        self._closed = False

    @property
    def judge(self) -> LangChainJudge:
        if self._judge is None:
            try:
                self._judge = LangChainJudge(self.model, max_concurrency=self.max_concurrency,
                                             timeout=self.timeout, max_retries=self.max_retries)
            except Exception as exc:
                raise JudgeConfigurationError("Could not initialize the configured judge.") from exc
        return self._judge

    def _metadata(self) -> dict:
        versions = {}
        for name in ("deepeval", "langchain-openai", "meeting-summary-eval"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "source"
        return {"model": self.model if isinstance(self.model, str) else self.model.model_name,
                "metrics": list(self.metric_names), "versions": versions,
                "methodology_version": "1", "sales_policy": "explicit_discussion_only",
                "scoring": "separate_scores_no_acceptance_gate", "max_concurrency": self.max_concurrency,
                "timestamp": datetime.now(timezone.utc).isoformat()}

    def _sync(self, coroutine_factory):
        if self._closed:
            raise RuntimeError("This evaluator is closed. Create a new evaluator.")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if self._runner is None:
                self._runner = asyncio.Runner()
            return self._runner.run(coroutine_factory())
        raise RuntimeError("An event loop is already running. Use await evaluator.aevaluate(...) or aevaluate_batch(...).")

    def close(self):
        """Release resources used by synchronous evaluation calls."""
        if self._closed:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if self._runner is None:
                self._runner = asyncio.Runner()
            try:
                if self._judge is not None:
                    self._runner.run(self._judge.aclose())
            finally:
                self._runner.close()
                self._closed = True
            return
        raise RuntimeError("Use await evaluator.aclose() in asynchronous code.")

    async def aclose(self):
        """Release owned model resources after asynchronous notebook use."""
        if self._closed:
            return
        if self._runner is not None:
            raise RuntimeError("Close synchronous evaluators with close() outside an event loop.")
        if self._judge is not None:
            await self._judge.aclose()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("This evaluator is closed. Create a new evaluator.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def evaluate(self, transcript: str, summary: str, meeting_id: str = "meeting") -> MeetingResult:
        return self._sync(lambda: self.aevaluate(transcript, summary, meeting_id))

    async def aevaluate(self, transcript: str, summary: str, meeting_id: str = "meeting") -> MeetingResult:
        if self._closed:
            raise RuntimeError("This evaluator is closed. Create a new evaluator.")
        from deepeval.test_case import LLMTestCase

        result = MeetingResult(meeting_id=meeting_id, metadata=self._metadata())
        result.metadata["diagnostics_status"] = "not_run"
        if not isinstance(transcript, str) or not isinstance(summary, str):
            return self._invalid(result, "Transcript and summary must be strings.")
        transcript, summary = normalize_text(transcript), normalize_text(summary)
        if not transcript.strip() or not summary.strip():
            return self._invalid(result, "Transcript and summary must both contain text.")
        result.metadata.update({"transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
                                "summary_sha256": hashlib.sha256(summary.encode()).hexdigest(),
                                "transcript_lines": len(transcript.splitlines()),
                                "summary_lines": len(summary.splitlines())})
        test_case = LLMTestCase(input=BUSINESS_INSTRUCTIONS, actual_output=summary,
                               retrieval_context=[transcript])
        try:
            judge = self.judge
            metric_objects = create_metrics(judge, names=tuple(n for n in self.metric_names if n != "structure"))
        except Exception as exc:
            result.errors.append(_error_message(exc))
            result.execution_status = "error"
            result.review_required = True
            result.review_reasons.append("Judge initialization failed.")
            result.metadata["judge_configuration_error"] = _global_error(exc)
            return result

        for name, metric in metric_objects.items():
            try:
                await metric.a_measure(test_case)
                details = {}
                if name == "faithfulness":
                    for attr in ("claims", "verdicts"):
                        details[attr] = _plain(getattr(metric, attr, []))
                    if not details["claims"]:
                        result.metrics[name] = MetricResult(name=name, status="not_applicable",
                            reason="No assessable claims were extracted; faithfulness is unscorable.", details=details)
                        result.review_reasons.append("No assessable summary claims.")
                        continue
                    if any(v.get("verdict") in {"no", "borderline"} for v in details["verdicts"]):
                        result.review_reasons.append("Faithfulness found contradicted or ambiguously supported claims.")
                if name in {"coverage", "clarity"}:
                    details["scoring_method"] = "normalized_rubric_without_logprob_weighting"
                if metric.score is None:
                    raise ValueError("The metric returned no score.")
                result.metrics[name] = MetricResult(name=name, score=metric.score, reason=metric.reason, details=details)
            except Exception as exc:
                message = _error_message(exc)
                result.metrics[name] = MetricResult(name=name, status="error", reason=message)
                result.errors.append(f"{name}: {message}")
                if _global_error(exc):
                    result.metadata["judge_configuration_error"] = True
                    break

        qa_present = None
        if not result.metadata.get("judge_configuration_error"):
            try:
                diagnostics = await judge.structured(diagnostic_prompt(transcript, summary), Diagnostics)
                result.findings = verify_evidence(diagnostics.findings, transcript, summary)
                qa_present = diagnostics.qa_present
                result.metadata["diagnostics_status"] = "completed"
                result.metadata["qa_present"] = qa_present
            except Exception as exc:
                result.metadata["diagnostics_status"] = "error"
                result.metadata["judge_configuration_error"] = _global_error(exc)
                result.errors.append(f"diagnostics: {_error_message(exc)}")
        if "structure" in self.metric_names:
            result.metrics["structure"], structure_findings = check_structure(summary, qa_present)
            result.findings.extend(structure_findings)
        for name in self.metric_names:
            if name not in result.metrics:
                result.metrics[name] = MetricResult(name=name, status="error", reason="Not run after judge configuration failure.")

        if any(f.severity == "high" for f in result.findings):
            result.review_reasons.append("Material judge finding requires human review.")
        if any(not evidence_complete(f) for f in result.findings):
            result.review_reasons.append("One or more findings have missing, ambiguous or unverified source quotations.")
        if any(f.kind == "uncertainty" for f in result.findings):
            result.review_reasons.append("Source interpretation or attribution is uncertain.")
        if result.metadata["diagnostics_status"] != "completed":
            result.review_reasons.append("Structured review did not complete.")
        failed = any(m.status == "error" for m in result.metrics.values()) or bool(result.errors)
        if failed:
            completed = any(m.status == "completed" for m in result.metrics.values()) or result.metadata["diagnostics_status"] == "completed"
            result.execution_status = "partial" if completed else "error"
            result.review_reasons.append("Evaluation was incomplete.")
        result.review_required = bool(result.review_reasons)
        return result

    @staticmethod
    def _invalid(result: MeetingResult, message: str) -> MeetingResult:
        result.execution_status = "invalid"
        result.errors.append(message)
        result.review_required = True
        result.review_reasons.append("Invalid or unmatched input.")
        return result

    def evaluate_batch(self, transcript_dir: str | Path, summary_dir: str | Path,
                       output_dir: str | Path | None = None) -> BatchResult:
        return self._sync(lambda: self.aevaluate_batch(transcript_dir, summary_dir, output_dir))

    async def aevaluate_batch(self, transcript_dir: str | Path, summary_dir: str | Path,
                              output_dir: str | Path | None = None) -> BatchResult:
        if self._closed:
            raise RuntimeError("This evaluator is closed. Create a new evaluator.")
        pairs = discover_pairs(transcript_dir, summary_dir)
        if not pairs:
            raise ValueError("No .txt or .md input files were found.")
        slots = asyncio.Semaphore(self.max_concurrency)
        configuration_failed = asyncio.Event()

        async def run(pair):
            result = MeetingResult(meeting_id=pair.meeting_id, metadata={"diagnostics_status": "not_run"})
            if pair.error:
                return self._invalid(result, pair.error)
            async with slots:
                if configuration_failed.is_set():
                    result.execution_status = "error"
                    result.errors.append("Not run after batch-wide judge configuration failure.")
                    result.review_required = True
                    result.review_reasons.append("Evaluation did not run.")
                    return result
                try:
                    transcript = pair.transcript_path.read_text(encoding="utf-8-sig")
                    summary = pair.summary_path.read_text(encoding="utf-8-sig")
                except (OSError, UnicodeError) as exc:
                    return self._invalid(result, f"Could not read UTF-8 input: {type(exc).__name__}.")
                try:
                    result = await self.aevaluate(transcript, summary, pair.meeting_id)
                except Exception as exc:
                    # Cancellation is a BaseException and deliberately propagates.
                    result.execution_status = "error"
                    result.errors.append(_error_message(exc))
                    result.review_required = True
                    result.review_reasons.append("Unexpected meeting evaluation failure.")
                result.metadata.update({"transcript_path": str(pair.transcript_path.resolve()),
                                        "summary_path": str(pair.summary_path.resolve())})
                if result.metadata.get("judge_configuration_error"):
                    configuration_failed.set()
                return result

        results = await asyncio.gather(*(run(pair) for pair in pairs))
        batch = BatchResult(run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8],
                            results=results, metadata=self._metadata())
        if output_dir is not None:
            from .reports import write_reports
            batch.output_dir = str(write_reports(batch, output_dir))
        return batch
