"""Small, serializable result types shared by the evaluator and reports."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    source: Literal["transcript", "summary"]
    quote: str
    line_start: int | None = None
    line_end: int | None = None
    verified: bool = False


class Finding(BaseModel):
    kind: Literal[
        "contradiction", "unsupported_claim", "missing_topic", "uncertainty",
        "structure", "clarity", "fidelity_attribution", "product_claim",
        "material_distortion", "certainty_inflation",
    ]
    severity: Literal["high", "medium", "low"]
    description: str
    section: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    suggested_prompt_change: str | None = None


class Diagnostics(BaseModel):
    qa_present: bool | None
    findings: list[Finding]


class MetricResult(BaseModel):
    name: str
    score: float | None = Field(default=None, ge=0, le=1)
    reason: str | None = None
    status: Literal["completed", "error", "not_applicable"] = "completed"
    details: dict[str, Any] = Field(default_factory=dict)


class MeetingResult(BaseModel):
    meeting_id: str
    execution_status: Literal["completed", "partial", "invalid", "error"] = "completed"
    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    review_required: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchResult(BaseModel):
    run_id: str
    results: list[MeetingResult]
    output_dir: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dataframe(self):
        """Return one comparison row per meeting (requires the notebooks extra)."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Install meeting-summary-eval[notebooks] to use Pandas.") from exc
        from .reports import score_rows

        return pd.DataFrame(score_rows(self.results))
