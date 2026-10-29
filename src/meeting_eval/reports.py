"""Local, inspectable reports for meeting evaluation runs."""

from __future__ import annotations

import csv
import html
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import BatchResult, Finding, MeetingResult, MetricResult


def _valid_score(metric: MetricResult | None) -> float | None:
    if metric is None or metric.status != "completed" or metric.score is None:
        return None
    score = float(metric.score)
    return score if math.isfinite(score) else None


def score_rows(results: list[MeetingResult]) -> list[dict[str, Any]]:
    """Flatten scores without treating missing or unsuccessful metrics as zero."""
    metric_names = sorted({name for result in results for name in result.metrics})
    rows = []
    for result in results:
        row: dict[str, Any] = {
            "meeting_id": result.meeting_id,
            "execution_status": result.execution_status,
            "review_required": result.review_required,
            "diagnostics_status": result.metadata.get("diagnostics_status", "unknown"),
            "review_reasons": "; ".join(result.review_reasons),
            "high_severity_findings": sum(f.severity == "high" for f in result.findings),
            "finding_count": len(result.findings),
            "errors": "; ".join(result.errors),
            "request_count": result.metadata.get("usage", {}).get("request_count"),
            "input_tokens": result.metadata.get("usage", {}).get("input_tokens"),
            "output_tokens": result.metadata.get("usage", {}).get("output_tokens"),
            "elapsed_seconds": result.metadata.get("elapsed_seconds"),
        }
        for name in metric_names:
            metric = result.metrics.get(name)
            row[f"{name}_score"] = _valid_score(metric)
            row[f"{name}_status"] = metric.status if metric is not None else "not_run"
        rows.append(row)
    return rows


def aggregate_results(results: list[MeetingResult]) -> dict[str, Any]:
    """Summarize valid metric scores and show issue-frequency denominators."""
    metric_names = sorted({name for result in results for name in result.metrics})
    metric_stats = {}
    for name in metric_names:
        scores = [score for result in results if (score := _valid_score(result.metrics.get(name))) is not None]
        statuses = Counter(
            result.metrics[name].status if name in result.metrics else "not_run"
            for result in results
        )
        metric_stats[name] = {
            "valid_count": len(scores),
            "meeting_denominator": len(results),
            "mean": statistics.mean(scores) if scores else None,
            "median": statistics.median(scores) if scores else None,
            "minimum": min(scores) if scores else None,
            "maximum": max(scores) if scores else None,
            "status_counts": dict(sorted(statuses.items())),
        }

    evaluated = [r for r in results if r.execution_status in {"completed", "partial"}]
    issue_counts: Counter[str] = Counter()
    issue_meetings: Counter[str] = Counter()
    for result in results:
        issue_counts.update(f.kind for f in result.findings)
        issue_meetings.update({f.kind for f in result.findings})
    diagnostics_completed = [r for r in results if r.metadata.get("diagnostics_status") == "completed"]
    structure_completed = [r for r in results if r.metrics.get("structure") is not None and r.metrics["structure"].status == "completed"]
    issues = {}
    for kind, count in sorted(issue_counts.items(), key=lambda item: (-issue_meetings[item[0]], item[0])):
        assessed = structure_completed if kind == "structure" else diagnostics_completed
        affected = sum(any(f.kind == kind for f in r.findings) for r in assessed)
        issues[kind] = {
            "finding_count": count,
            "meeting_count": issue_meetings[kind],
            "assessed_meeting_count": affected,
            "meeting_denominator": len(assessed),
            "meeting_fraction": affected / len(assessed) if assessed else None,
            "denominator_type": "completed_structure_metrics" if kind == "structure" else "completed_diagnostics",
        }
    usage = [r.metadata["usage"] for r in results if "usage" in r.metadata]
    requests = sum(item["request_count"] for item in usage)
    usage_complete = bool(requests) and all(
        not item["request_count"] or item.get("requests_with_token_usage") == item["request_count"]
        for item in usage
    )
    return {
        "meeting_count": len(results),
        "execution_status_counts": dict(sorted(Counter(r.execution_status for r in results).items())),
        "review_required_count": sum(r.review_required for r in results),
        "high_severity_finding_count": sum(f.severity == "high" for r in results for f in r.findings),
        "evaluated_meeting_count": len(evaluated),
        "diagnostics_completed_count": len(diagnostics_completed),
        "diagnostics_status_counts": dict(sorted(Counter(r.metadata.get("diagnostics_status", "unknown") for r in results).items())),
        "metrics": metric_stats,
        "issue_categories": issues,
        "usage": {
            "request_count": requests,
            "successful_requests": sum(item["successful_requests"] for item in usage),
            "failed_requests": sum(item["failed_requests"] for item in usage),
            "requests_with_token_usage": sum(item.get("requests_with_token_usage", 0) for item in usage),
            "input_tokens": sum(item.get("input_tokens") or 0 for item in usage) if usage_complete else None,
            "output_tokens": sum(item.get("output_tokens") or 0 for item in usage) if usage_complete else None,
            "total_tokens": sum(item.get("total_tokens") or 0 for item in usage) if usage_complete else None,
        },
        "issue_denominator_description": "Issue rates use only meetings with completed structured diagnostics. Structure issues use completed structure metrics. Failed, skipped, and unknown diagnostic states are excluded; finding counts include all returned findings.",
    }


def _csv_cell(value: Any) -> Any:
    # Spreadsheet applications may execute a text cell beginning with these chars.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    if isinstance(value, str) and value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


def write_csv(rows: list[dict[str, Any]], path: str | Path, *, fieldnames: list[str] | None = None) -> Path:
    """Export local analysis tables using the same formula protection as reports."""
    path = Path(path)
    names = fieldnames if fieldnames is not None else list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writerow({name: _csv_cell(name) for name in names})
        writer.writerows({name: _csv_cell(row.get(name)) for name in names} for row in rows)
    return path


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _label(value: str) -> str:
    return value.replace("_", " ").strip().capitalize()


def _score(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _finding_html(finding: Finding) -> str:
    evidence = []
    for item in finding.evidence:
        location = _label(item.source)
        if item.line_start is not None:
            location += f" · line {item.line_start}"
            if item.line_end is not None and item.line_end != item.line_start:
                location += f"–{item.line_end}"
        verification = "Quote verified" if item.verified else "Quote unverified — human review needed"
        evidence.append(
            f'<div class="evidence"><div class="evidence-label">{_e(location)}'
            f' <span class="verification">{_e(verification)}</span></div>'
            f'<blockquote>{_e(item.quote)}</blockquote></div>'
        )
    section = f'<span class="section-tag">{_e(finding.section)}</span>' if finding.section else ""
    suggestion = (
        f'<div class="suggestion"><strong>Prompt improvement</strong><p>{_e(finding.suggested_prompt_change)}</p></div>'
        if finding.suggested_prompt_change else ""
    )
    return (
        f'<article class="finding severity-{_e(finding.severity)}">'
        f'<div class="finding-heading"><span class="severity">{_e(finding.severity)} severity</span>'
        f'<strong>{_e(_label(finding.kind))}</strong>{section}</div>'
        f'<p class="preserve">{_e(finding.description)}</p>{"".join(evidence)}{suggestion}</article>'
    )


def _meeting_html(result: MeetingResult) -> str:
    rows = []
    for name, metric in result.metrics.items():
        score = _valid_score(metric)
        bar = f'<span class="score-bar" style="width:{max(0, min(1, score)) * 100:.1f}%"></span>' if score is not None else ""
        details = ""
        if metric.details:
            details = (
                '<details class="metric-details"><summary>Metric details</summary>'
                f'<pre>{_e(json.dumps(metric.details, indent=2, ensure_ascii=False, default=str))}</pre></details>'
            )
        rows.append(
            f'<tr><th scope="row">{_e(_label(name))}</th><td class="score-number">{_score(score)}'
            f'<span class="score-track">{bar}</span></td><td>{_e(_label(metric.status))}</td>'
            f'<td><div class="preserve">{_e(metric.reason or "No reasoning available.")}</div>{details}</td></tr>'
        )
    errors = (
        '<div class="error-box"><strong>Execution problems</strong><ul>'
        + "".join(f'<li>{_e(error)}</li>' for error in result.errors)
        + '</ul></div>'
        if result.errors else ""
    )
    reasons = (
        '<div class="review-reasons"><strong>Why review is needed</strong><ul>'
        + "".join(f'<li>{_e(reason)}</li>' for reason in result.review_reasons)
        + '</ul></div>'
        if result.review_reasons else ""
    )
    findings = "".join(_finding_html(f) for f in sorted(result.findings, key=lambda f: {"high": 0, "medium": 1, "low": 2}[f.severity]))
    if not findings:
        findings = '<p class="muted">No structured findings were returned. Check execution status and review reasons before interpreting this as a clean result.</p>'
    metadata = (
        '<details class="metadata"><summary>Run and input metadata</summary>'
        f'<pre>{_e(json.dumps(result.metadata, indent=2, ensure_ascii=False, default=str))}</pre></details>'
        if result.metadata else ""
    )
    badge = 'Review required' if result.review_required else 'No review flags'
    return (
        f'<details class="meeting" data-meeting="{_e(result.meeting_id.lower())}" data-review="{str(result.review_required).lower()}">'
        f'<summary><span class="meeting-id">{_e(result.meeting_id)}</span>'
        f'<span class="meeting-status">{_e(_label(result.execution_status))}</span>'
        f'<span class="badge {"review" if result.review_required else "neutral"}">{badge}</span></summary>'
        f'<div class="meeting-body">{errors}{reasons}<div class="table-wrap"><table class="metric-table">'
        '<thead><tr><th>Metric</th><th>Score / 1</th><th>Status</th><th>Reasoning</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div><h3>Findings and evidence</h3>{findings}{metadata}</div></details>'
    )


_CSS = """
:root{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172d35;background:#f3f5f3;line-height:1.55;font-size:15px}
*{box-sizing:border-box}body{margin:0}main{max-width:1220px;margin:auto;padding:44px 28px 70px}h1,h2,h3,p{margin-top:0}h1{font-size:clamp(1.8rem,4vw,2.65rem);line-height:1.12;letter-spacing:-.045em;max-width:850px;margin-bottom:16px}h2{font-size:1.3rem;letter-spacing:-.02em;margin-bottom:16px}h3{font-size:1rem;margin:26px 0 14px}.eyebrow{text-transform:uppercase;letter-spacing:.16em;font-size:.72rem;font-weight:750;color:#27705b;margin-bottom:14px}.subtitle{max-width:780px;color:#51656b}.run-id{font-size:.78rem;color:#61737a;overflow-wrap:anywhere}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:32px 0}.kpi{background:#fff;border:1px solid #dae2dd;border-radius:12px;padding:21px}.kpi strong{display:block;font-size:2rem;line-height:1.2;letter-spacing:-.04em}.kpi span{font-size:.79rem;color:#61737a}.panel{background:#fff;border:1px solid #dae2dd;border-radius:14px;padding:25px;margin-bottom:24px}.muted,.note{color:#61737a;font-size:.86rem}.note{margin:16px 0 0}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:.87rem;text-align:left}th,td{padding:13px 12px;border-bottom:1px solid #e8eeea;vertical-align:top}th{font-weight:650}thead th{text-transform:uppercase;letter-spacing:.06em;font-size:.67rem;color:#61737a;white-space:nowrap}tbody tr:last-child>*{border-bottom:0}.number{font-variant-numeric:tabular-nums}.metric-table th:first-child{width:16%}.metric-table td:last-child{min-width:250px}.metric-table td:nth-child(3){width:15%}.score-number{font-weight:700;font-variant-numeric:tabular-nums;min-width:90px}.score-track{background:#e9eeeb;border-radius:8px;height:5px;display:block;margin-top:7px;width:65px;overflow:hidden}.score-bar{display:block;height:100%;background:#27705b}.toolbar{display:flex;gap:18px;align-items:center;justify-content:space-between;margin-bottom:18px;flex-wrap:wrap}.toolbar input[type=search]{min-width:230px;padding:11px 13px;border:1px solid #c8d5cd;border-radius:8px;font:inherit;background:#fff}.toolbar label{font-size:.86rem;display:flex;align-items:center;gap:8px}.meeting{background:#fff;border:1px solid #dae2dd;border-radius:12px;margin-bottom:12px;overflow:hidden}.meeting>summary{display:flex;align-items:center;gap:14px;padding:20px 22px;cursor:pointer;list-style:none}.meeting>summary:before{content:'＋';color:#27705b}.meeting[open]>summary:before{content:'−'}.meeting-id{font-weight:700;overflow-wrap:anywhere}.meeting-status{font-size:.78rem;color:#61737a;margin-left:auto}.badge{font-size:.7rem;font-weight:650;border-radius:20px;padding:5px 11px;white-space:nowrap}.review{background:#fff0d6;color:#86531c}.neutral{background:#edf2ee;color:#4b6557}.meeting-body{border-top:1px solid #e8eeea;padding:22px}.review-reasons,.error-box{padding:16px 18px;border-radius:8px;font-size:.85rem;margin-bottom:20px;background:#fff7e9;border:1px solid #f0d7a9}.error-box{background:#fff1ef;border-color:#efc4bd;color:#8b3931}ul{padding-left:21px;margin:8px 0 0}.finding{border:1px solid #dce5de;border-left:4px solid #9fb5a6;border-radius:8px;padding:17px 19px;margin-bottom:14px;font-size:.87rem}.severity-high{border-left-color:#bc5c46}.severity-medium{border-left-color:#c19342}.finding-heading{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-bottom:10px}.severity{font-size:.65rem;text-transform:uppercase;letter-spacing:.06em;color:#687b70}.section-tag{font-size:.69rem;background:#edf2ee;border-radius:4px;padding:2px 7px;color:#4b6557}.evidence{margin-top:13px;background:#f5f7f5;border-radius:6px;padding:11px 13px}.evidence-label{font-size:.69rem;font-weight:650;color:#536b5e}.verification{font-weight:400;margin-left:8px}blockquote{white-space:pre-wrap;margin:8px 0 0;padding-left:12px;border-left:2px solid #b9cbbf;font-size:.82rem;overflow-wrap:anywhere}.suggestion{font-size:.8rem;padding:12px 0 0;color:#315d4a}.suggestion p{margin:4px 0 0;white-space:pre-wrap}.preserve{white-space:pre-wrap;overflow-wrap:anywhere}.metric-details,.metadata{margin-top:12px;font-size:.78rem;color:#61737a}.metric-details summary,.metadata summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.72rem;background:#f5f7f5;border-radius:6px;padding:14px;max-height:420px;overflow:auto}.legend{display:grid;grid-template-columns:1fr 1fr;gap:16px;font-size:.79rem;color:#61737a}.legend p{margin:0}.footer{font-size:.76rem;color:#61737a;margin-top:30px;border-top:1px solid #d7e1d9;padding-top:18px}button{padding:9px 12px;border:1px solid #c8d5cd;border-radius:7px;background:#fff;color:#315d4a;cursor:pointer;font:inherit;font-size:.8rem}button:hover{background:#edf2ee}[hidden]{display:none!important}input:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid #70a28b;outline-offset:3px}
@media(max-width:740px){main{padding:28px 15px}.kpis{grid-template-columns:1fr 1fr;gap:9px}.kpi{padding:16px}.kpi strong{font-size:1.6rem}.panel{padding:18px}.meeting>summary{padding:16px 13px;gap:9px;flex-wrap:wrap}.meeting-body{padding:15px}.meeting-status{margin-left:0}.legend{grid-template-columns:1fr}.badge{margin-left:auto}}
@media print{body{background:white}main{padding:0;max-width:none}.toolbar{display:none}.panel,.meeting,.kpi{break-inside:avoid;box-shadow:none}.meeting-body{display:block}.meeting>summary{display:flex}details:not([open])>*:not(summary){display:block}.metric-details pre,.metadata pre{max-height:none}.footer{break-inside:avoid}}
"""


def _report_html(batch: BatchResult, aggregate: dict[str, Any]) -> str:
    metric_names = list(aggregate["metrics"])
    comparison_headers = "".join(f'<th>{_e(_label(name))}</th>' for name in metric_names)
    comparison_rows = "".join(
        f'<tr><th scope="row">{_e(result.meeting_id)}</th><td>{_e(_label(result.execution_status))}</td>'
        + "".join(f'<td class="number">{_score(_valid_score(result.metrics.get(name)))}</td>' for name in metric_names)
        + f'<td>{"Required" if result.review_required else "No flags"}</td></tr>'
        for result in batch.results
    )
    comparison = (
        '<details class="metadata"><summary>Compare individual meeting scores</summary><div class="table-wrap">'
        f'<table><thead><tr><th>Meeting</th><th>Execution</th>{comparison_headers}<th>Review</th></tr></thead>'
        f'<tbody>{comparison_rows}</tbody></table></div></details>'
        if batch.results else ""
    )
    metric_rows = "".join(
        f'<tr><th scope="row">{_e(_label(name))}</th><td class="number">{stats["valid_count"]} / {stats["meeting_denominator"]}</td>'
        f'<td class="number">{_score(stats["mean"])}</td><td class="number">{_score(stats["median"])}</td>'
        f'<td class="number">{_score(stats["minimum"])} – {_score(stats["maximum"])}</td></tr>'
        for name, stats in aggregate["metrics"].items()
    ) or '<tr><td colspan="5" class="muted">No metrics were run.</td></tr>'
    issue_rows = "".join(
        f'<tr><th scope="row">{_e(_label(kind))}</th><td class="number">{str(stats["assessed_meeting_count"]) + " / " + str(stats["meeting_denominator"]) if stats["meeting_denominator"] else "Not assessed"}</td>'
        f'<td class="number">{stats["finding_count"]}</td></tr>'
        for kind, stats in aggregate["issue_categories"].items()
    ) or '<tr><td colspan="3" class="muted">No structured findings were returned.</td></tr>'
    statuses = " · ".join(f'{count} {_label(status).lower()}' for status, count in aggregate["execution_status_counts"].items()) or "No inputs"
    problem_count = sum(aggregate["execution_status_counts"].get(status, 0) for status in ("partial", "invalid", "error"))
    usage = aggregate["usage"]
    tokens = str(usage["total_tokens"]) if usage["total_tokens"] is not None else "Unavailable or incomplete"
    provenance = _e(json.dumps(batch.metadata, indent=2, ensure_ascii=False, default=str))
    meetings = "".join(_meeting_html(result) for result in batch.results)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Meeting summary evaluation report</title><style>{_CSS}</style></head>
<body><main>
<header><p class="eyebrow">Meeting Summary Evaluation</p><h1>Meeting summary quality,<br>with the evidence in view.</h1>
<p class="subtitle">Evaluations grounded in the meeting transcript and explicitly supplied context. Compare dimensions separately, then inspect the findings that matter for advisor and client conversations.</p>
<p class="run-id">Run {_e(batch.run_id)} · {_e(statuses)}</p>
<details class="metadata"><summary>Evaluation policy and judge configuration</summary><pre>{provenance}</pre></details></header>
<section class="kpis" aria-label="Run overview">
<div class="kpi"><strong>{aggregate["meeting_count"]}</strong><span>Input meetings</span></div>
<div class="kpi"><strong>{aggregate["review_required_count"]}</strong><span>Require review</span></div>
<div class="kpi"><strong>{aggregate["high_severity_finding_count"]}</strong><span>High severity findings</span></div>
<div class="kpi"><strong>{problem_count}</strong><span>Invalid inputs or evaluation errors</span></div></section>
<p class="note">Judge invocations: {usage["request_count"]} · Failed invocations: {usage["failed_requests"]} · Provider-reported tokens: {_e(tokens)}. Invocation counts exclude HTTP retries inside the SDK; incomplete usage remains unknown.</p>
<section class="panel"><h2>Scores across meetings</h2><div class="table-wrap"><table><thead><tr><th>Dimension</th><th>Valid / input meetings</th><th>Mean</th><th>Median</th><th>Range</th></tr></thead><tbody>{metric_rows}</tbody></table></div>
<p class="note">Scores range from 0 to 1; higher is better. Means and medians include only completed metrics with a valid score. Missing, failed, and inapplicable evaluations remain missing. These are evaluation scores, not probabilities of correctness. No weighted overall score is calculated.</p>{comparison}</section>
<section class="panel"><h2>Recurring issues</h2><div class="table-wrap"><table><thead><tr><th>Issue category</th><th>Meetings affected / assessed</th><th>Findings</th></tr></thead><tbody>{issue_rows}</tbody></table></div>
<p class="note"><strong>Structured diagnostic coverage: {aggregate["diagnostics_completed_count"]} / {aggregate["meeting_count"]} input meetings.</strong> {_e(aggregate["issue_denominator_description"])} A meeting is counted once per category, even when it has several findings of that kind.</p></section>
<section aria-labelledby="meeting-heading"><h2 id="meeting-heading">Inspect each meeting</h2><div class="toolbar"><input id="meeting-search" type="search" placeholder="Find a meeting ID" aria-label="Find a meeting ID"><label><input id="review-only" type="checkbox">Only meetings requiring review</label><button id="expand-all" type="button">Expand visible meetings</button></div>
<p id="filter-empty" class="muted" hidden>No meetings match this filter.</p>{meetings or '<p class="muted">No meeting results are available.</p>'}</section>
<section class="panel" style="margin-top:28px"><h2>Reading this report</h2><div class="legend"><p><strong>Execution status.</strong> Completed means the requested evaluation ran; it does not mean the summary passed. Partial means some requested work failed. Invalid means the input could not be evaluated. Error means evaluation failed.</p>
<p><strong>Evidence and judgment.</strong> Quote verification checks that cited text occurs in the transcript, supplied context, or summary; it does not prove the judge interpreted it correctly. Labeled suggested outreach is assessed separately from claims that a meeting recommendation was made. Unsupported claims and ambiguous passages require human judgment. A review flag does not constitute a confirmed financial or policy finding.</p></div></section>
<p class="footer">The transcript establishes what was said and can contain transcription errors. Supplied context remains a separate source. Review consequential claims and institutional view attribution against the cited passages. Detailed results and flat scores are saved beside this report as results.json and scores.csv.</p>
</main><script>
const search=document.getElementById('meeting-search');
const review=document.getElementById('review-only');
const cards=Array.from(document.querySelectorAll('.meeting'));
function filterMeetings(){{const query=search.value.toLowerCase().trim();let visible=0;for(const card of cards){{card.hidden=!card.dataset.meeting.includes(query)||(review.checked&&card.dataset.review!=='true');if(!card.hidden)visible++;}}document.getElementById('filter-empty').hidden=visible>0||cards.length===0;}}
search.addEventListener('input',filterMeetings);review.addEventListener('change',filterMeetings);
document.getElementById('expand-all').addEventListener('click',()=>{{for(const card of cards)if(!card.hidden)card.open=true;}});
</script></body></html>'''


def write_reports(batch: BatchResult, output_dir: str | Path) -> Path:
    """Write JSON, CSV, and self-contained HTML to a new run directory."""
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    dirname = re.sub(r"[^A-Za-z0-9._-]", "_", batch.run_id).strip(".")[:120] or "run"
    run_dir = root / dirname
    while True:
        try:
            run_dir.mkdir(exist_ok=False)
            break
        except FileExistsError:
            run_dir = root / f"{dirname}-{uuid4().hex[:8]}"
    batch.output_dir = str(run_dir)
    aggregate = aggregate_results(batch.results)
    payload = batch.model_dump(mode="json")
    payload["aggregate"] = aggregate
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    rows = score_rows(batch.results)
    fieldnames = list(rows[0]) if rows else ["meeting_id", "execution_status", "review_required", "diagnostics_status", "review_reasons", "high_severity_findings", "finding_count", "errors"]
    write_csv(rows, run_dir / "scores.csv", fieldnames=fieldnames)
    (run_dir / "report.html").write_text(_report_html(batch, aggregate), encoding="utf-8")
    return run_dir
