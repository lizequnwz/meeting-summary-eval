"""Save native DeepEval results as local JSON, CSV, and standalone HTML."""

import csv
import html
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path


_COLUMNS = ("meeting_id", "status", "alignment", "coverage", "score", "reason", "error")


def _csv_cell(value):
    if isinstance(value, str) and (
        value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n"))
    ):
        return "'" + value
    return value


def _escape(value):
    return html.escape(str(value), quote=True)


def _score(value):
    return "—" if value is None else f"{value:.2f}"


def _html(results, settings, created_at):
    rows, meetings = [], []
    for index, result in enumerate(results):
        breakdown = result["score_breakdown"] or {}
        scores = [breakdown.get("Alignment"), breakdown.get("Coverage"), result["score"]]
        rows.append(
            f'<tr><th scope="row"><a href="#meeting-{index}">{_escape(result["meeting_id"])}</a></th>'
            f'<td>{_escape(result["status"])}</td>'
            + "".join(f'<td class="number">{_score(score)}</td>' for score in scores)
            + "</tr>"
        )
        sections = []
        if result["error"]:
            sections.append(f'<p class="error">{_escape(result["error"])}</p>')
        if result["reason"]:
            sections.append(f'<p class="preserve">{_escape(result["reason"])}</p>')
        for label, key in (
            ("Assessment questions", "assessment_questions"),
            ("Extracted summary claims", "claims"),
            ("Alignment verdicts", "alignment_verdicts"),
            ("Coverage verdicts", "coverage_verdicts"),
            ("Extracted transcript statements", "truths"),
        ):
            if result[key]:
                detail = json.dumps(result[key], indent=2, ensure_ascii=False)
                sections.append(f"<h3>{label}</h3><pre>{_escape(detail)}</pre>")
        meetings.append(
            f'<details class="meeting" id="meeting-{index}"><summary>'
            f'{_escape(result["meeting_id"])} <span>{_escape(result["status"])} · '
            f'{_score(result["score"])}</span></summary><div class="body">'
            + "".join(sections)
            + "</div></details>"
        )
    completed = sum(result["status"] == "completed" for result in results)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meeting summary evaluation</title><style>
:root{{font-family:system-ui,sans-serif;color:#20352f;background:#f4f6f4;line-height:1.55}}
*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1080px;margin:auto;padding:40px 22px 64px}}
h1{{font-size:2rem;letter-spacing:-.04em;line-height:1.15;margin-bottom:12px}}h3{{font-size:.9rem;margin:24px 0 8px}}
p{{margin:12px 0}}.muted{{color:#5c7067;font-size:.9rem}}.panel,.meeting{{background:white;border:1px solid #d6e1da;border-radius:10px;margin:20px 0}}
.panel{{padding:20px;overflow-x:auto}}table{{width:100%;border-collapse:collapse;text-align:left;font-size:.9rem}}
th,td{{padding:12px;border-bottom:1px solid #e5ebe7}}thead th{{color:#5c7067;font-size:.8rem}}tbody tr:last-child>*{{border-bottom:0}}
.number{{font-variant-numeric:tabular-nums}}a{{color:#216b53}}summary{{padding:18px;cursor:pointer;font-weight:650;overflow-wrap:anywhere}}
summary span{{font-weight:400;color:#5c7067;margin-left:12px}}.body{{padding:0 20px 20px;border-top:1px solid #e5ebe7}}
pre{{font-size:.8rem;background:#f4f7f4;padding:14px;border-radius:6px;white-space:pre-wrap;overflow-wrap:anywhere}}
.preserve{{white-space:pre-wrap;overflow-wrap:anywhere}}.error{{color:#963e32;background:#fff1ed;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere}}
:focus-visible{{outline:3px solid #76aa94;outline-offset:3px}}@media(max-width:600px){{main{{padding:24px 12px}}.panel{{padding:8px}}th,td{{padding:10px 7px}}summary span{{display:block;margin:5px 0 0}}}}
@media print{{:root{{background:white}}main{{padding:0}}.meeting,.panel{{break-inside:avoid}}details:not([open])>*:not(summary){{display:block}}}}
</style></head><body><main>
<p class="muted">DeepEval · local evaluation results</p><h1>Meeting summary evaluation</h1>
<p>{completed} completed · {len(results) - completed} errors · {len(results)} meetings</p>
<p class="muted">All scores range from 0 to 1. Overall = min(alignment, coverage).
Coverage measures whether sampled transcript questions remain answerable from the summary.</p>
<div class="panel"><table><thead><tr><th>Meeting</th><th>Execution</th><th>Alignment</th><th>Coverage</th><th>Overall</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
{''.join(meetings)}
<details><summary>Run settings</summary><pre>{_escape(json.dumps(settings, indent=2, ensure_ascii=False))}</pre></details>
<p class="muted">Created {_escape(created_at)}. Judge verdicts require human interpretation; scores are not an acceptance gate.</p>
</main></body></html>"""


def write_reports(results: list[dict], output_dir: str | Path, settings: dict) -> Path:
    """Create a new run directory without overwriting previous evaluations."""
    created_at = datetime.now(timezone.utc).isoformat()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=created_at.replace(":", "-") + "-", dir=root))
    payload = {"created_at": created_at, "settings": settings, "results": results}
    (run_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (run_dir / "scores.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_COLUMNS)
        writer.writeheader()
        for result in results:
            breakdown = result["score_breakdown"] or {}
            row = {key: result.get(key) for key in _COLUMNS}
            row.update(alignment=breakdown.get("Alignment"), coverage=breakdown.get("Coverage"))
            writer.writerow({key: _csv_cell(value) for key, value in row.items()})
    (run_dir / "report.html").write_text(_html(results, settings, created_at), encoding="utf-8")
    return run_dir
