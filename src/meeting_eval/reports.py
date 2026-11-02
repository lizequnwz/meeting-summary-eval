"""Export one saved evaluation as an Excel workbook and standalone HTML."""

import html
import json
import math
from pathlib import Path
import tempfile

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .evaluation import load_evaluation


_DEFINITIONS = (
    ("Alignment", "Supported claims / all extracted summary claims. No claims gives 0."),
    ("Coverage", "Covered eligible questions / source-answerable questions. No eligible questions gives 0."),
    ("Overall", "The native minimum of alignment and coverage; scores are not an acceptance gate."),
    ("Excluded coverage", "Questions the transcript cannot answer are excluded from the coverage denominator."),
    ("Question generation cost", "Historical generation cost is provenance. It is current spending only when questions were generated in this run."),
    ("Unknown cost", "A blank cost means unavailable, not zero."),
)


def _escape(value):
    return html.escape(str(value), quote=True)


def _score(value):
    return "—" if value is None else f"{value:.2f}"


def _flatten(prefix, value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten(f"{prefix}.{key}" if prefix else key, item)
    else:
        yield prefix, json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value


def _run_details(run, source):
    yield "Run", "Saved evaluation", str(source.resolve())
    for key in ("schema_version", "created_at", "updated_at", "status", "settings"):
        for field, value in _flatten(key, run[key]):
            yield "Run", field, value
    for name, definition in _DEFINITIONS:
        yield "Definitions", name, definition
    for result in run["results"]:
        meeting_id = result["meeting_id"]
        for key in ("inputs", "question_generation_performed", "evaluation"):
            for field, value in _flatten(key, result[key]):
                yield meeting_id, field, value
        question_set = result["question_set"]
        if question_set is not None:
            for key, value in question_set.items():
                if key != "questions":
                    for field, item in _flatten(f"question_set.{key}", value):
                        yield meeting_id, field, item
        for index, truth in enumerate(result["truths"], 1):
            yield meeting_id, f"Transcript statement {index}", truth


def _append(sheet, values):
    """Store external text literally; openpyxl otherwise interprets '=' as a formula."""
    row = sheet.max_row + 1 if sheet.cell(1, 1).value is not None else 1
    for column, value in enumerate(values, 1):
        cell = sheet.cell(row, column)
        if isinstance(value, str):
            # Use Excel line breaks consistently; the saved JSON remains unchanged.
            value = value.replace("\r\n", "\n").replace("\r", "\n")
            if len(value) > 32767:
                raise ValueError(f"Excel text exceeds the 32767-character cell limit at {sheet.title}!{cell.coordinate}.")
            if ILLEGAL_CHARACTERS_RE.search(value):
                raise ValueError(f"Excel text contains unsupported control characters at {sheet.title}!{cell.coordinate}.")
            cell.value = value
            cell.data_type = "s"
        else:
            cell.value = value
        cell.alignment = Alignment(vertical="top", wrap_text=True)


def _sheet(workbook, name, headers, rows, widths, *, freeze="A2"):
    sheet = workbook.create_sheet(name)
    _append(sheet, headers)
    for row in rows:
        _append(sheet, row)
    for cell in sheet[1]:
        cell.font = Font(name="Calibri", bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="245B49")
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name="Calibri", size=11, color="20352F")
            cell.alignment = Alignment(
                vertical="top", wrap_text=True, indent=1,
                horizontal="right" if type(cell.value) in (int, float) else "left",
            )
    sheet.row_dimensions[1].height = 32
    sheet.freeze_panes = freeze
    sheet.auto_filter.ref = sheet.dimensions
    sheet.sheet_view.showGridLines = False
    for index, width in enumerate(widths, 1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    for row in sheet.iter_rows(min_row=2):
        lines = max(
            sum(max(1, math.ceil(len(line) / max(widths[index] - 3, 1)))
                for line in str(cell.value or "").split("\n"))
            for index, cell in enumerate(row)
        )
        sheet.row_dimensions[row[0].row].height = min(409, max(26, 15 * lines + 6))
    return sheet


def _workbook(run, source):
    workbook = Workbook()
    workbook.remove(workbook.active)
    score_rows = []
    coverage_rows = []
    alignment_rows = []
    for result in run["results"]:
        scores, counts = result["scores"] or {}, result["counts"] or {}
        evaluation = result["evaluation"] or {}
        question_set = result["question_set"] or {}
        generation = question_set.get("generation", {})
        score_rows.append((
            result["meeting_id"], result["status"], scores.get("alignment"),
            counts.get("alignment_supported"), counts.get("alignment_total"), scores.get("coverage"),
            counts.get("coverage_covered"), counts.get("coverage_eligible"), counts.get("coverage_total"),
            scores.get("overall"), result["reason"], result["error"], question_set.get("question_set_id"),
            evaluation.get("model"), evaluation.get("duration_seconds"), evaluation.get("cost"),
            result["question_generation_performed"],
            generation.get("cost") if result["question_generation_performed"] else None,
            (f'min({scores["alignment"]:.6g}, {scores["coverage"]:.6g}) = {scores["overall"]:.6g}'
             if scores else None),
        ))
        coverage_rows.extend((
            result["meeting_id"], item["question_id"], item["question"], item["transcript_answer"],
            item["summary_answer"], "yes" if item["in_denominator"] else "no", item["outcome"],
        ) for item in result["coverage"])
        alignment_rows.extend((
            result["meeting_id"], item["claim_id"], item["claim"], item["verdict"], item["reason"],
        ) for item in result["alignment"])
    scores = _sheet(workbook, "Scores", (
        "Meeting", "Execution", "Alignment", "Supported claims", "All claims", "Coverage",
        "Covered eligible questions", "Eligible questions", "All questions", "Overall", "Reason", "Error",
        "Question set", "Evaluation model", "Evaluation seconds", "Evaluation cost (USD)",
        "Questions generated this run", "Question generation cost this run (USD)", "Overall calculation",
    ), score_rows, (24, 16, 14, 18, 14, 14, 24, 19, 17, 14, 65, 55, 38, 24, 21, 23, 26, 31, 43))
    for row in scores.iter_rows(min_row=2):
        for column in (3, 6, 10):
            row[column - 1].number_format = "0.00"
        for column in (15, 16, 18):
            row[column - 1].number_format = "0.00####"
    coverage = _sheet(workbook, "Coverage", (
        "Meeting", "Question ID", "Question", "Transcript answerable", "Summary answerable",
        "In denominator", "Outcome",
    ), coverage_rows, (24, 16, 85, 24, 24, 20, 18), freeze="C2")
    alignment = _sheet(workbook, "Alignment", (
        "Meeting", "Claim ID", "Claim", "Native verdict", "Native reason",
    ), alignment_rows, (24, 16, 85, 20, 85), freeze="C2")
    for sheet, column, flagged in ((coverage, 7, {"missing"}), (alignment, 4, {"no", "borderline"})):
        for row in sheet.iter_rows(min_row=2):
            if row[column - 1].value in flagged:
                for cell in row:
                    cell.fill = PatternFill("solid", fgColor="FFF0E8")
    details = _sheet(workbook, "Run details", ("Scope", "Field", "Value"), _run_details(run, source), (25, 48, 105))
    for row in details.iter_rows(min_row=2):
        if row[1].value.endswith((".cost", ".duration_seconds")):
            row[2].number_format = "0.00####"
    return workbook


def _table(headers, rows):
    heading = "".join(f"<th scope=\"col\">{_escape(value)}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{_escape(value) if value is not None else '—'}</td>" for value in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>'


def _coverage_table(items):
    return _table(("ID", "Question", "Transcript", "Summary", "In denominator", "Outcome"), (
        (item["question_id"], item["question"], item["transcript_answer"], item["summary_answer"],
         "yes" if item["in_denominator"] else "no", item["outcome"]) for item in items
    ))


def _alignment_table(items):
    return _table(("ID", "Summary claim", "Native verdict", "Native reason"), (
        (item["claim_id"], item["claim"], item["verdict"], item["reason"]) for item in items
    ))


def _arithmetic(result):
    counts, scores = result["counts"], result["scores"]
    alignment = f'{counts["alignment_supported"]}/{counts["alignment_total"]} = {_score(scores["alignment"])}'
    coverage = f'{counts["coverage_covered"]}/{counts["coverage_eligible"]} = {_score(scores["coverage"])}'
    if counts["alignment_total"] == 0:
        alignment = "0 (no extracted claims)"
    if counts["coverage_eligible"] == 0:
        coverage = "0 (no source-answerable questions)"
    return (
        f'<p class="arithmetic">Alignment: {alignment}. Coverage: {coverage}. '
        f'Overall: min({_score(scores["alignment"])}, {_score(scores["coverage"])}) = {_score(scores["overall"])}.</p>'
    )


def _html(run, source):
    score_rows, meetings = [], []
    for index, result in enumerate(run["results"]):
        scores = result["scores"] or {}
        score_rows.append(
            f'<tr><th scope="row"><a href="#meeting-{index}">{_escape(result["meeting_id"])}</a></th>'
            f'<td>{_escape(result["status"])}</td>'
            + "".join(f'<td class="number">{_score(scores.get(key))}</td>' for key in ("alignment", "coverage", "overall"))
            + "</tr>"
        )
        sections = []
        if result["error"]:
            sections.append(f'<p class="error">{_escape(result["error"])}</p>')
        if result["status"] == "unprocessed":
            sections.append("<p>This meeting has not been evaluated.</p>")
        if result["reason"]:
            sections.append(f'<p class="preserve">{_escape(result["reason"])}</p>')
        if result["status"] == "completed":
            sections.append(_arithmetic(result))
            missing = [item for item in result["coverage"] if item["outcome"] == "missing"]
            unsupported = [item for item in result["alignment"] if item["verdict"] != "yes"]
            if missing:
                sections.append("<h3>Missing information</h3>" + _coverage_table(missing))
            if unsupported:
                sections.append("<h3>Unsupported or borderline claims</h3>" + _alignment_table(unsupported))
            if not missing and not unsupported:
                sections.append("<p>No missing sampled information or unsupported claims were identified.</p>")
            coverage = sorted(result["coverage"], key=lambda item: {"missing": 0, "excluded": 1, "covered": 2}[item["outcome"]])
            alignment = sorted(result["alignment"], key=lambda item: {"no": 0, "borderline": 1, "yes": 2}[item["verdict"]])
            sections.append(f'<details><summary>All coverage questions ({len(coverage)})</summary>{_coverage_table(coverage)}</details>')
            sections.append(f'<details><summary>All summary claims ({len(alignment)})</summary>{_alignment_table(alignment)}</details>')
        provenance = {key: result[key] for key in ("inputs", "question_set", "question_generation_performed", "evaluation", "truths")}
        sections.append('<details><summary>Native details and provenance</summary><pre>'
                        + _escape(json.dumps(provenance, indent=2, ensure_ascii=False)) + "</pre></details>")
        meetings.append(
            f'<section class="meeting" id="meeting-{index}"><h2>{_escape(result["meeting_id"])}</h2>'
            f'<p class="muted">Execution: {_escape(result["status"])}</p>' + "".join(sections) + "</section>"
        )
    completed = sum(result["status"] == "completed" for result in run["results"])
    failed = sum(result["status"] == "error" for result in run["results"])
    unprocessed = len(run["results"]) - completed - failed
    definitions = "".join(f"<dt>{_escape(label)}</dt><dd>{_escape(text)}</dd>" for label, text in _DEFINITIONS)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meeting summary evaluation</title><style>
:root{{font-family:system-ui,sans-serif;color:#20352f;background:#f4f6f4;line-height:1.55}}
*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1180px;margin:auto;padding:40px 22px 64px}}
h1{{font-size:2rem;letter-spacing:-.04em;line-height:1.15}}h2{{font-size:1.35rem;margin:0}}h3{{font-size:1rem;margin:24px 0 8px}}
p{{margin:12px 0}}.muted{{color:#5c7067;font-size:.9rem}}.panel,.meeting{{background:white;border:1px solid #d6e1da;border-radius:10px;margin:20px 0;padding:22px}}
.table-wrap{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;text-align:left;font-size:.9rem}}
th,td{{padding:12px;border-bottom:1px solid #e5ebe7;vertical-align:top;min-width:80px;overflow-wrap:anywhere}}thead th{{color:#5c7067;font-size:.8rem}}tbody tr:last-child>*{{border-bottom:0}}
.number{{font-variant-numeric:tabular-nums}}a{{color:#216b53}}summary{{padding:14px 0;cursor:pointer;font-weight:650;overflow-wrap:anywhere}}
pre{{font-size:.8rem;background:#f4f7f4;padding:14px;border-radius:6px;white-space:pre-wrap;overflow-wrap:anywhere}}
.preserve,.arithmetic{{white-space:pre-wrap;overflow-wrap:anywhere}}.arithmetic{{background:#eef5ef;padding:12px}}
.error{{color:#963e32;background:#fff1ed;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere}}
dt{{font-weight:650;margin-top:12px}}dd{{margin-left:0}}:focus-visible{{outline:3px solid #76aa94;outline-offset:3px}}
@media(max-width:600px){{main{{padding:24px 12px}}.panel,.meeting{{padding:14px}}th,td{{padding:10px 7px}}}}
@media print{{:root{{background:white}}main{{padding:0}}.meeting,.panel{{break-inside:avoid}}details:not([open])>*:not(summary){{display:block}}}}
</style></head><body><main>
<p class="muted">DeepEval · saved evaluation</p><h1>Meeting summary evaluation</h1>
<p>Run status: {_escape(run["status"])}. {completed} completed · {failed} errors · {unprocessed} unprocessed.</p>
<p class="muted">All scores range from 0 to 1. Overall = min(alignment, coverage). Read the verdicts alongside scores.</p>
<div class="panel table-wrap"><table><thead><tr><th>Meeting</th><th>Execution</th><th>Alignment</th><th>Coverage</th><th>Overall</th></tr></thead><tbody>{''.join(score_rows)}</tbody></table></div>
{''.join(meetings)}
<section class="panel"><h2>How to read the scores</h2><dl>{definitions}</dl></section>
<details class="panel"><summary>Run details</summary>{_table(("Scope", "Field", "Value"), _run_details(run, source))}</details>
<p class="muted">Saved {_escape(run["updated_at"])}. Reports use this saved evaluation without additional judging.</p>
</main></body></html>"""


def export_reports(results_json: str | Path, output_dir: str | Path | None = None) -> Path:
    """Render saved, validated results without reading inputs or contacting a judge."""
    source = Path(results_json)
    run = load_evaluation(source)
    destination = Path(output_dir) if output_dir is not None else source.parent
    if source.resolve() in {(destination / "report.xlsx").resolve(), (destination / "report.html").resolve()}:
        raise ValueError("Report destinations must differ from the saved evaluation file.")
    destination.mkdir(parents=True, exist_ok=True)
    # Build both files before replacing either existing report.
    with tempfile.TemporaryDirectory(prefix=".report-", dir=destination) as temporary:
        staging = Path(temporary)
        workbook = _workbook(run, source)
        try:
            workbook.save(staging / "report.xlsx")
        finally:
            workbook.close()
        (staging / "report.html").write_text(_html(run, source), encoding="utf-8")
        (staging / "report.xlsx").replace(destination / "report.xlsx")
        (staging / "report.html").replace(destination / "report.html")
    return destination
