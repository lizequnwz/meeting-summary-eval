# Workflow and reports

Run commands from the repository root after completing the [installation steps](../README.md#install).
New question preparation and summary evaluation make billable model calls. Loading saved
questions, saving records, and exporting reports do not.

Preparation and evaluation use LangChain chat models with structured output. A CLI model
name selects `ChatOpenAI`; Python and the notebook can also receive a configured
`BaseChatModel` instance. Defaults use `OPENAI_MODEL` or `gpt-6-luna`, with a 120-second
timeout and two retries. Endpoint precedence is `OPENAI_BASE_URL`, `OPENAI_API_BASE`, then
`https://api.openai.com/v1`. See the [model setup example](implementation.md#configure-a-langchain-model).

## Evaluate folders

```bash
.venv/bin/meeting-eval evaluate \
  --transcripts examples/data/transcripts \
  --summaries examples/data/summaries \
  --business-context examples/business-context.md \
  --questions results/questions \
  --output results
```

Transcript and summary folders must be flat. Files with `.txt` and `.md` extensions are
paired by case-sensitive filename stem; extensions are matched without regard to case.
For example, `meeting_001.txt` pairs with `meeting_001.md`. Multiple supported files with
the same stem in one folder are an error rather than a choice between files. Unsupported
extensions and nested files are not evaluated.

Meetings run sequentially. The command prints each meeting's scores or error and the
saved evaluation/report location. A new run directory is created for each evaluation.

| Option | Behavior |
| --- | --- |
| `--transcripts`, `--summaries` | Required folders to pair by meeting ID |
| `--business-context` | Optional UTF-8 file describing audience and purpose; omitting it uses stock native question generation |
| `--questions` | Shared question directory; defaults to `results/questions`, independently of `--output` |
| `--output` | Parent directory for new evaluation runs; defaults to `results` |
| `--n` | Positive question count; a new set defaults to five, while an existing set retains its count when this option is omitted |
| `--model` | Judge name; otherwise `OPENAI_MODEL` or the project's default is used |
| `--regenerate-questions` | Explicitly replace question sets for the applicable meetings in this batch |

Keep the same question directory and meeting IDs when comparing different summary
versions. A new summary can live in a different summary folder with the same filename
stem. Changing the summary alone reuses the questions and performs a new evaluation.

## Regenerate questions deliberately

A saved set becomes stale when the normalized transcript, business context, generation
prompt fingerprint, or explicitly requested count differs. The affected meeting receives
an error before scoring until regeneration is requested.

For example, after changing the business context, deliberately prepare ten questions:

```bash
.venv/bin/meeting-eval evaluate \
  --transcripts examples/data/transcripts \
  --summaries examples/data/summaries \
  --business-context examples/business-context.md \
  --questions results/questions \
  --output results \
  --n 10 \
  --regenerate-questions
```

The flag regenerates every applicable meeting in the batch, including sets that were
already current. Remove the flag on subsequent runs. Omitting `--n` while regenerating a
valid saved set retains its count. An invalid set can be recreated explicitly; without
an explicit count, that recreation defaults to five.

Changing the judge or endpoint does not invalidate a question set. The generation model
and evaluation model are recorded separately. Changing the question count with `--n`
does require regeneration if it differs from an existing set.

The current file at `results/questions/<meeting_id>.json` is replaced only after new
questions have been generated and validated successfully. Earlier saved evaluations
retain snapshots of their original sets. A new question sample changes the basis of a
coverage comparison; check the question-set ID when comparing scores.

## Use the notebook

Open [evaluate_meetings.ipynb](../notebooks/evaluate_meetings.ipynb) from the repository
root or its `notebooks` directory and select the repository's Python environment.

1. Load a transcript, summary, and optional business context. Choose the shared question
   directory, output directory, judge, and optional count.
2. Run the preparation cell. It generates a missing set or loads an existing matching set,
   then displays its questions and provenance. To replace a set deliberately, set
   `regenerate_questions = True`, run preparation, then return the setting to `False`.
3. Run evaluation/save. It creates a new run directory, evaluates once with those
   questions, and immediately saves the returned record to JSON.
4. Inspect the scores, counts, paired claims, coverage outcomes, and any execution error.
5. Run export to create Excel and HTML from that saved JSON.

Each execution of the evaluation cell makes new judging calls and creates a new run.
Only the questions are reused automatically. The notebook clears its generation-performed
flag after saving so rerunning evaluation does not count historical preparation as new
spending again.

After restarting the kernel, run only the standalone export cell: uncomment its
`results_json = Path(...)` line and enter the absolute path of an existing saved evaluation.
Source loading, question preparation, and evaluation cells are unnecessary for export.

## Read the outputs

A completed batch run normally contains:

```text
results/
  questions/
    meeting_001.json
  run-<timestamp>-<unique-suffix>/
    results.json
    report.xlsx
    report.html
```

`results.json` is the saved evaluation. It contains the exact question snapshot, verdicts,
scores, counts, and provenance needed to rebuild both reports. Input paths and normalized
text hashes identify the evaluated sources; full verbatim source files are not embedded.

The workbook has four sheets:

| Sheet | How to use it |
| --- | --- |
| **Scores** | Compare native scores with their claim/question numerators and denominators; inspect the overall calculation, reason, or execution error |
| **Coverage** | Inspect each frozen question, both answerability verdicts, denominator inclusion, and covered/missing/excluded outcome |
| **Alignment** | Read each extracted claim directly beside its native verdict and reason |
| **Run details** | Check source identifiers, context, question-set ID, generation and evaluation models, versions, timing, costs, and extracted transcript statements |

Workbook scores and counts are recorded numeric values. The overall calculation is
explanatory text, not an editable Excel scoring model. Displayed scores use two decimal
places; saved JSON and numeric workbook cells retain the underlying precision.

HTML presents missing sampled information and unsupported or borderline claims first
within each meeting. Expand the all-question, all-claim, and provenance sections to
investigate further. Question and claim IDs preserve their original correspondence even
when details are presented in issue-first order.

An excluded question did not receive a source `yes` and is outside the coverage
denominator. An execution error has blank scores; a native zero score is a measured
result. See [Methodology](methodology.md) for the exact rules and zero-denominator cases.

New preparation and evaluation calls leave USD cost fields empty. LangChain's standard
usage metadata provides token counts rather than currency, and the project does not
infer prices. Blank cost means unavailable, not zero. Numeric cost observations already
present in saved data are rendered as recorded; a reused set's generation cost remains
historical provenance. These fields are not a billing reconciliation. The reports provide
no invented per-question explanations or verified transcript passage citations.

## Rebuild reports without reevaluating

Replace the placeholder run directory with the path printed by an evaluation:

```bash
.venv/bin/meeting-eval report results/<run-directory>/results.json
```

This creates or replaces `report.xlsx` and `report.html` beside the JSON. To write them
elsewhere, add `--output /path/to/report-directory`. The saved JSON stays in its original
location and is not modified. API credentials, source files, and current question files
are unnecessary for this operation.

## Failures and recovery

The batch command writes an initial saved record before model calls and checkpoints after
question preparation and each meeting. Missing partners, duplicate stems, unreadable or
blank files, stale questions, and judge failures remain attached to meeting IDs. Other
meetings can still complete. A run status of `completed` means processing finished; check
the individual meeting statuses for errors.

On a handled interruption, completed records remain saved, remaining meetings are marked
`unprocessed`, and the command prints the partial JSON path. It attempts partial report
export, but report failure does not hide that saved path. This checkpoint behavior belongs
to the batch command: the notebook saves its single meeting after evaluation returns.

| Situation | Next step |
| --- | --- |
| Stale or invalid saved questions | Check the inputs and count, then explicitly regenerate the intended sets |
| Missing or duplicate input partners | Correct the file stems or duplicates and run evaluation again |
| Judge failure | Check the selected model, endpoint, credentials, and connection; provider error messages are intentionally redacted |
| Report export fails after evaluation | Keep the saved JSON and retry the `report` command after fixing the destination or export problem |
| Interrupted batch | Inspect or export the partial JSON; rerunning evaluation scores valid pairs again while reusing matching questions |
| Excel rejects a text field | Inspect the saved field for unsupported control characters or Excel's 32,767-character cell limit; JSON remains available |

CLI exit codes are `0` for successful execution, `1` for a finished evaluation containing
meeting errors, `2` for argument or top-level input/output/export errors, and `130` for a
handled keyboard interruption. Low quality scores do not cause a nonzero status. A
successful `report` command returns `0` even when the saved evaluation contains errors.

For implementation details and offline checks, see [Implementation](implementation.md).
