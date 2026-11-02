# Meeting summary evaluation

Evaluate an existing meeting summary against its transcript with DeepEval's native
`SummarizationMetric`. The evaluator records **alignment**, **coverage**, their minimum,
and inspectable questions, claims, answers, and verdicts. Coverage questions are generated
once per meeting and saved for reuse. Excel and HTML reports are rendered from the same
saved evaluation, without making more model calls.

The [business context](examples/business-context.md) describes the audience and purpose:
institutional sales professionals preparing advisor and client conversations from weekly
capital-markets meetings. Replace it for another audience or omit it for stock native
question generation. Contextual question generation prioritizes consequential transcript
facts; it does not add a separately scored rubric.

## Documentation

The [documentation index](docs/README.md) links to the [approach](docs/approach.md),
[scoring methodology](docs/methodology.md), [workflow and report guide](docs/workflow.md),
and [implementation reference](docs/implementation.md).

## Install

Use Python 3.11 or newer:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,notebooks]'
```

Install with `-e .` if you only need the Python helpers and batch command. DeepEval is
pinned to **4.3.2**. Set `OPENAI_API_KEY` in your environment or a local `.env` in the
working directory. Optional `OPENAI_MODEL` (default `gpt-6-luna`) and `OPENAI_BASE_URL`
select the judge and endpoint. Process environment settings override `.env`; an explicit
model overrides `OPENAI_MODEL`. The endpoint must support the selected model and structured
output.

## Evaluate in the notebook

Open the [notebook](notebooks/evaluate_meetings.ipynb):

```bash
.venv/bin/jupyter lab notebooks/evaluate_meetings.ipynb
```

Its cells load a meeting, prepare or load its frozen questions, let you review the question
list, evaluate once, and immediately save `results.json`. You can then inspect scores,
counts, paired claims and alignment verdicts, and question answers. A separate export cell
reads that JSON to produce `report.xlsx` and `report.html`; it also works after a kernel
restart by choosing the saved JSON path. Exporting does not reevaluate the summary or
require the source files.

## Evaluate folders

```bash
.venv/bin/meeting-eval evaluate \
  --transcripts examples/data/transcripts \
  --summaries examples/data/summaries \
  --business-context examples/business-context.md \
  --questions results/questions \
  --output results
```

Files in flat folders are paired by case-sensitive filename stem; `.txt` and `.md` are
supported. Meetings run sequentially with a fresh metric each time. Add `--model NAME`
to choose a judge. New question sets contain five questions by default. `--n 10` requests
a larger sample when preparing a set; omitting `--n` reuses a saved set's count.

The first evaluation of a meeting generates and saves transcript-specific factual
questions. Later evaluations load those exact questions. A changed transcript, business
context, or explicitly requested question count requires `--regenerate-questions` to
replace the saved set. Earlier evaluations retain their complete question set. Business
context changes topic selection only; it supplies no factual evidence. Frozen questions
remove variation in question selection, while claim extraction and judging can still vary.

Each evaluation creates a new run directory containing:

- `results.json`: the complete saved evaluation, run settings, question provenance,
  scores, counts, paired claims, and question answers.
- `report.xlsx`: an Excel workbook for reviewing scores and detailed verdicts.
- `report.html`: a standalone report containing the same recorded results.

Missing partners, duplicate stems, blank/unreadable files, and judge failures remain
associated with their meeting IDs. Successful results are saved as the batch progresses.
Failed scores stay empty. Execution errors produce a nonzero exit status; low quality
scores do not.

## Re-export a saved evaluation

```bash
.venv/bin/meeting-eval report results/<run-directory>/results.json
```

This creates or replaces `report.xlsx` and `report.html` beside the saved JSON. Add
`--output PATH` to write the reports elsewhere. It reads the saved evaluation only: no API
key, transcript, summary, or question-generation call is needed. A report failure leaves
the JSON available for another export attempt.

The notebook and CLI share the same Python helpers: `get_or_prepare_questions`,
`evaluate_meeting`, `new_run`, `save_evaluation`, `load_evaluation`, and `export_reports`.
`create_metric` constructs the native metric from an explicit `assessment_questions`
list; question generation belongs to the preparation helper. Create a fresh metric for
each meeting rather than reusing a metric instance.

## Read the scores

- **Alignment, 0–1:** the fraction of extracted summary claims judged supported by the
  transcript. Contradictory and unsupported/borderline claims reduce it. Inspect the
  supported-claim count and total extracted claims alongside the score.
- **Coverage, 0–1:** the fraction of source-answerable frozen questions also answerable
  from the summary. Inspect covered, eligible, and total question counts. This samples
  information preservation; it is not a percentage of all important information or
  transcript sentences.
- **Overall, 0–1:** DeepEval's minimum of alignment and coverage. There is no acceptance
  gate or additional comparative-rubric score.

Read the reason and verdicts alongside scores. Native alignment may penalize derived
outreach ideas absent from the transcript, even when labeled as suggestions. No custom
exception is added. The transcript establishes what was said, not external truth; model
extraction and judging can miss material errors. See [DeepEval's scoring
documentation](https://deepeval.com/docs/metrics-summarization).

## Check the pipeline

Run `.venv/bin/python -m pytest` for offline checks. CI checks installed dependencies,
frozen-question reuse, native metric integration, input failures, saved evaluations,
report exports, and notebook execution without calling a hosted judge.

The three fictional examples offer an optional live smoke check: `meeting_001` preserves
the main meaning, `meeting_002` omits central information, and `meeting_003` distorts
facts. Inspect explanations and verdicts instead of expecting exact scores. Preparing
new questions and evaluating summaries make billable model calls; offline tests establish
pipeline behavior, not judge accuracy. No human-reviewed calibration is included.

DeepEval telemetry is disabled. Transcripts and summaries are sent to the configured judge
endpoint, and saved question sets and evaluations can contain sensitive source text. Keep
credentials and real meeting data out of Git; generated artifacts belong under ignored
`results/`.
