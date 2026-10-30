# Meeting summary evaluation

Evaluate an existing meeting summary against its transcript with DeepEval's built-in
`SummarizationMetric`. One metric produces **alignment**, **coverage**, an overall score,
and inspectable questions, claims, verdicts, and reasoning.

The [business context](examples/business-context.md) describes the intended readers:
institutional sales professionals preparing advisor and client conversations from weekly
capital-markets meetings. It is the single place to describe audience and purpose; replace
it for another use case or omit it for stock DeepEval question generation.

## Install

Use Python 3.11 or newer:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,notebooks]'
```

Install with `-e .` if you only need the Python helper and batch command. DeepEval is pinned
to **4.3.2**. Set `OPENAI_API_KEY` in your environment or a local `.env` in the working
directory. Optional `OPENAI_MODEL` (default `gpt-6-luna`) and `OPENAI_BASE_URL` select the
judge and endpoint. Process environment settings override `.env`; an explicit `model`
overrides `OPENAI_MODEL`. The endpoint must support the selected model and structured output.

## Evaluate one meeting

```python
from pathlib import Path
from meeting_eval import create_metric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input=Path("examples/data/transcripts/meeting_001.txt").read_text(),
    actual_output=Path("examples/data/summaries/meeting_001.md").read_text(),
)
metric = create_metric(
    business_context=Path("examples/business-context.md").read_text(),
)
metric.measure(test_case)
print(metric.score_breakdown["Alignment"], metric.score_breakdown["Coverage"])
print(metric.score, metric.reason)
```

`create_metric(model=None, business_context=None, n=5, async_mode=True)` returns the native
`SummarizationMetric`. It uses score-only mode (`threshold=None`), LLM judging, and reasons.
The model can be a name or a native DeepEval model instance. In a notebook, use
`await metric.a_measure(test_case)` instead of `measure()`.

When supplied, business context changes only the native coverage-question prompt through
`SummarizationTemplate.generate_questions`. It guides which transcript facts matter to
readers; it never supplies factual evidence or required headings. Alignment and both score
calculations remain native. **Create a fresh metric for each meeting**: DeepEval retains
generated questions on the instance.

Open the [single notebook](notebooks/evaluate_meetings.ipynb) for an editable example that
also displays questions, claims, and verdicts:

```bash
.venv/bin/jupyter lab notebooks/evaluate_meetings.ipynb
```

## Evaluate folders

```bash
.venv/bin/meeting-eval \
  --transcripts examples/data/transcripts \
  --summaries examples/data/summaries \
  --business-context examples/business-context.md \
  --output results
```

Files in flat folders are paired by case-sensitive filename stem; `.txt` and `.md` are
supported. Meetings run sequentially with a fresh metric each time. Add `--model NAME` or
`--n 10` to change the judge or number of coverage questions; the default is five questions.

Each run creates a new directory containing `results.json` (native scores and details plus
run settings), `scores.csv` (one score row per meeting), and a standalone `report.html`
(score table and expandable details). Missing partners, duplicate stems, blank/unreadable
files, and judge failures are recorded against their meeting IDs. Successful meetings
remain available; failed scores stay empty. Execution errors produce a nonzero exit status;
low quality scores do not.

## Read the scores

- **Alignment, 0–1:** the fraction of extracted summary claims judged supported by the
  transcript. Contradictory and unsupported/borderline claims reduce it.
- **Coverage, 0–1:** the fraction of source-answerable assessment questions also answerable
  from the summary. This samples information preservation; it is not a percentage of all
  important information or transcript sentences.
- **Overall, 0–1:** DeepEval's minimum of alignment and coverage. There is no acceptance gate.

Read the reason and verdicts alongside scores. Native alignment may penalize derived
outreach ideas absent from the transcript, even if labeled as suggestions. No custom
exception is added. The transcript establishes what was said, not external truth; model
extraction and judgments can miss errors. Different question samples can change coverage
between runs. See [DeepEval's scoring documentation](https://deepeval.com/docs/metrics-summarization).

## Check the pipeline

Run `.venv/bin/python -m pytest` for offline checks. CI tests native metric integration,
input failures, exports, and notebook syntax without calling a hosted judge.

For an optional live smoke check, run the folder command above and inspect all three
fictional examples: `meeting_001` preserves the main meaning, `meeting_002` omits central
information, and `meeting_003` distorts facts. Look for sensible explanations and verdicts
instead of expecting exact scores. Notebook and CLI evaluations make billable model calls;
offline tests establish pipeline behavior, not judge accuracy.

DeepEval telemetry is disabled. Transcripts and summaries are sent to the configured judge
endpoint, and local outputs can contain sensitive source text. Keep credentials and real
meeting data out of Git; generated reports belong under ignored `results/`.
