# Meeting summary evaluation

Evaluate meeting summaries against the full transcript and optional explicitly supplied source context. DeepEval produces separate **faithfulness** and **important-information coverage** scores. Structured diagnostics add quotations, material issues, and prompt improvement suggestions.

The evaluator uses methodology **4**, with generic, source-grounded scoring defined in the [evaluation policy](src/meeting_eval/policy.py). Supply an existing summary in any format. Headings, section order, bullet counts, and indexing metadata are not scored.

**New to evaluation?** Start with the [learning guide](docs/learning-guide.md) and
[notebook learning path](notebooks/README.md). The foundations, DeepEval lab, and prompt
workbench run offline. They explain the methodology, show where to edit prompts, and
demonstrate why DeepEval does **not** require a question list for every metric.

Review the current prompts without model calls with `meeting-eval-prompts --show policy`,
or export a review bundle with `meeting-eval-prompts --output results/prompt-review-001`.

## Install

Python 3.11 or newer is required:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade -e '.[dev,notebooks]'
.venv/bin/python -m pip check
```

The project pins DeepEval **4.3.2** and `langchain-openai` **1.7.3**. Use the installed
environment as your notebook kernel. If importing `SingleTurnParams` fails, follow the
[notebook import troubleshooting steps](docs/learning-guide.md#fix-a-notebook-import-error)
to check and update that kernel's environment, then restart it.

Use `.venv/bin/python -m pip install --upgrade -e .` for the API and CLI without notebook dependencies. Provide `OPENAI_API_KEY` through your environment or a local `.env`. Optional `OPENAI_MODEL` and `OPENAI_BASE_URL` select the model and endpoint. Process settings take precedence over `.env`; an explicit constructor model takes precedence over `OPENAI_MODEL`. The default is `gpt-6-luna`; your endpoint must provide that model and strict structured-output support. Credentials and source text should never be committed.

## Evaluate a meeting

```python
from pathlib import Path
from meeting_eval import MeetingSummaryEvaluator

transcript = Path("examples/data/transcripts/meeting_001.txt").read_text()
summary = Path("examples/data/summaries/meeting_001.md").read_text()
with MeetingSummaryEvaluator(max_concurrency=4) as evaluator:
    result = evaluator.evaluate(
        transcript=transcript,
        summary=summary,
        meeting_id="meeting_001",
    )

print(result.execution_status, result.review_required)
for name, metric in result.metrics.items():
    print(name, metric.score, metric.reason)
for finding in result.findings:
    print(finding.severity, finding.kind, finding.description)
```

Pass `provided_context="..."` when the producer received an additional approved factual source, such as a glossary or participant context. Context is a separate source: its facts do not automatically become things said at the meeting. Missing context is not filled with outside knowledge. A [fictional context example](examples/data/contexts/meeting_001.md) demonstrates the distinction.

The generic evaluator needs no business configuration. To tailor importance and readability
to a particular audience, pass one optional `business_context` block when constructing it:

```python
business_context = Path("examples/business-context.md").read_text()
with MeetingSummaryEvaluator(business_context=business_context) as evaluator:
    result = evaluator.evaluate(transcript, summary)
```

The [example business context](examples/business-context.md) is the single place that
describes the existing Zoom workflow's audience and goals. Replace it for another meeting
type, or omit it. Business context guides salience and reader needs; it is not factual
evidence and cannot impose format grading. `provided_context` remains a separate factual source.

You can pass a configured `ChatOpenAI` instance as `model`. The caller owns and must close that model's clients. Evaluator-owned models close with `with`, `close()`, or `await aclose()`. Notebooks use `await evaluator.aevaluate(...)` and `await evaluator.aevaluate_batch(...)`; use a fresh evaluator for unrelated event loops.

## Evaluate a batch

Use `.txt` or `.md` inputs in flat transcript and summary folders, paired by case-sensitive filename stem. Per-meeting context uses the same stem in a third folder.

```python
with MeetingSummaryEvaluator() as evaluator:
    batch = evaluator.evaluate_batch(
        transcript_dir="examples/data/transcripts",
        summary_dir="examples/data/summaries",
        output_dir="results",
    )
print(batch.output_dir)
comparison = batch.to_dataframe()  # optional Pandas dependency
```

```bash
.venv/bin/meeting-eval \
  --transcripts examples/data/transcripts \
  --summaries examples/data/summaries \
  --output results \
  --max-concurrency 4
```

For shared source context, add `--context path/to/context.md`. For per-meeting files, add `--contexts path/to/context-folder`. These are mutually exclusive, as are Python's `provided_context` and `context_dir` batch arguments. Add `--business-context examples/business-context.md` to apply an audience and purpose to the whole run. See `meeting-eval --help` for metrics and request limits.

Each run creates a new directory containing `results.json`, `scores.csv`, and a standalone `report.html`. Missing partners, duplicates, empty inputs, read failures, and judge failures stay associated with meeting IDs. Successful meetings survive failures elsewhere. The CLI returns a nonzero status for incomplete execution, independently of summary-quality findings. Missing scores stay empty.

## Interpret results

- **Faithfulness, 0–1:** support for extracted claims, with ambiguity penalized. Clearly presented derived ideas may use supported premises; fabricated meeting statements, endorsements, promises, owners, or deadlines cannot. No particular label is required.
- **Coverage, 0–1:** a rubric judgment of whether consequential meeting information survives. It is not a percentage of transcript sentences. Context-only facts do not create meeting coverage requirements.
- **Review flags:** material findings, unsupported or ambiguous claims, uncertain attribution, unverified quotations, and incomplete evaluation prioritize human inspection. A high score can still hide a severe error.
- **Evidence:** `verified=True` means the exact quotation uniquely matches its indicated `transcript`, `context`, or `summary` source. It does not verify the judge's interpretation.

There is no composite score or automatic acceptance threshold. The transcript establishes what was said, not externally verified truth. Preserve uncertainty, corrections, and attribution. Assess meaning regardless of headings or layout; no structure metric or required summary template is used.

Enable optional clarity and conciseness scoring with `metrics=("faithfulness", "coverage", "clarity")`. This rubric assesses readable, economical writing without word-count or section quotas. [Methodology](docs/methodology.md) explains scoring boundaries and evidence limitations.

## Prompt experiments and calibration

The [single-meeting notebook](notebooks/01_single_meeting.ipynb) runs the complete evaluator; the [batch notebook](notebooks/02_batch_evaluation.ipynb) compares scores, findings, and exports. Both have **no saved outputs** and make billable requests when run. Optional individual metric calls in the single-meeting notebook are off by default; the new offline metric lab teaches those stages without duplicate spending. Start Jupyter with `.venv/bin/jupyter lab notebooks`.

The three bundled meetings are fictional: a faithful summary, important omissions, and material distortions. A separate [calibration manifest](calibration/fixtures.json) adds boundary cases for paraphrase, negation, units, attribution, generated outreach, context, unanswered Q&A, injection, and corrections. Expectations are draft test designs, **not human calibration labels**.

Check the manifest without model calls:

```bash
.venv/bin/python scripts/calibrate.py
```

Choose repeated judge trials explicitly:

```bash
.venv/bin/python scripts/calibrate.py --run --model gpt-6-luna --trials 3
```

This saves detailed results, per-case variability, and a separate human annotation template. Errors remain execution failures, not low summary scores. See [calibration workflow](docs/calibration.md) for blind human review, adjudication, holdouts, and paired prompt comparisons.

Compare saved baseline and candidate evaluations with matching judges, policy, business context, and meeting sources:

```bash
.venv/bin/meeting-eval-compare baseline/results.json candidate/results.json \
  --output results/comparison-001
```

The comparison preserves paired score changes and material findings, excludes incomplete pairs from aggregate deltas, and rejects changed or insufficiently recorded evaluation conditions. Summaries may differ; transcripts, supplied source context, and business context must match. Inspect material regressions even when a mean improves.

## Runtime, privacy, and verification

`max_concurrency` bounds judge requests, including calls within metrics. Token limits can require lower concurrency. Retries are bounded at the request layer; unsupported configuration requires correction. Full sources are retained; oversized inputs fail explicitly. A default evaluation normally makes about six judge requests before retries. Source length dominates cost. The offline suite and CI make no paid model calls.

Reports stay local and may contain sensitive quotations. DeepEval telemetry is disabled and automatic LangSmith tracing is suppressed for judge calls. Sources are sent to your configured endpoint. Keep outputs under ignored `results/` and follow your organization's data-handling rules.

Run `.venv/bin/python -m pytest` for offline checks. [Validation](docs/validation.md) distinguishes historical runs from the current methodology. Compare recorded prompt, policy, source, and judge provenance; inspect both improvements and regressions before changing a producer prompt.
