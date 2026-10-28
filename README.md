# Meeting summary evaluation

A small Python package for evaluating Zoom AI meeting summaries against their full meeting transcripts. DeepEval produces separate **faithfulness** and **important-information coverage** scores. A structured diagnostic review adds source quotations, missing topics, unsupported claims, and material-risk flags.

The intended readers are Fidelity Institutional sales professionals and the people improving the Zoom summary prompt. Scores describe transcript-grounded summary quality; they do not verify market facts or approve client communications.

## Install

Python 3.11 or newer is required. From the project root:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev,notebooks]'
```

The core package does not require Pandas or Jupyter. Install with `pip install -e .` if you only need the Python API and CLI.

Keep credentials in the existing `.env` file:

```dotenv
OPENAI_API_KEY=your-key
# OPENAI_MODEL=gpt-6-luna
# OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
```

Existing process environment settings take precedence over `.env`. A constructor `model` value takes precedence over `OPENAI_MODEL`. The default model is `gpt-6-luna`, which is listed in the [official OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna). Availability and structured-output support must also be provided by your endpoint. An unsupported model or endpoint produces an actionable error; the package does not silently choose a different model.

## Evaluate a meeting

```python
from pathlib import Path
from meeting_eval import MeetingSummaryEvaluator

evaluator = MeetingSummaryEvaluator(model="gpt-6-luna", max_concurrency=4)
result = evaluator.evaluate(
    transcript=Path("examples/data/transcripts/meeting_001.txt").read_text(encoding="utf-8"),
    summary=Path("examples/data/summaries/meeting_001.md").read_text(encoding="utf-8"),
    meeting_id="meeting_001",
)

for name, metric in result.metrics.items():
    print(name, metric.score, metric.reason)
print("Review required:", result.review_required)
for finding in result.findings:
    print(finding.severity, finding.kind, finding.description)
```

For an existing LangChain model, pass the instance directly:

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from meeting_eval import MeetingSummaryEvaluator

load_dotenv(".env", override=False)
judge = ChatOpenAI(model="gpt-6-luna", timeout=120, max_retries=2)
evaluator = MeetingSummaryEvaluator(model=judge, max_concurrency=4)
```

Avoid adding sampling parameters unless your model supports them. The adapter uses schema-based structured output. No web search, retrieval service, or reference summary is needed.

## Evaluate a batch

Place `.txt` or `.md` inputs in two flat folders, using matching, case-sensitive file stems:

```text
data/transcripts/meeting_001.txt
data/summaries/meeting_001.md
```

```python
batch = evaluator.evaluate_batch(
    transcript_dir="examples/data/transcripts",
    summary_dir="examples/data/summaries",
    output_dir="results",
)
print(batch.output_dir)
comparison = batch.to_dataframe()  # optional Pandas dependency
print(comparison)
```

The CLI runs the same workflow:

```bash
.venv/bin/meeting-eval \
  --transcripts examples/data/transcripts \
  --summaries examples/data/summaries \
  --output results \
  --max-concurrency 4
```

Use `--max-concurrency 1` for sequential debugging. Run `meeting-eval --help` for model and metric options. Each run writes detailed JSON, a CSV comparison, and a standalone HTML report to the run directory printed by the command. The HTML report can be opened locally without a server.

Missing partners, ambiguous duplicate stems, empty files, and evaluation errors are recorded against their meeting IDs. A problem with one meeting does not discard successful results from the others. Missing or failed scores remain empty rather than becoming zero.

When finished, call `evaluator.close()` in scripts or `await evaluator.aclose()` in notebooks. Scripts can also use `with MeetingSummaryEvaluator(...) as evaluator:`. Use a fresh evaluator when switching between synchronous execution and unrelated asynchronous event loops. A caller-supplied `ChatOpenAI` remains caller-owned.

## Read the results

- **Faithfulness, 0–1:** the proportion of extracted summary claims supported by the transcript, with ambiguous claims penalized. A high ratio can still hide one consequential error.
- **Coverage, 0–1:** a rubric-based judgment of whether important meeting information survives summarization. It is not the fraction of transcript sentences retained.
- **Review flags:** unsupported or ambiguous faithfulness claims, material diagnostic findings, uncertainty, missing or unverified quotations, and incomplete evaluation warrant human inspection independently of the scores. These are triage flags, not an acceptance gate.
- **Evidence:** `verified=True` means the exact quoted text was uniquely located in the indicated source. Missing or repeated quotations remain unverified without assigned source lines. Verification does not establish that the judge interpreted the quote correctly.

The default output has no weighted average or automatic pass/fail threshold. Sales implications must have been explicitly discussed; a plausible inference introduced by the summary is still unsupported under this project's policy. The transcript is the grounding source even when it contains transcription errors, which should produce an uncertainty finding.

Add optional checks with `metrics=("faithfulness", "coverage", "structure", "clarity")`. Structure checks the expected sections and executive bullet count; clarity uses a separate readability rubric. Q&A expectations depend on whether meaningful Q&A occurred in the transcript. These checks are optional because layout and style are secondary to accurate, useful content.

See [the methodology](docs/methodology.md) for scoring rubrics, metric tradeoffs, evidence limitations, and diagnostic categories.

## Experiment in Jupyter

```bash
.venv/bin/jupyter lab notebooks
```

1. [Single meeting evaluation](notebooks/01_single_meeting.ipynb): load a pair, create `ChatOpenAI`, run individual DeepEval metrics, and inspect the package's evidence-rich result.
2. [Batch evaluation and analysis](notebooks/02_batch_evaluation.ipynb): evaluate the examples concurrently, compare a Pandas table and score chart, inspect recurring findings, and export an analysis table.

The notebooks use `await evaluator.aevaluate(...)` and `await evaluator.aevaluate_batch(...)` because Jupyter already runs an event loop. Use the synchronous methods in ordinary scripts. Both notebooks make real, billable API calls. The single-meeting notebook deliberately evaluates the two metrics individually and then repeats them through the complete evaluator for teaching purposes.

All bundled meetings are explicitly fictional software fixtures. `meeting_001` is a faithful example, `meeting_002` deliberately omits the central themes, and `meeting_003` contains material distortions. Expect qualitative differences, not fixed scores: LLM judgments can vary.

## Runtime and limitations

- `max_concurrency` bounds in-flight judge requests across meetings and metric internals. The default is four; provider token-per-minute limits may require a lower value.
- Request timeouts and bounded retries handle transient failures. Authentication, model access, unsupported parameters, and invalid inputs require correction rather than endless retries.
- A default meeting evaluation requires approximately six judge requests before retries: claim/truth extraction, faithfulness judgments and reasoning, coverage, and structured diagnostics. Actual usage depends on the selected metrics and the installed DeepEval behavior. Full-transcript prompts dominate input-token usage.
- The full transcript is retained. Oversized inputs fail clearly instead of being silently truncated or replaced by a generated synopsis. Chunking is deliberately outside the initial version.
- Results stay in local files. DeepEval telemetry is disabled and automatic LangSmith tracing is suppressed for judge calls; no Confident AI evaluation upload is configured. Judge prompts are sent to your configured model endpoint. Reports may contain sensitive source quotations.
- Human calibration is still needed. Review a representative set of good and flawed summaries before using score trends to make operational decisions; synthetic examples demonstrate mechanics rather than establish judge accuracy.

## Validate and extend

```bash
.venv/bin/python -m pytest
```

Tests use controlled judge responses so they do not spend API credits. Live validation should include a model/schema compatibility check and all three fictional pairs. Preserve real API errors separately from summary-quality findings.

See the [validation record](docs/validation.md) for the executed notebook results, observed judge variation, and remaining calibration work.

To change the judge, pass another configured `ChatOpenAI` instance. To change evaluation policy, edit the reusable metric rubrics and diagnostic instructions together, record the change in run metadata, and rerun a fixed human-reviewed comparison set. Keep model and rubric versions consistent when comparing runs.
