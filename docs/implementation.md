# Implementation reference

The pipeline prepares a reusable question set, evaluates a summary once, saves a complete
record, then renders that record. Report rebuilding is independent of the original inputs
and judge. See the [README](../README.md) for commands and the
[notebook](../notebooks/evaluate_meetings.ipynb) for the shared Python workflow.

## Modules and boundaries

| Module | Responsibility |
| --- | --- |
| [`inputs.py`](../src/meeting_eval/inputs.py) | Normalize text; pair flat `.txt`/`.md` folders by case-sensitive filename stem; retain missing/duplicate-partner errors. |
| [`judge.py`](../src/meeting_eval/judge.py) | Adapt a LangChain chat model to DeepEval's judge interface; invoke Pydantic response schemas through structured output. |
| [`metrics.py`](../src/meeting_eval/metrics.py) | Resolve the judge, validate explicit questions, construct a fresh native DeepEval metric. |
| [`contradiction.py`](../src/meeting_eval/contradiction.py) | Independently extract summary claims and judge them against the full transcript; fail contradictions and expose unsupported warnings. |
| [`questions.py`](../src/meeting_eval/questions.py) | Generate, validate, fingerprint, persist, and reuse transcript-specific questions. |
| [`evaluation.py`](../src/meeting_eval/evaluation.py) | Evaluate a meeting; pair native verdicts with claims/questions; validate and save canonical run records. |
| [`storage.py`](../src/meeting_eval/storage.py) | Hash normalized content, sanitize endpoint provenance, normalize reported costs, create run directories, replace JSON files atomically. |
| [`reports.py`](../src/meeting_eval/reports.py) | Load saved evaluations and export Excel/HTML without judging or rereading source files. |
| [`cli.py`](../src/meeting_eval/cli.py) | Orchestrate sequential folder evaluations, checkpoints, interruption handling, and report-only commands. |
| [`__init__.py`](../src/meeting_eval/__init__.py) | Load local settings before DeepEval initialization and expose the shared helpers. |

The package loads `.env` from the working directory with existing environment values taking
precedence. It disables DeepEval telemetry, DeepEval's own dotenv loading, and legacy keyfile
loading. Public model arguments accept `str | BaseChatModel | None`. A model name or omitted
model creates LangChain's `ChatOpenAI`; a configured `BaseChatModel` instance is reused.
The default model name is `OPENAI_MODEL`, then `gpt-6-luna`. Endpoint precedence is
`OPENAI_BASE_URL`, `OPENAI_API_BASE`, then `https://api.openai.com/v1`. Default construction
uses a 120-second timeout and two retries. An injected model retains its own configuration.
Configured LangChain callbacks and tracing also retain their normal behavior and can
receive prompts and responses.

`LangChainJudge` is the internal bridge to DeepEval's `DeepEvalBaseLLM` interface. Every
generation call uses `model.with_structured_output(schema, include_raw=True)` followed by
`invoke` or `ainvoke`. The schemas are Pydantic models supplied by the native metric,
question preparation, or standalone contradiction metric. Injected models must support
that structured-output interface.
Response parsing failures remain evaluation errors; the adapter preserves native prompts
and scoring. Question preparation and metric creation each wrap the configured model with
the same adapter implementation.

## Public Python APIs

These helpers are exported by `meeting_eval`. Paths may be strings or `Path` objects.
Every `model` parameter accepts a model name, a LangChain `BaseChatModel`, or `None`.

| Helper | Contract |
| --- | --- |
| `await get_or_prepare_questions(meeting_id, transcript, *, business_context=None, questions_dir=Path("results/questions"), model=None, n=None, regenerate=False)` | Returns `(question_set, generated)`. `generated` is true only after newly generated questions have been validated and persisted. A matching saved set returns false without resolving a judge. |
| `await evaluate_meeting(meeting_id, transcript, summary, *, question_set, model=None, transcript_path=None, summary_path=None, question_generation_performed=False)` | Returns a completed/error meeting dictionary. Requires an explicit question snapshot and never generates questions. Pass the preparation flag through for spending provenance. Source paths are optional provenance; the supplied strings are evaluated. |
| `create_metric(model=None, *, assessment_questions, async_mode=True)` | Returns a fresh native `SummarizationMetric`, with `n=len(assessment_questions)`, `threshold=None`, `eval_mode="llm"`, and reasons enabled. Direct native measurement does not create a canonical saved run. |
| `SummaryContradictionMetric(model=None, *, async_mode=True)` | Returns a separate custom DeepEval metric. `measure(LLMTestCase(input=transcript, actual_output=summary))` or `await a_measure(...)` records a binary contradiction score, three-way claim verdicts, counts, and rates on the metric instance. Does not create a saved run or change the native pipeline. See [standalone usage](contradiction-metric.md). |
| `new_run(settings=None)` | Returns an empty `running` run with timestamps and installed package versions merged with supplied settings. |
| `create_run_directory(output_dir="results")` | Creates a unique writable `run-…` directory and returns its `Path`. |
| `save_evaluation(run, path)` | Updates `run["updated_at"]`, validates the record, writes JSON atomically, and returns the path. The caller sets run status. |
| `load_evaluation(path)` | Loads and validates saved JSON; returns the run dictionary without reading sources or creating a judge. |
| `export_reports(results_json, output_dir=None)` | Loads the saved run and writes `report.xlsx`/`report.html`; returns the destination directory. The default is the JSON file's parent. |

The orchestration helper is also available as
`await meeting_eval.cli.evaluate_batch(transcript_dir, summary_dir, *, model=None,
business_context=None, n=None, questions_dir="results/questions",
regenerate_questions=False, output_dir="results")`. It returns the saved `results.json`
path after final reporting. Its question directory is independent of its output directory.

For question preparation, count precedence is explicit `n`, otherwise the valid existing
set's count, otherwise five. This also applies to explicit regeneration. An unreadable or
invalid existing set supplies no usable previous count. A supplied `n` must be a positive
integer; booleans are rejected. Questions must be nonblank strings and distinct after
stripping whitespace and case-folding; their original text and order are retained.

## Configure a LangChain model

This example runs in the notebook or another asynchronous context. It reuses one explicit
`ChatOpenAI` instance for preparation and evaluation:

```python
import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from meeting_eval import evaluate_meeting, get_or_prepare_questions

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-6-luna"),
    base_url=(os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
              or "https://api.openai.com/v1"),
    timeout=120,
    max_retries=2,
)
transcript = Path("examples/data/transcripts/meeting_001.txt").read_text(encoding="utf-8-sig")
summary = Path("examples/data/summaries/meeting_001.md").read_text(encoding="utf-8-sig")
context = Path("examples/business-context.md").read_text(encoding="utf-8-sig")
question_set, generated = await get_or_prepare_questions(
    "meeting_001", transcript, business_context=context, model=model,
)
result = await evaluate_meeting(
    "meeting_001", transcript, summary, question_set=question_set, model=model,
    question_generation_performed=generated,
)
```

Passing `model=None` selects the same default configuration lazily; passing a string
changes the model name. Other configured LangChain chat models can be injected when they
support Pydantic structured output with `include_raw=True`. Their provider integration
must be installed in the environment. The returned result still needs to be added to a
run and saved; the notebook demonstrates that step.

This report-only example makes no model calls and constructs no model client. Choose an
existing run file:

```python
from pathlib import Path
from meeting_eval import export_reports, load_evaluation

results_json = Path("results/<existing-run-directory>/results.json")
run = load_evaluation(results_json)
for meeting in run["results"]:
    print(meeting["meeting_id"], meeting["status"], meeting["scores"])

report_directory = export_reports(results_json, output_dir="results/rebuilt-reports")
print(report_directory / "report.xlsx")
print(report_directory / "report.html")
```

## Question artifacts and identity

Each question artifact is stored as `<questions_dir>/<meeting_id>.json`. Meeting IDs must
be nonblank filename components, excluding `.`, `..`, path separators, and NUL.
Schema version 1 contains these exact top-level fields:

| Field | Value |
| --- | --- |
| `schema_version` | Integer `1`. |
| `meeting_id` | Identifier used for lookup and meeting matching. |
| `question_set_id` | Content fingerprint described below. |
| `transcript_sha256` | Hash of normalized transcript text. |
| `business_context` | Normalized audience/purpose text, or null. |
| `business_context_sha256` | Hash of that context, or null when context is omitted. |
| `n` | Positive question count, equal to `len(questions)`. |
| `prompt_sha256` | Fingerprint of the question-generation prompt implementation. |
| `questions` | Ordered list of distinct, nonblank strings. |
| `generated_at` | Timezone-aware ISO timestamp. |
| `generation` | `{model, endpoint, duration_seconds, cost}`; endpoint/cost may be null. |

`normalize_text` removes a leading U+FEFF BOM, converts CRLF to LF, and converts remaining CR
to LF. `text_hash` applies that normalization, encodes UTF-8, and returns the SHA-256 hex
digest. It does not trim other whitespace, change case, or normalize Unicode characters.
Folder inputs are decoded with `utf-8-sig` before normalization.

The prompt fingerprint hashes a sorted-key JSON object containing two rendered prompts:
`native` uses transcript placeholder `[TRANSCRIPT]`, `n=5`, and no context; `contextual`
uses the same transcript/count plus context placeholder `[BUSINESS CONTEXT]`. This fingerprints
both native DeepEval instructions and this project's audience-prioritization instructions
at those canonical inputs.
Serialization uses `ensure_ascii=False` and the normal `json.dumps` separators.

The question-set ID hashes sorted-key JSON containing exactly:
`transcript_sha256`, `business_context_sha256`, `prompt_sha256`, `n`, and `questions`.
That serialization uses `ensure_ascii=False` and compact separators `(",", ":")` before
`text_hash`. Question wording and order therefore affect the ID. Meeting ID, model, endpoint,
generation time/cost, and summary text are excluded. The meeting ID is checked separately.
A changed evaluation judge can reuse a matching question set.

Preparation compares a valid saved set against the current transcript, context, prompt
fingerprint, and explicitly requested count. A mismatch or invalid artifact raises an error
until `regenerate=True` is supplied. A missing artifact is generated once. Explicit
regeneration replaces it only after a valid response with exactly the requested number of
questions; a generation failure preserves the prior artifact.

`evaluate_meeting` validates the supplied snapshot and matches its meeting ID/transcript
hash. Current context/template/count matching belongs to preparation; evaluation does not
infer those settings. Historical snapshots remain usable for report rebuilding after the
current prompt changes because export validates their stored contents, not current templates.

## Saved evaluation schema

A run has `schema_version: 1`, `created_at`, `updated_at`, `status`, `settings`, and `results`.
Run status is `running`, `completed`, or `interrupted`. Settings include `deepeval_version`
and `app_version`; the batch helper also records `business_context`. Run completion describes
execution and can include meeting errors.

Each meeting record contains:

| Field | Value |
| --- | --- |
| `meeting_id`, `status` | Identifier; `unprocessed`, `completed`, or `error`. |
| `inputs` | `transcript` and `summary`, each `{path, sha256}`; unknown values are null. Recorded source paths are resolved by the evaluation helper. |
| `question_set` | Full saved question snapshot, or null if unavailable. |
| `question_generation_performed` | Whether this run generated that snapshot. |
| `scores` | `{alignment, coverage, overall}`, or null for incomplete/error rows. |
| `counts` | `{alignment_supported, alignment_total, coverage_covered, coverage_eligible, coverage_total}`, or null. |
| `alignment` | Ordered `{claim_id, claim, verdict, reason}` rows; IDs `C001`, `C002`, …; verdict `yes`, `no`, or `borderline`. |
| `coverage` | Ordered `{question_id, question, transcript_answer, summary_answer, in_denominator, outcome}` rows; IDs `Q001`, `Q002`, …; answers `yes`/`no`. |
| `truths` | Native extracted transcript statements. |
| `reason`, `error` | Native overall explanation or execution-error explanation; absent values are null. |
| `evaluation` | `{model, endpoint, started_at, duration_seconds, cost}`, or null if evaluation was not attempted. Model/endpoint/cost can be unavailable. |

Coverage outcomes are `excluded` when the transcript answer is `no`, `covered` for
transcript/summary `yes`/`yes`, and `missing` for `yes`/`no`. Only source-answerable questions
have `in_denominator: true`. Alignment counts only native `yes` verdicts as supported.
Both zero-denominator cases retain native score zero. Overall remains the native minimum.

Before completion, validation requires identical frozen question order, one coverage row per
question, one alignment verdict per extracted claim, and recognized verdict values. Native
coverage answers are stripped and lowercased to `yes`/`no`. Scores must be finite numbers in
`[0, 1]` and match complete verdict arithmetic within relative/absolute tolerance `1e-9`.
Saved-file validation also checks provenance, counts, IDs, statuses, and detail types.
Incomplete/error rows have null scores/counts and empty verdict arrays. No list mismatch is silently
truncated into a partial score. Structural validation does not establish semantic judge accuracy.

## Persistence and failure behavior

The batch helper creates a unique run directory and probes question-directory writability
before model calls. It saves the initial `running` record with every meeting `unprocessed`.
For each valid pair, it reads/hashes sources, prepares or reuses questions, saves the question
snapshot on the still-unprocessed row, evaluates, then saves the completed/error row.
Pairing/read/generation errors are associated with that meeting; later meetings continue.

On `KeyboardInterrupt` or asynchronous cancellation during the loop, the run becomes
`interrupted`, is saved, and report export is attempted before the interruption is re-raised.
Already saved results remain available; active or later meetings can remain `unprocessed`.
An in-flight model call may have incurred spending without a completed result. This is
checkpoint preservation, not an automatic resume implementation. A normal finish marks the
run `completed`, saves it, then exports reports.

JSON persistence serializes with `allow_nan=False`, writes a temporary file in the target
directory, flushes and `fsync`s it, then replaces the destination. Atomic replacement applies
to that individual file. It is not a transaction across question artifacts and run records.

Reports validate and load saved JSON, build both output files in a temporary destination
subdirectory, then replace Excel and HTML individually. A build failure preserves existing
reports; the two final replacements are not a cross-file transaction. The canonical JSON is
not changed, and an export failure can be retried using the report-only command.

Excel uses four sheets: Scores, Coverage, Alignment, and Run details. It retains numeric
scores/counts, makes source/judge strings literal cells, and uses LF display line breaks.
Unsupported control characters or cells exceeding Excel's 32,767-character limit raise a
readable error instead of silently truncating saved details. HTML escapes external text,
shows missing questions and non-`yes` claims first, and expands full details/provenance.
Neither report adds per-question reasoning or evidence absent from native results.

Provider exception messages are withheld because they can contain sources or credentials.
Generation raises a generic failure; evaluation records retain the exception class with a
generic failure explanation. Model provenance uses public `model_name`, `model`, or
`deployment_name` properties, then the chat model's class name. Endpoint provenance reads
known public endpoint properties or the explicitly configured default-model endpoint;
it does not serialize the model configuration. Sanitization removes user information,
query parameters, and fragments while retaining the scheme, host/port, and path. An
injected model's endpoint can be unavailable.

New LangChain preparation and evaluation calls leave USD `cost` values null. Standard
LangChain raw-message usage metadata describes tokens, not currency; the adapter does
not infer a dollar price or add token-tracking fields. Nullable cost fields remain part
of the saved schema, and reports render numeric observations already present in a saved
record. A reused question set's generation cost remains historical provenance. These
fields are not a billing reconciliation.

## Verification

Run `.venv/bin/python -m pytest` for offline checks. Tests use scripted LangChain chat models and
block socket connections. They cover question reuse/staleness and failed regeneration,
native arithmetic and malformed responses, safe errors/provenance, checkpointed interruption,
saved-data-only report rebuilding, Excel/HTML text safety, and notebook execution from both
supported working directories. Notebook tests also rebuild reports in a fresh namespace
after deleting source files and question artifacts.

[CI](../.github/workflows/tests.yml) runs on Python 3.11 and 3.13, installs the development
extra, runs `pip check`, and runs pytest. These checks establish pipeline behavior without
hosted judge calls; human review of real judgments remains separate.
