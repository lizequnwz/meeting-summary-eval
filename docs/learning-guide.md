# Learn, inspect, and use meeting-summary evaluation

Start by predicting whether a summary is accurate and complete yourself. Then inspect how
the software represents that judgment, and only then pay for a live judge. The notebooks
separate these steps so the first lessons need neither credentials nor model requests.

## Your learning path

| Order | Open | What you will do | Model calls |
| --- | --- | --- | --- |
| 1 | [00 — Evaluation foundations](../notebooks/00_evaluation_foundations.ipynb) | Label a tiny example, calculate support, distinguish omissions, severity, and execution errors. | None |
| 2 | [03 — DeepEval metric lab](../notebooks/03_deepeval_metric_lab.ipynb) | Run actual DeepEval metrics with clearly labeled prewritten judge responses; inspect claims, verdicts, questions, and rubrics. | None |
| 3 | [01 — Single meeting](../notebooks/01_single_meeting.ipynb) | Evaluate one fictional meeting with a real judge; inspect every claim and quotation. | Normally about 6; optional individual metrics add calls |
| 4 | [02 — Batch evaluation](../notebooks/02_batch_evaluation.ipynb) | Evaluate several meetings, inspect recurring issues, and open the saved report. | Normally about 6 per meeting |
| 5 | [04 — Prompt review and experiments](../notebooks/04_prompt_review_and_experiments.ipynb) | Review judge prompts and compare supplied summaries with fixed judging rules. | None by default; live evaluation requires explicit opt-in |

You can read lesson 04 before spending anything. The numbers preserve the existing
single-meeting and batch notebooks; the order above puts the conceptual lessons first.
Expect roughly 20–30 minutes for each of the first two lessons, then one focused practical
session. These are suggested study blocks, not measured completion times.

Install and start Jupyter from the repository root:

```bash
.venv/bin/python -m pip install -e '.[dev,notebooks]'
.venv/bin/jupyter lab notebooks
```

Use the project's Python environment as your kernel. If Jupyter offers the wrong one,
register it, then choose **Meeting Summary Eval**:

```bash
.venv/bin/python -m ipykernel install --user \
  --name meeting-summary-eval --display-name "Meeting Summary Eval"
```

Check
`sys.executable` in a notebook if an import fails. Do not paste an API key into a cell.

To execute just the offline lessons and save readable HTML previews:

```bash
.venv/bin/python scripts/run_learning_notebooks.py
# Or select one:
.venv/bin/python scripts/run_learning_notebooks.py --lesson deepeval
```

The runner uses the same Python interpreter as the command and writes a new directory
under ignored `results/learning/`. It adds an explicit offline guard inside each temporary
kernel and blocks outgoing Python socket connections. It leaves source notebooks clean.
Open the saved `.ipynb` or `.html` to read the worked outputs without rerunning cells.
Use ordinary Jupyter, rather than this runner, for deliberate live experiments.

## Does DeepEval require a list of questions?

**No. DeepEval is a toolkit of different metrics, not one universal question-based test.**

| Metric | Does it need your question list? | How it assesses this task |
| --- | --- | --- |
| `FaithfulnessMetric` | No | Extract summary claims and source statements, judge support, calculate a supported-claim ratio. |
| `GEval` for coverage | No | Apply the configured evaluation steps and importance rubric to the source and summary. |
| `SummarizationMetric` in LLM mode | Optional | Check alignment and question-based coverage; generate assessment questions if you do not supply them. |
| Package structure check | No | Parse headings, bullets, absence markers, and tags deterministically. |

Our default is **FaithfulnessMetric + GEval coverage + structured diagnostics**. The lab
also teaches SummarizationMetric as an alternative experiment; it is not silently added
to production scores. SummarizationMetric overlaps with both default dimensions, so a
third score should be adopted only if it adds a useful, independently checked signal.

For the installed DeepEval 4.3.2 LLM-mode SummarizationMetric:

- `assessment_questions` is optional. Without it, the default `n=5` requests generated questions.
- Questions should have yes/no answers and target specific important information in the source.
- Coverage counts questions answerable “yes” from the source whose answer is also “yes” from the summary, divided by source-“yes” questions. It does **not** reward both documents saying “no.”
- The overall score is the minimum of alignment and coverage. Check `score_breakdown` to see which dimension limited it.

That denominator matters. A question about a topic absent from the source does not test
whether the summary preserved an important meeting point. An incomplete question set can
also miss the very omission you care about. Write questions from the source before seeing
candidate summaries; review them for importance and keep them fixed across candidates.
Do not confuse real meeting Q&A with assessment questions or with `LLMTestCase.input`.
The lab demonstrates these distinctions using the actual installed implementation.
See [DeepEval Summarization](https://deepeval.com/docs/metrics-summarization),
[Faithfulness](https://deepeval.com/docs/metrics-faithfulness), and
[GEval](https://deepeval.com/docs/metrics-llm-evals).

## What goes into a DeepEval test case?

The correct field mapping depends on the metric. In this package:

| Field | Meaning |
| --- | --- |
| `input` | Evaluation instructions and format criteria. It does not have to be an English question. |
| `actual_output` | The generated meeting summary. |
| `retrieval_context` | Labeled full transcript and optional supplied context. This package has no retriever. |
| `expected_output` | Not required for these reference-free metrics. |

For a standalone SummarizationMetric, the source text goes in `input` and the summary in
`actual_output`. Reusing this package's test case unchanged would give it instructions
instead of the transcript as its source. The lab builds separate test cases deliberately.
Other DeepEval metrics may require expected answers or different inputs; check the
[test-case documentation](https://deepeval.com/docs/evaluation-test-cases) for that metric.

Reference-free scoring still needs independent validation. Human labels can describe
supported claims, important omissions, severity, and acceptable paraphrases without
requiring one canonical reference summary.

## Where should I review or edit the prompts?

Judge prompts tell this package how to evaluate a supplied summary. Review the shared
evaluation policy first, then the metric and diagnostic instructions. Keep those rules
fixed when comparing summaries so a judge change does not alter the benchmark.

| File | Role | When to edit |
| --- | --- | --- |
| [policy.py](../src/meeting_eval/policy.py) | Shared source, attribution, context, and outreach policy. | Deliberately change what counts as supported or acceptable. |
| [metrics.py](../src/meeting_eval/metrics.py) | Faithfulness extraction/verdict prompts, coverage steps/rubric, clarity rubric. | Correct demonstrated judge errors or redefine a scoring dimension. |
| [diagnostics.py](../src/meeting_eval/diagnostics.py) | Evidence-rich diagnostic prompt and deterministic structure checks. | Improve omission/severity findings, suggestions, evidence rules, or format checks. |
| [judge.py](../src/meeting_eval/judge.py) | System instructions, structured-output adapter, request limits, usage collection. | Change the model-call boundary, rather than editorial preferences. |

You can review effective prompts without opening Python files or making model requests:

```bash
.venv/bin/meeting-eval-prompts
.venv/bin/meeting-eval-prompts --show policy
.venv/bin/meeting-eval-prompts --show coverage
.venv/bin/meeting-eval-prompts --output results/prompt-review-001
```

The export contains the shared policy, system instruction, four
faithfulness-stage prompts, coverage/clarity steps and rubrics, diagnostic prompt, and a
fingerprint manifest. Source data are visible placeholders. DeepEval adds its own scoring
wrapper around the exported GEval steps; these are not stored private model thoughts.
Exports are **review copies**; editing them does not alter runtime instructions.

For an experiment, supply baseline and candidate summaries from the same sources and keep
the evaluation policy fixed. Lesson 04 records the inputs and compares compatible saved
runs. When changing evaluation criteria, update the methodology version, examples, and
related documentation together, and establish a new baseline. Restart a notebook kernel
after source prompt edits so imported constants do not remain stale.

## Use it on a real meeting

1. Read one transcript and its summary yourself. Note the main points, numbers,
   qualifiers, speaker attribution, and one error you would care about in a client conversation.
2. Save the exact producer inputs: transcript, summary, and any context the producer received.
   Start with the single-meeting notebook; do not tune a threshold on one case.
3. Inspect execution status first. A timeout or schema failure is an incomplete evaluation,
   not evidence that the summary is poor. Missing scores stay missing.
4. Read claims/verdicts, coverage reasoning, high-severity findings, and quotations. A matched
   quotation establishes its location, not the correctness of the judge's interpretation.
5. Disagree with the judge when the evidence warrants it. Record the disagreement and
   whether it was extraction loss, ambiguous source text, incorrect support, an omission,
   or a severity error. Add a focused regression case.
6. After several reviewed meetings, make one producer change. Generate baseline and
   candidate summaries on the same sources using the same summarizer configuration.
   Judge them under the same fixed policy and inspect paired results.
7. Repeat noisy cases and reserve examples that do not drive edits. Use the
   [calibration workflow](calibration.md) for independent review and missed-error analysis.

Obtain baseline and candidate summaries from the summarizer you intend to improve, then
supply them to lesson 04. This package evaluates those summaries; it does not generate them.

## Common surprises

- **A short summary is faithful but incomplete:** included claims can all be correct while central information is missing. Inspect coverage separately.
- **A high score still has a material error:** a supported-claim ratio can dilute one severe mistake. Read diagnostic findings.
- **The wording looks different but the meaning is right:** paraphrases should be accepted; lexical overlap is not the objective.
- **An outreach idea was never spoken:** labeled derived outreach can be valid; invented facts, endorsement, products, and guarantees remain unsupported.
- **A context fact becomes a meeting claim:** supplied background cannot establish that someone said or agreed to it in the meeting.
- **A structure score is perfect:** formatting does not establish faithfulness or coverage.
- **A repeated run changes slightly:** do not interpret one small difference as a producer improvement. Inspect the cases and repeat them.
- **An endpoint rejects a schema:** check model/structured-output support. Switching the score to zero would hide the infrastructure failure.

When ready to read the implementation, follow `policy.py → metrics.py → judge.py →
evaluator.py → diagnostics.py → reports.py`. Start with a real result beside the code.
The notebooks explain why those components exist before you need to understand every line.
