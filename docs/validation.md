# Validation record

## Current policy: methodology 3

The repository uses generic institutional attribution and six-section evaluation criteria,
separately supplied context, and labeled source-grounded outreach suggestions. Summaries
are supplied as inputs. Compare runs only when their recorded evaluation conditions match;
earlier methodology scores are not a comparison baseline for methodology 3.

On October 29, 2026, a fresh Python 3.13 environment installed the project with its
development and notebook extras and current compatible dependencies. **121 offline tests
passed**, including early notebook checks for missing or outdated DeepEval installations.
`pip check` found no broken requirements, and importing `SingleTurnParams` succeeded.
The tested versions included DeepEval **4.3.2**, `langchain-openai` **1.7.3**,
`langchain-core` **1.7.1**, `langsmith` **0.14.15**, OpenAI **3.41.0**, and
Pydantic **2.14.2**. The calibration manifest validated all **20 fictional cases**,
with human reviews still pending.
All three offline learning notebooks executed successfully with outgoing connections
disabled inside their kernels. The summary-comparison notebook used supplied summaries;
no live model calls were made.

Run the offline checks with:

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/calibrate.py
.venv/bin/python scripts/run_learning_notebooks.py
```

## Historical verification: October 28, 2026, methodology 2

The recorded verification used offline controlled judges and HTTP transports. It covers metric
execution, faithfulness prompts, source identity, quote verification, structure rules,
concurrency, client cleanup, error isolation, provenance, CSV/HTML safety, comparisons,
and the repeat-trial calibration workflow. **118 offline tests passed** on Python 3.13.
The complete pipeline test used the actual LangChain client, OpenAI SDK, and DeepEval
through simulated HTTP transports: two concurrent meetings, separate context, two saved
runs, 24 simulated provider invocations, isolated usage totals, verified context quotes,
and a paired comparison. Run the complete suite with:

```bash
.venv/bin/python -m pytest
```

The package was reinstalled locally as version 0.2.0 without fetching dependencies.
`pip check` reported no broken requirements. Both CLI help commands worked. A wheel
built successfully; package resources and provenance calculation were checked
from an extracted installation. `git diff --check` reported no whitespace errors.

The calibration workflow's nine focused tests passed, including preservation of supplied
context and failed metrics, blank human annotations, safe run provenance, and stopping
further requests after a judge configuration failure. `python scripts/calibrate.py`
validated all **20 fictional cases**, with all **20 human reviews pending**, without model
calls. These are draft design expectations, not calibrated accuracy evidence.

The live single-meeting and batch notebooks were validated as notebook documents, their code cells checked for syntax,
and all saved outputs and execution counts removed. Their live execution was **not**
repeated for methodology 2. The three teaching summaries satisfied the deterministic
structure checks at that revision; this says nothing about their semantic accuracy. Two examples deliberately
contain omissions or distortions despite valid structure.

The offline CI workflow repeats the suite, manifest validation, and clean-notebook checks
on Python 3.11 and 3.13. A configured CI workflow is not evidence that a remote CI run has
already succeeded.

## Historical offline learning path and prompt review

The foundations tutorial, DeepEval metric lab, and prompt workbench executed top-to-bottom
through local Jupyter kernels using the project's Python environment. Outgoing Python
socket connections were disabled inside those kernels. The DeepEval lab used actual
metric classes with explicitly authored judge responses; no hosted model was called.

Executed notebooks and HTML copies were saved under
`results/learning/20261029T021933Z-20370b65/`. All code cells completed without error.
Output tables and links back to lessons, documentation, and source files were checked
programmatically. Browser visual inspection of local HTML was unavailable because the
browser policy blocked local-file URLs. Source notebooks retain no execution outputs.
Optional live model cells in the prompt workbench were not executed.

The new `meeting-eval-prompts` command and prompt bundle export were exercised without
initializing a model. Exported file hashes were verified. The tutorial runner has its own
offline CI job; that configuration is not evidence of an already completed remote run.

## Historical integration evidence: methodology 1

Before this revision, the initial implementation passed 45 offline tests on Python 3.13
with DeepEval 4.3.2 and langchain-openai 1.7.3. A live schema probe and both original notebooks
ran through the then-configured endpoint. The old three-pair batch distinguished the
faithful, omission-heavy, and materially distorted fixtures; a faithful fixture varied
between 1.0 and approximately 0.93 faithfulness across runs.

Those runs used the old four-section producer assumptions and explicit-discussion-only
sales policy. Their results, notebook outputs, and local report paths are historical
integration evidence only. They do not validate the current six-section contract,
provided-context handling, or generated-outreach policy, and must not be used as the
baseline for a current-methodology comparison. Old saved notebook outputs were removed
to prevent their interpretation as current results.

## Remaining validation

No new billable judge run was performed for this revision. When deliberately scheduled,
first verify endpoint/schema compatibility, then inspect the fictional boundary cases and
repeat noisy cases. Preserve source, prompt, policy, and judge fingerprints with each run.

Before operational use, independently review representative real transcript/summary pairs,
adjudicate disagreement, inspect false positives and missed material errors, and reserve
a holdout. Follow the [calibration workflow](calibration.md). Neither synthetic fixtures,
model agreement, nor a high score establishes acceptable accuracy on real meetings.
