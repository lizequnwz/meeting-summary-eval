# Validation record — October 28, 2026

## Current policy: template 4.1 and methodology 2

The repository now uses the six-section producer contract, separately supplied context,
and labeled source-grounded outreach suggestions. These change evaluation meaning and
invalidate the original methodology's scores as a comparison baseline.

Current verification uses offline controlled judges and HTTP transports. It covers metric
execution, faithfulness prompts, source identity, quote verification, structure rules,
concurrency, client cleanup, error isolation, provenance, CSV/HTML safety, comparisons,
and the repeat-trial calibration workflow. **112 offline tests passed** on Python 3.13.
The complete pipeline test used the actual LangChain client, OpenAI SDK, and DeepEval
through simulated HTTP transports: two concurrent meetings, separate context, two saved
runs, 24 simulated provider invocations, isolated usage totals, verified context quotes,
and a paired comparison. Run the complete suite with:

```bash
.venv/bin/python -m pytest
```

The package was reinstalled locally as version 0.2.0 without fetching dependencies.
`pip check` reported no broken requirements. Both CLI help commands worked. A wheel
built successfully; its packaged v4.1 prompt and provenance calculation were checked
from an extracted installation. `git diff --check` reported no whitespace errors.

The calibration workflow's nine focused tests passed, including preservation of supplied
context and failed metrics, blank human annotations, safe run provenance, and stopping
further requests after a judge configuration failure. `python scripts/calibrate.py`
validated all **20 fictional cases**, with all **20 human reviews pending**, without model
calls. These are draft design expectations, not calibrated accuracy evidence.

Both notebooks were validated as notebook documents, their code cells checked for syntax,
and all saved outputs and execution counts removed. Their live execution was **not**
repeated for methodology 2. The three teaching summaries satisfy the current deterministic
structure checks; this says nothing about their semantic accuracy. Two examples deliberately
contain omissions or distortions despite valid structure.

The offline CI workflow repeats the suite, manifest validation, and clean-notebook checks
on Python 3.11 and 3.13. A configured CI workflow is not evidence that a remote CI run has
already succeeded.

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
baseline for a methodology-2 prompt comparison. Old saved notebook outputs were removed
to prevent their interpretation as current results.

## Remaining validation

No new billable judge run was performed for this revision. When deliberately scheduled,
first verify endpoint/schema compatibility, then inspect the fictional boundary cases and
repeat noisy cases. Preserve source, prompt, policy, and judge fingerprints with each run.

Before operational use, independently review representative real transcript/summary pairs,
adjudicate disagreement, inspect false positives and missed material errors, and reserve
a holdout. Follow the [calibration workflow](calibration.md). Neither synthetic fixtures,
model agreement, nor a high score establishes acceptable accuracy on Fidelity meetings.
