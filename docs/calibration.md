# Calibrating judge and producer changes

The [fixture manifest](../calibration/fixtures.json) contains fictional boundary cases. Each records a capability, source pair, optional context, design expectation, and **pending** human review. Expectations guide inspection; they are not measured accuracy or fabricated subject-matter approval.

Run `python scripts/calibrate.py` to validate the manifest offline. Repeated real trials require explicit `--run --model MODEL --trials N`. Use `--case ID` to restrict a deliberate trial. The runner evaluates fixed summaries; it does not generate summaries or modify a producer prompt.

Each run preserves detailed artifacts, a manifest fingerprint, trial settings, per-case score ranges and review-flag counts, and a source-only `review_cases.json` and separate `human_annotations.csv`. Repeated trials count as separate input evaluations, not a population prevalence estimate. API/schema errors remain execution failures. Running a model never promotes a case to human-reviewed.

Judge configuration failures stop further requests. The summary retains every selected case's requested, attempted, and unrun trial counts; unrun observations do not become fabricated scores. Safe run-level judge and evaluator provenance is retained for later compatible comparisons.

## Human review

1. Assemble representative real pairs with approved access and retention. Include good paraphrases, subtle unit/timeframe/negation errors, central omissions, attribution mistakes, derived ideas, unsupported advice, context-only facts, unanswered questions, corrections, and long or noisy transcripts.
2. Have subject-matter reviewers assess sources and summaries before seeing judge scores or draft design expectations; share only `review_cases.json` and the blank annotation template for a blind review. Annotate material issues, severity, review need, supported alternatives, and uncertainty. Record reviewer identity or an approved pseudonym and source version.
3. Adjudicate disagreements. Preserve initial annotations; do not replace them with model findings. Mark unresolved cases uncertain rather than manufacture a binary label.
4. Compare the judge against adjudicated material-issue and review labels. Report false positives and missed material issues with denominators and examples. Inspect numeric, omission, attribution, and recommendation cases separately where sample size permits.
5. Reserve a holdout that does not drive prompt edits. Record prompt, judge configuration, methodology, and input fingerprints. Do not claim calibrated performance from teaching fixtures or model agreement.

Annotation fields remain blank or `pending`. Keep completed annotations separately from generated output so reruns cannot overwrite human work.

## Prompt comparisons

Change one focused producer instruction at a time. Generate variants on the same approved meetings and evaluate with the same judge, methodology, and business context. Compare paired cases, including new and removed material findings. Score gains can coexist with consequential regressions. Repeat noisy cases and preserve infrastructure failures in the execution report.

Prefer concrete repairs: preserve corrected figures, preserve units and timeframes, label derived ideas, retain attribution where needed, or state that a question was unanswered. Promote changes after human review of improvements and regressions, not merely because the evaluator suggested them. Revise judge prompts when independent evidence shows systematic false rejection or missed errors.
