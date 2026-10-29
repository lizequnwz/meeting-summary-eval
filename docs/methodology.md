# Evaluation methodology 3

## Contract and source boundaries

The evaluator measures whether a capital markets summary serves sales professionals while preserving source meaning. The [evaluation policy](../src/meeting_eval/policy.py) defines source boundaries and the optional format checks: Executive Briefing; Economy & Market Updates; Advisor Outreach & Sales Actions; Market & Institutional Perspectives; Questions and Answers; Search Tags. Supply an existing summary for evaluation.

The transcript establishes what was said, including speaker roles, uncertainty, and explicit corrections. Optional supplied context supports additional facts when explicitly provided to the producer. A context-only fact is not evidence that a speaker discussed it. Neither source authorizes repairs using external market knowledge. Ambiguous or corrupted material requires uncertainty.

## Faithfulness

**Question:** Does each substantive claim preserve support, units, timeframe, direction, certainty, and attribution?

DeepEval's `FaithfulnessMetric` extracts source statements and summary claims, judges support, and calculates a supported-claim ratio. Ambiguity is penalized. Meeting-specific extraction and verdict instructions adapt the existing algorithm. No artificial truth-extraction limit is imposed. Empty summaries are invalid; no assessable claims means unscorable rather than perfect.

A score of 1 means all *extracted* claims were judged supported. Extraction and judgment can fail. Correct passages can dilute one severe error, so diagnostics remain independent of the ratio. Omissions affect faithfulness when remaining wording misrepresents the source, rather than merely because content was omitted.

Outreach has two different meanings:

- `Discussed action:` asserts that a source explicitly proposed the action; check the action and attribution.
- `Suggested outreach:` is a generated, relevant conversation idea grounded in supported premises. It need not have been proposed at the meeting. It cannot invent facts, products, promises, owners, deadlines, or institutional endorsements.

A suggestion must not turn a tentative outlook into an endorsed recommendation. Attributed forecasts describe what a speaker said, not already verified future outcomes. A participant question is not an answer. Employment alone does not make an individual's view an official institutional position. Preserve narrow attribution when it affects meaning, rather than requiring names on every bullet.

## Important-information coverage

**Question:** Does consequential meeting information survive at useful compression?

A fixed-step DeepEval `GEval` reads evaluation criteria, sources, and summary. It prioritizes central market and macro developments, significant figures/timeframes, institutional perspectives and qualifications, advisor implications, and substantive Q&A. Small talk, repetition, incidental details, and topics never discussed create no coverage requirement. Context-only facts create no requirement to insert them into the meeting account. Tags cannot replace explanatory coverage.

| Raw score | Meaning |
| --- | --- |
| 0–2 | Almost all consequential information is absent. |
| 3–5 | Several important insights or a central perspective are missing. |
| 6–7 | Main themes survive, with an important gap or qualification missing. |
| 8–9 | Consequential themes and qualifications survive; only minor gaps remain. |
| 10 | Important meaning and necessary qualifications survive with appropriate emphasis. |

Normalize the emitted score by ten. The adapter supplies no token log probabilities, so scoring uses the emitted value rather than probability weighting. This is a holistic rubric judgment, not a measured fraction of source topics. Unsupported additions earn no coverage credit. Headings and prose style have separate checks.

## Structure and clarity

Both are optional; the default remains faithfulness plus coverage. Structure evaluates exact headings and the format limits in the evaluation policy. Executive Briefing allows one to three bullets, no more than 30 words each; prefer two or three only when supported. If there are no substantive takeaways, use `No supported items.` instead of inventing a bullet. Economy and Outreach permit at most four bullets each, one or two sentences per bullet. Perspectives permits at most five. Use `No supported items.` when sections 2–4 have no supported items. Tags are a single comma-separated line of unique Title Case topics, at most ten; do not invent tags to reach a minimum. Semantic support remains the judge's responsibility.

Substantive questions may occur outside a formal Q&A segment. Preserve actual answers and state `No answer was provided.` for unanswered questions. When no substantive questions occurred, use `No substantive Q&A occurred.` Uncertain applicability remains visible rather than silently manufacturing an answer.

Clarity uses its own rubric: raw 0–2 for severely unusable prose, 3–5 for substantial ambiguity/repetition, 6–7 for usable writing needing editing, 8–9 for clear writing with small issues, and 10 for exceptionally readable, economical organization. It does not reward omitting necessary qualifications or grade source support, coverage, or exact headings.

## Findings, evidence, and review

A structured diagnostic call identifies consequential omissions, unsupported/contradictory claims, attribution errors, certainty inflation, and practical prompt improvements. High severity means an issue could materially change an advisor or client conversation. Lack of support is distinct from contradiction. Avoid duplicates and invented issues to fill categories.

Evidence identifies its `transcript`, `context`, or `summary` source. The application resets judge-supplied verification and offsets, then matches exact contiguous quotations. Only unique matches receive verified lines. Missing, repeated, or altered quotations stay unverified. Verification proves text provenance, not semantic support. Support-not-found claims can cite the summary without a source passage; absence is not directly quotable.

Review flags prioritize negative/ambiguous support verdicts, material findings, uncertainty, insufficient evidence, and incomplete execution. Completed execution does not mean an acceptable summary. Failed metrics are null; failed diagnostics do not imply a clean review. No composite score or automatic pass/fail threshold is calculated.

## Execution, comparability, and calibration

`ChatOpenAI` supplies structured output to fresh mutable DeepEval metrics for each meeting. Actual judge calls share a request limit. Retries do not silently switch models or truncate sources. Context-limit failures are execution errors. Trace upload is suppressed, while sources are sent to the configured judge endpoint.

Compare saved methodology versions, evaluator and policy fingerprints, source hashes, and safe effective judge settings. Package versions alone cannot identify a rubric revision. Keep inputs fixed, inspect failures independently, and repeat noisy cases to assess variation.

Offline tests establish execution and deterministic behavior, not semantic judge accuracy. The [calibration workflow](calibration.md) keeps synthetic design expectations separate from independent human judgments. Historical scores from earlier methodologies are integration evidence only and are not a comparison baseline for methodology 3.
