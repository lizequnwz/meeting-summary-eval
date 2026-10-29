# Evaluation methodology 4

## Generic policy and optional business context

The [evaluation policy](../src/meeting_eval/policy.py) applies to meeting summaries in any format. It evaluates source support, preservation of important meaning, and optional clarity and conciseness. It does not grade headings, section order, bullet counts, word quotas, or indexing metadata. There is no structure metric or required summary template.

One optional `business_context` block describes the audience, purpose, and kinds of information that matter to readers. The [example business context](../examples/business-context.md) centralizes the existing Zoom workflow's goals. Replace that block to evaluate another meeting type, or omit it for generic evaluation. Business context can guide importance and readability, but cannot establish facts, override source authority, or introduce format requirements.

The transcript establishes what was said, including speaker roles, uncertainty, and explicit corrections. Optional `provided_context` is a separate factual source that was also supplied to the summary producer. Its facts cannot establish that a speaker discussed or endorsed them. Missing information is not repaired with outside knowledge. Ambiguous or corrupted material requires uncertainty.

## Faithfulness

**Question:** Does each substantive claim preserve support, quantities, timeframe, certainty, and attribution?

DeepEval's `FaithfulnessMetric` extracts source statements and summary claims, judges support, and calculates a supported-claim ratio. Ambiguity is penalized. Generic meeting-specific extraction and verdict instructions adapt the existing algorithm. No artificial truth-extraction limit is imposed. Empty summaries are invalid; no assessable claims means unscorable rather than perfect.

A score of 1 means all *extracted* claims were judged supported. Extraction and judgment can fail. Correct passages can dilute one severe error, so diagnostics remain independent of the ratio. Omissions affect faithfulness when the remaining wording misrepresents the source, rather than merely because content was omitted.

Distinguish claims about what was discussed or agreed from clearly presented derived ideas. A proposed follow-up need not have been stated verbatim in the meeting when it is clearly an original suggestion. Its factual premises still require support, and it must not invent decisions, endorsements, promises, owners, or deadlines. This distinction depends on meaning; no exact label is required.

An attributed opinion or forecast records what a speaker said, not an already verified future outcome. A question is not an answer or an endorsed position. Preserve attribution when it affects meaning, without requiring names on every sentence.

## Important-information coverage

**Question:** Does consequential meeting information survive at useful compression?

A fixed-step DeepEval `GEval` reads the generic evaluation criteria, optional business context, sources, and summary. It identifies central themes, decisions, outcomes, actions, unresolved issues, risks, and qualifications according to the meeting's emphasis and consequences. The business context can clarify which details matter to the intended audience. It cannot create requirements for topics absent from the meeting.

Small talk, repetition, incidental details, and topics never discussed create no coverage requirement. Context-only facts create no requirement to insert them into the meeting account. Faithful paraphrases and equivalent meaning count regardless of location or layout. Unsupported additions earn no coverage credit.

| Raw score | Meaning |
| --- | --- |
| 0–2 | Almost all consequential information is absent. |
| 3–5 | Several important insights or a central perspective are missing. |
| 6–7 | Main themes survive, with an important gap or qualification missing. |
| 8–9 | Consequential themes and qualifications survive; only minor gaps remain. |
| 10 | Important meaning and necessary qualifications survive with appropriate emphasis. |

Normalize the emitted score by ten. The adapter supplies no token log probabilities, so scoring uses the emitted value rather than probability weighting. This is a holistic rubric judgment, not a measured fraction of source sentences or topics. Coverage does not grade formatting or prose style.

## Clarity and conciseness

Optional `clarity` scoring assesses readable, economical writing for the intended audience, using business context when supplied. It considers ambiguous wording, unexplained jargon, unnecessary repetition, and logical flow. It does not impose headings, a particular layout, a length target, or fixed section requirements. Brevity that removes necessary qualifications or attribution is not an improvement.

Clarity has its own rubric: raw 0–2 for severely unusable prose, 3–5 for substantial ambiguity or repetition, 6–7 for usable writing needing editing, 8–9 for clear writing with small issues, and 10 for exceptionally readable, economical expression. It does not score factual support or coverage.

## Findings, evidence, and review

A structured diagnostic call identifies consequential omissions, unsupported or contradictory claims, attribution errors, certainty inflation, and useful prompt improvements. Severity follows the likely consequence for the meeting's purpose and readers. Lack of support is distinct from contradiction. Avoid duplicate or invented issues. Missing headings, omitted empty sections, and different presentation formats do not produce findings.

Evidence identifies its `transcript`, `context`, or `summary` source. The application resets judge-supplied verification and offsets, then matches exact contiguous quotations. Only unique matches receive verified lines. Missing, repeated, or altered quotations stay unverified. Verification proves text provenance, not semantic support. Support-not-found claims can cite the summary without a source passage; absence is not directly quotable. Business context is not an evidence source.

Review flags prioritize negative or ambiguous support verdicts, material findings, uncertainty, insufficient evidence, and incomplete execution. Completed execution does not mean an acceptable summary. Failed metrics are null; failed diagnostics do not imply a clean review. No composite score or automatic pass/fail threshold is calculated.

## Execution, comparability, and calibration

`ChatOpenAI` supplies structured output to fresh mutable DeepEval metrics for each meeting. Actual judge calls share a request limit. Retries do not silently switch models or truncate sources. Context-limit failures are execution errors. Trace upload is suppressed, while sources are sent to the configured judge endpoint.

Compare saved methodology versions, evaluator fingerprints, business-context identity, source hashes, and safe effective judge settings. Changing business context changes the evaluation conditions and requires a new baseline. Package versions alone cannot identify a rubric revision. Keep conditions fixed, inspect failures independently, and repeat noisy cases to assess variation.

Offline tests establish execution and deterministic behavior, not semantic judge accuracy. The [calibration workflow](calibration.md) keeps synthetic design expectations separate from independent human judgments. Scores from earlier methodologies are historical integration evidence and are not a comparison baseline for methodology 4.
