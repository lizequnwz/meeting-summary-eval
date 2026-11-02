# Evaluation methodology

The project evaluates an existing summary against its meeting transcript using DeepEval
**4.3.2** and its native `SummarizationMetric` in `eval_mode="llm"`. The metric produces
alignment, coverage, an overall score, and a reason. The application prepares and freezes
coverage questions, checks the completeness of the returned judgments, and saves their
details. It preserves DeepEval's scoring rules.

The configuration is in [metrics.py](../src/meeting_eval/metrics.py); response validation
and saved score explanations are in [evaluation.py](../src/meeting_eval/evaluation.py).

## What the judge evaluates

The transcript supplies evidence about what was said in the meeting. Business context
describes the audience and purpose. It helps select relevant transcript information for
coverage questions and supplies no factual evidence.

Native evaluation uses three different collections:

| Collection | Origin and purpose | Count |
| --- | --- | --- |
| Transcript statements, called `truths` by DeepEval | Extracted from the transcript; used to judge support for summary claims | Chosen by the model |
| Summary `claims` | Extracted from the summary; each receives an alignment verdict | `M`, chosen by the model |
| Assessment questions | Prepared from the transcript, optionally guided by business context; used for coverage | `n`, chosen when preparing the question set |

`n` does not control how many claims or transcript statements are extracted. A meeting
with five questions can produce ten summary claims and eighteen transcript statements.
Alignment uses the ten claims as its denominator; coverage uses the eligible questions.

DeepEval's name `truths` describes extracted source statements, not independently verified
facts. A summary can accurately report a mistaken statement from a meeting. This evaluation
measures support in the source; it does not establish external truth. Extraction can also
omit or alter material information before the subsequent judgment.

## Prepare questions once, then reuse them

On first preparation, the judge receives the transcript and DeepEval's native question
generation prompt. With business context, the application adds guidance to select
consequential information for the audience: material economic and market developments,
figures and horizons, investment perspectives, risk and qualifications, attribution,
discussed actions, product discussions, and substantive or unanswered Q&A. Only topics
present in the transcript are eligible; these priorities impose no required sections or
category quotas. Without business context, generation uses the stock native prompt.

Questions concern specific transcript facts. For example, the fictional
[meeting_001 transcript](../examples/data/transcripts/meeting_001.txt) supports questions
such as:

- Did short-term yields decline by 10 basis points during the week while longer-term
  yields stayed roughly unchanged?
- Did Jordan say that no specific rate-cut date was forecast and that policy would depend
  on incoming data?
- Did Sam ask about near-term liquidity needs before discussing exposure to interest-rate
  volatility?

These examples illustrate the intended question shape; they are not a prescribed bank or
a record of a live generation. The native prompt requests closed-ended questions whose
answers are supported by the transcript. The application requires exactly `n` nonblank,
distinct questions, but that structural check cannot prove their factual quality.

The question set is saved before summary evaluation and reused across summary versions
and reruns. Reuse checks the transcript, business context, generation prompt fingerprint,
and any explicitly requested question count. Changes require explicit regeneration;
corrupt artifacts also require explicit regeneration. Changing only the summary or the
evaluation judge leaves the saved questions in place. Question preparation and artifact
validation are implemented in [questions.py](../src/meeting_eval/questions.py).

This produces a stable, transcript-specific coverage sample. Native coverage judges the
transcript and summary separately for each question, so questions must stand on their own.
An instruction such as “Does the summary preserve every important fact in the transcript?”
requires a comparative judgment and is unsuitable for this native question interface.

## Alignment: support for extracted summary claims

For every extracted summary claim, the native judge returns one verdict:

| Verdict | Native interpretation | Effect on alignment |
| --- | --- | --- |
| `yes` | Agrees with the source | Counts as supported |
| `no` | Directly contradicts the source | Remains in the denominator; does not count as supported |
| `borderline` | Lacks support or concerns information absent from the source | Remains in the denominator; does not count as supported |

Alignment is the number of `yes` verdicts divided by `M`, the total number of extracted
summary claims. With no extracted claims, native alignment is **0**. This is an unweighted
fraction: each extracted claim contributes one verdict, regardless of its business impact.

Read the claim and its verdict reason together. Native alignment can penalize derived
outreach ideas absent from the transcript even when the summary labels them as suggestions.
The application applies no special exception for them. Business context affects question
preparation only; it does not change claim extraction, alignment prompts, or the scoring
calculation.

## Coverage: answerability of the frozen questions

The judge assesses whether the transcript contains enough information to answer each
question, and separately whether the summary does. The saved `transcript_answer` and
`summary_answer` values are these **answerability judgments**, not extracted factual
answers or evidence quotations.

| Transcript answerable | Summary answerable | Saved outcome | Included in coverage denominator? |
| --- | --- | --- | --- |
| `yes` | `yes` | `covered` | Yes |
| `yes` | `no` | `missing` | Yes |
| `no` | Either | `excluded` | No |

Coverage is `covered / eligible`, where eligible questions are those with transcript
answerability `yes`. If none are eligible, native coverage is **0**. The application keeps
excluded questions visible so the denominator remains inspectable.

Coverage measures the preservation of sampled, source-answerable information. A score of
0.75 does not mean that 75% of every important fact, sentence, or topic was retained.
Answerability alone also does not establish that the summary's answer is correct; alignment
provides the separate check on support for its extracted claims.

## Overall score and a worked example

The overall score is DeepEval's **minimum of alignment and coverage**. It is not their
average. Consider a hypothetical completed evaluation:

| Measure | Native judgments | Calculation | Score |
| --- | --- | --- | --- |
| Alignment | 8 `yes`, 1 `no`, 1 `borderline` across 10 claims | `8 / 10` | 0.80 |
| Coverage | 3 covered, 1 missing, 1 excluded across 5 questions | `3 / 4` | 0.75 |
| Overall | The two scores above | `min(0.80, 0.75)` | 0.75 |

The excluded question contributes to neither side of the coverage fraction. Its presence
does not change the alignment denominator. A zero denominator produces the native zero
score described above and therefore an overall score of zero.

The metric uses `threshold=None` and returns scores without an acceptance gate. A completed
evaluation can have low or zero scores. An execution or response-integrity error instead
has empty scores; it is not converted into a low-quality score.

## Completeness and integrity checks

Before a response is accepted as completed, the application checks that:

- The evaluated questions exactly match the frozen list and its order.
- Every extracted claim has one alignment verdict, and every question has one coverage
  verdict.
- Alignment verdicts are `yes`, `no`, or `borderline`; both answerability verdicts are
  `yes` or `no` after whitespace and case normalization.
- Scores are finite numbers in the range 0–1 and agree with the complete native verdict
  fractions and their minimum.

These checks matter because a response can satisfy the native structured-output schema
while returning too few answers or verdicts. Such a response is recorded as an error with
empty scores. The fraction checks verify native results; they do not introduce an
alternative scoring method.

Saved question sets carry content identifiers and provenance hashes. Saved evaluations
include a snapshot of their question set, input hashes, verdicts, counts, and scores. Loading
an evaluation validates their consistency before export. Excel and HTML are rendered from
the same saved evaluation and make no new judgments.

The offline tests in [test_metrics.py](../tests/test_metrics.py),
[test_questions.py](../tests/test_questions.py), and
[test_evaluation.py](../tests/test_evaluation.py) exercise native integration, reuse and
invalidation, incomplete responses, excluded questions, zero denominators, and inconsistent
saved records. Their scripted responses verify pipeline behavior and arithmetic; they do
not establish the hosted judge's accuracy.

## Variability that remains

Freezing questions removes repeated question generation and keeps the sampled questions
constant. Every evaluation still extracts transcript statements and summary claims, judges
alignment, reassesses both texts' answerability, and generates a reason. Those model
responses can vary. Even the eligible coverage denominator can change if the judge changes
its transcript-answerability verdicts for the same questions.

Use the saved question-set identifier and judge provenance when comparing evaluations,
and inspect the underlying claims, verdicts, and counts alongside score changes. The
reported reason is a model-generated explanation, not a calibrated confidence estimate.
