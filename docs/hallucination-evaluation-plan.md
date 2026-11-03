# Proposed claim-level contradiction evaluation

Research date: November 3, 2026. Status: research and rollout recommendation.
The [standalone contradiction metric](contradiction-metric.md) now implements the
claim-level check as a separate Python API. Calibration and integration into the CLI,
notebook workflow, and saved reports remain proposed work.
The repository pins and installs DeepEval 4.3.2. The implementation observations below
were checked against that installed version as well as official documentation and source.
No live judge calls or latency benchmarks were run.

## Recommendation

Use a custom DeepEval metric with **summary claim extraction followed by batched
full-transcript judgment**. This follows the user's preference for contradiction-focused
scoring while retaining inspectable detail for each summary claim.

1. Extract atomic factual claims from the summary, preserving attribution, uncertainty,
   negation, scope and the original summary passage for each claim.
2. Send all claims, their original passages and the complete transcript to one structured
   judge call. Return one verdict, explanation and relevant transcript passage per claim.
   Do not extract or summarize the transcript before judging.
3. Calculate the contradiction rate locally and fail the contradiction check if any claim
   is contradicted. Show unsupported claims separately for review without automatically
   counting them as contradictions.

This normally takes two judge calls, with no extra overall-reason call. It provides more
inspectable findings than a single whole-summary G-Eval judgment. It still requires
calibration: missed claims and missed contradictions remain possible.

Keep contradiction detection, support warnings and coverage separate. A summary can have
zero detected contradictions while containing unsupported facts or omitting useful content.

## What the current alignment score means

The current [metric factory](../src/meeting_eval/metrics.py) creates native
`SummarizationMetric` with frozen coverage questions, `eval_mode="llm"`, and
`threshold=None`. It has no acceptance gate. Alignment is supported extracted claims
divided by all extracted claims; both contradictions (`no`) and missing support
(`borderline`) reduce it. The overall score is the minimum of alignment and coverage.
See [Methodology](methodology.md).

A 98% alignment score therefore does not mean that 2% of summaries hallucinate, or that
there is a 98% probability that a summary is correct. Claim extraction determines the
denominator and gives each claim equal weight.

The native implementation compares extracted summary claims against extracted transcript
statements, rather than the complete transcript at the final alignment stage. Extraction
can lose supporting evidence. Existing documentation also notes that labeled outreach
suggestions can be penalized for being absent from the transcript. These are plausible
sources of false alarms; the saved flagged claims must be reviewed before concluding that
the reported 97–98% is too strict.

## How DeepEval's built-in HallucinationMetric works

In the installed 4.3.2 LLM mode:

1. The test case requires `input`, `actual_output`, and `context`. The verdict prompt
   actually compares `actual_output` with `context`; `input` is not included in that
   comparison. A transcript must go in `context=[transcript]`.
2. One model call returns one yes/no verdict and reason for each context document.
   All documents are batched in that call. There is no separate extraction of summary
   claims or source facts.
3. The prompt directs the judge to reject contradictions and forgive missing detail.
   **It does not explicitly require evidence for every extra factual assertion in the
   output.** An invented deadline absent from the transcript is therefore a weak point
   in this definition, even if a particular judge sometimes catches it.
4. The score is **aligned context documents / total context verdicts**. Higher is better
   in 4.3.2. Older examples using zero as the ideal score have the opposite direction.
   Strict mode sets the passing threshold to 1 and converts any imperfect result to 0.
5. The default overall reason takes a second call. `include_reason=False` removes that
   call while retaining reasons on the individual context verdicts.

With one full transcript as the sole context document, a complete valid response already
has a binary score. Splitting it into chunks changes the denominator: the resulting
fraction measures document consistency, not the fraction of invented summary claims.
For any pilot of this metric, validate that every supplied context has exactly one
verdict. The shared native scoring helper returns 1 for an empty verdict list.

Sources: [official HallucinationMetric documentation](https://deepeval.com/docs/metrics-hallucination),
[verdict prompt](https://github.com/confident-ai/deepeval/blob/main/deepeval/metrics/hallucination/templates/generate_verdicts.txt),
[implementation](https://github.com/confident-ai/deepeval/blob/main/deepeval/metrics/hallucination/hallucination.py).
The installed files under `deepeval/metrics/hallucination/` and
`deepeval/metrics/utils/qag.py` were the version-specific reference; upstream main can change.

## Comparison and expected runtime

| Approach | What it judges | Normal judge calls per summary | Fit for this requirement |
| --- | --- | ---: | --- |
| Current SummarizationMetric | Extracted-claim support and question-based coverage | 6 | Useful diagnostics; lossy extraction and no acceptance gate |
| Built-in HallucinationMetric, LLM mode | Contradiction against each context document | 2 with overall reason; 1 without | Simple baseline, but weaker definition for unsupported additions |
| Built-in HallucinationMetric separately for each extracted summary claim | Each claim against the full transcript | M + 1 without overall reasons, for M claims | Native option; repeatedly submits the full transcript and lacks a separate unsupported verdict |
| Custom strict G-Eval with fixed steps | Entire summary under an explicit policy | 1, including reason | Simpler whole-summary alternative; less structured detail |
| Custom claim extraction and batched transcript judgment | Each summary claim against the complete transcript | 2, including per-claim explanations | Recommended for the revised requirement |

Counts exclude retries and one-time question preparation. Current summarization overlaps
some calls asynchronously. Two calls instead of six is a plausible latency improvement,
not a measured speedup. Transcript length, claim count, output length, judge model and
retries still matter. Adding the proposed metric alongside the current evaluator
increases total work from six to eight calls; existing coverage is not accelerated.

Explicit `evaluation_steps` skip G-Eval's step-generation call. Strict mode requests 0/1
and sets the threshold to 1; it does not make the judge deterministic. Put the binary
policy in the steps, without a graded rubric. Both candidates can reuse
[LangChainJudge](../src/meeting_eval/judge.py); no new dependency is needed.
See [official G-Eval documentation](https://deepeval.com/docs/metrics-llm-evals) and
[implementation](https://github.com/confident-ai/deepeval/blob/main/deepeval/metrics/g_eval/g_eval.py).

DeepEval's `FaithfulnessMetric` supplies a closely related established pattern: summary
claim extraction and three-way claim verdicts. In installed 4.3.2,
`penalize_ambiguous_claims=False` counts supported and borderline claims as passing, so
strict mode rejects contradictions. However, its LLM verdict stage still uses extracted
source truths, rather than the complete source. It is not a drop-in match for this
requirement. Adopt its claim/verdict pattern through a small custom `BaseMetric`; avoid
overriding private extraction methods to turn them into transcript passthroughs.
See [Faithfulness documentation](https://deepeval.com/docs/metrics-faithfulness) and
[implementation](https://github.com/confident-ai/deepeval/blob/main/deepeval/metrics/faithfulness/faithfulness.py).

## Proposed claim verdicts and pass/fail policy

Use the transcript as the sole factual evidence. Business context may define the audience
but cannot supply missing meeting facts. Judge the source as a record of what was said,
without independently deciding whether a participant's external-world claims are true.

Use three verdicts rather than conflating a lack of evidence with contrary evidence:

| Verdict | Meaning | Effect |
| --- | --- | --- |
| Supported | The transcript supports the claim, including faithful paraphrase or synthesis | No contradiction; supported |
| Contradicted | The claim conflicts with transcript evidence | Fails the contradiction check |
| Unsupported | The transcript does not establish the claim and provides no clear contradiction | Visible review warning; does not automatically fail the contradiction check |

For example, a Friday deadline is contradicted if the transcript specifies Thursday;
it is unsupported if the transcript contains no deadline. Names, figures, dates, owners,
products, decisions and causal claims all receive the same distinction. Preserve
attribution, negation, scope and uncertainty. Converting an explicitly tentative proposal
into an approved decision can be a contradiction; inventing a decision where none is
discussed is unsupported. Omitting a topic affects coverage; omitting a qualifier can
change the meaning of an asserted claim and must be judged.

The summary-level contradiction check passes with zero contradicted claims. A passing
summary can still carry unsupported-claim warnings. Do not describe that outcome as
proof that every summary claim is grounded.

Clearly separated new outreach questions or suggested actions are not automatically
factual assertions about what occurred in the meeting. Extract and judge their factual
premises, preserving the suggestion framing. A suggestion label cannot excuse an
invented guarantee, product or attribution.

For every claim, preserve its identifier, original summary passage, verdict, concise
explanation and relevant transcript quotation or source location. Unsupported findings
must explain the missing evidence without fabricating a quotation. Validate quotations
against the source text. Conflicting speakers and later corrections need contextual
judgment; disagreement somewhere in the transcript is not sufficient to contradict an
accurately attributed statement. Ambiguous or incomplete evidence receives a review
warning rather than a confident supported label.

Execution errors and malformed responses remain separate from content verdicts. Require
exactly one verdict for every claim, with no missing, duplicate or unexpected identifiers.
An unexpectedly empty extraction must not yield an automatic pass. Require the complete
transcript to fit the judge's context; do not silently truncate it.

Illustrative calibration cases:

| Transcript situation and summary treatment | Expected outcome |
| --- | --- |
| Equivalent paraphrase of a supported point | Pass |
| Supported point included, another topic omitted | Factuality pass; assess omission through coverage |
| Clearly labeled new outreach question with supported premises | Pass under proposed suggestion policy |
| New deadline or decision that the transcript never mentions | Unsupported warning |
| Summary deadline is Friday; transcript explicitly says Thursday | Contradicted; fail |
| A possible rate cut becomes a certain rate cut | Contradicted if explicit uncertainty conflicts with asserted certainty; otherwise unsupported |
| One speaker's opinion becomes a meeting-wide agreement | Contradicted if the transcript records disagreement; otherwise unsupported |

## Calibration before adoption

1. Review the current non-yes alignment claims against their full transcripts. Label
   genuine unsupported facts, contradictions, extraction mistakes and acceptable wording
   or suggestions. Do not assume that a lower rejection rate means a better metric.
2. Start with 30–50 human-reviewed cases, mixing real summaries and minimally edited
   variants. Include clean paraphrases, absent-but-invented facts, changed figures,
   false attribution, lost qualifications and suggestion boundaries. Include long
   transcripts with failures near the beginning, middle and end. This is an initial
   calibration set, not evidence of a universal zero error rate.
3. Reserve a holdout set before tuning the prompts. Compare current alignment,
   built-in HallucinationMetric and the proposed claim-level metric with the same judge
   and input pairs.
   Repeat a representative subset three times to inspect variability.
4. Report missed hallucinations and false alarms separately, plus execution failures
   and median/tail latency. Native strict mode cannot replace this accuracy check.
   Current saved timing can be reused; token usage and reliable cost are not currently
   recorded by the custom adapter. Do not claim dollar savings from call counts alone.
5. Require known contradictions in the holdout to fail and unsupported cases to receive
   warnings; investigate clean-case rejections. Do not use a blanket hallucination-failure
   target for this contradiction-focused policy. State remaining limitations. A clean
   finite holdout still cannot prove perfect future detection.

## Implementation sequence after selecting the policy

1. Add a dedicated custom DeepEval metric and factory, reuse `resolve_model` and the
   existing structured judge adapter, and define claim extraction and verdict schemas.
   Follow DeepEval's established extraction/verdict pattern. Give the policy a version.
2. Persist claims, original summary passages, three-way verdicts, source evidence,
   contradiction-check pass/fail, counts, judge and policy provenance, input hashes and
   per-metric duration. Calculate aggregates from complete verdicts; preserve errors as
   errors. Bump the saved schema and reject obsolete records without compatibility
   readers or migrations.
3. Show per-claim findings in JSON, Excel, HTML and the notebook. Report coverage
   independently. Export must continue to make no judge calls. Add focused offline checks
   for full-transcript mapping, two-call behavior, all three verdicts, evidence matching,
   incomplete responses, invalid records and denominator handling.
4. Use the existing and proposed metrics during the bounded calibration experiment. If
   validated, make the new metric the primary contradiction check. Decide whether every run
   still needs the six-call summarization/coverage evaluation. Do not promise end-to-end
   acceleration while continuing all existing calls, or keep a redundant factuality
   path as a permanent fallback.

Report the primary claim-level measure as:

`Detected contradiction rate = contradicted claims / all successfully judged factual claims`

Show unsupported counts separately. For example: "0 contradictions / 32 judged claims
(0%); 2 unsupported claims need review." A batch may additionally report the number of
summaries failing the contradiction check, with its separate summary-level denominator.
Show evaluation errors and unevaluated summaries; never count them as passes. Empty or
incomplete judgments have no valid rate. The extracted claim count is model-dependent.
Zero detected contradictions does not establish zero unsupported content or a true
zero-percent hallucination rate.
