# Standalone summary contradiction metric

`SummaryContradictionMetric` checks an existing summary against its complete meeting
transcript. It is independently callable from Python and implements DeepEval's metric
interface. The current CLI, notebook workflow, and saved JSON, Excel, and HTML reports
continue to use the native `SummarizationMetric`; they do not run or save this metric.

The metric distinguishes factual contradictions from claims the transcript does not
establish. **Any contradicted claim fails the check. Unsupported claims are warnings and
do not fail it.** A passing summary can therefore contain unsupported claims. This is a
contradiction check, not a guarantee of zero hallucinations.

## Run a check

Supply the full transcript as `input` and the full generated summary as `actual_output`:

```python
from meeting_eval import SummaryContradictionMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input=transcript,
    actual_output=summary,
)
metric = SummaryContradictionMetric(model="your-judge-model")
metric.measure(test_case)

print(metric.score, metric.is_successful(), metric.reason)
print(metric.counts)
print(metric.contradiction_rate, metric.unsupported_rate)
```

`transcript` and `summary` above are nonblank strings loaded by your application. The
model uses the same settings and LangChain adapter as the existing evaluator: omit
`model` to use `OPENAI_MODEL`, or pass a model name or a configured LangChain
`BaseChatModel` supporting structured output. Endpoint settings and credentials follow
the [installation instructions](../README.md#install). No additional package is needed.

In a notebook or another asynchronous application, await the asynchronous method:

```python
metric = SummaryContradictionMetric(model="your-judge-model")
await metric.a_measure(test_case)
print(metric.score, metric.reason)
```

Create a fresh metric for each meeting. Running it makes billable requests to the
configured judge and sends that judge the transcript and summary.

## How the judgment works

A successful check normally makes two judge calls, excluding retries:

1. Extract atomic factual claims from the complete summary, preserving attribution,
   uncertainty, negation, and scope. Each claim retains an exact summary quotation.
2. Send the complete transcript, complete summary, and all extracted claims together
   for one batched judgment. The transcript is not extracted or summarized before this
   stage. The judge first checks whether extraction faithfully covers the summary's
   factual assertions. Each claim receives a verdict, explanation, and transcript
   quotations; invalid extraction makes the check an error.

The metric calculates its score and overall reason locally. There is no third call to
generate an overall explanation. Two calls describe the implementation; no live latency
benchmark or human-reviewed accuracy calibration has been completed.

| Verdict | Meaning | Effect |
| --- | --- | --- |
| `supported` | The transcript supports the claim, including faithful paraphrase or synthesis | Does not fail |
| `contradicted` | The claim conflicts with transcript evidence | Fails the check |
| `unsupported` | The transcript neither establishes the claim nor clearly contradicts it | Warning; does not fail |

Judgment considers the complete transcript, including corrections and later decisions.
For example, a Thursday deadline contradicts a summary claiming Friday. A deadline absent
from the transcript is unsupported. Omissions and writing style are not scored here;
coverage remains a separate concern.

## Scores and details

- `score` is **1** when no claim is contradicted, otherwise **0**. The passing
  `threshold` is **1**.
- `contradiction_rate` is contradicted claims divided by all successfully judged claims.
- `unsupported_rate` is unsupported claims divided by that same total.
- `counts` contains `total`, `supported`, `contradicted`, and `unsupported` claim counts.

Rates are raw fractions between 0 and 1, not percentage values. Multiply by 100 for a
percentage display. Their denominator is the model's extracted claim set, which can vary
and may miss claims. Neither rate is a probability that a summary is correct, and a
claim-level rate is different from the fraction of summaries failing the check.

The structured records are available for inspection:

```python
claims = [claim.model_dump() for claim in metric.claims]
verdicts = [verdict.model_dump() for verdict in metric.verdicts]
print(claims)
print(verdicts)
```

Each `SummaryClaim` contains `text` and `summary_quote`. Verdicts correspond to the
extracted claim order through IDs such as `C001`, `C002`, and so on. Each `ClaimVerdict`
contains `claim_id`, `verdict`, `reason`, and `transcript_quotes`.

Summary quotations must match the supplied summary exactly. Supported and contradicted
verdicts require transcript quotations that match the supplied transcript exactly.
Quotation validation establishes that the passage exists; it does not prove that the
judge interpreted the passage correctly or that extraction found every factual claim.

Blank inputs, empty claim extraction, malformed responses, incomplete verdicts, and
invalid quotations raise an error instead of producing a passing score. The judgment stage
also rejects incomplete extraction or claims that misrepresent their summary passages;
these are extraction errors, not nonfailing unsupported claims. A summary with no factual
claims is unscorable. The metric records a safe explanation in `metric.error` and clears
prior results when a new check fails. Treat an error as unevaluated, not as zero
contradictions. The judge's extraction check can itself miss an error; it does not prove
completeness or accuracy.

## Current scope

The standalone metric is implemented in
[`src/meeting_eval/contradiction.py`](../src/meeting_eval/contradiction.py). Existing
native alignment and coverage scores keep their current definitions. Integration into
saved evaluations and reports, selection of a primary metric, and human-reviewed
calibration remain future work described in the
[evaluation plan](hallucination-evaluation-plan.md).

Use wording such as “0 detected contradictions across 32 judged claims; 2 unsupported
claims need review.” Zero detected contradictions establishes neither zero unsupported
content nor a true zero-percent hallucination rate.
