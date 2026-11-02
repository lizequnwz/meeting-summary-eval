# Meeting-summary content guidance

This checklist is derived from **FI CMSG Weekly Market Updates and Sales Insights —
Template 4.1**, recovered from Git history at
`2360f54^:src/meeting_eval/prompts/summary-template-v4.1.md`. Commit `2360f54` removed
that producer prompt. The producer prompt remains deleted.

This reference retains its substantive requirements for institutional sales readers.
It guides preparation of transcript-specific factual coverage questions and receives no
separate rubric scores. It omits prescribed headings, section order, bullet and word limits,
mandatory absence phrases, and search-tag formatting. An executive preview followed by useful supporting
detail is not itself a defect.

The transcript is the sole factual source, matching the current evaluator. Business
context describes audience and purpose; it supplies no meeting facts. This narrows the
historical template's allowance for explicitly supplied factual context.

## Content checklist

These comparative questions explain the content distinctions in the historical prompt.
They are guidance rather than questions submitted to the native metric. Meaning matters
wherever it appears in a summary, independent of section names or order.

1. **Q01 — Important takeaways:** Does the summary preserve the meeting's most consequential
   takeaways for advisor conversations, client concerns, portfolio discussions, or near-term
   outreach?
2. **Q02 — Economic and market developments:** Does the summary cover the distinct,
   materially important economic and market developments discussed, with their significance
   represented only to the extent supported by the transcript?
3. **Q03 — Investment perspectives:** Does the summary preserve significant investment,
   positioning, risk, and portfolio perspectives, including materially different viewpoints
   and tradeoffs?
4. **Q04 — Figures and scope:** Does the summary preserve material quantities, units,
   comparisons, direction of change, dates, time horizons, and the scope of the statements
   they describe?
5. **Q05 — Risks and qualifications:** Does the summary retain material risks, conditions,
   caveats, uncertainty, and disagreement without strengthening tentative statements or
   narrowing away important qualifications?
6. **Q06 — Attribution and endorsement:** Does the summary accurately distinguish
   observations, opinions, forecasts, and recommendations, using the narrowest supported
   attribution without inferring institutional endorsement from employment, a question,
   or a product mention?
7. **Q07 — Ambiguous or conflicting information:** Where the transcript is materially
   unclear or conflicting, does the summary preserve that uncertainty instead of silently
   resolving it or inventing a correction?
8. **Q08 — Discussed actions:** Does the summary faithfully report consequential actions,
   recommendations, and follow-ups explicitly discussed, preserving their scope and
   including owners or deadlines only when stated?
9. **Q09 — Suggested outreach:** When the summary proposes outreach ideas, are they clearly
   distinguishable from meeting-endorsed actions, exploratory, connected to supported
   insights, and free of invented factual premises, investment recommendations, promises,
   owners, or deadlines?
10. **Q10 — Products and solutions:** Are references to specific products or solutions
    supported by the transcript, without turning discussion into a recommendation,
    endorsement, launch, or performance promise?
11. **Q11 — Answered questions:** Does the summary preserve substantive questions and
    their answers wherever they occurred in the meeting, retaining material scope,
    qualifications, and disagreement when consolidating related questions?
12. **Q12 — Unanswered and generated questions:** Does the summary keep substantive
    unanswered questions explicitly unanswered, distinguish generated outreach questions
    from meeting Q&A, and avoid inventing questions, answers, or the absence of Q&A?
13. **Q13 — Direct quotations:** Are direct quotations exact and assigned to a speaker
    whose identity is supported by the transcript?
14. **Q14 — Factual grounding:** Are all meeting-related factual statements, causal
    explanations, and conclusions supported by the transcript, including those in takeaways, outreach
    premises, and tags, without adding outside knowledge or unsupported meeting events?

## Use in question preparation

The running evaluator uses native DeepEval alignment and coverage only. It does not
produce Yes/No/Not applicable answers to this checklist, extract evidence passages for
these checks, or calculate a comparative-rubric score.

When preparing a question set, prioritize consequential facts actually present in the
transcript: material quantities and horizons, risks and qualifications, attribution,
discussed actions, and substantive answered or unanswered questions. Do not force every
category into every meeting. Sparse sources need no invented outreach, products, quotes,
Q&A, or topics. Preserve the distinction between generated exploratory outreach and an
action discussed in the meeting; native alignment still applies its normal source-grounding
rules to the summary.

The resulting assessment questions must concern specific transcript information that can
be answered from the transcript and summary separately. The generic comparative questions
above cannot be passed directly to `SummarizationMetric.assessment_questions`, because
native coverage answers each question independently from each text rather than comparing
the texts in one rubric judgment.

Questions are generated on first use and saved with their provenance. Later evaluations
reuse the exact list. Changed transcript content, business context, or an explicitly
requested count requires deliberate regeneration. Saved evaluations embed the complete
question set so reports continue to describe the questions that were actually judged.

Prescribed headings, section order, bullet and word limits, mandatory absence phrases,
and search-tag formatting remain outside evaluation. No producer prompt is restored and
no human-reviewed calibration is added.
