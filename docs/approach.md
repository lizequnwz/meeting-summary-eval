# Evaluation approach

## Purpose and audience

The evaluator checks an existing meeting summary against the meeting transcript. Its
current business context is institutional sales professionals preparing advisor and
client conversations after a capital-markets meeting. Important content can include
material developments, figures, horizons, perspectives, risks, qualifications, actions,
and substantive questions and answers.

The evaluator returns two complementary measures: whether extracted summary claims are
supported, and whether selected transcript information survives in the summary. The
details explain the scores so a reader can investigate omissions and distortions.

Every LLM request uses a LangChain chat model with structured output. DeepEval retains
its native prompts and scoring behavior; the shared judge adapter connects those requests
to LangChain for question preparation and summary evaluation.

The transcript establishes what was said. It does not independently verify market facts,
forecasts, or investment opinions. Business context identifies reader needs and priorities;
it contributes no factual evidence about the meeting.

## End-to-end process

1. **Load inputs.** Pair transcript and summary files by meeting ID and normalize their
   text. Record input identifiers and report missing, ambiguous, unreadable, or blank inputs.
2. **Prepare or load questions.** For a new meeting, generate a factual question set from
   the transcript, guided by business context, and save it. For an existing meeting, load
   its matching frozen set.
3. **Evaluate the summary.** Create a fresh native DeepEval metric with the frozen
   questions. Extract transcript statements and summary claims, judge alignment, and
   answer the coverage questions against each text.
4. **Validate and save.** Check that the judgments are complete and that native scores
   match their verdicts. Save successful or failed meeting records in `results.json`.
5. **Export.** Load that saved evaluation to create the Excel workbook and standalone
   HTML report. This stage makes no model calls.

The batch command checkpoints as it progresses. The notebook exposes preparation,
evaluation/save, inspection, and export as separate cells, allowing questions to be
reviewed before scoring. See [Workflow and reports](workflow.md) for their exact behavior.

## Why questions are specific to a transcript

A general requirement such as "preserve material figures" describes a quality criterion.
A native coverage question must instead test an actual source fact, such as "Did annual
inflation decline from 3.4% to 3.2%?" DeepEval answers that question from the transcript
and summary separately.

The generation instructions and business context determine which facts matter. The
transcript determines which factual questions can be asked. The summary does not
participate in question preparation, so changing a summary cannot change its own test.

Question selection happens once for a new meeting. Subsequent summary versions reuse
the same list. This preserves a consistent sample for comparison and avoids repeating
the generation call. It does not make claim extraction or subsequent judgments
deterministic.

## Design decisions

| Decision | Reason |
| --- | --- |
| Native DeepEval scoring | Keep established score calculations and judgments instead of maintaining a second scoring algorithm |
| LangChain chat models | Use one model interface with structured output for preparation and all native judging requests |
| Frozen questions | Compare summary versions against the same sampled transcript facts |
| Explicit regeneration | Make changes to the evaluation sample deliberate and traceable |
| Fresh metric per evaluation | Avoid carrying native metric state between meetings or summary versions |
| Saved JSON as the reporting source | Keep Excel and HTML tied to the same recorded evaluation and allow offline re-export |
| Separate preparation, evaluation, and reporting | Reuse the same behavior in Python, the notebook, and the CLI without reevaluating just to export |
| Visible errors and denominators | Distinguish incomplete execution, zero evidence, and low measured summary quality |

A question-set content ID identifies its source, context, generation instructions, count,
and ordered questions. Saved evaluations embed the full set used at the time. Explicitly
replacing the current question file therefore does not rewrite historical evaluations.

## Boundaries

The workflow grades alignment and sampled information preservation. It does not generate
summaries, require particular headings or bullet counts, score against a universal
comparative checklist, or impose a pass/fail quality gate.

Native alignment can penalize a derived outreach idea that is absent from the transcript,
even when it is clearly labeled as a suggestion. The project does not add a scoring
exception for this. A correct factual premise and a useful generated suggestion should
still be examined in the underlying verdicts.

The saved details include extracted transcript statements, claims, questions, answerability
verdicts, and available native reasons. Extracted statements are model outputs, not
verified quotations with passage locations. No additional evidence-extraction call or
human-reviewed calibration is part of the implemented process.

See [Methodology](methodology.md) for formulas and interpretation, and
[Implementation](implementation.md) for the public APIs and record structure.
