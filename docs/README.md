# Evaluation documentation

This project evaluates existing meeting summaries against their transcripts. It uses
DeepEval's native alignment and coverage scores, freezes transcript-specific assessment
questions for reuse, and exports Excel and HTML from a saved evaluation.
All LLM requests use LangChain chat models with structured output; DeepEval's native
scoring remains unchanged.

| Document | What it explains |
| --- | --- |
| [Approach](approach.md) | The problem, evaluation boundaries, workflow, and reasons for the main design choices |
| [Methodology](methodology.md) | Claims versus questions, exact scoring rules, question selection, and interpretation limits |
| [Workflow and reports](workflow.md) | Running evaluations, regenerating questions, using the notebook, reading reports, and recovering from errors |
| [Implementation](implementation.md) | Module responsibilities, Python APIs, stored records, validation, persistence, and tests |
| [Standalone contradiction metric](contradiction-metric.md) | Independently checking full transcripts and summaries, binary contradiction scores, unsupported warnings, and per-claim evidence |

Start with **Approach** for an overview, **Methodology** to understand a score, or
**Workflow and reports** to run the evaluator. **Implementation** is intended for people
maintaining or extending the code.

The [repository README](../README.md) covers installation and a short getting-started
path. The [notebook](../notebooks/evaluate_meetings.ipynb) provides an editable single-meeting
workflow. The [business-context example](../examples/business-context.md) defines the
current audience and purpose; it can be replaced for another use case.

These documents describe the implemented workflow. They do not prescribe a summary
generation prompt or add separate format, rubric, or calibration scores.

The separate [hallucination evaluation plan](hallucination-evaluation-plan.md) researches
DeepEval's built-in metrics and proposes checking extracted summary claims against the
full transcript, with contradiction failures and separate unsupported-claim warnings.
The [standalone contradiction metric](contradiction-metric.md) now implements that check
as a separate Python API. The plan's calibration and saved-report rollout remain
proposals; the existing CLI and notebook workflow do not run the new metric.
