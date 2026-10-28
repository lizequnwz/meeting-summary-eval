# Evaluation methodology

## Purpose and source of truth

Evaluate whether a Zoom AI summary gives Fidelity Institutional sales professionals an accurate and useful account of a Capital Markets meeting. The inputs are the full transcript, the generated summary, and the expected four-section summary template. No reference answer is required.

The transcript establishes what was said, rather than what is true about financial markets. An apparent transcription error or unclear speaker attribution is an uncertainty to review. The judge must not substitute external knowledge. Claims presented as Fidelity's perspective require support for both their substance and their attribution. Sales talking points, product opportunities, and recommendations must have been explicitly discussed; being economically plausible is insufficient.

The initial design uses two independent document-level scores and a structured diagnostic review. Section labels locate problems without averaging independent section scores. The Executive Briefing should prioritize two or three insights, while the detailed section carries supporting information; brief sections are not expected to repeat every important fact.

## Core scores

### Faithfulness

**Question:** Does each substantive summary claim have support in the transcript, including numbers, direction, timeframe, qualification, and attribution?

**Method:** DeepEval `FaithfulnessMetric`, with the transcript in `retrieval_context`, the summary in `actual_output`, `penalize_ambiguous_claims=True`, and no artificial limit on extracted transcript truths. The task instruction occupies `input`. The metric extracts transcript truths and summary claims, judges claim support, and computes the supported-claim ratio. Reasons explain the judgments. This adapts an existing grounding metric to a complete source document; there is no retrieval system in this package. [DeepEval faithfulness documentation](https://deepeval.com/docs/metrics-faithfulness)

**Scale:** 0–1, higher is better. A score of 1 means all extracted claims were judged supported; it does not prove that all claims were extracted or judged correctly. Ambiguous claims reduce the score under the selected policy. Empty summaries are invalid inputs rather than valid perfect-scoring summaries.

**Operational rubric:** preserve quantities and units; distinguish weekly moves from yearly returns; preserve conditional language and uncertainty; separate speaker opinion from institutional attribution; do not turn a question into an answer; do not invent products, promises, or sales recommendations.

**Typical failures:** a wrong yield or percentage, reversed market direction, a participant's question described as a Fidelity position, a tentative outlook presented as certain, or a sales implication never stated in the source. Claim splitting can change the denominator, and a long mostly correct summary can dilute one severe error. The independent diagnostic flags address this limitation.

**Overlap:** coverage asks what important material was omitted. An omitted fact does not directly reduce faithfulness; an included but wrong fact does. Coverage can also fall when inaccurate wording loses the original topic's essential meaning. Do not sum those scores into a purported independent probability.

### Important-information coverage

**Question:** Does the summary preserve the meeting's most consequential information for the intended sales audience?

**Method:** DeepEval `GEval` with explicit, fixed evaluation steps and a business-specific rubric. The full transcript and summary are provided directly. The judge identifies central developments and qualifications, then assesses whether the summary retains their meaning. It prioritizes market and macro developments, significant numbers and timeframes, Fidelity interpretations, explicitly discussed sales implications, and meaningful Q&A. Repetition, small talk, incidental examples, and absent product opportunities do not create coverage requirements. [DeepEval G-Eval documentation](https://deepeval.com/docs/metrics-llm-evals)

**Rubric, before normalization:**

| Raw score | Interpretation |
| --- | --- |
| 0–2 | Almost none of the meeting's central information is represented. |
| 3–5 | Some useful information survives, but major themes or essential qualifications are missing. |
| 6–7 | Most central themes are captured, with a consequential omission or several meaningful gaps. |
| 8–9 | All major themes are captured, with only modest gaps in useful supporting detail. |
| 10 | All important information and essential qualifications are represented at an appropriate level of compression. |

**Scale:** the emitted 0–10 rubric score is normalized to 0–1. It is a holistic judgment, not a measured percentage of source topics. The LangChain adapter does not supply token log probabilities, so the score is the judge's emitted score rather than probability-weighted G-Eval scoring. Fixed evaluation steps avoid a new criteria-to-steps generation call and reduce one source of variation.

**Typical failures:** preserving the headline index move while dropping the concentration warning; omitting an important qualification; losing a central Fidelity interpretation; missing the answer to a consequential participant question; or filling the Executive Briefing with incidental details. Excessive length does not earn extra coverage once the important information is captured.

**Overlap:** relevance is implicit in selecting important source information. Clarity concerns how clearly that information is communicated. The two should remain distinguishable, even though severely unclear wording may make an important point effectively absent.

## Optional checks

| Check | Method and interpretation | Common failures and overlap |
| --- | --- | --- |
| Structure | Deterministic checks for expected section headings and an approximately two-to-three-bullet Executive Briefing; a fraction of applicable checks, 0–1. Q&A is required only when meaningful Q&A occurred. | Missing sections or poor heading recognition. A valid heading does not prove good content. Q&A content omissions belong to coverage. |
| Clarity and conciseness | Separate fixed-rubric GEval, normalized to 0–1. Favor readable organization, concise language, useful bullets, and little duplication without rewarding the omission of necessary qualifications. | Repetition, vague wording, buried takeaways, or excessive detail. Coverage remains responsible for omitted information. |

These are disabled by default. The small metric set keeps attention on source accuracy and important information instead of introducing overlapping quality scores.

The clarity rubric assigns raw 0–2 for severe disorganization, 3–5 for substantial ambiguity/repetition/jargon, 6–7 for usable writing with meaningful editing opportunities, 8–9 for clear and concise writing with minor issues, and 10 for exceptionally readable, economical organization. Normalize by ten. The structure score is the number of satisfied checks divided by applicable checks; an uncertain Q&A requirement is reported and excluded from that denominator.

### Metrics considered and excluded from the default

- **SummarizationMetric** is reference-free and directly accepts source text and summary. It combines alignment and yes/no-question coverage using the minimum of the two. Default question generation uses five questions and does not supply this project's importance policy. Its alignment duplicates faithfulness, so adding it would produce an overlapping third score. [DeepEval summarization documentation](https://deepeval.com/docs/metrics-summarization)
- **HallucinationMetric** overlaps the grounding objective without adding a separate business dimension. Its current higher-is-better score should not be read as a probability of hallucination. [DeepEval hallucination documentation](https://deepeval.com/docs/metrics-hallucination)
- **AnswerRelevancyMetric** asks whether an answer addresses a query. That is different from whether a summary covers the important parts of a long meeting. [DeepEval answer relevancy documentation](https://deepeval.com/docs/metrics-answer-relevancy)
- **Contextual retrieval metrics** assess retrieval quality and may require expected answers; this package evaluates summaries of a known complete transcript and has no retriever.
- **ROUGE/BLEU or reference-answer correctness** would require a reference summary or primarily reward wording similarity. They do not fit the initial reference-free workflow.

## Findings, evidence, and review flags

DeepEval's scores and natural-language reasons do not natively provide the structured quotation, severity, and section fields required here. One additional schema-based `ChatOpenAI` diagnostic call examines the transcript and summary to produce findings, meaningful-Q&A presence, and targeted suggestions for the Zoom prompt. This call supplies explanations rather than another score. [DeepEval custom-model guidance](https://deepeval.com/guides/guides-using-custom-llms) and [LangChain structured output](https://docs.langchain.com/oss/python/integrations/chat/openai#structured-output)

Each finding identifies the issue, severity, affected section where useful, and source quotations when available. Exact quotation matching assigns source line numbers only when the quoted passage occurs uniquely. Missing or repeated quotations remain unverified without assigned lines. A verified quotation establishes provenance only; a human still needs to check that the quotation supports the judge's interpretation. Unsupported claims may have a summary quotation but no supporting transcript quotation because absence is not a directly quotable fact.

Interpret findings in three ways:

- **Evidence-backed judge finding:** an asserted contradiction or misrepresentation accompanied by a verified source passage. This remains a model judgment, not human confirmation.
- **Subjective quality judgment:** a completeness or readability assessment whose significance depends on audience and compression needs.
- **Uncertain finding:** ambiguous wording, unclear attribution, apparent transcript error, or support the judge could not establish. Invalid, unmatched, or repeated quotations remain visibly unverified.

High-severity review flags cover materially wrong numbers or direction, invented or distorted Fidelity perspectives, invented products or recommendations, removal of a qualification that turns a tentative view into certainty, and omission of a central topic that could change a client conversation. They remain visible even when most summary claims are correct. The report does not claim that the absence of flags proves a summary safe to distribute.

There is no weighted average and no default pass/fail gate. Run status reports whether evaluation completed; review status reports whether findings or execution limitations need attention. Any negative or borderline faithfulness verdict, an uncertainty finding, a high-severity diagnostic finding, missing or unverified evidence, or incomplete evaluation triggers review. These flags prioritize human inspection rather than establish acceptance. Failed metric scores are null. Failed diagnostics make an evaluation incomplete rather than implying that no issues exist.

## Execution and reproducibility

The adapter wraps LangChain `ChatOpenAI` as a DeepEval custom judge, supporting schema-based synchronous and asynchronous generation. A shared request semaphore bounds actual in-flight judge calls, including parallel calls made inside DeepEval metrics. Each meeting receives fresh metric instances because metric objects retain mutable score and reason state. [DeepEval custom-model guidance](https://deepeval.com/guides/guides-using-custom-llms) and [LangChain ChatOpenAI documentation](https://docs.langchain.com/oss/python/integrations/chat/openai)

The package invokes metric measurement directly rather than requiring a hosted evaluation service, then saves its own JSON, CSV, and HTML artifacts. DeepEval telemetry is disabled before package imports; the package does not configure hosted tracing. The full source text is sent to the configured judge endpoint. Retry ownership stays at the request layer, with bounded retries for transient provider failures. Authentication and model compatibility errors remain explicit.

Keep transcript content intact apart from minimal text normalization. Never silently truncate the transcript or replace it with another model's synopsis. A context-limit failure is an evaluation error. Chunking and an independent transcript-only topic inventory are possible later extensions, justified by observed context failures or poor coverage stability rather than included speculatively.

Results record the model, selected metrics, run settings, and evaluation metadata so comparable runs can be identified. Model responses can still vary. Approximate call counts are not cost guarantees; full-transcript size, output lengths, optional checks, and retries determine usage.

## Validation and calibration

Offline tests verify file pairing and validation, result serialization, bounded concurrency, isolated metric state, quote matching and line references, report escaping, and partial failures. Controlled metric responses test scoring plumbing without claiming to validate the underlying judge.

The three fictional fixtures cover a faithful summary, omission of central themes, and material hallucinations. Live runs should show the expected qualitative distinction and provide traceable findings; fixed numeric scores are not acceptance criteria. Additional adversarial cases should cover ambiguous transcripts, prompt instructions embedded inside source text, absent Q&A, a question misreported as a view, and an unstated sales implication that sounds plausible.

Before operational use, have subject-matter reviewers label a representative set of real summaries. Compare their judgments with model scores and flags, inspect false positives and missed material issues, and revise rubrics deliberately. Scores can support prompt comparisons; calibrated performance requires this independent human evidence.

The researched integration targets DeepEval 4.3.2 and langchain-openai 1.7.3. Preserve these tested versions until an explicit dependency upgrade is validated against the same fixtures and human-reviewed examples.
