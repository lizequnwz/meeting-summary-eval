# Validation record — October 28, 2026

The project was installed in `.venv` on Python 3.13 with DeepEval 4.3.2 and
langchain-openai 1.7.3. The existing `.env` was used without editing its contents.

## Software and integration checks

- **45 offline tests passed**, including actual DeepEval metric execution with a
  controlled schema-aware judge, rubric normalization, request and meeting
  concurrency, partial failures, batch-wide judge failures, source verification,
  report escaping, and repeated synchronous calls through a real LangChain/SDK
  client with an offline HTTP transport.
- `pip check` reported no broken requirements. The CLI help and end-to-end sample
  batch were exercised.
- A live `gpt-6-luna` schema probe succeeded through the configured endpoint.
- Both notebooks executed top-to-bottom with real judge calls and saved outputs.
  Their HTML exports are under `results/validation/notebooks/`.

## Latest saved notebook batch

The detailed artifacts are under
`results/notebook/20261028T132433Z-f3067d5a/` (JSON, CSV and HTML).

These generated `results/` artifacts are local and excluded from Git; rerun the
notebooks or CLI to regenerate them. Executed notebook outputs remain included.

| Fictional meeting | Faithfulness | Coverage | Execution |
| --- | ---: | ---: | --- |
| 001: faithful summary | 0.929 | 1.0 | Completed |
| 002: major omissions | 0.750 | 0.3 | Completed |
| 003: material distortions | 0.143 | 0.0 | Completed |

All three structured diagnostic reviews completed. Every returned evidence quote
was uniquely matched to its source. The omitted concentration/liquidity themes
and the invented product, reversed market move, guarantees, and reversed Q&A were
detected in the flawed examples.

The faithful summary received one borderline faithfulness verdict about the
phrase “Use the explicitly discussed talking point”; the diagnostic review
returned no findings. The same fixture's single-meeting notebook run scored 1.0
faithfulness and 0.9 coverage without review flags. This exposes conservative
overinterpretation and judge variation rather than a deterministic difference in
the source. The implementation preserves the disagreement and flags it for
review; it does not tune scores to make the sample pass. These are synthetic
integration checks, not a measurement of accuracy on Fidelity meetings.

The notebook score chart was visually inspected without clipping. Full HTML
browser preview was unavailable because the browser policy disallowed local-file
viewing; HTML content, escaping and report generation were checked programmatically.

## Next validation step

Have subject-matter reviewers assess representative real transcript/summary pairs,
including good paraphrases and subtle financial misrepresentations. Inspect missed
material issues and false positives before setting thresholds or using scores for
operational acceptance. Compare runs with the same model and rubric version.
