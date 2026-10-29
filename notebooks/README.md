# Start here

Read the [learning guide](../docs/learning-guide.md), then follow this path:

1. [00 — Evaluation foundations](00_evaluation_foundations.ipynb): offline concepts and worked exercises.
2. [03 — DeepEval metric lab](03_deepeval_metric_lab.ipynb): offline execution of actual metrics, including optional assessment questions.
3. [01 — Single meeting](01_single_meeting.ipynb): first real judge call and evidence inspection.
4. [02 — Batch evaluation](02_batch_evaluation.ipynb): recurring issues and reports across meetings.
5. [04 — Prompt review and experiments](04_prompt_review_and_experiments.ipynb): offline judge-prompt review and supplied-summary comparisons, with optional live evaluation.

Only 01 and 02 make model calls by default. Lessons 00, 03, and 04 run offline with:

```bash
.venv/bin/python scripts/run_learning_notebooks.py
```

Run that command from the repository root. Executed notebooks and HTML previews go under
`results/learning/`; source notebooks remain clean. The runner uses the invoking Python
environment and blocks outgoing Python socket connections inside the lesson kernels.

To review the evaluation criteria immediately, open
[policy.py](../src/meeting_eval/policy.py), or run
`meeting-eval-prompts --show policy`. Review exports are copies; editing them does not
change the package's prompts.
