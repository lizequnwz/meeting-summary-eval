"""Execute the notebook offline, including export from a saved evaluation alone."""

import ast
import asyncio
from collections import Counter
import inspect
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "evaluate_meetings.ipynb"


def notebook_cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


async def execute_cells(cells, namespace):
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        code = compile(
            "".join(cell["source"]), str(NOTEBOOK), "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
        outcome = eval(code, namespace)
        if inspect.isawaitable(outcome):
            await outcome


def test_single_notebook_is_clean_and_valid():
    notebooks = list((ROOT / "notebooks").glob("*.ipynb"))
    assert [path.name for path in notebooks] == [NOTEBOOK.name]
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), str(NOTEBOOK), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


@pytest.mark.parametrize("start_in_notebooks", [False, True])
def test_notebook_evaluates_once_and_exports_saved_result(
    tmp_path, monkeypatch, judge, start_in_notebooks,
):
    import meeting_eval

    workspace = tmp_path / "workspace"
    shutil.copytree(ROOT / "examples", workspace / "examples")
    (workspace / "notebooks").mkdir()
    monkeypatch.chdir(workspace / "notebooks" if start_in_notebooks else workspace)

    prepare = meeting_eval.get_or_prepare_questions
    evaluate = meeting_eval.evaluate_meeting

    async def offline_prepare(*args, **kwargs):
        # The scripted judge supplies two questions; production keeps its normal default.
        assert kwargs["n"] is None
        kwargs.update(model=judge, n=2)
        return await prepare(*args, **kwargs)

    async def offline_evaluate(*args, **kwargs):
        kwargs["model"] = judge
        return await evaluate(*args, **kwargs)

    monkeypatch.setattr(meeting_eval, "get_or_prepare_questions", offline_prepare)
    monkeypatch.setattr(meeting_eval, "evaluate_meeting", offline_evaluate)
    namespace = {}
    cells = notebook_cells()
    asyncio.run(execute_cells(cells, namespace))

    results_json = namespace["results_json"]
    saved = meeting_eval.load_evaluation(results_json)
    assert saved["results"] == [namespace["result"]]
    assert saved["results"][0]["status"] == "completed"
    assert namespace["question_set"]["questions"] == saved["results"][0]["question_set"]["questions"]
    assert (results_json.parent / "report.xlsx").is_file()
    assert (results_json.parent / "report.html").is_file()
    stages = Counter(stage for stage, _ in judge.prompts)
    assert stages["Questions"] == 1
    assert stages["Truths"] == stages["Claims"] == stages["SummarizationScoreReason"] == 1

    # Restarted-kernel export must need only the chosen saved JSON, without sources or a judge.
    shutil.rmtree(workspace / "examples")
    shutil.rmtree(workspace / "results" / "questions")
    (results_json.parent / "report.xlsx").unlink()
    (results_json.parent / "report.html").unlink()
    previous_prompts = list(judge.prompts)

    def forbidden(*args, **kwargs):
        raise AssertionError("Export must not evaluate or prepare questions")

    monkeypatch.setattr(meeting_eval, "get_or_prepare_questions", forbidden)
    monkeypatch.setattr(meeting_eval, "evaluate_meeting", forbidden)
    monkeypatch.setattr(type(judge), "respond", forbidden)
    monkeypatch.setattr(type(judge), "bind_tools", forbidden)
    export_cell = next(cell for cell in cells if cell["id"] == "meeting-export")
    restarted_namespace = {"results_json": results_json}
    asyncio.run(execute_cells([export_cell], restarted_namespace))

    assert judge.prompts == previous_prompts
    assert (results_json.parent / "report.xlsx").is_file()
    assert (results_json.parent / "report.html").is_file()
    assert meeting_eval.load_evaluation(results_json) == saved
