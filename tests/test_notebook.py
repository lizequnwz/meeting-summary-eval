"""The single notebook stays editable, syntactically valid, and free of saved data."""

import ast
import json
from pathlib import Path


def test_single_notebook_is_clean_and_valid():
    root = Path(__file__).resolve().parents[1]
    notebooks = list((root / "notebooks").glob("*.ipynb"))
    assert [path.name for path in notebooks] == ["evaluate_meetings.ipynb"]
    notebook = json.loads(notebooks[0].read_text())
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), str(notebooks[0]), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
