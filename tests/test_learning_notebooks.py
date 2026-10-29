import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_learning_notebooks", ROOT / "scripts" / "run_learning_notebooks.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_tutorial_sources_are_clean_executable_notebooks_with_explicit_offline_mode():
    for path in runner.lesson_paths(list(runner.LESSONS)):
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("outputs", []) == []
                assert cell.get("execution_count") is None
                compile("".join(cell["source"]), str(path), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def test_runner_rejects_unmarked_lessons(tmp_path, monkeypatch):
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "live.ipynb").write_text(json.dumps({"metadata": {}, "cells": []}))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "LESSONS", {"live": "live.ipynb"})
    with pytest.raises(ValueError, match="not marked"):
        runner.lesson_paths(["live"])


def test_injected_guard_blocks_network_before_a_connection_is_attempted():
    import subprocess
    import sys

    probe = runner.OFFLINE_GUARD + '''
try:
    _tutorial_socket.create_connection(("example.invalid", 443))
except RuntimeError as exc:
    assert "disabled" in str(exc)
else:
    raise AssertionError("The tutorial guard allowed a network connection")
'''
    subprocess.run([sys.executable, "-c", probe], check=True, timeout=10)


def test_exported_html_retains_working_source_and_lesson_links(tmp_path):
    pytest.importorskip("bs4")
    source, output = tmp_path / "notebooks", tmp_path / "results" / "learning"
    source.mkdir()
    output.mkdir(parents=True)
    (source / "live.ipynb").write_text("Source")
    (output / "next.html").write_text("Next")
    html = '<a href="live.ipynb">Source</a><a href="next.html">Next</a><a href="https://example.com">Web</a><a href="#section">Anchor</a>'
    rebased = runner.rebase_html_links(html, source, output)
    assert 'href="../../notebooks/live.ipynb"' in rebased
    assert 'href="next.html"' in rebased
    assert 'href="https://example.com"' in rebased
    assert 'href="#section"' in rebased
