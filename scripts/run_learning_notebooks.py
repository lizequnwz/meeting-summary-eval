"""Execute the default-offline tutorials and save notebooks plus HTML previews."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from urllib.parse import unquote, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
LESSONS = {
    "foundations": "00_evaluation_foundations.ipynb",
    "deepeval": "03_deepeval_metric_lab.ipynb",
    "prompts": "04_prompt_review_and_experiments.ipynb",
}
OFFLINE_GUARD = '''# Inserted by the offline tutorial runner; the source notebook is unchanged.
import os as _tutorial_os
import socket as _tutorial_socket
_tutorial_os.environ["MEETING_EVAL_OFFLINE"] = "1"
def _tutorial_no_network(*args, **kwargs):
    raise RuntimeError("Outgoing connections are disabled in offline tutorial execution.")
_tutorial_socket.socket.connect = _tutorial_no_network
_tutorial_socket.socket.connect_ex = _tutorial_no_network
_tutorial_socket.create_connection = _tutorial_no_network
'''


def lesson_paths(names: list[str]) -> list[Path]:
    paths = [ROOT / "notebooks" / LESSONS[name] for name in dict.fromkeys(names)]
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if notebook.get("metadata", {}).get("meeting_eval", {}).get("offline") is not True:
            raise ValueError(f"{path.name} is not marked as an offline tutorial.")
    return paths


def rebase_html_links(html: str, source_dir: Path, output_dir: Path) -> str:
    """Keep lesson-to-lesson links, and repair links back to source/docs after export."""
    from bs4 import BeautifulSoup

    document = BeautifulSoup(html, "html.parser")
    for link in document.find_all("a", href=True):
        target = urlsplit(link["href"])
        if target.scheme or target.netloc or not target.path:
            continue
        local_path = unquote(target.path)
        if (output_dir / local_path).exists():
            continue
        original = (source_dir / local_path).resolve()
        if original.exists():
            relative = Path(os.path.relpath(original, output_dir)).as_posix()
            link["href"] = urlunsplit(("", "", relative, target.query, target.fragment))
    return str(document)


def run_lessons(names: list[str], output_dir: Path) -> Path:
    """Use this Python environment for kernels; never modify notebook source files."""
    import nbformat
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    paths = lesson_paths(names)
    run_dir = output_dir.expanduser().resolve() / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="meeting-eval-kernels-") as temporary:
        kernel_root = Path(temporary) / "kernels"
        spec_dir = kernel_root / "meeting-eval-learning"
        spec_dir.mkdir(parents=True)
        (spec_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Meeting evaluation tutorials", "language": "python",
            "env": {"MEETING_EVAL_OFFLINE": "1", "MPLCONFIGDIR": str(Path(temporary) / "matplotlib"),
                    "IPYTHONDIR": str(Path(temporary) / "ipython"),
                    "JUPYTER_RUNTIME_DIR": str(Path(temporary) / "runtime")},
        }), encoding="utf-8")
        kernel_specs = KernelSpecManager(kernel_dirs=[str(kernel_root)], ensure_native_kernel=False)
        for path in paths:
            print(f"Executing {path.name} offline...", flush=True)
            notebook = nbformat.read(path, as_version=4)
            notebook.cells.insert(0, nbformat.v4.new_code_cell(OFFLINE_GUARD))
            manager = AsyncKernelManager(kernel_name="meeting-eval-learning", kernel_spec_manager=kernel_specs,
                                         connection_file=str(Path(temporary) / f"kernel-{uuid4().hex}.json"))
            client = NotebookClient(notebook, km=manager, timeout=180, allow_errors=False,
                                    resources={"metadata": {"path": str(ROOT)}})
            try:
                client.execute(cleanup_kc=True)
            finally:
                nbformat.write(notebook, run_dir / path.name)
            html, _ = HTMLExporter(template_name="lab", embed_images=True).from_notebook_node(notebook)
            (run_dir / f"{path.stem}.html").write_text(
                rebase_html_links(html, path.parent, run_dir), encoding="utf-8")
        (run_dir / "execution.json").write_text(json.dumps({
            "python": sys.executable, "lessons": [path.name for path in paths],
            "offline": True, "network_guard": "Python socket connections disabled inside tutorial kernels",
        }, indent=2) + "\n", encoding="utf-8")
    return run_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesson", action="append", choices=tuple(LESSONS), help="Select a lesson; repeat for several")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "learning")
    args = parser.parse_args(argv)
    try:
        output = run_lessons(args.lesson or list(LESSONS), args.output)
    except ImportError:
        parser.exit(2, "Install notebook dependencies with: python -m pip install -e '.[notebooks]'\n")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Tutorial execution failed: {exc}\n")
    print(f"Executed notebooks and HTML previews: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
