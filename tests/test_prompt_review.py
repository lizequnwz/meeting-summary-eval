import hashlib
import json

import pytest

from meeting_eval.policy import PRODUCER_PROMPT
from meeting_eval.prompt_review import export_prompts, main, review_prompts


def test_exported_bundle_contains_current_prompts_and_verifiable_file_hashes(tmp_path):
    output = export_prompts(tmp_path / "review")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["producer_template_version"] == "4.1"
    assert len(manifest["evaluator_fingerprint"]) == 64
    assert len(manifest["prompts"]) == 10
    assert PRODUCER_PROMPT in (output / "producer.md").read_text()
    for entry in manifest["prompts"]:
        assert hashlib.sha256((output / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]
    assert "[TRANSCRIPT]" in (output / "diagnostics.md").read_text()
    with pytest.raises(FileExistsError):
        export_prompts(output)


def test_prompt_inspection_never_initializes_a_model(monkeypatch, capsys):
    from meeting_eval.judge import LangChainJudge

    def forbidden(*args, **kwargs):
        pytest.fail("Prompt inspection must not initialize a model")
    monkeypatch.setattr(LangChainJudge, "__init__", forbidden)
    assert main(["--show", "producer"]) == 0
    assert capsys.readouterr().out.strip() == PRODUCER_PROMPT.strip()
    assert set(review_prompts()) >= {"producer", "coverage", "diagnostics", "faithfulness-verdicts"}
