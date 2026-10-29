import hashlib
import json

import pytest

from meeting_eval.policy import METHODOLOGY_VERSION, POLICY
from meeting_eval.prompt_review import export_prompts, main, review_prompts


def test_exported_bundle_contains_current_prompts_and_verifiable_file_hashes(tmp_path):
    output = export_prompts(tmp_path / "review")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["methodology_version"] == METHODOLOGY_VERSION
    assert len(manifest["evaluator_fingerprint"]) == 64
    assert len(manifest["prompts"]) == 9
    assert POLICY in (output / "policy.md").read_text()
    assert not (output / "producer.md").exists()
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
    assert main(["--show", "policy"]) == 0
    assert capsys.readouterr().out.strip() == POLICY.strip()
    assert set(review_prompts()) >= {"policy", "coverage", "diagnostics", "faithfulness-verdicts"}
    assert "producer" not in review_prompts()


def test_business_context_is_a_separate_review_component_and_has_its_own_identity(tmp_path):
    context = "Audience: engineering leads. Prioritize launch decisions and risks."
    generic_prompts = review_prompts()
    prompts = review_prompts(business_context=context)
    assert prompts["business-context"] == context
    assert context in prompts["diagnostics"]
    for name in ("policy", "faithfulness-source", "faithfulness-claims", "faithfulness-verdicts", "faithfulness-reason"):
        assert prompts[name] == generic_prompts[name]
        assert context not in prompts[name]
    output = export_prompts(tmp_path / "review", business_context=context)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["business_context_sha256"] == hashlib.sha256(context.encode()).hexdigest()
    assert len(manifest["prompts"]) == 10
    assert context in (output / "business-context.md").read_text()
    assert manifest["evaluator_fingerprint"] == json.loads(
        (export_prompts(tmp_path / "generic") / "manifest.json").read_text())["evaluator_fingerprint"]
