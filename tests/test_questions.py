"""Question preparation is a one-call, validated, explicitly refreshed operation."""

import asyncio
from copy import deepcopy
import json

import pytest
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.utils import EvaluationCost

from meeting_eval.questions import get_or_prepare_questions, validate_question_set
from meeting_eval.storage import text_hash


class QuestionJudge(DeepEvalBaseLLM):
    def __init__(self, questions=None, *, cost=None, error=None):
        self.questions = questions if questions is not None else ["Is the outlook discussed?", "Is a risk identified?"]
        self.cost = cost
        self.error = error
        self.prompts = []
        self.base_url = "https://user:secret@offline.invalid/v1?api_key=hidden#private"

    def load_model(self):
        return self

    def get_model_name(self):
        return "offline-question-judge"

    def generate(self, prompt, schema=None):
        self.prompts.append((schema.__name__, prompt))
        if self.error:
            raise self.error
        result = schema.model_validate({"questions": self.questions})
        return (result, self.cost) if self.cost is not None else result

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)


def prepare(tmp_path, judge=None, **kwargs):
    options = {"questions_dir": tmp_path, "model": judge or QuestionJudge(), "n": 2}
    options.update(kwargs)
    meeting_id = options.pop("meeting_id", "meeting_001")
    transcript = options.pop("transcript", "SourceAlpha: the meeting transcript.")
    return asyncio.run(get_or_prepare_questions(meeting_id, transcript, **options))


def forbid_model(monkeypatch):
    monkeypatch.setattr("meeting_eval.questions.resolve_model", lambda *args: pytest.fail("No model should be created"))


def test_prepare_once_and_reuse_without_creating_a_judge(tmp_path, monkeypatch):
    judge = QuestionJudge(cost=EvaluationCost(0.0123, 100, 20))
    saved, generated = prepare(tmp_path, judge, business_context="AdvisorAudience")

    assert generated is True
    assert len(judge.prompts) == 1
    stage, prompt = judge.prompts[0]
    assert stage == "Questions"
    assert "SourceAlpha" in prompt and "AdvisorAudience" in prompt
    assert "never as factual evidence" in prompt
    assert "explicitly discussed actions" in prompt and "unanswered status" in prompt
    assert "not a mandatory checklist" in prompt and "standalone questions" in prompt
    assert saved["transcript_sha256"] == text_hash("SourceAlpha: the meeting transcript.")
    assert saved["business_context_sha256"] == text_hash("AdvisorAudience")
    assert saved["generation"]["cost"] == 0.0123
    assert saved["generation"]["endpoint"] == "https://offline.invalid/v1"
    assert saved["generation"]["duration_seconds"] >= 0
    assert json.loads((tmp_path / "meeting_001.json").read_text()) == saved
    validate_question_set(saved)

    forbid_model(monkeypatch)
    reused, generated = prepare(tmp_path, business_context="AdvisorAudience", model="another-judge", n=None)
    assert generated is False
    assert reused == saved


@pytest.mark.parametrize("changed", [
    {"transcript": "A changed transcript."},
    {"business_context": "A changed audience."},
    {"n": 3},
])
def test_stale_questions_fail_before_any_model_call(tmp_path, monkeypatch, changed):
    prepare(tmp_path)
    forbid_model(monkeypatch)
    with pytest.raises(ValueError, match="regenerate them explicitly"):
        prepare(tmp_path, **changed)


def test_prompt_changes_require_explicit_regeneration(tmp_path, monkeypatch):
    prepare(tmp_path)
    import meeting_eval.questions as module
    monkeypatch.setattr(module, "_CONTEXT_INSTRUCTIONS", module._CONTEXT_INSTRUCTIONS + "Changed policy.")
    forbid_model(monkeypatch)
    with pytest.raises(ValueError, match="regenerate them explicitly"):
        prepare(tmp_path)


def test_normalized_sources_match_and_regeneration_preserves_count(tmp_path):
    first, _ = prepare(tmp_path, transcript="\ufeffSource\r\nAlpha", business_context="\ufeffAudience\r\nPurpose")
    reused, generated = prepare(tmp_path, transcript="Source\nAlpha", business_context="Audience\nPurpose", n=None)
    assert generated is False
    assert reused == first

    regenerated, generated = prepare(tmp_path, transcript="Source changed", business_context="Audience\nPurpose",
                                       n=None, regenerate=True)
    assert generated is True
    assert regenerated["n"] == 2
    assert regenerated["question_set_id"] != first["question_set_id"]


def test_default_count_is_five_for_a_new_meeting(tmp_path):
    judge = QuestionJudge(questions=[f"Is fact {index} identified?" for index in range(5)])
    saved, _ = prepare(tmp_path, judge, n=None)
    assert saved["n"] == 5


def test_omitted_context_uses_stock_native_generation_prompt(tmp_path):
    from deepeval.metrics.summarization import SummarizationTemplate

    judge = QuestionJudge()
    prepare(tmp_path, judge)
    assert judge.prompts[0][1] == SummarizationTemplate.generate_questions(
        text="SourceAlpha: the meeting transcript.", n=2,
    )


def test_stable_content_id_does_not_include_time_or_model(tmp_path):
    first, _ = prepare(tmp_path)
    second, _ = prepare(tmp_path, regenerate=True)
    assert second["question_set_id"] == first["question_set_id"]


@pytest.mark.parametrize("corrupt", [
    lambda value: value.update(schema_version=2),
    lambda value: value.update(question_set_id="0" * 64),
    lambda value: value["questions"].__setitem__(0, "Tampered question?"),
    lambda value: value.update(n=True),
    lambda value: value.update(business_context="Changed without updating hash"),
    lambda value: value.update(meeting_id="another_meeting"),
    lambda value: value.update(extra="obsolete"),
    lambda value: value["generation"].update(cost=-1),
    lambda value: value["generation"].update(endpoint="https://user:secret@offline.invalid/v1"),
])
def test_corrupt_saved_questions_fail_without_a_model(tmp_path, monkeypatch, corrupt):
    saved, _ = prepare(tmp_path)
    corrupt(saved)
    (tmp_path / "meeting_001.json").write_text(json.dumps(saved))
    forbid_model(monkeypatch)
    with pytest.raises(ValueError, match="invalid; regenerate"):
        prepare(tmp_path)


def test_invalid_json_requires_regeneration(tmp_path, monkeypatch):
    (tmp_path / "meeting_001.json").write_text("{broken")
    with monkeypatch.context() as patch:
        forbid_model(patch)
        with pytest.raises(ValueError, match="invalid; regenerate"):
            prepare(tmp_path)
    saved, generated = prepare(tmp_path, regenerate=True)
    assert generated is True
    validate_question_set(saved)


@pytest.mark.parametrize("questions", [["Only one?"], ["Question?", " QUESTION? "], ["Question?", " "]])
def test_invalid_generation_never_replaces_saved_questions(tmp_path, questions):
    prepare(tmp_path)
    path = tmp_path / "meeting_001.json"
    original = path.read_bytes()
    with pytest.raises(ValueError):
        prepare(tmp_path, QuestionJudge(questions), regenerate=True)
    assert path.read_bytes() == original


def test_provider_failure_is_safe_and_preserves_previous_questions(tmp_path):
    prepare(tmp_path)
    path = tmp_path / "meeting_001.json"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Question generation failed") as error:
        prepare(tmp_path, QuestionJudge(error=RuntimeError("secret source and API key")), regenerate=True)
    assert "secret source" not in str(error.value)
    assert path.read_bytes() == original


def test_unwritable_destination_fails_before_model_and_preserves_saved_set(tmp_path, monkeypatch):
    prepare(tmp_path)
    original = (tmp_path / "meeting_001.json").read_bytes()
    forbid_model(monkeypatch)

    def denied(*args, **kwargs):
        raise PermissionError("Destination is not writable")

    monkeypatch.setattr("meeting_eval.questions.tempfile.NamedTemporaryFile", denied)
    with pytest.raises(PermissionError, match="not writable"):
        prepare(tmp_path, regenerate=True)
    assert (tmp_path / "meeting_001.json").read_bytes() == original


@pytest.mark.parametrize("invalid_target", ["parent_file", "target_directory"])
def test_invalid_destination_fails_before_model(tmp_path, monkeypatch, invalid_target):
    destination = tmp_path / "questions"
    if invalid_target == "parent_file":
        destination.write_text("Existing unrelated file")
    else:
        (destination / "meeting_001.json").mkdir(parents=True)
    forbid_model(monkeypatch)
    with pytest.raises((OSError, ValueError)):
        prepare(destination, regenerate=True)


def test_write_probe_does_not_leave_an_artifact_after_generation_failure(tmp_path):
    with pytest.raises(ValueError, match="Question generation failed"):
        prepare(tmp_path, QuestionJudge(error=RuntimeError("Provider failure")))
    assert list(tmp_path.iterdir()) == []


def test_unknown_native_cost_stays_unknown(tmp_path):
    saved, _ = prepare(tmp_path, QuestionJudge(cost=EvaluationCost(None, 10, 20)))
    assert saved["generation"]["cost"] is None


@pytest.mark.parametrize("kwargs", [
    {"n": 0}, {"n": True}, {"n": 1.5}, {"transcript": " "}, {"transcript": "\ufeff"},
    {"business_context": " "}, {"business_context": "\ufeff"}, {"meeting_id": "../other"},
])
def test_invalid_preparation_input_fails_before_model(tmp_path, monkeypatch, kwargs):
    forbid_model(monkeypatch)
    with pytest.raises(ValueError):
        prepare(tmp_path, **kwargs)


def test_historical_question_snapshot_can_be_validated_after_template_change(tmp_path, monkeypatch):
    saved, _ = prepare(tmp_path)
    original = deepcopy(saved)
    monkeypatch.setattr("meeting_eval.questions._prompt_sha256", lambda: "0" * 64)
    validate_question_set(saved)
    assert saved == original
