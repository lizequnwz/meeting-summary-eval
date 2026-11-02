"""Native judge completeness, provenance, score explanations, and saved snapshots."""

import asyncio
from copy import deepcopy
import json

import pytest

from meeting_eval import evaluate_meeting, get_or_prepare_questions, load_evaluation, new_run, save_evaluation
from meeting_eval.storage import safe_endpoint, text_hash


def prepare(tmp_path, judge):
    return asyncio.run(get_or_prepare_questions(
        "meeting", "SourceAlpha", questions_dir=tmp_path / "questions", model=judge, n=2,
    ))[0]


def evaluate(question_set, judge, **kwargs):
    return asyncio.run(evaluate_meeting(
        "meeting", "SourceAlpha", "SummaryAlpha", question_set=question_set, model=judge, **kwargs,
    ))


def test_native_judgments_are_paired_and_saved_without_rejudging(tmp_path, judge):
    frozen = prepare(tmp_path, judge)
    row = evaluate(frozen, judge, question_generation_performed=True)
    assert row["status"] == "completed"
    assert row["counts"] == {"alignment_supported": 1, "alignment_total": 3,
                             "coverage_covered": 1, "coverage_eligible": 2, "coverage_total": 2}
    assert row["alignment"][1] == {"claim_id": "C002", "claim": "Second claim", "verdict": "no", "reason": "The claim is no."}
    assert row["coverage"][1]["outcome"] == "missing"
    assert row["evaluation"]["model"] == judge.model_name
    assert row["evaluation"]["cost"] is None
    assert row["evaluation"]["duration_seconds"] >= 0
    run = new_run()
    run.update(status="completed", results=[row])
    original_calls = len(judge.prompts)
    path = save_evaluation(run, tmp_path / "results.json")
    assert load_evaluation(path) == run
    assert len(judge.prompts) == original_calls
    frozen["questions"][0] = "A later mutation"
    assert row["question_set"]["questions"][0] != "A later mutation"


@pytest.mark.parametrize("malformation", ["coverage", "alignment", "answerability"])
def test_incomplete_structured_responses_cannot_produce_completed_scores(tmp_path, judge, monkeypatch, malformation):
    frozen = prepare(tmp_path, judge)
    original = judge.respond

    def malformed(prompt, schema=None):
        if malformation == "coverage" and schema.__name__ == "Answers":
            return schema.model_validate({"answers": ["yes"]})
        if malformation == "alignment" and schema.__name__ == "Verdicts":
            return schema.model_validate({"verdicts": [{"verdict": "yes", "reason": None}]})
        if malformation == "answerability" and schema.__name__ == "Answers":
            return schema.model_validate({"answers": ["yes", "maybe"]})
        return original(prompt, schema)

    monkeypatch.setattr(judge, "respond", malformed)
    row = evaluate(frozen, judge)
    assert row["status"] == "error"
    assert "Incomplete evaluation" in row["error"]
    assert row["scores"] is row["counts"] is None
    assert row["alignment"] == row["coverage"] == []
    assert row["evaluation"]["duration_seconds"] >= 0


def test_coverage_excludes_source_unanswerable_questions(tmp_path, judge, monkeypatch):
    frozen = prepare(tmp_path, judge)
    original = judge.respond

    def source_unanswerable(prompt, schema=None):
        if schema.__name__ == "Answers":
            return schema.model_validate({"answers": ["yes", "no"]})
        return original(prompt, schema)

    monkeypatch.setattr(judge, "respond", source_unanswerable)
    row = evaluate(frozen, judge)
    assert row["status"] == "completed"
    assert row["scores"]["coverage"] == 1
    assert row["counts"]["coverage_eligible"] == row["counts"]["coverage_covered"] == 1
    assert row["counts"]["coverage_total"] == 2
    assert row["coverage"][1]["outcome"] == "excluded"


def test_zero_denominators_retain_native_zero_scores(tmp_path, judge, monkeypatch):
    frozen = prepare(tmp_path, judge)
    original = judge.respond

    def empty_details(prompt, schema=None):
        if schema.__name__ == "Claims":
            return schema.model_validate({"claims": []})
        if schema.__name__ == "Answers":
            return schema.model_validate({"answers": ["no", "no"]})
        return original(prompt, schema)

    monkeypatch.setattr(judge, "respond", empty_details)
    row = evaluate(frozen, judge)
    assert row["status"] == "completed"
    assert row["scores"] == {"alignment": 0, "coverage": 0, "overall": 0}
    assert row["counts"]["alignment_total"] == row["counts"]["coverage_eligible"] == 0


def test_changed_transcript_cannot_use_another_snapshot(tmp_path, judge):
    frozen = prepare(tmp_path, judge)
    previous_calls = len(judge.prompts)
    row = asyncio.run(evaluate_meeting(
        "meeting", "Changed SourceAlpha", "SummaryAlpha", question_set=frozen, model=judge,
    ))
    assert row["status"] == "error"
    assert "does not match" in row["error"]
    assert len(judge.prompts) == previous_calls


def test_provider_errors_do_not_expose_sources_or_credentials(tmp_path, judge, monkeypatch):
    frozen = prepare(tmp_path, judge)

    def failed(*args, **kwargs):
        raise RuntimeError("SecretSource Authorization: secret-key")

    monkeypatch.setattr(judge, "respond", failed)
    row = evaluate(frozen, judge)
    assert row["status"] == "error"
    assert "SecretSource" not in json.dumps(row)
    assert "secret-key" not in json.dumps(row)


def test_failing_model_metadata_cannot_escape_error_redaction(tmp_path, judge, monkeypatch):
    from meeting_eval.judge import LangChainJudge

    frozen = prepare(tmp_path, judge)

    def failed_name(self):
        raise RuntimeError("Private authentication detail")

    monkeypatch.setattr(LangChainJudge, "get_model_name", failed_name)
    row = evaluate(frozen, judge)
    assert row["status"] == "error"
    assert row["evaluation"]["model"] is None
    assert "Private authentication detail" not in json.dumps(row)


@pytest.mark.parametrize("mutation", ["score", "count", "order", "missing_verdict", "schema"])
def test_inconsistent_saved_results_are_rejected(tmp_path, judge, mutation):
    row = evaluate(prepare(tmp_path, judge), judge)
    run = new_run()
    run.update(status="completed", results=[row])
    changed = deepcopy(run)
    if mutation == "score":
        changed["results"][0]["scores"]["overall"] = float("nan")
    elif mutation == "count":
        changed["results"][0]["counts"]["coverage_eligible"] = 1
    elif mutation == "order":
        changed["results"][0]["coverage"].reverse()
    elif mutation == "missing_verdict":
        changed["results"][0]["coverage"].pop()
    else:
        changed["schema_version"] = 2
    path = tmp_path / "results.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        load_evaluation(path)
    with pytest.raises(ValueError):
        save_evaluation(changed, path)


def test_endpoint_and_normalized_input_provenance():
    assert safe_endpoint("https://user:secret@example.com/v1?api_key=secret#fragment") == "https://example.com/v1"
    assert text_hash("\ufeffSourceAlpha\r\n") == text_hash("SourceAlpha\n")


@pytest.mark.parametrize("field", ["reason", "truths", "evaluation", "inputs", "scores", "counts"])
def test_incomplete_error_record_is_rejected_before_export(tmp_path, field):
    from meeting_eval.evaluation import empty_result

    row = empty_result("meeting")
    row.update(status="error", error="Input unavailable.")
    del row[field]
    run = new_run()
    run.update(status="completed", results=[row])
    path = tmp_path / "results.json"
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="Invalid meeting record"):
        load_evaluation(path)
