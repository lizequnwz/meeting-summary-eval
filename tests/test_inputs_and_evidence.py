from meeting_eval.diagnostics import check_structure, evidence_complete, verify_evidence
from meeting_eval.inputs import discover_pairs, normalize_text
from meeting_eval.models import Evidence, Finding


def test_quotes_are_rechecked_and_lines_are_derived():
    finding = Finding(kind="missing_topic", severity="medium", description="An omission", evidence=[
        Evidence(source="transcript", quote="first\nsecond", line_start=999, verified=True),
        Evidence(source="summary", quote="not present", line_start=999, verified=True),
        Evidence(source="summary", quote="", verified=True),
    ])
    verify_evidence([finding], "intro\nfirst\nsecond\n", "summary")
    assert finding.evidence[0].line_start == 2
    assert finding.evidence[0].line_end == 3
    assert finding.evidence[0].verified
    assert not finding.evidence[1].verified
    assert finding.evidence[1].line_start is None
    assert not finding.evidence[2].verified


SUMMARY = """# Executive Briefing
- One important point.
- Another point.
## Sales Insights
No product opportunities were discussed.
## Detailed Market Updates and Fidelity Perspectives
Details.
"""


def test_structure_qa_absence_and_applicability():
    complete, findings = check_structure(SUMMARY, False)
    assert complete.score == 1 and not findings
    incomplete, findings = check_structure(SUMMARY, True)
    assert incomplete.score == 0.8
    assert findings[0].kind == "structure"
    uncertain, _ = check_structure(SUMMARY, None)
    assert uncertain.score == 1
    assert uncertain.details["qa_applicability_uncertain"]


def test_discovery_and_bom(tmp_path):
    t, s = tmp_path / "t", tmp_path / "s"
    t.mkdir()
    s.mkdir()
    (t / "one.TXT").write_text("x")
    (s / "one.md").write_text("x")
    (s / "ignore.pdf").write_text("x")
    assert len(discover_pairs(t, s)) == 1
    assert discover_pairs(t, s)[0].error is None
    assert normalize_text("\ufeffa\r\nb\rc") == "a\nb\nc"


def test_repeated_quotes_are_not_given_arbitrary_line_numbers():
    finding = Finding(kind="uncertainty", severity="medium", description="Ambiguous excerpt", evidence=[
        Evidence(source="transcript", quote="Yes.")])
    verify_evidence([finding], "Speaker A: Yes.\nSpeaker B: Yes.", "summary")
    assert not finding.evidence[0].verified
    assert finding.evidence[0].line_start is None


def test_correct_quote_from_wrong_source_is_incomplete_evidence():
    finding = Finding(kind="missing_topic", severity="medium", description="Omitted topic", evidence=[
        Evidence(source="summary", quote="summary", verified=True)])
    assert not evidence_complete(finding)
    unsupported = Finding(kind="unsupported_claim", severity="medium", description="Support not found", evidence=[
        Evidence(source="summary", quote="summary", verified=True)])
    assert evidence_complete(unsupported)
