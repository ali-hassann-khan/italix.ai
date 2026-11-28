import json

from scripts.evidence_facts import case_facts, hunk_groups, spans_for_python
from scripts.evaluate_evidence_facts import inspect_claims


def test_hunk_ranges_for_insertions_deletions_and_context():
    hunk = {"lines": "@@ -9,4 +9,4 @@\n outer\n-    old\n+    new\n same\n-  gone\n+  added"}
    a, b = list(hunk_groups(hunk))
    assert a == {"old": [{"line": 10, "text": "    old"}], "new": [{"line": 10, "text": "    new"}],
                 "preceding": "outer", "following": "same"}
    assert b["old"][0]["line"] == 12 and b["new"][0]["line"] == 12


def test_python_ast_extracts_public_signature_condition_and_test():
    source = "import math\nTIMEOUT = 30\ndef public(x: int) -> bool:\n    if x > TIMEOUT:\n        return True\ndef test_public():\n    assert public(31)\n"
    spans = spans_for_python(source)
    assert any(s["text"] == "def public(x: int) -> bool" and s["start"] == 3 for s in spans)
    assert any(s["kind"] == "condition" and s["text"] == "x > TIMEOUT" for s in spans)
    assert any(s["kind"] == "test" and s["name"] == "test_public" for s in spans)
    assert any(s["kind"] == "constant" and s["name"] == "TIMEOUT" for s in spans)


def test_non_ast_fallback_keeps_whitespace_inside_string_and_cross_file_locations(tmp_path):
    case = {"id": "x", "repo": "org/repo", "license": "MIT", "source_url": "https://example.com",
            "base": "a", "head": "b", "paths": ["a.rb", "b.rs"],
            "evidence": [{"path": "a.rb", "id": "H001", "lines": '@@ -1 +1 @@\n-label = "two  spaces"\n+label = "two spaces"'},
                         {"path": "b.rs", "id": "H002", "lines": '@@ -1 +1 @@\n-label = "old"\n+label = "new"'}]}
    data = case_facts(case, tmp_path, None)
    assert len(data["facts"]) == 2
    assert data["facts"][0]["locations"]["old"][0]["text"] == 'label = "two  spaces"'
    assert data["cross_file_links"][0]["facts_by_path"] == {"a.rb": ["F001"], "b.rs": ["F002"]}
    assert data["cross_file_links"][0]["status"] == "syntactic_cooccurrence"


def test_checker_detects_wrong_fact_and_reversed_literal():
    fact = {"id": "F001", "hunk": "H001", "path": "app.py",
            "locations": {"old": [{"text": "TIMEOUT = 15"}], "new": [{"text": "TIMEOUT = 30"}]}}
    raw = json.dumps({"claims": [{"text": "`TIMEOUT` changed from 30 to 15.", "evidence_ids": ["F001", "F999"]}]})
    claims, flags = inspect_claims(raw, {"F001": fact})
    assert claims[0]["evidence_ids"] == ["H001"]
    assert set(flags) == {"numeric_contradiction", "unknown_or_missing_fact"}


def test_pinned_line_verification_rejects_shifted_or_altered_evidence(monkeypatch, tmp_path):
    import pytest
    from scripts import evidence_facts

    data = {"id": "x", "base": "a", "head": "b", "facts": [
        {"id": "F001", "path": "a.py", "locations": {"old": [], "new": [{"line": 2, "text": "after"}]}}
    ]}
    monkeypatch.setattr(evidence_facts, "git", lambda *args: "before\nafter\n")
    assert evidence_facts.verify_lines(data, tmp_path) == 1
    data["facts"][0]["locations"]["new"][0]["line"] = 1
    with pytest.raises(ValueError, match="pinned source mismatch"):
        evidence_facts.verify_lines(data, tmp_path)


def test_id_only_ranker_rejects_invented_prose_and_repeated_ids():
    from scripts.compare_annotation_sources import parse_ids

    supplied = {"F001", "F002"}
    assert parse_ids('["F002", "F001"]', supplied) == ["F002", "F001"]
    assert parse_ids('["F001", "F001"]', supplied) == []
    assert parse_ids('["F007"]', supplied) == []
    assert parse_ids('{"claims":["improves speed"]}', supplied) == []
    assert parse_ids('["F001"] fixes latency', supplied) == []


def test_exact_renderer_and_weak_screen_treat_source_directions_separately():
    from scripts.compare_annotation_sources import message_screen, render

    fact = {"id": "F001", "hunk": "H001", "path": "config.py",
            "locations": {"old": [{"line": 3, "text": "TIMEOUT = 15"}],
                          "new": [{"line": 3, "text": "TIMEOUT = 30"}]}}
    statement = render(fact)
    assert "Removed at L3:\nTIMEOUT = 15" in statement["statement"]
    assert "Added at L3:\nTIMEOUT = 30" in statement["statement"]
    assert "unmatched_subject_literal" in message_screen("Use timeout 90", [fact])["flags"]
    assert message_screen("fix", [fact])["automatic_screen"] == "flagged"
