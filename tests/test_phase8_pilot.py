import hashlib
import json
from pathlib import Path

from italix.hybrid_triage import hybrid_triage, patch_evidence


def test_pilot_matches_published_labels_and_evidence():
    pilot = Path(__file__).resolve().parents[1] / "data/phase8/pilot-v1.jsonl"
    raw = pilot.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pilot.with_name("SHA256SUMS").read_text().split()[0]
    rows = [json.loads(line) for line in raw.splitlines()]
    assert 100 <= len(rows) <= 200
    assert len({row["id"] for row in rows}) == len({row["diff"] for row in rows}) == len(rows)
    for row in rows:
        output = hybrid_triage(row["diff"], explain=False, engine=None)
        assert output["result"] == row["policy"]["result"], row["id"]
        assert output["abstention_reason"] == row["policy"]["abstention_reason"], row["id"]
        if output["abstention_reason"] != "malformed_or_truncated_hunk":
            _, excerpts = patch_evidence(row["diff"])
            for item in row["semantic"]["evidence"]:
                for operation, key in (("-", "removed"), ("+", "added")):
                    assert any(e["path"] == item["path"] and e["operation"] == operation
                               and e["text"] == item[key] for e in excerpts), row["id"]


def test_phase10_pilot_has_unique_hunks_and_review_pending():
    pilot = Path(__file__).resolve().parents[1] / "data/phase10/pilot-v2.jsonl"
    raw = pilot.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pilot.with_name("SHA256SUMS").read_text().split()[0]
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 100
    seen = set()
    for row in rows:
        assert row["family"] and row["review_status"] == "pending_independent_review"
        result = hybrid_triage(row["diff"], explain=False, engine=None)
        assert result["result"] == row["policy"]["result"], row["id"]
        assert result["abstention_reason"] == row["policy"]["abstention_reason"], row["id"]
        if result["abstention_reason"] == "malformed_or_truncated_hunk":
            continue
        _, lines = patch_evidence(row["diff"])
        for evidence in row["semantic"]["evidence"]:
            pair = (evidence["removed"], evidence["added"])
            assert pair not in seen, row["id"]
            seen.add(pair)
            assert [(line["operation"], line["text"]) for line in lines if line["path"] == evidence["path"]] == [
                ("-", pair[0]), ("+", pair[1]),
            ]
    ruby = rows[43]
    assert ruby["semantic"]["evidence"][0]["removed"] == "cached || compute_price"
