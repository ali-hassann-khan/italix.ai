import json

import pytest

from scripts.annotate_real_changes import check_candidate, score_human, select
from scripts.prepare_specialization import digest


def case():
    return {
        "id": "c1", "repo": "org/example", "base": "a" * 40, "head": "b" * 40,
        "license": "MIT", "source_url": "https://example.org/c1", "license_url": "https://example.org/license",
        "notices": {}, "paths": ["app.py", "tests.py"],
        "evidence": [
            {"id": "H001", "path": "app.py", "lines": "@@ -99 +100 @@\n-TIMEOUT = 15\n+TIMEOUT = 30"},
            {"id": "H002", "path": "tests.py", "lines": "@@ -2 +2 @@\n-old_check()\n+new_check()"},
        ],
    }


def output(*claims):
    return {"candidate": json.dumps({"claims": [{"text": text, "evidence_ids": ids} for text, ids in claims]})}


def test_checker_detects_reversed_values_unknown_hunks_and_uncovered_files():
    result = check_candidate(case(), output(
        ("`TIMEOUT` changed from 30 to 15.", ["H001", "H003"]),
        ("`BOGUS` is expected in tests.py with threshold 99.", ["H001"]),
    ))
    assert result["flags"]["reversed_numeric_change"] == 1
    assert result["flags"]["missing_or_unknown_hunk"] == 1
    assert result["flags"]["number_missing_from_citation"] == 1  # Header offsets are excluded.
    assert result["flags"]["identifier_missing_from_citation"] == 1
    assert result["flags"]["changed_file_missing_from_citation"] == 1
    assert result["missing_files"] == ["tests.py"]


def test_model_agreement_does_not_hide_conflicting_numeric_claims_or_grant_review():
    reliable = output(("`TIMEOUT` changed from 15 to 30.", ["H001"]),
                      ("Add `new_check`.", ["H002"]))
    reversed_row = output(("`TIMEOUT` changed from 30 to 15.", ["H001"]),
                          ("Add `new_check`.", ["H002"]))
    result = select(case(), {"model1": reliable, "model2": reversed_row, "model3": reliable})
    assert result["selected"] == "model1"
    assert result["automatic_checks"] == "pass"
    assert result["human_review"] == "pending"
    assert result["review_priority"] == "urgent"
    assert result["conflicting_value_identifiers"] == ["TIMEOUT"]
    assert all(score == 1 for score in result["cross_model_hunk_jaccard"].values())


def test_human_scoring_requires_attestation_exact_results_and_calibration_membership(tmp_path):
    prior, audited, reviewer, target = [tmp_path / name for name in ("results", "audit", "human", "score")]
    prior.write_text(json.dumps({"corpus_sha256": "123", "records": [
        {"id": "c1", "provisional_claims": [{}, {}], "automatic_checks": "pass"},
    ]}))
    audited.write_text(json.dumps({"corpus_sha256": "123", "initial_review_packet_ids": ["c1"]}))
    values = {"results_sha256": digest(prior), "reviewer": "human-one", "human_attestation": False,
              "cases": [{"id": "c1", "claim_verdicts": ["supported", "unsupported"],
                         "required_facts": 3, "covered_facts": 1}]}
    reviewer.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="attested"):
        score_human(prior, audited, reviewer, target)
    values["human_attestation"] = True
    reviewer.write_text(json.dumps(values))
    score_human(prior, audited, reviewer, target)
    metrics = json.loads(target.read_text())
    assert metrics["human_claim_precision"] == .5
    assert metrics["human_required_fact_recall"] == 1/3


def test_fenced_bare_array_is_labeled_as_normalized_and_retains_the_same_claim():
    from scripts.annotate_real_changes import decode

    claim = {"text": "Add `new_check`.", "evidence_ids": ["H002"]}
    data, format = decode("```json\n" + json.dumps([claim]) + "\n```")
    assert data == [claim]
    assert format == "fenced_bare_array"


def test_qc_export_keeps_provisional_cases_and_ai_review_separate(tmp_path):
    from scripts.export_annotation_qc import export

    reviewed = {**case(), "diff": "+test"}
    pending = {**reviewed, "id": "c2"}
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(row) for row in (reviewed, pending)) + "\n")
    source_hash = digest(corpus)
    audited, observed, corrections = [tmp_path / name for name in ("audit", "results", "corrections")]
    audited.write_text(json.dumps({"corpus_sha256": source_hash, "initial_review_packet_ids": ["c1"]}))
    observed.write_text(json.dumps({"corpus_sha256": source_hash,
                                    "records": [{"id": "c1"}, {"id": "c2"}], "models": {}}))
    corrections.write_text("@c1 corrected\nH001|Updates the timeout.\n")
    target = tmp_path / "out"
    export(corpus, audited, observed, corrections, target)
    a, b = (json.loads(line) for line in (target / "qc-corpus.jsonl").read_text().splitlines())
    assert a["reference_claims"][0]["evidence_ids"] == ["H001"]
    assert b["disposition"] == "unresolved" and b["reference_claims"] is None
    assert all(row["human_review"] == "pending" and row["rights_clearance"] == "pending"
               and not row["training_eligible"] for row in (a, b))
    counts = json.loads((target / "manifest.json").read_text())["review_dispositions"]
    assert counts == {"accepted": 0, "corrected": 1, "rejected": 0, "unresolved": 1}


def test_copied_prompt_template_cannot_be_selected_as_reference():
    generic = output(("factual change", ["H001"]), ("factual change", ["H002"]))
    result = select(case(), {"model1": generic, "model2": generic, "model3": generic})
    assert result["selected"] is None
    assert result["review_priority"] == "urgent"
    assert result["comparisons"]["model1"]["flags"]["prompt_placeholder"] == 2
    assert result["comparisons"]["model1"]["flags"]["duplicate_claim"] == 1


def test_jsonl_reader_preserves_unicode_line_separators_in_source(tmp_path):
    from scripts.prepare_specialization import read_rows, write_rows

    path = tmp_path / "changes.jsonl"
    rows = [{"diff": "+example\u2028next\u0085line"}, {"diff": "+other"}]
    write_rows(path, rows)
    assert read_rows(path) == rows
