import json

import pytest

from italix.semantic_review import semantic_diagnostics
from scripts import prepare_specialization as prep
from scripts.train_specialization import plan


def test_semantic_diagnostics_flags_unverified_numeric_and_repetitive_claims():
    fact = {"id": "timeout", "anchors": ["timeout"], "numbers": ["30 seconds"],
            "evidence_ids": ["H001", "H002"]}
    paths = {"H001": "api.py", "H002": "worker.py"}
    claims = [
        {"text": "Timeout is 90 seconds. Risk: high.", "evidence_ids": ["H001"]},
        {"text": "Timeout is 30 seconds. Risk: low.", "evidence_ids": ["H001", "invented"]},
        {"text": "Performance doubled.", "evidence_ids": []},
        {"text": "Performance doubled.", "evidence_ids": []},
    ]
    report = semantic_diagnostics(json.dumps({"claims": claims}), [fact], paths)
    assert report["valid_json"]
    assert report["missing_facts"] == ["timeout"]
    assert report["missing_files"] == ["api.py", "worker.py"]
    assert report["numeric_mismatches"] == [{"claim": 0, "fact": "timeout", "unexpected": ["90 second"]}]
    assert report["possible_contradictions"] == ["timeout"]
    assert report["conflicting_risk_labels"] == ["high", "low"]
    assert report["unknown_evidence_ids"] == ["invented"]
    assert report["security_or_performance_claims_for_review"] == [2, 3]
    assert report["duplicate_claims"] == 1
    claims[1]["evidence_ids"] = ["H001", "H002"]
    assert not semantic_diagnostics(json.dumps({"claims": claims}), [fact], paths)["missing_files"]


def record(repo, symbol):
    path = symbol + ".py"
    diff = (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
            f"@@ -1 +1 @@\n-{symbol} = False\n+{symbol} = True\n")
    return {"id": repo, "repo": repo, "base": "a" * 40, "head": "b" * 40,
            "source_url": "https://example.org/" + repo, "license": "MIT",
            "license_url": "https://example.org/license", "rights_reviewed": True,
            "review_status": "independently_reviewed", "reviewer": "reviewer-one", "diff": diff,
            "facts": [{"id": "change", "text": f"{symbol} is now enabled.", "anchors": [symbol],
                       "numbers": [], "evidence_ids": ["H001"]}]}


def test_exports_keep_test_out_of_mlx_and_exclude_cross_repo_copies(tmp_path, monkeypatch):
    rows = [record("org/train", "enabled"), record("org/valid", "logged"),
            record("org/test", "confirmed"), record("fork/copy", "confirmed")]
    split_map = {row["repo"]: {"family": row["repo"], "split": row["repo"].split("/")[1]}
                 for row in rows}
    split_map["fork/copy"]["split"] = "train"
    inventory, splits = tmp_path / "inventory.jsonl", tmp_path / "splits.json"
    prep.write_rows(inventory, rows)
    splits.write_text(json.dumps(split_map))

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize):
            assert tokenize
            return [1, 2]

    monkeypatch.setattr(prep, "tokenizer", Tokenizer)
    with pytest.raises(ValueError, match="stage A requires 100 reviewed cases"):
        prep.build(inventory, splits, tmp_path / "too-small", stage="A")
    output = tmp_path / "out"
    prep.build(inventory, splits, output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["counts"] == {"train": 1, "valid": 1, "test": 1}
    assert manifest["rejections"] == {"exact patch duplicate": 1}
    assert {p.name for p in (output / "mlx").iterdir()} == {"train.jsonl", "valid.jsonl"}
    assert prep.read_rows(output / "review/test.jsonl")[0]["id"] == "org/test"
    for relative, checksum in manifest["sha256"].items():
        assert prep.digest(output / relative) == checksum
    with pytest.raises(ValueError, match="2,000"):
        plan(output, tmp_path / "run")
    split_map["fork/copy"]["family"] = "org/test"
    splits.write_text(json.dumps(split_map))
    with pytest.raises(ValueError, match="multiple splits"):
        prep.build(inventory, splits, tmp_path / "second")


def test_training_plan_uses_verified_train_and_development_only(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import train_specialization as training

    spec = SimpleNamespace(revision="a" * 40, cached_path=lambda: tmp_path / "cached-model")
    monkeypatch.setitem(training.LOCAL_MODELS, "qwen-base", spec)
    corpus = tmp_path / "corpus"
    data = corpus / "mlx"
    data.mkdir(parents=True)
    (data / "train.jsonl").write_text("{}\n" * 2000)
    (data / "valid.jsonl").write_text("{}\n")
    manifest = {"format": "italix-specialization-v1", "base_revision": spec.revision,
                "max_sequence_tokens": prep.MAX_TOKENS, "counts": {"train": 2000, "valid": 1},
                "sha256": {f"mlx/{path.name}": prep.digest(path) for path in data.iterdir()}}
    (corpus / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "run"
    reviewed = plan(corpus, output)
    assert str(output / "data") in reviewed["command"]
    assert reviewed["recipe"]["mask_prompt"] and not reviewed["recipe"]["test"]
    one_pass = plan(corpus, output, one_pass=True)
    assert one_pass["recipe"]["iters"] == 2000
    assert one_pass["recipe"]["lora_parameters"]["keys"] == ["self_attn.q_proj", "self_attn.v_proj"]
    assert not output.exists()
    (data / "valid.jsonl").write_text("{}\n{}\n")
    with pytest.raises(ValueError, match="integrity check"):
        plan(corpus, output)


def test_calibration_annotation_is_exact_and_includes_every_changed_path():
    from scripts.evaluate_calibration import annotate

    case = {"paths": ["app.py", "worker.py"], "evidence": [
        {"id": "H001", "path": "app.py", "lines": "@@ -1 +1 @@\n-timeout = 15\n+timeout = 30"},
        {"id": "H002", "path": "worker.py", "lines": "@@ -1 +1 @@\n-}\n+];"},
    ]}
    row = annotate(case)
    assert row["annotation_status"] == "automated_extractive_pending_human_review"
    assert [(f["evidence_ids"], f["changed_line"]) for f in row["facts"]] == [
        (["H001"], "+timeout = 30"), (["H002"], "+];"),
    ]


def test_cuda_assistant_prefix_is_masked_and_misalignment_refused():
    from scripts.train_specialization_cuda import encode

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize
            return [1, 2] if add_generation_prompt else [1, 2, 3, 4]

    row = {"messages": [{"role": "user"}, {"role": "assistant"}]}
    result = encode(row, Tokenizer())
    assert result["labels"] == [-100, -100, 3, 4]
    assert result["attention_mask"] == [1, 1, 1, 1]

    class WrongTokenizer(Tokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return [7] if add_generation_prompt else [1, 2]

    with pytest.raises(ValueError, match="prefix"):
        encode(row, WrongTokenizer())


def test_calibration_reports_markdown_fences_as_separate_diagnostic():
    from scripts.evaluate_calibration import diagnose

    case = {"evidence": [{"id": "H001", "path": "app.py"}],
            "facts": [{"id": "timeout", "anchors": ["timeout"], "numbers": [],
                       "evidence_ids": ["H001"]}]}
    text = '```json\n{"claims":[{"text":"Update timeout","evidence_ids":["H001"]}]}\n```'
    report = diagnose(case, text)
    assert not report["diagnostics"]["valid_json"]
    assert report["fence_tolerant_diagnostics"]["valid_json"]
    assert not report["fence_tolerant_diagnostics"]["missing_facts"]


def test_context_compression_preserves_hunks_changed_lines_and_marks_gaps():
    from italix.context_compression import compress

    original = [
        {"id": "H001", "path": "app.py", "lines":
         "@@ -1,7 +1,7 @@ handler\n first\n second\n third\n-before\n+after\n fourth\n fifth\n sixth"},
        {"id": "H002", "path": "test.py", "lines":
         "@@ -1 +1 @@\n-old\n+new\n\\ No newline at end of file"},
    ]
    compact = compress(original)
    assert [(h["id"], h["path"]) for h in compact] == [("H001", "app.py"), ("H002", "test.py")]
    assert "handler" in compact[0]["lines"] and "second" not in compact[0]["lines"]
    assert "\n-before\n+after\n" in compact[0]["lines"]
    assert "omitted" in compact[0]["lines"]
    assert compact[1] == original[1]


def test_prose_citation_after_punctuation_stays_with_the_claim():
    from scripts.calibrate_context import score

    case = {"evidence": [{"id": "H001", "path": "app.py"}],
            "facts": [{"id": "timeout", "anchors": ["timeout"], "numbers": [],
                       "evidence_ids": ["H001"]}]}
    report = score(case, "Update the timeout. [H001] Another sentence.", "prose")
    assert report["has_citations"]
    assert not report["diagnostics"]["missing_facts"]


def test_training_export_requires_review_of_exact_compressed_context(tmp_path, monkeypatch):
    from hashlib import sha256

    from italix.context_compression import compress

    rows = [record("org/one", "enabled"), record("org/two", "logged"), record("org/three", "confirmed")]
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps({row["repo"]: {"family": row["repo"], "split": split}
                                      for row, split in zip(rows, prep.SPLITS)}))
    inventory = tmp_path / "review.jsonl"
    prep.write_rows(inventory, rows)
    monkeypatch.setattr(prep, "tokenizer", lambda: type("Tokenizer", (), {
        "apply_chat_template": lambda self, messages, tokenize: [1, 2]
    })())
    with pytest.raises(ValueError, match="compressed context requires review"):
        prep.build(inventory, split_path, tmp_path / "rejected", compressed_context=True)
    for row in rows:
        context = prep.messages_for(compress(prep.hunks(row["diff"])))[1]["content"]
        row["reviewed_context_sha256"] = sha256(context.encode()).hexdigest()
    prep.write_rows(inventory, rows)
    output = tmp_path / "approved"
    prep.build(inventory, split_path, output, compressed_context=True, max_tokens=2048)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["max_sequence_tokens"] == 2048
    assert manifest["context_mode"] == "compressed"
