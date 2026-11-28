from decimal import Decimal
import json
from types import SimpleNamespace

import httpx
import pytest

from scripts.gemini_annotation import Gemini, connect, dispatch, load_authorized, validate
from scripts.prepare_specialization import digest


def example(key):
    fact = {"id": "F001", "hunk": "H001", "path": "app.py", "kind": ["value"],
            "locations": {"old": [{"line": 1, "text": "TIMEOUT = 10"}],
                          "new": [{"line": 1, "text": "TIMEOUT = 15"}]},
            "syntax": {"old": [], "new": []}, "symbols": []}
    return {"id": key, "repo": "org/repo", "source_url": "https://example.com/commit", "base": "a" * 40,
            "head": "b" * 40, "license": "MIT", "license_url": "https://example.com/license",
            "notices": {}, "paths": ["app.py"], "facts": [fact], "verified_source_lines": 2}


TEXT = '{"claims":[{"text":"TIMEOUT changed from 10 to 15.","evidence_ids":["F001"]}]}'


def test_structured_request_counts_first_and_records_reasoning_usage():
    def handler(request):
        assert request.headers["x-goog-api-key"] == "mock-only"
        payload = json.loads(request.content)
        if request.url.path.endswith(":countTokens"):
            assert "responseJsonSchema" in payload["generateContentRequest"]["generationConfig"]
            return httpx.Response(200, json={"totalTokens": 18})
        assert request.url.path.endswith(":generateContent")
        return httpx.Response(200, json={"candidates": [{"finishReason": "STOP", "content": {
            "parts": [{"text": TEXT}]}}], "usageMetadata": {"promptTokenCount": 18,
            "candidatesTokenCount": 12, "thoughtsTokenCount": 2}})
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="https://example.test",
                      headers={"x-goog-api-key": "mock-only"}) as client:
        api = Gemini(client, "gemini-test", 50)
        assert api("source", 17)["error"] == "input_token_ceiling"
        result = api("source", 18)
    assert (result["input"], result["output"]) == (18, 14)
    assert not validate(result["text"], example("case"))[1]
    assert "numeric_contradiction" in validate(TEXT.replace("10 to 15", "15 to 10"), example("case"))[1]


def test_provider_and_source_gate_is_bound_to_exact_manifest(tmp_path):
    manifest, review = tmp_path / "data", tmp_path / "approval"
    cases = [example(str(i)) for i in range(20)]
    manifest.write_text("\n".join(json.dumps(row) for row in [
        {"kind": "gemini-source-v1", "size": 20}, *cases]) + "\n")
    approved = {"manifest_sha256": digest(manifest), "model": "gemini-test",
                "provider": "gemini-developer-api-paid", "purpose": "local-qwen-student-training",
                "provider_permission_reference": "fixture permission",
                "source_rights_review_reference": "fixture approval", "approved_case_ids": [r["id"] for r in cases]}
    review.write_text(json.dumps(approved))
    assert len(load_authorized(manifest, review, "gemini-test")) == 20
    approved.pop("provider_permission_reference")
    review.write_text(json.dumps(approved))
    with pytest.raises(ValueError, match="scoped provider"):
        load_authorized(manifest, review, "gemini-test")


def test_budget_resume_accounts_for_interruption_and_does_not_retry_completed(tmp_path):
    source, approval = tmp_path / "input", tmp_path / "approval"
    source.write_text("fixture source")
    approval.write_text("fixture approval")
    args = SimpleNamespace(manifest=source, approval=approval, model="gemini-test",
                           input_rate=Decimal("1"), output_rate=Decimal("1"),
                           max_input=10, max_output=10, budget=Decimal("0.000030"),
                           concurrency=2, retries=0)
    called = []
    def request(content, limit):
        called.append((content, limit))
        return {"text": TEXT, "input": 9, "output": 9, "finish": "STOP"}
    with connect(tmp_path / "state") as db:
        with db:
            db.execute("INSERT INTO attempts(case_id,debit,status) VALUES ('prior',10,'in_flight')")
        cases = [example("a"), example("b")]
        assert dispatch(db, cases, request, args) == 1
        assert len(called) == 1
        assert db.execute("SELECT status FROM attempts WHERE case_id='prior'").fetchone()[0] == "interrupted_unknown"
        args.budget = Decimal("0.000100")
        assert dispatch(db, cases, request, args) == 2
        assert len(called) == 2
        assert all(not json.loads(r[0])["training_eligible"] for r in db.execute("SELECT payload FROM results"))


def test_rate_limit_mock_honors_retry_after():
    with httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(429, headers={"Retry-After": "7"})),
            base_url="https://example.test") as client:
        result = Gemini(client, "gemini-test", 10)("source", 100)
    assert result == {"error": "http_429", "retry": True, "uncertain": False, "retry_after": 7}


def test_retry_records_attempts_and_charges_ambiguous_provider_response(tmp_path, monkeypatch):
    from itertools import count

    import scripts.gemini_annotation as module

    clock = count()
    monkeypatch.setattr(module, "monotonic", lambda: next(clock))
    source, approval = tmp_path / "input", tmp_path / "approval"
    source.write_text("fixture source")
    approval.write_text("fixture approval")
    args = SimpleNamespace(manifest=source, approval=approval, model="gemini-test",
                           input_rate=Decimal("1"), output_rate=Decimal("1"),
                           max_input=10, max_output=10, budget=Decimal("0.000100"),
                           concurrency=1, retries=1)
    outcomes = iter([{"error": "http_503", "retry": True, "uncertain": True, "retry_after": 1},
                     {"text": TEXT, "input": 4, "output": 5, "finish": "STOP"}])
    with connect(tmp_path / "state") as db:
        assert dispatch(db, [example("a")], lambda *_: next(outcomes), args) == 1
        assert [tuple(row) for row in db.execute("SELECT status,debit FROM attempts ORDER BY attempt")] == [
            ("http_503", 20), ("reported", 9)]


def test_free_zero_cost_quota_pause_survives_resume_without_retry(tmp_path, monkeypatch):
    import scripts.gemini_annotation as module

    manifest, approval = tmp_path / "source", tmp_path / "receipt"
    manifest.write_text("source")
    approval.write_text("receipt")
    args = SimpleNamespace(manifest=manifest, approval=approval, model="gemini-test", tier="free",
                           input_rate=Decimal(0), output_rate=Decimal(0), budget=Decimal(0),
                           max_input=20, max_output=20, concurrency=4, retries=10)
    clock = [1000]
    monkeypatch.setattr(module, "time", lambda: clock[0])
    hits = []
    def request(*_):
        hits.append(1)
        return {"error": "http_429", "retry": True, "retry_after": 120, "uncertain": False}
    path = tmp_path / "checkpoint"
    with connect(path) as db:
        assert dispatch(db, [example("a"), example("b")], request, args) == 0
    with connect(path) as db:
        assert dispatch(db, [example("a"), example("b")], request, args) == 0
        assert len(hits) == 1
        assert db.execute("SELECT resume_after FROM quota_pause").fetchone()[0] == 1120
        clock[0] = 1121
        assert dispatch(db, [example("a")], lambda *_: {
            "text": TEXT, "input": 5, "output": 4, "finish": "STOP"}, args) == 1
        assert list(db.execute("SELECT debit FROM attempts"))[1][0] == 0
        assert not db.execute("SELECT 1 FROM quota_pause").fetchone()
        args.tier = "paid"
        with pytest.raises(ValueError, match="checkpoint recipe"):
            dispatch(db, [example("a")], request, args)


def test_free_authorization_requires_billing_privacy_key_match_and_no_email(tmp_path):
    from hashlib import sha256

    from scripts.gemini_annotation import verify_free_key

    manifest, review = tmp_path / "source", tmp_path / "receipt"
    cases = [example(str(i)) for i in range(20)]
    def prepare():
        manifest.write_text("\n".join(json.dumps(row) for row in [
            {"kind": "gemini-source-v1", "size": 20}, *cases]))
    prepare()
    approved = {"manifest_sha256": digest(manifest), "model": "gemini-test",
                "provider": "gemini-developer-api-free", "purpose": "local-qwen-student-training",
                "provider_permission_reference": "fixture permission",
                "source_rights_review_reference": "fixture approval", "approved_case_ids": [r["id"] for r in cases],
                "api_key_sha256": sha256(b"mock-key").hexdigest(),
                "unpaid_data_handling_approval_reference": "fixture unpaid approval",
                "public_nonsensitive_source_reviewed": True, "billing_disabled": False}
    review.write_text(json.dumps(approved))
    with pytest.raises(ValueError, match="billing or privacy"):
        load_authorized(manifest, review, "gemini-test", "free")
    approved["billing_disabled"] = True
    review.write_text(json.dumps(approved))
    assert len(load_authorized(manifest, review, "gemini-test", "free")) == 20
    verify_free_key("mock-key", review)
    with pytest.raises(ValueError, match="key does not match"):
        verify_free_key("different-key", review)
    cases[0]["facts"][0]["locations"]["new"][0]["text"] = "Email: contributor@example.com"
    prepare()
    approved["manifest_sha256"] = digest(manifest)
    review.write_text(json.dumps(approved))
    with pytest.raises(ValueError, match="email address"):
        load_authorized(manifest, review, "gemini-test", "free")


def test_evaluation_checkpoint_retains_claims_without_training_eligibility(tmp_path):
    manifest = tmp_path / "source"
    manifest.write_text("fixture")
    args = SimpleNamespace(manifest=manifest, model="gemini-test", tier="free", evaluation_only=True,
                           key_fingerprint="fixture", budget=Decimal(0), input_rate=Decimal(0),
                           output_rate=Decimal(0), max_input=20, max_output=20, concurrency=1, retries=0)
    with connect(tmp_path / "state") as db:
        assert dispatch(db, [example("a")], lambda *_: {
            "text": TEXT, "input": 10, "output": 10, "finish": "STOP"}, args) == 1
        saved = json.loads(db.execute("SELECT payload FROM results").fetchone()[0])
        assert saved["status"] == "screen_passed" and not saved["training_eligible"]
        assert saved["claims"][0]["evidence_ids"] == ["H001"]
        assert saved["summary"] == "TIMEOUT changed from 10 to 15."
        assert json.loads(db.execute("SELECT config FROM metadata").fetchone()[0])["purpose"] == "evaluation_only"


def test_semantic_review_scores_verdicts_and_excludes_out_of_scope_cases(tmp_path):
    from scripts.gemini_annotation import score_review, write_review_packet

    references = tmp_path / "references"
    references.write_text("\n".join(json.dumps(row) for row in [
        {"id": "a", "source_url": "https://example.test", "evidence": [],
         "ai_review": {"claims": [{"text": "Source fact", "evidence_ids": ["H001"]}]}},
        {"id": "b", "disposition": "rejected"},
    ]))
    review = tmp_path / "review"
    with connect(tmp_path / "state") as db:
        with db:
            for key in ("a", "b"):
                db.execute("INSERT INTO results VALUES (?,?)", (key, json.dumps({
                    "id": key, "summary": "candidate", "claims": [{"text": "candidate"}], "flags": []})))
        write_review_packet(db, review, references)
        data = json.loads(review.read_text())
        data.update(reviewer="fixture", review_type="ai_assisted")
        data["cases"][0].update(verdicts=["unsupported"], covered_fact_indices=[], usable=False)
        review.write_text(json.dumps(data))
        metrics = score_review(db, references, review)
        assert metrics["cases"] == 1
        assert metrics["factual_precision"] == metrics["required_fact_recall"] == 0
        assert metrics["counts"]["unsupported"] == 1 and metrics["counts"]["usable"] == 0


def test_key_failover_only_on_invalid_credentials():
    from scripts.gemini_annotation import CredentialFailover, parse_keys

    keys = parse_keys(" first, , second, first, third ")
    assert keys == ("first", "second", "third")
    seen = []

    def handler(request):
        key = request.headers["x-goog-api-key"]
        seen.append(key)
        if key == "first":
            return httpx.Response(400, json={"error": {"details": [{"reason": "API_KEY_INVALID"}]}})
        return httpx.Response(429)

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="https://example.test") as client:
        request = CredentialFailover(client, "gemini-test", 10, keys)
        assert request("example", 20)["error"] == "http_429"
    assert seen == ["first", "second"]


def test_two_pass_flags_an_omitted_file_and_accounts_for_all_stages():
    from scripts.gemini_annotation import EXTRACT_PROMPT, SYNTHESIS_PROMPT, two_pass

    case = example("a")
    case["paths"].append("other.py")
    case["facts"].append(case["facts"][0] | {"id": "F002", "path": "other.py"})
    calls = []
    def request(content, limit, prompt):
        packet = json.loads(content)
        calls.append(prompt)
        if prompt == SYNTHESIS_PROMPT:
            assert {f["path"] for f in packet["file_findings"]} == set(case["paths"])
            text = TEXT
        else:
            text = TEXT.replace("F001", packet[0]["id"])
        return {"text": text, "finish": "STOP", "input": 10, "output": 5}

    result = two_pass(request, case, 100)
    assert calls == [EXTRACT_PROMPT, EXTRACT_PROMPT, SYNTHESIS_PROMPT]
    assert result["flags"] == ["extracted_evidence_not_covered"]
    assert result["coverage"] == {"extracted_findings": 2, "citation_covered": 1}
    assert (result["input"], result["output"]) == (30, 15)


def test_quota_uses_larger_provider_delay_and_expired_key_has_safe_reason():
    from scripts.gemini_annotation import failure

    quota = httpx.Response(429, headers={"Retry-After": "12"}, json={"error": {"details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "90.5s"}
    ]}})
    assert failure(quota)["retry_after"] == 90.5
    expired = httpx.Response(400, json={"error": {"details": [{"reason": "API_KEY_EXPIRED"}]}})
    assert failure(expired)["credential_reason"] == "API_KEY_EXPIRED"
