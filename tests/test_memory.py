from __future__ import annotations

import json
import subprocess

from fastapi.testclient import TestClient

from italix.api import create_app
from italix.engines import Generation


OPERATOR_HEADERS = {"Authorization": "Bearer test-operator-token-0123456789abcde"}


class MemoryEngine:
    configured = False
    model = "fake-local"

    def __init__(self):
        self.calls = 0

    def generate(self, messages, max_tokens, temperature, adapter_path=None):
        self.calls += 1
        assert adapter_path is None
        assert "Evidence:" in messages[-1]["content"]
        return Generation(
            "The plan catalog was cached to reduce repeated database reads. This is recorded evidence. [DISCUSSION:msg-redis]",
            "fake-local",
            20,
            15,
        )


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Demo Team")
    git(repo, "config", "user.email", "demo@example.invalid")
    decision = repo / "decision.md"
    decision.write_text("Redis caches plan catalog reads for 60 seconds with PostgreSQL fallback.\n")
    git(repo, "add", "decision.md")
    git(repo, "commit", "-q", "-m", "Add Redis plan cache with database fallback")
    return repo


def make_client(tmp_path):
    engine = MemoryEngine()
    app = create_app(
        database_path=tmp_path / "italix.db",
        local_engine=engine,
        gemini_engine=MemoryEngine(),
        groq_engine=MemoryEngine(),
    )
    return TestClient(app, headers=OPERATOR_HEADERS), engine


def test_git_and_discussion_import_search_query_and_deduplication(tmp_path):
    client, engine = make_client(tmp_path)
    repo = make_repo(tmp_path)
    imported = client.post(
        "/v1/memory/import/git",
        json={"repository_path": str(repo), "max_commits": 10, "include_diff_summary": True},
    )
    assert imported.status_code == 200
    assert imported.json()["imported"] == 1
    assert imported.json()["code_executed"] is False
    duplicate = client.post(
        "/v1/memory/import/git", json={"repository_path": str(repo), "max_commits": 10}
    )
    assert duplicate.json()["duplicates"] == 1

    discussions = tmp_path / "approved.jsonl"
    discussions.write_text(
        json.dumps(
            {
                "message_id": "msg-redis", "channel": "architecture",
                "timestamp": "2026-02-03T14:05:00Z", "author": "SRE",
                "text": "Redis reduces duplicate plan catalog reads; PostgreSQL remains the source of truth.",
                "thread_id": "thread-redis", "permalink": "https://example.invalid/msg-redis",
            }
        ) + "\n"
    )
    discussion_import = client.post(
        "/v1/memory/import/discussions", json={"file_path": str(discussions)}
    )
    assert discussion_import.json()["imported"] == 1

    results = client.get("/v1/memory/search", params={"q": "Why Redis plan caching?"}).json()["data"]
    assert {item["kind"] for item in results} == {"git", "discussion"}
    source = client.get(f"/v1/memory/sources/{results[0]['id']}")
    assert source.status_code == 200
    answer = client.post(
        "/v1/memory/query", json={"question": "Why did we introduce Redis caching?"}
    ).json()
    assert answer["abstained"] is False
    assert answer["cloud_used"] is False
    assert answer["sources"]
    assert "[DISCUSSION:msg-redis]" in answer["answer"]
    assert engine.calls == 1

    with app_database(client).connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM training_queue").fetchone()[0] == 0


def app_database(client):
    return client.app.state.database


def test_memory_abstains_without_evidence_and_does_not_call_model(tmp_path):
    client, engine = make_client(tmp_path)
    answer = client.post(
        "/v1/memory/query", json={"question": "Why did we select a lunar database?"}
    ).json()
    assert answer["abstained"] is True
    assert answer["sources"] == []
    assert engine.calls == 0


def test_git_import_rejects_non_repository(tmp_path):
    client, _ = make_client(tmp_path)
    response = client.post(
        "/v1/memory/import/git", json={"repository_path": str(tmp_path)}
    )
    assert response.status_code == 400


def test_memory_redaction_retention_and_strict_citations(tmp_path):
    from datetime import datetime, timedelta, timezone

    client, engine = make_client(tmp_path)
    discussion = tmp_path / "approved.jsonl"
    discussion.write_text(json.dumps({
        "message_id": "cache", "author": "SRE", "channel": "architecture",
        "timestamp": "2025-03-01T00:00:00Z", "text": "Cache rollout. API_KEY=sample-local-key",
        "permalink": "https://user:pass@example.com/cache",
    }))
    source_id = client.post("/v1/memory/import/discussions", json={"file_path": str(discussion)}).json()["source_ids"][0]
    result = client.post("/v1/memory/search", json={"question": "Cache rollout"}).json()["data"][0]
    assert "[REDACTED]" in result["content"] and "sample-local-key" not in result["content"]
    assert result["reference"] is None
    assert client.post("/v1/memory/query", json={"question": "API_KEY=example-secret"}).status_code == 422
    assert engine.calls == 0
    result = client.post("/v1/memory/query", json={"question": "Cache rollout"}).json()
    assert result["reason"] == "invalid_or_missing_citations" and result["sources"] == []

    with app_database(client).connect() as connection:
        connection.execute("UPDATE memory_sources SET content='Cache rotation. token=legacy-fixture' WHERE id=?", (source_id,))
    assert client.post("/v1/memory/redact").json() == {"redacted": 1}
    result = client.post("/v1/memory/search", json={"question": "rotation"}).json()["data"][0]
    assert result["id"] == source_id and "legacy-fixture" not in result["content"]
    with app_database(client).connect() as connection:
        old = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        connection.execute("UPDATE memory_sources SET imported_at=? WHERE id=?", (old, source_id))
    assert client.post("/v1/memory/retention/purge", json={"retention_days": 90}).json() == {"deleted": 1}
    assert client.get(f"/v1/memory/sources/{source_id}").status_code == 404
    assert not client.post("/v1/memory/search", json={"question": "rotation"}).json()["data"]
    assert client.delete(f"/v1/memory/sources/{source_id}", headers={"Authorization": ""}).status_code == 401


def test_answer_rejects_single_generic_match_without_hiding_search_results(tmp_path):
    client, engine = make_client(tmp_path)
    source = tmp_path / "discussion.jsonl"
    source.write_text(json.dumps({"message_id": "earth", "channel": "decision", "author": "SRE",
                                  "timestamp": "2025-01-01T13:00:00Z",
                                  "text": "PostgreSQL is the database for the plan catalog."}))
    assert client.post("/v1/memory/import/discussions", json={"file_path": str(source)}).status_code == 200
    question = {"question": "Which lunar telemetry database was selected?"}
    assert client.post("/v1/memory/search", json=question).json()["data"]
    assert client.post("/v1/memory/query", json=question).json()["abstained"]
    assert engine.calls == 0


def test_answer_retrieval_matches_inflected_terms_for_repository_history(tmp_path):
    client, _ = make_client(tmp_path)
    memory = client.app.state.memory
    export = tmp_path / "history.jsonl"
    export.write_text(json.dumps({"message_id": "session", "channel": "history", "author": "SRE",
                                  "timestamp": "2025-01-06T13:00:00Z",
                                  "text": "Rename session_cookie_ttl to session_ttl in the session loader. The lifetime remains 900 seconds."}))
    memory.import_discussions(str(export))
    rows = memory.retrieve_for_answer("Which session setting was renamed, and did its duration change?")
    assert [row["metadata"]["citation"] for row in rows] == ["[DISCUSSION:session]"]
