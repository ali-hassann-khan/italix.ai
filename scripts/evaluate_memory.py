"""Compare local models on identical, redacted Developer Memory retrieval."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

from italix.database import Database
from italix.engines import LOCAL_MODELS, LocalMLXEngine
from italix.memory import MemoryService
from italix.privacy import contains_secret
from scripts.evaluate_product_models import MeasuredEngine

FIXTURE = Path(__file__).resolve().parents[1] / "data/phase11/memory-v1.json"
MODEL_IDS = tuple(LOCAL_MODELS)


def seed_sources(root, memory, sources):
    repo = root / "case-repo"
    repo.mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(root), "GIT_CONFIG_NOSYSTEM": "1"}

    def git(*args, **overrides):
        return subprocess.check_output(
            ["git", "-C", str(repo), "-c", "user.name=Fixtures", "-c",
             "user.email=fixtures@example.invalid", "-c", "core.hooksPath=/dev/null", *args],
            env={**env, **overrides}, text=True,
        ).strip()

    git("init", "-q", "--template=")
    by_key = {}
    discussions = []
    for source in sources:
        key, timestamp = source["key"], source["timestamp"]
        if source["kind"] == "disc":
            discussions.append({"message_id": key, "channel": "evaluation", "author": "Fixtures",
                                "timestamp": timestamp, "text": source["text"], "synthetic": True})
            by_key[key] = f"[DISCUSSION:{key}]"
        else:
            git("commit", "-q", "--allow-empty", "-m", f"Fixture {key}", "-m", source["text"],
                GIT_AUTHOR_DATE=timestamp, GIT_COMMITTER_DATE=timestamp)
            by_key[key] = f"[GIT:{git('rev-parse', 'HEAD')[:12]}]"
    memory.import_git(str(repo))
    path = root / "discussions.json"
    path.write_text(json.dumps(discussions))
    memory.import_discussions(str(path))
    return by_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", choices=MODEL_IDS, action="append")
    parser.add_argument("--answer-policy", action="store_true", help="Use current production answer retrieval")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    raw = FIXTURE.read_bytes()
    fixture = json.loads(raw)
    models = args.model or MODEL_IDS
    with TemporaryDirectory(prefix="italix-memory-eval-") as tmp, args.output.open("x") as output:
        root = Path(tmp)
        database = Database(root / "memory.sqlite")
        database.initialize()
        memory = MemoryService(database, None)
        labels = seed_sources(root, memory, fixture["sources"])
        packets = []
        for case in fixture["cases"]:
            question = case["question"]
            blocked = contains_secret(question)
            started = perf_counter()
            retrieve = memory.retrieve_for_answer if args.answer_policy else memory.search
            evidence = [] if blocked else retrieve(question)
            packets.append((case, evidence, blocked, (perf_counter() - started) * 1000))
        retrieval_digest = sha256(json.dumps([
            [case["id"], [[item["metadata"]["citation"], item["content"]] for item in evidence]]
            for case, evidence, _, _ in packets
        ]).encode()).hexdigest()

        def write(row):
            print(json.dumps(row), file=output, flush=True)

        write({"fixture_sha256": sha256(raw).hexdigest(), "retrieval_sha256": retrieval_digest,
               "models": {name: LOCAL_MODELS[name].revision for name in models},
               "review_status": fixture["review_status"],
               "retrieval": "answer_policy" if args.answer_policy else "phase11_search",
               "mlx": version("mlx"), "mlx_lm": version("mlx-lm"), "temperature": 0, "max_tokens": 220})
        for name in models:
            local = LocalMLXEngine(model_id=name)
            engine = MeasuredEngine(local)
            memory.local_engine = engine
            try:
                for case, evidence, blocked, retrieval_ms in packets:
                    engine.measurement = {}
                    expected = {labels[key] for key in case["expected_sources"]}
                    retrieved = {item["metadata"]["citation"] for item in evidence}
                    started = perf_counter()
                    result = ({"answer": "", "abstained": True, "reason": "credential_blocked", "sources": []}
                              if blocked else memory.answer_from_evidence(case["question"], evidence))
                    cited = {item["metadata"]["citation"] for item in result["sources"]}
                    write({"id": case["id"], "model": name, "category": case["category"],
                           "expected_sources": sorted(expected), "retrieved": sorted(retrieved),
                           "cited": sorted(cited), "abstained": result["abstained"],
                           "reason": result.get("reason"), "answer": result["answer"],
                           "retrieval_ms": retrieval_ms, "answer_ms": (perf_counter() - started) * 1000,
                           **engine.measurement})
            finally:
                local.unload()


if __name__ == "__main__":
    main()
