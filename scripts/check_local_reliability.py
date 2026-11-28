"""Measure a bounded offline gateway burst, model switches and cache failure recovery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import secrets
from tempfile import TemporaryDirectory
from threading import Barrier
from time import perf_counter

from fastapi.testclient import TestClient

from italix.engines import LOCAL_MODELS, LocalMLXEngine
from scripts.phase9_model_smoke import machine_memory

MESSAGES = [{"role": "user", "content": "Name the primary purpose of a version-control commit in one sentence."}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ballast-mib", type=int, choices=(0, 128, 256), default=0)
    parser.add_argument("--sustained", type=int, choices=(0, 80, 160), default=0,
                        help="Run a fixed-size sustained four-client sample after the burst")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["GEMINI_API_KEY"] = os.environ["GROQ_API_KEY"] = ""
    os.environ.pop("ITALIX_HYBRID_MODEL", None)
    os.environ["ITALIX_OPERATOR_TOKEN"] = token = secrets.token_urlsafe(32)
    os.environ["ITALIX_VIEWER_TOKEN"] = secrets.token_urlsafe(32)

    with TemporaryDirectory(prefix="italix-reliability-") as temporary, args.output.open("x") as report:
        root = Path(temporary)
        os.environ["ITALIX_DB_PATH"] = str(root / "initial.sqlite")
        from italix.api import create_app

        def write(row):
            print(json.dumps(row), file=report, flush=True)

        def generate(client):
            started = perf_counter()
            response = client.post("/v1/chat/completions", json={
                "model": "italix-auto", "workload": "general", "messages": MESSAGES,
                "max_tokens": 32, "temperature": 0, "capture": False, "allow_cloud": False,
            })
            data = response.json()
            return {"status": response.status_code, "elapsed_ms": (perf_counter() - started) * 1000,
                    "usage": data.get("usage"), "route": data.get("italix", {}).get("route")}

        engines = {name: LocalMLXEngine(model_id=name) for name in LOCAL_MODELS}
        base = engines["qwen-base"]
        ballast = bytearray(args.ballast_mib * 1024**2)
        for i in range(0, len(ballast), 4096):
            ballast[i] = (i // 4096) % 255 + 1
        write({"phase": "before", "ballast_mib": args.ballast_mib, **machine_memory()})
        try:
            app = create_app(database_path=root / "burst.sqlite", local_engine=base)
            headers = {"Authorization": f"Bearer {token}"}
            with TestClient(app, headers=headers) as client:
                for label in ("cold", "warm", "warm", "warm"):
                    row = generate(client)
                    write({"phase": label, **row})
                    assert row["status"] == 200 and row["route"] == "local"
                barrier = Barrier(4)

                def burst(_):
                    barrier.wait(timeout=15)
                    return generate(client)

                start = perf_counter()
                with ThreadPoolExecutor(max_workers=4) as pool:
                    responses = list(pool.map(burst, range(4)))
                write({"phase": "burst", "total_ms": (perf_counter() - start) * 1000, "responses": responses})
                assert all(row["status"] == 200 and row["route"] == "local" for row in responses)
                if args.sustained:
                    start = perf_counter()
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        sustained = list(pool.map(lambda _: generate(client), range(args.sustained)))
                    write({"phase": "sustained", "concurrency": 4,
                           "total_ms": (perf_counter() - start) * 1000, "responses": sustained})
                    assert all(row["status"] == 200 and row["route"] == "local" for row in sustained)
                for model_id in ("qwen3-4b", "qwen-coder-7b", "qwen-base"):
                    start = perf_counter()
                    result = engines[model_id].generate(MESSAGES, max_tokens=32, temperature=0.0)
                    write({"phase": "switch", "model": model_id, "elapsed_ms": (perf_counter() - start) * 1000,
                           "output_tokens": result.completion_tokens, **(result.metadata or {}), **machine_memory()})

                with TestClient(create_app(database_path=root / "missing.sqlite",
                                           local_engine=LocalMLXEngine(root / "absent")), headers=headers) as missing:
                    failed = generate(missing)
                    deterministic = missing.post("/v1/chat/completions", json={
                        "model": "pr-summary-deterministic-v1", "workload": "pr_summary",
                        "messages": [{"role": "user", "content": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-X = 1\n+X = 2"}],
                    })
                recovered = generate(client)
                write({"phase": "recovery", "missing": failed["status"],
                       "deterministic": deterministic.status_code, "subsequent": recovered})
                assert (failed["status"], deterministic.status_code, recovered["status"]) == (503, 200, 200)
        finally:
            for engine in engines.values():
                engine.unload()
            write({"phase": "after", **machine_memory()})


if __name__ == "__main__":
    main()
