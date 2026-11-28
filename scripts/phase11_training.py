"""Operator-only launcher for the one reviewed Phase 11 training plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from tempfile import TemporaryDirectory

import httpx

from italix.database import Database
from italix.engines import LOCAL_MODELS, cached_qwen_path
from italix.training_jobs import TrainingJobRunner

ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = ROOT / "artifacts/phase11/training-preflight"
DATABASE = PLAN_DIR / "lifecycle.sqlite"
JOB_ID = 1
CANDIDATE = "phase11-pr-lifecycle-25802ca2"
REVIEW_DIGEST = "2f6711af76c235963778291537aac62ef4bb18fe39288c77979f078efda78507"
MODEL_SHA256 = {
    "model.safetensors": "0979f33d1bc58afcf696d13f57977644e7b11a6f0eec3e631d8e9463d18c0717",
    "config.json": "636d3e2a15e8914b8cf82b05cc2288a811f9bd93c3bf1afc00cab701a70b47c0",
    "model.safetensors.index.json": "6b98634d5044f0e2ad45228a374f8445904e571f1082392d08a6ce54f5d517ca",
    "tokenizer.json": "a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf",
    "tokenizer_config.json": "f7c61e32b7a17d19bf8e7037dcb74079a833e53ea9801f24008cac68458f03b7",
    "added_tokens.json": "58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b",
    "special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
}
JOB_DIR = ROOT / f"artifacts/training_jobs/{JOB_ID}"
ADAPTER_DIR = ROOT / "artifacts/adapters" / CANDIDATE


def file_sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify():
    if not DATABASE.is_file():
        raise ValueError("isolated Phase 11 database is missing")
    manifest = json.loads((PLAN_DIR / "review.json").read_text())
    runner = TrainingJobRunner(Database(DATABASE), ROOT)
    review = runner.review(JOB_ID)
    if (review["review_digest"] != REVIEW_DIGEST or review["candidate_id"] != CANDIDATE
            or any(manifest.get(key) != value for key, value in review.items())):
        raise ValueError("review manifest or planned job has changed")
    if JOB_DIR.exists() or ADAPTER_DIR.exists():
        raise ValueError("job or adapter artifacts already exist; preserve them and inspect job status")
    fixture = ROOT / "data/phase11/training-preflight.json"
    if file_sha256(fixture) != manifest["fixture_sha256"]:
        raise ValueError("Phase 11 synthetic fixture checksum mismatch")
    expected = manifest["export_sha256"]
    with TemporaryDirectory(prefix="italix-phase11-verify-") as temporary:
        export = Path(temporary)
        counts = runner._export_data(JOB_ID, export, runner._spec("pr_summary"))
        if counts != (66, 6) or counts != (manifest["train_examples"], manifest["validation_examples"]):
            raise ValueError("unexpected export counts")
        for name in ("train.jsonl", "valid.jsonl"):
            if not (file_sha256(export / name) == file_sha256(PLAN_DIR / "reviewed-export" / name) == expected[name]):
                raise ValueError(f"reviewed export checksum mismatch for {name}")
    spec = LOCAL_MODELS["qwen-base"]
    model_path = spec.cached_path()
    config = json.loads((model_path / "config.json").read_text())
    if (spec.revision != review["model_revision"] or config.get("quantization", {}).get("bits") != 4
            or {p.name for p in model_path.glob("model*.safetensors")} != {"model.safetensors"}):
        raise ValueError("cached model revision, configuration or weight files differ from the plan")
    for name, expected_digest in MODEL_SHA256.items():
        if file_sha256(model_path / name) != expected_digest:
            raise ValueError(f"cached model checksum mismatch for {name}")
    return manifest


def request(client, method, path, json_body=None):
    response = client.request(method, path, json=json_body)
    if response.is_error:
        body = response.json()
        raise ValueError(f"HTTP {response.status_code}: {body.get('detail', 'request rejected')}")
    return response.json()


def read_job(client):
    jobs = request(client, "GET", "/v1/training/jobs?limit=200")["data"]
    job = next((j for j in jobs if j["id"] == JOB_ID and j["candidate_id"] == CANDIDATE), None)
    if job is None:
        raise ValueError("approved job is absent from this server")
    return job


def watch(client):
    while True:
        job = read_job(client)
        print(json.dumps(job, indent=2), flush=True)
        if job["status"] in {"completed", "failed"}:
            return
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "serve", "start", "status", "logs",
                                            "interrupt", "recover", "preview", "evaluate",
                                            "qualify", "qualification", "watch"))
    parser.add_argument("--confirm")
    parser.add_argument("--patch", type=Path)
    args = parser.parse_args()
    action = args.action
    if action == "verify":
        print(json.dumps(verify(), indent=2))
        return
    if action == "logs":
        os.execl("/usr/bin/tail", "tail", "-n", "+1", "-F", str(JOB_DIR / "training.log"))
    token = os.getenv("ITALIX_OPERATOR_TOKEN", "")
    if len(token) < 32:
        parser.error("set ITALIX_OPERATOR_TOKEN to a local token of at least 32 characters")
    port = int(os.getenv("ITALIX_TRAINING_PORT", "8011"))
    if not 1024 <= port <= 65535:
        parser.error("ITALIX_TRAINING_PORT must be between 1024 and 65535")
    if action == "serve":
        if not DATABASE.is_file():
            parser.error("isolated Phase 11 database is missing")
        viewer = os.getenv("ITALIX_VIEWER_TOKEN", "")
        if len(viewer) < 32 or token == viewer:
            parser.error("set a distinct ITALIX_VIEWER_TOKEN of at least 32 characters")
        if cached_qwen_path().resolve() != LOCAL_MODELS["qwen-base"].cached_path().resolve():
            parser.error("the ordinary gateway cache and the pinned reviewed model must resolve to the same snapshot")
        os.environ.update(ITALIX_DB_PATH=str(DATABASE))
        os.execv(sys.executable, [sys.executable, "-m", "uvicorn", "italix.api:app",
                                 "--host", "127.0.0.1", "--port", str(port), "--workers", "1"])
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=600,
                      headers={"Authorization": f"Bearer {token}"}) as client:
        job_path = f"/v1/training/jobs/{JOB_ID}"
        model_path = f"/v1/models/{CANDIDATE}"
        if action == "start":
            if args.confirm != CANDIDATE:
                parser.error(f"start requires --confirm {CANDIDATE}")
            manifest = verify()
            live_review = request(client, "GET", job_path + "/review")
            if any(manifest.get(key) != value for key, value in live_review.items()) or live_review.get("review_digest") != REVIEW_DIGEST:
                raise ValueError("the local server does not have the verified reviewed plan")
            result = request(client, "POST", job_path + "/execute", {"review_digest": REVIEW_DIGEST})
        elif action == "watch":
            watch(client)
            return
        elif action in {"status", "interrupt", "recover"}:
            job = read_job(client)
            if action == "status":
                result = job
            elif action == "interrupt":
                result = request(client, "POST", job_path + "/interrupt")
            else:
                if job["status"] != "running":
                    raise ValueError(f"job is {job['status']}; recovery only applies to an interrupted running job")
                result = request(client, "POST", "/v1/training/recover")
        else:
            models = request(client, "GET", "/v1/models")["data"]
            model = next((m for m in models if m["id"] == CANDIDATE and m["workload"] == "pr_summary"), None)
            if model is None or model["kind"] != "adapter" or model["active"]:
                raise ValueError("the completed candidate must exist and remain inactive")
            if action == "preview":
                if args.patch is None:
                    parser.error("preview requires --patch PATH")
                result = request(client, "POST", model_path + "/preview", {"content": args.patch.read_text()})
            elif action in {"evaluate", "qualify"}:
                flag = "false" if action == "evaluate" else "true"
                result = request(client, "POST", model_path + f"/evaluation?record={flag}",
                                 {"workload": "pr_summary", "split": "valid"})
            else:
                result = {"candidate_id": CANDIDATE, "status": model["status"],
                          "qualified": model["status"] == "qualified", "active": bool(model["active"])}
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, PermissionError, httpx.HTTPError) as exc:
        raise SystemExit(f"Phase 11 launcher: {exc}") from None
