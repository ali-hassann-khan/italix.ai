"""Prepare one synthetic PR-summary lifecycle plan using real cached inference."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets

from fastapi.testclient import TestClient

from italix.engines import LocalMLXEngine

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for the isolated database and review manifest")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(mode=0o700)
    for key in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        os.environ[key] = ""
    os.environ["ITALIX_OPERATOR_TOKEN"] = token = secrets.token_urlsafe(32)
    os.environ["ITALIX_VIEWER_TOKEN"] = secrets.token_urlsafe(32)
    os.environ["ITALIX_DB_PATH"] = str(out / "lifecycle.sqlite")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from italix.api import create_app

    engine = LocalMLXEngine(model_id="qwen-base")
    app = create_app(local_engine=engine)
    fixture = ROOT / "data/phase11/training-preflight.json"

    def verified(response):
        response.raise_for_status()
        return response.json()

    try:
        with TestClient(app, headers={"Authorization": f"Bearer {token}"}) as client:
            for row in json.loads(fixture.read_text()):
                body = verified(client.post("/v1/chat/completions", json={
                    "model": "qwen-base", "workload": "pr_summary",
                    "messages": [{"role": "user", "content": row["diff"]}],
                    "temperature": 0, "max_tokens": 160, "capture": True, "allow_cloud": False,
                }))
                trace = body["italix"]["trace_id"]
                feedback = verified(client.post(f"/v1/traces/{trace}/feedback", json={
                    "action": "correct", "corrected_response": row["correction"],
                }))
                if not feedback["queued"]:
                    raise RuntimeError("synthetic correction did not enter the queue")
            candidate_id = f"phase11-pr-lifecycle-{secrets.token_hex(4)}"
            planned = verified(client.post("/v1/training/jobs", json={
                "workload": "pr_summary", "candidate_id": candidate_id,
            }))
            reviewed = verified(client.get(f"/v1/training/jobs/{planned['id']}/review"))
            runner = app.state.training_runner
            train, validation = runner._export_data(planned["id"], out / "reviewed-export",
                                                     runner._spec("pr_summary"))
            manifest = {**reviewed, "fixture_sha256": sha256(fixture.read_bytes()).hexdigest(),
                        "train_examples": train, "validation_examples": validation,
                        "export_sha256": {name: sha256((out / "reviewed-export" / name).read_bytes()).hexdigest()
                                          for name in ("train.jsonl", "valid.jsonl")}}
            (out / "review.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps(manifest, indent=2))
    finally:
        engine.unload()


if __name__ == "__main__":
    main()
