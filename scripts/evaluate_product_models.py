"""Collect offline workload predictions for blinded review."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
from tempfile import TemporaryDirectory
from time import perf_counter

from italix.database import Database
from italix.engines import LOCAL_MODELS, LocalMLXEngine
from italix.hybrid_triage import ReviewRequired, patch_evidence
from italix.memory import MemoryService
from scripts.compare_local_models import TASKS

ROOT = Path(__file__).resolve().parents[1] / "data/phase10"
WORKLOADS = ("pr_summary", "semantic", "evidence", "chat", "memory")
PATCH_PROMPTS = {
    "pr_summary": TASKS["summary"],
    "semantic": "List the concrete before-and-after behavior shown by the changed lines. "
                "Give one bullet per file. If a claim needs unseen context, state that limitation.",
    "evidence": TASKS["evidence"],
}


class MeasuredEngine:
    def __init__(self, engine):
        self.engine = engine
        self.measurement = {}

    def generate(self, *args, **kwargs):
        result = self.engine.generate(*args, **kwargs)
        self.measurement = {
            "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
            **(result.metadata or {}),
        }
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=LOCAL_MODELS)
    parser.add_argument("workload", choices=WORKLOADS)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Run only one draft fixture")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    source = ROOT / ("pilot-v2.jsonl" if args.workload in PATCH_PROMPTS else "workload-fixtures.json")
    raw = source.read_bytes()
    digests = {name: digest for digest, name in (
        line.split() for line in (ROOT / "SHA256SUMS").read_text().splitlines()
    )}
    digest = hashlib.sha256(raw).hexdigest()
    if digests.get(source.name) != digest:
        parser.error("fixture checksum mismatch")
    bundle = json.loads(raw) if source.suffix == ".json" else None
    cases = bundle[args.workload] if bundle else [json.loads(line) for line in raw.splitlines()]
    reviewed = (bundle["review_status"] == "independently_reviewed" if bundle else
                all(case["review_status"] == "independently_reviewed" for case in cases))
    if not args.smoke and not reviewed:
        parser.error("full collection requires independent review")
    if args.smoke:
        cases = cases[:1]

    spec = LOCAL_MODELS[args.model]
    spec.cached_path()
    local = LocalMLXEngine(model_id=args.model)
    engine = MeasuredEngine(local)
    with TemporaryDirectory(prefix="italix-workload-eval-") as temporary, args.output.open("x") as output:
        def write(row):
            print(json.dumps(row), file=output, flush=True)

        write({"model": spec.repository, "revision": spec.revision, "workload": args.workload,
               "dataset_sha256": digest, "smoke": args.smoke, "temperature": 0,
               "max_tokens": 220 if args.workload == "memory" else 160,
               "mlx": version("mlx"), "mlx_lm": version("mlx-lm")})
        memory = None
        if args.workload == "memory":
            root = Path(temporary)
            db = Database(root / "memory.sqlite")
            db.initialize()
            memory = MemoryService(db, engine)
            path = root / "fixture.json"
            path.write_text(json.dumps(bundle["memory_sources"]))
            memory.import_discussions(str(path))
        try:
            for case in cases:
                started = perf_counter()
                engine.measurement = {}
                if memory is not None:
                    result = memory.query(case["question"])
                    prediction = {"answer": result["answer"], "abstained": result["abstained"],
                                  "cited": [row["metadata"]["citation"] for row in result["sources"]],
                                  "retrieved": [row["metadata"]["citation"] for row in result.get("retrieved", [])]}
                else:
                    system = PATCH_PROMPTS.get(args.workload, "Answer the developer request using the supplied facts.")
                    content = case.get("diff", case.get("prompt", ""))
                    if args.workload == "evidence":
                        try:
                            _, excerpts = patch_evidence(content)
                        except ReviewRequired:
                            excerpts = []
                        content = json.dumps(excerpts)
                    prediction = engine.generate(
                        [{"role": "system", "content": system}, {"role": "user", "content": content}],
                        max_tokens=160, temperature=0.0,
                    ).text
                write({"id": case["id"], "prediction": prediction, **engine.measurement,
                       "latency_ms": (perf_counter() - started) * 1000,
                       "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        finally:
            local.unload()


if __name__ == "__main__":
    main()
