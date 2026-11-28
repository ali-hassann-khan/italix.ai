"""Run a matched offline evaluation on one verified Phase 13 split."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
from time import perf_counter

from italix.engines import LOCAL_MODELS, LocalMLXEngine
from italix.privacy import redact
from italix.semantic_review import semantic_diagnostics
from scripts.prepare_specialization import digest, read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", choices=LOCAL_MODELS, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--confirm-test-manifest")
    args = parser.parse_args()
    if args.adapter and args.model != "qwen-base":
        parser.error("the specialized adapter is only compatible with qwen-base")
    adapter = args.adapter.resolve() if args.adapter else None
    if adapter and not all((adapter / name).is_file() for name in ("adapters.safetensors", "adapter_config.json")):
        parser.error("adapter is missing local artifacts")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    manifest_path = args.corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["format"] != "italix-specialization-v1":
        parser.error("expected a reviewed Phase 13 export")
    if adapter:
        plan_path = adapter.parent / "plan.json"
        plan = json.loads(plan_path.read_text()) if plan_path.is_file() else {}
        if (plan.get("manifest_sha256") != digest(manifest_path)
                or plan.get("base_revision") != LOCAL_MODELS["qwen-base"].revision):
            parser.error("adapter does not belong to this corpus and pinned base")
    if args.split == "test" and args.confirm_test_manifest != digest(manifest_path):
        parser.error("Phase 13 test access requires the frozen manifest digest")
    relative = f"review/{args.split}.jsonl"
    path = args.corpus / relative
    if path.is_symlink() or digest(path) != manifest["sha256"][relative]:
        parser.error("review split integrity check failed")
    LOCAL_MODELS[args.model].cached_path()
    model = LocalMLXEngine(model_id=args.model)
    failures = 0
    with args.output.open("x") as stream:
        def write(value):
            print(json.dumps(value, ensure_ascii=False), file=stream, flush=True)

        write({"revision": LOCAL_MODELS[args.model].revision, "model": args.model, "split": args.split,
               "manifest_sha256": digest(manifest_path), "split_sha256": digest(path),
               "adapter_sha256": digest(adapter / "adapters.safetensors") if adapter else None,
               "max_tokens": 256, "temperature": 0, "mlx": version("mlx"), "mlx_lm": version("mlx-lm")})
        try:
            cases = read_rows(path)
            for case in cases:
                started = perf_counter()
                try:
                    result = model.generate(case["messages"], max_tokens=256, temperature=0,
                                            adapter_path=str(adapter) if adapter else None)
                except (RuntimeError, OSError, ValueError, ImportError) as exc:
                    failures += 1
                    write({"id": case["id"], "error_type": type(exc).__name__,
                           "latency_ms": (perf_counter() - started) * 1000})
                    if failures == 1 and case is cases[0]:
                        raise SystemExit("Initial generation failed. See the local results file.")
                    continue
                elapsed = (perf_counter() - started) * 1000
                text = redact(result.text)
                write({"id": case["id"], "prediction": text, "latency_ms": elapsed,
                       "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                       **(result.metadata or {}),
                       "diagnostics": semantic_diagnostics(text, case["facts"], case["evidence_paths"]),
                       "semantic_review": "pending_human_adjudication"})
        finally:
            model.unload()
        write({"cases": len(cases), "runtime_failures": failures})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
