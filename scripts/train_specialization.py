"""Print or manually launch one isolated, reviewed MLX specialization run."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

from italix.engines import LOCAL_MODELS
from italix.privacy import redact
from scripts.prepare_specialization import MAX_TOKENS, SEQUENCE_LIMITS, digest

RECIPE = {
    "train": True, "test": False, "fine_tune_type": "lora", "num_layers": 4,
    "iters": 400, "learning_rate": 2e-5, "batch_size": 1,
    "max_seq_length": MAX_TOKENS, "mask_prompt": True, "seed": 43,
    "report_to": None, "steps_per_report": 10, "steps_per_eval": 100, "save_every": 100,
    "lora_parameters": {"rank": 4, "scale": 8.0, "dropout": 0.0},
}


def plan(corpus, output, *, one_pass=False):
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    spec = LOCAL_MODELS["qwen-base"]
    if (manifest["format"] != "italix-specialization-v1" or manifest["base_revision"] != spec.revision
            or manifest["max_sequence_tokens"] not in SEQUENCE_LIMITS or not 2000 <= manifest["counts"]["train"] <= 5000):
        raise ValueError("expected a reviewed Phase 13 corpus with 2,000–5,000 training examples")
    data = corpus / "mlx"
    names = {"train.jsonl", "valid.jsonl"}
    if output.exists() or {path.name for path in data.iterdir()} != names:
        raise ValueError("output must be new and the MLX directory must contain only train and valid")
    for name in names:
        path = data / name
        if path.is_symlink() or digest(path) != manifest["sha256"][f"mlx/{name}"]:
            raise ValueError(f"training input failed integrity check: {name}")
        with path.open() as stream:
            if sum(bool(line.strip()) for line in stream) != manifest["counts"][path.stem]:
                raise ValueError(f"training input count mismatch: {name}")
    model = spec.cached_path()
    result = {
        "manifest_sha256": digest(manifest_path), "base_revision": spec.revision,
        "counts": {name: manifest["counts"][name] for name in ("train", "valid")},
        "recipe": RECIPE | {"max_seq_length": manifest["max_sequence_tokens"]} | ({
            "iters": manifest["counts"]["train"],
            "lora_parameters": RECIPE["lora_parameters"] | {"keys": ["self_attn.q_proj", "self_attn.v_proj"]},
        } if one_pass else {}),
        "dependencies": {package: version(package) for package in ("mlx", "mlx-lm", "transformers")},
        "input_sha256": {f"mlx/{name}": manifest["sha256"][f"mlx/{name}"] for name in sorted(names)},
        "command": [sys.executable, "-m", "mlx_lm", "lora", "--model", str(model), "--data", str(output / "data"),
                    "--adapter-path", str(output / "adapter"), "-c", str(output / "recipe.yaml")],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--one-pass", action="store_true", help="budget one optimizer step per reviewed row")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-manifest")
    args = parser.parse_args()
    output = args.output.resolve()
    reviewed = plan(args.corpus.resolve(), output, one_pass=args.one_pass)
    print(json.dumps(reviewed, indent=2), flush=True)
    if not args.execute:
        return
    if args.confirm_manifest != reviewed["manifest_sha256"]:
        parser.error("execution requires the exact approved manifest SHA-256")
    output.mkdir(mode=0o700)
    snapshot = output / "data"
    snapshot.mkdir(mode=0o700)
    for name in ("train.jsonl", "valid.jsonl"):
        relative = f"mlx/{name}"
        frozen = snapshot / name
        frozen.write_bytes((args.corpus / relative).read_bytes())
        if digest(frozen) != reviewed["input_sha256"][relative]:
            raise ValueError(f"snapshot changed during preparation: {name}")
        frozen.chmod(0o400)
    (output / "plan.json").write_text(json.dumps(reviewed, indent=2) + "\n")
    # JSON is valid YAML and preserves the reviewed recipe exactly.
    (output / "recipe.yaml").write_text(json.dumps(reviewed["recipe"], indent=2) + "\n")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("ITALIX_") and key not in {"GEMINI_API_KEY", "GROQ_API_KEY"}}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    with (output / "training.log").open("x") as stream:
        with subprocess.Popen(reviewed["command"], env=env, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as process:
            for line in process.stdout:
                stream.write(redact(line))
                stream.flush()
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
