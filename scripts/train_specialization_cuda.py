"""Manually run a one-pass CUDA QLoRA comparison from local approved artifacts."""
import argparse
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path

from scripts.prepare_specialization import MAX_TOKENS, SEQUENCE_LIMITS, digest, read_rows


def encode(row, tokenizer, max_tokens=MAX_TOKENS):
    messages = row["messages"]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("missing final assistant message")
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    complete = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if complete[:len(prompt)] != prompt or not len(prompt) < len(complete) <= max_tokens:
        raise ValueError("template prefix or sequence length failed validation")
    return {"input_ids": complete, "attention_mask": [1] * len(complete),
            "labels": [-100] * len(prompt) + complete[len(prompt):]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("model", type=Path, help="existing local HF-format Qwen2.5-1.5B-Instruct snapshot")
    parser.add_argument("output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-manifest")
    parser.add_argument("--confirm-model")
    args = parser.parse_args()
    manifest_path = args.corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_hash = digest(manifest_path)
    if (manifest["format"] != "italix-specialization-v1" or manifest["max_sequence_tokens"] not in SEQUENCE_LIMITS
            or not 2000 <= manifest["counts"]["train"] <= 5000 or args.output.exists()):
        parser.error("expected an approved 2,000–5,000-case corpus and a fresh output")
    data = {}
    for split in ("train", "valid"):
        relative = f"mlx/{split}.jsonl"
        path = args.corpus / relative
        if path.is_symlink() or digest(path) != manifest["sha256"][relative]:
            parser.error(f"{split} hash mismatch")
        data[split] = read_rows(path)
        if not data[split] or len(data[split]) != manifest["counts"][split]:
            parser.error(f"{split} count mismatch")
        if digest(path) != manifest["sha256"][relative]:
            parser.error(f"{split} file changed during read")
    model_path = args.model.resolve()
    if not (model_path / "config.json").is_file():
        parser.error("existing HF-format model required; no download is performed")
    config = json.loads((model_path / "config.json").read_text())
    if (config.get("model_type"), config.get("num_hidden_layers"), config.get("hidden_size")) != ("qwen2", 28, 1536) or config.get("quantization") or config.get("quantization_config"):
        parser.error("expected an unquantized HF-format Qwen2.5-1.5B model for runtime NF4")
    weights = sorted(model_path.glob("model*.safetensors"))
    if not weights:
        parser.error("no local safetensors weights found")
    model_files = weights + sorted(p for p in model_path.iterdir()
                                   if p.is_file() and p.suffix in {".json", ".jinja", ".txt"})
    model_hashes = {}
    for path in model_files:
        with path.open("rb") as stream:
            model_hashes[path.name] = file_digest(stream, "sha256").hexdigest()
    model_identity = sha256(json.dumps(model_hashes, sort_keys=True).encode()).hexdigest()
    recipe = {"model_path": str(model_path), "manifest_sha256": manifest_hash,
              "model_sha256": model_identity, "model_files_sha256": model_hashes,
              "examples": manifest["counts"]["train"], "epochs": 1, "seed": 43, "lr": 2e-5,
              "batch": 1, "rank": 4, "lora_alpha": 32, "target_layers": [24, 25, 26, 27], "target_modules": ["q_proj", "v_proj"],
              "max_sequence_length": manifest["max_sequence_tokens"], "quantization": "NF4", "checkpoints_every": 250}
    print(json.dumps(recipe, indent=2), flush=True)
    if not args.execute:
        return
    if args.confirm_manifest != manifest_hash or args.confirm_model != model_identity:
        parser.error("execution requires the approved manifest and model hashes")

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("CUDA and BF16 support required")
    set_seed(recipe["seed"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    datasets = {split: [encode(row, tokenizer, recipe["max_sequence_length"]) for row in rows] for split, rows in data.items()}
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_compute_dtype=torch.bfloat16))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=32, lora_dropout=0., task_type="CAUSAL_LM",
                                           layers_to_transform=recipe["target_layers"], layers_pattern="layers",
                                           target_modules=recipe["target_modules"]))
    args.output.mkdir(mode=0o700)
    (args.output / "plan.json").write_text(json.dumps(recipe, indent=2) + "\n")
    trainer = Trainer(model=model, data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100),
                      train_dataset=datasets["train"], eval_dataset=datasets["valid"],
                      args=TrainingArguments(output_dir=str(args.output), num_train_epochs=1, seed=43,
                                             learning_rate=2e-5, per_device_train_batch_size=1,
                                             per_device_eval_batch_size=1, bf16=True, report_to=[],
                                             optim="paged_adamw_8bit", lr_scheduler_type="constant",
                                             logging_steps=10, save_steps=250, eval_strategy="steps",
                                             eval_steps=250, prediction_loss_only=True))
    trainer.train()
    trainer.save_model(str(args.output / "adapter"))


if __name__ == "__main__":
    main()
