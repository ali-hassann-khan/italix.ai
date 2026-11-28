# Phase 15 benchmark calibration

This phase uses the ten existing Phase 14 diagnostic IDs and the local 100-case calibration corpus only. It uses no final test inputs. Phase 14 artifacts remain intact. New outputs live under `$HOME/italix-phase15`.

## Diagnostic design

Compare full and compressed evidence with the previous structured JSON prompt and a concise prose prompt that asks for hunk citations. Models use their pinned snapshot, native chat template, no sampling and no adapter. Generate a greedy stream once per combination, then record the 192-, 384- and 512-token prefixes. A budget-limited prefix can end with buffered text or an incomplete JSON object. This design shares input and decode work across budgets. Prefix timings indicate observed stream latency rather than timings from separate launches with each budget.

Report native JSON and JSON recovered by removing an enclosing Markdown fence separately. A limit hit is a truncation signal. Lexical coverage of cited draft anchors, unmatched claims, unknown citations and repetition are diagnostics. Source entailment, factual correctness and importance of omissions require blind review.

On all ten cases, the 1.5B model produced 0/10 strict JSON and 8/10 fence-recoverable JSON with both full and compressed evidence at all three caps. At 384 tokens, compressed JSON reduced mean prompt length from 1,662 to 1,389 and mean measured latency from 2.16 to 1.92 seconds. Lexical cited-fact coverage went from 14.8% to 18.5%, while mean repeated four-word sequences rose from 0.3% to 6.2%. Full prose cited hunks on six cases and covered 7.4% of draft anchors. Compressed prose cited hunks on three cases and covered 11.1%. Raising this model's JSON output budget did not fix its formatting or omissions.

An AI-assisted source check found that the compressed JSON output removed the false claim that original kuberc code did not compare input to 'y', but still omitted the save-then-print behavior. The full prose output invented a change from input scanning to printing. The compressed prose output reversed the decline branch. In HTTPX, both JSON variants cited the mapping copies in Request and Response and omitted the Mapping type changes. These observations are diagnostic, pending human adjudication.


The two larger models were cross-checked on the same two diagnostic cases, Go DWARF and kubectl preferences. Qwen3-4B went from 0/2 parseable JSON at 192 to 1/2 at 384 with either context. The remaining case still reached 512. Qwen2.5-Coder-7B full-context JSON went from 0/2 at 192 to 2/2 after fence removal at 384. With compressed context it yielded two parseable fenced responses at 192, while cited draft-anchor coverage at 384 fell from 20% to 0%. The small cross-check supports using 384 for the next review pass. The coverage drop supports keeping full evidence visible to reviewers.

To rerun the complete diagnostic matrix into a new local directory:

```bash
cd /Users/alihassan/Projects/italix.ai
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
P14="$HOME/italix-phase14/pilot"
RERUN="$HOME/italix-phase15/rerun-01"
umask 077
mkdir "$RERUN"
for model in qwen-base qwen3-4b qwen-coder-7b; do
  python -m scripts.calibrate_context run "$P14/review-v2.jsonl" "$RERUN/$model.jsonl" \
    --prior "$P14/diagnostic-baseline/$model.jsonl" --model "$model" || break
done
python -m scripts.calibrate_context report "$RERUN"/qwen*.jsonl --corpus "$P14/review-v2.jsonl"
```

## Context audit

Compression retains the original order, exact added and removed lines, source path, hunk ID and header, no-newline markers, and one unchanged line next to each change. It marks omitted context. It removes no files or hunks and applies no length-based truncation.

On 100 cases, using the native Qwen2.5-1.5B tokenizer and provisional assistant targets, complete sequence totals fell from 192,809 to 161,714 tokens, down 16.1%. All 248 case-file pairs, 417 hunks and 248 required draft hunk references survived. A separate exact comparison verified all 8,294 changed lines and patch markers. Compression removed 3,273 unchanged context lines. For example, the compressed kuberc change retains the write and the new YAML regression, but omits nearby alias initialization and the unchanged input-scan error guard. The compressed HTTPX change retains the mapping copies and type changes, but omits adjacent constructor setup. These omissions illustrate why the training exporter requires the exact compressed packet to receive explicit human review.

| Training sequence cap | Full cases fitting | Compressed cases fitting |
| --- | ---: | ---: |
| 1,536 | 58/100 | 65/100 |
| 2,048 | 68/100 | 73/100 |
| 4,096 | 87/100 | 92/100 |

Mean compressed sequence length among fitting cases was 807, 916 and 1,327 tokens respectively. Scaling those three provisional mixtures to 3,000 rows gives approximately 2.42M, 2.75M and 3.98M tokens per pass. Real target rewrites can change these counts. Oversized cases should remain in review for restructuring or a larger reviewed sequence cap.

## Twenty-case review and release gates

The packet has the original ten diagnostic cases and ten more from the same calibration corpus, two extra cases per source where possible. Open the pinned before/after hunks, review the provisional reference, correct claims and evidence IDs or reject the record, and adjudicate each blind output. The reviewer can inspect the proposed compressed context and explicitly approve its hash. Export notes and keep the blind key private. Reference review, compressed context approval, and rights clearance remain separate. Browser submissions are evidentiary records for curator adjudication.

The build command only admits independently reviewed rows with rights metadata, pinned evidence and exact/near duplicate checks. The stage gate checks counts after all exclusions. Stages A, B and C require at least 100, 500 and 1,000 accepted records across splits. Stage D requires at least 3,000 training examples plus nonempty repository-family-disjoint development and test. Require human sampling per source and change family at each increment. Do not promote the supplied Phase 14 JSONL directly. Merge explicit source-rights approval, pinned license notices, and the curator-approved corrections into a new reviewed inventory. Preserve rejections rather than quietly replacing rows. Keep the completed review release and later additions on separate immutable manifests. The 100 acquired changes leave no margin for Stage A rejections. Refill through the same bounded Git screening after the calibration and source-rights approval.

```bash
cd /Users/alihassan/Projects/italix.ai
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
umask 077
P15="$HOME/italix-phase15"
open "$P15/review-20-blind.html"
# After human review and source-rights approval, set STAGE to the approved release.
STAGE=A
python -m scripts.prepare_specialization build "$P15/reviewed-stage-$STAGE.jsonl" \
  "$P15/splits-stage-$STAGE.json" "$P15/corpus-$STAGE" \
  --stage "$STAGE" --max-sequence-tokens 2048 --compressed-context
```

Use a full-context export for cases without compressed approvals. Omit `--compressed-context` for a full-context release. The compressed flag requires each reviewed row to contain `reviewed_context_sha256` matching the exact export.

## Practical training comparison

Use one pass with rank 4, last four layers, Q/V targets, gradient batch size one, seed 43, learning rate 2e-5, evaluation and checkpoints. MLX uses the pinned Apple-compatible quantized base and CUDA uses local original HF-format weights plus NF4/BF16. Apple uses `time -l` for CPU resource counters and the MLX training log for allocator peaks. Linux uses `time -v` and samples device-wide allocations with `nvidia-smi` if installed. CUDA needs a provisioned RTX A4000 host and locally verified weights. This phase did not run either training recipe.

Start candidate review around the 2,048-token limit, which fits eight more compressed draft cases than 1,536. MLX and CUDA can accept a 4,096-token reviewed manifest, but its 92% draft coverage comes with more long sequences and potentially higher activation memory. Neither training peak nor duration has been measured at 2,048 or 4,096. At batch one, a 3,000-case run takes 50 minutes per observed second/step plus checkpointing and validation. At 2,048 a pass has at most 6.144M unpadded tokens, versus 4.608M at 1,536 and 12.288M at 4,096. For the 151,936-token vocabulary, one unpadded batch-one logits tensor alone is 0.43–0.87 GiB at 1,536, 0.58–1.16 GiB at 2,048 and 1.16–2.32 GiB at 4,096 in 16- to 32-bit precision. Base weights, activations, gradients and allocator buffers are additional. Inspect the actual accepted sequence distribution and leave device headroom. MLX and CUDA optimizer/runtime differences make them separate runs. Run only the approved device path.

```bash
# Dry plan after release D exists.
CORPUS="$P15/corpus-D"
python -m scripts.train_specialization "$CORPUS" "$P15/mlx-run-01" --one-pass

# Manually execute after reviewing and approving the manifest and printed recipe.
APPROVED_CORPUS_SHA='paste approved manifest SHA-256'
scripts/monitor_specialization.sh "$P15/mlx-resource-01" \
  python -m scripts.train_specialization "$CORPUS" "$P15/mlx-run-01" \
  --one-pass --execute --confirm-manifest "$APPROVED_CORPUS_SHA"

# Alternative on the CUDA host. Run the dry plan first and review both digests.
HF_MODEL="$HOME/models/Qwen2.5-1.5B-Instruct"
python -m scripts.train_specialization_cuda "$CORPUS" "$HF_MODEL" "$P15/cuda-run-01"
APPROVED_MODEL_SHA='paste approved CUDA local-model fingerprint'
scripts/monitor_specialization.sh "$P15/cuda-resource-01" \
  python -m scripts.train_specialization_cuda "$CORPUS" "$HF_MODEL" "$P15/cuda-run-01" \
  --execute --confirm-manifest "$APPROVED_CORPUS_SHA" --confirm-model "$APPROVED_MODEL_SHA"
```
