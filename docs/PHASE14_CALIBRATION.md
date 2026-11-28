# Phase 14 real-world calibration

The acquisition contains 100 pinned public commit changes with five lines of unchanged context on either side of each hunk where available. It is a calibration corpus. It contains no independently human-reviewed labels and has not been admitted to the training exports or to any reserved ITALIX test. These are commit-level patches, some originating from squashed PRs. Multi-file counts do not imply full PR coverage.

| Repository | License screened at base and head | Cases |
| --- | --- | ---: |
| golang/go | BSD-3-Clause | 15 |
| kubernetes/kubectl | Apache-2.0 | 25 |
| cli/cli | MIT | 25 |
| pallets/flask | BSD-3-Clause | 25 |
| encode/httpx | BSD-3-Clause | 10 |

The five mirrors were shallow, bare and single branch with depth 45, 65, 95, 95 and 105 respectively. No full clone or dataset archive was acquired. Actual on-disk usage was about 74 MiB, including baseline results. The selected acquisition SHA-256 is `82d98121f75e46e150bccadaed9081ec76336443a7e24d831d5c7fad9407e4cb`. Each record contains exact base/head pins, repository URL, file paths and license notice content hashes from both revisions. The scanner checks regular files, changed source headers, file-level SPDX, nested license or notice files, generated markers and sensitive material. It is a conservative automated screen. Review attribution, separate licensing and redaction on each packet before admitting it to a redistributed corpus. The kubectl mirror can contain changes also present upstream in Kubernetes. Treat both as one repository family.

Of 431 examined commits, 324 were excluded and seven eligible cases were held out by the 100-case selection cap. Exclusion counts are first-hit categories. Common reasons included 134 excluded paths, 81 sensitive-source or sensitive-patch matches, 26 oversized patches, 26 small patches, 25 without a matching source/configuration extension, six generated files and four nested-license/notice cases. The credential scanner errs conservatively. A match does not establish a disclosure. No duplicate was flagged by the exact/near rule, and a separate check found no whitespace-only patches in the selected packet. Human reviewers should still group backports, related implementation and follow-up changes. The corpus has 83 multi-file cases and 248 case-file pairs.

## Human review

The 100 cases each have a provisional reference with exact changed-line citations. It uses one extract per changed file and misses behaviors spanning lines or subsystems. Four AI-assisted semantic spot-checks identify examples of those gaps. The file `verified-stats.json` checks all 248 draft excerpts against their hunks. It does not validate semantic completeness or model correctness.

Open the private HTML offline. Expand a case, inspect its pinned hunks, replace provisional quotes with atomic factual claims, and include all required hunk IDs. For each blind candidate claim, mark supported, contradicted or unsupported and record missing required facts, missing files and repetition. Export the notes before closing the page. Keep the `*.key.json` map private until adjudication. Confirm source and file-level licensing separately from technical accuracy. The same reviewer should never treat an automated draft or another model prediction as evidence.

```bash
cd /Users/alihassan/Projects/italix.ai
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
umask 077
P14="$HOME/italix-phase14/pilot"
open "$P14/diagnostic-baseline/review.html"
```

The matched offline inference protocol uses the same patches and claim instructions, native chat templates, greedy decoding and 192 generated tokens. The baseline reports latency and per-process high-water MLX allocations alongside anchor-based coverage, unmatched claims, impact language and repetition. Anchor coverage is a diagnostic proxy. In particular, a correct paraphrase can fail a lexically anchored check, and an unsupported statement can contain the correct anchor.

The first 10 cases completed on each model with zero runtime errors. This is a matched engineering sample. Strict JSON validity and complete fenced JSON are reported separately. The first latency includes model loading and the first request.

| Model | Strict JSON / after fence removal | Reached cap | Fact/file anchor coverage after fence removal | Mean repetition ratio | First request | Warm median / p95 | Generation tok/s | Peak MLX GB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B | 0/10 · 8/10 | 0/10 | 14.8% | 0.33% | 8.21 s | 1.50 / 2.43 s | 127.6 | 1.91 |
| Qwen3-4B | 1/10 · 1/10 | 8/10 | 0.0% | 1.47% | 27.80 s | 7.20 / 11.39 s | 54.2 | 3.86 |
| Qwen2.5-Coder-7B | 0/10 · 6/10 | 4/10 | 25.9% | 2.83% | 32.14 s | 9.38 / 12.17 s | 36.3 | 5.42 |

The unmatched-claim heuristic flagged 4, 1 and 4 cases respectively after fence removal. The impact keyword heuristic flagged zero parseable cases. Invalid or truncated output reduces the reach of both heuristics. Factual accuracy and unsupported-claim rates remain pending human review. An AI-assisted spot-check found that the 1.5B model incorrectly says the original kuberc logic did not compare against 'y'. The patch shows that it already did. The larger outputs cite the new regression but omit the main behavior of continuing to print the generated preferences, and both hit the output cap in that case. Preserve these as diagnostic observations. The complete 100-case comparison remains available after the review calibration.

To run the complete automated comparison on all 100 cases:

```bash
mkdir -p "$P14/full"
for model in qwen-base qwen3-4b qwen-coder-7b; do
  python -m scripts.evaluate_calibration run "$P14/review-v2.jsonl" \
    "$P14/full/$model.jsonl" --model "$model" || break
done
python -m scripts.evaluate_calibration report "$P14"/full/qwen*.jsonl
python -m scripts.evaluate_calibration packet "$P14/review-v2.jsonl" \
  "$P14/full/review.html" "$P14"/full/qwen*.jsonl
open "$P14/full/review.html"
```

Use a fresh result directory for retries to preserve completed results.

## Training sizing and manually approved execution

The measured native 1.5B tokenization of the 100 cases and provisional targets produced 180,725 prompt tokens plus 12,084 assistant tokens. Median full sequence was 1,335 tokens rounded, and p95 was 4,936. Only 58 fit the 1,536-token cap. Their mean was 976 total tokens. Preserve complete cross-file evidence when shortening packets or set oversized examples aside. Re-tokenize reviewed labels and review lost categories before expanding the corpus.

At 3,000 accepted examples and 976 tokens per sequence, one pass is approximately 2.93M presented tokens. Keep the final assistant-token estimate separate and recalculate it after human editing. A 400-step batch-one run covers a budget of 0.133 pass, or roughly 390K presented tokens. The proposed Phase 14 budget is 3,000 steps for batch one and one pass in a 3,000-row corpus. Effective sampling order can differ from an exact one-time traversal in MLX. Keep rank 4, final four layers, Q/V projection targets, learning rate 2e-5, constant schedule, max sequence 1,536 and seed 43 fixed. The four-layer Q/V recipe has about 77,824 trainable parameters under the pinned architecture.

No training throughput or memory was measured for this recipe. At an assumed 1–5 seconds per step, the MLX run would take about 50 minutes to 4 hours 10 minutes plus validation overhead. At an assumed 0.5–3 seconds per step, the CUDA run would take about 25 minutes to 2 hours 30 minutes plus validation overhead. Start resource budgeting at 4–8 GiB for the MLX run and 4–9 GiB VRAM for CUDA, including activations and runtime buffers. These are planning estimates, sensitive to sequence lengths, implementation and concurrent local workloads.

The launchers require an approved Phase 13-format human-reviewed corpus with 2,000–5,000 training rows. The calibration draft is ineligible. These commands print a dry plan:

```bash
CORPUS="$HOME/italix-phase13/corpus"
python -m scripts.train_specialization "$CORPUS" "$HOME/italix-phase14/mlx-run-01" --one-pass
# Run on the CUDA host with Python 3.11 and compatible torch, transformers,
# accelerate, bitsandbytes and peft already provisioned.
HF_MODEL="$HOME/models/Qwen2.5-1.5B-Instruct"
python -m scripts.train_specialization_cuda "$CORPUS" "$HF_MODEL" "$HOME/italix-phase14/cuda-run-01"
```

CUDA requires a pre-existing original Hugging Face-format safetensors snapshot. An MLX-quantized cache is a different format. The CUDA launcher fingerprints local model files, uses NF4/BF16 and masks prompt and padding labels. MLX and CUDA use distinct quantization and optimizer implementations. Treat results as separate runs. CUDA runtime was unavailable on the Mac. Only run one device path for a single approval.

```bash
# Manual execution only after reviewing the corpus and approved digests.
APPROVED_CORPUS_SHA='paste reviewed manifest digest'
python -m scripts.train_specialization "$CORPUS" "$HOME/italix-phase14/mlx-run-01" \
  --one-pass --execute --confirm-manifest "$APPROVED_CORPUS_SHA"

# Alternative manual command on the CUDA host.
APPROVED_MODEL_SHA='paste reviewed CUDA model fingerprint'
python -m scripts.train_specialization_cuda "$CORPUS" "$HF_MODEL" "$HOME/italix-phase14/cuda-run-01" \
  --execute --confirm-manifest "$APPROVED_CORPUS_SHA" --confirm-model "$APPROVED_MODEL_SHA"
```
