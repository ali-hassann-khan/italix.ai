# Phase 13 — Evidence-grounded PR summarization

## Public source audit

A release license and the rights to the source files, contribution text and personal data need separate review. Archive sizes are decimal and describe the published files. Phase 13 fetched metadata and documentation only.

| Candidate | Release and size | Content and recommendation |
| --- | --- | --- |
| CommitBench | 1,664,590 records, about 1.50 GB for the three CSVs. Dataset CC BY-NC 4.0. Repository code MIT. Java, Ruby, JavaScript, Go, PHP and Python. | Diffs, commit messages and repository/commit identifiers. No PR descriptions or review threads in the released schema. Commit text can omit details or overstate intent. Exclude release content from the product corpus under the current licensing assumption. Inspect upstream provenance independently for any newly acquired patches. [Official download and card](https://huggingface.co/datasets/Maxscha/commitbench) · [Size metadata](https://datasets-server.huggingface.co/size?dataset=Maxscha%2Fcommitbench) · [Paper](https://arxiv.org/html/2403.05188) |
| CodeReviewer | About 4.82 GB across quality, comment generation and refinement archives. CC BY 4.0 release. C, C++, C#, Go, Java, JavaScript, PHP, Python and Ruby. | Hunk-level diffs and code context. Comment and refinement tasks add review comments or reviewed code. Useful as an evidence-ranking lead, with weaker whole-PR coverage. Comment presence is a collection proxy in the quality task. Group review rounds together and verify source licensing, provenance and privacy before using any row. [Official downloads and license](https://zenodo.org/records/6900648) · [Primary paper](https://arxiv.org/html/2203.09095) |
| MCMD v2 | About 10.79 GB filtered and 39.45 GB raw. CC BY 4.0 release. Java, C#, C++, Python and JavaScript from 100 projects per language in the study. | Commit messages and changes with repository provenance. PR bodies and review conversations are outside the task. Candidate discovery only at first. Upstream source review, repeated changes and message noise make bulk ingestion a poor fit. [Official downloads and license](https://zenodo.org/records/5025758) · [Primary paper](https://arxiv.org/pdf/2107.05373) |
| PRSummarizer | 333K collected merged PRs, with 41,832 in the paper's filtered dataset, from approximately 1,000 Java repositories. Tooling MIT. A distinct permission grant for the released corpus was not verified. | PR descriptions, commit messages and IDs, and added source comments. Its documented input format lacks full patches and review threads. Descriptions can be missing and need factual verification even when present. Hold corpus material pending clearance. [Author repository and schema](https://github.com/Tbabm/PRSummarizer#dataset) · [Author download](https://drive.google.com/drive/folders/1VMByXOEmJDQL_JQY6l63NRiveUySY-Sq?usp=sharing) · [Primary paper](https://arxiv.org/abs/1909.06987) |
| CanItEdit | 105 authored Python examples, roughly 250 KB parquet. Dataset card lists MIT. | Before/after code, natural-language edit instructions and published test fields. Useful for checking numerical and behavioral rubric coverage. Small, authored and single-file. Keep separate from the product corpus. [Official card and download](https://huggingface.co/datasets/nuprl/CanItEdit) · [Author repository](https://github.com/nuprl/CanItEdit) |
| EditPackFT | About 22.6K Python edits. Dataset card lists MIT. | Single-file before/after code, diff and commit-message/instruction data with source metadata. Check per-row upstream license and groups for forks or duplicate file edits. Auxiliary single-file evidence data only after review. [Official card and download](https://huggingface.co/datasets/nuprl/EditPackFT) |

## Recommended first corpus

Aim for 3,000 training, 350 development and 450 new test changes, with 2,000–5,000 allowed in training. Start with a 100-case human review calibration before expanding. Select permissively licensed repositories with pinned revisions and retain license notices verified at those revisions, including file-level exceptions, generated and vendored material. Use authored references and review text only when its reuse rights and privacy have been checked. Store PR identifiers, base and head, source URL, family, permissions and reviewer pseudonym. Commit messages and PR descriptions may help discover candidates. Reviewers derive the reference from the code.

Assign complete upstream families, forks, cherry-picks and connected change series to one split before labelling. Balance across several languages, repositories, change types and single- or multi-file scope. Ensure multi-file validation includes a measured cross-file slice. Take the newest eligible revisions for the new test where possible. A curator holds that test and reviews repeat exposure and any overlap with previous benchmark inputs. Neither timing nor repository-disjoint splits can rule out base-model pretraining overlap. The existing reserved benchmark stays outside this workflow.

Require a reviewer distinct from the fact author to verify each assertion against the pinned patch. Record material facts as atomic claims with stable hunk IDs, key terms, and any reference quantities and units. Record security/performance claims only when the packet supplies sufficient reproducible evidence, and record the evidence basis in the review notes. Run a second review on disagreements, multi-file behavioral claims and a stratified sample. Report inter-reviewer agreement and unresolved counts. Unresolved examples stay excluded. Treat the independent-review field as a human attestation.

Remove secrets and identifying review metadata before export. Reject noisy or unrelated changes, binary and generated patches, inconsistent labels, unsupported behavior claims, oversized inputs and redundant families. The builder screens secrets, checks evidence, caps patches at 24 KB, eight files and 20 hunks, checks the actual base template against 1,536 sequence tokens, and drops exact or near-duplicate changed token sequences. Near matches use 5-token shingle Jaccard at 0.85. Review cloned, reordered and translated changes separately because heuristic deduplication misses some of them. Preserve local access controls and per-source deletion lineage.

## Offline operations

Use paths outside the checkout for data and outputs. Start with a private working directory and an approved JSONL source inventory. Each inventory record supplies `id`, lowercase `repo` in `owner/repo` form, `local_path`, full lowercase 40-digit `base` and `head` commit IDs, `source_url`, `license`, `license_url` and `rights_reviewed: true`. Base must be an ancestor of head. For a PR, choose the correct pinned diff base first. Prepare `splits.json` as a mapping from repo to objects with `family` and `split` keys. The split names are `train`, `valid` and `test`.

```bash
cd /Users/alihassan/Projects/italix.ai
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
umask 077
PHASE13_DIR="$HOME/italix-phase13"
mkdir -p "$PHASE13_DIR"
python -m scripts.prepare_specialization collect "$PHASE13_DIR/sources.jsonl" "$PHASE13_DIR/pending.jsonl"
```

Have reviewers create `reviewed.jsonl` from that inventory. Set `review_status` to `independently_reviewed` after human review and add the reviewer pseudonym and `facts`. For example, if both supplied hunks change a helper's timeout:

```json
{"id":"timeout","text":"The worker timeout is now 30 seconds.","anchors":["worker timeout"],"numbers":["30 seconds"],"evidence_ids":["H001","H002"]}
```

Use `impact_evidence: true` only with a reviewed justification. Keep adjudication notes and license notices with the source inventory.

```bash
python -m scripts.prepare_specialization build "$PHASE13_DIR/reviewed.jsonl" "$PHASE13_DIR/splits.json" "$PHASE13_DIR/corpus"
python -m scripts.train_specialization "$PHASE13_DIR/corpus" "$PHASE13_DIR/run-01"
```

The second command prints a plan. The proposed initial controlled budget is the pinned Qwen2.5 1.5B 4-bit model, standard assistant-masked LoRA, rank 4 over the final four layers, scale 8, dropout zero, learning rate 2e-5, batch size 1, seed 43, 1,536 tokens, and 400 steps. Checkpoint and development loss intervals are 100 steps. This budget samples a fraction of the proposed corpus. Record training-example exposure and causal-mask validation. Any full-corpus pass or extra run requires its own decision. Review the plan and hashes before manual execution.

```bash
# Manual execution only, after approving this reviewed corpus and recipe.
PHASE13_APPROVED_SHA256='paste the approved manifest SHA-256 here'
python -m scripts.train_specialization "$PHASE13_DIR/corpus" "$PHASE13_DIR/run-01" \
  --execute --confirm-manifest "$PHASE13_APPROVED_SHA256"
```

Execution verifies input hashes and counts and takes a read-only training/development copy inside the run directory. Save this directory intact with its plan, configuration, training log, and checkpoints. The launcher has no production database, registration, qualification or activation operation.

## Model comparison and human review

Run the pinned base models individually with the same inputs, claim schema, greedy decoding, and a 256-token generation budget. Each retains its native chat template. After the manual training run, include the candidate on the same development split. Logs include dataset/model digests, dependencies, token counts, generation throughput, MLX memory and wall-clock latency. Case one includes a cold load. Report warm mean, median and p95 separately and disclose failures and truncation.

```bash
for model in qwen-base qwen3-4b qwen-coder-7b; do
  python -m scripts.evaluate_specialization "$PHASE13_DIR/corpus" "$PHASE13_DIR/$model.valid.jsonl" --model "$model"
done
# After manual training completes
python -m scripts.evaluate_specialization "$PHASE13_DIR/corpus" "$PHASE13_DIR/candidate.valid.jsonl" \
  --model qwen-base --adapter "$PHASE13_DIR/run-01/adapter"
```

Freeze prompts, rubric, decoding, selected checkpoint and model hashes after development feedback. Give the separately curated Phase 13 test to a curator. Its CLI path requires the digest via `--split test --confirm-test-manifest`. Keep raw model IDs and runtimes out of review packets. Randomize output ordering with a recorded seed and keep the mapping private until adjudication.

The new diagnostics detect invalid JSON, incorrect values near known fact anchors, conflicting values or explicit risk labels, missing known facts and file coverage, unknown citation IDs, unmatched claims, unsubstantiated security/performance language and repetition. These checks flag suspects for humans. For example, a citation ID can be valid while failing to entail a claim. Review each claim against the patch as supported, contradicted, or unsupported, with a separate required-fact coverage and cross-file citation precision rubric. Record concrete calculation or behavior evidence and score false claims separately from omissions. Report supported claim precision, fact and file recall, cross-file evidence precision/recall, contradiction and repetition rates, alongside JSON and efficiency metrics. Do not use the diagnostic scores to alter existing qualification gates. The existing scaffolded baseline and deterministic output can be shown as product references in a separately labeled review.

## Storage estimates

The three cached model snapshots total about 6.94 GiB. A normalized corpus and its MLX/review exports should take a few hundred MB for 3,800 accepted examples. Budget 0.3–1 GB including review and rejected-input buffers, 1–4 GB for selectively acquired source objects with actual size measured before acquisition, and roughly 20–50 MB for rank-4 checkpoint files and logs. An early local inspection measured about 9.3 GiB free. Avoid full source clones and bulk archives. Longer sequences raise memory requirements and this recipe has no measured training duration or peak on the approved real corpus.
