# Phase 16 real-world annotation

## Source audit and label states

The 100 previously acquired changes were checked against their acquisition checksum, retained pinned base/head metadata, regenerated hunk IDs, and per-revision notice hashes. There are 83 multi-file cases, 248 changed file/case pairs and 417 hunks. All 248 prior extractive draft citations point to the exact claimed changed line. There are zero exact and near duplicate cases under the existing 5-token shingle cutoff of 0.85. Related changes, forks, upstream mirrors and renamed or reorganized implementations can still require manual grouping. File-level licensing retains its Phase 14 automated-screen status. Separate human rights clearance is still required.

Use three separate label fields. `annotation_origin` identifies model generation. `automatic_checks` describes deterministic validation, which tests evidence structure and lexical alignment. `human_review` stays pending in every generated row. An independently reviewed reference requires a human to check the pinned patch and supply adjudicated facts. Only the separately approved, rights-cleared corpus builder can use those labels for training. Calibration source inputs and model outputs are distinct from the existing reserved final test.

## Local annotation pipeline

Each model receives the same evidence-preserving compressed change, a fixed citation prompt, a 384-token output budget and greedy decoding via its native chat template. Generate with the models sequentially so they are unloaded between runs. Retain raw drafts, errors, pins, candidate hashes and timings. Matching requires the same immutable case order and recipe for all three models. The audit skips duplicate cases for future annotation while retaining their IDs in its report.

The decoder records strict JSON, fenced JSON and fenced bare arrays separately and preserves the original response. The review queue flags malformed or truncated outputs, missing or unknown hunk IDs, missing changed files, quoted identifiers and numbers absent from cited material, and simple reversed before/after values. The numerical check excludes diff header offsets. Cross-model checks compare cited hunks, source-backed identifiers and explicit numerical transitions. When sources agree lexically they may still share the same factual error. Lexical checks do not establish units, causality, execution behavior or entailment. An uncited filename expressed without an explicit filename or identifier can also escape lexical checks.

Choose a provisional claim set from one model based on fewer risky flags, better file coverage, explicit identifiers and hunk agreement. Keeping one source candidate intact preserves its claim/citation mapping. Review triage is urgent for material evidence flags, conflicting numerical claims or fewer than two parseable models. It is targeted for softer omissions or low evidence overlap, with the remaining cases eligible for sampling. Urgent, targeted and sampled cases are still pending human review.

## Pilot observations

Ten short calibration cases, two per source, completed on all three cached models without runtime failures. Qwen2.5-1.5B returned eight usable fenced arrays and two invalid outputs. Qwen3-4B returned eight strict JSON outputs and two invalid outputs. Qwen2.5-Coder-7B returned ten usable fenced JSON objects. The selected set contains four automatic passes and six flagged cases. The queue has four urgent, five targeted and one random-sample candidate. No explicit cross-model numeric contradiction appeared in this sample. That signal has a dedicated regression fixture.

An AI-assisted spot-check of the random-sample kubectl case found that every model cited the same hunks, while the selected draft omitted the consequential change from substring to prefix matching and the regression for embedded flag strings. It passed the lexical checks and is incomplete. Keep it in human calibration and review all 20 before accepting a low-risk sampling rate.

## Exact commands

Run from the existing checkout and virtual environment. Use fresh paths to preserve Phase 14–16 artifacts.

```bash
cd /Users/alihassan/Projects/italix.ai
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
umask 077

P14="$HOME/italix-phase14/pilot"
P15="$HOME/italix-phase15"
P16="$HOME/italix-phase16"
RUN="$P16/calibration-20"
mkdir -m 700 "$RUN"

python -m scripts.annotate_real_changes audit "$P14/review-v2.jsonl" \
  "$P14/acquisition.json" "$P14/notices" "$P15/review-20-blind.key.json" "$RUN/audit.json"

for model in qwen-base qwen3-4b qwen-coder-7b; do
  python -m scripts.annotate_real_changes generate "$P14/review-v2.jsonl" "$RUN/audit.json" \
    "$RUN/$model.jsonl" --model "$model" --scope calibration || break
done

python -m scripts.annotate_real_changes reconcile "$P14/review-v2.jsonl" \
  "$RUN/results.json" "$RUN"/qwen*.jsonl
python -m scripts.annotate_real_changes packet "$P14/review-v2.jsonl" \
  "$RUN/results.json" "$RUN/review.html"
open "$RUN/review.html"
```

To annotate all 100 acquired cases, select a new `RUN` directory and replace `--scope calibration` with `--scope all`. For larger approved acquisitions, keep a fresh directory for each source batch with its original `acquired.jsonl`, checksum manifest and notice files. Audit that batch with `--expansion` and generate with `--scope all`. `--offset` and `--limit` allow immutable batches such as 50 cases. Reconcile only files from the same shard; keep previous outputs for recoveries. Before pooling training examples, run the repository-family split and cross-batch near-duplicate checks in `scripts.prepare_specialization` across the whole collection. Assign mirrors such as kubectl/Kubernetes, forks, backports and copied patch families to one split before exporting.

## Human quality measurement and sampling

No completed independent review export was available. Review the initial 20 without seeing model identities. Establish atomic source facts first, then reveal the selected candidate and adjudicate each claim as supported, contradicted or unsupported. Record critical omitted facts, numeric/direction error, affected files and cited hunk IDs. A second reviewer checks all contradictions and source-rights questions and at least 20% of the remaining calibration cases. Keep qualification and review criteria frozen for the measured sample.

The scoring command accepts selected-candidate verdicts tied to the exact results file SHA-256. Use the following structure in a manually authored file and substitute the actual IDs, counts and verdicts. Do not treat an AI-assisted annotation or procedural test as a human submission.

```json
{
  "reviewer": "reviewer-pseudonym",
  "human_attestation": true,
  "results_sha256": "<SHA-256 of results.json>",
  "cases": [
    {"id": "<reviewed case>", "claim_verdicts": ["supported", "unsupported"], "required_facts": 4, "covered_facts": 2}
  ]
}
```

```bash
shasum -a 256 "$RUN/results.json"
python -m scripts.annotate_real_changes score-human "$RUN/results.json" \
  "$RUN/audit.json" "$RUN/human-verdicts.json" "$RUN/human-metrics.json"
```

Human metrics report judged claim precision, required-fact recall, contradiction counts, denominators, and breakdown by automatic check outcome. Missing reviews remain pending. The Phase 15 free-text notes may guide adjudication, but they are not a substitute for verdicts tied to these new candidates.

Review every urgently flagged case and all new source families at the next expansion. Within the targeted pool, sample across repository, language, multi-file ratio, length and disagreement. Once the 20-case calibration establishes acceptable reviewer agreement and limits for missed errors, start with a random 20% sample of automatically passing cases per stratum, with at least five where available. Include longer cases and cases where independent sources agree but cite few distinct hunks. Escalate a stratum to full review when a sampled case has a material wrong value, contradicted behavior or repeated unsupported claims. Retain uncertain and rejected examples with their reason codes. Sampling estimates failure rates and prioritizes work. Only individually adjudicated records can carry independently human-reviewed status.

## Cost guidance

Time includes local inference and first-load overhead. Sequential execution means the largest measured model allocation governs MLX memory; add process and OS headroom. Use a representative stratified shard before forecasting expansion. The immediate pilot prefers shorter cases from the 20-case review packet so a forecast extrapolated directly from it is optimistic. The ten-case pilot used 168.7 seconds across 30 generations. First-load-inclusive per-model totals were 16.9 seconds for 1.5B, 68.7 seconds for 4B and 83.1 seconds for 7B. MLX peaks were 1.78, 3.40 and 5.26 GB respectively. The measured short-shard rate extrapolates to 2.34, 4.69 and 14.06 device-hours for 500, 1,000 and 3,000 cases. A 1.5–2× planning multiplier gives about 3.5–4.7, 7–9.4 and 21–28.1 hours pending a representative shard. Human time must be budgeted separately. At four to eight minutes per case, individually reviewing 500, 1,000 and 3,000 candidates takes roughly 33–67, 67–133 and 200–400 reviewer-hours respectively, before second reviews.
