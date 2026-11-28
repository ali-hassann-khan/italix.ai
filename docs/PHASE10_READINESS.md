# Phase 10 review and release checklist

## Pilot v2

The [100-case revision](../data/phase10/pilot-v2.jsonl) retains Phase 9's policy
outcomes and original benchmark as separate artifacts. Case 044 now contains a
valid Ruby before/after edit. Cases 084 and 095 use fresh hunks in place of the
broken repeated patch. All 16 multi-file cases have unique pairs of changed
lines. Four intentionally truncated patches remain for abstention. Labels
focus on what the actual edited lines establish. All cases are pending human
review. The new split stays outside the production qualification gates.

### Reviewer worksheet

Review one family at a time. Generate a blind packet with the input hashes
and diffs. It omits draft labels.

```bash
python - <<'PY' > /tmp/italix-phase10-review.md
import hashlib, json
with open("data/phase10/pilot-v2.jsonl") as source:
    cases = sorted(map(json.loads, source), key=lambda r: (r["family"], r["id"]))
for case in cases:
    print(f"## {case['family']} / {case['id']}")
    print(hashlib.sha256(case["diff"].encode()).hexdigest())
    print("\n```diff")
    print(case["diff"].rstrip())
    print("```\n")
PY
```

Record one row per case. An adjudicator compares the review with the draft
only after the reviewer submits it.

| Case ID and input SHA-256 | Reviewer and date | Policy or abstention | Concrete before/after facts | Evidence line IDs | Uncertain claims | Verdict and rationale |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  | Accept / revise / exclude |

Record revisions and disagreements. Freeze inputs, rubric, family mapping,
review log and SHA-256 before marking cases independently reviewed. Report
macro-averages by family and keep all variants of one upstream change in the
same split. Keep the reserved final test sealed.

For real-world intake, CPython offers runtime and library behavior changes,
Go offers standard-library/API changes, and Kubernetes offers controller and
configuration changes. Check licensing and notices at each pinned commit,
including vendored subtrees. Store a source link and patch digest, use reviewed
code and tests for semantics, exclude credentials and personal text, and keep
repository-disjoint splits. Avoid duplicating upstream cherry-picks. No
repository patches are included in this revision.
[CPython license](https://github.com/python/cpython/blob/main/LICENSE) ·
[Go license](https://github.com/golang/go/blob/master/LICENSE) ·
[Kubernetes license](https://github.com/kubernetes/kubernetes/blob/master/LICENSE)

## Workload evaluation

Run each pinned model in a separate process with native templates, greedy
decoding, and the same per-workload request budgets. The runner records
latency, token counts, throughput, process RSS and peak MLX allocation. It
requires independent review for full collection and permits one draft smoke
case per invocation. Use new output paths.

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.evaluate_product_models \
  qwen3-4b memory /tmp/italix-memory-smoke.jsonl --smoke
```

| Workload | Human review or deterministic checks |
| --- | --- |
| PR summary | Credit facts grounded in the diff. Count missed files, misstated identifiers/values, and unsupported claims. |
| Semantic understanding | Check per-file before/after behavior and whether boundaries or unknown implementation details are stated accurately. |
| Evidence extraction | Score valid JSON and IDs, evidence precision/recall, and required-file coverage. |
| General chat | Check instruction following, arithmetic, and whether the answer admits facts the prompt omits. |
| Developer Memory | Score retrieval recall separately from citation accuracy, factual entailment, and abstention. Include contradictory or adjacent sources. |

Blind human reviewers to model identities. Score claims against approved facts
and evidence, then report latency and resource trade-offs alongside quality.
A valid Memory citation proves that a source was retrieved, while claim
entailment still needs review. The existing deterministic path policy remains
authoritative for triage.

## Local operator workflows

Learning Studio can create and review plans, explicitly execute a digest-bound
plan, poll step progress, preview a locally registered adapter, evaluate it,
and confirm qualified activation or rollback. The reviewed path pins the base
revision. In-process background execution remains tied to the local API
process. Recovery marks interrupted jobs failed and returns their examples to
the queue after the worker lock is released. It preserves partial artifacts and
requires a new reviewed plan. It does not resume an optimizer checkpoint.
The existing synchronous MCP and HTTP execution paths remain compatible.

Developer Memory allows explicitly approved local imports, source search,
local cited answers, deletion, credential rescreening, and an on-demand purge
based on import age. The browser sends search text in a POST body. Use the POST
search route for privacy. SQLite and FTS entries are updated together.
Deletion and redaction are logical, and backups, filesystem snapshots, or
older database pages may retain material. Periodic retention is an operator
action. Secret screening uses patterns and requires source sanitization.

## Release gates for specialization

1. Independently review the fixture packet and intake representative,
   licensed, context-rich patches.
2. Run the blinded workload comparison on the three pinned models and verify
   citation entailment, multi-file coverage, abstention, and realistic latency.
3. Review source-isolation, crash recovery and model artifact preview on a
   fresh checkout. Exercise real training only with separate approval.
4. Target measured model failures. Repeated summaries missing a second file or
   inventing runtime effects justify work on multi-file semantic supervision.
   Persistent evidence omissions justify measuring a discriminative evidence
   ranker against the existing LLM selector. Repeatable retrieval/citation
   failures justify a retrieval or grounded-answering experiment. Keep
   policy-defined labels separate from semantic model learning and evaluate on
   repository-disjoint, independently reviewed cases.
