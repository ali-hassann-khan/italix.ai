# Phase 11 product validation

## Installation and startup

The clean-room check cloned the published branch locally at `fc2f4e9`, then
copied the current source-only changes into that clone. It shared no virtual
environment, node_modules, application database, .env file, or workspace
artifacts. Installation used Python 3.11.16, Node 26.8.1, npm 12.0.2, and
SQLite 3.53.4 with FTS5. The pinned requirements resolved MLX-LM 0.30.6
with MLX 0.32.5 in the clean environment. The separate Memory and public
probes used the existing MLX 0.32.4 environment. All model identities were
pinned, and no new weights were downloaded.

- `pip install -r requirements.txt`, `pip check`, `npm ci --prefer-offline`,
  and Next.js production build passed.
- With empty HOME and Hugging Face cache, the full backend suite passed 94 tests.
  A first Streamlit AppTest timeout passed on retry with usage reporting
  disabled. Test configuration now explicitly disables Streamlit usage
  reporting and clears both cloud provider keys.
- Model-free learning and memory browser checks passed against isolated SQLite.
  The browser verified consent, redaction, search, deletion and same-origin
  enforcement. Frontend lint, typecheck, and 21 operator security checks passed.
- The live MCP stdio client discovered and called policy-only triage on fresh
  model-free FastAPI. Anonymous model requests received 401, viewers could
  list models, and viewer access to training jobs received 403.

Next.js 16.4.5 requires Node 20.9 or later. Browser integration tests also need
local Chrome under the documented macOS application path. Python's SQLite
must support FTS5 for Memory. The runner's measured training command uses
macOS `/usr/bin/time -l`. Model setup uses huggingface-hub and the training
runner uses PyYAML, both present on the resolved MLX-LM dependency graph.
Keep the optional Streamlit runner local and set
`STREAMLIT_BROWSER_GATHER_USAGE_STATS=false` for offline practice.

## Training run prepared for separate approval

The isolated [review manifest](../artifacts/phase11/training-preflight/review.json)
binds job 1 and candidate `phase11-pr-lifecycle-25802ca2`. Real local Qwen
generation created ten explicitly captured PR-summary traces, and ten
synthetic corrections passed through the authenticated API into the queue.
The export contains 56 curated and 10 corrected training items plus six
validation items. It passed the export's credential and duplicate checks.
Only the 1.5B base revision `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`
is eligible for this reviewed job.

The existing PR recipe is rank 4, four layers, 28 optimizer steps, LR 2e-5,
batch 1, sequence cap 512, seed 43, and assistant-token masking. The estimated
trainable count is 659,456. The review digest is
`2f6711af76c235963778291537aac62ef4bb18fe39288c77979f078efda78507`.
The actual exported SHA-256 values, model revision and reviewed parameters
are in the manifest. This is a lifecycle verification run, with the existing
PR qualification evaluation and gates. It is separate from future model
accuracy experiments.

The test suite exercised successful job completion, candidate registration,
queue consumption, interruption and failure recovery, and refusal to activate
a failed candidate using stubbed training and evaluation. No optimizer,
real-weight registration or actual qualification ran in Phase 11.

After explicit approval, start the gateway from the repository root with the
prepared isolated database, cloud keys blank and offline flags enabled. With
a configured operator token, POST the exact digest once. Keep the candidate
inactive and review returned artifacts, metrics and gate results.

```bash
ITALIX_DB_PATH="$PWD/artifacts/phase11/training-preflight/lifecycle.sqlite" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GEMINI_API_KEY= GROQ_API_KEY= \
  .venv/bin/python -m uvicorn italix.api:app --host 127.0.0.1 --port 8011
```

```bash
curl --fail-with-body -sS -X POST http://127.0.0.1:8011/v1/training/jobs/1/execute \
  -H "Authorization: Bearer ${ITALIX_OPERATOR_TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{"review_digest":"2f6711af76c235963778291537aac62ef4bb18fe39288c77979f078efda78507"}'
```

## Developer Memory

The [42-case fixture](../data/phase11/memory-v1.json) uses 23 original
synthetic sources, including six actual commits constructed in a temporary
fixture repository. All models received the same frozen, ordered and redacted
FTS5 evidence packets. The raw run records report the fixture and retrieval
SHA-256, model revisions, tokens, latency and peak MLX allocation. Complete
source retrieval was 38/39 where the draft lists expected sources. The
missing item was an older retention decision needed for the comparison.
One question matched no source and one was blocked as credential-like before
retrieval or inference.

| Model | Citation-valid answers / generated | Median generated latency | Mean decode throughput | Peak MLX |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5 1.5B | 6 / 40 | 651 ms | 139 tokens/s | 1.47 GB |
| Qwen3 4B | 40 / 40 | 2,397 ms | 58 tokens/s | 2.91 GB |
| Qwen2.5-Coder 7B | 32 / 40 | 2,734 ms | 35 tokens/s | 4.89 GB |

A citation that names a retrieved record can still accompany an unsupported
claim. An AI-assisted spot check found that Coder promoted a 4-second timeout
proposal to an adopted decision and answered an unrelated lunar database
question with PostgreSQL. Qwen3 handled the conflicting proposal but made
a categorical absence claim on a different unsupported question. Neither
finding substitutes for independent human review. Do not report a benchmark
answer-correctness rate before adjudication. The blind packet is
`artifacts/phase11/memory-blind-review.jsonl`. Keep the model mapping
`memory-review-key.json` from reviewers until their judgments are recorded.

## Public change pilot

The [provenance manifest](../data/phase11/public-pilot-manifest.json) records six
changes across 16 files. Commit and parent IDs, changed paths, license text
hashes, and input envelope hashes are recorded. The local pilot retains
before/after context and license text, along with Go patent terms. Cases stay
grouped by repository, plus any narrower change families discovered in review.
Send reviewers the local `artifacts/phase11/public-pilot/blind-review.md` and
its inputs, with a separate adjudication pass against draft facts.

Sources are [CPython](https://github.com/python/cpython), [Go](https://github.com/golang/go),
and [kubectl](https://github.com/kubernetes/kubectl), under the terms recorded
at the pinned commits. Tests and documentation in these patches provide
supporting context. No claims about running upstream tests are made. All
six cases still need independent human review.

Draft probes used one multi-file case per repository and the same three pinned
models. Probe inputs were patch hunks, while the review packet also contains
broader source context. Unfounded performance implications, implementation
omissions, and incomplete file coverage appeared in AI-assisted inspection.
Use these observations to choose a task and rubric. Keep the cases out of
future training splits if they form part of the comparison.

## Reliability and specialization

On the 16 GB M1 Pro, the fresh environment served a cold Qwen request in
2.93 s and subsequent requests at about 397 ms median. Four simultaneous
requests completed in 1.62 s wall time, with latency from 532 ms to 1.62 s.
The single MLX worker serialized generation. With approximately 6 GiB already
in system swap and a bounded 128 MiB ballast, switch plus short generation
took 3.61 s to Qwen3, 4.56 s to Coder, and 0.97 s back to base. Peak process
RSS was 2.19 GiB and peak MLX allocation was 4.49 GB. System counters
reported no swap-outs during this bounded run. A missing model yielded 503,
policy-independent deterministic summarization still succeeded, and local
generation continued to work afterward. These results cover short prompts and
bounded concurrency. Do not infer stability under sustained load or worst-case
context lengths.

A measurable specialization candidate is **cross-file evidence selection for
factual PR summaries**. Given a patch and surrounding context, return bounded
stable hunk IDs that cover each required implementation, test or documentation
fact. Compare evidence recall and precision, required-file coverage, invalid
IDs, unsupported claims in the downstream summary, and wall time on reviewed
repository-disjoint cases.

| Approach | Role in the first controlled comparison |
| --- | --- |
| Prompt optimization | Cheapest control. Use explicit file coverage and a fixed JSON schema. |
| Retrieval improvements | Compare lexical retrieval with coverage-aware candidate assembly before attributing misses to generation. |
| Discriminative evidence ranking | First targeted candidate. Rank hunks directly using graded relevance and file-coverage metrics. |
| Targeted LoRA | Compare only after reviewing a diverse corpus of corrected multi-file selections and summaries. Test whether better ranking translates to fewer unsupported claims. |

Before release, independently review both packets, enlarge the public pilot,
and measure sustained queue latency and model-switch churn under realistic
context sizes. Complete the separately approved real training run and qualify
its output with the unchanged gates. Check worker loss and operator recovery
again with real subprocess artifacts. Keep deterministic policy authoritative
and leave workload defaults unchanged until quality and resource targets are
met.
