# Phase 12 manual operator runbook

The only training plan covered here is Phase 11 job 1,
`phase11-pr-lifecycle-25802ca2`. Its approved digest is
`2f6711af76c235963778291537aac62ef4bb18fe39288c77979f078efda78507`.
The launcher regenerated the 66/6 export in a temporary directory and verified
it against the reviewed files and hashes. It validated the planned model
revision, the nine cached payload file hashes, and the recipe. It refuses
start if any of these change or if job or candidate artifacts already exist.

Training settings are unchanged. Rank 4, four layers, 28 steps, LR 2e-5, batch
1, limit 512, seed 43, scale 8 and zero dropout. The training and validation
dataset hashes are
`ab910112533d8feb59d8a194e2b850ce670390fc77b3d2b4c69cb99c545c4f0f` and
`f2a01afb455699c18066b8e5af1307f0ec1859ec7fa5ced50ec8af0cff64e018`.
The base revision is `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`.
No training started during preflight.

## Terminal 1, verify and serve

Run from the repository root. This creates isolated local credentials in an
ignored directory with owner-only permissions. Use the same file in each
terminal. The launcher pins the isolated Phase 11 database, a single loopback
worker, offline flags, and blank cloud keys.

```bash
umask 077
secrets_file=artifacts/phase11/training-preflight/operator-env.sh
if [ ! -e "$secrets_file" ]; then
  {
    printf 'export ITALIX_OPERATOR_TOKEN=%s\n' "$(openssl rand -hex 32)"
    printf 'export ITALIX_VIEWER_TOKEN=%s\n' "$(openssl rand -hex 32)"
    printf 'export ITALIX_TRAINING_PORT=8011\n'
  } > "$secrets_file"
fi
source "$secrets_file"
./scripts/run_phase11_training.sh verify
./scripts/run_phase11_training.sh serve
```

## Terminal 2, start and observe

The start call re-verifies the local files and compares the served review
digest before submitting the authenticated execution request. Launch once.
Status can be polled before, during and after execution. A lost response does
not justify retrying. Query status first.

```bash
source artifacts/phase11/training-preflight/operator-env.sh
./scripts/run_phase11_training.sh status
./scripts/run_phase11_training.sh start --confirm phase11-pr-lifecycle-25802ca2
./scripts/run_phase11_training.sh watch
```

In a third terminal, stream redacted logs and optionally list checkpoints.

```bash
./scripts/run_phase11_training.sh logs
find artifacts/adapters/phase11-pr-lifecycle-25802ca2 -maxdepth 1 -type f -print
```

Ctrl-C in a status or log terminal stops that display. To stop the optimizer,
keep the server running and run the authenticated interruption command.
It signals the process group for this reviewed worker only. Check status until
it becomes terminal. Completed checkpoints and existing artifacts remain in
place. A checkpoint interrupted during a write may be unusable.

```bash
source artifacts/phase11/training-preflight/operator-env.sh
./scripts/run_phase11_training.sh interrupt
./scripts/run_phase11_training.sh status
```

If the server was lost, restart `serve` from Terminal 1 and check status. Use
`recover` only if the job remains running and the previous process has exited.
The lock prevents recovery while an old worker is still alive. Recovery marks
the interrupted job failed and returns its examples to the queue. Normal
failures already return the examples. This job is single-use. Keep the
checkpoint, log, config and dataset export directories, and create a newly
reviewed plan and candidate for a separately authorized retry. The runner
does not resume optimizer state.

```bash
source artifacts/phase11/training-preflight/operator-env.sh
./scripts/run_phase11_training.sh status
./scripts/run_phase11_training.sh recover
./scripts/run_phase11_training.sh status
```

## Evaluate a completed candidate

Preview uses the following synthetic patch and never captures feedback. The
read-only evaluation computes the existing gate without updating the registry.
Qualification separately recomputes it and records the result. No command
activates the adapter.

```bash
source artifacts/phase11/training-preflight/operator-env.sh
reports="$(mktemp -d artifacts/phase11/training-preflight/evaluation.XXXXXX)"
./scripts/run_phase11_training.sh preview --patch data/phase12/preview.diff
./scripts/run_phase11_training.sh evaluate > "$reports/evaluation-preview.json"
./scripts/run_phase11_training.sh qualify > "$reports/qualification-report.json"
./scripts/run_phase11_training.sh qualification
```

## Phase 12 validation

Coverage-aware candidate selection gives each changed file a slot before
taking more lines from any file. The model can choose evidence, and
deterministic supplementation fills missing file slots within the output
budget. The hybrid response and Gateway show coverage. The path policy
continues to determine classification. The draft evidence fixture contains
ten original synthetic changes and references the six Phase 11 public cases
as evaluation-only. It needs independent review. Using the three-line output
budget, the prefix baseline covered 14 of 25 expected files and the coverage
baseline covered 24 of 25. The remaining file is beyond the budget of a
four-file patch. Covered paths and preferred synthetic line IDs are structural
checks, with semantic usefulness pending human review.

Memory answer retrieval now reranks by corroborating query terms. It keeps
source search available even when evidence is too weak for an answer, and the
prompt distinguishes proposals from adopted or superseded decisions. The
retrieval rule is a lexical heuristic. On the same 42-case Qwen3 diagnostic,
the unrelated lunar database question abstained before generation. Complete
source retrieval stayed at 38/39. Two generated answers failed citation
validation, leaving 37/39 citation-valid answers versus 40/40 before. These
diagnostic outcomes still require human entailment review.

A four-client sustained run completed 160 requests in 59.36 seconds with
zero failures, 1.48 s median and 1.53 s p95. Single-request warm median was
371 ms. The four-client latency reflects one shared MLX worker and queue
waiting. Under a 128 MiB ballast and existing system swap, model switching
remained functional and a missing-cache 503 did not prevent later inference.
The sample covers short prompts and about one minute of sustained activity.
