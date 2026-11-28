# Standalone Gemini annotation

## Authorization status

Live runs for a Gemini-to-local-Qwen teacher workflow are blocked. The Gemini Developer API terms restrict developing models that compete with the services. Google's ownership language for generated content leaves that restriction in place. The intended use has no verified permission in this workspace. Obtain written confirmation covering this exact use and account before launch. The separate Google Cloud terms also contain restrictions on using outputs to create similar models or replace Google models. Switching services leaves permission to be assessed independently. [Gemini Developer API terms](https://ai.google.dev/gemini-api/terms) · [Google Cloud service-specific terms](https://cloud.google.com/terms/service-terms)

The existing dataset records pinned code-license notices and review status. It has no clearance record for transmitting the exact packets to Gemini or using the resulting labels for training. Google's API terms require rights to submitted content. Paid Gemini Developer API data handling applies through an actively billed Cloud project. Google describes limited logging and possible processing outside the user's country. Unpaid service terms generally permit use for improvement and human review, subject to stated regional exceptions. A key file gives no evidence about billing or data transmission rights. [Google APIs terms](https://developers.google.com/terms#b_submission_of_content) · [Gemini data use terms](https://ai.google.dev/gemini-api/terms#data-use-paid)

The live CLI takes a local authorization receipt. It must come from an actual provider permission and per-case source rights review. The operator records the permission references, the exact prepared manifest SHA-256, the approved case IDs, the covered model, provider `gemini-developer-api-paid`, and purpose `local-qwen-student-training`. An authorization receipt needs these fields:

`manifest_sha256`, `model`, `provider`, `purpose`, `provider_permission_reference`, `source_rights_review_reference`, `approved_case_ids`.

The launcher checks scope and manifest binding locally. Legal validity and actual account billing must be verified through the responsible parties. This integration has no live smoke test. Server-side structured output support and billing behavior still need validation on an authorized account. [Gemini generation REST API](https://ai.google.dev/api/generate-content) · [Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing)

## Prepare calibration offline

```bash
cd /Users/alihassan/Projects/italix.ai
source .venv/bin/activate
umask 077
RUN="$(mktemp -d "$HOME/italix-gemini-annotation.XXXXXX")"
P14="$HOME/italix-phase14/pilot"
P15="$HOME/italix-phase15"
P16="$HOME/italix-phase16"
GOTOOLCHAIN=local GOPROXY=off GOSUMDB=off go build \
  -o "$RUN/go-index" scripts/go_change_index/main.go

python -m scripts.gemini_annotation prepare \
  "$P14/review-v2.jsonl" "$RUN/calibration.jsonl" \
  --size 20 --packet "$P15/review-20-blind.key.json" \
  --mirrors "$HOME/italix-phase14/sources" --go-index "$RUN/go-index"

shasum -a 256 "$RUN/calibration.jsonl"
```

## Prepare a bulk manifest offline

```bash
SIZE=100
POOL="$P16/expansion/training-candidate-pool.jsonl"
python -c 'import json,sys; print(json.dumps({k:v for p in sys.argv[1:] for k,v in json.load(open(p)).items()}))' \
  "$P16/expansion/source-catalog.json" "$P16/expansion/top-up-catalog.json" > "$RUN/catalog.json"

python -m scripts.gemini_annotation prepare \
  "$POOL" "$RUN/bulk-$SIZE.jsonl" --size "$SIZE" \
  --catalog "$RUN/catalog.json" --mirrors "$P16/expansion/mirrors" \
  --go-index "$RUN/go-index"
shasum -a 256 "$RUN/bulk-$SIZE.jsonl"
```

Supported bulk sizes are 100, 500, 1,000 and 3,000. Change `SIZE` and point `POOL` to a sufficiently large source inventory for that run. The existing pool contains 498 non-rejected source candidates. The command fails if the requested count is unavailable. The local preparation step validates pinned changes against known mirrors and excludes held splits. It does not grant transmission permission.

## Manual live commands after permission and clearance

Put the actual scoped receipts in `$RUN/calibration.approval.json` and `$RUN/bulk-$SIZE.approval.json` respectively. Enter the approved model and the verified paid per-million-token USD rates. Use the rates covering the chosen context sizes and billable reasoning tokens.

```bash
printf 'Approved Gemini model: '; read -r MODEL
printf 'Verified input USD per million tokens: '; read -r INPUT_RATE
printf 'Verified output USD per million tokens: '; read -r OUTPUT_RATE
printf 'Calibration budget in USD: '; read -r CALIBRATION_BUDGET

run_gemini() (
  set -o pipefail
  python -m scripts.gemini_annotation run "$RUN/$1.jsonl" \
    "$RUN/$1.approval.json" "$RUN/$1.sqlite" \
    --env .env --model "$MODEL" --budget "$2" \
    --input-rate "$INPUT_RATE" --output-rate "$OUTPUT_RATE" \
    --max-input 8192 --max-output 1024 --concurrency 2 --retries 3 \
    2>&1 | tee -a "$RUN/$1.log"
)

run_gemini calibration "$CALIBRATION_BUDGET"
```

Review the calibration before launching bulk work. Use the same function for launch and resumption:

```bash
printf 'Bulk budget in USD: '; read -r BULK_BUDGET
run_gemini "bulk-$SIZE" "$BULK_BUDGET"

# After interruption, repeat with the same source, receipt, model, rates and state.
run_gemini "bulk-$SIZE" "$BULK_BUDGET"
```

Budget, concurrency and retry limits can be adjusted on resume. The SQLite checkpoint persists usage reservations before requests, and skips completed records. An interrupted submitted attempt keeps its reservation and its case can be resubmitted. This means provider billing can include both attempts. The budget is a local accounting guard, subject to verified pricing, actual provider usage, taxes and other provider fees. Configure provider-side billing controls for invoice protection. Each case uses a token count request before generation. Log files contain completion progress and accounted cost. The SQLite ledger retains attempt status and reported prompt, completion and reasoning token totals without provider bodies.

## Local evaluation and export

```bash
python -m scripts.gemini_annotation evaluate "$RUN/calibration.sqlite" \
  --references "$P16/full-100/qc-final/qc-corpus.jsonl" \
  --output "$RUN/calibration-evaluation.json"
python -m scripts.gemini_annotation evaluate "$RUN/bulk-$SIZE.sqlite"
python -m scripts.gemini_annotation export "$RUN/bulk-$SIZE.sqlite" \
  --output "$RUN/bulk-$SIZE.provisional.jsonl"
```

Evaluation reports screening status and AI-reference citation overlap. Citation overlap is a diagnostic proxy for recall and does not measure entailment. Inspect drafts against the pinned source before admission. The exporter provides candidate provenance, selected evidence, provider model version and pending review status. Every generated record is ineligible for training until it passes existing independent human and rights review. Rejected drafts carry reason codes.

## Explicit free-tier runs

Use a key from a verified Free Tier project with Cloud Billing disabled, in a region where the approved use is available. Confirm this in Google AI Studio and Cloud Billing before each run. The Gemini inference endpoint has no client-side switch that forces requests from a paid project into free quota. `--tier free` never configures billing, changes accounts, or falls back to a paid tier. [Billing documentation](https://ai.google.dev/gemini-api/docs/billing) · [Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)

For a free run, the actual authorization receipt must name provider `gemini-developer-api-free` and preserve the existing provider training-use permission and per-case source transmission review. It also requires `billing_disabled: true`, `public_nonsensitive_source_reviewed: true`, an `unpaid_data_handling_approval_reference`, and the reviewed key's `api_key_sha256`. These are attestations from the responsible reviewers. The program cannot verify billing remotely. Google may use unpaid prompts and responses for product improvement and human review, subject to its regional terms. Keep sensitive, confidential, and personal data out of the packets. The CLI also screens for credentials and email addresses. [Gemini API terms](https://ai.google.dev/gemini-api/terms#data-use-unpaid)

If the project and key in `.env` have already been verified, this saves only the fingerprint to a fresh private file. The responsible reviewer uses that fingerprint when issuing the receipt.

```bash
cd /Users/alihassan/Projects/italix.ai
source .venv/bin/activate
umask 077
RUN="${RUN:-$HOME/italix-gemini-manual}"
mkdir -p "$RUN"
python - "$RUN/free-key.sha256" <<'PY'
from dotenv import dotenv_values
from hashlib import sha256
from pathlib import Path
import sys

key = dotenv_values(".env", interpolate=False).get("GEMINI_API_KEY")
if not key:
    raise SystemExit("GEMINI_API_KEY missing")
with Path(sys.argv[1]).open("x") as output:
    output.write(sha256(key.encode()).hexdigest() + "\n")
PY
```

Prepare the source manifest using the earlier offline commands. Set `MODEL` to the exact approved model listed for your free project and save the actual reviewed receipt as `$RUN/calibration.free.approval.json`. The same command starts or resumes the run.

```bash
printf 'Approved free-tier model: '; read -r MODEL
JOB=calibration
(
  set -o pipefail
  python -m scripts.gemini_annotation run \
    "$RUN/$JOB.jsonl" "$RUN/$JOB.free.approval.json" "$RUN/$JOB.free.sqlite" \
    --tier free --env .env --model "$MODEL" --concurrency 1 \
    2>&1 | tee -a "$RUN/$JOB.free.log"
)
python -m scripts.gemini_annotation evaluate "$RUN/$JOB.free.sqlite"
```

Omit the paid rate arguments. Free mode accounts all token costs at zero while recording reported usage. It sends one case at a time. On HTTP 429 it saves the checkpoint and pauses for the longer of `Retry-After` and 60 seconds. When the dashboard shows quota is available, rerun the same command after `resume_after_utc`. Longer quota windows can require a longer wait. Paid checkpoint files remain separate and cannot be switched to free mode. Use [AI Studio usage](https://aistudio.google.com/usage) to check project quota.
