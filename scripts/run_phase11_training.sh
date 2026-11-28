#!/bin/bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
export GEMINI_API_KEY="" GROQ_API_KEY="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ITALIX_HYBRID_MODEL=""
exec "$root/.venv/bin/python" -m scripts.phase11_training "$@"
