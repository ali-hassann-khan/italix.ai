#!/usr/bin/env bash
# Run an explicitly approved command in the foreground and record resource use.
set -euo pipefail
umask 077
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 NEW_RESOURCE_DIRECTORY COMMAND [ARGS...]" >&2
    exit 2
fi
resource_dir=$1
shift
mkdir -m 700 "$resource_dir"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
monitor=""
cleanup() {
    if [[ -n "$monitor" ]]; then
        kill "$monitor" 2>/dev/null || true
        wait "$monitor" 2>/dev/null || true
    fi
}
trap cleanup EXIT
if [[ $(uname -s) == Darwin ]]; then
    /usr/bin/time -l "$@" 2> >(tee "$resource_dir/resource.log" >&2)
else
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu             --format=csv --loop=2 > "$resource_dir/gpu.csv" 2> "$resource_dir/gpu-errors.log" &
        monitor=$!
    fi
    /usr/bin/time -v "$@" 2> >(tee "$resource_dir/resource.log" >&2)
fi
