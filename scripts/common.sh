#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$ROOT/.env" ]]; then
  set -a
  source "$ROOT/.env"
  set +a
fi
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_NO_USAGE_STATS=1
PY="$ROOT/.venv/bin/python"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
MODEL_REVISION="${MODEL_REVISION:-main}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8b}"
mkdir -p "$ROOT/.runtime"
umask 077
