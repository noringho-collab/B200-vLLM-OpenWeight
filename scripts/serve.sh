#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
: "${VLLM_API_KEY:?Set VLLM_API_KEY in your local .env before starting.}"
export VLLM_API_KEY
# Use the downloaded commit, not a mutable Hub main branch at every restart.
MODEL_PATH=$("$PY" - "$ROOT/.runtime/model.json" "$MODEL_ID" "$MODEL_REVISION" <<'PY'
import json
from pathlib import Path
import sys
record = json.loads(Path(sys.argv[1]).read_text())
if record['model_id'] != sys.argv[2]:
    raise SystemExit('MODEL_ID changed: rerun bash scripts/download.sh')
if sys.argv[3] != 'main' and record['revision'] != sys.argv[3]:
    raise SystemExit('Use MODEL_REVISION=main or the downloaded full commit SHA; rerun download if needed.')
if not Path(record['snapshot']).is_dir():
    raise SystemExit('Model cache missing: rerun download.sh')
print(record['snapshot'])
PY
)
args=(serve "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
  --tensor-parallel-size 1 --dtype bfloat16
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.40}"
  --max-model-len "${MAX_MODEL_LEN:-8192}"
  --max-num-seqs "${MAX_NUM_SEQS:-4}"
  --max-num-batched-tokens 8192
  --generation-config vllm --enforce-eager)
exec "$ROOT/.venv/bin/vllm" "${args[@]}"
