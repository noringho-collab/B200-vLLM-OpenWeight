#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
"$PY" "$ROOT/scripts/download_model.py" "$MODEL_ID" "$MODEL_REVISION" "$ROOT/.runtime/model.json"
