#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
export SERVED_MODEL_NAME
export API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:${PORT:-8000}}"
exec python3 "$ROOT/scripts/smoke.py"
