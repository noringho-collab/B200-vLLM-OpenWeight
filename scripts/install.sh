#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
python3 "$ROOT/scripts/preflight.py"
if [[ -e "$ROOT/.venv" && ! -f "$ROOT/.venv/.serving-owned" ]]; then
  echo 'Existing unowned .venv: preserve it and select a clean checkout/directory.' >&2
  exit 1
fi
if [[ ! -d "$ROOT/.venv" ]]; then
  python3 -m venv "$ROOT/.venv" || {
    echo 'venv/ensurepip unavailable: ask MSP for Python 3.12 venv support; do not use global pip.' >&2
    exit 1
  }
  touch "$ROOT/.venv/.serving-owned"
fi
"$PY" -m pip install --index-url https://pypi.org/simple 'pip==25.3'
# Official hashed CUDA wheels are selected explicitly in requirements-serving.txt.
"$PY" -m pip install --index-url https://pypi.org/simple --only-binary=:all: \
  -r "$ROOT/requirements-serving.lock"
"$PY" -m pip check
"$PY" "$ROOT/scripts/preflight.py" --gpu-check
"$PY" -m pip freeze > "$ROOT/.runtime/installed-packages.txt"
echo 'Installed and GPU checked. Next: bash scripts/download.sh'
