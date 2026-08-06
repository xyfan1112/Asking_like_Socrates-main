#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
SPLIT=${2:-train}
BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
[[ -n "$BOOTSTRAP_PY" ]] || { echo "[FAIL] Python not found" >&2; exit 2; }
PYTHON_BIN=$("$BOOTSTRAP_PY" "$ROOT/main_layer/resolve_runtime.py" --settings "$SETTINGS" --workload data)
"$PYTHON_BIN" "$ROOT/data_layer/official_socratic/00_manage_run_state.py" reset --settings "$SETTINGS" --split "$SPLIT"
echo "[OK] Old canonical Socratic outputs archived; Parquet and Direct data were kept."
