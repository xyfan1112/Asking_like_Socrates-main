#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
[[ -n "$BOOTSTRAP_PY" ]] || { echo "[FAIL] 找不到 Python" >&2; exit 2; }
DATA_PY=$("$BOOTSTRAP_PY" "$ROOT/main_layer/resolve_runtime.py" --settings "$SETTINGS" --workload data --field python)
JSONL=$("$DATA_PY" - "$SETTINGS" <<'PY'
import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8'))['paths']['vrsbench_result_jsonl'])
PY
)
"$DATA_PY" "$ROOT/test_layer/07a_check_vrsbench_k_runs.py" --jsonl "$JSONL" --k 5
"$DATA_PY" "$ROOT/test_layer/07_score_existing_vrsbench.py" --settings "$SETTINGS"
