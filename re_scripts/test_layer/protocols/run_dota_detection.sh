#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}; KEY=${2:?model key}; URL=${3:-http://127.0.0.1:8010/v1}; SPLIT=${4:-val}
RUN_POLICY=${5:-resume}
BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
[[ -n "$BOOTSTRAP_PY" ]] || { echo "[FAIL] 找不到 Python" >&2; exit 2; }
DATA_PY=$("$BOOTSTRAP_PY" "$ROOT/main_layer/resolve_runtime.py" --settings "$SETTINGS" --workload data --field python)
EXTRA=()
[[ "$RUN_POLICY" == "fresh" ]] && EXTRA+=(--fresh)
"$DATA_PY" "$ROOT/test_layer/01_raw_dota_obb_infer.py" --settings "$SETTINGS" --model-key "$KEY" --base-url "$URL" --split "$SPLIT" --protocol dota_class_conditioned "${EXTRA[@]}"
PRED=$("$DATA_PY" - "$SETTINGS" "$KEY" "$SPLIT" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'));print(f"{s['paths']['test_run_root']}/dota_detection/{sys.argv[2]}/{sys.argv[3]}_dota_class_conditioned_roi_v2_k1.jsonl")
PY
)
"$DATA_PY" "$ROOT/test_layer/02_raw_dota_obb_metrics.py" --settings "$SETTINGS" --pred "$PRED"
