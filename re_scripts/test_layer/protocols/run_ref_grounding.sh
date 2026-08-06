#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}; KEY=${2:?model key}; URL=${3:-http://127.0.0.1:8010/v1}; DATASET=${4:-dota_ref}; MODE=${5:-ref_grounding}
RUN_POLICY=${6:-resume}
BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
[[ -n "$BOOTSTRAP_PY" ]] || { echo "[FAIL] 找不到 Python" >&2; exit 2; }
DATA_PY=$("$BOOTSTRAP_PY" "$ROOT/main_layer/resolve_runtime.py" --settings "$SETTINGS" --workload data --field python)
EXTRA=()
[[ "$RUN_POLICY" == "fresh" ]] && EXTRA+=(--fresh)
"$DATA_PY" "$ROOT/test_layer/03_ref_grounding_infer.py" --settings "$SETTINGS" --model-key "$KEY" --base-url "$URL" --dataset "$DATASET" --protocol "$MODE" --split val "${EXTRA[@]}"
K=$("$DATA_PY" - "$SETTINGS" "$MODE" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'));print(s['evaluation']['protocols'][sys.argv[2]]['k'])
PY
)
PRED=$("$DATA_PY" - "$SETTINGS" "$KEY" "$DATASET" "$MODE" "$K" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'));print(f"{s['paths']['test_run_root']}/ref_grounding/{sys.argv[3]}/{sys.argv[2]}/val_{sys.argv[4]}_k{sys.argv[5]}.jsonl")
PY
)
"$DATA_PY" "$ROOT/test_layer/04_ref_grounding_metrics_k.py" --settings "$SETTINGS" --pred "$PRED"
