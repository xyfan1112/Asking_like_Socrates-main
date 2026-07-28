#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
KEY=${2:?model key}
URL=${3:-http://127.0.0.1:8010/v1}
RUN_POLICY=${4:-resume}
BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
[[ -n "$BOOTSTRAP_PY" ]] || { echo "[FAIL] 找不到 Python" >&2; exit 2; }
DATA_PY=$("$BOOTSTRAP_PY" "$ROOT/main_layer/resolve_runtime.py" --settings "$SETTINGS" --workload data --field python)
EXTRA=()
[[ "$RUN_POLICY" == "fresh" ]] && EXTRA+=(--fresh)
"$DATA_PY" "$ROOT/test_layer/10_ref_classification_infer.py" \
  --settings "$SETTINGS" --model-key "$KEY" --base-url "$URL" --split val "${EXTRA[@]}"
PRED=$("$DATA_PY" - "$SETTINGS" "$KEY" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
k=s['evaluation']['protocols']['ref_classification']['k']
print(f"{s['paths']['test_run_root']}/ref_classification/{sys.argv[2]}/val_k{k}.jsonl")
PY
)
"$DATA_PY" "$ROOT/test_layer/11_ref_classification_metrics.py" --settings "$SETTINGS" --pred "$PRED"
