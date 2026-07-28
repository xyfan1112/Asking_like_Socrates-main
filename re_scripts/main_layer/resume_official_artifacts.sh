#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
SPLIT=${2:-train}
API_LOG_ARG=${3:-}
BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
[[ -n "$BOOTSTRAP_PY" ]] || { echo "[FAIL] Python not found" >&2; exit 2; }
DATA_PY=$("$BOOTSTRAP_PY" "$ROOT/main_layer/resolve_runtime.py" --settings "$SETTINGS" --workload data --field python)
mapfile -d '' -t PATHS < <("$DATA_PY" - "$SETTINGS" "$SPLIT" <<'INNERPY'
import json,sys
from pathlib import Path
s=json.load(open(sys.argv[1],encoding='utf-8')); split=sys.argv[2]
raw=Path(s['official_socratic']['raw_output_dir'])
for v in (raw/f'dota128_{split}_official.jsonl', raw/f'dota128_{split}_official.manifest.json', raw):
    sys.stdout.write(str(v)); sys.stdout.write('\0')
INNERPY
)
RAW=${PATHS[0]}; MANIFEST=${PATHS[1]}; RAW_DIR=${PATHS[2]}
[[ -s "$RAW" ]] || { echo "[FAIL] canonical full raw missing: $RAW" >&2; exit 2; }
[[ -s "$MANIFEST" ]] || { echo "[FAIL] completed manifest missing: $MANIFEST" >&2; exit 2; }
if [[ -n "$API_LOG_ARG" ]]; then
  API_LOG=$API_LOG_ARG
else
  API_LOG=$(ls -t "$RAW_DIR"/local_api_calls_${SPLIT}_full_*.jsonl 2>/dev/null | head -1 || true)
fi
[[ -n "${API_LOG:-}" && -s "$API_LOG" ]] || { echo "[FAIL] full API log not found; pass it as third argument" >&2; exit 2; }
echo "[INFO] Reusing completed raw: $RAW"
echo "[INFO] Using API log: $API_LOG"
"$DATA_PY" "$ROOT/data_layer/official_socratic/02c_audit_full_generation.py" --raw "$RAW" --api-log "$API_LOG"
bash "$ROOT/data_layer/official_socratic/03_run_official_postproc.sh" "$SETTINGS" "$SPLIT"
"$DATA_PY" "$ROOT/data_layer/official_socratic/04_convert_official_to_llamafactory.py" --settings "$SETTINGS" --split "$SPLIT" --register
"$DATA_PY" "$ROOT/data_layer/05c_audit_lineage.py" --settings "$SETTINGS"
"$DATA_PY" "$ROOT/train_layer/01_validate_training_data.py" --settings "$SETTINGS"
"$DATA_PY" "$ROOT/train_layer/02b_generate_official_llamafactory_configs.py" --settings "$SETTINGS"
echo "[OK] Existing full raw was strictly filtered and converted; no regeneration was performed."
