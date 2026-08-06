#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
SPLIT=${2:-train}
MODE=${3:-full}
DEBUG_SAMPLES=${4:-2}
RUN_POLICY=${5:-auto}
BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
[[ -n "$BOOTSTRAP_PY" ]] || { echo "[FAIL] 找不到 Python" >&2; exit 2; }
DATA_PY=$("$BOOTSTRAP_PY" "$ROOT/main_layer/resolve_runtime.py" --settings "$SETTINGS" --workload data --field python)
AGENT_INPUT=$("$DATA_PY" - "$SETTINGS" "$SPLIT" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
print(f"{s['paths']['pipeline_work_root']}/agent_inputs/{sys.argv[2]}_agent_inputs.jsonl")
PY
)
if [[ ! -s "$AGENT_INPUT" ]]; then
  echo "[AUTO] 缺少 $AGENT_INPUT，先执行数据层。"
  bash "$ROOT/main_layer/run_data_full.sh" "$SETTINGS"
fi
"$DATA_PY" "$ROOT/data_layer/official_socratic/01_build_official_parquet.py" --settings "$SETTINGS" --split "$SPLIT"
"$DATA_PY" "$ROOT/data_layer/agents/check_three_agents.py" --settings "$SETTINGS"
bash "$ROOT/data_layer/official_socratic/02_run_official_generation.sh" "$SETTINGS" "$SPLIT" "$MODE" "$DEBUG_SAMPLES" "$RUN_POLICY"
if [[ "$MODE" == "debug" ]]; then
  echo "[DEBUG GATE PASS] debug audit met every v4.3.3 quality threshold."
  echo "[NEXT] Only now may you run official-socratic-full in full/fresh mode."
  exit 0
fi
bash "$ROOT/data_layer/official_socratic/03_run_official_postproc.sh" "$SETTINGS" "$SPLIT"
"$DATA_PY" "$ROOT/data_layer/official_socratic/04_convert_official_to_llamafactory.py" --settings "$SETTINGS" --split "$SPLIT" --register
"$DATA_PY" "$ROOT/data_layer/05c_audit_lineage.py" --settings "$SETTINGS"
"$DATA_PY" "$ROOT/train_layer/01_validate_training_data.py" --settings "$SETTINGS"
"$DATA_PY" "$ROOT/train_layer/02b_generate_official_llamafactory_configs.py" --settings "$SETTINGS"
echo "[OK] official SocraticAgent mainline completed for split=$SPLIT."
