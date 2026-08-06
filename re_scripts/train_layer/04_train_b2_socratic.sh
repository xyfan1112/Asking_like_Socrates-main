#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
source "$ROOT/train_layer/_runtime.sh"
resolve_sft_runtime "$ROOT" "$SETTINGS"
"$SFT_PYTHON" "$ROOT/train_layer/01_validate_training_data.py" --settings "$SETTINGS" --target b2
"$SFT_PYTHON" "$ROOT/train_layer/02_generate_configs.py" --settings "$SETTINGS"
"$SFT_PYTHON" "$ROOT/train_layer/02a_validate_llamafactory_contract.py" --settings "$SETTINGS" --target b2
ensure_agents_stopped_for_sft "$SETTINGS"
readarray -t META < <("$SFT_PYTHON" - "$SETTINGS" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
print(s['paths']['training_run_root']+'/configs/b2_socratic_lora.yaml')
print(s['paths']['training_run_root'])
print(str(s.get('v1_2_3',{}).get('sft',{}).get('freeze_vision_tower',s['training'].get('freeze_vision_tower',True))).lower())
PY
)
CFG=${META[0]}; RUN_ROOT=${META[1]}; FREEZE=${META[2]}
[[ -f "$CFG" ]] || { echo "[FAIL] config missing: $CFG" >&2; exit 2; }
REPORT_ARGS=(--settings "$SETTINGS" --config "$CFG" --target b2_socratic)
[[ "$FREEZE" == "true" ]] && REPORT_ARGS+=(--require-frozen-vision)
"$SFT_PYTHON" "$ROOT/tools/report_training_contract.py" "${REPORT_ARGS[@]}"
mkdir -p "$RUN_ROOT/logs"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$RUN_ROOT/logs/b2_socratic_train_${STAMP}.log"
echo "[TRAIN] B2 Socratic four-GPU DDP log=$LOG"
cd "$LLAMA_FACTORY_DIR"
set +e
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE="$WORLD_SIZE" \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1 \
  llamafactory_cli train "$CFG" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e
"$SFT_PYTHON" "$ROOT/tools/extract_training_log_report.py" --settings "$SETTINGS" --log "$LOG" --target b2_socratic || true
(( RC == 0 )) || { echo "[FAIL] B2 Socratic training failed rc=$RC; log=$LOG" >&2; exit "$RC"; }

ADAPTER_DIR=$("$SFT_PYTHON" - "$SETTINGS" <<'PYADAPTER'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
print(s['training']['b2_adapter_output'])
PYADAPTER
)
SCOPE_ARGS=(--settings "$SETTINGS" --adapter-dir "$ADAPTER_DIR" --target b2_socratic)
[[ "$FREEZE" == "true" ]] && SCOPE_ARGS+=(--require-frozen-vision)
"$SFT_PYTHON" "$ROOT/tools/audit_adapter_trainable_scope.py" "${SCOPE_ARGS[@]}"
echo "[PASS] B2 Socratic training completed; log=$LOG"
