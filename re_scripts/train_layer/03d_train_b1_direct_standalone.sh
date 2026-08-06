#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
source "$ROOT/train_layer/_runtime.sh"
resolve_sft_runtime "$ROOT" "$SETTINGS"

"$SFT_PYTHON" "$ROOT/train_layer/00_prepare_direct_only.py" --settings "$SETTINGS"
"$SFT_PYTHON" "$ROOT/train_layer/01d_validate_b1_direct_standalone.py" --settings "$SETTINGS"
"$SFT_PYTHON" "$ROOT/train_layer/02d_generate_b1_direct_standalone_config.py" --settings "$SETTINGS"
"$SFT_PYTHON" "$ROOT/train_layer/02e_validate_b1_direct_contract.py" --settings "$SETTINGS"
ensure_agents_stopped_for_sft "$SETTINGS"
readarray -t META < <("$SFT_PYTHON" - "$SETTINGS" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
print(s['paths']['training_run_root']+'/configs/b1_direct_standalone_lora.yaml')
print(s['paths']['training_run_root'])
print(str(s.get('v1_2_3',{}).get('sft',{}).get('freeze_vision_tower',s['training'].get('freeze_vision_tower',True))).lower())
PY
)
CFG=${META[0]}; RUN_ROOT=${META[1]}; FREEZE=${META[2]}
[[ -f "$CFG" ]] || { echo "[FAIL] B1 Direct config missing: $CFG" >&2; exit 2; }
REPORT_ARGS=(--settings "$SETTINGS" --config "$CFG" --target b1_direct)
[[ "$FREEZE" == "true" ]] && REPORT_ARGS+=(--require-frozen-vision)
"$SFT_PYTHON" "$ROOT/tools/report_training_contract.py" "${REPORT_ARGS[@]}"

mkdir -p "$RUN_ROOT/logs"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$RUN_ROOT/logs/b1_direct_train_${STAMP}.log"
echo "[TRAIN] B1 Direct DDP GPUs=$CUDA_DEVICES world_size=$WORLD_SIZE log=$LOG"
cd "$LLAMA_FACTORY_DIR"
set +e
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE="$WORLD_SIZE" \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1 \
  llamafactory_cli train "$CFG" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e
"$SFT_PYTHON" "$ROOT/tools/extract_training_log_report.py" --settings "$SETTINGS" --log "$LOG" --target b1_direct || true
(( RC == 0 )) || { echo "[FAIL] B1 Direct training failed rc=$RC; log=$LOG" >&2; exit "$RC"; }

ADAPTER_DIR=$("$SFT_PYTHON" - "$SETTINGS" <<'PYADAPTER'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
print(s['training']['b1_direct_standalone_adapter_output'])
PYADAPTER
)
SCOPE_ARGS=(--settings "$SETTINGS" --adapter-dir "$ADAPTER_DIR" --target b1_direct)
[[ "$FREEZE" == "true" ]] && SCOPE_ARGS+=(--require-frozen-vision)
"$SFT_PYTHON" "$ROOT/tools/audit_adapter_trainable_scope.py" "${SCOPE_ARGS[@]}"
echo "[PASS] B1 Direct training completed; log=$LOG"
