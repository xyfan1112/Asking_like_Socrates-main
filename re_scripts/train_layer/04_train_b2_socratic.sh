#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
source "$ROOT/train_layer/_runtime.sh"
resolve_sft_runtime "$ROOT" "$SETTINGS"
"$SFT_PYTHON" "$ROOT/train_layer/01_validate_training_data.py" --settings "$SETTINGS" --target b2
"$SFT_PYTHON" "$ROOT/train_layer/02_generate_configs.py" --settings "$SETTINGS"
"$SFT_PYTHON" "$ROOT/train_layer/02a_validate_llamafactory_contract.py" --settings "$SETTINGS" --target b2
ensure_agents_stopped_for_sft "$SETTINGS"
CFG=$("$SFT_PYTHON" - "$SETTINGS" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
print(s['paths']['training_run_root']+'/configs/b2_socratic_lora.yaml')
PY
)
[[ -f "$CFG" ]] || { echo "[FAIL] config missing: $CFG" >&2; exit 2; }
cd "$LLAMA_FACTORY_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE="$WORLD_SIZE" \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$LLAMAFACTORY_CLI" train "$CFG"
