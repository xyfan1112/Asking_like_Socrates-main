#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
source "$ROOT/train_layer/_runtime.sh"
resolve_sft_runtime "$ROOT" "$SETTINGS"

"$SFT_PYTHON" "$ROOT/train_layer/00_prepare_direct_only.py" --settings "$SETTINGS"
"$SFT_PYTHON" "$ROOT/train_layer/01d_validate_b1_direct_standalone.py" --settings "$SETTINGS"
"$SFT_PYTHON" "$ROOT/train_layer/02d_generate_b1_direct_standalone_config.py" --settings "$SETTINGS"
"$SFT_PYTHON" "$ROOT/train_layer/02e_validate_b1_direct_contract.py" --settings "$SETTINGS"
ensure_agents_stopped_for_sft "$SETTINGS"
CFG=$("$SFT_PYTHON" - "$SETTINGS" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
print(s['paths']['training_run_root']+'/configs/b1_direct_standalone_lora.yaml')
PY
)
[[ -f "$CFG" ]] || { echo "[FAIL] B1 Direct config missing: $CFG" >&2; exit 2; }
cd "$LLAMA_FACTORY_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE="$WORLD_SIZE" \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$LLAMAFACTORY_CLI" train "$CFG"
