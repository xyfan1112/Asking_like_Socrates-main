#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
GROUP=${2:-socratic}   # direct | socratic
source "$ROOT/train_layer/_runtime.sh"
resolve_sft_runtime "$ROOT" "$SETTINGS"
"$SFT_PYTHON" "$ROOT/train_layer/02b_generate_official_llamafactory_configs.py" --settings "$SETTINGS"
CFG=$("$SFT_PYTHON" - "$SETTINGS" "$GROUP" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8')); group=sys.argv[2]
name='b1_direct_official.yaml' if group=='direct' else 'b2_socratic_official.yaml'
print(s['paths']['training_run_root']+'/configs/'+name)
PY
)
[[ -f "$CFG" ]] || { echo "[FAIL] config missing: $CFG" >&2; exit 2; }
cd "$LLAMA_FACTORY_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE="$WORLD_SIZE" \
  llamafactory_cli train "$CFG"
