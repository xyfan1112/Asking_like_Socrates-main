#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
source "$ROOT/train_layer/_runtime.sh"
resolve_sft_runtime "$ROOT" "$SETTINGS"
"$SFT_PYTHON" "$ROOT/train_layer/02d_generate_b1_direct_standalone_config.py" --settings "$SETTINGS"
readarray -t META < <("$SFT_PYTHON" - "$SETTINGS" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8')); t=s['training']
print(s['paths']['training_run_root'])
print(t['b1_direct_standalone_adapter_output'])
print(t['b1_direct_standalone_merged_output'])
PY
)
RUN=${META[0]}; ADAPTER_ROOT=${META[1]}; MERGED=${META[2]}
resolve_adapter() {
  local root=$1
  if [[ -f "$root/adapter_config.json" ]] && { [[ -f "$root/adapter_model.safetensors" ]] || [[ -f "$root/adapter_model.bin" ]]; }; then
    echo "$root"; return 0
  fi
  local latest
  latest=$(find "$root" -maxdepth 2 -type f -name adapter_config.json 2>/dev/null | sed 's#/adapter_config.json$##' | sort -V | tail -n 1)
  [[ -n "$latest" ]] || return 1
  echo "$latest"
}
ADAPTER=$(resolve_adapter "$ADAPTER_ROOT") || { echo "[FAIL] B1 Direct adapter unavailable: $ADAPTER_ROOT" >&2; exit 2; }
BASE_CFG="$RUN/configs/b1_direct_standalone_export.yaml"
RESOLVED="$RUN/configs/b1_direct_standalone_export_resolved.yaml"
"$SFT_PYTHON" - "$BASE_CFG" "$RESOLVED" "$ADAPTER" "$MERGED" <<'PY'
import sys
from pathlib import Path
src=Path(sys.argv[1]).read_text(encoding='utf-8')
lines=[]
for line in src.splitlines():
    if line.startswith('adapter_name_or_path:'): line=f'adapter_name_or_path: {sys.argv[3]}'
    if line.startswith('export_dir:'): line=f'export_dir: {sys.argv[4]}'
    lines.append(line)
Path(sys.argv[2]).write_text('\n'.join(lines)+'\n',encoding='utf-8')
PY
cd "$LLAMA_FACTORY_DIR"
llamafactory_cli export "$RESOLVED"
[[ -f "$MERGED/config.json" ]] || { echo "[FAIL] merged config missing: $MERGED/config.json" >&2; exit 3; }
echo "[PASS] B1 Direct merged: $MERGED"
