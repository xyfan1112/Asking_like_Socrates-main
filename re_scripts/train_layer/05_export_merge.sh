#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
WHICH=${2:-both}   # b1 | b2 | both
source "$ROOT/train_layer/_runtime.sh"
resolve_sft_runtime "$ROOT" "$SETTINGS"
"$SFT_PYTHON" "$ROOT/train_layer/02_generate_configs.py" --settings "$SETTINGS"

readarray -t META < <("$SFT_PYTHON" - "$SETTINGS" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8')); t=s['training']
for value in [
    s['paths']['training_run_root'],
    t['b1_adapter_output'], t['b2_adapter_output'],
    t['b1_merged_output'], t['b2_merged_output'],
]:
    print(value)
PY
)
LF=$LLAMA_FACTORY_DIR
RUN=${META[0]}
B1_ADAPTER=${META[1]}
B2_ADAPTER=${META[2]}
B1_MERGED=${META[3]}
B2_MERGED=${META[4]}

resolve_adapter() {
  local root=$1
  if [[ -f "$root/adapter_config.json" ]] && { [[ -f "$root/adapter_model.safetensors" ]] || [[ -f "$root/adapter_model.bin" ]]; }; then
    echo "$root"; return 0
  fi
  local latest
  latest=$(find "$root" -maxdepth 2 -type f -name adapter_config.json 2>/dev/null \
    | sed 's#/adapter_config.json$##' \
    | sort -V | tail -n 1)
  if [[ -n "$latest" ]] && { [[ -f "$latest/adapter_model.safetensors" ]] || [[ -f "$latest/adapter_model.bin" ]]; }; then
    echo "[WARN] root adapter incomplete; using latest checkpoint: $latest" >&2
    echo "$latest"; return 0
  fi
  return 1
}

export_one() {
  local name=$1 adapter_root=$2 merged=$3 base_cfg=$4
  local adapter
  if ! adapter=$(resolve_adapter "$adapter_root"); then
    echo "[SKIP] $name adapter unavailable: $adapter_root" >&2
    return 10
  fi
  local cfg="$RUN/configs/${name}_export_resolved.yaml"
  "$SFT_PYTHON" - "$base_cfg" "$cfg" "$adapter" "$merged" <<'PY'
import sys
from pathlib import Path
src=Path(sys.argv[1]).read_text(encoding='utf-8')
lines=[]
for line in src.splitlines():
    if line.startswith('adapter_name_or_path:'):
        line=f'adapter_name_or_path: {sys.argv[3]}'
    if line.startswith('export_dir:'):
        line=f'export_dir: {sys.argv[4]}'
    lines.append(line)
Path(sys.argv[2]).write_text('\n'.join(lines)+'\n',encoding='utf-8')
PY
  echo "[INFO] merging $name: adapter=$adapter -> $merged"
  cd "$LF"
  llamafactory_cli export "$cfg"
  [[ -f "$merged/config.json" ]] || { echo "[FAIL] merged config missing: $merged/config.json" >&2; return 3; }
  echo "[OK] $name merged: $merged"
}

requested=0; succeeded=0; skipped=0
if [[ "$WHICH" == "b1" || "$WHICH" == "both" ]]; then
  requested=$((requested+1))
  if export_one b1 "$B1_ADAPTER" "$B1_MERGED" "$RUN/configs/b1_export.yaml"; then
    succeeded=$((succeeded+1))
  else
    rc=$?; [[ $rc -eq 10 ]] && skipped=$((skipped+1)) || exit $rc
  fi
fi
if [[ "$WHICH" == "b2" || "$WHICH" == "both" ]]; then
  requested=$((requested+1))
  if export_one b2 "$B2_ADAPTER" "$B2_MERGED" "$RUN/configs/b2_export.yaml"; then
    succeeded=$((succeeded+1))
  else
    rc=$?; [[ $rc -eq 10 ]] && skipped=$((skipped+1)) || exit $rc
  fi
fi
if [[ $requested -eq 0 ]]; then echo "[FAIL] WHICH must be b1, b2 or both" >&2; exit 2; fi
if [[ $succeeded -eq 0 ]]; then echo "[FAIL] no adapter was merged" >&2; exit 3; fi
echo "[SUMMARY] requested=$requested succeeded=$succeeded skipped=$skipped"
