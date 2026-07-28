#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
REWARD=${2:-hbb}
BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
[[ -n "$BOOTSTRAP_PY" ]] || { echo "[FAIL] 找不到 Python" >&2; exit 2; }
DATA_PY=$("$BOOTSTRAP_PY" "$ROOT/main_layer/resolve_runtime.py" --settings "$SETTINGS" --workload data --field python)
RL_PY=$("$BOOTSTRAP_PY" "$ROOT/main_layer/resolve_runtime.py" --settings "$SETTINGS" --workload rl --field python)
"$DATA_PY" "$ROOT/train_layer/rl/01_build_grounding_rl_data.py" --settings "$SETTINGS"
"$RL_PY" "$ROOT/train_layer/rl/00_preflight_rl.py" --settings "$SETTINGS"
OUTPUT=$("$DATA_PY" "$ROOT/train_layer/rl/02_generate_easy_r1_launch.py" --settings "$SETTINGS" --reward "$REWARD")
echo "$OUTPUT"
SCRIPT=$(echo "$OUTPUT" | "$DATA_PY" -c 'import json,sys; print(json.load(sys.stdin)["launch_script"])')
exec bash "$SCRIPT"
