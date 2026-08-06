#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
[[ -n "$BOOTSTRAP_PY" ]] || { echo "[FAIL] 找不到 Python" >&2; exit 2; }
DATA_PY=$("$BOOTSTRAP_PY" "$ROOT/main_layer/resolve_runtime.py" --settings "$SETTINGS" --workload data --field python)
SFT_PY=$("$BOOTSTRAP_PY" "$ROOT/main_layer/resolve_runtime.py" --settings "$SETTINGS" --workload sft --field python)
"$DATA_PY" "$ROOT/data_layer/07_filter_trajectories.py" --settings "$SETTINGS" --split train
"$DATA_PY" "$ROOT/data_layer/08_build_dota128_llamafactory.py" --settings "$SETTINGS"
"$SFT_PY" "$ROOT/train_layer/01_validate_training_data.py" --settings "$SETTINGS"
"$SFT_PY" "$ROOT/train_layer/02_generate_configs.py" --settings "$SETTINGS"
echo "训练数据和 B1/B2 配置已经准备好。请分别运行 train_layer/03_train_b1_direct.sh 与 04_train_b2_socratic.sh。"
