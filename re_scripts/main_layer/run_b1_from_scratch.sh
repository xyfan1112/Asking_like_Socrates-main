#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
MODE=${2:-fresh}
BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
[[ -n "$BOOTSTRAP_PY" ]] || { echo "[FAIL] 找不到 Python" >&2; exit 2; }
SFT_PY=$("$BOOTSTRAP_PY" "$ROOT/main_layer/resolve_runtime.py" --settings "$SETTINGS" --workload sft --field python)

echo "[B1 1/8] 重建 scene-disjoint DOTA→Ref→Agent/Direct 数据"
"$BOOTSTRAP_PY" "$ROOT/main_layer/run.py" data-full --settings "$SETTINGS"

echo "[B1 2/8] 启动并实测 Reasoner/Perceiver/Verifier"
bash "$ROOT/data_layer/agents/start_three_agents.sh" "$SETTINGS"

echo "[B1 3/8] 20条分层 debug；硬错误会阻止继续"
bash "$ROOT/main_layer/run_official_socratic_full.sh" "$SETTINGS" train debug 20 fresh

echo "[B1 4/8] 全量 fresh Socratic 生成并同时构建匹配的 B1/B2 数据"
bash "$ROOT/main_layer/run_official_socratic_full.sh" "$SETTINGS" train full 20 fresh

echo "[B1 5/8] 校验 B1/B2 行数、问题、图像和最终 GT 完全匹配"
"$SFT_PY" "$ROOT/train_layer/01_validate_training_data.py" --settings "$SETTINGS" --target all

echo "[B1 6/8] 生成并验证 LLaMA-Factory 配置"
"$SFT_PY" "$ROOT/train_layer/02_generate_configs.py" --settings "$SETTINGS"
"$SFT_PY" "$ROOT/train_layer/02a_validate_llamafactory_contract.py" --settings "$SETTINGS" --target b1

echo "[B1 7/8] 停止智能体并按需归档旧 B1 adapter"
bash "$ROOT/data_layer/agents/stop_three_agents.sh" "$SETTINGS"
if [[ "$MODE" == "fresh" ]]; then
  "$SFT_PY" - "$SETTINGS" <<'PY'
import json, shutil, sys
from datetime import datetime
from pathlib import Path
s=json.load(open(sys.argv[1],encoding='utf-8'))
p=Path(s['training']['b1_adapter_output'])
if p.exists():
    dst=p.with_name(p.name+'_archive_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    shutil.move(str(p),str(dst))
    print(f'[ARCHIVE] {p} -> {dst}')
PY
fi

echo "[B1 8/8] 训练匹配的 final-only B1"
bash "$ROOT/train_layer/03_train_b1_direct.sh" "$SETTINGS"
