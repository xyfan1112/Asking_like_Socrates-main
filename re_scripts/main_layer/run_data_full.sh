#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
[[ -n "$BOOTSTRAP_PY" ]] || { echo "[FAIL] 找不到 Python" >&2; exit 2; }
DATA_PY=$("$BOOTSTRAP_PY" "$ROOT/main_layer/resolve_runtime.py" --settings "$SETTINGS" --workload data --field python)

step() {
  local number=$1 title=$2
  shift 2
  echo
  echo "================================================================================"
  echo "[DATA ${number}/9] ${title}"
  echo "================================================================================"
  "$@"
}

require_file() {
  local path=$1 hint=$2
  if [[ ! -s "$path" ]]; then
    echo "[FAIL] 预期文件没有生成: $path" >&2
    echo "[HINT] $hint" >&2
    exit 3
  fi
  echo "[OK] $path"
}

json_passed() {
  local path=$1
  "$DATA_PY" - "$path" <<'PY'
import json,sys
p=sys.argv[1]
x=json.load(open(p,encoding='utf-8'))
raise SystemExit(0 if x.get('passed',False) else 2)
PY
}

PIPELINE_ROOT=$("$DATA_PY" - "$SETTINGS" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8')); print(s['paths']['pipeline_work_root'])
PY
)
REF_ROOT=$("$DATA_PY" - "$SETTINGS" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8')); print(s['paths']['dota128_ref_root'])
PY
)

step 1 "环境与路径预检" \
  "$DATA_PY" "$ROOT/main_layer/00_preflight.py" --settings "$SETTINGS"

step 2 "审计原始 DOTA OBB" \
  "$DATA_PY" "$ROOT/data_layer/01_audit_dota128.py" --settings "$SETTINGS"

step 3 "构建有效 Direct OBB 与唯一 DOTA-Ref" \
  "$DATA_PY" "$ROOT/data_layer/02_build_dota128_ref.py" --settings "$SETTINGS"
require_file "$REF_ROOT/train.jsonl" "查看 $REF_ROOT/build_report.json"
require_file "$REF_ROOT/train_all.jsonl" "查看 $REF_ROOT/build_report.json"

step 4 "严格校验像素边界、归一化范围、唯一性和类别遮蔽" \
  "$DATA_PY" "$ROOT/data_layer/03_validate_dota128_ref.py" --settings "$SETTINGS"
json_passed "$REF_ROOT/validation_report.json" || {
  echo "[FAIL] Ref 未通过校验，禁止继续生成训练数据。" >&2
  exit 4
}

step 5 "生成平衡抽样的 Ref 可视化" \
  "$DATA_PY" "$ROOT/data_layer/04_visualize_dota128_ref.py" --settings "$SETTINGS"
require_file "$PIPELINE_ROOT/ref_previews/preview_report.json" "查看 ref_previews 目录"
json_passed "$PIPELINE_ROOT/ref_previews/preview_report.json" || exit 5

step 6 "生成多智能体输入和 Direct SFT 数据" \
  "$DATA_PY" "$ROOT/data_layer/05_build_agent_inputs.py" --settings "$SETTINGS"
require_file "$PIPELINE_ROOT/agent_inputs/train_agent_inputs.jsonl" "查看 agent_inputs/build_report.json"
require_file "$PIPELINE_ROOT/agent_inputs/train_direct.json" "查看 agent_inputs/build_report.json"

step 7 "审计 DOTA→Ref→Agent/Direct 的类别与 OBB 血缘" \
  "$DATA_PY" "$ROOT/data_layer/05c_audit_lineage.py" --settings "$SETTINGS"

step 8 "准备 Grounding RL 数据（不启动训练）" \
  "$DATA_PY" "$ROOT/train_layer/rl/01_build_grounding_rl_data.py" --settings "$SETTINGS"

step 9 "数据层最终门控" \
  "$DATA_PY" "$ROOT/data_layer/05b_verify_data_outputs.py" --settings "$SETTINGS"

echo
echo "[DONE] 数据层全部通过。"
echo "下一步：python main_layer/run.py build-official-parquet --settings $SETTINGS --split train"
echo "最终报告：$PIPELINE_ROOT/reports/data_layer_final_report.json"
