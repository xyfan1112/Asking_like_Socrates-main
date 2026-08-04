#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/commands/user_config.sh"
RS="$REPO_ROOT/re_scripts"
[[ -d "$RS" ]] || { echo "[FAIL] re_scripts不存在: $RS" >&2; exit 2; }
if [[ ! -f "$BASE_SETTINGS" ]]; then
  if [[ -f "$RS/settings.json" ]]; then
    echo "[WARN] BASE_SETTINGS不存在，退回 $RS/settings.json"
    BASE_SETTINGS="$RS/settings.json"
  else
    echo "[FAIL] settings不存在: $BASE_SETTINGS" >&2; exit 2
  fi
fi
[[ -f "$CLASSES_FILE" ]] || { echo "[FAIL] classes.txt不存在: $CLASSES_FILE" >&2; exit 2; }
ZH_CFG="$ZH_OUTPUT_ROOT/config/settings.zh.v1.2.0.json"
PYTHON=$(python - "$BASE_SETTINGS" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
print(s.get('runtime',{}).get('workloads',{}).get('data',{}).get('python') or sys.executable)
PY
)
[[ -x "$PYTHON" ]] || PYTHON=$(command -v python3 || command -v python)

prepare() {
  cd "$RS"
  "$PYTHON" tools/materialize_language_settings.py \
    --base-settings "$BASE_SETTINGS" \
    --classes-file "$CLASSES_FILE" \
    --lang zh \
    --output-root "$ZH_OUTPUT_ROOT"
  "$PYTHON" main_layer/run.py check-taxonomy \
    --settings "$ZH_CFG" --classes-file "$CLASSES_FILE" --lang zh
  echo "[PASS] 中文独立配置: $ZH_CFG"
  echo "[INFO] 英文继续使用原配置，不创建新英文目录: $BASE_SETTINGS"
}

require_zh_cfg() {
  [[ -f "$ZH_CFG" ]] || prepare
}

datafull() {
  require_zh_cfg
  cd "$RS"
  "$PYTHON" main_layer/run.py data-full \
    --settings "$ZH_CFG" --classes-file "$CLASSES_FILE" --lang zh
}

agents() {
  require_zh_cfg
  cd "$RS"
  # 模型服务器本身没有语言。中文/英文由 official generation 的请求 Prompt 决定。
  "$PYTHON" main_layer/run.py stop-agents --settings "$ZH_CFG" || true
  "$PYTHON" main_layer/run.py start-agents --settings "$ZH_CFG"
}

debug40() {
  require_zh_cfg
  cd "$RS"
  "$PYTHON" main_layer/run.py official-socratic-full \
    --settings "$ZH_CFG" --classes-file "$CLASSES_FILE" --lang zh \
    train debug 40 fresh
}

inspect_tokens() {
  require_zh_cfg
  cd "$RS"
  "$PYTHON" main_layer/run.py inspect-api-truncation \
    --settings "$ZH_CFG" --classes-file "$CLASSES_FILE" --lang zh --mode debug
}

check_debug() {
  require_zh_cfg
  cd "$RS"
  "$PYTHON" main_layer/run.py check-debug-infra \
    --settings "$ZH_CFG" --classes-file "$CLASSES_FILE" --lang zh
  "$PYTHON" main_layer/run.py check-debug \
    --settings "$ZH_CFG" --classes-file "$CLASSES_FILE" --lang zh
}

safe_until_debug() {
  prepare
  datafull
  agents
  trap '"$PYTHON" "$RS/main_layer/run.py" stop-agents --settings "$ZH_CFG" || true' EXIT
  debug40
  inspect_tokens
  check_debug
  "$PYTHON" main_layer/run.py stop-agents --settings "$ZH_CFG" || true
  trap - EXIT
  echo "[PASS] 已完成到Debug40并停止三智能体。脚本不会自动运行Full。"
}

b0() {
  require_zh_cfg
  cd "$RS"
  "$PYTHON" main_layer/run.py stop-agents --settings "$ZH_CFG" || true
  "$PYTHON" main_layer/run.py stop-eval --settings "$ZH_CFG" rs_eot 8010 || true
  "$PYTHON" main_layer/run.py start-eval --settings "$ZH_CFG" rs_eot 1 8010
  trap '"$PYTHON" "$RS/main_layer/run.py" stop-eval --settings "$ZH_CFG" rs_eot 8010 || true' EXIT
  "$PYTHON" main_layer/run.py test-matrix \
    --settings "$ZH_CFG" rs_eot http://127.0.0.1:8010/v1 fresh
  "$PYTHON" main_layer/run.py stop-eval --settings "$ZH_CFG" rs_eot 8010
  trap - EXIT
  echo "[PASS] B0 RS-EoT评测完成。结果位于中文独立testing目录。"
}

train_baseline() {
  require_zh_cfg
  cd "$RS"
  "$PYTHON" main_layer/run.py stop-agents --settings "$ZH_CFG" || true
  echo "[INFO] 先验证matched B1。若不存在或不合格，则自动训练独立D1 Direct-only。"
  if "$PYTHON" main_layer/run.py validate-train --settings "$ZH_CFG" --target b1; then
    echo "[PATH] matched B1数据有效，训练B1。"
    "$PYTHON" main_layer/run.py train-b1 --settings "$ZH_CFG"
    "$PYTHON" main_layer/run.py merge --settings "$ZH_CFG" b1
  else
    echo "[PATH] matched B1不可用，构建并训练D1 Direct-only基线。"
    "$PYTHON" main_layer/run.py prepare-direct-sft --settings "$ZH_CFG"
    "$PYTHON" main_layer/run.py validate-d1 --settings "$ZH_CFG"
    "$PYTHON" main_layer/run.py generate-d1-config --settings "$ZH_CFG"
    "$PYTHON" main_layer/run.py train-d1 --settings "$ZH_CFG"
    "$PYTHON" main_layer/run.py merge-d1 --settings "$ZH_CFG"
    "$PYTHON" main_layer/run.py register-d1-model --settings "$ZH_CFG"
  fi
}

eval_d1() {
  require_zh_cfg
  cd "$RS"
  "$PYTHON" main_layer/run.py register-d1-model --settings "$ZH_CFG"
  "$PYTHON" main_layer/run.py stop-agents --settings "$ZH_CFG" || true
  "$PYTHON" main_layer/run.py stop-eval --settings "$ZH_CFG" rs_eot_d1_direct_only 8011 || true
  "$PYTHON" main_layer/run.py start-eval --settings "$ZH_CFG" rs_eot_d1_direct_only 1 8011
  trap '"$PYTHON" "$RS/main_layer/run.py" stop-eval --settings "$ZH_CFG" rs_eot_d1_direct_only 8011 || true' EXIT
  "$PYTHON" main_layer/run.py test-matrix \
    --settings "$ZH_CFG" rs_eot_d1_direct_only http://127.0.0.1:8011/v1 fresh
  "$PYTHON" main_layer/run.py stop-eval --settings "$ZH_CFG" rs_eot_d1_direct_only 8011
  trap - EXIT
}

full() {
  require_zh_cfg
  cd "$RS"
  # 两个检查有任何一个失败，set -e 会立即停止，不会降低门槛。
  check_debug
  "$PYTHON" main_layer/run.py official-socratic-full \
    --settings "$ZH_CFG" --classes-file "$CLASSES_FILE" --lang zh \
    train full 40 fresh
}

en_info() {
  echo "英文不变。继续使用原配置和原输出目录："
  echo "  $BASE_SETTINGS"
  echo "示例："
  echo "  cd $RS"
  echo "  $PYTHON main_layer/run.py data-full --settings $BASE_SETTINGS --classes-file $CLASSES_FILE --lang en"
  echo "  $PYTHON main_layer/run.py start-agents --settings $BASE_SETTINGS"
  echo "  $PYTHON main_layer/run.py official-socratic-full --settings $BASE_SETTINGS --classes-file $CLASSES_FILE --lang en train debug 40 fresh"
}

usage() {
  cat <<EOF
用法: bash commands/run_v1.2.0.sh <命令>

  prepare           创建独立中文settings并检查classes/labels
  datafull          仅运行中文data-full
  agents            启停三智能体；不传--lang
  debug40           仅运行中文fresh Debug40
  inspect-tokens    检查真实finish_reason；没有length就不增max_tokens
  check             检查ROI/crop基础设施和正式Debug门
  safe-until-debug  无脑运行：prepare→datafull→agents→Debug40→检查；不跑Full
  b0                评测未训练RS-EoT基座
  train-baseline    先验证matched B1；失败则训练D1 Direct-only
  eval-d1           评测D1 Direct-only合并模型
  full              仅在Debug全部PASS后运行fresh Full
  en-info           显示英文原配置运行方式（英文目录不变）
EOF
}

case "${1:-}" in
  prepare) prepare ;;
  datafull) datafull ;;
  agents) agents ;;
  debug40) debug40 ;;
  inspect-tokens) inspect_tokens ;;
  check) check_debug ;;
  safe-until-debug) safe_until_debug ;;
  b0) b0 ;;
  train-baseline) train_baseline ;;
  eval-d1) eval_d1 ;;
  full) full ;;
  en-info) en_info ;;
  *) usage; exit 2 ;;
esac
