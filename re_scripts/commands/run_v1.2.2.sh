#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/commands/user_config.sh"
RS="$REPO_ROOT/re_scripts"
ZH_CFG="$ZH_OUTPUT_ROOT/config/settings.zh.v1.2.2.json"
OMNI_CFG="$ZH_OUTPUT_ROOT/config/settings.zh.v1.2.2.omni-shared.json"

fail() { echo "[FAIL] $*" >&2; exit 2; }

resolve_base() {
  [[ -d "$RS" ]] || fail "re_scripts不存在: $RS"
  if [[ ! -f "$BASE_SETTINGS" ]]; then
    if [[ -f "$RS/settings.json" ]]; then
      echo "[WARN] BASE_SETTINGS不存在，退回 $RS/settings.json"
      BASE_SETTINGS="$RS/settings.json"
    else
      fail "settings不存在: $BASE_SETTINGS"
    fi
  fi
}
resolve_base

PYTHON=$(python - "$BASE_SETTINGS" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
print(s.get('runtime',{}).get('workloads',{}).get('data',{}).get('python') or sys.executable)
PY
)
[[ -x "$PYTHON" ]] || PYTHON=$(command -v python3 || command -v python)

require_classes() {
  [[ -f "$CLASSES_FILE" ]] || fail "classes.txt不存在: $CLASSES_FILE"
}

prepare() {
  require_classes
  cd "$RS"
  "$PYTHON" tools/materialize_language_settings.py \
    --base-settings "$BASE_SETTINGS" \
    --classes-file "$CLASSES_FILE" \
    --lang zh \
    --direct-qa-mode "$DIRECT_QA_MODE" \
    --output-root "$ZH_OUTPUT_ROOT"
  "$PYTHON" main_layer/run.py check-taxonomy \
    --settings "$ZH_CFG" --classes-file "$CLASSES_FILE" --lang zh
  echo "[PASS] 中文独立配置: $ZH_CFG"
  echo "[INFO] Direct模式: $DIRECT_QA_MODE"
  echo "[INFO] 英文继续使用原配置和原输出目录: $BASE_SETTINGS"
}

require_cfg() {
  require_classes
  [[ -f "$ZH_CFG" ]] || prepare
  local configured
  configured=$("$PYTHON" - "$ZH_CFG" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
print(s.get('data_conversion',{}).get('direct_qa_mode','legacy_class_roi_obb'))
PY
)
  [[ "$configured" == "$DIRECT_QA_MODE" ]] || fail \
    "现有中文settings的Direct模式=$configured，但user_config.sh=$DIRECT_QA_MODE。更换ZH_OUTPUT_ROOT或删除旧config后重新prepare。"
}

require_legacy_or_explicit_json_eval() {
  if [[ "$DIRECT_QA_MODE" != "legacy_class_roi_obb" && "${ALLOW_EXPERIMENTAL_JSON_TEST_MATRIX:-0}" != "1" ]]; then
    fail "当前Direct模式=$DIRECT_QA_MODE，不是原test-matrix协议。若接受跨协议迁移评测，请设置ALLOW_EXPERIMENTAL_JSON_TEST_MATRIX=1。"
  fi
  [[ "$DIRECT_QA_MODE" == "legacy_class_roi_obb" ]] || \
    echo "[WARN] JSON Direct模型正在执行原test-matrix；结果不可冒充legacy同协议消融。"
}

datafull() {
  require_cfg
  cd "$RS"
  "$PYTHON" main_layer/run.py data-full \
    --settings "$ZH_CFG" --classes-file "$CLASSES_FILE" --lang zh
}

b0() {
  require_cfg
  cd "$RS"
  "$PYTHON" main_layer/run.py stop-agents --settings "$ZH_CFG" || true
  "$PYTHON" main_layer/run.py stop-eval --settings "$ZH_CFG" rs_eot 8010 || true
  "$PYTHON" main_layer/run.py start-eval --settings "$ZH_CFG" rs_eot 1 8010
  trap '"$PYTHON" "$RS/main_layer/run.py" stop-eval --settings "$ZH_CFG" rs_eot 8010 || true' EXIT
  "$PYTHON" main_layer/run.py test-matrix \
    --settings "$ZH_CFG" rs_eot http://127.0.0.1:8010/v1 fresh
  "$PYTHON" main_layer/run.py stop-eval --settings "$ZH_CFG" rs_eot 8010
  trap - EXIT
  echo "[PASS] B0 RS-EoT评测完成。"
}

archive_direct_outputs() {
  require_cfg
  "$PYTHON" - "$ZH_CFG" <<'PY'
import json,shutil,sys
from datetime import datetime
from pathlib import Path
s=json.load(open(sys.argv[1],encoding='utf-8'))
t=s['training']; stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
for key in ('b1_direct_standalone_adapter_output','b1_direct_standalone_merged_output'):
    p=Path(t[key])
    if p.exists():
        dst=p.with_name(p.name+'_archive_'+stamp)
        shutil.move(str(p),str(dst))
        print(f'[ARCHIVE] {p} -> {dst}')
PY
}

force_direct_prepare() {
  require_cfg
  cd "$RS"
  "$PYTHON" main_layer/run.py prepare-direct-sft --settings "$ZH_CFG"
  "$PYTHON" main_layer/run.py validate-b1-direct-standalone --settings "$ZH_CFG"
  "$PYTHON" main_layer/run.py generate-b1-direct-config --settings "$ZH_CFG"
  "$PYTHON" main_layer/run.py validate-b1-direct-contract --settings "$ZH_CFG"
  echo "[PASS] B1-Direct-Standalone全部训练前门控通过。"
}

force_direct_train() {
  require_cfg
  cd "$RS"
  force_direct_prepare
  "$PYTHON" main_layer/run.py stop-agents --settings "$ZH_CFG" || true
  "$PYTHON" main_layer/run.py stop-eval --settings "$ZH_CFG" rs_eot 8010 || true
  "$PYTHON" main_layer/run.py stop-eval --settings "$ZH_CFG" rs-eot-b1-direct 8011 || true
  archive_direct_outputs
  "$PYTHON" main_layer/run.py train-b1-direct-standalone --settings "$ZH_CFG"
  "$PYTHON" main_layer/run.py merge-b1-direct-standalone --settings "$ZH_CFG"
  "$PYTHON" main_layer/run.py register-b1-direct-model --settings "$ZH_CFG"
  echo "[PASS] B1-Direct-Standalone训练与合并完成。"
}

eval_b1_direct() {
  require_cfg
  require_legacy_or_explicit_json_eval
  cd "$RS"
  "$PYTHON" main_layer/run.py register-b1-direct-model --settings "$ZH_CFG"
  "$PYTHON" main_layer/run.py stop-agents --settings "$ZH_CFG" || true
  "$PYTHON" main_layer/run.py stop-eval --settings "$ZH_CFG" rs-eot-b1-direct 8011 || true
  "$PYTHON" main_layer/run.py start-eval --settings "$ZH_CFG" rs-eot-b1-direct 1 8011
  trap '"$PYTHON" "$RS/main_layer/run.py" stop-eval --settings "$ZH_CFG" rs-eot-b1-direct 8011 || true' EXIT
  "$PYTHON" main_layer/run.py test-matrix \
    --settings "$ZH_CFG" rs-eot-b1-direct http://127.0.0.1:8011/v1 fresh
  "$PYTHON" main_layer/run.py stop-eval --settings "$ZH_CFG" rs-eot-b1-direct 8011
  trap - EXIT
  echo "[PASS] rs-eot-b1-direct评测完成。"
}

safe_direct() {
  require_legacy_or_explicit_json_eval
  prepare
  datafull
  b0
  force_direct_train
  eval_b1_direct
  echo "[PASS] 数据层→B0→B1 Direct训练/合并→B1 Direct测试完成；未运行Socratic。"
}

socratic_debug() {
  require_cfg
  cd "$RS"
  "$PYTHON" main_layer/run.py stop-agents --settings "$ZH_CFG" || true
  "$PYTHON" main_layer/run.py start-agents --settings "$ZH_CFG"
  trap '"$PYTHON" "$RS/main_layer/run.py" stop-agents --settings "$ZH_CFG" || true' EXIT
  "$PYTHON" main_layer/run.py official-socratic-full \
    --settings "$ZH_CFG" --classes-file "$CLASSES_FILE" --lang zh \
    train debug 40 fresh
  "$PYTHON" main_layer/run.py inspect-api-truncation \
    --settings "$ZH_CFG" --classes-file "$CLASSES_FILE" --lang zh --mode debug
  "$PYTHON" main_layer/run.py check-debug-infra \
    --settings "$ZH_CFG" --classes-file "$CLASSES_FILE" --lang zh
  "$PYTHON" main_layer/run.py check-debug \
    --settings "$ZH_CFG" --classes-file "$CLASSES_FILE" --lang zh
  "$PYTHON" main_layer/run.py stop-agents --settings "$ZH_CFG" || true
  trap - EXIT
}

full() {
  require_cfg
  cd "$RS"
  "$PYTHON" main_layer/run.py check-debug-infra \
    --settings "$ZH_CFG" --classes-file "$CLASSES_FILE" --lang zh
  "$PYTHON" main_layer/run.py check-debug \
    --settings "$ZH_CFG" --classes-file "$CLASSES_FILE" --lang zh
  "$PYTHON" main_layer/run.py official-socratic-full \
    --settings "$ZH_CFG" --classes-file "$CLASSES_FILE" --lang zh \
    train full 40 fresh
}

omni_download() {
  local modelscope_bin
  modelscope_bin="$(dirname "${OMNI_PYTHON:-/nonexistent/python}")/modelscope"
  if [[ ! -x "$modelscope_bin" ]]; then
    modelscope_bin=$(command -v modelscope || true)
  fi
  [[ -n "$modelscope_bin" && -x "$modelscope_bin" ]] || fail \
    "找不到modelscope命令。请在Omni环境安装modelscope后重试；脚本不会自动改环境。"
  mkdir -p "$(dirname "$OMNI_MODEL_PATH")"
  if [[ -f "$OMNI_MODEL_PATH/config.json" ]]; then
    echo "[INFO] 模型已存在: $OMNI_MODEL_PATH"
    return 0
  fi
  "$modelscope_bin" download \
    --model "$OMNI_MODELSCOPE_ID" \
    --local_dir "$OMNI_MODEL_PATH"
  [[ -f "$OMNI_MODEL_PATH/config.json" ]] || fail "下载结束但缺少config.json: $OMNI_MODEL_PATH"
  echo "[PASS] ModelScope模型下载完成: $OMNI_MODEL_PATH"
}

omni_preflight() {
  require_cfg
  [[ -x "$OMNI_PYTHON" ]] || fail "OMNI_PYTHON不可执行: $OMNI_PYTHON"
  [[ -d "$OMNI_MODEL_PATH" ]] || fail "OMNI_MODEL_PATH不存在: $OMNI_MODEL_PATH"
  local no_tts_flag="--require-no-tts"
  [[ "${OMNI_REQUIRE_NO_TTS:-1}" == "1" ]] || no_tts_flag="--no-require-no-tts"
  "$OMNI_PYTHON" "$RS/tools/check_qwen3_omni_4gpu.py" \
    --settings "$ZH_CFG" \
    --model-path "$OMNI_MODEL_PATH" \
    --model-id "$OMNI_MODELSCOPE_ID" \
    --python "$OMNI_PYTHON" \
    --tensor-parallel-size "$OMNI_TP_SIZE" \
    "$no_tts_flag"
}

omni_start() {
  omni_preflight
  bash "$HERE/commands/manage_omni_tp4.sh" start
}

omni_stop() { bash "$HERE/commands/manage_omni_tp4.sh" stop; }
omni_status() { bash "$HERE/commands/manage_omni_tp4.sh" status; }
omni_log() { bash "$HERE/commands/manage_omni_tp4.sh" log "${2:-200}"; }

omni_shared_settings() {
  require_cfg
  [[ -n "$OMNI_SERVED_NAME" ]] || fail "OMNI_SERVED_NAME为空"
  cd "$RS"
  "$PYTHON" main_layer/run.py materialize-shared-omni-agents \
    --settings "$ZH_CFG" \
    --output "$OMNI_CFG" \
    --base-url "$OMNI_BASE_URL" \
    --served-name "$OMNI_SERVED_NAME" \
    --model-path "$OMNI_MODEL_PATH" \
    --full-concurrency "$OMNI_FULL_CONCURRENCY" \
    --tensor-parallel-size "$OMNI_TP_SIZE" \
    --checkpoint-id "$OMNI_MODELSCOPE_ID"
  echo "[PASS] 共享Omni三逻辑角色配置: $OMNI_CFG"
  echo "[IMPORTANT] 不要对该配置运行start-agents。"
}

omni_smoke() {
  omni_shared_settings
  cd "$RS"
  "$PYTHON" main_layer/run.py check-shared-omni-endpoint --settings "$OMNI_CFG"
}

omni_benchmark() {
  omni_smoke
  mkdir -p "$OMNI_RUNTIME_DIR"
  local stamp report
  stamp=$(date +%Y%m%d_%H%M%S)
  report="$OMNI_RUNTIME_DIR/benchmark_${stamp}.json"
  "$PYTHON" "$RS/tools/benchmark_shared_omni_endpoint.py" \
    --settings "$OMNI_CFG" \
    --base-url "$OMNI_BASE_URL" \
    --served-name "$OMNI_SERVED_NAME" \
    --concurrency 1,2,4 \
    --requests-per-level 4 \
    --output "$report"
}

omni_shared_debug() {
  omni_smoke
  cd "$RS"
  "$PYTHON" main_layer/run.py official-socratic-full \
    --settings "$OMNI_CFG" --classes-file "$CLASSES_FILE" --lang zh \
    train debug 40 fresh
  "$PYTHON" main_layer/run.py inspect-api-truncation \
    --settings "$OMNI_CFG" --classes-file "$CLASSES_FILE" --lang zh --mode debug
  "$PYTHON" main_layer/run.py check-debug-infra \
    --settings "$OMNI_CFG" --classes-file "$CLASSES_FILE" --lang zh
  "$PYTHON" main_layer/run.py check-debug \
    --settings "$OMNI_CFG" --classes-file "$CLASSES_FILE" --lang zh
}

omni_shared_full() {
  omni_shared_settings
  cd "$RS"
  "$PYTHON" main_layer/run.py check-debug-infra \
    --settings "$OMNI_CFG" --classes-file "$CLASSES_FILE" --lang zh
  "$PYTHON" main_layer/run.py check-debug \
    --settings "$OMNI_CFG" --classes-file "$CLASSES_FILE" --lang zh
  "$PYTHON" main_layer/run.py official-socratic-full \
    --settings "$OMNI_CFG" --classes-file "$CLASSES_FILE" --lang zh \
    train full 40 fresh
}

omni_safe_debug() {
  prepare
  datafull
  omni_preflight
  omni_start
  trap 'bash "$HERE/commands/manage_omni_tp4.sh" stop || true' EXIT
  omni_smoke
  omni_benchmark
  omni_shared_debug
  omni_stop
  trap - EXIT
  echo "[PASS] 中文数据层、Omni TP4服务、smoke、benchmark和fresh Debug40全部完成。"
}

en_info() {
  echo "英文不变：继续使用原settings和原输出目录："
  echo "  $BASE_SETTINGS"
  echo "中文、Direct实验和Omni轨迹均写入："
  echo "  $ZH_OUTPUT_ROOT"
}

usage() {
  cat <<EOF
用法: bash commands/run_v1.2.2.sh <命令>

数据与Direct：
  prepare               创建中文独立settings并检查类别/标签
  datafull              生成Canonical、Direct、Ref和Agent输入
  b0                    评测未训练RS-EoT
  force-direct-prepare  Direct训练前全部门控
  force-direct-train    独立Direct训练并合并，不依赖Socratic
  eval-b1-direct        评测rs-eot-b1-direct
  safe-direct           prepare→datafull→B0→Direct训练→B1测试

原三服务Socratic：
  socratic-debug        原三服务中文fresh Debug40
  full                  原三服务仅在Debug PASS后运行Full

Qwen3-Omni AWQ-No-TTS共享TP4：
  omni-download         从ModelScope下载指定checkpoint
  omni-preflight        检查4卡、vLLM、AWQ、No-TTS和模型索引
  omni-start            启动一个模型、一个服务、TP=4
  omni-status           查看服务和四卡显存/利用率
  omni-log [N]          查看最后N行服务日志
  omni-stop             停止共享服务
  omni-shared-settings  三逻辑角色指向同一个TP4端点
  omni-smoke            Reasoner文本+Perceiver图像+Verifier文本smoke
  omni-benchmark        并发1/2/4部署性能测试
  omni-shared-debug     使用共享端点运行fresh Debug40
  omni-shared-full      Debug门控PASS后运行Full
  omni-safe-debug       无脑：prepare→datafull→preflight→start→smoke→benchmark→Debug→stop

  en-info               显示英文保持不变说明
EOF
}

case "${1:-}" in
  prepare) prepare ;;
  datafull) datafull ;;
  b0) b0 ;;
  force-direct-prepare) force_direct_prepare ;;
  force-direct-train) force_direct_train ;;
  eval-b1-direct) eval_b1_direct ;;
  safe-direct) safe_direct ;;
  socratic-debug) socratic_debug ;;
  full) full ;;
  omni-download) omni_download ;;
  omni-preflight) omni_preflight ;;
  omni-start) omni_start ;;
  omni-status) omni_status ;;
  omni-log) omni_log "$@" ;;
  omni-stop) omni_stop ;;
  omni-shared-settings) omni_shared_settings ;;
  omni-smoke) omni_smoke ;;
  omni-benchmark) omni_benchmark ;;
  omni-shared-debug) omni_shared_debug ;;
  omni-shared-full) omni_shared_full ;;
  omni-safe-debug) omni_safe_debug ;;
  en-info) en_info ;;
  *) usage; exit 2 ;;
esac
