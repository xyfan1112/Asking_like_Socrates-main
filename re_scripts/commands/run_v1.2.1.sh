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
ZH_CFG="$ZH_OUTPUT_ROOT/config/settings.zh.v1.2.1.json"
OMNI_CFG="$ZH_OUTPUT_ROOT/config/settings.zh.v1.2.1.omni-shared.json"
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
    --direct-qa-mode "$DIRECT_QA_MODE" \
    --output-root "$ZH_OUTPUT_ROOT"
  "$PYTHON" main_layer/run.py check-taxonomy \
    --settings "$ZH_CFG" --classes-file "$CLASSES_FILE" --lang zh
  echo "[PASS] 中文独立配置: $ZH_CFG"
  echo "[INFO] Direct模式: $DIRECT_QA_MODE"
  echo "[INFO] 英文继续使用原配置和原输出目录: $BASE_SETTINGS"
}

require_cfg() {
  [[ -f "$ZH_CFG" ]] || prepare
  local configured
  configured=$("$PYTHON" - "$ZH_CFG" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
print(s.get('data_conversion',{}).get('direct_qa_mode','legacy_class_roi_obb'))
PY
)
  [[ "$configured" == "$DIRECT_QA_MODE" ]] || {
    echo "[FAIL] 现有中文settings的Direct模式为 $configured，但user_config.sh为 $DIRECT_QA_MODE" >&2
    echo "       请更换ZH_OUTPUT_ROOT或删除旧config后重新prepare，禁止混写不同Direct模式。" >&2
    exit 2
  }
}


require_legacy_or_explicit_json_eval() {
  if [[ "$DIRECT_QA_MODE" != "legacy_class_roi_obb" && "${ALLOW_EXPERIMENTAL_JSON_TEST_MATRIX:-0}" != "1" ]]; then
    echo "[FAIL] 当前Direct模式=$DIRECT_QA_MODE，不是原test-matrix的legacy训练协议。" >&2
    echo "       模型可以训练，但继续原test-matrix属于跨协议迁移评测。" >&2
    echo "       确认接受后，在commands/user_config.sh设置 ALLOW_EXPERIMENTAL_JSON_TEST_MATRIX=1。" >&2
    exit 2
  fi
  if [[ "$DIRECT_QA_MODE" != "legacy_class_roi_obb" ]]; then
    echo "[WARN] 正在对JSON Direct模型执行原test-matrix；结果不可冒充legacy同协议消融。"
  fi
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
  echo "[PASS] 已完成：数据层→B0→B1 Direct训练/合并→B1 Direct测试。未运行Socratic Debug/Full。"
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

omni_preflight() {
  require_cfg
  cd "$RS"
  local omni_python="${OMNI_PYTHON:-}"
  if [[ ! -x "$omni_python" ]]; then
    echo "[WARN] OMNI_PYTHON不可执行，preflight退回当前Python：$PYTHON"
    omni_python="$PYTHON"
  fi
  local cmd=("$omni_python" "$RS/tools/check_qwen3_omni_4gpu.py" --settings "$ZH_CFG" --server-mode "$OMNI_SERVER_MODE")
  if [[ -n "${OMNI_MODEL_PATH:-}" ]]; then cmd+=(--model-path "$OMNI_MODEL_PATH"); fi
  "${cmd[@]}"
}

omni_launch_info() {
  [[ -n "${OMNI_MODEL_PATH:-}" ]] || {
    echo "[FAIL] 请先在commands/user_config.sh填写 OMNI_MODEL_PATH。" >&2
    exit 2
  }
  local vllm_bin
  vllm_bin="$(dirname "${OMNI_PYTHON:-$(command -v python)}")/vllm"
  if [[ "$OMNI_SERVER_MODE" == "vllm_thinker" ]]; then
    cat <<EOF
推荐轨迹服务：一个普通vLLM Thinker服务，4卡Tensor Parallel，不加载三份权重。
在独立Omni环境中运行：

CUDA_VISIBLE_DEVICES=0,1,2,3 "$vllm_bin" serve "$OMNI_MODEL_PATH" \\
  --served-model-name "$OMNI_SERVED_NAME" \\
  --host 127.0.0.1 --port 8091 \\
  --tensor-parallel-size 4 \\
  --trust-remote-code \\
  --dtype auto \\
  --max-model-len 12288 \\
  --max-num-seqs 4 \\
  --gpu-memory-utilization 0.90 \\
  --limit-mm-per-prompt '{"image":2,"video":0,"audio":0}' \\
  --enforce-eager

说明：第三方AWQ若启动时出现weight_packed、quantization或kernel错误，立即停止；
先改用官方BF16权重验证服务链路，或重新量化成当前vLLM明确兼容的权重。
EOF
  elif [[ "$OMNI_SERVER_MODE" == "vllm_omni" ]]; then
    cat <<EOF
vLLM-Omni多阶段服务命令（默认部署拓扑不保证自动占满4张A6000）：

CUDA_VISIBLE_DEVICES=0,1,2,3 "$vllm_bin" serve "$OMNI_MODEL_PATH" \\
  --omni \\
  --served-model-name "$OMNI_SERVED_NAME" \\
  --host 127.0.0.1 --port 8091

此模式会在请求中发送modalities=["text"]。不要把普通start-agents用于该模型。
若要精确使用4卡，必须依据你安装的vLLM-Omni版本编写stage overrides/deploy YAML，
本包不会猜测阶段显存分配。
EOF
  else
    echo "[FAIL] OMNI_SERVER_MODE必须是vllm_thinker或vllm_omni" >&2
    exit 2
  fi
}

omni_shared_settings() {
  require_cfg
  [[ "$OMNI_SERVER_MODE" == "vllm_thinker" || "$OMNI_SERVER_MODE" == "vllm_omni" ]] || {
    echo "[FAIL] OMNI_SERVER_MODE必须是vllm_thinker或vllm_omni" >&2; exit 2;
  }
  [[ -n "${OMNI_SERVED_NAME:-}" ]] || { echo "[FAIL] OMNI_SERVED_NAME为空" >&2; exit 2; }
  cd "$RS"
  local text_flag="--no-text-only"
  if [[ "$OMNI_SERVER_MODE" == "vllm_omni" ]]; then text_flag="--text-only"; fi
  local cmd=("$PYTHON" main_layer/run.py materialize-shared-omni-agents
    --settings "$ZH_CFG"
    --output "$OMNI_CFG"
    --base-url "$OMNI_BASE_URL"
    --served-name "$OMNI_SERVED_NAME"
    "$text_flag")
  if [[ -n "${OMNI_MODEL_PATH:-}" ]]; then cmd+=(--model-path "$OMNI_MODEL_PATH"); fi
  "${cmd[@]}"
  echo "[PASS] 共享Omni三逻辑角色配置：$OMNI_CFG"
  echo "[IMPORTANT] 不要对该配置运行start-agents；模型服务必须由你单独启动一次。"
}

omni_smoke() {
  omni_shared_settings
  cd "$RS"
  "$PYTHON" main_layer/run.py check-shared-omni-endpoint --settings "$OMNI_CFG"
}

omni_shared_debug() {
  omni_smoke
  cd "$RS"
  if [[ "$OMNI_SERVER_MODE" == "vllm_omni" ]]; then
    export ALS_OMNI_TEXT_ONLY=true
  else
    export ALS_OMNI_TEXT_ONLY=false
  fi
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
  if [[ "$OMNI_SERVER_MODE" == "vllm_omni" ]]; then
    export ALS_OMNI_TEXT_ONLY=true
  else
    export ALS_OMNI_TEXT_ONLY=false
  fi
  "$PYTHON" main_layer/run.py check-debug-infra \
    --settings "$OMNI_CFG" --classes-file "$CLASSES_FILE" --lang zh
  "$PYTHON" main_layer/run.py check-debug \
    --settings "$OMNI_CFG" --classes-file "$CLASSES_FILE" --lang zh
  "$PYTHON" main_layer/run.py official-socratic-full \
    --settings "$OMNI_CFG" --classes-file "$CLASSES_FILE" --lang zh \
    train full 40 fresh
}

en_info() {
  echo "英文不变：继续使用原settings和原输出目录："
  echo "  $BASE_SETTINGS"
  echo "本包不会自动把英文数据改成新的JSON Direct格式。"
}

usage() {
  cat <<EOF
用法: bash commands/run_v1.2.1.sh <命令>

  prepare               创建中文独立settings并检查类别/标签
  datafull              生成Canonical、Direct、Ref和Agent输入
  b0                    评测未训练RS-EoT
  force-direct-prepare  Direct训练前全部门控，不占GPU训练
  force-direct-train    强制独立Direct训练并合并，不依赖Socratic Full
  eval-b1-direct        评测rs-eot-b1-direct
  safe-direct           无脑顺序：prepare→datafull→B0→Direct训练→B1测试
  socratic-debug        可选：修复后的中文fresh Debug40
  full                  仅在Debug基础设施和正式门均PASS后运行Full
  omni-preflight        检查4×A6000、Omni环境和模型元数据；不安装、不启动
  omni-launch-info      打印一个共享Omni服务的建议启动命令
  omni-shared-settings  生成三逻辑角色共用一个外部Omni端点的settings
  omni-smoke            对共享端点做文本+图像真实请求
  omni-shared-debug     不启动三份模型，使用共享端点运行fresh Debug40
  omni-shared-full      仅在共享端点Debug门控PASS后运行Full
  en-info               显示英文保持不变的说明
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
  omni-preflight) omni_preflight ;;
  omni-launch-info) omni_launch_info ;;
  omni-shared-settings) omni_shared_settings ;;
  omni-smoke) omni_smoke ;;
  omni-shared-debug) omni_shared_debug ;;
  omni-shared-full) omni_shared_full ;;
  en-info) en_info ;;
  *) usage; exit 2 ;;
esac
