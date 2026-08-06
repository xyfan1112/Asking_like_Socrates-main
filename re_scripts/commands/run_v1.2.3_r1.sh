#!/usr/bin/env bash
set -Eeuo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/commands/user_config.sh"
RS=${RE_SCRIPTS_ROOT_OVERRIDE:-"$REPO_ROOT/re_scripts"}
BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
[[ -n "$BOOTSTRAP_PY" ]] || { echo "[FAIL] 找不到 bootstrap Python" >&2; exit 2; }

fail() { echo "[FAIL] $*" >&2; exit 2; }
info() { echo "[INFO] $*"; }

[[ -d "$RS" ]] || fail "re_scripts不存在: $RS"
[[ -f "$BASE_SETTINGS" ]] || fail "BASE_SETTINGS不存在: $BASE_SETTINGS。v1.2.3不再静默退回其他settings。"
case "$RUN_LANG" in en|zh) ;; *) fail "RUN_LANG必须是en或zh: $RUN_LANG" ;; esac
case "$DATASET_VARIANT" in
  raw) ACTIVE_DATASET_ROOT="$RAW_DATASET_ROOT" ;;
  scene_disjoint) ACTIVE_DATASET_ROOT="$SCENE_DISJOINT_DATASET_ROOT" ;;
  *) fail "DATASET_VARIANT必须是raw或scene_disjoint" ;;
esac
[[ -d "$RAW_DATASET_ROOT" ]] || fail "原始数据集目录不存在: $RAW_DATASET_ROOT"
if [[ -z "${CLASSES_FILE:-}" ]]; then CLASSES_FILE="$RAW_DATASET_ROOT/classes.txt"; fi
[[ -f "$CLASSES_FILE" ]] || fail "classes.txt不存在: $CLASSES_FILE"
if [[ "$RUN_LANG" == "en" ]]; then ACTIVE_OUTPUT_ROOT="$EN_OUTPUT_ROOT"; else ACTIVE_OUTPUT_ROOT="$ZH_OUTPUT_ROOT"; fi
ACTIVE_CFG="$ACTIVE_OUTPUT_ROOT/config/settings.${RUN_LANG}.v1.2.3.json"
OMNI_CFG="$ACTIVE_OUTPUT_ROOT/config/settings.${RUN_LANG}.v1.2.3.omni-shared.json"
INDEPENDENT_CFG="$ACTIVE_OUTPUT_ROOT/config/settings.${RUN_LANG}.v1.2.3.three-services.json"
export OMNI_RUNTIME_DIR="$ACTIVE_OUTPUT_ROOT/omni_tp4_service"
if [[ -z "${OMNI_SERVED_NAME:-}" ]]; then
  OMNI_SERVED_NAME=$(basename "$OMNI_MODELSCOPE_ID" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '-' | sed 's/^-*//; s/-*$//')
  export OMNI_SERVED_NAME
fi


validate_scene_disjoint_manifest() {
  local manifest="$SCENE_DISJOINT_DATASET_ROOT/scene_split_manifest.json"
  [[ -f "$manifest" ]] || return 1
  "$BOOTSTRAP_PY" - "$manifest" "$RAW_DATASET_ROOT" "$SCENE_DISJOINT_DATASET_ROOT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); raw=Path(sys.argv[2]).resolve(); out=Path(sys.argv[3]).resolve()
r=json.loads(p.read_text(encoding='utf-8'))
errors=[]
if not r.get('passed'): errors.append('passed=false')
if r.get('scene_overlap'): errors.append('scene_overlap_nonempty')
if Path(str(r.get('source_root',''))).resolve()!=raw: errors.append('source_root_mismatch')
if Path(str(r.get('output_root',''))).resolve()!=out: errors.append('output_root_mismatch')
counts=r.get('output_image_counts') or {}
if int(counts.get('train',0))<=0 or int(counts.get('val',0))<=0: errors.append('empty_split')
if errors:
    print('[FAIL] scene_split_manifest invalid:', ','.join(errors), file=sys.stderr)
    raise SystemExit(2)
print(f"[PASS] scene-disjoint manifest: train={counts.get('train')} val={counts.get('val')} overlap=0")
PY
}

build_scene_disjoint() {
  [[ -d "$RAW_DATASET_ROOT" ]] || fail "原始数据集目录不存在: $RAW_DATASET_ROOT"
  if [[ -d "$SCENE_DISJOINT_DATASET_ROOT" ]] && \
     find "$SCENE_DISJOINT_DATASET_ROOT" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    validate_scene_disjoint_manifest || fail "无泄漏目录已非空，但缺少/不匹配有效 manifest：$SCENE_DISJOINT_DATASET_ROOT。为防误删，脚本不会覆盖；请先打包检查或手工改名旧目录。"
    echo "[REUSE] 无泄漏数据已存在，不重复生成: $SCENE_DISJOINT_DATASET_ROOT"
    return 0
  fi
  mkdir -p "$(dirname "$SCENE_DISJOINT_DATASET_ROOT")" "$ACTIVE_OUTPUT_ROOT/config"
  echo "[BUILD] 从原始数据生成无场景泄漏副本：$RAW_DATASET_ROOT -> $SCENE_DISJOINT_DATASET_ROOT"
  "$BOOTSTRAP_PY" "$RS/main_layer/run.py" split-scenes \
    --settings "$BASE_SETTINGS" \
    --output-root "$SCENE_DISJOINT_DATASET_ROOT" \
    --settings-output "$ACTIVE_OUTPUT_ROOT/config/settings.scene_disjoint.generated.json"
  validate_scene_disjoint_manifest || fail "无泄漏数据生成后校验失败"
}

ensure_active_dataset() {
  case "$DATASET_VARIANT" in
    raw)
      [[ -d "$RAW_DATASET_ROOT" ]] || fail "原始数据集目录不存在: $RAW_DATASET_ROOT"
      ;;
    scene_disjoint)
      if [[ -d "$SCENE_DISJOINT_DATASET_ROOT" ]] && validate_scene_disjoint_manifest; then
        return 0
      fi
      if [[ "${AUTO_BUILD_SCENE_DISJOINT:-1}" == "1" ]]; then
        build_scene_disjoint
      else
        fail "无泄漏数据不存在或校验失败: $SCENE_DISJOINT_DATASET_ROOT。请运行 bash commands/run_v1.2.3.sh split-scenes"
      fi
      ;;
  esac
}

prepare() {
  ensure_active_dataset
  mkdir -p "$ACTIVE_OUTPUT_ROOT/config"
  local -a cmd
  cmd=("$BOOTSTRAP_PY" "$RS/tools/materialize_language_settings.py"
    --base-settings "$BASE_SETTINGS"
    --classes-file "$CLASSES_FILE"
    --lang "$RUN_LANG"
    --output-root "$ACTIVE_OUTPUT_ROOT"
    --output "$ACTIVE_CFG"
    --dataset-root "$ACTIVE_DATASET_ROOT"
    --dataset-variant "$DATASET_VARIANT"
    --direct-qa-mode "$DIRECT_QA_MODE"
    --eval-gpus "$EVAL_GPUS"
    --eval-base-port "$EVAL_BASE_PORT"
    --eval-gpu-memory-utilization "$EVAL_GPU_MEMORY_UTILIZATION"
    --eval-max-model-len "$EVAL_MAX_MODEL_LEN"
    --sft-visible-gpus "$SFT_VISIBLE_GPUS"
    --sft-world-size "$SFT_WORLD_SIZE"
    --sft-gradient-accum "$SFT_GRADIENT_ACCUMULATION")
  [[ "$RUN_LANG" == "en" ]] && cmd+=(--class-map "$CLASS_MAP_FILE")
  [[ -n "$DATA_PYTHON_OVERRIDE" ]] && cmd+=(--data-python "$DATA_PYTHON_OVERRIDE")
  [[ -n "$SFT_PYTHON_OVERRIDE" ]] && cmd+=(--sft-python "$SFT_PYTHON_OVERRIDE")
  [[ -n "$VLLM_PYTHON_OVERRIDE" ]] && cmd+=(--vllm-python "$VLLM_PYTHON_OVERRIDE")
  [[ "$REQUIRE_FROZEN_VISION_TOWER" == "1" ]] && cmd+=(--require-frozen-vision)
  "${cmd[@]}"
  # check-taxonomy must use the active generated classes file, not the source Chinese file in English mode.
  local active_classes
  active_classes=$("$BOOTSTRAP_PY" - "$ACTIVE_CFG" <<'PY'
import json,sys
print(json.load(open(sys.argv[1],encoding='utf-8'))['taxonomy']['classes_file'])
PY
)
  "$BOOTSTRAP_PY" "$RS/main_layer/run.py" check-taxonomy \
    --settings "$ACTIVE_CFG" --classes-file "$active_classes" --lang "$RUN_LANG"
  echo "[PASS] prepare完成。它只生成/验证配置，不生成QA、不启动模型、不训练。"
  print_modes
}

require_cfg() {
  ensure_active_dataset
  [[ -f "$ACTIVE_CFG" ]] || prepare
  "$BOOTSTRAP_PY" - "$ACTIVE_CFG" "$RUN_LANG" "$ACTIVE_DATASET_ROOT" "$DIRECT_QA_MODE" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8')); v=s.get('v1_2_3',{})
errors=[]
if v.get('version')!='1.2.3': errors.append('version')
if v.get('language')!=sys.argv[2]: errors.append('language')
if v.get('dataset_root')!=sys.argv[3]: errors.append('dataset_root')
if v.get('direct_qa_mode')!=sys.argv[4]: errors.append('direct_qa_mode')
if errors: raise SystemExit('resolved settings mismatch: '+','.join(errors)+'. rerun prepare after removing old config')
PY
}

active_classes() {
  require_cfg
  "$BOOTSTRAP_PY" - "$ACTIVE_CFG" <<'PY'
import json,sys
print(json.load(open(sys.argv[1],encoding='utf-8'))['taxonomy']['classes_file'])
PY
}

print_modes() {
  cat <<EOF2
[MODE] language=$RUN_LANG (default=en; one switch changes data/QA/trajectory/train/eval paths)
[MODE] dataset=$DATASET_VARIANT root=$ACTIVE_DATASET_ROOT
[MODE] Direct=$DIRECT_QA_MODE (default legacy_class_roi_obb)
[MODE] Eval=$EVAL_TOPOLOGY GPUs=$EVAL_GPUS ports=$EVAL_BASE_PORT-$((EVAL_BASE_PORT+3))
[MODE] SFT=LoRA DDP GPUs=$SFT_VISIBLE_GPUS world_size=$SFT_WORLD_SIZE; vision tower required frozen=$REQUIRE_FROZEN_VISION_TOWER
[MODE] Socratic=$SOCRATIC_TOPOLOGY; shared=one 8091 TP4 endpoint/three separate message contexts; independent=8001/8002/8003
[MODE] prepare只创建独立配置、类别映射和运行签名。
EOF2
}

status() {
  if [[ -f "$ACTIVE_CFG" ]]; then
    "$BOOTSTRAP_PY" "$RS/tools/status_v1_2_3.py" --settings "$ACTIVE_CFG" || true
  else
    echo "[STATUS] active settings尚未生成: $ACTIVE_CFG"
    print_modes
  fi
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
}

require_legacy_or_explicit_json_eval() {
  if [[ "$DIRECT_QA_MODE" != "legacy_class_roi_obb" && "$ALLOW_EXPERIMENTAL_JSON_TEST_MATRIX" != "1" ]]; then
    fail "Direct模式=$DIRECT_QA_MODE与原test-matrix不同；设置ALLOW_EXPERIMENTAL_JSON_TEST_MATRIX=1才允许跨协议评估。"
  fi
}

data() {
  require_cfg
  local classes; classes=$(active_classes)
  cd "$RS"
  "$BOOTSTRAP_PY" main_layer/run.py data-full --settings "$ACTIVE_CFG" --classes-file "$classes" --lang "$RUN_LANG"
  "$BOOTSTRAP_PY" "$RS/tools/summarize_pipeline_v1_2_3.py" --settings "$ACTIVE_CFG" || true
  echo "[PASS] 数据层完成：Canonical/Direct/Ref/Agent inputs。"
}

stop_independent_agents() {
  [[ -f "$ACTIVE_CFG" ]] && "$BOOTSTRAP_PY" "$RS/main_layer/run.py" stop-agents --settings "$ACTIVE_CFG" || true
  [[ -f "$INDEPENDENT_CFG" ]] && "$BOOTSTRAP_PY" "$RS/main_layer/run.py" stop-agents --settings "$INDEPENDENT_CFG" || true
}
stop_omni() { bash "$RS/commands/manage_omni_tp4.sh" stop >/dev/null 2>&1 || true; }
stop_socratic_all() { stop_independent_agents; stop_omni; }

known_eval_keys=(rs_eot rs_eot_b1_direct rs-eot-b1-direct rs_eot_b2_socratic)
read_eval_arrays() {
  IFS=',' read -r -a EVAL_GPU_ARRAY <<< "$EVAL_GPUS"
  if [[ "$EVAL_TOPOLOGY" == "single" ]]; then EVAL_GPU_ARRAY=("${EVAL_GPU_ARRAY[0]}"); fi
  EVAL_PORT_ARRAY=(); EVAL_URL_ARRAY=()
  for ((i=0;i<${#EVAL_GPU_ARRAY[@]};i++)); do
    EVAL_PORT_ARRAY+=("$((EVAL_BASE_PORT+i))")
    EVAL_URL_ARRAY+=("http://127.0.0.1:$((EVAL_BASE_PORT+i))/v1")
  done
  EVAL_URLS_CSV=$(IFS=,; echo "${EVAL_URL_ARRAY[*]}")
}
stop_eval_replicas() {
  [[ -f "$ACTIVE_CFG" ]] || return 0
  read_eval_arrays
  for port in "${EVAL_PORT_ARRAY[@]}"; do
    for key in "${known_eval_keys[@]}"; do
      bash "$RS/test_layer/stop_eval_model.sh" "$ACTIVE_CFG" "$key" "$port" >/dev/null 2>&1 || true
    done
  done
}
eval_model_preflight() {
  local key=$1
  require_cfg
  "$BOOTSTRAP_PY" - "$ACTIVE_CFG" "$key" <<'PY'
import json,sys
from pathlib import Path
s=json.load(open(sys.argv[1],encoding='utf-8')); key=sys.argv[2]
models=s.get('models',{})
if key not in models:
    raise SystemExit(f"[FAIL] evaluation model key missing in active settings: {key}")
p=Path(models[key]['path'])
print(f"[EVAL PREFLIGHT] key={key} path={p}")
if not (p/'config.json').is_file():
    stage={'rs_eot':'B0基础模型','rs_eot_b1_direct':'B1 Direct合并模型','rs-eot-b1-direct':'B1 Direct合并模型','rs_eot_b2_socratic':'B2 Socratic合并模型'}.get(key,key)
    hint={
      'rs_eot':'确认 /home/vieo/vieo/fxy_workspace/fxy/models/RS-EoT-7B 已完整下载。',
      'rs_eot_b1_direct':'先成功运行 b1-train；旧日志中的 CONFIG/CONTRACT PASS 不等于训练完成。',
      'rs-eot-b1-direct':'先成功运行 b1-train；旧日志中的 CONFIG/CONTRACT PASS 不等于训练完成。',
      'rs_eot_b2_socratic':'该模型只在 B2 训练并合并后存在；safe-direct 正常不会请求 B2。请核对实际命令和活动脚本。',
    }.get(key,'')
    raise SystemExit(f"[FAIL] {stage}缺少 config.json: {p/'config.json'}\n[HINT] {hint}")
PY
}

start_eval_replicas() {
  local key=$1
  eval_model_preflight "$key"
  read_eval_arrays; stop_eval_replicas
  stop_socratic_all
  local -a pids=()
  for ((i=0;i<${#EVAL_GPU_ARRAY[@]};i++)); do
    echo "[EVAL START] model=$key gpu=${EVAL_GPU_ARRAY[$i]} port=${EVAL_PORT_ARRAY[$i]}"
    bash "$RS/test_layer/start_eval_model.sh" "$ACTIVE_CFG" "$key" "${EVAL_GPU_ARRAY[$i]}" "${EVAL_PORT_ARRAY[$i]}" &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { stop_eval_replicas; fail "至少一个评估副本启动失败"; }
}
run_eval_matrix() {
  local key=$1
  require_legacy_or_explicit_json_eval
  read_eval_arrays
  if [[ "$EVAL_TOPOLOGY" == "replica4" ]]; then
    bash "$RS/main_layer/run_test_matrix_replica4.sh" "$ACTIVE_CFG" "$key" "$EVAL_URLS_CSV" fresh
  else
    bash "$RS/main_layer/run_test_matrix.sh" "$ACTIVE_CFG" "$key" "${EVAL_URL_ARRAY[0]}" fresh
  fi
}
eval_model() {
  local key=$1
  start_eval_replicas "$key"
  trap 'stop_eval_replicas' EXIT
  run_eval_matrix "$key"
  stop_eval_replicas; trap - EXIT
}

b0() { require_cfg; eval_model rs_eot; echo "[PASS] B0完成：基础RS-EoT未训练评估。"; }

archive_paths() {
  local mode=$1
  require_cfg
  "$BOOTSTRAP_PY" - "$ACTIVE_CFG" "$mode" <<'PY'
import json,shutil,sys
from datetime import datetime
from pathlib import Path
s=json.load(open(sys.argv[1],encoding='utf-8')); t=s['training']; mode=sys.argv[2]
keys={'b1':['b1_direct_standalone_adapter_output','b1_direct_standalone_merged_output'],'b2':['b2_adapter_output','b2_merged_output']}[mode]
stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
for key in keys:
 p=Path(t[key])
 if p.exists():
  dst=p.with_name(p.name+'_archive_'+stamp); shutil.move(str(p),str(dst)); print('[ARCHIVE]',p,'->',dst)
PY
}

b1_train_only() {
  require_cfg; stop_eval_replicas; stop_socratic_all; archive_paths b1
  cd "$RS"
  "$BOOTSTRAP_PY" main_layer/run.py train-b1-direct-standalone --settings "$ACTIVE_CFG"
  "$BOOTSTRAP_PY" main_layer/run.py merge-b1-direct-standalone --settings "$ACTIVE_CFG"
  "$BOOTSTRAP_PY" main_layer/run.py register-b1-direct-model --settings "$ACTIVE_CFG"
  "$BOOTSTRAP_PY" "$RS/tools/summarize_pipeline_v1_2_3.py" --settings "$ACTIVE_CFG" || true
  echo "[PASS] B1 Direct LoRA训练与合并完成。"
}
b1_eval_only() { require_cfg; "$BOOTSTRAP_PY" "$RS/main_layer/run.py" register-b1-direct-model --settings "$ACTIVE_CFG"; eval_model rs_eot_b1_direct; }
b1() { b1_train_only; b1_eval_only; echo "[PASS] B1结果完成。"; }

materialize_independent() {
  require_cfg
  "$BOOTSTRAP_PY" "$RS/tools/materialize_independent_agent_settings.py" \
    --settings "$ACTIVE_CFG" --output "$INDEPENDENT_CFG" \
    --reasoner-model "$REASONER_MODEL_PATH" --perceiver-model "$PERCEIVER_MODEL_PATH" --verifier-model "$VERIFIER_MODEL_PATH"
}
materialize_shared() {
  require_cfg
  "$BOOTSTRAP_PY" "$RS/tools/materialize_shared_omni_agent_settings.py" \
    --settings "$ACTIVE_CFG" --output "$OMNI_CFG" --base-url "$OMNI_BASE_URL" \
    --served-name "$OMNI_SERVED_NAME" --model-path "$OMNI_MODEL_PATH" \
    --full-concurrency "$OMNI_FULL_CONCURRENCY" --tensor-parallel-size "$OMNI_TP_SIZE" --checkpoint-id "$OMNI_MODELSCOPE_ID"
}

omni_preflight() {
  require_cfg
  [[ -x "$OMNI_PYTHON" ]] || fail "OMNI_PYTHON不可执行: $OMNI_PYTHON"
  [[ -f "$OMNI_MODEL_PATH/config.json" ]] || fail "Omni模型缺少config.json: $OMNI_MODEL_PATH"
  local flag=--require-no-tts; [[ "$OMNI_REQUIRE_NO_TTS" == "1" ]] || flag=--no-require-no-tts
  "$OMNI_PYTHON" "$RS/tools/check_qwen3_omni_4gpu.py" --settings "$ACTIVE_CFG" --model-path "$OMNI_MODEL_PATH" \
    --model-id "$OMNI_MODELSCOPE_ID" --python "$OMNI_PYTHON" --tensor-parallel-size "$OMNI_TP_SIZE" "$flag"
}
omni_start() { stop_eval_replicas; stop_independent_agents; omni_preflight; bash "$RS/commands/manage_omni_tp4.sh" start; }
omni_status() { bash "$RS/commands/manage_omni_tp4.sh" status; }
omni_log() { bash "$RS/commands/manage_omni_tp4.sh" log "${2:-200}"; }
omni_smoke() { materialize_shared; "$BOOTSTRAP_PY" "$RS/main_layer/run.py" check-shared-omni-endpoint --settings "$OMNI_CFG"; }

socratic_audits() {
  local cfg=$1 mode=$2 classes; classes=$(active_classes)
  "$BOOTSTRAP_PY" "$RS/main_layer/run.py" inspect-api-truncation --settings "$cfg" --classes-file "$classes" --lang "$RUN_LANG" --mode "$mode"
  "$BOOTSTRAP_PY" "$RS/main_layer/run.py" check-debug-infra --settings "$cfg" --classes-file "$classes" --lang "$RUN_LANG"
  "$BOOTSTRAP_PY" "$RS/main_layer/run.py" check-debug --settings "$cfg" --classes-file "$classes" --lang "$RUN_LANG"
}
socratic_generate() {
  local cfg=$1 mode=$2 samples=$3 classes; classes=$(active_classes)
  "$BOOTSTRAP_PY" "$RS/main_layer/run.py" official-socratic-full --settings "$cfg" --classes-file "$classes" --lang "$RUN_LANG" train "$mode" "$samples" fresh
}

socratic_one_debug() {
  require_cfg; materialize_shared; omni_start
  trap 'stop_omni' EXIT
  omni_smoke
  socratic_generate "$OMNI_CFG" debug "$SOCRATIC_DEBUG_SAMPLES"
  socratic_audits "$OMNI_CFG" debug
  stop_omni; trap - EXIT
  echo "[PASS] 共享Omni Debug完成。一个物理端点，三个逻辑角色的messages彼此分开。"
}
socratic_one_full() {
  require_cfg; materialize_shared; omni_start
  trap 'stop_omni' EXIT
  socratic_audits "$OMNI_CFG" debug
  socratic_generate "$OMNI_CFG" full "$SOCRATIC_FULL_TASK_SAMPLES"
  stop_omni; trap - EXIT
  "$BOOTSTRAP_PY" "$RS/tools/audit_b1_b2_direct_pairing.py" --settings "$ACTIVE_CFG"
  "$BOOTSTRAP_PY" "$RS/tools/summarize_pipeline_v1_2_3.py" --settings "$ACTIVE_CFG" || true
  echo "[PASS] 共享Omni Full及后处理/训练数据转换完成。"
}
socratic_three_debug() {
  require_cfg; stop_eval_replicas; stop_omni; materialize_independent
  "$BOOTSTRAP_PY" "$RS/main_layer/run.py" start-agents --settings "$INDEPENDENT_CFG"
  trap 'stop_independent_agents' EXIT
  socratic_generate "$INDEPENDENT_CFG" debug "$SOCRATIC_DEBUG_SAMPLES"
  socratic_audits "$INDEPENDENT_CFG" debug
  stop_independent_agents; trap - EXIT
  echo "[PASS] 原三服务Debug完成。"
}
socratic_three_full() {
  require_cfg; stop_eval_replicas; stop_omni; materialize_independent
  "$BOOTSTRAP_PY" "$RS/main_layer/run.py" start-agents --settings "$INDEPENDENT_CFG"
  trap 'stop_independent_agents' EXIT
  socratic_audits "$INDEPENDENT_CFG" debug
  socratic_generate "$INDEPENDENT_CFG" full "$SOCRATIC_FULL_TASK_SAMPLES"
  stop_independent_agents; trap - EXIT
  "$BOOTSTRAP_PY" "$RS/tools/audit_b1_b2_direct_pairing.py" --settings "$ACTIVE_CFG"
  "$BOOTSTRAP_PY" "$RS/tools/summarize_pipeline_v1_2_3.py" --settings "$ACTIVE_CFG" || true
  echo "[PASS] 原三服务Full及后处理/训练数据转换完成。"
}

b2_train_only() {
  require_cfg; stop_eval_replicas; stop_socratic_all
  "$BOOTSTRAP_PY" "$RS/tools/audit_b1_b2_direct_pairing.py" --settings "$ACTIVE_CFG"
  archive_paths b2
  "$BOOTSTRAP_PY" "$RS/main_layer/run.py" train-b2 --settings "$ACTIVE_CFG"
  "$BOOTSTRAP_PY" "$RS/main_layer/run.py" merge --settings "$ACTIVE_CFG" b2
  "$BOOTSTRAP_PY" "$RS/main_layer/run.py" register-b2-model --settings "$ACTIVE_CFG"
  "$BOOTSTRAP_PY" "$RS/tools/summarize_pipeline_v1_2_3.py" --settings "$ACTIVE_CFG" || true
  echo "[PASS] B2 Socratic LoRA训练与合并完成。"
}
b2_eval_only() { require_cfg; "$BOOTSTRAP_PY" "$RS/main_layer/run.py" register-b2-model --settings "$ACTIVE_CFG"; eval_model rs_eot_b2_socratic; }
b2() { b2_train_only; b2_eval_only; echo "[PASS] B2结果完成。"; }

compare() { require_cfg; "$BOOTSTRAP_PY" "$RS/main_layer/run.py" compare-b0-b1-b2 --settings "$ACTIVE_CFG"; }

socratic_debug() {
  case "$SOCRATIC_TOPOLOGY" in
    shared) socratic_one_debug ;;
    independent) socratic_three_debug ;;
    *) fail "SOCRATIC_TOPOLOGY必须是shared或independent: $SOCRATIC_TOPOLOGY" ;;
  esac
}
socratic_full() {
  case "$SOCRATIC_TOPOLOGY" in
    shared) socratic_one_full ;;
    independent) socratic_three_full ;;
    *) fail "SOCRATIC_TOPOLOGY必须是shared或independent: $SOCRATIC_TOPOLOGY" ;;
  esac
}

safe_direct() { echo "[FLOW] safe-direct只运行 B0(rs_eot) 与 B1(rs_eot_b1_direct)，不会请求B2。"; prepare; data; b0; b1; echo "[PASS] safe-direct完成；未启动Socratic。"; }
safe_b2_one() { prepare; data; socratic_one_debug; socratic_one_full; b2; }
safe_b2_three() { prepare; data; socratic_three_debug; socratic_three_full; b2; }
safe_all_one() { prepare; data; b0; b1; socratic_one_debug; socratic_one_full; b2; compare; }
safe_all_three() { prepare; data; b0; b1; socratic_three_debug; socratic_three_full; b2; compare; }
safe_b2() {
  case "$SOCRATIC_TOPOLOGY" in
    shared) safe_b2_one ;;
    independent) safe_b2_three ;;
    *) fail "SOCRATIC_TOPOLOGY必须是shared或independent: $SOCRATIC_TOPOLOGY" ;;
  esac
}
safe_all() {
  case "$SOCRATIC_TOPOLOGY" in
    shared) safe_all_one ;;
    independent) safe_all_three ;;
    *) fail "SOCRATIC_TOPOLOGY必须是shared或independent: $SOCRATIC_TOPOLOGY" ;;
  esac
}

collect() {
  local mode=${2:-error}; require_cfg
  "$BOOTSTRAP_PY" "$RS/tools/collect_support_bundle.py" --settings "$ACTIVE_CFG" --repo-root "$RS" --mode "$mode" \
    --max-log-mb "$SUPPORT_BUNDLE_MAX_LOG_MB" --sample-rows "$SUPPORT_BUNDLE_INCLUDE_SAMPLE_ROWS"
}

usage() {
cat <<'EOF2'
用法：bash commands/run_v1.2.3.sh <命令>

基础：
  status                 查看语言、数据、GPU拓扑、端口和B0/B1/B2产物状态
  split-scenes           从 yw128 生成/校验 yw128_scene_disjoint；已有有效manifest则复用
  prepare                先确保无泄漏数据存在，再生成/检查当前语言独立配置
  data                   OBB数据审计→Direct→Ref→Agent inputs
  b0                     四卡副本评估基础模型
  b1                     四卡LoRA Direct训练→合并→四卡评估
  b2                     使用现有严格过滤Socratic数据训练→合并→四卡评估
  compare                汇总B0/B1/B2

Socratic：
  socratic-debug         按 user_config.sh 的 SOCRATIC_TOPOLOGY 自动选共享/三服务
  socratic-full          按 SOCRATIC_TOPOLOGY 自动运行 Full
  socratic-one-debug     共享Omni：8091单端点、TP4、三独立messages，Debug40
  socratic-one-full      共享Omni，Debug门通过后Full
  socratic-three-debug   原Reasoner/Perceiver/Verifier三服务Debug40
  socratic-three-full    原三服务Full

无脑安全流程：
  safe-direct            prepare→data→B0→B1；完全不需要三个智能体
  safe-b2                按 SOCRATIC_TOPOLOGY：data→Debug/Full→B2
  safe-all               按 SOCRATIC_TOPOLOGY：data→B0→B1→B2→compare
  safe-b2-one            data→共享Omni Debug/Full→B2
  safe-b2-three          data→原三服务 Debug/Full→B2
  safe-all-one           data→B0→B1→共享Omni B2→compare
  safe-all-three         data→B0→B1→原三服务 B2→compare

服务与诊断：
  omni-preflight|omni-start|omni-status|omni-log|omni-stop
  collect error|data|socratic|train-b1|train-b2|eval-b0|eval-b1|eval-b2|all
EOF2
}

case "${1:-}" in
  status) status ;; split-scenes) build_scene_disjoint ;; prepare) prepare ;; data|datafull) data ;; b0) b0 ;;
  b1) b1 ;; b1-train) b1_train_only ;; b1-eval) b1_eval_only ;;
  b2) b2 ;; b2-train) b2_train_only ;; b2-eval) b2_eval_only ;; compare) compare ;;
  socratic-debug) socratic_debug ;; socratic-full) socratic_full ;;
  socratic-one-debug) socratic_one_debug ;; socratic-one-full) socratic_one_full ;;
  socratic-three-debug) socratic_three_debug ;; socratic-three-full) socratic_three_full ;;
  safe-direct) safe_direct ;; safe-b2) safe_b2 ;; safe-all) safe_all ;;
  safe-b2-one) safe_b2_one ;; safe-b2-three) safe_b2_three ;;
  safe-all-one) safe_all_one ;; safe-all-three) safe_all_three ;;
  omni-preflight) omni_preflight ;; omni-start) omni_start ;; omni-status) omni_status ;;
  omni-log) omni_log "$@" ;; omni-stop) stop_omni ;; collect) collect "$@" ;;
  *) usage; exit 2 ;;
esac
