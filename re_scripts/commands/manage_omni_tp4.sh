#!/usr/bin/env bash
# 兼容文件名保留为 manage_omni_tp4.sh；实现已改为可配置TP，不再固定四卡。
set -Eeuo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/commands/user_config.sh"

ACTIVE_ROOT="$EN_OUTPUT_ROOT"
[[ "${RUN_LANG:-en}" == "zh" ]] && ACTIVE_ROOT="$ZH_OUTPUT_ROOT"
if [[ -z "${OMNI_SERVED_NAME:-}" ]]; then
  OMNI_SERVED_NAME=$(basename "${OMNI_MODELSCOPE_ID:-qwen3-omni}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '-' | sed 's/^-*//; s/-*$//')
  export OMNI_SERVED_NAME
fi
RUNTIME_DIR=${OMNI_RUNTIME_DIR:-"$ACTIVE_ROOT/omni_shared_service"}
PID_FILE="$RUNTIME_DIR/server.pid"
CMD_FILE="$RUNTIME_DIR/server.command.txt"
LOG_FILE="$RUNTIME_DIR/server.log"
mkdir -p "$RUNTIME_DIR"

fail() { echo "[FAIL] $*" >&2; exit 2; }

validate_runtime() {
  [[ -x "${OMNI_PYTHON:-}" ]] || fail "OMNI_PYTHON不可执行: ${OMNI_PYTHON:-空}"
  "$OMNI_PYTHON" -c 'import vllm,torch; print("[PASS] vllm",vllm.__version__,"torch",torch.__version__)' \
    || fail "OMNI_PYTHON无法导入vllm/torch"

  local -a visible=()
  IFS=',' read -r -a visible <<< "${OMNI_VISIBLE_GPUS:-}"
  local count=0 item
  for item in "${visible[@]}"; do [[ -n "${item// /}" ]] && ((count+=1)); done
  [[ "${OMNI_TP_SIZE:-0}" =~ ^[0-9]+$ ]] || fail "OMNI_TP_SIZE必须是正整数"
  (( OMNI_TP_SIZE >= 1 )) || fail "OMNI_TP_SIZE必须>=1"
  (( count >= OMNI_TP_SIZE )) || fail "OMNI_VISIBLE_GPUS=$OMNI_VISIBLE_GPUS只有$count张卡，但OMNI_TP_SIZE=$OMNI_TP_SIZE"
}

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid=$(cat "$PID_FILE" 2>/dev/null || true)
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

wait_endpoint() {
  local timeout=${1:-900}
  "$OMNI_PYTHON" - "$OMNI_BASE_URL" "$OMNI_SERVED_NAME" "$timeout" <<'PY'
import json,sys,time,urllib.request
base,expected,timeout=sys.argv[1],sys.argv[2],int(sys.argv[3])
url=base.rstrip('/')+'/models'
deadline=time.time()+timeout
last=''
while time.time()<deadline:
    try:
        req=urllib.request.Request(url,headers={'Authorization':'Bearer EMPTY'})
        with urllib.request.urlopen(req,timeout=10) as r:
            data=json.loads(r.read())
        ids=[str(x.get('id')) for x in data.get('data',[])]
        if expected in ids:
            print('[PASS] Omni endpoint ready:', ids)
            raise SystemExit(0)
        last=f'served name not ready: {ids}'
    except Exception as exc:
        last=f'{type(exc).__name__}: {exc}'
    time.sleep(5)
print('[FAIL] endpoint timeout:',last,file=sys.stderr)
raise SystemExit(2)
PY
}

start_server() {
  validate_runtime
  [[ -d "$OMNI_MODEL_PATH" ]] || fail "模型目录不存在: $OMNI_MODEL_PATH"
  [[ -f "$OMNI_MODEL_PATH/config.json" ]] || fail "模型缺少config.json: $OMNI_MODEL_PATH/config.json"
  if is_running; then
    echo "[INFO] Omni服务已运行，PID=$(cat "$PID_FILE")"
    exit 0
  fi
  rm -f "$PID_FILE"

  local help_text profile util seqs eager
  help_text=$("$OMNI_PYTHON" -m vllm.entrypoints.cli.main serve --help=all 2>&1 || true)
  [[ "$help_text" == *"--tensor-parallel-size"* ]] || fail "当前vLLM CLI未显示--tensor-parallel-size"
  [[ "$help_text" == *"--max-model-len"* ]] || fail "当前vLLM CLI未显示--max-model-len"
  [[ "$help_text" == *"--max-num-seqs"* ]] || fail "当前vLLM CLI未显示--max-num-seqs"
  [[ "$help_text" == *"--gpu-memory-utilization"* ]] || fail "当前vLLM CLI未显示--gpu-memory-utilization"

  profile=${OMNI_PROFILE:-safe}
  case "$profile" in
    safe)
      util=${OMNI_SAFE_GPU_MEMORY_UTILIZATION:-${OMNI_GPU_MEMORY_UTILIZATION:-0.80}}
      seqs=${OMNI_SAFE_MAX_NUM_SEQS:-${OMNI_MAX_NUM_SEQS:-1}}
      eager=1
      ;;
    balanced)
      util=${OMNI_GPU_MEMORY_UTILIZATION:-0.85}
      seqs=${OMNI_MAX_NUM_SEQS:-2}
      eager=0
      ;;
    fast)
      util=${OMNI_FAST_GPU_MEMORY_UTILIZATION:-${OMNI_GPU_MEMORY_UTILIZATION:-0.85}}
      seqs=${OMNI_FAST_MAX_NUM_SEQS:-${OMNI_MAX_NUM_SEQS:-2}}
      eager=0
      ;;
    *) fail "OMNI_PROFILE必须是safe/balanced/fast，当前=$profile" ;;
  esac

  local -a cmd
  cmd=(
    "$OMNI_PYTHON" -m vllm.entrypoints.cli.main
    serve "$OMNI_MODEL_PATH"
    --served-model-name "$OMNI_SERVED_NAME"
    --host "$OMNI_HOST" --port "$OMNI_PORT"
    --tensor-parallel-size "$OMNI_TP_SIZE"
    --dtype auto
    --max-model-len "$OMNI_MAX_MODEL_LEN"
    --max-num-seqs "$seqs"
    --gpu-memory-utilization "$util"
  )

  [[ "$help_text" == *"--distributed-executor-backend"* ]] && cmd+=(--distributed-executor-backend mp)
  [[ "$help_text" == *"--limit-mm-per-prompt"* ]] && cmd+=(--limit-mm-per-prompt "$OMNI_MM_LIMITS")
  [[ "$help_text" == *"--allowed-local-media-path"* ]] && cmd+=(--allowed-local-media-path "$OMNI_ALLOWED_LOCAL_MEDIA_PATH")
  [[ "${OMNI_TRUST_REMOTE_CODE:-1}" == "1" && "$help_text" == *"--trust-remote-code"* ]] && cmd+=(--trust-remote-code)
  [[ "$eager" == "1" && "$help_text" == *"--enforce-eager"* ]] && cmd+=(--enforce-eager)
  if [[ "${OMNI_QUANTIZATION:-auto}" != "auto" ]]; then
    [[ "$help_text" == *"--quantization"* ]] || fail "当前vLLM不支持--quantization，但OMNI_QUANTIZATION=${OMNI_QUANTIZATION}"
    cmd+=(--quantization "$OMNI_QUANTIZATION")
  fi
  if [[ "${OMNI_ENABLE_PREFIX_CACHING:-0}" == "1" && "$help_text" == *"--enable-prefix-caching"* ]]; then
    cmd+=(--enable-prefix-caching)
  fi

  {
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$OMNI_VISIBLE_GPUS"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } > "$CMD_FILE"

  echo "[INFO] 启动一个Qwen3-Omni AWQ-No-TTS共享服务"
  echo "[INFO] physical_gpus=$OMNI_VISIBLE_GPUS TP=$OMNI_TP_SIZE；一个权重副本，三个逻辑角色共享端点"
  echo "[INFO] profile=$profile max_num_seqs=$seqs gpu_memory_utilization=$util max_model_len=$OMNI_MAX_MODEL_LEN"
  echo "[INFO] log=$LOG_FILE"
  echo "[CMD] $(cat "$CMD_FILE")"

  export CUDA_VISIBLE_DEVICES="$OMNI_VISIBLE_GPUS"
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

  if command -v setsid >/dev/null 2>&1; then
    nohup setsid "${cmd[@]}" >"$LOG_FILE" 2>&1 &
  else
    nohup "${cmd[@]}" >"$LOG_FILE" 2>&1 &
  fi
  local pid=$!
  echo "$pid" > "$PID_FILE"
  echo "[INFO] PID=$pid，等待端点，最长900秒"

  set +e
  wait_endpoint 900
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] 服务未就绪，最后160行日志：" >&2
    tail -n 160 "$LOG_FILE" >&2 || true
    if kill -0 "$pid" 2>/dev/null; then kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true; fi
    rm -f "$PID_FILE"
    exit 2
  fi
  echo "[PASS] Omni共享服务启动完成。"
}

stop_server() {
  if ! is_running; then
    echo "[INFO] Omni服务未运行。"
    rm -f "$PID_FILE"
    exit 0
  fi
  local pid
  pid=$(cat "$PID_FILE")
  echo "[INFO] 停止Omni服务 PID=$pid"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "[WARN] 正常停止超时，发送KILL"
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "[PASS] Omni服务已停止。"
}

status_server() {
  echo "runtime_dir=$RUNTIME_DIR"
  echo "base_url=$OMNI_BASE_URL"
  echo "model=$OMNI_MODEL_PATH"
  echo "profile=$OMNI_PROFILE"
  echo "physical_gpus=$OMNI_VISIBLE_GPUS"
  echo "tp_size=$OMNI_TP_SIZE"
  if is_running; then
    local pid
    pid=$(cat "$PID_FILE")
    echo "status=RUNNING pid=$pid"
    ps -o user,pid,ppid,pgid,etime,%cpu,%mem,cmd -p "$pid" || true
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader || true
    "$OMNI_PYTHON" - "$OMNI_BASE_URL" <<'PY' || true
import json,sys,urllib.request
url=sys.argv[1].rstrip('/')+'/models'
req=urllib.request.Request(url,headers={'Authorization':'Bearer EMPTY'})
with urllib.request.urlopen(req,timeout=10) as r:
    print(json.dumps(json.loads(r.read()),ensure_ascii=False,indent=2))
PY
  else
    echo "status=STOPPED"
  fi
}

case "${1:-}" in
  start) start_server ;;
  stop) stop_server ;;
  restart) stop_server; start_server ;;
  status) status_server ;;
  log) tail -n "${2:-200}" "$LOG_FILE" ;;
  *) echo "用法: bash commands/manage_omni_tp4.sh start|stop|restart|status|log [行数]"; exit 2 ;;
esac
