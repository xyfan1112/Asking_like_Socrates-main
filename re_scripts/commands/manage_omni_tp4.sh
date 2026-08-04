#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/commands/user_config.sh"

RUNTIME_DIR=${OMNI_RUNTIME_DIR:-"$ZH_OUTPUT_ROOT/omni_tp4_service"}
PID_FILE="$RUNTIME_DIR/server.pid"
CMD_FILE="$RUNTIME_DIR/server.command.txt"
LOG_FILE="$RUNTIME_DIR/server.log"
mkdir -p "$RUNTIME_DIR"

fail() { echo "[FAIL] $*" >&2; exit 2; }

resolve_vllm() {
  [[ -x "${OMNI_PYTHON:-}" ]] || fail "OMNI_PYTHON不可执行: ${OMNI_PYTHON:-空}"
  local bin
  bin="$(dirname "$OMNI_PYTHON")/vllm"
  [[ -x "$bin" ]] || fail "找不到vllm命令: $bin"
  printf '%s\n' "$bin"
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
            print('[PASS] Omni TP4 endpoint ready:', ids)
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
  [[ -d "$OMNI_MODEL_PATH" ]] || fail "模型目录不存在: $OMNI_MODEL_PATH"
  [[ "${OMNI_TP_SIZE:-}" == "4" ]] || fail "v1.2.2固定要求OMNI_TP_SIZE=4，当前=${OMNI_TP_SIZE:-空}"
  if is_running; then
    echo "[INFO] Omni服务已运行，PID=$(cat "$PID_FILE")"
    exit 0
  fi
  rm -f "$PID_FILE"

  local vllm_bin help_text profile util seqs eager
  vllm_bin=$(resolve_vllm)
  help_text=$($vllm_bin serve --help 2>&1 || true)
  [[ "$help_text" == *"--tensor-parallel-size"* ]] || fail "当前vLLM不支持--tensor-parallel-size"

  profile=${OMNI_PROFILE:-balanced}
  case "$profile" in
    safe)
      util=${OMNI_SAFE_GPU_MEMORY_UTILIZATION:-0.88}
      seqs=${OMNI_SAFE_MAX_NUM_SEQS:-2}
      eager=1
      ;;
    balanced)
      util=${OMNI_GPU_MEMORY_UTILIZATION:-0.92}
      seqs=${OMNI_MAX_NUM_SEQS:-4}
      eager=0
      ;;
    fast)
      util=${OMNI_FAST_GPU_MEMORY_UTILIZATION:-0.94}
      seqs=${OMNI_FAST_MAX_NUM_SEQS:-8}
      eager=0
      ;;
    *) fail "OMNI_PROFILE必须是safe/balanced/fast，当前=$profile" ;;
  esac

  local -a cmd
  cmd=(
    "$vllm_bin" serve "$OMNI_MODEL_PATH"
    --served-model-name "$OMNI_SERVED_NAME"
    --host "$OMNI_HOST" --port "$OMNI_PORT"
    --tensor-parallel-size 4
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

  echo "[INFO] 启动一个Qwen3-Omni AWQ-No-TTS服务"
  echo "[INFO] TP=4，每个请求由四张GPU共同计算，不启动三份权重"
  echo "[INFO] profile=$profile max_num_seqs=$seqs gpu_memory_utilization=$util"
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
    echo "[FAIL] 服务未就绪，最后120行日志：" >&2
    tail -n 120 "$LOG_FILE" >&2 || true
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; fi
    rm -f "$PID_FILE"
    exit 2
  fi
  echo "[PASS] Omni TP4服务启动完成。"
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
  echo "tp_size=$OMNI_TP_SIZE"
  if is_running; then
    local pid
    pid=$(cat "$PID_FILE")
    echo "status=RUNNING pid=$pid"
    ps -o pid,ppid,etime,%cpu,%mem,cmd -p "$pid" || true
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader || true
    "$OMNI_PYTHON" - "$OMNI_BASE_URL" <<'PY' || true
import json,sys,urllib.request
url=sys.argv[1].rstrip('/')+'/models'
req=urllib.request.Request(url,headers={'Authorization':'Bearer EMPTY'})
with urllib.request.urlopen(req,timeout=10) as r: print(json.dumps(json.loads(r.read()),ensure_ascii=False,indent=2))
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
