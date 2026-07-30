#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}

if [[ ! -f "$SETTINGS" ]]; then
  echo "[FAIL] settings 文件不存在: $SETTINGS" >&2
  exit 2
fi

BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
if [[ -z "$BOOTSTRAP_PY" ]]; then
  echo "[FAIL] 找不到用于解析 runtime 的 Python。" >&2
  exit 2
fi

VLLM_PY=$(
  "$BOOTSTRAP_PY" "$ROOT/main_layer/resolve_runtime.py" \
    --settings "$SETTINGS" --workload vllm --field python
)
DATA_PY=$(
  "$BOOTSTRAP_PY" "$ROOT/main_layer/resolve_runtime.py" \
    --settings "$SETTINGS" --workload data --field python
)

if [[ ! -x "$VLLM_PY" ]]; then
  echo "[FAIL] vLLM Python 不可执行: $VLLM_PY" >&2
  exit 2
fi

ENV_INFO=$(
  "$VLLM_PY" - <<'PY'
import sys
try:
    import torch
    import vllm
except Exception as exc:
    print(f"ERROR|{sys.executable}|{type(exc).__name__}: {exc}")
    raise SystemExit(3)
print(f"OK|{sys.executable}|{vllm.__version__}|{torch.__version__}|{torch.cuda.device_count()}")
PY
) || {
  echo "[FAIL] 目标 vLLM Python 无法导入 torch/vllm: $VLLM_PY" >&2
  exit 2
}
IFS='|' read -r _ PY_PATH VLLM_VERSION TORCH_VERSION GPU_COUNT <<<"$ENV_INFO"
echo "[INFO] vLLM Python: $PY_PATH"
echo "[INFO] vLLM=$VLLM_VERSION torch=$TORCH_VERSION visible_gpus=$GPU_COUNT"

read_cfg() {
  "$DATA_PY" - "$SETTINGS" "$@" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    value = json.load(f)
for key in sys.argv[2:]:
    value = value[key]
if isinstance(value, bool):
    print(str(value).lower())
else:
    print(value)
PY
}

RESULTS=$(read_cfg paths pipeline_work_root)
LOG_DIR="$RESULTS/agents/logs"
PID_DIR="$RESULTS/agents/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

start_role() {
  local role=$1
  local model_path served_name host port gpu util max_model_len max_num_seqs is_multimodal
  model_path=$(read_cfg agents "$role" model_path)
  served_name=$(read_cfg agents "$role" served_name)
  host=$(read_cfg agents "$role" host)
  port=$(read_cfg agents "$role" port)
  gpu=$(read_cfg agents "$role" gpu)
  util=$(read_cfg agents "$role" gpu_memory_utilization)
  max_model_len=$(read_cfg agents "$role" max_model_len)
  max_num_seqs=$(read_cfg agents "$role" max_num_seqs)
  is_multimodal=$(read_cfg agents "$role" is_multimodal)

  if [[ ! -d "$model_path" || ! -f "$model_path/config.json" ]]; then
    echo "[FAIL] $role 模型目录无效或缺少 config.json: $model_path" >&2
    return 1
  fi
  if ! [[ "$gpu" =~ ^[0-9]+$ ]] || (( gpu >= GPU_COUNT )); then
    echo "[FAIL] $role 指定 GPU=$gpu，但当前 Python 仅看到 $GPU_COUNT 张卡。" >&2
    return 1
  fi

  local pid_file="$PID_DIR/$role.pid"
  if [[ -f "$pid_file" ]]; then
    local old_pid
    old_pid=$(cat "$pid_file" 2>/dev/null || true)
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "[INFO] $role 已运行 PID=$old_pid"
      return 0
    fi
    rm -f "$pid_file"
  fi

  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$port$"; then
    if curl -fsS --max-time 2 "http://$host:$port/v1/models" >/dev/null 2>&1; then
      echo "[INFO] $role 端口 $port 已有健康服务，按已启动处理。"
      return 0
    fi
    echo "[FAIL] $role 端口 $port 被非健康/未知服务占用。请检查: ss -ltnp | grep :$port" >&2
    return 1
  fi

  local -a cmd=(
    "$VLLM_PY" -m vllm.entrypoints.openai.api_server
    --model "$model_path"
    --served-model-name "$served_name"
    --host "$host"
    --port "$port"
    --gpu-memory-utilization "$util"
    --max-model-len "$max_model_len"
    --max-num-seqs "$max_num_seqs"
    --trust-remote-code
  )
  # 双A6000首次跑通：关闭CUDA Graph，降低启动显存。
  cmd+=(--enforce-eager)

  if [[ "$is_multimodal" == "true" ]]; then
    # Perceiver只使用图像，不使用视频。
    cmd+=(--limit-mm-per-prompt '{"image":2,"video":0}')
  else
    # Reasoner和Verifier为纯文本角色。
    # 禁止所有视觉输入，并跳过最大多模态输入的显存profiling。
    cmd+=(
      --limit-mm-per-prompt '{"image":0,"video":0}'
      --skip-mm-profiling
    )
  fi

  {
    printf '[ENV] CUDA_VISIBLE_DEVICES=%q VLLM_USE_FLASHINFER_SAMPLER=0\n' "$gpu"
    printf '[CMD] '
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } >"$LOG_DIR/$role.command.txt"

  CUDA_VISIBLE_DEVICES="$gpu" \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  nohup setsid "${cmd[@]}" >"$LOG_DIR/$role.log" 2>&1 &
  local pid=$!
  echo "$pid" >"$pid_file"
  echo "已提交 $role: GPU=$gpu port=$port PID=$pid"
  echo "日志: $LOG_DIR/$role.log"
}

wait_role() {
  local role=$1 host port pid
  host=$(read_cfg agents "$role" host)
  port=$(read_cfg agents "$role" port)
  pid=$(cat "$PID_DIR/$role.pid" 2>/dev/null || true)

  for _ in $(seq 1 180); do
    if curl -fsS "http://$host:$port/v1/models" >/dev/null 2>&1; then
      echo "[READY] $role http://$host:$port/v1"
      return 0
    fi
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "[FAIL] $role 进程已提前退出，PID=$pid" >&2
      echo "===== $role 最近日志 =====" >&2
      tail -n 200 "$LOG_DIR/$role.log" >&2 || true
      return 1
    fi
    sleep 2
  done

  echo "[FAIL] $role 等待 360 秒仍未就绪" >&2
  tail -n 200 "$LOG_DIR/$role.log" >&2 || true
  return 1
}

# 官方核心是 Reasoner↔Perceiver 循环，Verifier 为可选终局验收角色。
# 双 A6000 默认部署：GPU0 Reasoner+Verifier，GPU1 Perceiver。
start_role reasoner
wait_role reasoner
start_role verifier
wait_role verifier
start_role perceiver
wait_role perceiver

"$DATA_PY" "$ROOT/data_layer/agents/check_three_agents.py" --settings "$SETTINGS"
nvidia-smi
