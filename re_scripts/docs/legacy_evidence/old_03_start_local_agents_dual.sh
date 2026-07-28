#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/nhl/Asking_like_Socrates"
SCRIPT_DIR="$ROOT/fined_scripts"
LOG_DIR="/home/nhl/fxy/results/local_agents"
PID_DIR="$LOG_DIR/pids"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate als_vllm

mkdir -p "$LOG_DIR" "$PID_DIR"

cat > "$ROOT/fined_scripts/local_agent_runtime.env" <<'EOF'
export LOCAL_REASONER_BASE_URL=http://127.0.0.1:8001/v1
export LOCAL_REASONER_MODEL=local-reasoner
export LOCAL_PERCEIVER_BASE_URL=http://127.0.0.1:8002/v1
export LOCAL_PERCEIVER_MODEL=local-perceiver
EOF

REASONER="/home/nhl/fxy/models/Qwen2.5-7B-Instruct-AWQ"
PERCEIVER="/home/nhl/fxy/models/Qwen2.5-VL-7B-Instruct-AWQ"

if pgrep -f "vllm serve.*local-reasoner" >/dev/null 2>&1; then
  echo "local-reasoner 已经运行。"
else
  CUDA_VISIBLE_DEVICES=0 nohup vllm serve "$REASONER" \
    --served-model-name local-reasoner \
    --host 127.0.0.1 \
    --port 8001 \
    --dtype auto \
    --gpu-memory-utilization 0.26 \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --enable-prefix-caching \
    > "$LOG_DIR/reasoner.log" 2>&1 &
  echo $! > "$PID_DIR/reasoner.pid"
fi

echo "等待 Reasoner..."
for i in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
    echo "Reasoner 就绪。"
    break
  fi
  sleep 2
  if [ "$i" -eq 120 ]; then
    tail -n 100 "$LOG_DIR/reasoner.log"
    exit 1
  fi
done

if pgrep -f "vllm serve.*local-perceiver" >/dev/null 2>&1; then
  echo "local-perceiver 已经运行。"
else
  CUDA_VISIBLE_DEVICES=0 nohup vllm serve "$PERCEIVER" \
    --served-model-name local-perceiver \
    --host 127.0.0.1 \
    --port 8002 \
    --dtype auto \
    --gpu-memory-utilization 0.50 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --limit-mm-per-prompt '{"image":1}' \
    --enable-prefix-caching \
    > "$LOG_DIR/perceiver.log" 2>&1 &
  echo $! > "$PID_DIR/perceiver.pid"
fi

echo "等待 Perceiver..."
for i in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8002/v1/models >/dev/null 2>&1; then
    echo "Perceiver 就绪。"
    break
  fi
  sleep 2
  if [ "$i" -eq 180 ]; then
    tail -n 100 "$LOG_DIR/perceiver.log"
    exit 1
  fi
done

echo "双本地智能体已经启动。"
nvidia-smi
