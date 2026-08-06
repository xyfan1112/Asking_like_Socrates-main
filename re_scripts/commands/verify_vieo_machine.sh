#!/usr/bin/env bash
# 只读检查目标机路径、环境、数据、模型和可用GPU；不启动服务。
set -Eeuo pipefail
WORKSPACE="/home/vieo/vieo/fxy_workspace"
PROJECT_ROOT="$WORKSPACE/Asking_like_Socrates-main"
FXY_ROOT="$WORKSPACE/fxy"
SFT_PY="/home/vieo/anaconda3/envs/als_sft/bin/python"
VLLM_PY="/home/vieo/anaconda3/envs/als_vllm/bin/python"
FAIL=0
check_file(){ [[ -f "$1" ]] && echo "[PASS] file $1" || { echo "[FAIL] file $1"; FAIL=1; }; }
check_dir(){ [[ -d "$1" ]] && echo "[PASS] dir  $1" || { echo "[FAIL] dir  $1"; FAIL=1; }; }
check_exec(){ [[ -x "$1" ]] && echo "[PASS] exec $1" || { echo "[FAIL] exec $1"; FAIL=1; }; }

check_dir "$PROJECT_ROOT/re_scripts"
check_dir "$FXY_ROOT/datasets/yw128"
check_dir "$FXY_ROOT/datasets/yw128_scene_disjoint"
check_file "$FXY_ROOT/datasets/yw128/classes.txt"
check_file "$FXY_ROOT/datasets/yw128_scene_disjoint/scene_split_manifest.json"
check_dir "$FXY_ROOT/models/Qwen3-Omni-30B-A3B-Instruct-AWQ-No-TTS"
check_file "$FXY_ROOT/models/Qwen3-Omni-30B-A3B-Instruct-AWQ-No-TTS/config.json"
check_exec "$SFT_PY"
check_exec "$VLLM_PY"

if [[ -x "$SFT_PY" ]]; then
  "$SFT_PY" -c 'import sys,torch,transformers,accelerate,peft,llamafactory; print("[PASS] als_sft",sys.executable,torch.__version__,transformers.__version__,llamafactory.__file__)' || FAIL=1
fi
if [[ -x "$VLLM_PY" ]]; then
  "$VLLM_PY" -c 'import sys,torch,vllm,transformers; print("[PASS] als_vllm",sys.executable,torch.__version__,vllm.__version__,transformers.__version__)' || FAIL=1
fi

echo "===== GPU 0,1,3 ====="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv || FAIL=1

echo "===== 当前GPU计算进程及所有者 ====="
for PID in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sort -u); do
  ps -o user,pid,ppid,etime,cmd -p "$PID" || true
done

echo "===== 旧绝对路径检查（活动源码/基础配置） ====="
if grep -RInE '/home/(yk|nhl)/' \
    "$PROJECT_ROOT/re_scripts/commands" \
    "$PROJECT_ROOT/re_scripts/main_layer" \
    "$PROJECT_ROOT/re_scripts/train_layer" \
    "$PROJECT_ROOT/re_scripts/tools" \
    "$PROJECT_ROOT/re_scripts/settings.json" 2>/dev/null; then
  echo "[WARN] 上面仍有旧路径；legacy文档不计，但活动源码/settings需要处理。"
  FAIL=1
else
  echo "[PASS] 活动源码和settings未发现/home/yk或/home/nhl"
fi

if (( FAIL != 0 )); then
  echo "[FAIL] vieo机器检查未通过"
  exit 2
fi
echo "[PASS] vieo机器只读检查通过"
