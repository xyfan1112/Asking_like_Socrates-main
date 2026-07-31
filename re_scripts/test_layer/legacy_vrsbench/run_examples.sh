#!/usr/bin/env bash
set -euo pipefail

# 统一路径。路径不同，只修改这里。
REPO_ROOT="/home/yk/桌面/Asking_like_Socrates"
EVAL_DIR="${REPO_ROOT}/eval_vrsbench"
DATA_ROOT="/home/yk/fxy/datasets/VRSBench"
IMAGE_DIR="${DATA_ROOT}/Images_val"
JSON_PATH="${DATA_ROOT}/VRSBench_EVAL_vqa.json"
MODEL_DIR="/home/yk/fxy/models/RS-EoT-7B"
RESULT_DIR="/home/yk/fxy/results/VRSBench_RS-EoT-7B"

mkdir -p "${RESULT_DIR}"

echo "1) 检查数据"
python "${EVAL_DIR}/01_check_vrsbench.py" \
  --json "${JSON_PATH}" \
  --image-dir "${IMAGE_DIR}"

echo
echo "2) 10 题流水线测试"
CUDA_VISIBLE_DEVICES=0 python "${EVAL_DIR}/02_infer_vrsbench.py" \
  --model "${MODEL_DIR}" \
  --json "${JSON_PATH}" \
  --image-dir "${IMAGE_DIR}" \
  --output "${RESULT_DIR}/debug_10_k1.jsonl" \
  --num-samples 1 \
  --max-questions 10 \
  --max-new-tokens 512 \
  --greedy

echo
echo "3) 计算 10 题本地分数"
python "${EVAL_DIR}/03_score_local.py" \
  --pred "${RESULT_DIR}/debug_10_k1.jsonl" \
  --k 1 \
  --save-summary "${RESULT_DIR}/debug_10_k1_summary.json"

echo
echo "调试流程完成。"
