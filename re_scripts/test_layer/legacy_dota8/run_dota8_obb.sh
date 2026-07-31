#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="als_vllm"
SCRIPT_DIR="/home/yk/Asking_like_Socrates/eval_vrsbench"
RESULT_DIR="/home/yk/fxy/dota8_vlm_obb_results"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$RESULT_DIR"

echo "============================================================"
echo "1/2 DOTA8 批量 OBB 推理"
echo "============================================================"

python "$SCRIPT_DIR/07_dota8_batch_obb.py" \
  2>&1 | tee "$RESULT_DIR/inference.log"

echo
echo "============================================================"
echo "2/2 计算 OBB 指标"
echo "============================================================"

python "$SCRIPT_DIR/08_eval_dota8_obb.py" \
  2>&1 | tee "$RESULT_DIR/evaluation.log"

echo
echo "全部完成。"
echo "结果目录：$RESULT_DIR"
