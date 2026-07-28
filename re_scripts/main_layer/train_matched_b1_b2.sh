#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/root/Asking_like_Socrates-main/re_scripts"
PY="/root/miniconda3/bin/python"

CFG="${1:-$ROOT/settings.json}"

DATA_DIR="/root/autodl-tmp/fxy/datasets4.3.3/dota128-llamafactory4.3.3"
RUN_DIR="/root/autodl-tmp/fxy/results4.3.3/dota128_training4.3.3"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$RUN_DIR/logs"
LOG_FILE="$LOG_DIR/train_matched_b1_b2_${STAMP}.log"
SNAPSHOT_FILE="$RUN_DIR/dataset_snapshot_${STAMP}.sha256"

mkdir -p "$LOG_DIR"
cd "$ROOT"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================"
echo "[START] Matched B1/B2 training"
echo "time     = $(date '+%F %T')"
echo "settings = $CFG"
echo "log      = $LOG_FILE"
echo "============================================================"

if [[ ! -f "$CFG" ]]; then
  echo "[FAIL] settings不存在：$CFG"
  exit 2
fi

echo
echo "[1/7] 停止三个轨迹生成Agent"

"$PY" main_layer/run.py stop-agents \
  --settings "$CFG" || true

echo
echo "[2/7] 验证训练数据"

"$PY" main_layer/run.py validate-train \
  --settings "$CFG"

echo
echo "[3/7] 保存数据快照"

for file in \
  dota128_b1_matched_train_official.json \
  dota128_b2_matched_train_official.json \
  dota128_ref_val_official.json
do
  if [[ ! -f "$DATA_DIR/$file" ]]; then
    echo "[FAIL] 缺少训练文件：$DATA_DIR/$file"
    exit 2
  fi
done

sha256sum \
  "$DATA_DIR/dota128_b1_matched_train_official.json" \
  "$DATA_DIR/dota128_b2_matched_train_official.json" \
  "$DATA_DIR/dota128_ref_val_official.json" \
  > "$SNAPSHOT_FILE"

cat "$SNAPSHOT_FILE"

echo
echo "[4/7] 重新生成B1/B2训练YAML"

"$PY" main_layer/run.py generate-train-config \
  --settings "$CFG"

echo
echo "[5/7] 验证LLaMA-Factory配置合同"

"$PY" main_layer/run.py validate-lf-contract \
  --settings "$CFG" \
  --target all

echo
echo "[6/7] 检查现有Adapter目录"

mapfile -t ADAPTER_DIRS < <(
  "$PY" - "$CFG" <<'PY'
import json
import sys

settings = json.load(open(sys.argv[1], encoding="utf-8"))

print(settings["training"]["b1_adapter_output"])
print(settings["training"]["b2_adapter_output"])
PY
)

B1_ADAPTER="${ADAPTER_DIRS[0]}"
B2_ADAPTER="${ADAPTER_DIRS[1]}"

adapter_complete() {
  local path="$1"

  [[ -f "$path/adapter_config.json" ]] &&
  [[ -s "$path/adapter_model.safetensors" ]]
}

prepare_adapter_dir() {
  local name="$1"
  local path="$2"

  if adapter_complete "$path"; then
    echo "[SKIP] $name Adapter已经完整：$path"
    return 10
  fi

  if [[ -d "$path" ]]; then
    local backup="${path}.incomplete_backup_${STAMP}"
    echo "[WARN] 发现不完整的$name目录"
    echo "[MOVE] $path -> $backup"
    mv "$path" "$backup"
  fi

  return 0
}

TRAIN_B1=1
TRAIN_B2=1

if prepare_adapter_dir "B1" "$B1_ADAPTER"; then
  TRAIN_B1=1
else
  status=$?
  if [[ "$status" -eq 10 ]]; then
    TRAIN_B1=0
  else
    exit "$status"
  fi
fi

if prepare_adapter_dir "B2" "$B2_ADAPTER"; then
  TRAIN_B2=1
else
  status=$?
  if [[ "$status" -eq 10 ]]; then
    TRAIN_B2=0
  else
    exit "$status"
  fi
fi

echo
echo "[7/7] 开始训练"

if [[ "$TRAIN_B1" -eq 1 ]]; then
  echo
  echo "------------------------------------------------------------"
  echo "[TRAIN] B1 Direct + final-only Ref"
  echo "------------------------------------------------------------"

  "$PY" main_layer/run.py train-b1 \
    --settings "$CFG"
else
  echo "[SKIP] B1已有完整Adapter"
fi

if [[ "$TRAIN_B2" -eq 1 ]]; then
  echo
  echo "------------------------------------------------------------"
  echo "[TRAIN] B2 Direct + Socratic Ref"
  echo "------------------------------------------------------------"

  "$PY" main_layer/run.py train-b2 \
    --settings "$CFG"
else
  echo "[SKIP] B2已有完整Adapter"
fi

echo
echo "============================================================"
echo "[PASS] B1/B2训练阶段完成"
echo "B1 Adapter = $B1_ADAPTER"
echo "B2 Adapter = $B2_ADAPTER"
echo "snapshot   = $SNAPSHOT_FILE"
echo "log        = $LOG_FILE"
echo "============================================================"

echo
echo "[IMPORTANT]"
echo "由于数据盘容量有限，本脚本不会自动执行 merge both。"
echo "请按 B1评测 -> 删除B1 merged -> B2评测 的顺序合并。"