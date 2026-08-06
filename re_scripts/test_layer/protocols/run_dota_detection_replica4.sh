#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}; KEY=${2:?model key}; URLS_CSV=${3:?comma-separated URLs}; SPLIT=${4:-val}; RUN_POLICY=${5:-fresh}
BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
[[ -n "$BOOTSTRAP_PY" ]] || { echo "[FAIL] 找不到 Python" >&2; exit 2; }
DATA_PY=$("$BOOTSTRAP_PY" "$ROOT/main_layer/resolve_runtime.py" --settings "$SETTINGS" --workload data --field python)
IFS=',' read -r -a URLS <<< "$URLS_CSV"
N=${#URLS[@]}; (( N > 0 )) || { echo "[FAIL] empty URL list" >&2; exit 2; }
FINAL=$("$DATA_PY" - "$SETTINGS" "$KEY" "$SPLIT" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
print(f"{s['paths']['test_run_root']}/dota_detection/{sys.argv[2]}/{sys.argv[3]}_dota_class_conditioned_roi_v2_k1.jsonl")
PY
)
mkdir -p "$(dirname "$FINAL")"
[[ "$RUN_POLICY" == "fresh" ]] && { rm -f "$FINAL" "${FINAL%.jsonl}.manifest.json"; rm -f "${FINAL%.jsonl}".shard*-of-*.jsonl*; }
PIDS=(); SHARDS=()
for ((i=0;i<N;i++)); do
  shard="${FINAL%.jsonl}.shard$(printf '%02d' "$i")-of-$(printf '%02d' "$N").jsonl"
  SHARDS+=("$shard")
  extra=(); [[ "$RUN_POLICY" == "fresh" ]] && extra+=(--fresh)
  "$DATA_PY" "$ROOT/test_layer/01_raw_dota_obb_infer.py" \
    --settings "$SETTINGS" --model-key "$KEY" --base-url "${URLS[$i]}" \
    --split "$SPLIT" --protocol dota_class_conditioned \
    --num-shards "$N" --shard-index "$i" --output "$shard" "${extra[@]}" \
    >"${shard%.jsonl}.log" 2>&1 &
  PIDS+=("$!")
  echo "[REPLICA4] dota_detection shard=$i/$N url=${URLS[$i]} pid=${PIDS[-1]}"
done
failed=0
for ((i=0;i<N;i++)); do
  if ! wait "${PIDS[$i]}"; then
    echo "[FAIL] dota_detection shard $i failed; tail follows" >&2
    tail -n 120 "${SHARDS[$i]%.jsonl}.log" >&2 || true
    failed=1
  fi
done
(( failed == 0 )) || exit 3
"$DATA_PY" "$ROOT/tools/merge_jsonl_shards.py" --output "$FINAL" --shards "${SHARDS[@]}"
"$DATA_PY" "$ROOT/test_layer/02_raw_dota_obb_metrics.py" --settings "$SETTINGS" --pred "$FINAL"
