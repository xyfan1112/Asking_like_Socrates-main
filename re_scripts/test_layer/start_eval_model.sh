#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
KEY=${2:?model key required}
GPU=${3:-1}
PORT=${4:-8010}

BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
[[ -n "$BOOTSTRAP_PY" ]] || { echo "[FAIL] 找不到 Python" >&2; exit 2; }

readarray -t CFG < <("$BOOTSTRAP_PY" - "$SETTINGS" "$KEY" "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[3]); s=json.load(open(sys.argv[1],encoding='utf-8')); key=sys.argv[2]
sys.path.insert(0,str(root/'main_layer'))
from config import resolve_model_workload, resolve_python
m=s['models'][key]; ev=s['runtime']['gpu_profiles']['eval_single_gpu']
workload=resolve_model_workload(s,key,'vllm')
print(resolve_python(s,workload))
print(m['path'])
print(m['served_name'])
print(s['paths']['test_run_root'])
print(ev.get('gpu_memory_utilization',.80))
print(workload)
print(m.get('architecture',''))
print(ev.get('max_model_len',4096))
print(m.get('adapter_path',''))
print(m.get('base_served_name',m['served_name']))
print(m.get('adapter_name',m['served_name']))
PY
)
PYTHON_BIN=${CFG[0]}; MODEL_PATH=${CFG[1]}; NAME=${CFG[2]}; WORK=${CFG[3]}; UTIL=${CFG[4]}; WORKLOAD=${CFG[5]}; ARCH=${CFG[6]}; MAX_MODEL_LEN=${CFG[7]}; ADAPTER_PATH=${CFG[8]}; BASE_NAME=${CFG[9]}; ADAPTER_NAME=${CFG[10]}
LOG_DIR="$WORK/servers"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${KEY}_${PORT}.log"; PIDFILE="$LOG_DIR/${KEY}_${PORT}.pid"; CMDFILE="$LOG_DIR/${KEY}_${PORT}.command.txt"
[[ -x "$PYTHON_BIN" ]] || { echo "[FAIL] vLLM Python invalid: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$MODEL_PATH/config.json" ]] || { echo "[FAIL] model config missing: $MODEL_PATH/config.json" >&2; exit 2; }
"$PYTHON_BIN" -c 'import vllm,torch,transformers; assert torch.cuda.is_available(); print(f"vLLM={vllm.__version__} transformers={transformers.__version__} torch={torch.__version__}")' || { echo "[FAIL] vLLM environment invalid: workload=$WORKLOAD python=$PYTHON_BIN" >&2; exit 2; }
echo "[INFO] key=$KEY served=$NAME architecture=$ARCH workload=$WORKLOAD python=$PYTHON_BIN model=$MODEL_PATH adapter=${ADAPTER_PATH:-none}"
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$PORT$"; then echo "[FAIL] port occupied: $PORT" >&2; exit 2; fi
if [[ -f "$PIDFILE" ]]; then old=$(cat "$PIDFILE" || true); kill -0 "$old" 2>/dev/null && { echo "[FAIL] pid already alive: $old" >&2; exit 2; }; rm -f "$PIDFILE"; fi

CMD=("$PYTHON_BIN" -m vllm.entrypoints.cli.main serve "$MODEL_PATH" --served-model-name "$BASE_NAME" --host 127.0.0.1 --port "$PORT" --gpu-memory-utilization "$UTIL" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 1 --limit-mm-per-prompt '{"image":2}' --trust-remote-code --enforce-eager)
if [[ -n "$ADAPTER_PATH" ]]; then
  [[ -f "$ADAPTER_PATH/adapter_config.json" ]] || { echo "[FAIL] adapter_config missing: $ADAPTER_PATH/adapter_config.json" >&2; exit 2; }
  CMD+=(--enable-lora --max-loras 1 --max-lora-rank 64 --lora-modules "${ADAPTER_NAME}=${ADAPTER_PATH}")
fi
printf 'CUDA_VISIBLE_DEVICES=%q VLLM_USE_FLASHINFER_SAMPLER=0 ' "$GPU" > "$CMDFILE"; printf '%q ' "${CMD[@]}" >> "$CMDFILE"; printf '\n' >> "$CMDFILE"
CUDA_VISIBLE_DEVICES="$GPU" VLLM_USE_FLASHINFER_SAMPLER=0 nohup setsid "${CMD[@]}" >"$LOG" 2>&1 &
pid=$!; echo "$pid" > "$PIDFILE"
echo "[INFO] submitted PID=$pid, log=$LOG"
for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/tmp/re_scripts_eval_models_${PORT}.json 2>/dev/null; then
    if "$BOOTSTRAP_PY" - /tmp/re_scripts_eval_models_${PORT}.json "$NAME" <<'PY'
import json,sys
ids={str(x.get('id')) for x in json.load(open(sys.argv[1],encoding='utf-8')).get('data',[])}
raise SystemExit(0 if sys.argv[2] in ids else 1)
PY
    then
      echo "[READY] URL=http://127.0.0.1:$PORT/v1 model=$NAME"; cat /tmp/re_scripts_eval_models_${PORT}.json; exit 0
    fi
  fi
  if ! kill -0 "$pid" 2>/dev/null; then echo "[FAIL] eval server exited" >&2; tail -n 200 "$LOG" >&2 || true; exit 3; fi
  sleep 2
done
echo "[FAIL] eval server/model not ready after 480 seconds" >&2; tail -n 200 "$LOG" >&2 || true; exit 4
