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
print(resolve_python(s,workload)); print(m['path']); print(m['served_name']); print(s['paths']['test_run_root']); print(ev.get('gpu_memory_utilization',.86)); print(workload); print(m.get('architecture',''))
PY
)
PYTHON_BIN=${CFG[0]}; MODEL_PATH=${CFG[1]}; NAME=${CFG[2]}; WORK=${CFG[3]}; UTIL=${CFG[4]}; WORKLOAD=${CFG[5]}; ARCH=${CFG[6]}
LOG_DIR="$WORK/servers"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${KEY}_${PORT}.log"; PIDFILE="$LOG_DIR/${KEY}_${PORT}.pid"; CMDFILE="$LOG_DIR/${KEY}_${PORT}.command.txt"
[[ -x "$PYTHON_BIN" ]] || { echo "[FAIL] vLLM Python invalid: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$MODEL_PATH/config.json" ]] || { echo "[FAIL] model config missing: $MODEL_PATH/config.json" >&2; exit 2; }
"$PYTHON_BIN" -c 'import vllm,torch,transformers; assert torch.cuda.is_available(); print(f"vLLM={vllm.__version__} transformers={transformers.__version__} torch={torch.__version__}")' || { echo "[FAIL] vLLM environment invalid: workload=$WORKLOAD python=$PYTHON_BIN" >&2; exit 2; }
echo "[INFO] model=$KEY architecture=$ARCH workload=$WORKLOAD python=$PYTHON_BIN"
"$PYTHON_BIN" - "$ARCH" <<'PY'
import sys
from packaging.version import Version
arch=sys.argv[1]
if arch == "qwen3_vl":
    import transformers, vllm
    problems=[]
    if Version(transformers.__version__.split("+")[0]) < Version("4.57.0"):
        problems.append(f"transformers={transformers.__version__} < 4.57.0")
    if Version(vllm.__version__.split("+")[0]) < Version("0.11.0"):
        problems.append(f"vllm={vllm.__version__} < 0.11.0")
    if problems:
        raise SystemExit(
            "Qwen3-VL runtime is incompatible: " + ", ".join(problems) +
            ". Use the separate qwen3_vllm workload; do not upgrade the working Qwen2.5 environment in place."
        )
PY
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$PORT$"; then echo "[FAIL] port occupied: $PORT" >&2; exit 2; fi
if [[ -f "$PIDFILE" ]]; then old=$(cat "$PIDFILE" || true); kill -0 "$old" 2>/dev/null && { echo "[FAIL] pid already alive: $old" >&2; exit 2; }; rm -f "$PIDFILE"; fi

CMD=("$PYTHON_BIN" -m vllm.entrypoints.openai.api_server --model "$MODEL_PATH" --served-model-name "$NAME" --host 127.0.0.1 --port "$PORT" --gpu-memory-utilization "$UTIL" --max-model-len 12288 --limit-mm-per-prompt '{"image":2}' --trust-remote-code)
printf 'CUDA_VISIBLE_DEVICES=%q VLLM_USE_FLASHINFER_SAMPLER=0 ' "$GPU" > "$CMDFILE"; printf '%q ' "${CMD[@]}" >> "$CMDFILE"; printf '\n' >> "$CMDFILE"
CUDA_VISIBLE_DEVICES="$GPU" VLLM_USE_FLASHINFER_SAMPLER=0 nohup setsid "${CMD[@]}" >"$LOG" 2>&1 &
pid=$!; echo "$pid" > "$PIDFILE"
echo "[INFO] submitted PID=$pid, log=$LOG"
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/tmp/re_scripts_eval_models.json 2>/dev/null; then
    echo "[READY] URL=http://127.0.0.1:$PORT/v1 model=$NAME"; cat /tmp/re_scripts_eval_models.json; exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then echo "[FAIL] eval server exited" >&2; tail -n 180 "$LOG" >&2 || true; exit 3; fi
  sleep 2
done
echo "[FAIL] eval server not ready after 360 seconds" >&2; tail -n 180 "$LOG" >&2 || true; exit 4
