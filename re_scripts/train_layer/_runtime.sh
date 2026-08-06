#!/usr/bin/env bash
# Shared runtime resolver for LLaMA-Factory commands.
set -euo pipefail

resolve_sft_runtime() {
  local root=$1 settings=$2
  local bootstrap_py=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
  [[ -n "$bootstrap_py" ]] || { echo "[FAIL] 找不到 Python" >&2; return 2; }
  readarray -t _SFT_CFG < <("$bootstrap_py" - "$settings" "$root" <<'PY'
import json,sys
from pathlib import Path
s=json.load(open(sys.argv[1],encoding='utf-8')); root=Path(sys.argv[2])
sys.path.insert(0,str(root/'main_layer'))
from config import resolve_llama_factory_dir, resolve_model_workload, resolve_python, resolve_training_template
key=s['training']['base_model_key']
workload=resolve_model_workload(s,key,'sft')
print(resolve_python(s,workload))
print(resolve_llama_factory_dir(s,key))
print(workload)
print(resolve_training_template(s,key))
PY
)
  SFT_PYTHON=${_SFT_CFG[0]}
  LLAMA_FACTORY_DIR=${_SFT_CFG[1]}
  SFT_WORKLOAD=${_SFT_CFG[2]}
  SFT_TEMPLATE=${_SFT_CFG[3]}
  SFT_BIN_DIR=$(dirname "$SFT_PYTHON")
  LLAMAFACTORY_CLI="$SFT_BIN_DIR/llamafactory-cli"
  LLAMAFACTORY_CLI_MODE=""
  if [[ ! -x "$SFT_PYTHON" ]]; then
    echo "[FAIL] SFT Python invalid: $SFT_PYTHON" >&2; return 2
  fi
  if ! "$SFT_PYTHON" - <<'PYENV'
import llamafactory
import torch
import transformers
print(
    "llamafactory={} transformers={} torch={}".format(
        getattr(llamafactory, "__version__", "unknown"),
        transformers.__version__,
        torch.__version__,
    )
)
PYENV
  then
    echo "[FAIL] SFT environment cannot import llamafactory/torch/transformers: $SFT_PYTHON" >&2
    echo "[HINT] Do not trust the activated environment name. Test the exact interpreter:" >&2
    echo "       $SFT_PYTHON -c 'import llamafactory,torch,transformers'" >&2
    echo "       $SFT_PYTHON -m pip list" >&2
    echo "[HINT] If bin/pip says required file not found after migration, its shebang still points to the old host path; rebuild with conda-pack/conda-unpack." >&2
    return 2
  fi
  # Prefer the exact interpreter over console-script shebangs. This remains
  # valid after conda-pack/conda-unpack even when bin/llamafactory-cli or pip
  # still contains the source host path.
  if "$SFT_PYTHON" -c 'import llamafactory.cli' >/dev/null 2>&1; then
    LLAMAFACTORY_CLI_MODE="python_module"
  elif [[ -x "$LLAMAFACTORY_CLI" ]]; then
    LLAMAFACTORY_CLI_MODE="console_script"
  else
    echo "[FAIL] Neither python -m llamafactory.cli nor $LLAMAFACTORY_CLI is available." >&2
    return 2
  fi
  CUDA_DEVICES=$("$SFT_PYTHON" - "$settings" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
profiles=s.get('runtime',{}).get('gpu_profiles',{})
p=profiles.get('sft_four_a6000') or profiles.get('sft_dual_a6000') or {}
print(p.get('visible_devices','0,1,2,3'))
PY
)
  WORLD_SIZE=$("$SFT_PYTHON" - "$settings" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
profiles=s.get('runtime',{}).get('gpu_profiles',{})
p=profiles.get('sft_four_a6000') or profiles.get('sft_dual_a6000') or {}
print(p.get('world_size',s['training'].get('world_size',4)))
PY
)
  echo "[INFO] SFT workload: $SFT_WORKLOAD"
  echo "[INFO] SFT Python: $SFT_PYTHON"
  echo "[INFO] LLaMA-Factory dir: $LLAMA_FACTORY_DIR"
  echo "[INFO] LLaMA-Factory CLI mode: $LLAMAFACTORY_CLI_MODE (console_script=$LLAMAFACTORY_CLI)"
  echo "[INFO] template: $SFT_TEMPLATE"
  local visible_count
  visible_count=$(awk -F, '{print NF}' <<<"$CUDA_DEVICES")
  if [[ "$visible_count" -ne "$WORLD_SIZE" ]]; then
    echo "[FAIL] visible GPU count=$visible_count but world_size=$WORLD_SIZE" >&2
    return 2
  fi
  echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_DEVICES; world_size=$WORLD_SIZE"
}

llamafactory_cli() {
  if [[ "${LLAMAFACTORY_CLI_MODE:-}" == "python_module" ]]; then
    "$SFT_PYTHON" -m llamafactory.cli "$@"
  else
    "$LLAMAFACTORY_CLI" "$@"
  fi
}

ensure_agents_stopped_for_sft() {
  local settings=$1
  local require auto_stop
  require=$($SFT_PYTHON - "$settings" <<'PYCFG'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
print(str(s.get('training',{}).get('require_agents_stopped',True)).lower())
PYCFG
)
  [[ "$require" == "true" ]] || return 0
  auto_stop=$($SFT_PYTHON - "$settings" <<'PYCFG'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
print(str(s.get('training',{}).get('auto_stop_agents_before_sft',True)).lower())
PYCFG
)

  local busy=0
  for port in 8001 8002 8003; do
    if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 1 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      busy=1
    fi
  done
  [[ $busy -eq 0 ]] && return 0

  if [[ "$auto_stop" == "true" ]]; then
    echo "[AUTO] 检测到三智能体服务，训练前自动停止。"
    bash "$ROOT/data_layer/agents/stop_three_agents.sh" "$settings"
  else
    echo "[FAIL] Agent services are still running on 8001/8002/8003." >&2
    echo "[HINT] Run: python main_layer/run.py stop-agents --settings $settings" >&2
    return 2
  fi

  for port in 8001 8002 8003; do
    if command -v curl >/dev/null 2>&1 && curl -fsS --max-time 1 "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "[FAIL] Agent service is still running on port ${port}." >&2
      return 2
    fi
  done
  echo "[GPU PRECHECK] agent ports are free"
}
