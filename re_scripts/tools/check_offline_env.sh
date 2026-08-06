#!/usr/bin/env bash
set -uo pipefail
ENV_PY=${1:?usage: check_offline_env.sh /path/to/env/bin/python [llama_factory_dir]}
LF_DIR=${2:-}
rc=0
echo "[CHECK] python=$ENV_PY"
if [[ ! -x "$ENV_PY" ]]; then echo "[FAIL] interpreter not executable"; exit 2; fi
"$ENV_PY" - <<'PY' || rc=1
import os,sys
print('sys.executable=',sys.executable)
print('version=',sys.version)
for name in ('torch','transformers','llamafactory','vllm','peft','datasets'):
 try:
  m=__import__(name); print(name,'OK',getattr(m,'__version__','unknown'),getattr(m,'__file__',''))
 except Exception as e:
  print(name,'FAIL',type(e).__name__,e)
PY
BIN_DIR=$(dirname "$ENV_PY")
for tool in pip llamafactory-cli vllm torchrun; do
  p="$BIN_DIR/$tool"
  if [[ -f "$p" ]]; then
    echo "[TOOL] $p"
    head -n 1 "$p" 2>/dev/null || true
    [[ -x "$p" ]] || echo "[WARN] not executable"
  else
    echo "[MISS] $p"
  fi
done
"$ENV_PY" -m pip --version || { echo "[FAIL] python -m pip unavailable"; rc=1; }
if [[ -n "$LF_DIR" ]]; then
  [[ -f "$LF_DIR/pyproject.toml" ]] || echo "[WARN] no pyproject.toml at $LF_DIR"
fi
exit "$rc"
