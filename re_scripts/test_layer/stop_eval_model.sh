#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if [[ ${1:-} == *.json && -f ${1:-} ]]; then
  SETTINGS=$1
  KEY=${2:?model key required}
  PORT=${3:-8010}
  BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
  PIDFILE=$("$BOOTSTRAP_PY" - "$SETTINGS" "$KEY" "$PORT" <<'PY'
import json,sys
s=json.load(open(sys.argv[1],encoding='utf-8'))
print(f"{s['paths']['test_run_root']}/servers/{sys.argv[2]}_{sys.argv[3]}.pid")
PY
)
else
  # Backward-compatible direct use: stop_eval_model.sh /path/to/server.pid
  PIDFILE=${1:?settings+model key or pid file required}
fi
[[ -f "$PIDFILE" ]] || { echo "[INFO] pid file absent"; exit 0; }
pid=$(cat "$PIDFILE" 2>/dev/null || true)
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep .5; done
  kill -9 -- "-$pid" 2>/dev/null || true
fi
rm -f "$PIDFILE"
echo "[STOPPED] $PIDFILE"
