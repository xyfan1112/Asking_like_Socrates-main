#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
[[ -n "$BOOTSTRAP_PY" ]] || { echo "[FAIL] 找不到 Python" >&2; exit 2; }
DATA_PY=$("$BOOTSTRAP_PY" "$ROOT/main_layer/resolve_runtime.py" --settings "$SETTINGS" --workload data --field python)

read_cfg() {
  "$DATA_PY" - "$SETTINGS" "$@" <<'PY'
import json,sys
v=json.load(open(sys.argv[1],encoding='utf-8'))
for k in sys.argv[2:]: v=v[k]
print(v)
PY
}

WORK=$(read_cfg paths pipeline_work_root)
PID_DIR="$WORK/agents/pids"
mkdir -p "$PID_DIR"

is_vllm_port_process() {
  local pid=$1 port=$2
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  local cmd
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
  [[ "$cmd" == *vllm* ]] || return 1
  [[ "$cmd" =~ --port([=[:space:]])${port}([[:space:]]|$) ]] || return 1
  return 0
}

listener_pids() {
  local port=$1
  local found=""
  if command -v lsof >/dev/null 2>&1; then
    found+=" $(lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  fi
  if command -v ss >/dev/null 2>&1; then
    found+=" $(ss -ltnp "sport = :$port" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 || true)"
  fi
  ps -u "$(id -u)" -o pid=,args= 2>/dev/null | while read -r pid args; do
    if [[ "$args" == *vllm* ]] && [[ "$args" =~ --port([=[:space:]])${port}([[:space:]]|$) ]]; then
      echo "$pid"
    fi
  done
  for pid in $found; do echo "$pid"; done
}

stop_pid() {
  local pid=$1 role=$2 port=$3
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  if ! is_vllm_port_process "$pid" "$port"; then
    echo "[WARN] 不停止非预期进程 PID=$pid role=$role port=$port" >&2
    return 0
  fi
  local pgid
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
  echo "[STOP] $role port=$port PID=$pid PGID=${pgid:-unknown}"
  if [[ -n "$pgid" ]]; then kill -TERM -- "-$pgid" 2>/dev/null || true; fi
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.5
  done
  if [[ -n "$pgid" ]]; then kill -KILL -- "-$pgid" 2>/dev/null || true; fi
  kill -KILL "$pid" 2>/dev/null || true
}

failed=0
for role in reasoner perceiver verifier; do
  port=$(read_cfg agents "$role" port)
  pid_file="$PID_DIR/$role.pid"
  pids=""
  if [[ -f "$pid_file" ]]; then pids+=" $(cat "$pid_file" 2>/dev/null || true)"; fi
  pids+=" $(listener_pids "$port" | sort -u | tr '\n' ' ')"
  for pid in $(echo "$pids" | xargs -n1 2>/dev/null | sort -u); do
    stop_pid "$pid" "$role" "$port"
  done
  rm -f "$pid_file"
  for _ in $(seq 1 20); do
    if ! curl -fsS --max-time 1 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then break; fi
    sleep 0.5
  done
  if curl -fsS --max-time 1 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
    echo "[FAIL] $role 仍在监听 port=$port" >&2
    failed=1
  else
    echo "[FREE] $role port=$port"
  fi
done

if [[ $failed -ne 0 ]]; then
  echo "[HINT] 检查: ss -ltnp | grep -E ':8001|:8002|:8003'" >&2
  exit 2
fi

echo "[AGENTS STOP] PASS"
