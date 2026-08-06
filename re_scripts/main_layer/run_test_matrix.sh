#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
KEY=${2:?model key}
URL=${3:-http://127.0.0.1:8010/v1}
RUN_POLICY=${4:-fresh}

declare -a STAGE_NAMES=()
declare -a STAGE_CODES=()

run_stage() {
  local name=$1
  shift
  echo "[MATRIX] START $name"
  "$@"
  local code=$?
  STAGE_NAMES+=("$name")
  STAGE_CODES+=("$code")
  echo "[MATRIX] END $name exit=$code"
}

run_stage dota_detection \
  bash "$ROOT/test_layer/protocols/run_dota_detection.sh" \
  "$SETTINGS" "$KEY" "$URL" val "$RUN_POLICY"
run_stage dota_ref_grounding_k1 \
  bash "$ROOT/test_layer/protocols/run_ref_grounding.sh" \
  "$SETTINGS" "$KEY" "$URL" dota_ref ref_grounding "$RUN_POLICY"
run_stage dota_ref_classification \
  bash "$ROOT/test_layer/protocols/run_ref_classification.sh" \
  "$SETTINGS" "$KEY" "$URL" "$RUN_POLICY"
run_stage dota_ref_grounding_k5 \
  bash "$ROOT/test_layer/protocols/run_ref_grounding.sh" \
  "$SETTINGS" "$KEY" "$URL" dota_ref ref_grounding_k5 "$RUN_POLICY"

BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
NAMES=$(IFS=,; echo "${STAGE_NAMES[*]}")
CODES=$(IFS=,; echo "${STAGE_CODES[*]}")
"$BOOTSTRAP_PY" - "$SETTINGS" "$KEY" "$RUN_POLICY" "$NAMES" "$CODES" <<'PY'
import json
import sys
from pathlib import Path

settings = json.load(open(sys.argv[1], encoding="utf-8"))
names = sys.argv[4].split(",") if sys.argv[4] else []
codes = [int(value) for value in sys.argv[5].split(",")] if sys.argv[5] else []
stages = [{"name": name, "exit_code": code, "passed": code == 0} for name, code in zip(names, codes)]
report = {
    "schema_version": "test_matrix_v4_3_3",
    "model_key": sys.argv[2],
    "run_policy": sys.argv[3],
    "passed": all(item["passed"] for item in stages),
    "stages": stages,
    "note": "All stages are attempted even if one protocol fails.",
}
output = Path(settings["paths"]["test_run_root"]) / "matrix" / sys.argv[2] / "test_matrix_status.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
print(f"[MATRIX] report={output}")
PY

for code in "${STAGE_CODES[@]}"; do
  if (( code != 0 )); then
    exit "$code"
  fi
done
echo "[OK] DOTA detection, Ref localization, Ref classification, and K=5 stability completed for $KEY."
