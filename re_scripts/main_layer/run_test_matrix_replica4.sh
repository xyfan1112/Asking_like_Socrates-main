#!/usr/bin/env bash
# Four replicated single-GPU servers; every evaluation protocol is sharded across all replicas.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
KEY=${2:?model key}
URLS_CSV=${3:-http://127.0.0.1:8010/v1,http://127.0.0.1:8011/v1,http://127.0.0.1:8012/v1,http://127.0.0.1:8013/v1}
RUN_POLICY=${4:-fresh}

declare -a STAGE_NAMES=()
declare -a STAGE_CODES=()
run_stage() {
  local name=$1; shift
  echo "[MATRIX REPLICA4] START $name"
  "$@"; local code=$?
  STAGE_NAMES+=("$name"); STAGE_CODES+=("$code")
  echo "[MATRIX REPLICA4] END $name exit=$code"
}

run_stage dota_detection \
  bash "$ROOT/test_layer/protocols/run_dota_detection_replica4.sh" \
  "$SETTINGS" "$KEY" "$URLS_CSV" val "$RUN_POLICY"
run_stage dota_ref_grounding_k1 \
  bash "$ROOT/test_layer/protocols/run_ref_grounding_replica4.sh" \
  "$SETTINGS" "$KEY" "$URLS_CSV" dota_ref ref_grounding "$RUN_POLICY"
run_stage dota_ref_classification \
  bash "$ROOT/test_layer/protocols/run_ref_classification_replica4.sh" \
  "$SETTINGS" "$KEY" "$URLS_CSV" "$RUN_POLICY"
run_stage dota_ref_grounding_k5 \
  bash "$ROOT/test_layer/protocols/run_ref_grounding_replica4.sh" \
  "$SETTINGS" "$KEY" "$URLS_CSV" dota_ref ref_grounding_k5 "$RUN_POLICY"

BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
NAMES=$(IFS=,; echo "${STAGE_NAMES[*]}")
CODES=$(IFS=,; echo "${STAGE_CODES[*]}")
"$BOOTSTRAP_PY" - "$SETTINGS" "$KEY" "$RUN_POLICY" "$NAMES" "$CODES" "$URLS_CSV" <<'PY'
import json,sys
from datetime import datetime, timezone
from pathlib import Path
s=json.load(open(sys.argv[1],encoding='utf-8'))
names=sys.argv[4].split(',') if sys.argv[4] else []
codes=[int(x) for x in sys.argv[5].split(',')] if sys.argv[5] else []
report={
  'schema_version':'test_matrix_v1_2_3_replica4',
  'model_key':sys.argv[2], 'run_policy':sys.argv[3],
  'topology':'four independent single-GPU vLLM replicas with stable SHA256 shards',
  'urls':sys.argv[6].split(','),
  'stages':[{'name':n,'exit_code':c,'passed':c==0} for n,c in zip(names,codes)],
  'passed':all(c==0 for c in codes),
  'finished_at':datetime.now(timezone.utc).isoformat(),
}
out=Path(s['paths']['test_run_root'])/'matrix'/sys.argv[2]/'test_matrix_status.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2)); print('[MATRIX] report=',out)
PY
for code in "${STAGE_CODES[@]}"; do (( code == 0 )) || exit "$code"; done
echo "[OK] replica4 matrix completed for $KEY"
