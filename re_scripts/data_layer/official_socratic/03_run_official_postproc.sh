#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
SPLIT=${2:-train}

BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
[[ -n "$BOOTSTRAP_PY" ]] || { echo "[FAIL] 找不到 Python" >&2; exit 2; }
mapfile -d '' -t CFG < <("$BOOTSTRAP_PY" - "$SETTINGS" "$SPLIT" "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[3]); s=json.load(open(sys.argv[1],encoding='utf-8')); split=sys.argv[2]; o=s['official_socratic']; t=s.get('trajectory',{})
sys.path.insert(0,str(root/'main_layer'))
from config import resolve_python
vals=[resolve_python(s,'data'),s['project']['project_root'],f"{o['raw_output_dir']}/dota128_{split}_official.jsonl",f"{o['raw_output_dir']}/dota128_{split}_official_strict.jsonl",f"{o['postproc_output_dir']}/dota128_{split}_official_merge.json",f"{o['raw_output_dir']}/dota128_{split}_official.manifest.json",str(int(t.get('min_strict_grounding_samples',20))),str(int(t.get('min_strict_classification_samples',20)))]
for v in vals: sys.stdout.write(v); sys.stdout.write('\0')
PY
)
PYTHON_BIN=${CFG[0]}; PROJECT_ROOT=${CFG[1]}; RAW=${CFG[2]}; STRICT_RAW=${CFG[3]}; OUT=${CFG[4]}; MANIFEST=${CFG[5]}; MIN_G=${CFG[6]}; MIN_C=${CFG[7]}
POSTPROC="$PROJECT_ROOT/SocraticAgent/postproc.py"
[[ -f "$RAW" ]] || { echo "[FAIL] canonical full raw missing: $RAW" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "[FAIL] v4.3 run manifest missing. Do not postprocess an old/mixed raw. Regenerate with policy=fresh." >&2; exit 2; }
"$PYTHON_BIN" - "$MANIFEST" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding='utf-8'))
if m.get('status')!='completed' or not str(m.get('package_version','')).startswith('4.3'):
 raise SystemExit('[FAIL] manifest is not a completed v4.3 run')
print(f"[OK] manifest signature={m.get('signature')} rows={m.get('rows')}")
PY
mkdir -p "$(dirname "$OUT")"

"$PYTHON_BIN" "$ROOT/data_layer/official_socratic/03a_audit_official_raw.py" --settings "$SETTINGS" --split "$SPLIT" --input "$RAW" --output "$STRICT_RAW"
REPORT="${STRICT_RAW%.jsonl}.report.json"
"$PYTHON_BIN" - "$REPORT" "$MIN_G" "$MIN_C" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding='utf-8')); by=r.get('accepted_by_task',{})
g=int(by.get('ref_grounding_obb',0)); c=int(by.get('ref_classification',0)); mg=int(sys.argv[2]); mc=int(sys.argv[3])
print(f"[STRICT COUNTS] grounding={g} classification={c} required={mg}/{mc}")
if g<mg or c<mc:
 raise SystemExit('[FAIL] strict Socratic data is not balanced or large enough for B2. Improve queries/generation before conversion.')
PY

cd "$PROJECT_ROOT"
"$PYTHON_BIN" "$POSTPROC" --data_path "$STRICT_RAW" --out_path "$OUT"
echo "[OK] official postproc output: $OUT"
