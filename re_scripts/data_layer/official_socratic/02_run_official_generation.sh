#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SETTINGS=${1:-$ROOT/settings.json}
SPLIT=${2:-train}
MODE=${3:-full}          # full | debug
DEBUG_SAMPLES=${4:-20}
RUN_POLICY=${5:-auto}    # auto | fresh | resume

BOOTSTRAP_PY=${PYTHON_BOOTSTRAP:-$(command -v python3 || command -v python || true)}
[[ -n "$BOOTSTRAP_PY" ]] || { echo "[FAIL] 找不到 Python" >&2; exit 2; }
[[ -f "$SETTINGS" ]] || { echo "[FAIL] settings not found: $SETTINGS" >&2; exit 2; }

mapfile -d '' -t CFG < <("$BOOTSTRAP_PY" - "$SETTINGS" "$SPLIT" "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[3]); s=json.load(open(sys.argv[1],encoding='utf-8')); split=sys.argv[2]
sys.path.insert(0,str(root/'main_layer'))
from config import resolve_python
o=s['official_socratic']; a=s['agents']; t=s.get('trajectory',{}); c=s.get('data_conversion',{})
profile=str(t.get('prompt_profile','official_general')).strip()
values=[
 resolve_python(s,'data'), s['project']['project_root'],
 f"{o['input_parquet_dir']}/dota128_{split}_official.parquet", o['raw_output_dir'],
 str(o['concurrency']), str(o['max_loop']), str(o['current_task_samples']),
 str(o['verify_inst']), str(o['image_meta_pre']), profile,
 a['reasoner']['base_url'],a['reasoner']['served_name'],str(a['reasoner']['max_tokens']),str(a['reasoner']['temperature']),
 a['perceiver']['base_url'],a['perceiver']['served_name'],str(a['perceiver']['max_tokens']),str(a['perceiver']['temperature']),
 a['verifier']['base_url'],a['verifier']['served_name'],str(a['verifier']['max_tokens']),str(a['verifier']['temperature']),
 str(c.get('coordinate_target','norm1000_obb')),
 str(bool(t.get('force_final_on_last_round',True))).lower(),
 str(int(t.get('max_repair_attempts',2))),
 str(float(t.get('question_similarity_threshold',0.82))),
 str(bool(t.get('require_perceiver_original_context',True))).lower(),
 str(bool(t.get('enable_perceiver_focus_crop',True))).lower(),
 str(int(t.get('focus_crop_long_side',1024))),
 str(float(t.get('focus_crop_padding_ratio',0.02))),
 str(bool(t.get('map_crop_coordinates',True))).lower(),
 str(bool(t.get('deterministic_structured_verifier',True))).lower(),
 str(int(t.get('reasoner_compact_max_chars',9000))),
 str(bool(t.get('classification_use_full_and_crop',True))).lower(),
 str(float(t.get('classification_crop_only_max_area',0.0))),
 str(bool(t.get('classification_reject_class_claims',False))).lower(),
 str(float(t.get('max_crop_coordinate_coverage',0.94))),
 str(bool(t.get('grounding_coordinate_crop_only',True))).lower(),
 str(bool(t.get('reject_unsolicited_coordinates',True))).lower(),
 str(bool(t.get('perceiver_fail_closed',True))).lower(),
 str(bool(t.get('reasoner_fail_closed',True))).lower(),
]
for v in values:
 sys.stdout.write(v); sys.stdout.write('\0')
PY
)
[[ ${#CFG[@]} -eq 41 ]] || { echo "[FAIL] expected 41 config fields, got ${#CFG[@]}" >&2; exit 2; }

PYTHON_BIN=${CFG[0]}; PROJECT_ROOT=${CFG[1]}; PARQUET=${CFG[2]}; OUTPUT_DIR=${CFG[3]}
CONCURRENCY=${CFG[4]}; MAX_LOOP=${CFG[5]}; CURRENT_TASK_SAMPLES=${CFG[6]}
VERIFY_INST=${CFG[7]}; IMAGE_META_PRE=${CFG[8]}; PROMPT_PROFILE=${CFG[9]}
export LOCAL_REASONER_BASE_URL=${CFG[10]}; export LOCAL_REASONER_MODEL=${CFG[11]}; export LOCAL_REASONER_MAX_TOKENS=${CFG[12]}; export LOCAL_REASONER_TEMPERATURE=${CFG[13]}
export LOCAL_PERCEIVER_BASE_URL=${CFG[14]}; export LOCAL_PERCEIVER_MODEL=${CFG[15]}; export LOCAL_PERCEIVER_MAX_TOKENS=${CFG[16]}; export LOCAL_PERCEIVER_TEMPERATURE=${CFG[17]}
export LOCAL_VERIFIER_BASE_URL=${CFG[18]}; export LOCAL_VERIFIER_MODEL=${CFG[19]}; export LOCAL_VERIFIER_MAX_TOKENS=${CFG[20]}; export LOCAL_VERIFIER_TEMPERATURE=${CFG[21]}
export ALS_COORDINATE_TARGET=${CFG[22]}; export ALS_FORCE_FINAL_ON_LAST_ROUND=${CFG[23]}; export ALS_MAX_REPAIR_ATTEMPTS=${CFG[24]}
export ALS_QUESTION_SIMILARITY_THRESHOLD=${CFG[25]}; export ALS_REQUIRE_PERCEIVER_CONTEXT=${CFG[26]}
export ALS_ENABLE_FOCUS_CROP=${CFG[27]}; export ALS_FOCUS_CROP_LONG_SIDE=${CFG[28]}; export ALS_FOCUS_CROP_PADDING_RATIO=${CFG[29]}
export ALS_MAP_CROP_COORDINATES=${CFG[30]}; export ALS_DETERMINISTIC_STRUCTURED_VERIFIER=${CFG[31]}; export ALS_REASONER_COMPACT_MAX_CHARS=${CFG[32]}
export ALS_CLASSIFICATION_USE_FULL_AND_CROP=${CFG[33]}; export ALS_CLASSIFICATION_CROP_ONLY_MAX_AREA=${CFG[34]}; export ALS_CLASSIFICATION_REJECT_CLASS_CLAIMS=${CFG[35]}
export ALS_MAX_CROP_COORDINATE_COVERAGE=${CFG[36]}; export ALS_GROUNDING_COORDINATE_CROP_ONLY=${CFG[37]}; export ALS_REJECT_UNSOLICITED_COORDINATES=${CFG[38]}
export ALS_PERCEIVER_FAIL_CLOSED=${CFG[39]}; export ALS_REASONER_FAIL_CLOSED=${CFG[40]}
export ALS_MAX_LOOP=$MAX_LOOP; export ALS_BYPASS_STRUCTURED_REWRITE=true

if [[ "$MODE" == "debug" ]]; then
  CONCURRENCY=1
fi

PROMPT_PROFILE=$("$BOOTSTRAP_PY" - "$ROOT" "$PROMPT_PROFILE" <<'PY'
import sys
from pathlib import Path
root=Path(sys.argv[1]); profile=sys.argv[2].strip(); sys.path.insert(0,str(root))
from data_layer.prompts.profiles import PROFILES
if profile not in PROFILES:
 raise SystemExit(f"Unknown prompt profile {profile!r}; available={sorted(PROFILES)}")
print(profile)
PY
)
export ALS_PROMPT_PROFILE="$PROMPT_PROFILE"

OFFICIAL_GENERATOR="$PROJECT_ROOT/SocraticAgent/generation.py"
ADAPTER_DIR="$ROOT/data_layer/official_socratic/local_api_adapter"
[[ -x "$PYTHON_BIN" ]] || { echo "[FAIL] data Python invalid: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$OFFICIAL_GENERATOR" ]] || { echo "[FAIL] official generator missing: $OFFICIAL_GENERATOR" >&2; exit 2; }
[[ -f "$PARQUET" ]] || { echo "[FAIL] parquet missing: $PARQUET" >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"

# A /models response alone does not prove the role can answer requests. Run a
# real chat completion against all three routes before creating/resuming output.
"$PYTHON_BIN" "$ROOT/data_layer/agents/check_three_agents.py" --settings "$SETTINGS"

GUARD_JSON=$("$PYTHON_BIN" "$ROOT/data_layer/official_socratic/00_manage_run_state.py" prepare \
  --settings "$SETTINGS" --split "$SPLIT" --mode "$MODE" --policy "$RUN_POLICY")
RESUME=$("$PYTHON_BIN" -c 'import json,sys; print(str(json.loads(sys.stdin.read())["resume"]).lower())' <<<"$GUARD_JSON")

RUN_TS=$(date +%Y%m%d%H%M%S)
export ALS_API_LOG_PATH="$OUTPUT_DIR/local_api_calls_${SPLIT}_${MODE}_${RUN_TS}.jsonl"
CMD=("$PYTHON_BIN" "$OFFICIAL_GENERATOR" --data-path "$PARQUET" --output-dir "$OUTPUT_DIR" \
  --concurrency "$CONCURRENCY" --max-loop "$MAX_LOOP" --current-task-samples "$CURRENT_TASK_SAMPLES" \
  --verify-inst "$VERIFY_INST" --image-meta-pre "$IMAGE_META_PRE")
if [[ "$MODE" == "debug" ]]; then
  CMD+=(--debug --debug-samples "$DEBUG_SAMPLES")
elif [[ "$MODE" == "full" ]]; then
  [[ "$RESUME" == "true" ]] && CMD+=(--resume)
else
  echo "[FAIL] MODE must be debug or full" >&2; exit 2
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$ADAPTER_DIR:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] Prompt profile: $ALS_PROMPT_PROFILE"
echo "[INFO] Split=$SPLIT mode=$MODE policy=$RUN_POLICY resume=$RESUME concurrency=$CONCURRENCY"
echo "[INFO] max_loop=$MAX_LOOP force_final=$ALS_FORCE_FINAL_ON_LAST_ROUND repair_attempts=$ALS_MAX_REPAIR_ATTEMPTS"
echo "[INFO] perceiver_original_context_required=$ALS_REQUIRE_PERCEIVER_CONTEXT question_similarity_threshold=$ALS_QUESTION_SIMILARITY_THRESHOLD"
echo "[INFO] coordinate_target=$ALS_COORDINATE_TARGET"
echo "[INFO] focus_crop=$ALS_ENABLE_FOCUS_CROP long_side=$ALS_FOCUS_CROP_LONG_SIDE padding=$ALS_FOCUS_CROP_PADDING_RATIO map_crop=$ALS_MAP_CROP_COORDINATES"
echo "[INFO] classification_full_and_crop=$ALS_CLASSIFICATION_USE_FULL_AND_CROP reject_class_claims=$ALS_CLASSIFICATION_REJECT_CLASS_CLAIMS"
echo "[INFO] grounding_coordinate_crop_only=$ALS_GROUNDING_COORDINATE_CROP_ONLY reject_unsolicited=$ALS_REJECT_UNSOLICITED_COORDINATES perceiver_fail_closed=$ALS_PERCEIVER_FAIL_CLOSED reasoner_fail_closed=$ALS_REASONER_FAIL_CLOSED"
echo "[INFO] Parquet: $PARQUET"
echo "[INFO] API call log: $ALS_API_LOG_PATH"
echo "[INFO] Running official SocraticAgent:"
printf '  %q' "${CMD[@]}"; echo
"${CMD[@]}"

if [[ "$MODE" == "full" ]]; then
  "$PYTHON_BIN" "$ROOT/data_layer/official_socratic/00_manage_run_state.py" finalize --settings "$SETTINGS" --split "$SPLIT"
  CANONICAL="$OUTPUT_DIR/dota128_${SPLIT}_official.jsonl"
  "$PYTHON_BIN" "$ROOT/data_layer/official_socratic/02c_audit_full_generation.py" --raw "$CANONICAL" --api-log "$ALS_API_LOG_PATH"
  echo "[GENERATION COMPLETE] Full raw completed. Run strict audit/postproc next."
else
  LATEST=$(ls -t "$OUTPUT_DIR"/*debug*.jsonl 2>/dev/null | grep -v local_api_calls | head -1 || true)
  [[ -n "$LATEST" ]] || { echo "[FAIL] debug raw missing" >&2; exit 3; }
  "$PYTHON_BIN" "$ROOT/data_layer/official_socratic/02b_audit_debug_generation.py" --settings "$SETTINGS" --raw "$LATEST" --api-log "$ALS_API_LOG_PATH"
  echo "[GENERATION COMPLETE] Debug run audited. Inspect the audit before full generation."
fi
