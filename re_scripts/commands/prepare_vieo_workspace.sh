#!/usr/bin/env bash
# 修正从 yk/nhl 复制到 vieo 后的生成型绝对路径和scene manifest。
set -Eeuo pipefail
WORKSPACE="/home/vieo/vieo/fxy_workspace"
PROJECT_ROOT="$WORKSPACE/Asking_like_Socrates-main"
FXY_ROOT="$WORKSPACE/fxy"
RS="$PROJECT_ROOT/re_scripts"
RAW="$FXY_ROOT/datasets/yw128"
SCENE="$FXY_ROOT/datasets/yw128_scene_disjoint"
RESULTS="$FXY_ROOT/results_ssh"
STAMP=$(date +%Y%m%d_%H%M%S)

fail() { echo "[FAIL] $*" >&2; exit 2; }
[[ "$(readlink -f "$RS")" == "$(readlink -f "$(cd "$(dirname "$0")/.." && pwd)")" ]] || fail "脚本不是从目标项目目录执行"
[[ -f "$RAW/classes.txt" ]] || fail "缺少: $RAW/classes.txt"
[[ -d "$SCENE/train" && -d "$SCENE/val" ]] || fail "无泄漏数据目录不完整: $SCENE"

if [[ ! -f "$SCENE/classes.txt" ]]; then
  cp "$RAW/classes.txt" "$SCENE/classes.txt"
  echo "[COPY] classes.txt -> scene_disjoint"
fi

MANIFEST="$SCENE/scene_split_manifest.json"
[[ -f "$MANIFEST" ]] || fail "缺少scene manifest: $MANIFEST"
cp -a "$MANIFEST" "${MANIFEST}.before_vieo_${STAMP}"
python3 - "$MANIFEST" "$RAW" "$SCENE" <<'PY'
import json,sys
from datetime import datetime, timezone
from pathlib import Path
p=Path(sys.argv[1]); raw=str(Path(sys.argv[2]).resolve()); out=str(Path(sys.argv[3]).resolve())
data=json.loads(p.read_text(encoding='utf-8'))
old_source=data.get('source_root'); old_output=data.get('output_root')
data['source_root']=raw
data['output_root']=out
data.setdefault('relocations',[]).append({
    'at': datetime.now(timezone.utc).isoformat(),
    'old_source_root': old_source,
    'old_output_root': old_output,
    'new_source_root': raw,
    'new_output_root': out,
    'reason': 'temporary migration to /home/vieo/vieo/fxy_workspace',
})
p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('[PASS] scene manifest relocated:',p)
print('  source_root=',raw)
print('  output_root=',out)
print('  overlap=',data.get('scene_overlap'))
PY

# 旧runtime_settings和活动settings保存了/home/yk或/home/nhl绝对路径，必须重新物化。
if [[ -d "$RESULTS" ]]; then
  while IFS= read -r -d '' d; do
    echo "[REMOVE GENERATED] $d"
    rm -rf "$d"
  done < <(find "$RESULTS" -type d -name runtime_settings -print0 2>/dev/null)

  while IFS= read -r -d '' f; do
    echo "[ARCHIVE GENERATED CFG] $f"
    mv "$f" "${f}.before_vieo_${STAMP}"
  done < <(find "$RESULTS" -type f \
    \( -name 'settings.zh.v1.2.3.json' -o -name 'settings.en.v1.2.3.json' \
       -o -name 'settings.zh.v1.2.3.manifest.json' -o -name 'settings.en.v1.2.3.manifest.json' \
       -o -name 'settings.zh.v1.2.3.omni-shared.json' -o -name 'settings.en.v1.2.3.omni-shared.json' \
       -o -name 'settings.zh.v1.2.3.three-services.json' -o -name 'settings.en.v1.2.3.three-services.json' \) \
    -print0 2>/dev/null)
fi

find "$RS/commands" "$RS/main_layer" "$RS/train_layer" "$RS/test_layer" "$RS/tools" \
  -type f \( -name '*.sh' -o -name '*.py' \) -exec chmod u+rx {} +

echo "[PASS] workspace路径迁移准备完成。"
echo "下一步："
echo "  cd $RS"
echo "  bash commands/run_v1.2.3.sh doctor"
echo "  bash commands/run_v1.2.3.sh eval-route-check"
