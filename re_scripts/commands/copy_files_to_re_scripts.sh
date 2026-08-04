#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/commands/user_config.sh"
TARGET="$REPO_ROOT/re_scripts"
[[ -d "$TARGET" ]] || { echo "[FAIL] re_scripts不存在: $TARGET" >&2; exit 2; }
cp -a "$HERE/files/." "$TARGET/"
PY=$(command -v python3 || command -v python)
"$PY" -m py_compile \
  "$TARGET/main_layer/run.py" \
  "$TARGET/main_layer/taxonomy.py" \
  "$TARGET/data_layer/qa_i18n.py" \
  "$TARGET/data_layer/prompts/profiles.py" \
  "$TARGET/data_layer/official_socratic/local_api_adapter/utils.py" \
  "$TARGET/tools/materialize_language_settings.py" \
  "$TARGET/tools/check_debug_infrastructure.py" \
  "$TARGET/tools/inspect_api_truncation.py" \
  "$TARGET/train_layer/01c_validate_d1_direct_only.py" \
  "$TARGET/train_layer/02c_generate_d1_direct_only_config.py"
bash -n \
  "$TARGET/train_layer/03c_train_d1_direct_only.sh" \
  "$TARGET/train_layer/05c_export_d1_direct_only.sh"
echo "[PASS] files/ 已覆盖到 $TARGET，Python和Shell静态语法检查通过。"
echo "[INFO] 可用 VS Code Discard Changes 回退已追踪文件；新增未追踪文件需手动删除或 git clean -fd 前先检查。"
