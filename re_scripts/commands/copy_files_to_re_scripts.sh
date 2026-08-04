#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
source "$HERE/commands/user_config.sh"
TARGET="$REPO_ROOT/re_scripts"
[[ -d "$TARGET" ]] || { echo "[FAIL] re_scripts不存在: $TARGET" >&2; exit 2; }

cp -a "$HERE/files/." "$TARGET/"
PY=$(command -v python3 || command -v python)

while IFS= read -r source; do
  rel=${source#"$HERE/files/"}
  "$PY" -m py_compile "$TARGET/$rel"
done < <(find "$HERE/files" -type f -name '*.py' | sort)

while IFS= read -r source; do
  rel=${source#"$HERE/files/"}
  bash -n "$TARGET/$rel"
done < <(find "$HERE/files" -type f -name '*.sh' | sort)

bash -n "$HERE/commands/run_v1.2.1.sh" "$HERE/commands/user_config.sh"

echo "[PASS] files/ 已覆盖到 $TARGET，补丁内全部Python和Shell静态语法检查通过。"
echo "[INFO] 可用 VS Code Discard Changes 回退已追踪文件；新增未追踪文件需手动删除。"
