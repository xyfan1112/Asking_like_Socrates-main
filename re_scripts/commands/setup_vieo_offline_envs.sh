#!/usr/bin/env bash
# 在 vieo 临时机离线解压 conda-pack 环境并执行 conda-unpack。
# 默认搜索：/home/vieo/vieo/fxy_workspace/fxy/offlinetransfer、offline、fxy 根目录。
set -Eeuo pipefail

WORKSPACE="/home/vieo/vieo/fxy_workspace"
FXY_ROOT="$WORKSPACE/fxy"
PROJECT_ROOT="$WORKSPACE/Asking_like_Socrates-main"
CONDA_ROOT="/home/vieo/anaconda3"
ENV_ROOT="$CONDA_ROOT/envs"
ARCHIVE_DIR=""
REPLACE=0

usage() {
  cat <<'TXT'
用法：
  bash re_scripts/commands/setup_vieo_offline_envs.sh
  bash re_scripts/commands/setup_vieo_offline_envs.sh --archive-dir /目录
  bash re_scripts/commands/setup_vieo_offline_envs.sh --replace

默认不会覆盖已存在且含 bin/python 的环境。
--replace 会先把旧环境改名备份，再解压；不会直接删除。
TXT
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive-dir) ARCHIVE_DIR=${2:?缺少目录}; shift 2 ;;
    --replace) REPLACE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[FAIL] 未知参数: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -d "$PROJECT_ROOT" ]] || { echo "[FAIL] 项目目录不存在: $PROJECT_ROOT" >&2; exit 2; }
mkdir -p "$ENV_ROOT"

find_archive() {
  local env_name=$1
  local -a roots=()
  if [[ -n "$ARCHIVE_DIR" ]]; then
    roots+=("$ARCHIVE_DIR")
  else
    roots+=("$FXY_ROOT/offlinetransfer" "$FXY_ROOT/offline" "$FXY_ROOT")
  fi
  local -a found=()
  local root
  for root in "${roots[@]}"; do
    [[ -d "$root" ]] || continue
    while IFS= read -r -d '' f; do found+=("$f"); done < <(
      find "$root" -maxdepth 3 -type f \
        \( -iname "*${env_name}*.tar.gz" -o -iname "*${env_name}*.tgz" \) \
        -print0 2>/dev/null
    )
  done
  # 去重
  local -A seen=(); local -a unique=(); local f
  for f in "${found[@]}"; do
    [[ -n "${seen[$f]:-}" ]] && continue
    seen[$f]=1; unique+=("$f")
  done
  if (( ${#unique[@]} == 0 )); then
    echo "[FAIL] 未找到 ${env_name} 的 tar.gz。搜索目录: ${roots[*]}" >&2
    return 2
  fi
  if (( ${#unique[@]} > 1 )); then
    echo "[FAIL] 找到多个 ${env_name} 压缩包，请用 --archive-dir 指定唯一目录：" >&2
    printf '  %s\n' "${unique[@]}" >&2
    return 2
  fi
  printf '%s\n' "${unique[0]}"
}

verify_sidecar() {
  local archive=$1
  local candidates=("${archive}.sha256" "${archive%.tar.gz}.sha256" "${archive%.tgz}.sha256")
  local sumfile=""
  local c
  for c in "${candidates[@]}"; do [[ -f "$c" ]] && { sumfile=$c; break; }; done
  if [[ -z "$sumfile" ]]; then
    echo "[WARN] 没有找到单独SHA256文件，计算当前摘要："
    sha256sum "$archive"
    return 0
  fi
  echo "[CHECK] SHA256: $sumfile"
  local expected actual
  expected=$(grep -Eo '[0-9a-fA-F]{64}' "$sumfile" | head -1 | tr 'A-F' 'a-f')
  actual=$(sha256sum "$archive" | awk '{print $1}')
  [[ -n "$expected" ]] || { echo "[FAIL] SHA256文件中没有64位摘要: $sumfile" >&2; return 2; }
  [[ "$expected" == "$actual" ]] || { echo "[FAIL] SHA256不一致: $archive" >&2; return 2; }
  echo "[PASS] SHA256一致: $actual"
}

install_env() {
  local name=$1 archive=$2 target="$ENV_ROOT/$name"
  verify_sidecar "$archive"

  if [[ -x "$target/bin/python" && $REPLACE -eq 0 ]]; then
    echo "[REUSE] 环境已存在，不覆盖: $target"
  else
    if [[ -e "$target" ]]; then
      local backup="${target}_backup_$(date +%Y%m%d_%H%M%S)"
      echo "[BACKUP] $target -> $backup"
      mv "$target" "$backup"
    fi
    local tmp
    tmp=$(mktemp -d "/tmp/${name}_unpack_XXXXXX")
    echo "[EXTRACT] $archive -> $tmp"
    tar -xzf "$archive" -C "$tmp"

    local source="$tmp"
    if [[ ! -x "$source/bin/python" ]]; then
      local -a tops=()
      while IFS= read -r -d '' d; do tops+=("$d"); done < <(find "$tmp" -mindepth 1 -maxdepth 1 -type d -print0)
      if (( ${#tops[@]} == 1 )) && [[ -x "${tops[0]}/bin/python" ]]; then
        source=${tops[0]}
      else
        echo "[FAIL] 压缩包解压后未找到 bin/python；它可能不是conda-pack归档: $archive" >&2
        find "$tmp" -maxdepth 2 -type f | head -30 >&2 || true
        rm -rf "$tmp"
        return 2
      fi
    fi

    mkdir -p "$target"
    cp -a "$source/." "$target/"
    rm -rf "$tmp"
  fi

  [[ -x "$target/bin/python" ]] || { echo "[FAIL] Python不可执行: $target/bin/python" >&2; return 2; }
  if [[ -x "$target/bin/conda-unpack" ]]; then
    echo "[RELOCATE] 执行 conda-unpack: $name"
    env PATH="$target/bin:$PATH" "$target/bin/conda-unpack"
  else
    echo "[FAIL] conda-unpack不存在: $target/bin/conda-unpack；请确认归档确由conda-pack生成" >&2
    return 2
  fi

  echo "[VERIFY] $name"
  "$target/bin/python" -V
  if [[ "$name" == "als_vllm" ]]; then
    "$target/bin/python" - <<'PY'
import sys, torch, vllm, transformers
print('python=', sys.executable)
print('torch=', torch.__version__, 'cuda=', torch.version.cuda, 'available=', torch.cuda.is_available())
print('vllm=', vllm.__version__)
print('transformers=', transformers.__version__)
PY
  else
    # 将项目内LLaMA-Factory源码注册到离线环境，无需联网pip安装。
    local site_dir
    site_dir=$("$target/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
    printf '%s\n' "$PROJECT_ROOT/LLaMA-Factory/src" > "$site_dir/asking_like_socrates_llamafactory.pth"
    "$target/bin/python" - <<'PY'
import sys, torch, transformers, accelerate, peft
print('python=', sys.executable)
print('torch=', torch.__version__, 'cuda=', torch.version.cuda, 'available=', torch.cuda.is_available())
print('transformers=', transformers.__version__)
print('accelerate=', accelerate.__version__)
print('peft=', peft.__version__)
try:
    import llamafactory
    print('llamafactory=', llamafactory.__file__)
except Exception as exc:
    raise SystemExit(f'llamafactory import failed: {type(exc).__name__}: {exc}')
PY
  fi
  echo "[PASS] $name 环境可用: $target"
}

SFT_ARCHIVE=$(find_archive als_sft)
VLLM_ARCHIVE=$(find_archive als_vllm)
echo "[FOUND] als_sft=$SFT_ARCHIVE"
echo "[FOUND] als_vllm=$VLLM_ARCHIVE"

install_env als_sft "$SFT_ARCHIVE"
install_env als_vllm "$VLLM_ARCHIVE"

echo "[PASS] 两个离线环境已部署。"
echo "下一步：bash $PROJECT_ROOT/re_scripts/commands/prepare_vieo_workspace.sh"
