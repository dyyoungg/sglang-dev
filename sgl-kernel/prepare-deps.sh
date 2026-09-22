#!/usr/bin/env bash
# Usage: bash prepare-deps.sh [destination] [parallel jobs]
# Downloads sources only; does not build or install packages.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == -h || ${1:-} == --help ]]; then
  echo "Usage: bash $0 [destination] [parallel jobs (default: 4)]"
  exit 0
fi
deps_dir=${1:-"$script_dir/../sgl-kernel-third-party"}
jobs=${2:-4}
if [[ ! $jobs =~ ^[1-9][0-9]*$ ]]; then
  echo "parallel jobs must be a positive integer" >&2
  exit 2
fi
for tool in git python3 flock; do
  command -v "$tool" >/dev/null || { echo "Missing tool: $tool" >&2; exit 2; }
done
mkdir -p -- "$deps_dir"
deps_dir=$(cd -- "$deps_dir" && pwd)
exec 9>"$deps_dir/.prepare-deps.lock"
flock -n 9 || { echo "Another prepare-deps process is using $deps_dir" >&2; exit 2; }
mkdir -p "$deps_dir/.logs"
manifest=$(mktemp)
trap 'rm -f -- "$manifest"' EXIT

# Read the exact FetchContent names and revisions used by this checkout.
python3 - "$script_dir/CMakeLists.txt" >"$manifest" <<'PY'
import pathlib
import re
import shlex
import sys

source = pathlib.Path(sys.argv[1]).read_text()
rows = []
for block in re.findall(r"FetchContent_Declare\s*\((.*?)\)", source, re.S):
    fields = shlex.split(block, comments=True)
    if "GIT_REPOSITORY" not in fields:
        continue
    name = fields[0]
    url = fields[fields.index("GIT_REPOSITORY") + 1]
    ref = fields[fields.index("GIT_TAG") + 1]
    if not re.fullmatch(r"repo-[a-z0-9-]+", name):
        raise SystemExit(f"Unsupported dependency name: {name}")
    if any(c in url + ref for c in "\n\r\t;$") or ref.startswith("-"):
        raise SystemExit(f"Unsupported URL/revision for {name}")
    directory = "sgl-attn" if name == "repo-flash-attention" else name[5:]
    rows.append((name, url, ref, directory))
if not rows or len({r[0] for r in rows}) != len(rows):
    raise SystemExit("Expected nonempty, unique Git FetchContent declarations")
for row in rows:
    print("\t".join(row))
PY

prepare_one() (
  set -euo pipefail
  name=$1 url=$2 ref=$3 directory=$4
  dest="$deps_dir/$directory"
  if [[ -e $dest ]]; then
    # Never overwrite another checkout or silently discard local changes.
    [[ $(git -C "$dest" rev-parse --show-toplevel) == "$dest" ]] || {
      echo "Not a standalone Git checkout: $dest"; exit 1;
    }
    status=$(git -C "$dest" status --porcelain --untracked-files=all)
    # CMake's FetchContent clone may use --no-checkout. An empty worktree then
    # appears as every tracked file deleted, but there is no user change to
    # preserve. Treat it as an incomplete checkout and restore it below.
    worktree_entry=$(find "$dest" -mindepth 1 -maxdepth 1 ! -name .git -print -quit)
    index_path=$(git -C "$dest" rev-parse --git-path index)
    if [[ $index_path != /* ]]; then index_path="$dest/$index_path"; fi
    if [[ -n $status ]]; then
      if [[ -z $worktree_entry && ! -e $index_path ]]; then
        echo "Empty --no-checkout clone detected; checking out the pinned revision."
      else
        echo "Local changes found in $dest; leave them intact and resolve manually."; exit 1;
      fi
    fi
    [[ $(git -C "$dest" remote get-url origin) == "$url" ||
       $(git -C "$dest" remote get-url origin) == "${url%.git}.git" ||
       $(git -C "$dest" remote get-url origin) == "${url%.git}" ]] || {
      echo "Unexpected origin in $dest (expected $url)"; exit 1;
    }
  else
    git clone --progress --depth 1 -- "$url" "$dest"
  fi

  # Shallow-fetch the pinned revision instead of downloading full history.
  if [[ $ref =~ ^[0-9a-fA-F]{40}$ ]] && git -C "$dest" cat-file -e "$ref^{commit}" 2>/dev/null; then
    expected=$(git -C "$dest" rev-parse "$ref^{commit}")
  else
    git -C "$dest" fetch --progress --depth 1 origin "$ref"
    expected=$(git -C "$dest" rev-parse 'FETCH_HEAD^{commit}')
  fi
  git -C "$dest" checkout --detach "$expected"
  git -C "$dest" submodule sync --recursive
  git -C "$dest" submodule update --init --recursive --depth 1 --jobs 1 --progress

  [[ $(git -C "$dest" rev-parse HEAD) == "$expected" ]]
  submodules=$(git -C "$dest" submodule status --recursive)
  if printf '%s\n' "$submodules" | grep -Eq '^[-+U]'; then
    echo "Uninitialized or mismatched submodule:"; printf '%s\n' "$submodules"; exit 1
  fi
  [[ -z $(git -C "$dest" status --porcelain --untracked-files=all) ]]
  git -C "$dest" fsck --connectivity-only
  echo "VERIFIED $name $expected"
)

# Keep at most jobs repositories active. Logs include Git transfer progress.
pids=() names=()
failed=0
wait_one() {
  if wait "${pids[$1]}"; then
    echo "[OK] ${names[$1]}"
  else
    echo "[FAIL] ${names[$1]} -- see $deps_dir/.logs/${names[$1]}.log" >&2
    failed=1
  fi
}
index=0
while IFS=$'\t' read -r name url ref directory; do
  echo "[START] $name -> $ref"
  prepare_one "$name" "$url" "$ref" "$directory" >"$deps_dir/.logs/$name.log" 2>&1 &
  pids+=("$!") names+=("$name")
  if (( ${#pids[@]} - index >= jobs )); then
    wait_one "$index"
    index=$((index + 1))
  fi
done <"$manifest"
while (( index < ${#pids[@]} )); do
  wait_one "$index"
  index=$((index + 1))
done
if (( failed )); then
  echo "Some dependencies failed. Check logs, then rerun the same command." >&2
  exit 1
fi

# An absolute-path preload file avoids FetchContent relative-path ambiguity.
python3 - "$manifest" "$deps_dir" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[2])
lines = ["# Generated by prepare-deps.sh; sources checked against CMakeLists.txt."]
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    name, _, _, directory = line.split("\t")
    path = str(root / directory).replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")
    lines.append(f'set(FETCHCONTENT_SOURCE_DIR_{name.upper()} "{path}" CACHE PATH "Local dependency" FORCE)')
out = root / "local-deps.cmake"
out.write_text("\n".join(lines) + "\n")
print(f"All dependencies verified. CMake preload: {out}")
PY
echo "Append -C$deps_dir/local-deps.cmake to your existing CMAKE_ARGS."
