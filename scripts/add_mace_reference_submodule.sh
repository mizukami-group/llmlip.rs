#!/usr/bin/env bash
set -euo pipefail

if ! ROOT="$(git rev-parse --show-toplevel 2>&1)"; then
  echo "error: run this inside an initialized git repository" >&2
  echo "failed command: git rev-parse --show-toplevel" >&2
  echo "output:" >&2
  echo "$ROOT" >&2
  echo "hint: git init && git add . && git commit -m 'Initial spec pack'" >&2
  exit 1
fi

cd "$ROOT"

mkdir -p third_party tools/reference fixtures/mace

is_registered_submodule() {
  [ -f .gitmodules ] \
    && git config --file .gitmodules --get-regexp '^submodule\..*\.path$' \
      | awk '$2 == "third_party/mace" { found = 1 } END { exit !found }'
}

if [ -e third_party/mace ] \
  && ! git -C third_party/mace rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: third_party/mace exists but is not a git checkout or submodule" >&2
  echo "remove or move it before adding the pinned reference implementation" >&2
  exit 1
fi

if git -C third_party/mace rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if ! is_registered_submodule; then
    echo "error: third_party/mace exists as a git checkout but is not registered in .gitmodules" >&2
    echo "move it aside and rerun this script so Git can add it as a pinned submodule" >&2
    exit 1
  fi
  echo "third_party/mace already exists as a registered submodule"
else
  git submodule add https://github.com/ACEsuit/mace.git third_party/mace
fi

MACE_REF="${MACE_REF:-v0.3.16}"
git -C third_party/mace fetch --tags
git -C third_party/mace checkout "$MACE_REF"
PINNED_COMMIT="$(git -C third_party/mace rev-parse HEAD)"
printf '%s\n' "$PINNED_COMMIT" > third_party/mace.PINNED_COMMIT.tmp
mv third_party/mace.PINNED_COMMIT.tmp third_party/mace.PINNED_COMMIT

cat <<MSG
Pinned MACE reference implementation:
  ref:    $MACE_REF
  commit: $PINNED_COMMIT

Next:
  git add .gitmodules third_party/mace third_party/mace.PINNED_COMMIT
  git commit -m "Add pinned MACE reference implementation"
MSG
