#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/clippy_changed_lines.sh [--base REF] [--target REF]

Runs cargo clippy with JSON output and fails only for clippy diagnostics whose
primary span intersects Rust lines changed between REF...TARGET.

Environment:
  CARGO       Cargo executable. Defaults to cargo.
  BASE_REF    Base commit or ref. Defaults to origin/$GITHUB_BASE_REF,
              origin/main, main, then master.
  TARGET_REF  Target commit or ref. Defaults to HEAD.
USAGE
}

cargo_bin="${CARGO:-cargo}"
base_ref="${BASE_REF:-}"
target_ref="${TARGET_REF:-HEAD}"

while (($#)); do
  case "$1" in
    --base)
      if (($# < 2)); then
        echo "error: --base requires a ref" >&2
        exit 2
      fi
      base_ref="$2"
      shift 2
      ;;
    --target)
      if (($# < 2)); then
        echo "error: --target requires a ref" >&2
        exit 2
      fi
      target_ref="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v "$cargo_bin" >/dev/null 2>&1; then
  echo "error: cargo is required; set CARGO to the cargo executable if it is not on PATH" >&2
  exit 127
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required to parse clippy JSON diagnostics" >&2
  exit 127
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

ref_exists() {
  git rev-parse --verify --quiet "$1^{commit}" >/dev/null
}

if [[ -z "$base_ref" ]]; then
  if [[ -n "${GITHUB_BASE_REF:-}" ]] && ref_exists "origin/${GITHUB_BASE_REF}"; then
    base_ref="origin/${GITHUB_BASE_REF}"
  elif ref_exists origin/main; then
    base_ref="origin/main"
  elif ref_exists main; then
    base_ref="main"
  elif ref_exists master; then
    base_ref="master"
  else
    echo "error: BASE_REF is not set and no default base ref was found" >&2
    exit 2
  fi
fi

if ! ref_exists "$base_ref"; then
  echo "error: base ref '$base_ref' is not available locally" >&2
  exit 2
fi

if ! ref_exists "$target_ref"; then
  echo "error: target ref '$target_ref' is not available locally" >&2
  exit 2
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

diff_file="$tmp_dir/diff.patch"
ranges_file="$tmp_dir/changed-ranges.tsv"
diagnostics_file="$tmp_dir/clippy.json"
diagnostics_tsv="$tmp_dir/clippy.tsv"
violations_file="$tmp_dir/violations.txt"

git diff --unified=0 --no-ext-diff --no-color --diff-filter=ACMR \
  "$base_ref...$target_ref" -- '*.rs' > "$diff_file"

awk '
  /^\+\+\+ b\// {
    file = substr($0, 7)
    next
  }
  /^@@ / {
    line = $0
    sub(/^@@ [^+]*\+/, "", line)
    sub(/ .*/, "", line)
    split(line, parts, ",")
    start = parts[1] + 0
    count = (parts[2] == "" ? 1 : parts[2] + 0)
    if (count > 0 && file != "") {
      print file "\t" start "\t" (start + count - 1)
    }
  }
' "$diff_file" > "$ranges_file"

if [[ ! -s "$ranges_file" ]]; then
  echo "No changed Rust lines; skipping changed-line clippy gate."
  exit 0
fi

set +e
"$cargo_bin" clippy --workspace --all-targets --message-format=json --color=never > "$diagnostics_file"
clippy_status=$?
set -e

if [[ $clippy_status -ne 0 && ! -s "$diagnostics_file" ]]; then
  echo "error: cargo clippy failed before emitting JSON diagnostics" >&2
  exit "$clippy_status"
fi

jq -r '
  select(.reason == "compiler-message")
  | .message as $message
  | ($message.code.code // "") as $code
  | select($code | startswith("clippy::"))
  | $message.spans[]?
  | select(.is_primary == true)
  | select((.file_name // "") | endswith(".rs"))
  | [
      ($message.level // ""),
      $code,
      (.file_name // ""),
      (.line_start // 0),
      (.line_end // .line_start // 0),
      (($message.message // "") | gsub("[\r\n\t]+"; " "))
    ]
  | @tsv
' "$diagnostics_file" > "$diagnostics_tsv"

awk -F '\t' -v root="$repo_root" '
  FNR == NR {
    ranges[$1] = ranges[$1] $2 "\t" $3 "\n"
    next
  }
  {
    level = $1
    code = $2
    file = $3
    start = $4 + 0
    end = $5 + 0
    message = $6

    if (index(file, root "/") == 1) {
      file = substr(file, length(root) + 2)
    }

    if (!(file in ranges)) {
      next
    }

    split(ranges[file], file_ranges, "\n")
    for (idx in file_ranges) {
      if (file_ranges[idx] == "") {
        continue
      }
      split(file_ranges[idx], bounds, "\t")
      first = bounds[1] + 0
      last = bounds[2] + 0
      if (start <= last && end >= first) {
        printf "%s:%d:%d: %s [%s] %s\n", file, start, end, level, code, message
        found = 1
        break
      }
    }
  }
  END {
    exit found ? 1 : 0
  }
' "$ranges_file" "$diagnostics_tsv" > "$violations_file" || {
  echo "Clippy diagnostics found on changed lines:"
  cat "$violations_file"
  exit 1
}

if [[ $clippy_status -ne 0 ]]; then
  echo "error: cargo clippy failed even though no clippy diagnostics intersected changed lines" >&2
  echo "Run '${cargo_bin} clippy --workspace --all-targets' for the full compiler output." >&2
  exit "$clippy_status"
fi

echo "No clippy diagnostics on changed lines."
