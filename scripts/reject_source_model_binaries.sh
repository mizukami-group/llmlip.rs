#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

tracked_binaries="$(
  git ls-files \
    -- '*.model' '*.nequip.zip' '*.nqx' '*.pt' '*.pth' '*.safetensors' '*.ckpt' '*.npz' '*.bin' |
    while IFS= read -r path; do
      # Tensor payloads of converted .e3rs fixtures are allowed only under
      # fixtures/ (private monorepo) or examples/ (audited tiny fixtures;
      # see docs/MODEL_WEIGHT_REDISTRIBUTION_AUDIT.md). Public staging is
      # additionally gated by release_split/allowlists/public_weight_fixtures.txt.
      if [[ ("$path" == fixtures/* || "$path" == examples/*) &&
        ("$path" == *.e3rs/weights/*.f64.bin ||
          "$path" == *.e3rs/weights/*.f32.bin) ]]; then
        continue
      fi
      printf '%s\n' "$path"
    done
)"

if [[ -z "$tracked_binaries" ]]; then
  echo "No tracked source model binaries found."
  exit 0
fi

cat >&2 <<'MSG'
error: tracked source model binaries are not allowed in this repository.

Upstream .model/.nequip.zip/.nqx/.pt/.pth/.safetensors/.ckpt/.npz files and raw
.bin files must stay in ignored local caches such as outputs/reference-artifacts/
and be represented by committed manifests, fixtures, hashes, or exported .e3rs
tensor payloads. Only .e3rs/weights/*.f64.bin and .e3rs/weights/*.f32.bin tensor
payloads under fixtures/ or examples/ are allowlisted, and each committed
payload needs an entry in docs/MODEL_WEIGHT_REDISTRIBUTION_AUDIT.md.

Tracked binary paths:
MSG

while IFS= read -r path; do
  printf '  %s\n' "$path" >&2
done <<< "$tracked_binaries"

exit 1
