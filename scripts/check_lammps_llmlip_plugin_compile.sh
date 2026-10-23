#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cxx="${CXX:-c++}"

"${cxx}" -std=c++17 -fsyntax-only \
  -I"${repo_root}/lammps_plugin/llmlip" \
  -I"${repo_root}/third_party/lammps/src" \
  -I"${repo_root}/third_party/lammps/src/STUBS" \
  -I"${repo_root}/crates/llmlip-capi/include" \
  "${repo_root}/lammps_plugin/llmlip/pair_llmlip.cpp" \
  "${repo_root}/lammps_plugin/llmlip/lammpsplugin_registration.cpp"
