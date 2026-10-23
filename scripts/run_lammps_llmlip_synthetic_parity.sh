#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" == "Darwin" ]]; then
  plugin_suffix="dylib"
else
  plugin_suffix="so"
fi

lmp="${LMP:-${repo_root}/target/lammps-build/lmp}"
plugin="${LLMLIP_LAMMPS_PLUGIN:-${repo_root}/target/lammps-llmlip-plugin/liblammps_llmlipplugin.${plugin_suffix}}"
model="${LLMLIP_MODEL:-${repo_root}/examples/model.e3rs}"
structure="${LLMLIP_STRUCTURE:-${repo_root}/examples/lammps/synthetic_two_atom.extxyz}"
work_dir="${repo_root}/target/lammps-llmlip-synthetic-parity"
input="${work_dir}/in.e3rs_synthetic"
lammps_log="${work_dir}/lammps.out"
lammps_forces="${work_dir}/lammps_forces.dump"
cli_forces="${work_dir}/cli_forces.extxyz"
cli_stdout="${work_dir}/cli.out"

if [[ ! -x "${lmp}" ]]; then
  echo "LAMMPS executable not found: ${lmp}" >&2
  echo "Build one with: cmake -S third_party/lammps/cmake -B target/lammps-build -D BUILD_MPI=off -D PKG_PLUGIN=on && cmake --build target/lammps-build" >&2
  exit 2
fi

if [[ ! -f "${plugin}" ]]; then
  echo "llmlip LAMMPS plugin not found: ${plugin}" >&2
  echo "Build it with: cargo build -p llmlip-capi --no-default-features --features lammps && cmake -S lammps_plugin/llmlip -B target/lammps-llmlip-plugin -DLLMLIP_CAPI_LIBRARY=${repo_root}/target/debug/libllmlip_capi.a && cmake --build target/lammps-llmlip-plugin" >&2
  exit 2
fi

mkdir -p "${work_dir}"

cat > "${input}" <<EOF
units metal
atom_style atomic
boundary f f f

region box block -5 5 -5 5 -5 5
create_box 2 box
create_atoms 1 single 0.0 0.0 0.0
create_atoms 2 single 0.0 0.0 1.2

mass 1 1.008
mass 2 15.999

newton on
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes

plugin load ${plugin}
pair_style llmlip model ${model} dtype f64 virial fdotr
pair_coeff * * H O

dump forces all custom 1 ${lammps_forces} id type fx fy fz
dump_modify forces sort id format float %.17g first yes

thermo_style custom step pe fmax
thermo_modify format float %.17g
run 0
EOF

"${lmp}" -log "${work_dir}/log.lammps" -in "${input}" > "${lammps_log}"

cargo run -q -p llmlip-cli -- predict \
  --model "${model}" \
  --structure "${structure}" \
  --energy \
  --forces-out "${cli_forces}" \
  > "${cli_stdout}"

python3 - "$lammps_log" "$lammps_forces" "$cli_stdout" "$cli_forces" <<'PY'
import math
import re
import sys

lammps_log, lammps_forces, cli_stdout, cli_forces = sys.argv[1:]

thermo_line = None
with open(lammps_log, encoding="utf-8") as handle:
    for line in handle:
        if re.match(r"^\s*0\s+[-+0-9.eE]+\s+[-+0-9.eE]+", line):
            thermo_line = line.strip()

if thermo_line is None:
    raise SystemExit("could not find LAMMPS thermo line")

_, lammps_energy, _lammps_thermo_fmax = thermo_line.split()[:3]
lammps_energy = float(lammps_energy)

with open(cli_stdout, encoding="utf-8") as handle:
    text = handle.read()
match = re.search(r"energy\s*=\s*([-+0-9.eE]+)", text)
if match is None:
    raise SystemExit("could not find CLI energy")
cli_energy = float(match.group(1))

cli_fmax = 0.0
cli_force_rows = []
with open(cli_forces, encoding="utf-8") as handle:
    lines = handle.readlines()[2:]
for line in lines:
    fields = line.split()
    fx, fy, fz = map(float, fields[-3:])
    cli_force_rows.append((fx, fy, fz))
    cli_fmax = max(cli_fmax, math.sqrt(fx * fx + fy * fy + fz * fz))

def read_lammps_dump(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()
    for index, line in enumerate(lines):
        if line.startswith("ITEM: ATOMS"):
            for row in lines[index + 1 :]:
                if row.startswith("ITEM:"):
                    break
                fields = row.split()
                if fields:
                    rows.append((int(fields[0]), tuple(map(float, fields[2:5]))))
            break
    if not rows:
        raise SystemExit(f"no ATOMS rows in {path}")
    return [force for _, force in sorted(rows)]

lammps_force_rows = read_lammps_dump(lammps_forces)
if len(lammps_force_rows) != len(cli_force_rows):
    raise SystemExit(f"force row count mismatch: {len(lammps_force_rows)} vs {len(cli_force_rows)}")
lammps_dump_fmax = max(math.sqrt(sum(value * value for value in force)) for force in lammps_force_rows)

if not math.isclose(lammps_energy, cli_energy, rel_tol=0.0, abs_tol=1e-8):
    raise SystemExit(f"energy mismatch: LAMMPS={lammps_energy:.17g} CLI={cli_energy:.17g}")
if not math.isclose(lammps_dump_fmax, cli_fmax, rel_tol=0.0, abs_tol=1e-8):
    raise SystemExit(f"fmax mismatch: LAMMPS={lammps_dump_fmax:.17g} CLI={cli_fmax:.17g}")
for atom, (lf, cf) in enumerate(zip(lammps_force_rows, cli_force_rows), start=1):
    for axis, (left, right) in enumerate(zip(lf, cf)):
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-8):
            raise SystemExit(
                f"force mismatch atom {atom} axis {axis}: LAMMPS={left:.17g} CLI={right:.17g}"
            )

print(f"synthetic parity ok: energy={lammps_energy:.17g} fmax={lammps_dump_fmax:.17g}")
PY
