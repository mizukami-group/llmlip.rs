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
work_dir="${repo_root}/target/lammps-llmlip-synthetic-pbc-parity"
input="${work_dir}/in.synthetic_pbc"
structure="${work_dir}/synthetic_pbc.extxyz"
lammps_log="${work_dir}/lammps.out"
lammps_forces="${work_dir}/lammps_forces.dump"
cli_forces="${work_dir}/cli_forces.extxyz"
cli_stdout="${work_dir}/cli.out"

if [[ ! -x "${lmp}" ]]; then
  echo "LAMMPS executable not found: ${lmp}" >&2
  exit 2
fi

if [[ ! -f "${plugin}" ]]; then
  echo "llmlip LAMMPS plugin not found: ${plugin}" >&2
  exit 2
fi

mkdir -p "${work_dir}"

cat > "${structure}" <<'EOF'
4
Lattice="12 0 0 0 12 0 0 0 12" Properties=species:S:1:pos:R:3 pbc="T T T"
H 0.20 0.25 0.30
O 11.55 0.25 0.30
H 4.75 5.10 5.20
O 5.65 5.60 5.35
EOF

cat > "${input}" <<EOF
units metal
atom_style atomic
boundary p p p

region box block 0 12 0 12 0 12
create_box 2 box
create_atoms 1 single 0.20 0.25 0.30
create_atoms 2 single 11.55 0.25 0.30
create_atoms 1 single 4.75 5.10 5.20
create_atoms 2 single 5.65 5.60 5.35

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
_, lammps_energy, lammps_fmax = thermo_line.split()[:3]
lammps_energy = float(lammps_energy)
lammps_fmax = float(lammps_fmax)

with open(cli_stdout, encoding="utf-8") as handle:
    text = handle.read()
match = re.search(r"energy\s*=\s*([-+0-9.eE]+)", text)
if match is None:
    raise SystemExit("could not find CLI energy")
cli_energy = float(match.group(1))

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

def read_extxyz_forces(path):
    forces = []
    with open(path, encoding="utf-8") as handle:
        for line in handle.readlines()[2:]:
            fields = line.split()
            forces.append(tuple(map(float, fields[-3:])))
    return forces

lammps_force_rows = read_lammps_dump(lammps_forces)
cli_force_rows = read_extxyz_forces(cli_forces)
if len(lammps_force_rows) != len(cli_force_rows):
    raise SystemExit(f"force row count mismatch: {len(lammps_force_rows)} vs {len(cli_force_rows)}")

cli_fmax = 0.0
for atom, (lf, cf) in enumerate(zip(lammps_force_rows, cli_force_rows), start=1):
    cli_fmax = max(cli_fmax, math.sqrt(sum(value * value for value in cf)))
    for axis, (left, right) in enumerate(zip(lf, cf)):
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-8):
            raise SystemExit(
                f"force mismatch atom {atom} axis {axis}: LAMMPS={left:.17g} CLI={right:.17g}"
            )

if not math.isclose(lammps_energy, cli_energy, rel_tol=0.0, abs_tol=1e-8):
    raise SystemExit(f"energy mismatch: LAMMPS={lammps_energy:.17g} CLI={cli_energy:.17g}")
if not math.isclose(lammps_fmax, cli_fmax, rel_tol=0.0, abs_tol=1e-8):
    raise SystemExit(f"fmax mismatch: LAMMPS={lammps_fmax:.17g} CLI={cli_fmax:.17g}")

print(f"synthetic PBC parity ok: energy={lammps_energy:.17g} fmax={lammps_fmax:.17g}")
PY
