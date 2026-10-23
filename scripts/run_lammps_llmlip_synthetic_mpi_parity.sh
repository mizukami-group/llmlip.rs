#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" == "Darwin" ]]; then
  plugin_suffix="dylib"
else
  plugin_suffix="so"
fi

mpirun_bin="${MPIRUN:-}"
if [[ -z "${mpirun_bin}" ]]; then
  mpirun_bin="$(command -v mpirun || command -v mpiexec || true)"
fi

lmp="${LMP_MPI:-${repo_root}/target/lammps-mpi-build/lmp}"
plugin="${LLMLIP_LAMMPS_PLUGIN:-${repo_root}/target/lammps-llmlip-plugin/liblammps_llmlipplugin.${plugin_suffix}}"
model="${LLMLIP_MODEL:-${repo_root}/examples/model.e3rs}"
work_dir="${repo_root}/target/lammps-llmlip-synthetic-mpi-parity"
input="${work_dir}/in.synthetic_mpi"
structure="${work_dir}/synthetic_mpi.extxyz"
cli_forces="${work_dir}/cli_forces.extxyz"
cli_stdout="${work_dir}/cli.out"

if [[ -z "${mpirun_bin}" ]]; then
  echo "mpirun/mpiexec not found; install MPI or set MPIRUN" >&2
  exit 2
fi

if [[ ! -x "${lmp}" ]]; then
  echo "MPI-enabled LAMMPS executable not found: ${lmp}" >&2
  echo "Build one with BUILD_MPI=on and PKG_PLUGIN=on, then set LMP_MPI if needed." >&2
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
H 0.4 0.5 0.6
O 1.6 0.5 0.6
H 3.8 4.1 4.2
O 6.9 7.2 7.1
EOF

cat > "${input}" <<EOF
units metal
atom_style atomic
boundary p p p

region box block 0 12 0 12 0 12
create_box 2 box
create_atoms 1 single 0.4 0.5 0.6
create_atoms 2 single 1.6 0.5 0.6
create_atoms 1 single 3.8 4.1 4.2
create_atoms 2 single 6.9 7.2 7.1

mass 1 1.008
mass 2 15.999

newton on
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes

plugin load ${plugin}
pair_style llmlip model ${model} dtype f64 virial fdotr
pair_coeff * * H O

thermo_style custom step pe fmax pxx pyy pzz pxy pxz pyz
thermo_modify format float %.17g
run 0
EOF

logs=()
for ranks in 1 2 4; do
  log="${work_dir}/ranks_${ranks}.out"
  "${mpirun_bin}" -np "${ranks}" "${lmp}" -log "${work_dir}/ranks_${ranks}.log.lammps" -in "${input}" > "${log}"
  logs+=("${ranks}:${log}")
done

cargo run -q -p llmlip-cli -- predict \
  --model "${model}" \
  --structure "${structure}" \
  --energy \
  --forces-out "${cli_forces}" \
  > "${cli_stdout}"

python3 - "$cli_stdout" "$cli_forces" "${logs[@]}" <<'PY'
import math
import re
import sys

cli_stdout, cli_forces = sys.argv[1:3]

def read_thermo(path):
    thermo = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if re.match(r"^\s*0\s+[-+0-9.eE]+", line):
                thermo = [float(value) for value in line.split()[:9]]
    if thermo is None:
        raise SystemExit(f"could not find LAMMPS thermo line in {path}")
    keys = ["step", "pe", "fmax", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz"]
    return dict(zip(keys, thermo))

records = []
for item in sys.argv[3:]:
    ranks, path = item.split(":", 1)
    records.append((int(ranks), read_thermo(path)))

base_ranks, base = records[0]

with open(cli_stdout, encoding="utf-8") as handle:
    cli_text = handle.read()
match = re.search(r"energy\s*=\s*([-+0-9.eE]+)", cli_text)
if match is None:
    raise SystemExit("could not find CLI energy")
cli_energy = float(match.group(1))

cli_fmax = 0.0
with open(cli_forces, encoding="utf-8") as handle:
    for line in handle.readlines()[2:]:
        fields = line.split()
        fx, fy, fz = map(float, fields[-3:])
        cli_fmax = max(cli_fmax, math.sqrt(fx * fx + fy * fy + fz * fz))

for key, expected, atol in [
    ("pe", cli_energy, 1e-8),
    ("fmax", cli_fmax, 1e-8),
]:
    if not math.isclose(base[key], expected, rel_tol=0.0, abs_tol=atol):
        raise SystemExit(
            f"rank-1 LAMMPS vs CLI {key} mismatch: "
            f"{base[key]:.17g} vs {expected:.17g}"
        )

for ranks, record in records[1:]:
    for key, atol in [
        ("pe", 1e-8),
        ("fmax", 1e-8),
        ("pxx", 1e-7),
        ("pyy", 1e-7),
        ("pzz", 1e-7),
        ("pxy", 1e-7),
        ("pxz", 1e-7),
        ("pyz", 1e-7),
    ]:
        if not math.isclose(base[key], record[key], rel_tol=0.0, abs_tol=atol):
            raise SystemExit(
                f"{key} mismatch for ranks {base_ranks} vs {ranks}: "
                f"{base[key]:.17g} vs {record[key]:.17g}"
            )

print(
    "synthetic MPI parity ok: "
    f"energy={base['pe']:.17g} fmax={base['fmax']:.17g} "
    f"pxx={base['pxx']:.17g} ranks={[r for r, _ in records]}"
)
PY
