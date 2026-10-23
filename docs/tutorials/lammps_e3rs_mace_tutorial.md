# LAMMPS llmlip/MACE Interface Tutorial

This tutorial explains how to build and use the LAMMPS `pair_style llmlip`
plugin. It is written for users who have little or no LAMMPS experience.

The current interface runs Rust `.e3rs` fixed-weight MLIP models from LAMMPS.
The most important supported use case in this branch is MACE-lite/MACE-Osaka
style `.e3rs` inference for energies and conservative forces.

Important terminology:

- **LAMMPS** is the molecular dynamics driver.
- **`pair_style llmlip`** is the LAMMPS pair style that calls the Rust inference
  runtime.
- **`.e3rs` model** is the Rust-readable exported model directory.
- **MACE** is the model family. This plugin does not load raw upstream MACE
  checkpoints directly; use an exported `.e3rs` model.
- If you have a raw MAISE or other potential file, it must first be converted
  into a supported `.e3rs` model format. Raw MAISE files are not loaded by this
  plugin.

## 1. What You Will Build

You need three pieces:

1. A LAMMPS executable built with plugin support.
2. The Rust C ABI library, `libllmlip_capi`.
3. The out-of-tree LAMMPS plugin, `liblammps_llmlipplugin`.

At runtime, LAMMPS loads the plugin:

```lammps
plugin load /path/to/liblammps_llmlipplugin.dylib
```

Then the input script selects the Rust model:

```lammps
pair_style llmlip model /path/to/model.e3rs dtype f32 virial fdotr
pair_coeff * * Si
```

On Linux the plugin suffix is usually `.so`. On macOS it is `.dylib`.

## 2. Prerequisites

Install:

- Rust toolchain with `cargo`.
- CMake.
- A C++ compiler.
- Git.
- Python 3 for helper benchmark scripts.
- Optional: MPI and `mpirun` if you want multi-rank LAMMPS.

On macOS, the recommended Rust build uses Apple's Accelerate BLAS:

```bash
cargo build -p llmlip-capi --release --no-default-features --features lammps,accelerate-blas,parallel
```

On Linux x86-64, use one supported BLAS provider, for example:

```bash
cargo build -p llmlip-capi --release --no-default-features --features lammps,openblas,parallel
```

Do not enable multiple BLAS providers at once.

## 3. Get the Repository and LAMMPS Source

From the repository root:

```bash
git submodule update --init --recursive
```

If `third_party/lammps` is not present yet, add the pinned LAMMPS checkout:

```bash
scripts/add_lammps_reference_submodule.sh
```

This keeps LAMMPS inside the repository under `third_party/lammps`, so the
plugin build can always find the same source tree.

## 4. Build a Serial LAMMPS Executable

For a first run, build serial LAMMPS without MPI:

```bash
cmake -S third_party/lammps/cmake \
  -B target/lammps-build \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_MPI=off \
  -D LAMMPS_EXCEPTIONS=on \
  -D LAMMPS_INSTALL_RPATH=on \
  -D PKG_PLUGIN=on

cmake --build target/lammps-build --parallel 8
```

The executable should be:

```bash
target/lammps-build/lmp
```

Check it:

```bash
target/lammps-build/lmp -help | head
```

## 5. Build the Rust C ABI

On macOS:

```bash
cargo build -p llmlip-capi --release --no-default-features --features lammps,accelerate-blas,parallel
```

On Linux x86-64 with OpenBLAS:

```bash
cargo build -p llmlip-capi --release --no-default-features --features lammps,openblas,parallel
```

The static library should be:

```bash
target/release/libllmlip_capi.a
```

## 6. Build the LAMMPS llmlip Plugin

Build the plugin against the pinned LAMMPS source and the Rust C ABI:

```bash
cmake -S lammps_plugin/llmlip \
  -B target/lammps-llmlip-plugin-release \
  -D CMAKE_BUILD_TYPE=Release \
  -D LAMMPS_SOURCE_DIR="$PWD/third_party/lammps" \
  -D LLMLIP_CAPI_INCLUDE_DIR="$PWD/crates/llmlip-capi/include" \
  -D LLMLIP_CAPI_LIBRARY="$PWD/target/release/libllmlip_capi.a"

cmake --build target/lammps-llmlip-plugin-release --parallel 8
```

Expected output:

```text
target/lammps-llmlip-plugin-release/liblammps_llmlipplugin.dylib  # macOS
target/lammps-llmlip-plugin-release/liblammps_llmlipplugin.so     # Linux
```

## 7. Minimal Single-Point Example

Create a file named `in.e3rs_si`:

```lammps
units metal
atom_style atomic
boundary p p p

lattice diamond 5.43
region box block 0 4 0 4 0 4 units lattice
create_box 1 box
create_atoms 1 box
mass 1 28.0855

newton on
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes

plugin load /ABS/PATH/TO/e3nn-core-rs/target/lammps-llmlip-plugin-release/liblammps_llmlipplugin.dylib

pair_style llmlip model /ABS/PATH/TO/e3nn-core-rs/fixtures/mace/v0_3_16_4d2da09/models/mace_osaka24_medium_runtime.e3rs dtype f32 virial fdotr
pair_coeff * * Si

thermo 1
thermo_style custom step atoms pe fmax
run 0
```

Replace `/ABS/PATH/TO/e3nn-core-rs` with your repository path. On Linux, use
`liblammps_llmlipplugin.so`.

Run it:

```bash
target/lammps-build/lmp -in in.e3rs_si
```

`run 0` performs a single energy/force evaluation without integrating dynamics.
This is the safest first test.

## 8. Minimal MD Example

After the single-point example works, add a time integrator:

```lammps
velocity all create 300.0 12345 mom yes rot yes dist gaussian
fix int all nve
timestep 0.001
run 10
```

Full example:

```lammps
units metal
atom_style atomic
boundary p p p

lattice diamond 5.43
region box block 0 4 0 4 0 4 units lattice
create_box 1 box
create_atoms 1 box
mass 1 28.0855

newton on
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes

plugin load /ABS/PATH/TO/e3nn-core-rs/target/lammps-llmlip-plugin-release/liblammps_llmlipplugin.dylib
pair_style llmlip model /ABS/PATH/TO/e3nn-core-rs/fixtures/mace/v0_3_16_4d2da09/models/mace_osaka24_medium_runtime.e3rs dtype f32 virial fdotr
pair_coeff * * Si

velocity all create 300.0 12345 mom yes rot yes dist gaussian
fix int all nve
timestep 0.001

thermo 1
thermo_style custom step temp pe etotal fmax
run 10
```

This is only a smoke test. It is not a recommendation for production MD
settings.

## 9. Understanding the LAMMPS Commands

### `units metal`

The plugin currently expects LAMMPS `metal` units:

- energy: eV
- distance: Angstrom
- force: eV/Angstrom
- time: ps

### `atom_style atomic`

Use this for simple atomic systems without bonds, angles, charges, or molecule
IDs.

### `boundary p p p`

This requests full 3D periodic boundary conditions.

### `newton on`

Required by the plugin. It controls how LAMMPS communicates force
contributions, especially for multi-rank runs.

### `plugin load ...`

Loads the out-of-tree `pair_style llmlip` plugin. The path must point to the
compiled plugin shared library.

### `pair_style llmlip ...`

Selects the Rust MLIP model and runtime options.

Common options:

```lammps
pair_style llmlip model MODEL_PATH dtype f64 virial fdotr
pair_style llmlip model MODEL_PATH dtype f32 virial fdotr
```

`dtype f32` uses optimized FP32 MACE-lite kernels when the model supports them.
The LAMMPS-facing energy, force, virial, and stress buffers remain double
precision.

### `pair_coeff * * Si`

Maps LAMMPS atom types to element symbols. For one atom type:

```lammps
pair_coeff * * Si
```

For two atom types, for example H and O:

```lammps
pair_coeff * * H O
```

The number of element names after `* *` must match the number of LAMMPS atom
types.

## 10. Rank-1 and MPI Behavior

For one MPI rank, the default path is compact:

```lammps
pair_style llmlip model MODEL_PATH dtype f32 virial fdotr
```

This sends only the local structure to Rust and lets Rust build the normal PBC
graph. It avoids the ghost-expanded debug graph and is much faster for rank 1.

To debug the MPI-style ghost/image-node handoff on one rank:

```lammps
pair_style llmlip model MODEL_PATH dtype f32 rank1_mode external_graph virial fdotr
```

Accepted rank-1 mode aliases:

- `rank1_mode structure`
- `rank1_mode compact`
- `rank1_mode external_graph`
- `rank1_mode ghost_graph`

For multi-rank MPI, the plugin always uses the external graph path because each
rank needs ghost/image atoms from neighboring ranks.

## 11. Optional MPI Build

Build MPI-enabled LAMMPS:

```bash
cmake -S third_party/lammps/cmake \
  -B target/lammps-mpi-build \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_MPI=on \
  -D LAMMPS_EXCEPTIONS=on \
  -D LAMMPS_INSTALL_RPATH=on \
  -D PKG_PLUGIN=on

cmake --build target/lammps-mpi-build --parallel 8
```

Run with four ranks:

```bash
mpirun -np 4 target/lammps-mpi-build/lmp -in in.e3rs_si
```

Current limitation: multi-rank MPI uses the ghost-expanded external graph path.
It is correct-oriented but not yet the final optimized strong-scaling design.

## 12. Verification and Benchmark Helpers

Useful local checks:

```bash
scripts/run_lammps_llmlip_mace_lite_pbc_single_rank_parity.sh
scripts/run_lammps_llmlip_mace_lite_pbc_mpi_parity.sh
scripts/run_lammps_llmlip_mace_lite_pbc_virial_fd_parity.sh
scripts/run_lammps_llmlip_cutoff_contract_parity.sh
```

Si PBC scaling benchmark:

```bash
RAYON_NUM_THREADS=16 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
MKL_NUM_THREADS=1 \
LLMLIP_SKIP_BUILD=1 \
scripts/benchmark_lammps_llmlip_si_pbc_scaling.sh
```

To force the old rank-1 ghost-expanded debug path:

```bash
LLMLIP_LAMMPS_SI_PBC_RANK1_MODE=external_graph \
scripts/benchmark_lammps_llmlip_si_pbc_scaling.sh
```

## 13. Troubleshooting

### LAMMPS says `Invalid command: plugin`

Your LAMMPS executable was not built with plugin support. Rebuild with:

```bash
-D PKG_PLUGIN=on
```

### LAMMPS cannot load the plugin

Check the plugin path and suffix:

```bash
ls target/lammps-llmlip-plugin-release/
```

Use `.dylib` on macOS and `.so` on Linux.

### `Pair style llmlip requires newton pair on`

Add:

```lammps
newton on
```

before `pair_style`.

### `Incorrect args for pair coefficients`

Make sure the number of element names matches the number of LAMMPS atom types:

```lammps
create_box 1 box
pair_coeff * * Si
```

or:

```lammps
create_box 2 box
pair_coeff * * H O
```

### `Requested pair_style llmlip dtype f32, but model does not support it`

Use `dtype f64`, or use a model that advertises FP32 support.

### MPI run is much slower than rank 1

This can happen because multi-rank currently uses the ghost-expanded external
graph path. The rank-1 compact path is optimized separately and is not the same
as the current MPI path.

## 14. Current Limitations

- Runtime Python is not required, but model export/conversion is a separate
  workflow.
- Raw upstream MACE checkpoints are not loaded directly by LAMMPS. Use exported
  `.e3rs` models.
- Raw MAISE potential files are not supported unless converted into a supported
  `.e3rs` model.
- Per-atom energy and per-atom virial are not implemented.
- Multi-rank MPI is correctness-oriented and still uses the ghost-expanded graph
  path.
- A future rank-1 MD cache should reuse neighbor candidates with a skin
  distance, similar to a Verlet list, to reduce the remaining graph-preparation
  overhead.
