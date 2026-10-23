# Local validation scope

Validation of this pre-release extraction was performed on macOS arm64.
No hosted CI is configured. Keep the committed Cargo lockfile when repeating checks.

## Checks performed

- Rust formatting, strict workspace Clippy and default workspace tests.
- Workspace tests with `parallel,accelerate-blas`.
- Rust 1.89 workspace/all-target compilation.
- Python binding wheel build and installation in a separate Python 3.11 environment.
- Python / ASE / OpenMM tests: 31 passed, 11 explicitly skipped without external
  model fixtures; 42 passed with separately supplied reference models.
- LAMMPS serial plugin build and CLI parity for synthetic nonperiodic and
  periodic structures. Reference source commit:
  `611ca3b7f8525ba802373b04f9e3d632d515f7e3`.
- Native OpenMM C++ evaluator bridge: compile, unit and real-library link checks
  (2 tests passed). This is not a registered OpenMM Force implementation.
- MACE-MP-0b3 checkpoint conversion using MACE v0.3.16 and e3nn 0.4.4,
  followed by comparison against four pinned upstream reference structures.

## MACE-MP-0b3 comparison

The following are maximum absolute differences over two nonperiodic and two
periodic reference structures. Tolerances preserve the model's existing runtime
acceptance criteria; they are not universal tolerances for other models.

| Quantity | Maximum difference | Acceptance tolerance |
|---|---:|---:|
| Energy (eV) | 5.01e-6 | 1e-5 nonperiodic / 2e-5 periodic |
| Force component (eV/Å) | 3.17e-5 | 5e-5 |
| Stress component (eV/Å³) | 4.44e-9 | 5e-6 |
| Virial component (eV) | 1.04e-5 | 5e-3 |

`tools/python/check_model_parity.py` repeats this check given an external model
and reference JSON. Neither the foundation weights nor the external reference
bundle are included in this source distribution.

## Optional data and limitations

Six Rust tests are explicitly ignored because they require a local MACE-MH-1
state export/trace. Eleven Python tests similarly require external model fixtures;
set `LLMLIP_TEST_REFERENCE_ROOT` to a compatible reference-data directory to run
them. Default tests require no upstream Python packages or foundation weights.

Linux, Windows, MPI, Apple Metal, non-Accelerate BLAS backends, all named baked
model providers, and a full rerun of every upstream model family were not
validated in this extraction. Historical performance tables are not new
measurements of this candidate. Registry publication and release artifacts are
separate steps.
