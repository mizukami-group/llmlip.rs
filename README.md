# llmlip.rs

[日本語](README.ja.md)

A Rust inference engine for fixed-weight machine-learning interatomic potentials.
It predicts energies and conservative forces, with stress and virial support
for selected architectures. The Rust library, CLI and C ABI do not require
Python or PyTorch at runtime. Python is required for the Python/ASE/OpenMM
adapters and for upstream checkpoint conversion.

This is a pre-release source tree prepared for source-only publication.
No prebuilt CLI, wheel, Python sdist, native plugin, container or foundation-model
weights are part of this distribution. Local build instructions are for users
to build their own copies. See [distribution policy](docs/distribution_policy.md).
The included synthetic model is for examples and interface tests, not scientific use.

## Build and run

Use Rust 1.89 or later and a native C/C++ build toolchain. Keep `Cargo.lock`:

```bash
cargo build --locked -p llmlip-cli --release
./target/release/llmlip predict \
  --model examples/model.e3rs --structure examples/input.extxyz \
  --energy --forces --json
```

The equivariant primitives come from [e3nn-core-rs](https://github.com/mizukami-group/e3nn-core-rs),
pinned to the v0.1.0 commit `9da369cf1d854da6af1a46928bc953c7966e9382`.
Initial builds require network access to fetch Cargo dependencies.

## Interfaces

| Interface | Documentation |
|---|---|
| Rust and CLI | [CLI](docs/cli.md), `crates/llmlip` |
| Python / ASE | [Python interface](docs/PYTHON_ASE_INTERFACE_SPEC.md) |
| OpenMM / OpenMM-ML | [OpenMM](docs/openmm.md) |
| LAMMPS | [LAMMPS tutorial](docs/tutorials/lammps_e3rs_mace_tutorial.md) |
| C ABI | [C ABI](docs/c_abi.md) |

For Python, create and activate a virtual environment, then run:

```bash
pip install maturin numpy ase pytest
maturin develop --manifest-path crates/llmlip-py/Cargo.toml
```

The OpenMM adapter additionally needs `openmm>=8.5` and `openmmml>=1.6`.
The native OpenMM directory provides a C++ evaluator bridge, not a registered
OpenMM Force plugin.

## Models and conversion

The runtime contains adapters for MACE, NequIP, Allegro, PET, Nequix, ORB,
eSEN and UMA subsets. Availability of precision modes, periodic boundaries
and stress depends on the model architecture. Unsupported requests return errors.
This is not a general importer for arbitrary upstream checkpoints.

See [conversion instructions](converters/README.md),
[model format](docs/MODEL_FORMAT_E3RS.md), and
[validation scope](docs/VALIDATION.md). The current extraction was checked
with synthetic fixtures and a locally converted MACE-MP-0b3 model; it does not
constitute fresh upstream validation of every supported family.

```bash
python3 converters/convert.py list
python3 converters/convert.py info pet-mad-s
```

Upstream libraries and checkpoints must be obtained separately for conversion.
No checkpoint downloads or model redistribution occur during ordinary builds/tests.
See [weight policy](docs/model_weight_policy.md) and
[third-party records](THIRD_PARTY_LICENSES.md).

## Local validation

No hosted CI is configured. Run locally:

```bash
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
pytest crates/llmlip-py/tests
```

Some optional tests require separately acquired reference models and report
explicit skips when those models are absent. See [validation scope](docs/VALIDATION.md).

## Optional build features

The default feature set is empty. `parallel` enables Rayon. `accelerate-blas`
is available on macOS; `openblas`, `intel-mkl` and `libxsmm` have their own
platform and native-library requirements. Select at most one BLAS provider.
`embed-model` embeds a model supplied via `LLMLIP_EMBED_MODEL`. Named baked
model features also require separately supplied artifacts; they are not part
of the default build. Apple Metal and Linux backends were not validated in
this extraction. Do not use `--all-features` as a portable validation command.

```bash
cargo test --locked --workspace --features parallel,accelerate-blas
```

## License and acknowledgements

Unless a file states otherwise, project code is dual licensed under
[MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE). The LAMMPS C++ interface
files are GPL-2.0-only; see [LAMMPS licensing](lammps_plugin/llmlip/README.md#license-and-distribution).
This exception does not relicense the standalone Rust library. Third-party
components and models retain their own terms; see [third-party notices](THIRD_PARTY_LICENSES.md).

Developed by Wataru Mizukami, Do Tuan Minh and Kenji Ishihara.
Supported by JST COI-NEXT “Quantum Software Research Hub”, JPMJPF2014.

Implementation sources and unresolved license scope are recorded in the
[source provenance review](docs/SOURCE_PROVENANCE.md).

## ORB license scope clarification

The ORB implementation module is Apache-2.0; see [scope and notices](docs/ORB_LICENSE_SCOPE.md).
Original MIT OR Apache-2.0 grants do not override it. The current combined
LAMMPS plugin is not cleared for redistribution under GPL-2.0-only.
Only the separately licensed source components are in this release scope.
