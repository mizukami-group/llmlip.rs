# LAMMPS `pair_style llmlip` Plugin

This directory holds the out-of-tree LAMMPS plugin skeleton for connecting
LAMMPS neighbor lists to the Rust `.e3rs` inference runtime through
`crates/llmlip-capi`.

For a beginner-oriented build and usage guide, see
`docs/tutorials/lammps_llmlip_mace_tutorial.md`.

Initial scope:

- one entry point: `pair_style llmlip`
- `units metal`
- `newton pair on`
- `pair_coeff * * Element...`
- rank-1 default compact structure prediction through the Rust PBC path
- optional rank-1 debug mode using the ghost-expanded external graph path
- f64 external-graph prediction
- f32 MACE-lite external-graph prediction when the Rust model advertises f32
  support; ABI outputs remain double-precision energy, forces, and virial slots
- Synthetic/SyntheticEquivariant model parity through the Rust C ABI
- MACE-lite external-graph prediction for the validated f64 subset, including
  local+ghost/image-node graphs from LAMMPS PBC/MPI neighbor lists
- direct virial handoff when the Rust model advertises external-graph virial
  support
- source-generated baked MACE-OAM Medium loading through
  `llmlip_model_load_baked_mace_oam_medium`, selected with
  `pair_style llmlip baked mace-oam-medium ...`

Unsupported paths fail closed: f32 for models that do not advertise f32 support,
`NULL` mappings, per-atom energy/virial, centroid stress, selected heads when
the model does not advertise head support, unsupported LAMMPS communication
modes, and models whose external graph contract is not implemented in Rust.

For a single MPI rank, `pair_style llmlip` defaults to the compact Rust structure
prediction path and does not request a ghost-expanded LAMMPS graph. To debug the
MPI-style handoff on one rank, request the legacy path explicitly:

```lammps
pair_style llmlip model /path/to/model.e3rs dtype f64 rank1_mode external_graph virial fdotr
```

For a baked redistribution build, compile `llmlip-capi` with the baked model
feature and omit the model path in the LAMMPS input:

```bash
cargo build -p llmlip-capi --release \
  --features baked-mace-oam-medium,apple-metal,accelerate-blas,parallel
```

```lammps
pair_style llmlip baked mace-oam-medium dtype f32 rank1_mode external_graph virial fdotr
pair_coeff * * H C O
```

`mace-oam0-medium` and `mace_oam_medium` are accepted as aliases for the baked
model token.
The symbols after `pair_coeff * *` must still match the LAMMPS atom types in
the data file.

The accepted `rank1_mode` values are `structure`/`compact` and
`external_graph`/`ghost_graph`. Multi-rank runs always use the external graph
path.

Build sketch:

```bash
cargo build -p llmlip-capi
cmake -S lammps_plugin/llmlip -B build/lammps-llmlip-plugin \
  -DLAMMPS_SOURCE_DIR="$PWD/third_party/lammps" \
  -DLLMLIP_CAPI_INCLUDE_DIR="$PWD/crates/llmlip-capi/include" \
  -DLLMLIP_CAPI_LIBRARY="$PWD/target/debug/libllmlip_capi.a"
cmake --build build/lammps-llmlip-plugin
```

For a local header/API check without building LAMMPS:

```bash
scripts/check_lammps_llmlip_plugin_compile.sh
```

For an opt-in local LAMMPS vs Rust CLI smoke parity check after building a
`PKG_PLUGIN=on` LAMMPS executable:

```bash
scripts/run_lammps_llmlip_synthetic_parity.sh
```

Additional opt-in checks cover MACE-lite direct-vs-fdotr virial parity and
MPI rank-split synthetic parity against the Rust CLI baseline:

```bash
scripts/run_lammps_llmlip_mace_lite_virial_parity.sh
scripts/run_lammps_llmlip_mace_lite_pbc_mpi_virial_parity.sh
scripts/run_lammps_llmlip_synthetic_mpi_parity.sh
scripts/run_lammps_llmlip_synthetic_pbc_parity.sh
scripts/run_lammps_llmlip_mace_lite_pbc_single_rank_parity.sh
scripts/run_lammps_llmlip_mace_lite_pbc_mpi_parity.sh
scripts/run_lammps_llmlip_mace_lite_triclinic_pbc_parity.sh
scripts/run_lammps_llmlip_mace_lite_pbc_virial_fd_parity.sh
scripts/run_lammps_llmlip_cutoff_contract_parity.sh
scripts/benchmark_lammps_llmlip_si_pbc_scaling.sh
```

The MPI check requires an MPI-enabled LAMMPS executable at
`target/lammps-mpi-build/lmp` or `LMP_MPI`, plus `mpirun`/`mpiexec`.

For a local bridge-overhead probe against the Rust external-graph baseline,
build a release serial LAMMPS with `PKG_PLUGIN=on` and run:

```bash
scripts/benchmark_lammps_llmlip_bridge_overhead.sh
```

The script reports LAMMPS `Pair` timer cost per step versus
`LlmlipModel::predict_external_graph` on the same two-atom graph. Results above
10% overhead must be documented as non-production performance evidence.


## License and distribution

The C++ interface files `pair_llmlip.cpp`, `pair_llmlip.h`, and
`lammpsplugin_registration.cpp` are licensed under **GPL-2.0-only**; the full
text is in [LICENSE-GPL-2.0](LICENSE-GPL-2.0). They directly use LAMMPS headers,
its Pair API and registration mechanism. This follows the
[LAMMPS interface licensing guidance](https://docs.lammps.org/stable/Intro_opensource.html).
Original Rust contributions retain MIT OR Apache-2.0, but the current library
also contains the Apache-2.0 ORB module. It is not a uniformly MIT-selectable library.

This distribution contains source only. It does not include LAMMPS itself or
a compiled plugin. Obtain LAMMPS separately and retain its notices. Building
and using a copy locally does not itself require publishing your private changes.

Before distributing a compiled plugin, modified LAMMPS, or a container containing
them, comply with GPLv2 for the combined work: preserve license and copyright
notices and provide the complete corresponding source and build scripts, or
another source-provision method permitted by GPLv2. Review every linked library
and model artifact in that exact build. Choose MIT where an MIT/Apache dependency
offers that choice; Apache-2.0-only code cannot simply be included in a GPLv2-only
combined distribution. Dynamic loading does not by itself remove these obligations.
See the [Apache compatibility guidance](https://www.apache.org/foundation/license-faq).
No compiled-plugin redistribution clearance is implied by a successful local build.

### 日本語

上記3つのC++連携ファイルはGPL-2.0-onlyです。Rustの独自部分はMIT OR Apache-2.0ですが、ORB部分はApache-2.0です。
今回の対象はソースのみで、LAMMPS本体やビルド済みプラグインは同梱しません。
各自のローカルビルド・利用と、結合バイナリの外部配布を区別してください。
後者ではGPLv2の通知、対応ソースとビルド手順の提供、リンク先ライブラリとモデルの
条件確認が必要です。動的ロードだけを理由にこれらを省略することはできません。

## ORB license scope clarification

The ORB implementation module is Apache-2.0; see [scope and notices](../../docs/ORB_LICENSE_SCOPE.md).
Original MIT OR Apache-2.0 grants do not override it. The current combined
LAMMPS plugin is not cleared for redistribution under GPL-2.0-only.
Only the separately licensed source components are in this release scope.
