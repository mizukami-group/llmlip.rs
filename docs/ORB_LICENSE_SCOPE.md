# ORB implementation license scope and LAMMPS builds

## Decision and evidence

The ORB implementation is distributed conservatively under **Apache-2.0**.
This applies to `crates/llmlip/src/orb_v3_omol.rs` as a whole, rather than trying
to classify every line of its model-specific logic as independently authored.
It does not assert that the file was copied wholesale. Earlier grants on original
contributions are not revoked; recipients must still respect upstream obligations
for any adapted material. Original code outside this exception retains its stated
MIT OR Apache-2.0 choice and the other third-party notices.

The upstream reference is [orb-models v0.6.2](https://github.com/orbital-materials/orb-models/tree/a5f1a382a72126feff0594b19891e03cde5f843e),
by Orbital Materials, under [Apache-2.0](../licenses/reference/orb.txt).
No separate upstream NOTICE file was found in that pinned tree. The project
[NOTICE](../NOTICE) records attribution and modifications; the module header
identifies the exception and modifications prominently.

| Rust area in orb_v3_omol.rs | Pinned upstream source | Correspondence / changes |
|---|---|---|
| `gnn_forward_impl` | `orb_models/common/models/gns.py`, `AttentionInteractionNetwork.forward` | Conditioning, sender/receiver attention, edge/node message construction, scatter and residual update sequence; Rust uses explicit buffers and fused/native kernels. |
| charge/spin sinusoidal embedding | `orb_models/common/models/nn_util.py`, `ChargeSpinEmbedding.forward` | f32 rounding before sinusoidal embedding and zero-spin handling are model-specific matches. |
| cutoff and edge attention | `nn_util.py::get_cutoff`, `gns.py` | Pinned cutoff and sigmoid attention behavior; explicit force derivatives added in Rust. |
| input/radial features and readout | `common/models/rbf.py`, conservative forcefield model interfaces | Reference conventions and weight layout; validated fixed-weight execution instead of PyTorch autograd. |

The loader in `model.rs` is project-specific `.e3rs` schema/validation and dispatch,
not an upstream ORB checkpoint loader. The exporter imports installed ORB code
and serializes its tensors; it does not bundle upstream package sources or weights.
Shared Rust matrix/BLAS helpers are project-native kernels; a helper name beginning
with `orb_` is not by itself evidence of copied ORB code. The upstream routines
examined use PyTorch operations, not these Rust SIMD/BLAS kernels.

Cargo/Python source-package metadata now records the cumulative license expression
`(MIT OR Apache-2.0) AND Apache-2.0 AND BSD-3-Clause AND Zlib`. The explicit Apache
term covers the ORB exception; BSD/Zlib notices cover the numerical material
recorded in [LICENSE-THIRD-PARTY](../LICENSE-THIRD-PARTY). A source package includes
the ORB source file even if an individual binary build disables it. Metadata is
not a license change to unrelated original source or separately fetched dependencies.

## Why the LAMMPS build differs

LAMMPS's pinned source and [official guidance](https://docs.lammps.org/Intro_opensource.html)
apply GPLv2. The three C++ interface files remain GPL-2.0-only. Apache-2.0-only
code is [incompatible with a GPLv2 combined distribution](https://www.apache.org/licenses/GPL-compatibility.html).
Neither selecting a different model at runtime nor dynamic loading removes
code already linked into the library. Source-only aggregation and private local
use must be distinguished from distributing that combined binary.

Normal Rust, CLI, Python and C ABI builds enable the `orb` feature by default,
preserving ORB support. The `lammps` build profile excludes it:

```bash
cargo build -p llmlip-capi --release --no-default-features --features lammps   --target-dir target/lammps-runtime
```

Add compatible CPU features as needed, for example `lammps,parallel,accelerate-blas`
on macOS, and review their native-library distribution terms separately.
Build this package separately: a workspace-wide build can unify Cargo features
and re-enable ORB through another package. `orb` and `lammps` together are a compile
error, including indirect ORB optimization features. `--all-features` intentionally
includes an invalid combination; it is not a supported build configuration.

With `orb` disabled, the module, public model types, dispatch variants and loader
routines are excluded by `cfg`. An ORB model request is rejected as an unsupported
architecture before tensor loading. Other model inference is unchanged.

The LAMMPS CMake configuration links a small check against the supplied C ABI
library and requires `llmlip_lammps_no_orb_build`. That symbol is exported only
by a `lammps` build. The C++ interface also references it, preventing accidental
use of the usual ORB-enabled library. Use a separate target directory so a later
normal build does not overwrite the chosen library. Old plugins/libraries must
be rebuilt together. There is no override that labels an ORB-enabled build safe.

## Distribution boundary

For a no-ORB combined binary, select the MIT grant for applicable original/core
code, retain BSD/MIT/Zlib notices, and meet GPLv2 requirements for the interface
and LAMMPS, including corresponding source and build information. Preserve the
full source archive's third-party notices even when giving corresponding source
that contains an optional, uncompiled ORB file. The package-wide Apache term is
not evidence that the excluded code is in the binary, and the no-ORB marker is
not a certification of every other linked component.

This change removes the identified ORB/GPLv2 conflict from the prescribed build.
It does not authorize distribution of model checkpoints, proprietary BLAS libraries,
or old wheels lacking license files. The current release scope remains
[source-only](distribution_policy.md); any binary distribution still requires an
inventory of its actual dependencies, required notices and corresponding source.
Using ORB from LAMMPS in a distributed combined binary would require a separate
compatible permission or a separately reviewed integration design. Do not assume
an exception exists or that all LAMMPS versions may be relicensed GPLv3.
