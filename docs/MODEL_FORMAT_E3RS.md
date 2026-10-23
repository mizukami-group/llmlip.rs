# `.e3rs` Model Format Specification

## Goal

Define a simple, explicit, Rust-readable model format for fixed-weight force inference.

Do not parse arbitrary PyTorch `state_dict` in v0.1. Use a Python exporter to convert a reference model into `.e3rs`.

## Format style

Initial v0.1 format is an unpacked directory:

```text
model.e3rs/
  manifest.toml
  weights/
    tensor_000.f64.bin
    tensor_001.f64.bin
  plans/
    tensor_product_000.json
```

A packed archive can be added later.

## Manifest fields

Example:

```toml
format_version = "0.1"
architecture = "synthetic-local-v1"
dtype = "f64"
length_unit = "angstrom"
energy_unit = "eV"
force_unit = "eV/angstrom"

[species]
symbols = ["H", "C", "O"]

[geometry]
cutoff = 5.0
edge_vector = "R_j + shift - R_i"
directed_edges = true

[spherical_harmonics]
l_max = 2
normalize_input = true
normalization = "component"
reference_convention = "pinned-fixture"

[validation]
reference_package = "e3nn-torch"
reference_version = "TO_BE_FILLED_BY_EXPORTER"
fixture_seed = 12345

[synthetic_radial]
centers = [0.5, 1.0, 1.5, 2.0]
gamma = 0.8

[[mlp_layers]]
weight = "edge_mlp.layers.0.weight"
bias = "edge_mlp.layers.0.bias"
activation = "tanh"

[[mlp_layers]]
weight = "edge_mlp.layers.1.weight"
bias = "edge_mlp.layers.1.bias"
activation = "identity"

[[tensors]]
name = "edge_mlp.layers.0.weight"
dtype = "f64"
shape = [3, 4]
layout = "row-major"
path = "weights/edge_mlp.layers.0.weight.f64.bin"
sha256 = "..."

# Tensor product plan sections are used by synthetic-equivariant-v1 and later
# Nequip-lite, but are not loaded by synthetic-local-v1.
```

## Optional dispersion section

Models may include a top-level `[dispersion]` section to request a fixed
dispersion correction composed with the base architecture:

```toml
[dispersion]
model = "dftd3.json"
correction = "dftd3-bj-coordination-v1"
terms = "two-body-bj-coordination-dependent-c6"
reference_package = "torch-dftd"
reference_version = "0.5.3"
source_sha256 = "52b7b35a957df55b813035ffde3a6b5a41dec585bdb1e281fb8ba67a729f3d92"
three_body = false
```

The first supported payload is a strict JSON file with
`schema_version = "dftd3-rust-subset-model-v1"`. It stores BJ damping
parameters, Grimme coordination parameters, element `rcov`/`r2r4` rows, and
coordination-dependent `C6` reference grids in eV/Angstrom units. The loader
requires `abc = false`, `cutoff_smoothing` to be `"none"` or `"cosine"`,
bidirectional directed edges, all manifest species to have element rows, and
all unordered manifest species pairs to have C6 grids. The optional
`table_scope` metadata may be
`"hco"` for the compact parity fixture or `"full"` for the Z=1..94 PBE D3(BJ)
two-body table generated from torch-dftd source data.

`source_sha256` and the JSON payload's `npz_sha256` are provenance checks for
the pinned reference data used by the exporter. They are not a load-time
integrity proof for an external `.npz` file: normal Rust inference reads only
the committed `.e3rs` manifest, tensor payloads, and DFTD3 JSON.

This is a model-format hook for the currently validated two-body DFTD3(BJ)
coordination path. The committed full-table payload covers the packaged
two-body PBE D3(BJ) data for elements H-Pu. It is not full D3 compatibility:
only the no-smoothing and cosine-cutoff variants are implemented, and ATM
three-body terms remain unsupported. `cutoff_smoothing = "none"` preserves the
pinned torch-dftd two-body values. `cutoff_smoothing = "cosine"` is an explicit
Rust-local opt-in taper and is not the default torch-dftd smoothing convention.
If ATM is requested through
`three_body = true` or `abc = true`, the loader rejects the model.

Unsupported element policy: DFTD3 dispersion corrections fail closed. The
loader must reject a model if any manifest species lacks an element row or if
any unordered manifest species pair lacks a `C6` grid in the DFTD3 payload. It
must not silently skip unsupported species, apply a zero dispersion correction,
or drop species from validation. The committed `table_scope = "full"` fixture is
therefore usable only for models whose manifest species are within H-Pu
(Z=1..94), and the loader rejects `table_scope = "full"` payloads that declare
element rows above Pu. Osaka26 artifacts that declare species above this range
require additional reference data or a new explicitly documented
unsupported-element policy before a DFTD3-corrected artifact can be generated
safely.

DFTD3 PBC scope is also fail-closed. The current DFTD3 implementation supports
non-periodic structures and full 3D PBC structures. Partial-PBC structures are
not fixture-pinned for DFTD3 and are rejected until a separate validation matrix
is added.

## Tensor binary format

For v0.1:

- little-endian f64 or f32 per `[[tensors]].dtype`,
- contiguous row-major,
- no compression,
- shape stored in manifest,
- optional `sha256` with the 64-character hex digest of the exact tensor
  payload bytes.

The loader must validate file length:

```text
file_bytes == product(shape) * sizeof(dtype)
```

When `sha256` is present, loaders and build-time source generators that consume
manifest tensor payloads must compare it against the exact on-disk bytes and
fail closed on mismatch. This is required for baked MACE runtime generation so
same-length finite but incorrect tensor substitutions cannot be baked silently.

Top-level `dtype = "f64"` remains the runtime arithmetic contract for the
current Rust models. Individual tensor entries may use `dtype = "f32"` when an
upstream fixed-weight artifact stores parameters in f32; the loader promotes
those payload values to f64 before constructing the Rust runtime model. This is
the intended bridge for production NeQUIP-OAM-L tensors and does not imply a
native f32 inference path.

For `pet-lite-v1`, PET tensor-payload models may additionally retain a native
f32 cache beside the promoted f64 values. That cache exists only to support the
explicit PET-MAD-S FP32 optimization path; the public prediction API still
returns f64 energy and force values, and unsupported PET configurations must
fail closed.

Across `.e3rs` runtimes, lower-precision compute dtypes are allowed to keep hot
feature, dense, readout, and VJP intermediates in f32, but the final
energy-reduction boundary is f64 by default. This includes atom-to-system
energy accumulation and public `Prediction.energy` values.

## Tensor product plan format

Example JSON:

```json
{
  "name": "tp0",
  "irreps_in1": "1x0e + 1x1o",
  "irreps_in2": "1x0e + 1x1o + 1x2e",
  "irreps_out": "1x0e + 1x1o",
  "weight_numel": 4,
  "instructions": [
    {
      "in1_segment": 0,
      "in2_segment": 0,
      "out_segment": 0,
      "cg_offset": 0,
      "cg_len": 1,
      "weight_offset": 0,
      "path_weight": 1.0
    }
  ],
  "cg_entries": [
    {"i_m": 0, "j_m": 0, "k_m": 0, "value": 1.0}
  ]
}
```

## Architecture names

Allowed:

```text
synthetic-local-v1
synthetic-equivariant-v1
nequip-lite-v1
nequip-oam-l-v1
nequix-v1
mace-lite-v1
mace-mh1-v1
mace-polar-v1
allegro-oam-l-v1
orb-v3-conservative-omol-v1
pet-lite-v1
esen-oam-v1
uma-v1
```

Unknown architectures must fail explicitly.

The `orb-v3-conservative-omol-v1` architecture is a pinned runtime contract
for `orbital-materials/orb-models` `orb-v3-conservative-omol`, not a general
ORB checkpoint loader. The first scope is FP64, non-PBC molecular energy and
conservative forces with explicit charge and spin metadata. The current loader
accepts two fail-closed states: `metadata_only_not_executable` to pin
provenance and architecture parameters, and `state_export_only` to validate the
145-tensor state export with f64/i64 row-major payloads. Metadata-only
artifacts remain explicitly unsupported for prediction; public state-export
prediction supports non-PBC energy and conservative forces with explicit or
defaulted charge/spin conditioning.
Validated state payloads are read through typed tensor accessors that require
the expected tensor name, dtype, and shape; future ORB kernels should not index
the raw external tensor map directly.
The Rust graph-preparation helper currently covers only non-PBC structures,
uses finite total charge and spin multiplicity values, emits directed edges
with the project convention `R_j - R_i`, emits zero unit shifts, and rejects
atoms with more than the pinned 120 outgoing neighbors. Public prediction and
prepared-input APIs use explicitly supplied `PredictionOptions.total_charge`
and `PredictionOptions.spin_multiplicity` when present; when either field is
absent, the ORB runtime defaults it to the pinned neutral baseline value
`0.0`. Non-finite charge/spin values still fail closed. PBC graph payloads,
stress, and virial remain outside the executable ORB contract until separate
fixtures pin their behavior. ORB atomic-number resolution is part of this
handoff. The current linear reference-energy table has 118 classes, so symbols
mapping outside atomic numbers `1..=117` are rejected until a fixture pins
upstream behavior for them.
The first executable feature handoff also pins ORB's non-PBC input feature
buffers: Bessel basis values, component-normalized `lmax=3` spherical
harmonics, polynomial cutoffs, flattened radial/angular edge features, atom
embedding rows, and charge/spin node conditioning. These buffers are validated
against a committed upstream feature trace before later encoder and
message-passing kernels consume them. The current encoder handoff replays the
pinned node and edge MLP/RMSNorm stacks and validates encoded node/edge latents
against the same trace. The current GNN handoff replays all five pinned
attention-interaction layers and validates every layer's node/edge latents
against the same trace. The current energy handoff replays the pinned
`heads.energy` MLP, batch-normalizer inverse transform, linear reference-energy
addition, and ZBL pair-repulsion energy against that trace. ZBL constants
preserve upstream precision quirks: registered buffers and covalent radii are
f32-origin values promoted to f64, while the Python literals `0.529` and
`14.3996` remain f64 literals. Public state-export prediction now supports
non-PBC energy and conservative forces with explicit or defaulted charge/spin
conditioning; stress and virial still reject explicitly. The energy-head tail
VJP is pinned from total energy back to final GNN node latents, the GNN input
VJP is pinned back to encoded node/edge latents plus charge/spin conditioning
and attention cutoffs, the encoder VJP is pinned back to node/edge input
feature buffers, and the geometric VJP is pinned from ORB edge features and
attention cutoffs back to edge vectors.

The user-provided MACE-OAM small/medium artifacts are scoped `mace-lite-v1`
family targets, not a new broad architecture string and not arbitrary MACE
checkpoint import. Their source artifact lock is
`fixtures/mace/v0_3_16_4d2da09/mace_oam_artifact_manifest.json`; runtime
exports should use manifest-backed f64 tensors plus an explicit MACE-OAM
runtime plan under the existing `mace-lite-v1` loader.

The `esen-oam-v1` architecture is the planned pinned FairChem eSEN-OAM runtime
contract for `selected_model = "esen-30m-oam"`, not broad FairChem
compatibility and not an arbitrary PyTorch checkpoint importer. The loader
currently recognizes the state-export metadata boundary and validates the
pinned OAM artifact identity, selected model, state schema, model constants,
Z=1..94 element coverage, tensor summary, tensor names, tensor dtypes, tensor
shapes, and tensor SHA256 payload digests. If the ignored local tensor payload
files are present, the loader also verifies every payload byte length and
SHA256 and decodes f64/i64 payloads into a typed tensor store before retaining
the bundle as payload-verified metadata. Metadata-only handoffs fail closed for
prediction and input preparation because the fixed tensor payloads are not
available. When the full verified FP64 tensor payload bundle is present, the
scoped eSEN-OAM runtime prepares inputs and runs non-PBC energy and
conservative-force prediction (stress and virial still reject) through the
pinned runtime graph and force-only VJP.

The `allegro-oam-l-v1` architecture is a planned pinned Allegro-OAM-L runtime
contract for `mir-group/Allegro-OAM-L:0.1`, not broad Allegro compatibility and
not an arbitrary `.nequip.zip` loader. The loader currently supports two
fail-closed states: an inspection-only manifest that returns "recognized but
not implemented", and a `state_export_only` manifest that parses the pinned
state JSON/tensor inventory and runs the narrow generated-state runtime. The
current runtime covers non-PBC energy/forces and full-PBC
energy/forces/stress/virial for the pinned contract; partial-PBC stress/virial,
broad Allegro compatibility, and arbitrary `.nequip.zip` loading remain
unsupported. The source `.nequip.zip` remains outside git under an ignored
reference cache.

Planned manifest section:

```toml
[allegro_oam]
model = "allegro_oam_l.json"
reference_package = "allegro"
reference_version = "0.8.2"
reference_commit = "6d05192cfc6047f22cb8b97cb86839667e9e484c"
selected_model = "mir-group/Allegro-OAM-L:0.1"
inspection_schema = "allegro-oam-inspection-v1"
artifact_sha256 = "3f0d3ca7bb136d4c2ee76278170fcbe756436f037e16c673a86e2c57d271c64e"
artifact_size_bytes = 80738705
runtime_status = "artifact_inspected_export_not_implemented"
```

Executable state-export handoff manifests must instead set:

```toml
runtime_status = "state_export_only"
```

and the referenced JSON must use `schema_version =
"allegro-oam-state-export-v1"` with pinned artifact identity, `cutoff = 7.0`,
`l_max = 3`, 89 loaded type names, 34 source state tensors, and the
`allegro-oam-execution-contract-v1` counts. Parity is claimed only for
fixtures/tests that explicitly compare a generated tensor bundle against the
pinned upstream Allegro reference; the manifest status alone is not a parity
claim.

The manifest contract additionally requires `geometry.cutoff = 7.0`,
`spherical_harmonics.l_max = 3`, `normalize_input = true`, component
normalization, directed edges, and the project edge convention
`R_j + shift - R_i`.

The `nequip-oam-l-v1` architecture is the pinned NeQUIP-OAM-L runtime contract,
not the general `nequip-lite-v1` format and not an arbitrary `.nequip.zip`
loader. It is loadable as an `LlmlipModel` only for the pinned OAM-L `.e3rs`
export:

- non-PBC energy and force inference against the pinned OAM-L force fixture,
- partial-PBC energy and force inference against a pinned H/C/O/Si OAM-L
  fixture,
- full-PBC energy, force, stress, and virial inference, currently validated on
  the pinned H-H-O and H/C/O/Si OAM-L fixtures,
- broader non-PBC energy and force validation on a pinned Li/Na/Al/S/Fe
  production-artifact fixture without intermediate captures,
- broader B/O/Ti/Ag non-PBC energy and force validation, plus Li/B/O/Ti
  triclinic full-PBC energy, force, stress, and virial validation,
- larger H/C/O/Si/Li/N/S/Ti full-PBC energy, force, stress, and virial
  validation with 52 directed edges, x-direction periodic image shifts, and a
  near-cutoff edge,
- `l_max = 3`,
- six exported ConvNet layers plus readout, per-type scale/shift, ZBL, and
  total-energy tail,
- manifest-backed fixed tensors promoted to f64 runtime arithmetic.

The runtime validates this checkpoint as a pinned production-artifact contract:
the JSON model spec must identify `mir-group/NequIP-OAM-L:0.1`, the declared
packaged-artifact SHA256 and byte size provenance fields, the pinned NequIP
reference revision, the `manifest-f32-f64-tensors-v1` state-dict format, the
exact six-layer tensor-product semantics, and all required manifest tensor
names and shapes. The original `.nequip.zip` package is a reference/export
input, not a Rust runtime input, so normal prediction does not rehash that
package at load time.

It must reject stress/virial requests unless all three PBC axes are enabled.
It is not a claim of full NeQUIP-OAM-L compatibility, arbitrary upstream NequIP
checkpoint loading, or arbitrary structure parity beyond the pinned OAM-L
fixtures.

The legacy architecture name `nequip-oam-prefix-v1` is still accepted for
previously generated fixtures, but new pinned OAM-L exports should use
`nequip-oam-l-v1`.

The `nequix-v1` architecture is the pinned NequiX-PFT runtime contract for
`nequix-mp-1-pft` and `nequix-oam-1-pft`. It is not a generic `.nqx` loader
and must reject unknown NequiX model keys or architecture variants. The
reference artifacts are pinned under `fixtures/nequip/v0_4_5_da0fb24/`; the
downloaded `.nqx` binaries are reference/export inputs only and must not be
required by normal Rust tests. Runtime support must use exported `.e3rs`
tensors/plans, promote fixed `float32` upstream tensors to f64 arithmetic, and
cross-check `selected_model`, artifact SHA256/size, `lmax = 3`, `cutoff = 6.0`,
`n_layers = 4`, hidden irreps, scale/shift, atom energies, and tensor inventory
against the pinned manifest before constructing an executable model.
The current loader recognizes this manifest contract, cross-checks the exported
state against the manifest, and supports energy and force prediction for the
pinned NequiX-PFT models. Stress and virial requests still fail closed because
they are not part of the pinned NequiX-PFT validation contract.

The `mace-mh1-v1` architecture is the scoped MACE-MH-1 `mh-1` runtime
contract. Its manifest section is:

```toml
[mace_mh1]
model = "mace_mh1.json"
```

The current loader accepts inspected exports with either
`schema_version = "mace-mh1-metadata-v1"` or
`schema_version = "mace-mh1-state-dict-v1"`. It validates the pinned official
GitHub release `mh-1` source, MACE v0.3.16 provenance, observed head list,
missing requested heads, species order, cutoff, `l_max`, nonlinear block
classes, and inspected state-dict tensor counts before constructing a model.
The state-dict schema additionally requires manifest-backed external tensors
with exact name, dtype, shape, and value-count coverage. Metadata-only
prediction remains unsupported; state-dict exports can execute energy and force
prediction for the pinned `mh-1` runtime path.
The loader pins the executable tensor boundary by requiring the
observed `node_embedding`, radial embedding, ZBL pair repulsion, per-head
atomic energy, scale/shift, readout, interaction-0 source/target embedding,
linear-up, `skip_tp`, `linear_res`, `linear_1`, `equivariant_nonlin.mul`,
`linear_2`, `alpha`/`beta`, `conv_tp_weights`, `conv_tp.output_mask`, and
`density_fn` tensor names/dtypes/shapes from the GitHub `mh-1` state export.
The Rust boundary can now build interaction-0 augmented edge MLP inputs,
tensor-backed TP-weight/density rows, interaction-0 `conv_tp` edge messages
plus receiver scatter, and the immediate post-scatter linear layers, density
normalization, Gate, reshape update, product0/readout0, interaction-1 edge
MLP plus `conv_tp` scatter, interaction-1 post-scatter, product1/readout1,
selected-head readout masking, and scale-shifted energy composition. The
force-only VJP now propagates through interaction-1 input linears back to
product0, through interaction-0 edge paths to directed edge vectors, and
scatters those edge-vector cotangents to atomic forces. Stress and virial
prediction still reject explicitly.
`PredictionOptions.selected_head` and the CLI `--head` flag are the supported
API surface for MACE-MH-1 head selection; unknown heads are rejected
explicitly.

The first committed metadata fixture is:

```text
fixtures/mace/v0_3_16_4d2da09/models/mace_mh1_metadata.e3rs/
```

The reference-only exporter for a local ignored tensor bundle is:

```bash
PYTHONPATH=tools/reference:third_party/mace TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  .venv-reference/bin/python tools/reference/export_mace_mh1_state.py \
  --force \
  --out-dir outputs/reference-artifacts/mace/mace-mh1/github-release/mace_mh1_state.e3rs
```

## Synthetic radial model fields

For the initial `synthetic-local-v1` implementation, the runtime loads a
minimal radial pair-energy validation model:

```toml
[synthetic_radial]
centers = [0.5, 1.0, 1.5, 2.0]
gamma = 0.8

[[mlp_layers]]
weight = "edge_mlp.layers.0.weight"
bias = "edge_mlp.layers.0.bias"
activation = "tanh"      # or "identity"
```

Layer tensors are referenced by name from `[[tensors]]`. Weight tensors are
row-major with shape `[out_dim, in_dim]`; bias tensors have shape `[out_dim]`.
The first layer input dimension must equal `synthetic_radial.centers.len()`,
and the final output dimension must be `1`.

## Synthetic equivariant model fields

`synthetic-equivariant-v1` currently reuses the radial baseline tensor format:
`[synthetic_radial]` declares the cutoff basis parameters and `[[mlp_layers]]`
declares the radial MLP. The runtime attaches the fixture-safe built-in `l=1`
SH/TP graph.

Explicit SH/TP model metadata is still reserved for a later format extension:

```toml
architecture = "synthetic-equivariant-v1"

[synthetic_equivariant]
message_irreps = "1x1o"
node_irreps = "1x1o"
energy_irreps = "1x0e"
radial_weight_numel = 1
```

The current loader validates those values exactly against the built-in graph.
Explicit TP plan files are reserved for a later format extension:

```toml
edge_tp_plan = "plans/edge_message_tp.json"
energy_tp_plan = "plans/energy_tp.json"
```

When explicit plan fields are enabled, the loader must validate:

- all referenced tensor and plan paths stay inside the model directory,
- SH settings match supported fixture-pinned conventions,
- TP plans declare irreps, instruction slices, dynamic weight counts, and CG
  entries explicitly,
- dynamic weight counts match the radial MLP output dimension,
- scalar energy contraction produces one scalar contribution per node or edge,
- no learned parameter gradients are implied by the format.

## Convention validation

The loader must reject a model if the runtime does not support the declared:

- dtype,
- `l_max`,
- SH normalization,
- edge convention,
- architecture,
- model format version.

The current `synthetic-local-v1` loader may reject `l_max > 2` until the T15
primitive extension is implemented. The Nequip-lite target format requires
`l_max <= 4`.

`synthetic-equivariant-v1` should initially use the already pinned
`l_max <= 2` primitive baseline, with a conservative `l=1` energy graph unless
additional higher-`l` energy-path fixtures are generated.

## Nequip-lite model fields

The planned `nequip-lite-v1` architecture must serialize only explicitly
supported fixed-weight blocks:

```toml
architecture = "nequip-lite-v1"

[spherical_harmonics]
l_max = 4
normalize_input = true
normalization = "component"
reference_convention = "pinned-fixture"

[nequip_lite]
num_layers = 1
feature_irreps = "32x0e + 32x1o + 16x2e + 8x3o + 4x4e"
```

The initial loader stores the exported Nequip-lite graph in a JSON file
referenced from the manifest:

```toml
[nequip_lite]
model = "nequip_lite.json"
```

The JSON schema records node type embeddings, Bessel/cutoff metadata, dense
linear maps, tensor-product plans, gate metadata, readout weights, and per-type
energy scale/shift values. For small deterministic fixtures, these values may
be embedded directly in `nequip_lite.json`. The loader also supports
manifest-backed f64 tensor references for the same fields:

```json
{
  "node_embedding_tensor": "node_embedding",
  "layers": [
    {
      "linear1": {"in_dim": 2, "out_dim": 2, "weight_tensor": "layers.0.linear1.weight"}
    }
  ],
  "readout": {"in_dim": 2, "out_dim": 1, "weight_tensor": "readout.weight"},
  "per_type_energy_scales_tensor": "per_type_energy_scales",
  "per_type_energy_shifts_tensor": "per_type_energy_shifts"
}
```

Each referenced tensor must be declared in manifest `[[tensors]]` as row-major
f64 data, or f32 data promoted to f64 at load time. Dense `weight_tensor`
references use row-major rank-2
`[out_dim, in_dim]` tensors; rank-1 `[out_dim * in_dim]` dense tensor
references remain accepted only for compatibility with early tensor-storage
fixtures. Embedded dense `weight` arrays remain flat row-major vectors. This
path is the required storage foundation for production
NeQUIP-OAM-L exports; it does not by itself imply OAM-L runtime parity.
Unsupported upstream NeQUIP modules must fail during export or load, not
degrade silently.

## Exporter requirements

Python exporter must write:

- manifest,
- tensors,
- tensor product plans,
- reference fixture for at least one structure,
- environment metadata.

Exporter must not be required at inference runtime.

## MACE-lite model fields

The planned `mace-lite-v1` architecture serializes only the explicitly
supported fixed-weight subset documented in `docs/MACE_LITE_SPEC.md`.

```toml
architecture = "mace-lite-v1"

[spherical_harmonics]
l_max = 2
normalize_input = true
normalization = "component"
reference_convention = "pinned-fixture"

[mace_lite]
model = "mace_lite.json"
reference_package = "mace"
reference_version = "0.3.16"
reference_commit = "4d2da09413ac1407f37cdbb6b81fa28e4c15655e"
```

The initial `mace_lite.json` schema records the fixed execution plan:

```json
{
  "architecture": "mace-lite-v1",
  "heads": ["Default"],
  "atomic_numbers": [1, 6, 8],
  "atomic_energies": "atomic_energies",
  "scale_shift": {"scale": "scale", "shift": "shift"},
  "radial": {
    "type": "bessel",
    "num_bessel": 8,
    "polynomial_cutoff_p": 6,
    "r_max": 5.0,
    "distance_transform": "None"
  },
  "pair_repulsion": null,
  "layers": [
    {
      "interaction": "RealAgnosticResidualInteractionBlock",
      "linear_up": "layers.0.linear_up",
      "conv_tp_plan": "plans/layers.0.conv_tp.json",
      "conv_tp_weights_mlp": "layers.0.conv_tp_weights",
      "linear": "layers.0.linear",
      "skip_tp_plan": "plans/layers.0.skip_tp.json",
      "product": {
        "symmetric_contraction_plan": "plans/layers.0.symmetric_contraction.json",
        "linear": "layers.0.product.linear",
        "correlation": 3
      }
    }
  ],
  "readouts": [
    {"kind": "linear", "weight": "readouts.0.linear"}
  ],
  "state_dict_format": "embedded-torch-state-dict-v1",
  "state_dict": []
}
```

Current compact MACE-lite fixtures cover `l_max = 2` and the scalar-hidden
two-layer `l_max = 3` path. Larger feature dimensions and vector-output
symmetric-contraction product blocks must be represented explicitly before a
representative MACE-MP0b `.e3rs` export is allowed.

`radial.distance_transform` defaults to `"None"` when omitted. The current
runtime accepts `"None"` and `"Agnesi"`; for Agnesi, cutoff remains a function
of the original edge distance while Bessel radial features use the transformed
distance.
`pair_repulsion` may be `null`/omitted or `{"type": "zbl",
"polynomial_cutoff_p": 6}`. The current ZBL path requires the pinned MACE
default constants. Its polynomial cutoff exponent is serialized separately
from the radial cutoff exponent because the pinned MACE-MP0b artifact uses
`p=5` for both radial and ZBL while older compact fixtures may use `p=6`.

The loader must reject unsupported upstream MACE features, including multihead
models, joint embeddings, attention interaction blocks,
foundation-model-specific extras, stress/virial-only paths, and backend
acceleration wrappers. PolarMACE must not be loaded through `mace-lite-v1`; it
uses the separate `mace-polar-v1` architecture.

### Pinned MACE-family runtime provenance

Pinned large-artifact MACE-family exports (`mace_mp0b_runtime`,
`mace_osaka26_runtime`, and `mace_osaka24_{small,medium,large}_runtime`) must
carry a `provenance` object in `mace_lite.json`. This is the current
MACE-family registry/manifest identity contract:

```json
{
  "provenance": {
    "source_family": "MACE-Osaka24",
    "selected_model": "mace-osaka24-small",
    "artifact": {
      "filename": "mace-osaka24-small.model",
      "url": "https://...",
      "sha256": "...64 hex chars...",
      "size_bytes": 32598220
    },
    "reference": {
      "name": "mace",
      "repository": "https://github.com/ACEsuit/mace",
      "commit": "4d2da09413ac1407f37cdbb6b81fa28e4c15655e",
      "package_version": "0.3.16"
    },
    "state_dict_format": "manifest-f64-tensors-v1",
    "upstream_state_tensor_count": 69,
    "runtime_state_tensor_count": 49,
    "archival_only_state_tensor_count": 20,
    "upstream_state_dict_inventory": "upstream_state_dict_inventory.json"
  }
}
```

The runtime validates the pinned source family, selected model, artifact URL,
artifact SHA256, artifact size, upstream MACE commit/package version,
state-dict storage format, tensor counts, and inventory filename for each
pinned family. The tensor counts must satisfy:

```text
upstream_state_tensor_count
  = runtime_state_tensor_count + archival_only_state_tensor_count
```

Schema validation and stale-provenance rejection are covered by
`mace_mp0b_runtime_rejects_stale_provenance`,
`mace_osaka26_runtime_rejects_stale_provenance`, and
`mace_osaka24_runtime_rejects_stale_provenance` in
`crates/llmlip/tests/mace_lite_reference.rs`. These tests reject missing
provenance, stale artifact hash, stale artifact URL, stale upstream commit,
stale state-dict format, stale inventory filename, unknown provenance fields,
and mismatched runtime tensor counts.

This contract is intentionally pinned-artifact oriented, not a generic MACE
checkpoint registry. The companion registry file
`fixtures/mace/v0_3_16_4d2da09/mace_family_runtime_registry.json` records the
pinned runtime entries, artifact hashes/sizes, tensor counts, element coverage,
cutoff, `l_max`, units, benchmark summary paths, benchmark summary SHA256
values, and observed speedup values. The registry is covered by
`mace_family_runtime_registry_matches_artifacts_and_benchmarks`, which compares
registry entries against each registered model manifest/spec and recomputes the
benchmark summary file hashes.

The Rust public API exposes generic pinned-runtime introspection through
`MaceLiteModel::active_runtime_plan_kind()` and
`MaceLiteModel::has_pinned_mace_runtime_plan()`. Legacy family-specific JSON
keys remain accepted for fixture compatibility:

- `mp0b_runtime_plan`
- `osaka26_runtime_plan`
- `osaka_runtime_plan`

New runtime-family additions should prefer the generic API boundary and add
schema-specific aliases only when a pinned artifact family requires backwards
compatible fixture keys.

The initial one-layer and two-layer MACE-lite fixtures embed the pinned
upstream Torch `state_dict` directly in `mace_lite.json` using
`state_dict_format = "embedded-torch-state-dict-v1"`. This is an intermediate,
explicit fixture format for loader/runtime bring-up; later larger MACE exports
may move large f64 tensors into the shared binary `[[tensors]]` manifest
representation.

For larger MACE-lite exports, including the representative MACE-MP0b path, the
model JSON may instead declare:

```json
"state_dict_format": "manifest-f64-tensors-v1"
```

In that mode `mace_lite.json` must leave `state_dict` empty/omitted, and every
fixed f64 state tensor needed by the exported subset must appear in the parent
`manifest.toml` as a `[[tensors]]` entry. Tensor names are still the canonical
state tensor names referenced by the MACE-lite execution plan. This is not
arbitrary PyTorch checkpoint loading; it is only binary storage for an explicit
`.e3rs` export.

The pinned MP0b runtime export must additionally include a top-level
`provenance` block in `mace_lite.json`. This block records the source family,
selected model, original artifact filename/URL/SHA256/byte size, MACE reference
repository/commit/package version, state-dict format, upstream/runtime/archival
state tensor counts, and `upstream_state_dict_inventory.json` path. When an
`mp0b_runtime_plan` is present, the Rust loader requires these fields to match
the pinned `mace-mp-0b3-medium.model` contract before prediction. This loader
check pins artifact identity and state inventory counts; it does not rehash the
ignored upstream `.model` file or compare every exported f64 tensor value at
normal prediction time. Tensor-value parity remains pinned by committed
fixtures and `upstream_state_dict_inventory.json`.

Large MACE-lite exports may include a top-level `conv_tp_plans` array in
`mace_lite.json`. Each entry records the layer index, input/output irreps,
dynamic `weight_numel`, expanded tensor-product instructions, and explicit
sparse CG entries using the same field names as the pinned reference fixture.
The MP0b runtime export currently stores the layer `1` plan this way so Rust
runtime energy/force/stress inference can run without reading fixture JSON.

### Symmetric-contraction plan format

MACE-lite product-basis blocks require a serialized symmetric-contraction plan.
The exact dense/sparse representation may evolve during T19, but every plan
must record:

- `irreps_in`,
- `irreps_out`,
- `correlation`,
- `num_elements`,
- `irrep_normalization`,
- `path_normalization`,
- `use_reduced_cg`,
- per-output-irrep `U_matrix_nu` data or a sparse equivalent,
- element-conditioned weight tensor names and shapes,
- layout metadata needed for forward and VJP wrt node features.

Stored weights and `U` tensors are fixed parameters; MACE-lite force inference
does not compute gradients wrt them.

The compact scalar fixtures use the legacy single-contraction form:

```json
"u_tensors": ["...U_matrix_1", "...U_matrix_2", "...U_matrix_3"],
"weight_tensors": ["...weights.0", "...weights.1", "...weights_max"]
```

MACE-MP0b product blocks that emit both scalar and vector irreps must use the
component form instead:

```json
"contractions": [
  {
    "kind": "scalar",
    "u_tensors": ["...contractions.0.U_matrix_1", "...U_matrix_2", "...U_matrix_3"],
    "weight_tensors": ["...contractions.0.weights.0", "...weights.1", "...weights_max"]
  },
  {
    "kind": "vector",
    "num_equivariance": 3,
    "u_tensors": ["...contractions.1.U_matrix_1", "...U_matrix_2", "...U_matrix_3"],
    "weight_tensors": ["...contractions.1.weights.0", "...weights.1", "...weights_max"]
  }
]
```

The two forms are mutually exclusive within one symmetric-contraction plan.

## MACE-POLAR model fields

`mace-polar-v1` is the separate architecture for MACE-POLAR-1. It is not a
general MACE checkpoint importer and must not reuse `mace-lite-v1` silently.

```toml
architecture = "mace-polar-v1"

[mace_polar]
model = "mace_polar.json"
reference_package = "mace"
reference_version = "0.3.16"
reference_commit = "4d2da09413ac1407f37cdbb6b81fa28e4c15655e"
selected_model = "polar-1-s"
# Allowed values are polar-1-s, polar-1-m, and polar-1-l.
```

The current loader recognizes this section as `mace-polar-v1`. Metadata-only and
inspection-only artifacts load as non-executable records, while a pinned
`polarmace-state-dict-v1` export with retained real tensors and execution
metadata may execute the restricted `polar-1-s`, `polar-1-m`, or `polar-1-l`
non-PBC neutral/spin=1 zero-field energy/forces subset. The loader validates
pinned `polar-1-m` and `polar-1-l` state-export manifests and retained tensor
inventories with model-specific constants and execution-plan layer counts.
Broader PolarMACE inputs and outputs documented
in `docs/MACE_POLAR_SPEC.md`, including reciprocal cell, volume, external field,
Fermi level, density coefficients, non-default total charge/spin, fixed-point
recursion settings beyond the pinned subset, and electrostatic outputs beyond
the current comparisons remain format/runtime extension points.

## `mace-polar-v1` metadata-only model file

For the initial PolarMACE loader slice, `[mace_polar].model` points to `mace_polar.json` with schema version `mace-polar-metadata-v1`. This file is a pinned provenance and inspection contract, not a converted tensor bundle.

Required pinned fields include:

- `architecture = "mace-polar-v1"`
- `runtime_status = "unsupported_metadata_only"`
- `selected_model` in the pinned set: `polar-1-s`, `polar-1-m`, or `polar-1-l`
- reference package `mace`, version `0.3.16`, commit `4d2da09413ac1407f37cdbb6b81fa28e4c15655e`
- artifact filename `MACE-POLAR-1-S.model`, upstream URL, SHA256 `e4495612037b3b3312633182882a38a694ecac9ea0be2b9889ac0b2a84a99510`, and byte size `33375439`
- atomic symbols and atomic numbers for elements 1 through 83
- inspected PolarMACE attributes: `r_max = 6.0`, `num_interactions = 1`, `num_recursion_steps = 2`, `atomic_multipoles_max_l = 1`, `field_feature_max_l = 1`, `field_feature_widths = [1.5, 3.0]`
- state-dict inspection counts: 245 tensors, 8203554 elements, dtype counts `{torch.bool: 3, torch.float32: 228, torch.int32: 1, torch.int64: 13}`
- full inventory path `../../polarmace_polar_1_s_full_inventory.json`, generated by the reference-only inspection tool with `--full-inventory`
- reference fixture paths `../../polarmace_polar_1_s_reference_20plus_f32.json` and `../../polarmace_polar_1_s_reference_20plus_f64.json` with 24 non-PBC structures, neutral charge, spin setting 1, and zero external field; the f64 fixture is the Rust f64-first parity target

The committed example lives at `fixtures/mace/v0_3_16_4d2da09/models/polarmace_polar_1_s_metadata.e3rs/`. Normal cargo tests must not require Python, PyTorch, MACE, e3nn, graph_longrange, or the untracked upstream `.model` artifact.

## `mace-polar-v1` tensor payload handoff

For f64 PolarMACE state exports, `[mace_polar].model` may point to a `polarmace-state-dict-v1` JSON such as `mace_polar_state.json`, and `[[tensors]]` entries may include `f64`, `f32`, `bool`, `i32`, and `i64` row-major payloads. This is a pinned exporter handoff only; it is not arbitrary PyTorch state_dict import. Public prediction is limited to the documented `polar-1-s/m/l` non-PBC neutral/spin=1/zero-field subset. Completion evidence for `polar-1-m/l` is recorded in `docs/MACE_POLAR_COMPLETION_AUDIT.md` and `docs/benchmark_summaries/2026-05-20-mace-polar-speedup.md`; stress, virial, PBC/cell, metadata-only, and incomplete state-export execution paths remain unsupported.

## eSEN-OAM model fields

`esen-oam-v1` is the planned architecture for FairChem `eSEN-30M-OAM`. It is a
pinned state-export handoff for the gated Hugging Face OMat24 checkpoint
`esen_30m_oam.pt`; the source checkpoint must remain outside git under ignored
reference-artifact caches.

```toml
architecture = "esen-oam-v1"

[esen_oam]
state = "esen_oam_state.json"
reference_package = "fairchem"
reference_version = "1.10.0"
reference_commit = "977a80328f2be44649b414a9907a1d6ef2f81e95"
selected_model = "esen-30m-oam"
state_schema = "esen-oam-state-dict-v1"
artifact_sha256 = "<64 hex chars>"
artifact_size_bytes = 1
```

The Rust loader accepts this section as a metadata-only state-export contract
for the pinned OAM checkpoint. It validates the committed state inventory and
then rejects Energy+Forces prediction until the executable Rust runtime graph is
implemented. Normal cargo tests must not require FairChem, PyTorch, ASE,
pymatgen, torch-geometric, Hugging Face authentication, or the gated checkpoint.

Before replacing this fail-closed state with an executable Rust runtime, the
reference artifact set must include:

- `fixtures/fairchem/current/esen_oam_artifact_manifest.json` with
  `scope.status = "inspected_reference_binary"`, artifact SHA256/byte size,
  checkpoint config, and tensor dtype/count/numel metadata.
- `outputs/reference-artifacts/fairchem/esen_oam_state_export.e3rs/` with
  `schema_version = "esen-oam-state-dict-v1"` and manifest-backed f64 tensor
  payloads preserving original FairChem tensor names.
- `fixtures/fairchem/current/esen_oam_state_manifest.toml` and
  `fixtures/fairchem/current/esen_oam_state_inventory.json` with the committed
  metadata-only copy of the 388 tensor contract used by normal Rust tests.
- `outputs/reference-artifacts/fairchem/esen_oam_model_tree.json` with
  `schema_version = "esen-oam-model-tree-v1"`, instantiated FairChem trainer
  and model types, module tree, selected head/backbone modules, config, and
  dtype inventories before and after f64 casting.
- `outputs/reference-artifacts/fairchem/esen_oam_upstream_benchmark_summary.json`
  with upstream FP64 Energy+Forces timings for 1 and 16 threads on the pinned
  deterministic structure set.
- `outputs/reference-artifacts/fairchem/esen_oam_radial_smoothness_scan.json`
  with the required 1D radial/cutoff value, derivative, central-difference
  self-consistency, and smoothness scan using the inspected model config.
- A passing `tools/reference/check_esen_oam_reference_readiness.py` report.

These reference files are evidence for implementation readiness only. They are
not a parity claim by themselves; parity still requires Rust FP64 outputs
against upstream full force arrays, energy finite-difference force checks, and
radial/cutoff value-derivative-smoothness tests.

The readiness report must also cross-check artifact identity. The SHA256 and
byte size recorded in the inspected artifact manifest, exported `.e3rs`
manifest, instantiated model-tree inspection, and upstream benchmark runs must
all refer to the same checkpoint.

## UMA-S-1.2 model fields

`uma-v1` is the pinned FairChem v2 UMA-S-1.2 runtime contract for
`selected_model = "uma-s-1p2"`. It is not broad FairChem compatibility and not
an arbitrary PyTorch checkpoint importer. The source checkpoint remains outside
git under ignored reference-artifact caches.

```toml
architecture = "uma-v1"

[uma]
state = "uma_state.json"
reference_package = "fairchem"
reference_version = "2.20.0"
reference_commit = "28292f090743f36c1151002964140b04d4b2c6af"
selected_model = "uma-s-1p2"
state_schema = "uma-s-1p2-state-dict-v1"
artifact_sha256 = "ba5c0d912efa22dc238e5fb1b5b7f66ee2e68c48c1b95b7cfd7fe1da5938398b"
artifact_size_bytes = 2333393167

[state_dict]
format = "manifest-f64-tensors-v1"
```

The loader validates the pinned FairChem v2 reference identity, checkpoint
SHA256/byte size, selected model, state schema, state-dict format, tensor
summary, tensor shapes, tensor dtypes, and tensor SHA256 metadata for the
exported `.e3rs` handoff. UMA payloads may be distributed either as the
manifest-listed individual `.bin` tensor files or as a root-level
`tensors.pack` single-file pack. If `tensors.pack` is present, runtime loading
uses it and does not require the individual `.bin` files. If `tensors.pack` is
absent but all `.bin` files are present, the loader falls back to the legacy
per-tensor path for backwards compatibility. Mixed partial `.bin` payloads
without a pack fail closed.

`tensors.pack` stores the same raw little-endian f64/i64 tensor bytes in
manifest order. It is read into safe aligned Rust storage, not memory-mapped.
Load-time validation checks the pack index ordering, data-section bounds,
8-byte alignment, manifest SHA256 for every tensor, and finite f64 values.
Metadata-only handoffs still fail closed for prediction and input preparation
because the fixed tensor payloads are not available. When the full verified
FP64 tensor payload bundle is present, the scoped OMAT runtime can prepare
inputs and run energy/conservative-force prediction through the UMA
Mixture-of-Linear-Experts graph, task/domain metadata, task-aware energy head,
and force-only VJP. The committed normal test suite remains payload-free; full
upstream FP64 parity and prepared-vs-unprepared runtime parity are gated behind
ignored tests that require the local 4.66 GB payload export under
`outputs/reference-artifacts/fairchem/`.

For verified UMA payloads, `load_llmlip_model` eagerly builds the expert-minor
MoLE relayout after state-export validation by default. This moves the one-time
FP64 transpose from the first FP64 prediction into model load, but it keeps an
additional in-memory copy of the 64-expert SO2 MoLE weights (approximately
2.3 GB for UMA-S-1.2). Set `LLMLIP_UMA_NO_PREBUILD_EXPERT_MINOR` to any value
before loading to skip that load-time prebuild and keep the original lazy
first-use behavior. The lazy path remains the authoritative fallback for
metadata-only models, direct internal construction paths, and any failed
best-effort prebuild.

Normal Rust tests must remain Python-free and must not require FairChem,
PyTorch, ASE, Hugging Face, Google Drive, or the 2.2 GB checkpoint binary.
