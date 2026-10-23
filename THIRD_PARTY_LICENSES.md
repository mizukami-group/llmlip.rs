# Third-party licenses and reference sources

## Scope and project license

This is a source-only distribution. Project-authored code is MIT OR Apache-2.0
unless a file states otherwise. The three LAMMPS C++ interface files named in
[their README](lammps_plugin/llmlip/README.md#license-and-distribution) are GPL-2.0-only.
The root licenses do not relicense third-party code, scientific data or model
weights. See [distribution policy](docs/distribution_policy.md).

## Source implementation provenance

The [function-level provenance review](docs/SOURCE_PROVENANCE.md) records upstream
implementation references, preserved license texts, and the unresolved ORB license question.
Cephes/SIMD attribution and terms are retained in
[LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY). This list is not a clearance for MIT-only distribution.

## Dependencies fetched separately

- [e3nn-core-rs](https://github.com/mizukami-group/e3nn-core-rs/tree/9da369cf1d854da6af1a46928bc953c7966e9382),
  pinned by Cargo.toml/Cargo.lock: MIT OR Apache-2.0. Its
  [third-party notices](https://github.com/mizukami-group/e3nn-core-rs/blob/9da369cf1d854da6af1a46928bc953c7966e9382/THIRD_PARTY_NOTICES.md)
  retain reference licenses and scientific data provenance, including the MACE/ASE
  covalent-radii table. These notices must accompany any applicable bundled code
  in a future binary distribution; neither numerical parity nor an upstream MIT
  label alone establishes rights to every underlying dataset.
- Other Rust crates are fetched at versions fixed in Cargo.lock, not vendored
  here. Their own licenses apply. Some declarations contain additional conditions
  (for example, Unicode-3.0 alongside MIT/Apache choices). Preserve all applicable
  notices when incorporating their code into a distribution.
- Python adapters use separately installed NumPy, ASE (LGPL-2.1-or-later),
  OpenMM and OpenMM-ML. The adapters do not include those packages. Their
  component licenses must be assessed separately if later bundled.
- LAMMPS is GPLv2 and obtained separately. See the plugin README and included
  GPL text for the interface. No LAMMPS executable is bundled.
- Optional native BLAS/accelerator providers are obtained separately; the
  project's code license does not grant redistribution rights to those libraries.

## Reference and conversion implementations

The following upstream implementations are used for inspection, conversion or
numerical reference comparisons. Their source trees and checkpoints are not
included here. An exporter importing an installed upstream package does not
change that package's license. This list records provenance, not a claim that
upstream implementation code was copied or that all model weights have the
same license as the upstream software.

| Upstream implementation | Reference revision | Code license / source |
|---|---|---|
| e3nn | 2aa7f58440a06b15352a2cbce01fa4c26f824969 | [MIT](https://github.com/e3nn/e3nn/blob/2aa7f58440a06b15352a2cbce01fa4c26f824969/LICENSE) |
| MACE | 4d2da09413ac1407f37cdbb6b81fa28e4c15655e | [MIT](https://github.com/ACEsuit/mace/blob/4d2da09413ac1407f37cdbb6b81fa28e4c15655e/LICENSE.md) |
| NequIP | 604ce73b33f067f1e55b69ed96c9e68b842e1409 | [MIT](https://github.com/mir-group/nequip/blob/604ce73b33f067f1e55b69ed96c9e68b842e1409/LICENSE) |
| Allegro | 6d05192cfc6047f22cb8b97cb86839667e9e484c | [MIT](https://github.com/mir-group/allegro/blob/6d05192cfc6047f22cb8b97cb86839667e9e484c/LICENSE) |
| NequiX | da0fb241f417dad1afafa4e723ad867667ee7445 | [MIT](https://github.com/atomicarchitects/nequix/blob/da0fb241f417dad1afafa4e723ad867667ee7445/LICENSE) |
| upet | 1f0b65c9efeb31abcfb9a14ca5859f0b3ce45e78 | [BSD-3-Clause](https://github.com/lab-cosmo/upet/blob/1f0b65c9efeb31abcfb9a14ca5859f0b3ce45e78/LICENSE) |
| fairchem | 977a80328f2be44649b414a9907a1d6ef2f81e95 | [MIT](https://github.com/FAIR-Chem/fairchem/blob/977a80328f2be44649b414a9907a1d6ef2f81e95/LICENSE.md) |
| ORB | a5f1a382a72126feff0594b19891e03cde5f843e | [Upstream license](https://github.com/orbital-materials/orb-models/blob/a5f1a382a72126feff0594b19891e03cde5f843e/LICENSE) |
| torch-dftd | v0.5.3 | [MIT](https://github.com/pfnet-research/torch-dftd/blob/v0.5.3/LICENSE) |

DFT-D3 reference data from torch-dftd were used in development; the full
parameter tables and D3 reference fixtures are **not included** in this source
candidate. Scientific references for the implementation are
[Grimme et al. (2010)](https://doi.org/10.1063/1.3382344) and
[Grimme et al. (2011)](https://doi.org/10.1002/jcc.21759).

## Included model material

Only the self-generated synthetic tensors under `examples/model.e3rs` are
included. Artifact manifests/inspection JSON record model identity, hashes,
shapes and execution metadata; they do not contain foundation-model tensors.
The synthetic example is for interface validation, not scientific prediction.

No foundation-model checkpoint or converted bundle, including PET-MAD-S, is
part of this distribution. [Model weight policy](docs/model_weight_policy.md)
distinguishes this project decision from rights granted by upstream licenses.
The converter registry links to upstream terms; license acceptance for a
checkpoint is separate from this project's source license.
