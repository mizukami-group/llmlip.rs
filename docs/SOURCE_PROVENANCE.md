# Source provenance and unresolved license review

This document identifies implementation references, not just package dependencies.
The current review does **not** clear every source file for an unqualified
MIT-only distribution. Do not interpret model parity, absence of literal copies,
or the root dual-license declaration as proof of independent authorship.

## Reading the evidence

- **Numerical implementation correspondence**: distinctive coefficient tables and
  control flow agree; translation or adaptation requires origin review.
- **Behavioral reimplementation**: model-specific computation and numerical choices
  follow inspected upstream code. This is stronger than merely citing a paper,
  but does not by itself establish copying of copyrightable expression.
- **Interface/reference use**: imports, conversion, layouts and validation target
  upstream APIs; no claim of verbatim copying follows from this alone.

The Rust implementations add explicit buffers, validation, fixed-weight execution,
manual force derivatives and CPU optimizations. These changes do not eliminate
any obligations attached to material actually adapted from upstream.

## Numerical helpers: Cephes and SIMD notices

`crates/llmlip/src/mace_polar.rs` contains `polar_mace_erfc_cephes_positive`,
`polar_mace_polevl`, and `polar_mace_p1evl`. The erfc helper has no call sites in
this source tree, but remains part of the source distribution. Its P/Q/R/S
coefficient tables and x < 8 rational-approximation split correspond to Cephes
`ndtr.c` / `erfc`; 24 of 29 coefficients match the f64 values exactly and all
29 differ by at most 8 ULP after decimal rounding. This is concrete evidence
of Cephes-family numerical material, not only the mathematical definition of erfc.

Comparison source: [SciPy Cephes ndtr.h](https://github.com/scipy/scipy/blob/6e246d0b54dd55dc69232a0caae6772228a7ac25/scipy/special/xsf/cephes/ndtr.h),
which retains Stephen L. Moshier's copyright notice and identifies a 2024 C++
translation by SciPy developers. This comparison revision is **not** established
as the historical source consulted by this project. The historical intermediary remains unproven. For the Cephes material, however,
SciPy explicitly records author-authorized BSD-3-Clause distribution in its
bundled-license notice. That specific notice, rather than its top-level license,
is retained verbatim in [cephes.txt](../licenses/reference/cephes.txt).

`crates/llmlip/src/blas_backend.rs::exp_ps_avx512` explicitly identifies its
polynomial as Cephes expf. Its six coefficients and split ln(2) constants also
occur in [sse_mathfun.h](https://github.com/RJVB/sse_mathfun/blob/0f462645698320643decdb5f79d211733b2a97d6/sse_mathfun.h).
That implementation carries Julien Pommier's zlib terms and RJVB extension notices.
The Rust version uses AVX-512 FMA, conversion rounding and scalef, with different
clamping from the SSE implementation. Coefficient identity does not prove that
this particular SSE file was copied. The Cephes BSD notice and the SSE zlib notice are both retained conservatively;
[LICENSE-THIRD-PARTY](../LICENSE-THIRD-PARTY) maps the functions and changes.
These permissive terms are compatible with original MIT/Apache contributions
when their conditions are preserved. Unlike the erfc
helper, this function is called by the x86 parallel UMA path.

## Model implementation map

All local paths below are relative to `crates/llmlip/src/`.
Upstream revisions and license texts are in the table below.

| Local implementation | Upstream file / symbol | Evidence and classification |
|---|---|---|
| `mace_polar.rs`: realspace descriptors and charge energy | graph_longrange `realspace_electrostatics.py::charges_energy_from_graph`, `features.py::project_to_features_batch`; MACE `modules/extensions.py::PolarMACE`, `modules/field_blocks.py` | Behavioral reimplementation: same pseudo-charge representation, field constant, erf smoothing and denominator regularizer `1e-6`; development records explicitly describe replaying graph_longrange feature projection. MACE alone is an incomplete attribution. |
| `orb_v3_omol.rs::gnn_forward_impl`, charge/spin embedding, edge attention | ORB `orb_models/common/models/gns.py::AttentionInteractionNetwork.forward`; `nn_util.py::ChargeSpinEmbedding.forward`, `get_cutoff` | Behavioral reimplementation: conditioning, separate send/receive attention, concatenated edge/node messages and residual updates; matching f32 rounding before sinusoidal charge/spin embedding. Rust fuses operations and adds explicit derivatives. Treat any adapted expression as retaining Apache-2.0 obligations; a MIT option for original additions does not override them. |
| `pet_lite.rs`: attention and cutoff factors | metatrain `src/metatrain/pet/modules/transformer.py::AttentionBlock.forward`, `manual_attention`, `CartesianTransformer` | Behavioral reimplementation: QKV/head layouts, log-clamped cutoff mask and attention scaling. upet is the wrapper; metatrain is the actual transformer reference. |
| `nequix.rs::nequix_rms_layer_norm` | NequiX `nequix/layer_norm.py::RMSLayerNorm.__call__` | Behavioral reimplementation: degree-balanced weights `1 / (irrep dimension * number of blocks)`, reductions, epsilon and affine application. Upstream explicitly traces this to fairchem eSEN and EquiformerV2. Retain that ancestry, including Yi-Lun Liao's notice, where adapted material applies. |
| `uma_mole.rs`: routing and mixture linear/SO2 functions | fairchem UMA `src/fairchem/core/models/uma/nn/mole.py::_softmax`, `MOLE.forward` | Behavioral reimplementation: softmax followed by addition of 0.005 without renormalization, expert-weight contraction, per-system routing. Training/checkpoint machinery is not reproduced. UMA reference revision is separately pinned below. |
| `esen_block.rs`, `esen_mlp.rs`, model dispatch | fairchem `src/fairchem/core/models/esen/esen_block.py`, `nn/so2_layers.py`, `nn/so3_layers.py`, `nn/activation.py`, `nn/layer_norm.py` | Model block and convention reference. Upstream rotation code also identifies e3nn ancestry; do not treat Meta's root notice as replacing nested notices. |
| `mace_lite.rs`, `mace_mh1.rs`, MACE model dispatch | MACE `modules/models.py`, `modules/blocks.py`, `modules/symmetric_contraction.py` | Model and contraction reference. MACE source is MIT; separate checkpoint restrictions do not automatically license this Rust code. Source headers also name Ilyes Batatia and Gregor Simm; field blocks name Will Baldwin and Ilyes Batatia. |
| NequIP / Allegro model paths and converters | pinned NequIP / Allegro models, irreps and interaction specifications | Architecture, artifact-layout and validation references; no whole-file translation established by this review. Keep upstream notices if adapted material is identified. |
| `dftd3.rs` and reference tooling | torch-dftd v0.5.3, D3/D3(BJ) scientific definitions | Numerical reference and parameter conventions. Full upstream parameter tables are excluded here; this does not establish rights to separately supplied tables. |

UMA reference: [fairchem 28292f090743f36c1151002964140b04d4b2c6af](https://github.com/FAIR-Chem/fairchem/tree/28292f090743f36c1151002964140b04d4b2c6af/src/fairchem/core/models/uma).
NequiX also uses e3nn-jax normalization conventions. The exact e3nn-jax revision
behind the normalized-SiLU constant has not been established in this review.

## Reference license texts

These are unchanged upstream license files retained for attribution and review.
Including them does not assert that every listed library's code was copied,
nor does it establish which historical intermediary was consulted.

Cephes BSD and SIMD zlib notices are additionally mapped in
[LICENSE-THIRD-PARTY](../LICENSE-THIRD-PARTY). Their notice omission has been
addressed; this does not resolve the separate ORB scope question.

| Reference | Revision | Retained text |
|---|---|---|
| mace | [4d2da09413ac1407f37cdbb6b81fa28e4c15655e](https://github.com/ACEsuit/mace/tree/4d2da09413ac1407f37cdbb6b81fa28e4c15655e) | [MIT](../licenses/reference/mace.txt) |
| nequip | [604ce73b33f067f1e55b69ed96c9e68b842e1409](https://github.com/mir-group/nequip/tree/604ce73b33f067f1e55b69ed96c9e68b842e1409) | [MIT](../licenses/reference/nequip.txt) |
| allegro | [6d05192cfc6047f22cb8b97cb86839667e9e484c](https://github.com/mir-group/allegro/tree/6d05192cfc6047f22cb8b97cb86839667e9e484c) | [MIT](../licenses/reference/allegro.txt) |
| nequix | [da0fb241f417dad1afafa4e723ad867667ee7445](https://github.com/atomicarchitects/nequix/tree/da0fb241f417dad1afafa4e723ad867667ee7445) | [MIT](../licenses/reference/nequix.txt) |
| fairchem | [977a80328f2be44649b414a9907a1d6ef2f81e95](https://github.com/FAIR-Chem/fairchem/tree/977a80328f2be44649b414a9907a1d6ef2f81e95) | [MIT](../licenses/reference/fairchem.txt) |
| equiformer-v2 | [977a80328f2be44649b414a9907a1d6ef2f81e95](https://github.com/FAIR-Chem/fairchem/tree/977a80328f2be44649b414a9907a1d6ef2f81e95) | [MIT](../licenses/reference/equiformer-v2.txt) |
| e3nn | [2aa7f58440a06b15352a2cbce01fa4c26f824969](https://github.com/e3nn/e3nn/tree/2aa7f58440a06b15352a2cbce01fa4c26f824969) | [MIT](../licenses/reference/e3nn.txt) |
| upet | [1f0b65c9efeb31abcfb9a14ca5859f0b3ce45e78](https://github.com/lab-cosmo/upet/tree/1f0b65c9efeb31abcfb9a14ca5859f0b3ce45e78) | [BSD-3-Clause](../licenses/reference/upet.txt) |
| metatrain | [21990f4796220f8984de6a726c164dda0ba2e802](https://github.com/metatensor/metatrain/tree/21990f4796220f8984de6a726c164dda0ba2e802) | [BSD-3-Clause](../licenses/reference/metatrain.txt) |
| orb | [a5f1a382a72126feff0594b19891e03cde5f843e](https://github.com/orbital-materials/orb-models/tree/a5f1a382a72126feff0594b19891e03cde5f843e) | [Apache-2.0](../licenses/reference/orb.txt) |
| graph-longrange | [0e21d5546c482d08388a08eb4d948e833227ce47](https://github.com/WillBaldwin0/graph_electrostatics/tree/0e21d5546c482d08388a08eb4d948e833227ce47) | [upstream MIT-labelled grant](../licenses/reference/graph-longrange.txt) |

The graph_longrange LICENSE at the pinned revision is a shortened MIT-labelled
permission grant: it lacks a copyright-owner line and the usual warranty paragraph.
It is retained verbatim, without inventing an owner or silently inserting clauses.
The upstream project identifies Will Baldwin as an author. We preserve the
actual permission grant verbatim and do not represent it as the standard full
MIT text or invent a copyright year/owner statement. A fully standardized
upstream notice remains an external clarification; the local notice cannot
supply rights absent from upstream.

## License assessment and remaining actions

MIT/BSD-licensed adaptations can coexist with original MIT/Apache-2.0 contributions
when their required copyright, permission and disclaimer texts are retained.
An attribution URL alone does not replace those texts.

For ORB-derived expression, Apache-2.0 section 4 requires its license, applicable
notices and prominent modification notices. The architecture correspondence alone
does not settle derivative-work status, but current evidence is insufficient to
certify all ORB implementation code as independently authored and MIT-only.
The complete ORB module now carries Apache-2.0 and modification notices; see
[ORB license scope](ORB_LICENSE_SCOPE.md). The aggregate package metadata no longer
offers an unrestricted MIT-only choice. Historical line-by-line authorship is
not claimed to have been established.

Apache-2.0-only material is not compatible with a GPL-2.0-only combined LAMMPS
plugin. The existing MIT option for original code cannot cure that mismatch.
A documented separation/exclusion of such material, a compatible upstream grant,
or substantiated independent implementation is needed before clearing that binary.
This is distinct from hosting separate source components together.

The separately fetched e3nn-core dependency has its own third-party notices,
including MACE/ASE numerical-table provenance. This review does not independently
clear every primitive in that separate repository. Python/C++ adapters and exporters
were checked for literal code matches in the earlier audit; lack of a match cannot
prove absence of translated or edited code. Model weight terms are separate.

The conservative ORB release boundary and modification notice are now recorded.
Required follow-up: verify
all license texts in any future package/binary. No source clearance is inferred
from passing numerical tests. See [distribution policy](distribution_policy.md).
