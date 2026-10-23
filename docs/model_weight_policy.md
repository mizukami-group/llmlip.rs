# Model weights

This source tree distributes only the small self-generated demonstration model
under `examples/model.e3rs`. It includes no upstream foundation checkpoint or
converted foundation-model tensor bundle. Artifact manifests record provenance
and hashes; they are not model weights.

Obtain upstream checkpoints from their official sources and review their terms
before downloading, converting, embedding or redistributing them. The code
license does not grant rights to upstream models. Registry records and
`THIRD_PARTY_LICENSES.md` capture provenance and license information available
when the model was integrated; consult the upstream terms for your intended use.

Keep locally acquired weights, converted models and upstream checkouts outside
the source tree or under ignored `outputs/` and `third_party/` directories.
Do not include them in source archives, wheels, releases or commits. Use the
bundled synthetic model for ordinary tests. Conversion tools require explicit
inputs and do not make model redistribution part of the build process.


## Project policy versus upstream permissions

All registered foundation models use the project's `converter_only` policy,
including PET-MAD-S. This means we do not distribute their checkpoints or
converted bundles. It is not a claim that upstream licenses forbid every
redistribution. Gated access also does not by itself determine redistribution
rights. Converting a model does not replace its original terms with MIT/Apache.

- [UMA](https://huggingface.co/facebook/UMA/blob/main/LICENSE) and
  [OMat24/eSEN](https://huggingface.co/facebook/OMAT24/blob/main/LICENSE) have
  custom terms with conditional redistribution provisions. Review the license
  applicable to the actual artifact, including acceptance and acceptable use.
- MACE-MH-1 and MACE-Polar use the
  [Academic Software Licence](https://github.com/gabor1/ASL/blob/main/ASL.md).
  It restricts use to academic non-commercial purposes and has conditional
  copying/modification/distribution provisions; it does not grant unrestricted
  commercial use and is not the MACE code's MIT license.
- PET-MAD-S is identified as BSD-3-Clause by its upstream model card. We still
  exclude its weights. Any future redistribution needs a separate review of
  the exact artifact and inclusion of its license text and copyright notices.

The registry records names and source links, not a blanket legal permission
boolean. `redistribution.policy` describes only what this project distributes.
