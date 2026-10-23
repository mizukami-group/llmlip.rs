# Python and ASE Interface Specification

## Purpose

This document turns the `llmlip_python_ase_handoff.zip` discussion notes into a
repository-local contract for completing the Python and ASE integration.

The target is a production-usable Python package named `llmlip` that exposes the
same fixed-weight Rust inference runtime used by the CLI. The package must
support both direct in-process PyO3 calls and a CLI-backed portability path.

## Required Context

Implementation work must follow:

- `AGENTS.md`
- `PROJECT_SPEC.md`
- `development_TASKS.md`
- `docs/PRODUCT_DIRECTION.md`
- `docs/RELEASE_AND_REFACTOR_PLAN.md`
- `docs/FORCE_ONLY_VJP_SPEC.md`
- `docs/MATH_CONVENTIONS.md`
- `docs/PBC_NEIGHBOR_LIST_SPEC.md`
- `docs/MODEL_FORMAT_E3RS.md`
- this document
- `docs/PYTHON_ASE_IMPLEMENTATION_PLAN.md`

The handoff ZIP is useful background, but this document and the implementation
plan are the authoritative repo-local summary.

## Scope

Implement:

- A new workspace crate `crates/llmlip-py`.
- A PyO3 extension module imported as `llmlip._llmlip`.
- A Python package `llmlip`.
- A single public `llmlip.Model` facade around `llmlip::LlmlipModel`.
- Direct model loading and prediction from Python.
- A single public ASE calculator, `llmlip.ase.LlmlipCalculator`.
- Backend selection via `backend="auto" | "direct" | "cli"`.
- A machine-readable CLI JSON path for the CLI backend.
- Python, ASE, CLI, packaging, and parity tests.
- Short user documentation for standalone Python and ASE usage.

## Non-goals

Do not add training, parameter gradients, generic AD, GPU support,
arbitrary PyTorch checkpoint loading, broad e3nn API compatibility, or a
separate public Python class per model architecture.

Do not make Rust runtime crates depend on Python, PyO3, NumPy, ASE, PyTorch, or
upstream ML packages. PyO3 and Python dependencies belong only to
`crates/llmlip-py` and Python test/docs tooling.

## Public Python API

Package layout:

```text
crates/llmlip-py/
  Cargo.toml
  pyproject.toml
  src/lib.rs
  python/llmlip/
    __init__.py
    _backend.py
    ase.py
    py.typed
```

Native module name:

```text
llmlip._llmlip
```

Core API:

```python
import llmlip

model = llmlip.Model("model.e3rs")
prediction = model.predict(
    species=["O", "H", "H"],
    positions=positions,          # shape (n, 3), Angstrom
    cell=None,                    # or shape (3, 3), row-major cell vectors
    pbc=(False, False, False),
    energy=True,
    forces=True,
    stress=False,
    virial=False,
    dtype="float64",              # "float64", "float32", "f64", "f32"
    head=None,
    total_charge=None,
    spin_multiplicity=None,
    mace_mh1_wigner_mode="true_float64",
)
```

`Prediction` exposes:

- `energy: float | None` in eV.
- `forces: numpy.ndarray | None` with shape `(n, 3)` in eV/Angstrom.
- `stress: numpy.ndarray | None` with shape `(3, 3)` in eV/Angstrom^3.
- `virial: numpy.ndarray | None` with shape `(3, 3)` in eV.

`Model`, `Prediction`, `PredictionInputRequirements`, and `ModelCapabilities`
must provide concise `__repr__` output for interactive debugging.

Returned arrays are `float64`, even when an internal f32 compute path is used,
matching Rust public output semantics.

## ASE API

```python
from llmlip.ase import LlmlipCalculator

calc = LlmlipCalculator("model.e3rs", backend="auto")
calc = LlmlipCalculator("model.e3rs", backend="direct")
calc = LlmlipCalculator("model.e3rs", backend="cli")
```

Constructor parameters:

- `model_path: str | Path | None`
- `model: llmlip.Model | None`
- `backend: "auto" | "direct" | "cli" | None`
- `binary: str | Path | None`
- `dtype: "float64" | "float32" | "model-native"`
- `head: str | None`
- `total_charge: float | None`
- `spin_multiplicity: float | None`
- `mace_mh1_wigner_mode: "true_float64" | "upstream_artifact"`

Rules:

- Exactly one of `model_path` or `model` is required for direct backend.
- CLI backend requires `model_path`.
- `backend=None` means `LLMLIP_ASE_BACKEND` or default `auto`.
- `auto` tries direct first, then CLI. If both fail, the exception must include
  both causes.
- Stress remains fail-fast for non-full-PBC ASE atoms unless model
  capabilities later prove a narrower safe case.
- Rust stress `(3, 3)` is converted to ASE Voigt order
  `[xx, yy, zz, yz, xz, xy]`.
- ASE cell filters use `virial = -volume * stress`; the Rust stress convention
  already satisfies `virial = -stress * volume`, so the ASE adapter must not
  flip the stress sign.

## Backend Contract

Use one internal backend protocol:

```python
class PredictionBackend(Protocol):
    def predict(self, request: PredictionRequest) -> Prediction: ...
    def close(self) -> None: ...
```

`DirectBackend`:

- Imports `llmlip._llmlip`.
- Keeps the Rust model loaded in-process.
- Converts Python inputs to owned Rust `Structure` values.
- Releases the GIL around Rust-only prediction work.

`CliBackend`:

- Binary resolution and CLI invocation live in
  `crates/llmlip-py/python/llmlip/_backend.py` (`CliBackend`).
- Resolution order: explicit `binary`, `LLMLIP_BINARY`, `PATH`, repo-local
  `target/release/llmlip`.
- Prefers CLI JSON once available.
- May keep legacy extXYZ/text parsing as a fallback.
- Emits a `RuntimeWarning` when selected because this backend starts one
  `llmlip` process per prediction and is unsuitable for MD-style repeated
  workflows unless the overhead is acceptable.
- A persistent `llmlip serve --protocol jsonl` path is a later optimization
  and is not exposed as a public `LlmlipCalculator` option until implemented.

## Rust API Extensions

Add a public method:

```rust
impl LlmlipModel {
    #[must_use]
    pub fn architecture(&self) -> &'static str;
}
```

Add and re-export a conservative `ModelCapabilities` type:

```rust
pub struct ModelCapabilities {
    pub architecture: &'static str,
    pub supports_energy: bool,
    pub supports_forces: bool,
    pub supports_stress: bool,
    pub supports_virial: bool,
    pub supports_float64: bool,
    pub supports_float32: bool,
    pub supports_prepared_input: bool,
    pub stress_requires_full_pbc: bool,
    pub virial_requires_full_pbc: bool,
    pub supported_heads: Vec<String>,
    pub input_requirements: PredictionInputRequirements,
}
```

Capabilities are for diagnostics and preflight only. Prediction-time Rust
validation remains the source of truth.

## CLI JSON Protocol

Add `llmlip predict --json`:

```bash
llmlip predict --model model.e3rs --structure input.extxyz --energy --forces --json
```

JSON stdout schema:

```json
{
  "schema_version": "llmlip-prediction-v1",
  "energy": -123.456,
  "forces": [[0.0, 0.1, 0.2]],
  "stress": null,
  "virial": null
}
```

Compatibility requirements:

- Without `--json`, existing text output and force extXYZ output must remain
  unchanged.
- Current `--forces-out PATH` behavior must remain.
- Existing `--forces PATH` alias must either continue to work or be replaced
  only with an explicit compatibility migration and tests.
- JSON mode must provide forces without requiring a force-output file.
- If both JSON and force extXYZ are requested, produce both.
- Errors remain stderr plus non-zero exit for the initial implementation.

## Error Mapping

Expose Python exceptions:

- `LlmlipError`
- `ModelLoadError`
- `PredictionError`
- `UnsupportedCapabilityError`
- `InvalidStructureError`
- `BackendUnavailableError`
- `CliBackendError`

Map important Rust `ModelError` variants to the most specific Python exception
available. Do not silently ignore unsupported dtype, head, stress, virial,
charge/spin, or prepared-input requests.

## Packaging

Use `maturin` with mixed Rust/Python layout:

- `python-source = "python"`
- `module-name = "llmlip._llmlip"`
- include `py.typed`
- optional dependency group `ase`
- test dependency group `test`

Feature forwarding from `llmlip-py` to `llmlip` must preserve existing CBLAS
selection rules. Only one of `accelerate-blas`, `intel-mkl`, and `openblas` may
be selected.

## Acceptance Criteria

The interface is complete when:

- `llmlip.Model` can load a `.e3rs` model and predict energy/forces from NumPy.
- `LlmlipCalculator` works with `backend="direct"`, `backend="cli"`, and
  `backend="auto"`.
- Direct and CLI results match on at least one small committed fixture.
- ASE energy/forces match direct model predictions.
- Stress conversion is tested for a full-PBC case where supported, and
  non-full-PBC stress fails clearly.
- Error paths are tested for invalid shape, missing backend, unsupported dtype,
  unsupported head, and missing charge/spin where applicable.
- A wheel build smoke test can import `llmlip`.
- User docs include standalone Python and ASE examples.

Quality gates:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
maturin develop --manifest-path crates/llmlip-py/Cargo.toml
python -m pytest crates/llmlip-py/tests
```
