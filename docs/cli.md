# llmlip CLI

English | [日本語](cli.ja.md)

`llmlip` is the command-line front end of the Rust-native fixed-weight equivariant
MLIP inference engine. It loads `.e3rs` model directories and predicts energy,
forces, stress, and virial for extXYZ structures on the CPU, with FP64 as the
default compute dtype.

## Prerequisites and build

```sh
cargo build --release -p llmlip-cli
./target/release/llmlip --help
```

The binary is named `llmlip` and is written to `target/release/llmlip`.

## Minimal working example

The repository bundles a synthetic demo model (`examples/model.e3rs`, H/O species,
FP64-only) and a 3-atom water-like structure (`examples/input.extxyz`):

```sh
$ llmlip validate-model --model examples/model.e3rs
model validation ok

$ llmlip predict --model examples/model.e3rs --structure examples/input.extxyz --energy
energy = 2.8927585729719403e0

$ llmlip predict --model examples/model.e3rs --structure examples/input.extxyz \
    --energy --forces --json
{
  "schema_version": "llmlip-prediction-v1",
  "energy": 2.8927585729719403,
  "forces": [
    [-0.0137316644530655, -0.017634105420898622, 0.0],
    [0.05699796467573395, -0.029977567698224745, 0.0],
    [-0.043266300222668454, 0.04761167311912337, 0.0]
  ],
  "stress": null,
  "virial": null
}
```

(The JSON above is reformatted for brevity; the actual output pretty-prints each force component on its own line. Values are verbatim.)

## Subcommands

llmlip provides five subcommands (plus the clap-generated `help`): `predict`, `finite-difference-forces`,
`benchmark-predict`, `batch-predict`, and `validate-model`.

### Common flags

These flags are shared by `predict`, `finite-difference-forces`, `benchmark-predict`,
and `batch-predict` (`--dtype` is not available on `finite-difference-forces`):

| Flag | Default | Description |
| --- | --- | --- |
| `--model <MODEL>` | required | Path to an `.e3rs` model directory. |
| `--dtype <DTYPE>` | `float64` | Compute dtype: `float64` or `float32` (aliases `fp64`/`f64`, `fp32`/`f32`). Models without float32 support return an explicit error. |
| `--head <HEAD>` | – | Model head name for multi-head architectures. Headless models reject this option explicitly instead of silently ignoring it. |
| `--total-charge <X>` | – | Total charge for models that require global charge conditioning. |
| `--spin-multiplicity <X>` | – | Spin multiplicity for models that require global spin conditioning. |
| `--mace-mh1-wigner-mode <MODE>` | `true-float64` | MACE-MH-1 only. `true-float64` uses the f64 Clebsch-Gordan constants compiled into the Rust runtime; `upstream-artifact` matches the upstream TorchScript artifact and is intended for reference comparisons. |

### predict

At least one of `--energy`, `--forces`, `--forces-out <PATH>`, `--stress`, or
`--virial` is required.

| Flag | Description |
| --- | --- |
| `--structure <STRUCTURE>` | Input extXYZ file (single frame). |
| `--energy` | Compute the total energy (text mode prints `energy = ...`). |
| `--forces [<PATH>]` | Request forces. Without `PATH` this is supported only with `--json`. With `PATH` it also writes legacy force extXYZ output. |
| `--forces-out <PATH>` | Request forces and write an extXYZ file with a `force` column (and the energy when requested). Cannot be combined with `--forces <PATH>`. |
| `--stress` | Compute the 3x3 stress tensor (text mode prints it in eV/angstrom^3, row-major). |
| `--virial` | Compute the 3x3 virial tensor (text mode prints it in eV, row-major). |
| `--json` | Emit a `llmlip-prediction-v1` JSON document on stdout instead of text output. |

### finite-difference-forces

Compares analytic forces against central finite differences of the energy and
prints a `llmlip-force-finite-difference-report-v1` JSON report on stdout.

| Flag | Default | Description |
| --- | --- | --- |
| `--structure <STRUCTURE>` | required | Input extXYZ file (single frame). |
| `--fd-eps <FD_EPS>` | `0.0001` | Central-difference displacement; must be finite and positive. |
| `--fd-tol <FD_TOL>` | `0.005` | Per-component tolerance on the absolute analytic-vs-FD delta; must be finite and non-negative. |
| `--fd-mode <FD_MODE>` | `max_force` | `representative` and `max_force` check the single component with the largest absolute force; `full_3n` checks all 3N components. |
| `--component-start <N>` | `0` | First component index when slicing a `full_3n` run. |
| `--component-limit <N>` | all | Maximum number of components checked from `--component-start`. |
| `--include-components` | off | Include every per-component record in the JSON report (`components` array). |

The report contains `base_energy`, delta statistics (`max_abs_delta`, `rms_delta`,
percentiles), `failed_component_count`, the up-to-10 `worst_components`, and a boolean
`passed`. The exit code is 0 even when `passed` is `false`; consumers must read
the `passed` field.

### benchmark-predict

Requires exactly one of `--structure <PATH>` (single-frame extXYZ) or
`--fixture <PATH>` (JSON fixture with a `structures` array), plus at least one of
`--energy`, `--forces`, `--stress`, `--virial`. Prints a
`llmlip-predict-benchmark-v1.1` JSON report on stdout with timing statistics
(`mean_ms`, `median_ms`, `p95_ms`, `min_ms`, `max_ms`, `stdev_ms`, `samples_ms`),
a `checksum`, a `measurement_scope` block, and an `environment` block (logical
CPU count and thread-related environment variables).

| Flag | Default | Description |
| --- | --- | --- |
| `--warmup <N>` | `3` | Untimed warmup passes. |
| `--iterations <N>` | `10` | Timed passes; must be at least 1. |
| `--prepared-graph` | off | Prepare supported graph/topology inputs once before timed samples. |
| `--profile-stages` | off | Emit eSEN-OAM stage-level timing counters in the JSON report. |
| `--label <LABEL>` | `llmlip_predict` | Label recorded in the report. |

### batch-predict

Requires at least one of `--energy`, `--forces`, `--stress`, `--virial`. Writes a
`llmlip-batch-predict-v1` JSON report to `--out` with per-structure results
(energy/forces/stress/virial, timing, atom count, element species) and prints
progress to stderr. Per-structure failures are recorded in each result's `error`
field and counted in `error_count` instead of aborting the run.

| Flag | Default | Description |
| --- | --- | --- |
| `--structures <STRUCTURES>` | required | Path to a multi-frame extXYZ file. |
| `--out <OUT>` | required | Output JSON file path for the full batch result. |
| `--progress-interval <N>` | `100` | Print progress every N structures (0 disables progress lines). |
| `--profile-stages` | off | Emit eSEN-OAM stage-level timing counters per structure. |

When the model requires `total_charge`/`spin_multiplicity`, the values are read
per frame from the extXYZ frame metadata; the `--total-charge` and
`--spin-multiplicity` CLI flags take precedence, and missing required values fail
that frame with an explicit error.

### validate-model

```sh
$ llmlip validate-model --model examples/model.e3rs
model validation ok
```

Takes only `--model` and exits non-zero with `error: failed to validate model: ...`
when the model directory is invalid.

## JSON output schema (`predict --json`)

The `predict --json` document always contains exactly these five keys:

| Key | Type | Notes |
| --- | --- | --- |
| `schema_version` | string | `"llmlip-prediction-v1"`. |
| `energy` | number or null | Total energy; `null` when `--energy` was not requested. |
| `forces` | array of `[fx, fy, fz]` or null | One row per atom, input order; `null` when forces were not requested. |
| `stress` | 3x3 array or null | Row-major; `null` when `--stress` was not requested. |
| `virial` | 3x3 array or null | Row-major; `null` when `--virial` was not requested. |

A requested quantity is never emitted as `null`: if the model cannot provide it,
the command fails with an error instead of printing the document.

## Error behavior

All errors are printed to stderr as `error: <message>` and the process exits
with code 1. Messages observed with the bundled demo model:

- No output flag: `predict requires at least one of --energy, --forces, --forces-out <PATH>, --stress, or --virial`.
- `--forces` without `PATH` and without `--json`: `--forces without PATH is supported only with --json; use --forces-out <PATH> for legacy extXYZ output`.
- `--forces <PATH>` combined with `--forces-out`: `use only one of --forces [PATH] or --forces-out <PATH>`.
- Stress/virial on an unsupported model is an explicit error, never silent zeros: `prediction failed: synthetic radial model does not support stress or virial prediction`. Models that do support stress/virial still require full PBC.
- `--dtype float32` on an FP64-only model: `prediction failed: prediction compute dtype Float32 is unsupported for this model`.

## Related documents

- [docs/MODEL_FORMAT_E3RS.md](MODEL_FORMAT_E3RS.md) — the `.e3rs` model directory format.
- [converters/README.md](../converters/README.md) — converting upstream checkpoints to `.e3rs`.
- [docs/PYTHON_ASE_INTERFACE_SPEC.md](PYTHON_ASE_INTERFACE_SPEC.md) — Python/ASE interface.
- [docs/c_abi.md](c_abi.md) — C ABI (`llmlip_capi.h`).
- [docs/openmm.md](openmm.md) — OpenMM integration.
