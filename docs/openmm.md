# Using llmlip models from OpenMM

English | [日本語](openmm.ja.md)

The `llmlip` Python package can drive [OpenMM](https://openmm.org) simulations
with a fixed-weight `.e3rs` model as the force provider. Two integration layers
are shipped: an [OpenMM-ML](https://github.com/openmm/openmm-ml) adapter
(`llmlip.openmmml`, registered under the potential name `llmlip`) and a
lower-level helper (`llmlip.openmm.create_llmlip_force`) that builds an
`openmm.PythonForce` whose callback calls `llmlip.Model.predict()` directly.
A native C++ C ABI bridge is also provided under
[openmm_plugin/llmlip](../openmm_plugin/llmlip/README.md) for host code that
wants to bypass `PythonForce`.

## Prerequisites

The package is not yet published on PyPI. Build it from the repository root
with [maturin](https://www.maturin.rs), then install the OpenMM dependencies
declared by the `openmm` optional-dependency group in
[crates/llmlip-py/pyproject.toml](../crates/llmlip-py/pyproject.toml):

```bash
pip install maturin numpy
maturin develop --release --manifest-path crates/llmlip-py/Cargo.toml
pip install "openmm>=8.5" "openmmml>=1.6"
```

For the baked MACE-OAM Medium extension used by the ASE/OpenMM default path on
Apple Silicon, build the Python module with the baked and Metal features:

```bash
maturin develop --release \
  --manifest-path crates/llmlip-py/Cargo.toml \
  --features baked-mace-oam-medium,apple-metal,accelerate-blas,parallel
```

At runtime, `llmlip` sets `LLMLIP_MACE_METAL=1` by default unless the caller
sets another value before importing the package.

Requirements checked at force-creation time:

- An OpenMM build that provides `openmm.PythonForce`; otherwise an
  `ImportError` is raised immediately.
- OpenMM-ML is needed only for the `llmlip.openmmml` adapter. The factory is
  exposed through the `openmmml.potentials` entry point
  (`llmlip = "llmlip.openmmml:LlmlipPotentialImplFactory"`); it can also be
  registered explicitly with
  `openmmml.MLPotential.registerImplFactory("llmlip", LlmlipPotentialImplFactory())`.

## Minimal working example

Uses the bundled synthetic demo model
[examples/model.e3rs](../examples/model.e3rs) (species H and O, float64 only):

```python
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential

topology = app.Topology()
chain = topology.addChain()
residue = topology.addResidue("MOL", chain)
for i, symbol in enumerate(["O", "H", "H"]):
    topology.addAtom(f"{symbol}{i}", app.Element.getBySymbol(symbol), residue)

potential = MLPotential("llmlip", modelPath="examples/model.e3rs")
system = potential.createSystem(topology, dtype="float64", removeCMMotion=False)

positions_ang = np.array(
    [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.927297, 0.0]]
)
context = openmm.Context(
    system,
    openmm.VerletIntegrator(0.001),
    openmm.Platform.getPlatformByName("Reference"),
)
context.setPositions((positions_ang / 10.0) * unit.nanometer)
state = context.getState(getEnergy=True, getForces=True)
print(state.getPotentialEnergy())  # kJ/mol
```

The lower-level helper returns a force without adding it to the system. The
system must already contain one particle per topology atom:

```python
from llmlip.openmm import create_llmlip_force

force = create_llmlip_force(topology, system, "examples/model.e3rs")
system.addForce(force)
```

In a baked MACE-OAM Medium build, omit `modelPath` / `model_path` to use the
compiled-in model:

```python
potential = MLPotential("llmlip")
system = potential.createSystem(topology, dtype="float32", removeCMMotion=False)

force = create_llmlip_force(topology, system, dtype="float32")
```

## Reference

`create_llmlip_force(topology, system, model_path=None, *, ...)` keyword
arguments. The OpenMM-ML `createSystem`/`createMixedSystem` paths forward
`dtype`, `precision`, `head`, `total_charge`, `spin_multiplicity`, and
`mace_mh1_wigner_mode` under the same names; the force group is set via
OpenMM-ML's own `forceGroup` argument, and only `modelPath`/`model_path`
(not a pre-loaded `model`) is accepted at `MLPotential` construction:

| Argument | Default | Meaning |
|---|---|---|
| `model_path` / `model` | baked MACE-OAM Medium when both are omitted | Path to a `.e3rs` model, or a pre-loaded `llmlip.Model`. Passing both is an error. Omitting both calls `llmlip.load_baked_model()`, which requires a build with `baked-mace-oam-medium`. |
| `atoms` | `None` (all atoms) | Indices of the ML subset. Non-ML atoms receive zero force from this force object. |
| `force_group` | `0` | OpenMM force group of the returned force. |
| `dtype` | `"float64"` | `"float64"`/`"f64"`/`"fp64"` or `"float32"`/`"f32"`/`"fp32"`. |
| `precision` | `None` | `"double"` or `"single"` (OpenMM-style alias). Must not conflict with `dtype`. |
| `head` | `None` | Prediction head; validated against the model capabilities. |
| `total_charge` | `None` | Required when the model declares it as an input requirement. OpenMM-ML alias: `charge`. |
| `spin_multiplicity` | `None` | Required when the model declares it as an input requirement. OpenMM-ML aliases: `multiplicity`, `spin`. |
| `mace_mh1_wigner_mode` | `"true_float64"` | MACE-MH1 only; alternative: `"upstream_artifact"`. |

Unit conversions performed by the adapter (llmlip models use Angstrom / eV /
eV/Angstrom; OpenMM uses nm / kJ/mol):

| Quantity | Direction | Factor |
|---|---|---|
| Positions | nm → Angstrom | × 10 |
| Energy | eV → kJ/mol | × 96.4853321233100184 |
| Forces | eV/Angstrom → kJ/mol/nm | × 964.853321233100184 |

## Error behavior and limitations

- Energy and forces only: the adapter calls `predict()` with
  `stress=False, virial=False`. There is no virial/stress path to OpenMM.
- The Python adapter still enters through a `PythonForce` callback. The
  underlying llmlip runtime may use the baked Metal path when the extension is
  built with `apple-metal` and `LLMLIP_MACE_METAL=1`; otherwise it uses the
  normal CPU path. The bundled tests exercise the OpenMM `Reference` platform.
- Fail-fast capability validation at force creation
  (`UnsupportedCapabilityError`): models without energy or force support,
  unsupported `dtype`, topology species outside the model's supported set,
  unknown `head`, and missing required `total_charge`/`spin_multiplicity`.
- `ValueError` at force creation: both `model_path` and `model`,
  conflicting `dtype`/`precision`, unknown dtype alias, topology atom count ≠
  system particle count, out-of-range/duplicate/non-1D `atoms`, ML atoms
  without an element, and unsupported extra keyword arguments in the OpenMM-ML
  adapter.
- Periodicity: if the topology has periodic box vectors or the system uses
  PBC, the box is treated as a full 3D PBC contract; partial PBC is
  intentionally unsupported. Positions are wrapped back into the primary cell
  before each `predict()` call. A singular box or non-finite fractional
  coordinates raise `PredictionError`.
- OpenMM-ML mixed systems work without interpolation
  (`createMixedSystem(..., interpolate=False)`). `interpolate=True` raises
  `openmm.OpenMMException` because OpenMM-ML tries to pickle the
  `PythonForce` callback, which binds the non-pickleable native
  `llmlip.Model`.

Tests covering all of the above:
[crates/llmlip-py/tests/test_openmm.py](../crates/llmlip-py/tests/test_openmm.py).

## Related documents

- ASE calculator and Python API: [docs/PYTHON_ASE_INTERFACE_SPEC.md](PYTHON_ASE_INTERFACE_SPEC.md)
- Python package overview: [crates/llmlip-py/README.md](../crates/llmlip-py/README.md)
- `.e3rs` model format: [docs/MODEL_FORMAT_E3RS.md](MODEL_FORMAT_E3RS.md)
