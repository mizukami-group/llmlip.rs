# llmlip Python bindings

Python, ASE, and OpenMM bindings for the Rust-native `llmlip`
fixed-weight MLIP inference runtime.

The public package is `llmlip`. The native module is private and imported as
`llmlip._llmlip`.

## OpenMM

Install the optional OpenMM dependencies with:

```bash
pip install maturin numpy "openmm>=8.5" "openmmml>=1.6"
maturin develop --manifest-path crates/llmlip-py/Cargo.toml
```

Use the OpenMM-ML entry point for a pure ML system:

```python
from openmmml import MLPotential

potential = MLPotential("llmlip", modelPath="model.e3rs")
system = potential.createSystem(topology, dtype="float64")
```

In a baked MACE-OAM Medium build, `modelPath` can be omitted:

```python
potential = MLPotential("llmlip")
system = potential.createSystem(topology, dtype="float64")
```

The lower-level OpenMM helper returns a force without adding it to the system:

```python
from llmlip.openmm import create_llmlip_force

force = create_llmlip_force(topology, system, "model.e3rs")
system.addForce(force)
```

With the baked extension, `create_llmlip_force(topology, system)` loads the
compiled-in model through `llmlip.load_baked_model()`.

The MVP uses `openmm.PythonForce` and calls `llmlip.Model.predict()` directly.
It assumes `.e3rs` model units are Angstrom for positions, eV for energy, and
eV/Angstrom for forces, matching the current manifest examples. OpenMM values
are converted to kJ/mol and kJ/mol/nm. The topology atom count must match the
system particle count, and every ML atom species must be supported by the llmlip
model when the model exposes a finite species set. For periodic systems, the
adapter wraps OpenMM positions back into the primary cell before calling
`predict()` so long unwrapped MD trajectories preserve periodic-neighbor
semantics.

Partial-PBC semantics and full OpenMM `ForceImpl` registration are deferred
follow-ups; full-periodic and non-periodic systems are the supported OpenMM
paths. A native C++ C ABI evaluator bridge lives under
`openmm_plugin/llmlip` for plugin/application code that wants to bypass
`PythonForce`.
OpenMM-ML mixed systems without interpolation are part of the MVP. In current
OpenMM-ML, interpolated mixed systems attempt to serialize the `PythonForce`
callback and fail because the callback binds the native `llmlip.Model` object,
which is not pickleable, so `createMixedSystem(..., interpolate=True)` is not
supported by this MVP.


## Distribution scope

This project currently prepares source-only distribution. Local build outputs
are for local use; wheels, Python sdists and native binaries are not release
assets. Before redistributing a build, collect the applicable license texts
and notices for all bundled components and inspect the final archive.
