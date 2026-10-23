# llmlip OpenMM Native C ABI Bridge

This directory contains the native C++ bridge used by an OpenMM plugin or
application to call `llmlip-capi` directly, without going through
`openmm.PythonForce`.

The bridge intentionally keeps OpenMM-specific `ForceImpl` registration out of
the core evaluator. It accepts OpenMM-native units at the C++ boundary:

- positions and periodic box vectors in nm
- energy returned in kJ/mol
- forces returned in kJ/mol/nm

The model can be loaded either from a `.e3rs` path or from the source-generated
baked MACE-OAM Medium provider compiled into `libllmlip_capi`.

## Build

```bash
cargo build -p llmlip-capi --release \
  --features baked-mace-oam-medium,apple-metal,accelerate-blas,parallel

cmake -S openmm_plugin/llmlip -B target/openmm-llmlip-capi-bridge \
  -DLLMLIP_CAPI_INCLUDE_DIR="$PWD/crates/llmlip-capi/include" \
  -DLLMLIP_CAPI_LIBRARY="$PWD/target/release/libllmlip_capi.a"

cmake --build target/openmm-llmlip-capi-bridge
```

When `LLMLIP_CAPI_LIBRARY` points at the Rust `staticlib`, final OpenMM plugin
or application links must also include the native dependencies selected by the
Cargo feature set. The bridge target propagates pthread/dl/m defaults where
applicable and accepts additional values through
`LLMLIP_CAPI_EXTRA_LINK_LIBS` / `LLMLIP_CAPI_EXTRA_LINK_OPTIONS`; for example,
Apple Metal/Accelerate builds should add the required Apple frameworks at the
final link. Using the `llmlip-capi` `cdylib` (`.dylib`/`.so`) is an alternative
when you prefer Cargo to own those transitive native dependencies.

With `LLMLIP_OPENMM_BUILD_TESTS=ON`, CMake also builds
`llmlip_openmm_capi_link_smoke`, a tiny executable that links the bridge against
the configured real `LLMLIP_CAPI_LIBRARY` so local tests catch C ABI symbol or
signature drift at link time.

## Usage Sketch

An OpenMM `ForceImpl` can own one `llmlip_openmm::Model`, then call
`predict()` from its force/energy callback after copying the current OpenMM
positions into a contiguous `double` array.

```cpp
llmlip_openmm::Model model =
    llmlip_openmm::Model::loadBakedMaceOamMedium();

llmlip_openmm::Prediction prediction;
model.predict(atomic_numbers, positions_nm, box_nm, true, true, true,
              llmlip_openmm::DType::F32, prediction);
```

This is the native substrate for a full OpenMM plugin. The existing Python
adapter remains the easiest way to use llmlip from OpenMM today.


## Distribution scope

This project currently prepares source-only distribution. Local build outputs
are for local use; wheels, Python sdists and native binaries are not release
assets. Before redistributing a build, collect the applicable license texts
and notices for all bundled components and inspect the final archive.
