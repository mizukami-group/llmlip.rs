# llmlip C ABI (`libllmlip_capi`)

English | [日本語](c_abi.ja.md)

The `llmlip-capi` crate exposes the llmlip inference engine as a plain C ABI:
load a `.e3rs` model directory once, then request energy / forces / virial /
stress for a structure or for an externally built neighbor graph. It is the
integration layer used by the LAMMPS plugin and by any C/C++/Fortran host code.

## What is provided

| Artifact | Path (after build) | Notes |
|---|---|---|
| C header | `crates/llmlip-capi/include/llmlip_capi.h` | single header, C and C++ (`extern "C"`) |
| Static library | `target/release/libllmlip_capi.a` | `crate-type = "staticlib"` |
| Shared library | `target/release/libllmlip_capi.dylib` (macOS) / `.so` (Linux) | `crate-type = "cdylib"` |
| Rust rlib | `target/release/libllmlip_capi.rlib` | for Rust consumers only |

All exported symbols are prefixed `llmlip_`. The baked MACE-OAM Medium loader is
always present in the C ABI; builds that do not enable
`baked-mace-oam-medium` return `LLMLIP_ERROR_UNSUPPORTED` from that loader
instead of omitting the symbol. `reference-fixtures` only gates internal tests.

## Build and link

```sh
cargo build -p llmlip-capi --release
```

For a redistribution build that embeds the MACE-OAM Medium weights and enables
the Apple-Silicon Metal path, build with the forwarded llmlip features:

```sh
cargo build -p llmlip-capi --release \
  --features baked-mace-oam-medium,apple-metal,accelerate-blas,parallel
```

Link either artifact against your program:

```sh
# static (no runtime library path needed)
cc -std=c11 -Icrates/llmlip-capi/include demo.c target/release/libllmlip_capi.a -o demo

# dynamic (set LD_LIBRARY_PATH / DYLD_LIBRARY_PATH to target/release at run time)
cc -std=c11 -Icrates/llmlip-capi/include demo.c -Ltarget/release -lllmlip_capi -o demo
```

When static linking pulls in extra system libraries on your platform, ask the
Rust toolchain for the exact list:
`RUSTFLAGS="--print native-static-libs" cargo rustc -p llmlip-capi --release --lib`.

## Minimal working example

Save as `demo.c`, build as above, and run from the repository root (the bundled
synthetic demo model lives at `examples/model.e3rs`):

```c
#include <stdio.h>
#include <llmlip_capi.h>

static int check(llmlip_status_t status, const char *what) {
  if (status != LLMLIP_OK) {
    fprintf(stderr, "%s failed (status %d): %s\n", what, (int)status, llmlip_last_error());
    return 1;
  }
  return 0;
}

int main(void) {
  llmlip_model_t *model = NULL;
  if (check(llmlip_model_load("examples/model.e3rs", &model), "llmlip_model_load")) return 1;
  /* Baked redistribution builds can instead call:
   * llmlip_model_load_baked_mace_oam_medium(&model)
   */

  llmlip_model_capabilities_t caps;
  if (check(llmlip_model_capabilities(model, &caps), "llmlip_model_capabilities")) return 1;
  printf("model cutoff: %.2f\n", caps.model_cutoff);

  /* Water molecule, no periodic boundary conditions. */
  int32_t atomic_numbers[3] = {8, 1, 1};
  double positions[9] = {
      0.000000, 0.000000, 0.0,  /* O */
      0.957200, 0.000000, 0.0,  /* H */
      -0.239987, 0.927297, 0.0, /* H */
  };
  llmlip_structure_view_t structure = {
      .n_atoms_total = 3,
      .n_atoms_owned = 3,
      .atomic_numbers = atomic_numbers,
      .positions = positions,
      .cell = NULL, /* may be NULL when all pbc flags are 0 */
      .pbc = {0, 0, 0},
  };

  llmlip_prediction_options_t options = {
      .compute_energy = 1,
      .compute_forces = 1,
      .dtype = LLMLIP_DTYPE_F64,
      .selected_head = NULL, /* NULL selects the default head */
  };

  double forces[9] = {0}; /* caller-owned: n_atoms_total * 3 doubles */
  llmlip_prediction_out_t out = {0};
  out.forces = forces;

  if (check(llmlip_predict_structure(model, &structure, &options, &out),
            "llmlip_predict_structure")) {
    llmlip_model_free(model);
    return 1;
  }

  if (out.has_energy) printf("energy: %.6f eV\n", out.energy);
  if (out.has_forces) {
    for (int atom = 0; atom < 3; ++atom) {
      printf("force[%d] = (%.6f, %.6f, %.6f)\n", atom, forces[3 * atom + 0],
             forces[3 * atom + 1], forces[3 * atom + 2]);
    }
  }

  llmlip_model_free(model);
  return 0;
}
```

Expected output: `model cutoff: 5.00`, a finite energy, and one force line per atom.

## API reference

| Function | Arguments | Returns | On error |
|---|---|---|---|
| `llmlip_model_load` | `model_dir` (NUL-terminated UTF-8 path to a `.e3rs` directory), `out` (receives handle) | `llmlip_status_t` | non-OK status, `*out` left `NULL` |
| `llmlip_model_load_baked_mace_oam_medium` | `out` (receives handle) | `llmlip_status_t` | non-OK status, `*out` left `NULL`; returns `LLMLIP_ERROR_UNSUPPORTED` when the library was built without `baked-mace-oam-medium` |
| `llmlip_model_free` | `model` handle | `void` | `NULL` is ignored; never call twice on the same handle |
| `llmlip_model_capabilities` | `model`, `out` (`llmlip_model_capabilities_t *`) | `llmlip_status_t` | non-OK status (`LLMLIP_ERROR_NULL_POINTER` on `NULL` args) |
| `llmlip_predict_structure` | `model`, `structure`, `options`, `out` | `llmlip_status_t` | non-OK status; `out` flags reset to 0 |
| `llmlip_predict_external_graph` | `model`, `structure`, `graph`, `options`, `out` | `llmlip_status_t` | non-OK status; `out` flags reset to 0 |
| `llmlip_last_error` | — | `const char *` (thread-local message; empty string when none) | — |
| `llmlip_clear_error` | — | `void` | — |

Input contracts (checked by the library, violations return an error status):

- All numeric inputs/outputs are `double` (FP64). `LLMLIP_DTYPE_F32` only switches
  the internal compute precision; results are still written as `double`.
- `out.forces` must point to a caller-owned buffer of `n_atoms_total * 3` doubles
  (atom-major xyz) whenever forces are requested; it may be `NULL` only when they
  are not. A `NULL` buffer while forces are returned fails with a status code.
- `llmlip_predict_structure` requires `n_atoms_owned == n_atoms_total`. `cell`
  (9 doubles, rows are lattice vectors) is required when any `pbc` flag is 1 and
  may be `NULL` otherwise; `pbc` entries must be 0 or 1.
- External graph: `edge_offsets` has `n_centers + 1` entries; the edge count is
  its last entry; `edge_vector` holds `3 * n_edges` doubles; `center_owned`
  entries must be 0 or 1.
- `virial` / `stress` are 9-double row-major fields, valid only when
  `has_virial` / `has_stress` is 1. Models that cannot compute stress or virial
  fail with an explicit error status instead of silently returning zeros.

## Status codes and errors

| Code | Value | Meaning |
|---|---|---|
| `LLMLIP_OK` | 0 | success |
| `LLMLIP_ERROR_NULL_POINTER` | 1 | a required pointer was `NULL` (including a missing `cell` for a periodic structure or a missing forces buffer) |
| `LLMLIP_ERROR_INVALID_ARGUMENT` | 2 | bad input data: negative counts, `pbc`/`center_owned` not 0/1, unknown atomic number, inconsistent external graph |
| `LLMLIP_ERROR_MODEL_LOAD` | 3 | the `.e3rs` directory could not be loaded |
| `LLMLIP_ERROR_UNSUPPORTED` | 4 | the model does not support the requested compute dtype, prediction head, or external-graph prediction |
| `LLMLIP_ERROR_PREDICTION` | 5 | the engine failed during prediction (e.g. stress/virial requested from a model that cannot compute them) |
| `LLMLIP_ERROR_PANIC` | 6 | a Rust panic was caught at the ABI boundary; the process stays alive |

After any non-OK status, `llmlip_last_error()` returns a human-readable message.
The message is stored thread-locally: read it on the same thread that received
the status, before the next llmlip C ABI call on that thread. The pointer is
owned by the library and stays valid until the next llmlip C ABI call on that
thread or `llmlip_clear_error()`. Do not free it.

## Thread safety and ownership

- A loaded model is read-only. The library guards its per-handle scratch buffers
  with mutexes, so prediction calls on the same handle from multiple threads are
  serialized per prediction function. The simplest robust pattern is one handle
  per thread, or external serialization of calls on a shared handle.
- The caller owns all input arrays and the forces buffer; they only need to stay
  valid for the duration of each call. The library never frees caller memory.
- The model handle is owned by the library and is released only by
  `llmlip_model_free` (exactly once; `NULL` is a no-op). Do not free a handle
  while another thread is still using it.

## Related documentation

- [`.e3rs` model format](MODEL_FORMAT_E3RS.md)
- [Converting published models to `.e3rs`](../converters/README.md)
- [LAMMPS `pair_style llmlip` plugin](../lammps_plugin/llmlip/README.md) — a complete consumer of this ABI
- [OpenMM native C ABI bridge](../openmm_plugin/llmlip/README.md)
