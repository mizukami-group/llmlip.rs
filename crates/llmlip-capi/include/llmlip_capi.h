#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llmlip_model llmlip_model_t;

typedef enum {
  LLMLIP_OK = 0,
  LLMLIP_ERROR_NULL_POINTER = 1,
  LLMLIP_ERROR_INVALID_ARGUMENT = 2,
  LLMLIP_ERROR_MODEL_LOAD = 3,
  LLMLIP_ERROR_UNSUPPORTED = 4,
  LLMLIP_ERROR_PREDICTION = 5,
  LLMLIP_ERROR_PANIC = 6,
} llmlip_status_t;

typedef enum {
  LLMLIP_DTYPE_F64 = 0,
  LLMLIP_DTYPE_F32 = 1,
} llmlip_dtype_t;

typedef struct {
  double model_cutoff;
  double mpi_halo_cutoff;
  int64_t message_passing_depth;

  int32_t supports_non_pbc;
  int32_t supports_partial_pbc;
  int32_t supports_full_pbc;
  int32_t supports_forces;
  int32_t supports_external_graph;
  int32_t supports_external_graph_ghost_atoms;
  int32_t supports_external_graph_global_virial;
  int32_t supports_external_graph_global_stress;
  int32_t supports_global_virial;
  int32_t supports_global_stress;
  int32_t supports_per_atom_energy;
  int32_t supports_per_atom_virial;

  int32_t supports_f64;
  int32_t supports_f32;

  int32_t requires_total_charge;
  int32_t requires_spin_multiplicity;
  int32_t supports_selected_head;
  int64_t supported_head_count;
} llmlip_model_capabilities_t;

typedef struct {
  int64_t n_atoms_total;
  int64_t n_atoms_owned;
  const int32_t *atomic_numbers;
  const double *positions;
  const double *cell;
  int32_t pbc[3];
} llmlip_structure_view_t;

typedef struct {
  int64_t n_centers;
  const int64_t *centers;
  const uint8_t *center_owned;
  const int64_t *edge_offsets;
  const int64_t *dst;
  const double *edge_vector;
  double model_cutoff;
} llmlip_external_graph_view_t;

typedef struct {
  int32_t compute_energy;
  int32_t compute_forces;
  int32_t compute_virial;
  int32_t compute_stress;

  llmlip_dtype_t dtype;
  const char *selected_head;

  int32_t has_total_charge;
  double total_charge;

  int32_t has_spin_multiplicity;
  double spin_multiplicity;
} llmlip_prediction_options_t;

typedef struct {
  int32_t has_energy;
  double energy;

  /*
   * If forces are requested, the caller must set `forces` to a writable buffer
   * of at least `structure.n_atoms_total * 3` doubles before calling
   * `llmlip_predict_external_graph` or `llmlip_predict_structure`.
   *
   * External-graph prediction writes forces for every supplied local and
   * ghost/image atom slot. Structure prediction currently requires
   * `n_atoms_owned == n_atoms_total` and writes the same `n_atoms_total * 3`
   * layout. The buffer uses atom-major xyz order:
   *
   *   forces[3 * atom + 0], forces[3 * atom + 1], forces[3 * atom + 2]
   *
   * The pointer may be NULL only when forces are not requested. If the pointer
   * is NULL while forces are returned, the call fails with a status code instead
   * of writing. Use standard `double` alignment.
   */
  int32_t has_forces;
  double *forces;

  int32_t has_virial;
  double virial[9];

  int32_t has_stress;
  double stress[9];
} llmlip_prediction_out_t;

llmlip_status_t llmlip_model_load(const char *model_dir, llmlip_model_t **out);
llmlip_status_t llmlip_model_load_baked_mace_oam_medium(llmlip_model_t **out);
void llmlip_model_free(llmlip_model_t *model);

llmlip_status_t llmlip_model_capabilities(
    const llmlip_model_t *model,
    llmlip_model_capabilities_t *out);

llmlip_status_t llmlip_predict_external_graph(
    const llmlip_model_t *model,
    const llmlip_structure_view_t *structure,
    const llmlip_external_graph_view_t *graph,
    const llmlip_prediction_options_t *options,
    llmlip_prediction_out_t *out);

llmlip_status_t llmlip_predict_structure(
    const llmlip_model_t *model,
    const llmlip_structure_view_t *structure,
    const llmlip_prediction_options_t *options,
    llmlip_prediction_out_t *out);

/*
 * Error strings are stored thread-locally. Read `llmlip_last_error()` from the
 * same thread that received a non-OK status, before the next llmlip C ABI call on
 * that thread. The returned pointer is owned by the library and remains valid
 * until another llmlip C ABI call on the same thread or `llmlip_clear_error()`.
 */
const char *llmlip_last_error(void);
void llmlip_clear_error(void);

#ifdef __cplusplus
}
#endif
