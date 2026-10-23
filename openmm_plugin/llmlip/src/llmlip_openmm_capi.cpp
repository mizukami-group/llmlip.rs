#include "llmlip_openmm_capi.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace llmlip_openmm {
namespace {

constexpr double kAngstromPerNm = 10.0;
constexpr double kEvToKjMol = 96.4853321233100184;
constexpr double kForceEvPerAngstromToKjMolPerNm = kEvToKjMol * kAngstromPerNm;

llmlip_dtype_t to_c_dtype(DType dtype) {
  return dtype == DType::F32 ? LLMLIP_DTYPE_F32 : LLMLIP_DTYPE_F64;
}

void check_status(llmlip_status_t status, const char *context) {
  if (status == LLMLIP_OK) return;
  const char *detail = llmlip_last_error();
  std::string message = context;
  if (detail != nullptr && detail[0] != '\0') {
    message += ": ";
    message += detail;
  }
  throw Error(message);
}

void validate_shape(std::size_t atom_count, const std::vector<double> &positions_nm) {
  if (atom_count == 0) {
    throw Error("atomic_numbers must contain at least one atom");
  }
  if (atom_count > static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    throw Error("atom_count exceeds llmlip C ABI int64 range");
  }
  if (atom_count > std::numeric_limits<std::size_t>::max() / 3) {
    throw Error("atom_count is too large for 3-coordinate buffers");
  }
  const std::size_t coordinate_count = atom_count * 3;
  if (positions_nm.size() != coordinate_count) {
    throw Error("positions_nm must contain exactly 3 values per atom");
  }
  for (double value : positions_nm) {
    if (!std::isfinite(value)) throw Error("positions_nm contains a non-finite value");
  }
}

double determinant3x3(const std::array<double, 9> &matrix) {
  return matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
         matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
         matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
}

std::array<double, 9> inverse3x3(const std::array<double, 9> &matrix) {
  const double det = determinant3x3(matrix);
  if (!std::isfinite(det) || std::fabs(det) < 1.0e-30) {
    throw Error("periodic OpenMM box is singular; cannot wrap llmlip positions");
  }
  const double inv_det = 1.0 / det;
  return {
      (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * inv_det,
      (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * inv_det,
      (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * inv_det,
      (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * inv_det,
      (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * inv_det,
      (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * inv_det,
      (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * inv_det,
      (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * inv_det,
      (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * inv_det,
  };
}

void wrap_positions_into_periodic_cell(std::vector<double> &positions_angstrom,
                                       const std::array<double, 9> &cell_angstrom) {
  const std::array<double, 9> inverse_cell = inverse3x3(cell_angstrom);
  for (std::size_t base = 0; base < positions_angstrom.size(); base += 3) {
    const double x = positions_angstrom[base + 0];
    const double y = positions_angstrom[base + 1];
    const double z = positions_angstrom[base + 2];
    double f0 = x * inverse_cell[0] + y * inverse_cell[3] + z * inverse_cell[6];
    double f1 = x * inverse_cell[1] + y * inverse_cell[4] + z * inverse_cell[7];
    double f2 = x * inverse_cell[2] + y * inverse_cell[5] + z * inverse_cell[8];
    if (!std::isfinite(f0) || !std::isfinite(f1) || !std::isfinite(f2)) {
      throw Error("periodic OpenMM positions produced non-finite fractional coordinates");
    }
    f0 -= std::floor(f0);
    f1 -= std::floor(f1);
    f2 -= std::floor(f2);
    positions_angstrom[base + 0] =
        f0 * cell_angstrom[0] + f1 * cell_angstrom[3] + f2 * cell_angstrom[6];
    positions_angstrom[base + 1] =
        f0 * cell_angstrom[1] + f1 * cell_angstrom[4] + f2 * cell_angstrom[7];
    positions_angstrom[base + 2] =
        f0 * cell_angstrom[2] + f1 * cell_angstrom[5] + f2 * cell_angstrom[8];
  }
}

}  // namespace

Model Model::loadPath(const std::string &model_path) {
  llmlip_model_t *model = nullptr;
  check_status(llmlip_model_load(model_path.c_str(), &model), "load llmlip model");
  return Model(model);
}

Model Model::loadBakedMaceOamMedium() {
  llmlip_model_t *model = nullptr;
  check_status(llmlip_model_load_baked_mace_oam_medium(&model),
               "load baked llmlip MACE-OAM Medium model");
  return Model(model);
}

Model::Model(llmlip_model_t *model) : model_(model) {
  if (model == nullptr) throw Error("llmlip model loader returned null");
  check_status(llmlip_model_capabilities(model_.get(), &capabilities_),
               "query llmlip model capabilities");
}

void Model::predict(const std::vector<int32_t> &atomic_numbers,
                    const std::vector<double> &positions_nm,
                    const std::optional<std::array<double, 9>> &box_vectors_nm,
                    bool periodic,
                    bool compute_energy,
                    bool compute_forces,
                    DType dtype,
                    Prediction &out) const {
  const std::size_t atom_count = atomic_numbers.size();
  validate_shape(atom_count, positions_nm);

  std::vector<double> positions_angstrom(positions_nm.size());
  for (std::size_t index = 0; index < positions_nm.size(); ++index) {
    positions_angstrom[index] = positions_nm[index] * kAngstromPerNm;
  }

  std::array<double, 9> cell_angstrom{};
  const double *cell_ptr = nullptr;
  if (periodic) {
    if (!box_vectors_nm.has_value()) {
      throw Error("periodic prediction requires box_vectors_nm");
    }
    for (std::size_t index = 0; index < cell_angstrom.size(); ++index) {
      if (!std::isfinite((*box_vectors_nm)[index])) {
        throw Error("box_vectors_nm contains a non-finite value");
      }
      cell_angstrom[index] = (*box_vectors_nm)[index] * kAngstromPerNm;
    }
    wrap_positions_into_periodic_cell(positions_angstrom, cell_angstrom);
    cell_ptr = cell_angstrom.data();
  }

  llmlip_structure_view_t structure{};
  structure.n_atoms_total = static_cast<int64_t>(atom_count);
  structure.n_atoms_owned = static_cast<int64_t>(atom_count);
  structure.atomic_numbers = atomic_numbers.data();
  structure.positions = positions_angstrom.data();
  structure.cell = cell_ptr;
  structure.pbc[0] = periodic ? 1 : 0;
  structure.pbc[1] = periodic ? 1 : 0;
  structure.pbc[2] = periodic ? 1 : 0;

  std::vector<double> forces_ev_per_angstrom;
  if (compute_forces) forces_ev_per_angstrom.assign(atom_count * 3, 0.0);

  llmlip_prediction_options_t options{};
  options.compute_energy = compute_energy ? 1 : 0;
  options.compute_forces = compute_forces ? 1 : 0;
  options.compute_virial = 0;
  options.compute_stress = 0;
  options.dtype = to_c_dtype(dtype);

  llmlip_prediction_out_t prediction{};
  prediction.forces = forces_ev_per_angstrom.empty() ? nullptr : forces_ev_per_angstrom.data();

  check_status(llmlip_predict_structure(model_.get(), &structure, &options, &prediction),
               "predict with llmlip model");

  out.has_energy = prediction.has_energy != 0;
  out.energy_kj_mol = prediction.energy * kEvToKjMol;
  out.has_forces = prediction.has_forces != 0;
  out.forces_kj_mol_per_nm.clear();
  if (out.has_forces) {
    out.forces_kj_mol_per_nm.resize(forces_ev_per_angstrom.size());
    for (std::size_t index = 0; index < forces_ev_per_angstrom.size(); ++index) {
      out.forces_kj_mol_per_nm[index] =
          forces_ev_per_angstrom[index] * kForceEvPerAngstromToKjMolPerNm;
    }
  }
}

}  // namespace llmlip_openmm
