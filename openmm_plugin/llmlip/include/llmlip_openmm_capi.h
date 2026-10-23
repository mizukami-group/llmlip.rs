#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "llmlip_capi.h"

namespace llmlip_openmm {

enum class DType { F64, F32 };

struct Prediction {
  bool has_energy = false;
  double energy_kj_mol = 0.0;
  bool has_forces = false;
  std::vector<double> forces_kj_mol_per_nm;
};

class Error : public std::runtime_error {
 public:
  explicit Error(const std::string &message) : std::runtime_error(message) {}
};

class Model {
 public:
  static Model loadPath(const std::string &model_path);
  static Model loadBakedMaceOamMedium();

  Model(Model &&) noexcept = default;
  Model &operator=(Model &&) noexcept = default;

  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;

  void predict(const std::vector<int32_t> &atomic_numbers,
               const std::vector<double> &positions_nm,
               const std::optional<std::array<double, 9>> &box_vectors_nm,
               bool periodic,
               bool compute_energy,
               bool compute_forces,
               DType dtype,
               Prediction &out) const;

  const llmlip_model_capabilities_t &capabilities() const { return capabilities_; }

 private:
  explicit Model(llmlip_model_t *model);

  struct Deleter {
    void operator()(llmlip_model_t *model) const {
      if (model != nullptr) llmlip_model_free(model);
    }
  };

  std::unique_ptr<llmlip_model_t, Deleter> model_;
  llmlip_model_capabilities_t capabilities_{};
};

}  // namespace llmlip_openmm
