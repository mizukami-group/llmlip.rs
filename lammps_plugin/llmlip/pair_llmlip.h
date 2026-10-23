// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026 Mizukami Group and llmlip contributors
// See LICENSE-GPL-2.0 in this directory.

/* -*- c++ -*- */
#ifdef PAIR_CLASS
// clang-format off
PairStyle(llmlip,PairLlmlip);
// clang-format on
#else

#ifndef LMP_PAIR_LLMLIP_H
#define LMP_PAIR_LLMLIP_H

#include "pair.h"

#include "llmlip_capi.h"

#include <cstdint>
#include <string>
#include <vector>

namespace LAMMPS_NS {

class PairLlmlip : public Pair {
 public:
  explicit PairLlmlip(class LAMMPS *);
  ~PairLlmlip() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  double memory_usage() override;

 protected:
  void allocate();

 private:
  enum class DType { F64, F32 };
  enum class VirialMode { Auto, Direct, Fdotr, Off };
  enum class ResolvedVirialMode { Direct, Fdotr, Off };
  enum class Rank1Mode { Structure, ExternalGraph };
  enum class ModelSource { Path, BakedMaceOamMedium };

  void parse_settings(int, char **);
  void load_model();
  void validate_model_capabilities();
  void validate_domain_capabilities();
  void ensure_comm_halo_cutoff();
  void build_type_to_atomic_number();
  void build_structure_view(bool include_ghosts);
  void build_external_graph_from_lammps_list();
  bool use_external_graph_path() const;
  ResolvedVirialMode resolved_virial_mode() const;
  llmlip_dtype_t capi_dtype() const;

  int atomic_number_for_symbol(const char *) const;
  void fail_on_llmlip(llmlip_status_t, const char *);

  llmlip_model_t *model_ = nullptr;
  llmlip_model_capabilities_t caps_{};

  DType dtype_ = DType::F64;
  VirialMode virial_mode_ = VirialMode::Auto;
  Rank1Mode rank1_mode_ = Rank1Mode::Structure;
  ModelSource model_source_ = ModelSource::Path;

  std::string model_path_;
  std::string selected_head_;

  bool has_total_charge_ = false;
  double total_charge_ = 0.0;

  bool has_spin_multiplicity_ = false;
  double spin_multiplicity_ = 0.0;

  std::vector<int32_t> type_to_atomic_number_;
  std::vector<int32_t> atomic_numbers_;
  std::vector<double> positions_;
  std::vector<double> cell_;

  std::vector<int64_t> centers_;
  std::vector<uint8_t> center_owned_;
  std::vector<int64_t> edge_offsets_;
  std::vector<int64_t> edge_dst_;
  std::vector<double> edge_vectors_;
  std::vector<double> force_buffer_;

  llmlip_structure_view_t structure_view_{};
  llmlip_external_graph_view_t graph_view_{};
};

}    // namespace LAMMPS_NS

#endif
#endif
