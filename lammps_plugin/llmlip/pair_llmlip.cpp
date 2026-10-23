// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026 Mizukami Group and llmlip contributors
// See LICENSE-GPL-2.0 in this directory.

#include "pair_llmlip.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "info.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

PairLlmlip::PairLlmlip(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  ghostneigh = 1;
  centroidstressflag = CENTROID_NOTAVAIL;
  no_virial_fdotr_compute = 0;
}

PairLlmlip::~PairLlmlip()
{
  if (copymode) return;

  if (model_) llmlip_model_free(model_);
  model_ = nullptr;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(map);
    memory->destroy(cutghost);
  }
}

void PairLlmlip::allocate()
{
  allocated = 1;
  const int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair_llmlip:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair_llmlip:cutsq");
  memory->create(map, n + 1, "pair_llmlip:map");
  memory->create(cutghost, n + 1, n + 1, "pair_llmlip:cutghost");

  for (int i = 1; i <= n; ++i)
    for (int j = i; j <= n; ++j) setflag[i][j] = 0;

  type_to_atomic_number_.assign(static_cast<std::size_t>(n + 1), 0);
}

void PairLlmlip::settings(int narg, char **arg)
{
  parse_settings(narg, arg);

  if (model_source_ == ModelSource::Path && model_path_.empty())
    error->all(FLERR, "Pair style llmlip requires 'model <path>' or 'baked mace-oam-medium'");

  load_model();
  validate_model_capabilities();

  no_virial_fdotr_compute = (resolved_virial_mode() == ResolvedVirialMode::Direct) ? 1 : 0;
}

void PairLlmlip::parse_settings(int narg, char **arg)
{
  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "model") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "pair_style llmlip model", error);
      if (model_source_ != ModelSource::Path || !model_path_.empty())
        error->all(FLERR, "Pair style llmlip accepts only one model source");
      model_source_ = ModelSource::Path;
      model_path_ = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "baked") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "pair_style llmlip baked", error);
      if (model_source_ != ModelSource::Path || !model_path_.empty())
        error->all(FLERR, "Pair style llmlip accepts only one model source");
      // Accept both user-facing and upstream artifact aliases. The "oam0"
      // spelling matches the pinned MACE-OAM artifact names.
      if (strcmp(arg[iarg + 1], "mace-oam-medium") == 0 ||
          strcmp(arg[iarg + 1], "mace-oam0-medium") == 0 ||
          strcmp(arg[iarg + 1], "mace_oam_medium") == 0) {
        model_source_ = ModelSource::BakedMaceOamMedium;
      } else {
        error->all(FLERR, "Unknown pair_style llmlip baked model: {}", arg[iarg + 1]);
      }
      iarg += 2;
    } else if (strcmp(arg[iarg], "dtype") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "pair_style llmlip dtype", error);
      if (strcmp(arg[iarg + 1], "f64") == 0)
        dtype_ = DType::F64;
      else if (strcmp(arg[iarg + 1], "f32") == 0)
        dtype_ = DType::F32;
      else
        error->all(FLERR, "Unknown pair_style llmlip dtype: {}", arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "virial") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "pair_style llmlip virial", error);
      if (strcmp(arg[iarg + 1], "direct") == 0)
        virial_mode_ = VirialMode::Direct;
      else if (strcmp(arg[iarg + 1], "fdotr") == 0)
        virial_mode_ = VirialMode::Fdotr;
      else if (strcmp(arg[iarg + 1], "off") == 0)
        virial_mode_ = VirialMode::Off;
      else
        error->all(FLERR, "Unknown pair_style llmlip virial mode: {}", arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "rank1_mode") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "pair_style llmlip rank1_mode", error);
      if (strcmp(arg[iarg + 1], "structure") == 0 || strcmp(arg[iarg + 1], "compact") == 0)
        rank1_mode_ = Rank1Mode::Structure;
      else if (strcmp(arg[iarg + 1], "external_graph") == 0 ||
               strcmp(arg[iarg + 1], "ghost_graph") == 0)
        rank1_mode_ = Rank1Mode::ExternalGraph;
      else
        error->all(FLERR, "Unknown pair_style llmlip rank1_mode: {}", arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "head") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "pair_style llmlip head", error);
      selected_head_ = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "total_charge") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "pair_style llmlip total_charge", error);
      total_charge_ = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      has_total_charge_ = true;
      iarg += 2;
    } else if (strcmp(arg[iarg], "spin_multiplicity") == 0) {
      if (iarg + 2 > narg)
        utils::missing_cmd_args(FLERR, "pair_style llmlip spin_multiplicity", error);
      spin_multiplicity_ = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      has_spin_multiplicity_ = true;
      iarg += 2;
    } else {
      error->all(FLERR, "Unknown pair_style llmlip keyword: {}", arg[iarg]);
    }
  }
}

void PairLlmlip::load_model()
{
  if (model_) {
    llmlip_model_free(model_);
    model_ = nullptr;
  }

  if (model_source_ == ModelSource::BakedMaceOamMedium) {
    fail_on_llmlip(llmlip_model_load_baked_mace_oam_medium(&model_),
                   "load baked llmlip MACE-OAM Medium model");
  } else {
    fail_on_llmlip(llmlip_model_load(model_path_.c_str(), &model_), "load llmlip model");
  }
  fail_on_llmlip(llmlip_model_capabilities(model_, &caps_), "query llmlip model capabilities");
}

void PairLlmlip::validate_model_capabilities()
{
  if (!caps_.supports_forces)
    error->all(FLERR, "Pair style llmlip requires an llmlip model with force prediction support");

  const bool external_graph_path = use_external_graph_path();
  if (external_graph_path && !caps_.supports_external_graph)
    error->all(FLERR, "Pair style llmlip external-graph path requires external graph support");

  if (external_graph_path && !caps_.supports_external_graph_ghost_atoms)
    error->all(FLERR,
               "Pair style llmlip requires external graph ghost/image atom support for LAMMPS");

  if (dtype_ == DType::F32 && !caps_.supports_f32)
    error->all(FLERR, "Requested pair_style llmlip dtype f32, but model does not support it");

  if (!caps_.supports_f64)
    error->all(FLERR, "Pair style llmlip requires f64 model support");

  if (caps_.requires_total_charge && !has_total_charge_)
    error->all(FLERR, "Pair style llmlip model requires total_charge");

  if (caps_.requires_spin_multiplicity && !has_spin_multiplicity_)
    error->all(FLERR, "Pair style llmlip model requires spin_multiplicity");

  if (!selected_head_.empty() && !caps_.supports_selected_head)
    error->all(FLERR, "Pair style llmlip model does not support selected heads");

  if (virial_mode_ == VirialMode::Direct) {
    if (external_graph_path && !caps_.supports_external_graph_global_virial)
      error->all(FLERR,
                 "Requested pair_style llmlip virial direct, but external graph path does not "
                 "support it");
    if (!external_graph_path && !caps_.supports_global_virial)
      error->all(FLERR,
                 "Requested pair_style llmlip virial direct, but structure path does not support it");
  }

  if (caps_.model_cutoff <= 0.0 || !std::isfinite(caps_.model_cutoff))
    error->all(FLERR, "Pair style llmlip model reported invalid model_cutoff");

  if (external_graph_path) {
    const double required_halo =
        caps_.model_cutoff * static_cast<double>(std::max<int64_t>(caps_.message_passing_depth, 1));
    if (caps_.mpi_halo_cutoff + 1.0e-12 < required_halo || !std::isfinite(caps_.mpi_halo_cutoff))
      error->all(FLERR, "Pair style llmlip model reported invalid mpi_halo_cutoff");
  }
}

void PairLlmlip::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  if (narg != 2 + atom->ntypes) error->all(FLERR, "Incorrect args for pair coefficients");

  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Pair style llmlip requires pair_coeff * * Element...");

  for (int i = 0; i < atom->ntypes; ++i) {
    if (strcmp(arg[2 + i], "NULL") == 0)
      error->all(FLERR, "Pair style llmlip does not support NULL atom type mappings yet");
  }

  map_element2type(narg - 2, arg + 2);
  build_type_to_atomic_number();
}

void PairLlmlip::build_type_to_atomic_number()
{
  type_to_atomic_number_.assign(static_cast<std::size_t>(atom->ntypes + 1), 0);

  for (int type = 1; type <= atom->ntypes; ++type) {
    const int element_index = map[type];
    if (element_index < 0)
      error->all(FLERR, "Pair style llmlip does not support NULL atom type mappings yet");

    const int atomic_number = atomic_number_for_symbol(elements[element_index]);
    if (atomic_number == 0)
      error->all(FLERR, "Pair style llmlip does not recognize element symbol '{}'",
                 elements[element_index]);

    type_to_atomic_number_[static_cast<std::size_t>(type)] = atomic_number;
  }
}

void PairLlmlip::init_style()
{
  if (model_ == nullptr) error->all(FLERR, "Pair style llmlip model is not loaded");

  if (strcmp(update->unit_style, "metal") != 0)
    error->all(FLERR, "Pair style llmlip currently supports only units metal");

  if (force->newton_pair == 0)
    error->all(FLERR, Error::NOLASTLINE, "Pair style llmlip requires newton pair on");

  if (atom->molecular != Atom::ATOMIC) {
    const bool special_ok = force->special_lj[1] == 1.0 && force->special_lj[2] == 1.0 &&
        force->special_lj[3] == 1.0 && force->special_coul[1] == 1.0 &&
        force->special_coul[2] == 1.0 && force->special_coul[3] == 1.0;

    if (!special_ok)
      error->all(FLERR,
                 "Pair style llmlip is a many-body MLIP and requires no special_bonds exclusions");
  }

  validate_domain_capabilities();

  if (use_external_graph_path()) {
    ensure_comm_halo_cutoff();
    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
  }
}

void PairLlmlip::validate_domain_capabilities()
{
  const int pbc_count = domain->xperiodic + domain->yperiodic + domain->zperiodic;
  if (pbc_count == 0 && !caps_.supports_non_pbc)
    error->all(FLERR, "Pair style llmlip model does not support non-periodic structures");

  if (pbc_count > 0 && pbc_count < 3 && !caps_.supports_partial_pbc)
    error->all(FLERR, "Pair style llmlip model does not support partial PBC structures");

  if (pbc_count == 3 && !caps_.supports_full_pbc)
    error->all(FLERR, "Pair style llmlip model does not support full PBC structures");
}

void PairLlmlip::ensure_comm_halo_cutoff()
{
  if (comm->mode != Comm::SINGLE)
    error->all(FLERR,
               "Pair style llmlip currently supports only comm_modify mode single; use comm_modify "
               "mode single before pair_style llmlip");

  const double required_halo = caps_.mpi_halo_cutoff;
  const double depth_halo =
      caps_.model_cutoff * static_cast<double>(std::max<int64_t>(caps_.message_passing_depth, 1));
  if (required_halo + 1.0e-12 < depth_halo)
    error->all(FLERR,
               "Pair style llmlip model communication halo {} is shorter than model_cutoff * "
               "message_passing_depth {}",
               required_halo, depth_halo);

  if (comm->cutghostuser + 1.0e-12 < required_halo) {
    comm->cutghostuser = required_halo;
    if (comm->me == 0)
      error->warning(FLERR,
                     "Pair style llmlip extended comm_modify cutoff to {} for model_cutoff {} and "
                     "message_passing_depth {}",
                     required_halo, caps_.model_cutoff, caps_.message_passing_depth);
  }
}

double PairLlmlip::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    error->all(FLERR, Error::NOLASTLINE,
               "All pair coeffs are not set. Status\n" + Info::get_pair_coeff_status(lmp));

  const double model_cut = caps_.model_cutoff;
  const double model_cut_sq = model_cut * model_cut;
  cutsq[i][j] = cutsq[j][i] = model_cut_sq;
  cutghost[i][j] = cutghost[j][i] = model_cut;
  return model_cut;
}

void PairLlmlip::build_structure_view(bool include_ghosts)
{
  const int nlocal = atom->nlocal;
  const int nall = include_ghosts ? atom->nlocal + atom->nghost : atom->nlocal;

  atomic_numbers_.resize(static_cast<std::size_t>(nall));
  positions_.resize(static_cast<std::size_t>(3 * nall));

  for (int i = 0; i < nall; ++i) {
    const int type = atom->type[i];
    atomic_numbers_[static_cast<std::size_t>(i)] = type_to_atomic_number_[type];
    positions_[static_cast<std::size_t>(3 * i + 0)] = atom->x[i][0];
    positions_[static_cast<std::size_t>(3 * i + 1)] = atom->x[i][1];
    positions_[static_cast<std::size_t>(3 * i + 2)] = atom->x[i][2];
  }

  cell_.resize(9);
  cell_[0] = domain->xprd;
  cell_[1] = 0.0;
  cell_[2] = 0.0;
  cell_[3] = domain->triclinic ? domain->xy : 0.0;
  cell_[4] = domain->yprd;
  cell_[5] = 0.0;
  cell_[6] = domain->triclinic ? domain->xz : 0.0;
  cell_[7] = domain->triclinic ? domain->yz : 0.0;
  cell_[8] = domain->zprd;

  structure_view_.n_atoms_total = nall;
  structure_view_.n_atoms_owned = nlocal;
  structure_view_.atomic_numbers = atomic_numbers_.data();
  structure_view_.positions = positions_.data();
  structure_view_.cell = cell_.data();
  structure_view_.pbc[0] = domain->xperiodic;
  structure_view_.pbc[1] = domain->yperiodic;
  structure_view_.pbc[2] = domain->zperiodic;
}

void PairLlmlip::build_external_graph_from_lammps_list()
{
  const int nlocal = atom->nlocal;
  const int nall = atom->nlocal + atom->nghost;
  const double model_cut = caps_.model_cutoff;
  const double model_cut_sq = model_cut * model_cut;
  const int ncenters = list->ghost ? list->inum + list->gnum : list->inum;
  if (!list->ghost)
    error->all(FLERR, "Pair style llmlip requires a full ghost neighbor list");
  if (ncenters != nall)
    error->all(FLERR,
               "Pair style llmlip expected neighbor centers to cover nlocal+nghost atoms ({}), got "
               "{}",
               nall, ncenters);

  centers_.resize(static_cast<std::size_t>(ncenters));
  center_owned_.resize(static_cast<std::size_t>(ncenters));
  edge_offsets_.resize(static_cast<std::size_t>(ncenters + 1));

  std::size_t candidate_edges = 0;
  for (int ii = 0; ii < ncenters; ++ii) {
    const int i = list->ilist[ii];
    candidate_edges += static_cast<std::size_t>(list->numneigh[i]);
  }

  edge_dst_.clear();
  edge_vectors_.clear();
  if (edge_dst_.capacity() < candidate_edges) edge_dst_.reserve(candidate_edges);
  if (edge_vectors_.capacity() < 3 * candidate_edges) edge_vectors_.reserve(3 * candidate_edges);

  int64_t edge_count = 0;

  for (int ii = 0; ii < ncenters; ++ii) {
    const int i = list->ilist[ii];
    centers_[static_cast<std::size_t>(ii)] = i;
    center_owned_[static_cast<std::size_t>(ii)] = (i < nlocal) ? 1 : 0;
    edge_offsets_[static_cast<std::size_t>(ii)] = edge_count;

    const double xi = atom->x[i][0];
    const double yi = atom->x[i][1];
    const double zi = atom->x[i][2];

    int *jlist = list->firstneigh[i];
    const int jnum = list->numneigh[i];

    for (int jj = 0; jj < jnum; ++jj) {
      const int j = jlist[jj] & NEIGHMASK;

      const double dx = atom->x[j][0] - xi;
      const double dy = atom->x[j][1] - yi;
      const double dz = atom->x[j][2] - zi;
      const double rsq = dx * dx + dy * dy + dz * dz;

      if (rsq >= model_cut_sq) continue;

      edge_dst_.push_back(j);
      edge_vectors_.push_back(dx);
      edge_vectors_.push_back(dy);
      edge_vectors_.push_back(dz);
      ++edge_count;
    }
  }

  edge_offsets_[static_cast<std::size_t>(ncenters)] = edge_count;

  graph_view_.n_centers = ncenters;
  graph_view_.centers = centers_.data();
  graph_view_.center_owned = center_owned_.data();
  graph_view_.edge_offsets = edge_offsets_.data();
  graph_view_.dst = edge_dst_.data();
  graph_view_.edge_vector = edge_vectors_.data();
  graph_view_.model_cutoff = model_cut;
}

void PairLlmlip::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  if (eflag_atom) error->all(FLERR, "Pair style llmlip does not yet support per-atom energy");

  if (vflag_atom || cvflag_atom)
    error->all(FLERR, "Pair style llmlip does not yet support per-atom virial/stress");

  const int nall = atom->nlocal + atom->nghost;
  const auto virial_mode = resolved_virial_mode();

  const bool external_graph_path = use_external_graph_path();
  build_structure_view(external_graph_path);
  if (external_graph_path) build_external_graph_from_lammps_list();

  const int force_atom_count = external_graph_path ? nall : atom->nlocal;
  const std::size_t force_len = static_cast<std::size_t>(3 * force_atom_count);
  if (force_buffer_.size() != force_len) force_buffer_.resize(force_len);
  std::fill(force_buffer_.begin(), force_buffer_.end(), 0.0);

  llmlip_prediction_options_t options{};
  options.compute_energy = eflag_global ? 1 : 0;
  options.compute_forces = 1;
  options.compute_virial =
      (vflag_global && virial_mode == ResolvedVirialMode::Direct) ? 1 : 0;
  options.compute_stress = 0;
  options.dtype = capi_dtype();
  options.selected_head = selected_head_.empty() ? nullptr : selected_head_.c_str();
  options.has_total_charge = has_total_charge_ ? 1 : 0;
  options.total_charge = total_charge_;
  options.has_spin_multiplicity = has_spin_multiplicity_ ? 1 : 0;
  options.spin_multiplicity = spin_multiplicity_;

  llmlip_prediction_out_t out{};
  out.forces = force_buffer_.data();

  if (external_graph_path)
    fail_on_llmlip(llmlip_predict_external_graph(model_, &structure_view_, &graph_view_, &options, &out),
                 "predict with llmlip external graph");
  else
    fail_on_llmlip(llmlip_predict_structure(model_, &structure_view_, &options, &out),
                 "predict with llmlip structure");

  if (out.has_forces) {
    for (int i = 0; i < force_atom_count; ++i) {
      atom->f[i][0] += out.forces[3 * i + 0];
      atom->f[i][1] += out.forces[3 * i + 1];
      atom->f[i][2] += out.forces[3 * i + 2];
    }
  }

  if (eflag_global && out.has_energy) eng_vdwl += out.energy;

  if (vflag_global && virial_mode == ResolvedVirialMode::Direct && out.has_virial) {
    const double *v = out.virial;
    virial[0] += v[0];
    virial[1] += v[4];
    virial[2] += v[8];
    virial[3] += v[1];
    virial[4] += v[2];
    virial[5] += v[5];
  } else if (vflag_fdotr && virial_mode == ResolvedVirialMode::Fdotr) {
    virial_fdotr_compute();
  }
}

bool PairLlmlip::use_external_graph_path() const
{
  return comm->nprocs > 1 || rank1_mode_ == Rank1Mode::ExternalGraph;
}

PairLlmlip::ResolvedVirialMode PairLlmlip::resolved_virial_mode() const
{
  if (virial_mode_ == VirialMode::Off) return ResolvedVirialMode::Off;
  if (virial_mode_ == VirialMode::Fdotr) return ResolvedVirialMode::Fdotr;
  if (virial_mode_ == VirialMode::Direct) return ResolvedVirialMode::Direct;
  if (use_external_graph_path())
    return caps_.supports_external_graph_global_virial ? ResolvedVirialMode::Direct
                                                       : ResolvedVirialMode::Fdotr;
  return caps_.supports_global_virial ? ResolvedVirialMode::Direct : ResolvedVirialMode::Fdotr;
}

llmlip_dtype_t PairLlmlip::capi_dtype() const
{
  return dtype_ == DType::F64 ? LLMLIP_DTYPE_F64 : LLMLIP_DTYPE_F32;
}

int PairLlmlip::atomic_number_for_symbol(const char *symbol) const
{
  static constexpr const char *kSymbols[] = {
      "",   "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na",
      "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",
      "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
      "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
      "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
      "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
      "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
      "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am",
      "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh",
      "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

  for (int z = 1; z < static_cast<int>(sizeof(kSymbols) / sizeof(kSymbols[0])); ++z) {
    if (strcmp(symbol, kSymbols[z]) == 0) return z;
  }
  return 0;
}

void PairLlmlip::fail_on_llmlip(llmlip_status_t status, const char *context)
{
  if (status == LLMLIP_OK) return;
  error->all(FLERR, "Failed to {}: {}", context, llmlip_last_error());
}

double PairLlmlip::memory_usage()
{
  double bytes = Pair::memory_usage();
  bytes += static_cast<double>(type_to_atomic_number_.capacity()) * sizeof(int32_t);
  bytes += static_cast<double>(atomic_numbers_.capacity()) * sizeof(int32_t);
  bytes += static_cast<double>(positions_.capacity()) * sizeof(double);
  bytes += static_cast<double>(cell_.capacity()) * sizeof(double);
  bytes += static_cast<double>(centers_.capacity()) * sizeof(int64_t);
  bytes += static_cast<double>(center_owned_.capacity()) * sizeof(uint8_t);
  bytes += static_cast<double>(edge_offsets_.capacity()) * sizeof(int64_t);
  bytes += static_cast<double>(edge_dst_.capacity()) * sizeof(int64_t);
  bytes += static_cast<double>(edge_vectors_.capacity()) * sizeof(double);
  bytes += static_cast<double>(force_buffer_.capacity()) * sizeof(double);
  return bytes;
}
