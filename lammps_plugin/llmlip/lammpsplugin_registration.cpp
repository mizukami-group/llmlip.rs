// SPDX-License-Identifier: GPL-2.0-only
// Copyright (c) 2026 Mizukami Group and llmlip contributors
// See LICENSE-GPL-2.0 in this directory.

#include "lammpsplugin.h"
#include "pair_llmlip.h"
#include "version.h"

using namespace LAMMPS_NS;

static Pair *pair_llmlip_creator(LAMMPS *lmp)
{
  return new PairLlmlip(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "llmlip";
  plugin.info = "llmlip Rust-native MLIP pair style";
  plugin.author = "mizukami-group";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &pair_llmlip_creator;
  plugin.handle = handle;

  (*register_plugin)(&plugin, lmp);
}
