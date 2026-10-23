# Source-only distribution policy

The current publication scope is the source repository and its source-tree
archive, including build instructions and the small synthetic example model.
It excludes prebuilt executables, Python wheels and Python sdists, native
plugins, containers, upstream model weights and converted foundation models.
Local test builds are not approved release assets.

Original project-authored code defaults to MIT OR Apache-2.0. The complete ORB
module is distributed under Apache-2.0; the aggregate crate is not MIT-only.
The three LAMMPS
C++ interface files identified in [the plugin README](../lammps_plugin/llmlip/README.md#license-and-distribution)
are GPL-2.0-only. Preserve file-specific notices and third-party terms.
The source archive must retain the root license files, third-party notices,
and the plugin's GPL text. Do not publish internal development history or logs.

Before any future binary or Python package distribution:

1. Inventory the exact components, features and native libraries included.
2. Include the project's license texts and each bundled component's required
   copyright, permission and notice texts; license metadata alone is insufficient.
3. Include the pinned core's third-party notices and review numeric data provenance.
4. For LAMMPS combined distributions, satisfy GPLv2 and provide corresponding source.
5. Review Intel MKL, ArmPL, OpenBLAS, libxsmm and platform-library terms for the
   actual linking configuration. Cargo metadata does not cover those libraries.
6. Inspect the final archive contents. The existing local wheel/sdist from
   preparation lacked license files and must not be distributed.
7. Review any model artifact separately; this source policy distributes no
   foundation weights, including PET models.

This policy limits the project's release scope. It does not remove rights that
recipients have under the applicable licenses or declare all downstream binary
redistribution forbidden.

## ORB license scope clarification

The ORB implementation module is Apache-2.0; see [scope and notices](../docs/ORB_LICENSE_SCOPE.md).
Original MIT OR Apache-2.0 grants do not override it. The current combined
LAMMPS plugin is not cleared for redistribution under GPL-2.0-only.
Only the separately licensed source components are in this release scope.
