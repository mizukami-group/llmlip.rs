# ORB implementation licensing

The complete `crates/llmlip/src/orb_v3_omol.rs` module is distributed under
Apache-2.0 as a conservative boundary. This resolves the release-policy
ambiguity without asserting that every line was copied or independently authored.
Original project code elsewhere retains its MIT OR Apache-2.0 grant; that
choice cannot remove the Apache conditions on this module. Cargo and Python
metadata describe the aggregate obligations, including BSD and zlib numerical
material. See [NOTICE](../NOTICE) for references and modification notices.

The current library includes ORB. Therefore this release does not clear a
GPL-2.0-only LAMMPS/llmlip combined binary for redistribution. Selecting the
MIT option on original code or using dynamic loading does not solve this.
A separately audited build excluding Apache-only material, or an appropriate
additional upstream grant, is required before such distribution. This source
release contains separately licensed components; it includes no combined binary.
