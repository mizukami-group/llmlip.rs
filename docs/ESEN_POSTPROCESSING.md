# eSEN postprocessing metadata

The local checkpoint exporter now writes a required `postprocessing` object
inside `esen_oam_state.json` with `energy_mean`, `energy_rmsd` and
`element_references`. The last field contains 119 finite values indexed by
atomic number. The supported pinned contract requires zero mean and a finite,
strictly positive RMSD. No model-specific numerical defaults are built into Rust.

Energy is raw network energy times RMSD plus the sum of elemental references;
the force derivative is scaled by the same RMSD. Element references are constant
with respect to positions. Unknown fields, missing data, unsupported mean,
invalid scale and invalid reference arrays fail during loading.

Old exports must be regenerated with `tools/reference/export_esen_oam_state.py`
using the user's authorized local checkpoint. No automatic download or license
acceptance is performed. The generated postprocessing values remain model data
and must not be committed or included in this source release. Do not copy values
from a previous source revision into a public migration example.
