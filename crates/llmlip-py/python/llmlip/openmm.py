"""OpenMM integration for llmlip fixed-weight MLIP inference.

The lower-level helper in this module creates an ``openmm.PythonForce`` whose
callback calls :class:`llmlip.Model` directly.  It intentionally does not depend
on OpenMM-ML; the OpenMM-ML adapter lives in :mod:`llmlip.openmmml`.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from . import Model, PredictionError, UnsupportedCapabilityError, load_baked_model

EV_TO_KJMOL = 96.4853321233100184
ANGSTROM_PER_NM = 10.0
FORCE_SCALE = EV_TO_KJMOL * ANGSTROM_PER_NM

_DTYPE_ALIASES = {
    "float64": "float64",
    "f64": "float64",
    "fp64": "float64",
    "float32": "float32",
    "f32": "float32",
    "fp32": "float32",
}

_PRECISION_TO_DTYPE = {
    "double": "float64",
    "single": "float32",
}


def create_llmlip_force(
    topology,
    system,
    model_path: str | Path | None = None,
    *,
    model: Model | None = None,
    atoms: Iterable[int] | None = None,
    force_group: int = 0,
    dtype: str | None = None,
    precision: Literal["single", "double"] | None = None,
    head: str | None = None,
    total_charge: float | None = None,
    spin_multiplicity: float | None = None,
    mace_mh1_wigner_mode: Literal["true_float64", "upstream_artifact"] = "true_float64",
):
    """Create an OpenMM ``PythonForce`` backed by ``llmlip.Model.predict()``.

    The returned force is not added to ``system``.  llmlip currently assumes
    Angstrom/eV/eV-per-Angstrom model units; OpenMM uses nm and kJ/mol.
    """

    try:
        import openmm
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise ImportError("llmlip.openmm requires OpenMM; install llmlip[openmm].") from exc
    if not hasattr(openmm, "PythonForce"):
        raise ImportError("llmlip.openmm requires an OpenMM build with PythonForce support")

    if model_path is not None and model is not None:
        raise ValueError("create_llmlip_force accepts only one of model_path or model")
    if model_path is None and model is None:
        model = load_baked_model()

    model_obj = model if model is not None else Model(str(model_path))
    normalized_dtype = _normalize_dtype(dtype=dtype, precision=precision)

    all_atoms = list(topology.atoms())
    _validate_topology_system_particle_count(all_atoms, system)
    included_atoms, original_indices, indices = _included_atoms_and_indices(all_atoms, atoms)
    _validate_included_atoms(included_atoms, original_indices)
    species = [atom.element.symbol for atom in included_atoms]

    _validate_model_capabilities(
        model_obj,
        species=species,
        dtype=normalized_dtype,
        head=head,
        total_charge=total_charge,
        spin_multiplicity=spin_multiplicity,
    )

    periodic = (
        topology.getPeriodicBoxVectors() is not None
    ) or system.usesPeriodicBoundaryConditions()

    compute = partial(
        _compute_llmlip,
        model=model_obj,
        species=species,
        indices=indices,
        periodic=periodic,
        dtype=normalized_dtype,
        head=head,
        total_charge=total_charge,
        spin_multiplicity=spin_multiplicity,
        mace_mh1_wigner_mode=mace_mh1_wigner_mode,
    )
    force = openmm.PythonForce(compute)
    force.setForceGroup(force_group)
    force.setUsesPeriodicBoundaryConditions(periodic)
    return force


def _compute_llmlip(
    state,
    *,
    model: Model,
    species: list[str],
    indices: np.ndarray | None,
    periodic: bool,
    dtype: str,
    head: str | None,
    total_charge: float | None,
    spin_multiplicity: float | None,
    mace_mh1_wigner_mode: str,
):
    from openmm import unit

    positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    positions = np.asarray(positions, dtype=np.float64)
    num_atoms_total = positions.shape[0]

    if indices is None:
        positions_ml = positions
    else:
        positions_ml = np.asarray(positions[indices], dtype=np.float64)

    if periodic:
        cell = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
        cell = np.asarray(cell, dtype=np.float64)
        # OpenMM box vectors are treated as a full 3D PBC contract here. Partial
        # PBC semantics are intentionally unsupported in this MVP adapter.
        pbc = (True, True, True)
        positions_ml = _wrap_positions_into_periodic_cell(positions_ml, cell)
    else:
        cell = None
        pbc = (False, False, False)

    prediction = model.predict(
        species=species,
        positions=positions_ml,
        cell=cell,
        pbc=pbc,
        energy=True,
        forces=True,
        stress=False,
        virial=False,
        dtype=dtype,
        head=head,
        total_charge=total_charge,
        spin_multiplicity=spin_multiplicity,
        mace_mh1_wigner_mode=mace_mh1_wigner_mode,
    )

    if prediction.energy is None:
        raise PredictionError("llmlip did not return requested energy")
    if prediction.forces is None:
        raise PredictionError("llmlip did not return requested forces")

    energy = float(prediction.energy) * EV_TO_KJMOL
    forces_ml = np.asarray(prediction.forces, dtype=np.float64) * FORCE_SCALE
    if forces_ml.shape != (len(species), 3):
        raise PredictionError(
            f"llmlip returned forces with shape {forces_ml.shape}, expected {(len(species), 3)}"
        )

    if indices is None:
        forces = forces_ml
    else:
        forces = np.zeros((num_atoms_total, 3), dtype=np.float64)
        forces[indices] = forces_ml

    return energy, forces


def _normalize_dtype(*, dtype: str | None, precision: str | None) -> str:
    if precision is not None:
        try:
            precision_dtype = _PRECISION_TO_DTYPE[precision]
        except KeyError as exc:
            raise ValueError(
                f"unsupported llmlip OpenMM precision: {precision!r}; "
                "expected 'single' or 'double'"
            ) from exc
        if dtype is None:
            return precision_dtype
        normalized_dtype = _normalize_dtype_alias(dtype)
        if normalized_dtype != precision_dtype:
            raise ValueError(
                f"conflicting dtype={dtype!r} and precision={precision!r}; "
                f"precision maps to {precision_dtype!r}"
            )
        return normalized_dtype

    return _normalize_dtype_alias("float64" if dtype is None else dtype)


def _normalize_dtype_alias(dtype: str) -> str:
    try:
        return _DTYPE_ALIASES[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported llmlip OpenMM dtype: {dtype!r}") from exc


def _validate_topology_system_particle_count(all_atoms, system) -> None:
    topology_count = len(all_atoms)
    system_count = system.getNumParticles()
    if topology_count != system_count:
        raise ValueError(
            "OpenMM topology atom count must match system particle count for llmlip; "
            f"topology has {topology_count} atoms but system has {system_count} particles"
        )


def _included_atoms_and_indices(all_atoms, atoms: Iterable[int] | None):
    if atoms is None:
        return all_atoms, list(range(len(all_atoms))), None

    original_indices = np.asarray(list(atoms), dtype=np.int64)
    if original_indices.ndim != 1:
        raise ValueError("atoms must be a one-dimensional iterable of atom indices")
    if np.any(original_indices < 0) or np.any(original_indices >= len(all_atoms)):
        raise ValueError(
            f"atoms contains an index outside [0, {len(all_atoms)}): "
            f"{original_indices.tolist()}"
        )
    if len(np.unique(original_indices)) != original_indices.size:
        raise ValueError(f"atoms contains duplicate indices: {original_indices.tolist()}")
    included_atoms = [all_atoms[int(index)] for index in original_indices]
    return included_atoms, original_indices.tolist(), original_indices


def _validate_included_atoms(included_atoms, original_indices: list[int]) -> None:
    missing = [
        original_index
        for original_index, atom in zip(original_indices, included_atoms, strict=True)
        if atom.element is None
    ]
    if missing:
        raise ValueError(
            "all llmlip ML atoms must have elements defined; "
            f"missing OpenMM atom indices {missing}"
        )


def _validate_model_capabilities(
    model: Model,
    *,
    species: list[str],
    dtype: str,
    head: str | None,
    total_charge: float | None,
    spin_multiplicity: float | None,
) -> None:
    capabilities = model.capabilities()

    if not capabilities.supports_energy:
        raise UnsupportedCapabilityError("OpenMM requires an llmlip model with energy support")
    if not capabilities.supports_forces:
        raise UnsupportedCapabilityError("OpenMM requires an llmlip model with force support")

    if dtype == "float32" and not capabilities.supports_float32:
        raise UnsupportedCapabilityError(
            "requested OpenMM dtype float32, but model does not support it"
        )
    if dtype == "float64" and not capabilities.supports_float64:
        raise UnsupportedCapabilityError(
            "requested OpenMM dtype float64, but model does not support it"
        )

    supported_species = list(capabilities.supported_species)
    if supported_species:
        supported_species_set = set(supported_species)
        unknown_species = sorted(
            {symbol for symbol in species if symbol not in supported_species_set}
        )
        if unknown_species:
            raise UnsupportedCapabilityError(
                "OpenMM topology contains species unsupported by this llmlip model: "
                f"{unknown_species!r}; supported species: {supported_species!r}"
            )

    if head is not None:
        supported_heads = list(capabilities.supported_heads)
        if head not in supported_heads:
            raise UnsupportedCapabilityError(
                f"unsupported llmlip prediction head {head!r}; "
                f"supported heads: {supported_heads!r}"
            )

    requirements = capabilities.input_requirements
    if requirements.total_charge and total_charge is None:
        raise UnsupportedCapabilityError("this llmlip model requires total_charge")
    if requirements.spin_multiplicity and spin_multiplicity is None:
        raise UnsupportedCapabilityError("this llmlip model requires spin_multiplicity")


def _wrap_positions_into_periodic_cell(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    try:
        inverse_cell = np.linalg.inv(cell)
    except np.linalg.LinAlgError as exc:
        raise PredictionError(
            "periodic OpenMM box is singular; cannot wrap llmlip positions"
        ) from exc

    fractional = positions @ inverse_cell
    if not np.all(np.isfinite(fractional)):
        raise PredictionError(
            "periodic OpenMM positions produced non-finite fractional coordinates"
        )
    wrapped = (fractional - np.floor(fractional)) @ cell
    return np.asarray(wrapped, dtype=np.float64)
