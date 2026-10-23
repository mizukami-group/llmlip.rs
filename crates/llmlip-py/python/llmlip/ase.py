from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from . import (
    BackendUnavailableError,
    Model,
    PredictionError,
    UnsupportedCapabilityError,
    load_baked_model,
)
from ._backend import CliBackend, DirectBackend, PredictionBackend, PredictionRequest


class LlmlipCalculator(Calculator):
    """ASE calculator backed by llmlip direct PyO3 or CLI execution."""

    implemented_properties = ["energy", "forces", "stress", "energies", "descriptors"]
    default_parameters: dict = {}

    def __init__(
        self,
        model_path: str | Path | None = None,
        model: Model | None = None,
        *,
        backend: Literal["auto", "direct", "cli"] | None = None,
        binary: str | Path | None = None,
        dtype: Literal["float64", "float32", "f64", "f32", "model-native"] = "float64",
        head: str | None = None,
        total_charge: float | None = None,
        spin_multiplicity: float | None = None,
        mace_mh1_wigner_mode: Literal["true_float64", "upstream_artifact"] = "true_float64",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if model_path is None and model is None:
            model = load_baked_model()
        if model_path is not None and model is not None:
            raise ValueError("LlmlipCalculator accepts only one of model_path or model")
        self.model_path = None if model_path is None else str(model_path)
        self.model = model
        self.dtype = dtype
        self.head = head
        self.total_charge = total_charge
        self.spin_multiplicity = spin_multiplicity
        self.mace_mh1_wigner_mode = mace_mh1_wigner_mode
        requested_backend = backend or os.environ.get("LLMLIP_ASE_BACKEND", "auto")
        self.backend_name = requested_backend
        self.backend = self._make_backend(requested_backend, binary)

    def _make_backend(
        self,
        backend: str,
        binary: str | Path | None,
    ) -> PredictionBackend:
        if backend == "direct":
            return DirectBackend(self.model_path, model=self.model)
        if backend == "cli":
            if self.model_path is None:
                raise BackendUnavailableError("CLI backend requires model_path")
            return CliBackend(self.model_path, binary=binary, use_json=True)
        if backend == "auto":
            direct_error: Exception | None = None
            try:
                return DirectBackend(self.model_path, model=self.model)
            except Exception as exc:
                direct_error = exc
            cli_error: Exception | None = None
            if self.model_path is not None:
                try:
                    return CliBackend(self.model_path, binary=binary, use_json=True)
                except Exception as exc:
                    cli_error = exc
            raise BackendUnavailableError(
                "No llmlip backend available. "
                f"Direct backend failed with: {direct_error!r}; "
                f"CLI backend failed with: {cli_error!r}"
            )
        raise ValueError(f"unsupported backend: {backend}")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if properties is None:
            properties = self.implemented_properties

        need_energy = "energy" in properties
        need_forces = "forces" in properties
        need_stress = "stress" in properties
        need_atomic_energies = "energies" in properties
        need_descriptors = "descriptors" in properties
        if need_stress and not bool(self.atoms.pbc.all()):
            raise UnsupportedCapabilityError(
                "Stress calculation requires full periodic boundary conditions "
                "(pbc=[True, True, True])."
            )

        request = PredictionRequest(
            species=self.atoms.get_chemical_symbols(),
            positions=np.asarray(self.atoms.positions, dtype=np.float64),
            cell=np.asarray(self.atoms.cell.array, dtype=np.float64)
            if self.atoms.pbc.any()
            else None,
            pbc=tuple(bool(value) for value in self.atoms.pbc),
            energy=need_energy or need_stress,
            forces=need_forces,
            stress=need_stress,
            virial=False,
            dtype=self.dtype,
            head=self.head,
            total_charge=self.total_charge,
            spin_multiplicity=self.spin_multiplicity,
            mace_mh1_wigner_mode=self.mace_mh1_wigner_mode,
        )
        prediction = None
        if need_energy or need_forces or need_stress:
            prediction = self.backend.predict(request)

        if need_energy:
            if prediction is None:
                raise PredictionError("model did not return requested energy")
            if prediction.energy is None:
                raise PredictionError("model did not return requested energy")
            self.results["energy"] = prediction.energy
        if need_forces:
            if prediction is None:
                raise PredictionError("model did not return requested forces")
            if prediction.forces is None:
                raise PredictionError("model did not return requested forces")
            self.results["forces"] = prediction.forces
        if need_stress:
            if prediction is None:
                raise PredictionError("model did not return requested stress")
            stress = prediction.stress
            if stress is None:
                raise PredictionError("model did not return requested stress")
            self.results["stress"] = _ase_voigt_stress_from_llmlip(stress)
        if need_atomic_energies and need_descriptors:
            descriptors, atomic_energies = self.backend.get_extra_outputs(request)
            self.results["energies"] = atomic_energies
            self.results["descriptors"] = descriptors
        elif need_atomic_energies:
            self.results["energies"] = self.backend.get_atomic_energies(request)
        elif need_descriptors:
            self.results["descriptors"] = self.backend.get_descriptors(request)

    def close(self) -> None:
        self.backend.close()


def _ase_voigt_stress_from_llmlip(stress: np.ndarray) -> np.ndarray:
    # ASE cell filters use virial = -volume * stress. llmlip stress already
    # satisfies virial = -stress * volume, so no sign flip is applied here.
    return np.array(
        [
            stress[0, 0],
            stress[1, 1],
            stress[2, 2],
            stress[1, 2],
            stress[0, 2],
            stress[0, 1],
        ],
        dtype=np.float64,
    )
