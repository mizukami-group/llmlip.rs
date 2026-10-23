"""OpenMM-ML adapter for llmlip fixed-weight MLIP inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def _import_openmmml():
    try:
        from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise ImportError("llmlip.openmmml requires OpenMM-ML; install llmlip[openmm].") from exc
    return MLPotentialImpl, MLPotentialImplFactory


class LlmlipPotentialImplFactory:
    """Factory loaded through the ``openmmml.potentials`` entry point."""

    def createImpl(
        self,
        name: str,
        modelPath: str | None = None,
        model_path: str | None = None,
        **args,
    ) -> Any:
        _import_openmmml()
        if args:
            raise ValueError(
                "unsupported llmlip OpenMM-ML keyword argument(s): "
                f"{sorted(args)!r}"
            )
        if modelPath is not None and model_path is not None:
            raise ValueError("MLPotential('llmlip') accepts only one of modelPath or model_path")
        return LlmlipPotentialImpl(name=name, model_path=modelPath or model_path)


class LlmlipPotentialImpl:
    """OpenMM-ML implementation for the ``llmlip`` potential name."""

    def __init__(self, name: str, model_path: str | Path | None):
        if name != "llmlip":
            raise ValueError(f"unsupported llmlip OpenMM-ML potential name: {name!r}")
        self.model_path = None if model_path is None else str(model_path)

    def getSupportedEmbeddings(self) -> list[str]:
        """Use OpenMM-ML's generic embedding implementations."""
        return []

    def getMLLongRange(self) -> bool | None:
        """The range depends on the selected model, so report it as unknown."""
        return None

    def addForces(
        self,
        topology,
        system,
        atoms: Iterable[int] | None,
        forceGroup: int,
        dtype: str | None = None,
        precision: str | None = None,
        head: str | None = None,
        total_charge: float | None = None,
        spin_multiplicity: float | None = None,
        mace_mh1_wigner_mode: str = "true_float64",
        **args,
    ) -> None:
        from .openmm import create_llmlip_force

        _populate_empty_openmmml_temp_system(topology, system)

        if total_charge is None and "charge" in args:
            total_charge = args.pop("charge")
        if spin_multiplicity is None:
            if "multiplicity" in args:
                spin_multiplicity = args.pop("multiplicity")
            elif "spin" in args:
                spin_multiplicity = args.pop("spin")
        if args:
            raise ValueError(
                "unsupported llmlip OpenMM-ML addForces keyword argument(s): "
                f"{sorted(args)!r}"
            )

        force = create_llmlip_force(
            topology,
            system,
            self.model_path,
            atoms=atoms,
            force_group=forceGroup,
            dtype=dtype,
            precision=precision,
            head=head,
            total_charge=total_charge,
            spin_multiplicity=spin_multiplicity,
            mace_mh1_wigner_mode=mace_mh1_wigner_mode,
        )
        system.addForce(force)


def _populate_empty_openmmml_temp_system(topology, system) -> None:
    if system.getNumParticles() != 0:
        return
    if topology.getPeriodicBoxVectors() is not None:
        system.setDefaultPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
    for atom in topology.atoms():
        system.addParticle(0 if atom.element is None else atom.element.mass)
