from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

import llmlip
import llmlip._llmlip  # noqa: F401
from llmlip._backend import CliBackend, DirectBackend, PredictionRequest


REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_ROOT = Path(os.environ.get("LLMLIP_TEST_REFERENCE_ROOT", REPO_ROOT))
MODEL_PATH = REPO_ROOT / "examples" / "model.e3rs"
PBC_MODEL_PATH = (
    REFERENCE_ROOT
    / "fixtures"
    / "nequip"
    / "v0_17_1_604ce73b"
    / "models"
    / "nequip_lite_lmax1_layers1.e3rs"
)
MACE_MODEL_PATH = (
    REFERENCE_ROOT
    / "fixtures"
    / "mace"
    / "v0_3_16_4d2da09"
    / "models"
    / "mace_lite_layers2.e3rs"
)
_BUILT_LLMLIP_BINARY: Path | None = None
SPECIES = ["O", "H", "H"]
POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.9572, 0.0, 0.0],
        [-0.239987, 0.927297, 0.0],
    ],
    dtype=np.float64,
)
PBC_SPECIES = ["H", "H", "O"]
PBC_POSITIONS = np.array(
    [
        [0.2, 0.3, 0.4],
        [8.05, 0.35, 0.45],
        [0.25, 8.2, 0.5],
    ],
    dtype=np.float64,
)
PBC_CELL = np.array(
    [
        [8.5, 0.0, 0.0],
        [0.0, 8.7, 0.0],
        [0.0, 0.0, 8.9],
    ],
    dtype=np.float64,
)
MACE_SPECIES = ["C", "H", "H"]
MACE_POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [0.0, 1.2, 0.0],
    ],
    dtype=np.float64,
)


def _llmlip_binary() -> Path:
    global _BUILT_LLMLIP_BINARY
    if _BUILT_LLMLIP_BINARY is not None:
        return _BUILT_LLMLIP_BINARY
    suffix = ".exe" if os.name == "nt" else ""
    binary = REPO_ROOT / "target" / "debug" / f"llmlip{suffix}"
    subprocess.run(["cargo", "build", "-p", "llmlip-cli"], cwd=REPO_ROOT, check=True)
    _BUILT_LLMLIP_BINARY = binary
    return binary


def _request() -> PredictionRequest:
    return PredictionRequest(
        species=SPECIES,
        positions=POSITIONS,
        cell=None,
        pbc=(False, False, False),
        energy=True,
        forces=True,
    )


def test_direct_model_predicts_energy_forces_and_capabilities():
    model = llmlip.Model(str(MODEL_PATH))
    assert model.architecture() == "synthetic-local-v1"

    requirements = model.input_requirements()
    assert requirements.total_charge is False
    assert requirements.spin_multiplicity is False

    capabilities = model.capabilities()
    assert capabilities.architecture == "synthetic-local-v1"
    assert capabilities.supports_energy is True
    assert capabilities.supports_forces is True
    assert capabilities.supports_float64 is True
    assert capabilities.supports_float32 is False
    assert capabilities.supported_heads == []

    prediction = model.predict(
        species=SPECIES,
        positions=POSITIONS,
        cell=None,
        pbc=(False, False, False),
        energy=True,
        forces=True,
    )
    assert isinstance(prediction.energy, float)
    assert math.isfinite(prediction.energy)
    assert prediction.forces.shape == (3, 3)
    assert prediction.forces.dtype == np.float64


def test_pyobjects_have_informative_repr():
    model = llmlip.Model(str(MODEL_PATH))
    assert "Model(" in repr(model)
    assert "synthetic-local-v1" in repr(model)
    assert str(MODEL_PATH) in repr(model)

    requirements = model.input_requirements()
    assert (
        repr(requirements)
        == "PredictionInputRequirements(total_charge=False, spin_multiplicity=False)"
    )

    capabilities = model.capabilities()
    assert "ModelCapabilities(" in repr(capabilities)
    assert 'architecture="synthetic-local-v1"' in repr(capabilities)
    assert "supports_forces=True" in repr(capabilities)

    prediction = model.predict(species=SPECIES, positions=POSITIONS, energy=True, forces=True)
    assert "Prediction(" in repr(prediction)
    assert "energy=" in repr(prediction)
    assert "forces=array(shape=(3, 3), dtype=float64)" in repr(prediction)
    assert "stress=None" in repr(prediction)


def test_direct_model_predicts_stress_and_virial_for_full_pbc():
    model = llmlip.Model(str(PBC_MODEL_PATH))
    prediction = model.predict(
        species=PBC_SPECIES,
        positions=PBC_POSITIONS,
        cell=PBC_CELL,
        pbc=(True, True, True),
        energy=True,
        forces=True,
        stress=True,
        virial=True,
    )

    assert isinstance(prediction.energy, float)
    assert prediction.forces.shape == (3, 3)
    assert prediction.stress.shape == (3, 3)
    assert prediction.virial.shape == (3, 3)
    assert np.all(np.isfinite(prediction.stress))
    assert np.all(np.isfinite(prediction.virial))


def test_direct_model_exposes_mace_descriptors_and_atomic_energies():
    model = llmlip.Model(str(MACE_MODEL_PATH))

    descriptors = model.get_descriptors(
        species=MACE_SPECIES,
        positions=MACE_POSITIONS,
        invariants_only=True,
        num_layers=-1,
    )
    assert descriptors.shape == (3, 2)
    assert descriptors.dtype == np.float64
    assert descriptors.flags.c_contiguous
    assert np.all(np.isfinite(descriptors))

    first_layer = model.get_descriptors(
        species=MACE_SPECIES,
        positions=MACE_POSITIONS,
        invariants_only=True,
        num_layers=1,
        dtype="float64",
    )
    assert first_layer.shape == (3, 1)

    atomic_energies = model.get_atomic_energies(
        species=MACE_SPECIES,
        positions=MACE_POSITIONS,
    )
    alias_energies = model.get_energies(
        species=MACE_SPECIES,
        positions=MACE_POSITIONS,
    )
    assert atomic_energies.shape == (3,)
    assert atomic_energies.dtype == np.float64
    assert atomic_energies.flags.c_contiguous
    np.testing.assert_allclose(alias_energies, atomic_energies, atol=0.0)

    prediction = model.predict(
        species=MACE_SPECIES,
        positions=MACE_POSITIONS,
        energy=True,
        forces=False,
    )
    assert atomic_energies.sum() == pytest.approx(prediction.energy, abs=1.0e-12)

    # Combined one-pass getter returns (descriptors, atomic_energies) bit-for-bit
    # equal to the two separate calls, without the duplicate forward pass.
    combined_descriptors, combined_atomic_energies = model.get_extra_outputs(
        species=MACE_SPECIES,
        positions=MACE_POSITIONS,
        invariants_only=True,
        num_layers=-1,
    )
    assert combined_descriptors.shape == (3, 2)
    assert combined_descriptors.dtype == np.float64
    assert combined_descriptors.flags.c_contiguous
    assert combined_atomic_energies.shape == (3,)
    np.testing.assert_array_equal(combined_descriptors, descriptors)
    np.testing.assert_array_equal(combined_atomic_energies, atomic_energies)


def test_direct_prediction_validates_shapes_and_lengths():
    model = llmlip.Model(str(MODEL_PATH))
    with pytest.raises(llmlip.InvalidStructureError, match="positions"):
        model.predict(species=SPECIES, positions=np.zeros((3, 2), dtype=np.float64))
    with pytest.raises(llmlip.InvalidStructureError, match="species length"):
        model.predict(species=SPECIES[:2], positions=POSITIONS)
    with pytest.raises(llmlip.InvalidStructureError, match="cell"):
        model.predict(
            species=SPECIES,
            positions=POSITIONS,
            cell=np.zeros((2, 3), dtype=np.float64),
            pbc=(True, True, True),
        )


def test_direct_prediction_maps_unsupported_requests_to_explicit_errors():
    model = llmlip.Model(str(MODEL_PATH))
    with pytest.raises(llmlip.UnsupportedCapabilityError, match="compute dtype"):
        model.predict(species=SPECIES, positions=POSITIONS, dtype="float32")
    with pytest.raises(llmlip.UnsupportedCapabilityError, match="unsupported prediction head"):
        model.predict(species=SPECIES, positions=POSITIONS, head="omat_pbe")


def test_direct_and_cli_backends_match_energy_and_forces():
    request = _request()
    direct = DirectBackend(MODEL_PATH)
    with pytest.warns(RuntimeWarning, match="one llmlip process per prediction"):
        cli = CliBackend(MODEL_PATH, binary=_llmlip_binary())

    direct_prediction = direct.predict(request)
    cli_prediction = cli.predict(request)

    assert cli.protocol == "predict-json"
    assert direct_prediction.energy == pytest.approx(cli_prediction.energy, abs=1.0e-12)
    np.testing.assert_allclose(direct_prediction.forces, cli_prediction.forces, atol=1.0e-12)


def test_direct_backend_get_extra_outputs_matches_separate_mace_outputs():
    request = PredictionRequest(
        species=MACE_SPECIES,
        positions=MACE_POSITIONS,
        cell=None,
        pbc=(False, False, False),
        energy=False,
        forces=False,
    )
    direct = DirectBackend(MACE_MODEL_PATH)

    descriptors = direct.get_descriptors(request)
    atomic_energies = direct.get_atomic_energies(request)
    combined_descriptors, combined_atomic_energies = direct.get_extra_outputs(request)

    np.testing.assert_array_equal(combined_descriptors, descriptors)
    np.testing.assert_array_equal(combined_atomic_energies, atomic_energies)


def test_cli_backend_reports_missing_binary():
    with pytest.raises(llmlip.BackendUnavailableError, match="explicit llmlip binary"):
        CliBackend(MODEL_PATH, binary=REPO_ROOT / "target" / "debug" / "missing-llmlip")


def test_ase_direct_and_cli_calculators_match():
    ase = pytest.importorskip("ase")
    from llmlip.ase import LlmlipCalculator

    direct_atoms = ase.Atoms(SPECIES, positions=POSITIONS)
    cli_atoms = ase.Atoms(SPECIES, positions=POSITIONS)
    direct_atoms.calc = LlmlipCalculator(MODEL_PATH, backend="direct")
    with pytest.warns(RuntimeWarning, match="one llmlip process per prediction"):
        cli_atoms.calc = LlmlipCalculator(MODEL_PATH, backend="cli", binary=_llmlip_binary())

    assert direct_atoms.get_potential_energy() == pytest.approx(
        cli_atoms.get_potential_energy(),
        abs=1.0e-12,
    )
    np.testing.assert_allclose(direct_atoms.get_forces(), cli_atoms.get_forces(), atol=1.0e-12)


def test_ase_auto_prefers_direct_and_can_fallback_to_cli(monkeypatch):
    ase = pytest.importorskip("ase")
    import llmlip.ase as llmlip_ase
    from llmlip.ase import LlmlipCalculator

    atoms = ase.Atoms(SPECIES, positions=POSITIONS)
    calc = LlmlipCalculator(MODEL_PATH, backend="auto", binary=_llmlip_binary())
    assert calc.backend.protocol == "direct"
    atoms.calc = calc
    assert math.isfinite(atoms.get_potential_energy())

    class BrokenDirectBackend:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("direct disabled for test")

    monkeypatch.setattr(llmlip_ase, "DirectBackend", BrokenDirectBackend)
    with pytest.warns(RuntimeWarning, match="one llmlip process per prediction"):
        fallback_calc = LlmlipCalculator(MODEL_PATH, backend="auto", binary=_llmlip_binary())
    assert fallback_calc.backend.protocol in {"predict-json", "legacy-extxyz"}


def test_ase_baked_default_constructs_direct_calculator(monkeypatch):
    ase = pytest.importorskip("ase")
    import llmlip.ase as llmlip_ase
    from llmlip.ase import LlmlipCalculator

    calls = []

    class FakePrediction:
        energy = 1.25
        forces = np.zeros((3, 3), dtype=np.float64)
        stress = None
        virial = None

    class FakeBakedModel:
        def predict(self, **kwargs):
            calls.append(kwargs)
            return FakePrediction()

    baked_model = FakeBakedModel()
    monkeypatch.setattr(llmlip_ase, "load_baked_model", lambda: baked_model)

    atoms = ase.Atoms(SPECIES, positions=POSITIONS)
    atoms.calc = LlmlipCalculator()

    assert atoms.calc.model_path is None
    assert atoms.calc.model is baked_model
    assert atoms.calc.backend.protocol == "direct"
    assert atoms.get_potential_energy() == pytest.approx(1.25)
    np.testing.assert_allclose(atoms.get_forces(), np.zeros((3, 3)), atol=0.0)
    assert calls[0]["energy"] is True
    assert calls[0]["forces"] is False


def test_ase_non_full_pbc_stress_fails_before_prediction():
    ase = pytest.importorskip("ase")
    from llmlip.ase import LlmlipCalculator

    atoms = ase.Atoms(
        SPECIES,
        positions=POSITIONS,
        cell=np.eye(3) * 10.0,
        pbc=[True, False, True],
    )
    atoms.calc = LlmlipCalculator(MODEL_PATH, backend="direct")
    with pytest.raises(llmlip.UnsupportedCapabilityError, match="full periodic"):
        atoms.get_stress()


def test_ase_direct_stress_matches_ase_virial_sign_and_voigt_order():
    ase = pytest.importorskip("ase")
    from llmlip.ase import LlmlipCalculator

    model = llmlip.Model(str(PBC_MODEL_PATH))
    direct_prediction = model.predict(
        species=PBC_SPECIES,
        positions=PBC_POSITIONS,
        cell=PBC_CELL,
        pbc=(True, True, True),
        stress=True,
        virial=True,
        forces=False,
    )
    volume = abs(np.linalg.det(PBC_CELL))
    expected_stress = np.array(
        [
            direct_prediction.stress[0, 0],
            direct_prediction.stress[1, 1],
            direct_prediction.stress[2, 2],
            direct_prediction.stress[1, 2],
            direct_prediction.stress[0, 2],
            direct_prediction.stress[0, 1],
        ],
        dtype=np.float64,
    )
    expected_stress_from_virial = np.array(
        [
            -direct_prediction.virial[0, 0] / volume,
            -direct_prediction.virial[1, 1] / volume,
            -direct_prediction.virial[2, 2] / volume,
            -direct_prediction.virial[1, 2] / volume,
            -direct_prediction.virial[0, 2] / volume,
            -direct_prediction.virial[0, 1] / volume,
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(expected_stress, expected_stress_from_virial, atol=1.0e-12)

    atoms = ase.Atoms(PBC_SPECIES, positions=PBC_POSITIONS, cell=PBC_CELL, pbc=True)
    atoms.calc = LlmlipCalculator(model=model, backend="direct")
    np.testing.assert_allclose(atoms.get_stress(), expected_stress, atol=1.0e-12)


def test_ase_direct_exposes_mace_descriptors_and_atomic_energies():
    ase = pytest.importorskip("ase")
    from llmlip.ase import LlmlipCalculator

    atoms = ase.Atoms(MACE_SPECIES, positions=MACE_POSITIONS)
    atoms.calc = LlmlipCalculator(MACE_MODEL_PATH, backend="direct")

    descriptors = atoms.calc.get_property("descriptors", atoms)
    atomic_energies = atoms.calc.get_property("energies", atoms)
    assert descriptors.shape == (3, 2)
    assert descriptors.dtype == np.float64
    assert atomic_energies.shape == (3,)
    assert atomic_energies.dtype == np.float64
    assert np.all(np.isfinite(descriptors))
    assert np.all(np.isfinite(atomic_energies))
    assert atomic_energies.sum() == pytest.approx(atoms.get_potential_energy(), abs=1.0e-12)


def test_ase_direct_uses_one_pass_for_mace_extra_outputs(monkeypatch):
    ase = pytest.importorskip("ase")
    from ase.calculators.calculator import all_changes

    import llmlip.ase as llmlip_ase
    from llmlip.ase import LlmlipCalculator

    class RecordingBackend:
        protocol = "direct"

        def __init__(self):
            self.extra_output_calls = 0
            self.descriptor_calls = 0
            self.atomic_energy_calls = 0

        def predict(self, _request):
            raise AssertionError("extra-only ASE request should not call predict")

        def get_extra_outputs(self, request, *, invariants_only=True, num_layers=-1):
            assert list(request.species) == MACE_SPECIES
            assert invariants_only is True
            assert num_layers == -1
            self.extra_output_calls += 1
            return (
                np.ones((len(MACE_SPECIES), 2), dtype=np.float64),
                np.arange(len(MACE_SPECIES), dtype=np.float64),
            )

        def get_descriptors(self, _request, *, invariants_only=True, num_layers=-1):
            self.descriptor_calls += 1
            raise AssertionError("combined ASE extra-output request should not split descriptors")

        def get_atomic_energies(self, _request):
            self.atomic_energy_calls += 1
            raise AssertionError("combined ASE extra-output request should not split energies")

        def close(self):
            pass

    backend = RecordingBackend()
    monkeypatch.setattr(llmlip_ase, "DirectBackend", lambda *_args, **_kwargs: backend)

    atoms = ase.Atoms(MACE_SPECIES, positions=MACE_POSITIONS)
    calc = LlmlipCalculator(MACE_MODEL_PATH, backend="direct")
    calc.calculate(atoms, properties=["energies", "descriptors"], system_changes=all_changes)

    assert backend.extra_output_calls == 1
    assert backend.descriptor_calls == 0
    assert backend.atomic_energy_calls == 0
    np.testing.assert_array_equal(calc.results["descriptors"], np.ones((3, 2)))
    np.testing.assert_array_equal(calc.results["energies"], np.arange(3, dtype=np.float64))


def test_ase_cli_rejects_mace_extra_outputs():
    ase = pytest.importorskip("ase")
    from llmlip.ase import LlmlipCalculator

    atoms = ase.Atoms(MACE_SPECIES, positions=MACE_POSITIONS)
    with pytest.warns(RuntimeWarning, match="one llmlip process per prediction"):
        atoms.calc = LlmlipCalculator(
            MACE_MODEL_PATH,
            backend="cli",
            binary=_llmlip_binary(),
        )
    with pytest.raises(llmlip.UnsupportedCapabilityError, match="backend='direct'"):
        atoms.calc.get_property("descriptors", atoms)
