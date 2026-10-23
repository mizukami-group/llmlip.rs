from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import llmlip
from llmlip.openmm import (
    ANGSTROM_PER_NM,
    EV_TO_KJMOL,
    FORCE_SCALE,
    _compute_llmlip,
    _normalize_dtype,
    _validate_model_capabilities,
    _wrap_positions_into_periodic_cell,
    create_llmlip_force,
)


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
ORB_METADATA_MODEL_PATH = (
    REFERENCE_ROOT
    / "fixtures"
    / "orb"
    / "v0_6_2_a5f1a38"
    / "models"
    / "orb_v3_conservative_omol_metadata.e3rs"
)
SPECIES = ["O", "H", "H"]
POSITIONS_ANG = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.9572, 0.0, 0.0],
        [-0.239987, 0.927297, 0.0],
    ],
    dtype=np.float64,
)
PBC_SPECIES = ["H", "H", "O"]
PBC_POSITIONS_ANG = np.array(
    [
        [0.2, 0.3, 0.4],
        [8.05, 0.35, 0.45],
        [0.25, 8.2, 0.5],
    ],
    dtype=np.float64,
)
PBC_CELL_ANG = np.array(
    [
        [8.5, 0.0, 0.0],
        [0.0, 8.7, 0.0],
        [0.0, 0.0, 8.9],
    ],
    dtype=np.float64,
)
TRICLINIC_CELL_ANG = np.array(
    [
        [8.5, 0.2, 0.1],
        [0.3, 8.7, 0.4],
        [0.1, 0.5, 8.9],
    ],
    dtype=np.float64,
)


def _openmm_modules():
    openmm = pytest.importorskip("openmm")
    app = pytest.importorskip("openmm.app")
    unit = pytest.importorskip("openmm.unit")
    return openmm, app, unit


def _require_pythonforce():
    openmm, _app, _unit = _openmm_modules()
    if not hasattr(openmm, "PythonForce"):
        pytest.skip("OpenMM build lacks PythonForce")
    return openmm


def _require_native_llmlip():
    if getattr(llmlip, "_NATIVE_IMPORT_ERROR", None) is not None:
        pytest.skip("llmlip native module is unavailable")


def _box_vectors(cell_ang: np.ndarray):
    openmm, _app, unit = _openmm_modules()
    return tuple(
        openmm.Vec3(*np.asarray(row / 10.0, dtype=np.float64)) * unit.nanometer
        for row in cell_ang
    )


def _minimal_topology(species: list[str], cell_ang: np.ndarray | None = None):
    _openmm, app, _unit = _openmm_modules()
    top = app.Topology()
    chain = top.addChain()
    residue = top.addResidue("MOL", chain)
    for index, symbol in enumerate(species):
        element = app.Element.getBySymbol(symbol) if symbol != "X" else None
        top.addAtom(f"{symbol}{index}", element, residue)
    if cell_ang is not None:
        top.setPeriodicBoxVectors(_box_vectors(cell_ang))
    return top


def _minimal_system(natoms: int, cell_ang: np.ndarray | None = None):
    openmm, _app, _unit = _openmm_modules()
    system = openmm.System()
    for _ in range(natoms):
        system.addParticle(1.0)
    if cell_ang is not None:
        system.setDefaultPeriodicBoxVectors(*_box_vectors(cell_ang))
    return system


def _context_energy_forces(system, positions_ang, cell_ang: np.ndarray | None = None):
    openmm, _app, unit = _openmm_modules()
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), platform)
    if cell_ang is not None:
        context.setPeriodicBoxVectors(*_box_vectors(cell_ang))
    context.setPositions((np.asarray(positions_ang, dtype=np.float64) / 10.0) * unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer
    )
    del context
    return energy, np.asarray(forces, dtype=np.float64)


def test_openmm_module_imports_without_openmm_dependency():
    import llmlip.openmm

    assert llmlip.openmm.EV_TO_KJMOL == pytest.approx(EV_TO_KJMOL)


def test_openmmml_module_imports_without_openmmml_dependency():
    import llmlip.openmmml

    assert llmlip.openmmml.LlmlipPotentialImplFactory.__name__ == "LlmlipPotentialImplFactory"


def test_dtype_and_precision_normalization():
    assert _normalize_dtype(dtype=None, precision=None) == "float64"
    assert _normalize_dtype(dtype=None, precision="single") == "float32"
    assert _normalize_dtype(dtype="f64", precision="double") == "float64"
    assert _normalize_dtype(dtype="float32", precision="single") == "float32"
    with pytest.raises(ValueError, match="conflicting"):
        _normalize_dtype(dtype="float64", precision="single")
    with pytest.raises(ValueError, match="unsupported"):
        _normalize_dtype(dtype="single", precision=None)
    with pytest.raises(ValueError, match="unsupported"):
        _normalize_dtype(dtype="double", precision=None)
    with pytest.raises(ValueError, match="unsupported"):
        _normalize_dtype(dtype="half", precision=None)


def test_charge_spin_requirements_are_validated_without_openmm():
    class Requirements:
        total_charge = True
        spin_multiplicity = True

    class Capabilities:
        supports_energy = True
        supports_forces = True
        supports_float64 = True
        supports_float32 = False
        supported_species: list[str] = []
        supported_heads: list[str] = []
        input_requirements = Requirements()

    class FakeModel:
        def capabilities(self):
            return Capabilities()

    with pytest.raises(llmlip.UnsupportedCapabilityError, match="total_charge"):
        _validate_model_capabilities(
            FakeModel(),
            species=SPECIES,
            dtype="float64",
            head=None,
            total_charge=None,
            spin_multiplicity=1.0,
        )
    with pytest.raises(llmlip.UnsupportedCapabilityError, match="spin_multiplicity"):
        _validate_model_capabilities(
            FakeModel(),
            species=SPECIES,
            dtype="float64",
            head=None,
            total_charge=0.0,
            spin_multiplicity=None,
        )


def test_real_model_capabilities_expose_input_requirements():
    _require_native_llmlip()
    model = llmlip.Model(str(MODEL_PATH))
    capabilities = model.capabilities()

    assert isinstance(capabilities.supported_species, list)
    assert isinstance(capabilities.supported_heads, list)
    assert hasattr(capabilities.input_requirements, "total_charge")
    assert hasattr(capabilities.input_requirements, "spin_multiplicity")

    pbc_capabilities = llmlip.Model(str(PBC_MODEL_PATH)).capabilities()
    assert set(PBC_SPECIES).issubset(set(pbc_capabilities.supported_species))


def test_create_llmlip_force_energy_and_force_parity():
    _require_pythonforce()
    topology = _minimal_topology(SPECIES)
    system = _minimal_system(len(SPECIES))
    force = create_llmlip_force(topology, system, MODEL_PATH)
    system.addForce(force)

    energy, forces = _context_energy_forces(system, POSITIONS_ANG)

    model = llmlip.Model(str(MODEL_PATH))
    pred = model.predict(species=SPECIES, positions=POSITIONS_ANG, energy=True, forces=True)

    assert energy == pytest.approx(pred.energy * EV_TO_KJMOL, rel=1.0e-10, abs=1.0e-10)
    np.testing.assert_allclose(forces, pred.forces * FORCE_SCALE, rtol=1.0e-10, atol=1.0e-10)


def test_openmm_reported_force_matches_energy_finite_difference_sign():
    _require_pythonforce()
    topology = _minimal_topology(SPECIES)
    system = _minimal_system(len(SPECIES))
    force = create_llmlip_force(topology, system, MODEL_PATH)
    system.addForce(force)

    _energy, forces = _context_energy_forces(system, POSITIONS_ANG)
    delta_ang = 1.0e-4
    atom = 1
    axis = 0
    positions_plus = POSITIONS_ANG.copy()
    positions_minus = POSITIONS_ANG.copy()
    positions_plus[atom, axis] += delta_ang
    positions_minus[atom, axis] -= delta_ang
    energy_plus, _forces_plus = _context_energy_forces(system, positions_plus)
    energy_minus, _forces_minus = _context_energy_forces(system, positions_minus)

    fd_force = -(energy_plus - energy_minus) / (2.0 * delta_ang / ANGSTROM_PER_NM)
    assert forces[atom, axis] == pytest.approx(fd_force, rel=1.0e-4, abs=1.0e-3)


def test_subset_atoms_zeroes_non_ml_forces():
    _require_pythonforce()
    topology = _minimal_topology(["O", "H", "H", "He"])
    system = _minimal_system(4)
    positions = np.vstack([POSITIONS_ANG, np.array([[5.0, 5.0, 5.0]], dtype=np.float64)])

    force = create_llmlip_force(topology, system, MODEL_PATH, atoms=[0, 1, 2], dtype="float64")
    system.addForce(force)

    energy, forces = _context_energy_forces(system, positions)
    model = llmlip.Model(str(MODEL_PATH))
    pred = model.predict(species=SPECIES, positions=POSITIONS_ANG, energy=True, forces=True)

    assert energy == pytest.approx(pred.energy * EV_TO_KJMOL, rel=1.0e-10, abs=1.0e-10)
    np.testing.assert_allclose(forces[:3], pred.forces * FORCE_SCALE, rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(forces[3], np.zeros(3), atol=1.0e-14)


def test_duplicate_subset_atoms_fail_before_prediction():
    _require_pythonforce()
    topology = _minimal_topology(["O", "H", "H", "He"])
    system = _minimal_system(4)

    with pytest.raises(ValueError, match="duplicate"):
        create_llmlip_force(topology, system, MODEL_PATH, atoms=[0, 0, 1], dtype="float64")


def test_topology_system_particle_count_mismatch_fails_at_force_creation():
    _require_pythonforce()
    topology = _minimal_topology(SPECIES)
    system = _minimal_system(len(SPECIES) + 1)

    with pytest.raises(ValueError, match="topology has 3 atoms but system has 4 particles"):
        create_llmlip_force(topology, system, MODEL_PATH, dtype="float64")


def test_unknown_model_species_fails_at_force_creation():
    _require_pythonforce()
    topology = _minimal_topology(["H", "C", "O"], cell_ang=PBC_CELL_ANG)
    system = _minimal_system(3, cell_ang=PBC_CELL_ANG)

    with pytest.raises(llmlip.UnsupportedCapabilityError, match="unsupported.*species"):
        create_llmlip_force(topology, system, PBC_MODEL_PATH, dtype="float64")


def test_real_charge_spin_requirement_model_fails_at_force_creation():
    _require_pythonforce()
    topology = _minimal_topology(SPECIES)
    system = _minimal_system(len(SPECIES))

    with pytest.raises(llmlip.UnsupportedCapabilityError, match="total_charge"):
        create_llmlip_force(topology, system, ORB_METADATA_MODEL_PATH, dtype="float64")
    with pytest.raises(llmlip.UnsupportedCapabilityError, match="spin_multiplicity"):
        create_llmlip_force(
            topology,
            system,
            ORB_METADATA_MODEL_PATH,
            dtype="float64",
            total_charge=0.0,
        )


@pytest.mark.parametrize(
    ("spin_alias", "expected_spin"),
    [
        ({"multiplicity": 2.0}, 2.0),
        ({"spin": 3.0}, 3.0),
    ],
)
def test_openmmml_create_system_forwards_charge_spin_aliases(
    monkeypatch, spin_alias, expected_spin
):
    openmm, _app, _unit = _openmm_modules()
    openmmml = pytest.importorskip("openmmml")
    from llmlip import openmm as openmm_module
    from llmlip.openmmml import LlmlipPotentialImplFactory

    seen = {}

    def fake_create_llmlip_force(topology, system, model_path, **kwargs):
        seen.update(
            {
                "topology": topology,
                "system": system,
                "model_path": model_path,
                **kwargs,
            }
        )
        return openmm.CustomExternalForce("0")

    monkeypatch.setattr(openmm_module, "create_llmlip_force", fake_create_llmlip_force)
    openmmml.MLPotential.registerImplFactory("llmlip", LlmlipPotentialImplFactory())

    topology = _minimal_topology(SPECIES)
    potential = openmmml.MLPotential("llmlip", modelPath=str(MODEL_PATH))
    system = potential.createSystem(
        topology,
        removeCMMotion=False,
        charge=-1.0,
        dtype="float64",
        **spin_alias,
    )

    assert system.getNumForces() == 1
    assert seen["topology"] is topology
    assert seen["system"] is system
    assert seen["model_path"] == str(MODEL_PATH)
    assert seen["atoms"] is None
    assert seen["force_group"] == 0
    assert seen["total_charge"] == -1.0
    assert seen["spin_multiplicity"] == expected_spin
    assert seen["dtype"] == "float64"


def test_openmmml_factory_allows_baked_default(monkeypatch):
    from llmlip import openmmml as openmmml_module
    from llmlip.openmmml import LlmlipPotentialImplFactory

    monkeypatch.setattr(openmmml_module, "_import_openmmml", lambda: (object, object))

    impl = LlmlipPotentialImplFactory().createImpl("llmlip")

    assert impl.model_path is None


def test_openmmml_factory_creates_system():
    _require_pythonforce()
    openmmml = pytest.importorskip("openmmml")
    from llmlip.openmmml import LlmlipPotentialImplFactory

    openmmml.MLPotential.registerImplFactory("llmlip", LlmlipPotentialImplFactory())

    topology = _minimal_topology(SPECIES)
    potential = openmmml.MLPotential("llmlip", modelPath=str(MODEL_PATH))
    system = potential.createSystem(topology, dtype="float64", removeCMMotion=False)
    assert system.getNumForces() == 1

    energy, forces = _context_energy_forces(system, POSITIONS_ANG)
    model = llmlip.Model(str(MODEL_PATH))
    pred = model.predict(species=SPECIES, positions=POSITIONS_ANG, energy=True, forces=True)

    assert energy == pytest.approx(pred.energy * EV_TO_KJMOL, rel=1.0e-10, abs=1.0e-10)
    np.testing.assert_allclose(forces, pred.forces * FORCE_SCALE, rtol=1.0e-10, atol=1.0e-10)


def test_openmmml_mixed_non_interpolate_system():
    _require_pythonforce()
    openmmml = pytest.importorskip("openmmml")
    from llmlip.openmmml import LlmlipPotentialImplFactory

    openmmml.MLPotential.registerImplFactory("llmlip", LlmlipPotentialImplFactory())

    topology = _minimal_topology(["O", "H", "H", "He"])
    mm_system = _minimal_system(4)
    potential = openmmml.MLPotential("llmlip", modelPath=str(MODEL_PATH))
    system = potential.createMixedSystem(
        topology,
        mm_system,
        atoms=[0, 1, 2],
        removeConstraints=False,
        interpolate=False,
        dtype="float64",
    )
    assert system.getNumParticles() == 4
    assert system.getNumForces() >= 1

    positions = np.vstack([POSITIONS_ANG, np.array([[5.0, 5.0, 5.0]], dtype=np.float64)])
    energy, forces = _context_energy_forces(system, positions)
    assert np.isfinite(energy)
    np.testing.assert_allclose(forces[3], np.zeros(3), atol=1.0e-14)


def test_openmmml_mixed_interpolate_documents_pythonforce_pickle_limitation():
    openmm = _require_pythonforce()
    openmmml = pytest.importorskip("openmmml")
    from llmlip.openmmml import LlmlipPotentialImplFactory

    openmmml.MLPotential.registerImplFactory("llmlip", LlmlipPotentialImplFactory())

    topology = _minimal_topology(["O", "H", "H", "He"])
    mm_system = _minimal_system(4)
    potential = openmmml.MLPotential("llmlip", modelPath=str(MODEL_PATH))
    with pytest.raises(openmm.OpenMMException, match="PythonForce.*pickl"):
        potential.createMixedSystem(
            topology,
            mm_system,
            atoms=[0, 1, 2],
            removeConstraints=False,
            interpolate=True,
            dtype="float64",
        )


def test_unsupported_dtype_fails_early_for_synthetic_model():
    _require_pythonforce()
    topology = _minimal_topology(SPECIES)
    system = _minimal_system(len(SPECIES))
    with pytest.raises(llmlip.UnsupportedCapabilityError, match="float32"):
        create_llmlip_force(topology, system, MODEL_PATH, dtype="float32")


def test_model_path_and_model_are_exclusive():
    _require_pythonforce()
    topology = _minimal_topology(SPECIES)
    system = _minimal_system(len(SPECIES))
    model = llmlip.Model(str(MODEL_PATH))
    with pytest.raises(ValueError, match="only one"):
        create_llmlip_force(topology, system, MODEL_PATH, model=model)


def test_create_llmlip_force_uses_baked_default(monkeypatch):
    _require_pythonforce()
    from llmlip import openmm as openmm_module

    class Requirements:
        total_charge = False
        spin_multiplicity = False

    class Capabilities:
        supports_energy = True
        supports_forces = True
        supports_float64 = True
        supports_float32 = False
        supported_species = SPECIES
        supported_heads: list[str] = []
        input_requirements = Requirements()

    class FakeModel:
        def capabilities(self):
            return Capabilities()

    loaded = []

    def fake_load_baked_model():
        loaded.append(True)
        return FakeModel()

    monkeypatch.setattr(openmm_module, "load_baked_model", fake_load_baked_model)
    topology = _minimal_topology(SPECIES)
    system = _minimal_system(len(SPECIES))

    force = create_llmlip_force(topology, system, dtype="float64")

    assert force.getForceGroup() == 0
    assert loaded == [True]


def test_non_ml_missing_element_is_allowed_but_ml_missing_element_fails():
    _require_pythonforce()
    topology = _minimal_topology(["O", "H", "H", "X"])
    system = _minimal_system(4)

    create_llmlip_force(topology, system, MODEL_PATH, atoms=[0, 1, 2], dtype="float64")
    with pytest.raises(ValueError, match="elements"):
        create_llmlip_force(topology, system, MODEL_PATH, atoms=[0, 1, 3], dtype="float64")


def test_periodic_position_wrapper_handles_triclinic_unwrapped_coordinates():
    fractional = np.array(
        [
            [0.05, 0.10, 0.15],
            [0.90, 0.20, 0.30],
            [0.25, 0.85, 0.40],
        ],
        dtype=np.float64,
    )
    in_box = fractional @ TRICLINIC_CELL_ANG
    lattice_drift = np.array(
        [
            [3.0, -4.0, 5.0],
            [-6.0, 2.0, -3.0],
            [4.0, 4.0, -5.0],
        ],
        dtype=np.float64,
    )
    unwrapped = in_box + lattice_drift @ TRICLINIC_CELL_ANG

    wrapped = _wrap_positions_into_periodic_cell(unwrapped, TRICLINIC_CELL_ANG)

    np.testing.assert_allclose(wrapped, in_box, rtol=1.0e-12, atol=1.0e-12)


def test_periodic_callback_wraps_subset_positions_before_prediction():
    _openmm, _app, unit = _openmm_modules()
    fractional = np.array(
        [
            [0.05, 0.10, 0.15],
            [0.90, 0.20, 0.30],
            [0.25, 0.85, 0.40],
        ],
        dtype=np.float64,
    )
    in_box = fractional @ TRICLINIC_CELL_ANG
    lattice_drift = np.array(
        [
            [5.0, -3.0, 4.0],
            [0.0, 0.0, 0.0],
            [-4.0, 6.0, -5.0],
        ],
        dtype=np.float64,
    )
    unwrapped = in_box + lattice_drift @ TRICLINIC_CELL_ANG
    indices = np.array([2, 0], dtype=np.int64)
    seen = {}

    class RecordingModel:
        def predict(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(
                energy=0.0,
                forces=np.zeros((len(kwargs["species"]), 3), dtype=np.float64),
            )

    class FakeState:
        def getPositions(self, asNumpy):
            assert asNumpy
            return unit.Quantity(unwrapped, unit.angstrom)

        def getPeriodicBoxVectors(self, asNumpy):
            assert asNumpy
            return unit.Quantity(TRICLINIC_CELL_ANG, unit.angstrom)

    energy, forces = _compute_llmlip(
        FakeState(),
        model=RecordingModel(),
        species=["O", "H"],
        indices=indices,
        periodic=True,
        dtype="float64",
        head=None,
        total_charge=None,
        spin_multiplicity=None,
        mace_mh1_wigner_mode="true_float64",
    )

    assert energy == 0.0
    assert forces.shape == (len(PBC_SPECIES), 3)
    np.testing.assert_allclose(
        seen["positions"],
        _wrap_positions_into_periodic_cell(unwrapped[indices], TRICLINIC_CELL_ANG),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_pbc_openmm_path_matches_direct_prediction():
    _require_pythonforce()
    topology = _minimal_topology(PBC_SPECIES, cell_ang=PBC_CELL_ANG)
    system = _minimal_system(len(PBC_SPECIES), cell_ang=PBC_CELL_ANG)
    force = create_llmlip_force(topology, system, PBC_MODEL_PATH, dtype="float64")
    system.addForce(force)

    energy, forces = _context_energy_forces(system, PBC_POSITIONS_ANG, cell_ang=PBC_CELL_ANG)

    model = llmlip.Model(str(PBC_MODEL_PATH))
    pred = model.predict(
        species=PBC_SPECIES,
        positions=PBC_POSITIONS_ANG,
        cell=PBC_CELL_ANG,
        pbc=(True, True, True),
        energy=True,
        forces=True,
    )

    assert energy == pytest.approx(pred.energy * EV_TO_KJMOL, rel=1.0e-10, abs=1.0e-10)
    np.testing.assert_allclose(forces, pred.forces * FORCE_SCALE, rtol=1.0e-10, atol=1.0e-10)
