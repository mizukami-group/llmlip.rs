from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import warnings
from typing import Literal, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from . import (
    BackendUnavailableError,
    CliBackendError,
    InvalidStructureError,
    Model,
    PredictionError,
    UnsupportedCapabilityError,
)


Dtype = Literal["float64", "float32", "f64", "f32", "model-native"]
ExtraOutputs = tuple[NDArray[np.float64], NDArray[np.float64]]
CLI_PROCESS_WARNING = (
    "CliBackend starts one llmlip process per prediction; use backend='direct' "
    "for MD or other repeated workflows. Persistent llmlip serve support is a "
    "planned follow-up, not part of this backend yet."
)


@dataclass(frozen=True)
class PredictionRequest:
    species: Sequence[str]
    positions: NDArray[np.float64]
    cell: NDArray[np.float64] | None
    pbc: tuple[bool, bool, bool]
    energy: bool = True
    forces: bool = True
    stress: bool = False
    virial: bool = False
    dtype: Dtype = "float64"
    head: str | None = None
    total_charge: float | None = None
    spin_multiplicity: float | None = None
    mace_mh1_wigner_mode: str = "true_float64"


@dataclass(frozen=True)
class Prediction:
    energy: float | None
    forces: NDArray[np.float64] | None
    stress: NDArray[np.float64] | None
    virial: NDArray[np.float64] | None


class PredictionBackend(Protocol):
    protocol: str

    def predict(self, request: PredictionRequest) -> Prediction: ...

    def get_descriptors(
        self,
        request: PredictionRequest,
        *,
        invariants_only: bool = True,
        num_layers: int = -1,
    ) -> NDArray[np.float64]: ...

    def get_atomic_energies(self, request: PredictionRequest) -> NDArray[np.float64]: ...

    def get_extra_outputs(
        self,
        request: PredictionRequest,
        *,
        invariants_only: bool = True,
        num_layers: int = -1,
    ) -> ExtraOutputs: ...

    def close(self) -> None: ...


class DirectBackend:
    protocol = "direct"

    def __init__(self, model_path: str | Path | None = None, *, model: Model | None = None):
        if model is None and model_path is None:
            raise BackendUnavailableError("direct backend requires model_path or model")
        try:
            self.model = model if model is not None else Model(str(model_path))
        except Exception as exc:
            raise BackendUnavailableError(f"direct backend unavailable: {exc}") from exc

    def predict(self, request: PredictionRequest) -> Prediction:
        dtype = _normalize_direct_dtype(self.model, request.dtype)
        try:
            pred = self.model.predict(
                species=list(request.species),
                positions=np.asarray(request.positions, dtype=np.float64),
                cell=None
                if request.cell is None
                else np.asarray(request.cell, dtype=np.float64),
                pbc=tuple(bool(value) for value in request.pbc),
                energy=request.energy,
                forces=request.forces,
                stress=request.stress,
                virial=request.virial,
                dtype=dtype,
                head=request.head,
                total_charge=request.total_charge,
                spin_multiplicity=request.spin_multiplicity,
                mace_mh1_wigner_mode=request.mace_mh1_wigner_mode,
            )
        except (InvalidStructureError, PredictionError):
            raise
        return Prediction(
            pred.energy,
            None if pred.forces is None else np.asarray(pred.forces, dtype=np.float64),
            None if pred.stress is None else np.asarray(pred.stress, dtype=np.float64),
            None if pred.virial is None else np.asarray(pred.virial, dtype=np.float64),
        )

    def get_descriptors(
        self,
        request: PredictionRequest,
        *,
        invariants_only: bool = True,
        num_layers: int = -1,
    ) -> NDArray[np.float64]:
        dtype = _normalize_direct_dtype(self.model, request.dtype)
        try:
            descriptors = self.model.get_descriptors(
                species=list(request.species),
                positions=np.asarray(request.positions, dtype=np.float64),
                cell=None
                if request.cell is None
                else np.asarray(request.cell, dtype=np.float64),
                pbc=tuple(bool(value) for value in request.pbc),
                invariants_only=bool(invariants_only),
                num_layers=int(num_layers),
                dtype=dtype,
            )
        except (InvalidStructureError, PredictionError):
            raise
        return np.asarray(descriptors, dtype=np.float64)

    def get_atomic_energies(self, request: PredictionRequest) -> NDArray[np.float64]:
        dtype = _normalize_direct_dtype(self.model, request.dtype)
        try:
            energies = self.model.get_atomic_energies(
                species=list(request.species),
                positions=np.asarray(request.positions, dtype=np.float64),
                cell=None
                if request.cell is None
                else np.asarray(request.cell, dtype=np.float64),
                pbc=tuple(bool(value) for value in request.pbc),
                dtype=dtype,
            )
        except (InvalidStructureError, PredictionError):
            raise
        return np.asarray(energies, dtype=np.float64)

    def get_extra_outputs(
        self,
        request: PredictionRequest,
        *,
        invariants_only: bool = True,
        num_layers: int = -1,
    ) -> ExtraOutputs:
        dtype = _normalize_direct_dtype(self.model, request.dtype)
        try:
            descriptors, energies = self.model.get_extra_outputs(
                species=list(request.species),
                positions=np.asarray(request.positions, dtype=np.float64),
                cell=None
                if request.cell is None
                else np.asarray(request.cell, dtype=np.float64),
                pbc=tuple(bool(value) for value in request.pbc),
                invariants_only=bool(invariants_only),
                num_layers=int(num_layers),
                dtype=dtype,
            )
        except (InvalidStructureError, PredictionError):
            raise
        return (
            np.asarray(descriptors, dtype=np.float64),
            np.asarray(energies, dtype=np.float64),
        )

    def close(self) -> None:
        pass


class _JsonProtocolUnavailable(Exception):
    pass


class CliBackend:
    def __init__(
        self,
        model_path: str | Path,
        binary: str | Path | None = None,
        *,
        use_json: bool = True,
    ):
        self.model_path = str(model_path)
        self.binary = _find_llmlip_binary(binary)
        self.use_json = use_json
        self.protocol = "predict-json" if use_json else "legacy-extxyz"
        warnings.warn(CLI_PROCESS_WARNING, RuntimeWarning, stacklevel=2)

    def predict(self, request: PredictionRequest) -> Prediction:
        if self.use_json:
            try:
                prediction = self._predict_json(request)
                self.protocol = "predict-json"
                return prediction
            except _JsonProtocolUnavailable:
                self.protocol = "legacy-extxyz"
        return self._predict_legacy(request)

    def get_descriptors(
        self,
        request: PredictionRequest,
        *,
        invariants_only: bool = True,
        num_layers: int = -1,
    ) -> NDArray[np.float64]:
        raise UnsupportedCapabilityError(
            "MACE descriptor extraction requires backend='direct'; "
            "the CLI backend does not expose extra outputs"
        )

    def get_atomic_energies(self, request: PredictionRequest) -> NDArray[np.float64]:
        raise UnsupportedCapabilityError(
            "MACE per-atom energy extraction requires backend='direct'; "
            "the CLI backend does not expose extra outputs"
        )

    def get_extra_outputs(
        self,
        request: PredictionRequest,
        *,
        invariants_only: bool = True,
        num_layers: int = -1,
    ) -> ExtraOutputs:
        raise UnsupportedCapabilityError(
            "MACE extra output extraction requires backend='direct'; "
            "the CLI backend does not expose extra outputs"
        )

    def _predict_json(self, request: PredictionRequest) -> Prediction:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.extxyz"
            _write_extxyz(request, input_path)
            cmd = [
                self.binary,
                "predict",
                "--model",
                self.model_path,
                "--structure",
                str(input_path),
                "--json",
            ]
            _append_prediction_flags(cmd, request, include_forces_path=None)
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                if _json_unavailable_stderr(result.stderr):
                    raise _JsonProtocolUnavailable(result.stderr)
                raise CliBackendError(
                    f"llmlip predict --json failed with exit code {result.returncode}:\n"
                    f"{result.stderr}"
                )
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise CliBackendError(
                    f"llmlip predict --json returned invalid JSON:\n{result.stdout}"
                ) from exc
            if data.get("schema_version") != "llmlip-prediction-v1":
                raise CliBackendError(
                    f"unsupported llmlip prediction JSON schema: {data.get('schema_version')!r}"
                )
            return _prediction_from_json(data)

    def _predict_legacy(self, request: PredictionRequest) -> Prediction:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.extxyz"
            forces_path = Path(tmpdir) / "forces.extxyz"
            _write_extxyz(request, input_path)
            cmd = [
                self.binary,
                "predict",
                "--model",
                self.model_path,
                "--structure",
                str(input_path),
            ]
            _append_prediction_flags(
                cmd,
                request,
                include_forces_path=str(forces_path) if request.forces else None,
            )
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise CliBackendError(
                    f"llmlip legacy predict failed with exit code {result.returncode}:\n"
                    f"{result.stderr}"
                )
            stdout = result.stdout
            return Prediction(
                _parse_energy(stdout) if request.energy else None,
                _parse_forces_extxyz(forces_path, len(request.species))
                if request.forces
                else None,
                _parse_matrix(stdout, "stress") if request.stress else None,
                _parse_matrix(stdout, "virial") if request.virial else None,
            )

    def close(self) -> None:
        pass


def _normalize_direct_dtype(model: Model, dtype: str) -> str:
    if dtype != "model-native":
        return dtype
    capabilities = model.capabilities()
    return "float32" if capabilities.supports_float32 else "float64"


def _normalize_cli_dtype(dtype: str) -> str:
    return "float64" if dtype == "model-native" else dtype


def _append_prediction_flags(
    cmd: list[str],
    request: PredictionRequest,
    *,
    include_forces_path: str | None,
) -> None:
    if request.energy:
        cmd.append("--energy")
    if request.forces:
        if include_forces_path is None:
            cmd.append("--forces")
        else:
            cmd.extend(["--forces-out", include_forces_path])
    if request.stress:
        cmd.append("--stress")
    if request.virial:
        cmd.append("--virial")
    cmd.extend(["--dtype", _normalize_cli_dtype(request.dtype)])
    if request.head is not None:
        cmd.extend(["--head", request.head])
    if request.total_charge is not None:
        cmd.extend(["--total-charge", str(request.total_charge)])
    if request.spin_multiplicity is not None:
        cmd.extend(["--spin-multiplicity", str(request.spin_multiplicity)])
    if request.mace_mh1_wigner_mode != "true_float64":
        cmd.extend(["--mace-mh1-wigner-mode", _normalize_wigner_for_cli(request.mace_mh1_wigner_mode)])


def _prediction_from_json(data: dict) -> Prediction:
    return Prediction(
        data.get("energy"),
        None if data.get("forces") is None else np.asarray(data["forces"], dtype=np.float64),
        None if data.get("stress") is None else np.asarray(data["stress"], dtype=np.float64),
        None if data.get("virial") is None else np.asarray(data["virial"], dtype=np.float64),
    )


def _normalize_wigner_for_cli(mode: str) -> str:
    return mode.replace("_", "-")


def _json_unavailable_stderr(stderr: str) -> bool:
    lowered = stderr.lower()
    return "unexpected argument '--json'" in lowered or "unrecognized option '--json'" in lowered


def _find_llmlip_binary(binary: str | Path | None) -> str:
    if binary is not None:
        candidate = Path(binary)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        raise BackendUnavailableError(f"Cannot locate explicit llmlip binary: {candidate}")

    candidates: list[Path] = []
    env = os.environ.get("LLMLIP_BINARY")
    if env:
        candidates.append(Path(env))
    on_path = shutil.which("llmlip")
    if on_path is not None:
        candidates.append(Path(on_path))
    repo_root = _repo_root_from_source()
    if repo_root is not None:
        candidates.append(repo_root / "target" / "debug" / "llmlip")
        candidates.append(repo_root / "target" / "release" / "llmlip")

    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise BackendUnavailableError(
        "Cannot locate llmlip binary. Checked: "
        + (", ".join(checked) if checked else "explicit binary, LLMLIP_BINARY, PATH")
    )


def _repo_root_from_source() -> Path | None:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "Cargo.toml").is_file() and (parent / "crates").is_dir():
            return parent
    return None


def _write_extxyz(request: PredictionRequest, path: Path) -> None:
    species = list(request.species)
    positions = np.asarray(request.positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise InvalidStructureError("positions must have shape (n, 3)")
    if len(species) != positions.shape[0]:
        raise InvalidStructureError("species length must match positions.shape[0]")

    lines = [str(len(species))]
    comment = ["Properties=species:S:1:pos:R:3"]
    if request.cell is not None and any(request.pbc):
        cell = np.asarray(request.cell, dtype=np.float64)
        if cell.shape != (3, 3):
            raise InvalidStructureError("cell must have shape (3, 3)")
        lattice = " ".join(f"{cell[i, j]:.16e}" for i in range(3) for j in range(3))
        comment.append(f'Lattice="{lattice}"')
    comment.append('pbc="' + " ".join("T" if value else "F" for value in request.pbc) + '"')
    lines.append(" ".join(comment))
    for symbol, (x, y, z) in zip(species, positions, strict=True):
        lines.append(f"{symbol} {x:.16e} {y:.16e} {z:.16e}")
    path.write_text("\n".join(lines) + "\n")


def _parse_energy(stdout: str) -> float:
    match = re.search(r"energy\s*=\s*([^\s]+)", stdout)
    if match is None:
        raise CliBackendError(f"Could not parse energy from llmlip output:\n{stdout}")
    return float(match.group(1))


def _parse_forces_extxyz(path: Path, natoms: int) -> NDArray[np.float64]:
    lines = path.read_text().splitlines()
    forces = np.zeros((natoms, 3), dtype=np.float64)
    for index in range(natoms):
        fields = lines[2 + index].split()
        forces[index, 0] = float(fields[4])
        forces[index, 1] = float(fields[5])
        forces[index, 2] = float(fields[6])
    return forces


def _parse_matrix(stdout: str, label: str) -> NDArray[np.float64]:
    pattern = re.escape(label) + r"[^\n]*=\s*\n"
    match = re.search(pattern, stdout)
    if match is None:
        raise CliBackendError(f"Could not find '{label}' matrix in llmlip output:\n{stdout}")
    rest = stdout[match.end() :]
    matrix = np.zeros((3, 3), dtype=np.float64)
    for row_index in range(3):
        row_match = re.search(r"\[([^\]]+)\]", rest)
        if row_match is None:
            raise CliBackendError(f"Could not parse row {row_index} of '{label}' matrix")
        values = row_match.group(1).split()
        for col_index in range(3):
            matrix[row_index, col_index] = float(values[col_index])
        rest = rest[row_match.end() :]
    return matrix
