#!/usr/bin/env python3
"""Write a PET-lite `.e3rs` export from pinned PET locks.

By default this script creates a metadata-only loader fixture. With
`--export-candidate-tensors`, it also writes the conservative-energy candidate
state tensors as manifest payloads, while still recording that PET runtime graph
execution is unsupported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any


EDGE_CONVENTION = "R_j + shift - R_i"
SCHEMA_VERSION = "pet-lite-model-v2"
STATE_INVENTORY_SCHEMA_VERSION = "pet-lite-state-inventory-v1"
ENERGY_POSTPROCESSING_SCHEMA_VERSION = "pet-lite-energy-postprocessing-v1"
STATE_DICT_FORMAT = "manifest-f32-f64-tensors-v1"
UPET_COMMIT = "1f0b65c9efeb31abcfb9a14ca5859f0b3ce45e78"
HF_REPO = "lab-cosmo/upet"
METADATA_RUNTIME_STATUS = "inventory_only_no_tensor_payloads"
PAYLOAD_RUNTIME_STATUS = "manifest_payloads_no_runtime_graph"
LARGE_PAYLOAD_REVIEW_THRESHOLD_BYTES = 64 * 1024 * 1024
TORCH_DTYPE_BYTES = {
    "torch.float32": 4,
    "torch.float64": 8,
    "torch.int64": 8,
    "torch.uint8": 1,
}

CHEMICAL_SYMBOLS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
]


class ExportError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_repo_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def toml_string(value: str) -> str:
    return json.dumps(value)


def toml_string_array(values: list[str]) -> str:
    return "[" + ", ".join(toml_string(value) for value in values) + "]"


def toml_int_array(values: list[int]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise ExportError(message)


def require_str(value: Any, path: str) -> str:
    expect(isinstance(value, str) and value, f"{path} must be a non-empty string")
    return value


def require_int(value: Any, path: str) -> int:
    expect(isinstance(value, int) and value > 0, f"{path} must be a positive integer")
    return value


def require_number(value: Any, path: str) -> float:
    expect(isinstance(value, (int, float)) and value > 0, f"{path} must be positive")
    return float(value)


def symbols_for_atomic_numbers(atomic_numbers: list[int]) -> list[str]:
    symbols: list[str] = []
    for number in atomic_numbers:
        expect(1 <= number <= len(CHEMICAL_SYMBOLS), f"unsupported atomic number {number}")
        symbols.append(CHEMICAL_SYMBOLS[number - 1])
    return symbols


def parse_selected_model(selected_model: str) -> tuple[str, str, str, str]:
    prefix = "pet-"
    expect(selected_model.startswith(prefix), f"unexpected PET selected_model {selected_model!r}")
    parts = selected_model.split("-")
    expect(len(parts) == 4, f"selected_model must look like pet-mad-xs-v1.5.0: {selected_model}")
    family = "-".join(parts[:2])
    size = parts[2]
    version = parts[3].removeprefix("v")
    checkpoint = f"{selected_model}.ckpt"
    return family, size, version, checkpoint


def model_manifest_toml(
    *,
    symbols: list[str],
    cutoff: float,
    reference_version: str,
    fixture_seed: int,
    tensors: list[dict[str, Any]],
) -> str:
    lines = [
        'format_version = "0.1"',
        'architecture = "pet-lite-v1"',
        'dtype = "f64"',
        'length_unit = "angstrom"',
        'energy_unit = "eV"',
        'force_unit = "eV/angstrom"',
        "",
        "[species]",
        f"symbols = {toml_string_array(symbols)}",
        "",
        "[geometry]",
        f"cutoff = {cutoff}",
        f"edge_vector = {toml_string(EDGE_CONVENTION)}",
        "directed_edges = true",
        "",
        "[spherical_harmonics]",
        "l_max = 0",
        "normalize_input = true",
        'normalization = "component"',
        'reference_convention = "pinned-fixture"',
        "",
        "[validation]",
        'reference_package = "upet"',
        f"reference_version = {toml_string(reference_version)}",
        f"fixture_seed = {fixture_seed}",
        "",
        "[pet_lite]",
        'model = "pet_lite.json"',
        'reference_package = "upet"',
        f"reference_commit = {toml_string(UPET_COMMIT)}",
    ]
    for tensor in tensors:
        lines.extend(
            [
                "",
                "[[tensors]]",
                f"name = {toml_string(tensor['name'])}",
                f"dtype = {toml_string(tensor['dtype'])}",
                f"shape = {toml_int_array(tensor['shape'])}",
                'layout = "row-major"',
                f"path = {toml_string(tensor['path'])}",
            ]
        )
    return "\n".join(lines) + "\n"


def build_model_spec(
    artifact: dict[str, Any],
    inspection: dict[str, Any],
    reference_fixture: dict[str, Any] | None,
    *,
    export_candidate_tensors: bool,
    energy_postprocessing: dict[str, Any] | None,
) -> dict[str, Any]:
    selected_model = require_str(artifact.get("selected_model"), "artifact.selected_model")
    expect(selected_model == inspection.get("selected_model"), "artifact and inspection model mismatch")
    if reference_fixture is not None:
        expect(
            selected_model == reference_fixture.get("model", {}).get("selected_model"),
            "artifact and reference fixture model mismatch",
        )

    family, size, version, checkpoint = parse_selected_model(selected_model)
    source = artifact["source"]
    model = artifact["model"]
    metadata = inspection["model_summary"]["pet_metadata"]
    hypers = metadata["hypers"]
    dependency_versions = inspection["environment"]["dependency_versions"]
    atomic_numbers = [require_int(value, "atomic_types[]") for value in metadata["atomic_types"]]
    symbols = symbols_for_atomic_numbers(atomic_numbers)

    cutoff = require_number(hypers["cutoff"], "hypers.cutoff")
    max_neighbors = require_int(hypers["num_neighbors_adaptive"], "hypers.num_neighbors_adaptive")
    if reference_fixture is not None:
        neighbor = reference_fixture["model"]["neighbor_selection"]
        expect(cutoff == neighbor["cutoff"], "inspection/reference cutoff mismatch")
        expect(max_neighbors == neighbor["max_neighbors"], "inspection/reference max-neighbor mismatch")
        tie_break = require_str(neighbor["tie_break"], "reference neighbor tie_break")
    else:
        tie_break = "not_pinned_for_runtime"

    expect(artifact["reference"]["commit"] == UPET_COMMIT, "unexpected UPET commit")
    expect(source["huggingface_repo"] == HF_REPO, "unexpected Hugging Face repo")
    expect(source["lfs_sha256"] == artifact["artifact"]["sha256"], "artifact hash mismatch")
    expect(source["size_bytes"] == artifact["artifact"]["size_bytes"], "artifact size mismatch")

    unsupported_features = [
        "non_conservative_forces",
        "non_conservative_stress",
        "uncertainty",
        "rotational_averaging",
        "torchscript_runtime",
        "arbitrary_checkpoint_import",
    ]
    state_inventory = state_inventory_summary(
        inspection["state_tensors"],
        export_candidate_tensors=export_candidate_tensors,
    )

    model_spec = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "family": family,
            "size": size,
            "version": version,
            "checkpoint": checkpoint,
        },
        "provenance": {
            "upet_commit": artifact["reference"]["commit"],
            "huggingface_repo": source["huggingface_repo"],
            "huggingface_repo_sha": source["huggingface_repo_sha"],
            "checkpoint_lfs_sha256": source["lfs_sha256"],
            "checkpoint_size_bytes": source["size_bytes"],
            "metatrain_version": dependency_versions["metatrain"],
            "metatomic_version": dependency_versions["metatomic-torch"],
        },
        "species": {
            "symbols": symbols,
            "atomic_numbers": atomic_numbers,
        },
        "neighbor_selection": {
            "kind": "adaptive-top-k",
            "cutoff": cutoff,
            "max_neighbors": max_neighbors,
            "tie_break": tie_break,
            "directed_edges": True,
            "edge_vector": EDGE_CONVENTION,
        },
        "architecture": {
            "edge_feature_dim": require_int(
                model["hyperparameters"]["edge_feature_dim"],
                "artifact.model.hyperparameters.edge_feature_dim",
            ),
            "gnn_layers": require_int(
                model["hyperparameters"]["gnn_layers"],
                "artifact.model.hyperparameters.gnn_layers",
            ),
            "attention_layers": require_int(
                model["hyperparameters"]["attention_layers"],
                "artifact.model.hyperparameters.attention_layers",
            ),
            "num_heads": require_int(hypers["num_heads"], "hypers.num_heads"),
            "unsupported_or_unpinned_fields": [
                "adaptive_neighbor_tie_break",
                "runtime_tensor_graph",
            ],
        },
        "outputs": {
            "energy": {"quantity": "energy", "unit": "eV"},
            "forces": {"source": "conservative_vjp"},
        },
        "state_dict_format": STATE_DICT_FORMAT,
        "state_tensor_inventory": {
            "path": "pet_state_inventory.json",
            "schema_version": STATE_INVENTORY_SCHEMA_VERSION,
            **state_inventory,
        },
        "unsupported_features": unsupported_features,
    }
    if energy_postprocessing is not None:
        model_spec["energy_postprocessing"] = energy_postprocessing
    return model_spec


def build_manifest(
    model_spec: dict[str, Any],
    artifact: dict[str, Any],
    reference_fixture: dict[str, Any] | None,
    tensors: list[dict[str, Any]],
) -> str:
    fixture_seed = 24_680
    if reference_fixture is not None:
        fixture_seed = require_int(reference_fixture["seed"], "reference_fixture.seed")
    return model_manifest_toml(
        symbols=model_spec["species"]["symbols"],
        cutoff=model_spec["neighbor_selection"]["cutoff"],
        reference_version=artifact["reference"]["package_version"],
        fixture_seed=fixture_seed,
        tensors=tensors,
    )


def state_tensor_inventory_digest(entries: list[dict[str, Any]]) -> str:
    hasher = hashlib.sha256()
    for entry in entries:
        shape = json.dumps(entry["shape"], separators=(",", ":"))
        for field in [
            entry["name"],
            entry["kind"],
            entry["dtype"],
            str(entry["numel"]),
            shape,
            entry["value_sha256"],
        ]:
            hasher.update(field.encode())
            hasher.update(b"\0")
        hasher.update(b"\n")
    return hasher.hexdigest()


def exportable_candidate(entry: dict[str, Any]) -> bool:
    return (
        classify_state_tensor(entry) == "candidate_conservative_energy_runtime"
        and manifest_dtype(entry["dtype"]) is not None
    )


def state_tensor_byte_count(entry: dict[str, Any]) -> int:
    dtype = require_str(entry.get("dtype"), "state_tensor.dtype")
    expect(dtype in TORCH_DTYPE_BYTES, f"unsupported state tensor dtype {dtype}")
    return require_int(entry.get("numel"), "state_tensor.numel") * TORCH_DTYPE_BYTES[dtype]


def candidate_payload_size_bytes(state_tensors: list[dict[str, Any]]) -> int:
    return sum(
        state_tensor_byte_count(entry)
        for entry in state_tensors
        if exportable_candidate(entry)
    )


def mib(byte_count: int) -> float:
    return byte_count / (1024.0 * 1024.0)


def enforce_payload_size_guard(
    *,
    inspection: dict[str, Any],
    allow_large_payload_export: bool,
) -> None:
    byte_count = candidate_payload_size_bytes(inspection["state_tensors"])
    if byte_count <= LARGE_PAYLOAD_REVIEW_THRESHOLD_BYTES or allow_large_payload_export:
        return
    selected_model = require_str(inspection.get("selected_model"), "inspection.selected_model")
    raise ExportError(
        f"{selected_model} conservative-energy candidate payload is "
        f"{byte_count:,} bytes ({mib(byte_count):.2f} MiB), exceeding the "
        f"{mib(LARGE_PAYLOAD_REVIEW_THRESHOLD_BYTES):.0f} MiB review threshold. "
        "Pass --allow-large-payload-export only after deciding how this large "
        "fixture should be stored."
    )


def state_inventory_summary(
    state_tensors: list[dict[str, Any]],
    *,
    export_candidate_tensors: bool,
) -> dict[str, Any]:
    entries = [
        {
            "name": tensor["name"],
            "kind": tensor["kind"],
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "numel": tensor["numel"],
            "value_sha256": tensor["value_sha256"],
        }
        for tensor in state_tensors
    ]
    manifest_eligible_count = sum(1 for tensor in state_tensors if manifest_dtype(tensor["dtype"]))
    manifest_tensor_count = (
        sum(1 for tensor in state_tensors if exportable_candidate(tensor))
        if export_candidate_tensors
        else 0
    )
    return {
        "upstream_state_tensor_count": len(entries),
        "manifest_eligible_float_tensor_count": manifest_eligible_count,
        "manifest_tensor_count": manifest_tensor_count,
        "runtime_tensor_count": 0,
        "archival_tensor_count": len(entries) - manifest_tensor_count,
        "unclassified_tensor_count": 0,
        "state_tensor_inventory_digest": state_tensor_inventory_digest(entries),
        "runtime_status": (
            PAYLOAD_RUNTIME_STATUS
            if export_candidate_tensors
            else METADATA_RUNTIME_STATUS
        ),
    }


def classify_state_tensor(entry: dict[str, Any]) -> str:
    name = entry["name"]
    dtype = entry["dtype"]
    if name in {
        "covariance_energy_uncertainty",
        "cholesky_energy_uncertainty",
        "multiplier_energy_uncertainty",
        "llpr_ensemble_layers.energy.weight",
    }:
        return "archival_llpr_uncertainty_wrapper"
    if dtype in {"torch.int64", "torch.uint8"}:
        if (
            "species_to_species_index" in name
            or "type_to_index" in name
            or "atomic_types" in name
        ):
            return "archival_species_or_index_buffer"
        return "archival_opaque_python_buffer"
    if name in {
        "model.additive_models.0.dummy_buffer",
        "model.scaler.dummy_buffer",
    }:
        return "archival_opaque_python_buffer"
    if name.startswith("model.additive_models.") or name.startswith("additive_models."):
        return "archival_opaque_python_buffer"
    if name.startswith("model.scaler.") or name.startswith("scaler."):
        return "archival_opaque_python_buffer"
    if "non_conservative" in name:
        return "archival_direct_nonconservative_head"
    return "candidate_conservative_energy_runtime"


def manifest_dtype(dtype: str) -> str | None:
    if dtype == "torch.float32":
        return "f32"
    if dtype == "torch.float64":
        return "f64"
    return None


def sanitize_tensor_filename(index: int, name: str, dtype: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if len(stem) > 120:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
        stem = f"{stem[:96]}_{digest}"
    return f"weights/{index:03}_{stem}.{dtype}.bin"


def tensor_value_sha256(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and sys.byteorder == "big"
    ):
        array = array.byteswap().newbyteorder("<")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def tensor_bytes(tensor: Any) -> bytes:
    array = tensor.detach().cpu().contiguous().numpy()
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and sys.byteorder == "big"
    ):
        array = array.byteswap().newbyteorder("<")
    return array.tobytes(order="C")


def import_reference(upet_dir: Path) -> tuple[Any, Any]:
    sys.path.insert(0, str((upet_dir / "src").resolve()))
    try:
        from warp._src import utils as wp_utils  # type: ignore[import-not-found]

        if not hasattr(wp_utils, "warn"):
            wp_utils.warn = warnings.warn
        import upet  # type: ignore[import-not-found]
        from metatrain.utils.io import load_model as load_metatrain_model  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - reference environment only
        raise ExportError(
            "failed to import PET export dependencies. Use the reference venv "
            "with upet/metatrain installed.\n"
            f"{type(exc).__name__}: {exc}"
        ) from exc
    upet_file = Path(upet.__file__).resolve()
    expected = (upet_dir / "src" / "upet").resolve()
    if not str(upet_file).startswith(str(expected)):
        raise ExportError(
            "imported upet does not come from the pinned checkout\n"
            f"imported: {upet_file}\nexpected under: {expected}"
        )
    return upet, load_metatrain_model


def candidate_tensor_payloads(
    *,
    upet_dir: Path,
    checkpoint: Path,
    inspection: dict[str, Any],
) -> list[dict[str, Any]]:
    if not checkpoint.exists():
        raise ExportError(f"missing PET checkpoint for tensor export: {checkpoint}")
    _upet, load_metatrain_model = import_reference(upet_dir)
    loaded_model = load_metatrain_model(str(checkpoint))
    state = loaded_model.state_dict()
    payloads: list[dict[str, Any]] = []
    for index, entry in enumerate(inspection["state_tensors"]):
        if not exportable_candidate(entry):
            continue
        name = entry["name"]
        if name not in state:
            raise ExportError(f"checkpoint state_dict missing tensor {name}")
        tensor = state[name]
        dtype = manifest_dtype(entry["dtype"])
        assert dtype is not None
        expect(str(tensor.dtype) == entry["dtype"], f"tensor {name} dtype drifted")
        expect(list(tensor.shape) == entry["shape"], f"tensor {name} shape drifted")
        value_sha256 = tensor_value_sha256(tensor)
        expect(
            value_sha256 == entry["value_sha256"],
            f"tensor {name} value digest drifted: {value_sha256}",
        )
        path = sanitize_tensor_filename(index, name, dtype)
        payloads.append(
            {
                "index": index,
                "name": name,
                "dtype": dtype,
                "shape": entry["shape"],
                "layout": "row-major",
                "path": path,
                "value_sha256": value_sha256,
                "bytes": tensor_bytes(tensor),
            }
        )
    return payloads


def energy_postprocessing_spec(
    *,
    upet_dir: Path,
    checkpoint: Path,
    inspection: dict[str, Any],
) -> dict[str, Any]:
    if not checkpoint.exists():
        raise ExportError(f"missing PET checkpoint for post-processing export: {checkpoint}")
    _upet, load_metatrain_model = import_reference(upet_dir)
    try:
        import metatensor.torch as mts  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - reference environment only
        raise ExportError(
            "failed to import metatensor.torch for PET post-processing export"
        ) from exc

    state = load_metatrain_model(str(checkpoint)).state_dict()
    scaler_name = first_present_state_key(
        state,
        [
            "model.scaler.energy_scaler_buffer",
            "scaler.energy_scaler_buffer",
        ],
        "post-processing energy scaler",
    )
    composition_name = first_present_state_key(
        state,
        [
            "model.additive_models.0.energy_composition_buffer",
            "additive_models.0.energy_composition_buffer",
        ],
        "post-processing energy composition",
    )

    scale_map = mts.load_buffer(state[scaler_name])
    scale_block = scale_map.block()
    expect(
        list(scale_block.samples.names) == ["atomic_type"],
        "PET energy scaler samples must be atomic_type labels",
    )
    expect(
        scale_block.samples.values.tolist() == [[-1]],
        "PET energy scaler must be a global atomic_type=-1 scale",
    )
    expect(
        list(scale_block.properties.names) == ["energy"],
        "PET energy scaler properties must be energy labels",
    )
    expect(
        scale_block.properties.values.tolist() == [[0]],
        "PET energy scaler must contain one energy property",
    )
    scale_values = scale_block.values.detach().cpu().reshape(-1).tolist()
    expect(len(scale_values) == 1, "PET energy scaler must contain exactly one value")
    scale = float(scale_values[0])
    expect(math.isfinite(scale) and scale > 0.0, "PET energy scaler value must be positive")

    composition_map = mts.load_buffer(state[composition_name])
    composition_block = composition_map.block()
    expect(
        list(composition_block.samples.names) == ["center_type"],
        "PET composition samples must be center_type labels",
    )
    expect(
        list(composition_block.properties.names) == ["energy"],
        "PET composition properties must be energy labels",
    )
    expect(
        composition_block.properties.values.tolist() == [[0]],
        "PET composition must contain one energy property",
    )
    atomic_numbers = [
        int(value)
        for value in composition_block.samples.column("center_type").detach().cpu().tolist()
    ]
    values = [
        float(value)
        for value in composition_block.values.detach().cpu().reshape(-1).tolist()
    ]
    expected_atomic_numbers = [
        require_int(value, "inspection atomic_types[]")
        for value in inspection["model_summary"]["pet_metadata"]["atomic_types"]
    ]
    expect(
        atomic_numbers == expected_atomic_numbers,
        "PET composition center_type labels must match inspected atomic types",
    )
    expect(
        len(values) == len(atomic_numbers),
        "PET composition values length must match atomic number labels",
    )
    for index, value in enumerate(values):
        expect(math.isfinite(value), f"PET composition value {index} must be finite")

    return {
        "schema_version": ENERGY_POSTPROCESSING_SCHEMA_VERSION,
        "order": ["scale", "composition_additive"],
        "scale": {
            "target": "energy",
            "kind": "global_per_structure_scale_applied_to_per_atom",
            "value": scale,
            "source_state_tensor": scaler_name,
        },
        "composition_additive": {
            "target": "energy",
            "sample_name": "center_type",
            "atomic_numbers": atomic_numbers,
            "values": values,
            "unit": "eV",
            "source_state_tensor": composition_name,
        },
    }


def first_present_state_key(
    state: dict[str, Any],
    candidates: list[str],
    context: str,
) -> str:
    for name in candidates:
        if name in state:
            return name
    raise ExportError(
        f"checkpoint state_dict missing {context}; tried {', '.join(candidates)}"
    )


def write_tensor_payloads(out: Path, payloads: list[dict[str, Any]]) -> None:
    for payload in payloads:
        path = out / payload["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload["bytes"])


def build_state_inventory(
    *,
    artifact: dict[str, Any],
    inspection: dict[str, Any],
    artifact_manifest_path: Path,
    inspection_path: Path,
    reference_fixture_path: Path | None,
    manifest_tensors: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest_by_name = {tensor["name"]: tensor for tensor in manifest_tensors}
    entries: list[dict[str, Any]] = []
    for index, tensor in enumerate(inspection["state_tensors"]):
        manifest_dtype_value = manifest_dtype(tensor["dtype"])
        payload = manifest_by_name.get(tensor["name"])
        manifest_tensor = None
        if payload is not None:
            manifest_tensor = {
                "name": payload["name"],
                "dtype": payload["dtype"],
                "shape": payload["shape"],
                "layout": payload["layout"],
                "path": payload["path"],
                "value_sha256": payload["value_sha256"],
            }
        entries.append(
            {
                "index": index,
                "name": tensor["name"],
                "kind": tensor["kind"],
                "dtype": tensor["dtype"],
                "manifest_dtype": manifest_dtype_value,
                "shape": tensor["shape"],
                "numel": tensor["numel"],
                "value_sha256": tensor["value_sha256"],
                "classification": classify_state_tensor(tensor),
                "manifest_tensor": manifest_tensor,
            }
        )
    dtype_counts = Counter(entry["dtype"] for entry in entries)
    kind_counts = Counter(entry["kind"] for entry in entries)
    classification_counts = Counter(entry["classification"] for entry in entries)
    manifest_eligible_count = sum(1 for entry in entries if entry["manifest_dtype"] is not None)
    manifest_tensor_count = len(manifest_tensors)
    return {
        "schema_version": STATE_INVENTORY_SCHEMA_VERSION,
        "selected_model": artifact["selected_model"],
        "state_dict_format": STATE_DICT_FORMAT,
        "artifact_manifest": stable_repo_path(artifact_manifest_path),
        "inspection": stable_repo_path(inspection_path),
        "reference_fixture": stable_repo_path(reference_fixture_path),
        "reference": {
            "name": artifact["reference"]["name"],
            "repository": artifact["reference"]["repository"],
            "commit": artifact["reference"]["commit"],
            "package_version": artifact["reference"]["package_version"],
        },
        "artifact": {
            "checkpoint_sha256": artifact["source"]["lfs_sha256"],
            "checkpoint_size_bytes": artifact["source"]["size_bytes"],
            "binary_tracked_by_git": False,
        },
        "summary": {
            "upstream_state_tensor_count": len(entries),
            "manifest_tensor_count": manifest_tensor_count,
            "manifest_eligible_float_tensor_count": manifest_eligible_count,
            "runtime_tensor_count": 0,
            "archival_tensor_count": len(entries) - manifest_tensor_count,
            "unclassified_tensor_count": 0,
            "dtype_counts": dict(sorted(dtype_counts.items())),
            "kind_counts": dict(sorted(kind_counts.items())),
            "classification_counts": dict(sorted(classification_counts.items())),
            "state_tensor_inventory_digest": state_tensor_inventory_digest(entries),
            "runtime_status": (
                PAYLOAD_RUNTIME_STATUS
                if manifest_tensor_count
                else METADATA_RUNTIME_STATUS
            ),
        },
        "entries": entries,
        "notes": [
            (
                "This export contains conservative-energy candidate binary tensor payloads but no executable runtime graph."
                if manifest_tensor_count
                else "This metadata-only export intentionally omits binary tensor payloads."
            ),
            "No upstream state key is dropped; every tensor is listed with an explicit classification.",
            "prediction remains UnsupportedRuntime until the PET execution graph and required runtime tensors are pinned.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-manifest", required=True, type=Path)
    parser.add_argument("--inspection", required=True, type=Path)
    parser.add_argument("--reference-fixture", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--upet-dir", type=Path, default=Path("third_party/upet"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--export-candidate-tensors",
        action="store_true",
        help="write conservative-energy candidate tensors as manifest payloads",
    )
    parser.add_argument(
        "--allow-large-payload-export",
        action="store_true",
        help=(
            "allow exporting candidate tensor payloads above the review threshold; "
            "intended for ignored/local or explicitly approved large artifacts"
        ),
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing output directory")
    args = parser.parse_args()

    artifact = load_json(args.artifact_manifest)
    inspection = load_json(args.inspection)
    reference_fixture = load_json(args.reference_fixture) if args.reference_fixture else None

    checkpoint = args.checkpoint or Path(artifact["artifact"]["cache_path"])
    if args.export_candidate_tensors:
        enforce_payload_size_guard(
            inspection=inspection,
            allow_large_payload_export=args.allow_large_payload_export,
        )
    payloads = (
        candidate_tensor_payloads(
            upet_dir=args.upet_dir,
            checkpoint=checkpoint,
            inspection=inspection,
        )
        if args.export_candidate_tensors
        else []
    )
    energy_postprocessing = (
        energy_postprocessing_spec(
            upet_dir=args.upet_dir,
            checkpoint=checkpoint,
            inspection=inspection,
        )
        if args.export_candidate_tensors
        else None
    )
    model_spec = build_model_spec(
        artifact,
        inspection,
        reference_fixture,
        export_candidate_tensors=args.export_candidate_tensors,
        energy_postprocessing=energy_postprocessing,
    )
    expected_payloads = model_spec["state_tensor_inventory"]["manifest_tensor_count"]
    expect(
        len(payloads) == expected_payloads,
        f"expected {expected_payloads} tensor payloads, generated {len(payloads)}",
    )
    manifest = build_manifest(model_spec, artifact, reference_fixture, payloads)

    if args.out.exists():
        if not args.force:
            raise ExportError(f"{args.out} already exists; pass --force to overwrite")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    (args.out / "manifest.toml").write_text(manifest, encoding="utf-8")
    write_tensor_payloads(args.out, payloads)
    write_json(args.out / "pet_lite.json", model_spec)
    write_json(
        args.out / "pet_state_inventory.json",
        build_state_inventory(
            artifact=artifact,
            inspection=inspection,
            artifact_manifest_path=args.artifact_manifest,
            inspection_path=args.inspection,
            reference_fixture_path=args.reference_fixture,
            manifest_tensors=payloads,
        ),
    )

    print(
        json.dumps(
            {
                "model_dir": str(args.out),
                "schema_version": SCHEMA_VERSION,
                "selected_model": artifact["selected_model"],
                "species": len(model_spec["species"]["symbols"]),
                "manifest_tensor_count": len(payloads),
                "state_inventory": str(args.out / "pet_state_inventory.json"),
                "runtime_status": model_spec["state_tensor_inventory"]["runtime_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
