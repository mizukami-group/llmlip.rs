#!/usr/bin/env python3
"""Export pinned MACE-MH-1 state tensors into an .e3rs-style directory.

This script is reference-only. It may import Python, PyTorch, e3nn, ASE, and
upstream MACE inside the pinned reference environment, but Rust runtime and
normal cargo tests must not depend on any of them.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from generate_mace_fixtures import hash_file, import_mace_artifact_reference, read_json, write_json
from inspect_mace_mh1_artifact import classify_state_tensor
from mace_mh1_artifact import MACE_COMMIT


SCHEMA_VERSION = "mace-mh1-state-dict-v1"
RUNTIME_STATUS = "state_export_only"
FAMILY = "MACE-MH-1"
DEFAULT_HEAD = "omat_pbe"
DOCUMENTED_HEADS = [
    "omat_pbe",
    "omol",
    "spice_wB97M",
    "oc20_usemppbe",
    "matpes_r2scan",
]


def toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_scalar(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML scalar value: {value!r}")


def clean_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if tensor.ndim == 0:
            return tensor.item()
        return tensor.tolist()
    if hasattr(value, "tolist"):
        return clean_value(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [clean_value(item) for item in value]
    return repr(value)


def storage_dtype_name(dtype: Any) -> str:
    aliases = {
        "torch.bool": "bool",
        "torch.float32": "f32",
        "torch.float64": "f64",
        "torch.int32": "i32",
        "torch.int64": "i64",
        "bool": "bool",
        "float32": "f32",
        "float64": "f64",
        "int32": "i32",
        "int64": "i64",
    }
    return aliases.get(str(dtype), str(dtype).removeprefix("torch."))


def tensor_to_export_array(tensor: Any, output_dtype: str) -> tuple[Any, str]:
    import numpy
    import torch

    detached = tensor.detach().cpu().contiguous()
    if detached.is_floating_point():
        detached = detached.to(dtype=torch.float64 if output_dtype == "f64" else torch.float32)
    elif detached.is_complex():
        raise RuntimeError(f"complex state_dict tensor is unsupported: {tensor.dtype}")
    array = detached.numpy()
    if array.dtype.byteorder == ">":
        array = array.byteswap().newbyteorder("<")
    elif array.dtype.byteorder == "=":
        array = array.astype(numpy.dtype(array.dtype).newbyteorder("<"), copy=False)
    return array, storage_dtype_name(detached.dtype)


def tensor_entry(name: str, tensor: Any, index: int, output_dtype: str, tensors_dir: Path) -> dict[str, Any]:
    array, stored_dtype = tensor_to_export_array(tensor, output_dtype)
    file_name = f"tensor_{index:05d}.bin"
    path = tensors_dir / file_name
    payload = array.tobytes(order="C")
    path.write_bytes(payload)
    return {
        "byte_size": len(payload),
        "dtype": stored_dtype,
        "layout": "row-major",
        "name": name,
        "numel": int(tensor.detach().numel()),
        "path": f"tensors/{file_name}",
        "sha256": hash_file(path),
        "shape": [int(dim) for dim in tensor.detach().shape],
        "source_dtype": storage_dtype_name(tensor.detach().dtype),
    }


def module_class_list(modules: Any) -> list[str]:
    return [type(module).__name__ for module in list(modules)]


def state_classification_counts(state_dict: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for name, tensor in state_dict.items():
        classification, _reason = classify_state_tensor(
            name,
            str(tensor.detach().dtype),
            int(tensor.detach().numel()),
        )
        counts[classification] += 1
    return dict(counts)


def tensor_shape(state_dict: dict[str, Any], name: str) -> list[int]:
    tensor = state_dict[name]
    return [int(dim) for dim in tensor.detach().shape]


def mace_type_symbols(ase: Any, atomic_numbers: list[int]) -> list[str]:
    return [ase.data.chemical_symbols[number] for number in atomic_numbers]


def patch_e3nn_spherical_harmonics(model: Any) -> None:
    try:
        from e3nn.o3._spherical_harmonics import _spherical_harmonics
    except Exception as exc:  # pragma: no cover - reference environment dependent
        raise SystemExit(f"failed to import e3nn spherical harmonics helper: {exc}") from exc
    for module in model.modules():
        if module.__class__.__name__ == "SphericalHarmonics" and not hasattr(module, "sph_func"):
            module.sph_func = _spherical_harmonics


def patch_e3nn_activation_paths(model: Any) -> None:
    for module in model.modules():
        if module.__class__.__name__ != "Activation" or hasattr(module, "paths"):
            continue
        if not hasattr(module, "irreps_in") or not hasattr(module, "acts"):
            raise SystemExit(
                "loaded e3nn Activation is missing paths and lacks irreps_in/acts "
                "needed to reconstruct it"
            )
        module.paths = [
            (mul, (ir.l, ir.p), act)
            for (mul, ir), act in zip(module.irreps_in, module.acts)
        ]


def write_e3rs_manifest(path: Path, *, state_json: dict[str, Any], tensors: list[dict[str, Any]]) -> None:
    lines = [
        'format_version = "0.1"',
        'architecture = "mace-mh1-v1"',
        'dtype = "f64"',
        'length_unit = "angstrom"',
        'energy_unit = "eV"',
        'force_unit = "eV/angstrom"',
        "",
        "[species]",
        f"symbols = {toml_scalar(state_json['type_names'])}",
        "",
        "[geometry]",
        f"cutoff = {state_json['cutoff']}",
        'edge_vector = "R_j + shift - R_i"',
        "directed_edges = true",
        "",
        "[spherical_harmonics]",
        f"l_max = {state_json['l_max']}",
        "normalize_input = true",
        'normalization = "component"',
        'reference_convention = "pinned-fixture"',
        "",
        "[validation]",
        'reference_package = "mace"',
        f"reference_version = {toml_scalar(state_json['reference']['package_version'])}",
        "fixture_seed = 20240520",
        "",
        "[mace_mh1]",
        'model = "mace_mh1_state.json"',
    ]
    for tensor in tensors:
        lines.extend(
            [
                "",
                "[[tensors]]",
                f"name = {toml_scalar(tensor['name'])}",
                f"dtype = {toml_scalar(tensor['dtype'])}",
                f"shape = {toml_scalar(tensor['shape'])}",
                f"layout = {toml_scalar(tensor['layout'])}",
                f"path = {toml_scalar(tensor['path'])}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_temp_out(out_dir: Path, force: bool) -> Path:
    if out_dir.exists() and not force:
        raise FileExistsError(f"out-dir already exists: {out_dir}; pass --force to replace it")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp.", dir=str(out_dir.parent)))


def replace_out_dir(temp_dir: Path, out_dir: Path, force: bool) -> None:
    if out_dir.exists():
        if not force:
            raise FileExistsError(f"out-dir already exists: {out_dir}; pass --force to replace it")
        if out_dir.is_dir() and not out_dir.is_symlink():
            shutil.rmtree(out_dir)
        else:
            out_dir.unlink()
    os.replace(temp_dir, out_dir)


def build_state_json(
    *,
    ase: Any,
    e3nn: Any,
    mace: Any,
    torch: Any,
    model: Any,
    manifest: dict[str, Any],
    tensors: list[dict[str, Any]],
) -> dict[str, Any]:
    state_dict = model.state_dict()
    observed_heads = list(getattr(model, "heads", []))
    missing_requested_heads = [head for head in DOCUMENTED_HEADS if head not in observed_heads]
    extra_observed_heads = [head for head in observed_heads if head not in DOCUMENTED_HEADS]
    atomic_numbers = [int(value) for value in model.atomic_numbers.detach().cpu().tolist()]
    counts = state_classification_counts(state_dict)
    scale_shift = getattr(model, "scale_shift", None)
    radial_embedding = getattr(model, "radial_embedding", None)
    return {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "runtime_status": RUNTIME_STATUS,
        "selected_model": manifest["selected_model"],
        "default_head": DEFAULT_HEAD,
        "heads": observed_heads,
        "requested_heads": DOCUMENTED_HEADS,
        "missing_requested_heads": missing_requested_heads,
        "extra_observed_heads": extra_observed_heads,
        "type_names": mace_type_symbols(ase, atomic_numbers),
        "atomic_numbers": atomic_numbers,
        "cutoff": float(model.r_max),
        "l_max": 3,
        "num_interactions": int(model.num_interactions),
        "artifact": {
            "source_key": manifest["selected_artifact"]["source_key"],
            "filename": manifest["selected_artifact"]["filename"],
            "url": manifest["selected_artifact"]["url"],
            "sha256": manifest["selected_artifact"]["sha256"],
            "size_bytes": manifest["selected_artifact"]["size_bytes"],
        },
        "reference": {
            "name": "mace",
            "repository": "https://github.com/ACEsuit/mace",
            "commit": manifest["reference"]["commit"],
            "package_version": getattr(mace, "__version__", "unknown"),
        },
        "state_dict": {
            "tensor_count": len(tensors),
            "runtime_tensor_count": counts.get("runtime_tensor_candidate", 0),
            "runtime_control_tensor_count": counts.get("runtime_control_tensor", 0),
            "deterministic_reconstructed_tensor_count": counts.get("deterministic_reconstructed", 0),
            "empty_tensor_count": counts.get("empty_tensor", 0),
            "metadata_tensor_count": counts.get("metadata", 0),
        },
        "head_tensor_shapes": {
            "atomic_energies": tensor_shape(state_dict, "atomic_energies_fn.atomic_energies"),
            "scale_shift_scale": tensor_shape(state_dict, "scale_shift.scale"),
            "scale_shift_shift": tensor_shape(state_dict, "scale_shift.shift"),
            "readout_output_masks": [
                tensor_shape(state_dict, "readouts.0.linear.output_mask"),
                tensor_shape(state_dict, "readouts.1.linear_1.output_mask"),
                tensor_shape(state_dict, "readouts.1.linear_2.output_mask"),
            ],
        },
        "execution_plan": {
            "interaction_classes": module_class_list(getattr(model, "interactions", [])),
            "product_classes": module_class_list(getattr(model, "products", [])),
            "readout_classes": module_class_list(getattr(model, "readouts", [])),
            "radial_embedding_class": type(radial_embedding).__name__,
            "radial_distance_transform_class": type(radial_embedding.distance_transform).__name__,
            "radial_cutoff_class": type(radial_embedding.cutoff_fn).__name__,
            "radial_bessel_class": type(radial_embedding.bessel_fn).__name__,
            "pair_repulsion_class": type(model.pair_repulsion_fn).__name__,
            "scale_shift_class": type(scale_shift).__name__,
        },
        "provenance": {
            "normal_cargo_test_dependency": False,
            "python_executable": Path(sys.executable).name,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "e3nn_version": getattr(e3nn, "__version__", "unknown"),
        },
        "tensors": tensors,
    }


def export(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.manifest)
    actual_sha = hash_file(args.artifact)
    expected_sha = manifest["selected_artifact"]["sha256"]
    if actual_sha != expected_sha:
        raise RuntimeError(f"artifact SHA256 mismatch: expected {expected_sha}, got {actual_sha}")
    if manifest["reference"]["commit"] != MACE_COMMIT:
        raise RuntimeError(
            f"manifest reference commit must be {MACE_COMMIT}, got {manifest['reference']['commit']}"
        )

    ase, e3nn, mace, _np, torch, _torch_functional, _o3, _refs, model = (
        import_mace_artifact_reference(args, args.artifact)
    )
    patch_e3nn_spherical_harmonics(model)
    patch_e3nn_activation_paths(model)
    temp_dir = prepare_temp_out(args.out_dir.resolve(), args.force)
    try:
        tensors_dir = temp_dir / "tensors"
        tensors_dir.mkdir(parents=True)
        tensors = [
            tensor_entry(name, tensor, index, args.dtype, tensors_dir)
            for index, (name, tensor) in enumerate(model.state_dict().items())
        ]
        state_json = build_state_json(
            ase=ase,
            e3nn=e3nn,
            mace=mace,
            torch=torch,
            model=model,
            manifest=manifest,
            tensors=tensors,
        )
        write_e3rs_manifest(temp_dir / "manifest.toml", state_json=state_json, tensors=tensors)
        write_json(temp_dir / "mace_mh1_state.json", state_json)
        report = {
            "schema_version": "mace-mh1-state-export-report-v1",
            "status": "exported",
            "out_dir": str(args.out_dir),
            "artifact_sha256": actual_sha,
            "dtype": args.dtype,
            "tensor_count": len(tensors),
            "dtype_counts": dict(Counter(tensor["dtype"] for tensor in tensors)),
            "source_dtype_counts": dict(Counter(tensor["source_dtype"] for tensor in tensors)),
            "selected_model": manifest["selected_model"],
            "heads": state_json["heads"],
            "missing_requested_heads": state_json["missing_requested_heads"],
            "normal_cargo_test_dependency": False,
        }
        write_json(temp_dir / "export_report.json", report)
        replace_out_dir(temp_dir, args.out_dir.resolve(), args.force)
        return report
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mace-dir", type=Path, default=Path("third_party/mace"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("fixtures/mace/v0_3_16_4d2da09/mace_mh1_artifact_manifest.json"),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("outputs/reference-artifacts/mace/mace-mh1/github-release/mace-mh-1.model"),
    )
    parser.add_argument("--e3nn-compat-path", type=Path)
    parser.add_argument("--dtype", choices=["f64", "f32"], default="f64")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/reference-artifacts/mace/mace-mh1/github-release/mace_mh1_state.e3rs"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    report = export(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
