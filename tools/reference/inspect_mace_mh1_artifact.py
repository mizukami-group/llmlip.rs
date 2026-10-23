#!/usr/bin/env python3
"""Inspect a pinned MACE-MH-1 artifact without making it a Rust dependency."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

from generate_mace_fixtures import hash_file, import_mace_artifact_reference, read_json, write_json


SCHEMA_VERSION = "mace-mh1-inspection-v1"


def classify_state_tensor(name: str, dtype: str, numel: int | None) -> tuple[str, str]:
    if numel == 0:
        return "empty_tensor", "zero-size upstream tensor; validate as empty but do not serialize as runtime weight"
    if name in {"atomic_numbers", "r_max", "num_interactions"}:
        return "metadata", "model identity/shape metadata; serialize in manifest or plan, not as a learned weight"
    if "._compiled_" in name or "._w3j_" in name:
        return "deterministic_reconstructed", "compiled tensor-product Wigner buffer; represent through explicit TP plans"
    if name.endswith(".output_mask"):
        return "runtime_control_tensor", "mask controls compact irreps layout and must be serialized or reconstructed explicitly"
    if name.endswith("_zeroed"):
        return "runtime_control_tensor", "boolean contraction-control flag; must be serialized or validated explicitly"
    if dtype in {"torch.int32", "torch.int64", "torch.bool"}:
        return "runtime_control_tensor", "non-floating control tensor required for exact runtime behavior"
    return "runtime_tensor_candidate", "floating tensor candidate for manifest-backed runtime serialization"


def clean_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if tensor.ndim == 0:
            return tensor.item()
        return {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "values": tensor.tolist() if tensor.numel() <= 16 else None,
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [clean_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): clean_value(item) for key, item in value.items()}
    return repr(value)


def tensor_entry(name: str, tensor: Any) -> dict[str, Any]:
    detached = tensor.detach().cpu() if hasattr(tensor, "detach") else tensor
    shape = list(detached.shape) if hasattr(detached, "shape") else []
    numel = int(detached.numel()) if hasattr(detached, "numel") else None
    dtype = str(getattr(detached, "dtype", "unknown"))
    classification, classification_reason = classify_state_tensor(name, dtype, numel)
    return {
        "name": name,
        "dtype": dtype,
        "shape": shape,
        "numel": numel,
        "classification": classification,
        "classification_reason": classification_reason,
    }


def summarize_state_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, dict[str, Any]] = {}
    by_dtype: dict[str, int] = {}
    for entry in entries:
        classification = entry["classification"]
        class_summary = by_class.setdefault(
            classification,
            {"tensor_count": 0, "numel": 0, "examples": []},
        )
        class_summary["tensor_count"] += 1
        class_summary["numel"] += int(entry["numel"] or 0)
        if len(class_summary["examples"]) < 8:
            class_summary["examples"].append(entry["name"])
        by_dtype[entry["dtype"]] = by_dtype.get(entry["dtype"], 0) + 1
    return {
        "by_classification": dict(sorted(by_class.items())),
        "by_dtype": dict(sorted(by_dtype.items())),
        "runtime_tensor_candidate_count": by_class.get(
            "runtime_tensor_candidate", {}
        ).get("tensor_count", 0),
        "runtime_control_tensor_count": by_class.get("runtime_control_tensor", {}).get(
            "tensor_count", 0
        ),
        "archive_or_metadata_count": sum(
            by_class.get(name, {}).get("tensor_count", 0)
            for name in ("deterministic_reconstructed", "empty_tensor", "metadata")
        ),
    }


def module_class_list(modules: Any) -> list[str]:
    return [type(module).__name__ for module in list(modules)]


def maybe_module_class(module: Any) -> str | None:
    if module is None:
        return None
    return type(module).__name__


def build_inspection(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.manifest)
    expected_sha = manifest["selected_artifact"]["sha256"]
    actual_sha = hash_file(args.artifact)
    if actual_sha != expected_sha:
        raise SystemExit(
            "artifact SHA256 does not match manifest selected_artifact\n"
            f"expected: {expected_sha}\n"
            f"actual:   {actual_sha}"
        )

    (
        _ase,
        e3nn,
        mace,
        _np,
        torch,
        _torch_functional,
        _o3,
        _refs,
        model,
    ) = import_mace_artifact_reference(args, args.artifact)
    state_dict = model.state_dict()
    observed_heads = list(getattr(model, "heads", []))
    requested_heads = list(manifest.get("requested_heads", manifest.get("heads", [])))
    missing_requested_heads = [head for head in requested_heads if head not in observed_heads]
    extra_observed_heads = [head for head in observed_heads if head not in requested_heads]

    interaction_classes = module_class_list(getattr(model, "interactions", []))
    product_classes = module_class_list(getattr(model, "products", []))
    readout_classes = module_class_list(getattr(model, "readouts", []))
    nonlinear_interaction_blocks = [
        cls for cls in interaction_classes if "NonLinear" in cls or "Nonlinear" in cls
    ]

    radial_embedding = getattr(model, "radial_embedding", None)
    distance_transform = getattr(radial_embedding, "distance_transform", None)
    cutoff_fn = getattr(radial_embedding, "cutoff_fn", None)
    bessel_fn = getattr(radial_embedding, "bessel_fn", None)
    scale_shift = getattr(model, "scale_shift", None)

    state_entries = [tensor_entry(name, tensor) for name, tensor in state_dict.items()]

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": {
            "source_key": manifest["selected_artifact"]["source_key"],
            "filename": manifest["selected_artifact"]["filename"],
            "url": manifest["selected_artifact"]["url"],
            "sha256": actual_sha,
            "size_bytes": args.artifact.stat().st_size,
        },
        "reference": {
            "name": "mace",
            "repository": "https://github.com/ACEsuit/mace",
            "commit": manifest["reference"]["commit"],
            "package_version": getattr(mace, "__version__", "unknown"),
        },
        "python": {
            "python_version": platform.python_version(),
            "executable": Path(sys.executable).name,
            "torch_version": torch.__version__,
            "e3nn_version": getattr(e3nn, "__version__", "unknown"),
        },
        "model": {
            "class": type(model).__name__,
            "heads": observed_heads,
            "requested_heads": requested_heads,
            "missing_requested_heads": missing_requested_heads,
            "extra_observed_heads": extra_observed_heads,
            "r_max": clean_value(getattr(model, "r_max", None)),
            "num_interactions": int(getattr(model, "num_interactions", len(interaction_classes))),
            "atomic_numbers_count": int(len(getattr(model, "atomic_numbers", []))),
            "atomic_numbers": clean_value(getattr(model, "atomic_numbers", None)),
            "interaction_classes": interaction_classes,
            "product_classes": product_classes,
            "readout_classes": readout_classes,
            "radial_embedding_class": maybe_module_class(radial_embedding),
            "radial_distance_transform_class": maybe_module_class(distance_transform),
            "radial_cutoff_class": maybe_module_class(cutoff_fn),
            "radial_bessel_class": maybe_module_class(bessel_fn),
            "pair_repulsion_class": maybe_module_class(getattr(model, "pair_repulsion_fn", None)),
            "scale_shift_class": maybe_module_class(scale_shift),
            "scale_shift_scale": clean_value(getattr(scale_shift, "scale", None)),
            "scale_shift_shift": clean_value(getattr(scale_shift, "shift", None)),
        },
        "state_dict": {
            "tensor_count": len(state_dict),
            "classification_summary": summarize_state_entries(state_entries),
            "entries": state_entries,
        },
        "gap_analysis": {
            "current_mace_lite_single_head_only": observed_heads != ["Default"],
            "requires_head_selection_api": len(observed_heads) > 1,
            "requires_nonlinear_interaction_block_support": bool(nonlinear_interaction_blocks),
            "nonlinear_interaction_blocks": nonlinear_interaction_blocks,
            "requires_source_target_embedding_support": any(
                "source_embedding" in entry["name"] or "target_embedding" in entry["name"]
                for entry in state_entries
            ),
            "requires_equivariant_nonlinearity_support": any(
                "equivariant_nonlin" in entry["name"] for entry in state_entries
            ),
            "requires_multihead_atomic_energies": any(
                entry["name"] == "atomic_energies_fn.atomic_energies"
                and len(entry["shape"]) == 2
                and entry["shape"][0] > 1
                for entry in state_entries
            ),
            "requested_head_mismatch_blocks_full_user_scope": bool(missing_requested_heads),
            "notes": [
                (
                    "The Rust runtime must reject unknown heads explicitly. "
                    "Upstream MACECalculator currently warns and falls back for "
                    "some missing head names."
                ),
                (
                    "MACE-MH-1 cannot claim support for requested heads absent "
                    "from the serialized model without an official artifact or "
                    "documented alias."
                ),
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("fixtures/mace/current/mace_mh1_inspection.json"),
    )
    args = parser.parse_args()

    inspection = build_inspection(args)
    write_json(args.out, inspection)
    print(json.dumps(inspection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
