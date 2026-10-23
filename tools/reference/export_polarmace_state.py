#!/usr/bin/env python3
"""Export a pinned PolarMACE artifact state_dict into an .e3rs-style directory.

This script is reference-only. It may import Python, PyTorch, MACE, e3nn, and
optional PolarMACE dependencies in the reference Python environment, but Rust
runtime and normal cargo tests must not depend on any of them.

The trust boundary is the pinned artifact manifest plus SHA verification. After
that verification, the script loads the upstream pickled artifact and exports
only raw state_dict tensor payloads and metadata for later Rust-side inspection
or exporter development. It does not claim PolarMACE runtime support.
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

from polarmace_artifact import hash_file, validate_mace_tree


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clean_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [clean_value(item) for item in value]
    if hasattr(value, "tolist"):
        return clean_value(value.tolist())
    return str(value)


def toml_quote(value: str) -> str:
    return json.dumps(value)


def toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return toml_quote(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_scalar(item) for item in value) + "]"
    if value is None:
        return toml_quote("null")
    raise TypeError(f"unsupported TOML scalar value: {value!r}")


def write_manifest_toml(path: Path, manifest: dict[str, Any], tensors: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    for key in [
        "format_version",
        "architecture",
        "schema_version",
        "dtype",
        "runtime_status",
        "selected_model",
        "artifact_sha256",
        "reference_commit",
        "normal_cargo_test_dependency",
    ]:
        lines.append(f"{key} = {toml_scalar(manifest[key])}")
    lines.append("")
    lines.append("[provenance]")
    for key, value in manifest["provenance"].items():
        lines.append(f"{key} = {toml_scalar(value)}")
    lines.append("")
    lines.append("[state_dict]")
    for key, value in manifest["state_dict"].items():
        if isinstance(value, dict):
            continue
        lines.append(f"{key} = {toml_scalar(value)}")
    for key, value in manifest["state_dict"].items():
        if not isinstance(value, dict):
            continue
        lines.append("")
        lines.append(f"[state_dict.{key}]")
        for nested_key, nested_value in value.items():
            lines.append(f"{nested_key} = {toml_scalar(nested_value)}")
    for tensor in tensors:
        lines.append("")
        lines.append("[[tensors]]")
        for key in [
            "name",
            "path",
            "dtype",
            "source_dtype",
            "shape",
            "numel",
            "byte_size",
            "layout",
            "sha256",
        ]:
            lines.append(f"{key} = {toml_scalar(tensor[key])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def polar_type_symbols() -> list[str]:
    return [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
        "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",
        "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
        "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
        "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
        "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    ]


def write_e3rs_manifest_toml(path: Path, manifest: dict[str, Any], tensors: list[dict[str, Any]]) -> None:
    if manifest["dtype"] != "f64":
        raise RuntimeError("normal mace-polar-v1 .e3rs manifest export currently requires --dtype f64")
    lines = [
        'format_version = "0.1"',
        'architecture = "mace-polar-v1"',
        'dtype = "f64"',
        'length_unit = "angstrom"',
        'energy_unit = "eV"',
        'force_unit = "eV/angstrom"',
        "",
        "[species]",
        f"symbols = {toml_scalar(polar_type_symbols())}",
        "",
        "[geometry]",
        "cutoff = 6.0",
        'edge_vector = "R_j + shift - R_i"',
        "directed_edges = true",
        "",
        "[spherical_harmonics]",
        "l_max = 1",
        "normalize_input = true",
        'normalization = "component"',
        'reference_convention = "pinned-fixture"',
        "",
        "[validation]",
        'reference_package = "mace"',
        f"reference_version = {toml_scalar(manifest['provenance']['mace_package_version'])}",
        "fixture_seed = 20240518",
        "",
        "[mace_polar]",
        'model = "mace_polar_state.json"',
        'reference_package = "mace"',
        f"reference_version = {toml_scalar(manifest['provenance']['mace_package_version'])}",
        f"reference_commit = {toml_scalar(manifest['reference_commit'])}",
        f"selected_model = {toml_scalar(manifest['selected_model'])}",
    ]
    for tensor in tensors:
        lines.append("")
        lines.append("[[tensors]]")
        for key in ["name", "path", "dtype", "shape", "layout"]:
            lines.append(f"{key} = {toml_scalar(tensor[key])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def validate_artifact_manifest(manifest: dict[str, Any], artifact_path: Path) -> str:
    if not artifact_path.exists():
        raise FileNotFoundError(f"artifact does not exist: {artifact_path}")
    actual_sha256 = hash_file(artifact_path)
    expected_sha256 = manifest.get("expected_sha256")
    recorded_sha256 = manifest.get("sha256")
    if expected_sha256 and recorded_sha256 and expected_sha256 != recorded_sha256:
        raise RuntimeError(
            "artifact manifest has conflicting "
            f"expected_sha256={expected_sha256} and sha256={recorded_sha256}"
        )
    expected_sha256 = expected_sha256 or recorded_sha256
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{artifact_path} SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    if not expected_sha256:
        raise RuntimeError("artifact manifest must record expected_sha256 or sha256")
    return actual_sha256


def torch_dtype_name(dtype: Any) -> str:
    text = str(dtype)
    return text.removeprefix("torch.")


def storage_dtype_name(dtype: Any) -> str:
    name = torch_dtype_name(dtype)
    aliases = {
        "float16": "f16",
        "float32": "f32",
        "float64": "f64",
        "bfloat16": "bf16",
        "bool": "bool",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "uint8": "u8",
    }
    return aliases.get(name, name)


def tensor_to_export_array(tensor: Any, output_dtype: str) -> tuple[Any, str]:
    import numpy
    import torch

    detached = tensor.detach().cpu().contiguous()
    if detached.is_floating_point():
        torch_dtype = torch.float64 if output_dtype == "f64" else torch.float32
        detached = detached.to(dtype=torch_dtype)
    elif detached.is_complex():
        raise RuntimeError(f"complex state_dict tensor is unsupported: {tensor.dtype}")
    array = detached.numpy()
    if array.dtype.byteorder == ">":
        array = array.byteswap().newbyteorder("<")
    elif array.dtype.byteorder == "=":
        little = numpy.dtype(array.dtype).newbyteorder("<")
        array = array.astype(little, copy=False)
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


def exported_tensor_ref(tensor_by_name: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    entry = tensor_by_name.get(name)
    if entry is None:
        return {
            "name": name,
            "status": "missing",
            "blocker": f"state export has no tensor named {name!r}",
        }
    return {
        "name": name,
        "status": "exported",
        "path": entry["path"],
        "dtype": entry["dtype"],
        "source_dtype": entry["source_dtype"],
        "shape": entry["shape"],
        "numel": entry["numel"],
        "layout": entry["layout"],
        "sha256": entry["sha256"],
    }


def linear_runtime_metadata(
    module: Any,
    *,
    state_prefix: str,
    tensor_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from generate_polarmace_manual_backbone_replay import e3nn_linear_metadata

    return {
        **e3nn_linear_metadata(module),
        "state_tensor_refs": {
            name: exported_tensor_ref(tensor_by_name, f"{state_prefix}.{name}")
            for name in ["weight", "bias", "output_mask"]
        },
    }


def gate_runtime_metadata(
    gate: Any,
    *,
    state_prefix: str,
    tensor_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from generate_polarmace_manual_backbone_replay import (
        activation_metadata,
        clean_value as replay_clean_value,
        irreps_partitions,
        tensor_product_instructions,
    )

    sc = getattr(gate, "sc", None)
    sc_irreps_outs = list(getattr(sc, "irreps_outs", [])) if sc is not None else []
    sc_instructions = replay_clean_value(
        getattr(getattr(sc, "cut", None), "instructions", None)
    )

    input_names = ["scalars", "gates", "gated"]
    input_partitions = []
    offset = 0
    for index, irreps in enumerate(sc_irreps_outs):
        dim = int(irreps.dim)
        input_partitions.append(
            {
                "name": input_names[index] if index < len(input_names) else str(index),
                "irreps": str(irreps),
                "start": offset,
                "end": offset + dim,
                "dim": dim,
                "extract_instruction": (
                    sc_instructions[index]
                    if isinstance(sc_instructions, list)
                    and index < len(sc_instructions)
                    else None
                ),
                "subpartitions": irreps_partitions(irreps, start=offset),
            }
        )
        offset += dim

    scalar_dim = int(getattr(gate, "irreps_scalars").dim)
    gated_dim = int(getattr(gate, "irreps_gated").dim)
    mul = getattr(gate, "mul")
    return {
        "class_name": f"{type(gate).__module__}.{type(gate).__name__}",
        "irreps_scalars": str(getattr(gate, "irreps_scalars", "unknown")),
        "irreps_gates": str(getattr(gate, "irreps_gates", "unknown")),
        "irreps_gated": str(getattr(gate, "irreps_gated", "unknown")),
        "irreps_in": str(getattr(gate, "irreps_in", "unknown")),
        "irreps_out": str(getattr(gate, "irreps_out", "unknown")),
        "layout": {
            "input_dim": int(getattr(gate, "irreps_in").dim),
            "output_dim": scalar_dim + gated_dim,
            "input_partition_source": "gate.sc _Sortcut output order: scalars, gates, gated.",
            "input_layout": "e3nn contiguous irreps direct-sum layout from gate.irreps_in.",
            "output_layout": "torch.cat([act_scalars(scalars), mul(gated, act_gates(gates))], dim=-1).",
            "mul_input_order": "gated x activated_gates",
        },
        "input_partitions": input_partitions,
        "output_partitions": [
            {
                "name": "scalar_activation",
                "irreps": str(getattr(gate, "irreps_scalars", "unknown")),
                "start": 0,
                "end": scalar_dim,
                "dim": scalar_dim,
                "subpartitions": irreps_partitions(getattr(gate, "irreps_scalars")),
            },
            {
                "name": "gated_product",
                "irreps": str(getattr(gate, "irreps_gated", "unknown")),
                "start": scalar_dim,
                "end": scalar_dim + gated_dim,
                "dim": gated_dim,
                "subpartitions": irreps_partitions(
                    getattr(gate, "irreps_gated"), start=scalar_dim
                ),
            },
        ],
        "scalar_activation": "silu",
        "gate_activation": "sigmoid",
        "activation_modules": {
            "scalar": activation_metadata(getattr(gate, "act_scalars")),
            "gate": activation_metadata(getattr(gate, "act_gates")),
        },
        "normalize2mom": {
            "scalar_activation": [
                act["normalize2mom_constant"]
                for act in activation_metadata(getattr(gate, "act_scalars"))["acts"]
            ],
            "gate_activation": [
                act["normalize2mom_constant"]
                for act in activation_metadata(getattr(gate, "act_gates"))["acts"]
            ],
        },
        "mul": {
            "class_name": f"{type(mul).__module__}.{type(mul).__name__}",
            "irreps_in1": str(getattr(mul, "irreps_in1", "unknown")),
            "irreps_in2": str(getattr(mul, "irreps_in2", "unknown")),
            "irreps_out": str(getattr(mul, "irreps_out", "unknown")),
            "weight_numel": int(getattr(mul, "weight_numel", -1)),
            "instructions": tensor_product_instructions(mul),
            "state_tensor_refs": {
                name: exported_tensor_ref(tensor_by_name, f"{state_prefix}.mul.{name}")
                for name in ["weight", "output_mask"]
            },
        },
    }


def symmetric_contraction_scalar_runtime_metadata(
    module: Any,
    *,
    state_prefix: str,
    tensor_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from generate_polarmace_manual_backbone_replay import optional_module_attr_or_blocker

    contractions = list(getattr(module, "contractions", []))
    if not contractions:
        return {
            "status": "blocked",
            "blocker": "symmetric_contractions module has no contractions",
        }

    contraction = contractions[0]
    state = contraction.state_dict()
    num_elements = int(state["weights.1"].shape[0]) if "weights.1" in state else None
    num_features = getattr(contraction, "num_features", None)
    if num_features is not None:
        num_features = int(num_features)
    num_ell = int(state["U_matrix_1"].shape[0]) if "U_matrix_1" in state else None
    tensor_names = [
        "U_matrix_1",
        "U_matrix_2",
        "U_matrix_3",
        "weights.0",
        "weights.1",
        "weights_max",
        "weights_0_zeroed",
        "weights_1_zeroed",
        "weights_max_zeroed",
    ]

    return {
        "status": "exported",
        "kind": "scalar",
        "index": 0,
        "module_class": f"{type(module).__module__}.{type(module).__name__}",
        "contraction_class": f"{type(contraction).__module__}.{type(contraction).__name__}",
        "layout": {
            "input": "[node, feature, ell]",
            "output": "[node, feature]",
            "effective_node_attrs": "[node, element]",
        },
        "num_elements": num_elements,
        "num_features": num_features,
        "num_ell": num_ell,
        "correlation": clean_value(getattr(contraction, "correlation", None)),
        "irreps_in": str(getattr(module, "irreps_in", "unknown")),
        "irreps_out": str(getattr(module, "irreps_out", "unknown")),
        "coupling_irreps": clean_value(
            [str(irrep) for irrep in getattr(contraction, "coupling_irreps", [])]
        ),
        "normalization": {
            "irrep_normalization": optional_module_attr_or_blocker(
                contraction, "irrep_normalization"
            ),
            "path_normalization": optional_module_attr_or_blocker(
                contraction, "path_normalization"
            ),
            "use_reduced_cg": optional_module_attr_or_blocker(
                contraction, "use_reduced_cg"
            ),
        },
        "semantic_weight_mapping": {
            "weights.0": "weights_u2",
            "weights.1": "weights_u1",
            "weights_max": "weights_u3",
        },
        "state_tensor_roles": {
            "U_matrix_1": "correlation_1_U",
            "U_matrix_2": "correlation_2_U",
            "U_matrix_3": "correlation_3_U",
            "weights.0": "weights_u2",
            "weights.1": "weights_u1",
            "weights_max": "weights_u3",
            "weights_0_zeroed": "weights_u2_zero_flag",
            "weights_1_zeroed": "weights_u1_zero_flag",
            "weights_max_zeroed": "weights_u3_zero_flag",
        },
        "state_tensor_refs": {
            name: exported_tensor_ref(tensor_by_name, f"{state_prefix}.{name}")
            for name in tensor_names
        },
    }


def scale_shift_runtime_metadata(
    scale_shift: Any,
    *,
    heads: list[Any],
    tensor_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "class_name": f"{type(scale_shift).__module__}.{type(scale_shift).__name__}",
        "heads": clean_value(heads),
        "scale_attr_present": getattr(scale_shift, "scale", None) is not None,
        "shift_attr_present": getattr(scale_shift, "shift", None) is not None,
        "state_tensor_refs": {
            name: exported_tensor_ref(tensor_by_name, f"scale_shift.{name}")
            for name in ["scale", "shift"]
        },
    }


def runtime_execution_plan_metadata(
    model: Any,
    tensors: list[dict[str, Any]],
    np_module: Any,
    torch_module: Any,
) -> dict[str, Any]:
    from generate_polarmace_manual_backbone_replay import tensor_product_plan_pin

    tensor_by_name = {entry["name"]: entry for entry in tensors}
    blockers = [
        {
            "component": "polar_long_range_tail",
            "status": "not_exported",
            "blocker": (
                "This metadata block covers only the local MACE backbone. "
                "PolarMACE graph_longrange density/electrostatics, field-dependent "
                "charge recursion, GTO caches, and electrostatic force VJP still "
                "require separate fixture-pinned execution plans."
            ),
        }
    ]
    layers = []
    interactions = list(getattr(model, "interactions", []))
    products = list(getattr(model, "products", []))
    readouts = list(getattr(model, "readouts", []))
    for layer, (interaction, product) in enumerate(zip(interactions, products)):
        readout_index = layer if len(readouts) > 1 else 0
        readout = readouts[readout_index] if readouts else None
        layer_entry: dict[str, Any] = {
            "layer": layer,
            "interaction_class": f"{type(interaction).__module__}.{type(interaction).__name__}",
            "product_class": f"{type(product).__module__}.{type(product).__name__}",
            "interaction": {},
            "product": {},
        }
        if hasattr(interaction, "conv_tp"):
            layer_entry["interaction"]["conv_tp_plan"] = tensor_product_plan_pin(
                interaction.conv_tp, np_module, torch_module
            )
            layer_entry["interaction"]["conv_tp_state_tensor_refs"] = {
                name: exported_tensor_ref(
                    tensor_by_name, f"interactions.{layer}.conv_tp.{name}"
                )
                for name in ["weight", "output_mask"]
            }
        else:
            layer_entry["interaction"]["conv_tp_plan"] = {
                "status": "blocked",
                "blocker": f"interactions.{layer} has no conv_tp attribute",
            }
        if hasattr(interaction, "equivariant_nonlin"):
            layer_entry["interaction"]["equivariant_nonlin"] = {
                "gate_plan": gate_runtime_metadata(
                    interaction.equivariant_nonlin,
                    state_prefix=f"interactions.{layer}.equivariant_nonlin",
                    tensor_by_name=tensor_by_name,
                )
            }
        else:
            layer_entry["interaction"]["equivariant_nonlin"] = {
                "status": "not_applicable",
                "blocker": f"interactions.{layer} has no equivariant_nonlin attribute",
            }
        if hasattr(product, "symmetric_contractions"):
            layer_entry["product"]["scalar_symmetric_contraction_plan"] = (
                symmetric_contraction_scalar_runtime_metadata(
                    product.symmetric_contractions,
                    state_prefix=f"products.{layer}.symmetric_contractions.contractions.0",
                    tensor_by_name=tensor_by_name,
                )
            )
        if hasattr(product, "linear"):
            layer_entry["product"]["linear"] = linear_runtime_metadata(
                product.linear,
                state_prefix=f"products.{layer}.linear",
                tensor_by_name=tensor_by_name,
            )
        if readout is not None:
            layer_entry["readout"] = {
                "readout_index": readout_index,
                "readout_class": f"{type(readout).__module__}.{type(readout).__name__}",
            }
            if hasattr(readout, "linear"):
                layer_entry["readout"]["linear"] = linear_runtime_metadata(
                    readout.linear,
                    state_prefix=f"readouts.{readout_index}.linear",
                    tensor_by_name=tensor_by_name,
                )
        layers.append(layer_entry)

    return {
        "schema_version": "polarmace-runtime-execution-plan-metadata-v1",
        "runtime_status": "metadata_only_not_executable",
        "description": (
            "Reference-only non-tensor execution metadata for the local "
            "PolarMACE MACE backbone. It is intended to let Rust inspect plans "
            "without importing Python, but it does not make mace-polar-v1 executable."
        ),
        "non_tensor_metadata_only": True,
        "layer_count": len(layers),
        "layers": layers,
        "scale_shift": (
            scale_shift_runtime_metadata(
                model.scale_shift,
                heads=list(getattr(model, "heads", [])),
                tensor_by_name=tensor_by_name,
            )
            if hasattr(model, "scale_shift")
            else {
                "status": "not_applicable",
                "blocker": "model has no scale_shift attribute",
            }
        ),
        "blockers": blockers,
    }


def load_trusted_model(artifact_path: Path) -> Any:
    import torch

    # Trust boundary: the artifact SHA was verified before this point.
    # `weights_only=False` is required because upstream MACE artifacts are full
    # pickled model objects, not plain state dictionaries.
    try:
        return torch.load(artifact_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(artifact_path, map_location="cpu")


def prepare_temp_out(out_dir: Path, force: bool) -> Path:
    if out_dir.exists() and not force:
        raise FileExistsError(f"out-dir already exists: {out_dir}; pass --force to replace it")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{out_dir.name}.tmp.",
            dir=str(out_dir.parent),
        )
    )


def replace_out_dir(temp_dir: Path, out_dir: Path, force: bool) -> None:
    if out_dir.exists():
        if not force:
            raise FileExistsError(f"out-dir already exists: {out_dir}; pass --force to replace it")
        if out_dir.is_dir() and not out_dir.is_symlink():
            shutil.rmtree(out_dir)
        else:
            out_dir.unlink()
    os.replace(temp_dir, out_dir)


def export(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    manifest_path = resolve_under_root(root, args.artifact_manifest)
    artifact_manifest = read_json(manifest_path)
    artifact_path = resolve_under_root(root, Path(artifact_manifest["artifact_path"]))
    artifact_sha256 = validate_artifact_manifest(artifact_manifest, artifact_path)

    mace_dir = resolve_under_root(root, args.mace_dir).resolve()
    reference_commit = validate_mace_tree(mace_dir)
    manifest_commit = artifact_manifest.get("mace_commit")
    if manifest_commit and manifest_commit != reference_commit:
        raise RuntimeError(
            f"artifact manifest mace_commit={manifest_commit} does not match pinned checkout {reference_commit}"
        )

    if args.e3nn_compat_path is not None:
        sys.path.insert(0, str(resolve_under_root(root, args.e3nn_compat_path).resolve()))
    sys.path.insert(0, str(mace_dir))

    import numpy
    import torch

    model = load_trusted_model(artifact_path)
    state_dict = model.state_dict()
    temp_dir = prepare_temp_out(args.out_dir.resolve(), args.force)
    try:
        tensors_dir = temp_dir / "tensors"
        tensors_dir.mkdir(parents=True)
        tensors: list[dict[str, Any]] = []
        for index, (name, tensor) in enumerate(state_dict.items()):
            if not hasattr(tensor, "detach"):
                raise RuntimeError(f"state_dict entry is not a tensor: {name}")
            tensors.append(tensor_entry(name, tensor, index, args.dtype, tensors_dir))

        dtype_counts = dict(Counter(entry["dtype"] for entry in tensors))
        source_dtype_counts = dict(Counter(entry["source_dtype"] for entry in tensors))
        total_numel = sum(entry["numel"] for entry in tensors)
        total_bytes = sum(entry["byte_size"] for entry in tensors)
        selected_model = artifact_manifest.get("selected_model")
        runtime_plan = runtime_execution_plan_metadata(model, tensors, numpy, torch)

        provenance = {
            "artifact_manifest_path": str(manifest_path.relative_to(root)),
            "artifact_path": str(artifact_path.relative_to(root)),
            "artifact_schema_version": clean_value(artifact_manifest.get("schema_version")),
            "mace_commit": reference_commit,
            "mace_package_version": clean_value(artifact_manifest.get("mace_package_version")),
            "numpy_version": numpy.__version__,
            "python_executable": sys.executable,
            "python_platform": platform.platform(),
            "python_version": sys.version,
            "source_url": clean_value(artifact_manifest.get("source_url")),
            "torch_version": torch.__version__,
        }
        manifest = {
            "architecture": "mace-polar-v1",
            "artifact_sha256": artifact_sha256,
            "dtype": args.dtype,
            "format_version": "0.1",
            "normal_cargo_test_dependency": False,
            "provenance": provenance,
            "reference_commit": reference_commit,
            "runtime_status": "state_export_only",
            "schema_version": "polarmace-state-export-v1",
            "selected_model": clean_value(selected_model),
            "state_dict": {
                "dtype_counts": dtype_counts,
                "numel": total_numel,
                "source_dtype_counts": source_dtype_counts,
                "tensor_count": len(tensors),
                "total_bytes": total_bytes,
            },
        }
        state_json = {
            "artifact": artifact_manifest,
            "artifact_sha256_verified": artifact_sha256,
            "model": {
                "class_name": model.__class__.__name__,
                "module": model.__class__.__module__,
            },
            "normal_cargo_test_dependency": False,
            "provenance": provenance,
            "runtime_status": "state_export_only",
            "runtime_execution_plan": runtime_plan,
            "schema_version": "polarmace-state-dict-v1",
            "selected_model": clean_value(selected_model),
            "state_dict": manifest["state_dict"],
            "tensors": tensors,
        }
        export_report = {
            "artifact_sha256": artifact_sha256,
            "dtype": args.dtype,
            "normal_cargo_test_dependency": False,
            "out_dir": str(args.out_dir),
            "reference_commit": reference_commit,
            "runtime_status": "state_export_only",
            "runtime_execution_plan": {
                "schema_version": runtime_plan["schema_version"],
                "runtime_status": runtime_plan["runtime_status"],
                "layer_count": runtime_plan["layer_count"],
                "blocker_count": len(runtime_plan["blockers"]),
            },
            "schema_version": "polarmace-state-export-report-v1",
            "selected_model": clean_value(selected_model),
            "state_dict": manifest["state_dict"],
            "status": "exported",
            "written_files": [
                "manifest.toml",
                "state_export_manifest.toml",
                "mace_polar_state.json",
                "export_report.json",
                *[entry["path"] for entry in tensors],
            ],
        }
        write_manifest_toml(temp_dir / "state_export_manifest.toml", manifest, tensors)
        if args.dtype == "f64":
            write_e3rs_manifest_toml(temp_dir / "manifest.toml", manifest, tensors)
        else:
            write_manifest_toml(temp_dir / "manifest.toml", manifest, tensors)
        write_json(temp_dir / "mace_polar_state.json", state_json)
        write_json(temp_dir / "export_report.json", export_report)
        replace_out_dir(temp_dir, args.out_dir.resolve(), args.force)
        return export_report
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dtype", choices=["f32", "f64"], default="f64")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mace-dir", default="third_party/mace", type=Path)
    parser.add_argument("--e3nn-compat-path", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    report = export(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
