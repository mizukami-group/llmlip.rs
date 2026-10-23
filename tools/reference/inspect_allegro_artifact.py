#!/usr/bin/env python3
"""
Inspect the pinned Allegro-OAM-L packaged model artifact.

The inspection report is reference metadata for exporter planning. It is not an
`.e3rs` export and does not imply Rust inference parity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as std_importlib_metadata
import json
import math
import platform
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "allegro-oam-inspection-v1"


def checked_output(args: list[str]) -> str:
    import subprocess

    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.output.strip()) from exc


def validate_pinned_tree(path: Path, name: str) -> str:
    if not path.exists():
        raise SystemExit(f"missing pinned {name} reference directory: {path}")
    commit = checked_output(["git", "-C", str(path), "rev-parse", "HEAD"])
    pin_path = path.with_name(f"{path.name}.PINNED_COMMIT")
    if not pin_path.exists():
        raise SystemExit(f"missing pinned commit marker: {pin_path}")
    pinned_commit = pin_path.read_text(encoding="utf-8").strip()
    if commit != pinned_commit:
        raise SystemExit(
            f"{name} reference checkout does not match pinned commit marker\n"
            f"checkout: {commit}\n"
            f"marker:   {pinned_commit}"
        )
    return commit


def patch_py39_importlib_metadata() -> None:
    try:
        import importlib_metadata as backport  # type: ignore[import-not-found]
    except Exception:
        return
    for name in ("packages_distributions", "version", "PackageNotFoundError"):
        if not hasattr(std_importlib_metadata, name) and hasattr(backport, name):
            setattr(std_importlib_metadata, name, getattr(backport, name))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def stable_path(path: Path) -> str:
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def import_reference(
    allegro_dir: Path, nequip_dir: Path, e3nn_dir: Path
) -> tuple[Any, Any, Any, Any, Any]:
    sys.path.insert(0, str(allegro_dir.resolve()))
    sys.path.insert(0, str(nequip_dir.resolve()))
    sys.path.insert(0, str(e3nn_dir.resolve()))
    patch_py39_importlib_metadata()
    try:
        import torch  # type: ignore[import-not-found]
        import nequip  # type: ignore[import-not-found]
        import allegro  # type: ignore[import-not-found]
        import e3nn  # type: ignore[import-not-found]
        from nequip.model.saved_models.package import (  # type: ignore[import-not-found]
            ModelFromPackage,
            ModelTypeNamesFromPackage,
        )
    except Exception as exc:
        raise SystemExit(
            "failed to import pinned Allegro inspection dependencies. Run with "
            "PYTHONPATH including third_party/allegro, third_party/nequip, and "
            "third_party/e3nn-torch, and install tools/reference/requirements.txt.\n"
            f"{type(exc).__name__}: {exc}"
        ) from exc

    for module, expected_dir, label in (
        (allegro, allegro_dir, "allegro"),
        (nequip, nequip_dir, "nequip"),
        (e3nn, e3nn_dir, "e3nn"),
    ):
        imported_file = Path(module.__file__).resolve()
        if not str(imported_file).startswith(str(expected_dir.resolve())):
            raise SystemExit(
                f"imported {label} does not come from the pinned checkout\n"
                f"imported: {imported_file}\nexpected under: {expected_dir.resolve()}"
            )
    return torch, nequip, allegro, ModelFromPackage, ModelTypeNamesFromPackage


def tensor_value_sha256(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and sys.byteorder == "big"
    ):
        array = array.byteswap().newbyteorder("<")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def tensor_summary(name: str, tensor: Any) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "value_sha256": tensor_value_sha256(tensor),
    }


def tensor_contract_summary(name: str, tensor: Any) -> dict[str, Any]:
    value = tensor_summary(name, tensor)
    value["nonzero_count"] = int((tensor.detach().cpu() != 0).sum().item())
    return value


def scalar_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if int(tensor.numel()) == 1:
            return tensor.item()
        return str(value)
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def stable_module_name(value: str) -> str:
    return re.sub(r"<torch_package_\d+>", "<torch_package>", value)


def stable_class(module: Any) -> tuple[str, str]:
    class_name = type(module).__name__
    module_name = stable_module_name(type(module).__module__)
    return class_name, module_name


def direct_parameter_numel(module: Any) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters(recurse=False)))


def direct_buffer_numel(module: Any) -> int:
    return int(sum(buffer.numel() for buffer in module.buffers(recurse=False)))


def module_extra(module: Any) -> dict[str, str]:
    fields = (
        "irreps_in",
        "irreps_out",
        "irreps",
        "irreps_edge_sh",
        "irreps_edge_attr",
        "feature_irreps_in",
        "feature_irreps_out",
        "r_max",
        "avg_num_neighbors",
        "num_layers",
        "num_tensor_features",
        "latent_dim",
    )
    extra = {}
    for field in fields:
        if hasattr(module, field):
            try:
                extra[field] = str(getattr(module, field))
            except Exception:
                extra[field] = "<unprintable>"
    return extra


def module_summary(name: str, module: Any) -> dict[str, Any]:
    class_name, module_name = stable_class(module)
    return {
        "name": name,
        "class": class_name,
        "module": module_name,
        "direct_parameter_numel": direct_parameter_numel(module),
        "direct_buffer_numel": direct_buffer_numel(module),
        "child_count": len(list(module.children())),
        "extra": module_extra(module),
    }


def dtype_totals(state: dict[str, Any]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for tensor in state.values():
        dtype = str(tensor.dtype)
        entry = totals.setdefault(dtype, {"count": 0, "numel": 0})
        entry["count"] += 1
        entry["numel"] += int(tensor.numel())
    return totals


def summarize_named_modules(model: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
    modules = [module_summary(name, module) for name, module in model.named_modules()]
    counts = Counter(item["class"] for item in modules)
    return modules, dict(sorted(counts.items()))


def irrep_blocks(irreps: Any) -> list[dict[str, Any]]:
    blocks = []
    offset = 0
    for index, mul_ir in enumerate(irreps):
        dim = int(mul_ir.ir.dim)
        block_dim = int(mul_ir.mul) * dim
        blocks.append(
            {
                "index": index,
                "repr": str(mul_ir),
                "mul": int(mul_ir.mul),
                "l": int(mul_ir.ir.l),
                "parity": "e" if int(mul_ir.ir.p) == 1 else "o",
                "dim": dim,
                "offset": offset,
                "block_dim": block_dim,
            }
        )
        offset += block_dim
    return blocks


def tensor_product_path_table(irreps_in1: Any, irreps_in2: Any, irreps_out: Any) -> list[dict[str, Any]]:
    blocks1 = irrep_blocks(irreps_in1)
    blocks2 = irrep_blocks(irreps_in2)
    blocks_out = irrep_blocks(irreps_out)
    paths = []
    for out in blocks_out:
        parity_out = 1 if out["parity"] == "e" else -1
        for in1 in blocks1:
            parity_in1 = 1 if in1["parity"] == "e" else -1
            for in2 in blocks2:
                parity_in2 = 1 if in2["parity"] == "e" else -1
                if parity_in1 * parity_in2 != parity_out:
                    continue
                if abs(in1["l"] - in2["l"]) <= out["l"] <= in1["l"] + in2["l"]:
                    paths.append(
                        {
                            "path_index": len(paths),
                            "in1_index": in1["index"],
                            "in2_index": in2["index"],
                            "out_index": out["index"],
                            "l_in1": in1["l"],
                            "l_in2": in2["l"],
                            "l_out": out["l"],
                            "parity_in1": in1["parity"],
                            "parity_in2": in2["parity"],
                            "parity_out": out["parity"],
                            "component_normalization": math.sqrt(2 * out["l"] + 1),
                        }
                    )
    return paths


def direct_tensor_ref(state: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in state:
        raise SystemExit(f"expected tensor missing from state_dict: {name}")
    return tensor_contract_summary(name, state[name])


def named_modules_by_class(model: Any, class_name: str) -> list[tuple[str, Any]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if stable_class(module)[0] == class_name
    ]


def scalar_mlp_summary(name: str, module: Any, state: dict[str, Any]) -> dict[str, Any]:
    layers = []
    if not hasattr(module, "mlp"):
        raise SystemExit(f"expected ScalarMLPFunction-like module to have .mlp: {name}")
    for index, child in enumerate(module.mlp):
        class_name, module_name = stable_class(child)
        layer_name = f"{name}.mlp.{index}"
        layer: dict[str, Any] = {
            "index": index,
            "class": class_name,
            "module": module_name,
        }
        if hasattr(child, "weight"):
            weight_name = f"{layer_name}.weight"
            layer.update(
                {
                    "in_features": int(getattr(child, "in_features")),
                    "out_features": int(getattr(child, "out_features")),
                    "alpha": scalar_value(getattr(child, "alpha")),
                    "bias": scalar_value(getattr(child, "bias")),
                    "weight": direct_tensor_ref(state, weight_name),
                }
            )
        layers.append(layer)
    return {
        "name": name,
        "class": stable_class(module)[0],
        "bias": scalar_value(getattr(module, "bias", None)),
        "num_layers": int(getattr(module, "num_layers")),
        "layers": layers,
    }


def contracter_summary(name: str, module: Any, state: dict[str, Any], torch: Any) -> dict[str, Any]:
    weights_name = f"{name}.weights"
    w3j_name = f"{name}.w3j"
    if not hasattr(module, "weights") or not hasattr(module, "w3j"):
        raise SystemExit(f"expected Contracter tensors missing on {name}")
    weight_w3j_einsum = str(getattr(module, "_weight_w3j_einstr"))
    ww3j = torch.einsum(weight_w3j_einsum, module.weights, module.w3j).detach().contiguous()
    diagonal = bool(getattr(module, "w3j_is_ij_diagonal"))
    forward_formula = (
        "outer = x1 * x2; out = sum_i outer[z,u,i] * ww3j[u,i,k]"
        if diagonal
        else "outer = x1[:,:,i] * x2[:,:,j]; out = sum_ij outer[z,u,i,j] * ww3j[u,i,j,k]"
    )
    contract_einsum_semantics = "zui,uik->zuk" if diagonal else "zuij,uijk->zuk"
    return {
        "name": name,
        "class": stable_class(module)[0],
        "irreps_in1": str(getattr(module, "irreps_in1")),
        "irreps_in2": str(getattr(module, "irreps_in2")),
        "irreps_out": str(getattr(module, "irreps_out")),
        "irrep_blocks_in1": irrep_blocks(getattr(module, "irreps_in1")),
        "irrep_blocks_in2": irrep_blocks(getattr(module, "irreps_in2")),
        "irrep_blocks_out": irrep_blocks(getattr(module, "irreps_out")),
        "mul": int(getattr(module, "mul")),
        "base_dim1": int(getattr(module, "base_dim1")),
        "base_dim2": int(getattr(module, "base_dim2")),
        "base_dim_out": int(getattr(module, "base_dim_out")),
        "num_paths": int(getattr(module, "num_paths")),
        "path_channel_coupling": bool(getattr(module, "path_channel_coupling")),
        "scatter_factor": scalar_value(getattr(module, "scatter_factor")),
        "irrep_normalization": str(getattr(module, "irrep_normalization")),
        "w3j_is_ij_diagonal": diagonal,
        "weight_w3j_einsum": weight_w3j_einsum,
        "contract_einsum_semantics": contract_einsum_semantics,
        "forward_formula": forward_formula,
        "instructions": getattr(module, "instructions", None),
        "instructions_source": "generated_from_irreps_output_major",
        "path_table": tensor_product_path_table(
            getattr(module, "irreps_in1"),
            getattr(module, "irreps_in2"),
            getattr(module, "irreps_out"),
        ),
        "weights": direct_tensor_ref(state, weights_name),
        "w3j": direct_tensor_ref(state, w3j_name),
        "precontracted_ww3j": tensor_contract_summary(
            f"{name}.precontracted_ww3j", ww3j
        ),
        "contract_input_shapes": {
            "x1": ["edge_count", "mul", "base_dim1"],
            "x2": ["edge_count_or_scatter_source_count", "mul", "base_dim2"],
            "idxs": ["edge_count"],
            "output": ["edge_count", "mul", "base_dim_out"],
        },
    }


def weighted_channels_summary(name: str, module: Any) -> dict[str, Any]:
    if not hasattr(module, "_rtoi"):
        raise SystemExit(f"expected MakeWeightedChannels buffer missing on {name}")
    individual = bool(getattr(module, "weight_individual_irreps"))
    return {
        "name": name,
        "class": stable_class(module)[0],
        "weight_numel": int(getattr(module, "weight_numel")),
        "multiplicity_out": int(getattr(module, "multiplicity_out")),
        "weight_individual_irreps": individual,
        "alpha": scalar_value(getattr(module, "alpha")),
        "num_irreps": int(getattr(module, "_num_irreps")),
        "rtoi": tensor_contract_summary(f"{name}._rtoi", getattr(module, "_rtoi")),
        "weights_shape": (
            ["edge_count", "multiplicity_out", "num_irreps"]
            if individual
            else ["edge_count", "multiplicity_out"]
        ),
        "forward_formula": (
            "aux = weights.reshape(z*u,r) @ rtoi[r,i]; out[z,u,i] = edge_attr[z,i] * aux[z,u,i]"
            if individual
            else "out[z,u,i] = weights[z,u] * alpha * edge_attr[z,i]"
        ),
    }


def product_type_embedding_summary(name: str, module: Any, state: dict[str, Any]) -> dict[str, Any]:
    basis_name = f"{name}.basis_linear"
    return {
        "name": name,
        "class": stable_class(module)[0],
        "in_field": str(getattr(module, "in_field")),
        "out_field": str(getattr(module, "out_field")),
        "edge_type_field": str(getattr(module, "edge_type_field")),
        "num_types": int(getattr(module, "num_types")),
        "center_embed": direct_tensor_ref(state, f"{name}.center_embed.weight"),
        "neighbor_embed": direct_tensor_ref(state, f"{name}.neighbor_embed.weight"),
        "basis_linear": scalar_mlp_summary(basis_name, getattr(module, "basis_linear"), state),
        "forward_formula": (
            "type_embed = concat(center_embed[edge_type[0]], neighbor_embed[edge_type[1]]); "
            "out = type_embed * basis_linear(edge_embedding)"
        ),
    }


def edgewise_reduce_summary(name: str, module: Any) -> dict[str, Any]:
    return {
        "name": name,
        "class": stable_class(module)[0],
        "field": str(getattr(module, "field")),
        "out_field": str(getattr(module, "out_field")),
        "reduce": str(getattr(module, "reduce")),
        "factor": scalar_value(getattr(module, "_factor", None)),
        "edge_index_destination_row": 0,
        "sqrt2_force_pair_normalization": True,
        "forward_formula": (
            "scatter edge field to edge_index[0], apply AvgNumNeighborsNorm factor, "
            "then divide by sqrt(2)"
        ),
    }


def allegro_module_summary(name: str, module: Any) -> dict[str, Any]:
    return {
        "name": name,
        "class": stable_class(module)[0],
        "num_layers": int(getattr(module, "num_layers")),
        "num_scalar_features": int(getattr(module, "num_scalar_features")),
        "num_tensor_features": int(getattr(module, "num_tensor_features")),
        "tensor_track_allowed_irreps": str(getattr(module, "tensor_track_allowed_irreps")),
        "n_scalar_outs": list(getattr(module, "_n_scalar_outs")),
    }


def radial_and_sh_summary(model: Any, state: dict[str, Any]) -> dict[str, Any]:
    modules_by_name = dict(model.named_modules())
    edge_norm = modules_by_name["model.func.edge_norm"]
    bessel = modules_by_name["model.func.radial_chemical_embed.bessel_encode"]
    cutoff = modules_by_name["model.func.radial_chemical_embed.bessel_encode.cutoff"]
    tensor_embed = modules_by_name["model.func.tensor_embed"]
    sh = modules_by_name["model.func.tensor_embed.sh"]
    return {
        "edge_norm": {
            "name": "model.func.edge_norm",
            "r_max": scalar_value(getattr(edge_norm, "r_max")),
            "rmax_recip": direct_tensor_ref(state, "model.func.edge_norm._rmax_recip"),
            "norm_length_field": str(getattr(edge_norm, "norm_length_field")),
            "edge_type_field": str(getattr(edge_norm, "edge_type_field")),
            "num_types": int(getattr(edge_norm, "num_types")),
        },
        "bessel": {
            "name": "model.func.radial_chemical_embed.bessel_encode",
            "num_bessels": int(getattr(bessel, "num_bessels")),
            "trainable": bool(getattr(bessel, "trainable")),
            "cutoff_p": scalar_value(getattr(cutoff, "p")),
            "norm_length_field": str(getattr(bessel, "norm_length_field")),
            "edge_invariant_field": str(getattr(bessel, "edge_invariant_field")),
            "weights": direct_tensor_ref(
                state,
                "model.func.radial_chemical_embed.bessel_encode.bessel_weights",
            ),
            "cutoff_module": stable_class(cutoff)[0],
        },
        "tensor_embed": {
            "name": "model.func.tensor_embed",
            "scalar_embedding_in_field": str(getattr(tensor_embed, "scalar_embedding_in_field")),
            "tensor_basis_out_field": str(getattr(tensor_embed, "tensor_basis_out_field")),
            "tensor_embedding_out_field": str(
                getattr(tensor_embed, "tensor_embedding_out_field")
            ),
        },
        "spherical_harmonics": {
            "name": "model.func.tensor_embed.sh",
            "irreps_in": str(getattr(sh, "irreps_in")),
            "irreps_out": str(getattr(sh, "irreps_out")),
            "normalize": bool(getattr(sh, "normalize")),
            "normalization": str(getattr(sh, "normalization")),
        },
    }


def execution_contract(model: Any, state: dict[str, Any], torch: Any) -> dict[str, Any]:
    modules_by_name = dict(model.named_modules())
    contracters = [
        contracter_summary(name, module, state, torch)
        for name, module in named_modules_by_class(model, "Contracter")
    ]
    weighted_channels = [
        weighted_channels_summary(name, module)
        for name, module in named_modules_by_class(model, "MakeWeightedChannels")
    ]
    product_embeddings = [
        product_type_embedding_summary(name, module, state)
        for name, module in named_modules_by_class(model, "ProductTypeEmbedding")
    ]
    edgewise_reduces = [
        edgewise_reduce_summary(name, module)
        for name, module in named_modules_by_class(model, "EdgewiseReduce")
    ]
    scalar_mlps = [
        scalar_mlp_summary(name, module, state)
        for name, module in named_modules_by_class(model, "ScalarMLPFunction")
    ]
    per_type = modules_by_name["model.func.per_type_energy_scale_shift"]
    return {
        "schema_version": "allegro-oam-execution-contract-v1",
        "status": "export_plan_pinned_runtime_not_implemented",
        "radial_and_spherical": radial_and_sh_summary(model, state),
        "product_type_embeddings": product_embeddings,
        "scalar_mlps": scalar_mlps,
        "weighted_channels": weighted_channels,
        "allegro_module": allegro_module_summary("model.func.allegro", modules_by_name["model.func.allegro"]),
        "contracters": contracters,
        "edgewise_reduces": edgewise_reduces,
        "per_type_scale_shift": {
            "name": "model.func.per_type_energy_scale_shift",
            "field": str(getattr(per_type, "field")),
            "out_field": str(getattr(per_type, "out_field")),
            "num_types": int(getattr(per_type, "num_types")),
            "scales": direct_tensor_ref(
                state, "model.func.per_type_energy_scale_shift.scales"
            ),
            "shifts": direct_tensor_ref(
                state, "model.func.per_type_energy_scale_shift.shifts"
            ),
        },
        "unsupported_branches": [
            "training",
            "GPU",
            "arbitrary Allegro or NeQuIP package loading",
            "runtime Python or torch dependency in Rust crates",
            "parameter gradients",
        ],
    }


def contract_summary(modules: list[dict[str, Any]]) -> dict[str, Any]:
    interesting = {
        "ProductTypeEmbedding",
        "MakeWeightedChannels",
        "Contracter",
        "Allegro_Module",
        "TwoBodyBesselScalarEmbed",
        "TwoBodySphericalHarmonicTensorEmbed",
        "EdgewiseReduce",
        "PerTypeScaleShift",
        "ScalarMLP",
    }
    return {
        "class_counts": {
            class_name: sum(1 for module in modules if module["class"] == class_name)
            for class_name in sorted(interesting)
        },
        "module_names": [
            {
                "name": module["name"],
                "class": module["class"],
                "extra": module["extra"],
            }
            for module in modules
            if module["class"] in interesting
        ],
        "notes": [
            "This report pins module and tensor inventory plus a detailed execution contract for exporter planning.",
            "Exporter/runtime work must still generate `.e3rs` payloads and prove Rust energy/force parity.",
        ],
    }


def inspect_artifact(args: argparse.Namespace) -> dict[str, Any]:
    allegro_commit = validate_pinned_tree(args.allegro_dir, "Allegro")
    nequip_commit = validate_pinned_tree(args.nequip_dir, "NeQUIP")
    e3nn_commit = validate_pinned_tree(args.e3nn_dir, "e3nn")
    manifest = load_json(args.manifest)
    artifact = manifest["artifact"]
    package_path = args.package or Path(str(artifact["cache_path"]))
    if not package_path.exists():
        raise SystemExit(f"cached Allegro package is missing: {package_path}")

    sha256, size_bytes = sha256_file(package_path)
    expected_sha256 = artifact.get("sha256")
    expected_size = artifact.get("size_bytes")
    if expected_sha256 and sha256 != expected_sha256:
        raise SystemExit(
            "cached Allegro artifact SHA256 does not match manifest\n"
            f"expected: {expected_sha256}\nactual:   {sha256}"
        )
    if expected_size and size_bytes != int(expected_size):
        raise SystemExit(
            "cached Allegro artifact size does not match manifest\n"
            f"expected: {expected_size}\nactual:   {size_bytes}"
        )

    torch, nequip, allegro, ModelFromPackage, ModelTypeNamesFromPackage = import_reference(
        args.allegro_dir, args.nequip_dir, args.e3nn_dir
    )
    type_names = list(ModelTypeNamesFromPackage(str(package_path)))
    packaged = ModelFromPackage(str(package_path), compile_mode=args.compile_mode)
    packaged_keys = list(packaged.keys()) if hasattr(packaged, "keys") else []
    model = packaged[args.model_key] if args.model_key else packaged
    state = model.state_dict()
    state_tensors = [tensor_summary(name, tensor) for name, tensor in sorted(state.items())]
    modules, module_class_counts = summarize_named_modules(model)
    metadata = {
        str(key): str(value) for key, value in getattr(model, "metadata", {}).items()
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "selected_model": manifest["selected_model"],
        "artifact_manifest": stable_path(args.manifest),
        "package": {
            "path": stable_path(package_path),
            "filename": package_path.name,
            "sha256": sha256,
            "size_bytes": size_bytes,
            "compile_mode": args.compile_mode,
            "model_key": args.model_key,
            "packaged_model_keys": packaged_keys,
        },
        "reference": {
            "allegro_repository": str(args.allegro_dir),
            "allegro_commit": allegro_commit,
            "allegro_package_version": getattr(allegro, "__version__", "unknown"),
            "nequip_repository": str(args.nequip_dir),
            "nequip_commit": nequip_commit,
            "nequip_package_version": getattr(nequip, "__version__", "unknown"),
            "e3nn_repository": str(args.e3nn_dir),
            "e3nn_commit": e3nn_commit,
            "e3nn_package_version": getattr(
                sys.modules.get("e3nn"), "__version__", "unknown"
            ),
            "torch_version": getattr(torch, "__version__", "unknown"),
            "python_version": platform.python_version(),
        },
        "model": {
            "class": stable_class(model)[0],
            "module": stable_class(model)[1],
            "metadata": metadata,
            "type_names": type_names,
            "num_types": len(type_names),
            "state_dict": {
                "count": len(state_tensors),
                "dtype_totals": dtype_totals(state),
                "tensors": state_tensors,
            },
            "modules": {
                "count": len(modules),
                "class_counts": module_class_counts,
                "items": modules,
            },
        },
        "allegro_contract": contract_summary(modules),
        "execution_contract": execution_contract(model, state, torch),
        "runtime_status": "artifact_inspected_export_not_implemented",
        "normal_cargo_test_dependency": False,
        "notes": [
            "This inspection verifies package identity and records module/tensor inventory plus exporter-planning execution details.",
            "It is not an `.e3rs` export and does not claim Rust energy/force parity.",
            "The loaded package currently exposes 89 type names; compare against model-card metadata before claiming supported species.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allegro-dir", type=Path, default=Path("third_party/allegro"))
    parser.add_argument("--nequip-dir", type=Path, default=Path("third_party/nequip"))
    parser.add_argument("--e3nn-dir", type=Path, default=Path("third_party/e3nn-torch"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("fixtures/allegro/current/allegro_oam_l_artifact_manifest.json"),
    )
    parser.add_argument("--package", type=Path)
    parser.add_argument("--compile-mode", default="eager")
    parser.add_argument("--model-key", default="sole_model")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("fixtures/allegro/current/allegro_oam_l_inspection.json"),
    )
    args = parser.parse_args()

    report = inspect_artifact(args)
    write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
