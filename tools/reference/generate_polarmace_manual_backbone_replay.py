#!/usr/bin/env python3
"""Generate a compact PolarMACE manual-backbone replay fixture.

This is reference tooling only. It manually replays the local MACE backbone in
upstream PolarMACE through `interaction_energy` and the concatenated backbone
`node_feats`, then compares those tensors against the upstream forward output.
It intentionally stops before graph_longrange density/electrostatics recursion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from generate_e3nn_fixtures import tensor_product_plan_metadata
from polarmace_fixture_structures import named_structure_spec


PINNED_MACE_COMMIT = "4d2da09413ac1407f37cdbb6b81fa28e4c15655e"
PINNED_ARTIFACT_SHA256 = (
    "e4495612037b3b3312633182882a38a694ecac9ea0be2b9889ac0b2a84a99510"
)
SCHEMA_VERSION = "polarmace-manual-backbone-replay-v1"
BACKBONE_RUNTIME_STATUS = "reference_manual_backbone_replay_only"


BACKBONE_HOOK_TARGETS = [
    "node_embedding",
    "radial_embedding",
    "spherical_harmonics",
    "atomic_energies_fn",
    "interactions.0",
    "products.0",
    "readouts.0",
    "scale_shift",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pinned_commit(mace_dir: Path) -> str:
    git_pointer = mace_dir / ".git"
    if git_pointer.is_file():
        text = git_pointer.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            git_dir = (mace_dir / text.split(":", 1)[1].strip()).resolve()
        else:
            raise RuntimeError(f"Unsupported .git file format in {git_pointer}")
    elif git_pointer.is_dir():
        git_dir = git_pointer
    else:
        raise RuntimeError(f"Cannot find git metadata for {mace_dir}")

    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref:"):
        ref = head.split(" ", 1)[1].strip()
        ref_path = git_dir / ref
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
        packed_refs = git_dir / "packed-refs"
        if packed_refs.exists():
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if not line or line.startswith("#") or line.startswith("^"):
                    continue
                commit, packed_ref = line.split(" ", 1)
                if packed_ref == ref:
                    return commit
        raise RuntimeError(f"Cannot resolve {ref} in {git_dir}")
    return head


def clean_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return clean_value(value.item())
    if isinstance(value, np.ndarray):
        return clean_value(value.tolist())
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [clean_value(v) for v in value]
    if isinstance(value, tuple):
        return [clean_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): clean_value(v) for k, v in value.items()}
    return str(value)


def tensor_summary(tensor: Any, *, max_values: int = 8) -> dict[str, Any]:
    import torch

    with torch.no_grad():
        detached = tensor.detach().cpu()
        flat = detached.reshape(-1)
        summary: dict[str, Any] = {
            "kind": "tensor",
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "numel": int(detached.numel()),
        }
        if detached.numel() == 0:
            summary.update({"finite": True})
            return summary
        if torch.is_floating_point(detached):
            finite = torch.isfinite(flat)
            summary["finite"] = bool(finite.all().item())
            if bool(finite.any().item()):
                finite_flat = flat[finite]
                summary.update(
                    {
                        "min": float(finite_flat.min().item()),
                        "max": float(finite_flat.max().item()),
                        "mean": float(finite_flat.mean().item()),
                        "max_abs": float(finite_flat.abs().max().item()),
                    }
                )
        else:
            summary["finite"] = True
            summary.update(
                {
                    "min": clean_value(flat.min().item()),
                    "max": clean_value(flat.max().item()),
                }
            )
        if detached.numel() <= max_values:
            summary["values"] = clean_value(detached.tolist())
        return summary


def tensor_exact_values(tensor: Any) -> dict[str, Any]:
    """Return full tensor values for small convention-pinning diagnostics."""
    summary = tensor_summary(tensor, max_values=0)
    summary["values"] = clean_value(tensor.detach().cpu().tolist())
    return summary


def tensor_exact_rows(tensor: Any) -> dict[str, Any]:
    """Return full row-wise tensor values for small 2D replay tensors."""
    summary = tensor_summary(tensor, max_values=0)
    detached = tensor.detach().cpu()
    if len(detached.shape) != 2:
        raise ValueError(f"expected a 2D tensor, got shape {list(detached.shape)}")
    summary["rows"] = [
        {"row": row, "values": clean_value(detached[row].tolist())}
        for row in range(int(detached.shape[0]))
    ]
    return summary


def tensor_selected_rows(tensor: Any, rows: list[int]) -> dict[str, Any]:
    """Return selected row-wise tensor values for compact input-side pins."""
    summary = tensor_summary(tensor, max_values=0)
    detached = tensor.detach().cpu()
    if len(detached.shape) != 2:
        raise ValueError(f"expected a 2D tensor, got shape {list(detached.shape)}")
    summary["rows"] = [
        {"row": row, "values": clean_value(detached[row].tolist())} for row in rows
    ]
    return summary


def linear_boundary_pin(
    module: Any,
    input_tensor: Any,
    output_tensor: Any,
    *,
    input_rows: list[int] | None = None,
) -> dict[str, Any]:
    """Return exact output rows and compact input/parameter pins for a Linear boundary."""
    if input_rows is None:
        input_rows = list(range(int(input_tensor.shape[0])))
    weight = getattr(module, "weight", None)
    output_mask = getattr(module, "output_mask", None)
    bias = getattr(module, "bias", None)
    return {
        "input_rows": tensor_selected_rows(input_tensor, input_rows),
        "output_rows": tensor_exact_rows(output_tensor),
        "weight": tensor_exact_values(weight) if weight is not None else None,
        "output_mask": tensor_exact_values(output_mask)
        if output_mask is not None
        else None,
        "bias": tensor_exact_values(bias) if bias is not None else None,
    }


def module_parameter_pin(module: Any) -> dict[str, Any]:
    """Return exact non-recursive parameters and buffers for a module boundary."""
    return {
        "module_class": f"{type(module).__module__}.{type(module).__name__}",
        "parameters": {
            name: tensor_exact_values(value)
            for name, value in module.named_parameters(recurse=False)
        },
        "buffers": {
            name: tensor_exact_values(value)
            for name, value in module.named_buffers(recurse=False)
        },
        "metadata": clean_value(
            {
                "in_features": getattr(module, "in_features", None),
                "out_features": getattr(module, "out_features", None),
                "normalized_shape": getattr(module, "normalized_shape", None),
                "eps": getattr(module, "eps", None),
                "elementwise_affine": getattr(module, "elementwise_affine", None),
            }
        ),
    }


def e3nn_linear_metadata(module: Any) -> dict[str, Any]:
    """Return convention-sensitive e3nn Linear constants without guessing."""
    instructions = []
    for instruction in getattr(module, "instructions", []):
        if hasattr(instruction, "_asdict"):
            instructions.append(clean_value(instruction._asdict()))
        else:
            instructions.append(clean_value(vars(instruction)))
    return {
        "module_class": f"{type(module).__module__}.{type(module).__name__}",
        "irreps_in": str(getattr(module, "irreps_in", "unknown")),
        "irreps_out": str(getattr(module, "irreps_out", "unknown")),
        "weight_numel": int(getattr(module, "weight_numel", -1)),
        "instructions": instructions,
    }


def tensor_or_clean_attr(value: Any) -> Any:
    """Return exact tensor attrs and JSON-safe scalar attrs."""
    try:
        import torch

        if torch.is_tensor(value):
            return tensor_exact_values(value)
    except Exception:
        pass
    return clean_value(value)


def readout_boundary_pin(
    readout: Any,
    product_out: Any,
    node_heads: Any,
    readout_out: Any,
    node_es: Any,
    *,
    readout_index: int,
) -> dict[str, Any]:
    """Pin readouts[layer](product_out, node_heads) and selected node energies."""
    linear = getattr(readout, "linear", None)
    if linear is not None:
        linear_pin = {
            **linear_boundary_pin(linear, product_out, readout_out),
            "metadata": e3nn_linear_metadata(linear),
        }
    else:
        linear_pin = {
            "metadata": {
                "metadata_missing": [
                    "readout has no `.linear` attribute; raw_output is pinned only",
                ],
            },
        }
    return {
        "description": f"Layer-0 readout boundary: readouts[{readout_index}](product_out, node_heads), then select [num_atoms_arange, node_heads].",
        "readout_call": f"readouts[{readout_index}](product_out, node_heads)",
        "readout_class": f"{type(readout).__module__}.{type(readout).__name__}",
        "metadata": {
            "readout_index": readout_index,
            "node_heads": tensor_exact_values(node_heads),
        },
        "input_name": "product_out",
        "input_rows": tensor_exact_rows(product_out),
        "linear": linear_pin,
        "raw_output": tensor_exact_values(readout_out),
        "selected_node_es": tensor_exact_values(node_es),
    }


def scale_shift_boundary_pin(
    scale_shift: Any,
    node_inter_es_unscaled: Any,
    node_heads: Any,
    node_inter_es: Any,
    interaction_energy: Any,
) -> dict[str, Any]:
    """Pin scale_shift(node_inter_es_unscaled, node_heads) and graph scatter."""
    state = scale_shift.state_dict() if hasattr(scale_shift, "state_dict") else {}
    scale_attr = getattr(scale_shift, "scale", None)
    shift_attr = getattr(scale_shift, "shift", None)
    return {
        "description": "Scale-shift boundary after layer-0 readout sum: scale_shift(node_inter_es_unscaled, node_heads), then scatter_sum to interaction_energy.",
        "scale_shift_call": "scale_shift(node_inter_es_unscaled, node_heads)",
        "scale_shift_class": f"{type(scale_shift).__module__}.{type(scale_shift).__name__}",
        "metadata": {
            "node_heads": tensor_exact_values(node_heads),
            "scale_attr": tensor_or_clean_attr(scale_attr),
            "shift_attr": tensor_or_clean_attr(shift_attr),
            "state_dict": {
                name: tensor_exact_values(value) for name, value in state.items()
            },
        },
        "node_inter_es_unscaled": tensor_exact_values(node_inter_es_unscaled),
        "node_inter_es": tensor_exact_values(node_inter_es),
        "interaction_energy": tensor_exact_values(interaction_energy),
    }


def sequential_boundary_pin(
    module: Any,
    input_tensor: Any,
    *,
    target_indices: set[int],
) -> tuple[Any, dict[str, Any]]:
    """Run a Sequential-like net and pin selected layer parameters and outputs."""
    current = input_tensor
    layers: dict[str, Any] = {}
    for index, layer in enumerate(module.net):
        layer_input = current
        current = layer(current)
        if index in target_indices:
            layers[str(index)] = {
                **module_parameter_pin(layer),
                "input_rows": tensor_exact_rows(layer_input),
                "output_rows": tensor_exact_rows(current),
            }
    return current, layers


def tensor_exact_or_blocker(
    state: dict[str, Any],
    name: str,
    *,
    owner: str,
) -> dict[str, Any]:
    """Return a full state tensor payload or a precise extraction blocker."""
    if name not in state:
        return {
            "status": "blocked",
            "blocker": f"{owner} state_dict does not contain required tensor {name!r}",
        }
    return tensor_exact_values(state[name])


def missing_attr_blocker(owner: Any, name: str) -> dict[str, Any]:
    """Return explicit metadata for attrs that cannot be extracted safely."""
    return {
        "status": "blocked",
        "value": None,
        "blocker": (
            f"{type(owner).__module__}.{type(owner).__name__} has no non-None "
            f"{name!r} attribute, and no {name!r} entry exists in state_dict"
        ),
    }


def optional_module_attr_or_blocker(owner: Any, name: str) -> Any:
    value = getattr(owner, name, None)
    if value is None:
        state = owner.state_dict() if hasattr(owner, "state_dict") else {}
        if name in state:
            return tensor_exact_values(state[name])
        return missing_attr_blocker(owner, name)
    return clean_value(value)


def symmetric_contraction_scalar_plan(
    module: Any,
    contraction: Any,
    *,
    index: int,
    input_tensor: Any,
    effective_node_attrs: Any,
    use_agnostic_product: bool,
) -> dict[str, Any]:
    """Return explicit scalar symmetric-contraction replay metadata."""
    state = contraction.state_dict()
    owner = f"contractions[{index}]"

    num_elements = None
    if "weights.1" in state:
        num_elements = int(state["weights.1"].shape[0])
    elif effective_node_attrs is not None:
        num_elements = int(effective_node_attrs.shape[1])

    num_features = getattr(contraction, "num_features", None)
    if num_features is None and input_tensor is not None:
        num_features = int(input_tensor.shape[1])
    if num_features is not None:
        num_features = int(num_features)

    num_ell = None
    if input_tensor is not None:
        num_ell = int(input_tensor.shape[2])
    elif "U_matrix_1" in state:
        num_ell = int(state["U_matrix_1"].shape[0])

    scalar_plan = {
        "kind": "scalar",
        "index": index,
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
        "use_agnostic_product": use_agnostic_product,
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
        "state_tensors_full": {
            name: tensor_exact_or_blocker(state, name, owner=owner)
            for name in [
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
        },
    }

    return scalar_plan


def symmetric_contractions_metadata(
    module: Any,
    *,
    input_tensor: Any | None = None,
    effective_node_attrs: Any | None = None,
    use_agnostic_product: bool = False,
) -> dict[str, Any]:
    """Return safe symmetric-contraction metadata and replay payloads."""
    contractions = []
    for index, contraction in enumerate(getattr(module, "contractions", [])):
        coupling_irreps = getattr(contraction, "coupling_irreps", None)
        contraction_metadata = {
            "index": index,
            "module_class": f"{type(contraction).__module__}.{type(contraction).__name__}",
            "correlation": clean_value(getattr(contraction, "correlation", None)),
            "num_features": clean_value(getattr(contraction, "num_features", None)),
            "irreps_in": str(getattr(module, "irreps_in", "unknown")),
            "irreps_out": str(getattr(module, "irreps_out", "unknown")),
            "coupling_irreps": clean_value(
                [str(irrep) for irrep in coupling_irreps]
                if coupling_irreps is not None
                else None
            ),
            "state_tensors": {
                name: tensor_summary(value, max_values=0)
                for name, value in contraction.state_dict().items()
            },
            "metadata_blockers": [
                missing_attr_blocker(contraction, name)
                for name in [
                    "irrep_normalization",
                    "path_normalization",
                    "use_reduced_cg",
                ]
                if getattr(contraction, name, None) is None
            ],
        }
        if index == 0 and input_tensor is not None:
            contraction_metadata["scalar_plan"] = symmetric_contraction_scalar_plan(
                module,
                contraction,
                index=index,
                input_tensor=input_tensor,
                effective_node_attrs=effective_node_attrs,
                use_agnostic_product=use_agnostic_product,
            )
        contractions.append(contraction_metadata)

    return {
        "module_class": f"{type(module).__module__}.{type(module).__name__}",
        "irreps_in": str(getattr(module, "irreps_in", "unknown")),
        "irreps_out": str(getattr(module, "irreps_out", "unknown")),
        "layout": {
            "input": "[node, feature, ell]",
            "output": "[node, feature]",
            "effective_node_attrs": "[node, element]",
        },
        "use_agnostic_product": use_agnostic_product,
        "metadata_blockers": [
            missing_attr_blocker(module, name)
            for name in [
                "irrep_normalization",
                "path_normalization",
                "use_reduced_cg",
            ]
            if getattr(module, name, None) is None
        ],
        "contractions": contractions,
    }


def product_boundary_pin(
    product: Any,
    input_tensor: Any,
    skip_connection: Any,
    node_attrs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Replay and pin the layer-local product boundary only."""
    import torch

    use_agnostic_product = bool(getattr(product, "use_agnostic_product", False))
    use_sc = bool(getattr(product, "use_sc", False))
    effective_node_attrs = node_attrs
    if use_agnostic_product:
        effective_node_attrs = torch.ones(
            (input_tensor.shape[0], 1),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )

    symmetric_out = product.symmetric_contractions(input_tensor, effective_node_attrs)
    product_linear_out = product.linear(symmetric_out)
    skip_added = (
        product_linear_out + skip_connection
        if use_sc and skip_connection is not None
        else product_linear_out
    )
    product_out = product(
        node_feats=input_tensor,
        sc=skip_connection,
        node_attrs=node_attrs,
    )

    return product_out, {
        "description": "Layer-local product boundary only: symmetric_contractions(interaction_out, effective_node_attrs), product.linear, then optional skip add.",
        "product_class": f"{type(product).__module__}.{type(product).__name__}",
        "path_constants": {
            "use_agnostic_product": use_agnostic_product,
            "use_sc": use_sc,
            "effective_node_attrs": (
                "ones([num_nodes, 1])"
                if use_agnostic_product
                else "node_attrs"
            ),
        },
        "symmetric_contractions": {
            "metadata": symmetric_contractions_metadata(
                product.symmetric_contractions,
                input_tensor=input_tensor,
                effective_node_attrs=effective_node_attrs,
                use_agnostic_product=use_agnostic_product,
            ),
            "input": tensor_exact_values(input_tensor),
            "effective_node_attrs": tensor_exact_rows(effective_node_attrs),
            "output": tensor_exact_rows(symmetric_out),
        },
        "linear": {
            **linear_boundary_pin(product.linear, symmetric_out, product_linear_out),
            "metadata": e3nn_linear_metadata(product.linear),
        },
        "skip_added": {
            "description": "product.linear output plus skip_connection when use_sc is true and skip_connection is present.",
            "skip_connection": (
                tensor_exact_rows(skip_connection)
                if skip_connection is not None
                else None
            ),
            "output": tensor_exact_rows(skip_added),
        },
        "product_out": tensor_exact_rows(product_out),
        "product_out_max_abs_diff_vs_skip_added": max_abs_diff(product_out, skip_added),
    }


def tensor_product_plan_pin(tp: Any, np_module: Any, torch_module: Any) -> dict[str, Any]:
    """Return expanded tensor-product plan metadata, or a precise blocker."""
    try:
        from e3nn import o3

        meta = tensor_product_plan_metadata(tp, np_module, torch_module, o3)
        return {
            "status": "exported",
            "irreps_in1": str(tp.irreps_in1),
            "irreps_in2": str(tp.irreps_in2),
            "irreps_out": str(tp.irreps_out),
            "weight_numel": int(tp.weight_numel),
            "instruction_in1_segment": meta["instruction_in1_segment"].tolist(),
            "instruction_in2_segment": meta["instruction_in2_segment"].tolist(),
            "instruction_out_segment": meta["instruction_out_segment"].tolist(),
            "instruction_cg_offset": meta["instruction_cg_offset"].tolist(),
            "instruction_cg_len": meta["instruction_cg_len"].tolist(),
            "instruction_weight_offset": meta["instruction_weight_offset"].tolist(),
            "instruction_path_weight": meta["instruction_path_weight"].tolist(),
            "cg_i_m": meta["cg_i_m"].tolist(),
            "cg_j_m": meta["cg_j_m"].tolist(),
            "cg_k_m": meta["cg_k_m"].tolist(),
            "cg_value": meta["cg_value"].tolist(),
        }
    except Exception as exc:
        return {
            "status": "blocked",
            "blocker": f"{type(exc).__name__}: {exc}",
            "irreps_in1": str(getattr(tp, "irreps_in1", "unknown")),
            "irreps_in2": str(getattr(tp, "irreps_in2", "unknown")),
            "irreps_out": str(getattr(tp, "irreps_out", "unknown")),
            "weight_numel": int(getattr(tp, "weight_numel", -1)),
        }


def irreps_partitions(irreps: Any, *, start: int = 0) -> list[dict[str, Any]]:
    """Return contiguous e3nn Irreps partitions with exact dimensions."""
    offset = int(start)
    partitions = []
    for index, (mul, irrep) in enumerate(irreps):
        dim = int(mul) * int(irrep.dim)
        partitions.append(
            {
                "index": index,
                "mul": int(mul),
                "irrep": str(irrep),
                "irrep_dim": int(irrep.dim),
                "dim": dim,
                "start": offset,
                "end": offset + dim,
            }
        )
        offset += dim
    return partitions


def activation_metadata(module: Any) -> dict[str, Any]:
    """Return activation names and normalize2mom constants from upstream modules."""
    acts = []
    for index, act in enumerate(getattr(module, "acts", [])):
        fn = getattr(act, "f", None)
        acts.append(
            {
                "index": index,
                "class_name": f"{type(act).__module__}.{type(act).__name__}",
                "function": getattr(fn, "__name__", str(fn)),
                "normalize2mom_constant": clean_value(getattr(act, "cst", None)),
            }
        )
    return {
        "class_name": f"{type(module).__module__}.{type(module).__name__}",
        "irreps_in": str(getattr(module, "irreps_in", "unknown")),
        "irreps_out": str(getattr(module, "irreps_out", "unknown")),
        "acts": acts,
    }


def tensor_product_instructions(module: Any) -> list[dict[str, Any]]:
    """Return JSON-safe tensor-product instructions from upstream metadata."""
    instructions = []
    for instruction in getattr(module, "instructions", []):
        if hasattr(instruction, "_asdict"):
            instructions.append(clean_value(instruction._asdict()))
        else:
            instructions.append(clean_value(vars(instruction)))
    return instructions


def gate_plan_metadata(
    gate: Any,
    gate_input: Any,
    scalar_activation_rows: Any,
    gate_activation_rows: Any,
    gated_product_rows: Any,
) -> dict[str, Any]:
    """Pin machine-readable e3nn Gate replay metadata without guessed constants."""
    sc = getattr(gate, "sc", None)
    sc_irreps_outs = list(getattr(sc, "irreps_outs", [])) if sc is not None else []
    sc_instructions = clean_value(
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
    mul_output_mask = getattr(mul, "output_mask", None)
    mul_weight = getattr(mul, "weight", None)

    return {
        "class_name": f"{type(gate).__module__}.{type(gate).__name__}",
        "irreps_scalars": str(getattr(gate, "irreps_scalars", "unknown")),
        "irreps_gates": str(getattr(gate, "irreps_gates", "unknown")),
        "irreps_gated": str(getattr(gate, "irreps_gated", "unknown")),
        "irreps_in": str(getattr(gate, "irreps_in", "unknown")),
        "irreps_out": str(getattr(gate, "irreps_out", "unknown")),
        "layout": {
            "input_dim": int(gate_input.shape[-1]),
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
        "activation_input_partitions": {
            "scalar_activation": irreps_partitions(
                getattr(getattr(gate, "act_scalars"), "irreps_in")
            ),
            "gate_activation": irreps_partitions(
                getattr(getattr(gate, "act_gates"), "irreps_in"),
                start=input_partitions[1]["start"] if len(input_partitions) > 1 else 0,
            ),
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
            "output_mask": (
                tensor_exact_values(mul_output_mask)
                if mul_output_mask is not None
                else None
            ),
            "weight": tensor_exact_values(mul_weight) if mul_weight is not None else None,
        },
        "scalar_activation_rows": tensor_exact_rows(scalar_activation_rows),
        "gate_activation_rows": tensor_exact_rows(gate_activation_rows),
        "gated_product_rows": tensor_exact_rows(gated_product_rows),
    }


def node_embedding_inputs(model: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Return compact upstream inputs needed to replay node_embedding."""
    import torch
    from ase.data import atomic_numbers as ase_atomic_numbers

    model_atomic_numbers = [
        int(z) for z in model.atomic_numbers.detach().cpu().reshape(-1).tolist()
    ]
    node_atomic_numbers = [int(ase_atomic_numbers[symbol]) for symbol in spec["symbols"]]
    type_indices = [model_atomic_numbers.index(z) for z in node_atomic_numbers]
    unique_type_indices = sorted(set(type_indices))

    linear = model.node_embedding.linear
    output_mask = linear.output_mask.detach().cpu()
    hidden_dim = int(output_mask.numel())
    node_attr_dim = len(model_atomic_numbers)
    weight = linear.weight.detach().cpu().reshape(node_attr_dim, hidden_dim)
    bias = linear.bias.detach().cpu() if torch.is_tensor(linear.bias) else None

    return {
        "node_atomic_numbers": node_atomic_numbers,
        "model_atomic_numbers": model_atomic_numbers,
        "type_indices_by_node": type_indices,
        "node_attr_dim": node_attr_dim,
        "hidden_dim": hidden_dim,
        "path_weight": 1.0 / math.sqrt(float(node_attr_dim)),
        "weight_rows": tensor_selected_rows(weight, unique_type_indices),
        "output_mask": tensor_exact_values(output_mask),
        "bias": tensor_exact_values(bias) if bias is not None else None,
    }


def summarize(value: Any, *, max_values: int = 8) -> Any:
    import torch

    if torch.is_tensor(value):
        return tensor_summary(value, max_values=max_values)
    if isinstance(value, dict):
        return {str(k): summarize(v, max_values=max_values) for k, v in value.items()}
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "len": len(value),
            "items": [summarize(v, max_values=max_values) for v in value],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "len": len(value),
            "items": [summarize(v, max_values=max_values) for v in value],
        }
    return clean_value(value)


def max_abs_diff(a: Any, b: Any) -> float:
    import torch

    if a is None or b is None:
        return float("nan")
    with torch.no_grad():
        lhs = a.detach().cpu()
        rhs = b.detach().cpu()
        if lhs.shape != rhs.shape:
            return float("inf")
        if lhs.numel() == 0:
            return 0.0
        return float((lhs - rhs).abs().max().item())


def build_atoms(spec: dict[str, Any]) -> Any:
    from ase import Atoms

    atoms = Atoms(
        symbols=spec["symbols"],
        positions=np.asarray(spec["positions"], dtype=float),
        pbc=spec["pbc"],
    )
    atoms.info["charge"] = float(spec["total_charge"])
    atoms.info["spin"] = float(spec["total_spin"])
    atoms.info["external_field"] = list(spec["external_field"])
    return atoms


def make_batch(atoms: Any, calc: Any, model: Any, dtype: Any) -> dict[str, Any]:
    from mace import data
    from mace.tools import torch_geometric

    config = data.config_from_atoms(
        atoms,
        key_specification=data.KeySpecification(
            info_keys={
                "total_spin": "spin",
                "total_charge": "charge",
                "external_field": "external_field",
            },
            arrays_keys={},
        ),
        head_name=model.heads[0],
    )
    atomic_data = data.AtomicData.from_config(
        config,
        z_table=calc.z_table,
        cutoff=float(model.r_max),
        heads=model.heads,
    )
    batch = next(
        iter(torch_geometric.dataloader.DataLoader([atomic_data], batch_size=1))
    ).to_dict()
    coerced = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            value = value.to("cpu")
            if getattr(value, "is_floating_point", lambda: False)():
                value = value.to(dtype=dtype)
        coerced[key] = value
    return coerced


def interaction_kwargs_dict(ctx: Any) -> dict[str, Any]:
    kwargs = getattr(ctx, "interaction_kwargs", {})
    if isinstance(kwargs, dict):
        return dict(kwargs)
    if hasattr(kwargs, "_asdict"):
        return dict(kwargs._asdict())
    if hasattr(kwargs, "__dict__"):
        return dict(vars(kwargs))
    return {}


def manual_backbone_replay(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    import torch
    from mace.modules.extensions import _permute_to_e3nn_convention
    from mace.modules.utils import prepare_graph
    from mace.tools.scatter import scatter_sum

    ctx = prepare_graph(
        batch,
        compute_virials=False,
        compute_stress=False,
        compute_displacement=False,
        lammps_mliap=False,
    )
    kwargs = interaction_kwargs_dict(ctx)
    is_lammps = bool(getattr(ctx, "is_lammps", False))
    lammps_natoms = kwargs.get("lammps_natoms")
    lammps_class = kwargs.get("lammps_class")

    num_atoms_arange = ctx.num_atoms_arange
    num_graphs = ctx.num_graphs
    node_heads = ctx.node_heads
    vectors = ctx.vectors
    lengths = ctx.lengths
    data = batch

    node_e0 = model.atomic_energies_fn(data["node_attrs"])[num_atoms_arange, node_heads]
    e0 = scatter_sum(
        src=node_e0,
        index=data["batch"],
        dim=0,
        dim_size=num_graphs,
    ).to(vectors.dtype)

    node_feats_initial = model.node_embedding(data["node_attrs"])
    edge_attrs = model.spherical_harmonics(_permute_to_e3nn_convention(vectors))
    edge_feats, cutoff = model.radial_embedding(
        lengths,
        data["node_attrs"],
        data["edge_index"],
        model.atomic_numbers,
    )

    if hasattr(model, "pair_repulsion"):
        pair_node_energy = model.pair_repulsion_fn(
            lengths,
            data["node_attrs"],
            data["edge_index"],
            model.atomic_numbers,
        )
        if is_lammps and lammps_natoms is not None:
            pair_node_energy = pair_node_energy[: lammps_natoms[0]]
    else:
        pair_node_energy = torch.zeros_like(node_e0)

    node_es_list = []
    node_feats_list = []
    layer_records = []
    node_feats = node_feats_initial

    for layer, (interaction, product) in enumerate(zip(model.interactions, model.products)):
        node_attrs_slice = data["node_attrs"]
        if is_lammps and layer > 0 and lammps_natoms is not None:
            node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]

        node_feats_in = node_feats
        local_boundary: dict[str, Any] = {}
        if hasattr(interaction, "skip_tp"):
            skip_tp_out = interaction.skip_tp(node_feats_in)
            local_boundary["skip_tp"] = linear_boundary_pin(
                interaction.skip_tp,
                node_feats_in,
                skip_tp_out,
            )
        linear_up_out = None
        linear_res_out = None
        conv_tp_scatter_message = None
        node_density = None
        if hasattr(interaction, "linear_up"):
            linear_up_out = interaction.linear_up(node_feats_in)
            local_boundary["linear_up"] = linear_boundary_pin(
                interaction.linear_up,
                node_feats_in,
                linear_up_out,
            )
            boundary_node_attrs = node_attrs_slice
            if is_lammps:
                node_feats_attrs = torch.cat(
                    [linear_up_out, node_attrs_slice],
                    dim=-1,
                )
                node_feats_attrs = interaction.handle_lammps(
                    node_feats_attrs,
                    lammps_class=lammps_class,
                    lammps_natoms=lammps_natoms,
                    first_layer=(layer == 0),
                )
                boundary_node_attrs = node_feats_attrs[:, linear_up_out.shape[-1] :]
            if hasattr(interaction, "source_embedding") and hasattr(
                interaction, "target_embedding"
            ):
                source_embedding_out = interaction.source_embedding(boundary_node_attrs)
                target_embedding_out = interaction.target_embedding(boundary_node_attrs)
                local_boundary["source_embedding"] = linear_boundary_pin(
                    interaction.source_embedding,
                    boundary_node_attrs,
                    source_embedding_out,
                )
                local_boundary["target_embedding"] = linear_boundary_pin(
                    interaction.target_embedding,
                    boundary_node_attrs,
                    target_embedding_out,
                )
                edge_aug = torch.cat(
                    [
                        edge_feats,
                        source_embedding_out[data["edge_index"][0]],
                        target_embedding_out[data["edge_index"][1]],
                    ],
                    dim=-1,
                )
                local_boundary["edge_aug"] = {
                    "description": "Radial MLP input immediately before conv_tp_weights.",
                    "rows": tensor_exact_rows(edge_aug),
                }
                if hasattr(interaction, "conv_tp_weights"):
                    conv_tp_weights_raw, conv_tp_weight_layers = sequential_boundary_pin(
                        interaction.conv_tp_weights,
                        edge_aug,
                        target_indices={0, 1, 3, 4, 6, 7, 9},
                    )
                    conv_tp_weights_cutoff_applied = (
                        conv_tp_weights_raw * cutoff if cutoff is not None else None
                    )
                    local_boundary["conv_tp_weights"] = {
                        "description": "Raw conv_tp_weights(edge_aug) before optional cutoff multiplication.",
                        "net_layers": conv_tp_weight_layers,
                        "raw_output": tensor_exact_rows(conv_tp_weights_raw),
                        "cutoff_applied_output": (
                            tensor_exact_rows(conv_tp_weights_cutoff_applied)
                            if conv_tp_weights_cutoff_applied is not None
                            else None
                        ),
                    }
                    if layer == 0 and hasattr(interaction, "conv_tp"):
                        conv_tp_source = linear_up_out[data["edge_index"][0]]
                        conv_tp_edge_messages = interaction.conv_tp(
                            conv_tp_source,
                            edge_attrs,
                            conv_tp_weights_cutoff_applied,
                        )
                        conv_tp_scatter_message = scatter_sum(
                            src=conv_tp_edge_messages,
                            index=data["edge_index"][1],
                            dim=0,
                            dim_size=int(node_feats_in.shape[0]),
                        )
                        conv_tp_output_mask = getattr(
                            interaction.conv_tp, "output_mask", None
                        )
                        local_boundary["conv_tp"] = {
                            "description": "interactions.0.conv_tp(linear_up[source], edge_attrs, cutoff_applied conv_tp_weights) and receiver scatter_sum over edge_index[1].",
                            "source_feature_name": "linear_up_out[edge_index[0]]",
                            "edge_attr_name": "edge_attrs",
                            "weight_name": "conv_tp_weights.cutoff_applied_output",
                            "scatter_index": "edge_index[1]",
                            "conv_tp_plan": tensor_product_plan_pin(
                                interaction.conv_tp, np, torch
                            ),
                            "output_mask": (
                                tensor_exact_values(conv_tp_output_mask)
                                if conv_tp_output_mask is not None
                                else None
                            ),
                            "edge_messages": tensor_exact_rows(conv_tp_edge_messages),
                            "scatter_message": tensor_exact_rows(
                                conv_tp_scatter_message
                            ),
                        }
                if hasattr(interaction, "density_fn"):
                    density_raw, density_layers = sequential_boundary_pin(
                        interaction.density_fn,
                        edge_aug,
                        target_indices={0, 1, 3},
                    )
                    edge_density = torch.tanh(density_raw**2)
                    edge_density_cutoff_applied = (
                        edge_density * cutoff if cutoff is not None else None
                    )
                    node_density = (
                        scatter_sum(
                            src=edge_density_cutoff_applied,
                            index=data["edge_index"][1],
                            dim=0,
                            dim_size=int(node_feats_in.shape[0]),
                        )
                        if edge_density_cutoff_applied is not None
                        else None
                    )
                    local_boundary["density_fn"] = {
                        "description": "Raw density_fn(edge_aug), post tanh(square) edge_density, optional cutoff-applied edge_density, and upstream density scatter to receiver/target nodes.",
                        "upstream_node_density_name": "density",
                        "upstream_node_density_description": "Upstream MACE computes density = scatter_sum(src=edge_density, index=edge_index[1], dim=0, dim_size=num_nodes) after applying cutoff to edge_density.",
                        "net_layers": density_layers,
                        "raw_output": tensor_exact_rows(density_raw),
                        "edge_density_post_tanh_square": tensor_exact_rows(edge_density),
                        "edge_density_cutoff_applied": (
                            tensor_exact_rows(edge_density_cutoff_applied)
                            if edge_density_cutoff_applied is not None
                            else None
                        ),
                        "node_density": (
                            tensor_exact_rows(node_density)
                            if node_density is not None
                            else None
                        ),
                    }
                if (
                    layer == 0
                    and conv_tp_scatter_message is not None
                    and node_density is not None
                    and linear_up_out is not None
                ):
                    required_post_scatter_attrs = [
                        "linear_res",
                        "linear_1",
                        "alpha",
                        "beta",
                        "equivariant_nonlin",
                        "linear_2",
                        "reshape",
                    ]
                    missing_attrs = [
                        attr
                        for attr in required_post_scatter_attrs
                        if not hasattr(interaction, attr)
                    ]
                    if missing_attrs:
                        raise RuntimeError(
                            "Cannot pin interactions.0 post-scatter boundary; "
                            f"missing upstream attributes: {missing_attrs}"
                        )

                    linear_res_out = interaction.linear_res(linear_up_out)
                    message_after_linear_1 = interaction.linear_1(
                        conv_tp_scatter_message
                    )
                    density_denominator = node_density * interaction.beta + interaction.alpha
                    message_after_density_norm = (
                        message_after_linear_1 / density_denominator
                    )
                    message_after_residual_add = (
                        message_after_density_norm + linear_res_out
                    )
                    (
                        gate_scalars_in,
                        gate_gates_in,
                        gate_gated_in,
                    ) = interaction.equivariant_nonlin.sc(message_after_residual_add)
                    gate_scalar_activation_out = (
                        interaction.equivariant_nonlin.act_scalars(gate_scalars_in)
                    )
                    gate_activation_out = interaction.equivariant_nonlin.act_gates(
                        gate_gates_in
                    )
                    gate_gated_product_out = interaction.equivariant_nonlin.mul(
                        gate_gated_in,
                        gate_activation_out,
                    )
                    equivariant_nonlin_out = interaction.equivariant_nonlin(
                        message_after_residual_add
                    )
                    linear_2_out = interaction.linear_2(equivariant_nonlin_out)
                    reshaped_interaction_out = interaction.reshape(linear_2_out)

                    local_boundary["linear_res"] = linear_boundary_pin(
                        interaction.linear_res,
                        linear_up_out,
                        linear_res_out,
                    )
                    local_boundary["linear_1"] = linear_boundary_pin(
                        interaction.linear_1,
                        conv_tp_scatter_message,
                        message_after_linear_1,
                    )
                    local_boundary["alpha"] = tensor_exact_values(interaction.alpha)
                    local_boundary["beta"] = tensor_exact_values(interaction.beta)
                    local_boundary["density_denominator"] = {
                        "description": "density * beta + alpha denominator used immediately after linear_1.",
                        "rows": tensor_exact_rows(density_denominator),
                    }
                    local_boundary["message_after_linear_1"] = tensor_exact_rows(
                        message_after_linear_1
                    )
                    local_boundary["message_after_density_norm"] = tensor_exact_rows(
                        message_after_density_norm
                    )
                    local_boundary["message_after_residual_add"] = tensor_exact_rows(
                        message_after_residual_add
                    )
                    local_boundary["equivariant_nonlin"] = {
                        "input_rows": tensor_exact_rows(message_after_residual_add),
                        "output_rows": tensor_exact_rows(equivariant_nonlin_out),
                        "gate_plan": gate_plan_metadata(
                            interaction.equivariant_nonlin,
                            message_after_residual_add,
                            gate_scalar_activation_out,
                            gate_activation_out,
                            gate_gated_product_out,
                        ),
                    }
                    local_boundary["linear_2"] = linear_boundary_pin(
                        interaction.linear_2,
                        equivariant_nonlin_out,
                        linear_2_out,
                    )
                    local_boundary["reshaped_interaction_out"] = tensor_exact_values(
                        reshaped_interaction_out
                    )
        node_feats, sc = interaction(
            node_attrs=node_attrs_slice,
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=data["edge_index"],
            cutoff=cutoff,
            first_layer=(layer == 0),
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
        )

        if is_lammps and layer == 0 and lammps_natoms is not None:
            node_attrs_slice = node_attrs_slice[: lammps_natoms[0]]

        interaction_out = node_feats
        node_feats, product_boundary = product_boundary_pin(
            product,
            node_feats,
            sc,
            node_attrs_slice,
        )
        local_boundary["product"] = product_boundary
        node_feats_list.append(node_feats)

        feat_idx = -1 if len(model.readouts) == 1 else min(layer, len(model.readouts) - 1)
        readout_index = feat_idx if feat_idx >= 0 else len(model.readouts) + feat_idx
        readout_out = model.readouts[feat_idx](node_feats, node_heads)
        node_es = readout_out[
            num_atoms_arange,
            node_heads,
        ]
        node_es_list.append(node_es)
        if layer == 0:
            local_boundary["readout"] = readout_boundary_pin(
                model.readouts[feat_idx],
                node_feats,
                node_heads,
                readout_out,
                node_es,
                readout_index=readout_index,
            )

        layer_records.append(
            {
                "layer": layer,
                "interaction_class": f"{type(interaction).__module__}.{type(interaction).__name__}",
                "product_class": f"{type(product).__module__}.{type(product).__name__}",
                "readout_class": f"{type(model.readouts[feat_idx]).__module__}.{type(model.readouts[feat_idx]).__name__}",
                "readout_index": feat_idx,
                "node_feats_in": tensor_summary(node_feats_in),
                "skip_connection": None if sc is None else tensor_summary(sc),
                "interaction_out": tensor_summary(interaction_out),
                "product_out": tensor_summary(node_feats),
                "node_es": tensor_summary(node_es),
                "local_backbone_boundary": local_boundary,
            }
        )

    node_feats_out = torch.cat(node_feats_list, dim=-1)
    node_inter_es_unscaled = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
    node_inter_es = model.scale_shift(node_inter_es_unscaled, node_heads)
    inter_e = scatter_sum(
        node_inter_es,
        data["batch"],
        dim=-1,
        dim_size=num_graphs,
    )
    if len(layer_records) == 1:
        layer_records[0]["local_backbone_boundary"]["scale_shift"] = (
            scale_shift_boundary_pin(
                model.scale_shift,
                node_inter_es_unscaled,
                node_heads,
                node_inter_es,
                inter_e,
            )
        )

    return {
        "ctx": ctx,
        "outputs": {
            "node_e0": node_e0,
            "e0": e0,
            "vectors": vectors,
            "lengths": lengths,
            "node_feats_initial": node_feats_initial,
            "edge_attrs": edge_attrs,
            "edge_feats": edge_feats,
            "cutoff": cutoff,
            "pair_node_energy": pair_node_energy,
            "node_feats_out": node_feats_out,
            "node_inter_es_unscaled": node_inter_es_unscaled,
            "node_inter_es": node_inter_es,
            "interaction_energy": inter_e,
        },
        "layers": layer_records,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = repo_root()
    mace_dir = (root / args.mace_dir).resolve()
    sys.path.insert(0, str(mace_dir))

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    import torch
    import mace
    from mace.calculators.mace import MACECalculator

    manifest_path = (root / args.manifest).resolve()
    manifest = read_json(manifest_path)
    artifact_path = (root / manifest["artifact_path"]).resolve()
    artifact_sha = sha256_file(artifact_path)
    expected_sha = manifest.get("sha256") or manifest.get("expected_sha256")
    if artifact_sha != expected_sha:
        raise RuntimeError("Artifact SHA256 does not match manifest")

    mace_commit = read_pinned_commit(mace_dir)
    if mace_commit != PINNED_MACE_COMMIT:
        raise RuntimeError(f"Unexpected MACE commit: {mace_commit}")
    if mace_commit != manifest["mace_commit"]:
        raise RuntimeError("MACE commit does not match manifest")

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.set_default_dtype(dtype)

    calc = MACECalculator(
        model_paths=str(artifact_path),
        model_type="PolarMACE",
        device="cpu",
        default_dtype=args.dtype,
    )
    model = calc.models[0]
    model.eval()

    spec = named_structure_spec(args.structure_name)
    atoms = build_atoms(spec)
    batch = make_batch(atoms, calc, model, dtype)

    with torch.no_grad():
        replay = manual_backbone_replay(model, dict(batch))
        forward_out = model(
            dict(batch),
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
        )

    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    calc_results = dict(calc.results)

    replay_outputs = replay["outputs"]
    manual_vs_forward = {
        "interaction_energy_max_abs_diff": max_abs_diff(
            replay_outputs["interaction_energy"], forward_out.get("interaction_energy")
        ),
        "node_feats_max_abs_diff": max_abs_diff(
            replay_outputs["node_feats_out"], forward_out.get("node_feats")
        ),
    }
    tolerance = 1.0e-5 if args.dtype == "float32" else 1.0e-10
    for name, diff in manual_vs_forward.items():
        if not math.isfinite(diff) or diff > tolerance:
            raise RuntimeError(f"Manual replay mismatch for {name}: {diff} > {tolerance}")

    output_keys = [str(k) for k in sorted(forward_out.keys())]
    calc_result_keys = [str(k) for k in sorted(calc_results.keys())]
    return {
        "schema_version": SCHEMA_VERSION,
        "architecture": "mace-polar-v1",
        "selected_model": clean_value(manifest.get("selected_model")),
        "runtime_status": BACKBONE_RUNTIME_STATUS,
        "normal_cargo_test_dependency": False,
        "artifact_manifest": str(Path(args.manifest)),
        "artifact_sha256_verified": artifact_sha,
        "artifact": clean_value(manifest),
        "reference": {
            "mace_repository": "https://github.com/ACEsuit/mace",
            "mace_commit": mace_commit,
            "mace_package_version": getattr(mace, "__version__", "unknown"),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dtype": args.dtype,
        },
        "conventions": {
            "edge_vector": "R_j + shift - R_i",
            "edge_index": "Upstream MACE edge_index[0] is sender/source and edge_index[1] is receiver/target.",
            "spherical_harmonics_input": "PolarMACE permutes vectors to e3nn coordinate convention with [y, z, x] before spherical harmonics.",
            "length_unit": "angstrom",
            "energy_unit": "eV",
            "force_unit": "eV/angstrom",
            "pbc": spec["pbc"],
        },
        "structure": {
            **spec,
            "energy": energy,
            "forces": clean_value(forces.tolist()),
        },
        "batch": {
            "edge_index": tensor_exact_values(batch["edge_index"]),
            "shifts": summarize(batch.get("shifts")),
            "unit_shifts": summarize(batch.get("unit_shifts")),
            "node_attrs": summarize(batch.get("node_attrs")),
            "batch": summarize(batch.get("batch")),
            "head": summarize(batch.get("head")),
            "total_charge": summarize(batch.get("total_charge")),
            "total_spin": summarize(batch.get("total_spin")),
            "external_field": summarize(batch.get("external_field")),
        },
        "manual_backbone_inputs": {
            "node_embedding": node_embedding_inputs(model, spec),
        },
        "manual_backbone_outputs": {
            key: (
                tensor_exact_values(value)
                if key in {"vectors", "node_inter_es", "node_inter_es_unscaled"}
                else tensor_exact_rows(value)
                if key in {"edge_attrs", "edge_feats", "node_feats_initial", "node_feats_out"}
                else tensor_summary(value)
            )
            for key, value in replay_outputs.items()
        },
        "manual_backbone_layers": replay["layers"],
        "manual_vs_forward": clean_value(manual_vs_forward),
        "forward_outputs": {
            "keys": output_keys,
            "energy": summarize(forward_out.get("energy")),
            "interaction_energy": summarize(forward_out.get("interaction_energy")),
            "node_feats": summarize(forward_out.get("node_feats")),
        },
        "calculator_result_keys": calc_result_keys,
        "calculator_results": {
            key: summarize(value) for key, value in calc_results.items()
        },
        "hook_targets": BACKBONE_HOOK_TARGETS,
        "captured_manual_layer_count": len(replay["layers"]),
        "captured_manual_output_keys": sorted(replay_outputs.keys()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "This fixture manually replays only the local MACE backbone inside upstream PolarMACE.",
            "It stops after interaction_energy and concatenated backbone node_feats.",
            "It does not replay graph_longrange density/electrostatics, fixed-point charge recursion, or electrostatic outputs.",
            "The Rust runtime is not considered implemented or parity-complete by this fixture alone.",
        ],
        "unreplayed_polarmace_sections": [
            "fermi_level/external_field/external_potential handling after backbone",
            "charges_to_mul_ir and spin_charge_density construction",
            "lr_source_maps and density_coefficients recursion",
            "compute_k_vectors_flat and reciprocal-cell long-range geometry",
            "electric_potential_descriptor.precompute_geometry/forward_dynamic locals",
            "layer_feature_mixer and fukui_source_map charge correction",
            "field_dependent_charges_maps fixed-point updates",
            "local_electron_energy, external_field_contribution, coulomb_energy, dipole",
            "force-only VJP through all PolarMACE energy terms",
        ],
    }


def parse_args() -> argparse.Namespace:
    root = repo_root()
    return argparse.Namespace(**vars(_parser(root).parse_args()))


def _parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="fixtures/mace/v0_3_16_4d2da09/polarmace_polar_1_s_artifact_manifest.json",
        help="Path to the pinned PolarMACE artifact manifest, relative to repo root.",
    )
    parser.add_argument(
        "--mace-dir",
        default="third_party/mace",
        help="Pinned upstream MACE checkout, relative to repo root.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
    )
    parser.add_argument(
        "--structure-name",
        default="water",
        help="Structure name from tools/reference/polarmace_fixture_structures.py.",
    )
    parser.add_argument(
        "--out",
        default=str(
            root
            / "fixtures/mace/v0_3_16_4d2da09/polarmace_polar_1_s_water_manual_backbone_replay.json"
        ),
        help="Output fixture path.",
    )
    return parser


def main() -> None:
    args = parse_args()
    payload = run(args)
    out = Path(args.out)
    if not out.is_absolute():
        out = repo_root() / out
    write_json(out, payload)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
