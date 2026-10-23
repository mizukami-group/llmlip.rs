#!/usr/bin/env python3
"""Export pinned NequiX `.nqx` artifacts into a narrow `.e3rs` state bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ELEMENT_SYMBOLS = [
    "",
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
]


def checked_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.output.strip()) from exc


def validate_nequix_tree(nequix_dir: Path) -> str:
    if not nequix_dir.exists():
        raise SystemExit(f"missing pinned NequiX reference directory: {nequix_dir}")
    commit = checked_output(["git", "-C", str(nequix_dir), "rev-parse", "HEAD"])
    pin_path = nequix_dir.with_name(f"{nequix_dir.name}.PINNED_COMMIT")
    if not pin_path.exists():
        raise SystemExit(f"missing pinned commit marker: {pin_path}")
    pinned_commit = pin_path.read_text(encoding="utf-8").strip()
    if commit != pinned_commit:
        raise SystemExit(
            "NequiX reference checkout does not match pinned commit marker\n"
            f"checkout: {commit}\nmarker:   {pinned_commit}"
        )
    return commit


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def hash_file(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def path_key_to_string(key: Any) -> str:
    if hasattr(key, "name"):
        return str(key.name)
    if hasattr(key, "idx"):
        return str(key.idx)
    if hasattr(key, "key"):
        return str(key.key)
    return str(key)


def tree_path_to_string(path: tuple[Any, ...]) -> str:
    return ".".join(path_key_to_string(key) for key in path)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


_TP_PLAN_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}


def irreps_segments(irreps: Any) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    start = 0
    for term_index, mul_ir in enumerate(irreps):
        mul = int(mul_ir.mul)
        irrep = mul_ir.ir
        dim = int(irrep.dim)
        for mul_index in range(mul):
            segments.append(
                {
                    "term_index": term_index,
                    "mul_index": mul_index,
                    "irrep": str(irrep),
                    "start": start,
                    "dim": dim,
                }
            )
            start += dim
    return segments


def tensor_product_plan_spec(
    e3nn: Any,
    irreps_in1_text: str,
    irreps_in2_text: str,
    filter_ir_out_text: str,
) -> dict[str, Any]:
    cache_key = (irreps_in1_text, irreps_in2_text, filter_ir_out_text)
    if cache_key in _TP_PLAN_CACHE:
        return _TP_PLAN_CACHE[cache_key]

    irreps_in1 = e3nn.Irreps(irreps_in1_text)
    irreps_in2 = e3nn.Irreps(irreps_in2_text)
    filter_ir_out = e3nn.Irreps(filter_ir_out_text)
    probe = e3nn.tensor_product(
        e3nn.IrrepsArray(irreps_in1, np.zeros(irreps_in1.dim, dtype=np.float64)),
        e3nn.IrrepsArray(irreps_in2, np.zeros(irreps_in2.dim, dtype=np.float64)),
        filter_ir_out=filter_ir_out,
    )
    irreps_out = probe.irreps
    in1_segments = irreps_segments(irreps_in1)
    in2_segments = irreps_segments(irreps_in2)
    out_segments = irreps_segments(irreps_out)
    out_lookup: dict[int, tuple[int, int]] = {}
    for segment_index, segment in enumerate(out_segments):
        for component in range(segment["dim"]):
            out_lookup[segment["start"] + component] = (segment_index, component)

    entries_by_out: dict[int, tuple[int, int, list[tuple[int, int, int, float]]]] = {}
    for in1_index, in1_segment in enumerate(in1_segments):
        for in2_index, in2_segment in enumerate(in2_segments):
            pairs = [
                (i_component, j_component)
                for i_component in range(in1_segment["dim"])
                for j_component in range(in2_segment["dim"])
            ]
            x = np.zeros((len(pairs), irreps_in1.dim), dtype=np.float64)
            y = np.zeros((len(pairs), irreps_in2.dim), dtype=np.float64)
            for row, (i_component, j_component) in enumerate(pairs):
                x[row, in1_segment["start"] + i_component] = 1.0
                y[row, in2_segment["start"] + j_component] = 1.0
            z = np.asarray(
                e3nn.tensor_product(
                    e3nn.IrrepsArray(irreps_in1, x),
                    e3nn.IrrepsArray(irreps_in2, y),
                    filter_ir_out=filter_ir_out,
                ).array,
                dtype=np.float64,
            )
            pair_entries: dict[int, list[tuple[int, int, int, float]]] = {}
            for row, (i_component, j_component) in enumerate(pairs):
                nonzero = np.flatnonzero(np.abs(z[row]) > 1.0e-12)
                for flat_index in nonzero:
                    out_index, k_component = out_lookup[int(flat_index)]
                    pair_entries.setdefault(out_index, []).append(
                        (
                            i_component,
                            j_component,
                            k_component,
                            float(z[row, flat_index]),
                        )
                    )
            for out_index, entries in pair_entries.items():
                if out_index in entries_by_out:
                    raise SystemExit(
                        "duplicate NequiX tensor-product output segment "
                        f"{out_index} for {irreps_in1_text} x {irreps_in2_text}"
                    )
                entries_by_out[out_index] = (in1_index, in2_index, entries)

    missing = [index for index in range(len(out_segments)) if index not in entries_by_out]
    if missing:
        raise SystemExit(
            "NequiX tensor-product plan generation missed output segments "
            f"{missing[:8]} for {irreps_in1_text} x {irreps_in2_text}"
        )

    instruction_in1_segment: list[int] = []
    instruction_in2_segment: list[int] = []
    instruction_out_segment: list[int] = []
    instruction_cg_offset: list[int] = []
    instruction_cg_len: list[int] = []
    instruction_weight_offset: list[int] = []
    instruction_path_weight: list[float] = []
    cg_i_m: list[int] = []
    cg_j_m: list[int] = []
    cg_k_m: list[int] = []
    cg_value: list[float] = []

    for out_index in range(len(out_segments)):
        in1_index, in2_index, entries = entries_by_out[out_index]
        instruction_in1_segment.append(in1_index)
        instruction_in2_segment.append(in2_index)
        instruction_out_segment.append(out_index)
        instruction_cg_offset.append(len(cg_value))
        instruction_cg_len.append(len(entries))
        instruction_weight_offset.append(out_index)
        instruction_path_weight.append(1.0)
        for i_component, j_component, k_component, value in entries:
            cg_i_m.append(i_component)
            cg_j_m.append(j_component)
            cg_k_m.append(k_component)
            cg_value.append(value)

    spec = {
        "schema_version": "e3nn-core-tensor-product-plan-v1",
        "generation": "sampled from pinned e3nn-jax tensor_product basis vectors",
        "irreps_in1": str(irreps_in1),
        "irreps_in2": str(irreps_in2),
        "irreps_out": str(irreps_out),
        "instruction_in1_segment": instruction_in1_segment,
        "instruction_in2_segment": instruction_in2_segment,
        "instruction_out_segment": instruction_out_segment,
        "instruction_cg_offset": instruction_cg_offset,
        "instruction_cg_len": instruction_cg_len,
        "instruction_weight_offset": instruction_weight_offset,
        "instruction_path_weight": instruction_path_weight,
        "cg_i_m": cg_i_m,
        "cg_j_m": cg_j_m,
        "cg_k_m": cg_k_m,
        "cg_value": cg_value,
    }
    _TP_PLAN_CACHE[cache_key] = spec
    return spec


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plan_tensor_entry(tensor: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "index": tensor["index"],
        "name": tensor["name"],
        "tree_path": tensor["tree_path"],
        "path": tensor["path"],
        "dtype": tensor["dtype"],
        "shape": tensor["shape"],
        "sha256": tensor["sha256"],
    }
    if "] " in tensor["tree_path"]:
        irreps = tensor["tree_path"].split("] ", 1)[1].split(",", 1)
        if len(irreps) == 2:
            entry["irreps_in"] = irreps[0]
            entry["irreps_out"] = irreps[1]
    return entry


def grouped_layer_tensors(tensors: list[dict[str, Any]]) -> dict[int, dict[str, list[dict[str, Any]]]]:
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for tensor in tensors:
        path = tensor["tree_path"]
        if not path.startswith("layers."):
            continue
        parts = path.split(".", 3)
        if len(parts) < 4:
            continue
        layer_index = int(parts[1])
        block = parts[2]
        if block == "radial_mlp":
            key = "radial_mlp"
        elif block == "linear_1":
            key = "linear_1"
        elif block == "linear_2":
            key = "linear_2"
        elif block == "skip":
            key = "skip"
        elif block == "layer_norm":
            key = "layer_norm"
        else:
            key = "other"
        grouped.setdefault(layer_index, {}).setdefault(key, []).append(tensor)
    for layer in grouped.values():
        for entries in layer.values():
            entries.sort(key=lambda value: value["index"])
    return grouped


def build_execution_plan(
    selected_model: str,
    model_config: dict[str, Any],
    layer_metadata: list[dict[str, Any]],
    tensors: list[dict[str, Any]],
    e3nn: Any,
) -> dict[str, Any]:
    grouped = grouped_layer_tensors(tensors)
    atom_energies = [tensor for tensor in tensors if tensor["tree_path"] == "atom_energies"]
    readout = [tensor for tensor in tensors if tensor["tree_path"].startswith("readout.")]
    layers = []
    for metadata in layer_metadata:
        index = metadata["index"]
        layer_tensors = grouped.get(index, {})
        radial_weights = layer_tensors.get("radial_mlp", [])
        radial_weight_numel = (
            int(radial_weights[-1]["shape"][1]) if radial_weights and len(radial_weights[-1]["shape"]) == 2 else 0
        )
        layers.append(
            {
                **metadata,
                "input_radial_basis": {
                    "basis": "bessel",
                    "basis_size": int(model_config["radial_basis_size"]),
                    "cutoff": float(model_config["cutoff"]),
                    "cutoff_function": "polynomial",
                    "polynomial_p": float(model_config["radial_polynomial_p"]),
                },
                "spherical_harmonics_irreps": "1x0e+1x1o+1x2e+1x3o",
                "radial_mlp": {
                    "activation": "silu",
                    "use_bias": False,
                    "weights": [plan_tensor_entry(tensor) for tensor in radial_weights],
                    "output_weight_numel": radial_weight_numel,
                },
                "linear_1": {
                    "weights": [
                        plan_tensor_entry(tensor)
                        for tensor in layer_tensors.get("linear_1", [])
                    ],
                },
                "tensor_product": {
                    "shared_weights": False,
                    "internal_weights": False,
                    "kernel": False,
                    "filter_ir_out": metadata["tp_irreps"],
                    "radial_weight_numel": radial_weight_numel,
                    "scatter": "sum_messages_by_receiver_then_divide_sqrt_avg_n_neighbors",
                    "plan": tensor_product_plan_spec(
                        e3nn,
                        metadata["linear_1_output_irreps"],
                        "1x0e+1x1o+1x2e+1x3o",
                        metadata["tp_irreps"],
                    ),
                },
                "linear_2": {
                    "weights": [
                        plan_tensor_entry(tensor)
                        for tensor in layer_tensors.get("linear_2", [])
                    ],
                },
                "skip": {
                    "linear_type": "vanilla",
                    "weights": [
                        plan_tensor_entry(tensor)
                        for tensor in layer_tensors.get("skip", [])
                    ],
                },
                "layer_norm": {
                    "kind": "rms",
                    "centering": False,
                    "std_balance_degrees": True,
                    "affine_weights": [
                        plan_tensor_entry(tensor)
                        for tensor in layer_tensors.get("layer_norm", [])
                    ],
                },
                "gate": {
                    "even_act": "silu",
                    "odd_act": "tanh",
                    "even_gate_act": "silu",
                },
            }
        )

    return {
        "schema_version": "nequix-execution-plan-v1",
        "selected_model": selected_model,
        "scope": "fixed NequiX-PFT f64 Rust runtime handoff; not an arbitrary .nqx plan",
        "atom_energies": [plan_tensor_entry(tensor) for tensor in atom_energies],
        "layers": layers,
        "readout": {
            "weights": [plan_tensor_entry(tensor) for tensor in readout],
        },
        "energy_tail": {
            "node_energy": "readout(last_features.scalar_channels)",
            "total_energy": "sum((node_energy * scale + shift) + atom_energies[species])",
        },
    }


def toml_string_array(values: list[str]) -> str:
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"


def element_symbol(atomic_number: int) -> str:
    if atomic_number <= 0 or atomic_number >= len(ELEMENT_SYMBOLS):
        raise SystemExit(f"unsupported NequiX atomic number in export: {atomic_number}")
    return ELEMENT_SYMBOLS[atomic_number]


def write_manifest_toml(out_dir: Path, state: dict[str, Any]) -> None:
    config = state["model_config"]
    symbols = [element_symbol(int(value)) for value in config["atomic_numbers"]]
    reference = state["reference"]
    artifact = state["artifact"]
    selected_model = state["selected_model"]
    manifest = f"""format_version = "0.1"
architecture = "nequix-v1"
dtype = "f64"
length_unit = "angstrom"
energy_unit = "eV"
force_unit = "eV/angstrom"

[species]
symbols = {toml_string_array(symbols)}

[geometry]
cutoff = {float(config["cutoff"]):.16g}
edge_vector = "R_j + shift - R_i"
directed_edges = true

[spherical_harmonics]
l_max = {int(config["lmax"])}
normalize_input = true
normalization = "component"
reference_convention = "pinned-fixture"

[nequix]
model = "nequix_state.json"
reference_package = "{reference['name']}"
reference_version = "{reference['package_version']}"
reference_commit = "{reference['commit']}"
selected_model = "{selected_model}"
state_schema = "nequix-state-dict-v1"
artifact_sha256 = "{artifact['sha256']}"
artifact_size_bytes = {artifact['size_bytes']}
"""
    (out_dir / "manifest.toml").write_text(manifest, encoding="utf-8")


def prepare_out_dir(out_dir: Path, force: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise SystemExit(f"output directory is not empty; pass --force to replace: {out_dir}")
        shutil.rmtree(out_dir)
    (out_dir / "weights").mkdir(parents=True, exist_ok=True)
    (out_dir / "plans").mkdir(parents=True, exist_ok=True)


def export_state(manifest: dict[str, Any], nequix_dir: Path, out_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(nequix_dir))
    import e3nn_jax as e3nn  # type: ignore[import-not-found]
    import jax.tree_util as jtu  # type: ignore[import-not-found]
    from nequix.model import load_model  # type: ignore[import-not-found]

    artifact = manifest["artifact"]
    artifact_path = Path(artifact["cache_path"])
    if not artifact_path.exists():
        raise SystemExit(f"cached artifact missing: {artifact_path}")
    sha256, size = hash_file(artifact_path)
    if sha256 != artifact["sha256"] or size != artifact["size_bytes"]:
        raise SystemExit(
            "artifact hash/size mismatch\n"
            f"expected sha={artifact['sha256']} size={artifact['size_bytes']}\n"
            f"actual   sha={sha256} size={size}"
        )

    model, config = load_model(str(artifact_path), kernel=False)
    leaves_with_path, treedef = jtu.tree_flatten_with_path(model)
    tensors: list[dict[str, Any]] = []
    dtype_counts: dict[str, int] = {}
    total_numel = 0
    total_bytes_f64 = 0

    for index, (path, leaf) in enumerate(leaves_with_path):
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            continue
        source = np.ascontiguousarray(np.asarray(leaf))
        source_dtype = str(source.dtype)
        exported = np.ascontiguousarray(source.astype("<f8", copy=False))
        payload = exported.tobytes()
        weight_path = Path("weights") / f"tensor_{len(tensors):04d}.f64.bin"
        (out_dir / weight_path).write_bytes(payload)

        numel = int(exported.size)
        nbytes = int(len(payload))
        dtype_counts[source_dtype] = dtype_counts.get(source_dtype, 0) + 1
        total_numel += numel
        total_bytes_f64 += nbytes
        tensors.append(
            {
                "index": len(tensors),
                "name": f"tensor_{len(tensors):04d}",
                "tree_path": tree_path_to_string(path),
                "source_dtype": source_dtype,
                "source_shape": list(source.shape),
                "source_numel": int(source.size),
                "source_nbytes": int(source.nbytes),
                "source_sha256": hash_bytes(source.tobytes()),
                "dtype": "f64",
                "shape": list(exported.shape),
                "layout": "row-major",
                "path": weight_path.as_posix(),
                "numel": numel,
                "nbytes": nbytes,
                "sha256": hash_bytes(payload),
            }
        )

    layers = []
    for index, layer in enumerate(model.layers):
        layers.append(
            {
                "index": index,
                "output_irreps": str(layer.output_irreps),
                "linear_1_output_irreps": str(layer.linear_1.irreps_out),
                "tp_irreps": str(layer.tp_irreps),
                "avg_n_neighbors": float(layer.avg_n_neighbors),
                "index_weights": bool(layer.index_weights),
                "layer_norm": layer.layer_norm is not None,
                "kernel_enabled": bool(layer.kernel),
            }
        )

    plan = build_execution_plan(manifest["selected_model"], config, layers, tensors, e3nn)
    write_json(out_dir / "plans" / "nequix_plan.json", plan)

    return {
        "schema_version": "nequix-state-dict-v1",
        "architecture": "nequix-v1",
        "selected_model": manifest["selected_model"],
        "reference": manifest["reference"],
        "artifact": {
            "filename": artifact["filename"],
            "sha256": sha256,
            "size_bytes": size,
            "source_url": manifest["source"]["url"],
        },
        "source": manifest["source"],
        "python": {
            "version": platform.python_version(),
            "executable_name": Path(sys.executable).name,
            "packages": {
                "jax": package_version("jax"),
                "jaxlib": package_version("jaxlib"),
                "equinox": package_version("equinox"),
                "e3nn-jax": package_version("e3nn-jax"),
                "jraph": package_version("jraph"),
                "numpy": package_version("numpy"),
            },
        },
        "model_config": config,
        "runtime_contract": {
            "kernel": False,
            "source_tree": "jax-pytree",
            "state_dtype": "f64",
            "runtime_arithmetic_dtype": "f64",
            "unsupported": [
                "training",
                "generic_ad",
                "parameter_gradients",
                "gpu_kernels",
                "arbitrary_nqx_loading",
            ],
        },
        "plans": [{"name": "nequix_plan", "path": "plans/nequix_plan.json"}],
        "pytree": {
            "treedef": str(treedef),
            "source_dtype_counts": dtype_counts,
            "tensor_count": len(tensors),
            "total_numel": total_numel,
            "total_bytes_f64": total_bytes_f64,
        },
        "tensors": tensors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nequix-dir", type=Path, default=Path("third_party/nequix"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    validate_nequix_tree(args.nequix_dir)
    manifest = load_json(args.manifest)
    prepare_out_dir(args.out_dir, args.force)
    state = export_state(manifest, args.nequix_dir, args.out_dir)
    write_manifest_toml(args.out_dir, state)
    write_json(args.out_dir / "nequix_state.json", state)
    print(
        "wrote "
        f"{args.out_dir} tensors={state['pytree']['tensor_count']} "
        f"numel={state['pytree']['total_numel']} "
        f"bytes_f64={state['pytree']['total_bytes_f64']}"
    )


if __name__ == "__main__":
    main()
