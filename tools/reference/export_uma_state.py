#!/usr/bin/env python3
"""Export UMA-S-1.2 checkpoint tensors into a pinned `.e3rs` handoff.

This is a reference-only converter. Rust runtime must read only the produced
`.e3rs` files and must not depend on Python, PyTorch, or FairChem.

When ``--write-tensors`` is set, the exporter writes per-tensor ``.bin`` files
and a single mmap-friendly ``tensors.pack`` consumed by the Rust runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any, Mapping

MODEL_KEY = "uma-s-1p2"
ARCHITECTURE = "uma-v1"
STATE_SCHEMA = "uma-s-1p2-state-dict-v1"
STATE_DICT_FORMAT = "manifest-f64-tensors-v1"
RUNTIME_STATUS = "state_dict_export_only_runtime_not_implemented"
CHECKPOINT_SHA256 = "ba5c0d912efa22dc238e5fb1b5b7f66ee2e68c48c1b95b7cfd7fe1da5938398b"
CHECKPOINT_SIZE_BYTES = 2_333_393_167
FAIRCHEM_VERSION = "2.20.0"
FAIRCHEM_COMMIT = "28292f090743f36c1151002964140b04d4b2c6af"
DEFAULT_CHECKPOINT = Path("outputs/reference-artifacts/fairchem/uma_s_1p2.pt")
DEFAULT_OUT = Path("outputs/reference-artifacts/fairchem/uma_s_1p2_state_export.e3rs")
UMA_TENSOR_PACK_FILENAME = "tensors.pack"
PACK_MAGIC = b"LLMLIPMPK1"
PACK_VERSION = 1
PACK_HEADER_START = 26
PACK_DTYPE_TAG_F64 = 1
PACK_DTYPE_TAG_I64 = 2
INFERENCE_STATE_PREFIX = "ema_state_dict.module."
RAW_STATE_PREFIX = "model_state_dict."
EMA_METADATA_TENSOR = "ema_state_dict.n_averaged"
ELEMENT_SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
    "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
    "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
    "U", "Np", "Pu",
]


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize_name(name: str) -> str:
    safe = []
    for ch in name:
        safe.append(ch if ch.isalnum() or ch in ".-_" else "_")
    return "".join(safe).strip("._") or "tensor"


def walk_tensors(obj: Any) -> list[tuple[str, Any]]:
    import torch

    found: list[tuple[str, Any]] = []
    seen: set[int] = set()

    def walk(value: Any, prefix: str) -> None:
        ident = id(value)
        if ident in seen:
            return
        seen.add(ident)
        if torch.is_tensor(value):
            found.append((prefix or "tensor", value.detach().cpu()))
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                walk(child, f"{prefix}.{key}" if prefix else str(key))
            return
        if isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                walk(child, f"{prefix}[{i}]")
            return
        for attr in ("state_dict",):
            if hasattr(value, attr):
                try:
                    child = getattr(value, attr)()
                except Exception:
                    continue
                walk(child, f"{prefix}.{attr}" if prefix else attr)
                return
        if hasattr(value, "__dict__"):
            for key, child in vars(value).items():
                if key.startswith("_"):
                    continue
                walk(child, f"{prefix}.{key}" if prefix else key)

    walk(obj, "")
    dedup: dict[str, Any] = {}
    for name, tensor in found:
        candidate = name
        suffix = 1
        while candidate in dedup:
            suffix += 1
            candidate = f"{name}#{suffix}"
        dedup[candidate] = tensor
    return sorted(dedup.items(), key=lambda item: item[0])


def export_dtype(tensor: Any) -> str:
    import torch

    if tensor.dtype.is_floating_point:
        return "f64"
    if tensor.dtype == torch.int64:
        return "i64"
    if tensor.dtype == torch.int32:
        return "i32"
    if tensor.dtype == torch.bool:
        return "bool"
    if tensor.dtype == torch.uint8:
        return "u8"
    raise TypeError(f"unsupported UMA tensor dtype {tensor.dtype}")


def tensor_bytes(tensor: Any, dtype: str) -> bytes:
    import numpy as np
    import torch

    if dtype == "f64":
        array = tensor.to(dtype=torch.float64).contiguous().numpy().astype("<f8", copy=False)
    elif dtype == "i64":
        array = tensor.to(dtype=torch.int64).contiguous().numpy().astype("<i8", copy=False)
    elif dtype == "i32":
        array = tensor.to(dtype=torch.int32).contiguous().numpy().astype("<i4", copy=False)
    elif dtype in {"bool", "u8"}:
        array = tensor.to(dtype=torch.uint8).contiguous().numpy().astype("u1", copy=False)
    else:
        raise TypeError(f"unsupported export dtype {dtype}")
    assert isinstance(array, np.ndarray)
    return array.tobytes(order="C")


def tensor_shape(tensor: Any) -> list[int]:
    return [int(dim) for dim in tensor.shape] or [1]


def tensor_state_role(name: str) -> str:
    if name.startswith(INFERENCE_STATE_PREFIX):
        return "ema"
    if name == EMA_METADATA_TENSOR:
        return "ema_metadata"
    if name.startswith(RAW_STATE_PREFIX):
        return "raw"
    return "unknown"


def tensor_inference_key(name: str) -> str | None:
    if name.startswith(INFERENCE_STATE_PREFIX):
        return name[len(INFERENCE_STATE_PREFIX) :]
    return None


def tensor_semantic_group(name: str) -> str:
    key = tensor_inference_key(name)
    if key is None and name.startswith(RAW_STATE_PREFIX):
        key = name[len(RAW_STATE_PREFIX) :]
    if key is None:
        return "swa_metadata" if name == EMA_METADATA_TENSOR else "unknown"

    if key.startswith("backbone.blocks."):
        parts = key.split(".")
        if len(parts) >= 3 and parts[2].isdigit():
            return f"backbone_block_{parts[2]}"
        return "backbone_blocks"
    if key.startswith("backbone.edge_degree_embedding."):
        return "edge_degree_embedding"
    if key.startswith("backbone.dataset_embedding."):
        return "dataset_embedding"
    if key.startswith("backbone.routing_mlp."):
        return "routing_mlp"
    if key.startswith("output_heads.energyandforcehead."):
        return "energy_force_head"
    if key.startswith("backbone."):
        return "backbone_global"
    return "unknown"


def to_jsonable(value: Any) -> Any:
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except Exception:
        pass
    else:
        if isinstance(value, (DictConfig, ListConfig)):
            return to_jsonable(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if hasattr(value, "__dict__"):
        return {
            str(key): to_jsonable(child)
            for key, child in vars(value).items()
            if not key.startswith("_")
        }
    return repr(value)


def tensor_numel(shape: list[int]) -> int:
    return int(math.prod(shape))


def pack_dtype_tag(dtype: str) -> int:
    if dtype == "f64":
        return PACK_DTYPE_TAG_F64
    if dtype == "i64":
        return PACK_DTYPE_TAG_I64
    raise TypeError(
        f"write_uma_tensor_pack: unsupported dtype {dtype!r}; supported: f64, i64"
    )


def align_u64(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def write_pack_index_entry(
    out: bytearray,
    entry: dict[str, Any],
    data_rel_offset: int,
    byte_len: int,
) -> None:
    name = str(entry["name"])
    name_bytes = name.encode("utf-8")
    out.extend(struct.pack("<I", len(name_bytes)))
    out.extend(name_bytes)
    out.append(pack_dtype_tag(str(entry["exported_dtype"])))
    shape = [int(dim) for dim in entry["shape"]]
    out.extend(struct.pack("<I", len(shape)))
    for dim in shape:
        out.extend(struct.pack("<Q", dim))
    out.extend(struct.pack("<Q", data_rel_offset))
    out.extend(struct.pack("<Q", byte_len))


def write_uma_tensor_pack(out_dir: Path, entries: list[dict[str, Any]], payloads: list[bytes]) -> Path:
    """manifest 順のテンソル payload から ``tensors.pack`` を書き出す。"""
    if len(entries) != len(payloads):
        raise ValueError(
            "write_uma_tensor_pack: entries/payloads length mismatch "
            f"({len(entries)} vs {len(payloads)})"
        )
    index_bytes = bytearray()
    data_bytes = bytearray()
    for entry, payload in zip(entries, payloads, strict=True):
        expected_bytes = int(entry["bytes"])
        if len(payload) != expected_bytes:
            raise ValueError(
                "write_uma_tensor_pack: payload byte mismatch for "
                f"{entry['name']!r}: expected {expected_bytes}, got {len(payload)}"
            )
        data_rel_offset = len(data_bytes)
        write_pack_index_entry(index_bytes, entry, data_rel_offset, len(payload))
        data_bytes.extend(payload)

    data_offset = align_u64(PACK_HEADER_START + len(index_bytes), 8)
    padding_len = data_offset - PACK_HEADER_START - len(index_bytes)
    file_bytes = bytearray()
    file_bytes.extend(PACK_MAGIC)
    file_bytes.extend(struct.pack("<I", PACK_VERSION))
    file_bytes.extend(struct.pack("<I", len(entries)))
    file_bytes.extend(struct.pack("<Q", data_offset))
    file_bytes.extend(index_bytes)
    file_bytes.extend(b"\x00" * padding_len)
    if len(file_bytes) != data_offset:
        raise RuntimeError(
            "write_uma_tensor_pack: internal layout error: "
            f"header_end={len(file_bytes)} data_offset={data_offset}"
        )
    file_bytes.extend(data_bytes)

    pack_path = out_dir / UMA_TENSOR_PACK_FILENAME
    pack_path.write_bytes(file_bytes)
    return pack_path


def write_manifest(out_dir: Path, tensors: list[dict[str, Any]], metadata_only: bool) -> None:
    manifest = [
        'format_version = "0.1"',
        f'architecture = "{ARCHITECTURE}"',
        'dtype = "f64"',
        'length_unit = "angstrom"',
        'energy_unit = "eV"',
        'force_unit = "eV/angstrom"',
        '',
        '[species]',
        'symbols = [' + ", ".join(json.dumps(symbol) for symbol in ELEMENT_SYMBOLS) + ']',
        '',
        '[geometry]',
        'cutoff = 6.0',
        'edge_vector = "R_j + shift - R_i"',
        'directed_edges = true',
        '',
        '[spherical_harmonics]',
        'l_max = 2',
        'normalize_input = true',
        'normalization = "component"',
        'reference_convention = "pinned-fixture"',
        '',
        '[uma]',
        'state = "uma_state.json"',
        'reference_package = "fairchem"',
        f'reference_version = "{FAIRCHEM_VERSION}"',
        f'reference_commit = "{FAIRCHEM_COMMIT}"',
        f'selected_model = "{MODEL_KEY}"',
        f'state_schema = "{STATE_SCHEMA}"',
        f'artifact_sha256 = "{CHECKPOINT_SHA256}"',
        f'artifact_size_bytes = {CHECKPOINT_SIZE_BYTES}',
        '',
        '[state_dict]',
        f'format = "{STATE_DICT_FORMAT}"',
    ]
    if metadata_only:
        manifest.append('# Tensor payload files were not written; this manifest is for inventory checks only.')
    else:
        manifest.append(f'# Runtime tensor payloads are shipped as mmap {UMA_TENSOR_PACK_FILENAME}.')
    for entry in tensors:
        manifest.extend([
            '',
            '[[tensors]]',
            f'name = {json.dumps(entry["name"])}',
            f'path = {json.dumps(entry["file"])}',
            f'dtype = {json.dumps(entry["exported_dtype"])}',
            'shape = [' + ", ".join(str(dim) for dim in entry["shape"]) + ']',
            f'sha256 = {json.dumps(entry["sha256"])}',
        ])
    out_dir.joinpath("manifest.toml").write_text("\n".join(manifest) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--write-tensors", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint_path}")
    if args.checkpoint_path.stat().st_size != CHECKPOINT_SIZE_BYTES:
        raise SystemExit("checkpoint byte size mismatch")
    if digest_file(args.checkpoint_path) != CHECKPOINT_SHA256:
        raise SystemExit("checkpoint SHA256 mismatch")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise SystemExit(f"output directory is non-empty; pass --force: {args.out_dir}")

    import torch

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir = args.out_dir / "tensors"
    if args.write_tensors:
        tensor_dir.mkdir(parents=True, exist_ok=True)

    obj = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    tensors = walk_tensors(obj)
    entries: list[dict[str, Any]] = []
    payloads: list[bytes] = []
    dtype_counts: dict[str, int] = {}
    active_semantic_group_counts: dict[str, int] = {}
    active_semantic_group_numel: dict[str, int] = {}
    total_numel = 0
    total_bytes = 0
    active_total_numel = 0
    active_total_bytes = 0

    for index, (name, tensor) in enumerate(tensors):
        dtype = export_dtype(tensor)
        shape = tensor_shape(tensor)
        numel = tensor_numel(shape)
        file_name = f"tensors/{index:05d}_{sanitize_name(name)}.{dtype}.bin"
        payload = tensor_bytes(tensor, dtype)
        sha256 = hashlib.sha256(payload).hexdigest()
        inference_key = tensor_inference_key(name)
        active_for_inference = inference_key is not None
        semantic_group = tensor_semantic_group(name)
        if args.write_tensors:
            (args.out_dir / file_name).write_bytes(payload)
        payloads.append(payload)
        entries.append({
            "name": name,
            "file": file_name,
            "exported_dtype": dtype,
            "source_dtype": str(tensor.dtype),
            "shape": shape,
            "numel": numel,
            "bytes": len(payload),
            "sha256": sha256,
            "state_role": tensor_state_role(name),
            "active_for_inference": active_for_inference,
            "inference_key": inference_key,
            "semantic_group": semantic_group,
        })
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        total_numel += numel
        total_bytes += len(payload)
        if active_for_inference:
            active_total_numel += numel
            active_total_bytes += len(payload)
            active_semantic_group_counts[semantic_group] = (
                active_semantic_group_counts.get(semantic_group, 0) + 1
            )
            active_semantic_group_numel[semantic_group] = (
                active_semantic_group_numel.get(semantic_group, 0) + numel
            )

    state = {
        "schema_version": STATE_SCHEMA,
        "selected_model": MODEL_KEY,
        "state_dict_format": STATE_DICT_FORMAT,
        "reference": {
            "package": "fairchem-core",
            "version": FAIRCHEM_VERSION,
            "commit": FAIRCHEM_COMMIT,
        },
        "artifact": {
            "path": str(args.checkpoint_path),
            "sha256": CHECKPOINT_SHA256,
            "size_bytes": CHECKPOINT_SIZE_BYTES,
        },
        "execution_contract": {
            "status": RUNTIME_STATUS,
            "runtime_note": "UMA Rust runtime graph and force-only VJP are not implemented by this exporter.",
        },
        "inference_state": {
            "source": "ema_state_dict.module",
            "checkpoint_loader": "MLIPPredictUnit load_inference_model(use_ema=True)",
            "active_tensor_count": sum(1 for entry in entries if entry["active_for_inference"]),
            "active_total_numel": active_total_numel,
            "active_total_bytes": active_total_bytes,
        },
        "tensor_summary": {
            "count": len(entries),
            "total_numel": total_numel,
            "total_bytes": total_bytes,
            "dtype_counts": dtype_counts,
            "active_inference_count": sum(
                1 for entry in entries if entry["active_for_inference"]
            ),
            "active_inference_numel": active_total_numel,
            "active_inference_bytes": active_total_bytes,
            "active_semantic_group_counts": active_semantic_group_counts,
            "active_semantic_group_numel": active_semantic_group_numel,
        },
        "model_config": to_jsonable(getattr(obj, "model_config", None)),
        "tasks_config": to_jsonable(getattr(obj, "tasks_config", None)),
        "tensors": entries,
    }
    (args.out_dir / "uma_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    write_manifest(args.out_dir, entries, metadata_only=not args.write_tensors)
    if args.write_tensors:
        pack_path = write_uma_tensor_pack(args.out_dir, entries, payloads)
        print(f"wrote {pack_path}", file=sys.stderr)
    print(json.dumps(state["tensor_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
