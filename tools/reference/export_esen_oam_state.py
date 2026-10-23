#!/usr/bin/env python3
"""Export eSEN-OAM checkpoint tensors into an `.e3rs` state bundle.

This exporter is reference tooling. It converts checkpoint tensors to a stable
binary payload plus JSON/TOML metadata; it does not imply Rust runtime support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "esen-oam-state-dict-v1"
STATE_DICT_FORMAT = "manifest-f64-tensors-v1"
MODEL_KEY = "esen-30m-oam"
REPOSITORY = "https://github.com/facebookresearch/fairchem.git"
CHEMICAL_SYMBOLS_Z1_TO_Z94 = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu",
]


def checked_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.output.strip()) from exc


def validate_fairchem_tree(fairchem_dir: Path) -> str:
    if not fairchem_dir.exists():
        raise SystemExit(f"missing pinned FairChem reference directory: {fairchem_dir}")
    commit = checked_output(["git", "-C", str(fairchem_dir), "rev-parse", "HEAD"])
    pin_path = fairchem_dir.with_name(f"{fairchem_dir.name}.PINNED_COMMIT")
    if not pin_path.exists():
        raise SystemExit(f"missing pinned commit marker: {pin_path}")
    pinned_commit = pin_path.read_text(encoding="utf-8").strip()
    if commit != pinned_commit:
        raise SystemExit(
            "FairChem reference checkout does not match pinned commit marker\n"
            f"checkout: {commit}\nmarker:   {pinned_commit}"
        )
    return commit


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def validate_manifest_artifact_identity(
    artifact_manifest: dict[str, Any] | None,
    checkpoint_sha256: str,
    checkpoint_size: int,
) -> None:
    if artifact_manifest is None:
        return
    scope = artifact_manifest.get("scope")
    if isinstance(scope, dict) and scope.get("status") == "identity_mismatch":
        raise SystemExit(
            "artifact manifest reports checkpoint identity mismatch; refusing to export eSEN-OAM state\n"
            f"identity_warnings: {scope.get('identity_warnings')!r}"
        )
    metadata = artifact_manifest.get("checkpoint_metadata")
    identity = metadata.get("checkpoint_identity") if isinstance(metadata, dict) else None
    if isinstance(identity, dict):
        if identity.get("archive_contains_mptrj") is True:
            raise SystemExit(
                "checkpoint archive identity contains 'mptrj'; refusing to export as eSEN-OAM\n"
                f"archive_prefixes={identity.get('archive_prefixes')!r}, head_keys={identity.get('head_keys')!r}"
            )
        if identity.get("contains_oam_or_omat") is False:
            raise SystemExit(
                "checkpoint identity hints do not contain 'oam' or 'omat'; refusing to export as eSEN-OAM\n"
                f"archive_prefixes={identity.get('archive_prefixes')!r}, head_keys={identity.get('head_keys')!r}"
            )
    artifact = artifact_manifest.get("artifact")
    if not isinstance(artifact, dict):
        return
    expected_sha256 = artifact.get("sha256")
    expected_size = artifact.get("size_bytes")
    if expected_sha256 is not None and expected_sha256 != checkpoint_sha256:
        raise SystemExit(
            "checkpoint SHA256 does not match artifact manifest\n"
            f"manifest: {expected_sha256}\nactual:   {checkpoint_sha256}"
        )
    if expected_size is not None and expected_size != checkpoint_size:
        raise SystemExit(
            "checkpoint byte size does not match artifact manifest\n"
            f"manifest: {expected_size}\nactual:   {checkpoint_size}"
        )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "tensor"


def stringify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): stringify(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [stringify(child) for child in value]
    if hasattr(value, "tolist"):
        return stringify(value.tolist())
    return repr(value)


def iter_tensors(value: Any, prefix: str = ""):
    import torch  # type: ignore

    if torch.is_tensor(value):
        yield prefix or "tensor", value
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_name = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_tensors(child, child_name)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_name = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from iter_tensors(child, child_name)


def extract_postprocessing(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Read only the supplied checkpoint; never use pretrained fallback values."""
    try:
        normalizer = checkpoint["normalizers"]["energy"]
        mean = float(normalizer["mean"])
        rmsd = float(normalizer["rmsd"])
        raw = checkpoint["elementrefs"]["energy"]["element_references"]
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().tolist()
        refs = [float(v) for v in raw]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint requires energy normalizer and element references") from exc
    if mean != 0.0 or not math.isfinite(rmsd) or rmsd <= 0:
        raise ValueError("only zero energy mean and finite positive RMSD are supported")
    if len(refs) != 119 or not all(math.isfinite(v) for v in refs):
        raise ValueError("element references must contain 119 finite values")
    return {"energy_mean": mean, "energy_rmsd": rmsd, "element_references": refs}


def checkpoint_state_and_config(path: Path) -> tuple[Any, Any, list[str], dict[str, Any]]:
    import torch  # type: ignore

    # This is trusted reference tooling for a pinned checkpoint. Keep
    # weights_only explicit so PyTorch >=2.6 default changes do not alter
    # behavior.
    checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("eSEN checkpoint must be a mapping with postprocessing data")
    top_keys = sorted(str(key) for key in checkpoint.keys())
    state = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    return state, checkpoint.get("config"), top_keys, extract_postprocessing(checkpoint)


def export_tensor(name: str, tensor: Any, tensor_dir: Path, used_files: set[str]) -> dict[str, Any]:
    import torch  # type: ignore

    if tensor.is_floating_point():
        array = tensor.detach().cpu().to(dtype=torch.float64).contiguous().numpy()
        exported_dtype = "f64"
        suffix = ".f64.bin"
    elif tensor.dtype == torch.bool:
        array = tensor.detach().cpu().to(dtype=torch.uint8).contiguous().numpy()
        exported_dtype = "u8"
        suffix = ".u8.bin"
    else:
        array = tensor.detach().cpu().to(dtype=torch.int64).contiguous().numpy()
        exported_dtype = "i64"
        suffix = ".i64.bin"

    base = sanitize_name(name)
    filename = base + suffix
    counter = 1
    while filename in used_files:
        filename = f"{base}_{counter}{suffix}"
        counter += 1
    used_files.add(filename)

    path = tensor_dir / filename
    payload = np.asarray(array).tobytes(order="C")
    path.write_bytes(payload)
    return {
        "name": name,
        "file": str(path.relative_to(tensor_dir.parent)),
        "shape": list(tensor.shape),
        "original_dtype": str(tensor.dtype),
        "exported_dtype": exported_dtype,
        "numel": int(tensor.numel()),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def toml_string(value: str) -> str:
    return json.dumps(value)


def toml_string_array(values: list[str]) -> str:
    return "[" + ", ".join(toml_string(value) for value in values) + "]"


def walk_config_numbers(value: Any, key_names: set[str]) -> list[float]:
    found: list[float] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in key_names and isinstance(child, (int, float)) and float(child) > 0:
                found.append(float(child))
            found.extend(walk_config_numbers(child, key_names))
    elif isinstance(value, list):
        for child in value:
            found.extend(walk_config_numbers(child, key_names))
    return found


def infer_cutoff(config: Any) -> float:
    values = walk_config_numbers(config, {"cutoff", "max_radius", "radius"})
    return values[0] if values else 6.0


def infer_lmax(config: Any) -> int:
    values = walk_config_numbers(config, {"lmax", "l_max", "max_l"})
    return int(values[0]) if values else 4


def manifest_toml(
    *,
    tensors: list[dict[str, Any]],
    config: Any,
    fairchem_commit: str,
    checkpoint_sha256: str,
    checkpoint_size: int,
) -> str:
    cutoff = infer_cutoff(config)
    lmax = infer_lmax(config)
    lines = [
        'format_version = "0.1"',
        'architecture = "esen-oam-v1"',
        'dtype = "f64"',
        'length_unit = "angstrom"',
        'energy_unit = "eV"',
        'force_unit = "eV/angstrom"',
        '',
        "[species]",
        f"symbols = {toml_string_array(CHEMICAL_SYMBOLS_Z1_TO_Z94)}",
        '',
        "[geometry]",
        f"cutoff = {cutoff:.12g}",
        'edge_vector = "R_j + shift - R_i"',
        "directed_edges = true",
        '',
        "[spherical_harmonics]",
        f"l_max = {lmax}",
        "normalize_input = true",
        'normalization = "component"',
        'reference_convention = "pinned-fixture"',
        '',
        "[esen_oam]",
        f"state = {toml_string('esen_oam_state.json')}",
        'reference_package = "fairchem"',
        'reference_version = "1.10.0"',
        f"reference_commit = {toml_string(fairchem_commit)}",
        f"selected_model = {toml_string(MODEL_KEY)}",
        f"state_schema = {toml_string(SCHEMA_VERSION)}",
        f"artifact_sha256 = {toml_string(checkpoint_sha256)}",
        f"artifact_size_bytes = {checkpoint_size}",
        '',
        '[state_dict]',
        f'format = {toml_string(STATE_DICT_FORMAT)}',
        '',
    ]
    for tensor in tensors:
        shape = ", ".join(str(int(value)) for value in tensor["shape"])
        lines.extend(
            [
                '[[tensors]]',
                f'name = {toml_string(tensor["name"])}',
                f'path = {toml_string(tensor["file"])}',
                f'dtype = {toml_string(tensor["exported_dtype"])}',
                f'shape = [{shape}]',
                f'sha256 = {toml_string(tensor["sha256"])}',
                '',
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fairchem-dir", type=Path, default=Path("third_party/fairchem"))
    parser.add_argument("--artifact-manifest", type=Path, default=Path("fixtures/fairchem/current/esen_oam_artifact_manifest.json"))
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--out", type=Path, default=Path("outputs/reference-artifacts/fairchem/esen_oam_state_export.e3rs"))
    args = parser.parse_args()

    fairchem_commit = validate_fairchem_tree(args.fairchem_dir)
    artifact_manifest = load_json(args.artifact_manifest) if args.artifact_manifest.exists() else None
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve() if args.checkpoint_path else None
    if checkpoint_path is None and artifact_manifest is not None:
        artifact_path = artifact_manifest.get("artifact", {}).get("path")
        checkpoint_path = Path(artifact_path).expanduser().resolve() if artifact_path else None
    if checkpoint_path is None:
        raise SystemExit("pass --checkpoint-path or provide an artifact manifest with artifact.path")
    if not checkpoint_path.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    checkpoint_sha256, checkpoint_size = sha256_file(checkpoint_path)
    validate_manifest_artifact_identity(artifact_manifest, checkpoint_sha256, checkpoint_size)
    state, config, top_keys, postprocessing = checkpoint_state_and_config(checkpoint_path)

    out = args.out
    tensor_dir = out / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    tensors: list[dict[str, Any]] = []
    used_files: set[str] = set()
    for name, tensor in iter_tensors(state):
        tensors.append(export_tensor(name, tensor, tensor_dir, used_files))
    if not tensors:
        raise SystemExit(f"no tensors found in checkpoint: {checkpoint_path}")

    report = {
        "schema_version": SCHEMA_VERSION,
        "selected_model": MODEL_KEY,
        "state_dict_format": STATE_DICT_FORMAT,
        "reference": {
            "repository": REPOSITORY,
            "commit": fairchem_commit,
        },
        "artifact": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "size_bytes": checkpoint_size,
            "manifest": str(args.artifact_manifest) if artifact_manifest else None,
        },
        "checkpoint_top_level_keys": top_keys,
        "postprocessing": postprocessing,
        "checkpoint_config": stringify(config),
        "tensor_summary": {
            "count": len(tensors),
            "total_numel": sum(tensor["numel"] for tensor in tensors),
            "total_bytes": sum(tensor["bytes"] for tensor in tensors),
        },
        "tensors": tensors,
        "execution_contract": {
            "status": "state_dict_export_only_runtime_not_implemented",
            "notes": [
                "All floating tensors are exported as f64 little-endian C-order payloads.",
                "This bundle preserves original checkpoint tensor names and config for future fixture-driven Rust implementation.",
            ],
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
    }

    (out / "esen_oam_state.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "manifest.toml").write_text(
        manifest_toml(
            tensors=tensors,
            config=config,
            fairchem_commit=fairchem_commit,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_size=checkpoint_size,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
