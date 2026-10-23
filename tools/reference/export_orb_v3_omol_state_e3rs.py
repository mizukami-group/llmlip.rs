#!/usr/bin/env python3
"""Export pinned ORB v3 Conservative OMol state tensors to an ignored `.e3rs`.

This is reference/export tooling only. The generated directory is large and
must stay under ignored `outputs/` paths unless a later task explicitly defines
a committed compact payload policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Z = 1..=83 (H..Bi): the OMol25 element coverage. Kept in sync with
# ORB_ELEMENT_SYMBOLS_BY_Z / ORB_MAX_SUPPORTED_ATOMIC_NUMBER in
# crates/llmlip/src/orb_v3_omol.rs.
OMOL25_ELEMENT_SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi",
]


DEFAULT_WEIGHTS_URL = (
    "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/"
    "forcefields/orb-v3-conservative-omol-20250820.ckpt"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def toml_quote(value: str) -> str:
    return json.dumps(value)


def import_reference(orb_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(orb_dir.resolve()))
    try:
        import torch  # type: ignore[import-not-found]
        import orb_models  # type: ignore[import-not-found]
        from orb_models.forcefield import pretrained  # type: ignore[import-not-found]
    except Exception as exc:
        raise SystemExit(
            "failed to import ORB reference dependencies. Use "
            "`.venv-orb-reference` as documented in tools/reference/README.md.\n"
            f"{type(exc).__name__}: {exc}"
        ) from exc

    imported = Path(orb_models.__file__).resolve()
    expected = orb_dir.resolve()
    if not str(imported).startswith(str(expected)):
        raise SystemExit(
            "imported orb_models does not come from the pinned checkout\n"
            f"imported: {imported}\nexpected under: {expected}"
        )
    return {
        "torch": torch,
        "orb_models": orb_models,
        "pretrained": pretrained,
        "orb_file": str(imported),
    }


def resolve_weights_path(weights_path: str) -> dict[str, Any]:
    if "://" in weights_path:
        try:
            from cached_path import cached_path  # type: ignore[import-not-found]
        except Exception as exc:
            raise SystemExit(
                "URL weights require cached_path in the ORB reference environment.\n"
                f"{type(exc).__name__}: {exc}"
            ) from exc
        resolved = Path(cached_path(weights_path))
    else:
        resolved = Path(weights_path)
    metadata: dict[str, Any] = {"input": weights_path}
    if resolved.exists() and resolved.is_file():
        metadata["size_bytes"] = resolved.stat().st_size
        metadata["sha256"] = sha256_file(resolved)
    metadata["load_path"] = str(resolved) if resolved.exists() else weights_path
    return metadata


def dtype_name(torch: Any, tensor: Any) -> tuple[str, str]:
    if tensor.dtype == torch.float64:
        return "f64", "<f8"
    if tensor.dtype == torch.float32:
        return "f32", "<f4"
    if tensor.dtype == torch.int64:
        return "i64", "<i8"
    raise SystemExit(f"unsupported ORB state tensor dtype {tensor.dtype}")


def sanitize_tensor_filename(index: int, name: str, dtype: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    safe = "_".join(part for part in safe.split("_") if part)
    return f"tensor_{index:04d}_{safe}.{dtype}.bin"


def write_tensor_file(torch: Any, tensor: Any, path: Path) -> tuple[str, int]:
    dtype, numpy_dtype = dtype_name(torch, tensor)
    array = tensor.detach().cpu().contiguous().numpy().astype(numpy_dtype, copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)
    return dtype, path.stat().st_size


def metadata_json(
    ref: dict[str, Any],
    weights: dict[str, Any],
    entries: list[dict[str, Any]],
    total_numel: int,
) -> dict[str, Any]:
    return {
        "schema_version": "orb-v3-omol-metadata-v1",
        "architecture": "orb-v3-conservative-omol-v1",
        "selected_model": "orb-v3-conservative-omol",
        "runtime_status": "state_export_only",
        "reference": {
            "repository": "https://github.com/orbital-materials/orb-models",
            "tag": "v0.6.2",
            "package_version": getattr(ref["orb_models"], "__version__", "unknown"),
            "commit": read_text_if_exists(Path("third_party/orb-models.PINNED_COMMIT")),
        },
        "checkpoint": {
            "filename": "orb-v3-conservative-omol-20250820.ckpt",
            "url": DEFAULT_WEIGHTS_URL,
            "sha256": weights.get("sha256"),
            "size_bytes": weights.get("size_bytes"),
        },
        "graph": {
            "cutoff_angstrom": 6.0,
            "max_num_neighbors": 120,
            "requires_charge": True,
            "requires_spin": True,
            "supports_pbc": False,
        },
        "architecture_parameters": {
            "latent_dim": 256,
            "message_passing_steps": 5,
            "base_mlp_hidden_dim": 1024,
            "base_mlp_depth": 2,
            "head_mlp_hidden_dim": 256,
            "head_mlp_depth": 1,
            "activation": "silu",
            "radial_basis": "bessel",
            "bessel_num_bases": 8,
            "sh_lmax": 3,
            "sh_normalization": "component",
            "sh_normalize": True,
            "outer_product_with_cutoff": True,
            "mlp_norm": "rms_norm",
            "attention_gate": "sigmoid",
            "zbl_pair_repulsion": True,
            "conservative_forces": True,
            "confidence_head": True,
        },
        "state_inventory": {
            "schema_version": "orb-v3-omol-state-export-inventory-v1",
            "tensor_count": len(entries),
            "total_numel": total_numel,
            "entries": entries,
        },
    }


def manifest_toml(ref: dict[str, Any], entries: list[dict[str, Any]]) -> str:
    reference_version = getattr(ref["orb_models"], "__version__", "unknown")
    lines = [
        'format_version = "0.1"',
        'architecture = "orb-v3-conservative-omol-v1"',
        'dtype = "f64"',
        'length_unit = "angstrom"',
        'energy_unit = "eV"',
        'force_unit = "eV/angstrom"',
        "",
        "[species]",
        # OMol25 training coverage: the first 83 elements, H through Bi
        # (arXiv:2505.08762). The checkpoint tables are padded to the full
        # periodic table, but only these species are trained/supported.
        "symbols = ["
        + ", ".join(f'"{s}"' for s in OMOL25_ELEMENT_SYMBOLS)
        + "]",
        "",
        "[geometry]",
        "cutoff = 6.0",
        'edge_vector = "R_j + shift - R_i"',
        "directed_edges = true",
        "",
        "[spherical_harmonics]",
        "l_max = 3",
        "normalize_input = true",
        'normalization = "component"',
        'reference_convention = "pinned-fixture"',
        "",
        "[validation]",
        'reference_package = "orb-models"',
        f"reference_version = {toml_quote(reference_version)}",
        "fixture_seed = 20260524",
        "",
        "[orb_v3_omol]",
        'model = "orb_v3_omol.json"',
        'reference_package = "orb-models"',
        'reference_tag = "v0.6.2"',
        f"reference_version = {toml_quote(reference_version)}",
        'reference_commit = "a5f1a382a72126feff0594b19891e03cde5f843e"',
        'selected_model = "orb-v3-conservative-omol"',
        'checkpoint_filename = "orb-v3-conservative-omol-20250820.ckpt"',
        f"checkpoint_url = {toml_quote(DEFAULT_WEIGHTS_URL)}",
        'checkpoint_sha256 = "c284e99c45df928ae28443fb27223188cc2c33cced593488a4d28595e75cb6e8"',
        "checkpoint_size_bytes = 103417970",
        'runtime_status = "state_export_only"',
        "",
    ]
    for entry in entries:
        lines.extend(
            [
                "[[tensors]]",
                f"name = {toml_quote(entry['name'])}",
                f"dtype = {toml_quote(entry['dtype'])}",
                f"shape = {json.dumps(entry['shape'])}",
                'layout = "row-major"',
                f"path = {toml_quote(entry['path'])}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orb-dir", type=Path, default=Path("third_party/orb-models"))
    parser.add_argument("--weights-path", default=DEFAULT_WEIGHTS_URL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["float64"], default="float64")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/reference-artifacts/orb/orb_v3_conservative_omol_state_export.e3rs"),
    )
    args = parser.parse_args()

    ref = import_reference(args.orb_dir)
    weights = resolve_weights_path(args.weights_path)
    model, _adapter = ref["pretrained"].orb_v3_conservative_omol(
        weights_path=weights["load_path"],
        device=args.device,
        precision=args.precision,
        compile=False,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    total_numel = 0
    for index, (name, tensor) in enumerate(model.state_dict().items()):
        dtype, _numpy_dtype = dtype_name(ref["torch"], tensor)
        rel_path = Path("weights") / sanitize_tensor_filename(index, name, dtype)
        abs_path = args.out / rel_path
        written_dtype, size_bytes = write_tensor_file(ref["torch"], tensor, abs_path)
        assert written_dtype == dtype
        numel = int(tensor.numel())
        total_numel += numel
        entries.append(
            {
                "index": index,
                "name": name,
                "dtype": dtype,
                "shape": list(tensor.shape),
                "numel": numel,
                "path": rel_path.as_posix(),
                "size_bytes": size_bytes,
                "sha256": sha256_file(abs_path),
            }
        )

    (args.out / "orb_v3_omol.json").write_text(
        json.dumps(metadata_json(ref, weights, entries, total_numel), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (args.out / "manifest.toml").write_text(manifest_toml(ref, entries), encoding="utf-8")
    report = {
        "schema_version": "orb-v3-omol-state-export-report-v1",
        "path": str(args.out),
        "tensor_count": len(entries),
        "total_numel": total_numel,
        "total_tensor_bytes": sum(entry["size_bytes"] for entry in entries),
    }
    (args.out / "export_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
