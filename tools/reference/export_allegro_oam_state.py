#!/usr/bin/env python3
"""Export the pinned Allegro-OAM-L package into an `.e3rs` state bundle.

The generated bundle is an ignored local artifact for the narrow
`allegro-oam-l-v1` Rust runtime slice. Runtime parity is claimed only by the
explicit ignored Rust/upstream comparison and finite-difference tests, not by
export success alone.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from inspect_allegro_artifact import (
    SCHEMA_VERSION as INSPECTION_SCHEMA_VERSION,
    import_reference,
    load_json,
    sha256_file,
    stable_path,
    tensor_value_sha256,
    validate_pinned_tree,
    write_json,
)


STATE_EXPORT_SCHEMA_VERSION = "allegro-oam-state-export-v1"
STATE_INVENTORY_SCHEMA_VERSION = "allegro-oam-upstream-state-dict-inventory-v1"
RUNTIME_STATUS = "state_export_only"
ARCHITECTURE = "allegro-oam-l-v1"


def toml_string_array(values: list[str]) -> str:
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"


def tensor_relpath(name: str, dtype: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return Path("weights") / f"{safe_name}.{dtype}.bin"


def tensor_dtype(tensor: Any) -> str:
    dtype = str(tensor.dtype)
    if dtype == "torch.float32":
        return "f32"
    if dtype == "torch.float64":
        return "f64"
    raise SystemExit(f"unsupported Allegro tensor dtype for export: {dtype}")


def tensor_shape(tensor: Any) -> list[int]:
    return [int(value) for value in tensor.shape]


def write_tensor(path: Path, tensor: Any, dtype: str) -> int:
    if dtype == "f32":
        array = np.ascontiguousarray(tensor.detach().cpu().numpy().astype("<f4", copy=False))
    elif dtype == "f64":
        array = np.ascontiguousarray(tensor.detach().cpu().numpy().astype("<f8", copy=False))
    else:
        raise SystemExit(f"unsupported tensor dtype: {dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(array.tobytes(order="C"))
    return int(array.size)


def tensor_entry(
    name: str,
    tensor: Any,
    dtype: str,
    relpath: Path,
    numel: int,
    source: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "dtype": dtype,
        "source": source,
        "source_dtype": str(tensor.dtype),
        "shape": tensor_shape(tensor),
        "numel": numel,
        "path": relpath.as_posix(),
        "value_sha256": tensor_value_sha256(tensor),
    }


def tensor_manifest_entries(tensors: list[dict[str, Any]]) -> str:
    sections = []
    for tensor in tensors:
        shape = ", ".join(str(value) for value in tensor["shape"])
        sections.append(
            "\n".join(
                [
                    "[[tensors]]",
                    f'name = "{tensor["name"]}"',
                    f'dtype = "{tensor["dtype"]}"',
                    f"shape = [{shape}]",
                    'layout = "row-major"',
                    f'path = "{tensor["path"]}"',
                ]
            )
        )
    return "\n\n".join(sections) + "\n"


def prepare_out_dir(out_dir: Path, force: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise SystemExit(f"output directory is not empty; pass --force to replace: {out_dir}")
        shutil.rmtree(out_dir)
    (out_dir / "weights").mkdir(parents=True, exist_ok=True)


def manifest_toml(
    artifact_manifest: dict[str, Any],
    type_names: list[str],
    tensor_entries: list[dict[str, Any]],
) -> str:
    reference = artifact_manifest["reference"]
    artifact = artifact_manifest["artifact"]
    return f"""format_version = "0.1"
architecture = "{ARCHITECTURE}"
dtype = "f64"
length_unit = "angstrom"
energy_unit = "eV"
force_unit = "eV/angstrom"

[species]
symbols = {toml_string_array(type_names)}

[geometry]
cutoff = 7.0
edge_vector = "R_j + shift - R_i"
directed_edges = true

[spherical_harmonics]
l_max = 3
normalize_input = true
normalization = "component"
reference_convention = "pinned-fixture"

[validation]
reference_package = "allegro"
reference_version = "0.8.2"
fixture_seed = 1

[allegro_oam]
model = "allegro_oam_l.json"
reference_package = "{reference["name"]}"
reference_version = "{reference["package_version"]}"
reference_commit = "{reference["commit"]}"
selected_model = "{artifact_manifest["selected_model"]}"
inspection_schema = "{INSPECTION_SCHEMA_VERSION}"
artifact_sha256 = "{artifact["sha256"]}"
artifact_size_bytes = {artifact["size_bytes"]}
runtime_status = "{RUNTIME_STATUS}"

""" + tensor_manifest_entries(tensor_entries)


def state_export_json(
    artifact_manifest: dict[str, Any],
    inspection: dict[str, Any],
    type_names: list[str],
) -> dict[str, Any]:
    execution = inspection["execution_contract"]
    state_dict = inspection["model"]["state_dict"]
    return {
        "schema_version": STATE_EXPORT_SCHEMA_VERSION,
        "selected_model": artifact_manifest["selected_model"],
        "artifact": {
            "sha256": artifact_manifest["artifact"]["sha256"],
            "size_bytes": artifact_manifest["artifact"]["size_bytes"],
        },
        "model_config": {
            "cutoff": 7.0,
            "l_max": 3,
            "num_types": len(type_names),
            "num_layers": int(execution["allegro_module"]["num_layers"]),
        },
        "type_names": type_names,
        "state_dict": {
            "count": int(state_dict["count"]),
            "total_numel": sum(
                int(total["numel"]) for total in state_dict["dtype_totals"].values()
            ),
        },
        "execution_contract": {
            "schema_version": execution["schema_version"],
            "status": execution["status"],
            "contracter_count": len(execution["contracters"]),
            "weighted_channel_count": len(execution["weighted_channels"]),
            "scalar_mlp_count": len(execution["scalar_mlps"]),
            "plan": execution,
        },
    }


def inventory_json(
    artifact_manifest: dict[str, Any],
    tensor_entries: list[dict[str, Any]],
    skipped_empty_tensors: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": STATE_INVENTORY_SCHEMA_VERSION,
        "selected_model": artifact_manifest["selected_model"],
        "state_dict_format": "manifest-f32-f64-tensors-v1",
        "tensor_count": len(tensor_entries),
        "skipped_empty_tensor_count": len(skipped_empty_tensors),
        "total_numel": sum(int(tensor["numel"]) for tensor in tensor_entries),
        "dtype_counts": {
            dtype: sum(1 for tensor in tensor_entries if tensor["dtype"] == dtype)
            for dtype in sorted({tensor["dtype"] for tensor in tensor_entries})
        },
        "source_counts": {
            source: sum(1 for tensor in tensor_entries if tensor.get("source") == source)
            for source in sorted({str(tensor.get("source")) for tensor in tensor_entries})
        },
        "tensors": tensor_entries,
        "skipped_empty_tensors": skipped_empty_tensors,
    }


def export_precontracted_contracter_tensors(
    out_dir: Path,
    torch: Any,
    state: dict[str, Any],
    inspection: dict[str, Any],
) -> list[dict[str, Any]]:
    derived_entries = []
    for contracter in inspection["execution_contract"]["contracters"]:
        weights_name = contracter["weights"]["name"]
        w3j_name = contracter["w3j"]["name"]
        output_name = contracter["precontracted_ww3j"]["name"]
        if weights_name not in state:
            raise SystemExit(f"Contracter weights missing from state dict: {weights_name}")
        if w3j_name not in state:
            raise SystemExit(f"Contracter w3j missing from state dict: {w3j_name}")
        ww3j = torch.einsum(
            str(contracter["weight_w3j_einsum"]),
            state[weights_name],
            state[w3j_name],
        ).detach().contiguous()
        actual_shape = tensor_shape(ww3j)
        expected_shape = [int(value) for value in contracter["precontracted_ww3j"]["shape"]]
        if actual_shape != expected_shape:
            raise SystemExit(
                f"precontracted shape mismatch for {output_name}: "
                f"expected {expected_shape}, got {actual_shape}"
            )
        actual_sha256 = tensor_value_sha256(ww3j)
        expected_sha256 = str(contracter["precontracted_ww3j"]["value_sha256"])
        if actual_sha256 != expected_sha256:
            raise SystemExit(
                f"precontracted SHA256 mismatch for {output_name}\n"
                f"expected: {expected_sha256}\nactual:   {actual_sha256}"
            )
        dtype = tensor_dtype(ww3j)
        relpath = tensor_relpath(output_name, dtype)
        numel = write_tensor(out_dir / relpath, ww3j, dtype)
        derived_entries.append(
            tensor_entry(
                output_name,
                ww3j,
                dtype,
                relpath,
                numel,
                "derived_precontracted_contracter",
            )
        )
    return derived_entries


def export_weighted_channel_rtoi_tensors(
    out_dir: Path,
    model: Any,
    inspection: dict[str, Any],
) -> list[dict[str, Any]]:
    modules = dict(model.named_modules())
    buffer_entries = []
    for weighted_channel in inspection["execution_contract"]["weighted_channels"]:
        module_name = weighted_channel["name"]
        output_name = weighted_channel["rtoi"]["name"]
        if module_name not in modules:
            raise SystemExit(f"MakeWeightedChannels module missing from package: {module_name}")
        module = modules[module_name]
        if not hasattr(module, "_rtoi"):
            raise SystemExit(f"MakeWeightedChannels _rtoi missing on module: {module_name}")
        rtoi = module._rtoi.detach().contiguous()
        actual_shape = tensor_shape(rtoi)
        expected_shape = [int(value) for value in weighted_channel["rtoi"]["shape"]]
        if actual_shape != expected_shape:
            raise SystemExit(
                f"_rtoi shape mismatch for {output_name}: "
                f"expected {expected_shape}, got {actual_shape}"
            )
        actual_sha256 = tensor_value_sha256(rtoi)
        expected_sha256 = str(weighted_channel["rtoi"]["value_sha256"])
        if actual_sha256 != expected_sha256:
            raise SystemExit(
                f"_rtoi SHA256 mismatch for {output_name}\n"
                f"expected: {expected_sha256}\nactual:   {actual_sha256}"
            )
        dtype = tensor_dtype(rtoi)
        relpath = tensor_relpath(output_name, dtype)
        numel = write_tensor(out_dir / relpath, rtoi, dtype)
        buffer_entries.append(
            tensor_entry(
                output_name,
                rtoi,
                dtype,
                relpath,
                numel,
                "module_buffer_weighted_channel",
            )
        )
    return buffer_entries


def export_state(args: argparse.Namespace) -> dict[str, Any]:
    allegro_commit = validate_pinned_tree(args.allegro_dir, "Allegro")
    nequip_commit = validate_pinned_tree(args.nequip_dir, "NeQUIP")
    e3nn_commit = validate_pinned_tree(args.e3nn_dir, "e3nn")
    artifact_manifest = load_json(args.manifest)
    inspection = load_json(args.inspection)
    artifact = artifact_manifest["artifact"]
    package_path = args.package or Path(str(artifact["cache_path"]))
    expected_sha256 = artifact["sha256"]
    expected_size = int(artifact["size_bytes"])
    actual_sha256, actual_size = sha256_file(package_path)
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            "cached Allegro artifact SHA256 does not match manifest\n"
            f"expected: {expected_sha256}\nactual:   {actual_sha256}"
        )
    if actual_size != expected_size:
        raise SystemExit(
            "cached Allegro artifact size does not match manifest\n"
            f"expected: {expected_size}\nactual:   {actual_size}"
        )

    torch, nequip, allegro, ModelFromPackage, ModelTypeNamesFromPackage = import_reference(
        args.allegro_dir, args.nequip_dir, args.e3nn_dir
    )
    type_names = list(ModelTypeNamesFromPackage(str(package_path)))
    packaged = ModelFromPackage(str(package_path), compile_mode=args.compile_mode)
    model = packaged[args.model_key] if args.model_key else packaged
    state = model.state_dict()
    prepare_out_dir(args.out_dir, args.force)

    tensor_entries = []
    skipped_empty_tensors = []
    for name, tensor in sorted(state.items()):
        if int(tensor.numel()) == 0:
            skipped_empty_tensors.append(
                {
                    "name": name,
                    "source_dtype": str(tensor.dtype),
                    "shape": tensor_shape(tensor),
                    "numel": 0,
                    "reason": "manifest tensor specs reject zero-sized shapes",
                    "value_sha256": tensor_value_sha256(tensor),
                }
            )
            continue
        dtype = tensor_dtype(tensor)
        relpath = tensor_relpath(name, dtype)
        numel = write_tensor(args.out_dir / relpath, tensor, dtype)
        entry = tensor_entry(name, tensor, dtype, relpath, numel, "state_dict")
        tensor_entries.append(entry)

    tensor_entries.extend(
        export_precontracted_contracter_tensors(args.out_dir, torch, state, inspection)
    )
    tensor_entries.extend(
        export_weighted_channel_rtoi_tensors(args.out_dir, model, inspection)
    )

    (args.out_dir / "manifest.toml").write_text(
        manifest_toml(artifact_manifest, type_names, tensor_entries), encoding="utf-8"
    )
    write_json(
        args.out_dir / "allegro_oam_l.json",
        state_export_json(artifact_manifest, inspection, type_names),
    )
    write_json(
        args.out_dir / "upstream_state_dict_inventory.json",
        inventory_json(artifact_manifest, tensor_entries, skipped_empty_tensors),
    )

    report = {
        "schema_version": "allegro-oam-state-export-report-v1",
        "status": "exported_state_only_runtime_not_implemented",
        "out_dir": stable_path(args.out_dir),
        "selected_model": artifact_manifest["selected_model"],
        "artifact": {
            "sha256": actual_sha256,
            "size_bytes": actual_size,
            "path": stable_path(package_path),
        },
        "reference": {
            "allegro_commit": allegro_commit,
            "allegro_package_version": getattr(allegro, "__version__", "unknown"),
            "nequip_commit": nequip_commit,
            "nequip_package_version": getattr(nequip, "__version__", "unknown"),
            "e3nn_commit": e3nn_commit,
        },
        "state_dict": {
            "manifest_tensor_count": len(tensor_entries),
            "skipped_empty_tensor_count": len(skipped_empty_tensors),
            "total_numel": sum(int(tensor["numel"]) for tensor in tensor_entries),
            "tensor_dtypes": sorted({tensor["dtype"] for tensor in tensor_entries}),
            "source_counts": {
                source: sum(1 for tensor in tensor_entries if tensor.get("source") == source)
                for source in sorted({str(tensor.get("source")) for tensor in tensor_entries})
            },
        },
        "outputs": [
            "manifest.toml",
            "allegro_oam_l.json",
            "upstream_state_dict_inventory.json",
            "weights/",
        ],
    }
    write_json(args.out_dir / "export_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allegro-dir", type=Path, default=Path("third_party/allegro"))
    parser.add_argument("--nequip-dir", type=Path, default=Path("third_party/nequip"))
    parser.add_argument("--e3nn-dir", type=Path, default=Path("third_party/e3nn-torch"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("fixtures/allegro/v0_8_2_6d05192/allegro_oam_l_artifact_manifest.json"),
    )
    parser.add_argument(
        "--inspection",
        type=Path,
        default=Path("fixtures/allegro/v0_8_2_6d05192/allegro_oam_l_inspection.json"),
    )
    parser.add_argument("--package", type=Path)
    parser.add_argument("--compile-mode", default="eager")
    parser.add_argument("--model-key", default="sole_model")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/reference-artifacts/allegro/allegro_oam_l_state_export.e3rs"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    report = export_state(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
