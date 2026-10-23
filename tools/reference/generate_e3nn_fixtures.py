#!/usr/bin/env python3
"""
Generate e3nn reference fixtures for e3nn-core-rs.

This script is intentionally outside the Rust runtime. It may depend on Python,
PyTorch, NumPy, and the pinned e3nn submodule.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def checked_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command, stderr=subprocess.STDOUT, text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        output = exc.output.strip()
        raise SystemExit(
            "command failed: "
            + " ".join(command)
            + ("\noutput:\n" + output if output else "")
        ) from exc


def validate_reference_tree(e3nn_dir: Path) -> str:
    if not e3nn_dir.exists():
        raise SystemExit(
            f"missing pinned e3nn reference directory: {e3nn_dir}\n"
            "run scripts/add_e3nn_reference_submodule.sh from an initialized git "
            "repository before generating fixtures"
        )

    commit = checked_output(["git", "-C", str(e3nn_dir), "rev-parse", "HEAD"])
    pin_path = e3nn_dir.with_name(f"{e3nn_dir.name}.PINNED_COMMIT")
    if not pin_path.exists():
        raise SystemExit(
            f"missing pinned commit marker: {pin_path}\n"
            "run scripts/add_e3nn_reference_submodule.sh; do not generate "
            "fixtures from an unpinned checkout"
        )

    pinned_commit = pin_path.read_text(encoding="utf-8").strip()
    if commit != pinned_commit:
        raise SystemExit(
            "e3nn reference checkout does not match pinned commit marker\n"
            f"checkout: {commit}\n"
            f"marker:   {pinned_commit}"
        )
    return commit


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def import_reference_dependencies(e3nn_dir: Path) -> tuple[Any, Any, Any, Any]:
    missing: list[str] = []
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        missing.append("numpy")
        np = None
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        missing.append("torch")
        torch = None
    try:
        import e3nn  # type: ignore[import-not-found]
        from e3nn import o3  # type: ignore[import-not-found]
    except ImportError:
        missing.append("e3nn")
        e3nn = None
        o3 = None

    if missing:
        raise SystemExit(
            "missing Python reference dependencies: "
            + ", ".join(sorted(set(missing)))
            + "\ncreate .venv-reference and run: "
            "python -m pip install -r tools/reference/requirements.txt"
        )

    e3nn_file = Path(e3nn.__file__).resolve()
    expected_root = e3nn_dir.resolve()
    if not is_relative_to(e3nn_file, expected_root):
        raise SystemExit(
            "imported e3nn does not come from the pinned reference checkout\n"
            f"imported: {e3nn_file}\n"
            f"expected under: {expected_root}\n"
            "reinstall the reference environment with: "
            "python -m pip install -r tools/reference/requirements.txt"
        )
    return np, torch, e3nn, o3


def write_manifest(
    out_dir: Path,
    e3nn_commit: str,
    e3nn: Any,
    torch: Any,
    np: Any,
    seed: int,
    fixture_metadata: list[dict[str, Any]],
) -> None:

    manifest = {
        "schema_version": "0.3.0",
        "reference": {
            "name": "e3nn-torch",
            "repository": "https://github.com/e3nn/e3nn",
            "commit": e3nn_commit,
            "package_version": getattr(e3nn, "__version__", "unknown"),
        },
        "python": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "seed": seed,
        "dtype": "float64",
        "conventions": {
            "edge_direction": "r_ij = R_j + shift_ij - R_i",
            "spherical_harmonics_normalization": "component",
            "spherical_harmonics_normalize_input": True,
            "spherical_harmonics_normalize": True,
            "primary_irreps_layout": "e3nn-compatible contiguous blocks",
        },
        "fixtures": fixture_metadata,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def irrep_term_segment_indices(irreps: Any) -> list[list[int]]:
    segment_indices: list[list[int]] = []
    next_segment = 0
    for mul, _irrep in irreps:
        term_segments = []
        for _ in range(mul):
            term_segments.append(next_segment)
            next_segment += 1
        segment_indices.append(term_segments)
    return segment_indices


def tensor_product_plan_metadata(tp: Any, np: Any, torch: Any, o3: Any) -> dict[str, Any]:
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        in1_segments = irrep_term_segment_indices(tp.irreps_in1)
        in2_segments = irrep_term_segment_indices(tp.irreps_in2)
        out_segments = irrep_term_segment_indices(tp.irreps_out)

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

        weight_offset = 0
        for instruction in tp.instructions:
            if instruction.connection_mode not in ("uvw", "uvu"):
                raise SystemExit(
                    "fixture metadata generation currently supports only "
                    f"uvw/uvu tensor-product instructions, got {instruction.connection_mode}"
                )
            if not instruction.has_weight:
                raise SystemExit("expected fully-connected tensor product paths to use weights")

            irreps_in1_term = tp.irreps_in1[instruction.i_in1].ir
            irreps_in2_term = tp.irreps_in2[instruction.i_in2].ir
            irreps_out_term = tp.irreps_out[instruction.i_out].ir
            cg = o3.wigner_3j(
                irreps_in1_term.l, irreps_in2_term.l, irreps_out_term.l
            )
            nonzero = (cg.abs() > 1.0e-14).nonzero()
            cg_offset = len(cg_value)
            for i_m, j_m, k_m in nonzero.tolist():
                cg_i_m.append(i_m)
                cg_j_m.append(j_m)
                cg_k_m.append(k_m)
                cg_value.append(float(cg[i_m, j_m, k_m]))
            cg_len = len(cg_value) - cg_offset

            if instruction.connection_mode == "uvw":
                mul1, mul2, mul_out = instruction.path_shape
                for u in range(mul1):
                    for v in range(mul2):
                        for w in range(mul_out):
                            instruction_in1_segment.append(in1_segments[instruction.i_in1][u])
                            instruction_in2_segment.append(in2_segments[instruction.i_in2][v])
                            instruction_out_segment.append(out_segments[instruction.i_out][w])
                            instruction_cg_offset.append(cg_offset)
                            instruction_cg_len.append(cg_len)
                            instruction_weight_offset.append(
                                weight_offset + (u * mul2 * mul_out) + (v * mul_out) + w
                            )
                            instruction_path_weight.append(float(instruction.path_weight))
                weight_offset += mul1 * mul2 * mul_out
            else:
                mul1, mul2 = instruction.path_shape
                for u in range(mul1):
                    for v in range(mul2):
                        instruction_in1_segment.append(in1_segments[instruction.i_in1][u])
                        instruction_in2_segment.append(in2_segments[instruction.i_in2][v])
                        instruction_out_segment.append(out_segments[instruction.i_out][u])
                        instruction_cg_offset.append(cg_offset)
                        instruction_cg_len.append(cg_len)
                        instruction_weight_offset.append(
                            weight_offset + (u * mul2) + v
                        )
                        instruction_path_weight.append(float(instruction.path_weight))
                weight_offset += mul1 * mul2

        if weight_offset != tp.weight_numel:
            raise SystemExit(
                "expanded tensor-product metadata weight count mismatch: "
                f"expanded={weight_offset}, e3nn={tp.weight_numel}"
            )

        return {
            "instruction_in1_segment": np.array(instruction_in1_segment, dtype=np.int64),
            "instruction_in2_segment": np.array(instruction_in2_segment, dtype=np.int64),
            "instruction_out_segment": np.array(instruction_out_segment, dtype=np.int64),
            "instruction_cg_offset": np.array(instruction_cg_offset, dtype=np.int64),
            "instruction_cg_len": np.array(instruction_cg_len, dtype=np.int64),
            "instruction_weight_offset": np.array(instruction_weight_offset, dtype=np.int64),
            "instruction_path_weight": np.array(instruction_path_weight, dtype=np.float64),
            "cg_i_m": np.array(cg_i_m, dtype=np.int64),
            "cg_j_m": np.array(cg_j_m, dtype=np.int64),
            "cg_k_m": np.array(cg_k_m, dtype=np.int64),
            "cg_value": np.array(cg_value, dtype=np.float64),
        }
    finally:
        torch.set_default_dtype(previous_dtype)


def generate_spherical_harmonics(
    out_dir: Path, seed: int, np: Any, torch: Any, o3: Any, lmax: int = 2
) -> dict[str, Any]:
    torch.manual_seed(seed if lmax == 2 else seed + lmax)
    vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
            [-0.25, 0.5, 1.25],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    y = o3.spherical_harmonics(
        list(range(lmax + 1)), vectors, normalize=True, normalization="component"
    )
    grad_y = torch.randn_like(y)
    scalar = (y * grad_y).sum()
    scalar.backward()
    path = out_dir / f"spherical_harmonics_lmax{lmax}_component_normalized_f64.npz"
    arrays = {
        "operation": "spherical_harmonics",
        "lmax": np.array([lmax], dtype=np.int64),
        "vectors": vectors.detach().cpu().numpy(),
        "y": y.detach().cpu().numpy(),
        "grad_y": grad_y.detach().cpu().numpy(),
        "grad_vectors": vectors.grad.detach().cpu().numpy(),
    }
    np.savez(path, **arrays)
    return {
        "filename": path.name,
        "operation": "spherical_harmonics",
        "lmax": lmax,
        "tensor_shapes": {
            "vectors": list(arrays["vectors"].shape),
            "y": list(arrays["y"].shape),
            "grad_y": list(arrays["grad_y"].shape),
            "grad_vectors": list(arrays["grad_vectors"].shape),
        },
        "conventions": {
            "normalization": "component",
            "normalize_input": True,
            "normalize": True,
        },
    }


def generate_tensor_product_smoke(
    out_dir: Path, seed: int, np: Any, torch: Any, o3: Any
) -> dict[str, Any]:
    torch.manual_seed(seed + 1)
    irreps_in1 = o3.Irreps("2x0e + 1x1o")
    irreps_in2 = o3.Irreps("1x0e + 1x1o")
    irreps_out = o3.Irreps("2x0e + 1x1o")
    tp = o3.FullyConnectedTensorProduct(
        irreps_in1, irreps_in2, irreps_out, shared_weights=False
    ).to(dtype=torch.float64)
    batch = 4
    x = torch.randn(batch, irreps_in1.dim, dtype=torch.float64, requires_grad=True)
    y = torch.randn(batch, irreps_in2.dim, dtype=torch.float64, requires_grad=True)
    w = torch.randn(batch, tp.weight_numel, dtype=torch.float64, requires_grad=True)
    out = tp(x, y, w)
    grad_out = torch.randn_like(out)
    scalar = (out * grad_out).sum()
    scalar.backward()
    path = out_dir / "fully_connected_tensor_product_smoke_f64.npz"
    plan_metadata = tensor_product_plan_metadata(tp, np, torch, o3)
    arrays = {
        "operation": "fully_connected_tensor_product",
        "irreps_in1": str(irreps_in1),
        "irreps_in2": str(irreps_in2),
        "irreps_out": str(irreps_out),
        "x": x.detach().cpu().numpy(),
        "y": y.detach().cpu().numpy(),
        "w": w.detach().cpu().numpy(),
        "out": out.detach().cpu().numpy(),
        "grad_out": grad_out.detach().cpu().numpy(),
        "grad_x": x.grad.detach().cpu().numpy(),
        "grad_y": y.grad.detach().cpu().numpy(),
        "grad_w": w.grad.detach().cpu().numpy(),
        "weight_numel": np.array([tp.weight_numel], dtype=np.int64),
        **plan_metadata,
    }
    np.savez(path, **arrays)
    return {
        "filename": path.name,
        "operation": "fully_connected_tensor_product",
        "irreps_in1": str(irreps_in1),
        "irreps_in2": str(irreps_in2),
        "irreps_out": str(irreps_out),
        "weight_numel": tp.weight_numel,
        "tensor_shapes": {
            "x": list(arrays["x"].shape),
            "y": list(arrays["y"].shape),
            "w": list(arrays["w"].shape),
            "out": list(arrays["out"].shape),
            "grad_out": list(arrays["grad_out"].shape),
            "grad_x": list(arrays["grad_x"].shape),
            "grad_y": list(arrays["grad_y"].shape),
            "grad_w": list(arrays["grad_w"].shape),
            "instructions": list(arrays["instruction_in1_segment"].shape),
            "cg_entries": list(arrays["cg_value"].shape),
        },
    }


def generate_tensor_product_lmax4_smoke(
    out_dir: Path, seed: int, np: Any, torch: Any, o3: Any
) -> dict[str, Any]:
    torch.manual_seed(seed + 3)
    irreps_in1 = o3.Irreps("1x3o + 1x4e")
    irreps_in2 = o3.Irreps("1x1o + 1x2e")
    irreps_out = o3.Irreps("1x2e + 1x3o + 1x4e")
    tp = o3.FullyConnectedTensorProduct(
        irreps_in1, irreps_in2, irreps_out, shared_weights=False
    ).to(dtype=torch.float64)
    batch = 3
    x = (0.01 * torch.randn(batch, irreps_in1.dim, dtype=torch.float64)).requires_grad_()
    y = (0.01 * torch.randn(batch, irreps_in2.dim, dtype=torch.float64)).requires_grad_()
    w = (0.01 * torch.randn(batch, tp.weight_numel, dtype=torch.float64)).requires_grad_()
    out = tp(x, y, w)
    grad_out = torch.randn_like(out)
    scalar = (out * grad_out).sum()
    scalar.backward()
    path = out_dir / "fully_connected_tensor_product_lmax4_smoke_f64.npz"
    plan_metadata = tensor_product_plan_metadata(tp, np, torch, o3)
    arrays = {
        "operation": "fully_connected_tensor_product",
        "irreps_in1": str(irreps_in1),
        "irreps_in2": str(irreps_in2),
        "irreps_out": str(irreps_out),
        "x": x.detach().cpu().numpy(),
        "y": y.detach().cpu().numpy(),
        "w": w.detach().cpu().numpy(),
        "out": out.detach().cpu().numpy(),
        "grad_out": grad_out.detach().cpu().numpy(),
        "grad_x": x.grad.detach().cpu().numpy(),
        "grad_y": y.grad.detach().cpu().numpy(),
        "grad_w": w.grad.detach().cpu().numpy(),
        "weight_numel": np.array([tp.weight_numel], dtype=np.int64),
        **plan_metadata,
    }
    np.savez(path, **arrays)
    return {
        "filename": path.name,
        "operation": "fully_connected_tensor_product",
        "irreps_in1": str(irreps_in1),
        "irreps_in2": str(irreps_in2),
        "irreps_out": str(irreps_out),
        "weight_numel": tp.weight_numel,
        "tensor_shapes": {
            "x": list(arrays["x"].shape),
            "y": list(arrays["y"].shape),
            "w": list(arrays["w"].shape),
            "out": list(arrays["out"].shape),
            "grad_out": list(arrays["grad_out"].shape),
            "grad_x": list(arrays["grad_x"].shape),
            "grad_y": list(arrays["grad_y"].shape),
            "grad_w": list(arrays["grad_w"].shape),
            "instructions": list(arrays["instruction_in1_segment"].shape),
            "cg_entries": list(arrays["cg_value"].shape),
        },
    }


def generate_synthetic_equivariant_v1(
    out_dir: Path, seed: int, np: Any, torch: Any, o3: Any
) -> dict[str, Any]:
    torch.manual_seed(seed + 2)
    cutoff = 3.5
    centers = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
    gamma = 0.8
    layer0_weight = torch.tensor(
        [
            [0.8, -0.4, 0.2, 0.5],
            [-0.3, 0.7, -0.6, 0.1],
            [0.25, 0.35, -0.45, 0.9],
        ],
        dtype=torch.float64,
    )
    layer0_bias = torch.tensor([0.05, -0.1, 0.2], dtype=torch.float64)
    layer1_weight = torch.tensor([[1.1, -0.75, 0.4]], dtype=torch.float64)
    layer1_bias = torch.tensor([0.03], dtype=torch.float64)
    inv_sqrt_3 = 0.5773502691896258
    node_energy_scale = 0.5

    species = np.array(
        [
            ["H", "H", "O"],
            ["O", "H", "H"],
            ["O", "H", "H"],
        ]
    )
    positions_cases = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.1, -0.2, 0.4], [-0.6, 1.3, 0.7]],
            [[0.0, 0.0, 0.0], [2.8, 0.0, 0.0], [0.0, 2.8, 0.0]],
            [[0.0, 0.0, 0.0], [2.8, 0.0, 0.0], [-1.4, 2.4248711305964283, 0.0]],
        ],
        dtype=torch.float64,
    )

    def edge_weight(distance: Any) -> Any:
        basis = torch.exp(-gamma * (distance - centers).pow(2))
        cutoff_value = 0.5 * (torch.cos(torch.pi * distance / cutoff) + 1.0)
        weighted_basis = cutoff_value * basis
        hidden = torch.tanh(layer0_weight @ weighted_basis + layer0_bias)
        return (layer1_weight @ hidden + layer1_bias)[0]

    def energy_for_positions(positions: Any) -> tuple[Any, int]:
        node_features = [
            torch.zeros(3, dtype=torch.float64) for _ in range(positions.shape[0])
        ]
        edge_count = 0
        for src in range(positions.shape[0]):
            for dst in range(positions.shape[0]):
                if src == dst:
                    continue
                r = positions[dst] - positions[src]
                distance = torch.linalg.norm(r)
                if bool((distance < cutoff).detach().cpu().item()):
                    y = o3.spherical_harmonics(
                        [0, 1], r, normalize=True, normalization="component"
                    )
                    node_features[src] = node_features[src] + edge_weight(distance) * y[1:4]
                    edge_count += 1

        energy = torch.zeros((), dtype=torch.float64)
        for feature in node_features:
            energy = energy + node_energy_scale * inv_sqrt_3 * torch.dot(feature, feature)
        return energy, edge_count

    energies = []
    forces = []
    edge_counts = []
    for case in positions_cases:
        positions = case.clone().detach().requires_grad_(True)
        energy, edge_count = energy_for_positions(positions)
        energy.backward()
        energies.append(float(energy.detach().cpu()))
        forces.append((-positions.grad).detach().cpu().numpy())
        edge_counts.append(edge_count)

    path = out_dir / "synthetic_equivariant_v1_energy_force_f64.npz"
    arrays = {
        "operation": np.array("synthetic_equivariant_energy_force"),
        "architecture": np.array("synthetic-equivariant-v1"),
        "species": species,
        "positions": positions_cases.detach().cpu().numpy(),
        "pbc": np.zeros((positions_cases.shape[0], 3), dtype=np.bool_),
        "energy": np.array(energies, dtype=np.float64),
        "forces": np.array(forces, dtype=np.float64),
        "edge_count": np.array(edge_counts, dtype=np.int64),
        "cutoff": np.array(cutoff, dtype=np.float64),
        "centers": centers.detach().cpu().numpy(),
        "gamma": np.array(gamma, dtype=np.float64),
        "mlp_layer0_weight": layer0_weight.detach().cpu().numpy(),
        "mlp_layer0_bias": layer0_bias.detach().cpu().numpy(),
        "mlp_layer1_weight": layer1_weight.detach().cpu().numpy(),
        "mlp_layer1_bias": layer1_bias.detach().cpu().numpy(),
        "mlp_activations": np.array(["tanh", "identity"]),
        "sh_lmax": np.array(1, dtype=np.int64),
        "sh_normalization": np.array("component"),
        "sh_normalize_input": np.array(True, dtype=np.bool_),
        "edge_message_instruction_in1_segment": np.array([0], dtype=np.int64),
        "edge_message_instruction_in2_segment": np.array([0], dtype=np.int64),
        "edge_message_instruction_out_segment": np.array([0], dtype=np.int64),
        "edge_message_instruction_cg_offset": np.array([0], dtype=np.int64),
        "edge_message_instruction_cg_len": np.array([3], dtype=np.int64),
        "edge_message_instruction_weight_offset": np.array([0], dtype=np.int64),
        "edge_message_instruction_path_weight": np.array([1.0], dtype=np.float64),
        "edge_message_cg_i_m": np.array([0, 0, 0], dtype=np.int64),
        "edge_message_cg_j_m": np.array([0, 1, 2], dtype=np.int64),
        "edge_message_cg_k_m": np.array([0, 1, 2], dtype=np.int64),
        "edge_message_cg_value": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        "node_energy_scale": np.array(node_energy_scale, dtype=np.float64),
        "node_energy_instruction_in1_segment": np.array([0], dtype=np.int64),
        "node_energy_instruction_in2_segment": np.array([0], dtype=np.int64),
        "node_energy_instruction_out_segment": np.array([0], dtype=np.int64),
        "node_energy_instruction_cg_offset": np.array([0], dtype=np.int64),
        "node_energy_instruction_cg_len": np.array([3], dtype=np.int64),
        "node_energy_instruction_weight_offset": np.array([-1], dtype=np.int64),
        "node_energy_instruction_path_weight": np.array([1.0], dtype=np.float64),
        "node_energy_cg_i_m": np.array([0, 1, 2], dtype=np.int64),
        "node_energy_cg_j_m": np.array([0, 1, 2], dtype=np.int64),
        "node_energy_cg_k_m": np.array([0, 0, 0], dtype=np.int64),
        "node_energy_cg_value": np.array([inv_sqrt_3] * 3, dtype=np.float64),
    }
    np.savez(path, **arrays)
    return {
        "filename": path.name,
        "operation": "synthetic_equivariant_energy_force",
        "architecture": "synthetic-equivariant-v1",
        "model": {
            "cutoff": cutoff,
            "centers": centers.detach().cpu().numpy().tolist(),
            "gamma": gamma,
            "mlp_activations": ["tanh", "identity"],
            "sh_lmax": 1,
            "sh_normalization": "component",
            "sh_normalize_input": True,
            "edge_message_irreps": "1x0e x 1x1o -> 1x1o",
            "node_energy_irreps": "1x1o x 1x1o -> 1x0e",
        },
        "tensor_shapes": {
            "species": list(arrays["species"].shape),
            "positions": list(arrays["positions"].shape),
            "pbc": list(arrays["pbc"].shape),
            "energy": list(arrays["energy"].shape),
            "forces": list(arrays["forces"].shape),
            "edge_count": list(arrays["edge_count"].shape),
            "centers": list(arrays["centers"].shape),
            "mlp_layer0_weight": list(arrays["mlp_layer0_weight"].shape),
            "mlp_layer0_bias": list(arrays["mlp_layer0_bias"].shape),
            "mlp_layer1_weight": list(arrays["mlp_layer1_weight"].shape),
            "mlp_layer1_bias": list(arrays["mlp_layer1_bias"].shape),
            "mlp_activations": list(arrays["mlp_activations"].shape),
            "edge_message_instructions": list(
                arrays["edge_message_instruction_in1_segment"].shape
            ),
            "edge_message_cg_entries": list(arrays["edge_message_cg_value"].shape),
            "node_energy_instructions": list(
                arrays["node_energy_instruction_in1_segment"].shape
            ),
            "node_energy_cg_entries": list(arrays["node_energy_cg_value"].shape),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("fixtures/e3nn_torch/current"))
    parser.add_argument("--e3nn-dir", type=Path, default=Path("third_party/e3nn-torch"))
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    e3nn_commit = validate_reference_tree(args.e3nn_dir)
    np, torch, e3nn, o3 = import_reference_dependencies(args.e3nn_dir)

    args.out.mkdir(parents=True, exist_ok=True)
    fixture_metadata = [
        generate_spherical_harmonics(args.out, args.seed, np, torch, o3, lmax=2),
        generate_spherical_harmonics(args.out, args.seed, np, torch, o3, lmax=3),
        generate_spherical_harmonics(args.out, args.seed, np, torch, o3, lmax=4),
        generate_tensor_product_smoke(args.out, args.seed, np, torch, o3),
        generate_tensor_product_lmax4_smoke(args.out, args.seed, np, torch, o3),
        generate_synthetic_equivariant_v1(args.out, args.seed, np, torch, o3),
    ]
    write_manifest(args.out, e3nn_commit, e3nn, torch, np, args.seed, fixture_metadata)
    print(f"wrote e3nn reference fixtures to {args.out}")


if __name__ == "__main__":
    main()
