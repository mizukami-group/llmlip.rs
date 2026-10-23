#!/usr/bin/env python3
"""
Generate MACE-lite fixtures from the pinned MACE reference.

The representative MACE-MP0b artifact is inspected before runtime export. The
script writes structured readiness reports and only emits `.e3rs` exports for
the documented scoped runtime path.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_e3nn_fixtures import tensor_product_plan_metadata


SCHEMA_VERSION = "mace-lite-export-report-v1"
LOADER_SMOKE_SCHEMA_VERSION = "mace-lite-loader-smoke-v1"
REFERENCE_SCHEMA_VERSION = "mace-lite-reference-v1"
PBC_REFERENCE_SCHEMA_VERSION = "mace-lite-pbc-reference-v1"
MP0B_FLOW_SCHEMA_VERSION = "mace-mp0b-layer-flow-reference-v1"
MP0B_RUNTIME_EXPORT_SCHEMA_VERSION = "mace-mp0b-runtime-v1"
OSAKA26_RUNTIME_EXPORT_SCHEMA_VERSION = "mace-osaka26-runtime-v1"
OSAKA26_REFERENCE_SCHEMA_VERSION = "mace-osaka26-reference-v1"
MP0B_BROAD_SCHEMA_VERSION = "mace-mp0b-broad-reference-v4"
MP0B_MPTRJ_MINI_SCHEMA_VERSION = "mace-mp0b-mptrj-mini-reference-v1"
OSAKA26_MPTRJ_MINI_SCHEMA_VERSION = "mace-osaka26-mptrj-mini-reference-v1"
MACE_PERIODIC_PATHOLOGICAL_SCHEMA_VERSION = "mace-pbc-pathological-reference-v1"
REFERENCE_REPOSITORY = "https://github.com/ACEsuit/mace"
PINNED_MACE_COMMIT = "4d2da09413ac1407f37cdbb6b81fa28e4c15655e"
PINNED_MACE_PACKAGE_VERSION = "0.3.16"
LOADER_SMOKE_SEED = 97531
REFERENCE_SEED = 20240513


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def toml_string(value: str) -> str:
    return json.dumps(value)


def toml_string_array(values: list[str]) -> str:
    return "[" + ", ".join(toml_string(value) for value in values) + "]"


def toml_int_array(values: list[int]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def read_pinned_commit(mace_dir: Path) -> str:
    pin_path = mace_dir.with_name(f"{mace_dir.name}.PINNED_COMMIT")
    if not pin_path.exists():
        raise SystemExit(f"missing pinned MACE commit file: {pin_path}")
    commit = pin_path.read_text(encoding="utf-8").strip()
    if not commit:
        raise SystemExit(f"empty pinned MACE commit file: {pin_path}")
    return commit


def read_mace_version(mace_dir: Path) -> str:
    version_path = mace_dir / "mace" / "__version__.py"
    for line in version_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit(f"could not read MACE version from {version_path}")


def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def manifest_source_url(manifest: dict[str, Any]) -> str:
    source = manifest["source"]
    return source.get("url") or source["asset_url"]


def as_list(tensor: Any) -> list[Any]:
    return tensor.detach().cpu().tolist()


def tensor_payload(tensor: Any) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "values": as_list(tensor),
    }


def scalar_number(value: Any) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())
    return float(value)


def import_mace_reference(mace_dir: Path) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    sys.path.insert(0, str(mace_dir.resolve()))
    try:
        import ase
        import e3nn
        import mace
        import numpy as np
        import torch
        import torch.nn.functional as torch_functional
        from e3nn import o3
        from mace import data, modules, tools
        from mace.modules.utils import get_edge_vectors_and_lengths
        from mace.tools.scatter import scatter_sum
        from mace.tools import torch_geometric
    except Exception as exc:
        raise SystemExit(
            "failed to import pinned MACE reference dependencies for the "
            f"MACE-lite fixture.\n{type(exc).__name__}: {exc}"
        ) from exc

    mace_file = Path(mace.__file__).resolve()
    if not str(mace_file).startswith(str(mace_dir.resolve())):
        raise SystemExit(
            "imported mace does not come from the pinned checkout\n"
            f"imported: {mace_file}\nexpected under: {mace_dir.resolve()}"
        )
    return ase, e3nn, mace, np, torch, torch_functional, o3, (
        data,
        modules,
        tools,
        torch_geometric,
        get_edge_vectors_and_lengths,
        scatter_sum,
    )


def import_mace_mp0b_reference(args: argparse.Namespace) -> tuple[Any, ...]:
    return import_mace_artifact_reference(args, args.artifact)


def ensure_mace_reference_default_dtype(torch: Any, torch_dtype: Any) -> None:
    # MACE AtomicData.from_config materializes positions/cell/edge tensors using
    # torch's current default dtype. Converting the batch after construction is
    # too late because f32-rounded geometry cannot be recovered.
    if torch.get_default_dtype() != torch_dtype:
        torch.set_default_dtype(torch_dtype)
    if torch.get_default_dtype() != torch_dtype:
        raise RuntimeError(
            "failed to set torch default dtype before MACE AtomicData.from_config: "
            f"got {torch.get_default_dtype()}, expected {torch_dtype}"
        )


def assert_floating_batch_dtype(
    torch: Any,
    batch: dict[str, Any],
    torch_dtype: Any,
    context: str,
) -> None:
    mismatches = [
        f"{key}={value.dtype}"
        for key, value in batch.items()
        if torch.is_tensor(value)
        and value.is_floating_point()
        and value.dtype != torch_dtype
    ]
    if mismatches:
        raise RuntimeError(
            f"{context} produced floating tensors with the wrong dtype "
            f"({', '.join(mismatches)}); expected {torch_dtype}"
        )


def repair_loaded_e3nn_modules(model: Any) -> None:
    """Restore runtime-only e3nn fields omitted by older pickled artifacts."""
    from e3nn.nn._activation import Activation
    from e3nn.o3._spherical_harmonics import SphericalHarmonics, _spherical_harmonics

    for module in model.modules():
        if isinstance(module, SphericalHarmonics) and not hasattr(module, "sph_func"):
            module.sph_func = _spherical_harmonics
        if isinstance(module, Activation) and not hasattr(module, "paths"):
            module.paths = [
                (mul, (ell, parity), act)
                for (mul, (ell, parity)), act in zip(module.irreps_in, module.acts)
            ]


def install_e3nn_codegen_compat() -> None:
    try:
        import e3nn.util.codegen._mixin as codegen_mixin
        import torch
        from torch import fx
    except Exception:
        return

    codegen_mixin_cls = codegen_mixin.CodeGenMixin
    if getattr(codegen_mixin_cls, "_e3rs_codegen_compat", False):
        return

    def compat_setstate(self: Any, d: dict[str, Any]) -> None:
        d = d.copy()
        codegen_state = d.pop("__codegen__", None)
        if hasattr(super(codegen_mixin_cls, self), "__setstate__"):
            super(codegen_mixin_cls, self).__setstate__(d)
        else:
            self.__dict__.update(d)

        if codegen_state is not None:
            for fname, encoded in codegen_state.items():
                assert isinstance(fname, str)
                if isinstance(encoded, bytes):
                    try:
                        smod = pickle.loads(encoded)
                    except Exception:
                        smod = torch.jit.load(io.BytesIO(encoded))
                else:
                    buffer_type, buffer = encoded[0], encoded[1]
                    assert isinstance(buffer_type, str)
                    assert isinstance(buffer, bytes)
                    if buffer_type == "fx":
                        smod = pickle.loads(buffer)
                    elif buffer_type == "torchscript":
                        smod = torch.jit.load(io.BytesIO(buffer))
                    else:
                        raise NotImplementedError(buffer_type)
                assert isinstance(
                    smod,
                    (fx.GraphModule, torch._dynamo.OptimizedModule, torch.jit.ScriptModule),
                )
                setattr(self, fname, smod)
            self.__codegen__ = list(codegen_state.keys())

    codegen_mixin_cls.__setstate__ = compat_setstate
    codegen_mixin_cls._e3rs_codegen_compat = True


def import_mace_artifact_reference(
    args: argparse.Namespace, artifact: Path
) -> tuple[Any, ...]:
    if args.e3nn_compat_path is not None:
        sys.path.insert(0, str(args.e3nn_compat_path.resolve()))
    # These trusted pinned reference artifacts are used only under tools/reference.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    ase, e3nn, mace, np, torch, torch_functional, o3, refs = import_mace_reference(args.mace_dir)
    install_e3nn_codegen_compat()
    model = torch.load(artifact, map_location="cpu", weights_only=False).eval()
    repair_loaded_e3nn_modules(model)
    return ase, e3nn, mace, np, torch, torch_functional, o3, refs, model


def build_one_layer_reference_model(
    np: Any,
    torch: Any,
    torch_functional: Any,
    o3: Any,
    modules: Any,
    tools: Any,
) -> tuple[Any, Any]:
    return build_reference_model(np, torch, torch_functional, o3, modules, tools, 1)


def build_two_layer_reference_model(
    np: Any,
    torch: Any,
    torch_functional: Any,
    o3: Any,
    modules: Any,
    tools: Any,
) -> tuple[Any, Any]:
    return build_reference_model(np, torch, torch_functional, o3, modules, tools, 2)


def build_reference_model(
    np: Any,
    torch: Any,
    torch_functional: Any,
    o3: Any,
    modules: Any,
    tools: Any,
    num_interactions: int,
    interaction_cls_name: str = "RealAgnosticResidualInteractionBlock",
    interaction_cls_first_name: str | None = None,
    distance_transform: str = "None",
    pair_repulsion: bool = False,
    max_ell: int = 2,
) -> tuple[Any, Any]:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(REFERENCE_SEED)
    np.random.seed(REFERENCE_SEED)
    if interaction_cls_first_name is None:
        interaction_cls_first_name = interaction_cls_name
    table = tools.AtomicNumberTable([1, 6])
    model = modules.ScaleShiftMACE(
        r_max=4.0,
        num_bessel=1,
        num_polynomial_cutoff=6,
        max_ell=max_ell,
        interaction_cls=modules.interaction_classes[interaction_cls_name],
        interaction_cls_first=modules.interaction_classes[interaction_cls_first_name],
        num_interactions=num_interactions,
        num_elements=len(table),
        hidden_irreps=o3.Irreps("1x0e"),
        MLP_irreps=o3.Irreps("1x0e"),
        gate=torch_functional.silu,
        atomic_energies=np.array([0.0, 0.0], dtype=np.float64),
        avg_num_neighbors=1.0,
        atomic_numbers=table.zs,
        correlation=3,
        pair_repulsion=pair_repulsion,
        apply_cutoff=True,
        use_reduced_cg=False,
        use_so3=False,
        use_agnostic_product=False,
        use_last_readout_only=False,
        use_embedding_readout=False,
        distance_transform=distance_transform,
        radial_MLP=[1],
        radial_type="bessel",
        heads=["Default"],
        cueq_config=None,
        embedding_specs=None,
        oeq_config=None,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    ).eval()
    return model, table


def deterministic_reference_structures(ase: Any) -> list[tuple[str, Any]]:
    return [
        (
            "ch2_nonperiodic",
            ase.Atoms(
                numbers=[6, 1, 1],
                positions=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0]],
                cell=[10.0, 10.0, 10.0],
                pbc=[False, False, False],
            ),
        ),
        (
            "c2h2_nonperiodic",
            ase.Atoms(
                numbers=[6, 1, 6, 1],
                positions=[
                    [0.0, 0.0, 0.0],
                    [0.9, 0.2, 0.1],
                    [1.8, 0.7, -0.2],
                    [2.6, 0.8, 0.3],
                ],
                cell=[12.0, 12.0, 12.0],
                pbc=[False, False, False],
            ),
        ),
    ]


def deterministic_pbc_reference_structures(ase: Any) -> list[tuple[str, Any]]:
    return [
        (
            "ch2_orthorhombic_pbc",
            ase.Atoms(
                numbers=[6, 1, 1],
                positions=[
                    [0.2, 0.3, 0.4],
                    [8.05, 0.35, 0.45],
                    [0.25, 8.2, 0.5],
                ],
                cell=[8.5, 8.7, 8.9],
                pbc=[True, True, True],
            ),
        ),
    ]


def run_one_layer_structure(
    data: Any,
    torch_geometric: Any,
    get_edge_vectors_and_lengths: Any,
    scatter_sum: Any,
    torch: Any,
    model: Any,
    table: Any,
    name: str,
    atoms: Any,
) -> dict[str, Any]:
    ensure_mace_reference_default_dtype(torch, torch.float64)
    config = data.config_from_atoms(atoms, head_name="Default")
    atomic_data = data.AtomicData.from_config(
        config,
        z_table=table,
        cutoff=float(model.r_max),
        heads=model.heads,
    )
    batch = next(
        iter(torch_geometric.dataloader.DataLoader([atomic_data], batch_size=1))
    ).to_dict()
    assert_floating_batch_dtype(torch, batch, torch.float64, "MACE AtomicData.from_config")
    return run_reference_structure(
        get_edge_vectors_and_lengths,
        scatter_sum,
        torch,
        model,
        name,
        atoms,
        batch,
        include_one_layer_intermediates=True,
    )


def run_two_layer_structure(
    data: Any,
    torch_geometric: Any,
    get_edge_vectors_and_lengths: Any,
    scatter_sum: Any,
    torch: Any,
    model: Any,
    table: Any,
    name: str,
    atoms: Any,
    include_stress_virials: bool = False,
) -> dict[str, Any]:
    ensure_mace_reference_default_dtype(torch, torch.float64)
    config = data.config_from_atoms(atoms, head_name="Default")
    atomic_data = data.AtomicData.from_config(
        config,
        z_table=table,
        cutoff=float(model.r_max),
        heads=model.heads,
    )
    batch = next(
        iter(torch_geometric.dataloader.DataLoader([atomic_data], batch_size=1))
    ).to_dict()
    assert_floating_batch_dtype(torch, batch, torch.float64, "MACE AtomicData.from_config")
    return run_reference_structure(
        get_edge_vectors_and_lengths,
        scatter_sum,
        torch,
        model,
        name,
        atoms,
        batch,
        include_one_layer_intermediates=False,
        include_stress_virials=include_stress_virials,
    )


def run_reference_structure(
    get_edge_vectors_and_lengths: Any,
    scatter_sum: Any,
    torch: Any,
    model: Any,
    name: str,
    atoms: Any,
    batch: dict[str, Any],
    include_one_layer_intermediates: bool,
    include_stress_virials: bool = False,
) -> dict[str, Any]:
    output = model(
        batch,
        training=False,
        compute_force=True,
        compute_virials=include_stress_virials,
        compute_stress=include_stress_virials,
    )
    outputs = {
        "energy": as_list(output["energy"]),
        "node_energy": as_list(output["node_energy"]),
        "interaction_energy": as_list(output["interaction_energy"]),
        "forces": as_list(output["forces"]),
        "node_feats": as_list(output["node_feats"]),
    }
    if include_stress_virials:
        outputs["virials"] = as_list(output["virials"])
        if "stress" in output:
            outputs["stress"] = as_list(output["stress"])
    batch_positions = as_list(batch["positions"]) if "positions" in batch else atoms.positions.tolist()
    if "cell" in batch:
        batch_cell = as_list(batch["cell"])
        if len(batch_cell) == 1:
            batch_cell = batch_cell[0]
    else:
        batch_cell = atoms.cell.array.tolist()
    structure = {
        "name": name,
        "atomic_numbers": list(map(int, atoms.numbers.tolist())),
        "positions": batch_positions,
        "cell": batch_cell,
        "pbc": list(map(bool, atoms.pbc.tolist())),
        "batch": {
            "edge_index": as_list(batch["edge_index"]),
            "shifts": as_list(batch["shifts"]),
            "unit_shifts": as_list(batch["unit_shifts"]),
            "node_attrs": as_list(batch["node_attrs"]),
            "head": as_list(batch["head"]),
        },
        "outputs": outputs,
    }
    if include_one_layer_intermediates:
        structure["intermediates"] = one_layer_forward_intermediates(
            get_edge_vectors_and_lengths, scatter_sum, torch, model, batch
        )
    return structure


def deterministic_mp0b_flow_structure(ase: Any) -> tuple[str, Any]:
    return (
        "hco_nonperiodic_mp0b_flow",
        ase.Atoms(
            numbers=[6, 1, 8],
            positions=[[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.2, 1.1, 0.1]],
            cell=[12.0, 12.0, 12.0],
            pbc=[False, False, False],
        ),
    )


def deterministic_mp0b_flow_structures(ase: Any) -> list[tuple[str, Any, bool, bool]]:
    name, atoms = deterministic_mp0b_flow_structure(ase)
    return [
        (name, atoms, True, False),
        (
            "hcon_nonperiodic_mp0b_validation",
            ase.Atoms(
                numbers=[6, 1, 8, 7],
                positions=[
                    [0.2, 0.1, 0.3],
                    [1.4, 0.2, 0.5],
                    [0.6, 1.5, 0.7],
                    [1.8, 1.2, 1.4],
                ],
                cell=[16.0, 16.0, 16.0],
                pbc=[False, False, False],
            ),
            False,
            False,
        ),
        (
            "hco_pbc_mp0b_stress",
            ase.Atoms(
                numbers=[6, 1, 8],
                positions=[
                    [0.25, 0.5, 0.75],
                    [12.0, 0.75, 1.0],
                    [0.5, 12.25, 1.25],
                ],
                cell=[12.5, 12.75, 13.0],
                pbc=[True, True, True],
            ),
            False,
            True,
        ),
        (
            "hcon_pbc_mp0b_stress",
            ase.Atoms(
                numbers=[6, 1, 8, 7],
                positions=[
                    [0.3, 0.4, 0.5],
                    [12.6, 0.7, 0.9],
                    [0.6, 12.8, 1.2],
                    [0.8, 0.9, 12.9],
                ],
                cell=[13.0, 13.25, 13.5],
                pbc=[True, True, True],
            ),
            False,
            True,
        ),
    ]


def deterministic_mp0b_broad_structures(ase: Any) -> list[tuple[str, Any, bool]]:
    return [
        (
            "mp0b_broader_species_nonperiodic",
            ase.Atoms(
                numbers=[1, 6, 8, 14, 26, 29],
                positions=[
                    [0.0, 0.0, 0.0],
                    [1.2, 0.2, 0.1],
                    [0.3, 1.4, 0.2],
                    [1.7, 1.5, 0.4],
                    [2.7, 0.5, 1.3],
                    [3.3, 1.8, 1.8],
                ],
                cell=[18.0, 18.0, 18.0],
                pbc=[False, False, False],
            ),
            False,
        ),
        (
            "mp0b_near_cutoff_mixed_nonperiodic",
            ase.Atoms(
                numbers=[3, 6, 8, 11, 13, 16, 17, 26],
                positions=[
                    [0.0, 0.0, 0.0],
                    [1.4, 0.1, 0.2],
                    [0.2, 1.5, 0.3],
                    [2.1, 1.7, 0.8],
                    [3.4, 0.6, 1.9],
                    [4.6, 2.0, 2.2],
                    [5.85, 0.0, 0.0],
                    [6.7, 1.6, 2.8],
                ],
                cell=[24.0, 24.0, 24.0],
                pbc=[False, False, False],
            ),
            False,
        ),
        (
            "mp0b_dense_midperiod_nonperiodic",
            ase.Atoms(
                numbers=[1, 5, 7, 8, 14, 22, 29, 47],
                positions=[
                    [0.1, 0.2, 0.3],
                    [1.0, 0.4, 0.2],
                    [0.4, 1.1, 0.5],
                    [1.3, 1.2, 0.7],
                    [2.0, 0.3, 1.4],
                    [2.5, 1.4, 1.1],
                    [3.2, 0.8, 2.0],
                    [3.8, 1.9, 2.4],
                ],
                cell=[22.0, 22.0, 22.0],
                pbc=[False, False, False],
            ),
            False,
        ),
        (
            "mp0b_high_z_species_nonperiodic",
            ase.Atoms(
                numbers=[35, 53, 56, 74, 79, 82, 92, 94],
                positions=[
                    [0.2, 0.1, 0.3],
                    [1.8, 0.4, 0.6],
                    [0.6, 2.0, 0.7],
                    [2.5, 1.7, 1.1],
                    [3.7, 0.5, 1.6],
                    [4.6, 2.2, 2.0],
                    [5.4, 0.8, 2.7],
                    [6.2, 2.5, 3.2],
                ],
                cell=[28.0, 28.0, 28.0],
                pbc=[False, False, False],
            ),
            False,
        ),
        (
            "mp0b_full_pbc_mixed_orthorhombic",
            ase.Atoms(
                numbers=[6, 8, 14, 26, 29],
                positions=[
                    [0.2, 0.3, 0.4],
                    [7.9, 0.6, 0.7],
                    [0.7, 8.2, 0.9],
                    [2.3, 1.8, 2.2],
                    [6.6, 7.5, 8.1],
                ],
                cell=[8.4, 8.8, 9.2],
                pbc=[True, True, True],
            ),
            True,
        ),
        (
            "mp0b_full_pbc_mixed_triclinic",
            ase.Atoms(
                numbers=[3, 8, 13, 16],
                positions=[
                    [0.4, 0.5, 0.6],
                    [7.7, 0.8, 0.9],
                    [1.0, 7.9, 1.2],
                    [2.2, 2.5, 7.8],
                ],
                cell=[
                    [8.3, 0.2, 0.1],
                    [0.4, 8.7, 0.3],
                    [0.2, 0.5, 9.1],
                ],
                pbc=[True, True, True],
            ),
            True,
        ),
        (
            "mp0b_large_mixed_full_pbc_near_cutoff",
            ase.Atoms(
                numbers=[1, 6, 8, 14, 3, 7, 16, 22],
                # The x positions deliberately create wrapped image edges in a
                # 12.8 A cell, including an H-Li pair at about 5.93 A to pin
                # behavior near MACE-MP0b's strict 6.0 A cutoff.
                positions=[
                    [0.25, 0.35, 0.40],
                    [1.55, 0.70, 0.65],
                    [0.85, 1.95, 0.95],
                    [2.60, 1.35, 1.55],
                    [6.18, 0.45, 0.48],
                    [6.35, 2.20, 1.10],
                    [11.55, 0.85, 0.60],
                    [3.45, 4.85, 1.25],
                ],
                cell=[12.8, 12.9, 13.0],
                pbc=[True, True, True],
            ),
            True,
        ),
        (
            "mp0b_multi_image_1d_pbc",
            ase.Atoms(
                numbers=[14, 8],
                positions=[
                    [0.2, 0.1, 0.1],
                    [2.7, 0.1, 0.1],
                ],
                cell=[5.0, 8.0, 8.0],
                pbc=[True, False, False],
            ),
            True,
        ),
        (
            "mp0b_multi_image_2d_pbc",
            ase.Atoms(
                numbers=[1, 8, 14],
                positions=[
                    [0.15, 0.2, 0.4],
                    [2.65, 0.25, 0.5],
                    [0.25, 2.75, 0.6],
                ],
                cell=[5.2, 5.4, 9.0],
                pbc=[True, True, False],
            ),
            True,
        ),
    ]


def deterministic_osaka26_reference_structures(ase: Any) -> list[tuple[str, Any, bool]]:
    return [
        (
            "osaka26_hco_nonperiodic",
            ase.Atoms(
                numbers=[1, 6, 8],
                positions=[
                    [0.0, 0.0, 0.0],
                    [1.15, 0.2, 0.1],
                    [0.3, 1.35, 0.2],
                ],
                cell=[18.0, 18.0, 18.0],
                pbc=[False, False, False],
            ),
            False,
        ),
        (
            "osaka26_cf_h_nonperiodic",
            ase.Atoms(
                numbers=[98, 1, 8],
                positions=[
                    [0.2, 0.1, 0.3],
                    [1.8, 0.2, 0.4],
                    [0.4, 1.9, 0.5],
                ],
                cell=[20.0, 20.0, 20.0],
                pbc=[False, False, False],
            ),
            False,
        ),
        (
            "osaka26_hco_full_pbc",
            ase.Atoms(
                numbers=[1, 6, 8],
                positions=[
                    [0.3, 0.4, 0.5],
                    [11.7, 0.6, 0.8],
                    [0.5, 11.9, 1.0],
                ],
                cell=[12.0, 12.25, 12.5],
                pbc=[True, True, True],
            ),
            True,
        ),
    ]


DETERMINISTIC_FRACTIONAL_TEMPLATE = [
    [0.12, 0.23, 0.18],
    [0.82, 0.44, 0.67],
    [0.41, 0.77, 0.33],
    [0.63, 0.12, 0.31],
    [0.28, 0.53, 0.42],
    [0.57, 0.31, 0.76],
    [0.16, 0.68, 0.71],
    [0.93, 0.21, 0.58],
]


def _orthorhombic_cell(a: float, b: float, c: float) -> list[list[float]]:
    return [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]]


def _triclinic_cell(
    a: float,
    b: float,
    c: float,
    shear_xy: float,
    shear_xz: float,
    shear_yz: float,
) -> list[list[float]]:
    return [[a, 0.0, 0.0], [shear_xy, b, 0.0], [shear_xz, shear_yz, c]]


def _fractional_to_cartesian(
    cell: list[list[float]], fraction: tuple[float, float, float]
) -> list[float]:
    fx, fy, fz = fraction
    return [
        cell[0][0] * fx + cell[1][0] * fy + cell[2][0] * fz,
        cell[0][1] * fx + cell[1][1] * fy + cell[2][1] * fz,
        cell[0][2] * fx + cell[1][2] * fy + cell[2][2] * fz,
    ]


def _fractional_positions(
    cell: list[list[float]],
    atom_count: int,
    seed: int,
    phase: tuple[float, float, float],
) -> list[list[float]]:
    positions = []
    for atom in range(atom_count):
        base = DETERMINISTIC_FRACTIONAL_TEMPLATE[(seed * 3 + atom * 2) % len(DETERMINISTIC_FRACTIONAL_TEMPLATE)]
        fractions = (
            (base[0] + phase[0] + 0.017 * atom) % 1.0,
            (base[1] + phase[1] + 0.019 * seed) % 1.0,
            (base[2] + phase[2] + 0.013 * (atom + 1)) % 1.0,
        )
        positions.append(_fractional_to_cartesian(cell, fractions))
    return positions


def _build_periodic_structure(
    ase: Any,
    name: str,
    numbers: list[int],
    cell: list[list[float]],
    pbc: list[bool],
    seed: int,
) -> tuple[str, Any, bool, bool]:
    return (
        name,
        ase.Atoms(
            numbers=numbers,
            positions=_fractional_positions(
                cell,
                atom_count=len(numbers),
                seed=seed,
                phase=(
                    (0.07 * seed + 0.01 * len(numbers)) % 1.0,
                    (0.13 * seed + 0.03 * len(numbers)) % 1.0,
                    (0.19 * seed + 0.05 * len(numbers)) % 1.0,
                ),
            ),
            cell=cell,
            pbc=pbc,
        ),
        False,
        True,
    )


def deterministic_mp0b_mptrj_mini_structures(ase: Any) -> list[tuple[str, Any, bool, bool]]:
    species_templates = [
        [1, 6, 8, 1],
        [1, 3, 8, 6],
        [6, 8, 14, 1],
        [1, 6, 8, 7],
        [1, 26, 8, 1],
    ]
    structures = []
    full_orth_counter = 0
    full_triclinic_counter = 0
    partial_xy_counter = 0
    partial_x_counter = 0
    for index in range(20):
        species = species_templates[index % len(species_templates)]
        if index == 0:
            cell = _orthorhombic_cell(5.00, 5.20, 5.10)
            structures.append(
                _build_periodic_structure(
                    ase,
                    "mp0b_mptrj_mini_full_pbc",
                    species,
                    cell,
                    [True, True, True],
                    index,
                )
            )
            continue

        if index == 1:
            cell = _orthorhombic_cell(4.88, 5.12, 5.34)
            structures.append(
                _build_periodic_structure(
                    ase,
                    "mp0b_mptrj_mini_partial_pbc",
                    species,
                    cell,
                    [True, True, False],
                    index,
                )
            )
            continue

        mode = index % 4
        if mode == 0:
            cell = _orthorhombic_cell(
                4.90 + 0.07 * (index % 4),
                5.05 + 0.09 * ((index + 1) % 4),
                5.28 + 0.06 * ((index + 3) % 4),
            )
            name = f"mp0b_mptrj_mini_full_orth_{full_orth_counter:02d}"
            full_orth_counter += 1
            pbc = [True, True, True]
        elif mode == 1:
            cell = _triclinic_cell(
                4.95 + 0.05 * (index % 4),
                5.12 + 0.07 * ((index + 2) % 4),
                5.18 + 0.05 * ((index + 1) % 4),
                0.14 + 0.01 * ((index) % 3),
                0.08 + 0.01 * ((index + 2) % 3),
                0.06 + 0.01 * ((index + 1) % 3),
            )
            name = f"mp0b_mptrj_mini_full_triclinic_{full_triclinic_counter:02d}"
            full_triclinic_counter += 1
            pbc = [True, True, True]
        elif mode == 2:
            cell = _orthorhombic_cell(
                5.34 + 0.09 * ((index + 1) % 3),
                5.50 + 0.11 * ((index + 2) % 3),
                7.10 + 0.12 * ((index + 3) % 3),
            )
            name = f"mp0b_mptrj_mini_partial_xy_pbc_{partial_xy_counter:02d}"
            partial_xy_counter += 1
            pbc = [True, True, False]
        else:
            cell = _orthorhombic_cell(
                6.72 + 0.10 * ((index + 3) % 3),
                8.16 + 0.06 * ((index + 2) % 3),
                4.20 + 0.08 * ((index + 1) % 3),
            )
            name = f"mp0b_mptrj_mini_partial_x_pbc_{partial_x_counter:02d}"
            partial_x_counter += 1
            pbc = [True, False, False]
        structures.append(_build_periodic_structure(ase, name, species, cell, pbc, index))
    return structures


def deterministic_osaka26_mptrj_mini_structures(ase: Any) -> list[tuple[str, Any, bool, bool]]:
    species_templates = [
        [98, 1, 8],
        [1, 6, 8],
        [1, 26, 8],
        [98, 6, 1, 1],
        [1, 3, 8, 98],
        [1, 6, 7, 8],
    ]
    structures = []
    full_orth_counter = 0
    full_triclinic_counter = 0
    partial_yz_counter = 0
    partial_xz_counter = 0
    for index in range(20):
        species = species_templates[index % len(species_templates)]
        if index == 0:
            cell = _orthorhombic_cell(5.60, 5.80, 6.00)
            structures.append(
                _build_periodic_structure(
                    ase,
                    "osaka26_mptrj_mini_full_pbc",
                    species,
                    cell,
                    [True, True, True],
                    index,
                )
            )
            continue

        if index == 1:
            cell = _orthorhombic_cell(5.20, 5.40, 5.10)
            structures.append(
                _build_periodic_structure(
                    ase,
                    "osaka26_mptrj_mini_hco_full_pbc",
                    species,
                    cell,
                    [True, True, True],
                    index,
                )
            )
            continue

        mode = index % 4
        if mode == 0:
            cell = _orthorhombic_cell(
                4.82 + 0.08 * (index % 4),
                5.02 + 0.06 * ((index + 1) % 4),
                5.28 + 0.07 * ((index + 2) % 4),
            )
            name = f"osaka26_mptrj_mini_full_orth_{full_orth_counter:02d}"
            full_orth_counter += 1
            pbc = [True, True, True]
        elif mode == 1:
            cell = _triclinic_cell(
                4.95 + 0.04 * ((index + 1) % 4),
                5.11 + 0.05 * ((index + 3) % 4),
                5.32 + 0.06 * ((index + 2) % 4),
                0.10 + 0.01 * ((index) % 3),
                0.07 + 0.01 * ((index + 1) % 3),
                -0.04 - 0.01 * ((index + 2) % 3),
            )
            name = f"osaka26_mptrj_mini_full_triclinic_{full_triclinic_counter:02d}"
            full_triclinic_counter += 1
            pbc = [True, True, True]
        elif mode == 2:
            cell = _orthorhombic_cell(
                6.40 + 0.06 * ((index + 2) % 3),
                5.40 + 0.10 * ((index + 3) % 3),
                6.20 + 0.09 * ((index + 1) % 3),
            )
            name = f"osaka26_mptrj_mini_partial_yz_pbc_{partial_yz_counter:02d}"
            partial_yz_counter += 1
            pbc = [False, True, True]
        else:
            cell = _orthorhombic_cell(
                5.30 + 0.10 * ((index + 3) % 3),
                8.10 + 0.05 * ((index + 1) % 3),
                5.90 + 0.06 * ((index + 2) % 3),
            )
            name = f"osaka26_mptrj_mini_partial_xz_pbc_{partial_xz_counter:02d}"
            partial_xz_counter += 1
            pbc = [True, False, True]
        structures.append(_build_periodic_structure(ase, name, species, cell, pbc, index))
    return structures


def deterministic_periodic_pathological_structures(
    ase: Any,
) -> list[tuple[str, Any, bool, bool]]:
    species_templates = [
        [1, 8, 6, 1],
        [1, 1, 8, 14],
        [1, 6, 6, 7],
        [1, 3, 8, 16],
        [1, 6, 8, 1, 14],
    ]
    structures = []
    full_orth_counter = 0
    full_triclinic_counter = 0
    partial_xy_counter = 0
    partial_xz_counter = 0
    for index in range(20):
        species = species_templates[index % len(species_templates)]
        if index == 0:
            cell = _orthorhombic_cell(5.00, 5.40, 5.50)
            structures.append(
                _build_periodic_structure(
                    ase,
                    "mp0b_pathological_full_pbc",
                    species,
                    cell,
                    [True, True, True],
                    index,
                )
            )
            continue

        if index == 1:
            cell = _orthorhombic_cell(6.00, 6.00, 7.80)
            structures.append(
                _build_periodic_structure(
                    ase,
                    "mp0b_pathological_partial_pbc_2d",
                    species,
                    cell,
                    [True, True, False],
                    index,
                )
            )
            continue
        if index == 2:
            cell = _orthorhombic_cell(20.0, 20.0, 20.0)
            structures.append(
                (
                    "mp0b_pathological_large_box_isolated_atom_00",
                    ase.Atoms(
                        numbers=[1],
                        positions=[[10.0, 10.0, 10.0]],
                        cell=cell,
                        pbc=[True, True, True],
                    ),
                    False,
                    True,
                )
            )
            continue
        if index == 3:
            structures.append(
                (
                    "mp0b_pathological_fractional_boundary_00",
                    ase.Atoms(
                        numbers=[1, 6, 8, 1],
                        positions=[
                            [0.05, 0.95, 0.03],
                            [4.62, 0.08, 0.07],
                            [0.12, 4.58, 1.35],
                            [4.55, 2.40, 4.52],
                        ],
                        cell=[4.8, 4.75, 4.7],
                        pbc=[True, True, True],
                    ),
                    False,
                    True,
                )
            )
            continue

        mode = index % 4
        if mode == 0:
            cell = _orthorhombic_cell(
                4.70 + 0.06 * (index % 4),
                5.20 + 0.04 * ((index + 1) % 4),
                4.90 + 0.07 * ((index + 2) % 4),
            )
            name = f"mp0b_pathological_full_stretched_orth_{full_orth_counter:02d}"
            full_orth_counter += 1
            pbc = [True, True, True]
        elif mode == 1:
            cell = _triclinic_cell(
                4.80 + 0.08 * ((index + 2) % 4),
                5.30 + 0.06 * ((index + 1) % 4),
                5.40 + 0.04 * ((index + 3) % 4),
                0.18 + 0.02 * ((index) % 3),
                0.10 + 0.02 * ((index + 1) % 3),
                0.11 + 0.01 * ((index + 2) % 3),
            )
            name = f"mp0b_pathological_full_triclinic_{full_triclinic_counter:02d}"
            full_triclinic_counter += 1
            pbc = [True, True, True]
        elif mode == 2:
            cell = _orthorhombic_cell(
                5.90 + 0.05 * ((index + 3) % 3),
                6.50 + 0.07 * ((index + 1) % 3),
                8.60 + 0.06 * ((index + 2) % 3),
            )
            name = f"mp0b_pathological_partial_xy_pbc_{partial_xy_counter:02d}"
            partial_xy_counter += 1
            pbc = [True, True, False]
        else:
            cell = _orthorhombic_cell(
                7.20 + 0.07 * ((index + 1) % 3),
                6.10 + 0.05 * ((index + 2) % 3),
                4.50 + 0.08 * ((index + 3) % 3),
            )
            name = f"mp0b_pathological_partial_xz_pbc_{partial_xz_counter:02d}"
            partial_xz_counter += 1
            pbc = [True, False, True]

        structures.append(_build_periodic_structure(ase, name, species, cell, pbc, index))
    return structures


def run_mp0b_layer_flow_structure(
    data: Any,
    torch_geometric: Any,
    get_edge_vectors_and_lengths: Any,
    scatter_sum: Any,
    torch: Any,
    model: Any,
    table: Any,
    name: str,
    atoms: Any,
    include_intermediates: bool = True,
    include_stress_virials: bool = False,
) -> dict[str, Any]:
    ensure_mace_reference_default_dtype(torch, torch.float64)
    config = data.config_from_atoms(atoms, head_name=model.heads[0])
    atomic_data = data.AtomicData.from_config(
        config,
        z_table=table,
        cutoff=float(model.r_max),
        heads=model.heads,
    )
    batch = next(
        iter(torch_geometric.dataloader.DataLoader([atomic_data], batch_size=1))
    ).to_dict()
    assert_floating_batch_dtype(torch, batch, torch.float64, "MACE AtomicData.from_config")
    batch = {
        key: value.to(torch.float64)
        if torch.is_tensor(value) and value.is_floating_point()
        else value
        for key, value in batch.items()
    }
    assert_floating_batch_dtype(torch, batch, torch.float64, "MACE reference batch")
    output = model(
        batch,
        training=False,
        compute_force=True,
        compute_virials=include_stress_virials,
        compute_stress=include_stress_virials,
    )
    outputs = {
        "energy": as_list(output["energy"]),
        "node_energy": as_list(output["node_energy"]),
        "interaction_energy": as_list(output["interaction_energy"]),
        "forces": as_list(output["forces"]),
        "node_feats": as_list(output["node_feats"]),
    }
    if include_stress_virials:
        outputs["virials"] = as_list(output["virials"])
        if "stress" in output:
            outputs["stress"] = as_list(output["stress"])
    structure = {
        "name": name,
        "atomic_numbers": list(map(int, atoms.numbers.tolist())),
        "positions": atoms.positions.tolist(),
        "cell": atoms.cell.array.tolist(),
        "pbc": list(map(bool, atoms.pbc.tolist())),
        "batch": {
            "edge_index": as_list(batch["edge_index"]),
            "shifts": as_list(batch["shifts"]),
            "unit_shifts": as_list(batch["unit_shifts"]),
            "node_attrs": as_list(batch["node_attrs"]),
            "head": as_list(batch["head"]),
        },
        "outputs": outputs,
    }
    if include_intermediates:
        with torch.no_grad():
            structure["intermediates"] = mp0b_layer_flow_intermediates(
                get_edge_vectors_and_lengths, scatter_sum, torch, model, batch
            )
    return structure


def mp0b_layer_flow_intermediates(
    get_edge_vectors_and_lengths: Any,
    scatter_sum: Any,
    torch: Any,
    model: Any,
    batch: dict[str, Any],
) -> dict[str, Any]:
    positions = batch["positions"]
    edge_index = batch["edge_index"]
    node_attrs = batch["node_attrs"]
    node_heads = batch["head"][batch["batch"]].to(torch.int64)
    num_nodes = positions.shape[0]
    num_graphs = int(batch["ptr"].numel() - 1)
    num_atoms_arange = torch.arange(num_nodes, device=positions.device)

    vectors, lengths = get_edge_vectors_and_lengths(
        positions=positions,
        edge_index=edge_index,
        shifts=batch["shifts"],
    )
    node_e0 = model.atomic_energies_fn(node_attrs)[num_atoms_arange, node_heads]
    e0 = scatter_sum(src=node_e0, index=batch["batch"], dim=0, dim_size=num_graphs).to(
        vectors.dtype
    )
    node_feats = model.node_embedding(node_attrs)
    edge_attrs = model.spherical_harmonics(vectors)
    edge_feats, cutoff = model.radial_embedding(
        lengths, node_attrs, edge_index, model.atomic_numbers
    )
    pair_node_energy = model.pair_repulsion_fn(
        lengths, node_attrs, edge_index, model.atomic_numbers
    )
    node_feats_list = []
    layer_records = []
    for layer, (interaction, product) in enumerate(
        zip(model.interactions, model.products)
    ):
        receiver = edge_index[1]
        node_feats_in = node_feats
        sc = (
            interaction.skip_tp(node_feats, node_attrs)
            if "Residual" in type(interaction).__name__
            else None
        )
        linear_up = interaction.linear_up(node_feats)
        tp_weights_raw = interaction.conv_tp_weights(edge_feats)
        edge_density_raw = torch.tanh(interaction.density_fn(edge_feats) ** 2)
        tp_weights = tp_weights_raw if cutoff is None else tp_weights_raw * cutoff
        edge_density = edge_density_raw if cutoff is None else edge_density_raw * cutoff
        density = scatter_sum(
            src=edge_density,
            index=receiver,
            dim=0,
            dim_size=num_nodes,
        )
        edge_messages = interaction.conv_tp(linear_up[edge_index[0]], edge_attrs, tp_weights)
        scatter_message = scatter_sum(
            src=edge_messages,
            index=edge_index[1],
            dim=0,
            dim_size=num_nodes,
        )
        linear_message = interaction.linear(scatter_message) / (density + 1)
        if sc is None:
            selected_message = interaction.skip_tp(linear_message, node_attrs)
            interaction_sc = None
        else:
            selected_message = linear_message
            interaction_sc = sc
        interaction_out = interaction.reshape(selected_message)
        product_symmetric = product.symmetric_contractions(interaction_out, node_attrs)
        product_linear = product.linear(product_symmetric)
        product_out = (
            product_linear + interaction_sc
            if interaction_sc is not None
            else product_linear
        )
        layer_records.append(
            {
                "layer": layer,
                "interaction_class": f"{type(interaction).__module__}.{type(interaction).__name__}",
                "product_class": f"{type(product).__module__}.{type(product).__name__}",
                "node_feats_in": as_list(node_feats_in),
                "skip_connection": None if sc is None else as_list(sc),
                "linear_up": as_list(linear_up),
                "tp_weights_raw": as_list(tp_weights_raw),
                "tp_weights": as_list(tp_weights),
                "edge_density_raw": as_list(edge_density_raw),
                "edge_density": as_list(edge_density),
                "density": as_list(density),
                "edge_messages": as_list(edge_messages),
                "scatter_message": as_list(scatter_message),
                "linear_message": as_list(linear_message),
                "selected_message": as_list(selected_message),
                "interaction_out": as_list(interaction_out),
                "product_symmetric": as_list(product_symmetric),
                "product_linear": as_list(product_linear),
                "product_out": as_list(product_out),
            }
        )
        node_feats = product_out
        node_feats_list.append(node_feats)

    readout_records = []
    node_es_list = [pair_node_energy]
    for layer, readout in enumerate(model.readouts):
        feat_idx = -1 if len(model.readouts) == 1 else layer
        all_heads = readout(node_feats_list[feat_idx], node_heads)
        selected = all_heads[num_atoms_arange, node_heads]
        node_es_list.append(selected)
        readout_records.append(
            {
                "layer": layer,
                "class": f"{type(readout).__module__}.{type(readout).__name__}",
                "source_layer": feat_idx,
                "all_heads": as_list(all_heads),
                "selected": as_list(selected),
            }
        )
    node_inter_es_raw = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
    node_inter_es = model.scale_shift(node_inter_es_raw, node_heads)
    inter_e = scatter_sum(node_inter_es, batch["batch"], dim=-1, dim_size=num_graphs)
    total_energy = e0 + inter_e
    return {
        "vectors": as_list(vectors),
        "lengths": as_list(lengths),
        "node_e0": as_list(node_e0),
        "e0": as_list(e0),
        "node_embedding": as_list(model.node_embedding(node_attrs)),
        "edge_attrs": as_list(edge_attrs),
        "edge_feats": as_list(edge_feats),
        "cutoff": None if cutoff is None else as_list(cutoff),
        "pair_node_energy": as_list(pair_node_energy),
        "layers": layer_records,
        "readouts": readout_records,
        "node_inter_energy_raw": as_list(node_inter_es_raw),
        "node_inter_energy": as_list(node_inter_es),
        "interaction_energy": as_list(inter_e),
        "total_energy": as_list(total_energy),
        "node_feats_out": as_list(torch.cat(node_feats_list, dim=-1)),
    }


def one_layer_forward_intermediates(
    get_edge_vectors_and_lengths: Any,
    scatter_sum: Any,
    torch: Any,
    model: Any,
    batch: dict[str, Any],
) -> dict[str, Any]:
    positions = batch["positions"]
    edge_index = batch["edge_index"]
    node_attrs = batch["node_attrs"]
    node_heads = batch["head"][batch["batch"]].to(torch.int64)
    num_nodes = positions.shape[0]
    num_graphs = int(batch["ptr"].numel() - 1)
    num_atoms_arange = torch.arange(num_nodes, device=positions.device)

    vectors, lengths = get_edge_vectors_and_lengths(
        positions=positions,
        edge_index=edge_index,
        shifts=batch["shifts"],
    )
    node_e0 = model.atomic_energies_fn(node_attrs)[num_atoms_arange, node_heads]
    e0 = scatter_sum(
        src=node_e0,
        index=batch["batch"],
        dim=0,
        dim_size=num_graphs,
    ).to(vectors.dtype)
    node_embedding = model.node_embedding(node_attrs)
    edge_attrs = model.spherical_harmonics(vectors)
    edge_feats, cutoff = model.radial_embedding(
        lengths, node_attrs, edge_index, model.atomic_numbers
    )
    polynomial_cutoff = model.radial_embedding.cutoff_fn(lengths)

    interaction = model.interactions[0]
    product = model.products[0]
    readout = model.readouts[0]

    sc = interaction.skip_tp(node_embedding, node_attrs)
    linear_up = interaction.linear_up(node_embedding)
    tp_weights_raw = interaction.conv_tp_weights(edge_feats)
    tp_weights = tp_weights_raw if cutoff is None else tp_weights_raw * cutoff
    edge_messages = interaction.conv_tp(
        linear_up[edge_index[0]], edge_attrs, tp_weights
    )
    scatter_message = scatter_sum(
        src=edge_messages,
        index=edge_index[1],
        dim=0,
        dim_size=linear_up.shape[0],
    )
    linear_message = interaction.linear(scatter_message) / interaction.avg_num_neighbors
    reshaped_message = interaction.reshape(linear_message)
    interaction_out, interaction_sc = interaction(
        node_attrs=node_attrs,
        node_feats=node_embedding,
        edge_attrs=edge_attrs,
        edge_feats=edge_feats,
        edge_index=edge_index,
        cutoff=cutoff,
        first_layer=True,
    )
    product_out = product(node_feats=interaction_out, sc=interaction_sc, node_attrs=node_attrs)
    readout_all_heads = readout(product_out, node_heads)
    readout_node_energy = readout_all_heads[num_atoms_arange, node_heads]
    node_inter_energy = model.scale_shift(readout_node_energy, node_heads)
    interaction_energy = scatter_sum(
        src=node_inter_energy,
        index=batch["batch"],
        dim=-1,
        dim_size=num_graphs,
    )

    return {
        "vectors": as_list(vectors),
        "lengths": as_list(lengths),
        "node_e0": as_list(node_e0),
        "e0": as_list(e0),
        "node_embedding": as_list(node_embedding),
        "edge_attrs": as_list(edge_attrs),
        "edge_feats": as_list(edge_feats),
        "polynomial_cutoff": as_list(polynomial_cutoff),
        "interaction_cutoff_argument": None if cutoff is None else as_list(cutoff),
        "skip_connection": as_list(sc),
        "linear_up": as_list(linear_up),
        "tp_weights_raw": as_list(tp_weights_raw),
        "tp_weights": as_list(tp_weights),
        "edge_messages": as_list(edge_messages),
        "scatter_message": as_list(scatter_message),
        "linear_message": as_list(linear_message),
        "reshaped_message": as_list(reshaped_message),
        "interaction_out": as_list(interaction_out),
        "interaction_sc": as_list(interaction_sc),
        "product_out": as_list(product_out),
        "readout_all_heads": as_list(readout_all_heads),
        "readout_node_energy": as_list(readout_node_energy),
        "node_inter_energy": as_list(node_inter_energy),
        "interaction_energy": as_list(interaction_energy),
    }


def model_state_dict(model: Any) -> list[dict[str, Any]]:
    return [
        {"name": name, **tensor_payload(tensor)}
        for name, tensor in model.state_dict().items()
    ]


def interaction_summary(interaction: Any) -> dict[str, Any]:
    return {
        "class": f"{type(interaction).__module__}.{type(interaction).__name__}",
        "avg_num_neighbors": float(interaction.avg_num_neighbors),
        "hidden_irreps": str(interaction.hidden_irreps),
        "target_irreps": str(interaction.target_irreps),
        "conv_tp": {
            "irreps_in1": str(interaction.conv_tp.irreps_in1),
            "irreps_in2": str(interaction.conv_tp.irreps_in2),
            "irreps_out": str(interaction.conv_tp.irreps_out),
            "weight_numel": int(interaction.conv_tp.weight_numel),
            "instructions": [
                {
                    "i_in1": int(instruction.i_in1),
                    "i_in2": int(instruction.i_in2),
                    "i_out": int(instruction.i_out),
                    "connection_mode": instruction.connection_mode,
                    "has_weight": bool(instruction.has_weight),
                    "path_weight": float(instruction.path_weight),
                    "path_shape": list(instruction.path_shape),
                }
                for instruction in interaction.conv_tp.instructions
            ],
        },
        "skip_tp": {
            "irreps_in1": str(interaction.skip_tp.irreps_in1),
            "irreps_in2": str(interaction.skip_tp.irreps_in2),
            "irreps_out": str(interaction.skip_tp.irreps_out),
            "weight_numel": int(interaction.skip_tp.weight_numel),
        },
    }


def tensor_product_plan_json(tp: Any, np: Any, torch: Any, o3: Any) -> dict[str, Any]:
    meta = tensor_product_plan_metadata(tp, np, torch, o3)
    return {
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


def product_summary(product: Any) -> dict[str, Any]:
    symmetric = product.symmetric_contractions
    return {
        "class": f"{type(product).__module__}.{type(product).__name__}",
        "symmetric_contraction_class": (
            f"{type(symmetric).__module__}.{type(symmetric).__name__}"
        ),
        "irreps_in": str(symmetric.irreps_in),
        "irreps_out": str(symmetric.irreps_out),
        "linear_irreps_in": str(product.linear.irreps_in),
        "linear_irreps_out": str(product.linear.irreps_out),
    }


def readout_summary(readout: Any) -> dict[str, Any]:
    entry = {"class": f"{type(readout).__module__}.{type(readout).__name__}"}
    if hasattr(readout, "linear"):
        entry.update(
            {
                "linear_irreps_in": str(readout.linear.irreps_in),
                "linear_irreps_out": str(readout.linear.irreps_out),
            }
        )
    elif hasattr(readout, "linear_1") and hasattr(readout, "linear_2"):
        entry.update(
            {
                "linear_1_irreps_in": str(readout.linear_1.irreps_in),
                "linear_1_irreps_out": str(readout.linear_1.irreps_out),
                "linear_2_irreps_in": str(readout.linear_2.irreps_in),
                "linear_2_irreps_out": str(readout.linear_2.irreps_out),
            }
        )
    else:
        raise SystemExit(f"unsupported MACE-lite readout block: {type(readout)}")
    return entry


def model_module_summary(model: Any) -> dict[str, Any]:
    return {
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "num_interactions": int(model.num_interactions.detach().cpu().item()),
        "r_max": float(model.r_max.detach().cpu().item()),
        "heads": list(model.heads),
        "atomic_numbers": as_list(model.atomic_numbers),
        "interaction_blocks": [
            interaction_summary(interaction) for interaction in model.interactions
        ],
        "products": [product_summary(product) for product in model.products],
        "readouts": [readout_summary(readout) for readout in model.readouts],
    }


def generate_reference_fixture(
    mace_dir: Path,
    num_interactions: int,
    interaction_cls_name: str = "RealAgnosticResidualInteractionBlock",
    interaction_cls_first_name: str | None = None,
    fixture_name: str | None = None,
    distance_transform: str = "None",
    pair_repulsion: bool = False,
    max_ell: int = 2,
) -> dict[str, Any]:
    if num_interactions not in (1, 2):
        raise SystemExit("MACE-lite reference fixtures currently support 1 or 2 layers")
    if interaction_cls_first_name is None:
        interaction_cls_first_name = interaction_cls_name
    if fixture_name is None:
        fixture_name = f"mace_lite_layers{num_interactions}"
    ase, e3nn, mace, np, torch, torch_functional, o3, mace_parts = import_mace_reference(
        mace_dir
    )
    (
        data,
        modules,
        tools,
        torch_geometric,
        get_edge_vectors_and_lengths,
        scatter_sum,
    ) = mace_parts
    model, table = build_reference_model(
        np,
        torch,
        torch_functional,
        o3,
        modules,
        tools,
        num_interactions,
        interaction_cls_name=interaction_cls_name,
        interaction_cls_first_name=interaction_cls_first_name,
        distance_transform=distance_transform,
        pair_repulsion=pair_repulsion,
        max_ell=max_ell,
    )
    run_structure = run_one_layer_structure if num_interactions == 1 else run_two_layer_structure
    structures = [
        run_structure(
            data,
            torch_geometric,
            get_edge_vectors_and_lengths,
            scatter_sum,
            torch,
            model,
            table,
            name,
            atoms,
        )
        for name, atoms in deterministic_reference_structures(ase)
    ]
    supported_blocks = [
        "ScaleShiftMACE",
        "LinearNodeEmbeddingBlock",
        "BesselBasis",
        "PolynomialCutoff",
        interaction_cls_first_name,
        "TensorProduct",
        "EquivariantProductBasisBlock",
        "SymmetricContraction",
        "LinearReadoutBlock",
        "ScaleShiftBlock",
    ]
    if distance_transform == "Agnesi":
        supported_blocks.insert(4, "AgnesiTransform")
    elif distance_transform != "None":
        raise SystemExit(f"unsupported MACE-lite distance_transform: {distance_transform}")
    if pair_repulsion:
        supported_blocks.insert(5 if distance_transform == "Agnesi" else 4, "ZBLBasis")
    if num_interactions > 1 and interaction_cls_name not in supported_blocks:
        supported_blocks.insert(-1, interaction_cls_name)
    if num_interactions > 1:
        supported_blocks.insert(-1, "NonLinearReadoutBlock")
    config = {
        "r_max": 4.0,
        "num_bessel": 1,
        "num_polynomial_cutoff": 6,
        "max_ell": max_ell,
        "num_interactions": num_interactions,
        "num_elements": 2,
        "hidden_irreps": "1x0e",
        "MLP_irreps": "1x0e",
        "radial_MLP": [1],
        "radial_type": "bessel",
        "correlation": 3,
        "avg_num_neighbors": 1.0,
        "pair_repulsion": pair_repulsion,
        "distance_transform": distance_transform,
        "heads": ["Default"],
    }
    if (
        interaction_cls_name != "RealAgnosticResidualInteractionBlock"
        or interaction_cls_first_name != "RealAgnosticResidualInteractionBlock"
    ):
        config["interaction_cls"] = interaction_cls_name
        config["interaction_cls_first"] = interaction_cls_first_name
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "reference": {
            "name": "mace",
            "repository": REFERENCE_REPOSITORY,
            "commit": read_pinned_commit(mace_dir),
            "package_version": read_mace_version(mace_dir),
        },
        "python": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "ase_version": ase.__version__,
            "e3nn_version": getattr(e3nn, "__version__", "unknown"),
            "mace_version": getattr(mace, "__version__", "unknown"),
        },
        "dtype": "float64",
        "seed": REFERENCE_SEED,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "architecture": "mace-lite-v1",
            "name": fixture_name,
            "supported_blocks": supported_blocks,
            "config": config,
            "modules": model_module_summary(model),
            "state_dict": model_state_dict(model),
        },
        "structures": structures,
        "conventions": {
            "edge_direction": "r_ij = R_j + shift_ij - R_i",
            "dtype": "float64",
            "forces": "F = -dE/dR from upstream MACE/PyTorch autograd",
            "spherical_harmonics_normalization": "component",
            "spherical_harmonics_normalize_input": True,
        },
    }


def generate_one_layer_reference_fixture(mace_dir: Path) -> dict[str, Any]:
    return generate_reference_fixture(mace_dir, 1)


def generate_two_layer_reference_fixture(mace_dir: Path) -> dict[str, Any]:
    return generate_reference_fixture(mace_dir, 2)


def generate_density_residual_reference_fixture(mace_dir: Path) -> dict[str, Any]:
    return generate_reference_fixture(
        mace_dir,
        2,
        interaction_cls_name="RealAgnosticDensityResidualInteractionBlock",
        interaction_cls_first_name="RealAgnosticDensityResidualInteractionBlock",
        fixture_name="mace_lite_density_residual_layers2",
    )


def generate_density_mixed_reference_fixture(mace_dir: Path) -> dict[str, Any]:
    return generate_reference_fixture(
        mace_dir,
        2,
        interaction_cls_name="RealAgnosticDensityResidualInteractionBlock",
        interaction_cls_first_name="RealAgnosticDensityInteractionBlock",
        fixture_name="mace_lite_density_mixed_layers2",
    )


def generate_agnesi_density_mixed_reference_fixture(mace_dir: Path) -> dict[str, Any]:
    return generate_reference_fixture(
        mace_dir,
        2,
        interaction_cls_name="RealAgnosticDensityResidualInteractionBlock",
        interaction_cls_first_name="RealAgnosticDensityInteractionBlock",
        fixture_name="mace_lite_agnesi_density_mixed_layers2",
        distance_transform="Agnesi",
    )


def generate_agnesi_zbl_density_mixed_reference_fixture(mace_dir: Path) -> dict[str, Any]:
    return generate_reference_fixture(
        mace_dir,
        2,
        interaction_cls_name="RealAgnosticDensityResidualInteractionBlock",
        interaction_cls_first_name="RealAgnosticDensityInteractionBlock",
        fixture_name="mace_lite_agnesi_zbl_density_mixed_layers2",
        distance_transform="Agnesi",
        pair_repulsion=True,
    )


def generate_lmax3_reference_fixture(mace_dir: Path) -> dict[str, Any]:
    return generate_reference_fixture(
        mace_dir,
        2,
        fixture_name="mace_lite_lmax3_layers2",
        max_ell=3,
    )


def generate_pbc_reference_fixture(
    mace_dir: Path,
    num_interactions: int,
    interaction_cls_name: str = "RealAgnosticResidualInteractionBlock",
    interaction_cls_first_name: str | None = None,
    fixture_name: str | None = None,
    model_fixture: str | None = None,
    distance_transform: str = "None",
    pair_repulsion: bool = False,
    max_ell: int = 2,
) -> dict[str, Any]:
    if num_interactions != 2:
        raise SystemExit("MACE-lite PBC reference fixture currently supports 2 layers")
    if interaction_cls_first_name is None:
        interaction_cls_first_name = interaction_cls_name
    if fixture_name is None:
        fixture_name = f"mace_lite_layers{num_interactions}"
    if model_fixture is None:
        model_fixture = f"models/{fixture_name}.e3rs"
    ase, e3nn, mace, np, torch, torch_functional, o3, mace_parts = import_mace_reference(
        mace_dir
    )
    (
        data,
        modules,
        tools,
        torch_geometric,
        get_edge_vectors_and_lengths,
        scatter_sum,
    ) = mace_parts
    model, table = build_reference_model(
        np,
        torch,
        torch_functional,
        o3,
        modules,
        tools,
        num_interactions,
        interaction_cls_name=interaction_cls_name,
        interaction_cls_first_name=interaction_cls_first_name,
        distance_transform=distance_transform,
        pair_repulsion=pair_repulsion,
        max_ell=max_ell,
    )
    structures = [
        run_two_layer_structure(
            data,
            torch_geometric,
            get_edge_vectors_and_lengths,
            scatter_sum,
            torch,
            model,
            table,
            name,
            atoms,
            include_stress_virials=True,
        )
        for name, atoms in deterministic_pbc_reference_structures(ase)
    ]
    config = {
        "r_max": 4.0,
        "num_bessel": 1,
        "num_polynomial_cutoff": 6,
        "max_ell": max_ell,
        "num_interactions": num_interactions,
        "num_elements": 2,
        "hidden_irreps": "1x0e",
        "MLP_irreps": "1x0e",
        "radial_MLP": [1],
        "radial_type": "bessel",
        "correlation": 3,
        "avg_num_neighbors": 1.0,
        "pair_repulsion": pair_repulsion,
        "distance_transform": distance_transform,
        "heads": ["Default"],
    }
    if (
        interaction_cls_name != "RealAgnosticResidualInteractionBlock"
        or interaction_cls_first_name != "RealAgnosticResidualInteractionBlock"
    ):
        config["interaction_cls"] = interaction_cls_name
        config["interaction_cls_first"] = interaction_cls_first_name
    return {
        "schema_version": PBC_REFERENCE_SCHEMA_VERSION,
        "reference": {
            "name": "mace",
            "repository": REFERENCE_REPOSITORY,
            "commit": read_pinned_commit(mace_dir),
            "package_version": read_mace_version(mace_dir),
        },
        "python": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "ase_version": ase.__version__,
            "e3nn_version": getattr(e3nn, "__version__", "unknown"),
            "mace_version": getattr(mace, "__version__", "unknown"),
        },
        "dtype": "float64",
        "seed": REFERENCE_SEED,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "architecture": "mace-lite-v1",
            "name": fixture_name,
            "model_fixture": model_fixture,
            "config": config,
        },
        "structures": structures,
        "conventions": {
            "edge_direction": "r_ij = R_j + shift_ij - R_i",
            "dtype": "float64",
            "forces": "F = -dE/dR from upstream MACE/PyTorch autograd",
            "stress": "stress = dE/dD / abs(det(cell)) for symmetric displacement D",
            "virials": "virials = -dE/dD for symmetric displacement D",
            "spherical_harmonics_normalization": "component",
            "spherical_harmonics_normalize_input": True,
        },
    }


def generate_density_residual_pbc_reference_fixture(mace_dir: Path) -> dict[str, Any]:
    return generate_pbc_reference_fixture(
        mace_dir,
        2,
        interaction_cls_name="RealAgnosticDensityResidualInteractionBlock",
        interaction_cls_first_name="RealAgnosticDensityResidualInteractionBlock",
        fixture_name="mace_lite_density_residual_layers2",
        model_fixture="models/mace_lite_density_residual_layers2.e3rs",
    )


def generate_density_mixed_pbc_reference_fixture(mace_dir: Path) -> dict[str, Any]:
    return generate_pbc_reference_fixture(
        mace_dir,
        2,
        interaction_cls_name="RealAgnosticDensityResidualInteractionBlock",
        interaction_cls_first_name="RealAgnosticDensityInteractionBlock",
        fixture_name="mace_lite_density_mixed_layers2",
        model_fixture="models/mace_lite_density_mixed_layers2.e3rs",
    )


def generate_agnesi_density_mixed_pbc_reference_fixture(mace_dir: Path) -> dict[str, Any]:
    return generate_pbc_reference_fixture(
        mace_dir,
        2,
        interaction_cls_name="RealAgnosticDensityResidualInteractionBlock",
        interaction_cls_first_name="RealAgnosticDensityInteractionBlock",
        fixture_name="mace_lite_agnesi_density_mixed_layers2",
        model_fixture="models/mace_lite_agnesi_density_mixed_layers2.e3rs",
        distance_transform="Agnesi",
    )


def generate_agnesi_zbl_density_mixed_pbc_reference_fixture(mace_dir: Path) -> dict[str, Any]:
    return generate_pbc_reference_fixture(
        mace_dir,
        2,
        interaction_cls_name="RealAgnosticDensityResidualInteractionBlock",
        interaction_cls_first_name="RealAgnosticDensityInteractionBlock",
        fixture_name="mace_lite_agnesi_zbl_density_mixed_layers2",
        model_fixture="models/mace_lite_agnesi_zbl_density_mixed_layers2.e3rs",
        distance_transform="Agnesi",
        pair_repulsion=True,
    )


def generate_lmax3_pbc_reference_fixture(mace_dir: Path) -> dict[str, Any]:
    return generate_pbc_reference_fixture(
        mace_dir,
        2,
        fixture_name="mace_lite_lmax3_layers2",
        model_fixture="models/mace_lite_lmax3_layers2.e3rs",
        max_ell=3,
    )


def generate_mp0b_layer_flow_fixture(args: argparse.Namespace) -> dict[str, Any]:
    (
        ase,
        _e3nn,
        mace,
        np,
        torch,
        _torch_functional,
        o3,
        refs,
        model,
    ) = import_mace_mp0b_reference(args)
    ensure_mace_reference_default_dtype(torch, torch.float64)
    model = model.to(torch.float64).eval()
    data, _modules, tools, torch_geometric, get_edge_vectors_and_lengths, scatter_sum = refs
    atomic_numbers = [int(value) for value in model.atomic_numbers.detach().cpu().tolist()]
    manifest = read_json(args.manifest)
    expected_sha256 = manifest["artifact"]["sha256"]
    actual_sha256 = hash_file(args.artifact)
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            "MACE-MP0b artifact SHA256 mismatch:\n"
            f"expected: {expected_sha256}\n"
            f"actual:   {actual_sha256}"
        )
    table = tools.AtomicNumberTable(atomic_numbers)
    structures = [
        run_mp0b_layer_flow_structure(
            data,
            torch_geometric,
            get_edge_vectors_and_lengths,
            scatter_sum,
            torch,
            model,
            table,
            name,
            atoms,
            include_intermediates=include_intermediates,
            include_stress_virials=include_stress_virials,
        )
        for name, atoms, include_intermediates, include_stress_virials in deterministic_mp0b_flow_structures(
            ase
        )
    ]
    return {
        "schema_version": MP0B_FLOW_SCHEMA_VERSION,
        "reference": {
            "name": "mace",
            "repository": REFERENCE_REPOSITORY,
            "commit": read_pinned_commit(args.mace_dir),
            "package_version": getattr(mace, "__version__", "unknown"),
        },
        "artifact": {
            "manifest": str(args.manifest),
            "filename": args.artifact.name,
            "sha256": actual_sha256,
        },
        "python": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "mace_version": getattr(mace, "__version__", "unknown"),
        },
        "dtype": "float64",
        "reference_compute_dtype": "torch.float64",
        "torch_default_dtype_before_atomic_data": "torch.float64",
        "model": {
            "architecture": "mace-mp0b-layer-flow-reference",
            "class": f"{type(model).__module__}.{type(model).__name__}",
            "heads": list(model.heads),
            "atomic_numbers": atomic_numbers,
            "r_max": float(model.r_max.detach().cpu().item()),
            "num_interactions": int(model.num_interactions.detach().cpu().item()),
            "num_bessel": int(model.radial_embedding.bessel_fn.bessel_weights.numel()),
            "radial_cutoff_p": int(model.radial_embedding.cutoff_fn.p.detach().cpu().item()),
            "pair_repulsion_cutoff_p": int(
                model.pair_repulsion_fn.p.detach().cpu().item()
            ),
            "max_ell": 3,
            "hidden_irreps": "128x0e+128x1o+128x2e+128x3o",
            "conv_tp_plans": [
                {
                    "layer": layer_index,
                    **tensor_product_plan_json(
                        model.interactions[layer_index].conv_tp, np, torch, o3
                    ),
                }
                for layer_index in range(2)
            ],
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "structures": structures,
        "conventions": {
            "edge_vector": "R_j + shift - R_i",
            "force_reference": "F = -dE/dR from upstream MACE/PyTorch autograd",
            "stress": "stress = dE/dD / abs(det(cell)) for symmetric displacement D",
            "virials": "virials = -dE/dD for symmetric displacement D",
            "runtime_claim": "forward intermediates plus upstream force/stress reference; this is not a broad Rust MACE-MP0b representative-structure parity claim",
        },
    }


def generate_mp0b_broad_reference_fixture(args: argparse.Namespace) -> dict[str, Any]:
    (
        ase,
        _e3nn,
        mace,
        _np,
        torch,
        _torch_functional,
        _o3,
        refs,
        model,
    ) = import_mace_mp0b_reference(args)
    ensure_mace_reference_default_dtype(torch, torch.float64)
    model = model.to(torch.float64).eval()
    data, _modules, tools, torch_geometric, get_edge_vectors_and_lengths, scatter_sum = refs
    atomic_numbers = [int(value) for value in model.atomic_numbers.detach().cpu().tolist()]
    manifest = read_json(args.manifest)
    expected_sha256 = manifest["artifact"]["sha256"]
    actual_sha256 = hash_file(args.artifact)
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            "MACE-MP0b artifact SHA256 mismatch:\n"
            f"expected: {expected_sha256}\n"
            f"actual:   {actual_sha256}"
        )
    table = tools.AtomicNumberTable(atomic_numbers)
    structures = [
        run_mp0b_layer_flow_structure(
            data,
            torch_geometric,
            get_edge_vectors_and_lengths,
            scatter_sum,
            torch,
            model,
            table,
            name,
            atoms,
            include_intermediates=False,
            include_stress_virials=include_stress_virials,
        )
        for name, atoms, include_stress_virials in deterministic_mp0b_broad_structures(
            ase
        )
    ]
    return {
        "schema_version": MP0B_BROAD_SCHEMA_VERSION,
        "reference": {
            "name": "mace",
            "repository": REFERENCE_REPOSITORY,
            "commit": read_pinned_commit(args.mace_dir),
            "package_version": getattr(mace, "__version__", "unknown"),
        },
        "artifact": {
            "manifest": str(args.manifest),
            "filename": args.artifact.name,
            "sha256": actual_sha256,
        },
        "python": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "mace_version": getattr(mace, "__version__", "unknown"),
        },
        "dtype": "float64",
        "reference_compute_dtype": "torch.float64",
        "torch_default_dtype_before_atomic_data": "torch.float64",
        "model": {
            "architecture": "mace-mp0b-broad-reference",
            "class": f"{type(model).__module__}.{type(model).__name__}",
            "heads": list(model.heads),
            "atomic_numbers": atomic_numbers,
            "r_max": float(model.r_max.detach().cpu().item()),
            "num_interactions": int(model.num_interactions.detach().cpu().item()),
            "max_ell": 3,
            "hidden_irreps": "128x0e+128x1o+128x2e+128x3o",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "structures": structures,
        "conventions": {
            "edge_vector": "R_j + shift - R_i",
            "force_reference": "F = -dE/dR from upstream MACE/PyTorch autograd",
            "stress": "stress = dE/dD / abs(det(reference cell)) for symmetric displacement D; for partial PBC, MACE expands non-periodic cell rows to (max_abs_position + 1) * 5 * cutoff identity rows before stress normalization",
            "virials": "virials = -dE/dD for symmetric displacement D",
            "runtime_claim": (
                "broader chemistry, near-cutoff, full-PBC stress/virial, and "
                "partial-PBC multi-image diagnostics for the scoped MP0b "
                "runtime, including MACE reference-cell partial-PBC stress"
            ),
        },
    }


def generate_osaka26_reference_fixture(args: argparse.Namespace) -> dict[str, Any]:
    (
        ase,
        _e3nn,
        mace,
        _np,
        torch,
        _torch_functional,
        _o3,
        refs,
        model,
    ) = import_mace_artifact_reference(args, args.artifact)
    ensure_mace_reference_default_dtype(torch, torch.float64)
    model = model.to(torch.float64).eval()
    data, _modules, tools, torch_geometric, get_edge_vectors_and_lengths, scatter_sum = refs
    atomic_numbers = [int(value) for value in model.atomic_numbers.detach().cpu().tolist()]
    manifest = read_json(args.manifest)
    expected_sha256 = manifest["artifact"]["sha256"]
    actual_sha256 = hash_file(args.artifact)
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            "MACE-Osaka26 artifact SHA256 mismatch:\n"
            f"expected: {expected_sha256}\n"
            f"actual:   {actual_sha256}"
        )
    table = tools.AtomicNumberTable(atomic_numbers)
    structures = [
        run_mp0b_layer_flow_structure(
            data,
            torch_geometric,
            get_edge_vectors_and_lengths,
            scatter_sum,
            torch,
            model,
            table,
            name,
            atoms,
            include_intermediates=False,
            include_stress_virials=include_stress_virials,
        )
        for name, atoms, include_stress_virials in deterministic_osaka26_reference_structures(
            ase
        )
    ]
    return {
        "schema_version": OSAKA26_REFERENCE_SCHEMA_VERSION,
        "reference": {
            "name": "mace",
            "repository": REFERENCE_REPOSITORY,
            "commit": read_pinned_commit(args.mace_dir),
            "package_version": getattr(mace, "__version__", "unknown"),
        },
        "artifact": {
            "manifest": str(args.manifest),
            "filename": args.artifact.name,
            "sha256": actual_sha256,
        },
        "python": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "mace_version": getattr(mace, "__version__", "unknown"),
        },
        "dtype": "float64",
        "reference_compute_dtype": "torch.float64",
        "torch_default_dtype_before_atomic_data": "torch.float64",
        "model": {
            "architecture": "mace-osaka26-reference",
            "class": f"{type(model).__module__}.{type(model).__name__}",
            "heads": list(model.heads),
            "atomic_numbers": atomic_numbers,
            "r_max": float(model.r_max.detach().cpu().item()),
            "num_interactions": int(model.num_interactions.detach().cpu().item()),
            "num_bessel": int(model.radial_embedding.bessel_fn.bessel_weights.numel()),
            "radial_cutoff_p": int(model.radial_embedding.cutoff_fn.p.detach().cpu().item()),
            "pair_repulsion_cutoff_p": int(
                model.pair_repulsion_fn.p.detach().cpu().item()
            ),
            "max_ell": 3,
            "hidden_irreps": "128x0e+128x1o+128x2e+128x3o",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "structures": structures,
        "conventions": {
            "edge_vector": "R_j + shift - R_i",
            "force_reference": "F = -dE/dR from upstream MACE/PyTorch autograd",
            "stress": "stress = dE/dD / abs(det(cell)) for symmetric displacement D",
            "virials": "virials = -dE/dD for symmetric displacement D",
            "runtime_claim": "upstream MACE-Osaka26 energy/force/stress reference for scoped Rust runtime parity",
        },
    }


def generate_mp0b_mptrj_mini_reference_fixture(args: argparse.Namespace) -> dict[str, Any]:
    (
        ase,
        _e3nn,
        mace,
        np,
        torch,
        _torch_functional,
        o3,
        refs,
        model,
    ) = import_mace_mp0b_reference(args)
    ensure_mace_reference_default_dtype(torch, torch.float64)
    model = model.to(torch.float64).eval()
    data, _modules, tools, torch_geometric, get_edge_vectors_and_lengths, scatter_sum = refs
    atomic_numbers = [int(value) for value in model.atomic_numbers.detach().cpu().tolist()]
    manifest = read_json(args.manifest)
    expected_sha256 = manifest["artifact"]["sha256"]
    actual_sha256 = hash_file(args.artifact)
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            "MACE-MP0b artifact SHA256 mismatch:\n"
            f"expected: {expected_sha256}\n"
            f"actual:   {actual_sha256}"
        )
    table = tools.AtomicNumberTable(atomic_numbers)
    structures = [
        run_mp0b_layer_flow_structure(
            data,
            torch_geometric,
            get_edge_vectors_and_lengths,
            scatter_sum,
            torch,
            model,
            table,
            name,
            atoms,
            include_intermediates=False,
            include_stress_virials=include_stress_virials,
        )
        for name, atoms, _include_intermediates, include_stress_virials in deterministic_mp0b_mptrj_mini_structures(
            ase
        )
    ]
    return {
        "schema_version": MP0B_MPTRJ_MINI_SCHEMA_VERSION,
        "reference": {
            "name": "mace",
            "repository": REFERENCE_REPOSITORY,
            "commit": read_pinned_commit(args.mace_dir),
            "package_version": getattr(mace, "__version__", "unknown"),
        },
        "artifact": {
            "manifest": str(args.manifest),
            "filename": args.artifact.name,
            "sha256": actual_sha256,
        },
        "python": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "mace_version": getattr(mace, "__version__", "unknown"),
        },
        "dtype": "float64",
        "reference_compute_dtype": "torch.float64",
        "torch_default_dtype_before_atomic_data": "torch.float64",
        "model": {
            "architecture": "mace-mp0b-mptrj-mini-reference",
            "class": f"{type(model).__module__}.{type(model).__name__}",
            "heads": list(model.heads),
            "atomic_numbers": atomic_numbers,
            "r_max": float(model.r_max.detach().cpu().item()),
            "num_interactions": int(model.num_interactions.detach().cpu().item()),
            "num_bessel": int(model.radial_embedding.bessel_fn.bessel_weights.numel()),
            "radial_cutoff_p": int(model.radial_embedding.cutoff_fn.p.detach().cpu().item()),
            "pair_repulsion_cutoff_p": int(
                model.pair_repulsion_fn.p.detach().cpu().item()
            ),
            "max_ell": 3,
            "hidden_irreps": "128x0e+128x1o+128x2e+128x3o",
            "conv_tp_plans": [
                {
                    "layer": layer_index,
                    **tensor_product_plan_json(
                        model.interactions[layer_index].conv_tp, np, torch, o3
                    ),
                }
                for layer_index in range(2)
            ],
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "structures": structures,
        "conventions": {
            "dataset_source": "deterministic MPtrj-like periodic regression structures; not raw sampled MPtrj dataset records",
            "edge_vector": "R_j + shift - R_i",
            "force_reference": "F = -dE/dR from upstream MACE/PyTorch autograd",
            "stress": "stress = dE/dD / abs(det(reference cell)) for full-PBC symmetric displacement D; for partial PBC, MACE expands non-periodic cell rows to (max_abs_position + 1) * 5 * cutoff identity rows before stress normalization",
            "virials": "virials = -dE/dD for symmetric displacement D",
            "runtime_claim": "upstream MP0b mini periodic parity fixture for focused regression coverage",
        },
    }


def generate_osaka26_mptrj_mini_reference_fixture(args: argparse.Namespace) -> dict[str, Any]:
    (
        ase,
        _e3nn,
        mace,
        _np,
        torch,
        _torch_functional,
        _o3,
        refs,
        model,
    ) = import_mace_artifact_reference(args, args.artifact)
    ensure_mace_reference_default_dtype(torch, torch.float64)
    model = model.to(torch.float64).eval()
    data, _modules, tools, torch_geometric, get_edge_vectors_and_lengths, scatter_sum = refs
    atomic_numbers = [int(value) for value in model.atomic_numbers.detach().cpu().tolist()]
    manifest = read_json(args.manifest)
    expected_sha256 = manifest["artifact"]["sha256"]
    actual_sha256 = hash_file(args.artifact)
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            "MACE-Osaka26 artifact SHA256 mismatch:\n"
            f"expected: {expected_sha256}\n"
            f"actual:   {actual_sha256}"
        )
    table = tools.AtomicNumberTable(atomic_numbers)
    structures = [
        run_mp0b_layer_flow_structure(
            data,
            torch_geometric,
            get_edge_vectors_and_lengths,
            scatter_sum,
            torch,
            model,
            table,
            name,
            atoms,
            include_intermediates=False,
            include_stress_virials=include_stress_virials,
        )
        for name, atoms, _include_intermediates, include_stress_virials in deterministic_osaka26_mptrj_mini_structures(
            ase
        )
    ]
    return {
        "schema_version": OSAKA26_MPTRJ_MINI_SCHEMA_VERSION,
        "reference": {
            "name": "mace",
            "repository": REFERENCE_REPOSITORY,
            "commit": read_pinned_commit(args.mace_dir),
            "package_version": getattr(mace, "__version__", "unknown"),
        },
        "artifact": {
            "manifest": str(args.manifest),
            "filename": args.artifact.name,
            "sha256": actual_sha256,
        },
        "python": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "mace_version": getattr(mace, "__version__", "unknown"),
        },
        "dtype": "float64",
        "reference_compute_dtype": "torch.float64",
        "torch_default_dtype_before_atomic_data": "torch.float64",
        "model": {
            "architecture": "mace-osaka26-mptrj-mini-reference",
            "class": f"{type(model).__module__}.{type(model).__name__}",
            "heads": list(model.heads),
            "atomic_numbers": atomic_numbers,
            "r_max": float(model.r_max.detach().cpu().item()),
            "num_interactions": int(model.num_interactions.detach().cpu().item()),
            "num_bessel": int(model.radial_embedding.bessel_fn.bessel_weights.numel()),
            "radial_cutoff_p": int(model.radial_embedding.cutoff_fn.p.detach().cpu().item()),
            "pair_repulsion_cutoff_p": int(
                model.pair_repulsion_fn.p.detach().cpu().item()
            ),
            "max_ell": 3,
            "hidden_irreps": "128x0e+128x1o+128x2e+128x3o",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "structures": structures,
        "conventions": {
            "dataset_source": "deterministic MPtrj-like periodic regression structures; not raw sampled MPtrj dataset records",
            "edge_vector": "R_j + shift - R_i",
            "force_reference": "F = -dE/dR from upstream MACE/PyTorch autograd",
            "stress": "stress = dE/dD / abs(det(reference cell)) for full-PBC symmetric displacement D; for partial PBC, MACE expands non-periodic cell rows to (max_abs_position + 1) * 5 * cutoff identity rows before stress normalization",
            "virials": "virials = -dE/dD for symmetric displacement D",
            "runtime_claim": "upstream Osaka26 mini periodic parity fixture for focused regression coverage",
        },
    }


def generate_periodic_pathological_reference_fixture(args: argparse.Namespace) -> dict[str, Any]:
    (
        ase,
        _e3nn,
        mace,
        np,
        torch,
        _torch_functional,
        o3,
        refs,
        model,
    ) = import_mace_mp0b_reference(args)
    data, _modules, tools, torch_geometric, get_edge_vectors_and_lengths, scatter_sum = refs
    atomic_numbers = [int(value) for value in model.atomic_numbers.detach().cpu().tolist()]
    manifest = read_json(args.manifest)
    expected_sha256 = manifest["artifact"]["sha256"]
    actual_sha256 = hash_file(args.artifact)
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            "MACE-MP0b artifact SHA256 mismatch:\n"
            f"expected: {expected_sha256}\n"
            f"actual:   {actual_sha256}"
        )
    table = tools.AtomicNumberTable(atomic_numbers)
    structures = [
        run_mp0b_layer_flow_structure(
            data,
            torch_geometric,
            get_edge_vectors_and_lengths,
            scatter_sum,
            torch,
            model,
            table,
            name,
            atoms,
            include_intermediates=False,
            include_stress_virials=include_stress_virials,
        )
        for name, atoms, _include_intermediates, include_stress_virials in deterministic_periodic_pathological_structures(
            ase
        )
    ]
    return {
        "schema_version": MACE_PERIODIC_PATHOLOGICAL_SCHEMA_VERSION,
        "reference": {
            "name": "mace",
            "repository": REFERENCE_REPOSITORY,
            "commit": read_pinned_commit(args.mace_dir),
            "package_version": getattr(mace, "__version__", "unknown"),
        },
        "artifact": {
            "manifest": str(args.manifest),
            "filename": args.artifact.name,
            "sha256": actual_sha256,
        },
        "python": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "mace_version": getattr(mace, "__version__", "unknown"),
        },
        "dtype": "float64",
        "model": {
            "architecture": "mace-pbc-pathological-reference",
            "class": f"{type(model).__module__}.{type(model).__name__}",
            "heads": list(model.heads),
            "atomic_numbers": atomic_numbers,
            "r_max": float(model.r_max.detach().cpu().item()),
            "num_interactions": int(model.num_interactions.detach().cpu().item()),
            "num_bessel": int(model.radial_embedding.bessel_fn.bessel_weights.numel()),
            "radial_cutoff_p": int(model.radial_embedding.cutoff_fn.p.detach().cpu().item()),
            "pair_repulsion_cutoff_p": int(
                model.pair_repulsion_fn.p.detach().cpu().item()
            ),
            "max_ell": 3,
            "hidden_irreps": "128x0e+128x1o+128x2e+128x3o",
            "conv_tp_plans": [
                {
                    "layer": layer_index,
                    **tensor_product_plan_json(
                        model.interactions[layer_index].conv_tp, np, torch, o3
                    ),
                }
                for layer_index in range(2)
            ],
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "structures": structures,
        "conventions": {
            "dataset_source": "deterministic pathological periodic regression structures",
            "edge_vector": "R_j + shift - R_i",
            "force_reference": "F = -dE/dR from upstream MACE/PyTorch autograd",
            "stress": "stress = dE/dD / abs(det(reference cell)) for symmetric displacement D; for partial PBC, MACE expands non-periodic cell rows to (max_abs_position + 1) * 5 * cutoff identity rows before stress normalization",
            "virials": "virials = -dE/dD for symmetric displacement D",
            "runtime_claim": "pathological periodic edge-case parity fixture for scoped MP0b stress/virial regression",
        },
    }


def mp0b_contraction_prefix(product_index: int, contraction_index: int) -> str:
    return (
        f"products.{product_index}.symmetric_contractions."
        f"contractions.{contraction_index}"
    )


def mp0b_u_tensor_names(product_index: int, contraction_index: int) -> list[str]:
    prefix = mp0b_contraction_prefix(product_index, contraction_index)
    return [f"{prefix}.U_matrix_{order}" for order in [1, 2, 3]]


def mp0b_weight_tensor_names(product_index: int, contraction_index: int) -> list[str]:
    prefix = mp0b_contraction_prefix(product_index, contraction_index)
    return [
        f"{prefix}.weights.0",
        f"{prefix}.weights.1",
        f"{prefix}.weights_max",
    ]


def mp0b_component_spec(
    product_index: int,
    contraction_index: int,
    kind: str,
    num_equivariance: int | None = None,
) -> dict[str, Any]:
    component: dict[str, Any] = {
        "kind": kind,
        "u_tensors": mp0b_u_tensor_names(product_index, contraction_index),
        "weight_tensors": mp0b_weight_tensor_names(product_index, contraction_index),
    }
    if num_equivariance is not None:
        component["num_equivariance"] = num_equivariance
    return component


def mp0b_runtime_tensor_names() -> list[str]:
    radial_and_pair = [
        "radial_embedding.distance_transform.q",
        "radial_embedding.distance_transform.p",
        "radial_embedding.distance_transform.a",
        "pair_repulsion_fn.c",
        "pair_repulsion_fn.p",
        "pair_repulsion_fn.a_exp",
        "pair_repulsion_fn.a_prefactor",
    ]
    product0_scalar = mp0b_u_tensor_names(0, 0) + mp0b_weight_tensor_names(0, 0)
    product0_vector = mp0b_u_tensor_names(0, 1) + mp0b_weight_tensor_names(0, 1)
    product1_scalar = mp0b_u_tensor_names(1, 0) + mp0b_weight_tensor_names(1, 0)
    product_linear = [
        "products.0.linear.weight",
        "products.0.linear.output_mask",
        "products.1.linear.weight",
        "products.1.linear.output_mask",
    ]
    residual_skip = [
        "interactions.0.conv_tp.output_mask",
        "interactions.0.conv_tp_weights.layer0.weight",
        "interactions.0.conv_tp_weights.layer1.weight",
        "interactions.0.conv_tp_weights.layer2.weight",
        "interactions.0.conv_tp_weights.layer3.weight",
        "interactions.0.density_fn.layer0.weight",
        "interactions.0.linear.weight",
        "interactions.0.linear.output_mask",
        "interactions.0.skip_tp.weight",
        "interactions.0.skip_tp.output_mask",
        "interactions.1.skip_tp.weight",
        "interactions.1.skip_tp.output_mask",
        "interactions.1.linear_up.weight",
        "interactions.1.linear_up.output_mask",
        "interactions.1.conv_tp.output_mask",
        "interactions.1.conv_tp_weights.layer0.weight",
        "interactions.1.conv_tp_weights.layer1.weight",
        "interactions.1.conv_tp_weights.layer2.weight",
        "interactions.1.conv_tp_weights.layer3.weight",
        "interactions.1.density_fn.layer0.weight",
        "interactions.1.linear.weight",
        "interactions.1.linear.output_mask",
    ]
    readout_and_energy = [
        "node_embedding.linear.weight",
        "node_embedding.linear.output_mask",
        "interactions.0.linear_up.weight",
        "interactions.0.linear_up.output_mask",
        "atomic_energies_fn.atomic_energies",
        "readouts.0.linear.weight",
        "readouts.0.linear.output_mask",
        "readouts.1.linear_1.weight",
        "readouts.1.linear_1.output_mask",
        "readouts.1.linear_2.weight",
        "readouts.1.linear_2.output_mask",
        "scale_shift.scale",
        "scale_shift.shift",
    ]
    return (
        radial_and_pair
        + product0_scalar
        + product0_vector
        + product1_scalar
        + product_linear
        + residual_skip
        + readout_and_energy
    )


def scalar_contraction_prefix(product_index: int, contraction_index: int = 0) -> str:
    return (
        f"products.{product_index}.symmetric_contractions."
        f"contractions.{contraction_index}"
    )


def scalar_u_tensor_names(product_index: int, contraction_index: int = 0) -> list[str]:
    prefix = scalar_contraction_prefix(product_index, contraction_index)
    return [f"{prefix}.U_matrix_{order}" for order in [1, 2, 3]]


def scalar_weight_tensor_names(product_index: int, contraction_index: int = 0) -> list[str]:
    prefix = scalar_contraction_prefix(product_index, contraction_index)
    return [
        f"{prefix}.weights.0",
        f"{prefix}.weights.1",
        f"{prefix}.weights_max",
    ]


def osaka26_runtime_tensor_names() -> list[str]:
    radial_and_pair = [
        "radial_embedding.distance_transform.q",
        "radial_embedding.distance_transform.p",
        "radial_embedding.distance_transform.a",
        "pair_repulsion_fn.c",
        "pair_repulsion_fn.p",
        "pair_repulsion_fn.a_exp",
        "pair_repulsion_fn.a_prefactor",
    ]
    products = []
    for product_index in range(2):
        products.extend(scalar_u_tensor_names(product_index))
        products.extend(scalar_weight_tensor_names(product_index))
        products.extend(
            [
                f"products.{product_index}.linear.weight",
                f"products.{product_index}.linear.output_mask",
            ]
        )
    interactions = []
    for layer_index in range(2):
        interactions.extend(
            [
                f"interactions.{layer_index}.linear_up.weight",
                f"interactions.{layer_index}.linear_up.output_mask",
                f"interactions.{layer_index}.conv_tp.output_mask",
                f"interactions.{layer_index}.conv_tp_weights.layer0.weight",
                f"interactions.{layer_index}.conv_tp_weights.layer1.weight",
                f"interactions.{layer_index}.conv_tp_weights.layer2.weight",
                f"interactions.{layer_index}.conv_tp_weights.layer3.weight",
                f"interactions.{layer_index}.linear.weight",
                f"interactions.{layer_index}.linear.output_mask",
                f"interactions.{layer_index}.skip_tp.weight",
                f"interactions.{layer_index}.skip_tp.output_mask",
            ]
        )
    readout_and_energy = [
        "node_embedding.linear.weight",
        "node_embedding.linear.output_mask",
        "atomic_energies_fn.atomic_energies",
        "readouts.0.linear.weight",
        "readouts.0.linear.output_mask",
        "readouts.1.linear_1.weight",
        "readouts.1.linear_1.output_mask",
        "readouts.1.linear_2.weight",
        "readouts.1.linear_2.output_mask",
        "scale_shift.scale",
        "scale_shift.shift",
    ]
    return radial_and_pair + interactions + products + readout_and_energy


def tensor_relpath(name: str) -> Path:
    return Path("weights") / f"{name.replace('.', '/')}.f64.bin"


def write_manifest_f64_tensor(path: Path, tensor: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = tensor.detach().cpu().double().contiguous().numpy().reshape(-1)
    values = values.astype("<f8", copy=False)
    with path.open("wb") as handle:
        values.tofile(handle)


def tensor_value_sha256(tensor: Any) -> str:
    values = tensor.detach().cpu().contiguous().numpy().reshape(-1)
    if values.dtype.kind not in {"f", "i", "u"}:
        raise TypeError(f"unsupported tensor dtype for inventory hash: {values.dtype}")
    values = values.astype(values.dtype.newbyteorder("<"), copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()


def mp0b_state_tensor_kind(name: str, tensor: Any, model: Any) -> tuple[str, bool]:
    parameter = dict(model.named_parameters()).get(name)
    buffer = dict(model.named_buffers()).get(name)
    if parameter is not None:
        return "parameter", bool(parameter.requires_grad)
    if buffer is not None:
        return "buffer", False
    return "state", False


def mace_archival_state_class(name: str, kind: str, requires_grad: bool, numel: int) -> str:
    if kind == "parameter" and requires_grad:
        return "runtime_trainable_parameter"
    if numel == 0 and name.endswith(".bias"):
        return "empty_bias_tensor"
    if numel == 0 and name.endswith(".weight"):
        return "empty_internal_weight_tensor"
    if name in {"atomic_numbers", "r_max", "num_interactions"}:
        return "model_metadata_constant"
    if name.startswith("radial_embedding.bessel_fn."):
        return "deterministic_bessel_metadata"
    if name == "radial_embedding.distance_transform.covalent_radii":
        return "distance_transform_covalent_radii_buffer"
    if name.startswith("radial_embedding.cutoff_fn."):
        return "radial_cutoff_metadata"
    if name == "pair_repulsion_fn.covalent_radii":
        return "pair_repulsion_covalent_radii_buffer"
    if name == "pair_repulsion_fn.r_max" or name.startswith("pair_repulsion_fn.cutoff."):
        return "pair_repulsion_cutoff_metadata"
    if "._compiled_main_left_right._w3j_" in name:
        return "compiled_tensor_product_wigner_buffer"
    return "runtime_buffer"


def mp0b_archival_state_class(name: str, kind: str, requires_grad: bool, numel: int) -> str:
    return mace_archival_state_class(name, kind, requires_grad, numel)


def write_mp0b_state_dict_inventory(path: Path, model: Any, runtime_names: set[str]) -> None:
    write_state_dict_inventory(
        path,
        model,
        runtime_names,
        "mace-mp0b-upstream-state-dict-inventory-v1",
        (
            "This inventory pins every upstream state_dict tensor by dtype, "
            "shape, and value hash. Only runtime_tensor=true entries are "
            "loaded through manifest [[tensors]] by the current scoped "
            "MP0b runtime."
        ),
    )


def write_state_dict_inventory(
    path: Path,
    model: Any,
    runtime_names: set[str],
    schema_version: str,
    scope_note: str,
) -> None:
    state_dict = model.state_dict()
    entries = []
    for name, tensor in state_dict.items():
        kind, requires_grad = mp0b_state_tensor_kind(name, tensor, model)
        numel = int(tensor.numel())
        entries.append(
            {
                "name": name,
                "kind": kind,
                "requires_grad": requires_grad,
                "dtype": str(tensor.dtype),
                "shape": [int(dim) for dim in tensor.shape],
                "numel": numel,
                "nonzero": int((tensor.detach().cpu() != 0).sum().item()),
                "value_sha256": tensor_value_sha256(tensor),
                "runtime_tensor": name in runtime_names,
                "classification": mace_archival_state_class(
                    name,
                    kind,
                    requires_grad,
                    numel,
                ),
            }
        )
    write_json(
        path,
        {
            "schema_version": schema_version,
            "state_tensor_count": len(entries),
            "runtime_tensor_count": sum(1 for entry in entries if entry["runtime_tensor"]),
            "archival_only_tensor_count": sum(
                1 for entry in entries if not entry["runtime_tensor"]
            ),
            "entries": entries,
            "scope_note": scope_note,
        },
    )


def mp0b_runtime_model_json(
    atomic_numbers: list[int],
    type_names: list[str],
    cutoff: float,
    num_bessel: int,
    heads: list[str],
    radial_cutoff_p: int,
    pair_repulsion_p: int,
    conv_tp_plans: list[dict[str, Any]],
    runtime_plan: dict[str, Any],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    irreps_in = "128x0e+128x1o+128x2e+128x3o"
    product0_contraction = {
        "irreps_in": irreps_in,
        "irreps_out": "128x0e+128x1o",
        "num_elements": len(atomic_numbers),
        "irrep_normalization": "component",
        "path_normalization": "element",
        "use_reduced_cg": False,
        "contractions": [
            mp0b_component_spec(0, 0, "scalar"),
            mp0b_component_spec(0, 1, "vector", num_equivariance=3),
        ],
    }
    product1_contraction = {
        "irreps_in": irreps_in,
        "irreps_out": "128x0e",
        "num_elements": len(atomic_numbers),
        "irrep_normalization": "component",
        "path_normalization": "element",
        "use_reduced_cg": False,
        "u_tensors": mp0b_u_tensor_names(1, 0),
        "weight_tensors": mp0b_weight_tensor_names(1, 0),
    }
    spec = {
        "schema_version": "mace-lite-model-v1",
        "name": "mace_mp0b_runtime",
        "type_names": type_names,
        "atomic_numbers": atomic_numbers,
        "cutoff": cutoff,
        "l_max": 3,
        "heads": heads,
        "radial": {
            "type": "bessel",
            "num_bessel": num_bessel,
            "polynomial_cutoff_p": radial_cutoff_p,
            "r_max": cutoff,
            "distance_transform": "Agnesi",
        },
        "pair_repulsion": {"type": "zbl", "polynomial_cutoff_p": pair_repulsion_p},
        "layers": [
            {
                "interaction": "RealAgnosticDensityInteractionBlock",
                "avg_num_neighbors_norm": 1.0,
                "product": {
                    "correlation": 3,
                    "symmetric_contraction": product0_contraction,
                },
            },
            {
                "interaction": "RealAgnosticDensityResidualInteractionBlock",
                "avg_num_neighbors_norm": 1.0,
                "product": {
                    "correlation": 3,
                    "symmetric_contraction": product1_contraction,
                },
            },
        ],
        "readouts": [{"kind": "linear"}, {"kind": "nonlinear"}],
        "mp0b_runtime_plan": runtime_plan,
        "conv_tp_plans": conv_tp_plans,
        "state_dict_format": "manifest-f64-tensors-v1",
        "state_dict": [],
        "unsupported_features": [],
    }
    if provenance is not None:
        spec["provenance"] = provenance
    return spec


def mp0b_runtime_plan_json(
    state_dict: dict[str, Any],
    atomic_numbers: list[int],
    num_bessel: int,
    l_max: int,
    conv_tp_plans: list[dict[str, Any]],
) -> dict[str, Any]:
    def shape(name: str) -> list[int]:
        return [int(dim) for dim in state_dict[name].shape]

    def numel(name: str) -> int:
        return int(state_dict[name].numel())

    num_features = numel("node_embedding.linear.output_mask")
    sh_dim = (l_max + 1) ** 2
    hidden_row_len = numel("interactions.0.linear.output_mask")
    product0_row_len = numel("products.0.linear.output_mask")
    product1_row_len = numel("products.1.linear.output_mask")
    layers = []
    for layer_index, plan in enumerate(conv_tp_plans):
        conv_tp_weight_shape = shape(f"interactions.{layer_index}.conv_tp_weights.layer3.weight")
        density_shape = shape(f"interactions.{layer_index}.density_fn.layer0.weight")
        layers.append(
            {
                "layer": layer_index,
                "interaction": (
                    "RealAgnosticDensityInteractionBlock"
                    if layer_index == 0
                    else "RealAgnosticDensityResidualInteractionBlock"
                ),
                "linear_up_input_dim": num_features if layer_index == 0 else product0_row_len,
                "linear_up_output_dim": numel(
                    f"interactions.{layer_index}.linear_up.output_mask"
                ),
                "conv_tp_weight_numel": int(plan["weight_numel"]),
                "conv_tp_output_dim": numel(f"interactions.{layer_index}.conv_tp.output_mask"),
                "conv_tp_weight_mlp_hidden_dim": conv_tp_weight_shape[0],
                "density_input_dim": density_shape[0],
                "density_output_dim": density_shape[1],
                "linear_message_output_dim": numel(
                    f"interactions.{layer_index}.linear.output_mask"
                ),
                "skip_output_dim": numel(f"interactions.{layer_index}.skip_tp.output_mask"),
                "skip_kind": "message_selector" if layer_index == 0 else "residual_add",
                "product_output_dim": product0_row_len if layer_index == 0 else product1_row_len,
                "product_kind": "scalar_vector" if layer_index == 0 else "scalar",
            }
        )
    return {
        "schema_version": "mace-mp0b-runtime-plan-v1",
        "num_elements": len(atomic_numbers),
        "num_features": num_features,
        "l_max": l_max,
        "sh_dim": sh_dim,
        "num_bessel": num_bessel,
        "hidden_row_len": hidden_row_len,
        "product0_row_len": product0_row_len,
        "product1_row_len": product1_row_len,
        "node_embedding": {
            "input_row_len": len(atomic_numbers),
            "in_features": len(atomic_numbers),
            "out_features": num_features,
            "path_weight": 1.0 / (len(atomic_numbers) ** 0.5),
        },
        "layers": layers,
        "readouts": [
            {"layer": 0, "kind": "linear", "input_dim": product0_row_len},
            {"layer": 1, "kind": "nonlinear", "input_dim": product1_row_len},
        ],
    }


def mp0b_runtime_manifest_toml(
    type_names: list[str],
    tensor_entries: list[dict[str, Any]],
) -> str:
    lines = [
        'format_version = "0.1"',
        'architecture = "mace-lite-v1"',
        'dtype = "f64"',
        'length_unit = "angstrom"',
        'energy_unit = "eV"',
        'force_unit = "eV/angstrom"',
        "",
        "[species]",
        f"symbols = {toml_string_array(type_names)}",
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
        'reference_package = "mace"',
        'reference_version = "0.3.16"',
        "fixture_seed = 20240513",
        "",
        "[mace_lite]",
        'model = "mace_lite.json"',
    ]
    for entry in tensor_entries:
        lines.extend(
            [
                "",
                "[[tensors]]",
                f"name = {toml_string(entry['name'])}",
                'dtype = "f64"',
                f"shape = {toml_int_array(entry['shape'])}",
                'layout = "row-major"',
                f"path = {toml_string(entry['path'])}",
                f"sha256 = {toml_string(entry['sha256'])}",
            ]
        )
    return "\n".join(lines) + "\n"


def runtime_manifest_toml(
    type_names: list[str],
    cutoff: float,
    l_max: int,
    tensor_entries: list[dict[str, Any]],
) -> str:
    lines = [
        'format_version = "0.1"',
        'architecture = "mace-lite-v1"',
        'dtype = "f64"',
        'length_unit = "angstrom"',
        'energy_unit = "eV"',
        'force_unit = "eV/angstrom"',
        "",
        "[species]",
        f"symbols = {toml_string_array(type_names)}",
        "",
        "[geometry]",
        f"cutoff = {cutoff}",
        'edge_vector = "R_j + shift - R_i"',
        "directed_edges = true",
        "",
        "[spherical_harmonics]",
        f"l_max = {l_max}",
        "normalize_input = true",
        'normalization = "component"',
        'reference_convention = "pinned-fixture"',
        "",
        "[validation]",
        'reference_package = "mace"',
        'reference_version = "0.3.16"',
        "fixture_seed = 20240513",
        "",
        "[mace_lite]",
        'model = "mace_lite.json"',
    ]
    for entry in tensor_entries:
        lines.extend(
            [
                "",
                "[[tensors]]",
                f"name = {toml_string(entry['name'])}",
                'dtype = "f64"',
                f"shape = {toml_int_array(entry['shape'])}",
                'layout = "row-major"',
                f"path = {toml_string(entry['path'])}",
                f"sha256 = {toml_string(entry['sha256'])}",
            ]
        )
    return "\n".join(lines) + "\n"


def scalar_symmetric_contraction_plan(
    product_index: int,
    atomic_numbers: list[int],
) -> dict[str, Any]:
    return {
        "irreps_in": "128x0e+128x1o+128x2e+128x3o",
        "irreps_out": "128x0e",
        "num_elements": len(atomic_numbers),
        "irrep_normalization": "component",
        "path_normalization": "element",
        "use_reduced_cg": False,
        "u_tensors": scalar_u_tensor_names(product_index),
        "weight_tensors": scalar_weight_tensor_names(product_index),
    }


def osaka26_runtime_plan_json(
    state_dict: dict[str, Any],
    atomic_numbers: list[int],
    num_bessel: int,
    l_max: int,
    conv_tp_plans: list[dict[str, Any]],
    avg_num_neighbors: list[float],
) -> dict[str, Any]:
    def numel(name: str) -> int:
        return int(state_dict[name].numel())

    num_features = numel("node_embedding.linear.output_mask")
    sh_dim = (l_max + 1) ** 2
    hidden_row_len = numel("interactions.0.linear.output_mask")
    product_row_len = numel("products.0.linear.output_mask")
    layers = []
    for layer_index, plan in enumerate(conv_tp_plans):
        layers.append(
            {
                "layer": layer_index,
                "interaction": "RealAgnosticResidualInteractionBlock",
                "linear_up_input_dim": num_features,
                "linear_up_output_dim": numel(
                    f"interactions.{layer_index}.linear_up.output_mask"
                ),
                "conv_tp_weight_numel": int(plan["weight_numel"]),
                "conv_tp_output_dim": numel(f"interactions.{layer_index}.conv_tp.output_mask"),
                "conv_tp_weight_mlp_hidden_dim": int(
                    state_dict[f"interactions.{layer_index}.conv_tp_weights.layer0.weight"].shape[1]
                ),
                "density_input_dim": 0,
                "density_output_dim": 0,
                "linear_message_output_dim": numel(
                    f"interactions.{layer_index}.linear.output_mask"
                ),
                "skip_output_dim": numel(f"interactions.{layer_index}.skip_tp.output_mask"),
                "skip_kind": "residual_add",
                "product_output_dim": product_row_len,
                "product_kind": "scalar",
            }
        )
    return {
        "schema_version": "mace-osaka26-runtime-plan-v1",
        "num_elements": len(atomic_numbers),
        "num_features": num_features,
        "l_max": l_max,
        "sh_dim": sh_dim,
        "num_bessel": num_bessel,
        "hidden_row_len": hidden_row_len,
        "product0_row_len": product_row_len,
        "product1_row_len": product_row_len,
        "node_embedding": {
            "input_row_len": len(atomic_numbers),
            "in_features": len(atomic_numbers),
            "out_features": num_features,
            "path_weight": 1.0 / (len(atomic_numbers) ** 0.5),
        },
        "layers": layers,
        "readouts": [
            {"layer": 0, "kind": "linear", "input_dim": product_row_len},
            {"layer": 1, "kind": "nonlinear", "input_dim": product_row_len},
        ],
    }


def osaka26_runtime_model_json(
    atomic_numbers: list[int],
    type_names: list[str],
    cutoff: float,
    num_bessel: int,
    heads: list[str],
    radial_cutoff_p: int,
    pair_repulsion_p: int,
    avg_num_neighbors: list[float],
    conv_tp_plans: list[dict[str, Any]],
    runtime_plan: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "mace-lite-model-v1",
        "name": "mace_osaka26_runtime",
        "type_names": type_names,
        "atomic_numbers": atomic_numbers,
        "cutoff": cutoff,
        "l_max": 3,
        "heads": heads,
        "radial": {
            "type": "bessel",
            "num_bessel": num_bessel,
            "polynomial_cutoff_p": radial_cutoff_p,
            "r_max": cutoff,
            "distance_transform": "Agnesi",
        },
        "pair_repulsion": {"type": "zbl", "polynomial_cutoff_p": pair_repulsion_p},
        "layers": [
            {
                "interaction": "RealAgnosticResidualInteractionBlock",
                "avg_num_neighbors_norm": avg_num_neighbors[layer_index],
                "product": {
                    "correlation": 3,
                    "symmetric_contraction": scalar_symmetric_contraction_plan(
                        layer_index, atomic_numbers
                    ),
                },
            }
            for layer_index in range(2)
        ],
        "readouts": [{"kind": "linear"}, {"kind": "nonlinear"}],
        "osaka26_runtime_plan": runtime_plan,
        "conv_tp_plans": conv_tp_plans,
        "provenance": provenance,
        "state_dict_format": "manifest-f64-tensors-v1",
        "state_dict": [],
        "unsupported_features": [],
    }


def validate_osaka26_runtime_model(model: Any, state_dict: dict[str, Any]) -> None:
    atomic_numbers = [int(value) for value in model.atomic_numbers.detach().cpu().tolist()]
    expected_atomic_numbers = list(range(1, 97)) + [98]
    checks = [
        ("atomic_numbers", atomic_numbers, expected_atomic_numbers),
        ("num_interactions", int(model.num_interactions.detach().cpu().item()), 2),
        ("r_max", float(model.r_max.detach().cpu().item()), 6.0),
        ("num_bessel", int(model.radial_embedding.bessel_fn.bessel_weights.numel()), 10),
        ("radial_cutoff_p", int(model.radial_embedding.cutoff_fn.p.detach().cpu().item()), 5),
        ("pair_repulsion_p", int(model.pair_repulsion_fn.p.detach().cpu().item()), 5),
    ]
    for name, actual, expected in checks:
        if actual != expected:
            raise SystemExit(
                f"MACE-Osaka26 artifact {name} mismatch:\n"
                f"expected: {expected}\n"
                f"actual:   {actual}"
            )
    if canonical_heads(list(model.heads)) is None:
        raise SystemExit(f"unsupported MACE-Osaka26 heads: {model.heads!r}")
    for layer_index, interaction in enumerate(model.interactions):
        if type(interaction).__name__ != "RealAgnosticResidualInteractionBlock":
            raise SystemExit(
                f"MACE-Osaka26 interaction {layer_index} is {type(interaction).__name__}"
            )
        if hasattr(interaction, "density_fn"):
            raise SystemExit("MACE-Osaka26 runtime exporter does not expect density_fn tensors")
        expected_shapes = {
            f"interactions.{layer_index}.linear_up.output_mask": [128],
            f"interactions.{layer_index}.conv_tp.output_mask": [2048],
            f"interactions.{layer_index}.linear.output_mask": [2048],
            f"interactions.{layer_index}.skip_tp.weight": [1589248],
            f"interactions.{layer_index}.skip_tp.output_mask": [128],
        }
        for tensor_name, expected_shape in expected_shapes.items():
            actual_shape = [int(dim) for dim in state_dict[tensor_name].shape]
            if actual_shape != expected_shape:
                raise SystemExit(
                    f"MACE-Osaka26 tensor {tensor_name} shape mismatch:\n"
                    f"expected: {expected_shape}\n"
                    f"actual:   {actual_shape}"
                )
        if int(interaction.conv_tp.weight_numel) != 512:
            raise SystemExit(
                f"MACE-Osaka26 conv_tp {layer_index} weight_numel mismatch: "
                f"{interaction.conv_tp.weight_numel}"
            )
    for product_index in range(2):
        for tensor_name, expected_shape in {
            f"products.{product_index}.linear.weight": [16384],
            f"products.{product_index}.linear.output_mask": [128],
        }.items():
            actual_shape = [int(dim) for dim in state_dict[tensor_name].shape]
            if actual_shape != expected_shape:
                raise SystemExit(
                    f"MACE-Osaka26 tensor {tensor_name} shape mismatch:\n"
                    f"expected: {expected_shape}\n"
                    f"actual:   {actual_shape}"
                )


def write_mp0b_runtime_model_fixture(
    out: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    (
        ase,
        _e3nn,
        mace,
        np,
        torch,
        _torch_functional,
        o3,
        _refs,
        model,
    ) = import_mace_mp0b_reference(args)
    manifest = read_json(args.manifest)
    expected_sha256 = manifest["artifact"]["sha256"]
    actual_sha256 = hash_file(args.artifact)
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            "MACE-MP0b artifact SHA256 mismatch:\n"
            f"expected: {expected_sha256}\n"
            f"actual:   {actual_sha256}"
        )
    expected_size_bytes = int(manifest["artifact"]["size_bytes"])
    actual_size_bytes = args.artifact.stat().st_size
    if actual_size_bytes != expected_size_bytes:
        raise SystemExit(
            "MACE-MP0b artifact size mismatch:\n"
            f"expected: {expected_size_bytes}\n"
            f"actual:   {actual_size_bytes}"
        )
    actual_commit = read_pinned_commit(args.mace_dir)
    if actual_commit != PINNED_MACE_COMMIT:
        raise SystemExit(
            "MACE reference commit mismatch:\n"
            f"expected: {PINNED_MACE_COMMIT}\n"
            f"actual:   {actual_commit}"
        )
    actual_package_version = getattr(mace, "__version__", "unknown")
    if actual_package_version != PINNED_MACE_PACKAGE_VERSION:
        raise SystemExit(
            "MACE package version mismatch:\n"
            f"expected: {PINNED_MACE_PACKAGE_VERSION}\n"
            f"actual:   {actual_package_version}"
        )

    atomic_numbers = [int(value) for value in model.atomic_numbers.detach().cpu().tolist()]
    type_names = [ase.data.chemical_symbols[number] for number in atomic_numbers]
    heads = canonical_heads(list(model.heads))
    if heads is None:
        raise SystemExit(f"unsupported MACE-MP0b heads: {model.heads!r}")
    cutoff = float(model.r_max.detach().cpu().item())
    num_bessel = int(model.radial_embedding.bessel_fn.bessel_weights.numel())
    radial_cutoff_p = int(model.radial_embedding.cutoff_fn.p.detach().cpu().item())
    pair_repulsion_p = int(model.pair_repulsion_fn.p.detach().cpu().item())
    conv_tp_plans = [
        {
            "layer": layer_index,
            **tensor_product_plan_json(model.interactions[layer_index].conv_tp, np, torch, o3),
        }
        for layer_index in range(2)
    ]
    state_dict = model.state_dict()
    runtime_plan = mp0b_runtime_plan_json(
        state_dict,
        atomic_numbers,
        num_bessel,
        3,
        conv_tp_plans,
    )

    runtime_tensor_names = mp0b_runtime_tensor_names()
    runtime_tensor_name_set = set(runtime_tensor_names)
    tensor_entries = []
    total_bytes = 0
    for name in runtime_tensor_names:
        if name not in state_dict:
            raise SystemExit(f"MACE-MP0b state_dict is missing {name!r}")
        tensor = state_dict[name]
        relpath = tensor_relpath(name)
        tensor_path = out / relpath
        write_manifest_f64_tensor(tensor_path, tensor)
        shape = [int(dim) for dim in tensor.shape]
        tensor_entries.append(
            {
                "name": name,
                "shape": shape,
                "path": relpath.as_posix(),
                "sha256": hash_file(tensor_path),
            }
        )
        total_bytes += max(1, int(tensor.numel())) * 8

    write_text(out / "manifest.toml", mp0b_runtime_manifest_toml(type_names, tensor_entries))
    write_mp0b_state_dict_inventory(
        out / "upstream_state_dict_inventory.json",
        model,
        runtime_tensor_name_set,
    )
    write_json(
        out / "mace_lite.json",
        mp0b_runtime_model_json(
            atomic_numbers,
            type_names,
            cutoff,
            num_bessel,
            heads,
            radial_cutoff_p,
            pair_repulsion_p,
            conv_tp_plans,
            runtime_plan,
            {
                "source_family": manifest["family"],
                "selected_model": manifest["selected_model"],
                "artifact": {
                    "filename": manifest["artifact"]["filename"],
                    "url": manifest_source_url(manifest),
                    "sha256": actual_sha256,
                    "size_bytes": actual_size_bytes,
                },
                "reference": {
                    "name": "mace",
                    "repository": REFERENCE_REPOSITORY,
                    "commit": actual_commit,
                    "package_version": actual_package_version,
                },
                "state_dict_format": "manifest-f64-tensors-v1",
                "upstream_state_tensor_count": len(state_dict),
                "runtime_state_tensor_count": len(tensor_entries),
                "archival_only_state_tensor_count": len(state_dict) - len(tensor_entries),
                "upstream_state_dict_inventory": "upstream_state_dict_inventory.json",
            },
        ),
    )
    return {
        "schema_version": MP0B_RUNTIME_EXPORT_SCHEMA_VERSION,
        "model": str(out),
        "reference": {
            "name": "mace",
            "repository": REFERENCE_REPOSITORY,
            "commit": read_pinned_commit(args.mace_dir),
            "package_version": getattr(mace, "__version__", "unknown"),
        },
        "artifact": {
            "manifest": str(args.manifest),
            "filename": args.artifact.name,
            "sha256": actual_sha256,
        },
        "l_max": 3,
        "num_layers": 2,
        "num_bessel": num_bessel,
        "state_tensors": len(tensor_entries),
        "state_tensor_bytes": total_bytes,
        "upstream_state_tensors": len(state_dict),
        "archival_only_state_tensors": len(state_dict) - len(tensor_entries),
        "purpose": (
            "scoped MP0b runtime with a full upstream state_dict "
            "inventory plus manifest f64 tensors for the documented "
            "runtime path: scalar/vector symmetric "
            "contraction, product-linear, message-selector/residual skip, "
            "explicit layer-0/layer-1 conv_tp plans and output masks, "
            "radial MLP/density, linear, readout, ZBL, and scale/shift tensors"
        ),
    }


def write_osaka26_runtime_model_fixture(
    out: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    (
        ase,
        _e3nn,
        mace,
        np,
        torch,
        _torch_functional,
        o3,
        _refs,
        model,
    ) = import_mace_artifact_reference(args, args.artifact)
    manifest = read_json(args.manifest)
    expected_sha256 = manifest["artifact"]["sha256"]
    actual_sha256 = hash_file(args.artifact)
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            "MACE-Osaka26 artifact SHA256 mismatch:\n"
            f"expected: {expected_sha256}\n"
            f"actual:   {actual_sha256}"
        )
    expected_size_bytes = int(manifest["artifact"]["size_bytes"])
    actual_size_bytes = args.artifact.stat().st_size
    if actual_size_bytes != expected_size_bytes:
        raise SystemExit(
            "MACE-Osaka26 artifact size mismatch:\n"
            f"expected: {expected_size_bytes}\n"
            f"actual:   {actual_size_bytes}"
        )
    actual_commit = read_pinned_commit(args.mace_dir)
    if actual_commit != PINNED_MACE_COMMIT:
        raise SystemExit(
            "MACE reference commit mismatch:\n"
            f"expected: {PINNED_MACE_COMMIT}\n"
            f"actual:   {actual_commit}"
        )
    actual_package_version = getattr(mace, "__version__", "unknown")
    if actual_package_version != PINNED_MACE_PACKAGE_VERSION:
        raise SystemExit(
            "MACE package version mismatch:\n"
            f"expected: {PINNED_MACE_PACKAGE_VERSION}\n"
            f"actual:   {actual_package_version}"
        )

    state_dict = model.state_dict()
    validate_osaka26_runtime_model(model, state_dict)
    atomic_numbers = [int(value) for value in model.atomic_numbers.detach().cpu().tolist()]
    type_names = [ase.data.chemical_symbols[number] for number in atomic_numbers]
    heads = canonical_heads(list(model.heads))
    if heads is None:
        raise SystemExit(f"unsupported MACE-Osaka26 heads: {model.heads!r}")
    cutoff = float(model.r_max.detach().cpu().item())
    num_bessel = int(model.radial_embedding.bessel_fn.bessel_weights.numel())
    radial_cutoff_p = int(model.radial_embedding.cutoff_fn.p.detach().cpu().item())
    pair_repulsion_p = int(model.pair_repulsion_fn.p.detach().cpu().item())
    avg_num_neighbors = [
        scalar_number(interaction.avg_num_neighbors) for interaction in model.interactions
    ]
    conv_tp_plans = [
        {
            "layer": layer_index,
            **tensor_product_plan_json(model.interactions[layer_index].conv_tp, np, torch, o3),
        }
        for layer_index in range(2)
    ]
    runtime_plan = osaka26_runtime_plan_json(
        state_dict,
        atomic_numbers,
        num_bessel,
        3,
        conv_tp_plans,
        avg_num_neighbors,
    )

    runtime_tensor_names = osaka26_runtime_tensor_names()
    runtime_tensor_name_set = set(runtime_tensor_names)
    tensor_entries = []
    total_bytes = 0
    for name in runtime_tensor_names:
        if name not in state_dict:
            raise SystemExit(f"MACE-Osaka26 state_dict is missing {name!r}")
        tensor = state_dict[name]
        relpath = tensor_relpath(name)
        tensor_path = out / relpath
        write_manifest_f64_tensor(tensor_path, tensor)
        shape = [int(dim) for dim in tensor.shape]
        tensor_entries.append(
            {
                "name": name,
                "shape": shape,
                "path": relpath.as_posix(),
                "sha256": hash_file(tensor_path),
            }
        )
        total_bytes += max(1, int(tensor.numel())) * 8

    write_text(out / "manifest.toml", runtime_manifest_toml(type_names, cutoff, 3, tensor_entries))
    write_state_dict_inventory(
        out / "upstream_state_dict_inventory.json",
        model,
        runtime_tensor_name_set,
        "mace-osaka26-upstream-state-dict-inventory-v1",
        (
            "This inventory pins every upstream state_dict tensor by dtype, "
            "shape, and value hash. Only runtime_tensor=true entries are "
            "loaded through manifest [[tensors]] by the current scoped "
            "MACE-Osaka26 runtime."
        ),
    )
    write_json(
        out / "mace_lite.json",
        osaka26_runtime_model_json(
            atomic_numbers,
            type_names,
            cutoff,
            num_bessel,
            heads,
            radial_cutoff_p,
            pair_repulsion_p,
            avg_num_neighbors,
            conv_tp_plans,
            runtime_plan,
            {
                "source_family": manifest["family"],
                "selected_model": manifest["selected_model"],
                "artifact": {
                    "filename": manifest["artifact"]["filename"],
                    "url": manifest_source_url(manifest),
                    "sha256": actual_sha256,
                    "size_bytes": actual_size_bytes,
                },
                "reference": {
                    "name": "mace",
                    "repository": REFERENCE_REPOSITORY,
                    "commit": actual_commit,
                    "package_version": actual_package_version,
                },
                "state_dict_format": "manifest-f64-tensors-v1",
                "upstream_state_tensor_count": len(state_dict),
                "runtime_state_tensor_count": len(tensor_entries),
                "archival_only_state_tensor_count": len(state_dict) - len(tensor_entries),
                "upstream_state_dict_inventory": "upstream_state_dict_inventory.json",
            },
        ),
    )
    return {
        "schema_version": OSAKA26_RUNTIME_EXPORT_SCHEMA_VERSION,
        "model": str(out),
        "reference": {
            "name": "mace",
            "repository": REFERENCE_REPOSITORY,
            "commit": read_pinned_commit(args.mace_dir),
            "package_version": getattr(mace, "__version__", "unknown"),
        },
        "artifact": {
            "manifest": str(args.manifest),
            "filename": args.artifact.name,
            "sha256": actual_sha256,
            "size_bytes": actual_size_bytes,
        },
        "l_max": 3,
        "num_layers": 2,
        "num_bessel": num_bessel,
        "state_tensors": len(tensor_entries),
        "state_tensor_bytes": total_bytes,
        "upstream_state_tensors": len(state_dict),
        "archival_only_state_tensors": len(state_dict) - len(tensor_entries),
        "purpose": (
            "scoped MACE-Osaka26 runtime export with manifest f64 tensors, "
            "scalar-only product contractions, residual interaction blocks, "
            "explicit layer-0/layer-1 conv_tp plans, residual skip tensor "
            "products, readouts, ZBL, and scale/shift tensors"
        ),
    }


def model_manifest_toml(l_max: int = 2) -> str:
    return f"""format_version = "0.1"
architecture = "mace-lite-v1"
dtype = "f64"
length_unit = "angstrom"
energy_unit = "eV"
force_unit = "eV/angstrom"

[species]
symbols = ["H", "C"]

[geometry]
cutoff = 4.0
edge_vector = "R_j + shift - R_i"
directed_edges = true

[spherical_harmonics]
l_max = {l_max}
normalize_input = true
normalization = "component"
reference_convention = "pinned-fixture"

[validation]
reference_package = "mace"
reference_version = "0.3.16"
fixture_seed = 20240513

[mace_lite]
model = "mace_lite.json"
"""


def model_layers_json(
    num_interactions: int, interactions: list[str] | None = None, l_max: int = 2
) -> list[dict[str, Any]]:
    if interactions is None:
        interactions = ["RealAgnosticResidualInteractionBlock"] * num_interactions
    if len(interactions) != num_interactions:
        raise ValueError("interaction list must match num_interactions")
    irreps_in = "+".join(
        f"1x{ell}{'e' if ell % 2 == 0 else 'o'}" for ell in range(l_max + 1)
    )
    return [
        {
            "interaction": interactions[index],
            "avg_num_neighbors_norm": 1.0,
            "product": {
                "correlation": 3,
                "symmetric_contraction": {
                    "irreps_in": irreps_in,
                    "irreps_out": "1x0e",
                    "num_elements": 2,
                    "irrep_normalization": "component",
                    "path_normalization": "element",
                    "use_reduced_cg": False,
                    "u_tensors": [
                        f"products.{index}.symmetric_contractions.contractions.0.U_matrix_1",
                        f"products.{index}.symmetric_contractions.contractions.0.U_matrix_2",
                        f"products.{index}.symmetric_contractions.contractions.0.U_matrix_3",
                    ],
                    "weight_tensors": [
                        f"products.{index}.symmetric_contractions.contractions.0.weights.0",
                        f"products.{index}.symmetric_contractions.contractions.0.weights.1",
                        f"products.{index}.symmetric_contractions.contractions.0.weights_max",
                    ],
                },
            },
        }
        for index in range(num_interactions)
    ]


def model_readouts_json(num_interactions: int) -> list[dict[str, str]]:
    return [
        {"kind": "linear"} if index == 0 else {"kind": "nonlinear"}
        for index in range(num_interactions)
    ]


def mace_lite_model_json(reference_fixture: dict[str, Any]) -> dict[str, Any]:
    num_interactions = int(reference_fixture["model"]["config"]["num_interactions"])
    config = reference_fixture["model"]["config"]
    l_max = int(config.get("max_ell", 2))
    default_interaction = "RealAgnosticResidualInteractionBlock"
    interaction_cls = config.get("interaction_cls", default_interaction)
    interaction_cls_first = config.get("interaction_cls_first", interaction_cls)
    interactions = [interaction_cls_first] + [interaction_cls] * (num_interactions - 1)
    return {
        "schema_version": "mace-lite-model-v1",
        "name": reference_fixture["model"]["name"],
        "type_names": ["H", "C"],
        "atomic_numbers": [1, 6],
        "cutoff": 4.0,
        "l_max": l_max,
        "heads": ["Default"],
        "radial": {
            "type": "bessel",
            "num_bessel": 1,
            "polynomial_cutoff_p": 6,
            "r_max": 4.0,
            "distance_transform": config.get("distance_transform", "None"),
        },
        "pair_repulsion": (
            {"type": "zbl", "polynomial_cutoff_p": 6}
            if config.get("pair_repulsion", False)
            else None
        ),
        "layers": model_layers_json(num_interactions, interactions, l_max=l_max),
        "readouts": model_readouts_json(num_interactions),
        "state_dict_format": "embedded-torch-state-dict-v1",
        "state_dict": reference_fixture["model"]["state_dict"],
        "unsupported_features": [],
    }


def write_model_fixture(
    out: Path,
    mace_dir: Path,
    num_interactions: int,
    fixture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if fixture is None:
        fixture = generate_reference_fixture(mace_dir, num_interactions)
    l_max = int(fixture["model"]["config"].get("max_ell", 2))
    write_text(out / "manifest.toml", model_manifest_toml(l_max))
    write_json(out / "mace_lite.json", mace_lite_model_json(fixture))
    return {
        "name": fixture["model"]["name"],
        "model": str(out),
        "state_dict_tensors": len(fixture["model"]["state_dict"]),
        "structures": [
            {
                "name": structure["name"],
                "num_atoms": len(structure["atomic_numbers"]),
                "energy": structure["outputs"]["energy"],
            }
            for structure in fixture["structures"]
        ],
    }


def write_one_layer_model_fixture(out: Path, mace_dir: Path) -> dict[str, Any]:
    return write_model_fixture(out, mace_dir, 1)


def write_two_layer_model_fixture(out: Path, mace_dir: Path) -> dict[str, Any]:
    return write_model_fixture(out, mace_dir, 2)


def write_density_residual_model_fixture(out: Path, mace_dir: Path) -> dict[str, Any]:
    return write_model_fixture(
        out,
        mace_dir,
        2,
        fixture=generate_density_residual_reference_fixture(mace_dir),
    )


def write_density_mixed_model_fixture(out: Path, mace_dir: Path) -> dict[str, Any]:
    return write_model_fixture(
        out,
        mace_dir,
        2,
        fixture=generate_density_mixed_reference_fixture(mace_dir),
    )


def write_agnesi_density_mixed_model_fixture(out: Path, mace_dir: Path) -> dict[str, Any]:
    return write_model_fixture(
        out,
        mace_dir,
        2,
        fixture=generate_agnesi_density_mixed_reference_fixture(mace_dir),
    )


def write_agnesi_zbl_density_mixed_model_fixture(out: Path, mace_dir: Path) -> dict[str, Any]:
    return write_model_fixture(
        out,
        mace_dir,
        2,
        fixture=generate_agnesi_zbl_density_mixed_reference_fixture(mace_dir),
    )


def write_lmax3_model_fixture(out: Path, mace_dir: Path) -> dict[str, Any]:
    return write_model_fixture(
        out,
        mace_dir,
        2,
        fixture=generate_lmax3_reference_fixture(mace_dir),
    )


def class_counts(inspection: dict[str, Any]) -> set[str]:
    return {entry["class"] for entry in inspection.get("module_class_counts", [])}


def top_level_classes(inspection: dict[str, Any]) -> dict[str, str]:
    return {
        entry["name"]: entry["class"]
        for entry in inspection.get("top_level_modules", [])
    }


def unsupported_features(inspection: dict[str, Any]) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    classes = class_counts(inspection)
    model = inspection["model"]

    if model["class"] != "mace.modules.models.ScaleShiftMACE":
        blockers.append(
            {
                "feature": "model_class",
                "observed": model["class"],
                "required_action": "add a fixture-pinned exporter path for this MACE model class",
            }
        )

    if model.get("parameter_dtypes") != ["torch.float64"]:
        blockers.append(
            {
                "feature": "dtype",
                "observed": ", ".join(model.get("parameter_dtypes", [])),
                "required_action": "support or reject non-f64 MACE artifacts explicitly",
            }
        )

    if model.get("heads") not in (["default"], ["Default"]):
        blockers.append(
            {
                "feature": "heads",
                "observed": str(model.get("heads")),
                "required_action": "support one default head only or add multihead fixtures",
            }
        )

    return blockers


def canonical_heads(raw_heads: Any) -> list[str] | None:
    if raw_heads == ["default"] or raw_heads == ["Default"]:
        return ["Default"]
    return None


def build_report(inspection_path: Path, inspection: dict[str, Any]) -> dict[str, Any]:
    blockers = unsupported_features(inspection)
    raw_heads = inspection["model"]["heads"]
    canonical = canonical_heads(raw_heads)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked" if blockers else "ready_for_scoped_runtime",
        "inspection": str(inspection_path),
        "artifact": inspection["artifact"],
        "reference_environment": inspection["environment"],
        "model": {
            "class": inspection["model"]["class"],
            "parameter_dtypes": inspection["model"]["parameter_dtypes"],
            "num_interactions": inspection["model"]["num_interactions"],
            "observed_max_l": inspection["model"]["observed_max_l"],
            "r_max": inspection["model"]["r_max"],
            "heads": inspection["model"]["heads"],
            "canonical_heads": canonical,
            "num_parameters": inspection["model"]["num_parameters"],
        },
        "head_policy": {
            "raw_upstream_heads": raw_heads,
            "canonical_e3rs_heads": canonical,
            "policy": (
                "single raw 'default' or 'Default' head is accepted and canonicalized "
                "to '.e3rs' head 'Default'; all multihead cases remain unsupported"
            ),
        },
        "supported_today": [
            "pinned artifact manifest",
            "SHA256-verified artifact inspection",
            "f64 ScaleShiftMACE model detection",
            "single default head detection with explicit canonicalization policy",
            "compact RealAgnosticDensityResidualInteractionBlock fixture/runtime path",
            "compact RealAgnosticDensityInteractionBlock first-layer fixture/runtime path",
            "compact AgnesiTransform radial distance-transform fixture/runtime path",
            "compact ZBLBasis pair-repulsion fixture/runtime path",
            "compact scalar-hidden l_max=3 symmetric-contraction fixture/runtime path",
            "MACE-MP0b vector-output symmetric-contraction primitive fixture/core VJP path",
            "core scalar+vector product-output concatenation and VJP accumulation path",
            "MACE-MP0b large-feature layer-flow forward reference fixture",
            "MACE-MP0b runtime energy/force parity on two non-periodic structures",
            "MACE-MP0b broad fixture energy/force parity on three non-periodic chemistry/near-cutoff representatives",
            "MACE-MP0b runtime energy/force/stress/virial parity on two PBC structures",
            "MACE-MP0b broad fixture energy/force/stress/virial parity on two additional full-PBC structures",
            "MACE-MP0b broad fixture energy/force/stress/virial parity for two partial-PBC multi-image diagnostics",
        ],
        "blocked_features": blockers,
        "scope_limitations": [
            "ready_for_scoped_runtime covers the committed MP0b runtime fixture set, not arbitrary MACE-MP0b compatibility",
            "broad PBC stress parity uses MACE's reference-cell normalization for partial-PBC diagnostics",
        ],
    }


def loader_smoke_manifest_toml() -> str:
    return """format_version = "0.1"
architecture = "mace-lite-v1"
dtype = "f64"
length_unit = "angstrom"
energy_unit = "eV"
force_unit = "eV/angstrom"

[species]
symbols = ["H", "C", "O"]

[geometry]
cutoff = 5.0
edge_vector = "R_j + shift - R_i"
directed_edges = true

[spherical_harmonics]
l_max = 2
normalize_input = true
normalization = "component"
reference_convention = "pinned-fixture"

[validation]
reference_package = "mace"
reference_version = "0.3.16"
fixture_seed = 97531

[mace_lite]
model = "mace_lite.json"
"""


def loader_smoke_model_json() -> dict[str, Any]:
    return {
        "schema_version": "mace-lite-model-v1",
        "name": "mace_lite_loader_smoke",
        "type_names": ["H", "C", "O"],
        "atomic_numbers": [1, 6, 8],
        "cutoff": 5.0,
        "l_max": 2,
        "heads": ["Default"],
        "radial": {
            "type": "bessel",
            "num_bessel": 8,
            "polynomial_cutoff_p": 6,
            "r_max": 5.0,
        },
        "layers": [
            {
                "interaction": "RealAgnosticInteractionBlock",
                "avg_num_neighbors_norm": 1.0,
                "product": {
                    "correlation": 3,
                    "symmetric_contraction": {
                        "irreps_in": "1x0e+1x1o+1x2e",
                        "irreps_out": "1x0e",
                        "num_elements": 3,
                        "irrep_normalization": "component",
                        "path_normalization": "element",
                        "use_reduced_cg": False,
                        "u_tensors": [
                            "layers.0.product.U_matrix_1",
                            "layers.0.product.U_matrix_2",
                            "layers.0.product.U_matrix_3",
                        ],
                        "weight_tensors": [
                            "layers.0.product.weights.0",
                            "layers.0.product.weights.1",
                            "layers.0.product.weights_max",
                        ],
                    },
                },
            }
        ],
        "readouts": [{"kind": "linear"}],
        "unsupported_features": [],
    }


def write_loader_smoke_fixture(out: Path, mace_dir: Path) -> dict[str, Any]:
    model_rel = "models/mace_lite_loader_smoke.e3rs"
    model_dir = out / model_rel
    reference = {
        "name": "mace",
        "repository": REFERENCE_REPOSITORY,
        "commit": read_pinned_commit(mace_dir),
        "package_version": read_mace_version(mace_dir),
    }
    manifest = {
        "schema_version": LOADER_SMOKE_SCHEMA_VERSION,
        "reference": reference,
        "dtype": "float64",
        "seed": LOADER_SMOKE_SEED,
        "models": [
            {
                "name": "mace_lite_loader_smoke",
                "model": model_rel,
                "l_max": 2,
                "num_layers": 1,
                "num_structures": 0,
                "purpose": "schema and loader validation only; no energy/force parity claim",
            }
        ],
    }
    write_text(model_dir / "manifest.toml", loader_smoke_manifest_toml())
    write_json(model_dir / "mace_lite.json", loader_smoke_model_json())
    write_json(out / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mace-dir", type=Path, default=Path("third_party/mace"))
    parser.add_argument(
        "--inspection",
        type=Path,
        default=Path("fixtures/mace/v0_3_16_4d2da09/mace_mp0b_inspection.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("fixtures/mace/v0_3_16_4d2da09/mace_mp0b_artifact_manifest.json"),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("outputs/reference-artifacts/mace/mace-mp-0b3-medium.model"),
    )
    parser.add_argument("--e3nn-compat-path", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("fixtures/mace/current/mace_lite_export_block_report.json"),
    )
    parser.add_argument(
        "--expect-blocked",
        action="store_true",
        help="return success only when the current artifact is explicitly blocked",
    )
    parser.add_argument(
        "--emit-loader-smoke",
        action="store_true",
        help="write a deterministic mace-lite-v1 loader smoke .e3rs fixture",
    )
    parser.add_argument(
        "--loader-smoke-out",
        type=Path,
        default=Path("fixtures/mace/current"),
    )
    parser.add_argument(
        "--emit-one-layer-reference",
        action="store_true",
        help="write a deterministic one-layer non-periodic MACE-lite reference fixture",
    )
    parser.add_argument(
        "--one-layer-reference-out",
        type=Path,
        default=Path("fixtures/mace/current/mace_lite_reference_f64.json"),
    )
    parser.add_argument(
        "--emit-one-layer-model",
        action="store_true",
        help="write a loadable deterministic one-layer MACE-lite .e3rs model fixture",
    )
    parser.add_argument(
        "--one-layer-model-out",
        type=Path,
        default=Path("fixtures/mace/current/models/mace_lite_layers1.e3rs"),
    )
    parser.add_argument(
        "--emit-two-layer-reference",
        action="store_true",
        help="write a deterministic two-layer non-periodic MACE-lite reference fixture",
    )
    parser.add_argument(
        "--two-layer-reference-out",
        type=Path,
        default=Path("fixtures/mace/current/mace_lite_layers2_reference_f64.json"),
    )
    parser.add_argument(
        "--emit-two-layer-model",
        action="store_true",
        help="write a loadable deterministic two-layer MACE-lite .e3rs model fixture",
    )
    parser.add_argument(
        "--two-layer-model-out",
        type=Path,
        default=Path("fixtures/mace/current/models/mace_lite_layers2.e3rs"),
    )
    parser.add_argument(
        "--emit-two-layer-pbc-reference",
        action="store_true",
        help="write a deterministic two-layer PBC MACE-lite reference fixture",
    )
    parser.add_argument(
        "--two-layer-pbc-reference-out",
        type=Path,
        default=Path("fixtures/mace/current/mace_lite_layers2_pbc_reference_f64.json"),
    )
    parser.add_argument(
        "--emit-density-residual-reference",
        action="store_true",
        help="write a deterministic two-layer density-residual MACE-lite reference fixture",
    )
    parser.add_argument(
        "--density-residual-reference-out",
        type=Path,
        default=Path(
            "fixtures/mace/current/mace_lite_density_residual_layers2_reference_f64.json"
        ),
    )
    parser.add_argument(
        "--emit-density-residual-model",
        action="store_true",
        help="write a loadable deterministic two-layer density-residual MACE-lite .e3rs model fixture",
    )
    parser.add_argument(
        "--density-residual-model-out",
        type=Path,
        default=Path("fixtures/mace/current/models/mace_lite_density_residual_layers2.e3rs"),
    )
    parser.add_argument(
        "--emit-density-residual-pbc-reference",
        action="store_true",
        help="write a deterministic two-layer density-residual PBC MACE-lite reference fixture",
    )
    parser.add_argument(
        "--density-residual-pbc-reference-out",
        type=Path,
        default=Path(
            "fixtures/mace/current/mace_lite_density_residual_layers2_pbc_reference_f64.json"
        ),
    )
    parser.add_argument(
        "--emit-density-mixed-reference",
        action="store_true",
        help="write a deterministic two-layer MACE-lite fixture with first non-residual density block",
    )
    parser.add_argument(
        "--density-mixed-reference-out",
        type=Path,
        default=Path("fixtures/mace/current/mace_lite_density_mixed_layers2_reference_f64.json"),
    )
    parser.add_argument(
        "--emit-density-mixed-model",
        action="store_true",
        help="write a loadable deterministic two-layer mixed-density MACE-lite .e3rs model fixture",
    )
    parser.add_argument(
        "--density-mixed-model-out",
        type=Path,
        default=Path("fixtures/mace/current/models/mace_lite_density_mixed_layers2.e3rs"),
    )
    parser.add_argument(
        "--emit-density-mixed-pbc-reference",
        action="store_true",
        help="write a deterministic two-layer mixed-density PBC MACE-lite reference fixture",
    )
    parser.add_argument(
        "--density-mixed-pbc-reference-out",
        type=Path,
        default=Path(
            "fixtures/mace/current/mace_lite_density_mixed_layers2_pbc_reference_f64.json"
        ),
    )
    parser.add_argument(
        "--emit-agnesi-density-mixed-reference",
        action="store_true",
        help="write a deterministic mixed-density MACE-lite fixture with Agnesi distance transform",
    )
    parser.add_argument(
        "--agnesi-density-mixed-reference-out",
        type=Path,
        default=Path(
            "fixtures/mace/current/mace_lite_agnesi_density_mixed_layers2_reference_f64.json"
        ),
    )
    parser.add_argument(
        "--emit-agnesi-density-mixed-model",
        action="store_true",
        help="write a loadable mixed-density Agnesi MACE-lite .e3rs model fixture",
    )
    parser.add_argument(
        "--agnesi-density-mixed-model-out",
        type=Path,
        default=Path(
            "fixtures/mace/current/models/mace_lite_agnesi_density_mixed_layers2.e3rs"
        ),
    )
    parser.add_argument(
        "--emit-agnesi-density-mixed-pbc-reference",
        action="store_true",
        help="write a deterministic mixed-density Agnesi PBC MACE-lite reference fixture",
    )
    parser.add_argument(
        "--agnesi-density-mixed-pbc-reference-out",
        type=Path,
        default=Path(
            "fixtures/mace/current/mace_lite_agnesi_density_mixed_layers2_pbc_reference_f64.json"
        ),
    )
    parser.add_argument(
        "--emit-agnesi-zbl-density-mixed-reference",
        action="store_true",
        help="write a deterministic mixed-density MACE-lite fixture with Agnesi and ZBL",
    )
    parser.add_argument(
        "--agnesi-zbl-density-mixed-reference-out",
        type=Path,
        default=Path(
            "fixtures/mace/current/mace_lite_agnesi_zbl_density_mixed_layers2_reference_f64.json"
        ),
    )
    parser.add_argument(
        "--emit-agnesi-zbl-density-mixed-model",
        action="store_true",
        help="write a loadable mixed-density Agnesi+ZBL MACE-lite .e3rs model fixture",
    )
    parser.add_argument(
        "--agnesi-zbl-density-mixed-model-out",
        type=Path,
        default=Path(
            "fixtures/mace/current/models/mace_lite_agnesi_zbl_density_mixed_layers2.e3rs"
        ),
    )
    parser.add_argument(
        "--emit-agnesi-zbl-density-mixed-pbc-reference",
        action="store_true",
        help="write a deterministic mixed-density Agnesi+ZBL PBC MACE-lite reference fixture",
    )
    parser.add_argument(
        "--agnesi-zbl-density-mixed-pbc-reference-out",
        type=Path,
        default=Path(
            "fixtures/mace/current/mace_lite_agnesi_zbl_density_mixed_layers2_pbc_reference_f64.json"
        ),
    )
    parser.add_argument(
        "--emit-lmax3-reference",
        action="store_true",
        help="write a deterministic two-layer scalar-hidden MACE-lite l_max=3 fixture",
    )
    parser.add_argument(
        "--lmax3-reference-out",
        type=Path,
        default=Path("fixtures/mace/current/mace_lite_lmax3_layers2_reference_f64.json"),
    )
    parser.add_argument(
        "--emit-lmax3-model",
        action="store_true",
        help="write a loadable deterministic two-layer scalar-hidden MACE-lite l_max=3 .e3rs model fixture",
    )
    parser.add_argument(
        "--lmax3-model-out",
        type=Path,
        default=Path("fixtures/mace/current/models/mace_lite_lmax3_layers2.e3rs"),
    )
    parser.add_argument(
        "--emit-lmax3-pbc-reference",
        action="store_true",
        help="write a deterministic two-layer scalar-hidden MACE-lite l_max=3 PBC reference fixture",
    )
    parser.add_argument(
        "--lmax3-pbc-reference-out",
        type=Path,
        default=Path("fixtures/mace/current/mace_lite_lmax3_layers2_pbc_reference_f64.json"),
    )
    parser.add_argument(
        "--emit-mp0b-flow-reference",
        action="store_true",
        help="write a deterministic representative MACE-MP0b layer-flow reference fixture",
    )
    parser.add_argument(
        "--mp0b-flow-reference-out",
        type=Path,
        default=Path("fixtures/mace/current/mace_mp0b_layer_flow_reference_f64.json"),
    )
    parser.add_argument(
        "--emit-mp0b-broad-reference",
        action="store_true",
        help="write broader MACE-MP0b chemistry and multi-image PBC diagnostics",
    )
    parser.add_argument(
        "--mp0b-broad-reference-out",
        type=Path,
        default=Path("fixtures/mace/current/mace_mp0b_broad_reference_f64.json"),
    )
    parser.add_argument(
        "--emit-mp0b-mptrj-mini-reference",
        action="store_true",
        help="write deterministic compact MP0b mini periodic parity fixture",
    )
    parser.add_argument(
        "--mp0b-mptrj-mini-reference-out",
        type=Path,
        default=Path("fixtures/mace/current/mace_mp0b_mptrj_mini_reference_f64.json"),
    )
    parser.add_argument(
        "--emit-periodic-pathological-reference",
        action="store_true",
        help="write deterministic MP0b mini periodic pathological edge-case fixture",
    )
    parser.add_argument(
        "--periodic-pathological-reference-out",
        type=Path,
        default=Path("fixtures/mace/current/mace_pbc_pathological_reference_f64.json"),
    )
    parser.add_argument(
        "--emit-osaka26-reference",
        action="store_true",
        help="write deterministic MACE-Osaka26 upstream parity reference data",
    )
    parser.add_argument(
        "--osaka26-reference-out",
        type=Path,
        default=Path("fixtures/mace/current/mace_osaka26_reference_f64.json"),
    )
    parser.add_argument(
        "--emit-osaka26-mptrj-mini-reference",
        action="store_true",
        help="write deterministic compact Osaka26 mini periodic parity fixture",
    )
    parser.add_argument(
        "--osaka26-mptrj-mini-reference-out",
        type=Path,
        default=Path(
            "fixtures/mace/current/mace_osaka26_mptrj_mini_reference_f64.json"
        ),
    )
    parser.add_argument(
        "--emit-mp0b-runtime-model",
        "--emit-mp0b-skeleton-model",
        dest="emit_mp0b_runtime_model",
        action="store_true",
        help="write a scoped MACE-MP0b .e3rs runtime export with manifest f64 tensors",
    )
    parser.add_argument(
        "--mp0b-runtime-model-out",
        "--mp0b-skeleton-model-out",
        dest="mp0b_runtime_model_out",
        type=Path,
        default=Path("fixtures/mace/current/models/mace_mp0b_runtime.e3rs"),
    )
    parser.add_argument(
        "--emit-osaka26-runtime-model",
        action="store_true",
        help="write a scoped MACE-Osaka26 .e3rs runtime export with manifest f64 tensors",
    )
    parser.add_argument(
        "--osaka26-runtime-model-out",
        type=Path,
        default=Path("fixtures/mace/current/models/mace_osaka26_runtime.e3rs"),
    )
    args = parser.parse_args()

    if args.emit_osaka26_runtime_model:
        summary = write_osaka26_runtime_model_fixture(args.osaka26_runtime_model_out, args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.emit_mp0b_runtime_model:
        summary = write_mp0b_runtime_model_fixture(args.mp0b_runtime_model_out, args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.emit_mp0b_broad_reference:
        fixture = generate_mp0b_broad_reference_fixture(args)
        write_json(args.mp0b_broad_reference_out, fixture)
        print(
            json.dumps(
                {
                    "schema_version": fixture["schema_version"],
                    "out": str(args.mp0b_broad_reference_out),
                    "reference": fixture["reference"],
                    "model": fixture["model"],
                    "structures": [
                        {
                            "name": structure["name"],
                            "num_atoms": len(structure["atomic_numbers"]),
                            "num_edges": len(structure["batch"]["edge_index"][0]),
                            "energy": structure["outputs"]["energy"],
                        }
                        for structure in fixture["structures"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.emit_mp0b_mptrj_mini_reference:
        fixture = generate_mp0b_mptrj_mini_reference_fixture(args)
        write_json(args.mp0b_mptrj_mini_reference_out, fixture)
        print(
            json.dumps(
                {
                    "schema_version": fixture["schema_version"],
                    "out": str(args.mp0b_mptrj_mini_reference_out),
                    "reference": fixture["reference"],
                    "model": fixture["model"],
                    "structures": [
                        {
                            "name": structure["name"],
                            "num_atoms": len(structure["atomic_numbers"]),
                            "num_edges": len(structure["batch"]["edge_index"][0]),
                            "energy": structure["outputs"]["energy"],
                        }
                        for structure in fixture["structures"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.emit_periodic_pathological_reference:
        fixture = generate_periodic_pathological_reference_fixture(args)
        write_json(args.periodic_pathological_reference_out, fixture)
        print(
            json.dumps(
                {
                    "schema_version": fixture["schema_version"],
                    "out": str(args.periodic_pathological_reference_out),
                    "reference": fixture["reference"],
                    "model": fixture["model"]["architecture"],
                    "structures": [
                        {
                            "name": structure["name"],
                            "num_atoms": len(structure["atomic_numbers"]),
                            "num_edges": len(structure["batch"]["edge_index"][0]),
                            "energy": structure["outputs"]["energy"],
                        }
                        for structure in fixture["structures"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.emit_osaka26_reference:
        fixture = generate_osaka26_reference_fixture(args)
        write_json(args.osaka26_reference_out, fixture)
        print(
            json.dumps(
                {
                    "schema_version": fixture["schema_version"],
                    "out": str(args.osaka26_reference_out),
                    "reference": fixture["reference"],
                    "model": fixture["model"],
                    "structures": [
                        {
                            "name": structure["name"],
                            "num_atoms": len(structure["atomic_numbers"]),
                            "num_edges": len(structure["batch"]["edge_index"][0]),
                            "energy": structure["outputs"]["energy"],
                        }
                        for structure in fixture["structures"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.emit_osaka26_mptrj_mini_reference:
        fixture = generate_osaka26_mptrj_mini_reference_fixture(args)
        write_json(args.osaka26_mptrj_mini_reference_out, fixture)
        print(
            json.dumps(
                {
                    "schema_version": fixture["schema_version"],
                    "out": str(args.osaka26_mptrj_mini_reference_out),
                    "reference": fixture["reference"],
                    "model": fixture["model"],
                    "structures": [
                        {
                            "name": structure["name"],
                            "num_atoms": len(structure["atomic_numbers"]),
                            "num_edges": len(structure["batch"]["edge_index"][0]),
                            "energy": structure["outputs"]["energy"],
                        }
                        for structure in fixture["structures"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.emit_mp0b_flow_reference:
        fixture = generate_mp0b_layer_flow_fixture(args)
        write_json(args.mp0b_flow_reference_out, fixture)
        print(
            json.dumps(
                {
                    "schema_version": fixture["schema_version"],
                    "out": str(args.mp0b_flow_reference_out),
                    "reference": fixture["reference"],
                    "model": fixture["model"],
                    "structures": [
                        {
                            "name": structure["name"],
                            "num_atoms": len(structure["atomic_numbers"]),
                            "energy": structure["outputs"]["energy"],
                        }
                        for structure in fixture["structures"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.emit_lmax3_pbc_reference:
        fixture = generate_lmax3_pbc_reference_fixture(args.mace_dir)
        write_json(args.lmax3_pbc_reference_out, fixture)
        print(
            json.dumps(
                {
                    "schema_version": fixture["schema_version"],
                    "out": str(args.lmax3_pbc_reference_out),
                    "reference": fixture["reference"],
                    "model": fixture["model"]["name"],
                    "structures": [
                        {
                            "name": structure["name"],
                            "num_atoms": len(structure["atomic_numbers"]),
                            "energy": structure["outputs"]["energy"],
                        }
                        for structure in fixture["structures"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.emit_lmax3_reference:
        fixture = generate_lmax3_reference_fixture(args.mace_dir)
        write_json(args.lmax3_reference_out, fixture)
        print(
            json.dumps(
                {
                    "schema_version": fixture["schema_version"],
                    "out": str(args.lmax3_reference_out),
                    "reference": fixture["reference"],
                    "model": fixture["model"]["name"],
                    "structures": [
                        {
                            "name": structure["name"],
                            "num_atoms": len(structure["atomic_numbers"]),
                            "energy": structure["outputs"]["energy"],
                        }
                        for structure in fixture["structures"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.emit_lmax3_model:
        summary = write_lmax3_model_fixture(args.lmax3_model_out, args.mace_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.emit_agnesi_zbl_density_mixed_pbc_reference:
        fixture = generate_agnesi_zbl_density_mixed_pbc_reference_fixture(args.mace_dir)
        write_json(args.agnesi_zbl_density_mixed_pbc_reference_out, fixture)
        print(
            json.dumps(
                {
                    "schema_version": fixture["schema_version"],
                    "out": str(args.agnesi_zbl_density_mixed_pbc_reference_out),
                    "reference": fixture["reference"],
                    "model": fixture["model"]["name"],
                    "structures": [
                        {
                            "name": structure["name"],
                            "num_atoms": len(structure["atomic_numbers"]),
                            "energy": structure["outputs"]["energy"],
                        }
                        for structure in fixture["structures"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.emit_agnesi_zbl_density_mixed_reference:
        fixture = generate_agnesi_zbl_density_mixed_reference_fixture(args.mace_dir)
        write_json(args.agnesi_zbl_density_mixed_reference_out, fixture)
        print(
            json.dumps(
                {
                    "schema_version": fixture["schema_version"],
                    "out": str(args.agnesi_zbl_density_mixed_reference_out),
                    "reference": fixture["reference"],
                    "model": fixture["model"]["name"],
                    "structures": [
                        {
                            "name": structure["name"],
                            "num_atoms": len(structure["atomic_numbers"]),
                            "energy": structure["outputs"]["energy"],
                        }
                        for structure in fixture["structures"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.emit_agnesi_zbl_density_mixed_model:
        summary = write_agnesi_zbl_density_mixed_model_fixture(
            args.agnesi_zbl_density_mixed_model_out, args.mace_dir
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.emit_agnesi_density_mixed_pbc_reference:
        fixture = generate_agnesi_density_mixed_pbc_reference_fixture(args.mace_dir)
        write_json(args.agnesi_density_mixed_pbc_reference_out, fixture)
        print(
            json.dumps(
                {
                    "schema_version": fixture["schema_version"],
                    "out": str(args.agnesi_density_mixed_pbc_reference_out),
                    "reference": fixture["reference"],
                    "model": fixture["model"]["name"],
                    "structures": [
                        {
                            "name": structure["name"],
                            "num_atoms": len(structure["atomic_numbers"]),
                            "energy": structure["outputs"]["energy"],
                        }
                        for structure in fixture["structures"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.emit_agnesi_density_mixed_reference:
        fixture = generate_agnesi_density_mixed_reference_fixture(args.mace_dir)
        write_json(args.agnesi_density_mixed_reference_out, fixture)
        print(
            json.dumps(
                {
                    "schema_version": fixture["schema_version"],
                    "out": str(args.agnesi_density_mixed_reference_out),
                    "reference": fixture["reference"],
                    "model": fixture["model"]["name"],
                    "structures": [
                        {
                            "name": structure["name"],
                            "num_atoms": len(structure["atomic_numbers"]),
                            "energy": structure["outputs"]["energy"],
                        }
                        for structure in fixture["structures"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.emit_agnesi_density_mixed_model:
        summary = write_agnesi_density_mixed_model_fixture(
            args.agnesi_density_mixed_model_out, args.mace_dir
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.emit_density_mixed_pbc_reference:
        fixture = generate_density_mixed_pbc_reference_fixture(args.mace_dir)
        write_json(args.density_mixed_pbc_reference_out, fixture)
        print(
            json.dumps(
                {
                    "schema_version": fixture["schema_version"],
                    "out": str(args.density_mixed_pbc_reference_out),
                    "reference": fixture["reference"],
                    "model": fixture["model"]["name"],
                    "structures": [
                        {
                            "name": structure["name"],
                            "num_atoms": len(structure["atomic_numbers"]),
                            "energy": structure["outputs"]["energy"],
                        }
                        for structure in fixture["structures"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.emit_density_mixed_reference:
        fixture = generate_density_mixed_reference_fixture(args.mace_dir)
        write_json(args.density_mixed_reference_out, fixture)
        print(
            json.dumps(
                {
                    "schema_version": fixture["schema_version"],
                    "out": str(args.density_mixed_reference_out),
                    "reference": fixture["reference"],
                    "model": fixture["model"]["name"],
                    "structures": [
                        {
                            "name": structure["name"],
                            "num_atoms": len(structure["atomic_numbers"]),
                            "energy": structure["outputs"]["energy"],
                        }
                        for structure in fixture["structures"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.emit_density_mixed_model:
        summary = write_density_mixed_model_fixture(args.density_mixed_model_out, args.mace_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.emit_density_residual_pbc_reference:
        fixture = generate_density_residual_pbc_reference_fixture(args.mace_dir)
        write_json(args.density_residual_pbc_reference_out, fixture)
        print(
            json.dumps(
                {
                    "schema_version": fixture["schema_version"],
                    "out": str(args.density_residual_pbc_reference_out),
                    "reference": fixture["reference"],
                    "model": fixture["model"]["name"],
                    "structures": [
                        {
                            "name": structure["name"],
                            "num_atoms": len(structure["atomic_numbers"]),
                            "energy": structure["outputs"]["energy"],
                        }
                        for structure in fixture["structures"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.emit_density_residual_reference:
        fixture = generate_density_residual_reference_fixture(args.mace_dir)
        write_json(args.density_residual_reference_out, fixture)
        print(
            json.dumps(
                {
                    "schema_version": fixture["schema_version"],
                    "out": str(args.density_residual_reference_out),
                    "reference": fixture["reference"],
                    "model": fixture["model"]["name"],
                    "structures": [
                        {
                            "name": structure["name"],
                            "num_atoms": len(structure["atomic_numbers"]),
                            "energy": structure["outputs"]["energy"],
                        }
                        for structure in fixture["structures"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.emit_density_residual_model:
        summary = write_density_residual_model_fixture(
            args.density_residual_model_out, args.mace_dir
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.emit_two_layer_pbc_reference:
        fixture = generate_pbc_reference_fixture(args.mace_dir, 2)
        write_json(args.two_layer_pbc_reference_out, fixture)
        print(
            json.dumps(
                {
                    "schema_version": fixture["schema_version"],
                    "out": str(args.two_layer_pbc_reference_out),
                    "reference": fixture["reference"],
                    "model": fixture["model"]["name"],
                    "structures": [
                        {
                            "name": structure["name"],
                            "num_atoms": len(structure["atomic_numbers"]),
                            "energy": structure["outputs"]["energy"],
                        }
                        for structure in fixture["structures"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.emit_two_layer_model:
        summary = write_two_layer_model_fixture(args.two_layer_model_out, args.mace_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.emit_one_layer_model:
        summary = write_one_layer_model_fixture(args.one_layer_model_out, args.mace_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.emit_two_layer_reference:
        fixture = generate_two_layer_reference_fixture(args.mace_dir)
        write_json(args.two_layer_reference_out, fixture)
        print(
            json.dumps(
                {
                    "schema_version": fixture["schema_version"],
                    "out": str(args.two_layer_reference_out),
                    "reference": fixture["reference"],
                    "model": fixture["model"]["name"],
                    "structures": [
                        {
                            "name": structure["name"],
                            "num_atoms": len(structure["atomic_numbers"]),
                            "energy": structure["outputs"]["energy"],
                        }
                        for structure in fixture["structures"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.emit_one_layer_reference:
        fixture = generate_one_layer_reference_fixture(args.mace_dir)
        write_json(args.one_layer_reference_out, fixture)
        print(
            json.dumps(
                {
                    "schema_version": fixture["schema_version"],
                    "out": str(args.one_layer_reference_out),
                    "reference": fixture["reference"],
                    "model": fixture["model"]["name"],
                    "structures": [
                        {
                            "name": structure["name"],
                            "num_atoms": len(structure["atomic_numbers"]),
                            "energy": structure["outputs"]["energy"],
                        }
                        for structure in fixture["structures"]
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.emit_loader_smoke:
        manifest = write_loader_smoke_fixture(args.loader_smoke_out, args.mace_dir)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    inspection = read_json(args.inspection)
    report = build_report(args.inspection, inspection)
    write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))

    blocked = report["status"] == "blocked"
    if args.expect_blocked:
        if not blocked:
            raise SystemExit("expected MACE export to be blocked, but it is ready")
        return
    if blocked:
        raise SystemExit(
            "MACE export is blocked by unsupported artifact features; "
            f"see {args.out}"
        )
    return


if __name__ == "__main__":
    main()
