#!/usr/bin/env python3
"""Shared PolarMACE reference fixture structure catalog."""

from __future__ import annotations

from typing import Any


def molecule_set() -> list[dict[str, Any]]:
    return [
        {"name": "hydrogen", "symbols": "H2", "positions": [[0.0, 0.0, 0.0], [0.741, 0.0, 0.0]]},
        {"name": "nitrogen", "symbols": "N2", "positions": [[0.0, 0.0, 0.0], [1.098, 0.0, 0.0]]},
        {"name": "oxygen", "symbols": "O2", "positions": [[0.0, 0.0, 0.0], [1.208, 0.0, 0.0]]},
        {"name": "carbon_monoxide", "symbols": "CO", "positions": [[0.0, 0.0, 0.0], [1.128, 0.0, 0.0]]},
        {"name": "carbon_dioxide", "symbols": "CO2", "positions": [[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]]},
        {"name": "water", "symbols": "OH2", "positions": [[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.239, 0.927, 0.0]]},
        {"name": "ammonia", "symbols": "NH3", "positions": [[0.0, 0.0, 0.0], [0.94, 0.0, 0.0], [-0.313, 0.886, 0.0], [-0.313, -0.443, 0.767]]},
        {"name": "methane", "symbols": "CH4", "positions": [[0.0, 0.0, 0.0], [0.629, 0.629, 0.629], [-0.629, -0.629, 0.629], [-0.629, 0.629, -0.629], [0.629, -0.629, -0.629]]},
        {"name": "ethane", "symbols": "C2H6", "positions": [[-0.77, 0.0, 0.0], [0.77, 0.0, 0.0], [-1.17, 1.0, 0.0], [-1.17, -0.5, 0.866], [-1.17, -0.5, -0.866], [1.17, -1.0, 0.0], [1.17, 0.5, 0.866], [1.17, 0.5, -0.866]]},
        {"name": "ethylene", "symbols": "C2H4", "positions": [[-0.67, 0.0, 0.0], [0.67, 0.0, 0.0], [-1.23, 0.92, 0.0], [-1.23, -0.92, 0.0], [1.23, 0.92, 0.0], [1.23, -0.92, 0.0]]},
        {"name": "acetylene", "symbols": "C2H2", "positions": [[-0.60, 0.0, 0.0], [0.60, 0.0, 0.0], [-1.66, 0.0, 0.0], [1.66, 0.0, 0.0]]},
        {"name": "methanol", "symbols": "COH4", "positions": [[0.0, 0.0, 0.0], [1.43, 0.0, 0.0], [1.84, 0.89, 0.0], [-0.63, 0.9, 0.0], [-0.63, -0.45, 0.78], [-0.63, -0.45, -0.78]]},
        {"name": "ethanol", "symbols": "C2OH6", "positions": [[-0.75, 0.0, 0.0], [0.75, 0.0, 0.0], [1.43, 1.2, 0.0], [2.37, 1.05, 0.0], [-1.15, 1.0, 0.0], [-1.15, -0.5, 0.866], [-1.15, -0.5, -0.866], [1.1, -0.55, 0.9], [1.1, -0.55, -0.9]]},
        {"name": "formaldehyde", "symbols": "COH2", "positions": [[0.0, 0.0, 0.0], [1.21, 0.0, 0.0], [-0.58, 0.94, 0.0], [-0.58, -0.94, 0.0]]},
        {"name": "acetone", "symbols": "C3OH6", "positions": [[0.0, 0.0, 0.0], [1.22, 0.0, 0.0], [-1.48, 0.0, 0.0], [0.75, 1.28, 0.0], [-1.87, 1.0, 0.0], [-1.87, -0.5, 0.866], [-1.87, -0.5, -0.866], [1.15, 1.8, 0.9], [1.15, 1.8, -0.9], [1.82, 1.0, 0.0]]},
        {"name": "benzene", "symbols": "C6H6", "positions": [[1.397, 0.0, 0.0], [0.6985, 1.2098, 0.0], [-0.6985, 1.2098, 0.0], [-1.397, 0.0, 0.0], [-0.6985, -1.2098, 0.0], [0.6985, -1.2098, 0.0], [2.48, 0.0, 0.0], [1.24, 2.148, 0.0], [-1.24, 2.148, 0.0], [-2.48, 0.0, 0.0], [-1.24, -2.148, 0.0], [1.24, -2.148, 0.0]]},
        {"name": "pyridine", "symbols": "NC5H5", "positions": [[1.35, 0.0, 0.0], [0.675, 1.169, 0.0], [-0.675, 1.169, 0.0], [-1.35, 0.0, 0.0], [-0.675, -1.169, 0.0], [0.675, -1.169, 0.0], [1.17, 2.02, 0.0], [-1.17, 2.02, 0.0], [-2.42, 0.0, 0.0], [-1.17, -2.02, 0.0], [1.17, -2.02, 0.0]]},
        {"name": "hydrogen_fluoride", "symbols": "HF", "positions": [[0.0, 0.0, 0.0], [0.917, 0.0, 0.0]]},
        {"name": "hydrogen_chloride", "symbols": "HCl", "positions": [[0.0, 0.0, 0.0], [1.275, 0.0, 0.0]]},
        {"name": "sulfur_dioxide", "symbols": "SO2", "positions": [[0.0, 0.0, 0.0], [1.43, 0.0, 0.0], [-0.715, 1.238, 0.0]]},
        {"name": "hydrogen_sulfide", "symbols": "SH2", "positions": [[0.0, 0.0, 0.0], [1.34, 0.0, 0.0], [-0.44, 1.27, 0.0]]},
        {"name": "phosphine", "symbols": "PH3", "positions": [[0.0, 0.0, 0.0], [1.42, 0.0, 0.0], [-0.47, 1.34, 0.0], [-0.47, -0.67, 1.16]]},
        {"name": "silane", "symbols": "SiH4", "positions": [[0.0, 0.0, 0.0], [0.85, 0.85, 0.85], [-0.85, -0.85, 0.85], [-0.85, 0.85, -0.85], [0.85, -0.85, -0.85]]},
        {"name": "sodium_chloride", "symbols": "NaCl", "positions": [[0.0, 0.0, 0.0], [2.36, 0.0, 0.0]]},
    ]


def reference_20plus_molecule_set() -> list[dict[str, Any]]:
    """Return the deterministic 24-structure non-PBC PolarMACE parity set."""
    return molecule_set()


def named_structure_spec(name: str) -> dict[str, Any]:
    """Return a neutral singlet zero-field non-PBC structure by catalog name."""
    from ase.symbols import string2symbols

    for spec in molecule_set():
        if spec["name"] == name:
            return {
                "name": spec["name"],
                "symbols": string2symbols(spec["symbols"]),
                "positions": spec["positions"],
                "total_charge": 0.0,
                "total_spin": 1.0,
                "external_field": [0.0, 0.0, 0.0],
                "pbc": [False, False, False],
            }
    known = ", ".join(spec["name"] for spec in molecule_set())
    raise RuntimeError(f"PolarMACE fixture catalog is missing {name!r}; known: {known}")


def water_spec() -> dict[str, Any]:
    """Return the canonical water structure used by detailed SCF replay fixtures."""
    return named_structure_spec("water")
