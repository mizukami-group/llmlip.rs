"""Validate an external MACE-MP-0b3 model against pinned flow references.

Tolerances match the original MP0b runtime acceptance tests, in eV and Å.
This check intentionally does not apply these tolerances to other model families.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ase.data import chemical_symbols
from llmlip import Model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    model = Model(args.model)
    with args.reference.open() as stream:
        reference = json.load(stream)
    rows = []
    failures = []
    for structure in reference['structures']:
        expected = structure['outputs']
        periodic = any(structure['pbc'])
        options = dict(
            species=[chemical_symbols[z] for z in structure['atomic_numbers']],
            positions=np.array(structure['positions']),
            cell=np.array(structure['cell']) if periodic else None,
            pbc=tuple(structure['pbc']),
        )
        pred = model.predict(**options, energy=True, forces=True,
                             stress=periodic, virial=periodic)
        row = {'structure': structure['name']}
        comparisons = [('energy', 'energy', 2e-5 if periodic else 1e-5),
                       ('forces', 'forces', 5e-5)]
        if periodic:
            comparisons += [('stress', 'stress', 5e-6), ('virial', 'virials', 5e-3)]
        for key, reference_key, tolerance in comparisons:
            actual = np.asarray(getattr(pred, key))
            wanted = np.asarray(expected[reference_key]).reshape(actual.shape)
            error = float(np.max(np.abs(actual - wanted)))
            row[key + '_max_abs'] = error
            if not np.isfinite(error) or error > tolerance:
                failures.append(f"{structure['name']}: {key}: {error} > {tolerance}")
        rows.append(row)
    print(json.dumps(rows, indent=2))
    if not rows:
        raise SystemExit('Reference contains no structures')
    if failures:
        raise SystemExit('\n'.join(failures))


if __name__ == '__main__':
    main()
