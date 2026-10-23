use llmlip::{Cell, Structure};

pub fn ase_si_diamond_cubic_8() -> Structure {
    let a = 5.43;
    let half = a / 2.0;
    let quarter = a / 4.0;
    let three_quarters = 3.0 * a / 4.0;
    Structure::new(
        vec!["Si".to_owned(); 8],
        vec![
            [0.0, 0.0, 0.0],
            [quarter, quarter, quarter],
            [0.0, half, half],
            [quarter, three_quarters, three_quarters],
            [half, 0.0, half],
            [three_quarters, quarter, three_quarters],
            [half, half, 0.0],
            [three_quarters, three_quarters, quarter],
        ],
    )
    .with_cell(
        Cell::new([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]),
        [true, true, true],
    )
}

pub fn ase_si_diamond_cubic_8_perturbed() -> Structure {
    let mut structure = ase_si_diamond_cubic_8();
    let deltas = [
        [0.031, -0.017, 0.011],
        [-0.022, 0.019, -0.013],
        [0.014, 0.027, -0.018],
        [-0.018, -0.021, 0.025],
        [0.026, 0.008, -0.016],
        [-0.011, -0.024, 0.021],
        [0.017, -0.015, -0.019],
        [-0.028, 0.023, 0.009],
    ];
    for (position, delta) in structure.positions.iter_mut().zip(deltas) {
        for axis in 0..3 {
            position[axis] += delta[axis];
        }
    }
    structure
}

pub fn hea_bcc(cells: usize) -> Structure {
    let a = 2.8;
    let symbols = ["Fe", "Ni", "Co", "Cr", "Cu"];
    let atom_count = 2 * cells * cells * cells;
    let mut species = Vec::with_capacity(atom_count);
    let mut positions = Vec::with_capacity(atom_count);
    let mut index = 0usize;
    for ix in 0..cells {
        for iy in 0..cells {
            for iz in 0..cells {
                let base = [ix as f64 * a, iy as f64 * a, iz as f64 * a];
                for frac in [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]] {
                    species.push(symbols[index % symbols.len()].to_owned());
                    positions.push([
                        base[0] + frac[0] * a,
                        base[1] + frac[1] * a,
                        base[2] + frac[2] * a,
                    ]);
                    index += 1;
                }
            }
        }
    }
    let cell_len = cells as f64 * a;
    Structure::new(species, positions).with_cell(
        Cell::new([
            [cell_len, 0.0, 0.0],
            [0.0, cell_len, 0.0],
            [0.0, 0.0, cell_len],
        ]),
        [true, true, true],
    )
}

pub fn hea_bcc_perturbed(cells: usize) -> Structure {
    let mut structure = hea_bcc(cells);
    for (index, position) in structure.positions.iter_mut().enumerate() {
        let delta = [
            (((index * 37 + 11) % 17) as f64 - 8.0) * 0.003,
            (((index * 53 + 7) % 19) as f64 - 9.0) * 0.003,
            (((index * 71 + 5) % 23) as f64 - 11.0) * 0.003,
        ];
        for axis in 0..3 {
            position[axis] += delta[axis];
        }
    }
    structure
}
