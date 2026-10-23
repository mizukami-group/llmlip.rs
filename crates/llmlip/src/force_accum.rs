use e3nn_core::Vec3;
use thiserror::Error;

use crate::geometry::{cross, dot, Cell};

pub type Mat3 = [[f64; 3]; 3];

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ForceAccumError {
    #[error("force index {index} is out of bounds for force buffer length {len}")]
    IndexOutOfBounds { index: usize, len: usize },

    #[error("stress cell volume must be finite and positive")]
    InvalidCellVolume,
}

/// Scatter an edge-vector cotangent into atomic forces.
///
/// The project edge convention is `r_ij = R_dst + shift - R_src`.
/// Given `bar_r = dE / dr_ij`, conservative forces receive:
///
/// ```text
/// F_src +=  bar_r
/// F_dst += -bar_r
/// ```
pub fn scatter_edge_vector_vjp(
    src: usize,
    dst: usize,
    bar_r: Vec3,
    forces: &mut [Vec3],
) -> Result<(), ForceAccumError> {
    if src >= forces.len() {
        return Err(ForceAccumError::IndexOutOfBounds {
            index: src,
            len: forces.len(),
        });
    }
    if dst >= forces.len() {
        return Err(ForceAccumError::IndexOutOfBounds {
            index: dst,
            len: forces.len(),
        });
    }
    forces[src][0] += bar_r[0];
    forces[src][1] += bar_r[1];
    forces[src][2] += bar_r[2];
    forces[dst][0] -= bar_r[0];
    forces[dst][1] -= bar_r[1];
    forces[dst][2] -= bar_r[2];
    Ok(())
}

pub fn scatter_edge_vector_vjp_compensated(
    src: usize,
    dst: usize,
    bar_r: Vec3,
    forces: &mut [Vec3],
    compensation: &mut [Vec3],
) -> Result<(), ForceAccumError> {
    if src >= forces.len() {
        return Err(ForceAccumError::IndexOutOfBounds {
            index: src,
            len: forces.len(),
        });
    }
    if dst >= forces.len() {
        return Err(ForceAccumError::IndexOutOfBounds {
            index: dst,
            len: forces.len(),
        });
    }
    for (axis, bar) in bar_r.iter().enumerate() {
        force_kahan_add(&mut forces[src][axis], &mut compensation[src][axis], *bar);
        force_kahan_add(&mut forces[dst][axis], &mut compensation[dst][axis], -*bar);
    }
    Ok(())
}

#[inline(always)]
fn force_kahan_add(sum: &mut f64, comp: &mut f64, term: f64) {
    let y = term - *comp;
    let t = *sum + y;
    *comp = (t - *sum) - y;
    *sum = t;
}

/// Accumulate the displacement-gradient contribution for one edge.
///
/// Stress/virial reference paths in NequIP and MACE apply an infinitesimal
/// row-vector displacement `D` as `r' = r + r D`. For an edge cotangent
/// `bar_r = dE / dr`, the contribution to `dE / dD` is:
///
/// ```text
/// grad_D[row][col] += r[row] * bar_r[col]
/// ```
pub fn accumulate_edge_displacement_gradient(
    edge_vector: Vec3,
    bar_r: Vec3,
    displacement_gradient: &mut Mat3,
) {
    for row in 0..3 {
        for (col, bar_component) in bar_r.iter().enumerate() {
            displacement_gradient[row][col] += edge_vector[row] * bar_component;
        }
    }
}

#[must_use]
pub fn virial_from_displacement_gradient(displacement_gradient: Mat3) -> Mat3 {
    scale_mat3(displacement_gradient, -1.0)
}

pub fn stress_from_displacement_gradient(
    displacement_gradient: Mat3,
    cell: &Cell,
) -> Result<Mat3, ForceAccumError> {
    let volume = cell_volume(cell);
    if !volume.is_finite() || volume <= 0.0 {
        return Err(ForceAccumError::InvalidCellVolume);
    }
    Ok(scale_mat3(displacement_gradient, 1.0 / volume))
}

#[must_use]
pub fn cell_volume(cell: &Cell) -> f64 {
    let [a, b, c] = cell.vectors;
    dot(a, cross(b, c)).abs()
}

#[must_use]
fn scale_mat3(matrix: Mat3, scale: f64) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = matrix[row][col] * scale;
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::edge::edge_vector;
    use e3nn_core::geometry::dot;

    const FD_EPS: f64 = 1e-6;
    const TOL: f64 = 1e-9;

    fn assert_vec_close(actual: Vec3, expected: Vec3) {
        for (axis, (actual_component, expected_component)) in
            actual.iter().zip(expected.iter()).enumerate()
        {
            assert!(
                (*actual_component - *expected_component).abs() < TOL,
                "axis {axis}: expected {}, got {}",
                expected_component,
                actual_component
            );
        }
    }

    fn assert_mat_close(actual: Mat3, expected: Mat3, tol: f64) {
        for row in 0..3 {
            for col in 0..3 {
                assert!(
                    (actual[row][col] - expected[row][col]).abs() < tol,
                    "row {row} col {col}: expected {}, got {}",
                    expected[row][col],
                    actual[row][col]
                );
            }
        }
    }

    fn pair_energy(pos: &[Vec3], src: usize, dst: usize, shift_cart: Vec3) -> f64 {
        let r = edge_vector(pos[src], pos[dst], shift_cart);
        0.5 * dot(r, r)
    }

    fn transform_row_vector(vector: Vec3, displacement: Mat3) -> Vec3 {
        let mut out = vector;
        for row in 0..3 {
            for (col, out_col) in out.iter_mut().enumerate() {
                *out_col += vector[row] * displacement[row][col];
            }
        }
        out
    }

    fn transformed_pair_energy(
        pos_src: Vec3,
        pos_dst: Vec3,
        cell: Cell,
        unit_shift: [i32; 3],
        displacement: Mat3,
    ) -> f64 {
        let src = transform_row_vector(pos_src, displacement);
        let dst = transform_row_vector(pos_dst, displacement);
        let shifted_cell = Cell::new(
            cell.vectors
                .map(|vector| transform_row_vector(vector, displacement)),
        );
        let shift_cart = shifted_cell.cartesian_shift(unit_shift);
        let r = edge_vector(src, dst, shift_cart);
        0.5 * dot(r, r)
    }

    fn finite_difference_force(
        pos: &[Vec3],
        atom: usize,
        axis: usize,
        src: usize,
        dst: usize,
        shift_cart: Vec3,
    ) -> f64 {
        let mut plus = pos.to_vec();
        plus[atom][axis] += FD_EPS;
        let e_plus = pair_energy(&plus, src, dst, shift_cart);

        let mut minus = pos.to_vec();
        minus[atom][axis] -= FD_EPS;
        let e_minus = pair_energy(&minus, src, dst, shift_cart);

        -(e_plus - e_minus) / (2.0 * FD_EPS)
    }

    #[test]
    fn scatter_matches_pair_energy_finite_difference_signs() {
        let src = 0;
        let dst = 1;
        let shift_cart = [0.25, -0.5, 0.75];
        let positions = vec![[1.0, -2.0, 0.5], [2.5, 1.0, -1.5]];
        let bar_r = edge_vector(positions[src], positions[dst], shift_cart);

        let mut forces = vec![[0.0; 3]; positions.len()];
        scatter_edge_vector_vjp(src, dst, bar_r, &mut forces).unwrap();

        for atom in [src, dst] {
            for (axis, force_component) in forces[atom].iter().enumerate() {
                let expected =
                    finite_difference_force(&positions, atom, axis, src, dst, shift_cart);
                assert!(
                    (*force_component - expected).abs() < TOL,
                    "atom {atom} axis {axis}: scatter {}, finite-difference {expected}",
                    force_component,
                );
            }
        }
    }

    #[test]
    fn scatter_uses_src_plus_dst_minus_convention() {
        let bar_r = [1.25, -2.5, 3.75];
        let mut forces = vec![[10.0, 20.0, 30.0], [-4.0, -5.0, -6.0], [0.5, 0.25, -0.5]];

        scatter_edge_vector_vjp(2, 0, bar_r, &mut forces).unwrap();

        assert_vec_close(forces[2], [1.75, -2.25, 3.25]);
        assert_vec_close(forces[0], [8.75, 22.5, 26.25]);
        assert_vec_close(forces[1], [-4.0, -5.0, -6.0]);
    }

    #[test]
    fn pair_scatter_has_zero_net_force() {
        let bar_r = [-0.5, 4.0, 2.25];
        let mut forces = vec![[0.0; 3]; 2];

        scatter_edge_vector_vjp(0, 1, bar_r, &mut forces).unwrap();

        for (left, right) in forces[0].iter().zip(forces[1].iter()) {
            assert!((*left + *right).abs() < TOL);
        }
    }

    #[test]
    fn scatter_reports_out_of_bounds_indices() {
        let mut forces = vec![[0.0; 3]; 1];

        let err = scatter_edge_vector_vjp(0, 1, [1.0, 0.0, 0.0], &mut forces)
            .expect_err("dst index should be checked");

        assert_eq!(err, ForceAccumError::IndexOutOfBounds { index: 1, len: 1 });
    }

    #[test]
    fn displacement_gradient_matches_cell_and_position_finite_difference() {
        let pos_src = [0.2, 0.3, -0.4];
        let pos_dst = [1.1, -0.7, 0.5];
        let cell = Cell::from_vectors([2.0, 0.1, 0.0], [0.0, 2.5, 0.2], [0.1, 0.0, 3.0]);
        let unit_shift = [1, -1, 0];
        let edge_vector = edge_vector(pos_src, pos_dst, cell.cartesian_shift(unit_shift));
        let bar_r = edge_vector;
        let mut displacement_gradient = [[0.0; 3]; 3];

        accumulate_edge_displacement_gradient(edge_vector, bar_r, &mut displacement_gradient);

        let mut expected = [[0.0; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                let mut plus = [[0.0; 3]; 3];
                plus[row][col] = FD_EPS;
                let e_plus = transformed_pair_energy(pos_src, pos_dst, cell, unit_shift, plus);

                let mut minus = [[0.0; 3]; 3];
                minus[row][col] = -FD_EPS;
                let e_minus = transformed_pair_energy(pos_src, pos_dst, cell, unit_shift, minus);

                expected[row][col] = (e_plus - e_minus) / (2.0 * FD_EPS);
            }
        }

        assert_mat_close(displacement_gradient, expected, 1.0e-8);
    }

    #[test]
    fn stress_and_virial_follow_reference_sign_convention() {
        let cell = Cell::from_vectors([2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]);
        let displacement_gradient = [[2.4, -1.2, 0.6], [0.0, 3.6, -0.3], [1.8, 0.9, -4.8]];

        let stress = stress_from_displacement_gradient(displacement_gradient, &cell).unwrap();
        let virial = virial_from_displacement_gradient(displacement_gradient);

        assert_mat_close(
            stress,
            [
                [0.1, -0.05, 0.025],
                [0.0, 0.15, -0.0125],
                [0.075, 0.0375, -0.2],
            ],
            1.0e-12,
        );
        assert_mat_close(
            virial,
            [[-2.4, 1.2, -0.6], [-0.0, -3.6, 0.3], [-1.8, -0.9, 4.8]],
            1.0e-12,
        );
    }

    #[test]
    fn stress_rejects_singular_cell() {
        let cell = Cell::from_vectors([1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let err = stress_from_displacement_gradient([[0.0; 3]; 3], &cell)
            .expect_err("zero volume should fail");

        assert_eq!(err, ForceAccumError::InvalidCellVolume);
    }
}
