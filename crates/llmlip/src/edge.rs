//! Edge representation and force accumulation for the project edge convention.

use crate::{
    force_accum::{scatter_edge_vector_vjp, ForceAccumError},
    geometry::{add, norm, sub, Vec3},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    pub src: usize,
    pub dst: usize,
    pub shift: [i32; 3],
    pub shift_cart: Vec3,
    pub vector: Vec3,
    pub distance: f64,
}

impl Edge {
    #[must_use]
    pub fn new(
        src: usize,
        dst: usize,
        pos_src: Vec3,
        pos_dst: Vec3,
        shift: [i32; 3],
        shift_cart: Vec3,
    ) -> Self {
        let vector = edge_vector(pos_src, pos_dst, shift_cart);
        let distance = norm(vector);
        Self {
            src,
            dst,
            shift,
            shift_cart,
            vector,
            distance,
        }
    }
}

/// Project-wide edge convention: `r_ij = R_j + shift_cart - R_i`.
#[must_use]
pub fn edge_vector(pos_src: Vec3, pos_dst: Vec3, shift_cart: Vec3) -> Vec3 {
    sub(add(pos_dst, shift_cart), pos_src)
}

/// Accumulate `bar_r = dE / d r_ij` into atomic forces.
///
/// With `r_ij = R_j + shift_ij - R_i` and `F = -dE/dR`, force scatter is:
///
/// `forces[src] += bar_r`
/// `forces[dst] -= bar_r`
pub fn accumulate_edge_force(
    forces: &mut [Vec3],
    edge: &Edge,
    bar_r: Vec3,
) -> Result<(), ForceAccumError> {
    scatter_edge_vector_vjp(edge.src, edge.dst, bar_r, forces)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::geometry::{dot, ZERO_VEC3};

    fn assert_vec_close(actual: Vec3, expected: Vec3) {
        for (axis, (actual_component, expected_component)) in
            actual.iter().zip(expected.iter()).enumerate()
        {
            assert!(
                (*actual_component - *expected_component).abs() < 1e-12,
                "axis {axis}: expected {}, got {}",
                expected_component,
                actual_component
            );
        }
    }

    #[test]
    fn non_periodic_edge_vector_uses_project_convention() {
        let edge = Edge::new(0, 1, [1.0, 2.0, 3.0], [4.0, 6.0, 8.0], [0, 0, 0], ZERO_VEC3);

        assert_vec_close(edge.vector, [3.0, 4.0, 5.0]);
        assert!((edge.distance - 50.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn pair_energy_force_scatter_has_project_sign() {
        let edge = Edge::new(
            0,
            1,
            [0.0, 0.0, 0.0],
            [2.0, -1.0, 0.5],
            [0, 0, 0],
            ZERO_VEC3,
        );
        let energy = 0.5 * dot(edge.vector, edge.vector);
        let mut forces = vec![ZERO_VEC3; 2];

        accumulate_edge_force(&mut forces, &edge, edge.vector).unwrap();

        assert!((energy - 2.625).abs() < 1e-12);
        assert_vec_close(forces[0], [2.0, -1.0, 0.5]);
        assert_vec_close(forces[1], [-2.0, 1.0, -0.5]);
    }

    #[test]
    fn force_scatter_conserves_net_force_for_pair() {
        let edge = Edge::new(
            0,
            1,
            [0.0, 0.0, 0.0],
            [2.0, -1.0, 0.5],
            [0, 0, 0],
            ZERO_VEC3,
        );
        let mut forces = vec![ZERO_VEC3; 2];

        accumulate_edge_force(&mut forces, &edge, edge.vector).unwrap();

        assert_vec_close(
            [
                forces[0][0] + forces[1][0],
                forces[0][1] + forces[1][1],
                forces[0][2] + forces[1][2],
            ],
            ZERO_VEC3,
        );
    }
}
