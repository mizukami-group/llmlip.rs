//! Geometry helpers for structure and neighbor-list construction.

use thiserror::Error;

pub type Vec3 = e3nn_core::Vec3;

pub const ZERO_VEC3: Vec3 = [0.0, 0.0, 0.0];

/// Periodic simulation cell.
///
/// The matrix is stored row-major as Cartesian lattice vectors `[a, b, c]`.
/// Fractional coordinates are converted as:
///
/// `cart = frac[0] * a + frac[1] * b + frac[2] * c`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell {
    pub vectors: [Vec3; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinimumImage {
    pub shift: [i32; 3],
    pub shift_cart: Vec3,
    pub vector: Vec3,
}

#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum MinimumImageError {
    #[error("cell is singular and cannot be used for minimum-image PBC")]
    SingularCell,

    #[error("minimum-image fractional coordinate axis {axis} must be finite, got {value}")]
    NonFiniteFractional { axis: usize, value: f64 },

    #[error("minimum-image shift axis {axis} is outside the i32 edge-shift range, got {value}")]
    ShiftOutOfRange { axis: usize, value: f64 },

    #[error(
        "minimum-image local search shift axis {axis} overflowed for base {base} and delta {delta}"
    )]
    ShiftOverflow { axis: usize, base: i32, delta: i32 },
}

impl Cell {
    #[must_use]
    pub fn new(vectors: [Vec3; 3]) -> Self {
        Self { vectors }
    }

    #[must_use]
    pub fn from_vectors(a: Vec3, b: Vec3, c: Vec3) -> Self {
        Self::new([a, b, c])
    }

    #[must_use]
    pub fn cartesian_shift(&self, shift: [i32; 3]) -> Vec3 {
        add(
            add(
                scale(self.vectors[0], f64::from(shift[0])),
                scale(self.vectors[1], f64::from(shift[1])),
            ),
            scale(self.vectors[2], f64::from(shift[2])),
        )
    }

    #[must_use]
    pub fn fractional_to_cartesian(&self, frac: Vec3) -> Vec3 {
        add(
            add(
                scale(self.vectors[0], frac[0]),
                scale(self.vectors[1], frac[1]),
            ),
            scale(self.vectors[2], frac[2]),
        )
    }

    #[must_use]
    pub fn cartesian_to_fractional(&self, cart: Vec3) -> Option<Vec3> {
        let [a, b, c] = self.vectors;
        let volume = dot(a, cross(b, c));
        if !volume.is_finite() || volume.abs() <= f64::EPSILON {
            return None;
        }

        let frac = [
            dot(cart, cross(b, c)) / volume,
            dot(cart, cross(c, a)) / volume,
            dot(cart, cross(a, b)) / volume,
        ];
        if frac.iter().all(|value| value.is_finite()) {
            Some(frac)
        } else {
            None
        }
    }

    pub fn minimum_image(
        &self,
        delta_cart: Vec3,
        pbc: [bool; 3],
    ) -> Result<MinimumImage, MinimumImageError> {
        let frac = self
            .cartesian_to_fractional(delta_cart)
            .ok_or(MinimumImageError::SingularCell)?;
        let mut base_shift = [0_i32; 3];

        for (axis, shift_component) in base_shift.iter_mut().enumerate() {
            if !frac[axis].is_finite() {
                return Err(MinimumImageError::NonFiniteFractional {
                    axis,
                    value: frac[axis],
                });
            }
            if pbc[axis] {
                let images = frac[axis].round();
                let shift = -images;
                if shift < f64::from(i32::MIN) + 1.0 || shift > f64::from(i32::MAX) - 1.0 {
                    return Err(MinimumImageError::ShiftOutOfRange { axis, value: shift });
                }
                *shift_component = shift as i32;
            }
        }

        let mut best_shift = base_shift;
        let mut best_shift_cart = self.cartesian_shift(best_shift);
        let mut best_vector = add(delta_cart, best_shift_cart);
        let mut best_norm2 = dot(best_vector, best_vector);

        for delta_a in -1..=1 {
            if !pbc[0] && delta_a != 0 {
                continue;
            }
            for delta_b in -1..=1 {
                if !pbc[1] && delta_b != 0 {
                    continue;
                }
                for delta_c in -1..=1 {
                    if !pbc[2] && delta_c != 0 {
                        continue;
                    }
                    let deltas = [delta_a, delta_b, delta_c];
                    let mut candidate_shift = [0_i32; 3];
                    for axis in 0..3 {
                        candidate_shift[axis] = base_shift[axis].checked_add(deltas[axis]).ok_or(
                            MinimumImageError::ShiftOverflow {
                                axis,
                                base: base_shift[axis],
                                delta: deltas[axis],
                            },
                        )?;
                    }
                    let candidate_shift_cart = self.cartesian_shift(candidate_shift);
                    let candidate_vector = add(delta_cart, candidate_shift_cart);
                    let candidate_norm2 = dot(candidate_vector, candidate_vector);
                    if candidate_norm2 < best_norm2 {
                        best_shift = candidate_shift;
                        best_shift_cart = candidate_shift_cart;
                        best_vector = candidate_vector;
                        best_norm2 = candidate_norm2;
                    }
                }
            }
        }

        Ok(MinimumImage {
            shift: best_shift,
            shift_cart: best_shift_cart,
            vector: best_vector,
        })
    }
}

#[must_use]
pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[must_use]
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[must_use]
pub fn scale(a: Vec3, s: f64) -> Vec3 {
    [s * a[0], s * a[1], s * a[2]]
}

#[must_use]
pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[must_use]
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[must_use]
pub fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

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
    fn cell_uses_row_major_lattice_vectors() {
        let cell = Cell::from_vectors([2.0, 0.0, 0.0], [0.5, 3.0, 0.0], [0.0, 0.25, 4.0]);

        assert_vec_close(
            cell.fractional_to_cartesian([0.5, 0.25, 0.75]),
            [1.125, 0.9375, 3.0],
        );
        assert_vec_close(
            cell.cartesian_to_fractional([1.125, 0.9375, 3.0])
                .expect("non-singular cell"),
            [0.5, 0.25, 0.75],
        );
    }

    #[test]
    fn minimum_image_searches_neighboring_shifts_for_skewed_cells() {
        let cell = Cell::from_vectors([1.0, 0.0, 0.0], [0.99, 0.1, 0.0], [0.0, 0.0, 10.0]);
        let delta = cell.fractional_to_cartesian([0.49, 0.49, 0.0]);

        let image = cell
            .minimum_image(delta, [true, true, false])
            .expect("non-singular cell");

        assert_eq!(image.shift, [0, -1, 0]);
        assert_vec_close(image.vector, [-0.0149, -0.051, 0.0]);
    }

    #[test]
    fn minimum_image_rejects_unrepresentable_shifts() {
        let cell = Cell::from_vectors([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);

        let err = cell
            .minimum_image([3.0e9, 0.0, 0.0], [true, false, false])
            .expect_err("shift does not fit edge shift format");

        assert!(matches!(
            err,
            MinimumImageError::ShiftOutOfRange { axis: 0, .. }
        ));
    }
}
