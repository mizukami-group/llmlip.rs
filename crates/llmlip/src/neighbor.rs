//! Correctness-first neighbor list construction.

use std::collections::HashMap;

use thiserror::Error;

use crate::{
    edge::Edge,
    geometry::{add, dot, norm, sub, MinimumImageError, ZERO_VEC3},
    structure::Structure,
    Vec3,
};

const MAX_MULTI_IMAGE_ATOMS: usize = 10_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeMode {
    /// Store both `i -> j` and `j -> i` when both are inside the cutoff.
    Directed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NeighborList {
    pub edges: Vec<Edge>,
    pub cutoff: f64,
    pub edge_mode: EdgeMode,
}

impl NeighborList {
    #[must_use]
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    pub fn filtered_by_cutoff(&self, cutoff: f64) -> Result<Self, NeighborListError> {
        if !cutoff.is_finite() || cutoff < 0.0 {
            return Err(NeighborListError::InvalidCutoff(cutoff));
        }
        if cutoff > self.cutoff {
            return Err(NeighborListError::SharedCutoffTooSmall {
                requested: cutoff,
                available: self.cutoff,
            });
        }
        Ok(Self {
            edges: self
                .edges
                .iter()
                .filter(|edge| edge.distance < cutoff)
                .copied()
                .collect(),
            cutoff,
            edge_mode: self.edge_mode,
        })
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum NeighborListError {
    #[error("structure has {species} species labels but {positions} positions")]
    SpeciesPositionMismatch { species: usize, positions: usize },

    #[error("periodic boundary conditions require a cell")]
    MissingCellForPbc,

    #[error("cell is singular and cannot be used for minimum-image PBC")]
    SingularCell,

    #[error("position for atom {atom} axis {axis} must be finite, got {value}")]
    InvalidPosition {
        atom: usize,
        axis: usize,
        value: f64,
    },

    #[error("cell vector {vector} axis {axis} must be finite, got {value}")]
    InvalidCellVector {
        vector: usize,
        axis: usize,
        value: f64,
    },

    #[error(transparent)]
    MinimumImage(#[from] MinimumImageError),

    #[error("cutoff must be finite and non-negative, got {0}")]
    InvalidCutoff(f64),

    #[error("PBC cutoff {cutoff} exceeds the conservative minimum-image limit {limit}")]
    CutoffExceedsMinimumImage { cutoff: f64, limit: f64 },

    #[error(
        "shared neighbor-list cutoff {available} is smaller than requested cutoff {requested}"
    )]
    SharedCutoffTooSmall { requested: f64, available: f64 },

    #[error("PBC cutoff {cutoff} would expand neighbor-list images to {image_atoms} atoms ({image_count} images x {atom_count} atoms), exceeding safety limit {limit}; reduce the cutoff or enlarge the thin periodic cell")]
    MultiImageExpansionTooLarge {
        cutoff: f64,
        image_count: usize,
        atom_count: usize,
        image_atoms: usize,
        limit: usize,
    },
}

/// Build a directed all-pairs neighbor list using `distance < cutoff`.
pub fn all_pairs_neighbor_list(
    structure: &Structure,
    cutoff: f64,
) -> Result<NeighborList, NeighborListError> {
    validate_neighbor_list_inputs(structure, cutoff, true)?;

    // Hoist structure-level lookups out of the O(N^2) pair loop (§6.3). These are
    // loop-invariant, so resolving them once is pure code motion: under PBC the
    // cell is already guaranteed present by `validate_neighbor_list_inputs`
    // above, so the `ok_or` cannot change behavior.
    let has_pbc = uses_pbc(structure);
    let pbc = structure.pbc;
    let cell = if has_pbc {
        Some(structure.cell.ok_or(NeighborListError::MissingCellForPbc)?)
    } else {
        None
    };

    let mut edges = Vec::new();

    for (src, _) in structure.positions.iter().enumerate() {
        for (dst, _) in structure.positions.iter().enumerate() {
            if src == dst {
                continue;
            }

            let edge = edge_for_pair(structure, src, dst, has_pbc, cell, pbc)?;
            if edge.distance < cutoff {
                edges.push(edge);
            }
        }
    }

    Ok(NeighborList {
        edges,
        cutoff,
        edge_mode: EdgeMode::Directed,
    })
}

/// Build a directed all-pairs neighbor list by enumerating every periodic image
/// whose Cartesian edge vector satisfies `distance < cutoff`.
///
/// The zero self edge is skipped, but nonzero self-image edges are kept. This
/// is needed for upstream-compatible MACE/NequIP style PBC graphs when the
/// cutoff exceeds the single-image limit.
pub fn all_pairs_multi_image_neighbor_list(
    structure: &Structure,
    cutoff: f64,
) -> Result<NeighborList, NeighborListError> {
    validate_neighbor_list_inputs(structure, cutoff, false)?;
    if !uses_pbc(structure) {
        return all_pairs_neighbor_list(structure, cutoff);
    }

    let cell = structure.cell.ok_or(NeighborListError::MissingCellForPbc)?;
    let max_shifts = multi_image_shift_bounds(structure, cutoff)?;
    validate_multi_image_expansion(structure, cutoff, max_shifts)?;
    let mut edges = Vec::new();

    for (src, pos_src) in structure.positions.iter().enumerate() {
        for shift_a in -max_shifts[0]..=max_shifts[0] {
            for shift_b in -max_shifts[1]..=max_shifts[1] {
                for shift_c in -max_shifts[2]..=max_shifts[2] {
                    let shift = [shift_a, shift_b, shift_c];
                    if shift
                        .iter()
                        .zip(structure.pbc)
                        .any(|(component, is_periodic)| *component != 0 && !is_periodic)
                    {
                        continue;
                    }
                    if shift == [0, 0, 0] {
                        for (dst, pos_dst) in structure.positions.iter().enumerate() {
                            if src == dst {
                                continue;
                            }
                            let edge = Edge::new(src, dst, *pos_src, *pos_dst, shift, ZERO_VEC3);
                            if edge.distance < cutoff {
                                edges.push(edge);
                            }
                        }
                    } else {
                        let shift_cart = cell.cartesian_shift(shift);
                        for (dst, pos_dst) in structure.positions.iter().enumerate() {
                            let edge = Edge::new(src, dst, *pos_src, *pos_dst, shift, shift_cart);
                            if edge.distance < cutoff {
                                edges.push(edge);
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(NeighborList {
        edges,
        cutoff,
        edge_mode: EdgeMode::Directed,
    })
}

/// Build a directed cell-list neighbor list that supports non-PBC and
/// multi-image PBC edges with the same edge convention as
/// [`all_pairs_multi_image_neighbor_list`].
pub fn cell_list_multi_image_neighbor_list(
    structure: &Structure,
    cutoff: f64,
) -> Result<NeighborList, NeighborListError> {
    validate_neighbor_list_inputs(structure, cutoff, false)?;
    if cutoff == 0.0 || structure.positions.is_empty() {
        return Ok(NeighborList {
            edges: Vec::new(),
            cutoff,
            edge_mode: EdgeMode::Directed,
        });
    }

    let has_pbc = uses_pbc(structure);
    let cell = if has_pbc {
        Some(structure.cell.ok_or(NeighborListError::MissingCellForPbc)?)
    } else {
        None
    };
    let max_shifts = if has_pbc {
        let bounds = multi_image_shift_bounds(structure, cutoff)?;
        validate_multi_image_expansion(structure, cutoff, bounds)?;
        bounds
    } else {
        [0, 0, 0]
    };

    let mut image_atoms = Vec::new();
    let mut min_corner = [f64::INFINITY; 3];
    for shift_a in -max_shifts[0]..=max_shifts[0] {
        for shift_b in -max_shifts[1]..=max_shifts[1] {
            for shift_c in -max_shifts[2]..=max_shifts[2] {
                let shift = [shift_a, shift_b, shift_c];
                if shift
                    .iter()
                    .zip(structure.pbc)
                    .any(|(component, is_periodic)| *component != 0 && !is_periodic)
                {
                    continue;
                }
                let shift_cart = cell.map_or(ZERO_VEC3, |periodic_cell| {
                    periodic_cell.cartesian_shift(shift)
                });
                for (dst, pos_dst) in structure.positions.iter().enumerate() {
                    let position = add(*pos_dst, shift_cart);
                    for axis in 0..3 {
                        min_corner[axis] = min_corner[axis].min(position[axis]);
                    }
                    image_atoms.push(CellImageAtom {
                        dst,
                        shift,
                        shift_cart,
                        position,
                    });
                }
            }
        }
    }

    let bin_size = cell_list_bin_size(cutoff);
    let bin_radius = (cutoff / bin_size).ceil() as i64;

    let mut bins: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for (image_index, image_atom) in image_atoms.iter().enumerate() {
        bins.entry(cell_bin(image_atom.position, min_corner, bin_size))
            .or_default()
            .push(image_index);
    }

    let cutoff_squared = cutoff * cutoff;
    let mut edges = Vec::new();
    for (src, pos_src) in structure.positions.iter().enumerate() {
        let src_bin = cell_bin(*pos_src, min_corner, bin_size);
        for offset_a in -bin_radius..=bin_radius {
            for offset_b in -bin_radius..=bin_radius {
                for offset_c in -bin_radius..=bin_radius {
                    let neighbor_bin = [
                        src_bin[0] + offset_a,
                        src_bin[1] + offset_b,
                        src_bin[2] + offset_c,
                    ];
                    let Some(indices) = bins.get(&neighbor_bin) else {
                        continue;
                    };
                    for image_index in indices {
                        let image_atom = &image_atoms[*image_index];
                        if src == image_atom.dst && image_atom.shift == [0, 0, 0] {
                            continue;
                        }
                        push_cell_list_edge_if_within_cutoff(
                            &mut edges,
                            cutoff_squared,
                            src,
                            *pos_src,
                            image_atom,
                        );
                    }
                }
            }
        }
    }

    edges.sort_unstable_by_key(|edge| {
        (
            edge.src,
            edge.shift[0],
            edge.shift[1],
            edge.shift[2],
            edge.dst,
        )
    });

    Ok(NeighborList {
        edges,
        cutoff,
        edge_mode: EdgeMode::Directed,
    })
}

#[derive(Clone, Copy, Debug)]
struct CellImageAtom {
    dst: usize,
    shift: [i32; 3],
    shift_cart: Vec3,
    position: Vec3,
}

fn cell_list_bin_size(cutoff: f64) -> f64 {
    cutoff * 0.5
}

fn cell_bin(position: Vec3, min_corner: Vec3, bin_size: f64) -> [i64; 3] {
    [
        ((position[0] - min_corner[0]) / bin_size).floor() as i64,
        ((position[1] - min_corner[1]) / bin_size).floor() as i64,
        ((position[2] - min_corner[2]) / bin_size).floor() as i64,
    ]
}

fn push_cell_list_edge_if_within_cutoff(
    edges: &mut Vec<Edge>,
    cutoff_squared: f64,
    src: usize,
    pos_src: Vec3,
    image_atom: &CellImageAtom,
) {
    let vector = sub(image_atom.position, pos_src);
    let distance_squared = dot(vector, vector);
    if distance_squared < cutoff_squared {
        edges.push(Edge {
            src,
            dst: image_atom.dst,
            shift: image_atom.shift,
            shift_cart: image_atom.shift_cart,
            vector,
            distance: distance_squared.sqrt(),
        });
    }
}

fn validate_neighbor_list_inputs(
    structure: &Structure,
    cutoff: f64,
    enforce_single_image_cutoff: bool,
) -> Result<(), NeighborListError> {
    let has_pbc = uses_pbc(structure);

    if structure.species.len() != structure.positions.len() {
        return Err(NeighborListError::SpeciesPositionMismatch {
            species: structure.species.len(),
            positions: structure.positions.len(),
        });
    }
    if !cutoff.is_finite() || cutoff < 0.0 {
        return Err(NeighborListError::InvalidCutoff(cutoff));
    }
    validate_positions(structure)?;
    if has_pbc && structure.cell.is_none() {
        return Err(NeighborListError::MissingCellForPbc);
    }
    if has_pbc {
        validate_cell_vectors(structure)?;
    }
    if has_pbc
        && structure
            .cell
            .and_then(|cell| cell.cartesian_to_fractional(ZERO_VEC3))
            .is_none()
    {
        return Err(NeighborListError::SingularCell);
    }
    if has_pbc && enforce_single_image_cutoff {
        validate_single_image_cutoff(structure, cutoff)?;
    }
    Ok(())
}

fn edge_for_pair(
    structure: &Structure,
    src: usize,
    dst: usize,
    has_pbc: bool,
    cell: Option<crate::geometry::Cell>,
    pbc: [bool; 3],
) -> Result<Edge, NeighborListError> {
    let pos_src = structure.positions[src];
    let pos_dst = structure.positions[dst];

    if has_pbc {
        let cell = cell.ok_or(NeighborListError::MissingCellForPbc)?;
        let delta = sub(pos_dst, pos_src);
        let image = cell.minimum_image(delta, pbc).map_err(|err| match err {
            MinimumImageError::SingularCell => NeighborListError::SingularCell,
            other => NeighborListError::MinimumImage(other),
        })?;
        Ok(Edge::new(
            src,
            dst,
            pos_src,
            pos_dst,
            image.shift,
            image.shift_cart,
        ))
    } else {
        Ok(Edge::new(src, dst, pos_src, pos_dst, [0, 0, 0], ZERO_VEC3))
    }
}

fn uses_pbc(structure: &Structure) -> bool {
    structure.pbc.iter().any(|is_periodic| *is_periodic)
}

fn validate_positions(structure: &Structure) -> Result<(), NeighborListError> {
    for (atom, position) in structure.positions.iter().enumerate() {
        for (axis, value) in position.iter().enumerate() {
            if !value.is_finite() {
                return Err(NeighborListError::InvalidPosition {
                    atom,
                    axis,
                    value: *value,
                });
            }
        }
    }
    Ok(())
}

fn validate_cell_vectors(structure: &Structure) -> Result<(), NeighborListError> {
    let cell = structure.cell.ok_or(NeighborListError::MissingCellForPbc)?;
    for (vector, cell_vector) in cell.vectors.iter().enumerate() {
        for (axis, value) in cell_vector.iter().enumerate() {
            if !value.is_finite() {
                return Err(NeighborListError::InvalidCellVector {
                    vector,
                    axis,
                    value: *value,
                });
            }
        }
    }
    Ok(())
}

fn validate_single_image_cutoff(
    structure: &Structure,
    cutoff: f64,
) -> Result<(), NeighborListError> {
    let limit = single_image_cutoff_limit(structure)?;
    if cutoff > limit {
        Err(NeighborListError::CutoffExceedsMinimumImage { cutoff, limit })
    } else {
        Ok(())
    }
}

pub(crate) fn single_image_cutoff_limit(structure: &Structure) -> Result<f64, NeighborListError> {
    let cell = structure.cell.ok_or(NeighborListError::MissingCellForPbc)?;
    Ok(0.5 * shortest_periodic_lower_bound(&cell.vectors, structure.pbc)?)
}

fn multi_image_shift_bounds(
    structure: &Structure,
    cutoff: f64,
) -> Result<[i32; 3], NeighborListError> {
    let cell = structure.cell.ok_or(NeighborListError::MissingCellForPbc)?;
    let reciprocal_norms = reciprocal_axis_norms(cell)?;
    let fractional_spans = fractional_position_spans(structure, cell)?;
    let mut bounds = [0_i32; 3];
    for (axis, is_periodic) in structure.pbc.iter().enumerate() {
        if *is_periodic {
            let bound = (fractional_spans[axis] + cutoff * reciprocal_norms[axis]).ceil();
            if !bound.is_finite() || bound > f64::from(i32::MAX) {
                return Err(NeighborListError::MinimumImage(
                    MinimumImageError::ShiftOutOfRange { axis, value: bound },
                ));
            }
            bounds[axis] = bound.max(0.0) as i32;
        }
    }
    Ok(bounds)
}

fn reciprocal_axis_norms(cell: crate::geometry::Cell) -> Result<Vec3, NeighborListError> {
    let frac_x = cell
        .cartesian_to_fractional([1.0, 0.0, 0.0])
        .ok_or(NeighborListError::SingularCell)?;
    let frac_y = cell
        .cartesian_to_fractional([0.0, 1.0, 0.0])
        .ok_or(NeighborListError::SingularCell)?;
    let frac_z = cell
        .cartesian_to_fractional([0.0, 0.0, 1.0])
        .ok_or(NeighborListError::SingularCell)?;
    Ok([
        norm([frac_x[0], frac_y[0], frac_z[0]]),
        norm([frac_x[1], frac_y[1], frac_z[1]]),
        norm([frac_x[2], frac_y[2], frac_z[2]]),
    ])
}

fn fractional_position_spans(
    structure: &Structure,
    cell: crate::geometry::Cell,
) -> Result<Vec3, NeighborListError> {
    let mut min_frac = [f64::INFINITY; 3];
    let mut max_frac = [f64::NEG_INFINITY; 3];
    let mut has_position = false;
    for position in &structure.positions {
        let frac = cell
            .cartesian_to_fractional(*position)
            .ok_or(NeighborListError::SingularCell)?;
        has_position = true;
        for axis in 0..3 {
            min_frac[axis] = min_frac[axis].min(frac[axis]);
            max_frac[axis] = max_frac[axis].max(frac[axis]);
        }
    }
    if !has_position {
        return Ok(ZERO_VEC3);
    }
    Ok([
        max_frac[0] - min_frac[0],
        max_frac[1] - min_frac[1],
        max_frac[2] - min_frac[2],
    ])
}

fn validate_multi_image_expansion(
    structure: &Structure,
    cutoff: f64,
    max_shifts: [i32; 3],
) -> Result<(), NeighborListError> {
    let mut image_count = 1usize;
    for (axis, is_periodic) in structure.pbc.iter().enumerate() {
        let axis_images = if *is_periodic {
            usize::try_from(i64::from(max_shifts[axis]) * 2 + 1).map_err(|_| {
                NeighborListError::MinimumImage(MinimumImageError::ShiftOutOfRange {
                    axis,
                    value: f64::from(max_shifts[axis]),
                })
            })?
        } else {
            1
        };
        image_count = image_count.checked_mul(axis_images).ok_or(
            NeighborListError::MultiImageExpansionTooLarge {
                cutoff,
                image_count: usize::MAX,
                atom_count: structure.positions.len(),
                image_atoms: usize::MAX,
                limit: MAX_MULTI_IMAGE_ATOMS,
            },
        )?;
    }
    let image_atoms = image_count.checked_mul(structure.positions.len()).ok_or(
        NeighborListError::MultiImageExpansionTooLarge {
            cutoff,
            image_count,
            atom_count: structure.positions.len(),
            image_atoms: usize::MAX,
            limit: MAX_MULTI_IMAGE_ATOMS,
        },
    )?;
    if image_atoms > MAX_MULTI_IMAGE_ATOMS {
        return Err(NeighborListError::MultiImageExpansionTooLarge {
            cutoff,
            image_count,
            atom_count: structure.positions.len(),
            image_atoms,
            limit: MAX_MULTI_IMAGE_ATOMS,
        });
    }
    Ok(())
}

fn shortest_periodic_lower_bound(
    vectors: &[crate::Vec3; 3],
    pbc: [bool; 3],
) -> Result<f64, NeighborListError> {
    let mut periodic = Vec::new();
    for (axis, is_periodic) in pbc.iter().enumerate() {
        if *is_periodic {
            periodic.push(vectors[axis]);
        }
    }

    let mut local_shortest = f64::INFINITY;
    for da in -1..=1 {
        for db in -1..=1 {
            for dc in -1..=1 {
                let coeffs = [da, db, dc];
                if coeffs == [0, 0, 0] {
                    continue;
                }
                if coeffs
                    .iter()
                    .zip(pbc)
                    .any(|(coeff, is_periodic)| *coeff != 0 && !is_periodic)
                {
                    continue;
                }
                let shift = [
                    f64::from(da) * vectors[0][0]
                        + f64::from(db) * vectors[1][0]
                        + f64::from(dc) * vectors[2][0],
                    f64::from(da) * vectors[0][1]
                        + f64::from(db) * vectors[1][1]
                        + f64::from(dc) * vectors[2][1],
                    f64::from(da) * vectors[0][2]
                        + f64::from(db) * vectors[1][2]
                        + f64::from(dc) * vectors[2][2],
                ];
                let length = norm(shift);
                if length > 0.0 && length < local_shortest {
                    local_shortest = length;
                }
            }
        }
    }

    let singular_lower_bound = gram_smallest_eigenvalue_lower_bound(&periodic)
        .ok_or(NeighborListError::SingularCell)?
        .sqrt();
    let lower_bound = local_shortest.min(singular_lower_bound);
    if lower_bound.is_finite() && lower_bound > 0.0 {
        Ok(lower_bound)
    } else {
        Err(NeighborListError::SingularCell)
    }
}

fn gram_smallest_eigenvalue_lower_bound(vectors: &[crate::Vec3]) -> Option<f64> {
    match vectors.len() {
        0 => Some(f64::INFINITY),
        1 => Some(dot(vectors[0], vectors[0])),
        2 => {
            let a = dot(vectors[0], vectors[0]);
            let b = dot(vectors[0], vectors[1]);
            let d = dot(vectors[1], vectors[1]);
            let trace = a + d;
            let discriminant = ((a - d) * (a - d) + 4.0 * b * b).sqrt();
            Some(0.5 * (trace - discriminant))
        }
        3 => {
            let mut matrix = [[0.0; 3]; 3];
            for row in 0..3 {
                for col in 0..3 {
                    matrix[row][col] = dot(vectors[row], vectors[col]);
                }
            }
            Some(jacobi_smallest_eigenvalue_lower_bound(matrix))
        }
        _ => None,
    }
    .filter(|value| value.is_finite() && *value > 0.0)
}

fn jacobi_smallest_eigenvalue_lower_bound(mut matrix: [[f64; 3]; 3]) -> f64 {
    for _ in 0..24 {
        let mut pivot = (0, 1);
        let mut pivot_abs = matrix[0][1].abs();
        for (row, col) in [(0, 2), (1, 2)] {
            let value = matrix[row][col].abs();
            if value > pivot_abs {
                pivot = (row, col);
                pivot_abs = value;
            }
        }
        if pivot_abs <= 1.0e-14 {
            break;
        }

        let (p, q) = pivot;
        let tau = (matrix[q][q] - matrix[p][p]) / (2.0 * matrix[p][q]);
        let tau_sign = if tau >= 0.0 { 1.0 } else { -1.0 };
        let t = tau_sign / (tau.abs() + (1.0 + tau * tau).sqrt());
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let app = matrix[p][p];
        let aqq = matrix[q][q];
        let apq = matrix[p][q];
        matrix[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        matrix[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;

        for r in [0_usize, 1, 2] {
            if r == p || r == q {
                continue;
            }
            let arp = matrix[r][p];
            let arq = matrix[r][q];
            matrix[r][p] = c * arp - s * arq;
            matrix[p][r] = matrix[r][p];
            matrix[r][q] = s * arp + c * arq;
            matrix[q][r] = matrix[r][q];
        }
    }

    // If tiny off-diagonal residuals remain, Gershgorin gives a conservative
    // lower bound and prevents near-singular cells from getting an optimistic
    // single-image cutoff.
    let diagonal_min = matrix[0][0].min(matrix[1][1]).min(matrix[2][2]);
    diagonal_min.min(symmetric_gershgorin_lower_bound(matrix))
}

fn symmetric_gershgorin_lower_bound(matrix: [[f64; 3]; 3]) -> f64 {
    let mut lower_bound = f64::INFINITY;
    for (row, values) in matrix.iter().enumerate() {
        let mut off_diagonal_sum = 0.0;
        for (col, value) in values.iter().enumerate() {
            if row != col {
                off_diagonal_sum += value.abs();
            }
        }
        lower_bound = lower_bound.min(values[row] - off_diagonal_sum);
    }
    lower_bound
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::{
        edge::{accumulate_edge_force, edge_vector},
        geometry::{dot, Cell, Vec3},
    };

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

    fn edge(list: &NeighborList, src: usize, dst: usize) -> Edge {
        *list
            .edges
            .iter()
            .find(|edge| edge.src == src && edge.dst == dst)
            .expect("edge exists")
    }

    #[test]
    fn non_periodic_all_pairs_edge_vectors_are_directed() {
        let structure = Structure::new(
            vec!["H".to_owned(), "He".to_owned()],
            vec![[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]],
        );

        let list = all_pairs_neighbor_list(&structure, 10.0).expect("neighbor list");

        assert_eq!(list.edge_mode, EdgeMode::Directed);
        assert_eq!(list.len(), 2);
        assert_vec_close(edge(&list, 0, 1).vector, [3.0, 4.0, 5.0]);
        assert_vec_close(edge(&list, 1, 0).vector, [-3.0, -4.0, -5.0]);
    }

    #[test]
    fn orthorhombic_pbc_uses_minimum_image_shift() {
        let cell = Cell::from_vectors([10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]);
        let structure = Structure::new(
            vec!["H".to_owned(), "H".to_owned()],
            vec![[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]],
        )
        .with_cell(cell, [true, true, true]);

        let list = all_pairs_neighbor_list(&structure, 2.0).expect("neighbor list");

        assert_eq!(list.len(), 2);
        let forward = edge(&list, 0, 1);
        assert_eq!(forward.shift, [-1, 0, 0]);
        assert_vec_close(forward.shift_cart, [-10.0, 0.0, 0.0]);
        assert_vec_close(forward.vector, [-1.0, 0.0, 0.0]);

        let reverse = edge(&list, 1, 0);
        assert_eq!(reverse.shift, [1, 0, 0]);
        assert_vec_close(reverse.shift_cart, [10.0, 0.0, 0.0]);
        assert_vec_close(reverse.vector, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn triclinic_pbc_uses_row_major_minimum_image() {
        let cell = Cell::from_vectors([2.0, 0.0, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 3.0]);
        let pos_src = cell.fractional_to_cartesian([0.9, 0.1, 0.0]);
        let pos_dst = cell.fractional_to_cartesian([0.1, 0.1, 0.0]);
        let structure =
            Structure::new(vec!["H".to_owned(), "H".to_owned()], vec![pos_src, pos_dst])
                .with_cell(cell, [true, true, true]);

        let list = all_pairs_neighbor_list(&structure, 0.8).expect("neighbor list");
        let forward = edge(&list, 0, 1);

        assert_eq!(forward.shift, [1, 0, 0]);
        assert_vec_close(forward.shift_cart, [2.0, 0.0, 0.0]);
        assert_vec_close(forward.vector, [0.4, 0.0, 0.0]);
    }

    #[test]
    fn cutoff_rule_is_strictly_less_than_cutoff() {
        let structure = Structure::new(
            vec!["H".to_owned(), "H".to_owned()],
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        );

        let at_cutoff = all_pairs_neighbor_list(&structure, 1.0).expect("neighbor list");
        let above_cutoff = all_pairs_neighbor_list(&structure, 1.0 + 1e-12).expect("neighbor list");

        assert!(at_cutoff.is_empty());
        assert_eq!(above_cutoff.len(), 2);
    }

    #[test]
    fn cell_list_matches_all_pairs_for_non_periodic_edges() {
        let structure = Structure::new(
            vec![
                "H".to_owned(),
                "O".to_owned(),
                "C".to_owned(),
                "H".to_owned(),
                "N".to_owned(),
            ],
            vec![
                [0.0, 0.0, 0.0],
                [0.7, 0.1, 0.2],
                [2.1, -0.4, 0.3],
                [3.4, 1.2, -0.2],
                [-1.2, 0.5, 0.4],
            ],
        );

        let expected = all_pairs_multi_image_neighbor_list(&structure, 2.2).expect("all-pairs");
        let actual = cell_list_multi_image_neighbor_list(&structure, 2.2).expect("cell-list");

        assert_eq!(actual.edge_mode, EdgeMode::Directed);
        assert_eq!(actual.edges, expected.edges);
    }

    #[test]
    fn filtered_neighbor_list_preserves_edge_order_and_rejects_larger_cutoff() {
        let structure = Structure::new(
            vec!["H".to_owned(), "O".to_owned(), "C".to_owned()],
            vec![[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [1.8, 0.0, 0.0]],
        );
        let list = cell_list_multi_image_neighbor_list(&structure, 2.0).expect("cell-list");

        let filtered = list.filtered_by_cutoff(1.1).expect("filtered list");

        assert_eq!(filtered.cutoff, 1.1);
        assert_eq!(
            filtered
                .edges
                .iter()
                .map(|edge| (edge.src, edge.dst))
                .collect::<Vec<_>>(),
            vec![(0, 1), (1, 0), (1, 2), (2, 1)]
        );
        let err = filtered
            .filtered_by_cutoff(1.2)
            .expect_err("cannot request beyond available cutoff");
        assert!(matches!(
            err,
            NeighborListError::SharedCutoffTooSmall { .. }
        ));
    }

    #[test]
    fn directed_pair_energy_uses_half_weight_to_avoid_double_counting() {
        let structure = Structure::new(
            vec!["H".to_owned(), "H".to_owned()],
            vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        );
        let list = all_pairs_neighbor_list(&structure, 3.0).expect("neighbor list");
        let mut energy = 0.0;
        let mut forces = vec![ZERO_VEC3; structure.len()];

        for edge in &list.edges {
            energy += 0.25 * dot(edge.vector, edge.vector);
            accumulate_edge_force(
                &mut forces,
                edge,
                [
                    0.5 * edge.vector[0],
                    0.5 * edge.vector[1],
                    0.5 * edge.vector[2],
                ],
            )
            .unwrap();
        }

        assert_eq!(list.len(), 2);
        assert!((energy - 2.0).abs() < 1e-12);
        assert_vec_close(forces[0], [2.0, 0.0, 0.0]);
        assert_vec_close(forces[1], [-2.0, 0.0, 0.0]);
    }

    #[test]
    fn pbc_requires_cell() {
        let mut structure = Structure::new(vec!["H".to_owned()], vec![[0.0, 0.0, 0.0]]);
        structure.pbc = [true, false, false];

        let err = all_pairs_neighbor_list(&structure, 1.0).expect_err("missing cell");

        assert_eq!(err, NeighborListError::MissingCellForPbc);
    }

    #[test]
    fn pbc_rejects_singular_cell() {
        let cell = Cell::from_vectors([1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let structure = Structure::new(vec!["H".to_owned()], vec![[0.0, 0.0, 0.0]])
            .with_cell(cell, [true, false, false]);

        let err = all_pairs_neighbor_list(&structure, 1.0).expect_err("singular cell");

        assert_eq!(err, NeighborListError::SingularCell);
    }

    #[test]
    fn pbc_cutoff_must_fit_single_image_limit() {
        let cell = Cell::from_vectors([2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]);
        let structure = Structure::new(vec!["H".to_owned()], vec![[0.0, 0.0, 0.0]])
            .with_cell(cell, [true, true, true]);

        let err = all_pairs_neighbor_list(&structure, 1.0 + 1.0e-12)
            .expect_err("cutoff larger than half the shortest cell vector should fail");

        assert!(matches!(
            err,
            NeighborListError::CutoffExceedsMinimumImage { .. }
        ));
    }

    #[test]
    fn multi_image_pbc_keeps_nonzero_self_images_and_duplicate_pair_images() {
        let cell = Cell::from_vectors([5.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]);
        let structure = Structure::new(
            vec!["Si".to_owned(), "O".to_owned()],
            vec![[0.2, 0.1, 0.1], [2.7, 0.1, 0.1]],
        )
        .with_cell(cell, [true, false, false]);

        let list = all_pairs_multi_image_neighbor_list(&structure, 6.0).expect("multi-image list");

        let actual = list
            .edges
            .iter()
            .map(|edge| (edge.src, edge.dst, edge.shift, edge.vector))
            .collect::<Vec<_>>();
        assert_eq!(
            actual
                .iter()
                .map(|(src, dst, shift, _)| (*src, *dst, *shift))
                .collect::<Vec<_>>(),
            vec![
                (0, 0, [-1, 0, 0]),
                (0, 1, [-1, 0, 0]),
                (0, 1, [0, 0, 0]),
                (0, 0, [1, 0, 0]),
                (1, 1, [-1, 0, 0]),
                (1, 0, [0, 0, 0]),
                (1, 0, [1, 0, 0]),
                (1, 1, [1, 0, 0]),
            ]
        );
        assert_vec_close(actual[0].3, [-5.0, 0.0, 0.0]);
        assert_vec_close(actual[1].3, [-2.5, 0.0, 0.0]);
        assert_vec_close(actual[2].3, [2.5, 0.0, 0.0]);
        assert_vec_close(actual[3].3, [5.0, 0.0, 0.0]);
    }

    #[test]
    fn cell_list_matches_all_pairs_for_multi_image_pbc_edges() {
        let cell = Cell::from_vectors([5.0, 0.0, 0.0], [0.3, 4.5, 0.0], [0.1, 0.2, 6.0]);
        let structure = Structure::new(
            vec!["Si".to_owned(), "O".to_owned(), "H".to_owned()],
            vec![[0.2, 0.1, 0.1], [2.7, 0.1, 0.1], [1.1, 3.9, 5.8]],
        )
        .with_cell(cell, [true, true, true]);

        let expected = all_pairs_multi_image_neighbor_list(&structure, 5.3).expect("all-pairs");
        let actual = cell_list_multi_image_neighbor_list(&structure, 5.3).expect("cell-list");

        assert_eq!(actual.edge_mode, EdgeMode::Directed);
        assert_eq!(actual.edges, expected.edges);
    }

    #[test]
    fn multi_image_shift_bounds_include_skewed_mptrj_component_two_edges() {
        let cell = Cell::from_vectors(
            [6.248_343_46, -0.040_041_21, 0.064_742_36],
            [-0.034_787_78, 6.363_223_82, 0.135_674_88],
            [0.117_717_2, 0.246_632_07, 10.876_403_21],
        );
        let structure = Structure::new(
            vec!["H".to_owned(), "H".to_owned()],
            vec![
                [-0.696_388_93, 2.182_930_49, 10.527_082_48],
                [6.094_228_76, 2.914_893, 1.442_282_49],
            ],
        )
        .with_cell(cell, [true, true, true]);

        let list = all_pairs_multi_image_neighbor_list(&structure, 6.0)
            .expect("skewed multi-image neighbor list");

        let forward = list
            .edges
            .iter()
            .find(|edge| edge.src == 0 && edge.dst == 1 && edge.shift == [-2, 0, 1])
            .expect("MPTrj local 7825 missing-edge representative should be included");
        assert!(forward.distance < 6.0);
        assert_vec_close(forward.vector, [-5.588_352_03, 1.058_677, 1.662_118_5]);

        assert!(list
            .edges
            .iter()
            .any(|edge| edge.src == 1 && edge.dst == 0 && edge.shift == [2, 0, -1]));
    }

    #[test]
    fn multi_image_shift_bounds_include_translated_mptrj_component_three_shift() {
        // Same skewed MPTrj-derived cell as
        // `multi_image_shift_bounds_include_skewed_mptrj_component_two_edges`, but the
        // second atom is translated by +a (one `a` lattice vector). This is an
        // equivalent PBC structure, yet the required minimum-image edge now needs
        // shift = [-3, 0, 1] instead of [-2, 0, 1]. The OLD scalar bound
        // (`ceil(cutoff / lower_bound) + 1`, which is 2 here) under-counts and misses
        // this edge; the conservative per-axis bound includes it. This is the
        // regression guard that actually fails on the pre-fix implementation.
        let cell = Cell::from_vectors(
            [6.248_343_46, -0.040_041_21, 0.064_742_36],
            [-0.034_787_78, 6.363_223_82, 0.135_674_88],
            [0.117_717_2, 0.246_632_07, 10.876_403_21],
        );
        let structure = Structure::new(
            vec!["H".to_owned(), "H".to_owned()],
            vec![
                [-0.696_388_93, 2.182_930_49, 10.527_082_48],
                // second atom translated by +a relative to the two-edge test
                [12.342_572_22, 2.874_851_79, 1.507_024_85],
            ],
        )
        .with_cell(cell, [true, true, true]);

        let list = all_pairs_multi_image_neighbor_list(&structure, 6.0)
            .expect("translated skewed multi-image neighbor list");

        let forward = list
            .edges
            .iter()
            .find(|edge| edge.src == 0 && edge.dst == 1 && edge.shift == [-3, 0, 1])
            .expect("translated MPTrj representative needs shift [-3, 0, 1]");
        assert!(forward.distance < 6.0);
        assert_vec_close(forward.vector, [-5.588_352_03, 1.058_677, 1.662_118_5]);

        assert!(list
            .edges
            .iter()
            .any(|edge| edge.src == 1 && edge.dst == 0 && edge.shift == [3, 0, -1]));
    }

    #[test]
    fn skewed_pbc_cutoff_uses_short_lattice_combinations() {
        let cell = Cell::from_vectors([1.0, 0.0, 0.0], [0.99, 0.1, 0.0], [0.0, 0.0, 10.0]);
        let structure = Structure::new(vec!["H".to_owned()], vec![[0.0, 0.0, 0.0]])
            .with_cell(cell, [true, true, false]);

        let err = all_pairs_neighbor_list(&structure, 0.2)
            .expect_err("a - b is much shorter than either basis vector");

        assert!(matches!(
            err,
            NeighborListError::CutoffExceedsMinimumImage { .. }
        ));
    }

    #[test]
    fn jacobi_smallest_eigenvalue_lower_bound_handles_equal_diagonal_near_singular_pivot() {
        let smallest = jacobi_smallest_eigenvalue_lower_bound([
            [1.0, 1.0, 0.0],
            [1.0, 1.0 + 1.0e-15, 0.0],
            [0.0, 0.0, 4.0],
        ]);

        assert!(
            smallest.abs() <= 1.0e-12,
            "nearly rank-deficient Gram block should not get an overestimated lower bound: {smallest}"
        );
    }

    #[test]
    fn rejects_non_finite_positions_before_edge_filtering() {
        let structure = Structure::new(
            vec!["H".to_owned(), "H".to_owned()],
            vec![[0.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0]],
        );

        let err = all_pairs_neighbor_list(&structure, 1.0)
            .expect_err("NaN positions should not silently drop edges");

        assert!(matches!(
            err,
            NeighborListError::InvalidPosition {
                atom: 1,
                axis: 0,
                ..
            }
        ));
    }

    #[test]
    fn rejects_non_finite_pbc_cell_vectors() {
        let cell = Cell::from_vectors([1.0, 0.0, 0.0], [0.0, f64::INFINITY, 0.0], [0.0, 0.0, 1.0]);
        let structure = Structure::new(vec!["H".to_owned()], vec![[0.0, 0.0, 0.0]])
            .with_cell(cell, [true, true, false]);

        let err =
            all_pairs_neighbor_list(&structure, 0.1).expect_err("non-finite PBC cell should fail");

        assert!(matches!(
            err,
            NeighborListError::InvalidCellVector {
                vector: 1,
                axis: 1,
                ..
            }
        ));
    }

    #[test]
    fn edge_vector_helper_matches_edge_record_convention() {
        let vector = edge_vector([1.0, 2.0, 3.0], [4.0, 6.0, 8.0], [0.5, 0.0, -1.0]);

        assert_vec_close(vector, [3.5, 4.0, 4.0]);
    }

    #[test]
    fn thin_pbc_cell_large_cutoff_reports_expansion_before_allocation() {
        let cell = Cell::from_vectors([0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]);
        let structure = Structure::new(vec!["H".to_owned()], vec![[0.0, 0.0, 0.0]])
            .with_cell(cell, [true, true, true]);

        let err = cell_list_multi_image_neighbor_list(&structure, 50.0)
            .expect_err("thin cells and large cutoffs should be guarded");

        assert!(matches!(
            err,
            NeighborListError::MultiImageExpansionTooLarge { .. }
        ));
    }

    #[test]
    fn multi_image_shift_bound_never_undercounts_edges() {
        // Brute-force reference: enumerate a generous, fixed +/-8 image range
        // (independent of `multi_image_shift_bounds`) using the exact same edge
        // convention and `distance < cutoff` filter as production, then return the
        // edge set as sorted (src, dst, shift) keys.
        fn brute_force_edge_keys(
            structure: &Structure,
            cutoff: f64,
        ) -> Vec<(usize, usize, [i32; 3])> {
            const RANGE: i32 = 8;
            let cell = structure.cell.expect("cell");
            let mut keys = Vec::new();
            for (src, pos_src) in structure.positions.iter().enumerate() {
                for shift_a in -RANGE..=RANGE {
                    for shift_b in -RANGE..=RANGE {
                        for shift_c in -RANGE..=RANGE {
                            let shift = [shift_a, shift_b, shift_c];
                            if shift
                                .iter()
                                .zip(structure.pbc)
                                .any(|(component, is_periodic)| *component != 0 && !is_periodic)
                            {
                                continue;
                            }
                            let shift_cart = if shift == [0, 0, 0] {
                                ZERO_VEC3
                            } else {
                                cell.cartesian_shift(shift)
                            };
                            for (dst, pos_dst) in structure.positions.iter().enumerate() {
                                if shift == [0, 0, 0] && src == dst {
                                    continue;
                                }
                                let edge =
                                    Edge::new(src, dst, *pos_src, *pos_dst, shift, shift_cart);
                                if edge.distance < cutoff {
                                    keys.push((src, dst, shift));
                                }
                            }
                        }
                    }
                }
            }
            keys.sort_unstable();
            keys
        }

        // (label, cell vectors, positions, pbc, cutoff). The first three share an
        // EXACT integer cutoff/spacing ratio (cutoff == spacing == 3.0) and just
        // below / just above it; then a skewed triclinic cell and a thin 2D cell.
        let cubic = Cell::from_vectors([3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]);
        let cubic_positions = vec![[0.0, 0.0, 0.0], [1.4, 0.2, 0.1]];
        let skewed = Cell::from_vectors([2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.3, 0.4, 2.5]);
        let skewed_positions = vec![[0.1, 0.1, 0.1], [1.2, 0.9, 0.7]];
        let thin = Cell::from_vectors([1.0, 0.0, 0.0], [0.99, 0.2, 0.0], [0.0, 0.0, 10.0]);
        let thin_positions = vec![[0.0, 0.0, 0.0], [0.4, 0.05, 0.0]];

        type NeighborCase = (&'static str, Cell, Vec<Vec3>, [bool; 3], f64);
        let cases: Vec<NeighborCase> = vec![
            (
                "cutoff exactly equals spacing (integer ratio)",
                cubic,
                cubic_positions.clone(),
                [true, true, true],
                3.0,
            ),
            (
                "cutoff just below integer ratio",
                cubic,
                cubic_positions.clone(),
                [true, true, true],
                2.999_999,
            ),
            (
                "cutoff just above integer ratio",
                cubic,
                cubic_positions,
                [true, true, true],
                3.000_001,
            ),
            (
                "skewed triclinic cell",
                skewed,
                skewed_positions,
                [true, true, true],
                4.0,
            ),
            (
                "thin 2D-periodic cell",
                thin,
                thin_positions,
                [true, true, false],
                0.6,
            ),
        ];

        for (label, cell, positions, pbc, cutoff) in cases {
            let symbols = vec!["H".to_owned(); positions.len()];
            let structure = Structure::new(symbols, positions).with_cell(cell, pbc);

            let list = all_pairs_multi_image_neighbor_list(&structure, cutoff)
                .unwrap_or_else(|err| panic!("{label}: multi-image list failed: {err:?}"));
            let mut actual_keys = list
                .edges
                .iter()
                .map(|edge| (edge.src, edge.dst, edge.shift))
                .collect::<Vec<_>>();
            actual_keys.sort_unstable();

            let reference_keys = brute_force_edge_keys(&structure, cutoff);

            assert_eq!(
                actual_keys, reference_keys,
                "{label}: edge set differs from generous brute-force reference (cutoff {cutoff})"
            );
        }
    }
}
