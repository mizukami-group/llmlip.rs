//! Borrowed external graph views for MD-engine integrations.

use std::ops::Range;

use thiserror::Error;

use crate::{
    edge::Edge,
    geometry::{norm, Vec3, ZERO_VEC3},
    neighbor::{EdgeMode, NeighborList},
};

const CUTOFF_TOLERANCE: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug)]
pub struct ExternalGraphView<'a> {
    pub atom_count: usize,
    pub owned_atom_count: usize,
    pub atomic_numbers: &'a [u8],
    pub center_atom_indices: &'a [usize],
    pub center_owned: &'a [bool],
    pub edge_offsets: &'a [usize],
    pub edge_dst: &'a [usize],
    pub edge_vectors: &'a [Vec3],
    pub model_cutoff: f64,
}

impl<'a> ExternalGraphView<'a> {
    #[must_use]
    pub fn center_count(&self) -> usize {
        self.center_atom_indices.len()
    }

    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edge_dst.len()
    }

    #[must_use]
    pub fn center_edge_range(&self, center_index: usize) -> Range<usize> {
        self.edge_offsets[center_index]..self.edge_offsets[center_index + 1]
    }

    #[must_use]
    pub fn is_owned_atom(&self, atom_index: usize) -> bool {
        atom_index < self.owned_atom_count
    }

    #[must_use]
    pub fn edge_distance(&self, edge_index: usize) -> f64 {
        norm(self.edge_vectors[edge_index])
    }

    pub fn validate(&self) -> Result<(), ExternalGraphError> {
        if !self.model_cutoff.is_finite() || self.model_cutoff <= 0.0 {
            return Err(ExternalGraphError::InvalidModelCutoff(self.model_cutoff));
        }
        if self.atomic_numbers.len() != self.atom_count {
            return Err(ExternalGraphError::AtomicNumberCount {
                atom_count: self.atom_count,
                atomic_numbers: self.atomic_numbers.len(),
            });
        }
        if self.owned_atom_count > self.atom_count {
            return Err(ExternalGraphError::OwnedAtomCount {
                atom_count: self.atom_count,
                owned_atom_count: self.owned_atom_count,
            });
        }
        for (atom, atomic_number) in self.atomic_numbers.iter().enumerate() {
            if *atomic_number == 0 {
                return Err(ExternalGraphError::InvalidAtomicNumber {
                    atom,
                    atomic_number: *atomic_number,
                });
            }
        }

        let center_count = self.center_count();
        if self.center_owned.len() != center_count {
            return Err(ExternalGraphError::CenterOwnedCount {
                center_count,
                center_owned: self.center_owned.len(),
            });
        }
        if self.edge_offsets.len() != center_count + 1 {
            return Err(ExternalGraphError::EdgeOffsetCount {
                center_count,
                edge_offsets: self.edge_offsets.len(),
            });
        }
        if let Some(first) = self.edge_offsets.first() {
            if *first != 0 {
                return Err(ExternalGraphError::FirstEdgeOffset { first: *first });
            }
        }
        for offset_index in 1..self.edge_offsets.len() {
            let previous = self.edge_offsets[offset_index - 1];
            let current = self.edge_offsets[offset_index];
            if current < previous {
                return Err(ExternalGraphError::NonMonotonicEdgeOffsets {
                    offset_index,
                    previous,
                    current,
                });
            }
        }
        let edge_count = self.edge_offsets.last().copied().unwrap_or(0);
        if edge_count != self.edge_dst.len() || edge_count != self.edge_vectors.len() {
            return Err(ExternalGraphError::EdgeCountMismatch {
                edge_offsets_last: edge_count,
                edge_dst: self.edge_dst.len(),
                edge_vectors: self.edge_vectors.len(),
            });
        }

        for (center_index, atom) in self.center_atom_indices.iter().copied().enumerate() {
            if atom >= self.atom_count {
                return Err(ExternalGraphError::CenterAtomOutOfRange {
                    center_index,
                    atom,
                    atom_count: self.atom_count,
                });
            }
            let expected_owned = atom < self.owned_atom_count;
            if self.center_owned[center_index] != expected_owned {
                return Err(ExternalGraphError::CenterOwnershipMismatch {
                    center_index,
                    atom,
                    owned_atom_count: self.owned_atom_count,
                    center_owned: self.center_owned[center_index],
                });
            }
        }

        for (edge_index, dst) in self.edge_dst.iter().copied().enumerate() {
            if dst >= self.atom_count {
                return Err(ExternalGraphError::EdgeDstOutOfRange {
                    edge_index,
                    dst,
                    atom_count: self.atom_count,
                });
            }
            let vector = self.edge_vectors[edge_index];
            for (axis, value) in vector.iter().copied().enumerate() {
                if !value.is_finite() {
                    return Err(ExternalGraphError::NonFiniteEdgeVector {
                        edge_index,
                        axis,
                        value,
                    });
                }
            }
            let distance = norm(vector);
            if !distance.is_finite() {
                return Err(ExternalGraphError::NonFiniteEdgeDistance {
                    edge_index,
                    distance,
                });
            }
            if distance >= self.model_cutoff {
                return Err(ExternalGraphError::EdgeOutsideCutoff {
                    edge_index,
                    distance,
                    cutoff: self.model_cutoff,
                });
            }
        }

        Ok(())
    }

    pub fn validate_complete_centers(&self) -> Result<(), ExternalGraphError> {
        self.validate()?;
        if self.center_count() != self.atom_count {
            return Err(ExternalGraphError::IncompleteCenterCount {
                atom_count: self.atom_count,
                center_count: self.center_count(),
            });
        }
        let mut seen = vec![false; self.atom_count];
        for (center_index, atom) in self.center_atom_indices.iter().copied().enumerate() {
            if seen[atom] {
                return Err(ExternalGraphError::DuplicateCenter { center_index, atom });
            }
            seen[atom] = true;
            let expected_owned = self.is_owned_atom(atom);
            if self.center_owned[center_index] != expected_owned {
                return Err(ExternalGraphError::CenterOwnershipMismatch {
                    center_index,
                    atom,
                    owned_atom_count: self.owned_atom_count,
                    center_owned: self.center_owned[center_index],
                });
            }
        }
        for (atom, present) in seen.iter().copied().enumerate() {
            if !present {
                return Err(ExternalGraphError::MissingCenter { atom });
            }
        }
        Ok(())
    }

    pub fn validate_for_model_cutoff(&self, model_cutoff: f64) -> Result<(), ExternalGraphError> {
        self.validate()?;
        if (self.model_cutoff - model_cutoff).abs() > CUTOFF_TOLERANCE {
            return Err(ExternalGraphError::ModelCutoffMismatch {
                graph_cutoff: self.model_cutoff,
                model_cutoff,
            });
        }
        Ok(())
    }

    pub fn to_neighbor_list(&self) -> Result<NeighborList, ExternalGraphError> {
        self.validate()?;
        let mut edges = Vec::with_capacity(self.edge_count());
        for (center_index, src) in self.center_atom_indices.iter().copied().enumerate() {
            for edge_index in self.center_edge_range(center_index) {
                let vector = self.edge_vectors[edge_index];
                edges.push(Edge {
                    src,
                    dst: self.edge_dst[edge_index],
                    shift: [0; 3],
                    shift_cart: ZERO_VEC3,
                    vector,
                    distance: norm(vector),
                });
            }
        }
        Ok(NeighborList {
            edges,
            cutoff: self.model_cutoff,
            edge_mode: EdgeMode::Directed,
        })
    }
}

#[derive(Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum ExternalGraphError {
    #[error("external graph model cutoff must be finite and positive, got {0}")]
    InvalidModelCutoff(f64),

    #[error("external graph has {atomic_numbers} atomic numbers for {atom_count} atoms")]
    AtomicNumberCount {
        atom_count: usize,
        atomic_numbers: usize,
    },

    #[error("external graph owned atom count {owned_atom_count} exceeds atom count {atom_count}")]
    OwnedAtomCount {
        atom_count: usize,
        owned_atom_count: usize,
    },

    #[error("external graph atom {atom} has invalid atomic number {atomic_number}")]
    InvalidAtomicNumber { atom: usize, atomic_number: u8 },

    #[error("external graph has {center_owned} center-owned flags for {center_count} centers")]
    CenterOwnedCount {
        center_count: usize,
        center_owned: usize,
    },

    #[error("external graph has {edge_offsets} edge offsets for {center_count} centers")]
    EdgeOffsetCount {
        center_count: usize,
        edge_offsets: usize,
    },

    #[error("external graph first edge offset must be 0, got {first}")]
    FirstEdgeOffset { first: usize },

    #[error("external graph edge offsets are not monotonic at {offset_index}: previous {previous}, current {current}")]
    NonMonotonicEdgeOffsets {
        offset_index: usize,
        previous: usize,
        current: usize,
    },

    #[error("external graph edge count mismatch: offsets end at {edge_offsets_last}, dst len {edge_dst}, vectors len {edge_vectors}")]
    EdgeCountMismatch {
        edge_offsets_last: usize,
        edge_dst: usize,
        edge_vectors: usize,
    },

    #[error(
        "external graph center {center_index} atom index {atom} is outside atom count {atom_count}"
    )]
    CenterAtomOutOfRange {
        center_index: usize,
        atom: usize,
        atom_count: usize,
    },

    #[error("external graph center {center_index} ownership mismatch for atom {atom}: owned_atom_count {owned_atom_count}, center_owned {center_owned}")]
    CenterOwnershipMismatch {
        center_index: usize,
        atom: usize,
        owned_atom_count: usize,
        center_owned: bool,
    },

    #[error("external graph edge {edge_index} dst atom {dst} is outside atom count {atom_count}")]
    EdgeDstOutOfRange {
        edge_index: usize,
        dst: usize,
        atom_count: usize,
    },

    #[error("external graph edge {edge_index} vector axis {axis} must be finite, got {value}")]
    NonFiniteEdgeVector {
        edge_index: usize,
        axis: usize,
        value: f64,
    },

    #[error("external graph edge {edge_index} distance must be finite, got {distance}")]
    NonFiniteEdgeDistance { edge_index: usize, distance: f64 },

    #[error(
        "external graph edge {edge_index} distance {distance} is outside model cutoff {cutoff}"
    )]
    EdgeOutsideCutoff {
        edge_index: usize,
        distance: f64,
        cutoff: f64,
    },

    #[error("external graph cutoff {graph_cutoff} does not match model cutoff {model_cutoff}")]
    ModelCutoffMismatch {
        graph_cutoff: f64,
        model_cutoff: f64,
    },

    #[error("external graph has {center_count} centers for {atom_count} atoms")]
    IncompleteCenterCount {
        atom_count: usize,
        center_count: usize,
    },

    #[error("external graph atom {atom} appears as a duplicate center at center {center_index}")]
    DuplicateCenter { center_index: usize, atom: usize },

    #[error("external graph atom {atom} is missing from center list")]
    MissingCenter { atom: usize },
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn valid_graph() -> ExternalGraphView<'static> {
        ExternalGraphView {
            atom_count: 2,
            owned_atom_count: 2,
            atomic_numbers: &[1, 8],
            center_atom_indices: &[0, 1],
            center_owned: &[true, true],
            edge_offsets: &[0, 1, 2],
            edge_dst: &[1, 0],
            edge_vectors: &[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            model_cutoff: 3.5,
        }
    }

    #[test]
    fn validates_well_formed_external_graph() {
        valid_graph().validate().unwrap();
    }

    #[test]
    fn validates_complete_centers() {
        valid_graph().validate_complete_centers().unwrap();
    }

    #[test]
    fn rejects_shape_mismatch() {
        let graph = ExternalGraphView {
            edge_offsets: &[0, 1],
            ..valid_graph()
        };

        assert!(matches!(
            graph.validate(),
            Err(ExternalGraphError::EdgeOffsetCount {
                center_count: 2,
                edge_offsets: 2,
            })
        ));
    }

    #[test]
    fn rejects_ownership_mismatch() {
        let graph = ExternalGraphView {
            owned_atom_count: 1,
            ..valid_graph()
        };

        assert!(matches!(
            graph.validate(),
            Err(ExternalGraphError::CenterOwnershipMismatch {
                center_index: 1,
                atom: 1,
                owned_atom_count: 1,
                center_owned: true,
            })
        ));
    }

    #[test]
    fn rejects_duplicate_complete_center() {
        let graph = ExternalGraphView {
            center_atom_indices: &[0, 0],
            center_owned: &[true, true],
            edge_offsets: &[0, 0, 0],
            edge_dst: &[],
            edge_vectors: &[],
            ..valid_graph()
        };

        assert!(matches!(
            graph.validate_complete_centers(),
            Err(ExternalGraphError::DuplicateCenter {
                center_index: 1,
                atom: 0,
            })
        ));
    }

    #[test]
    fn rejects_missing_complete_center() {
        let graph = ExternalGraphView {
            atom_count: 3,
            atomic_numbers: &[1, 8, 1],
            center_atom_indices: &[0, 1],
            center_owned: &[true, true],
            edge_offsets: &[0, 0, 0],
            edge_dst: &[],
            edge_vectors: &[],
            ..valid_graph()
        };

        assert!(matches!(
            graph.validate_complete_centers(),
            Err(ExternalGraphError::IncompleteCenterCount {
                atom_count: 3,
                center_count: 2,
            })
        ));
    }

    #[test]
    fn rejects_edges_outside_cutoff() {
        let graph = ExternalGraphView {
            model_cutoff: 1.0,
            ..valid_graph()
        };

        assert!(matches!(
            graph.validate(),
            Err(ExternalGraphError::EdgeOutsideCutoff { edge_index: 0, .. })
        ));
    }
}
