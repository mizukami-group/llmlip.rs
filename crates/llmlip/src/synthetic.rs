//! Synthetic radial local energy model for end-to-end force validation.
//!
//! This is the first T09 model: fixed weights, radial-only, and conservative
//! forces through an explicit force-only reverse pass. It also serves as the
//! first model-level PBC force-sign validation target before enabling PBC in
//! SH/TP production-lite models.

use e3nn_core::{
    radial::{
        cosine_cutoff_forward, cutoff_gaussian_radial_vjp, distance_vjp_accumulate,
        gaussian_radial_basis_forward,
    },
    CoreError, FixedMlp, MlpCache, Vec3,
};
use thiserror::Error;

use crate::{
    all_pairs_neighbor_list,
    external_graph::{ExternalGraphError, ExternalGraphView},
    force_accum::{scatter_edge_vector_vjp, ForceAccumError},
    structure::{Prediction, PredictionOptions, Structure},
    NeighborListError,
};

const DIRECTED_EDGE_ENERGY_SCALE: f64 = 0.5;

#[derive(Debug, Error)]
pub enum SyntheticModelError {
    #[error(transparent)]
    Core(#[from] CoreError),

    #[error(transparent)]
    NeighborList(#[from] NeighborListError),

    #[error(transparent)]
    ForceAccum(#[from] ForceAccumError),

    #[error(transparent)]
    ExternalGraph(#[from] ExternalGraphError),

    #[error("synthetic radial model requires non-periodic structures")]
    PeriodicUnsupported,

    #[error("synthetic radial model does not support stress or virial prediction")]
    UnsupportedStressVirial,

    #[error("synthetic radial model MLP output dimension must be 1, got {0}")]
    InvalidMlpOutput(usize),

    #[error("synthetic radial model centers length {centers} does not match MLP input dimension {input_dim}")]
    RadialMlpShape { centers: usize, input_dim: usize },

    #[error("invalid synthetic model parameter: {0}")]
    InvalidParameter(String),
}

#[derive(Clone, Debug)]
pub struct SyntheticRadialModel {
    cutoff: f64,
    centers: Vec<f64>,
    gamma: f64,
    distance_eps: f64,
    edge_mlp: FixedMlp,
}

impl SyntheticRadialModel {
    pub fn new(
        cutoff: f64,
        centers: Vec<f64>,
        gamma: f64,
        edge_mlp: FixedMlp,
    ) -> Result<Self, SyntheticModelError> {
        validate_radial_parameters(cutoff, &centers, gamma)?;
        if edge_mlp.input_dim() != centers.len() {
            return Err(SyntheticModelError::RadialMlpShape {
                centers: centers.len(),
                input_dim: edge_mlp.input_dim(),
            });
        }
        if edge_mlp.output_dim() != 1 {
            return Err(SyntheticModelError::InvalidMlpOutput(edge_mlp.output_dim()));
        }
        Ok(Self {
            cutoff,
            centers,
            gamma,
            distance_eps: 1.0e-12,
            edge_mlp,
        })
    }

    #[must_use]
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    #[must_use]
    pub fn centers(&self) -> &[f64] {
        &self.centers
    }

    #[must_use]
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    #[must_use]
    pub fn distance_eps(&self) -> f64 {
        self.distance_eps
    }

    #[must_use]
    pub fn edge_mlp(&self) -> &FixedMlp {
        &self.edge_mlp
    }

    pub fn predict(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
    ) -> Result<Prediction, SyntheticModelError> {
        if options.compute_stress || options.compute_virial {
            return Err(SyntheticModelError::UnsupportedStressVirial);
        }
        if !options.compute_energy && !options.compute_forces {
            return Ok(Prediction::default());
        }

        let out = self.evaluate(structure, options.compute_forces)?;
        Ok(Prediction {
            energy: options.compute_energy.then_some(out.energy),
            forces: options.compute_forces.then_some(out.forces),
            stress: None,
            virial: None,
        })
    }

    pub fn energy(&self, structure: &Structure) -> Result<f64, SyntheticModelError> {
        self.evaluate(structure, false).map(|out| out.energy)
    }

    pub fn energy_and_forces(
        &self,
        structure: &Structure,
    ) -> Result<(f64, Vec<Vec3>), SyntheticModelError> {
        let out = self.evaluate(structure, true)?;
        Ok((out.energy, out.forces))
    }

    pub(crate) fn predict_external_graph(
        &self,
        graph: &ExternalGraphView<'_>,
        options: &PredictionOptions,
    ) -> Result<Prediction, SyntheticModelError> {
        if options.compute_stress || options.compute_virial {
            return Err(SyntheticModelError::UnsupportedStressVirial);
        }
        if !options.compute_energy && !options.compute_forces {
            return Ok(Prediction::default());
        }

        let out = self.evaluate_external_graph(graph, options.compute_forces)?;
        Ok(Prediction {
            energy: options.compute_energy.then_some(out.energy),
            forces: options.compute_forces.then_some(out.forces),
            stress: None,
            virial: None,
        })
    }

    fn evaluate(
        &self,
        structure: &Structure,
        compute_forces: bool,
    ) -> Result<ModelOutput, SyntheticModelError> {
        let neighbor_list = all_pairs_neighbor_list(structure, self.cutoff)?;
        let mut energy = 0.0;
        let mut forces = if compute_forces {
            vec![[0.0; 3]; structure.len()]
        } else {
            Vec::new()
        };
        let mut basis = vec![0.0; self.centers.len()];
        let mut weighted_basis = vec![0.0; self.centers.len()];
        let mut bar_weighted_basis = vec![0.0; self.centers.len()];
        let mut mlp_cache = MlpCache::default();
        let mut edge_energy = [0.0; 1];

        for edge in &neighbor_list.edges {
            gaussian_radial_basis_forward(edge.distance, &self.centers, self.gamma, &mut basis)?;
            let cutoff = cosine_cutoff_forward(edge.distance, self.cutoff)?;
            for (target, value) in weighted_basis.iter_mut().zip(&basis) {
                *target = cutoff.value * value;
            }

            self.edge_mlp
                .forward(&weighted_basis, &mut edge_energy, &mut mlp_cache)?;
            energy += DIRECTED_EDGE_ENERGY_SCALE * edge_energy[0];

            if compute_forces {
                // FixedMlp VJPs accumulate, so reset this per directed edge.
                bar_weighted_basis.fill(0.0);
                self.edge_mlp.vjp_input_only(
                    &mlp_cache,
                    &[DIRECTED_EDGE_ENERGY_SCALE],
                    &mut bar_weighted_basis,
                )?;
                let bar_distance = cutoff_gaussian_radial_vjp(
                    edge.distance,
                    self.cutoff,
                    &self.centers,
                    self.gamma,
                    &basis,
                    &bar_weighted_basis,
                )?;
                let mut bar_r = [0.0; 3];
                distance_vjp_accumulate(edge.vector, bar_distance, self.distance_eps, &mut bar_r)?;
                scatter_edge_vector_vjp(edge.src, edge.dst, bar_r, &mut forces)?;
            }
        }

        Ok(ModelOutput { energy, forces })
    }

    fn evaluate_external_graph(
        &self,
        graph: &ExternalGraphView<'_>,
        compute_forces: bool,
    ) -> Result<ModelOutput, SyntheticModelError> {
        graph.validate_for_model_cutoff(self.cutoff)?;

        let mut energy = 0.0;
        let mut forces = if compute_forces {
            vec![[0.0; 3]; graph.atom_count]
        } else {
            Vec::new()
        };
        let mut basis = vec![0.0; self.centers.len()];
        let mut weighted_basis = vec![0.0; self.centers.len()];
        let mut bar_weighted_basis = vec![0.0; self.centers.len()];
        let mut mlp_cache = MlpCache::default();
        let mut edge_energy = [0.0; 1];

        for center_index in 0..graph.center_count() {
            if !graph.center_owned[center_index] {
                continue;
            }
            let src = graph.center_atom_indices[center_index];
            for edge_index in graph.center_edge_range(center_index) {
                let dst = graph.edge_dst[edge_index];
                let vector = graph.edge_vectors[edge_index];
                let distance = graph.edge_distance(edge_index);

                gaussian_radial_basis_forward(distance, &self.centers, self.gamma, &mut basis)?;
                let cutoff = cosine_cutoff_forward(distance, self.cutoff)?;
                for (target, value) in weighted_basis.iter_mut().zip(&basis) {
                    *target = cutoff.value * value;
                }

                self.edge_mlp
                    .forward(&weighted_basis, &mut edge_energy, &mut mlp_cache)?;
                energy += DIRECTED_EDGE_ENERGY_SCALE * edge_energy[0];

                if compute_forces {
                    bar_weighted_basis.fill(0.0);
                    self.edge_mlp.vjp_input_only(
                        &mlp_cache,
                        &[DIRECTED_EDGE_ENERGY_SCALE],
                        &mut bar_weighted_basis,
                    )?;
                    let bar_distance = cutoff_gaussian_radial_vjp(
                        distance,
                        self.cutoff,
                        &self.centers,
                        self.gamma,
                        &basis,
                        &bar_weighted_basis,
                    )?;
                    let mut bar_r = [0.0; 3];
                    distance_vjp_accumulate(vector, bar_distance, self.distance_eps, &mut bar_r)?;
                    scatter_edge_vector_vjp(src, dst, bar_r, &mut forces)?;
                }
            }
        }

        Ok(ModelOutput { energy, forces })
    }
}

pub(crate) fn validate_radial_parameters(
    cutoff: f64,
    centers: &[f64],
    gamma: f64,
) -> Result<(), SyntheticModelError> {
    if !cutoff.is_finite() || cutoff <= 0.0 {
        return Err(SyntheticModelError::InvalidParameter(format!(
            "cutoff must be finite and positive, got {cutoff}"
        )));
    }
    if centers.is_empty() {
        return Err(SyntheticModelError::InvalidParameter(
            "centers must not be empty".to_owned(),
        ));
    }
    if centers.iter().any(|value| !value.is_finite()) {
        return Err(SyntheticModelError::InvalidParameter(
            "centers must contain only finite values".to_owned(),
        ));
    }
    if !gamma.is_finite() || gamma <= 0.0 {
        return Err(SyntheticModelError::InvalidParameter(format!(
            "gamma must be finite and positive, got {gamma}"
        )));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct ModelOutput {
    energy: f64,
    forces: Vec<Vec3>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use e3nn_core::{Activation, DenseLayer};

    const FD_EPS: f64 = 1.0e-6;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {expected:.16e}, got {actual:.16e}, diff {:.3e}",
            (actual - expected).abs()
        );
    }

    fn assert_vec_close(actual: Vec3, expected: Vec3, tol: f64) {
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() <= tol,
                "axis {axis}: expected {:.16e}, got {:.16e}, diff {:.3e}",
                expected[axis],
                actual[axis],
                (actual[axis] - expected[axis]).abs()
            );
        }
    }

    fn model() -> SyntheticRadialModel {
        let mlp = FixedMlp::new(vec![
            DenseLayer::new(
                4,
                3,
                vec![
                    0.8, -0.4, 0.2, 0.5, -0.3, 0.7, -0.6, 0.1, 0.25, 0.35, -0.45, 0.9,
                ],
                vec![0.05, -0.1, 0.2],
                Activation::Tanh,
            )
            .unwrap(),
            DenseLayer::new(
                3,
                1,
                vec![1.1, -0.75, 0.4],
                vec![0.03],
                Activation::Identity,
            )
            .unwrap(),
        ])
        .unwrap();
        SyntheticRadialModel::new(3.5, vec![0.5, 1.0, 1.5, 2.0], 0.8, mlp).unwrap()
    }

    fn structure() -> Structure {
        Structure::new(
            vec!["H".to_owned(), "H".to_owned(), "O".to_owned()],
            vec![[0.0, 0.0, 0.0], [1.1, -0.2, 0.4], [-0.6, 1.3, 0.7]],
        )
    }

    fn translated(structure: &Structure, shift: Vec3) -> Structure {
        let mut translated = Structure::new(
            structure.species.clone(),
            structure
                .positions
                .iter()
                .map(|pos| [pos[0] + shift[0], pos[1] + shift[1], pos[2] + shift[2]])
                .collect(),
        );
        translated.cell = structure.cell;
        translated.pbc = structure.pbc;
        translated
    }

    fn rotate_z(v: Vec3) -> Vec3 {
        let c = 0.6_f64;
        let s = 0.8_f64;
        [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]
    }

    fn rotated(structure: &Structure) -> Structure {
        Structure::new(
            structure.species.clone(),
            structure.positions.iter().copied().map(rotate_z).collect(),
        )
    }

    fn finite_difference_force(
        model: &SyntheticRadialModel,
        structure: &Structure,
        atom: usize,
        axis: usize,
    ) -> f64 {
        let mut plus = structure.clone();
        plus.positions[atom][axis] += FD_EPS;
        let e_plus = model.energy(&plus).unwrap();

        let mut minus = structure.clone();
        minus.positions[atom][axis] -= FD_EPS;
        let e_minus = model.energy(&minus).unwrap();

        -(e_plus - e_minus) / (2.0 * FD_EPS)
    }

    #[test]
    fn returns_energy_and_forces() {
        let model = model();
        let structure = structure();
        let (energy, forces) = model.energy_and_forces(&structure).unwrap();

        assert!(energy.is_finite());
        assert_eq!(forces.len(), structure.len());
        assert!(forces.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn prediction_options_control_outputs() {
        let model = model();
        let structure = structure();

        let neither = model
            .predict(
                &structure,
                &PredictionOptions {
                    compute_energy: false,
                    compute_forces: false,
                    ..PredictionOptions::default()
                },
            )
            .unwrap();
        assert!(neither.energy.is_none());
        assert!(neither.forces.is_none());

        let energy_only = model
            .predict(
                &structure,
                &PredictionOptions {
                    compute_energy: true,
                    compute_forces: false,
                    ..PredictionOptions::default()
                },
            )
            .unwrap();
        assert!(energy_only.energy.is_some());
        assert!(energy_only.forces.is_none());

        let forces_only = model
            .predict(
                &structure,
                &PredictionOptions {
                    compute_energy: false,
                    compute_forces: true,
                    ..PredictionOptions::default()
                },
            )
            .unwrap();
        assert!(forces_only.energy.is_none());
        assert!(forces_only.forces.is_some());

        let both = model
            .predict(
                &structure,
                &PredictionOptions {
                    compute_energy: true,
                    compute_forces: true,
                    ..PredictionOptions::default()
                },
            )
            .unwrap();
        assert_close(energy_only.energy.unwrap(), both.energy.unwrap(), 1.0e-12);
        for (force, expected) in forces_only
            .forces
            .unwrap()
            .iter()
            .zip(both.forces.unwrap().iter())
        {
            assert_vec_close(*force, *expected, 1.0e-12);
        }
    }

    #[test]
    fn forces_match_energy_finite_difference() {
        let model = model();
        let structure = structure();
        let (_energy, forces) = model.energy_and_forces(&structure).unwrap();

        // This end-to-end path includes cutoff, Gaussian radial features, tanh,
        // directed-edge accumulation, and force scatter, so use the project
        // recommended f64 force tolerance from docs/TESTING_SPEC.md.
        for (atom, force) in forces.iter().enumerate() {
            for (axis, force_component) in force.iter().enumerate() {
                let expected = finite_difference_force(&model, &structure, atom, axis);
                assert_close(*force_component, expected, 1.0e-5);
            }
        }
    }

    #[test]
    fn pairs_outside_cutoff_do_not_contribute() {
        let model = model();
        let structure = Structure::new(
            vec!["H".to_owned(), "H".to_owned()],
            vec![[0.0, 0.0, 0.0], [1.05 * model.cutoff(), 0.0, 0.0]],
        );
        let (energy, forces) = model.energy_and_forces(&structure).unwrap();

        assert_close(energy, 0.0, 0.0);
        for force in forces {
            assert_vec_close(force, [0.0; 3], 0.0);
        }
    }

    #[test]
    fn finite_difference_forces_hold_near_cutoff() {
        let model = model();
        let structure = Structure::new(
            vec!["H".to_owned(), "H".to_owned()],
            vec![[0.0, 0.0, 0.0], [0.95 * model.cutoff(), 0.0, 0.0]],
        );
        let (_energy, forces) = model.energy_and_forces(&structure).unwrap();

        for (atom, force) in forces.iter().enumerate() {
            let expected = finite_difference_force(&model, &structure, atom, 0);
            assert_close(force[0], expected, 1.0e-5);
        }
    }

    #[test]
    fn energy_is_translation_invariant() {
        let model = model();
        let structure = structure();
        let shifted = translated(&structure, [2.0, -3.0, 0.75]);

        assert_close(
            model.energy(&structure).unwrap(),
            model.energy(&shifted).unwrap(),
            1.0e-12,
        );
    }

    #[test]
    fn net_force_is_zero() {
        let model = model();
        let structure = structure();
        let (_energy, forces) = model.energy_and_forces(&structure).unwrap();
        let net = forces.iter().fold([0.0; 3], |mut total, force| {
            total[0] += force[0];
            total[1] += force[1];
            total[2] += force[2];
            total
        });

        assert_vec_close(net, [0.0; 3], 1.0e-12);
    }

    #[test]
    fn radial_model_is_rotation_invariant_and_force_covariant() {
        let model = model();
        let structure = structure();
        let rotated_structure = rotated(&structure);
        let (energy, forces) = model.energy_and_forces(&structure).unwrap();
        let (rotated_energy, rotated_forces) = model.energy_and_forces(&rotated_structure).unwrap();

        assert_close(energy, rotated_energy, 1.0e-12);
        for atom in 0..structure.len() {
            assert_vec_close(rotated_forces[atom], rotate_z(forces[atom]), 1.0e-10);
        }
    }

    #[test]
    fn orthorhombic_pbc_matches_equivalent_minimum_image_structure() {
        let model = model();
        let pbc_structure = Structure::new(
            vec!["H".to_owned(), "H".to_owned()],
            vec![[0.2, 0.0, 0.0], [9.7, 0.0, 0.0]],
        )
        .with_cell(
            crate::Cell::from_vectors([10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]),
            [true, false, false],
        );
        let equivalent = Structure::new(
            vec!["H".to_owned(), "H".to_owned()],
            vec![[0.2, 0.0, 0.0], [-0.3, 0.0, 0.0]],
        );

        let (pbc_energy, pbc_forces) = model.energy_and_forces(&pbc_structure).unwrap();
        let (equivalent_energy, equivalent_forces) = model.energy_and_forces(&equivalent).unwrap();

        assert_close(pbc_energy, equivalent_energy, 1.0e-12);
        for (actual, expected) in pbc_forces.iter().zip(&equivalent_forces) {
            assert_vec_close(*actual, *expected, 1.0e-12);
        }
    }

    #[test]
    fn pbc_forces_match_energy_finite_difference() {
        let model = model();
        let structure = Structure::new(
            vec!["H".to_owned(), "H".to_owned(), "O".to_owned()],
            vec![[0.2, 0.3, 0.4], [7.6, 0.5, 0.2], [1.2, 2.0, 0.9]],
        )
        .with_cell(
            crate::Cell::from_vectors([8.0, 0.0, 0.0], [0.3, 8.5, 0.0], [0.0, 0.2, 9.0]),
            [true, true, false],
        );
        let (_energy, forces) = model.energy_and_forces(&structure).unwrap();

        for (atom, force) in forces.iter().enumerate() {
            for (axis, force_component) in force.iter().enumerate() {
                let expected = finite_difference_force(&model, &structure, atom, axis);
                assert_close(*force_component, expected, 1.0e-5);
            }
        }
    }

    #[test]
    fn pbc_energy_is_translation_invariant() {
        let model = model();
        let structure = Structure::new(
            vec!["H".to_owned(), "H".to_owned(), "O".to_owned()],
            vec![[0.2, 0.3, 0.4], [7.6, 0.5, 0.2], [1.2, 2.0, 0.9]],
        )
        .with_cell(
            crate::Cell::from_vectors([8.0, 0.0, 0.0], [0.3, 8.5, 0.0], [0.0, 0.2, 9.0]),
            [true, true, false],
        );
        let shifted = translated(&structure, [1.1, -0.7, 0.5]);

        assert_close(
            model.energy(&structure).unwrap(),
            model.energy(&shifted).unwrap(),
            1.0e-12,
        );
    }

    #[test]
    fn validates_model_shapes() {
        let bad_input = FixedMlp::new(vec![DenseLayer::new(
            3,
            1,
            vec![1.0; 3],
            vec![0.0],
            Activation::Identity,
        )
        .unwrap()])
        .unwrap();
        assert!(matches!(
            SyntheticRadialModel::new(3.5, vec![0.5, 1.0], 0.8, bad_input),
            Err(SyntheticModelError::RadialMlpShape { .. })
        ));

        let bad_output = FixedMlp::new(vec![DenseLayer::new(
            2,
            2,
            vec![1.0; 4],
            vec![0.0; 2],
            Activation::Identity,
        )
        .unwrap()])
        .unwrap();
        assert!(matches!(
            SyntheticRadialModel::new(3.5, vec![0.5, 1.0], 0.8, bad_output),
            Err(SyntheticModelError::InvalidMlpOutput(2))
        ));
    }
}
