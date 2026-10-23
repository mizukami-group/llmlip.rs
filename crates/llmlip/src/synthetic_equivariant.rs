//! Small SH/TP synthetic local energy model for force-path validation.
//!
//! This model is intentionally narrow: l=1 spherical harmonics, explicit
//! hand-checkable tensor-product plans, fixed weights, and force-only VJPs.
//! It validates the SH/TP energy graph before any Nequip-lite work.

use e3nn_core::{
    radial::{
        cosine_cutoff_forward, cutoff_gaussian_radial_vjp, distance_vjp_accumulate,
        gaussian_radial_basis_forward,
    },
    CgEntry, FixedMlp, Irreps, MlpCache, SphericalHarmonicsPlan, TensorProductInstruction,
    TensorProductPlan, Vec3,
};

use crate::{
    all_pairs_neighbor_list, external_graph::ExternalGraphView,
    force_accum::scatter_edge_vector_vjp, synthetic::validate_radial_parameters, Prediction,
    PredictionOptions, Structure, SyntheticModelError,
};

const NODE_ENERGY_SCALE: f64 = 0.5;
const INV_SQRT_3: f64 = 0.577_350_269_189_625_8;

#[derive(Clone, Debug)]
pub struct SyntheticEquivariantModel {
    cutoff: f64,
    centers: Vec<f64>,
    gamma: f64,
    distance_eps: f64,
    edge_mlp: FixedMlp,
    sh_plan: SphericalHarmonicsPlan,
    edge_message_tp: TensorProductPlan,
    node_energy_tp: TensorProductPlan,
}

impl SyntheticEquivariantModel {
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
            sh_plan: SphericalHarmonicsPlan::component(1, true)?,
            edge_message_tp: edge_message_tp_plan()?,
            node_energy_tp: node_energy_tp_plan()?,
        })
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

    #[must_use]
    pub fn cutoff(&self) -> f64 {
        self.cutoff
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
        if structure.pbc.iter().any(|value| *value) {
            return Err(SyntheticModelError::PeriodicUnsupported);
        }

        let neighbor_list = all_pairs_neighbor_list(structure, self.cutoff)?;
        let mut node_features = vec![[0.0; 3]; structure.len()];
        let mut edge_caches = Vec::new();
        let mut basis = vec![0.0; self.centers.len()];
        let mut weighted_basis = vec![0.0; self.centers.len()];
        let mut edge_weight = [0.0; 1];
        let mut sh_out = vec![0.0; self.sh_plan.output_dim()];
        let mut message = [0.0; 3];

        if compute_forces {
            edge_caches.reserve(neighbor_list.edges.len());
        }

        for edge in &neighbor_list.edges {
            gaussian_radial_basis_forward(edge.distance, &self.centers, self.gamma, &mut basis)?;
            let cutoff = cosine_cutoff_forward(edge.distance, self.cutoff)?;
            for (target, value) in weighted_basis.iter_mut().zip(&basis) {
                *target = cutoff.value * value;
            }

            let mut mlp_cache = MlpCache::default();
            self.edge_mlp
                .forward(&weighted_basis, &mut edge_weight, &mut mlp_cache)?;
            self.sh_plan.forward(edge.vector, &mut sh_out)?;
            let sh_vector = [sh_out[1], sh_out[2], sh_out[3]];
            self.edge_message_tp
                .forward(&[1.0], &sh_vector, Some(&edge_weight), &mut message)?;

            for (target, value) in node_features[edge.src].iter_mut().zip(message) {
                *target += value;
            }

            if compute_forces {
                edge_caches.push(EdgeCache {
                    src: edge.src,
                    dst: edge.dst,
                    vector: edge.vector,
                    distance: edge.distance,
                    basis: basis.clone(),
                    mlp_cache,
                    weight: edge_weight,
                    sh_vector,
                });
            }
        }

        let mut energy = 0.0;
        let mut node_bars = if compute_forces {
            vec![[0.0; 3]; structure.len()]
        } else {
            Vec::new()
        };
        let mut node_energy = [0.0; 1];
        let mut grad_x = [0.0; 3];
        let mut grad_y = [0.0; 3];

        for (node_index, feature) in node_features.iter().enumerate() {
            self.node_energy_tp
                .forward(feature, feature, None, &mut node_energy)?;
            energy += NODE_ENERGY_SCALE * node_energy[0];

            if compute_forces {
                grad_x.fill(0.0);
                grad_y.fill(0.0);
                self.node_energy_tp.vjp(
                    feature,
                    feature,
                    None,
                    &[NODE_ENERGY_SCALE],
                    &mut grad_x,
                    &mut grad_y,
                    None,
                )?;
                for axis in 0..3 {
                    node_bars[node_index][axis] += grad_x[axis] + grad_y[axis];
                }
            }
        }

        let mut forces = if compute_forces {
            vec![[0.0; 3]; structure.len()]
        } else {
            Vec::new()
        };

        if compute_forces {
            self.reverse_edges(&edge_caches, &node_bars, &mut forces)?;
        }

        Ok(ModelOutput { energy, forces })
    }

    fn evaluate_external_graph(
        &self,
        graph: &ExternalGraphView<'_>,
        compute_forces: bool,
    ) -> Result<ModelOutput, SyntheticModelError> {
        graph.validate_for_model_cutoff(self.cutoff)?;

        let mut node_features = vec![[0.0; 3]; graph.atom_count];
        let mut edge_caches = Vec::new();
        let mut basis = vec![0.0; self.centers.len()];
        let mut weighted_basis = vec![0.0; self.centers.len()];
        let mut edge_weight = [0.0; 1];
        let mut sh_out = vec![0.0; self.sh_plan.output_dim()];
        let mut message = [0.0; 3];

        if compute_forces {
            edge_caches.reserve(graph.edge_count());
        }

        for center_index in 0..graph.center_count() {
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

                let mut mlp_cache = MlpCache::default();
                self.edge_mlp
                    .forward(&weighted_basis, &mut edge_weight, &mut mlp_cache)?;
                self.sh_plan.forward(vector, &mut sh_out)?;
                let sh_vector = [sh_out[1], sh_out[2], sh_out[3]];
                self.edge_message_tp.forward(
                    &[1.0],
                    &sh_vector,
                    Some(&edge_weight),
                    &mut message,
                )?;

                for (target, value) in node_features[src].iter_mut().zip(message) {
                    *target += value;
                }

                if compute_forces {
                    edge_caches.push(EdgeCache {
                        src,
                        dst,
                        vector,
                        distance,
                        basis: basis.clone(),
                        mlp_cache,
                        weight: edge_weight,
                        sh_vector,
                    });
                }
            }
        }

        let mut energy = 0.0;
        let mut node_bars = if compute_forces {
            vec![[0.0; 3]; graph.atom_count]
        } else {
            Vec::new()
        };
        let mut node_energy = [0.0; 1];
        let mut grad_x = [0.0; 3];
        let mut grad_y = [0.0; 3];

        for center_index in 0..graph.center_count() {
            if !graph.center_owned[center_index] {
                continue;
            }
            let node_index = graph.center_atom_indices[center_index];
            let feature = &node_features[node_index];
            self.node_energy_tp
                .forward(feature, feature, None, &mut node_energy)?;
            energy += NODE_ENERGY_SCALE * node_energy[0];

            if compute_forces {
                grad_x.fill(0.0);
                grad_y.fill(0.0);
                self.node_energy_tp.vjp(
                    feature,
                    feature,
                    None,
                    &[NODE_ENERGY_SCALE],
                    &mut grad_x,
                    &mut grad_y,
                    None,
                )?;
                for axis in 0..3 {
                    node_bars[node_index][axis] += grad_x[axis] + grad_y[axis];
                }
            }
        }

        let mut forces = if compute_forces {
            vec![[0.0; 3]; graph.atom_count]
        } else {
            Vec::new()
        };

        if compute_forces {
            self.reverse_edges(&edge_caches, &node_bars, &mut forces)?;
        }

        Ok(ModelOutput { energy, forces })
    }

    fn reverse_edges(
        &self,
        edge_caches: &[EdgeCache],
        node_bars: &[Vec3],
        forces: &mut [Vec3],
    ) -> Result<(), SyntheticModelError> {
        let mut grad_constant = [0.0; 1];
        let mut grad_sh_vector = [0.0; 3];
        let mut grad_weight = [0.0; 1];
        let mut grad_weighted_basis = vec![0.0; self.centers.len()];
        let mut grad_sh_full = vec![0.0; self.sh_plan.output_dim()];

        for edge in edge_caches.iter().rev() {
            grad_constant.fill(0.0);
            grad_sh_vector.fill(0.0);
            grad_weight.fill(0.0);
            self.edge_message_tp.vjp(
                &[1.0],
                &edge.sh_vector,
                Some(&edge.weight),
                &node_bars[edge.src],
                &mut grad_constant,
                &mut grad_sh_vector,
                Some(&mut grad_weight),
            )?;

            grad_weighted_basis.fill(0.0);
            self.edge_mlp.vjp_input_only(
                &edge.mlp_cache,
                &grad_weight,
                &mut grad_weighted_basis,
            )?;
            let bar_distance = cutoff_gaussian_radial_vjp(
                edge.distance,
                self.cutoff,
                &self.centers,
                self.gamma,
                &edge.basis,
                &grad_weighted_basis,
            )?;

            let mut bar_r = [0.0; 3];
            distance_vjp_accumulate(edge.vector, bar_distance, self.distance_eps, &mut bar_r)?;

            grad_sh_full.fill(0.0);
            grad_sh_full[1] = grad_sh_vector[0];
            grad_sh_full[2] = grad_sh_vector[1];
            grad_sh_full[3] = grad_sh_vector[2];
            self.sh_plan
                .vjp_edge_vector(edge.vector, &grad_sh_full, &mut bar_r)?;

            scatter_edge_vector_vjp(edge.src, edge.dst, bar_r, forces)?;
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
struct ModelOutput {
    energy: f64,
    forces: Vec<Vec3>,
}

#[derive(Clone, Debug)]
struct EdgeCache {
    src: usize,
    dst: usize,
    vector: Vec3,
    distance: f64,
    basis: Vec<f64>,
    mlp_cache: MlpCache,
    weight: [f64; 1],
    sh_vector: Vec3,
}

fn scalar_irreps() -> Result<Irreps, SyntheticModelError> {
    Ok("1x0e".parse::<Irreps>()?)
}

fn vector_irreps() -> Result<Irreps, SyntheticModelError> {
    Ok("1x1o".parse::<Irreps>()?)
}

fn edge_message_tp_plan() -> Result<TensorProductPlan, SyntheticModelError> {
    Ok(TensorProductPlan::new(
        scalar_irreps()?,
        vector_irreps()?,
        vector_irreps()?,
        vec![TensorProductInstruction {
            in1_segment: 0,
            in2_segment: 0,
            out_segment: 0,
            cg_offset: 0,
            cg_len: 3,
            weight_offset: Some(0),
            path_weight: 1.0,
        }],
        vec![
            CgEntry {
                i_m: 0,
                j_m: 0,
                k_m: 0,
                value: 1.0,
            },
            CgEntry {
                i_m: 0,
                j_m: 1,
                k_m: 1,
                value: 1.0,
            },
            CgEntry {
                i_m: 0,
                j_m: 2,
                k_m: 2,
                value: 1.0,
            },
        ],
        1,
    )?)
}

fn node_energy_tp_plan() -> Result<TensorProductPlan, SyntheticModelError> {
    Ok(TensorProductPlan::new(
        vector_irreps()?,
        vector_irreps()?,
        scalar_irreps()?,
        vec![TensorProductInstruction {
            in1_segment: 0,
            in2_segment: 0,
            out_segment: 0,
            cg_offset: 0,
            cg_len: 3,
            weight_offset: None,
            path_weight: 1.0,
        }],
        vec![
            CgEntry {
                i_m: 0,
                j_m: 0,
                k_m: 0,
                value: INV_SQRT_3,
            },
            CgEntry {
                i_m: 1,
                j_m: 1,
                k_m: 0,
                value: INV_SQRT_3,
            },
            CgEntry {
                i_m: 2,
                j_m: 2,
                k_m: 0,
                value: INV_SQRT_3,
            },
        ],
        0,
    )?)
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

    fn model() -> SyntheticEquivariantModel {
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
        SyntheticEquivariantModel::new(3.5, vec![0.5, 1.0, 1.5, 2.0], 0.8, mlp).unwrap()
    }

    fn structure() -> Structure {
        Structure::new(
            vec!["H".to_owned(), "H".to_owned(), "O".to_owned()],
            vec![[0.0, 0.0, 0.0], [1.1, -0.2, 0.4], [-0.6, 1.3, 0.7]],
        )
    }

    fn translated(structure: &Structure, shift: Vec3) -> Structure {
        Structure::new(
            structure.species.clone(),
            structure
                .positions
                .iter()
                .map(|pos| [pos[0] + shift[0], pos[1] + shift[1], pos[2] + shift[2]])
                .collect(),
        )
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
        model: &SyntheticEquivariantModel,
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
    fn returns_finite_energy_and_forces() {
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
    }

    #[test]
    fn forces_match_energy_finite_difference() {
        let model = model();
        let structure = structure();
        let (_energy, forces) = model.energy_and_forces(&structure).unwrap();

        for (atom, force) in forces.iter().enumerate() {
            for (axis, force_component) in force.iter().enumerate() {
                let expected = finite_difference_force(&model, &structure, atom, axis);
                assert_close(*force_component, expected, 2.0e-5);
            }
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
    fn forces_are_rotation_covariant() {
        let model = model();
        let structure = structure();
        let rotated_structure = rotated(&structure);
        let (energy, forces) = model.energy_and_forces(&structure).unwrap();
        let (rotated_energy, rotated_forces) = model.energy_and_forces(&rotated_structure).unwrap();

        assert_close(energy, rotated_energy, 1.0e-10);
        for atom in 0..structure.len() {
            assert_vec_close(rotated_forces[atom], rotate_z(forces[atom]), 1.0e-8);
        }
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

        assert_vec_close(net, [0.0; 3], 1.0e-10);
    }

    #[test]
    fn energy_is_angularly_sensitive() {
        let model = model();
        let right_angle = Structure::new(
            vec!["O".to_owned(), "H".to_owned(), "H".to_owned()],
            vec![[0.0, 0.0, 0.0], [2.8, 0.0, 0.0], [0.0, 2.8, 0.0]],
        );
        let obtuse_angle = Structure::new(
            vec!["O".to_owned(), "H".to_owned(), "H".to_owned()],
            vec![
                [0.0, 0.0, 0.0],
                [2.8, 0.0, 0.0],
                [-1.4, 2.424_871_130_596_428_3, 0.0],
            ],
        );

        assert_close(
            distance(right_angle.positions[0], right_angle.positions[1]),
            distance(obtuse_angle.positions[0], obtuse_angle.positions[1]),
            1.0e-12,
        );
        assert_close(
            distance(right_angle.positions[0], right_angle.positions[2]),
            distance(obtuse_angle.positions[0], obtuse_angle.positions[2]),
            1.0e-12,
        );
        assert!(distance(right_angle.positions[1], right_angle.positions[2]) > model.cutoff);
        assert!(distance(obtuse_angle.positions[1], obtuse_angle.positions[2]) > model.cutoff);

        let energy_diff =
            (model.energy(&right_angle).unwrap() - model.energy(&obtuse_angle).unwrap()).abs();
        assert!(
            energy_diff > 1.0e-8,
            "expected angular energy sensitivity, diff={energy_diff:.6e}"
        );
    }

    #[test]
    fn rejects_pbc_for_initial_equivariant_model() {
        let model = model();
        let mut structure = structure();
        structure.pbc = [true, false, false];

        assert!(matches!(
            model.energy(&structure),
            Err(SyntheticModelError::PeriodicUnsupported)
        ));
    }

    fn distance(a: Vec3, b: Vec3) -> f64 {
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let dz = b[2] - a[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}
