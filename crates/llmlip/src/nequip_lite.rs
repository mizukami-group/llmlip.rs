//! Narrow fixed-weight Nequip-lite runtime exported from deterministic `NeQuIP` models.

use std::{
    collections::{HashMap, HashSet},
    f64::consts::PI,
};

use e3nn_core::{
    radial::distance_vjp_accumulate, CgEntry, CoreError, Irreps, ShNormalization,
    SphericalHarmonicsPlan, TensorProductInstruction, TensorProductPlan, Vec3,
};
use serde::Deserialize;
use thiserror::Error;

use crate::{
    all_pairs_neighbor_list,
    force_accum::{
        accumulate_edge_displacement_gradient, scatter_edge_vector_vjp,
        stress_from_displacement_gradient, virial_from_displacement_gradient, ForceAccumError,
        Mat3,
    },
    NeighborListError, Prediction, PredictionOptions, Structure,
};

const MODEL_SCHEMA_VERSION: &str = "nequip-lite-model-v1";
const DISTANCE_EPS: f64 = 1.0e-12;

#[derive(Clone, Debug)]
pub struct NequipLiteExternalTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
}

#[derive(Debug, Error)]
pub enum NequipLiteError {
    #[error(transparent)]
    Core(#[from] CoreError),

    #[error(transparent)]
    NeighborList(#[from] NeighborListError),

    #[error(transparent)]
    ForceAccum(#[from] ForceAccumError),

    #[error("invalid Nequip-lite model: {0}")]
    InvalidModel(String),

    #[error("Nequip-lite stress prediction requires a finite periodic cell")]
    StressRequiresCell,

    #[error("unknown species {0:?} for Nequip-lite model")]
    UnknownSpecies(String),
}

#[derive(Clone, Debug)]
pub struct NequipLiteModel {
    type_names: Vec<String>,
    cutoff: f64,
    num_bessels: usize,
    polynomial_cutoff_p: usize,
    edge_embedding_factor: f64,
    node_embedding: Vec<Vec<f64>>,
    layers: Vec<NequipLiteLayer>,
    readout: DenseMatrix,
    per_type_energy_scales: Vec<f64>,
    per_type_energy_shifts: Vec<f64>,
    sh_plan: SphericalHarmonicsPlan,
}

impl NequipLiteModel {
    pub fn from_json_str(text: &str) -> Result<Self, NequipLiteError> {
        Self::from_json_str_with_external_tensors(text, HashMap::new())
    }

    pub fn from_json_str_with_external_tensors(
        text: &str,
        external_tensors: HashMap<String, NequipLiteExternalTensor>,
    ) -> Result<Self, NequipLiteError> {
        let spec: ModelSpec = serde_json::from_str(text)
            .map_err(|err| invalid_model(format!("invalid Nequip-lite JSON: {err}")))?;
        Self::from_spec(spec, &external_tensors)
    }

    pub fn predict(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
    ) -> Result<Prediction, NequipLiteError> {
        let needs_reverse =
            options.compute_forces || options.compute_stress || options.compute_virial;
        if !options.compute_energy && !needs_reverse {
            return Ok(Prediction::default());
        }
        let out = self.evaluate(
            structure,
            needs_reverse,
            options.compute_stress,
            options.compute_virial,
        )?;
        Ok(Prediction {
            energy: options.compute_energy.then_some(out.energy),
            forces: options.compute_forces.then_some(out.forces),
            stress: if options.compute_stress {
                out.stress
            } else {
                None
            },
            virial: if options.compute_virial {
                out.virial
            } else {
                None
            },
        })
    }

    pub fn energy(&self, structure: &Structure) -> Result<f64, NequipLiteError> {
        self.evaluate(structure, false, false, false)
            .map(|out| out.energy)
    }

    pub fn energy_and_forces(
        &self,
        structure: &Structure,
    ) -> Result<(f64, Vec<Vec3>), NequipLiteError> {
        let out = self.evaluate(structure, true, false, false)?;
        Ok((out.energy, out.forces))
    }

    #[must_use]
    pub fn type_names(&self) -> &[String] {
        &self.type_names
    }

    #[must_use]
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    pub fn l_max(&self) -> usize {
        self.sh_plan.l_max
    }

    fn from_spec(
        spec: ModelSpec,
        external_tensors: &HashMap<String, NequipLiteExternalTensor>,
    ) -> Result<Self, NequipLiteError> {
        if spec.schema_version != MODEL_SCHEMA_VERSION {
            return Err(invalid_model(format!(
                "schema_version must be {MODEL_SCHEMA_VERSION:?}, got {:?}",
                spec.schema_version
            )));
        }
        if spec.type_names.is_empty() {
            return Err(invalid_model("type_names must not be empty"));
        }
        validate_type_names(&spec.type_names)?;
        if !spec.cutoff.is_finite() || spec.cutoff <= 0.0 {
            return Err(invalid_model(format!(
                "cutoff must be finite and positive, got {}",
                spec.cutoff
            )));
        }
        if spec.l_max > 4 {
            return Err(invalid_model(format!(
                "l_max={} is unsupported; Nequip-lite supports l_max <= 4",
                spec.l_max
            )));
        }
        if spec.num_bessels == 0 {
            return Err(invalid_model("num_bessels must be nonzero"));
        }
        if spec.polynomial_cutoff_p < 2 {
            return Err(invalid_model("polynomial_cutoff_p must be >= 2"));
        }
        if !spec.edge_embedding_factor.is_finite() {
            return Err(invalid_model("edge_embedding_factor must be finite"));
        }
        let node_embedding = matrix_from_embedded_or_tensor(
            "node_embedding",
            spec.node_embedding,
            spec.node_embedding_tensor,
            external_tensors,
        )?;
        if node_embedding.len() != spec.type_names.len() {
            return Err(invalid_model(format!(
                "node_embedding row count {} does not match type_names length {}",
                node_embedding.len(),
                spec.type_names.len()
            )));
        }
        let per_type_energy_scales = vector_from_embedded_or_tensor(
            "per_type_energy_scales",
            spec.per_type_energy_scales,
            spec.per_type_energy_scales_tensor,
            external_tensors,
        )?;
        let per_type_energy_shifts = vector_from_embedded_or_tensor(
            "per_type_energy_shifts",
            spec.per_type_energy_shifts,
            spec.per_type_energy_shifts_tensor,
            external_tensors,
        )?;
        if spec.layers.is_empty() {
            return Err(invalid_model("layers must not be empty"));
        }
        if per_type_energy_scales.len() != spec.type_names.len()
            || per_type_energy_shifts.len() != spec.type_names.len()
        {
            return Err(invalid_model(
                "per-type scale/shift lengths must match type_names",
            ));
        }

        let sh_plan = SphericalHarmonicsPlan::new(spec.l_max, true, ShNormalization::Component)?;
        let mut layers = Vec::with_capacity(spec.layers.len());
        let mut current_dim = checked_rectangular("node_embedding", &node_embedding)?;
        for (index, layer_spec) in spec.layers.into_iter().enumerate() {
            let layer = NequipLiteLayer::from_spec(layer_spec, external_tensors)?;
            if layer.linear1.in_dim != current_dim {
                return Err(invalid_model(format!(
                    "layer {index} linear1 input dimension {} does not match previous feature dimension {current_dim}",
                    layer.linear1.in_dim
                )));
            }
            if layer.linear1.out_dim != layer.tensor_product.input1_dim() {
                return Err(invalid_model(format!(
                    "layer {index} linear1 output dimension does not match tensor-product input1"
                )));
            }
            if layer.tensor_product.input2_dim() != sh_plan.output_dim() {
                return Err(invalid_model(format!(
                    "layer {index} tensor-product input2 dimension {} does not match SH dimension {}",
                    layer.tensor_product.input2_dim(),
                    sh_plan.output_dim()
                )));
            }
            if layer.edge_mlp.in_dim != spec.num_bessels
                || layer.edge_mlp.out_dim != layer.tensor_product.weight_numel
            {
                return Err(invalid_model(format!(
                    "layer {index} edge_mlp dimensions must be num_bessels -> tensor-product weight_numel"
                )));
            }
            if layer.linear2.in_dim != layer.tensor_product.output_dim()
                || layer.linear2.out_dim != layer.gate.input_dim
            {
                return Err(invalid_model(format!(
                    "layer {index} linear2 dimensions do not match tensor-product/gate"
                )));
            }
            current_dim = layer.gate.output_dim;
            layers.push(layer);
        }
        let readout = DenseMatrix::from_spec(spec.readout, external_tensors)?;
        if readout.in_dim != current_dim || readout.out_dim != 1 {
            return Err(invalid_model(format!(
                "readout must map final feature dimension {current_dim} to one scalar"
            )));
        }

        Ok(Self {
            type_names: spec.type_names,
            cutoff: spec.cutoff,
            num_bessels: spec.num_bessels,
            polynomial_cutoff_p: spec.polynomial_cutoff_p,
            edge_embedding_factor: spec.edge_embedding_factor,
            node_embedding,
            layers,
            readout,
            per_type_energy_scales,
            per_type_energy_shifts,
            sh_plan,
        })
    }

    fn evaluate(
        &self,
        structure: &Structure,
        compute_forces: bool,
        compute_stress: bool,
        compute_virial: bool,
    ) -> Result<ModelOutput, NequipLiteError> {
        let compute_displacement_gradient = compute_stress || compute_virial;
        // For NeQuIP parity, edge.src is the convolution center/receiver and
        // edge.dst is the neighbor/sender. The vector points receiver -> sender.
        let type_ids = self.type_ids(structure)?;
        let neighbor_list = all_pairs_neighbor_list(structure, self.cutoff)?;
        let edge_inputs = self.edge_inputs(&neighbor_list.edges)?;

        let mut features = type_ids
            .iter()
            .map(|type_id| self.node_embedding[*type_id].clone())
            .collect::<Vec<_>>();
        let mut layer_caches = Vec::new();
        if compute_forces {
            layer_caches.reserve(self.layers.len());
        }

        for layer in &self.layers {
            let (next_features, cache) =
                layer.forward(&features, &neighbor_list.edges, &edge_inputs)?;
            if compute_forces {
                layer_caches.push(cache);
            }
            features = next_features;
        }

        let mut energy = 0.0;
        let mut feature_bars = if compute_forces {
            vec![vec![0.0; self.readout.in_dim]; structure.len()]
        } else {
            Vec::new()
        };
        let mut raw_energy = [0.0; 1];

        for (node_index, feature) in features.iter().enumerate() {
            self.readout.forward(feature, &mut raw_energy)?;
            let type_id = type_ids[node_index];
            energy += self.per_type_energy_scales[type_id] * raw_energy[0]
                + self.per_type_energy_shifts[type_id];
            if compute_forces {
                self.readout.vjp(
                    &[self.per_type_energy_scales[type_id]],
                    &mut feature_bars[node_index],
                )?;
            }
        }

        let mut forces = if compute_forces {
            vec![[0.0; 3]; structure.len()]
        } else {
            Vec::new()
        };

        let mut stress = None;
        let mut virial = None;
        if compute_forces {
            let displacement_gradient = self.reverse_layers(
                &neighbor_list.edges,
                &edge_inputs,
                &layer_caches,
                feature_bars,
                &mut forces,
                compute_displacement_gradient,
            )?;
            if compute_displacement_gradient {
                let displacement_gradient = symmetric_part(displacement_gradient);
                if compute_stress {
                    stress = Some(stress_from_displacement_gradient(
                        displacement_gradient,
                        structure
                            .cell
                            .as_ref()
                            .ok_or(NequipLiteError::StressRequiresCell)?,
                    )?);
                }
                if compute_virial {
                    virial = Some(virial_from_displacement_gradient(displacement_gradient));
                }
            }
        }

        Ok(ModelOutput {
            energy,
            forces,
            stress,
            virial,
        })
    }

    fn reverse_layers(
        &self,
        edges: &[crate::Edge],
        edge_inputs: &[EdgeInput],
        layer_caches: &[LayerCache],
        mut feature_bars: Vec<Vec<f64>>,
        forces: &mut [Vec3],
        compute_displacement_gradient: bool,
    ) -> Result<Mat3, NequipLiteError> {
        let mut displacement_gradient = [[0.0; 3]; 3];
        for (layer, cache) in self.layers.iter().zip(layer_caches).rev() {
            let node_count = cache.gate_input.len();
            let mut gate_input_bars = vec![vec![0.0; layer.gate.input_dim]; node_count];
            for node in 0..node_count {
                layer.gate.vjp(
                    &cache.gate_input[node],
                    &feature_bars[node],
                    &mut gate_input_bars[node],
                )?;
            }

            let mut scatter_bars = vec![vec![0.0; layer.linear2.in_dim]; node_count];
            for node in 0..node_count {
                layer
                    .linear2
                    .vjp(&gate_input_bars[node], &mut scatter_bars[node])?;
            }

            let mut normed_bars = vec![vec![0.0; layer.linear1.out_dim]; node_count];
            let mut edge_attr_bars = vec![vec![0.0; self.sh_plan.output_dim()]; edges.len()];
            let mut edge_embedding_bars = vec![vec![0.0; self.num_bessels]; edges.len()];

            let mut grad_x = vec![0.0; layer.tensor_product.input1_dim()];
            let mut grad_edge_attr = vec![0.0; layer.tensor_product.input2_dim()];
            let mut grad_weight = vec![0.0; layer.tensor_product.weight_numel];
            for (edge_index, edge) in edges.iter().enumerate() {
                grad_x.fill(0.0);
                grad_edge_attr.fill(0.0);
                grad_weight.fill(0.0);
                layer.tensor_product.vjp(
                    &cache.normed_features[edge.dst],
                    &edge_inputs[edge_index].edge_attr,
                    Some(&cache.edge_weights[edge_index]),
                    &scatter_bars[edge.src],
                    &mut grad_x,
                    &mut grad_edge_attr,
                    Some(&mut grad_weight),
                )?;
                for (target, value) in normed_bars[edge.dst].iter_mut().zip(&grad_x) {
                    *target += *value;
                }
                for (target, value) in edge_attr_bars[edge_index].iter_mut().zip(&grad_edge_attr) {
                    *target += *value;
                }
                layer
                    .edge_mlp
                    .vjp(&grad_weight, &mut edge_embedding_bars[edge_index])?;
            }

            for (edge_index, edge) in edges.iter().enumerate() {
                let mut bar_r = [0.0; 3];
                self.sh_plan.vjp_edge_vector(
                    edge.vector,
                    &edge_attr_bars[edge_index],
                    &mut bar_r,
                )?;
                let bar_distance = self
                    .edge_embedding_vjp(&edge_inputs[edge_index], &edge_embedding_bars[edge_index]);
                distance_vjp_accumulate(edge.vector, bar_distance, DISTANCE_EPS, &mut bar_r)?;
                scatter_edge_vector_vjp(edge.src, edge.dst, bar_r, forces)?;
                if compute_displacement_gradient {
                    accumulate_edge_displacement_gradient(
                        edge.vector,
                        bar_r,
                        &mut displacement_gradient,
                    );
                }
            }

            let mut previous_feature_bars =
                vec![vec![0.0; layer.linear1.in_dim]; cache.input_node_count];
            let mut linear1_bar = vec![0.0; layer.linear1.out_dim];
            for node in 0..cache.input_node_count {
                linear1_bar.copy_from_slice(&normed_bars[node]);
                for value in &mut linear1_bar {
                    *value *= layer.avg_num_neighbors_norm;
                }
                layer
                    .linear1
                    .vjp(&linear1_bar, &mut previous_feature_bars[node])?;
            }
            feature_bars = previous_feature_bars;
        }
        Ok(displacement_gradient)
    }

    fn type_ids(&self, structure: &Structure) -> Result<Vec<usize>, NequipLiteError> {
        structure
            .species
            .iter()
            .map(|species| {
                self.type_names
                    .iter()
                    .position(|name| name == species)
                    .ok_or_else(|| NequipLiteError::UnknownSpecies(species.clone()))
            })
            .collect()
    }

    fn edge_inputs(&self, edges: &[crate::Edge]) -> Result<Vec<EdgeInput>, NequipLiteError> {
        let mut out = Vec::with_capacity(edges.len());
        let mut edge_attr = vec![0.0; self.sh_plan.output_dim()];
        for edge in edges {
            self.sh_plan.forward(edge.vector, &mut edge_attr)?;
            let radial = self.edge_embedding(edge.distance);
            out.push(EdgeInput {
                norm_length: radial.norm_length,
                bessel: radial.bessel,
                cutoff: radial.cutoff,
                edge_embedding: radial.edge_embedding,
                edge_attr: edge_attr.clone(),
            });
        }
        Ok(out)
    }

    fn edge_embedding(&self, distance: f64) -> RadialEmbedding {
        let norm_length = distance / self.cutoff;
        let cutoff = polynomial_cutoff(norm_length, self.polynomial_cutoff_p);
        let mut bessel = vec![0.0; self.num_bessels];
        let mut edge_embedding = vec![0.0; self.num_bessels];
        for k in 0..self.num_bessels {
            let n = (k + 1) as f64;
            bessel[k] = n * sinc_pi(n * norm_length);
            edge_embedding[k] = self.edge_embedding_factor * cutoff * bessel[k];
        }
        RadialEmbedding {
            norm_length,
            bessel,
            cutoff,
            edge_embedding,
        }
    }

    fn edge_embedding_vjp(&self, input: &EdgeInput, grad_embedding: &[f64]) -> f64 {
        let cutoff_deriv =
            polynomial_cutoff_derivative(input.norm_length, self.polynomial_cutoff_p);
        let mut bar_norm_length = 0.0;
        for (k, grad) in grad_embedding.iter().enumerate() {
            let n = (k + 1) as f64;
            let bessel_deriv = n * n * sinc_pi_derivative(n * input.norm_length);
            bar_norm_length += self.edge_embedding_factor
                * grad
                * (input.cutoff * bessel_deriv + input.bessel[k] * cutoff_deriv);
        }
        bar_norm_length / self.cutoff
    }
}

#[derive(Clone, Debug)]
struct NequipLiteLayer {
    linear1: DenseMatrix,
    avg_num_neighbors_norm: f64,
    edge_mlp: DenseMatrix,
    tensor_product: TensorProductPlan,
    linear2: DenseMatrix,
    gate: GateSpec,
}

impl NequipLiteLayer {
    fn from_spec(
        spec: LayerSpec,
        external_tensors: &HashMap<String, NequipLiteExternalTensor>,
    ) -> Result<Self, NequipLiteError> {
        let input_irreps = spec.input_irreps.parse::<Irreps>()?;
        if input_irreps.dim() != spec.linear1.in_dim {
            return Err(invalid_model(format!(
                "input_irreps dimension {} does not match linear1 input dimension {}",
                input_irreps.dim(),
                spec.linear1.in_dim
            )));
        }
        if !spec.avg_num_neighbors_norm.is_finite() {
            return Err(invalid_model("avg_num_neighbors_norm must be finite"));
        }
        Ok(Self {
            linear1: DenseMatrix::from_spec(spec.linear1, external_tensors)?,
            avg_num_neighbors_norm: spec.avg_num_neighbors_norm,
            edge_mlp: DenseMatrix::from_spec(spec.edge_mlp, external_tensors)?,
            tensor_product: tensor_product_from_spec(spec.tensor_product)?,
            linear2: DenseMatrix::from_spec(spec.linear2, external_tensors)?,
            gate: GateSpec::from_spec(spec.gate)?,
        })
    }

    fn forward(
        &self,
        features: &[Vec<f64>],
        edges: &[crate::Edge],
        edge_inputs: &[EdgeInput],
    ) -> Result<(Vec<Vec<f64>>, LayerCache), NequipLiteError> {
        let mut linear1 = vec![vec![0.0; self.linear1.out_dim]; features.len()];
        for (out, feature) in linear1.iter_mut().zip(features) {
            self.linear1.forward(feature, out)?;
        }

        for feature in &mut linear1 {
            for value in feature {
                *value *= self.avg_num_neighbors_norm;
            }
        }
        let normed_features = linear1;

        let mut scatter = vec![vec![0.0; self.tensor_product.output_dim()]; features.len()];
        let mut edge_weights = vec![vec![0.0; self.tensor_product.weight_numel]; edges.len()];
        let mut message = vec![0.0; self.tensor_product.output_dim()];
        for (edge_index, edge) in edges.iter().enumerate() {
            self.edge_mlp.forward(
                &edge_inputs[edge_index].edge_embedding,
                &mut edge_weights[edge_index],
            )?;
            self.tensor_product.forward(
                &normed_features[edge.dst],
                &edge_inputs[edge_index].edge_attr,
                Some(&edge_weights[edge_index]),
                &mut message,
            )?;
            for (target, value) in scatter[edge.src].iter_mut().zip(&message) {
                *target += value;
            }
        }

        let mut gate_input = vec![vec![0.0; self.linear2.out_dim]; features.len()];
        for (out, value) in gate_input.iter_mut().zip(&scatter) {
            self.linear2.forward(value, out)?;
        }

        let mut output = vec![vec![0.0; self.gate.output_dim]; features.len()];
        for (out, value) in output.iter_mut().zip(&gate_input) {
            self.gate.forward(value, out)?;
        }

        Ok((
            output,
            LayerCache {
                input_node_count: features.len(),
                normed_features,
                gate_input,
                edge_weights,
            },
        ))
    }
}

#[derive(Clone, Debug)]
struct DenseMatrix {
    in_dim: usize,
    out_dim: usize,
    weight: Vec<f64>,
}

impl DenseMatrix {
    fn from_spec(
        spec: DenseSpec,
        external_tensors: &HashMap<String, NequipLiteExternalTensor>,
    ) -> Result<Self, NequipLiteError> {
        if spec.in_dim == 0 || spec.out_dim == 0 {
            return Err(invalid_model("dense dimensions must be nonzero"));
        }
        let expected_len = spec
            .in_dim
            .checked_mul(spec.out_dim)
            .ok_or_else(|| invalid_model("dense weight shape overflows"))?;
        let weight = dense_weight_from_embedded_or_tensor(
            "dense.weight",
            spec.weight,
            spec.weight_tensor,
            external_tensors,
            spec.out_dim,
            spec.in_dim,
        )?;
        if weight.len() != expected_len {
            return Err(invalid_model(format!(
                "dense weight length mismatch for {} -> {}",
                spec.in_dim, spec.out_dim
            )));
        }
        if weight.iter().any(|value| !value.is_finite()) {
            return Err(invalid_model("dense weights must be finite"));
        }
        Ok(Self {
            in_dim: spec.in_dim,
            out_dim: spec.out_dim,
            weight,
        })
    }

    fn forward(&self, input: &[f64], out: &mut [f64]) -> Result<(), NequipLiteError> {
        if input.len() != self.in_dim || out.len() != self.out_dim {
            return Err(invalid_model("dense forward shape mismatch"));
        }
        for (out_index, out_value) in out.iter_mut().enumerate() {
            let row = out_index * self.in_dim;
            *out_value = input
                .iter()
                .enumerate()
                .map(|(in_index, value)| self.weight[row + in_index] * value)
                .sum();
        }
        Ok(())
    }

    fn vjp(&self, grad_out: &[f64], grad_input: &mut [f64]) -> Result<(), NequipLiteError> {
        if grad_out.len() != self.out_dim || grad_input.len() != self.in_dim {
            return Err(invalid_model("dense VJP shape mismatch"));
        }
        // VJP accumulates into grad_input so callers can combine multiple paths.
        for (out_index, bar_out) in grad_out.iter().enumerate() {
            let row = out_index * self.in_dim;
            for (in_index, target) in grad_input.iter_mut().enumerate() {
                *target += self.weight[row + in_index] * bar_out;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct GateSpec {
    input_dim: usize,
    output_dim: usize,
    scalar_indices: Vec<usize>,
    scalar_activation_scale: f64,
    gate_activation_scale: f64,
    gated_groups: Vec<GateGroup>,
}

impl GateSpec {
    fn from_spec(spec: GateSpecJson) -> Result<Self, NequipLiteError> {
        if spec.scalar_activation != "silu" || spec.gate_activation != "silu" {
            return Err(invalid_model(
                "Nequip-lite gate currently supports only silu",
            ));
        }
        if spec.input_dim == 0 || spec.output_dim == 0 {
            return Err(invalid_model("gate dimensions must be nonzero"));
        }
        if !spec.scalar_activation_scale.is_finite() || !spec.gate_activation_scale.is_finite() {
            return Err(invalid_model("gate activation scales must be finite"));
        }
        let mut output_dim = spec.scalar_indices.len();
        for index in &spec.scalar_indices {
            if *index >= spec.input_dim {
                return Err(invalid_model("gate scalar index out of bounds"));
            }
        }
        let mut groups = Vec::with_capacity(spec.gated_groups.len());
        for group in spec.gated_groups {
            if group.gate_index >= spec.input_dim {
                return Err(invalid_model("gate index out of bounds"));
            }
            if group.output_start != output_dim {
                return Err(invalid_model("gated group output_start is not contiguous"));
            }
            for index in &group.component_indices {
                if *index >= spec.input_dim {
                    return Err(invalid_model("gated component index out of bounds"));
                }
            }
            output_dim += group.component_indices.len();
            groups.push(GateGroup {
                gate_index: group.gate_index,
                component_indices: group.component_indices,
                output_start: group.output_start,
            });
        }
        if output_dim != spec.output_dim {
            return Err(invalid_model(format!(
                "gate output dimension mismatch: declared {}, reconstructed {}",
                spec.output_dim, output_dim
            )));
        }
        Ok(Self {
            input_dim: spec.input_dim,
            output_dim: spec.output_dim,
            scalar_indices: spec.scalar_indices,
            scalar_activation_scale: spec.scalar_activation_scale,
            gate_activation_scale: spec.gate_activation_scale,
            gated_groups: groups,
        })
    }

    fn forward(&self, input: &[f64], out: &mut [f64]) -> Result<(), NequipLiteError> {
        if input.len() != self.input_dim || out.len() != self.output_dim {
            return Err(invalid_model("gate forward shape mismatch"));
        }
        for (out_index, input_index) in self.scalar_indices.iter().enumerate() {
            out[out_index] = self.scalar_activation_scale * silu(input[*input_index]);
        }
        for group in &self.gated_groups {
            let gate = self.gate_activation_scale * silu(input[group.gate_index]);
            for (offset, component_index) in group.component_indices.iter().enumerate() {
                out[group.output_start + offset] = gate * input[*component_index];
            }
        }
        Ok(())
    }

    fn vjp(
        &self,
        input: &[f64],
        grad_out: &[f64],
        grad_input: &mut [f64],
    ) -> Result<(), NequipLiteError> {
        if input.len() != self.input_dim
            || grad_out.len() != self.output_dim
            || grad_input.len() != self.input_dim
        {
            return Err(invalid_model("gate VJP shape mismatch"));
        }
        for (out_index, input_index) in self.scalar_indices.iter().enumerate() {
            grad_input[*input_index] += grad_out[out_index]
                * self.scalar_activation_scale
                * silu_derivative(input[*input_index]);
        }
        for group in &self.gated_groups {
            let gate = self.gate_activation_scale * silu(input[group.gate_index]);
            let mut grad_gate = 0.0;
            for (offset, component_index) in group.component_indices.iter().enumerate() {
                let bar_out = grad_out[group.output_start + offset];
                grad_input[*component_index] += bar_out * gate;
                grad_gate += bar_out * input[*component_index];
            }
            grad_input[group.gate_index] +=
                grad_gate * self.gate_activation_scale * silu_derivative(input[group.gate_index]);
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct GateGroup {
    gate_index: usize,
    component_indices: Vec<usize>,
    output_start: usize,
}

#[derive(Clone, Debug)]
struct EdgeInput {
    norm_length: f64,
    bessel: Vec<f64>,
    cutoff: f64,
    edge_embedding: Vec<f64>,
    edge_attr: Vec<f64>,
}

#[derive(Clone, Debug)]
struct RadialEmbedding {
    norm_length: f64,
    bessel: Vec<f64>,
    cutoff: f64,
    edge_embedding: Vec<f64>,
}

#[derive(Clone, Debug)]
struct LayerCache {
    input_node_count: usize,
    normed_features: Vec<Vec<f64>>,
    gate_input: Vec<Vec<f64>>,
    edge_weights: Vec<Vec<f64>>,
}

#[derive(Clone, Debug)]
struct ModelOutput {
    energy: f64,
    forces: Vec<Vec3>,
    stress: Option<Mat3>,
    virial: Option<Mat3>,
}

#[must_use]
fn symmetric_part(matrix: Mat3) -> Mat3 {
    let mut out = matrix;
    for row in 0..3 {
        for col in 0..row {
            let value = 0.5 * (matrix[row][col] + matrix[col][row]);
            out[row][col] = value;
            out[col][row] = value;
        }
    }
    out
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelSpec {
    schema_version: String,
    #[serde(rename = "name")]
    _name: String,
    type_names: Vec<String>,
    cutoff: f64,
    l_max: usize,
    num_bessels: usize,
    polynomial_cutoff_p: usize,
    edge_embedding_factor: f64,
    node_embedding: Option<Vec<Vec<f64>>>,
    node_embedding_tensor: Option<String>,
    layers: Vec<LayerSpec>,
    readout: DenseSpec,
    per_type_energy_scales: Option<Vec<f64>>,
    per_type_energy_scales_tensor: Option<String>,
    per_type_energy_shifts: Option<Vec<f64>>,
    per_type_energy_shifts_tensor: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LayerSpec {
    input_irreps: String,
    linear1: DenseSpec,
    avg_num_neighbors_norm: f64,
    edge_mlp: DenseSpec,
    tensor_product: TensorProductSpec,
    linear2: DenseSpec,
    gate: GateSpecJson,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DenseSpec {
    in_dim: usize,
    out_dim: usize,
    weight: Option<Vec<f64>>,
    weight_tensor: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GateSpecJson {
    input_dim: usize,
    output_dim: usize,
    scalar_indices: Vec<usize>,
    scalar_activation: String,
    scalar_activation_scale: f64,
    gate_activation: String,
    gate_activation_scale: f64,
    gated_groups: Vec<GateGroupJson>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GateGroupJson {
    gate_index: usize,
    component_indices: Vec<usize>,
    output_start: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorProductSpec {
    irreps_in1: String,
    irreps_in2: String,
    irreps_out: String,
    weight_numel: usize,
    instruction_in1_segment: Vec<usize>,
    instruction_in2_segment: Vec<usize>,
    instruction_out_segment: Vec<usize>,
    instruction_cg_offset: Vec<usize>,
    instruction_cg_len: Vec<usize>,
    instruction_weight_offset: Vec<usize>,
    instruction_path_weight: Vec<f64>,
    cg_i_m: Vec<usize>,
    cg_j_m: Vec<usize>,
    cg_k_m: Vec<usize>,
    cg_value: Vec<f64>,
}

fn tensor_product_from_spec(spec: TensorProductSpec) -> Result<TensorProductPlan, NequipLiteError> {
    let instruction_len = spec.instruction_in1_segment.len();
    for (name, len) in [
        (
            "instruction_in2_segment",
            spec.instruction_in2_segment.len(),
        ),
        (
            "instruction_out_segment",
            spec.instruction_out_segment.len(),
        ),
        ("instruction_cg_offset", spec.instruction_cg_offset.len()),
        ("instruction_cg_len", spec.instruction_cg_len.len()),
        (
            "instruction_weight_offset",
            spec.instruction_weight_offset.len(),
        ),
        (
            "instruction_path_weight",
            spec.instruction_path_weight.len(),
        ),
    ] {
        if len != instruction_len {
            return Err(invalid_model(format!(
                "{name} length {len} does not match instruction length {instruction_len}"
            )));
        }
    }
    let cg_len = spec.cg_value.len();
    for (name, len) in [
        ("cg_i_m", spec.cg_i_m.len()),
        ("cg_j_m", spec.cg_j_m.len()),
        ("cg_k_m", spec.cg_k_m.len()),
    ] {
        if len != cg_len {
            return Err(invalid_model(format!(
                "{name} length {len} does not match cg length {cg_len}"
            )));
        }
    }

    let instructions = (0..instruction_len)
        .map(|index| TensorProductInstruction {
            in1_segment: spec.instruction_in1_segment[index],
            in2_segment: spec.instruction_in2_segment[index],
            out_segment: spec.instruction_out_segment[index],
            cg_offset: spec.instruction_cg_offset[index],
            cg_len: spec.instruction_cg_len[index],
            weight_offset: Some(spec.instruction_weight_offset[index]),
            path_weight: spec.instruction_path_weight[index],
        })
        .collect::<Vec<_>>();
    let cg_entries = (0..cg_len)
        .map(|index| CgEntry {
            i_m: spec.cg_i_m[index],
            j_m: spec.cg_j_m[index],
            k_m: spec.cg_k_m[index],
            value: spec.cg_value[index],
        })
        .collect::<Vec<_>>();

    Ok(TensorProductPlan::new(
        spec.irreps_in1.parse::<Irreps>()?,
        spec.irreps_in2.parse::<Irreps>()?,
        spec.irreps_out.parse::<Irreps>()?,
        instructions,
        cg_entries,
        spec.weight_numel,
    )?)
}

fn checked_rectangular(name: &str, rows: &[Vec<f64>]) -> Result<usize, NequipLiteError> {
    let width = rows
        .first()
        .ok_or_else(|| invalid_model(format!("{name} must not be empty")))?
        .len();
    if width == 0 {
        return Err(invalid_model(format!("{name} row width must be nonzero")));
    }
    if rows.iter().any(|row| row.len() != width) {
        return Err(invalid_model(format!("{name} must be rectangular")));
    }
    if rows.iter().flatten().any(|value| !value.is_finite()) {
        return Err(invalid_model(format!("{name} values must be finite")));
    }
    Ok(width)
}

fn vector_from_embedded_or_tensor(
    name: &str,
    embedded: Option<Vec<f64>>,
    tensor_name: Option<String>,
    external_tensors: &HashMap<String, NequipLiteExternalTensor>,
) -> Result<Vec<f64>, NequipLiteError> {
    match (embedded, tensor_name) {
        (Some(values), None) => {
            if values.iter().any(|value| !value.is_finite()) {
                return Err(invalid_model(format!("{name} values must be finite")));
            }
            Ok(values)
        }
        (None, Some(tensor_name)) => {
            let tensor = external_tensors.get(&tensor_name).ok_or_else(|| {
                invalid_model(format!("{name} references missing tensor {tensor_name:?}"))
            })?;
            if tensor.shape.len() != 1 {
                return Err(invalid_model(format!(
                    "{name} tensor {tensor_name:?} must have rank 1, got shape {:?}",
                    tensor.shape
                )));
            }
            if tensor.values.iter().any(|value| !value.is_finite()) {
                return Err(invalid_model(format!(
                    "{name} tensor {tensor_name:?} values must be finite"
                )));
            }
            Ok(tensor.values.clone())
        }
        (Some(_), Some(tensor_name)) => Err(invalid_model(format!(
            "{name} must use either embedded values or tensor {tensor_name:?}, not both"
        ))),
        (None, None) => Err(invalid_model(format!(
            "{name} must provide embedded values or a tensor reference"
        ))),
    }
}

fn matrix_from_embedded_or_tensor(
    name: &str,
    embedded: Option<Vec<Vec<f64>>>,
    tensor_name: Option<String>,
    external_tensors: &HashMap<String, NequipLiteExternalTensor>,
) -> Result<Vec<Vec<f64>>, NequipLiteError> {
    match (embedded, tensor_name) {
        (Some(rows), None) => {
            checked_rectangular(name, &rows)?;
            Ok(rows)
        }
        (None, Some(tensor_name)) => {
            let tensor = external_tensors.get(&tensor_name).ok_or_else(|| {
                invalid_model(format!("{name} references missing tensor {tensor_name:?}"))
            })?;
            if tensor.shape.len() != 2 {
                return Err(invalid_model(format!(
                    "{name} tensor {tensor_name:?} must have rank 2, got shape {:?}",
                    tensor.shape
                )));
            }
            let rows = tensor.shape[0];
            let cols = tensor.shape[1];
            if rows == 0 || cols == 0 {
                return Err(invalid_model(format!(
                    "{name} tensor {tensor_name:?} dimensions must be nonzero"
                )));
            }
            if tensor.values.len() != rows * cols {
                return Err(invalid_model(format!(
                    "{name} tensor {tensor_name:?} value length {} does not match shape {:?}",
                    tensor.values.len(),
                    tensor.shape
                )));
            }
            if tensor.values.iter().any(|value| !value.is_finite()) {
                return Err(invalid_model(format!(
                    "{name} tensor {tensor_name:?} values must be finite"
                )));
            }
            Ok(tensor
                .values
                .chunks(cols)
                .map(<[f64]>::to_vec)
                .collect::<Vec<_>>())
        }
        (Some(_), Some(tensor_name)) => Err(invalid_model(format!(
            "{name} must use either embedded values or tensor {tensor_name:?}, not both"
        ))),
        (None, None) => Err(invalid_model(format!(
            "{name} must provide embedded values or a tensor reference"
        ))),
    }
}

fn dense_weight_from_embedded_or_tensor(
    name: &str,
    embedded: Option<Vec<f64>>,
    tensor_name: Option<String>,
    external_tensors: &HashMap<String, NequipLiteExternalTensor>,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f64>, NequipLiteError> {
    let expected_len = out_dim
        .checked_mul(in_dim)
        .ok_or_else(|| invalid_model("dense weight shape overflows"))?;
    match (embedded, tensor_name) {
        (Some(values), None) => {
            if values.iter().any(|value| !value.is_finite()) {
                return Err(invalid_model(format!("{name} values must be finite")));
            }
            Ok(values)
        }
        (None, Some(tensor_name)) => {
            let tensor = external_tensors.get(&tensor_name).ok_or_else(|| {
                invalid_model(format!("{name} references missing tensor {tensor_name:?}"))
            })?;
            match tensor.shape.as_slice() {
                [len] if *len == expected_len => {}
                [rows, cols] if *rows == out_dim && *cols == in_dim => {}
                [len] => {
                    return Err(invalid_model(format!(
                        "{name} tensor {tensor_name:?} rank-1 length {len} does not match expected {expected_len}"
                    )));
                }
                [rows, cols] => {
                    return Err(invalid_model(format!(
                        "{name} tensor {tensor_name:?} shape [{rows}, {cols}] does not match expected [{out_dim}, {in_dim}]"
                    )));
                }
                _ => {
                    return Err(invalid_model(format!(
                        "{name} tensor {tensor_name:?} must have rank 1 or rank 2, got shape {:?}",
                        tensor.shape
                    )));
                }
            }
            if tensor.values.len() != expected_len {
                return Err(invalid_model(format!(
                    "{name} tensor {tensor_name:?} value length {} does not match expected {expected_len}",
                    tensor.values.len()
                )));
            }
            if tensor.values.iter().any(|value| !value.is_finite()) {
                return Err(invalid_model(format!(
                    "{name} tensor {tensor_name:?} values must be finite"
                )));
            }
            Ok(tensor.values.clone())
        }
        (Some(_), Some(tensor_name)) => Err(invalid_model(format!(
            "{name} must use either embedded values or tensor {tensor_name:?}, not both"
        ))),
        (None, None) => Err(invalid_model(format!(
            "{name} must provide embedded values or a tensor reference"
        ))),
    }
}

fn validate_type_names(type_names: &[String]) -> Result<(), NequipLiteError> {
    let mut seen = HashSet::new();
    for type_name in type_names {
        if type_name.trim().is_empty() {
            return Err(invalid_model("type_names must not contain blank names"));
        }
        if !seen.insert(type_name.as_str()) {
            return Err(invalid_model(format!(
                "type_names contains duplicate entry {type_name:?}"
            )));
        }
    }
    Ok(())
}

fn polynomial_cutoff(x: f64, p: usize) -> f64 {
    if x >= 1.0 {
        return 0.0;
    }
    let p = p as f64;
    1.0 - ((p + 1.0) * (p + 2.0) / 2.0) * x.powf(p) + p * (p + 2.0) * x.powf(p + 1.0)
        - (p * (p + 1.0) / 2.0) * x.powf(p + 2.0)
}

fn polynomial_cutoff_derivative(x: f64, p: usize) -> f64 {
    if x >= 1.0 {
        return 0.0;
    }
    let p = p as f64;
    -((p + 1.0) * (p + 2.0) / 2.0) * p * x.powf(p - 1.0) + p * (p + 2.0) * (p + 1.0) * x.powf(p)
        - (p * (p + 1.0) / 2.0) * (p + 2.0) * x.powf(p + 1.0)
}

fn sinc_pi(x: f64) -> f64 {
    if x.abs() < 1.0e-8 {
        1.0 - PI * PI * x * x / 6.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

fn sinc_pi_derivative(x: f64) -> f64 {
    if x.abs() < 1.0e-8 {
        -PI * PI * x / 3.0
    } else {
        (PI * x * (PI * x).cos() - (PI * x).sin()) / (PI * x * x)
    }
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn silu_derivative(x: f64) -> f64 {
    let sigmoid = 1.0 / (1.0 + (-x).exp());
    sigmoid + x * sigmoid * (1.0 - sigmoid)
}

fn invalid_model(message: impl Into<String>) -> NequipLiteError {
    NequipLiteError::InvalidModel(message.into())
}
