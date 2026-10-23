//! Metadata gate for the pinned `allegro-oam-l-v1` runtime contract.
//!
//! This module validates the narrow Allegro-OAM-L state handoff and exported
//! tensor inventory, and implements the pinned runtime slice.

use std::collections::{BTreeSet, HashMap};

use e3nn_core::{radial::distance_vjp_accumulate, SphericalHarmonicsPlan};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        target_arch = "x86_64"
    ),
    all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
))]
use crate::blas_backend;
use crate::neighbor::cell_list_multi_image_neighbor_list;
use crate::{
    all_pairs_neighbor_list,
    force_accum::{
        accumulate_edge_displacement_gradient, scatter_edge_vector_vjp,
        stress_from_displacement_gradient, virial_from_displacement_gradient, ForceAccumError,
        Mat3,
    },
    geometry::Cell,
    NeighborListError, Prediction, PredictionOptions, Structure,
};

const STATE_SCHEMA_VERSION: &str = "allegro-oam-state-export-v1";
const EXECUTION_CONTRACT_SCHEMA_VERSION: &str = "allegro-oam-execution-contract-v1";
const SELECTED_MODEL: &str = "mir-group/Allegro-OAM-L:0.1";
const ARTIFACT_SHA256: &str = "3f0d3ca7bb136d4c2ee76278170fcbe756436f037e16c673a86e2c57d271c64e";
const ARTIFACT_SIZE_BYTES: u64 = 80_738_705;
const EXPECTED_CUTOFF: f64 = 7.0;
const EXPECTED_L_MAX: usize = 3;
const EXPECTED_NUM_TYPES: usize = 89;
const EXPECTED_NUM_LAYERS: usize = 4;
const EXPECTED_STATE_TENSOR_COUNT: usize = 34;
const EXPECTED_STATE_TENSOR_NUMEL: usize = 10_005_951;
const EXPECTED_CONTRACTER_COUNT: usize = 4;
const EXPECTED_WEIGHTED_CHANNEL_COUNT: usize = 2;
const EXPECTED_SCALAR_MLP_COUNT: usize = 9;
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        target_arch = "x86_64"
    ),
    all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
))]
const ALLEGRO_BLAS_ROW_THRESHOLD: usize = 16;
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        target_arch = "x86_64"
    ),
    all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
))]
const ALLEGRO_BLAS_OUT_DIM_THRESHOLD: usize = 64;
#[cfg(feature = "parallel")]
const ALLEGRO_PARALLEL_ELEMENTWISE_THRESHOLD: usize = 16_384;

#[derive(Debug, Error)]
pub enum AllegroOamError {
    #[error("invalid Allegro-OAM-L model: {0}")]
    InvalidModel(String),

    #[error(transparent)]
    NeighborList(#[from] NeighborListError),

    #[error(transparent)]
    ForceAccum(#[from] ForceAccumError),

    #[error(transparent)]
    Core(#[from] e3nn_core::CoreError),

    #[error("Allegro-OAM-L runtime request is not supported by the pinned slice")]
    RuntimeNotImplemented,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AllegroWeightedChannelsShape {
    pub rows: usize,
    pub multiplicity_out: usize,
    pub num_irreps: usize,
    pub base_dim: usize,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct AllegroWeightedChannelsVjp {
    pub edge_attrs: Vec<f64>,
    pub weights: Vec<f64>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AllegroLinearShape {
    pub rows: usize,
    pub in_dim: usize,
    pub out_dim: usize,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum AllegroScalarMlpLayer<'a> {
    Linear {
        weight: &'a [f64],
        in_dim: usize,
        out_dim: usize,
        alpha: f64,
    },
    Silu,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AllegroProductTypeEmbeddingShape {
    pub rows: usize,
    pub num_types: usize,
    pub radial_dim: usize,
    pub type_dim: usize,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AllegroProductTypeEmbeddingInputs<'a> {
    pub radial_embedding: &'a [f64],
    pub center_types: &'a [usize],
    pub neighbor_types: &'a [usize],
    pub center_embed: &'a [f64],
    pub neighbor_embed: &'a [f64],
    pub basis_weight: &'a [f64],
    pub basis_alpha: f64,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct AllegroProductTypeEmbeddingVjp {
    pub radial_embedding: Vec<f64>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AllegroRadialEmbeddingShape {
    pub rows: usize,
    pub num_bessels: usize,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AllegroContracterMode {
    Full,
    Diagonal,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AllegroContracterPrecontractedShape {
    pub rows: usize,
    pub multiplicity: usize,
    pub base_dim1: usize,
    pub base_dim2: usize,
    pub base_dim_out: usize,
    pub mode: AllegroContracterMode,
}

#[derive(Clone, Debug)]
pub struct AllegroOamExternalTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct AllegroOamPreparedGraph {
    atom_count: usize,
    pbc: [bool; 3],
    cell: Option<Cell>,
    atom_types: Vec<usize>,
    edge_sources: Vec<usize>,
    edge_dests: Vec<usize>,
    edge_vectors: Vec<e3nn_core::Vec3>,
    center_types: Vec<usize>,
    neighbor_types: Vec<usize>,
    normed_lengths: Vec<f64>,
}

impl AllegroOamPreparedGraph {
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.atom_count
    }

    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edge_sources.len()
    }

    #[must_use]
    pub fn pbc(&self) -> [bool; 3] {
        self.pbc
    }
}

#[derive(Clone, Debug)]
pub struct AllegroOamModel {
    selected_model: String,
    type_names: Vec<String>,
    cutoff: f64,
    l_max: usize,
    num_layers: usize,
    state_tensor_count: usize,
    state_tensor_numel: usize,
    loaded_tensor_count: usize,
    loaded_tensor_numel: usize,
    artifact_sha256: String,
    artifact_size_bytes: u64,
    execution_contract_status: String,
    execution_plan: Option<Value>,
    tensors: HashMap<String, AllegroOamExternalTensor>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct AllegroForwardTrace {
    total_energy: f64,
    atom_types: Vec<usize>,
    edge_sources: Vec<usize>,
    edge_dests: Vec<usize>,
    edge_vectors: Vec<e3nn_core::Vec3>,
    center_types: Vec<usize>,
    neighbor_types: Vec<usize>,
    normed_lengths: Vec<f64>,
    edge_attrs: Vec<f64>,
    radial: Vec<f64>,
    type_embedding: Vec<f64>,
    edge_embedding: Vec<f64>,
    edge_embedding_mlp_trace: Option<AllegroScalarMlpForwardTrace>,
    tensor_embed_weights: Vec<f64>,
    first_projection: Vec<f64>,
    scalar_blocks: Vec<Vec<f64>>,
    layers: Vec<AllegroLayerForward>,
    edge_features: Vec<f64>,
    edge_energy: Vec<f64>,
    edge_readout_mlp_trace: Option<AllegroScalarMlpForwardTrace>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct AllegroLayerForward {
    tensor_features_in: Vec<f64>,
    env_weights: Vec<f64>,
    gathered_env: Vec<f64>,
    contracted_len: usize,
    scalar_tp_len: usize,
    latent_input: Vec<f64>,
    latent_output_len: usize,
    latent_mlp_trace: Option<AllegroScalarMlpForwardTrace>,
}

#[derive(Clone, Debug)]
struct AllegroVjp {
    forces: Vec<e3nn_core::Vec3>,
    displacement_gradient: Mat3,
}

#[derive(Clone, Debug)]
struct AllegroScalarMlpSiluTrace {
    dim: usize,
    derivatives: Vec<f64>,
}

#[derive(Clone, Debug)]
struct AllegroScalarMlpForwardTrace {
    output_dim: usize,
    silu_inputs: Vec<AllegroScalarMlpSiluTrace>,
}

impl AllegroOamModel {
    pub fn from_json_str_with_external_tensors(
        text: &str,
        external_tensors: HashMap<String, AllegroOamExternalTensor>,
    ) -> Result<Self, AllegroOamError> {
        let spec: StateSpec = serde_json::from_str(text)
            .map_err(|err| invalid_model(format!("invalid Allegro-OAM-L JSON: {err}")))?;
        Self::from_spec(spec, external_tensors)
    }

    pub fn predict(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
    ) -> Result<Prediction, AllegroOamError> {
        if !options.compute_energy
            && !options.compute_forces
            && !options.compute_stress
            && !options.compute_virial
        {
            return Ok(Prediction::default());
        }
        let has_pbc = structure.pbc.iter().any(|enabled| *enabled);
        let compute_displacement_gradient = options.compute_stress || options.compute_virial;
        if has_pbc && structure.pbc != [true, true, true] {
            return Err(AllegroOamError::RuntimeNotImplemented);
        }
        if compute_displacement_gradient && structure.pbc != [true, true, true] {
            return Err(AllegroOamError::RuntimeNotImplemented);
        }
        let prepared = self.prepare_graph(structure)?;
        self.predict_prepared(&prepared, options)
    }

    pub fn predict_prepared(
        &self,
        prepared: &AllegroOamPreparedGraph,
        options: &PredictionOptions,
    ) -> Result<Prediction, AllegroOamError> {
        if !options.compute_energy
            && !options.compute_forces
            && !options.compute_stress
            && !options.compute_virial
        {
            return Ok(Prediction::default());
        }
        let compute_displacement_gradient = options.compute_stress || options.compute_virial;
        if compute_displacement_gradient && prepared.pbc != [true, true, true] {
            return Err(AllegroOamError::RuntimeNotImplemented);
        }
        let capture_mlp_trace = options.compute_forces || compute_displacement_gradient;
        let forward = self.predict_forward_trace(prepared, capture_mlp_trace)?;
        let vjp = if options.compute_forces || compute_displacement_gradient {
            Some(self.predict_vjp_from_forward(&forward, compute_displacement_gradient)?)
        } else {
            None
        };
        let displacement_gradient = if compute_displacement_gradient {
            Some(symmetric_part(
                vjp.as_ref()
                    .ok_or(AllegroOamError::RuntimeNotImplemented)?
                    .displacement_gradient,
            ))
        } else {
            None
        };
        let stress = if options.compute_stress {
            let cell = prepared
                .cell
                .as_ref()
                .ok_or(AllegroOamError::RuntimeNotImplemented)?;
            Some(stress_from_displacement_gradient(
                displacement_gradient.ok_or(AllegroOamError::RuntimeNotImplemented)?,
                cell,
            )?)
        } else {
            None
        };
        let virial = if options.compute_virial {
            Some(virial_from_displacement_gradient(
                displacement_gradient.ok_or(AllegroOamError::RuntimeNotImplemented)?,
            ))
        } else {
            None
        };
        Ok(Prediction {
            energy: options.compute_energy.then_some(forward.total_energy),
            forces: if options.compute_forces {
                vjp.map(|value| value.forces)
            } else {
                None
            },
            stress,
            virial,
        })
    }

    pub fn prepare_graph(
        &self,
        structure: &Structure,
    ) -> Result<AllegroOamPreparedGraph, AllegroOamError> {
        let has_pbc = structure.pbc.iter().any(|enabled| *enabled);
        if has_pbc && structure.pbc != [true, true, true] {
            return Err(AllegroOamError::RuntimeNotImplemented);
        }
        let neighbor_list = if has_pbc {
            cell_list_multi_image_neighbor_list(structure, self.cutoff)?
        } else {
            all_pairs_neighbor_list(structure, self.cutoff)?
        };
        let edge_count = neighbor_list.edges.len();
        if edge_count == 0 {
            return Err(invalid_model(
                "Allegro-OAM-L energy runtime requires at least one directed edge",
            ));
        }
        let atom_types = self.atom_types(structure)?;
        let edge_sources = neighbor_list
            .edges
            .iter()
            .map(|edge| edge.src)
            .collect::<Vec<_>>();
        let edge_dests = neighbor_list
            .edges
            .iter()
            .map(|edge| edge.dst)
            .collect::<Vec<_>>();
        let edge_vectors = neighbor_list
            .edges
            .iter()
            .map(|edge| edge.vector)
            .collect::<Vec<_>>();
        let center_types = neighbor_list
            .edges
            .iter()
            .map(|edge| atom_types[edge.src])
            .collect::<Vec<_>>();
        let neighbor_types = neighbor_list
            .edges
            .iter()
            .map(|edge| atom_types[edge.dst])
            .collect::<Vec<_>>();
        let rmax_recip = self.tensor_values("model.func.edge_norm._rmax_recip", &[])?[0];
        let normed_lengths = neighbor_list
            .edges
            .iter()
            .map(|edge| edge.distance * rmax_recip)
            .collect::<Vec<_>>();
        Ok(AllegroOamPreparedGraph {
            atom_count: structure.len(),
            pbc: structure.pbc,
            cell: structure.cell,
            atom_types,
            edge_sources,
            edge_dests,
            edge_vectors,
            center_types,
            neighbor_types,
            normed_lengths,
        })
    }

    #[must_use]
    pub fn selected_model(&self) -> &str {
        &self.selected_model
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
    pub fn l_max(&self) -> usize {
        self.l_max
    }

    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    #[must_use]
    pub fn state_tensor_count(&self) -> usize {
        self.state_tensor_count
    }

    #[must_use]
    pub fn state_tensor_numel(&self) -> usize {
        self.state_tensor_numel
    }

    #[must_use]
    pub fn loaded_tensor_count(&self) -> usize {
        self.loaded_tensor_count
    }

    #[must_use]
    pub fn loaded_tensor_numel(&self) -> usize {
        self.loaded_tensor_numel
    }

    #[must_use]
    pub fn tensors(&self) -> &HashMap<String, AllegroOamExternalTensor> {
        &self.tensors
    }

    #[must_use]
    pub fn tensor_by_name(&self, name: &str) -> Option<&AllegroOamExternalTensor> {
        self.tensors.get(name)
    }

    #[must_use]
    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    #[must_use]
    pub fn artifact_size_bytes(&self) -> u64 {
        self.artifact_size_bytes
    }

    #[must_use]
    pub fn execution_contract_status(&self) -> &str {
        &self.execution_contract_status
    }

    #[must_use]
    pub fn execution_plan(&self) -> Option<&Value> {
        self.execution_plan.as_ref()
    }

    pub fn runtime_required_tensor_names(&self) -> Result<Vec<String>, AllegroOamError> {
        let plan = self
            .execution_plan
            .as_ref()
            .ok_or_else(|| invalid_model("Allegro execution plan is missing"))?;
        runtime_required_tensor_names_from_plan(plan)
    }

    pub fn validate_runtime_tensor_handoff(&self) -> Result<(), AllegroOamError> {
        for name in self.runtime_required_tensor_names()? {
            if !self.tensors.contains_key(&name) {
                return Err(invalid_model(format!(
                    "runtime tensor required by Allegro plan is missing: {name}"
                )));
            }
        }
        Ok(())
    }

    fn predict_forward_trace(
        &self,
        prepared: &AllegroOamPreparedGraph,
        capture_mlp_trace: bool,
    ) -> Result<AllegroForwardTrace, AllegroOamError> {
        let edge_count = prepared.edge_sources.len();
        if edge_count == 0 {
            return Err(invalid_model(
                "Allegro-OAM-L energy runtime requires at least one directed edge",
            ));
        }
        let radial = allegro_radial_bessel_embedding_forward(
            &prepared.normed_lengths,
            self.tensor_values(
                "model.func.radial_chemical_embed.bessel_encode.bessel_weights",
                &[1, 12],
            )?,
            8.0,
            AllegroRadialEmbeddingShape {
                rows: edge_count,
                num_bessels: 12,
            },
        )?;

        let mut edge_attrs = vec![0.0; edge_count * 16];
        let sh_plan = SphericalHarmonicsPlan::component(3, true)?;
        for (edge_index, edge_vector) in prepared.edge_vectors.iter().copied().enumerate() {
            sh_plan.forward(
                edge_vector,
                &mut edge_attrs[edge_index * 16..(edge_index + 1) * 16],
            )?;
        }

        let type_embedding = allegro_product_type_embedding_forward(
            AllegroProductTypeEmbeddingInputs {
                radial_embedding: &radial,
                center_types: &prepared.center_types,
                neighbor_types: &prepared.neighbor_types,
                center_embed: self.tensor_values(
                    "model.func.radial_chemical_embed.type_embed.center_embed.weight",
                    &[EXPECTED_NUM_TYPES, 256],
                )?,
                neighbor_embed: self.tensor_values(
                    "model.func.radial_chemical_embed.type_embed.neighbor_embed.weight",
                    &[EXPECTED_NUM_TYPES, 256],
                )?,
                basis_weight: self.tensor_values(
                    "model.func.radial_chemical_embed.type_embed.basis_linear.mlp.0.weight",
                    &[12, 512],
                )?,
                basis_alpha: 0.288_675_129_413_604_74,
            },
            AllegroProductTypeEmbeddingShape {
                rows: edge_count,
                num_types: EXPECTED_NUM_TYPES,
                radial_dim: 12,
                type_dim: 256,
            },
        )?;
        let (edge_embedding, edge_embedding_mlp_trace) = self.run_scalar_mlp_maybe_traced(
            &type_embedding,
            edge_count,
            512,
            &[
                (
                    "model.func.scalar_embed_mlp.mlp_module.mlp.0.weight",
                    512,
                    512,
                    0.044_194_173_067_808_15,
                ),
                (
                    "model.func.scalar_embed_mlp.mlp_module.mlp.2.weight",
                    512,
                    256,
                    0.0625,
                ),
            ],
            capture_mlp_trace,
        )?;

        let tensor_embed_weights = self.run_scalar_mlp(
            &edge_embedding,
            edge_count,
            256,
            &[(
                "model.func.tensor_embed.env_embed_linear.mlp.0.weight",
                256,
                256,
                0.0625,
            )],
        )?;
        let tensor_embed_rtoi =
            self.tensor_values("model.func.tensor_embed._edge_weighter._rtoi", &[4, 16])?;
        let mut tensor_features = allegro_make_weighted_channels_forward(
            &edge_attrs,
            &tensor_embed_weights,
            tensor_embed_rtoi,
            AllegroWeightedChannelsShape {
                rows: edge_count,
                multiplicity_out: 64,
                num_irreps: 4,
                base_dim: 16,
            },
        )?;

        let first_projection = self.run_scalar_mlp(
            &edge_embedding,
            edge_count,
            256,
            &[(
                "model.func.allegro.first_layer_env_embed_projection.mlp.0.weight",
                256,
                512,
                0.0625,
            )],
        )?;
        let mut scalar_blocks = vec![select_columns(&first_projection, edge_count, 512, 0, 256)?];
        let mut env_weights = select_columns(&first_projection, edge_count, 512, 256, 256)?;
        let env_rtoi = self.tensor_values("model.func.allegro._env_weighter._rtoi", &[4, 16])?;
        let mut layers = Vec::with_capacity(EXPECTED_NUM_LAYERS);

        for layer_index in 0..EXPECTED_NUM_LAYERS {
            let tensor_features_in = tensor_features.clone();
            let layer_env_weights = env_weights.clone();
            let env_edge_features = allegro_make_weighted_channels_forward(
                &edge_attrs,
                &env_weights,
                env_rtoi,
                AllegroWeightedChannelsShape {
                    rows: edge_count,
                    multiplicity_out: 64,
                    num_irreps: 4,
                    base_dim: 16,
                },
            )?;
            let gathered_env = allegro_scatter_center_gather_edge_features_forward(
                &env_edge_features,
                &prepared.edge_sources,
                prepared.atom_count,
                64,
                16,
                0.010_007_007_358_585_517,
            )?;
            let mode = if layer_index == 3 {
                AllegroContracterMode::Diagonal
            } else {
                AllegroContracterMode::Full
            };
            let base_dim_out = if layer_index == 3 { 1 } else { 16 };
            let contracted = allegro_contracter_forward_precontracted(
                &tensor_features,
                &gathered_env,
                self.tensor_values(
                    &format!("model.func.allegro.tps.{layer_index}.precontracted_ww3j"),
                    if layer_index == 3 {
                        &[64, 16, 1]
                    } else {
                        &[64, 16, 16, 16]
                    },
                )?,
                AllegroContracterPrecontractedShape {
                    rows: edge_count,
                    multiplicity: 64,
                    base_dim1: 16,
                    base_dim2: 16,
                    base_dim_out,
                    mode,
                },
            )?;
            let scalar_tp = first_tensor_component(&contracted, edge_count, 64, base_dim_out)?;
            let latent_input =
                concat_scalar_blocks_with_tp(&scalar_blocks, &scalar_tp, edge_count)?;
            let (latent_output, latent_mlp_trace) = self.run_scalar_mlp_maybe_traced(
                &latent_input,
                edge_count,
                latent_input.len() / edge_count,
                latent_specs(layer_index)?,
                capture_mlp_trace,
            )?;
            layers.push(AllegroLayerForward {
                tensor_features_in,
                env_weights: layer_env_weights,
                gathered_env,
                contracted_len: contracted.len(),
                scalar_tp_len: scalar_tp.len(),
                latent_input,
                latent_output_len: latent_output.len(),
                latent_mlp_trace,
            });
            let scalar_out_dim = 256;
            scalar_blocks.push(select_columns(
                &latent_output,
                edge_count,
                latent_output.len() / edge_count,
                0,
                scalar_out_dim,
            )?);
            if layer_index < 3 {
                env_weights = select_columns(
                    &latent_output,
                    edge_count,
                    latent_output.len() / edge_count,
                    scalar_out_dim,
                    256,
                )?;
            }
            tensor_features = contracted;
        }

        let edge_features = concat_scalar_blocks(&scalar_blocks, edge_count)?;
        let (edge_energy, edge_readout_mlp_trace) = self.run_scalar_mlp_maybe_traced(
            &edge_features,
            edge_count,
            1280,
            &[
                (
                    "model.func.edge_readout.mlp_module.mlp.0.weight",
                    1280,
                    128,
                    0.027_950_849_384_069_443,
                ),
                (
                    "model.func.edge_readout.mlp_module.mlp.2.weight",
                    128,
                    1,
                    0.125,
                ),
            ],
            capture_mlp_trace,
        )?;
        let (_, total_energy) = allegro_edgewise_reduce_scale_shift_sum_forward(
            &edge_energy,
            &prepared.edge_sources,
            &prepared.atom_types,
            self.tensor_values("model.func.per_type_energy_scale_shift.scales", &[89, 1])?,
            self.tensor_values("model.func.per_type_energy_scale_shift.shifts", &[89, 1])?,
            0.007_076_022_762_639_500_6,
            prepared.atom_count,
        )?;
        Ok(AllegroForwardTrace {
            total_energy,
            atom_types: prepared.atom_types.clone(),
            edge_sources: prepared.edge_sources.clone(),
            edge_dests: prepared.edge_dests.clone(),
            edge_vectors: prepared.edge_vectors.clone(),
            center_types: prepared.center_types.clone(),
            neighbor_types: prepared.neighbor_types.clone(),
            normed_lengths: prepared.normed_lengths.clone(),
            edge_attrs,
            radial,
            type_embedding,
            edge_embedding,
            edge_embedding_mlp_trace,
            tensor_embed_weights,
            first_projection,
            scalar_blocks,
            layers,
            edge_features,
            edge_energy,
            edge_readout_mlp_trace,
        })
    }

    fn predict_vjp_from_forward(
        &self,
        forward: &AllegroForwardTrace,
        compute_displacement_gradient: bool,
    ) -> Result<AllegroVjp, AllegroOamError> {
        self.validate_forward_trace(forward)?;
        let edge_count = forward.edge_sources.len();
        let num_atoms = forward.atom_types.len();
        let mut grad_edge_attrs =
            vec![0.0; checked_len("Allegro grad edge attrs", &[edge_count, 16])?];

        let grad_edge_energy = allegro_edgewise_reduce_scale_shift_sum_vjp_edge_energy(
            &forward.edge_sources,
            &forward.atom_types,
            self.tensor_values("model.func.per_type_energy_scale_shift.scales", &[89, 1])?,
            0.007_076_022_762_639_500_6,
            num_atoms,
            1.0,
        )?;
        let grad_edge_features = self.run_scalar_mlp_vjp_maybe_traced(
            &forward.edge_features,
            edge_count,
            1280,
            &[
                (
                    "model.func.edge_readout.mlp_module.mlp.0.weight",
                    1280,
                    128,
                    0.027_950_849_384_069_443,
                ),
                (
                    "model.func.edge_readout.mlp_module.mlp.2.weight",
                    128,
                    1,
                    0.125,
                ),
            ],
            forward.edge_readout_mlp_trace.as_ref(),
            &grad_edge_energy,
        )?;
        let mut grad_scalar_blocks =
            concat_blocks_vjp(&grad_edge_features, edge_count, &[256, 256, 256, 256, 256])?;
        let mut grad_env_weights_by_layer =
            vec![
                vec![0.0; checked_len("Allegro grad env weights", &[edge_count, 256])?];
                EXPECTED_NUM_LAYERS
            ];
        let mut grad_tensor_features = vec![
            0.0;
            forward
                .layers
                .last()
                .ok_or_else(|| invalid_model("Allegro forward trace has no layers"))?
                .contracted_len
        ];
        let env_rtoi = self.tensor_values("model.func.allegro._env_weighter._rtoi", &[4, 16])?;

        for layer_index in (0..EXPECTED_NUM_LAYERS).rev() {
            let layer = &forward.layers[layer_index];
            let latent_output_dim = layer.latent_output_len / edge_count;
            let mut grad_latent_output = vec![0.0; layer.latent_output_len];
            select_columns_vjp_add(
                &grad_scalar_blocks[layer_index + 1],
                edge_count,
                latent_output_dim,
                0,
                256,
                &mut grad_latent_output,
            )?;
            if layer_index < 3 {
                select_columns_vjp_add(
                    &grad_env_weights_by_layer[layer_index + 1],
                    edge_count,
                    latent_output_dim,
                    256,
                    256,
                    &mut grad_latent_output,
                )?;
            }

            let grad_latent_input = self.run_scalar_mlp_vjp_maybe_traced(
                &layer.latent_input,
                edge_count,
                layer.latent_input.len() / edge_count,
                latent_specs(layer_index)?,
                layer.latent_mlp_trace.as_ref(),
                &grad_latent_output,
            )?;
            let mut widths = vec![256; layer_index + 1];
            widths.push(64);
            let mut grad_scalar_tp =
                vec![0.0; checked_len("Allegro grad scalar TP", &[edge_count, 64])?];
            {
                let mut grad_parts = grad_scalar_blocks[..=layer_index]
                    .iter_mut()
                    .map(Vec::as_mut_slice)
                    .collect::<Vec<_>>();
                grad_parts.push(&mut grad_scalar_tp);
                concat_blocks_vjp_add(&grad_latent_input, edge_count, &widths, &mut grad_parts)?;
            }

            let mode = if layer_index == 3 {
                AllegroContracterMode::Diagonal
            } else {
                AllegroContracterMode::Full
            };
            let base_dim_out = if layer_index == 3 { 1 } else { 16 };
            let mut grad_contracted = grad_tensor_features;
            first_tensor_component_vjp_add(
                &grad_scalar_tp,
                edge_count,
                64,
                base_dim_out,
                &mut grad_contracted,
            )?;
            let (grad_tensor_features_in, grad_gathered_env) =
                allegro_contracter_vjp_inputs_precontracted(
                    &layer.tensor_features_in,
                    &layer.gathered_env,
                    self.tensor_values(
                        &format!("model.func.allegro.tps.{layer_index}.precontracted_ww3j"),
                        if layer_index == 3 {
                            &[64, 16, 1]
                        } else {
                            &[64, 16, 16, 16]
                        },
                    )?,
                    &grad_contracted,
                    AllegroContracterPrecontractedShape {
                        rows: edge_count,
                        multiplicity: 64,
                        base_dim1: 16,
                        base_dim2: 16,
                        base_dim_out,
                        mode,
                    },
                )?;
            grad_tensor_features = grad_tensor_features_in;
            let grad_env_edge_features = allegro_scatter_center_gather_edge_features_vjp(
                &forward.edge_sources,
                num_atoms,
                64,
                16,
                0.010_007_007_358_585_517,
                &grad_gathered_env,
            )?;
            let env_vjp = allegro_make_weighted_channels_vjp_inputs(
                &forward.edge_attrs,
                &layer.env_weights,
                env_rtoi,
                &grad_env_edge_features,
                AllegroWeightedChannelsShape {
                    rows: edge_count,
                    multiplicity_out: 64,
                    num_irreps: 4,
                    base_dim: 16,
                },
            )?;
            add_assign(&mut grad_edge_attrs, &env_vjp.edge_attrs)?;
            add_assign(
                &mut grad_env_weights_by_layer[layer_index],
                &env_vjp.weights,
            )?;
        }

        let tensor_embed_rtoi =
            self.tensor_values("model.func.tensor_embed._edge_weighter._rtoi", &[4, 16])?;
        let tensor_embed_vjp = allegro_make_weighted_channels_vjp_inputs(
            &forward.edge_attrs,
            &forward.tensor_embed_weights,
            tensor_embed_rtoi,
            &grad_tensor_features,
            AllegroWeightedChannelsShape {
                rows: edge_count,
                multiplicity_out: 64,
                num_irreps: 4,
                base_dim: 16,
            },
        )?;
        add_assign(&mut grad_edge_attrs, &tensor_embed_vjp.edge_attrs)?;
        let mut grad_edge_embedding = self.run_scalar_mlp_vjp(
            &forward.edge_embedding,
            edge_count,
            256,
            &[(
                "model.func.tensor_embed.env_embed_linear.mlp.0.weight",
                256,
                256,
                0.0625,
            )],
            &tensor_embed_vjp.weights,
        )?;

        let mut grad_first_projection = vec![0.0; forward.first_projection.len()];
        select_columns_vjp_add(
            &grad_scalar_blocks[0],
            edge_count,
            512,
            0,
            256,
            &mut grad_first_projection,
        )?;
        select_columns_vjp_add(
            &grad_env_weights_by_layer[0],
            edge_count,
            512,
            256,
            256,
            &mut grad_first_projection,
        )?;
        let grad_edge_embedding_from_projection = self.run_scalar_mlp_vjp(
            &forward.edge_embedding,
            edge_count,
            256,
            &[(
                "model.func.allegro.first_layer_env_embed_projection.mlp.0.weight",
                256,
                512,
                0.0625,
            )],
            &grad_first_projection,
        )?;
        add_assign(
            &mut grad_edge_embedding,
            &grad_edge_embedding_from_projection,
        )?;
        let grad_type_embedding = self.run_scalar_mlp_vjp_maybe_traced(
            &forward.type_embedding,
            edge_count,
            512,
            &[
                (
                    "model.func.scalar_embed_mlp.mlp_module.mlp.0.weight",
                    512,
                    512,
                    0.044_194_173_067_808_15,
                ),
                (
                    "model.func.scalar_embed_mlp.mlp_module.mlp.2.weight",
                    512,
                    256,
                    0.0625,
                ),
            ],
            forward.edge_embedding_mlp_trace.as_ref(),
            &grad_edge_embedding,
        )?;
        let product_vjp = allegro_product_type_embedding_vjp_inputs(
            AllegroProductTypeEmbeddingInputs {
                radial_embedding: &forward.radial,
                center_types: &forward.center_types,
                neighbor_types: &forward.neighbor_types,
                center_embed: self.tensor_values(
                    "model.func.radial_chemical_embed.type_embed.center_embed.weight",
                    &[EXPECTED_NUM_TYPES, 256],
                )?,
                neighbor_embed: self.tensor_values(
                    "model.func.radial_chemical_embed.type_embed.neighbor_embed.weight",
                    &[EXPECTED_NUM_TYPES, 256],
                )?,
                basis_weight: self.tensor_values(
                    "model.func.radial_chemical_embed.type_embed.basis_linear.mlp.0.weight",
                    &[12, 512],
                )?,
                basis_alpha: 0.288_675_129_413_604_74,
            },
            &grad_type_embedding,
            AllegroProductTypeEmbeddingShape {
                rows: edge_count,
                num_types: EXPECTED_NUM_TYPES,
                radial_dim: 12,
                type_dim: 256,
            },
        )?;
        let grad_normed_lengths = allegro_radial_bessel_embedding_vjp_normed_lengths(
            &forward.normed_lengths,
            self.tensor_values(
                "model.func.radial_chemical_embed.bessel_encode.bessel_weights",
                &[1, 12],
            )?,
            8.0,
            &product_vjp.radial_embedding,
            AllegroRadialEmbeddingShape {
                rows: edge_count,
                num_bessels: 12,
            },
        )?;

        let rmax_recip = self.tensor_values("model.func.edge_norm._rmax_recip", &[])?[0];
        let sh_plan = SphericalHarmonicsPlan::component(3, true)?;
        let mut forces = vec![[0.0; 3]; num_atoms];
        let mut displacement_gradient = [[0.0; 3]; 3];
        for edge_index in 0..edge_count {
            let mut bar_r = [0.0; 3];
            sh_plan.vjp_edge_vector(
                forward.edge_vectors[edge_index],
                &grad_edge_attrs[edge_index * 16..(edge_index + 1) * 16],
                &mut bar_r,
            )?;
            distance_vjp_accumulate(
                forward.edge_vectors[edge_index],
                grad_normed_lengths[edge_index] * rmax_recip,
                1.0e-12,
                &mut bar_r,
            )?;
            scatter_edge_vector_vjp(
                forward.edge_sources[edge_index],
                forward.edge_dests[edge_index],
                bar_r,
                &mut forces,
            )?;
            if compute_displacement_gradient {
                accumulate_edge_displacement_gradient(
                    forward.edge_vectors[edge_index],
                    bar_r,
                    &mut displacement_gradient,
                );
            }
        }
        Ok(AllegroVjp {
            forces,
            displacement_gradient,
        })
    }

    fn validate_forward_trace(&self, forward: &AllegroForwardTrace) -> Result<(), AllegroOamError> {
        let edge_count = forward.edge_sources.len();
        if edge_count == 0 {
            return Err(invalid_model("Allegro forward trace has no directed edges"));
        }
        let atom_count = forward.atom_types.len();
        if atom_count == 0 {
            return Err(invalid_model("Allegro forward trace has no atoms"));
        }
        expect_len(
            "Allegro trace edge_dests",
            forward.edge_dests.len(),
            edge_count,
        )?;
        expect_len(
            "Allegro trace edge_vectors",
            forward.edge_vectors.len(),
            edge_count,
        )?;
        expect_len(
            "Allegro trace center_types",
            forward.center_types.len(),
            edge_count,
        )?;
        expect_len(
            "Allegro trace neighbor_types",
            forward.neighbor_types.len(),
            edge_count,
        )?;
        expect_len(
            "Allegro trace normed_lengths",
            forward.normed_lengths.len(),
            edge_count,
        )?;
        for (edge_index, (&source, &dest)) in forward
            .edge_sources
            .iter()
            .zip(forward.edge_dests.iter())
            .enumerate()
        {
            if source >= atom_count || dest >= atom_count {
                return Err(invalid_model(format!(
                    "Allegro trace edge {edge_index} references atoms {source}->{dest}, but atom count is {atom_count}"
                )));
            }
        }
        expect_len(
            "Allegro trace edge_attrs",
            forward.edge_attrs.len(),
            checked_len("Allegro trace edge_attrs", &[edge_count, 16])?,
        )?;
        expect_len(
            "Allegro trace radial",
            forward.radial.len(),
            checked_len("Allegro trace radial", &[edge_count, 12])?,
        )?;
        expect_len(
            "Allegro trace type_embedding",
            forward.type_embedding.len(),
            checked_len("Allegro trace type_embedding", &[edge_count, 512])?,
        )?;
        expect_len(
            "Allegro trace edge_embedding",
            forward.edge_embedding.len(),
            checked_len("Allegro trace edge_embedding", &[edge_count, 256])?,
        )?;
        expect_len(
            "Allegro trace tensor_embed_weights",
            forward.tensor_embed_weights.len(),
            checked_len("Allegro trace tensor_embed_weights", &[edge_count, 256])?,
        )?;
        expect_len(
            "Allegro trace first_projection",
            forward.first_projection.len(),
            checked_len("Allegro trace first_projection", &[edge_count, 512])?,
        )?;
        expect_len(
            "Allegro trace edge_features",
            forward.edge_features.len(),
            checked_len("Allegro trace edge_features", &[edge_count, 1280])?,
        )?;
        expect_len(
            "Allegro trace edge_energy",
            forward.edge_energy.len(),
            edge_count,
        )?;

        expect_len(
            "Allegro trace scalar_blocks",
            forward.scalar_blocks.len(),
            EXPECTED_NUM_LAYERS + 1,
        )?;
        for (block_index, block) in forward.scalar_blocks.iter().enumerate() {
            expect_len(
                &format!("Allegro trace scalar block {block_index}"),
                block.len(),
                checked_len("Allegro trace scalar block", &[edge_count, 256])?,
            )?;
        }

        expect_len(
            "Allegro trace layers",
            forward.layers.len(),
            EXPECTED_NUM_LAYERS,
        )?;
        for (layer_index, layer) in forward.layers.iter().enumerate() {
            expect_len(
                &format!("Allegro trace layer {layer_index} tensor_features_in"),
                layer.tensor_features_in.len(),
                checked_len(
                    "Allegro trace layer tensor_features_in",
                    &[edge_count, 64, 16],
                )?,
            )?;
            expect_len(
                &format!("Allegro trace layer {layer_index} env_weights"),
                layer.env_weights.len(),
                checked_len("Allegro trace layer env_weights", &[edge_count, 256])?,
            )?;
            expect_len(
                &format!("Allegro trace layer {layer_index} gathered_env"),
                layer.gathered_env.len(),
                checked_len("Allegro trace layer gathered_env", &[edge_count, 64, 16])?,
            )?;
            let contracted_base_dim = if layer_index == 3 { 1 } else { 16 };
            expect_len(
                &format!("Allegro trace layer {layer_index} contracted"),
                layer.contracted_len,
                checked_len(
                    "Allegro trace layer contracted",
                    &[edge_count, 64, contracted_base_dim],
                )?,
            )?;
            expect_len(
                &format!("Allegro trace layer {layer_index} scalar_tp"),
                layer.scalar_tp_len,
                checked_len("Allegro trace layer scalar_tp", &[edge_count, 64])?,
            )?;
            let latent_input_dim = 256 * (layer_index + 1) + 64;
            expect_len(
                &format!("Allegro trace layer {layer_index} latent_input"),
                layer.latent_input.len(),
                checked_len(
                    "Allegro trace layer latent_input",
                    &[edge_count, latent_input_dim],
                )?,
            )?;
            let latent_output_dim = if layer_index < 3 { 512 } else { 256 };
            expect_len(
                &format!("Allegro trace layer {layer_index} latent_output"),
                layer.latent_output_len,
                checked_len(
                    "Allegro trace layer latent_output",
                    &[edge_count, latent_output_dim],
                )?,
            )?;
        }
        Ok(())
    }

    fn atom_types(&self, structure: &Structure) -> Result<Vec<usize>, AllegroOamError> {
        if structure.species.len() != structure.positions.len() {
            return Err(invalid_model(format!(
                "structure has {} species labels but {} positions",
                structure.species.len(),
                structure.positions.len()
            )));
        }
        structure
            .species
            .iter()
            .map(|species| {
                self.type_names
                    .iter()
                    .position(|known| known == species)
                    .ok_or_else(|| {
                        invalid_model(format!(
                            "species {species:?} is absent from the pinned Allegro-OAM-L type table"
                        ))
                    })
            })
            .collect()
    }

    fn tensor_values(
        &self,
        name: &str,
        expected_shape: &[usize],
    ) -> Result<&[f64], AllegroOamError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| invalid_model(format!("required tensor {name:?} is missing")))?;
        if tensor.shape != expected_shape {
            return Err(invalid_model(format!(
                "tensor {name:?} shape must be {expected_shape:?}, got {:?}",
                tensor.shape
            )));
        }
        let expected_len = checked_len(name, expected_shape)?;
        expect_len(name, tensor.values.len(), expected_len)?;
        Ok(&tensor.values)
    }

    fn run_scalar_mlp(
        &self,
        input: &[f64],
        rows: usize,
        input_dim: usize,
        specs: &[(&str, usize, usize, f64)],
    ) -> Result<Vec<f64>, AllegroOamError> {
        let mut layers = Vec::new();
        for (index, (name, in_dim, out_dim, alpha)) in specs.iter().copied().enumerate() {
            if index > 0 {
                layers.push(AllegroScalarMlpLayer::Silu);
            }
            layers.push(AllegroScalarMlpLayer::Linear {
                weight: self.tensor_values(name, &[in_dim, out_dim])?,
                in_dim,
                out_dim,
                alpha,
            });
        }
        allegro_scalar_mlp_forward(input, rows, input_dim, &layers)
    }

    fn run_scalar_mlp_maybe_traced(
        &self,
        input: &[f64],
        rows: usize,
        input_dim: usize,
        specs: &[(&str, usize, usize, f64)],
        capture_trace: bool,
    ) -> Result<(Vec<f64>, Option<AllegroScalarMlpForwardTrace>), AllegroOamError> {
        let layers = self.scalar_mlp_layers(specs)?;
        if capture_trace {
            let (output, trace) =
                allegro_scalar_mlp_forward_traced(input, rows, input_dim, &layers)?;
            Ok((output, Some(trace)))
        } else {
            allegro_scalar_mlp_forward(input, rows, input_dim, &layers).map(|output| (output, None))
        }
    }

    fn run_scalar_mlp_vjp(
        &self,
        input: &[f64],
        rows: usize,
        input_dim: usize,
        specs: &[(&str, usize, usize, f64)],
        grad_out: &[f64],
    ) -> Result<Vec<f64>, AllegroOamError> {
        let mut layers = Vec::new();
        for (index, (name, in_dim, out_dim, alpha)) in specs.iter().copied().enumerate() {
            if index > 0 {
                layers.push(AllegroScalarMlpLayer::Silu);
            }
            layers.push(AllegroScalarMlpLayer::Linear {
                weight: self.tensor_values(name, &[in_dim, out_dim])?,
                in_dim,
                out_dim,
                alpha,
            });
        }
        allegro_scalar_mlp_vjp_input(input, rows, input_dim, &layers, grad_out)
    }

    fn run_scalar_mlp_vjp_maybe_traced(
        &self,
        input: &[f64],
        rows: usize,
        input_dim: usize,
        specs: &[(&str, usize, usize, f64)],
        trace: Option<&AllegroScalarMlpForwardTrace>,
        grad_out: &[f64],
    ) -> Result<Vec<f64>, AllegroOamError> {
        let layers = self.scalar_mlp_layers(specs)?;
        if let Some(trace) = trace {
            allegro_scalar_mlp_vjp_input_from_trace(rows, input_dim, &layers, trace, grad_out)
        } else {
            allegro_scalar_mlp_vjp_input(input, rows, input_dim, &layers, grad_out)
        }
    }

    fn scalar_mlp_layers<'a>(
        &'a self,
        specs: &[(&str, usize, usize, f64)],
    ) -> Result<Vec<AllegroScalarMlpLayer<'a>>, AllegroOamError> {
        let mut layers = Vec::new();
        for (index, (name, in_dim, out_dim, alpha)) in specs.iter().copied().enumerate() {
            if index > 0 {
                layers.push(AllegroScalarMlpLayer::Silu);
            }
            layers.push(AllegroScalarMlpLayer::Linear {
                weight: self.tensor_values(name, &[in_dim, out_dim])?,
                in_dim,
                out_dim,
                alpha,
            });
        }
        Ok(layers)
    }

    fn from_spec(
        spec: StateSpec,
        external_tensors: HashMap<String, AllegroOamExternalTensor>,
    ) -> Result<Self, AllegroOamError> {
        expect_str("schema_version", &spec.schema_version, STATE_SCHEMA_VERSION)?;
        expect_str("selected_model", &spec.selected_model, SELECTED_MODEL)?;
        expect_str("artifact.sha256", &spec.artifact.sha256, ARTIFACT_SHA256)?;
        if spec.artifact.size_bytes != ARTIFACT_SIZE_BYTES {
            return Err(invalid_model(format!(
                "artifact.size_bytes must be {ARTIFACT_SIZE_BYTES}, got {}",
                spec.artifact.size_bytes
            )));
        }
        if (spec.model_config.cutoff - EXPECTED_CUTOFF).abs() > 1.0e-12 {
            return Err(invalid_model(format!(
                "model_config.cutoff must be {EXPECTED_CUTOFF}, got {}",
                spec.model_config.cutoff
            )));
        }
        if spec.model_config.l_max != EXPECTED_L_MAX {
            return Err(invalid_model(format!(
                "model_config.l_max must be {EXPECTED_L_MAX}, got {}",
                spec.model_config.l_max
            )));
        }
        if spec.model_config.num_layers != EXPECTED_NUM_LAYERS {
            return Err(invalid_model(format!(
                "model_config.num_layers must be {EXPECTED_NUM_LAYERS}, got {}",
                spec.model_config.num_layers
            )));
        }
        if spec.type_names.len() != EXPECTED_NUM_TYPES {
            return Err(invalid_model(format!(
                "type_names length must be {EXPECTED_NUM_TYPES}, got {}",
                spec.type_names.len()
            )));
        }
        if spec.model_config.num_types != spec.type_names.len() {
            return Err(invalid_model(format!(
                "model_config.num_types must match type_names length {}, got {}",
                spec.type_names.len(),
                spec.model_config.num_types
            )));
        }
        if spec.state_dict.count != EXPECTED_STATE_TENSOR_COUNT {
            return Err(invalid_model(format!(
                "state_dict.count must be {EXPECTED_STATE_TENSOR_COUNT}, got {}",
                spec.state_dict.count
            )));
        }
        if spec.state_dict.total_numel != EXPECTED_STATE_TENSOR_NUMEL {
            return Err(invalid_model(format!(
                "state_dict.total_numel must be {EXPECTED_STATE_TENSOR_NUMEL}, got {}",
                spec.state_dict.total_numel
            )));
        }
        expect_str(
            "execution_contract.schema_version",
            &spec.execution_contract.schema_version,
            EXECUTION_CONTRACT_SCHEMA_VERSION,
        )?;
        if spec.execution_contract.contracter_count != EXPECTED_CONTRACTER_COUNT {
            return Err(invalid_model(format!(
                "execution_contract.contracter_count must be {EXPECTED_CONTRACTER_COUNT}, got {}",
                spec.execution_contract.contracter_count
            )));
        }
        if spec.execution_contract.weighted_channel_count != EXPECTED_WEIGHTED_CHANNEL_COUNT {
            return Err(invalid_model(format!(
                "execution_contract.weighted_channel_count must be {EXPECTED_WEIGHTED_CHANNEL_COUNT}, got {}",
                spec.execution_contract.weighted_channel_count
            )));
        }
        if spec.execution_contract.scalar_mlp_count != EXPECTED_SCALAR_MLP_COUNT {
            return Err(invalid_model(format!(
                "execution_contract.scalar_mlp_count must be {EXPECTED_SCALAR_MLP_COUNT}, got {}",
                spec.execution_contract.scalar_mlp_count
            )));
        }
        let plan = spec
            .execution_contract
            .plan
            .as_ref()
            .ok_or_else(|| invalid_model("Allegro execution plan is missing"))?;
        validate_runtime_plan_contract(plan)?;

        let mut loaded_tensor_numel = 0usize;
        for (name, tensor) in &external_tensors {
            let expected_numel = checked_product(&tensor.shape).map_err(|_| {
                invalid_model(format!("external tensor {name:?} shape product overflows"))
            })?;
            if expected_numel != tensor.values.len() {
                return Err(invalid_model(format!(
                    "external tensor {name:?} values length must match shape product {expected_numel}, got {}",
                    tensor.values.len()
                )));
            }
            loaded_tensor_numel = loaded_tensor_numel
                .checked_add(tensor.values.len())
                .ok_or_else(|| invalid_model("external tensor numel overflows"))?;
        }
        let model = Self {
            selected_model: spec.selected_model,
            type_names: spec.type_names,
            cutoff: spec.model_config.cutoff,
            l_max: spec.model_config.l_max,
            num_layers: spec.model_config.num_layers,
            state_tensor_count: spec.state_dict.count,
            state_tensor_numel: spec.state_dict.total_numel,
            loaded_tensor_count: external_tensors.len(),
            loaded_tensor_numel,
            artifact_sha256: spec.artifact.sha256,
            artifact_size_bytes: spec.artifact.size_bytes,
            execution_contract_status: spec.execution_contract.status,
            execution_plan: spec.execution_contract.plan,
            tensors: external_tensors,
        };
        model.validate_runtime_tensor_handoff()?;
        Ok(model)
    }
}

type ScalarMlpSpec = (&'static str, usize, usize, f64);

fn latent_specs(layer_index: usize) -> Result<&'static [ScalarMlpSpec], AllegroOamError> {
    match layer_index {
        0 => Ok(&[
            (
                "model.func.allegro.latents.0.mlp.0.weight",
                320,
                1024,
                0.055_901_698_768_138_885,
            ),
            (
                "model.func.allegro.latents.0.mlp.2.weight",
                1024,
                1024,
                0.044_194_173_067_808_15,
            ),
            (
                "model.func.allegro.latents.0.mlp.4.weight",
                1024,
                512,
                0.044_194_173_067_808_15,
            ),
        ]),
        1 => Ok(&[
            (
                "model.func.allegro.latents.1.mlp.0.weight",
                576,
                1024,
                0.041_666_667_908_430_1,
            ),
            (
                "model.func.allegro.latents.1.mlp.2.weight",
                1024,
                1024,
                0.044_194_173_067_808_15,
            ),
            (
                "model.func.allegro.latents.1.mlp.4.weight",
                1024,
                512,
                0.044_194_173_067_808_15,
            ),
        ]),
        2 => Ok(&[
            (
                "model.func.allegro.latents.2.mlp.0.weight",
                832,
                1024,
                0.034_668_762_236_833_57,
            ),
            (
                "model.func.allegro.latents.2.mlp.2.weight",
                1024,
                1024,
                0.044_194_173_067_808_15,
            ),
            (
                "model.func.allegro.latents.2.mlp.4.weight",
                1024,
                512,
                0.044_194_173_067_808_15,
            ),
        ]),
        3 => Ok(&[
            (
                "model.func.allegro.latents.3.mlp.0.weight",
                1088,
                1024,
                0.030_316_952_615_976_334,
            ),
            (
                "model.func.allegro.latents.3.mlp.2.weight",
                1024,
                1024,
                0.044_194_173_067_808_15,
            ),
            (
                "model.func.allegro.latents.3.mlp.4.weight",
                1024,
                256,
                0.044_194_173_067_808_15,
            ),
        ]),
        _ => Err(invalid_model(format!(
            "Allegro latent layer index must be 0..4, got {layer_index}"
        ))),
    }
}

fn select_columns(
    values: &[f64],
    rows: usize,
    cols: usize,
    offset: usize,
    width: usize,
) -> Result<Vec<f64>, AllegroOamError> {
    let input_len = checked_len("select_columns input", &[rows, cols])?;
    expect_len("select_columns input", values.len(), input_len)?;
    if match offset.checked_add(width) {
        Some(end) => end > cols,
        None => true,
    } {
        return Err(invalid_model(format!(
            "select_columns offset {offset} width {width} exceed column count {cols}"
        )));
    }
    let mut out = vec![0.0; checked_len("select_columns output", &[rows, width])?];
    for row in 0..rows {
        let input_offset = row * cols + offset;
        let output_offset = row * width;
        out[output_offset..output_offset + width]
            .copy_from_slice(&values[input_offset..input_offset + width]);
    }
    Ok(out)
}

#[allow(dead_code)]
fn select_columns_vjp(
    grad_out: &[f64],
    rows: usize,
    cols: usize,
    offset: usize,
    width: usize,
) -> Result<Vec<f64>, AllegroOamError> {
    let output_len = checked_len("select_columns VJP grad_out", &[rows, width])?;
    expect_len("select_columns VJP grad_out", grad_out.len(), output_len)?;
    if match offset.checked_add(width) {
        Some(end) => end > cols,
        None => true,
    } {
        return Err(invalid_model(format!(
            "select_columns VJP offset {offset} width {width} exceed column count {cols}"
        )));
    }
    let mut grad_values = vec![0.0; checked_len("select_columns VJP input", &[rows, cols])?];
    for row in 0..rows {
        let input_offset = row * cols + offset;
        let output_offset = row * width;
        grad_values[input_offset..input_offset + width]
            .copy_from_slice(&grad_out[output_offset..output_offset + width]);
    }
    Ok(grad_values)
}

fn select_columns_vjp_add(
    grad_out: &[f64],
    rows: usize,
    cols: usize,
    offset: usize,
    width: usize,
    grad_values: &mut [f64],
) -> Result<(), AllegroOamError> {
    let output_len = checked_len("select_columns VJP add grad_out", &[rows, width])?;
    expect_len(
        "select_columns VJP add grad_out",
        grad_out.len(),
        output_len,
    )?;
    let input_len = checked_len("select_columns VJP add input", &[rows, cols])?;
    expect_len("select_columns VJP add input", grad_values.len(), input_len)?;
    if match offset.checked_add(width) {
        Some(end) => end > cols,
        None => true,
    } {
        return Err(invalid_model(format!(
            "select_columns VJP add offset {offset} width {width} exceed column count {cols}"
        )));
    }
    for row in 0..rows {
        let input_offset = row * cols + offset;
        let output_offset = row * width;
        for col in 0..width {
            grad_values[input_offset + col] += grad_out[output_offset + col];
        }
    }
    Ok(())
}

fn first_tensor_component(
    values: &[f64],
    rows: usize,
    multiplicity: usize,
    base_dim: usize,
) -> Result<Vec<f64>, AllegroOamError> {
    let input_len = checked_len(
        "first tensor component input",
        &[rows, multiplicity, base_dim],
    )?;
    expect_len("first tensor component input", values.len(), input_len)?;
    let mut out = vec![0.0; checked_len("first tensor component output", &[rows, multiplicity])?];
    for row in 0..rows {
        for channel in 0..multiplicity {
            out[row * multiplicity + channel] = values[(row * multiplicity + channel) * base_dim];
        }
    }
    Ok(out)
}

#[allow(dead_code)]
fn first_tensor_component_vjp(
    grad_out: &[f64],
    rows: usize,
    multiplicity: usize,
    base_dim: usize,
) -> Result<Vec<f64>, AllegroOamError> {
    let grad_out_len = checked_len("first tensor component VJP grad_out", &[rows, multiplicity])?;
    expect_len(
        "first tensor component VJP grad_out",
        grad_out.len(),
        grad_out_len,
    )?;
    let mut grad_values = vec![
        0.0;
        checked_len(
            "first tensor component VJP input",
            &[rows, multiplicity, base_dim],
        )?
    ];
    for row in 0..rows {
        for channel in 0..multiplicity {
            grad_values[(row * multiplicity + channel) * base_dim] =
                grad_out[row * multiplicity + channel];
        }
    }
    Ok(grad_values)
}

fn first_tensor_component_vjp_add(
    grad_out: &[f64],
    rows: usize,
    multiplicity: usize,
    base_dim: usize,
    grad_values: &mut [f64],
) -> Result<(), AllegroOamError> {
    let grad_out_len = checked_len(
        "first tensor component VJP add grad_out",
        &[rows, multiplicity],
    )?;
    expect_len(
        "first tensor component VJP add grad_out",
        grad_out.len(),
        grad_out_len,
    )?;
    let grad_values_len = checked_len(
        "first tensor component VJP add input",
        &[rows, multiplicity, base_dim],
    )?;
    expect_len(
        "first tensor component VJP add input",
        grad_values.len(),
        grad_values_len,
    )?;
    for row in 0..rows {
        for channel in 0..multiplicity {
            grad_values[(row * multiplicity + channel) * base_dim] +=
                grad_out[row * multiplicity + channel];
        }
    }
    Ok(())
}

fn concat_scalar_blocks_with_tp(
    scalar_blocks: &[Vec<f64>],
    scalar_tp: &[f64],
    rows: usize,
) -> Result<Vec<f64>, AllegroOamError> {
    let mut block_refs = scalar_blocks
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<&[f64]>>();
    block_refs.push(scalar_tp);
    concat_blocks(&block_refs, rows)
}

fn concat_scalar_blocks(
    scalar_blocks: &[Vec<f64>],
    rows: usize,
) -> Result<Vec<f64>, AllegroOamError> {
    let block_refs = scalar_blocks
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<&[f64]>>();
    concat_blocks(&block_refs, rows)
}

fn add_assign(dst: &mut [f64], src: &[f64]) -> Result<(), AllegroOamError> {
    expect_len("add_assign source", src.len(), dst.len())?;
    for (dst_value, src_value) in dst.iter_mut().zip(src.iter().copied()) {
        *dst_value += src_value;
    }
    Ok(())
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

fn concat_blocks(blocks: &[&[f64]], rows: usize) -> Result<Vec<f64>, AllegroOamError> {
    if rows == 0 {
        return Err(invalid_model("concat_blocks rows must be positive"));
    }
    let mut total_dim = 0usize;
    let mut widths = Vec::with_capacity(blocks.len());
    for (index, block) in blocks.iter().enumerate() {
        if block.len() % rows != 0 {
            return Err(invalid_model(format!(
                "concat block {index} length {} is not divisible by row count {rows}",
                block.len()
            )));
        }
        let width = block.len() / rows;
        if width == 0 {
            return Err(invalid_model(format!("concat block {index} is empty")));
        }
        total_dim = total_dim
            .checked_add(width)
            .ok_or_else(|| invalid_model("concat block width overflows usize"))?;
        widths.push(width);
    }
    let mut out = vec![0.0; checked_len("concat output", &[rows, total_dim])?];
    for row in 0..rows {
        let mut out_col = 0usize;
        for (block, width) in blocks.iter().zip(widths.iter().copied()) {
            let input_offset = row * width;
            let output_offset = row * total_dim + out_col;
            out[output_offset..output_offset + width]
                .copy_from_slice(&block[input_offset..input_offset + width]);
            out_col += width;
        }
    }
    Ok(out)
}

#[allow(dead_code)]
fn concat_blocks_vjp(
    grad_out: &[f64],
    rows: usize,
    widths: &[usize],
) -> Result<Vec<Vec<f64>>, AllegroOamError> {
    if rows == 0 {
        return Err(invalid_model("concat_blocks VJP rows must be positive"));
    }
    if widths.is_empty() {
        return Err(invalid_model("concat_blocks VJP widths must be non-empty"));
    }
    let mut total_dim = 0usize;
    for width in widths {
        if *width == 0 {
            return Err(invalid_model("concat_blocks VJP width must be positive"));
        }
        total_dim = total_dim
            .checked_add(*width)
            .ok_or_else(|| invalid_model("concat_blocks VJP width overflows usize"))?;
    }
    let grad_out_len = checked_len("concat_blocks VJP grad_out", &[rows, total_dim])?;
    expect_len("concat_blocks VJP grad_out", grad_out.len(), grad_out_len)?;
    let mut grad_blocks = widths
        .iter()
        .map(|width| vec![0.0; rows * width])
        .collect::<Vec<_>>();
    for row in 0..rows {
        let mut input_col = 0usize;
        for (block, width) in grad_blocks.iter_mut().zip(widths.iter().copied()) {
            let output_offset = row * total_dim + input_col;
            let block_offset = row * width;
            block[block_offset..block_offset + width]
                .copy_from_slice(&grad_out[output_offset..output_offset + width]);
            input_col += width;
        }
    }
    Ok(grad_blocks)
}

fn concat_blocks_vjp_add(
    grad_out: &[f64],
    rows: usize,
    widths: &[usize],
    grad_blocks: &mut [&mut [f64]],
) -> Result<(), AllegroOamError> {
    if rows == 0 {
        return Err(invalid_model("concat_blocks VJP add rows must be positive"));
    }
    if widths.is_empty() {
        return Err(invalid_model(
            "concat_blocks VJP add widths must be non-empty",
        ));
    }
    expect_len(
        "concat_blocks VJP add block count",
        grad_blocks.len(),
        widths.len(),
    )?;
    let mut total_dim = 0usize;
    for (index, width) in widths.iter().copied().enumerate() {
        if width == 0 {
            return Err(invalid_model(
                "concat_blocks VJP add width must be positive",
            ));
        }
        total_dim = total_dim
            .checked_add(width)
            .ok_or_else(|| invalid_model("concat_blocks VJP add width overflows usize"))?;
        let block_len = checked_len("concat_blocks VJP add block", &[rows, width])?;
        expect_len(
            &format!("concat_blocks VJP add block {index}"),
            grad_blocks[index].len(),
            block_len,
        )?;
    }
    let grad_out_len = checked_len("concat_blocks VJP add grad_out", &[rows, total_dim])?;
    expect_len(
        "concat_blocks VJP add grad_out",
        grad_out.len(),
        grad_out_len,
    )?;
    for row in 0..rows {
        let mut input_col = 0usize;
        for (block, width) in grad_blocks.iter_mut().zip(widths.iter().copied()) {
            let output_offset = row * total_dim + input_col;
            let block_offset = row * width;
            for col in 0..width {
                block[block_offset + col] += grad_out[output_offset + col];
            }
            input_col += width;
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StateSpec {
    schema_version: String,
    selected_model: String,
    artifact: ArtifactSpec,
    model_config: ModelConfigSpec,
    type_names: Vec<String>,
    state_dict: StateDictSpec,
    execution_contract: ExecutionContractSpec,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactSpec {
    sha256: String,
    size_bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelConfigSpec {
    cutoff: f64,
    l_max: usize,
    num_types: usize,
    num_layers: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StateDictSpec {
    count: usize,
    total_numel: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionContractSpec {
    schema_version: String,
    status: String,
    contracter_count: usize,
    weighted_channel_count: usize,
    scalar_mlp_count: usize,
    plan: Option<Value>,
}

fn checked_product(shape: &[usize]) -> Result<usize, ()> {
    shape
        .iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(*value).ok_or(()))
}

fn validate_runtime_plan_contract(plan: &Value) -> Result<(), AllegroOamError> {
    expect_str(
        "execution_contract.plan.schema_version",
        string_field(plan, "schema_version")?,
        EXECUTION_CONTRACT_SCHEMA_VERSION,
    )?;
    expect_str(
        "execution_contract.plan.status",
        string_field(plan, "status")?,
        "export_plan_pinned_runtime_not_implemented",
    )?;
    let unsupported_branches = string_vec_field(plan, "unsupported_branches")?;
    let expected_unsupported = [
        "training",
        "GPU",
        "arbitrary Allegro or NeQuIP package loading",
        "runtime Python or torch dependency in Rust crates",
        "parameter gradients",
    ];
    if unsupported_branches != expected_unsupported {
        return Err(invalid_model(format!(
            "Allegro execution plan unsupported_branches must be {expected_unsupported:?}, got {unsupported_branches:?}"
        )));
    }

    let radial = field(plan, "radial_and_spherical")?;
    let edge_norm = field(radial, "edge_norm")?;
    expect_f64_field(
        "radial_and_spherical.edge_norm.r_max",
        edge_norm,
        "r_max",
        7.0,
    )?;
    expect_usize_field(
        "radial_and_spherical.edge_norm.num_types",
        edge_norm,
        "num_types",
        EXPECTED_NUM_TYPES,
    )?;
    expect_tensor_shape(
        "radial_and_spherical.edge_norm.rmax_recip",
        field(edge_norm, "rmax_recip")?,
        &[],
    )?;

    let bessel = field(radial, "bessel")?;
    expect_usize_field(
        "radial_and_spherical.bessel.num_bessels",
        bessel,
        "num_bessels",
        12,
    )?;
    expect_f64_field(
        "radial_and_spherical.bessel.cutoff_p",
        bessel,
        "cutoff_p",
        8.0,
    )?;
    expect_tensor_shape(
        "radial_and_spherical.bessel.weights",
        field(bessel, "weights")?,
        &[1, 12],
    )?;

    let sh = field(radial, "spherical_harmonics")?;
    expect_str(
        "radial_and_spherical.spherical_harmonics.normalization",
        string_field(sh, "normalization")?,
        "component",
    )?;
    expect_bool_field(
        "radial_and_spherical.spherical_harmonics.normalize",
        sh,
        "normalize",
        true,
    )?;
    expect_str(
        "radial_and_spherical.spherical_harmonics.irreps_out",
        string_field(sh, "irreps_out")?,
        "1x0e+1x1o+1x2e+1x3o",
    )?;

    let product_type_embeddings = array_field(plan, "product_type_embeddings")?;
    expect_len(
        "product_type_embeddings length",
        product_type_embeddings.len(),
        1,
    )?;
    let product_type_embedding = &product_type_embeddings[0];
    expect_tensor_shape(
        "product_type_embeddings[0].center_embed",
        field(product_type_embedding, "center_embed")?,
        &[EXPECTED_NUM_TYPES, 256],
    )?;
    expect_tensor_shape(
        "product_type_embeddings[0].neighbor_embed",
        field(product_type_embedding, "neighbor_embed")?,
        &[EXPECTED_NUM_TYPES, 256],
    )?;
    let basis_layers = array_field(field(product_type_embedding, "basis_linear")?, "layers")?;
    expect_scalar_linear_layers(
        "product_type_embeddings[0].basis_linear.layers",
        basis_layers,
        &[(12, 512)],
    )?;

    let scalar_mlps = array_field(plan, "scalar_mlps")?;
    expect_len(
        "scalar_mlps length",
        scalar_mlps.len(),
        EXPECTED_SCALAR_MLP_COUNT,
    )?;
    for (index, mlp) in scalar_mlps.iter().enumerate() {
        expect_str(
            &format!("scalar_mlps[{index}].name"),
            string_field(mlp, "name")?,
            expected_scalar_mlp_name(index)?,
        )?;
        expect_scalar_linear_layers(
            &format!("scalar_mlps[{index}].layers"),
            array_field(mlp, "layers")?,
            expected_scalar_mlp_dims(index)?,
        )?;
    }

    let weighted_channels = array_field(plan, "weighted_channels")?;
    expect_len(
        "weighted_channels length",
        weighted_channels.len(),
        EXPECTED_WEIGHTED_CHANNEL_COUNT,
    )?;
    for (index, weighted_channel) in weighted_channels.iter().enumerate() {
        expect_usize_field(
            &format!("weighted_channels[{index}].multiplicity_out"),
            weighted_channel,
            "multiplicity_out",
            64,
        )?;
        expect_usize_field(
            &format!("weighted_channels[{index}].num_irreps"),
            weighted_channel,
            "num_irreps",
            4,
        )?;
        expect_usize_field(
            &format!("weighted_channels[{index}].weight_numel"),
            weighted_channel,
            "weight_numel",
            256,
        )?;
        expect_tensor_shape(
            &format!("weighted_channels[{index}].rtoi"),
            field(weighted_channel, "rtoi")?,
            &[4, 16],
        )?;
    }

    let contracters = array_field(plan, "contracters")?;
    expect_len(
        "contracters length",
        contracters.len(),
        EXPECTED_CONTRACTER_COUNT,
    )?;
    for (index, contracter) in contracters.iter().enumerate() {
        expect_f64_field(
            &format!("contracters[{index}].scatter_factor"),
            contracter,
            "scatter_factor",
            0.010_007_007_358_585_517,
        )?;
        expect_bool_field(
            &format!("contracters[{index}].w3j_is_ij_diagonal"),
            contracter,
            "w3j_is_ij_diagonal",
            index == 3,
        )?;
        let expected_shape = if index == 3 {
            &[64, 16, 1][..]
        } else {
            &[64, 16, 16, 16][..]
        };
        expect_tensor_shape(
            &format!("contracters[{index}].precontracted_ww3j"),
            field(contracter, "precontracted_ww3j")?,
            expected_shape,
        )?;
    }

    let edgewise_reduces = array_field(plan, "edgewise_reduces")?;
    expect_len("edgewise_reduces length", edgewise_reduces.len(), 1)?;
    let edgewise_reduce = &edgewise_reduces[0];
    expect_usize_field(
        "edgewise_reduces[0].edge_index_destination_row",
        edgewise_reduce,
        "edge_index_destination_row",
        0,
    )?;
    expect_f64_field(
        "edgewise_reduces[0].factor",
        edgewise_reduce,
        "factor",
        0.007_076_022_762_639_500_6,
    )?;
    expect_str(
        "edgewise_reduces[0].reduce",
        string_field(edgewise_reduce, "reduce")?,
        "sum",
    )?;

    let scale_shift = field(plan, "per_type_scale_shift")?;
    expect_usize_field(
        "per_type_scale_shift.num_types",
        scale_shift,
        "num_types",
        EXPECTED_NUM_TYPES,
    )?;
    expect_tensor_shape(
        "per_type_scale_shift.scales",
        field(scale_shift, "scales")?,
        &[89, 1],
    )?;
    expect_tensor_shape(
        "per_type_scale_shift.shifts",
        field(scale_shift, "shifts")?,
        &[89, 1],
    )?;

    let allegro = field(plan, "allegro_module")?;
    expect_usize_field(
        "allegro_module.num_layers",
        allegro,
        "num_layers",
        EXPECTED_NUM_LAYERS,
    )?;
    expect_usize_field(
        "allegro_module.num_scalar_features",
        allegro,
        "num_scalar_features",
        256,
    )?;
    expect_usize_field(
        "allegro_module.num_tensor_features",
        allegro,
        "num_tensor_features",
        64,
    )?;

    Ok(())
}

fn expected_scalar_mlp_name(index: usize) -> Result<&'static str, AllegroOamError> {
    match index {
        0 => Ok("model.func.radial_chemical_embed.type_embed.basis_linear"),
        1 => Ok("model.func.scalar_embed_mlp.mlp_module"),
        2 => Ok("model.func.tensor_embed.env_embed_linear"),
        3 => Ok("model.func.allegro.first_layer_env_embed_projection"),
        4 => Ok("model.func.allegro.latents.0"),
        5 => Ok("model.func.allegro.latents.1"),
        6 => Ok("model.func.allegro.latents.2"),
        7 => Ok("model.func.allegro.latents.3"),
        8 => Ok("model.func.edge_readout.mlp_module"),
        _ => Err(invalid_model(format!(
            "unexpected scalar MLP index in Allegro execution plan: {index}"
        ))),
    }
}

fn expected_scalar_mlp_dims(index: usize) -> Result<&'static [(usize, usize)], AllegroOamError> {
    match index {
        0 => Ok(&[(12, 512)]),
        1 => Ok(&[(512, 512), (512, 256)]),
        2 => Ok(&[(256, 256)]),
        3 => Ok(&[(256, 512)]),
        4 => Ok(&[(320, 1024), (1024, 1024), (1024, 512)]),
        5 => Ok(&[(576, 1024), (1024, 1024), (1024, 512)]),
        6 => Ok(&[(832, 1024), (1024, 1024), (1024, 512)]),
        7 => Ok(&[(1088, 1024), (1024, 1024), (1024, 256)]),
        8 => Ok(&[(1280, 128), (128, 1)]),
        _ => Err(invalid_model(format!(
            "unexpected scalar MLP index in Allegro execution plan: {index}"
        ))),
    }
}

fn expect_scalar_linear_layers(
    label: &str,
    layers: &[Value],
    expected_dims: &[(usize, usize)],
) -> Result<(), AllegroOamError> {
    let mut actual_dims = Vec::new();
    for layer in layers {
        let class = string_field(layer, "class")?;
        if class == "ScalarLinearLayer" {
            let in_features = usize_field(layer, "in_features")?;
            let out_features = usize_field(layer, "out_features")?;
            expect_tensor_shape(
                &format!("{label} ScalarLinearLayer weight"),
                field(layer, "weight")?,
                &[in_features, out_features],
            )?;
            actual_dims.push((in_features, out_features));
        } else if class != "SiLU" {
            return Err(invalid_model(format!(
                "{label} contains unsupported layer class {class:?}"
            )));
        }
    }
    if actual_dims != expected_dims {
        return Err(invalid_model(format!(
            "{label} linear dims must be {expected_dims:?}, got {actual_dims:?}"
        )));
    }
    Ok(())
}

fn expect_tensor_shape(
    label: &str,
    tensor_ref: &Value,
    expected: &[usize],
) -> Result<(), AllegroOamError> {
    let shape = shape_field(tensor_ref, "shape")?;
    if shape != expected {
        return Err(invalid_model(format!(
            "{label}.shape must be {expected:?}, got {shape:?}"
        )));
    }
    Ok(())
}

fn expect_usize_field(
    label: &str,
    value: &Value,
    field_name: &str,
    expected: usize,
) -> Result<(), AllegroOamError> {
    let actual = usize_field(value, field_name)?;
    if actual != expected {
        return Err(invalid_model(format!(
            "{label} must be {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn expect_f64_field(
    label: &str,
    value: &Value,
    field_name: &str,
    expected: f64,
) -> Result<(), AllegroOamError> {
    let actual = f64_field(value, field_name)?;
    if (actual - expected).abs() > 1.0e-15 {
        return Err(invalid_model(format!(
            "{label} must be {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn expect_bool_field(
    label: &str,
    value: &Value,
    field_name: &str,
    expected: bool,
) -> Result<(), AllegroOamError> {
    let actual = bool_field(value, field_name)?;
    if actual != expected {
        return Err(invalid_model(format!(
            "{label} must be {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn runtime_required_tensor_names_from_plan(plan: &Value) -> Result<Vec<String>, AllegroOamError> {
    let mut names = BTreeSet::new();
    insert_tensor_name(
        &mut names,
        field(
            field(field(plan, "radial_and_spherical")?, "edge_norm")?,
            "rmax_recip",
        )?,
    )?;
    insert_tensor_name(
        &mut names,
        field(
            field(field(plan, "radial_and_spherical")?, "bessel")?,
            "weights",
        )?,
    )?;

    for embedding in array_field(plan, "product_type_embeddings")? {
        insert_tensor_name(&mut names, field(embedding, "center_embed")?)?;
        insert_tensor_name(&mut names, field(embedding, "neighbor_embed")?)?;
        for layer in array_field(field(embedding, "basis_linear")?, "layers")? {
            if field(layer, "class")?.as_str() == Some("ScalarLinearLayer") {
                insert_tensor_name(&mut names, field(layer, "weight")?)?;
            }
        }
    }

    for mlp in array_field(plan, "scalar_mlps")? {
        for layer in array_field(mlp, "layers")? {
            if field(layer, "class")?.as_str() == Some("ScalarLinearLayer") {
                insert_tensor_name(&mut names, field(layer, "weight")?)?;
            }
        }
    }

    for weighted_channel in array_field(plan, "weighted_channels")? {
        insert_tensor_name(&mut names, field(weighted_channel, "rtoi")?)?;
    }

    for contracter in array_field(plan, "contracters")? {
        insert_tensor_name(&mut names, field(contracter, "precontracted_ww3j")?)?;
    }

    let scale_shift = field(plan, "per_type_scale_shift")?;
    insert_tensor_name(&mut names, field(scale_shift, "scales")?)?;
    insert_tensor_name(&mut names, field(scale_shift, "shifts")?)?;

    Ok(names.into_iter().collect())
}

fn insert_tensor_name(names: &mut BTreeSet<String>, value: &Value) -> Result<(), AllegroOamError> {
    let name = string_field(value, "name")?;
    names.insert(name.to_owned());
    Ok(())
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value, AllegroOamError> {
    value
        .get(name)
        .ok_or_else(|| invalid_model(format!("Allegro execution plan field {name:?} is missing")))
}

fn array_field<'a>(value: &'a Value, name: &str) -> Result<&'a [Value], AllegroOamError> {
    field(value, name)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| {
            invalid_model(format!(
                "Allegro execution plan field {name:?} must be an array"
            ))
        })
}

fn string_field<'a>(value: &'a Value, name: &str) -> Result<&'a str, AllegroOamError> {
    field(value, name)?.as_str().ok_or_else(|| {
        invalid_model(format!(
            "Allegro execution plan field {name:?} must be a string"
        ))
    })
}

fn usize_field(value: &Value, name: &str) -> Result<usize, AllegroOamError> {
    let raw = field(value, name)?.as_u64().ok_or_else(|| {
        invalid_model(format!(
            "Allegro execution plan field {name:?} must be an unsigned integer"
        ))
    })?;
    usize::try_from(raw).map_err(|_| {
        invalid_model(format!(
            "Allegro execution plan field {name:?} is too large for this platform"
        ))
    })
}

fn f64_field(value: &Value, name: &str) -> Result<f64, AllegroOamError> {
    field(value, name)?.as_f64().ok_or_else(|| {
        invalid_model(format!(
            "Allegro execution plan field {name:?} must be a finite number"
        ))
    })
}

fn bool_field(value: &Value, name: &str) -> Result<bool, AllegroOamError> {
    field(value, name)?.as_bool().ok_or_else(|| {
        invalid_model(format!(
            "Allegro execution plan field {name:?} must be a boolean"
        ))
    })
}

fn shape_field(value: &Value, name: &str) -> Result<Vec<usize>, AllegroOamError> {
    array_field(value, name)?
        .iter()
        .map(|item| {
            let raw = item.as_u64().ok_or_else(|| {
                invalid_model(format!(
                    "Allegro execution plan shape field {name:?} must contain unsigned integers"
                ))
            })?;
            usize::try_from(raw).map_err(|_| {
                invalid_model(format!(
                    "Allegro execution plan shape field {name:?} is too large for this platform"
                ))
            })
        })
        .collect()
}

fn string_vec_field(value: &Value, name: &str) -> Result<Vec<String>, AllegroOamError> {
    array_field(value, name)?
        .iter()
        .map(|item| {
            item.as_str().map(str::to_owned).ok_or_else(|| {
                invalid_model(format!(
                    "Allegro execution plan field {name:?} must contain strings"
                ))
            })
        })
        .collect()
}

#[allow(dead_code)]
/// Expands Allegro `MakeWeightedChannels` weights over irrep component blocks.
pub(crate) fn allegro_make_weighted_channels_forward(
    edge_attrs: &[f64],
    weights: &[f64],
    rtoi: &[f64],
    shape: AllegroWeightedChannelsShape,
) -> Result<Vec<f64>, AllegroOamError> {
    ensure_positive(
        "weighted channels",
        &[
            shape.rows,
            shape.multiplicity_out,
            shape.num_irreps,
            shape.base_dim,
        ],
    )?;
    let edge_attr_len = checked_len("edge_attrs", &[shape.rows, shape.base_dim])?;
    let weight_len = checked_len(
        "weighted channel weights",
        &[shape.rows, shape.multiplicity_out, shape.num_irreps],
    )?;
    let rtoi_len = checked_len("weighted channel rtoi", &[shape.num_irreps, shape.base_dim])?;
    let output_len = checked_len(
        "weighted channel output",
        &[shape.rows, shape.multiplicity_out, shape.base_dim],
    )?;
    expect_len("edge_attrs", edge_attrs.len(), edge_attr_len)?;
    expect_len("weighted channel weights", weights.len(), weight_len)?;
    expect_len("weighted channel rtoi", rtoi.len(), rtoi_len)?;

    let mut out = vec![0.0; output_len];
    #[cfg(feature = "parallel")]
    if output_len >= ALLEGRO_PARALLEL_ELEMENTWISE_THRESHOLD && rayon::current_num_threads() > 1 {
        let output_row_dim = shape.multiplicity_out * shape.base_dim;
        let weight_row_dim = shape.multiplicity_out * shape.num_irreps;
        out.par_chunks_mut(output_row_dim)
            .enumerate()
            .for_each(|(row, out_row)| {
                let edge_attr_row = &edge_attrs[row * shape.base_dim..(row + 1) * shape.base_dim];
                let weight_row = &weights[row * weight_row_dim..(row + 1) * weight_row_dim];
                for channel in 0..shape.multiplicity_out {
                    for component in 0..shape.base_dim {
                        let mut expanded_weight = 0.0;
                        for irrep in 0..shape.num_irreps {
                            let weight_index = channel * shape.num_irreps + irrep;
                            let rtoi_index = irrep * shape.base_dim + component;
                            expanded_weight += weight_row[weight_index] * rtoi[rtoi_index];
                        }
                        out_row[channel * shape.base_dim + component] =
                            edge_attr_row[component] * expanded_weight;
                    }
                }
            });
        return Ok(out);
    }
    for row in 0..shape.rows {
        for channel in 0..shape.multiplicity_out {
            for component in 0..shape.base_dim {
                let mut expanded_weight = 0.0;
                for irrep in 0..shape.num_irreps {
                    let weight_index =
                        (row * shape.multiplicity_out + channel) * shape.num_irreps + irrep;
                    let rtoi_index = irrep * shape.base_dim + component;
                    expanded_weight += weights[weight_index] * rtoi[rtoi_index];
                }
                let edge_attr_index = row * shape.base_dim + component;
                let out_index =
                    (row * shape.multiplicity_out + channel) * shape.base_dim + component;
                out[out_index] = edge_attrs[edge_attr_index] * expanded_weight;
            }
        }
    }
    Ok(out)
}

#[allow(dead_code)]
pub(crate) fn allegro_make_weighted_channels_vjp_inputs(
    edge_attrs: &[f64],
    weights: &[f64],
    rtoi: &[f64],
    grad_out: &[f64],
    shape: AllegroWeightedChannelsShape,
) -> Result<AllegroWeightedChannelsVjp, AllegroOamError> {
    ensure_positive(
        "weighted channels VJP",
        &[
            shape.rows,
            shape.multiplicity_out,
            shape.num_irreps,
            shape.base_dim,
        ],
    )?;
    let edge_attr_len = checked_len(
        "weighted channels VJP edge_attrs",
        &[shape.rows, shape.base_dim],
    )?;
    let weight_len = checked_len(
        "weighted channels VJP weights",
        &[shape.rows, shape.multiplicity_out, shape.num_irreps],
    )?;
    let rtoi_len = checked_len(
        "weighted channels VJP rtoi",
        &[shape.num_irreps, shape.base_dim],
    )?;
    let output_len = checked_len(
        "weighted channels VJP grad_out",
        &[shape.rows, shape.multiplicity_out, shape.base_dim],
    )?;
    expect_len(
        "weighted channels VJP edge_attrs",
        edge_attrs.len(),
        edge_attr_len,
    )?;
    expect_len("weighted channels VJP weights", weights.len(), weight_len)?;
    expect_len("weighted channels VJP rtoi", rtoi.len(), rtoi_len)?;
    expect_len("weighted channels VJP grad_out", grad_out.len(), output_len)?;

    let mut grad_edge_attrs = vec![0.0; edge_attr_len];
    let mut grad_weights = vec![0.0; weight_len];
    #[cfg(feature = "parallel")]
    if output_len >= ALLEGRO_PARALLEL_ELEMENTWISE_THRESHOLD && rayon::current_num_threads() > 1 {
        let output_row_dim = shape.multiplicity_out * shape.base_dim;
        let weight_row_dim = shape.multiplicity_out * shape.num_irreps;
        grad_edge_attrs
            .par_chunks_mut(shape.base_dim)
            .zip(grad_weights.par_chunks_mut(weight_row_dim))
            .enumerate()
            .for_each(|(row, (grad_edge_row, grad_weight_row))| {
                let edge_attr_row = &edge_attrs[row * shape.base_dim..(row + 1) * shape.base_dim];
                let weight_row = &weights[row * weight_row_dim..(row + 1) * weight_row_dim];
                let grad_out_row = &grad_out[row * output_row_dim..(row + 1) * output_row_dim];
                for channel in 0..shape.multiplicity_out {
                    for component in 0..shape.base_dim {
                        let out_index = channel * shape.base_dim + component;
                        let bar_out = grad_out_row[out_index];
                        let mut expanded_weight = 0.0;
                        for irrep in 0..shape.num_irreps {
                            let weight_index = channel * shape.num_irreps + irrep;
                            let rtoi_index = irrep * shape.base_dim + component;
                            let rtoi_value = rtoi[rtoi_index];
                            expanded_weight += weight_row[weight_index] * rtoi_value;
                            grad_weight_row[weight_index] +=
                                bar_out * edge_attr_row[component] * rtoi_value;
                        }
                        grad_edge_row[component] += bar_out * expanded_weight;
                    }
                }
            });
        return Ok(AllegroWeightedChannelsVjp {
            edge_attrs: grad_edge_attrs,
            weights: grad_weights,
        });
    }
    for row in 0..shape.rows {
        for channel in 0..shape.multiplicity_out {
            for component in 0..shape.base_dim {
                let edge_attr_index = row * shape.base_dim + component;
                let out_index =
                    (row * shape.multiplicity_out + channel) * shape.base_dim + component;
                let bar_out = grad_out[out_index];
                let mut expanded_weight = 0.0;
                for irrep in 0..shape.num_irreps {
                    let weight_index =
                        (row * shape.multiplicity_out + channel) * shape.num_irreps + irrep;
                    let rtoi_index = irrep * shape.base_dim + component;
                    let rtoi_value = rtoi[rtoi_index];
                    expanded_weight += weights[weight_index] * rtoi_value;
                    grad_weights[weight_index] +=
                        bar_out * edge_attrs[edge_attr_index] * rtoi_value;
                }
                grad_edge_attrs[edge_attr_index] += bar_out * expanded_weight;
            }
        }
    }
    Ok(AllegroWeightedChannelsVjp {
        edge_attrs: grad_edge_attrs,
        weights: grad_weights,
    })
}

#[allow(dead_code)]
pub(crate) fn allegro_radial_bessel_embedding_forward(
    normed_lengths: &[f64],
    bessel_weights: &[f64],
    cutoff_p: f64,
    shape: AllegroRadialEmbeddingShape,
) -> Result<Vec<f64>, AllegroOamError> {
    ensure_positive("radial embedding", &[shape.rows, shape.num_bessels])?;
    if !cutoff_p.is_finite() || cutoff_p <= 0.0 {
        return Err(invalid_model(format!(
            "radial embedding cutoff_p must be positive finite, got {cutoff_p}"
        )));
    }
    expect_len("radial normed lengths", normed_lengths.len(), shape.rows)?;
    expect_len(
        "radial bessel weights",
        bessel_weights.len(),
        shape.num_bessels,
    )?;
    let output_len = checked_len("radial embedding output", &[shape.rows, shape.num_bessels])?;
    let mut out = vec![0.0; output_len];
    for row in 0..shape.rows {
        let normed = normed_lengths[row];
        if !normed.is_finite() || normed < 0.0 {
            return Err(invalid_model(format!(
                "radial normed length must be non-negative finite, got {normed}"
            )));
        }
        let cutoff = allegro_polynomial_cutoff(normed, cutoff_p);
        for basis_index in 0..shape.num_bessels {
            let weight = bessel_weights[basis_index];
            if !weight.is_finite() {
                return Err(invalid_model(format!(
                    "radial bessel weight must be finite, got {weight}"
                )));
            }
            out[row * shape.num_bessels + basis_index] =
                allegro_torch_sinc(normed * weight) * weight * cutoff;
        }
    }
    Ok(out)
}

#[allow(dead_code)]
pub(crate) fn allegro_radial_bessel_embedding_vjp_normed_lengths(
    normed_lengths: &[f64],
    bessel_weights: &[f64],
    cutoff_p: f64,
    grad_out: &[f64],
    shape: AllegroRadialEmbeddingShape,
) -> Result<Vec<f64>, AllegroOamError> {
    ensure_positive("radial embedding VJP", &[shape.rows, shape.num_bessels])?;
    if !cutoff_p.is_finite() || cutoff_p <= 0.0 {
        return Err(invalid_model(format!(
            "radial embedding VJP cutoff_p must be positive finite, got {cutoff_p}"
        )));
    }
    expect_len(
        "radial embedding VJP normed lengths",
        normed_lengths.len(),
        shape.rows,
    )?;
    expect_len(
        "radial embedding VJP bessel weights",
        bessel_weights.len(),
        shape.num_bessels,
    )?;
    let grad_out_len = checked_len(
        "radial embedding VJP grad_out",
        &[shape.rows, shape.num_bessels],
    )?;
    expect_len(
        "radial embedding VJP grad_out",
        grad_out.len(),
        grad_out_len,
    )?;

    let mut grad_normed = vec![0.0; shape.rows];
    for row in 0..shape.rows {
        let normed = normed_lengths[row];
        if !normed.is_finite() || normed < 0.0 {
            return Err(invalid_model(format!(
                "radial embedding VJP normed length must be non-negative finite, got {normed}"
            )));
        }
        let cutoff = allegro_polynomial_cutoff(normed, cutoff_p);
        let cutoff_derivative = allegro_polynomial_cutoff_derivative(normed, cutoff_p);
        for basis_index in 0..shape.num_bessels {
            let weight = bessel_weights[basis_index];
            if !weight.is_finite() {
                return Err(invalid_model(format!(
                    "radial embedding VJP bessel weight must be finite, got {weight}"
                )));
            }
            let argument = normed * weight;
            let derivative = weight
                * (allegro_torch_sinc_derivative(argument) * weight * cutoff
                    + allegro_torch_sinc(argument) * cutoff_derivative);
            grad_normed[row] += grad_out[row * shape.num_bessels + basis_index] * derivative;
        }
    }
    Ok(grad_normed)
}

#[allow(dead_code)]
pub(crate) fn allegro_polynomial_cutoff(x: f64, p: f64) -> f64 {
    if x >= 1.0 {
        return 0.0;
    }
    1.0 - ((p + 1.0) * (p + 2.0) / 2.0) * x.powf(p) + p * (p + 2.0) * x.powf(p + 1.0)
        - (p * (p + 1.0) / 2.0) * x.powf(p + 2.0)
}

#[allow(dead_code)]
pub(crate) fn allegro_polynomial_cutoff_derivative(x: f64, p: f64) -> f64 {
    if x >= 1.0 {
        return 0.0;
    }
    if x == 0.0 && p > 1.0 {
        return 0.0;
    }
    let a = (p + 1.0) * (p + 2.0) / 2.0;
    let b = p * (p + 2.0);
    let c = p * (p + 1.0) / 2.0;
    -a * p * x.powf(p - 1.0) + b * (p + 1.0) * x.powf(p) - c * (p + 2.0) * x.powf(p + 1.0)
}

#[allow(dead_code)]
pub(crate) fn allegro_torch_sinc(x: f64) -> f64 {
    let pix = std::f64::consts::PI * x;
    if pix.abs() < 1.0e-12 {
        1.0
    } else {
        pix.sin() / pix
    }
}

#[allow(dead_code)]
pub(crate) fn allegro_torch_sinc_derivative(x: f64) -> f64 {
    let pix = std::f64::consts::PI * x;
    if pix.abs() < 1.0e-8 {
        0.0
    } else {
        (pix * pix.cos() - pix.sin()) / (std::f64::consts::PI * x * x)
    }
}

#[allow(dead_code)]
/// Applies a row-major batch to an Allegro ScalarMLP linear layer.
///
/// Pinned Allegro stores ScalarMLP weights as `[in_features, out_features]`
/// and applies each layer-specific alpha after the matrix multiply.
pub(crate) fn allegro_input_major_linear_forward(
    input: &[f64],
    weight: &[f64],
    alpha: f64,
    shape: AllegroLinearShape,
) -> Result<Vec<f64>, AllegroOamError> {
    ensure_positive("linear", &[shape.rows, shape.in_dim, shape.out_dim])?;
    if !alpha.is_finite() {
        return Err(invalid_model(format!(
            "linear alpha must be finite, got {alpha}"
        )));
    }
    let input_len = checked_len("linear input", &[shape.rows, shape.in_dim])?;
    let weight_len = checked_len("linear weight", &[shape.in_dim, shape.out_dim])?;
    let output_len = checked_len("linear output", &[shape.rows, shape.out_dim])?;
    expect_len("linear input", input.len(), input_len)?;
    expect_len("linear weight", weight.len(), weight_len)?;

    let mut out = vec![0.0; output_len];
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        ),
        all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
    ))]
    if shape.rows >= ALLEGRO_BLAS_ROW_THRESHOLD && shape.out_dim >= ALLEGRO_BLAS_OUT_DIM_THRESHOLD {
        blas_backend::dense_forward(input, shape.rows, shape.in_dim, weight, &mut out, alpha);
        return Ok(out);
    }
    for row in 0..shape.rows {
        for out_index in 0..shape.out_dim {
            let mut value = 0.0;
            for in_index in 0..shape.in_dim {
                value += input[row * shape.in_dim + in_index]
                    * weight[in_index * shape.out_dim + out_index];
            }
            out[row * shape.out_dim + out_index] = alpha * value;
        }
    }
    Ok(out)
}

#[allow(dead_code)]
pub(crate) fn allegro_input_major_linear_vjp_input(
    weight: &[f64],
    alpha: f64,
    grad_out: &[f64],
    shape: AllegroLinearShape,
) -> Result<Vec<f64>, AllegroOamError> {
    ensure_positive("linear VJP", &[shape.rows, shape.in_dim, shape.out_dim])?;
    if !alpha.is_finite() {
        return Err(invalid_model(format!(
            "linear VJP alpha must be finite, got {alpha}"
        )));
    }
    let weight_len = checked_len("linear VJP weight", &[shape.in_dim, shape.out_dim])?;
    let grad_out_len = checked_len("linear VJP grad_out", &[shape.rows, shape.out_dim])?;
    let grad_input_len = checked_len("linear VJP grad_input", &[shape.rows, shape.in_dim])?;
    expect_len("linear VJP weight", weight.len(), weight_len)?;
    expect_len("linear VJP grad_out", grad_out.len(), grad_out_len)?;

    let mut grad_input = vec![0.0; grad_input_len];
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        ),
        all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
    ))]
    if shape.rows >= ALLEGRO_BLAS_ROW_THRESHOLD && shape.out_dim >= ALLEGRO_BLAS_OUT_DIM_THRESHOLD {
        blas_backend::dense_vjp_input(
            shape.rows,
            shape.in_dim,
            weight,
            grad_out,
            &mut grad_input,
            alpha,
        );
        return Ok(grad_input);
    }
    for row in 0..shape.rows {
        for in_index in 0..shape.in_dim {
            let mut value = 0.0;
            for out_index in 0..shape.out_dim {
                value += grad_out[row * shape.out_dim + out_index]
                    * alpha
                    * weight[in_index * shape.out_dim + out_index];
            }
            grad_input[row * shape.in_dim + in_index] = value;
        }
    }
    Ok(grad_input)
}

#[allow(dead_code)]
pub(crate) fn allegro_silu(value: f64) -> f64 {
    value / (1.0 + (-value).exp())
}

#[allow(dead_code)]
pub(crate) fn allegro_silu_derivative(value: f64) -> f64 {
    let sigmoid = 1.0 / (1.0 + (-value).exp());
    sigmoid * (1.0 + value * (1.0 - sigmoid))
}

fn allegro_silu_and_derivative(value: f64) -> (f64, f64) {
    let sigmoid = 1.0 / (1.0 + (-value).exp());
    (value * sigmoid, sigmoid * (1.0 + value * (1.0 - sigmoid)))
}

fn allegro_apply_silu_in_place(values: &mut [f64]) {
    #[cfg(feature = "parallel")]
    if values.len() >= ALLEGRO_PARALLEL_ELEMENTWISE_THRESHOLD && rayon::current_num_threads() > 1 {
        values.par_iter_mut().for_each(|value| {
            *value = allegro_silu(*value);
        });
        return;
    }
    for value in values {
        *value = allegro_silu(*value);
    }
}

fn allegro_apply_silu_with_derivative_trace(values: &mut [f64]) -> Vec<f64> {
    let mut derivatives = vec![0.0; values.len()];
    #[cfg(feature = "parallel")]
    if values.len() >= ALLEGRO_PARALLEL_ELEMENTWISE_THRESHOLD && rayon::current_num_threads() > 1 {
        values
            .par_iter_mut()
            .zip(derivatives.par_iter_mut())
            .for_each(|(value, derivative)| {
                let (activated, local_derivative) = allegro_silu_and_derivative(*value);
                *value = activated;
                *derivative = local_derivative;
            });
        return derivatives;
    }
    for (value, derivative) in values.iter_mut().zip(&mut derivatives) {
        let (activated, local_derivative) = allegro_silu_and_derivative(*value);
        *value = activated;
        *derivative = local_derivative;
    }
    derivatives
}

fn allegro_multiply_by_silu_derivatives(bars: &mut [f64], derivatives: &[f64]) {
    #[cfg(feature = "parallel")]
    if bars.len() >= ALLEGRO_PARALLEL_ELEMENTWISE_THRESHOLD && rayon::current_num_threads() > 1 {
        bars.par_iter_mut()
            .zip(derivatives.par_iter())
            .for_each(|(bar, derivative)| {
                *bar *= *derivative;
            });
        return;
    }
    for (bar, derivative) in bars.iter_mut().zip(derivatives) {
        *bar *= *derivative;
    }
}

#[allow(dead_code)]
pub(crate) fn allegro_scalar_mlp_forward(
    input: &[f64],
    rows: usize,
    input_dim: usize,
    layers: &[AllegroScalarMlpLayer<'_>],
) -> Result<Vec<f64>, AllegroOamError> {
    ensure_positive("scalar MLP input", &[rows, input_dim])?;
    let input_len = checked_len("scalar MLP input", &[rows, input_dim])?;
    expect_len("scalar MLP input", input.len(), input_len)?;
    let mut current: Vec<f64> = Vec::new();
    let mut current_dim = input_dim;
    let mut first = true;
    for (layer_index, layer) in layers.iter().enumerate() {
        match *layer {
            AllegroScalarMlpLayer::Linear {
                weight,
                in_dim,
                out_dim,
                alpha,
            } => {
                if in_dim != current_dim {
                    return Err(invalid_model(format!(
                        "scalar MLP layer {layer_index} input dimension must be {current_dim}, got {in_dim}"
                    )));
                }
                let source: &[f64] = if first { input } else { &current };
                current = allegro_input_major_linear_forward(
                    source,
                    weight,
                    alpha,
                    AllegroLinearShape {
                        rows,
                        in_dim,
                        out_dim,
                    },
                )?;
                current_dim = out_dim;
            }
            AllegroScalarMlpLayer::Silu => {
                if first {
                    current = input.to_vec();
                }
                allegro_apply_silu_in_place(&mut current);
            }
        }
        first = false;
    }
    Ok(current)
}

fn allegro_scalar_mlp_forward_traced(
    input: &[f64],
    rows: usize,
    input_dim: usize,
    layers: &[AllegroScalarMlpLayer<'_>],
) -> Result<(Vec<f64>, AllegroScalarMlpForwardTrace), AllegroOamError> {
    ensure_positive("scalar MLP traced input", &[rows, input_dim])?;
    let input_len = checked_len("scalar MLP traced input", &[rows, input_dim])?;
    expect_len("scalar MLP traced input", input.len(), input_len)?;
    let mut current: Vec<f64> = Vec::new();
    let mut current_dim = input_dim;
    let mut silu_inputs = Vec::new();
    let mut first = true;
    for (layer_index, layer) in layers.iter().enumerate() {
        match *layer {
            AllegroScalarMlpLayer::Linear {
                weight,
                in_dim,
                out_dim,
                alpha,
            } => {
                if in_dim != current_dim {
                    return Err(invalid_model(format!(
                        "scalar MLP traced layer {layer_index} input dimension must be {current_dim}, got {in_dim}"
                    )));
                }
                let source: &[f64] = if first { input } else { &current };
                current = allegro_input_major_linear_forward(
                    source,
                    weight,
                    alpha,
                    AllegroLinearShape {
                        rows,
                        in_dim,
                        out_dim,
                    },
                )?;
                current_dim = out_dim;
            }
            AllegroScalarMlpLayer::Silu => {
                if first {
                    current = input.to_vec();
                }
                let derivatives = allegro_apply_silu_with_derivative_trace(&mut current);
                silu_inputs.push(AllegroScalarMlpSiluTrace {
                    dim: current_dim,
                    derivatives,
                });
            }
        }
        first = false;
    }
    Ok((
        current,
        AllegroScalarMlpForwardTrace {
            output_dim: current_dim,
            silu_inputs,
        },
    ))
}

#[allow(dead_code)]
pub(crate) fn allegro_scalar_mlp_vjp_input(
    input: &[f64],
    rows: usize,
    input_dim: usize,
    layers: &[AllegroScalarMlpLayer<'_>],
    grad_out: &[f64],
) -> Result<Vec<f64>, AllegroOamError> {
    ensure_positive("scalar MLP VJP input", &[rows, input_dim])?;
    let input_len = checked_len("scalar MLP VJP input", &[rows, input_dim])?;
    expect_len("scalar MLP VJP input", input.len(), input_len)?;

    if layers
        .iter()
        .all(|layer| matches!(layer, AllegroScalarMlpLayer::Linear { .. }))
    {
        let mut current_dim = input_dim;
        for (layer_index, layer) in layers.iter().enumerate() {
            let AllegroScalarMlpLayer::Linear {
                in_dim, out_dim, ..
            } = *layer
            else {
                unreachable!("non-linear layer was filtered above");
            };
            if in_dim != current_dim {
                return Err(invalid_model(format!(
                    "scalar MLP VJP layer {layer_index} input dimension must be {current_dim}, got {in_dim}"
                )));
            }
            current_dim = out_dim;
        }
        let grad_out_len = checked_len("scalar MLP VJP grad_out", &[rows, current_dim])?;
        expect_len("scalar MLP VJP grad_out", grad_out.len(), grad_out_len)?;
        let mut current_bar = grad_out.to_vec();
        for layer in layers.iter().rev() {
            let AllegroScalarMlpLayer::Linear {
                weight,
                in_dim,
                out_dim,
                alpha,
            } = *layer
            else {
                unreachable!("non-linear layer was filtered above");
            };
            current_bar = allegro_input_major_linear_vjp_input(
                weight,
                alpha,
                &current_bar,
                AllegroLinearShape {
                    rows,
                    in_dim,
                    out_dim,
                },
            )?;
        }
        return Ok(current_bar);
    }

    let mut activations = Vec::with_capacity(layers.len() + 1);
    let mut dims = Vec::with_capacity(layers.len() + 1);
    activations.push(input.to_vec());
    dims.push(input_dim);

    let mut current = input.to_vec();
    let mut current_dim = input_dim;
    for (layer_index, layer) in layers.iter().enumerate() {
        match *layer {
            AllegroScalarMlpLayer::Linear {
                weight,
                in_dim,
                out_dim,
                alpha,
            } => {
                if in_dim != current_dim {
                    return Err(invalid_model(format!(
                        "scalar MLP VJP layer {layer_index} input dimension must be {current_dim}, got {in_dim}"
                    )));
                }
                current = allegro_input_major_linear_forward(
                    &current,
                    weight,
                    alpha,
                    AllegroLinearShape {
                        rows,
                        in_dim,
                        out_dim,
                    },
                )?;
                current_dim = out_dim;
            }
            AllegroScalarMlpLayer::Silu => {
                allegro_apply_silu_in_place(&mut current);
            }
        }
        activations.push(current.clone());
        dims.push(current_dim);
    }

    let output_dim = *dims
        .last()
        .ok_or_else(|| invalid_model("scalar MLP VJP has no output dimension"))?;
    let grad_out_len = checked_len("scalar MLP VJP grad_out", &[rows, output_dim])?;
    expect_len("scalar MLP VJP grad_out", grad_out.len(), grad_out_len)?;
    let mut current_bar = grad_out.to_vec();

    for (layer_index, layer) in layers.iter().enumerate().rev() {
        match *layer {
            AllegroScalarMlpLayer::Linear {
                weight,
                in_dim,
                out_dim,
                alpha,
            } => {
                current_bar = allegro_input_major_linear_vjp_input(
                    weight,
                    alpha,
                    &current_bar,
                    AllegroLinearShape {
                        rows,
                        in_dim,
                        out_dim,
                    },
                )?;
            }
            AllegroScalarMlpLayer::Silu => {
                let activation_input = &activations[layer_index];
                let expected_len =
                    checked_len("scalar MLP VJP SiLU input", &[rows, dims[layer_index]])?;
                expect_len(
                    "scalar MLP VJP SiLU input",
                    activation_input.len(),
                    expected_len,
                )?;
                expect_len("scalar MLP VJP SiLU grad", current_bar.len(), expected_len)?;
                for (bar, value) in current_bar.iter_mut().zip(activation_input) {
                    *bar *= allegro_silu_derivative(*value);
                }
            }
        }
    }
    Ok(current_bar)
}

fn allegro_scalar_mlp_vjp_input_from_trace(
    rows: usize,
    input_dim: usize,
    layers: &[AllegroScalarMlpLayer<'_>],
    trace: &AllegroScalarMlpForwardTrace,
    grad_out: &[f64],
) -> Result<Vec<f64>, AllegroOamError> {
    ensure_positive("scalar MLP traced VJP input", &[rows, input_dim])?;
    let mut current_dim = input_dim;
    let mut silu_count = 0usize;
    for (layer_index, layer) in layers.iter().enumerate() {
        match *layer {
            AllegroScalarMlpLayer::Linear {
                in_dim, out_dim, ..
            } => {
                if in_dim != current_dim {
                    return Err(invalid_model(format!(
                        "scalar MLP traced VJP layer {layer_index} input dimension must be {current_dim}, got {in_dim}"
                    )));
                }
                current_dim = out_dim;
            }
            AllegroScalarMlpLayer::Silu => {
                silu_count += 1;
            }
        }
    }
    if trace.output_dim != current_dim {
        return Err(invalid_model(format!(
            "scalar MLP traced VJP output dimension must be {current_dim}, got {}",
            trace.output_dim
        )));
    }
    expect_len(
        "scalar MLP traced VJP SiLU trace count",
        trace.silu_inputs.len(),
        silu_count,
    )?;
    let grad_out_len = checked_len("scalar MLP traced VJP grad_out", &[rows, current_dim])?;
    expect_len(
        "scalar MLP traced VJP grad_out",
        grad_out.len(),
        grad_out_len,
    )?;

    let mut current_bar = grad_out.to_vec();
    let mut silu_index = trace.silu_inputs.len();
    for layer in layers.iter().rev() {
        match *layer {
            AllegroScalarMlpLayer::Linear {
                weight,
                in_dim,
                out_dim,
                alpha,
            } => {
                current_bar = allegro_input_major_linear_vjp_input(
                    weight,
                    alpha,
                    &current_bar,
                    AllegroLinearShape {
                        rows,
                        in_dim,
                        out_dim,
                    },
                )?;
            }
            AllegroScalarMlpLayer::Silu => {
                silu_index = silu_index.checked_sub(1).ok_or_else(|| {
                    invalid_model("scalar MLP traced VJP encountered too many SiLU layers")
                })?;
                let activation_input = &trace.silu_inputs[silu_index];
                let expected_len = checked_len(
                    "scalar MLP traced VJP SiLU input",
                    &[rows, activation_input.dim],
                )?;
                expect_len(
                    "scalar MLP traced VJP SiLU derivative",
                    activation_input.derivatives.len(),
                    expected_len,
                )?;
                expect_len(
                    "scalar MLP traced VJP SiLU grad",
                    current_bar.len(),
                    expected_len,
                )?;
                allegro_multiply_by_silu_derivatives(
                    &mut current_bar,
                    &activation_input.derivatives,
                );
            }
        }
    }
    if silu_index != 0 {
        return Err(invalid_model(format!(
            "scalar MLP traced VJP left {silu_index} unused SiLU traces"
        )));
    }
    Ok(current_bar)
}

#[allow(dead_code)]
/// Computes the pinned Allegro `ProductTypeEmbedding` forward path.
pub(crate) fn allegro_product_type_embedding_forward(
    inputs: AllegroProductTypeEmbeddingInputs<'_>,
    shape: AllegroProductTypeEmbeddingShape,
) -> Result<Vec<f64>, AllegroOamError> {
    ensure_positive(
        "product type embedding",
        &[
            shape.rows,
            shape.num_types,
            shape.radial_dim,
            shape.type_dim,
        ],
    )?;
    let output_dim = shape
        .type_dim
        .checked_mul(2)
        .ok_or_else(|| invalid_model("product type embedding output dimension overflows usize"))?;
    expect_len(
        "product type center types",
        inputs.center_types.len(),
        shape.rows,
    )?;
    expect_len(
        "product type neighbor types",
        inputs.neighbor_types.len(),
        shape.rows,
    )?;
    let radial_len = checked_len(
        "product type radial embedding",
        &[shape.rows, shape.radial_dim],
    )?;
    let type_embed_len = checked_len(
        "product type embedding table",
        &[shape.num_types, shape.type_dim],
    )?;
    let basis_weight_len =
        checked_len("product type basis weight", &[shape.radial_dim, output_dim])?;
    expect_len(
        "product type radial embedding",
        inputs.radial_embedding.len(),
        radial_len,
    )?;
    expect_len(
        "product type center embedding table",
        inputs.center_embed.len(),
        type_embed_len,
    )?;
    expect_len(
        "product type neighbor embedding table",
        inputs.neighbor_embed.len(),
        type_embed_len,
    )?;
    expect_len(
        "product type basis weight",
        inputs.basis_weight.len(),
        basis_weight_len,
    )?;

    let basis = allegro_input_major_linear_forward(
        inputs.radial_embedding,
        inputs.basis_weight,
        inputs.basis_alpha,
        AllegroLinearShape {
            rows: shape.rows,
            in_dim: shape.radial_dim,
            out_dim: output_dim,
        },
    )?;
    let output_len = checked_len("product type output", &[shape.rows, output_dim])?;
    let mut out = vec![0.0; output_len];
    for row in 0..shape.rows {
        let center_type = inputs.center_types[row];
        let neighbor_type = inputs.neighbor_types[row];
        if center_type >= shape.num_types {
            return Err(invalid_model(format!(
                "center type index {center_type} is out of bounds for {} types",
                shape.num_types
            )));
        }
        if neighbor_type >= shape.num_types {
            return Err(invalid_model(format!(
                "neighbor type index {neighbor_type} is out of bounds for {} types",
                shape.num_types
            )));
        }
        for component in 0..shape.type_dim {
            let center_component = inputs.center_embed[center_type * shape.type_dim + component];
            let neighbor_component =
                inputs.neighbor_embed[neighbor_type * shape.type_dim + component];
            out[row * output_dim + component] =
                center_component * basis[row * output_dim + component];
            out[row * output_dim + shape.type_dim + component] =
                neighbor_component * basis[row * output_dim + shape.type_dim + component];
        }
    }
    Ok(out)
}

#[allow(dead_code)]
pub(crate) fn allegro_product_type_embedding_vjp_inputs(
    inputs: AllegroProductTypeEmbeddingInputs<'_>,
    grad_out: &[f64],
    shape: AllegroProductTypeEmbeddingShape,
) -> Result<AllegroProductTypeEmbeddingVjp, AllegroOamError> {
    ensure_positive(
        "product type embedding VJP",
        &[
            shape.rows,
            shape.num_types,
            shape.radial_dim,
            shape.type_dim,
        ],
    )?;
    let output_dim = shape.type_dim.checked_mul(2).ok_or_else(|| {
        invalid_model("product type embedding VJP output dimension overflows usize")
    })?;
    expect_len(
        "product type VJP center types",
        inputs.center_types.len(),
        shape.rows,
    )?;
    expect_len(
        "product type VJP neighbor types",
        inputs.neighbor_types.len(),
        shape.rows,
    )?;
    let radial_len = checked_len(
        "product type VJP radial embedding",
        &[shape.rows, shape.radial_dim],
    )?;
    let type_embed_len = checked_len(
        "product type VJP embedding table",
        &[shape.num_types, shape.type_dim],
    )?;
    let basis_weight_len = checked_len(
        "product type VJP basis weight",
        &[shape.radial_dim, output_dim],
    )?;
    let grad_out_len = checked_len("product type VJP grad_out", &[shape.rows, output_dim])?;
    expect_len(
        "product type VJP radial embedding",
        inputs.radial_embedding.len(),
        radial_len,
    )?;
    expect_len(
        "product type VJP center embedding table",
        inputs.center_embed.len(),
        type_embed_len,
    )?;
    expect_len(
        "product type VJP neighbor embedding table",
        inputs.neighbor_embed.len(),
        type_embed_len,
    )?;
    expect_len(
        "product type VJP basis weight",
        inputs.basis_weight.len(),
        basis_weight_len,
    )?;
    expect_len("product type VJP grad_out", grad_out.len(), grad_out_len)?;

    let mut grad_basis = vec![0.0; grad_out_len];
    for row in 0..shape.rows {
        let center_type = inputs.center_types[row];
        let neighbor_type = inputs.neighbor_types[row];
        if center_type >= shape.num_types {
            return Err(invalid_model(format!(
                "center type index {center_type} is out of bounds for {} types",
                shape.num_types
            )));
        }
        if neighbor_type >= shape.num_types {
            return Err(invalid_model(format!(
                "neighbor type index {neighbor_type} is out of bounds for {} types",
                shape.num_types
            )));
        }
        for component in 0..shape.type_dim {
            let center_component = inputs.center_embed[center_type * shape.type_dim + component];
            let neighbor_component =
                inputs.neighbor_embed[neighbor_type * shape.type_dim + component];
            grad_basis[row * output_dim + component] =
                grad_out[row * output_dim + component] * center_component;
            grad_basis[row * output_dim + shape.type_dim + component] =
                grad_out[row * output_dim + shape.type_dim + component] * neighbor_component;
        }
    }

    let grad_radial = allegro_input_major_linear_vjp_input(
        inputs.basis_weight,
        inputs.basis_alpha,
        &grad_basis,
        AllegroLinearShape {
            rows: shape.rows,
            in_dim: shape.radial_dim,
            out_dim: output_dim,
        },
    )?;
    Ok(AllegroProductTypeEmbeddingVjp {
        radial_embedding: grad_radial,
    })
}

#[allow(dead_code)]
pub(crate) fn allegro_edgewise_reduce_scale_shift_sum_forward(
    edge_energy: &[f64],
    edge_sources: &[usize],
    atom_types: &[usize],
    scales: &[f64],
    shifts: &[f64],
    factor: f64,
    num_atoms: usize,
) -> Result<(Vec<f64>, f64), AllegroOamError> {
    if num_atoms == 0 {
        return Err(invalid_model("edgewise reduce num_atoms must be positive"));
    }
    if !factor.is_finite() {
        return Err(invalid_model(format!(
            "edgewise reduce factor must be finite, got {factor}"
        )));
    }
    expect_len(
        "edgewise reduce edge sources",
        edge_sources.len(),
        edge_energy.len(),
    )?;
    expect_len("edgewise reduce atom types", atom_types.len(), num_atoms)?;
    expect_len("edgewise reduce shifts", shifts.len(), scales.len())?;
    if scales.is_empty() {
        return Err(invalid_model("edgewise reduce scales must be non-empty"));
    }

    let mut atomic_energy = vec![0.0; num_atoms];
    for (edge_index, energy) in edge_energy.iter().enumerate() {
        let src = edge_sources[edge_index];
        if src >= num_atoms {
            return Err(invalid_model(format!(
                "edge source index {src} is out of bounds for {num_atoms} atoms"
            )));
        }
        atomic_energy[src] += *energy * factor;
    }
    let mut total_energy = 0.0;
    for atom in 0..num_atoms {
        let atom_type = atom_types[atom];
        if atom_type >= scales.len() {
            return Err(invalid_model(format!(
                "atom type index {atom_type} is out of bounds for {} types",
                scales.len()
            )));
        }
        let scale = scales[atom_type];
        let shift = shifts[atom_type];
        if !scale.is_finite() || !shift.is_finite() {
            return Err(invalid_model(format!(
                "scale/shift for atom type {atom_type} must be finite"
            )));
        }
        atomic_energy[atom] = atomic_energy[atom] * scale + shift;
        total_energy += atomic_energy[atom];
    }
    Ok((atomic_energy, total_energy))
}

#[allow(dead_code)]
pub(crate) fn allegro_edgewise_reduce_scale_shift_sum_vjp_edge_energy(
    edge_sources: &[usize],
    atom_types: &[usize],
    scales: &[f64],
    factor: f64,
    num_atoms: usize,
    bar_total_energy: f64,
) -> Result<Vec<f64>, AllegroOamError> {
    if num_atoms == 0 {
        return Err(invalid_model(
            "edgewise reduce VJP num_atoms must be positive",
        ));
    }
    if !factor.is_finite() || !bar_total_energy.is_finite() {
        return Err(invalid_model(format!(
            "edgewise reduce VJP factor and bar_total_energy must be finite, got {factor} and {bar_total_energy}"
        )));
    }
    expect_len(
        "edgewise reduce VJP atom types",
        atom_types.len(),
        num_atoms,
    )?;
    if scales.is_empty() {
        return Err(invalid_model(
            "edgewise reduce VJP scales must be non-empty",
        ));
    }

    let mut grad_edge_energy = vec![0.0; edge_sources.len()];
    for (edge_index, src) in edge_sources.iter().copied().enumerate() {
        if src >= num_atoms {
            return Err(invalid_model(format!(
                "edge source index {src} is out of bounds for {num_atoms} atoms"
            )));
        }
        let atom_type = atom_types[src];
        if atom_type >= scales.len() {
            return Err(invalid_model(format!(
                "atom type index {atom_type} is out of bounds for {} types",
                scales.len()
            )));
        }
        let scale = scales[atom_type];
        if !scale.is_finite() {
            return Err(invalid_model(format!(
                "scale for atom type {atom_type} must be finite"
            )));
        }
        grad_edge_energy[edge_index] = bar_total_energy * factor * scale;
    }
    Ok(grad_edge_energy)
}

#[allow(dead_code)]
pub(crate) fn allegro_scatter_center_gather_edge_features_forward(
    edge_features: &[f64],
    edge_sources: &[usize],
    num_nodes: usize,
    multiplicity: usize,
    base_dim: usize,
    scatter_factor: f64,
) -> Result<Vec<f64>, AllegroOamError> {
    if num_nodes == 0 {
        return Err(invalid_model("scatter/gather num_nodes must be positive"));
    }
    ensure_positive("scatter/gather", &[multiplicity, base_dim])?;
    if !scatter_factor.is_finite() {
        return Err(invalid_model(format!(
            "scatter/gather factor must be finite, got {scatter_factor}"
        )));
    }
    let edge_count = edge_sources.len();
    let edge_feature_len = checked_len(
        "scatter/gather edge features",
        &[edge_count, multiplicity, base_dim],
    )?;
    expect_len(
        "scatter/gather edge features",
        edge_features.len(),
        edge_feature_len,
    )?;
    let feature_dim = multiplicity
        .checked_mul(base_dim)
        .ok_or_else(|| invalid_model("scatter/gather feature dimension overflows usize"))?;
    let node_feature_len = checked_len("scatter/gather node features", &[num_nodes, feature_dim])?;
    let mut node_features = vec![0.0; node_feature_len];
    for (edge_index, src) in edge_sources.iter().copied().enumerate() {
        if src >= num_nodes {
            return Err(invalid_model(format!(
                "scatter/gather edge source index {src} is out of bounds for {num_nodes} nodes"
            )));
        }
        let edge_offset = edge_index * feature_dim;
        let node_offset = src * feature_dim;
        for feature in 0..feature_dim {
            node_features[node_offset + feature] +=
                edge_features[edge_offset + feature] * scatter_factor;
        }
    }
    let mut gathered = vec![0.0; edge_feature_len];
    for (edge_index, src) in edge_sources.iter().copied().enumerate() {
        let edge_offset = edge_index * feature_dim;
        let node_offset = src * feature_dim;
        gathered[edge_offset..edge_offset + feature_dim]
            .copy_from_slice(&node_features[node_offset..node_offset + feature_dim]);
    }
    Ok(gathered)
}

#[allow(dead_code)]
pub(crate) fn allegro_scatter_center_gather_edge_features_vjp(
    edge_sources: &[usize],
    num_nodes: usize,
    multiplicity: usize,
    base_dim: usize,
    scatter_factor: f64,
    grad_out: &[f64],
) -> Result<Vec<f64>, AllegroOamError> {
    if num_nodes == 0 {
        return Err(invalid_model(
            "scatter/gather VJP num_nodes must be positive",
        ));
    }
    ensure_positive("scatter/gather VJP", &[multiplicity, base_dim])?;
    if !scatter_factor.is_finite() {
        return Err(invalid_model(format!(
            "scatter/gather VJP factor must be finite, got {scatter_factor}"
        )));
    }
    let edge_count = edge_sources.len();
    let feature_dim = multiplicity
        .checked_mul(base_dim)
        .ok_or_else(|| invalid_model("scatter/gather VJP feature dimension overflows usize"))?;
    let edge_feature_len = checked_len(
        "scatter/gather VJP edge features",
        &[edge_count, feature_dim],
    )?;
    expect_len(
        "scatter/gather VJP grad_out",
        grad_out.len(),
        edge_feature_len,
    )?;

    let node_feature_len = checked_len(
        "scatter/gather VJP node features",
        &[num_nodes, feature_dim],
    )?;
    let mut node_bars = vec![0.0; node_feature_len];
    for (edge_index, src) in edge_sources.iter().copied().enumerate() {
        if src >= num_nodes {
            return Err(invalid_model(format!(
                "scatter/gather VJP edge source index {src} is out of bounds for {num_nodes} nodes"
            )));
        }
        let edge_offset = edge_index * feature_dim;
        let node_offset = src * feature_dim;
        for feature in 0..feature_dim {
            node_bars[node_offset + feature] += grad_out[edge_offset + feature];
        }
    }

    let mut grad_edge_features = vec![0.0; edge_feature_len];
    for (edge_index, src) in edge_sources.iter().copied().enumerate() {
        let edge_offset = edge_index * feature_dim;
        let node_offset = src * feature_dim;
        for feature in 0..feature_dim {
            grad_edge_features[edge_offset + feature] =
                scatter_factor * node_bars[node_offset + feature];
        }
    }
    Ok(grad_edge_features)
}

#[allow(dead_code)]
/// Contracts edge-aligned tensor features with precontracted Allegro Wigner weights.
///
/// The caller must apply Allegro's scatter normalization, node scatter, and
/// edge index-select before passing `x2`; this kernel is only the final
/// `zuij,uijk->zuk` or `zui,uik->zuk` contraction.
pub(crate) fn allegro_contracter_forward_precontracted(
    x1: &[f64],
    x2: &[f64],
    precontracted_ww3j: &[f64],
    shape: AllegroContracterPrecontractedShape,
) -> Result<Vec<f64>, AllegroOamError> {
    ensure_positive(
        "contracter",
        &[
            shape.rows,
            shape.multiplicity,
            shape.base_dim1,
            shape.base_dim2,
            shape.base_dim_out,
        ],
    )?;
    let x1_len = checked_len(
        "contracter x1",
        &[shape.rows, shape.multiplicity, shape.base_dim1],
    )?;
    let x2_base_dim = match shape.mode {
        AllegroContracterMode::Full => shape.base_dim2,
        AllegroContracterMode::Diagonal => {
            if shape.base_dim2 != shape.base_dim1 {
                return Err(invalid_model(format!(
                    "diagonal contracter base_dim2 must equal base_dim1 {}, got {}",
                    shape.base_dim1, shape.base_dim2
                )));
            }
            shape.base_dim1
        }
    };
    let x2_len = checked_len(
        "contracter x2",
        &[shape.rows, shape.multiplicity, x2_base_dim],
    )?;
    let ww3j_len = match shape.mode {
        AllegroContracterMode::Full => checked_len(
            "contracter precontracted_ww3j",
            &[
                shape.multiplicity,
                shape.base_dim1,
                shape.base_dim2,
                shape.base_dim_out,
            ],
        )?,
        AllegroContracterMode::Diagonal => checked_len(
            "contracter precontracted_ww3j",
            &[shape.multiplicity, shape.base_dim1, shape.base_dim_out],
        )?,
    };
    let output_len = checked_len(
        "contracter output",
        &[shape.rows, shape.multiplicity, shape.base_dim_out],
    )?;
    expect_len("contracter x1", x1.len(), x1_len)?;
    expect_len("contracter x2", x2.len(), x2_len)?;
    expect_len(
        "contracter precontracted_ww3j",
        precontracted_ww3j.len(),
        ww3j_len,
    )?;

    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        ),
        all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
    ))]
    if shape.mode == AllegroContracterMode::Full && shape.rows >= ALLEGRO_BLAS_ROW_THRESHOLD {
        return Ok(allegro_contracter_forward_full_blas(
            x1,
            x2,
            precontracted_ww3j,
            shape,
            output_len,
        ));
    }

    let mut out = vec![0.0; output_len];
    for row in 0..shape.rows {
        for channel in 0..shape.multiplicity {
            let out_base = (row * shape.multiplicity + channel) * shape.base_dim_out;
            match shape.mode {
                AllegroContracterMode::Full => {
                    for in1_component in 0..shape.base_dim1 {
                        let x1_index =
                            (row * shape.multiplicity + channel) * shape.base_dim1 + in1_component;
                        let x1_value = x1[x1_index];
                        for in2_component in 0..shape.base_dim2 {
                            let x2_index = (row * shape.multiplicity + channel) * shape.base_dim2
                                + in2_component;
                            let scale = x1_value * x2[x2_index];
                            if scale == 0.0 {
                                continue;
                            }
                            let ww3j_base = ((channel * shape.base_dim1 + in1_component)
                                * shape.base_dim2
                                + in2_component)
                                * shape.base_dim_out;
                            let weights =
                                &precontracted_ww3j[ww3j_base..ww3j_base + shape.base_dim_out];
                            for out_component in 0..shape.base_dim_out {
                                out[out_base + out_component] += scale * weights[out_component];
                            }
                        }
                    }
                }
                AllegroContracterMode::Diagonal => {
                    for out_component in 0..shape.base_dim_out {
                        let mut value = 0.0;
                        for in1_component in 0..shape.base_dim1 {
                            let x1_index = (row * shape.multiplicity + channel) * shape.base_dim1
                                + in1_component;
                            let x2_index = (row * shape.multiplicity + channel) * shape.base_dim1
                                + in1_component;
                            let ww3j_index = (channel * shape.base_dim1 + in1_component)
                                * shape.base_dim_out
                                + out_component;
                            value += x1[x1_index] * x2[x2_index] * precontracted_ww3j[ww3j_index];
                        }
                        out[out_base + out_component] = value;
                    }
                }
            }
        }
    }
    Ok(out)
}

#[allow(dead_code)]
pub(crate) fn allegro_contracter_vjp_inputs_precontracted(
    x1: &[f64],
    x2: &[f64],
    precontracted_ww3j: &[f64],
    grad_out: &[f64],
    shape: AllegroContracterPrecontractedShape,
) -> Result<(Vec<f64>, Vec<f64>), AllegroOamError> {
    ensure_positive(
        "contracter VJP",
        &[
            shape.rows,
            shape.multiplicity,
            shape.base_dim1,
            shape.base_dim2,
            shape.base_dim_out,
        ],
    )?;
    let x1_len = checked_len(
        "contracter VJP x1",
        &[shape.rows, shape.multiplicity, shape.base_dim1],
    )?;
    let x2_base_dim = match shape.mode {
        AllegroContracterMode::Full => shape.base_dim2,
        AllegroContracterMode::Diagonal => {
            if shape.base_dim2 != shape.base_dim1 {
                return Err(invalid_model(format!(
                    "diagonal contracter VJP base_dim2 must equal base_dim1 {}, got {}",
                    shape.base_dim1, shape.base_dim2
                )));
            }
            shape.base_dim1
        }
    };
    let x2_len = checked_len(
        "contracter VJP x2",
        &[shape.rows, shape.multiplicity, x2_base_dim],
    )?;
    let ww3j_len = match shape.mode {
        AllegroContracterMode::Full => checked_len(
            "contracter VJP precontracted_ww3j",
            &[
                shape.multiplicity,
                shape.base_dim1,
                shape.base_dim2,
                shape.base_dim_out,
            ],
        )?,
        AllegroContracterMode::Diagonal => checked_len(
            "contracter VJP precontracted_ww3j",
            &[shape.multiplicity, shape.base_dim1, shape.base_dim_out],
        )?,
    };
    let grad_out_len = checked_len(
        "contracter VJP grad_out",
        &[shape.rows, shape.multiplicity, shape.base_dim_out],
    )?;
    expect_len("contracter VJP x1", x1.len(), x1_len)?;
    expect_len("contracter VJP x2", x2.len(), x2_len)?;
    expect_len(
        "contracter VJP precontracted_ww3j",
        precontracted_ww3j.len(),
        ww3j_len,
    )?;
    expect_len("contracter VJP grad_out", grad_out.len(), grad_out_len)?;

    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        ),
        all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
    ))]
    if shape.mode == AllegroContracterMode::Full && shape.rows >= ALLEGRO_BLAS_ROW_THRESHOLD {
        return Ok(allegro_contracter_vjp_full_blas(
            x1,
            x2,
            precontracted_ww3j,
            grad_out,
            shape,
            x1_len,
            x2_len,
        ));
    }

    let mut grad_x1 = vec![0.0; x1_len];
    let mut grad_x2 = vec![0.0; x2_len];
    for row in 0..shape.rows {
        for channel in 0..shape.multiplicity {
            let out_base = (row * shape.multiplicity + channel) * shape.base_dim_out;
            match shape.mode {
                AllegroContracterMode::Full => {
                    let grad_out_row = &grad_out[out_base..out_base + shape.base_dim_out];
                    for in1_component in 0..shape.base_dim1 {
                        let x1_index =
                            (row * shape.multiplicity + channel) * shape.base_dim1 + in1_component;
                        let x1_value = x1[x1_index];
                        let mut grad_x1_value = 0.0;
                        for in2_component in 0..shape.base_dim2 {
                            let x2_index = (row * shape.multiplicity + channel) * shape.base_dim2
                                + in2_component;
                            let x2_value = x2[x2_index];
                            let ww3j_base = ((channel * shape.base_dim1 + in1_component)
                                * shape.base_dim2
                                + in2_component)
                                * shape.base_dim_out;
                            let weights =
                                &precontracted_ww3j[ww3j_base..ww3j_base + shape.base_dim_out];
                            let mut grad_x2_value = 0.0;
                            for out_component in 0..shape.base_dim_out {
                                let bar_out = grad_out_row[out_component];
                                let weight = weights[out_component];
                                grad_x1_value += bar_out * x2_value * weight;
                                grad_x2_value += bar_out * x1_value * weight;
                            }
                            grad_x2[x2_index] += grad_x2_value;
                        }
                        grad_x1[x1_index] += grad_x1_value;
                    }
                }
                AllegroContracterMode::Diagonal => {
                    for out_component in 0..shape.base_dim_out {
                        let out_index = out_base + out_component;
                        let bar_out = grad_out[out_index];
                        for in1_component in 0..shape.base_dim1 {
                            let x1_index = (row * shape.multiplicity + channel) * shape.base_dim1
                                + in1_component;
                            let x2_index = (row * shape.multiplicity + channel) * shape.base_dim1
                                + in1_component;
                            let ww3j_index = (channel * shape.base_dim1 + in1_component)
                                * shape.base_dim_out
                                + out_component;
                            let weight = precontracted_ww3j[ww3j_index];
                            grad_x1[x1_index] += bar_out * x2[x2_index] * weight;
                            grad_x2[x2_index] += bar_out * x1[x1_index] * weight;
                        }
                    }
                }
            }
        }
    }
    Ok((grad_x1, grad_x2))
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        target_arch = "x86_64"
    ),
    all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
))]
fn allegro_contracter_forward_full_blas(
    x1: &[f64],
    x2: &[f64],
    precontracted_ww3j: &[f64],
    shape: AllegroContracterPrecontractedShape,
    output_len: usize,
) -> Vec<f64> {
    let kron_dim = shape.base_dim1 * shape.base_dim2;
    let x1_row_stride = shape.multiplicity * shape.base_dim1;
    let x2_row_stride = shape.multiplicity * shape.base_dim2;
    let out_row_stride = shape.multiplicity * shape.base_dim_out;
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 {
        let channel_outputs = (0..shape.multiplicity)
            .into_par_iter()
            .map(|channel| {
                let mut kron = vec![0.0; shape.rows * kron_dim];
                let mut channel_out = vec![0.0; shape.rows * shape.base_dim_out];
                for row in 0..shape.rows {
                    let x1_base = row * x1_row_stride + channel * shape.base_dim1;
                    let x2_base = row * x2_row_stride + channel * shape.base_dim2;
                    let kron_base = row * kron_dim;
                    for in1_component in 0..shape.base_dim1 {
                        let x1_value = x1[x1_base + in1_component];
                        let kron_i_base = kron_base + in1_component * shape.base_dim2;
                        for in2_component in 0..shape.base_dim2 {
                            kron[kron_i_base + in2_component] =
                                x1_value * x2[x2_base + in2_component];
                        }
                    }
                }
                let weight_base = channel * kron_dim * shape.base_dim_out;
                blas_backend::dense_forward(
                    &kron,
                    shape.rows,
                    kron_dim,
                    &precontracted_ww3j[weight_base..weight_base + kron_dim * shape.base_dim_out],
                    &mut channel_out,
                    1.0,
                );
                channel_out
            })
            .collect::<Vec<_>>();
        let mut out = vec![0.0; output_len];
        for (channel, channel_out) in channel_outputs.iter().enumerate() {
            for row in 0..shape.rows {
                let source_base = row * shape.base_dim_out;
                let target_base = row * out_row_stride + channel * shape.base_dim_out;
                out[target_base..target_base + shape.base_dim_out]
                    .copy_from_slice(&channel_out[source_base..source_base + shape.base_dim_out]);
            }
        }
        return out;
    }

    let mut kron = vec![0.0; shape.rows * kron_dim];
    let mut out = vec![0.0; output_len];
    for channel in 0..shape.multiplicity {
        for row in 0..shape.rows {
            let x1_base = row * x1_row_stride + channel * shape.base_dim1;
            let x2_base = row * x2_row_stride + channel * shape.base_dim2;
            let kron_base = row * kron_dim;
            for in1_component in 0..shape.base_dim1 {
                let x1_value = x1[x1_base + in1_component];
                let kron_i_base = kron_base + in1_component * shape.base_dim2;
                for in2_component in 0..shape.base_dim2 {
                    kron[kron_i_base + in2_component] = x1_value * x2[x2_base + in2_component];
                }
            }
        }
        let weight_base = channel * kron_dim * shape.base_dim_out;
        let out_base = channel * shape.base_dim_out;
        blas_backend::dense_forward_strided(
            &kron,
            shape.rows,
            kron_dim,
            kron_dim,
            &precontracted_ww3j[weight_base..weight_base + kron_dim * shape.base_dim_out],
            &mut out[out_base..],
            shape.base_dim_out,
            out_row_stride,
            1.0,
            0.0,
        );
    }
    out
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        target_arch = "x86_64"
    ),
    all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
))]
fn allegro_contracter_vjp_full_blas(
    x1: &[f64],
    x2: &[f64],
    precontracted_ww3j: &[f64],
    grad_out: &[f64],
    shape: AllegroContracterPrecontractedShape,
    x1_len: usize,
    x2_len: usize,
) -> (Vec<f64>, Vec<f64>) {
    let kron_dim = shape.base_dim1 * shape.base_dim2;
    let x1_row_stride = shape.multiplicity * shape.base_dim1;
    let x2_row_stride = shape.multiplicity * shape.base_dim2;
    let out_row_stride = shape.multiplicity * shape.base_dim_out;
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 {
        let channel_grads = (0..shape.multiplicity)
            .into_par_iter()
            .map(|channel| {
                let mut grad_kron = vec![0.0; shape.rows * kron_dim];
                let mut grad_x1_channel = vec![0.0; shape.rows * shape.base_dim1];
                let mut grad_x2_channel = vec![0.0; shape.rows * shape.base_dim2];
                let weight_base = channel * kron_dim * shape.base_dim_out;
                let out_base = channel * shape.base_dim_out;
                blas_backend::dense_vjp_input_strided(
                    shape.rows,
                    kron_dim,
                    &precontracted_ww3j[weight_base..weight_base + kron_dim * shape.base_dim_out],
                    shape.base_dim_out,
                    &grad_out[out_base..],
                    out_row_stride,
                    &mut grad_kron,
                    kron_dim,
                    1.0,
                    0.0,
                );
                for row in 0..shape.rows {
                    let x1_base = row * x1_row_stride + channel * shape.base_dim1;
                    let x2_base = row * x2_row_stride + channel * shape.base_dim2;
                    let channel_x1_base = row * shape.base_dim1;
                    let channel_x2_base = row * shape.base_dim2;
                    let kron_base = row * kron_dim;
                    for in1_component in 0..shape.base_dim1 {
                        let x1_value = x1[x1_base + in1_component];
                        let mut grad_x1_value = 0.0;
                        let kron_i_base = kron_base + in1_component * shape.base_dim2;
                        for in2_component in 0..shape.base_dim2 {
                            let grad_pair = grad_kron[kron_i_base + in2_component];
                            grad_x1_value += grad_pair * x2[x2_base + in2_component];
                            grad_x2_channel[channel_x2_base + in2_component] +=
                                grad_pair * x1_value;
                        }
                        grad_x1_channel[channel_x1_base + in1_component] += grad_x1_value;
                    }
                }
                (grad_x1_channel, grad_x2_channel)
            })
            .collect::<Vec<_>>();
        let mut grad_x1 = vec![0.0; x1_len];
        let mut grad_x2 = vec![0.0; x2_len];
        for (channel, (grad_x1_channel, grad_x2_channel)) in channel_grads.iter().enumerate() {
            for row in 0..shape.rows {
                let target_x1_base = row * x1_row_stride + channel * shape.base_dim1;
                let target_x2_base = row * x2_row_stride + channel * shape.base_dim2;
                let source_x1_base = row * shape.base_dim1;
                let source_x2_base = row * shape.base_dim2;
                grad_x1[target_x1_base..target_x1_base + shape.base_dim1].copy_from_slice(
                    &grad_x1_channel[source_x1_base..source_x1_base + shape.base_dim1],
                );
                grad_x2[target_x2_base..target_x2_base + shape.base_dim2].copy_from_slice(
                    &grad_x2_channel[source_x2_base..source_x2_base + shape.base_dim2],
                );
            }
        }
        return (grad_x1, grad_x2);
    }

    let mut grad_kron = vec![0.0; shape.rows * kron_dim];
    let mut grad_x1 = vec![0.0; x1_len];
    let mut grad_x2 = vec![0.0; x2_len];
    for channel in 0..shape.multiplicity {
        grad_kron.fill(0.0);
        let weight_base = channel * kron_dim * shape.base_dim_out;
        let out_base = channel * shape.base_dim_out;
        blas_backend::dense_vjp_input_strided(
            shape.rows,
            kron_dim,
            &precontracted_ww3j[weight_base..weight_base + kron_dim * shape.base_dim_out],
            shape.base_dim_out,
            &grad_out[out_base..],
            out_row_stride,
            &mut grad_kron,
            kron_dim,
            1.0,
            0.0,
        );
        for row in 0..shape.rows {
            let x1_base = row * x1_row_stride + channel * shape.base_dim1;
            let x2_base = row * x2_row_stride + channel * shape.base_dim2;
            let kron_base = row * kron_dim;
            for in1_component in 0..shape.base_dim1 {
                let x1_value = x1[x1_base + in1_component];
                let mut grad_x1_value = 0.0;
                let kron_i_base = kron_base + in1_component * shape.base_dim2;
                for in2_component in 0..shape.base_dim2 {
                    let grad_pair = grad_kron[kron_i_base + in2_component];
                    grad_x1_value += grad_pair * x2[x2_base + in2_component];
                    grad_x2[x2_base + in2_component] += grad_pair * x1_value;
                }
                grad_x1[x1_base + in1_component] += grad_x1_value;
            }
        }
    }
    (grad_x1, grad_x2)
}

#[allow(dead_code)]
fn checked_len(label: &str, dims: &[usize]) -> Result<usize, AllegroOamError> {
    checked_product(dims)
        .map_err(|_| invalid_model(format!("{label} shape product overflows usize")))
}

#[allow(dead_code)]
fn ensure_positive(label: &str, dims: &[usize]) -> Result<(), AllegroOamError> {
    if dims.iter().all(|dim| *dim > 0) {
        Ok(())
    } else {
        Err(invalid_model(format!(
            "{label} dimensions must be positive, got {dims:?}"
        )))
    }
}

#[allow(dead_code)]
fn expect_len(label: &str, actual: usize, expected: usize) -> Result<(), AllegroOamError> {
    if actual == expected {
        Ok(())
    } else {
        Err(invalid_model(format!(
            "{label} length must be {expected}, got {actual}"
        )))
    }
}

fn expect_str(field: &str, actual: &str, expected: &str) -> Result<(), AllegroOamError> {
    if actual == expected {
        Ok(())
    } else {
        Err(invalid_model(format!(
            "{field} must be {expected:?}, got {actual:?}"
        )))
    }
}

fn invalid_model(message: impl Into<String>) -> AllegroOamError {
    AllegroOamError::InvalidModel(message.into())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn minimal_runtime_plan() -> Value {
        serde_json::json!({
            "radial_and_spherical": {
                "edge_norm": {"rmax_recip": {"name": "rmax"}},
                "bessel": {"weights": {"name": "bessel"}},
            },
            "product_type_embeddings": [{
                "center_embed": {"name": "center"},
                "neighbor_embed": {"name": "neighbor"},
                "basis_linear": {
                    "layers": [
                        {"class": "ScalarLinearLayer", "weight": {"name": "type_basis"}}
                    ]
                }
            }],
            "scalar_mlps": [{
                "layers": [
                    {"class": "ScalarLinearLayer", "weight": {"name": "w0"}},
                    {"class": "SiLU"},
                    {"class": "ScalarLinearLayer", "weight": {"name": "w1"}}
                ]
            }],
            "weighted_channels": [{"rtoi": {"name": "rtoi"}}],
            "contracters": [{"precontracted_ww3j": {"name": "contract"}}],
            "per_type_scale_shift": {
                "scales": {"name": "scales"},
                "shifts": {"name": "shifts"}
            }
        })
    }

    fn model_with_runtime_tensors(plan: Value, names: &[String]) -> AllegroOamModel {
        let tensors = names
            .iter()
            .map(|name| {
                (
                    name.clone(),
                    AllegroOamExternalTensor {
                        shape: vec![1],
                        values: vec![0.0],
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        AllegroOamModel {
            selected_model: SELECTED_MODEL.to_owned(),
            type_names: Vec::new(),
            cutoff: EXPECTED_CUTOFF,
            l_max: EXPECTED_L_MAX,
            num_layers: EXPECTED_NUM_LAYERS,
            state_tensor_count: EXPECTED_STATE_TENSOR_COUNT,
            state_tensor_numel: EXPECTED_STATE_TENSOR_NUMEL,
            loaded_tensor_count: tensors.len(),
            loaded_tensor_numel: tensors.len(),
            artifact_sha256: ARTIFACT_SHA256.to_owned(),
            artifact_size_bytes: ARTIFACT_SIZE_BYTES,
            execution_contract_status: "export_plan_pinned_runtime_not_implemented".to_owned(),
            execution_plan: Some(plan),
            tensors,
        }
    }

    #[test]
    fn runtime_required_tensor_names_collects_energy_forward_handoff() {
        let names = runtime_required_tensor_names_from_plan(&minimal_runtime_plan())
            .expect("runtime names must collect");

        assert_eq!(
            names,
            vec![
                "bessel".to_owned(),
                "center".to_owned(),
                "contract".to_owned(),
                "neighbor".to_owned(),
                "rmax".to_owned(),
                "rtoi".to_owned(),
                "scales".to_owned(),
                "shifts".to_owned(),
                "type_basis".to_owned(),
                "w0".to_owned(),
                "w1".to_owned(),
            ]
        );
    }

    #[test]
    fn runtime_tensor_handoff_reports_missing_plan_tensor() {
        let plan = minimal_runtime_plan();
        let names =
            runtime_required_tensor_names_from_plan(&plan).expect("runtime names must collect");
        let model = model_with_runtime_tensors(plan.clone(), &names);
        model
            .validate_runtime_tensor_handoff()
            .expect("complete handoff must validate");

        let mut missing_names = names;
        missing_names.retain(|name| name != "contract");
        let model = model_with_runtime_tensors(plan, &missing_names);
        let err = model
            .validate_runtime_tensor_handoff()
            .expect_err("missing tensor must be rejected");

        assert!(matches!(err, AllegroOamError::InvalidModel(_)));
    }

    #[test]
    fn selection_and_concat_vjps_match_finite_difference() {
        let values = [0.2, -0.4, 0.8, 1.1, -0.7, 0.5, 1.3, -1.5];
        let select_grad = [0.6, -0.9, 1.2, 0.4];
        let select_analytic =
            select_columns_vjp(&select_grad, 2, 4, 1, 2).expect("select VJP must evaluate");
        let mut select_add = vec![0.0; select_analytic.len()];
        select_columns_vjp_add(&select_grad, 2, 4, 1, 2, &mut select_add)
            .expect("select VJP add must evaluate");
        assert_eq!(select_add, select_analytic);
        for (index, actual) in select_analytic.iter().enumerate() {
            let finite_difference = central_difference_vec(&values, index, |candidate| {
                let out = select_columns(candidate, 2, 4, 1, 2).expect("select must evaluate");
                dot(&out, &select_grad)
            });
            assert_close(*actual, finite_difference, 2.0e-9);
        }

        let tensor_values = [
            0.3, -0.6, 1.2, 0.7, -1.1, 0.4, 0.9, -0.2, 1.5, -1.4, 0.8, 0.1,
        ];
        let first_grad = [0.5, -0.8, 1.1, 0.6];
        let first_analytic = first_tensor_component_vjp(&first_grad, 2, 2, 3)
            .expect("first component VJP must evaluate");
        let mut first_add = vec![0.0; first_analytic.len()];
        first_tensor_component_vjp_add(&first_grad, 2, 2, 3, &mut first_add)
            .expect("first component VJP add must evaluate");
        assert_eq!(first_add, first_analytic);
        for (index, actual) in first_analytic.iter().enumerate() {
            let finite_difference = central_difference_vec(&tensor_values, index, |candidate| {
                let out =
                    first_tensor_component(candidate, 2, 2, 3).expect("first component forward");
                dot(&out, &first_grad)
            });
            assert_close(*actual, finite_difference, 2.0e-9);
        }

        let block0 = [0.1, -0.2, 0.3, 0.4];
        let block1 = [0.7, -0.9];
        let concat_grad = [0.5, -0.6, 1.2, 0.8, -1.0, 0.4];
        let concat_analytic =
            concat_blocks_vjp(&concat_grad, 2, &[2, 1]).expect("concat VJP must evaluate");
        let mut concat_add0 = vec![0.0; concat_analytic[0].len()];
        let mut concat_add1 = vec![0.0; concat_analytic[1].len()];
        concat_blocks_vjp_add(
            &concat_grad,
            2,
            &[2, 1],
            &mut [&mut concat_add0, &mut concat_add1],
        )
        .expect("concat VJP add must evaluate");
        assert_eq!(concat_add0, concat_analytic[0]);
        assert_eq!(concat_add1, concat_analytic[1]);
        for (index, actual) in concat_analytic[0].iter().enumerate() {
            let finite_difference = central_difference_vec(&block0, index, |candidate| {
                let out = concat_blocks(&[candidate, &block1], 2).expect("concat must evaluate");
                dot(&out, &concat_grad)
            });
            assert_close(*actual, finite_difference, 2.0e-9);
        }
        for (index, actual) in concat_analytic[1].iter().enumerate() {
            let finite_difference = central_difference_vec(&block1, index, |candidate| {
                let out = concat_blocks(&[&block0, candidate], 2).expect("concat must evaluate");
                dot(&out, &concat_grad)
            });
            assert_close(*actual, finite_difference, 2.0e-9);
        }
    }

    #[test]
    fn make_weighted_channels_matches_hand_check() {
        let edge_attrs = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0];
        let rtoi = [1.0, 0.0, 1.0, 0.0, 2.0, 0.0];
        let weights = [10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0];

        let out = allegro_make_weighted_channels_forward(
            &edge_attrs,
            &weights,
            &rtoi,
            AllegroWeightedChannelsShape {
                rows: 2,
                multiplicity_out: 2,
                num_irreps: 2,
                base_dim: 3,
            },
        )
        .expect("weighted channels must evaluate");

        assert_eq!(
            out,
            vec![20.0, 120.0, 50.0, 60.0, 240.0, 150.0, 7.0, 44.0, 13.0, 21.0, 88.0, 39.0,]
        );
    }

    #[test]
    fn make_weighted_channels_vjp_matches_finite_difference() {
        let edge_attrs = [0.4, -0.8, 1.3, 0.7, -1.1, 0.2];
        let weights = [0.5, -0.3, 0.9, 1.4, -0.6, 0.8, 0.2, -1.0];
        let rtoi = [1.0, 0.0, 0.5, -0.2, 0.3, 1.0];
        let grad_out = [
            0.7, -0.4, 0.2, 1.1, -0.9, 0.3, -0.5, 0.8, 0.6, -1.2, 0.4, 0.9,
        ];
        let shape = AllegroWeightedChannelsShape {
            rows: 2,
            multiplicity_out: 2,
            num_irreps: 2,
            base_dim: 3,
        };

        let analytic = allegro_make_weighted_channels_vjp_inputs(
            &edge_attrs,
            &weights,
            &rtoi,
            &grad_out,
            shape,
        )
        .expect("weighted channels VJP must evaluate");

        for index in 0..edge_attrs.len() {
            let finite_difference = central_difference_vec(&edge_attrs, index, |candidate| {
                let out = allegro_make_weighted_channels_forward(candidate, &weights, &rtoi, shape)
                    .expect("weighted channels forward must evaluate");
                dot(&out, &grad_out)
            });
            assert_close(analytic.edge_attrs[index], finite_difference, 2.0e-9);
        }
        for index in 0..weights.len() {
            let finite_difference = central_difference_vec(&weights, index, |candidate| {
                let out =
                    allegro_make_weighted_channels_forward(&edge_attrs, candidate, &rtoi, shape)
                        .expect("weighted channels forward must evaluate");
                dot(&out, &grad_out)
            });
            assert_close(analytic.weights[index], finite_difference, 2.0e-9);
        }
    }

    #[test]
    fn input_major_linear_matches_hand_check() {
        let input = [2.0, 3.0, 5.0, 7.0];
        let weight = [11.0, 13.0, 17.0, 19.0, 23.0, 29.0];

        let out = allegro_input_major_linear_forward(
            &input,
            &weight,
            0.5,
            AllegroLinearShape {
                rows: 2,
                in_dim: 2,
                out_dim: 3,
            },
        )
        .expect("linear layer must evaluate");

        assert_eq!(out, vec![39.5, 47.5, 60.5, 94.0, 113.0, 144.0]);
    }

    #[test]
    fn input_major_linear_vjp_input_matches_finite_difference() {
        let input = [0.2, -0.4, 1.1, 0.7, -1.3, 0.5];
        let weight = [0.3, -0.2, 0.8, 1.2, -0.6, 0.4];
        let grad_out = [0.9, -0.7, 0.2, 1.1];
        let shape = AllegroLinearShape {
            rows: 2,
            in_dim: 3,
            out_dim: 2,
        };

        let analytic = allegro_input_major_linear_vjp_input(&weight, 0.75, &grad_out, shape)
            .expect("linear VJP must evaluate");

        for (index, actual) in analytic.iter().enumerate() {
            let finite_difference = central_difference_vec(&input, index, |candidate| {
                let out = allegro_input_major_linear_forward(candidate, &weight, 0.75, shape)
                    .expect("linear forward must evaluate");
                dot(&out, &grad_out)
            });
            assert_close(*actual, finite_difference, 1.0e-9);
        }
    }

    #[test]
    fn scalar_mlp_forward_matches_hand_check() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight0 = [1.0, 2.0, 3.0, 4.0];
        let weight1 = [5.0, 6.0];

        let out = allegro_scalar_mlp_forward(
            &input,
            2,
            2,
            &[
                AllegroScalarMlpLayer::Linear {
                    weight: &weight0,
                    in_dim: 2,
                    out_dim: 2,
                    alpha: 1.0,
                },
                AllegroScalarMlpLayer::Silu,
                AllegroScalarMlpLayer::Linear {
                    weight: &weight1,
                    in_dim: 2,
                    out_dim: 1,
                    alpha: 0.5,
                },
            ],
        )
        .expect("scalar MLP must evaluate");

        let hidden0 = [7.0, 10.0, 15.0, 22.0];
        let expected0 = 0.5 * (5.0 * allegro_silu(hidden0[0]) + 6.0 * allegro_silu(hidden0[1]));
        let expected1 = 0.5 * (5.0 * allegro_silu(hidden0[2]) + 6.0 * allegro_silu(hidden0[3]));
        assert_close(out[0], expected0, 1.0e-12);
        assert_close(out[1], expected1, 1.0e-12);
    }

    #[test]
    fn scalar_mlp_vjp_input_matches_finite_difference() {
        let input = [0.25, -0.5, 0.75, 1.25];
        let weight0 = [0.3, -0.7, 1.1, 0.2];
        let weight1 = [0.8, -0.4];
        let grad_out = [0.6, -1.2];
        let layers = [
            AllegroScalarMlpLayer::Linear {
                weight: &weight0,
                in_dim: 2,
                out_dim: 2,
                alpha: 0.9,
            },
            AllegroScalarMlpLayer::Silu,
            AllegroScalarMlpLayer::Linear {
                weight: &weight1,
                in_dim: 2,
                out_dim: 1,
                alpha: 1.3,
            },
        ];

        let analytic = allegro_scalar_mlp_vjp_input(&input, 2, 2, &layers, &grad_out)
            .expect("scalar MLP VJP must evaluate");

        for (index, actual) in analytic.iter().enumerate() {
            let finite_difference = central_difference_vec(&input, index, |candidate| {
                let out = allegro_scalar_mlp_forward(candidate, 2, 2, &layers)
                    .expect("scalar MLP forward must evaluate");
                dot(&out, &grad_out)
            });
            assert_close(*actual, finite_difference, 2.0e-9);
        }
    }

    #[test]
    fn scalar_mlp_traced_vjp_matches_recompute_vjp_and_finite_difference() {
        let input = [0.25, -0.5, 0.75, 1.25];
        let weight0 = [0.3, -0.7, 1.1, 0.2];
        let weight1 = [0.8, -0.4, 0.5, -0.9];
        let weight2 = [1.1, -0.6];
        let grad_out = [0.6, -1.2];
        let layers = [
            AllegroScalarMlpLayer::Linear {
                weight: &weight0,
                in_dim: 2,
                out_dim: 2,
                alpha: 0.9,
            },
            AllegroScalarMlpLayer::Silu,
            AllegroScalarMlpLayer::Linear {
                weight: &weight1,
                in_dim: 2,
                out_dim: 2,
                alpha: 0.7,
            },
            AllegroScalarMlpLayer::Silu,
            AllegroScalarMlpLayer::Linear {
                weight: &weight2,
                in_dim: 2,
                out_dim: 1,
                alpha: 1.3,
            },
        ];

        let (traced_out, trace) = allegro_scalar_mlp_forward_traced(&input, 2, 2, &layers)
            .expect("traced scalar MLP forward must evaluate");
        let plain_out = allegro_scalar_mlp_forward(&input, 2, 2, &layers)
            .expect("plain scalar MLP forward must evaluate");
        assert_eq!(trace.silu_inputs.len(), 2);
        for (actual, expected) in traced_out.iter().zip(&plain_out) {
            assert_close(*actual, *expected, 1.0e-12);
        }

        let traced = allegro_scalar_mlp_vjp_input_from_trace(2, 2, &layers, &trace, &grad_out)
            .expect("traced scalar MLP VJP must evaluate");
        let recomputed = allegro_scalar_mlp_vjp_input(&input, 2, 2, &layers, &grad_out)
            .expect("recomputed scalar MLP VJP must evaluate");
        for (actual, expected) in traced.iter().zip(&recomputed) {
            assert_close(*actual, *expected, 1.0e-12);
        }

        for (index, actual) in traced.iter().enumerate() {
            let finite_difference = central_difference_vec(&input, index, |candidate| {
                let out = allegro_scalar_mlp_forward(candidate, 2, 2, &layers)
                    .expect("scalar MLP forward must evaluate");
                dot(&out, &grad_out)
            });
            assert_close(*actual, finite_difference, 2.0e-9);
        }
    }

    #[test]
    fn scalar_mlp_forward_rejects_layer_dimension_mismatch() {
        let err = allegro_scalar_mlp_forward(
            &[1.0, 2.0],
            1,
            2,
            &[AllegroScalarMlpLayer::Linear {
                weight: &[1.0, 2.0],
                in_dim: 1,
                out_dim: 2,
                alpha: 1.0,
            }],
        )
        .expect_err("dimension mismatch must be rejected");

        assert!(matches!(err, AllegroOamError::InvalidModel(_)));
    }

    #[test]
    fn radial_bessel_embedding_matches_hand_check() {
        let out = allegro_radial_bessel_embedding_forward(
            &[0.0, 0.5, 1.2],
            &[1.0, 2.0],
            6.0,
            AllegroRadialEmbeddingShape {
                rows: 3,
                num_bessels: 2,
            },
        )
        .expect("radial embedding must evaluate");

        let cutoff_half = 0.855_468_75;
        assert_close(out[0], 1.0, 1.0e-12);
        assert_close(out[1], 2.0, 1.0e-12);
        assert_close(out[2], cutoff_half * 2.0 / std::f64::consts::PI, 1.0e-12);
        assert_close(out[3], 0.0, 1.0e-12);
        assert_close(out[4], 0.0, 1.0e-12);
        assert_close(out[5], 0.0, 1.0e-12);
    }

    #[test]
    fn radial_bessel_embedding_vjp_matches_finite_difference() {
        let normed_lengths = [0.2, 0.55, 0.83];
        let bessel_weights = [1.0, 2.0, 3.5];
        let grad_out = [0.3, -0.7, 1.1, -0.4, 0.6, 0.8, 1.2, -0.9, 0.5];
        let shape = AllegroRadialEmbeddingShape {
            rows: 3,
            num_bessels: 3,
        };

        let analytic = allegro_radial_bessel_embedding_vjp_normed_lengths(
            &normed_lengths,
            &bessel_weights,
            8.0,
            &grad_out,
            shape,
        )
        .expect("radial VJP must evaluate");

        for (index, actual) in analytic.iter().enumerate() {
            let finite_difference = central_difference_vec(&normed_lengths, index, |candidate| {
                let out =
                    allegro_radial_bessel_embedding_forward(candidate, &bessel_weights, 8.0, shape)
                        .expect("radial forward must evaluate");
                dot(&out, &grad_out)
            });
            assert_close(*actual, finite_difference, 3.0e-8);
        }
    }

    #[test]
    fn product_type_embedding_matches_hand_check() {
        let radial_embedding = [2.0, 3.0, 5.0, 7.0];
        let center_types = [1, 0];
        let neighbor_types = [2, 1];
        let center_embed = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let neighbor_embed = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let basis_weight = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let out = allegro_product_type_embedding_forward(
            AllegroProductTypeEmbeddingInputs {
                radial_embedding: &radial_embedding,
                center_types: &center_types,
                neighbor_types: &neighbor_types,
                center_embed: &center_embed,
                neighbor_embed: &neighbor_embed,
                basis_weight: &basis_weight,
                basis_alpha: 0.5,
            },
            AllegroProductTypeEmbeddingShape {
                rows: 2,
                num_types: 3,
                radial_dim: 2,
                type_dim: 2,
            },
        )
        .expect("product type embedding must evaluate");

        assert_eq!(
            out,
            vec![25.5, 44.0, 675.0, 960.0, 20.0, 52.0, 960.0, 1520.0,]
        );
    }

    #[test]
    fn product_type_embedding_vjp_matches_finite_difference() {
        let radial_embedding = [0.2, -0.5, 0.7, 1.1];
        let center_types = [1, 0];
        let neighbor_types = [2, 1];
        let center_embed = [1.0, -0.2, 0.3, 0.8, -1.1, 0.5];
        let neighbor_embed = [0.4, -0.7, 1.2, 0.6, -0.3, 0.9];
        let basis_weight = [0.5, -0.4, 0.2, 0.7, -0.6, 0.3, 0.8, -1.0];
        let grad_out = [0.9, -0.5, 0.2, 1.1, -0.8, 0.4, 0.6, -1.2];
        let shape = AllegroProductTypeEmbeddingShape {
            rows: 2,
            num_types: 3,
            radial_dim: 2,
            type_dim: 2,
        };

        let analytic = allegro_product_type_embedding_vjp_inputs(
            AllegroProductTypeEmbeddingInputs {
                radial_embedding: &radial_embedding,
                center_types: &center_types,
                neighbor_types: &neighbor_types,
                center_embed: &center_embed,
                neighbor_embed: &neighbor_embed,
                basis_weight: &basis_weight,
                basis_alpha: 0.75,
            },
            &grad_out,
            shape,
        )
        .expect("product type VJP must evaluate");

        for (index, actual) in analytic.radial_embedding.iter().enumerate() {
            let finite_difference = central_difference_vec(&radial_embedding, index, |candidate| {
                let out = allegro_product_type_embedding_forward(
                    AllegroProductTypeEmbeddingInputs {
                        radial_embedding: candidate,
                        center_types: &center_types,
                        neighbor_types: &neighbor_types,
                        center_embed: &center_embed,
                        neighbor_embed: &neighbor_embed,
                        basis_weight: &basis_weight,
                        basis_alpha: 0.75,
                    },
                    shape,
                )
                .expect("product type forward must evaluate");
                dot(&out, &grad_out)
            });
            assert_close(*actual, finite_difference, 2.0e-9);
        }
    }

    #[test]
    fn product_type_embedding_rejects_out_of_range_type() {
        let err = allegro_product_type_embedding_forward(
            AllegroProductTypeEmbeddingInputs {
                radial_embedding: &[1.0],
                center_types: &[2],
                neighbor_types: &[0],
                center_embed: &[1.0, 2.0],
                neighbor_embed: &[3.0, 4.0],
                basis_weight: &[5.0, 6.0],
                basis_alpha: 1.0,
            },
            AllegroProductTypeEmbeddingShape {
                rows: 1,
                num_types: 2,
                radial_dim: 1,
                type_dim: 1,
            },
        )
        .expect_err("out-of-range type must be rejected");

        assert!(matches!(err, AllegroOamError::InvalidModel(_)));
    }

    #[test]
    fn edgewise_reduce_scale_shift_sum_matches_hand_check() {
        let (atomic, total) = allegro_edgewise_reduce_scale_shift_sum_forward(
            &[1.0, 2.0, 3.0],
            &[0, 1, 0],
            &[1, 0],
            &[10.0, 100.0],
            &[1.0, 2.0],
            0.5,
            2,
        )
        .expect("edgewise reduce must evaluate");

        assert_eq!(atomic, vec![202.0, 11.0]);
        assert_eq!(total, 213.0);
    }

    #[test]
    fn edgewise_reduce_scale_shift_sum_vjp_matches_finite_difference() {
        let edge_energy = [0.4, -0.7, 1.2, 0.9];
        let edge_sources = [0, 1, 0, 2];
        let atom_types = [1, 0, 2];
        let scales = [0.5, -1.25, 2.0];
        let shifts = [0.1, -0.2, 0.3];
        let factor = 0.75;
        let bar_total = -1.4;

        let analytic = allegro_edgewise_reduce_scale_shift_sum_vjp_edge_energy(
            &edge_sources,
            &atom_types,
            &scales,
            factor,
            3,
            bar_total,
        )
        .expect("edgewise reduce VJP must evaluate");

        for (index, actual) in analytic.iter().enumerate() {
            let finite_difference = central_difference_vec(&edge_energy, index, |candidate| {
                let (_, total) = allegro_edgewise_reduce_scale_shift_sum_forward(
                    candidate,
                    &edge_sources,
                    &atom_types,
                    &scales,
                    &shifts,
                    factor,
                    3,
                )
                .expect("edgewise reduce forward must evaluate");
                bar_total * total
            });
            assert_close(*actual, finite_difference, 2.0e-9);
        }
    }

    #[test]
    fn edgewise_reduce_rejects_out_of_range_source() {
        let err = allegro_edgewise_reduce_scale_shift_sum_forward(
            &[1.0],
            &[2],
            &[0],
            &[1.0],
            &[0.0],
            1.0,
            1,
        )
        .expect_err("out-of-range source must be rejected");

        assert!(matches!(err, AllegroOamError::InvalidModel(_)));
    }

    #[test]
    fn scatter_center_gather_edge_features_matches_hand_check() {
        let out = allegro_scatter_center_gather_edge_features_forward(
            &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
            &[0, 1, 0, 1],
            2,
            1,
            2,
            0.5,
        )
        .expect("scatter/gather must evaluate");

        assert_eq!(out, vec![5.5, 11.0, 16.5, 22.0, 5.5, 11.0, 16.5, 22.0]);
    }

    #[test]
    fn scatter_center_gather_edge_features_vjp_matches_finite_difference() {
        let edge_features = [0.3, -0.4, 0.9, 1.2, -0.7, 0.6, 1.5, -1.1];
        let edge_sources = [0, 1, 0, 1];
        let grad_out = [0.8, -0.5, 0.2, 1.0, -0.3, 0.7, 1.1, -0.9];

        let analytic =
            allegro_scatter_center_gather_edge_features_vjp(&edge_sources, 2, 1, 2, 0.5, &grad_out)
                .expect("scatter/gather VJP must evaluate");

        for (index, actual) in analytic.iter().enumerate() {
            let finite_difference = central_difference_vec(&edge_features, index, |candidate| {
                let out = allegro_scatter_center_gather_edge_features_forward(
                    candidate,
                    &edge_sources,
                    2,
                    1,
                    2,
                    0.5,
                )
                .expect("scatter/gather forward must evaluate");
                dot(&out, &grad_out)
            });
            assert_close(*actual, finite_difference, 2.0e-9);
        }
    }

    #[test]
    fn scatter_center_gather_rejects_out_of_range_source() {
        let err =
            allegro_scatter_center_gather_edge_features_forward(&[1.0, 2.0], &[1], 1, 1, 2, 1.0)
                .expect_err("out-of-range source must be rejected");

        assert!(matches!(err, AllegroOamError::InvalidModel(_)));
    }

    #[test]
    fn contracter_forward_precontracted_matches_hand_check_non_diagonal() {
        let x1 = [2.0, 3.0, 11.0, 13.0, 23.0, 29.0, 41.0, 43.0];
        let x2 = [5.0, 7.0, 17.0, 19.0, 31.0, 37.0, 47.0, 53.0];
        let precontracted_ww3j = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
        ];

        let out = allegro_contracter_forward_precontracted(
            &x1,
            &x2,
            &precontracted_ww3j,
            AllegroContracterPrecontractedShape {
                rows: 2,
                multiplicity: 2,
                base_dim1: 2,
                base_dim2: 2,
                base_dim_out: 2,
                mode: AllegroContracterMode::Full,
            },
        )
        .expect("contracter must evaluate");

        assert_eq!(
            out,
            vec![274.0, 334.0, 36_480.0, 45_120.0, 15_272.0, 18_808.0, 345_040.0, 429_040.0,]
        );
    }

    #[test]
    fn contracter_vjp_precontracted_matches_finite_difference_non_diagonal() {
        let x1 = [0.2, -0.3, 0.7, 1.1, -0.5, 0.9, 1.4, -1.2];
        let x2 = [0.6, -0.8, 1.3, -1.5, 0.4, 0.2, -0.7, 1.0];
        let precontracted_ww3j = [
            0.3, -0.2, 0.5, 0.7, -0.4, 0.6, 0.1, -0.9, 1.1, -0.8, 0.2, 0.4, -0.6, 0.9, 0.5, -0.3,
        ];
        let grad_out = [0.4, -1.1, 0.8, 0.6, -0.2, 0.9, 1.2, -0.7];
        let shape = AllegroContracterPrecontractedShape {
            rows: 2,
            multiplicity: 2,
            base_dim1: 2,
            base_dim2: 2,
            base_dim_out: 2,
            mode: AllegroContracterMode::Full,
        };

        let (grad_x1, grad_x2) = allegro_contracter_vjp_inputs_precontracted(
            &x1,
            &x2,
            &precontracted_ww3j,
            &grad_out,
            shape,
        )
        .expect("contracter VJP must evaluate");

        for (index, actual) in grad_x1.iter().enumerate() {
            let finite_difference = central_difference_vec(&x1, index, |candidate| {
                let out = allegro_contracter_forward_precontracted(
                    candidate,
                    &x2,
                    &precontracted_ww3j,
                    shape,
                )
                .expect("contracter forward must evaluate");
                dot(&out, &grad_out)
            });
            assert_close(*actual, finite_difference, 3.0e-9);
        }
        for (index, actual) in grad_x2.iter().enumerate() {
            let finite_difference = central_difference_vec(&x2, index, |candidate| {
                let out = allegro_contracter_forward_precontracted(
                    &x1,
                    candidate,
                    &precontracted_ww3j,
                    shape,
                )
                .expect("contracter forward must evaluate");
                dot(&out, &grad_out)
            });
            assert_close(*actual, finite_difference, 3.0e-9);
        }
    }

    #[test]
    fn contracter_forward_precontracted_matches_hand_check_diagonal() {
        let x1 = [
            2.0, 3.0, 5.0, 17.0, 19.0, 23.0, 41.0, 43.0, 47.0, 67.0, 71.0, 73.0,
        ];
        let x2 = [
            7.0, 11.0, 13.0, 29.0, 31.0, 37.0, 53.0, 59.0, 61.0, 79.0, 83.0, 89.0,
        ];
        let precontracted_ww3j = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
        ];

        let out = allegro_contracter_forward_precontracted(
            &x1,
            &x2,
            &precontracted_ww3j,
            AllegroContracterPrecontractedShape {
                rows: 2,
                multiplicity: 2,
                base_dim1: 3,
                base_dim2: 3,
                base_dim_out: 2,
                mode: AllegroContracterMode::Diagonal,
            },
        )
        .expect("diagonal contracter must evaluate");

        assert_eq!(
            out,
            vec![438.0, 550.0, 65_150.0, 84_480.0, 24_119.0, 31_696.0, 554_570.0, 731_400.0,]
        );
    }

    #[test]
    fn contracter_vjp_precontracted_matches_finite_difference_diagonal() {
        let x1 = [0.2, -0.3, 0.7, 1.1, -0.5, 0.9];
        let x2 = [0.6, -0.8, 1.3, -1.5, 0.4, 0.2];
        let precontracted_ww3j = [0.3, -0.2, 0.5, 0.7, -0.4, 0.6];
        let grad_out = [0.4, -1.1, 0.8, 0.6];
        let shape = AllegroContracterPrecontractedShape {
            rows: 2,
            multiplicity: 1,
            base_dim1: 3,
            base_dim2: 3,
            base_dim_out: 2,
            mode: AllegroContracterMode::Diagonal,
        };

        let (grad_x1, grad_x2) = allegro_contracter_vjp_inputs_precontracted(
            &x1,
            &x2,
            &precontracted_ww3j,
            &grad_out,
            shape,
        )
        .expect("diagonal contracter VJP must evaluate");

        for (index, actual) in grad_x1.iter().enumerate() {
            let finite_difference = central_difference_vec(&x1, index, |candidate| {
                let out = allegro_contracter_forward_precontracted(
                    candidate,
                    &x2,
                    &precontracted_ww3j,
                    shape,
                )
                .expect("diagonal contracter forward must evaluate");
                dot(&out, &grad_out)
            });
            assert_close(*actual, finite_difference, 3.0e-9);
        }
        for (index, actual) in grad_x2.iter().enumerate() {
            let finite_difference = central_difference_vec(&x2, index, |candidate| {
                let out = allegro_contracter_forward_precontracted(
                    &x1,
                    candidate,
                    &precontracted_ww3j,
                    shape,
                )
                .expect("diagonal contracter forward must evaluate");
                dot(&out, &grad_out)
            });
            assert_close(*actual, finite_difference, 3.0e-9);
        }
    }

    #[test]
    fn contracter_rejects_shape_mismatch() {
        let err = allegro_contracter_forward_precontracted(
            &[1.0, 2.0],
            &[3.0, 4.0],
            &[5.0],
            AllegroContracterPrecontractedShape {
                rows: 1,
                multiplicity: 1,
                base_dim1: 2,
                base_dim2: 2,
                base_dim_out: 2,
                mode: AllegroContracterMode::Full,
            },
        )
        .expect_err("short precontracted tensor must be rejected");

        assert!(matches!(err, AllegroOamError::InvalidModel(_)));
    }

    #[test]
    fn allegro_kernel_vjps_reject_shape_mismatches() {
        let linear_err = allegro_input_major_linear_vjp_input(
            &[1.0],
            1.0,
            &[1.0, 2.0],
            AllegroLinearShape {
                rows: 1,
                in_dim: 2,
                out_dim: 1,
            },
        )
        .expect_err("short linear VJP weight must be rejected");
        assert!(matches!(linear_err, AllegroOamError::InvalidModel(_)));

        let mlp_err = allegro_scalar_mlp_vjp_input(
            &[1.0, 2.0],
            1,
            2,
            &[AllegroScalarMlpLayer::Linear {
                weight: &[1.0, 2.0],
                in_dim: 1,
                out_dim: 2,
                alpha: 1.0,
            }],
            &[1.0, 2.0],
        )
        .expect_err("scalar MLP VJP dimension mismatch must be rejected");
        assert!(matches!(mlp_err, AllegroOamError::InvalidModel(_)));

        let weighted_err = allegro_make_weighted_channels_vjp_inputs(
            &[1.0],
            &[1.0],
            &[1.0],
            &[1.0],
            AllegroWeightedChannelsShape {
                rows: 1,
                multiplicity_out: 1,
                num_irreps: 1,
                base_dim: 2,
            },
        )
        .expect_err("weighted channels VJP shape mismatch must be rejected");
        assert!(matches!(weighted_err, AllegroOamError::InvalidModel(_)));

        let reduce_err = allegro_edgewise_reduce_scale_shift_sum_vjp_edge_energy(
            &[0],
            &[1],
            &[1.0],
            1.0,
            1,
            1.0,
        )
        .expect_err("edgewise reduce VJP type mismatch must be rejected");
        assert!(matches!(reduce_err, AllegroOamError::InvalidModel(_)));

        let scatter_err =
            allegro_scatter_center_gather_edge_features_vjp(&[1], 1, 1, 1, 1.0, &[1.0])
                .expect_err("scatter/gather VJP source mismatch must be rejected");
        assert!(matches!(scatter_err, AllegroOamError::InvalidModel(_)));

        let contracter_err = allegro_contracter_vjp_inputs_precontracted(
            &[1.0, 2.0],
            &[3.0, 4.0],
            &[5.0],
            &[1.0, 2.0],
            AllegroContracterPrecontractedShape {
                rows: 1,
                multiplicity: 1,
                base_dim1: 2,
                base_dim2: 2,
                base_dim_out: 2,
                mode: AllegroContracterMode::Full,
            },
        )
        .expect_err("short contracter VJP tensor must be rejected");
        assert!(matches!(contracter_err, AllegroOamError::InvalidModel(_)));
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {actual} to be within {tolerance} of {expected}"
        );
    }

    fn dot(values: &[f64], cotangents: &[f64]) -> f64 {
        assert_eq!(values.len(), cotangents.len());
        values
            .iter()
            .zip(cotangents)
            .map(|(value, cotangent)| value * cotangent)
            .sum()
    }

    fn central_difference_vec<F>(values: &[f64], index: usize, f: F) -> f64
    where
        F: Fn(&[f64]) -> f64,
    {
        let eps = 1.0e-6;
        let mut plus = values.to_vec();
        plus[index] += eps;
        let mut minus = values.to_vec();
        minus[index] -= eps;
        (f(&plus) - f(&minus)) / (2.0 * eps)
    }
}
