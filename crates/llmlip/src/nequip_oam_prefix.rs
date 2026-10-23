//! Artifact-backed NeQUIP-OAM-L runtime contract.
//!
//! This is intentionally narrow: it loads the pinned `nequip-oam-l-v1` export
//! or legacy `nequip-oam-prefix-v1` fixtures and runs the six ConvNet layers
//! plus readout, per-type scale/shift, ZBL, and total-energy tail. Force VJP is
//! currently fixture-validated on pinned non-PBC, partial-PBC, and full-PBC
//! structures. Stress/virial VJP is fixture-validated only for pinned full-PBC
//! structures.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
};

use e3nn_core::{
    radial::distance_vjp_accumulate, CgEntry, CoreError, Irreps, LayoutSegment, ShNormalization,
    SphericalHarmonicsPlan, TensorProductInstruction, TensorProductPlan, Vec3,
};
use serde_json::Value;
use thiserror::Error;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    all_pairs_neighbor_list,
    force_accum::{
        accumulate_edge_displacement_gradient, scatter_edge_vector_vjp,
        stress_from_displacement_gradient, virial_from_displacement_gradient, ForceAccumError,
        Mat3,
    },
    neighbor::cell_list_multi_image_neighbor_list,
    NeighborListError, Prediction, PredictionComputeDtype, PredictionOptions, Structure,
};

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas", feature = "libxsmm"),
        target_os = "linux",
        target_arch = "x86_64"
    )
))]
mod portable_blas {
    pub fn dense_forward(
        input: &[f64],
        rows: usize,
        in_dim: usize,
        weight: &[f64],
        out: &mut [f64],
        alpha: f64,
    ) {
        crate::blas_backend::dense_forward(input, rows, in_dim, weight, out, alpha);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dense_forward_strided_block(
        input: &[f64],
        rows: usize,
        input_stride: usize,
        input_offset: usize,
        block_in_dim: usize,
        weight: &[f64],
        block_out_dim: usize,
        out: &mut [f64],
        output_stride: usize,
        output_offset: usize,
        alpha: f64,
    ) {
        crate::blas_backend::dense_forward_strided(
            &input[input_offset..],
            rows,
            block_in_dim,
            input_stride,
            weight,
            &mut out[output_offset..],
            block_out_dim,
            output_stride,
            alpha,
            1.0,
        );
    }

    pub fn dense_vjp_input(
        rows: usize,
        in_dim: usize,
        weight: &[f64],
        bar_out: &[f64],
        bar_input: &mut [f64],
        alpha: f64,
    ) {
        crate::blas_backend::dense_vjp_input(rows, in_dim, weight, bar_out, bar_input, alpha);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dense_vjp_input_strided_block(
        rows: usize,
        input_stride: usize,
        input_offset: usize,
        block_in_dim: usize,
        weight: &[f64],
        bar_out: &[f64],
        output_stride: usize,
        output_offset: usize,
        block_out_dim: usize,
        bar_input: &mut [f64],
        alpha: f64,
    ) {
        crate::blas_backend::dense_vjp_input_strided(
            rows,
            block_in_dim,
            weight,
            block_out_dim,
            &bar_out[output_offset..],
            output_stride,
            &mut bar_input[input_offset..],
            input_stride,
            alpha,
            1.0,
        );
    }
}

#[derive(Clone, Debug)]
pub struct NequipOamPrefixExternalTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
    pub values_f32: Option<Vec<f32>>,
}

const PINNED_OAM_L_TYPE_NAMES: &[&str] = &[
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Ac", "Th", "Pa", "U", "Np", "Pu",
];
const DISTANCE_EPS: f64 = 1.0e-12;
const OAM_MODEL_SCHEMA_VERSION: &str = "nequip-oam-prefix-model-v1";
const OAM_SELECTED_MODEL: &str = "mir-group/NequIP-OAM-L:0.1";
const OAM_STATE_DICT_FORMAT: &str = "manifest-f32-f64-tensors-v1";
const OAM_ARTIFACT_SHA256: &str =
    "5d01a4fab228abb3cdb6ace0033f93993729956bca6a42234a2a8816825b9a0f";
const OAM_ARTIFACT_SIZE_BYTES: usize = 78_464_590;
const OAM_REFERENCE_NAME: &str = "nequip";
const OAM_REFERENCE_REPOSITORY: &str = "https://github.com/mir-group/nequip";
const OAM_REFERENCE_COMMIT: &str = "604ce73b33f067f1e55b69ed96c9e68b842e1409";
const OAM_REFERENCE_PACKAGE_VERSION: &str = "0.17.1";
const OAM_L_MAX: usize = 3;
const OAM_CUTOFF: f64 = 6.0;
const OAM_NUM_BESSELS: usize = 8;
const OAM_LAYER_COUNT: usize = 6;
const OAM_INPUT_IRREPS: &str = "48x0e";
const OAM_EDGE_ATTR_IRREPS: &str = "1x0e+1x1o+1x2e+1x3o";
const OAM_HIDDEN_IRREPS: &str = "128x0e+64x1o+32x2e+32x3o";
const OAM_CONV_OUTPUT_IRREPS: &str = "256x0e+64x1o+32x2e+32x3o";
const OAM_LAYER0_TP_OUT_IRREPS: &str = "48x0e+48x1o+48x2e+48x3o";
const OAM_LAYER1_TP_OUT_IRREPS: &str = concat!(
    "128x0e+64x0e+32x0e+32x0e+128x1o+64x1o+64x1o+32x1o+32x1o+32x1o+",
    "128x2e+64x2e+64x2e+32x2e+32x2e+32x2e+32x2e+128x3o+64x3o+",
    "32x3o+32x3o+32x3o+32x3o"
);
const OAM_LAYER5_TP_OUT_IRREPS: &str = "128x0e+64x0e+32x0e+32x0e";
const OAM_LAYER1_LINEAR2_IN_IRREPS: &str = "256x0e+352x1o+384x2e+320x3o";
const OAM_LAYER5_LINEAR2_IN_IRREPS: &str = "256x0e";
const OAM_SCATTER_NORM_FACTOR: f64 = 0.126_987_731_701_272_26;
const OAM_EDGE_EMBEDDING_FACTOR: f64 = 0.174_532_925_199_432_95;
const OAM_POLYNOMIAL_CUTOFF_P: f64 = 5.0;
const OAM_EDGE_MLP_ALPHA0: f64 = 0.353_553_384_542_465_2;
const OAM_EDGE_MLP_ALPHA1: f64 = 0.125;
const OAM_READOUT_ALPHA: f64 = 0.088_388_346_135_616_3;
const OAM_GATE_SILU_SCALE: f64 = 1.679_176_792_398_941_8;
const OAM_EMPTY_OUTPUT_MASK: &str =
    "sole_model.model.func.layer5_convnet.equivariant_nonlin.mul.output_mask";
#[cfg(feature = "parallel")]
const OAM_TP_FORWARD_PARALLEL_MIN_EDGES: usize = 512;
#[cfg(feature = "parallel")]
const OAM_TP_VJP_PARALLEL_MIN_EDGES: usize = 512;
#[cfg(feature = "parallel")]
const OAM_TP_VJP_PARALLEL_TARGET_CHUNKS: usize = 16;
#[cfg(feature = "parallel")]
const OAM_COMPACT_LINEAR_PARALLEL_MIN_ROWS: usize = 32;
#[cfg(feature = "parallel")]
const OAM_TP_VJP_DST_OWNED_MAX_EDGES: usize = 1024;
#[cfg(all(
    feature = "parallel",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
const OAM_INPUT_MAJOR_PARALLEL_BLAS_MIN_ROWS: usize = 1024;
#[cfg(all(
    feature = "parallel",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
const OAM_INPUT_MAJOR_PARALLEL_BLAS_TARGET_CHUNKS: usize = 16;
const OAM_TP_RUN_STACK_LIMIT: usize = 128;

#[derive(Debug, Error)]
pub enum NequipOamPrefixError {
    #[error(transparent)]
    Core(#[from] CoreError),

    #[error(transparent)]
    NeighborList(#[from] NeighborListError),

    #[error(transparent)]
    ForceAccum(#[from] ForceAccumError),

    #[error("invalid NeQUIP-OAM prefix model: {0}")]
    InvalidModel(String),

    #[error("invalid NeQUIP-OAM prefix structure: {0}")]
    InvalidStructure(String),

    #[error("non-PBC NeQUIP-OAM prefix structures must not specify a cell")]
    NonPbcCell,

    #[error("NeQUIP-OAM prefix stress/virial requires all three PBC axes to be enabled")]
    StressVirialRequiresFullPbc,

    #[error("unknown species {0:?} for NeQUIP-OAM prefix model")]
    UnknownSpecies(String),

    #[error("NeQUIP-OAM prefix model is missing required state tensor {0:?}")]
    MissingStateTensor(String),

    #[error("NeQUIP-OAM prefix state tensor {name:?} has shape {actual:?}, expected {expected:?}")]
    StateTensorShape {
        name: String,
        actual: Vec<usize>,
        expected: Vec<usize>,
    },

    #[error("prediction compute dtype {0:?} is unsupported for NeQUIP-OAM prefix")]
    UnsupportedComputeDtype(PredictionComputeDtype),
}

#[derive(Clone, Debug)]
pub struct NequipOamPrefixModel {
    type_names: Vec<String>,
    spec: Value,
    tensors: HashMap<String, NequipOamPrefixExternalTensor>,
    cutoff: f64,
    polynomial_cutoff_p: f64,
    polynomial_cutoff_p_f32: f32,
    edge_embedding_factor: f64,
    edge_embedding_factor_f32: f32,
    readout_alpha: f64,
    readout_alpha_f32: f32,
    sh_plan: SphericalHarmonicsPlan,
    layer_plans: Vec<OamLayerPlan>,
    tp_plans: Vec<OamTensorProductPlan>,
    gate_plans: Vec<OamGatePlan>,
    sc_contraction_cache: Arc<Mutex<HashMap<Vec<usize>, Arc<ScalarAttrContractionsByLayer>>>>,
    sc_contraction_f32_cache:
        Arc<Mutex<HashMap<Vec<usize>, Arc<ScalarAttrContractionsByLayerF32>>>>,
}

/// Prepared fixed-structure input for the pinned NeQUIP-OAM-L runtime.
#[derive(Clone, Debug)]
pub struct NequipOamPrefixPreparedInput {
    structure: Structure,
    edge_count: usize,
}

impl NequipOamPrefixPreparedInput {
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.structure.len()
    }

    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }
}

#[derive(Clone, Debug)]
struct NequipOamLayerCache {
    normed_linear1: Vec<f64>,
    hidden_silu_derivative: Vec<f64>,
    edge_mlp: Vec<f64>,
    gate: Option<OamGateCache>,
    gate_input: Vec<f64>,
}

#[derive(Clone, Debug)]
struct NequipOamLayerCacheF32 {
    normed_linear1: Vec<f32>,
    hidden_silu_derivative: Vec<f32>,
    edge_mlp: Vec<f32>,
    gate: Option<OamGateCacheF32>,
    gate_input: Vec<f32>,
}

#[derive(Clone, Debug)]
struct NequipOamPrefixOutput {
    energy: f64,
    forces: Vec<Vec3>,
    stress: Option<Mat3>,
    virial: Option<Mat3>,
}

#[derive(Clone, Debug)]
struct NequipOamPrefixVjp {
    forces: Vec<Vec3>,
    displacement_gradient: Mat3,
}

type NequipOamPrefixOutputTensors = (Vec<Vec3>, Option<Mat3>, Option<Mat3>);
type ScalarAttrContractionsByLayer = Vec<Option<ScalarAttrContractions>>;
type ScalarAttrContractionsByLayerF32 = Vec<Option<ScalarAttrContractionsF32>>;

#[derive(Clone, Debug)]
struct ActiveTypeRows {
    active_rows: Vec<usize>,
    row_type_slots: Vec<usize>,
    #[allow(dead_code)]
    type_row_offsets: Vec<usize>,
    #[allow(dead_code)]
    type_rows: Vec<usize>,
}

#[derive(Clone, Debug)]
struct ScalarAttrContractions {
    input_dim: usize,
    output_dim: usize,
    blocks: Vec<ScalarAttrBlockContraction>,
}

#[derive(Clone, Debug)]
struct ScalarAttrBlockContraction {
    mul_in: usize,
    mul_out: usize,
    ir_dim: usize,
    input_offset: usize,
    output_offset: usize,
    scale: f64,
    values: Vec<f64>,
}

#[derive(Clone, Debug)]
struct ScalarAttrContractionsF32 {
    input_dim: usize,
    output_dim: usize,
    blocks: Vec<ScalarAttrBlockContractionF32>,
}

#[derive(Clone, Debug)]
struct ScalarAttrBlockContractionF32 {
    mul_in: usize,
    mul_out: usize,
    ir_dim: usize,
    input_offset: usize,
    output_offset: usize,
    scale: f32,
    values: Vec<f32>,
}

#[derive(Clone, Debug)]
struct OamTensorProductInstruction {
    in1_start: usize,
    in2_start: usize,
    out_start: usize,
    cg_offset: usize,
    cg_len: usize,
    weight_offset: usize,
    path_weight: f64,
}

#[derive(Clone, Debug)]
struct OamTensorProductRun {
    in1_start: usize,
    in1_step: usize,
    in2_start: usize,
    out_start: usize,
    out_step: usize,
    cg_offset: usize,
    cg_len: usize,
    weight_offset: usize,
    path_weight: f64,
    len: usize,
}

#[derive(Clone, Debug)]
struct OamTensorProductLayoutRun {
    in1_start: usize,
    in1_step: usize,
    in1_m_step: usize,
    in2_start: usize,
    out_start: usize,
    out_step: usize,
    out_m_step: usize,
    cg_offset: usize,
    cg_len: usize,
    weight_offset: usize,
    len: usize,
}

#[derive(Clone, Debug)]
struct OamTensorProductLayoutInstruction {
    in1_start: usize,
    in1_m_step: usize,
    in2_start: usize,
    out_start: usize,
    out_m_step: usize,
    cg_offset: usize,
    cg_len: usize,
    weight_offset: usize,
    path_weight: f64,
}

#[derive(Clone, Debug)]
struct OamTpLayoutMaps {
    original_to_tp: Vec<usize>,
    segment_m_steps: Vec<usize>,
}

#[derive(Clone, Debug)]
struct OamScaledCgEntry {
    i_m: usize,
    j_m: usize,
    k_m: usize,
    scale: f64,
    scale_f32: f32,
}

#[derive(Clone, Debug)]
struct OamGateGroup {
    gate_index: usize,
    output_start: usize,
    component_indices: Vec<usize>,
}

#[derive(Clone, Debug)]
struct OamGatePlan {
    input_dim: usize,
    output_dim: usize,
    scalar_scale: f64,
    gate_scale: f64,
    scalar_indices: Vec<usize>,
    groups: Vec<OamGateGroup>,
}

#[derive(Clone, Debug)]
struct OamGateCache {
    scalar_derivatives: Vec<f64>,
    gate_values: Vec<f64>,
    gate_derivatives: Vec<f64>,
}

#[derive(Clone, Debug)]
struct OamGateCacheF32 {
    scalar_derivatives: Vec<f32>,
    gate_values: Vec<f32>,
    gate_derivatives: Vec<f32>,
}

#[derive(Clone, Debug)]
struct OamLayerPlan {
    edge_mlp_alpha0: f64,
    edge_mlp_alpha1: f64,
    edge_mlp_alpha0_f32: f32,
    edge_mlp_alpha1_f32: f32,
    scatter_norm: f64,
    scatter_norm_f32: f32,
}

#[cfg(feature = "parallel")]
type OamParallelVjpChunkResult<T> = Result<OamParallelVjpChunk<T>, NequipOamPrefixError>;

#[cfg(feature = "parallel")]
#[derive(Debug)]
struct OamParallelVjpChunk<T> {
    edge_start: usize,
    edge_end: usize,
    edge_attr_bars: Vec<T>,
    hidden_silu_bars: Vec<T>,
}

#[derive(Clone, Debug)]
struct OamTensorProductPlan {
    input1_dim: usize,
    input2_dim: usize,
    output_dim: usize,
    weight_numel: usize,
    runs: Vec<OamTensorProductRun>,
    tp_layout_runs: Vec<OamTensorProductLayoutRun>,
    input1_original_to_tp: Vec<usize>,
    output_original_to_tp: Vec<usize>,
    cg_entries: Vec<OamScaledCgEntry>,
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct OamCompactLinearBlock {
    mul_in: usize,
    mul_out: usize,
    ir_dim: usize,
    input_offset: usize,
    output_offset: usize,
    weight_offset: usize,
    path_weight: f64,
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct OamCompactLinearPlan {
    input_dim: usize,
    output_dim: usize,
    weight_numel: usize,
    blocks: Vec<OamCompactLinearBlock>,
}

impl OamTensorProductPlan {
    fn from_plan(plan: TensorProductPlan) -> Result<Self, NequipOamPrefixError> {
        let layouts = plan.checked_layouts()?;
        let input1_layout_maps = oam_tp_layout_maps(&layouts.in1, plan.input1_dim())?;
        let output_layout_maps = oam_tp_layout_maps(&layouts.out, plan.output_dim())?;
        let mut instructions = Vec::with_capacity(plan.instructions.len());
        let mut tp_layout_instructions = Vec::with_capacity(plan.instructions.len());
        for instruction in &plan.instructions {
            let in1_segment = layouts.in1[instruction.in1_segment];
            let in2_segment = layouts.in2[instruction.in2_segment];
            let out_segment = layouts.out[instruction.out_segment];
            let weight_offset = instruction
                .weight_offset
                .ok_or_else(|| invalid_model("OAM tensor product instruction is unweighted"))?;
            instructions.push(OamTensorProductInstruction {
                in1_start: in1_segment.start,
                in2_start: in2_segment.start,
                out_start: out_segment.start,
                cg_offset: instruction.cg_offset,
                cg_len: instruction.cg_len,
                weight_offset,
                path_weight: instruction.path_weight,
            });
            tp_layout_instructions.push(OamTensorProductLayoutInstruction {
                in1_start: input1_layout_maps.original_to_tp[in1_segment.start],
                in1_m_step: input1_layout_maps.segment_m_steps[instruction.in1_segment],
                in2_start: in2_segment.start,
                out_start: output_layout_maps.original_to_tp[out_segment.start],
                out_m_step: output_layout_maps.segment_m_steps[instruction.out_segment],
                cg_offset: instruction.cg_offset,
                cg_len: instruction.cg_len,
                weight_offset,
                path_weight: instruction.path_weight,
            });
        }
        let mut runs = oam_tensor_product_runs(&instructions);
        let mut tp_layout_runs = oam_tensor_product_layout_runs(&tp_layout_instructions);
        let mut cg_entries = Vec::new();
        for run in &mut runs {
            let original_cg_offset = run.cg_offset;
            run.cg_offset = cg_entries.len();
            for cg in &plan.cg_entries[original_cg_offset..original_cg_offset + run.cg_len] {
                let scale = run.path_weight * cg.value;
                cg_entries.push(OamScaledCgEntry {
                    i_m: cg.i_m,
                    j_m: cg.j_m,
                    k_m: cg.k_m,
                    scale,
                    scale_f32: scale as f32,
                });
            }
        }
        for run in &mut tp_layout_runs {
            let matching_run = runs
                .iter()
                .find(|candidate| {
                    candidate.weight_offset == run.weight_offset && candidate.len == run.len
                })
                .ok_or_else(|| invalid_model("missing matching OAM TP layout run"))?;
            run.cg_offset = matching_run.cg_offset;
        }
        Ok(Self {
            input1_dim: plan.input1_dim(),
            input2_dim: plan.input2_dim(),
            output_dim: plan.output_dim(),
            weight_numel: plan.weight_numel,
            runs,
            tp_layout_runs,
            input1_original_to_tp: input1_layout_maps.original_to_tp,
            output_original_to_tp: output_layout_maps.original_to_tp,
            cg_entries,
        })
    }

    #[must_use]
    fn input1_dim(&self) -> usize {
        self.input1_dim
    }

    #[must_use]
    fn input2_dim(&self) -> usize {
        self.input2_dim
    }

    #[must_use]
    fn output_dim(&self) -> usize {
        self.output_dim
    }
}

impl OamGatePlan {
    fn from_json(gate: &Value) -> Result<Self, NequipOamPrefixError> {
        let input_dim = usize_field(gate, "input_dim")?;
        let output_dim = usize_field(gate, "output_dim")?;
        let scalar_scale = f64_field(gate, "scalar_activation_scale")?;
        let gate_scale = f64_field(gate, "gate_activation_scale")?;
        let scalar_indices = usize_array_field(gate, "scalar_indices")?;
        let mut groups = Vec::new();
        for group in array_field(gate, "gated_groups")? {
            groups.push(OamGateGroup {
                gate_index: usize_field(group, "gate_index")?,
                output_start: usize_field(group, "output_start")?,
                component_indices: usize_array_field(group, "component_indices")?,
            });
        }
        Ok(Self {
            input_dim,
            output_dim,
            scalar_scale,
            gate_scale,
            scalar_indices,
            groups,
        })
    }
}

impl OamLayerPlan {
    fn from_json(layer: &Value) -> Result<Self, NequipOamPrefixError> {
        let edge_mlp_alphas = f64_array_field(layer, "edge_mlp_layer_alphas")?;
        if edge_mlp_alphas.len() != 2 {
            return Err(invalid_model("edge_mlp_layer_alphas must have length 2"));
        }
        let edge_mlp_alpha0 = edge_mlp_alphas[0];
        let edge_mlp_alpha1 = edge_mlp_alphas[1];
        let scatter_norm = f64_field(layer, "scatter_norm_factor")?;
        Ok(Self {
            edge_mlp_alpha0,
            edge_mlp_alpha1,
            edge_mlp_alpha0_f32: edge_mlp_alpha0 as f32,
            edge_mlp_alpha1_f32: edge_mlp_alpha1 as f32,
            scatter_norm,
            scatter_norm_f32: scatter_norm as f32,
        })
    }
}

#[derive(Clone, Debug)]
struct OamEdgePlan {
    edges_by_dst: Vec<usize>,
    dst_offsets: Vec<usize>,
    #[cfg(feature = "parallel")]
    edges_by_src: Vec<usize>,
    #[cfg(feature = "parallel")]
    src_offsets: Vec<usize>,
    #[cfg(feature = "parallel")]
    has_src: bool,
}

impl OamEdgePlan {
    fn new(edges: &[crate::Edge], node_count: usize, _include_src: bool) -> Self {
        let (edges_by_dst, dst_offsets) =
            edge_indices_grouped_by_node(edges, node_count, |edge| edge.dst);
        #[cfg(feature = "parallel")]
        let (edges_by_src, src_offsets, has_src) = if _include_src {
            let (edges_by_src, src_offsets) =
                edge_indices_grouped_by_node(edges, node_count, |edge| edge.src);
            (edges_by_src, src_offsets, true)
        } else {
            (Vec::new(), vec![0; node_count + 1], false)
        };
        Self {
            edges_by_dst,
            dst_offsets,
            #[cfg(feature = "parallel")]
            edges_by_src,
            #[cfg(feature = "parallel")]
            src_offsets,
            #[cfg(feature = "parallel")]
            has_src,
        }
    }

    #[cfg(feature = "parallel")]
    fn edges_for_src(&self, src: usize) -> &[usize] {
        debug_assert!(self.has_src);
        &self.edges_by_src[self.src_offsets[src]..self.src_offsets[src + 1]]
    }
}

#[cfg(feature = "parallel")]
fn oam_tp_forward_parallel_enabled(edge_count: usize) -> bool {
    rayon::current_num_threads() > 1 && edge_count >= OAM_TP_FORWARD_PARALLEL_MIN_EDGES
}

fn oam_tensor_product_runs(
    instructions: &[OamTensorProductInstruction],
) -> Vec<OamTensorProductRun> {
    let mut runs = Vec::new();
    let mut start = 0;
    while start < instructions.len() {
        let first = &instructions[start];
        let mut len = 1;
        let mut in1_step = 0;
        let mut out_step = 0;
        if let Some(next) = instructions.get(start + 1) {
            if next.in2_start == first.in2_start
                && next.cg_offset == first.cg_offset
                && next.cg_len == first.cg_len
                && next.path_weight == first.path_weight
                && next.weight_offset == first.weight_offset + 1
                && next.in1_start >= first.in1_start
                && next.out_start >= first.out_start
            {
                in1_step = next.in1_start - first.in1_start;
                out_step = next.out_start - first.out_start;
                len = 2;
                while let Some(instruction) = instructions.get(start + len) {
                    if instruction.in2_start != first.in2_start
                        || instruction.cg_offset != first.cg_offset
                        || instruction.cg_len != first.cg_len
                        || instruction.path_weight != first.path_weight
                        || instruction.weight_offset != first.weight_offset + len
                        || instruction.in1_start != first.in1_start + len * in1_step
                        || instruction.out_start != first.out_start + len * out_step
                    {
                        break;
                    }
                    len += 1;
                }
            }
        }
        runs.push(OamTensorProductRun {
            in1_start: first.in1_start,
            in1_step,
            in2_start: first.in2_start,
            out_start: first.out_start,
            out_step,
            cg_offset: first.cg_offset,
            cg_len: first.cg_len,
            weight_offset: first.weight_offset,
            path_weight: first.path_weight,
            len,
        });
        start += len;
    }
    runs
}

fn oam_tp_layout_maps(
    segments: &[LayoutSegment],
    dim: usize,
) -> Result<OamTpLayoutMaps, NequipOamPrefixError> {
    let mut term_bases = Vec::new();
    let mut term_muls = Vec::new();
    for segment in segments {
        if segment.term_index >= term_bases.len() {
            term_bases.resize(segment.term_index + 1, segment.start);
            term_muls.resize(segment.term_index + 1, 0);
        }
        if term_muls[segment.term_index] == 0 {
            term_bases[segment.term_index] = segment.start;
        }
        term_muls[segment.term_index] += 1;
    }

    let mut original_to_tp = vec![0; dim];
    let mut segment_m_steps = Vec::with_capacity(segments.len());
    for segment in segments {
        let mul = term_muls[segment.term_index];
        segment_m_steps.push(mul);
        for m in 0..segment.len {
            let original_index = segment.start + m;
            let tp_index = term_bases[segment.term_index] + m * mul + segment.mul_index;
            if original_index >= dim || tp_index >= dim {
                return Err(invalid_model("OAM TP layout map index out of bounds"));
            }
            original_to_tp[original_index] = tp_index;
        }
    }
    Ok(OamTpLayoutMaps {
        original_to_tp,
        segment_m_steps,
    })
}

fn oam_tensor_product_layout_runs(
    instructions: &[OamTensorProductLayoutInstruction],
) -> Vec<OamTensorProductLayoutRun> {
    let mut runs = Vec::new();
    let mut start = 0;
    while start < instructions.len() {
        let first = &instructions[start];
        let mut len = 1;
        let mut in1_step = 0;
        let mut out_step = 0;
        if let Some(next) = instructions.get(start + 1) {
            if next.in2_start == first.in2_start
                && next.in1_m_step == first.in1_m_step
                && next.out_m_step == first.out_m_step
                && next.cg_offset == first.cg_offset
                && next.cg_len == first.cg_len
                && next.path_weight == first.path_weight
                && next.weight_offset == first.weight_offset + 1
                && next.in1_start >= first.in1_start
                && next.out_start >= first.out_start
            {
                in1_step = next.in1_start - first.in1_start;
                out_step = next.out_start - first.out_start;
                len = 2;
                while let Some(instruction) = instructions.get(start + len) {
                    if instruction.in2_start != first.in2_start
                        || instruction.in1_m_step != first.in1_m_step
                        || instruction.out_m_step != first.out_m_step
                        || instruction.cg_offset != first.cg_offset
                        || instruction.cg_len != first.cg_len
                        || instruction.path_weight != first.path_weight
                        || instruction.weight_offset != first.weight_offset + len
                        || instruction.in1_start != first.in1_start + len * in1_step
                        || instruction.out_start != first.out_start + len * out_step
                    {
                        break;
                    }
                    len += 1;
                }
            }
        }
        runs.push(OamTensorProductLayoutRun {
            in1_start: first.in1_start,
            in1_step,
            in1_m_step: first.in1_m_step,
            in2_start: first.in2_start,
            out_start: first.out_start,
            out_step,
            out_m_step: first.out_m_step,
            cg_offset: first.cg_offset,
            cg_len: first.cg_len,
            weight_offset: first.weight_offset,
            len,
        });
        start += len;
    }
    runs
}

#[allow(dead_code)]
fn tensor_product_forward_accumulate_oam(
    plan: &OamTensorProductPlan,
    x: &[f64],
    y: &[f64],
    w_edge: &[f64],
    out: &mut [f64],
) {
    debug_assert_eq!(x.len(), plan.input1_dim());
    debug_assert_eq!(y.len(), plan.input2_dim());
    debug_assert_eq!(w_edge.len(), plan.weight_numel);
    debug_assert_eq!(out.len(), plan.output_dim());
    for run in &plan.runs {
        let cg_end = run.cg_offset + run.cg_len;
        let cg_entries = &plan.cg_entries[run.cg_offset..cg_end];
        if run.len == 1 {
            let weight = w_edge[run.weight_offset];
            for cg in cg_entries {
                out[run.out_start + cg.k_m] +=
                    weight * cg.scale * y[run.in2_start + cg.j_m] * x[run.in1_start + cg.i_m];
            }
            continue;
        }
        let weight_end = run.weight_offset + run.len;
        let weights = &w_edge[run.weight_offset..weight_end];
        if run.in1_step == 1 && run.out_step == 1 {
            for cg in cg_entries {
                let y_scale = cg.scale * y[run.in2_start + cg.j_m];
                let x_start = run.in1_start + cg.i_m;
                let out_start = run.out_start + cg.k_m;
                let x_values = &x[x_start..x_start + run.len];
                let out_values = &mut out[out_start..out_start + run.len];
                for ((out_value, &x_value), &weight) in
                    out_values.iter_mut().zip(x_values).zip(weights)
                {
                    *out_value += y_scale * weight * x_value;
                }
            }
            continue;
        }
        if run.out_step == 1 {
            for cg in cg_entries {
                let y_scale = cg.scale * y[run.in2_start + cg.j_m];
                let mut in1_index = run.in1_start + cg.i_m;
                let out_start = run.out_start + cg.k_m;
                let out_values = &mut out[out_start..out_start + run.len];
                for (out_value, &weight) in out_values.iter_mut().zip(weights) {
                    *out_value += y_scale * weight * x[in1_index];
                    in1_index += run.in1_step;
                }
            }
            continue;
        }
        if run.in1_step == 1 {
            for cg in cg_entries {
                let y_scale = cg.scale * y[run.in2_start + cg.j_m];
                let x_start = run.in1_start + cg.i_m;
                let mut out_index = run.out_start + cg.k_m;
                let x_values = &x[x_start..x_start + run.len];
                for (&x_value, &weight) in x_values.iter().zip(weights) {
                    out[out_index] += y_scale * weight * x_value;
                    out_index += run.out_step;
                }
            }
            continue;
        }
        for cg in cg_entries {
            let y_scale = cg.scale * y[run.in2_start + cg.j_m];
            let mut in1_index = run.in1_start + cg.i_m;
            let mut out_index = run.out_start + cg.k_m;
            for &weight in weights {
                out[out_index] += y_scale * weight * x[in1_index];
                in1_index += run.in1_step;
                out_index += run.out_step;
            }
        }
    }
}

#[inline(always)]
fn tensor_product_forward_accumulate_oam_tp_layout(
    plan: &OamTensorProductPlan,
    x: &[f64],
    y: &[f64],
    w_edge: &[f64],
    out: &mut [f64],
) {
    debug_assert_eq!(x.len(), plan.input1_dim());
    debug_assert_eq!(y.len(), plan.input2_dim());
    debug_assert_eq!(w_edge.len(), plan.weight_numel);
    debug_assert_eq!(out.len(), plan.output_dim());
    for run in &plan.tp_layout_runs {
        let cg_end = run.cg_offset + run.cg_len;
        let cg_entries = &plan.cg_entries[run.cg_offset..cg_end];
        let weight_end = run.weight_offset + run.len;
        let weights = &w_edge[run.weight_offset..weight_end];
        if run.in1_step == 1 && run.out_step == 1 {
            for cg in cg_entries {
                let y_scale = cg.scale * y[run.in2_start + cg.j_m];
                let x_start = run.in1_start + cg.i_m * run.in1_m_step;
                let out_start = run.out_start + cg.k_m * run.out_m_step;
                let x_values = &x[x_start..x_start + run.len];
                let out_values = &mut out[out_start..out_start + run.len];
                for ((out_value, &x_value), &weight) in
                    out_values.iter_mut().zip(x_values).zip(weights)
                {
                    *out_value += y_scale * weight * x_value;
                }
            }
            continue;
        }
        if run.out_step == 1 {
            for cg in cg_entries {
                let y_scale = cg.scale * y[run.in2_start + cg.j_m];
                let mut in1_index = run.in1_start + cg.i_m * run.in1_m_step;
                let out_start = run.out_start + cg.k_m * run.out_m_step;
                let out_values = &mut out[out_start..out_start + run.len];
                for (out_value, &weight) in out_values.iter_mut().zip(weights) {
                    *out_value += y_scale * weight * x[in1_index];
                    in1_index += run.in1_step;
                }
            }
            continue;
        }
        if run.in1_step == 1 {
            for cg in cg_entries {
                let y_scale = cg.scale * y[run.in2_start + cg.j_m];
                let x_start = run.in1_start + cg.i_m * run.in1_m_step;
                let mut out_index = run.out_start + cg.k_m * run.out_m_step;
                let x_values = &x[x_start..x_start + run.len];
                for (&x_value, &weight) in x_values.iter().zip(weights) {
                    out[out_index] += y_scale * weight * x_value;
                    out_index += run.out_step;
                }
            }
            continue;
        }
        for cg in cg_entries {
            let y_scale = cg.scale * y[run.in2_start + cg.j_m];
            let mut in1_index = run.in1_start + cg.i_m * run.in1_m_step;
            let mut out_index = run.out_start + cg.k_m * run.out_m_step;
            for &weight in weights {
                out[out_index] += y_scale * weight * x[in1_index];
                in1_index += run.in1_step;
                out_index += run.out_step;
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn tensor_product_forward_scatter_oam_parallel(
    plan: &OamTensorProductPlan,
    edges: &[crate::Edge],
    edge_plan: &OamEdgePlan,
    edge_attrs: &[f64],
    sh_dim: usize,
    normed_linear1: &[f64],
    edge_mlp: &[f64],
    tp_weight_numel: usize,
    tp_scatter: &mut [f64],
) {
    let output_dim = plan.output_dim();
    let input1_dim = plan.input1_dim();
    let node_count = tp_scatter.len() / output_dim;

    tp_scatter
        .par_chunks_mut(output_dim)
        .enumerate()
        .for_each(|(src, scatter_row)| {
            let edge_indices = edge_plan.edges_for_src(src);
            for &edge_index in edge_indices {
                let edge = &edges[edge_index];
                tensor_product_forward_accumulate_oam(
                    plan,
                    &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                    &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                    &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                    scatter_row,
                );
            }
        });
    debug_assert_eq!(node_count + 1, edge_plan.src_offsets.len());
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn tensor_product_forward_scatter_oam_parallel_tp_layout(
    plan: &OamTensorProductPlan,
    edges: &[crate::Edge],
    edge_plan: &OamEdgePlan,
    edge_attrs: &[f64],
    sh_dim: usize,
    normed_linear1: &[f64],
    edge_mlp: &[f64],
    tp_weight_numel: usize,
    tp_scatter: &mut [f64],
) {
    let output_dim = plan.output_dim();
    let input1_dim = plan.input1_dim();
    let node_count = tp_scatter.len() / output_dim;

    tp_scatter
        .par_chunks_mut(output_dim)
        .enumerate()
        .for_each(|(src, scatter_row)| {
            let edge_indices = edge_plan.edges_for_src(src);
            for &edge_index in edge_indices {
                let edge = &edges[edge_index];
                tensor_product_forward_accumulate_oam_tp_layout(
                    plan,
                    &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                    &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                    &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                    scatter_row,
                );
            }
        });
    debug_assert_eq!(node_count + 1, edge_plan.src_offsets.len());
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn tensor_product_forward_scatter_oam_parallel_f32(
    plan: &OamTensorProductPlan,
    edges: &[crate::Edge],
    edge_plan: &OamEdgePlan,
    edge_attrs: &[f32],
    sh_dim: usize,
    normed_linear1: &[f32],
    edge_mlp: &[f32],
    tp_weight_numel: usize,
    tp_scatter: &mut [f32],
) {
    let output_dim = plan.output_dim();
    let input1_dim = plan.input1_dim();
    let node_count = tp_scatter.len() / output_dim;

    tp_scatter
        .par_chunks_mut(output_dim)
        .enumerate()
        .for_each(|(src, scatter_row)| {
            let edge_indices = edge_plan.edges_for_src(src);
            for &edge_index in edge_indices {
                let edge = &edges[edge_index];
                tensor_product_forward_accumulate_oam_f32(
                    plan,
                    &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                    &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                    &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                    scatter_row,
                );
            }
        });
    debug_assert_eq!(node_count + 1, edge_plan.src_offsets.len());
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn tensor_product_forward_scatter_oam_parallel_tp_layout_f32(
    plan: &OamTensorProductPlan,
    edges: &[crate::Edge],
    edge_plan: &OamEdgePlan,
    edge_attrs: &[f32],
    sh_dim: usize,
    normed_linear1: &[f32],
    edge_mlp: &[f32],
    tp_weight_numel: usize,
    tp_scatter: &mut [f32],
) {
    let output_dim = plan.output_dim();
    let input1_dim = plan.input1_dim();
    let node_count = tp_scatter.len() / output_dim;

    tp_scatter
        .par_chunks_mut(output_dim)
        .enumerate()
        .for_each(|(src, scatter_row)| {
            let edge_indices = edge_plan.edges_for_src(src);
            for &edge_index in edge_indices {
                let edge = &edges[edge_index];
                tensor_product_forward_accumulate_oam_tp_layout_f32(
                    plan,
                    &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                    &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                    &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                    scatter_row,
                );
            }
        });
    debug_assert_eq!(node_count + 1, edge_plan.src_offsets.len());
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn tensor_product_vjp_accumulate_oam(
    plan: &OamTensorProductPlan,
    x: &[f64],
    y: &[f64],
    w_edge: &[f64],
    grad_out: &[f64],
    grad_x: &mut [f64],
    grad_y: &mut [f64],
    grad_w_edge: &mut [f64],
) {
    debug_assert_eq!(x.len(), plan.input1_dim());
    debug_assert_eq!(y.len(), plan.input2_dim());
    debug_assert_eq!(w_edge.len(), plan.weight_numel);
    debug_assert_eq!(grad_out.len(), plan.output_dim());
    debug_assert_eq!(grad_x.len(), plan.input1_dim());
    debug_assert_eq!(grad_y.len(), plan.input2_dim());
    debug_assert_eq!(grad_w_edge.len(), plan.weight_numel);
    for run in &plan.runs {
        let cg_end = run.cg_offset + run.cg_len;
        let cg_entries = &plan.cg_entries[run.cg_offset..cg_end];
        let weight_end = run.weight_offset + run.len;
        if run.len == 1 {
            let weight = w_edge[run.weight_offset];
            let mut grad_weight = 0.0;
            for cg in cg_entries {
                let x_index = run.in1_start + cg.i_m;
                let y_index = run.in2_start + cg.j_m;
                let out_index = run.out_start + cg.k_m;
                let scale = cg.scale;
                let bar_out = grad_out[out_index];
                let x_value = x[x_index];
                let y_value = y[y_index];
                let weighted_scale = scale * weight;
                grad_x[x_index] += weighted_scale * y_value * bar_out;
                grad_y[y_index] += weighted_scale * x_value * bar_out;
                grad_weight += scale * x_value * y_value * bar_out;
            }
            grad_w_edge[run.weight_offset] += grad_weight;
            continue;
        }
        if run.len <= 32 {
            tensor_product_vjp_accumulate_oam_stack::<32>(
                run,
                cg_entries,
                x,
                y,
                w_edge,
                grad_out,
                grad_x,
                grad_y,
                grad_w_edge,
            );
        } else if run.len <= 64 {
            tensor_product_vjp_accumulate_oam_stack::<64>(
                run,
                cg_entries,
                x,
                y,
                w_edge,
                grad_out,
                grad_x,
                grad_y,
                grad_w_edge,
            );
        } else if run.len <= OAM_TP_RUN_STACK_LIMIT {
            tensor_product_vjp_accumulate_oam_stack::<OAM_TP_RUN_STACK_LIMIT>(
                run,
                cg_entries,
                x,
                y,
                w_edge,
                grad_out,
                grad_x,
                grad_y,
                grad_w_edge,
            );
        } else {
            for (slot, weight_offset) in (run.weight_offset..weight_end).enumerate() {
                let weight = w_edge[weight_offset];
                let mut grad_weight = 0.0;
                for cg in cg_entries {
                    let x_index = run.in1_start + cg.i_m + slot * run.in1_step;
                    let y_index = run.in2_start + cg.j_m;
                    let out_index = run.out_start + cg.k_m + slot * run.out_step;
                    let scale = cg.scale;
                    let bar_out = grad_out[out_index];
                    let x_value = x[x_index];
                    let y_value = y[y_index];
                    let weighted_scale = scale * weight;
                    grad_x[x_index] += weighted_scale * y_value * bar_out;
                    grad_y[y_index] += weighted_scale * x_value * bar_out;
                    grad_weight += scale * x_value * y_value * bar_out;
                }
                grad_w_edge[weight_offset] += grad_weight;
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn tensor_product_vjp_accumulate_oam_stack<const N: usize>(
    run: &OamTensorProductRun,
    cg_entries: &[OamScaledCgEntry],
    x: &[f64],
    y: &[f64],
    w_edge: &[f64],
    grad_out: &[f64],
    grad_x: &mut [f64],
    grad_y: &mut [f64],
    grad_w_edge: &mut [f64],
) {
    debug_assert!(run.len <= N);
    let weight_end = run.weight_offset + run.len;
    let mut grad_weights = [0.0; N];
    if run.in1_step == 1 && run.out_step == 1 {
        let weights = &w_edge[run.weight_offset..weight_end];
        for cg in cg_entries {
            let y_index = run.in2_start + cg.j_m;
            let scale = cg.scale;
            let y_value = y[y_index];
            let mut grad_y_value = 0.0;
            let x_start = run.in1_start + cg.i_m;
            let out_start = run.out_start + cg.k_m;
            let x_values = &x[x_start..x_start + run.len];
            let grad_x_values = &mut grad_x[x_start..x_start + run.len];
            let grad_out_values = &grad_out[out_start..out_start + run.len];
            for slot in 0..run.len {
                let weight = weights[slot];
                let bar_out = grad_out_values[slot];
                let x_value = x_values[slot];
                let weighted_scale = scale * weight;
                grad_x_values[slot] += weighted_scale * y_value * bar_out;
                grad_y_value += weighted_scale * x_value * bar_out;
                grad_weights[slot] += scale * x_value * y_value * bar_out;
            }
            grad_y[y_index] += grad_y_value;
        }
    } else if run.out_step == 1 {
        let weights = &w_edge[run.weight_offset..weight_end];
        for cg in cg_entries {
            let y_index = run.in2_start + cg.j_m;
            let scale = cg.scale;
            let y_value = y[y_index];
            let mut grad_y_value = 0.0;
            let mut in1_index = run.in1_start + cg.i_m;
            let out_start = run.out_start + cg.k_m;
            let grad_out_values = &grad_out[out_start..out_start + run.len];
            for slot in 0..run.len {
                let weight = weights[slot];
                let bar_out = grad_out_values[slot];
                let x_value = x[in1_index];
                let weighted_scale = scale * weight;
                grad_x[in1_index] += weighted_scale * y_value * bar_out;
                grad_y_value += weighted_scale * x_value * bar_out;
                grad_weights[slot] += scale * x_value * y_value * bar_out;
                in1_index += run.in1_step;
            }
            grad_y[y_index] += grad_y_value;
        }
    } else if run.in1_step == 1 {
        let weights = &w_edge[run.weight_offset..weight_end];
        for cg in cg_entries {
            let y_index = run.in2_start + cg.j_m;
            let scale = cg.scale;
            let y_value = y[y_index];
            let mut grad_y_value = 0.0;
            let x_start = run.in1_start + cg.i_m;
            let mut out_index = run.out_start + cg.k_m;
            let x_values = &x[x_start..x_start + run.len];
            let grad_x_values = &mut grad_x[x_start..x_start + run.len];
            for slot in 0..run.len {
                let weight = weights[slot];
                let bar_out = grad_out[out_index];
                let x_value = x_values[slot];
                let weighted_scale = scale * weight;
                grad_x_values[slot] += weighted_scale * y_value * bar_out;
                grad_y_value += weighted_scale * x_value * bar_out;
                grad_weights[slot] += scale * x_value * y_value * bar_out;
                out_index += run.out_step;
            }
            grad_y[y_index] += grad_y_value;
        }
    } else {
        for cg in cg_entries {
            let y_index = run.in2_start + cg.j_m;
            let scale = cg.scale;
            let y_value = y[y_index];
            let mut grad_y_value = 0.0;
            let mut in1_index = run.in1_start + cg.i_m;
            let mut out_index = run.out_start + cg.k_m;
            for (slot, weight_offset) in (run.weight_offset..weight_end).enumerate() {
                let weight = w_edge[weight_offset];
                let bar_out = grad_out[out_index];
                let x_value = x[in1_index];
                let weighted_scale = scale * weight;
                grad_x[in1_index] += weighted_scale * y_value * bar_out;
                grad_y_value += weighted_scale * x_value * bar_out;
                grad_weights[slot] += scale * x_value * y_value * bar_out;
                in1_index += run.in1_step;
                out_index += run.out_step;
            }
            grad_y[y_index] += grad_y_value;
        }
    }
    for (slot, weight_offset) in (run.weight_offset..weight_end).enumerate() {
        grad_w_edge[weight_offset] += grad_weights[slot];
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn tensor_product_vjp_accumulate_oam_tp_layout(
    plan: &OamTensorProductPlan,
    x: &[f64],
    y: &[f64],
    w_edge: &[f64],
    grad_out: &[f64],
    grad_x: &mut [f64],
    grad_y: &mut [f64],
    grad_w_edge: &mut [f64],
) {
    debug_assert_eq!(x.len(), plan.input1_dim());
    debug_assert_eq!(y.len(), plan.input2_dim());
    debug_assert_eq!(w_edge.len(), plan.weight_numel);
    debug_assert_eq!(grad_out.len(), plan.output_dim());
    debug_assert_eq!(grad_x.len(), plan.input1_dim());
    debug_assert_eq!(grad_y.len(), plan.input2_dim());
    debug_assert_eq!(grad_w_edge.len(), plan.weight_numel);
    for run in &plan.tp_layout_runs {
        let cg_end = run.cg_offset + run.cg_len;
        let cg_entries = &plan.cg_entries[run.cg_offset..cg_end];
        if run.len <= 32 {
            tensor_product_vjp_accumulate_oam_tp_layout_stack::<32>(
                run,
                cg_entries,
                x,
                y,
                w_edge,
                grad_out,
                grad_x,
                grad_y,
                grad_w_edge,
            );
        } else if run.len <= 64 {
            tensor_product_vjp_accumulate_oam_tp_layout_stack::<64>(
                run,
                cg_entries,
                x,
                y,
                w_edge,
                grad_out,
                grad_x,
                grad_y,
                grad_w_edge,
            );
        } else if run.len <= OAM_TP_RUN_STACK_LIMIT {
            tensor_product_vjp_accumulate_oam_tp_layout_stack::<OAM_TP_RUN_STACK_LIMIT>(
                run,
                cg_entries,
                x,
                y,
                w_edge,
                grad_out,
                grad_x,
                grad_y,
                grad_w_edge,
            );
        } else {
            let weight_end = run.weight_offset + run.len;
            for (slot, weight_offset) in (run.weight_offset..weight_end).enumerate() {
                let weight = w_edge[weight_offset];
                let mut grad_weight = 0.0;
                for cg in cg_entries {
                    let x_index = run.in1_start + cg.i_m * run.in1_m_step + slot * run.in1_step;
                    let y_index = run.in2_start + cg.j_m;
                    let out_index = run.out_start + cg.k_m * run.out_m_step + slot * run.out_step;
                    let scale = cg.scale;
                    let bar_out = grad_out[out_index];
                    let x_value = x[x_index];
                    let y_value = y[y_index];
                    let weighted_scale = scale * weight;
                    grad_x[x_index] += weighted_scale * y_value * bar_out;
                    grad_y[y_index] += weighted_scale * x_value * bar_out;
                    grad_weight += scale * x_value * y_value * bar_out;
                }
                grad_w_edge[weight_offset] += grad_weight;
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tensor_product_vjp_accumulate_oam_tp_layout_stack<const N: usize>(
    run: &OamTensorProductLayoutRun,
    cg_entries: &[OamScaledCgEntry],
    x: &[f64],
    y: &[f64],
    w_edge: &[f64],
    grad_out: &[f64],
    grad_x: &mut [f64],
    grad_y: &mut [f64],
    grad_w_edge: &mut [f64],
) {
    debug_assert!(run.len <= N);
    let weight_end = run.weight_offset + run.len;
    let mut grad_weights = [0.0; N];
    let weights = &w_edge[run.weight_offset..weight_end];
    if run.in1_step == 1 && run.out_step == 1 {
        for cg in cg_entries {
            let y_index = run.in2_start + cg.j_m;
            let scale = cg.scale;
            let y_value = y[y_index];
            let mut grad_y_value = 0.0;
            let x_start = run.in1_start + cg.i_m * run.in1_m_step;
            let out_start = run.out_start + cg.k_m * run.out_m_step;
            let x_values = &x[x_start..x_start + run.len];
            let grad_x_values = &mut grad_x[x_start..x_start + run.len];
            let grad_out_values = &grad_out[out_start..out_start + run.len];
            for slot in 0..run.len {
                let weight = weights[slot];
                let bar_out = grad_out_values[slot];
                let x_value = x_values[slot];
                let weighted_scale = scale * weight;
                grad_x_values[slot] += weighted_scale * y_value * bar_out;
                grad_y_value += weighted_scale * x_value * bar_out;
                grad_weights[slot] += scale * x_value * y_value * bar_out;
            }
            grad_y[y_index] += grad_y_value;
        }
    } else if run.out_step == 1 {
        for cg in cg_entries {
            let y_index = run.in2_start + cg.j_m;
            let scale = cg.scale;
            let y_value = y[y_index];
            let mut grad_y_value = 0.0;
            let mut in1_index = run.in1_start + cg.i_m * run.in1_m_step;
            let out_start = run.out_start + cg.k_m * run.out_m_step;
            let grad_out_values = &grad_out[out_start..out_start + run.len];
            for slot in 0..run.len {
                let weight = weights[slot];
                let bar_out = grad_out_values[slot];
                let x_value = x[in1_index];
                let weighted_scale = scale * weight;
                grad_x[in1_index] += weighted_scale * y_value * bar_out;
                grad_y_value += weighted_scale * x_value * bar_out;
                grad_weights[slot] += scale * x_value * y_value * bar_out;
                in1_index += run.in1_step;
            }
            grad_y[y_index] += grad_y_value;
        }
    } else if run.in1_step == 1 {
        for cg in cg_entries {
            let y_index = run.in2_start + cg.j_m;
            let scale = cg.scale;
            let y_value = y[y_index];
            let mut grad_y_value = 0.0;
            let x_start = run.in1_start + cg.i_m * run.in1_m_step;
            let mut out_index = run.out_start + cg.k_m * run.out_m_step;
            let x_values = &x[x_start..x_start + run.len];
            let grad_x_values = &mut grad_x[x_start..x_start + run.len];
            for slot in 0..run.len {
                let weight = weights[slot];
                let bar_out = grad_out[out_index];
                let x_value = x_values[slot];
                let weighted_scale = scale * weight;
                grad_x_values[slot] += weighted_scale * y_value * bar_out;
                grad_y_value += weighted_scale * x_value * bar_out;
                grad_weights[slot] += scale * x_value * y_value * bar_out;
                out_index += run.out_step;
            }
            grad_y[y_index] += grad_y_value;
        }
    } else {
        for cg in cg_entries {
            let y_index = run.in2_start + cg.j_m;
            let scale = cg.scale;
            let y_value = y[y_index];
            let mut grad_y_value = 0.0;
            let mut in1_index = run.in1_start + cg.i_m * run.in1_m_step;
            let mut out_index = run.out_start + cg.k_m * run.out_m_step;
            for slot in 0..run.len {
                let weight = weights[slot];
                let bar_out = grad_out[out_index];
                let x_value = x[in1_index];
                let weighted_scale = scale * weight;
                grad_x[in1_index] += weighted_scale * y_value * bar_out;
                grad_y_value += weighted_scale * x_value * bar_out;
                grad_weights[slot] += scale * x_value * y_value * bar_out;
                in1_index += run.in1_step;
                out_index += run.out_step;
            }
            grad_y[y_index] += grad_y_value;
        }
    }
    for (slot, weight_offset) in (run.weight_offset..weight_end).enumerate() {
        grad_w_edge[weight_offset] += grad_weights[slot];
    }
}

#[allow(dead_code)]
fn tensor_product_forward_accumulate_oam_f32(
    plan: &OamTensorProductPlan,
    x: &[f32],
    y: &[f32],
    w_edge: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), plan.input1_dim());
    debug_assert_eq!(y.len(), plan.input2_dim());
    debug_assert_eq!(w_edge.len(), plan.weight_numel);
    debug_assert_eq!(out.len(), plan.output_dim());
    for run in &plan.runs {
        let cg_end = run.cg_offset + run.cg_len;
        let cg_entries = &plan.cg_entries[run.cg_offset..cg_end];
        if run.len == 1 {
            let weight = w_edge[run.weight_offset];
            for cg in cg_entries {
                out[run.out_start + cg.k_m] +=
                    weight * cg.scale_f32 * y[run.in2_start + cg.j_m] * x[run.in1_start + cg.i_m];
            }
            continue;
        }
        let weight_end = run.weight_offset + run.len;
        let weights = &w_edge[run.weight_offset..weight_end];
        if run.in1_step == 1 && run.out_step == 1 {
            for cg in cg_entries {
                let y_scale = cg.scale_f32 * y[run.in2_start + cg.j_m];
                let x_start = run.in1_start + cg.i_m;
                let out_start = run.out_start + cg.k_m;
                let x_values = &x[x_start..x_start + run.len];
                let out_values = &mut out[out_start..out_start + run.len];
                for ((out_value, &x_value), &weight) in
                    out_values.iter_mut().zip(x_values).zip(weights)
                {
                    *out_value += y_scale * weight * x_value;
                }
            }
            continue;
        }
        if run.out_step == 1 {
            for cg in cg_entries {
                let y_scale = cg.scale_f32 * y[run.in2_start + cg.j_m];
                let mut in1_index = run.in1_start + cg.i_m;
                let out_start = run.out_start + cg.k_m;
                let out_values = &mut out[out_start..out_start + run.len];
                for (out_value, &weight) in out_values.iter_mut().zip(weights) {
                    *out_value += y_scale * weight * x[in1_index];
                    in1_index += run.in1_step;
                }
            }
            continue;
        }
        if run.in1_step == 1 {
            for cg in cg_entries {
                let y_scale = cg.scale_f32 * y[run.in2_start + cg.j_m];
                let x_start = run.in1_start + cg.i_m;
                let mut out_index = run.out_start + cg.k_m;
                let x_values = &x[x_start..x_start + run.len];
                for (&x_value, &weight) in x_values.iter().zip(weights) {
                    out[out_index] += y_scale * weight * x_value;
                    out_index += run.out_step;
                }
            }
            continue;
        }
        for cg in cg_entries {
            let y_scale = cg.scale_f32 * y[run.in2_start + cg.j_m];
            let mut in1_index = run.in1_start + cg.i_m;
            let mut out_index = run.out_start + cg.k_m;
            for &weight in weights {
                out[out_index] += y_scale * weight * x[in1_index];
                in1_index += run.in1_step;
                out_index += run.out_step;
            }
        }
    }
}

#[inline(always)]
fn tensor_product_forward_accumulate_oam_tp_layout_f32(
    plan: &OamTensorProductPlan,
    x: &[f32],
    y: &[f32],
    w_edge: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), plan.input1_dim());
    debug_assert_eq!(y.len(), plan.input2_dim());
    debug_assert_eq!(w_edge.len(), plan.weight_numel);
    debug_assert_eq!(out.len(), plan.output_dim());
    for run in &plan.tp_layout_runs {
        let cg_end = run.cg_offset + run.cg_len;
        let cg_entries = &plan.cg_entries[run.cg_offset..cg_end];
        let weight_end = run.weight_offset + run.len;
        let weights = &w_edge[run.weight_offset..weight_end];
        if run.in1_step == 1 && run.out_step == 1 {
            for cg in cg_entries {
                let y_scale = cg.scale_f32 * y[run.in2_start + cg.j_m];
                let x_start = run.in1_start + cg.i_m * run.in1_m_step;
                let out_start = run.out_start + cg.k_m * run.out_m_step;
                let x_values = &x[x_start..x_start + run.len];
                let out_values = &mut out[out_start..out_start + run.len];
                for ((out_value, &x_value), &weight) in
                    out_values.iter_mut().zip(x_values).zip(weights)
                {
                    *out_value += y_scale * weight * x_value;
                }
            }
            continue;
        }
        if run.out_step == 1 {
            for cg in cg_entries {
                let y_scale = cg.scale_f32 * y[run.in2_start + cg.j_m];
                let mut in1_index = run.in1_start + cg.i_m * run.in1_m_step;
                let out_start = run.out_start + cg.k_m * run.out_m_step;
                let out_values = &mut out[out_start..out_start + run.len];
                for (out_value, &weight) in out_values.iter_mut().zip(weights) {
                    *out_value += y_scale * weight * x[in1_index];
                    in1_index += run.in1_step;
                }
            }
            continue;
        }
        if run.in1_step == 1 {
            for cg in cg_entries {
                let y_scale = cg.scale_f32 * y[run.in2_start + cg.j_m];
                let x_start = run.in1_start + cg.i_m * run.in1_m_step;
                let mut out_index = run.out_start + cg.k_m * run.out_m_step;
                let x_values = &x[x_start..x_start + run.len];
                for (&x_value, &weight) in x_values.iter().zip(weights) {
                    out[out_index] += y_scale * weight * x_value;
                    out_index += run.out_step;
                }
            }
            continue;
        }
        for cg in cg_entries {
            let y_scale = cg.scale_f32 * y[run.in2_start + cg.j_m];
            let mut in1_index = run.in1_start + cg.i_m * run.in1_m_step;
            let mut out_index = run.out_start + cg.k_m * run.out_m_step;
            for &weight in weights {
                out[out_index] += y_scale * weight * x[in1_index];
                in1_index += run.in1_step;
                out_index += run.out_step;
            }
        }
    }
}

#[allow(clippy::too_many_arguments, dead_code)]
fn tensor_product_vjp_accumulate_oam_f32(
    plan: &OamTensorProductPlan,
    x: &[f32],
    y: &[f32],
    w_edge: &[f32],
    grad_out: &[f32],
    grad_x: &mut [f32],
    grad_y: &mut [f32],
    grad_w_edge: &mut [f32],
) {
    debug_assert_eq!(x.len(), plan.input1_dim());
    debug_assert_eq!(y.len(), plan.input2_dim());
    debug_assert_eq!(w_edge.len(), plan.weight_numel);
    debug_assert_eq!(grad_out.len(), plan.output_dim());
    debug_assert_eq!(grad_x.len(), plan.input1_dim());
    debug_assert_eq!(grad_y.len(), plan.input2_dim());
    debug_assert_eq!(grad_w_edge.len(), plan.weight_numel);
    for run in &plan.runs {
        let cg_end = run.cg_offset + run.cg_len;
        let cg_entries = &plan.cg_entries[run.cg_offset..cg_end];
        let weight_end = run.weight_offset + run.len;
        if run.len == 1 {
            let weight = w_edge[run.weight_offset];
            let mut grad_weight = 0.0;
            for cg in cg_entries {
                let x_index = run.in1_start + cg.i_m;
                let y_index = run.in2_start + cg.j_m;
                let out_index = run.out_start + cg.k_m;
                let scale = cg.scale_f32;
                let bar_out = grad_out[out_index];
                let x_value = x[x_index];
                let y_value = y[y_index];
                let weighted_scale = scale * weight;
                grad_x[x_index] += weighted_scale * y_value * bar_out;
                grad_y[y_index] += weighted_scale * x_value * bar_out;
                grad_weight += scale * x_value * y_value * bar_out;
            }
            grad_w_edge[run.weight_offset] += grad_weight;
            continue;
        }
        if run.len <= 32 {
            tensor_product_vjp_accumulate_oam_stack_f32::<32>(
                run,
                cg_entries,
                x,
                y,
                w_edge,
                grad_out,
                grad_x,
                grad_y,
                grad_w_edge,
            );
        } else if run.len <= 64 {
            tensor_product_vjp_accumulate_oam_stack_f32::<64>(
                run,
                cg_entries,
                x,
                y,
                w_edge,
                grad_out,
                grad_x,
                grad_y,
                grad_w_edge,
            );
        } else if run.len <= OAM_TP_RUN_STACK_LIMIT {
            tensor_product_vjp_accumulate_oam_stack_f32::<OAM_TP_RUN_STACK_LIMIT>(
                run,
                cg_entries,
                x,
                y,
                w_edge,
                grad_out,
                grad_x,
                grad_y,
                grad_w_edge,
            );
        } else {
            for (slot, weight_offset) in (run.weight_offset..weight_end).enumerate() {
                let weight = w_edge[weight_offset];
                let mut grad_weight = 0.0;
                for cg in cg_entries {
                    let x_index = run.in1_start + cg.i_m + slot * run.in1_step;
                    let y_index = run.in2_start + cg.j_m;
                    let out_index = run.out_start + cg.k_m + slot * run.out_step;
                    let scale = cg.scale_f32;
                    let bar_out = grad_out[out_index];
                    let x_value = x[x_index];
                    let y_value = y[y_index];
                    let weighted_scale = scale * weight;
                    grad_x[x_index] += weighted_scale * y_value * bar_out;
                    grad_y[y_index] += weighted_scale * x_value * bar_out;
                    grad_weight += scale * x_value * y_value * bar_out;
                }
                grad_w_edge[weight_offset] += grad_weight;
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tensor_product_vjp_accumulate_oam_stack_f32<const N: usize>(
    run: &OamTensorProductRun,
    cg_entries: &[OamScaledCgEntry],
    x: &[f32],
    y: &[f32],
    w_edge: &[f32],
    grad_out: &[f32],
    grad_x: &mut [f32],
    grad_y: &mut [f32],
    grad_w_edge: &mut [f32],
) {
    debug_assert!(run.len <= N);
    let weight_end = run.weight_offset + run.len;
    let mut grad_weights = [0.0; N];
    if run.in1_step == 1 && run.out_step == 1 {
        let weights = &w_edge[run.weight_offset..weight_end];
        for cg in cg_entries {
            let y_index = run.in2_start + cg.j_m;
            let scale = cg.scale_f32;
            let y_value = y[y_index];
            let mut grad_y_value = 0.0;
            let x_start = run.in1_start + cg.i_m;
            let out_start = run.out_start + cg.k_m;
            let x_values = &x[x_start..x_start + run.len];
            let grad_x_values = &mut grad_x[x_start..x_start + run.len];
            let grad_out_values = &grad_out[out_start..out_start + run.len];
            for slot in 0..run.len {
                let weight = weights[slot];
                let bar_out = grad_out_values[slot];
                let x_value = x_values[slot];
                let weighted_scale = scale * weight;
                grad_x_values[slot] += weighted_scale * y_value * bar_out;
                grad_y_value += weighted_scale * x_value * bar_out;
                grad_weights[slot] += scale * x_value * y_value * bar_out;
            }
            grad_y[y_index] += grad_y_value;
        }
    } else if run.out_step == 1 {
        let weights = &w_edge[run.weight_offset..weight_end];
        for cg in cg_entries {
            let y_index = run.in2_start + cg.j_m;
            let scale = cg.scale_f32;
            let y_value = y[y_index];
            let mut grad_y_value = 0.0;
            let mut in1_index = run.in1_start + cg.i_m;
            let out_start = run.out_start + cg.k_m;
            let grad_out_values = &grad_out[out_start..out_start + run.len];
            for slot in 0..run.len {
                let weight = weights[slot];
                let bar_out = grad_out_values[slot];
                let x_value = x[in1_index];
                let weighted_scale = scale * weight;
                grad_x[in1_index] += weighted_scale * y_value * bar_out;
                grad_y_value += weighted_scale * x_value * bar_out;
                grad_weights[slot] += scale * x_value * y_value * bar_out;
                in1_index += run.in1_step;
            }
            grad_y[y_index] += grad_y_value;
        }
    } else if run.in1_step == 1 {
        let weights = &w_edge[run.weight_offset..weight_end];
        for cg in cg_entries {
            let y_index = run.in2_start + cg.j_m;
            let scale = cg.scale_f32;
            let y_value = y[y_index];
            let mut grad_y_value = 0.0;
            let x_start = run.in1_start + cg.i_m;
            let mut out_index = run.out_start + cg.k_m;
            let x_values = &x[x_start..x_start + run.len];
            let grad_x_values = &mut grad_x[x_start..x_start + run.len];
            for slot in 0..run.len {
                let weight = weights[slot];
                let bar_out = grad_out[out_index];
                let x_value = x_values[slot];
                let weighted_scale = scale * weight;
                grad_x_values[slot] += weighted_scale * y_value * bar_out;
                grad_y_value += weighted_scale * x_value * bar_out;
                grad_weights[slot] += scale * x_value * y_value * bar_out;
                out_index += run.out_step;
            }
            grad_y[y_index] += grad_y_value;
        }
    } else {
        for cg in cg_entries {
            let y_index = run.in2_start + cg.j_m;
            let scale = cg.scale_f32;
            let y_value = y[y_index];
            let mut grad_y_value = 0.0;
            let mut in1_index = run.in1_start + cg.i_m;
            let mut out_index = run.out_start + cg.k_m;
            for (slot, weight_offset) in (run.weight_offset..weight_end).enumerate() {
                let weight = w_edge[weight_offset];
                let bar_out = grad_out[out_index];
                let x_value = x[in1_index];
                let weighted_scale = scale * weight;
                grad_x[in1_index] += weighted_scale * y_value * bar_out;
                grad_y_value += weighted_scale * x_value * bar_out;
                grad_weights[slot] += scale * x_value * y_value * bar_out;
                in1_index += run.in1_step;
                out_index += run.out_step;
            }
            grad_y[y_index] += grad_y_value;
        }
    }
    for (slot, weight_offset) in (run.weight_offset..weight_end).enumerate() {
        grad_w_edge[weight_offset] += grad_weights[slot];
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn tensor_product_vjp_accumulate_oam_tp_layout_f32(
    plan: &OamTensorProductPlan,
    x: &[f32],
    y: &[f32],
    w_edge: &[f32],
    grad_out: &[f32],
    grad_x: &mut [f32],
    grad_y: &mut [f32],
    grad_w_edge: &mut [f32],
) {
    debug_assert_eq!(x.len(), plan.input1_dim());
    debug_assert_eq!(y.len(), plan.input2_dim());
    debug_assert_eq!(w_edge.len(), plan.weight_numel);
    debug_assert_eq!(grad_out.len(), plan.output_dim());
    debug_assert_eq!(grad_x.len(), plan.input1_dim());
    debug_assert_eq!(grad_y.len(), plan.input2_dim());
    debug_assert_eq!(grad_w_edge.len(), plan.weight_numel);
    for run in &plan.tp_layout_runs {
        let cg_end = run.cg_offset + run.cg_len;
        let cg_entries = &plan.cg_entries[run.cg_offset..cg_end];
        if run.len <= 32 {
            tensor_product_vjp_accumulate_oam_tp_layout_stack_f32::<32>(
                run,
                cg_entries,
                x,
                y,
                w_edge,
                grad_out,
                grad_x,
                grad_y,
                grad_w_edge,
            );
        } else if run.len <= 64 {
            tensor_product_vjp_accumulate_oam_tp_layout_stack_f32::<64>(
                run,
                cg_entries,
                x,
                y,
                w_edge,
                grad_out,
                grad_x,
                grad_y,
                grad_w_edge,
            );
        } else if run.len <= OAM_TP_RUN_STACK_LIMIT {
            tensor_product_vjp_accumulate_oam_tp_layout_stack_f32::<OAM_TP_RUN_STACK_LIMIT>(
                run,
                cg_entries,
                x,
                y,
                w_edge,
                grad_out,
                grad_x,
                grad_y,
                grad_w_edge,
            );
        } else {
            let weight_end = run.weight_offset + run.len;
            for (slot, weight_offset) in (run.weight_offset..weight_end).enumerate() {
                let weight = w_edge[weight_offset];
                let mut grad_weight = 0.0;
                for cg in cg_entries {
                    let x_index = run.in1_start + cg.i_m * run.in1_m_step + slot * run.in1_step;
                    let y_index = run.in2_start + cg.j_m;
                    let out_index = run.out_start + cg.k_m * run.out_m_step + slot * run.out_step;
                    let scale = cg.scale_f32;
                    let bar_out = grad_out[out_index];
                    let x_value = x[x_index];
                    let y_value = y[y_index];
                    let weighted_scale = scale * weight;
                    grad_x[x_index] += weighted_scale * y_value * bar_out;
                    grad_y[y_index] += weighted_scale * x_value * bar_out;
                    grad_weight += scale * x_value * y_value * bar_out;
                }
                grad_w_edge[weight_offset] += grad_weight;
            }
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tensor_product_vjp_accumulate_oam_tp_layout_stack_f32<const N: usize>(
    run: &OamTensorProductLayoutRun,
    cg_entries: &[OamScaledCgEntry],
    x: &[f32],
    y: &[f32],
    w_edge: &[f32],
    grad_out: &[f32],
    grad_x: &mut [f32],
    grad_y: &mut [f32],
    grad_w_edge: &mut [f32],
) {
    debug_assert!(run.len <= N);
    let weight_end = run.weight_offset + run.len;
    let mut grad_weights = [0.0; N];
    let weights = &w_edge[run.weight_offset..weight_end];
    if run.in1_step == 1 && run.out_step == 1 {
        for cg in cg_entries {
            let y_index = run.in2_start + cg.j_m;
            let scale = cg.scale_f32;
            let y_value = y[y_index];
            let mut grad_y_value = 0.0;
            let x_start = run.in1_start + cg.i_m * run.in1_m_step;
            let out_start = run.out_start + cg.k_m * run.out_m_step;
            let x_values = &x[x_start..x_start + run.len];
            let grad_x_values = &mut grad_x[x_start..x_start + run.len];
            let grad_out_values = &grad_out[out_start..out_start + run.len];
            for slot in 0..run.len {
                let weight = weights[slot];
                let bar_out = grad_out_values[slot];
                let x_value = x_values[slot];
                let weighted_scale = scale * weight;
                grad_x_values[slot] += weighted_scale * y_value * bar_out;
                grad_y_value += weighted_scale * x_value * bar_out;
                grad_weights[slot] += scale * x_value * y_value * bar_out;
            }
            grad_y[y_index] += grad_y_value;
        }
    } else if run.out_step == 1 {
        for cg in cg_entries {
            let y_index = run.in2_start + cg.j_m;
            let scale = cg.scale_f32;
            let y_value = y[y_index];
            let mut grad_y_value = 0.0;
            let mut in1_index = run.in1_start + cg.i_m * run.in1_m_step;
            let out_start = run.out_start + cg.k_m * run.out_m_step;
            let grad_out_values = &grad_out[out_start..out_start + run.len];
            for slot in 0..run.len {
                let weight = weights[slot];
                let bar_out = grad_out_values[slot];
                let x_value = x[in1_index];
                let weighted_scale = scale * weight;
                grad_x[in1_index] += weighted_scale * y_value * bar_out;
                grad_y_value += weighted_scale * x_value * bar_out;
                grad_weights[slot] += scale * x_value * y_value * bar_out;
                in1_index += run.in1_step;
            }
            grad_y[y_index] += grad_y_value;
        }
    } else if run.in1_step == 1 {
        for cg in cg_entries {
            let y_index = run.in2_start + cg.j_m;
            let scale = cg.scale_f32;
            let y_value = y[y_index];
            let mut grad_y_value = 0.0;
            let x_start = run.in1_start + cg.i_m * run.in1_m_step;
            let mut out_index = run.out_start + cg.k_m * run.out_m_step;
            let x_values = &x[x_start..x_start + run.len];
            let grad_x_values = &mut grad_x[x_start..x_start + run.len];
            for slot in 0..run.len {
                let weight = weights[slot];
                let bar_out = grad_out[out_index];
                let x_value = x_values[slot];
                let weighted_scale = scale * weight;
                grad_x_values[slot] += weighted_scale * y_value * bar_out;
                grad_y_value += weighted_scale * x_value * bar_out;
                grad_weights[slot] += scale * x_value * y_value * bar_out;
                out_index += run.out_step;
            }
            grad_y[y_index] += grad_y_value;
        }
    } else {
        for cg in cg_entries {
            let y_index = run.in2_start + cg.j_m;
            let scale = cg.scale_f32;
            let y_value = y[y_index];
            let mut grad_y_value = 0.0;
            let mut in1_index = run.in1_start + cg.i_m * run.in1_m_step;
            let mut out_index = run.out_start + cg.k_m * run.out_m_step;
            for slot in 0..run.len {
                let weight = weights[slot];
                let bar_out = grad_out[out_index];
                let x_value = x[in1_index];
                let weighted_scale = scale * weight;
                grad_x[in1_index] += weighted_scale * y_value * bar_out;
                grad_y_value += weighted_scale * x_value * bar_out;
                grad_weights[slot] += scale * x_value * y_value * bar_out;
                in1_index += run.in1_step;
                out_index += run.out_step;
            }
            grad_y[y_index] += grad_y_value;
        }
    }
    for (slot, weight_offset) in (run.weight_offset..weight_end).enumerate() {
        grad_w_edge[weight_offset] += grad_weights[slot];
    }
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn tensor_product_vjp_scatter_oam_parallel_direct(
    plan: &OamTensorProductPlan,
    edges: &[crate::Edge],
    edge_attrs: &[f64],
    sh_dim: usize,
    normed_linear1: &[f64],
    edge_mlp: &[f64],
    tp_weight_numel: usize,
    tp_scatter_bars: &[f64],
    normed_linear1_bars: &mut [f64],
    edge_attr_bars: &mut [f64],
    edge_mlp_bars: &mut [f64],
    num_nodes: usize,
) {
    let edge_count = edges.len();
    if edge_count == 0 {
        return;
    }
    let input1_dim = plan.input1_dim();
    let output_dim = plan.output_dim();
    let chunk_count = edge_count.clamp(1, OAM_TP_VJP_PARALLEL_TARGET_CHUNKS);
    let edge_chunk_len = edge_count.div_ceil(chunk_count).max(1);
    let local_linear1_len = num_nodes * input1_dim;

    let linear1_bar_chunks: Vec<Vec<f64>> = edge_attr_bars
        .par_chunks_mut(edge_chunk_len * sh_dim)
        .zip(edge_mlp_bars.par_chunks_mut(edge_chunk_len * tp_weight_numel))
        .enumerate()
        .map(|(chunk_index, (edge_attr_chunk, edge_mlp_chunk))| {
            let mut local_linear1_bars = vec![0.0; local_linear1_len];
            let edge_start = chunk_index * edge_chunk_len;
            let chunk_edges = edge_attr_chunk.len() / sh_dim;
            debug_assert_eq!(edge_mlp_chunk.len(), chunk_edges * tp_weight_numel);
            for local_edge in 0..chunk_edges {
                let edge_index = edge_start + local_edge;
                let edge = &edges[edge_index];
                tensor_product_vjp_accumulate_oam(
                    plan,
                    &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                    &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                    &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                    &tp_scatter_bars[edge.src * output_dim..(edge.src + 1) * output_dim],
                    &mut local_linear1_bars[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                    &mut edge_attr_chunk[local_edge * sh_dim..(local_edge + 1) * sh_dim],
                    &mut edge_mlp_chunk
                        [local_edge * tp_weight_numel..(local_edge + 1) * tp_weight_numel],
                );
            }
            local_linear1_bars
        })
        .collect();

    for local_linear1_bars in linear1_bar_chunks {
        for (target, value) in normed_linear1_bars.iter_mut().zip(local_linear1_bars) {
            *target += value;
        }
    }
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn tensor_product_vjp_scatter_oam_parallel_direct_fused_edge_mlp2(
    plan: &OamTensorProductPlan,
    edges: &[crate::Edge],
    edge_attrs: &[f64],
    sh_dim: usize,
    normed_linear1: &[f64],
    edge_mlp: &[f64],
    tp_weight_numel: usize,
    tp_scatter_bars: &[f64],
    edge_mlp_2_weight: &[f64],
    edge_mlp_2_alpha: f64,
    normed_linear1_bars: &mut [f64],
    edge_attr_bars: &mut [f64],
    hidden_silu_bars: &mut [f64],
    num_nodes: usize,
) -> Result<(), NequipOamPrefixError> {
    let edge_count = edges.len();
    if edge_count == 0 {
        return Ok(());
    }
    let input1_dim = plan.input1_dim();
    let output_dim = plan.output_dim();
    let chunk_count = edge_count.clamp(1, OAM_TP_VJP_PARALLEL_TARGET_CHUNKS);
    let edge_chunk_len = edge_count.div_ceil(chunk_count).max(1);
    let local_linear1_len = num_nodes * input1_dim;

    let linear1_bar_chunks: Vec<Result<Vec<f64>, NequipOamPrefixError>> = edge_attr_bars
        .par_chunks_mut(edge_chunk_len * sh_dim)
        .zip(hidden_silu_bars.par_chunks_mut(edge_chunk_len * 128))
        .enumerate()
        .map(|(chunk_index, (edge_attr_chunk, hidden_silu_chunk))| {
            let mut local_linear1_bars = vec![0.0; local_linear1_len];
            let chunk_edges = edge_attr_chunk.len() / sh_dim;
            let mut local_edge_mlp_bars = vec![0.0; chunk_edges * tp_weight_numel];
            let edge_start = chunk_index * edge_chunk_len;
            debug_assert_eq!(hidden_silu_chunk.len(), chunk_edges * 128);
            for local_edge in 0..chunk_edges {
                let edge_index = edge_start + local_edge;
                let edge = &edges[edge_index];
                tensor_product_vjp_accumulate_oam(
                    plan,
                    &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                    &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                    &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                    &tp_scatter_bars[edge.src * output_dim..(edge.src + 1) * output_dim],
                    &mut local_linear1_bars[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                    &mut edge_attr_chunk[local_edge * sh_dim..(local_edge + 1) * sh_dim],
                    &mut local_edge_mlp_bars
                        [local_edge * tp_weight_numel..(local_edge + 1) * tp_weight_numel],
                );
            }
            input_major_linear_vjp_input_accumulate(
                &local_edge_mlp_bars,
                chunk_edges,
                128,
                tp_weight_numel,
                edge_mlp_2_weight,
                edge_mlp_2_alpha,
                hidden_silu_chunk,
            )?;
            Ok(local_linear1_bars)
        })
        .collect();

    for local_linear1_bars in linear1_bar_chunks {
        for (target, value) in normed_linear1_bars.iter_mut().zip(local_linear1_bars?) {
            *target += value;
        }
    }
    Ok(())
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn tensor_product_vjp_scatter_oam_parallel_direct_f32(
    plan: &OamTensorProductPlan,
    edges: &[crate::Edge],
    edge_attrs: &[f32],
    sh_dim: usize,
    normed_linear1: &[f32],
    edge_mlp: &[f32],
    tp_weight_numel: usize,
    tp_scatter_bars: &[f32],
    normed_linear1_bars: &mut [f32],
    edge_attr_bars: &mut [f32],
    edge_mlp_bars: &mut [f32],
    num_nodes: usize,
) {
    let edge_count = edges.len();
    if edge_count == 0 {
        return;
    }
    let input1_dim = plan.input1_dim();
    let output_dim = plan.output_dim();
    let chunk_count = edge_count.clamp(1, OAM_TP_VJP_PARALLEL_TARGET_CHUNKS);
    let edge_chunk_len = edge_count.div_ceil(chunk_count).max(1);
    let local_linear1_len = num_nodes * input1_dim;

    let linear1_bar_chunks: Vec<Vec<f32>> = edge_attr_bars
        .par_chunks_mut(edge_chunk_len * sh_dim)
        .zip(edge_mlp_bars.par_chunks_mut(edge_chunk_len * tp_weight_numel))
        .enumerate()
        .map(|(chunk_index, (edge_attr_chunk, edge_mlp_chunk))| {
            let mut local_linear1_bars = vec![0.0; local_linear1_len];
            let edge_start = chunk_index * edge_chunk_len;
            let chunk_edges = edge_attr_chunk.len() / sh_dim;
            debug_assert_eq!(edge_mlp_chunk.len(), chunk_edges * tp_weight_numel);
            for local_edge in 0..chunk_edges {
                let edge_index = edge_start + local_edge;
                let edge = &edges[edge_index];
                tensor_product_vjp_accumulate_oam_f32(
                    plan,
                    &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                    &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                    &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                    &tp_scatter_bars[edge.src * output_dim..(edge.src + 1) * output_dim],
                    &mut local_linear1_bars[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                    &mut edge_attr_chunk[local_edge * sh_dim..(local_edge + 1) * sh_dim],
                    &mut edge_mlp_chunk
                        [local_edge * tp_weight_numel..(local_edge + 1) * tp_weight_numel],
                );
            }
            local_linear1_bars
        })
        .collect();

    for local_linear1_bars in linear1_bar_chunks {
        for (target, value) in normed_linear1_bars.iter_mut().zip(local_linear1_bars) {
            *target += value;
        }
    }
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn tensor_product_vjp_scatter_oam_parallel_direct_fused_edge_mlp2_f32(
    plan: &OamTensorProductPlan,
    edges: &[crate::Edge],
    edge_attrs: &[f32],
    sh_dim: usize,
    normed_linear1: &[f32],
    edge_mlp: &[f32],
    tp_weight_numel: usize,
    tp_scatter_bars: &[f32],
    edge_mlp_2_weight: &[f32],
    edge_mlp_2_alpha: f32,
    normed_linear1_bars: &mut [f32],
    edge_attr_bars: &mut [f32],
    hidden_silu_bars: &mut [f32],
    num_nodes: usize,
) -> Result<(), NequipOamPrefixError> {
    let edge_count = edges.len();
    if edge_count == 0 {
        return Ok(());
    }
    let input1_dim = plan.input1_dim();
    let output_dim = plan.output_dim();
    let chunk_count = edge_count.clamp(1, OAM_TP_VJP_PARALLEL_TARGET_CHUNKS);
    let edge_chunk_len = edge_count.div_ceil(chunk_count).max(1);
    let local_linear1_len = num_nodes * input1_dim;

    let linear1_bar_chunks: Vec<Result<Vec<f32>, NequipOamPrefixError>> = edge_attr_bars
        .par_chunks_mut(edge_chunk_len * sh_dim)
        .zip(hidden_silu_bars.par_chunks_mut(edge_chunk_len * 128))
        .enumerate()
        .map(|(chunk_index, (edge_attr_chunk, hidden_silu_chunk))| {
            let mut local_linear1_bars = vec![0.0; local_linear1_len];
            let chunk_edges = edge_attr_chunk.len() / sh_dim;
            let mut local_edge_mlp_bars = vec![0.0; chunk_edges * tp_weight_numel];
            let edge_start = chunk_index * edge_chunk_len;
            debug_assert_eq!(hidden_silu_chunk.len(), chunk_edges * 128);
            for local_edge in 0..chunk_edges {
                let edge_index = edge_start + local_edge;
                let edge = &edges[edge_index];
                tensor_product_vjp_accumulate_oam_f32(
                    plan,
                    &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                    &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                    &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                    &tp_scatter_bars[edge.src * output_dim..(edge.src + 1) * output_dim],
                    &mut local_linear1_bars[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                    &mut edge_attr_chunk[local_edge * sh_dim..(local_edge + 1) * sh_dim],
                    &mut local_edge_mlp_bars
                        [local_edge * tp_weight_numel..(local_edge + 1) * tp_weight_numel],
                );
            }
            input_major_linear_vjp_input_accumulate_f32(
                &local_edge_mlp_bars,
                chunk_edges,
                128,
                tp_weight_numel,
                edge_mlp_2_weight,
                edge_mlp_2_alpha,
                hidden_silu_chunk,
            )?;
            Ok(local_linear1_bars)
        })
        .collect();

    for local_linear1_bars in linear1_bar_chunks {
        for (target, value) in normed_linear1_bars.iter_mut().zip(local_linear1_bars?) {
            *target += value;
        }
    }
    Ok(())
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn tensor_product_vjp_scatter_oam_parallel_direct_fused_edge_mlp2_tp_layout(
    plan: &OamTensorProductPlan,
    edges: &[crate::Edge],
    edge_plan: &OamEdgePlan,
    edge_attrs: &[f64],
    sh_dim: usize,
    normed_linear1: &[f64],
    edge_mlp: &[f64],
    tp_weight_numel: usize,
    tp_scatter_bars: &[f64],
    edge_mlp_2_weight: &[f64],
    edge_mlp_2_alpha: f64,
    normed_linear1_bars: &mut [f64],
    edge_attr_bars: &mut [f64],
    hidden_silu_bars: &mut [f64],
    num_nodes: usize,
) -> Result<(), NequipOamPrefixError> {
    let edge_count = edges.len();
    if edge_count == 0 {
        return Ok(());
    }
    let input1_dim = plan.input1_dim();
    let output_dim = plan.output_dim();
    if edge_count >= OAM_TP_VJP_DST_OWNED_MAX_EDGES {
        let chunk_count = edge_count.clamp(1, OAM_TP_VJP_PARALLEL_TARGET_CHUNKS);
        let edge_chunk_len = edge_count.div_ceil(chunk_count).max(1);
        let local_linear1_len = num_nodes * input1_dim;

        let linear1_bar_chunks: Vec<Result<Vec<f64>, NequipOamPrefixError>> = edge_attr_bars
            .par_chunks_mut(edge_chunk_len * sh_dim)
            .zip(hidden_silu_bars.par_chunks_mut(edge_chunk_len * 128))
            .enumerate()
            .map(|(chunk_index, (edge_attr_chunk, hidden_silu_chunk))| {
                let mut local_linear1_bars = vec![0.0; local_linear1_len];
                let chunk_edges = edge_attr_chunk.len() / sh_dim;
                let mut local_edge_mlp_bars = vec![0.0; chunk_edges * tp_weight_numel];
                let edge_start = chunk_index * edge_chunk_len;
                debug_assert_eq!(hidden_silu_chunk.len(), chunk_edges * 128);
                for local_edge in 0..chunk_edges {
                    let edge_index = edge_start + local_edge;
                    let edge = &edges[edge_index];
                    tensor_product_vjp_accumulate_oam_tp_layout(
                        plan,
                        &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                        &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                        &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                        &tp_scatter_bars[edge.src * output_dim..(edge.src + 1) * output_dim],
                        &mut local_linear1_bars[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                        &mut edge_attr_chunk[local_edge * sh_dim..(local_edge + 1) * sh_dim],
                        &mut local_edge_mlp_bars
                            [local_edge * tp_weight_numel..(local_edge + 1) * tp_weight_numel],
                    );
                }
                input_major_linear_vjp_input_accumulate(
                    &local_edge_mlp_bars,
                    chunk_edges,
                    128,
                    tp_weight_numel,
                    edge_mlp_2_weight,
                    edge_mlp_2_alpha,
                    hidden_silu_chunk,
                )?;
                Ok(local_linear1_bars)
            })
            .collect();

        for local_linear1_bars in linear1_bar_chunks {
            for (target, value) in normed_linear1_bars.iter_mut().zip(local_linear1_bars?) {
                *target += value;
            }
        }
        return Ok(());
    }
    debug_assert_eq!(edge_plan.dst_offsets.len(), num_nodes + 1);
    let node_chunk_count = num_nodes.clamp(1, OAM_TP_VJP_PARALLEL_TARGET_CHUNKS);
    let node_chunk_len = num_nodes.div_ceil(node_chunk_count).max(1);

    let chunk_results: Vec<OamParallelVjpChunkResult<f64>> = normed_linear1_bars
        .par_chunks_mut(node_chunk_len * input1_dim)
        .enumerate()
        .map(|(node_chunk_index, normed_linear1_chunk)| {
            let dst_start = node_chunk_index * node_chunk_len;
            let dst_count = normed_linear1_chunk.len() / input1_dim;
            let edge_start = edge_plan.dst_offsets[dst_start];
            let edge_end = edge_plan.dst_offsets[dst_start + dst_count];
            let chunk_edges = edge_end - edge_start;
            let mut local_edge_attr_bars = vec![0.0; chunk_edges * sh_dim];
            let mut local_edge_mlp_bars = vec![0.0; chunk_edges * tp_weight_numel];
            let mut local_hidden_silu_bars = vec![0.0; chunk_edges * 128];
            for (local_edge, &edge_index) in edge_plan.edges_by_dst[edge_start..edge_end]
                .iter()
                .enumerate()
            {
                let edge = &edges[edge_index];
                let dst_local = edge.dst - dst_start;
                tensor_product_vjp_accumulate_oam_tp_layout(
                    plan,
                    &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                    &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                    &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                    &tp_scatter_bars[edge.src * output_dim..(edge.src + 1) * output_dim],
                    &mut normed_linear1_chunk[dst_local * input1_dim..(dst_local + 1) * input1_dim],
                    &mut local_edge_attr_bars[local_edge * sh_dim..(local_edge + 1) * sh_dim],
                    &mut local_edge_mlp_bars
                        [local_edge * tp_weight_numel..(local_edge + 1) * tp_weight_numel],
                );
            }
            input_major_linear_vjp_input_accumulate(
                &local_edge_mlp_bars,
                chunk_edges,
                128,
                tp_weight_numel,
                edge_mlp_2_weight,
                edge_mlp_2_alpha,
                &mut local_hidden_silu_bars,
            )?;
            Ok(OamParallelVjpChunk {
                edge_start,
                edge_end,
                edge_attr_bars: local_edge_attr_bars,
                hidden_silu_bars: local_hidden_silu_bars,
            })
        })
        .collect();

    for result in chunk_results {
        let chunk = result?;
        for (local_edge, &edge_index) in edge_plan.edges_by_dst[chunk.edge_start..chunk.edge_end]
            .iter()
            .enumerate()
        {
            let edge_attr_bar = &mut edge_attr_bars[edge_index * sh_dim..(edge_index + 1) * sh_dim];
            let chunk_edge_attr_bar =
                &chunk.edge_attr_bars[local_edge * sh_dim..(local_edge + 1) * sh_dim];
            for (target, value) in edge_attr_bar.iter_mut().zip(chunk_edge_attr_bar) {
                *target += *value;
            }
            hidden_silu_bars[edge_index * 128..(edge_index + 1) * 128]
                .copy_from_slice(&chunk.hidden_silu_bars[local_edge * 128..(local_edge + 1) * 128]);
        }
    }
    Ok(())
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn tensor_product_vjp_scatter_oam_parallel_direct_fused_edge_mlp2_tp_layout_f32(
    plan: &OamTensorProductPlan,
    edges: &[crate::Edge],
    edge_plan: &OamEdgePlan,
    edge_attrs: &[f32],
    sh_dim: usize,
    normed_linear1: &[f32],
    edge_mlp: &[f32],
    tp_weight_numel: usize,
    tp_scatter_bars: &[f32],
    edge_mlp_2_weight: &[f32],
    edge_mlp_2_alpha: f32,
    normed_linear1_bars: &mut [f32],
    edge_attr_bars: &mut [f32],
    hidden_silu_bars: &mut [f32],
    num_nodes: usize,
) -> Result<(), NequipOamPrefixError> {
    let edge_count = edges.len();
    if edge_count == 0 {
        return Ok(());
    }
    let input1_dim = plan.input1_dim();
    let output_dim = plan.output_dim();
    if edge_count >= OAM_TP_VJP_DST_OWNED_MAX_EDGES {
        let chunk_count = edge_count.clamp(1, OAM_TP_VJP_PARALLEL_TARGET_CHUNKS);
        let edge_chunk_len = edge_count.div_ceil(chunk_count).max(1);
        let local_linear1_len = num_nodes * input1_dim;

        let linear1_bar_chunks: Vec<Result<Vec<f32>, NequipOamPrefixError>> = edge_attr_bars
            .par_chunks_mut(edge_chunk_len * sh_dim)
            .zip(hidden_silu_bars.par_chunks_mut(edge_chunk_len * 128))
            .enumerate()
            .map(|(chunk_index, (edge_attr_chunk, hidden_silu_chunk))| {
                let mut local_linear1_bars = vec![0.0; local_linear1_len];
                let chunk_edges = edge_attr_chunk.len() / sh_dim;
                let mut local_edge_mlp_bars = vec![0.0; chunk_edges * tp_weight_numel];
                let edge_start = chunk_index * edge_chunk_len;
                debug_assert_eq!(hidden_silu_chunk.len(), chunk_edges * 128);
                for local_edge in 0..chunk_edges {
                    let edge_index = edge_start + local_edge;
                    let edge = &edges[edge_index];
                    tensor_product_vjp_accumulate_oam_tp_layout_f32(
                        plan,
                        &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                        &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                        &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                        &tp_scatter_bars[edge.src * output_dim..(edge.src + 1) * output_dim],
                        &mut local_linear1_bars[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                        &mut edge_attr_chunk[local_edge * sh_dim..(local_edge + 1) * sh_dim],
                        &mut local_edge_mlp_bars
                            [local_edge * tp_weight_numel..(local_edge + 1) * tp_weight_numel],
                    );
                }
                input_major_linear_vjp_input_accumulate_f32(
                    &local_edge_mlp_bars,
                    chunk_edges,
                    128,
                    tp_weight_numel,
                    edge_mlp_2_weight,
                    edge_mlp_2_alpha,
                    hidden_silu_chunk,
                )?;
                Ok(local_linear1_bars)
            })
            .collect();

        for local_linear1_bars in linear1_bar_chunks {
            for (target, value) in normed_linear1_bars.iter_mut().zip(local_linear1_bars?) {
                *target += value;
            }
        }
        return Ok(());
    }
    debug_assert_eq!(edge_plan.dst_offsets.len(), num_nodes + 1);
    let node_chunk_count = num_nodes.clamp(1, OAM_TP_VJP_PARALLEL_TARGET_CHUNKS);
    let node_chunk_len = num_nodes.div_ceil(node_chunk_count).max(1);

    let chunk_results: Vec<OamParallelVjpChunkResult<f32>> = normed_linear1_bars
        .par_chunks_mut(node_chunk_len * input1_dim)
        .enumerate()
        .map(|(node_chunk_index, normed_linear1_chunk)| {
            let dst_start = node_chunk_index * node_chunk_len;
            let dst_count = normed_linear1_chunk.len() / input1_dim;
            let edge_start = edge_plan.dst_offsets[dst_start];
            let edge_end = edge_plan.dst_offsets[dst_start + dst_count];
            let chunk_edges = edge_end - edge_start;
            let mut local_edge_attr_bars = vec![0.0; chunk_edges * sh_dim];
            let mut local_edge_mlp_bars = vec![0.0; chunk_edges * tp_weight_numel];
            let mut local_hidden_silu_bars = vec![0.0; chunk_edges * 128];
            for (local_edge, &edge_index) in edge_plan.edges_by_dst[edge_start..edge_end]
                .iter()
                .enumerate()
            {
                let edge = &edges[edge_index];
                let dst_local = edge.dst - dst_start;
                tensor_product_vjp_accumulate_oam_tp_layout_f32(
                    plan,
                    &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                    &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                    &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                    &tp_scatter_bars[edge.src * output_dim..(edge.src + 1) * output_dim],
                    &mut normed_linear1_chunk[dst_local * input1_dim..(dst_local + 1) * input1_dim],
                    &mut local_edge_attr_bars[local_edge * sh_dim..(local_edge + 1) * sh_dim],
                    &mut local_edge_mlp_bars
                        [local_edge * tp_weight_numel..(local_edge + 1) * tp_weight_numel],
                );
            }
            input_major_linear_vjp_input_accumulate_f32(
                &local_edge_mlp_bars,
                chunk_edges,
                128,
                tp_weight_numel,
                edge_mlp_2_weight,
                edge_mlp_2_alpha,
                &mut local_hidden_silu_bars,
            )?;
            Ok(OamParallelVjpChunk {
                edge_start,
                edge_end,
                edge_attr_bars: local_edge_attr_bars,
                hidden_silu_bars: local_hidden_silu_bars,
            })
        })
        .collect();

    for result in chunk_results {
        let chunk = result?;
        for (local_edge, &edge_index) in edge_plan.edges_by_dst[chunk.edge_start..chunk.edge_end]
            .iter()
            .enumerate()
        {
            let edge_attr_bar = &mut edge_attr_bars[edge_index * sh_dim..(edge_index + 1) * sh_dim];
            let chunk_edge_attr_bar =
                &chunk.edge_attr_bars[local_edge * sh_dim..(local_edge + 1) * sh_dim];
            for (target, value) in edge_attr_bar.iter_mut().zip(chunk_edge_attr_bar) {
                *target += *value;
            }
            hidden_silu_bars[edge_index * 128..(edge_index + 1) * 128]
                .copy_from_slice(&chunk.hidden_silu_bars[local_edge * 128..(local_edge + 1) * 128]);
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "parallel"))]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn tensor_product_vjp_scatter_oam_parallel_direct_fused_linear2_edge_mlp2_edge_chunks(
    plan: &OamTensorProductPlan,
    edges: &[crate::Edge],
    edge_attrs: &[f64],
    sh_dim: usize,
    normed_linear1: &[f64],
    edge_mlp: &[f64],
    tp_weight_numel: usize,
    linear2_plan: &OamCompactLinearPlan,
    gate_input_bars: &[f64],
    linear2_weight: &[f64],
    edge_mlp_2_weight: &[f64],
    edge_mlp_2_alpha: f64,
    normed_linear1_bars: &mut [f64],
    edge_attr_bars: &mut [f64],
    hidden_silu_bars: &mut [f64],
    num_nodes: usize,
) -> Result<(), NequipOamPrefixError> {
    let edge_count = edges.len();
    if edge_count == 0 {
        return Ok(());
    }
    let input1_dim = plan.input1_dim();
    let output_dim = plan.output_dim();
    if linear2_plan.input_dim != output_dim
        || gate_input_bars.len() != num_nodes * linear2_plan.output_dim
        || linear2_weight.len() != linear2_plan.weight_numel
        || normed_linear1.len() != num_nodes * input1_dim
        || edge_attrs.len() != edge_count * sh_dim
        || edge_mlp.len() != edge_count * tp_weight_numel
        || hidden_silu_bars.len() != edge_count * 128
    {
        return Err(invalid_model(
            "parallel edge-chunk fused linear2/TP/edge-MLP2 VJP shape mismatch",
        ));
    }
    let chunk_count = edge_count.clamp(1, OAM_TP_VJP_PARALLEL_TARGET_CHUNKS);
    let edge_chunk_len = edge_count.div_ceil(chunk_count).max(1);
    let local_linear1_len = num_nodes * input1_dim;

    let linear1_bar_chunks: Vec<Result<Vec<f64>, NequipOamPrefixError>> = edge_attr_bars
        .par_chunks_mut(edge_chunk_len * sh_dim)
        .zip(hidden_silu_bars.par_chunks_mut(edge_chunk_len * 128))
        .enumerate()
        .map(|(chunk_index, (edge_attr_chunk, hidden_silu_chunk))| {
            let mut local_linear1_bars = vec![0.0; local_linear1_len];
            let chunk_edges = edge_attr_chunk.len() / sh_dim;
            let mut local_edge_mlp_bars = vec![0.0; chunk_edges * tp_weight_numel];
            let mut tp_scatter_bar_row = vec![0.0; output_dim];
            let mut current_src = usize::MAX;
            let edge_start = chunk_index * edge_chunk_len;
            debug_assert_eq!(hidden_silu_chunk.len(), chunk_edges * 128);
            for local_edge in 0..chunk_edges {
                let edge_index = edge_start + local_edge;
                let edge = &edges[edge_index];
                if edge.src != current_src {
                    compact_linear_vjp_input_row_into_oam(
                        linear2_plan,
                        &gate_input_bars[edge.src * linear2_plan.output_dim
                            ..(edge.src + 1) * linear2_plan.output_dim],
                        linear2_weight,
                        &mut tp_scatter_bar_row,
                    );
                    current_src = edge.src;
                }
                tensor_product_vjp_accumulate_oam(
                    plan,
                    &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                    &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                    &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                    &tp_scatter_bar_row,
                    &mut local_linear1_bars[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                    &mut edge_attr_chunk[local_edge * sh_dim..(local_edge + 1) * sh_dim],
                    &mut local_edge_mlp_bars
                        [local_edge * tp_weight_numel..(local_edge + 1) * tp_weight_numel],
                );
            }
            input_major_linear_vjp_input_accumulate(
                &local_edge_mlp_bars,
                chunk_edges,
                128,
                tp_weight_numel,
                edge_mlp_2_weight,
                edge_mlp_2_alpha,
                hidden_silu_chunk,
            )?;
            Ok(local_linear1_bars)
        })
        .collect();

    for local_linear1_bars in linear1_bar_chunks {
        for (target, value) in normed_linear1_bars.iter_mut().zip(local_linear1_bars?) {
            *target += value;
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "parallel"))]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn tensor_product_vjp_scatter_oam_parallel_direct_fused_linear2_edge_mlp2_edge_chunks_f32(
    plan: &OamTensorProductPlan,
    edges: &[crate::Edge],
    edge_attrs: &[f32],
    sh_dim: usize,
    normed_linear1: &[f32],
    edge_mlp: &[f32],
    tp_weight_numel: usize,
    linear2_plan: &OamCompactLinearPlan,
    gate_input_bars: &[f32],
    linear2_weight: &[f32],
    edge_mlp_2_weight: &[f32],
    edge_mlp_2_alpha: f32,
    normed_linear1_bars: &mut [f32],
    edge_attr_bars: &mut [f32],
    hidden_silu_bars: &mut [f32],
    num_nodes: usize,
) -> Result<(), NequipOamPrefixError> {
    let edge_count = edges.len();
    if edge_count == 0 {
        return Ok(());
    }
    let input1_dim = plan.input1_dim();
    let output_dim = plan.output_dim();
    if linear2_plan.input_dim != output_dim
        || gate_input_bars.len() != num_nodes * linear2_plan.output_dim
        || linear2_weight.len() != linear2_plan.weight_numel
        || normed_linear1.len() != num_nodes * input1_dim
        || edge_attrs.len() != edge_count * sh_dim
        || edge_mlp.len() != edge_count * tp_weight_numel
        || hidden_silu_bars.len() != edge_count * 128
    {
        return Err(invalid_model(
            "parallel edge-chunk fused f32 linear2/TP/edge-MLP2 VJP shape mismatch",
        ));
    }
    let chunk_count = edge_count.clamp(1, OAM_TP_VJP_PARALLEL_TARGET_CHUNKS);
    let edge_chunk_len = edge_count.div_ceil(chunk_count).max(1);
    let local_linear1_len = num_nodes * input1_dim;

    let linear1_bar_chunks: Vec<Result<Vec<f32>, NequipOamPrefixError>> = edge_attr_bars
        .par_chunks_mut(edge_chunk_len * sh_dim)
        .zip(hidden_silu_bars.par_chunks_mut(edge_chunk_len * 128))
        .enumerate()
        .map(|(chunk_index, (edge_attr_chunk, hidden_silu_chunk))| {
            let mut local_linear1_bars = vec![0.0; local_linear1_len];
            let chunk_edges = edge_attr_chunk.len() / sh_dim;
            let mut local_edge_mlp_bars = vec![0.0; chunk_edges * tp_weight_numel];
            let mut tp_scatter_bar_row = vec![0.0; output_dim];
            let mut current_src = usize::MAX;
            let edge_start = chunk_index * edge_chunk_len;
            debug_assert_eq!(hidden_silu_chunk.len(), chunk_edges * 128);
            for local_edge in 0..chunk_edges {
                let edge_index = edge_start + local_edge;
                let edge = &edges[edge_index];
                if edge.src != current_src {
                    compact_linear_vjp_input_row_into_oam_f32(
                        linear2_plan,
                        &gate_input_bars[edge.src * linear2_plan.output_dim
                            ..(edge.src + 1) * linear2_plan.output_dim],
                        linear2_weight,
                        &mut tp_scatter_bar_row,
                    );
                    current_src = edge.src;
                }
                tensor_product_vjp_accumulate_oam_f32(
                    plan,
                    &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                    &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                    &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                    &tp_scatter_bar_row,
                    &mut local_linear1_bars[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                    &mut edge_attr_chunk[local_edge * sh_dim..(local_edge + 1) * sh_dim],
                    &mut local_edge_mlp_bars
                        [local_edge * tp_weight_numel..(local_edge + 1) * tp_weight_numel],
                );
            }
            input_major_linear_vjp_input_accumulate_f32(
                &local_edge_mlp_bars,
                chunk_edges,
                128,
                tp_weight_numel,
                edge_mlp_2_weight,
                edge_mlp_2_alpha,
                hidden_silu_chunk,
            )?;
            Ok(local_linear1_bars)
        })
        .collect();

    for local_linear1_bars in linear1_bar_chunks {
        for (target, value) in normed_linear1_bars.iter_mut().zip(local_linear1_bars?) {
            *target += value;
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "parallel"))]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn tensor_product_vjp_scatter_oam_direct_fused_linear2_edge_mlp2(
    plan: &OamTensorProductPlan,
    edges: &[crate::Edge],
    edge_plan: &OamEdgePlan,
    edge_attrs: &[f64],
    sh_dim: usize,
    normed_linear1: &[f64],
    edge_mlp: &[f64],
    tp_weight_numel: usize,
    linear2_plan: &OamCompactLinearPlan,
    gate_input_bars: &[f64],
    linear2_weight: &[f64],
    edge_mlp_2_weight: &[f64],
    edge_mlp_2_alpha: f64,
    normed_linear1_bars: &mut [f64],
    edge_attr_bars: &mut [f64],
    hidden_silu_bars: &mut [f64],
    num_nodes: usize,
) -> Result<(), NequipOamPrefixError> {
    let edge_count = edges.len();
    let input1_dim = plan.input1_dim();
    let output_dim = plan.output_dim();
    if linear2_plan.input_dim != output_dim
        || gate_input_bars.len() != num_nodes * linear2_plan.output_dim
        || linear2_weight.len() != linear2_plan.weight_numel
        || normed_linear1.len() != num_nodes * input1_dim
        || edge_attrs.len() != edge_count * sh_dim
        || edge_mlp.len() != edge_count * tp_weight_numel
        || hidden_silu_bars.len() != edge_count * 128
    {
        return Err(invalid_model(
            "fused linear2/TP/edge-MLP2 VJP shape mismatch",
        ));
    }

    let mut edge_mlp_bars = vec![0.0; edge_count * tp_weight_numel];
    let mut tp_scatter_bar_row = vec![0.0; output_dim];
    let mut dst_grad_x = vec![0.0; input1_dim];
    for src in 0..num_nodes {
        let edge_range = edge_plan.src_offsets[src]..edge_plan.src_offsets[src + 1];
        if edge_range.is_empty() {
            continue;
        }
        compact_linear_vjp_input_row_into_oam(
            linear2_plan,
            &gate_input_bars[src * linear2_plan.output_dim..(src + 1) * linear2_plan.output_dim],
            linear2_weight,
            &mut tp_scatter_bar_row,
        );
        for &edge_index in &edge_plan.edges_by_src[edge_range] {
            let edge = &edges[edge_index];
            dst_grad_x.fill(0.0);
            tensor_product_vjp_accumulate_oam(
                plan,
                &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                &tp_scatter_bar_row,
                &mut dst_grad_x,
                &mut edge_attr_bars[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                &mut edge_mlp_bars
                    [edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
            );
            let target =
                &mut normed_linear1_bars[edge.dst * input1_dim..(edge.dst + 1) * input1_dim];
            for (target, value) in target.iter_mut().zip(&dst_grad_x) {
                *target += *value;
            }
        }
    }
    input_major_linear_vjp_input_accumulate(
        &edge_mlp_bars,
        edge_count,
        128,
        tp_weight_numel,
        edge_mlp_2_weight,
        edge_mlp_2_alpha,
        hidden_silu_bars,
    )
}

#[cfg(all(test, feature = "parallel"))]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn tensor_product_vjp_scatter_oam_direct_fused_linear2_edge_mlp2_f32(
    plan: &OamTensorProductPlan,
    edges: &[crate::Edge],
    edge_plan: &OamEdgePlan,
    edge_attrs: &[f32],
    sh_dim: usize,
    normed_linear1: &[f32],
    edge_mlp: &[f32],
    tp_weight_numel: usize,
    linear2_plan: &OamCompactLinearPlan,
    gate_input_bars: &[f32],
    linear2_weight: &[f32],
    edge_mlp_2_weight: &[f32],
    edge_mlp_2_alpha: f32,
    normed_linear1_bars: &mut [f32],
    edge_attr_bars: &mut [f32],
    hidden_silu_bars: &mut [f32],
    num_nodes: usize,
) -> Result<(), NequipOamPrefixError> {
    let edge_count = edges.len();
    let input1_dim = plan.input1_dim();
    let output_dim = plan.output_dim();
    if linear2_plan.input_dim != output_dim
        || gate_input_bars.len() != num_nodes * linear2_plan.output_dim
        || linear2_weight.len() != linear2_plan.weight_numel
        || normed_linear1.len() != num_nodes * input1_dim
        || edge_attrs.len() != edge_count * sh_dim
        || edge_mlp.len() != edge_count * tp_weight_numel
        || hidden_silu_bars.len() != edge_count * 128
    {
        return Err(invalid_model(
            "fused f32 linear2/TP/edge-MLP2 VJP shape mismatch",
        ));
    }

    let mut edge_mlp_bars = vec![0.0; edge_count * tp_weight_numel];
    let mut tp_scatter_bar_row = vec![0.0; output_dim];
    let mut dst_grad_x = vec![0.0; input1_dim];
    for src in 0..num_nodes {
        let edge_range = edge_plan.src_offsets[src]..edge_plan.src_offsets[src + 1];
        if edge_range.is_empty() {
            continue;
        }
        compact_linear_vjp_input_row_into_oam_f32(
            linear2_plan,
            &gate_input_bars[src * linear2_plan.output_dim..(src + 1) * linear2_plan.output_dim],
            linear2_weight,
            &mut tp_scatter_bar_row,
        );
        for &edge_index in &edge_plan.edges_by_src[edge_range] {
            let edge = &edges[edge_index];
            dst_grad_x.fill(0.0);
            tensor_product_vjp_accumulate_oam_f32(
                plan,
                &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                &tp_scatter_bar_row,
                &mut dst_grad_x,
                &mut edge_attr_bars[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                &mut edge_mlp_bars
                    [edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
            );
            let target =
                &mut normed_linear1_bars[edge.dst * input1_dim..(edge.dst + 1) * input1_dim];
            for (target, value) in target.iter_mut().zip(&dst_grad_x) {
                *target += *value;
            }
        }
    }
    input_major_linear_vjp_input_accumulate_f32(
        &edge_mlp_bars,
        edge_count,
        128,
        tp_weight_numel,
        edge_mlp_2_weight,
        edge_mlp_2_alpha,
        hidden_silu_bars,
    )
}

#[cfg(all(test, feature = "parallel"))]
#[allow(dead_code)]
struct OamFusedLinear2ChunkBars {
    normed_linear1_bars: Vec<f64>,
    edge_indices: Vec<usize>,
    edge_attr_bars: Vec<f64>,
    hidden_silu_bars: Vec<f64>,
}

#[cfg(all(test, feature = "parallel"))]
#[allow(dead_code)]
struct OamFusedLinear2ChunkBarsF32 {
    normed_linear1_bars: Vec<f32>,
    edge_indices: Vec<usize>,
    edge_attr_bars: Vec<f32>,
    hidden_silu_bars: Vec<f32>,
}

#[cfg(all(test, feature = "parallel"))]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn tensor_product_vjp_scatter_oam_parallel_direct_fused_linear2_edge_mlp2(
    plan: &OamTensorProductPlan,
    edges: &[crate::Edge],
    edge_plan: &OamEdgePlan,
    edge_attrs: &[f64],
    sh_dim: usize,
    normed_linear1: &[f64],
    edge_mlp: &[f64],
    tp_weight_numel: usize,
    linear2_plan: &OamCompactLinearPlan,
    gate_input_bars: &[f64],
    linear2_weight: &[f64],
    edge_mlp_2_weight: &[f64],
    edge_mlp_2_alpha: f64,
    normed_linear1_bars: &mut [f64],
    edge_attr_bars: &mut [f64],
    hidden_silu_bars: &mut [f64],
    num_nodes: usize,
) -> Result<(), NequipOamPrefixError> {
    let edge_count = edges.len();
    let input1_dim = plan.input1_dim();
    let output_dim = plan.output_dim();
    if edge_count == 0 {
        return Ok(());
    }
    if linear2_plan.input_dim != output_dim
        || gate_input_bars.len() != num_nodes * linear2_plan.output_dim
        || linear2_weight.len() != linear2_plan.weight_numel
        || normed_linear1.len() != num_nodes * input1_dim
        || edge_attrs.len() != edge_count * sh_dim
        || edge_mlp.len() != edge_count * tp_weight_numel
        || hidden_silu_bars.len() != edge_count * 128
    {
        return Err(invalid_model(
            "parallel fused linear2/TP/edge-MLP2 VJP shape mismatch",
        ));
    }

    let source_chunk_count = num_nodes.clamp(1, OAM_TP_VJP_PARALLEL_TARGET_CHUNKS);
    let source_chunk_len = num_nodes.div_ceil(source_chunk_count).max(1);
    let local_linear1_len = num_nodes * input1_dim;
    let chunks = (0..source_chunk_count)
        .into_par_iter()
        .map(|chunk_index| {
            let src_start = chunk_index * source_chunk_len;
            let src_end = (src_start + source_chunk_len).min(num_nodes);
            let edge_total = (src_start..src_end)
                .map(|src| edge_plan.src_offsets[src + 1] - edge_plan.src_offsets[src])
                .sum::<usize>();
            let mut local_linear1_bars = vec![0.0; local_linear1_len];
            let mut edge_indices = Vec::with_capacity(edge_total);
            let mut local_edge_attr_bars = vec![0.0; edge_total * sh_dim];
            let mut local_edge_mlp_bars = vec![0.0; edge_total * tp_weight_numel];
            let mut tp_scatter_bar_row = vec![0.0; output_dim];
            let mut local_edge = 0;
            for src in src_start..src_end {
                let edge_range = edge_plan.src_offsets[src]..edge_plan.src_offsets[src + 1];
                if edge_range.is_empty() {
                    continue;
                }
                compact_linear_vjp_input_row_into_oam(
                    linear2_plan,
                    &gate_input_bars
                        [src * linear2_plan.output_dim..(src + 1) * linear2_plan.output_dim],
                    linear2_weight,
                    &mut tp_scatter_bar_row,
                );
                for &edge_index in &edge_plan.edges_by_src[edge_range] {
                    let edge = &edges[edge_index];
                    edge_indices.push(edge_index);
                    tensor_product_vjp_accumulate_oam(
                        plan,
                        &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                        &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                        &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                        &tp_scatter_bar_row,
                        &mut local_linear1_bars[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                        &mut local_edge_attr_bars[local_edge * sh_dim..(local_edge + 1) * sh_dim],
                        &mut local_edge_mlp_bars
                            [local_edge * tp_weight_numel..(local_edge + 1) * tp_weight_numel],
                    );
                    local_edge += 1;
                }
            }
            let local_hidden_silu_bars = input_major_linear_vjp_input(
                &local_edge_mlp_bars,
                edge_total,
                128,
                tp_weight_numel,
                edge_mlp_2_weight,
                edge_mlp_2_alpha,
            )?;
            Ok(OamFusedLinear2ChunkBars {
                normed_linear1_bars: local_linear1_bars,
                edge_indices,
                edge_attr_bars: local_edge_attr_bars,
                hidden_silu_bars: local_hidden_silu_bars,
            })
        })
        .collect::<Vec<Result<OamFusedLinear2ChunkBars, NequipOamPrefixError>>>();

    for chunk in chunks {
        let chunk = chunk?;
        for (target, value) in normed_linear1_bars
            .iter_mut()
            .zip(chunk.normed_linear1_bars)
        {
            *target += value;
        }
        for (local_edge, edge_index) in chunk.edge_indices.into_iter().enumerate() {
            let edge_attr_target =
                &mut edge_attr_bars[edge_index * sh_dim..(edge_index + 1) * sh_dim];
            let edge_attr_source =
                &chunk.edge_attr_bars[local_edge * sh_dim..(local_edge + 1) * sh_dim];
            for (target, value) in edge_attr_target.iter_mut().zip(edge_attr_source) {
                *target += *value;
            }
            let hidden_target = &mut hidden_silu_bars[edge_index * 128..(edge_index + 1) * 128];
            let hidden_source = &chunk.hidden_silu_bars[local_edge * 128..(local_edge + 1) * 128];
            for (target, value) in hidden_target.iter_mut().zip(hidden_source) {
                *target += *value;
            }
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "parallel"))]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn tensor_product_vjp_scatter_oam_parallel_direct_fused_linear2_edge_mlp2_f32(
    plan: &OamTensorProductPlan,
    edges: &[crate::Edge],
    edge_plan: &OamEdgePlan,
    edge_attrs: &[f32],
    sh_dim: usize,
    normed_linear1: &[f32],
    edge_mlp: &[f32],
    tp_weight_numel: usize,
    linear2_plan: &OamCompactLinearPlan,
    gate_input_bars: &[f32],
    linear2_weight: &[f32],
    edge_mlp_2_weight: &[f32],
    edge_mlp_2_alpha: f32,
    normed_linear1_bars: &mut [f32],
    edge_attr_bars: &mut [f32],
    hidden_silu_bars: &mut [f32],
    num_nodes: usize,
) -> Result<(), NequipOamPrefixError> {
    let edge_count = edges.len();
    let input1_dim = plan.input1_dim();
    let output_dim = plan.output_dim();
    if edge_count == 0 {
        return Ok(());
    }
    if linear2_plan.input_dim != output_dim
        || gate_input_bars.len() != num_nodes * linear2_plan.output_dim
        || linear2_weight.len() != linear2_plan.weight_numel
        || normed_linear1.len() != num_nodes * input1_dim
        || edge_attrs.len() != edge_count * sh_dim
        || edge_mlp.len() != edge_count * tp_weight_numel
        || hidden_silu_bars.len() != edge_count * 128
    {
        return Err(invalid_model(
            "parallel fused f32 linear2/TP/edge-MLP2 VJP shape mismatch",
        ));
    }

    let source_chunk_count = num_nodes.clamp(1, OAM_TP_VJP_PARALLEL_TARGET_CHUNKS);
    let source_chunk_len = num_nodes.div_ceil(source_chunk_count).max(1);
    let local_linear1_len = num_nodes * input1_dim;
    let chunks = (0..source_chunk_count)
        .into_par_iter()
        .map(|chunk_index| {
            let src_start = chunk_index * source_chunk_len;
            let src_end = (src_start + source_chunk_len).min(num_nodes);
            let edge_total = (src_start..src_end)
                .map(|src| edge_plan.src_offsets[src + 1] - edge_plan.src_offsets[src])
                .sum::<usize>();
            let mut local_linear1_bars = vec![0.0; local_linear1_len];
            let mut edge_indices = Vec::with_capacity(edge_total);
            let mut local_edge_attr_bars = vec![0.0; edge_total * sh_dim];
            let mut local_edge_mlp_bars = vec![0.0; edge_total * tp_weight_numel];
            let mut tp_scatter_bar_row = vec![0.0; output_dim];
            let mut local_edge = 0;
            for src in src_start..src_end {
                let edge_range = edge_plan.src_offsets[src]..edge_plan.src_offsets[src + 1];
                if edge_range.is_empty() {
                    continue;
                }
                compact_linear_vjp_input_row_into_oam_f32(
                    linear2_plan,
                    &gate_input_bars
                        [src * linear2_plan.output_dim..(src + 1) * linear2_plan.output_dim],
                    linear2_weight,
                    &mut tp_scatter_bar_row,
                );
                for &edge_index in &edge_plan.edges_by_src[edge_range] {
                    let edge = &edges[edge_index];
                    edge_indices.push(edge_index);
                    tensor_product_vjp_accumulate_oam_f32(
                        plan,
                        &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                        &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                        &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                        &tp_scatter_bar_row,
                        &mut local_linear1_bars[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                        &mut local_edge_attr_bars[local_edge * sh_dim..(local_edge + 1) * sh_dim],
                        &mut local_edge_mlp_bars
                            [local_edge * tp_weight_numel..(local_edge + 1) * tp_weight_numel],
                    );
                    local_edge += 1;
                }
            }
            let local_hidden_silu_bars = input_major_linear_vjp_input_f32(
                &local_edge_mlp_bars,
                edge_total,
                128,
                tp_weight_numel,
                edge_mlp_2_weight,
                edge_mlp_2_alpha,
            )?;
            Ok(OamFusedLinear2ChunkBarsF32 {
                normed_linear1_bars: local_linear1_bars,
                edge_indices,
                edge_attr_bars: local_edge_attr_bars,
                hidden_silu_bars: local_hidden_silu_bars,
            })
        })
        .collect::<Vec<Result<OamFusedLinear2ChunkBarsF32, NequipOamPrefixError>>>();

    for chunk in chunks {
        let chunk = chunk?;
        for (target, value) in normed_linear1_bars
            .iter_mut()
            .zip(chunk.normed_linear1_bars)
        {
            *target += value;
        }
        for (local_edge, edge_index) in chunk.edge_indices.into_iter().enumerate() {
            let edge_attr_target =
                &mut edge_attr_bars[edge_index * sh_dim..(edge_index + 1) * sh_dim];
            let edge_attr_source =
                &chunk.edge_attr_bars[local_edge * sh_dim..(local_edge + 1) * sh_dim];
            for (target, value) in edge_attr_target.iter_mut().zip(edge_attr_source) {
                *target += *value;
            }
            let hidden_target = &mut hidden_silu_bars[edge_index * 128..(edge_index + 1) * 128];
            let hidden_source = &chunk.hidden_silu_bars[local_edge * 128..(local_edge + 1) * 128];
            for (target, value) in hidden_target.iter_mut().zip(hidden_source) {
                *target += *value;
            }
        }
    }
    Ok(())
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
struct OamVjpEdgeBars {
    edge_index: usize,
    edge_attr_bars: Vec<f64>,
    edge_mlp_bars: Vec<f64>,
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
struct OamVjpDstBucket {
    linear1_bars: Vec<f64>,
    edge_bars: Vec<OamVjpEdgeBars>,
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn tensor_product_vjp_scatter_oam_parallel(
    plan: &OamTensorProductPlan,
    edges: &[crate::Edge],
    edge_plan: &OamEdgePlan,
    edge_attrs: &[f64],
    sh_dim: usize,
    normed_linear1: &[f64],
    edge_mlp: &[f64],
    tp_weight_numel: usize,
    tp_scatter_bars: &[f64],
    normed_linear1_bars: &mut [f64],
    edge_attr_bars: &mut [f64],
    edge_mlp_bars: &mut [f64],
    num_nodes: usize,
) {
    if edges.len() >= OAM_TP_VJP_PARALLEL_MIN_EDGES {
        tensor_product_vjp_scatter_oam_parallel_direct(
            plan,
            edges,
            edge_attrs,
            sh_dim,
            normed_linear1,
            edge_mlp,
            tp_weight_numel,
            tp_scatter_bars,
            normed_linear1_bars,
            edge_attr_bars,
            edge_mlp_bars,
            num_nodes,
        );
        return;
    }
    let input1_dim = plan.input1_dim();
    let output_dim = plan.output_dim();
    debug_assert_eq!(edge_plan.dst_offsets.len(), num_nodes + 1);

    let dst_buckets = (0..num_nodes)
        .into_par_iter()
        .map(|dst| {
            let mut linear1_bars = vec![0.0; input1_dim];
            let mut edge_bars = Vec::with_capacity(
                edge_plan.dst_offsets[dst + 1].saturating_sub(edge_plan.dst_offsets[dst]),
            );
            for &edge_index in
                &edge_plan.edges_by_dst[edge_plan.dst_offsets[dst]..edge_plan.dst_offsets[dst + 1]]
            {
                let mut edge_attr_bar = vec![0.0; sh_dim];
                let mut edge_mlp_bar = vec![0.0; tp_weight_numel];
                tensor_product_vjp_accumulate_oam(
                    plan,
                    &normed_linear1[dst * input1_dim..(dst + 1) * input1_dim],
                    &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                    &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                    &tp_scatter_bars[edges[edge_index].src * output_dim
                        ..(edges[edge_index].src + 1) * output_dim],
                    &mut linear1_bars,
                    &mut edge_attr_bar,
                    &mut edge_mlp_bar,
                );
                edge_bars.push(OamVjpEdgeBars {
                    edge_index,
                    edge_attr_bars: edge_attr_bar,
                    edge_mlp_bars: edge_mlp_bar,
                });
            }
            OamVjpDstBucket {
                linear1_bars,
                edge_bars,
            }
        })
        .collect::<Vec<_>>();

    for (dst, bucket) in dst_buckets.into_iter().enumerate() {
        let target = &mut normed_linear1_bars[dst * input1_dim..(dst + 1) * input1_dim];
        for (target, value) in target.iter_mut().zip(bucket.linear1_bars) {
            *target += value;
        }
        for edge_bars in bucket.edge_bars {
            let edge_attr_target = &mut edge_attr_bars
                [edge_bars.edge_index * sh_dim..(edge_bars.edge_index + 1) * sh_dim];
            for (target, value) in edge_attr_target.iter_mut().zip(edge_bars.edge_attr_bars) {
                *target += value;
            }
            let edge_mlp_target = &mut edge_mlp_bars[edge_bars.edge_index * tp_weight_numel
                ..(edge_bars.edge_index + 1) * tp_weight_numel];
            for (target, value) in edge_mlp_target.iter_mut().zip(edge_bars.edge_mlp_bars) {
                *target += value;
            }
        }
    }
}

impl NequipOamPrefixModel {
    pub fn from_json_str_with_external_tensors(
        text: &str,
        type_names: Vec<String>,
        tensors: HashMap<String, NequipOamPrefixExternalTensor>,
    ) -> Result<Self, NequipOamPrefixError> {
        let spec: Value = serde_json::from_str(text)
            .map_err(|err| invalid_model(format!("invalid NeQUIP-OAM prefix JSON: {err}")))?;
        expect_str(&spec, "schema_version", OAM_MODEL_SCHEMA_VERSION)?;
        expect_str(&spec, "selected_model", OAM_SELECTED_MODEL)?;
        expect_str(&spec, "state_dict_format", OAM_STATE_DICT_FORMAT)?;
        validate_pinned_type_names(&type_names)?;
        validate_oam_artifact_contract(&spec)?;
        let prefix = object_field(&spec, "prefix")?;
        let l_max = usize_field(prefix, "l_max")?;
        if l_max != OAM_L_MAX {
            return Err(invalid_model(format!(
                "only pinned OAM-L l_max={OAM_L_MAX} prefix is supported, got {l_max}"
            )));
        }
        let cutoff = f64_field(prefix, "cutoff")?;
        if (cutoff - OAM_CUTOFF).abs() > 1.0e-12 {
            return Err(invalid_model(format!(
                "only pinned OAM-L cutoff {OAM_CUTOFF} is supported, got {cutoff}"
            )));
        }
        let polynomial_cutoff_p = f64_field(prefix, "polynomial_cutoff_p")?;
        let edge_embedding_factor = f64_field(prefix, "edge_embedding_factor")?;
        let readout_alpha = f64_field(object_field(prefix, "readout")?, "alpha")?;
        let layers = array_field(prefix, "layers")?;
        if layers.len() != OAM_LAYER_COUNT {
            return Err(invalid_model(format!(
                "only pinned OAM-L {OAM_LAYER_COUNT}-layer prefix is supported, got {} layers",
                layers.len()
            )));
        }
        validate_oam_prefix_contract(prefix, &tensors)?;
        let sh_plan = SphericalHarmonicsPlan::new(l_max, true, ShNormalization::Component)?;
        let mut layer_plans = Vec::with_capacity(layers.len());
        let mut tp_plans = Vec::with_capacity(layers.len());
        let mut gate_plans = Vec::with_capacity(layers.len());
        for layer in layers {
            layer_plans.push(OamLayerPlan::from_json(layer)?);
            let tp_plan = tensor_product_plan_from_json(object_field(layer, "tensor_product")?)?;
            tp_plans.push(OamTensorProductPlan::from_plan(tp_plan)?);
            gate_plans.push(OamGatePlan::from_json(object_field(layer, "gate")?)?);
        }
        Ok(Self {
            type_names,
            spec,
            tensors,
            cutoff,
            polynomial_cutoff_p,
            polynomial_cutoff_p_f32: polynomial_cutoff_p as f32,
            edge_embedding_factor,
            edge_embedding_factor_f32: edge_embedding_factor as f32,
            readout_alpha,
            readout_alpha_f32: readout_alpha as f32,
            sh_plan,
            layer_plans,
            tp_plans,
            gate_plans,
            sc_contraction_cache: Arc::new(Mutex::new(HashMap::new())),
            sc_contraction_f32_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    pub fn predict(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
    ) -> Result<Prediction, NequipOamPrefixError> {
        if !options.compute_energy
            && !options.compute_forces
            && !options.compute_stress
            && !options.compute_virial
        {
            return Ok(Prediction::default());
        }
        let output = self.evaluate(structure, options)?;
        Ok(Prediction {
            energy: options.compute_energy.then_some(output.energy),
            forces: options.compute_forces.then_some(output.forces),
            stress: output.stress,
            virial: output.virial,
        })
    }

    pub fn predict_with_compute_dtype(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
        dtype: PredictionComputeDtype,
    ) -> Result<Prediction, NequipOamPrefixError> {
        match dtype {
            PredictionComputeDtype::Float64 => self.predict(structure, options),
            PredictionComputeDtype::Float32 => self.predict_energy_f32(structure, options),
        }
    }

    pub fn prepare_prediction_input(
        &self,
        structure: &Structure,
    ) -> Result<NequipOamPrefixPreparedInput, NequipOamPrefixError> {
        let has_pbc = structure.pbc.iter().any(|enabled| *enabled);
        if !has_pbc && structure.cell.is_some() {
            return Err(NequipOamPrefixError::NonPbcCell);
        }
        self.type_ids(structure)?;
        let neighbor_list = if has_pbc {
            cell_list_multi_image_neighbor_list(structure, self.cutoff())?
        } else {
            all_pairs_neighbor_list(structure, self.cutoff())?
        };
        for edge in &neighbor_list.edges {
            if edge.distance <= 0.0 {
                return Err(NequipOamPrefixError::InvalidStructure(format!(
                    "zero-distance directed edge {} -> {} is not supported by ZBL",
                    edge.src, edge.dst
                )));
            }
        }
        Ok(NequipOamPrefixPreparedInput {
            structure: structure.clone(),
            edge_count: neighbor_list.edges.len(),
        })
    }

    pub fn predict_prepared(
        &self,
        prepared: &NequipOamPrefixPreparedInput,
        options: &PredictionOptions,
    ) -> Result<Prediction, NequipOamPrefixError> {
        self.predict(&prepared.structure, options)
    }

    pub fn predict_prepared_with_compute_dtype(
        &self,
        prepared: &NequipOamPrefixPreparedInput,
        options: &PredictionOptions,
        dtype: PredictionComputeDtype,
    ) -> Result<Prediction, NequipOamPrefixError> {
        self.predict_with_compute_dtype(&prepared.structure, options, dtype)
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
        self.layer_plans.len()
    }

    fn evaluate(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
    ) -> Result<NequipOamPrefixOutput, NequipOamPrefixError> {
        let compute_vjp =
            options.compute_forces || options.compute_stress || options.compute_virial;
        let compute_displacement_gradient = options.compute_stress || options.compute_virial;
        let has_pbc = structure.pbc.iter().any(|enabled| *enabled);
        if !has_pbc && structure.cell.is_some() {
            return Err(NequipOamPrefixError::NonPbcCell);
        }
        if compute_displacement_gradient && structure.pbc != [true, true, true] {
            return Err(NequipOamPrefixError::StressVirialRequiresFullPbc);
        }
        let prefix = object_field(&self.spec, "prefix")?;
        let layers = array_field(prefix, "layers")?;
        let type_ids = self.type_ids(structure)?;
        let neighbor_list = if has_pbc {
            cell_list_multi_image_neighbor_list(structure, self.cutoff())?
        } else {
            all_pairs_neighbor_list(structure, self.cutoff())?
        };
        for edge in &neighbor_list.edges {
            if edge.distance <= 0.0 {
                return Err(NequipOamPrefixError::InvalidStructure(format!(
                    "zero-distance directed edge {} -> {} is not supported by ZBL",
                    edge.src, edge.dst
                )));
            }
        }
        #[cfg(feature = "parallel")]
        let use_parallel_forward = oam_tp_forward_parallel_enabled(neighbor_list.edges.len());
        #[cfg(not(feature = "parallel"))]
        let use_parallel_forward = false;
        let edge_plan = if compute_vjp || use_parallel_forward {
            Some(OamEdgePlan::new(
                &neighbor_list.edges,
                structure.len(),
                use_parallel_forward,
            ))
        } else {
            None
        };

        let sh_dim = self.sh_plan.output_dim();
        let mut edge_attrs = vec![0.0; neighbor_list.edges.len() * sh_dim];
        // Per-edge SH precompute: each edge writes only its own disjoint
        // `edge_attrs[i*sh_dim..(i+1)*sh_dim]` slice, so the parallel and serial
        // results are bit-identical regardless of thread count.
        let sh_done_parallel;
        #[cfg(feature = "parallel")]
        {
            sh_done_parallel = use_parallel_forward;
            if use_parallel_forward {
                let sh_plan = &self.sh_plan;
                edge_attrs
                    .par_chunks_mut(sh_dim)
                    .zip(neighbor_list.edges.par_iter())
                    .for_each(|(attr_row, edge)| {
                        sh_plan.forward_unchecked(edge.vector, attr_row);
                    });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            sh_done_parallel = false;
        }
        if !sh_done_parallel {
            let mut edge_attr = vec![0.0; sh_dim];
            for (edge, attr_row) in neighbor_list
                .edges
                .iter()
                .zip(edge_attrs.chunks_mut(sh_dim))
            {
                self.sh_plan.forward(edge.vector, &mut edge_attr)?;
                attr_row.copy_from_slice(&edge_attr);
            }
        }

        let rmax_recip = self.tensor_scalar("sole_model.model.func.edge_norm._rmax_recip")?;
        let bessel_weights = self.tensor(
            "sole_model.model.func.bessel_encode.bessel_weights",
            &[1, 8],
        )?;
        let cutoff_p = self.polynomial_cutoff_p;
        let edge_embedding_factor = self.edge_embedding_factor;
        let bessel_len = bessel_weights.len();
        let mut edge_cutoff = vec![0.0; neighbor_list.edges.len()];
        let mut factored_embedding = vec![0.0; neighbor_list.edges.len() * bessel_len];
        // Per-edge Bessel precompute: edge i writes only `edge_cutoff[i]` and
        // `factored_embedding[i*bessel_len..(i+1)*bessel_len]`; disjoint per edge.
        let bessel_done_parallel;
        #[cfg(feature = "parallel")]
        {
            bessel_done_parallel = use_parallel_forward;
            if use_parallel_forward {
                factored_embedding
                    .par_chunks_mut(bessel_len)
                    .zip(edge_cutoff.par_iter_mut())
                    .zip(neighbor_list.edges.par_iter())
                    .for_each(|((embedding_row, cutoff_slot), edge)| {
                        let normed = edge.distance * rmax_recip;
                        let cutoff = polynomial_cutoff(normed, cutoff_p);
                        *cutoff_slot = cutoff;
                        for (slot, weight) in embedding_row.iter_mut().zip(bessel_weights.iter()) {
                            let value = torch_sinc(normed * *weight) * *weight * cutoff;
                            *slot = value * edge_embedding_factor;
                        }
                    });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            bessel_done_parallel = false;
        }
        if !bessel_done_parallel {
            for ((edge, cutoff_slot), embedding_row) in neighbor_list
                .edges
                .iter()
                .zip(edge_cutoff.iter_mut())
                .zip(factored_embedding.chunks_mut(bessel_len))
            {
                let normed = edge.distance * rmax_recip;
                let cutoff = polynomial_cutoff(normed, cutoff_p);
                *cutoff_slot = cutoff;
                for (slot, weight) in embedding_row.iter_mut().zip(bessel_weights.iter()) {
                    let value = torch_sinc(normed * *weight) * *weight * cutoff;
                    *slot = value * edge_embedding_factor;
                }
            }
        }

        let type_embedding = self.tensor(
            "sole_model.model.func.type_embed.embed_module.weight",
            &[self.type_names.len(), 48],
        )?;
        let mut node_features = Vec::with_capacity(type_ids.len() * 48);
        for type_id in &type_ids {
            let row = *type_id * 48;
            node_features.extend_from_slice(&type_embedding[row..row + 48]);
        }
        let active_type_rows = active_type_rows(&type_ids, self.type_names.len(), structure.len())?;
        let sc_contractions_by_layer = self.sc_contractions_for_active_types(&active_type_rows)?;

        let mut layer_input = node_features.clone();
        let mut layer_caches = Vec::new();
        let mut layer_inputs = Vec::new();
        if compute_vjp {
            layer_caches.reserve(layers.len());
            layer_inputs.reserve(layers.len());
        }
        let mut tp_scatter_workspace = Vec::new();
        for (layer_index, layer_spec) in layers.iter().enumerate() {
            if compute_vjp {
                layer_inputs.push(layer_input.clone());
            }
            let (next_input, cache) = self.layer_forward_with_cache(
                layer_index,
                structure.len(),
                &neighbor_list.edges,
                edge_plan.as_ref(),
                &edge_attrs,
                &factored_embedding,
                &active_type_rows,
                &sc_contractions_by_layer,
                &layer_input,
                layer_spec,
                &mut tp_scatter_workspace,
            )?;
            if let Some(cache) = cache {
                layer_caches.push(cache);
            }
            layer_input = next_input;
        }

        let readout_weight = self.tensor(
            "sole_model.model.func.per_atom_energy_readout.mlp_module.mlp.0.weight",
            &[128, 1],
        )?;
        let readout_alpha = self.readout_alpha;
        let readout = input_major_linear(
            &layer_input,
            structure.len(),
            128,
            1,
            readout_weight,
            readout_alpha,
        )?;

        let scales = self.tensor(
            "sole_model.model.func.per_type_energy_scale_shift.scales",
            &[self.type_names.len(), 1],
        )?;
        let shifts = self.tensor(
            "sole_model.model.func.per_type_energy_scale_shift.shifts",
            &[self.type_names.len(), 1],
        )?;
        let mut atomic_energy = Vec::with_capacity(structure.len());
        let mut readout_bars = if compute_vjp {
            vec![0.0; structure.len()]
        } else {
            Vec::new()
        };
        for (node, type_id) in type_ids.iter().enumerate() {
            atomic_energy.push(readout[node] * scales[*type_id] + shifts[*type_id]);
            if compute_vjp {
                readout_bars[node] += scales[*type_id];
            }
        }
        let feature_bars = if compute_vjp {
            input_major_linear_vjp_input(
                &readout_bars,
                structure.len(),
                128,
                1,
                readout_weight,
                readout_alpha,
            )?
        } else {
            Vec::new()
        };

        let atomic_numbers = self.tensor(
            "sole_model.model.func.pair_potential.atomic_numbers",
            &[self.type_names.len()],
        )?;
        let qqr2exesquare =
            self.tensor_scalar("sole_model.model.func.pair_potential._qqr2exesquare")?;
        let mut edge_distance_bars = if compute_vjp {
            vec![0.0; neighbor_list.edges.len()]
        } else {
            Vec::new()
        };
        for (edge_index, edge) in neighbor_list.edges.iter().enumerate() {
            let (raw, raw_derivative) = nequip_zbl_edge_energy_and_derivative(
                edge.distance,
                type_ids[edge.src],
                type_ids[edge.dst],
                atomic_numbers,
                qqr2exesquare,
            );
            atomic_energy[edge.src] += raw * edge_cutoff[edge_index];
            if compute_vjp {
                let normed = edge.distance * rmax_recip;
                edge_distance_bars[edge_index] += raw_derivative * edge_cutoff[edge_index]
                    + raw * polynomial_cutoff_derivative(normed, cutoff_p) * rmax_recip;
            }
        }

        let energy = atomic_energy.iter().sum();
        let (forces, stress, virial) = if compute_vjp {
            let edge_plan = edge_plan
                .as_ref()
                .ok_or_else(|| invalid_model("missing OAM edge plan for force VJP"))?;
            let vjp = self.reverse_forces(
                &neighbor_list.edges,
                edge_plan,
                &edge_attrs,
                rmax_recip,
                bessel_weights,
                cutoff_p,
                edge_embedding_factor,
                &active_type_rows,
                &sc_contractions_by_layer,
                &layer_inputs,
                &layer_caches,
                feature_bars,
                edge_distance_bars,
                structure.len(),
                compute_displacement_gradient,
            )?;
            vjp.into_output_tensors(structure, options)?
        } else {
            (Vec::new(), None, None)
        };

        Ok(NequipOamPrefixOutput {
            energy,
            forces,
            stress,
            virial,
        })
    }

    fn predict_energy_f32(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
    ) -> Result<Prediction, NequipOamPrefixError> {
        if !options.compute_energy
            && !options.compute_forces
            && !options.compute_stress
            && !options.compute_virial
        {
            return Ok(Prediction::default());
        }
        if options.compute_stress || options.compute_virial {
            return Err(NequipOamPrefixError::UnsupportedComputeDtype(
                PredictionComputeDtype::Float32,
            ));
        }
        let compute_vjp = options.compute_forces;
        if !self.has_native_f32_state_tensors() {
            return Err(NequipOamPrefixError::UnsupportedComputeDtype(
                PredictionComputeDtype::Float32,
            ));
        }

        let has_pbc = structure.pbc.iter().any(|enabled| *enabled);
        if !has_pbc && structure.cell.is_some() {
            return Err(NequipOamPrefixError::NonPbcCell);
        }
        let prefix = object_field(&self.spec, "prefix")?;
        let layers = array_field(prefix, "layers")?;
        let type_ids = self.type_ids(structure)?;
        let neighbor_list = if has_pbc {
            cell_list_multi_image_neighbor_list(structure, self.cutoff())?
        } else {
            all_pairs_neighbor_list(structure, self.cutoff())?
        };
        for edge in &neighbor_list.edges {
            if edge.distance <= 0.0 {
                return Err(NequipOamPrefixError::InvalidStructure(format!(
                    "zero-distance directed edge {} -> {} is not supported by ZBL",
                    edge.src, edge.dst
                )));
            }
        }
        #[cfg(feature = "parallel")]
        let use_parallel_forward = oam_tp_forward_parallel_enabled(neighbor_list.edges.len());
        #[cfg(not(feature = "parallel"))]
        let use_parallel_forward = false;
        let edge_plan = if compute_vjp || use_parallel_forward {
            Some(OamEdgePlan::new(
                &neighbor_list.edges,
                structure.len(),
                use_parallel_forward,
            ))
        } else {
            None
        };

        let sh_dim = self.sh_plan.output_dim();
        let mut edge_attrs = vec![0.0f32; neighbor_list.edges.len() * sh_dim];
        // Per-edge SH precompute (f64 SH then cast to f32, matching the serial
        // path exactly): edge i writes only `edge_attrs[i*sh_dim..(i+1)*sh_dim]`,
        // so parallel and serial results are bit-identical.
        let sh_done_parallel;
        #[cfg(feature = "parallel")]
        {
            sh_done_parallel = use_parallel_forward;
            if use_parallel_forward {
                let sh_plan = &self.sh_plan;
                let edge_count = neighbor_list.edges.len();
                let chunk_count = edge_count.clamp(1, OAM_TP_VJP_PARALLEL_TARGET_CHUNKS);
                let edge_chunk_len = edge_count.div_ceil(chunk_count).max(1);
                edge_attrs
                    .par_chunks_mut(edge_chunk_len * sh_dim)
                    .enumerate()
                    .for_each(|(chunk_index, attr_chunk)| {
                        let edge_start = chunk_index * edge_chunk_len;
                        let chunk_edges = attr_chunk.len() / sh_dim;
                        let mut sh_f64 = vec![0.0; sh_dim];
                        for local_edge in 0..chunk_edges {
                            let edge = &neighbor_list.edges[edge_start + local_edge];
                            sh_plan.forward_unchecked(edge.vector, &mut sh_f64);
                            let attr_row =
                                &mut attr_chunk[local_edge * sh_dim..(local_edge + 1) * sh_dim];
                            for (dst, &src) in attr_row.iter_mut().zip(sh_f64.iter()) {
                                *dst = src as f32;
                            }
                        }
                    });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            sh_done_parallel = false;
        }
        if !sh_done_parallel {
            let mut edge_attr = vec![0.0; sh_dim];
            for (edge, attr_row) in neighbor_list
                .edges
                .iter()
                .zip(edge_attrs.chunks_mut(sh_dim))
            {
                self.sh_plan.forward(edge.vector, &mut edge_attr)?;
                for (dst, &src) in attr_row.iter_mut().zip(edge_attr.iter()) {
                    *dst = src as f32;
                }
            }
        }

        let rmax_recip = self.tensor_scalar_f32("sole_model.model.func.edge_norm._rmax_recip")?;
        let bessel_weights = self.tensor_f32_cow(
            "sole_model.model.func.bessel_encode.bessel_weights",
            &[1, 8],
        )?;
        let cutoff_p = self.polynomial_cutoff_p_f32;
        let edge_embedding_factor = self.edge_embedding_factor_f32;
        let bessel_len = bessel_weights.len();
        let mut edge_cutoff = vec![0.0f32; neighbor_list.edges.len()];
        let mut factored_embedding = vec![0.0f32; neighbor_list.edges.len() * bessel_len];
        // Per-edge Bessel precompute: edge i writes only `edge_cutoff[i]` and
        // `factored_embedding[i*bessel_len..(i+1)*bessel_len]`; disjoint per edge.
        let bessel_done_parallel;
        #[cfg(feature = "parallel")]
        {
            bessel_done_parallel = use_parallel_forward;
            if use_parallel_forward {
                let bessel_weights = bessel_weights.as_ref();
                factored_embedding
                    .par_chunks_mut(bessel_len)
                    .zip(edge_cutoff.par_iter_mut())
                    .zip(neighbor_list.edges.par_iter())
                    .for_each(|((embedding_row, cutoff_slot), edge)| {
                        let normed = edge.distance as f32 * rmax_recip;
                        let cutoff =
                            polynomial_cutoff(f64::from(normed), f64::from(cutoff_p)) as f32;
                        *cutoff_slot = cutoff;
                        for (slot, &weight) in embedding_row.iter_mut().zip(bessel_weights.iter()) {
                            let value =
                                torch_sinc(f64::from(normed * weight)) as f32 * weight * cutoff;
                            *slot = value * edge_embedding_factor;
                        }
                    });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            bessel_done_parallel = false;
        }
        if !bessel_done_parallel {
            for ((edge, cutoff_slot), embedding_row) in neighbor_list
                .edges
                .iter()
                .zip(edge_cutoff.iter_mut())
                .zip(factored_embedding.chunks_mut(bessel_len))
            {
                let normed = edge.distance as f32 * rmax_recip;
                let cutoff = polynomial_cutoff(f64::from(normed), f64::from(cutoff_p)) as f32;
                *cutoff_slot = cutoff;
                for (slot, &weight) in embedding_row.iter_mut().zip(bessel_weights.iter()) {
                    let value = torch_sinc(f64::from(normed * weight)) as f32 * weight * cutoff;
                    *slot = value * edge_embedding_factor;
                }
            }
        }

        let type_embedding = self.tensor_f32_cow(
            "sole_model.model.func.type_embed.embed_module.weight",
            &[self.type_names.len(), 48],
        )?;
        let mut node_features = Vec::with_capacity(type_ids.len() * 48);
        for type_id in &type_ids {
            let row = *type_id * 48;
            node_features.extend_from_slice(&type_embedding[row..row + 48]);
        }
        let active_type_rows = active_type_rows(&type_ids, self.type_names.len(), structure.len())?;
        let sc_contractions_by_layer =
            self.sc_contractions_for_active_types_f32(&active_type_rows)?;

        let mut layer_input = node_features;
        let mut layer_caches = Vec::new();
        let mut layer_inputs = Vec::new();
        if compute_vjp {
            layer_caches.reserve(layers.len());
            layer_inputs.reserve(layers.len());
        }
        let mut tp_scatter_workspace = Vec::new();
        for (layer_index, layer_spec) in layers.iter().enumerate() {
            if compute_vjp {
                layer_inputs.push(layer_input.clone());
            }
            let (next_input, cache) = self.layer_forward_f32(
                layer_index,
                structure.len(),
                &neighbor_list.edges,
                edge_plan.as_ref(),
                &edge_attrs,
                &factored_embedding,
                &active_type_rows,
                &sc_contractions_by_layer,
                &layer_input,
                layer_spec,
                &mut tp_scatter_workspace,
            )?;
            if let Some(cache) = cache {
                layer_caches.push(cache);
            }
            layer_input = next_input;
        }

        let readout_weight = self.tensor_f32_cow(
            "sole_model.model.func.per_atom_energy_readout.mlp_module.mlp.0.weight",
            &[128, 1],
        )?;
        let readout_alpha = self.readout_alpha_f32;
        let readout = input_major_linear_f32(
            &layer_input,
            structure.len(),
            128,
            1,
            &readout_weight,
            readout_alpha,
        )?;

        let scales = self.tensor_f32_cow(
            "sole_model.model.func.per_type_energy_scale_shift.scales",
            &[self.type_names.len(), 1],
        )?;
        let shifts = self.tensor_f32_cow(
            "sole_model.model.func.per_type_energy_scale_shift.shifts",
            &[self.type_names.len(), 1],
        )?;
        let mut atomic_energy = Vec::with_capacity(structure.len());
        let mut readout_bars = if compute_vjp {
            vec![0.0; structure.len()]
        } else {
            Vec::new()
        };
        for (node, type_id) in type_ids.iter().enumerate() {
            atomic_energy.push(f64::from(
                readout[node] * scales[*type_id] + shifts[*type_id],
            ));
            if compute_vjp {
                readout_bars[node] += scales[*type_id];
            }
        }
        let feature_bars = if compute_vjp {
            input_major_linear_vjp_input_f32(
                &readout_bars,
                structure.len(),
                128,
                1,
                &readout_weight,
                readout_alpha,
            )?
        } else {
            Vec::new()
        };

        let atomic_numbers = self.tensor(
            "sole_model.model.func.pair_potential.atomic_numbers",
            &[self.type_names.len()],
        )?;
        let qqr2exesquare =
            self.tensor_scalar("sole_model.model.func.pair_potential._qqr2exesquare")?;
        let mut edge_distance_bars = if compute_vjp {
            vec![0.0; neighbor_list.edges.len()]
        } else {
            Vec::new()
        };
        for (edge_index, edge) in neighbor_list.edges.iter().enumerate() {
            let (raw, raw_derivative) = nequip_zbl_edge_energy_and_derivative(
                edge.distance,
                type_ids[edge.src],
                type_ids[edge.dst],
                atomic_numbers,
                qqr2exesquare,
            );
            atomic_energy[edge.src] += raw * f64::from(edge_cutoff[edge_index]);
            if compute_vjp {
                let normed = edge.distance * f64::from(rmax_recip);
                edge_distance_bars[edge_index] += raw_derivative
                    * f64::from(edge_cutoff[edge_index])
                    + raw
                        * polynomial_cutoff_derivative(normed, f64::from(cutoff_p))
                        * f64::from(rmax_recip);
            }
        }

        let forces = if compute_vjp {
            let edge_plan = edge_plan
                .as_ref()
                .ok_or_else(|| invalid_model("missing OAM f32 edge plan for force VJP"))?;
            let vjp = self.reverse_forces_f32(
                &neighbor_list.edges,
                edge_plan,
                &edge_attrs,
                rmax_recip,
                &bessel_weights,
                cutoff_p,
                edge_embedding_factor,
                &active_type_rows,
                &sc_contractions_by_layer,
                &layer_inputs,
                &layer_caches,
                feature_bars,
                edge_distance_bars,
                structure.len(),
            )?;
            Some(vjp.forces)
        } else {
            None
        };

        Ok(Prediction {
            energy: options.compute_energy.then_some(atomic_energy.iter().sum()),
            forces,
            stress: None,
            virial: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_forward_with_cache(
        &self,
        layer_index: usize,
        num_nodes: usize,
        edges: &[crate::Edge],
        edge_plan: Option<&OamEdgePlan>,
        edge_attrs: &[f64],
        factored_embedding: &[f64],
        active_type_rows: &ActiveTypeRows,
        sc_contractions_by_layer: &ScalarAttrContractionsByLayer,
        layer_input: &[f64],
        layer_spec: &Value,
        tp_scatter_workspace: &mut Vec<f64>,
    ) -> Result<(Vec<f64>, Option<NequipOamLayerCache>), NequipOamPrefixError> {
        #[cfg(not(feature = "parallel"))]
        let _ = edge_plan;
        let layer_plan = &self.layer_plans[layer_index];
        let tensor_prefix = format!("sole_model.model.func.layer{layer_index}_convnet");
        let tp_plan = &self.tp_plans[layer_index];
        let linear1_weight = self.tensor_by_numel(
            &format!("{tensor_prefix}.conv.linear_1.weight"),
            usize_field(object_field(layer_spec, "linear1")?, "weight_numel")?,
        )?;
        let mut normed_linear1_tp = compact_linear_forward_mapped(
            layer_input,
            num_nodes,
            object_field(layer_spec, "linear1")?,
            linear1_weight,
            None,
            Some(&tp_plan.input1_original_to_tp),
        )?;

        let tp_weight_numel = tp_plan.weight_numel;
        let edge_mlp_0_weight = self.tensor(
            &format!("{tensor_prefix}.conv.edge_mlp.mlp.0.weight"),
            &[8, 128],
        )?;
        let edge_mlp_2_weight = self.tensor(
            &format!("{tensor_prefix}.conv.edge_mlp.mlp.2.weight"),
            &[128, tp_weight_numel],
        )?;
        let mut hidden = input_major_linear(
            factored_embedding,
            edges.len(),
            8,
            128,
            edge_mlp_0_weight,
            layer_plan.edge_mlp_alpha0,
        )?;
        let mut hidden_silu_derivative = Vec::with_capacity(hidden.len());
        for value in &mut hidden {
            let (activated, derivative) = silu_and_derivative(*value);
            *value = activated;
            hidden_silu_derivative.push(derivative);
        }
        let edge_mlp = input_major_linear(
            &hidden,
            edges.len(),
            128,
            tp_weight_numel,
            edge_mlp_2_weight,
            layer_plan.edge_mlp_alpha1,
        )?;

        let scatter_norm = layer_plan.scatter_norm;
        for value in &mut normed_linear1_tp {
            *value *= scatter_norm;
        }
        let sh_dim = self.sh_plan.output_dim();
        let linear2_spec = object_field(layer_spec, "linear2")?;
        let linear2_weight = self.tensor_by_numel(
            &format!("{tensor_prefix}.conv.linear_2.weight"),
            usize_field(linear2_spec, "weight_numel")?,
        )?;
        tp_scatter_workspace.resize(num_nodes * tp_plan.output_dim(), 0.0);
        tp_scatter_workspace.fill(0.0);
        #[cfg(feature = "parallel")]
        if oam_tp_forward_parallel_enabled(edges.len()) {
            let edge_plan =
                edge_plan.ok_or_else(|| invalid_model("missing OAM edge plan for TP forward"))?;
            tensor_product_forward_scatter_oam_parallel_tp_layout(
                tp_plan,
                edges,
                edge_plan,
                edge_attrs,
                sh_dim,
                &normed_linear1_tp,
                &edge_mlp,
                tp_weight_numel,
                tp_scatter_workspace,
            );
        } else {
            for (edge_index, edge) in edges.iter().enumerate() {
                let scatter_row = &mut tp_scatter_workspace
                    [edge.src * tp_plan.output_dim()..(edge.src + 1) * tp_plan.output_dim()];
                tensor_product_forward_accumulate_oam_tp_layout(
                    tp_plan,
                    &normed_linear1_tp
                        [edge.dst * tp_plan.input1_dim()..(edge.dst + 1) * tp_plan.input1_dim()],
                    &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                    &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                    scatter_row,
                );
            }
        }
        #[cfg(not(feature = "parallel"))]
        for (edge_index, edge) in edges.iter().enumerate() {
            let scatter_row = &mut tp_scatter_workspace
                [edge.src * tp_plan.output_dim()..(edge.src + 1) * tp_plan.output_dim()];
            tensor_product_forward_accumulate_oam_tp_layout(
                tp_plan,
                &normed_linear1_tp
                    [edge.dst * tp_plan.input1_dim()..(edge.dst + 1) * tp_plan.input1_dim()],
                &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                scatter_row,
            );
        }
        let mut conv = compact_linear_forward_mapped(
            tp_scatter_workspace,
            num_nodes,
            linear2_spec,
            linear2_weight,
            Some(&tp_plan.output_original_to_tp),
            None,
        )?;
        if layer_spec.get("self_connection").is_some() {
            let contractions = sc_contractions_by_layer[layer_index]
                .as_ref()
                .ok_or_else(|| invalid_model("missing scalar-attribute contraction cache"))?;
            #[cfg(feature = "parallel")]
            if rayon::current_num_threads() > 1 && num_nodes >= OAM_COMPACT_LINEAR_PARALLEL_MIN_ROWS
            {
                scalar_attr_tensor_product_forward_accumulate_parallel(
                    layer_input,
                    active_type_rows,
                    num_nodes,
                    contractions,
                    &mut conv,
                )?;
            } else {
                scalar_attr_tensor_product_forward_accumulate(
                    layer_input,
                    active_type_rows,
                    num_nodes,
                    contractions,
                    &mut conv,
                )?;
            }
            #[cfg(not(feature = "parallel"))]
            scalar_attr_tensor_product_forward_accumulate(
                layer_input,
                active_type_rows,
                num_nodes,
                contractions,
                &mut conv,
            )?;
        }
        #[cfg(feature = "parallel")]
        let cache_gate = rayon::current_num_threads() > 1;
        #[cfg(not(feature = "parallel"))]
        let cache_gate = false;
        let gate_plan = &self.gate_plans[layer_index];
        let (output, gate) = if cache_gate {
            let (output, gate) = gate_forward_prepared(&conv, num_nodes, gate_plan)?;
            (output, Some(gate))
        } else {
            (gate_forward_plan(&conv, num_nodes, gate_plan)?, None)
        };
        Ok((
            output,
            Some(NequipOamLayerCache {
                normed_linear1: normed_linear1_tp,
                hidden_silu_derivative,
                edge_mlp,
                gate,
                gate_input: conv,
            }),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_forward_f32(
        &self,
        layer_index: usize,
        num_nodes: usize,
        edges: &[crate::Edge],
        edge_plan: Option<&OamEdgePlan>,
        edge_attrs: &[f32],
        factored_embedding: &[f32],
        active_type_rows: &ActiveTypeRows,
        sc_contractions_by_layer: &ScalarAttrContractionsByLayerF32,
        layer_input: &[f32],
        layer_spec: &Value,
        tp_scatter_workspace: &mut Vec<f32>,
    ) -> Result<(Vec<f32>, Option<NequipOamLayerCacheF32>), NequipOamPrefixError> {
        #[cfg(not(feature = "parallel"))]
        let _ = edge_plan;
        let layer_plan = &self.layer_plans[layer_index];
        let tensor_prefix = format!("sole_model.model.func.layer{layer_index}_convnet");
        let tp_plan = &self.tp_plans[layer_index];
        let linear1_weight = self.tensor_by_numel_f32_cow(
            &format!("{tensor_prefix}.conv.linear_1.weight"),
            usize_field(object_field(layer_spec, "linear1")?, "weight_numel")?,
        )?;
        let mut normed_linear1_tp = compact_linear_forward_mapped_f32(
            layer_input,
            num_nodes,
            object_field(layer_spec, "linear1")?,
            &linear1_weight,
            None,
            Some(&tp_plan.input1_original_to_tp),
        )?;

        let tp_weight_numel = tp_plan.weight_numel;
        let edge_mlp_0_weight = self.tensor_f32_cow(
            &format!("{tensor_prefix}.conv.edge_mlp.mlp.0.weight"),
            &[8, 128],
        )?;
        let edge_mlp_2_weight = self.tensor_f32_cow(
            &format!("{tensor_prefix}.conv.edge_mlp.mlp.2.weight"),
            &[128, tp_weight_numel],
        )?;
        let mut hidden = input_major_linear_f32(
            factored_embedding,
            edges.len(),
            8,
            128,
            &edge_mlp_0_weight,
            layer_plan.edge_mlp_alpha0_f32,
        )?;
        let mut hidden_silu_derivative = Vec::with_capacity(hidden.len());
        for value in &mut hidden {
            let (activated, derivative) = silu_and_derivative_f32(*value);
            *value = activated;
            hidden_silu_derivative.push(derivative);
        }
        let edge_mlp = input_major_linear_f32(
            &hidden,
            edges.len(),
            128,
            tp_weight_numel,
            &edge_mlp_2_weight,
            layer_plan.edge_mlp_alpha1_f32,
        )?;

        let scatter_norm = layer_plan.scatter_norm_f32;
        for value in &mut normed_linear1_tp {
            *value *= scatter_norm;
        }
        let sh_dim = self.sh_plan.output_dim();
        tp_scatter_workspace.resize(num_nodes * tp_plan.output_dim(), 0.0);
        tp_scatter_workspace.fill(0.0);
        #[cfg(feature = "parallel")]
        if oam_tp_forward_parallel_enabled(edges.len()) {
            let edge_plan = edge_plan
                .ok_or_else(|| invalid_model("missing OAM f32 edge plan for TP forward"))?;
            tensor_product_forward_scatter_oam_parallel_tp_layout_f32(
                tp_plan,
                edges,
                edge_plan,
                edge_attrs,
                sh_dim,
                &normed_linear1_tp,
                &edge_mlp,
                tp_weight_numel,
                tp_scatter_workspace,
            );
        } else {
            for (edge_index, edge) in edges.iter().enumerate() {
                let scatter_row = &mut tp_scatter_workspace
                    [edge.src * tp_plan.output_dim()..(edge.src + 1) * tp_plan.output_dim()];
                tensor_product_forward_accumulate_oam_tp_layout_f32(
                    tp_plan,
                    &normed_linear1_tp
                        [edge.dst * tp_plan.input1_dim()..(edge.dst + 1) * tp_plan.input1_dim()],
                    &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                    &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                    scatter_row,
                );
            }
        }
        #[cfg(not(feature = "parallel"))]
        for (edge_index, edge) in edges.iter().enumerate() {
            let scatter_row = &mut tp_scatter_workspace
                [edge.src * tp_plan.output_dim()..(edge.src + 1) * tp_plan.output_dim()];
            tensor_product_forward_accumulate_oam_tp_layout_f32(
                tp_plan,
                &normed_linear1_tp
                    [edge.dst * tp_plan.input1_dim()..(edge.dst + 1) * tp_plan.input1_dim()],
                &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                scatter_row,
            );
        }
        let linear2_spec = object_field(layer_spec, "linear2")?;
        let linear2_weight = self.tensor_by_numel_f32_cow(
            &format!("{tensor_prefix}.conv.linear_2.weight"),
            usize_field(linear2_spec, "weight_numel")?,
        )?;
        let mut conv = compact_linear_forward_mapped_f32(
            tp_scatter_workspace,
            num_nodes,
            linear2_spec,
            &linear2_weight,
            Some(&tp_plan.output_original_to_tp),
            None,
        )?;
        if layer_spec.get("self_connection").is_some() {
            let contractions = sc_contractions_by_layer[layer_index]
                .as_ref()
                .ok_or_else(|| invalid_model("missing scalar-attribute f32 contraction cache"))?;
            #[cfg(feature = "parallel")]
            if rayon::current_num_threads() > 1 && num_nodes >= OAM_COMPACT_LINEAR_PARALLEL_MIN_ROWS
            {
                scalar_attr_tensor_product_forward_accumulate_parallel_f32(
                    layer_input,
                    active_type_rows,
                    num_nodes,
                    contractions,
                    &mut conv,
                )?;
            } else {
                scalar_attr_tensor_product_forward_accumulate_f32(
                    layer_input,
                    active_type_rows,
                    num_nodes,
                    contractions,
                    &mut conv,
                )?;
            }
            #[cfg(not(feature = "parallel"))]
            scalar_attr_tensor_product_forward_accumulate_f32(
                layer_input,
                active_type_rows,
                num_nodes,
                contractions,
                &mut conv,
            )?;
        }
        let gate_plan = &self.gate_plans[layer_index];
        let (output, gate) = gate_forward_prepared_f32(&conv, num_nodes, gate_plan)?;
        let gate = Some(gate);
        Ok((
            output,
            Some(NequipOamLayerCacheF32 {
                normed_linear1: normed_linear1_tp,
                hidden_silu_derivative,
                edge_mlp,
                gate,
                gate_input: conv,
            }),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn reverse_forces(
        &self,
        edges: &[crate::Edge],
        edge_plan: &OamEdgePlan,
        edge_attrs: &[f64],
        rmax_recip: f64,
        bessel_weights: &[f64],
        cutoff_p: f64,
        edge_embedding_factor: f64,
        active_type_rows: &ActiveTypeRows,
        sc_contractions_by_layer: &ScalarAttrContractionsByLayer,
        layer_inputs: &[Vec<f64>],
        layer_caches: &[NequipOamLayerCache],
        mut feature_bars: Vec<f64>,
        mut edge_distance_bars: Vec<f64>,
        num_nodes: usize,
        compute_displacement_gradient: bool,
    ) -> Result<NequipOamPrefixVjp, NequipOamPrefixError> {
        let prefix = object_field(&self.spec, "prefix")?;
        let layers = array_field(prefix, "layers")?;
        let sh_dim = self.sh_plan.output_dim();
        let edge_count = edges.len();
        let mut edge_attr_bars = vec![0.0; edge_count * sh_dim];
        let mut factored_embedding_bars = vec![0.0; edge_count * bessel_weights.len()];

        for layer_index in (0..layers.len()).rev() {
            let layer_spec = &layers[layer_index];
            let layer_plan = &self.layer_plans[layer_index];
            let cache = &layer_caches[layer_index];
            let layer_input = &layer_inputs[layer_index];
            let tensor_prefix = format!("sole_model.model.func.layer{layer_index}_convnet");

            let gate_plan = &self.gate_plans[layer_index];
            let gate_input_bars = if let Some(gate_cache) = &cache.gate {
                gate_vjp_input_prepared(
                    &cache.gate_input,
                    num_nodes,
                    gate_plan,
                    gate_cache,
                    &feature_bars,
                )?
            } else {
                gate_vjp_input_plan(&cache.gate_input, num_nodes, gate_plan, &feature_bars)?
            };

            let tp_plan = &self.tp_plans[layer_index];
            let linear2_spec = object_field(layer_spec, "linear2")?;
            let linear2_weight = self.tensor_by_numel(
                &format!("{tensor_prefix}.conv.linear_2.weight"),
                usize_field(linear2_spec, "weight_numel")?,
            )?;
            let tp_scatter_bars_tp = compact_linear_vjp_input_mapped(
                &gate_input_bars,
                num_nodes,
                linear2_spec,
                linear2_weight,
                None,
                Some(&tp_plan.output_original_to_tp),
            )?;

            let mut previous_feature_bars = vec![0.0; layer_input.len()];
            if layer_spec.get("self_connection").is_some() {
                let sc_contractions = sc_contractions_by_layer[layer_index]
                    .as_ref()
                    .ok_or_else(|| invalid_model("missing scalar-attribute contraction cache"))?;
                scalar_attr_tensor_product_vjp_input_accumulate_auto(
                    &gate_input_bars,
                    layer_input,
                    active_type_rows,
                    num_nodes,
                    sc_contractions,
                    &mut previous_feature_bars,
                )?;
            }

            let tp_weight_numel = tp_plan.weight_numel;
            let mut normed_linear1_bars = vec![0.0; cache.normed_linear1.len()];
            let input1_dim = tp_plan.input1_dim();
            let edge_mlp_2_weight = self.tensor(
                &format!("{tensor_prefix}.conv.edge_mlp.mlp.2.weight"),
                &[128, tp_weight_numel],
            )?;
            #[cfg(feature = "parallel")]
            let hidden_silu_bars = if rayon::current_num_threads() > 1
                && edge_count >= OAM_TP_VJP_PARALLEL_MIN_EDGES
            {
                let mut fused_hidden_silu_bars = vec![0.0; edge_count * 128];
                tensor_product_vjp_scatter_oam_parallel_direct_fused_edge_mlp2_tp_layout(
                    tp_plan,
                    edges,
                    edge_plan,
                    edge_attrs,
                    sh_dim,
                    &cache.normed_linear1,
                    &cache.edge_mlp,
                    tp_weight_numel,
                    &tp_scatter_bars_tp,
                    edge_mlp_2_weight,
                    layer_plan.edge_mlp_alpha1,
                    &mut normed_linear1_bars,
                    &mut edge_attr_bars,
                    &mut fused_hidden_silu_bars,
                    num_nodes,
                )?;
                fused_hidden_silu_bars
            } else {
                let mut edge_mlp_bars = vec![0.0; cache.edge_mlp.len()];
                let mut dst_grad_x = vec![0.0; input1_dim];
                for dst in 0..num_nodes {
                    let edge_range = edge_plan.dst_offsets[dst]..edge_plan.dst_offsets[dst + 1];
                    if edge_range.is_empty() {
                        continue;
                    }
                    dst_grad_x.fill(0.0);
                    for &edge_index in &edge_plan.edges_by_dst[edge_range] {
                        let edge = &edges[edge_index];
                        let edge_attr_bar =
                            &mut edge_attr_bars[edge_index * sh_dim..(edge_index + 1) * sh_dim];
                        let edge_mlp_bar = &mut edge_mlp_bars
                            [edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel];
                        tensor_product_vjp_accumulate_oam_tp_layout(
                            tp_plan,
                            &cache.normed_linear1[edge.dst * tp_plan.input1_dim()
                                ..(edge.dst + 1) * tp_plan.input1_dim()],
                            &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                            &cache.edge_mlp
                                [edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                            &tp_scatter_bars_tp[edge.src * tp_plan.output_dim()
                                ..(edge.src + 1) * tp_plan.output_dim()],
                            &mut dst_grad_x,
                            edge_attr_bar,
                            edge_mlp_bar,
                        );
                    }
                    let target = &mut normed_linear1_bars[dst * input1_dim..(dst + 1) * input1_dim];
                    for (target, value) in target.iter_mut().zip(&dst_grad_x) {
                        *target += *value;
                    }
                }
                input_major_linear_vjp_input(
                    &edge_mlp_bars,
                    edge_count,
                    128,
                    tp_weight_numel,
                    edge_mlp_2_weight,
                    layer_plan.edge_mlp_alpha1,
                )?
            };
            #[cfg(not(feature = "parallel"))]
            let hidden_silu_bars = {
                let mut edge_mlp_bars = vec![0.0; cache.edge_mlp.len()];
                let mut dst_grad_x = vec![0.0; input1_dim];
                for dst in 0..num_nodes {
                    let edge_range = edge_plan.dst_offsets[dst]..edge_plan.dst_offsets[dst + 1];
                    if edge_range.is_empty() {
                        continue;
                    }
                    dst_grad_x.fill(0.0);
                    for &edge_index in &edge_plan.edges_by_dst[edge_range] {
                        let edge = &edges[edge_index];
                        let edge_attr_bar =
                            &mut edge_attr_bars[edge_index * sh_dim..(edge_index + 1) * sh_dim];
                        let edge_mlp_bar = &mut edge_mlp_bars
                            [edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel];
                        tensor_product_vjp_accumulate_oam_tp_layout(
                            tp_plan,
                            &cache.normed_linear1[edge.dst * tp_plan.input1_dim()
                                ..(edge.dst + 1) * tp_plan.input1_dim()],
                            &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                            &cache.edge_mlp
                                [edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                            &tp_scatter_bars_tp[edge.src * tp_plan.output_dim()
                                ..(edge.src + 1) * tp_plan.output_dim()],
                            &mut dst_grad_x,
                            edge_attr_bar,
                            edge_mlp_bar,
                        );
                    }
                    let target = &mut normed_linear1_bars[dst * input1_dim..(dst + 1) * input1_dim];
                    for (target, value) in target.iter_mut().zip(&dst_grad_x) {
                        *target += *value;
                    }
                }
                input_major_linear_vjp_input(
                    &edge_mlp_bars,
                    edge_count,
                    128,
                    tp_weight_numel,
                    edge_mlp_2_weight,
                    layer_plan.edge_mlp_alpha1,
                )?
            };

            let mut hidden_pre_bars = hidden_silu_bars;
            for (bar, derivative) in hidden_pre_bars
                .iter_mut()
                .zip(&cache.hidden_silu_derivative)
            {
                *bar *= *derivative;
            }
            let edge_mlp_0_weight = self.tensor(
                &format!("{tensor_prefix}.conv.edge_mlp.mlp.0.weight"),
                &[8, 128],
            )?;
            input_major_linear_vjp_input_accumulate(
                &hidden_pre_bars,
                edge_count,
                8,
                128,
                edge_mlp_0_weight,
                layer_plan.edge_mlp_alpha0,
                &mut factored_embedding_bars,
            )?;

            let linear1_spec = object_field(layer_spec, "linear1")?;
            let linear1_weight = self.tensor_by_numel(
                &format!("{tensor_prefix}.conv.linear_1.weight"),
                usize_field(linear1_spec, "weight_numel")?,
            )?;
            let scatter_norm = layer_plan.scatter_norm;
            for value in &mut normed_linear1_bars {
                *value *= scatter_norm;
            }
            let linear1_input_bars = compact_linear_vjp_input_mapped(
                &normed_linear1_bars,
                num_nodes,
                linear1_spec,
                linear1_weight,
                Some(&tp_plan.input1_original_to_tp),
                None,
            )?;
            for (target, value) in previous_feature_bars.iter_mut().zip(linear1_input_bars) {
                *target += value;
            }
            feature_bars = previous_feature_bars;
        }

        let mut forces = vec![[0.0; 3]; num_nodes];
        let mut displacement_gradient = [[0.0; 3]; 3];
        for (edge_index, edge) in edges.iter().enumerate() {
            let mut bar_r = [0.0; 3];
            self.sh_plan.vjp_edge_vector(
                edge.vector,
                &edge_attr_bars[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                &mut bar_r,
            )?;
            edge_distance_bars[edge_index] += factored_embedding_distance_vjp(
                edge.distance,
                rmax_recip,
                bessel_weights,
                cutoff_p,
                edge_embedding_factor,
                &factored_embedding_bars
                    [edge_index * bessel_weights.len()..(edge_index + 1) * bessel_weights.len()],
            );
            distance_vjp_accumulate(
                edge.vector,
                edge_distance_bars[edge_index],
                DISTANCE_EPS,
                &mut bar_r,
            )?;
            scatter_edge_vector_vjp(edge.src, edge.dst, bar_r, &mut forces)?;
            if compute_displacement_gradient {
                accumulate_edge_displacement_gradient(
                    edge.vector,
                    bar_r,
                    &mut displacement_gradient,
                );
            }
        }
        Ok(NequipOamPrefixVjp {
            forces,
            displacement_gradient,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn reverse_forces_f32(
        &self,
        edges: &[crate::Edge],
        edge_plan: &OamEdgePlan,
        edge_attrs: &[f32],
        rmax_recip: f32,
        bessel_weights: &[f32],
        cutoff_p: f32,
        edge_embedding_factor: f32,
        active_type_rows: &ActiveTypeRows,
        sc_contractions_by_layer: &ScalarAttrContractionsByLayerF32,
        layer_inputs: &[Vec<f32>],
        layer_caches: &[NequipOamLayerCacheF32],
        mut feature_bars: Vec<f32>,
        mut edge_distance_bars: Vec<f64>,
        num_nodes: usize,
    ) -> Result<NequipOamPrefixVjp, NequipOamPrefixError> {
        let prefix = object_field(&self.spec, "prefix")?;
        let layers = array_field(prefix, "layers")?;
        let sh_dim = self.sh_plan.output_dim();
        let edge_count = edges.len();
        let mut edge_attr_bars = vec![0.0; edge_count * sh_dim];
        let mut factored_embedding_bars = vec![0.0; edge_count * bessel_weights.len()];

        for layer_index in (0..layers.len()).rev() {
            let layer_spec = &layers[layer_index];
            let layer_plan = &self.layer_plans[layer_index];
            let cache = &layer_caches[layer_index];
            let layer_input = &layer_inputs[layer_index];
            let tensor_prefix = format!("sole_model.model.func.layer{layer_index}_convnet");

            let gate_plan = &self.gate_plans[layer_index];
            let gate_cache = cache
                .gate
                .as_ref()
                .ok_or_else(|| invalid_model("missing f32 prepared gate cache"))?;
            let gate_input_bars = gate_vjp_input_prepared_f32(
                &cache.gate_input,
                num_nodes,
                gate_plan,
                gate_cache,
                &feature_bars,
            )?;

            let tp_plan = &self.tp_plans[layer_index];
            let linear2_spec = object_field(layer_spec, "linear2")?;
            let linear2_weight = self.tensor_by_numel_f32_cow(
                &format!("{tensor_prefix}.conv.linear_2.weight"),
                usize_field(linear2_spec, "weight_numel")?,
            )?;
            let tp_scatter_bars_tp = compact_linear_vjp_input_mapped_f32(
                &gate_input_bars,
                num_nodes,
                linear2_spec,
                &linear2_weight,
                None,
                Some(&tp_plan.output_original_to_tp),
            )?;

            let mut previous_feature_bars = vec![0.0; layer_input.len()];
            if layer_spec.get("self_connection").is_some() {
                let sc_contractions =
                    sc_contractions_by_layer[layer_index]
                        .as_ref()
                        .ok_or_else(|| {
                            invalid_model("missing scalar-attribute f32 contraction cache")
                        })?;
                scalar_attr_tensor_product_vjp_input_accumulate_auto_f32(
                    &gate_input_bars,
                    layer_input,
                    active_type_rows,
                    num_nodes,
                    sc_contractions,
                    &mut previous_feature_bars,
                )?;
            }

            let tp_weight_numel = tp_plan.weight_numel;
            let mut normed_linear1_bars = vec![0.0; cache.normed_linear1.len()];
            let input1_dim = tp_plan.input1_dim();
            let edge_mlp_2_weight = self.tensor_f32_cow(
                &format!("{tensor_prefix}.conv.edge_mlp.mlp.2.weight"),
                &[128, tp_weight_numel],
            )?;
            #[cfg(feature = "parallel")]
            let hidden_silu_bars = if rayon::current_num_threads() > 1
                && edge_count >= OAM_TP_VJP_PARALLEL_MIN_EDGES
            {
                let mut fused_hidden_silu_bars = vec![0.0; edge_count * 128];
                tensor_product_vjp_scatter_oam_parallel_direct_fused_edge_mlp2_tp_layout_f32(
                    tp_plan,
                    edges,
                    edge_plan,
                    edge_attrs,
                    sh_dim,
                    &cache.normed_linear1,
                    &cache.edge_mlp,
                    tp_weight_numel,
                    &tp_scatter_bars_tp,
                    &edge_mlp_2_weight,
                    layer_plan.edge_mlp_alpha1_f32,
                    &mut normed_linear1_bars,
                    &mut edge_attr_bars,
                    &mut fused_hidden_silu_bars,
                    num_nodes,
                )?;
                fused_hidden_silu_bars
            } else {
                let mut edge_mlp_bars = vec![0.0; cache.edge_mlp.len()];
                let mut dst_grad_x = vec![0.0; input1_dim];
                for dst in 0..num_nodes {
                    let edge_range = edge_plan.dst_offsets[dst]..edge_plan.dst_offsets[dst + 1];
                    if edge_range.is_empty() {
                        continue;
                    }
                    dst_grad_x.fill(0.0);
                    for &edge_index in &edge_plan.edges_by_dst[edge_range] {
                        let edge = &edges[edge_index];
                        let edge_attr_bar =
                            &mut edge_attr_bars[edge_index * sh_dim..(edge_index + 1) * sh_dim];
                        let edge_mlp_bar = &mut edge_mlp_bars
                            [edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel];
                        tensor_product_vjp_accumulate_oam_tp_layout_f32(
                            tp_plan,
                            &cache.normed_linear1[edge.dst * tp_plan.input1_dim()
                                ..(edge.dst + 1) * tp_plan.input1_dim()],
                            &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                            &cache.edge_mlp
                                [edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                            &tp_scatter_bars_tp[edge.src * tp_plan.output_dim()
                                ..(edge.src + 1) * tp_plan.output_dim()],
                            &mut dst_grad_x,
                            edge_attr_bar,
                            edge_mlp_bar,
                        );
                    }
                    let target = &mut normed_linear1_bars[dst * input1_dim..(dst + 1) * input1_dim];
                    for (target, value) in target.iter_mut().zip(&dst_grad_x) {
                        *target += *value;
                    }
                }
                input_major_linear_vjp_input_f32(
                    &edge_mlp_bars,
                    edge_count,
                    128,
                    tp_weight_numel,
                    &edge_mlp_2_weight,
                    layer_plan.edge_mlp_alpha1_f32,
                )?
            };
            #[cfg(not(feature = "parallel"))]
            let hidden_silu_bars = {
                let mut edge_mlp_bars = vec![0.0; cache.edge_mlp.len()];
                let mut dst_grad_x = vec![0.0; input1_dim];
                for dst in 0..num_nodes {
                    let edge_range = edge_plan.dst_offsets[dst]..edge_plan.dst_offsets[dst + 1];
                    if edge_range.is_empty() {
                        continue;
                    }
                    dst_grad_x.fill(0.0);
                    for &edge_index in &edge_plan.edges_by_dst[edge_range] {
                        let edge = &edges[edge_index];
                        let edge_attr_bar =
                            &mut edge_attr_bars[edge_index * sh_dim..(edge_index + 1) * sh_dim];
                        let edge_mlp_bar = &mut edge_mlp_bars
                            [edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel];
                        tensor_product_vjp_accumulate_oam_tp_layout_f32(
                            tp_plan,
                            &cache.normed_linear1[edge.dst * tp_plan.input1_dim()
                                ..(edge.dst + 1) * tp_plan.input1_dim()],
                            &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                            &cache.edge_mlp
                                [edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                            &tp_scatter_bars_tp[edge.src * tp_plan.output_dim()
                                ..(edge.src + 1) * tp_plan.output_dim()],
                            &mut dst_grad_x,
                            edge_attr_bar,
                            edge_mlp_bar,
                        );
                    }
                    let target = &mut normed_linear1_bars[dst * input1_dim..(dst + 1) * input1_dim];
                    for (target, value) in target.iter_mut().zip(&dst_grad_x) {
                        *target += *value;
                    }
                }
                input_major_linear_vjp_input_f32(
                    &edge_mlp_bars,
                    edge_count,
                    128,
                    tp_weight_numel,
                    &edge_mlp_2_weight,
                    layer_plan.edge_mlp_alpha1_f32,
                )?
            };

            let mut hidden_pre_bars = hidden_silu_bars;
            for (bar, derivative) in hidden_pre_bars
                .iter_mut()
                .zip(&cache.hidden_silu_derivative)
            {
                *bar *= *derivative;
            }
            let edge_mlp_0_weight = self.tensor_f32_cow(
                &format!("{tensor_prefix}.conv.edge_mlp.mlp.0.weight"),
                &[8, 128],
            )?;
            input_major_linear_vjp_input_accumulate_f32(
                &hidden_pre_bars,
                edge_count,
                8,
                128,
                &edge_mlp_0_weight,
                layer_plan.edge_mlp_alpha0_f32,
                &mut factored_embedding_bars,
            )?;

            let linear1_spec = object_field(layer_spec, "linear1")?;
            let linear1_weight = self.tensor_by_numel_f32_cow(
                &format!("{tensor_prefix}.conv.linear_1.weight"),
                usize_field(linear1_spec, "weight_numel")?,
            )?;
            let scatter_norm = layer_plan.scatter_norm_f32;
            for value in &mut normed_linear1_bars {
                *value *= scatter_norm;
            }
            let linear1_input_bars = compact_linear_vjp_input_mapped_f32(
                &normed_linear1_bars,
                num_nodes,
                linear1_spec,
                &linear1_weight,
                Some(&tp_plan.input1_original_to_tp),
                None,
            )?;
            for (target, value) in previous_feature_bars.iter_mut().zip(linear1_input_bars) {
                *target += value;
            }
            feature_bars = previous_feature_bars;
        }

        let mut forces = vec![[0.0; 3]; num_nodes];
        let mut edge_attr_bar_f64 = vec![0.0; sh_dim];
        for (edge_index, edge) in edges.iter().enumerate() {
            let mut bar_r = [0.0; 3];
            for (target, &value) in edge_attr_bar_f64
                .iter_mut()
                .zip(&edge_attr_bars[edge_index * sh_dim..(edge_index + 1) * sh_dim])
            {
                *target = f64::from(value);
            }
            self.sh_plan
                .vjp_edge_vector(edge.vector, &edge_attr_bar_f64, &mut bar_r)?;
            edge_distance_bars[edge_index] += factored_embedding_distance_vjp_f32(
                edge.distance,
                rmax_recip,
                bessel_weights,
                cutoff_p,
                edge_embedding_factor,
                &factored_embedding_bars
                    [edge_index * bessel_weights.len()..(edge_index + 1) * bessel_weights.len()],
            );
            distance_vjp_accumulate(
                edge.vector,
                edge_distance_bars[edge_index],
                DISTANCE_EPS,
                &mut bar_r,
            )?;
            scatter_edge_vector_vjp(edge.src, edge.dst, bar_r, &mut forces)?;
        }
        Ok(NequipOamPrefixVjp {
            forces,
            displacement_gradient: [[0.0; 3]; 3],
        })
    }

    fn type_ids(&self, structure: &Structure) -> Result<Vec<usize>, NequipOamPrefixError> {
        structure
            .species
            .iter()
            .map(|species| {
                self.type_names
                    .iter()
                    .position(|candidate| candidate == species)
                    .ok_or_else(|| NequipOamPrefixError::UnknownSpecies(species.clone()))
            })
            .collect()
    }

    fn sc_contractions_for_active_types(
        &self,
        active_type_rows: &ActiveTypeRows,
    ) -> Result<Arc<ScalarAttrContractionsByLayer>, NequipOamPrefixError> {
        let key = active_type_rows.active_rows.clone();
        if let Some(cached) = self
            .sc_contraction_cache
            .lock()
            .map_err(|_| invalid_model("scalar-attribute contraction cache is poisoned"))?
            .get(&key)
            .cloned()
        {
            return Ok(cached);
        }

        let computed = Arc::new(self.build_sc_contractions(active_type_rows)?);
        let mut cache = self
            .sc_contraction_cache
            .lock()
            .map_err(|_| invalid_model("scalar-attribute contraction cache is poisoned"))?;
        Ok(cache
            .entry(key)
            .or_insert_with(|| Arc::clone(&computed))
            .clone())
    }

    fn build_sc_contractions(
        &self,
        active_type_rows: &ActiveTypeRows,
    ) -> Result<ScalarAttrContractionsByLayer, NequipOamPrefixError> {
        let prefix = object_field(&self.spec, "prefix")?;
        let layers = array_field(prefix, "layers")?;
        let type_embedding = self.tensor(
            "sole_model.model.func.type_embed.embed_module.weight",
            &[self.type_names.len(), 48],
        )?;
        // The contraction cache is keyed only by active type rows. That is
        // valid for the pinned OAM-L contract because scalar attributes are the
        // fixed per-type embedding table, not per-node learned features.
        debug_assert_eq!(type_embedding.len(), self.type_names.len() * 48);
        let mut contractions_by_layer = Vec::with_capacity(layers.len());
        for (layer_index, layer_spec) in layers.iter().enumerate() {
            let Some(self_connection) = layer_spec.get("self_connection") else {
                contractions_by_layer.push(None);
                continue;
            };
            let tensor_prefix = format!("sole_model.model.func.layer{layer_index}_convnet");
            let sc_weight = self.tensor_by_numel(
                &format!("{tensor_prefix}.conv.sc.weight"),
                usize_field(self_connection, "weight_numel")?,
            )?;
            contractions_by_layer.push(Some(scalar_attr_tensor_product_contractions(
                type_embedding,
                active_type_rows,
                self.type_names.len(),
                self_connection,
                sc_weight,
            )?));
        }
        Ok(contractions_by_layer)
    }

    fn sc_contractions_for_active_types_f32(
        &self,
        active_type_rows: &ActiveTypeRows,
    ) -> Result<Arc<ScalarAttrContractionsByLayerF32>, NequipOamPrefixError> {
        let key = active_type_rows.active_rows.clone();
        if let Some(cached) = self
            .sc_contraction_f32_cache
            .lock()
            .map_err(|_| invalid_model("scalar-attribute f32 contraction cache is poisoned"))?
            .get(&key)
            .cloned()
        {
            return Ok(cached);
        }

        let computed = Arc::new(self.build_sc_contractions_f32(active_type_rows)?);
        let mut cache = self
            .sc_contraction_f32_cache
            .lock()
            .map_err(|_| invalid_model("scalar-attribute f32 contraction cache is poisoned"))?;
        Ok(cache
            .entry(key)
            .or_insert_with(|| Arc::clone(&computed))
            .clone())
    }

    fn build_sc_contractions_f32(
        &self,
        active_type_rows: &ActiveTypeRows,
    ) -> Result<ScalarAttrContractionsByLayerF32, NequipOamPrefixError> {
        let prefix = object_field(&self.spec, "prefix")?;
        let layers = array_field(prefix, "layers")?;
        let type_embedding = self.tensor_f32_cow(
            "sole_model.model.func.type_embed.embed_module.weight",
            &[self.type_names.len(), 48],
        )?;
        debug_assert_eq!(type_embedding.len(), self.type_names.len() * 48);
        let mut contractions_by_layer = Vec::with_capacity(layers.len());
        for (layer_index, layer_spec) in layers.iter().enumerate() {
            let Some(self_connection) = layer_spec.get("self_connection") else {
                contractions_by_layer.push(None);
                continue;
            };
            let tensor_prefix = format!("sole_model.model.func.layer{layer_index}_convnet");
            let sc_weight = self.tensor_by_numel_f32_cow(
                &format!("{tensor_prefix}.conv.sc.weight"),
                usize_field(self_connection, "weight_numel")?,
            )?;
            contractions_by_layer.push(Some(scalar_attr_tensor_product_contractions_f32(
                &type_embedding,
                active_type_rows,
                self.type_names.len(),
                self_connection,
                &sc_weight,
            )?));
        }
        Ok(contractions_by_layer)
    }

    fn tensor(&self, name: &str, expected_shape: &[usize]) -> Result<&[f64], NequipOamPrefixError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| NequipOamPrefixError::MissingStateTensor(name.to_owned()))?;
        if tensor.shape.as_slice() != expected_shape {
            return Err(NequipOamPrefixError::StateTensorShape {
                name: name.to_owned(),
                actual: tensor.shape.clone(),
                expected: expected_shape.to_vec(),
            });
        }
        Ok(&tensor.values)
    }

    fn tensor_by_numel(
        &self,
        name: &str,
        expected_numel: usize,
    ) -> Result<&[f64], NequipOamPrefixError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| NequipOamPrefixError::MissingStateTensor(name.to_owned()))?;
        if tensor.values.len() != expected_numel {
            return Err(NequipOamPrefixError::StateTensorShape {
                name: name.to_owned(),
                actual: tensor.shape.clone(),
                expected: vec![expected_numel],
            });
        }
        Ok(&tensor.values)
    }

    fn tensor_f32_cow(
        &self,
        name: &str,
        expected_shape: &[usize],
    ) -> Result<Cow<'_, [f32]>, NequipOamPrefixError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| NequipOamPrefixError::MissingStateTensor(name.to_owned()))?;
        if tensor.shape.as_slice() != expected_shape {
            return Err(NequipOamPrefixError::StateTensorShape {
                name: name.to_owned(),
                actual: tensor.shape.clone(),
                expected: expected_shape.to_vec(),
            });
        }
        Ok(if let Some(values) = tensor.values_f32.as_deref() {
            Cow::Borrowed(values)
        } else {
            Cow::Owned(tensor.values.iter().map(|&value| value as f32).collect())
        })
    }

    fn tensor_by_numel_f32_cow(
        &self,
        name: &str,
        expected_numel: usize,
    ) -> Result<Cow<'_, [f32]>, NequipOamPrefixError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| NequipOamPrefixError::MissingStateTensor(name.to_owned()))?;
        if tensor.values.len() != expected_numel {
            return Err(NequipOamPrefixError::StateTensorShape {
                name: name.to_owned(),
                actual: tensor.shape.clone(),
                expected: vec![expected_numel],
            });
        }
        Ok(if let Some(values) = tensor.values_f32.as_deref() {
            Cow::Borrowed(values)
        } else {
            Cow::Owned(tensor.values.iter().map(|&value| value as f32).collect())
        })
    }

    fn tensor_scalar_f32(&self, name: &str) -> Result<f32, NequipOamPrefixError> {
        let tensor = self.tensor_f32_cow(name, &[])?;
        tensor
            .first()
            .copied()
            .ok_or_else(|| invalid_model(format!("scalar f32 tensor {name:?} is empty")))
    }

    #[must_use]
    pub fn native_f32_state_tensor_count(&self) -> usize {
        self.tensors
            .values()
            .filter(|tensor| tensor.values_f32.is_some())
            .count()
    }

    #[must_use]
    pub fn has_native_f32_state_tensors(&self) -> bool {
        self.native_f32_state_tensor_count() > 0
    }

    #[must_use]
    pub fn state_tensor_values_f32(&self, name: &str) -> Option<&[f32]> {
        self.tensors
            .get(name)
            .and_then(|tensor| tensor.values_f32.as_deref())
    }

    fn tensor_scalar(&self, name: &str) -> Result<f64, NequipOamPrefixError> {
        let tensor = self.tensor(name, &[])?;
        tensor
            .first()
            .copied()
            .ok_or_else(|| invalid_model(format!("scalar tensor {name:?} is empty")))
    }
}

impl NequipOamPrefixVjp {
    fn into_output_tensors(
        self,
        structure: &Structure,
        options: &PredictionOptions,
    ) -> Result<NequipOamPrefixOutputTensors, NequipOamPrefixError> {
        let displacement_gradient = symmetric_part(self.displacement_gradient);
        let stress = if options.compute_stress {
            let cell = structure
                .cell
                .as_ref()
                .ok_or(NequipOamPrefixError::StressVirialRequiresFullPbc)?;
            Some(stress_from_displacement_gradient(
                displacement_gradient,
                cell,
            )?)
        } else {
            None
        };
        let virial = if options.compute_virial {
            Some(virial_from_displacement_gradient(displacement_gradient))
        } else {
            None
        };
        Ok((self.forces, stress, virial))
    }
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

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        target_arch = "x86_64"
    )
))]
fn compact_linear_allows_packed_blas() -> bool {
    true
}

#[allow(dead_code)]
fn compact_linear_forward(
    input: &[f64],
    row_count: usize,
    spec: &Value,
    weight: &[f64],
) -> Result<Vec<f64>, NequipOamPrefixError> {
    let input_dim = usize_field(spec, "input_dim")?;
    let output_dim = usize_field(spec, "output_dim")?;
    if input.len() != row_count * input_dim {
        return Err(invalid_model("compact linear input shape mismatch"));
    }
    if weight.len() != usize_field(spec, "weight_numel")? {
        return Err(invalid_model("compact linear weight shape mismatch"));
    }
    let mut out = vec![0.0; row_count * output_dim];
    for block in array_field(spec, "blocks")? {
        let mul_in = usize_field(block, "mul_in")?;
        let mul_out = usize_field(block, "mul_out")?;
        let ir_dim = usize_field(block, "ir_dim")?;
        let input_offset = usize_field(block, "input_offset")?;
        let output_offset = usize_field(block, "output_offset")?;
        let weight_offset = usize_field(block, "weight_offset")?;
        let path_weight = f64_field(block, "path_weight")?;
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if row_count >= 16 && ir_dim == 1 && mul_in >= 48 && mul_out >= 48 {
            let weight_end = weight_offset + mul_in * mul_out;
            portable_blas::dense_forward_strided_block(
                input,
                row_count,
                input_dim,
                input_offset,
                mul_in,
                &weight[weight_offset..weight_end],
                mul_out,
                &mut out,
                output_dim,
                output_offset,
                path_weight,
            );
            continue;
        }
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if compact_linear_allows_packed_blas()
            && row_count >= 16
            && ir_dim > 1
            && mul_in >= 32
            && mul_out >= 32
        {
            let weight_end = weight_offset + mul_in * mul_out;
            let block_weight = &weight[weight_offset..weight_end];
            let mut packed_input = vec![0.0; row_count * mul_in];
            let mut packed_output = vec![0.0; row_count * mul_out];
            for m in 0..ir_dim {
                for row in 0..row_count {
                    let input_row = row * input_dim + input_offset;
                    let packed_row = row * mul_in;
                    for in_mul in 0..mul_in {
                        packed_input[packed_row + in_mul] = input[input_row + in_mul * ir_dim + m];
                    }
                }
                portable_blas::dense_forward(
                    &packed_input,
                    row_count,
                    mul_in,
                    block_weight,
                    &mut packed_output,
                    path_weight,
                );
                for row in 0..row_count {
                    let output_row = row * output_dim + output_offset;
                    let packed_row = row * mul_out;
                    for out_mul in 0..mul_out {
                        out[output_row + out_mul * ir_dim + m] +=
                            packed_output[packed_row + out_mul];
                    }
                }
            }
            continue;
        }
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && row_count >= OAM_COMPACT_LINEAR_PARALLEL_MIN_ROWS {
            out.par_chunks_mut(output_dim)
                .enumerate()
                .for_each(|(row, out_row)| {
                    let input_row = row * input_dim + input_offset;
                    for in_mul in 0..mul_in {
                        let weight_row = weight_offset + in_mul * mul_out;
                        for m in 0..ir_dim {
                            let x = path_weight * input[input_row + in_mul * ir_dim + m];
                            for out_mul in 0..mul_out {
                                out_row[output_offset + out_mul * ir_dim + m] +=
                                    x * weight[weight_row + out_mul];
                            }
                        }
                    }
                });
            continue;
        }
        for row in 0..row_count {
            let input_row = row * input_dim + input_offset;
            let output_row = row * output_dim + output_offset;
            for in_mul in 0..mul_in {
                let weight_row = weight_offset + in_mul * mul_out;
                for m in 0..ir_dim {
                    let x = path_weight * input[input_row + in_mul * ir_dim + m];
                    for out_mul in 0..mul_out {
                        out[output_row + out_mul * ir_dim + m] += x * weight[weight_row + out_mul];
                    }
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
#[allow(dead_code)]
fn compact_linear_forward_plan(
    input: &[f64],
    row_count: usize,
    plan: &OamCompactLinearPlan,
    weight: &[f64],
) -> Result<Vec<f64>, NequipOamPrefixError> {
    let input_dim = plan.input_dim;
    let output_dim = plan.output_dim;
    if input.len() != row_count * input_dim {
        return Err(invalid_model("compact linear input shape mismatch"));
    }
    if weight.len() != plan.weight_numel {
        return Err(invalid_model("compact linear weight shape mismatch"));
    }
    let mut out = vec![0.0; row_count * output_dim];
    for block in &plan.blocks {
        let mul_in = block.mul_in;
        let mul_out = block.mul_out;
        let ir_dim = block.ir_dim;
        let input_offset = block.input_offset;
        let output_offset = block.output_offset;
        let weight_offset = block.weight_offset;
        let path_weight = block.path_weight;
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if row_count >= 16 && ir_dim == 1 && mul_in >= 48 && mul_out >= 48 {
            let weight_end = weight_offset + mul_in * mul_out;
            portable_blas::dense_forward_strided_block(
                input,
                row_count,
                input_dim,
                input_offset,
                mul_in,
                &weight[weight_offset..weight_end],
                mul_out,
                &mut out,
                output_dim,
                output_offset,
                path_weight,
            );
            continue;
        }
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if compact_linear_allows_packed_blas()
            && row_count >= 16
            && ir_dim > 1
            && mul_in >= 32
            && mul_out >= 32
        {
            let weight_end = weight_offset + mul_in * mul_out;
            let block_weight = &weight[weight_offset..weight_end];
            let mut packed_input = vec![0.0; row_count * mul_in];
            let mut packed_output = vec![0.0; row_count * mul_out];
            for m in 0..ir_dim {
                for row in 0..row_count {
                    let input_row = row * input_dim + input_offset;
                    let packed_row = row * mul_in;
                    for in_mul in 0..mul_in {
                        packed_input[packed_row + in_mul] = input[input_row + in_mul * ir_dim + m];
                    }
                }
                portable_blas::dense_forward(
                    &packed_input,
                    row_count,
                    mul_in,
                    block_weight,
                    &mut packed_output,
                    path_weight,
                );
                for row in 0..row_count {
                    let output_row = row * output_dim + output_offset;
                    let packed_row = row * mul_out;
                    for out_mul in 0..mul_out {
                        out[output_row + out_mul * ir_dim + m] +=
                            packed_output[packed_row + out_mul];
                    }
                }
            }
            continue;
        }
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && row_count >= OAM_COMPACT_LINEAR_PARALLEL_MIN_ROWS {
            out.par_chunks_mut(output_dim)
                .enumerate()
                .for_each(|(row, out_row)| {
                    let input_row = row * input_dim + input_offset;
                    for in_mul in 0..mul_in {
                        let weight_row = weight_offset + in_mul * mul_out;
                        for m in 0..ir_dim {
                            let x = path_weight * input[input_row + in_mul * ir_dim + m];
                            for out_mul in 0..mul_out {
                                out_row[output_offset + out_mul * ir_dim + m] +=
                                    x * weight[weight_row + out_mul];
                            }
                        }
                    }
                });
            continue;
        }
        for row in 0..row_count {
            let input_row = row * input_dim + input_offset;
            let output_row = row * output_dim + output_offset;
            for in_mul in 0..mul_in {
                let weight_row = weight_offset + in_mul * mul_out;
                for m in 0..ir_dim {
                    let x = path_weight * input[input_row + in_mul * ir_dim + m];
                    for out_mul in 0..mul_out {
                        out[output_row + out_mul * ir_dim + m] += x * weight[weight_row + out_mul];
                    }
                }
            }
        }
    }
    Ok(out)
}

#[allow(dead_code)]
fn compact_linear_forward_f32(
    input: &[f32],
    row_count: usize,
    spec: &Value,
    weight: &[f32],
) -> Result<Vec<f32>, NequipOamPrefixError> {
    let input_dim = usize_field(spec, "input_dim")?;
    let output_dim = usize_field(spec, "output_dim")?;
    if input.len() != row_count * input_dim {
        return Err(invalid_model("compact f32 linear input shape mismatch"));
    }
    if weight.len() != usize_field(spec, "weight_numel")? {
        return Err(invalid_model("compact f32 linear weight shape mismatch"));
    }
    let mut out = vec![0.0; row_count * output_dim];
    for block in array_field(spec, "blocks")? {
        let mul_in = usize_field(block, "mul_in")?;
        let mul_out = usize_field(block, "mul_out")?;
        let ir_dim = usize_field(block, "ir_dim")?;
        let input_offset = usize_field(block, "input_offset")?;
        let output_offset = usize_field(block, "output_offset")?;
        let weight_offset = usize_field(block, "weight_offset")?;
        let path_weight = f64_field(block, "path_weight")? as f32;
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if row_count >= 16 && ir_dim == 1 && mul_in >= 48 && mul_out >= 48 {
            let weight_end = weight_offset + mul_in * mul_out;
            crate::blas_backend::dense_forward_strided_f32(
                &input[input_offset..],
                row_count,
                mul_in,
                input_dim,
                &weight[weight_offset..weight_end],
                &mut out[output_offset..],
                mul_out,
                output_dim,
                path_weight,
                1.0,
            );
            continue;
        }
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if compact_linear_allows_packed_blas()
            && row_count >= 16
            && ir_dim > 1
            && mul_in >= 32
            && mul_out >= 32
        {
            let weight_end = weight_offset + mul_in * mul_out;
            let block_weight = &weight[weight_offset..weight_end];
            let mut packed_input = vec![0.0; row_count * mul_in];
            let mut packed_output = vec![0.0; row_count * mul_out];
            for m in 0..ir_dim {
                for row in 0..row_count {
                    let input_row = row * input_dim + input_offset;
                    let packed_row = row * mul_in;
                    for in_mul in 0..mul_in {
                        packed_input[packed_row + in_mul] = input[input_row + in_mul * ir_dim + m];
                    }
                }
                crate::blas_backend::dense_forward_f32(
                    &packed_input,
                    row_count,
                    mul_in,
                    block_weight,
                    &mut packed_output,
                    path_weight,
                );
                for row in 0..row_count {
                    let output_row = row * output_dim + output_offset;
                    let packed_row = row * mul_out;
                    for out_mul in 0..mul_out {
                        out[output_row + out_mul * ir_dim + m] +=
                            packed_output[packed_row + out_mul];
                    }
                }
            }
            continue;
        }
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && row_count >= OAM_COMPACT_LINEAR_PARALLEL_MIN_ROWS {
            out.par_chunks_mut(output_dim)
                .enumerate()
                .for_each(|(row, out_row)| {
                    let input_row = row * input_dim + input_offset;
                    for in_mul in 0..mul_in {
                        let weight_row = weight_offset + in_mul * mul_out;
                        for m in 0..ir_dim {
                            let x = path_weight * input[input_row + in_mul * ir_dim + m];
                            for out_mul in 0..mul_out {
                                out_row[output_offset + out_mul * ir_dim + m] +=
                                    x * weight[weight_row + out_mul];
                            }
                        }
                    }
                });
            continue;
        }
        for row in 0..row_count {
            let input_row = row * input_dim + input_offset;
            let output_row = row * output_dim + output_offset;
            for in_mul in 0..mul_in {
                let weight_row = weight_offset + in_mul * mul_out;
                for m in 0..ir_dim {
                    let x = path_weight * input[input_row + in_mul * ir_dim + m];
                    for out_mul in 0..mul_out {
                        out[output_row + out_mul * ir_dim + m] += x * weight[weight_row + out_mul];
                    }
                }
            }
        }
    }
    Ok(out)
}

#[inline(always)]
fn mapped_or_identity(map: Option<&[usize]>, index: usize) -> usize {
    map.map_or(index, |map| map[index])
}

#[allow(dead_code)]
fn compact_linear_block_map_is_identity(
    map: Option<&[usize]>,
    offset: usize,
    mul: usize,
    ir_dim: usize,
) -> bool {
    let Some(map) = map else {
        return true;
    };
    for mul_index in 0..mul {
        for m in 0..ir_dim {
            let index = offset + mul_index * ir_dim + m;
            if map[index] != index {
                return false;
            }
        }
    }
    true
}

#[allow(dead_code)]
fn compact_linear_forward_mapped(
    input: &[f64],
    row_count: usize,
    spec: &Value,
    weight: &[f64],
    input_map: Option<&[usize]>,
    output_map: Option<&[usize]>,
) -> Result<Vec<f64>, NequipOamPrefixError> {
    let input_dim = usize_field(spec, "input_dim")?;
    let output_dim = usize_field(spec, "output_dim")?;
    if input.len() != row_count * input_dim {
        return Err(invalid_model("mapped compact linear input shape mismatch"));
    }
    if input_map.is_some_and(|map| map.len() != input_dim)
        || output_map.is_some_and(|map| map.len() != output_dim)
    {
        return Err(invalid_model("mapped compact linear layout map mismatch"));
    }
    if weight.len() != usize_field(spec, "weight_numel")? {
        return Err(invalid_model("mapped compact linear weight shape mismatch"));
    }
    let mut out = vec![0.0; row_count * output_dim];
    for block in array_field(spec, "blocks")? {
        let mul_in = usize_field(block, "mul_in")?;
        let mul_out = usize_field(block, "mul_out")?;
        let ir_dim = usize_field(block, "ir_dim")?;
        let input_offset = usize_field(block, "input_offset")?;
        let output_offset = usize_field(block, "output_offset")?;
        let weight_offset = usize_field(block, "weight_offset")?;
        let path_weight = f64_field(block, "path_weight")?;
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        let input_identity =
            compact_linear_block_map_is_identity(input_map, input_offset, mul_in, ir_dim);
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        let output_identity =
            compact_linear_block_map_is_identity(output_map, output_offset, mul_out, ir_dim);
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if input_identity
            && output_identity
            && row_count >= 16
            && ir_dim == 1
            && mul_in >= 48
            && mul_out >= 48
        {
            let weight_end = weight_offset + mul_in * mul_out;
            portable_blas::dense_forward_strided_block(
                input,
                row_count,
                input_dim,
                input_offset,
                mul_in,
                &weight[weight_offset..weight_end],
                mul_out,
                &mut out,
                output_dim,
                output_offset,
                path_weight,
            );
            continue;
        }
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if compact_linear_allows_packed_blas()
            && row_count >= 16
            && ir_dim > 1
            && mul_in >= 32
            && mul_out >= 32
        {
            let weight_end = weight_offset + mul_in * mul_out;
            let block_weight = &weight[weight_offset..weight_end];
            let mut packed_input = vec![0.0; row_count * mul_in];
            let mut packed_output = vec![0.0; row_count * mul_out];
            for m in 0..ir_dim {
                for row in 0..row_count {
                    let input_row = row * input_dim;
                    let packed_row = row * mul_in;
                    for in_mul in 0..mul_in {
                        let original_index = input_offset + in_mul * ir_dim + m;
                        packed_input[packed_row + in_mul] =
                            input[input_row + mapped_or_identity(input_map, original_index)];
                    }
                }
                portable_blas::dense_forward(
                    &packed_input,
                    row_count,
                    mul_in,
                    block_weight,
                    &mut packed_output,
                    path_weight,
                );
                for row in 0..row_count {
                    let output_row = row * output_dim;
                    let packed_row = row * mul_out;
                    for out_mul in 0..mul_out {
                        let original_index = output_offset + out_mul * ir_dim + m;
                        out[output_row + mapped_or_identity(output_map, original_index)] +=
                            packed_output[packed_row + out_mul];
                    }
                }
            }
            continue;
        }
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && row_count >= OAM_COMPACT_LINEAR_PARALLEL_MIN_ROWS {
            out.par_chunks_mut(output_dim)
                .enumerate()
                .for_each(|(row, out_row)| {
                    let input_row = row * input_dim;
                    for in_mul in 0..mul_in {
                        let weight_row = weight_offset + in_mul * mul_out;
                        for m in 0..ir_dim {
                            let original_input_index = input_offset + in_mul * ir_dim + m;
                            let x = path_weight
                                * input[input_row
                                    + mapped_or_identity(input_map, original_input_index)];
                            for out_mul in 0..mul_out {
                                let original_output_index = output_offset + out_mul * ir_dim + m;
                                out_row[mapped_or_identity(output_map, original_output_index)] +=
                                    x * weight[weight_row + out_mul];
                            }
                        }
                    }
                });
            continue;
        }
        for row in 0..row_count {
            let input_row = row * input_dim;
            let output_row = row * output_dim;
            for in_mul in 0..mul_in {
                let weight_row = weight_offset + in_mul * mul_out;
                for m in 0..ir_dim {
                    let original_input_index = input_offset + in_mul * ir_dim + m;
                    let x = path_weight
                        * input[input_row + mapped_or_identity(input_map, original_input_index)];
                    for out_mul in 0..mul_out {
                        let original_output_index = output_offset + out_mul * ir_dim + m;
                        out[output_row + mapped_or_identity(output_map, original_output_index)] +=
                            x * weight[weight_row + out_mul];
                    }
                }
            }
        }
    }
    Ok(out)
}

#[allow(dead_code)]
fn compact_linear_forward_mapped_f32(
    input: &[f32],
    row_count: usize,
    spec: &Value,
    weight: &[f32],
    input_map: Option<&[usize]>,
    output_map: Option<&[usize]>,
) -> Result<Vec<f32>, NequipOamPrefixError> {
    let input_dim = usize_field(spec, "input_dim")?;
    let output_dim = usize_field(spec, "output_dim")?;
    if input.len() != row_count * input_dim {
        return Err(invalid_model(
            "mapped compact f32 linear input shape mismatch",
        ));
    }
    if input_map.is_some_and(|map| map.len() != input_dim)
        || output_map.is_some_and(|map| map.len() != output_dim)
    {
        return Err(invalid_model(
            "mapped compact f32 linear layout map mismatch",
        ));
    }
    if weight.len() != usize_field(spec, "weight_numel")? {
        return Err(invalid_model(
            "mapped compact f32 linear weight shape mismatch",
        ));
    }
    let mut out = vec![0.0; row_count * output_dim];
    for block in array_field(spec, "blocks")? {
        let mul_in = usize_field(block, "mul_in")?;
        let mul_out = usize_field(block, "mul_out")?;
        let ir_dim = usize_field(block, "ir_dim")?;
        let input_offset = usize_field(block, "input_offset")?;
        let output_offset = usize_field(block, "output_offset")?;
        let weight_offset = usize_field(block, "weight_offset")?;
        let path_weight = f64_field(block, "path_weight")? as f32;
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        let input_identity =
            compact_linear_block_map_is_identity(input_map, input_offset, mul_in, ir_dim);
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        let output_identity =
            compact_linear_block_map_is_identity(output_map, output_offset, mul_out, ir_dim);
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if input_identity
            && output_identity
            && row_count >= 16
            && ir_dim == 1
            && mul_in >= 48
            && mul_out >= 48
        {
            let weight_end = weight_offset + mul_in * mul_out;
            crate::blas_backend::dense_forward_strided_f32(
                &input[input_offset..],
                row_count,
                mul_in,
                input_dim,
                &weight[weight_offset..weight_end],
                &mut out[output_offset..],
                mul_out,
                output_dim,
                path_weight,
                1.0,
            );
            continue;
        }
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if compact_linear_allows_packed_blas()
            && row_count >= 16
            && ir_dim > 1
            && mul_in >= 32
            && mul_out >= 32
        {
            let weight_end = weight_offset + mul_in * mul_out;
            let block_weight = &weight[weight_offset..weight_end];
            let mut packed_input = vec![0.0; row_count * mul_in];
            let mut packed_output = vec![0.0; row_count * mul_out];
            for m in 0..ir_dim {
                for row in 0..row_count {
                    let input_row = row * input_dim;
                    let packed_row = row * mul_in;
                    for in_mul in 0..mul_in {
                        let original_index = input_offset + in_mul * ir_dim + m;
                        packed_input[packed_row + in_mul] =
                            input[input_row + mapped_or_identity(input_map, original_index)];
                    }
                }
                crate::blas_backend::dense_forward_f32(
                    &packed_input,
                    row_count,
                    mul_in,
                    block_weight,
                    &mut packed_output,
                    path_weight,
                );
                for row in 0..row_count {
                    let output_row = row * output_dim;
                    let packed_row = row * mul_out;
                    for out_mul in 0..mul_out {
                        let original_index = output_offset + out_mul * ir_dim + m;
                        out[output_row + mapped_or_identity(output_map, original_index)] +=
                            packed_output[packed_row + out_mul];
                    }
                }
            }
            continue;
        }
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && row_count >= OAM_COMPACT_LINEAR_PARALLEL_MIN_ROWS {
            out.par_chunks_mut(output_dim)
                .enumerate()
                .for_each(|(row, out_row)| {
                    let input_row = row * input_dim;
                    for in_mul in 0..mul_in {
                        let weight_row = weight_offset + in_mul * mul_out;
                        for m in 0..ir_dim {
                            let original_input_index = input_offset + in_mul * ir_dim + m;
                            let x = path_weight
                                * input[input_row
                                    + mapped_or_identity(input_map, original_input_index)];
                            for out_mul in 0..mul_out {
                                let original_output_index = output_offset + out_mul * ir_dim + m;
                                out_row[mapped_or_identity(output_map, original_output_index)] +=
                                    x * weight[weight_row + out_mul];
                            }
                        }
                    }
                });
            continue;
        }
        for row in 0..row_count {
            let input_row = row * input_dim;
            let output_row = row * output_dim;
            for in_mul in 0..mul_in {
                let weight_row = weight_offset + in_mul * mul_out;
                for m in 0..ir_dim {
                    let original_input_index = input_offset + in_mul * ir_dim + m;
                    let x = path_weight
                        * input[input_row + mapped_or_identity(input_map, original_input_index)];
                    for out_mul in 0..mul_out {
                        let original_output_index = output_offset + out_mul * ir_dim + m;
                        out[output_row + mapped_or_identity(output_map, original_output_index)] +=
                            x * weight[weight_row + out_mul];
                    }
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
#[allow(dead_code)]
fn compact_linear_forward_plan_f32(
    input: &[f32],
    row_count: usize,
    plan: &OamCompactLinearPlan,
    weight: &[f32],
) -> Result<Vec<f32>, NequipOamPrefixError> {
    let input_dim = plan.input_dim;
    let output_dim = plan.output_dim;
    if input.len() != row_count * input_dim {
        return Err(invalid_model("compact f32 linear input shape mismatch"));
    }
    if weight.len() != plan.weight_numel {
        return Err(invalid_model("compact f32 linear weight shape mismatch"));
    }
    let mut out = vec![0.0; row_count * output_dim];
    for block in &plan.blocks {
        let mul_in = block.mul_in;
        let mul_out = block.mul_out;
        let ir_dim = block.ir_dim;
        let input_offset = block.input_offset;
        let output_offset = block.output_offset;
        let weight_offset = block.weight_offset;
        let path_weight = block.path_weight as f32;
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if row_count >= 16 && ir_dim == 1 && mul_in >= 48 && mul_out >= 48 {
            let weight_end = weight_offset + mul_in * mul_out;
            crate::blas_backend::dense_forward_strided_f32(
                &input[input_offset..],
                row_count,
                mul_in,
                input_dim,
                &weight[weight_offset..weight_end],
                &mut out[output_offset..],
                mul_out,
                output_dim,
                path_weight,
                1.0,
            );
            continue;
        }
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if compact_linear_allows_packed_blas()
            && row_count >= 16
            && ir_dim > 1
            && mul_in >= 32
            && mul_out >= 32
        {
            let weight_end = weight_offset + mul_in * mul_out;
            let block_weight = &weight[weight_offset..weight_end];
            let mut packed_input = vec![0.0; row_count * mul_in];
            let mut packed_output = vec![0.0; row_count * mul_out];
            for m in 0..ir_dim {
                for row in 0..row_count {
                    let input_row = row * input_dim + input_offset;
                    let packed_row = row * mul_in;
                    for in_mul in 0..mul_in {
                        packed_input[packed_row + in_mul] = input[input_row + in_mul * ir_dim + m];
                    }
                }
                crate::blas_backend::dense_forward_f32(
                    &packed_input,
                    row_count,
                    mul_in,
                    block_weight,
                    &mut packed_output,
                    path_weight,
                );
                for row in 0..row_count {
                    let output_row = row * output_dim + output_offset;
                    let packed_row = row * mul_out;
                    for out_mul in 0..mul_out {
                        out[output_row + out_mul * ir_dim + m] +=
                            packed_output[packed_row + out_mul];
                    }
                }
            }
            continue;
        }
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && row_count >= OAM_COMPACT_LINEAR_PARALLEL_MIN_ROWS {
            out.par_chunks_mut(output_dim)
                .enumerate()
                .for_each(|(row, out_row)| {
                    let input_row = row * input_dim + input_offset;
                    for in_mul in 0..mul_in {
                        let weight_row = weight_offset + in_mul * mul_out;
                        for m in 0..ir_dim {
                            let x = path_weight * input[input_row + in_mul * ir_dim + m];
                            for out_mul in 0..mul_out {
                                out_row[output_offset + out_mul * ir_dim + m] +=
                                    x * weight[weight_row + out_mul];
                            }
                        }
                    }
                });
            continue;
        }
        for row in 0..row_count {
            let input_row = row * input_dim + input_offset;
            let output_row = row * output_dim + output_offset;
            for in_mul in 0..mul_in {
                let weight_row = weight_offset + in_mul * mul_out;
                for m in 0..ir_dim {
                    let x = path_weight * input[input_row + in_mul * ir_dim + m];
                    for out_mul in 0..mul_out {
                        out[output_row + out_mul * ir_dim + m] += x * weight[weight_row + out_mul];
                    }
                }
            }
        }
    }
    Ok(out)
}

#[allow(dead_code)]
fn compact_linear_vjp_input(
    grad_out: &[f64],
    row_count: usize,
    spec: &Value,
    weight: &[f64],
) -> Result<Vec<f64>, NequipOamPrefixError> {
    let mut grad_input = Vec::new();
    compact_linear_vjp_input_into(grad_out, row_count, spec, weight, &mut grad_input)?;
    Ok(grad_input)
}

#[cfg(test)]
#[allow(dead_code)]
fn compact_linear_vjp_input_plan(
    grad_out: &[f64],
    row_count: usize,
    plan: &OamCompactLinearPlan,
    weight: &[f64],
) -> Result<Vec<f64>, NequipOamPrefixError> {
    let mut grad_input = Vec::new();
    compact_linear_vjp_input_into_plan(grad_out, row_count, plan, weight, &mut grad_input)?;
    Ok(grad_input)
}

#[allow(dead_code)]
fn compact_linear_vjp_input_into(
    grad_out: &[f64],
    row_count: usize,
    spec: &Value,
    weight: &[f64],
    grad_input: &mut Vec<f64>,
) -> Result<(), NequipOamPrefixError> {
    let input_dim = usize_field(spec, "input_dim")?;
    let output_dim = usize_field(spec, "output_dim")?;
    if grad_out.len() != row_count * output_dim {
        return Err(invalid_model("compact linear VJP output shape mismatch"));
    }
    if weight.len() != usize_field(spec, "weight_numel")? {
        return Err(invalid_model("compact linear VJP weight shape mismatch"));
    }
    grad_input.resize(row_count * input_dim, 0.0);
    grad_input.fill(0.0);
    for block in array_field(spec, "blocks")? {
        let mul_in = usize_field(block, "mul_in")?;
        let mul_out = usize_field(block, "mul_out")?;
        let ir_dim = usize_field(block, "ir_dim")?;
        let input_offset = usize_field(block, "input_offset")?;
        let output_offset = usize_field(block, "output_offset")?;
        let weight_offset = usize_field(block, "weight_offset")?;
        let path_weight = f64_field(block, "path_weight")?;
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if row_count >= 16 && ir_dim == 1 && mul_in >= 48 && mul_out >= 48 {
            let weight_end = weight_offset + mul_in * mul_out;
            portable_blas::dense_vjp_input_strided_block(
                row_count,
                input_dim,
                input_offset,
                mul_in,
                &weight[weight_offset..weight_end],
                grad_out,
                output_dim,
                output_offset,
                mul_out,
                grad_input,
                path_weight,
            );
            continue;
        }
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if compact_linear_allows_packed_blas()
            && row_count >= 16
            && ir_dim > 1
            && mul_in >= 32
            && mul_out >= 32
        {
            let weight_end = weight_offset + mul_in * mul_out;
            let block_weight = &weight[weight_offset..weight_end];
            let mut packed_grad_out = vec![0.0; row_count * mul_out];
            let mut packed_grad_input = vec![0.0; row_count * mul_in];
            for m in 0..ir_dim {
                for row in 0..row_count {
                    let output_row = row * output_dim + output_offset;
                    let packed_row = row * mul_out;
                    for out_mul in 0..mul_out {
                        packed_grad_out[packed_row + out_mul] =
                            grad_out[output_row + out_mul * ir_dim + m];
                    }
                }
                packed_grad_input.fill(0.0);
                portable_blas::dense_vjp_input(
                    row_count,
                    mul_in,
                    block_weight,
                    &packed_grad_out,
                    &mut packed_grad_input,
                    path_weight,
                );
                for row in 0..row_count {
                    let input_row = row * input_dim + input_offset;
                    let packed_row = row * mul_in;
                    for in_mul in 0..mul_in {
                        grad_input[input_row + in_mul * ir_dim + m] +=
                            packed_grad_input[packed_row + in_mul];
                    }
                }
            }
            continue;
        }
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && row_count >= OAM_COMPACT_LINEAR_PARALLEL_MIN_ROWS {
            grad_input
                .par_chunks_mut(input_dim)
                .enumerate()
                .for_each(|(row, grad_input_row)| {
                    let output_row = row * output_dim + output_offset;
                    for in_mul in 0..mul_in {
                        let weight_row = weight_offset + in_mul * mul_out;
                        for m in 0..ir_dim {
                            let mut value = 0.0;
                            for out_mul in 0..mul_out {
                                value += weight[weight_row + out_mul]
                                    * grad_out[output_row + out_mul * ir_dim + m];
                            }
                            grad_input_row[input_offset + in_mul * ir_dim + m] +=
                                path_weight * value;
                        }
                    }
                });
            continue;
        }
        for row in 0..row_count {
            let input_row = row * input_dim + input_offset;
            let output_row = row * output_dim + output_offset;
            for in_mul in 0..mul_in {
                let weight_row = weight_offset + in_mul * mul_out;
                for m in 0..ir_dim {
                    let mut value = 0.0;
                    for out_mul in 0..mul_out {
                        value += weight[weight_row + out_mul]
                            * grad_out[output_row + out_mul * ir_dim + m];
                    }
                    grad_input[input_row + in_mul * ir_dim + m] += path_weight * value;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(dead_code)]
fn compact_linear_vjp_input_into_plan(
    grad_out: &[f64],
    row_count: usize,
    plan: &OamCompactLinearPlan,
    weight: &[f64],
    grad_input: &mut Vec<f64>,
) -> Result<(), NequipOamPrefixError> {
    let input_dim = plan.input_dim;
    let output_dim = plan.output_dim;
    if grad_out.len() != row_count * output_dim {
        return Err(invalid_model("compact linear VJP output shape mismatch"));
    }
    if weight.len() != plan.weight_numel {
        return Err(invalid_model("compact linear VJP weight shape mismatch"));
    }
    grad_input.resize(row_count * input_dim, 0.0);
    grad_input.fill(0.0);
    for block in &plan.blocks {
        let mul_in = block.mul_in;
        let mul_out = block.mul_out;
        let ir_dim = block.ir_dim;
        let input_offset = block.input_offset;
        let output_offset = block.output_offset;
        let weight_offset = block.weight_offset;
        let path_weight = block.path_weight;
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if row_count >= 16 && ir_dim == 1 && mul_in >= 48 && mul_out >= 48 {
            let weight_end = weight_offset + mul_in * mul_out;
            portable_blas::dense_vjp_input_strided_block(
                row_count,
                input_dim,
                input_offset,
                mul_in,
                &weight[weight_offset..weight_end],
                grad_out,
                output_dim,
                output_offset,
                mul_out,
                grad_input,
                path_weight,
            );
            continue;
        }
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if compact_linear_allows_packed_blas()
            && row_count >= 16
            && ir_dim > 1
            && mul_in >= 32
            && mul_out >= 32
        {
            let weight_end = weight_offset + mul_in * mul_out;
            let block_weight = &weight[weight_offset..weight_end];
            let mut packed_grad_out = vec![0.0; row_count * mul_out];
            let mut packed_grad_input = vec![0.0; row_count * mul_in];
            for m in 0..ir_dim {
                for row in 0..row_count {
                    let output_row = row * output_dim + output_offset;
                    let packed_row = row * mul_out;
                    for out_mul in 0..mul_out {
                        packed_grad_out[packed_row + out_mul] =
                            grad_out[output_row + out_mul * ir_dim + m];
                    }
                }
                packed_grad_input.fill(0.0);
                portable_blas::dense_vjp_input(
                    row_count,
                    mul_in,
                    block_weight,
                    &packed_grad_out,
                    &mut packed_grad_input,
                    path_weight,
                );
                for row in 0..row_count {
                    let input_row = row * input_dim + input_offset;
                    let packed_row = row * mul_in;
                    for in_mul in 0..mul_in {
                        grad_input[input_row + in_mul * ir_dim + m] +=
                            packed_grad_input[packed_row + in_mul];
                    }
                }
            }
            continue;
        }
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && row_count >= OAM_COMPACT_LINEAR_PARALLEL_MIN_ROWS {
            grad_input
                .par_chunks_mut(input_dim)
                .enumerate()
                .for_each(|(row, grad_input_row)| {
                    let output_row = row * output_dim + output_offset;
                    for in_mul in 0..mul_in {
                        let weight_row = weight_offset + in_mul * mul_out;
                        for m in 0..ir_dim {
                            let mut value = 0.0;
                            for out_mul in 0..mul_out {
                                value += weight[weight_row + out_mul]
                                    * grad_out[output_row + out_mul * ir_dim + m];
                            }
                            grad_input_row[input_offset + in_mul * ir_dim + m] +=
                                path_weight * value;
                        }
                    }
                });
            continue;
        }
        for row in 0..row_count {
            let input_row = row * input_dim + input_offset;
            let output_row = row * output_dim + output_offset;
            for in_mul in 0..mul_in {
                let weight_row = weight_offset + in_mul * mul_out;
                for m in 0..ir_dim {
                    let mut value = 0.0;
                    for out_mul in 0..mul_out {
                        value += weight[weight_row + out_mul]
                            * grad_out[output_row + out_mul * ir_dim + m];
                    }
                    grad_input[input_row + in_mul * ir_dim + m] += path_weight * value;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(dead_code)]
fn compact_linear_vjp_input_row_into_oam(
    plan: &OamCompactLinearPlan,
    grad_out_row: &[f64],
    weight: &[f64],
    grad_input_row: &mut [f64],
) {
    debug_assert_eq!(grad_out_row.len(), plan.output_dim);
    debug_assert_eq!(weight.len(), plan.weight_numel);
    debug_assert_eq!(grad_input_row.len(), plan.input_dim);
    grad_input_row.fill(0.0);
    for block in &plan.blocks {
        for in_mul in 0..block.mul_in {
            let weight_row = block.weight_offset + in_mul * block.mul_out;
            for m in 0..block.ir_dim {
                let mut value = 0.0;
                for out_mul in 0..block.mul_out {
                    value += weight[weight_row + out_mul]
                        * grad_out_row[block.output_offset + out_mul * block.ir_dim + m];
                }
                grad_input_row[block.input_offset + in_mul * block.ir_dim + m] +=
                    block.path_weight * value;
            }
        }
    }
}

#[allow(dead_code)]
fn compact_linear_vjp_input_mapped(
    grad_out: &[f64],
    row_count: usize,
    spec: &Value,
    weight: &[f64],
    grad_out_map: Option<&[usize]>,
    grad_input_map: Option<&[usize]>,
) -> Result<Vec<f64>, NequipOamPrefixError> {
    let input_dim = usize_field(spec, "input_dim")?;
    let output_dim = usize_field(spec, "output_dim")?;
    if grad_out.len() != row_count * output_dim {
        return Err(invalid_model(
            "mapped compact linear VJP output shape mismatch",
        ));
    }
    if grad_out_map.is_some_and(|map| map.len() != output_dim)
        || grad_input_map.is_some_and(|map| map.len() != input_dim)
    {
        return Err(invalid_model(
            "mapped compact linear VJP layout map mismatch",
        ));
    }
    if weight.len() != usize_field(spec, "weight_numel")? {
        return Err(invalid_model(
            "mapped compact linear VJP weight shape mismatch",
        ));
    }
    let mut grad_input = vec![0.0; row_count * input_dim];
    for block in array_field(spec, "blocks")? {
        let mul_in = usize_field(block, "mul_in")?;
        let mul_out = usize_field(block, "mul_out")?;
        let ir_dim = usize_field(block, "ir_dim")?;
        let input_offset = usize_field(block, "input_offset")?;
        let output_offset = usize_field(block, "output_offset")?;
        let weight_offset = usize_field(block, "weight_offset")?;
        let path_weight = f64_field(block, "path_weight")?;
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        let input_identity =
            compact_linear_block_map_is_identity(grad_input_map, input_offset, mul_in, ir_dim);
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        let output_identity =
            compact_linear_block_map_is_identity(grad_out_map, output_offset, mul_out, ir_dim);
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if input_identity
            && output_identity
            && row_count >= 16
            && ir_dim == 1
            && mul_in >= 48
            && mul_out >= 48
        {
            let weight_end = weight_offset + mul_in * mul_out;
            portable_blas::dense_vjp_input_strided_block(
                row_count,
                input_dim,
                input_offset,
                mul_in,
                &weight[weight_offset..weight_end],
                grad_out,
                output_dim,
                output_offset,
                mul_out,
                &mut grad_input,
                path_weight,
            );
            continue;
        }
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if compact_linear_allows_packed_blas()
            && row_count >= 16
            && ir_dim > 1
            && mul_in >= 32
            && mul_out >= 32
        {
            let weight_end = weight_offset + mul_in * mul_out;
            let block_weight = &weight[weight_offset..weight_end];
            let mut packed_grad_out = vec![0.0; row_count * mul_out];
            let mut packed_grad_input = vec![0.0; row_count * mul_in];
            for m in 0..ir_dim {
                for row in 0..row_count {
                    let output_row = row * output_dim;
                    let packed_row = row * mul_out;
                    for out_mul in 0..mul_out {
                        let original_index = output_offset + out_mul * ir_dim + m;
                        packed_grad_out[packed_row + out_mul] =
                            grad_out[output_row + mapped_or_identity(grad_out_map, original_index)];
                    }
                }
                packed_grad_input.fill(0.0);
                portable_blas::dense_vjp_input(
                    row_count,
                    mul_in,
                    block_weight,
                    &packed_grad_out,
                    &mut packed_grad_input,
                    path_weight,
                );
                for row in 0..row_count {
                    let input_row = row * input_dim;
                    let packed_row = row * mul_in;
                    for in_mul in 0..mul_in {
                        let original_index = input_offset + in_mul * ir_dim + m;
                        grad_input
                            [input_row + mapped_or_identity(grad_input_map, original_index)] +=
                            packed_grad_input[packed_row + in_mul];
                    }
                }
            }
            continue;
        }
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && row_count >= OAM_COMPACT_LINEAR_PARALLEL_MIN_ROWS {
            grad_input
                .par_chunks_mut(input_dim)
                .enumerate()
                .for_each(|(row, grad_input_row)| {
                    let output_row = row * output_dim;
                    for in_mul in 0..mul_in {
                        let weight_row = weight_offset + in_mul * mul_out;
                        for m in 0..ir_dim {
                            let mut value = 0.0;
                            for out_mul in 0..mul_out {
                                let original_output_index = output_offset + out_mul * ir_dim + m;
                                value += weight[weight_row + out_mul]
                                    * grad_out[output_row
                                        + mapped_or_identity(grad_out_map, original_output_index)];
                            }
                            let original_input_index = input_offset + in_mul * ir_dim + m;
                            grad_input_row
                                [mapped_or_identity(grad_input_map, original_input_index)] +=
                                path_weight * value;
                        }
                    }
                });
            continue;
        }
        for row in 0..row_count {
            let input_row = row * input_dim;
            let output_row = row * output_dim;
            for in_mul in 0..mul_in {
                let weight_row = weight_offset + in_mul * mul_out;
                for m in 0..ir_dim {
                    let mut value = 0.0;
                    for out_mul in 0..mul_out {
                        let original_output_index = output_offset + out_mul * ir_dim + m;
                        value += weight[weight_row + out_mul]
                            * grad_out[output_row
                                + mapped_or_identity(grad_out_map, original_output_index)];
                    }
                    let original_input_index = input_offset + in_mul * ir_dim + m;
                    grad_input
                        [input_row + mapped_or_identity(grad_input_map, original_input_index)] +=
                        path_weight * value;
                }
            }
        }
    }
    Ok(grad_input)
}

#[allow(dead_code)]
fn compact_linear_vjp_input_mapped_f32(
    grad_out: &[f32],
    row_count: usize,
    spec: &Value,
    weight: &[f32],
    grad_out_map: Option<&[usize]>,
    grad_input_map: Option<&[usize]>,
) -> Result<Vec<f32>, NequipOamPrefixError> {
    let input_dim = usize_field(spec, "input_dim")?;
    let output_dim = usize_field(spec, "output_dim")?;
    if grad_out.len() != row_count * output_dim {
        return Err(invalid_model(
            "mapped compact f32 linear VJP output shape mismatch",
        ));
    }
    if grad_out_map.is_some_and(|map| map.len() != output_dim)
        || grad_input_map.is_some_and(|map| map.len() != input_dim)
    {
        return Err(invalid_model(
            "mapped compact f32 linear VJP layout map mismatch",
        ));
    }
    if weight.len() != usize_field(spec, "weight_numel")? {
        return Err(invalid_model(
            "mapped compact f32 linear VJP weight shape mismatch",
        ));
    }
    let mut grad_input = vec![0.0; row_count * input_dim];
    for block in array_field(spec, "blocks")? {
        let mul_in = usize_field(block, "mul_in")?;
        let mul_out = usize_field(block, "mul_out")?;
        let ir_dim = usize_field(block, "ir_dim")?;
        let input_offset = usize_field(block, "input_offset")?;
        let output_offset = usize_field(block, "output_offset")?;
        let weight_offset = usize_field(block, "weight_offset")?;
        let path_weight = f64_field(block, "path_weight")? as f32;
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        let input_identity =
            compact_linear_block_map_is_identity(grad_input_map, input_offset, mul_in, ir_dim);
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        let output_identity =
            compact_linear_block_map_is_identity(grad_out_map, output_offset, mul_out, ir_dim);
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if input_identity
            && output_identity
            && row_count >= 16
            && ir_dim == 1
            && mul_in >= 48
            && mul_out >= 48
        {
            let weight_end = weight_offset + mul_in * mul_out;
            crate::blas_backend::dense_vjp_input_strided_f32(
                row_count,
                mul_in,
                &weight[weight_offset..weight_end],
                mul_out,
                &grad_out[output_offset..],
                output_dim,
                &mut grad_input[input_offset..],
                input_dim,
                path_weight,
                1.0,
            );
            continue;
        }
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if compact_linear_allows_packed_blas()
            && row_count >= 16
            && ir_dim > 1
            && mul_in >= 32
            && mul_out >= 32
        {
            let weight_end = weight_offset + mul_in * mul_out;
            let block_weight = &weight[weight_offset..weight_end];
            let mut packed_grad_out = vec![0.0; row_count * mul_out];
            let mut packed_grad_input = vec![0.0; row_count * mul_in];
            for m in 0..ir_dim {
                for row in 0..row_count {
                    let output_row = row * output_dim;
                    let packed_row = row * mul_out;
                    for out_mul in 0..mul_out {
                        let original_index = output_offset + out_mul * ir_dim + m;
                        packed_grad_out[packed_row + out_mul] =
                            grad_out[output_row + mapped_or_identity(grad_out_map, original_index)];
                    }
                }
                packed_grad_input.fill(0.0);
                crate::blas_backend::dense_vjp_input_f32(
                    row_count,
                    mul_in,
                    block_weight,
                    &packed_grad_out,
                    &mut packed_grad_input,
                    path_weight,
                );
                for row in 0..row_count {
                    let input_row = row * input_dim;
                    let packed_row = row * mul_in;
                    for in_mul in 0..mul_in {
                        let original_index = input_offset + in_mul * ir_dim + m;
                        grad_input
                            [input_row + mapped_or_identity(grad_input_map, original_index)] +=
                            packed_grad_input[packed_row + in_mul];
                    }
                }
            }
            continue;
        }
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && row_count >= OAM_COMPACT_LINEAR_PARALLEL_MIN_ROWS {
            grad_input
                .par_chunks_mut(input_dim)
                .enumerate()
                .for_each(|(row, grad_input_row)| {
                    let output_row = row * output_dim;
                    for in_mul in 0..mul_in {
                        let weight_row = weight_offset + in_mul * mul_out;
                        for m in 0..ir_dim {
                            let mut value = 0.0;
                            for out_mul in 0..mul_out {
                                let original_output_index = output_offset + out_mul * ir_dim + m;
                                value += weight[weight_row + out_mul]
                                    * grad_out[output_row
                                        + mapped_or_identity(grad_out_map, original_output_index)];
                            }
                            let original_input_index = input_offset + in_mul * ir_dim + m;
                            grad_input_row
                                [mapped_or_identity(grad_input_map, original_input_index)] +=
                                path_weight * value;
                        }
                    }
                });
            continue;
        }
        for row in 0..row_count {
            let input_row = row * input_dim;
            let output_row = row * output_dim;
            for in_mul in 0..mul_in {
                let weight_row = weight_offset + in_mul * mul_out;
                for m in 0..ir_dim {
                    let mut value = 0.0;
                    for out_mul in 0..mul_out {
                        let original_output_index = output_offset + out_mul * ir_dim + m;
                        value += weight[weight_row + out_mul]
                            * grad_out[output_row
                                + mapped_or_identity(grad_out_map, original_output_index)];
                    }
                    let original_input_index = input_offset + in_mul * ir_dim + m;
                    grad_input
                        [input_row + mapped_or_identity(grad_input_map, original_input_index)] +=
                        path_weight * value;
                }
            }
        }
    }
    Ok(grad_input)
}

#[allow(dead_code)]
fn compact_linear_vjp_input_f32(
    grad_out: &[f32],
    row_count: usize,
    spec: &Value,
    weight: &[f32],
) -> Result<Vec<f32>, NequipOamPrefixError> {
    let mut grad_input = Vec::new();
    compact_linear_vjp_input_into_f32(grad_out, row_count, spec, weight, &mut grad_input)?;
    Ok(grad_input)
}

#[cfg(test)]
#[allow(dead_code)]
fn compact_linear_vjp_input_plan_f32(
    grad_out: &[f32],
    row_count: usize,
    plan: &OamCompactLinearPlan,
    weight: &[f32],
) -> Result<Vec<f32>, NequipOamPrefixError> {
    let mut grad_input = Vec::new();
    compact_linear_vjp_input_into_plan_f32(grad_out, row_count, plan, weight, &mut grad_input)?;
    Ok(grad_input)
}

#[allow(dead_code)]
fn compact_linear_vjp_input_into_f32(
    grad_out: &[f32],
    row_count: usize,
    spec: &Value,
    weight: &[f32],
    grad_input: &mut Vec<f32>,
) -> Result<(), NequipOamPrefixError> {
    let input_dim = usize_field(spec, "input_dim")?;
    let output_dim = usize_field(spec, "output_dim")?;
    if grad_out.len() != row_count * output_dim {
        return Err(invalid_model(
            "compact f32 linear VJP output shape mismatch",
        ));
    }
    if weight.len() != usize_field(spec, "weight_numel")? {
        return Err(invalid_model(
            "compact f32 linear VJP weight shape mismatch",
        ));
    }
    grad_input.resize(row_count * input_dim, 0.0);
    grad_input.fill(0.0);
    for block in array_field(spec, "blocks")? {
        let mul_in = usize_field(block, "mul_in")?;
        let mul_out = usize_field(block, "mul_out")?;
        let ir_dim = usize_field(block, "ir_dim")?;
        let input_offset = usize_field(block, "input_offset")?;
        let output_offset = usize_field(block, "output_offset")?;
        let weight_offset = usize_field(block, "weight_offset")?;
        let path_weight = f64_field(block, "path_weight")? as f32;
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if row_count >= 16 && ir_dim == 1 && mul_in >= 48 && mul_out >= 48 {
            let weight_end = weight_offset + mul_in * mul_out;
            crate::blas_backend::dense_vjp_input_strided_f32(
                row_count,
                mul_in,
                &weight[weight_offset..weight_end],
                mul_out,
                &grad_out[output_offset..],
                output_dim,
                &mut grad_input[input_offset..],
                input_dim,
                path_weight,
                1.0,
            );
            continue;
        }
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if compact_linear_allows_packed_blas()
            && row_count >= 16
            && ir_dim > 1
            && mul_in >= 32
            && mul_out >= 32
        {
            let weight_end = weight_offset + mul_in * mul_out;
            let block_weight = &weight[weight_offset..weight_end];
            let mut packed_grad_out = vec![0.0; row_count * mul_out];
            let mut packed_grad_input = vec![0.0; row_count * mul_in];
            for m in 0..ir_dim {
                for row in 0..row_count {
                    let output_row = row * output_dim + output_offset;
                    let packed_row = row * mul_out;
                    for out_mul in 0..mul_out {
                        packed_grad_out[packed_row + out_mul] =
                            grad_out[output_row + out_mul * ir_dim + m];
                    }
                }
                packed_grad_input.fill(0.0);
                crate::blas_backend::dense_vjp_input_f32(
                    row_count,
                    mul_in,
                    block_weight,
                    &packed_grad_out,
                    &mut packed_grad_input,
                    path_weight,
                );
                for row in 0..row_count {
                    let input_row = row * input_dim + input_offset;
                    let packed_row = row * mul_in;
                    for in_mul in 0..mul_in {
                        grad_input[input_row + in_mul * ir_dim + m] +=
                            packed_grad_input[packed_row + in_mul];
                    }
                }
            }
            continue;
        }
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && row_count >= OAM_COMPACT_LINEAR_PARALLEL_MIN_ROWS {
            grad_input
                .par_chunks_mut(input_dim)
                .enumerate()
                .for_each(|(row, grad_input_row)| {
                    let output_row = row * output_dim + output_offset;
                    for in_mul in 0..mul_in {
                        let weight_row = weight_offset + in_mul * mul_out;
                        for m in 0..ir_dim {
                            let mut value = 0.0;
                            for out_mul in 0..mul_out {
                                value += weight[weight_row + out_mul]
                                    * grad_out[output_row + out_mul * ir_dim + m];
                            }
                            grad_input_row[input_offset + in_mul * ir_dim + m] +=
                                path_weight * value;
                        }
                    }
                });
            continue;
        }
        for row in 0..row_count {
            let input_row = row * input_dim + input_offset;
            let output_row = row * output_dim + output_offset;
            for in_mul in 0..mul_in {
                let weight_row = weight_offset + in_mul * mul_out;
                for m in 0..ir_dim {
                    let mut value = 0.0;
                    for out_mul in 0..mul_out {
                        value += weight[weight_row + out_mul]
                            * grad_out[output_row + out_mul * ir_dim + m];
                    }
                    grad_input[input_row + in_mul * ir_dim + m] += path_weight * value;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(dead_code)]
fn compact_linear_vjp_input_into_plan_f32(
    grad_out: &[f32],
    row_count: usize,
    plan: &OamCompactLinearPlan,
    weight: &[f32],
    grad_input: &mut Vec<f32>,
) -> Result<(), NequipOamPrefixError> {
    let input_dim = plan.input_dim;
    let output_dim = plan.output_dim;
    if grad_out.len() != row_count * output_dim {
        return Err(invalid_model(
            "compact f32 linear VJP output shape mismatch",
        ));
    }
    if weight.len() != plan.weight_numel {
        return Err(invalid_model(
            "compact f32 linear VJP weight shape mismatch",
        ));
    }
    grad_input.resize(row_count * input_dim, 0.0);
    grad_input.fill(0.0);
    for block in &plan.blocks {
        let mul_in = block.mul_in;
        let mul_out = block.mul_out;
        let ir_dim = block.ir_dim;
        let input_offset = block.input_offset;
        let output_offset = block.output_offset;
        let weight_offset = block.weight_offset;
        let path_weight = block.path_weight as f32;
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if row_count >= 16 && ir_dim == 1 && mul_in >= 48 && mul_out >= 48 {
            let weight_end = weight_offset + mul_in * mul_out;
            crate::blas_backend::dense_vjp_input_strided_f32(
                row_count,
                mul_in,
                &weight[weight_offset..weight_end],
                mul_out,
                &grad_out[output_offset..],
                output_dim,
                &mut grad_input[input_offset..],
                input_dim,
                path_weight,
                1.0,
            );
            continue;
        }
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        if compact_linear_allows_packed_blas()
            && row_count >= 16
            && ir_dim > 1
            && mul_in >= 32
            && mul_out >= 32
        {
            let weight_end = weight_offset + mul_in * mul_out;
            let block_weight = &weight[weight_offset..weight_end];
            let mut packed_grad_out = vec![0.0; row_count * mul_out];
            let mut packed_grad_input = vec![0.0; row_count * mul_in];
            for m in 0..ir_dim {
                for row in 0..row_count {
                    let output_row = row * output_dim + output_offset;
                    let packed_row = row * mul_out;
                    for out_mul in 0..mul_out {
                        packed_grad_out[packed_row + out_mul] =
                            grad_out[output_row + out_mul * ir_dim + m];
                    }
                }
                packed_grad_input.fill(0.0);
                crate::blas_backend::dense_vjp_input_f32(
                    row_count,
                    mul_in,
                    block_weight,
                    &packed_grad_out,
                    &mut packed_grad_input,
                    path_weight,
                );
                for row in 0..row_count {
                    let input_row = row * input_dim + input_offset;
                    let packed_row = row * mul_in;
                    for in_mul in 0..mul_in {
                        grad_input[input_row + in_mul * ir_dim + m] +=
                            packed_grad_input[packed_row + in_mul];
                    }
                }
            }
            continue;
        }
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && row_count >= OAM_COMPACT_LINEAR_PARALLEL_MIN_ROWS {
            grad_input
                .par_chunks_mut(input_dim)
                .enumerate()
                .for_each(|(row, grad_input_row)| {
                    let output_row = row * output_dim + output_offset;
                    for in_mul in 0..mul_in {
                        let weight_row = weight_offset + in_mul * mul_out;
                        for m in 0..ir_dim {
                            let mut value = 0.0;
                            for out_mul in 0..mul_out {
                                value += weight[weight_row + out_mul]
                                    * grad_out[output_row + out_mul * ir_dim + m];
                            }
                            grad_input_row[input_offset + in_mul * ir_dim + m] +=
                                path_weight * value;
                        }
                    }
                });
            continue;
        }
        for row in 0..row_count {
            let input_row = row * input_dim + input_offset;
            let output_row = row * output_dim + output_offset;
            for in_mul in 0..mul_in {
                let weight_row = weight_offset + in_mul * mul_out;
                for m in 0..ir_dim {
                    let mut value = 0.0;
                    for out_mul in 0..mul_out {
                        value += weight[weight_row + out_mul]
                            * grad_out[output_row + out_mul * ir_dim + m];
                    }
                    grad_input[input_row + in_mul * ir_dim + m] += path_weight * value;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(dead_code)]
fn compact_linear_vjp_input_row_into_oam_f32(
    plan: &OamCompactLinearPlan,
    grad_out_row: &[f32],
    weight: &[f32],
    grad_input_row: &mut [f32],
) {
    debug_assert_eq!(grad_out_row.len(), plan.output_dim);
    debug_assert_eq!(weight.len(), plan.weight_numel);
    debug_assert_eq!(grad_input_row.len(), plan.input_dim);
    grad_input_row.fill(0.0);
    for block in &plan.blocks {
        for in_mul in 0..block.mul_in {
            let weight_row = block.weight_offset + in_mul * block.mul_out;
            for m in 0..block.ir_dim {
                let mut value = 0.0;
                for out_mul in 0..block.mul_out {
                    value += weight[weight_row + out_mul]
                        * grad_out_row[block.output_offset + out_mul * block.ir_dim + m];
                }
                grad_input_row[block.input_offset + in_mul * block.ir_dim + m] +=
                    block.path_weight as f32 * value;
            }
        }
    }
}

fn scalar_attr_tensor_product_contractions(
    attrs: &[f64],
    active_type_rows: &ActiveTypeRows,
    row_count: usize,
    spec: &Value,
    weight: &[f64],
) -> Result<ScalarAttrContractions, NequipOamPrefixError> {
    let input_dim = usize_field(spec, "input1_dim")?;
    let attr_dim = usize_field(spec, "input2_dim")?;
    let output_dim = usize_field(spec, "output_dim")?;
    if attrs.len() != row_count * attr_dim
        || active_type_rows
            .active_rows
            .iter()
            .any(|&row| row >= row_count)
    {
        return Err(invalid_model(
            "scalar-attribute tensor product contraction shape mismatch",
        ));
    }
    if weight.len() != usize_field(spec, "weight_numel")? {
        return Err(invalid_model(
            "scalar-attribute tensor product weight shape mismatch",
        ));
    }
    let mut contractions = Vec::new();
    for block in array_field(spec, "blocks")? {
        let mul_in = usize_field(block, "mul_in")?;
        let attr_mul = usize_field(block, "attr_mul")?;
        let mul_out = usize_field(block, "mul_out")?;
        let ir_dim = usize_field(block, "ir_dim")?;
        let input_offset = usize_field(block, "input_offset")?;
        let output_offset = usize_field(block, "output_offset")?;
        let weight_offset = usize_field(block, "weight_offset")?;
        let scale = f64_field(block, "path_weight")? * f64_field(block, "cg_scalar")?;
        if attr_mul > attr_dim {
            return Err(invalid_model("scalar-attribute block attr shape mismatch"));
        }
        if input_offset + mul_in * ir_dim > input_dim
            || output_offset + mul_out * ir_dim > output_dim
        {
            return Err(invalid_model(
                "scalar-attribute block input/output shape mismatch",
            ));
        }
        if weight_offset + mul_in * attr_mul * mul_out > weight.len() {
            return Err(invalid_model(
                "scalar-attribute block weight shape mismatch",
            ));
        }
        let contracted_stride = mul_in * mul_out;
        let mut values = vec![0.0; active_type_rows.active_rows.len() * contracted_stride];
        for (slot, &row) in active_type_rows.active_rows.iter().enumerate() {
            if row >= row_count {
                return Err(invalid_model(
                    "scalar-attribute active type row out of bounds",
                ));
            }
            let attr_row = &attrs[row * attr_dim..(row + 1) * attr_dim];
            let contracted_base = slot * contracted_stride;
            for in_mul in 0..mul_in {
                for out_mul in 0..mul_out {
                    let mut value = 0.0;
                    for (attr, &attr_value) in attr_row.iter().enumerate().take(attr_mul) {
                        let weight_index =
                            weight_offset + in_mul * attr_mul * mul_out + attr * mul_out + out_mul;
                        value += weight[weight_index] * attr_value;
                    }
                    values[contracted_base + in_mul * mul_out + out_mul] = value;
                }
            }
        }
        contractions.push(ScalarAttrBlockContraction {
            mul_in,
            mul_out,
            ir_dim,
            input_offset,
            output_offset,
            scale,
            values,
        });
    }
    Ok(ScalarAttrContractions {
        input_dim,
        output_dim,
        blocks: contractions,
    })
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        target_arch = "x86_64"
    )
))]
fn scalar_attr_grouped_blas_is_available<T>(
    active_type_rows: &ActiveTypeRows,
    contractions: &[T],
) -> bool
where
    T: ScalarAttrBlockShape,
{
    contractions.iter().any(|contraction| {
        contraction.mul_in() >= 32
            && contraction.mul_out() >= 32
            && active_type_rows
                .type_row_offsets
                .windows(2)
                .any(|range| range[1] - range[0] >= 16)
    })
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        target_arch = "x86_64"
    )
))]
trait ScalarAttrBlockShape {
    fn mul_in(&self) -> usize;
    fn mul_out(&self) -> usize;
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        target_arch = "x86_64"
    )
))]
impl ScalarAttrBlockShape for ScalarAttrBlockContraction {
    fn mul_in(&self) -> usize {
        self.mul_in
    }

    fn mul_out(&self) -> usize {
        self.mul_out
    }
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        target_arch = "x86_64"
    )
))]
impl ScalarAttrBlockShape for ScalarAttrBlockContractionF32 {
    fn mul_in(&self) -> usize {
        self.mul_in
    }

    fn mul_out(&self) -> usize {
        self.mul_out
    }
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        target_arch = "x86_64"
    )
))]
fn scalar_attr_tensor_product_forward_accumulate_grouped_blas(
    input: &[f64],
    active_type_rows: &ActiveTypeRows,
    contractions: &ScalarAttrContractions,
    out: &mut [f64],
) {
    let input_dim = contractions.input_dim;
    let output_dim = contractions.output_dim;
    let mut packed_input = Vec::new();
    let mut packed_output = Vec::new();
    for contraction in &contractions.blocks {
        let contracted_stride = contraction.mul_in * contraction.mul_out;
        for slot in 0..active_type_rows.active_rows.len() {
            let rows = &active_type_rows.type_rows[active_type_rows.type_row_offsets[slot]
                ..active_type_rows.type_row_offsets[slot + 1]];
            let contracted_base = slot * contracted_stride;
            if rows.len() >= 16 && contraction.mul_in >= 32 && contraction.mul_out >= 32 {
                let contracted =
                    &contraction.values[contracted_base..contracted_base + contracted_stride];
                packed_input.resize(rows.len() * contraction.mul_in, 0.0);
                packed_output.resize(rows.len() * contraction.mul_out, 0.0);
                for m in 0..contraction.ir_dim {
                    for (packed_row, &row) in rows.iter().enumerate() {
                        let input_row = row * input_dim + contraction.input_offset;
                        let packed_row_start = packed_row * contraction.mul_in;
                        for in_mul in 0..contraction.mul_in {
                            packed_input[packed_row_start + in_mul] =
                                input[input_row + in_mul * contraction.ir_dim + m];
                        }
                    }
                    portable_blas::dense_forward(
                        &packed_input,
                        rows.len(),
                        contraction.mul_in,
                        contracted,
                        &mut packed_output,
                        contraction.scale,
                    );
                    for (packed_row, &row) in rows.iter().enumerate() {
                        let output_row = row * output_dim + contraction.output_offset;
                        let packed_row_start = packed_row * contraction.mul_out;
                        for out_mul in 0..contraction.mul_out {
                            out[output_row + out_mul * contraction.ir_dim + m] +=
                                packed_output[packed_row_start + out_mul];
                        }
                    }
                }
            } else {
                for &row in rows {
                    let input_row = row * input_dim + contraction.input_offset;
                    let output_row = row * output_dim + contraction.output_offset;
                    for out_mul in 0..contraction.mul_out {
                        for m in 0..contraction.ir_dim {
                            let mut value = 0.0;
                            for in_mul in 0..contraction.mul_in {
                                let x = input[input_row + in_mul * contraction.ir_dim + m];
                                value += x * contraction.values
                                    [contracted_base + in_mul * contraction.mul_out + out_mul];
                            }
                            out[output_row + out_mul * contraction.ir_dim + m] +=
                                contraction.scale * value;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        target_arch = "x86_64"
    )
))]
fn scalar_attr_tensor_product_vjp_input_accumulate_grouped_blas(
    grad_out: &[f64],
    active_type_rows: &ActiveTypeRows,
    contractions: &ScalarAttrContractions,
    grad_input: &mut [f64],
) {
    let input_dim = contractions.input_dim;
    let output_dim = contractions.output_dim;
    let mut packed_grad_out = Vec::new();
    let mut packed_grad_input = Vec::new();
    for contraction in &contractions.blocks {
        let contracted_stride = contraction.mul_in * contraction.mul_out;
        for slot in 0..active_type_rows.active_rows.len() {
            let rows = &active_type_rows.type_rows[active_type_rows.type_row_offsets[slot]
                ..active_type_rows.type_row_offsets[slot + 1]];
            let contracted_base = slot * contracted_stride;
            if rows.len() >= 16 && contraction.mul_in >= 32 && contraction.mul_out >= 32 {
                let contracted =
                    &contraction.values[contracted_base..contracted_base + contracted_stride];
                packed_grad_out.resize(rows.len() * contraction.mul_out, 0.0);
                packed_grad_input.resize(rows.len() * contraction.mul_in, 0.0);
                for m in 0..contraction.ir_dim {
                    for (packed_row, &row) in rows.iter().enumerate() {
                        let output_row = row * output_dim + contraction.output_offset;
                        let packed_row_start = packed_row * contraction.mul_out;
                        for out_mul in 0..contraction.mul_out {
                            packed_grad_out[packed_row_start + out_mul] =
                                grad_out[output_row + out_mul * contraction.ir_dim + m];
                        }
                    }
                    packed_grad_input.fill(0.0);
                    portable_blas::dense_vjp_input(
                        rows.len(),
                        contraction.mul_in,
                        contracted,
                        &packed_grad_out,
                        &mut packed_grad_input,
                        contraction.scale,
                    );
                    for (packed_row, &row) in rows.iter().enumerate() {
                        let input_row = row * input_dim + contraction.input_offset;
                        let packed_row_start = packed_row * contraction.mul_in;
                        for in_mul in 0..contraction.mul_in {
                            grad_input[input_row + in_mul * contraction.ir_dim + m] +=
                                packed_grad_input[packed_row_start + in_mul];
                        }
                    }
                }
            } else {
                for &row in rows {
                    let input_row = row * input_dim + contraction.input_offset;
                    let output_row = row * output_dim + contraction.output_offset;
                    for out_mul in 0..contraction.mul_out {
                        for m in 0..contraction.ir_dim {
                            let bar_out = grad_out[output_row + out_mul * contraction.ir_dim + m];
                            for in_mul in 0..contraction.mul_in {
                                let input_index = input_row + in_mul * contraction.ir_dim + m;
                                grad_input[input_index] += contraction.scale
                                    * contraction.values
                                        [contracted_base + in_mul * contraction.mul_out + out_mul]
                                    * bar_out;
                            }
                        }
                    }
                }
            }
        }
    }
}

fn scalar_attr_tensor_product_forward_accumulate(
    input: &[f64],
    active_type_rows: &ActiveTypeRows,
    row_count: usize,
    contractions: &ScalarAttrContractions,
    out: &mut [f64],
) -> Result<(), NequipOamPrefixError> {
    let input_dim = contractions.input_dim;
    let output_dim = contractions.output_dim;
    if input.len() != row_count * input_dim
        || out.len() != row_count * output_dim
        || active_type_rows.row_type_slots.len() != row_count
    {
        return Err(invalid_model(
            "scalar-attribute tensor product input shape mismatch",
        ));
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    ))]
    if scalar_attr_grouped_blas_is_available(active_type_rows, &contractions.blocks) {
        scalar_attr_tensor_product_forward_accumulate_grouped_blas(
            input,
            active_type_rows,
            contractions,
            out,
        );
        return Ok(());
    }
    for contraction in &contractions.blocks {
        let contracted_stride = contraction.mul_in * contraction.mul_out;
        for row in 0..row_count {
            let slot = active_type_rows.row_type_slots[row];
            let contracted_base = slot * contracted_stride;
            debug_assert!(contracted_base + contracted_stride <= contraction.values.len());
            let input_row = row * input_dim + contraction.input_offset;
            let output_row = row * output_dim + contraction.output_offset;
            for out_mul in 0..contraction.mul_out {
                for m in 0..contraction.ir_dim {
                    let mut value = 0.0;
                    for in_mul in 0..contraction.mul_in {
                        let x = input[input_row + in_mul * contraction.ir_dim + m];
                        value += x * contraction.values
                            [contracted_base + in_mul * contraction.mul_out + out_mul];
                    }
                    out[output_row + out_mul * contraction.ir_dim + m] += contraction.scale * value;
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "parallel")]
fn scalar_attr_tensor_product_forward_accumulate_parallel(
    input: &[f64],
    active_type_rows: &ActiveTypeRows,
    row_count: usize,
    contractions: &ScalarAttrContractions,
    out: &mut [f64],
) -> Result<(), NequipOamPrefixError> {
    let input_dim = contractions.input_dim;
    let output_dim = contractions.output_dim;
    if input.len() != row_count * input_dim
        || out.len() != row_count * output_dim
        || active_type_rows.row_type_slots.len() != row_count
    {
        return Err(invalid_model(
            "parallel scalar-attribute tensor product input shape mismatch",
        ));
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    ))]
    if scalar_attr_grouped_blas_is_available(active_type_rows, &contractions.blocks) {
        scalar_attr_tensor_product_forward_accumulate_grouped_blas(
            input,
            active_type_rows,
            contractions,
            out,
        );
        return Ok(());
    }
    for contraction in &contractions.blocks {
        let contracted_stride = contraction.mul_in * contraction.mul_out;
        out.par_chunks_mut(output_dim)
            .enumerate()
            .for_each(|(row, out_row)| {
                let slot = active_type_rows.row_type_slots[row];
                let contracted_base = slot * contracted_stride;
                debug_assert!(contracted_base + contracted_stride <= contraction.values.len());
                let input_row = row * input_dim + contraction.input_offset;
                for out_mul in 0..contraction.mul_out {
                    for m in 0..contraction.ir_dim {
                        let mut value = 0.0;
                        for in_mul in 0..contraction.mul_in {
                            let x = input[input_row + in_mul * contraction.ir_dim + m];
                            value += x * contraction.values
                                [contracted_base + in_mul * contraction.mul_out + out_mul];
                        }
                        out_row[contraction.output_offset + out_mul * contraction.ir_dim + m] +=
                            contraction.scale * value;
                    }
                }
            });
    }
    Ok(())
}

fn scalar_attr_tensor_product_vjp_input_accumulate(
    grad_out: &[f64],
    input: &[f64],
    active_type_rows: &ActiveTypeRows,
    row_count: usize,
    contractions: &ScalarAttrContractions,
    grad_input: &mut [f64],
) -> Result<(), NequipOamPrefixError> {
    let input_dim = contractions.input_dim;
    let output_dim = contractions.output_dim;
    if input.len() != row_count * input_dim
        || grad_out.len() != row_count * output_dim
        || grad_input.len() != row_count * input_dim
        || active_type_rows.row_type_slots.len() != row_count
    {
        return Err(invalid_model(
            "scalar-attribute tensor product VJP input shape mismatch",
        ));
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    ))]
    if scalar_attr_grouped_blas_is_available(active_type_rows, &contractions.blocks) {
        scalar_attr_tensor_product_vjp_input_accumulate_grouped_blas(
            grad_out,
            active_type_rows,
            contractions,
            grad_input,
        );
        return Ok(());
    }
    for contraction in &contractions.blocks {
        let contracted_stride = contraction.mul_in * contraction.mul_out;
        for row in 0..row_count {
            let slot = active_type_rows.row_type_slots[row];
            let contracted_base = slot * contracted_stride;
            debug_assert!(contracted_base + contracted_stride <= contraction.values.len());
            let input_row = row * input_dim + contraction.input_offset;
            let output_row = row * output_dim + contraction.output_offset;
            for out_mul in 0..contraction.mul_out {
                for m in 0..contraction.ir_dim {
                    let bar_out = grad_out[output_row + out_mul * contraction.ir_dim + m];
                    for in_mul in 0..contraction.mul_in {
                        let input_index = input_row + in_mul * contraction.ir_dim + m;
                        grad_input[input_index] += contraction.scale
                            * contraction.values
                                [contracted_base + in_mul * contraction.mul_out + out_mul]
                            * bar_out;
                    }
                }
            }
        }
    }
    Ok(())
}

#[inline(always)]
fn scalar_attr_tensor_product_vjp_input_accumulate_auto(
    grad_out: &[f64],
    input: &[f64],
    active_type_rows: &ActiveTypeRows,
    row_count: usize,
    contractions: &ScalarAttrContractions,
    grad_input: &mut [f64],
) -> Result<(), NequipOamPrefixError> {
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && row_count >= OAM_COMPACT_LINEAR_PARALLEL_MIN_ROWS {
        return scalar_attr_tensor_product_vjp_input_accumulate_parallel(
            grad_out,
            input,
            active_type_rows,
            row_count,
            contractions,
            grad_input,
        );
    }
    scalar_attr_tensor_product_vjp_input_accumulate(
        grad_out,
        input,
        active_type_rows,
        row_count,
        contractions,
        grad_input,
    )
}

#[cfg(feature = "parallel")]
#[inline(never)]
fn scalar_attr_tensor_product_vjp_input_accumulate_parallel(
    grad_out: &[f64],
    input: &[f64],
    active_type_rows: &ActiveTypeRows,
    row_count: usize,
    contractions: &ScalarAttrContractions,
    grad_input: &mut [f64],
) -> Result<(), NequipOamPrefixError> {
    let input_dim = contractions.input_dim;
    let output_dim = contractions.output_dim;
    if input.len() != row_count * input_dim
        || grad_out.len() != row_count * output_dim
        || grad_input.len() != row_count * input_dim
        || active_type_rows.row_type_slots.len() != row_count
    {
        return Err(invalid_model(
            "parallel scalar-attribute tensor product VJP input shape mismatch",
        ));
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    ))]
    if scalar_attr_grouped_blas_is_available(active_type_rows, &contractions.blocks) {
        scalar_attr_tensor_product_vjp_input_accumulate_grouped_blas(
            grad_out,
            active_type_rows,
            contractions,
            grad_input,
        );
        return Ok(());
    }
    for contraction in &contractions.blocks {
        let contracted_stride = contraction.mul_in * contraction.mul_out;
        grad_input
            .par_chunks_mut(input_dim)
            .enumerate()
            .for_each(|(row, grad_input_row)| {
                let slot = active_type_rows.row_type_slots[row];
                let contracted_base = slot * contracted_stride;
                debug_assert!(contracted_base + contracted_stride <= contraction.values.len());
                let output_row = row * output_dim + contraction.output_offset;
                for out_mul in 0..contraction.mul_out {
                    for m in 0..contraction.ir_dim {
                        let bar_out = grad_out[output_row + out_mul * contraction.ir_dim + m];
                        for in_mul in 0..contraction.mul_in {
                            grad_input_row
                                [contraction.input_offset + in_mul * contraction.ir_dim + m] +=
                                contraction.scale
                                    * contraction.values
                                        [contracted_base + in_mul * contraction.mul_out + out_mul]
                                    * bar_out;
                        }
                    }
                }
            });
    }
    Ok(())
}

#[allow(dead_code)]
fn scalar_attr_tensor_product_contractions_f32(
    attrs: &[f32],
    active_type_rows: &ActiveTypeRows,
    row_count: usize,
    spec: &Value,
    weight: &[f32],
) -> Result<ScalarAttrContractionsF32, NequipOamPrefixError> {
    let input_dim = usize_field(spec, "input1_dim")?;
    let attr_dim = usize_field(spec, "input2_dim")?;
    let output_dim = usize_field(spec, "output_dim")?;
    if attrs.len() != row_count * attr_dim
        || active_type_rows
            .active_rows
            .iter()
            .any(|&row| row >= row_count)
    {
        return Err(invalid_model(
            "scalar-attribute f32 tensor product contraction shape mismatch",
        ));
    }
    if weight.len() != usize_field(spec, "weight_numel")? {
        return Err(invalid_model(
            "scalar-attribute f32 tensor product weight shape mismatch",
        ));
    }
    let mut contractions = Vec::new();
    for block in array_field(spec, "blocks")? {
        let mul_in = usize_field(block, "mul_in")?;
        let attr_mul = usize_field(block, "attr_mul")?;
        let mul_out = usize_field(block, "mul_out")?;
        let ir_dim = usize_field(block, "ir_dim")?;
        let input_offset = usize_field(block, "input_offset")?;
        let output_offset = usize_field(block, "output_offset")?;
        let weight_offset = usize_field(block, "weight_offset")?;
        let scale = (f64_field(block, "path_weight")? * f64_field(block, "cg_scalar")?) as f32;
        if attr_mul > attr_dim {
            return Err(invalid_model(
                "scalar-attribute f32 block attr shape mismatch",
            ));
        }
        if input_offset + mul_in * ir_dim > input_dim
            || output_offset + mul_out * ir_dim > output_dim
        {
            return Err(invalid_model(
                "scalar-attribute f32 block input/output shape mismatch",
            ));
        }
        if weight_offset + mul_in * attr_mul * mul_out > weight.len() {
            return Err(invalid_model(
                "scalar-attribute f32 block weight shape mismatch",
            ));
        }
        let contracted_stride = mul_in * mul_out;
        let mut values = vec![0.0; active_type_rows.active_rows.len() * contracted_stride];
        for (slot, &row) in active_type_rows.active_rows.iter().enumerate() {
            if row >= row_count {
                return Err(invalid_model(
                    "scalar-attribute f32 active type row out of bounds",
                ));
            }
            let attr_row = &attrs[row * attr_dim..(row + 1) * attr_dim];
            let contracted_base = slot * contracted_stride;
            for in_mul in 0..mul_in {
                for out_mul in 0..mul_out {
                    let mut value = 0.0;
                    for (attr, &attr_value) in attr_row.iter().enumerate().take(attr_mul) {
                        let weight_index =
                            weight_offset + in_mul * attr_mul * mul_out + attr * mul_out + out_mul;
                        value += weight[weight_index] * attr_value;
                    }
                    values[contracted_base + in_mul * mul_out + out_mul] = value;
                }
            }
        }
        contractions.push(ScalarAttrBlockContractionF32 {
            mul_in,
            mul_out,
            ir_dim,
            input_offset,
            output_offset,
            scale,
            values,
        });
    }
    Ok(ScalarAttrContractionsF32 {
        input_dim,
        output_dim,
        blocks: contractions,
    })
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        target_arch = "x86_64"
    )
))]
fn scalar_attr_tensor_product_forward_accumulate_grouped_blas_f32(
    input: &[f32],
    active_type_rows: &ActiveTypeRows,
    contractions: &ScalarAttrContractionsF32,
    out: &mut [f32],
) {
    let input_dim = contractions.input_dim;
    let output_dim = contractions.output_dim;
    let mut packed_input = Vec::new();
    let mut packed_output = Vec::new();
    for contraction in &contractions.blocks {
        let contracted_stride = contraction.mul_in * contraction.mul_out;
        for slot in 0..active_type_rows.active_rows.len() {
            let rows = &active_type_rows.type_rows[active_type_rows.type_row_offsets[slot]
                ..active_type_rows.type_row_offsets[slot + 1]];
            let contracted_base = slot * contracted_stride;
            if rows.len() >= 16 && contraction.mul_in >= 32 && contraction.mul_out >= 32 {
                let contracted =
                    &contraction.values[contracted_base..contracted_base + contracted_stride];
                packed_input.resize(rows.len() * contraction.mul_in, 0.0);
                packed_output.resize(rows.len() * contraction.mul_out, 0.0);
                for m in 0..contraction.ir_dim {
                    for (packed_row, &row) in rows.iter().enumerate() {
                        let input_row = row * input_dim + contraction.input_offset;
                        let packed_row_start = packed_row * contraction.mul_in;
                        for in_mul in 0..contraction.mul_in {
                            packed_input[packed_row_start + in_mul] =
                                input[input_row + in_mul * contraction.ir_dim + m];
                        }
                    }
                    crate::blas_backend::dense_forward_f32(
                        &packed_input,
                        rows.len(),
                        contraction.mul_in,
                        contracted,
                        &mut packed_output,
                        contraction.scale,
                    );
                    for (packed_row, &row) in rows.iter().enumerate() {
                        let output_row = row * output_dim + contraction.output_offset;
                        let packed_row_start = packed_row * contraction.mul_out;
                        for out_mul in 0..contraction.mul_out {
                            out[output_row + out_mul * contraction.ir_dim + m] +=
                                packed_output[packed_row_start + out_mul];
                        }
                    }
                }
            } else {
                for &row in rows {
                    let input_row = row * input_dim + contraction.input_offset;
                    let output_row = row * output_dim + contraction.output_offset;
                    for out_mul in 0..contraction.mul_out {
                        for m in 0..contraction.ir_dim {
                            let mut value = 0.0;
                            for in_mul in 0..contraction.mul_in {
                                let x = input[input_row + in_mul * contraction.ir_dim + m];
                                value += x * contraction.values
                                    [contracted_base + in_mul * contraction.mul_out + out_mul];
                            }
                            out[output_row + out_mul * contraction.ir_dim + m] +=
                                contraction.scale * value;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        target_arch = "x86_64"
    )
))]
fn scalar_attr_tensor_product_vjp_input_accumulate_grouped_blas_f32(
    grad_out: &[f32],
    active_type_rows: &ActiveTypeRows,
    contractions: &ScalarAttrContractionsF32,
    grad_input: &mut [f32],
) {
    let input_dim = contractions.input_dim;
    let output_dim = contractions.output_dim;
    let mut packed_grad_out = Vec::new();
    let mut packed_grad_input = Vec::new();
    for contraction in &contractions.blocks {
        let contracted_stride = contraction.mul_in * contraction.mul_out;
        for slot in 0..active_type_rows.active_rows.len() {
            let rows = &active_type_rows.type_rows[active_type_rows.type_row_offsets[slot]
                ..active_type_rows.type_row_offsets[slot + 1]];
            let contracted_base = slot * contracted_stride;
            if rows.len() >= 16 && contraction.mul_in >= 32 && contraction.mul_out >= 32 {
                let contracted =
                    &contraction.values[contracted_base..contracted_base + contracted_stride];
                packed_grad_out.resize(rows.len() * contraction.mul_out, 0.0);
                packed_grad_input.resize(rows.len() * contraction.mul_in, 0.0);
                for m in 0..contraction.ir_dim {
                    for (packed_row, &row) in rows.iter().enumerate() {
                        let output_row = row * output_dim + contraction.output_offset;
                        let packed_row_start = packed_row * contraction.mul_out;
                        for out_mul in 0..contraction.mul_out {
                            packed_grad_out[packed_row_start + out_mul] =
                                grad_out[output_row + out_mul * contraction.ir_dim + m];
                        }
                    }
                    packed_grad_input.fill(0.0);
                    crate::blas_backend::dense_vjp_input_f32(
                        rows.len(),
                        contraction.mul_in,
                        contracted,
                        &packed_grad_out,
                        &mut packed_grad_input,
                        contraction.scale,
                    );
                    for (packed_row, &row) in rows.iter().enumerate() {
                        let input_row = row * input_dim + contraction.input_offset;
                        let packed_row_start = packed_row * contraction.mul_in;
                        for in_mul in 0..contraction.mul_in {
                            grad_input[input_row + in_mul * contraction.ir_dim + m] +=
                                packed_grad_input[packed_row_start + in_mul];
                        }
                    }
                }
            } else {
                for &row in rows {
                    let input_row = row * input_dim + contraction.input_offset;
                    let output_row = row * output_dim + contraction.output_offset;
                    for out_mul in 0..contraction.mul_out {
                        for m in 0..contraction.ir_dim {
                            let bar_out = grad_out[output_row + out_mul * contraction.ir_dim + m];
                            for in_mul in 0..contraction.mul_in {
                                let input_index = input_row + in_mul * contraction.ir_dim + m;
                                grad_input[input_index] += contraction.scale
                                    * contraction.values
                                        [contracted_base + in_mul * contraction.mul_out + out_mul]
                                    * bar_out;
                            }
                        }
                    }
                }
            }
        }
    }
}

#[allow(dead_code)]
fn scalar_attr_tensor_product_forward_accumulate_f32(
    input: &[f32],
    active_type_rows: &ActiveTypeRows,
    row_count: usize,
    contractions: &ScalarAttrContractionsF32,
    out: &mut [f32],
) -> Result<(), NequipOamPrefixError> {
    let input_dim = contractions.input_dim;
    let output_dim = contractions.output_dim;
    if input.len() != row_count * input_dim
        || out.len() != row_count * output_dim
        || active_type_rows.row_type_slots.len() != row_count
    {
        return Err(invalid_model(
            "scalar-attribute f32 tensor product input shape mismatch",
        ));
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    ))]
    if scalar_attr_grouped_blas_is_available(active_type_rows, &contractions.blocks) {
        scalar_attr_tensor_product_forward_accumulate_grouped_blas_f32(
            input,
            active_type_rows,
            contractions,
            out,
        );
        return Ok(());
    }
    for contraction in &contractions.blocks {
        let contracted_stride = contraction.mul_in * contraction.mul_out;
        for row in 0..row_count {
            let slot = active_type_rows.row_type_slots[row];
            let contracted_base = slot * contracted_stride;
            debug_assert!(contracted_base + contracted_stride <= contraction.values.len());
            let input_row = row * input_dim + contraction.input_offset;
            let output_row = row * output_dim + contraction.output_offset;
            for out_mul in 0..contraction.mul_out {
                for m in 0..contraction.ir_dim {
                    let mut value = 0.0;
                    for in_mul in 0..contraction.mul_in {
                        let x = input[input_row + in_mul * contraction.ir_dim + m];
                        value += x * contraction.values
                            [contracted_base + in_mul * contraction.mul_out + out_mul];
                    }
                    out[output_row + out_mul * contraction.ir_dim + m] += contraction.scale * value;
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn scalar_attr_tensor_product_forward_accumulate_parallel_f32(
    input: &[f32],
    active_type_rows: &ActiveTypeRows,
    row_count: usize,
    contractions: &ScalarAttrContractionsF32,
    out: &mut [f32],
) -> Result<(), NequipOamPrefixError> {
    let input_dim = contractions.input_dim;
    let output_dim = contractions.output_dim;
    if input.len() != row_count * input_dim
        || out.len() != row_count * output_dim
        || active_type_rows.row_type_slots.len() != row_count
    {
        return Err(invalid_model(
            "parallel scalar-attribute f32 tensor product input shape mismatch",
        ));
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    ))]
    if scalar_attr_grouped_blas_is_available(active_type_rows, &contractions.blocks) {
        scalar_attr_tensor_product_forward_accumulate_grouped_blas_f32(
            input,
            active_type_rows,
            contractions,
            out,
        );
        return Ok(());
    }
    for contraction in &contractions.blocks {
        let contracted_stride = contraction.mul_in * contraction.mul_out;
        out.par_chunks_mut(output_dim)
            .enumerate()
            .for_each(|(row, out_row)| {
                let slot = active_type_rows.row_type_slots[row];
                let contracted_base = slot * contracted_stride;
                debug_assert!(contracted_base + contracted_stride <= contraction.values.len());
                let input_row = row * input_dim + contraction.input_offset;
                for out_mul in 0..contraction.mul_out {
                    for m in 0..contraction.ir_dim {
                        let mut value = 0.0;
                        for in_mul in 0..contraction.mul_in {
                            let x = input[input_row + in_mul * contraction.ir_dim + m];
                            value += x * contraction.values
                                [contracted_base + in_mul * contraction.mul_out + out_mul];
                        }
                        out_row[contraction.output_offset + out_mul * contraction.ir_dim + m] +=
                            contraction.scale * value;
                    }
                }
            });
    }
    Ok(())
}

#[allow(dead_code)]
fn scalar_attr_tensor_product_vjp_input_accumulate_f32(
    grad_out: &[f32],
    input: &[f32],
    active_type_rows: &ActiveTypeRows,
    row_count: usize,
    contractions: &ScalarAttrContractionsF32,
    grad_input: &mut [f32],
) -> Result<(), NequipOamPrefixError> {
    let input_dim = contractions.input_dim;
    let output_dim = contractions.output_dim;
    if input.len() != row_count * input_dim
        || grad_out.len() != row_count * output_dim
        || grad_input.len() != row_count * input_dim
        || active_type_rows.row_type_slots.len() != row_count
    {
        return Err(invalid_model(
            "scalar-attribute f32 tensor product VJP input shape mismatch",
        ));
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    ))]
    if scalar_attr_grouped_blas_is_available(active_type_rows, &contractions.blocks) {
        scalar_attr_tensor_product_vjp_input_accumulate_grouped_blas_f32(
            grad_out,
            active_type_rows,
            contractions,
            grad_input,
        );
        return Ok(());
    }
    for contraction in &contractions.blocks {
        let contracted_stride = contraction.mul_in * contraction.mul_out;
        for row in 0..row_count {
            let slot = active_type_rows.row_type_slots[row];
            let contracted_base = slot * contracted_stride;
            debug_assert!(contracted_base + contracted_stride <= contraction.values.len());
            let input_row = row * input_dim + contraction.input_offset;
            let output_row = row * output_dim + contraction.output_offset;
            for out_mul in 0..contraction.mul_out {
                for m in 0..contraction.ir_dim {
                    let bar_out = grad_out[output_row + out_mul * contraction.ir_dim + m];
                    for in_mul in 0..contraction.mul_in {
                        let input_index = input_row + in_mul * contraction.ir_dim + m;
                        grad_input[input_index] += contraction.scale
                            * contraction.values
                                [contracted_base + in_mul * contraction.mul_out + out_mul]
                            * bar_out;
                    }
                }
            }
        }
    }
    Ok(())
}

#[inline(always)]
fn scalar_attr_tensor_product_vjp_input_accumulate_auto_f32(
    grad_out: &[f32],
    input: &[f32],
    active_type_rows: &ActiveTypeRows,
    row_count: usize,
    contractions: &ScalarAttrContractionsF32,
    grad_input: &mut [f32],
) -> Result<(), NequipOamPrefixError> {
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && row_count >= OAM_COMPACT_LINEAR_PARALLEL_MIN_ROWS {
        return scalar_attr_tensor_product_vjp_input_accumulate_parallel_f32(
            grad_out,
            input,
            active_type_rows,
            row_count,
            contractions,
            grad_input,
        );
    }
    scalar_attr_tensor_product_vjp_input_accumulate_f32(
        grad_out,
        input,
        active_type_rows,
        row_count,
        contractions,
        grad_input,
    )
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
#[inline(never)]
fn scalar_attr_tensor_product_vjp_input_accumulate_parallel_f32(
    grad_out: &[f32],
    input: &[f32],
    active_type_rows: &ActiveTypeRows,
    row_count: usize,
    contractions: &ScalarAttrContractionsF32,
    grad_input: &mut [f32],
) -> Result<(), NequipOamPrefixError> {
    let input_dim = contractions.input_dim;
    let output_dim = contractions.output_dim;
    if input.len() != row_count * input_dim
        || grad_out.len() != row_count * output_dim
        || grad_input.len() != row_count * input_dim
        || active_type_rows.row_type_slots.len() != row_count
    {
        return Err(invalid_model(
            "parallel scalar-attribute f32 tensor product VJP input shape mismatch",
        ));
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    ))]
    if scalar_attr_grouped_blas_is_available(active_type_rows, &contractions.blocks) {
        scalar_attr_tensor_product_vjp_input_accumulate_grouped_blas_f32(
            grad_out,
            active_type_rows,
            contractions,
            grad_input,
        );
        return Ok(());
    }
    for contraction in &contractions.blocks {
        let contracted_stride = contraction.mul_in * contraction.mul_out;
        grad_input
            .par_chunks_mut(input_dim)
            .enumerate()
            .for_each(|(row, grad_input_row)| {
                let slot = active_type_rows.row_type_slots[row];
                let contracted_base = slot * contracted_stride;
                debug_assert!(contracted_base + contracted_stride <= contraction.values.len());
                let output_row = row * output_dim + contraction.output_offset;
                for out_mul in 0..contraction.mul_out {
                    for m in 0..contraction.ir_dim {
                        let bar_out = grad_out[output_row + out_mul * contraction.ir_dim + m];
                        for in_mul in 0..contraction.mul_in {
                            grad_input_row
                                [contraction.input_offset + in_mul * contraction.ir_dim + m] +=
                                contraction.scale
                                    * contraction.values
                                        [contracted_base + in_mul * contraction.mul_out + out_mul]
                                    * bar_out;
                        }
                    }
                }
            });
    }
    Ok(())
}

fn active_type_rows(
    type_ids: &[usize],
    type_count: usize,
    row_count: usize,
) -> Result<ActiveTypeRows, NequipOamPrefixError> {
    if type_ids.len() != row_count {
        return Err(invalid_model("scalar-attribute type-id shape mismatch"));
    }
    if type_ids.iter().any(|&type_id| type_id >= type_count) {
        return Err(invalid_model("scalar-attribute type-id out of bounds"));
    }
    let mut active_rows = type_ids.to_vec();
    active_rows.sort_unstable();
    active_rows.dedup();
    let mut row_type_slots = Vec::with_capacity(row_count);
    let mut type_counts = vec![0usize; active_rows.len()];
    for type_id in type_ids {
        let slot = active_rows
            .binary_search(type_id)
            .map_err(|_| invalid_model("scalar-attribute type-id slot missing"))?;
        row_type_slots.push(slot);
        type_counts[slot] += 1;
    }
    let mut type_row_offsets = Vec::with_capacity(active_rows.len() + 1);
    type_row_offsets.push(0);
    let mut total = 0;
    for &count in &type_counts {
        total += count;
        type_row_offsets.push(total);
    }
    let mut type_rows = vec![0usize; row_count];
    let mut cursors = type_row_offsets[..active_rows.len()].to_vec();
    for (row, &slot) in row_type_slots.iter().enumerate() {
        let cursor = &mut cursors[slot];
        type_rows[*cursor] = row;
        *cursor += 1;
    }
    Ok(ActiveTypeRows {
        active_rows,
        row_type_slots,
        type_row_offsets,
        type_rows,
    })
}

fn edge_indices_grouped_by_node(
    edges: &[crate::Edge],
    node_count: usize,
    node_of: impl Fn(&crate::Edge) -> usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut offsets = vec![0usize; node_count];
    for edge in edges {
        let node = node_of(edge);
        debug_assert!(node < node_count);
        offsets[node] += 1;
    }

    let mut grouped_offsets = Vec::with_capacity(node_count + 1);
    grouped_offsets.push(0);
    let mut total = 0;
    for offset in &mut offsets {
        let count = *offset;
        *offset = total;
        total += count;
        grouped_offsets.push(total);
    }

    let mut edge_indices = vec![0usize; edges.len()];
    for (edge_index, edge) in edges.iter().enumerate() {
        let node = node_of(edge);
        let slot = offsets[node];
        edge_indices[slot] = edge_index;
        offsets[node] += 1;
    }
    (edge_indices, grouped_offsets)
}

#[cfg(test)]
fn edge_indices_grouped_by_dst(edges: &[crate::Edge], node_count: usize) -> Vec<usize> {
    edge_indices_grouped_by_node(edges, node_count, |edge| edge.dst).0
}

fn validate_oam_artifact_contract(spec: &Value) -> Result<(), NequipOamPrefixError> {
    let artifact = object_field(spec, "artifact")?;
    validate_oam_str(
        "artifact.sha256",
        str_field(artifact, "sha256")?,
        OAM_ARTIFACT_SHA256,
    )?;
    validate_oam_usize(
        "artifact.size_bytes",
        usize_field(artifact, "size_bytes")?,
        OAM_ARTIFACT_SIZE_BYTES,
    )?;

    let reference = object_field(spec, "reference")?;
    validate_oam_str(
        "reference.name",
        str_field(reference, "name")?,
        OAM_REFERENCE_NAME,
    )?;
    validate_oam_str(
        "reference.repository",
        str_field(reference, "repository")?,
        OAM_REFERENCE_REPOSITORY,
    )?;
    validate_oam_str(
        "reference.commit",
        str_field(reference, "commit")?,
        OAM_REFERENCE_COMMIT,
    )?;
    validate_oam_str(
        "reference.package_version",
        str_field(reference, "package_version")?,
        OAM_REFERENCE_PACKAGE_VERSION,
    )?;
    Ok(())
}

fn validate_oam_prefix_contract(
    prefix: &Value,
    tensors: &HashMap<String, NequipOamPrefixExternalTensor>,
) -> Result<(), NequipOamPrefixError> {
    validate_oam_usize(
        "prefix.num_bessels",
        usize_field(prefix, "num_bessels")?,
        OAM_NUM_BESSELS,
    )?;
    validate_oam_str(
        "prefix.input_irreps",
        str_field(prefix, "input_irreps")?,
        OAM_INPUT_IRREPS,
    )?;
    validate_oam_str(
        "prefix.edge_attr_irreps",
        str_field(prefix, "edge_attr_irreps")?,
        OAM_EDGE_ATTR_IRREPS,
    )?;
    validate_oam_str(
        "prefix.layer0_conv_output_irreps",
        str_field(prefix, "layer0_conv_output_irreps")?,
        OAM_CONV_OUTPUT_IRREPS,
    )?;
    validate_oam_str(
        "prefix.layer0_output_irreps",
        str_field(prefix, "layer0_output_irreps")?,
        OAM_HIDDEN_IRREPS,
    )?;
    validate_oam_f64(
        "prefix.scatter_norm_factor",
        f64_field(prefix, "scatter_norm_factor")?,
        OAM_SCATTER_NORM_FACTOR,
        1.0e-14,
    )?;
    validate_oam_f64(
        "prefix.edge_embedding_factor",
        f64_field(prefix, "edge_embedding_factor")?,
        OAM_EDGE_EMBEDDING_FACTOR,
        1.0e-14,
    )?;
    validate_oam_f64(
        "prefix.polynomial_cutoff_p",
        f64_field(prefix, "polynomial_cutoff_p")?,
        OAM_POLYNOMIAL_CUTOFF_P,
        1.0e-14,
    )?;
    validate_oam_edge_mlp_alphas("prefix.edge_mlp_layer_alphas", prefix)?;
    validate_oam_readout(object_field(prefix, "readout")?)?;
    validate_oam_scale_shift(object_field(prefix, "per_type_energy_scale_shift")?)?;
    validate_oam_pair_potential(object_field(prefix, "pair_potential")?)?;
    validate_oam_tensor_names(prefix, tensors)?;

    let layers = array_field(prefix, "layers")?;
    let prefix_tp = object_field(prefix, "tensor_product")?;
    validate_oam_tensor_product_spec("prefix.tensor_product", prefix_tp, expected_oam_layer(0))?;
    validate_oam_tensor_product(
        "prefix.tensor_product",
        &tensor_product_plan_from_json(prefix_tp)?,
        expected_oam_layer(0),
    )?;
    validate_oam_gate(
        "prefix.gate",
        object_field(prefix, "gate")?,
        expected_oam_layer(0),
    )?;
    for (layer_index, layer) in layers.iter().enumerate() {
        validate_oam_layer(layer_index, layer)?;
    }
    validate_oam_state_shapes(tensors)?;
    Ok(())
}

#[derive(Clone, Copy)]
struct OamExpectedLayer {
    input_irreps: &'static str,
    conv_output_irreps: &'static str,
    output_irreps: &'static str,
    tp_irreps_out: &'static str,
    tp_weight_numel: usize,
    tp_instruction_count: usize,
    tp_cg_count: usize,
    linear1_input_dim: usize,
    linear1_output_dim: usize,
    linear1_weight_numel: usize,
    linear1_block_count: usize,
    linear2_irreps_in: &'static str,
    linear2_input_dim: usize,
    linear2_output_dim: usize,
    linear2_weight_numel: usize,
    linear2_block_count: usize,
    self_connection: Option<OamExpectedSelfConnection>,
    gate_input_dim: usize,
    gate_output_dim: usize,
    gate_group_count: usize,
}

#[derive(Clone, Copy)]
struct OamExpectedSelfConnection {
    output_irreps: &'static str,
    output_dim: usize,
    weight_numel: usize,
    block_count: usize,
}

fn expected_oam_layer(layer: usize) -> OamExpectedLayer {
    match layer {
        0 => OamExpectedLayer {
            input_irreps: OAM_INPUT_IRREPS,
            conv_output_irreps: OAM_CONV_OUTPUT_IRREPS,
            output_irreps: OAM_HIDDEN_IRREPS,
            tp_irreps_out: OAM_LAYER0_TP_OUT_IRREPS,
            tp_weight_numel: 192,
            tp_instruction_count: 192,
            tp_cg_count: 16,
            linear1_input_dim: 48,
            linear1_output_dim: 48,
            linear1_weight_numel: 2_304,
            linear1_block_count: 1,
            linear2_irreps_in: OAM_LAYER0_TP_OUT_IRREPS,
            linear2_input_dim: 768,
            linear2_output_dim: 832,
            linear2_weight_numel: 18_432,
            linear2_block_count: 4,
            self_connection: None,
            gate_input_dim: 832,
            gate_output_dim: 704,
            gate_group_count: 128,
        },
        1..=4 => OamExpectedLayer {
            input_irreps: OAM_HIDDEN_IRREPS,
            conv_output_irreps: OAM_CONV_OUTPUT_IRREPS,
            output_irreps: OAM_HIDDEN_IRREPS,
            tp_irreps_out: OAM_LAYER1_TP_OUT_IRREPS,
            tp_weight_numel: 1_312,
            tp_instruction_count: 1_312,
            tp_cg_count: 353,
            linear1_input_dim: 704,
            linear1_output_dim: 704,
            linear1_weight_numel: 22_528,
            linear1_block_count: 4,
            linear2_irreps_in: OAM_LAYER1_LINEAR2_IN_IRREPS,
            linear2_input_dim: 5_472,
            linear2_output_dim: 832,
            linear2_weight_numel: 110_592,
            linear2_block_count: 4,
            self_connection: Some(OamExpectedSelfConnection {
                output_irreps: OAM_CONV_OUTPUT_IRREPS,
                output_dim: 832,
                weight_numel: 1_867_776,
                block_count: 4,
            }),
            gate_input_dim: 832,
            gate_output_dim: 704,
            gate_group_count: 128,
        },
        5 => OamExpectedLayer {
            input_irreps: OAM_HIDDEN_IRREPS,
            conv_output_irreps: "128x0e",
            output_irreps: "128x0e",
            tp_irreps_out: OAM_LAYER5_TP_OUT_IRREPS,
            tp_weight_numel: 256,
            tp_instruction_count: 256,
            tp_cg_count: 16,
            linear1_input_dim: 704,
            linear1_output_dim: 704,
            linear1_weight_numel: 22_528,
            linear1_block_count: 4,
            linear2_irreps_in: OAM_LAYER5_LINEAR2_IN_IRREPS,
            linear2_input_dim: 256,
            linear2_output_dim: 128,
            linear2_weight_numel: 32_768,
            linear2_block_count: 1,
            self_connection: Some(OamExpectedSelfConnection {
                output_irreps: "128x0e",
                output_dim: 128,
                weight_numel: 786_432,
                block_count: 1,
            }),
            gate_input_dim: 128,
            gate_output_dim: 128,
            gate_group_count: 0,
        },
        _ => unreachable!("OAM-L prefix validation only calls layers 0..6"),
    }
}

fn validate_oam_layer(layer_index: usize, layer: &Value) -> Result<(), NequipOamPrefixError> {
    let expected = expected_oam_layer(layer_index);
    validate_oam_str(
        &format!("layers[{layer_index}].name"),
        str_field(layer, "name")?,
        &format!("layer{layer_index}"),
    )?;
    validate_oam_str(
        &format!("layers[{layer_index}].input_irreps"),
        str_field(layer, "input_irreps")?,
        expected.input_irreps,
    )?;
    validate_oam_str(
        &format!("layers[{layer_index}].conv_output_irreps"),
        str_field(layer, "conv_output_irreps")?,
        expected.conv_output_irreps,
    )?;
    validate_oam_str(
        &format!("layers[{layer_index}].output_irreps"),
        str_field(layer, "output_irreps")?,
        expected.output_irreps,
    )?;
    validate_oam_f64(
        &format!("layers[{layer_index}].scatter_norm_factor"),
        f64_field(layer, "scatter_norm_factor")?,
        OAM_SCATTER_NORM_FACTOR,
        1.0e-14,
    )?;
    validate_oam_edge_mlp_alphas(
        &format!("layers[{layer_index}].edge_mlp_layer_alphas"),
        layer,
    )?;
    validate_oam_linear(
        &format!("layers[{layer_index}].linear1"),
        object_field(layer, "linear1")?,
        expected.input_irreps,
        expected.input_irreps,
        expected.linear1_input_dim,
        expected.linear1_output_dim,
        expected.linear1_weight_numel,
        expected.linear1_block_count,
    )?;
    let tp_spec = object_field(layer, "tensor_product")?;
    validate_oam_tensor_product_spec(
        &format!("layers[{layer_index}].tensor_product"),
        tp_spec,
        expected,
    )?;
    let tp_plan = tensor_product_plan_from_json(tp_spec)?;
    validate_oam_tensor_product(
        &format!("layers[{layer_index}].tensor_product"),
        &tp_plan,
        expected,
    )?;
    validate_oam_usize(
        &format!("layers[{layer_index}].tensor_product.output_dim"),
        tp_plan.output_dim(),
        expected.linear2_input_dim,
    )?;
    validate_oam_linear(
        &format!("layers[{layer_index}].linear2"),
        object_field(layer, "linear2")?,
        expected.linear2_irreps_in,
        expected.conv_output_irreps,
        expected.linear2_input_dim,
        expected.linear2_output_dim,
        expected.linear2_weight_numel,
        expected.linear2_block_count,
    )?;
    validate_oam_self_connection(layer_index, layer.get("self_connection"), expected)?;
    validate_oam_gate(
        &format!("layers[{layer_index}].gate"),
        object_field(layer, "gate")?,
        expected,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_oam_linear(
    field: &str,
    spec: &Value,
    expected_irreps_in: &str,
    expected_irreps_out: &str,
    expected_input_dim: usize,
    expected_output_dim: usize,
    expected_weight_numel: usize,
    expected_block_count: usize,
) -> Result<(), NequipOamPrefixError> {
    validate_oam_str(
        &format!("{field}.kind"),
        str_field(spec, "kind")?,
        "o3-linear-compact-v1",
    )?;
    validate_oam_str(
        &format!("{field}.irreps_in"),
        str_field(spec, "irreps_in")?,
        expected_irreps_in,
    )?;
    validate_oam_str(
        &format!("{field}.irreps_out"),
        str_field(spec, "irreps_out")?,
        expected_irreps_out,
    )?;
    validate_oam_usize(
        &format!("{field}.input_dim"),
        usize_field(spec, "input_dim")?,
        expected_input_dim,
    )?;
    validate_oam_usize(
        &format!("{field}.output_dim"),
        usize_field(spec, "output_dim")?,
        expected_output_dim,
    )?;
    validate_oam_usize(
        &format!("{field}.weight_numel"),
        usize_field(spec, "weight_numel")?,
        expected_weight_numel,
    )?;
    validate_oam_usize(
        &format!("{field}.blocks.len"),
        array_field(spec, "blocks")?.len(),
        expected_block_count,
    )?;
    Ok(())
}

fn validate_oam_tensor_product_spec(
    field: &str,
    spec: &Value,
    expected: OamExpectedLayer,
) -> Result<(), NequipOamPrefixError> {
    validate_oam_str(
        &format!("{field}.irreps_in1"),
        str_field(spec, "irreps_in1")?,
        expected.input_irreps,
    )?;
    validate_oam_str(
        &format!("{field}.irreps_in2"),
        str_field(spec, "irreps_in2")?,
        OAM_EDGE_ATTR_IRREPS,
    )?;
    validate_oam_str(
        &format!("{field}.irreps_out"),
        str_field(spec, "irreps_out")?,
        expected.tp_irreps_out,
    )?;
    validate_oam_usize(
        &format!("{field}.weight_numel"),
        usize_field(spec, "weight_numel")?,
        expected.tp_weight_numel,
    )?;
    validate_oam_usize(
        &format!("{field}.instruction_count"),
        array_field(spec, "instruction_weight_offset")?.len(),
        expected.tp_instruction_count,
    )?;
    validate_oam_usize(
        &format!("{field}.cg_count"),
        array_field(spec, "cg_value")?.len(),
        expected.tp_cg_count,
    )?;
    Ok(())
}

fn validate_oam_tensor_product(
    field: &str,
    plan: &TensorProductPlan,
    expected: OamExpectedLayer,
) -> Result<(), NequipOamPrefixError> {
    validate_oam_irreps(
        &format!("{field}.irreps_in1"),
        &plan.irreps_in1,
        expected.input_irreps,
    )?;
    validate_oam_irreps(
        &format!("{field}.irreps_in2"),
        &plan.irreps_in2,
        OAM_EDGE_ATTR_IRREPS,
    )?;
    validate_oam_irreps(
        &format!("{field}.irreps_out"),
        &plan.irreps_out,
        expected.tp_irreps_out,
    )?;
    validate_oam_usize(
        &format!("{field}.weight_numel"),
        plan.weight_numel,
        expected.tp_weight_numel,
    )?;
    validate_oam_usize(
        &format!("{field}.instruction_count"),
        plan.instructions.len(),
        expected.tp_instruction_count,
    )?;
    validate_oam_usize(
        &format!("{field}.cg_count"),
        plan.cg_entries.len(),
        expected.tp_cg_count,
    )?;
    Ok(())
}

fn validate_oam_self_connection(
    layer_index: usize,
    actual: Option<&Value>,
    expected: OamExpectedLayer,
) -> Result<(), NequipOamPrefixError> {
    match (actual, expected.self_connection) {
        (None, None) => Ok(()),
        (Some(_), None) => Err(invalid_model(format!(
            "layers[{layer_index}].self_connection must be absent for the pinned OAM-L layer"
        ))),
        (None, Some(_)) => Err(invalid_model(format!(
            "layers[{layer_index}].self_connection is required for the pinned OAM-L layer"
        ))),
        (Some(spec), Some(expected_sc)) => {
            let field = format!("layers[{layer_index}].self_connection");
            validate_oam_str(
                &format!("{field}.kind"),
                str_field(spec, "kind")?,
                "scalar-attribute-uvw-compact-v1",
            )?;
            validate_oam_str(
                &format!("{field}.irreps_in1"),
                str_field(spec, "irreps_in1")?,
                expected.input_irreps,
            )?;
            validate_oam_str(
                &format!("{field}.irreps_in2"),
                str_field(spec, "irreps_in2")?,
                OAM_INPUT_IRREPS,
            )?;
            validate_oam_str(
                &format!("{field}.irreps_out"),
                str_field(spec, "irreps_out")?,
                expected_sc.output_irreps,
            )?;
            validate_oam_usize(
                &format!("{field}.input1_dim"),
                usize_field(spec, "input1_dim")?,
                expected.linear1_input_dim,
            )?;
            validate_oam_usize(
                &format!("{field}.input2_dim"),
                usize_field(spec, "input2_dim")?,
                48,
            )?;
            validate_oam_usize(
                &format!("{field}.output_dim"),
                usize_field(spec, "output_dim")?,
                expected_sc.output_dim,
            )?;
            validate_oam_usize(
                &format!("{field}.weight_numel"),
                usize_field(spec, "weight_numel")?,
                expected_sc.weight_numel,
            )?;
            validate_oam_usize(
                &format!("{field}.blocks.len"),
                array_field(spec, "blocks")?.len(),
                expected_sc.block_count,
            )?;
            Ok(())
        }
    }
}

fn validate_oam_gate(
    field: &str,
    gate: &Value,
    expected: OamExpectedLayer,
) -> Result<(), NequipOamPrefixError> {
    validate_oam_usize(
        &format!("{field}.input_dim"),
        usize_field(gate, "input_dim")?,
        expected.gate_input_dim,
    )?;
    validate_oam_usize(
        &format!("{field}.output_dim"),
        usize_field(gate, "output_dim")?,
        expected.gate_output_dim,
    )?;
    validate_oam_str(
        &format!("{field}.scalar_activation"),
        str_field(gate, "scalar_activation")?,
        "silu",
    )?;
    validate_oam_str(
        &format!("{field}.gate_activation"),
        str_field(gate, "gate_activation")?,
        "silu",
    )?;
    validate_oam_f64(
        &format!("{field}.scalar_activation_scale"),
        f64_field(gate, "scalar_activation_scale")?,
        OAM_GATE_SILU_SCALE,
        1.0e-14,
    )?;
    validate_oam_f64(
        &format!("{field}.gate_activation_scale"),
        f64_field(gate, "gate_activation_scale")?,
        OAM_GATE_SILU_SCALE,
        1.0e-14,
    )?;
    validate_oam_usize(
        &format!("{field}.scalar_indices.len"),
        array_field(gate, "scalar_indices")?.len(),
        128,
    )?;
    validate_oam_usize(
        &format!("{field}.gated_groups.len"),
        array_field(gate, "gated_groups")?.len(),
        expected.gate_group_count,
    )?;
    Ok(())
}

fn validate_oam_edge_mlp_alphas(field: &str, spec: &Value) -> Result<(), NequipOamPrefixError> {
    let alphas = f64_array_field(spec, "edge_mlp_layer_alphas")?;
    validate_oam_usize(&format!("{field}.len"), alphas.len(), 2)?;
    validate_oam_f64(
        &format!("{field}[0]"),
        alphas[0],
        OAM_EDGE_MLP_ALPHA0,
        1.0e-14,
    )?;
    validate_oam_f64(
        &format!("{field}[1]"),
        alphas[1],
        OAM_EDGE_MLP_ALPHA1,
        1.0e-14,
    )?;
    Ok(())
}

fn validate_oam_readout(readout: &Value) -> Result<(), NequipOamPrefixError> {
    validate_oam_str(
        "prefix.readout.kind",
        str_field(readout, "kind")?,
        "scalar-linear-v1",
    )?;
    validate_oam_usize(
        "prefix.readout.input_dim",
        usize_field(readout, "input_dim")?,
        128,
    )?;
    validate_oam_usize(
        "prefix.readout.output_dim",
        usize_field(readout, "output_dim")?,
        1,
    )?;
    validate_oam_str(
        "prefix.readout.weight",
        str_field(readout, "weight")?,
        "sole_model.model.func.per_atom_energy_readout.mlp_module.mlp.0.weight",
    )?;
    validate_oam_f64(
        "prefix.readout.alpha",
        f64_field(readout, "alpha")?,
        OAM_READOUT_ALPHA,
        1.0e-14,
    )?;
    Ok(())
}

fn validate_oam_scale_shift(scale_shift: &Value) -> Result<(), NequipOamPrefixError> {
    validate_oam_str(
        "prefix.per_type_energy_scale_shift.scales",
        str_field(scale_shift, "scales")?,
        "sole_model.model.func.per_type_energy_scale_shift.scales",
    )?;
    validate_oam_str(
        "prefix.per_type_energy_scale_shift.shifts",
        str_field(scale_shift, "shifts")?,
        "sole_model.model.func.per_type_energy_scale_shift.shifts",
    )?;
    Ok(())
}

fn validate_oam_pair_potential(pair: &Value) -> Result<(), NequipOamPrefixError> {
    validate_oam_str(
        "prefix.pair_potential.kind",
        str_field(pair, "kind")?,
        "nequip-zbl-v1",
    )?;
    validate_oam_str(
        "prefix.pair_potential.atomic_numbers",
        str_field(pair, "atomic_numbers")?,
        "sole_model.model.func.pair_potential.atomic_numbers",
    )?;
    validate_oam_str(
        "prefix.pair_potential.qqr2exesquare",
        str_field(pair, "qqr2exesquare")?,
        "sole_model.model.func.pair_potential._qqr2exesquare",
    )?;
    validate_oam_bool(
        "prefix.pair_potential.uses_edge_cutoff_key",
        bool_field(pair, "uses_edge_cutoff_key")?,
        true,
    )?;
    validate_oam_str(
        "prefix.pair_potential.edge_center",
        str_field(pair, "edge_center")?,
        "edge_index[0]",
    )?;
    let constants = object_field(pair, "constants")?;
    validate_oam_f64(
        "prefix.pair_potential.constants.pzbl",
        f64_field(constants, "pzbl")?,
        0.23,
        1.0e-14,
    )?;
    validate_oam_f64(
        "prefix.pair_potential.constants.a0",
        f64_field(constants, "a0")?,
        0.4685,
        1.0e-14,
    )?;
    validate_oam_f64_array(
        "prefix.pair_potential.constants.c",
        &f64_array_field(constants, "c")?,
        &[0.02817, 0.28022, 0.50986, 0.18175],
        1.0e-14,
    )?;
    validate_oam_f64_array(
        "prefix.pair_potential.constants.d",
        &f64_array_field(constants, "d")?,
        &[-0.20162, -0.40290, -0.94229, -3.19980],
        1.0e-14,
    )?;
    Ok(())
}

fn validate_oam_tensor_names(
    prefix: &Value,
    tensors: &HashMap<String, NequipOamPrefixExternalTensor>,
) -> Result<(), NequipOamPrefixError> {
    let expected_names = expected_oam_tensor_names();
    let tensor_names = str_array_field(prefix, "tensor_names")?;
    if tensor_names != expected_names {
        return Err(invalid_model(
            "prefix.tensor_names must match the pinned NeQUIP-OAM-L state tensor order",
        ));
    }
    let empty_tensor_names = str_array_field(prefix, "empty_tensor_names")?;
    if empty_tensor_names.as_slice() != [OAM_EMPTY_OUTPUT_MASK] {
        return Err(invalid_model(
            "prefix.empty_tensor_names must match the pinned NeQUIP-OAM-L empty state tensors",
        ));
    }
    let expected_non_empty = expected_names
        .iter()
        .filter(|name| name.as_str() != OAM_EMPTY_OUTPUT_MASK)
        .map(String::as_str)
        .collect::<HashSet<_>>();
    validate_oam_usize(
        "external_state_tensor_count",
        tensors.len(),
        expected_non_empty.len(),
    )?;
    for name in &expected_non_empty {
        if !tensors.contains_key(*name) {
            return Err(NequipOamPrefixError::MissingStateTensor((*name).to_owned()));
        }
    }
    for name in tensors.keys() {
        if !expected_non_empty.contains(name.as_str()) {
            return Err(invalid_model(format!(
                "unexpected external NeQUIP-OAM-L state tensor {name:?}"
            )));
        }
    }
    Ok(())
}

fn expected_oam_tensor_names() -> Vec<String> {
    let mut names = vec![
        "sole_model.model.func.edge_norm._rmax_recip".to_owned(),
        "sole_model.model.func.bessel_encode.bessel_weights".to_owned(),
        "sole_model.model.func.type_embed.embed_module.weight".to_owned(),
        "sole_model.model.func.per_atom_energy_readout.mlp_module.mlp.0.weight".to_owned(),
        "sole_model.model.func.per_type_energy_scale_shift.scales".to_owned(),
        "sole_model.model.func.per_type_energy_scale_shift.shifts".to_owned(),
        "sole_model.model.func.pair_potential.atomic_numbers".to_owned(),
        "sole_model.model.func.pair_potential._qqr2exesquare".to_owned(),
    ];
    for layer_index in 0..OAM_LAYER_COUNT {
        let prefix = format!("sole_model.model.func.layer{layer_index}_convnet");
        names.extend([
            format!("{prefix}.equivariant_nonlin.mul.output_mask"),
            format!("{prefix}.conv.linear_1.weight"),
            format!("{prefix}.conv.linear_1.output_mask"),
            format!("{prefix}.conv.tp_scatter.tp.output_mask"),
            format!("{prefix}.conv.edge_mlp.mlp.0.weight"),
            format!("{prefix}.conv.edge_mlp.mlp.2.weight"),
            format!("{prefix}.conv.linear_2.weight"),
            format!("{prefix}.conv.linear_2.output_mask"),
        ]);
        if layer_index > 0 {
            names.extend([
                format!("{prefix}.conv.sc.weight"),
                format!("{prefix}.conv.sc.output_mask"),
            ]);
        }
    }
    names
}

fn validate_oam_state_shapes(
    tensors: &HashMap<String, NequipOamPrefixExternalTensor>,
) -> Result<(), NequipOamPrefixError> {
    validate_oam_state_shape(tensors, "sole_model.model.func.edge_norm._rmax_recip", &[])?;
    validate_oam_state_shape(
        tensors,
        "sole_model.model.func.bessel_encode.bessel_weights",
        &[1, 8],
    )?;
    validate_oam_state_shape(
        tensors,
        "sole_model.model.func.type_embed.embed_module.weight",
        &[PINNED_OAM_L_TYPE_NAMES.len(), 48],
    )?;
    validate_oam_state_shape(
        tensors,
        "sole_model.model.func.per_atom_energy_readout.mlp_module.mlp.0.weight",
        &[128, 1],
    )?;
    validate_oam_state_shape(
        tensors,
        "sole_model.model.func.per_type_energy_scale_shift.scales",
        &[PINNED_OAM_L_TYPE_NAMES.len(), 1],
    )?;
    validate_oam_state_shape(
        tensors,
        "sole_model.model.func.per_type_energy_scale_shift.shifts",
        &[PINNED_OAM_L_TYPE_NAMES.len(), 1],
    )?;
    validate_oam_state_shape(
        tensors,
        "sole_model.model.func.pair_potential.atomic_numbers",
        &[PINNED_OAM_L_TYPE_NAMES.len()],
    )?;
    validate_oam_state_shape(
        tensors,
        "sole_model.model.func.pair_potential._qqr2exesquare",
        &[],
    )?;

    for layer_index in 0..OAM_LAYER_COUNT {
        let prefix = format!("sole_model.model.func.layer{layer_index}_convnet");
        if layer_index != 5 {
            validate_oam_state_shape(
                tensors,
                &format!("{prefix}.equivariant_nonlin.mul.output_mask"),
                &[576],
            )?;
        }
        let expected = expected_oam_layer(layer_index);
        validate_oam_state_shape(
            tensors,
            &format!("{prefix}.conv.linear_1.weight"),
            &[expected.linear1_weight_numel],
        )?;
        validate_oam_state_shape(
            tensors,
            &format!("{prefix}.conv.linear_1.output_mask"),
            &[expected.linear1_output_dim],
        )?;
        validate_oam_state_shape(
            tensors,
            &format!("{prefix}.conv.tp_scatter.tp.output_mask"),
            &[expected.linear2_input_dim],
        )?;
        validate_oam_state_shape(
            tensors,
            &format!("{prefix}.conv.edge_mlp.mlp.0.weight"),
            &[OAM_NUM_BESSELS, 128],
        )?;
        validate_oam_state_shape(
            tensors,
            &format!("{prefix}.conv.edge_mlp.mlp.2.weight"),
            &[128, expected.tp_weight_numel],
        )?;
        validate_oam_state_shape(
            tensors,
            &format!("{prefix}.conv.linear_2.weight"),
            &[expected.linear2_weight_numel],
        )?;
        validate_oam_state_shape(
            tensors,
            &format!("{prefix}.conv.linear_2.output_mask"),
            &[expected.linear2_output_dim],
        )?;
        if let Some(self_connection) = expected.self_connection {
            validate_oam_state_shape(
                tensors,
                &format!("{prefix}.conv.sc.weight"),
                &[self_connection.weight_numel],
            )?;
            validate_oam_state_shape(
                tensors,
                &format!("{prefix}.conv.sc.output_mask"),
                &[self_connection.output_dim],
            )?;
        }
    }
    Ok(())
}

fn validate_oam_state_shape(
    tensors: &HashMap<String, NequipOamPrefixExternalTensor>,
    name: &str,
    expected: &[usize],
) -> Result<(), NequipOamPrefixError> {
    let tensor = tensors
        .get(name)
        .ok_or_else(|| NequipOamPrefixError::MissingStateTensor(name.to_owned()))?;
    let expected_numel = if expected.is_empty() {
        1
    } else {
        expected.iter().copied().product::<usize>()
    };
    if tensor.shape.as_slice() != expected || tensor.values.len() != expected_numel {
        return Err(NequipOamPrefixError::StateTensorShape {
            name: name.to_owned(),
            actual: tensor.shape.clone(),
            expected: expected.to_vec(),
        });
    }
    Ok(())
}

fn validate_oam_irreps(
    field: &str,
    actual: &Irreps,
    expected: &str,
) -> Result<(), NequipOamPrefixError> {
    let expected_irreps = expected.parse::<Irreps>()?;
    if actual != &expected_irreps {
        return Err(invalid_model(format!(
            "pinned NeQUIP-OAM-L contract requires {field} to be {expected:?}, got {:?}",
            actual.to_string()
        )));
    }
    Ok(())
}

fn validate_oam_usize(
    field: &str,
    actual: usize,
    expected: usize,
) -> Result<(), NequipOamPrefixError> {
    if actual != expected {
        return Err(invalid_model(format!(
            "pinned NeQUIP-OAM-L contract requires {field} to be {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn validate_oam_str(field: &str, actual: &str, expected: &str) -> Result<(), NequipOamPrefixError> {
    if actual != expected {
        return Err(invalid_model(format!(
            "pinned NeQUIP-OAM-L contract requires {field} to be {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

fn validate_oam_bool(
    field: &str,
    actual: bool,
    expected: bool,
) -> Result<(), NequipOamPrefixError> {
    if actual != expected {
        return Err(invalid_model(format!(
            "pinned NeQUIP-OAM-L contract requires {field} to be {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn validate_oam_f64(
    field: &str,
    actual: f64,
    expected: f64,
    tolerance: f64,
) -> Result<(), NequipOamPrefixError> {
    if (actual - expected).abs() > tolerance {
        return Err(invalid_model(format!(
            "pinned NeQUIP-OAM-L contract requires {field} to be {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn validate_oam_f64_array(
    field: &str,
    actual: &[f64],
    expected: &[f64],
    tolerance: f64,
) -> Result<(), NequipOamPrefixError> {
    validate_oam_usize(&format!("{field}.len"), actual.len(), expected.len())?;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        validate_oam_f64(&format!("{field}[{index}]"), *actual, *expected, tolerance)?;
    }
    Ok(())
}

fn tensor_product_plan_from_json(spec: &Value) -> Result<TensorProductPlan, NequipOamPrefixError> {
    let instruction_in1 = usize_array_field(spec, "instruction_in1_segment")?;
    let instruction_in2 = usize_array_field(spec, "instruction_in2_segment")?;
    let instruction_out = usize_array_field(spec, "instruction_out_segment")?;
    let instruction_cg_offset = usize_array_field(spec, "instruction_cg_offset")?;
    let instruction_cg_len = usize_array_field(spec, "instruction_cg_len")?;
    let instruction_weight_offset = usize_array_field(spec, "instruction_weight_offset")?;
    let instruction_path_weight = f64_array_field(spec, "instruction_path_weight")?;
    ensure_same_len(
        "tensor product instruction arrays",
        &[
            ("instruction_in1_segment", instruction_in1.len()),
            ("instruction_in2_segment", instruction_in2.len()),
            ("instruction_out_segment", instruction_out.len()),
            ("instruction_cg_offset", instruction_cg_offset.len()),
            ("instruction_cg_len", instruction_cg_len.len()),
            ("instruction_weight_offset", instruction_weight_offset.len()),
            ("instruction_path_weight", instruction_path_weight.len()),
        ],
    )?;
    let instructions = (0..instruction_in1.len())
        .map(|index| TensorProductInstruction {
            in1_segment: instruction_in1[index],
            in2_segment: instruction_in2[index],
            out_segment: instruction_out[index],
            cg_offset: instruction_cg_offset[index],
            cg_len: instruction_cg_len[index],
            weight_offset: Some(instruction_weight_offset[index]),
            path_weight: instruction_path_weight[index],
        })
        .collect::<Vec<_>>();
    let cg_i_m = usize_array_field(spec, "cg_i_m")?;
    let cg_j_m = usize_array_field(spec, "cg_j_m")?;
    let cg_k_m = usize_array_field(spec, "cg_k_m")?;
    let cg_value = f64_array_field(spec, "cg_value")?;
    ensure_same_len(
        "tensor product CG arrays",
        &[
            ("cg_i_m", cg_i_m.len()),
            ("cg_j_m", cg_j_m.len()),
            ("cg_k_m", cg_k_m.len()),
            ("cg_value", cg_value.len()),
        ],
    )?;
    let cg_entries = (0..cg_value.len())
        .map(|index| CgEntry {
            i_m: cg_i_m[index],
            j_m: cg_j_m[index],
            k_m: cg_k_m[index],
            value: cg_value[index],
        })
        .collect::<Vec<_>>();
    Ok(TensorProductPlan::new(
        str_field(spec, "irreps_in1")?
            .parse::<Irreps>()
            .map_err(|err| invalid_model(format!("invalid TP irreps_in1: {err}")))?,
        str_field(spec, "irreps_in2")?
            .parse::<Irreps>()
            .map_err(|err| invalid_model(format!("invalid TP irreps_in2: {err}")))?,
        str_field(spec, "irreps_out")?
            .parse::<Irreps>()
            .map_err(|err| invalid_model(format!("invalid TP irreps_out: {err}")))?,
        instructions,
        cg_entries,
        usize_field(spec, "weight_numel")?,
    )?)
}

#[cfg(test)]
fn gate_forward(
    input: &[f64],
    row_count: usize,
    gate: &Value,
) -> Result<Vec<f64>, NequipOamPrefixError> {
    let input_dim = usize_field(gate, "input_dim")?;
    let output_dim = usize_field(gate, "output_dim")?;
    let scalar_scale = f64_field(gate, "scalar_activation_scale")?;
    let gate_scale = f64_field(gate, "gate_activation_scale")?;
    let scalar_indices = usize_array_field(gate, "scalar_indices")?;
    let mut out = vec![0.0; row_count * output_dim];
    for row in 0..row_count {
        for (out_index, input_index) in scalar_indices.iter().enumerate() {
            out[row * output_dim + out_index] =
                scalar_scale * silu(input[row * input_dim + input_index]);
        }
        for group in array_field(gate, "gated_groups")? {
            let gate_index = usize_field(group, "gate_index")?;
            let output_start = usize_field(group, "output_start")?;
            let gate_value = gate_scale * silu(input[row * input_dim + gate_index]);
            for (offset, component_index) in usize_array_field(group, "component_indices")?
                .iter()
                .enumerate()
            {
                out[row * output_dim + output_start + offset] =
                    gate_value * input[row * input_dim + component_index];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
fn gate_vjp_input(
    input: &[f64],
    row_count: usize,
    gate: &Value,
    grad_out: &[f64],
) -> Result<Vec<f64>, NequipOamPrefixError> {
    let input_dim = usize_field(gate, "input_dim")?;
    let output_dim = usize_field(gate, "output_dim")?;
    if input.len() != row_count * input_dim || grad_out.len() != row_count * output_dim {
        return Err(invalid_model("gate VJP shape mismatch"));
    }
    let scalar_scale = f64_field(gate, "scalar_activation_scale")?;
    let gate_scale = f64_field(gate, "gate_activation_scale")?;
    let scalar_indices = usize_array_field(gate, "scalar_indices")?;
    let mut grad_input = vec![0.0; row_count * input_dim];
    for row in 0..row_count {
        for (out_index, input_index) in scalar_indices.iter().enumerate() {
            let input_value = input[row * input_dim + input_index];
            grad_input[row * input_dim + input_index] += grad_out[row * output_dim + out_index]
                * scalar_scale
                * silu_derivative(input_value);
        }
        for group in array_field(gate, "gated_groups")? {
            let gate_index = usize_field(group, "gate_index")?;
            let output_start = usize_field(group, "output_start")?;
            let gate_input = input[row * input_dim + gate_index];
            let (gate_silu, gate_derivative) = silu_and_derivative(gate_input);
            let gate_value = gate_scale * gate_silu;
            let mut grad_gate = 0.0;
            for (offset, component_index) in usize_array_field(group, "component_indices")?
                .iter()
                .enumerate()
            {
                let bar_out = grad_out[row * output_dim + output_start + offset];
                let component_value = input[row * input_dim + component_index];
                grad_input[row * input_dim + component_index] += bar_out * gate_value;
                grad_gate += bar_out * component_value;
            }
            grad_input[row * input_dim + gate_index] += grad_gate * gate_scale * gate_derivative;
        }
    }
    Ok(grad_input)
}

fn gate_forward_prepared(
    input: &[f64],
    row_count: usize,
    plan: &OamGatePlan,
) -> Result<(Vec<f64>, OamGateCache), NequipOamPrefixError> {
    if input.len() != row_count * plan.input_dim {
        return Err(invalid_model("prepared gate input shape mismatch"));
    }
    let mut out = vec![0.0; row_count * plan.output_dim];
    let mut scalar_derivatives = vec![0.0; row_count * plan.scalar_indices.len()];
    let mut gate_values = vec![0.0; row_count * plan.groups.len()];
    let mut gate_derivatives = vec![0.0; row_count * plan.groups.len()];
    for row in 0..row_count {
        let input_row = row * plan.input_dim;
        let output_row = row * plan.output_dim;
        let scalar_cache_row = row * plan.scalar_indices.len();
        for (out_index, &input_index) in plan.scalar_indices.iter().enumerate() {
            let (activated, derivative) = silu_and_derivative(input[input_row + input_index]);
            out[output_row + out_index] = plan.scalar_scale * activated;
            scalar_derivatives[scalar_cache_row + out_index] = derivative;
        }
        let gate_cache_row = row * plan.groups.len();
        for (group_index, group) in plan.groups.iter().enumerate() {
            let (gate_silu, gate_derivative) =
                silu_and_derivative(input[input_row + group.gate_index]);
            let gate_value = plan.gate_scale * gate_silu;
            gate_values[gate_cache_row + group_index] = gate_value;
            gate_derivatives[gate_cache_row + group_index] = gate_derivative;
            for (offset, &component_index) in group.component_indices.iter().enumerate() {
                out[output_row + group.output_start + offset] =
                    gate_value * input[input_row + component_index];
            }
        }
    }
    Ok((
        out,
        OamGateCache {
            scalar_derivatives,
            gate_values,
            gate_derivatives,
        },
    ))
}

fn gate_vjp_input_prepared(
    input: &[f64],
    row_count: usize,
    plan: &OamGatePlan,
    cache: &OamGateCache,
    grad_out: &[f64],
) -> Result<Vec<f64>, NequipOamPrefixError> {
    if input.len() != row_count * plan.input_dim
        || grad_out.len() != row_count * plan.output_dim
        || cache.scalar_derivatives.len() != row_count * plan.scalar_indices.len()
        || cache.gate_values.len() != row_count * plan.groups.len()
        || cache.gate_derivatives.len() != row_count * plan.groups.len()
    {
        return Err(invalid_model("prepared gate VJP shape mismatch"));
    }
    let mut grad_input = vec![0.0; row_count * plan.input_dim];
    for row in 0..row_count {
        let input_row = row * plan.input_dim;
        let output_row = row * plan.output_dim;
        let scalar_cache_row = row * plan.scalar_indices.len();
        for (out_index, &input_index) in plan.scalar_indices.iter().enumerate() {
            grad_input[input_row + input_index] += grad_out[output_row + out_index]
                * plan.scalar_scale
                * cache.scalar_derivatives[scalar_cache_row + out_index];
        }
        let gate_cache_row = row * plan.groups.len();
        for (group_index, group) in plan.groups.iter().enumerate() {
            let gate_value = cache.gate_values[gate_cache_row + group_index];
            let gate_derivative = cache.gate_derivatives[gate_cache_row + group_index];
            let mut grad_gate = 0.0;
            for (offset, &component_index) in group.component_indices.iter().enumerate() {
                let bar_out = grad_out[output_row + group.output_start + offset];
                grad_input[input_row + component_index] += bar_out * gate_value;
                grad_gate += bar_out * input[input_row + component_index];
            }
            grad_input[input_row + group.gate_index] +=
                grad_gate * plan.gate_scale * gate_derivative;
        }
    }
    Ok(grad_input)
}

fn gate_forward_plan(
    input: &[f64],
    row_count: usize,
    plan: &OamGatePlan,
) -> Result<Vec<f64>, NequipOamPrefixError> {
    if input.len() != row_count * plan.input_dim {
        return Err(invalid_model("prepared uncached gate input shape mismatch"));
    }
    let mut out = vec![0.0; row_count * plan.output_dim];
    for row in 0..row_count {
        let input_row = row * plan.input_dim;
        let output_row = row * plan.output_dim;
        for (out_index, &input_index) in plan.scalar_indices.iter().enumerate() {
            out[output_row + out_index] = plan.scalar_scale * silu(input[input_row + input_index]);
        }
        for group in &plan.groups {
            let gate_value = plan.gate_scale * silu(input[input_row + group.gate_index]);
            for (offset, &component_index) in group.component_indices.iter().enumerate() {
                out[output_row + group.output_start + offset] =
                    gate_value * input[input_row + component_index];
            }
        }
    }
    Ok(out)
}

fn gate_vjp_input_plan(
    input: &[f64],
    row_count: usize,
    plan: &OamGatePlan,
    grad_out: &[f64],
) -> Result<Vec<f64>, NequipOamPrefixError> {
    if input.len() != row_count * plan.input_dim || grad_out.len() != row_count * plan.output_dim {
        return Err(invalid_model("prepared uncached gate VJP shape mismatch"));
    }
    let mut grad_input = vec![0.0; row_count * plan.input_dim];
    for row in 0..row_count {
        let input_row = row * plan.input_dim;
        let output_row = row * plan.output_dim;
        for (out_index, &input_index) in plan.scalar_indices.iter().enumerate() {
            grad_input[input_row + input_index] += grad_out[output_row + out_index]
                * plan.scalar_scale
                * silu_derivative(input[input_row + input_index]);
        }
        for group in &plan.groups {
            let gate_input = input[input_row + group.gate_index];
            let (gate_silu, gate_derivative) = silu_and_derivative(gate_input);
            let gate_value = plan.gate_scale * gate_silu;
            let mut grad_gate = 0.0;
            for (offset, &component_index) in group.component_indices.iter().enumerate() {
                let bar_out = grad_out[output_row + group.output_start + offset];
                grad_input[input_row + component_index] += bar_out * gate_value;
                grad_gate += bar_out * input[input_row + component_index];
            }
            grad_input[input_row + group.gate_index] +=
                grad_gate * plan.gate_scale * gate_derivative;
        }
    }
    Ok(grad_input)
}

#[allow(dead_code)]
fn gate_forward_f32(
    input: &[f32],
    row_count: usize,
    gate: &Value,
) -> Result<Vec<f32>, NequipOamPrefixError> {
    let input_dim = usize_field(gate, "input_dim")?;
    let output_dim = usize_field(gate, "output_dim")?;
    let scalar_scale = f64_field(gate, "scalar_activation_scale")? as f32;
    let gate_scale = f64_field(gate, "gate_activation_scale")? as f32;
    let scalar_indices = usize_array_field(gate, "scalar_indices")?;
    let mut out = vec![0.0; row_count * output_dim];
    for row in 0..row_count {
        for (out_index, input_index) in scalar_indices.iter().enumerate() {
            out[row * output_dim + out_index] =
                scalar_scale * silu_f32(input[row * input_dim + input_index]);
        }
        for group in array_field(gate, "gated_groups")? {
            let gate_index = usize_field(group, "gate_index")?;
            let output_start = usize_field(group, "output_start")?;
            let gate_value = gate_scale * silu_f32(input[row * input_dim + gate_index]);
            for (offset, component_index) in usize_array_field(group, "component_indices")?
                .iter()
                .enumerate()
            {
                out[row * output_dim + output_start + offset] =
                    gate_value * input[row * input_dim + component_index];
            }
        }
    }
    Ok(out)
}

#[allow(dead_code)]
fn gate_vjp_input_f32(
    input: &[f32],
    row_count: usize,
    gate: &Value,
    grad_out: &[f32],
) -> Result<Vec<f32>, NequipOamPrefixError> {
    let input_dim = usize_field(gate, "input_dim")?;
    let output_dim = usize_field(gate, "output_dim")?;
    if input.len() != row_count * input_dim || grad_out.len() != row_count * output_dim {
        return Err(invalid_model("gate f32 VJP shape mismatch"));
    }
    let scalar_scale = f64_field(gate, "scalar_activation_scale")? as f32;
    let gate_scale = f64_field(gate, "gate_activation_scale")? as f32;
    let scalar_indices = usize_array_field(gate, "scalar_indices")?;
    let mut grad_input = vec![0.0; row_count * input_dim];
    for row in 0..row_count {
        for (out_index, input_index) in scalar_indices.iter().enumerate() {
            let input_value = input[row * input_dim + input_index];
            grad_input[row * input_dim + input_index] += grad_out[row * output_dim + out_index]
                * scalar_scale
                * silu_derivative_f32(input_value);
        }
        for group in array_field(gate, "gated_groups")? {
            let gate_index = usize_field(group, "gate_index")?;
            let output_start = usize_field(group, "output_start")?;
            let gate_input = input[row * input_dim + gate_index];
            let (gate_silu, gate_derivative) = silu_and_derivative_f32(gate_input);
            let gate_value = gate_scale * gate_silu;
            let mut grad_gate = 0.0;
            for (offset, component_index) in usize_array_field(group, "component_indices")?
                .iter()
                .enumerate()
            {
                let bar_out = grad_out[row * output_dim + output_start + offset];
                let component_value = input[row * input_dim + component_index];
                grad_input[row * input_dim + component_index] += bar_out * gate_value;
                grad_gate += bar_out * component_value;
            }
            grad_input[row * input_dim + gate_index] += grad_gate * gate_scale * gate_derivative;
        }
    }
    Ok(grad_input)
}

fn gate_forward_prepared_f32(
    input: &[f32],
    row_count: usize,
    plan: &OamGatePlan,
) -> Result<(Vec<f32>, OamGateCacheF32), NequipOamPrefixError> {
    if input.len() != row_count * plan.input_dim {
        return Err(invalid_model("prepared f32 gate input shape mismatch"));
    }
    let scalar_scale = plan.scalar_scale as f32;
    let gate_scale = plan.gate_scale as f32;
    let mut out = vec![0.0; row_count * plan.output_dim];
    let mut scalar_derivatives = vec![0.0; row_count * plan.scalar_indices.len()];
    let mut gate_values = vec![0.0; row_count * plan.groups.len()];
    let mut gate_derivatives = vec![0.0; row_count * plan.groups.len()];
    for row in 0..row_count {
        let input_row = row * plan.input_dim;
        let output_row = row * plan.output_dim;
        let scalar_cache_row = row * plan.scalar_indices.len();
        for (out_index, &input_index) in plan.scalar_indices.iter().enumerate() {
            let (activated, derivative) = silu_and_derivative_f32(input[input_row + input_index]);
            out[output_row + out_index] = scalar_scale * activated;
            scalar_derivatives[scalar_cache_row + out_index] = derivative;
        }
        let gate_cache_row = row * plan.groups.len();
        for (group_index, group) in plan.groups.iter().enumerate() {
            let (gate_silu, gate_derivative) =
                silu_and_derivative_f32(input[input_row + group.gate_index]);
            let gate_value = gate_scale * gate_silu;
            gate_values[gate_cache_row + group_index] = gate_value;
            gate_derivatives[gate_cache_row + group_index] = gate_derivative;
            for (offset, &component_index) in group.component_indices.iter().enumerate() {
                out[output_row + group.output_start + offset] =
                    gate_value * input[input_row + component_index];
            }
        }
    }
    Ok((
        out,
        OamGateCacheF32 {
            scalar_derivatives,
            gate_values,
            gate_derivatives,
        },
    ))
}

fn gate_vjp_input_prepared_f32(
    input: &[f32],
    row_count: usize,
    plan: &OamGatePlan,
    cache: &OamGateCacheF32,
    grad_out: &[f32],
) -> Result<Vec<f32>, NequipOamPrefixError> {
    if input.len() != row_count * plan.input_dim
        || grad_out.len() != row_count * plan.output_dim
        || cache.scalar_derivatives.len() != row_count * plan.scalar_indices.len()
        || cache.gate_values.len() != row_count * plan.groups.len()
        || cache.gate_derivatives.len() != row_count * plan.groups.len()
    {
        return Err(invalid_model("prepared f32 gate VJP shape mismatch"));
    }
    let scalar_scale = plan.scalar_scale as f32;
    let gate_scale = plan.gate_scale as f32;
    let mut grad_input = vec![0.0; row_count * plan.input_dim];
    for row in 0..row_count {
        let input_row = row * plan.input_dim;
        let output_row = row * plan.output_dim;
        let scalar_cache_row = row * plan.scalar_indices.len();
        for (out_index, &input_index) in plan.scalar_indices.iter().enumerate() {
            grad_input[input_row + input_index] += grad_out[output_row + out_index]
                * scalar_scale
                * cache.scalar_derivatives[scalar_cache_row + out_index];
        }
        let gate_cache_row = row * plan.groups.len();
        for (group_index, group) in plan.groups.iter().enumerate() {
            let gate_value = cache.gate_values[gate_cache_row + group_index];
            let gate_derivative = cache.gate_derivatives[gate_cache_row + group_index];
            let mut grad_gate = 0.0;
            for (offset, &component_index) in group.component_indices.iter().enumerate() {
                let bar_out = grad_out[output_row + group.output_start + offset];
                grad_input[input_row + component_index] += bar_out * gate_value;
                grad_gate += bar_out * input[input_row + component_index];
            }
            grad_input[input_row + group.gate_index] += grad_gate * gate_scale * gate_derivative;
        }
    }
    Ok(grad_input)
}

#[cfg(test)]
fn gate_forward_plan_f32(
    input: &[f32],
    row_count: usize,
    plan: &OamGatePlan,
) -> Result<Vec<f32>, NequipOamPrefixError> {
    if input.len() != row_count * plan.input_dim {
        return Err(invalid_model(
            "prepared uncached f32 gate input shape mismatch",
        ));
    }
    let scalar_scale = plan.scalar_scale as f32;
    let gate_scale = plan.gate_scale as f32;
    let mut out = vec![0.0; row_count * plan.output_dim];
    for row in 0..row_count {
        let input_row = row * plan.input_dim;
        let output_row = row * plan.output_dim;
        for (out_index, &input_index) in plan.scalar_indices.iter().enumerate() {
            out[output_row + out_index] = scalar_scale * silu_f32(input[input_row + input_index]);
        }
        for group in &plan.groups {
            let gate_value = gate_scale * silu_f32(input[input_row + group.gate_index]);
            for (offset, &component_index) in group.component_indices.iter().enumerate() {
                out[output_row + group.output_start + offset] =
                    gate_value * input[input_row + component_index];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
fn gate_vjp_input_plan_f32(
    input: &[f32],
    row_count: usize,
    plan: &OamGatePlan,
    grad_out: &[f32],
) -> Result<Vec<f32>, NequipOamPrefixError> {
    if input.len() != row_count * plan.input_dim || grad_out.len() != row_count * plan.output_dim {
        return Err(invalid_model(
            "prepared uncached f32 gate VJP shape mismatch",
        ));
    }
    let scalar_scale = plan.scalar_scale as f32;
    let gate_scale = plan.gate_scale as f32;
    let mut grad_input = vec![0.0; row_count * plan.input_dim];
    for row in 0..row_count {
        let input_row = row * plan.input_dim;
        let output_row = row * plan.output_dim;
        for (out_index, &input_index) in plan.scalar_indices.iter().enumerate() {
            grad_input[input_row + input_index] += grad_out[output_row + out_index]
                * scalar_scale
                * silu_derivative_f32(input[input_row + input_index]);
        }
        for group in &plan.groups {
            let gate_input = input[input_row + group.gate_index];
            let (gate_silu, gate_derivative) = silu_and_derivative_f32(gate_input);
            let gate_value = gate_scale * gate_silu;
            let mut grad_gate = 0.0;
            for (offset, &component_index) in group.component_indices.iter().enumerate() {
                let bar_out = grad_out[output_row + group.output_start + offset];
                grad_input[input_row + component_index] += bar_out * gate_value;
                grad_gate += bar_out * input[input_row + component_index];
            }
            grad_input[input_row + group.gate_index] += grad_gate * gate_scale * gate_derivative;
        }
    }
    Ok(grad_input)
}

fn input_major_linear(
    input: &[f64],
    row_count: usize,
    in_dim: usize,
    out_dim: usize,
    weight: &[f64],
    alpha: f64,
) -> Result<Vec<f64>, NequipOamPrefixError> {
    if input.len() != row_count * in_dim || weight.len() != in_dim * out_dim {
        return Err(invalid_model("input-major linear shape mismatch"));
    }
    let mut out = vec![0.0; row_count * out_dim];
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    ))]
    #[cfg(feature = "parallel")]
    if row_count >= OAM_INPUT_MAJOR_PARALLEL_BLAS_MIN_ROWS
        && out_dim >= 64
        && rayon::current_num_threads() > 1
    {
        let chunk_count = row_count.clamp(1, OAM_INPUT_MAJOR_PARALLEL_BLAS_TARGET_CHUNKS);
        let chunk_rows = row_count.div_ceil(chunk_count).max(1);
        out.par_chunks_mut(chunk_rows * out_dim)
            .enumerate()
            .for_each(|(chunk_index, out_chunk)| {
                let row_start = chunk_index * chunk_rows;
                let rows = out_chunk.len() / out_dim;
                let input_start = row_start * in_dim;
                crate::blas_backend::dense_forward(
                    &input[input_start..input_start + rows * in_dim],
                    rows,
                    in_dim,
                    weight,
                    out_chunk,
                    alpha,
                );
            });
        return Ok(out);
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    ))]
    if row_count >= 16 && out_dim >= 64 {
        crate::blas_backend::dense_forward(input, row_count, in_dim, weight, &mut out, alpha);
        return Ok(out);
    }
    for row in 0..row_count {
        for out_index in 0..out_dim {
            let mut value = 0.0;
            for in_index in 0..in_dim {
                value += weight[in_index * out_dim + out_index] * input[row * in_dim + in_index];
            }
            out[row * out_dim + out_index] = alpha * value;
        }
    }
    Ok(out)
}

#[allow(dead_code)]
fn input_major_linear_f32(
    input: &[f32],
    row_count: usize,
    in_dim: usize,
    out_dim: usize,
    weight: &[f32],
    alpha: f32,
) -> Result<Vec<f32>, NequipOamPrefixError> {
    if input.len() != row_count * in_dim || weight.len() != in_dim * out_dim {
        return Err(invalid_model("input-major f32 linear shape mismatch"));
    }
    let mut out = vec![0.0; row_count * out_dim];
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    ))]
    if row_count >= 16 && out_dim >= 64 {
        crate::blas_backend::dense_forward_f32(input, row_count, in_dim, weight, &mut out, alpha);
        return Ok(out);
    }
    for row in 0..row_count {
        for out_index in 0..out_dim {
            let mut value = 0.0;
            for in_index in 0..in_dim {
                value += weight[in_index * out_dim + out_index] * input[row * in_dim + in_index];
            }
            out[row * out_dim + out_index] = alpha * value;
        }
    }
    Ok(out)
}

fn input_major_linear_vjp_input(
    grad_out: &[f64],
    row_count: usize,
    in_dim: usize,
    out_dim: usize,
    weight: &[f64],
    alpha: f64,
) -> Result<Vec<f64>, NequipOamPrefixError> {
    if grad_out.len() != row_count * out_dim || weight.len() != in_dim * out_dim {
        return Err(invalid_model("input-major linear VJP shape mismatch"));
    }
    let mut grad_input = vec![0.0; row_count * in_dim];
    input_major_linear_vjp_input_accumulate(
        grad_out,
        row_count,
        in_dim,
        out_dim,
        weight,
        alpha,
        &mut grad_input,
    )?;
    Ok(grad_input)
}

fn input_major_linear_vjp_input_accumulate(
    grad_out: &[f64],
    row_count: usize,
    in_dim: usize,
    out_dim: usize,
    weight: &[f64],
    alpha: f64,
    grad_input: &mut [f64],
) -> Result<(), NequipOamPrefixError> {
    if grad_out.len() != row_count * out_dim
        || weight.len() != in_dim * out_dim
        || grad_input.len() != row_count * in_dim
    {
        return Err(invalid_model("input-major linear VJP shape mismatch"));
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    ))]
    #[cfg(feature = "parallel")]
    if row_count >= OAM_INPUT_MAJOR_PARALLEL_BLAS_MIN_ROWS
        && out_dim >= 64
        && rayon::current_num_threads() > 1
    {
        let chunk_count = row_count.clamp(1, OAM_INPUT_MAJOR_PARALLEL_BLAS_TARGET_CHUNKS);
        let chunk_rows = row_count.div_ceil(chunk_count).max(1);
        grad_input
            .par_chunks_mut(chunk_rows * in_dim)
            .enumerate()
            .for_each(|(chunk_index, grad_input_chunk)| {
                let row_start = chunk_index * chunk_rows;
                let rows = grad_input_chunk.len() / in_dim;
                let grad_out_start = row_start * out_dim;
                crate::blas_backend::dense_vjp_input(
                    rows,
                    in_dim,
                    weight,
                    &grad_out[grad_out_start..grad_out_start + rows * out_dim],
                    grad_input_chunk,
                    alpha,
                );
            });
        return Ok(());
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    ))]
    if row_count >= 16 && out_dim >= 64 {
        crate::blas_backend::dense_vjp_input(
            row_count, in_dim, weight, grad_out, grad_input, alpha,
        );
        return Ok(());
    }
    for row in 0..row_count {
        for out_index in 0..out_dim {
            let bar_out = grad_out[row * out_dim + out_index];
            for in_index in 0..in_dim {
                grad_input[row * in_dim + in_index] +=
                    alpha * weight[in_index * out_dim + out_index] * bar_out;
            }
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn input_major_linear_vjp_input_f32(
    grad_out: &[f32],
    row_count: usize,
    in_dim: usize,
    out_dim: usize,
    weight: &[f32],
    alpha: f32,
) -> Result<Vec<f32>, NequipOamPrefixError> {
    if grad_out.len() != row_count * out_dim || weight.len() != in_dim * out_dim {
        return Err(invalid_model("input-major f32 linear VJP shape mismatch"));
    }
    let mut grad_input = vec![0.0; row_count * in_dim];
    input_major_linear_vjp_input_accumulate_f32(
        grad_out,
        row_count,
        in_dim,
        out_dim,
        weight,
        alpha,
        &mut grad_input,
    )?;
    Ok(grad_input)
}

#[allow(dead_code)]
fn input_major_linear_vjp_input_accumulate_f32(
    grad_out: &[f32],
    row_count: usize,
    in_dim: usize,
    out_dim: usize,
    weight: &[f32],
    alpha: f32,
    grad_input: &mut [f32],
) -> Result<(), NequipOamPrefixError> {
    if grad_out.len() != row_count * out_dim
        || weight.len() != in_dim * out_dim
        || grad_input.len() != row_count * in_dim
    {
        return Err(invalid_model("input-major f32 linear VJP shape mismatch"));
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    ))]
    if row_count >= 16 && out_dim >= 64 {
        crate::blas_backend::dense_vjp_input_f32(
            row_count, in_dim, weight, grad_out, grad_input, alpha,
        );
        return Ok(());
    }
    for row in 0..row_count {
        for out_index in 0..out_dim {
            let bar_out = grad_out[row * out_dim + out_index];
            for in_index in 0..in_dim {
                grad_input[row * in_dim + in_index] +=
                    alpha * weight[in_index * out_dim + out_index] * bar_out;
            }
        }
    }
    Ok(())
}

fn polynomial_cutoff(x: f64, p: f64) -> f64 {
    if x >= 1.0 {
        return 0.0;
    }
    1.0 - ((p + 1.0) * (p + 2.0) / 2.0) * x.powf(p) + p * (p + 2.0) * x.powf(p + 1.0)
        - (p * (p + 1.0) / 2.0) * x.powf(p + 2.0)
}

fn polynomial_cutoff_derivative(x: f64, p: f64) -> f64 {
    if x >= 1.0 {
        return 0.0;
    }
    let a = (p + 1.0) * (p + 2.0) / 2.0;
    let b = p * (p + 2.0);
    let c = p * (p + 1.0) / 2.0;
    -a * p * x.powf(p - 1.0) + b * (p + 1.0) * x.powf(p) - c * (p + 2.0) * x.powf(p + 1.0)
}

fn torch_sinc(x: f64) -> f64 {
    let pix = std::f64::consts::PI * x;
    if pix.abs() < 1.0e-12 {
        1.0
    } else {
        pix.sin() / pix
    }
}

fn torch_sinc_derivative(x: f64) -> f64 {
    if x.abs() < 1.0e-8 {
        return -std::f64::consts::PI * std::f64::consts::PI * x / 3.0;
    }
    let pix = std::f64::consts::PI * x;
    (pix * pix.cos() - pix.sin()) / (std::f64::consts::PI * x * x)
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

#[allow(dead_code)]
fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn silu_and_derivative(x: f64) -> (f64, f64) {
    let sigmoid = 1.0 / (1.0 + (-x).exp());
    let value = x * sigmoid;
    (value, sigmoid + x * sigmoid * (1.0 - sigmoid))
}

#[allow(dead_code)]
fn silu_and_derivative_f32(x: f32) -> (f32, f32) {
    let sigmoid = 1.0 / (1.0 + (-x).exp());
    let value = x * sigmoid;
    (value, sigmoid + x * sigmoid * (1.0 - sigmoid))
}

fn silu_derivative(x: f64) -> f64 {
    let sigmoid = 1.0 / (1.0 + (-x).exp());
    sigmoid + x * sigmoid * (1.0 - sigmoid)
}

#[allow(dead_code)]
fn silu_derivative_f32(x: f32) -> f32 {
    let sigmoid = 1.0 / (1.0 + (-x).exp());
    sigmoid + x * sigmoid * (1.0 - sigmoid)
}

fn factored_embedding_distance_vjp(
    distance: f64,
    rmax_recip: f64,
    bessel_weights: &[f64],
    cutoff_p: f64,
    edge_embedding_factor: f64,
    grad_embedding: &[f64],
) -> f64 {
    let normed = distance * rmax_recip;
    let cutoff = polynomial_cutoff(normed, cutoff_p);
    let cutoff_derivative = polynomial_cutoff_derivative(normed, cutoff_p);
    let mut grad_normed = 0.0;
    for (weight, grad) in bessel_weights.iter().zip(grad_embedding) {
        let arg = normed * *weight;
        grad_normed += grad
            * edge_embedding_factor
            * *weight
            * (torch_sinc_derivative(arg) * *weight * cutoff + torch_sinc(arg) * cutoff_derivative);
    }
    grad_normed * rmax_recip
}

fn factored_embedding_distance_vjp_f32(
    distance: f64,
    rmax_recip: f32,
    bessel_weights: &[f32],
    cutoff_p: f32,
    edge_embedding_factor: f32,
    grad_embedding: &[f32],
) -> f64 {
    let rmax_recip = f64::from(rmax_recip);
    let cutoff_p = f64::from(cutoff_p);
    let edge_embedding_factor = f64::from(edge_embedding_factor);
    let normed = distance * rmax_recip;
    let cutoff = polynomial_cutoff(normed, cutoff_p);
    let cutoff_derivative = polynomial_cutoff_derivative(normed, cutoff_p);
    let mut grad_normed = 0.0;
    for (&weight, &grad) in bessel_weights.iter().zip(grad_embedding) {
        let weight = f64::from(weight);
        let arg = normed * weight;
        grad_normed += f64::from(grad)
            * edge_embedding_factor
            * weight
            * (torch_sinc_derivative(arg) * weight * cutoff + torch_sinc(arg) * cutoff_derivative);
    }
    grad_normed * rmax_recip
}

fn nequip_zbl_edge_energy_and_derivative(
    distance: f64,
    src_type: usize,
    dst_type: usize,
    atomic_numbers: &[f64],
    qqr2exesquare: f64,
) -> (f64, f64) {
    let zi = atomic_numbers[src_type];
    let zj = atomic_numbers[dst_type];
    let dx_ddistance = (zi.powf(0.23) + zj.powf(0.23)) / 0.46850;
    let x = dx_ddistance * distance;
    let coefficients = [0.02817, 0.28022, 0.50986, 0.18175];
    let exponents = [-0.20162, -0.40290, -0.94229, -3.19980];
    let mut psi = 0.0;
    let mut dpsi_dx = 0.0;
    for (coefficient, exponent) in coefficients.iter().zip(exponents) {
        let exp_value = (exponent * x).exp();
        psi += coefficient * exp_value;
        dpsi_dx += coefficient * exponent * exp_value;
    }
    let prefactor = qqr2exesquare * zi * zj;
    let value = prefactor * psi / distance;
    let derivative = prefactor * (dpsi_dx * dx_ddistance / distance - psi / (distance * distance));
    (value, derivative)
}

fn expect_str(value: &Value, key: &str, expected: &str) -> Result<(), NequipOamPrefixError> {
    let actual = str_field(value, key)?;
    if actual == expected {
        Ok(())
    } else {
        Err(invalid_model(format!(
            "{key} must be {expected:?}, got {actual:?}"
        )))
    }
}

fn object_field<'a>(value: &'a Value, key: &str) -> Result<&'a Value, NequipOamPrefixError> {
    value
        .get(key)
        .filter(|item| item.is_object())
        .ok_or_else(|| invalid_model(format!("missing object field {key:?}")))
}

fn array_field<'a>(value: &'a Value, key: &str) -> Result<&'a [Value], NequipOamPrefixError> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid_model(format!("missing array field {key:?}")))
}

fn str_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, NequipOamPrefixError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_model(format!("missing string field {key:?}")))
}

fn bool_field(value: &Value, key: &str) -> Result<bool, NequipOamPrefixError> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| invalid_model(format!("missing bool field {key:?}")))
}

fn f64_field(value: &Value, key: &str) -> Result<f64, NequipOamPrefixError> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid_model(format!("missing f64 field {key:?}")))
}

fn usize_field(value: &Value, key: &str) -> Result<usize, NequipOamPrefixError> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .ok_or_else(|| invalid_model(format!("missing usize field {key:?}")))
}

fn f64_array_field(value: &Value, key: &str) -> Result<Vec<f64>, NequipOamPrefixError> {
    array_field(value, key)?
        .iter()
        .map(|item| {
            item.as_f64()
                .ok_or_else(|| invalid_model(format!("field {key:?} contains non-f64 item")))
        })
        .collect()
}

fn usize_array_field(value: &Value, key: &str) -> Result<Vec<usize>, NequipOamPrefixError> {
    array_field(value, key)?
        .iter()
        .map(|item| {
            item.as_u64()
                .map(|value| value as usize)
                .ok_or_else(|| invalid_model(format!("field {key:?} contains non-usize item")))
        })
        .collect()
}

fn str_array_field(value: &Value, key: &str) -> Result<Vec<String>, NequipOamPrefixError> {
    array_field(value, key)?
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid_model(format!("field {key:?} contains non-string item")))
        })
        .collect()
}

fn ensure_same_len(context: &str, lengths: &[(&str, usize)]) -> Result<(), NequipOamPrefixError> {
    let Some((expected_name, expected_len)) = lengths.first() else {
        return Ok(());
    };
    for (name, len) in &lengths[1..] {
        if len != expected_len {
            return Err(invalid_model(format!(
                "{context} length mismatch: {expected_name} has {expected_len}, {name} has {len}"
            )));
        }
    }
    Ok(())
}

fn validate_pinned_type_names(type_names: &[String]) -> Result<(), NequipOamPrefixError> {
    if type_names.len() != PINNED_OAM_L_TYPE_NAMES.len()
        || type_names
            .iter()
            .zip(PINNED_OAM_L_TYPE_NAMES)
            .any(|(actual, expected)| actual != expected)
    {
        return Err(invalid_model(
            "type_names must match the pinned NeQUIP-OAM-L package order",
        ));
    }
    Ok(())
}

fn invalid_model(message: impl Into<String>) -> NequipOamPrefixError {
    NequipOamPrefixError::InvalidModel(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn type_order_rejection_model_spec() -> String {
        serde_json::json!({
            "schema_version": "nequip-oam-prefix-model-v1",
            "selected_model": "mir-group/NequIP-OAM-L:0.1",
            "state_dict_format": "manifest-f32-f64-tensors-v1",
            "prefix": {
                "l_max": 3,
                "cutoff": 6.0,
                "layers": [{}, {}, {}, {}, {}, {}]
            }
        })
        .to_string()
    }

    fn pinned_type_names() -> Vec<String> {
        PINNED_OAM_L_TYPE_NAMES
            .iter()
            .map(|symbol| (*symbol).to_owned())
            .collect()
    }

    fn tiny_oam_tp_plan_with_singleton_and_stride_run() -> OamTensorProductPlan {
        OamTensorProductPlan {
            input1_dim: 4,
            input2_dim: 2,
            output_dim: 4,
            weight_numel: 3,
            runs: vec![
                OamTensorProductRun {
                    in1_start: 0,
                    in1_step: 0,
                    in2_start: 0,
                    out_start: 0,
                    out_step: 0,
                    cg_offset: 0,
                    cg_len: 1,
                    weight_offset: 0,
                    path_weight: 1.5,
                    len: 1,
                },
                OamTensorProductRun {
                    in1_start: 1,
                    in1_step: 1,
                    in2_start: 0,
                    out_start: 1,
                    out_step: 1,
                    cg_offset: 1,
                    cg_len: 1,
                    weight_offset: 1,
                    path_weight: -0.5,
                    len: 2,
                },
            ],
            tp_layout_runs: Vec::new(),
            input1_original_to_tp: (0..4).collect(),
            output_original_to_tp: (0..4).collect(),
            cg_entries: vec![
                OamScaledCgEntry {
                    i_m: 0,
                    j_m: 1,
                    k_m: 0,
                    scale: 0.375,
                    scale_f32: 0.375,
                },
                OamScaledCgEntry {
                    i_m: 0,
                    j_m: 0,
                    k_m: 0,
                    scale: -1.0,
                    scale_f32: -1.0,
                },
            ],
        }
    }

    fn assert_f32_matches_f64(actual: &[f32], expected: &[f64], tolerance: f64, context: &str) {
        assert_eq!(actual.len(), expected.len(), "{context} length mismatch");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let actual = f64::from(actual);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{context}[{index}] mismatch: actual={actual}, expected={expected}, diff={}",
                (actual - expected).abs()
            );
        }
    }

    fn assert_f64_slices_close(actual: &[f64], expected: &[f64], tolerance: f64, context: &str) {
        assert_eq!(actual.len(), expected.len(), "{context} length mismatch");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{context}[{index}] mismatch: actual={actual}, expected={expected}, diff={}",
                (actual - expected).abs()
            );
        }
    }

    fn input_major_linear_scalar_reference(
        input: &[f64],
        row_count: usize,
        in_dim: usize,
        out_dim: usize,
        weight: &[f64],
        alpha: f64,
    ) -> Vec<f64> {
        let mut out = vec![0.0; row_count * out_dim];
        for row in 0..row_count {
            for out_index in 0..out_dim {
                let mut value = 0.0;
                for in_index in 0..in_dim {
                    value +=
                        weight[in_index * out_dim + out_index] * input[row * in_dim + in_index];
                }
                out[row * out_dim + out_index] = alpha * value;
            }
        }
        out
    }

    fn input_major_linear_vjp_scalar_reference(
        grad_out: &[f64],
        row_count: usize,
        in_dim: usize,
        out_dim: usize,
        weight: &[f64],
        alpha: f64,
    ) -> Vec<f64> {
        let mut grad_input = vec![0.0; row_count * in_dim];
        for row in 0..row_count {
            for out_index in 0..out_dim {
                let bar_out = grad_out[row * out_dim + out_index];
                for in_index in 0..in_dim {
                    grad_input[row * in_dim + in_index] +=
                        alpha * weight[in_index * out_dim + out_index] * bar_out;
                }
            }
        }
        grad_input
    }

    #[test]
    fn edge_indices_grouped_by_dst_preserves_bucket_order() {
        let edge = |src, dst| {
            crate::Edge::new(
                src,
                dst,
                [src as f64, 0.0, 0.0],
                [dst as f64, 1.0, 0.0],
                [0, 0, 0],
                [0.0; 3],
            )
        };
        let edges = vec![edge(0, 2), edge(1, 0), edge(2, 2), edge(0, 1), edge(2, 0)];

        assert_eq!(edge_indices_grouped_by_dst(&edges, 3), vec![1, 4, 3, 0, 2]);
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn oam_input_major_linear_forward_and_vjp_match_scalar_reference() {
        let row_count = 16;
        let in_dim = 8;
        let out_dim = 64;
        let alpha = -0.625;
        let input = (0..row_count * in_dim)
            .map(|index| ((index % 17) as f64 - 8.0) * 0.03125)
            .collect::<Vec<_>>();
        let weight = (0..in_dim * out_dim)
            .map(|index| ((index % 29) as f64 - 14.0) * 0.00625)
            .collect::<Vec<_>>();
        let grad_out = (0..row_count * out_dim)
            .map(|index| ((index % 23) as f64 - 11.0) * 0.0125)
            .collect::<Vec<_>>();

        let out = input_major_linear(&input, row_count, in_dim, out_dim, &weight, alpha)
            .expect("f64 input-major linear should run");
        let expected_out =
            input_major_linear_scalar_reference(&input, row_count, in_dim, out_dim, &weight, alpha);
        assert_f64_slices_close(&out, &expected_out, 1.0e-12, "input-major linear out");

        let grad_input =
            input_major_linear_vjp_input(&grad_out, row_count, in_dim, out_dim, &weight, alpha)
                .expect("f64 input-major linear VJP should run");
        let expected_grad_input = input_major_linear_vjp_scalar_reference(
            &grad_out, row_count, in_dim, out_dim, &weight, alpha,
        );
        assert_f64_slices_close(
            &grad_input,
            &expected_grad_input,
            1.0e-12,
            "input-major linear grad_input",
        );
    }

    #[test]
    fn oam_f32_tensor_product_forward_and_vjp_match_f64_tiny_plan() {
        let plan = tiny_oam_tp_plan_with_singleton_and_stride_run();
        let x = [2.0, 3.0, 5.0, 7.0];
        let y = [11.0, 13.0];
        let w_edge = [17.0, 19.0, 23.0];
        let grad_out = [29.0, 31.0, 37.0, 0.0];

        let mut out = [0.0; 4];
        let mut grad_x = [0.0; 4];
        let mut grad_y = [0.0; 2];
        let mut grad_w_edge = [0.0; 3];
        tensor_product_forward_accumulate_oam(&plan, &x, &y, &w_edge, &mut out);
        tensor_product_vjp_accumulate_oam(
            &plan,
            &x,
            &y,
            &w_edge,
            &grad_out,
            &mut grad_x,
            &mut grad_y,
            &mut grad_w_edge,
        );

        let x_f32 = x.map(|value| value as f32);
        let y_f32 = y.map(|value| value as f32);
        let w_edge_f32 = w_edge.map(|value| value as f32);
        let grad_out_f32 = grad_out.map(|value| value as f32);
        let mut out_f32 = [0.0; 4];
        let mut grad_x_f32 = [0.0; 4];
        let mut grad_y_f32 = [0.0; 2];
        let mut grad_w_edge_f32 = [0.0; 3];
        tensor_product_forward_accumulate_oam_f32(&plan, &x_f32, &y_f32, &w_edge_f32, &mut out_f32);
        tensor_product_vjp_accumulate_oam_f32(
            &plan,
            &x_f32,
            &y_f32,
            &w_edge_f32,
            &grad_out_f32,
            &mut grad_x_f32,
            &mut grad_y_f32,
            &mut grad_w_edge_f32,
        );

        assert_f32_matches_f64(&out_f32, &out, 1.0e-3, "forward out");
        assert_f32_matches_f64(&grad_x_f32, &grad_x, 1.0e-3, "grad_x");
        assert_f32_matches_f64(&grad_y_f32, &grad_y, 1.0e-3, "grad_y");
        assert_f32_matches_f64(&grad_w_edge_f32, &grad_w_edge, 1.0e-3, "grad_w_edge");
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn oam_f32_input_major_linear_forward_and_vjp_match_f64() {
        let row_count = 16;
        let in_dim = 8;
        let out_dim = 64;
        let alpha = -0.75;
        let input = (0..row_count * in_dim)
            .map(|index| ((index % 17) as f64 - 8.0) * 0.03125)
            .collect::<Vec<_>>();
        let weight = (0..in_dim * out_dim)
            .map(|index| ((index % 23) as f64 - 11.0) * 0.0078125)
            .collect::<Vec<_>>();
        let grad_out = (0..row_count * out_dim)
            .map(|index| ((index % 19) as f64 - 9.0) * 0.015625)
            .collect::<Vec<_>>();

        let out = input_major_linear(&input, row_count, in_dim, out_dim, &weight, alpha)
            .expect("f64 input-major linear should run");
        let grad_input =
            input_major_linear_vjp_input(&grad_out, row_count, in_dim, out_dim, &weight, alpha)
                .expect("f64 input-major linear VJP should run");

        let input_f32 = input.iter().map(|&value| value as f32).collect::<Vec<_>>();
        let weight_f32 = weight.iter().map(|&value| value as f32).collect::<Vec<_>>();
        let grad_out_f32 = grad_out
            .iter()
            .map(|&value| value as f32)
            .collect::<Vec<_>>();
        let out_f32 = input_major_linear_f32(
            &input_f32,
            row_count,
            in_dim,
            out_dim,
            &weight_f32,
            alpha as f32,
        )
        .expect("f32 input-major linear should run");
        let grad_input_f32 = input_major_linear_vjp_input_f32(
            &grad_out_f32,
            row_count,
            in_dim,
            out_dim,
            &weight_f32,
            alpha as f32,
        )
        .expect("f32 input-major linear VJP should run");

        assert_f32_matches_f64(&out_f32, &out, 5.0e-5, "input-major linear out");
        assert_f32_matches_f64(
            &grad_input_f32,
            &grad_input,
            5.0e-5,
            "input-major linear grad_input",
        );
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn oam_f32_compact_linear_forward_and_vjp_match_f64() {
        let row_count = 16;
        let scalar_weight_numel = 48 * 48;
        let tensor_weight_numel = 32 * 32;
        let spec = serde_json::json!({
            "input_dim": 112,
            "output_dim": 112,
            "weight_numel": scalar_weight_numel + tensor_weight_numel,
            "blocks": [
                {
                    "mul_in": 48,
                    "mul_out": 48,
                    "ir_dim": 1,
                    "input_offset": 0,
                    "output_offset": 0,
                    "weight_offset": 0,
                    "path_weight": 0.75
                },
                {
                    "mul_in": 32,
                    "mul_out": 32,
                    "ir_dim": 2,
                    "input_offset": 48,
                    "output_offset": 48,
                    "weight_offset": scalar_weight_numel,
                    "path_weight": -0.5
                }
            ]
        });
        let input_dim = 112;
        let output_dim = 112;
        let input = (0..row_count * input_dim)
            .map(|index| ((index % 29) as f64 - 14.0) * 0.00390625)
            .collect::<Vec<_>>();
        let weight = (0..scalar_weight_numel + tensor_weight_numel)
            .map(|index| ((index % 31) as f64 - 15.0) * 0.001953125)
            .collect::<Vec<_>>();
        let grad_out = (0..row_count * output_dim)
            .map(|index| ((index % 37) as f64 - 18.0) * 0.0029296875)
            .collect::<Vec<_>>();

        let out = compact_linear_forward(&input, row_count, &spec, &weight)
            .expect("f64 compact linear should run");
        let grad_input = compact_linear_vjp_input(&grad_out, row_count, &spec, &weight)
            .expect("f64 compact linear VJP should run");

        let input_f32 = input.iter().map(|&value| value as f32).collect::<Vec<_>>();
        let weight_f32 = weight.iter().map(|&value| value as f32).collect::<Vec<_>>();
        let grad_out_f32 = grad_out
            .iter()
            .map(|&value| value as f32)
            .collect::<Vec<_>>();
        let out_f32 = compact_linear_forward_f32(&input_f32, row_count, &spec, &weight_f32)
            .expect("f32 compact linear should run");
        let grad_input_f32 =
            compact_linear_vjp_input_f32(&grad_out_f32, row_count, &spec, &weight_f32)
                .expect("f32 compact linear VJP should run");

        assert_f32_matches_f64(&out_f32, &out, 1.0e-4, "compact linear out");
        assert_f32_matches_f64(
            &grad_input_f32,
            &grad_input,
            1.0e-4,
            "compact linear grad_input",
        );
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn oam_f32_gate_forward_and_vjp_match_f64() {
        let row_count = 5;
        let gate = serde_json::json!({
            "input_dim": 7,
            "output_dim": 5,
            "scalar_activation_scale": 1.25,
            "gate_activation_scale": -0.75,
            "scalar_indices": [0, 2],
            "gated_groups": [
                {
                    "gate_index": 1,
                    "output_start": 2,
                    "component_indices": [3, 4, 5]
                }
            ]
        });
        let input = (0..row_count * 7)
            .map(|index| ((index % 13) as f64 - 6.0) * 0.125)
            .collect::<Vec<_>>();
        let grad_out = (0..row_count * 5)
            .map(|index| ((index % 11) as f64 - 5.0) * 0.0625)
            .collect::<Vec<_>>();

        let out = gate_forward(&input, row_count, &gate).expect("f64 gate should run");
        let grad_input =
            gate_vjp_input(&input, row_count, &gate, &grad_out).expect("f64 gate VJP should run");
        let gate_plan = OamGatePlan::from_json(&gate).expect("prepared gate plan should build");
        let out_plan = gate_forward_plan(&input, row_count, &gate_plan)
            .expect("prepared uncached gate should run");
        let grad_input_plan = gate_vjp_input_plan(&input, row_count, &gate_plan, &grad_out)
            .expect("prepared uncached gate VJP should run");
        let (out_prepared, gate_cache) =
            gate_forward_prepared(&input, row_count, &gate_plan).expect("prepared gate should run");
        let grad_input_prepared =
            gate_vjp_input_prepared(&input, row_count, &gate_plan, &gate_cache, &grad_out)
                .expect("prepared gate VJP should run");
        assert_f64_slices_close(&out_plan, &out, 1.0e-15, "prepared uncached gate out");
        assert_f64_slices_close(
            &grad_input_plan,
            &grad_input,
            1.0e-15,
            "prepared uncached gate grad_input",
        );
        assert_f64_slices_close(&out_prepared, &out, 1.0e-15, "prepared gate out");
        assert_f64_slices_close(
            &grad_input_prepared,
            &grad_input,
            1.0e-15,
            "prepared gate grad_input",
        );

        let input_f32 = input.iter().map(|&value| value as f32).collect::<Vec<_>>();
        let grad_out_f32 = grad_out
            .iter()
            .map(|&value| value as f32)
            .collect::<Vec<_>>();
        let out_f32 = gate_forward_f32(&input_f32, row_count, &gate).expect("f32 gate should run");
        let grad_input_f32 = gate_vjp_input_f32(&input_f32, row_count, &gate, &grad_out_f32)
            .expect("f32 gate VJP should run");
        let (out_f32_prepared, gate_cache_f32) =
            gate_forward_prepared_f32(&input_f32, row_count, &gate_plan)
                .expect("prepared f32 gate should run");
        let out_f32_plan = gate_forward_plan_f32(&input_f32, row_count, &gate_plan)
            .expect("prepared uncached f32 gate should run");
        let grad_input_f32_plan =
            gate_vjp_input_plan_f32(&input_f32, row_count, &gate_plan, &grad_out_f32)
                .expect("prepared uncached f32 gate VJP should run");
        let grad_input_f32_prepared = gate_vjp_input_prepared_f32(
            &input_f32,
            row_count,
            &gate_plan,
            &gate_cache_f32,
            &grad_out_f32,
        )
        .expect("prepared f32 gate VJP should run");
        assert_eq!(out_f32_plan, out_f32);
        assert_eq!(grad_input_f32_plan, grad_input_f32);
        assert_eq!(out_f32_prepared, out_f32);
        assert_eq!(grad_input_f32_prepared, grad_input_f32);

        assert_f32_matches_f64(&out_f32, &out, 1.0e-6, "gate out");
        assert_f32_matches_f64(&grad_input_f32, &grad_input, 1.0e-6, "gate grad_input");
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn oam_f32_scalar_attr_tensor_product_matches_f64() {
        let row_count = 4;
        let type_count = 3;
        let type_ids = [2, 0, 2, 1];
        let active_type_rows = active_type_rows(&type_ids, type_count, row_count)
            .expect("active type rows should build");
        let spec = serde_json::json!({
            "input1_dim": 5,
            "input2_dim": 3,
            "output_dim": 4,
            "weight_numel": 12,
            "blocks": [
                {
                    "mul_in": 2,
                    "attr_mul": 3,
                    "mul_out": 2,
                    "ir_dim": 2,
                    "input_offset": 1,
                    "output_offset": 0,
                    "weight_offset": 0,
                    "path_weight": 0.8,
                    "cg_scalar": -1.25
                }
            ]
        });
        let attrs = (0..type_count * 3)
            .map(|index| ((index % 7) as f64 - 3.0) * 0.0625)
            .collect::<Vec<_>>();
        let weight = (0..12)
            .map(|index| ((index % 5) as f64 - 2.0) * 0.125)
            .collect::<Vec<_>>();
        let input = (0..row_count * 5)
            .map(|index| ((index % 9) as f64 - 4.0) * 0.03125)
            .collect::<Vec<_>>();
        let grad_out = (0..row_count * 4)
            .map(|index| ((index % 11) as f64 - 5.0) * 0.046875)
            .collect::<Vec<_>>();

        let contractions = scalar_attr_tensor_product_contractions(
            &attrs,
            &active_type_rows,
            type_count,
            &spec,
            &weight,
        )
        .expect("f64 scalar attr contractions should build");
        let mut out = vec![0.0; row_count * 4];
        scalar_attr_tensor_product_forward_accumulate(
            &input,
            &active_type_rows,
            row_count,
            &contractions,
            &mut out,
        )
        .expect("f64 scalar attr forward should run");
        let mut grad_input = vec![0.0; row_count * 5];
        scalar_attr_tensor_product_vjp_input_accumulate(
            &grad_out,
            &input,
            &active_type_rows,
            row_count,
            &contractions,
            &mut grad_input,
        )
        .expect("f64 scalar attr VJP should run");

        let attrs_f32 = attrs.iter().map(|&value| value as f32).collect::<Vec<_>>();
        let weight_f32 = weight.iter().map(|&value| value as f32).collect::<Vec<_>>();
        let input_f32 = input.iter().map(|&value| value as f32).collect::<Vec<_>>();
        let grad_out_f32 = grad_out
            .iter()
            .map(|&value| value as f32)
            .collect::<Vec<_>>();
        let contractions_f32 = scalar_attr_tensor_product_contractions_f32(
            &attrs_f32,
            &active_type_rows,
            type_count,
            &spec,
            &weight_f32,
        )
        .expect("f32 scalar attr contractions should build");
        let mut out_f32 = vec![0.0; row_count * 4];
        scalar_attr_tensor_product_forward_accumulate_f32(
            &input_f32,
            &active_type_rows,
            row_count,
            &contractions_f32,
            &mut out_f32,
        )
        .expect("f32 scalar attr forward should run");
        let mut grad_input_f32 = vec![0.0; row_count * 5];
        scalar_attr_tensor_product_vjp_input_accumulate_f32(
            &grad_out_f32,
            &input_f32,
            &active_type_rows,
            row_count,
            &contractions_f32,
            &mut grad_input_f32,
        )
        .expect("f32 scalar attr VJP should run");

        assert_f32_matches_f64(&out_f32, &out, 1.0e-6, "scalar attr out");
        assert_f32_matches_f64(
            &grad_input_f32,
            &grad_input,
            1.0e-6,
            "scalar attr grad_input",
        );
    }

    #[test]
    #[allow(clippy::expect_used)]
    fn oam_scalar_attr_grouped_blas_matches_scalar_reference() {
        let row_count = 18;
        let type_count = 1;
        let type_ids = vec![0; row_count];
        let active_type_rows = active_type_rows(&type_ids, type_count, row_count)
            .expect("active type rows should build");
        let mul_in = 32;
        let attr_mul = 3;
        let mul_out = 32;
        let ir_dim = 2;
        let input_dim = mul_in * ir_dim;
        let output_dim = mul_out * ir_dim;
        let spec = serde_json::json!({
            "input1_dim": input_dim,
            "input2_dim": attr_mul,
            "output_dim": output_dim,
            "weight_numel": mul_in * attr_mul * mul_out,
            "blocks": [
                {
                    "mul_in": mul_in,
                    "attr_mul": attr_mul,
                    "mul_out": mul_out,
                    "ir_dim": ir_dim,
                    "input_offset": 0,
                    "output_offset": 0,
                    "weight_offset": 0,
                    "path_weight": -0.375,
                    "cg_scalar": 1.25
                }
            ]
        });
        let attrs = (0..type_count * attr_mul)
            .map(|index| ((index % 7) as f64 - 3.0) * 0.0625)
            .collect::<Vec<_>>();
        let weight = (0..mul_in * attr_mul * mul_out)
            .map(|index| ((index % 17) as f64 - 8.0) * 0.0078125)
            .collect::<Vec<_>>();
        let input = (0..row_count * input_dim)
            .map(|index| ((index % 23) as f64 - 11.0) * 0.00390625)
            .collect::<Vec<_>>();
        let grad_out = (0..row_count * output_dim)
            .map(|index| ((index % 19) as f64 - 9.0) * 0.005859375)
            .collect::<Vec<_>>();

        let contractions = scalar_attr_tensor_product_contractions(
            &attrs,
            &active_type_rows,
            type_count,
            &spec,
            &weight,
        )
        .expect("f64 scalar attr contractions should build");
        let mut out = vec![0.0; row_count * output_dim];
        scalar_attr_tensor_product_forward_accumulate(
            &input,
            &active_type_rows,
            row_count,
            &contractions,
            &mut out,
        )
        .expect("f64 scalar attr forward should run");
        let mut grad_input = vec![0.0; row_count * input_dim];
        scalar_attr_tensor_product_vjp_input_accumulate(
            &grad_out,
            &input,
            &active_type_rows,
            row_count,
            &contractions,
            &mut grad_input,
        )
        .expect("f64 scalar attr VJP should run");

        let scale = -0.375 * 1.25;
        let mut contracted = vec![0.0; mul_in * mul_out];
        for in_mul in 0..mul_in {
            for out_mul in 0..mul_out {
                let mut value = 0.0;
                for attr in 0..attr_mul {
                    value += weight[in_mul * attr_mul * mul_out + attr * mul_out + out_mul]
                        * attrs[attr];
                }
                contracted[in_mul * mul_out + out_mul] = value;
            }
        }

        let mut expected_out = vec![0.0; out.len()];
        let mut expected_grad_input = vec![0.0; grad_input.len()];
        for row in 0..row_count {
            let input_row = row * input_dim;
            let output_row = row * output_dim;
            for in_mul in 0..mul_in {
                for m in 0..ir_dim {
                    let input_index = input_row + in_mul * ir_dim + m;
                    let x = input[input_index];
                    for out_mul in 0..mul_out {
                        let contracted_value = contracted[in_mul * mul_out + out_mul];
                        let output_index = output_row + out_mul * ir_dim + m;
                        expected_out[output_index] += scale * x * contracted_value;
                        expected_grad_input[input_index] +=
                            scale * contracted_value * grad_out[output_index];
                    }
                }
            }
        }

        assert_f64_slices_close(&out, &expected_out, 2.0e-13, "grouped scalar attr forward");
        assert_f64_slices_close(
            &grad_input,
            &expected_grad_input,
            2.0e-13,
            "grouped scalar attr VJP",
        );
    }

    #[cfg(feature = "parallel")]
    fn tiny_oam_tp_plan() -> OamTensorProductPlan {
        OamTensorProductPlan {
            input1_dim: 1,
            input2_dim: 1,
            output_dim: 1,
            weight_numel: 1,
            runs: vec![OamTensorProductRun {
                in1_start: 0,
                in1_step: 0,
                in2_start: 0,
                out_start: 0,
                out_step: 0,
                cg_offset: 0,
                cg_len: 1,
                weight_offset: 0,
                path_weight: 1.0,
                len: 1,
            }],
            tp_layout_runs: vec![OamTensorProductLayoutRun {
                in1_start: 0,
                in1_step: 0,
                in1_m_step: 1,
                in2_start: 0,
                out_start: 0,
                out_step: 0,
                out_m_step: 1,
                cg_offset: 0,
                cg_len: 1,
                weight_offset: 0,
                len: 1,
            }],
            input1_original_to_tp: vec![0],
            output_original_to_tp: vec![0],
            cg_entries: vec![OamScaledCgEntry {
                i_m: 0,
                j_m: 0,
                k_m: 0,
                scale: 1.0,
                scale_f32: 1.0,
            }],
        }
    }

    #[cfg(feature = "parallel")]
    fn tiny_oam_linear_plan() -> OamCompactLinearPlan {
        OamCompactLinearPlan {
            input_dim: 1,
            output_dim: 1,
            weight_numel: 1,
            blocks: vec![OamCompactLinearBlock {
                mul_in: 1,
                mul_out: 1,
                ir_dim: 1,
                input_offset: 0,
                output_offset: 0,
                weight_offset: 0,
                path_weight: 0.5,
            }],
        }
    }

    #[cfg(feature = "parallel")]
    #[allow(clippy::type_complexity)]
    fn serial_tiny_vjp_scatter(
        plan: &OamTensorProductPlan,
        edges: &[crate::Edge],
        edge_attrs: &[f64],
        normed_linear1: &[f64],
        edge_mlp: &[f64],
        tp_scatter_bars: &[f64],
        num_nodes: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let input1_dim = plan.input1_dim();
        let mut normed_linear1_bars = vec![0.0; num_nodes * input1_dim];
        let mut edge_attr_bars = vec![0.0; edge_attrs.len()];
        let mut edge_mlp_bars = vec![0.0; edge_mlp.len()];
        let mut dst_grad_x = vec![0.0; input1_dim];
        let mut current_dst = None;
        for edge_index in edge_indices_grouped_by_dst(edges, num_nodes) {
            let edge = &edges[edge_index];
            if current_dst != Some(edge.dst) {
                if let Some(dst) = current_dst {
                    let target = &mut normed_linear1_bars[dst * input1_dim..(dst + 1) * input1_dim];
                    for (target, value) in target.iter_mut().zip(&dst_grad_x) {
                        *target += *value;
                    }
                    dst_grad_x.fill(0.0);
                }
                current_dst = Some(edge.dst);
            }
            tensor_product_vjp_accumulate_oam(
                plan,
                &normed_linear1[edge.dst..edge.dst + 1],
                &edge_attrs[edge_index..edge_index + 1],
                &edge_mlp[edge_index..edge_index + 1],
                &tp_scatter_bars[edge.src..edge.src + 1],
                &mut dst_grad_x,
                &mut edge_attr_bars[edge_index..edge_index + 1],
                &mut edge_mlp_bars[edge_index..edge_index + 1],
            );
        }
        if let Some(dst) = current_dst {
            let target = &mut normed_linear1_bars[dst * input1_dim..(dst + 1) * input1_dim];
            for (target, value) in target.iter_mut().zip(&dst_grad_x) {
                *target += *value;
            }
        }
        (normed_linear1_bars, edge_attr_bars, edge_mlp_bars)
    }

    #[cfg(feature = "parallel")]
    #[allow(clippy::type_complexity)]
    fn serial_tiny_vjp_scatter_tp_layout(
        plan: &OamTensorProductPlan,
        edges: &[crate::Edge],
        edge_attrs: &[f64],
        normed_linear1: &[f64],
        edge_mlp: &[f64],
        tp_scatter_bars: &[f64],
        num_nodes: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let input1_dim = plan.input1_dim();
        let sh_dim = plan.input2_dim();
        let output_dim = plan.output_dim();
        let tp_weight_numel = plan.weight_numel;
        let mut normed_linear1_bars = vec![0.0; num_nodes * input1_dim];
        let mut edge_attr_bars = vec![0.0; edge_attrs.len()];
        let mut edge_mlp_bars = vec![0.0; edge_mlp.len()];
        let mut dst_grad_x = vec![0.0; input1_dim];
        let mut current_dst = None;
        for edge_index in edge_indices_grouped_by_dst(edges, num_nodes) {
            let edge = &edges[edge_index];
            if current_dst != Some(edge.dst) {
                if let Some(dst) = current_dst {
                    let target = &mut normed_linear1_bars[dst * input1_dim..(dst + 1) * input1_dim];
                    for (target, value) in target.iter_mut().zip(&dst_grad_x) {
                        *target += *value;
                    }
                    dst_grad_x.fill(0.0);
                }
                current_dst = Some(edge.dst);
            }
            tensor_product_vjp_accumulate_oam_tp_layout(
                plan,
                &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                &tp_scatter_bars[edge.src * output_dim..(edge.src + 1) * output_dim],
                &mut dst_grad_x,
                &mut edge_attr_bars[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                &mut edge_mlp_bars
                    [edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
            );
        }
        if let Some(dst) = current_dst {
            let target = &mut normed_linear1_bars[dst * input1_dim..(dst + 1) * input1_dim];
            for (target, value) in target.iter_mut().zip(&dst_grad_x) {
                *target += *value;
            }
        }
        (normed_linear1_bars, edge_attr_bars, edge_mlp_bars)
    }

    #[cfg(feature = "parallel")]
    #[allow(clippy::type_complexity)]
    fn serial_tiny_vjp_scatter_tp_layout_f32(
        plan: &OamTensorProductPlan,
        edges: &[crate::Edge],
        edge_attrs: &[f32],
        normed_linear1: &[f32],
        edge_mlp: &[f32],
        tp_scatter_bars: &[f32],
        num_nodes: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let input1_dim = plan.input1_dim();
        let sh_dim = plan.input2_dim();
        let output_dim = plan.output_dim();
        let tp_weight_numel = plan.weight_numel;
        let mut normed_linear1_bars = vec![0.0; num_nodes * input1_dim];
        let mut edge_attr_bars = vec![0.0; edge_attrs.len()];
        let mut edge_mlp_bars = vec![0.0; edge_mlp.len()];
        let mut dst_grad_x = vec![0.0; input1_dim];
        let mut current_dst = None;
        for edge_index in edge_indices_grouped_by_dst(edges, num_nodes) {
            let edge = &edges[edge_index];
            if current_dst != Some(edge.dst) {
                if let Some(dst) = current_dst {
                    let target = &mut normed_linear1_bars[dst * input1_dim..(dst + 1) * input1_dim];
                    for (target, value) in target.iter_mut().zip(&dst_grad_x) {
                        *target += *value;
                    }
                    dst_grad_x.fill(0.0);
                }
                current_dst = Some(edge.dst);
            }
            tensor_product_vjp_accumulate_oam_tp_layout_f32(
                plan,
                &normed_linear1[edge.dst * input1_dim..(edge.dst + 1) * input1_dim],
                &edge_attrs[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                &edge_mlp[edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
                &tp_scatter_bars[edge.src * output_dim..(edge.src + 1) * output_dim],
                &mut dst_grad_x,
                &mut edge_attr_bars[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                &mut edge_mlp_bars
                    [edge_index * tp_weight_numel..(edge_index + 1) * tp_weight_numel],
            );
        }
        if let Some(dst) = current_dst {
            let target = &mut normed_linear1_bars[dst * input1_dim..(dst + 1) * input1_dim];
            for (target, value) in target.iter_mut().zip(&dst_grad_x) {
                *target += *value;
            }
        }
        (normed_linear1_bars, edge_attr_bars, edge_mlp_bars)
    }

    #[cfg(feature = "parallel")]
    fn dst_owned_branch_test_edges(edge_count: usize, num_nodes: usize) -> Vec<crate::Edge> {
        (0..edge_count)
            .map(|index| {
                let src = if index % 97 < 11 {
                    3 % num_nodes
                } else {
                    (index / 3) % num_nodes
                };
                let mut dst = if index < edge_count / 2 {
                    2 % num_nodes
                } else if index % 31 < 5 {
                    5 % num_nodes
                } else {
                    (index * 5 + 1) % num_nodes
                };
                if dst == src {
                    dst = (dst + 1) % num_nodes;
                }
                crate::Edge::new(
                    src,
                    dst,
                    [src as f64, (index % 13) as f64 * 0.01, 0.0],
                    [dst as f64 + 1.0, (index % 17) as f64 * 0.02, 0.0],
                    [0, 0, 0],
                    [0.0; 3],
                )
            })
            .collect()
    }

    #[cfg(feature = "parallel")]
    #[test]
    #[allow(clippy::expect_used)]
    fn parallel_oam_vjp_scatter_matches_dst_order_independent_of_thread_count() {
        let plan = tiny_oam_tp_plan();
        let edge = || crate::Edge::new(1, 0, [0.0; 3], [1.0, 0.0, 0.0], [0, 0, 0], [0.0; 3]);
        let edges = vec![edge(), edge(), edge(), edge()];
        let edge_attrs = vec![1.0e16, 1.0, -1.0e16, 1.0];
        let edge_mlp = vec![1.0; edges.len()];
        let normed_linear1 = vec![2.0, 3.0];
        let tp_scatter_bars = vec![0.0, 1.0];
        let edge_plan = OamEdgePlan::new(&edges, 2, true);
        let expected = serial_tiny_vjp_scatter(
            &plan,
            &edges,
            &edge_attrs,
            &normed_linear1,
            &edge_mlp,
            &tp_scatter_bars,
            2,
        );

        for thread_count in [2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .expect("test thread pool should build");
            let mut normed_linear1_bars = vec![0.0; normed_linear1.len()];
            let mut edge_attr_bars = vec![0.0; edge_attrs.len()];
            let mut edge_mlp_bars = vec![0.0; edge_mlp.len()];
            pool.install(|| {
                tensor_product_vjp_scatter_oam_parallel(
                    &plan,
                    &edges,
                    &edge_plan,
                    &edge_attrs,
                    1,
                    &normed_linear1,
                    &edge_mlp,
                    1,
                    &tp_scatter_bars,
                    &mut normed_linear1_bars,
                    &mut edge_attr_bars,
                    &mut edge_mlp_bars,
                    2,
                );
            });
            assert_eq!(
                normed_linear1_bars, expected.0,
                "thread_count={thread_count}"
            );
            assert_eq!(edge_attr_bars, expected.1, "thread_count={thread_count}");
            assert_eq!(edge_mlp_bars, expected.2, "thread_count={thread_count}");
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    #[allow(clippy::expect_used)]
    fn parallel_oam_vjp_direct_large_edge_path_matches_serial() {
        let plan = tiny_oam_tp_plan();
        let edges = (0..(OAM_TP_VJP_PARALLEL_MIN_EDGES + 17))
            .map(|index| {
                let src = index % 2;
                let dst = 1 - src;
                crate::Edge::new(
                    src,
                    dst,
                    [src as f64, 0.0, 0.0],
                    [dst as f64 + 2.0, 0.0, 0.0],
                    [0, 0, 0],
                    [0.0; 3],
                )
            })
            .collect::<Vec<_>>();
        let edge_attrs = (0..edges.len())
            .map(|index| (index % 7) as f64 + 1.0)
            .collect::<Vec<_>>();
        let edge_mlp = vec![1.0; edges.len()];
        let normed_linear1 = vec![2.0, 3.0];
        let tp_scatter_bars = vec![5.0, 7.0];
        let edge_plan = OamEdgePlan::new(&edges, 2, true);
        let expected = serial_tiny_vjp_scatter(
            &plan,
            &edges,
            &edge_attrs,
            &normed_linear1,
            &edge_mlp,
            &tp_scatter_bars,
            2,
        );

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("test thread pool should build");
        let mut normed_linear1_bars = vec![0.0; normed_linear1.len()];
        let mut edge_attr_bars = vec![0.0; edge_attrs.len()];
        let mut edge_mlp_bars = vec![0.0; edge_mlp.len()];
        pool.install(|| {
            tensor_product_vjp_scatter_oam_parallel(
                &plan,
                &edges,
                &edge_plan,
                &edge_attrs,
                1,
                &normed_linear1,
                &edge_mlp,
                1,
                &tp_scatter_bars,
                &mut normed_linear1_bars,
                &mut edge_attr_bars,
                &mut edge_mlp_bars,
                2,
            );
        });

        assert_eq!(normed_linear1_bars, expected.0);
        assert_eq!(edge_attr_bars, expected.1);
        assert_eq!(edge_mlp_bars, expected.2);
    }

    #[cfg(feature = "parallel")]
    #[test]
    #[allow(clippy::expect_used)]
    fn parallel_oam_vjp_direct_fused_edge_mlp2_matches_two_step_path() {
        let plan = tiny_oam_tp_plan();
        let edges = (0..(OAM_TP_VJP_PARALLEL_MIN_EDGES + 17))
            .map(|index| {
                let src = index % 2;
                let dst = 1 - src;
                crate::Edge::new(
                    src,
                    dst,
                    [src as f64, 0.0, 0.0],
                    [dst as f64 + 2.0, 0.0, 0.0],
                    [0, 0, 0],
                    [0.0; 3],
                )
            })
            .collect::<Vec<_>>();
        let edge_attrs = (0..edges.len())
            .map(|index| (index % 5) as f64 + 0.5)
            .collect::<Vec<_>>();
        let edge_mlp = vec![1.0; edges.len()];
        let normed_linear1 = vec![2.0, 3.0];
        let tp_scatter_bars = vec![5.0, 7.0];
        let edge_mlp_2_weight = (0..128)
            .map(|index| 0.25 + index as f64 * 0.001)
            .collect::<Vec<_>>();
        let edge_mlp_2_alpha = 0.125;

        let expected = serial_tiny_vjp_scatter(
            &plan,
            &edges,
            &edge_attrs,
            &normed_linear1,
            &edge_mlp,
            &tp_scatter_bars,
            2,
        );
        let expected_hidden_silu_bars = input_major_linear_vjp_input(
            &expected.2,
            edges.len(),
            128,
            1,
            &edge_mlp_2_weight,
            edge_mlp_2_alpha,
        )
        .expect("two-step edge MLP2 VJP should run");

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("test thread pool should build");
        let mut normed_linear1_bars = vec![0.0; normed_linear1.len()];
        let mut edge_attr_bars = vec![0.0; edge_attrs.len()];
        let mut hidden_silu_bars = vec![0.0; edges.len() * 128];
        pool.install(|| {
            tensor_product_vjp_scatter_oam_parallel_direct_fused_edge_mlp2(
                &plan,
                &edges,
                &edge_attrs,
                1,
                &normed_linear1,
                &edge_mlp,
                1,
                &tp_scatter_bars,
                &edge_mlp_2_weight,
                edge_mlp_2_alpha,
                &mut normed_linear1_bars,
                &mut edge_attr_bars,
                &mut hidden_silu_bars,
                2,
            )
            .expect("fused direct TP VJP should run");
        });

        assert_eq!(normed_linear1_bars, expected.0);
        assert_eq!(edge_attr_bars, expected.1);
        assert_eq!(hidden_silu_bars, expected_hidden_silu_bars);
    }

    #[cfg(feature = "parallel")]
    #[test]
    #[allow(clippy::expect_used)]
    fn parallel_oam_vjp_direct_fused_edge_mlp2_tp_layout_accumulates_edge_attrs_across_layers() {
        let plan = tiny_oam_tp_plan();
        let num_nodes = 7;
        let edge_count = OAM_TP_VJP_PARALLEL_MIN_EDGES + 143;
        assert!(edge_count >= OAM_TP_VJP_PARALLEL_MIN_EDGES);
        assert!(edge_count < OAM_TP_VJP_DST_OWNED_MAX_EDGES);
        let edges = dst_owned_branch_test_edges(edge_count, num_nodes);
        let edge_plan = OamEdgePlan::new(&edges, num_nodes, true);
        let sh_dim = plan.input2_dim();
        let tp_weight_numel = plan.weight_numel;
        let edge_attrs = (0..edge_count)
            .map(|index| ((index % 11) as f64 - 5.0) * 0.125)
            .collect::<Vec<_>>();
        let edge_mlp_first = (0..edge_count)
            .map(|index| 0.75 + (index % 7) as f64 * 0.03125)
            .collect::<Vec<_>>();
        let edge_mlp_second = (0..edge_count)
            .map(|index| 0.5 - (index % 5) as f64 * 0.015625)
            .collect::<Vec<_>>();
        let normed_linear1 = (0..num_nodes)
            .map(|node| 1.0 + node as f64 * 0.25)
            .collect::<Vec<_>>();
        let tp_scatter_bars_first = (0..num_nodes)
            .map(|node| 0.5 + node as f64 * 0.125)
            .collect::<Vec<_>>();
        let tp_scatter_bars_second = (0..num_nodes)
            .map(|node| 1.25 - node as f64 * 0.0625)
            .collect::<Vec<_>>();
        let edge_mlp_2_weight = (0..128)
            .map(|index| 0.25 + index as f64 * 0.001)
            .collect::<Vec<_>>();
        let edge_mlp_2_alpha = 0.125;

        let expected_first = serial_tiny_vjp_scatter_tp_layout(
            &plan,
            &edges,
            &edge_attrs,
            &normed_linear1,
            &edge_mlp_first,
            &tp_scatter_bars_first,
            num_nodes,
        );
        let expected_second = serial_tiny_vjp_scatter_tp_layout(
            &plan,
            &edges,
            &edge_attrs,
            &normed_linear1,
            &edge_mlp_second,
            &tp_scatter_bars_second,
            num_nodes,
        );
        let expected_hidden_first = input_major_linear_vjp_input(
            &expected_first.2,
            edges.len(),
            128,
            tp_weight_numel,
            &edge_mlp_2_weight,
            edge_mlp_2_alpha,
        )
        .expect("first edge MLP2 VJP should run");
        let expected_hidden_second = input_major_linear_vjp_input(
            &expected_second.2,
            edges.len(),
            128,
            tp_weight_numel,
            &edge_mlp_2_weight,
            edge_mlp_2_alpha,
        )
        .expect("second edge MLP2 VJP should run");

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("test thread pool should build");
        let mut normed_linear1_bars_first = vec![0.0; normed_linear1.len()];
        let mut edge_attr_bars = (0..edge_count)
            .map(|index| ((index % 3) as f64 - 1.0) * 0.03125)
            .collect::<Vec<_>>();
        let initial_edge_attr_bars = edge_attr_bars.clone();
        let mut hidden_silu_bars_first = vec![0.0; edges.len() * 128];
        pool.install(|| {
            tensor_product_vjp_scatter_oam_parallel_direct_fused_edge_mlp2_tp_layout(
                &plan,
                &edges,
                &edge_plan,
                &edge_attrs,
                sh_dim,
                &normed_linear1,
                &edge_mlp_first,
                tp_weight_numel,
                &tp_scatter_bars_first,
                &edge_mlp_2_weight,
                edge_mlp_2_alpha,
                &mut normed_linear1_bars_first,
                &mut edge_attr_bars,
                &mut hidden_silu_bars_first,
                num_nodes,
            )
            .expect("first fused TP-layout VJP should run");
        });

        let mut expected_edge_attr_after_first = initial_edge_attr_bars.clone();
        for (target, value) in expected_edge_attr_after_first
            .iter_mut()
            .zip(&expected_first.1)
        {
            *target += *value;
        }
        assert_f64_slices_close(
            &normed_linear1_bars_first,
            &expected_first.0,
            1.0e-12,
            "first normed_linear1_bars",
        );
        assert_f64_slices_close(
            &edge_attr_bars,
            &expected_edge_attr_after_first,
            1.0e-12,
            "first accumulated edge_attr_bars",
        );
        assert_f64_slices_close(
            &hidden_silu_bars_first,
            &expected_hidden_first,
            1.0e-12,
            "first hidden_silu_bars",
        );

        let mut normed_linear1_bars_second = vec![0.0; normed_linear1.len()];
        let mut hidden_silu_bars_second = vec![0.0; edges.len() * 128];
        pool.install(|| {
            tensor_product_vjp_scatter_oam_parallel_direct_fused_edge_mlp2_tp_layout(
                &plan,
                &edges,
                &edge_plan,
                &edge_attrs,
                sh_dim,
                &normed_linear1,
                &edge_mlp_second,
                tp_weight_numel,
                &tp_scatter_bars_second,
                &edge_mlp_2_weight,
                edge_mlp_2_alpha,
                &mut normed_linear1_bars_second,
                &mut edge_attr_bars,
                &mut hidden_silu_bars_second,
                num_nodes,
            )
            .expect("second fused TP-layout VJP should run");
        });

        let mut expected_edge_attr_after_second = expected_edge_attr_after_first;
        for (target, value) in expected_edge_attr_after_second
            .iter_mut()
            .zip(&expected_second.1)
        {
            *target += *value;
        }
        assert_f64_slices_close(
            &normed_linear1_bars_second,
            &expected_second.0,
            1.0e-12,
            "second normed_linear1_bars",
        );
        assert_f64_slices_close(
            &edge_attr_bars,
            &expected_edge_attr_after_second,
            1.0e-12,
            "second accumulated edge_attr_bars",
        );
        assert_f64_slices_close(
            &hidden_silu_bars_second,
            &expected_hidden_second,
            1.0e-12,
            "second hidden_silu_bars",
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    #[allow(clippy::expect_used)]
    fn parallel_oam_vjp_direct_fused_edge_mlp2_tp_layout_f32_accumulates_edge_attrs_across_layers()
    {
        let plan = tiny_oam_tp_plan();
        let num_nodes = 7;
        let edge_count = OAM_TP_VJP_PARALLEL_MIN_EDGES + 143;
        assert!(edge_count >= OAM_TP_VJP_PARALLEL_MIN_EDGES);
        assert!(edge_count < OAM_TP_VJP_DST_OWNED_MAX_EDGES);
        let edges = dst_owned_branch_test_edges(edge_count, num_nodes);
        let edge_plan = OamEdgePlan::new(&edges, num_nodes, true);
        let sh_dim = plan.input2_dim();
        let tp_weight_numel = plan.weight_numel;
        let edge_attrs = (0..edge_count)
            .map(|index| ((index % 11) as f32 - 5.0) * 0.125)
            .collect::<Vec<_>>();
        let edge_mlp_first = (0..edge_count)
            .map(|index| 0.75 + (index % 7) as f32 * 0.03125)
            .collect::<Vec<_>>();
        let edge_mlp_second = (0..edge_count)
            .map(|index| 0.5 - (index % 5) as f32 * 0.015625)
            .collect::<Vec<_>>();
        let normed_linear1 = (0..num_nodes)
            .map(|node| 1.0 + node as f32 * 0.25)
            .collect::<Vec<_>>();
        let tp_scatter_bars_first = (0..num_nodes)
            .map(|node| 0.5 + node as f32 * 0.125)
            .collect::<Vec<_>>();
        let tp_scatter_bars_second = (0..num_nodes)
            .map(|node| 1.25 - node as f32 * 0.0625)
            .collect::<Vec<_>>();
        let edge_mlp_2_weight = (0..128)
            .map(|index| 0.25 + index as f32 * 0.001)
            .collect::<Vec<_>>();
        let edge_mlp_2_alpha = 0.125;

        let expected_first = serial_tiny_vjp_scatter_tp_layout_f32(
            &plan,
            &edges,
            &edge_attrs,
            &normed_linear1,
            &edge_mlp_first,
            &tp_scatter_bars_first,
            num_nodes,
        );
        let expected_second = serial_tiny_vjp_scatter_tp_layout_f32(
            &plan,
            &edges,
            &edge_attrs,
            &normed_linear1,
            &edge_mlp_second,
            &tp_scatter_bars_second,
            num_nodes,
        );
        let expected_hidden_first = input_major_linear_vjp_input_f32(
            &expected_first.2,
            edges.len(),
            128,
            tp_weight_numel,
            &edge_mlp_2_weight,
            edge_mlp_2_alpha,
        )
        .expect("first f32 edge MLP2 VJP should run");
        let expected_hidden_second = input_major_linear_vjp_input_f32(
            &expected_second.2,
            edges.len(),
            128,
            tp_weight_numel,
            &edge_mlp_2_weight,
            edge_mlp_2_alpha,
        )
        .expect("second f32 edge MLP2 VJP should run");

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("test thread pool should build");
        let mut normed_linear1_bars_first = vec![0.0; normed_linear1.len()];
        let mut edge_attr_bars = (0..edge_count)
            .map(|index| ((index % 3) as f32 - 1.0) * 0.03125)
            .collect::<Vec<_>>();
        let initial_edge_attr_bars = edge_attr_bars.clone();
        let mut hidden_silu_bars_first = vec![0.0; edges.len() * 128];
        pool.install(|| {
            tensor_product_vjp_scatter_oam_parallel_direct_fused_edge_mlp2_tp_layout_f32(
                &plan,
                &edges,
                &edge_plan,
                &edge_attrs,
                sh_dim,
                &normed_linear1,
                &edge_mlp_first,
                tp_weight_numel,
                &tp_scatter_bars_first,
                &edge_mlp_2_weight,
                edge_mlp_2_alpha,
                &mut normed_linear1_bars_first,
                &mut edge_attr_bars,
                &mut hidden_silu_bars_first,
                num_nodes,
            )
            .expect("first f32 fused TP-layout VJP should run");
        });

        let mut expected_edge_attr_after_first = initial_edge_attr_bars.clone();
        for (target, value) in expected_edge_attr_after_first
            .iter_mut()
            .zip(&expected_first.1)
        {
            *target += *value;
        }
        assert_eq!(normed_linear1_bars_first, expected_first.0);
        assert_eq!(edge_attr_bars, expected_edge_attr_after_first);
        assert_eq!(hidden_silu_bars_first, expected_hidden_first);

        let mut normed_linear1_bars_second = vec![0.0; normed_linear1.len()];
        let mut hidden_silu_bars_second = vec![0.0; edges.len() * 128];
        pool.install(|| {
            tensor_product_vjp_scatter_oam_parallel_direct_fused_edge_mlp2_tp_layout_f32(
                &plan,
                &edges,
                &edge_plan,
                &edge_attrs,
                sh_dim,
                &normed_linear1,
                &edge_mlp_second,
                tp_weight_numel,
                &tp_scatter_bars_second,
                &edge_mlp_2_weight,
                edge_mlp_2_alpha,
                &mut normed_linear1_bars_second,
                &mut edge_attr_bars,
                &mut hidden_silu_bars_second,
                num_nodes,
            )
            .expect("second f32 fused TP-layout VJP should run");
        });

        let mut expected_edge_attr_after_second = expected_edge_attr_after_first;
        for (target, value) in expected_edge_attr_after_second
            .iter_mut()
            .zip(&expected_second.1)
        {
            *target += *value;
        }
        assert_eq!(normed_linear1_bars_second, expected_second.0);
        assert_eq!(edge_attr_bars, expected_edge_attr_after_second);
        assert_eq!(hidden_silu_bars_second, expected_hidden_second);
    }

    #[cfg(feature = "parallel")]
    #[test]
    #[allow(clippy::expect_used)]
    fn parallel_oam_vjp_direct_fused_linear2_matches_two_step_path() {
        let plan = tiny_oam_tp_plan();
        let linear2_plan = tiny_oam_linear_plan();
        let edges = (0..(OAM_TP_VJP_PARALLEL_MIN_EDGES + 17))
            .map(|index| {
                let src = index % 2;
                let dst = 1 - src;
                crate::Edge::new(
                    src,
                    dst,
                    [src as f64, 0.0, 0.0],
                    [dst as f64 + 2.0, 0.0, 0.0],
                    [0, 0, 0],
                    [0.0; 3],
                )
            })
            .collect::<Vec<_>>();
        let edge_attrs = (0..edges.len())
            .map(|index| (index % 5) as f64 + 0.5)
            .collect::<Vec<_>>();
        let edge_mlp = vec![1.0; edges.len()];
        let normed_linear1 = vec![2.0, 3.0];
        let gate_input_bars = vec![5.0, 7.0];
        let linear2_weight = vec![0.25];
        let edge_mlp_2_weight = (0..128)
            .map(|index| 0.25 + index as f64 * 0.001)
            .collect::<Vec<_>>();
        let edge_mlp_2_alpha = 0.125;
        let mut tp_scatter_bars = vec![0.0; 2];
        for src in 0..2 {
            compact_linear_vjp_input_row_into_oam(
                &linear2_plan,
                &gate_input_bars[src..src + 1],
                &linear2_weight,
                &mut tp_scatter_bars[src..src + 1],
            );
        }

        let expected = serial_tiny_vjp_scatter(
            &plan,
            &edges,
            &edge_attrs,
            &normed_linear1,
            &edge_mlp,
            &tp_scatter_bars,
            2,
        );
        let expected_hidden_silu_bars = input_major_linear_vjp_input(
            &expected.2,
            edges.len(),
            128,
            1,
            &edge_mlp_2_weight,
            edge_mlp_2_alpha,
        )
        .expect("two-step edge MLP2 VJP should run");

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("test thread pool should build");
        let mut normed_linear1_bars = vec![0.0; normed_linear1.len()];
        let mut edge_attr_bars = vec![0.0; edge_attrs.len()];
        let mut hidden_silu_bars = vec![0.0; edges.len() * 128];
        pool.install(|| {
            tensor_product_vjp_scatter_oam_parallel_direct_fused_linear2_edge_mlp2_edge_chunks(
                &plan,
                &edges,
                &edge_attrs,
                1,
                &normed_linear1,
                &edge_mlp,
                1,
                &linear2_plan,
                &gate_input_bars,
                &linear2_weight,
                &edge_mlp_2_weight,
                edge_mlp_2_alpha,
                &mut normed_linear1_bars,
                &mut edge_attr_bars,
                &mut hidden_silu_bars,
                2,
            )
            .expect("fused linear2 TP VJP should run");
        });

        assert_eq!(normed_linear1_bars, expected.0);
        assert_eq!(edge_attr_bars, expected.1);
        assert_eq!(hidden_silu_bars, expected_hidden_silu_bars);
    }

    #[test]
    fn rejects_non_pinned_type_order() {
        let mut type_names = pinned_type_names();
        type_names.swap(0, 1);

        let result = NequipOamPrefixModel::from_json_str_with_external_tensors(
            &type_order_rejection_model_spec(),
            type_names,
            HashMap::new(),
        );
        let err = result.err();
        assert!(err.is_some(), "non-pinned type order should be rejected");
        let Some(err) = err else {
            return;
        };

        assert!(matches!(
            err,
            NequipOamPrefixError::InvalidModel(message)
                if message.contains("type_names must match")
        ));
    }

    #[test]
    fn mismatched_tensor_product_instruction_lengths_return_error() {
        let spec = serde_json::json!({
            "instruction_in1_segment": [0, 1],
            "instruction_in2_segment": [0],
            "instruction_out_segment": [0, 1],
            "instruction_cg_offset": [0, 1],
            "instruction_cg_len": [1, 1],
            "instruction_weight_offset": [0, 1],
            "instruction_path_weight": [1.0, 1.0],
            "cg_i_m": [],
            "cg_j_m": [],
            "cg_k_m": [],
            "cg_value": [],
            "irreps_in1": "1x0e",
            "irreps_in2": "1x0e",
            "irreps_out": "1x0e",
            "weight_numel": 2
        });

        let result = tensor_product_plan_from_json(&spec);
        let err = result.err();
        assert!(
            err.is_some(),
            "mismatched instruction arrays should be rejected"
        );
        let Some(err) = err else {
            return;
        };

        assert!(matches!(
            err,
            NequipOamPrefixError::InvalidModel(message)
                if message.contains("instruction arrays length mismatch")
        ));
    }

    #[test]
    fn mismatched_tensor_product_cg_lengths_return_error() {
        let spec = serde_json::json!({
            "instruction_in1_segment": [],
            "instruction_in2_segment": [],
            "instruction_out_segment": [],
            "instruction_cg_offset": [],
            "instruction_cg_len": [],
            "instruction_weight_offset": [],
            "instruction_path_weight": [],
            "cg_i_m": [0],
            "cg_j_m": [],
            "cg_k_m": [0],
            "cg_value": [1.0],
            "irreps_in1": "1x0e",
            "irreps_in2": "1x0e",
            "irreps_out": "1x0e",
            "weight_numel": 0
        });

        let result = tensor_product_plan_from_json(&spec);
        let err = result.err();
        assert!(err.is_some(), "mismatched CG arrays should be rejected");
        let Some(err) = err else {
            return;
        };

        assert!(matches!(
            err,
            NequipOamPrefixError::InvalidModel(message)
                if message.contains("CG arrays length mismatch")
        ));
    }
}
