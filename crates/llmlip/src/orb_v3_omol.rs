// SPDX-License-Identifier: Apache-2.0
// ORB implementation reference: Orbital Materials, orb-models v0.6.2,
// commit a5f1a382a72126feff0594b19891e03cde5f843e.
// This project conservatively distributes this module under Apache-2.0.
// Modified implementation: Rust fixed-weight buffers, explicit input VJPs,
// fused attention, and CPU parallel/BLAS paths; not upstream Python source.
// See ../../../LICENSE-APACHE, ../../../NOTICE and
// ../../../docs/ORB_LICENSE_SCOPE.md for sources and modification details.
//! State-export gate for the pinned `orb-v3-conservative-omol-v1` runtime contract.

use std::{
    collections::{BTreeMap, HashMap},
    time::Instant,
};

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
use std::sync::Arc;

use e3nn_core::{radial::mace_covalent_radius, sh::SphericalHarmonicsPlan};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::Deserialize;
use thiserror::Error;

use crate::{
    neighbor::{all_pairs_neighbor_list, NeighborListError},
    structure::{Prediction, PredictionComputeDtype, PredictionOptions, Structure},
    Vec3,
};

pub const ARCHITECTURE: &str = "orb-v3-conservative-omol-v1";
pub const SCHEMA_VERSION: &str = "orb-v3-omol-metadata-v1";
pub const RUNTIME_STATUS_METADATA_ONLY: &str = "metadata_only_not_executable";
pub const RUNTIME_STATUS_STATE_EXPORT: &str = "state_export_only";
pub const REFERENCE_PACKAGE: &str = "orb-models";
pub const REFERENCE_TAG: &str = "v0.6.2";
pub const REFERENCE_VERSION: &str = "0.3.2.dev";
pub const REFERENCE_COMMIT: &str = "a5f1a382a72126feff0594b19891e03cde5f843e";
pub const SELECTED_MODEL: &str = "orb-v3-conservative-omol";
pub const CHECKPOINT_FILENAME: &str = "orb-v3-conservative-omol-20250820.ckpt";
pub const CHECKPOINT_URL: &str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3-conservative-omol-20250820.ckpt";
pub const CHECKPOINT_SHA256: &str =
    "c284e99c45df928ae28443fb27223188cc2c33cced593488a4d28595e75cb6e8";
pub const CHECKPOINT_SIZE_BYTES: u64 = 103_417_970;

#[must_use]
pub const fn supports_float32_compute_dtype() -> bool {
    cfg!(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))
}

fn validate_orb_compute_dtype(dtype: PredictionComputeDtype) -> Result<(), OrbV3OmolError> {
    match dtype {
        PredictionComputeDtype::Float64 => Ok(()),
        PredictionComputeDtype::Float32 if supports_float32_compute_dtype() => Ok(()),
        PredictionComputeDtype::Float32 => Err(OrbV3OmolError::UnsupportedComputeDtype(dtype)),
    }
}

pub const CUTOFF_ANGSTROM: f64 = 6.0;
pub const MAX_NUM_NEIGHBORS: usize = 120;
pub const BESSEL_NUM_BASES: usize = 8;
pub const SH_LMAX: usize = 3;
pub const LATENT_DIM: usize = 256;
pub const MESSAGE_PASSING_STEPS: usize = 5;
pub const BASE_MLP_HIDDEN_DIM: usize = 1024;
pub const BASE_MLP_DEPTH: usize = 2;
pub const HEAD_MLP_HIDDEN_DIM: usize = 256;
pub const HEAD_MLP_DEPTH: usize = 1;
const DEFAULT_TOTAL_CHARGE: f64 = 0.0;
const DEFAULT_SPIN_MULTIPLICITY: f64 = 0.0;
const CHARGE_SPIN_MATCH_TOLERANCE: f64 = 1.0e-12;
pub const STATE_EXPORT_INVENTORY_SCHEMA: &str = "orb-v3-omol-state-export-inventory-v1";
pub const STATE_TENSOR_COUNT: usize = 145;
pub const STATE_TOTAL_NUMEL: usize = 25_839_740;
pub const ORB_ANGULAR_DIM: usize = (SH_LMAX + 1) * (SH_LMAX + 1);
pub const ORB_EDGE_FEATURE_DIM: usize = BESSEL_NUM_BASES * ORB_ANGULAR_DIM;
const ORB_REFERENCE_ENERGY_WEIGHT: &str = "heads.energy.reference.linear.weight";
const ORB_REFERENCE_ENERGY_CLASSES: usize = 118;
// OMol25 training coverage is the first 83 elements, H through Bi
// (arXiv:2505.08762: "83 elements"). The checkpoint's embedding/reference
// tables are padded out to the full periodic table, so without this bound the
// runtime would silently extrapolate for untrained species (Z >= 84).
const ORB_MAX_SUPPORTED_ATOMIC_NUMBER: usize = 83;
const ORB_BESSEL_WEIGHTS: &str = "model.rbf_transform.bessel_weights";
const ORB_BESSEL_R_MAX: &str = "model.rbf_transform.r_max";
const ORB_BESSEL_PREFACTOR: &str = "model.rbf_transform.prefactor";
const ORB_ATOM_EMBEDDING_WEIGHT: &str = "model.atom_emb.embeddings.weight";
const ORB_ATOM_EMBEDDING_ROWS: usize = 119;
const ORB_CHARGE_EMBEDDING_W: &str = "model.conditioner.charge_embedding.W";
const ORB_SPIN_EMBEDDING_W: &str = "model.conditioner.spin_embedding.W";
const ORB_CONDITIONING_HALF_DIM: usize = LATENT_DIM / 2;
const ORB_CONDITIONING_FREQUENCIES: usize = ORB_CONDITIONING_HALF_DIM / 2;
const ORB_ENCODER_NODE_PREFIX: &str = "model._encoder._node_fn";
const ORB_ENCODER_EDGE_PREFIX: &str = "model._encoder._edge_fn";
const ORB_ENCODER_HIDDEN_DIM: usize = BASE_MLP_HIDDEN_DIM;
const ORB_GNN_INPUT_DIM: usize = LATENT_DIM * 3;
const ORB_ENERGY_HEAD_PREFIX: &str = "heads.energy";
const ORB_ZBL_C: [f64; 4] = [
    0.1817999929189682,
    0.5098999738693237,
    0.2802000045776367,
    0.02817000076174736,
];
const ORB_ZBL_D: [f64; 4] = [
    3.200000047683716,
    0.942300021648407,
    0.4027999937534332,
    0.20160000026226044,
];
const ORB_ZBL_A_EXP: f64 = 0.30000001192092896;
const ORB_ZBL_A_PREFACTOR: f64 = 0.4542999863624573;
const ORB_ZBL_BOHR_ANGSTROM: f64 = 0.529;
const ORB_ZBL_COULOMB_EV_ANGSTROM: f64 = 14.3996;
const ORB_ZBL_CUTOFF_P: i32 = 6;
#[cfg(feature = "parallel")]
const ORB_PARALLEL_VALUE_THRESHOLD: usize = 65_536;
#[cfg(feature = "parallel")]
const ORB_PARALLEL_ROW_THRESHOLD: usize = 256;
#[allow(dead_code)]
#[cfg(all(feature = "parallel", feature = "orb-f32-edge-mlp"))]
const ORB_F32_CAST_PARALLEL_ROW_THRESHOLD: usize = 4_096;
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        target_arch = "x86_64"
    )
))]
const ORB_BATCHED_EXP_THRESHOLD: usize = 4_096;
const ORB_DUPLICATE_COMPONENT_TOLERANCE: f64 = 1e-12;

#[derive(Clone, Debug, Default)]
pub struct OrbProfileBlasShapeBreakdown {
    pub label: String,
    pub rows: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub full_input_dim: usize,
    pub column_offset: usize,
    pub calls: usize,
    pub ms: f64,
}

#[derive(Clone, Debug, Default)]
pub struct OrbProfileKernelBreakdown {
    pub label: String,
    pub rows: usize,
    pub width: usize,
    pub calls: usize,
    pub ms: f64,
}

#[derive(Clone, Debug, Default)]
struct OrbProfileBreakdown {
    blas_ms: f64,
    activation_ms: f64,
    scatter_reduction_ms: f64,
    blas_shapes: Vec<OrbProfileBlasShapeBreakdown>,
    kernels: Vec<OrbProfileKernelBreakdown>,
}

#[cfg(feature = "orb-profile-breakdown")]
mod orb_profile_breakdown {
    use std::{
        cell::RefCell,
        collections::BTreeMap,
        time::{Duration, Instant},
    };

    use super::{OrbProfileBlasShapeBreakdown, OrbProfileBreakdown, OrbProfileKernelBreakdown};

    #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
    struct BlasShapeKey {
        label: &'static str,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        full_input_dim: usize,
        column_offset: usize,
    }

    #[derive(Default)]
    struct BlasShapeAccumulator {
        calls: usize,
        elapsed: Duration,
    }

    #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
    struct KernelKey {
        label: &'static str,
        rows: usize,
        width: usize,
    }

    #[derive(Default)]
    struct KernelAccumulator {
        calls: usize,
        elapsed: Duration,
    }

    #[derive(Default)]
    struct Accumulator {
        blas: Duration,
        activation: Duration,
        scatter_reduction: Duration,
        blas_shapes: BTreeMap<BlasShapeKey, BlasShapeAccumulator>,
        kernels: BTreeMap<KernelKey, KernelAccumulator>,
    }

    thread_local! {
        static ACCUMULATOR: RefCell<Accumulator> = RefCell::new(Accumulator::default());
    }

    pub fn reset() {
        ACCUMULATOR.with(|accumulator| *accumulator.borrow_mut() = Accumulator::default());
    }

    pub fn snapshot() -> OrbProfileBreakdown {
        ACCUMULATOR.with(|accumulator| {
            let accumulator = accumulator.borrow();
            OrbProfileBreakdown {
                blas_ms: accumulator.blas.as_secs_f64() * 1000.0,
                activation_ms: accumulator.activation.as_secs_f64() * 1000.0,
                scatter_reduction_ms: accumulator.scatter_reduction.as_secs_f64() * 1000.0,
                blas_shapes: accumulator
                    .blas_shapes
                    .iter()
                    .map(|(key, value)| OrbProfileBlasShapeBreakdown {
                        label: key.label.to_owned(),
                        rows: key.rows,
                        input_dim: key.input_dim,
                        output_dim: key.output_dim,
                        full_input_dim: key.full_input_dim,
                        column_offset: key.column_offset,
                        calls: value.calls,
                        ms: value.elapsed.as_secs_f64() * 1000.0,
                    })
                    .collect(),
                kernels: accumulator
                    .kernels
                    .iter()
                    .map(|(key, value)| OrbProfileKernelBreakdown {
                        label: key.label.to_owned(),
                        rows: key.rows,
                        width: key.width,
                        calls: value.calls,
                        ms: value.elapsed.as_secs_f64() * 1000.0,
                    })
                    .collect(),
            }
        })
    }

    #[allow(dead_code)]
    pub fn measure_blas_shape<T>(
        label: &'static str,
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        full_input_dim: usize,
        column_offset: usize,
        f: impl FnOnce() -> T,
    ) -> T {
        let key = BlasShapeKey {
            label,
            rows,
            input_dim,
            output_dim,
            full_input_dim,
            column_offset,
        };
        measure(
            |accumulator, elapsed| {
                accumulator.blas += elapsed;
                let entry = accumulator.blas_shapes.entry(key).or_default();
                entry.calls += 1;
                entry.elapsed += elapsed;
            },
            f,
        )
    }

    pub fn measure_activation<T>(f: impl FnOnce() -> T) -> T {
        measure(|accumulator, elapsed| accumulator.activation += elapsed, f)
    }

    pub fn measure_kernel<T>(
        label: &'static str,
        rows: usize,
        width: usize,
        f: impl FnOnce() -> T,
    ) -> T {
        let key = KernelKey { label, rows, width };
        measure(
            |accumulator, elapsed| {
                let entry = accumulator.kernels.entry(key).or_default();
                entry.calls += 1;
                entry.elapsed += elapsed;
            },
            f,
        )
    }

    pub fn measure_scatter_reduction<T>(f: impl FnOnce() -> T) -> T {
        measure(
            |accumulator, elapsed| accumulator.scatter_reduction += elapsed,
            f,
        )
    }

    fn measure<T>(record: impl FnOnce(&mut Accumulator, Duration), f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let output = f();
        let elapsed = start.elapsed();
        ACCUMULATOR.with(|accumulator| record(&mut accumulator.borrow_mut(), elapsed));
        output
    }
}

#[cfg(feature = "orb-profile-breakdown")]
#[allow(unused_macros)]
macro_rules! orb_profile_blas_shape {
    ($label:expr, $rows:expr, $input_dim:expr, $output_dim:expr, $full_input_dim:expr, $column_offset:expr, $body:block) => {
        orb_profile_breakdown::measure_blas_shape(
            $label,
            $rows,
            $input_dim,
            $output_dim,
            $full_input_dim,
            $column_offset,
            || $body,
        )
    };
}

#[cfg(not(feature = "orb-profile-breakdown"))]
#[allow(unused_macros)]
macro_rules! orb_profile_blas_shape {
    ($label:expr, $rows:expr, $input_dim:expr, $output_dim:expr, $full_input_dim:expr, $column_offset:expr, $body:block) => {{
        let _ = (
            $label,
            $rows,
            $input_dim,
            $output_dim,
            $full_input_dim,
            $column_offset,
        );
        $body
    }};
}

#[cfg(feature = "orb-profile-breakdown")]
macro_rules! orb_profile_activation {
    ($body:block) => {
        orb_profile_breakdown::measure_activation(|| $body)
    };
}

#[cfg(not(feature = "orb-profile-breakdown"))]
macro_rules! orb_profile_activation {
    ($body:block) => {
        $body
    };
}

#[cfg(feature = "orb-profile-breakdown")]
macro_rules! orb_profile_kernel {
    ($label:expr, $rows:expr, $width:expr, $body:block) => {
        orb_profile_breakdown::measure_kernel($label, $rows, $width, || $body)
    };
}

#[cfg(not(feature = "orb-profile-breakdown"))]
macro_rules! orb_profile_kernel {
    ($label:expr, $rows:expr, $width:expr, $body:block) => {{
        let _ = ($label, $rows, $width);
        $body
    }};
}

#[cfg(feature = "orb-profile-breakdown")]
macro_rules! orb_profile_scatter_reduction {
    ($body:block) => {
        orb_profile_breakdown::measure_scatter_reduction(|| $body)
    };
}

#[cfg(not(feature = "orb-profile-breakdown"))]
macro_rules! orb_profile_scatter_reduction {
    ($body:block) => {
        $body
    };
}

#[derive(Clone, Debug)]
pub struct OrbV3OmolModel {
    spec: OrbV3OmolSpec,
    external_tensors: BTreeMap<String, OrbV3OmolExternalTensor>,
    tensor_index: HashMap<String, usize>,
    tensor_values_f64: Vec<Vec<f64>>,
    gnn_weight_cache: Option<OrbGnnWeightPlanCache>,
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    encoder_node_mlp_packed: Option<OrbFixedRowsMlpPackedDense>,
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    encoder_edge_mlp_packed: Option<OrbFixedRowsMlpPackedDense>,
    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    encoder_node_mlp_f32: Option<OrbMlpF32Dense>,
    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    encoder_edge_mlp_f32: Option<OrbMlpF32Dense>,
}

#[derive(Clone, Debug)]
pub struct OrbV3OmolPreparedGraph {
    pub n_atoms: usize,
    pub atomic_numbers: Vec<usize>,
    pub senders: Vec<usize>,
    pub receivers: Vec<usize>,
    pub edge_vectors: Vec<Vec3>,
    pub unit_shifts: Vec<Vec3>,
    pub total_charge: f64,
    pub spin_multiplicity: f64,
}

#[derive(Clone, Debug)]
pub struct OrbV3OmolInputFeatures {
    pub edge_distances: Vec<f64>,
    pub attention_cutoffs: Vec<f64>,
    pub radial_basis: Vec<f64>,
    pub angular_basis: Vec<f64>,
    pub edge_features: Vec<f64>,
    pub node_features: Vec<f64>,
    pub node_conditioning: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct OrbV3OmolEncodedFeatures {
    pub nodes: Vec<f64>,
    pub edges: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct OrbV3OmolInputFeatureCotangents {
    pub node_features: Vec<f64>,
    pub edge_features: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct OrbV3OmolGnnLayerOutput {
    pub nodes: Vec<f64>,
    pub edges: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct OrbV3OmolGnnOutput {
    pub layers: Vec<OrbV3OmolGnnLayerOutput>,
    layer_caches: Vec<OrbV3OmolGnnLayerCache>,
}

#[derive(Clone, Debug)]
pub struct OrbV3OmolGnnInputCotangents {
    pub encoded: OrbV3OmolEncodedFeatures,
    pub node_conditioning: Vec<f64>,
    pub attention_cutoffs: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct OrbV3OmolEnergyBreakdown {
    pub normalized_energy_head: f64,
    pub reference_energy: f64,
    pub denormalized_without_zbl: f64,
    pub zbl_energy: f64,
    pub total_energy: f64,
}

#[derive(Clone, Debug)]
pub struct OrbV3OmolEnergyTailVjp {
    pub breakdown: OrbV3OmolEnergyBreakdown,
    pub final_node_cotangents: Vec<f64>,
    pub zbl_forces: Vec<Vec3>,
}

#[derive(Clone, Debug)]
pub struct OrbV3OmolEnergyForcesProfile {
    pub atom_count: usize,
    pub edge_count: usize,
    pub total_ms: f64,
    pub input_features_ms: f64,
    pub encoder_forward_ms: f64,
    pub gnn_weight_plan_ms: f64,
    pub gnn_forward_ms: f64,
    pub energy_tail_vjp_ms: f64,
    pub final_cotangent_ms: f64,
    pub gnn_input_vjp_ms: f64,
    pub encoder_vjp_ms: f64,
    pub geometry_forces_ms: f64,
    pub zbl_force_accum_ms: f64,
    pub force_vjp_ms: f64,
    pub blas_ms: f64,
    pub activation_ms: f64,
    pub scatter_reduction_ms: f64,
    pub blas_shapes: Vec<OrbProfileBlasShapeBreakdown>,
    pub kernels: Vec<OrbProfileKernelBreakdown>,
    pub total_energy: f64,
    pub force_checksum: f64,
    pub force_l2_norm: f64,
    pub force_max_abs: f64,
    pub forces: Vec<Vec3>,
}

#[derive(Clone, Debug)]
struct OrbV3OmolGnnLayerCache {
    receive_attn: Vec<f64>,
    receive_attn_sigmoid: Vec<f64>,
    send_attn: Vec<f64>,
    send_attn_sigmoid: Vec<f64>,
    updated_edges: Vec<f64>,
    edge_mlp: OrbMlpRmsNormForwardCache,
    node_mlp: OrbMlpRmsNormForwardCache,
}

#[derive(Clone, Debug)]
struct OrbMlpRmsNormForwardCache {
    layer0_silu_derivative: Vec<f64>,
    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    layer0_silu_derivative_f32: Option<Vec<f32>>,
    layer1_silu_derivative: Vec<f64>,
    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    layer1_silu_derivative_f32: Option<Vec<f32>>,
    layer2: Vec<f64>,
    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    layer2_f32: Option<Vec<f32>>,
    layer2_inv_rms: Vec<f64>,
    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    layer2_inv_rms_f32: Option<Vec<f32>>,
}

type OrbNodeEdgeListRefs<'a> = (&'a [Vec<usize>], &'a [Vec<usize>]);

#[derive(Clone, Debug)]
struct OrbEdgeAttentionForward {
    values: Vec<f64>,
    sigmoid: Vec<f64>,
}

#[derive(Clone, Debug)]
struct OrbV3OmolEncodedFeatureCache {
    node_mlp: OrbMlpRmsNormForwardCache,
    edge_mlp: OrbMlpRmsNormForwardCache,
}

#[derive(Clone, Copy)]
struct OrbMlpRmsNormWeights<'a> {
    layer0_weight: &'a [f64],
    layer0_bias: &'a [f64],
    layer1_weight: &'a [f64],
    layer1_bias: &'a [f64],
    layer2_weight: &'a [f64],
    layer2_bias: &'a [f64],
    rms_weight: &'a [f64],
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    fixed_rows_packed: Option<&'a OrbFixedRowsMlpPackedDense>,
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    edge_packed: Option<&'a OrbEdgeMlpPackedDense>,
    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    mlp_f32: Option<&'a OrbMlpF32Dense>,
    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    edge_f32: Option<&'a OrbEdgeMlpF32Dense>,
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
#[derive(Clone, Debug)]
struct OrbFixedRowsMlpPackedDense {
    rows: usize,
    input_dim: usize,
    labels: OrbFixedRowsMlpPackedLabels,
    layer0_forward: Arc<crate::blas_backend::MklPackedDgemmBMatrix>,
    layer0_input_vjp: Arc<crate::blas_backend::MklPackedDgemmBMatrix>,
    layer1_forward: Arc<crate::blas_backend::MklPackedDgemmBMatrix>,
    layer1_input_vjp: Arc<crate::blas_backend::MklPackedDgemmBMatrix>,
    layer2_forward: Arc<crate::blas_backend::MklPackedDgemmBMatrix>,
    layer2_input_vjp: Arc<crate::blas_backend::MklPackedDgemmBMatrix>,
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
#[derive(Clone, Copy, Debug)]
struct OrbFixedRowsMlpPackedLabels {
    layer0_forward: &'static str,
    layer0_input_vjp: &'static str,
    layer1_forward: &'static str,
    layer1_input_vjp: &'static str,
    layer2_forward: &'static str,
    layer2_input_vjp: &'static str,
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
const ORB_ENCODER_NODE_MLP_PACKED_LABELS: OrbFixedRowsMlpPackedLabels =
    OrbFixedRowsMlpPackedLabels {
        layer0_forward: "encoder_node_mlp_layer0_forward_mkl_packed",
        layer0_input_vjp: "encoder_node_mlp_layer0_input_vjp_mkl_packed",
        layer1_forward: "encoder_node_mlp_layer1_forward_mkl_packed",
        layer1_input_vjp: "encoder_node_mlp_layer1_input_vjp_mkl_packed",
        layer2_forward: "encoder_node_mlp_layer2_forward_mkl_packed",
        layer2_input_vjp: "encoder_node_mlp_layer2_input_vjp_mkl_packed",
    };

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
const ORB_ENCODER_EDGE_MLP_PACKED_LABELS: OrbFixedRowsMlpPackedLabels =
    OrbFixedRowsMlpPackedLabels {
        layer0_forward: "encoder_edge_mlp_layer0_forward_mkl_packed",
        layer0_input_vjp: "encoder_edge_mlp_layer0_input_vjp_mkl_packed",
        layer1_forward: "encoder_edge_mlp_layer1_forward_mkl_packed",
        layer1_input_vjp: "encoder_edge_mlp_layer1_input_vjp_mkl_packed",
        layer2_forward: "encoder_edge_mlp_layer2_forward_mkl_packed",
        layer2_input_vjp: "encoder_edge_mlp_layer2_input_vjp_mkl_packed",
    };

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
const ORB_GNN_NODE_MLP_PACKED_LABELS: OrbFixedRowsMlpPackedLabels = OrbFixedRowsMlpPackedLabels {
    layer0_forward: "node_mlp_layer0_forward_mkl_packed",
    layer0_input_vjp: "node_mlp_layer0_input_vjp_mkl_packed",
    layer1_forward: "node_mlp_layer1_forward_mkl_packed",
    layer1_input_vjp: "node_mlp_layer1_input_vjp_mkl_packed",
    layer2_forward: "node_mlp_layer2_forward_mkl_packed",
    layer2_input_vjp: "node_mlp_layer2_input_vjp_mkl_packed",
};

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
#[derive(Clone, Debug)]
struct OrbEdgeMlpPackedDense {
    layer0_edge_forward: Arc<crate::blas_backend::MklPackedDgemmBMatrix>,
    layer0_edge_vjp: Arc<crate::blas_backend::MklPackedDgemmBMatrix>,
    layer0_sender_forward: Arc<crate::blas_backend::MklPackedDgemmBMatrix>,
    layer0_receiver_forward: Arc<crate::blas_backend::MklPackedDgemmBMatrix>,
    layer0_sender_vjp: Arc<crate::blas_backend::MklPackedDgemmBMatrix>,
    layer0_receiver_vjp: Arc<crate::blas_backend::MklPackedDgemmBMatrix>,
    layer1_forward: Arc<crate::blas_backend::MklPackedDgemmBMatrix>,
    layer1_input_vjp: Arc<crate::blas_backend::MklPackedDgemmBMatrix>,
    layer2_forward: Arc<crate::blas_backend::MklPackedDgemmBMatrix>,
    layer2_input_vjp: Arc<crate::blas_backend::MklPackedDgemmBMatrix>,
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
#[derive(Clone, Debug)]
struct OrbEdgeMlpF32Dense {
    layer0_weight: Vec<f32>,
    layer0_bias: Vec<f32>,
    layer1_weight: Vec<f32>,
    layer1_weight_input_major: Vec<f32>,
    layer1_bias: Vec<f32>,
    layer2_weight: Vec<f32>,
    layer2_bias: Vec<f32>,
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
#[derive(Clone, Debug)]
struct OrbMlpF32Dense {
    input_dim: usize,
    layer0_weight: Vec<f32>,
    layer0_bias: Vec<f32>,
    layer1_weight: Vec<f32>,
    layer1_bias: Vec<f32>,
    layer2_weight: Vec<f32>,
    layer2_bias: Vec<f32>,
}

struct OrbEdgeMlpF32Layer12Forward {
    layer2_f32: Vec<f32>,
    layer1_silu_derivative_f32: Vec<f32>,
}

struct OrbEdgeMlpF32Forward {
    layer0_silu_derivative_f32: Vec<f32>,
    layer2_f32: Vec<f32>,
    layer1_silu_derivative_f32: Vec<f32>,
}

#[derive(Clone, Copy)]
struct OrbLinearWeights<'a> {
    weight: &'a [f64],
    bias: &'a [f64],
}

struct OrbGnnLayerWeights<'a> {
    cond_node: OrbLinearWeights<'a>,
    receive_attn: OrbLinearWeights<'a>,
    send_attn: OrbLinearWeights<'a>,
    attention_pair_weight: Vec<f64>,
    attention_pair_bias: [f64; 2],
    edge_mlp: OrbMlpRmsNormWeights<'a>,
    node_mlp: OrbMlpRmsNormWeights<'a>,
}

struct OrbGnnInitialCotangents {
    nodes: Vec<f64>,
    edges: Vec<f64>,
}

struct OrbEdgeAttentionPairVjpBuffers<'a> {
    receive_sigmoid: &'a [f64],
    send_sigmoid: &'a [f64],
    grad_receive_attention: &'a [f64],
    grad_send_attention: &'a [f64],
    grad_edges: &'a mut [f64],
    grad_attention_cutoffs: &'a mut [f64],
}

struct OrbEdgeMlpInputCotangents {
    edges: Vec<f64>,
    nodes: Vec<f64>,
}

#[derive(Clone, Debug)]
struct OrbGnnWeightPlanCache {
    layers: Vec<OrbGnnLayerWeightCache>,
}

#[derive(Clone, Debug)]
struct OrbGnnLayerWeightCache {
    cond_node_weight_key: String,
    cond_node_bias_key: String,
    receive_attn_weight_key: String,
    receive_attn_bias_key: String,
    send_attn_weight_key: String,
    send_attn_bias_key: String,
    edge_mlp_keys: OrbMlpRmsNormWeightKeys,
    node_mlp_keys: OrbMlpRmsNormWeightKeys,
    attention_pair_weight: Vec<f64>,
    attention_pair_bias: [f64; 2],
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    node_mlp_packed: Option<OrbFixedRowsMlpPackedDense>,
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    edge_mlp_packed: Option<OrbEdgeMlpPackedDense>,
    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    node_mlp_f32: Option<OrbMlpF32Dense>,
    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    edge_mlp_f32: Option<OrbEdgeMlpF32Dense>,
}

#[derive(Clone, Debug)]
struct OrbMlpRmsNormWeightKeys {
    layer0_weight: String,
    layer0_bias: String,
    layer1_weight: String,
    layer1_bias: String,
    layer2_weight: String,
    layer2_bias: String,
    rms_weight: String,
}

#[derive(Clone, Debug)]
struct OrbDuplicateComponentPlan {
    components: Vec<Vec<usize>>,
    representative: OrbV3OmolPreparedGraph,
}

impl OrbV3OmolPreparedGraph {
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.n_atoms
    }

    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.senders.len()
    }
}

#[derive(Clone, Debug)]
pub struct OrbV3OmolExternalTensor {
    pub shape: Vec<usize>,
    pub path: String,
    pub values: OrbV3OmolExternalTensorValues,
    pub byte_sha256: String,
}

impl OrbV3OmolExternalTensor {
    #[must_use]
    pub fn dtype(&self) -> &'static str {
        match &self.values {
            OrbV3OmolExternalTensorValues::F64(_) => "f64",
            OrbV3OmolExternalTensorValues::I64(_) => "i64",
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        match &self.values {
            OrbV3OmolExternalTensorValues::F64(values) => values.len(),
            OrbV3OmolExternalTensorValues::I64(values) => values.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Debug)]
pub enum OrbV3OmolExternalTensorValues {
    F64(Vec<f64>),
    I64(Vec<i64>),
}

impl OrbV3OmolModel {
    pub fn from_json_str(text: &str) -> Result<Self, OrbV3OmolError> {
        let spec: OrbV3OmolSpec = serde_json::from_str(text)?;
        validate_spec(&spec)?;
        if spec.runtime_status == RUNTIME_STATUS_STATE_EXPORT {
            return invalid("state_export_only ORB metadata requires external tensors");
        }
        Ok(Self {
            spec,
            external_tensors: BTreeMap::new(),
            tensor_index: HashMap::new(),
            tensor_values_f64: Vec::new(),
            gnn_weight_cache: None,
            #[cfg(all(
                feature = "orb-fixed-shape-dense",
                feature = "intel-mkl",
                target_os = "linux",
                target_arch = "x86_64"
            ))]
            encoder_node_mlp_packed: None,
            #[cfg(all(
                feature = "orb-fixed-shape-dense",
                feature = "intel-mkl",
                target_os = "linux",
                target_arch = "x86_64"
            ))]
            encoder_edge_mlp_packed: None,
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            encoder_node_mlp_f32: None,
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            encoder_edge_mlp_f32: None,
        })
    }

    pub fn from_json_str_with_external_tensors(
        text: &str,
        external_tensors: BTreeMap<String, OrbV3OmolExternalTensor>,
    ) -> Result<Self, OrbV3OmolError> {
        let spec: OrbV3OmolSpec = serde_json::from_str(text)?;
        validate_spec(&spec)?;
        validate_state_export_tensors(&spec, &external_tensors)?;
        let (tensor_index, tensor_values_f64) = build_tensor_index(&external_tensors);
        let gnn_weight_cache = if spec.runtime_status == RUNTIME_STATUS_STATE_EXPORT {
            Some(build_gnn_weight_cache(&tensor_index, &tensor_values_f64)?)
        } else {
            None
        };
        #[cfg(all(
            feature = "orb-fixed-shape-dense",
            feature = "intel-mkl",
            target_os = "linux",
            target_arch = "x86_64"
        ))]
        let encoder_node_mlp_packed = if spec.runtime_status == RUNTIME_STATUS_STATE_EXPORT {
            Some(build_encoder_node_mlp_packed(
                &tensor_index,
                &tensor_values_f64,
            )?)
        } else {
            None
        };
        #[cfg(all(
            feature = "orb-fixed-shape-dense",
            feature = "intel-mkl",
            target_os = "linux",
            target_arch = "x86_64"
        ))]
        let encoder_edge_mlp_packed = if spec.runtime_status == RUNTIME_STATUS_STATE_EXPORT {
            Some(build_encoder_edge_mlp_packed(
                &tensor_index,
                &tensor_values_f64,
            )?)
        } else {
            None
        };
        #[cfg(all(
            feature = "orb-f32-edge-mlp",
            any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    any(feature = "intel-mkl", feature = "openblas"),
                    target_os = "linux",
                    target_arch = "x86_64"
                )
            )
        ))]
        let encoder_node_mlp_f32 = if spec.runtime_status == RUNTIME_STATUS_STATE_EXPORT {
            Some(build_encoder_mlp_f32(
                ORB_ENCODER_NODE_PREFIX,
                LATENT_DIM,
                &tensor_index,
                &tensor_values_f64,
            )?)
        } else {
            None
        };
        #[cfg(all(
            feature = "orb-f32-edge-mlp",
            any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    any(feature = "intel-mkl", feature = "openblas"),
                    target_os = "linux",
                    target_arch = "x86_64"
                )
            )
        ))]
        let encoder_edge_mlp_f32 = if spec.runtime_status == RUNTIME_STATUS_STATE_EXPORT {
            Some(build_encoder_mlp_f32(
                ORB_ENCODER_EDGE_PREFIX,
                ORB_EDGE_FEATURE_DIM,
                &tensor_index,
                &tensor_values_f64,
            )?)
        } else {
            None
        };
        Ok(Self {
            spec,
            external_tensors,
            tensor_index,
            tensor_values_f64,
            gnn_weight_cache,
            #[cfg(all(
                feature = "orb-fixed-shape-dense",
                feature = "intel-mkl",
                target_os = "linux",
                target_arch = "x86_64"
            ))]
            encoder_node_mlp_packed,
            #[cfg(all(
                feature = "orb-fixed-shape-dense",
                feature = "intel-mkl",
                target_os = "linux",
                target_arch = "x86_64"
            ))]
            encoder_edge_mlp_packed,
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            encoder_node_mlp_f32,
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            encoder_edge_mlp_f32,
        })
    }

    #[must_use]
    pub fn selected_model(&self) -> &str {
        &self.spec.selected_model
    }

    #[must_use]
    pub fn runtime_status(&self) -> &str {
        &self.spec.runtime_status
    }

    #[must_use]
    pub fn reference_commit(&self) -> &str {
        &self.spec.reference.commit
    }

    #[must_use]
    pub fn artifact_sha256(&self) -> &str {
        &self.spec.checkpoint.sha256
    }

    #[must_use]
    pub fn artifact_size_bytes(&self) -> u64 {
        self.spec.checkpoint.size_bytes
    }

    #[must_use]
    pub fn cutoff_angstrom(&self) -> f64 {
        self.spec.graph.cutoff_angstrom
    }

    #[must_use]
    pub fn max_num_neighbors(&self) -> usize {
        self.spec.graph.max_num_neighbors
    }

    #[must_use]
    pub fn architecture_parameters(&self) -> &OrbArchitectureParameters {
        &self.spec.architecture_parameters
    }

    #[must_use]
    pub fn state_tensor_count(&self) -> usize {
        self.external_tensors.len()
    }

    #[must_use]
    pub fn state_tensor_total_numel(&self) -> usize {
        self.external_tensors
            .values()
            .map(OrbV3OmolExternalTensor::len)
            .sum()
    }

    #[must_use]
    pub fn state_tensor_dtype_count(&self, dtype: &str) -> usize {
        self.external_tensors
            .values()
            .filter(|tensor| tensor.dtype() == dtype)
            .count()
    }

    pub fn state_tensor_f64(
        &self,
        name: &str,
        expected_shape: &[usize],
    ) -> Result<&[f64], OrbV3OmolError> {
        let tensor = self.state_tensor(name, expected_shape)?;
        match &tensor.values {
            OrbV3OmolExternalTensorValues::F64(values) => Ok(values),
            OrbV3OmolExternalTensorValues::I64(_) => {
                Err(OrbV3OmolError::StateTensorDtypeMismatch {
                    name: name.to_owned(),
                    expected: "f64",
                    actual: tensor.dtype(),
                })
            }
        }
    }

    pub fn state_tensor_i64(
        &self,
        name: &str,
        expected_shape: &[usize],
    ) -> Result<&[i64], OrbV3OmolError> {
        let tensor = self.state_tensor(name, expected_shape)?;
        match &tensor.values {
            OrbV3OmolExternalTensorValues::I64(values) => Ok(values),
            OrbV3OmolExternalTensorValues::F64(_) => {
                Err(OrbV3OmolError::StateTensorDtypeMismatch {
                    name: name.to_owned(),
                    expected: "i64",
                    actual: tensor.dtype(),
                })
            }
        }
    }

    fn state_tensor(
        &self,
        name: &str,
        expected_shape: &[usize],
    ) -> Result<&OrbV3OmolExternalTensor, OrbV3OmolError> {
        let tensor =
            self.external_tensors
                .get(name)
                .ok_or_else(|| OrbV3OmolError::MissingStateTensor {
                    name: name.to_owned(),
                })?;
        if tensor.shape != expected_shape {
            return Err(OrbV3OmolError::StateTensorShapeMismatch {
                name: name.to_owned(),
                expected: expected_shape.to_vec(),
                actual: tensor.shape.clone(),
            });
        }
        Ok(tensor)
    }

    pub fn prepare_non_pbc_graph(
        &self,
        structure: &Structure,
        total_charge: f64,
        spin_multiplicity: f64,
    ) -> Result<OrbV3OmolPreparedGraph, OrbV3OmolError> {
        if structure.pbc.iter().any(|is_periodic| *is_periodic) {
            return Err(OrbV3OmolError::PbcUnsupported);
        }
        validate_charge_spin_values(total_charge, spin_multiplicity)?;
        let neighbor_list = all_pairs_neighbor_list(structure, self.cutoff_angstrom())?;
        let mut neighbors_per_sender = vec![0usize; structure.len()];
        for edge in &neighbor_list.edges {
            neighbors_per_sender[edge.src] += 1;
        }
        if let Some((atom, count)) = neighbors_per_sender
            .iter()
            .copied()
            .enumerate()
            .find(|(_, count)| *count > self.max_num_neighbors())
        {
            return Err(OrbV3OmolError::TooManyNeighbors {
                atom,
                count,
                max: self.max_num_neighbors(),
            });
        }
        let atomic_numbers = structure
            .species
            .iter()
            .enumerate()
            .map(|(atom, symbol)| orb_atomic_number(symbol, atom))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(OrbV3OmolPreparedGraph {
            n_atoms: structure.len(),
            atomic_numbers,
            senders: neighbor_list.edges.iter().map(|edge| edge.src).collect(),
            receivers: neighbor_list.edges.iter().map(|edge| edge.dst).collect(),
            edge_vectors: neighbor_list.edges.iter().map(|edge| edge.vector).collect(),
            unit_shifts: vec![[0.0; 3]; neighbor_list.edges.len()],
            total_charge,
            spin_multiplicity,
        })
    }

    pub fn reference_energy_ev(
        &self,
        graph: &OrbV3OmolPreparedGraph,
    ) -> Result<f64, OrbV3OmolError> {
        if graph.atomic_numbers.len() != graph.n_atoms {
            return invalid(format!(
                "ORB prepared graph atomic_numbers length must be {}, got {}",
                graph.n_atoms,
                graph.atomic_numbers.len()
            ));
        }
        let weights =
            self.state_tensor_f64(ORB_REFERENCE_ENERGY_WEIGHT, &[ORB_REFERENCE_ENERGY_CLASSES])?;
        let mut energy = 0.0;
        for (atom, &atomic_number) in graph.atomic_numbers.iter().enumerate() {
            if atomic_number > ORB_MAX_SUPPORTED_ATOMIC_NUMBER {
                return Err(OrbV3OmolError::UnsupportedAtomicNumber {
                    atom,
                    atomic_number,
                });
            }
            energy += weights[atomic_number];
        }
        Ok(energy)
    }

    pub fn input_features(
        &self,
        graph: &OrbV3OmolPreparedGraph,
    ) -> Result<OrbV3OmolInputFeatures, OrbV3OmolError> {
        validate_graph_lengths(graph)?;

        let bessel_weights = self.state_tensor_f64(ORB_BESSEL_WEIGHTS, &[BESSEL_NUM_BASES])?;
        let r_max = self.state_tensor_f64(ORB_BESSEL_R_MAX, &[])?[0];
        let prefactor = self.state_tensor_f64(ORB_BESSEL_PREFACTOR, &[])?[0];
        let atom_embedding = self.state_tensor_f64(
            ORB_ATOM_EMBEDDING_WEIGHT,
            &[ORB_ATOM_EMBEDDING_ROWS, LATENT_DIM],
        )?;
        let charge_w =
            self.state_tensor_f64(ORB_CHARGE_EMBEDDING_W, &[ORB_CONDITIONING_FREQUENCIES])?;
        let spin_w =
            self.state_tensor_f64(ORB_SPIN_EMBEDDING_W, &[ORB_CONDITIONING_FREQUENCIES])?;

        if !r_max.is_finite() || r_max <= 0.0 {
            return invalid(format!(
                "ORB bessel r_max must be positive and finite, got {r_max}"
            ));
        }
        if !prefactor.is_finite() {
            return invalid(format!(
                "ORB bessel prefactor must be finite, got {prefactor}"
            ));
        }

        let edge_count = graph.edge_count();
        let mut edge_distances = vec![0.0; edge_count];
        let mut attention_cutoffs = vec![0.0; edge_count];
        let mut radial_basis = vec![0.0; edge_count * BESSEL_NUM_BASES];
        let mut angular_basis = vec![0.0; edge_count * ORB_ANGULAR_DIM];
        let mut edge_features = vec![0.0; edge_count * ORB_EDGE_FEATURE_DIM];
        let sh_plan = SphericalHarmonicsPlan::component(SH_LMAX, true)?;

        let compute_edge = |edge: usize,
                            d: &mut [f64],
                            c: &mut [f64],
                            radial_row: &mut [f64],
                            angular_row: &mut [f64],
                            edge_row: &mut [f64]|
         -> Result<(), OrbV3OmolError> {
            let vector = graph.edge_vectors[edge];
            let distance = vec3_norm(vector);
            if !distance.is_finite() || distance <= 0.0 {
                return invalid(format!(
                    "ORB edge {edge} distance must be positive and finite, got {distance}"
                ));
            }
            d[0] = distance;
            let cutoff = orb_cutoff(distance, r_max);
            c[0] = cutoff;
            for (basis, (&weight, out)) in
                bessel_weights.iter().zip(radial_row.iter_mut()).enumerate()
            {
                if !weight.is_finite() {
                    return invalid(format!(
                        "ORB bessel weight {basis} must be finite, got {weight}"
                    ));
                }
                *out = prefactor * (weight * distance).sin() / distance;
            }
            sh_plan.forward(vector, angular_row)?;
            for basis in 0..BESSEL_NUM_BASES {
                for angular in 0..ORB_ANGULAR_DIM {
                    edge_row[basis * ORB_ANGULAR_DIM + angular] =
                        cutoff * radial_row[basis] * angular_row[angular];
                }
            }
            Ok(())
        };

        #[cfg(feature = "parallel")]
        let run_serial =
            edge_count < ORB_PARALLEL_ROW_THRESHOLD || rayon::current_num_threads() <= 1;
        #[cfg(not(feature = "parallel"))]
        let run_serial = true;

        #[cfg(feature = "parallel")]
        if !run_serial {
            edge_distances
                .par_chunks_exact_mut(1)
                .zip(attention_cutoffs.par_chunks_exact_mut(1))
                .zip(radial_basis.par_chunks_exact_mut(BESSEL_NUM_BASES))
                .zip(angular_basis.par_chunks_exact_mut(ORB_ANGULAR_DIM))
                .zip(edge_features.par_chunks_exact_mut(ORB_EDGE_FEATURE_DIM))
                .enumerate()
                .try_for_each(|(edge, ((((dr, cr), rr), ar), er))| {
                    compute_edge(edge, dr, cr, rr, ar, er)
                })?;
        }

        if run_serial {
            edge_distances
                .chunks_exact_mut(1)
                .zip(attention_cutoffs.chunks_exact_mut(1))
                .zip(radial_basis.chunks_exact_mut(BESSEL_NUM_BASES))
                .zip(angular_basis.chunks_exact_mut(ORB_ANGULAR_DIM))
                .zip(edge_features.chunks_exact_mut(ORB_EDGE_FEATURE_DIM))
                .enumerate()
                .try_for_each(|(edge, ((((dr, cr), rr), ar), er))| {
                    compute_edge(edge, dr, cr, rr, ar, er)
                })?;
        }

        let mut node_features = vec![0.0; graph.n_atoms * LATENT_DIM];
        for (atom, &atomic_number) in graph.atomic_numbers.iter().enumerate() {
            if atomic_number >= ORB_ATOM_EMBEDDING_ROWS {
                return Err(OrbV3OmolError::UnsupportedAtomicNumber {
                    atom,
                    atomic_number,
                });
            }
            let src_start = atomic_number * LATENT_DIM;
            let dst_start = atom * LATENT_DIM;
            node_features[dst_start..dst_start + LATENT_DIM]
                .copy_from_slice(&atom_embedding[src_start..src_start + LATENT_DIM]);
        }

        let mut conditioning_row = vec![0.0; LATENT_DIM];
        fill_charge_spin_embedding(
            graph.total_charge,
            charge_w,
            &mut conditioning_row[..ORB_CONDITIONING_HALF_DIM],
            false,
        )?;
        fill_charge_spin_embedding(
            graph.spin_multiplicity,
            spin_w,
            &mut conditioning_row[ORB_CONDITIONING_HALF_DIM..],
            true,
        )?;
        let mut node_conditioning = vec![0.0; graph.n_atoms * LATENT_DIM];
        for atom in 0..graph.n_atoms {
            node_conditioning[atom * LATENT_DIM..(atom + 1) * LATENT_DIM]
                .copy_from_slice(&conditioning_row);
        }

        Ok(OrbV3OmolInputFeatures {
            edge_distances,
            attention_cutoffs,
            radial_basis,
            angular_basis,
            edge_features,
            node_features,
            node_conditioning,
        })
    }

    pub fn encode_input_features(
        &self,
        features: &OrbV3OmolInputFeatures,
    ) -> Result<OrbV3OmolEncodedFeatures, OrbV3OmolError> {
        let (encoded, _) = self.encode_input_features_with_cache(features)?;
        Ok(encoded)
    }

    fn encode_input_features_with_cache(
        &self,
        features: &OrbV3OmolInputFeatures,
    ) -> Result<(OrbV3OmolEncodedFeatures, OrbV3OmolEncodedFeatureCache), OrbV3OmolError> {
        self.encode_input_features_with_cache_for_compute_dtype(
            features,
            PredictionComputeDtype::Float64,
        )
    }

    fn encode_input_features_with_cache_for_compute_dtype(
        &self,
        features: &OrbV3OmolInputFeatures,
        dtype: PredictionComputeDtype,
    ) -> Result<(OrbV3OmolEncodedFeatures, OrbV3OmolEncodedFeatureCache), OrbV3OmolError> {
        let node_rows = checked_row_count(
            "ORB node input features",
            &features.node_features,
            LATENT_DIM,
        )?;
        let edge_rows = checked_row_count(
            "ORB edge input features",
            &features.edge_features,
            ORB_EDGE_FEATURE_DIM,
        )?;
        let (nodes, node_mlp) = self.mlp_rms_norm_forward_with_cache_for_compute_dtype(
            ORB_ENCODER_NODE_PREFIX,
            &features.node_features,
            node_rows,
            LATENT_DIM,
            dtype,
        )?;
        let (edges, edge_mlp) = self.mlp_rms_norm_forward_with_cache_for_compute_dtype(
            ORB_ENCODER_EDGE_PREFIX,
            &features.edge_features,
            edge_rows,
            ORB_EDGE_FEATURE_DIM,
            dtype,
        )?;
        Ok((
            OrbV3OmolEncodedFeatures { nodes, edges },
            OrbV3OmolEncodedFeatureCache { node_mlp, edge_mlp },
        ))
    }

    pub fn encode_input_features_vjp(
        &self,
        features: &OrbV3OmolInputFeatures,
        encoded_cotangents: &OrbV3OmolEncodedFeatures,
    ) -> Result<OrbV3OmolInputFeatureCotangents, OrbV3OmolError> {
        let (_, cache) = self.encode_input_features_with_cache(features)?;
        self.encode_input_features_vjp_cached(features, encoded_cotangents, &cache)
    }

    fn encode_input_features_vjp_cached(
        &self,
        features: &OrbV3OmolInputFeatures,
        encoded_cotangents: &OrbV3OmolEncodedFeatures,
        cache: &OrbV3OmolEncodedFeatureCache,
    ) -> Result<OrbV3OmolInputFeatureCotangents, OrbV3OmolError> {
        self.encode_input_features_vjp_cached_for_compute_dtype(
            features,
            encoded_cotangents,
            cache,
            PredictionComputeDtype::Float64,
        )
    }

    fn encode_input_features_vjp_cached_for_compute_dtype(
        &self,
        features: &OrbV3OmolInputFeatures,
        encoded_cotangents: &OrbV3OmolEncodedFeatures,
        cache: &OrbV3OmolEncodedFeatureCache,
        dtype: PredictionComputeDtype,
    ) -> Result<OrbV3OmolInputFeatureCotangents, OrbV3OmolError> {
        let node_rows = checked_row_count(
            "ORB node input features",
            &features.node_features,
            LATENT_DIM,
        )?;
        let edge_rows = checked_row_count(
            "ORB edge input features",
            &features.edge_features,
            ORB_EDGE_FEATURE_DIM,
        )?;
        expect_len(
            "ORB encoded node cotangents",
            &encoded_cotangents.nodes,
            node_rows * LATENT_DIM,
        )?;
        expect_len(
            "ORB encoded edge cotangents",
            &encoded_cotangents.edges,
            edge_rows * LATENT_DIM,
        )?;
        let node_features = self.mlp_rms_norm_input_vjp_cached_for_compute_dtype(
            ORB_ENCODER_NODE_PREFIX,
            &cache.node_mlp,
            &encoded_cotangents.nodes,
            node_rows,
            LATENT_DIM,
            dtype,
        )?;
        let edge_features = self.mlp_rms_norm_input_vjp_cached_for_compute_dtype(
            ORB_ENCODER_EDGE_PREFIX,
            &cache.edge_mlp,
            &encoded_cotangents.edges,
            edge_rows,
            ORB_EDGE_FEATURE_DIM,
            dtype,
        )?;
        Ok(OrbV3OmolInputFeatureCotangents {
            node_features,
            edge_features,
        })
    }

    fn encode_input_feature_edge_vjp_cached_for_geometry_with_compute_dtype(
        &self,
        features: &OrbV3OmolInputFeatures,
        encoded_edge_cotangents: &[f64],
        cache: &OrbV3OmolEncodedFeatureCache,
        dtype: PredictionComputeDtype,
    ) -> Result<OrbV3OmolInputFeatureCotangents, OrbV3OmolError> {
        let edge_rows = checked_row_count(
            "ORB edge input features",
            &features.edge_features,
            ORB_EDGE_FEATURE_DIM,
        )?;
        expect_len(
            "ORB encoded edge cotangents",
            encoded_edge_cotangents,
            edge_rows * LATENT_DIM,
        )?;
        let edge_features = self.mlp_rms_norm_input_vjp_cached_for_compute_dtype(
            ORB_ENCODER_EDGE_PREFIX,
            &cache.edge_mlp,
            encoded_edge_cotangents,
            edge_rows,
            ORB_EDGE_FEATURE_DIM,
            dtype,
        )?;
        Ok(OrbV3OmolInputFeatureCotangents {
            node_features: Vec::new(),
            edge_features,
        })
    }

    pub fn gnn_forward(
        &self,
        graph: &OrbV3OmolPreparedGraph,
        features: &OrbV3OmolInputFeatures,
        encoded: &OrbV3OmolEncodedFeatures,
    ) -> Result<OrbV3OmolGnnOutput, OrbV3OmolError> {
        let weights = self.gnn_layer_weight_plan()?;
        self.gnn_forward_impl(graph, features, encoded, &weights, true)
    }

    fn gnn_forward_for_energy_tail_with_weights(
        &self,
        graph: &OrbV3OmolPreparedGraph,
        features: &OrbV3OmolInputFeatures,
        encoded: &OrbV3OmolEncodedFeatures,
        weights_by_layer: &[OrbGnnLayerWeights<'_>],
    ) -> Result<OrbV3OmolGnnOutput, OrbV3OmolError> {
        self.gnn_forward_impl(graph, features, encoded, weights_by_layer, false)
    }

    fn gnn_forward_impl(
        &self,
        graph: &OrbV3OmolPreparedGraph,
        features: &OrbV3OmolInputFeatures,
        encoded: &OrbV3OmolEncodedFeatures,
        weights_by_layer: &[OrbGnnLayerWeights<'_>],
        retain_all_layer_outputs: bool,
    ) -> Result<OrbV3OmolGnnOutput, OrbV3OmolError> {
        validate_graph_lengths(graph)?;
        if weights_by_layer.len() != MESSAGE_PASSING_STEPS {
            return invalid(format!(
                "ORB GNN weight plan requires {MESSAGE_PASSING_STEPS} layers, got {}",
                weights_by_layer.len()
            ));
        }
        let node_rows = graph.n_atoms;
        let edge_rows = graph.edge_count();
        expect_len(
            "ORB encoded node features",
            &encoded.nodes,
            node_rows * LATENT_DIM,
        )?;
        expect_len(
            "ORB encoded edge features",
            &encoded.edges,
            edge_rows * LATENT_DIM,
        )?;
        expect_len(
            "ORB node conditioning",
            &features.node_conditioning,
            node_rows * LATENT_DIM,
        )?;
        expect_len(
            "ORB attention cutoffs",
            &features.attention_cutoffs,
            edge_rows,
        )?;

        let mut nodes = encoded.nodes.clone();
        let mut edges = encoded.edges.clone();
        let mut layers = Vec::with_capacity(MESSAGE_PASSING_STEPS);
        let mut layer_caches = Vec::with_capacity(MESSAGE_PASSING_STEPS);
        let mut node_input = vec![0.0; node_rows * ORB_GNN_INPUT_DIM];
        #[cfg(feature = "parallel")]
        let parallel_node_edge_lists =
            if edge_rows >= ORB_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
                Some((
                    build_node_edge_lists(&graph.senders, node_rows),
                    build_node_edge_lists(&graph.receivers, node_rows),
                ))
            } else {
                None
            };
        #[cfg(feature = "parallel")]
        let parallel_node_edge_list_refs: Option<OrbNodeEdgeListRefs<'_>> =
            if edge_rows >= ORB_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
                let (sender_lists, receiver_lists) =
                    parallel_node_edge_lists.as_ref().ok_or_else(|| {
                        OrbV3OmolError::InvalidSpec(
                            "ORB parallel edge lists are missing for parallel scatter".to_owned(),
                        )
                    })?;
                Some((sender_lists.as_slice(), receiver_lists.as_slice()))
            } else {
                None
            };
        for (layer, weights) in weights_by_layer.iter().enumerate() {
            if retain_all_layer_outputs {
                let cond_nodes = orb_linear_forward(
                    &features.node_conditioning,
                    node_rows,
                    LATENT_DIM,
                    weights.cond_node.weight,
                    weights.cond_node.bias,
                    LATENT_DIM,
                )?;
                for (node_value, cond_value) in nodes.iter_mut().zip(&cond_nodes) {
                    *node_value += *cond_value;
                }
            } else {
                let cond_node = orb_linear_forward(
                    &features.node_conditioning[..LATENT_DIM],
                    1,
                    LATENT_DIM,
                    weights.cond_node.weight,
                    weights.cond_node.bias,
                    LATENT_DIM,
                )?;
                for node in nodes.chunks_exact_mut(LATENT_DIM) {
                    for (node_value, cond_value) in node.iter_mut().zip(&cond_node) {
                        *node_value += *cond_value;
                    }
                }
            }

            let (receive_attn, send_attn) = orb_edge_attention_pair_forward(
                &edges,
                features,
                &weights.attention_pair_weight,
                &weights.attention_pair_bias,
            )?;

            let (updated_edges, edge_mlp) = orb_edge_mlp_rms_norm_forward_with_cache(
                &edges,
                &nodes,
                &graph.senders,
                &graph.receivers,
                node_rows,
                weights.edge_mlp,
            )?;

            orb_profile_scatter_reduction!({
                orb_profile_kernel!(
                    "gnn_forward_node_input_fill",
                    node_rows,
                    ORB_GNN_INPUT_DIM,
                    {
                        node_input.fill(0.0);
                    }
                );
                orb_profile_kernel!("gnn_forward_node_input_self", node_rows, LATENT_DIM, {
                    #[cfg(feature = "parallel")]
                    if node_rows >= ORB_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
                        node_input
                            .par_chunks_exact_mut(ORB_GNN_INPUT_DIM)
                            .zip(nodes.par_chunks_exact(LATENT_DIM))
                            .for_each(|(dst, node)| {
                                dst[..LATENT_DIM].copy_from_slice(node);
                            });
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        for node in 0..node_rows {
                            let dst = &mut node_input
                                [node * ORB_GNN_INPUT_DIM..(node + 1) * ORB_GNN_INPUT_DIM];
                            dst[..LATENT_DIM].copy_from_slice(
                                &nodes[node * LATENT_DIM..(node + 1) * LATENT_DIM],
                            );
                        }
                    }
                    #[cfg(feature = "parallel")]
                    if node_rows < ORB_PARALLEL_ROW_THRESHOLD || rayon::current_num_threads() <= 1 {
                        for node in 0..node_rows {
                            let dst = &mut node_input
                                [node * ORB_GNN_INPUT_DIM..(node + 1) * ORB_GNN_INPUT_DIM];
                            dst[..LATENT_DIM].copy_from_slice(
                                &nodes[node * LATENT_DIM..(node + 1) * LATENT_DIM],
                            );
                        }
                    }
                });
                orb_profile_kernel!("gnn_forward_edge_to_node", edge_rows, LATENT_DIM, {
                    #[cfg(feature = "parallel")]
                    if edge_rows >= ORB_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
                        if let Some((sender_lists, receiver_lists)) = parallel_node_edge_list_refs {
                            node_input
                                .par_chunks_exact_mut(ORB_GNN_INPUT_DIM)
                                .enumerate()
                                .for_each(|(node, node_row)| {
                                    for &edge in &sender_lists[node] {
                                        let eu = &updated_edges
                                            [edge * LATENT_DIM..(edge + 1) * LATENT_DIM];
                                        let attn = send_attn.values[edge];
                                        orb_add_scaled_row_in_place(
                                            &mut node_row[2 * LATENT_DIM..3 * LATENT_DIM],
                                            eu,
                                            attn,
                                        );
                                    }
                                    for &edge in &receiver_lists[node] {
                                        let eu = &updated_edges
                                            [edge * LATENT_DIM..(edge + 1) * LATENT_DIM];
                                        let attn = receive_attn.values[edge];
                                        orb_add_scaled_row_in_place(
                                            &mut node_row[LATENT_DIM..2 * LATENT_DIM],
                                            eu,
                                            attn,
                                        );
                                    }
                                });
                        }
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        for edge in 0..edge_rows {
                            let sender = graph.senders[edge];
                            let receiver = graph.receivers[edge];
                            let edge_update =
                                &updated_edges[edge * LATENT_DIM..(edge + 1) * LATENT_DIM];
                            let sent_start = sender * ORB_GNN_INPUT_DIM + 2 * LATENT_DIM;
                            let received_start = receiver * ORB_GNN_INPUT_DIM + LATENT_DIM;
                            orb_add_scaled_row_in_place(
                                &mut node_input[sent_start..sent_start + LATENT_DIM],
                                edge_update,
                                send_attn.values[edge],
                            );
                            orb_add_scaled_row_in_place(
                                &mut node_input[received_start..received_start + LATENT_DIM],
                                edge_update,
                                receive_attn.values[edge],
                            );
                        }
                    }
                    #[cfg(feature = "parallel")]
                    if edge_rows < ORB_PARALLEL_ROW_THRESHOLD || rayon::current_num_threads() <= 1 {
                        for edge in 0..edge_rows {
                            let sender = graph.senders[edge];
                            let receiver = graph.receivers[edge];
                            let edge_update =
                                &updated_edges[edge * LATENT_DIM..(edge + 1) * LATENT_DIM];
                            let sent_start = sender * ORB_GNN_INPUT_DIM + 2 * LATENT_DIM;
                            let received_start = receiver * ORB_GNN_INPUT_DIM + LATENT_DIM;
                            orb_add_scaled_row_in_place(
                                &mut node_input[sent_start..sent_start + LATENT_DIM],
                                edge_update,
                                send_attn.values[edge],
                            );
                            orb_add_scaled_row_in_place(
                                &mut node_input[received_start..received_start + LATENT_DIM],
                                edge_update,
                                receive_attn.values[edge],
                            );
                        }
                    }
                });
            });
            let (updated_nodes, node_mlp) = orb_mlp_rms_norm_forward_with_cache(
                &node_input,
                node_rows,
                ORB_GNN_INPUT_DIM,
                weights.node_mlp,
            )?;

            for (edge_value, update) in edges.iter_mut().zip(&updated_edges) {
                *edge_value += *update;
            }
            for (node_value, update) in nodes.iter_mut().zip(&updated_nodes) {
                *node_value += *update;
            }
            let is_final_layer = layer + 1 == MESSAGE_PASSING_STEPS;
            layers.push(OrbV3OmolGnnLayerOutput {
                nodes: if retain_all_layer_outputs || is_final_layer {
                    nodes.clone()
                } else {
                    Vec::new()
                },
                edges: if retain_all_layer_outputs {
                    edges.clone()
                } else {
                    Vec::new()
                },
            });
            layer_caches.push(OrbV3OmolGnnLayerCache {
                receive_attn: receive_attn.values,
                receive_attn_sigmoid: receive_attn.sigmoid,
                send_attn: send_attn.values,
                send_attn_sigmoid: send_attn.sigmoid,
                updated_edges,
                edge_mlp,
                node_mlp,
            });
        }
        Ok(OrbV3OmolGnnOutput {
            layers,
            layer_caches,
        })
    }

    pub fn gnn_input_vjp(
        &self,
        graph: &OrbV3OmolPreparedGraph,
        features: &OrbV3OmolInputFeatures,
        encoded: &OrbV3OmolEncodedFeatures,
        gnn: &OrbV3OmolGnnOutput,
        final_layer_cotangents: &OrbV3OmolGnnLayerOutput,
    ) -> Result<OrbV3OmolGnnInputCotangents, OrbV3OmolError> {
        let weights = self.gnn_layer_weight_plan()?;
        Self::gnn_input_vjp_impl(
            graph,
            features,
            encoded,
            gnn,
            &weights,
            OrbGnnInitialCotangents {
                nodes: final_layer_cotangents.nodes.clone(),
                edges: final_layer_cotangents.edges.clone(),
            },
            true,
        )
    }

    fn gnn_input_vjp_for_geometry_with_weights(
        &self,
        graph: &OrbV3OmolPreparedGraph,
        features: &OrbV3OmolInputFeatures,
        encoded: &OrbV3OmolEncodedFeatures,
        gnn: &OrbV3OmolGnnOutput,
        weights_by_layer: &[OrbGnnLayerWeights<'_>],
        final_node_cotangents: Vec<f64>,
    ) -> Result<OrbV3OmolGnnInputCotangents, OrbV3OmolError> {
        Self::gnn_input_vjp_impl(
            graph,
            features,
            encoded,
            gnn,
            weights_by_layer,
            OrbGnnInitialCotangents {
                nodes: final_node_cotangents,
                edges: vec![0.0; graph.edge_count() * LATENT_DIM],
            },
            false,
        )
    }

    fn gnn_input_vjp_impl(
        graph: &OrbV3OmolPreparedGraph,
        features: &OrbV3OmolInputFeatures,
        encoded: &OrbV3OmolEncodedFeatures,
        gnn: &OrbV3OmolGnnOutput,
        weights_by_layer: &[OrbGnnLayerWeights<'_>],
        initial_cotangents: OrbGnnInitialCotangents,
        compute_node_conditioning: bool,
    ) -> Result<OrbV3OmolGnnInputCotangents, OrbV3OmolError> {
        let OrbGnnInitialCotangents {
            nodes: mut grad_nodes,
            edges: mut grad_edges,
        } = initial_cotangents;
        validate_graph_lengths(graph)?;
        let node_rows = graph.n_atoms;
        let edge_rows = graph.edge_count();
        if weights_by_layer.len() != MESSAGE_PASSING_STEPS {
            return invalid(format!(
                "ORB GNN VJP weight plan requires {MESSAGE_PASSING_STEPS} layers, got {}",
                weights_by_layer.len()
            ));
        }
        expect_len(
            "ORB encoded node features",
            &encoded.nodes,
            node_rows * LATENT_DIM,
        )?;
        expect_len(
            "ORB encoded edge features",
            &encoded.edges,
            edge_rows * LATENT_DIM,
        )?;
        expect_len(
            "ORB node conditioning",
            &features.node_conditioning,
            node_rows * LATENT_DIM,
        )?;
        expect_len(
            "ORB attention cutoffs",
            &features.attention_cutoffs,
            edge_rows,
        )?;
        if gnn.layers.len() != MESSAGE_PASSING_STEPS {
            return invalid(format!(
                "ORB GNN VJP requires {MESSAGE_PASSING_STEPS} layers, got {}",
                gnn.layers.len()
            ));
        }
        if gnn.layer_caches.len() != MESSAGE_PASSING_STEPS {
            return invalid(format!(
                "ORB GNN VJP requires {MESSAGE_PASSING_STEPS} layer caches, got {}",
                gnn.layer_caches.len()
            ));
        }
        for (layer_index, layer_output) in gnn.layers.iter().enumerate() {
            if !layer_output.nodes.is_empty() || layer_index + 1 == MESSAGE_PASSING_STEPS {
                expect_len(
                    &format!("ORB GNN layer {layer_index} node features"),
                    &layer_output.nodes,
                    node_rows * LATENT_DIM,
                )?;
            }
            if !layer_output.edges.is_empty() {
                expect_len(
                    &format!("ORB GNN layer {layer_index} edge features"),
                    &layer_output.edges,
                    edge_rows * LATENT_DIM,
                )?;
            }
        }
        expect_len(
            "ORB final GNN node cotangents",
            &grad_nodes,
            node_rows * LATENT_DIM,
        )?;
        expect_len(
            "ORB final GNN edge cotangents",
            &grad_edges,
            edge_rows * LATENT_DIM,
        )?;

        let mut grad_node_conditioning = if compute_node_conditioning {
            vec![0.0; node_rows * LATENT_DIM]
        } else {
            Vec::new()
        };
        let mut grad_attention_cutoffs = vec![0.0; edge_rows];
        let mut grad_updated_edges_workspace = vec![0.0; edge_rows * LATENT_DIM];
        let mut grad_receive_attn = vec![0.0; edge_rows];
        let mut grad_send_attn = vec![0.0; edge_rows];
        #[cfg(feature = "parallel")]
        let parallel_node_edge_lists =
            if edge_rows >= ORB_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
                Some((
                    build_node_edge_lists(&graph.senders, node_rows),
                    build_node_edge_lists(&graph.receivers, node_rows),
                ))
            } else {
                None
            };
        #[cfg(feature = "parallel")]
        let parallel_node_edge_list_refs: Option<OrbNodeEdgeListRefs<'_>> =
            if edge_rows >= ORB_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
                let (sender_lists, receiver_lists) =
                    parallel_node_edge_lists.as_ref().ok_or_else(|| {
                        OrbV3OmolError::InvalidSpec(
                            "ORB parallel edge lists are missing for parallel VJP scatter"
                                .to_owned(),
                        )
                    })?;
                Some((sender_lists.as_slice(), receiver_lists.as_slice()))
            } else {
                None
            };
        #[cfg(not(feature = "parallel"))]
        let parallel_node_edge_list_refs = None;

        for (layer, weights) in weights_by_layer.iter().enumerate().rev() {
            let cache = gnn.layer_caches.get(layer).ok_or_else(|| {
                OrbV3OmolError::InvalidSpec(format!("ORB GNN layer {layer} cache is missing"))
            })?;
            expect_len(
                &format!("ORB GNN layer {layer} receive attention cache"),
                &cache.receive_attn,
                edge_rows,
            )?;
            expect_len(
                &format!("ORB GNN layer {layer} receive attention sigmoid cache"),
                &cache.receive_attn_sigmoid,
                edge_rows,
            )?;
            expect_len(
                &format!("ORB GNN layer {layer} send attention cache"),
                &cache.send_attn,
                edge_rows,
            )?;
            expect_len(
                &format!("ORB GNN layer {layer} send attention sigmoid cache"),
                &cache.send_attn_sigmoid,
                edge_rows,
            )?;
            expect_len(
                &format!("ORB GNN layer {layer} updated edge cache"),
                &cache.updated_edges,
                edge_rows * LATENT_DIM,
            )?;
            let grad_node_input = orb_mlp_rms_norm_input_vjp_cached(
                &cache.node_mlp,
                &grad_nodes,
                node_rows,
                ORB_GNN_INPUT_DIM,
                weights.node_mlp,
            )?;
            let mut grad_conditioned_nodes = grad_nodes;
            grad_updated_edges_workspace.copy_from_slice(&grad_edges);
            let mut grad_edges_in = grad_edges;
            orb_profile_scatter_reduction!({
                orb_profile_kernel!("gnn_vjp_node_input_conditioned", node_rows, LATENT_DIM, {
                    for node in 0..node_rows {
                        let grad_row = &grad_node_input
                            [node * ORB_GNN_INPUT_DIM..(node + 1) * ORB_GNN_INPUT_DIM];
                        let conditioned_grad =
                            &mut grad_conditioned_nodes[node * LATENT_DIM..(node + 1) * LATENT_DIM];
                        orb_add_row_in_place(conditioned_grad, &grad_row[..LATENT_DIM]);
                    }
                });
            });

            grad_receive_attn.fill(0.0);
            grad_send_attn.fill(0.0);
            orb_profile_scatter_reduction!({
                orb_profile_kernel!("gnn_vjp_attention_edge_scatter", edge_rows, LATENT_DIM, {
                    #[cfg(feature = "parallel")]
                    if edge_rows >= ORB_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
                        grad_updated_edges_workspace
                            .par_chunks_exact_mut(LATENT_DIM)
                            .zip(grad_send_attn.par_iter_mut())
                            .zip(grad_receive_attn.par_iter_mut())
                            .enumerate()
                            .for_each(
                                |(edge, ((edge_grad, grad_send_attn), grad_receive_attn))| {
                                    let sender = graph.senders[edge];
                                    let receiver = graph.receivers[edge];
                                    let edge_update = &cache.updated_edges
                                        [edge * LATENT_DIM..(edge + 1) * LATENT_DIM];
                                    let sent_grad = &grad_node_input[sender * ORB_GNN_INPUT_DIM
                                        + 2 * LATENT_DIM
                                        ..sender * ORB_GNN_INPUT_DIM + 3 * LATENT_DIM];
                                    let received_grad =
                                        &grad_node_input[receiver * ORB_GNN_INPUT_DIM + LATENT_DIM
                                            ..receiver * ORB_GNN_INPUT_DIM + 2 * LATENT_DIM];
                                    let (send_dot, receive_dot) =
                                        orb_add_two_scaled_rows_and_two_dots(
                                            edge_grad,
                                            sent_grad,
                                            cache.send_attn[edge],
                                            received_grad,
                                            cache.receive_attn[edge],
                                            edge_update,
                                        );
                                    *grad_send_attn += send_dot;
                                    *grad_receive_attn += receive_dot;
                                },
                            );
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        for edge in 0..edge_rows {
                            let sender = graph.senders[edge];
                            let receiver = graph.receivers[edge];
                            let edge_update =
                                &cache.updated_edges[edge * LATENT_DIM..(edge + 1) * LATENT_DIM];
                            let edge_grad = &mut grad_updated_edges_workspace
                                [edge * LATENT_DIM..(edge + 1) * LATENT_DIM];
                            let sent_grad = &grad_node_input[sender * ORB_GNN_INPUT_DIM
                                + 2 * LATENT_DIM
                                ..sender * ORB_GNN_INPUT_DIM + 3 * LATENT_DIM];
                            let received_grad = &grad_node_input[receiver * ORB_GNN_INPUT_DIM
                                + LATENT_DIM
                                ..receiver * ORB_GNN_INPUT_DIM + 2 * LATENT_DIM];
                            let (send_dot, receive_dot) = orb_add_two_scaled_rows_and_two_dots(
                                edge_grad,
                                sent_grad,
                                cache.send_attn[edge],
                                received_grad,
                                cache.receive_attn[edge],
                                edge_update,
                            );
                            grad_send_attn[edge] += send_dot;
                            grad_receive_attn[edge] += receive_dot;
                        }
                    }
                    #[cfg(feature = "parallel")]
                    if edge_rows < ORB_PARALLEL_ROW_THRESHOLD || rayon::current_num_threads() <= 1 {
                        for edge in 0..edge_rows {
                            let sender = graph.senders[edge];
                            let receiver = graph.receivers[edge];
                            let edge_update =
                                &cache.updated_edges[edge * LATENT_DIM..(edge + 1) * LATENT_DIM];
                            let edge_grad = &mut grad_updated_edges_workspace
                                [edge * LATENT_DIM..(edge + 1) * LATENT_DIM];
                            let sent_grad = &grad_node_input[sender * ORB_GNN_INPUT_DIM
                                + 2 * LATENT_DIM
                                ..sender * ORB_GNN_INPUT_DIM + 3 * LATENT_DIM];
                            let received_grad = &grad_node_input[receiver * ORB_GNN_INPUT_DIM
                                + LATENT_DIM
                                ..receiver * ORB_GNN_INPUT_DIM + 2 * LATENT_DIM];
                            let (send_dot, receive_dot) = orb_add_two_scaled_rows_and_two_dots(
                                edge_grad,
                                sent_grad,
                                cache.send_attn[edge],
                                received_grad,
                                cache.receive_attn[edge],
                                edge_update,
                            );
                            grad_send_attn[edge] += send_dot;
                            grad_receive_attn[edge] += receive_dot;
                        }
                    }
                });
            });

            let grad_edge_mlp = orb_edge_mlp_rms_norm_input_vjp_cached(
                &cache.edge_mlp,
                &grad_updated_edges_workspace,
                &graph.senders,
                &graph.receivers,
                node_rows,
                weights.edge_mlp,
                parallel_node_edge_list_refs,
            )?;
            orb_profile_scatter_reduction!({
                orb_profile_kernel!("gnn_vjp_edge_mlp_cotangent_add", edge_rows, LATENT_DIM, {
                    orb_add_row_in_place(&mut grad_edges_in, &grad_edge_mlp.edges);
                    orb_add_row_in_place(&mut grad_conditioned_nodes, &grad_edge_mlp.nodes);
                });
            });

            orb_edge_attention_pair_input_vjp(
                &features.attention_cutoffs,
                weights.receive_attn,
                weights.send_attn,
                OrbEdgeAttentionPairVjpBuffers {
                    receive_sigmoid: &cache.receive_attn_sigmoid,
                    send_sigmoid: &cache.send_attn_sigmoid,
                    grad_receive_attention: &grad_receive_attn,
                    grad_send_attention: &grad_send_attn,
                    grad_edges: &mut grad_edges_in,
                    grad_attention_cutoffs: &mut grad_attention_cutoffs,
                },
            )?;

            if compute_node_conditioning {
                let grad_conditioning_layer = orb_linear_input_vjp(
                    &grad_conditioned_nodes,
                    node_rows,
                    LATENT_DIM,
                    weights.cond_node.weight,
                    LATENT_DIM,
                )?;
                for (dst, src) in grad_node_conditioning
                    .iter_mut()
                    .zip(grad_conditioning_layer)
                {
                    *dst += src;
                }
            }

            grad_nodes = grad_conditioned_nodes;
            grad_edges = grad_edges_in;
        }

        Ok(OrbV3OmolGnnInputCotangents {
            encoded: OrbV3OmolEncodedFeatures {
                nodes: grad_nodes,
                edges: grad_edges,
            },
            node_conditioning: grad_node_conditioning,
            attention_cutoffs: grad_attention_cutoffs,
        })
    }

    pub fn energy_from_gnn(
        &self,
        graph: &OrbV3OmolPreparedGraph,
        gnn: &OrbV3OmolGnnOutput,
    ) -> Result<OrbV3OmolEnergyBreakdown, OrbV3OmolError> {
        validate_graph_lengths(graph)?;
        let last = gnn.layers.last().ok_or_else(|| {
            OrbV3OmolError::InvalidSpec("ORB GNN output has no layers".to_owned())
        })?;
        expect_len(
            "ORB final node features",
            &last.nodes,
            graph.n_atoms * LATENT_DIM,
        )?;
        let mean_nodes = mean_node_features(&last.nodes, graph.n_atoms)?;
        let hidden = orb_linear_silu_forward(
            &mean_nodes,
            1,
            LATENT_DIM,
            self.state_tensor_f64(
                &format!("{ORB_ENERGY_HEAD_PREFIX}.mlp.NN-0.weight"),
                &[HEAD_MLP_HIDDEN_DIM, LATENT_DIM],
            )?,
            self.state_tensor_f64(
                &format!("{ORB_ENERGY_HEAD_PREFIX}.mlp.NN-0.bias"),
                &[HEAD_MLP_HIDDEN_DIM],
            )?,
            HEAD_MLP_HIDDEN_DIM,
        )?;
        let normalized = orb_linear_forward(
            &hidden,
            1,
            HEAD_MLP_HIDDEN_DIM,
            self.state_tensor_f64(
                &format!("{ORB_ENERGY_HEAD_PREFIX}.mlp.NN-1.weight"),
                &[1, HEAD_MLP_HIDDEN_DIM],
            )?,
            self.state_tensor_f64(&format!("{ORB_ENERGY_HEAD_PREFIX}.mlp.NN-1.bias"), &[1])?,
            1,
        )?[0];
        let running_mean = self.state_tensor_f64(
            &format!("{ORB_ENERGY_HEAD_PREFIX}.normalizer.bn.running_mean"),
            &[1],
        )?[0];
        let running_var = self.state_tensor_f64(
            &format!("{ORB_ENERGY_HEAD_PREFIX}.normalizer.bn.running_var"),
            &[1],
        )?[0];
        if !running_var.is_finite() || running_var < 0.0 {
            return invalid(format!(
                "ORB energy running_var must be finite and non-negative, got {running_var}"
            ));
        }
        let reference_energy = self.reference_energy_ev(graph)?;
        let denormalized_without_zbl = (normalized * running_var.sqrt() + running_mean)
            * graph.n_atoms as f64
            + reference_energy;
        let zbl_energy = self.zbl_energy_ev(graph)?;
        Ok(OrbV3OmolEnergyBreakdown {
            normalized_energy_head: normalized,
            reference_energy,
            denormalized_without_zbl,
            zbl_energy,
            total_energy: denormalized_without_zbl + zbl_energy,
        })
    }

    pub fn energy_tail_vjp_from_gnn(
        &self,
        graph: &OrbV3OmolPreparedGraph,
        gnn: &OrbV3OmolGnnOutput,
    ) -> Result<OrbV3OmolEnergyTailVjp, OrbV3OmolError> {
        validate_graph_lengths(graph)?;
        let last = gnn.layers.last().ok_or_else(|| {
            OrbV3OmolError::InvalidSpec("ORB GNN output has no layers".to_owned())
        })?;
        expect_len(
            "ORB final node features",
            &last.nodes,
            graph.n_atoms * LATENT_DIM,
        )?;
        let mean_nodes = mean_node_features(&last.nodes, graph.n_atoms)?;
        let layer0_weight = self.state_tensor_f64(
            &format!("{ORB_ENERGY_HEAD_PREFIX}.mlp.NN-0.weight"),
            &[HEAD_MLP_HIDDEN_DIM, LATENT_DIM],
        )?;
        let layer0_bias = self.state_tensor_f64(
            &format!("{ORB_ENERGY_HEAD_PREFIX}.mlp.NN-0.bias"),
            &[HEAD_MLP_HIDDEN_DIM],
        )?;
        let layer1_weight = self.state_tensor_f64(
            &format!("{ORB_ENERGY_HEAD_PREFIX}.mlp.NN-1.weight"),
            &[1, HEAD_MLP_HIDDEN_DIM],
        )?;
        let layer1_bias =
            self.state_tensor_f64(&format!("{ORB_ENERGY_HEAD_PREFIX}.mlp.NN-1.bias"), &[1])?;
        let pre0 = orb_linear_forward(
            &mean_nodes,
            1,
            LATENT_DIM,
            layer0_weight,
            layer0_bias,
            HEAD_MLP_HIDDEN_DIM,
        )?;
        let hidden = pre0
            .iter()
            .map(|&value| orb_silu(value))
            .collect::<Vec<_>>();
        let normalized = orb_linear_forward(
            &hidden,
            1,
            HEAD_MLP_HIDDEN_DIM,
            layer1_weight,
            layer1_bias,
            1,
        )?[0];
        let running_mean = self.state_tensor_f64(
            &format!("{ORB_ENERGY_HEAD_PREFIX}.normalizer.bn.running_mean"),
            &[1],
        )?[0];
        let running_var = self.state_tensor_f64(
            &format!("{ORB_ENERGY_HEAD_PREFIX}.normalizer.bn.running_var"),
            &[1],
        )?[0];
        if !running_var.is_finite() || running_var < 0.0 {
            return invalid(format!(
                "ORB energy running_var must be finite and non-negative, got {running_var}"
            ));
        }
        let reference_energy = self.reference_energy_ev(graph)?;
        let energy_scale = running_var.sqrt() * graph.n_atoms as f64;
        let denormalized_without_zbl =
            normalized * energy_scale + running_mean * graph.n_atoms as f64 + reference_energy;
        let (zbl_energy, zbl_forces) = self.zbl_energy_and_forces_ev(graph)?;
        let mut grad_pre0 = vec![0.0; HEAD_MLP_HIDDEN_DIM];
        for channel in 0..HEAD_MLP_HIDDEN_DIM {
            grad_pre0[channel] =
                energy_scale * layer1_weight[channel] * orb_silu_derivative(pre0[channel]);
        }
        let mut grad_mean = vec![0.0; LATENT_DIM];
        for hidden_channel in 0..HEAD_MLP_HIDDEN_DIM {
            let grad = grad_pre0[hidden_channel];
            let weight_row =
                &layer0_weight[hidden_channel * LATENT_DIM..(hidden_channel + 1) * LATENT_DIM];
            for channel in 0..LATENT_DIM {
                grad_mean[channel] += grad * weight_row[channel];
            }
        }
        let mut final_node_cotangents = vec![0.0; graph.n_atoms * LATENT_DIM];
        let mean_scale = 1.0 / graph.n_atoms as f64;
        for atom in 0..graph.n_atoms {
            let row = &mut final_node_cotangents[atom * LATENT_DIM..(atom + 1) * LATENT_DIM];
            for channel in 0..LATENT_DIM {
                row[channel] = grad_mean[channel] * mean_scale;
            }
        }
        Ok(OrbV3OmolEnergyTailVjp {
            breakdown: OrbV3OmolEnergyBreakdown {
                normalized_energy_head: normalized,
                reference_energy,
                denormalized_without_zbl,
                zbl_energy,
                total_energy: denormalized_without_zbl + zbl_energy,
            },
            final_node_cotangents,
            zbl_forces,
        })
    }

    pub fn zbl_energy_ev(&self, graph: &OrbV3OmolPreparedGraph) -> Result<f64, OrbV3OmolError> {
        validate_graph_lengths(graph)?;
        if graph.n_atoms == 0 {
            return invalid("ORB ZBL graph must contain at least one atom");
        }
        let mut edge_sum = 0.0;
        for edge in 0..graph.edge_count() {
            let sender = graph.senders[edge];
            let receiver = graph.receivers[edge];
            if sender >= graph.n_atoms || receiver >= graph.n_atoms {
                return invalid(format!(
                    "ORB ZBL edge {edge} sender/receiver out of bounds: {sender}->{receiver}, nodes {}",
                    graph.n_atoms
                ));
            }
            let distance = vec3_norm(graph.edge_vectors[edge]);
            if !distance.is_finite() || distance <= 0.0 {
                return invalid(format!(
                    "ORB ZBL edge {edge} distance must be positive and finite, got {distance}"
                ));
            }
            let z_sender = graph.atomic_numbers[sender] + 1;
            let z_receiver = graph.atomic_numbers[receiver] + 1;
            let a = ORB_ZBL_A_PREFACTOR * ORB_ZBL_BOHR_ANGSTROM
                / ((z_sender as f64).powf(ORB_ZBL_A_EXP) + (z_receiver as f64).powf(ORB_ZBL_A_EXP));
            let r_over_a = distance / a;
            let phi = ORB_ZBL_C
                .iter()
                .zip(ORB_ZBL_D)
                .map(|(&c, d)| c * (-d * r_over_a).exp())
                .sum::<f64>();
            let raw =
                ORB_ZBL_COULOMB_EV_ANGSTROM * z_sender as f64 * z_receiver as f64 * phi / distance;
            let cutoff_radius = f64::from(mace_covalent_radius(z_sender)? as f32)
                + f64::from(mace_covalent_radius(z_receiver)? as f32);
            edge_sum += 0.5 * raw * orb_zbl_cutoff(distance, cutoff_radius);
        }
        Ok(edge_sum / graph.n_atoms as f64)
    }

    pub fn zbl_energy_and_forces_ev(
        &self,
        graph: &OrbV3OmolPreparedGraph,
    ) -> Result<(f64, Vec<Vec3>), OrbV3OmolError> {
        validate_graph_lengths(graph)?;
        if graph.n_atoms == 0 {
            return invalid("ORB ZBL graph must contain at least one atom");
        }
        let mut edge_sum = 0.0;
        let mut forces = vec![[0.0; 3]; graph.n_atoms];
        let edge_scale = 0.5 / graph.n_atoms as f64;
        for edge in 0..graph.edge_count() {
            let sender = graph.senders[edge];
            let receiver = graph.receivers[edge];
            if sender >= graph.n_atoms || receiver >= graph.n_atoms {
                return invalid(format!(
                    "ORB ZBL edge {edge} sender/receiver out of bounds: {sender}->{receiver}, nodes {}",
                    graph.n_atoms
                ));
            }
            let vector = graph.edge_vectors[edge];
            let distance = vec3_norm(vector);
            if !distance.is_finite() || distance <= 0.0 {
                return invalid(format!(
                    "ORB ZBL edge {edge} distance must be positive and finite, got {distance}"
                ));
            }
            let z_sender = graph.atomic_numbers[sender] + 1;
            let z_receiver = graph.atomic_numbers[receiver] + 1;
            let a = ORB_ZBL_A_PREFACTOR * ORB_ZBL_BOHR_ANGSTROM
                / ((z_sender as f64).powf(ORB_ZBL_A_EXP) + (z_receiver as f64).powf(ORB_ZBL_A_EXP));
            let r_over_a = distance / a;
            let mut phi = 0.0;
            let mut dphi_dr = 0.0;
            for (&c, d) in ORB_ZBL_C.iter().zip(ORB_ZBL_D) {
                let value = c * (-d * r_over_a).exp();
                phi += value;
                dphi_dr += value * (-d / a);
            }
            let charge_scale = ORB_ZBL_COULOMB_EV_ANGSTROM * z_sender as f64 * z_receiver as f64;
            let raw = charge_scale * phi / distance;
            let raw_derivative = charge_scale * (dphi_dr / distance - phi / (distance * distance));
            let cutoff_radius = f64::from(mace_covalent_radius(z_sender)? as f32)
                + f64::from(mace_covalent_radius(z_receiver)? as f32);
            let cutoff = orb_zbl_cutoff(distance, cutoff_radius);
            let cutoff_derivative = orb_zbl_cutoff_derivative(distance, cutoff_radius);
            edge_sum += 0.5 * raw * cutoff;
            let d_edge_energy_dr = edge_scale * (raw_derivative * cutoff + raw * cutoff_derivative);
            let grad = [
                d_edge_energy_dr * vector[0] / distance,
                d_edge_energy_dr * vector[1] / distance,
                d_edge_energy_dr * vector[2] / distance,
            ];
            for (axis, value) in grad.iter().enumerate() {
                forces[sender][axis] += *value;
                forces[receiver][axis] -= *value;
            }
        }
        Ok((edge_sum / graph.n_atoms as f64, forces))
    }

    pub fn input_feature_geometry_edge_vector_vjp(
        &self,
        graph: &OrbV3OmolPreparedGraph,
        features: &OrbV3OmolInputFeatures,
        feature_cotangents: &OrbV3OmolInputFeatureCotangents,
        attention_cutoff_cotangents: &[f64],
    ) -> Result<Vec<Vec3>, OrbV3OmolError> {
        validate_graph_lengths(graph)?;
        let edge_rows = graph.edge_count();
        expect_len(
            "ORB feature geometry edge distances",
            &features.edge_distances,
            edge_rows,
        )?;
        expect_len(
            "ORB feature geometry attention cutoffs",
            &features.attention_cutoffs,
            edge_rows,
        )?;
        expect_len(
            "ORB feature geometry radial basis",
            &features.radial_basis,
            edge_rows * BESSEL_NUM_BASES,
        )?;
        expect_len(
            "ORB feature geometry angular basis",
            &features.angular_basis,
            edge_rows * ORB_ANGULAR_DIM,
        )?;
        expect_len(
            "ORB feature geometry edge-feature cotangents",
            &feature_cotangents.edge_features,
            edge_rows * ORB_EDGE_FEATURE_DIM,
        )?;
        expect_len(
            "ORB feature geometry attention-cutoff cotangents",
            attention_cutoff_cotangents,
            edge_rows,
        )?;

        let bessel_weights = self.state_tensor_f64(ORB_BESSEL_WEIGHTS, &[BESSEL_NUM_BASES])?;
        let r_max = self.state_tensor_f64(ORB_BESSEL_R_MAX, &[])?[0];
        let prefactor = self.state_tensor_f64(ORB_BESSEL_PREFACTOR, &[])?[0];
        if !r_max.is_finite() || r_max <= 0.0 {
            return invalid(format!(
                "ORB bessel r_max must be positive and finite, got {r_max}"
            ));
        }
        if !prefactor.is_finite() {
            return invalid(format!(
                "ORB bessel prefactor must be finite, got {prefactor}"
            ));
        }

        let sh_plan = SphericalHarmonicsPlan::component(SH_LMAX, true)?;
        let mut edge_vector_cotangents = vec![[0.0; 3]; edge_rows];
        for edge in 0..edge_rows {
            let vector = graph.edge_vectors[edge];
            let distance = vec3_norm(vector);
            if !distance.is_finite() || distance <= 0.0 {
                return invalid(format!(
                    "ORB edge {edge} distance must be positive and finite, got {distance}"
                ));
            }
            let cutoff = features.attention_cutoffs[edge];
            let radial_row =
                &features.radial_basis[edge * BESSEL_NUM_BASES..(edge + 1) * BESSEL_NUM_BASES];
            let angular_row =
                &features.angular_basis[edge * ORB_ANGULAR_DIM..(edge + 1) * ORB_ANGULAR_DIM];
            let edge_feature_grad = &feature_cotangents.edge_features
                [edge * ORB_EDGE_FEATURE_DIM..(edge + 1) * ORB_EDGE_FEATURE_DIM];

            let mut grad_cutoff = attention_cutoff_cotangents[edge];
            let mut grad_distance = 0.0;
            let mut grad_angular = [0.0; ORB_ANGULAR_DIM];
            for basis in 0..BESSEL_NUM_BASES {
                let radial = radial_row[basis];
                let mut grad_radial = 0.0;
                for angular in 0..ORB_ANGULAR_DIM {
                    let grad = edge_feature_grad[basis * ORB_ANGULAR_DIM + angular];
                    grad_cutoff += grad * radial * angular_row[angular];
                    grad_radial += grad * cutoff * angular_row[angular];
                    grad_angular[angular] += grad * cutoff * radial;
                }
                let weight = bessel_weights[basis];
                if !weight.is_finite() {
                    return invalid(format!(
                        "ORB bessel weight {basis} must be finite, got {weight}"
                    ));
                }
                let phase = weight * distance;
                let radial_derivative = prefactor * (weight * phase.cos() * distance - phase.sin())
                    / (distance * distance);
                grad_distance += grad_radial * radial_derivative;
            }
            grad_distance += grad_cutoff * orb_cutoff_derivative(distance, r_max);

            let mut grad_vector = [
                grad_distance * vector[0] / distance,
                grad_distance * vector[1] / distance,
                grad_distance * vector[2] / distance,
            ];
            sh_plan.vjp_edge_vector(vector, &grad_angular, &mut grad_vector)?;
            edge_vector_cotangents[edge] = grad_vector;
        }
        Ok(edge_vector_cotangents)
    }

    pub fn input_feature_geometry_forces(
        &self,
        graph: &OrbV3OmolPreparedGraph,
        features: &OrbV3OmolInputFeatures,
        feature_cotangents: &OrbV3OmolInputFeatureCotangents,
        attention_cutoff_cotangents: &[f64],
    ) -> Result<Vec<Vec3>, OrbV3OmolError> {
        let edge_vector_cotangents = self.input_feature_geometry_edge_vector_vjp(
            graph,
            features,
            feature_cotangents,
            attention_cutoff_cotangents,
        )?;
        let mut forces = vec![[0.0; 3]; graph.n_atoms];
        for edge in 0..edge_vector_cotangents.len() {
            let sender = graph.senders[edge];
            let receiver = graph.receivers[edge];
            if sender >= graph.n_atoms || receiver >= graph.n_atoms {
                return invalid(format!(
                    "ORB edge {edge} sender/receiver out of bounds: {sender}->{receiver}, nodes {}",
                    graph.n_atoms
                ));
            }
        }
        orb_profile_scatter_reduction!({
            for (edge, &bar_vector) in edge_vector_cotangents.iter().enumerate() {
                let sender = graph.senders[edge];
                let receiver = graph.receivers[edge];
                for (axis, value) in bar_vector.iter().enumerate() {
                    forces[sender][axis] += *value;
                    forces[receiver][axis] -= *value;
                }
            }
        });
        Ok(forces)
    }

    pub fn energy_for_non_pbc_structure(
        &self,
        structure: &Structure,
        total_charge: f64,
        spin_multiplicity: f64,
    ) -> Result<OrbV3OmolEnergyBreakdown, OrbV3OmolError> {
        if self.runtime_status() != RUNTIME_STATUS_STATE_EXPORT {
            return Err(OrbV3OmolError::RuntimeNotImplemented);
        }
        let graph = self.prepare_non_pbc_graph(structure, total_charge, spin_multiplicity)?;
        self.energy_for_prepared_graph(&graph)
    }

    pub fn energy_for_prepared_graph(
        &self,
        graph: &OrbV3OmolPreparedGraph,
    ) -> Result<OrbV3OmolEnergyBreakdown, OrbV3OmolError> {
        self.energy_for_prepared_graph_with_compute_dtype(graph, PredictionComputeDtype::Float64)
    }

    fn energy_for_prepared_graph_with_compute_dtype(
        &self,
        graph: &OrbV3OmolPreparedGraph,
        dtype: PredictionComputeDtype,
    ) -> Result<OrbV3OmolEnergyBreakdown, OrbV3OmolError> {
        validate_orb_compute_dtype(dtype)?;
        if let Some(plan) = duplicate_component_plan(graph)? {
            return self.energy_for_duplicate_components_with_compute_dtype(&plan, dtype);
        }
        self.energy_for_prepared_graph_core_with_compute_dtype(graph, dtype)
    }

    fn energy_for_prepared_graph_core_with_compute_dtype(
        &self,
        graph: &OrbV3OmolPreparedGraph,
        dtype: PredictionComputeDtype,
    ) -> Result<OrbV3OmolEnergyBreakdown, OrbV3OmolError> {
        if self.runtime_status() != RUNTIME_STATUS_STATE_EXPORT {
            return Err(OrbV3OmolError::RuntimeNotImplemented);
        }
        let features = self.input_features(graph)?;
        let (encoded, _) =
            self.encode_input_features_with_cache_for_compute_dtype(&features, dtype)?;
        let gnn_weights = self.gnn_layer_weight_plan_for_compute_dtype(dtype)?;
        let gnn = self.gnn_forward_for_energy_tail_with_weights(
            graph,
            &features,
            &encoded,
            &gnn_weights,
        )?;
        self.energy_from_gnn(graph, &gnn)
    }

    fn energy_for_duplicate_components_with_compute_dtype(
        &self,
        plan: &OrbDuplicateComponentPlan,
        dtype: PredictionComputeDtype,
    ) -> Result<OrbV3OmolEnergyBreakdown, OrbV3OmolError> {
        let count = plan.components.len();
        let representative =
            self.energy_for_prepared_graph_core_with_compute_dtype(&plan.representative, dtype)?;
        Ok(scale_duplicate_component_energy(representative, count))
    }

    pub fn energy_forces_for_non_pbc_structure(
        &self,
        structure: &Structure,
        total_charge: f64,
        spin_multiplicity: f64,
    ) -> Result<(OrbV3OmolEnergyBreakdown, Vec<Vec3>), OrbV3OmolError> {
        if self.runtime_status() != RUNTIME_STATUS_STATE_EXPORT {
            return Err(OrbV3OmolError::RuntimeNotImplemented);
        }
        let graph = self.prepare_non_pbc_graph(structure, total_charge, spin_multiplicity)?;
        self.energy_forces_for_prepared_graph(&graph)
    }

    pub fn energy_forces_for_prepared_graph(
        &self,
        graph: &OrbV3OmolPreparedGraph,
    ) -> Result<(OrbV3OmolEnergyBreakdown, Vec<Vec3>), OrbV3OmolError> {
        self.energy_forces_for_prepared_graph_with_compute_dtype(
            graph,
            PredictionComputeDtype::Float64,
        )
    }

    fn energy_forces_for_prepared_graph_with_compute_dtype(
        &self,
        graph: &OrbV3OmolPreparedGraph,
        dtype: PredictionComputeDtype,
    ) -> Result<(OrbV3OmolEnergyBreakdown, Vec<Vec3>), OrbV3OmolError> {
        validate_orb_compute_dtype(dtype)?;
        if let Some(plan) = duplicate_component_plan(graph)? {
            return self.energy_forces_for_duplicate_components_with_compute_dtype(&plan, dtype);
        }
        self.energy_forces_for_prepared_graph_core_with_compute_dtype(graph, dtype)
    }

    fn energy_forces_for_prepared_graph_core_with_compute_dtype(
        &self,
        graph: &OrbV3OmolPreparedGraph,
        dtype: PredictionComputeDtype,
    ) -> Result<(OrbV3OmolEnergyBreakdown, Vec<Vec3>), OrbV3OmolError> {
        if self.runtime_status() != RUNTIME_STATUS_STATE_EXPORT {
            return Err(OrbV3OmolError::RuntimeNotImplemented);
        }
        let features = self.input_features(graph)?;
        let (encoded, encoded_cache) =
            self.encode_input_features_with_cache_for_compute_dtype(&features, dtype)?;
        let gnn_weights = self.gnn_layer_weight_plan_for_compute_dtype(dtype)?;
        let gnn = self.gnn_forward_for_energy_tail_with_weights(
            graph,
            &features,
            &encoded,
            &gnn_weights,
        )?;
        let tail_vjp = self.energy_tail_vjp_from_gnn(graph, &gnn)?;
        let OrbV3OmolEnergyTailVjp {
            breakdown,
            final_node_cotangents,
            zbl_forces,
        } = tail_vjp;
        gnn.layers.last().ok_or_else(|| {
            OrbV3OmolError::InvalidSpec("ORB GNN output has no layers".to_owned())
        })?;
        let gnn_cotangents = self.gnn_input_vjp_for_geometry_with_weights(
            graph,
            &features,
            &encoded,
            &gnn,
            &gnn_weights,
            final_node_cotangents,
        )?;
        let feature_cotangents = self
            .encode_input_feature_edge_vjp_cached_for_geometry_with_compute_dtype(
                &features,
                &gnn_cotangents.encoded.edges,
                &encoded_cache,
                dtype,
            )?;
        let mut forces = self.input_feature_geometry_forces(
            graph,
            &features,
            &feature_cotangents,
            &gnn_cotangents.attention_cutoffs,
        )?;
        for (force, zbl_force) in forces.iter_mut().zip(&zbl_forces) {
            for axis in 0..3 {
                force[axis] += zbl_force[axis];
            }
        }
        Ok((breakdown, forces))
    }

    fn energy_forces_for_duplicate_components_with_compute_dtype(
        &self,
        plan: &OrbDuplicateComponentPlan,
        dtype: PredictionComputeDtype,
    ) -> Result<(OrbV3OmolEnergyBreakdown, Vec<Vec3>), OrbV3OmolError> {
        let count = plan.components.len();
        let (representative_energy, representative_forces) = self
            .energy_forces_for_prepared_graph_core_with_compute_dtype(
                &plan.representative,
                dtype,
            )?;
        let (_, representative_zbl_forces) = self.zbl_energy_and_forces_ev(&plan.representative)?;
        let energy = scale_duplicate_component_energy(representative_energy, count);
        let total_atoms = plan.components.iter().map(Vec::len).sum();
        let forces = reconstruct_duplicate_component_forces(
            plan,
            total_atoms,
            &representative_forces,
            &representative_zbl_forces,
        )?;
        Ok((energy, forces))
    }

    #[doc(hidden)]
    pub fn profile_energy_forces_for_prepared_graph(
        &self,
        graph: &OrbV3OmolPreparedGraph,
    ) -> Result<OrbV3OmolEnergyForcesProfile, OrbV3OmolError> {
        self.profile_energy_forces_for_prepared_graph_with_compute_dtype(
            graph,
            PredictionComputeDtype::Float64,
        )
    }

    #[doc(hidden)]
    pub fn profile_energy_forces_for_prepared_graph_with_compute_dtype(
        &self,
        graph: &OrbV3OmolPreparedGraph,
        dtype: PredictionComputeDtype,
    ) -> Result<OrbV3OmolEnergyForcesProfile, OrbV3OmolError> {
        validate_orb_compute_dtype(dtype)?;
        let total_start = Instant::now();
        if let Some(plan) = duplicate_component_plan(graph)? {
            return self.profile_energy_forces_for_duplicate_components(
                graph,
                &plan,
                dtype,
                total_start,
            );
        }
        let (mut profile, _) =
            self.profile_energy_forces_for_prepared_graph_core_with_compute_dtype(graph, dtype)?;
        profile.total_ms = elapsed_ms(total_start);
        Ok(profile)
    }

    fn profile_energy_forces_for_prepared_graph_core_with_compute_dtype(
        &self,
        graph: &OrbV3OmolPreparedGraph,
        dtype: PredictionComputeDtype,
    ) -> Result<(OrbV3OmolEnergyForcesProfile, OrbV3OmolEnergyBreakdown), OrbV3OmolError> {
        if self.runtime_status() != RUNTIME_STATUS_STATE_EXPORT {
            return Err(OrbV3OmolError::RuntimeNotImplemented);
        }
        validate_orb_compute_dtype(dtype)?;
        #[cfg(feature = "orb-profile-breakdown")]
        orb_profile_breakdown::reset();
        let total_start = Instant::now();

        let start = Instant::now();
        let features = self.input_features(graph)?;
        let input_features_ms = elapsed_ms(start);

        let start = Instant::now();
        let (encoded, encoded_cache) =
            self.encode_input_features_with_cache_for_compute_dtype(&features, dtype)?;
        let encoder_forward_ms = elapsed_ms(start);

        let start = Instant::now();
        let gnn_weights = self.gnn_layer_weight_plan_for_compute_dtype(dtype)?;
        let gnn_weight_plan_ms = elapsed_ms(start);

        let start = Instant::now();
        let gnn = self.gnn_forward_for_energy_tail_with_weights(
            graph,
            &features,
            &encoded,
            &gnn_weights,
        )?;
        let gnn_forward_ms = elapsed_ms(start);

        let start = Instant::now();
        let tail_vjp = self.energy_tail_vjp_from_gnn(graph, &gnn)?;
        let energy_tail_vjp_ms = elapsed_ms(start);

        let start = Instant::now();
        gnn.layers.last().ok_or_else(|| {
            OrbV3OmolError::InvalidSpec("ORB GNN output has no layers".to_owned())
        })?;
        let OrbV3OmolEnergyTailVjp {
            breakdown,
            final_node_cotangents,
            zbl_forces,
        } = tail_vjp;
        let final_cotangent_ms = elapsed_ms(start);

        let start = Instant::now();
        let gnn_cotangents = self.gnn_input_vjp_for_geometry_with_weights(
            graph,
            &features,
            &encoded,
            &gnn,
            &gnn_weights,
            final_node_cotangents,
        )?;
        let gnn_input_vjp_ms = elapsed_ms(start);

        let start = Instant::now();
        let feature_cotangents = self
            .encode_input_feature_edge_vjp_cached_for_geometry_with_compute_dtype(
                &features,
                &gnn_cotangents.encoded.edges,
                &encoded_cache,
                dtype,
            )?;
        let encoder_vjp_ms = elapsed_ms(start);

        let start = Instant::now();
        let mut forces = self.input_feature_geometry_forces(
            graph,
            &features,
            &feature_cotangents,
            &gnn_cotangents.attention_cutoffs,
        )?;
        let geometry_forces_ms = elapsed_ms(start);

        let start = Instant::now();
        for (force, zbl_force) in forces.iter_mut().zip(&zbl_forces) {
            for axis in 0..3 {
                force[axis] += zbl_force[axis];
            }
        }
        let zbl_force_accum_ms = elapsed_ms(start);
        let force_checksum = forces.iter().flatten().sum::<f64>();
        let force_vjp_ms = energy_tail_vjp_ms
            + final_cotangent_ms
            + gnn_input_vjp_ms
            + encoder_vjp_ms
            + geometry_forces_ms
            + zbl_force_accum_ms;
        #[cfg(feature = "orb-profile-breakdown")]
        let profile_breakdown = orb_profile_breakdown::snapshot();
        #[cfg(not(feature = "orb-profile-breakdown"))]
        let profile_breakdown = OrbProfileBreakdown::default();
        let total_ms = elapsed_ms(total_start);
        let force_l2_norm = forces
            .iter()
            .flatten()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let force_max_abs = forces
            .iter()
            .flatten()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);

        let profile = OrbV3OmolEnergyForcesProfile {
            atom_count: graph.n_atoms,
            edge_count: graph.edge_count(),
            total_ms,
            input_features_ms,
            encoder_forward_ms,
            gnn_weight_plan_ms,
            gnn_forward_ms,
            energy_tail_vjp_ms,
            final_cotangent_ms,
            gnn_input_vjp_ms,
            encoder_vjp_ms,
            geometry_forces_ms,
            zbl_force_accum_ms,
            force_vjp_ms,
            blas_ms: profile_breakdown.blas_ms,
            activation_ms: profile_breakdown.activation_ms,
            scatter_reduction_ms: profile_breakdown.scatter_reduction_ms,
            blas_shapes: profile_breakdown.blas_shapes,
            kernels: profile_breakdown.kernels,
            total_energy: breakdown.total_energy,
            force_checksum,
            force_l2_norm,
            force_max_abs,
            forces,
        };
        Ok((profile, breakdown))
    }

    fn profile_energy_forces_for_duplicate_components(
        &self,
        graph: &OrbV3OmolPreparedGraph,
        plan: &OrbDuplicateComponentPlan,
        dtype: PredictionComputeDtype,
        total_start: Instant,
    ) -> Result<OrbV3OmolEnergyForcesProfile, OrbV3OmolError> {
        let count = plan.components.len();
        let (mut profile, representative_energy) = self
            .profile_energy_forces_for_prepared_graph_core_with_compute_dtype(
                &plan.representative,
                dtype,
            )?;

        let start = Instant::now();
        let (_, representative_zbl_forces) = self.zbl_energy_and_forces_ev(&plan.representative)?;
        let energy = scale_duplicate_component_energy(representative_energy, count);
        let forces = reconstruct_duplicate_component_forces(
            plan,
            graph.n_atoms,
            &profile.forces,
            &representative_zbl_forces,
        )?;
        let duplicate_force_ms = elapsed_ms(start);
        let force_checksum = forces.iter().flatten().sum::<f64>();
        let force_l2_norm = forces
            .iter()
            .flatten()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let force_max_abs = forces
            .iter()
            .flatten()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max);

        profile.atom_count = graph.n_atoms;
        profile.edge_count = graph.edge_count();
        profile.total_ms = elapsed_ms(total_start);
        profile.zbl_force_accum_ms += duplicate_force_ms;
        profile.force_vjp_ms += duplicate_force_ms;
        profile.total_energy = energy.total_energy;
        profile.force_checksum = force_checksum;
        profile.force_l2_norm = force_l2_norm;
        profile.force_max_abs = force_max_abs;
        profile.forces = forces;
        Ok(profile)
    }

    pub fn predict_prepared(
        &self,
        prepared: &OrbV3OmolPreparedGraph,
        options: &PredictionOptions,
    ) -> Result<Prediction, OrbV3OmolError> {
        self.predict_prepared_with_compute_dtype(prepared, options, PredictionComputeDtype::Float64)
    }

    pub fn predict_prepared_with_compute_dtype(
        &self,
        prepared: &OrbV3OmolPreparedGraph,
        options: &PredictionOptions,
        dtype: PredictionComputeDtype,
    ) -> Result<Prediction, OrbV3OmolError> {
        validate_orb_compute_dtype(dtype)?;
        if options.compute_stress || options.compute_virial {
            return Err(OrbV3OmolError::RuntimeNotImplemented);
        }
        if !options.compute_energy && !options.compute_forces {
            return Ok(Prediction::default());
        }
        let (total_charge, spin_multiplicity) = charge_spin_from_prediction_options(options)?;
        if charge_spin_metadata_matches(
            prepared.total_charge,
            prepared.spin_multiplicity,
            total_charge,
            spin_multiplicity,
        ) {
            if options.compute_forces {
                let (energy, forces) =
                    self.energy_forces_for_prepared_graph_with_compute_dtype(prepared, dtype)?;
                return Ok(Prediction {
                    energy: options.compute_energy.then_some(energy.total_energy),
                    forces: Some(forces),
                    ..Prediction::default()
                });
            }
            let energy = self.energy_for_prepared_graph_with_compute_dtype(prepared, dtype)?;
            return Ok(Prediction {
                energy: Some(energy.total_energy),
                ..Prediction::default()
            });
        }
        let mut graph = prepared.clone();
        graph.total_charge = total_charge;
        graph.spin_multiplicity = spin_multiplicity;
        if options.compute_forces {
            let (energy, forces) =
                self.energy_forces_for_prepared_graph_with_compute_dtype(&graph, dtype)?;
            return Ok(Prediction {
                energy: options.compute_energy.then_some(energy.total_energy),
                forces: Some(forces),
                ..Prediction::default()
            });
        }
        let energy = self.energy_for_prepared_graph_with_compute_dtype(&graph, dtype)?;
        Ok(Prediction {
            energy: Some(energy.total_energy),
            ..Prediction::default()
        })
    }

    fn mlp_rms_norm_forward_with_cache_for_compute_dtype(
        &self,
        prefix: &str,
        input: &[f64],
        rows: usize,
        input_dim: usize,
        dtype: PredictionComputeDtype,
    ) -> Result<(Vec<f64>, OrbMlpRmsNormForwardCache), OrbV3OmolError> {
        orb_mlp_rms_norm_forward_with_cache(
            input,
            rows,
            input_dim,
            self.mlp_rms_norm_weights_for_compute_dtype(prefix, input_dim, dtype)?,
        )
    }

    fn mlp_rms_norm_weights(
        &self,
        prefix: &str,
        input_dim: usize,
    ) -> Result<OrbMlpRmsNormWeights<'_>, OrbV3OmolError> {
        self.mlp_rms_norm_weights_for_compute_dtype(
            prefix,
            input_dim,
            PredictionComputeDtype::Float64,
        )
    }

    fn mlp_rms_norm_weights_for_compute_dtype(
        &self,
        prefix: &str,
        input_dim: usize,
        dtype: PredictionComputeDtype,
    ) -> Result<OrbMlpRmsNormWeights<'_>, OrbV3OmolError> {
        #[cfg(all(
            feature = "orb-fixed-shape-dense",
            feature = "intel-mkl",
            target_os = "linux",
            target_arch = "x86_64"
        ))]
        let fixed_rows_packed = if prefix == ORB_ENCODER_NODE_PREFIX && input_dim == LATENT_DIM {
            self.encoder_node_mlp_packed.as_ref()
        } else if prefix == ORB_ENCODER_EDGE_PREFIX && input_dim == ORB_EDGE_FEATURE_DIM {
            self.encoder_edge_mlp_packed.as_ref()
        } else {
            None
        };
        #[cfg(all(
            feature = "orb-f32-edge-mlp",
            any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    any(feature = "intel-mkl", feature = "openblas"),
                    target_os = "linux",
                    target_arch = "x86_64"
                )
            )
        ))]
        let mlp_f32 = if dtype == PredictionComputeDtype::Float32 {
            if prefix == ORB_ENCODER_NODE_PREFIX && input_dim == LATENT_DIM {
                self.encoder_node_mlp_f32.as_ref()
            } else if prefix == ORB_ENCODER_EDGE_PREFIX && input_dim == ORB_EDGE_FEATURE_DIM {
                self.encoder_edge_mlp_f32.as_ref()
            } else {
                None
            }
        } else {
            None
        };
        #[cfg(not(all(
            feature = "orb-f32-edge-mlp",
            any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    any(feature = "intel-mkl", feature = "openblas"),
                    target_os = "linux",
                    target_arch = "x86_64"
                )
            )
        )))]
        let _ = dtype;
        Ok(OrbMlpRmsNormWeights {
            layer0_weight: self.state_tensor_f64(
                &format!("{prefix}.mlp.NN-0.weight"),
                &[ORB_ENCODER_HIDDEN_DIM, input_dim],
            )?,
            layer0_bias: self.state_tensor_f64(
                &format!("{prefix}.mlp.NN-0.bias"),
                &[ORB_ENCODER_HIDDEN_DIM],
            )?,
            layer1_weight: self.state_tensor_f64(
                &format!("{prefix}.mlp.NN-1.weight"),
                &[ORB_ENCODER_HIDDEN_DIM, ORB_ENCODER_HIDDEN_DIM],
            )?,
            layer1_bias: self.state_tensor_f64(
                &format!("{prefix}.mlp.NN-1.bias"),
                &[ORB_ENCODER_HIDDEN_DIM],
            )?,
            layer2_weight: self.state_tensor_f64(
                &format!("{prefix}.mlp.NN-2.weight"),
                &[LATENT_DIM, ORB_ENCODER_HIDDEN_DIM],
            )?,
            layer2_bias: self
                .state_tensor_f64(&format!("{prefix}.mlp.NN-2.bias"), &[LATENT_DIM])?,
            rms_weight: self
                .state_tensor_f64(&format!("{prefix}.layer_norm.weight"), &[LATENT_DIM])?,
            #[cfg(all(
                feature = "orb-fixed-shape-dense",
                feature = "intel-mkl",
                target_os = "linux",
                target_arch = "x86_64"
            ))]
            fixed_rows_packed,
            #[cfg(all(
                feature = "orb-fixed-shape-dense",
                feature = "intel-mkl",
                target_os = "linux",
                target_arch = "x86_64"
            ))]
            edge_packed: None,
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            mlp_f32,
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            edge_f32: None,
        })
    }

    fn mlp_rms_norm_input_vjp_cached_for_compute_dtype(
        &self,
        prefix: &str,
        cache: &OrbMlpRmsNormForwardCache,
        grad_output: &[f64],
        rows: usize,
        input_dim: usize,
        dtype: PredictionComputeDtype,
    ) -> Result<Vec<f64>, OrbV3OmolError> {
        orb_mlp_rms_norm_input_vjp_cached(
            cache,
            grad_output,
            rows,
            input_dim,
            self.mlp_rms_norm_weights_for_compute_dtype(prefix, input_dim, dtype)?,
        )
    }

    fn gnn_layer_weight_plan(&self) -> Result<Vec<OrbGnnLayerWeights<'_>>, OrbV3OmolError> {
        self.gnn_layer_weight_plan_for_compute_dtype(PredictionComputeDtype::Float64)
    }

    fn gnn_layer_weight_plan_for_compute_dtype(
        &self,
        dtype: PredictionComputeDtype,
    ) -> Result<Vec<OrbGnnLayerWeights<'_>>, OrbV3OmolError> {
        validate_orb_compute_dtype(dtype)?;
        if let Some(cache) = &self.gnn_weight_cache {
            return self.gnn_layer_weight_plan_from_cache(cache, dtype);
        }
        let mut weights = Vec::with_capacity(MESSAGE_PASSING_STEPS);
        for layer in 0..MESSAGE_PASSING_STEPS {
            let prefix = format!("model.gnn_stacks.{layer}");
            weights.push(self.gnn_layer_weights(&prefix)?);
        }
        Ok(weights)
    }

    fn gnn_layer_weight_plan_from_cache<'a>(
        &'a self,
        cache: &'a OrbGnnWeightPlanCache,
        dtype: PredictionComputeDtype,
    ) -> Result<Vec<OrbGnnLayerWeights<'a>>, OrbV3OmolError> {
        #[cfg(not(all(
            feature = "orb-f32-edge-mlp",
            any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    any(feature = "intel-mkl", feature = "openblas"),
                    target_os = "linux",
                    target_arch = "x86_64"
                )
            )
        )))]
        let _ = dtype;
        let mut weights = Vec::with_capacity(cache.layers.len());
        for layer_cache in &cache.layers {
            let edge_mlp = self.mlp_rms_norm_weights_from_keys(&layer_cache.edge_mlp_keys)?;
            #[cfg(all(
                feature = "orb-fixed-shape-dense",
                feature = "intel-mkl",
                target_os = "linux",
                target_arch = "x86_64"
            ))]
            let edge_mlp = OrbMlpRmsNormWeights {
                edge_packed: layer_cache.edge_mlp_packed.as_ref(),
                ..edge_mlp
            };
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            let edge_mlp = OrbMlpRmsNormWeights {
                edge_f32: if dtype == PredictionComputeDtype::Float32 {
                    layer_cache.edge_mlp_f32.as_ref()
                } else {
                    None
                },
                ..edge_mlp
            };
            let node_mlp = self.mlp_rms_norm_weights_from_keys(&layer_cache.node_mlp_keys)?;
            #[cfg(all(
                feature = "orb-fixed-shape-dense",
                feature = "intel-mkl",
                target_os = "linux",
                target_arch = "x86_64"
            ))]
            let node_mlp = OrbMlpRmsNormWeights {
                fixed_rows_packed: layer_cache.node_mlp_packed.as_ref(),
                ..node_mlp
            };
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            let node_mlp = OrbMlpRmsNormWeights {
                mlp_f32: if dtype == PredictionComputeDtype::Float32 {
                    layer_cache.node_mlp_f32.as_ref()
                } else {
                    None
                },
                ..node_mlp
            };
            weights.push(OrbGnnLayerWeights {
                cond_node: OrbLinearWeights {
                    weight: self.cached_f64(&layer_cache.cond_node_weight_key)?,
                    bias: self.cached_f64(&layer_cache.cond_node_bias_key)?,
                },
                receive_attn: OrbLinearWeights {
                    weight: self.cached_f64(&layer_cache.receive_attn_weight_key)?,
                    bias: self.cached_f64(&layer_cache.receive_attn_bias_key)?,
                },
                send_attn: OrbLinearWeights {
                    weight: self.cached_f64(&layer_cache.send_attn_weight_key)?,
                    bias: self.cached_f64(&layer_cache.send_attn_bias_key)?,
                },
                attention_pair_weight: layer_cache.attention_pair_weight.clone(),
                attention_pair_bias: layer_cache.attention_pair_bias,
                edge_mlp,
                node_mlp,
            });
        }
        Ok(weights)
    }

    fn cached_f64(&self, key: &str) -> Result<&[f64], OrbV3OmolError> {
        let &index =
            self.tensor_index
                .get(key)
                .ok_or_else(|| OrbV3OmolError::MissingStateTensor {
                    name: key.to_owned(),
                })?;
        Ok(&self.tensor_values_f64[index])
    }

    fn mlp_rms_norm_weights_from_keys(
        &self,
        keys: &OrbMlpRmsNormWeightKeys,
    ) -> Result<OrbMlpRmsNormWeights<'_>, OrbV3OmolError> {
        Ok(OrbMlpRmsNormWeights {
            layer0_weight: self.cached_f64(&keys.layer0_weight)?,
            layer0_bias: self.cached_f64(&keys.layer0_bias)?,
            layer1_weight: self.cached_f64(&keys.layer1_weight)?,
            layer1_bias: self.cached_f64(&keys.layer1_bias)?,
            layer2_weight: self.cached_f64(&keys.layer2_weight)?,
            layer2_bias: self.cached_f64(&keys.layer2_bias)?,
            rms_weight: self.cached_f64(&keys.rms_weight)?,
            #[cfg(all(
                feature = "orb-fixed-shape-dense",
                feature = "intel-mkl",
                target_os = "linux",
                target_arch = "x86_64"
            ))]
            fixed_rows_packed: None,
            #[cfg(all(
                feature = "orb-fixed-shape-dense",
                feature = "intel-mkl",
                target_os = "linux",
                target_arch = "x86_64"
            ))]
            edge_packed: None,
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            mlp_f32: None,
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            edge_f32: None,
        })
    }

    fn gnn_layer_weights(&self, prefix: &str) -> Result<OrbGnnLayerWeights<'_>, OrbV3OmolError> {
        let receive_attn = OrbLinearWeights {
            weight: self
                .state_tensor_f64(&format!("{prefix}._receive_attn.weight"), &[1, LATENT_DIM])?,
            bias: self.state_tensor_f64(&format!("{prefix}._receive_attn.bias"), &[1])?,
        };
        let send_attn = OrbLinearWeights {
            weight: self
                .state_tensor_f64(&format!("{prefix}._send_attn.weight"), &[1, LATENT_DIM])?,
            bias: self.state_tensor_f64(&format!("{prefix}._send_attn.bias"), &[1])?,
        };
        let mut attention_pair_weight = Vec::with_capacity(2 * LATENT_DIM);
        attention_pair_weight.extend_from_slice(receive_attn.weight);
        attention_pair_weight.extend_from_slice(send_attn.weight);
        let attention_pair_bias = [receive_attn.bias[0], send_attn.bias[0]];
        Ok(OrbGnnLayerWeights {
            cond_node: OrbLinearWeights {
                weight: self.state_tensor_f64(
                    &format!("{prefix}._cond_node_proj.weight"),
                    &[LATENT_DIM, LATENT_DIM],
                )?,
                bias: self
                    .state_tensor_f64(&format!("{prefix}._cond_node_proj.bias"), &[LATENT_DIM])?,
            },
            receive_attn,
            send_attn,
            attention_pair_weight,
            attention_pair_bias,
            edge_mlp: self
                .mlp_rms_norm_weights(&format!("{prefix}._edge_mlp"), ORB_GNN_INPUT_DIM)?,
            node_mlp: self
                .mlp_rms_norm_weights(&format!("{prefix}._node_mlp"), ORB_GNN_INPUT_DIM)?,
        })
    }

    pub fn predict(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
    ) -> Result<Prediction, OrbV3OmolError> {
        self.predict_with_compute_dtype(structure, options, PredictionComputeDtype::Float64)
    }

    pub fn predict_with_compute_dtype(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
        dtype: PredictionComputeDtype,
    ) -> Result<Prediction, OrbV3OmolError> {
        validate_orb_compute_dtype(dtype)?;
        if options.compute_stress || options.compute_virial {
            return Err(OrbV3OmolError::RuntimeNotImplemented);
        }
        if !options.compute_energy && !options.compute_forces {
            return Ok(Prediction::default());
        }
        if self.runtime_status() != RUNTIME_STATUS_STATE_EXPORT {
            return Err(OrbV3OmolError::RuntimeNotImplemented);
        }
        let (total_charge, spin_multiplicity) = charge_spin_from_prediction_options(options)?;
        let graph = self.prepare_non_pbc_graph(structure, total_charge, spin_multiplicity)?;
        if options.compute_forces {
            let (energy, forces) =
                self.energy_forces_for_prepared_graph_with_compute_dtype(&graph, dtype)?;
            return Ok(Prediction {
                energy: options.compute_energy.then_some(energy.total_energy),
                forces: Some(forces),
                ..Prediction::default()
            });
        }
        let energy = self.energy_for_prepared_graph_with_compute_dtype(&graph, dtype)?;
        Ok(Prediction {
            energy: Some(energy.total_energy),
            ..Prediction::default()
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OrbV3OmolSpec {
    schema_version: String,
    architecture: String,
    selected_model: String,
    runtime_status: String,
    reference: OrbReferenceSpec,
    checkpoint: OrbCheckpointSpec,
    graph: OrbGraphSpec,
    architecture_parameters: OrbArchitectureParameters,
    #[serde(default)]
    state_inventory: Option<OrbStateInventorySpec>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OrbReferenceSpec {
    repository: String,
    tag: String,
    package_version: String,
    commit: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OrbCheckpointSpec {
    filename: String,
    url: String,
    sha256: String,
    size_bytes: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OrbGraphSpec {
    cutoff_angstrom: f64,
    max_num_neighbors: usize,
    requires_charge: bool,
    requires_spin: bool,
    supports_pbc: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrbArchitectureParameters {
    pub latent_dim: usize,
    pub message_passing_steps: usize,
    pub base_mlp_hidden_dim: usize,
    pub base_mlp_depth: usize,
    pub head_mlp_hidden_dim: usize,
    pub head_mlp_depth: usize,
    pub activation: String,
    pub radial_basis: String,
    pub bessel_num_bases: usize,
    pub sh_lmax: usize,
    pub sh_normalization: String,
    pub sh_normalize: bool,
    pub outer_product_with_cutoff: bool,
    pub mlp_norm: String,
    pub attention_gate: String,
    pub zbl_pair_repulsion: bool,
    pub conservative_forces: bool,
    pub confidence_head: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OrbStateInventorySpec {
    schema_version: String,
    tensor_count: usize,
    total_numel: usize,
    entries: Vec<OrbStateTensorSpec>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OrbStateTensorSpec {
    index: usize,
    name: String,
    dtype: String,
    shape: Vec<usize>,
    numel: usize,
    path: String,
    size_bytes: usize,
    sha256: String,
}

#[derive(Debug, Error)]
pub enum OrbV3OmolError {
    #[error("invalid ORB-v3 Conservative OMol metadata: {0}")]
    InvalidSpec(String),

    #[error("ORB-v3 Conservative OMol runtime is not implemented")]
    RuntimeNotImplemented,

    #[error("ORB-v3 Conservative OMol prediction compute dtype {0:?} is unsupported")]
    UnsupportedComputeDtype(PredictionComputeDtype),

    #[error("unsupported ORB-v3 Conservative OMol element symbol {symbol:?} at atom {atom}")]
    UnsupportedElementSymbol { atom: usize, symbol: String },

    #[error("unsupported ORB-v3 Conservative OMol atomic number {atomic_number} at atom {atom}")]
    UnsupportedAtomicNumber { atom: usize, atomic_number: usize },

    #[error("missing ORB-v3 Conservative OMol state tensor {name:?}")]
    MissingStateTensor { name: String },

    #[error(
        "ORB-v3 Conservative OMol state tensor {name:?} must have dtype {expected}, got {actual}"
    )]
    StateTensorDtypeMismatch {
        name: String,
        expected: &'static str,
        actual: &'static str,
    },

    #[error("ORB-v3 Conservative OMol state tensor {name:?} must have shape {expected:?}, got {actual:?}")]
    StateTensorShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("ORB-v3 Conservative OMol PBC graph construction is not implemented")]
    PbcUnsupported,

    #[error(
        "ORB-v3 Conservative OMol atom {atom} has {count} neighbors, exceeding pinned max {max}"
    )]
    TooManyNeighbors {
        atom: usize,
        count: usize,
        max: usize,
    },

    #[error(transparent)]
    NeighborList(#[from] NeighborListError),

    #[error(transparent)]
    Core(#[from] e3nn_core::CoreError),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

fn build_tensor_index(
    tensors: &BTreeMap<String, OrbV3OmolExternalTensor>,
) -> (HashMap<String, usize>, Vec<Vec<f64>>) {
    let mut index = HashMap::with_capacity(tensors.len());
    let mut values = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        let idx = values.len();
        index.insert(name.clone(), idx);
        match &tensor.values {
            OrbV3OmolExternalTensorValues::F64(v) => values.push(v.clone()),
            OrbV3OmolExternalTensorValues::I64(_) => values.push(Vec::new()),
        }
    }
    (index, values)
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
fn lookup_cached_tensor_f64<'a>(
    tensor_index: &HashMap<String, usize>,
    tensor_values: &'a [Vec<f64>],
    key: &str,
) -> Result<&'a [f64], OrbV3OmolError> {
    let &idx = tensor_index
        .get(key)
        .ok_or_else(|| OrbV3OmolError::MissingStateTensor {
            name: key.to_owned(),
        })?;
    Ok(&tensor_values[idx])
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
fn build_encoder_node_mlp_packed(
    tensor_index: &HashMap<String, usize>,
    tensor_values: &[Vec<f64>],
) -> Result<OrbFixedRowsMlpPackedDense, OrbV3OmolError> {
    let layer0_key = format!("{ORB_ENCODER_NODE_PREFIX}.mlp.NN-0.weight");
    let layer1_key = format!("{ORB_ENCODER_NODE_PREFIX}.mlp.NN-1.weight");
    let layer2_key = format!("{ORB_ENCODER_NODE_PREFIX}.mlp.NN-2.weight");
    build_orb_fixed_rows_mlp_packed(
        128,
        LATENT_DIM,
        ORB_ENCODER_NODE_MLP_PACKED_LABELS,
        lookup_cached_tensor_f64(tensor_index, tensor_values, &layer0_key)?,
        lookup_cached_tensor_f64(tensor_index, tensor_values, &layer1_key)?,
        lookup_cached_tensor_f64(tensor_index, tensor_values, &layer2_key)?,
    )
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
fn build_encoder_edge_mlp_packed(
    tensor_index: &HashMap<String, usize>,
    tensor_values: &[Vec<f64>],
) -> Result<OrbFixedRowsMlpPackedDense, OrbV3OmolError> {
    let layer0_key = format!("{ORB_ENCODER_EDGE_PREFIX}.mlp.NN-0.weight");
    let layer1_key = format!("{ORB_ENCODER_EDGE_PREFIX}.mlp.NN-1.weight");
    let layer2_key = format!("{ORB_ENCODER_EDGE_PREFIX}.mlp.NN-2.weight");
    build_orb_fixed_rows_mlp_packed(
        2304,
        ORB_EDGE_FEATURE_DIM,
        ORB_ENCODER_EDGE_MLP_PACKED_LABELS,
        lookup_cached_tensor_f64(tensor_index, tensor_values, &layer0_key)?,
        lookup_cached_tensor_f64(tensor_index, tensor_values, &layer1_key)?,
        lookup_cached_tensor_f64(tensor_index, tensor_values, &layer2_key)?,
    )
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
fn build_orb_fixed_rows_mlp_packed(
    rows: usize,
    input_dim: usize,
    labels: OrbFixedRowsMlpPackedLabels,
    layer0_weight: &[f64],
    layer1_weight: &[f64],
    layer2_weight: &[f64],
) -> Result<OrbFixedRowsMlpPackedDense, OrbV3OmolError> {
    let layer0_expected = ORB_ENCODER_HIDDEN_DIM * input_dim;
    if layer0_weight.len() != layer0_expected {
        return invalid(format!(
            "ORB fixed-row MLP layer0 packed weight length must be {layer0_expected}, got {}",
            layer0_weight.len()
        ));
    }
    let layer1_expected = ORB_ENCODER_HIDDEN_DIM * ORB_ENCODER_HIDDEN_DIM;
    if layer1_weight.len() != layer1_expected {
        return invalid(format!(
            "ORB fixed-row MLP layer1 packed weight length must be {layer1_expected}, got {}",
            layer1_weight.len()
        ));
    }
    let layer2_expected = LATENT_DIM * ORB_ENCODER_HIDDEN_DIM;
    if layer2_weight.len() != layer2_expected {
        return invalid(format!(
            "ORB fixed-row MLP layer2 packed weight length must be {layer2_expected}, got {}",
            layer2_weight.len()
        ));
    }
    let layer0_forward = crate::blas_backend::MklPackedDgemmBMatrix::pack_output_major_weight(
        layer0_weight,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        input_dim,
        true,
    )
    .map_err(OrbV3OmolError::InvalidSpec)?;
    let layer0_input_vjp = crate::blas_backend::MklPackedDgemmBMatrix::pack_output_major_weight(
        layer0_weight,
        rows,
        input_dim,
        ORB_ENCODER_HIDDEN_DIM,
        false,
    )
    .map_err(OrbV3OmolError::InvalidSpec)?;
    let layer1_forward = crate::blas_backend::MklPackedDgemmBMatrix::pack_output_major_weight(
        layer1_weight,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        true,
    )
    .map_err(OrbV3OmolError::InvalidSpec)?;
    let layer1_input_vjp = crate::blas_backend::MklPackedDgemmBMatrix::pack_output_major_weight(
        layer1_weight,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        false,
    )
    .map_err(OrbV3OmolError::InvalidSpec)?;
    let layer2_forward = crate::blas_backend::MklPackedDgemmBMatrix::pack_output_major_weight(
        layer2_weight,
        rows,
        LATENT_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        true,
    )
    .map_err(OrbV3OmolError::InvalidSpec)?;
    let layer2_input_vjp = crate::blas_backend::MklPackedDgemmBMatrix::pack_output_major_weight(
        layer2_weight,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        LATENT_DIM,
        false,
    )
    .map_err(OrbV3OmolError::InvalidSpec)?;
    Ok(OrbFixedRowsMlpPackedDense {
        rows,
        input_dim,
        labels,
        layer0_forward: Arc::new(layer0_forward),
        layer0_input_vjp: Arc::new(layer0_input_vjp),
        layer1_forward: Arc::new(layer1_forward),
        layer1_input_vjp: Arc::new(layer1_input_vjp),
        layer2_forward: Arc::new(layer2_forward),
        layer2_input_vjp: Arc::new(layer2_input_vjp),
    })
}

fn build_gnn_weight_cache(
    tensor_index: &HashMap<String, usize>,
    tensor_values: &[Vec<f64>],
) -> Result<OrbGnnWeightPlanCache, OrbV3OmolError> {
    let mut layers = Vec::with_capacity(MESSAGE_PASSING_STEPS);
    for layer in 0..MESSAGE_PASSING_STEPS {
        let prefix = format!("model.gnn_stacks.{layer}");
        let recv_w_key = format!("{prefix}._receive_attn.weight");
        let recv_b_key = format!("{prefix}._receive_attn.bias");
        let send_w_key = format!("{prefix}._send_attn.weight");
        let send_b_key = format!("{prefix}._send_attn.bias");

        let lookup = |key: &str| -> Result<&[f64], OrbV3OmolError> {
            let &idx = tensor_index
                .get(key)
                .ok_or_else(|| OrbV3OmolError::MissingStateTensor {
                    name: key.to_owned(),
                })?;
            Ok(&tensor_values[idx])
        };

        let recv_w = lookup(&recv_w_key)?;
        let recv_b = lookup(&recv_b_key)?;
        let send_w = lookup(&send_w_key)?;
        let send_b = lookup(&send_b_key)?;

        let mut attention_pair_weight = Vec::with_capacity(2 * LATENT_DIM);
        attention_pair_weight.extend_from_slice(recv_w);
        attention_pair_weight.extend_from_slice(send_w);
        let attention_pair_bias = [recv_b[0], send_b[0]];

        let mlp_keys = |mlp_prefix: &str| OrbMlpRmsNormWeightKeys {
            layer0_weight: format!("{mlp_prefix}.mlp.NN-0.weight"),
            layer0_bias: format!("{mlp_prefix}.mlp.NN-0.bias"),
            layer1_weight: format!("{mlp_prefix}.mlp.NN-1.weight"),
            layer1_bias: format!("{mlp_prefix}.mlp.NN-1.bias"),
            layer2_weight: format!("{mlp_prefix}.mlp.NN-2.weight"),
            layer2_bias: format!("{mlp_prefix}.mlp.NN-2.bias"),
            rms_weight: format!("{mlp_prefix}.layer_norm.weight"),
        };
        let edge_mlp_keys = mlp_keys(&format!("{prefix}._edge_mlp"));
        let node_mlp_keys = mlp_keys(&format!("{prefix}._node_mlp"));
        #[cfg(all(
            feature = "orb-fixed-shape-dense",
            feature = "intel-mkl",
            target_os = "linux",
            target_arch = "x86_64"
        ))]
        let node_mlp_packed = Some(build_orb_fixed_rows_mlp_packed(
            128,
            ORB_GNN_INPUT_DIM,
            ORB_GNN_NODE_MLP_PACKED_LABELS,
            lookup(&node_mlp_keys.layer0_weight)?,
            lookup(&node_mlp_keys.layer1_weight)?,
            lookup(&node_mlp_keys.layer2_weight)?,
        )?);
        #[cfg(all(
            feature = "orb-fixed-shape-dense",
            feature = "intel-mkl",
            target_os = "linux",
            target_arch = "x86_64"
        ))]
        let edge_mlp_packed = Some(build_orb_edge_mlp_packed(
            lookup(&edge_mlp_keys.layer0_weight)?,
            lookup(&edge_mlp_keys.layer1_weight)?,
            lookup(&edge_mlp_keys.layer2_weight)?,
        )?);
        #[cfg(all(
            feature = "orb-f32-edge-mlp",
            any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    any(feature = "intel-mkl", feature = "openblas"),
                    target_os = "linux",
                    target_arch = "x86_64"
                )
            )
        ))]
        let edge_mlp_f32 = Some(build_orb_edge_mlp_f32(
            lookup(&edge_mlp_keys.layer0_weight)?,
            lookup(&edge_mlp_keys.layer0_bias)?,
            lookup(&edge_mlp_keys.layer1_weight)?,
            lookup(&edge_mlp_keys.layer1_bias)?,
            lookup(&edge_mlp_keys.layer2_weight)?,
            lookup(&edge_mlp_keys.layer2_bias)?,
        )?);
        #[cfg(all(
            feature = "orb-f32-edge-mlp",
            any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    any(feature = "intel-mkl", feature = "openblas"),
                    target_os = "linux",
                    target_arch = "x86_64"
                )
            )
        ))]
        let node_mlp_f32 = Some(build_orb_mlp_f32(
            ORB_GNN_INPUT_DIM,
            lookup(&node_mlp_keys.layer0_weight)?,
            lookup(&node_mlp_keys.layer0_bias)?,
            lookup(&node_mlp_keys.layer1_weight)?,
            lookup(&node_mlp_keys.layer1_bias)?,
            lookup(&node_mlp_keys.layer2_weight)?,
            lookup(&node_mlp_keys.layer2_bias)?,
        )?);

        layers.push(OrbGnnLayerWeightCache {
            cond_node_weight_key: format!("{prefix}._cond_node_proj.weight"),
            cond_node_bias_key: format!("{prefix}._cond_node_proj.bias"),
            receive_attn_weight_key: recv_w_key,
            receive_attn_bias_key: recv_b_key,
            send_attn_weight_key: send_w_key,
            send_attn_bias_key: send_b_key,
            edge_mlp_keys,
            node_mlp_keys,
            attention_pair_weight,
            attention_pair_bias,
            #[cfg(all(
                feature = "orb-fixed-shape-dense",
                feature = "intel-mkl",
                target_os = "linux",
                target_arch = "x86_64"
            ))]
            node_mlp_packed,
            #[cfg(all(
                feature = "orb-fixed-shape-dense",
                feature = "intel-mkl",
                target_os = "linux",
                target_arch = "x86_64"
            ))]
            edge_mlp_packed,
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            node_mlp_f32,
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            edge_mlp_f32,
        });
    }
    Ok(OrbGnnWeightPlanCache { layers })
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn build_orb_mlp_f32(
    input_dim: usize,
    layer0_weight: &[f64],
    layer0_bias: &[f64],
    layer1_weight: &[f64],
    layer1_bias: &[f64],
    layer2_weight: &[f64],
    layer2_bias: &[f64],
) -> Result<OrbMlpF32Dense, OrbV3OmolError> {
    let layer0_expected = ORB_ENCODER_HIDDEN_DIM * input_dim;
    if layer0_weight.len() != layer0_expected {
        return invalid(format!(
            "ORB f32 MLP layer0 weight length must be {layer0_expected}, got {}",
            layer0_weight.len()
        ));
    }
    if layer0_bias.len() != ORB_ENCODER_HIDDEN_DIM {
        return invalid(format!(
            "ORB f32 MLP layer0 bias length must be {}, got {}",
            ORB_ENCODER_HIDDEN_DIM,
            layer0_bias.len()
        ));
    }
    let layer1_expected = ORB_ENCODER_HIDDEN_DIM * ORB_ENCODER_HIDDEN_DIM;
    if layer1_weight.len() != layer1_expected {
        return invalid(format!(
            "ORB f32 MLP layer1 weight length must be {layer1_expected}, got {}",
            layer1_weight.len()
        ));
    }
    if layer1_bias.len() != ORB_ENCODER_HIDDEN_DIM {
        return invalid(format!(
            "ORB f32 MLP layer1 bias length must be {}, got {}",
            ORB_ENCODER_HIDDEN_DIM,
            layer1_bias.len()
        ));
    }
    let layer2_expected = LATENT_DIM * ORB_ENCODER_HIDDEN_DIM;
    if layer2_weight.len() != layer2_expected {
        return invalid(format!(
            "ORB f32 MLP layer2 weight length must be {layer2_expected}, got {}",
            layer2_weight.len()
        ));
    }
    if layer2_bias.len() != LATENT_DIM {
        return invalid(format!(
            "ORB f32 MLP layer2 bias length must be {LATENT_DIM}, got {}",
            layer2_bias.len()
        ));
    }
    Ok(OrbMlpF32Dense {
        input_dim,
        layer0_weight: layer0_weight.iter().map(|&value| value as f32).collect(),
        layer0_bias: layer0_bias.iter().map(|&value| value as f32).collect(),
        layer1_weight: layer1_weight.iter().map(|&value| value as f32).collect(),
        layer1_bias: layer1_bias.iter().map(|&value| value as f32).collect(),
        layer2_weight: layer2_weight.iter().map(|&value| value as f32).collect(),
        layer2_bias: layer2_bias.iter().map(|&value| value as f32).collect(),
    })
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn build_encoder_mlp_f32(
    prefix: &str,
    input_dim: usize,
    tensor_index: &HashMap<String, usize>,
    tensor_values: &[Vec<f64>],
) -> Result<OrbMlpF32Dense, OrbV3OmolError> {
    let lookup = |key: &str| -> Result<&[f64], OrbV3OmolError> {
        let &idx = tensor_index
            .get(key)
            .ok_or_else(|| OrbV3OmolError::MissingStateTensor {
                name: key.to_owned(),
            })?;
        Ok(&tensor_values[idx])
    };
    let layer0_weight_key = format!("{prefix}.mlp.NN-0.weight");
    let layer0_bias_key = format!("{prefix}.mlp.NN-0.bias");
    let layer1_weight_key = format!("{prefix}.mlp.NN-1.weight");
    let layer1_bias_key = format!("{prefix}.mlp.NN-1.bias");
    let layer2_weight_key = format!("{prefix}.mlp.NN-2.weight");
    let layer2_bias_key = format!("{prefix}.mlp.NN-2.bias");
    build_orb_mlp_f32(
        input_dim,
        lookup(&layer0_weight_key)?,
        lookup(&layer0_bias_key)?,
        lookup(&layer1_weight_key)?,
        lookup(&layer1_bias_key)?,
        lookup(&layer2_weight_key)?,
        lookup(&layer2_bias_key)?,
    )
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn build_orb_edge_mlp_f32(
    layer0_weight: &[f64],
    layer0_bias: &[f64],
    layer1_weight: &[f64],
    layer1_bias: &[f64],
    layer2_weight: &[f64],
    layer2_bias: &[f64],
) -> Result<OrbEdgeMlpF32Dense, OrbV3OmolError> {
    let layer0_expected = ORB_ENCODER_HIDDEN_DIM * ORB_GNN_INPUT_DIM;
    if layer0_weight.len() != layer0_expected {
        return invalid(format!(
            "ORB edge MLP f32 layer0 weight length must be {layer0_expected}, got {}",
            layer0_weight.len()
        ));
    }
    if layer0_bias.len() != ORB_ENCODER_HIDDEN_DIM {
        return invalid(format!(
            "ORB edge MLP f32 layer0 bias length must be {}, got {}",
            ORB_ENCODER_HIDDEN_DIM,
            layer0_bias.len()
        ));
    }
    let layer1_expected = ORB_ENCODER_HIDDEN_DIM * ORB_ENCODER_HIDDEN_DIM;
    if layer1_weight.len() != layer1_expected {
        return invalid(format!(
            "ORB edge MLP f32 layer1 weight length must be {layer1_expected}, got {}",
            layer1_weight.len()
        ));
    }
    if layer1_bias.len() != ORB_ENCODER_HIDDEN_DIM {
        return invalid(format!(
            "ORB edge MLP f32 layer1 bias length must be {}, got {}",
            ORB_ENCODER_HIDDEN_DIM,
            layer1_bias.len()
        ));
    }
    let layer2_expected = LATENT_DIM * ORB_ENCODER_HIDDEN_DIM;
    if layer2_weight.len() != layer2_expected {
        return invalid(format!(
            "ORB edge MLP f32 layer2 weight length must be {layer2_expected}, got {}",
            layer2_weight.len()
        ));
    }
    if layer2_bias.len() != LATENT_DIM {
        return invalid(format!(
            "ORB edge MLP f32 layer2 bias length must be {}, got {}",
            LATENT_DIM,
            layer2_bias.len()
        ));
    }
    let layer1_weight = layer1_weight
        .iter()
        .map(|&value| value as f32)
        .collect::<Vec<_>>();
    let mut layer1_weight_input_major = vec![0.0_f32; layer1_weight.len()];
    for out_channel in 0..ORB_ENCODER_HIDDEN_DIM {
        for in_channel in 0..ORB_ENCODER_HIDDEN_DIM {
            layer1_weight_input_major[in_channel * ORB_ENCODER_HIDDEN_DIM + out_channel] =
                layer1_weight[out_channel * ORB_ENCODER_HIDDEN_DIM + in_channel];
        }
    }
    Ok(OrbEdgeMlpF32Dense {
        layer0_weight: layer0_weight.iter().map(|&value| value as f32).collect(),
        layer0_bias: layer0_bias.iter().map(|&value| value as f32).collect(),
        layer1_weight,
        layer1_weight_input_major,
        layer1_bias: layer1_bias.iter().map(|&value| value as f32).collect(),
        layer2_weight: layer2_weight.iter().map(|&value| value as f32).collect(),
        layer2_bias: layer2_bias.iter().map(|&value| value as f32).collect(),
    })
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
fn build_orb_edge_mlp_packed(
    layer0_weight: &[f64],
    layer1_weight: &[f64],
    layer2_weight: &[f64],
) -> Result<OrbEdgeMlpPackedDense, OrbV3OmolError> {
    let layer0_expected = ORB_ENCODER_HIDDEN_DIM * ORB_GNN_INPUT_DIM;
    if layer0_weight.len() != layer0_expected {
        return invalid(format!(
            "ORB edge MLP layer0 packed weight length must be {layer0_expected}, got {}",
            layer0_weight.len()
        ));
    }
    let layer1_expected = ORB_ENCODER_HIDDEN_DIM * ORB_ENCODER_HIDDEN_DIM;
    if layer1_weight.len() != layer1_expected {
        return invalid(format!(
            "ORB edge MLP layer1 packed weight length must be {layer1_expected}, got {}",
            layer1_weight.len()
        ));
    }
    let layer2_expected = LATENT_DIM * ORB_ENCODER_HIDDEN_DIM;
    if layer2_weight.len() != layer2_expected {
        return invalid(format!(
            "ORB edge MLP layer2 packed weight length must be {layer2_expected}, got {}",
            layer2_weight.len()
        ));
    }
    let layer0_edge_forward = crate::blas_backend::MklPackedDgemmBMatrix::pack_strided_b_matrix(
        layer0_weight,
        0,
        ORB_GNN_INPUT_DIM,
        2304,
        ORB_ENCODER_HIDDEN_DIM,
        LATENT_DIM,
        true,
    )
    .map_err(OrbV3OmolError::InvalidSpec)?;
    let layer0_edge_vjp = crate::blas_backend::MklPackedDgemmBMatrix::pack_strided_b_matrix(
        layer0_weight,
        0,
        ORB_GNN_INPUT_DIM,
        2304,
        LATENT_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        false,
    )
    .map_err(OrbV3OmolError::InvalidSpec)?;
    let layer0_sender_forward = crate::blas_backend::MklPackedDgemmBMatrix::pack_strided_b_matrix(
        layer0_weight,
        LATENT_DIM,
        ORB_GNN_INPUT_DIM,
        128,
        ORB_ENCODER_HIDDEN_DIM,
        LATENT_DIM,
        true,
    )
    .map_err(OrbV3OmolError::InvalidSpec)?;
    let layer0_receiver_forward =
        crate::blas_backend::MklPackedDgemmBMatrix::pack_strided_b_matrix(
            layer0_weight,
            2 * LATENT_DIM,
            ORB_GNN_INPUT_DIM,
            128,
            ORB_ENCODER_HIDDEN_DIM,
            LATENT_DIM,
            true,
        )
        .map_err(OrbV3OmolError::InvalidSpec)?;
    let layer0_sender_vjp = crate::blas_backend::MklPackedDgemmBMatrix::pack_strided_b_matrix(
        layer0_weight,
        LATENT_DIM,
        ORB_GNN_INPUT_DIM,
        128,
        LATENT_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        false,
    )
    .map_err(OrbV3OmolError::InvalidSpec)?;
    let layer0_receiver_vjp = crate::blas_backend::MklPackedDgemmBMatrix::pack_strided_b_matrix(
        layer0_weight,
        2 * LATENT_DIM,
        ORB_GNN_INPUT_DIM,
        128,
        LATENT_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        false,
    )
    .map_err(OrbV3OmolError::InvalidSpec)?;
    let layer1_forward = crate::blas_backend::MklPackedDgemmBMatrix::pack_output_major_weight(
        layer1_weight,
        2304,
        ORB_ENCODER_HIDDEN_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        true,
    )
    .map_err(OrbV3OmolError::InvalidSpec)?;
    let layer1_input_vjp = crate::blas_backend::MklPackedDgemmBMatrix::pack_output_major_weight(
        layer1_weight,
        2304,
        ORB_ENCODER_HIDDEN_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        false,
    )
    .map_err(OrbV3OmolError::InvalidSpec)?;
    let layer2_forward = crate::blas_backend::MklPackedDgemmBMatrix::pack_output_major_weight(
        layer2_weight,
        2304,
        LATENT_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        true,
    )
    .map_err(OrbV3OmolError::InvalidSpec)?;
    let layer2_input_vjp = crate::blas_backend::MklPackedDgemmBMatrix::pack_output_major_weight(
        layer2_weight,
        2304,
        ORB_ENCODER_HIDDEN_DIM,
        LATENT_DIM,
        false,
    )
    .map_err(OrbV3OmolError::InvalidSpec)?;
    Ok(OrbEdgeMlpPackedDense {
        layer0_edge_forward: Arc::new(layer0_edge_forward),
        layer0_edge_vjp: Arc::new(layer0_edge_vjp),
        layer0_sender_forward: Arc::new(layer0_sender_forward),
        layer0_receiver_forward: Arc::new(layer0_receiver_forward),
        layer0_sender_vjp: Arc::new(layer0_sender_vjp),
        layer0_receiver_vjp: Arc::new(layer0_receiver_vjp),
        layer1_forward: Arc::new(layer1_forward),
        layer1_input_vjp: Arc::new(layer1_input_vjp),
        layer2_forward: Arc::new(layer2_forward),
        layer2_input_vjp: Arc::new(layer2_input_vjp),
    })
}

fn validate_graph_lengths(graph: &OrbV3OmolPreparedGraph) -> Result<(), OrbV3OmolError> {
    if graph.atomic_numbers.len() != graph.n_atoms {
        return invalid(format!(
            "ORB prepared graph atomic_numbers length must be {}, got {}",
            graph.n_atoms,
            graph.atomic_numbers.len()
        ));
    }
    let edge_count = graph.senders.len();
    if graph.receivers.len() != edge_count {
        return invalid(format!(
            "ORB prepared graph receivers length must be {edge_count}, got {}",
            graph.receivers.len()
        ));
    }
    if graph.edge_vectors.len() != edge_count {
        return invalid(format!(
            "ORB prepared graph edge_vectors length must be {edge_count}, got {}",
            graph.edge_vectors.len()
        ));
    }
    if graph.unit_shifts.len() != edge_count {
        return invalid(format!(
            "ORB prepared graph unit_shifts length must be {edge_count}, got {}",
            graph.unit_shifts.len()
        ));
    }
    Ok(())
}

fn duplicate_component_plan(
    graph: &OrbV3OmolPreparedGraph,
) -> Result<Option<OrbDuplicateComponentPlan>, OrbV3OmolError> {
    validate_graph_lengths(graph)?;
    let components = connected_components(graph)?;
    if components.len() < 2 {
        return Ok(None);
    }
    let representative = subgraph_for_component(graph, &components[0])?;
    for component in &components[1..] {
        let candidate = subgraph_for_component(graph, component)?;
        if !component_graphs_equivalent(&representative, &candidate) {
            return Ok(None);
        }
    }
    Ok(Some(OrbDuplicateComponentPlan {
        components,
        representative,
    }))
}

fn connected_components(graph: &OrbV3OmolPreparedGraph) -> Result<Vec<Vec<usize>>, OrbV3OmolError> {
    let mut adjacency = vec![Vec::new(); graph.n_atoms];
    for edge in 0..graph.edge_count() {
        let sender = graph.senders[edge];
        let receiver = graph.receivers[edge];
        if sender >= graph.n_atoms || receiver >= graph.n_atoms {
            return invalid(format!(
                "ORB prepared graph edge {edge} sender/receiver out of bounds: {sender}->{receiver}, nodes {}",
                graph.n_atoms
            ));
        }
        adjacency[sender].push(receiver);
        adjacency[receiver].push(sender);
    }

    let mut seen = vec![false; graph.n_atoms];
    let mut components = Vec::new();
    for atom in 0..graph.n_atoms {
        if seen[atom] {
            continue;
        }
        let mut component = Vec::new();
        let mut stack = vec![atom];
        seen[atom] = true;
        while let Some(current) = stack.pop() {
            component.push(current);
            for &neighbor in &adjacency[current] {
                if !seen[neighbor] {
                    seen[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    Ok(components)
}

fn subgraph_for_component(
    graph: &OrbV3OmolPreparedGraph,
    component: &[usize],
) -> Result<OrbV3OmolPreparedGraph, OrbV3OmolError> {
    let mut local_index = vec![usize::MAX; graph.n_atoms];
    let mut atomic_numbers = Vec::with_capacity(component.len());
    for (local, &global) in component.iter().enumerate() {
        if global >= graph.n_atoms {
            return invalid(format!(
                "ORB component atom index {global} out of bounds for {} atoms",
                graph.n_atoms
            ));
        }
        local_index[global] = local;
        atomic_numbers.push(graph.atomic_numbers[global]);
    }

    let mut senders = Vec::new();
    let mut receivers = Vec::new();
    let mut edge_vectors = Vec::new();
    let mut unit_shifts = Vec::new();
    for edge in 0..graph.edge_count() {
        let sender = graph.senders[edge];
        let receiver = graph.receivers[edge];
        let local_sender = local_index[sender];
        if local_sender == usize::MAX {
            continue;
        }
        let local_receiver = local_index[receiver];
        if local_receiver == usize::MAX {
            return invalid(format!(
                "ORB component edge {edge} crosses disconnected components: {sender}->{receiver}"
            ));
        }
        senders.push(local_sender);
        receivers.push(local_receiver);
        edge_vectors.push(graph.edge_vectors[edge]);
        unit_shifts.push(graph.unit_shifts[edge]);
    }

    Ok(OrbV3OmolPreparedGraph {
        n_atoms: component.len(),
        atomic_numbers,
        senders,
        receivers,
        edge_vectors,
        unit_shifts,
        total_charge: graph.total_charge,
        spin_multiplicity: graph.spin_multiplicity,
    })
}

fn component_graphs_equivalent(
    reference: &OrbV3OmolPreparedGraph,
    candidate: &OrbV3OmolPreparedGraph,
) -> bool {
    if reference.n_atoms != candidate.n_atoms
        || reference.atomic_numbers != candidate.atomic_numbers
        || reference.senders != candidate.senders
        || reference.receivers != candidate.receivers
        || reference.edge_vectors.len() != candidate.edge_vectors.len()
        || reference.unit_shifts.len() != candidate.unit_shifts.len()
    {
        return false;
    }
    reference
        .edge_vectors
        .iter()
        .zip(&candidate.edge_vectors)
        .all(|(&lhs, &rhs)| vec3_close(lhs, rhs, ORB_DUPLICATE_COMPONENT_TOLERANCE))
        && reference
            .unit_shifts
            .iter()
            .zip(&candidate.unit_shifts)
            .all(|(&lhs, &rhs)| vec3_close(lhs, rhs, ORB_DUPLICATE_COMPONENT_TOLERANCE))
}

fn vec3_close(lhs: Vec3, rhs: Vec3, tolerance: f64) -> bool {
    (0..3).all(|axis| (lhs[axis] - rhs[axis]).abs() <= tolerance)
}

fn scale_duplicate_component_energy(
    representative: OrbV3OmolEnergyBreakdown,
    count: usize,
) -> OrbV3OmolEnergyBreakdown {
    let component_count = count as f64;
    let denormalized_without_zbl = representative.denormalized_without_zbl * component_count;
    OrbV3OmolEnergyBreakdown {
        normalized_energy_head: representative.normalized_energy_head,
        reference_energy: representative.reference_energy * component_count,
        denormalized_without_zbl,
        zbl_energy: representative.zbl_energy,
        total_energy: denormalized_without_zbl + representative.zbl_energy,
    }
}

fn reconstruct_duplicate_component_forces(
    plan: &OrbDuplicateComponentPlan,
    total_atoms: usize,
    representative_forces: &[Vec3],
    representative_zbl_forces: &[Vec3],
) -> Result<Vec<Vec3>, OrbV3OmolError> {
    if plan.components.is_empty() {
        return invalid("ORB duplicate component force expansion requires at least one component");
    }
    if representative_zbl_forces.len() != representative_forces.len() {
        return invalid(format!(
            "ORB duplicate representative ZBL force count mismatch: expected {}, got {}",
            representative_forces.len(),
            representative_zbl_forces.len()
        ));
    }
    let mut forces = vec![[0.0; 3]; total_atoms];
    let zbl_scale = 1.0 / plan.components.len() as f64;
    for component in &plan.components {
        if component.len() != representative_forces.len() {
            return invalid(format!(
                "ORB duplicate component atom count mismatch: expected {}, got {}",
                representative_forces.len(),
                component.len()
            ));
        }
        for (local_atom, &global_atom) in component.iter().enumerate() {
            if global_atom >= forces.len() {
                return invalid(format!(
                    "ORB duplicate component atom index {global_atom} out of bounds for {} atoms",
                    forces.len()
                ));
            }
            for axis in 0..3 {
                let gnn_force = representative_forces[local_atom][axis]
                    - representative_zbl_forces[local_atom][axis];
                forces[global_atom][axis] =
                    gnn_force + representative_zbl_forces[local_atom][axis] * zbl_scale;
            }
        }
    }
    Ok(forces)
}

fn vec3_norm(vector: Vec3) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn orb_cutoff(distance: f64, r_max: f64) -> f64 {
    if distance >= r_max {
        return 0.0;
    }
    let x = distance / r_max;
    1.0 - 15.0 * x.powi(4) + 24.0 * x.powi(5) - 10.0 * x.powi(6)
}

fn orb_cutoff_derivative(distance: f64, r_max: f64) -> f64 {
    if distance >= r_max {
        return 0.0;
    }
    let x = distance / r_max;
    (-60.0 * x.powi(3) + 120.0 * x.powi(4) - 60.0 * x.powi(5)) / r_max
}

fn fill_charge_spin_embedding(
    value: f64,
    weights: &[f64],
    out: &mut [f64],
    zero_when_value_is_zero: bool,
) -> Result<(), OrbV3OmolError> {
    if weights.len() != ORB_CONDITIONING_FREQUENCIES || out.len() != ORB_CONDITIONING_HALF_DIM {
        return invalid(format!(
            "ORB charge/spin embedding expects {} weights and {} outputs, got {}/{}",
            ORB_CONDITIONING_FREQUENCIES,
            ORB_CONDITIONING_HALF_DIM,
            weights.len(),
            out.len()
        ));
    }
    if !value.is_finite() {
        return invalid(format!("ORB charge/spin value must be finite, got {value}"));
    }
    if zero_when_value_is_zero && value == 0.0 {
        out.fill(0.0);
        return Ok(());
    }
    // Upstream ORB builds these Fourier features through float32 tensors even
    // in an otherwise float64 reference run. Preserve that f32-origin rounding
    // before applying the f64 trigonometric operations.
    let value = f64::from(value as f32);
    for frequency in 0..ORB_CONDITIONING_FREQUENCIES {
        let weight = weights[frequency];
        if !weight.is_finite() {
            return invalid(format!(
                "ORB charge/spin embedding weight {frequency} must be finite, got {weight}"
            ));
        }
        let projected = value * weight * 2.0 * std::f64::consts::PI;
        out[frequency] = projected.sin();
        out[ORB_CONDITIONING_FREQUENCIES + frequency] = projected.cos();
    }
    Ok(())
}

fn checked_row_count(label: &str, values: &[f64], width: usize) -> Result<usize, OrbV3OmolError> {
    if width == 0 {
        return invalid(format!("{label} width must be nonzero"));
    }
    if !values.len().is_multiple_of(width) {
        return invalid(format!(
            "{label} length {} must be divisible by width {width}",
            values.len()
        ));
    }
    Ok(values.len() / width)
}

pub(crate) fn charge_spin_from_prediction_options(
    options: &PredictionOptions,
) -> Result<(f64, f64), OrbV3OmolError> {
    let total_charge = options.total_charge.unwrap_or(DEFAULT_TOTAL_CHARGE);
    let spin_multiplicity = options
        .spin_multiplicity
        .unwrap_or(DEFAULT_SPIN_MULTIPLICITY);
    validate_charge_spin_values(total_charge, spin_multiplicity)?;
    Ok((total_charge, spin_multiplicity))
}

#[must_use]
pub(crate) fn supported_element_symbols() -> Vec<String> {
    ORB_ELEMENT_SYMBOLS_BY_Z[1..=ORB_MAX_SUPPORTED_ATOMIC_NUMBER]
        .iter()
        .map(|symbol| (*symbol).to_owned())
        .collect()
}

fn validate_charge_spin_values(
    total_charge: f64,
    spin_multiplicity: f64,
) -> Result<(), OrbV3OmolError> {
    if !total_charge.is_finite() {
        return invalid(format!("total_charge must be finite, got {total_charge}"));
    }
    if !spin_multiplicity.is_finite() {
        return invalid(format!(
            "spin_multiplicity must be finite, got {spin_multiplicity}"
        ));
    }
    Ok(())
}

fn charge_spin_metadata_matches(
    prepared_total_charge: f64,
    prepared_spin_multiplicity: f64,
    total_charge: f64,
    spin_multiplicity: f64,
) -> bool {
    (prepared_total_charge - total_charge).abs() <= CHARGE_SPIN_MATCH_TOLERANCE
        && (prepared_spin_multiplicity - spin_multiplicity).abs() <= CHARGE_SPIN_MATCH_TOLERANCE
}

fn expect_len<T>(label: &str, values: &[T], expected: usize) -> Result<(), OrbV3OmolError> {
    if values.len() != expected {
        return invalid(format!(
            "{label} length must be {expected}, got {}",
            values.len()
        ));
    }
    Ok(())
}

fn mean_node_features(nodes: &[f64], rows: usize) -> Result<Vec<f64>, OrbV3OmolError> {
    if rows == 0 {
        return invalid("ORB node feature mean requires at least one row");
    }
    expect_len("ORB node feature mean input", nodes, rows * LATENT_DIM)?;
    let mut mean = vec![0.0; LATENT_DIM];
    for row in 0..rows {
        let node = &nodes[row * LATENT_DIM..(row + 1) * LATENT_DIM];
        for channel in 0..LATENT_DIM {
            mean[channel] += node[channel];
        }
    }
    let scale = 1.0 / rows as f64;
    for value in &mut mean {
        *value *= scale;
    }
    Ok(mean)
}

fn orb_linear_silu_forward(
    input: &[f64],
    rows: usize,
    input_dim: usize,
    weight: &[f64],
    bias: &[f64],
    output_dim: usize,
) -> Result<Vec<f64>, OrbV3OmolError> {
    let mut output = orb_linear_forward(input, rows, input_dim, weight, bias, output_dim)?;
    orb_silu_in_place(&mut output);
    Ok(output)
}

fn orb_edge_attention_pair_forward(
    edges: &[f64],
    features: &OrbV3OmolInputFeatures,
    pair_weight: &[f64],
    pair_bias: &[f64; 2],
) -> Result<(OrbEdgeAttentionForward, OrbEdgeAttentionForward), OrbV3OmolError> {
    let edge_rows = features.attention_cutoffs.len();
    if pair_weight.len() != 2 * LATENT_DIM {
        return invalid(format!(
            "ORB paired attention weight length must be {}, got {}",
            2 * LATENT_DIM,
            pair_weight.len()
        ));
    }
    let logits = orb_linear_forward(edges, edge_rows, LATENT_DIM, pair_weight, pair_bias, 2)?;
    let sigmoid = orb_sigmoid_values(&logits);
    let mut receive_values = vec![0.0; edge_rows];
    let mut receive_sigmoid = vec![0.0; edge_rows];
    let mut send_values = vec![0.0; edge_rows];
    let mut send_sigmoid = vec![0.0; edge_rows];
    for edge in 0..edge_rows {
        let cutoff = features.attention_cutoffs[edge];
        let receive = sigmoid[2 * edge];
        let send = sigmoid[2 * edge + 1];
        receive_sigmoid[edge] = receive;
        receive_values[edge] = receive * cutoff;
        send_sigmoid[edge] = send;
        send_values[edge] = send * cutoff;
    }
    Ok((
        OrbEdgeAttentionForward {
            values: receive_values,
            sigmoid: receive_sigmoid,
        },
        OrbEdgeAttentionForward {
            values: send_values,
            sigmoid: send_sigmoid,
        },
    ))
}

fn orb_edge_attention_pair_input_vjp(
    attention_cutoffs: &[f64],
    receive_weights: OrbLinearWeights<'_>,
    send_weights: OrbLinearWeights<'_>,
    buffers: OrbEdgeAttentionPairVjpBuffers<'_>,
) -> Result<(), OrbV3OmolError> {
    let edge_rows = attention_cutoffs.len();
    expect_len(
        "ORB receive attention sigmoid cache",
        buffers.receive_sigmoid,
        edge_rows,
    )?;
    expect_len(
        "ORB send attention sigmoid cache",
        buffers.send_sigmoid,
        edge_rows,
    )?;
    expect_len(
        "ORB receive attention VJP cotangents",
        buffers.grad_receive_attention,
        edge_rows,
    )?;
    expect_len(
        "ORB send attention VJP cotangents",
        buffers.grad_send_attention,
        edge_rows,
    )?;
    expect_len(
        "ORB attention VJP edge cotangents",
        buffers.grad_edges,
        edge_rows * LATENT_DIM,
    )?;
    expect_len(
        "ORB attention cutoff cotangents",
        buffers.grad_attention_cutoffs,
        edge_rows,
    )?;
    if receive_weights.weight.len() != LATENT_DIM || receive_weights.bias.len() != 1 {
        return invalid(format!(
            "ORB receive attention weights must have weight len {LATENT_DIM} and bias len 1, got {} and {}",
            receive_weights.weight.len(),
            receive_weights.bias.len()
        ));
    }
    if send_weights.weight.len() != LATENT_DIM || send_weights.bias.len() != 1 {
        return invalid(format!(
            "ORB send attention weights must have weight len {LATENT_DIM} and bias len 1, got {} and {}",
            send_weights.weight.len(),
            send_weights.bias.len()
        ));
    }
    for (edge, &attention_cutoff) in attention_cutoffs.iter().enumerate() {
        let receive_sigmoid = buffers.receive_sigmoid[edge];
        let send_sigmoid = buffers.send_sigmoid[edge];
        buffers.grad_attention_cutoffs[edge] += buffers.grad_receive_attention[edge]
            * receive_sigmoid
            + buffers.grad_send_attention[edge] * send_sigmoid;
        let receive_grad_logit = buffers.grad_receive_attention[edge]
            * attention_cutoff
            * receive_sigmoid
            * (1.0 - receive_sigmoid);
        let send_grad_logit = buffers.grad_send_attention[edge]
            * attention_cutoff
            * send_sigmoid
            * (1.0 - send_sigmoid);
        let grad_edge_row = &mut buffers.grad_edges[edge * LATENT_DIM..(edge + 1) * LATENT_DIM];
        for (channel, grad_edge) in grad_edge_row.iter_mut().enumerate() {
            *grad_edge += receive_grad_logit * receive_weights.weight[channel]
                + send_grad_logit * send_weights.weight[channel];
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(dead_code)]
fn orb_mlp_rms_norm_input_vjp(
    input: &[f64],
    grad_output: &[f64],
    rows: usize,
    input_dim: usize,
    weights: OrbMlpRmsNormWeights<'_>,
) -> Result<Vec<f64>, OrbV3OmolError> {
    let (_, cache) = orb_mlp_rms_norm_forward_with_cache(input, rows, input_dim, weights)?;
    orb_mlp_rms_norm_input_vjp_cached(&cache, grad_output, rows, input_dim, weights)
}

fn orb_mlp_rms_norm_forward_with_cache(
    input: &[f64],
    rows: usize,
    input_dim: usize,
    weights: OrbMlpRmsNormWeights<'_>,
) -> Result<(Vec<f64>, OrbMlpRmsNormForwardCache), OrbV3OmolError> {
    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    if let Some(f32_dense) = weights.mlp_f32 {
        if let Some(result) =
            orb_mlp_rms_norm_forward_f32(input, rows, input_dim, weights, f32_dense)?
        {
            return Ok(result);
        }
    }

    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    let mut layer0 = orb_mlp_layer0_forward(input, rows, input_dim, weights)?;
    #[cfg(not(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    )))]
    let mut layer0 = orb_linear_forward(
        input,
        rows,
        input_dim,
        weights.layer0_weight,
        weights.layer0_bias,
        ORB_ENCODER_HIDDEN_DIM,
    )?;
    let layer0_silu_derivative = orb_silu_in_place_with_derivative(&mut layer0);
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    let mut layer1 = orb_mlp_layer1_forward(&layer0, rows, weights)?;
    #[cfg(not(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    )))]
    let mut layer1 = orb_linear_forward(
        &layer0,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        weights.layer1_weight,
        weights.layer1_bias,
        ORB_ENCODER_HIDDEN_DIM,
    )?;
    let layer1_silu_derivative = orb_silu_in_place_with_derivative(&mut layer1);
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    let layer2 = orb_mlp_layer2_forward(&layer1, rows, weights)?;
    #[cfg(not(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    )))]
    let layer2 = orb_linear_forward(
        &layer1,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        weights.layer2_weight,
        weights.layer2_bias,
        LATENT_DIM,
    )?;
    let (output, layer2_inv_rms) =
        orb_rms_norm_forward_with_inv_rms(&layer2, rows, LATENT_DIM, weights.rms_weight)?;
    Ok((
        output,
        OrbMlpRmsNormForwardCache {
            layer0_silu_derivative,
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            layer0_silu_derivative_f32: None,
            layer1_silu_derivative,
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            layer1_silu_derivative_f32: None,
            layer2,
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            layer2_f32: None,
            layer2_inv_rms,
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            layer2_inv_rms_f32: None,
        },
    ))
}

fn orb_mlp_rms_norm_input_vjp_cached(
    cache: &OrbMlpRmsNormForwardCache,
    grad_output: &[f64],
    rows: usize,
    input_dim: usize,
    weights: OrbMlpRmsNormWeights<'_>,
) -> Result<Vec<f64>, OrbV3OmolError> {
    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    if let Some(f32_dense) = weights.mlp_f32 {
        match (
            cache.layer2_f32.as_deref(),
            cache.layer2_inv_rms_f32.as_deref(),
        ) {
            (Some(layer2_f32), Some(layer2_inv_rms_f32)) => {
                let grad_layer2_f32 = orb_rms_norm_input_vjp_with_inv_rms_f32_labeled(
                    "mlp_rms_norm_input_vjp_f32",
                    layer2_f32,
                    layer2_inv_rms_f32,
                    grad_output,
                    rows,
                    LATENT_DIM,
                    weights.rms_weight,
                )?;
                if let Some(grad_input) = orb_mlp_rms_norm_input_vjp_f32(
                    &grad_layer2_f32,
                    cache,
                    rows,
                    input_dim,
                    f32_dense,
                )? {
                    return Ok(grad_input);
                }
                return invalid("ORB f32 MLP RMSNorm VJP cache could not dispatch");
            }
            (Some(_), None) | (None, Some(_)) => {
                return invalid("ORB f32 MLP RMSNorm VJP cache is incomplete");
            }
            (None, None) => {}
        }
    }

    expect_len(
        "ORB cached MLP RMSNorm layer2",
        &cache.layer2,
        rows * LATENT_DIM,
    )?;
    expect_len(
        "ORB cached MLP RMSNorm layer2 inv_rms",
        &cache.layer2_inv_rms,
        rows,
    )?;
    let grad_layer2 = orb_rms_norm_input_vjp_with_inv_rms(
        &cache.layer2,
        &cache.layer2_inv_rms,
        grad_output,
        rows,
        LATENT_DIM,
        weights.rms_weight,
    )?;
    expect_len(
        "ORB cached MLP RMSNorm layer0 SiLU derivative",
        &cache.layer0_silu_derivative,
        rows * ORB_ENCODER_HIDDEN_DIM,
    )?;
    expect_len(
        "ORB cached MLP RMSNorm layer1 SiLU derivative",
        &cache.layer1_silu_derivative,
        rows * ORB_ENCODER_HIDDEN_DIM,
    )?;
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    let mut grad_pre1 = orb_mlp_layer2_input_vjp(&grad_layer2, rows, weights)?;
    #[cfg(not(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    )))]
    let mut grad_pre1 = orb_linear_input_vjp(
        &grad_layer2,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        weights.layer2_weight,
        LATENT_DIM,
    )?;
    orb_mul_in_place(&mut grad_pre1, &cache.layer1_silu_derivative)?;
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    let mut grad_pre0 = orb_mlp_layer1_input_vjp(&grad_pre1, rows, weights)?;
    #[cfg(not(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    )))]
    let mut grad_pre0 = orb_linear_input_vjp(
        &grad_pre1,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        weights.layer1_weight,
        ORB_ENCODER_HIDDEN_DIM,
    )?;
    orb_mul_in_place(&mut grad_pre0, &cache.layer0_silu_derivative)?;
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    let result = orb_mlp_layer0_input_vjp(&grad_pre0, rows, input_dim, weights);
    #[cfg(not(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    )))]
    let result = orb_linear_input_vjp(
        &grad_pre0,
        rows,
        input_dim,
        weights.layer0_weight,
        ORB_ENCODER_HIDDEN_DIM,
    );
    result
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
fn orb_mlp_layer0_forward(
    input: &[f64],
    rows: usize,
    input_dim: usize,
    weights: OrbMlpRmsNormWeights<'_>,
) -> Result<Vec<f64>, OrbV3OmolError> {
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    if let Some(packed) = weights.fixed_rows_packed {
        if rows == packed.rows
            && input_dim == packed.input_dim
            && input.len() == rows * input_dim
            && weights.layer0_bias.len() == ORB_ENCODER_HIDDEN_DIM
        {
            let mut output = orb_profile_blas_shape!(
                packed.labels.layer0_forward,
                rows,
                input_dim,
                ORB_ENCODER_HIDDEN_DIM,
                input_dim,
                0,
                { packed.layer0_forward.compute_no_trans_a_alloc(input) }
            );
            orb_mkl_packed_linear_bias_add(
                &mut output,
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                weights.layer0_bias,
            );
            return Ok(output);
        }
    }
    orb_linear_forward(
        input,
        rows,
        input_dim,
        weights.layer0_weight,
        weights.layer0_bias,
        ORB_ENCODER_HIDDEN_DIM,
    )
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
fn orb_mlp_layer1_forward(
    input: &[f64],
    rows: usize,
    weights: OrbMlpRmsNormWeights<'_>,
) -> Result<Vec<f64>, OrbV3OmolError> {
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    if let Some(packed) = weights.fixed_rows_packed {
        if rows == packed.rows
            && input.len() == rows * ORB_ENCODER_HIDDEN_DIM
            && weights.layer1_bias.len() == ORB_ENCODER_HIDDEN_DIM
        {
            let mut output = orb_profile_blas_shape!(
                packed.labels.layer1_forward,
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                0,
                { packed.layer1_forward.compute_no_trans_a_alloc(input) }
            );
            orb_mkl_packed_linear_bias_add(
                &mut output,
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                weights.layer1_bias,
            );
            return Ok(output);
        }
    }
    orb_linear_forward(
        input,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        weights.layer1_weight,
        weights.layer1_bias,
        ORB_ENCODER_HIDDEN_DIM,
    )
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
fn orb_mlp_layer2_forward(
    input: &[f64],
    rows: usize,
    weights: OrbMlpRmsNormWeights<'_>,
) -> Result<Vec<f64>, OrbV3OmolError> {
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    if let Some(packed) = weights.fixed_rows_packed {
        if rows == packed.rows
            && input.len() == rows * ORB_ENCODER_HIDDEN_DIM
            && weights.layer2_bias.len() == LATENT_DIM
        {
            let mut output = orb_profile_blas_shape!(
                packed.labels.layer2_forward,
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                LATENT_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                0,
                { packed.layer2_forward.compute_no_trans_a_alloc(input) }
            );
            orb_mkl_packed_linear_bias_add(&mut output, rows, LATENT_DIM, weights.layer2_bias);
            return Ok(output);
        }
    }
    orb_linear_forward(
        input,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        weights.layer2_weight,
        weights.layer2_bias,
        LATENT_DIM,
    )
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
fn orb_mlp_layer2_input_vjp(
    grad_output: &[f64],
    rows: usize,
    weights: OrbMlpRmsNormWeights<'_>,
) -> Result<Vec<f64>, OrbV3OmolError> {
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    if let Some(packed) = weights.fixed_rows_packed {
        if rows == packed.rows && grad_output.len() == rows * LATENT_DIM {
            return Ok(orb_profile_blas_shape!(
                packed.labels.layer2_input_vjp,
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                LATENT_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                0,
                {
                    packed
                        .layer2_input_vjp
                        .compute_no_trans_a_alloc(grad_output)
                }
            ));
        }
    }
    orb_linear_input_vjp(
        grad_output,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        weights.layer2_weight,
        LATENT_DIM,
    )
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
fn orb_mlp_layer1_input_vjp(
    grad_output: &[f64],
    rows: usize,
    weights: OrbMlpRmsNormWeights<'_>,
) -> Result<Vec<f64>, OrbV3OmolError> {
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    if let Some(packed) = weights.fixed_rows_packed {
        if rows == packed.rows && grad_output.len() == rows * ORB_ENCODER_HIDDEN_DIM {
            return Ok(orb_profile_blas_shape!(
                packed.labels.layer1_input_vjp,
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                0,
                {
                    packed
                        .layer1_input_vjp
                        .compute_no_trans_a_alloc(grad_output)
                }
            ));
        }
    }
    orb_linear_input_vjp(
        grad_output,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        weights.layer1_weight,
        ORB_ENCODER_HIDDEN_DIM,
    )
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
fn orb_mlp_layer0_input_vjp(
    grad_output: &[f64],
    rows: usize,
    input_dim: usize,
    weights: OrbMlpRmsNormWeights<'_>,
) -> Result<Vec<f64>, OrbV3OmolError> {
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    if let Some(packed) = weights.fixed_rows_packed {
        if rows == packed.rows
            && input_dim == packed.input_dim
            && grad_output.len() == rows * ORB_ENCODER_HIDDEN_DIM
        {
            return Ok(orb_profile_blas_shape!(
                packed.labels.layer0_input_vjp,
                rows,
                input_dim,
                ORB_ENCODER_HIDDEN_DIM,
                input_dim,
                0,
                {
                    packed
                        .layer0_input_vjp
                        .compute_no_trans_a_alloc(grad_output)
                }
            ));
        }
    }
    orb_linear_input_vjp(
        grad_output,
        rows,
        input_dim,
        weights.layer0_weight,
        ORB_ENCODER_HIDDEN_DIM,
    )
}

fn orb_tiled_rows(row: &[f64], rows: usize) -> Vec<f64> {
    let cols = row.len();
    let mut output = vec![0.0; rows * cols];
    for output_row in output.chunks_exact_mut(cols) {
        output_row.copy_from_slice(row);
    }
    output
}

fn orb_edge_mlp_layer0_edge_forward_add(
    edges: &[f64],
    rows: usize,
    weights: OrbMlpRmsNormWeights<'_>,
    output: &mut [f64],
) -> Result<(), OrbV3OmolError> {
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    if rows == 2304
        && edges.len() == rows * LATENT_DIM
        && output.len() == rows * ORB_ENCODER_HIDDEN_DIM
    {
        if let Some(packed) = weights.edge_packed {
            orb_profile_blas_shape!(
                "edge_mlp_layer0_edge_forward_mkl_packed",
                rows,
                LATENT_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                ORB_GNN_INPUT_DIM,
                0,
                {
                    packed
                        .layer0_edge_forward
                        .compute_no_trans_a_into(edges, output, 1.0);
                }
            );
            return Ok(());
        }
    }
    orb_linear_forward_column_block_add(
        edges,
        rows,
        LATENT_DIM,
        weights.layer0_weight,
        ORB_GNN_INPUT_DIM,
        0,
        ORB_ENCODER_HIDDEN_DIM,
        output,
        1.0,
        1.0,
    )
}

fn orb_edge_mlp_layer0_node_forward(
    nodes: &[f64],
    node_rows: usize,
    weights: OrbMlpRmsNormWeights<'_>,
    column_offset: usize,
    output: &mut [f64],
) -> Result<(), OrbV3OmolError> {
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    if node_rows == 128
        && nodes.len() == node_rows * LATENT_DIM
        && output.len() == node_rows * ORB_ENCODER_HIDDEN_DIM
    {
        if let Some(packed) = weights.edge_packed {
            let (label, matrix) = if column_offset == LATENT_DIM {
                (
                    "edge_mlp_layer0_sender_forward_mkl_packed",
                    &packed.layer0_sender_forward,
                )
            } else if column_offset == 2 * LATENT_DIM {
                (
                    "edge_mlp_layer0_receiver_forward_mkl_packed",
                    &packed.layer0_receiver_forward,
                )
            } else {
                ("", &packed.layer0_sender_forward)
            };
            if !label.is_empty() {
                orb_profile_blas_shape!(
                    label,
                    node_rows,
                    LATENT_DIM,
                    ORB_ENCODER_HIDDEN_DIM,
                    ORB_GNN_INPUT_DIM,
                    column_offset,
                    {
                        matrix.compute_no_trans_a_into(nodes, output, 0.0);
                    }
                );
                return Ok(());
            }
        }
    }
    orb_linear_forward_column_block_add(
        nodes,
        node_rows,
        LATENT_DIM,
        weights.layer0_weight,
        ORB_GNN_INPUT_DIM,
        column_offset,
        ORB_ENCODER_HIDDEN_DIM,
        output,
        1.0,
        0.0,
    )
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn orb_f64_to_f32_labeled(
    label: &'static str,
    values: &[f64],
    rows: usize,
    width: usize,
) -> Vec<f32> {
    orb_profile_kernel!(label, rows, width, {
        #[cfg(feature = "parallel")]
        {
            if rows >= ORB_F32_CAST_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
                values.par_iter().map(|&value| value as f32).collect()
            } else {
                values.iter().map(|&value| value as f32).collect()
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            values.iter().map(|&value| value as f32).collect()
        }
    })
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn orb_f32_to_f64_labeled(
    label: &'static str,
    values: &[f32],
    rows: usize,
    width: usize,
) -> Vec<f64> {
    orb_profile_kernel!(label, rows, width, {
        #[cfg(feature = "parallel")]
        {
            if rows >= ORB_F32_CAST_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
                values.par_iter().map(|&value| f64::from(value)).collect()
            } else {
                values.iter().map(|&value| f64::from(value)).collect()
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            values.iter().map(|&value| f64::from(value)).collect()
        }
    })
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn orb_mlp_rms_norm_forward_f32(
    input: &[f64],
    rows: usize,
    input_dim: usize,
    weights: OrbMlpRmsNormWeights<'_>,
    f32_dense: &OrbMlpF32Dense,
) -> Result<Option<(Vec<f64>, OrbMlpRmsNormForwardCache)>, OrbV3OmolError> {
    if input_dim != f32_dense.input_dim || input.len() != rows * input_dim {
        return Ok(None);
    }
    let input_f32 =
        orb_f64_to_f32_labeled("mlp_layer0_forward_f32_cast_in", input, rows, input_dim);
    let mut layer0_f32 = vec![0.0_f32; rows * ORB_ENCODER_HIDDEN_DIM];
    orb_profile_blas_shape!(
        "mlp_layer0_forward_f32_sgemm",
        rows,
        input_dim,
        ORB_ENCODER_HIDDEN_DIM,
        input_dim,
        0,
        {
            crate::blas_backend::dense_forward_output_major_weight_f32(
                &input_f32,
                rows,
                input_dim,
                &f32_dense.layer0_weight,
                &mut layer0_f32,
                ORB_ENCODER_HIDDEN_DIM,
                1.0,
                0.0,
            );
        }
    );
    let layer0_silu_derivative_f32 = orb_add_bias_silu_in_place_with_derivative_f32_labeled(
        "mlp_layer0_bias_silu_with_derivative_f32",
        &mut layer0_f32,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        &f32_dense.layer0_bias,
    )?;

    let mut layer1_f32 = vec![0.0_f32; rows * ORB_ENCODER_HIDDEN_DIM];
    orb_profile_blas_shape!(
        "mlp_layer1_forward_f32_sgemm",
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        0,
        {
            crate::blas_backend::dense_forward_output_major_weight_f32(
                &layer0_f32,
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                &f32_dense.layer1_weight,
                &mut layer1_f32,
                ORB_ENCODER_HIDDEN_DIM,
                1.0,
                0.0,
            );
        }
    );
    let layer1_silu_derivative_f32 = orb_add_bias_silu_in_place_with_derivative_f32_labeled(
        "mlp_layer1_bias_silu_with_derivative_f32",
        &mut layer1_f32,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        &f32_dense.layer1_bias,
    )?;

    let mut layer2_f32 = vec![0.0_f32; rows * LATENT_DIM];
    orb_profile_blas_shape!(
        "mlp_layer2_forward_f32_sgemm",
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        LATENT_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        0,
        {
            crate::blas_backend::dense_forward_output_major_weight_f32(
                &layer1_f32,
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                &f32_dense.layer2_weight,
                &mut layer2_f32,
                LATENT_DIM,
                1.0,
                0.0,
            );
        }
    );
    orb_add_bias_in_place_f32_labeled(
        "mlp_layer2_forward_f32_bias_add",
        &mut layer2_f32,
        rows,
        LATENT_DIM,
        &f32_dense.layer2_bias,
    )?;
    let (output, layer2_inv_rms_f32) = orb_rms_norm_forward_with_inv_rms_f32_labeled(
        "mlp_rms_norm_forward_f32",
        &layer2_f32,
        rows,
        LATENT_DIM,
        weights.rms_weight,
    )?;
    Ok(Some((
        output,
        OrbMlpRmsNormForwardCache {
            layer0_silu_derivative: Vec::new(),
            layer0_silu_derivative_f32: Some(layer0_silu_derivative_f32),
            layer1_silu_derivative: Vec::new(),
            layer1_silu_derivative_f32: Some(layer1_silu_derivative_f32),
            layer2: Vec::new(),
            layer2_f32: Some(layer2_f32),
            layer2_inv_rms: Vec::new(),
            layer2_inv_rms_f32: Some(layer2_inv_rms_f32),
        },
    )))
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn orb_mlp_rms_norm_input_vjp_f32(
    grad_layer2_f32: &[f32],
    cache: &OrbMlpRmsNormForwardCache,
    rows: usize,
    input_dim: usize,
    f32_dense: &OrbMlpF32Dense,
) -> Result<Option<Vec<f64>>, OrbV3OmolError> {
    if input_dim != f32_dense.input_dim || grad_layer2_f32.len() != rows * LATENT_DIM {
        return Ok(None);
    }
    let Some(layer0_silu_derivative) = cache.layer0_silu_derivative_f32.as_deref() else {
        return Ok(None);
    };
    let Some(layer1_silu_derivative) = cache.layer1_silu_derivative_f32.as_deref() else {
        return Ok(None);
    };
    expect_len(
        "ORB f32 MLP layer0 SiLU derivative",
        layer0_silu_derivative,
        rows * ORB_ENCODER_HIDDEN_DIM,
    )?;
    expect_len(
        "ORB f32 MLP layer1 SiLU derivative",
        layer1_silu_derivative,
        rows * ORB_ENCODER_HIDDEN_DIM,
    )?;

    let mut grad_pre1_f32 = vec![0.0_f32; rows * ORB_ENCODER_HIDDEN_DIM];
    orb_profile_blas_shape!(
        "mlp_layer2_input_vjp_f32_sgemm",
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        LATENT_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        0,
        {
            crate::blas_backend::dense_vjp_input_output_major_weight_f32(
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                LATENT_DIM,
                &f32_dense.layer2_weight,
                grad_layer2_f32,
                &mut grad_pre1_f32,
                1.0,
                0.0,
            );
        }
    );
    orb_mul_in_place_f32_labeled(
        "mlp_layer1_derivative_mul_f32",
        &mut grad_pre1_f32,
        layer1_silu_derivative,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
    )?;

    let mut grad_pre0_f32 = vec![0.0_f32; rows * ORB_ENCODER_HIDDEN_DIM];
    orb_profile_blas_shape!(
        "mlp_layer1_input_vjp_f32_sgemm",
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        0,
        {
            crate::blas_backend::dense_vjp_input_output_major_weight_f32(
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                &f32_dense.layer1_weight,
                &grad_pre1_f32,
                &mut grad_pre0_f32,
                1.0,
                0.0,
            );
        }
    );
    orb_mul_in_place_f32_labeled(
        "mlp_layer0_derivative_mul_f32",
        &mut grad_pre0_f32,
        layer0_silu_derivative,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
    )?;

    let mut grad_input_f32 = vec![0.0_f32; rows * input_dim];
    orb_profile_blas_shape!(
        "mlp_layer0_input_vjp_f32_sgemm",
        rows,
        input_dim,
        ORB_ENCODER_HIDDEN_DIM,
        input_dim,
        0,
        {
            crate::blas_backend::dense_vjp_input_output_major_weight_f32(
                rows,
                input_dim,
                ORB_ENCODER_HIDDEN_DIM,
                &f32_dense.layer0_weight,
                &grad_pre0_f32,
                &mut grad_input_f32,
                1.0,
                0.0,
            );
        }
    );
    Ok(Some(orb_f32_to_f64_labeled(
        "mlp_layer0_input_vjp_f32_cast_out",
        &grad_input_f32,
        rows,
        input_dim,
    )))
}

fn orb_edge_mlp_layer1_forward(
    input: &[f64],
    rows: usize,
    weights: OrbMlpRmsNormWeights<'_>,
) -> Result<Vec<f64>, OrbV3OmolError> {
    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    if input.len() == rows * ORB_ENCODER_HIDDEN_DIM {
        if let Some(f32_dense) = weights.edge_f32 {
            let input_f32 = orb_f64_to_f32_labeled(
                "edge_mlp_layer1_forward_f32_cast_in",
                input,
                rows,
                ORB_ENCODER_HIDDEN_DIM,
            );
            let mut output_f32 = vec![0.0_f32; rows * ORB_ENCODER_HIDDEN_DIM];
            orb_profile_blas_shape!(
                "edge_mlp_layer1_forward_f32_sgemm",
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                0,
                {
                    crate::blas_backend::dense_forward_output_major_weight_f32(
                        &input_f32,
                        rows,
                        ORB_ENCODER_HIDDEN_DIM,
                        &f32_dense.layer1_weight,
                        &mut output_f32,
                        ORB_ENCODER_HIDDEN_DIM,
                        1.0,
                        0.0,
                    );
                }
            );
            return Ok(orb_profile_kernel!(
                "edge_mlp_layer1_forward_f32_bias_cast_out",
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                {
                    let mut output = Vec::with_capacity(rows * ORB_ENCODER_HIDDEN_DIM);
                    for output_row in output_f32.chunks_exact(ORB_ENCODER_HIDDEN_DIM) {
                        for (value, bias) in output_row.iter().zip(&f32_dense.layer1_bias) {
                            output.push(f64::from(*value + *bias));
                        }
                    }
                    output
                }
            ));
        }
    }
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    if rows == 2304
        && input.len() == rows * ORB_ENCODER_HIDDEN_DIM
        && weights.layer1_bias.len() == ORB_ENCODER_HIDDEN_DIM
    {
        if let Some(packed) = weights.edge_packed {
            let mut output = orb_profile_blas_shape!(
                "edge_mlp_layer1_forward_mkl_packed",
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                0,
                { packed.layer1_forward.compute_no_trans_a_alloc(input) }
            );
            orb_mkl_packed_linear_bias_add(
                &mut output,
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                weights.layer1_bias,
            );
            return Ok(output);
        }
    }
    orb_linear_forward_labeled(
        "edge_mlp_layer1_forward",
        input,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        weights.layer1_weight,
        weights.layer1_bias,
        ORB_ENCODER_HIDDEN_DIM,
    )
}

fn orb_edge_mlp_layer1_forward_with_silu_derivative(
    input: &[f64],
    rows: usize,
    weights: OrbMlpRmsNormWeights<'_>,
) -> Result<(Vec<f64>, Vec<f64>), OrbV3OmolError> {
    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    if input.len() == rows * ORB_ENCODER_HIDDEN_DIM && weights.edge_f32.is_some() {
        let mut output = orb_edge_mlp_layer1_forward(input, rows, weights)?;
        let derivative = orb_silu_in_place_with_derivative_labeled(
            "edge_mlp_layer1_silu_with_derivative",
            &mut output,
        );
        return Ok((output, derivative));
    }

    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    if rows == 2304
        && input.len() == rows * ORB_ENCODER_HIDDEN_DIM
        && weights.layer1_bias.len() == ORB_ENCODER_HIDDEN_DIM
    {
        if let Some(packed) = weights.edge_packed {
            let mut output = orb_profile_blas_shape!(
                "edge_mlp_layer1_forward_mkl_packed",
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                0,
                { packed.layer1_forward.compute_no_trans_a_alloc(input) }
            );
            let derivative = orb_add_bias_silu_in_place_with_derivative_labeled(
                "edge_mlp_layer1_bias_silu_with_derivative_mkl_packed",
                &mut output,
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                weights.layer1_bias,
            )?;
            return Ok((output, derivative));
        }
    }

    let mut output = orb_edge_mlp_layer1_forward(input, rows, weights)?;
    let derivative = orb_silu_in_place_with_derivative_labeled(
        "edge_mlp_layer1_silu_with_derivative",
        &mut output,
    );
    Ok((output, derivative))
}

fn orb_edge_mlp_layer1_input_vjp(
    grad_output: &[f64],
    rows: usize,
    weights: OrbMlpRmsNormWeights<'_>,
) -> Result<Vec<f64>, OrbV3OmolError> {
    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    if grad_output.len() == rows * ORB_ENCODER_HIDDEN_DIM {
        if let Some(f32_dense) = weights.edge_f32 {
            let grad_output_f32 = orb_f64_to_f32_labeled(
                "edge_mlp_layer1_input_vjp_f32_cast_in",
                grad_output,
                rows,
                ORB_ENCODER_HIDDEN_DIM,
            );
            let mut grad_input_f32 = vec![0.0_f32; rows * ORB_ENCODER_HIDDEN_DIM];
            orb_profile_blas_shape!(
                "edge_mlp_layer1_input_vjp_f32_sgemm",
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                0,
                {
                    crate::blas_backend::dense_vjp_input_input_major_weight_f32(
                        rows,
                        ORB_ENCODER_HIDDEN_DIM,
                        ORB_ENCODER_HIDDEN_DIM,
                        &f32_dense.layer1_weight_input_major,
                        &grad_output_f32,
                        &mut grad_input_f32,
                        1.0,
                        0.0,
                    );
                }
            );
            return Ok(orb_f32_to_f64_labeled(
                "edge_mlp_layer1_input_vjp_f32_cast_out",
                &grad_input_f32,
                rows,
                ORB_ENCODER_HIDDEN_DIM,
            ));
        }
    }
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    if rows == 2304 && grad_output.len() == rows * ORB_ENCODER_HIDDEN_DIM {
        if let Some(packed) = weights.edge_packed {
            return Ok(orb_profile_blas_shape!(
                "edge_mlp_layer1_input_vjp_mkl_packed",
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                0,
                {
                    packed
                        .layer1_input_vjp
                        .compute_no_trans_a_alloc(grad_output)
                }
            ));
        }
    }
    orb_linear_input_vjp_labeled(
        "edge_mlp_layer1_input_vjp",
        grad_output,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        weights.layer1_weight,
        ORB_ENCODER_HIDDEN_DIM,
    )
}

fn orb_edge_mlp_layer2_forward(
    input: &[f64],
    rows: usize,
    weights: OrbMlpRmsNormWeights<'_>,
) -> Result<Vec<f64>, OrbV3OmolError> {
    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    if input.len() == rows * ORB_ENCODER_HIDDEN_DIM {
        if let Some(f32_dense) = weights.edge_f32 {
            let input_f32 = orb_f64_to_f32_labeled(
                "edge_mlp_layer2_forward_f32_cast_in",
                input,
                rows,
                ORB_ENCODER_HIDDEN_DIM,
            );
            let mut output_f32 = vec![0.0_f32; rows * LATENT_DIM];
            orb_profile_blas_shape!(
                "edge_mlp_layer2_forward_f32_sgemm",
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                LATENT_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                0,
                {
                    crate::blas_backend::dense_forward_output_major_weight_f32(
                        &input_f32,
                        rows,
                        ORB_ENCODER_HIDDEN_DIM,
                        &f32_dense.layer2_weight,
                        &mut output_f32,
                        LATENT_DIM,
                        1.0,
                        0.0,
                    );
                }
            );
            return Ok(orb_profile_kernel!(
                "edge_mlp_layer2_forward_f32_bias_cast_out",
                rows,
                LATENT_DIM,
                {
                    let mut output = Vec::with_capacity(rows * LATENT_DIM);
                    for output_row in output_f32.chunks_exact(LATENT_DIM) {
                        for (value, bias) in output_row.iter().zip(&f32_dense.layer2_bias) {
                            output.push(f64::from(*value + *bias));
                        }
                    }
                    output
                }
            ));
        }
    }
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    if rows == 2304
        && input.len() == rows * ORB_ENCODER_HIDDEN_DIM
        && weights.layer2_bias.len() == LATENT_DIM
    {
        if let Some(packed) = weights.edge_packed {
            let mut output = orb_profile_blas_shape!(
                "edge_mlp_layer2_forward_mkl_packed",
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                LATENT_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                0,
                { packed.layer2_forward.compute_no_trans_a_alloc(input) }
            );
            orb_mkl_packed_linear_bias_add(&mut output, rows, LATENT_DIM, weights.layer2_bias);
            return Ok(output);
        }
    }
    orb_linear_forward_labeled(
        "edge_mlp_layer2_forward",
        input,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        weights.layer2_weight,
        weights.layer2_bias,
        LATENT_DIM,
    )
}

fn orb_edge_mlp_layer2_input_vjp(
    grad_output: &[f64],
    rows: usize,
    weights: OrbMlpRmsNormWeights<'_>,
) -> Result<Vec<f64>, OrbV3OmolError> {
    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    if grad_output.len() == rows * LATENT_DIM {
        if let Some(f32_dense) = weights.edge_f32 {
            let grad_output_f32 = orb_f64_to_f32_labeled(
                "edge_mlp_layer2_input_vjp_f32_cast_in",
                grad_output,
                rows,
                LATENT_DIM,
            );
            let mut grad_input_f32 = vec![0.0_f32; rows * ORB_ENCODER_HIDDEN_DIM];
            orb_profile_blas_shape!(
                "edge_mlp_layer2_input_vjp_f32_sgemm",
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                LATENT_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                0,
                {
                    crate::blas_backend::dense_vjp_input_output_major_weight_f32(
                        rows,
                        ORB_ENCODER_HIDDEN_DIM,
                        LATENT_DIM,
                        &f32_dense.layer2_weight,
                        &grad_output_f32,
                        &mut grad_input_f32,
                        1.0,
                        0.0,
                    );
                }
            );
            return Ok(orb_f32_to_f64_labeled(
                "edge_mlp_layer2_input_vjp_f32_cast_out",
                &grad_input_f32,
                rows,
                ORB_ENCODER_HIDDEN_DIM,
            ));
        }
    }
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    if rows == 2304 && grad_output.len() == rows * LATENT_DIM {
        if let Some(packed) = weights.edge_packed {
            return Ok(orb_profile_blas_shape!(
                "edge_mlp_layer2_input_vjp_mkl_packed",
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                LATENT_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                0,
                {
                    packed
                        .layer2_input_vjp
                        .compute_no_trans_a_alloc(grad_output)
                }
            ));
        }
    }
    orb_linear_input_vjp_labeled(
        "edge_mlp_layer2_input_vjp",
        grad_output,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        weights.layer2_weight,
        LATENT_DIM,
    )
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn orb_tiled_rows_f32(values: &[f32], rows: usize) -> Vec<f32> {
    orb_profile_kernel!("edge_mlp_layer0_f32_bias_fill", rows, values.len(), {
        let mut output = Vec::with_capacity(rows * values.len());
        for _ in 0..rows {
            output.extend_from_slice(values);
        }
        output
    })
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn orb_add_two_rows_in_place_f32(dst: &mut [f32], lhs: &[f32], rhs: &[f32]) {
    debug_assert_eq!(dst.len(), lhs.len());
    debug_assert_eq!(dst.len(), rhs.len());
    for channel in 0..dst.len() {
        dst[channel] += lhs[channel] + rhs[channel];
    }
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
#[allow(unsafe_code)]
fn orb_silu_in_place_with_derivative_f32_labeled(
    profile_label: &'static str,
    values: &mut [f32],
    rows: usize,
    width: usize,
) -> Result<Vec<f32>, OrbV3OmolError> {
    expect_len("ORB f32 SiLU values", values, rows * width)?;
    let mut derivative = vec![0.0_f32; values.len()];
    #[cfg(feature = "parallel")]
    if values.len() >= ORB_PARALLEL_VALUE_THRESHOLD && rayon::current_num_threads() > 1 {
        return Ok(orb_profile_activation!({
            orb_profile_kernel!(profile_label, rows, width, {
                #[cfg(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))]
                {
                    if orb_libmvec_shim_runtime_available() {
                        values
                            .par_chunks_exact_mut(width)
                            .zip(derivative.par_chunks_exact_mut(width))
                            .for_each(|(value_row, derivative_row)| unsafe {
                                llmlip_orb_f32_silu_derivative_mvec(
                                    value_row.as_mut_ptr(),
                                    derivative_row.as_mut_ptr(),
                                    width,
                                );
                            });
                    } else {
                        values
                            .par_iter_mut()
                            .zip(derivative.par_iter_mut())
                            .for_each(|(value, derivative)| {
                                let input = *value;
                                let sigmoid = orb_sigmoid_f32(input);
                                *derivative = sigmoid * (1.0 + input * (1.0 - sigmoid));
                                *value = input * sigmoid;
                            });
                    }
                    derivative
                }
                #[cfg(not(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu")))]
                {
                    values
                        .par_iter_mut()
                        .zip(derivative.par_iter_mut())
                        .for_each(|(value, derivative)| {
                            let input = *value;
                            let sigmoid = orb_sigmoid_f32(input);
                            *derivative = sigmoid * (1.0 + input * (1.0 - sigmoid));
                            *value = input * sigmoid;
                        });
                    derivative
                }
            })
        }));
    }
    Ok(orb_profile_activation!({
        orb_profile_kernel!(profile_label, rows, width, {
            #[cfg(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))]
            {
                if orb_libmvec_shim_runtime_available() {
                    unsafe {
                        llmlip_orb_f32_silu_derivative_mvec(
                            values.as_mut_ptr(),
                            derivative.as_mut_ptr(),
                            values.len(),
                        );
                    }
                } else {
                    for (value, derivative) in values.iter_mut().zip(&mut derivative) {
                        let input = *value;
                        let sigmoid = orb_sigmoid_f32(input);
                        *derivative = sigmoid * (1.0 + input * (1.0 - sigmoid));
                        *value = input * sigmoid;
                    }
                }
                derivative
            }
            #[cfg(not(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu")))]
            {
                for (value, derivative) in values.iter_mut().zip(&mut derivative) {
                    let input = *value;
                    let sigmoid = orb_sigmoid_f32(input);
                    *derivative = sigmoid * (1.0 + input * (1.0 - sigmoid));
                    *value = input * sigmoid;
                }
                derivative
            }
        })
    }))
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn orb_add_bias_in_place_f32_labeled(
    profile_label: &'static str,
    values: &mut [f32],
    rows: usize,
    width: usize,
    bias: &[f32],
) -> Result<(), OrbV3OmolError> {
    expect_len("ORB f32 bias-add values", values, rows * width)?;
    expect_len("ORB f32 bias-add bias", bias, width)?;
    orb_profile_kernel!(profile_label, rows, width, {
        #[cfg(feature = "parallel")]
        {
            if values.len() >= ORB_PARALLEL_VALUE_THRESHOLD && rayon::current_num_threads() > 1 {
                values.par_chunks_exact_mut(width).for_each(|row| {
                    for channel in 0..width {
                        row[channel] += bias[channel];
                    }
                });
            } else {
                for row in values.chunks_exact_mut(width) {
                    for channel in 0..width {
                        row[channel] += bias[channel];
                    }
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            for row in values.chunks_exact_mut(width) {
                for channel in 0..width {
                    row[channel] += bias[channel];
                }
            }
        }
    });
    Ok(())
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn orb_rms_norm_forward_with_inv_rms_f32_labeled(
    profile_label: &'static str,
    input: &[f32],
    rows: usize,
    width: usize,
    weight: &[f64],
) -> Result<(Vec<f64>, Vec<f32>), OrbV3OmolError> {
    expect_len("ORB f32 RMSNorm input", input, rows * width)?;
    expect_len("ORB f32 RMSNorm weight", weight, width)?;
    let mut output = vec![0.0; rows * width];
    let mut inv_rms_values = vec![0.0_f32; rows];
    orb_profile_kernel!(profile_label, rows, width, {
        #[cfg(feature = "parallel")]
        {
            if rows >= ORB_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
                output
                    .par_chunks_exact_mut(width)
                    .zip(input.par_chunks_exact(width))
                    .zip(inv_rms_values.par_iter_mut())
                    .for_each(|((output_row, input_row), inv_rms_value)| {
                        let mean_square =
                            input_row.iter().map(|value| value * value).sum::<f32>() / width as f32;
                        let inv_rms = (mean_square + f32::EPSILON).sqrt().recip();
                        *inv_rms_value = inv_rms;
                        for channel in 0..width {
                            output_row[channel] =
                                f64::from(input_row[channel] * inv_rms * weight[channel] as f32);
                        }
                    });
            } else {
                for row in 0..rows {
                    let input_row = &input[row * width..(row + 1) * width];
                    let output_row = &mut output[row * width..(row + 1) * width];
                    let mean_square =
                        input_row.iter().map(|value| value * value).sum::<f32>() / width as f32;
                    let inv_rms = (mean_square + f32::EPSILON).sqrt().recip();
                    inv_rms_values[row] = inv_rms;
                    for channel in 0..width {
                        output_row[channel] =
                            f64::from(input_row[channel] * inv_rms * weight[channel] as f32);
                    }
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            for row in 0..rows {
                let input_row = &input[row * width..(row + 1) * width];
                let output_row = &mut output[row * width..(row + 1) * width];
                let mean_square =
                    input_row.iter().map(|value| value * value).sum::<f32>() / width as f32;
                let inv_rms = (mean_square + f32::EPSILON).sqrt().recip();
                inv_rms_values[row] = inv_rms;
                for channel in 0..width {
                    output_row[channel] =
                        f64::from(input_row[channel] * inv_rms * weight[channel] as f32);
                }
            }
        }
    });
    Ok((output, inv_rms_values))
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn orb_rms_norm_input_vjp_with_inv_rms_f32_labeled(
    profile_label: &'static str,
    input: &[f32],
    inv_rms_values: &[f32],
    grad_output: &[f64],
    rows: usize,
    width: usize,
    weight: &[f64],
) -> Result<Vec<f32>, OrbV3OmolError> {
    expect_len("ORB f32 RMSNorm VJP input", input, rows * width)?;
    expect_len("ORB f32 RMSNorm VJP inv_rms", inv_rms_values, rows)?;
    expect_len("ORB f32 RMSNorm VJP grad_output", grad_output, rows * width)?;
    expect_len("ORB f32 RMSNorm VJP weight", weight, width)?;
    let mut grad_input = vec![0.0_f32; rows * width];
    orb_profile_kernel!(profile_label, rows, width, {
        #[cfg(feature = "parallel")]
        {
            if rows >= ORB_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
                grad_input
                    .par_chunks_exact_mut(width)
                    .zip(input.par_chunks_exact(width))
                    .zip(grad_output.par_chunks_exact(width))
                    .zip(inv_rms_values.par_iter())
                    .for_each(
                        |(((grad_input_row, input_row), grad_output_row), &inv_rms)| {
                            let mut weighted_dot = 0.0_f32;
                            for channel in 0..width {
                                weighted_dot += grad_output_row[channel] as f32
                                    * weight[channel] as f32
                                    * input_row[channel];
                            }
                            let inv_rms_cubed = inv_rms * inv_rms * inv_rms;
                            let correction_scale = inv_rms_cubed * weighted_dot / width as f32;
                            for channel in 0..width {
                                grad_input_row[channel] = grad_output_row[channel] as f32
                                    * weight[channel] as f32
                                    * inv_rms
                                    - input_row[channel] * correction_scale;
                            }
                        },
                    );
            } else {
                for row in 0..rows {
                    let input_row = &input[row * width..(row + 1) * width];
                    let grad_output_row = &grad_output[row * width..(row + 1) * width];
                    let grad_input_row = &mut grad_input[row * width..(row + 1) * width];
                    let inv_rms = inv_rms_values[row];
                    let mut weighted_dot = 0.0_f32;
                    for channel in 0..width {
                        weighted_dot += grad_output_row[channel] as f32
                            * weight[channel] as f32
                            * input_row[channel];
                    }
                    let inv_rms_cubed = inv_rms * inv_rms * inv_rms;
                    let correction_scale = inv_rms_cubed * weighted_dot / width as f32;
                    for channel in 0..width {
                        grad_input_row[channel] =
                            grad_output_row[channel] as f32 * weight[channel] as f32 * inv_rms
                                - input_row[channel] * correction_scale;
                    }
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            for row in 0..rows {
                let input_row = &input[row * width..(row + 1) * width];
                let grad_output_row = &grad_output[row * width..(row + 1) * width];
                let grad_input_row = &mut grad_input[row * width..(row + 1) * width];
                let inv_rms = inv_rms_values[row];
                let mut weighted_dot = 0.0_f32;
                for channel in 0..width {
                    weighted_dot += grad_output_row[channel] as f32
                        * weight[channel] as f32
                        * input_row[channel];
                }
                let inv_rms_cubed = inv_rms * inv_rms * inv_rms;
                let correction_scale = inv_rms_cubed * weighted_dot / width as f32;
                for channel in 0..width {
                    grad_input_row[channel] =
                        grad_output_row[channel] as f32 * weight[channel] as f32 * inv_rms
                            - input_row[channel] * correction_scale;
                }
            }
        }
    });
    Ok(grad_input)
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn orb_edge_mlp_forward_f32(
    edges: &[f64],
    nodes: &[f64],
    senders: &[usize],
    receivers: &[usize],
    node_rows: usize,
    weights: OrbMlpRmsNormWeights<'_>,
) -> Result<Option<OrbEdgeMlpF32Forward>, OrbV3OmolError> {
    let edge_rows = senders.len();
    if edges.len() != edge_rows * LATENT_DIM || nodes.len() != node_rows * LATENT_DIM {
        return Ok(None);
    }
    let Some(f32_dense) = weights.edge_f32 else {
        return Ok(None);
    };
    let edges_f32 = orb_f64_to_f32_labeled(
        "edge_mlp_layer0_f32_edge_cast_in",
        edges,
        edge_rows,
        LATENT_DIM,
    );
    let nodes_f32 = orb_f64_to_f32_labeled(
        "edge_mlp_layer0_f32_node_cast_in",
        nodes,
        node_rows,
        LATENT_DIM,
    );
    let mut layer0 = orb_tiled_rows_f32(&f32_dense.layer0_bias, edge_rows);
    orb_profile_blas_shape!(
        "edge_mlp_layer0_edge_forward_f32_sgemm",
        edge_rows,
        LATENT_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        ORB_GNN_INPUT_DIM,
        0,
        {
            crate::blas_backend::dense_forward_output_major_weight_column_block_add_f32(
                &edges_f32,
                edge_rows,
                LATENT_DIM,
                &f32_dense.layer0_weight,
                ORB_GNN_INPUT_DIM,
                0,
                ORB_ENCODER_HIDDEN_DIM,
                &mut layer0,
                1.0,
                1.0,
            );
        }
    );
    let mut sender_projection = vec![0.0_f32; node_rows * ORB_ENCODER_HIDDEN_DIM];
    orb_profile_blas_shape!(
        "edge_mlp_layer0_sender_forward_f32_sgemm",
        node_rows,
        LATENT_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        ORB_GNN_INPUT_DIM,
        LATENT_DIM,
        {
            crate::blas_backend::dense_forward_output_major_weight_column_block_add_f32(
                &nodes_f32,
                node_rows,
                LATENT_DIM,
                &f32_dense.layer0_weight,
                ORB_GNN_INPUT_DIM,
                LATENT_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                &mut sender_projection,
                1.0,
                0.0,
            );
        }
    );
    let mut receiver_projection = vec![0.0_f32; node_rows * ORB_ENCODER_HIDDEN_DIM];
    orb_profile_blas_shape!(
        "edge_mlp_layer0_receiver_forward_f32_sgemm",
        node_rows,
        LATENT_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        ORB_GNN_INPUT_DIM,
        2 * LATENT_DIM,
        {
            crate::blas_backend::dense_forward_output_major_weight_column_block_add_f32(
                &nodes_f32,
                node_rows,
                LATENT_DIM,
                &f32_dense.layer0_weight,
                ORB_GNN_INPUT_DIM,
                2 * LATENT_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                &mut receiver_projection,
                1.0,
                0.0,
            );
        }
    );
    orb_profile_scatter_reduction!({
        orb_profile_kernel!(
            "edge_mlp_layer0_sender_receiver_gather_f32",
            edge_rows,
            ORB_ENCODER_HIDDEN_DIM,
            {
                #[cfg(feature = "parallel")]
                if edge_rows >= ORB_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
                    layer0
                        .par_chunks_exact_mut(ORB_ENCODER_HIDDEN_DIM)
                        .enumerate()
                        .for_each(|(edge, output_row)| {
                            let sender = senders[edge];
                            let receiver = receivers[edge];
                            let sender_row = &sender_projection[sender * ORB_ENCODER_HIDDEN_DIM
                                ..(sender + 1) * ORB_ENCODER_HIDDEN_DIM];
                            let receiver_row = &receiver_projection[receiver
                                * ORB_ENCODER_HIDDEN_DIM
                                ..(receiver + 1) * ORB_ENCODER_HIDDEN_DIM];
                            orb_add_two_rows_in_place_f32(output_row, sender_row, receiver_row);
                        });
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for edge in 0..edge_rows {
                        let output_row = &mut layer0
                            [edge * ORB_ENCODER_HIDDEN_DIM..(edge + 1) * ORB_ENCODER_HIDDEN_DIM];
                        let sender = senders[edge];
                        let receiver = receivers[edge];
                        let sender_row = &sender_projection[sender * ORB_ENCODER_HIDDEN_DIM
                            ..(sender + 1) * ORB_ENCODER_HIDDEN_DIM];
                        let receiver_row = &receiver_projection[receiver * ORB_ENCODER_HIDDEN_DIM
                            ..(receiver + 1) * ORB_ENCODER_HIDDEN_DIM];
                        orb_add_two_rows_in_place_f32(output_row, sender_row, receiver_row);
                    }
                }
                #[cfg(feature = "parallel")]
                if edge_rows < ORB_PARALLEL_ROW_THRESHOLD || rayon::current_num_threads() <= 1 {
                    for edge in 0..edge_rows {
                        let output_row = &mut layer0
                            [edge * ORB_ENCODER_HIDDEN_DIM..(edge + 1) * ORB_ENCODER_HIDDEN_DIM];
                        let sender = senders[edge];
                        let receiver = receivers[edge];
                        let sender_row = &sender_projection[sender * ORB_ENCODER_HIDDEN_DIM
                            ..(sender + 1) * ORB_ENCODER_HIDDEN_DIM];
                        let receiver_row = &receiver_projection[receiver * ORB_ENCODER_HIDDEN_DIM
                            ..(receiver + 1) * ORB_ENCODER_HIDDEN_DIM];
                        orb_add_two_rows_in_place_f32(output_row, sender_row, receiver_row);
                    }
                }
            }
        );
    });
    let layer0_silu_derivative_f32 = orb_silu_in_place_with_derivative_f32_labeled(
        "edge_mlp_layer0_silu_with_derivative_f32",
        &mut layer0,
        edge_rows,
        ORB_ENCODER_HIDDEN_DIM,
    )?;
    let layer12 = orb_edge_mlp_layer1_layer2_forward_f32_from_f32(&layer0, edge_rows, f32_dense)?;
    Ok(Some(OrbEdgeMlpF32Forward {
        layer0_silu_derivative_f32,
        layer2_f32: layer12.layer2_f32,
        layer1_silu_derivative_f32: layer12.layer1_silu_derivative_f32,
    }))
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn orb_edge_mlp_layer1_layer2_forward_f32_from_f32(
    input_f32: &[f32],
    rows: usize,
    f32_dense: &OrbEdgeMlpF32Dense,
) -> Result<OrbEdgeMlpF32Layer12Forward, OrbV3OmolError> {
    expect_len(
        "ORB edge MLP f32 layer1/layer2 input",
        input_f32,
        rows * ORB_ENCODER_HIDDEN_DIM,
    )?;
    let mut layer1_f32 = vec![0.0_f32; rows * ORB_ENCODER_HIDDEN_DIM];
    orb_profile_blas_shape!(
        "edge_mlp_layer1_forward_f32_sgemm",
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        0,
        {
            crate::blas_backend::dense_forward_output_major_weight_f32(
                input_f32,
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                &f32_dense.layer1_weight,
                &mut layer1_f32,
                ORB_ENCODER_HIDDEN_DIM,
                1.0,
                0.0,
            );
        }
    );
    let layer1_silu_derivative_f32 = orb_add_bias_silu_in_place_with_derivative_f32_labeled(
        "edge_mlp_layer1_bias_silu_with_derivative_f32",
        &mut layer1_f32,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        &f32_dense.layer1_bias,
    )?;
    let mut layer2_f32 = vec![0.0_f32; rows * LATENT_DIM];
    orb_profile_blas_shape!(
        "edge_mlp_layer2_forward_f32_sgemm",
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        LATENT_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        0,
        {
            crate::blas_backend::dense_forward_output_major_weight_f32(
                &layer1_f32,
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                &f32_dense.layer2_weight,
                &mut layer2_f32,
                LATENT_DIM,
                1.0,
                0.0,
            );
        }
    );
    orb_add_bias_in_place_f32_labeled(
        "edge_mlp_layer2_forward_f32_bias_add",
        &mut layer2_f32,
        rows,
        LATENT_DIM,
        &f32_dense.layer2_bias,
    )?;
    Ok(OrbEdgeMlpF32Layer12Forward {
        layer2_f32,
        layer1_silu_derivative_f32,
    })
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn orb_edge_mlp_layer1_layer2_forward_f32(
    input: &[f64],
    rows: usize,
    weights: OrbMlpRmsNormWeights<'_>,
) -> Result<Option<OrbEdgeMlpF32Layer12Forward>, OrbV3OmolError> {
    if input.len() != rows * ORB_ENCODER_HIDDEN_DIM {
        return Ok(None);
    }
    let Some(f32_dense) = weights.edge_f32 else {
        return Ok(None);
    };
    let input_f32 = orb_f64_to_f32_labeled(
        "edge_mlp_layer1_layer2_f32_cast_in",
        input,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
    );
    Ok(Some(orb_edge_mlp_layer1_layer2_forward_f32_from_f32(
        &input_f32, rows, f32_dense,
    )?))
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn orb_edge_mlp_layer2_layer1_input_vjp_f32(
    grad_layer2_f32: &[f32],
    layer1_silu_derivative: Option<&[f32]>,
    rows: usize,
    weights: OrbMlpRmsNormWeights<'_>,
) -> Result<Option<Vec<f32>>, OrbV3OmolError> {
    if grad_layer2_f32.len() != rows * LATENT_DIM {
        return Ok(None);
    }
    let Some(f32_dense) = weights.edge_f32 else {
        return Ok(None);
    };
    let Some(layer1_silu_derivative) = layer1_silu_derivative else {
        return Ok(None);
    };
    expect_len(
        "ORB f32 edge MLP layer1 SiLU derivative",
        layer1_silu_derivative,
        rows * ORB_ENCODER_HIDDEN_DIM,
    )?;
    let mut grad_pre1_f32 = vec![0.0_f32; rows * ORB_ENCODER_HIDDEN_DIM];
    orb_profile_blas_shape!(
        "edge_mlp_layer2_input_vjp_f32_sgemm",
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        LATENT_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        0,
        {
            crate::blas_backend::dense_vjp_input_output_major_weight_f32(
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                LATENT_DIM,
                &f32_dense.layer2_weight,
                grad_layer2_f32,
                &mut grad_pre1_f32,
                1.0,
                0.0,
            );
        }
    );
    orb_mul_in_place_f32_labeled(
        "edge_mlp_layer1_derivative_mul_f32",
        &mut grad_pre1_f32,
        layer1_silu_derivative,
        rows,
        ORB_ENCODER_HIDDEN_DIM,
    )?;
    let mut grad_pre0_f32 = vec![0.0_f32; rows * ORB_ENCODER_HIDDEN_DIM];
    orb_profile_blas_shape!(
        "edge_mlp_layer1_input_vjp_f32_sgemm",
        rows,
        ORB_ENCODER_HIDDEN_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        0,
        {
            crate::blas_backend::dense_vjp_input_input_major_weight_f32(
                rows,
                ORB_ENCODER_HIDDEN_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                &f32_dense.layer1_weight_input_major,
                &grad_pre1_f32,
                &mut grad_pre0_f32,
                1.0,
                0.0,
            );
        }
    );
    Ok(Some(grad_pre0_f32))
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn orb_add_row_in_place_f32(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst += *src;
    }
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn orb_add_two_rows_sequential_in_place_f32(dst: &mut [f32], lhs: &[f32], rhs: &[f32]) {
    debug_assert_eq!(dst.len(), lhs.len());
    debug_assert_eq!(dst.len(), rhs.len());
    for channel in 0..dst.len() {
        dst[channel] += lhs[channel];
        dst[channel] += rhs[channel];
    }
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn orb_add_four_rows_sequential_in_place_f32(
    dst: &mut [f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    debug_assert_eq!(dst.len(), c.len());
    debug_assert_eq!(dst.len(), d.len());
    for channel in 0..dst.len() {
        dst[channel] += a[channel];
        dst[channel] += b[channel];
        dst[channel] += c[channel];
        dst[channel] += d[channel];
    }
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
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
fn scatter_add_node_centric_f32(
    target: &mut [f32],
    source: &[f32],
    node_edge_lists: &[Vec<usize>],
    width: usize,
) {
    target
        .par_chunks_exact_mut(width)
        .zip(node_edge_lists.par_iter())
        .for_each(|(node_row, edges)| {
            let mut quads = edges.chunks_exact(4);
            for quad in &mut quads {
                let a = quad[0];
                let b = quad[1];
                let c = quad[2];
                let d = quad[3];
                let a_row = &source[a * width..(a + 1) * width];
                let b_row = &source[b * width..(b + 1) * width];
                let c_row = &source[c * width..(c + 1) * width];
                let d_row = &source[d * width..(d + 1) * width];
                orb_add_four_rows_sequential_in_place_f32(node_row, a_row, b_row, c_row, d_row);
            }
            let mut pairs = quads.remainder().chunks_exact(2);
            for pair in &mut pairs {
                let lhs = pair[0];
                let rhs = pair[1];
                let lhs_row = &source[lhs * width..(lhs + 1) * width];
                let rhs_row = &source[rhs * width..(rhs + 1) * width];
                orb_add_two_rows_sequential_in_place_f32(node_row, lhs_row, rhs_row);
            }
            if let [edge] = pairs.remainder() {
                let src_row = &source[*edge * width..(*edge + 1) * width];
                orb_add_row_in_place_f32(node_row, src_row);
            }
        });
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn orb_edge_mlp_layer0_input_vjp_f32(
    grad_pre0: &mut [f32],
    layer0_silu_derivative: Option<&[f32]>,
    senders: &[usize],
    receivers: &[usize],
    node_rows: usize,
    weights: OrbMlpRmsNormWeights<'_>,
    cached_node_edge_lists: Option<OrbNodeEdgeListRefs<'_>>,
) -> Result<Option<OrbEdgeMlpInputCotangents>, OrbV3OmolError> {
    #[cfg(not(feature = "parallel"))]
    let _ = cached_node_edge_lists;
    let edge_rows = senders.len();
    if receivers.len() != edge_rows {
        return Ok(None);
    }
    let Some(f32_dense) = weights.edge_f32 else {
        return Ok(None);
    };
    let Some(layer0_silu_derivative) = layer0_silu_derivative else {
        return Ok(None);
    };
    expect_len(
        "ORB f32 edge MLP layer0 VJP preactivation cotangents",
        grad_pre0,
        edge_rows * ORB_ENCODER_HIDDEN_DIM,
    )?;
    expect_len(
        "ORB f32 edge MLP layer0 SiLU derivative",
        layer0_silu_derivative,
        edge_rows * ORB_ENCODER_HIDDEN_DIM,
    )?;
    orb_mul_in_place_f32_labeled(
        "edge_mlp_layer0_derivative_mul_f32",
        grad_pre0,
        layer0_silu_derivative,
        edge_rows,
        ORB_ENCODER_HIDDEN_DIM,
    )?;

    let mut grad_edges_f32 = vec![0.0_f32; edge_rows * LATENT_DIM];
    orb_profile_blas_shape!(
        "edge_mlp_layer0_edge_vjp_f32_sgemm",
        edge_rows,
        LATENT_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        ORB_GNN_INPUT_DIM,
        0,
        {
            crate::blas_backend::dense_vjp_input_output_major_weight_column_block_add_f32(
                edge_rows,
                LATENT_DIM,
                ORB_GNN_INPUT_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                0,
                &f32_dense.layer0_weight,
                grad_pre0,
                &mut grad_edges_f32,
                1.0,
                0.0,
            );
        }
    );

    let mut sender_pre0_f32 = vec![0.0_f32; node_rows * ORB_ENCODER_HIDDEN_DIM];
    let mut receiver_pre0_f32 = vec![0.0_f32; node_rows * ORB_ENCODER_HIDDEN_DIM];
    #[cfg(feature = "parallel")]
    let cached_node_edge_lists = if edge_rows >= ORB_PARALLEL_ROW_THRESHOLD
        && rayon::current_num_threads() > 1
    {
        Some(cached_node_edge_lists.ok_or_else(|| {
            OrbV3OmolError::InvalidSpec(
                "ORB cached node-edge lists are missing for f32 edge MLP VJP scatter".to_owned(),
            )
        })?)
    } else {
        cached_node_edge_lists
    };
    orb_profile_scatter_reduction!({
        orb_profile_kernel!(
            "edge_mlp_vjp_sender_receiver_scatter_f32",
            edge_rows,
            ORB_ENCODER_HIDDEN_DIM,
            {
                #[cfg(feature = "parallel")]
                if edge_rows >= ORB_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
                    if let Some((sender_lists, receiver_lists)) = cached_node_edge_lists {
                        scatter_add_node_centric_f32(
                            &mut sender_pre0_f32,
                            grad_pre0,
                            sender_lists,
                            ORB_ENCODER_HIDDEN_DIM,
                        );
                        scatter_add_node_centric_f32(
                            &mut receiver_pre0_f32,
                            grad_pre0,
                            receiver_lists,
                            ORB_ENCODER_HIDDEN_DIM,
                        );
                    }
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for edge in 0..edge_rows {
                        let grad_row = &grad_pre0
                            [edge * ORB_ENCODER_HIDDEN_DIM..(edge + 1) * ORB_ENCODER_HIDDEN_DIM];
                        let sender = senders[edge];
                        let receiver = receivers[edge];
                        let sender_row = &mut sender_pre0_f32[sender * ORB_ENCODER_HIDDEN_DIM
                            ..(sender + 1) * ORB_ENCODER_HIDDEN_DIM];
                        let receiver_row = &mut receiver_pre0_f32[receiver * ORB_ENCODER_HIDDEN_DIM
                            ..(receiver + 1) * ORB_ENCODER_HIDDEN_DIM];
                        orb_add_row_in_place_f32(sender_row, grad_row);
                        orb_add_row_in_place_f32(receiver_row, grad_row);
                    }
                }
                #[cfg(feature = "parallel")]
                if edge_rows < ORB_PARALLEL_ROW_THRESHOLD || rayon::current_num_threads() <= 1 {
                    for edge in 0..edge_rows {
                        let grad_row = &grad_pre0
                            [edge * ORB_ENCODER_HIDDEN_DIM..(edge + 1) * ORB_ENCODER_HIDDEN_DIM];
                        let sender = senders[edge];
                        let receiver = receivers[edge];
                        let sender_row = &mut sender_pre0_f32[sender * ORB_ENCODER_HIDDEN_DIM
                            ..(sender + 1) * ORB_ENCODER_HIDDEN_DIM];
                        let receiver_row = &mut receiver_pre0_f32[receiver * ORB_ENCODER_HIDDEN_DIM
                            ..(receiver + 1) * ORB_ENCODER_HIDDEN_DIM];
                        orb_add_row_in_place_f32(sender_row, grad_row);
                        orb_add_row_in_place_f32(receiver_row, grad_row);
                    }
                }
            }
        );
    });

    let mut grad_nodes_f32 = vec![0.0_f32; node_rows * LATENT_DIM];
    orb_profile_blas_shape!(
        "edge_mlp_layer0_sender_vjp_f32_sgemm",
        node_rows,
        LATENT_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        ORB_GNN_INPUT_DIM,
        LATENT_DIM,
        {
            crate::blas_backend::dense_vjp_input_output_major_weight_column_block_add_f32(
                node_rows,
                LATENT_DIM,
                ORB_GNN_INPUT_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                LATENT_DIM,
                &f32_dense.layer0_weight,
                &sender_pre0_f32,
                &mut grad_nodes_f32,
                1.0,
                0.0,
            );
        }
    );
    orb_profile_blas_shape!(
        "edge_mlp_layer0_receiver_vjp_f32_sgemm",
        node_rows,
        LATENT_DIM,
        ORB_ENCODER_HIDDEN_DIM,
        ORB_GNN_INPUT_DIM,
        2 * LATENT_DIM,
        {
            crate::blas_backend::dense_vjp_input_output_major_weight_column_block_add_f32(
                node_rows,
                LATENT_DIM,
                ORB_GNN_INPUT_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                2 * LATENT_DIM,
                &f32_dense.layer0_weight,
                &receiver_pre0_f32,
                &mut grad_nodes_f32,
                1.0,
                1.0,
            );
        }
    );

    let grad_edges = orb_f32_to_f64_labeled(
        "edge_mlp_layer0_edge_vjp_f32_cast_out",
        &grad_edges_f32,
        edge_rows,
        LATENT_DIM,
    );
    let grad_nodes = orb_f32_to_f64_labeled(
        "edge_mlp_layer0_node_vjp_f32_cast_out",
        &grad_nodes_f32,
        node_rows,
        LATENT_DIM,
    );
    Ok(Some(OrbEdgeMlpInputCotangents {
        edges: grad_edges,
        nodes: grad_nodes,
    }))
}

fn orb_edge_mlp_layer0_edge_vjp_add(
    grad_pre0: &[f64],
    rows: usize,
    weights: OrbMlpRmsNormWeights<'_>,
    grad_edges: &mut [f64],
) -> Result<(), OrbV3OmolError> {
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    if rows == 2304
        && grad_pre0.len() == rows * ORB_ENCODER_HIDDEN_DIM
        && grad_edges.len() == rows * LATENT_DIM
    {
        if let Some(packed) = weights.edge_packed {
            orb_profile_blas_shape!(
                "edge_mlp_layer0_edge_vjp_mkl_packed",
                rows,
                LATENT_DIM,
                ORB_ENCODER_HIDDEN_DIM,
                ORB_GNN_INPUT_DIM,
                0,
                {
                    packed
                        .layer0_edge_vjp
                        .compute_no_trans_a_into(grad_pre0, grad_edges, 0.0);
                }
            );
            return Ok(());
        }
    }
    orb_linear_vjp_column_block_add(
        grad_pre0,
        rows,
        LATENT_DIM,
        ORB_GNN_INPUT_DIM,
        weights.layer0_weight,
        ORB_ENCODER_HIDDEN_DIM,
        0,
        grad_edges,
        1.0,
        0.0,
    )
}

fn orb_edge_mlp_layer0_node_vjp_add(
    grad_pre0: &[f64],
    node_rows: usize,
    weights: OrbMlpRmsNormWeights<'_>,
    column_offset: usize,
    grad_nodes: &mut [f64],
    beta: f64,
) -> Result<(), OrbV3OmolError> {
    #[cfg(all(
        feature = "orb-fixed-shape-dense",
        feature = "intel-mkl",
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    if node_rows == 128
        && grad_pre0.len() == node_rows * ORB_ENCODER_HIDDEN_DIM
        && grad_nodes.len() == node_rows * LATENT_DIM
    {
        if let Some(packed) = weights.edge_packed {
            let (label, matrix) = if column_offset == LATENT_DIM {
                (
                    "edge_mlp_layer0_sender_vjp_mkl_packed",
                    &packed.layer0_sender_vjp,
                )
            } else if column_offset == 2 * LATENT_DIM {
                (
                    "edge_mlp_layer0_receiver_vjp_mkl_packed",
                    &packed.layer0_receiver_vjp,
                )
            } else {
                ("", &packed.layer0_sender_vjp)
            };
            if !label.is_empty() {
                orb_profile_blas_shape!(
                    label,
                    node_rows,
                    LATENT_DIM,
                    ORB_ENCODER_HIDDEN_DIM,
                    ORB_GNN_INPUT_DIM,
                    column_offset,
                    {
                        matrix.compute_no_trans_a_into(grad_pre0, grad_nodes, beta);
                    }
                );
                return Ok(());
            }
        }
    }
    orb_linear_vjp_column_block_add(
        grad_pre0,
        node_rows,
        LATENT_DIM,
        ORB_GNN_INPUT_DIM,
        weights.layer0_weight,
        ORB_ENCODER_HIDDEN_DIM,
        column_offset,
        grad_nodes,
        1.0,
        beta,
    )
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
fn orb_mkl_packed_linear_bias_add(
    output: &mut [f64],
    rows: usize,
    output_dim: usize,
    bias: &[f64],
) {
    #[cfg(feature = "parallel")]
    if rows >= ORB_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
        orb_profile_kernel!("linear_bias_add", rows, output_dim, {
            output
                .par_chunks_exact_mut(output_dim)
                .for_each(|output_row| {
                    for (value, bias_value) in output_row.iter_mut().zip(bias) {
                        *value += *bias_value;
                    }
                });
        });
        return;
    }
    orb_profile_kernel!("linear_bias_add", rows, output_dim, {
        for output_row in output.chunks_exact_mut(output_dim) {
            for (value, bias_value) in output_row.iter_mut().zip(bias) {
                *value += *bias_value;
            }
        }
    });
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
fn orb_add_bias_silu_in_place_with_derivative_labeled(
    profile_label: &'static str,
    values: &mut [f64],
    rows: usize,
    width: usize,
    bias: &[f64],
) -> Result<Vec<f64>, OrbV3OmolError> {
    expect_len("ORB fused bias+SiLU values", values, rows * width)?;
    expect_len("ORB fused bias+SiLU bias", bias, width)?;
    let mut derivative = vec![0.0; values.len()];
    #[cfg(feature = "parallel")]
    if values.len() >= ORB_PARALLEL_VALUE_THRESHOLD && rayon::current_num_threads() > 1 {
        return Ok(orb_profile_activation!({
            orb_profile_kernel!(profile_label, rows, width, {
                values
                    .par_chunks_exact_mut(width)
                    .zip(derivative.par_chunks_exact_mut(width))
                    .for_each(|(value_row, derivative_row)| {
                        for channel in 0..width {
                            let input = value_row[channel] + bias[channel];
                            let sigmoid = orb_sigmoid(input);
                            derivative_row[channel] = sigmoid * (1.0 + input * (1.0 - sigmoid));
                            value_row[channel] = input * sigmoid;
                        }
                    });
                derivative
            })
        }));
    }
    Ok(orb_profile_activation!({
        orb_profile_kernel!(profile_label, rows, width, {
            for (value_row, derivative_row) in values
                .chunks_exact_mut(width)
                .zip(derivative.chunks_exact_mut(width))
            {
                for channel in 0..width {
                    let input = value_row[channel] + bias[channel];
                    let sigmoid = orb_sigmoid(input);
                    derivative_row[channel] = sigmoid * (1.0 + input * (1.0 - sigmoid));
                    value_row[channel] = input * sigmoid;
                }
            }
            derivative
        })
    }))
}

fn orb_edge_mlp_rms_norm_forward_with_cache(
    edges: &[f64],
    nodes: &[f64],
    senders: &[usize],
    receivers: &[usize],
    node_rows: usize,
    weights: OrbMlpRmsNormWeights<'_>,
) -> Result<(Vec<f64>, OrbMlpRmsNormForwardCache), OrbV3OmolError> {
    let edge_rows = senders.len();
    if receivers.len() != edge_rows {
        return invalid(format!(
            "ORB edge MLP receiver length must be {edge_rows}, got {}",
            receivers.len()
        ));
    }
    expect_len("ORB edge MLP edge features", edges, edge_rows * LATENT_DIM)?;
    expect_len("ORB edge MLP node features", nodes, node_rows * LATENT_DIM)?;
    for edge in 0..edge_rows {
        let sender = senders[edge];
        let receiver = receivers[edge];
        if sender >= node_rows || receiver >= node_rows {
            return invalid(format!(
                "ORB edge MLP edge {edge} sender/receiver out of bounds: {sender}->{receiver}, nodes {node_rows}"
            ));
        }
    }

    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    let f32_forward =
        orb_edge_mlp_forward_f32(edges, nodes, senders, receivers, node_rows, weights)?;
    #[cfg(not(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    )))]
    let f32_forward: Option<OrbEdgeMlpF32Forward> = None;

    let (
        layer2,
        layer2_f32,
        layer0_silu_derivative,
        layer0_silu_derivative_f32,
        layer1_silu_derivative,
        layer1_silu_derivative_f32,
    ) = if let Some(f32_forward) = f32_forward {
        (
            Vec::new(),
            Some(f32_forward.layer2_f32),
            Vec::new(),
            Some(f32_forward.layer0_silu_derivative_f32),
            Vec::new(),
            Some(f32_forward.layer1_silu_derivative_f32),
        )
    } else {
        let mut layer0 = orb_tiled_rows(weights.layer0_bias, edge_rows);
        orb_edge_mlp_layer0_edge_forward_add(edges, edge_rows, weights, &mut layer0)?;
        let mut sender_projection = vec![0.0; node_rows * ORB_ENCODER_HIDDEN_DIM];
        orb_edge_mlp_layer0_node_forward(
            nodes,
            node_rows,
            weights,
            LATENT_DIM,
            &mut sender_projection,
        )?;
        let mut receiver_projection = vec![0.0; node_rows * ORB_ENCODER_HIDDEN_DIM];
        orb_edge_mlp_layer0_node_forward(
            nodes,
            node_rows,
            weights,
            2 * LATENT_DIM,
            &mut receiver_projection,
        )?;
        orb_profile_scatter_reduction!({
            orb_profile_kernel!(
                "edge_mlp_layer0_sender_receiver_gather",
                edge_rows,
                ORB_ENCODER_HIDDEN_DIM,
                {
                    #[cfg(feature = "parallel")]
                    if edge_rows >= ORB_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
                        layer0
                            .par_chunks_exact_mut(ORB_ENCODER_HIDDEN_DIM)
                            .enumerate()
                            .for_each(|(edge, output_row)| {
                                let sender = senders[edge];
                                let receiver = receivers[edge];
                                let sender_row = &sender_projection[sender * ORB_ENCODER_HIDDEN_DIM
                                    ..(sender + 1) * ORB_ENCODER_HIDDEN_DIM];
                                let receiver_row = &receiver_projection[receiver
                                    * ORB_ENCODER_HIDDEN_DIM
                                    ..(receiver + 1) * ORB_ENCODER_HIDDEN_DIM];
                                orb_add_two_rows_in_place(output_row, sender_row, receiver_row);
                            });
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        for edge in 0..edge_rows {
                            let output_row = &mut layer0[edge * ORB_ENCODER_HIDDEN_DIM
                                ..(edge + 1) * ORB_ENCODER_HIDDEN_DIM];
                            let sender = senders[edge];
                            let receiver = receivers[edge];
                            let sender_row = &sender_projection[sender * ORB_ENCODER_HIDDEN_DIM
                                ..(sender + 1) * ORB_ENCODER_HIDDEN_DIM];
                            let receiver_row = &receiver_projection[receiver
                                * ORB_ENCODER_HIDDEN_DIM
                                ..(receiver + 1) * ORB_ENCODER_HIDDEN_DIM];
                            orb_add_two_rows_in_place(output_row, sender_row, receiver_row);
                        }
                    }
                    #[cfg(feature = "parallel")]
                    if edge_rows < ORB_PARALLEL_ROW_THRESHOLD || rayon::current_num_threads() <= 1 {
                        for edge in 0..edge_rows {
                            let output_row = &mut layer0[edge * ORB_ENCODER_HIDDEN_DIM
                                ..(edge + 1) * ORB_ENCODER_HIDDEN_DIM];
                            let sender = senders[edge];
                            let receiver = receivers[edge];
                            let sender_row = &sender_projection[sender * ORB_ENCODER_HIDDEN_DIM
                                ..(sender + 1) * ORB_ENCODER_HIDDEN_DIM];
                            let receiver_row = &receiver_projection[receiver
                                * ORB_ENCODER_HIDDEN_DIM
                                ..(receiver + 1) * ORB_ENCODER_HIDDEN_DIM];
                            orb_add_two_rows_in_place(output_row, sender_row, receiver_row);
                        }
                    }
                }
            );
        });

        let layer0_silu_derivative = orb_silu_in_place_with_derivative_labeled(
            "edge_mlp_layer0_silu_with_derivative",
            &mut layer0,
        );
        #[cfg(all(
            feature = "orb-f32-edge-mlp",
            any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    any(feature = "intel-mkl", feature = "openblas"),
                    target_os = "linux",
                    target_arch = "x86_64"
                )
            )
        ))]
        let f32_layer1_layer2 =
            orb_edge_mlp_layer1_layer2_forward_f32(&layer0, edge_rows, weights)?;
        #[cfg(not(all(
            feature = "orb-f32-edge-mlp",
            any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    any(feature = "intel-mkl", feature = "openblas"),
                    target_os = "linux",
                    target_arch = "x86_64"
                )
            )
        )))]
        let f32_layer1_layer2: Option<OrbEdgeMlpF32Layer12Forward> = None;

        if let Some(f32_forward) = f32_layer1_layer2 {
            (
                Vec::new(),
                Some(f32_forward.layer2_f32),
                layer0_silu_derivative,
                None,
                Vec::new(),
                Some(f32_forward.layer1_silu_derivative_f32),
            )
        } else {
            let (layer1, layer1_silu_derivative) =
                orb_edge_mlp_layer1_forward_with_silu_derivative(&layer0, edge_rows, weights)?;
            let layer2 = orb_edge_mlp_layer2_forward(&layer1, edge_rows, weights)?;
            (
                layer2,
                None,
                layer0_silu_derivative,
                None,
                layer1_silu_derivative,
                None,
            )
        }
    };
    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    let (output, layer2_inv_rms, layer2_inv_rms_f32) = if let Some(layer2_f32_ref) =
        layer2_f32.as_deref()
    {
        let (output, layer2_inv_rms_f32) = orb_rms_norm_forward_with_inv_rms_f32_labeled(
            "edge_mlp_rms_norm_forward_f32",
            layer2_f32_ref,
            edge_rows,
            LATENT_DIM,
            weights.rms_weight,
        )?;
        (output, Vec::new(), Some(layer2_inv_rms_f32))
    } else {
        let (output, layer2_inv_rms) =
            orb_rms_norm_forward_with_inv_rms(&layer2, edge_rows, LATENT_DIM, weights.rms_weight)?;
        (output, layer2_inv_rms, None)
    };
    #[cfg(not(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    )))]
    let (output, layer2_inv_rms) =
        orb_rms_norm_forward_with_inv_rms(&layer2, edge_rows, LATENT_DIM, weights.rms_weight)?;
    #[cfg(not(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    )))]
    {
        let _ = &layer2_f32;
        let _ = &layer0_silu_derivative_f32;
        let _ = &layer1_silu_derivative_f32;
    }
    Ok((
        output,
        OrbMlpRmsNormForwardCache {
            layer0_silu_derivative,
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            layer0_silu_derivative_f32,
            layer1_silu_derivative,
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            layer1_silu_derivative_f32,
            layer2,
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            layer2_f32,
            layer2_inv_rms,
            #[cfg(all(
                feature = "orb-f32-edge-mlp",
                any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        target_arch = "x86_64"
                    )
                )
            ))]
            layer2_inv_rms_f32,
        },
    ))
}

fn orb_edge_mlp_rms_norm_input_vjp_cached(
    cache: &OrbMlpRmsNormForwardCache,
    grad_output: &[f64],
    senders: &[usize],
    receivers: &[usize],
    node_rows: usize,
    weights: OrbMlpRmsNormWeights<'_>,
    cached_node_edge_lists: Option<OrbNodeEdgeListRefs<'_>>,
) -> Result<OrbEdgeMlpInputCotangents, OrbV3OmolError> {
    #[cfg(not(feature = "parallel"))]
    let _ = cached_node_edge_lists;
    let edge_rows = senders.len();
    if receivers.len() != edge_rows {
        return invalid(format!(
            "ORB edge MLP VJP receiver length must be {edge_rows}, got {}",
            receivers.len()
        ));
    }
    for edge in 0..edge_rows {
        let sender = senders[edge];
        let receiver = receivers[edge];
        if sender >= node_rows || receiver >= node_rows {
            return invalid(format!(
                "ORB edge MLP VJP edge {edge} sender/receiver out of bounds: {sender}->{receiver}, nodes {node_rows}"
            ));
        }
    }
    #[cfg(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    ))]
    let f32_grad_pre0 = {
        if weights.edge_f32.is_some() {
            match (
                cache.layer2_f32.as_deref(),
                cache.layer2_inv_rms_f32.as_deref(),
            ) {
                (Some(layer2_f32), Some(layer2_inv_rms_f32)) => {
                    let grad_layer2_f32 = orb_rms_norm_input_vjp_with_inv_rms_f32_labeled(
                        "edge_mlp_rms_norm_input_vjp_f32",
                        layer2_f32,
                        layer2_inv_rms_f32,
                        grad_output,
                        edge_rows,
                        LATENT_DIM,
                        weights.rms_weight,
                    )?;
                    if let Some(mut grad_pre0_f32) = orb_edge_mlp_layer2_layer1_input_vjp_f32(
                        &grad_layer2_f32,
                        cache.layer1_silu_derivative_f32.as_deref(),
                        edge_rows,
                        weights,
                    )? {
                        if let Some(cotangents) = orb_edge_mlp_layer0_input_vjp_f32(
                            &mut grad_pre0_f32,
                            cache.layer0_silu_derivative_f32.as_deref(),
                            senders,
                            receivers,
                            node_rows,
                            weights,
                            cached_node_edge_lists,
                        )? {
                            return Ok(cotangents);
                        }
                        Some(orb_f32_to_f64_labeled(
                            "edge_mlp_layer2_layer1_vjp_f32_cast_out",
                            &grad_pre0_f32,
                            edge_rows,
                            ORB_ENCODER_HIDDEN_DIM,
                        ))
                    } else {
                        return invalid("ORB f32 edge MLP layer1/2 VJP cache could not dispatch");
                    }
                }
                (Some(_), None) | (None, Some(_)) => {
                    return invalid("ORB f32 edge MLP RMSNorm VJP cache is incomplete");
                }
                (None, None) => None,
            }
        } else {
            None
        }
    };
    #[cfg(not(all(
        feature = "orb-f32-edge-mlp",
        any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )
    )))]
    let f32_grad_pre0: Option<Vec<f64>> = None;

    let grad_layer2 = if f32_grad_pre0.is_none() {
        expect_len(
            "ORB cached edge MLP RMSNorm layer2",
            &cache.layer2,
            edge_rows * LATENT_DIM,
        )?;
        expect_len(
            "ORB cached edge MLP RMSNorm layer2 inv_rms",
            &cache.layer2_inv_rms,
            edge_rows,
        )?;
        orb_rms_norm_input_vjp_with_inv_rms(
            &cache.layer2,
            &cache.layer2_inv_rms,
            grad_output,
            edge_rows,
            LATENT_DIM,
            weights.rms_weight,
        )?
    } else {
        Vec::new()
    };

    let mut grad_pre0 = if let Some(grad_pre0) = f32_grad_pre0 {
        expect_len(
            "ORB cached edge MLP RMSNorm layer0 SiLU derivative",
            &cache.layer0_silu_derivative,
            edge_rows * ORB_ENCODER_HIDDEN_DIM,
        )?;
        grad_pre0
    } else {
        expect_len(
            "ORB cached edge MLP RMSNorm layer1 SiLU derivative",
            &cache.layer1_silu_derivative,
            edge_rows * ORB_ENCODER_HIDDEN_DIM,
        )?;
        let mut grad_pre1 = orb_edge_mlp_layer2_input_vjp(&grad_layer2, edge_rows, weights)?;
        orb_mul_in_place_labeled(
            "edge_mlp_layer1_derivative_mul",
            &mut grad_pre1,
            &cache.layer1_silu_derivative,
        )?;
        let grad_pre0 = orb_edge_mlp_layer1_input_vjp(&grad_pre1, edge_rows, weights)?;
        expect_len(
            "ORB cached edge MLP RMSNorm layer0 SiLU derivative",
            &cache.layer0_silu_derivative,
            edge_rows * ORB_ENCODER_HIDDEN_DIM,
        )?;
        grad_pre0
    };
    orb_mul_in_place_labeled(
        "edge_mlp_layer0_derivative_mul",
        &mut grad_pre0,
        &cache.layer0_silu_derivative,
    )?;

    let mut grad_edges = vec![0.0; edge_rows * LATENT_DIM];
    orb_edge_mlp_layer0_edge_vjp_add(&grad_pre0, edge_rows, weights, &mut grad_edges)?;
    let mut sender_pre0 = vec![0.0; node_rows * ORB_ENCODER_HIDDEN_DIM];
    let mut receiver_pre0 = vec![0.0; node_rows * ORB_ENCODER_HIDDEN_DIM];
    #[cfg(feature = "parallel")]
    let cached_node_edge_lists =
        if edge_rows >= ORB_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
            Some(cached_node_edge_lists.ok_or_else(|| {
                OrbV3OmolError::InvalidSpec(
                    "ORB cached node-edge lists are missing for edge MLP VJP scatter".to_owned(),
                )
            })?)
        } else {
            cached_node_edge_lists
        };
    orb_profile_scatter_reduction!({
        orb_profile_kernel!(
            "edge_mlp_vjp_sender_receiver_scatter",
            edge_rows,
            ORB_ENCODER_HIDDEN_DIM,
            {
                #[cfg(feature = "parallel")]
                if edge_rows >= ORB_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
                    if let Some((sender_lists, receiver_lists)) = cached_node_edge_lists {
                        scatter_add_node_centric(
                            &mut sender_pre0,
                            &grad_pre0,
                            sender_lists,
                            ORB_ENCODER_HIDDEN_DIM,
                        );
                        scatter_add_node_centric(
                            &mut receiver_pre0,
                            &grad_pre0,
                            receiver_lists,
                            ORB_ENCODER_HIDDEN_DIM,
                        );
                    }
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for edge in 0..edge_rows {
                        let grad_row = &grad_pre0
                            [edge * ORB_ENCODER_HIDDEN_DIM..(edge + 1) * ORB_ENCODER_HIDDEN_DIM];
                        let sender = senders[edge];
                        let receiver = receivers[edge];
                        let sender_row = &mut sender_pre0[sender * ORB_ENCODER_HIDDEN_DIM
                            ..(sender + 1) * ORB_ENCODER_HIDDEN_DIM];
                        let receiver_row = &mut receiver_pre0[receiver * ORB_ENCODER_HIDDEN_DIM
                            ..(receiver + 1) * ORB_ENCODER_HIDDEN_DIM];
                        orb_add_row_in_place(sender_row, grad_row);
                        orb_add_row_in_place(receiver_row, grad_row);
                    }
                }
                #[cfg(feature = "parallel")]
                if edge_rows < ORB_PARALLEL_ROW_THRESHOLD || rayon::current_num_threads() <= 1 {
                    for edge in 0..edge_rows {
                        let grad_row = &grad_pre0
                            [edge * ORB_ENCODER_HIDDEN_DIM..(edge + 1) * ORB_ENCODER_HIDDEN_DIM];
                        let sender = senders[edge];
                        let receiver = receivers[edge];
                        let sender_row = &mut sender_pre0[sender * ORB_ENCODER_HIDDEN_DIM
                            ..(sender + 1) * ORB_ENCODER_HIDDEN_DIM];
                        let receiver_row = &mut receiver_pre0[receiver * ORB_ENCODER_HIDDEN_DIM
                            ..(receiver + 1) * ORB_ENCODER_HIDDEN_DIM];
                        orb_add_row_in_place(sender_row, grad_row);
                        orb_add_row_in_place(receiver_row, grad_row);
                    }
                }
            }
        );
    });
    let mut grad_nodes = vec![0.0; node_rows * LATENT_DIM];
    orb_edge_mlp_layer0_node_vjp_add(
        &sender_pre0,
        node_rows,
        weights,
        LATENT_DIM,
        &mut grad_nodes,
        0.0,
    )?;
    orb_edge_mlp_layer0_node_vjp_add(
        &receiver_pre0,
        node_rows,
        weights,
        2 * LATENT_DIM,
        &mut grad_nodes,
        1.0,
    )?;
    Ok(OrbEdgeMlpInputCotangents {
        edges: grad_edges,
        nodes: grad_nodes,
    })
}

fn orb_linear_forward(
    input: &[f64],
    rows: usize,
    input_dim: usize,
    weight: &[f64],
    bias: &[f64],
    output_dim: usize,
) -> Result<Vec<f64>, OrbV3OmolError> {
    orb_linear_forward_labeled(
        "linear_forward",
        input,
        rows,
        input_dim,
        weight,
        bias,
        output_dim,
    )
}

fn orb_linear_forward_labeled(
    profile_label: &'static str,
    input: &[f64],
    rows: usize,
    input_dim: usize,
    weight: &[f64],
    bias: &[f64],
    output_dim: usize,
) -> Result<Vec<f64>, OrbV3OmolError> {
    let _ = profile_label;
    if input.len() != rows * input_dim {
        return invalid(format!(
            "ORB linear input length must be {}, got {}",
            rows * input_dim,
            input.len()
        ));
    }
    if weight.len() != output_dim * input_dim {
        return invalid(format!(
            "ORB linear weight length must be {}, got {}",
            output_dim * input_dim,
            weight.len()
        ));
    }
    if bias.len() != output_dim {
        return invalid(format!(
            "ORB linear bias length must be {output_dim}, got {}",
            bias.len()
        ));
    }

    if rows == 0 {
        return Ok(Vec::new());
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        ),
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
        all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
    ))]
    {
        let mut output =
            orb_profile_blas_shape!(profile_label, rows, input_dim, output_dim, input_dim, 0, {
                crate::blas_backend::dense_forward_output_major_weight_alloc(
                    input, rows, input_dim, weight, output_dim, 1.0,
                )
            });
        #[cfg(feature = "parallel")]
        if rows >= ORB_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
            orb_profile_kernel!("linear_bias_add", rows, output_dim, {
                output
                    .par_chunks_exact_mut(output_dim)
                    .for_each(|output_row| {
                        for (value, bias_value) in output_row.iter_mut().zip(bias) {
                            *value += *bias_value;
                        }
                    });
            });
            return Ok(output);
        }
        orb_profile_kernel!("linear_bias_add", rows, output_dim, {
            for output_row in output.chunks_exact_mut(output_dim) {
                for (value, bias_value) in output_row.iter_mut().zip(bias) {
                    *value += *bias_value;
                }
            }
        });
        Ok(output)
    }
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        ),
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
        all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
    )))]
    {
        let mut output = vec![0.0; rows * output_dim];
        for row in 0..rows {
            let input_row = &input[row * input_dim..(row + 1) * input_dim];
            let output_row = &mut output[row * output_dim..(row + 1) * output_dim];
            for out_col in 0..output_dim {
                let weight_row = &weight[out_col * input_dim..(out_col + 1) * input_dim];
                let mut value = bias[out_col];
                for (&input_value, &weight_value) in input_row.iter().zip(weight_row) {
                    value += input_value * weight_value;
                }
                output_row[out_col] = value;
            }
        }
        Ok(output)
    }
}

#[allow(clippy::too_many_arguments)]
fn orb_linear_forward_column_block_add(
    input: &[f64],
    rows: usize,
    selected_in_dim: usize,
    weight: &[f64],
    full_in_dim: usize,
    column_offset: usize,
    out_dim: usize,
    output: &mut [f64],
    alpha: f64,
    beta: f64,
) -> Result<(), OrbV3OmolError> {
    if column_offset > full_in_dim || selected_in_dim > full_in_dim.saturating_sub(column_offset) {
        return invalid(format!(
            "ORB column-block linear offset {column_offset} width {selected_in_dim} exceeds full input dim {full_in_dim}"
        ));
    }
    if input.len() != rows * selected_in_dim {
        return invalid(format!(
            "ORB column-block linear input length must be {}, got {}",
            rows * selected_in_dim,
            input.len()
        ));
    }
    if weight.len() != out_dim * full_in_dim {
        return invalid(format!(
            "ORB column-block linear weight length must be {}, got {}",
            out_dim * full_in_dim,
            weight.len()
        ));
    }
    if output.len() != rows * out_dim {
        return invalid(format!(
            "ORB column-block linear output length must be {}, got {}",
            rows * out_dim,
            output.len()
        ));
    }
    if rows == 0 {
        return Ok(());
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        ),
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
        all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
    ))]
    {
        orb_profile_blas_shape!(
            "linear_forward_column_block_add",
            rows,
            selected_in_dim,
            out_dim,
            full_in_dim,
            column_offset,
            {
                crate::blas_backend::dense_forward_output_major_weight_column_block_add(
                    input,
                    rows,
                    selected_in_dim,
                    weight,
                    full_in_dim,
                    column_offset,
                    out_dim,
                    output,
                    alpha,
                    beta,
                );
            }
        );
        Ok(())
    }
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        ),
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
        all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
    )))]
    {
        for row in 0..rows {
            let input_row = &input[row * selected_in_dim..(row + 1) * selected_in_dim];
            let output_row = &mut output[row * out_dim..(row + 1) * out_dim];
            for out_col in 0..out_dim {
                let weight_row = &weight[out_col * full_in_dim + column_offset
                    ..out_col * full_in_dim + column_offset + selected_in_dim];
                let mut value = if beta == 0.0 {
                    0.0
                } else {
                    beta * output_row[out_col]
                };
                for (&input_value, &weight_value) in input_row.iter().zip(weight_row) {
                    value += alpha * input_value * weight_value;
                }
                output_row[out_col] = value;
            }
        }
        Ok(())
    }
}

fn orb_linear_input_vjp(
    grad_output: &[f64],
    rows: usize,
    input_dim: usize,
    weight: &[f64],
    output_dim: usize,
) -> Result<Vec<f64>, OrbV3OmolError> {
    orb_linear_input_vjp_labeled(
        "linear_input_vjp",
        grad_output,
        rows,
        input_dim,
        weight,
        output_dim,
    )
}

fn orb_linear_input_vjp_labeled(
    profile_label: &'static str,
    grad_output: &[f64],
    rows: usize,
    input_dim: usize,
    weight: &[f64],
    output_dim: usize,
) -> Result<Vec<f64>, OrbV3OmolError> {
    let _ = profile_label;
    if grad_output.len() != rows * output_dim {
        return invalid(format!(
            "ORB linear VJP grad_output length must be {}, got {}",
            rows * output_dim,
            grad_output.len()
        ));
    }
    if weight.len() != output_dim * input_dim {
        return invalid(format!(
            "ORB linear VJP weight length must be {}, got {}",
            output_dim * input_dim,
            weight.len()
        ));
    }
    if rows == 0 {
        return Ok(Vec::new());
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        ),
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
        all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
    ))]
    {
        let grad_input =
            orb_profile_blas_shape!(profile_label, rows, input_dim, output_dim, input_dim, 0, {
                crate::blas_backend::dense_vjp_input_output_major_weight_alloc(
                    rows,
                    input_dim,
                    output_dim,
                    weight,
                    grad_output,
                    1.0,
                )
            });
        Ok(grad_input)
    }
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        ),
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
        all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
    )))]
    {
        let mut grad_input = vec![0.0; rows * input_dim];
        for row in 0..rows {
            let grad_row = &grad_output[row * output_dim..(row + 1) * output_dim];
            let grad_input_row = &mut grad_input[row * input_dim..(row + 1) * input_dim];
            for out_col in 0..output_dim {
                let grad = grad_row[out_col];
                let weight_row = &weight[out_col * input_dim..(out_col + 1) * input_dim];
                for input_col in 0..input_dim {
                    grad_input_row[input_col] += grad * weight_row[input_col];
                }
            }
        }
        Ok(grad_input)
    }
}

#[allow(clippy::too_many_arguments)]
fn orb_linear_vjp_column_block_add(
    grad_output: &[f64],
    rows: usize,
    selected_in_dim: usize,
    full_in_dim: usize,
    weight: &[f64],
    output_dim: usize,
    column_offset: usize,
    grad_input: &mut [f64],
    alpha: f64,
    beta: f64,
) -> Result<(), OrbV3OmolError> {
    if column_offset > full_in_dim || selected_in_dim > full_in_dim.saturating_sub(column_offset) {
        return invalid(format!(
            "ORB column-block linear VJP offset {column_offset} width {selected_in_dim} exceeds full input dim {full_in_dim}"
        ));
    }
    if grad_output.len() != rows * output_dim {
        return invalid(format!(
            "ORB column-block linear VJP grad_output length must be {}, got {}",
            rows * output_dim,
            grad_output.len()
        ));
    }
    if weight.len() != output_dim * full_in_dim {
        return invalid(format!(
            "ORB column-block linear VJP weight length must be {}, got {}",
            output_dim * full_in_dim,
            weight.len()
        ));
    }
    if grad_input.len() != rows * selected_in_dim {
        return invalid(format!(
            "ORB column-block linear VJP grad_input length must be {}, got {}",
            rows * selected_in_dim,
            grad_input.len()
        ));
    }
    if rows == 0 {
        return Ok(());
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        ),
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
        all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
    ))]
    {
        orb_profile_blas_shape!(
            "linear_vjp_column_block_add",
            rows,
            selected_in_dim,
            output_dim,
            full_in_dim,
            column_offset,
            {
                crate::blas_backend::dense_vjp_input_output_major_weight_column_block_add(
                    rows,
                    selected_in_dim,
                    full_in_dim,
                    output_dim,
                    column_offset,
                    weight,
                    grad_output,
                    grad_input,
                    alpha,
                    beta,
                );
            }
        );
        Ok(())
    }
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        ),
        all(feature = "openblas", target_os = "linux", target_arch = "x86_64"),
        all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
    )))]
    {
        if beta == 0.0 {
            grad_input.fill(0.0);
        } else if beta != 1.0 {
            for value in grad_input.iter_mut() {
                *value *= beta;
            }
        }
        for row in 0..rows {
            let grad_row = &grad_output[row * output_dim..(row + 1) * output_dim];
            let grad_input_row =
                &mut grad_input[row * selected_in_dim..(row + 1) * selected_in_dim];
            for out_col in 0..output_dim {
                let grad = alpha * grad_row[out_col];
                let weight_row = &weight[out_col * full_in_dim + column_offset
                    ..out_col * full_in_dim + column_offset + selected_in_dim];
                for input_col in 0..selected_in_dim {
                    grad_input_row[input_col] += grad * weight_row[input_col];
                }
            }
        }
        Ok(())
    }
}

fn orb_silu(value: f64) -> f64 {
    value / (1.0 + (-value).exp())
}

fn orb_silu_in_place_with_derivative(values: &mut [f64]) -> Vec<f64> {
    orb_silu_in_place_with_derivative_labeled("silu_with_derivative", values)
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
#[allow(dead_code)]
#[allow(unsafe_code)]
unsafe extern "C" {
    fn llmlip_orb_f64_silu_derivative_mvec(values: *mut f64, derivative: *mut f64, len: usize);
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
#[allow(dead_code)]
#[inline]
fn orb_libmvec_shim_runtime_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[allow(unsafe_code)]
fn orb_silu_in_place_with_derivative_labeled(
    profile_label: &'static str,
    values: &mut [f64],
) -> Vec<f64> {
    let len = values.len();
    #[cfg(feature = "parallel")]
    if values.len() >= ORB_PARALLEL_VALUE_THRESHOLD && rayon::current_num_threads() > 1 {
        return orb_profile_activation!({
            orb_profile_kernel!(profile_label, len, 1, {
                #[cfg(all(
                    feature = "orb-f32-edge-mlp",
                    target_os = "linux",
                    target_arch = "x86_64",
                    target_env = "gnu"
                ))]
                {
                    let mut derivative = vec![0.0; values.len()];
                    if orb_libmvec_shim_runtime_available() {
                        values
                            .par_chunks_mut(4096)
                            .zip(derivative.par_chunks_mut(4096))
                            .for_each(|(value_chunk, derivative_chunk)| unsafe {
                                llmlip_orb_f64_silu_derivative_mvec(
                                    value_chunk.as_mut_ptr(),
                                    derivative_chunk.as_mut_ptr(),
                                    value_chunk.len(),
                                );
                            });
                    } else {
                        values
                            .par_iter_mut()
                            .zip(derivative.par_iter_mut())
                            .for_each(|(value, derivative)| {
                                let input = *value;
                                let sigmoid = orb_sigmoid(input);
                                *value = input * sigmoid;
                                *derivative = sigmoid * (1.0 + input * (1.0 - sigmoid));
                            });
                    }
                    derivative
                }
                #[cfg(not(all(
                    feature = "orb-f32-edge-mlp",
                    target_os = "linux",
                    target_arch = "x86_64",
                    target_env = "gnu"
                )))]
                {
                    let derivative = values
                        .par_iter_mut()
                        .map(|value| {
                            let input = *value;
                            let sigmoid = orb_sigmoid(input);
                            *value = input * sigmoid;
                            sigmoid * (1.0 + input * (1.0 - sigmoid))
                        })
                        .collect();
                    derivative
                }
            })
        });
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    ))]
    if values.len() >= ORB_BATCHED_EXP_THRESHOLD {
        return orb_profile_activation!({
            orb_profile_kernel!(profile_label, len, 1, {
                let mut derivative = vec![0.0; values.len()];
                orb_exp_neg_values_in_place(values, &mut derivative);
                for (value, exp_neg) in values.iter_mut().zip(&mut derivative) {
                    let input = -*value;
                    let sigmoid = 1.0 / (1.0 + *exp_neg);
                    *exp_neg = sigmoid * (1.0 + input * (1.0 - sigmoid));
                    *value = input * sigmoid;
                }
                derivative
            })
        });
    }
    orb_profile_activation!({
        orb_profile_kernel!(profile_label, len, 1, {
            let mut derivative = Vec::with_capacity(len);
            for value in values.iter_mut() {
                let input = *value;
                let sigmoid = orb_sigmoid(input);
                *value = input * sigmoid;
                derivative.push(sigmoid * (1.0 + input * (1.0 - sigmoid)));
            }
            derivative
        })
    })
}

fn orb_silu_in_place(values: &mut [f64]) {
    let len = values.len();
    #[cfg(feature = "parallel")]
    if values.len() >= ORB_PARALLEL_VALUE_THRESHOLD && rayon::current_num_threads() > 1 {
        orb_profile_activation!({
            orb_profile_kernel!("silu", len, 1, {
                values.par_iter_mut().for_each(|value| {
                    *value = orb_silu(*value);
                });
            });
        });
        return;
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    ))]
    if values.len() >= ORB_BATCHED_EXP_THRESHOLD {
        orb_profile_activation!({
            orb_profile_kernel!("silu", len, 1, {
                let mut exp_neg = vec![0.0; values.len()];
                orb_exp_neg_values_in_place(values, &mut exp_neg);
                for (value, exp_neg) in values.iter_mut().zip(exp_neg) {
                    let input = -*value;
                    *value = input / (1.0 + exp_neg);
                }
            });
        });
        return;
    }
    orb_profile_activation!({
        orb_profile_kernel!("silu", len, 1, {
            for value in values {
                *value = orb_silu(*value);
            }
        });
    });
}

fn orb_silu_derivative(value: f64) -> f64 {
    let sigmoid = 1.0 / (1.0 + (-value).exp());
    sigmoid * (1.0 + value * (1.0 - sigmoid))
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
#[allow(dead_code)]
fn orb_sigmoid_f32(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    target_os = "linux",
    target_arch = "x86_64",
    target_env = "gnu"
))]
#[allow(dead_code)]
#[allow(unsafe_code)]
unsafe extern "C" {
    fn llmlip_orb_f32_silu_derivative_mvec(values: *mut f32, derivative: *mut f32, len: usize);
    fn llmlip_orb_f32_add_bias_silu_derivative_mvec(
        values: *mut f32,
        derivative: *mut f32,
        bias: *const f32,
        len: usize,
    );
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
#[allow(unsafe_code)]
fn orb_add_bias_silu_in_place_with_derivative_f32_labeled(
    profile_label: &'static str,
    values: &mut [f32],
    rows: usize,
    width: usize,
    bias: &[f32],
) -> Result<Vec<f32>, OrbV3OmolError> {
    expect_len("ORB f32 fused bias+SiLU values", values, rows * width)?;
    expect_len("ORB f32 fused bias+SiLU bias", bias, width)?;
    let mut derivative = vec![0.0_f32; values.len()];
    #[cfg(feature = "parallel")]
    if values.len() >= ORB_PARALLEL_VALUE_THRESHOLD && rayon::current_num_threads() > 1 {
        return Ok(orb_profile_activation!({
            orb_profile_kernel!(profile_label, rows, width, {
                #[cfg(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))]
                {
                    if orb_libmvec_shim_runtime_available() {
                        values
                            .par_chunks_exact_mut(width)
                            .zip(derivative.par_chunks_exact_mut(width))
                            .for_each(|(value_row, derivative_row)| unsafe {
                                llmlip_orb_f32_add_bias_silu_derivative_mvec(
                                    value_row.as_mut_ptr(),
                                    derivative_row.as_mut_ptr(),
                                    bias.as_ptr(),
                                    width,
                                );
                            });
                    } else {
                        values
                            .par_chunks_exact_mut(width)
                            .zip(derivative.par_chunks_exact_mut(width))
                            .for_each(|(value_row, derivative_row)| {
                                for channel in 0..width {
                                    let input = value_row[channel] + bias[channel];
                                    let sigmoid = orb_sigmoid_f32(input);
                                    derivative_row[channel] =
                                        sigmoid * (1.0 + input * (1.0 - sigmoid));
                                    value_row[channel] = input * sigmoid;
                                }
                            });
                    }
                    derivative
                }
                #[cfg(not(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu")))]
                {
                    values
                        .par_chunks_exact_mut(width)
                        .zip(derivative.par_chunks_exact_mut(width))
                        .for_each(|(value_row, derivative_row)| {
                            for channel in 0..width {
                                let input = value_row[channel] + bias[channel];
                                let sigmoid = orb_sigmoid_f32(input);
                                derivative_row[channel] = sigmoid * (1.0 + input * (1.0 - sigmoid));
                                value_row[channel] = input * sigmoid;
                            }
                        });
                    derivative
                }
            })
        }));
    }
    Ok(orb_profile_activation!({
        orb_profile_kernel!(profile_label, rows, width, {
            #[cfg(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))]
            {
                if orb_libmvec_shim_runtime_available() {
                    for (value_row, derivative_row) in values
                        .chunks_exact_mut(width)
                        .zip(derivative.chunks_exact_mut(width))
                    {
                        unsafe {
                            llmlip_orb_f32_add_bias_silu_derivative_mvec(
                                value_row.as_mut_ptr(),
                                derivative_row.as_mut_ptr(),
                                bias.as_ptr(),
                                width,
                            );
                        }
                    }
                } else {
                    for (value_row, derivative_row) in values
                        .chunks_exact_mut(width)
                        .zip(derivative.chunks_exact_mut(width))
                    {
                        for channel in 0..width {
                            let input = value_row[channel] + bias[channel];
                            let sigmoid = orb_sigmoid_f32(input);
                            derivative_row[channel] = sigmoid * (1.0 + input * (1.0 - sigmoid));
                            value_row[channel] = input * sigmoid;
                        }
                    }
                }
                derivative
            }
            #[cfg(not(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu")))]
            {
                for (value_row, derivative_row) in values
                    .chunks_exact_mut(width)
                    .zip(derivative.chunks_exact_mut(width))
                {
                    for channel in 0..width {
                        let input = value_row[channel] + bias[channel];
                        let sigmoid = orb_sigmoid_f32(input);
                        derivative_row[channel] = sigmoid * (1.0 + input * (1.0 - sigmoid));
                        value_row[channel] = input * sigmoid;
                    }
                }
                derivative
            }
        })
    }))
}

#[cfg(all(
    feature = "orb-f32-edge-mlp",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
fn orb_mul_in_place_f32_labeled(
    profile_label: &'static str,
    values: &mut [f32],
    rhs: &[f32],
    rows: usize,
    width: usize,
) -> Result<(), OrbV3OmolError> {
    if values.len() != rhs.len() || values.len() != rows * width {
        return invalid(format!(
            "ORB f32 mul inputs must both have length {}, got {} and {}",
            rows * width,
            values.len(),
            rhs.len()
        ));
    }
    #[cfg(feature = "parallel")]
    if values.len() >= ORB_PARALLEL_VALUE_THRESHOLD && rayon::current_num_threads() > 1 {
        orb_profile_kernel!(profile_label, rows, width, {
            values
                .par_iter_mut()
                .zip(rhs.par_iter())
                .for_each(|(value, rhs)| *value *= *rhs);
        });
        return Ok(());
    }
    orb_profile_kernel!(profile_label, rows, width, {
        for (value, rhs) in values.iter_mut().zip(rhs) {
            *value *= *rhs;
        }
    });
    Ok(())
}

#[cfg(feature = "parallel")]
fn build_node_edge_lists(indices: &[usize], node_count: usize) -> Vec<Vec<usize>> {
    let mut lists: Vec<Vec<usize>> = vec![Vec::new(); node_count];
    for (edge, &node) in indices.iter().enumerate() {
        lists[node].push(edge);
    }
    lists
}

#[cfg(feature = "parallel")]
fn scatter_add_node_centric(
    target: &mut [f64],
    source: &[f64],
    node_edge_lists: &[Vec<usize>],
    width: usize,
) {
    debug_assert_eq!(target.len(), node_edge_lists.len() * width);
    target
        .par_chunks_exact_mut(width)
        .zip(node_edge_lists.par_iter())
        .for_each(|(node_row, edges)| {
            scatter_add_edge_list_to_row(node_row, source, edges, width);
        });
}

#[cfg(feature = "parallel")]
fn scatter_add_edge_list_to_row(
    node_row: &mut [f64],
    source: &[f64],
    edges: &[usize],
    width: usize,
) {
    let mut quads = edges.chunks_exact(4);
    for quad in &mut quads {
        let a = quad[0];
        let b = quad[1];
        let c = quad[2];
        let d = quad[3];
        let a_row = &source[a * width..(a + 1) * width];
        let b_row = &source[b * width..(b + 1) * width];
        let c_row = &source[c * width..(c + 1) * width];
        let d_row = &source[d * width..(d + 1) * width];
        orb_add_four_rows_sequential_in_place(node_row, a_row, b_row, c_row, d_row);
    }
    let mut pairs = quads.remainder().chunks_exact(2);
    for pair in &mut pairs {
        let lhs = pair[0];
        let rhs = pair[1];
        let lhs_row = &source[lhs * width..(lhs + 1) * width];
        let rhs_row = &source[rhs * width..(rhs + 1) * width];
        orb_add_two_rows_sequential_in_place(node_row, lhs_row, rhs_row);
    }
    if let [edge] = pairs.remainder() {
        let src_row = &source[*edge * width..(*edge + 1) * width];
        orb_add_row_in_place(node_row, src_row);
    }
}

#[inline(always)]
#[allow(unsafe_code)]
fn orb_add_row_in_place(dst: &mut [f64], src: &[f64]) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: the helper only performs unaligned loads/stores inside the
        // provided slices and handles the scalar tail explicitly.
        unsafe { orb_add_row_in_place_avx2(dst, src) };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime AVX2 detection; slice bounds are
            // checked by the helper's loop limits.
            unsafe { orb_add_row_in_place_avx2(dst, src) };
            return;
        }
        for (dst, &src) in dst.iter_mut().zip(src) {
            *dst += src;
        }
    }
}

#[inline(always)]
#[allow(unsafe_code)]
fn orb_add_scaled_row_in_place(dst: &mut [f64], src: &[f64], scale: f64) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        // SAFETY: the helper only performs unaligned loads/stores inside the
        // provided slices and handles the scalar tail explicitly.
        unsafe { orb_add_scaled_row_in_place_avx2_fma(dst, src, scale) };
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: guarded by runtime AVX2/FMA detection; slice bounds are
            // checked by the helper's loop limits.
            unsafe { orb_add_scaled_row_in_place_avx2_fma(dst, src, scale) };
            return;
        }
        for (dst, &src) in dst.iter_mut().zip(src) {
            *dst += src * scale;
        }
    }
}

#[inline(always)]
#[allow(unsafe_code)]
fn orb_add_two_rows_in_place(dst: &mut [f64], lhs: &[f64], rhs: &[f64]) {
    debug_assert_eq!(dst.len(), lhs.len());
    debug_assert_eq!(dst.len(), rhs.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: the helper only performs unaligned loads/stores inside the
        // provided slices and handles the scalar tail explicitly.
        unsafe { orb_add_two_rows_in_place_avx2(dst, lhs, rhs) };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime AVX2 detection; slice bounds are
            // checked by the helper's loop limits.
            unsafe { orb_add_two_rows_in_place_avx2(dst, lhs, rhs) };
            return;
        }
        for ((dst, &lhs), &rhs) in dst.iter_mut().zip(lhs).zip(rhs) {
            *dst += lhs + rhs;
        }
    }
}

#[cfg(feature = "parallel")]
#[inline(always)]
#[allow(unsafe_code)]
fn orb_add_two_rows_sequential_in_place(dst: &mut [f64], lhs: &[f64], rhs: &[f64]) {
    debug_assert_eq!(dst.len(), lhs.len());
    debug_assert_eq!(dst.len(), rhs.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: the helper only performs unaligned loads/stores inside the
        // provided slices and handles the scalar tail explicitly.
        unsafe { orb_add_two_rows_sequential_in_place_avx2(dst, lhs, rhs) };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime AVX2 detection; slice bounds are
            // checked by the helper's loop limits.
            unsafe { orb_add_two_rows_sequential_in_place_avx2(dst, lhs, rhs) };
            return;
        }
        for ((dst, &lhs), &rhs) in dst.iter_mut().zip(lhs).zip(rhs) {
            *dst += lhs;
            *dst += rhs;
        }
    }
}

#[cfg(feature = "parallel")]
#[inline(always)]
#[allow(unsafe_code)]
fn orb_add_four_rows_sequential_in_place(
    dst: &mut [f64],
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    debug_assert_eq!(dst.len(), c.len());
    debug_assert_eq!(dst.len(), d.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: the helper only performs unaligned loads/stores inside the
        // provided slices and handles the scalar tail explicitly.
        unsafe { orb_add_four_rows_sequential_in_place_avx2(dst, a, b, c, d) };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime AVX2 detection; slice bounds are
            // checked by the helper's loop limits.
            unsafe { orb_add_four_rows_sequential_in_place_avx2(dst, a, b, c, d) };
            return;
        }
        for ((((dst, &a), &b), &c), &d) in dst.iter_mut().zip(a).zip(b).zip(c).zip(d) {
            *dst += a;
            *dst += b;
            *dst += c;
            *dst += d;
        }
    }
}

#[inline(always)]
#[allow(unsafe_code)]
fn orb_mul_row_in_place(dst: &mut [f64], scale: &[f64]) {
    debug_assert_eq!(dst.len(), scale.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: the helper only performs unaligned loads/stores inside the
        // provided slices and handles the scalar tail explicitly.
        unsafe { orb_mul_row_in_place_avx2(dst, scale) };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by runtime AVX2 detection; slice bounds are
            // checked by the helper's loop limits.
            unsafe { orb_mul_row_in_place_avx2(dst, scale) };
            return;
        }
        for (dst, &scale) in dst.iter_mut().zip(scale) {
            *dst *= scale;
        }
    }
}

#[inline(always)]
#[allow(unsafe_code)]
fn orb_add_two_scaled_rows_and_two_dots(
    dst: &mut [f64],
    lhs: &[f64],
    lhs_scale: f64,
    rhs: &[f64],
    rhs_scale: f64,
    dot_row: &[f64],
) -> (f64, f64) {
    debug_assert_eq!(dst.len(), lhs.len());
    debug_assert_eq!(dst.len(), rhs.len());
    debug_assert_eq!(dst.len(), dot_row.len());
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        // SAFETY: the helper only performs unaligned loads/stores inside the
        // provided slices and handles the scalar tail explicitly.
        unsafe {
            orb_add_two_scaled_rows_and_two_dots_avx2_fma(
                dst, lhs, lhs_scale, rhs, rhs_scale, dot_row,
            )
        }
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: guarded by runtime AVX2/FMA detection; slice bounds are
            // checked by the helper's loop limits.
            return unsafe {
                orb_add_two_scaled_rows_and_two_dots_avx2_fma(
                    dst, lhs, lhs_scale, rhs, rhs_scale, dot_row,
                )
            };
        }
        let mut lhs_dot = 0.0;
        let mut rhs_dot = 0.0;
        for (((dst, &lhs), &rhs), &dot) in dst.iter_mut().zip(lhs).zip(rhs).zip(dot_row) {
            *dst += lhs * lhs_scale + rhs * rhs_scale;
            lhs_dot += lhs * dot;
            rhs_dot += rhs * dot;
        }
        (lhs_dot, rhs_dot)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn orb_add_row_in_place_avx2(dst: &mut [f64], src: &[f64]) {
    use std::arch::x86_64::{__m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};

    let mut offset = 0;
    while offset + 4 <= dst.len() {
        let sum: __m256d = unsafe {
            let dst_v = _mm256_loadu_pd(dst.as_ptr().add(offset));
            let src_v = _mm256_loadu_pd(src.as_ptr().add(offset));
            _mm256_add_pd(dst_v, src_v)
        };
        unsafe {
            _mm256_storeu_pd(dst.as_mut_ptr().add(offset), sum);
        }
        offset += 4;
    }
    for (dst, &src) in dst[offset..].iter_mut().zip(&src[offset..]) {
        *dst += src;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
unsafe fn orb_add_scaled_row_in_place_avx2_fma(dst: &mut [f64], src: &[f64], scale: f64) {
    use std::arch::x86_64::{
        __m256d, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };

    let scale_v = _mm256_set1_pd(scale);
    let mut offset = 0;
    while offset + 4 <= dst.len() {
        let result: __m256d = unsafe {
            let dst_v = _mm256_loadu_pd(dst.as_ptr().add(offset));
            let src_v = _mm256_loadu_pd(src.as_ptr().add(offset));
            _mm256_fmadd_pd(src_v, scale_v, dst_v)
        };
        unsafe {
            _mm256_storeu_pd(dst.as_mut_ptr().add(offset), result);
        }
        offset += 4;
    }
    for (dst, &src) in dst[offset..].iter_mut().zip(&src[offset..]) {
        *dst += src * scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn orb_add_two_rows_in_place_avx2(dst: &mut [f64], lhs: &[f64], rhs: &[f64]) {
    use std::arch::x86_64::{__m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};

    let mut offset = 0;
    while offset + 4 <= dst.len() {
        let result: __m256d = unsafe {
            let dst_v = _mm256_loadu_pd(dst.as_ptr().add(offset));
            let lhs_v = _mm256_loadu_pd(lhs.as_ptr().add(offset));
            let rhs_v = _mm256_loadu_pd(rhs.as_ptr().add(offset));
            _mm256_add_pd(dst_v, _mm256_add_pd(lhs_v, rhs_v))
        };
        unsafe {
            _mm256_storeu_pd(dst.as_mut_ptr().add(offset), result);
        }
        offset += 4;
    }
    for ((dst, &lhs), &rhs) in dst[offset..]
        .iter_mut()
        .zip(&lhs[offset..])
        .zip(&rhs[offset..])
    {
        *dst += lhs + rhs;
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn orb_add_two_rows_sequential_in_place_avx2(dst: &mut [f64], lhs: &[f64], rhs: &[f64]) {
    use std::arch::x86_64::{__m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};

    let mut offset = 0;
    while offset + 4 <= dst.len() {
        let result: __m256d = unsafe {
            let dst_v = _mm256_loadu_pd(dst.as_ptr().add(offset));
            let lhs_v = _mm256_loadu_pd(lhs.as_ptr().add(offset));
            let rhs_v = _mm256_loadu_pd(rhs.as_ptr().add(offset));
            _mm256_add_pd(_mm256_add_pd(dst_v, lhs_v), rhs_v)
        };
        unsafe {
            _mm256_storeu_pd(dst.as_mut_ptr().add(offset), result);
        }
        offset += 4;
    }
    for ((dst, &lhs), &rhs) in dst[offset..]
        .iter_mut()
        .zip(&lhs[offset..])
        .zip(&rhs[offset..])
    {
        *dst += lhs;
        *dst += rhs;
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn orb_add_four_rows_sequential_in_place_avx2(
    dst: &mut [f64],
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
) {
    use std::arch::x86_64::{__m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};

    let mut offset = 0;
    while offset + 4 <= dst.len() {
        let result: __m256d = unsafe {
            let dst_v = _mm256_loadu_pd(dst.as_ptr().add(offset));
            let a_v = _mm256_loadu_pd(a.as_ptr().add(offset));
            let b_v = _mm256_loadu_pd(b.as_ptr().add(offset));
            let c_v = _mm256_loadu_pd(c.as_ptr().add(offset));
            let d_v = _mm256_loadu_pd(d.as_ptr().add(offset));
            _mm256_add_pd(
                _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(dst_v, a_v), b_v), c_v),
                d_v,
            )
        };
        unsafe {
            _mm256_storeu_pd(dst.as_mut_ptr().add(offset), result);
        }
        offset += 4;
    }
    for ((((dst, &a), &b), &c), &d) in dst[offset..]
        .iter_mut()
        .zip(&a[offset..])
        .zip(&b[offset..])
        .zip(&c[offset..])
        .zip(&d[offset..])
    {
        *dst += a;
        *dst += b;
        *dst += c;
        *dst += d;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn orb_mul_row_in_place_avx2(dst: &mut [f64], scale: &[f64]) {
    use std::arch::x86_64::{__m256d, _mm256_loadu_pd, _mm256_mul_pd, _mm256_storeu_pd};

    let mut offset = 0;
    while offset + 4 <= dst.len() {
        let result: __m256d = unsafe {
            let dst_v = _mm256_loadu_pd(dst.as_ptr().add(offset));
            let scale_v = _mm256_loadu_pd(scale.as_ptr().add(offset));
            _mm256_mul_pd(dst_v, scale_v)
        };
        unsafe {
            _mm256_storeu_pd(dst.as_mut_ptr().add(offset), result);
        }
        offset += 4;
    }
    for (dst, &scale) in dst[offset..].iter_mut().zip(&scale[offset..]) {
        *dst *= scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_code)]
unsafe fn orb_add_two_scaled_rows_and_two_dots_avx2_fma(
    dst: &mut [f64],
    lhs: &[f64],
    lhs_scale: f64,
    rhs: &[f64],
    rhs_scale: f64,
    dot_row: &[f64],
) -> (f64, f64) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd,
        _mm256_storeu_pd,
    };

    let lhs_scale_v = _mm256_set1_pd(lhs_scale);
    let rhs_scale_v = _mm256_set1_pd(rhs_scale);
    let mut lhs_dot_v = _mm256_set1_pd(0.0);
    let mut rhs_dot_v = _mm256_set1_pd(0.0);
    let mut offset = 0;
    while offset + 4 <= dst.len() {
        unsafe {
            let dst_v = _mm256_loadu_pd(dst.as_ptr().add(offset));
            let lhs_v = _mm256_loadu_pd(lhs.as_ptr().add(offset));
            let rhs_v = _mm256_loadu_pd(rhs.as_ptr().add(offset));
            let dot_v = _mm256_loadu_pd(dot_row.as_ptr().add(offset));
            let scaled = _mm256_fmadd_pd(rhs_v, rhs_scale_v, _mm256_mul_pd(lhs_v, lhs_scale_v));
            _mm256_storeu_pd(dst.as_mut_ptr().add(offset), _mm256_add_pd(dst_v, scaled));
            lhs_dot_v = _mm256_fmadd_pd(lhs_v, dot_v, lhs_dot_v);
            rhs_dot_v = _mm256_fmadd_pd(rhs_v, dot_v, rhs_dot_v);
        }
        offset += 4;
    }
    let mut lhs_lanes = [0.0; 4];
    let mut rhs_lanes = [0.0; 4];
    unsafe {
        _mm256_storeu_pd(lhs_lanes.as_mut_ptr(), lhs_dot_v);
        _mm256_storeu_pd(rhs_lanes.as_mut_ptr(), rhs_dot_v);
    }
    let mut lhs_dot = lhs_lanes.iter().sum::<f64>();
    let mut rhs_dot = rhs_lanes.iter().sum::<f64>();
    for (((dst, &lhs), &rhs), &dot) in dst[offset..]
        .iter_mut()
        .zip(&lhs[offset..])
        .zip(&rhs[offset..])
        .zip(&dot_row[offset..])
    {
        *dst += lhs * lhs_scale + rhs * rhs_scale;
        lhs_dot += lhs * dot;
        rhs_dot += rhs * dot;
    }
    (lhs_dot, rhs_dot)
}

fn orb_mul_in_place(values: &mut [f64], scale: &[f64]) -> Result<(), OrbV3OmolError> {
    orb_mul_in_place_labeled("mul_in_place", values, scale)
}

fn orb_mul_in_place_labeled(
    profile_label: &'static str,
    values: &mut [f64],
    scale: &[f64],
) -> Result<(), OrbV3OmolError> {
    expect_len("ORB elementwise multiply scale", scale, values.len())?;
    let len = values.len();
    #[cfg(feature = "parallel")]
    if values.len() >= ORB_PARALLEL_VALUE_THRESHOLD && rayon::current_num_threads() > 1 {
        orb_profile_kernel!(profile_label, len, 1, {
            values
                .par_iter_mut()
                .zip(scale.par_iter())
                .for_each(|(value, &scale)| {
                    *value *= scale;
                });
        });
        return Ok(());
    }
    orb_profile_kernel!(profile_label, len, 1, {
        orb_mul_row_in_place(values, scale);
    });
    Ok(())
}

fn orb_sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn orb_sigmoid_values(input: &[f64]) -> Vec<f64> {
    let len = input.len();
    let mut output = vec![0.0; input.len()];
    #[cfg(feature = "parallel")]
    if input.len() >= ORB_PARALLEL_VALUE_THRESHOLD && rayon::current_num_threads() > 1 {
        return orb_profile_activation!({
            orb_profile_kernel!("sigmoid_values", len, 1, {
                output
                    .par_iter_mut()
                    .zip(input.par_iter())
                    .for_each(|(output, &input)| {
                        *output = orb_sigmoid(input);
                    });
                output
            })
        });
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    ))]
    if input.len() >= ORB_BATCHED_EXP_THRESHOLD {
        return orb_profile_activation!({
            orb_profile_kernel!("sigmoid_values", len, 1, {
                orb_exp_neg_values(input, &mut output);
                for output in &mut output {
                    *output = 1.0 / (1.0 + *output);
                }
                output
            })
        });
    }
    orb_profile_activation!({
        orb_profile_kernel!("sigmoid_values", len, 1, {
            for (output, &input) in output.iter_mut().zip(input) {
                *output = orb_sigmoid(input);
            }
            output
        })
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
fn orb_exp_neg_values(input: &[f64], output: &mut [f64]) {
    use std::cell::RefCell;

    debug_assert_eq!(input.len(), output.len());
    thread_local! {
        static NEG_EXP_INPUT: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    }
    NEG_EXP_INPUT.with(|cell| {
        let mut neg = cell.borrow_mut();
        if neg.len() < input.len() {
            neg.resize(input.len(), 0.0);
        }
        for (dst, &value) in neg[..input.len()].iter_mut().zip(input) {
            *dst = -value;
        }
        crate::blas_backend::exp_f64(&neg[..input.len()], output);
    });
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        target_arch = "x86_64"
    )
))]
fn orb_exp_neg_values_in_place(input: &mut [f64], output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());
    for value in input.iter_mut() {
        *value = -*value;
    }
    crate::blas_backend::exp_f64(input, output);
}

fn orb_zbl_cutoff(distance: f64, radius: f64) -> f64 {
    if distance >= radius {
        return 0.0;
    }
    let x = distance / radius;
    1.0 - 28.0 * x.powi(ORB_ZBL_CUTOFF_P) + 48.0 * x.powi(ORB_ZBL_CUTOFF_P + 1)
        - 21.0 * x.powi(ORB_ZBL_CUTOFF_P + 2)
}

fn orb_zbl_cutoff_derivative(distance: f64, radius: f64) -> f64 {
    if distance >= radius {
        return 0.0;
    }
    let x = distance / radius;
    (-28.0 * ORB_ZBL_CUTOFF_P as f64 * x.powi(ORB_ZBL_CUTOFF_P - 1)
        + 48.0 * (ORB_ZBL_CUTOFF_P + 1) as f64 * x.powi(ORB_ZBL_CUTOFF_P)
        - 21.0 * (ORB_ZBL_CUTOFF_P + 2) as f64 * x.powi(ORB_ZBL_CUTOFF_P + 1))
        / radius
}

#[cfg(test)]
#[allow(dead_code)]
fn orb_rms_norm_forward(
    input: &[f64],
    rows: usize,
    width: usize,
    weight: &[f64],
) -> Result<Vec<f64>, OrbV3OmolError> {
    let (output, _) = orb_rms_norm_forward_with_inv_rms(input, rows, width, weight)?;
    Ok(output)
}

fn orb_rms_norm_forward_row_256(input_row: &[f64], output_row: &mut [f64], weight: &[f64]) -> f64 {
    debug_assert_eq!(input_row.len(), LATENT_DIM);
    debug_assert_eq!(output_row.len(), LATENT_DIM);
    debug_assert_eq!(weight.len(), LATENT_DIM);

    let mut sum_square = 0.0;
    for &value in input_row {
        sum_square += value * value;
    }
    let mean_square = sum_square / LATENT_DIM as f64;
    let inv_rms = (mean_square + f64::EPSILON).sqrt().recip();
    for column in 0..LATENT_DIM {
        output_row[column] = input_row[column] * inv_rms * weight[column];
    }
    inv_rms
}

fn orb_rms_norm_forward_with_inv_rms(
    input: &[f64],
    rows: usize,
    width: usize,
    weight: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), OrbV3OmolError> {
    if input.len() != rows * width {
        return invalid(format!(
            "ORB RMSNorm input length must be {}, got {}",
            rows * width,
            input.len()
        ));
    }
    if weight.len() != width {
        return invalid(format!(
            "ORB RMSNorm weight length must be {width}, got {}",
            weight.len()
        ));
    }

    let mut output = vec![0.0; input.len()];
    let mut inv_rms_values = vec![0.0; rows];
    #[cfg(feature = "parallel")]
    if rows >= ORB_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
        return orb_profile_kernel!("rms_norm_forward", rows, width, {
            if width == LATENT_DIM {
                output
                    .par_chunks_exact_mut(LATENT_DIM)
                    .zip(input.par_chunks_exact(LATENT_DIM))
                    .zip(inv_rms_values.par_iter_mut())
                    .for_each(|((output_row, input_row), inv_rms_value)| {
                        *inv_rms_value =
                            orb_rms_norm_forward_row_256(input_row, output_row, weight);
                    });
            } else {
                output
                    .par_chunks_exact_mut(width)
                    .zip(input.par_chunks_exact(width))
                    .zip(inv_rms_values.par_iter_mut())
                    .for_each(|((output_row, input_row), inv_rms_value)| {
                        let mean_square =
                            input_row.iter().map(|value| value * value).sum::<f64>() / width as f64;
                        let inv_rms = (mean_square + f64::EPSILON).sqrt().recip();
                        *inv_rms_value = inv_rms;
                        for column in 0..width {
                            output_row[column] = input_row[column] * inv_rms * weight[column];
                        }
                    });
            }
            Ok((output, inv_rms_values))
        });
    }
    orb_profile_kernel!("rms_norm_forward", rows, width, {
        for row in 0..rows {
            let input_row = &input[row * width..(row + 1) * width];
            let output_row = &mut output[row * width..(row + 1) * width];
            let inv_rms = if width == LATENT_DIM {
                orb_rms_norm_forward_row_256(input_row, output_row, weight)
            } else {
                let mean_square =
                    input_row.iter().map(|value| value * value).sum::<f64>() / width as f64;
                let inv_rms = (mean_square + f64::EPSILON).sqrt().recip();
                for column in 0..width {
                    output_row[column] = input_row[column] * inv_rms * weight[column];
                }
                inv_rms
            };
            inv_rms_values[row] = inv_rms;
        }
        Ok((output, inv_rms_values))
    })
}

#[cfg(test)]
#[allow(dead_code)]
fn orb_rms_norm_input_vjp(
    input: &[f64],
    grad_output: &[f64],
    rows: usize,
    width: usize,
    weight: &[f64],
) -> Result<Vec<f64>, OrbV3OmolError> {
    if input.len() != rows * width {
        return invalid(format!(
            "ORB RMSNorm VJP input length must be {}, got {}",
            rows * width,
            input.len()
        ));
    }
    if grad_output.len() != rows * width {
        return invalid(format!(
            "ORB RMSNorm VJP grad_output length must be {}, got {}",
            rows * width,
            grad_output.len()
        ));
    }
    if weight.len() != width {
        return invalid(format!(
            "ORB RMSNorm VJP weight length must be {width}, got {}",
            weight.len()
        ));
    }
    let mut inv_rms_values = vec![0.0; rows];
    for row in 0..rows {
        let input_row = &input[row * width..(row + 1) * width];
        let mean_square = input_row.iter().map(|value| value * value).sum::<f64>() / width as f64;
        inv_rms_values[row] = (mean_square + f64::EPSILON).sqrt().recip();
    }
    orb_rms_norm_input_vjp_with_inv_rms(input, &inv_rms_values, grad_output, rows, width, weight)
}

fn orb_rms_norm_input_vjp_row_256(
    input_row: &[f64],
    grad_output_row: &[f64],
    inv_rms: f64,
    weight: &[f64],
    grad_input_row: &mut [f64],
) {
    debug_assert_eq!(input_row.len(), LATENT_DIM);
    debug_assert_eq!(grad_output_row.len(), LATENT_DIM);
    debug_assert_eq!(weight.len(), LATENT_DIM);
    debug_assert_eq!(grad_input_row.len(), LATENT_DIM);

    let mut weighted_dot = 0.0;
    for column in 0..LATENT_DIM {
        weighted_dot += grad_output_row[column] * weight[column] * input_row[column];
    }
    let inv_rms_cubed = inv_rms * inv_rms * inv_rms;
    let correction_scale = inv_rms_cubed * weighted_dot / LATENT_DIM as f64;
    for column in 0..LATENT_DIM {
        grad_input_row[column] = grad_output_row[column] * weight[column] * inv_rms
            - input_row[column] * correction_scale;
    }
}

fn orb_rms_norm_input_vjp_with_inv_rms(
    input: &[f64],
    inv_rms_values: &[f64],
    grad_output: &[f64],
    rows: usize,
    width: usize,
    weight: &[f64],
) -> Result<Vec<f64>, OrbV3OmolError> {
    if input.len() != rows * width {
        return invalid(format!(
            "ORB RMSNorm VJP input length must be {}, got {}",
            rows * width,
            input.len()
        ));
    }
    if inv_rms_values.len() != rows {
        return invalid(format!(
            "ORB RMSNorm VJP inv_rms length must be {rows}, got {}",
            inv_rms_values.len()
        ));
    }
    if grad_output.len() != rows * width {
        return invalid(format!(
            "ORB RMSNorm VJP grad_output length must be {}, got {}",
            rows * width,
            grad_output.len()
        ));
    }
    if weight.len() != width {
        return invalid(format!(
            "ORB RMSNorm VJP weight length must be {width}, got {}",
            weight.len()
        ));
    }
    let mut grad_input = vec![0.0; input.len()];
    #[cfg(feature = "parallel")]
    if rows >= ORB_PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1 {
        return orb_profile_kernel!("rms_norm_input_vjp", rows, width, {
            if width == LATENT_DIM {
                grad_input
                    .par_chunks_exact_mut(LATENT_DIM)
                    .zip(input.par_chunks_exact(LATENT_DIM))
                    .zip(grad_output.par_chunks_exact(LATENT_DIM))
                    .zip(inv_rms_values.par_iter())
                    .for_each(
                        |(((grad_input_row, input_row), grad_output_row), &inv_rms)| {
                            orb_rms_norm_input_vjp_row_256(
                                input_row,
                                grad_output_row,
                                inv_rms,
                                weight,
                                grad_input_row,
                            );
                        },
                    );
            } else {
                grad_input
                    .par_chunks_exact_mut(width)
                    .zip(input.par_chunks_exact(width))
                    .zip(grad_output.par_chunks_exact(width))
                    .zip(inv_rms_values.par_iter())
                    .for_each(
                        |(((grad_input_row, input_row), grad_output_row), &inv_rms)| {
                            let weighted_dot = grad_output_row
                                .iter()
                                .zip(weight)
                                .zip(input_row)
                                .map(|((&grad, &scale), &value)| grad * scale * value)
                                .sum::<f64>();
                            let inv_rms_cubed = inv_rms * inv_rms * inv_rms;
                            let correction_scale = inv_rms_cubed * weighted_dot / width as f64;
                            for column in 0..width {
                                grad_input_row[column] =
                                    grad_output_row[column] * weight[column] * inv_rms
                                        - input_row[column] * correction_scale;
                            }
                        },
                    );
            }
            Ok(grad_input)
        });
    }
    orb_profile_kernel!("rms_norm_input_vjp", rows, width, {
        for row in 0..rows {
            let input_row = &input[row * width..(row + 1) * width];
            let grad_output_row = &grad_output[row * width..(row + 1) * width];
            let grad_input_row = &mut grad_input[row * width..(row + 1) * width];
            let inv_rms = inv_rms_values[row];
            if width == LATENT_DIM {
                orb_rms_norm_input_vjp_row_256(
                    input_row,
                    grad_output_row,
                    inv_rms,
                    weight,
                    grad_input_row,
                );
            } else {
                let weighted_dot = grad_output_row
                    .iter()
                    .zip(weight)
                    .zip(input_row)
                    .map(|((&grad, &scale), &value)| grad * scale * value)
                    .sum::<f64>();
                let inv_rms_cubed = inv_rms * inv_rms * inv_rms;
                let correction_scale = inv_rms_cubed * weighted_dot / width as f64;
                for column in 0..width {
                    grad_input_row[column] = grad_output_row[column] * weight[column] * inv_rms
                        - input_row[column] * correction_scale;
                }
            }
        }
        Ok(grad_input)
    })
}

fn validate_spec(spec: &OrbV3OmolSpec) -> Result<(), OrbV3OmolError> {
    expect("schema_version", &spec.schema_version, SCHEMA_VERSION)?;
    expect("architecture", &spec.architecture, ARCHITECTURE)?;
    expect("selected_model", &spec.selected_model, SELECTED_MODEL)?;
    validate_runtime_status_and_inventory(spec)?;
    expect(
        "reference.repository",
        &spec.reference.repository,
        "https://github.com/orbital-materials/orb-models",
    )?;
    expect("reference.tag", &spec.reference.tag, REFERENCE_TAG)?;
    expect(
        "reference.package_version",
        &spec.reference.package_version,
        REFERENCE_VERSION,
    )?;
    expect("reference.commit", &spec.reference.commit, REFERENCE_COMMIT)?;
    expect(
        "checkpoint.filename",
        &spec.checkpoint.filename,
        CHECKPOINT_FILENAME,
    )?;
    expect("checkpoint.url", &spec.checkpoint.url, CHECKPOINT_URL)?;
    expect(
        "checkpoint.sha256",
        &spec.checkpoint.sha256,
        CHECKPOINT_SHA256,
    )?;
    if spec.checkpoint.size_bytes != CHECKPOINT_SIZE_BYTES {
        return invalid(format!(
            "checkpoint.size_bytes must be {CHECKPOINT_SIZE_BYTES}, got {}",
            spec.checkpoint.size_bytes
        ));
    }
    if (spec.graph.cutoff_angstrom - CUTOFF_ANGSTROM).abs() > 1.0e-12 {
        return invalid(format!(
            "graph.cutoff_angstrom must be {CUTOFF_ANGSTROM}, got {}",
            spec.graph.cutoff_angstrom
        ));
    }
    if spec.graph.max_num_neighbors != MAX_NUM_NEIGHBORS {
        return invalid(format!(
            "graph.max_num_neighbors must be {MAX_NUM_NEIGHBORS}, got {}",
            spec.graph.max_num_neighbors
        ));
    }
    if !spec.graph.requires_charge {
        return invalid("graph.requires_charge must be true");
    }
    if !spec.graph.requires_spin {
        return invalid("graph.requires_spin must be true");
    }
    if spec.graph.supports_pbc {
        return invalid("graph.supports_pbc must remain false until PBC fixtures are pinned");
    }
    validate_architecture_parameters(&spec.architecture_parameters)
}

fn validate_runtime_status_and_inventory(spec: &OrbV3OmolSpec) -> Result<(), OrbV3OmolError> {
    match spec.runtime_status.as_str() {
        RUNTIME_STATUS_METADATA_ONLY => {
            if spec.state_inventory.is_some() {
                return invalid("metadata-only ORB metadata must not include state_inventory");
            }
            Ok(())
        }
        RUNTIME_STATUS_STATE_EXPORT => {
            let inventory = spec.state_inventory.as_ref().ok_or_else(|| {
                OrbV3OmolError::InvalidSpec(
                    "state_export_only ORB metadata requires state_inventory".to_owned(),
                )
            })?;
            expect(
                "state_inventory.schema_version",
                &inventory.schema_version,
                STATE_EXPORT_INVENTORY_SCHEMA,
            )?;
            expect_usize(
                "state_inventory.tensor_count",
                inventory.tensor_count,
                STATE_TENSOR_COUNT,
            )?;
            expect_usize(
                "state_inventory.total_numel",
                inventory.total_numel,
                STATE_TOTAL_NUMEL,
            )?;
            if inventory.entries.len() != STATE_TENSOR_COUNT {
                return invalid(format!(
                    "state_inventory.entries length must be {STATE_TENSOR_COUNT}, got {}",
                    inventory.entries.len()
                ));
            }
            for (expected_index, entry) in inventory.entries.iter().enumerate() {
                validate_state_inventory_entry(expected_index, entry)?;
            }
            Ok(())
        }
        other => invalid(format!(
            "runtime_status must be {RUNTIME_STATUS_METADATA_ONLY:?} or {RUNTIME_STATUS_STATE_EXPORT:?}, got {other:?}"
        )),
    }
}

fn validate_state_inventory_entry(
    expected_index: usize,
    entry: &OrbStateTensorSpec,
) -> Result<(), OrbV3OmolError> {
    if entry.index != expected_index {
        return invalid(format!(
            "state_inventory.entries[{expected_index}].index must be {expected_index}, got {}",
            entry.index
        ));
    }
    if entry.name.trim().is_empty() {
        return invalid(format!(
            "state_inventory.entries[{expected_index}].name must not be empty"
        ));
    }
    if !matches!(entry.dtype.as_str(), "f64" | "i64") {
        return invalid(format!(
            "state_inventory entry {} dtype must be \"f64\" or \"i64\", got {:?}",
            entry.name, entry.dtype
        ));
    }
    if entry.shape.contains(&0) {
        return invalid(format!(
            "state_inventory entry {} shape dimensions must be nonzero, got {:?}",
            entry.name, entry.shape
        ));
    }
    let expected_numel = checked_numel(&entry.shape)?;
    if entry.numel != expected_numel {
        return invalid(format!(
            "state_inventory entry {} numel must be {}, got {}",
            entry.name, expected_numel, entry.numel
        ));
    }
    let dtype_size = if entry.dtype == "f64" {
        std::mem::size_of::<f64>()
    } else {
        std::mem::size_of::<i64>()
    };
    let expected_size = entry.numel.checked_mul(dtype_size).ok_or_else(|| {
        OrbV3OmolError::InvalidSpec(format!(
            "state_inventory entry {} size_bytes overflows",
            entry.name
        ))
    })?;
    if entry.size_bytes != expected_size {
        return invalid(format!(
            "state_inventory entry {} size_bytes must be {}, got {}",
            entry.name, expected_size, entry.size_bytes
        ));
    }
    if entry.path.trim().is_empty() || entry.path.starts_with('/') || entry.path.contains("..") {
        return invalid(format!(
            "state_inventory entry {} path must be relative and stay in the model directory",
            entry.name
        ));
    }
    if entry.sha256.len() != 64 || !entry.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return invalid(format!(
            "state_inventory entry {} sha256 must be 64 hex characters",
            entry.name
        ));
    }
    Ok(())
}

fn validate_state_export_tensors(
    spec: &OrbV3OmolSpec,
    external_tensors: &BTreeMap<String, OrbV3OmolExternalTensor>,
) -> Result<(), OrbV3OmolError> {
    if spec.runtime_status != RUNTIME_STATUS_STATE_EXPORT {
        return invalid("external ORB tensors are only valid for state_export_only metadata");
    }
    let inventory = spec.state_inventory.as_ref().ok_or_else(|| {
        OrbV3OmolError::InvalidSpec(
            "state_export_only ORB metadata requires state_inventory".to_owned(),
        )
    })?;
    if external_tensors.len() != inventory.tensor_count {
        return invalid(format!(
            "external tensor count must be {}, got {}",
            inventory.tensor_count,
            external_tensors.len()
        ));
    }
    let mut total_numel = 0usize;
    for entry in &inventory.entries {
        let tensor = external_tensors.get(&entry.name).ok_or_else(|| {
            OrbV3OmolError::InvalidSpec(format!("missing external ORB tensor {:?}", entry.name))
        })?;
        if tensor.dtype() != entry.dtype {
            return invalid(format!(
                "external ORB tensor {} dtype must be {}, got {}",
                entry.name,
                entry.dtype,
                tensor.dtype()
            ));
        }
        if tensor.shape != entry.shape {
            return invalid(format!(
                "external ORB tensor {} shape must be {:?}, got {:?}",
                entry.name, entry.shape, tensor.shape
            ));
        }
        if tensor.path != entry.path {
            return invalid(format!(
                "external ORB tensor {} path must be {:?}, got {:?}",
                entry.name, entry.path, tensor.path
            ));
        }
        if tensor.len() != entry.numel {
            return invalid(format!(
                "external ORB tensor {} numel must be {}, got {}",
                entry.name,
                entry.numel,
                tensor.len()
            ));
        }
        if tensor.byte_sha256 != entry.sha256 {
            return invalid(format!(
                "external ORB tensor {} sha256 must be {}, got {}",
                entry.name, entry.sha256, tensor.byte_sha256
            ));
        }
        total_numel = total_numel.checked_add(tensor.len()).ok_or_else(|| {
            OrbV3OmolError::InvalidSpec("external ORB tensor total numel overflows".to_owned())
        })?;
    }
    if total_numel != inventory.total_numel {
        return invalid(format!(
            "external ORB tensor total numel must be {}, got {}",
            inventory.total_numel, total_numel
        ));
    }
    Ok(())
}

fn validate_architecture_parameters(
    params: &OrbArchitectureParameters,
) -> Result<(), OrbV3OmolError> {
    expect_usize("latent_dim", params.latent_dim, LATENT_DIM)?;
    expect_usize(
        "message_passing_steps",
        params.message_passing_steps,
        MESSAGE_PASSING_STEPS,
    )?;
    expect_usize(
        "base_mlp_hidden_dim",
        params.base_mlp_hidden_dim,
        BASE_MLP_HIDDEN_DIM,
    )?;
    expect_usize("base_mlp_depth", params.base_mlp_depth, BASE_MLP_DEPTH)?;
    expect_usize(
        "head_mlp_hidden_dim",
        params.head_mlp_hidden_dim,
        HEAD_MLP_HIDDEN_DIM,
    )?;
    expect_usize("head_mlp_depth", params.head_mlp_depth, HEAD_MLP_DEPTH)?;
    expect("activation", &params.activation, "silu")?;
    expect("radial_basis", &params.radial_basis, "bessel")?;
    expect_usize(
        "bessel_num_bases",
        params.bessel_num_bases,
        BESSEL_NUM_BASES,
    )?;
    expect_usize("sh_lmax", params.sh_lmax, SH_LMAX)?;
    expect("sh_normalization", &params.sh_normalization, "component")?;
    if !params.sh_normalize {
        return invalid("sh_normalize must be true");
    }
    if !params.outer_product_with_cutoff {
        return invalid("outer_product_with_cutoff must be true");
    }
    expect("mlp_norm", &params.mlp_norm, "rms_norm")?;
    expect("attention_gate", &params.attention_gate, "sigmoid")?;
    if !params.zbl_pair_repulsion {
        return invalid("zbl_pair_repulsion must be true");
    }
    if !params.conservative_forces {
        return invalid("conservative_forces must be true");
    }
    if !params.confidence_head {
        return invalid("confidence_head must be true");
    }
    Ok(())
}

fn orb_atomic_number(symbol: &str, atom: usize) -> Result<usize, OrbV3OmolError> {
    let atomic_number = ORB_ELEMENT_SYMBOLS_BY_Z
        .iter()
        .position(|candidate| *candidate == symbol)
        .ok_or_else(|| OrbV3OmolError::UnsupportedElementSymbol {
            atom,
            symbol: symbol.to_owned(),
        })?;
    if atomic_number == 0 || atomic_number > ORB_MAX_SUPPORTED_ATOMIC_NUMBER {
        return Err(OrbV3OmolError::UnsupportedElementSymbol {
            atom,
            symbol: symbol.to_owned(),
        });
    }
    Ok(atomic_number)
}

const ORB_ELEMENT_SYMBOLS_BY_Z: [&str; 119] = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

fn expect(field: &str, actual: &str, expected: &str) -> Result<(), OrbV3OmolError> {
    if actual == expected {
        Ok(())
    } else {
        invalid(format!("{field} must be {expected:?}, got {actual:?}"))
    }
}

fn expect_usize(field: &str, actual: usize, expected: usize) -> Result<(), OrbV3OmolError> {
    if actual == expected {
        Ok(())
    } else {
        invalid(format!("{field} must be {expected}, got {actual}"))
    }
}

fn checked_numel(shape: &[usize]) -> Result<usize, OrbV3OmolError> {
    shape.iter().try_fold(1_usize, |acc, value| {
        acc.checked_mul(*value).ok_or_else(|| {
            OrbV3OmolError::InvalidSpec(format!("shape product overflows for {shape:?}"))
        })
    })
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn invalid<T>(message: impl Into<String>) -> Result<T, OrbV3OmolError> {
    Err(OrbV3OmolError::InvalidSpec(message.into()))
}
