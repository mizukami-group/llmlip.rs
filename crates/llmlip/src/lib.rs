//! MLIP force runtime assembly.
//!
//! This crate will contain structures, neighbor lists, model loading, and force accumulation.

#[cfg(any(
    all(feature = "accelerate-blas", feature = "intel-mkl"),
    all(feature = "accelerate-blas", feature = "openblas"),
    all(feature = "intel-mkl", feature = "openblas")
))]
compile_error!(
    "select exactly one CBLAS provider feature: `accelerate-blas`, `intel-mkl`, or `openblas`"
);

#[cfg(all(
    feature = "openblas",
    not(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))
))]
compile_error!("`openblas` is currently supported only for Linux x86_64/aarch64 builds");

#[cfg(all(
    feature = "intel-mkl-threaded",
    not(all(target_os = "linux", target_arch = "x86_64"))
))]
compile_error!("`intel-mkl-threaded` is supported only for Linux x86_64 builds");

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod allegro_oam;
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
))]
pub(crate) mod blas_backend;
pub mod dftd3;
pub mod dispersion;
pub mod edge;
#[cfg(feature = "embed-model")]
pub mod embedded;
pub mod esen_block;
pub mod esen_mlp;
pub mod external_graph;
pub mod extxyz;
pub mod force_accum;
pub mod geometry;

// Model-specific equivariant kernels moved out of the public `e3nn-core` crate
// (MACE symmetric contraction, FairChem eSEN blocks/MLP, UMA mixture-of-experts).
pub mod mace;
pub mod mace_lite;
pub mod mace_mh1;
pub mod mace_polar;
pub mod model;
pub mod neighbor;
pub mod nequip_lite;
pub mod nequip_oam_prefix;
pub mod nequix;
pub mod orb_v3_omol;
pub mod pet_lite;
pub mod structure;
pub mod synthetic;
pub mod synthetic_equivariant;
pub mod uma_mole;
mod uma_tensor_pack;

pub use allegro_oam::{
    AllegroOamError, AllegroOamExternalTensor, AllegroOamModel, AllegroOamPreparedGraph,
};
pub use dftd3::{
    Dftd3BjCoordinationModel, Dftd3BjMethod, Dftd3BjParameters, Dftd3BjTwoBodyModel,
    Dftd3C6Reference, Dftd3CoordinationPairCoefficient, Dftd3CoordinationParameters,
    Dftd3CutoffSmoothing, Dftd3ElementCoefficient, Dftd3Error, Dftd3PairCoefficient,
    Dftd3SupportedTerms,
};
pub use dispersion::{DispersionCorrectedError, DispersionCorrectedModel, DispersionCorrection};
pub use edge::{accumulate_edge_force, edge_vector, Edge};
pub use external_graph::{ExternalGraphError, ExternalGraphView};
pub use extxyz::{
    parse_extxyz, parse_extxyz_frames, parse_extxyz_multi, read_extxyz, read_extxyz_frames,
    read_extxyz_multi, write_extxyz_with_forces, ExtXyzError, ExtXyzFrame, ExtXyzFrameMetadata,
};
pub use force_accum::{
    accumulate_edge_displacement_gradient, cell_volume, scatter_edge_vector_vjp,
    stress_from_displacement_gradient, virial_from_displacement_gradient, ForceAccumError, Mat3,
};
pub use geometry::{Cell, Vec3};

pub use mace_lite::{MaceLiteError, MaceLiteModel, MaceRuntimePlanKind};
pub use mace_mh1::{
    MaceMh1Error, MaceMh1ExternalTensor, MaceMh1ExternalTensorValues, MaceMh1Model,
    MaceMh1PreparedInput,
};
pub use mace_polar::{
    MacePolarAuxiliaryPrediction, MacePolarError, MacePolarExternalTensor,
    MacePolarExternalTensorValues, MacePolarModel,
};
pub use model::{
    load_llmlip_model, pack_uma_model_tensors, validate_llmlip_model, EsenOamProfile,
    ExternalGraphModelCapabilities, LlmlipModel, ModelError, PreparedPredictionInput,
    PreparedPredictionWorkspace, UmaBlockReplay, UmaBlockTensorRefs, UmaDataset, UmaEnergyReplay,
    UmaModel, UmaPreparedInput, UmaProfile, UmaRadialMlpTensorRefs, UmaRuntimeModel,
    UmaRuntimeTensorView, UmaTensorPayloadStatus, UmaTensorRef,
};
pub use neighbor::{
    all_pairs_multi_image_neighbor_list, all_pairs_neighbor_list, EdgeMode, NeighborList,
    NeighborListError,
};
pub use nequip_lite::{NequipLiteError, NequipLiteModel};
pub use nequip_oam_prefix::{NequipOamPrefixError, NequipOamPrefixModel};
pub use nequix::{NequixError, NequixModel};
pub use orb_v3_omol::{
    OrbV3OmolEncodedFeatures, OrbV3OmolEnergyBreakdown, OrbV3OmolEnergyTailVjp, OrbV3OmolError,
    OrbV3OmolGnnInputCotangents, OrbV3OmolGnnLayerOutput, OrbV3OmolGnnOutput,
    OrbV3OmolInputFeatureCotangents, OrbV3OmolInputFeatures, OrbV3OmolModel,
    OrbV3OmolPreparedGraph,
};
pub use pet_lite::{
    PetLiteError, PetLiteExternalTensor, PetLiteModel, PetLitePreparedBatchF32Workspace,
    PetLitePreparedBatchWorkspace, PetLiteProfile, PetLiteStageTiming,
};
pub use structure::{
    MaceMh1WignerMode, ModelCapabilities, Prediction, PredictionComputeDtype,
    PredictionInputRequirements, PredictionOptions, Structure,
};
pub use synthetic::{SyntheticModelError, SyntheticRadialModel};
pub use synthetic_equivariant::SyntheticEquivariantModel;
pub use uma_tensor_pack::{uma_tensor_pack_exists, UMA_TENSOR_PACK_FILENAME};
