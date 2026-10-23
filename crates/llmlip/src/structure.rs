use crate::{
    force_accum::Mat3,
    geometry::{Cell, Vec3},
};

#[derive(Clone, Debug)]
pub struct Structure {
    pub species: Vec<String>,
    pub positions: Vec<Vec3>,
    pub cell: Option<Cell>,
    pub pbc: [bool; 3],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MaceMh1WignerMode {
    /// Use the f64 Clebsch-Gordan constants compiled into the Rust runtime.
    #[default]
    TrueFloat64,
    /// Match the official upstream MACE-MH-1 TorchScript artifact, where
    /// generic Wigner constants were captured as f32 buffers and later promoted
    /// by `model.double()`. This mode is intended for reference comparisons.
    UpstreamArtifact,
}

#[derive(Clone, Debug, Default)]
pub struct PredictionOptions {
    pub compute_energy: bool,
    pub compute_forces: bool,
    pub compute_stress: bool,
    pub compute_virial: bool,
    /// Optional molecular total charge for models that require global charge
    /// conditioning.
    pub total_charge: Option<f64>,
    /// Optional spin multiplicity for models that require global spin
    /// conditioning.
    pub spin_multiplicity: Option<f64>,
    /// Optional model head name for multi-head architectures.
    ///
    /// Headless models reject this option explicitly instead of silently
    /// ignoring it.
    pub selected_head: Option<String>,
    /// MACE-MH-1 Wigner/CG constant mode.
    ///
    /// The default keeps the Rust runtime on true f64 constants. Use
    /// [`MaceMh1WignerMode::UpstreamArtifact`] only when reproducing the
    /// official upstream TorchScript artifact bit-for-bit.
    pub mace_mh1_wigner_mode: MaceMh1WignerMode,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PredictionInputRequirements {
    pub total_charge: bool,
    pub spin_multiplicity: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelCapabilities {
    pub architecture: &'static str,
    pub supports_energy: bool,
    pub supports_forces: bool,
    pub supports_stress: bool,
    pub supports_virial: bool,
    pub supports_float64: bool,
    pub supports_float32: bool,
    pub supports_prepared_input: bool,
    pub stress_requires_full_pbc: bool,
    pub virial_requires_full_pbc: bool,
    /// Model-supported species symbols. An empty list means the architecture
    /// does not expose a finite species set through this capability surface.
    pub supported_species: Vec<String>,
    pub supported_heads: Vec<String>,
    pub input_requirements: PredictionInputRequirements,
}

impl PredictionInputRequirements {
    #[must_use]
    pub const fn none() -> Self {
        Self {
            total_charge: false,
            spin_multiplicity: false,
        }
    }

    #[must_use]
    pub const fn charge_spin() -> Self {
        Self {
            total_charge: true,
            spin_multiplicity: true,
        }
    }

    #[must_use]
    pub const fn requires_any(self) -> bool {
        self.total_charge || self.spin_multiplicity
    }
}

impl PredictionOptions {
    /// Return the selected model head, if one was requested.
    #[must_use]
    pub fn selected_head(&self) -> Option<&str> {
        self.selected_head.as_deref()
    }

    /// Return a copy of these prediction options with a selected model head.
    #[must_use]
    pub fn with_selected_head(mut self, selected_head: impl Into<String>) -> Self {
        self.selected_head = Some(selected_head.into());
        self
    }

    /// Return a copy of these prediction options with a MACE-MH-1 Wigner/CG
    /// constant mode.
    #[must_use]
    pub fn with_mace_mh1_wigner_mode(mut self, mode: MaceMh1WignerMode) -> Self {
        self.mace_mh1_wigner_mode = mode;
        self
    }

    /// Return a copy of these prediction options with molecular charge/spin
    /// metadata.
    #[must_use]
    pub fn with_charge_spin(mut self, total_charge: f64, spin_multiplicity: f64) -> Self {
        self.total_charge = Some(total_charge);
        self.spin_multiplicity = Some(spin_multiplicity);
        self
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PredictionComputeDtype {
    #[default]
    Float64,
    /// Use native f32 kernels where an optimized model path provides them.
    ///
    /// Scalar prediction outputs remain f64. Model runtimes should promote
    /// final scalar energy reductions to f64 by default, even when inner
    /// feature kernels execute in f32.
    Float32,
}

#[derive(Clone, Debug, Default)]
pub struct Prediction {
    pub energy: Option<f64>,
    pub forces: Option<Vec<Vec3>>,
    pub stress: Option<Mat3>,
    pub virial: Option<Mat3>,
}

impl Structure {
    #[must_use]
    pub fn new(species: Vec<String>, positions: Vec<Vec3>) -> Self {
        Self {
            species,
            positions,
            cell: None,
            pbc: [false; 3],
        }
    }

    #[must_use]
    pub fn with_cell(mut self, cell: Cell, pbc: [bool; 3]) -> Self {
        self.cell = Some(cell);
        self.pbc = pbc;
        self
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}
