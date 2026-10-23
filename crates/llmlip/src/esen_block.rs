//! Composed FairChem eSEN block primitives.

use e3nn_core::activation::{
    fairchem_esen_gate_activation_forward, fairchem_esen_gate_activation_forward_with_cache_f32,
    fairchem_esen_gate_activation_vjp, fairchem_esen_gate_activation_vjp_with_cache_f32,
    EsenGateActivationConfig,
};
use e3nn_core::mlp::{silu, silu_derivative};
use e3nn_core::normalization::{
    fairchem_esen_rms_norm_sh_forward, fairchem_esen_rms_norm_sh_vjp, EsenRmsNormShConfig,
};
use e3nn_core::radial::{
    fairchem_esen_edge_envelope_modulation_forward, fairchem_esen_edge_envelope_modulation_vjp,
};
use e3nn_core::scatter::{
    fairchem_esen_edge_to_node_scaled_scatter_forward,
    fairchem_esen_edge_to_node_scaled_scatter_vjp_edges,
    fairchem_esen_source_target_concat_forward, fairchem_esen_source_target_concat_vjp_nodes,
};
use e3nn_core::so2::{
    fairchem_esen_so2_convolution_forward, fairchem_esen_so2_convolution_vjp,
    EsenSo2ConvolutionConfig,
};
use e3nn_core::so3::{
    fairchem_esen_so3_batched_matmul_forward, fairchem_esen_so3_batched_matmul_vjp_input,
    fairchem_esen_so3_batched_matmul_vjp_matrix, fairchem_esen_so3_from_grid_forward,
    fairchem_esen_so3_from_grid_vjp, fairchem_esen_so3_linear_forward,
    fairchem_esen_so3_linear_forward_f32, fairchem_esen_so3_linear_vjp,
    fairchem_esen_so3_linear_vjp_f32, fairchem_esen_so3_linear_vjp_write_f32,
    fairchem_esen_so3_to_grid_forward, fairchem_esen_so3_to_grid_vjp, EsenSo3BatchedMatmulConfig,
    EsenSo3GridConfig, EsenSo3LinearConfig,
};
use e3nn_core::{CoreError, Result};
use std::sync::OnceLock;
use std::time::Instant;

fn esen_block_trace_timings_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("LLMLIP_UMA_TRACE_TIMINGS").ok().as_deref(),
            Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
        )
    })
}

#[inline(always)]
fn esen_block_trace_start() -> Option<Instant> {
    esen_block_trace_timings_enabled().then(Instant::now)
}

#[inline(always)]
fn trace_esen_block_stage(stage: &'static str, start: Option<Instant>) {
    if let Some(start) = start {
        eprintln!(
            "uma_trace stage={stage} elapsed_ms={:.6}",
            start.elapsed().as_secs_f64() * 1e3
        );
    }
}

#[cfg(all(
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
))]
fn esen_atomwise_scalar_gate_sve_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("LLMLIP_DISABLE_SIMD").is_none()
            && std::env::var_os("LLMLIP_DISABLE_SVE").is_none()
            && std::env::var_os("LLMLIP_ESEN_ATOMWISE_DISABLE_SCALAR_GATE_SVE").is_none()
            && std::arch::is_aarch64_feature_detected!("sve")
    })
}

#[cfg(all(
    feature = "armpl",
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
))]
fn esen_atomwise_so3_linear_sve_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("LLMLIP_DISABLE_SIMD").is_none()
            && std::env::var_os("LLMLIP_DISABLE_SVE").is_none()
            && std::env::var_os("LLMLIP_ESEN_ATOMWISE_DISABLE_SO3_LINEAR_SVE").is_none()
            && std::arch::is_aarch64_feature_detected!("sve")
    })
}

#[cfg(all(
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
))]
#[allow(unsafe_code)]
unsafe extern "C" {
    fn llmlip_uma_spectral_scalar_gate_mlp_forward_sve_f32(
        nodes: usize,
        input_node_stride: usize,
        sphere_channels: usize,
        gate_node_stride: usize,
        input: *const f32,
        weight: *const f32,
        bias: *const f32,
        gate_pre: *mut f32,
        gates: *mut f32,
    );

    fn llmlip_uma_spectral_scalar_gate_mlp_vjp_sve_f32(
        nodes: usize,
        input_node_stride: usize,
        sphere_channels: usize,
        gate_node_stride: usize,
        gate_pre: *const f32,
        weight: *const f32,
        grad_gates: *const f32,
        grad_input: *mut f32,
    );
}

#[cfg(all(
    feature = "armpl",
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
))]
#[allow(unsafe_code)]
unsafe extern "C" {
    fn llmlip_uma_spectral_so3_linear_forward_l2_sve_f32(
        nodes: usize,
        channels: usize,
        input: *const f32,
        weight_in_by_out: *const f32,
        bias: *const f32,
        out: *mut f32,
    );

    fn llmlip_uma_spectral_so3_linear_vjp_l2_sve_f32(
        nodes: usize,
        channels: usize,
        weight_out_by_in: *const f32,
        grad_out: *const f32,
        grad_input: *mut f32,
    );
}

#[cfg(all(
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
))]
#[allow(unsafe_code)]
fn spectral_scalar_gate_mlp_forward_sve_f32(
    config: EsenSpectralAtomwiseConfig,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    gate_pre: &mut [f32],
    gates: &mut [f32],
) -> bool {
    let input_node_stride = config.input_node_stride();
    if !esen_atomwise_scalar_gate_sve_enabled()
        || input_node_stride == 0
        || !input.len().is_multiple_of(input_node_stride)
        || weight.len() != config.scalar_gate_weight_len()
        || bias.len() != config.scalar_gate_bias_len()
        || gate_pre.len() != input.len() / input_node_stride * config.gate_node_stride()
        || gates.len() != gate_pre.len()
    {
        return false;
    }
    let nodes = input.len() / input_node_stride;
    unsafe {
        llmlip_uma_spectral_scalar_gate_mlp_forward_sve_f32(
            nodes,
            input_node_stride,
            config.sphere_channels,
            config.gate_node_stride(),
            input.as_ptr(),
            weight.as_ptr(),
            bias.as_ptr(),
            gate_pre.as_mut_ptr(),
            gates.as_mut_ptr(),
        );
    }
    true
}

#[cfg(all(
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
))]
#[allow(unsafe_code)]
fn spectral_scalar_gate_mlp_vjp_sve_f32(
    config: EsenSpectralAtomwiseConfig,
    input: &[f32],
    gate_pre: &[f32],
    weight: &[f32],
    grad_gates: &[f32],
    grad_input: &mut [f32],
) -> bool {
    let input_node_stride = config.input_node_stride();
    if !esen_atomwise_scalar_gate_sve_enabled()
        || input_node_stride == 0
        || !input.len().is_multiple_of(input_node_stride)
        || gate_pre.len() != input.len() / input_node_stride * config.gate_node_stride()
        || weight.len() != config.scalar_gate_weight_len()
        || grad_gates.len() != gate_pre.len()
        || grad_input.len() != input.len()
    {
        return false;
    }
    let nodes = input.len() / input_node_stride;
    unsafe {
        llmlip_uma_spectral_scalar_gate_mlp_vjp_sve_f32(
            nodes,
            input_node_stride,
            config.sphere_channels,
            config.gate_node_stride(),
            gate_pre.as_ptr(),
            weight.as_ptr(),
            grad_gates.as_ptr(),
            grad_input.as_mut_ptr(),
        );
    }
    true
}

#[cfg(all(
    feature = "armpl",
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
))]
#[allow(unsafe_code)]
fn spectral_so3_linear_forward_l2_sve_f32(
    config: EsenSpectralAtomwiseConfig,
    input: &[f32],
    weight_in_by_out: &[f32],
    bias: &[f32],
    out: &mut [f32],
) -> bool {
    let channels = config.sphere_channels;
    let input_node_stride = config.input_node_stride();
    if !esen_atomwise_so3_linear_sve_enabled()
        || config.l_max != 2
        || config.sphere_channels != config.hidden_channels
        || !input.len().is_multiple_of(input_node_stride)
        || weight_in_by_out.len() != 3 * channels * channels
        || bias.len() != channels
        || out.len() != input.len()
    {
        return false;
    }
    let nodes = input.len() / input_node_stride;
    unsafe {
        llmlip_uma_spectral_so3_linear_forward_l2_sve_f32(
            nodes,
            channels,
            input.as_ptr(),
            weight_in_by_out.as_ptr(),
            bias.as_ptr(),
            out.as_mut_ptr(),
        );
    }
    true
}

#[cfg(all(
    feature = "armpl",
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
))]
#[allow(unsafe_code)]
fn spectral_so3_linear_vjp_l2_sve_f32(
    config: EsenSpectralAtomwiseConfig,
    weight_out_by_in: &[f32],
    grad_out: &[f32],
    grad_input: &mut [f32],
) -> bool {
    let channels = config.sphere_channels;
    let input_node_stride = config.input_node_stride();
    if !esen_atomwise_so3_linear_sve_enabled()
        || config.l_max != 2
        || config.sphere_channels != config.hidden_channels
        || !grad_input.len().is_multiple_of(input_node_stride)
        || grad_out.len() != grad_input.len()
        || weight_out_by_in.len() != 3 * channels * channels
    {
        return false;
    }
    let nodes = grad_input.len() / input_node_stride;
    unsafe {
        llmlip_uma_spectral_so3_linear_vjp_l2_sve_f32(
            nodes,
            channels,
            weight_out_by_in.as_ptr(),
            grad_out.as_ptr(),
            grad_input.as_mut_ptr(),
        );
    }
    true
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EsenEdgewiseGateConfig {
    num_nodes: usize,
    l_max: usize,
    m_max: usize,
    sphere_channels: usize,
    hidden_channels: usize,
    node_offset: usize,
    use_envelope: bool,
    cutoff: f64,
    envelope_exponent: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EsenGridAtomwiseConfig {
    grid: EsenSo3GridConfig,
    hidden_channels: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EsenSpectralAtomwiseConfig {
    l_max: usize,
    sphere_channels: usize,
    hidden_channels: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EsenGridBlockConfig {
    norm1: EsenRmsNormShConfig,
    norm2: EsenRmsNormShConfig,
    edge: EsenEdgewiseGateConfig,
    atomwise: EsenGridAtomwiseConfig,
}

impl EsenEdgewiseGateConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_nodes: usize,
        l_max: usize,
        m_max: usize,
        sphere_channels: usize,
        hidden_channels: usize,
        node_offset: usize,
        use_envelope: bool,
        cutoff: f64,
        envelope_exponent: usize,
    ) -> Result<Self> {
        if num_nodes == 0 {
            return Err(invalid_esen_block("num_nodes must be nonzero"));
        }
        if m_max > l_max {
            return Err(invalid_esen_block(format!(
                "m_max must be <= l_max, got {m_max} > {l_max}"
            )));
        }
        if sphere_channels == 0 || hidden_channels == 0 {
            return Err(invalid_esen_block(format!(
                "channel dimensions must be nonzero, got sphere={sphere_channels}, hidden={hidden_channels}"
            )));
        }
        if use_envelope {
            if !cutoff.is_finite() || cutoff <= 0.0 {
                return Err(invalid_esen_block(format!(
                    "cutoff must be positive and finite when envelope is enabled, got {cutoff}"
                )));
            }
            if envelope_exponent == 0 {
                return Err(invalid_esen_block(
                    "envelope_exponent must be nonzero when envelope is enabled",
                ));
            }
        }
        reduced_coefficient_count(l_max, m_max)
            .and_then(|count| count.checked_mul(sphere_channels))
            .and_then(|_| reduced_coefficient_count(l_max, m_max)?.checked_mul(hidden_channels))
            .ok_or_else(|| invalid_esen_block("Edgewise dimensions overflow"))?;
        Ok(Self {
            num_nodes,
            l_max,
            m_max,
            sphere_channels,
            hidden_channels,
            node_offset,
            use_envelope,
            cutoff,
            envelope_exponent,
        })
    }

    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    #[must_use]
    pub fn l_max(&self) -> usize {
        self.l_max
    }

    #[must_use]
    pub fn m_max(&self) -> usize {
        self.m_max
    }

    #[must_use]
    pub fn sphere_channels(&self) -> usize {
        self.sphere_channels
    }

    #[must_use]
    pub fn hidden_channels(&self) -> usize {
        self.hidden_channels
    }

    #[must_use]
    pub fn node_offset(&self) -> usize {
        self.node_offset
    }

    #[must_use]
    pub fn use_envelope(&self) -> bool {
        self.use_envelope
    }

    #[must_use]
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    #[must_use]
    pub fn envelope_exponent(&self) -> usize {
        self.envelope_exponent
    }

    #[must_use]
    pub fn coefficient_count(&self) -> usize {
        (0..=self.l_max)
            .map(|degree| 2 * self.m_max.min(degree) + 1)
            .sum()
    }

    #[must_use]
    pub fn node_feature_dim(&self) -> usize {
        self.coefficient_count() * self.sphere_channels
    }

    #[must_use]
    pub fn edge_concat_feature_dim(&self) -> usize {
        2 * self.node_feature_dim()
    }

    #[must_use]
    pub fn hidden_feature_dim(&self) -> usize {
        self.coefficient_count() * self.hidden_channels
    }

    #[must_use]
    pub fn gate_feature_dim(&self) -> usize {
        self.l_max * self.hidden_channels
    }

    pub fn conv1_config(&self) -> Result<EsenSo2ConvolutionConfig> {
        EsenSo2ConvolutionConfig::new(
            self.l_max,
            self.m_max,
            2 * self.sphere_channels,
            self.hidden_channels,
            self.gate_feature_dim(),
        )
        .map_err(|err| invalid_esen_block(err.to_string()))
    }

    pub fn conv2_config(&self) -> Result<EsenSo2ConvolutionConfig> {
        EsenSo2ConvolutionConfig::new(
            self.l_max,
            self.m_max,
            self.hidden_channels,
            self.sphere_channels,
            0,
        )
        .map_err(|err| invalid_esen_block(err.to_string()))
    }

    pub fn gate_config(&self) -> Result<EsenGateActivationConfig> {
        EsenGateActivationConfig::new(self.l_max, self.m_max, self.hidden_channels)
            .map_err(|err| invalid_esen_block(err.to_string()))
    }
}

impl EsenGridAtomwiseConfig {
    pub fn new(
        coefficient_count: usize,
        sphere_channels: usize,
        hidden_channels: usize,
        lat_resolution: usize,
        long_resolution: usize,
    ) -> Result<Self> {
        if hidden_channels == 0 {
            return Err(invalid_esen_block("hidden_channels must be nonzero"));
        }
        let grid = EsenSo3GridConfig::new(
            coefficient_count,
            sphere_channels,
            lat_resolution,
            long_resolution,
        )
        .map_err(|err| invalid_esen_block(err.to_string()))?;
        hidden_channels
            .checked_mul(sphere_channels)
            .and_then(|_| hidden_channels.checked_mul(hidden_channels))
            .and_then(|_| sphere_channels.checked_mul(hidden_channels))
            .ok_or_else(|| invalid_esen_block("GridAtomwise dimensions overflow"))?;
        Ok(Self {
            grid,
            hidden_channels,
        })
    }

    #[must_use]
    pub fn grid_config(&self) -> EsenSo3GridConfig {
        self.grid
    }

    #[must_use]
    pub fn coefficient_count(&self) -> usize {
        self.grid.coefficient_count()
    }

    #[must_use]
    pub fn sphere_channels(&self) -> usize {
        self.grid.channels()
    }

    #[must_use]
    pub fn hidden_channels(&self) -> usize {
        self.hidden_channels
    }

    #[must_use]
    pub fn grid_points(&self) -> usize {
        self.grid.grid_points()
    }

    #[must_use]
    pub fn coefficient_node_stride(&self) -> usize {
        self.grid.coefficient_node_stride()
    }

    #[must_use]
    pub fn grid_node_stride(&self) -> usize {
        self.grid.grid_node_stride()
    }

    #[must_use]
    pub fn weight1_len(&self) -> usize {
        self.hidden_channels * self.sphere_channels()
    }

    #[must_use]
    pub fn weight2_len(&self) -> usize {
        self.hidden_channels * self.hidden_channels
    }

    #[must_use]
    pub fn weight3_len(&self) -> usize {
        self.sphere_channels() * self.hidden_channels
    }
}

impl EsenSpectralAtomwiseConfig {
    pub fn new(l_max: usize, sphere_channels: usize, hidden_channels: usize) -> Result<Self> {
        if l_max == 0 {
            return Err(invalid_esen_block(
                "l_max must be positive for SpectralAtomwise gate path",
            ));
        }
        if sphere_channels == 0 || hidden_channels == 0 {
            return Err(invalid_esen_block(format!(
                "channel dimensions must be nonzero, got sphere={sphere_channels}, hidden={hidden_channels}"
            )));
        }
        (l_max + 1)
            .checked_mul(l_max + 1)
            .and_then(|count| count.checked_mul(sphere_channels))
            .and_then(|_| (l_max + 1).checked_mul(hidden_channels))
            .ok_or_else(|| invalid_esen_block("SpectralAtomwise dimensions overflow"))?;
        Ok(Self {
            l_max,
            sphere_channels,
            hidden_channels,
        })
    }

    #[must_use]
    pub fn l_max(&self) -> usize {
        self.l_max
    }

    #[must_use]
    pub fn sphere_channels(&self) -> usize {
        self.sphere_channels
    }

    #[must_use]
    pub fn hidden_channels(&self) -> usize {
        self.hidden_channels
    }

    #[must_use]
    pub fn coefficient_count(&self) -> usize {
        (self.l_max + 1) * (self.l_max + 1)
    }

    #[must_use]
    pub fn input_node_stride(&self) -> usize {
        self.coefficient_count() * self.sphere_channels
    }

    #[must_use]
    pub fn hidden_node_stride(&self) -> usize {
        self.coefficient_count() * self.hidden_channels
    }

    #[must_use]
    pub fn gate_node_stride(&self) -> usize {
        self.l_max * self.hidden_channels
    }

    #[must_use]
    pub fn scalar_gate_weight_len(&self) -> usize {
        self.gate_node_stride() * self.sphere_channels
    }

    #[must_use]
    pub fn scalar_gate_bias_len(&self) -> usize {
        self.gate_node_stride()
    }

    pub fn so3_linear1_config(&self) -> Result<EsenSo3LinearConfig> {
        EsenSo3LinearConfig::new(self.l_max, self.sphere_channels, self.hidden_channels)
            .map_err(|err| invalid_esen_block(err.to_string()))
    }

    pub fn so3_linear2_config(&self) -> Result<EsenSo3LinearConfig> {
        EsenSo3LinearConfig::new(self.l_max, self.hidden_channels, self.sphere_channels)
            .map_err(|err| invalid_esen_block(err.to_string()))
    }

    pub fn gate_config(&self) -> Result<EsenGateActivationConfig> {
        EsenGateActivationConfig::new(self.l_max, self.l_max, self.hidden_channels)
            .map_err(|err| invalid_esen_block(err.to_string()))
    }
}

impl EsenGridBlockConfig {
    pub fn new(
        norm1: EsenRmsNormShConfig,
        norm2: EsenRmsNormShConfig,
        edge: EsenEdgewiseGateConfig,
        atomwise: EsenGridAtomwiseConfig,
    ) -> Result<Self> {
        let node_stride = edge.node_feature_dim();
        if norm1.node_stride() != node_stride {
            return Err(invalid_esen_block(format!(
                "norm1 node stride {} must match Edgewise node feature dim {node_stride}",
                norm1.node_stride()
            )));
        }
        if norm2.node_stride() != node_stride {
            return Err(invalid_esen_block(format!(
                "norm2 node stride {} must match Edgewise node feature dim {node_stride}",
                norm2.node_stride()
            )));
        }
        if atomwise.coefficient_node_stride() != node_stride {
            return Err(invalid_esen_block(format!(
                "GridAtomwise node stride {} must match Edgewise node feature dim {node_stride}",
                atomwise.coefficient_node_stride()
            )));
        }
        if norm1.l_max() != norm2.l_max() || norm1.channels() != norm2.channels() {
            return Err(invalid_esen_block(
                "norm1 and norm2 must use the same l_max and channels",
            ));
        }
        if edge.num_nodes() == 0 {
            return Err(invalid_esen_block("edge.num_nodes must be nonzero"));
        }
        Ok(Self {
            norm1,
            norm2,
            edge,
            atomwise,
        })
    }

    #[must_use]
    pub fn norm1_config(&self) -> EsenRmsNormShConfig {
        self.norm1
    }

    #[must_use]
    pub fn norm2_config(&self) -> EsenRmsNormShConfig {
        self.norm2
    }

    #[must_use]
    pub fn edge_config(&self) -> EsenEdgewiseGateConfig {
        self.edge
    }

    #[must_use]
    pub fn atomwise_config(&self) -> EsenGridAtomwiseConfig {
        self.atomwise
    }

    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.edge.num_nodes()
    }

    #[must_use]
    pub fn node_stride(&self) -> usize {
        self.edge.node_feature_dim()
    }
}

/// FairChem eSEN `Edgewise` gate-activation path.
///
/// This composes the checkpoint-independent primitives in the same dataflow as
/// the FairChem module: source/target concat, Wigner rotation, two SO2
/// convolutions, gate activation, optional polynomial envelope, inverse
/// rotation, and target-node accumulation. Weight gradients are intentionally
/// out of scope for this fixed-weight inference crate.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_edgewise_gate_forward(
    config: EsenEdgewiseGateConfig,
    edge_sources: &[usize],
    edge_targets: &[usize],
    node_features: &[f64],
    wigner: &[f64],
    conv1_radial: Option<&[f64]>,
    conv1_m0_weight: &[f64],
    conv1_m0_bias: &[f64],
    conv1_m_weights: &[&[f64]],
    conv2_radial: Option<&[f64]>,
    conv2_m0_weight: &[f64],
    conv2_m0_bias: &[f64],
    conv2_m_weights: &[&[f64]],
    edge_distances: Option<&[f64]>,
    wigner_inv: &[f64],
    out_node_features: &mut [f64],
) -> Result<()> {
    let edges = check_edgewise_common_shapes(
        "fairchem_esen_edgewise_gate_forward",
        config,
        edge_sources,
        edge_targets,
        node_features,
        wigner,
        edge_distances,
        wigner_inv,
        out_node_features,
    )?;
    let intermediates = run_edgewise_gate_intermediates(
        config,
        edge_sources,
        edge_targets,
        node_features,
        wigner,
        conv1_radial,
        conv1_m0_weight,
        conv1_m0_bias,
        conv1_m_weights,
        conv2_radial,
        conv2_m0_weight,
        conv2_m0_bias,
        conv2_m_weights,
        edge_distances,
        wigner_inv,
    )?;

    out_node_features.fill(0.0);
    fairchem_esen_edge_to_node_scaled_scatter_forward(
        config.num_nodes,
        config.node_feature_dim(),
        edge_targets,
        config.node_offset,
        1.0,
        &intermediates.rotated_back,
        out_node_features,
    )?;
    debug_assert_eq!(
        intermediates.rotated_back.len(),
        edges * config.node_feature_dim()
    );
    Ok(())
}

/// Accumulate input VJPs of [`fairchem_esen_edgewise_gate_forward`].
///
/// The gradients are accumulated into the provided buffers. Model parameter
/// gradients are deliberately not computed.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_edgewise_gate_vjp(
    config: EsenEdgewiseGateConfig,
    edge_sources: &[usize],
    edge_targets: &[usize],
    node_features: &[f64],
    wigner: &[f64],
    conv1_radial: Option<&[f64]>,
    conv1_m0_weight: &[f64],
    conv1_m0_bias: &[f64],
    conv1_m_weights: &[&[f64]],
    conv2_radial: Option<&[f64]>,
    conv2_m0_weight: &[f64],
    conv2_m0_bias: &[f64],
    conv2_m_weights: &[&[f64]],
    edge_distances: Option<&[f64]>,
    wigner_inv: &[f64],
    grad_out_node_features: &[f64],
    grad_node_features: &mut [f64],
    grad_wigner: &mut [f64],
    grad_conv1_radial: Option<&mut [f64]>,
    grad_conv2_radial: Option<&mut [f64]>,
    mut grad_edge_distances: Option<&mut [f64]>,
    grad_wigner_inv: &mut [f64],
) -> Result<()> {
    let edges = check_edgewise_common_shapes(
        "fairchem_esen_edgewise_gate_vjp",
        config,
        edge_sources,
        edge_targets,
        node_features,
        wigner,
        edge_distances,
        wigner_inv,
        grad_out_node_features,
    )?;
    check_len(
        "fairchem_esen_edgewise_gate_vjp grad_node_features",
        node_features.len(),
        grad_node_features.len(),
    )?;
    check_len(
        "fairchem_esen_edgewise_gate_vjp grad_wigner",
        wigner.len(),
        grad_wigner.len(),
    )?;
    check_len(
        "fairchem_esen_edgewise_gate_vjp grad_wigner_inv",
        wigner_inv.len(),
        grad_wigner_inv.len(),
    )?;
    if config.use_envelope {
        let Some(distances) = edge_distances else {
            return Err(invalid_esen_block(
                "edge_distances are required when envelope is enabled",
            ));
        };
        let Some(grad_distances) = grad_edge_distances.as_deref() else {
            return Err(invalid_esen_block(
                "grad_edge_distances are required when envelope is enabled",
            ));
        };
        check_len(
            "fairchem_esen_edgewise_gate_vjp grad_edge_distances",
            distances.len(),
            grad_distances.len(),
        )?;
    }

    let intermediates = run_edgewise_gate_intermediates(
        config,
        edge_sources,
        edge_targets,
        node_features,
        wigner,
        conv1_radial,
        conv1_m0_weight,
        conv1_m0_bias,
        conv1_m_weights,
        conv2_radial,
        conv2_m0_weight,
        conv2_m0_bias,
        conv2_m_weights,
        edge_distances,
        wigner_inv,
    )?;

    let coefficient_count = config.coefficient_count();
    let node_feature_dim = config.node_feature_dim();
    let forward_matmul_config = EsenSo3BatchedMatmulConfig::new(
        coefficient_count,
        coefficient_count,
        2 * config.sphere_channels,
    )
    .map_err(|err| invalid_esen_block(err.to_string()))?;
    let inverse_matmul_config = EsenSo3BatchedMatmulConfig::new(
        coefficient_count,
        coefficient_count,
        config.sphere_channels,
    )
    .map_err(|err| invalid_esen_block(err.to_string()))?;

    let mut grad_rotated_back = vec![0.0; edges * node_feature_dim];
    fairchem_esen_edge_to_node_scaled_scatter_vjp_edges(
        config.num_nodes,
        node_feature_dim,
        edge_targets,
        config.node_offset,
        1.0,
        grad_out_node_features,
        &mut grad_rotated_back,
    )?;

    let mut grad_after_envelope = vec![0.0; intermediates.after_envelope.len()];
    fairchem_esen_so3_batched_matmul_vjp_input(
        inverse_matmul_config,
        wigner_inv,
        &grad_rotated_back,
        &mut grad_after_envelope,
    )?;
    fairchem_esen_so3_batched_matmul_vjp_matrix(
        inverse_matmul_config,
        &intermediates.after_envelope,
        &grad_rotated_back,
        grad_wigner_inv,
    )?;

    let mut grad_conv2_out = vec![0.0; intermediates.conv2_out.len()];
    if config.use_envelope {
        let Some(distances) = edge_distances else {
            return Err(invalid_esen_block(
                "edge_distances are required when envelope is enabled",
            ));
        };
        let Some(ref mut grad_distances) = grad_edge_distances else {
            return Err(invalid_esen_block(
                "grad_edge_distances are required when envelope is enabled",
            ));
        };
        fairchem_esen_edge_envelope_modulation_vjp(
            node_feature_dim,
            distances,
            config.cutoff,
            config.envelope_exponent,
            &intermediates.conv2_out,
            &grad_after_envelope,
            &mut grad_conv2_out,
            grad_distances,
        )?;
    } else {
        grad_conv2_out.copy_from_slice(&grad_after_envelope);
    }

    let mut grad_activated = vec![0.0; intermediates.activated.len()];
    fairchem_esen_so2_convolution_vjp(
        config.conv2_config()?,
        &intermediates.activated,
        conv2_radial,
        conv2_m0_weight,
        conv2_m_weights,
        &grad_conv2_out,
        None,
        &mut grad_activated,
        grad_conv2_radial,
    )?;

    let mut grad_gates = vec![0.0; intermediates.gates.len()];
    let mut grad_conv1_out = vec![0.0; intermediates.conv1_out.len()];
    fairchem_esen_gate_activation_vjp(
        config.gate_config()?,
        &intermediates.gates,
        &intermediates.conv1_out,
        &grad_activated,
        &mut grad_gates,
        &mut grad_conv1_out,
    )?;

    let mut grad_rotated = vec![0.0; intermediates.rotated.len()];
    fairchem_esen_so2_convolution_vjp(
        config.conv1_config()?,
        &intermediates.rotated,
        conv1_radial,
        conv1_m0_weight,
        conv1_m_weights,
        &grad_conv1_out,
        Some(&grad_gates),
        &mut grad_rotated,
        grad_conv1_radial,
    )?;

    let mut grad_concat = vec![0.0; intermediates.concatenated.len()];
    fairchem_esen_so3_batched_matmul_vjp_input(
        forward_matmul_config,
        wigner,
        &grad_rotated,
        &mut grad_concat,
    )?;
    fairchem_esen_so3_batched_matmul_vjp_matrix(
        forward_matmul_config,
        &intermediates.concatenated,
        &grad_rotated,
        grad_wigner,
    )?;
    fairchem_esen_source_target_concat_vjp_nodes(
        config.edge_concat_feature_dim() / 2,
        edge_sources,
        edge_targets,
        config.node_offset,
        &grad_concat,
        grad_node_features,
    )
}

/// FairChem eSEN `GridAtomwise` path:
/// `SO3_Grid.to_grid -> Linear -> SiLU -> Linear -> SiLU -> Linear -> from_grid`.
///
/// The dense layers are pointwise over grid locations and row-major:
/// `weight1 = [hidden, sphere]`, `weight2 = [hidden, hidden]`, and
/// `weight3 = [sphere, hidden]`. FairChem's grid MLP uses no biases here.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_grid_atomwise_forward(
    config: EsenGridAtomwiseConfig,
    input: &[f64],
    to_grid_mat: &[f64],
    from_grid_mat: &[f64],
    weight1: &[f64],
    weight2: &[f64],
    weight3: &[f64],
    out: &mut [f64],
) -> Result<()> {
    let intermediates = run_grid_atomwise_intermediates(
        config,
        input,
        to_grid_mat,
        weight1,
        weight2,
        weight3,
        out,
    )?;
    fairchem_esen_so3_from_grid_forward(config.grid, &intermediates.grid_out, from_grid_mat, out)
}

/// Accumulate input VJP of [`fairchem_esen_grid_atomwise_forward`].
///
/// Model parameter and grid-matrix gradients are out of scope for this
/// fixed-weight runtime; this only propagates to input coefficients.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_grid_atomwise_vjp(
    config: EsenGridAtomwiseConfig,
    input: &[f64],
    to_grid_mat: &[f64],
    from_grid_mat: &[f64],
    weight1: &[f64],
    weight2: &[f64],
    weight3: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
) -> Result<()> {
    let intermediates = run_grid_atomwise_intermediates(
        config,
        input,
        to_grid_mat,
        weight1,
        weight2,
        weight3,
        grad_out,
    )?;
    check_len(
        "fairchem_esen_grid_atomwise_vjp grad_input",
        input.len(),
        grad_input.len(),
    )?;

    let nodes = input.len() / config.coefficient_node_stride();
    let mut grad_grid_out = vec![0.0; nodes * config.grid_node_stride()];
    fairchem_esen_so3_from_grid_vjp(config.grid, from_grid_mat, grad_out, &mut grad_grid_out)?;

    let mut grad_grid_in = vec![0.0; intermediates.grid_in.len()];
    grid_point_mlp_vjp(
        config,
        &intermediates.pre1,
        &intermediates.pre2,
        weight1,
        weight2,
        weight3,
        &grad_grid_out,
        &mut grad_grid_in,
    );
    fairchem_esen_so3_to_grid_vjp(config.grid, to_grid_mat, &grad_grid_in, grad_input)
}

/// FairChem eSEN `SpectralAtomwise` path.
///
/// This composes the scalar gate MLP, `SO3_Linear`, `GateActivation`, and the
/// final `SO3_Linear`. The scalar gate MLP is `Linear -> SiLU` over the scalar
/// coefficient row only, with row-major weight `[lmax * hidden, sphere]`.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_spectral_atomwise_forward(
    config: EsenSpectralAtomwiseConfig,
    input: &[f64],
    scalar_gate_weight: &[f64],
    scalar_gate_bias: &[f64],
    linear1_weight: &[f64],
    linear1_bias: &[f64],
    linear2_weight: &[f64],
    linear2_bias: &[f64],
    out: &mut [f64],
) -> Result<()> {
    let intermediates = run_spectral_atomwise_intermediates(
        config,
        input,
        scalar_gate_weight,
        scalar_gate_bias,
        linear1_weight,
        linear1_bias,
        out,
    )?;
    fairchem_esen_so3_linear_forward(
        config.so3_linear2_config()?,
        &intermediates.activated,
        linear2_weight,
        linear2_bias,
        out,
    )
}

/// Accumulate input VJP of [`fairchem_esen_spectral_atomwise_forward`].
///
/// Fixed model weights are constants; no parameter gradients are computed.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_spectral_atomwise_vjp(
    config: EsenSpectralAtomwiseConfig,
    input: &[f64],
    scalar_gate_weight: &[f64],
    scalar_gate_bias: &[f64],
    linear1_weight: &[f64],
    linear1_bias: &[f64],
    linear2_weight: &[f64],
    linear2_bias: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
) -> Result<()> {
    let intermediates = run_spectral_atomwise_intermediates(
        config,
        input,
        scalar_gate_weight,
        scalar_gate_bias,
        linear1_weight,
        linear1_bias,
        grad_out,
    )?;
    check_len(
        "fairchem_esen_spectral_atomwise_vjp grad_input",
        input.len(),
        grad_input.len(),
    )?;
    check_len(
        "fairchem_esen_spectral_atomwise_vjp linear2_bias",
        config.sphere_channels,
        linear2_bias.len(),
    )?;

    let mut grad_activated = vec![0.0; intermediates.activated.len()];
    fairchem_esen_so3_linear_vjp(
        config.so3_linear2_config()?,
        linear2_weight,
        grad_out,
        &mut grad_activated,
    )?;

    let mut grad_gates = vec![0.0; intermediates.gates.len()];
    let mut grad_hidden = vec![0.0; intermediates.hidden.len()];
    fairchem_esen_gate_activation_vjp(
        config.gate_config()?,
        &intermediates.gates,
        &intermediates.hidden,
        &grad_activated,
        &mut grad_gates,
        &mut grad_hidden,
    )?;

    fairchem_esen_so3_linear_vjp(
        config.so3_linear1_config()?,
        linear1_weight,
        &grad_hidden,
        grad_input,
    )?;
    spectral_scalar_gate_mlp_vjp(
        config,
        input,
        &intermediates.gate_pre,
        scalar_gate_weight,
        &grad_gates,
        grad_input,
    );
    Ok(())
}

/// f32 twin of [`fairchem_esen_spectral_atomwise_forward`].
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_spectral_atomwise_forward_f32(
    config: EsenSpectralAtomwiseConfig,
    input: &[f32],
    scalar_gate_weight: &[f32],
    scalar_gate_bias: &[f32],
    linear1_weight: &[f32],
    linear1_bias: &[f32],
    linear2_weight: &[f32],
    linear2_bias: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let intermediates = run_spectral_atomwise_intermediates_f32(
        config,
        input,
        scalar_gate_weight,
        scalar_gate_bias,
        linear1_weight,
        None,
        linear1_bias,
        out,
    )?;
    fairchem_esen_so3_linear_forward_f32(
        config.so3_linear2_config()?,
        &intermediates.activated,
        linear2_weight,
        linear2_bias,
        out,
    )
}

/// f32 spectral atomwise forward that forces the BLAS-gated scalar path.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_spectral_atomwise_forward_f32_blas_gate(
    config: EsenSpectralAtomwiseConfig,
    input: &[f32],
    scalar_gate_weight: &[f32],
    scalar_gate_bias: &[f32],
    linear1_weight: &[f32],
    linear1_bias: &[f32],
    linear2_weight: &[f32],
    linear2_bias: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let intermediates = run_spectral_atomwise_intermediates_f32_blas_gate(
        config,
        input,
        scalar_gate_weight,
        scalar_gate_bias,
        linear1_weight,
        linear1_bias,
        out,
    )?;
    fairchem_esen_so3_linear_forward_f32(
        config.so3_linear2_config()?,
        &intermediates.activated,
        linear2_weight,
        linear2_bias,
        out,
    )
}

/// f32 spectral atomwise forward that retains intermediates for a later VJP.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_spectral_atomwise_forward_with_cache_f32(
    config: EsenSpectralAtomwiseConfig,
    input: &[f32],
    scalar_gate_weight: &[f32],
    scalar_gate_bias: &[f32],
    linear1_weight: &[f32],
    linear1_weight_in_by_out: Option<&[f32]>,
    linear1_bias: &[f32],
    linear2_weight: &[f32],
    linear2_weight_in_by_out: Option<&[f32]>,
    linear2_bias: &[f32],
    out: &mut [f32],
    cache: &mut EsenSpectralAtomwiseForwardCacheF32,
) -> Result<()> {
    let intermediates = run_spectral_atomwise_intermediates_f32(
        config,
        input,
        scalar_gate_weight,
        scalar_gate_bias,
        linear1_weight,
        linear1_weight_in_by_out,
        linear1_bias,
        out,
    )?;
    let stage_start = esen_block_trace_start();
    #[cfg(all(
        feature = "armpl",
        target_arch = "aarch64",
        target_os = "linux",
        llmlip_aarch64_sve_kernels
    ))]
    let used_sve = linear2_weight_in_by_out.is_some_and(|weight_in_by_out| {
        spectral_so3_linear_forward_l2_sve_f32(
            config,
            &intermediates.activated,
            weight_in_by_out,
            linear2_bias,
            out,
        )
    });
    #[cfg(not(all(
        feature = "armpl",
        target_arch = "aarch64",
        target_os = "linux",
        llmlip_aarch64_sve_kernels
    )))]
    let used_sve = {
        let _ = linear2_weight_in_by_out;
        false
    };
    if !used_sve {
        fairchem_esen_so3_linear_forward_f32(
            config.so3_linear2_config()?,
            &intermediates.activated,
            linear2_weight,
            linear2_bias,
            out,
        )?;
    }
    trace_esen_block_stage("f32.atomwise.forward.so3_linear2", stage_start);
    *cache = intermediates;
    Ok(())
}

/// f32 twin of [`fairchem_esen_spectral_atomwise_vjp`].
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_spectral_atomwise_vjp_f32(
    config: EsenSpectralAtomwiseConfig,
    input: &[f32],
    scalar_gate_weight: &[f32],
    scalar_gate_bias: &[f32],
    linear1_weight: &[f32],
    linear1_bias: &[f32],
    linear2_weight: &[f32],
    linear2_bias: &[f32],
    grad_out: &[f32],
    grad_input: &mut [f32],
) -> Result<()> {
    let intermediates = run_spectral_atomwise_intermediates_f32(
        config,
        input,
        scalar_gate_weight,
        scalar_gate_bias,
        linear1_weight,
        None,
        linear1_bias,
        grad_out,
    )?;
    fairchem_esen_spectral_atomwise_vjp_from_cache_f32(
        config,
        input,
        scalar_gate_weight,
        linear1_weight,
        None,
        linear2_weight,
        None,
        linear2_bias,
        grad_out,
        &intermediates,
        grad_input,
    )
}

/// f32 spectral atomwise input VJP using intermediates saved by the forward pass.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_spectral_atomwise_vjp_from_cache_f32(
    config: EsenSpectralAtomwiseConfig,
    input: &[f32],
    scalar_gate_weight: &[f32],
    linear1_weight: &[f32],
    linear1_weight_in_by_out: Option<&[f32]>,
    linear2_weight: &[f32],
    linear2_weight_in_by_out: Option<&[f32]>,
    linear2_bias: &[f32],
    grad_out: &[f32],
    cache: &EsenSpectralAtomwiseForwardCacheF32,
    grad_input: &mut [f32],
) -> Result<()> {
    check_len(
        "fairchem_esen_spectral_atomwise_vjp_from_cache_f32 grad_input",
        input.len(),
        grad_input.len(),
    )?;
    check_len(
        "fairchem_esen_spectral_atomwise_vjp_from_cache_f32 grad_out",
        input.len(),
        grad_out.len(),
    )?;
    check_len(
        "fairchem_esen_spectral_atomwise_vjp_from_cache_f32 linear2_bias",
        config.sphere_channels,
        linear2_bias.len(),
    )?;
    let nodes = input.len() / config.input_node_stride();
    check_len(
        "fairchem_esen_spectral_atomwise_vjp_from_cache_f32 gate_pre",
        nodes * config.gate_node_stride(),
        cache.gate_pre.len(),
    )?;
    check_len(
        "fairchem_esen_spectral_atomwise_vjp_from_cache_f32 gates",
        nodes * config.gate_node_stride(),
        cache.gates.len(),
    )?;
    check_len(
        "fairchem_esen_spectral_atomwise_vjp_from_cache_f32 gate_sigmoids",
        nodes * config.gate_node_stride(),
        cache.gate_sigmoids.len(),
    )?;
    check_len(
        "fairchem_esen_spectral_atomwise_vjp_from_cache_f32 scalar_silu_derivatives",
        nodes * config.sphere_channels,
        cache.scalar_silu_derivatives.len(),
    )?;
    check_len(
        "fairchem_esen_spectral_atomwise_vjp_from_cache_f32 hidden",
        nodes * config.hidden_node_stride(),
        cache.hidden.len(),
    )?;
    check_len(
        "fairchem_esen_spectral_atomwise_vjp_from_cache_f32 activated",
        nodes * config.hidden_node_stride(),
        cache.activated.len(),
    )?;

    let mut grad_activated = vec![0.0_f32; cache.activated.len()];
    let stage_start = esen_block_trace_start();
    #[cfg(all(
        feature = "armpl",
        target_arch = "aarch64",
        target_os = "linux",
        llmlip_aarch64_sve_kernels
    ))]
    let used_sve = linear2_weight_in_by_out.is_some_and(|_| {
        spectral_so3_linear_vjp_l2_sve_f32(config, linear2_weight, grad_out, &mut grad_activated)
    });
    #[cfg(not(all(
        feature = "armpl",
        target_arch = "aarch64",
        target_os = "linux",
        llmlip_aarch64_sve_kernels
    )))]
    let used_sve = {
        let _ = linear2_weight_in_by_out;
        false
    };
    if !used_sve {
        fairchem_esen_so3_linear_vjp_f32(
            config.so3_linear2_config()?,
            linear2_weight,
            grad_out,
            &mut grad_activated,
        )?;
    }
    trace_esen_block_stage("f32.atomwise.vjp.so3_linear2", stage_start);

    let mut grad_gates = vec![0.0_f32; cache.gates.len()];
    let mut grad_hidden = vec![0.0_f32; cache.hidden.len()];
    let stage_start = esen_block_trace_start();
    fairchem_esen_gate_activation_vjp_with_cache_f32(
        config.gate_config()?,
        &cache.gates,
        &cache.hidden,
        &cache.gate_sigmoids,
        &cache.scalar_silu_derivatives,
        &grad_activated,
        &mut grad_gates,
        &mut grad_hidden,
    )?;
    trace_esen_block_stage("f32.atomwise.vjp.gate_activation", stage_start);

    let stage_start = esen_block_trace_start();
    #[cfg(all(
        feature = "armpl",
        target_arch = "aarch64",
        target_os = "linux",
        llmlip_aarch64_sve_kernels
    ))]
    let used_sve = linear1_weight_in_by_out.is_some_and(|_| {
        spectral_so3_linear_vjp_l2_sve_f32(config, linear1_weight, &grad_hidden, grad_input)
    });
    #[cfg(not(all(
        feature = "armpl",
        target_arch = "aarch64",
        target_os = "linux",
        llmlip_aarch64_sve_kernels
    )))]
    let used_sve = {
        let _ = linear1_weight_in_by_out;
        false
    };
    if !used_sve {
        fairchem_esen_so3_linear_vjp_f32(
            config.so3_linear1_config()?,
            linear1_weight,
            &grad_hidden,
            grad_input,
        )?;
    }
    trace_esen_block_stage("f32.atomwise.vjp.so3_linear1", stage_start);
    let stage_start = esen_block_trace_start();
    spectral_scalar_gate_mlp_vjp_f32(
        config,
        input,
        &cache.gate_pre,
        scalar_gate_weight,
        &grad_gates,
        grad_input,
    );
    trace_esen_block_stage("f32.atomwise.vjp.scalar_gate_mlp", stage_start);
    Ok(())
}

/// Write-only f32 twin of [`fairchem_esen_spectral_atomwise_vjp`].
///
/// This overwrites every `grad_input` component with the SO3-linear
/// contribution before accumulating the scalar-gate contribution into the
/// l=0 channels.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_spectral_atomwise_vjp_write_f32(
    config: EsenSpectralAtomwiseConfig,
    input: &[f32],
    scalar_gate_weight: &[f32],
    scalar_gate_bias: &[f32],
    linear1_weight: &[f32],
    linear1_bias: &[f32],
    linear2_weight: &[f32],
    linear2_bias: &[f32],
    grad_out: &[f32],
    grad_input: &mut [f32],
) -> Result<()> {
    let intermediates = run_spectral_atomwise_intermediates_f32(
        config,
        input,
        scalar_gate_weight,
        scalar_gate_bias,
        linear1_weight,
        None,
        linear1_bias,
        grad_out,
    )?;
    check_len(
        "fairchem_esen_spectral_atomwise_vjp_write_f32 grad_input",
        input.len(),
        grad_input.len(),
    )?;
    check_len(
        "fairchem_esen_spectral_atomwise_vjp_write_f32 linear2_bias",
        config.sphere_channels,
        linear2_bias.len(),
    )?;

    let mut grad_activated = vec![0.0_f32; intermediates.activated.len()];
    fairchem_esen_so3_linear_vjp_f32(
        config.so3_linear2_config()?,
        linear2_weight,
        grad_out,
        &mut grad_activated,
    )?;

    let mut grad_gates = vec![0.0_f32; intermediates.gates.len()];
    let mut grad_hidden = vec![0.0_f32; intermediates.hidden.len()];
    fairchem_esen_gate_activation_vjp_with_cache_f32(
        config.gate_config()?,
        &intermediates.gates,
        &intermediates.hidden,
        &intermediates.gate_sigmoids,
        &intermediates.scalar_silu_derivatives,
        &grad_activated,
        &mut grad_gates,
        &mut grad_hidden,
    )?;

    fairchem_esen_so3_linear_vjp_write_f32(
        config.so3_linear1_config()?,
        linear1_weight,
        &grad_hidden,
        grad_input,
    )?;
    spectral_scalar_gate_mlp_vjp_f32(
        config,
        input,
        &intermediates.gate_pre,
        scalar_gate_weight,
        &grad_gates,
        grad_input,
    );
    Ok(())
}

/// Write-only f32 VJP paired with
/// [`fairchem_esen_spectral_atomwise_forward_f32_blas_gate`].
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_spectral_atomwise_vjp_write_f32_blas_gate(
    config: EsenSpectralAtomwiseConfig,
    input: &[f32],
    scalar_gate_weight: &[f32],
    scalar_gate_bias: &[f32],
    linear1_weight: &[f32],
    linear1_bias: &[f32],
    linear2_weight: &[f32],
    linear2_bias: &[f32],
    grad_out: &[f32],
    grad_input: &mut [f32],
) -> Result<()> {
    let intermediates = run_spectral_atomwise_intermediates_f32_blas_gate(
        config,
        input,
        scalar_gate_weight,
        scalar_gate_bias,
        linear1_weight,
        linear1_bias,
        grad_out,
    )?;
    check_len(
        "fairchem_esen_spectral_atomwise_vjp_write_f32_blas_gate grad_input",
        input.len(),
        grad_input.len(),
    )?;
    check_len(
        "fairchem_esen_spectral_atomwise_vjp_write_f32_blas_gate linear2_bias",
        config.sphere_channels,
        linear2_bias.len(),
    )?;

    let mut grad_activated = vec![0.0_f32; intermediates.activated.len()];
    fairchem_esen_so3_linear_vjp_f32(
        config.so3_linear2_config()?,
        linear2_weight,
        grad_out,
        &mut grad_activated,
    )?;

    let mut grad_gates = vec![0.0_f32; intermediates.gates.len()];
    let mut grad_hidden = vec![0.0_f32; intermediates.hidden.len()];
    fairchem_esen_gate_activation_vjp_with_cache_f32(
        config.gate_config()?,
        &intermediates.gates,
        &intermediates.hidden,
        &intermediates.gate_sigmoids,
        &intermediates.scalar_silu_derivatives,
        &grad_activated,
        &mut grad_gates,
        &mut grad_hidden,
    )?;

    fairchem_esen_so3_linear_vjp_write_f32(
        config.so3_linear1_config()?,
        linear1_weight,
        &grad_hidden,
        grad_input,
    )?;
    spectral_scalar_gate_mlp_vjp_f32_blas_gate(
        config,
        input,
        &intermediates.gate_pre,
        scalar_gate_weight,
        &grad_gates,
        grad_input,
    );
    Ok(())
}

/// FairChem eSEN block with gate Edgewise and GridAtomwise branches.
///
/// This composes:
/// `norm1 -> Edgewise -> residual -> norm2 -> GridAtomwise -> residual`.
/// It currently requires the Edgewise reduced coefficient layout to match the
/// block's full node layout; that is the `m_max == l_max` case. Unknown OAM
/// checkpoint layouts must be validated before relaxing this contract.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_grid_block_forward(
    config: EsenGridBlockConfig,
    edge_sources: &[usize],
    edge_targets: &[usize],
    node_features: &[f64],
    norm1_weight: Option<&[f64]>,
    norm1_bias: Option<&[f64]>,
    wigner: &[f64],
    conv1_radial: Option<&[f64]>,
    conv1_m0_weight: &[f64],
    conv1_m0_bias: &[f64],
    conv1_m_weights: &[&[f64]],
    conv2_radial: Option<&[f64]>,
    conv2_m0_weight: &[f64],
    conv2_m0_bias: &[f64],
    conv2_m_weights: &[&[f64]],
    edge_distances: Option<&[f64]>,
    wigner_inv: &[f64],
    norm2_weight: Option<&[f64]>,
    norm2_bias: Option<&[f64]>,
    to_grid_mat: &[f64],
    from_grid_mat: &[f64],
    grid_weight1: &[f64],
    grid_weight2: &[f64],
    grid_weight3: &[f64],
    out: &mut [f64],
) -> Result<()> {
    let intermediates = run_grid_block_intermediates(
        config,
        edge_sources,
        edge_targets,
        node_features,
        norm1_weight,
        norm1_bias,
        wigner,
        conv1_radial,
        conv1_m0_weight,
        conv1_m0_bias,
        conv1_m_weights,
        conv2_radial,
        conv2_m0_weight,
        conv2_m0_bias,
        conv2_m_weights,
        edge_distances,
        wigner_inv,
        norm2_weight,
        norm2_bias,
        to_grid_mat,
        from_grid_mat,
        grid_weight1,
        grid_weight2,
        grid_weight3,
        out,
    )?;
    for ((target, atomwise), residual) in out
        .iter_mut()
        .zip(intermediates.atomwise_out.iter())
        .zip(intermediates.after_edge_residual.iter())
    {
        *target = atomwise + residual;
    }
    Ok(())
}

/// Accumulate input VJPs of [`fairchem_esen_grid_block_forward`].
#[allow(clippy::too_many_arguments)]
pub fn fairchem_esen_grid_block_vjp(
    config: EsenGridBlockConfig,
    edge_sources: &[usize],
    edge_targets: &[usize],
    node_features: &[f64],
    norm1_weight: Option<&[f64]>,
    norm1_bias: Option<&[f64]>,
    wigner: &[f64],
    conv1_radial: Option<&[f64]>,
    conv1_m0_weight: &[f64],
    conv1_m0_bias: &[f64],
    conv1_m_weights: &[&[f64]],
    conv2_radial: Option<&[f64]>,
    conv2_m0_weight: &[f64],
    conv2_m0_bias: &[f64],
    conv2_m_weights: &[&[f64]],
    edge_distances: Option<&[f64]>,
    wigner_inv: &[f64],
    norm2_weight: Option<&[f64]>,
    norm2_bias: Option<&[f64]>,
    to_grid_mat: &[f64],
    from_grid_mat: &[f64],
    grid_weight1: &[f64],
    grid_weight2: &[f64],
    grid_weight3: &[f64],
    grad_out: &[f64],
    grad_node_features: &mut [f64],
    grad_wigner: &mut [f64],
    grad_conv1_radial: Option<&mut [f64]>,
    grad_conv2_radial: Option<&mut [f64]>,
    grad_edge_distances: Option<&mut [f64]>,
    grad_wigner_inv: &mut [f64],
) -> Result<()> {
    let intermediates = run_grid_block_intermediates(
        config,
        edge_sources,
        edge_targets,
        node_features,
        norm1_weight,
        norm1_bias,
        wigner,
        conv1_radial,
        conv1_m0_weight,
        conv1_m0_bias,
        conv1_m_weights,
        conv2_radial,
        conv2_m0_weight,
        conv2_m0_bias,
        conv2_m_weights,
        edge_distances,
        wigner_inv,
        norm2_weight,
        norm2_bias,
        to_grid_mat,
        from_grid_mat,
        grid_weight1,
        grid_weight2,
        grid_weight3,
        grad_out,
    )?;
    check_len(
        "fairchem_esen_grid_block_vjp grad_node_features",
        node_features.len(),
        grad_node_features.len(),
    )?;

    let mut grad_after_edge_residual = grad_out.to_vec();
    let mut grad_norm2_out = vec![0.0; node_features.len()];
    fairchem_esen_grid_atomwise_vjp(
        config.atomwise,
        &intermediates.norm2_out,
        to_grid_mat,
        from_grid_mat,
        grid_weight1,
        grid_weight2,
        grid_weight3,
        grad_out,
        &mut grad_norm2_out,
    )?;
    fairchem_esen_rms_norm_sh_vjp(
        config.norm2,
        &intermediates.after_edge_residual,
        norm2_weight,
        norm2_bias,
        &grad_norm2_out,
        &mut grad_after_edge_residual,
    )?;

    for (target, grad) in grad_node_features
        .iter_mut()
        .zip(grad_after_edge_residual.iter())
    {
        *target += grad;
    }

    let mut grad_norm1_out = vec![0.0; node_features.len()];
    fairchem_esen_edgewise_gate_vjp(
        config.edge,
        edge_sources,
        edge_targets,
        &intermediates.norm1_out,
        wigner,
        conv1_radial,
        conv1_m0_weight,
        conv1_m0_bias,
        conv1_m_weights,
        conv2_radial,
        conv2_m0_weight,
        conv2_m0_bias,
        conv2_m_weights,
        edge_distances,
        wigner_inv,
        &grad_after_edge_residual,
        &mut grad_norm1_out,
        grad_wigner,
        grad_conv1_radial,
        grad_conv2_radial,
        grad_edge_distances,
        grad_wigner_inv,
    )?;
    fairchem_esen_rms_norm_sh_vjp(
        config.norm1,
        node_features,
        norm1_weight,
        norm1_bias,
        &grad_norm1_out,
        grad_node_features,
    )
}

struct EdgewiseGateIntermediates {
    concatenated: Vec<f64>,
    rotated: Vec<f64>,
    conv1_out: Vec<f64>,
    gates: Vec<f64>,
    activated: Vec<f64>,
    conv2_out: Vec<f64>,
    after_envelope: Vec<f64>,
    rotated_back: Vec<f64>,
}

struct GridBlockIntermediates {
    norm1_out: Vec<f64>,
    after_edge_residual: Vec<f64>,
    norm2_out: Vec<f64>,
    atomwise_out: Vec<f64>,
}

struct GridAtomwiseIntermediates {
    grid_in: Vec<f64>,
    pre1: Vec<f64>,
    pre2: Vec<f64>,
    grid_out: Vec<f64>,
}

struct SpectralAtomwiseIntermediates {
    gate_pre: Vec<f64>,
    gates: Vec<f64>,
    hidden: Vec<f64>,
    activated: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct EsenSpectralAtomwiseForwardCacheF32 {
    gate_pre: Vec<f32>,
    gates: Vec<f32>,
    gate_sigmoids: Vec<f32>,
    scalar_silu_derivatives: Vec<f32>,
    hidden: Vec<f32>,
    activated: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn run_grid_block_intermediates(
    config: EsenGridBlockConfig,
    edge_sources: &[usize],
    edge_targets: &[usize],
    node_features: &[f64],
    norm1_weight: Option<&[f64]>,
    norm1_bias: Option<&[f64]>,
    wigner: &[f64],
    conv1_radial: Option<&[f64]>,
    conv1_m0_weight: &[f64],
    conv1_m0_bias: &[f64],
    conv1_m_weights: &[&[f64]],
    conv2_radial: Option<&[f64]>,
    conv2_m0_weight: &[f64],
    conv2_m0_bias: &[f64],
    conv2_m_weights: &[&[f64]],
    edge_distances: Option<&[f64]>,
    wigner_inv: &[f64],
    norm2_weight: Option<&[f64]>,
    norm2_bias: Option<&[f64]>,
    to_grid_mat: &[f64],
    from_grid_mat: &[f64],
    grid_weight1: &[f64],
    grid_weight2: &[f64],
    grid_weight3: &[f64],
    output_like: &[f64],
) -> Result<GridBlockIntermediates> {
    check_len(
        "fairchem_esen_grid_block input",
        config.num_nodes() * config.node_stride(),
        node_features.len(),
    )?;
    check_len(
        "fairchem_esen_grid_block output_like",
        node_features.len(),
        output_like.len(),
    )?;

    let mut norm1_out = vec![0.0; node_features.len()];
    fairchem_esen_rms_norm_sh_forward(
        config.norm1,
        node_features,
        norm1_weight,
        norm1_bias,
        &mut norm1_out,
    )?;

    let mut edge_out = vec![0.0; node_features.len()];
    fairchem_esen_edgewise_gate_forward(
        config.edge,
        edge_sources,
        edge_targets,
        &norm1_out,
        wigner,
        conv1_radial,
        conv1_m0_weight,
        conv1_m0_bias,
        conv1_m_weights,
        conv2_radial,
        conv2_m0_weight,
        conv2_m0_bias,
        conv2_m_weights,
        edge_distances,
        wigner_inv,
        &mut edge_out,
    )?;

    let mut after_edge_residual = vec![0.0; node_features.len()];
    for ((target, input), edge) in after_edge_residual
        .iter_mut()
        .zip(node_features.iter())
        .zip(edge_out.iter())
    {
        *target = input + edge;
    }

    let mut norm2_out = vec![0.0; node_features.len()];
    fairchem_esen_rms_norm_sh_forward(
        config.norm2,
        &after_edge_residual,
        norm2_weight,
        norm2_bias,
        &mut norm2_out,
    )?;

    let mut atomwise_out = vec![0.0; node_features.len()];
    fairchem_esen_grid_atomwise_forward(
        config.atomwise,
        &norm2_out,
        to_grid_mat,
        from_grid_mat,
        grid_weight1,
        grid_weight2,
        grid_weight3,
        &mut atomwise_out,
    )?;
    Ok(GridBlockIntermediates {
        norm1_out,
        after_edge_residual,
        norm2_out,
        atomwise_out,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_spectral_atomwise_intermediates(
    config: EsenSpectralAtomwiseConfig,
    input: &[f64],
    scalar_gate_weight: &[f64],
    scalar_gate_bias: &[f64],
    linear1_weight: &[f64],
    linear1_bias: &[f64],
    output_like: &[f64],
) -> Result<SpectralAtomwiseIntermediates> {
    let nodes = check_spectral_atomwise_shapes(
        "fairchem_esen_spectral_atomwise",
        config,
        input,
        scalar_gate_weight,
        scalar_gate_bias,
        linear1_weight,
        linear1_bias,
        output_like,
    )?;
    let mut gate_pre = vec![0.0; nodes * config.gate_node_stride()];
    let mut gates = vec![0.0; gate_pre.len()];
    spectral_scalar_gate_mlp_forward(
        config,
        input,
        scalar_gate_weight,
        scalar_gate_bias,
        &mut gate_pre,
        &mut gates,
    );

    let mut hidden = vec![0.0; nodes * config.hidden_node_stride()];
    fairchem_esen_so3_linear_forward(
        config.so3_linear1_config()?,
        input,
        linear1_weight,
        linear1_bias,
        &mut hidden,
    )?;
    let mut activated = vec![0.0; hidden.len()];
    fairchem_esen_gate_activation_forward(config.gate_config()?, &gates, &hidden, &mut activated)?;
    Ok(SpectralAtomwiseIntermediates {
        gate_pre,
        gates,
        hidden,
        activated,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_spectral_atomwise_intermediates_f32(
    config: EsenSpectralAtomwiseConfig,
    input: &[f32],
    scalar_gate_weight: &[f32],
    scalar_gate_bias: &[f32],
    linear1_weight: &[f32],
    linear1_weight_in_by_out: Option<&[f32]>,
    linear1_bias: &[f32],
    output_like: &[f32],
) -> Result<EsenSpectralAtomwiseForwardCacheF32> {
    let nodes = check_spectral_atomwise_shapes_f32(
        "fairchem_esen_spectral_atomwise_f32",
        config,
        input,
        scalar_gate_weight,
        scalar_gate_bias,
        linear1_weight,
        linear1_bias,
        output_like,
    )?;
    let mut gate_pre = vec![0.0_f32; nodes * config.gate_node_stride()];
    let mut gates = vec![0.0_f32; gate_pre.len()];
    let stage_start = esen_block_trace_start();
    spectral_scalar_gate_mlp_forward_f32(
        config,
        input,
        scalar_gate_weight,
        scalar_gate_bias,
        &mut gate_pre,
        &mut gates,
    );
    trace_esen_block_stage("f32.atomwise.forward.scalar_gate_mlp", stage_start);

    let mut hidden = vec![0.0_f32; nodes * config.hidden_node_stride()];
    let stage_start = esen_block_trace_start();
    #[cfg(all(
        feature = "armpl",
        target_arch = "aarch64",
        target_os = "linux",
        llmlip_aarch64_sve_kernels
    ))]
    let used_sve = linear1_weight_in_by_out.is_some_and(|weight_in_by_out| {
        spectral_so3_linear_forward_l2_sve_f32(
            config,
            input,
            weight_in_by_out,
            linear1_bias,
            &mut hidden,
        )
    });
    #[cfg(not(all(
        feature = "armpl",
        target_arch = "aarch64",
        target_os = "linux",
        llmlip_aarch64_sve_kernels
    )))]
    let used_sve = {
        let _ = linear1_weight_in_by_out;
        false
    };
    if !used_sve {
        fairchem_esen_so3_linear_forward_f32(
            config.so3_linear1_config()?,
            input,
            linear1_weight,
            linear1_bias,
            &mut hidden,
        )?;
    }
    trace_esen_block_stage("f32.atomwise.forward.so3_linear1", stage_start);
    let mut activated = vec![0.0_f32; hidden.len()];
    let mut gate_sigmoids = vec![0.0_f32; gates.len()];
    let mut scalar_silu_derivatives = vec![0.0_f32; nodes * config.sphere_channels];
    let stage_start = esen_block_trace_start();
    fairchem_esen_gate_activation_forward_with_cache_f32(
        config.gate_config()?,
        &gates,
        &hidden,
        &mut activated,
        &mut gate_sigmoids,
        &mut scalar_silu_derivatives,
    )?;
    trace_esen_block_stage("f32.atomwise.forward.gate_activation", stage_start);
    Ok(EsenSpectralAtomwiseForwardCacheF32 {
        gate_pre,
        gates,
        gate_sigmoids,
        scalar_silu_derivatives,
        hidden,
        activated,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_spectral_atomwise_intermediates_f32_blas_gate(
    config: EsenSpectralAtomwiseConfig,
    input: &[f32],
    scalar_gate_weight: &[f32],
    scalar_gate_bias: &[f32],
    linear1_weight: &[f32],
    linear1_bias: &[f32],
    output_like: &[f32],
) -> Result<EsenSpectralAtomwiseForwardCacheF32> {
    let nodes = check_spectral_atomwise_shapes_f32(
        "fairchem_esen_spectral_atomwise_f32_blas_gate",
        config,
        input,
        scalar_gate_weight,
        scalar_gate_bias,
        linear1_weight,
        linear1_bias,
        output_like,
    )?;
    let mut gate_pre = vec![0.0_f32; nodes * config.gate_node_stride()];
    let mut gates = vec![0.0_f32; gate_pre.len()];
    spectral_scalar_gate_mlp_forward_f32_blas_gate(
        config,
        input,
        scalar_gate_weight,
        scalar_gate_bias,
        &mut gate_pre,
        &mut gates,
    );

    let mut hidden = vec![0.0_f32; nodes * config.hidden_node_stride()];
    fairchem_esen_so3_linear_forward_f32(
        config.so3_linear1_config()?,
        input,
        linear1_weight,
        linear1_bias,
        &mut hidden,
    )?;
    let mut activated = vec![0.0_f32; hidden.len()];
    let mut gate_sigmoids = vec![0.0_f32; gates.len()];
    let mut scalar_silu_derivatives = vec![0.0_f32; nodes * config.sphere_channels];
    fairchem_esen_gate_activation_forward_with_cache_f32(
        config.gate_config()?,
        &gates,
        &hidden,
        &mut activated,
        &mut gate_sigmoids,
        &mut scalar_silu_derivatives,
    )?;
    Ok(EsenSpectralAtomwiseForwardCacheF32 {
        gate_pre,
        gates,
        gate_sigmoids,
        scalar_silu_derivatives,
        hidden,
        activated,
    })
}

fn spectral_scalar_gate_mlp_forward(
    config: EsenSpectralAtomwiseConfig,
    input: &[f64],
    weight: &[f64],
    bias: &[f64],
    gate_pre: &mut [f64],
    gates: &mut [f64],
) {
    let nodes = input.len() / config.input_node_stride();
    for node in 0..nodes {
        let input_start = node * config.input_node_stride();
        let gate_start = node * config.gate_node_stride();
        for gate_index in 0..config.gate_node_stride() {
            let weight_start = gate_index * config.sphere_channels;
            let mut value = bias[gate_index];
            for channel in 0..config.sphere_channels {
                value += weight[weight_start + channel] * input[input_start + channel];
            }
            gate_pre[gate_start + gate_index] = value;
            gates[gate_start + gate_index] = silu(value);
        }
    }
}

fn spectral_scalar_gate_mlp_vjp(
    config: EsenSpectralAtomwiseConfig,
    input: &[f64],
    gate_pre: &[f64],
    weight: &[f64],
    grad_gates: &[f64],
    grad_input: &mut [f64],
) {
    let nodes = input.len() / config.input_node_stride();
    for node in 0..nodes {
        let input_start = node * config.input_node_stride();
        let gate_start = node * config.gate_node_stride();
        for gate_index in 0..config.gate_node_stride() {
            let grad_pre = grad_gates[gate_start + gate_index]
                * silu_derivative(gate_pre[gate_start + gate_index]);
            let weight_start = gate_index * config.sphere_channels;
            for channel in 0..config.sphere_channels {
                grad_input[input_start + channel] += weight[weight_start + channel] * grad_pre;
            }
        }
    }
}

fn spectral_scalar_gate_mlp_forward_f32(
    config: EsenSpectralAtomwiseConfig,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    gate_pre: &mut [f32],
    gates: &mut [f32],
) {
    #[cfg(all(
        target_arch = "aarch64",
        target_os = "linux",
        llmlip_aarch64_sve_kernels
    ))]
    if spectral_scalar_gate_mlp_forward_sve_f32(config, input, weight, bias, gate_pre, gates) {
        return;
    }

    let nodes = input.len() / config.input_node_stride();
    spectral_scalar_gate_mlp_forward_f32_scalar(
        config, nodes, input, weight, bias, gate_pre, gates,
    );
}

fn spectral_scalar_gate_mlp_forward_f32_blas_gate(
    config: EsenSpectralAtomwiseConfig,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    gate_pre: &mut [f32],
    gates: &mut [f32],
) {
    let nodes = input.len() / config.input_node_stride();
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    if nodes >= SPECTRAL_ATOMWISE_F32_BLAS_MIN_NODES
        && spectral_scalar_gate_mlp_forward_f32_blas(
            config, nodes, input, weight, bias, gate_pre, gates,
        )
    {
        return;
    }

    spectral_scalar_gate_mlp_forward_f32_scalar(
        config, nodes, input, weight, bias, gate_pre, gates,
    );
}

#[inline(always)]
fn spectral_scalar_gate_mlp_forward_f32_scalar(
    config: EsenSpectralAtomwiseConfig,
    nodes: usize,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    gate_pre: &mut [f32],
    gates: &mut [f32],
) {
    for node in 0..nodes {
        let input_start = node * config.input_node_stride();
        let gate_start = node * config.gate_node_stride();
        for gate_index in 0..config.gate_node_stride() {
            let weight_start = gate_index * config.sphere_channels;
            let mut value = bias[gate_index];
            for channel in 0..config.sphere_channels {
                value += weight[weight_start + channel] * input[input_start + channel];
            }
            gate_pre[gate_start + gate_index] = value;
            gates[gate_start + gate_index] = silu(value as f64) as f32;
        }
    }
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
const SPECTRAL_ATOMWISE_F32_BLAS_MIN_NODES: usize = 16;

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[inline(never)]
fn spectral_scalar_gate_mlp_forward_f32_blas(
    config: EsenSpectralAtomwiseConfig,
    nodes: usize,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    gate_pre: &mut [f32],
    gates: &mut [f32],
) -> bool {
    let sphere = config.sphere_channels;
    let gate_stride = config.gate_node_stride();
    let input_stride = config.input_node_stride();
    let mut scalar_input = vec![0.0_f32; nodes * sphere];
    for node in 0..nodes {
        let src = node * input_stride;
        let dst = node * sphere;
        scalar_input[dst..dst + sphere].copy_from_slice(&input[src..src + sphere]);
    }
    crate::blas_backend::dense_forward_output_major_weight_f32(
        &scalar_input,
        nodes,
        sphere,
        weight,
        gate_pre,
        gate_stride,
        1.0,
        0.0,
    );
    for (row_pre, row_gates) in gate_pre
        .chunks_exact_mut(gate_stride)
        .zip(gates.chunks_exact_mut(gate_stride))
    {
        for gate_index in 0..gate_stride {
            let value = row_pre[gate_index] + bias[gate_index];
            row_pre[gate_index] = value;
            row_gates[gate_index] = silu(value as f64) as f32;
        }
    }
    true
}

fn spectral_scalar_gate_mlp_vjp_f32(
    config: EsenSpectralAtomwiseConfig,
    input: &[f32],
    gate_pre: &[f32],
    weight: &[f32],
    grad_gates: &[f32],
    grad_input: &mut [f32],
) {
    #[cfg(all(
        target_arch = "aarch64",
        target_os = "linux",
        llmlip_aarch64_sve_kernels
    ))]
    if spectral_scalar_gate_mlp_vjp_sve_f32(config, input, gate_pre, weight, grad_gates, grad_input)
    {
        return;
    }

    let nodes = input.len() / config.input_node_stride();
    spectral_scalar_gate_mlp_vjp_f32_scalar(
        config, nodes, gate_pre, weight, grad_gates, grad_input,
    );
}

fn spectral_scalar_gate_mlp_vjp_f32_blas_gate(
    config: EsenSpectralAtomwiseConfig,
    input: &[f32],
    gate_pre: &[f32],
    weight: &[f32],
    grad_gates: &[f32],
    grad_input: &mut [f32],
) {
    let nodes = input.len() / config.input_node_stride();
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    if nodes >= SPECTRAL_ATOMWISE_F32_BLAS_MIN_NODES
        && spectral_scalar_gate_mlp_vjp_f32_blas(
            config, nodes, gate_pre, weight, grad_gates, grad_input,
        )
    {
        return;
    }

    spectral_scalar_gate_mlp_vjp_f32_scalar(
        config, nodes, gate_pre, weight, grad_gates, grad_input,
    );
}

#[inline(always)]
fn spectral_scalar_gate_mlp_vjp_f32_scalar(
    config: EsenSpectralAtomwiseConfig,
    nodes: usize,
    gate_pre: &[f32],
    weight: &[f32],
    grad_gates: &[f32],
    grad_input: &mut [f32],
) {
    for node in 0..nodes {
        let input_start = node * config.input_node_stride();
        let gate_start = node * config.gate_node_stride();
        for gate_index in 0..config.gate_node_stride() {
            let grad_pre = grad_gates[gate_start + gate_index]
                * silu_derivative(gate_pre[gate_start + gate_index] as f64) as f32;
            let weight_start = gate_index * config.sphere_channels;
            for channel in 0..config.sphere_channels {
                grad_input[input_start + channel] += weight[weight_start + channel] * grad_pre;
            }
        }
    }
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[inline(never)]
fn spectral_scalar_gate_mlp_vjp_f32_blas(
    config: EsenSpectralAtomwiseConfig,
    nodes: usize,
    gate_pre: &[f32],
    weight: &[f32],
    grad_gates: &[f32],
    grad_input: &mut [f32],
) -> bool {
    let sphere = config.sphere_channels;
    let gate_stride = config.gate_node_stride();
    let input_stride = config.input_node_stride();
    let mut grad_pre = vec![0.0_f32; nodes * gate_stride];
    for ((grad_pre_value, &grad_gate), &pre) in grad_pre
        .iter_mut()
        .zip(grad_gates.iter())
        .zip(gate_pre.iter())
    {
        *grad_pre_value = grad_gate * silu_derivative(pre as f64) as f32;
    }
    let mut grad_scalar_input = vec![0.0_f32; nodes * sphere];
    crate::blas_backend::dense_vjp_input_output_major_weight_f32(
        nodes,
        sphere,
        gate_stride,
        weight,
        &grad_pre,
        &mut grad_scalar_input,
        1.0,
        0.0,
    );
    for node in 0..nodes {
        let src = node * sphere;
        let dst = node * input_stride;
        for channel in 0..sphere {
            grad_input[dst + channel] += grad_scalar_input[src + channel];
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn run_grid_atomwise_intermediates(
    config: EsenGridAtomwiseConfig,
    input: &[f64],
    to_grid_mat: &[f64],
    weight1: &[f64],
    weight2: &[f64],
    weight3: &[f64],
    output_like: &[f64],
) -> Result<GridAtomwiseIntermediates> {
    let nodes = check_grid_atomwise_shapes(
        "fairchem_esen_grid_atomwise",
        config,
        input,
        to_grid_mat,
        weight1,
        weight2,
        weight3,
        output_like,
    )?;
    let mut grid_in = vec![0.0; nodes * config.grid_node_stride()];
    fairchem_esen_so3_to_grid_forward(config.grid, input, to_grid_mat, &mut grid_in)?;
    let mut pre1 = vec![0.0; nodes * config.grid_points() * config.hidden_channels];
    let mut pre2 = vec![0.0; pre1.len()];
    let mut grid_out = vec![0.0; grid_in.len()];
    grid_point_mlp_forward(
        config,
        &grid_in,
        weight1,
        weight2,
        weight3,
        &mut pre1,
        &mut pre2,
        &mut grid_out,
    );
    Ok(GridAtomwiseIntermediates {
        grid_in,
        pre1,
        pre2,
        grid_out,
    })
}

#[allow(clippy::too_many_arguments)]
fn grid_point_mlp_forward(
    config: EsenGridAtomwiseConfig,
    grid_in: &[f64],
    weight1: &[f64],
    weight2: &[f64],
    weight3: &[f64],
    pre1: &mut [f64],
    pre2: &mut [f64],
    grid_out: &mut [f64],
) {
    let sphere = config.sphere_channels();
    let hidden = config.hidden_channels;
    let points = grid_in.len() / sphere;
    for point in 0..points {
        let input_start = point * sphere;
        let hidden_start = point * hidden;
        let output_start = point * sphere;
        for hidden_index in 0..hidden {
            let weight_start = hidden_index * sphere;
            let mut value = 0.0;
            for channel in 0..sphere {
                value += weight1[weight_start + channel] * grid_in[input_start + channel];
            }
            pre1[hidden_start + hidden_index] = value;
        }
        for hidden_index in 0..hidden {
            let weight_start = hidden_index * hidden;
            let mut value = 0.0;
            for in_hidden in 0..hidden {
                value += weight2[weight_start + in_hidden] * silu(pre1[hidden_start + in_hidden]);
            }
            pre2[hidden_start + hidden_index] = value;
        }
        for channel in 0..sphere {
            let weight_start = channel * hidden;
            let mut value = 0.0;
            for hidden_index in 0..hidden {
                value +=
                    weight3[weight_start + hidden_index] * silu(pre2[hidden_start + hidden_index]);
            }
            grid_out[output_start + channel] = value;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn grid_point_mlp_vjp(
    config: EsenGridAtomwiseConfig,
    pre1: &[f64],
    pre2: &[f64],
    weight1: &[f64],
    weight2: &[f64],
    weight3: &[f64],
    grad_grid_out: &[f64],
    grad_grid_in: &mut [f64],
) {
    let sphere = config.sphere_channels();
    let hidden = config.hidden_channels;
    let points = grad_grid_in.len() / sphere;
    for point in 0..points {
        let input_start = point * sphere;
        let hidden_start = point * hidden;
        let output_start = point * sphere;
        let mut grad_pre2 = vec![0.0; hidden];
        for channel in 0..sphere {
            let grad = grad_grid_out[output_start + channel];
            let weight_start = channel * hidden;
            for hidden_index in 0..hidden {
                grad_pre2[hidden_index] += weight3[weight_start + hidden_index]
                    * grad
                    * silu_derivative(pre2[hidden_start + hidden_index]);
            }
        }

        let mut grad_pre1 = vec![0.0; hidden];
        for (hidden_out, &grad) in grad_pre2.iter().enumerate() {
            let weight_start = hidden_out * hidden;
            for hidden_in in 0..hidden {
                grad_pre1[hidden_in] += weight2[weight_start + hidden_in]
                    * grad
                    * silu_derivative(pre1[hidden_start + hidden_in]);
            }
        }

        for (hidden_index, &grad) in grad_pre1.iter().enumerate() {
            let weight_start = hidden_index * sphere;
            for channel in 0..sphere {
                grad_grid_in[input_start + channel] += weight1[weight_start + channel] * grad;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_edgewise_gate_intermediates(
    config: EsenEdgewiseGateConfig,
    edge_sources: &[usize],
    edge_targets: &[usize],
    node_features: &[f64],
    wigner: &[f64],
    conv1_radial: Option<&[f64]>,
    conv1_m0_weight: &[f64],
    conv1_m0_bias: &[f64],
    conv1_m_weights: &[&[f64]],
    conv2_radial: Option<&[f64]>,
    conv2_m0_weight: &[f64],
    conv2_m0_bias: &[f64],
    conv2_m_weights: &[&[f64]],
    edge_distances: Option<&[f64]>,
    wigner_inv: &[f64],
) -> Result<EdgewiseGateIntermediates> {
    let edges = edge_sources.len();
    let coefficient_count = config.coefficient_count();
    let node_feature_dim = config.node_feature_dim();
    let concat_feature_dim = config.edge_concat_feature_dim();
    let hidden_feature_dim = config.hidden_feature_dim();

    let mut concatenated = vec![0.0; edges * concat_feature_dim];
    fairchem_esen_source_target_concat_forward(
        node_feature_dim,
        edge_sources,
        edge_targets,
        config.node_offset,
        node_features,
        &mut concatenated,
    )?;

    let mut rotated = vec![0.0; edges * concat_feature_dim];
    fairchem_esen_so3_batched_matmul_forward(
        EsenSo3BatchedMatmulConfig::new(
            coefficient_count,
            coefficient_count,
            2 * config.sphere_channels,
        )
        .map_err(|err| invalid_esen_block(err.to_string()))?,
        wigner,
        &concatenated,
        &mut rotated,
    )?;

    let mut conv1_out = vec![0.0; edges * hidden_feature_dim];
    let mut gates = vec![0.0; edges * config.gate_feature_dim()];
    fairchem_esen_so2_convolution_forward(
        config.conv1_config()?,
        &rotated,
        conv1_radial,
        conv1_m0_weight,
        conv1_m0_bias,
        conv1_m_weights,
        &mut conv1_out,
        Some(&mut gates),
    )?;

    let mut activated = vec![0.0; conv1_out.len()];
    fairchem_esen_gate_activation_forward(
        config.gate_config()?,
        &gates,
        &conv1_out,
        &mut activated,
    )?;

    let mut conv2_out = vec![0.0; edges * node_feature_dim];
    fairchem_esen_so2_convolution_forward(
        config.conv2_config()?,
        &activated,
        conv2_radial,
        conv2_m0_weight,
        conv2_m0_bias,
        conv2_m_weights,
        &mut conv2_out,
        None,
    )?;

    let mut after_envelope = vec![0.0; conv2_out.len()];
    if config.use_envelope {
        let Some(distances) = edge_distances else {
            return Err(invalid_esen_block(
                "edge_distances are required when envelope is enabled",
            ));
        };
        fairchem_esen_edge_envelope_modulation_forward(
            node_feature_dim,
            distances,
            config.cutoff,
            config.envelope_exponent,
            &conv2_out,
            &mut after_envelope,
        )?;
    } else {
        after_envelope.copy_from_slice(&conv2_out);
    }

    let mut rotated_back = vec![0.0; edges * node_feature_dim];
    fairchem_esen_so3_batched_matmul_forward(
        EsenSo3BatchedMatmulConfig::new(
            coefficient_count,
            coefficient_count,
            config.sphere_channels,
        )
        .map_err(|err| invalid_esen_block(err.to_string()))?,
        wigner_inv,
        &after_envelope,
        &mut rotated_back,
    )?;

    Ok(EdgewiseGateIntermediates {
        concatenated,
        rotated,
        conv1_out,
        gates,
        activated,
        conv2_out,
        after_envelope,
        rotated_back,
    })
}

#[allow(clippy::too_many_arguments)]
fn check_edgewise_common_shapes(
    op: &'static str,
    config: EsenEdgewiseGateConfig,
    edge_sources: &[usize],
    edge_targets: &[usize],
    node_features: &[f64],
    wigner: &[f64],
    edge_distances: Option<&[f64]>,
    wigner_inv: &[f64],
    output_like: &[f64],
) -> Result<usize> {
    if edge_sources.len() != edge_targets.len() {
        return Err(CoreError::ShapeMismatch {
            op,
            expected: edge_sources.len(),
            actual: edge_targets.len(),
        });
    }
    let edges = edge_sources.len();
    check_len(
        op,
        config.num_nodes * config.node_feature_dim(),
        node_features.len(),
    )?;
    check_len(
        op,
        config.num_nodes * config.node_feature_dim(),
        output_like.len(),
    )?;
    let matrix_len = edges * config.coefficient_count() * config.coefficient_count();
    check_len(
        "fairchem_esen_edgewise_gate wigner",
        matrix_len,
        wigner.len(),
    )?;
    check_len(
        "fairchem_esen_edgewise_gate wigner_inv",
        matrix_len,
        wigner_inv.len(),
    )?;
    if config.use_envelope {
        let Some(distances) = edge_distances else {
            return Err(invalid_esen_block(
                "edge_distances are required when envelope is enabled",
            ));
        };
        check_len(
            "fairchem_esen_edgewise_gate edge_distances",
            edges,
            distances.len(),
        )?;
    } else if let Some(distances) = edge_distances {
        check_len(
            "fairchem_esen_edgewise_gate edge_distances",
            edges,
            distances.len(),
        )?;
    }
    for &node in edge_sources.iter().chain(edge_targets.iter()) {
        if node < config.node_offset {
            return Err(invalid_esen_block(format!(
                "edge endpoint {node} is below node_offset {}",
                config.node_offset
            )));
        }
        let local = node - config.node_offset;
        if local >= config.num_nodes {
            return Err(invalid_esen_block(format!(
                "edge endpoint {node} maps to local node {local}, but num_nodes is {}",
                config.num_nodes
            )));
        }
    }
    Ok(edges)
}

#[allow(clippy::too_many_arguments)]
fn check_grid_atomwise_shapes(
    op: &'static str,
    config: EsenGridAtomwiseConfig,
    input: &[f64],
    to_grid_mat: &[f64],
    weight1: &[f64],
    weight2: &[f64],
    weight3: &[f64],
    output_like: &[f64],
) -> Result<usize> {
    check_len(op, input.len(), output_like.len())?;
    if !input.len().is_multiple_of(config.coefficient_node_stride()) {
        return Err(invalid_esen_block(format!(
            "{op} input length {} is not a multiple of coefficient node stride {}",
            input.len(),
            config.coefficient_node_stride()
        )));
    }
    check_len(
        "fairchem_esen_grid_atomwise to_grid_mat",
        config.grid.matrix_len(),
        to_grid_mat.len(),
    )?;
    check_len(
        "fairchem_esen_grid_atomwise weight1",
        config.weight1_len(),
        weight1.len(),
    )?;
    check_len(
        "fairchem_esen_grid_atomwise weight2",
        config.weight2_len(),
        weight2.len(),
    )?;
    check_len(
        "fairchem_esen_grid_atomwise weight3",
        config.weight3_len(),
        weight3.len(),
    )?;
    Ok(input.len() / config.coefficient_node_stride())
}

#[allow(clippy::too_many_arguments)]
fn check_spectral_atomwise_shapes(
    op: &'static str,
    config: EsenSpectralAtomwiseConfig,
    input: &[f64],
    scalar_gate_weight: &[f64],
    scalar_gate_bias: &[f64],
    linear1_weight: &[f64],
    linear1_bias: &[f64],
    output_like: &[f64],
) -> Result<usize> {
    check_len(op, input.len(), output_like.len())?;
    if !input.len().is_multiple_of(config.input_node_stride()) {
        return Err(invalid_esen_block(format!(
            "{op} input length {} is not a multiple of input node stride {}",
            input.len(),
            config.input_node_stride()
        )));
    }
    check_len(
        "fairchem_esen_spectral_atomwise scalar_gate_weight",
        config.scalar_gate_weight_len(),
        scalar_gate_weight.len(),
    )?;
    check_len(
        "fairchem_esen_spectral_atomwise scalar_gate_bias",
        config.scalar_gate_bias_len(),
        scalar_gate_bias.len(),
    )?;
    let linear1_config = config.so3_linear1_config()?;
    check_len(
        "fairchem_esen_spectral_atomwise linear1_weight",
        linear1_config.weight_len(),
        linear1_weight.len(),
    )?;
    check_len(
        "fairchem_esen_spectral_atomwise linear1_bias",
        config.hidden_channels,
        linear1_bias.len(),
    )?;
    Ok(input.len() / config.input_node_stride())
}

#[allow(clippy::too_many_arguments)]
fn check_spectral_atomwise_shapes_f32(
    op: &'static str,
    config: EsenSpectralAtomwiseConfig,
    input: &[f32],
    scalar_gate_weight: &[f32],
    scalar_gate_bias: &[f32],
    linear1_weight: &[f32],
    linear1_bias: &[f32],
    output_like: &[f32],
) -> Result<usize> {
    check_len(op, input.len(), output_like.len())?;
    if !input.len().is_multiple_of(config.input_node_stride()) {
        return Err(invalid_esen_block(format!(
            "{op} input length {} is not a multiple of input node stride {}",
            input.len(),
            config.input_node_stride()
        )));
    }
    check_len(
        "fairchem_esen_spectral_atomwise_f32 scalar_gate_weight",
        config.scalar_gate_weight_len(),
        scalar_gate_weight.len(),
    )?;
    check_len(
        "fairchem_esen_spectral_atomwise_f32 scalar_gate_bias",
        config.scalar_gate_bias_len(),
        scalar_gate_bias.len(),
    )?;
    let linear1_config = config.so3_linear1_config()?;
    check_len(
        "fairchem_esen_spectral_atomwise_f32 linear1_weight",
        linear1_config.weight_len(),
        linear1_weight.len(),
    )?;
    check_len(
        "fairchem_esen_spectral_atomwise_f32 linear1_bias",
        config.hidden_channels,
        linear1_bias.len(),
    )?;
    Ok(input.len() / config.input_node_stride())
}

fn reduced_coefficient_count(l_max: usize, m_max: usize) -> Option<usize> {
    let mut count = 0usize;
    for degree in 0..=l_max {
        count = count.checked_add(2 * m_max.min(degree) + 1)?;
    }
    Some(count)
}

fn check_len(op: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected != actual {
        return Err(CoreError::ShapeMismatch {
            op,
            expected,
            actual,
        });
    }
    Ok(())
}

fn invalid_esen_block(message: impl Into<String>) -> CoreError {
    CoreError::Unsupported(format!("invalid eSEN block primitive: {}", message.into()))
}

#[cfg(test)]
#[cfg(all(
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
))]
mod tests {
    use super::*;

    #[cfg(all(
        target_arch = "aarch64",
        target_os = "linux",
        llmlip_aarch64_sve_kernels
    ))]
    #[test]
    fn spectral_scalar_gate_mlp_sve_wrappers_match_scalar_reference() -> Result<()> {
        if !std::arch::is_aarch64_feature_detected!("sve") {
            return Ok(());
        }

        let config = EsenSpectralAtomwiseConfig::new(2, 13, 7)?;
        let nodes = 3usize;
        let input_len = nodes * config.input_node_stride();
        let gate_len = nodes * config.gate_node_stride();
        let input = (0..input_len)
            .map(|i| (i as f32 * 0.017 + 0.13).sin() * 0.4)
            .collect::<Vec<_>>();
        let weight = (0..config.scalar_gate_weight_len())
            .map(|i| (i as f32 * 0.031 + 0.07).cos() * 0.25)
            .collect::<Vec<_>>();
        let bias = (0..config.scalar_gate_bias_len())
            .map(|i| (i as f32 * 0.19).sin() * 0.05)
            .collect::<Vec<_>>();

        let mut expected_pre = vec![0.0_f32; gate_len];
        let mut expected_gates = vec![0.0_f32; gate_len];
        for node in 0..nodes {
            let input_start = node * config.input_node_stride();
            let gate_start = node * config.gate_node_stride();
            for gate_index in 0..config.gate_node_stride() {
                let weight_start = gate_index * config.sphere_channels;
                let mut value = bias[gate_index];
                for channel in 0..config.sphere_channels {
                    value += weight[weight_start + channel] * input[input_start + channel];
                }
                expected_pre[gate_start + gate_index] = value;
                expected_gates[gate_start + gate_index] = silu(value as f64) as f32;
            }
        }

        let mut actual_pre = vec![f32::NAN; gate_len];
        let mut actual_gates = vec![f32::NAN; gate_len];
        assert!(spectral_scalar_gate_mlp_forward_sve_f32(
            config,
            &input,
            &weight,
            &bias,
            &mut actual_pre,
            &mut actual_gates,
        ));
        for (name, actual, expected, tolerance) in [
            ("gate_pre", &actual_pre, &expected_pre, 2.0e-5_f32),
            ("gates", &actual_gates, &expected_gates, 2.0e-5_f32),
        ] {
            for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
                let abs = (actual - expected).abs();
                assert!(
                    abs <= tolerance,
                    "{name}[{index}] actual={actual:.8e} expected={expected:.8e} abs={abs:.3e}"
                );
            }
        }

        let grad_gates = (0..gate_len)
            .map(|i| (i as f32 * 0.023 + 0.11).sin() * 0.2)
            .collect::<Vec<_>>();
        let base_grad_input = (0..input_len)
            .map(|i| (i as f32 * 0.029 + 0.05).cos() * 0.01)
            .collect::<Vec<_>>();
        let mut expected_grad_input = base_grad_input.clone();
        for node in 0..nodes {
            let input_start = node * config.input_node_stride();
            let gate_start = node * config.gate_node_stride();
            for gate_index in 0..config.gate_node_stride() {
                let grad_pre = grad_gates[gate_start + gate_index]
                    * silu_derivative(expected_pre[gate_start + gate_index] as f64) as f32;
                let weight_start = gate_index * config.sphere_channels;
                for channel in 0..config.sphere_channels {
                    expected_grad_input[input_start + channel] +=
                        weight[weight_start + channel] * grad_pre;
                }
            }
        }

        let mut actual_grad_input = base_grad_input;
        assert!(spectral_scalar_gate_mlp_vjp_sve_f32(
            config,
            &input,
            &expected_pre,
            &weight,
            &grad_gates,
            &mut actual_grad_input,
        ));
        for (index, (&actual, &expected)) in actual_grad_input
            .iter()
            .zip(expected_grad_input.iter())
            .enumerate()
        {
            let abs = (actual - expected).abs();
            assert!(
                abs <= 5.0e-5,
                "grad_input[{index}] actual={actual:.8e} expected={expected:.8e} abs={abs:.3e}"
            );
        }
        Ok(())
    }

    #[cfg(all(
        feature = "armpl",
        target_arch = "aarch64",
        target_os = "linux",
        llmlip_aarch64_sve_kernels
    ))]
    #[test]
    fn spectral_so3_linear_sve_wrappers_match_blas_reference() -> Result<()> {
        if !std::arch::is_aarch64_feature_detected!("sve") {
            return Ok(());
        }

        let config = EsenSpectralAtomwiseConfig::new(2, 16, 16)?;
        let linear_config = config.so3_linear1_config()?;
        let nodes = 5usize;
        let input_len = nodes * config.input_node_stride();
        let weight_len = 3 * config.sphere_channels * config.sphere_channels;
        let input = (0..input_len)
            .map(|i| (i as f32 * 0.013 + 0.17).sin() * 0.3)
            .collect::<Vec<_>>();
        let weight = (0..weight_len)
            .map(|i| (i as f32 * 0.019 + 0.23).cos() * 0.2)
            .collect::<Vec<_>>();
        let bias = (0..config.sphere_channels)
            .map(|i| (i as f32 * 0.11).sin() * 0.04)
            .collect::<Vec<_>>();
        let mut weight_in_by_out = vec![0.0_f32; weight_len];
        let per_degree = config.sphere_channels * config.sphere_channels;
        for degree in 0..3 {
            let base = degree * per_degree;
            for out_channel in 0..config.sphere_channels {
                for in_channel in 0..config.sphere_channels {
                    weight_in_by_out[base + in_channel * config.sphere_channels + out_channel] =
                        weight[base + out_channel * config.sphere_channels + in_channel];
                }
            }
        }

        let mut expected_forward = vec![0.0_f32; input_len];
        fairchem_esen_so3_linear_forward_f32(
            linear_config,
            &input,
            &weight,
            &bias,
            &mut expected_forward,
        )?;
        let mut actual_forward = vec![f32::NAN; input_len];
        assert!(spectral_so3_linear_forward_l2_sve_f32(
            config,
            &input,
            &weight_in_by_out,
            &bias,
            &mut actual_forward,
        ));
        for (index, (&actual, &expected)) in actual_forward
            .iter()
            .zip(expected_forward.iter())
            .enumerate()
        {
            let abs = (actual - expected).abs();
            assert!(
                abs <= 2.0e-5,
                "forward[{index}] actual={actual:.8e} expected={expected:.8e} abs={abs:.3e}"
            );
        }

        let grad_out = (0..input_len)
            .map(|i| (i as f32 * 0.007 + 0.41).cos() * 0.2)
            .collect::<Vec<_>>();
        let base_grad_input = (0..input_len)
            .map(|i| (i as f32 * 0.021 + 0.09).sin() * 0.01)
            .collect::<Vec<_>>();
        let mut expected_grad_input = base_grad_input.clone();
        fairchem_esen_so3_linear_vjp_f32(
            linear_config,
            &weight,
            &grad_out,
            &mut expected_grad_input,
        )?;
        let mut actual_grad_input = base_grad_input;
        assert!(spectral_so3_linear_vjp_l2_sve_f32(
            config,
            &weight,
            &grad_out,
            &mut actual_grad_input,
        ));
        for (index, (&actual, &expected)) in actual_grad_input
            .iter()
            .zip(expected_grad_input.iter())
            .enumerate()
        {
            let abs = (actual - expected).abs();
            assert!(
                abs <= 2.0e-5,
                "vjp[{index}] actual={actual:.8e} expected={expected:.8e} abs={abs:.3e}"
            );
        }
        Ok(())
    }
}
