//! FairChem UMA Mixture-of-Linear-Experts primitives.
//!
//! UMA uses MoLE layers as dense linear layers whose weight matrix is mixed
//! per system from expert weights. The learned bias is shared across experts.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use e3nn_core::{
    so2::{
        fairchem_esen_so2_radial_modulation_forward, EsenSo2ConvolutionConfig,
        EsenSo2RadialModulationConfig,
    },
    so3::{
        fairchem_esen_l_to_m_forward, fairchem_esen_m_to_l_forward, EsenCoefficientMappingConfig,
    },
    CoreError, Result,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UmaMoleLinearConfig {
    pub systems: usize,
    pub experts: usize,
    pub in_dim: usize,
    pub out_dim: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UmaRoutingMlpConfig {
    pub rows: usize,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub experts: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UmaMoleSo2ConvolutionConfig {
    pub systems: usize,
    pub experts: usize,
    pub so2: EsenSo2ConvolutionConfig,
}

#[derive(Clone, Copy, Debug)]
pub struct UmaRoutingMlpWeights<'a> {
    pub layer0_weight: &'a [f64],
    pub layer0_bias: &'a [f64],
    pub layer2_weight: &'a [f64],
    pub layer2_bias: &'a [f64],
    pub layer4_weight: &'a [f64],
    pub layer4_bias: &'a [f64],
}

#[derive(Debug)]
pub struct UmaRoutingMlpWorkspaceMut<'a> {
    pub hidden0: &'a mut [f64],
    pub hidden1: &'a mut [f64],
    pub logits: &'a mut [f64],
    pub coefficients: &'a mut [f64],
}

/// FairChem UMA adds this epsilon after softmax; rows intentionally sum to
/// `1.0 + experts * UMA_MOLE_SOFTMAX_EPSILON` rather than one.
pub const UMA_MOLE_SOFTMAX_EPSILON: f64 = 0.005;

pub fn fairchem_uma_mole_softmax_plus_epsilon_rows(
    logits: &[f64],
    rows: usize,
    experts: usize,
    out: &mut [f64],
) -> Result<()> {
    if rows == 0 || experts == 0 {
        return Err(invalid_mole(format!(
            "UMA MoLE softmax dimensions must be nonzero, got rows={rows} experts={experts}"
        )));
    }
    check_len("UMA MoLE softmax logits", rows * experts, logits.len())?;
    check_len("UMA MoLE softmax out", rows * experts, out.len())?;
    for row in 0..rows {
        let start = row * experts;
        let row_logits = &logits[start..start + experts];
        let row_out = &mut out[start..start + experts];
        let max_value = row_logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !max_value.is_finite() {
            return Err(invalid_mole("UMA MoLE softmax logits must be finite"));
        }
        let mut sum = 0.0;
        for (value, output) in row_logits.iter().zip(row_out.iter_mut()) {
            if !value.is_finite() {
                return Err(invalid_mole("UMA MoLE softmax logits must be finite"));
            }
            let exp_value = (*value - max_value).exp();
            *output = exp_value;
            sum += exp_value;
        }
        if !(sum.is_finite() && sum > 0.0) {
            return Err(invalid_mole("UMA MoLE softmax normalization is invalid"));
        }
        for output in row_out {
            *output = *output / sum + UMA_MOLE_SOFTMAX_EPSILON;
        }
    }
    Ok(())
}

pub fn fairchem_uma_routing_mlp_forward(
    config: UmaRoutingMlpConfig,
    input: &[f64],
    weights: UmaRoutingMlpWeights<'_>,
    workspace: UmaRoutingMlpWorkspaceMut<'_>,
) -> Result<()> {
    validate_routing_mlp_config(config)?;
    check_len(
        "UMA routing MLP input",
        config.rows * config.input_dim,
        input.len(),
    )?;
    check_len(
        "UMA routing MLP layer0_weight",
        config.hidden_dim * config.input_dim,
        weights.layer0_weight.len(),
    )?;
    check_len(
        "UMA routing MLP layer0_bias",
        config.hidden_dim,
        weights.layer0_bias.len(),
    )?;
    check_len(
        "UMA routing MLP layer2_weight",
        config.hidden_dim * config.hidden_dim,
        weights.layer2_weight.len(),
    )?;
    check_len(
        "UMA routing MLP layer2_bias",
        config.hidden_dim,
        weights.layer2_bias.len(),
    )?;
    check_len(
        "UMA routing MLP layer4_weight",
        config.experts * config.hidden_dim,
        weights.layer4_weight.len(),
    )?;
    check_len(
        "UMA routing MLP layer4_bias",
        config.experts,
        weights.layer4_bias.len(),
    )?;
    check_len(
        "UMA routing MLP hidden0",
        config.rows * config.hidden_dim,
        workspace.hidden0.len(),
    )?;
    check_len(
        "UMA routing MLP hidden1",
        config.rows * config.hidden_dim,
        workspace.hidden1.len(),
    )?;
    check_len(
        "UMA routing MLP logits",
        config.rows * config.experts,
        workspace.logits.len(),
    )?;
    check_len(
        "UMA routing MLP coefficients",
        config.rows * config.experts,
        workspace.coefficients.len(),
    )?;

    dense_silu_rows(
        input,
        config.rows,
        config.input_dim,
        config.hidden_dim,
        weights.layer0_weight,
        weights.layer0_bias,
        workspace.hidden0,
    );
    dense_silu_rows(
        workspace.hidden0,
        config.rows,
        config.hidden_dim,
        config.hidden_dim,
        weights.layer2_weight,
        weights.layer2_bias,
        workspace.hidden1,
    );
    dense_silu_rows(
        workspace.hidden1,
        config.rows,
        config.hidden_dim,
        config.experts,
        weights.layer4_weight,
        weights.layer4_bias,
        workspace.logits,
    );
    // FairChem applies SiLU after the final routing linear before
    // softmax-plus-epsilon, so these logits are post-activation values.
    fairchem_uma_mole_softmax_plus_epsilon_rows(
        workspace.logits,
        config.rows,
        config.experts,
        workspace.coefficients,
    )
}

pub fn fairchem_uma_mole_linear_forward(
    config: UmaMoleLinearConfig,
    input: &[f64],
    expert_weights: &[f64],
    bias: &[f64],
    coefficients: &[f64],
    system_sizes: &[usize],
    out: &mut [f64],
) -> Result<()> {
    validate_mole_config(config)?;
    let rows = check_system_sizes(config.systems, system_sizes)?;
    check_len("UMA MoLE linear input", rows * config.in_dim, input.len())?;
    check_len(
        "UMA MoLE linear expert_weights",
        config.experts * config.out_dim * config.in_dim,
        expert_weights.len(),
    )?;
    check_len("UMA MoLE linear bias", config.out_dim, bias.len())?;
    check_len(
        "UMA MoLE linear coefficients",
        config.systems * config.experts,
        coefficients.len(),
    )?;
    check_len("UMA MoLE linear out", rows * config.out_dim, out.len())?;

    let mut row_offset = 0usize;
    for system in 0..config.systems {
        let coeff_row = &coefficients[system * config.experts..(system + 1) * config.experts];
        for local_row in 0..system_sizes[system] {
            let row = row_offset + local_row;
            let input_row = &input[row * config.in_dim..(row + 1) * config.in_dim];
            let out_row = &mut out[row * config.out_dim..(row + 1) * config.out_dim];
            out_row.copy_from_slice(bias);
            for (expert, coefficient) in coeff_row.iter().copied().enumerate() {
                let expert_start = expert * config.out_dim * config.in_dim;
                for (out_index, out_value) in out_row.iter_mut().enumerate() {
                    let weight_row_start = expert_start + out_index * config.in_dim;
                    let mut dot = 0.0;
                    for in_index in 0..config.in_dim {
                        dot += expert_weights[weight_row_start + in_index] * input_row[in_index];
                    }
                    *out_value += coefficient * dot;
                }
            }
        }
        row_offset += system_sizes[system];
    }
    Ok(())
}

pub fn fairchem_uma_mole_linear_vjp_input_add(
    config: UmaMoleLinearConfig,
    grad_out: &[f64],
    expert_weights: &[f64],
    coefficients: &[f64],
    system_sizes: &[usize],
    grad_input: &mut [f64],
) -> Result<()> {
    validate_mole_config(config)?;
    let rows = check_system_sizes(config.systems, system_sizes)?;
    check_len(
        "UMA MoLE linear grad_out",
        rows * config.out_dim,
        grad_out.len(),
    )?;
    check_len(
        "UMA MoLE linear expert_weights",
        config.experts * config.out_dim * config.in_dim,
        expert_weights.len(),
    )?;
    check_len(
        "UMA MoLE linear coefficients",
        config.systems * config.experts,
        coefficients.len(),
    )?;
    check_len(
        "UMA MoLE linear grad_input",
        rows * config.in_dim,
        grad_input.len(),
    )?;

    let mut row_offset = 0usize;
    for system in 0..config.systems {
        let coeff_row = &coefficients[system * config.experts..(system + 1) * config.experts];
        for local_row in 0..system_sizes[system] {
            let row = row_offset + local_row;
            let grad_out_row = &grad_out[row * config.out_dim..(row + 1) * config.out_dim];
            let grad_input_row = &mut grad_input[row * config.in_dim..(row + 1) * config.in_dim];
            for (expert, coefficient) in coeff_row.iter().copied().enumerate() {
                let expert_start = expert * config.out_dim * config.in_dim;
                for (out_index, grad_out_value) in grad_out_row.iter().copied().enumerate() {
                    let scaled_grad = coefficient * grad_out_value;
                    let weight_row_start = expert_start + out_index * config.in_dim;
                    for (in_index, grad_input_value) in grad_input_row.iter_mut().enumerate() {
                        *grad_input_value +=
                            expert_weights[weight_row_start + in_index] * scaled_grad;
                    }
                }
            }
        }
        row_offset += system_sizes[system];
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn fairchem_uma_mole_so2_convolution_forward(
    config: UmaMoleSo2ConvolutionConfig,
    input_l_major: &[f64],
    radial: Option<&[f64]>,
    m0_expert_weights: &[f64],
    m0_bias: &[f64],
    m_expert_weights: &[&[f64]],
    coefficients: &[f64],
    system_sizes: &[usize],
    out_l_major: &mut [f64],
    extra_m0_out: Option<&mut [f64]>,
) -> Result<()> {
    validate_mole_so2_config(config)?;
    let rows = check_system_sizes(config.systems, system_sizes)?;
    check_len(
        "UMA MoLE SO2 input",
        rows * config.so2.input_node_stride(),
        input_l_major.len(),
    )?;
    if let Some(radial_values) = radial {
        check_len(
            "UMA MoLE SO2 radial",
            rows * config.so2.radial_node_stride(),
            radial_values.len(),
        )?;
    }
    check_len(
        "UMA MoLE SO2 m0_expert_weights",
        config.experts * config.so2.m0_weight_len(),
        m0_expert_weights.len(),
    )?;
    check_len(
        "UMA MoLE SO2 m0_bias",
        config.so2.m0_bias_len(),
        m0_bias.len(),
    )?;
    check_len(
        "UMA MoLE SO2 m_expert_weights",
        config.so2.m_max(),
        m_expert_weights.len(),
    )?;
    for order in 1..=config.so2.m_max() {
        check_len(
            "UMA MoLE SO2 m expert weight",
            config.experts * config.so2.m_conv_weight_len(order),
            m_expert_weights[order - 1].len(),
        )?;
    }
    check_len(
        "UMA MoLE SO2 coefficients",
        config.systems * config.experts,
        coefficients.len(),
    )?;
    check_len(
        "UMA MoLE SO2 output",
        rows * config.so2.output_node_stride(),
        out_l_major.len(),
    )?;
    if let Some(extra) = extra_m0_out.as_ref() {
        check_len(
            "UMA MoLE SO2 extra_m0_out",
            rows * config.so2.extra_m0_output_channels(),
            extra.len(),
        )?;
    } else if config.so2.extra_m0_output_channels() > 0 {
        return Err(invalid_mole(
            "UMA MoLE SO2 convolution requires extra_m0_out for configured extra channels",
        ));
    }

    if config.systems == 1 {
        let coeff_row = &coefficients[..config.experts];
        let mixed_m0 = mix_expert_weights(
            config.experts,
            config.so2.m0_weight_len(),
            m0_expert_weights,
            coeff_row,
        )?;
        let mut mixed_m_weights = Vec::with_capacity(config.so2.m_max());
        for order in 1..=config.so2.m_max() {
            mixed_m_weights.push(mix_expert_weights(
                config.experts,
                config.so2.m_conv_weight_len(order),
                m_expert_weights[order - 1],
                coeff_row,
            )?);
        }
        let mixed_m_weight_refs = mixed_m_weights
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();
        return e3nn_core::so2::fairchem_esen_so2_convolution_forward(
            config.so2,
            input_l_major,
            radial,
            &mixed_m0,
            m0_bias,
            &mixed_m_weight_refs,
            out_l_major,
            extra_m0_out,
        );
    }

    let mapping = EsenCoefficientMappingConfig::new(config.so2.l_max(), config.so2.m_max())
        .map_err(|err| invalid_mole(err.to_string()))?;
    let mut input_m_major = vec![0.0; input_l_major.len()];
    let mut out_m_major = vec![0.0; out_l_major.len()];
    fairchem_esen_l_to_m_forward(
        mapping,
        config.so2.in_channels(),
        input_l_major,
        &mut input_m_major,
    )?;

    run_mole_so2_convolution_forward_m_major(
        config,
        rows,
        &input_m_major,
        radial,
        m0_expert_weights,
        m0_bias,
        m_expert_weights,
        coefficients,
        system_sizes,
        &mut out_m_major,
        extra_m0_out,
    )?;
    fairchem_esen_m_to_l_forward(
        mapping,
        config.so2.out_channels(),
        &out_m_major,
        out_l_major,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn fairchem_uma_mole_so2_convolution_vjp(
    config: UmaMoleSo2ConvolutionConfig,
    input_l_major: &[f64],
    radial: Option<&[f64]>,
    m0_expert_weights: &[f64],
    m_expert_weights: &[&[f64]],
    coefficients: &[f64],
    system_sizes: &[usize],
    grad_out_l_major: &[f64],
    grad_extra_m0: Option<&[f64]>,
    grad_input_l_major: &mut [f64],
    grad_radial: Option<&mut [f64]>,
) -> Result<()> {
    validate_mole_so2_config(config)?;
    if config.systems != 1 {
        return Err(invalid_mole(
            "UMA MoLE SO2 VJP currently supports single-system inference only",
        ));
    }
    let rows = check_system_sizes(config.systems, system_sizes)?;
    check_len(
        "UMA MoLE SO2 VJP input",
        rows * config.so2.input_node_stride(),
        input_l_major.len(),
    )?;
    check_len(
        "UMA MoLE SO2 VJP grad_input",
        input_l_major.len(),
        grad_input_l_major.len(),
    )?;
    check_len(
        "UMA MoLE SO2 VJP m0_expert_weights",
        config.experts * config.so2.m0_weight_len(),
        m0_expert_weights.len(),
    )?;
    check_len(
        "UMA MoLE SO2 VJP m_expert_weights",
        config.so2.m_max(),
        m_expert_weights.len(),
    )?;
    for order in 1..=config.so2.m_max() {
        check_len(
            "UMA MoLE SO2 VJP m expert weight",
            config.experts * config.so2.m_conv_weight_len(order),
            m_expert_weights[order - 1].len(),
        )?;
    }
    check_len(
        "UMA MoLE SO2 VJP coefficients",
        config.experts,
        coefficients.len(),
    )?;

    let coeff_row = &coefficients[..config.experts];
    let mixed_m0 = mix_expert_weights(
        config.experts,
        config.so2.m0_weight_len(),
        m0_expert_weights,
        coeff_row,
    )?;
    let mut mixed_m_weights = Vec::with_capacity(config.so2.m_max());
    for order in 1..=config.so2.m_max() {
        mixed_m_weights.push(mix_expert_weights(
            config.experts,
            config.so2.m_conv_weight_len(order),
            m_expert_weights[order - 1],
            coeff_row,
        )?);
    }
    let mixed_m_weight_refs = mixed_m_weights
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    e3nn_core::so2::fairchem_esen_so2_convolution_vjp(
        config.so2,
        input_l_major,
        radial,
        &mixed_m0,
        &mixed_m_weight_refs,
        grad_out_l_major,
        grad_extra_m0,
        grad_input_l_major,
        grad_radial,
    )
}

fn validate_mole_config(config: UmaMoleLinearConfig) -> Result<()> {
    if config.systems == 0 || config.experts == 0 || config.in_dim == 0 || config.out_dim == 0 {
        return Err(invalid_mole(format!(
            "UMA MoLE linear dimensions must be nonzero, got systems={} experts={} in_dim={} out_dim={}",
            config.systems, config.experts, config.in_dim, config.out_dim
        )));
    }
    Ok(())
}

fn validate_mole_so2_config(config: UmaMoleSo2ConvolutionConfig) -> Result<()> {
    if config.systems == 0 || config.experts == 0 {
        return Err(invalid_mole(format!(
            "UMA MoLE SO2 dimensions must be nonzero, got systems={} experts={}",
            config.systems, config.experts
        )));
    }
    Ok(())
}

fn validate_routing_mlp_config(config: UmaRoutingMlpConfig) -> Result<()> {
    if config.rows == 0 || config.input_dim == 0 || config.hidden_dim == 0 || config.experts == 0 {
        return Err(invalid_mole(format!(
            "UMA routing MLP dimensions must be nonzero, got rows={} input_dim={} hidden_dim={} experts={}",
            config.rows, config.input_dim, config.hidden_dim, config.experts
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_mole_so2_convolution_forward_m_major(
    config: UmaMoleSo2ConvolutionConfig,
    rows: usize,
    input_m_major: &[f64],
    radial: Option<&[f64]>,
    m0_expert_weights: &[f64],
    m0_bias: &[f64],
    m_expert_weights: &[&[f64]],
    coefficients: &[f64],
    system_sizes: &[usize],
    out_m_major: &mut [f64],
    mut extra_m0_out: Option<&mut [f64]>,
) -> Result<()> {
    let mut coeff_offset = 0usize;
    let mut radial_offset = 0usize;
    for order in 0..=config.so2.m_max() {
        let copies = if order == 0 { 1 } else { 2 };
        let m_size = config.so2.m_size(order).ok_or_else(|| {
            invalid_mole(format!(
                "UMA MoLE SO2 order {order} exceeds l_max {}",
                config.so2.l_max()
            ))
        })?;
        let input_feature_dim = m_size * config.so2.in_channels();
        let output_feature_dim = m_size * config.so2.out_channels();
        let mut block_input = vec![0.0; rows * copies * input_feature_dim];
        let mut block_out = vec![0.0; rows * copies * output_feature_dim];
        gather_m_major_block(
            rows,
            config.so2.coefficient_count(),
            config.so2.in_channels(),
            coeff_offset,
            copies * m_size,
            input_m_major,
            &mut block_input,
        );
        let block_radial = radial.map(|radial_values| {
            gather_radial_block(
                rows,
                config.so2.radial_node_stride(),
                radial_offset,
                input_feature_dim,
                radial_values,
            )
        });
        if order == 0 {
            let mut modulated = block_input;
            if let Some(block_radial_values) = block_radial.as_deref() {
                let radial_config = EsenSo2RadialModulationConfig::new(input_feature_dim, 1)?;
                let mut modulated_with_radial = vec![0.0; modulated.len()];
                fairchem_esen_so2_radial_modulation_forward(
                    radial_config,
                    &modulated,
                    block_radial_values,
                    &mut modulated_with_radial,
                )?;
                modulated = modulated_with_radial;
            }
            let mut dense_out = vec![0.0; rows * config.so2.m0_output_feature_dim()];
            fairchem_uma_mole_linear_forward(
                UmaMoleLinearConfig {
                    systems: config.systems,
                    experts: config.experts,
                    in_dim: input_feature_dim,
                    out_dim: config.so2.m0_output_feature_dim(),
                },
                &modulated,
                m0_expert_weights,
                m0_bias,
                coefficients,
                system_sizes,
                &mut dense_out,
            )?;
            for row in 0..rows {
                let out_start = row * config.so2.m0_output_feature_dim();
                if config.so2.extra_m0_output_channels() > 0 {
                    let Some(extra_values) = extra_m0_out.as_mut() else {
                        return Err(invalid_mole("validated UMA extra m0 output is missing"));
                    };
                    let extra_start = row * config.so2.extra_m0_output_channels();
                    extra_values[extra_start..extra_start + config.so2.extra_m0_output_channels()]
                        .copy_from_slice(
                            &dense_out
                                [out_start..out_start + config.so2.extra_m0_output_channels()],
                        );
                }
                let main_start = out_start + config.so2.extra_m0_output_channels();
                let block_start = row * output_feature_dim;
                block_out[block_start..block_start + output_feature_dim]
                    .copy_from_slice(&dense_out[main_start..main_start + output_feature_dim]);
            }
        } else {
            let mut modulated = block_input;
            if let Some(block_radial_values) = block_radial.as_deref() {
                let radial_config = EsenSo2RadialModulationConfig::new(input_feature_dim, 2)?;
                let mut modulated_with_radial = vec![0.0; modulated.len()];
                fairchem_esen_so2_radial_modulation_forward(
                    radial_config,
                    &modulated,
                    block_radial_values,
                    &mut modulated_with_radial,
                )?;
                modulated = modulated_with_radial;
            }
            mole_so2_m_conv_forward(
                config.systems,
                config.experts,
                input_feature_dim,
                output_feature_dim,
                &modulated,
                m_expert_weights[order - 1],
                coefficients,
                system_sizes,
                &mut block_out,
            )?;
        }
        scatter_m_major_block(
            rows,
            config.so2.coefficient_count(),
            config.so2.out_channels(),
            coeff_offset,
            copies * m_size,
            &block_out,
            out_m_major,
        );
        coeff_offset += copies * m_size;
        radial_offset += input_feature_dim;
    }
    Ok(())
}

fn mix_expert_weights(
    experts: usize,
    per_expert_len: usize,
    expert_weights: &[f64],
    coefficients: &[f64],
) -> Result<Vec<f64>> {
    check_len(
        "UMA MoLE mix expert weights",
        experts * per_expert_len,
        expert_weights.len(),
    )?;
    check_len("UMA MoLE mix coefficients", experts, coefficients.len())?;
    let mut mixed = vec![0.0; per_expert_len];
    for (expert, coefficient) in coefficients.iter().copied().enumerate() {
        let start = expert * per_expert_len;
        for index in 0..per_expert_len {
            mixed[index] += coefficient * expert_weights[start + index];
        }
    }
    Ok(mixed)
}

#[allow(clippy::too_many_arguments)]
fn mole_so2_m_conv_forward(
    systems: usize,
    experts: usize,
    input_len: usize,
    output_len: usize,
    input: &[f64],
    expert_weights: &[f64],
    coefficients: &[f64],
    system_sizes: &[usize],
    out: &mut [f64],
) -> Result<()> {
    let rows = check_system_sizes(systems, system_sizes)?;
    check_len(
        "UMA MoLE SO2 m-conv input",
        rows * 2 * input_len,
        input.len(),
    )?;
    check_len(
        "UMA MoLE SO2 m-conv expert_weights",
        experts * 2 * output_len * input_len,
        expert_weights.len(),
    )?;
    check_len(
        "UMA MoLE SO2 m-conv coefficients",
        systems * experts,
        coefficients.len(),
    )?;
    check_len("UMA MoLE SO2 m-conv out", rows * 2 * output_len, out.len())?;
    out.fill(0.0);

    let mut row_offset = 0usize;
    for system in 0..systems {
        let coeff_row = &coefficients[system * experts..(system + 1) * experts];
        for local_row in 0..system_sizes[system] {
            let row = row_offset + local_row;
            let input_start = row * 2 * input_len;
            let real_input = &input[input_start..input_start + input_len];
            let imag_input = &input[input_start + input_len..input_start + 2 * input_len];
            let out_start = row * 2 * output_len;
            for (expert, coefficient) in coeff_row.iter().copied().enumerate() {
                let expert_start = expert * 2 * output_len * input_len;
                for out_index in 0..output_len {
                    let direct_weight_start = expert_start + out_index * input_len;
                    let mixed_weight_start = expert_start + (output_len + out_index) * input_len;
                    let mut real_value = 0.0;
                    let mut imag_value = 0.0;
                    for in_index in 0..input_len {
                        let direct_weight = expert_weights[direct_weight_start + in_index];
                        let mixed_weight = expert_weights[mixed_weight_start + in_index];
                        real_value += direct_weight * real_input[in_index]
                            - mixed_weight * imag_input[in_index];
                        imag_value += direct_weight * imag_input[in_index]
                            + mixed_weight * real_input[in_index];
                    }
                    out[out_start + out_index] += coefficient * real_value;
                    out[out_start + output_len + out_index] += coefficient * imag_value;
                }
            }
        }
        row_offset += system_sizes[system];
    }
    Ok(())
}

fn gather_m_major_block(
    rows: usize,
    coefficient_count: usize,
    channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    source: &[f64],
    target: &mut [f64],
) {
    let source_row_stride = coefficient_count * channels;
    let target_row_stride = block_coefficients * channels;
    for row in 0..rows {
        let source_start = row * source_row_stride + coefficient_offset * channels;
        let target_start = row * target_row_stride;
        target[target_start..target_start + target_row_stride]
            .copy_from_slice(&source[source_start..source_start + target_row_stride]);
    }
}

fn scatter_m_major_block(
    rows: usize,
    coefficient_count: usize,
    channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    source: &[f64],
    target: &mut [f64],
) {
    let target_row_stride = coefficient_count * channels;
    let source_row_stride = block_coefficients * channels;
    for row in 0..rows {
        let source_start = row * source_row_stride;
        let target_start = row * target_row_stride + coefficient_offset * channels;
        target[target_start..target_start + source_row_stride]
            .copy_from_slice(&source[source_start..source_start + source_row_stride]);
    }
}

fn gather_radial_block(
    rows: usize,
    radial_row_stride: usize,
    radial_offset: usize,
    feature_dim: usize,
    source: &[f64],
) -> Vec<f64> {
    let mut target = vec![0.0; rows * feature_dim];
    for row in 0..rows {
        let source_start = row * radial_row_stride + radial_offset;
        let target_start = row * feature_dim;
        target[target_start..target_start + feature_dim]
            .copy_from_slice(&source[source_start..source_start + feature_dim]);
    }
    target
}

fn dense_silu_rows(
    input: &[f64],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    weight: &[f64],
    bias: &[f64],
    out: &mut [f64],
) {
    for row in 0..rows {
        let input_row = &input[row * in_dim..(row + 1) * in_dim];
        let out_row = &mut out[row * out_dim..(row + 1) * out_dim];
        for out_index in 0..out_dim {
            let weight_row = &weight[out_index * in_dim..(out_index + 1) * in_dim];
            let mut value = bias[out_index];
            for in_index in 0..in_dim {
                value += weight_row[in_index] * input_row[in_index];
            }
            out_row[out_index] = silu(value);
        }
    }
}

fn silu(value: f64) -> f64 {
    value / (1.0 + (-value).exp())
}

fn check_system_sizes(systems: usize, system_sizes: &[usize]) -> Result<usize> {
    check_len("UMA MoLE system_sizes", systems, system_sizes.len())?;
    let mut rows = 0usize;
    for size in system_sizes {
        rows = rows
            .checked_add(*size)
            .ok_or_else(|| invalid_mole("UMA MoLE system_sizes overflow"))?;
    }
    if rows == 0 {
        return Err(invalid_mole("UMA MoLE requires at least one input row"));
    }
    Ok(rows)
}

fn check_len(name: &str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(invalid_mole(format!(
            "{name} length mismatch: expected {expected}, got {actual}"
        )))
    }
}

fn invalid_mole(message: impl Into<String>) -> CoreError {
    CoreError::InvalidUmaMole(message.into())
}

fn mix_expert_weights_f32(
    experts: usize,
    per_expert_len: usize,
    expert_weights: &[f32],
    coefficients: &[f32],
) -> Result<Vec<f32>> {
    check_len(
        "UMA MoLE mix expert weights f32",
        experts * per_expert_len,
        expert_weights.len(),
    )?;
    check_len("UMA MoLE mix coefficients f32", experts, coefficients.len())?;
    let mut mixed = vec![0.0f32; per_expert_len];
    for (expert, coefficient) in coefficients.iter().copied().enumerate() {
        let start = expert * per_expert_len;
        for index in 0..per_expert_len {
            mixed[index] += coefficient * expert_weights[start + index];
        }
    }
    Ok(mixed)
}

fn l_to_m_permutation_apply_f32(
    permutation: &[usize],
    channels: usize,
    input: &[f32],
    out: &mut [f32],
) {
    let node_stride = permutation.len() * channels;
    for node_start in (0..input.len()).step_by(node_stride) {
        for (out_index, &src_index) in permutation.iter().enumerate() {
            let src_start = node_start + src_index * channels;
            let dst_start = node_start + out_index * channels;
            out[dst_start..dst_start + channels]
                .copy_from_slice(&input[src_start..src_start + channels]);
        }
    }
}

fn m_to_l_permutation_apply_f32(
    permutation: &[usize],
    channels: usize,
    m_major: &[f32],
    out: &mut [f32],
) {
    let node_stride = permutation.len() * channels;
    for node_start in (0..m_major.len()).step_by(node_stride) {
        for (src_index, &dst_index) in permutation.iter().enumerate() {
            let src_start = node_start + src_index * channels;
            let dst_start = node_start + dst_index * channels;
            out[dst_start..dst_start + channels]
                .copy_from_slice(&m_major[src_start..src_start + channels]);
        }
    }
}

fn gather_m_major_block_f32(
    rows: usize,
    coefficient_count: usize,
    channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    source: &[f32],
    target: &mut [f32],
) {
    let source_row_stride = coefficient_count * channels;
    let target_row_stride = block_coefficients * channels;
    for row in 0..rows {
        let source_start = row * source_row_stride + coefficient_offset * channels;
        let target_start = row * target_row_stride;
        target[target_start..target_start + target_row_stride]
            .copy_from_slice(&source[source_start..source_start + target_row_stride]);
    }
}

fn scatter_m_major_block_f32(
    rows: usize,
    coefficient_count: usize,
    channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    source: &[f32],
    target: &mut [f32],
) {
    let target_row_stride = coefficient_count * channels;
    let source_row_stride = block_coefficients * channels;
    for row in 0..rows {
        let source_start = row * source_row_stride;
        let target_start = row * target_row_stride + coefficient_offset * channels;
        target[target_start..target_start + source_row_stride]
            .copy_from_slice(&source[source_start..source_start + source_row_stride]);
    }
}

fn gather_radial_block_f32(
    rows: usize,
    radial_row_stride: usize,
    radial_offset: usize,
    feature_dim: usize,
    source: &[f32],
) -> Vec<f32> {
    let mut target = vec![0.0f32; rows * feature_dim];
    for row in 0..rows {
        let source_start = row * radial_row_stride + radial_offset;
        let target_start = row * feature_dim;
        target[target_start..target_start + feature_dim]
            .copy_from_slice(&source[source_start..source_start + feature_dim]);
    }
    target
}

fn radial_modulate_f32(
    rows: usize,
    copies: usize,
    feature_dim: usize,
    block_input: &[f32],
    radial: &[f32],
    out: &mut [f32],
) {
    for row in 0..rows {
        for copy in 0..copies {
            let base = row * copies * feature_dim + copy * feature_dim;
            let rad_start = row * feature_dim;
            for f in 0..feature_dim {
                out[base + f] = block_input[base + f] * radial[rad_start + f];
            }
        }
    }
}

fn dense_forward_with_bias_f32(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    weight: &[f32],
    bias: &[f32],
    input: &[f32],
    out: &mut [f32],
) {
    // weight is output-major [out_dim, in_dim]
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    {
        crate::blas_backend::dense_forward_output_major_weight_f32(
            input, rows, in_dim, weight, out, out_dim, 1.0, 0.0,
        );
        for row in 0..rows {
            let row_start = row * out_dim;
            for (j, &b) in bias.iter().enumerate() {
                out[row_start + j] += b;
            }
        }
    }
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    for row in 0..rows {
        let in_start = row * in_dim;
        let out_start = row * out_dim;
        for out_index in 0..out_dim {
            let weight_row_start = out_index * in_dim;
            let mut value = bias[out_index];
            for in_index in 0..in_dim {
                value += weight[weight_row_start + in_index] * input[in_start + in_index];
            }
            out[out_start + out_index] = value;
        }
    }
}

fn mole_so2_m_conv_forward_single_system_f32(
    input_len: usize,
    output_len: usize,
    rows: usize,
    input: &[f32],
    mixed_weight: &[f32],
    out: &mut [f32],
) {
    // input: [rows, 2, input_len]  (real then imag)
    // mixed_weight: [2, output_len, input_len] = [direct; mixed] output-major for each half
    // out: [rows, 2, output_len]
    // real_out[row, j] = sum_i (W_direct[j,i]*real_in[row,i] - W_mixed[j,i]*imag_in[row,i])
    // imag_out[row, j] = sum_i (W_direct[j,i]*imag_in[row,i] + W_mixed[j,i]*real_in[row,i])

    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    {
        // Single fused SGEMM: out = input @ [[W0^T, W1^T], [-W1^T, W0^T]].
        // input is [rows, 2*input_len] and out is [rows, 2*output_len],
        // both in block-interleaved (real then imag) layout — no de/re-interleave needed.
        crate::blas_backend::so2_mconv_forward_set_fused_f32(
            rows,
            input_len,
            output_len,
            mixed_weight,
            input,
            out,
        );
    }

    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    {
        let w_direct = &mixed_weight[..output_len * input_len];
        let w_mixed = &mixed_weight[output_len * input_len..2 * output_len * input_len];
        for row in 0..rows {
            let in_start = row * 2 * input_len;
            let real_input = &input[in_start..in_start + input_len];
            let imag_input = &input[in_start + input_len..in_start + 2 * input_len];
            let out_start = row * 2 * output_len;
            for out_index in 0..output_len {
                let weight_row_start = out_index * input_len;
                let mut real_value = 0.0f32;
                let mut imag_value = 0.0f32;
                for in_index in 0..input_len {
                    let dw = w_direct[weight_row_start + in_index];
                    let mw = w_mixed[weight_row_start + in_index];
                    real_value += dw * real_input[in_index] - mw * imag_input[in_index];
                    imag_value += dw * imag_input[in_index] + mw * real_input[in_index];
                }
                out[out_start + out_index] = real_value;
                out[out_start + output_len + out_index] = imag_value;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_single_system_so2_conv_forward_f32(
    config: UmaMoleSo2ConvolutionConfig,
    rows: usize,
    input_l_major: &[f32],
    radial: Option<&[f32]>,
    mixed_m0: &[f32],
    m0_bias: &[f32],
    mixed_m_weights: &[Vec<f32>],
    out_l_major: &mut [f32],
    extra_m0_out: Option<&mut [f32]>,
) -> Result<()> {
    let mapping = EsenCoefficientMappingConfig::new(config.so2.l_max(), config.so2.m_max())
        .map_err(|err| invalid_mole(err.to_string()))?;
    let permutation = e3nn_core::so3::fairchem_esen_l_to_m_permutation(mapping);

    let mut input_m_major = vec![0.0f32; input_l_major.len()];
    l_to_m_permutation_apply_f32(
        &permutation,
        config.so2.in_channels(),
        input_l_major,
        &mut input_m_major,
    );

    let mut out_m_major = vec![0.0f32; out_l_major.len()];

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && rows >= 512 && config.so2.m_max() > 0 {
        run_single_system_so2_conv_forward_f32_parallel(
            config,
            rows,
            &input_m_major,
            radial,
            mixed_m0,
            m0_bias,
            mixed_m_weights,
            &mut out_m_major,
            extra_m0_out,
        )?;
        m_to_l_permutation_apply_f32(
            &permutation,
            config.so2.out_channels(),
            &out_m_major,
            out_l_major,
        );
        return Ok(());
    }

    let mut coeff_offset = 0usize;
    let mut radial_offset = 0usize;
    let mut extra_m0_out = extra_m0_out;

    for order in 0..=config.so2.m_max() {
        let copies = if order == 0 { 1 } else { 2 };
        let m_size = config.so2.m_size(order).ok_or_else(|| {
            invalid_mole(format!(
                "UMA MoLE SO2 f32 order {order} exceeds l_max {}",
                config.so2.l_max()
            ))
        })?;
        let input_feature_dim = m_size * config.so2.in_channels();
        let output_feature_dim = m_size * config.so2.out_channels();
        let block_rows = copies * m_size;

        let mut block_input = vec![0.0f32; rows * block_rows * config.so2.in_channels()];
        gather_m_major_block_f32(
            rows,
            config.so2.coefficient_count(),
            config.so2.in_channels(),
            coeff_offset,
            block_rows,
            &input_m_major,
            &mut block_input,
        );

        let modulated_input: Vec<f32> = if let Some(radial_values) = radial {
            let block_radial = gather_radial_block_f32(
                rows,
                config.so2.radial_node_stride(),
                radial_offset,
                input_feature_dim,
                radial_values,
            );
            let mut modulated = vec![0.0f32; block_input.len()];
            radial_modulate_f32(
                rows,
                copies,
                input_feature_dim,
                &block_input,
                &block_radial,
                &mut modulated,
            );
            modulated
        } else {
            block_input
        };

        let mut block_out = vec![0.0f32; rows * block_rows * config.so2.out_channels()];
        if order == 0 {
            let mut dense_out = vec![0.0f32; rows * config.so2.m0_output_feature_dim()];
            dense_forward_with_bias_f32(
                rows,
                input_feature_dim,
                config.so2.m0_output_feature_dim(),
                mixed_m0,
                m0_bias,
                &modulated_input,
                &mut dense_out,
            );
            for row in 0..rows {
                let out_start = row * config.so2.m0_output_feature_dim();
                if config.so2.extra_m0_output_channels() > 0 {
                    let extra_values = extra_m0_out
                        .as_mut()
                        .ok_or_else(|| invalid_mole("validated UMA extra m0 output is missing"))?;
                    let extra_start = row * config.so2.extra_m0_output_channels();
                    extra_values[extra_start..extra_start + config.so2.extra_m0_output_channels()]
                        .copy_from_slice(
                            &dense_out
                                [out_start..out_start + config.so2.extra_m0_output_channels()],
                        );
                }
                let main_src_start = out_start + config.so2.extra_m0_output_channels();
                let block_start = row * output_feature_dim;
                block_out[block_start..block_start + output_feature_dim].copy_from_slice(
                    &dense_out[main_src_start..main_src_start + output_feature_dim],
                );
            }
        } else {
            mole_so2_m_conv_forward_single_system_f32(
                input_feature_dim,
                output_feature_dim,
                rows,
                &modulated_input,
                &mixed_m_weights[order - 1],
                &mut block_out,
            );
        }
        scatter_m_major_block_f32(
            rows,
            config.so2.coefficient_count(),
            config.so2.out_channels(),
            coeff_offset,
            block_rows,
            &block_out,
            &mut out_m_major,
        );
        coeff_offset += block_rows;
        radial_offset += input_feature_dim;
    }

    m_to_l_permutation_apply_f32(
        &permutation,
        config.so2.out_channels(),
        &out_m_major,
        out_l_major,
    );
    Ok(())
}

/// Result of one m-order pass in the parallel f32 SO2 forward.
#[cfg(feature = "parallel")]
struct So2ConvF32ForwardOrderResult {
    coeff_offset: usize,
    block_rows: usize,
    block_out: Vec<f32>,
    /// Non-None only for order == 0 when extra_m0_output_channels > 0.
    extra_block: Option<Vec<f32>>,
}

/// Result of one m-order pass in the parallel f32 SO2 VJP.
#[cfg(feature = "parallel")]
struct So2ConvF32VjpOrderResult {
    coeff_offset: usize,
    radial_offset: usize,
    block_rows: usize,
    input_feature_dim: usize,
    /// Gradient w.r.t. the m-major input block (accumulate into grad_input_m_major).
    grad_input_block: Vec<f32>,
    /// Gradient w.r.t. the per-order radial block (scatter into grad_radial).
    /// Present only when `radial` is Some.
    grad_radial_block: Option<Vec<f32>>,
}

/// Parallel m-order forward for `run_single_system_so2_conv_forward_f32`.
/// Enabled only when the `parallel` feature is active and `rows >= 512`.
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn run_single_system_so2_conv_forward_f32_parallel(
    config: UmaMoleSo2ConvolutionConfig,
    rows: usize,
    input_m_major: &[f32],
    radial: Option<&[f32]>,
    mixed_m0: &[f32],
    m0_bias: &[f32],
    mixed_m_weights: &[Vec<f32>],
    out_m_major: &mut [f32],
    mut extra_m0_out: Option<&mut [f32]>,
) -> Result<()> {
    // Pre-compute per-order offsets so each rayon task can index independently.
    let mut coeff_offsets = Vec::with_capacity(config.so2.m_max() + 1);
    let mut radial_offsets = Vec::with_capacity(config.so2.m_max() + 1);
    let mut coeff_offset = 0usize;
    let mut radial_offset = 0usize;
    for order in 0..=config.so2.m_max() {
        coeff_offsets.push(coeff_offset);
        radial_offsets.push(radial_offset);
        let copies = if order == 0 { 1 } else { 2 };
        let m_size = config.so2.m_size(order).ok_or_else(|| {
            invalid_mole(format!(
                "UMA SO2 f32 parallel fwd: order {order} > l_max {}",
                config.so2.l_max()
            ))
        })?;
        coeff_offset += copies * m_size;
        radial_offset += m_size * config.so2.in_channels();
    }

    let results = (0..=config.so2.m_max())
        .into_par_iter()
        .map(|order| -> Result<So2ConvF32ForwardOrderResult> {
            let copies = if order == 0 { 1 } else { 2 };
            let m_size = config.so2.m_size(order).ok_or_else(|| {
                invalid_mole(format!(
                    "UMA SO2 f32 parallel fwd: order {order} > l_max {}",
                    config.so2.l_max()
                ))
            })?;
            let input_feature_dim = m_size * config.so2.in_channels();
            let output_feature_dim = m_size * config.so2.out_channels();
            let block_rows = copies * m_size;

            let mut block_input = vec![0.0f32; rows * block_rows * config.so2.in_channels()];
            gather_m_major_block_f32(
                rows,
                config.so2.coefficient_count(),
                config.so2.in_channels(),
                coeff_offsets[order],
                block_rows,
                input_m_major,
                &mut block_input,
            );

            let modulated_input: Vec<f32> = if let Some(radial_values) = radial {
                let block_radial = gather_radial_block_f32(
                    rows,
                    config.so2.radial_node_stride(),
                    radial_offsets[order],
                    input_feature_dim,
                    radial_values,
                );
                let mut modulated = vec![0.0f32; block_input.len()];
                radial_modulate_f32(
                    rows,
                    copies,
                    input_feature_dim,
                    &block_input,
                    &block_radial,
                    &mut modulated,
                );
                modulated
            } else {
                block_input
            };

            let mut block_out = vec![0.0f32; rows * block_rows * config.so2.out_channels()];
            let mut extra_block = None;
            if order == 0 {
                let mut dense_out = vec![0.0f32; rows * config.so2.m0_output_feature_dim()];
                dense_forward_with_bias_f32(
                    rows,
                    input_feature_dim,
                    config.so2.m0_output_feature_dim(),
                    mixed_m0,
                    m0_bias,
                    &modulated_input,
                    &mut dense_out,
                );
                if config.so2.extra_m0_output_channels() > 0 {
                    let mut extra = vec![0.0f32; rows * config.so2.extra_m0_output_channels()];
                    for row in 0..rows {
                        let out_start = row * config.so2.m0_output_feature_dim();
                        let extra_start = row * config.so2.extra_m0_output_channels();
                        extra[extra_start..extra_start + config.so2.extra_m0_output_channels()]
                            .copy_from_slice(
                                &dense_out
                                    [out_start..out_start + config.so2.extra_m0_output_channels()],
                            );
                    }
                    extra_block = Some(extra);
                }
                for row in 0..rows {
                    let out_start = row * config.so2.m0_output_feature_dim();
                    let main_src_start = out_start + config.so2.extra_m0_output_channels();
                    let block_start = row * output_feature_dim;
                    block_out[block_start..block_start + output_feature_dim].copy_from_slice(
                        &dense_out[main_src_start..main_src_start + output_feature_dim],
                    );
                }
            } else {
                mole_so2_m_conv_forward_single_system_f32(
                    input_feature_dim,
                    output_feature_dim,
                    rows,
                    &modulated_input,
                    &mixed_m_weights[order - 1],
                    &mut block_out,
                );
            }
            Ok(So2ConvF32ForwardOrderResult {
                coeff_offset: coeff_offsets[order],
                block_rows,
                block_out,
                extra_block,
            })
        })
        .collect::<Vec<_>>();

    // Sequential scatter phase — no data races since each order targets distinct regions.
    for result in results {
        let result = result?;
        scatter_m_major_block_f32(
            rows,
            config.so2.coefficient_count(),
            config.so2.out_channels(),
            result.coeff_offset,
            result.block_rows,
            &result.block_out,
            out_m_major,
        );
        if let (Some(extra), Some(extra_dst)) = (result.extra_block, extra_m0_out.as_deref_mut()) {
            extra_dst.copy_from_slice(&extra);
        }
    }
    Ok(())
}

/// Parallel m-order VJP for `run_single_system_so2_conv_vjp_f32`.
/// Enabled only when the `parallel` feature is active and `rows >= 512`.
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn run_single_system_so2_conv_vjp_f32_parallel(
    config: UmaMoleSo2ConvolutionConfig,
    rows: usize,
    input_m_major: &[f32],
    radial: Option<&[f32]>,
    mixed_m0: &[f32],
    m0_output_feature_dim: usize,
    mixed_m_weights: &[Vec<f32>],
    grad_out_m_major: &[f32],
    extra_m0_grad_out: Option<&[f32]>,
    grad_input_m_major: &mut [f32],
    mut grad_radial: Option<&mut [f32]>,
) -> Result<()> {
    // Pre-compute per-order offsets.
    let mut coeff_offsets = Vec::with_capacity(config.so2.m_max() + 1);
    let mut radial_offsets = Vec::with_capacity(config.so2.m_max() + 1);
    let mut coeff_offset = 0usize;
    let mut radial_offset = 0usize;
    for order in 0..=config.so2.m_max() {
        coeff_offsets.push(coeff_offset);
        radial_offsets.push(radial_offset);
        let copies = if order == 0 { 1 } else { 2 };
        let m_size = config.so2.m_size(order).ok_or_else(|| {
            invalid_mole(format!(
                "UMA SO2 f32 parallel vjp: order {order} > l_max {}",
                config.so2.l_max()
            ))
        })?;
        coeff_offset += copies * m_size;
        radial_offset += m_size * config.so2.in_channels();
    }

    let results = (0..=config.so2.m_max())
        .into_par_iter()
        .map(|order| -> Result<So2ConvF32VjpOrderResult> {
            let copies = if order == 0 { 1 } else { 2 };
            let m_size = config.so2.m_size(order).ok_or_else(|| {
                invalid_mole(format!(
                    "UMA SO2 f32 parallel vjp: order {order} > l_max {}",
                    config.so2.l_max()
                ))
            })?;
            let input_feature_dim = m_size * config.so2.in_channels();
            let output_feature_dim = m_size * config.so2.out_channels();
            let block_rows = copies * m_size;

            // Gather grad_out block.
            let mut block_grad_out = vec![0.0f32; rows * block_rows * config.so2.out_channels()];
            if order == 0 {
                gather_m_major_block_f32(
                    rows,
                    config.so2.coefficient_count(),
                    config.so2.out_channels(),
                    coeff_offsets[order],
                    block_rows,
                    grad_out_m_major,
                    &mut block_grad_out,
                );
                // Reconstruct full m0 grad including extra channels.
                let mut full_grad_out = vec![0.0f32; rows * m0_output_feature_dim];
                for row in 0..rows {
                    let full_start = row * m0_output_feature_dim;
                    if config.so2.extra_m0_output_channels() > 0 {
                        if let Some(extra) = extra_m0_grad_out {
                            let extra_start = row * config.so2.extra_m0_output_channels();
                            full_grad_out
                                [full_start..full_start + config.so2.extra_m0_output_channels()]
                                .copy_from_slice(
                                    &extra[extra_start
                                        ..extra_start + config.so2.extra_m0_output_channels()],
                                );
                        }
                    }
                    let main_start = full_start + config.so2.extra_m0_output_channels();
                    let block_start = row * output_feature_dim;
                    full_grad_out[main_start..main_start + output_feature_dim].copy_from_slice(
                        &block_grad_out[block_start..block_start + output_feature_dim],
                    );
                }
                // VJP of dense (no bias): grad_modulated = grad_out * W.
                let mut grad_modulated = vec![0.0f32; rows * input_feature_dim];
                #[cfg(any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        any(target_arch = "x86_64", target_arch = "aarch64")
                    )
                ))]
                {
                    crate::blas_backend::dense_vjp_input_output_major_weight_f32(
                        rows,
                        input_feature_dim,
                        m0_output_feature_dim,
                        mixed_m0,
                        &full_grad_out,
                        &mut grad_modulated,
                        1.0,
                        0.0,
                    );
                }
                #[cfg(not(any(
                    all(feature = "accelerate-blas", target_os = "macos"),
                    all(
                        any(feature = "intel-mkl", feature = "openblas"),
                        target_os = "linux",
                        any(target_arch = "x86_64", target_arch = "aarch64")
                    )
                )))]
                for row in 0..rows {
                    let out_start = row * m0_output_feature_dim;
                    let in_start = row * input_feature_dim;
                    for in_i in 0..input_feature_dim {
                        let mut acc = 0.0f32;
                        for out_j in 0..m0_output_feature_dim {
                            acc += mixed_m0[out_j * input_feature_dim + in_i]
                                * full_grad_out[out_start + out_j];
                        }
                        grad_modulated[in_start + in_i] += acc;
                    }
                }

                // Propagate through radial modulation if present.
                if let Some(radial_values) = radial {
                    let block_radial = gather_radial_block_f32(
                        rows,
                        config.so2.radial_node_stride(),
                        radial_offsets[order],
                        input_feature_dim,
                        radial_values,
                    );
                    let block_input = gather_m_major_block_f32_alloc(
                        rows,
                        config.so2.coefficient_count(),
                        config.so2.in_channels(),
                        coeff_offsets[order],
                        block_rows,
                        input_m_major,
                    );
                    let mut grad_block_input = vec![0.0f32; rows * input_feature_dim];
                    let radial_block_len = rows * input_feature_dim;
                    let mut grad_radial_block = vec![0.0f32; radial_block_len];
                    radial_modulate_vjp_f32(
                        rows,
                        copies,
                        input_feature_dim,
                        &block_input,
                        &block_radial,
                        &grad_modulated,
                        &mut grad_block_input,
                        &mut grad_radial_block,
                    );
                    return Ok(So2ConvF32VjpOrderResult {
                        coeff_offset: coeff_offsets[order],
                        radial_offset: radial_offsets[order],
                        block_rows,
                        input_feature_dim,
                        grad_input_block: grad_block_input,
                        grad_radial_block: Some(grad_radial_block),
                    });
                }
                return Ok(So2ConvF32VjpOrderResult {
                    coeff_offset: coeff_offsets[order],
                    radial_offset: radial_offsets[order],
                    block_rows,
                    input_feature_dim,
                    grad_input_block: grad_modulated,
                    grad_radial_block: None,
                });
            }

            // order > 0
            gather_m_major_block_f32(
                rows,
                config.so2.coefficient_count(),
                config.so2.out_channels(),
                coeff_offsets[order],
                block_rows,
                grad_out_m_major,
                &mut block_grad_out,
            );

            let mut grad_modulated = vec![0.0f32; rows * block_rows * config.so2.in_channels()];
            mole_so2_m_conv_vjp_input_f32(
                input_feature_dim,
                output_feature_dim,
                rows,
                &mixed_m_weights[order - 1],
                &block_grad_out,
                &mut grad_modulated,
            );

            if let Some(radial_values) = radial {
                let block_radial = gather_radial_block_f32(
                    rows,
                    config.so2.radial_node_stride(),
                    radial_offsets[order],
                    input_feature_dim,
                    radial_values,
                );
                let block_input = gather_m_major_block_f32_alloc(
                    rows,
                    config.so2.coefficient_count(),
                    config.so2.in_channels(),
                    coeff_offsets[order],
                    block_rows,
                    input_m_major,
                );
                let mut grad_block_input =
                    vec![0.0f32; rows * block_rows * config.so2.in_channels()];
                let radial_block_len = rows * input_feature_dim;
                let mut grad_radial_block = vec![0.0f32; radial_block_len];
                radial_modulate_vjp_f32(
                    rows,
                    copies,
                    input_feature_dim,
                    &block_input,
                    &block_radial,
                    &grad_modulated,
                    &mut grad_block_input,
                    &mut grad_radial_block,
                );
                return Ok(So2ConvF32VjpOrderResult {
                    coeff_offset: coeff_offsets[order],
                    radial_offset: radial_offsets[order],
                    block_rows,
                    input_feature_dim,
                    grad_input_block: grad_block_input,
                    grad_radial_block: Some(grad_radial_block),
                });
            }
            Ok(So2ConvF32VjpOrderResult {
                coeff_offset: coeff_offsets[order],
                radial_offset: radial_offsets[order],
                block_rows,
                input_feature_dim,
                grad_input_block: grad_modulated,
                grad_radial_block: None,
            })
        })
        .collect::<Vec<_>>();

    // Sequential scatter phase — accumulate per-order grads into shared buffers.
    for result in results {
        let result = result?;
        scatter_add_m_major_block_f32(
            rows,
            config.so2.coefficient_count(),
            config.so2.in_channels(),
            result.coeff_offset,
            result.block_rows,
            &result.grad_input_block,
            grad_input_m_major,
        );
        if let (Some(grad_radial_block), Some(grad_radial_values)) =
            (result.grad_radial_block, grad_radial.as_deref_mut())
        {
            for row in 0..rows {
                let src_start = row * result.input_feature_dim;
                let dst_start = row * config.so2.radial_node_stride() + result.radial_offset;
                for f in 0..result.input_feature_dim {
                    grad_radial_values[dst_start + f] += grad_radial_block[src_start + f];
                }
            }
        }
    }
    Ok(())
}

/// Parallel m-order forward for `run_single_system_so2_conv_forward_precomputed_f32`.
/// Enabled only when the BLAS feature and the `parallel` feature are both active
/// and `rows >= 512`.
#[cfg(all(
    feature = "parallel",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )
))]
#[allow(clippy::too_many_arguments)]
fn run_single_system_so2_conv_forward_precomputed_f32_parallel(
    config: UmaMoleSo2ConvolutionConfig,
    rows: usize,
    input_m_major: &[f32],
    radial: Option<&[f32]>,
    mixed_m0: &[f32],
    m0_bias: &[f32],
    fused_m_weights_fwd: &[&[f32]],
    out_m_major: &mut [f32],
    mut extra_m0_out: Option<&mut [f32]>,
) -> Result<()> {
    // Pre-compute per-order offsets.
    let mut coeff_offsets = Vec::with_capacity(config.so2.m_max() + 1);
    let mut radial_offsets = Vec::with_capacity(config.so2.m_max() + 1);
    let mut coeff_offset = 0usize;
    let mut radial_offset = 0usize;
    for order in 0..=config.so2.m_max() {
        coeff_offsets.push(coeff_offset);
        radial_offsets.push(radial_offset);
        let copies = if order == 0 { 1 } else { 2 };
        let m_size = config.so2.m_size(order).ok_or_else(|| {
            invalid_mole(format!(
                "UMA SO2 f32 parallel precomputed fwd: order {order} > l_max {}",
                config.so2.l_max()
            ))
        })?;
        coeff_offset += copies * m_size;
        radial_offset += m_size * config.so2.in_channels();
    }

    let results = (0..=config.so2.m_max())
        .into_par_iter()
        .map(|order| -> Result<So2ConvF32ForwardOrderResult> {
            let copies = if order == 0 { 1 } else { 2 };
            let m_size = config.so2.m_size(order).ok_or_else(|| {
                invalid_mole(format!(
                    "UMA SO2 f32 parallel precomputed fwd: order {order} > l_max {}",
                    config.so2.l_max()
                ))
            })?;
            let input_feature_dim = m_size * config.so2.in_channels();
            let output_feature_dim = m_size * config.so2.out_channels();
            let block_rows = copies * m_size;

            let mut block_input = vec![0.0f32; rows * block_rows * config.so2.in_channels()];
            gather_m_major_block_f32(
                rows,
                config.so2.coefficient_count(),
                config.so2.in_channels(),
                coeff_offsets[order],
                block_rows,
                input_m_major,
                &mut block_input,
            );

            let modulated_input: Vec<f32> = if let Some(radial_values) = radial {
                let block_radial = gather_radial_block_f32(
                    rows,
                    config.so2.radial_node_stride(),
                    radial_offsets[order],
                    input_feature_dim,
                    radial_values,
                );
                let mut modulated = vec![0.0f32; block_input.len()];
                radial_modulate_f32(
                    rows,
                    copies,
                    input_feature_dim,
                    &block_input,
                    &block_radial,
                    &mut modulated,
                );
                modulated
            } else {
                block_input
            };

            let mut block_out = vec![0.0f32; rows * block_rows * config.so2.out_channels()];
            let mut extra_block = None;
            if order == 0 {
                let mut dense_out = vec![0.0f32; rows * config.so2.m0_output_feature_dim()];
                dense_forward_with_bias_f32(
                    rows,
                    input_feature_dim,
                    config.so2.m0_output_feature_dim(),
                    mixed_m0,
                    m0_bias,
                    &modulated_input,
                    &mut dense_out,
                );
                if config.so2.extra_m0_output_channels() > 0 {
                    let mut extra = vec![0.0f32; rows * config.so2.extra_m0_output_channels()];
                    for row in 0..rows {
                        let out_start = row * config.so2.m0_output_feature_dim();
                        let extra_start = row * config.so2.extra_m0_output_channels();
                        extra[extra_start..extra_start + config.so2.extra_m0_output_channels()]
                            .copy_from_slice(
                                &dense_out
                                    [out_start..out_start + config.so2.extra_m0_output_channels()],
                            );
                    }
                    extra_block = Some(extra);
                }
                for row in 0..rows {
                    let out_start = row * config.so2.m0_output_feature_dim();
                    let main_src_start = out_start + config.so2.extra_m0_output_channels();
                    let block_start = row * output_feature_dim;
                    block_out[block_start..block_start + output_feature_dim].copy_from_slice(
                        &dense_out[main_src_start..main_src_start + output_feature_dim],
                    );
                }
            } else {
                let idx = order - 1;
                crate::blas_backend::so2_mconv_forward_set_precomputed_fused_f32(
                    rows,
                    input_feature_dim,
                    output_feature_dim,
                    fused_m_weights_fwd[idx],
                    &modulated_input,
                    &mut block_out,
                );
            }
            Ok(So2ConvF32ForwardOrderResult {
                coeff_offset: coeff_offsets[order],
                block_rows,
                block_out,
                extra_block,
            })
        })
        .collect::<Vec<_>>();

    for result in results {
        let result = result?;
        scatter_m_major_block_f32(
            rows,
            config.so2.coefficient_count(),
            config.so2.out_channels(),
            result.coeff_offset,
            result.block_rows,
            &result.block_out,
            out_m_major,
        );
        if let (Some(extra), Some(extra_dst)) = (result.extra_block, extra_m0_out.as_deref_mut()) {
            extra_dst.copy_from_slice(&extra);
        }
    }
    Ok(())
}

/// Parallel m-order VJP for `run_single_system_so2_conv_vjp_precomputed_f32`.
#[cfg(all(
    feature = "parallel",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )
))]
#[allow(clippy::too_many_arguments)]
fn run_single_system_so2_conv_vjp_precomputed_f32_parallel(
    config: UmaMoleSo2ConvolutionConfig,
    rows: usize,
    input_m_major: &[f32],
    radial: Option<&[f32]>,
    mixed_m0: &[f32],
    m0_output_feature_dim: usize,
    fused_m_weights_vjp: &[&[f32]],
    grad_out_m_major: &[f32],
    extra_m0_grad_out: Option<&[f32]>,
    grad_input_m_major: &mut [f32],
    mut grad_radial: Option<&mut [f32]>,
) -> Result<()> {
    let mut coeff_offsets = Vec::with_capacity(config.so2.m_max() + 1);
    let mut radial_offsets = Vec::with_capacity(config.so2.m_max() + 1);
    let mut coeff_offset = 0usize;
    let mut radial_offset = 0usize;
    for order in 0..=config.so2.m_max() {
        coeff_offsets.push(coeff_offset);
        radial_offsets.push(radial_offset);
        let copies = if order == 0 { 1 } else { 2 };
        let m_size = config.so2.m_size(order).ok_or_else(|| {
            invalid_mole(format!(
                "UMA SO2 f32 parallel precomputed vjp: order {order} > l_max {}",
                config.so2.l_max()
            ))
        })?;
        coeff_offset += copies * m_size;
        radial_offset += m_size * config.so2.in_channels();
    }

    let results = (0..=config.so2.m_max())
        .into_par_iter()
        .map(|order| -> Result<So2ConvF32VjpOrderResult> {
            let copies = if order == 0 { 1 } else { 2 };
            let m_size = config.so2.m_size(order).ok_or_else(|| {
                invalid_mole(format!(
                    "UMA SO2 f32 parallel precomputed vjp: order {order} > l_max {}",
                    config.so2.l_max()
                ))
            })?;
            let input_feature_dim = m_size * config.so2.in_channels();
            let output_feature_dim = m_size * config.so2.out_channels();
            let block_rows = copies * m_size;

            let mut block_grad_out = vec![0.0f32; rows * block_rows * config.so2.out_channels()];
            if order == 0 {
                gather_m_major_block_f32(
                    rows,
                    config.so2.coefficient_count(),
                    config.so2.out_channels(),
                    coeff_offsets[order],
                    block_rows,
                    grad_out_m_major,
                    &mut block_grad_out,
                );
                let mut full_grad_out = vec![0.0f32; rows * m0_output_feature_dim];
                for row in 0..rows {
                    let full_start = row * m0_output_feature_dim;
                    if config.so2.extra_m0_output_channels() > 0 {
                        if let Some(extra) = extra_m0_grad_out {
                            let extra_start = row * config.so2.extra_m0_output_channels();
                            full_grad_out
                                [full_start..full_start + config.so2.extra_m0_output_channels()]
                                .copy_from_slice(
                                    &extra[extra_start
                                        ..extra_start + config.so2.extra_m0_output_channels()],
                                );
                        }
                    }
                    let main_start = full_start + config.so2.extra_m0_output_channels();
                    let block_start = row * output_feature_dim;
                    full_grad_out[main_start..main_start + output_feature_dim].copy_from_slice(
                        &block_grad_out[block_start..block_start + output_feature_dim],
                    );
                }
                let mut grad_modulated = vec![0.0f32; rows * input_feature_dim];
                crate::blas_backend::dense_vjp_input_output_major_weight_f32(
                    rows,
                    input_feature_dim,
                    m0_output_feature_dim,
                    mixed_m0,
                    &full_grad_out,
                    &mut grad_modulated,
                    1.0,
                    0.0,
                );
                if let Some(radial_values) = radial {
                    let block_radial = gather_radial_block_f32(
                        rows,
                        config.so2.radial_node_stride(),
                        radial_offsets[order],
                        input_feature_dim,
                        radial_values,
                    );
                    let block_input = gather_m_major_block_f32_alloc(
                        rows,
                        config.so2.coefficient_count(),
                        config.so2.in_channels(),
                        coeff_offsets[order],
                        block_rows,
                        input_m_major,
                    );
                    let mut grad_block_input = vec![0.0f32; rows * input_feature_dim];
                    let radial_block_len = rows * input_feature_dim;
                    let mut grad_radial_block = vec![0.0f32; radial_block_len];
                    radial_modulate_vjp_f32(
                        rows,
                        copies,
                        input_feature_dim,
                        &block_input,
                        &block_radial,
                        &grad_modulated,
                        &mut grad_block_input,
                        &mut grad_radial_block,
                    );
                    return Ok(So2ConvF32VjpOrderResult {
                        coeff_offset: coeff_offsets[order],
                        radial_offset: radial_offsets[order],
                        block_rows,
                        input_feature_dim,
                        grad_input_block: grad_block_input,
                        grad_radial_block: Some(grad_radial_block),
                    });
                }
                return Ok(So2ConvF32VjpOrderResult {
                    coeff_offset: coeff_offsets[order],
                    radial_offset: radial_offsets[order],
                    block_rows,
                    input_feature_dim,
                    grad_input_block: grad_modulated,
                    grad_radial_block: None,
                });
            }

            // order > 0: use precomputed fused VJP weight
            gather_m_major_block_f32(
                rows,
                config.so2.coefficient_count(),
                config.so2.out_channels(),
                coeff_offsets[order],
                block_rows,
                grad_out_m_major,
                &mut block_grad_out,
            );

            let idx = order - 1;
            let mut grad_modulated = vec![0.0f32; rows * block_rows * config.so2.in_channels()];
            crate::blas_backend::so2_mconv_vjp_input_add_precomputed_fused_f32(
                rows,
                input_feature_dim,
                output_feature_dim,
                fused_m_weights_vjp[idx],
                &block_grad_out,
                &mut grad_modulated,
            );

            if let Some(radial_values) = radial {
                let block_radial = gather_radial_block_f32(
                    rows,
                    config.so2.radial_node_stride(),
                    radial_offsets[order],
                    input_feature_dim,
                    radial_values,
                );
                let block_input = gather_m_major_block_f32_alloc(
                    rows,
                    config.so2.coefficient_count(),
                    config.so2.in_channels(),
                    coeff_offsets[order],
                    block_rows,
                    input_m_major,
                );
                let mut grad_block_input =
                    vec![0.0f32; rows * block_rows * config.so2.in_channels()];
                let radial_block_len = rows * input_feature_dim;
                let mut grad_radial_block = vec![0.0f32; radial_block_len];
                radial_modulate_vjp_f32(
                    rows,
                    copies,
                    input_feature_dim,
                    &block_input,
                    &block_radial,
                    &grad_modulated,
                    &mut grad_block_input,
                    &mut grad_radial_block,
                );
                return Ok(So2ConvF32VjpOrderResult {
                    coeff_offset: coeff_offsets[order],
                    radial_offset: radial_offsets[order],
                    block_rows,
                    input_feature_dim,
                    grad_input_block: grad_block_input,
                    grad_radial_block: Some(grad_radial_block),
                });
            }
            Ok(So2ConvF32VjpOrderResult {
                coeff_offset: coeff_offsets[order],
                radial_offset: radial_offsets[order],
                block_rows,
                input_feature_dim,
                grad_input_block: grad_modulated,
                grad_radial_block: None,
            })
        })
        .collect::<Vec<_>>();

    for result in results {
        let result = result?;
        scatter_add_m_major_block_f32(
            rows,
            config.so2.coefficient_count(),
            config.so2.in_channels(),
            result.coeff_offset,
            result.block_rows,
            &result.grad_input_block,
            grad_input_m_major,
        );
        if let (Some(grad_radial_block), Some(grad_radial_values)) =
            (result.grad_radial_block, grad_radial.as_deref_mut())
        {
            for row in 0..rows {
                let src_start = row * result.input_feature_dim;
                let dst_start = row * config.so2.radial_node_stride() + result.radial_offset;
                for f in 0..result.input_feature_dim {
                    grad_radial_values[dst_start + f] += grad_radial_block[src_start + f];
                }
            }
        }
    }
    Ok(())
}

/// Precomputed-fused internal forward: like `run_single_system_so2_conv_forward_f32`
/// but uses precomputed fused weights for m>0 orders, bypassing the rebuild cost.
///
/// `fused_m_weights_fwd[i]` is the `[2*in_dim, 2*out_dim]` prebuilt fused weight for
/// m-order `i+1`.  `mixed_m_weights[i]` is the plain weight (used for the scalar VJP
/// fallback path on non-BLAS builds).
#[allow(clippy::too_many_arguments)]
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
fn run_single_system_so2_conv_forward_precomputed_f32(
    config: UmaMoleSo2ConvolutionConfig,
    rows: usize,
    input_l_major: &[f32],
    radial: Option<&[f32]>,
    mixed_m0: &[f32],
    m0_bias: &[f32],
    mixed_m_weights: &[&[f32]],     // plain mixed weights
    fused_m_weights_fwd: &[&[f32]], // precomputed fused weights
    out_l_major: &mut [f32],
    extra_m0_out: Option<&mut [f32]>,
) -> Result<()> {
    let mapping = EsenCoefficientMappingConfig::new(config.so2.l_max(), config.so2.m_max())
        .map_err(|err| invalid_mole(err.to_string()))?;
    let permutation = e3nn_core::so3::fairchem_esen_l_to_m_permutation(mapping);

    let mut input_m_major = vec![0.0f32; input_l_major.len()];
    l_to_m_permutation_apply_f32(
        &permutation,
        config.so2.in_channels(),
        input_l_major,
        &mut input_m_major,
    );

    let mut out_m_major = vec![0.0f32; out_l_major.len()];

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && rows >= 512 && config.so2.m_max() > 0 {
        run_single_system_so2_conv_forward_precomputed_f32_parallel(
            config,
            rows,
            &input_m_major,
            radial,
            mixed_m0,
            m0_bias,
            fused_m_weights_fwd,
            &mut out_m_major,
            extra_m0_out,
        )?;
        m_to_l_permutation_apply_f32(
            &permutation,
            config.so2.out_channels(),
            &out_m_major,
            out_l_major,
        );
        return Ok(());
    }

    let mut coeff_offset = 0usize;
    let mut radial_offset = 0usize;
    let mut extra_m0_out = extra_m0_out;

    for order in 0..=config.so2.m_max() {
        let copies = if order == 0 { 1 } else { 2 };
        let m_size = config.so2.m_size(order).ok_or_else(|| {
            invalid_mole(format!(
                "UMA MoLE SO2 f32 order {order} exceeds l_max {}",
                config.so2.l_max()
            ))
        })?;
        let input_feature_dim = m_size * config.so2.in_channels();
        let output_feature_dim = m_size * config.so2.out_channels();
        let block_rows = copies * m_size;

        let mut block_input = vec![0.0f32; rows * block_rows * config.so2.in_channels()];
        gather_m_major_block_f32(
            rows,
            config.so2.coefficient_count(),
            config.so2.in_channels(),
            coeff_offset,
            block_rows,
            &input_m_major,
            &mut block_input,
        );

        let modulated_input: Vec<f32> = if let Some(radial_values) = radial {
            let block_radial = gather_radial_block_f32(
                rows,
                config.so2.radial_node_stride(),
                radial_offset,
                input_feature_dim,
                radial_values,
            );
            let mut modulated = vec![0.0f32; block_input.len()];
            radial_modulate_f32(
                rows,
                copies,
                input_feature_dim,
                &block_input,
                &block_radial,
                &mut modulated,
            );
            modulated
        } else {
            block_input
        };

        let mut block_out = vec![0.0f32; rows * block_rows * config.so2.out_channels()];
        if order == 0 {
            let mut dense_out = vec![0.0f32; rows * config.so2.m0_output_feature_dim()];
            dense_forward_with_bias_f32(
                rows,
                input_feature_dim,
                config.so2.m0_output_feature_dim(),
                mixed_m0,
                m0_bias,
                &modulated_input,
                &mut dense_out,
            );
            for row in 0..rows {
                let out_start = row * config.so2.m0_output_feature_dim();
                if config.so2.extra_m0_output_channels() > 0 {
                    let extra_values = extra_m0_out
                        .as_mut()
                        .ok_or_else(|| invalid_mole("validated UMA extra m0 output is missing"))?;
                    let extra_start = row * config.so2.extra_m0_output_channels();
                    extra_values[extra_start..extra_start + config.so2.extra_m0_output_channels()]
                        .copy_from_slice(
                            &dense_out
                                [out_start..out_start + config.so2.extra_m0_output_channels()],
                        );
                }
                let main_src_start = out_start + config.so2.extra_m0_output_channels();
                let block_start = row * output_feature_dim;
                block_out[block_start..block_start + output_feature_dim].copy_from_slice(
                    &dense_out[main_src_start..main_src_start + output_feature_dim],
                );
            }
        } else {
            let idx = order - 1;
            // Use the precomputed fused weight for this order.
            crate::blas_backend::so2_mconv_forward_set_precomputed_fused_f32(
                rows,
                input_feature_dim,
                output_feature_dim,
                fused_m_weights_fwd[idx],
                &modulated_input,
                &mut block_out,
            );
            let _ = mixed_m_weights; // not used in BLAS path
        }
        scatter_m_major_block_f32(
            rows,
            config.so2.coefficient_count(),
            config.so2.out_channels(),
            coeff_offset,
            block_rows,
            &block_out,
            &mut out_m_major,
        );
        coeff_offset += block_rows;
        radial_offset += input_feature_dim;
    }

    m_to_l_permutation_apply_f32(
        &permutation,
        config.so2.out_channels(),
        &out_m_major,
        out_l_major,
    );
    Ok(())
}

/// f32 twin of [`fairchem_uma_mole_so2_convolution_forward`].
///
/// Native f32 path for the single-system case: mixes expert weights in f32,
/// then runs the SO2 convolution in f32. Multi-system falls back to the f64
/// upcast path.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_uma_mole_so2_convolution_forward_f32(
    config: UmaMoleSo2ConvolutionConfig,
    input_l_major: &[f32],
    radial: Option<&[f32]>,
    m0_expert_weights: &[f32],
    m0_bias: &[f32],
    m_expert_weights: &[&[f32]],
    coefficients: &[f32],
    system_sizes: &[usize],
    out_l_major: &mut [f32],
    extra_m0_out: Option<&mut [f32]>,
) -> Result<()> {
    validate_mole_so2_config(config)?;
    let rows = check_system_sizes(config.systems, system_sizes)?;

    if config.systems == 1 {
        let coeff_row = &coefficients[..config.experts];
        let mixed_m0 = mix_expert_weights_f32(
            config.experts,
            config.so2.m0_weight_len(),
            m0_expert_weights,
            coeff_row,
        )?;
        let m0_bias_f32: Vec<f32> = m0_bias.to_vec();
        let mut mixed_m_weights = Vec::with_capacity(config.so2.m_max());
        for order in 1..=config.so2.m_max() {
            mixed_m_weights.push(mix_expert_weights_f32(
                config.experts,
                config.so2.m_conv_weight_len(order),
                m_expert_weights[order - 1],
                coeff_row,
            )?);
        }
        check_len(
            "UMA MoLE SO2 f32 input",
            rows * config.so2.input_node_stride(),
            input_l_major.len(),
        )?;
        if let Some(radial_values) = radial {
            check_len(
                "UMA MoLE SO2 f32 radial",
                rows * config.so2.radial_node_stride(),
                radial_values.len(),
            )?;
        }
        check_len(
            "UMA MoLE SO2 f32 output",
            rows * config.so2.output_node_stride(),
            out_l_major.len(),
        )?;
        if let Some(extra) = extra_m0_out.as_ref() {
            check_len(
                "UMA MoLE SO2 f32 extra_m0_out",
                rows * config.so2.extra_m0_output_channels(),
                extra.len(),
            )?;
        } else if config.so2.extra_m0_output_channels() > 0 {
            return Err(invalid_mole(
                "UMA MoLE SO2 f32 convolution requires extra_m0_out for configured extra channels",
            ));
        }
        return run_single_system_so2_conv_forward_f32(
            config,
            rows,
            input_l_major,
            radial,
            &mixed_m0,
            &m0_bias_f32,
            &mixed_m_weights,
            out_l_major,
            extra_m0_out,
        );
    }

    // Multi-system fallback: upcast to f64 and call the f64 path.
    let input_f64: Vec<f64> = input_l_major.iter().map(|&x| x as f64).collect();
    let radial_f64: Option<Vec<f64>> = radial.map(|r| r.iter().map(|&x| x as f64).collect());
    let m0_expert_f64: Vec<f64> = m0_expert_weights.iter().map(|&x| x as f64).collect();
    let m0_bias_f64: Vec<f64> = m0_bias.iter().map(|&x| x as f64).collect();
    let m_expert_f64_vecs: Vec<Vec<f64>> = m_expert_weights
        .iter()
        .map(|s| s.iter().map(|&x| x as f64).collect())
        .collect();
    let m_expert_refs: Vec<&[f64]> = m_expert_f64_vecs.iter().map(Vec::as_slice).collect();
    let coefficients_f64: Vec<f64> = coefficients.iter().map(|&x| x as f64).collect();
    let mut out_f64: Vec<f64> = vec![0.0; out_l_major.len()];
    let mut extra_m0_f64: Option<Vec<f64>> = extra_m0_out.as_ref().map(|e| vec![0.0f64; e.len()]);

    fairchem_uma_mole_so2_convolution_forward(
        config,
        &input_f64,
        radial_f64.as_deref(),
        &m0_expert_f64,
        &m0_bias_f64,
        &m_expert_refs,
        &coefficients_f64,
        system_sizes,
        &mut out_f64,
        extra_m0_f64.as_deref_mut(),
    )?;

    for (dst, &src) in out_l_major.iter_mut().zip(out_f64.iter()) {
        *dst = src as f32;
    }
    if let (Some(extra_dst), Some(extra_src)) = (extra_m0_out, extra_m0_f64.as_ref()) {
        for (dst, &src) in extra_dst.iter_mut().zip(extra_src.iter()) {
            *dst = src as f32;
        }
    }
    Ok(())
}

/// f32 twin of [`fairchem_uma_mole_so2_convolution_vjp`].
///
/// Mixes expert weights in f32 to produce a single mixed weight matrix,
/// then upcasts only the mixed weights (not all expert weights) to f64
/// for the SO2 VJP.
#[allow(clippy::too_many_arguments)]
pub fn fairchem_uma_mole_so2_convolution_vjp_f32(
    config: UmaMoleSo2ConvolutionConfig,
    input_l_major: &[f32],
    radial: Option<&[f32]>,
    m0_expert_weights: &[f32],
    m_expert_weights: &[&[f32]],
    coefficients: &[f32],
    system_sizes: &[usize],
    grad_out_l_major: &[f32],
    grad_extra_m0: Option<&[f32]>,
    grad_input_l_major: &mut [f32],
    grad_radial: Option<&mut [f32]>,
) -> Result<()> {
    validate_mole_so2_config(config)?;
    if config.systems != 1 {
        return Err(invalid_mole(
            "UMA MoLE SO2 VJP f32 currently supports single-system inference only",
        ));
    }
    let rows = check_system_sizes(config.systems, system_sizes)?;

    let coeff_row = &coefficients[..config.experts];
    let mixed_m0 = mix_expert_weights_f32(
        config.experts,
        config.so2.m0_weight_len(),
        m0_expert_weights,
        coeff_row,
    )?;
    let mut mixed_m_weights = Vec::with_capacity(config.so2.m_max());
    for order in 1..=config.so2.m_max() {
        mixed_m_weights.push(mix_expert_weights_f32(
            config.experts,
            config.so2.m_conv_weight_len(order),
            m_expert_weights[order - 1],
            coeff_row,
        )?);
    }

    let m0_output_feature_dim = config.so2.m0_output_feature_dim();
    run_single_system_so2_conv_vjp_f32(
        config,
        rows,
        input_l_major,
        radial,
        &mixed_m0,
        m0_output_feature_dim,
        &mixed_m_weights,
        grad_out_l_major,
        grad_extra_m0,
        grad_input_l_major,
        grad_radial,
    )
}

/// Precomputed-fused fast path for the single-system f32 SO2 forward convolution.
///
/// Like `fairchem_uma_mole_so2_convolution_forward_f32` but takes precomputed
/// fused weight matrices (`fused_m_weights_fwd`) that were built at weight-loading
/// time.  Eliminates both the per-call `mix_expert_weights_f32` allocation and the
/// per-call rebuild of the `[[W0^T, W1^T], [-W1^T, W0^T]]` fused matrix.
///
/// Callers must pass `experts == 1` and `coefficients == [1.0]`.
/// `m_weights` are the plain (un-fused) mixed weights — still needed for m=0 and
/// for non-BLAS code paths.
/// `fused_m_weights_fwd[i]` is the precomputed `[2*in_dim, 2*out_dim]` fused weight
/// for m-order `i+1`.
#[allow(clippy::too_many_arguments)]
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub fn fairchem_uma_mole_so2_convolution_forward_precomputed_f32(
    so2: EsenSo2ConvolutionConfig,
    rows: usize,
    input_l_major: &[f32],
    radial: Option<&[f32]>,
    m0_mixed_weight: &[f32],
    m0_bias: &[f32],
    m_mixed_weights: &[&[f32]], // plain mixed weights (un-fused), one per m-order
    fused_m_weights_fwd: &[&[f32]], // precomputed fused weights, one per m-order
    out_l_major: &mut [f32],
    extra_m0_out: Option<&mut [f32]>,
) -> Result<()> {
    run_single_system_so2_conv_forward_precomputed_f32(
        UmaMoleSo2ConvolutionConfig {
            systems: 1,
            experts: 1,
            so2,
        },
        rows,
        input_l_major,
        radial,
        m0_mixed_weight,
        m0_bias,
        m_mixed_weights,
        fused_m_weights_fwd,
        out_l_major,
        extra_m0_out,
    )
}

/// Precomputed-fused fast path for the single-system f32 SO2 VJP.
///
/// Like `fairchem_uma_mole_so2_convolution_vjp_f32` but takes precomputed
/// fused VJP weight matrices (`fused_m_weights_vjp`) that were built at
/// weight-loading time.  Eliminates both the per-call `mix_expert_weights_f32`
/// allocation and the per-call rebuild of the `[[W0, -W1], [W1, W0]]` matrix.
///
/// `fused_m_weights_vjp[i]` is the precomputed `[2*out_dim, 2*in_dim]` fused
/// weight for m-order `i+1`.
#[allow(clippy::too_many_arguments)]
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub fn fairchem_uma_mole_so2_convolution_vjp_precomputed_f32(
    so2: EsenSo2ConvolutionConfig,
    rows: usize,
    input_l_major: &[f32],
    radial: Option<&[f32]>,
    m0_mixed_weight: &[f32],
    m_mixed_weights: &[&[f32]],     // plain mixed weights (un-fused)
    fused_m_weights_vjp: &[&[f32]], // precomputed fused VJP weights
    grad_out_l_major: &[f32],
    grad_extra_m0: Option<&[f32]>,
    grad_input_l_major: &mut [f32],
    grad_radial: Option<&mut [f32]>,
) -> Result<()> {
    let m0_output_feature_dim = so2.m0_output_feature_dim();
    run_single_system_so2_conv_vjp_precomputed_f32(
        UmaMoleSo2ConvolutionConfig {
            systems: 1,
            experts: 1,
            so2,
        },
        rows,
        input_l_major,
        radial,
        m0_mixed_weight,
        m0_output_feature_dim,
        m_mixed_weights,
        fused_m_weights_vjp,
        grad_out_l_major,
        grad_extra_m0,
        grad_input_l_major,
        grad_radial,
    )
}

// ---------------------------------------------------------------------------
// Native f32 SO2 VJP helpers
// ---------------------------------------------------------------------------

/// Accumulate a contiguous block into a row of `grad_input_m_major`
/// (VJP of `gather_m_major_block_f32`).
fn scatter_add_m_major_block_f32(
    rows: usize,
    coefficient_count: usize,
    channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    source: &[f32],
    target: &mut [f32],
) {
    let target_row_stride = coefficient_count * channels;
    let source_row_stride = block_coefficients * channels;
    for row in 0..rows {
        let source_start = row * source_row_stride;
        let target_start = row * target_row_stride + coefficient_offset * channels;
        for i in 0..source_row_stride {
            target[target_start + i] += source[source_start + i];
        }
    }
}

/// VJP of `radial_modulate_f32`.
///
/// Forward: `modulated[row, copy, f] = block_input[row, copy, f] * radial[row, f]`
///
/// VJP accumulates (does NOT zero first):
/// `grad_block_input[row, copy, f] += grad_modulated[row, copy, f] * radial[row, f]`
/// `grad_radial[row, f]            += sum_copy(grad_modulated[row, copy, f] * block_input[row, copy, f])`
#[allow(clippy::too_many_arguments)]
fn radial_modulate_vjp_f32(
    rows: usize,
    copies: usize,
    feature_dim: usize,
    block_input: &[f32],
    radial: &[f32],
    grad_modulated: &[f32],
    grad_block_input: &mut [f32],
    grad_radial: &mut [f32],
) {
    for row in 0..rows {
        let rad_start = row * feature_dim;
        for copy in 0..copies {
            let base = row * copies * feature_dim + copy * feature_dim;
            for f in 0..feature_dim {
                let gm = grad_modulated[base + f];
                grad_block_input[base + f] += gm * radial[rad_start + f];
                grad_radial[rad_start + f] += gm * block_input[base + f];
            }
        }
    }
}

/// VJP of `mole_so2_m_conv_forward_single_system_f32` with respect to the input.
///
/// Forward complex matmul (per row, for each (real, imag) pair):
/// `real_out = real_in * W_direct^T - imag_in * W_mixed^T`
/// `imag_out = imag_in * W_direct^T + real_in * W_mixed^T`
///
/// VJP (accumulates into `grad_input`):
/// `grad_real_in += W_direct * grad_real_out + W_mixed * grad_imag_out`
/// `grad_imag_in += W_direct * grad_imag_out - W_mixed * grad_real_out`
///
/// With BLAS: uses a single fused SGEMM via `so2_mconv_vjp_input_add_fused_f32`
/// (one `[rows, 2*out] @ [2*out, 2*in]` call instead of four `[rows, out] @ [out, in]`
/// calls), which yields the same FLOP count but with fewer BLAS dispatch overheads and
/// better SIMD utilisation on the doubled-size matrix.
fn mole_so2_m_conv_vjp_input_f32(
    input_len: usize,
    output_len: usize,
    rows: usize,
    mixed_weight: &[f32],
    grad_out: &[f32],       // [rows, 2, output_len]
    grad_input: &mut [f32], // [rows, 2, input_len]  (accumulates)
) {
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    {
        // Single fused SGEMM: grad_input += grad_out @ [[W0,-W1],[W1,W0]].
        // grad_out is [rows, 2*output_len] and grad_input is [rows, 2*input_len],
        // both in block-interleaved (real then imag) layout — no de/re-interleave needed.
        crate::blas_backend::so2_mconv_vjp_input_add_fused_f32(
            rows,
            input_len,
            output_len,
            mixed_weight,
            grad_out,
            grad_input,
        );
    }

    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    {
        let w_direct = &mixed_weight[..output_len * input_len];
        let w_mixed = &mixed_weight[output_len * input_len..2 * output_len * input_len];
        for row in 0..rows {
            let out_start = row * 2 * output_len;
            let grad_real_out = &grad_out[out_start..out_start + output_len];
            let grad_imag_out = &grad_out[out_start + output_len..out_start + 2 * output_len];
            let in_start = row * 2 * input_len;
            for in_index in 0..input_len {
                let mut g_real = 0.0f32;
                let mut g_imag = 0.0f32;
                for out_index in 0..output_len {
                    let w_row = out_index * input_len + in_index;
                    let wd = w_direct[w_row];
                    let wm = w_mixed[w_row];
                    g_real += wd * grad_real_out[out_index] + wm * grad_imag_out[out_index];
                    g_imag += wd * grad_imag_out[out_index] - wm * grad_real_out[out_index];
                }
                grad_input[in_start + in_index] += g_real;
                grad_input[in_start + input_len + in_index] += g_imag;
            }
        }
    }
}

/// Native f32 single-system SO2 convolution VJP with respect to the input.
///
/// Mirrors the structure of `run_single_system_so2_conv_forward_f32` in reverse.
/// Does NOT compute weight gradients (weight gradients are not needed for UMA force VJP).
#[allow(clippy::too_many_arguments)]
fn run_single_system_so2_conv_vjp_f32(
    config: UmaMoleSo2ConvolutionConfig,
    rows: usize,
    input_l_major: &[f32],
    radial: Option<&[f32]>,
    mixed_m0: &[f32],
    m0_output_feature_dim: usize,
    mixed_m_weights: &[Vec<f32>],
    grad_out_l_major: &[f32],
    extra_m0_grad_out: Option<&[f32]>,
    grad_input_l_major: &mut [f32],
    mut grad_radial: Option<&mut [f32]>,
) -> Result<()> {
    let mapping = EsenCoefficientMappingConfig::new(config.so2.l_max(), config.so2.m_max())
        .map_err(|err| invalid_mole(err.to_string()))?;
    let permutation = e3nn_core::so3::fairchem_esen_l_to_m_permutation(mapping);

    // Permute input and grad_out from l-major to m-major.
    let mut input_m_major = vec![0.0f32; input_l_major.len()];
    l_to_m_permutation_apply_f32(
        &permutation,
        config.so2.in_channels(),
        input_l_major,
        &mut input_m_major,
    );

    let mut grad_out_m_major = vec![0.0f32; grad_out_l_major.len()];
    l_to_m_permutation_apply_f32(
        &permutation,
        config.so2.out_channels(),
        grad_out_l_major,
        &mut grad_out_m_major,
    );

    let mut grad_input_m_major = vec![0.0f32; input_l_major.len()];

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && rows >= 512 && config.so2.m_max() > 0 {
        run_single_system_so2_conv_vjp_f32_parallel(
            config,
            rows,
            &input_m_major,
            radial,
            mixed_m0,
            m0_output_feature_dim,
            mixed_m_weights,
            &grad_out_m_major,
            extra_m0_grad_out,
            &mut grad_input_m_major,
            grad_radial.as_deref_mut(),
        )?;
        let mut grad_input_l_temp = vec![0.0f32; input_l_major.len()];
        m_to_l_permutation_apply_f32(
            &permutation,
            config.so2.in_channels(),
            &grad_input_m_major,
            &mut grad_input_l_temp,
        );
        for (dst, src) in grad_input_l_major.iter_mut().zip(grad_input_l_temp.iter()) {
            *dst += src;
        }
        return Ok(());
    }

    let mut coeff_offset = 0usize;
    let mut radial_offset = 0usize;

    for order in 0..=config.so2.m_max() {
        let copies = if order == 0 { 1 } else { 2 };
        let m_size = config.so2.m_size(order).ok_or_else(|| {
            invalid_mole(format!(
                "UMA MoLE SO2 f32 VJP order {order} exceeds l_max {}",
                config.so2.l_max()
            ))
        })?;
        let input_feature_dim = m_size * config.so2.in_channels();
        let output_feature_dim = m_size * config.so2.out_channels();
        let block_rows = copies * m_size;

        // Gather grad_out block.
        let mut block_grad_out = vec![0.0f32; rows * block_rows * config.so2.out_channels()];
        if order == 0 {
            // Gather the main m0 output channels from grad_out_m_major.
            gather_m_major_block_f32(
                rows,
                config.so2.coefficient_count(),
                config.so2.out_channels(),
                coeff_offset,
                block_rows,
                &grad_out_m_major,
                &mut block_grad_out,
            );
            // Reconstruct the full m0 grad including any extra channels.
            let mut full_grad_out = vec![0.0f32; rows * m0_output_feature_dim];
            for row in 0..rows {
                let full_start = row * m0_output_feature_dim;
                if config.so2.extra_m0_output_channels() > 0 {
                    if let Some(extra) = extra_m0_grad_out {
                        let extra_start = row * config.so2.extra_m0_output_channels();
                        full_grad_out
                            [full_start..full_start + config.so2.extra_m0_output_channels()]
                            .copy_from_slice(
                                &extra[extra_start
                                    ..extra_start + config.so2.extra_m0_output_channels()],
                            );
                    }
                }
                let main_start = full_start + config.so2.extra_m0_output_channels();
                let block_start = row * output_feature_dim;
                full_grad_out[main_start..main_start + output_feature_dim].copy_from_slice(
                    &block_grad_out[block_start..block_start + output_feature_dim],
                );
            }
            // VJP of dense (no bias): grad_modulated = grad_out * W.
            let mut grad_modulated = vec![0.0f32; rows * input_feature_dim];
            #[cfg(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    any(feature = "intel-mkl", feature = "openblas"),
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            ))]
            {
                crate::blas_backend::dense_vjp_input_output_major_weight_f32(
                    rows,
                    input_feature_dim,
                    m0_output_feature_dim,
                    mixed_m0,
                    &full_grad_out,
                    &mut grad_modulated,
                    1.0,
                    0.0,
                );
            }
            #[cfg(not(any(
                all(feature = "accelerate-blas", target_os = "macos"),
                all(
                    any(feature = "intel-mkl", feature = "openblas"),
                    target_os = "linux",
                    any(target_arch = "x86_64", target_arch = "aarch64")
                )
            )))]
            for row in 0..rows {
                let out_start = row * m0_output_feature_dim;
                let in_start = row * input_feature_dim;
                for in_i in 0..input_feature_dim {
                    let mut acc = 0.0f32;
                    for out_j in 0..m0_output_feature_dim {
                        acc += mixed_m0[out_j * input_feature_dim + in_i]
                            * full_grad_out[out_start + out_j];
                    }
                    grad_modulated[in_start + in_i] += acc;
                }
            }

            // Propagate through radial modulation if present.
            if let Some(radial_values) = radial {
                let block_radial = gather_radial_block_f32(
                    rows,
                    config.so2.radial_node_stride(),
                    radial_offset,
                    input_feature_dim,
                    radial_values,
                );
                let block_input = gather_m_major_block_f32_alloc(
                    rows,
                    config.so2.coefficient_count(),
                    config.so2.in_channels(),
                    coeff_offset,
                    block_rows,
                    &input_m_major,
                );
                let mut grad_block_input = vec![0.0f32; rows * input_feature_dim];
                let grad_radial = grad_radial
                    .as_deref_mut()
                    .ok_or_else(|| invalid_mole("grad_radial required when radial is present"))?;
                // We need the radial slice for this block in grad_radial.
                let radial_block_len = rows * input_feature_dim;
                let mut grad_radial_block = vec![0.0f32; radial_block_len];
                radial_modulate_vjp_f32(
                    rows,
                    copies,
                    input_feature_dim,
                    &block_input,
                    &block_radial,
                    &grad_modulated,
                    &mut grad_block_input,
                    &mut grad_radial_block,
                );
                scatter_add_m_major_block_f32(
                    rows,
                    config.so2.coefficient_count(),
                    config.so2.in_channels(),
                    coeff_offset,
                    block_rows,
                    &grad_block_input,
                    &mut grad_input_m_major,
                );
                // Scatter radial gradients back.
                for row in 0..rows {
                    let src_start = row * input_feature_dim;
                    let dst_start = row * config.so2.radial_node_stride() + radial_offset;
                    for f in 0..input_feature_dim {
                        grad_radial[dst_start + f] += grad_radial_block[src_start + f];
                    }
                }
            } else {
                scatter_add_m_major_block_f32(
                    rows,
                    config.so2.coefficient_count(),
                    config.so2.in_channels(),
                    coeff_offset,
                    block_rows,
                    &grad_modulated,
                    &mut grad_input_m_major,
                );
            }
        } else {
            gather_m_major_block_f32(
                rows,
                config.so2.coefficient_count(),
                config.so2.out_channels(),
                coeff_offset,
                block_rows,
                &grad_out_m_major,
                &mut block_grad_out,
            );

            // VJP of the m-conv.
            let mut grad_modulated = vec![0.0f32; rows * block_rows * config.so2.in_channels()];
            mole_so2_m_conv_vjp_input_f32(
                input_feature_dim,
                output_feature_dim,
                rows,
                &mixed_m_weights[order - 1],
                &block_grad_out,
                &mut grad_modulated,
            );

            // Propagate through radial modulation if present.
            if let Some(radial_values) = radial {
                let block_radial = gather_radial_block_f32(
                    rows,
                    config.so2.radial_node_stride(),
                    radial_offset,
                    input_feature_dim,
                    radial_values,
                );
                let block_input = gather_m_major_block_f32_alloc(
                    rows,
                    config.so2.coefficient_count(),
                    config.so2.in_channels(),
                    coeff_offset,
                    block_rows,
                    &input_m_major,
                );
                let mut grad_block_input =
                    vec![0.0f32; rows * block_rows * config.so2.in_channels()];
                let grad_radial = grad_radial
                    .as_deref_mut()
                    .ok_or_else(|| invalid_mole("grad_radial required when radial is present"))?;
                let radial_block_len = rows * input_feature_dim;
                let mut grad_radial_block = vec![0.0f32; radial_block_len];
                radial_modulate_vjp_f32(
                    rows,
                    copies,
                    input_feature_dim,
                    &block_input,
                    &block_radial,
                    &grad_modulated,
                    &mut grad_block_input,
                    &mut grad_radial_block,
                );
                scatter_add_m_major_block_f32(
                    rows,
                    config.so2.coefficient_count(),
                    config.so2.in_channels(),
                    coeff_offset,
                    block_rows,
                    &grad_block_input,
                    &mut grad_input_m_major,
                );
                for row in 0..rows {
                    let src_start = row * input_feature_dim;
                    let dst_start = row * config.so2.radial_node_stride() + radial_offset;
                    for f in 0..input_feature_dim {
                        grad_radial[dst_start + f] += grad_radial_block[src_start + f];
                    }
                }
            } else {
                scatter_add_m_major_block_f32(
                    rows,
                    config.so2.coefficient_count(),
                    config.so2.in_channels(),
                    coeff_offset,
                    block_rows,
                    &grad_modulated,
                    &mut grad_input_m_major,
                );
            }
        }
        coeff_offset += block_rows;
        radial_offset += input_feature_dim;
    }

    // Permute grad_input from m-major back to l-major (accumulate).
    let mut grad_input_l_temp = vec![0.0f32; input_l_major.len()];
    m_to_l_permutation_apply_f32(
        &permutation,
        config.so2.in_channels(),
        &grad_input_m_major,
        &mut grad_input_l_temp,
    );
    for (dst, src) in grad_input_l_major.iter_mut().zip(grad_input_l_temp.iter()) {
        *dst += src;
    }
    Ok(())
}

/// Precomputed-fused internal VJP: like `run_single_system_so2_conv_vjp_f32` but
/// uses precomputed fused VJP weights for m>0 orders.
///
/// `fused_m_weights_vjp[i]` is the `[2*out_dim, 2*in_dim]` prebuilt fused weight for
/// m-order `i+1`.
#[allow(clippy::too_many_arguments)]
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
fn run_single_system_so2_conv_vjp_precomputed_f32(
    config: UmaMoleSo2ConvolutionConfig,
    rows: usize,
    input_l_major: &[f32],
    radial: Option<&[f32]>,
    mixed_m0: &[f32],
    m0_output_feature_dim: usize,
    mixed_m_weights: &[&[f32]],     // plain mixed weights (for m0 path)
    fused_m_weights_vjp: &[&[f32]], // precomputed fused VJP weights
    grad_out_l_major: &[f32],
    extra_m0_grad_out: Option<&[f32]>,
    grad_input_l_major: &mut [f32],
    mut grad_radial: Option<&mut [f32]>,
) -> Result<()> {
    let mapping = EsenCoefficientMappingConfig::new(config.so2.l_max(), config.so2.m_max())
        .map_err(|err| invalid_mole(err.to_string()))?;
    let permutation = e3nn_core::so3::fairchem_esen_l_to_m_permutation(mapping);

    let mut input_m_major = vec![0.0f32; input_l_major.len()];
    l_to_m_permutation_apply_f32(
        &permutation,
        config.so2.in_channels(),
        input_l_major,
        &mut input_m_major,
    );

    let mut grad_out_m_major = vec![0.0f32; grad_out_l_major.len()];
    l_to_m_permutation_apply_f32(
        &permutation,
        config.so2.out_channels(),
        grad_out_l_major,
        &mut grad_out_m_major,
    );

    let mut grad_input_m_major = vec![0.0f32; input_l_major.len()];

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && rows >= 512 && config.so2.m_max() > 0 {
        run_single_system_so2_conv_vjp_precomputed_f32_parallel(
            config,
            rows,
            &input_m_major,
            radial,
            mixed_m0,
            m0_output_feature_dim,
            fused_m_weights_vjp,
            &grad_out_m_major,
            extra_m0_grad_out,
            &mut grad_input_m_major,
            grad_radial.as_deref_mut(),
        )?;
        let mut grad_input_l_temp = vec![0.0f32; input_l_major.len()];
        m_to_l_permutation_apply_f32(
            &permutation,
            config.so2.in_channels(),
            &grad_input_m_major,
            &mut grad_input_l_temp,
        );
        for (dst, src) in grad_input_l_major.iter_mut().zip(grad_input_l_temp.iter()) {
            *dst += src;
        }
        return Ok(());
    }

    let mut coeff_offset = 0usize;
    let mut radial_offset = 0usize;

    for order in 0..=config.so2.m_max() {
        let copies = if order == 0 { 1 } else { 2 };
        let m_size = config.so2.m_size(order).ok_or_else(|| {
            invalid_mole(format!(
                "UMA MoLE SO2 f32 VJP order {order} exceeds l_max {}",
                config.so2.l_max()
            ))
        })?;
        let input_feature_dim = m_size * config.so2.in_channels();
        let output_feature_dim = m_size * config.so2.out_channels();
        let block_rows = copies * m_size;

        let mut block_grad_out = vec![0.0f32; rows * block_rows * config.so2.out_channels()];
        if order == 0 {
            gather_m_major_block_f32(
                rows,
                config.so2.coefficient_count(),
                config.so2.out_channels(),
                coeff_offset,
                block_rows,
                &grad_out_m_major,
                &mut block_grad_out,
            );
            let mut full_grad_out = vec![0.0f32; rows * m0_output_feature_dim];
            for row in 0..rows {
                let full_start = row * m0_output_feature_dim;
                if config.so2.extra_m0_output_channels() > 0 {
                    if let Some(extra) = extra_m0_grad_out {
                        let extra_start = row * config.so2.extra_m0_output_channels();
                        full_grad_out
                            [full_start..full_start + config.so2.extra_m0_output_channels()]
                            .copy_from_slice(
                                &extra[extra_start
                                    ..extra_start + config.so2.extra_m0_output_channels()],
                            );
                    }
                }
                let main_start = full_start + config.so2.extra_m0_output_channels();
                let block_start = row * output_feature_dim;
                full_grad_out[main_start..main_start + output_feature_dim].copy_from_slice(
                    &block_grad_out[block_start..block_start + output_feature_dim],
                );
            }
            let mut grad_modulated = vec![0.0f32; rows * input_feature_dim];
            crate::blas_backend::dense_vjp_input_output_major_weight_f32(
                rows,
                input_feature_dim,
                m0_output_feature_dim,
                mixed_m0,
                &full_grad_out,
                &mut grad_modulated,
                1.0,
                0.0,
            );

            let _ = mixed_m_weights; // not used here

            if let Some(radial_values) = radial {
                let block_radial = gather_radial_block_f32(
                    rows,
                    config.so2.radial_node_stride(),
                    radial_offset,
                    input_feature_dim,
                    radial_values,
                );
                let block_input = gather_m_major_block_f32_alloc(
                    rows,
                    config.so2.coefficient_count(),
                    config.so2.in_channels(),
                    coeff_offset,
                    block_rows,
                    &input_m_major,
                );
                let mut grad_block_input = vec![0.0f32; rows * input_feature_dim];
                let grad_radial = grad_radial
                    .as_deref_mut()
                    .ok_or_else(|| invalid_mole("grad_radial required when radial is present"))?;
                let radial_block_len = rows * input_feature_dim;
                let mut grad_radial_block = vec![0.0f32; radial_block_len];
                radial_modulate_vjp_f32(
                    rows,
                    copies,
                    input_feature_dim,
                    &block_input,
                    &block_radial,
                    &grad_modulated,
                    &mut grad_block_input,
                    &mut grad_radial_block,
                );
                scatter_add_m_major_block_f32(
                    rows,
                    config.so2.coefficient_count(),
                    config.so2.in_channels(),
                    coeff_offset,
                    block_rows,
                    &grad_block_input,
                    &mut grad_input_m_major,
                );
                for row in 0..rows {
                    let src_start = row * input_feature_dim;
                    let dst_start = row * config.so2.radial_node_stride() + radial_offset;
                    for f in 0..input_feature_dim {
                        grad_radial[dst_start + f] += grad_radial_block[src_start + f];
                    }
                }
            } else {
                scatter_add_m_major_block_f32(
                    rows,
                    config.so2.coefficient_count(),
                    config.so2.in_channels(),
                    coeff_offset,
                    block_rows,
                    &grad_modulated,
                    &mut grad_input_m_major,
                );
            }
        } else {
            gather_m_major_block_f32(
                rows,
                config.so2.coefficient_count(),
                config.so2.out_channels(),
                coeff_offset,
                block_rows,
                &grad_out_m_major,
                &mut block_grad_out,
            );

            let idx = order - 1;
            let mut grad_modulated = vec![0.0f32; rows * block_rows * config.so2.in_channels()];
            // Use the precomputed fused VJP weight — no rebuild overhead.
            crate::blas_backend::so2_mconv_vjp_input_add_precomputed_fused_f32(
                rows,
                input_feature_dim,
                output_feature_dim,
                fused_m_weights_vjp[idx],
                &block_grad_out,
                &mut grad_modulated,
            );

            if let Some(radial_values) = radial {
                let block_radial = gather_radial_block_f32(
                    rows,
                    config.so2.radial_node_stride(),
                    radial_offset,
                    input_feature_dim,
                    radial_values,
                );
                let block_input = gather_m_major_block_f32_alloc(
                    rows,
                    config.so2.coefficient_count(),
                    config.so2.in_channels(),
                    coeff_offset,
                    block_rows,
                    &input_m_major,
                );
                let mut grad_block_input =
                    vec![0.0f32; rows * block_rows * config.so2.in_channels()];
                let grad_radial = grad_radial
                    .as_deref_mut()
                    .ok_or_else(|| invalid_mole("grad_radial required when radial is present"))?;
                let radial_block_len = rows * input_feature_dim;
                let mut grad_radial_block = vec![0.0f32; radial_block_len];
                radial_modulate_vjp_f32(
                    rows,
                    copies,
                    input_feature_dim,
                    &block_input,
                    &block_radial,
                    &grad_modulated,
                    &mut grad_block_input,
                    &mut grad_radial_block,
                );
                scatter_add_m_major_block_f32(
                    rows,
                    config.so2.coefficient_count(),
                    config.so2.in_channels(),
                    coeff_offset,
                    block_rows,
                    &grad_block_input,
                    &mut grad_input_m_major,
                );
                for row in 0..rows {
                    let src_start = row * input_feature_dim;
                    let dst_start = row * config.so2.radial_node_stride() + radial_offset;
                    for f in 0..input_feature_dim {
                        grad_radial[dst_start + f] += grad_radial_block[src_start + f];
                    }
                }
            } else {
                scatter_add_m_major_block_f32(
                    rows,
                    config.so2.coefficient_count(),
                    config.so2.in_channels(),
                    coeff_offset,
                    block_rows,
                    &grad_modulated,
                    &mut grad_input_m_major,
                );
            }
        }
        coeff_offset += block_rows;
        radial_offset += input_feature_dim;
    }

    let mut grad_input_l_temp = vec![0.0f32; input_l_major.len()];
    m_to_l_permutation_apply_f32(
        &permutation,
        config.so2.in_channels(),
        &grad_input_m_major,
        &mut grad_input_l_temp,
    );
    for (dst, src) in grad_input_l_major.iter_mut().zip(grad_input_l_temp.iter()) {
        *dst += src;
    }
    Ok(())
}

/// Variant of `gather_m_major_block_f32` that returns a newly allocated Vec.
fn gather_m_major_block_f32_alloc(
    rows: usize,
    coefficient_count: usize,
    channels: usize,
    coefficient_offset: usize,
    block_coefficients: usize,
    source: &[f32],
) -> Vec<f32> {
    let mut target = vec![0.0f32; rows * block_coefficients * channels];
    gather_m_major_block_f32(
        rows,
        coefficient_count,
        channels,
        coefficient_offset,
        block_coefficients,
        source,
        &mut target,
    );
    target
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_plus_epsilon_matches_fairchem_contract() {
        let logits = [0.0, 0.0, 2.0, -2.0];
        let mut out = [0.0; 4];
        assert!(fairchem_uma_mole_softmax_plus_epsilon_rows(&logits, 2, 2, &mut out).is_ok());

        assert_close(out[0], 0.505, 1.0e-15);
        assert_close(out[1], 0.505, 1.0e-15);
        let exp_pos = 2.0_f64.exp();
        let exp_neg = (-2.0_f64).exp();
        assert_close(
            out[2],
            exp_pos / (exp_pos + exp_neg) + UMA_MOLE_SOFTMAX_EPSILON,
            1.0e-15,
        );
        assert_close(
            out[3],
            exp_neg / (exp_pos + exp_neg) + UMA_MOLE_SOFTMAX_EPSILON,
            1.0e-15,
        );
    }

    #[test]
    fn routing_mlp_forward_applies_three_silu_layers_and_softmax_epsilon() {
        let config = UmaRoutingMlpConfig {
            rows: 1,
            input_dim: 2,
            hidden_dim: 2,
            experts: 2,
        };
        let input = [1.0, -1.0];
        let layer0_weight = [1.0, 0.0, 0.0, 1.0];
        let layer0_bias = [0.0, 0.0];
        let layer2_weight = [1.0, 0.0, 0.0, 1.0];
        let layer2_bias = [0.5, -0.5];
        let layer4_weight = [1.0, 0.0, 0.0, 1.0];
        let layer4_bias = [0.0, 0.0];
        let mut hidden0 = [0.0; 2];
        let mut hidden1 = [0.0; 2];
        let mut logits = [0.0; 2];
        let mut coefficients = [0.0; 2];

        assert!(fairchem_uma_routing_mlp_forward(
            config,
            &input,
            UmaRoutingMlpWeights {
                layer0_weight: &layer0_weight,
                layer0_bias: &layer0_bias,
                layer2_weight: &layer2_weight,
                layer2_bias: &layer2_bias,
                layer4_weight: &layer4_weight,
                layer4_bias: &layer4_bias,
            },
            UmaRoutingMlpWorkspaceMut {
                hidden0: &mut hidden0,
                hidden1: &mut hidden1,
                logits: &mut logits,
                coefficients: &mut coefficients,
            },
        )
        .is_ok());

        let expected_hidden0 = [local_silu(1.0), local_silu(-1.0)];
        let expected_hidden1 = [
            local_silu(expected_hidden0[0] + 0.5),
            local_silu(expected_hidden0[1] - 0.5),
        ];
        let expected_logits = [
            local_silu(expected_hidden1[0]),
            local_silu(expected_hidden1[1]),
        ];
        let exp0 = expected_logits[0].exp();
        let exp1 = expected_logits[1].exp();

        assert_close(hidden0[0], expected_hidden0[0], 1.0e-15);
        assert_close(hidden0[1], expected_hidden0[1], 1.0e-15);
        assert_close(hidden1[0], expected_hidden1[0], 1.0e-15);
        assert_close(hidden1[1], expected_hidden1[1], 1.0e-15);
        assert_close(logits[0], expected_logits[0], 1.0e-15);
        assert_close(logits[1], expected_logits[1], 1.0e-15);
        assert_close(
            coefficients[0],
            exp0 / (exp0 + exp1) + UMA_MOLE_SOFTMAX_EPSILON,
            1.0e-15,
        );
        assert_close(
            coefficients[1],
            exp1 / (exp0 + exp1) + UMA_MOLE_SOFTMAX_EPSILON,
            1.0e-15,
        );
    }

    #[test]
    fn routing_mlp_zero_weights_give_uniform_plus_epsilon_coefficients() {
        let config = UmaRoutingMlpConfig {
            rows: 2,
            input_dim: 3,
            hidden_dim: 4,
            experts: 5,
        };
        let input = [1.0; 6];
        let layer0_weight = [0.0; 12];
        let layer0_bias = [0.0; 4];
        let layer2_weight = [0.0; 16];
        let layer2_bias = [0.0; 4];
        let layer4_weight = [0.0; 20];
        let layer4_bias = [0.0; 5];
        let mut hidden0 = [1.0; 8];
        let mut hidden1 = [1.0; 8];
        let mut logits = [1.0; 10];
        let mut coefficients = [0.0; 10];

        assert!(fairchem_uma_routing_mlp_forward(
            config,
            &input,
            UmaRoutingMlpWeights {
                layer0_weight: &layer0_weight,
                layer0_bias: &layer0_bias,
                layer2_weight: &layer2_weight,
                layer2_bias: &layer2_bias,
                layer4_weight: &layer4_weight,
                layer4_bias: &layer4_bias,
            },
            UmaRoutingMlpWorkspaceMut {
                hidden0: &mut hidden0,
                hidden1: &mut hidden1,
                logits: &mut logits,
                coefficients: &mut coefficients,
            },
        )
        .is_ok());

        for value in coefficients {
            assert_close(value, 0.2 + UMA_MOLE_SOFTMAX_EPSILON, 1.0e-15);
        }
    }

    #[test]
    fn mole_linear_forward_matches_hand_computed_segmented_mix() {
        let config = UmaMoleLinearConfig {
            systems: 2,
            experts: 2,
            in_dim: 2,
            out_dim: 2,
        };
        let input = [1.0, 2.0, -1.0, 0.5, 3.0, -2.0];
        let expert_weights = [
            1.0, 2.0, 3.0, 4.0, // expert 0
            -1.0, 0.5, 2.0, -0.5, // expert 1
        ];
        let bias = [0.25, -0.75];
        let coefficients = [0.2, 0.8, 1.5, -0.5];
        let system_sizes = [2, 1];
        let mut out = [0.0; 6];

        assert!(fairchem_uma_mole_linear_forward(
            config,
            &input,
            &expert_weights,
            &bias,
            &coefficients,
            &system_sizes,
            &mut out,
        )
        .is_ok());

        let expected = [1.25, 2.25, 1.25, -2.75, 0.75, -2.75];
        assert_slice_close(&out, &expected, 1.0e-15);
    }

    #[test]
    fn mole_so2_convolution_matches_single_expert_esen_convolution() -> Result<()> {
        let so2 = EsenSo2ConvolutionConfig::new(1, 1, 1, 1, 1)?;
        let input = [
            1.0, 2.0, 3.0, -0.5, // node 0 l-major coefficients
            -1.0, 0.5, -0.25, 0.75, // node 1
        ];
        let radial = [
            0.5, 0.75, -0.25, // node 0 m0 then m1 radial
            1.25, -0.5, 0.9, // node 1
        ];
        let m0_weight = [
            0.1, 0.2, // extra m0 output
            0.3, 0.4, // l=0/l=1 m0 output
            -0.5, 0.25,
        ];
        let m0_bias = [0.01, -0.02, 0.03];
        let m1_weight = [0.7, -0.2];
        let mut esen_out = [0.0; 8];
        let mut esen_extra = [0.0; 2];
        let mut uma_out = [0.0; 8];
        let mut uma_extra = [0.0; 2];

        e3nn_core::so2::fairchem_esen_so2_convolution_forward(
            so2,
            &input,
            Some(&radial),
            &m0_weight,
            &m0_bias,
            &[&m1_weight],
            &mut esen_out,
            Some(&mut esen_extra),
        )?;
        fairchem_uma_mole_so2_convolution_forward(
            UmaMoleSo2ConvolutionConfig {
                systems: 1,
                experts: 1,
                so2,
            },
            &input,
            Some(&radial),
            &m0_weight,
            &m0_bias,
            &[&m1_weight],
            &[1.0],
            &[2],
            &mut uma_out,
            Some(&mut uma_extra),
        )?;

        assert_slice_close(&uma_out, &esen_out, 1.0e-15);
        assert_slice_close(&uma_extra, &esen_extra, 1.0e-15);
        Ok(())
    }

    #[test]
    fn mole_linear_vjp_input_matches_finite_difference() {
        let config = UmaMoleLinearConfig {
            systems: 2,
            experts: 2,
            in_dim: 2,
            out_dim: 2,
        };
        let input = [0.7, -1.1, 2.0, 0.25, -0.4, 1.3];
        let expert_weights = [0.5, -0.2, 1.1, 0.3, -0.7, 0.9, 0.4, -1.2];
        let bias = [0.1, -0.2];
        let coefficients = [0.3, 0.7, 1.2, -0.2];
        let system_sizes = [2, 1];
        let grad_out = [0.2, -0.5, 1.0, 0.4, -0.7, 0.9];
        let mut grad_input = [0.0; 6];

        assert!(fairchem_uma_mole_linear_vjp_input_add(
            config,
            &grad_out,
            &expert_weights,
            &coefficients,
            &system_sizes,
            &mut grad_input,
        )
        .is_ok());

        let eps = 1.0e-6;
        for index in 0..input.len() {
            let mut plus = input;
            let mut minus = input;
            plus[index] += eps;
            minus[index] -= eps;
            let fd = (objective(
                config,
                &plus,
                &expert_weights,
                &bias,
                &coefficients,
                &system_sizes,
                &grad_out,
            ) - objective(
                config,
                &minus,
                &expert_weights,
                &bias,
                &coefficients,
                &system_sizes,
                &grad_out,
            )) / (2.0 * eps);
            assert_close(grad_input[index], fd, 1.0e-9);
        }
    }

    fn objective(
        config: UmaMoleLinearConfig,
        input: &[f64],
        expert_weights: &[f64],
        bias: &[f64],
        coefficients: &[f64],
        system_sizes: &[usize],
        grad_out: &[f64],
    ) -> f64 {
        let mut out = vec![0.0; grad_out.len()];
        let result = fairchem_uma_mole_linear_forward(
            config,
            input,
            expert_weights,
            bias,
            coefficients,
            system_sizes,
            &mut out,
        );
        assert!(result.is_ok());
        out.iter()
            .zip(grad_out.iter())
            .map(|(value, grad)| value * grad)
            .sum()
    }

    #[test]
    fn so2_convolution_forward_f32_matches_f64_single_system() -> Result<()> {
        let so2 = EsenSo2ConvolutionConfig::new(1, 1, 2, 2, 1)?;
        let config = UmaMoleSo2ConvolutionConfig {
            systems: 1,
            experts: 2,
            so2,
        };
        // input: [2 nodes, 4 coefficients * 2 channels] = [2, 8]
        let input_f64: Vec<f64> = (0..16).map(|i| (i as f64 - 7.5) * 0.1).collect();
        let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();
        // radial: [2 nodes, radial_node_stride]
        let radial_stride = so2.radial_node_stride();
        let radial_f64: Vec<f64> = (0..2 * radial_stride)
            .map(|i| (i as f64 * 0.07 - 0.3).sin())
            .collect();
        let radial_f32: Vec<f32> = radial_f64.iter().map(|&x| x as f32).collect();
        // m0 expert weights: [2 experts, m0_weight_len]
        let m0_weight_len = so2.m0_weight_len();
        let m0_expert_f64: Vec<f64> = (0..2 * m0_weight_len)
            .map(|i| (i as f64 * 0.13 - 1.0) * 0.1)
            .collect();
        let m0_expert_f32: Vec<f32> = m0_expert_f64.iter().map(|&x| x as f32).collect();
        let m0_bias_f64: Vec<f64> = (0..so2.m0_bias_len())
            .map(|i| (i as f64 * 0.17 - 0.5) * 0.05)
            .collect();
        let m0_bias_f32: Vec<f32> = m0_bias_f64.iter().map(|&x| x as f32).collect();
        // m1 expert weights: [2 experts, m1_conv_weight_len]
        let m1_weight_len = so2.m_conv_weight_len(1);
        let m1_expert_f64: Vec<f64> = (0..2 * m1_weight_len)
            .map(|i| (i as f64 * 0.11 + 0.2) * 0.1)
            .collect();
        let m1_expert_f32: Vec<f32> = m1_expert_f64.iter().map(|&x| x as f32).collect();
        // coefficients: [1 system * 2 experts]
        let coefficients_f64 = [0.6f64, 0.4];
        let coefficients_f32: Vec<f32> = coefficients_f64.iter().map(|&x| x as f32).collect();
        let system_sizes = [2usize];

        let output_stride = so2.output_node_stride();
        let extra_channels = so2.extra_m0_output_channels();
        let mut out_f64 = vec![0.0f64; 2 * output_stride];
        let mut extra_f64 = vec![0.0f64; 2 * extra_channels];
        let mut out_f32 = vec![0.0f32; 2 * output_stride];
        let mut extra_f32 = vec![0.0f32; 2 * extra_channels];

        fairchem_uma_mole_so2_convolution_forward(
            config,
            &input_f64,
            Some(&radial_f64),
            &m0_expert_f64,
            &m0_bias_f64,
            &[&m1_expert_f64],
            &coefficients_f64,
            &system_sizes,
            &mut out_f64,
            Some(&mut extra_f64),
        )?;
        fairchem_uma_mole_so2_convolution_forward_f32(
            config,
            &input_f32,
            Some(&radial_f32),
            &m0_expert_f32,
            &m0_bias_f32,
            &[&m1_expert_f32],
            &coefficients_f32,
            &system_sizes,
            &mut out_f32,
            Some(&mut extra_f32),
        )?;

        // f32 results should be close to the f64 values when cast to f32.
        for (actual, expected) in out_f32.iter().zip(out_f64.iter()) {
            let expected_f32 = *expected as f32;
            let diff = (actual - expected_f32).abs();
            assert!(
                diff <= 1.0e-5_f32,
                "out mismatch: actual={actual} expected={expected_f32} diff={diff}"
            );
        }
        for (actual, expected) in extra_f32.iter().zip(extra_f64.iter()) {
            let expected_f32 = *expected as f32;
            let diff = (actual - expected_f32).abs();
            assert!(
                diff <= 1.0e-5_f32,
                "extra mismatch: actual={actual} expected={expected_f32} diff={diff}"
            );
        }
        Ok(())
    }

    #[test]
    fn so2_convolution_vjp_f32_matches_f64_single_system() -> Result<()> {
        let so2 = EsenSo2ConvolutionConfig::new(1, 1, 2, 2, 0)?;
        let config = UmaMoleSo2ConvolutionConfig {
            systems: 1,
            experts: 2,
            so2,
        };
        let input_f64: Vec<f64> = (0..16).map(|i| (i as f64 - 7.5) * 0.1).collect();
        let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();
        let radial_stride = so2.radial_node_stride();
        let radial_f64: Vec<f64> = (0..2 * radial_stride)
            .map(|i| (i as f64 * 0.07 - 0.3).sin())
            .collect();
        let radial_f32: Vec<f32> = radial_f64.iter().map(|&x| x as f32).collect();
        let m0_weight_len = so2.m0_weight_len();
        let m0_expert_f64: Vec<f64> = (0..2 * m0_weight_len)
            .map(|i| (i as f64 * 0.13 - 1.0) * 0.1)
            .collect();
        let m0_expert_f32: Vec<f32> = m0_expert_f64.iter().map(|&x| x as f32).collect();
        let m1_weight_len = so2.m_conv_weight_len(1);
        let m1_expert_f64: Vec<f64> = (0..2 * m1_weight_len)
            .map(|i| (i as f64 * 0.11 + 0.2) * 0.1)
            .collect();
        let m1_expert_f32: Vec<f32> = m1_expert_f64.iter().map(|&x| x as f32).collect();
        let coefficients_f64 = [0.6f64, 0.4];
        let coefficients_f32: Vec<f32> = coefficients_f64.iter().map(|&x| x as f32).collect();
        let system_sizes = [2usize];

        let output_stride = so2.output_node_stride();
        let grad_out_f64: Vec<f64> = (0..2 * output_stride)
            .map(|i| (i as f64 * 0.09 - 0.8) * 0.1)
            .collect();
        let grad_out_f32: Vec<f32> = grad_out_f64.iter().map(|&x| x as f32).collect();
        let mut grad_input_f64 = vec![0.0f64; input_f64.len()];
        let mut grad_radial_f64 = vec![0.0f64; radial_f64.len()];
        let mut grad_input_f32 = vec![0.0f32; input_f32.len()];
        let mut grad_radial_f32 = vec![0.0f32; radial_f32.len()];

        fairchem_uma_mole_so2_convolution_vjp(
            config,
            &input_f64,
            Some(&radial_f64),
            &m0_expert_f64,
            &[&m1_expert_f64],
            &coefficients_f64,
            &system_sizes,
            &grad_out_f64,
            None,
            &mut grad_input_f64,
            Some(&mut grad_radial_f64),
        )?;
        fairchem_uma_mole_so2_convolution_vjp_f32(
            config,
            &input_f32,
            Some(&radial_f32),
            &m0_expert_f32,
            &[&m1_expert_f32],
            &coefficients_f32,
            &system_sizes,
            &grad_out_f32,
            None,
            &mut grad_input_f32,
            Some(&mut grad_radial_f32),
        )?;

        for (actual, expected) in grad_input_f32.iter().zip(grad_input_f64.iter()) {
            let expected_f32 = *expected as f32;
            let diff = (actual - expected_f32).abs();
            assert!(
                diff <= 1.0e-4_f32,
                "grad_input mismatch: actual={actual} expected={expected_f32} diff={diff}"
            );
        }
        for (actual, expected) in grad_radial_f32.iter().zip(grad_radial_f64.iter()) {
            let expected_f32 = *expected as f32;
            let diff = (actual - expected_f32).abs();
            assert!(
                diff <= 1.0e-4_f32,
                "grad_radial mismatch: actual={actual} expected={expected_f32} diff={diff}"
            );
        }
        Ok(())
    }

    fn assert_slice_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_close(*actual, *expected, tol);
        }
    }

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tol,
            "actual={actual:?} expected={expected:?} diff={diff:?} tol={tol:?}"
        );
    }

    fn local_silu(value: f64) -> f64 {
        value / (1.0 + (-value).exp())
    }
}
