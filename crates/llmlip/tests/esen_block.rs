#![allow(clippy::too_many_arguments, clippy::unwrap_used)]

use e3nn_core::mlp::silu;
use e3nn_core::normalization::{
    fairchem_esen_rms_norm_sh_forward, EsenRmsNormShConfig, EsenRmsNormShNormalization,
};
use llmlip::esen_block::{
    fairchem_esen_edgewise_gate_forward, fairchem_esen_edgewise_gate_vjp,
    fairchem_esen_grid_atomwise_forward, fairchem_esen_grid_atomwise_vjp,
    fairchem_esen_grid_block_forward, fairchem_esen_grid_block_vjp,
    fairchem_esen_spectral_atomwise_forward, fairchem_esen_spectral_atomwise_forward_f32_blas_gate,
    fairchem_esen_spectral_atomwise_vjp, fairchem_esen_spectral_atomwise_vjp_write_f32_blas_gate,
    EsenEdgewiseGateConfig, EsenGridAtomwiseConfig, EsenGridBlockConfig,
    EsenSpectralAtomwiseConfig,
};

#[test]
fn edgewise_gate_forward_accumulates_only_target_nodes() {
    let fixture = EdgewiseFixture::new();
    let mut out = vec![99.0; fixture.node_features.len()];

    fixture.forward(&mut out);

    let node_stride = fixture.config.node_feature_dim();
    assert!(out[..node_stride].iter().all(|value| *value == 0.0));
    assert!(out[node_stride..2 * node_stride]
        .iter()
        .any(|value| value.abs() > 1.0e-12));
    assert!(out[2 * node_stride..3 * node_stride]
        .iter()
        .any(|value| value.abs() > 1.0e-12));
}

#[test]
fn edgewise_gate_vjp_matches_selected_finite_differences() {
    let mut fixture = EdgewiseFixture::new();
    let mut grad_out = vec![0.0; fixture.node_features.len()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.17 * index as f64 - 0.3).sin();
    }

    let mut grad_node_features = vec![0.0; fixture.node_features.len()];
    let mut grad_wigner = vec![0.0; fixture.wigner.len()];
    let mut grad_conv1_radial = vec![0.0; fixture.conv1_radial.len()];
    let mut grad_conv2_radial = vec![0.0; fixture.conv2_radial.len()];
    let mut grad_edge_distances = vec![0.0; fixture.edge_distances.len()];
    let mut grad_wigner_inv = vec![0.0; fixture.wigner_inv.len()];

    fairchem_esen_edgewise_gate_vjp(
        fixture.config,
        &fixture.edge_sources,
        &fixture.edge_targets,
        &fixture.node_features,
        &fixture.wigner,
        Some(&fixture.conv1_radial),
        &fixture.conv1_m0_weight,
        &fixture.conv1_m0_bias,
        &[&fixture.conv1_m1_weight],
        Some(&fixture.conv2_radial),
        &fixture.conv2_m0_weight,
        &fixture.conv2_m0_bias,
        &[&fixture.conv2_m1_weight],
        Some(&fixture.edge_distances),
        &fixture.wigner_inv,
        &grad_out,
        &mut grad_node_features,
        &mut grad_wigner,
        Some(&mut grad_conv1_radial),
        Some(&mut grad_conv2_radial),
        Some(&mut grad_edge_distances),
        &mut grad_wigner_inv,
    )
    .unwrap();

    assert_fd(
        grad_node_features[3],
        finite_difference(&mut fixture, &grad_out, |fixture, delta| {
            fixture.node_features[3] += delta;
        }),
        "node_features[3]",
    );
    assert_fd(
        grad_wigner[5],
        finite_difference(&mut fixture, &grad_out, |fixture, delta| {
            fixture.wigner[5] += delta;
        }),
        "wigner[5]",
    );
    assert_fd(
        grad_conv1_radial[2],
        finite_difference(&mut fixture, &grad_out, |fixture, delta| {
            fixture.conv1_radial[2] += delta;
        }),
        "conv1_radial[2]",
    );
    assert_fd(
        grad_conv2_radial[1],
        finite_difference(&mut fixture, &grad_out, |fixture, delta| {
            fixture.conv2_radial[1] += delta;
        }),
        "conv2_radial[1]",
    );
    assert_fd(
        grad_edge_distances[0],
        finite_difference(&mut fixture, &grad_out, |fixture, delta| {
            fixture.edge_distances[0] += delta;
        }),
        "edge_distances[0]",
    );
    assert_fd(
        grad_wigner_inv[8],
        finite_difference(&mut fixture, &grad_out, |fixture, delta| {
            fixture.wigner_inv[8] += delta;
        }),
        "wigner_inv[8]",
    );
}

#[test]
fn grid_atomwise_forward_matches_independent_formula() {
    let fixture = GridAtomwiseFixture::new();
    let mut out = vec![0.0; fixture.input.len()];

    fixture.forward(&mut out);

    let expected = independent_grid_atomwise_forward(&fixture);
    assert_eq!(out.len(), expected.len());
    for (index, (actual, expected)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "index={index}, actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn grid_atomwise_vjp_matches_finite_difference() {
    let mut fixture = GridAtomwiseFixture::new();
    let mut grad_out = vec![0.0; fixture.input.len()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.13 * index as f64 - 0.2).cos();
    }
    let mut grad_input = vec![0.0; fixture.input.len()];

    fairchem_esen_grid_atomwise_vjp(
        fixture.config,
        &fixture.input,
        &fixture.to_grid,
        &fixture.from_grid,
        &fixture.weight1,
        &fixture.weight2,
        &fixture.weight3,
        &grad_out,
        &mut grad_input,
    )
    .unwrap();

    let step = 1.0e-6;
    for (index, &analytic) in grad_input.iter().enumerate() {
        let original = fixture.input[index];
        fixture.input[index] = original + step;
        let plus = fixture.objective(&grad_out);
        fixture.input[index] = original - step;
        let minus = fixture.objective(&grad_out);
        fixture.input[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (analytic - fd).abs() <= 2.0e-8,
            "input index={index}, grad={}, fd={fd}",
            analytic
        );
    }
}

#[test]
fn spectral_atomwise_forward_matches_independent_formula() {
    let fixture = SpectralAtomwiseFixture::new();
    let mut out = vec![0.0; fixture.input.len()];

    fixture.forward(&mut out);

    let expected = independent_spectral_atomwise_forward(&fixture);
    assert_eq!(out.len(), expected.len());
    for (index, (actual, expected)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "index={index}, actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn spectral_atomwise_vjp_matches_finite_difference() {
    let mut fixture = SpectralAtomwiseFixture::new();
    let mut grad_out = vec![0.0; fixture.input.len()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.09 * index as f64 + 0.3).sin();
    }
    let mut grad_input = vec![0.0; fixture.input.len()];

    fairchem_esen_spectral_atomwise_vjp(
        fixture.config,
        &fixture.input,
        &fixture.scalar_gate_weight,
        &fixture.scalar_gate_bias,
        &fixture.linear1_weight,
        &fixture.linear1_bias,
        &fixture.linear2_weight,
        &fixture.linear2_bias,
        &grad_out,
        &mut grad_input,
    )
    .unwrap();

    let step = 1.0e-6;
    for (index, &analytic) in grad_input.iter().enumerate() {
        let original = fixture.input[index];
        fixture.input[index] = original + step;
        let plus = fixture.objective(&grad_out);
        fixture.input[index] = original - step;
        let minus = fixture.objective(&grad_out);
        fixture.input[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (analytic - fd).abs() <= 3.0e-8,
            "input index={index}, grad={analytic}, fd={fd}"
        );
    }
}

#[test]
fn spectral_atomwise_f32_forward_matches_f64_reference_at_blas_gate_size() {
    let config = EsenSpectralAtomwiseConfig::new(2, 8, 8).unwrap();
    let nodes = 16usize;
    let input_f32 = values_f32(nodes * config.input_node_stride(), 0.013, -0.4);
    let scalar_gate_weight_f32 = values_f32(config.scalar_gate_weight_len(), -0.017, 0.2);
    let scalar_gate_bias_f32 = values_f32(config.scalar_gate_bias_len(), 0.019, -0.1);
    let linear1_weight_f32 = values_f32(
        config.so3_linear1_config().unwrap().weight_len(),
        0.011,
        0.3,
    );
    let linear1_bias_f32 = values_f32(config.hidden_channels(), -0.023, 0.1);
    let linear2_weight_f32 = values_f32(
        config.so3_linear2_config().unwrap().weight_len(),
        -0.007,
        -0.2,
    );
    let linear2_bias_f32 = values_f32(config.sphere_channels(), 0.029, 0.05);

    let input_f64 = to_f64(&input_f32);
    let scalar_gate_weight_f64 = to_f64(&scalar_gate_weight_f32);
    let scalar_gate_bias_f64 = to_f64(&scalar_gate_bias_f32);
    let linear1_weight_f64 = to_f64(&linear1_weight_f32);
    let linear1_bias_f64 = to_f64(&linear1_bias_f32);
    let linear2_weight_f64 = to_f64(&linear2_weight_f32);
    let linear2_bias_f64 = to_f64(&linear2_bias_f32);

    let mut actual = vec![0.0_f32; input_f32.len()];
    let mut expected = vec![0.0_f64; input_f64.len()];
    fairchem_esen_spectral_atomwise_forward_f32_blas_gate(
        config,
        &input_f32,
        &scalar_gate_weight_f32,
        &scalar_gate_bias_f32,
        &linear1_weight_f32,
        &linear1_bias_f32,
        &linear2_weight_f32,
        &linear2_bias_f32,
        &mut actual,
    )
    .unwrap();
    fairchem_esen_spectral_atomwise_forward(
        config,
        &input_f64,
        &scalar_gate_weight_f64,
        &scalar_gate_bias_f64,
        &linear1_weight_f64,
        &linear1_bias_f64,
        &linear2_weight_f64,
        &linear2_bias_f64,
        &mut expected,
    )
    .unwrap();

    assert_f32_matches_f64(&actual, &expected, 8.0e-5, 2.0e-5);
}

#[test]
fn spectral_atomwise_f32_vjp_matches_f64_reference_at_blas_gate_size() {
    let config = EsenSpectralAtomwiseConfig::new(2, 8, 8).unwrap();
    let nodes = 16usize;
    let input_f32 = values_f32(nodes * config.input_node_stride(), 0.013, -0.4);
    let scalar_gate_weight_f32 = values_f32(config.scalar_gate_weight_len(), -0.017, 0.2);
    let scalar_gate_bias_f32 = values_f32(config.scalar_gate_bias_len(), 0.019, -0.1);
    let linear1_weight_f32 = values_f32(
        config.so3_linear1_config().unwrap().weight_len(),
        0.011,
        0.3,
    );
    let linear1_bias_f32 = values_f32(config.hidden_channels(), -0.023, 0.1);
    let linear2_weight_f32 = values_f32(
        config.so3_linear2_config().unwrap().weight_len(),
        -0.007,
        -0.2,
    );
    let linear2_bias_f32 = values_f32(config.sphere_channels(), 0.029, 0.05);
    let grad_out_f32 = values_f32(input_f32.len(), -0.031, 0.7);

    let input_f64 = to_f64(&input_f32);
    let scalar_gate_weight_f64 = to_f64(&scalar_gate_weight_f32);
    let scalar_gate_bias_f64 = to_f64(&scalar_gate_bias_f32);
    let linear1_weight_f64 = to_f64(&linear1_weight_f32);
    let linear1_bias_f64 = to_f64(&linear1_bias_f32);
    let linear2_weight_f64 = to_f64(&linear2_weight_f32);
    let linear2_bias_f64 = to_f64(&linear2_bias_f32);
    let grad_out_f64 = to_f64(&grad_out_f32);

    let mut actual = vec![0.0_f32; input_f32.len()];
    let mut expected = vec![0.0_f64; input_f64.len()];
    fairchem_esen_spectral_atomwise_vjp_write_f32_blas_gate(
        config,
        &input_f32,
        &scalar_gate_weight_f32,
        &scalar_gate_bias_f32,
        &linear1_weight_f32,
        &linear1_bias_f32,
        &linear2_weight_f32,
        &linear2_bias_f32,
        &grad_out_f32,
        &mut actual,
    )
    .unwrap();
    fairchem_esen_spectral_atomwise_vjp(
        config,
        &input_f64,
        &scalar_gate_weight_f64,
        &scalar_gate_bias_f64,
        &linear1_weight_f64,
        &linear1_bias_f64,
        &linear2_weight_f64,
        &linear2_bias_f64,
        &grad_out_f64,
        &mut expected,
    )
    .unwrap();

    assert_f32_matches_f64(&actual, &expected, 1.5e-4, 3.0e-5);
}

#[test]
fn grid_block_forward_matches_manual_composition() {
    let fixture = GridBlockFixture::new();
    let mut out = vec![0.0; fixture.node_features.len()];

    fixture.forward(&mut out);

    let expected = fixture.manual_forward();
    assert_eq!(out.len(), expected.len());
    for (index, (actual, expected)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "index={index}, actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn grid_block_vjp_matches_selected_finite_differences() {
    let mut fixture = GridBlockFixture::new();
    let mut grad_out = vec![0.0; fixture.node_features.len()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.071 * index as f64 - 0.25).cos();
    }

    let mut grad_node_features = vec![0.0; fixture.node_features.len()];
    let mut grad_wigner = vec![0.0; fixture.wigner.len()];
    let mut grad_conv1_radial = vec![0.0; fixture.conv1_radial.len()];
    let mut grad_conv2_radial = vec![0.0; fixture.conv2_radial.len()];
    let mut grad_edge_distances = vec![0.0; fixture.edge_distances.len()];
    let mut grad_wigner_inv = vec![0.0; fixture.wigner_inv.len()];

    fairchem_esen_grid_block_vjp(
        fixture.config,
        &fixture.edge_sources,
        &fixture.edge_targets,
        &fixture.node_features,
        Some(&fixture.norm1_weight),
        Some(&fixture.norm1_bias),
        &fixture.wigner,
        Some(&fixture.conv1_radial),
        &fixture.conv1_m0_weight,
        &fixture.conv1_m0_bias,
        &[&fixture.conv1_m1_weight],
        Some(&fixture.conv2_radial),
        &fixture.conv2_m0_weight,
        &fixture.conv2_m0_bias,
        &[&fixture.conv2_m1_weight],
        Some(&fixture.edge_distances),
        &fixture.wigner_inv,
        Some(&fixture.norm2_weight),
        Some(&fixture.norm2_bias),
        &fixture.to_grid,
        &fixture.from_grid,
        &fixture.grid_weight1,
        &fixture.grid_weight2,
        &fixture.grid_weight3,
        &grad_out,
        &mut grad_node_features,
        &mut grad_wigner,
        Some(&mut grad_conv1_radial),
        Some(&mut grad_conv2_radial),
        Some(&mut grad_edge_distances),
        &mut grad_wigner_inv,
    )
    .unwrap();

    assert_fd(
        grad_node_features[2],
        finite_difference_grid_block(&mut fixture, &grad_out, |fixture, delta| {
            fixture.node_features[2] += delta;
        }),
        "grid_block node_features[2]",
    );
    assert_fd(
        grad_wigner[6],
        finite_difference_grid_block(&mut fixture, &grad_out, |fixture, delta| {
            fixture.wigner[6] += delta;
        }),
        "grid_block wigner[6]",
    );
    assert_fd(
        grad_conv1_radial[1],
        finite_difference_grid_block(&mut fixture, &grad_out, |fixture, delta| {
            fixture.conv1_radial[1] += delta;
        }),
        "grid_block conv1_radial[1]",
    );
    assert_fd(
        grad_conv2_radial[2],
        finite_difference_grid_block(&mut fixture, &grad_out, |fixture, delta| {
            fixture.conv2_radial[2] += delta;
        }),
        "grid_block conv2_radial[2]",
    );
    assert_fd(
        grad_edge_distances[1],
        finite_difference_grid_block(&mut fixture, &grad_out, |fixture, delta| {
            fixture.edge_distances[1] += delta;
        }),
        "grid_block edge_distances[1]",
    );
    assert_fd(
        grad_wigner_inv[9],
        finite_difference_grid_block(&mut fixture, &grad_out, |fixture, delta| {
            fixture.wigner_inv[9] += delta;
        }),
        "grid_block wigner_inv[9]",
    );
}

struct EdgewiseFixture {
    config: EsenEdgewiseGateConfig,
    edge_sources: Vec<usize>,
    edge_targets: Vec<usize>,
    node_features: Vec<f64>,
    wigner: Vec<f64>,
    conv1_radial: Vec<f64>,
    conv1_m0_weight: Vec<f64>,
    conv1_m0_bias: Vec<f64>,
    conv1_m1_weight: Vec<f64>,
    conv2_radial: Vec<f64>,
    conv2_m0_weight: Vec<f64>,
    conv2_m0_bias: Vec<f64>,
    conv2_m1_weight: Vec<f64>,
    edge_distances: Vec<f64>,
    wigner_inv: Vec<f64>,
}

struct GridAtomwiseFixture {
    config: EsenGridAtomwiseConfig,
    input: Vec<f64>,
    to_grid: Vec<f64>,
    from_grid: Vec<f64>,
    weight1: Vec<f64>,
    weight2: Vec<f64>,
    weight3: Vec<f64>,
}

struct SpectralAtomwiseFixture {
    config: EsenSpectralAtomwiseConfig,
    input: Vec<f64>,
    scalar_gate_weight: Vec<f64>,
    scalar_gate_bias: Vec<f64>,
    linear1_weight: Vec<f64>,
    linear1_bias: Vec<f64>,
    linear2_weight: Vec<f64>,
    linear2_bias: Vec<f64>,
}

struct GridBlockFixture {
    config: EsenGridBlockConfig,
    edge_sources: Vec<usize>,
    edge_targets: Vec<usize>,
    node_features: Vec<f64>,
    norm1_weight: Vec<f64>,
    norm1_bias: Vec<f64>,
    wigner: Vec<f64>,
    conv1_radial: Vec<f64>,
    conv1_m0_weight: Vec<f64>,
    conv1_m0_bias: Vec<f64>,
    conv1_m1_weight: Vec<f64>,
    conv2_radial: Vec<f64>,
    conv2_m0_weight: Vec<f64>,
    conv2_m0_bias: Vec<f64>,
    conv2_m1_weight: Vec<f64>,
    edge_distances: Vec<f64>,
    wigner_inv: Vec<f64>,
    norm2_weight: Vec<f64>,
    norm2_bias: Vec<f64>,
    to_grid: Vec<f64>,
    from_grid: Vec<f64>,
    grid_weight1: Vec<f64>,
    grid_weight2: Vec<f64>,
    grid_weight3: Vec<f64>,
}

impl GridAtomwiseFixture {
    fn new() -> Self {
        let config = EsenGridAtomwiseConfig::new(3, 2, 3, 2, 2).unwrap();
        Self {
            config,
            input: values(2 * config.coefficient_node_stride(), 0.07, -0.5),
            to_grid: values(config.grid_config().matrix_len(), 0.11, 0.2),
            from_grid: values(config.grid_config().matrix_len(), -0.09, 0.6),
            weight1: values(config.weight1_len(), 0.05, -0.3),
            weight2: values(config.weight2_len(), -0.04, 0.4),
            weight3: values(config.weight3_len(), 0.06, -0.1),
        }
    }

    fn forward(&self, out: &mut [f64]) {
        fairchem_esen_grid_atomwise_forward(
            self.config,
            &self.input,
            &self.to_grid,
            &self.from_grid,
            &self.weight1,
            &self.weight2,
            &self.weight3,
            out,
        )
        .unwrap();
    }

    fn objective(&self, grad_out: &[f64]) -> f64 {
        let mut out = vec![0.0; self.input.len()];
        self.forward(&mut out);
        out.iter()
            .zip(grad_out.iter())
            .map(|(value, grad)| value * grad)
            .sum()
    }
}

impl SpectralAtomwiseFixture {
    fn new() -> Self {
        let config = EsenSpectralAtomwiseConfig::new(2, 2, 3).unwrap();
        Self {
            config,
            input: values(2 * config.input_node_stride(), 0.04, -0.6),
            scalar_gate_weight: values(config.scalar_gate_weight_len(), 0.05, 0.2),
            scalar_gate_bias: values(config.scalar_gate_bias_len(), -0.03, 0.1),
            linear1_weight: values(
                config.so3_linear1_config().unwrap().weight_len(),
                0.06,
                -0.4,
            ),
            linear1_bias: values(config.hidden_channels(), 0.07, 0.3),
            linear2_weight: values(
                config.so3_linear2_config().unwrap().weight_len(),
                -0.05,
                0.5,
            ),
            linear2_bias: values(config.sphere_channels(), -0.02, -0.1),
        }
    }

    fn forward(&self, out: &mut [f64]) {
        fairchem_esen_spectral_atomwise_forward(
            self.config,
            &self.input,
            &self.scalar_gate_weight,
            &self.scalar_gate_bias,
            &self.linear1_weight,
            &self.linear1_bias,
            &self.linear2_weight,
            &self.linear2_bias,
            out,
        )
        .unwrap();
    }

    fn objective(&self, grad_out: &[f64]) -> f64 {
        let mut out = vec![0.0; self.input.len()];
        self.forward(&mut out);
        out.iter()
            .zip(grad_out.iter())
            .map(|(value, grad)| value * grad)
            .sum()
    }
}

impl GridBlockFixture {
    fn new() -> Self {
        let edge = EsenEdgewiseGateConfig::new(3, 1, 1, 1, 1, 0, true, 4.0, 5).unwrap();
        let norm1 =
            EsenRmsNormShConfig::new(1, 1, 1.0e-6, EsenRmsNormShNormalization::Component, true)
                .unwrap();
        let norm2 =
            EsenRmsNormShConfig::new(1, 1, 1.0e-6, EsenRmsNormShNormalization::Component, true)
                .unwrap();
        let atomwise = EsenGridAtomwiseConfig::new(edge.coefficient_count(), 1, 2, 2, 2).unwrap();
        let config = EsenGridBlockConfig::new(norm1, norm2, edge, atomwise).unwrap();
        let conv1_config = edge.conv1_config().unwrap();
        let conv2_config = edge.conv2_config().unwrap();
        Self {
            config,
            edge_sources: vec![0, 1],
            edge_targets: vec![1, 2],
            node_features: values(config.num_nodes() * config.node_stride(), 0.043, -0.7),
            norm1_weight: values(2, 0.031, 1.2),
            norm1_bias: values(1, 0.017, -0.1),
            wigner: matrix_values(2, edge.coefficient_count(), 0.021),
            conv1_radial: values(2 * conv1_config.radial_node_stride(), 0.067, 0.4),
            conv1_m0_weight: values(conv1_config.m0_weight_len(), 0.052, -0.2),
            conv1_m0_bias: values(conv1_config.m0_bias_len(), 0.019, 0.1),
            conv1_m1_weight: values(conv1_config.m_conv_weight_len(1), -0.044, 0.3),
            conv2_radial: values(2 * conv2_config.radial_node_stride(), -0.061, 0.8),
            conv2_m0_weight: values(conv2_config.m0_weight_len(), 0.083, -0.1),
            conv2_m0_bias: values(conv2_config.m0_bias_len(), -0.026, 0.05),
            conv2_m1_weight: values(conv2_config.m_conv_weight_len(1), 0.097, 0.2),
            edge_distances: vec![1.1, 2.0],
            wigner_inv: matrix_values(2, edge.coefficient_count(), -0.018),
            norm2_weight: values(2, -0.027, 0.9),
            norm2_bias: values(1, 0.023, 0.2),
            to_grid: values(atomwise.grid_config().matrix_len(), 0.037, -0.3),
            from_grid: values(atomwise.grid_config().matrix_len(), -0.041, 0.5),
            grid_weight1: values(atomwise.weight1_len(), 0.071, -0.4),
            grid_weight2: values(atomwise.weight2_len(), -0.039, 0.2),
            grid_weight3: values(atomwise.weight3_len(), 0.063, -0.2),
        }
    }

    fn forward(&self, out: &mut [f64]) {
        fairchem_esen_grid_block_forward(
            self.config,
            &self.edge_sources,
            &self.edge_targets,
            &self.node_features,
            Some(&self.norm1_weight),
            Some(&self.norm1_bias),
            &self.wigner,
            Some(&self.conv1_radial),
            &self.conv1_m0_weight,
            &self.conv1_m0_bias,
            &[&self.conv1_m1_weight],
            Some(&self.conv2_radial),
            &self.conv2_m0_weight,
            &self.conv2_m0_bias,
            &[&self.conv2_m1_weight],
            Some(&self.edge_distances),
            &self.wigner_inv,
            Some(&self.norm2_weight),
            Some(&self.norm2_bias),
            &self.to_grid,
            &self.from_grid,
            &self.grid_weight1,
            &self.grid_weight2,
            &self.grid_weight3,
            out,
        )
        .unwrap();
    }

    fn manual_forward(&self) -> Vec<f64> {
        let mut norm1_out = vec![0.0; self.node_features.len()];
        fairchem_esen_rms_norm_sh_forward(
            self.config.norm1_config(),
            &self.node_features,
            Some(&self.norm1_weight),
            Some(&self.norm1_bias),
            &mut norm1_out,
        )
        .unwrap();
        let mut edge_out = vec![0.0; self.node_features.len()];
        fairchem_esen_edgewise_gate_forward(
            self.config.edge_config(),
            &self.edge_sources,
            &self.edge_targets,
            &norm1_out,
            &self.wigner,
            Some(&self.conv1_radial),
            &self.conv1_m0_weight,
            &self.conv1_m0_bias,
            &[&self.conv1_m1_weight],
            Some(&self.conv2_radial),
            &self.conv2_m0_weight,
            &self.conv2_m0_bias,
            &[&self.conv2_m1_weight],
            Some(&self.edge_distances),
            &self.wigner_inv,
            &mut edge_out,
        )
        .unwrap();
        let mut after_edge = vec![0.0; self.node_features.len()];
        for ((target, input), edge) in after_edge
            .iter_mut()
            .zip(self.node_features.iter())
            .zip(edge_out.iter())
        {
            *target = input + edge;
        }
        let mut norm2_out = vec![0.0; self.node_features.len()];
        fairchem_esen_rms_norm_sh_forward(
            self.config.norm2_config(),
            &after_edge,
            Some(&self.norm2_weight),
            Some(&self.norm2_bias),
            &mut norm2_out,
        )
        .unwrap();
        let mut atomwise_out = vec![0.0; self.node_features.len()];
        fairchem_esen_grid_atomwise_forward(
            self.config.atomwise_config(),
            &norm2_out,
            &self.to_grid,
            &self.from_grid,
            &self.grid_weight1,
            &self.grid_weight2,
            &self.grid_weight3,
            &mut atomwise_out,
        )
        .unwrap();
        for (value, residual) in atomwise_out.iter_mut().zip(after_edge.iter()) {
            *value += residual;
        }
        atomwise_out
    }

    fn objective(&self, grad_out: &[f64]) -> f64 {
        let mut out = vec![0.0; self.node_features.len()];
        self.forward(&mut out);
        out.iter()
            .zip(grad_out.iter())
            .map(|(value, grad)| value * grad)
            .sum()
    }
}

impl EdgewiseFixture {
    fn new() -> Self {
        let config = EsenEdgewiseGateConfig::new(3, 1, 1, 1, 1, 0, true, 4.0, 5).unwrap();
        let edge_sources = vec![0, 1];
        let edge_targets = vec![1, 2];
        let node_features = values(config.num_nodes() * config.node_feature_dim(), 0.11, -0.4);
        let wigner = matrix_values(2, config.coefficient_count(), 0.03);
        let wigner_inv = matrix_values(2, config.coefficient_count(), -0.02);
        let conv1_config = config.conv1_config().unwrap();
        let conv2_config = config.conv2_config().unwrap();
        Self {
            config,
            edge_sources,
            edge_targets,
            node_features,
            wigner,
            conv1_radial: values(2 * conv1_config.radial_node_stride(), 0.07, 0.6),
            conv1_m0_weight: values(conv1_config.m0_weight_len(), 0.05, -0.2),
            conv1_m0_bias: values(conv1_config.m0_bias_len(), 0.03, 0.1),
            conv1_m1_weight: values(conv1_config.m_conv_weight_len(1), -0.04, 0.3),
            conv2_radial: values(2 * conv2_config.radial_node_stride(), -0.06, 0.8),
            conv2_m0_weight: values(conv2_config.m0_weight_len(), 0.08, -0.1),
            conv2_m0_bias: values(conv2_config.m0_bias_len(), -0.02, 0.05),
            conv2_m1_weight: values(conv2_config.m_conv_weight_len(1), 0.09, 0.2),
            edge_distances: vec![1.2, 2.1],
            wigner_inv,
        }
    }

    fn forward(&self, out: &mut [f64]) {
        fairchem_esen_edgewise_gate_forward(
            self.config,
            &self.edge_sources,
            &self.edge_targets,
            &self.node_features,
            &self.wigner,
            Some(&self.conv1_radial),
            &self.conv1_m0_weight,
            &self.conv1_m0_bias,
            &[&self.conv1_m1_weight],
            Some(&self.conv2_radial),
            &self.conv2_m0_weight,
            &self.conv2_m0_bias,
            &[&self.conv2_m1_weight],
            Some(&self.edge_distances),
            &self.wigner_inv,
            out,
        )
        .unwrap();
    }

    fn objective(&self, grad_out: &[f64]) -> f64 {
        let mut out = vec![0.0; self.node_features.len()];
        self.forward(&mut out);
        out.iter()
            .zip(grad_out.iter())
            .map(|(value, grad)| value * grad)
            .sum()
    }
}

fn finite_difference(
    fixture: &mut EdgewiseFixture,
    grad_out: &[f64],
    mut perturb: impl FnMut(&mut EdgewiseFixture, f64),
) -> f64 {
    let step = 1.0e-6;
    perturb(fixture, step);
    let plus = fixture.objective(grad_out);
    perturb(fixture, -2.0 * step);
    let minus = fixture.objective(grad_out);
    perturb(fixture, step);
    (plus - minus) / (2.0 * step)
}

fn finite_difference_grid_block(
    fixture: &mut GridBlockFixture,
    grad_out: &[f64],
    mut perturb: impl FnMut(&mut GridBlockFixture, f64),
) -> f64 {
    let step = 1.0e-6;
    perturb(fixture, step);
    let plus = fixture.objective(grad_out);
    perturb(fixture, -2.0 * step);
    let minus = fixture.objective(grad_out);
    perturb(fixture, step);
    (plus - minus) / (2.0 * step)
}

fn assert_fd(analytic: f64, fd: f64, label: &str) {
    assert!(
        (analytic - fd).abs() <= 2.0e-8,
        "{label}: analytic={analytic}, fd={fd}, diff={}",
        (analytic - fd).abs()
    );
}

fn values(len: usize, scale: f64, shift: f64) -> Vec<f64> {
    (0..len)
        .map(|index| (scale * index as f64 + shift).sin())
        .collect()
}

fn values_f32(len: usize, scale: f64, shift: f64) -> Vec<f32> {
    values(len, scale, shift)
        .into_iter()
        .map(|value| value as f32)
        .collect()
}

fn to_f64(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&value| value as f64).collect()
}

fn assert_f32_matches_f64(actual: &[f32], expected: &[f64], abs_tol: f64, rel_tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let abs = (actual as f64 - expected).abs();
        let rel = abs / expected.abs().max(1.0e-6);
        assert!(
            abs <= abs_tol || rel <= rel_tol,
            "index={index}, actual={actual:.8e}, expected={expected:.8e}, abs={abs:.3e}, rel={rel:.3e}"
        );
    }
}

fn matrix_values(edges: usize, coefficient_count: usize, offdiag_scale: f64) -> Vec<f64> {
    let mut values = vec![0.0; edges * coefficient_count * coefficient_count];
    for edge in 0..edges {
        for row in 0..coefficient_count {
            for column in 0..coefficient_count {
                let index =
                    edge * coefficient_count * coefficient_count + row * coefficient_count + column;
                values[index] = if row == column {
                    1.0 + 0.01 * edge as f64
                } else {
                    offdiag_scale * (1 + edge + row + column) as f64
                };
            }
        }
    }
    values
}

fn independent_grid_atomwise_forward(fixture: &GridAtomwiseFixture) -> Vec<f64> {
    let config = fixture.config;
    let nodes = fixture.input.len() / config.coefficient_node_stride();
    let sphere = config.sphere_channels();
    let hidden = config.hidden_channels();
    let coefficients = config.coefficient_count();
    let grid_points = config.grid_points();
    let mut out = vec![0.0; fixture.input.len()];
    for node in 0..nodes {
        let input_node_start = node * config.coefficient_node_stride();
        let out_node_start = node * config.coefficient_node_stride();
        let mut grid = vec![0.0; grid_points * sphere];
        for point in 0..grid_points {
            for channel in 0..sphere {
                let mut value = 0.0;
                for coefficient in 0..coefficients {
                    value += fixture.to_grid[point * coefficients + coefficient]
                        * fixture.input[input_node_start + coefficient * sphere + channel];
                }
                grid[point * sphere + channel] = value;
            }
        }

        let mut grid_out = vec![0.0; grid_points * sphere];
        for point in 0..grid_points {
            let mut hidden1 = vec![0.0; hidden];
            for (hidden_index, hidden_value) in hidden1.iter_mut().enumerate() {
                for channel in 0..sphere {
                    *hidden_value += fixture.weight1[hidden_index * sphere + channel]
                        * grid[point * sphere + channel];
                }
                *hidden_value = silu(*hidden_value);
            }
            let mut hidden2 = vec![0.0; hidden];
            for (hidden_index, hidden_value) in hidden2.iter_mut().enumerate() {
                for (in_hidden, hidden1_value) in hidden1.iter().enumerate() {
                    *hidden_value +=
                        fixture.weight2[hidden_index * hidden + in_hidden] * hidden1_value;
                }
                *hidden_value = silu(*hidden_value);
            }
            for channel in 0..sphere {
                for (hidden_index, hidden2_value) in hidden2.iter().enumerate() {
                    grid_out[point * sphere + channel] +=
                        fixture.weight3[channel * hidden + hidden_index] * hidden2_value;
                }
            }
        }

        for coefficient in 0..coefficients {
            for channel in 0..sphere {
                let mut value = 0.0;
                for point in 0..grid_points {
                    value += fixture.from_grid[point * coefficients + coefficient]
                        * grid_out[point * sphere + channel];
                }
                out[out_node_start + coefficient * sphere + channel] = value;
            }
        }
    }
    out
}

fn independent_spectral_atomwise_forward(fixture: &SpectralAtomwiseFixture) -> Vec<f64> {
    let config = fixture.config;
    let nodes = fixture.input.len() / config.input_node_stride();
    let coefficients = config.coefficient_count();
    let sphere = config.sphere_channels();
    let hidden = config.hidden_channels();
    let mut out = vec![0.0; fixture.input.len()];
    for node in 0..nodes {
        let input_start = node * config.input_node_stride();
        let out_start = node * config.input_node_stride();
        let mut gates = vec![0.0; config.gate_node_stride()];
        for (gate_index, gate_value) in gates.iter_mut().enumerate() {
            let weight_start = gate_index * sphere;
            let mut value = fixture.scalar_gate_bias[gate_index];
            for channel in 0..sphere {
                value += fixture.scalar_gate_weight[weight_start + channel]
                    * fixture.input[input_start + channel];
            }
            *gate_value = silu(value);
        }

        let mut hidden_features = vec![0.0; coefficients * hidden];
        for coefficient in 0..coefficients {
            let degree = degree_for_full_coefficient(coefficient);
            for hidden_channel in 0..hidden {
                let mut value = if coefficient == 0 {
                    fixture.linear1_bias[hidden_channel]
                } else {
                    0.0
                };
                let weight_start = degree * hidden * sphere + hidden_channel * sphere;
                for channel in 0..sphere {
                    value += fixture.linear1_weight[weight_start + channel]
                        * fixture.input[input_start + coefficient * sphere + channel];
                }
                hidden_features[coefficient * hidden + hidden_channel] = value;
            }
        }

        let mut activated = vec![0.0; hidden_features.len()];
        for hidden_channel in 0..hidden {
            activated[hidden_channel] = silu(hidden_features[hidden_channel]);
        }
        for degree in 1..=config.l_max() {
            let gate_start = (degree - 1) * hidden;
            let coefficient_start = degree * degree;
            let coefficient_end = (degree + 1) * (degree + 1);
            for coefficient in coefficient_start..coefficient_end {
                for hidden_channel in 0..hidden {
                    let index = coefficient * hidden + hidden_channel;
                    activated[index] =
                        hidden_features[index] * sigmoid(gates[gate_start + hidden_channel]);
                }
            }
        }

        for coefficient in 0..coefficients {
            let degree = degree_for_full_coefficient(coefficient);
            for channel in 0..sphere {
                let mut value = if coefficient == 0 {
                    fixture.linear2_bias[channel]
                } else {
                    0.0
                };
                let weight_start = degree * sphere * hidden + channel * hidden;
                for hidden_channel in 0..hidden {
                    value += fixture.linear2_weight[weight_start + hidden_channel]
                        * activated[coefficient * hidden + hidden_channel];
                }
                out[out_start + coefficient * sphere + channel] = value;
            }
        }
    }
    out
}

fn degree_for_full_coefficient(coefficient: usize) -> usize {
    let mut degree = 0usize;
    while (degree + 1) * (degree + 1) <= coefficient {
        degree += 1;
    }
    degree
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}
