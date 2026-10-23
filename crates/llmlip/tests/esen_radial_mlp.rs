#![allow(clippy::unwrap_used)]

use llmlip::esen_mlp::{
    EsenRadialMlp, EsenRadialMlpBatchCache, EsenRadialMlpCache, EsenRadialMlpLayer,
};

#[test]
fn esen_radial_mlp_forward_matches_independent_formula() {
    let mlp = sample_mlp();
    let input = [0.4, -1.2, 0.7];
    let mut out = [0.0; 2];
    let mut cache = EsenRadialMlpCache::default();

    mlp.forward(&input, &mut out, &mut cache).unwrap();

    let hidden_linear = dense(
        &[
            0.2, -0.4, 0.5, 1.0, -0.3, 0.7, -0.8, 0.6, 0.1, 0.9, -0.2, 0.3,
        ],
        &[0.1, -0.2, 0.3, 0.4],
        3,
        4,
        &input,
    );
    let hidden_norm = layer_norm(
        &hidden_linear,
        &[1.1, 0.8, -0.6, 1.3],
        &[0.2, -0.1, 0.05, -0.3],
        1.0e-5,
    );
    let hidden: Vec<_> = hidden_norm.iter().copied().map(silu_reference).collect();
    let expected = dense(
        &[0.5, -1.0, 0.25, 0.75, -0.4, 0.9, 1.2, -0.8],
        &[-0.05, 0.15],
        4,
        2,
        &hidden,
    );

    for (actual, expected) in out.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() <= 1.0e-14);
    }
}

#[test]
fn esen_radial_mlp_vjp_matches_finite_difference() {
    let mlp = sample_mlp();
    let mut input = [0.4, -1.2, 0.7];
    let grad_out = [0.6, -0.9];
    let mut out = [0.0; 2];
    let mut cache = EsenRadialMlpCache::default();
    let mut grad_input = [0.0; 3];

    mlp.forward(&input, &mut out, &mut cache).unwrap();
    mlp.vjp_input_only(&cache, &grad_out, &mut grad_input)
        .unwrap();

    let step = 1.0e-6;
    for index in 0..input.len() {
        let original = input[index];
        input[index] = original + step;
        let plus = scalar_objective(&mlp, &input, &grad_out);
        input[index] = original - step;
        let minus = scalar_objective(&mlp, &input, &grad_out);
        input[index] = original;
        let fd = (plus - minus) / (2.0 * step);
        assert!(
            (grad_input[index] - fd).abs() <= 5.0e-8,
            "index={index}, grad={}, fd={fd}",
            grad_input[index]
        );
    }
}

#[test]
fn esen_radial_mlp_forward_batch_matches_per_row_forward_and_vjp_cache() {
    let mlp = sample_mlp();
    let rows = 4;
    let mut input = vec![0.0; rows * mlp.input_dim()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = (0.17 * index as f64).sin() - 0.4;
    }
    let mut batch_out = vec![0.0; rows * mlp.output_dim()];
    let mut batch_caches = Vec::new();

    mlp.forward_batch(rows, &input, &mut batch_out, &mut batch_caches)
        .unwrap();

    assert_eq!(batch_caches.len(), rows);
    for (row, batch_cache) in batch_caches.iter().enumerate().take(rows) {
        let input_start = row * mlp.input_dim();
        let output_start = row * mlp.output_dim();
        let mut row_out = vec![0.0; mlp.output_dim()];
        let mut row_cache = EsenRadialMlpCache::default();
        mlp.forward(
            &input[input_start..input_start + mlp.input_dim()],
            &mut row_out,
            &mut row_cache,
        )
        .unwrap();
        for (actual, expected) in batch_out[output_start..output_start + mlp.output_dim()]
            .iter()
            .zip(row_out.iter())
        {
            assert!((actual - expected).abs() <= 1.0e-13);
        }

        let grad_out = [0.6, -0.9];
        let mut batch_grad_input = vec![0.0; mlp.input_dim()];
        let mut row_grad_input = vec![0.0; mlp.input_dim()];
        mlp.vjp_input_only(batch_cache, &grad_out, &mut batch_grad_input)
            .unwrap();
        mlp.vjp_input_only(&row_cache, &grad_out, &mut row_grad_input)
            .unwrap();
        for (actual, expected) in batch_grad_input.iter().zip(row_grad_input.iter()) {
            assert!((actual - expected).abs() <= 1.0e-13);
        }
    }

    let mut grad_out = vec![0.0; rows * mlp.output_dim()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.11 * index as f64).cos();
    }
    let mut batch_grad_input = vec![0.0; rows * mlp.input_dim()];
    mlp.vjp_input_only_batch(&batch_caches, rows, &grad_out, &mut batch_grad_input)
        .unwrap();
    for (row, batch_cache) in batch_caches.iter().enumerate().take(rows) {
        let mut row_grad_input = vec![0.0; mlp.input_dim()];
        let grad_start = row * mlp.output_dim();
        mlp.vjp_input_only(
            batch_cache,
            &grad_out[grad_start..grad_start + mlp.output_dim()],
            &mut row_grad_input,
        )
        .unwrap();
        let input_start = row * mlp.input_dim();
        for (actual, expected) in batch_grad_input[input_start..input_start + mlp.input_dim()]
            .iter()
            .zip(row_grad_input.iter())
        {
            assert!((actual - expected).abs() <= 1.0e-13);
        }
    }
}

#[test]
fn esen_radial_mlp_contiguous_cache_vjp_matches_per_row_vjp() {
    let mlp = sample_mlp();
    let rows = 7;
    let mut input = vec![0.0; rows * mlp.input_dim()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = (0.13 * index as f64).sin() + 0.2 * (0.07 * index as f64).cos();
    }

    let mut contiguous_out = vec![0.0; rows * mlp.output_dim()];
    let mut contiguous_cache = EsenRadialMlpBatchCache::default();
    mlp.forward_batch_contiguous_cache(rows, &input, &mut contiguous_out, &mut contiguous_cache)
        .unwrap();

    let mut grad_out = vec![0.0; rows * mlp.output_dim()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.19 * index as f64).cos() - 0.1;
    }
    let mut contiguous_grad_input = vec![0.0; rows * mlp.input_dim()];
    mlp.vjp_input_only_batch_contiguous_cache(
        &contiguous_cache,
        rows,
        &grad_out,
        &mut contiguous_grad_input,
    )
    .unwrap();

    for row in 0..rows {
        let input_start = row * mlp.input_dim();
        let output_start = row * mlp.output_dim();
        let mut row_out = vec![0.0; mlp.output_dim()];
        let mut row_cache = EsenRadialMlpCache::default();
        mlp.forward(
            &input[input_start..input_start + mlp.input_dim()],
            &mut row_out,
            &mut row_cache,
        )
        .unwrap();
        for (actual, expected) in contiguous_out[output_start..output_start + mlp.output_dim()]
            .iter()
            .zip(row_out.iter())
        {
            assert!((actual - expected).abs() <= 1.0e-13);
        }

        let mut row_grad_input = vec![0.0; mlp.input_dim()];
        mlp.vjp_input_only(
            &row_cache,
            &grad_out[output_start..output_start + mlp.output_dim()],
            &mut row_grad_input,
        )
        .unwrap();
        for (actual, expected) in contiguous_grad_input[input_start..input_start + mlp.input_dim()]
            .iter()
            .zip(row_grad_input.iter())
        {
            assert!((actual - expected).abs() <= 1.0e-13);
        }
    }
}

#[test]
fn esen_radial_mlp_final_linear_block_vjp_matches_full_vjp() {
    let mlp = sample_mlp();
    let rows = 5;
    let mut input = vec![0.0; rows * mlp.input_dim()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = (0.23 * index as f64).sin() - 0.3;
    }

    let mut out = vec![0.0; rows * mlp.output_dim()];
    let mut cache = EsenRadialMlpBatchCache::default();
    mlp.forward_batch_contiguous_cache(rows, &input, &mut out, &mut cache)
        .unwrap();

    let mut grad_out = vec![0.0; rows * mlp.output_dim()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.17 * index as f64).cos() + 0.2;
    }
    let mut full_grad_input = vec![0.0; rows * mlp.input_dim()];
    mlp.vjp_input_only_batch_contiguous_cache(&cache, rows, &grad_out, &mut full_grad_input)
        .unwrap();

    let mut final_input_grad = vec![0.0; rows * mlp.final_linear_input_dim()];
    for output_offset in 0..mlp.output_dim() {
        let mut block = vec![0.0; rows];
        for row in 0..rows {
            block[row] = grad_out[row * mlp.output_dim() + output_offset];
        }
        mlp.accumulate_final_linear_input_vjp_from_output_block(
            rows,
            output_offset,
            1,
            &block,
            &mut final_input_grad,
        )
        .unwrap();
    }

    let mut split_grad_input = vec![0.0; rows * mlp.input_dim()];
    mlp.vjp_input_only_batch_contiguous_cache_from_final_linear_input_grad(
        &cache,
        rows,
        &final_input_grad,
        &mut split_grad_input,
    )
    .unwrap();

    for (actual, expected) in split_grad_input.iter().zip(full_grad_input.iter()) {
        assert!((actual - expected).abs() <= 1.0e-13);
    }
}

#[test]
fn esen_radial_mlp_final_linear_input_and_output_blocks_match_full_forward() {
    let mlp = sample_mlp();
    let rows = 6;
    let mut input = vec![0.0; rows * mlp.input_dim()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = (0.29 * index as f64).sin() + 0.1 * (0.31 * index as f64).cos();
    }

    let mut full_out = vec![0.0; rows * mlp.output_dim()];
    let mut full_cache = EsenRadialMlpBatchCache::default();
    mlp.forward_batch_contiguous_cache(rows, &input, &mut full_out, &mut full_cache)
        .unwrap();

    let mut final_input = vec![0.0; rows * mlp.final_linear_input_dim()];
    let mut split_cache = EsenRadialMlpBatchCache::default();
    mlp.forward_batch_final_linear_input_contiguous_cache(
        rows,
        &input,
        &mut final_input,
        &mut split_cache,
    )
    .unwrap();

    let mut split_out = vec![0.0; full_out.len()];
    for output_offset in 0..mlp.output_dim() {
        let mut block = vec![0.0; rows];
        mlp.final_linear_forward_output_block(rows, &final_input, output_offset, 1, &mut block)
            .unwrap();
        for row in 0..rows {
            split_out[row * mlp.output_dim() + output_offset] = block[row];
        }
    }

    for (actual, expected) in split_out.iter().zip(full_out.iter()) {
        assert!((actual - expected).abs() <= 1.0e-13);
    }

    let row_stride = mlp.output_dim() + 3;
    let feature_offset = 2;
    let output_offset = 1;
    let output_len = mlp.output_dim() - output_offset;
    let mut strided = vec![0.0; rows * row_stride];
    mlp.final_linear_forward_output_block_strided(
        rows,
        &final_input,
        output_offset,
        output_len,
        row_stride,
        feature_offset,
        &mut strided,
    )
    .unwrap();
    for row in 0..rows {
        let expected_start = row * mlp.output_dim() + output_offset;
        let actual_start = row * row_stride + feature_offset;
        for index in 0..output_len {
            assert!(
                (strided[actual_start + index] - full_out[expected_start + index]).abs() <= 1.0e-13
            );
        }
    }

    let mut grad_out = vec![0.0; rows * mlp.output_dim()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.37 * index as f64).cos() - 0.2;
    }
    let mut full_grad_input = vec![0.0; rows * mlp.input_dim()];
    mlp.vjp_input_only_batch_contiguous_cache(&full_cache, rows, &grad_out, &mut full_grad_input)
        .unwrap();

    let mut final_input_grad = vec![0.0; rows * mlp.final_linear_input_dim()];
    for output_offset in 0..mlp.output_dim() {
        let mut block = vec![0.0; rows];
        for row in 0..rows {
            block[row] = grad_out[row * mlp.output_dim() + output_offset];
        }
        mlp.accumulate_final_linear_input_vjp_from_output_block(
            rows,
            output_offset,
            1,
            &block,
            &mut final_input_grad,
        )
        .unwrap();
    }

    let mut split_grad_input = vec![0.0; rows * mlp.input_dim()];
    mlp.vjp_input_only_batch_contiguous_cache_from_final_linear_input_grad(
        &split_cache,
        rows,
        &final_input_grad,
        &mut split_grad_input,
    )
    .unwrap();

    for (actual, expected) in split_grad_input.iter().zip(full_grad_input.iter()) {
        assert!((actual - expected).abs() <= 1.0e-13);
    }
}

#[test]
fn esen_radial_mlp_wide_batch_forward_and_vjp_match_scalar_rows() {
    let first = EsenRadialMlpLayer::new(
        9,
        64,
        deterministic_values(9 * 64, 0.017, 0.0013),
        deterministic_values(64, -0.11, 0.002),
        Some((
            deterministic_values(64, 0.8, 0.0009),
            deterministic_values(64, -0.05, 0.0011),
        )),
    )
    .unwrap();
    let second = EsenRadialMlpLayer::new(
        64,
        32,
        deterministic_values(64 * 32, -0.013, 0.0007),
        deterministic_values(32, 0.03, -0.0015),
        None,
    )
    .unwrap();
    let mlp = EsenRadialMlp::new(vec![first, second], 1.0e-5).unwrap();
    let rows = 19;
    let mut input = vec![0.0; rows * mlp.input_dim()];
    for (index, value) in input.iter_mut().enumerate() {
        *value = (0.07 * index as f64).sin() - 0.25 * (0.13 * index as f64).cos();
    }

    let mut batch_out = vec![0.0; rows * mlp.output_dim()];
    let mut batch_cache = EsenRadialMlpBatchCache::default();
    mlp.forward_batch_contiguous_cache(rows, &input, &mut batch_out, &mut batch_cache)
        .unwrap();

    let mut scalar_caches = Vec::with_capacity(rows);
    let mut expected_out = vec![0.0; batch_out.len()];
    for row in 0..rows {
        let input_start = row * mlp.input_dim();
        let output_start = row * mlp.output_dim();
        let mut row_cache = EsenRadialMlpCache::default();
        mlp.forward(
            &input[input_start..input_start + mlp.input_dim()],
            &mut expected_out[output_start..output_start + mlp.output_dim()],
            &mut row_cache,
        )
        .unwrap();
        scalar_caches.push(row_cache);
    }
    assert_close_slice(&batch_out, &expected_out, 1.0e-10, "wide forward");

    let mut grad_out = vec![0.0; rows * mlp.output_dim()];
    for (index, value) in grad_out.iter_mut().enumerate() {
        *value = (0.19 * index as f64).cos() + 0.1 * (0.03 * index as f64).sin();
    }
    let mut batch_grad_input = vec![0.0; rows * mlp.input_dim()];
    mlp.vjp_input_only_batch_contiguous_cache(&batch_cache, rows, &grad_out, &mut batch_grad_input)
        .unwrap();

    let mut expected_grad_input = vec![0.0; batch_grad_input.len()];
    for (row, row_cache) in scalar_caches.iter().enumerate() {
        let grad_out_start = row * mlp.output_dim();
        let grad_input_start = row * mlp.input_dim();
        mlp.vjp_input_only(
            row_cache,
            &grad_out[grad_out_start..grad_out_start + mlp.output_dim()],
            &mut expected_grad_input[grad_input_start..grad_input_start + mlp.input_dim()],
        )
        .unwrap();
    }
    assert_close_slice(
        &batch_grad_input,
        &expected_grad_input,
        1.0e-10,
        "wide VJP input",
    );
}

#[test]
fn esen_radial_mlp_rejects_missing_hidden_layer_norm() {
    let first = EsenRadialMlpLayer::new(2, 3, vec![0.1; 6], vec![0.0; 3], None).unwrap();
    let second = EsenRadialMlpLayer::new(3, 1, vec![0.2; 3], vec![0.0; 1], None).unwrap();
    assert!(EsenRadialMlp::new(vec![first, second], 1.0e-5).is_err());
}

fn sample_mlp() -> EsenRadialMlp {
    let first = EsenRadialMlpLayer::new(
        3,
        4,
        vec![
            0.2, -0.4, 0.5, 1.0, -0.3, 0.7, -0.8, 0.6, 0.1, 0.9, -0.2, 0.3,
        ],
        vec![0.1, -0.2, 0.3, 0.4],
        Some((vec![1.1, 0.8, -0.6, 1.3], vec![0.2, -0.1, 0.05, -0.3])),
    )
    .unwrap();
    let second = EsenRadialMlpLayer::new(
        4,
        2,
        vec![0.5, -1.0, 0.25, 0.75, -0.4, 0.9, 1.2, -0.8],
        vec![-0.05, 0.15],
        None,
    )
    .unwrap();
    EsenRadialMlp::new(vec![first, second], 1.0e-5).unwrap()
}

fn deterministic_values(len: usize, start: f64, step: f64) -> Vec<f64> {
    (0..len)
        .map(|index| start + step * (index % 97) as f64)
        .collect()
}

fn assert_close_slice(actual: &[f64], expected: &[f64], tolerance: f64, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context} length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}[{index}] mismatch: actual={actual:.15e} expected={expected:.15e} diff={:.15e}",
            (actual - expected).abs()
        );
    }
}

fn scalar_objective(mlp: &EsenRadialMlp, input: &[f64], grad_out: &[f64]) -> f64 {
    let mut out = vec![0.0; mlp.output_dim()];
    let mut cache = EsenRadialMlpCache::default();
    mlp.forward(input, &mut out, &mut cache).unwrap();
    out.iter()
        .zip(grad_out.iter())
        .map(|(value, grad)| value * grad)
        .sum()
}

fn dense(weight: &[f64], bias: &[f64], in_dim: usize, out_dim: usize, input: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; out_dim];
    for out_index in 0..out_dim {
        let row_start = out_index * in_dim;
        let mut value = bias[out_index];
        for in_index in 0..in_dim {
            value += weight[row_start + in_index] * input[in_index];
        }
        out[out_index] = value;
    }
    out
}

fn layer_norm(input: &[f64], weight: &[f64], bias: &[f64], eps: f64) -> Vec<f64> {
    let mean = input.iter().sum::<f64>() / input.len() as f64;
    let variance = input
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / input.len() as f64;
    let inv_std = (variance + eps).powf(-0.5);
    input
        .iter()
        .zip(weight.iter())
        .zip(bias.iter())
        .map(|((value, weight), bias)| (value - mean) * inv_std * weight + bias)
        .collect()
}

fn silu_reference(value: f64) -> f64 {
    value / (1.0 + (-value).exp())
}
