//! Runtime for the pinned `nequix-v1` contract.
//!
//! Validates the pinned NequiX-PFT state handoff, loads exported f64 tensor
//! payloads, and runs forward energy plus conservative force/stress/virial VJP
//! inference. Global stress/virial are exposed for full-PBC internal prediction
//! after pinned upstream full-PBC stress fixtures and strain finite-difference
//! validation.

use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
};

use e3nn_core::{
    radial::distance_vjp_accumulate, CgEntry, Irreps, SphericalHarmonicsPlan,
    TensorProductInstruction, TensorProductPlan,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::Deserialize;
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
use crate::force_accum::{
    accumulate_edge_displacement_gradient, stress_from_displacement_gradient,
    virial_from_displacement_gradient, ForceAccumError, Mat3,
};
use crate::geometry::Cell;
use crate::neighbor::cell_list_multi_image_neighbor_list;
use crate::{scatter_edge_vector_vjp, Prediction, PredictionOptions, Structure};

const STATE_SCHEMA_VERSION: &str = "nequix-state-dict-v1";
const ARCHITECTURE: &str = "nequix-v1";
const EXPECTED_CUTOFF: f64 = 6.0;
const EXPECTED_L_MAX: usize = 3;
const EXPECTED_N_LAYERS: usize = 4;
const EXPECTED_TENSOR_COUNT: usize = 63;
const EXPECTED_TOTAL_NUMEL: usize = 707_658;
const EXPECTED_TOTAL_BYTES_F64: usize = 5_661_264;
const E3NN_JAX_NORMALIZED_SILU_SCALE: f64 = 1.676_562_259;
const TP_PLAN_SCHEMA_VERSION: &str = "e3nn-core-tensor-product-plan-v1";
const NEQUIX_DISTANCE_EPS: f64 = 1.0e-12;
#[cfg(feature = "parallel")]
const NEQUIX_PARALLEL_EDGE_THRESHOLD: usize = 128;
#[cfg(feature = "parallel")]
const NEQUIX_PARALLEL_DENSE_ROW_THRESHOLD: usize = 512;
#[cfg(feature = "parallel")]
const NEQUIX_PARALLEL_ACTIVATION_LEN_THRESHOLD: usize = 4096;

/// Numerical dtype used for the heavy linear (matmul) boundaries of the NeQuiX
/// compute path. `Float64` is the default, exact path. `Float32` mirrors the
/// MACE-MH-1 "f32-promoted boundaries" precedent: the dense / irrep-linear
/// matmuls (the operations the f64 path delegates to BLAS) are computed in f32
/// and then promoted back to f64 at the boundary, while every non-matmul
/// operation (SiLU, gate, RMS layer-norm, tensor product, scatter, spherical
/// harmonics, radial basis) stays in f64. Because the forward cache stores the
/// f32-rounded (f64-promoted) matmul outputs and the analytic VJP re-uses the
/// same f32-promoted matmul boundaries, both force = -dE/dR and the stress /
/// virial displacement gradient (which derive from the same edge-vector VJP)
/// stay consistent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NequixComputeDtype {
    Float64,
    Float32,
}

impl NequixComputeDtype {
    #[inline]
    fn is_f32(self) -> bool {
        matches!(self, NequixComputeDtype::Float32)
    }
}

#[must_use]
pub fn nequix_radial_basis(
    distance: f64,
    num_basis: usize,
    cutoff: f64,
    polynomial_p: f64,
) -> Vec<f64> {
    let mut output = vec![0.0; num_basis];
    nequix_radial_basis_into(distance, cutoff, polynomial_p, &mut output);
    output
}

fn nequix_radial_basis_into(distance: f64, cutoff: f64, polynomial_p: f64, output: &mut [f64]) {
    let cutoff_value = nequix_polynomial_cutoff(distance, cutoff, polynomial_p);
    for (index, value) in output.iter_mut().enumerate() {
        *value = nequix_bessel_basis_value(distance, index + 1, cutoff) * cutoff_value;
    }
}

#[must_use]
pub fn nequix_bessel_basis_value(distance: f64, basis_index: usize, cutoff: f64) -> f64 {
    let weight = basis_index as f64 * std::f64::consts::PI;
    let prefactor = 2.0 / cutoff;
    if distance == 0.0 {
        prefactor * weight / cutoff
    } else {
        prefactor * (weight * distance / cutoff).sin() / distance
    }
}

#[must_use]
pub fn nequix_polynomial_cutoff(distance: f64, cutoff: f64, p: f64) -> f64 {
    let x = distance / cutoff;
    let value = 1.0 - ((p + 1.0) * (p + 2.0) / 2.0) * x.powf(p) + p * (p + 2.0) * x.powf(p + 1.0)
        - (p * (p + 1.0) / 2.0) * x.powf(p + 2.0);
    if x < 1.0 {
        value
    } else {
        0.0
    }
}

#[must_use]
pub fn nequix_bessel_basis_derivative(distance: f64, basis_index: usize, cutoff: f64) -> f64 {
    if distance == 0.0 {
        return 0.0;
    }
    let weight = basis_index as f64 * std::f64::consts::PI;
    let angle = weight * distance / cutoff;
    let prefactor = 2.0 / cutoff;
    prefactor * ((weight / cutoff) * angle.cos() * distance - angle.sin()) / (distance * distance)
}

#[must_use]
pub fn nequix_polynomial_cutoff_derivative(distance: f64, cutoff: f64, p: f64) -> f64 {
    let x = distance / cutoff;
    if x >= 1.0 {
        return 0.0;
    }
    let a = (p + 1.0) * (p + 2.0) / 2.0;
    let b = p * (p + 2.0);
    let c = p * (p + 1.0) / 2.0;
    (-a * p * x.powf(p - 1.0) + b * (p + 1.0) * x.powf(p) - c * (p + 2.0) * x.powf(p + 1.0))
        / cutoff
}

pub fn nequix_radial_basis_vjp_distance(
    distance: f64,
    num_basis: usize,
    cutoff: f64,
    polynomial_p: f64,
    grad_basis: &[f64],
) -> Result<f64, NequixError> {
    if grad_basis.len() != num_basis {
        return Err(invalid_model(format!(
            "radial basis VJP cotangent length must be num_basis = {num_basis}, got {}",
            grad_basis.len()
        )));
    }
    let cutoff_value = nequix_polynomial_cutoff(distance, cutoff, polynomial_p);
    let cutoff_derivative = nequix_polynomial_cutoff_derivative(distance, cutoff, polynomial_p);
    let mut grad_distance = 0.0;
    for (basis, grad) in grad_basis.iter().enumerate() {
        let basis_index = basis + 1;
        let bessel = nequix_bessel_basis_value(distance, basis_index, cutoff);
        let bessel_derivative = nequix_bessel_basis_derivative(distance, basis_index, cutoff);
        grad_distance += grad * (bessel_derivative * cutoff_value + bessel * cutoff_derivative);
    }
    Ok(grad_distance)
}

pub fn nequix_dense_forward(
    input: &[f64],
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out_dim: usize,
) -> Result<Vec<f64>, NequixError> {
    nequix_dense_forward_scaled(input, rows, in_dim, weight, out_dim, 1.0)
}

pub fn nequix_dense_forward_scaled(
    input: &[f64],
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out_dim: usize,
    alpha: f64,
) -> Result<Vec<f64>, NequixError> {
    nequix_dense_forward_scaled_dtype(
        input,
        rows,
        in_dim,
        weight,
        out_dim,
        alpha,
        NequixComputeDtype::Float64,
    )
}

fn nequix_dense_forward_scaled_dtype(
    input: &[f64],
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out_dim: usize,
    alpha: f64,
    dtype: NequixComputeDtype,
) -> Result<Vec<f64>, NequixError> {
    let mut output = vec![0.0; rows * out_dim];
    nequix_dense_forward_scaled_into_dtype(
        input,
        rows,
        in_dim,
        weight,
        out_dim,
        alpha,
        &mut output,
        dtype,
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn nequix_dense_forward_scaled_into_dtype(
    input: &[f64],
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out_dim: usize,
    alpha: f64,
    output: &mut [f64],
    dtype: NequixComputeDtype,
) -> Result<(), NequixError> {
    if input.len() != rows * in_dim {
        return Err(invalid_model(format!(
            "dense input length must be rows * in_dim = {}, got {}",
            rows * in_dim,
            input.len()
        )));
    }
    if weight.len() != in_dim * out_dim {
        return Err(invalid_model(format!(
            "dense weight length must be in_dim * out_dim = {}, got {}",
            in_dim * out_dim,
            weight.len()
        )));
    }
    if output.len() != rows * out_dim {
        return Err(invalid_model(format!(
            "dense output length must be rows * out_dim = {}, got {}",
            rows * out_dim,
            output.len()
        )));
    }
    output.fill(0.0);
    if dtype.is_f32() {
        return nequix_dense_forward_scaled_into_f32_promoted(
            input, rows, in_dim, weight, out_dim, alpha, output,
        );
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
    {
        blas_backend::dense_forward(input, rows, in_dim, weight, output, alpha);
        Ok(())
    }
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        ),
        all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
    )))]
    {
        for row in 0..rows {
            let input_row = &input[row * in_dim..(row + 1) * in_dim];
            let output_row = &mut output[row * out_dim..(row + 1) * out_dim];
            for in_index in 0..in_dim {
                let input_value = input_row[in_index];
                if input_value == 0.0 {
                    continue;
                }
                let weight_row = &weight[in_index * out_dim..(in_index + 1) * out_dim];
                for out_index in 0..out_dim {
                    output_row[out_index] += alpha * input_value * weight_row[out_index];
                }
            }
        }
        Ok(())
    }
}

/// f32-promoted dense forward boundary: cast inputs/weights to f32, run the
/// matmul in f32 (via BLAS when available, scalar fallback otherwise), then
/// promote the result back to f64. This matches the MACE-MH-1 precedent of
/// computing the heavy linear in single precision while keeping the rest of the
/// pipeline in f64.
fn nequix_dense_forward_scaled_into_f32_promoted(
    input: &[f64],
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out_dim: usize,
    alpha: f64,
    output: &mut [f64],
) -> Result<(), NequixError> {
    let input_f32: Vec<f32> = input.iter().map(|&value| value as f32).collect();
    let weight_f32: Vec<f32> = weight.iter().map(|&value| value as f32).collect();
    let alpha_f32 = alpha as f32;
    let mut output_f32 = vec![0.0_f32; rows * out_dim];
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    ))]
    {
        blas_backend::dense_forward_f32(
            &input_f32,
            rows,
            in_dim,
            &weight_f32,
            &mut output_f32,
            alpha_f32,
        );
    }
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )))]
    {
        for row in 0..rows {
            let input_row = &input_f32[row * in_dim..(row + 1) * in_dim];
            let output_row = &mut output_f32[row * out_dim..(row + 1) * out_dim];
            for in_index in 0..in_dim {
                let input_value = input_row[in_index];
                if input_value == 0.0 {
                    continue;
                }
                let weight_row = &weight_f32[in_index * out_dim..(in_index + 1) * out_dim];
                for out_index in 0..out_dim {
                    output_row[out_index] += alpha_f32 * input_value * weight_row[out_index];
                }
            }
        }
    }
    for (dst, &value) in output.iter_mut().zip(&output_f32) {
        *dst = value as f64;
    }
    Ok(())
}

pub fn nequix_dense_forward_scaled_vjp_input(
    grad_output: &[f64],
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out_dim: usize,
    alpha: f64,
) -> Result<Vec<f64>, NequixError> {
    nequix_dense_forward_scaled_vjp_input_dtype(
        grad_output,
        rows,
        in_dim,
        weight,
        out_dim,
        alpha,
        NequixComputeDtype::Float64,
    )
}

fn nequix_dense_forward_scaled_vjp_input_dtype(
    grad_output: &[f64],
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out_dim: usize,
    alpha: f64,
    dtype: NequixComputeDtype,
) -> Result<Vec<f64>, NequixError> {
    let mut grad_input = vec![0.0; rows * in_dim];
    nequix_dense_forward_scaled_vjp_input_into_dtype(
        grad_output,
        rows,
        in_dim,
        weight,
        out_dim,
        alpha,
        &mut grad_input,
        dtype,
    )?;
    Ok(grad_input)
}

#[allow(clippy::too_many_arguments)]
fn nequix_dense_forward_scaled_vjp_input_into_dtype(
    grad_output: &[f64],
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out_dim: usize,
    alpha: f64,
    grad_input: &mut [f64],
    dtype: NequixComputeDtype,
) -> Result<(), NequixError> {
    if grad_output.len() != rows * out_dim {
        return Err(invalid_model(format!(
            "dense VJP grad_output length must be rows * out_dim = {}, got {}",
            rows * out_dim,
            grad_output.len()
        )));
    }
    if weight.len() != in_dim * out_dim {
        return Err(invalid_model(format!(
            "dense VJP weight length must be in_dim * out_dim = {}, got {}",
            in_dim * out_dim,
            weight.len()
        )));
    }
    if grad_input.len() != rows * in_dim {
        return Err(invalid_model(format!(
            "dense VJP grad_input length must be rows * in_dim = {}, got {}",
            rows * in_dim,
            grad_input.len()
        )));
    }
    grad_input.fill(0.0);
    if dtype.is_f32() {
        return nequix_dense_forward_scaled_vjp_input_into_f32_promoted(
            grad_output,
            rows,
            in_dim,
            weight,
            out_dim,
            alpha,
            grad_input,
        );
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
    {
        blas_backend::dense_vjp_input(rows, in_dim, weight, grad_output, grad_input, alpha);
        Ok(())
    }
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        ),
        all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
    )))]
    {
        for row in 0..rows {
            let grad_output_row = &grad_output[row * out_dim..(row + 1) * out_dim];
            let grad_input_row = &mut grad_input[row * in_dim..(row + 1) * in_dim];
            for input_index in 0..in_dim {
                let mut sum = 0.0;
                for output_index in 0..out_dim {
                    sum += grad_output_row[output_index]
                        * weight[input_index * out_dim + output_index];
                }
                grad_input_row[input_index] += alpha * sum;
            }
        }
        Ok(())
    }
}

/// f32-promoted dense VJP boundary mirroring
/// [`nequix_dense_forward_scaled_into_f32_promoted`]: the transpose matmul is
/// computed in f32 and promoted back to f64, so the analytic backward matches
/// the f32-promoted forward exactly.
fn nequix_dense_forward_scaled_vjp_input_into_f32_promoted(
    grad_output: &[f64],
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out_dim: usize,
    alpha: f64,
    grad_input: &mut [f64],
) -> Result<(), NequixError> {
    debug_assert_eq!(grad_output.len(), rows * out_dim);
    debug_assert_eq!(weight.len(), in_dim * out_dim);
    let grad_output_f32: Vec<f32> = grad_output.iter().map(|&value| value as f32).collect();
    let weight_f32: Vec<f32> = weight.iter().map(|&value| value as f32).collect();
    let alpha_f32 = alpha as f32;
    let mut grad_input_f32 = vec![0.0_f32; rows * in_dim];
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    ))]
    {
        blas_backend::dense_vjp_input_f32(
            rows,
            in_dim,
            &weight_f32,
            &grad_output_f32,
            &mut grad_input_f32,
            alpha_f32,
        );
    }
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )))]
    {
        for row in 0..rows {
            let grad_output_row = &grad_output_f32[row * out_dim..(row + 1) * out_dim];
            let grad_input_row = &mut grad_input_f32[row * in_dim..(row + 1) * in_dim];
            for input_index in 0..in_dim {
                let mut sum = 0.0_f32;
                for output_index in 0..out_dim {
                    sum += grad_output_row[output_index]
                        * weight_f32[input_index * out_dim + output_index];
                }
                grad_input_row[input_index] += alpha_f32 * sum;
            }
        }
    }
    for (dst, &value) in grad_input.iter_mut().zip(&grad_input_f32) {
        *dst = value as f64;
    }
    Ok(())
}

fn nequix_dense_forward_scaled_maybe_parallel_rows(
    input: &[f64],
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out_dim: usize,
    alpha: f64,
    dtype: NequixComputeDtype,
) -> Result<Vec<f64>, NequixError> {
    #[cfg(feature = "parallel")]
    if nequix_should_parallelize_dense_rows(rows) {
        if input.len() != rows * in_dim {
            return Err(invalid_model(format!(
                "dense input length must be rows * in_dim = {}, got {}",
                rows * in_dim,
                input.len()
            )));
        }
        let chunk_rows = nequix_parallel_row_chunk_len(rows);
        let mut output = vec![0.0; rows * out_dim];
        output
            .par_chunks_mut(chunk_rows * out_dim)
            .zip(input.par_chunks(chunk_rows * in_dim))
            .try_for_each(|(output_chunk, input_chunk)| {
                nequix_dense_forward_scaled_into_dtype(
                    input_chunk,
                    input_chunk.len() / in_dim,
                    in_dim,
                    weight,
                    out_dim,
                    alpha,
                    output_chunk,
                    dtype,
                )
            })?;
        return Ok(output);
    }
    nequix_dense_forward_scaled_dtype(input, rows, in_dim, weight, out_dim, alpha, dtype)
}

fn nequix_dense_vjp_input_maybe_parallel_rows(
    grad_output: &[f64],
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out_dim: usize,
    alpha: f64,
    dtype: NequixComputeDtype,
) -> Result<Vec<f64>, NequixError> {
    #[cfg(feature = "parallel")]
    if nequix_should_parallelize_dense_rows(rows) {
        if grad_output.len() != rows * out_dim {
            return Err(invalid_model(format!(
                "dense VJP grad_output length must be rows * out_dim = {}, got {}",
                rows * out_dim,
                grad_output.len()
            )));
        }
        let chunk_rows = nequix_parallel_row_chunk_len(rows);
        let mut grad_input = vec![0.0; rows * in_dim];
        grad_input
            .par_chunks_mut(chunk_rows * in_dim)
            .zip(grad_output.par_chunks(chunk_rows * out_dim))
            .try_for_each(|(grad_input_chunk, grad_chunk)| {
                nequix_dense_forward_scaled_vjp_input_into_dtype(
                    grad_chunk,
                    grad_chunk.len() / out_dim,
                    in_dim,
                    weight,
                    out_dim,
                    alpha,
                    grad_input_chunk,
                    dtype,
                )
            })?;
        return Ok(grad_input);
    }
    nequix_dense_forward_scaled_vjp_input_dtype(
        grad_output,
        rows,
        in_dim,
        weight,
        out_dim,
        alpha,
        dtype,
    )
}

fn nequix_dense_forward_scaled_maybe_parallel_rows_with_silu_cache(
    input: &[f64],
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out_dim: usize,
    alpha: f64,
    dtype: NequixComputeDtype,
) -> Result<(Vec<f64>, Vec<f64>), NequixError> {
    #[cfg(feature = "parallel")]
    if nequix_should_parallelize_dense_rows(rows) {
        if input.len() != rows * in_dim {
            return Err(invalid_model(format!(
                "dense input length must be rows * in_dim = {}, got {}",
                rows * in_dim,
                input.len()
            )));
        }
        let chunk_rows = nequix_parallel_row_chunk_len(rows);
        let mut activated = vec![0.0; rows * out_dim];
        let mut silu_derivative = vec![0.0; rows * out_dim];
        activated
            .par_chunks_mut(chunk_rows * out_dim)
            .zip(silu_derivative.par_chunks_mut(chunk_rows * out_dim))
            .zip(input.par_chunks(chunk_rows * in_dim))
            .try_for_each(|((activated_chunk, derivative_chunk), input_chunk)| {
                nequix_dense_forward_scaled_into_dtype(
                    input_chunk,
                    input_chunk.len() / in_dim,
                    in_dim,
                    weight,
                    out_dim,
                    alpha,
                    activated_chunk,
                    dtype,
                )?;
                for (dst, derivative) in activated_chunk.iter_mut().zip(derivative_chunk) {
                    let (activated, silu_derivative) = silu_with_derivative(*dst);
                    *dst = activated;
                    *derivative = silu_derivative;
                }
                Ok::<(), NequixError>(())
            })?;
        return Ok((activated, silu_derivative));
    }
    let mut output =
        nequix_dense_forward_scaled_dtype(input, rows, in_dim, weight, out_dim, alpha, dtype)?;
    let mut silu_derivative = vec![0.0; output.len()];
    for (value, derivative) in output.iter_mut().zip(&mut silu_derivative) {
        let (activated, cached_derivative) = silu_with_derivative(*value);
        *value = activated;
        *derivative = cached_derivative;
    }
    Ok((output, silu_derivative))
}

fn nequix_silu_in_place(values: &mut [f64]) {
    #[cfg(feature = "parallel")]
    if values.len() >= NEQUIX_PARALLEL_ACTIVATION_LEN_THRESHOLD && rayon::current_num_threads() > 1
    {
        values
            .par_iter_mut()
            .for_each(|value| *value = silu(*value));
        return;
    }
    values.iter_mut().for_each(|value| *value = silu(*value));
}

fn nequix_mul_silu_derivative_in_place(grad: &mut [f64], raw: &[f64]) {
    #[cfg(feature = "parallel")]
    if grad.len() >= NEQUIX_PARALLEL_ACTIVATION_LEN_THRESHOLD && rayon::current_num_threads() > 1 {
        grad.par_iter_mut()
            .zip(raw.par_iter())
            .for_each(|(bar, raw)| *bar *= silu_derivative(*raw));
        return;
    }
    for (bar, raw) in grad.iter_mut().zip(raw) {
        *bar *= silu_derivative(*raw);
    }
}

fn nequix_mul_cached_silu_derivative_in_place(grad: &mut [f64], derivative: &[f64]) {
    for (bar, derivative) in grad.iter_mut().zip(derivative) {
        *bar *= derivative;
    }
}

#[allow(clippy::too_many_arguments)]
fn nequix_dense_vjp_input_maybe_parallel_rows_with_silu_derivative(
    grad_output: &[f64],
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out_dim: usize,
    alpha: f64,
    silu_derivative: &[f64],
    dtype: NequixComputeDtype,
) -> Result<Vec<f64>, NequixError> {
    if silu_derivative.len() != rows * in_dim {
        return Err(invalid_model(format!(
            "dense VJP SiLU derivative length must be rows * in_dim = {}, got {}",
            rows * in_dim,
            silu_derivative.len()
        )));
    }
    #[cfg(feature = "parallel")]
    if nequix_should_parallelize_dense_rows(rows) {
        if grad_output.len() != rows * out_dim {
            return Err(invalid_model(format!(
                "dense VJP grad_output length must be rows * out_dim = {}, got {}",
                rows * out_dim,
                grad_output.len()
            )));
        }
        let chunk_rows = nequix_parallel_row_chunk_len(rows);
        let mut grad_input = vec![0.0; rows * in_dim];
        grad_input
            .par_chunks_mut(chunk_rows * in_dim)
            .zip(grad_output.par_chunks(chunk_rows * out_dim))
            .zip(silu_derivative.par_chunks(chunk_rows * in_dim))
            .try_for_each(|((grad_input_chunk, grad_chunk), derivative_chunk)| {
                nequix_dense_forward_scaled_vjp_input_into_dtype(
                    grad_chunk,
                    grad_chunk.len() / out_dim,
                    in_dim,
                    weight,
                    out_dim,
                    alpha,
                    grad_input_chunk,
                    dtype,
                )?;
                nequix_mul_cached_silu_derivative_in_place(grad_input_chunk, derivative_chunk);
                Ok::<(), NequixError>(())
            })?;
        return Ok(grad_input);
    }
    let mut grad_input = nequix_dense_forward_scaled_vjp_input_dtype(
        grad_output,
        rows,
        in_dim,
        weight,
        out_dim,
        alpha,
        dtype,
    )?;
    nequix_mul_cached_silu_derivative_in_place(&mut grad_input, silu_derivative);
    Ok(grad_input)
}

pub fn nequix_radial_mlp_forward(
    radial_basis: &[f64],
    rows: usize,
    weights: &[&NequixTensor],
) -> Result<Vec<f64>, NequixError> {
    if weights.is_empty() {
        return Err(invalid_model("radial MLP must contain at least one weight"));
    }
    let first_shape = weights[0].shape.as_slice();
    if first_shape.len() != 2 {
        return Err(invalid_model(format!(
            "radial MLP weight must be rank 2, got {:?}",
            weights[0].shape
        )));
    }
    let mut current = radial_basis.to_vec();
    let mut current_dim = first_shape[0];
    if current.len() != rows * current_dim {
        return Err(invalid_model(format!(
            "radial MLP input length must be rows * first_dim = {}, got {}",
            rows * current_dim,
            current.len()
        )));
    }
    for (index, weight) in weights.iter().enumerate() {
        if weight.shape.len() != 2 {
            return Err(invalid_model(format!(
                "radial MLP weight {} must be rank 2, got {:?}",
                weight.tree_path, weight.shape
            )));
        }
        if weight.shape[0] != current_dim {
            return Err(invalid_model(format!(
                "radial MLP weight {} input dim must be {}, got {}",
                weight.tree_path, current_dim, weight.shape[0]
            )));
        }
        let out_dim = weight.shape[1];
        current = nequix_dense_forward_scaled_maybe_parallel_rows(
            &current,
            rows,
            current_dim,
            &weight.values,
            out_dim,
            1.0,
            NequixComputeDtype::Float64,
        )?;
        current_dim = out_dim;
        if index + 1 != weights.len() {
            nequix_silu_in_place(&mut current);
        }
    }
    Ok(current)
}

struct NequixRadialMlpCache {
    output: Vec<f64>,
    silu_derivatives: [Vec<f64>; 2],
}

fn nequix_radial_mlp_forward_cache(
    radial_basis: &[f64],
    rows: usize,
    weights: &[&NequixTensor],
    dtype: NequixComputeDtype,
) -> Result<NequixRadialMlpCache, NequixError> {
    if weights.len() != 3 {
        return Err(invalid_model(format!(
            "cached radial MLP expects exactly 3 weights, got {}",
            weights.len()
        )));
    }
    if weights[0].shape.len() != 2 || weights[1].shape.len() != 2 || weights[2].shape.len() != 2 {
        return Err(invalid_model("cached radial MLP weights must be rank 2"));
    }
    let mut current = radial_basis.to_vec();
    let mut current_dim = weights[0].shape[0];
    let mut silu_derivatives = [Vec::new(), Vec::new()];
    if current.len() != rows * current_dim {
        return Err(invalid_model(format!(
            "radial MLP input length must be rows * first_dim = {}, got {}",
            rows * current_dim,
            current.len()
        )));
    }
    for (index, weight) in weights.iter().enumerate() {
        if weight.shape.len() != 2 {
            return Err(invalid_model(format!(
                "radial MLP weight {} must be rank 2, got {:?}",
                weight.tree_path, weight.shape
            )));
        }
        if weight.shape[0] != current_dim {
            return Err(invalid_model(format!(
                "radial MLP weight {} input dim must be {}, got {}",
                weight.tree_path, current_dim, weight.shape[0]
            )));
        }
        let out_dim = weight.shape[1];
        if index < 2 {
            let (activated, silu_derivative) =
                nequix_dense_forward_scaled_maybe_parallel_rows_with_silu_cache(
                    &current,
                    rows,
                    current_dim,
                    &weight.values,
                    out_dim,
                    1.0,
                    dtype,
                )?;
            current = activated;
            silu_derivatives[index] = silu_derivative;
        } else {
            current = nequix_dense_forward_scaled_maybe_parallel_rows(
                &current,
                rows,
                current_dim,
                &weight.values,
                out_dim,
                1.0,
                dtype,
            )?;
        }
        current_dim = out_dim;
    }
    Ok(NequixRadialMlpCache {
        output: current,
        silu_derivatives,
    })
}

pub fn nequix_radial_mlp_vjp_input(
    radial_basis: &[f64],
    rows: usize,
    weights: &[&NequixTensor],
    grad_output: &[f64],
) -> Result<Vec<f64>, NequixError> {
    if weights.is_empty() {
        return Err(invalid_model(
            "radial MLP VJP must contain at least one weight",
        ));
    }
    let first_shape = weights[0].shape.as_slice();
    if first_shape.len() != 2 {
        return Err(invalid_model(format!(
            "radial MLP VJP weight must be rank 2, got {:?}",
            weights[0].shape
        )));
    }
    let mut activations = vec![radial_basis.to_vec()];
    let mut pre_activations = Vec::new();
    let mut current_dim = first_shape[0];
    if radial_basis.len() != rows * current_dim {
        return Err(invalid_model(format!(
            "radial MLP VJP input length must be rows * first_dim = {}, got {}",
            rows * current_dim,
            radial_basis.len()
        )));
    }
    for (index, weight) in weights.iter().enumerate() {
        if weight.shape.len() != 2 {
            return Err(invalid_model(format!(
                "radial MLP VJP weight {} must be rank 2, got {:?}",
                weight.tree_path, weight.shape
            )));
        }
        if weight.shape[0] != current_dim {
            return Err(invalid_model(format!(
                "radial MLP VJP weight {} input dim must be {}, got {}",
                weight.tree_path, current_dim, weight.shape[0]
            )));
        }
        let out_dim = weight.shape[1];
        let previous = &activations[activations.len() - 1];
        let mut current = nequix_dense_forward_scaled_maybe_parallel_rows(
            previous,
            rows,
            current_dim,
            &weight.values,
            out_dim,
            1.0,
            NequixComputeDtype::Float64,
        )?;
        current_dim = out_dim;
        if index + 1 != weights.len() {
            pre_activations.push(current.clone());
            nequix_silu_in_place(&mut current);
        }
        activations.push(current);
    }
    if grad_output.len() != rows * current_dim {
        return Err(invalid_model(format!(
            "radial MLP VJP grad_output length must be rows * output_dim = {}, got {}",
            rows * current_dim,
            grad_output.len()
        )));
    }
    let mut grad = grad_output.to_vec();
    for layer_index in (0..weights.len()).rev() {
        if layer_index + 1 != weights.len() {
            let pre_activation = &pre_activations[layer_index];
            nequix_mul_silu_derivative_in_place(&mut grad, pre_activation);
        }
        let weight = weights[layer_index];
        grad = nequix_dense_forward_scaled_vjp_input(
            &grad,
            rows,
            weight.shape[0],
            &weight.values,
            weight.shape[1],
            1.0,
        )?;
    }
    Ok(grad)
}

fn nequix_radial_mlp_vjp_input_from_cache(
    radial_basis: &[f64],
    rows: usize,
    weights: &[&NequixTensor],
    silu_derivatives: &[Vec<f64>; 2],
    grad_output: &[f64],
    dtype: NequixComputeDtype,
) -> Result<Vec<f64>, NequixError> {
    if weights.len() != 3 {
        return Err(invalid_model(format!(
            "cached radial MLP VJP expects exactly 3 weights, got {}",
            weights.len()
        )));
    }
    if weights[0].shape.len() != 2 || weights[1].shape.len() != 2 || weights[2].shape.len() != 2 {
        return Err(invalid_model(
            "cached radial MLP VJP weights must be rank 2",
        ));
    }
    let input_dim = weights[0].shape[0];
    if radial_basis.len() != rows * input_dim {
        return Err(invalid_model(format!(
            "radial MLP VJP input length must be rows * first_dim = {}, got {}",
            rows * input_dim,
            radial_basis.len()
        )));
    }
    let output_dim = weights[2].shape[1];
    if grad_output.len() != rows * output_dim {
        return Err(invalid_model(format!(
            "radial MLP VJP grad_output length must be rows * output_dim = {}, got {}",
            rows * output_dim,
            grad_output.len()
        )));
    }
    let mut grad = nequix_dense_vjp_input_maybe_parallel_rows_with_silu_derivative(
        grad_output,
        rows,
        weights[2].shape[0],
        &weights[2].values,
        weights[2].shape[1],
        1.0,
        &silu_derivatives[1],
        dtype,
    )?;
    grad = nequix_dense_vjp_input_maybe_parallel_rows_with_silu_derivative(
        &grad,
        rows,
        weights[1].shape[0],
        &weights[1].values,
        weights[1].shape[1],
        1.0,
        &silu_derivatives[0],
        dtype,
    )?;
    nequix_dense_vjp_input_maybe_parallel_rows(
        &grad,
        rows,
        weights[0].shape[0],
        &weights[0].values,
        weights[0].shape[1],
        1.0,
        dtype,
    )
}

pub fn nequix_scatter_sum_by_receiver(
    edge_values: &[f64],
    edge_count: usize,
    value_dim: usize,
    receivers: &[usize],
    output_size: usize,
) -> Result<Vec<f64>, NequixError> {
    if edge_values.len() != edge_count * value_dim {
        return Err(invalid_model(format!(
            "scatter edge value length must be edge_count * value_dim = {}, got {}",
            edge_count * value_dim,
            edge_values.len()
        )));
    }
    if receivers.len() != edge_count {
        return Err(invalid_model(format!(
            "scatter receiver count must be edge_count = {edge_count}, got {}",
            receivers.len()
        )));
    }
    let mut output = vec![0.0; output_size * value_dim];
    for (edge_index, receiver) in receivers.iter().enumerate() {
        if *receiver >= output_size {
            return Err(invalid_model(format!(
                "scatter receiver index {receiver} is out of bounds for output size {output_size}"
            )));
        }
        let edge_row = &edge_values[edge_index * value_dim..(edge_index + 1) * value_dim];
        let out_row = &mut output[*receiver * value_dim..(*receiver + 1) * value_dim];
        for (out, value) in out_row.iter_mut().zip(edge_row) {
            *out += value;
        }
    }
    Ok(output)
}

pub fn nequix_gather_rows(
    values: &[f64],
    rows: usize,
    dim: usize,
    indices: &[usize],
) -> Result<Vec<f64>, NequixError> {
    if values.len() != rows * dim {
        return Err(invalid_model(format!(
            "gather value length must be rows * dim = {}, got {}",
            rows * dim,
            values.len()
        )));
    }
    let mut output = vec![0.0; indices.len() * dim];
    for (out_row_index, input_row_index) in indices.iter().enumerate() {
        if *input_row_index >= rows {
            return Err(invalid_model(format!(
                "gather row index {input_row_index} is out of bounds for row count {rows}"
            )));
        }
        let input_row = &values[*input_row_index * dim..(*input_row_index + 1) * dim];
        let output_row = &mut output[out_row_index * dim..(out_row_index + 1) * dim];
        output_row.copy_from_slice(input_row);
    }
    Ok(output)
}

pub fn nequix_edgewise_multiply(
    left: &[f64],
    right: &[f64],
    rows: usize,
    dim: usize,
) -> Result<Vec<f64>, NequixError> {
    let expected = rows * dim;
    if left.len() != expected || right.len() != expected {
        return Err(invalid_model(format!(
            "edgewise multiply inputs must both have length {expected}, got {} and {}",
            left.len(),
            right.len()
        )));
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .collect())
}

pub fn nequix_multiply_irrep_channels_by_weights(
    values: &[f64],
    rows: usize,
    irrep_blocks: &[(usize, usize)],
    weights: &[f64],
) -> Result<Vec<f64>, NequixError> {
    let value_dim = irrep_blocks.iter().try_fold(0usize, |acc, (mul, ell)| {
        acc.checked_add(mul * (2 * ell + 1))
            .ok_or_else(|| invalid_model("irrep value dimension overflows usize"))
    })?;
    let weight_dim = irrep_blocks.iter().try_fold(0usize, |acc, (mul, _)| {
        acc.checked_add(*mul)
            .ok_or_else(|| invalid_model("irrep weight dimension overflows usize"))
    })?;
    if values.len() != rows * value_dim {
        return Err(invalid_model(format!(
            "irrep value length must be rows * value_dim = {}, got {}",
            rows * value_dim,
            values.len()
        )));
    }
    if weights.len() != rows * weight_dim {
        return Err(invalid_model(format!(
            "irrep weight length must be rows * weight_dim = {}, got {}",
            rows * weight_dim,
            weights.len()
        )));
    }
    let mut output = values.to_vec();
    for row in 0..rows {
        let mut value_offset = row * value_dim;
        let mut weight_offset = row * weight_dim;
        for (mul, ell) in irrep_blocks {
            let component_count = 2 * ell + 1;
            for _ in 0..*mul {
                let weight = weights[weight_offset];
                weight_offset += 1;
                for component in 0..component_count {
                    output[value_offset + component] *= weight;
                }
                value_offset += component_count;
            }
        }
    }
    Ok(output)
}

pub fn nequix_scalar_sh_tensor_product(
    scalars: &[f64],
    rows: usize,
    scalar_channels: usize,
    spherical_harmonics: &[f64],
    l_max: usize,
) -> Result<Vec<f64>, NequixError> {
    let sh_dim = (l_max + 1) * (l_max + 1);
    if scalars.len() != rows * scalar_channels {
        return Err(invalid_model(format!(
            "scalar TP input length must be rows * scalar_channels = {}, got {}",
            rows * scalar_channels,
            scalars.len()
        )));
    }
    if spherical_harmonics.len() != rows * sh_dim {
        return Err(invalid_model(format!(
            "scalar TP SH length must be rows * sh_dim = {}, got {}",
            rows * sh_dim,
            spherical_harmonics.len()
        )));
    }
    let output_dim: usize = (0..=l_max).map(|ell| scalar_channels * (2 * ell + 1)).sum();
    let mut output = vec![0.0; rows * output_dim];
    for row in 0..rows {
        let scalar_row = &scalars[row * scalar_channels..(row + 1) * scalar_channels];
        let sh_row = &spherical_harmonics[row * sh_dim..(row + 1) * sh_dim];
        let output_row = &mut output[row * output_dim..(row + 1) * output_dim];
        let mut out_offset = 0;
        for ell in 0..=l_max {
            let component_count = 2 * ell + 1;
            let sh_offset = ell * ell;
            for scalar in scalar_row {
                for component in 0..component_count {
                    output_row[out_offset + component] = scalar * sh_row[sh_offset + component];
                }
                out_offset += component_count;
            }
        }
    }
    Ok(output)
}

pub struct NequixIrrepLinearBlock<'a> {
    pub input_offset: usize,
    pub input_mul: usize,
    pub output_offset: usize,
    pub output_mul: usize,
    pub ell: usize,
    pub weight: &'a [f64],
    pub alpha: f64,
}

pub struct NequixNormBlock<'a> {
    pub offset: usize,
    pub mul: usize,
    pub ell: usize,
    pub affine_weight: &'a [f64],
}

pub struct NequixGateBlock {
    pub input_offset: usize,
    pub output_offset: usize,
    pub gate_offset: usize,
    pub mul: usize,
    pub ell: usize,
}

#[derive(Deserialize)]
struct NequixTensorProductPlanSpec {
    schema_version: String,
    irreps_in1: String,
    irreps_in2: String,
    irreps_out: String,
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

pub fn nequix_tensor_product_plan_from_json(
    value: &serde_json::Value,
) -> Result<TensorProductPlan, NequixError> {
    let spec: NequixTensorProductPlanSpec = serde_json::from_value(value.clone())
        .map_err(|err| invalid_model(format!("invalid NequiX tensor-product plan JSON: {err}")))?;
    if spec.schema_version != TP_PLAN_SCHEMA_VERSION {
        return Err(invalid_model(format!(
            "unsupported NequiX tensor-product plan schema '{}', expected '{}'",
            spec.schema_version, TP_PLAN_SCHEMA_VERSION
        )));
    }

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
                "NequiX tensor-product plan {name} length {len} does not match instruction length {instruction_len}"
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
                "NequiX tensor-product plan {name} length {len} does not match CG length {cg_len}"
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
    let weight_numel = instruction_len;

    TensorProductPlan::new(
        spec.irreps_in1
            .parse::<Irreps>()
            .map_err(|err| invalid_model(format!("invalid NequiX TP input irreps: {err}")))?,
        spec.irreps_in2
            .parse::<Irreps>()
            .map_err(|err| invalid_model(format!("invalid NequiX TP SH irreps: {err}")))?,
        spec.irreps_out
            .parse::<Irreps>()
            .map_err(|err| invalid_model(format!("invalid NequiX TP output irreps: {err}")))?,
        instructions,
        cg_entries,
        weight_numel,
    )
    .map_err(|err| invalid_model(format!("invalid NequiX tensor-product plan: {err}")))
}

fn nequix_tensor_product_runtime_plan(
    plan: &TensorProductPlan,
) -> Result<NequixTensorProductRuntimePlan, NequixError> {
    let layouts = plan
        .checked_layouts()
        .map_err(|err| invalid_model(format!("NeQuiX TP layouts failed: {err}")))?;
    let instructions = plan
        .instructions
        .iter()
        .map(|instruction| {
            let in1 = layouts.in1[instruction.in1_segment];
            let in2 = layouts.in2[instruction.in2_segment];
            let out = layouts.out[instruction.out_segment];
            let weight_index = instruction
                .weight_offset
                .ok_or_else(|| invalid_model("NeQuiX TP instruction is missing radial weight"))?;
            let kind = nequix_tensor_product_instruction_kind(
                &plan.cg_entries,
                instruction.cg_offset,
                instruction.cg_len,
                instruction.path_weight,
            );
            Ok(NequixTensorProductRuntimeInstruction {
                input1_start: in1.start,
                input2_start: in2.start,
                out_start: out.start,
                cg_offset: instruction.cg_offset,
                cg_len: instruction.cg_len,
                weight_index,
                path_weight: instruction.path_weight,
                kind,
            })
        })
        .collect::<Result<Vec<_>, NequixError>>()?;
    Ok(NequixTensorProductRuntimePlan {
        unit_path_weights: instructions
            .iter()
            .all(|instruction| instruction.path_weight == 1.0),
        instructions,
        cg_entries: plan.cg_entries.clone(),
        input2_dim: plan.input2_dim(),
        output_dim: plan.output_dim(),
        weight_numel: plan.weight_numel,
    })
}

fn nequix_tensor_product_instruction_kind(
    cg_entries: &[CgEntry],
    cg_offset: usize,
    cg_len: usize,
    path_weight: f64,
) -> NequixTensorProductRuntimeInstructionKind {
    if path_weight != 1.0 || cg_len == 0 || cg_offset + cg_len > cg_entries.len() {
        return NequixTensorProductRuntimeInstructionKind::Generic;
    }
    let entries = &cg_entries[cg_offset..cg_offset + cg_len];
    let coefficient = entries[0].value;
    let same_coefficient = entries.iter().all(|entry| entry.value == coefficient);
    if !same_coefficient {
        return NequixTensorProductRuntimeInstructionKind::Generic;
    }
    if entries.iter().enumerate().all(|(component, entry)| {
        entry.i_m == 0 && entry.j_m == component && entry.k_m == component
    }) {
        return NequixTensorProductRuntimeInstructionKind::ScalarTimesY {
            component_count: cg_len,
            coefficient,
        };
    }
    if entries.iter().enumerate().all(|(component, entry)| {
        entry.i_m == component && entry.j_m == 0 && entry.k_m == component
    }) {
        return NequixTensorProductRuntimeInstructionKind::XTimesScalar {
            component_count: cg_len,
            coefficient,
        };
    }
    if entries.iter().enumerate().all(|(component, entry)| {
        entry.i_m == component && entry.j_m == component && entry.k_m == 0
    }) {
        return NequixTensorProductRuntimeInstructionKind::DotToScalar {
            component_count: cg_len,
            coefficient,
        };
    }
    NequixTensorProductRuntimeInstructionKind::Generic
}

pub fn nequix_irrep_linear_forward(
    input: &[f64],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    blocks: &[NequixIrrepLinearBlock<'_>],
) -> Result<Vec<f64>, NequixError> {
    nequix_irrep_linear_forward_dtype(
        input,
        rows,
        input_dim,
        output_dim,
        blocks,
        NequixComputeDtype::Float64,
    )
}

fn nequix_irrep_linear_forward_dtype(
    input: &[f64],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    blocks: &[NequixIrrepLinearBlock<'_>],
    dtype: NequixComputeDtype,
) -> Result<Vec<f64>, NequixError> {
    if dtype.is_f32() {
        return nequix_irrep_linear_forward_f32_promoted(
            input, rows, input_dim, output_dim, blocks,
        );
    }
    if input.len() != rows * input_dim {
        return Err(invalid_model(format!(
            "irrep linear input length must be rows * input_dim = {}, got {}",
            rows * input_dim,
            input.len()
        )));
    }
    let mut output = vec![0.0; rows * output_dim];
    for block in blocks {
        let component_count = 2 * block.ell + 1;
        if block.weight.len() != block.input_mul * block.output_mul {
            return Err(invalid_model(format!(
                "irrep linear weight length must be input_mul * output_mul = {}, got {}",
                block.input_mul * block.output_mul,
                block.weight.len()
            )));
        }
        if block.input_offset + block.input_mul * component_count > input_dim {
            return Err(invalid_model("irrep linear input block is out of bounds"));
        }
        if block.output_offset + block.output_mul * component_count > output_dim {
            return Err(invalid_model("irrep linear output block is out of bounds"));
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
        if block.ell == 0 {
            blas_backend::dense_forward_strided(
                &input[block.input_offset..],
                rows,
                block.input_mul,
                input_dim,
                block.weight,
                &mut output[block.output_offset..],
                block.output_mul,
                output_dim,
                block.alpha,
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
            ),
            all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
        ))]
        {
            let packed_rows = rows * component_count;
            let mut packed_input = vec![0.0; packed_rows * block.input_mul];
            for row in 0..rows {
                let input_row = &input[row * input_dim..(row + 1) * input_dim];
                for component in 0..component_count {
                    let packed_row = &mut packed_input[(row * component_count + component)
                        * block.input_mul
                        ..(row * component_count + component + 1) * block.input_mul];
                    for input_mul in 0..block.input_mul {
                        packed_row[input_mul] =
                            input_row[block.input_offset + input_mul * component_count + component];
                    }
                }
            }
            let mut packed_output = vec![0.0; packed_rows * block.output_mul];
            blas_backend::dense_forward(
                &packed_input,
                packed_rows,
                block.input_mul,
                block.weight,
                &mut packed_output,
                block.alpha,
            );
            for row in 0..rows {
                let output_row = &mut output[row * output_dim..(row + 1) * output_dim];
                for component in 0..component_count {
                    let packed_row = &packed_output[(row * component_count + component)
                        * block.output_mul
                        ..(row * component_count + component + 1) * block.output_mul];
                    for output_mul in 0..block.output_mul {
                        output_row
                            [block.output_offset + output_mul * component_count + component] +=
                            packed_row[output_mul];
                    }
                }
            }
            continue;
        }
        #[cfg(not(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            ),
            all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
        )))]
        {
            for row in 0..rows {
                let input_row = &input[row * input_dim..(row + 1) * input_dim];
                let output_row = &mut output[row * output_dim..(row + 1) * output_dim];
                for input_mul in 0..block.input_mul {
                    for output_mul in 0..block.output_mul {
                        let weight =
                            block.alpha * block.weight[input_mul * block.output_mul + output_mul];
                        if weight == 0.0 {
                            continue;
                        }
                        for component in 0..component_count {
                            let input_index =
                                block.input_offset + input_mul * component_count + component;
                            let output_index =
                                block.output_offset + output_mul * component_count + component;
                            output_row[output_index] += weight * input_row[input_index];
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

/// f32-promoted irrep-linear forward boundary. Mirrors the f64 packed/strided
/// matmul structure but computes each block contribution in f32 (BLAS f32 when
/// available, scalar fallback otherwise) and promotes the block result back to
/// f64 before accumulating into the output. Keeps the rest of the pipeline in
/// f64, matching the MACE-MH-1 precedent.
fn nequix_irrep_linear_forward_f32_promoted(
    input: &[f64],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    blocks: &[NequixIrrepLinearBlock<'_>],
) -> Result<Vec<f64>, NequixError> {
    if input.len() != rows * input_dim {
        return Err(invalid_model(format!(
            "irrep linear input length must be rows * input_dim = {}, got {}",
            rows * input_dim,
            input.len()
        )));
    }
    let input_f32: Vec<f32> = input.iter().map(|&value| value as f32).collect();
    let mut output = vec![0.0_f64; rows * output_dim];
    for block in blocks {
        let component_count = 2 * block.ell + 1;
        if block.weight.len() != block.input_mul * block.output_mul {
            return Err(invalid_model(format!(
                "irrep linear weight length must be input_mul * output_mul = {}, got {}",
                block.input_mul * block.output_mul,
                block.weight.len()
            )));
        }
        if block.input_offset + block.input_mul * component_count > input_dim {
            return Err(invalid_model("irrep linear input block is out of bounds"));
        }
        if block.output_offset + block.output_mul * component_count > output_dim {
            return Err(invalid_model("irrep linear output block is out of bounds"));
        }
        let weight_f32: Vec<f32> = block.weight.iter().map(|&value| value as f32).collect();
        let alpha_f32 = block.alpha as f32;
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        {
            if block.ell == 0 {
                // Strided f32 GEMM writes into a fresh per-block f32 strided
                // output, then promote-add into the f64 output.
                let mut block_output_f32 = vec![0.0_f32; rows * output_dim];
                blas_backend::dense_forward_strided_f32(
                    &input_f32[block.input_offset..],
                    rows,
                    block.input_mul,
                    input_dim,
                    &weight_f32,
                    &mut block_output_f32[block.output_offset..],
                    block.output_mul,
                    output_dim,
                    alpha_f32,
                    0.0,
                );
                for (dst, src) in output.iter_mut().zip(&block_output_f32) {
                    *dst += *src as f64;
                }
                continue;
            }
            let packed_rows = rows * component_count;
            let mut packed_input = vec![0.0_f32; packed_rows * block.input_mul];
            for row in 0..rows {
                let input_row = &input_f32[row * input_dim..(row + 1) * input_dim];
                for component in 0..component_count {
                    let packed_row = &mut packed_input[(row * component_count + component)
                        * block.input_mul
                        ..(row * component_count + component + 1) * block.input_mul];
                    for input_mul in 0..block.input_mul {
                        packed_row[input_mul] =
                            input_row[block.input_offset + input_mul * component_count + component];
                    }
                }
            }
            let mut packed_output = vec![0.0_f32; packed_rows * block.output_mul];
            blas_backend::dense_forward_f32(
                &packed_input,
                packed_rows,
                block.input_mul,
                &weight_f32,
                &mut packed_output,
                alpha_f32,
            );
            for row in 0..rows {
                let output_row = &mut output[row * output_dim..(row + 1) * output_dim];
                for component in 0..component_count {
                    let packed_row = &packed_output[(row * component_count + component)
                        * block.output_mul
                        ..(row * component_count + component + 1) * block.output_mul];
                    for output_mul in 0..block.output_mul {
                        output_row
                            [block.output_offset + output_mul * component_count + component] +=
                            packed_row[output_mul] as f64;
                    }
                }
            }
            continue;
        }
        #[cfg(not(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )))]
        {
            for row in 0..rows {
                let input_row = &input_f32[row * input_dim..(row + 1) * input_dim];
                let output_row = &mut output[row * output_dim..(row + 1) * output_dim];
                for input_mul in 0..block.input_mul {
                    for output_mul in 0..block.output_mul {
                        let weight =
                            alpha_f32 * weight_f32[input_mul * block.output_mul + output_mul];
                        if weight == 0.0 {
                            continue;
                        }
                        for component in 0..component_count {
                            let input_index =
                                block.input_offset + input_mul * component_count + component;
                            let output_index =
                                block.output_offset + output_mul * component_count + component;
                            output_row[output_index] += (weight * input_row[input_index]) as f64;
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

pub fn nequix_irrep_linear_vjp_input(
    grad_output: &[f64],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    blocks: &[NequixIrrepLinearBlock<'_>],
) -> Result<Vec<f64>, NequixError> {
    nequix_irrep_linear_vjp_input_dtype(
        grad_output,
        rows,
        input_dim,
        output_dim,
        blocks,
        NequixComputeDtype::Float64,
    )
}

fn nequix_irrep_linear_vjp_input_dtype(
    grad_output: &[f64],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    blocks: &[NequixIrrepLinearBlock<'_>],
    dtype: NequixComputeDtype,
) -> Result<Vec<f64>, NequixError> {
    if dtype.is_f32() {
        return nequix_irrep_linear_vjp_input_f32_promoted(
            grad_output,
            rows,
            input_dim,
            output_dim,
            blocks,
        );
    }
    if grad_output.len() != rows * output_dim {
        return Err(invalid_model(format!(
            "irrep linear VJP grad_output length must be rows * output_dim = {}, got {}",
            rows * output_dim,
            grad_output.len()
        )));
    }
    let mut grad_input = vec![0.0; rows * input_dim];
    for block in blocks {
        let component_count = 2 * block.ell + 1;
        if block.weight.len() != block.input_mul * block.output_mul {
            return Err(invalid_model(format!(
                "irrep linear VJP weight length must be input_mul * output_mul = {}, got {}",
                block.input_mul * block.output_mul,
                block.weight.len()
            )));
        }
        if block.input_offset + block.input_mul * component_count > input_dim {
            return Err(invalid_model(
                "irrep linear VJP input block is out of bounds",
            ));
        }
        if block.output_offset + block.output_mul * component_count > output_dim {
            return Err(invalid_model(
                "irrep linear VJP output block is out of bounds",
            ));
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
        if block.ell == 0 {
            blas_backend::dense_vjp_input_strided(
                rows,
                block.input_mul,
                block.weight,
                block.output_mul,
                &grad_output[block.output_offset..],
                output_dim,
                &mut grad_input[block.input_offset..],
                input_dim,
                block.alpha,
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
            ),
            all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
        ))]
        {
            let packed_rows = rows * component_count;
            let mut packed_grad_output = vec![0.0; packed_rows * block.output_mul];
            for row in 0..rows {
                let grad_output_row = &grad_output[row * output_dim..(row + 1) * output_dim];
                for component in 0..component_count {
                    let packed_row = &mut packed_grad_output[(row * component_count + component)
                        * block.output_mul
                        ..(row * component_count + component + 1) * block.output_mul];
                    for output_mul in 0..block.output_mul {
                        packed_row[output_mul] = grad_output_row
                            [block.output_offset + output_mul * component_count + component];
                    }
                }
            }
            let mut packed_grad_input = vec![0.0; packed_rows * block.input_mul];
            blas_backend::dense_vjp_input(
                packed_rows,
                block.input_mul,
                block.weight,
                &packed_grad_output,
                &mut packed_grad_input,
                block.alpha,
            );
            for row in 0..rows {
                let grad_input_row = &mut grad_input[row * input_dim..(row + 1) * input_dim];
                for component in 0..component_count {
                    let packed_row = &packed_grad_input[(row * component_count + component)
                        * block.input_mul
                        ..(row * component_count + component + 1) * block.input_mul];
                    for input_mul in 0..block.input_mul {
                        grad_input_row
                            [block.input_offset + input_mul * component_count + component] +=
                            packed_row[input_mul];
                    }
                }
            }
            continue;
        }
        #[cfg(not(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            ),
            all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
        )))]
        {
            for row in 0..rows {
                let grad_output_row = &grad_output[row * output_dim..(row + 1) * output_dim];
                let grad_input_row = &mut grad_input[row * input_dim..(row + 1) * input_dim];
                for input_mul in 0..block.input_mul {
                    for output_mul in 0..block.output_mul {
                        let weight =
                            block.alpha * block.weight[input_mul * block.output_mul + output_mul];
                        if weight == 0.0 {
                            continue;
                        }
                        for component in 0..component_count {
                            let input_index =
                                block.input_offset + input_mul * component_count + component;
                            let output_index =
                                block.output_offset + output_mul * component_count + component;
                            grad_input_row[input_index] += weight * grad_output_row[output_index];
                        }
                    }
                }
            }
        }
    }
    Ok(grad_input)
}

/// f32-promoted irrep-linear VJP boundary mirroring
/// [`nequix_irrep_linear_forward_f32_promoted`]: each block's transpose matmul
/// is computed in f32 and promoted to f64, keeping the analytic backward
/// consistent with the f32-promoted forward.
fn nequix_irrep_linear_vjp_input_f32_promoted(
    grad_output: &[f64],
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    blocks: &[NequixIrrepLinearBlock<'_>],
) -> Result<Vec<f64>, NequixError> {
    if grad_output.len() != rows * output_dim {
        return Err(invalid_model(format!(
            "irrep linear VJP grad_output length must be rows * output_dim = {}, got {}",
            rows * output_dim,
            grad_output.len()
        )));
    }
    let grad_output_f32: Vec<f32> = grad_output.iter().map(|&value| value as f32).collect();
    let mut grad_input = vec![0.0_f64; rows * input_dim];
    for block in blocks {
        let component_count = 2 * block.ell + 1;
        if block.weight.len() != block.input_mul * block.output_mul {
            return Err(invalid_model(format!(
                "irrep linear VJP weight length must be input_mul * output_mul = {}, got {}",
                block.input_mul * block.output_mul,
                block.weight.len()
            )));
        }
        if block.input_offset + block.input_mul * component_count > input_dim {
            return Err(invalid_model(
                "irrep linear VJP input block is out of bounds",
            ));
        }
        if block.output_offset + block.output_mul * component_count > output_dim {
            return Err(invalid_model(
                "irrep linear VJP output block is out of bounds",
            ));
        }
        let weight_f32: Vec<f32> = block.weight.iter().map(|&value| value as f32).collect();
        let alpha_f32 = block.alpha as f32;
        #[cfg(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        ))]
        {
            if block.ell == 0 {
                let mut block_grad_input_f32 = vec![0.0_f32; rows * input_dim];
                blas_backend::dense_vjp_input_strided_f32(
                    rows,
                    block.input_mul,
                    &weight_f32,
                    block.output_mul,
                    &grad_output_f32[block.output_offset..],
                    output_dim,
                    &mut block_grad_input_f32[block.input_offset..],
                    input_dim,
                    alpha_f32,
                    0.0,
                );
                for (dst, src) in grad_input.iter_mut().zip(&block_grad_input_f32) {
                    *dst += *src as f64;
                }
                continue;
            }
            let packed_rows = rows * component_count;
            let mut packed_grad_output = vec![0.0_f32; packed_rows * block.output_mul];
            for row in 0..rows {
                let grad_output_row = &grad_output_f32[row * output_dim..(row + 1) * output_dim];
                for component in 0..component_count {
                    let packed_row = &mut packed_grad_output[(row * component_count + component)
                        * block.output_mul
                        ..(row * component_count + component + 1) * block.output_mul];
                    for output_mul in 0..block.output_mul {
                        packed_row[output_mul] = grad_output_row
                            [block.output_offset + output_mul * component_count + component];
                    }
                }
            }
            let mut packed_grad_input = vec![0.0_f32; packed_rows * block.input_mul];
            blas_backend::dense_vjp_input_f32(
                packed_rows,
                block.input_mul,
                &weight_f32,
                &packed_grad_output,
                &mut packed_grad_input,
                alpha_f32,
            );
            for row in 0..rows {
                let grad_input_row = &mut grad_input[row * input_dim..(row + 1) * input_dim];
                for component in 0..component_count {
                    let packed_row = &packed_grad_input[(row * component_count + component)
                        * block.input_mul
                        ..(row * component_count + component + 1) * block.input_mul];
                    for input_mul in 0..block.input_mul {
                        grad_input_row
                            [block.input_offset + input_mul * component_count + component] +=
                            packed_row[input_mul] as f64;
                    }
                }
            }
            continue;
        }
        #[cfg(not(any(
            all(feature = "accelerate-blas", target_os = "macos"),
            all(
                any(feature = "intel-mkl", feature = "openblas"),
                target_os = "linux",
                target_arch = "x86_64"
            )
        )))]
        {
            for row in 0..rows {
                let grad_output_row = &grad_output_f32[row * output_dim..(row + 1) * output_dim];
                let grad_input_row = &mut grad_input[row * input_dim..(row + 1) * input_dim];
                for input_mul in 0..block.input_mul {
                    for output_mul in 0..block.output_mul {
                        let weight =
                            alpha_f32 * weight_f32[input_mul * block.output_mul + output_mul];
                        if weight == 0.0 {
                            continue;
                        }
                        for component in 0..component_count {
                            let input_index =
                                block.input_offset + input_mul * component_count + component;
                            let output_index =
                                block.output_offset + output_mul * component_count + component;
                            grad_input_row[input_index] +=
                                (weight * grad_output_row[output_index]) as f64;
                        }
                    }
                }
            }
        }
    }
    Ok(grad_input)
}

fn nequix_norm_block_count(
    blocks: &[NequixNormBlock<'_>],
    operation: &str,
) -> Result<f64, NequixError> {
    if blocks.is_empty() {
        return Err(invalid_model(format!(
            "{operation} must contain at least one block"
        )));
    }
    Ok(blocks.len() as f64)
}

pub fn nequix_rms_layer_norm(
    input: &[f64],
    rows: usize,
    input_dim: usize,
    blocks: &[NequixNormBlock<'_>],
    eps: f64,
) -> Result<Vec<f64>, NequixError> {
    if input.len() != rows * input_dim {
        return Err(invalid_model(format!(
            "RMS norm input length must be rows * input_dim = {}, got {}",
            rows * input_dim,
            input.len()
        )));
    }
    let block_count = nequix_norm_block_count(blocks, "RMS norm")?;
    for block in blocks {
        if block.affine_weight.len() != block.mul {
            return Err(invalid_model(format!(
                "RMS norm affine weight length must be mul = {}, got {}",
                block.mul,
                block.affine_weight.len()
            )));
        }
        let component_count = 2 * block.ell + 1;
        if block.offset + block.mul * component_count > input_dim {
            return Err(invalid_model("RMS norm block is out of bounds"));
        }
    }
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let input_row = &input[row * input_dim..(row + 1) * input_dim];
        let output_row = &mut output[row * input_dim..(row + 1) * input_dim];
        let mut norm_sum = 0.0;
        for block in blocks {
            let component_count = 2 * block.ell + 1;
            let weight = 1.0 / (component_count as f64 * block_count);
            let mut component_norm_sum = 0.0;
            for component in 0..component_count {
                let mut mul_sum = 0.0;
                for mul in 0..block.mul {
                    let index = block.offset + mul * component_count + component;
                    mul_sum += input_row[index] * input_row[index] * weight;
                }
                component_norm_sum += mul_sum;
            }
            norm_sum += component_norm_sum / component_count as f64;
        }
        let norm = (norm_sum + eps).powf(-0.5);
        for block in blocks {
            let component_count = 2 * block.ell + 1;
            for mul in 0..block.mul {
                let affine = block.affine_weight[mul];
                for component in 0..component_count {
                    let index = block.offset + mul * component_count + component;
                    output_row[index] = input_row[index] * affine * norm;
                }
            }
        }
    }
    Ok(output)
}

pub fn nequix_rms_layer_norm_vjp_input(
    input: &[f64],
    grad_output: &[f64],
    rows: usize,
    input_dim: usize,
    blocks: &[NequixNormBlock<'_>],
    eps: f64,
) -> Result<Vec<f64>, NequixError> {
    if input.len() != rows * input_dim || grad_output.len() != rows * input_dim {
        return Err(invalid_model(format!(
            "RMS norm VJP input and grad_output must both have length {}, got {} and {}",
            rows * input_dim,
            input.len(),
            grad_output.len()
        )));
    }
    let block_count = nequix_norm_block_count(blocks, "RMS norm VJP")?;
    for block in blocks {
        if block.affine_weight.len() != block.mul {
            return Err(invalid_model(format!(
                "RMS norm VJP affine weight length must be mul = {}, got {}",
                block.mul,
                block.affine_weight.len()
            )));
        }
        let component_count = 2 * block.ell + 1;
        if block.offset + block.mul * component_count > input_dim {
            return Err(invalid_model("RMS norm VJP block is out of bounds"));
        }
    }
    let mut grad_input = vec![0.0; input.len()];
    for row in 0..rows {
        let input_row = &input[row * input_dim..(row + 1) * input_dim];
        let grad_output_row = &grad_output[row * input_dim..(row + 1) * input_dim];
        let grad_input_row = &mut grad_input[row * input_dim..(row + 1) * input_dim];

        let mut norm_sum = 0.0;
        for block in blocks {
            let component_count = 2 * block.ell + 1;
            let weight = 1.0 / (component_count as f64 * block_count);
            let mut component_norm_sum = 0.0;
            for component in 0..component_count {
                let mut mul_sum = 0.0;
                for mul in 0..block.mul {
                    let index = block.offset + mul * component_count + component;
                    mul_sum += input_row[index] * input_row[index] * weight;
                }
                component_norm_sum += mul_sum;
            }
            norm_sum += component_norm_sum / component_count as f64;
        }
        let norm = (norm_sum + eps).powf(-0.5);
        let mut grad_norm = 0.0;
        for block in blocks {
            let component_count = 2 * block.ell + 1;
            for mul in 0..block.mul {
                let affine = block.affine_weight[mul];
                for component in 0..component_count {
                    let index = block.offset + mul * component_count + component;
                    grad_input_row[index] += grad_output_row[index] * affine * norm;
                    grad_norm += grad_output_row[index] * input_row[index] * affine;
                }
            }
        }
        let grad_norm_sum = -0.5 * grad_norm * norm * norm * norm;
        for block in blocks {
            let component_count = 2 * block.ell + 1;
            let weight = 1.0 / (component_count as f64 * block_count);
            let norm_sum_scale = weight / component_count as f64;
            for mul in 0..block.mul {
                for component in 0..component_count {
                    let index = block.offset + mul * component_count + component;
                    grad_input_row[index] +=
                        grad_norm_sum * 2.0 * input_row[index] * norm_sum_scale;
                }
            }
        }
    }
    Ok(grad_input)
}

pub fn nequix_gate_forward(
    input: &[f64],
    rows: usize,
    input_dim: usize,
    scalar_count: usize,
    output_dim: usize,
    blocks: &[NequixGateBlock],
) -> Result<Vec<f64>, NequixError> {
    if input.len() != rows * input_dim {
        return Err(invalid_model(format!(
            "gate input length must be rows * input_dim = {}, got {}",
            rows * input_dim,
            input.len()
        )));
    }
    if scalar_count > input_dim || scalar_count > output_dim {
        return Err(invalid_model("gate scalar count is out of bounds"));
    }
    for block in blocks {
        let component_count = 2 * block.ell + 1;
        if block.input_offset + block.mul * component_count > input_dim {
            return Err(invalid_model("gate input block is out of bounds"));
        }
        if block.output_offset + block.mul * component_count > output_dim {
            return Err(invalid_model("gate output block is out of bounds"));
        }
        if block.gate_offset + block.mul > input_dim {
            return Err(invalid_model("gate scalar block is out of bounds"));
        }
    }
    let mut output = vec![0.0; rows * output_dim];
    for row in 0..rows {
        let input_row = &input[row * input_dim..(row + 1) * input_dim];
        let output_row = &mut output[row * output_dim..(row + 1) * output_dim];
        for scalar in 0..scalar_count {
            output_row[scalar] = nequix_normalized_silu(input_row[scalar]);
        }
        for block in blocks {
            let component_count = 2 * block.ell + 1;
            for mul in 0..block.mul {
                let gate = nequix_normalized_silu(input_row[block.gate_offset + mul]);
                for component in 0..component_count {
                    let input_index = block.input_offset + mul * component_count + component;
                    let output_index = block.output_offset + mul * component_count + component;
                    output_row[output_index] = gate * input_row[input_index];
                }
            }
        }
    }
    Ok(output)
}

pub fn nequix_gate_vjp_input(
    input: &[f64],
    grad_output: &[f64],
    rows: usize,
    input_dim: usize,
    scalar_count: usize,
    output_dim: usize,
    blocks: &[NequixGateBlock],
) -> Result<Vec<f64>, NequixError> {
    if input.len() != rows * input_dim || grad_output.len() != rows * output_dim {
        return Err(invalid_model(format!(
            "gate VJP input length must be {} and grad_output length must be {}, got {} and {}",
            rows * input_dim,
            rows * output_dim,
            input.len(),
            grad_output.len()
        )));
    }
    if scalar_count > input_dim || scalar_count > output_dim {
        return Err(invalid_model("gate VJP scalar count is out of bounds"));
    }
    for block in blocks {
        let component_count = 2 * block.ell + 1;
        if block.input_offset + block.mul * component_count > input_dim {
            return Err(invalid_model("gate VJP input block is out of bounds"));
        }
        if block.output_offset + block.mul * component_count > output_dim {
            return Err(invalid_model("gate VJP output block is out of bounds"));
        }
        if block.gate_offset + block.mul > input_dim {
            return Err(invalid_model("gate VJP scalar block is out of bounds"));
        }
    }
    let mut grad_input = vec![0.0; rows * input_dim];
    for row in 0..rows {
        let input_row = &input[row * input_dim..(row + 1) * input_dim];
        let grad_output_row = &grad_output[row * output_dim..(row + 1) * output_dim];
        let grad_input_row = &mut grad_input[row * input_dim..(row + 1) * input_dim];
        for scalar in 0..scalar_count {
            grad_input_row[scalar] +=
                grad_output_row[scalar] * nequix_normalized_silu_derivative(input_row[scalar]);
        }
        for block in blocks {
            let component_count = 2 * block.ell + 1;
            for mul in 0..block.mul {
                let gate_index = block.gate_offset + mul;
                let gate = nequix_normalized_silu(input_row[gate_index]);
                let mut grad_gate = 0.0;
                for component in 0..component_count {
                    let input_index = block.input_offset + mul * component_count + component;
                    let output_index = block.output_offset + mul * component_count + component;
                    let bar = grad_output_row[output_index];
                    grad_input_row[input_index] += bar * gate;
                    grad_gate += bar * input_row[input_index];
                }
                grad_input_row[gate_index] +=
                    grad_gate * nequix_normalized_silu_derivative(input_row[gate_index]);
            }
        }
    }
    Ok(grad_input)
}

pub fn add_same_shape(left: &[f64], right: &[f64]) -> Result<Vec<f64>, NequixError> {
    if left.len() != right.len() {
        return Err(invalid_model(format!(
            "add inputs must have the same length, got {} and {}",
            left.len(),
            right.len()
        )));
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| left + right)
        .collect())
}

fn add_assign(left: &mut [f64], right: &[f64]) -> Result<(), NequixError> {
    add_assign_slice(left, right)
}

fn add_assign_slice(left: &mut [f64], right: &[f64]) -> Result<(), NequixError> {
    if left.len() != right.len() {
        return Err(invalid_model(format!(
            "add_assign inputs must have the same length, got {} and {}",
            left.len(),
            right.len()
        )));
    }
    for (left, right) in left.iter_mut().zip(right) {
        *left += right;
    }
    Ok(())
}

fn nequix_tensor_product_accumulate_fast(
    plan: &NequixTensorProductRuntimePlan,
    x: &[f64],
    y: &[f64],
    weights: &[f64],
    output_scale: f64,
    out: &mut [f64],
) {
    for instruction in &plan.instructions {
        let weight = weights[instruction.weight_index];
        match instruction.kind {
            NequixTensorProductRuntimeInstructionKind::ScalarTimesY {
                component_count,
                coefficient,
            } => {
                let scale = output_scale * weight * coefficient * x[instruction.input1_start];
                for component in 0..component_count {
                    let out_index = instruction.out_start + component;
                    out[out_index] =
                        scale.mul_add(y[instruction.input2_start + component], out[out_index]);
                }
                continue;
            }
            NequixTensorProductRuntimeInstructionKind::XTimesScalar {
                component_count,
                coefficient,
            } => {
                let scale = output_scale * weight * coefficient;
                let y0 = y[instruction.input2_start];
                for component in 0..component_count {
                    let out_index = instruction.out_start + component;
                    out[out_index] = (scale * x[instruction.input1_start + component])
                        .mul_add(y0, out[out_index]);
                }
                continue;
            }
            NequixTensorProductRuntimeInstructionKind::DotToScalar {
                component_count,
                coefficient,
            } => {
                let scale = output_scale * weight * coefficient;
                let out_index = instruction.out_start;
                for component in 0..component_count {
                    out[out_index] = (scale * x[instruction.input1_start + component])
                        .mul_add(y[instruction.input2_start + component], out[out_index]);
                }
                continue;
            }
            NequixTensorProductRuntimeInstructionKind::Generic => {}
        }
        let scale = if plan.unit_path_weights {
            output_scale * weight
        } else {
            output_scale * instruction.path_weight * weight
        };
        let cg_end = instruction.cg_offset + instruction.cg_len;
        for cg in &plan.cg_entries[instruction.cg_offset..cg_end] {
            let out_index = instruction.out_start + cg.k_m;
            out[out_index] = (scale * cg.value * x[instruction.input1_start + cg.i_m])
                .mul_add(y[instruction.input2_start + cg.j_m], out[out_index]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn nequix_tensor_product_vjp_fast(
    plan: &NequixTensorProductRuntimePlan,
    x: &[f64],
    y: &[f64],
    weights: &[f64],
    grad_out: &[f64],
    grad_out_scale: f64,
    grad_x: &mut [f64],
    grad_y: &mut [f64],
    grad_w: &mut [f64],
) {
    grad_x.fill(0.0);
    grad_y.fill(0.0);
    grad_w.fill(0.0);
    for instruction in &plan.instructions {
        let weight = weights[instruction.weight_index];
        match instruction.kind {
            NequixTensorProductRuntimeInstructionKind::ScalarTimesY {
                component_count,
                coefficient,
            } => {
                let x0 = x[instruction.input1_start];
                let grad_y_scale = coefficient * weight * x0;
                let mut grad_weight = 0.0;
                for component in 0..component_count {
                    let y_index = instruction.input2_start + component;
                    let out_index = instruction.out_start + component;
                    let bar_out = grad_out_scale * grad_out[out_index];
                    grad_x[instruction.input1_start] = (coefficient * weight * y[y_index])
                        .mul_add(bar_out, grad_x[instruction.input1_start]);
                    grad_y[y_index] = grad_y_scale.mul_add(bar_out, grad_y[y_index]);
                    grad_weight = (coefficient * x0 * y[y_index]).mul_add(bar_out, grad_weight);
                }
                grad_w[instruction.weight_index] += grad_weight;
                continue;
            }
            NequixTensorProductRuntimeInstructionKind::XTimesScalar {
                component_count,
                coefficient,
            } => {
                let y0 = y[instruction.input2_start];
                let grad_x_scale = coefficient * weight * y0;
                let mut grad_weight = 0.0;
                for component in 0..component_count {
                    let x_index = instruction.input1_start + component;
                    let out_index = instruction.out_start + component;
                    let bar_out = grad_out_scale * grad_out[out_index];
                    grad_x[x_index] = grad_x_scale.mul_add(bar_out, grad_x[x_index]);
                    grad_y[instruction.input2_start] = (coefficient * weight * x[x_index])
                        .mul_add(bar_out, grad_y[instruction.input2_start]);
                    grad_weight = (coefficient * x[x_index] * y0).mul_add(bar_out, grad_weight);
                }
                grad_w[instruction.weight_index] += grad_weight;
                continue;
            }
            NequixTensorProductRuntimeInstructionKind::DotToScalar {
                component_count,
                coefficient,
            } => {
                let bar_out = grad_out_scale * grad_out[instruction.out_start];
                let mut grad_weight = 0.0;
                for component in 0..component_count {
                    let x_index = instruction.input1_start + component;
                    let y_index = instruction.input2_start + component;
                    grad_x[x_index] =
                        (coefficient * weight * y[y_index]).mul_add(bar_out, grad_x[x_index]);
                    grad_y[y_index] =
                        (coefficient * weight * x[x_index]).mul_add(bar_out, grad_y[y_index]);
                    grad_weight =
                        (coefficient * x[x_index] * y[y_index]).mul_add(bar_out, grad_weight);
                }
                grad_w[instruction.weight_index] += grad_weight;
                continue;
            }
            NequixTensorProductRuntimeInstructionKind::Generic => {}
        }
        let cg_end = instruction.cg_offset + instruction.cg_len;
        let mut grad_weight = 0.0;
        for cg in &plan.cg_entries[instruction.cg_offset..cg_end] {
            let x_index = instruction.input1_start + cg.i_m;
            let y_index = instruction.input2_start + cg.j_m;
            let out_index = instruction.out_start + cg.k_m;
            let scale = if plan.unit_path_weights {
                cg.value
            } else {
                instruction.path_weight * cg.value
            };
            let bar_out = grad_out_scale * grad_out[out_index];
            grad_x[x_index] = (scale * weight * y[y_index]).mul_add(bar_out, grad_x[x_index]);
            grad_y[y_index] = (scale * weight * x[x_index]).mul_add(bar_out, grad_y[y_index]);
            grad_weight = (scale * x[x_index] * y[y_index]).mul_add(bar_out, grad_weight);
        }
        grad_w[instruction.weight_index] += grad_weight;
    }
}

pub fn scale_rows(values: &mut [f64], scale: f64) {
    values.iter_mut().for_each(|value| *value *= scale);
}

#[cfg(feature = "parallel")]
fn nequix_should_parallelize_edges(edge_count: usize) -> bool {
    edge_count >= NEQUIX_PARALLEL_EDGE_THRESHOLD && rayon::current_num_threads() > 1
}

#[cfg(feature = "parallel")]
fn nequix_parallel_edge_chunk_len(edge_count: usize) -> usize {
    (edge_count / (rayon::current_num_threads().max(1) * 2)).max(1)
}

#[cfg(feature = "parallel")]
fn nequix_should_parallelize_dense_rows(rows: usize) -> bool {
    rows >= NEQUIX_PARALLEL_DENSE_ROW_THRESHOLD && rayon::current_num_threads() > 1
}

#[cfg(feature = "parallel")]
fn nequix_parallel_row_chunk_len(rows: usize) -> usize {
    (rows / rayon::current_num_threads().max(1)).max(1)
}

#[must_use]
pub fn silu(value: f64) -> f64 {
    value / (1.0 + (-value).exp())
}

#[must_use]
fn silu_with_derivative(value: f64) -> (f64, f64) {
    let sigmoid = 1.0 / (1.0 + (-value).exp());
    (value * sigmoid, sigmoid * (1.0 + value * (1.0 - sigmoid)))
}

#[must_use]
pub fn silu_derivative(value: f64) -> f64 {
    let sigmoid = 1.0 / (1.0 + (-value).exp());
    sigmoid * (1.0 + value * (1.0 - sigmoid))
}

#[must_use]
pub fn nequix_normalized_silu(value: f64) -> f64 {
    E3NN_JAX_NORMALIZED_SILU_SCALE * silu(value)
}

#[must_use]
pub fn nequix_normalized_silu_derivative(value: f64) -> f64 {
    E3NN_JAX_NORMALIZED_SILU_SCALE * silu_derivative(value)
}

#[derive(Debug, Error)]
pub enum NequixError {
    #[error("I/O error for {path}: {source}")]
    Io { path: PathBuf, source: io::Error },

    #[error("invalid NequiX model: {0}")]
    InvalidModel(String),

    #[error("NequiX tensor {name} file size mismatch: expected {expected_bytes} bytes for shape {shape:?}, got {actual_bytes}")]
    TensorSizeMismatch {
        name: String,
        shape: Vec<usize>,
        expected_bytes: usize,
        actual_bytes: usize,
    },

    #[error("NequiX does not support stress/virial prediction")]
    RuntimeNotImplemented,

    #[error("nequix-v1 stress/virial requires full PBC")]
    StressVirialRequiresFullPbc,

    #[error("nequix-v1 stress/virial requires a periodic cell")]
    StressVirialRequiresCell,

    #[error(transparent)]
    ForceAccum(#[from] ForceAccumError),
}

#[derive(Clone, Debug)]
pub struct NequixTensor {
    pub tree_path: String,
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct NequixModel {
    selected_model: String,
    type_names: Vec<String>,
    cutoff: f64,
    l_max: usize,
    n_layers: usize,
    tensor_count: usize,
    total_numel: usize,
    artifact_sha256: String,
    artifact_size_bytes: u64,
    tensors: Vec<NequixTensor>,
    tensor_lookup: HashMap<String, usize>,
    weight_refs: Option<NequixWeightRefs>,
    tp_plans: Vec<TensorProductPlan>,
    tp_runtime_plans: Vec<NequixTensorProductRuntimePlan>,
    avg_n_neighbors: Vec<f64>,
    radial_polynomial_p: f64,
    scale: f64,
    shift: f64,
}

struct NequixGraphCache {
    atom_count: usize,
    edge_count: usize,
    senders: Vec<usize>,
    receivers: Vec<usize>,
    #[cfg(feature = "parallel")]
    receiver_edges: Vec<Vec<usize>>,
    edge_vectors: Vec<[f64; 3]>,
    distances: Vec<f64>,
    radial_basis: Vec<f64>,
    sh_values: Vec<f64>,
    sh_dim: usize,
}

struct NequixLayerCache {
    layer: usize,
    message_dim: usize,
    aggregated_dim: usize,
    pre_norm_dim: usize,
    output_dim: usize,
    messages: Vec<f64>,
    radial_message: Vec<f64>,
    radial_silu_derivatives: [Vec<f64>; 2],
    pre_norm: Vec<f64>,
    normed: Vec<f64>,
    features_out: Vec<f64>,
}

struct NequixForwardCache {
    graph: NequixGraphCache,
    layers: Vec<NequixLayerCache>,
    energy: f64,
}

struct NequixForceReplay {
    forces: Vec<[f64; 3]>,
    displacement_gradient: Mat3,
}

type NequixLayerVjp = (Vec<f64>, Vec<f64>, Vec<f64>);

fn nequix_full_pbc_cell_for_stress_virial(
    pbc: [bool; 3],
    cell: Option<&Cell>,
) -> Result<&Cell, NequixError> {
    if pbc != [true, true, true] {
        return Err(NequixError::StressVirialRequiresFullPbc);
    }
    cell.ok_or(NequixError::StressVirialRequiresCell)
}

fn nequix_stress_virial_from_displacement_gradient(
    displacement_gradient: Mat3,
    pbc: [bool; 3],
    cell: Option<&Cell>,
    compute_stress: bool,
    compute_virial: bool,
) -> Result<(Option<Mat3>, Option<Mat3>), NequixError> {
    let displacement_gradient = nequix_symmetric_part(displacement_gradient);
    let cell = nequix_full_pbc_cell_for_stress_virial(pbc, cell)?;
    let stress = if compute_stress {
        Some(stress_from_displacement_gradient(
            displacement_gradient,
            cell,
        )?)
    } else {
        None
    };
    let virial = if compute_virial {
        Some(virial_from_displacement_gradient(displacement_gradient))
    } else {
        None
    };
    Ok((stress, virial))
}

fn nequix_symmetric_part(matrix: Mat3) -> Mat3 {
    let mut symmetric = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            symmetric[row][col] = 0.5 * (matrix[row][col] + matrix[col][row]);
        }
    }
    symmetric
}

#[derive(Clone, Copy, Debug)]
struct NequixTensorProductRuntimeInstruction {
    input1_start: usize,
    input2_start: usize,
    out_start: usize,
    cg_offset: usize,
    cg_len: usize,
    weight_index: usize,
    path_weight: f64,
    kind: NequixTensorProductRuntimeInstructionKind,
}

#[derive(Clone, Copy, Debug)]
enum NequixTensorProductRuntimeInstructionKind {
    Generic,
    ScalarTimesY {
        component_count: usize,
        coefficient: f64,
    },
    XTimesScalar {
        component_count: usize,
        coefficient: f64,
    },
    DotToScalar {
        component_count: usize,
        coefficient: f64,
    },
}

#[derive(Clone, Debug)]
struct NequixTensorProductRuntimePlan {
    unit_path_weights: bool,
    instructions: Vec<NequixTensorProductRuntimeInstruction>,
    cg_entries: Vec<CgEntry>,
    input2_dim: usize,
    output_dim: usize,
    weight_numel: usize,
}

#[derive(Clone, Debug)]
struct NequixWeightRefs {
    atom_energies: usize,
    readout: usize,
    layers: Vec<NequixLayerWeightRefs>,
}

#[derive(Clone, Debug)]
struct NequixLayerWeightRefs {
    radial_mlp: [usize; 3],
    linear_1: Vec<usize>,
    linear_2: Vec<usize>,
    skip: Vec<usize>,
    layer_norm: Vec<usize>,
}

fn nequix_tensor_index(
    tensor_lookup: &HashMap<String, usize>,
    tree_path: &str,
) -> Result<usize, NequixError> {
    tensor_lookup
        .get(tree_path)
        .copied()
        .ok_or_else(|| invalid_model(format!("missing NeQuiX tensor {tree_path}")))
}

fn nequix_radial_weight_refs(
    tensor_lookup: &HashMap<String, usize>,
    layer: usize,
) -> Result<[usize; 3], NequixError> {
    Ok([
        nequix_tensor_index(
            tensor_lookup,
            &format!("layers.{layer}.radial_mlp.layers.0.weights"),
        )?,
        nequix_tensor_index(
            tensor_lookup,
            &format!("layers.{layer}.radial_mlp.layers.1.weights"),
        )?,
        nequix_tensor_index(
            tensor_lookup,
            &format!("layers.{layer}.radial_mlp.layers.2.weights"),
        )?,
    ])
}

fn nequix_norm_weight_refs(
    tensor_lookup: &HashMap<String, usize>,
    layer: usize,
    count: usize,
) -> Result<Vec<usize>, NequixError> {
    (0..count)
        .map(|index| {
            nequix_tensor_index(
                tensor_lookup,
                &format!("layers.{layer}.layer_norm.affine_weight.{index}"),
            )
        })
        .collect()
}

fn nequix_hidden_linear_weight_refs(
    tensor_lookup: &HashMap<String, usize>,
    layer: usize,
    block: &str,
) -> Result<Vec<usize>, NequixError> {
    Ok(vec![
        nequix_tensor_index(
            tensor_lookup,
            &format!("layers.{layer}.{block}._weights.w[0,0] 128x0e,128x0e"),
        )?,
        nequix_tensor_index(
            tensor_lookup,
            &format!("layers.{layer}.{block}._weights.w[1,1] 64x1o,64x1o"),
        )?,
        nequix_tensor_index(
            tensor_lookup,
            &format!("layers.{layer}.{block}._weights.w[2,2] 32x2e,32x2e"),
        )?,
        nequix_tensor_index(
            tensor_lookup,
            &format!("layers.{layer}.{block}._weights.w[3,3] 32x3o,32x3o"),
        )?,
    ])
}

fn build_nequix_weight_refs(
    tensor_lookup: &HashMap<String, usize>,
) -> Result<Option<NequixWeightRefs>, NequixError> {
    if tensor_lookup.is_empty() {
        return Ok(None);
    }

    let mut layers = Vec::with_capacity(EXPECTED_N_LAYERS);
    layers.push(NequixLayerWeightRefs {
        radial_mlp: nequix_radial_weight_refs(tensor_lookup, 0)?,
        linear_1: vec![nequix_tensor_index(
            tensor_lookup,
            "layers.0.linear_1._weights.w[0,0] 89x0e,89x0e",
        )?],
        linear_2: vec![
            nequix_tensor_index(
                tensor_lookup,
                "layers.0.linear_2._weights.w[0,0] 89x0e,256x0e",
            )?,
            nequix_tensor_index(
                tensor_lookup,
                "layers.0.linear_2._weights.w[1,1] 89x1o,64x1o",
            )?,
            nequix_tensor_index(
                tensor_lookup,
                "layers.0.linear_2._weights.w[2,2] 89x2e,32x2e",
            )?,
            nequix_tensor_index(
                tensor_lookup,
                "layers.0.linear_2._weights.w[3,3] 89x3o,32x3o",
            )?,
        ],
        skip: vec![nequix_tensor_index(
            tensor_lookup,
            "layers.0.skip._weights.w[0,0] 89x0e,256x0e",
        )?],
        layer_norm: nequix_norm_weight_refs(tensor_lookup, 0, 4)?,
    });

    for layer in 1..=2 {
        layers.push(NequixLayerWeightRefs {
            radial_mlp: nequix_radial_weight_refs(tensor_lookup, layer)?,
            linear_1: nequix_hidden_linear_weight_refs(tensor_lookup, layer, "linear_1")?,
            linear_2: vec![
                nequix_tensor_index(
                    tensor_lookup,
                    &format!("layers.{layer}.linear_2._weights.w[0,0] 256x0e,256x0e"),
                )?,
                nequix_tensor_index(
                    tensor_lookup,
                    &format!("layers.{layer}.linear_2._weights.w[1,1] 352x1o,64x1o"),
                )?,
                nequix_tensor_index(
                    tensor_lookup,
                    &format!("layers.{layer}.linear_2._weights.w[2,2] 384x2e,32x2e"),
                )?,
                nequix_tensor_index(
                    tensor_lookup,
                    &format!("layers.{layer}.linear_2._weights.w[3,3] 320x3o,32x3o"),
                )?,
            ],
            skip: vec![
                nequix_tensor_index(
                    tensor_lookup,
                    &format!("layers.{layer}.skip._weights.w[0,0] 128x0e,256x0e"),
                )?,
                nequix_tensor_index(
                    tensor_lookup,
                    &format!("layers.{layer}.skip._weights.w[1,1] 64x1o,64x1o"),
                )?,
                nequix_tensor_index(
                    tensor_lookup,
                    &format!("layers.{layer}.skip._weights.w[2,2] 32x2e,32x2e"),
                )?,
                nequix_tensor_index(
                    tensor_lookup,
                    &format!("layers.{layer}.skip._weights.w[3,3] 32x3o,32x3o"),
                )?,
            ],
            layer_norm: nequix_norm_weight_refs(tensor_lookup, layer, 4)?,
        });
    }

    layers.push(NequixLayerWeightRefs {
        radial_mlp: nequix_radial_weight_refs(tensor_lookup, 3)?,
        linear_1: nequix_hidden_linear_weight_refs(tensor_lookup, 3, "linear_1")?,
        linear_2: vec![nequix_tensor_index(
            tensor_lookup,
            "layers.3.linear_2._weights.w[0,0] 256x0e,128x0e",
        )?],
        skip: vec![nequix_tensor_index(
            tensor_lookup,
            "layers.3.skip._weights.w[0,0] 128x0e,128x0e",
        )?],
        layer_norm: nequix_norm_weight_refs(tensor_lookup, 3, 1)?,
    });

    Ok(Some(NequixWeightRefs {
        atom_energies: nequix_tensor_index(tensor_lookup, "atom_energies")?,
        readout: nequix_tensor_index(tensor_lookup, "readout._weights.w[0,0] 128x0e,1x0e")?,
        layers,
    }))
}

impl NequixModel {
    pub fn from_json_str(text: &str) -> Result<Self, NequixError> {
        let spec: StateSpec = serde_json::from_str(text)
            .map_err(|err| invalid_model(format!("invalid NequiX JSON: {err}")))?;
        Self::from_spec(spec, None)
    }

    pub fn from_json_str_with_root(text: &str, root: &Path) -> Result<Self, NequixError> {
        let spec: StateSpec = serde_json::from_str(text)
            .map_err(|err| invalid_model(format!("invalid NequiX JSON: {err}")))?;
        Self::from_spec(spec, Some(root))
    }

    pub fn predict(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
    ) -> Result<Prediction, NequixError> {
        self.predict_with_compute_dtype(structure, options, NequixComputeDtype::Float64)
    }

    /// Opt-in compute-dtype prediction. `Float64` is the default exact path;
    /// `Float32` runs the MACE-MH-1-style f32-promoted matmul boundaries while
    /// keeping the rest of the pipeline (and the conservative force / stress /
    /// virial VJP) in f64. Energy, forces, and the stress/virial displacement
    /// gradient all derive from the same f32-promoted forward cache and VJP
    /// chain, so they stay mutually consistent.
    pub fn predict_with_compute_dtype(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
        dtype: NequixComputeDtype,
    ) -> Result<Prediction, NequixError> {
        if options.compute_stress || options.compute_virial {
            nequix_full_pbc_cell_for_stress_virial(structure.pbc, structure.cell.as_ref())?;
        }
        if options.compute_forces || options.compute_stress || options.compute_virial {
            let cache = self.forward_cache(structure, dtype)?;
            let replay =
                self.vjp_forces_and_displacement_gradient(&cache, options.compute_forces, dtype)?;
            let energy = if options.compute_energy {
                Some(cache.energy)
            } else {
                None
            };
            let (stress, virial) = if options.compute_stress || options.compute_virial {
                nequix_stress_virial_from_displacement_gradient(
                    replay.displacement_gradient,
                    structure.pbc,
                    structure.cell.as_ref(),
                    options.compute_stress,
                    options.compute_virial,
                )?
            } else {
                (None, None)
            };
            return Ok(Prediction {
                energy,
                forces: options.compute_forces.then_some(replay.forces),
                stress,
                virial,
            });
        }
        let energy = if options.compute_energy {
            Some(match dtype {
                NequixComputeDtype::Float64 => self.forward_energy(structure)?,
                NequixComputeDtype::Float32 => self.forward_cache(structure, dtype)?.energy,
            })
        } else {
            None
        };
        Ok(Prediction {
            energy,
            forces: None,
            stress: None,
            virial: None,
        })
    }

    fn tensor_ref(&self, index: usize) -> Result<&NequixTensor, NequixError> {
        self.tensors
            .get(index)
            .ok_or_else(|| invalid_model(format!("missing NequiX tensor index {index}")))
    }

    fn weight_refs(&self) -> Result<&NequixWeightRefs, NequixError> {
        self.weight_refs
            .as_ref()
            .ok_or_else(|| invalid_model("NeQuiX runtime weights are not loaded"))
    }

    fn layer_weight_refs(&self, layer: usize) -> Result<&NequixLayerWeightRefs, NequixError> {
        self.weight_refs()?.layers.get(layer).ok_or_else(|| {
            invalid_model(format!(
                "missing NeQuiX layer weight refs for layer {layer}"
            ))
        })
    }

    fn radial_weight_refs(&self, layer: usize) -> Result<[&NequixTensor; 3], NequixError> {
        let refs = &self.layer_weight_refs(layer)?.radial_mlp;
        Ok([
            self.tensor_ref(refs[0])?,
            self.tensor_ref(refs[1])?,
            self.tensor_ref(refs[2])?,
        ])
    }

    fn forward_cache(
        &self,
        structure: &Structure,
        dtype: NequixComputeDtype,
    ) -> Result<NequixForwardCache, NequixError> {
        if self.tensors.len() != EXPECTED_TENSOR_COUNT {
            return Err(invalid_model(
                "NequiX prediction requires loaded f64 tensor payloads",
            ));
        }
        if self.tp_plans.len() != EXPECTED_N_LAYERS {
            return Err(invalid_model(
                "NequiX prediction requires loaded tensor-product plans",
            ));
        }
        if self.tp_runtime_plans.len() != EXPECTED_N_LAYERS {
            return Err(invalid_model(
                "NequiX prediction requires loaded runtime tensor-product plans",
            ));
        }
        let atom_count = structure.len();
        let species_indices = self.species_indices(structure)?;
        let mut features = vec![0.0; atom_count * 89];
        for (atom_index, species_index) in species_indices.iter().enumerate() {
            features[atom_index * 89 + species_index] = 1.0;
        }
        let graph = self.graph_cache(structure)?;
        let mut layers = Vec::with_capacity(EXPECTED_N_LAYERS);

        layers.push(self.forward_layer0_cache(&features, &graph, dtype)?);
        for layer in 1..3 {
            let current = layers
                .last()
                .ok_or_else(|| invalid_model("NequiX forward cache lost layer state"))?
                .features_out
                .as_slice();
            let layer_cache = self.forward_hidden_layer_cache(layer, current, &graph, dtype)?;
            layers.push(layer_cache);
        }
        let current = layers
            .last()
            .ok_or_else(|| invalid_model("NequiX forward cache lost layer state"))?
            .features_out
            .as_slice();
        layers.push(self.forward_final_layer_cache(current, &graph, dtype)?);

        let weights = self.weight_refs()?;
        let readout_weight = self.tensor_ref(weights.readout)?;
        let current = layers
            .last()
            .ok_or_else(|| invalid_model("NequiX forward cache lost layer state"))?
            .features_out
            .as_slice();
        let node_energy_raw = nequix_dense_forward_scaled_dtype(
            current,
            atom_count,
            128,
            &readout_weight.values,
            1,
            1.0 / 128.0_f64.sqrt(),
            dtype,
        )?;
        let atom_energies = self.tensor_ref(weights.atom_energies)?;
        let mut energy = 0.0;
        for (atom_index, species_index) in species_indices.iter().enumerate() {
            energy += node_energy_raw[atom_index] * self.scale
                + self.shift
                + atom_energies.values[*species_index];
        }
        Ok(NequixForwardCache {
            graph,
            layers,
            energy,
        })
    }

    fn graph_cache(&self, structure: &Structure) -> Result<NequixGraphCache, NequixError> {
        let neighbor_list = cell_list_multi_image_neighbor_list(structure, self.cutoff)
            .map_err(|err| invalid_model(format!("NequiX neighbor list failed: {err}")))?;
        let senders = neighbor_list
            .edges
            .iter()
            .map(|edge| edge.dst)
            .collect::<Vec<_>>();
        let receivers = neighbor_list
            .edges
            .iter()
            .map(|edge| edge.src)
            .collect::<Vec<_>>();
        let edge_vectors = neighbor_list
            .edges
            .iter()
            .map(|edge| edge.vector)
            .collect::<Vec<_>>();
        let distances = neighbor_list
            .edges
            .iter()
            .map(|edge| edge.distance)
            .collect::<Vec<_>>();
        let edge_count = neighbor_list.edges.len();
        #[cfg(feature = "parallel")]
        let receiver_edges = {
            let mut receiver_edges = vec![Vec::new(); structure.len()];
            for (edge_index, receiver) in receivers.iter().enumerate() {
                receiver_edges[*receiver].push(edge_index);
            }
            receiver_edges
        };
        let sh_plan = SphericalHarmonicsPlan::component(3, true)
            .map_err(|err| invalid_model(format!("NequiX SH plan failed: {err}")))?;
        let sh_dim = sh_plan.output_dim();
        let mut sh_values = vec![0.0; edge_count * sh_dim];
        let mut radial_basis = vec![0.0; edge_count * 8];
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && edge_count >= NEQUIX_PARALLEL_EDGE_THRESHOLD {
            sh_values
                .par_chunks_mut(sh_dim)
                .zip(radial_basis.par_chunks_mut(8))
                .enumerate()
                .try_for_each(|(edge_index, (sh_row, radial_row))| {
                    sh_plan
                        .forward(edge_vectors[edge_index], sh_row)
                        .map_err(|err| invalid_model(format!("NequiX SH forward failed: {err}")))?;
                    nequix_radial_basis_into(
                        distances[edge_index],
                        self.cutoff,
                        self.radial_polynomial_p,
                        radial_row,
                    );
                    Ok::<(), NequixError>(())
                })?;
        }
        #[cfg(not(feature = "parallel"))]
        for edge_index in 0..edge_count {
            sh_plan
                .forward(
                    edge_vectors[edge_index],
                    &mut sh_values[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                )
                .map_err(|err| invalid_model(format!("NequiX SH forward failed: {err}")))?;
            nequix_radial_basis_into(
                distances[edge_index],
                self.cutoff,
                self.radial_polynomial_p,
                &mut radial_basis[edge_index * 8..(edge_index + 1) * 8],
            );
        }
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() <= 1 || edge_count < NEQUIX_PARALLEL_EDGE_THRESHOLD {
            for edge_index in 0..edge_count {
                sh_plan
                    .forward(
                        edge_vectors[edge_index],
                        &mut sh_values[edge_index * sh_dim..(edge_index + 1) * sh_dim],
                    )
                    .map_err(|err| invalid_model(format!("NequiX SH forward failed: {err}")))?;
                nequix_radial_basis_into(
                    distances[edge_index],
                    self.cutoff,
                    self.radial_polynomial_p,
                    &mut radial_basis[edge_index * 8..(edge_index + 1) * 8],
                );
            }
        }
        Ok(NequixGraphCache {
            atom_count: structure.len(),
            edge_count,
            senders,
            receivers,
            #[cfg(feature = "parallel")]
            receiver_edges,
            edge_vectors,
            distances,
            radial_basis,
            sh_values,
            sh_dim,
        })
    }

    fn forward_layer0_cache(
        &self,
        features: &[f64],
        graph: &NequixGraphCache,
        dtype: NequixComputeDtype,
    ) -> Result<NequixLayerCache, NequixError> {
        let refs = self.layer_weight_refs(0)?;
        let linear1_weight = self.tensor_ref(refs.linear_1[0])?;
        let messages = nequix_dense_forward_scaled_dtype(
            features,
            graph.atom_count,
            89,
            &linear1_weight.values,
            89,
            1.0 / 89.0_f64.sqrt(),
            dtype,
        )?;
        let (aggregated, radial_cache) = self.tensor_product_scatter_with_radial(
            0,
            &messages,
            graph.atom_count,
            89,
            graph,
            dtype,
        )?;
        let NequixRadialMlpCache {
            output: radial_message,
            silu_derivatives: radial_silu_derivatives,
        } = radial_cache;
        let linear2_scale = 1.0 / 89.0_f64.sqrt();
        let linear2 = nequix_irrep_linear_forward_dtype(
            &aggregated,
            graph.atom_count,
            1424,
            832,
            &self.layer0_linear2_blocks(linear2_scale)?,
            dtype,
        )?;
        let skip = nequix_irrep_linear_forward_dtype(
            features,
            graph.atom_count,
            89,
            832,
            &self.layer0_skip_blocks()?,
            dtype,
        )?;
        let pre_norm = add_same_shape(&linear2, &skip)?;
        let normed = self.norm_layer(0, &pre_norm, graph.atom_count, 832)?;
        let features_out = self.gate_hidden(&normed, graph.atom_count, 832, 704)?;
        Ok(NequixLayerCache {
            layer: 0,
            message_dim: 89,
            aggregated_dim: 1424,
            pre_norm_dim: 832,
            output_dim: 704,
            messages,
            radial_message,
            radial_silu_derivatives,
            pre_norm,
            normed,
            features_out,
        })
    }

    fn forward_hidden_layer_cache(
        &self,
        layer: usize,
        features: &[f64],
        graph: &NequixGraphCache,
        dtype: NequixComputeDtype,
    ) -> Result<NequixLayerCache, NequixError> {
        let messages = self.hidden_linear(
            layer,
            "linear_1",
            features,
            graph.atom_count,
            704,
            704,
            dtype,
        )?;
        let (aggregated, radial_cache) = self.tensor_product_scatter_with_radial(
            layer,
            &messages,
            graph.atom_count,
            704,
            graph,
            dtype,
        )?;
        let NequixRadialMlpCache {
            output: radial_message,
            silu_derivatives: radial_silu_derivatives,
        } = radial_cache;
        let linear2 = self.hidden_linear2(layer, &aggregated, graph.atom_count, dtype)?;
        let skip = self.hidden_skip(layer, features, graph.atom_count, 832, dtype)?;
        let pre_norm = add_same_shape(&linear2, &skip)?;
        let normed = self.norm_layer(layer, &pre_norm, graph.atom_count, 832)?;
        let features_out = self.gate_hidden(&normed, graph.atom_count, 832, 704)?;
        Ok(NequixLayerCache {
            layer,
            message_dim: 704,
            aggregated_dim: 5472,
            pre_norm_dim: 832,
            output_dim: 704,
            messages,
            radial_message,
            radial_silu_derivatives,
            pre_norm,
            normed,
            features_out,
        })
    }

    fn forward_final_layer_cache(
        &self,
        features: &[f64],
        graph: &NequixGraphCache,
        dtype: NequixComputeDtype,
    ) -> Result<NequixLayerCache, NequixError> {
        let messages =
            self.hidden_linear(3, "linear_1", features, graph.atom_count, 704, 704, dtype)?;
        let (aggregated, radial_cache) = self.tensor_product_scatter_with_radial(
            3,
            &messages,
            graph.atom_count,
            704,
            graph,
            dtype,
        )?;
        let NequixRadialMlpCache {
            output: radial_message,
            silu_derivatives: radial_silu_derivatives,
        } = radial_cache;
        let refs = self.layer_weight_refs(3)?;
        let linear2_weight = self.tensor_ref(refs.linear_2[0])?;
        let linear2 = nequix_irrep_linear_forward_dtype(
            &aggregated,
            graph.atom_count,
            256,
            128,
            &[NequixIrrepLinearBlock {
                input_offset: 0,
                input_mul: 256,
                output_offset: 0,
                output_mul: 128,
                ell: 0,
                weight: &linear2_weight.values,
                alpha: 1.0 / 256.0_f64.sqrt(),
            }],
            dtype,
        )?;
        let skip_weight = self.tensor_ref(refs.skip[0])?;
        let skip = nequix_irrep_linear_forward_dtype(
            features,
            graph.atom_count,
            704,
            128,
            &[NequixIrrepLinearBlock {
                input_offset: 0,
                input_mul: 128,
                output_offset: 0,
                output_mul: 128,
                ell: 0,
                weight: &skip_weight.values,
                alpha: 1.0 / 128.0_f64.sqrt(),
            }],
            dtype,
        )?;
        let pre_norm = add_same_shape(&linear2, &skip)?;
        let norm_weight = self.tensor_ref(refs.layer_norm[0])?;
        let normed = nequix_rms_layer_norm(
            &pre_norm,
            graph.atom_count,
            128,
            &[NequixNormBlock {
                offset: 0,
                mul: 128,
                ell: 0,
                affine_weight: &norm_weight.values,
            }],
            1.0e-12,
        )?;
        let features_out = nequix_gate_forward(&normed, graph.atom_count, 128, 128, 128, &[])?;
        Ok(NequixLayerCache {
            layer: 3,
            message_dim: 704,
            aggregated_dim: 256,
            pre_norm_dim: 128,
            output_dim: 128,
            messages,
            radial_message,
            radial_silu_derivatives,
            pre_norm,
            normed,
            features_out,
        })
    }

    fn forward_energy(&self, structure: &Structure) -> Result<f64, NequixError> {
        if self.tensors.len() != EXPECTED_TENSOR_COUNT {
            return Err(invalid_model(
                "NequiX energy prediction requires loaded f64 tensor payloads",
            ));
        }
        if self.tp_plans.len() != EXPECTED_N_LAYERS {
            return Err(invalid_model(
                "NequiX energy prediction requires loaded tensor-product plans",
            ));
        }
        let atom_count = structure.len();
        let species_indices = self.species_indices(structure)?;
        let mut features = vec![0.0; atom_count * 89];
        for (atom_index, species_index) in species_indices.iter().enumerate() {
            features[atom_index * 89 + species_index] = 1.0;
        }

        let graph = self.graph_cache(structure)?;

        let features = self.forward_layer0(&features, &graph)?;
        let features = self.forward_hidden_layer(1, &features, &graph)?;
        let features = self.forward_hidden_layer(2, &features, &graph)?;
        let features = self.forward_final_layer(&features, &graph)?;
        let readout_weight = self.required_tensor("readout._weights.w[0,0] 128x0e,1x0e")?;
        let node_energy_raw = nequix_dense_forward_scaled(
            &features,
            atom_count,
            128,
            &readout_weight.values,
            1,
            1.0 / 128.0_f64.sqrt(),
        )?;
        let atom_energies = self.required_tensor("atom_energies")?;
        let mut energy = 0.0;
        for (atom_index, species_index) in species_indices.iter().enumerate() {
            energy += node_energy_raw[atom_index] * self.scale
                + self.shift
                + atom_energies.values[*species_index];
        }
        Ok(energy)
    }

    fn forward_layer0(
        &self,
        features: &[f64],
        graph: &NequixGraphCache,
    ) -> Result<Vec<f64>, NequixError> {
        let linear1_weight =
            self.required_tensor("layers.0.linear_1._weights.w[0,0] 89x0e,89x0e")?;
        let messages = nequix_dense_forward_scaled(
            features,
            graph.atom_count,
            89,
            &linear1_weight.values,
            89,
            1.0 / 89.0_f64.sqrt(),
        )?;
        let (aggregated, _) = self.tensor_product_scatter_with_radial(
            0,
            &messages,
            graph.atom_count,
            89,
            graph,
            NequixComputeDtype::Float64,
        )?;
        let linear2_scale = 1.0 / 89.0_f64.sqrt();
        let linear2 = nequix_irrep_linear_forward(
            &aggregated,
            graph.atom_count,
            1424,
            832,
            &[
                NequixIrrepLinearBlock {
                    input_offset: 0,
                    input_mul: 89,
                    output_offset: 0,
                    output_mul: 256,
                    ell: 0,
                    weight: &self
                        .required_tensor("layers.0.linear_2._weights.w[0,0] 89x0e,256x0e")?
                        .values,
                    alpha: linear2_scale,
                },
                NequixIrrepLinearBlock {
                    input_offset: 89,
                    input_mul: 89,
                    output_offset: 256,
                    output_mul: 64,
                    ell: 1,
                    weight: &self
                        .required_tensor("layers.0.linear_2._weights.w[1,1] 89x1o,64x1o")?
                        .values,
                    alpha: linear2_scale,
                },
                NequixIrrepLinearBlock {
                    input_offset: 356,
                    input_mul: 89,
                    output_offset: 448,
                    output_mul: 32,
                    ell: 2,
                    weight: &self
                        .required_tensor("layers.0.linear_2._weights.w[2,2] 89x2e,32x2e")?
                        .values,
                    alpha: linear2_scale,
                },
                NequixIrrepLinearBlock {
                    input_offset: 801,
                    input_mul: 89,
                    output_offset: 608,
                    output_mul: 32,
                    ell: 3,
                    weight: &self
                        .required_tensor("layers.0.linear_2._weights.w[3,3] 89x3o,32x3o")?
                        .values,
                    alpha: linear2_scale,
                },
            ],
        )?;
        let skip = nequix_irrep_linear_forward(
            features,
            graph.atom_count,
            89,
            832,
            &[NequixIrrepLinearBlock {
                input_offset: 0,
                input_mul: 89,
                output_offset: 0,
                output_mul: 256,
                ell: 0,
                weight: &self
                    .required_tensor("layers.0.skip._weights.w[0,0] 89x0e,256x0e")?
                    .values,
                alpha: 1.0 / 89.0_f64.sqrt(),
            }],
        )?;
        self.norm_and_gate(
            0,
            &add_same_shape(&linear2, &skip)?,
            graph.atom_count,
            832,
            704,
        )
    }

    fn forward_hidden_layer(
        &self,
        layer: usize,
        features: &[f64],
        graph: &NequixGraphCache,
    ) -> Result<Vec<f64>, NequixError> {
        let messages = self.hidden_linear(
            layer,
            "linear_1",
            features,
            graph.atom_count,
            704,
            704,
            NequixComputeDtype::Float64,
        )?;
        let (aggregated, _) = self.tensor_product_scatter_with_radial(
            layer,
            &messages,
            graph.atom_count,
            704,
            graph,
            NequixComputeDtype::Float64,
        )?;
        let linear2 = self.hidden_linear2(
            layer,
            &aggregated,
            graph.atom_count,
            NequixComputeDtype::Float64,
        )?;
        let skip = self.hidden_skip(
            layer,
            features,
            graph.atom_count,
            832,
            NequixComputeDtype::Float64,
        )?;
        self.norm_and_gate(
            layer,
            &add_same_shape(&linear2, &skip)?,
            graph.atom_count,
            832,
            704,
        )
    }

    fn forward_final_layer(
        &self,
        features: &[f64],
        graph: &NequixGraphCache,
    ) -> Result<Vec<f64>, NequixError> {
        let messages = self.hidden_linear(
            3,
            "linear_1",
            features,
            graph.atom_count,
            704,
            704,
            NequixComputeDtype::Float64,
        )?;
        let (aggregated, _) = self.tensor_product_scatter_with_radial(
            3,
            &messages,
            graph.atom_count,
            704,
            graph,
            NequixComputeDtype::Float64,
        )?;
        let linear2_weight =
            self.required_tensor("layers.3.linear_2._weights.w[0,0] 256x0e,128x0e")?;
        let linear2 = nequix_irrep_linear_forward(
            &aggregated,
            graph.atom_count,
            256,
            128,
            &[NequixIrrepLinearBlock {
                input_offset: 0,
                input_mul: 256,
                output_offset: 0,
                output_mul: 128,
                ell: 0,
                weight: &linear2_weight.values,
                alpha: 1.0 / 256.0_f64.sqrt(),
            }],
        )?;
        let skip_weight = self.required_tensor("layers.3.skip._weights.w[0,0] 128x0e,128x0e")?;
        let skip = nequix_irrep_linear_forward(
            features,
            graph.atom_count,
            704,
            128,
            &[NequixIrrepLinearBlock {
                input_offset: 0,
                input_mul: 128,
                output_offset: 0,
                output_mul: 128,
                ell: 0,
                weight: &skip_weight.values,
                alpha: 1.0 / 128.0_f64.sqrt(),
            }],
        )?;
        let norm_weight = self.required_tensor("layers.3.layer_norm.affine_weight.0")?;
        let normed = nequix_rms_layer_norm(
            &add_same_shape(&linear2, &skip)?,
            graph.atom_count,
            128,
            &[NequixNormBlock {
                offset: 0,
                mul: 128,
                ell: 0,
                affine_weight: &norm_weight.values,
            }],
            1.0e-12,
        )?;
        nequix_gate_forward(&normed, graph.atom_count, 128, 128, 128, &[])
    }

    fn tensor_product_scatter_with_radial(
        &self,
        layer: usize,
        messages: &[f64],
        atom_count: usize,
        message_dim: usize,
        graph: &NequixGraphCache,
        dtype: NequixComputeDtype,
    ) -> Result<(Vec<f64>, NequixRadialMlpCache), NequixError> {
        let senders = &graph.senders;
        let receivers = &graph.receivers;
        let radial_basis = &graph.radial_basis;
        let sh_values = &graph.sh_values;
        let edge_count = senders.len();
        if receivers.len() != edge_count {
            return Err(invalid_model(format!(
                "NequiX layer {layer} receiver count {} does not match sender count {edge_count}",
                receivers.len()
            )));
        }
        let radial_weights = self.radial_weight_refs(layer)?;
        let radial_cache =
            nequix_radial_mlp_forward_cache(radial_basis, edge_count, &radial_weights, dtype)?;
        let radial_message = &radial_cache.output;
        let plan = self
            .tp_runtime_plans
            .get(layer)
            .ok_or_else(|| invalid_model(format!("missing NequiX TP plan for layer {layer}")))?;
        let sh_dim = plan.input2_dim;
        if sh_values.len() != edge_count * sh_dim {
            return Err(invalid_model(format!(
                "NequiX SH value length must be edge_count * sh_dim = {}, got {}",
                edge_count * sh_dim,
                sh_values.len()
            )));
        }
        let output_dim = plan.output_dim;
        let avg_n_neighbors = *self.avg_n_neighbors.get(layer).ok_or_else(|| {
            invalid_model(format!("missing NeQuiX avg_n_neighbors for layer {layer}"))
        })?;
        if !avg_n_neighbors.is_finite() || avg_n_neighbors <= 0.0 {
            return Err(invalid_model(format!(
                "NeQuiX avg_n_neighbors for layer {layer} must be finite and positive, got {avg_n_neighbors}"
            )));
        }
        let scatter_scale = 1.0 / avg_n_neighbors.sqrt();
        let mut aggregated = vec![0.0; atom_count * output_dim];
        #[cfg(feature = "parallel")]
        if nequix_should_parallelize_edges(edge_count) {
            if graph.receiver_edges.len() != atom_count {
                return Err(invalid_model(format!(
                    "NequiX receiver bucket count {} does not match atom count {atom_count}",
                    graph.receiver_edges.len()
                )));
            }
            aggregated
                .par_chunks_mut(output_dim)
                .enumerate()
                .try_for_each(|(receiver, out_row)| {
                    for &edge_index in &graph.receiver_edges[receiver] {
                        let sender = senders[edge_index];
                        if sender >= atom_count {
                            return Err(invalid_model(format!(
                                "NequiX layer {layer} sender index {sender} is out of bounds for atom count {atom_count}"
                            )));
                        }
                        let message_start = sender * message_dim;
                        let sh_start = edge_index * sh_dim;
                        let weight_start = edge_index * plan.weight_numel;
                        nequix_tensor_product_accumulate_fast(
                            plan,
                            &messages[message_start..message_start + message_dim],
                            &sh_values[sh_start..sh_start + sh_dim],
                            &radial_message[weight_start..weight_start + plan.weight_numel],
                            scatter_scale,
                            out_row,
                        );
                    }
                    Ok::<(), NequixError>(())
                })?;
            return Ok((aggregated, radial_cache));
        }
        for (edge_index, receiver) in receivers.iter().enumerate().take(edge_count) {
            if *receiver >= atom_count {
                return Err(invalid_model(format!(
                    "NequiX layer {layer} receiver index {receiver} is out of bounds for atom count {atom_count}"
                )));
            }
            let sender = senders[edge_index];
            if sender >= atom_count {
                return Err(invalid_model(format!(
                    "NequiX layer {layer} sender index {sender} is out of bounds for atom count {atom_count}"
                )));
            }
            let message_start = sender * message_dim;
            let sh_start = edge_index * sh_dim;
            let weight_start = edge_index * plan.weight_numel;
            let out_row = &mut aggregated[*receiver * output_dim..(*receiver + 1) * output_dim];
            nequix_tensor_product_accumulate_fast(
                plan,
                &messages[message_start..message_start + message_dim],
                &sh_values[sh_start..sh_start + sh_dim],
                &radial_message[weight_start..weight_start + plan.weight_numel],
                scatter_scale,
                out_row,
            );
        }
        Ok((aggregated, radial_cache))
    }

    fn layer0_linear2_blocks(
        &self,
        alpha: f64,
    ) -> Result<Vec<NequixIrrepLinearBlock<'_>>, NequixError> {
        let refs = &self.layer_weight_refs(0)?.linear_2;
        Ok(vec![
            NequixIrrepLinearBlock {
                input_offset: 0,
                input_mul: 89,
                output_offset: 0,
                output_mul: 256,
                ell: 0,
                weight: &self.tensor_ref(refs[0])?.values,
                alpha,
            },
            NequixIrrepLinearBlock {
                input_offset: 89,
                input_mul: 89,
                output_offset: 256,
                output_mul: 64,
                ell: 1,
                weight: &self.tensor_ref(refs[1])?.values,
                alpha,
            },
            NequixIrrepLinearBlock {
                input_offset: 356,
                input_mul: 89,
                output_offset: 448,
                output_mul: 32,
                ell: 2,
                weight: &self.tensor_ref(refs[2])?.values,
                alpha,
            },
            NequixIrrepLinearBlock {
                input_offset: 801,
                input_mul: 89,
                output_offset: 608,
                output_mul: 32,
                ell: 3,
                weight: &self.tensor_ref(refs[3])?.values,
                alpha,
            },
        ])
    }

    fn layer0_skip_blocks(&self) -> Result<Vec<NequixIrrepLinearBlock<'_>>, NequixError> {
        let refs = &self.layer_weight_refs(0)?.skip;
        Ok(vec![NequixIrrepLinearBlock {
            input_offset: 0,
            input_mul: 89,
            output_offset: 0,
            output_mul: 256,
            ell: 0,
            weight: &self.tensor_ref(refs[0])?.values,
            alpha: 1.0 / 89.0_f64.sqrt(),
        }])
    }

    #[allow(clippy::too_many_arguments)]
    fn hidden_linear(
        &self,
        layer: usize,
        block: &str,
        input: &[f64],
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        dtype: NequixComputeDtype,
    ) -> Result<Vec<f64>, NequixError> {
        let refs = match block {
            "linear_1" => &self.layer_weight_refs(layer)?.linear_1,
            _ => {
                return Err(invalid_model(format!(
                    "unsupported NeQuiX hidden linear block {block}"
                )));
            }
        };
        let weight_0 = self.tensor_ref(refs[0])?;
        let weight_1 = self.tensor_ref(refs[1])?;
        let weight_2 = self.tensor_ref(refs[2])?;
        let weight_3 = self.tensor_ref(refs[3])?;
        nequix_irrep_linear_forward_dtype(
            input,
            rows,
            input_dim,
            output_dim,
            &[
                NequixIrrepLinearBlock {
                    input_offset: 0,
                    input_mul: 128,
                    output_offset: 0,
                    output_mul: 128,
                    ell: 0,
                    weight: &weight_0.values,
                    alpha: 1.0 / 128.0_f64.sqrt(),
                },
                NequixIrrepLinearBlock {
                    input_offset: 128,
                    input_mul: 64,
                    output_offset: 128,
                    output_mul: 64,
                    ell: 1,
                    weight: &weight_1.values,
                    alpha: 1.0 / 64.0_f64.sqrt(),
                },
                NequixIrrepLinearBlock {
                    input_offset: 320,
                    input_mul: 32,
                    output_offset: 320,
                    output_mul: 32,
                    ell: 2,
                    weight: &weight_2.values,
                    alpha: 1.0 / 32.0_f64.sqrt(),
                },
                NequixIrrepLinearBlock {
                    input_offset: 480,
                    input_mul: 32,
                    output_offset: 480,
                    output_mul: 32,
                    ell: 3,
                    weight: &weight_3.values,
                    alpha: 1.0 / 32.0_f64.sqrt(),
                },
            ],
            dtype,
        )
    }

    fn hidden_linear2(
        &self,
        layer: usize,
        input: &[f64],
        rows: usize,
        dtype: NequixComputeDtype,
    ) -> Result<Vec<f64>, NequixError> {
        let refs = &self.layer_weight_refs(layer)?.linear_2;
        let weight_0 = self.tensor_ref(refs[0])?;
        let weight_1 = self.tensor_ref(refs[1])?;
        let weight_2 = self.tensor_ref(refs[2])?;
        let weight_3 = self.tensor_ref(refs[3])?;
        nequix_irrep_linear_forward_dtype(
            input,
            rows,
            5472,
            832,
            &[
                NequixIrrepLinearBlock {
                    input_offset: 0,
                    input_mul: 256,
                    output_offset: 0,
                    output_mul: 256,
                    ell: 0,
                    weight: &weight_0.values,
                    alpha: 1.0 / 256.0_f64.sqrt(),
                },
                NequixIrrepLinearBlock {
                    input_offset: 256,
                    input_mul: 352,
                    output_offset: 256,
                    output_mul: 64,
                    ell: 1,
                    weight: &weight_1.values,
                    alpha: 1.0 / 352.0_f64.sqrt(),
                },
                NequixIrrepLinearBlock {
                    input_offset: 1312,
                    input_mul: 384,
                    output_offset: 448,
                    output_mul: 32,
                    ell: 2,
                    weight: &weight_2.values,
                    alpha: 1.0 / 384.0_f64.sqrt(),
                },
                NequixIrrepLinearBlock {
                    input_offset: 3232,
                    input_mul: 320,
                    output_offset: 608,
                    output_mul: 32,
                    ell: 3,
                    weight: &weight_3.values,
                    alpha: 1.0 / 320.0_f64.sqrt(),
                },
            ],
            dtype,
        )
    }

    fn hidden_skip(
        &self,
        layer: usize,
        input: &[f64],
        rows: usize,
        output_dim: usize,
        dtype: NequixComputeDtype,
    ) -> Result<Vec<f64>, NequixError> {
        let refs = &self.layer_weight_refs(layer)?.skip;
        let weight_0 = self.tensor_ref(refs[0])?;
        let weight_1 = self.tensor_ref(refs[1])?;
        let weight_2 = self.tensor_ref(refs[2])?;
        let weight_3 = self.tensor_ref(refs[3])?;
        nequix_irrep_linear_forward_dtype(
            input,
            rows,
            704,
            output_dim,
            &[
                NequixIrrepLinearBlock {
                    input_offset: 0,
                    input_mul: 128,
                    output_offset: 0,
                    output_mul: 256,
                    ell: 0,
                    weight: &weight_0.values,
                    alpha: 1.0 / 128.0_f64.sqrt(),
                },
                NequixIrrepLinearBlock {
                    input_offset: 128,
                    input_mul: 64,
                    output_offset: 256,
                    output_mul: 64,
                    ell: 1,
                    weight: &weight_1.values,
                    alpha: 1.0 / 64.0_f64.sqrt(),
                },
                NequixIrrepLinearBlock {
                    input_offset: 320,
                    input_mul: 32,
                    output_offset: 448,
                    output_mul: 32,
                    ell: 2,
                    weight: &weight_2.values,
                    alpha: 1.0 / 32.0_f64.sqrt(),
                },
                NequixIrrepLinearBlock {
                    input_offset: 480,
                    input_mul: 32,
                    output_offset: 608,
                    output_mul: 32,
                    ell: 3,
                    weight: &weight_3.values,
                    alpha: 1.0 / 32.0_f64.sqrt(),
                },
            ],
            dtype,
        )
    }

    fn norm_and_gate(
        &self,
        layer: usize,
        pre_norm: &[f64],
        rows: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Vec<f64>, NequixError> {
        let path_0 = format!("layers.{layer}.layer_norm.affine_weight.0");
        let path_1 = format!("layers.{layer}.layer_norm.affine_weight.1");
        let path_2 = format!("layers.{layer}.layer_norm.affine_weight.2");
        let path_3 = format!("layers.{layer}.layer_norm.affine_weight.3");
        let weight_0 = self.required_tensor(&path_0)?;
        let weight_1 = self.required_tensor(&path_1)?;
        let weight_2 = self.required_tensor(&path_2)?;
        let weight_3 = self.required_tensor(&path_3)?;
        let normed = nequix_rms_layer_norm(
            pre_norm,
            rows,
            input_dim,
            &[
                NequixNormBlock {
                    offset: 0,
                    mul: 256,
                    ell: 0,
                    affine_weight: &weight_0.values,
                },
                NequixNormBlock {
                    offset: 256,
                    mul: 64,
                    ell: 1,
                    affine_weight: &weight_1.values,
                },
                NequixNormBlock {
                    offset: 448,
                    mul: 32,
                    ell: 2,
                    affine_weight: &weight_2.values,
                },
                NequixNormBlock {
                    offset: 608,
                    mul: 32,
                    ell: 3,
                    affine_weight: &weight_3.values,
                },
            ],
            1.0e-12,
        )?;
        nequix_gate_forward(
            &normed,
            rows,
            input_dim,
            128,
            output_dim,
            &[
                NequixGateBlock {
                    input_offset: 256,
                    output_offset: 128,
                    gate_offset: 128,
                    mul: 64,
                    ell: 1,
                },
                NequixGateBlock {
                    input_offset: 448,
                    output_offset: 320,
                    gate_offset: 192,
                    mul: 32,
                    ell: 2,
                },
                NequixGateBlock {
                    input_offset: 608,
                    output_offset: 480,
                    gate_offset: 224,
                    mul: 32,
                    ell: 3,
                },
            ],
        )
    }

    fn norm_layer(
        &self,
        layer: usize,
        pre_norm: &[f64],
        rows: usize,
        input_dim: usize,
    ) -> Result<Vec<f64>, NequixError> {
        let refs = &self.layer_weight_refs(layer)?.layer_norm;
        let weight_0 = self.tensor_ref(refs[0])?;
        let weight_1 = self.tensor_ref(refs[1])?;
        let weight_2 = self.tensor_ref(refs[2])?;
        let weight_3 = self.tensor_ref(refs[3])?;
        nequix_rms_layer_norm(
            pre_norm,
            rows,
            input_dim,
            &[
                NequixNormBlock {
                    offset: 0,
                    mul: 256,
                    ell: 0,
                    affine_weight: &weight_0.values,
                },
                NequixNormBlock {
                    offset: 256,
                    mul: 64,
                    ell: 1,
                    affine_weight: &weight_1.values,
                },
                NequixNormBlock {
                    offset: 448,
                    mul: 32,
                    ell: 2,
                    affine_weight: &weight_2.values,
                },
                NequixNormBlock {
                    offset: 608,
                    mul: 32,
                    ell: 3,
                    affine_weight: &weight_3.values,
                },
            ],
            1.0e-12,
        )
    }

    fn gate_hidden(
        &self,
        normed: &[f64],
        rows: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Vec<f64>, NequixError> {
        nequix_gate_forward(
            normed,
            rows,
            input_dim,
            128,
            output_dim,
            &[
                NequixGateBlock {
                    input_offset: 256,
                    output_offset: 128,
                    gate_offset: 128,
                    mul: 64,
                    ell: 1,
                },
                NequixGateBlock {
                    input_offset: 448,
                    output_offset: 320,
                    gate_offset: 192,
                    mul: 32,
                    ell: 2,
                },
                NequixGateBlock {
                    input_offset: 608,
                    output_offset: 480,
                    gate_offset: 224,
                    mul: 32,
                    ell: 3,
                },
            ],
        )
    }

    fn vjp_forces_and_displacement_gradient(
        &self,
        cache: &NequixForwardCache,
        emit_forces: bool,
        dtype: NequixComputeDtype,
    ) -> Result<NequixForceReplay, NequixError> {
        let graph = &cache.graph;
        let readout_weight = self.tensor_ref(self.weight_refs()?.readout)?;
        let grad_node_energy = vec![self.scale; graph.atom_count];
        let mut grad_features = nequix_dense_forward_scaled_vjp_input_dtype(
            &grad_node_energy,
            graph.atom_count,
            128,
            &readout_weight.values,
            1,
            1.0 / 128.0_f64.sqrt(),
            dtype,
        )?;
        let mut grad_radial_basis = vec![0.0; graph.edge_count * 8];
        let mut grad_sh_values = vec![0.0; graph.edge_count * graph.sh_dim];
        for layer in cache.layers.iter().rev() {
            let (grad_previous, layer_grad_radial, layer_grad_sh) =
                self.vjp_layer(layer, graph, &grad_features, dtype)?;
            add_assign(&mut grad_radial_basis, &layer_grad_radial)?;
            add_assign(&mut grad_sh_values, &layer_grad_sh)?;
            grad_features = grad_previous;
        }

        let sh_plan = SphericalHarmonicsPlan::component(3, true)
            .map_err(|err| invalid_model(format!("NequiX SH plan failed: {err}")))?;
        let mut forces = emit_forces.then(|| vec![[0.0; 3]; graph.atom_count]);
        let mut displacement_gradient = [[0.0; 3]; 3];
        for edge_index in 0..graph.edge_count {
            let mut bar_r = [0.0; 3];
            let sh_start = edge_index * graph.sh_dim;
            sh_plan
                .vjp_edge_vector(
                    graph.edge_vectors[edge_index],
                    &grad_sh_values[sh_start..sh_start + graph.sh_dim],
                    &mut bar_r,
                )
                .map_err(|err| invalid_model(format!("NequiX SH VJP failed: {err}")))?;
            let radial_start = edge_index * 8;
            let bar_distance = nequix_radial_basis_vjp_distance(
                graph.distances[edge_index],
                8,
                self.cutoff,
                self.radial_polynomial_p,
                &grad_radial_basis[radial_start..radial_start + 8],
            )?;
            distance_vjp_accumulate(
                graph.edge_vectors[edge_index],
                bar_distance,
                NEQUIX_DISTANCE_EPS,
                &mut bar_r,
            )
            .map_err(|err| invalid_model(format!("NequiX distance VJP failed: {err}")))?;
            if let Some(forces) = &mut forces {
                scatter_edge_vector_vjp(
                    graph.receivers[edge_index],
                    graph.senders[edge_index],
                    bar_r,
                    forces,
                )
                .map_err(|err| invalid_model(format!("NequiX force scatter failed: {err}")))?;
            }
            accumulate_edge_displacement_gradient(
                graph.edge_vectors[edge_index],
                bar_r,
                &mut displacement_gradient,
            );
        }
        Ok(NequixForceReplay {
            forces: forces.unwrap_or_default(),
            displacement_gradient,
        })
    }

    fn vjp_layer(
        &self,
        layer: &NequixLayerCache,
        graph: &NequixGraphCache,
        grad_output: &[f64],
        dtype: NequixComputeDtype,
    ) -> Result<NequixLayerVjp, NequixError> {
        match layer.layer {
            0 => self.vjp_layer0(layer, graph, grad_output, dtype),
            1 | 2 => self.vjp_hidden_layer(layer, graph, grad_output, dtype),
            3 => self.vjp_final_layer(layer, graph, grad_output, dtype),
            _ => Err(invalid_model(format!(
                "unsupported NeQuiX layer {}",
                layer.layer
            ))),
        }
    }

    fn vjp_layer0(
        &self,
        layer: &NequixLayerCache,
        graph: &NequixGraphCache,
        grad_output: &[f64],
        dtype: NequixComputeDtype,
    ) -> Result<NequixLayerVjp, NequixError> {
        let grad_normed =
            self.gate_hidden_vjp(&layer.normed, grad_output, graph.atom_count, 832, 704)?;
        let grad_pre_norm =
            self.norm_layer_vjp(0, &layer.pre_norm, &grad_normed, graph.atom_count, 832)?;
        let grad_aggregated = nequix_irrep_linear_vjp_input_dtype(
            &grad_pre_norm,
            graph.atom_count,
            1424,
            832,
            &self.layer0_linear2_blocks(1.0 / 89.0_f64.sqrt())?,
            dtype,
        )?;
        let grad_skip_input = nequix_irrep_linear_vjp_input_dtype(
            &grad_pre_norm,
            graph.atom_count,
            89,
            832,
            &self.layer0_skip_blocks()?,
            dtype,
        )?;
        let (grad_messages, grad_radial, grad_sh) =
            self.tensor_product_scatter_vjp(layer, graph, &grad_aggregated, dtype)?;
        let refs = self.layer_weight_refs(0)?;
        let linear1_weight = self.tensor_ref(refs.linear_1[0])?;
        let grad_linear1_input = nequix_dense_forward_scaled_vjp_input_dtype(
            &grad_messages,
            graph.atom_count,
            89,
            &linear1_weight.values,
            89,
            1.0 / 89.0_f64.sqrt(),
            dtype,
        )?;
        let grad_previous = add_same_shape(&grad_skip_input, &grad_linear1_input)?;
        Ok((grad_previous, grad_radial, grad_sh))
    }

    fn vjp_hidden_layer(
        &self,
        layer: &NequixLayerCache,
        graph: &NequixGraphCache,
        grad_output: &[f64],
        dtype: NequixComputeDtype,
    ) -> Result<NequixLayerVjp, NequixError> {
        let grad_normed = self.gate_hidden_vjp(
            &layer.normed,
            grad_output,
            graph.atom_count,
            layer.pre_norm_dim,
            layer.output_dim,
        )?;
        let grad_pre_norm = self.norm_layer_vjp(
            layer.layer,
            &layer.pre_norm,
            &grad_normed,
            graph.atom_count,
            layer.pre_norm_dim,
        )?;
        let grad_aggregated =
            self.hidden_linear2_vjp(layer.layer, &grad_pre_norm, graph.atom_count, dtype)?;
        let grad_skip_input =
            self.hidden_skip_vjp(layer.layer, &grad_pre_norm, graph.atom_count, 832, dtype)?;
        let (grad_messages, grad_radial, grad_sh) =
            self.tensor_product_scatter_vjp(layer, graph, &grad_aggregated, dtype)?;
        let grad_linear_input = self.hidden_linear_vjp(
            layer.layer,
            "linear_1",
            &grad_messages,
            graph.atom_count,
            704,
            704,
            dtype,
        )?;
        let grad_previous = add_same_shape(&grad_skip_input, &grad_linear_input)?;
        Ok((grad_previous, grad_radial, grad_sh))
    }

    fn vjp_final_layer(
        &self,
        layer: &NequixLayerCache,
        graph: &NequixGraphCache,
        grad_output: &[f64],
        dtype: NequixComputeDtype,
    ) -> Result<NequixLayerVjp, NequixError> {
        let grad_normed = nequix_gate_vjp_input(
            &layer.normed,
            grad_output,
            graph.atom_count,
            128,
            128,
            128,
            &[],
        )?;
        let refs = self.layer_weight_refs(3)?;
        let norm_weight = self.tensor_ref(refs.layer_norm[0])?;
        let grad_pre_norm = nequix_rms_layer_norm_vjp_input(
            &layer.pre_norm,
            &grad_normed,
            graph.atom_count,
            128,
            &[NequixNormBlock {
                offset: 0,
                mul: 128,
                ell: 0,
                affine_weight: &norm_weight.values,
            }],
            1.0e-12,
        )?;
        let linear2_weight = self.tensor_ref(refs.linear_2[0])?;
        let grad_aggregated = nequix_irrep_linear_vjp_input_dtype(
            &grad_pre_norm,
            graph.atom_count,
            256,
            128,
            &[NequixIrrepLinearBlock {
                input_offset: 0,
                input_mul: 256,
                output_offset: 0,
                output_mul: 128,
                ell: 0,
                weight: &linear2_weight.values,
                alpha: 1.0 / 256.0_f64.sqrt(),
            }],
            dtype,
        )?;
        let skip_weight = self.tensor_ref(refs.skip[0])?;
        let grad_skip_input = nequix_irrep_linear_vjp_input_dtype(
            &grad_pre_norm,
            graph.atom_count,
            704,
            128,
            &[NequixIrrepLinearBlock {
                input_offset: 0,
                input_mul: 128,
                output_offset: 0,
                output_mul: 128,
                ell: 0,
                weight: &skip_weight.values,
                alpha: 1.0 / 128.0_f64.sqrt(),
            }],
            dtype,
        )?;
        let (grad_messages, grad_radial, grad_sh) =
            self.tensor_product_scatter_vjp(layer, graph, &grad_aggregated, dtype)?;
        let grad_linear_input = self.hidden_linear_vjp(
            3,
            "linear_1",
            &grad_messages,
            graph.atom_count,
            704,
            704,
            dtype,
        )?;
        let grad_previous = add_same_shape(&grad_skip_input, &grad_linear_input)?;
        Ok((grad_previous, grad_radial, grad_sh))
    }

    fn tensor_product_scatter_vjp(
        &self,
        layer: &NequixLayerCache,
        graph: &NequixGraphCache,
        grad_aggregated: &[f64],
        dtype: NequixComputeDtype,
    ) -> Result<NequixLayerVjp, NequixError> {
        let plan = self.tp_runtime_plans.get(layer.layer).ok_or_else(|| {
            invalid_model(format!("missing NequiX TP plan for layer {}", layer.layer))
        })?;
        if grad_aggregated.len() != graph.atom_count * layer.aggregated_dim {
            return Err(invalid_model(format!(
                "NeQuiX TP scatter VJP grad_aggregated length must be {}, got {}",
                graph.atom_count * layer.aggregated_dim,
                grad_aggregated.len()
            )));
        }
        let avg_n_neighbors = *self.avg_n_neighbors.get(layer.layer).ok_or_else(|| {
            invalid_model(format!(
                "missing NequiX avg_n_neighbors for layer {}",
                layer.layer
            ))
        })?;
        let scatter_scale = 1.0 / avg_n_neighbors.sqrt();
        let mut grad_messages = vec![0.0; graph.atom_count * layer.message_dim];
        let mut grad_radial_message = vec![0.0; graph.edge_count * plan.weight_numel];
        let mut grad_sh = vec![0.0; graph.edge_count * graph.sh_dim];
        #[cfg(feature = "parallel")]
        if nequix_should_parallelize_edges(graph.edge_count) {
            let min_len = nequix_parallel_edge_chunk_len(graph.edge_count);
            grad_messages = grad_radial_message
                .par_chunks_mut(plan.weight_numel)
                .zip(grad_sh.par_chunks_mut(graph.sh_dim))
                .enumerate()
                .with_min_len(min_len)
                .fold(
                    || {
                        (
                            vec![0.0; graph.atom_count * layer.message_dim],
                            vec![0.0; layer.message_dim],
                        )
                    },
                    |(mut local_grad_messages, mut grad_x), (edge_index, (grad_w, grad_y))| {
                        let receiver = graph.receivers[edge_index];
                        let sender = graph.senders[edge_index];
                        let grad_out_start = receiver * layer.aggregated_dim;
                        let message_start = sender * layer.message_dim;
                        let sh_start = edge_index * graph.sh_dim;
                        let radial_start = edge_index * plan.weight_numel;
                        nequix_tensor_product_vjp_fast(
                            plan,
                            &layer.messages[message_start..message_start + layer.message_dim],
                            &graph.sh_values[sh_start..sh_start + graph.sh_dim],
                            &layer.radial_message[radial_start..radial_start + plan.weight_numel],
                            &grad_aggregated[grad_out_start..grad_out_start + layer.aggregated_dim],
                            scatter_scale,
                            &mut grad_x,
                            grad_y,
                            grad_w,
                        );
                        let local_row = &mut local_grad_messages
                            [message_start..message_start + layer.message_dim];
                        for (dst, src) in local_row.iter_mut().zip(&grad_x) {
                            *dst += src;
                        }
                        (local_grad_messages, grad_x)
                    },
                )
                .map(|(local_grad_messages, _)| local_grad_messages)
                .reduce(
                    || vec![0.0; graph.atom_count * layer.message_dim],
                    |mut left, right| {
                        for (left, right) in left.iter_mut().zip(right) {
                            *left += right;
                        }
                        left
                    },
                );
            let radial_weights = self.radial_weight_refs(layer.layer)?;
            let grad_radial_basis = nequix_radial_mlp_vjp_input_from_cache(
                &graph.radial_basis,
                graph.edge_count,
                &radial_weights,
                &layer.radial_silu_derivatives,
                &grad_radial_message,
                dtype,
            )?;
            return Ok((grad_messages, grad_radial_basis, grad_sh));
        }
        let mut grad_x = vec![0.0; layer.message_dim];
        let mut grad_y = vec![0.0; graph.sh_dim];
        let mut grad_w = vec![0.0; plan.weight_numel];
        for edge_index in 0..graph.edge_count {
            let receiver = graph.receivers[edge_index];
            let sender = graph.senders[edge_index];
            let grad_out_start = receiver * layer.aggregated_dim;
            let message_start = sender * layer.message_dim;
            let sh_start = edge_index * graph.sh_dim;
            let radial_start = edge_index * plan.weight_numel;
            nequix_tensor_product_vjp_fast(
                plan,
                &layer.messages[message_start..message_start + layer.message_dim],
                &graph.sh_values[sh_start..sh_start + graph.sh_dim],
                &layer.radial_message[radial_start..radial_start + plan.weight_numel],
                &grad_aggregated[grad_out_start..grad_out_start + layer.aggregated_dim],
                scatter_scale,
                &mut grad_x,
                &mut grad_y,
                &mut grad_w,
            );
            add_assign_slice(
                &mut grad_messages[message_start..message_start + layer.message_dim],
                &grad_x,
            )?;
            add_assign_slice(&mut grad_sh[sh_start..sh_start + graph.sh_dim], &grad_y)?;
            grad_radial_message[radial_start..radial_start + plan.weight_numel]
                .copy_from_slice(&grad_w);
        }
        let radial_weights = self.radial_weight_refs(layer.layer)?;
        let grad_radial_basis = nequix_radial_mlp_vjp_input_from_cache(
            &graph.radial_basis,
            graph.edge_count,
            &radial_weights,
            &layer.radial_silu_derivatives,
            &grad_radial_message,
            dtype,
        )?;
        Ok((grad_messages, grad_radial_basis, grad_sh))
    }

    fn gate_hidden_vjp(
        &self,
        normed: &[f64],
        grad_output: &[f64],
        rows: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Vec<f64>, NequixError> {
        nequix_gate_vjp_input(
            normed,
            grad_output,
            rows,
            input_dim,
            128,
            output_dim,
            &[
                NequixGateBlock {
                    input_offset: 256,
                    output_offset: 128,
                    gate_offset: 128,
                    mul: 64,
                    ell: 1,
                },
                NequixGateBlock {
                    input_offset: 448,
                    output_offset: 320,
                    gate_offset: 192,
                    mul: 32,
                    ell: 2,
                },
                NequixGateBlock {
                    input_offset: 608,
                    output_offset: 480,
                    gate_offset: 224,
                    mul: 32,
                    ell: 3,
                },
            ],
        )
    }

    fn norm_layer_vjp(
        &self,
        layer: usize,
        pre_norm: &[f64],
        grad_normed: &[f64],
        rows: usize,
        input_dim: usize,
    ) -> Result<Vec<f64>, NequixError> {
        let refs = &self.layer_weight_refs(layer)?.layer_norm;
        let weight_0 = self.tensor_ref(refs[0])?;
        let weight_1 = self.tensor_ref(refs[1])?;
        let weight_2 = self.tensor_ref(refs[2])?;
        let weight_3 = self.tensor_ref(refs[3])?;
        nequix_rms_layer_norm_vjp_input(
            pre_norm,
            grad_normed,
            rows,
            input_dim,
            &[
                NequixNormBlock {
                    offset: 0,
                    mul: 256,
                    ell: 0,
                    affine_weight: &weight_0.values,
                },
                NequixNormBlock {
                    offset: 256,
                    mul: 64,
                    ell: 1,
                    affine_weight: &weight_1.values,
                },
                NequixNormBlock {
                    offset: 448,
                    mul: 32,
                    ell: 2,
                    affine_weight: &weight_2.values,
                },
                NequixNormBlock {
                    offset: 608,
                    mul: 32,
                    ell: 3,
                    affine_weight: &weight_3.values,
                },
            ],
            1.0e-12,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn hidden_linear_vjp(
        &self,
        layer: usize,
        block: &str,
        grad_output: &[f64],
        rows: usize,
        input_dim: usize,
        output_dim: usize,
        dtype: NequixComputeDtype,
    ) -> Result<Vec<f64>, NequixError> {
        let refs = match block {
            "linear_1" => &self.layer_weight_refs(layer)?.linear_1,
            "skip" => &self.layer_weight_refs(layer)?.skip,
            _ => {
                return Err(invalid_model(format!(
                    "unsupported NeQuiX hidden linear block {block}"
                )))
            }
        };
        let weight_0 = self.tensor_ref(refs[0])?;
        let weight_1 = self.tensor_ref(refs[1])?;
        let weight_2 = self.tensor_ref(refs[2])?;
        let weight_3 = self.tensor_ref(refs[3])?;
        nequix_irrep_linear_vjp_input_dtype(
            grad_output,
            rows,
            input_dim,
            output_dim,
            &[
                NequixIrrepLinearBlock {
                    input_offset: 0,
                    input_mul: 128,
                    output_offset: 0,
                    output_mul: 128,
                    ell: 0,
                    weight: &weight_0.values,
                    alpha: 1.0 / 128.0_f64.sqrt(),
                },
                NequixIrrepLinearBlock {
                    input_offset: 128,
                    input_mul: 64,
                    output_offset: 128,
                    output_mul: 64,
                    ell: 1,
                    weight: &weight_1.values,
                    alpha: 1.0 / 64.0_f64.sqrt(),
                },
                NequixIrrepLinearBlock {
                    input_offset: 320,
                    input_mul: 32,
                    output_offset: 320,
                    output_mul: 32,
                    ell: 2,
                    weight: &weight_2.values,
                    alpha: 1.0 / 32.0_f64.sqrt(),
                },
                NequixIrrepLinearBlock {
                    input_offset: 480,
                    input_mul: 32,
                    output_offset: 480,
                    output_mul: 32,
                    ell: 3,
                    weight: &weight_3.values,
                    alpha: 1.0 / 32.0_f64.sqrt(),
                },
            ],
            dtype,
        )
    }

    fn hidden_linear2_vjp(
        &self,
        layer: usize,
        grad_output: &[f64],
        rows: usize,
        dtype: NequixComputeDtype,
    ) -> Result<Vec<f64>, NequixError> {
        let refs = &self.layer_weight_refs(layer)?.linear_2;
        let weight_0 = self.tensor_ref(refs[0])?;
        let weight_1 = self.tensor_ref(refs[1])?;
        let weight_2 = self.tensor_ref(refs[2])?;
        let weight_3 = self.tensor_ref(refs[3])?;
        nequix_irrep_linear_vjp_input_dtype(
            grad_output,
            rows,
            5472,
            832,
            &[
                NequixIrrepLinearBlock {
                    input_offset: 0,
                    input_mul: 256,
                    output_offset: 0,
                    output_mul: 256,
                    ell: 0,
                    weight: &weight_0.values,
                    alpha: 1.0 / 256.0_f64.sqrt(),
                },
                NequixIrrepLinearBlock {
                    input_offset: 256,
                    input_mul: 352,
                    output_offset: 256,
                    output_mul: 64,
                    ell: 1,
                    weight: &weight_1.values,
                    alpha: 1.0 / 352.0_f64.sqrt(),
                },
                NequixIrrepLinearBlock {
                    input_offset: 1312,
                    input_mul: 384,
                    output_offset: 448,
                    output_mul: 32,
                    ell: 2,
                    weight: &weight_2.values,
                    alpha: 1.0 / 384.0_f64.sqrt(),
                },
                NequixIrrepLinearBlock {
                    input_offset: 3232,
                    input_mul: 320,
                    output_offset: 608,
                    output_mul: 32,
                    ell: 3,
                    weight: &weight_3.values,
                    alpha: 1.0 / 320.0_f64.sqrt(),
                },
            ],
            dtype,
        )
    }

    fn hidden_skip_vjp(
        &self,
        layer: usize,
        grad_output: &[f64],
        rows: usize,
        output_dim: usize,
        dtype: NequixComputeDtype,
    ) -> Result<Vec<f64>, NequixError> {
        let refs = &self.layer_weight_refs(layer)?.skip;
        let weight_0 = self.tensor_ref(refs[0])?;
        let weight_1 = self.tensor_ref(refs[1])?;
        let weight_2 = self.tensor_ref(refs[2])?;
        let weight_3 = self.tensor_ref(refs[3])?;
        nequix_irrep_linear_vjp_input_dtype(
            grad_output,
            rows,
            704,
            output_dim,
            &[
                NequixIrrepLinearBlock {
                    input_offset: 0,
                    input_mul: 128,
                    output_offset: 0,
                    output_mul: 256,
                    ell: 0,
                    weight: &weight_0.values,
                    alpha: 1.0 / 128.0_f64.sqrt(),
                },
                NequixIrrepLinearBlock {
                    input_offset: 128,
                    input_mul: 64,
                    output_offset: 256,
                    output_mul: 64,
                    ell: 1,
                    weight: &weight_1.values,
                    alpha: 1.0 / 64.0_f64.sqrt(),
                },
                NequixIrrepLinearBlock {
                    input_offset: 320,
                    input_mul: 32,
                    output_offset: 448,
                    output_mul: 32,
                    ell: 2,
                    weight: &weight_2.values,
                    alpha: 1.0 / 32.0_f64.sqrt(),
                },
                NequixIrrepLinearBlock {
                    input_offset: 480,
                    input_mul: 32,
                    output_offset: 608,
                    output_mul: 32,
                    ell: 3,
                    weight: &weight_3.values,
                    alpha: 1.0 / 32.0_f64.sqrt(),
                },
            ],
            dtype,
        )
    }

    fn species_indices(&self, structure: &Structure) -> Result<Vec<usize>, NequixError> {
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
            .map(|symbol| {
                self.type_names
                    .iter()
                    .position(|candidate| candidate == symbol)
                    .ok_or_else(|| invalid_model(format!("unsupported NequiX species {symbol}")))
            })
            .collect()
    }

    fn required_tensor(&self, tree_path: &str) -> Result<&NequixTensor, NequixError> {
        self.tensor_by_tree_path(tree_path)
            .ok_or_else(|| invalid_model(format!("missing NequiX tensor {tree_path}")))
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
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    #[must_use]
    pub fn total_numel(&self) -> usize {
        self.total_numel
    }

    #[must_use]
    pub fn loaded_tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Returns `true` when the runtime tensor payloads and tensor-product plans
    /// required for a forward/backward pass are loaded. This gates the opt-in
    /// f32 compute path (and matches the prerequisites checked by the f64
    /// prediction entry points).
    #[must_use]
    pub fn supports_runtime_prediction(&self) -> bool {
        self.tensors.len() == EXPECTED_TENSOR_COUNT
            && self.tp_plans.len() == EXPECTED_N_LAYERS
            && self.tp_runtime_plans.len() == EXPECTED_N_LAYERS
            && self.weight_refs.is_some()
    }

    #[must_use]
    pub fn tensors(&self) -> &[NequixTensor] {
        &self.tensors
    }

    #[must_use]
    pub fn tensor_by_tree_path(&self, tree_path: &str) -> Option<&NequixTensor> {
        self.tensor_lookup
            .get(tree_path)
            .and_then(|index| self.tensors.get(*index))
    }

    #[must_use]
    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    #[must_use]
    pub fn artifact_size_bytes(&self) -> u64 {
        self.artifact_size_bytes
    }

    fn from_spec(spec: StateSpec, root: Option<&Path>) -> Result<Self, NequixError> {
        expect_str("schema_version", &spec.schema_version, STATE_SCHEMA_VERSION)?;
        expect_str("architecture", &spec.architecture, ARCHITECTURE)?;
        expect_str(
            "runtime_contract.state_dtype",
            &spec.runtime_contract.state_dtype,
            "f64",
        )?;
        expect_str(
            "runtime_contract.runtime_arithmetic_dtype",
            &spec.runtime_contract.runtime_arithmetic_dtype,
            "f64",
        )?;
        if spec.runtime_contract.kernel {
            return Err(invalid_model(
                "runtime_contract.kernel must be false for pinned NequiX-PFT exports",
            ));
        }
        if (spec.model_config.cutoff - EXPECTED_CUTOFF).abs() > 1.0e-12 {
            return Err(invalid_model(format!(
                "model_config.cutoff must be {EXPECTED_CUTOFF}, got {}",
                spec.model_config.cutoff
            )));
        }
        if spec.model_config.lmax != EXPECTED_L_MAX {
            return Err(invalid_model(format!(
                "model_config.lmax must be {EXPECTED_L_MAX}, got {}",
                spec.model_config.lmax
            )));
        }
        if spec.model_config.n_layers != EXPECTED_N_LAYERS {
            return Err(invalid_model(format!(
                "model_config.n_layers must be {EXPECTED_N_LAYERS}, got {}",
                spec.model_config.n_layers
            )));
        }
        if spec.pytree.tensor_count != EXPECTED_TENSOR_COUNT {
            return Err(invalid_model(format!(
                "pytree.tensor_count must be {EXPECTED_TENSOR_COUNT}, got {}",
                spec.pytree.tensor_count
            )));
        }
        if spec.pytree.total_numel != EXPECTED_TOTAL_NUMEL {
            return Err(invalid_model(format!(
                "pytree.total_numel must be {EXPECTED_TOTAL_NUMEL}, got {}",
                spec.pytree.total_numel
            )));
        }
        if spec.pytree.total_bytes_f64 != EXPECTED_TOTAL_BYTES_F64 {
            return Err(invalid_model(format!(
                "pytree.total_bytes_f64 must be {EXPECTED_TOTAL_BYTES_F64}, got {}",
                spec.pytree.total_bytes_f64
            )));
        }
        let tensors = if let Some(tensors) = spec.tensors {
            if tensors.len() != spec.pytree.tensor_count {
                return Err(invalid_model(format!(
                    "tensors length must match pytree.tensor_count {}, got {}",
                    spec.pytree.tensor_count,
                    tensors.len()
                )));
            }
            if let Some(root) = root {
                load_exported_tensors(root, &tensors)?
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let type_names = spec
            .model_config
            .atomic_numbers
            .iter()
            .map(|atomic_number| {
                element_symbol(*atomic_number)
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        invalid_model(format!("unsupported atomic number {atomic_number}"))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let tensor_lookup = tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| (tensor.tree_path.clone(), index))
            .collect();
        let weight_refs = build_nequix_weight_refs(&tensor_lookup)?;
        let tp_plans = if let (Some(root), false) = (root, tensors.is_empty()) {
            load_execution_plans(root, &spec.plans)?
        } else {
            Vec::new()
        };
        let tp_runtime_plans = tp_plans
            .iter()
            .map(nequix_tensor_product_runtime_plan)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            selected_model: spec.selected_model,
            type_names,
            cutoff: spec.model_config.cutoff,
            l_max: spec.model_config.lmax,
            n_layers: spec.model_config.n_layers,
            tensor_count: spec.pytree.tensor_count,
            total_numel: spec.pytree.total_numel,
            artifact_sha256: spec.artifact.sha256,
            artifact_size_bytes: spec.artifact.size_bytes,
            tensors,
            tensor_lookup,
            weight_refs,
            tp_plans,
            tp_runtime_plans,
            avg_n_neighbors: spec.model_config.avg_n_neighbors_by_layer(),
            radial_polynomial_p: spec.model_config.radial_polynomial_p,
            scale: spec.model_config.scale,
            shift: spec.model_config.shift,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StateSpec {
    schema_version: String,
    architecture: String,
    selected_model: String,
    artifact: ArtifactSpec,
    model_config: ModelConfigSpec,
    runtime_contract: RuntimeContractSpec,
    pytree: PytreeSpec,
    tensors: Option<Vec<TensorRecordSpec>>,
    #[allow(dead_code)]
    reference: serde_json::Value,
    #[allow(dead_code)]
    source: serde_json::Value,
    #[allow(dead_code)]
    python: serde_json::Value,
    #[allow(dead_code)]
    plans: Vec<PlanRecordSpec>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactSpec {
    sha256: String,
    size_bytes: u64,
    #[allow(dead_code)]
    filename: String,
    #[allow(dead_code)]
    source_url: String,
}

#[derive(Debug, Deserialize)]
struct ModelConfigSpec {
    cutoff: f64,
    lmax: usize,
    n_layers: usize,
    atomic_numbers: Vec<u8>,
    avg_n_neighbors: f64,
    radial_polynomial_p: f64,
    scale: f64,
    shift: f64,
    #[allow(dead_code)]
    hidden_irreps: String,
    #[allow(dead_code)]
    layer_norm: bool,
    #[allow(dead_code)]
    radial_basis_size: usize,
    #[allow(dead_code)]
    radial_mlp_layers: usize,
    #[allow(dead_code)]
    radial_mlp_size: usize,
}

impl ModelConfigSpec {
    fn avg_n_neighbors_by_layer(&self) -> Vec<f64> {
        vec![self.avg_n_neighbors; self.n_layers]
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanRecordSpec {
    #[allow(dead_code)]
    name: String,
    path: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RuntimeContractSpec {
    kernel: bool,
    state_dtype: String,
    runtime_arithmetic_dtype: String,
    #[allow(dead_code)]
    source_tree: String,
    #[allow(dead_code)]
    unsupported: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PytreeSpec {
    tensor_count: usize,
    total_numel: usize,
    total_bytes_f64: usize,
    #[allow(dead_code)]
    source_dtype_counts: serde_json::Value,
    #[allow(dead_code)]
    treedef: String,
}

#[derive(Debug, Deserialize)]
struct TensorRecordSpec {
    tree_path: String,
    dtype: String,
    shape: Vec<usize>,
    path: PathBuf,
    #[allow(dead_code)]
    index: usize,
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    layout: String,
    #[allow(dead_code)]
    nbytes: usize,
    #[allow(dead_code)]
    numel: usize,
    #[allow(dead_code)]
    sha256: String,
    #[allow(dead_code)]
    source_dtype: String,
    #[allow(dead_code)]
    source_nbytes: usize,
    #[allow(dead_code)]
    source_numel: usize,
    #[allow(dead_code)]
    source_sha256: String,
    #[allow(dead_code)]
    source_shape: Vec<usize>,
}

fn load_exported_tensors(
    root: &Path,
    specs: &[TensorRecordSpec],
) -> Result<Vec<NequixTensor>, NequixError> {
    specs
        .iter()
        .map(|spec| {
            expect_str("tensor dtype", &spec.dtype, "f64")?;
            let values = read_f64_tensor(root, spec)?;
            Ok(NequixTensor {
                tree_path: spec.tree_path.clone(),
                shape: spec.shape.clone(),
                values,
            })
        })
        .collect()
}

fn load_execution_plans(
    root: &Path,
    specs: &[PlanRecordSpec],
) -> Result<Vec<TensorProductPlan>, NequixError> {
    let plan_spec = specs
        .iter()
        .find(|spec| spec.path == Path::new("plans/nequix_plan.json"))
        .or_else(|| specs.first())
        .ok_or_else(|| invalid_model("NequiX export must contain a runtime plan record"))?;
    let path = root.join(&plan_spec.path);
    let text = fs::read_to_string(&path).map_err(|source| NequixError::Io {
        path: path.clone(),
        source,
    })?;
    let value = serde_json::from_str::<serde_json::Value>(&text)
        .map_err(|err| invalid_model(format!("invalid NequiX execution plan JSON: {err}")))?;
    let layers = value["layers"]
        .as_array()
        .ok_or_else(|| invalid_model("NequiX execution plan must contain layers array"))?;
    if layers.len() != EXPECTED_N_LAYERS {
        return Err(invalid_model(format!(
            "NequiX execution plan must contain {EXPECTED_N_LAYERS} layers, got {}",
            layers.len()
        )));
    }
    layers
        .iter()
        .map(|layer| nequix_tensor_product_plan_from_json(&layer["tensor_product"]["plan"]))
        .collect()
}

fn read_f64_tensor(root: &Path, spec: &TensorRecordSpec) -> Result<Vec<f64>, NequixError> {
    let expected_bytes = spec.shape.iter().try_fold(8usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            invalid_model(format!(
                "tensor {} shape {:?} byte count overflows usize",
                spec.tree_path, spec.shape
            ))
        })
    })?;
    let path = root.join(&spec.path);
    let bytes = fs::read(&path).map_err(|source| NequixError::Io {
        path: path.clone(),
        source,
    })?;
    if bytes.len() != expected_bytes {
        return Err(NequixError::TensorSizeMismatch {
            name: spec.tree_path.clone(),
            shape: spec.shape.clone(),
            expected_bytes,
            actual_bytes: bytes.len(),
        });
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| {
            let mut value = [0_u8; 8];
            value.copy_from_slice(chunk);
            f64::from_le_bytes(value)
        })
        .collect())
}

fn element_symbol(atomic_number: u8) -> Option<&'static str> {
    match atomic_number {
        1 => Some("H"),
        2 => Some("He"),
        3 => Some("Li"),
        4 => Some("Be"),
        5 => Some("B"),
        6 => Some("C"),
        7 => Some("N"),
        8 => Some("O"),
        9 => Some("F"),
        10 => Some("Ne"),
        11 => Some("Na"),
        12 => Some("Mg"),
        13 => Some("Al"),
        14 => Some("Si"),
        15 => Some("P"),
        16 => Some("S"),
        17 => Some("Cl"),
        18 => Some("Ar"),
        19 => Some("K"),
        20 => Some("Ca"),
        21 => Some("Sc"),
        22 => Some("Ti"),
        23 => Some("V"),
        24 => Some("Cr"),
        25 => Some("Mn"),
        26 => Some("Fe"),
        27 => Some("Co"),
        28 => Some("Ni"),
        29 => Some("Cu"),
        30 => Some("Zn"),
        31 => Some("Ga"),
        32 => Some("Ge"),
        33 => Some("As"),
        34 => Some("Se"),
        35 => Some("Br"),
        36 => Some("Kr"),
        37 => Some("Rb"),
        38 => Some("Sr"),
        39 => Some("Y"),
        40 => Some("Zr"),
        41 => Some("Nb"),
        42 => Some("Mo"),
        43 => Some("Tc"),
        44 => Some("Ru"),
        45 => Some("Rh"),
        46 => Some("Pd"),
        47 => Some("Ag"),
        48 => Some("Cd"),
        49 => Some("In"),
        50 => Some("Sn"),
        51 => Some("Sb"),
        52 => Some("Te"),
        53 => Some("I"),
        54 => Some("Xe"),
        55 => Some("Cs"),
        56 => Some("Ba"),
        57 => Some("La"),
        58 => Some("Ce"),
        59 => Some("Pr"),
        60 => Some("Nd"),
        61 => Some("Pm"),
        62 => Some("Sm"),
        63 => Some("Eu"),
        64 => Some("Gd"),
        65 => Some("Tb"),
        66 => Some("Dy"),
        67 => Some("Ho"),
        68 => Some("Er"),
        69 => Some("Tm"),
        70 => Some("Yb"),
        71 => Some("Lu"),
        72 => Some("Hf"),
        73 => Some("Ta"),
        74 => Some("W"),
        75 => Some("Re"),
        76 => Some("Os"),
        77 => Some("Ir"),
        78 => Some("Pt"),
        79 => Some("Au"),
        80 => Some("Hg"),
        81 => Some("Tl"),
        82 => Some("Pb"),
        83 => Some("Bi"),
        89 => Some("Ac"),
        90 => Some("Th"),
        91 => Some("Pa"),
        92 => Some("U"),
        93 => Some("Np"),
        94 => Some("Pu"),
        _ => None,
    }
}

fn expect_str(name: &str, actual: &str, expected: &str) -> Result<(), NequixError> {
    if actual == expected {
        Ok(())
    } else {
        Err(invalid_model(format!(
            "{name} must be {expected:?}, got {actual:?}"
        )))
    }
}

fn invalid_model(message: impl Into<String>) -> NequixError {
    NequixError::InvalidModel(message.into())
}
