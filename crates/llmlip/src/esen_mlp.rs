//! eSEN-specific scalar MLP primitives.

use e3nn_core::{CoreError, Result};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::OnceLock;
use std::time::Instant;

fn esen_mlp_trace_timings_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("LLMLIP_UMA_TRACE_TIMINGS").ok().as_deref(),
            Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
        )
    })
}

#[inline(always)]
fn esen_mlp_trace_start() -> Option<Instant> {
    esen_mlp_trace_timings_enabled().then(Instant::now)
}

#[inline(always)]
fn trace_esen_mlp_layer_stage(
    prefix: &'static str,
    layer_index: usize,
    suffix: &'static str,
    start: Option<Instant>,
) {
    if let Some(start) = start {
        eprintln!(
            "uma_trace stage={prefix}.layer{layer_index}.{suffix} elapsed_ms={:.6}",
            start.elapsed().as_secs_f64() * 1e3
        );
    }
}

#[inline(always)]
fn trace_esen_mlp_stage(stage: &'static str, start: Option<Instant>) {
    if let Some(start) = start {
        eprintln!(
            "uma_trace stage={stage} elapsed_ms={:.6}",
            start.elapsed().as_secs_f64() * 1e3
        );
    }
}

#[allow(clippy::uninit_vec, unsafe_code)]
fn uninit_f32_vec(len: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(len);
    // Invariant: callers only use this for scratch buffers that are fully
    // overwritten by dense or row-wise kernels before any read.
    unsafe {
        values.set_len(len);
    }
    values
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(clippy::too_many_arguments)]
mod portable_blas {
    use e3nn_core::Result;

    pub fn dense_forward_transposed_weight_strided_output(
        input: &[f64],
        rows: usize,
        in_dim: usize,
        weight_out_by_in: &[f64],
        out_dim: usize,
        out_row_stride: usize,
        out_feature_offset: usize,
        out: &mut [f64],
    ) -> Result<()> {
        super::ensure_blas_dims(&[
            ("rows", rows),
            ("in_dim", in_dim),
            ("out_dim", out_dim),
            ("out_row_stride", out_row_stride),
        ])?;
        crate::blas_backend::dense_forward_output_major_weight_strided_output(
            input,
            rows,
            in_dim,
            weight_out_by_in,
            out_dim,
            out,
            out_row_stride,
            out_feature_offset,
            1.0,
            0.0,
        );
        Ok(())
    }
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
fn ensure_blas_dims(dimensions: &[(&str, usize)]) -> Result<()> {
    for &(name, value) in dimensions {
        if value > i32::MAX as usize {
            return Err(invalid_mlp(format!(
                "eSEN radial MLP BLAS dimension {name}={value} exceeds c_int::MAX"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct EsenRadialMlpLayer {
    in_dim: usize,
    out_dim: usize,
    weight: Vec<f64>,
    bias: Vec<f64>,
    layer_norm_weight: Option<Vec<f64>>,
    layer_norm_bias: Option<Vec<f64>>,
    weight_f32: Vec<f32>,
    weight_f32_in_by_out: Vec<f32>,
    bias_f32: Vec<f32>,
    layer_norm_weight_f32: Option<Vec<f32>>,
    layer_norm_bias_f32: Option<Vec<f32>>,
}

impl EsenRadialMlpLayer {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        weight: Vec<f64>,
        bias: Vec<f64>,
        layer_norm: Option<(Vec<f64>, Vec<f64>)>,
    ) -> Result<Self> {
        let (layer_norm_weight, layer_norm_bias) = match layer_norm {
            Some((weight, bias)) => (Some(weight), Some(bias)),
            None => (None, None),
        };
        let weight_f32 = weight.iter().map(|&value| value as f32).collect();
        let weight_f32_in_by_out =
            transpose_output_major_f64_to_input_major_f32(in_dim, out_dim, &weight);
        let bias_f32 = bias.iter().map(|&value| value as f32).collect();
        let layer_norm_weight_f32 = layer_norm_weight
            .as_ref()
            .map(|values| values.iter().map(|&value| value as f32).collect());
        let layer_norm_bias_f32 = layer_norm_bias
            .as_ref()
            .map(|values| values.iter().map(|&value| value as f32).collect());
        let layer = Self {
            in_dim,
            out_dim,
            weight,
            bias,
            layer_norm_weight,
            layer_norm_bias,
            weight_f32,
            weight_f32_in_by_out,
            bias_f32,
            layer_norm_weight_f32,
            layer_norm_bias_f32,
        };
        layer.validate()?;
        Ok(layer)
    }

    #[must_use]
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    #[must_use]
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    #[must_use]
    pub fn uses_layer_norm_silu(&self) -> bool {
        self.layer_norm_weight.is_some()
    }

    fn validate(&self) -> Result<()> {
        if self.in_dim == 0 || self.out_dim == 0 {
            return Err(invalid_mlp(format!(
                "eSEN radial MLP layer dimensions must be nonzero, got {} -> {}",
                self.in_dim, self.out_dim
            )));
        }
        check_len(
            "esen_radial_mlp layer weight",
            self.in_dim * self.out_dim,
            self.weight.len(),
        )?;
        check_len("esen_radial_mlp layer bias", self.out_dim, self.bias.len())?;
        check_len(
            "esen_radial_mlp layer f32 weight",
            self.in_dim * self.out_dim,
            self.weight_f32.len(),
        )?;
        check_len(
            "esen_radial_mlp layer f32 forward weight",
            self.in_dim * self.out_dim,
            self.weight_f32_in_by_out.len(),
        )?;
        check_len(
            "esen_radial_mlp layer f32 bias",
            self.out_dim,
            self.bias_f32.len(),
        )?;
        match (&self.layer_norm_weight, &self.layer_norm_bias) {
            (Some(weight), Some(bias)) => {
                check_len(
                    "esen_radial_mlp layer_norm_weight",
                    self.out_dim,
                    weight.len(),
                )?;
                check_len("esen_radial_mlp layer_norm_bias", self.out_dim, bias.len())?;
                let Some(weight_f32) = &self.layer_norm_weight_f32 else {
                    return Err(invalid_mlp(
                        "layer norm f32 weight must be present when f64 weight is present",
                    ));
                };
                let Some(bias_f32) = &self.layer_norm_bias_f32 else {
                    return Err(invalid_mlp(
                        "layer norm f32 bias must be present when f64 bias is present",
                    ));
                };
                check_len(
                    "esen_radial_mlp layer_norm_weight_f32",
                    self.out_dim,
                    weight_f32.len(),
                )?;
                check_len(
                    "esen_radial_mlp layer_norm_bias_f32",
                    self.out_dim,
                    bias_f32.len(),
                )?;
            }
            (None, None) => {
                if self.layer_norm_weight_f32.is_some() || self.layer_norm_bias_f32.is_some() {
                    return Err(invalid_mlp(
                        "layer norm f32 weight and bias must be absent when f64 layer norm is absent",
                    ));
                }
            }
            _ => {
                return Err(invalid_mlp(
                    "layer norm weight and bias must be both present or both absent",
                ));
            }
        }
        if self
            .weight
            .iter()
            .chain(self.bias.iter())
            .chain(self.layer_norm_weight.iter().flatten())
            .chain(self.layer_norm_bias.iter().flatten())
            .any(|value| !value.is_finite())
        {
            return Err(invalid_mlp("eSEN radial MLP parameters must be finite"));
        }
        if self
            .weight_f32
            .iter()
            .chain(self.bias_f32.iter())
            .chain(self.layer_norm_weight_f32.iter().flatten())
            .chain(self.layer_norm_bias_f32.iter().flatten())
            .any(|value| !value.is_finite())
        {
            return Err(invalid_mlp("eSEN radial MLP f32 parameters must be finite"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EsenRadialMlpCache {
    layer_caches: Vec<EsenRadialMlpLayerCache>,
}

impl EsenRadialMlpCache {
    pub fn clear(&mut self) {
        self.layer_caches.clear();
    }
}

#[derive(Clone, Debug, PartialEq)]
struct EsenRadialMlpLayerCache {
    linear_output: Vec<f64>,
    silu_derivative: Option<Vec<f64>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EsenRadialMlpBatchCache {
    rows: usize,
    layer_caches: Vec<EsenRadialMlpBatchLayerCache>,
}

impl EsenRadialMlpBatchCache {
    pub fn clear(&mut self) {
        self.rows = 0;
        self.layer_caches.clear();
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }
}

#[derive(Clone, Debug, PartialEq)]
struct EsenRadialMlpBatchLayerCache {
    linear_output: Vec<f64>,
    silu_derivative: Option<Vec<f64>>,
}

/// f32-native batch cache for [`EsenRadialMlp`].
///
/// Populated by [`EsenRadialMlp::forward_batch_f32_with_f32_cache`] and
/// consumed by [`EsenRadialMlp::vjp_input_only_batch_contiguous_cache_f32`].
/// Keeps all cached activations in f32 (unlike [`EsenRadialMlpBatchCache`]
/// which stores f32-precision values in f64 containers). The f64 path is
/// completely unaffected by this type.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EsenRadialMlpBatchCacheF32 {
    rows: usize,
    layer_caches: Vec<EsenRadialMlpBatchLayerCacheF32>,
}

impl EsenRadialMlpBatchCacheF32 {
    pub fn clear(&mut self) {
        self.rows = 0;
        self.layer_caches.clear();
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }
}

#[derive(Clone, Debug, PartialEq)]
struct EsenRadialMlpBatchLayerCacheF32 {
    linear_output: Vec<f32>,
    silu_derivative: Option<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EsenRadialMlp {
    layers: Vec<EsenRadialMlpLayer>,
    layer_norm_eps: f64,
}

impl EsenRadialMlp {
    pub fn new(layers: Vec<EsenRadialMlpLayer>, layer_norm_eps: f64) -> Result<Self> {
        if layers.is_empty() {
            return Err(invalid_mlp(
                "eSEN radial MLP must contain at least one layer",
            ));
        }
        if !layer_norm_eps.is_finite() || layer_norm_eps <= 0.0 {
            return Err(invalid_mlp(format!(
                "layer_norm_eps must be positive and finite, got {layer_norm_eps}"
            )));
        }
        for (index, layer) in layers.iter().enumerate() {
            layer.validate()?;
            let is_last = index + 1 == layers.len();
            if is_last && layer.uses_layer_norm_silu() {
                return Err(invalid_mlp(
                    "FairChem RadialMLP final linear layer must not use LayerNorm/SiLU",
                ));
            }
            if !is_last && !layer.uses_layer_norm_silu() {
                return Err(invalid_mlp(
                    "FairChem RadialMLP hidden layers must use LayerNorm/SiLU",
                ));
            }
            if index > 0 {
                let previous = &layers[index - 1];
                if previous.out_dim != layer.in_dim {
                    return Err(invalid_mlp(format!(
                        "eSEN radial MLP layer {index} input dim {} does not match previous output dim {}",
                        layer.in_dim, previous.out_dim
                    )));
                }
            }
        }
        Ok(Self {
            layers,
            layer_norm_eps,
        })
    }

    #[must_use]
    pub fn input_dim(&self) -> usize {
        self.layers[0].in_dim
    }

    #[must_use]
    pub fn output_dim(&self) -> usize {
        self.layers[self.layers.len() - 1].out_dim
    }

    #[must_use]
    pub fn final_linear_input_dim(&self) -> usize {
        self.layers[self.layers.len() - 1].in_dim
    }

    pub fn forward(
        &self,
        input: &[f64],
        out: &mut [f64],
        cache: &mut EsenRadialMlpCache,
    ) -> Result<()> {
        check_len(
            "esen_radial_mlp_forward input",
            self.input_dim(),
            input.len(),
        )?;
        check_len(
            "esen_radial_mlp_forward output",
            self.output_dim(),
            out.len(),
        )?;
        cache.clear();

        let mut current = input.to_vec();
        for layer in &self.layers {
            let linear_output = dense_forward(layer, &current);
            if let (Some(layer_norm_weight), Some(layer_norm_bias)) =
                (&layer.layer_norm_weight, &layer.layer_norm_bias)
            {
                let layer_norm_output = layer_norm_forward(
                    &linear_output,
                    layer_norm_weight,
                    layer_norm_bias,
                    self.layer_norm_eps,
                );
                let mut next = Vec::with_capacity(layer_norm_output.len());
                let mut silu_derivative = Vec::with_capacity(layer_norm_output.len());
                for value in &layer_norm_output {
                    let (activated, derivative) = silu_value_and_derivative(*value);
                    next.push(activated);
                    silu_derivative.push(derivative);
                }
                current = next;
                cache.layer_caches.push(EsenRadialMlpLayerCache {
                    linear_output,
                    silu_derivative: Some(silu_derivative),
                });
            } else {
                current = linear_output.clone();
                cache.layer_caches.push(EsenRadialMlpLayerCache {
                    linear_output,
                    silu_derivative: None,
                });
            }
        }
        out.copy_from_slice(&current);
        Ok(())
    }

    pub fn forward_batch(
        &self,
        rows: usize,
        input: &[f64],
        out: &mut [f64],
        caches: &mut Vec<EsenRadialMlpCache>,
    ) -> Result<()> {
        check_len(
            "esen_radial_mlp_forward_batch input",
            rows * self.input_dim(),
            input.len(),
        )?;
        check_len(
            "esen_radial_mlp_forward_batch output",
            rows * self.output_dim(),
            out.len(),
        )?;
        caches.clear();
        caches.resize_with(rows, EsenRadialMlpCache::default);

        let mut current: Cow<'_, [f64]> = Cow::Borrowed(input);
        for layer in &self.layers {
            let mut linear_output = vec![0.0; rows * layer.out_dim];
            dense_forward_rows(layer, rows, current.as_ref(), &mut linear_output)?;
            if let (Some(layer_norm_weight), Some(layer_norm_bias)) =
                (&layer.layer_norm_weight, &layer.layer_norm_bias)
            {
                let mut next = vec![0.0; linear_output.len()];
                let mut silu_derivatives = vec![0.0; linear_output.len()];

                #[cfg(feature = "parallel")]
                if rayon::current_num_threads() > 1 && rows >= e3nn_core::esen_parallel_min_nodes()
                {
                    caches
                        .par_iter_mut()
                        .zip(linear_output.par_chunks(layer.out_dim))
                        .zip(next.par_chunks_mut(layer.out_dim))
                        .zip(silu_derivatives.par_chunks_mut(layer.out_dim))
                        .for_each(|(((cache, linear_row), next_row), silu_derivative_row)| {
                            layer_norm_forward_into(
                                linear_row,
                                layer_norm_weight,
                                layer_norm_bias,
                                self.layer_norm_eps,
                                next_row,
                            );
                            for index in 0..layer.out_dim {
                                let (activated, derivative) =
                                    silu_value_and_derivative(next_row[index]);
                                next_row[index] = activated;
                                silu_derivative_row[index] = derivative;
                            }
                            cache.layer_caches.push(EsenRadialMlpLayerCache {
                                linear_output: linear_row.to_vec(),
                                silu_derivative: Some(silu_derivative_row.to_vec()),
                            });
                        });
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for (row, cache) in caches.iter_mut().enumerate().take(rows) {
                        let start = row * layer.out_dim;
                        let end = start + layer.out_dim;
                        layer_norm_forward_into(
                            &linear_output[start..end],
                            layer_norm_weight,
                            layer_norm_bias,
                            self.layer_norm_eps,
                            &mut next[start..end],
                        );
                        for index in start..end {
                            let (activated, derivative) = silu_value_and_derivative(next[index]);
                            next[index] = activated;
                            silu_derivatives[index] = derivative;
                        }
                        cache.layer_caches.push(EsenRadialMlpLayerCache {
                            linear_output: linear_output[start..end].to_vec(),
                            silu_derivative: Some(silu_derivatives[start..end].to_vec()),
                        });
                    }
                }
                #[cfg(feature = "parallel")]
                if rayon::current_num_threads() <= 1 || rows < e3nn_core::esen_parallel_min_nodes()
                {
                    for (row, cache) in caches.iter_mut().enumerate().take(rows) {
                        let start = row * layer.out_dim;
                        let end = start + layer.out_dim;
                        layer_norm_forward_into(
                            &linear_output[start..end],
                            layer_norm_weight,
                            layer_norm_bias,
                            self.layer_norm_eps,
                            &mut next[start..end],
                        );
                        for index in start..end {
                            let (activated, derivative) = silu_value_and_derivative(next[index]);
                            next[index] = activated;
                            silu_derivatives[index] = derivative;
                        }
                        cache.layer_caches.push(EsenRadialMlpLayerCache {
                            linear_output: linear_output[start..end].to_vec(),
                            silu_derivative: Some(silu_derivatives[start..end].to_vec()),
                        });
                    }
                }
                current = Cow::Owned(next);
            } else {
                #[cfg(feature = "parallel")]
                if rayon::current_num_threads() > 1 && rows >= e3nn_core::esen_parallel_min_nodes()
                {
                    caches
                        .par_iter_mut()
                        .zip(linear_output.par_chunks(layer.out_dim))
                        .for_each(|(cache, linear_row)| {
                            cache.layer_caches.push(EsenRadialMlpLayerCache {
                                linear_output: linear_row.to_vec(),
                                silu_derivative: None,
                            });
                        });
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for (row, cache) in caches.iter_mut().enumerate().take(rows) {
                        let start = row * layer.out_dim;
                        let end = start + layer.out_dim;
                        cache.layer_caches.push(EsenRadialMlpLayerCache {
                            linear_output: linear_output[start..end].to_vec(),
                            silu_derivative: None,
                        });
                    }
                }
                #[cfg(feature = "parallel")]
                if rayon::current_num_threads() <= 1 || rows < e3nn_core::esen_parallel_min_nodes()
                {
                    for (row, cache) in caches.iter_mut().enumerate().take(rows) {
                        let start = row * layer.out_dim;
                        let end = start + layer.out_dim;
                        cache.layer_caches.push(EsenRadialMlpLayerCache {
                            linear_output: linear_output[start..end].to_vec(),
                            silu_derivative: None,
                        });
                    }
                }
                current = Cow::Owned(linear_output);
            }
        }
        out.copy_from_slice(current.as_ref());
        Ok(())
    }

    pub fn forward_batch_contiguous_cache(
        &self,
        rows: usize,
        input: &[f64],
        out: &mut [f64],
        cache: &mut EsenRadialMlpBatchCache,
    ) -> Result<()> {
        check_len(
            "esen_radial_mlp_forward_batch_contiguous_cache input",
            rows * self.input_dim(),
            input.len(),
        )?;
        check_len(
            "esen_radial_mlp_forward_batch_contiguous_cache output",
            rows * self.output_dim(),
            out.len(),
        )?;
        cache.clear();
        cache.rows = rows;

        let mut current: Cow<'_, [f64]> = Cow::Borrowed(input);
        for layer in &self.layers {
            if !layer.uses_layer_norm_silu() {
                dense_forward_rows(layer, rows, current.as_ref(), out)?;
                cache.layer_caches.push(EsenRadialMlpBatchLayerCache {
                    linear_output: Vec::new(),
                    silu_derivative: None,
                });
                return Ok(());
            }

            let mut linear_output = vec![0.0; rows * layer.out_dim];
            dense_forward_rows(layer, rows, current.as_ref(), &mut linear_output)?;
            let (Some(layer_norm_weight), Some(layer_norm_bias)) =
                (&layer.layer_norm_weight, &layer.layer_norm_bias)
            else {
                return Err(invalid_mlp(
                    "FairChem RadialMLP hidden layer is missing LayerNorm/SiLU cache inputs",
                ));
            };
            let mut next = vec![0.0; linear_output.len()];
            let mut silu_derivatives = vec![0.0; linear_output.len()];

            // Fast path: vectorized f64 LayerNorm+SiLU (ArmPL SVE `exp`). The
            // scalar path below is the fallback when the SVE kernels are
            // unavailable/disabled.
            if !layer_norm_silu_sve_rows_f64(
                rows,
                layer.out_dim,
                &linear_output,
                layer_norm_weight,
                layer_norm_bias,
                self.layer_norm_eps,
                &mut next,
                &mut silu_derivatives,
            ) {
                #[cfg(feature = "parallel")]
                if rayon::current_num_threads() > 1 && rows >= e3nn_core::esen_parallel_min_nodes()
                {
                    linear_output
                        .par_chunks(layer.out_dim)
                        .zip(next.par_chunks_mut(layer.out_dim))
                        .zip(silu_derivatives.par_chunks_mut(layer.out_dim))
                        .for_each(|((linear_row, next_row), silu_derivative_row)| {
                            layer_norm_forward_into(
                                linear_row,
                                layer_norm_weight,
                                layer_norm_bias,
                                self.layer_norm_eps,
                                next_row,
                            );
                            for index in 0..layer.out_dim {
                                let (activated, derivative) =
                                    silu_value_and_derivative(next_row[index]);
                                next_row[index] = activated;
                                silu_derivative_row[index] = derivative;
                            }
                        });
                }
                #[cfg(feature = "parallel")]
                if rayon::current_num_threads() <= 1 || rows < e3nn_core::esen_parallel_min_nodes()
                {
                    serial_layer_norm_silu_rows(
                        rows,
                        layer.out_dim,
                        &linear_output,
                        layer_norm_weight,
                        layer_norm_bias,
                        self.layer_norm_eps,
                        &mut next,
                        &mut silu_derivatives,
                    );
                }
                #[cfg(not(feature = "parallel"))]
                serial_layer_norm_silu_rows(
                    rows,
                    layer.out_dim,
                    &linear_output,
                    layer_norm_weight,
                    layer_norm_bias,
                    self.layer_norm_eps,
                    &mut next,
                    &mut silu_derivatives,
                );
            }
            cache.layer_caches.push(EsenRadialMlpBatchLayerCache {
                linear_output,
                silu_derivative: Some(silu_derivatives),
            });
            current = Cow::Owned(next);
        }
        Ok(())
    }

    /// f32-native twin of [`Self::forward_batch_contiguous_cache`].
    ///
    /// Runs the MLP in f32, converting layer weights to f32 at call time.
    /// The cache is populated with f32-precision values cast to f64, which
    /// enables VJP callers to work at f32 fidelity.
    pub fn forward_batch_f32(
        &self,
        rows: usize,
        input: &[f32],
        output: &mut [f32],
        cache: &mut EsenRadialMlpBatchCache,
    ) -> Result<()> {
        check_len(
            "esen_radial_mlp_forward_batch_f32 input",
            rows * self.input_dim(),
            input.len(),
        )?;
        check_len(
            "esen_radial_mlp_forward_batch_f32 output",
            rows * self.output_dim(),
            output.len(),
        )?;
        cache.clear();
        cache.rows = rows;

        let mut current: Cow<'_, [f32]> = Cow::Borrowed(input);

        for (layer_index, layer) in self.layers.iter().enumerate() {
            if !layer.uses_layer_norm_silu() {
                let stage_start = esen_mlp_trace_start();
                cache.layer_caches.push(EsenRadialMlpBatchLayerCache {
                    linear_output: Vec::new(),
                    silu_derivative: None,
                });
                trace_esen_mlp_layer_stage(
                    "f32.radial.forward",
                    layer_index,
                    "cache_push",
                    stage_start,
                );

                let stage_start = esen_mlp_trace_start();
                dense_forward_rows_f32(
                    rows,
                    layer.in_dim,
                    layer.out_dim,
                    &layer.weight_f32,
                    &layer.weight_f32_in_by_out,
                    &layer.bias_f32,
                    current.as_ref(),
                    output,
                );
                trace_esen_mlp_layer_stage("f32.radial.forward", layer_index, "dense", stage_start);
                return Ok(());
            }

            let stage_start = esen_mlp_trace_start();
            let mut linear_output_f32 = uninit_f32_vec(rows * layer.out_dim);
            trace_esen_mlp_layer_stage(
                "f32.radial.forward",
                layer_index,
                "linear_alloc",
                stage_start,
            );

            let stage_start = esen_mlp_trace_start();
            dense_forward_rows_f32(
                rows,
                layer.in_dim,
                layer.out_dim,
                &layer.weight_f32,
                &layer.weight_f32_in_by_out,
                &layer.bias_f32,
                current.as_ref(),
                &mut linear_output_f32,
            );
            trace_esen_mlp_layer_stage("f32.radial.forward", layer_index, "dense", stage_start);

            let (Some(layer_norm_weight), Some(layer_norm_bias)) =
                (&layer.layer_norm_weight_f32, &layer.layer_norm_bias_f32)
            else {
                return Err(invalid_mlp(
                    "eSEN radial MLP f32 hidden layer is missing LayerNorm/SiLU cache inputs",
                ));
            };
            let eps_f32 = self.layer_norm_eps as f32;

            let stage_start = esen_mlp_trace_start();
            let mut next = uninit_f32_vec(linear_output_f32.len());
            let mut silu_derivatives_f32 = uninit_f32_vec(linear_output_f32.len());
            trace_esen_mlp_layer_stage(
                "f32.radial.forward",
                layer_index,
                "activation_alloc",
                stage_start,
            );

            let stage_start = esen_mlp_trace_start();
            if !layer_norm_silu_sve_rows_f32(
                rows,
                layer.out_dim,
                &linear_output_f32,
                layer_norm_weight,
                layer_norm_bias,
                eps_f32,
                &mut next,
                &mut silu_derivatives_f32,
            ) {
                #[cfg(feature = "parallel")]
                if rayon::current_num_threads() > 1 && rows >= e3nn_core::esen_parallel_min_nodes()
                {
                    use rayon::prelude::*;
                    linear_output_f32
                        .par_chunks(layer.out_dim)
                        .zip(next.par_chunks_mut(layer.out_dim))
                        .zip(silu_derivatives_f32.par_chunks_mut(layer.out_dim))
                        .for_each(|((linear_row, next_row), silu_derivative_row)| {
                            layer_norm_forward_into_f32(
                                linear_row,
                                layer_norm_weight,
                                layer_norm_bias,
                                eps_f32,
                                next_row,
                            );
                            silu_value_and_derivative_f32_in_place(next_row, silu_derivative_row);
                        });
                }
                #[cfg(feature = "parallel")]
                if rayon::current_num_threads() <= 1 || rows < e3nn_core::esen_parallel_min_nodes()
                {
                    for row in 0..rows {
                        let start = row * layer.out_dim;
                        let end = start + layer.out_dim;
                        layer_norm_forward_into_f32(
                            &linear_output_f32[start..end],
                            layer_norm_weight,
                            layer_norm_bias,
                            eps_f32,
                            &mut next[start..end],
                        );
                        silu_value_and_derivative_f32_in_place(
                            &mut next[start..end],
                            &mut silu_derivatives_f32[start..end],
                        );
                    }
                }
                #[cfg(not(feature = "parallel"))]
                for row in 0..rows {
                    let start = row * layer.out_dim;
                    let end = start + layer.out_dim;
                    layer_norm_forward_into_f32(
                        &linear_output_f32[start..end],
                        layer_norm_weight,
                        layer_norm_bias,
                        eps_f32,
                        &mut next[start..end],
                    );
                    silu_value_and_derivative_f32_in_place(
                        &mut next[start..end],
                        &mut silu_derivatives_f32[start..end],
                    );
                }
            }
            trace_esen_mlp_layer_stage("f32.radial.forward", layer_index, "norm_silu", stage_start);

            let stage_start = esen_mlp_trace_start();
            let linear_output_f64: Vec<f64> = linear_output_f32.iter().map(|&x| x as f64).collect();
            let silu_derivatives_f64: Vec<f64> =
                silu_derivatives_f32.iter().map(|&x| x as f64).collect();
            cache.layer_caches.push(EsenRadialMlpBatchLayerCache {
                linear_output: linear_output_f64,
                silu_derivative: Some(silu_derivatives_f64),
            });
            current = Cow::Owned(next);
            trace_esen_mlp_layer_stage(
                "f32.radial.forward",
                layer_index,
                "cache_cast_push",
                stage_start,
            );
        }
        // Fallback: if all layers had norm/silu (shouldn't happen per architecture),
        // copy current to output.
        let stage_start = esen_mlp_trace_start();
        for (dst, &src) in output.iter_mut().zip(current.iter()) {
            *dst = src;
        }
        trace_esen_mlp_stage("f32.radial.forward.output_copy", stage_start);
        Ok(())
    }

    /// f32-native twin of [`Self::forward_batch_f32`] that stores activations
    /// in an [`EsenRadialMlpBatchCacheF32`] (all-f32 storage) for use with
    /// [`Self::vjp_input_only_batch_contiguous_cache_f32`].
    ///
    /// The f64 cache path ([`EsenRadialMlpBatchCache`]) is not touched.
    pub fn forward_batch_f32_with_f32_cache(
        &self,
        rows: usize,
        input: &[f32],
        output: &mut [f32],
        cache: &mut EsenRadialMlpBatchCacheF32,
    ) -> Result<()> {
        check_len(
            "esen_radial_mlp_forward_batch_f32_with_f32_cache input",
            rows * self.input_dim(),
            input.len(),
        )?;
        check_len(
            "esen_radial_mlp_forward_batch_f32_with_f32_cache output",
            rows * self.output_dim(),
            output.len(),
        )?;
        cache.clear();
        cache.rows = rows;

        let mut current: Cow<'_, [f32]> = Cow::Borrowed(input);

        for (layer_index, layer) in self.layers.iter().enumerate() {
            if !layer.uses_layer_norm_silu() {
                let stage_start = esen_mlp_trace_start();
                cache.layer_caches.push(EsenRadialMlpBatchLayerCacheF32 {
                    linear_output: Vec::new(),
                    silu_derivative: None,
                });
                trace_esen_mlp_layer_stage(
                    "f32.radial.forward",
                    layer_index,
                    "cache_push",
                    stage_start,
                );

                let stage_start = esen_mlp_trace_start();
                dense_forward_rows_f32(
                    rows,
                    layer.in_dim,
                    layer.out_dim,
                    &layer.weight_f32,
                    &layer.weight_f32_in_by_out,
                    &layer.bias_f32,
                    current.as_ref(),
                    output,
                );
                trace_esen_mlp_layer_stage("f32.radial.forward", layer_index, "dense", stage_start);
                return Ok(());
            }

            let stage_start = esen_mlp_trace_start();
            let mut linear_output_f32 = uninit_f32_vec(rows * layer.out_dim);
            trace_esen_mlp_layer_stage(
                "f32.radial.forward",
                layer_index,
                "linear_alloc",
                stage_start,
            );

            let stage_start = esen_mlp_trace_start();
            dense_forward_rows_f32(
                rows,
                layer.in_dim,
                layer.out_dim,
                &layer.weight_f32,
                &layer.weight_f32_in_by_out,
                &layer.bias_f32,
                current.as_ref(),
                &mut linear_output_f32,
            );
            trace_esen_mlp_layer_stage("f32.radial.forward", layer_index, "dense", stage_start);

            let (Some(layer_norm_weight), Some(layer_norm_bias)) =
                (&layer.layer_norm_weight_f32, &layer.layer_norm_bias_f32)
            else {
                return Err(invalid_mlp(
                    "eSEN radial MLP f32 hidden layer is missing LayerNorm/SiLU cache inputs",
                ));
            };
            let eps_f32 = self.layer_norm_eps as f32;

            let stage_start = esen_mlp_trace_start();
            let mut next = uninit_f32_vec(linear_output_f32.len());
            let mut silu_derivatives_f32 = uninit_f32_vec(linear_output_f32.len());
            trace_esen_mlp_layer_stage(
                "f32.radial.forward",
                layer_index,
                "activation_alloc",
                stage_start,
            );

            let stage_start = esen_mlp_trace_start();
            if !layer_norm_silu_sve_rows_f32(
                rows,
                layer.out_dim,
                &linear_output_f32,
                layer_norm_weight,
                layer_norm_bias,
                eps_f32,
                &mut next,
                &mut silu_derivatives_f32,
            ) {
                #[cfg(feature = "parallel")]
                if rayon::current_num_threads() > 1 && rows >= e3nn_core::esen_parallel_min_nodes()
                {
                    use rayon::prelude::*;
                    linear_output_f32
                        .par_chunks(layer.out_dim)
                        .zip(next.par_chunks_mut(layer.out_dim))
                        .zip(silu_derivatives_f32.par_chunks_mut(layer.out_dim))
                        .for_each(|((linear_row, next_row), silu_derivative_row)| {
                            layer_norm_forward_into_f32(
                                linear_row,
                                layer_norm_weight,
                                layer_norm_bias,
                                eps_f32,
                                next_row,
                            );
                            silu_value_and_derivative_f32_in_place(next_row, silu_derivative_row);
                        });
                }
                #[cfg(feature = "parallel")]
                if rayon::current_num_threads() <= 1 || rows < e3nn_core::esen_parallel_min_nodes()
                {
                    for row in 0..rows {
                        let start = row * layer.out_dim;
                        let end = start + layer.out_dim;
                        layer_norm_forward_into_f32(
                            &linear_output_f32[start..end],
                            layer_norm_weight,
                            layer_norm_bias,
                            eps_f32,
                            &mut next[start..end],
                        );
                        silu_value_and_derivative_f32_in_place(
                            &mut next[start..end],
                            &mut silu_derivatives_f32[start..end],
                        );
                    }
                }
                #[cfg(not(feature = "parallel"))]
                for row in 0..rows {
                    let start = row * layer.out_dim;
                    let end = start + layer.out_dim;
                    layer_norm_forward_into_f32(
                        &linear_output_f32[start..end],
                        layer_norm_weight,
                        layer_norm_bias,
                        eps_f32,
                        &mut next[start..end],
                    );
                    silu_value_and_derivative_f32_in_place(
                        &mut next[start..end],
                        &mut silu_derivatives_f32[start..end],
                    );
                }
            }
            trace_esen_mlp_layer_stage("f32.radial.forward", layer_index, "norm_silu", stage_start);

            let stage_start = esen_mlp_trace_start();
            cache.layer_caches.push(EsenRadialMlpBatchLayerCacheF32 {
                linear_output: linear_output_f32,
                silu_derivative: Some(silu_derivatives_f32),
            });
            current = Cow::Owned(next);
            trace_esen_mlp_layer_stage(
                "f32.radial.forward",
                layer_index,
                "cache_push",
                stage_start,
            );
        }
        // Fallback: if all layers had norm/silu (shouldn't happen per architecture).
        let stage_start = esen_mlp_trace_start();
        for (dst, &src) in output.iter_mut().zip(current.iter()) {
            *dst = src;
        }
        trace_esen_mlp_stage("f32.radial.forward.output_copy", stage_start);
        Ok(())
    }

    pub fn forward_batch_final_linear_input_contiguous_cache(
        &self,
        rows: usize,
        input: &[f64],
        final_linear_input: &mut [f64],
        cache: &mut EsenRadialMlpBatchCache,
    ) -> Result<()> {
        check_len(
            "esen_radial_mlp_forward_final_linear_input input",
            rows * self.input_dim(),
            input.len(),
        )?;
        check_len(
            "esen_radial_mlp_forward_final_linear_input final input",
            rows * self.final_linear_input_dim(),
            final_linear_input.len(),
        )?;
        cache.clear();
        cache.rows = rows;

        let mut current: Cow<'_, [f64]> = Cow::Borrowed(input);
        for layer in &self.layers {
            if !layer.uses_layer_norm_silu() {
                final_linear_input.copy_from_slice(current.as_ref());
                cache.layer_caches.push(EsenRadialMlpBatchLayerCache {
                    linear_output: Vec::new(),
                    silu_derivative: None,
                });
                return Ok(());
            }

            let mut linear_output = vec![0.0; rows * layer.out_dim];
            dense_forward_rows(layer, rows, current.as_ref(), &mut linear_output)?;
            let (Some(layer_norm_weight), Some(layer_norm_bias)) =
                (&layer.layer_norm_weight, &layer.layer_norm_bias)
            else {
                return Err(invalid_mlp(
                    "FairChem RadialMLP hidden layer is missing LayerNorm/SiLU cache inputs",
                ));
            };
            let mut next = vec![0.0; linear_output.len()];
            let mut silu_derivatives = vec![0.0; linear_output.len()];

            #[cfg(feature = "parallel")]
            if rayon::current_num_threads() > 1 && rows >= e3nn_core::esen_parallel_min_nodes() {
                linear_output
                    .par_chunks(layer.out_dim)
                    .zip(next.par_chunks_mut(layer.out_dim))
                    .zip(silu_derivatives.par_chunks_mut(layer.out_dim))
                    .for_each(|((linear_row, next_row), silu_derivative_row)| {
                        layer_norm_forward_into(
                            linear_row,
                            layer_norm_weight,
                            layer_norm_bias,
                            self.layer_norm_eps,
                            next_row,
                        );
                        for index in 0..layer.out_dim {
                            let (activated, derivative) =
                                silu_value_and_derivative(next_row[index]);
                            next_row[index] = activated;
                            silu_derivative_row[index] = derivative;
                        }
                    });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for row in 0..rows {
                    let start = row * layer.out_dim;
                    let end = start + layer.out_dim;
                    layer_norm_forward_into(
                        &linear_output[start..end],
                        layer_norm_weight,
                        layer_norm_bias,
                        self.layer_norm_eps,
                        &mut next[start..end],
                    );
                    for index in start..end {
                        let (activated, derivative) = silu_value_and_derivative(next[index]);
                        next[index] = activated;
                        silu_derivatives[index] = derivative;
                    }
                }
            }
            #[cfg(feature = "parallel")]
            if rayon::current_num_threads() <= 1 || rows < e3nn_core::esen_parallel_min_nodes() {
                for row in 0..rows {
                    let start = row * layer.out_dim;
                    let end = start + layer.out_dim;
                    layer_norm_forward_into(
                        &linear_output[start..end],
                        layer_norm_weight,
                        layer_norm_bias,
                        self.layer_norm_eps,
                        &mut next[start..end],
                    );
                    for index in start..end {
                        let (activated, derivative) = silu_value_and_derivative(next[index]);
                        next[index] = activated;
                        silu_derivatives[index] = derivative;
                    }
                }
            }
            cache.layer_caches.push(EsenRadialMlpBatchLayerCache {
                linear_output,
                silu_derivative: Some(silu_derivatives),
            });
            current = Cow::Owned(next);
        }
        Ok(())
    }

    pub fn final_linear_forward_output_block(
        &self,
        rows: usize,
        final_linear_input: &[f64],
        output_offset: usize,
        output_len: usize,
        out_block: &mut [f64],
    ) -> Result<()> {
        let final_layer = &self.layers[self.layers.len() - 1];
        if output_offset
            .checked_add(output_len)
            .is_none_or(|end| end > final_layer.out_dim)
        {
            return Err(invalid_mlp(format!(
                "eSEN radial MLP final output block {output_offset}..{} is out of range for output dim {}",
                output_offset.saturating_add(output_len),
                final_layer.out_dim
            )));
        }
        check_len(
            "esen_radial_mlp_final_linear_forward final input",
            rows * final_layer.in_dim,
            final_linear_input.len(),
        )?;
        check_len(
            "esen_radial_mlp_final_linear_forward output block",
            rows * output_len,
            out_block.len(),
        )?;
        dense_forward_rows_with_weight_bias(
            rows,
            output_len,
            final_layer.in_dim,
            &final_layer.weight[output_offset * final_layer.in_dim
                ..(output_offset + output_len) * final_layer.in_dim],
            &final_layer.bias[output_offset..output_offset + output_len],
            final_linear_input,
            out_block,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn final_linear_forward_output_block_strided(
        &self,
        rows: usize,
        final_linear_input: &[f64],
        output_offset: usize,
        output_len: usize,
        out_row_stride: usize,
        out_feature_offset: usize,
        out: &mut [f64],
    ) -> Result<()> {
        let final_layer = &self.layers[self.layers.len() - 1];
        if output_offset
            .checked_add(output_len)
            .is_none_or(|end| end > final_layer.out_dim)
        {
            return Err(invalid_mlp(format!(
                "eSEN radial MLP final output block {output_offset}..{} is out of range for output dim {}",
                output_offset.saturating_add(output_len),
                final_layer.out_dim
            )));
        }
        if out_feature_offset
            .checked_add(output_len)
            .is_none_or(|end| end > out_row_stride)
        {
            return Err(invalid_mlp(format!(
                "eSEN radial MLP strided output block {out_feature_offset}..{} exceeds row stride {out_row_stride}",
                out_feature_offset.saturating_add(output_len)
            )));
        }
        check_len(
            "esen_radial_mlp_final_linear_forward_strided final input",
            rows * final_layer.in_dim,
            final_linear_input.len(),
        )?;
        let required_out_len = if rows == 0 {
            0
        } else {
            (rows - 1)
                .checked_mul(out_row_stride)
                .and_then(|start| start.checked_add(out_feature_offset))
                .and_then(|start| start.checked_add(output_len))
                .ok_or_else(|| invalid_mlp("eSEN radial MLP strided output length overflow"))?
        };
        if out.len() < required_out_len {
            return Err(invalid_mlp(format!(
                "esen_radial_mlp_final_linear_forward_strided output length must be at least {required_out_len}, got {}",
                out.len()
            )));
        }
        dense_forward_rows_with_weight_bias_strided_output(
            rows,
            output_len,
            final_layer.in_dim,
            &final_layer.weight[output_offset * final_layer.in_dim
                ..(output_offset + output_len) * final_layer.in_dim],
            &final_layer.bias[output_offset..output_offset + output_len],
            final_linear_input,
            out_row_stride,
            out_feature_offset,
            out,
        )
    }

    pub fn vjp_input_only(
        &self,
        cache: &EsenRadialMlpCache,
        grad_out: &[f64],
        grad_input: &mut [f64],
    ) -> Result<()> {
        check_len(
            "esen_radial_mlp_vjp_input_only grad_out",
            self.output_dim(),
            grad_out.len(),
        )?;
        check_len(
            "esen_radial_mlp_vjp_input_only grad_input",
            self.input_dim(),
            grad_input.len(),
        )?;
        if cache.layer_caches.len() != self.layers.len() {
            return Err(invalid_mlp(format!(
                "eSEN radial MLP cache layer count mismatch: expected {}, got {}",
                self.layers.len(),
                cache.layer_caches.len()
            )));
        }

        let mut grad_current: Cow<'_, [f64]> = Cow::Borrowed(grad_out);
        for layer_index in (0..self.layers.len()).rev() {
            let layer = &self.layers[layer_index];
            let layer_cache = &cache.layer_caches[layer_index];
            let grad_linear: Cow<'_, [f64]> =
                if let (Some(layer_norm_weight), Some(silu_derivative)) =
                    (&layer.layer_norm_weight, &layer_cache.silu_derivative)
                {
                    let mut values = vec![0.0; layer.out_dim];
                    layer_norm_silu_vjp_input_into(
                        &layer_cache.linear_output,
                        layer_norm_weight,
                        grad_current.as_ref(),
                        silu_derivative,
                        self.layer_norm_eps,
                        &mut values,
                    );
                    Cow::Owned(values)
                } else {
                    grad_current
                };
            grad_current = Cow::Owned(dense_vjp_input(layer, grad_linear.as_ref()));
        }
        for (target, value) in grad_input.iter_mut().zip(grad_current.iter()) {
            *target += value;
        }
        Ok(())
    }

    pub fn vjp_input_only_batch(
        &self,
        caches: &[EsenRadialMlpCache],
        rows: usize,
        grad_out: &[f64],
        grad_input: &mut [f64],
    ) -> Result<()> {
        check_len(
            "esen_radial_mlp_vjp_input_only_batch cache",
            rows,
            caches.len(),
        )?;
        check_len(
            "esen_radial_mlp_vjp_input_only_batch grad_out",
            rows * self.output_dim(),
            grad_out.len(),
        )?;
        check_len(
            "esen_radial_mlp_vjp_input_only_batch grad_input",
            rows * self.input_dim(),
            grad_input.len(),
        )?;
        for cache in caches {
            if cache.layer_caches.len() != self.layers.len() {
                return Err(invalid_mlp(format!(
                    "eSEN radial MLP cache layer count mismatch: expected {}, got {}",
                    self.layers.len(),
                    cache.layer_caches.len()
                )));
            }
        }

        let mut grad_current: Cow<'_, [f64]> = Cow::Borrowed(grad_out);
        for layer_index in (0..self.layers.len()).rev() {
            let layer = &self.layers[layer_index];
            let grad_linear: Cow<'_, [f64]> = if let Some(layer_norm_weight) =
                &layer.layer_norm_weight
            {
                let mut values = vec![0.0; rows * layer.out_dim];
                #[cfg(feature = "parallel")]
                if rayon::current_num_threads() > 1 && rows >= e3nn_core::esen_parallel_min_nodes()
                {
                    values
                        .par_chunks_mut(layer.out_dim)
                        .enumerate()
                        .try_for_each(|(row, values_row)| -> Result<()> {
                            let layer_cache = &caches[row].layer_caches[layer_index];
                            let Some(silu_derivative) = &layer_cache.silu_derivative else {
                                return Err(invalid_mlp(
                                    "eSEN radial MLP missing SiLU derivative cache",
                                ));
                            };
                            let start = row * layer.out_dim;
                            layer_norm_silu_vjp_input_into(
                                &layer_cache.linear_output,
                                layer_norm_weight,
                                &grad_current.as_ref()[start..start + layer.out_dim],
                                silu_derivative,
                                self.layer_norm_eps,
                                values_row,
                            );
                            Ok(())
                        })?;
                }
                #[cfg(not(feature = "parallel"))]
                {
                    for (row, cache) in caches.iter().enumerate().take(rows) {
                        let layer_cache = &cache.layer_caches[layer_index];
                        let Some(silu_derivative) = &layer_cache.silu_derivative else {
                            return Err(invalid_mlp(
                                "eSEN radial MLP missing SiLU derivative cache",
                            ));
                        };
                        let start = row * layer.out_dim;
                        let end = start + layer.out_dim;
                        layer_norm_silu_vjp_input_into(
                            &layer_cache.linear_output,
                            layer_norm_weight,
                            &grad_current.as_ref()[start..end],
                            silu_derivative,
                            self.layer_norm_eps,
                            &mut values[start..end],
                        );
                    }
                }
                #[cfg(feature = "parallel")]
                if rayon::current_num_threads() <= 1 || rows < e3nn_core::esen_parallel_min_nodes()
                {
                    for (row, cache) in caches.iter().enumerate().take(rows) {
                        let layer_cache = &cache.layer_caches[layer_index];
                        let Some(silu_derivative) = &layer_cache.silu_derivative else {
                            return Err(invalid_mlp(
                                "eSEN radial MLP missing SiLU derivative cache",
                            ));
                        };
                        let start = row * layer.out_dim;
                        let end = start + layer.out_dim;
                        layer_norm_silu_vjp_input_into(
                            &layer_cache.linear_output,
                            layer_norm_weight,
                            &grad_current.as_ref()[start..end],
                            silu_derivative,
                            self.layer_norm_eps,
                            &mut values[start..end],
                        );
                    }
                }
                Cow::Owned(values)
            } else {
                grad_current
            };
            let mut next_grad = vec![0.0; rows * layer.in_dim];
            dense_vjp_input_rows(layer, rows, grad_linear.as_ref(), &mut next_grad)?;
            grad_current = Cow::Owned(next_grad);
        }
        for (target, value) in grad_input.iter_mut().zip(grad_current.iter()) {
            *target += value;
        }
        Ok(())
    }

    pub fn vjp_input_only_batch_contiguous_cache(
        &self,
        cache: &EsenRadialMlpBatchCache,
        rows: usize,
        grad_out: &[f64],
        grad_input: &mut [f64],
    ) -> Result<()> {
        if cache.rows != rows {
            return Err(invalid_mlp(format!(
                "eSEN radial MLP batch cache row count mismatch: expected {rows}, got {}",
                cache.rows
            )));
        }
        check_len(
            "esen_radial_mlp_vjp_input_only_batch_contiguous_cache layer cache",
            self.layers.len(),
            cache.layer_caches.len(),
        )?;
        check_len(
            "esen_radial_mlp_vjp_input_only_batch_contiguous_cache grad_out",
            rows * self.output_dim(),
            grad_out.len(),
        )?;
        check_len(
            "esen_radial_mlp_vjp_input_only_batch_contiguous_cache grad_input",
            rows * self.input_dim(),
            grad_input.len(),
        )?;
        for (layer, layer_cache) in self.layers.iter().zip(cache.layer_caches.iter()) {
            if layer.uses_layer_norm_silu() {
                check_len(
                    "eSEN radial MLP batch layer linear cache",
                    rows * layer.out_dim,
                    layer_cache.linear_output.len(),
                )?;
                let Some(silu_derivative) = &layer_cache.silu_derivative else {
                    return Err(invalid_mlp("eSEN radial MLP missing SiLU derivative cache"));
                };
                check_len(
                    "eSEN radial MLP batch layer SiLU derivative cache",
                    rows * layer.out_dim,
                    silu_derivative.len(),
                )?;
            }
        }

        let mut grad_current: Cow<'_, [f64]> = Cow::Borrowed(grad_out);
        for layer_index in (0..self.layers.len()).rev() {
            let layer = &self.layers[layer_index];
            let layer_cache = &cache.layer_caches[layer_index];
            let grad_linear: Cow<'_, [f64]> =
                if let Some(layer_norm_weight) = &layer.layer_norm_weight {
                    let Some(silu_derivative) = &layer_cache.silu_derivative else {
                        return Err(invalid_mlp("eSEN radial MLP missing SiLU derivative cache"));
                    };
                    let mut values = vec![0.0; rows * layer.out_dim];
                    // Fast path: vectorized f64 LayerNorm+SiLU VJP (ArmPL SVE
                    // `exp`). Falls back to the scalar path when unavailable.
                    if !layer_norm_silu_vjp_sve_rows_f64(
                        rows,
                        layer.out_dim,
                        &layer_cache.linear_output,
                        layer_norm_weight,
                        grad_current.as_ref(),
                        silu_derivative,
                        self.layer_norm_eps,
                        &mut values,
                    ) {
                        #[cfg(feature = "parallel")]
                        if rayon::current_num_threads() > 1
                            && rows >= e3nn_core::esen_parallel_min_nodes()
                        {
                            values
                                .par_chunks_mut(layer.out_dim)
                                .zip(layer_cache.linear_output.par_chunks(layer.out_dim))
                                .zip(silu_derivative.par_chunks(layer.out_dim))
                                .zip(grad_current.as_ref().par_chunks(layer.out_dim))
                                .for_each(
                                    |(
                                        ((values_row, linear_row), silu_derivative_row),
                                        grad_current_row,
                                    )| {
                                        layer_norm_silu_vjp_input_into(
                                            linear_row,
                                            layer_norm_weight,
                                            grad_current_row,
                                            silu_derivative_row,
                                            self.layer_norm_eps,
                                            values_row,
                                        );
                                    },
                                );
                        }
                        #[cfg(feature = "parallel")]
                        if rayon::current_num_threads() <= 1
                            || rows < e3nn_core::esen_parallel_min_nodes()
                        {
                            serial_layer_norm_silu_vjp_rows(
                                rows,
                                layer.out_dim,
                                &layer_cache.linear_output,
                                layer_norm_weight,
                                grad_current.as_ref(),
                                silu_derivative,
                                self.layer_norm_eps,
                                &mut values,
                            );
                        }
                        #[cfg(not(feature = "parallel"))]
                        serial_layer_norm_silu_vjp_rows(
                            rows,
                            layer.out_dim,
                            &layer_cache.linear_output,
                            layer_norm_weight,
                            grad_current.as_ref(),
                            silu_derivative,
                            self.layer_norm_eps,
                            &mut values,
                        );
                    }
                    Cow::Owned(values)
                } else {
                    grad_current
                };
            let mut next_grad = vec![0.0; rows * layer.in_dim];
            dense_vjp_input_rows(layer, rows, grad_linear.as_ref(), &mut next_grad)?;
            grad_current = Cow::Owned(next_grad);
        }
        for (target, value) in grad_input.iter_mut().zip(grad_current.iter()) {
            *target += value;
        }
        Ok(())
    }

    /// f32-native VJP: analogous to [`Self::vjp_input_only_batch_contiguous_cache`]
    /// but uses [`EsenRadialMlpBatchCacheF32`] (all-f32 activations) and computes
    /// the entire backward in f32 via sgemm / pure-Rust f32 paths.
    ///
    /// The f64 VJP path is completely unaffected.
    pub fn vjp_input_only_batch_contiguous_cache_f32(
        &self,
        cache: &EsenRadialMlpBatchCacheF32,
        rows: usize,
        grad_out: &[f32],
        grad_input: &mut [f32],
    ) -> Result<()> {
        let grad_current =
            self.vjp_input_only_batch_contiguous_cache_f32_values(cache, rows, grad_out)?;
        check_len(
            "esen_radial_mlp_vjp_f32 grad_input",
            rows * self.input_dim(),
            grad_input.len(),
        )?;
        for (target, value) in grad_input.iter_mut().zip(grad_current.iter()) {
            *target += value;
        }
        Ok(())
    }

    /// f32-native VJP that accumulates the final f32 input gradient directly
    /// into an f64 destination. This keeps the f32 math path unchanged while
    /// avoiding an intermediate f32 output buffer at UMA force-accumulation
    /// call sites.
    pub fn vjp_input_only_batch_contiguous_cache_f32_accumulate_f64(
        &self,
        cache: &EsenRadialMlpBatchCacheF32,
        rows: usize,
        grad_out: &[f32],
        grad_input: &mut [f64],
    ) -> Result<()> {
        let grad_current =
            self.vjp_input_only_batch_contiguous_cache_f32_values(cache, rows, grad_out)?;
        check_len(
            "esen_radial_mlp_vjp_f32_accumulate_f64 grad_input",
            rows * self.input_dim(),
            grad_input.len(),
        )?;
        for (target, value) in grad_input.iter_mut().zip(grad_current.iter()) {
            *target += f64::from(*value);
        }
        Ok(())
    }

    fn vjp_input_only_batch_contiguous_cache_f32_values(
        &self,
        cache: &EsenRadialMlpBatchCacheF32,
        rows: usize,
        grad_out: &[f32],
    ) -> Result<Vec<f32>> {
        if cache.rows != rows {
            return Err(invalid_mlp(format!(
                "eSEN radial MLP f32 batch cache row count mismatch: expected {rows}, got {}",
                cache.rows
            )));
        }
        check_len(
            "esen_radial_mlp_vjp_f32 layer cache",
            self.layers.len(),
            cache.layer_caches.len(),
        )?;
        check_len(
            "esen_radial_mlp_vjp_f32 grad_out",
            rows * self.output_dim(),
            grad_out.len(),
        )?;
        // Validate cache shape up front.
        for (layer, layer_cache) in self.layers.iter().zip(cache.layer_caches.iter()) {
            if layer.uses_layer_norm_silu() {
                check_len(
                    "eSEN radial MLP f32 batch layer linear cache",
                    rows * layer.out_dim,
                    layer_cache.linear_output.len(),
                )?;
                let Some(silu_derivative) = &layer_cache.silu_derivative else {
                    return Err(invalid_mlp(
                        "eSEN radial MLP f32 missing SiLU derivative cache",
                    ));
                };
                check_len(
                    "eSEN radial MLP f32 batch layer SiLU derivative cache",
                    rows * layer.out_dim,
                    silu_derivative.len(),
                )?;
            }
        }

        let mut grad_current: Cow<'_, [f32]> = Cow::Borrowed(grad_out);

        for layer_index in (0..self.layers.len()).rev() {
            let layer = &self.layers[layer_index];
            let layer_cache = &cache.layer_caches[layer_index];
            // LayerNorm+SiLU backward (hidden layers only).
            let grad_linear: Cow<'_, [f32]> = if let Some(layer_norm_weight) =
                &layer.layer_norm_weight_f32
            {
                let Some(silu_derivative) = &layer_cache.silu_derivative else {
                    return Err(invalid_mlp(
                        "eSEN radial MLP f32 missing SiLU derivative cache",
                    ));
                };
                let eps_f32 = self.layer_norm_eps as f32;
                let stage_start = esen_mlp_trace_start();
                let mut values = uninit_f32_vec(rows * layer.out_dim);
                trace_esen_mlp_layer_stage(
                    "f32.radial.vjp",
                    layer_index,
                    "norm_silu_alloc",
                    stage_start,
                );

                let stage_start = esen_mlp_trace_start();
                if !layer_norm_silu_vjp_sve_rows_f32(
                    rows,
                    layer.out_dim,
                    &layer_cache.linear_output,
                    layer_norm_weight,
                    grad_current.as_ref(),
                    silu_derivative,
                    eps_f32,
                    &mut values,
                ) {
                    #[cfg(feature = "parallel")]
                    if rayon::current_num_threads() > 1
                        && rows >= e3nn_core::esen_parallel_min_nodes()
                    {
                        values
                            .par_chunks_mut(layer.out_dim)
                            .zip(layer_cache.linear_output.par_chunks(layer.out_dim))
                            .zip(silu_derivative.par_chunks(layer.out_dim))
                            .zip(grad_current.as_ref().par_chunks(layer.out_dim))
                            .for_each(
                                |(((values_row, linear_row), silu_derivative_row), grad_row)| {
                                    layer_norm_silu_vjp_input_into_f32(
                                        linear_row,
                                        layer_norm_weight,
                                        grad_row,
                                        silu_derivative_row,
                                        eps_f32,
                                        values_row,
                                    );
                                },
                            );
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        for row in 0..rows {
                            let start = row * layer.out_dim;
                            let end = start + layer.out_dim;
                            layer_norm_silu_vjp_input_into_f32(
                                &layer_cache.linear_output[start..end],
                                layer_norm_weight,
                                &grad_current.as_ref()[start..end],
                                &silu_derivative[start..end],
                                eps_f32,
                                &mut values[start..end],
                            );
                        }
                    }
                    #[cfg(feature = "parallel")]
                    if rayon::current_num_threads() <= 1
                        || rows < e3nn_core::esen_parallel_min_nodes()
                    {
                        for row in 0..rows {
                            let start = row * layer.out_dim;
                            let end = start + layer.out_dim;
                            layer_norm_silu_vjp_input_into_f32(
                                &layer_cache.linear_output[start..end],
                                layer_norm_weight,
                                &grad_current.as_ref()[start..end],
                                &silu_derivative[start..end],
                                eps_f32,
                                &mut values[start..end],
                            );
                        }
                    }
                }
                trace_esen_mlp_layer_stage("f32.radial.vjp", layer_index, "norm_silu", stage_start);
                Cow::Owned(values)
            } else {
                grad_current
            };

            // Linear layer backward: grad_input = grad_linear @ weight  (weight is [out_dim, in_dim])
            let stage_start = esen_mlp_trace_start();
            let mut next_grad = uninit_f32_vec(rows * layer.in_dim);
            trace_esen_mlp_layer_stage("f32.radial.vjp", layer_index, "dense_alloc", stage_start);

            let stage_start = esen_mlp_trace_start();
            dense_vjp_input_rows_f32(layer, rows, grad_linear.as_ref(), &mut next_grad);
            trace_esen_mlp_layer_stage("f32.radial.vjp", layer_index, "dense", stage_start);
            grad_current = Cow::Owned(next_grad);
        }
        Ok(grad_current.into_owned())
    }

    pub fn accumulate_final_linear_input_vjp_from_output_block(
        &self,
        rows: usize,
        output_offset: usize,
        output_len: usize,
        grad_output_block: &[f64],
        grad_final_linear_input: &mut [f64],
    ) -> Result<()> {
        let final_layer = &self.layers[self.layers.len() - 1];
        if output_offset
            .checked_add(output_len)
            .is_none_or(|end| end > final_layer.out_dim)
        {
            return Err(invalid_mlp(format!(
                "eSEN radial MLP final output block {output_offset}..{} is out of range for output dim {}",
                output_offset.saturating_add(output_len),
                final_layer.out_dim
            )));
        }
        check_len(
            "esen_radial_mlp_final_linear_vjp grad_output_block",
            rows * output_len,
            grad_output_block.len(),
        )?;
        check_len(
            "esen_radial_mlp_final_linear_vjp grad_final_linear_input",
            rows * final_layer.in_dim,
            grad_final_linear_input.len(),
        )?;
        dense_vjp_input_rows_add_with_weight(
            rows,
            output_len,
            final_layer.in_dim,
            &final_layer.weight[output_offset * final_layer.in_dim
                ..(output_offset + output_len) * final_layer.in_dim],
            grad_output_block,
            grad_final_linear_input,
        )
    }

    pub fn vjp_input_only_batch_contiguous_cache_from_final_linear_input_grad(
        &self,
        cache: &EsenRadialMlpBatchCache,
        rows: usize,
        grad_final_linear_input: &[f64],
        grad_input: &mut [f64],
    ) -> Result<()> {
        if self.layers.len() == 1 {
            check_len(
                "esen_radial_mlp_vjp_from_final_linear_input grad_final_linear_input",
                rows * self.input_dim(),
                grad_final_linear_input.len(),
            )?;
            check_len(
                "esen_radial_mlp_vjp_from_final_linear_input grad_input",
                rows * self.input_dim(),
                grad_input.len(),
            )?;
            for (target, value) in grad_input.iter_mut().zip(grad_final_linear_input.iter()) {
                *target += value;
            }
            return Ok(());
        }

        if cache.rows != rows {
            return Err(invalid_mlp(format!(
                "eSEN radial MLP batch cache row count mismatch: expected {rows}, got {}",
                cache.rows
            )));
        }
        check_len(
            "esen_radial_mlp_vjp_from_final_linear_input layer cache",
            self.layers.len(),
            cache.layer_caches.len(),
        )?;
        check_len(
            "esen_radial_mlp_vjp_from_final_linear_input grad_final_linear_input",
            rows * self.final_linear_input_dim(),
            grad_final_linear_input.len(),
        )?;
        check_len(
            "esen_radial_mlp_vjp_from_final_linear_input grad_input",
            rows * self.input_dim(),
            grad_input.len(),
        )?;

        let mut grad_current: Cow<'_, [f64]> = Cow::Borrowed(grad_final_linear_input);
        for layer_index in (0..self.layers.len() - 1).rev() {
            let layer = &self.layers[layer_index];
            let layer_cache = &cache.layer_caches[layer_index];
            let Some(layer_norm_weight) = &layer.layer_norm_weight else {
                return Err(invalid_mlp(
                    "eSEN radial MLP hidden layer is missing LayerNorm weight",
                ));
            };
            check_len(
                "eSEN radial MLP batch layer linear cache",
                rows * layer.out_dim,
                layer_cache.linear_output.len(),
            )?;
            let Some(silu_derivative) = &layer_cache.silu_derivative else {
                return Err(invalid_mlp("eSEN radial MLP missing SiLU derivative cache"));
            };
            check_len(
                "eSEN radial MLP batch layer SiLU derivative cache",
                rows * layer.out_dim,
                silu_derivative.len(),
            )?;

            let mut grad_linear = vec![0.0; rows * layer.out_dim];
            #[cfg(feature = "parallel")]
            if rayon::current_num_threads() > 1 && rows >= e3nn_core::esen_parallel_min_nodes() {
                grad_linear
                    .par_chunks_mut(layer.out_dim)
                    .zip(layer_cache.linear_output.par_chunks(layer.out_dim))
                    .zip(silu_derivative.par_chunks(layer.out_dim))
                    .zip(grad_current.as_ref().par_chunks(layer.out_dim))
                    .for_each(
                        |(((grad_linear_row, linear_row), silu_derivative_row), grad_row)| {
                            layer_norm_silu_vjp_input_into(
                                linear_row,
                                layer_norm_weight,
                                grad_row,
                                silu_derivative_row,
                                self.layer_norm_eps,
                                grad_linear_row,
                            );
                        },
                    );
            }
            #[cfg(not(feature = "parallel"))]
            {
                for row in 0..rows {
                    let start = row * layer.out_dim;
                    let end = start + layer.out_dim;
                    layer_norm_silu_vjp_input_into(
                        &layer_cache.linear_output[start..end],
                        layer_norm_weight,
                        &grad_current.as_ref()[start..end],
                        &silu_derivative[start..end],
                        self.layer_norm_eps,
                        &mut grad_linear[start..end],
                    );
                }
            }
            #[cfg(feature = "parallel")]
            if rayon::current_num_threads() <= 1 || rows < e3nn_core::esen_parallel_min_nodes() {
                for row in 0..rows {
                    let start = row * layer.out_dim;
                    let end = start + layer.out_dim;
                    layer_norm_silu_vjp_input_into(
                        &layer_cache.linear_output[start..end],
                        layer_norm_weight,
                        &grad_current.as_ref()[start..end],
                        &silu_derivative[start..end],
                        self.layer_norm_eps,
                        &mut grad_linear[start..end],
                    );
                }
            }

            let mut next_grad = vec![0.0; rows * layer.in_dim];
            dense_vjp_input_rows(layer, rows, &grad_linear, &mut next_grad)?;
            grad_current = Cow::Owned(next_grad);
        }

        for (target, value) in grad_input.iter_mut().zip(grad_current.iter()) {
            *target += value;
        }
        Ok(())
    }
}

fn silu_value_and_derivative_f32(value: f32) -> (f32, f32) {
    let sigmoid = if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    };
    (value * sigmoid, sigmoid * (1.0 + value * (1.0 - sigmoid)))
}

#[cfg(all(
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
))]
#[inline(always)]
fn esen_mlp_f32_sve_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("LLMLIP_DISABLE_SIMD").is_none()
            && std::env::var_os("LLMLIP_DISABLE_SVE").is_none()
            && matches!(
                std::env::var("LLMLIP_ESEN_MLP_ENABLE_SVE_SILU")
                    .ok()
                    .as_deref(),
                Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
            )
            && std::arch::is_aarch64_feature_detected!("sve")
    })
}

#[cfg(all(
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
))]
#[inline(always)]
fn esen_mlp_f32_sve_layer_norm_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("LLMLIP_DISABLE_SIMD").is_none()
            && std::env::var_os("LLMLIP_DISABLE_SVE").is_none()
            && matches!(
                std::env::var("LLMLIP_ESEN_MLP_ENABLE_SVE_LAYER_NORM")
                    .ok()
                    .as_deref(),
                Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
            )
            && std::arch::is_aarch64_feature_detected!("sve")
    })
}

/// f64 LayerNorm+SiLU SVE kernel gate. Unlike the f32 SVE silu/layer-norm
/// kernels (which are opt-in), the f64 kernels are enabled by default: the f64
/// path otherwise uses scalar libm transcendentals, so vectorizing with the
/// 1-ULP ArmPL SVE `exp` is the intended fast path. Opt out with
/// `LLMLIP_DISABLE_SVE` / `LLMLIP_DISABLE_SIMD` /
/// `LLMLIP_ESEN_MLP_DISABLE_SVE_LAYER_NORM_F64`.
#[cfg(all(
    feature = "armpl",
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
))]
#[inline(always)]
fn esen_mlp_f64_sve_layer_norm_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("LLMLIP_DISABLE_SIMD").is_none()
            && std::env::var_os("LLMLIP_DISABLE_SVE").is_none()
            && std::env::var_os("LLMLIP_ESEN_MLP_DISABLE_SVE_LAYER_NORM_F64").is_none()
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
    fn llmlip_uma_silu_derivative_sve_f32(len: usize, values: *mut f32, derivatives: *mut f32);
    fn llmlip_uma_layer_norm_silu_rows_sve_f32(
        rows: usize,
        dim: usize,
        input: *const f32,
        weight: *const f32,
        bias: *const f32,
        eps: f32,
        out: *mut f32,
        derivatives: *mut f32,
    );
    fn llmlip_uma_layer_norm_silu_vjp_rows_sve_f32(
        rows: usize,
        dim: usize,
        input: *const f32,
        weight: *const f32,
        grad_out: *const f32,
        silu_derivative: *const f32,
        eps: f32,
        out: *mut f32,
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
    fn llmlip_uma_layer_norm_silu_rows_sve_f64(
        rows: usize,
        dim: usize,
        input: *const f64,
        weight: *const f64,
        bias: *const f64,
        eps: f64,
        out: *mut f64,
        derivatives: *mut f64,
    );
    fn llmlip_uma_layer_norm_silu_vjp_rows_sve_f64(
        rows: usize,
        dim: usize,
        input: *const f64,
        weight: *const f64,
        grad_out: *const f64,
        silu_derivative: *const f64,
        eps: f64,
        out: *mut f64,
    );
}

#[cfg(all(
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
))]
#[inline(always)]
#[allow(unsafe_code)]
fn silu_value_and_derivative_sve_f32_in_place(values: &mut [f32], derivatives: &mut [f32]) -> bool {
    if values.len() != derivatives.len() || values.len() < 32 || !esen_mlp_f32_sve_enabled() {
        return false;
    }
    unsafe {
        llmlip_uma_silu_derivative_sve_f32(
            values.len(),
            values.as_mut_ptr(),
            derivatives.as_mut_ptr(),
        );
    }
    true
}

#[cfg(all(
    feature = "parallel",
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
))]
const ESEN_MLP_SVE_LAYER_NORM_CHUNK_ROWS: usize = 32;

#[cfg(all(
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_code)]
fn layer_norm_silu_sve_rows_f32(
    rows: usize,
    dim: usize,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    out: &mut [f32],
    derivatives: &mut [f32],
) -> bool {
    if !esen_mlp_f32_sve_layer_norm_enabled() {
        return false;
    }
    let Some(total_len) = rows.checked_mul(dim) else {
        return false;
    };
    if rows == 0
        || dim < 32
        || input.len() < total_len
        || out.len() < total_len
        || derivatives.len() < total_len
        || weight.len() < dim
        || bias.len() < dim
    {
        return false;
    }

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && rows >= e3nn_core::esen_parallel_min_nodes() {
        let chunk_len = ESEN_MLP_SVE_LAYER_NORM_CHUNK_ROWS * dim;
        input[..total_len]
            .par_chunks(chunk_len)
            .zip(out[..total_len].par_chunks_mut(chunk_len))
            .zip(derivatives[..total_len].par_chunks_mut(chunk_len))
            .for_each(|((input_chunk, out_chunk), derivative_chunk)| {
                let chunk_rows = input_chunk.len() / dim;
                unsafe {
                    llmlip_uma_layer_norm_silu_rows_sve_f32(
                        chunk_rows,
                        dim,
                        input_chunk.as_ptr(),
                        weight.as_ptr(),
                        bias.as_ptr(),
                        eps,
                        out_chunk.as_mut_ptr(),
                        derivative_chunk.as_mut_ptr(),
                    );
                }
            });
        return true;
    }

    unsafe {
        llmlip_uma_layer_norm_silu_rows_sve_f32(
            rows,
            dim,
            input.as_ptr(),
            weight.as_ptr(),
            bias.as_ptr(),
            eps,
            out.as_mut_ptr(),
            derivatives.as_mut_ptr(),
        );
    }
    true
}

#[cfg(all(
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_code)]
fn layer_norm_silu_vjp_sve_rows_f32(
    rows: usize,
    dim: usize,
    input: &[f32],
    weight: &[f32],
    grad_out: &[f32],
    silu_derivative: &[f32],
    eps: f32,
    out: &mut [f32],
) -> bool {
    if !esen_mlp_f32_sve_layer_norm_enabled() {
        return false;
    }
    let Some(total_len) = rows.checked_mul(dim) else {
        return false;
    };
    if rows == 0
        || dim < 32
        || input.len() < total_len
        || grad_out.len() < total_len
        || silu_derivative.len() < total_len
        || out.len() < total_len
        || weight.len() < dim
    {
        return false;
    }

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && rows >= e3nn_core::esen_parallel_min_nodes() {
        let chunk_len = ESEN_MLP_SVE_LAYER_NORM_CHUNK_ROWS * dim;
        input[..total_len]
            .par_chunks(chunk_len)
            .zip(grad_out[..total_len].par_chunks(chunk_len))
            .zip(silu_derivative[..total_len].par_chunks(chunk_len))
            .zip(out[..total_len].par_chunks_mut(chunk_len))
            .for_each(
                |(((input_chunk, grad_chunk), derivative_chunk), out_chunk)| {
                    let chunk_rows = input_chunk.len() / dim;
                    unsafe {
                        llmlip_uma_layer_norm_silu_vjp_rows_sve_f32(
                            chunk_rows,
                            dim,
                            input_chunk.as_ptr(),
                            weight.as_ptr(),
                            grad_chunk.as_ptr(),
                            derivative_chunk.as_ptr(),
                            eps,
                            out_chunk.as_mut_ptr(),
                        );
                    }
                },
            );
        return true;
    }

    unsafe {
        llmlip_uma_layer_norm_silu_vjp_rows_sve_f32(
            rows,
            dim,
            input.as_ptr(),
            weight.as_ptr(),
            grad_out.as_ptr(),
            silu_derivative.as_ptr(),
            eps,
            out.as_mut_ptr(),
        );
    }
    true
}

#[cfg(not(all(
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
)))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn layer_norm_silu_sve_rows_f32(
    _rows: usize,
    _dim: usize,
    _input: &[f32],
    _weight: &[f32],
    _bias: &[f32],
    _eps: f32,
    _out: &mut [f32],
    _derivatives: &mut [f32],
) -> bool {
    false
}

#[cfg(not(all(
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
)))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn layer_norm_silu_vjp_sve_rows_f32(
    _rows: usize,
    _dim: usize,
    _input: &[f32],
    _weight: &[f32],
    _grad_out: &[f32],
    _silu_derivative: &[f32],
    _eps: f32,
    _out: &mut [f32],
) -> bool {
    false
}

// --- f64 LayerNorm+SiLU SVE wrappers (ArmPL libamath `exp`) -----------------
// The f64 UMA radial-MLP path otherwise runs scalar libm transcendentals; these
// wrappers route the per-row LayerNorm+SiLU through the SVE kernels, parallel
// over row chunks. Result matches the scalar reference to ~1 ULP (verified by
// the energy/force MAE against upstream FP64).

#[cfg(all(
    feature = "armpl",
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_code)]
fn layer_norm_silu_sve_rows_f64(
    rows: usize,
    dim: usize,
    input: &[f64],
    weight: &[f64],
    bias: &[f64],
    eps: f64,
    out: &mut [f64],
    derivatives: &mut [f64],
) -> bool {
    if !esen_mlp_f64_sve_layer_norm_enabled() {
        return false;
    }
    let Some(total_len) = rows.checked_mul(dim) else {
        return false;
    };
    if rows == 0
        || dim == 0
        || input.len() < total_len
        || out.len() < total_len
        || derivatives.len() < total_len
        || weight.len() < dim
        || bias.len() < dim
    {
        return false;
    }

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && rows >= e3nn_core::esen_parallel_min_nodes() {
        let chunk_len = ESEN_MLP_SVE_LAYER_NORM_CHUNK_ROWS * dim;
        input[..total_len]
            .par_chunks(chunk_len)
            .zip(out[..total_len].par_chunks_mut(chunk_len))
            .zip(derivatives[..total_len].par_chunks_mut(chunk_len))
            .for_each(|((input_chunk, out_chunk), derivative_chunk)| {
                let chunk_rows = input_chunk.len() / dim;
                unsafe {
                    llmlip_uma_layer_norm_silu_rows_sve_f64(
                        chunk_rows,
                        dim,
                        input_chunk.as_ptr(),
                        weight.as_ptr(),
                        bias.as_ptr(),
                        eps,
                        out_chunk.as_mut_ptr(),
                        derivative_chunk.as_mut_ptr(),
                    );
                }
            });
        return true;
    }

    unsafe {
        llmlip_uma_layer_norm_silu_rows_sve_f64(
            rows,
            dim,
            input.as_ptr(),
            weight.as_ptr(),
            bias.as_ptr(),
            eps,
            out.as_mut_ptr(),
            derivatives.as_mut_ptr(),
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
#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[allow(unsafe_code)]
fn layer_norm_silu_vjp_sve_rows_f64(
    rows: usize,
    dim: usize,
    input: &[f64],
    weight: &[f64],
    grad_out: &[f64],
    silu_derivative: &[f64],
    eps: f64,
    out: &mut [f64],
) -> bool {
    if !esen_mlp_f64_sve_layer_norm_enabled() {
        return false;
    }
    let Some(total_len) = rows.checked_mul(dim) else {
        return false;
    };
    if rows == 0
        || dim == 0
        || input.len() < total_len
        || grad_out.len() < total_len
        || silu_derivative.len() < total_len
        || out.len() < total_len
        || weight.len() < dim
    {
        return false;
    }

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && rows >= e3nn_core::esen_parallel_min_nodes() {
        let chunk_len = ESEN_MLP_SVE_LAYER_NORM_CHUNK_ROWS * dim;
        input[..total_len]
            .par_chunks(chunk_len)
            .zip(grad_out[..total_len].par_chunks(chunk_len))
            .zip(silu_derivative[..total_len].par_chunks(chunk_len))
            .zip(out[..total_len].par_chunks_mut(chunk_len))
            .for_each(|(((input_chunk, grad_chunk), deriv_chunk), out_chunk)| {
                let chunk_rows = input_chunk.len() / dim;
                unsafe {
                    llmlip_uma_layer_norm_silu_vjp_rows_sve_f64(
                        chunk_rows,
                        dim,
                        input_chunk.as_ptr(),
                        weight.as_ptr(),
                        grad_chunk.as_ptr(),
                        deriv_chunk.as_ptr(),
                        eps,
                        out_chunk.as_mut_ptr(),
                    );
                }
            });
        return true;
    }

    unsafe {
        llmlip_uma_layer_norm_silu_vjp_rows_sve_f64(
            rows,
            dim,
            input.as_ptr(),
            weight.as_ptr(),
            grad_out.as_ptr(),
            silu_derivative.as_ptr(),
            eps,
            out.as_mut_ptr(),
        );
    }
    true
}

#[cfg(not(all(
    feature = "armpl",
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
)))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn layer_norm_silu_sve_rows_f64(
    _rows: usize,
    _dim: usize,
    _input: &[f64],
    _weight: &[f64],
    _bias: &[f64],
    _eps: f64,
    _out: &mut [f64],
    _derivatives: &mut [f64],
) -> bool {
    false
}

#[cfg(not(all(
    feature = "armpl",
    target_arch = "aarch64",
    target_os = "linux",
    llmlip_aarch64_sve_kernels
)))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn layer_norm_silu_vjp_sve_rows_f64(
    _rows: usize,
    _dim: usize,
    _input: &[f64],
    _weight: &[f64],
    _grad_out: &[f64],
    _silu_derivative: &[f64],
    _eps: f64,
    _out: &mut [f64],
) -> bool {
    false
}

/// Vectorized f32 logistic sigmoid for NEON (Apple Silicon / aarch64).
///
/// `exp(-y)` is approximated by range-reducing `y` to `r = y - n·ln2` (so the result is
/// `2^n · exp(r)`) and evaluating a degree-5 Taylor polynomial of `exp(r)` on the reduced
/// range. The `2^n` scale is applied by injecting `n` into the IEEE-754 exponent field.
/// This trades a few ULP of accuracy for avoiding a per-lane libm call; the maximum error
/// versus the exact scalar `silu_value_and_derivative_f32` (value and derivative) is pinned
/// below ~2e-6 by `neon_silu_matches_scalar_within_tolerance`. The scalar tail in
/// `silu_value_and_derivative_f32_in_place` uses exact libm for the remaining < 4 lanes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_code)]
fn sigmoid_f32x4_approx(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::{
        vaddq_f32, vaddq_s32, vcvtq_s32_f32, vdivq_f32, vdupq_n_f32, vdupq_n_s32, vfmaq_f32,
        vmaxq_f32, vminq_f32, vmulq_f32, vnegq_f32, vreinterpretq_f32_s32, vrndnq_f32, vshlq_n_s32,
        vsubq_f32,
    };

    unsafe {
        let one = vdupq_n_f32(1.0);
        let y = vminq_f32(
            vmaxq_f32(vnegq_f32(x), vdupq_n_f32(-88.0)),
            vdupq_n_f32(88.0),
        );
        let n_f = vrndnq_f32(vmulq_f32(y, vdupq_n_f32(std::f32::consts::LOG2_E)));
        let r = vsubq_f32(y, vmulq_f32(n_f, vdupq_n_f32(std::f32::consts::LN_2)));

        let mut poly = vdupq_n_f32(1.0 / 120.0);
        poly = vfmaq_f32(vdupq_n_f32(1.0 / 24.0), poly, r);
        poly = vfmaq_f32(vdupq_n_f32(1.0 / 6.0), poly, r);
        poly = vfmaq_f32(vdupq_n_f32(0.5), poly, r);
        poly = vfmaq_f32(one, poly, r);
        poly = vfmaq_f32(one, poly, r);

        let exponent = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n_f), vdupq_n_s32(127)), 23);
        let exp_neg = vmulq_f32(poly, vreinterpretq_f32_s32(exponent));
        vdivq_f32(one, vaddq_f32(one, exp_neg))
    }
}

#[inline]
#[allow(unsafe_code)]
fn silu_value_and_derivative_f32_in_place(values: &mut [f32], derivatives: &mut [f32]) {
    debug_assert_eq!(values.len(), derivatives.len());
    #[cfg(all(
        target_arch = "aarch64",
        target_os = "linux",
        llmlip_aarch64_sve_kernels
    ))]
    if silu_value_and_derivative_sve_f32_in_place(values, derivatives) {
        return;
    }

    let mut index = 0usize;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::{
            vaddq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32, vsubq_f32,
        };

        let one = vdupq_n_f32(1.0);
        while index + 4 <= values.len() {
            let x = vld1q_f32(values.as_ptr().add(index));
            let sigmoid = sigmoid_f32x4_approx(x);
            vst1q_f32(values.as_mut_ptr().add(index), vmulq_f32(x, sigmoid));
            let derivative = vmulq_f32(
                sigmoid,
                vaddq_f32(one, vmulq_f32(x, vsubq_f32(one, sigmoid))),
            );
            vst1q_f32(derivatives.as_mut_ptr().add(index), derivative);
            index += 4;
        }
    }

    while index < values.len() {
        let (activated, derivative) = silu_value_and_derivative_f32(values[index]);
        values[index] = activated;
        derivatives[index] = derivative;
        index += 1;
    }
}

#[allow(unsafe_code)]
fn layer_norm_forward_into_f32(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    out: &mut [f32],
) {
    let len = input.len();
    let dim = len as f32;
    let mut sum = 0.0f32;
    let mut index = 0usize;

    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    unsafe {
        use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32};

        let mut sum_v = vdupq_n_f32(0.0);
        while index + 4 <= len {
            sum_v = vaddq_f32(sum_v, vld1q_f32(input.as_ptr().add(index)));
            index += 4;
        }
        sum += vaddvq_f32(sum_v);
    }
    while index < len {
        sum += input[index];
        index += 1;
    }
    let mean = sum / dim;

    let mut variance_sum = 0.0f32;
    index = 0;
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    unsafe {
        use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32, vsubq_f32};

        let mean_v = vdupq_n_f32(mean);
        let mut variance_v = vdupq_n_f32(0.0);
        while index + 4 <= len {
            let centered = vsubq_f32(vld1q_f32(input.as_ptr().add(index)), mean_v);
            variance_v = vaddq_f32(
                variance_v,
                std::arch::aarch64::vmulq_f32(centered, centered),
            );
            index += 4;
        }
        variance_sum += vaddvq_f32(variance_v);
    }
    while index < len {
        let c = input[index] - mean;
        variance_sum += c * c;
        index += 1;
    }
    let variance = variance_sum / dim;
    let inv_std = (variance + eps).powf(-0.5);

    index = 0;
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    unsafe {
        use std::arch::aarch64::{
            vdupq_n_f32, vfmaq_f32, vld1q_f32, vmulq_f32, vst1q_f32, vsubq_f32,
        };

        let mean_v = vdupq_n_f32(mean);
        let inv_std_v = vdupq_n_f32(inv_std);
        while index + 4 <= len {
            let normalized = vmulq_f32(
                vsubq_f32(vld1q_f32(input.as_ptr().add(index)), mean_v),
                inv_std_v,
            );
            let value = vfmaq_f32(
                vld1q_f32(bias.as_ptr().add(index)),
                normalized,
                vld1q_f32(weight.as_ptr().add(index)),
            );
            vst1q_f32(out.as_mut_ptr().add(index), value);
            index += 4;
        }
    }
    while index < len {
        out[index] = (input[index] - mean) * inv_std * weight[index] + bias[index];
        index += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn dense_forward_rows_f32(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    weight_out_by_in: &[f32],
    weight_in_by_out: &[f32],
    bias: &[f32],
    input: &[f32],
    out: &mut [f32],
) {
    // weight_out_by_in is output-major [out_dim, in_dim]; weight_in_by_out is
    // the pre-transposed [in_dim, out_dim] twin used by BLAS forward.
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    let _ = weight_out_by_in;
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    let _ = weight_in_by_out;

    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    {
        crate::blas_backend::dense_forward_input_major_weight_f32(
            input,
            rows,
            in_dim,
            weight_in_by_out,
            out,
            out_dim,
            1.0,
            0.0,
        );
        add_bias_rows_f32(rows, out_dim, bias, out);
    }
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    let _ = weight_in_by_out;
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
                value += weight_out_by_in[weight_row_start + in_index] * input[in_start + in_index];
            }
            out[out_start + out_index] = value;
        }
    }
}

fn transpose_output_major_f64_to_input_major_f32(
    in_dim: usize,
    out_dim: usize,
    weight_out_by_in: &[f64],
) -> Vec<f32> {
    debug_assert!(weight_out_by_in.len() >= in_dim * out_dim);
    let mut transposed = vec![0.0_f32; in_dim * out_dim];
    for out_index in 0..out_dim {
        let src_row = out_index * in_dim;
        for in_index in 0..in_dim {
            transposed[in_index * out_dim + out_index] =
                weight_out_by_in[src_row + in_index] as f32;
        }
    }
    transposed
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
fn add_bias_rows_f32(rows: usize, row_len: usize, bias: &[f32], out: &mut [f32]) {
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && rows.saturating_mul(row_len) >= 65_536 {
        out.par_chunks_mut(row_len).take(rows).for_each(|row| {
            for (value, bias_value) in row.iter_mut().zip(bias.iter()) {
                *value += *bias_value;
            }
        });
        return;
    }

    for row in out.chunks_exact_mut(row_len).take(rows) {
        for (value, bias_value) in row.iter_mut().zip(bias.iter()) {
            *value += *bias_value;
        }
    }
}

fn silu_value_and_derivative(value: f64) -> (f64, f64) {
    let sigmoid = if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    };
    (value * sigmoid, sigmoid * (1.0 + value * (1.0 - sigmoid)))
}

fn dense_forward(layer: &EsenRadialMlpLayer, input: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; layer.out_dim];
    for (out_index, out_value) in out.iter_mut().enumerate() {
        let row_start = out_index * layer.in_dim;
        let mut value = layer.bias[out_index];
        for (in_index, input_value) in input.iter().enumerate() {
            value += layer.weight[row_start + in_index] * input_value;
        }
        *out_value = value;
    }
    out
}

fn dense_forward_rows(
    layer: &EsenRadialMlpLayer,
    rows: usize,
    input: &[f64],
    out: &mut [f64],
) -> Result<()> {
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    {
        ensure_blas_dims(&[
            ("rows", rows),
            ("in_dim", layer.in_dim),
            ("out_dim", layer.out_dim),
        ])?;
        crate::blas_backend::dense_forward_output_major_weight(
            input,
            rows,
            layer.in_dim,
            &layer.weight,
            out,
            layer.out_dim,
            1.0,
            0.0,
        );
        add_bias_rows(rows, layer.out_dim, &layer.bias, out);
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
        for row in 0..rows {
            let input_start = row * layer.in_dim;
            let out_start = row * layer.out_dim;
            let row_out = dense_forward(layer, &input[input_start..input_start + layer.in_dim]);
            out[out_start..out_start + layer.out_dim].copy_from_slice(&row_out);
        }
    }
    Ok(())
}

fn dense_forward_rows_with_weight_bias(
    rows: usize,
    out_dim: usize,
    in_dim: usize,
    weight_out_by_in: &[f64],
    bias: &[f64],
    input: &[f64],
    out: &mut [f64],
) -> Result<()> {
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    {
        ensure_blas_dims(&[("rows", rows), ("in_dim", in_dim), ("out_dim", out_dim)])?;
        crate::blas_backend::dense_forward_output_major_weight(
            input,
            rows,
            in_dim,
            weight_out_by_in,
            out,
            out_dim,
            1.0,
            0.0,
        );
        add_bias_rows(rows, out_dim, bias, out);
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
        for row in 0..rows {
            let input_start = row * in_dim;
            let out_start = row * out_dim;
            for out_index in 0..out_dim {
                let weight_row_start = out_index * in_dim;
                let mut value = bias[out_index];
                for in_index in 0..in_dim {
                    value += weight_out_by_in[weight_row_start + in_index]
                        * input[input_start + in_index];
                }
                out[out_start + out_index] = value;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dense_forward_rows_with_weight_bias_strided_output(
    rows: usize,
    out_dim: usize,
    in_dim: usize,
    weight_out_by_in: &[f64],
    bias: &[f64],
    input: &[f64],
    out_row_stride: usize,
    out_feature_offset: usize,
    out: &mut [f64],
) -> Result<()> {
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    {
        portable_blas::dense_forward_transposed_weight_strided_output(
            input,
            rows,
            in_dim,
            weight_out_by_in,
            out_dim,
            out_row_stride,
            out_feature_offset,
            out,
        )?;
        add_bias_rows_strided(rows, out_dim, bias, out_row_stride, out_feature_offset, out);
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
        for row in 0..rows {
            let input_start = row * in_dim;
            let out_start = row * out_row_stride + out_feature_offset;
            for out_index in 0..out_dim {
                let weight_row_start = out_index * in_dim;
                let mut value = bias[out_index];
                for in_index in 0..in_dim {
                    value += weight_out_by_in[weight_row_start + in_index]
                        * input[input_start + in_index];
                }
                out[out_start + out_index] = value;
            }
        }
    }
    Ok(())
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
fn add_bias_rows(rows: usize, row_len: usize, bias: &[f64], out: &mut [f64]) {
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && rows >= e3nn_core::esen_parallel_min_nodes() {
        out.par_chunks_mut(row_len).for_each(|row| {
            for (value, bias_value) in row.iter_mut().zip(bias.iter()) {
                *value += *bias_value;
            }
        });
        return;
    }

    for row in out.chunks_exact_mut(row_len).take(rows) {
        for (value, bias_value) in row.iter_mut().zip(bias.iter()) {
            *value += *bias_value;
        }
    }
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
fn add_bias_rows_strided(
    rows: usize,
    row_len: usize,
    bias: &[f64],
    out_row_stride: usize,
    out_feature_offset: usize,
    out: &mut [f64],
) {
    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && rows >= e3nn_core::esen_parallel_min_nodes() {
        out.par_chunks_mut(out_row_stride)
            .take(rows)
            .for_each(|row| {
                let target = &mut row[out_feature_offset..out_feature_offset + row_len];
                for (value, bias_value) in target.iter_mut().zip(bias.iter()) {
                    *value += *bias_value;
                }
            });
        return;
    }

    for row in out.chunks_exact_mut(out_row_stride).take(rows) {
        let target = &mut row[out_feature_offset..out_feature_offset + row_len];
        for (value, bias_value) in target.iter_mut().zip(bias.iter()) {
            *value += *bias_value;
        }
    }
}

fn dense_vjp_input(layer: &EsenRadialMlpLayer, grad_out: &[f64]) -> Vec<f64> {
    let mut grad_input = vec![0.0; layer.in_dim];
    for (out_index, grad) in grad_out.iter().enumerate() {
        let row_start = out_index * layer.in_dim;
        for (in_index, grad_input_value) in grad_input.iter_mut().enumerate() {
            *grad_input_value += layer.weight[row_start + in_index] * grad;
        }
    }
    grad_input
}

fn dense_vjp_input_rows(
    layer: &EsenRadialMlpLayer,
    rows: usize,
    grad_out: &[f64],
    grad_input: &mut [f64],
) -> Result<()> {
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    {
        ensure_blas_dims(&[
            ("rows", rows),
            ("in_dim", layer.in_dim),
            ("out_dim", layer.out_dim),
        ])?;
        crate::blas_backend::dense_vjp_input_output_major_weight(
            rows,
            layer.in_dim,
            layer.out_dim,
            &layer.weight,
            grad_out,
            grad_input,
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
    {
        for row in 0..rows {
            let grad_out_start = row * layer.out_dim;
            let grad_input_start = row * layer.in_dim;
            let row_grad = dense_vjp_input(
                layer,
                &grad_out[grad_out_start..grad_out_start + layer.out_dim],
            );
            grad_input[grad_input_start..grad_input_start + layer.in_dim]
                .copy_from_slice(&row_grad);
        }
    }
    Ok(())
}

fn dense_vjp_input_rows_add_with_weight(
    rows: usize,
    out_dim: usize,
    in_dim: usize,
    weight_out_by_in: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
) -> Result<()> {
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    {
        ensure_blas_dims(&[("rows", rows), ("in_dim", in_dim), ("out_dim", out_dim)])?;
        crate::blas_backend::dense_vjp_input_output_major_weight(
            rows,
            in_dim,
            out_dim,
            weight_out_by_in,
            grad_out,
            grad_input,
            1.0,
            1.0,
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
        for row in 0..rows {
            let grad_out_start = row * out_dim;
            let grad_input_start = row * in_dim;
            let grad_input_row = &mut grad_input[grad_input_start..grad_input_start + in_dim];
            for out_index in 0..out_dim {
                let grad = grad_out[grad_out_start + out_index];
                let weight_row_start = out_index * in_dim;
                for in_index in 0..in_dim {
                    grad_input_row[in_index] +=
                        weight_out_by_in[weight_row_start + in_index] * grad;
                }
            }
        }
    }
    Ok(())
}

fn layer_norm_forward(input: &[f64], weight: &[f64], bias: &[f64], eps: f64) -> Vec<f64> {
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

/// Serial per-row LayerNorm+SiLU with derivative cache (scalar fallback for the
/// f64 SVE path).
#[allow(clippy::too_many_arguments)]
fn serial_layer_norm_silu_rows(
    rows: usize,
    dim: usize,
    input: &[f64],
    weight: &[f64],
    bias: &[f64],
    eps: f64,
    out: &mut [f64],
    derivatives: &mut [f64],
) {
    for row in 0..rows {
        let start = row * dim;
        let end = start + dim;
        layer_norm_forward_into(&input[start..end], weight, bias, eps, &mut out[start..end]);
        for index in start..end {
            let (activated, derivative) = silu_value_and_derivative(out[index]);
            out[index] = activated;
            derivatives[index] = derivative;
        }
    }
}

fn layer_norm_forward_into(input: &[f64], weight: &[f64], bias: &[f64], eps: f64, out: &mut [f64]) {
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
    for (((out_value, value), weight), bias) in out
        .iter_mut()
        .zip(input.iter())
        .zip(weight.iter())
        .zip(bias.iter())
    {
        *out_value = (*value - mean) * inv_std * weight + bias;
    }
}

/// Serial per-row LayerNorm+SiLU VJP (scalar fallback for the f64 SVE path).
#[allow(clippy::too_many_arguments)]
fn serial_layer_norm_silu_vjp_rows(
    rows: usize,
    dim: usize,
    input: &[f64],
    weight: &[f64],
    grad_out: &[f64],
    silu_derivative: &[f64],
    eps: f64,
    out: &mut [f64],
) {
    for row in 0..rows {
        let start = row * dim;
        let end = start + dim;
        layer_norm_silu_vjp_input_into(
            &input[start..end],
            weight,
            &grad_out[start..end],
            &silu_derivative[start..end],
            eps,
            &mut out[start..end],
        );
    }
}

fn layer_norm_silu_vjp_input_into(
    input: &[f64],
    weight: &[f64],
    grad_out: &[f64],
    silu_derivative: &[f64],
    eps: f64,
    out: &mut [f64],
) {
    let dim = input.len();
    let mean = input.iter().sum::<f64>() / dim as f64;
    let variance = input
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / dim as f64;
    let inv_std = (variance + eps).powf(-0.5);
    let mut mean_grad = 0.0;
    let mut mean_grad_normalized = 0.0;
    for index in 0..dim {
        let normalized = (input[index] - mean) * inv_std;
        let grad_normalized = grad_out[index] * silu_derivative[index] * weight[index];
        mean_grad += grad_normalized;
        mean_grad_normalized += grad_normalized * normalized;
    }
    mean_grad /= dim as f64;
    mean_grad_normalized /= dim as f64;
    for index in 0..dim {
        let normalized = (input[index] - mean) * inv_std;
        let grad_normalized = grad_out[index] * silu_derivative[index] * weight[index];
        out[index] = inv_std * (grad_normalized - mean_grad - normalized * mean_grad_normalized);
    }
}

/// f32-native linear-layer VJP: grad_input[rows, in_dim] = grad_out[rows, out_dim] @ weight[out_dim, in_dim].
///
/// Overwrites (sets, not accumulates) `grad_input`. Uses sgemm on BLAS builds;
/// falls back to pure Rust otherwise.
fn dense_vjp_input_rows_f32(
    layer: &EsenRadialMlpLayer,
    rows: usize,
    grad_out: &[f32],
    grad_input: &mut [f32],
) {
    // beta = 0.0 → set (overwrite, not accumulate); alpha = 1.0.
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
            layer.in_dim,
            layer.out_dim,
            &layer.weight_f32,
            grad_out,
            grad_input,
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
    // Pure-Rust fallback: grad_input[r, i] = sum_j grad_out[r, j] * weight[j, i]
    // weight is [out_dim, in_dim] row-major
    for row in 0..rows {
        let grad_out_start = row * layer.out_dim;
        let grad_input_start = row * layer.in_dim;
        for in_index in 0..layer.in_dim {
            let mut acc = 0.0f32;
            for out_index in 0..layer.out_dim {
                acc += grad_out[grad_out_start + out_index]
                    * layer.weight_f32[out_index * layer.in_dim + in_index];
            }
            grad_input[grad_input_start + in_index] = acc;
        }
    }
}

/// f32-native LayerNorm+SiLU VJP for a single row (analogous to
/// [`layer_norm_silu_vjp_input_into`] but in f32).
#[allow(unsafe_code)]
fn layer_norm_silu_vjp_input_into_f32(
    input: &[f32],
    weight: &[f32],
    grad_out: &[f32],
    silu_derivative: &[f32],
    eps: f32,
    out: &mut [f32],
) {
    let dim = input.len();
    let dim_f = dim as f32;
    let mut sum = 0.0f32;
    let mut index = 0usize;
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    unsafe {
        use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32};

        let mut sum_v = vdupq_n_f32(0.0);
        while index + 4 <= dim {
            sum_v = vaddq_f32(sum_v, vld1q_f32(input.as_ptr().add(index)));
            index += 4;
        }
        sum += vaddvq_f32(sum_v);
    }
    while index < dim {
        sum += input[index];
        index += 1;
    }
    let mean = sum / dim_f;

    let mut variance_sum = 0.0f32;
    index = 0;
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    unsafe {
        use std::arch::aarch64::{
            vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vsubq_f32,
        };

        let mean_v = vdupq_n_f32(mean);
        let mut variance_v = vdupq_n_f32(0.0);
        while index + 4 <= dim {
            let centered = vsubq_f32(vld1q_f32(input.as_ptr().add(index)), mean_v);
            variance_v = vaddq_f32(variance_v, vmulq_f32(centered, centered));
            index += 4;
        }
        variance_sum += vaddvq_f32(variance_v);
    }
    while index < dim {
        let centered = input[index] - mean;
        variance_sum += centered * centered;
        index += 1;
    }
    let variance = variance_sum / dim_f;
    let inv_std = (variance + eps).powf(-0.5);
    let mut mean_grad = 0.0f32;
    let mut mean_grad_normalized = 0.0f32;
    index = 0;
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    unsafe {
        use std::arch::aarch64::{
            vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vsubq_f32,
        };

        let mean_v = vdupq_n_f32(mean);
        let inv_std_v = vdupq_n_f32(inv_std);
        let mut mean_grad_v = vdupq_n_f32(0.0);
        let mut mean_grad_normalized_v = vdupq_n_f32(0.0);
        while index + 4 <= dim {
            let normalized = vmulq_f32(
                vsubq_f32(vld1q_f32(input.as_ptr().add(index)), mean_v),
                inv_std_v,
            );
            let grad_normalized = vmulq_f32(
                vmulq_f32(
                    vld1q_f32(grad_out.as_ptr().add(index)),
                    vld1q_f32(silu_derivative.as_ptr().add(index)),
                ),
                vld1q_f32(weight.as_ptr().add(index)),
            );
            mean_grad_v = vaddq_f32(mean_grad_v, grad_normalized);
            mean_grad_normalized_v = vaddq_f32(
                mean_grad_normalized_v,
                vmulq_f32(grad_normalized, normalized),
            );
            index += 4;
        }
        mean_grad += vaddvq_f32(mean_grad_v);
        mean_grad_normalized += vaddvq_f32(mean_grad_normalized_v);
    }
    while index < dim {
        let normalized = (input[index] - mean) * inv_std;
        let grad_normalized = grad_out[index] * silu_derivative[index] * weight[index];
        mean_grad += grad_normalized;
        mean_grad_normalized += grad_normalized * normalized;
        index += 1;
    }
    mean_grad /= dim_f;
    mean_grad_normalized /= dim_f;

    index = 0;
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    unsafe {
        use std::arch::aarch64::{vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32, vsubq_f32};

        let mean_v = vdupq_n_f32(mean);
        let inv_std_v = vdupq_n_f32(inv_std);
        let mean_grad_v = vdupq_n_f32(mean_grad);
        let mean_grad_normalized_v = vdupq_n_f32(mean_grad_normalized);
        while index + 4 <= dim {
            let normalized = vmulq_f32(
                vsubq_f32(vld1q_f32(input.as_ptr().add(index)), mean_v),
                inv_std_v,
            );
            let grad_normalized = vmulq_f32(
                vmulq_f32(
                    vld1q_f32(grad_out.as_ptr().add(index)),
                    vld1q_f32(silu_derivative.as_ptr().add(index)),
                ),
                vld1q_f32(weight.as_ptr().add(index)),
            );
            let centered_grad = vsubq_f32(
                vsubq_f32(grad_normalized, mean_grad_v),
                vmulq_f32(normalized, mean_grad_normalized_v),
            );
            vst1q_f32(
                out.as_mut_ptr().add(index),
                vmulq_f32(inv_std_v, centered_grad),
            );
            index += 4;
        }
    }
    while index < dim {
        let normalized = (input[index] - mean) * inv_std;
        let grad_normalized = grad_out[index] * silu_derivative[index] * weight[index];
        out[index] = inv_std * (grad_normalized - mean_grad - normalized * mean_grad_normalized);
        index += 1;
    }
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

fn invalid_mlp(message: impl Into<String>) -> CoreError {
    CoreError::InvalidMlp(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pins the accuracy of the NEON degree-5 Taylor `exp` approximation in
    // `sigmoid_f32x4_approx`: the vectorized SiLU value and its derivative must stay within a
    // few ULP of the exact scalar `silu_value_and_derivative_f32` across the activation range.
    // Guards against a future polynomial/range-reduction change silently degrading precision.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_silu_matches_scalar_within_tolerance() {
        // Dense sweep over the range SiLU activations realistically span, including the
        // saturating tails where the approximation is worst. Length is a non-multiple of 4
        // so the scalar-tail path (last < 4 lanes) is exercised alongside the NEON body.
        let mut inputs: Vec<f32> = Vec::new();
        let mut x = -40.0_f32;
        while x <= 40.0 {
            inputs.push(x);
            x += 0.013;
        }
        assert!(
            !inputs.len().is_multiple_of(4),
            "want a scalar tail remainder"
        );

        let mut values = inputs.clone();
        let mut derivatives = vec![0.0_f32; inputs.len()];
        silu_value_and_derivative_f32_in_place(&mut values, &mut derivatives);

        let mut max_value_err = 0.0_f32;
        let mut max_deriv_err = 0.0_f32;
        for (i, &input) in inputs.iter().enumerate() {
            let (ref_value, ref_deriv) = silu_value_and_derivative_f32(input);
            max_value_err = max_value_err.max((values[i] - ref_value).abs());
            max_deriv_err = max_deriv_err.max((derivatives[i] - ref_deriv).abs());
        }

        assert!(
            max_value_err < 2.0e-6,
            "NEON SiLU value error {max_value_err:e} exceeds 2e-6 tolerance"
        );
        assert!(
            max_deriv_err < 2.0e-6,
            "NEON SiLU derivative error {max_deriv_err:e} exceeds 2e-6 tolerance"
        );
    }

    #[cfg(all(
        target_arch = "aarch64",
        target_os = "linux",
        llmlip_aarch64_sve_kernels
    ))]
    #[test]
    #[allow(unsafe_code)]
    fn sve_layer_norm_silu_f32_matches_scalar_within_tolerance() {
        if !std::arch::is_aarch64_feature_detected!("sve") {
            return;
        }

        let rows = 3usize;
        let dim = 37usize;
        let eps = 1.0e-5_f32;
        let input: Vec<f32> = (0..rows * dim)
            .map(|i| ((i as f32) * 0.031 - 1.7).sin() * 2.5)
            .collect();
        let weight: Vec<f32> = (0..dim)
            .map(|i| 0.7 + ((i as f32) * 0.017).cos() * 0.2)
            .collect();
        let bias: Vec<f32> = (0..dim)
            .map(|i| ((i as f32) * 0.041 - 0.3).sin() * 0.4)
            .collect();

        let mut expected = vec![0.0_f32; rows * dim];
        let mut expected_derivatives = vec![0.0_f32; rows * dim];
        for row in 0..rows {
            let start = row * dim;
            let end = start + dim;
            layer_norm_forward_into_f32(
                &input[start..end],
                &weight,
                &bias,
                eps,
                &mut expected[start..end],
            );
            for index in start..end {
                let (activated, derivative) = silu_value_and_derivative_f32(expected[index]);
                expected[index] = activated;
                expected_derivatives[index] = derivative;
            }
        }

        let mut actual = vec![0.0_f32; rows * dim];
        let mut actual_derivatives = vec![0.0_f32; rows * dim];
        // Safety: buffers are sized for rows * dim; weight and bias have dim entries.
        unsafe {
            llmlip_uma_layer_norm_silu_rows_sve_f32(
                rows,
                dim,
                input.as_ptr(),
                weight.as_ptr(),
                bias.as_ptr(),
                eps,
                actual.as_mut_ptr(),
                actual_derivatives.as_mut_ptr(),
            );
        }

        let max_value_err = actual
            .iter()
            .zip(expected.iter())
            .map(|(&a, &e)| (a - e).abs())
            .fold(0.0_f32, f32::max);
        let max_deriv_err = actual_derivatives
            .iter()
            .zip(expected_derivatives.iter())
            .map(|(&a, &e)| (a - e).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_value_err <= 2.0e-5,
            "SVE LayerNorm+SiLU f32 value error {max_value_err:e} exceeds 2e-5"
        );
        assert!(
            max_deriv_err <= 2.0e-5,
            "SVE LayerNorm+SiLU f32 derivative error {max_deriv_err:e} exceeds 2e-5"
        );

        let grad_out: Vec<f32> = (0..rows * dim)
            .map(|i| ((i as f32) * 0.019 + 0.2).cos() * 0.3)
            .collect();
        let mut expected_vjp = vec![0.0_f32; rows * dim];
        for row in 0..rows {
            let start = row * dim;
            let end = start + dim;
            layer_norm_silu_vjp_input_into_f32(
                &input[start..end],
                &weight,
                &grad_out[start..end],
                &expected_derivatives[start..end],
                eps,
                &mut expected_vjp[start..end],
            );
        }

        let mut actual_vjp = vec![0.0_f32; rows * dim];
        // Safety: all row-major buffers have rows * dim entries; weight has dim entries.
        unsafe {
            llmlip_uma_layer_norm_silu_vjp_rows_sve_f32(
                rows,
                dim,
                input.as_ptr(),
                weight.as_ptr(),
                grad_out.as_ptr(),
                actual_derivatives.as_ptr(),
                eps,
                actual_vjp.as_mut_ptr(),
            );
        }
        let max_vjp_err = actual_vjp
            .iter()
            .zip(expected_vjp.iter())
            .map(|(&a, &e)| (a - e).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_vjp_err <= 8.0e-5,
            "SVE LayerNorm+SiLU f32 VJP error {max_vjp_err:e} exceeds 8e-5"
        );
    }

    #[allow(clippy::unwrap_used)]
    fn make_test_mlp(in_dim: usize, hidden_dim: usize, out_dim: usize) -> EsenRadialMlp {
        // Hidden layer: weight [hidden_dim, in_dim] output-major, with LayerNorm+SiLU.
        let hidden_weight: Vec<f64> = (0..hidden_dim * in_dim)
            .map(|i| (i as f64 * 0.07 - 1.5) * 0.1)
            .collect();
        let hidden_bias: Vec<f64> = (0..hidden_dim)
            .map(|i| (i as f64 * 0.13 - 0.5) * 0.05)
            .collect();
        let ln_weight: Vec<f64> = (0..hidden_dim).map(|i| 0.9 + i as f64 * 0.02).collect();
        let ln_bias: Vec<f64> = (0..hidden_dim).map(|i| i as f64 * 0.05 - 0.1).collect();
        let hidden_layer = EsenRadialMlpLayer::new(
            in_dim,
            hidden_dim,
            hidden_weight,
            hidden_bias,
            Some((ln_weight, ln_bias)),
        )
        .unwrap();
        // Final layer: weight [out_dim, hidden_dim] output-major, no LayerNorm.
        let final_weight: Vec<f64> = (0..out_dim * hidden_dim)
            .map(|i| (i as f64 * 0.11 - 1.0) * 0.1)
            .collect();
        let final_bias: Vec<f64> = (0..out_dim)
            .map(|i| (i as f64 * 0.09 - 0.3) * 0.05)
            .collect();
        let final_layer =
            EsenRadialMlpLayer::new(hidden_dim, out_dim, final_weight, final_bias, None).unwrap();
        EsenRadialMlp::new(vec![hidden_layer, final_layer], 1.0e-5).unwrap()
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn forward_batch_f32_matches_f64_output() {
        let mlp = make_test_mlp(4, 8, 3);
        let rows = 5;
        let input_f64: Vec<f64> = (0..rows * 4)
            .map(|i| (i as f64 * 0.09 - 2.0) * 0.5)
            .collect();
        let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();

        let mut out_f64 = vec![0.0f64; rows * 3];
        let mut cache_f64 = EsenRadialMlpBatchCache::default();
        mlp.forward_batch_contiguous_cache(rows, &input_f64, &mut out_f64, &mut cache_f64)
            .unwrap();

        let mut out_f32 = vec![0.0f32; rows * 3];
        let mut cache_f32 = EsenRadialMlpBatchCache::default();
        mlp.forward_batch_f32(rows, &input_f32, &mut out_f32, &mut cache_f32)
            .unwrap();

        for (actual, expected) in out_f32.iter().zip(out_f64.iter()) {
            let expected_f32 = *expected as f32;
            let diff = (actual - expected_f32).abs();
            assert!(
                diff <= 5.0e-5_f32,
                "output mismatch: actual={actual} expected={expected_f32} diff={diff}"
            );
        }
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn forward_batch_f32_cache_layer_count_matches_f64() {
        let mlp = make_test_mlp(3, 6, 2);
        let rows = 3;
        let input_f32: Vec<f32> = (0..rows * 3).map(|i| i as f32 * 0.1).collect();
        let input_f64: Vec<f64> = input_f32.iter().map(|&x| x as f64).collect();

        let mut out_f32 = vec![0.0f32; rows * 2];
        let mut cache_f32 = EsenRadialMlpBatchCache::default();
        mlp.forward_batch_f32(rows, &input_f32, &mut out_f32, &mut cache_f32)
            .unwrap();

        let mut out_f64 = vec![0.0f64; rows * 2];
        let mut cache_f64 = EsenRadialMlpBatchCache::default();
        mlp.forward_batch_contiguous_cache(rows, &input_f64, &mut out_f64, &mut cache_f64)
            .unwrap();

        assert_eq!(cache_f32.layer_caches.len(), cache_f64.layer_caches.len());
        assert_eq!(cache_f32.rows, rows);
    }

    /// Verifies that `vjp_input_only_batch_contiguous_cache_f32` (sgemm-backed
    /// all-f32 backward) agrees with the f64 VJP to within f32 precision.
    #[test]
    #[allow(clippy::unwrap_used)]
    fn vjp_f32_cache_matches_f64_vjp() {
        let mlp = make_test_mlp(4, 8, 3);
        let rows = 5;
        let input_f64: Vec<f64> = (0..rows * 4)
            .map(|i| (i as f64 * 0.09 - 2.0) * 0.5)
            .collect();
        let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();
        let grad_out_f64: Vec<f64> = (0..rows * 3)
            .map(|i| (i as f64 * 0.13 - 1.0) * 0.2)
            .collect();
        let grad_out_f32: Vec<f32> = grad_out_f64.iter().map(|&x| x as f32).collect();

        // f64 reference VJP
        let mut cache_f64 = EsenRadialMlpBatchCache::default();
        let mut out_f64 = vec![0.0f64; rows * 3];
        mlp.forward_batch_contiguous_cache(rows, &input_f64, &mut out_f64, &mut cache_f64)
            .unwrap();
        let mut grad_input_f64 = vec![0.0f64; rows * 4];
        mlp.vjp_input_only_batch_contiguous_cache(
            &cache_f64,
            rows,
            &grad_out_f64,
            &mut grad_input_f64,
        )
        .unwrap();

        // f32 native VJP
        let mut cache_f32 = EsenRadialMlpBatchCacheF32::default();
        let mut out_f32 = vec![0.0f32; rows * 3];
        mlp.forward_batch_f32_with_f32_cache(rows, &input_f32, &mut out_f32, &mut cache_f32)
            .unwrap();
        let mut grad_input_f32 = vec![0.0f32; rows * 4];
        mlp.vjp_input_only_batch_contiguous_cache_f32(
            &cache_f32,
            rows,
            &grad_out_f32,
            &mut grad_input_f32,
        )
        .unwrap();

        // Tolerance: f32 rounding gives ~1e-7 absolute error at unit scale;
        // use 1e-4 absolute threshold to cover the full test range.
        for (i, (actual, expected)) in grad_input_f32.iter().zip(grad_input_f64.iter()).enumerate()
        {
            let expected_f32 = *expected as f32;
            let diff = (actual - expected_f32).abs();
            assert!(
                diff <= 1e-4_f32,
                "grad_input[{i}] mismatch: f32={actual} f64_ref={expected_f32} diff={diff}"
            );
        }
    }

    #[cfg(all(
        feature = "parallel",
        feature = "armpl",
        target_arch = "aarch64",
        target_os = "linux",
        llmlip_aarch64_sve_kernels
    ))]
    fn f64_sve_disabled_by_env() -> bool {
        std::env::var_os("LLMLIP_DISABLE_SIMD").is_some()
            || std::env::var_os("LLMLIP_DISABLE_SVE").is_some()
            || std::env::var_os("LLMLIP_ESEN_MLP_DISABLE_SVE_LAYER_NORM_F64").is_some()
    }

    #[cfg(all(
        feature = "parallel",
        feature = "armpl",
        target_arch = "aarch64",
        target_os = "linux",
        llmlip_aarch64_sve_kernels
    ))]
    #[test]
    fn layer_norm_silu_sve_rows_f64_matches_scalar_reference(
    ) -> std::result::Result<(), rayon::ThreadPoolBuildError> {
        if !std::arch::is_aarch64_feature_detected!("sve") || f64_sve_disabled_by_env() {
            return Ok(());
        }

        let rows = 77usize;
        let dim = 31usize;
        let input = (0..rows * dim)
            .map(|i| (i as f64 * 0.017 + 0.13).sin() * 0.7)
            .collect::<Vec<_>>();
        let weight = (0..dim)
            .map(|i| 0.8 + (i as f64 * 0.031 + 0.19).cos() * 0.2)
            .collect::<Vec<_>>();
        let bias = (0..dim)
            .map(|i| (i as f64 * 0.047 + 0.07).sin() * 0.15)
            .collect::<Vec<_>>();
        let eps = 1.0e-5;

        let mut expected = vec![0.0; rows * dim];
        let mut expected_derivatives = vec![0.0; rows * dim];
        serial_layer_norm_silu_rows(
            rows,
            dim,
            &input,
            &weight,
            &bias,
            eps,
            &mut expected,
            &mut expected_derivatives,
        );

        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
        pool.install(|| {
            let mut actual = vec![f64::NAN; rows * dim];
            let mut actual_derivatives = vec![f64::NAN; rows * dim];
            assert!(layer_norm_silu_sve_rows_f64(
                rows,
                dim,
                &input,
                &weight,
                &bias,
                eps,
                &mut actual,
                &mut actual_derivatives,
            ));
            for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
                let abs = (actual - expected).abs();
                assert!(
                    abs <= 2.0e-11,
                    "forward[{index}] actual={actual:.16e} expected={expected:.16e} abs={abs:.3e}"
                );
            }
            for (index, (&actual, &expected)) in actual_derivatives
                .iter()
                .zip(expected_derivatives.iter())
                .enumerate()
            {
                let abs = (actual - expected).abs();
                assert!(
                    abs <= 2.0e-11,
                    "derivative[{index}] actual={actual:.16e} expected={expected:.16e} abs={abs:.3e}"
                );
            }
        });
        Ok(())
    }

    #[cfg(all(
        feature = "parallel",
        feature = "armpl",
        target_arch = "aarch64",
        target_os = "linux",
        llmlip_aarch64_sve_kernels
    ))]
    #[test]
    fn layer_norm_silu_vjp_sve_rows_f64_matches_scalar_reference(
    ) -> std::result::Result<(), rayon::ThreadPoolBuildError> {
        if !std::arch::is_aarch64_feature_detected!("sve") || f64_sve_disabled_by_env() {
            return Ok(());
        }

        let rows = 77usize;
        let dim = 31usize;
        let input = (0..rows * dim)
            .map(|i| (i as f64 * 0.023 + 0.11).cos() * 0.6)
            .collect::<Vec<_>>();
        let weight = (0..dim)
            .map(|i| 0.75 + (i as f64 * 0.029 + 0.17).sin() * 0.25)
            .collect::<Vec<_>>();
        let bias = (0..dim)
            .map(|i| (i as f64 * 0.041 + 0.03).cos() * 0.12)
            .collect::<Vec<_>>();
        let grad_out = (0..rows * dim)
            .map(|i| (i as f64 * 0.019 + 0.31).sin() * 0.4)
            .collect::<Vec<_>>();
        let eps = 1.0e-5;

        let mut forward = vec![0.0; rows * dim];
        let mut silu_derivatives = vec![0.0; rows * dim];
        serial_layer_norm_silu_rows(
            rows,
            dim,
            &input,
            &weight,
            &bias,
            eps,
            &mut forward,
            &mut silu_derivatives,
        );

        let mut expected = vec![0.0; rows * dim];
        serial_layer_norm_silu_vjp_rows(
            rows,
            dim,
            &input,
            &weight,
            &grad_out,
            &silu_derivatives,
            eps,
            &mut expected,
        );

        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
        pool.install(|| {
            let mut actual = vec![f64::NAN; rows * dim];
            assert!(layer_norm_silu_vjp_sve_rows_f64(
                rows,
                dim,
                &input,
                &weight,
                &grad_out,
                &silu_derivatives,
                eps,
                &mut actual,
            ));
            for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
                let abs = (actual - expected).abs();
                assert!(
                    abs <= 2.0e-11,
                    "vjp[{index}] actual={actual:.16e} expected={expected:.16e} abs={abs:.3e}"
                );
            }
        });
        Ok(())
    }
}
