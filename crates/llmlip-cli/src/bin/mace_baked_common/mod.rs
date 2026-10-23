//! Shared command logic for MACE baked binaries.

use std::time::Instant;

use clap::ValueEnum;
use llmlip::{
    read_extxyz, write_extxyz_with_forces, LlmlipModel, Prediction, PredictionComputeDtype,
    PredictionOptions, Structure,
};
use serde::Serialize;

const PREDICTION_SCHEMA_VERSION: &str = "llmlip-baked-mace-prediction-v1";
const BENCHMARK_SCHEMA_VERSION: &str = "llmlip-baked-mace-benchmark-v1";

#[derive(Clone, Copy)]
pub struct BakedRuntime {
    model_id: &'static str,
    load: fn() -> Result<LlmlipModel, String>,
}

impl BakedRuntime {
    const fn new(model_id: &'static str, load: fn() -> Result<LlmlipModel, String>) -> Self {
        Self { model_id, load }
    }
}

#[cfg(feature = "baked-mace-oam-medium")]
#[allow(dead_code)]
pub const MACE_OAM_MEDIUM_RUNTIME: BakedRuntime =
    BakedRuntime::new("mace-oam0-medium-baked", load_mace_oam_medium);

#[cfg(feature = "baked-mace-oam-small")]
#[allow(dead_code)]
pub const MACE_OAM_SMALL_RUNTIME: BakedRuntime =
    BakedRuntime::new("mace-oam0-small-baked", load_mace_oam_small);

#[cfg(feature = "baked-mace-mp0b")]
#[allow(dead_code)]
pub const MACE_MP0B_RUNTIME: BakedRuntime =
    BakedRuntime::new("mace-mp-0b3-medium-baked", load_mace_mp0b);

#[cfg(feature = "baked-mace-osaka24-small")]
#[allow(dead_code)]
pub const MACE_OSAKA24_SMALL_RUNTIME: BakedRuntime =
    BakedRuntime::new("mace-osaka24-small-baked", load_mace_osaka24_small);

#[cfg(feature = "baked-mace-osaka24-medium")]
#[allow(dead_code)]
pub const MACE_OSAKA24_MEDIUM_RUNTIME: BakedRuntime =
    BakedRuntime::new("mace-osaka24-medium-baked", load_mace_osaka24_medium);

#[cfg(feature = "baked-mace-osaka24-large")]
#[allow(dead_code)]
pub const MACE_OSAKA24_LARGE_RUNTIME: BakedRuntime =
    BakedRuntime::new("mace-osaka24-large-baked", load_mace_osaka24_large);

#[cfg(feature = "baked-mace-osaka26")]
#[allow(dead_code)]
pub const MACE_OSAKA26_RUNTIME: BakedRuntime =
    BakedRuntime::new("mace-osaka26-small-baked", load_mace_osaka26);

#[derive(Clone, Copy, Debug, ValueEnum)]
#[value(rename_all = "lower")]
pub enum ComputeDtype {
    #[value(alias = "fp64", alias = "f64")]
    Float64,
    #[value(alias = "fp32", alias = "f32")]
    Float32,
}

impl From<ComputeDtype> for PredictionComputeDtype {
    fn from(value: ComputeDtype) -> Self {
        match value {
            ComputeDtype::Float64 => Self::Float64,
            ComputeDtype::Float32 => Self::Float32,
        }
    }
}

impl ComputeDtype {
    pub const fn report_name(self) -> &'static str {
        match self {
            Self::Float64 => "float64",
            Self::Float32 => "float32",
        }
    }
}

#[derive(Debug, Serialize)]
struct PredictionReport<'a> {
    schema_version: &'static str,
    model: &'static str,
    structure: &'a str,
    atom_count: usize,
    compute_dtype: &'static str,
    load_runtime_ms: f64,
    predict_runtime_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    energy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    forces: Option<Vec<[f64; 3]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stress: Option<[[f64; 3]; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    virial: Option<[[f64; 3]; 3]>,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport<'a> {
    schema_version: &'static str,
    model: &'static str,
    structure: &'a str,
    atom_count: usize,
    requested_outputs: RequestedOutputs,
    compute_dtype: &'static str,
    warmup: usize,
    iterations: usize,
    load_runtime_ms: f64,
    mean_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    min_ms: f64,
    max_ms: f64,
    samples_ms: Vec<f64>,
    checksum: f64,
}

#[derive(Debug, Serialize)]
struct RequestedOutputs {
    energy: bool,
    forces: bool,
    stress: bool,
    virial: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn run_predict(
    runtime: BakedRuntime,
    structure_path: &str,
    energy: bool,
    forces: bool,
    forces_out: Option<String>,
    stress: bool,
    virial: bool,
    json: bool,
    dtype: ComputeDtype,
) -> Result<(), String> {
    let compute_forces = forces || forces_out.is_some();
    validate_requested_outputs(energy, compute_forces, stress, virial, "predict")?;
    let structure =
        read_extxyz(structure_path).map_err(|err| format!("failed to read structure: {err}"))?;
    let (model, load_runtime_ms) = load_baked_model(runtime)?;
    let options = PredictionOptions {
        compute_energy: energy,
        compute_forces,
        compute_stress: stress,
        compute_virial: virial,
        ..PredictionOptions::default()
    };
    let predict_start = Instant::now();
    let prediction = model
        .predict_with_compute_dtype(&structure, &options, dtype.into())
        .map_err(|err| format!("prediction failed: {err}"))?;
    let predict_runtime_ms = elapsed_ms(predict_start);

    if let Some(path) = forces_out {
        let force_values = prediction
            .forces
            .as_ref()
            .ok_or_else(|| "model did not return requested forces".to_owned())?;
        write_extxyz_with_forces(&path, &structure, force_values, prediction.energy)
            .map_err(|err| format!("failed to write forces: {err}"))?;
    }

    if json {
        write_prediction_json(
            structure_path,
            &structure,
            prediction,
            runtime,
            dtype,
            load_runtime_ms,
            predict_runtime_ms,
        )
    } else {
        print_prediction_text(&prediction, energy, stress, virial)?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_benchmark_predict(
    runtime: BakedRuntime,
    structure_path: &str,
    energy: bool,
    forces: bool,
    stress: bool,
    virial: bool,
    warmup: usize,
    iterations: usize,
    dtype: ComputeDtype,
) -> Result<(), String> {
    validate_requested_outputs(energy, forces, stress, virial, "benchmark-predict")?;
    if iterations == 0 {
        return Err("--iterations must be at least 1".to_owned());
    }
    let structure =
        read_extxyz(structure_path).map_err(|err| format!("failed to read structure: {err}"))?;
    let (model, load_runtime_ms) = load_baked_model(runtime)?;
    let options = PredictionOptions {
        compute_energy: energy,
        compute_forces: forces,
        compute_stress: stress,
        compute_virial: virial,
        ..PredictionOptions::default()
    };
    let mut checksum = 0.0;
    for _ in 0..warmup {
        let prediction = model
            .predict_with_compute_dtype(&structure, &options, dtype.into())
            .map_err(|err| format!("warmup prediction failed: {err}"))?;
        checksum += prediction_checksum(&prediction);
    }

    let mut samples_ms = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let prediction = model
            .predict_with_compute_dtype(&structure, &options, dtype.into())
            .map_err(|err| format!("benchmark prediction failed: {err}"))?;
        samples_ms.push(elapsed_ms(start));
        checksum += prediction_checksum(&prediction);
    }
    let mut sorted = samples_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let mean_ms = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;
    let report = BenchmarkReport {
        schema_version: BENCHMARK_SCHEMA_VERSION,
        model: runtime.model_id,
        structure: structure_path,
        atom_count: structure.len(),
        requested_outputs: RequestedOutputs {
            energy,
            forces,
            stress,
            virial,
        },
        compute_dtype: dtype.report_name(),
        warmup,
        iterations,
        load_runtime_ms,
        mean_ms,
        median_ms: percentile(&sorted, 50.0)?,
        p95_ms: percentile(&sorted, 95.0)?,
        min_ms: *sorted
            .first()
            .ok_or_else(|| "benchmark produced no samples".to_owned())?,
        max_ms: *sorted
            .last()
            .ok_or_else(|| "benchmark produced no samples".to_owned())?,
        samples_ms,
        checksum,
    };
    serde_json::to_writer_pretty(std::io::stdout(), &report)
        .map_err(|err| format!("failed to write benchmark JSON: {err}"))?;
    println!();
    Ok(())
}

fn load_baked_model(runtime: BakedRuntime) -> Result<(LlmlipModel, f64), String> {
    let start = Instant::now();
    let model = (runtime.load)()?;
    Ok((model, elapsed_ms(start)))
}

#[cfg(feature = "baked-mace-oam-medium")]
#[allow(dead_code)]
fn load_mace_oam_medium() -> Result<LlmlipModel, String> {
    LlmlipModel::load_baked_mace_oam_medium()
        .map_err(|err| format!("failed to construct baked MACE-OAM Medium model: {err}"))
}

#[cfg(feature = "baked-mace-oam-small")]
#[allow(dead_code)]
fn load_mace_oam_small() -> Result<LlmlipModel, String> {
    LlmlipModel::load_baked_mace_oam_small()
        .map_err(|err| format!("failed to construct baked MACE-OAM Small model: {err}"))
}

#[cfg(feature = "baked-mace-mp0b")]
#[allow(dead_code)]
fn load_mace_mp0b() -> Result<LlmlipModel, String> {
    LlmlipModel::load_baked_mace_mp0b()
        .map_err(|err| format!("failed to construct baked MACE-MP-0b model: {err}"))
}

#[cfg(feature = "baked-mace-osaka24-small")]
#[allow(dead_code)]
fn load_mace_osaka24_small() -> Result<LlmlipModel, String> {
    LlmlipModel::load_baked_mace_osaka24_small()
        .map_err(|err| format!("failed to construct baked MACE-Osaka24 Small model: {err}"))
}

#[cfg(feature = "baked-mace-osaka24-medium")]
#[allow(dead_code)]
fn load_mace_osaka24_medium() -> Result<LlmlipModel, String> {
    LlmlipModel::load_baked_mace_osaka24_medium()
        .map_err(|err| format!("failed to construct baked MACE-Osaka24 Medium model: {err}"))
}

#[cfg(feature = "baked-mace-osaka24-large")]
#[allow(dead_code)]
fn load_mace_osaka24_large() -> Result<LlmlipModel, String> {
    LlmlipModel::load_baked_mace_osaka24_large()
        .map_err(|err| format!("failed to construct baked MACE-Osaka24 Large model: {err}"))
}

#[cfg(feature = "baked-mace-osaka26")]
#[allow(dead_code)]
fn load_mace_osaka26() -> Result<LlmlipModel, String> {
    LlmlipModel::load_baked_mace_osaka26()
        .map_err(|err| format!("failed to construct baked MACE-Osaka26 model: {err}"))
}

fn validate_requested_outputs(
    energy: bool,
    forces: bool,
    stress: bool,
    virial: bool,
    command: &str,
) -> Result<(), String> {
    if energy || forces || stress || virial {
        Ok(())
    } else {
        Err(format!(
            "{command} requires at least one of --energy, --forces, --stress, or --virial"
        ))
    }
}

fn write_prediction_json(
    structure_path: &str,
    structure: &Structure,
    prediction: Prediction,
    runtime: BakedRuntime,
    dtype: ComputeDtype,
    load_runtime_ms: f64,
    predict_runtime_ms: f64,
) -> Result<(), String> {
    let report = PredictionReport {
        schema_version: PREDICTION_SCHEMA_VERSION,
        model: runtime.model_id,
        structure: structure_path,
        atom_count: structure.len(),
        compute_dtype: dtype.report_name(),
        load_runtime_ms,
        predict_runtime_ms,
        energy: prediction.energy,
        forces: prediction.forces,
        stress: prediction.stress,
        virial: prediction.virial,
    };
    serde_json::to_writer_pretty(std::io::stdout(), &report)
        .map_err(|err| format!("failed to write prediction JSON: {err}"))?;
    println!();
    Ok(())
}

fn print_prediction_text(
    prediction: &Prediction,
    energy: bool,
    stress: bool,
    virial: bool,
) -> Result<(), String> {
    if energy {
        let value = prediction
            .energy
            .ok_or_else(|| "model did not return requested energy".to_owned())?;
        println!("energy = {value:.16e}");
    }
    if stress {
        let value = prediction
            .stress
            .ok_or_else(|| {
                "model did not return requested stress; the model may not support stress or the structure may lack full PBC"
                    .to_owned()
            })?;
        print_matrix("stress", "eV/angstrom^3", value);
    }
    if virial {
        let value = prediction
            .virial
            .ok_or_else(|| {
                "model did not return requested virial; the model may not support virial or the structure may lack full PBC"
                    .to_owned()
            })?;
        print_matrix("virial", "eV", value);
    }
    Ok(())
}

fn print_matrix(name: &str, unit: &str, value: [[f64; 3]; 3]) {
    println!("{name} ({unit}) =");
    for row in value {
        println!("  {:.16e} {:.16e} {:.16e}", row[0], row[1], row[2]);
    }
}

fn prediction_checksum(prediction: &Prediction) -> f64 {
    let mut checksum = prediction.energy.unwrap_or(0.0);
    if let Some(forces) = &prediction.forces {
        checksum += forces
            .iter()
            .flat_map(|force| force.iter())
            .copied()
            .sum::<f64>();
    }
    if let Some(stress) = prediction.stress {
        checksum += stress.iter().flatten().copied().sum::<f64>();
    }
    if let Some(virial) = prediction.virial {
        checksum += virial.iter().flatten().copied().sum::<f64>();
    }
    checksum
}

fn percentile(sorted: &[f64], percentile: f64) -> Result<f64, String> {
    if sorted.is_empty() {
        return Err("benchmark produced no samples".to_owned());
    }
    let index = ((percentile / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    Ok(sorted[index])
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1.0e3
}
