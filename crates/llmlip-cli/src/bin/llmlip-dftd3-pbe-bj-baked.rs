//! Self-contained DFTD3 PBE BJ inference binary.

use std::time::Instant;

use clap::{Parser, Subcommand};
use llmlip::{
    read_extxyz, write_extxyz_with_forces, Dftd3BjCoordinationModel, Dftd3BjParameters,
    Dftd3C6Reference, Dftd3CoordinationPairCoefficient, Dftd3CoordinationParameters,
    Dftd3CutoffSmoothing, Dftd3ElementCoefficient, Prediction, PredictionOptions, Structure,
};
use serde::{Deserialize, Serialize};

const FULL_TABLE_FIXTURE: &str =
    include_str!("../../../../fixtures/dftd3/v0_5_3/dftd3_bj_pbe_full_table_f64.json");
const MODEL_ID: &str = "dftd3-pbe-bj-full-table";
const PREDICTION_SCHEMA_VERSION: &str = "llmlip-baked-dftd3-prediction-v1";
const BENCHMARK_SCHEMA_VERSION: &str = "llmlip-baked-dftd3-benchmark-v1";

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Debug, Parser)]
#[command(name = "llmlip-dftd3-pbe-bj-baked")]
#[command(about = "Self-contained DFTD3 PBE BJ two-body inference binary")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Predict DFTD3 energy and/or derivatives.
    Predict {
        #[arg(long)]
        structure: String,
        #[arg(long)]
        energy: bool,
        #[arg(long)]
        forces: bool,
        #[arg(long = "forces-out", value_name = "PATH")]
        forces_out: Option<String>,
        #[arg(long)]
        stress: bool,
        #[arg(long)]
        virial: bool,
        #[arg(long)]
        json: bool,
    },
    /// Benchmark repeated load-once DFTD3 predictions.
    BenchmarkPredict {
        #[arg(long)]
        structure: String,
        #[arg(long)]
        energy: bool,
        #[arg(long)]
        forces: bool,
        #[arg(long)]
        stress: bool,
        #[arg(long)]
        virial: bool,
        #[arg(long, default_value_t = 3)]
        warmup: usize,
        #[arg(long, default_value_t = 10)]
        iterations: usize,
    },
}

#[derive(Debug, Deserialize)]
struct ReferenceFixtureSpec {
    rust_subset_model: Dftd3ModelSpec,
}

#[derive(Debug, Deserialize)]
struct Dftd3ModelSpec {
    parameters: Dftd3ParametersSpec,
    coordination: Dftd3CoordinationSpec,
    elements: Vec<Dftd3ElementSpec>,
    pair_coefficients: Vec<Dftd3PairSpec>,
    cutoff_smoothing: String,
}

#[derive(Debug, Deserialize)]
struct Dftd3ParametersSpec {
    s6: f64,
    s8: f64,
    a1: f64,
    a2_angstrom: f64,
    cutoff_angstrom: f64,
}

#[derive(Debug, Deserialize)]
struct Dftd3CoordinationSpec {
    k1: f64,
    k3: f64,
    cutoff_angstrom: f64,
}

#[derive(Debug, Deserialize)]
struct Dftd3ElementSpec {
    species: String,
    covalent_radius_angstrom: f64,
    r2r4_angstrom: f64,
}

#[derive(Debug, Deserialize)]
struct Dftd3PairSpec {
    species_a: String,
    species_b: String,
    references: Vec<Dftd3C6ReferenceSpec>,
}

#[derive(Debug, Deserialize)]
struct Dftd3C6ReferenceSpec {
    c6_ev_angstrom6: f64,
    coordination_a: f64,
    coordination_b: f64,
}

#[derive(Debug, Serialize)]
struct PredictionReport<'a> {
    schema_version: &'static str,
    model: &'static str,
    structure: &'a str,
    atom_count: usize,
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

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    match Cli::parse().command {
        Command::Predict {
            structure,
            energy,
            forces,
            forces_out,
            stress,
            virial,
            json,
        } => run_predict(&structure, energy, forces, forces_out, stress, virial, json),
        Command::BenchmarkPredict {
            structure,
            energy,
            forces,
            stress,
            virial,
            warmup,
            iterations,
        } => run_benchmark_predict(
            &structure, energy, forces, stress, virial, warmup, iterations,
        ),
    }
}

fn run_predict(
    structure_path: &str,
    energy: bool,
    forces: bool,
    forces_out: Option<String>,
    stress: bool,
    virial: bool,
    json: bool,
) -> Result<(), String> {
    let compute_forces = forces || forces_out.is_some();
    validate_requested_outputs(energy, compute_forces, stress, virial, "predict")?;
    let structure =
        read_extxyz(structure_path).map_err(|err| format!("failed to read structure: {err}"))?;
    let (model, load_runtime_ms) = load_baked_model()?;
    let options = PredictionOptions {
        compute_energy: energy,
        compute_forces,
        compute_stress: stress,
        compute_virial: virial,
        ..PredictionOptions::default()
    };
    let predict_start = Instant::now();
    let prediction = model
        .predict(&structure, &options)
        .map_err(|err| format!("prediction failed: {err}"))?;
    let predict_runtime_ms = elapsed_ms(predict_start);

    if let Some(path) = forces_out {
        let force_values = prediction
            .forces
            .as_ref()
            .ok_or_else(|| "DFTD3 model did not return requested forces".to_owned())?;
        write_extxyz_with_forces(&path, &structure, force_values, prediction.energy)
            .map_err(|err| format!("failed to write forces: {err}"))?;
    }

    if json {
        write_prediction_json(
            structure_path,
            &structure,
            prediction,
            load_runtime_ms,
            predict_runtime_ms,
        )
    } else {
        print_prediction_text(&prediction, energy, stress, virial)?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn run_benchmark_predict(
    structure_path: &str,
    energy: bool,
    forces: bool,
    stress: bool,
    virial: bool,
    warmup: usize,
    iterations: usize,
) -> Result<(), String> {
    validate_requested_outputs(energy, forces, stress, virial, "benchmark-predict")?;
    if iterations == 0 {
        return Err("--iterations must be at least 1".to_owned());
    }
    let structure =
        read_extxyz(structure_path).map_err(|err| format!("failed to read structure: {err}"))?;
    let (model, load_runtime_ms) = load_baked_model()?;
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
            .predict(&structure, &options)
            .map_err(|err| format!("warmup prediction failed: {err}"))?;
        checksum += prediction_checksum(&prediction);
    }

    let mut samples_ms = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let prediction = model
            .predict(&structure, &options)
            .map_err(|err| format!("benchmark prediction failed: {err}"))?;
        samples_ms.push(elapsed_ms(start));
        checksum += prediction_checksum(&prediction);
    }
    let mut sorted = samples_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let mean_ms = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;
    let report = BenchmarkReport {
        schema_version: BENCHMARK_SCHEMA_VERSION,
        model: MODEL_ID,
        structure: structure_path,
        atom_count: structure.len(),
        requested_outputs: RequestedOutputs {
            energy,
            forces,
            stress,
            virial,
        },
        warmup,
        iterations,
        load_runtime_ms,
        mean_ms,
        median_ms: percentile(&sorted, 50.0),
        p95_ms: percentile(&sorted, 95.0),
        min_ms: sorted[0],
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

fn load_baked_model() -> Result<(Dftd3BjCoordinationModel, f64), String> {
    let start = Instant::now();
    let fixture: ReferenceFixtureSpec = serde_json::from_str(FULL_TABLE_FIXTURE)
        .map_err(|err| format!("failed to parse baked DFTD3 fixture: {err}"))?;
    let spec = fixture.rust_subset_model;
    let parameters = Dftd3BjParameters::new(
        spec.parameters.s6,
        spec.parameters.s8,
        spec.parameters.a1,
        spec.parameters.a2_angstrom,
        spec.parameters.cutoff_angstrom,
    );
    let coordination = Dftd3CoordinationParameters::new(
        spec.coordination.k1,
        spec.coordination.k3,
        spec.coordination.cutoff_angstrom,
    );
    let elements = spec
        .elements
        .into_iter()
        .map(|element| {
            Dftd3ElementCoefficient::new(
                element.species,
                element.covalent_radius_angstrom,
                element.r2r4_angstrom,
            )
        })
        .collect();
    let pair_coefficients = spec
        .pair_coefficients
        .into_iter()
        .map(|pair| {
            let references = pair
                .references
                .into_iter()
                .map(|reference| {
                    Dftd3C6Reference::new(
                        reference.c6_ev_angstrom6,
                        reference.coordination_a,
                        reference.coordination_b,
                    )
                })
                .collect();
            Dftd3CoordinationPairCoefficient::new(pair.species_a, pair.species_b, references)
        })
        .collect();
    let smoothing = match spec.cutoff_smoothing.as_str() {
        "none" => Dftd3CutoffSmoothing::None,
        "cosine" => Dftd3CutoffSmoothing::Cosine,
        other => return Err(format!("unsupported DFTD3 cutoff_smoothing {other:?}")),
    };
    let model =
        Dftd3BjCoordinationModel::new(parameters, coordination, elements, pair_coefficients)
            .map_err(|err| format!("failed to construct baked DFTD3 model: {err}"))?
            .with_cutoff_smoothing(smoothing);
    Ok((model, elapsed_ms(start)))
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
    load_runtime_ms: f64,
    predict_runtime_ms: f64,
) -> Result<(), String> {
    let report = PredictionReport {
        schema_version: PREDICTION_SCHEMA_VERSION,
        model: MODEL_ID,
        structure: structure_path,
        atom_count: structure.len(),
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
            .ok_or_else(|| "DFTD3 model did not return requested energy".to_owned())?;
        println!("energy = {value:.16e}");
    }
    if stress {
        let value = prediction
            .stress
            .ok_or_else(|| "DFTD3 model did not return requested stress".to_owned())?;
        print_matrix("stress", "eV/angstrom^3", value);
    }
    if virial {
        let value = prediction
            .virial
            .ok_or_else(|| "DFTD3 model did not return requested virial".to_owned())?;
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

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let index = ((percentile / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[index]
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1.0e3
}
