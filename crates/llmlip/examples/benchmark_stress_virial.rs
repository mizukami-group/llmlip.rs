use std::{
    env, fs,
    path::PathBuf,
    time::{Duration, Instant},
};

use llmlip::{
    read_extxyz_frames, LlmlipModel, MaceMh1WignerMode, PredictionComputeDtype, PredictionOptions,
    Structure,
};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    let model = LlmlipModel::load(&args.model)?;
    let structure = load_structure(&args.structure, args.frame_index)?;
    let options = PredictionOptions {
        compute_energy: args.compute_energy,
        compute_forces: args.compute_forces,
        compute_stress: args.compute_stress,
        compute_virial: args.compute_virial,
        selected_head: args.head.clone(),
        mace_mh1_wigner_mode: args.mace_mh1_wigner_mode,
        ..PredictionOptions::default()
    };
    let prepared = if args.prepared {
        Some(model.prepare_prediction_input_with_options(&structure, &options)?)
    } else {
        None
    };

    for _ in 0..args.warmup {
        black_box_prediction(run_prediction(
            &model,
            &structure,
            prepared.as_ref(),
            &options,
            args.compute_dtype,
        )?);
    }

    let mut samples_ms = Vec::with_capacity(args.iterations);
    let mut last_prediction = None;
    for _ in 0..args.iterations {
        let start = Instant::now();
        let prediction = run_prediction(
            &model,
            &structure,
            prepared.as_ref(),
            &options,
            args.compute_dtype,
        )?;
        samples_ms.push(duration_ms(start.elapsed()));
        last_prediction = Some(prediction);
    }

    let prediction = last_prediction.ok_or("no prediction samples recorded")?;
    let capabilities = model.capabilities();
    let force_max_abs = prediction.forces.as_ref().map(|forces| {
        forces
            .iter()
            .flatten()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
    });
    let stats = stats(&samples_ms);
    let output = json!({
        "schema_version": "llmlip-stress-virial-benchmark-v1",
        "model_path": args.model,
        "structure_path": args.structure,
        "frame_index": args.frame_index,
        "label": args.label,
        "architecture": capabilities.architecture,
        "capabilities": {
            "supports_energy": capabilities.supports_energy,
            "supports_forces": capabilities.supports_forces,
            "supports_stress": capabilities.supports_stress,
            "supports_virial": capabilities.supports_virial,
            "stress_requires_full_pbc": capabilities.stress_requires_full_pbc,
            "virial_requires_full_pbc": capabilities.virial_requires_full_pbc,
        },
        "options": {
            "prepared": args.prepared,
            "compute_energy": args.compute_energy,
            "compute_forces": args.compute_forces,
            "compute_stress": args.compute_stress,
            "compute_virial": args.compute_virial,
            "compute_dtype": format!("{:?}", args.compute_dtype),
            "selected_head": args.head,
            "mace_mh1_wigner_mode": format!("{:?}", args.mace_mh1_wigner_mode),
        },
        "structure": {
            "natoms": structure.len(),
            "symbols": structure.species,
            "positions_angstrom": structure.positions,
            "cell_angstrom": structure.cell.map(|cell| cell.vectors),
            "pbc": structure.pbc,
        },
        "outputs": {
            "energy_ev": prediction.energy,
            "forces_ev_per_angstrom": prediction.forces,
            "force_max_abs_ev_per_angstrom": force_max_abs,
            "stress_eV_per_angstrom3": prediction.stress,
            "virial_eV": prediction.virial,
        },
        "timing": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "samples_ms": samples_ms,
            "mean_ms": stats.mean_ms,
            "median_ms": stats.median_ms,
            "min_ms": stats.min_ms,
            "max_ms": stats.max_ms,
        },
        "env": {
            "RAYON_NUM_THREADS": env::var("RAYON_NUM_THREADS").ok(),
            "OMP_NUM_THREADS": env::var("OMP_NUM_THREADS").ok(),
            "MKL_NUM_THREADS": env::var("MKL_NUM_THREADS").ok(),
            "OPENBLAS_NUM_THREADS": env::var("OPENBLAS_NUM_THREADS").ok(),
            "VECLIB_MAXIMUM_THREADS": env::var("VECLIB_MAXIMUM_THREADS").ok(),
        },
    });

    let text = serde_json::to_string_pretty(&output)?;
    if let Some(out) = args.out {
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(out, text + "\n")?;
    } else {
        println!("{text}");
    }
    Ok(())
}

#[derive(Debug)]
struct Args {
    model: PathBuf,
    structure: PathBuf,
    frame_index: usize,
    label: Option<String>,
    head: Option<String>,
    warmup: usize,
    iterations: usize,
    prepared: bool,
    compute_energy: bool,
    compute_forces: bool,
    compute_stress: bool,
    compute_virial: bool,
    compute_dtype: PredictionComputeDtype,
    mace_mh1_wigner_mode: MaceMh1WignerMode,
    out: Option<PathBuf>,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut model = None;
        let mut structure = None;
        let mut frame_index = 0usize;
        let mut label = None;
        let mut head = None;
        let mut warmup = 1usize;
        let mut iterations = 3usize;
        let mut prepared = false;
        let mut compute_energy = false;
        let mut compute_forces = false;
        let mut compute_stress = false;
        let mut compute_virial = false;
        let mut saw_output_flag = false;
        let mut compute_dtype = PredictionComputeDtype::Float64;
        let mut mace_mh1_wigner_mode = MaceMh1WignerMode::TrueFloat64;
        let mut out = None;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => model = Some(PathBuf::from(next_value(&mut args, "--model")?)),
                "--structure" => {
                    structure = Some(PathBuf::from(next_value(&mut args, "--structure")?));
                }
                "--frame-index" => frame_index = next_value(&mut args, "--frame-index")?.parse()?,
                "--label" => label = Some(next_value(&mut args, "--label")?),
                "--head" => head = Some(next_value(&mut args, "--head")?),
                "--warmup" => warmup = next_value(&mut args, "--warmup")?.parse()?,
                "--iterations" => iterations = next_value(&mut args, "--iterations")?.parse()?,
                "--prepared" => prepared = true,
                "--energy" => {
                    compute_energy = true;
                    saw_output_flag = true;
                }
                "--forces" => {
                    compute_forces = true;
                    saw_output_flag = true;
                }
                "--stress" => {
                    compute_stress = true;
                    saw_output_flag = true;
                }
                "--virial" => {
                    compute_virial = true;
                    saw_output_flag = true;
                }
                "--compute-dtype" => {
                    compute_dtype = match next_value(&mut args, "--compute-dtype")?.as_str() {
                        "f64" | "float64" => PredictionComputeDtype::Float64,
                        "f32" | "float32" => PredictionComputeDtype::Float32,
                        other => return Err(format!("unknown compute dtype: {other}").into()),
                    };
                }
                "--mace-mh1-wigner-mode" => {
                    mace_mh1_wigner_mode =
                        match next_value(&mut args, "--mace-mh1-wigner-mode")?.as_str() {
                            "true-f64" | "true_float64" => MaceMh1WignerMode::TrueFloat64,
                            "upstream-artifact" | "upstream_artifact" => {
                                MaceMh1WignerMode::UpstreamArtifact
                            }
                            other => {
                                return Err(format!("unknown MACE-MH-1 Wigner mode: {other}").into())
                            }
                        };
                }
                "--out" => out = Some(PathBuf::from(next_value(&mut args, "--out")?)),
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument: {other}").into()),
            }
        }
        if iterations == 0 {
            return Err("--iterations must be greater than 0".into());
        }
        if !saw_output_flag {
            compute_energy = true;
            compute_forces = true;
            compute_stress = true;
            compute_virial = true;
        }
        Ok(Self {
            model: model.ok_or("--model is required")?,
            structure: structure.ok_or("--structure is required")?,
            frame_index,
            label,
            head,
            warmup,
            iterations,
            prepared,
            compute_energy,
            compute_forces,
            compute_stress,
            compute_virial,
            compute_dtype,
            mace_mh1_wigner_mode,
            out,
        })
    }
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("{name} requires a value").into())
}

fn print_help() {
    println!(
        "Usage: cargo run -p llmlip --example benchmark_stress_virial -- \\\n  --model MODEL.e3rs --structure STRUCTURE.extxyz [--frame-index N] \\\n  [--head HEAD] [--prepared] [--energy] [--forces] [--stress] [--virial] \\\n  [--compute-dtype f64|f32] \\\n  [--mace-mh1-wigner-mode true-f64|upstream-artifact] \\\n  [--warmup N] [--iterations N] [--out PATH]"
    );
}

fn load_structure(
    path: &PathBuf,
    frame_index: usize,
) -> Result<Structure, Box<dyn std::error::Error>> {
    let frames = read_extxyz_frames(path)?;
    let frame = frames.get(frame_index).ok_or_else(|| {
        format!(
            "frame index {frame_index} out of range for {}",
            path.display()
        )
    })?;
    Ok(frame.structure.clone())
}

fn run_prediction(
    model: &LlmlipModel,
    structure: &Structure,
    prepared: Option<&llmlip::PreparedPredictionInput>,
    options: &PredictionOptions,
    dtype: PredictionComputeDtype,
) -> Result<llmlip::Prediction, Box<dyn std::error::Error>> {
    Ok(if let Some(prepared) = prepared {
        model.predict_prepared_with_compute_dtype(prepared, options, dtype)?
    } else {
        model.predict_with_compute_dtype(structure, options, dtype)?
    })
}

fn black_box_prediction(prediction: llmlip::Prediction) {
    std::hint::black_box(prediction);
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[derive(Debug)]
struct Stats {
    mean_ms: f64,
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

fn stats(samples: &[f64]) -> Stats {
    assert!(!samples.is_empty(), "stats requires at least one sample");
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean_ms = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let median_ms = if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2])
    } else {
        sorted[sorted.len() / 2]
    };
    Stats {
        mean_ms,
        median_ms,
        min_ms: sorted[0],
        max_ms: sorted[sorted.len() - 1],
    }
}
