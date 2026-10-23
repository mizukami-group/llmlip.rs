use std::{
    env, fs,
    path::PathBuf,
    time::{Duration, Instant},
};

use llmlip::{LlmlipModel, PredictionOptions, Structure};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde_json::{json, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    let model = LlmlipModel::load(&args.model)?;
    let fixture = read_json(&args.fixture)?;
    let structures = structures_from_fixture(&fixture)?;
    let options = PredictionOptions {
        compute_energy: true,
        compute_forces: args.forces,
        compute_stress: false,
        compute_virial: false,
        selected_head: None,
        ..PredictionOptions::default()
    };

    let mut checksum = 0.0;
    for _ in 0..args.warmup {
        checksum += run_pass(&model, &structures, &options)?;
    }
    let mut samples_ms = Vec::with_capacity(args.iterations);
    for _ in 0..args.iterations {
        let start = Instant::now();
        checksum += run_pass(&model, &structures, &options)?;
        samples_ms.push(duration_ms(start.elapsed()));
    }

    let stats = stats(&samples_ms);
    let output = json!({
        "schema_version": "nequix-rust-benchmark-v1",
        "model": args.model,
        "fixture": args.fixture,
        "structure_count": structures.len(),
        "forces": args.forces,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "checksum": checksum,
        "mean_ms": stats.mean_ms,
        "median_ms": stats.median_ms,
        "p95_ms": stats.p95_ms,
        "min_ms": stats.min_ms,
        "max_ms": stats.max_ms,
        "samples_ms": samples_ms,
        "env": {
            "RAYON_NUM_THREADS": env::var("RAYON_NUM_THREADS").ok(),
            "OMP_NUM_THREADS": env::var("OMP_NUM_THREADS").ok(),
            "MKL_NUM_THREADS": env::var("MKL_NUM_THREADS").ok(),
            "OPENBLAS_NUM_THREADS": env::var("OPENBLAS_NUM_THREADS").ok(),
            "VECLIB_MAXIMUM_THREADS": env::var("VECLIB_MAXIMUM_THREADS").ok(),
        },
    });
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

struct Args {
    model: PathBuf,
    fixture: PathBuf,
    warmup: usize,
    iterations: usize,
    forces: bool,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut model = None;
        let mut fixture = None;
        let mut warmup = 5usize;
        let mut iterations = 20usize;
        let mut forces = true;
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => model = Some(PathBuf::from(next_value(&mut args, "--model")?)),
                "--fixture" => fixture = Some(PathBuf::from(next_value(&mut args, "--fixture")?)),
                "--warmup" => warmup = next_value(&mut args, "--warmup")?.parse()?,
                "--iterations" => iterations = next_value(&mut args, "--iterations")?.parse()?,
                "--energy-only" => forces = false,
                "--forces" => forces = true,
                other => return Err(format!("unknown argument {other}").into()),
            }
        }
        if iterations == 0 {
            return Err("--iterations must be greater than 0".into());
        }

        Ok(Self {
            model: model.ok_or("--model is required")?,
            fixture: fixture.ok_or("--fixture is required")?,
            warmup,
            iterations,
            forces,
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

fn read_json(path: &PathBuf) -> Result<Value, Box<dyn std::error::Error>> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn structures_from_fixture(value: &Value) -> Result<Vec<Structure>, Box<dyn std::error::Error>> {
    let cases = value["structures"]
        .as_array()
        .ok_or("fixture must contain structures array")?;
    cases
        .iter()
        .map(|case| {
            let structure = &case["structure"];
            let symbols = structure["symbols"]
                .as_array()
                .ok_or("structure must contain symbols")?
                .iter()
                .map(|symbol| {
                    symbol
                        .as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| "symbol must be a string".into())
                })
                .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
            let positions = structure["positions_angstrom"]
                .as_array()
                .ok_or("structure must contain positions_angstrom")?
                .iter()
                .map(|row| {
                    let row = row.as_array().ok_or("position row must be an array")?;
                    if row.len() != 3 {
                        return Err("position row must have length 3".into());
                    }
                    Ok([
                        row[0].as_f64().ok_or("position x must be f64")?,
                        row[1].as_f64().ok_or("position y must be f64")?,
                        row[2].as_f64().ok_or("position z must be f64")?,
                    ])
                })
                .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
            Ok(Structure::new(symbols, positions))
        })
        .collect()
}

fn run_pass(
    model: &LlmlipModel,
    structures: &[Structure],
    options: &PredictionOptions,
) -> Result<f64, Box<dyn std::error::Error>> {
    #[cfg(feature = "parallel")]
    {
        structures
            .par_iter()
            .map(|structure| {
                let prediction = model
                    .predict(structure, options)
                    .map_err(|err| err.to_string())?;
                let mut checksum = prediction.energy.unwrap_or(0.0);
                if let Some(forces) = prediction.forces {
                    checksum += forces.iter().flatten().sum::<f64>();
                }
                Ok::<f64, String>(checksum)
            })
            .try_reduce(|| 0.0, |left, right| Ok::<f64, String>(left + right))
            .map_err(Into::into)
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut checksum = 0.0;
        for structure in structures {
            let prediction = model.predict(structure, options)?;
            if let Some(energy) = prediction.energy {
                checksum += energy;
            }
            if let Some(forces) = prediction.forces {
                checksum += forces.iter().flatten().sum::<f64>();
            }
        }
        Ok(checksum)
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

struct Stats {
    mean_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

fn stats(samples: &[f64]) -> Stats {
    assert!(!samples.is_empty(), "stats requires at least one sample");
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean_ms = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let median_ms = if sorted.len().is_multiple_of(2) {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
    } else {
        sorted[sorted.len() / 2]
    };
    let p95_index = ((sorted.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Stats {
        mean_ms,
        median_ms,
        p95_ms: sorted[p95_index],
        min_ms: sorted[0],
        max_ms: sorted[sorted.len() - 1],
    }
}
