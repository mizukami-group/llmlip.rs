use llmlip::{read_extxyz, LlmlipModel, PredictionComputeDtype, PredictionOptions};
use std::{env, error::Error, time::Instant};

#[derive(Clone)]
struct Config {
    model_path: String,
    structures: Vec<(usize, String)>,
    iterations: usize,
    mode: Mode,
    head: Option<String>,
    dtype: PredictionComputeDtype,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Both,
    EnergyOnly,
    FullOnly,
}

impl Config {
    fn from_args() -> Result<Self, Box<dyn Error>> {
        let mut model_path =
            String::from("fixtures/mace/v0_3_16_4d2da09/models/mace_mp0b_runtime.e3rs");
        let mut structures: Vec<(usize, String)> = vec![
            (3, String::from("/tmp/input_3.extxyz")),
            (6, String::from("/tmp/input_6.extxyz")),
            (12, String::from("/tmp/input_complex.extxyz")),
        ];
        let mut iterations = 20usize;
        let mut mode = Mode::Both;
        let mut head = None;
        let mut dtype = PredictionComputeDtype::Float64;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => {
                    model_path = args.next().ok_or("--model requires a path")?.to_owned();
                }
                "--iter" => {
                    iterations = args
                        .next()
                        .ok_or("--iter requires a number")?
                        .parse()
                        .map_err(|_| "invalid --iter")?;
                }
                "--head" => {
                    head = Some(args.next().ok_or("--head requires a value")?);
                }
                "--dtype" => {
                    dtype = match args.next().ok_or("--dtype requires f32 or f64")?.as_str() {
                        "f32" => PredictionComputeDtype::Float32,
                        "f64" => PredictionComputeDtype::Float64,
                        _ => return Err("--dtype must be f32 or f64".into()),
                    };
                }
                "--structure" => {
                    let path = args.next().ok_or("--structure requires a path")?.to_owned();
                    structures.push((0, path));
                }
                "--only-structures" => {
                    structures.clear();
                    for path in args.by_ref() {
                        structures.push((0, path));
                    }
                    break;
                }
                "--energy-only" => {
                    mode = Mode::EnergyOnly;
                }
                "--full-only" => {
                    mode = Mode::FullOnly;
                }
                _ => {}
            }
        }

        Ok(Self {
            model_path,
            structures,
            iterations,
            mode,
            head,
            dtype,
        })
    }
}

fn time_prediction(
    model: &LlmlipModel,
    structure: &llmlip::Structure,
    options: PredictionOptions,
    iterations: usize,
    dtype: PredictionComputeDtype,
) -> Result<(f64, f64), Box<dyn Error>> {
    let mut checks = 0.0f64;
    let _warmup = model.predict_with_compute_dtype(structure, &options, dtype)?;

    let start = Instant::now();
    for _ in 0..iterations {
        let prediction = model.predict_with_compute_dtype(structure, &options, dtype)?;
        if let Some(energy) = prediction.energy {
            checks += energy;
        }
        if let Some(forces) = prediction.forces {
            checks += forces.iter().flatten().sum::<f64>();
        }
    }
    std::hint::black_box(checks);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok((elapsed_ms, checks))
}

fn main() -> Result<(), Box<dyn Error>> {
    let cfg = Config::from_args()?;

    let model = LlmlipModel::load(&cfg.model_path)?;

    for (atoms, path) in cfg.structures {
        let structure = read_extxyz(&path)?;
        if atoms != 0 && structure.len() != atoms {
            eprintln!(
                "warning: requested {atoms} atoms but file {path} has {}",
                structure.len()
            );
        }
        let atoms = structure.len();

        let options_energy = PredictionOptions {
            compute_energy: true,
            compute_forces: false,
            selected_head: cfg.head.clone(),
            ..PredictionOptions::default()
        };
        let options_both = PredictionOptions {
            compute_energy: true,
            compute_forces: true,
            selected_head: cfg.head.clone(),
            ..PredictionOptions::default()
        };

        let (energy_ms, energy_checksum) = if cfg.mode == Mode::FullOnly {
            (0.0, 0.0)
        } else {
            time_prediction(
                &model,
                &structure,
                options_energy,
                cfg.iterations,
                cfg.dtype,
            )?
        };
        let (full_ms, full_checksum) = if cfg.mode == Mode::EnergyOnly {
            (0.0, 0.0)
        } else {
            time_prediction(&model, &structure, options_both, cfg.iterations, cfg.dtype)?
        };

        let timed_ms = if cfg.mode == Mode::EnergyOnly {
            energy_ms
        } else {
            full_ms
        };
        let atoms_per_s =
            structure.len() as f64 * cfg.iterations as f64 / (timed_ms / 1000.0).max(1e-12);
        let force_share = if full_ms > energy_ms && cfg.mode == Mode::Both {
            ((full_ms - energy_ms) / full_ms) * 100.0
        } else {
            0.0
        };

        println!(
            "atoms={atoms} iters={iters} energy_ms={energy_ms:.3} full_ms={full_ms:.3} per_iter_full_ms={iter_ms:.3} atoms_per_s={aps:.1} force_share_pct={force_share:.1} checks={checksum:.6} checks_full={checksum_full:.6}",
            atoms = atoms,
            iters = cfg.iterations,
            energy_ms = energy_ms,
            full_ms = full_ms,
            iter_ms = full_ms / cfg.iterations as f64,
            aps = atoms_per_s,
            force_share = force_share,
            checksum = energy_checksum,
            checksum_full = full_checksum,
        );
    }

    Ok(())
}
