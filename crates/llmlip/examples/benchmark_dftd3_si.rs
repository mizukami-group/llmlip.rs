use std::{env, hint::black_box, path::PathBuf, process, time::Instant};

use llmlip::{
    neighbor::cell_list_multi_image_neighbor_list, read_extxyz, Cell, Dftd3BjCoordinationModel,
    Dftd3BjParameters, Dftd3C6Reference, Dftd3CoordinationPairCoefficient,
    Dftd3CoordinationParameters, Dftd3CutoffSmoothing, Dftd3ElementCoefficient, PredictionOptions,
    Structure,
};
use serde_json::{json, Value};

const DEFAULT_DFTD3_FIXTURE: &str = "fixtures/dftd3/v0_5_3/dftd3_bj_pbe_full_table_f64.json";

#[derive(Debug)]
struct Args {
    dftd3_fixture: PathBuf,
    structure: Option<PathBuf>,
    diamond: Option<[usize; 3]>,
    lattice: f64,
    cell_skew: [f64; 3],
    cutoff: Option<f64>,
    coordination_cutoff: Option<f64>,
    cutoff_smoothing: Dftd3CutoffSmoothing,
    warmup: usize,
    iterations: usize,
    forces: bool,
    stress: bool,
    virial: bool,
    measure_neighbor_build: bool,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    if args.iterations == 0 {
        return Err("--iterations must be at least 1".to_owned());
    }

    let model_json = read_json(&args.dftd3_fixture)?;
    let model = load_full_table_model(
        &model_json["rust_subset_model"],
        args.cutoff,
        args.coordination_cutoff.or(args.cutoff),
        args.cutoff_smoothing,
    )?;
    let structure = if let Some(path) = &args.structure {
        read_extxyz(path).map_err(|err| format!("failed to read {}: {err}", path.display()))?
    } else {
        let repeats = args
            .diamond
            .ok_or_else(|| "provide --structure or --diamond NX NY NZ".to_owned())?;
        diamond_supercell(repeats, args.lattice, args.cell_skew)
    };

    let options = PredictionOptions {
        compute_energy: true,
        compute_forces: args.forces,
        compute_stress: args.stress,
        compute_virial: args.virial,
        ..PredictionOptions::default()
    };
    let neighbor_cutoff = model.required_neighbor_cutoff();

    let mut checksum = 0.0;
    for _ in 0..args.warmup {
        if args.measure_neighbor_build {
            let neighbors =
                cell_list_multi_image_neighbor_list(black_box(&structure), neighbor_cutoff)
                    .map_err(|err| format!("neighbor warmup failed: {err}"))?;
            checksum += neighbors.len() as f64;
        }
        let prediction = model
            .predict(black_box(&structure), black_box(&options))
            .map_err(|err| format!("DFTD3 warmup failed: {err}"))?;
        checksum += prediction.energy.unwrap_or(0.0);
    }

    let mut neighbor_ms = Vec::with_capacity(args.iterations);
    let mut predict_ms = Vec::with_capacity(args.iterations);
    let mut edge_count = 0usize;
    for _ in 0..args.iterations {
        if args.measure_neighbor_build {
            let start = Instant::now();
            let neighbors =
                cell_list_multi_image_neighbor_list(black_box(&structure), neighbor_cutoff)
                    .map_err(|err| format!("neighbor build failed: {err}"))?;
            neighbor_ms.push(elapsed_ms(start));
            edge_count = neighbors.len();
            checksum += edge_count as f64;
        }

        let start = Instant::now();
        let prediction = model
            .predict(black_box(&structure), black_box(&options))
            .map_err(|err| format!("DFTD3 prediction failed: {err}"))?;
        predict_ms.push(elapsed_ms(start));
        checksum += prediction.energy.unwrap_or(0.0);
        if let Some(forces) = prediction.forces {
            checksum += forces
                .iter()
                .flat_map(|force| force.iter())
                .map(|value| value.abs())
                .sum::<f64>();
        }
        if let Some(stress) = prediction.stress {
            checksum += stress
                .iter()
                .flatten()
                .map(|value| value.abs())
                .sum::<f64>();
        }
        if let Some(virial) = prediction.virial {
            checksum += virial
                .iter()
                .flatten()
                .map(|value| value.abs())
                .sum::<f64>();
        }
    }

    let report = json!({
        "schema_version": "llmlip-dftd3-si-benchmark-v1",
        "atom_count": structure.len(),
        "cell": structure.cell.map(|cell| cell.vectors),
        "pbc": structure.pbc,
        "dftd3_fixture": args.dftd3_fixture,
        "dftd3_cutoff_angstrom": model.parameters().cutoff,
        "coordination_cutoff_angstrom": model.coordination_parameters().cutoff,
        "cutoff_smoothing": format!("{:?}", model.cutoff_smoothing()),
        "neighbor_cutoff_angstrom": neighbor_cutoff,
        "edge_count": if args.measure_neighbor_build { json!(edge_count) } else { Value::Null },
        "forces": args.forces,
        "stress": args.stress,
        "virial": args.virial,
        "measure_neighbor_build": args.measure_neighbor_build,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "neighbor_build_ms": if neighbor_ms.is_empty() { Value::Null } else { summarize(&neighbor_ms) },
        "predict_total_ms": summarize(&predict_ms),
        "predict_minus_neighbor_median_ms": if neighbor_ms.is_empty() {
            Value::Null
        } else {
            json!(percentile(&predict_ms, 50.0) - percentile(&neighbor_ms, 50.0))
        },
        "checksum": black_box(checksum),
        "thread_env": thread_env(),
    });
    serde_json::to_writer_pretty(std::io::stdout(), &report)
        .map_err(|err| format!("failed to write JSON report: {err}"))?;
    println!();
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut args = env::args().skip(1);
    let mut parsed = Args {
        dftd3_fixture: PathBuf::from(DEFAULT_DFTD3_FIXTURE),
        structure: None,
        diamond: None,
        lattice: 5.43,
        cell_skew: [0.0; 3],
        cutoff: None,
        coordination_cutoff: None,
        cutoff_smoothing: Dftd3CutoffSmoothing::None,
        warmup: 3,
        iterations: 10,
        forces: true,
        stress: false,
        virial: false,
        measure_neighbor_build: true,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dftd3-fixture" => {
                parsed.dftd3_fixture = PathBuf::from(next_value("--dftd3-fixture", &mut args)?);
            }
            "--structure" => {
                parsed.structure = Some(PathBuf::from(next_value("--structure", &mut args)?));
            }
            "--diamond" => {
                parsed.diamond = Some([
                    parse_usize("--diamond NX", args.next())?,
                    parse_usize("--diamond NY", args.next())?,
                    parse_usize("--diamond NZ", args.next())?,
                ]);
            }
            "--lattice" => parsed.lattice = parse_f64("--lattice", args.next())?,
            "--cell-skew" => {
                parsed.cell_skew = [
                    parse_f64("--cell-skew XY", args.next())?,
                    parse_f64("--cell-skew XZ", args.next())?,
                    parse_f64("--cell-skew YZ", args.next())?,
                ];
            }
            "--cutoff" => parsed.cutoff = Some(parse_f64("--cutoff", args.next())?),
            "--coordination-cutoff" => {
                parsed.coordination_cutoff = Some(parse_f64("--coordination-cutoff", args.next())?);
            }
            "--cutoff-smoothing" => {
                parsed.cutoff_smoothing =
                    parse_cutoff_smoothing(&next_value("--cutoff-smoothing", &mut args)?)?;
            }
            "--warmup" => parsed.warmup = parse_usize("--warmup", args.next())?,
            "--iterations" => parsed.iterations = parse_usize("--iterations", args.next())?,
            "--energy-only" => parsed.forces = false,
            "--forces" => parsed.forces = true,
            "--stress" => parsed.stress = true,
            "--no-stress" => parsed.stress = false,
            "--virial" => parsed.virial = true,
            "--no-virial" => parsed.virial = false,
            "--skip-neighbor-build" => parsed.measure_neighbor_build = false,
            "--measure-neighbor-build" => parsed.measure_neighbor_build = true,
            "--help" | "-h" => return Err(usage()),
            other => return Err(format!("unknown argument {other:?}\n{}", usage())),
        }
    }
    if parsed.structure.is_some() && parsed.diamond.is_some() {
        return Err("provide only one of --structure or --diamond".to_owned());
    }
    Ok(parsed)
}

fn usage() -> String {
    "Usage: cargo run --release -p llmlip --example benchmark_dftd3_si -- \\
       [--structure STRUCTURE.extxyz | --diamond NX NY NZ] [--cutoff ANG] \\
       [--cell-skew XY XZ YZ] [--coordination-cutoff ANG] [--cutoff-smoothing none|cosine] \\
       [--warmup N] [--iterations N] [--energy-only] [--stress] [--virial] \\
       [--skip-neighbor-build]"
        .to_owned()
}

fn next_value(
    name: &'static str,
    args: &mut impl Iterator<Item = String>,
) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{name} requires a value"))
}

fn parse_usize(name: &'static str, value: Option<String>) -> Result<usize, String> {
    let value = value.ok_or_else(|| format!("{name} requires a value"))?;
    value
        .parse::<usize>()
        .map_err(|_| format!("{name} must be an unsigned integer, got {value:?}"))
}

fn parse_f64(name: &'static str, value: Option<String>) -> Result<f64, String> {
    let value = value.ok_or_else(|| format!("{name} requires a value"))?;
    value
        .parse::<f64>()
        .map_err(|_| format!("{name} must be a finite number, got {value:?}"))
        .and_then(|parsed| {
            if parsed.is_finite() {
                Ok(parsed)
            } else {
                Err(format!("{name} must be finite, got {value:?}"))
            }
        })
}

fn parse_cutoff_smoothing(value: &str) -> Result<Dftd3CutoffSmoothing, String> {
    match value {
        "none" => Ok(Dftd3CutoffSmoothing::None),
        "cosine" => Ok(Dftd3CutoffSmoothing::Cosine),
        other => Err(format!(
            "--cutoff-smoothing must be none or cosine, got {other:?}"
        )),
    }
}

fn read_json(path: &PathBuf) -> Result<Value, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    serde_json::from_str(&text).map_err(|err| format!("failed to parse {}: {err}", path.display()))
}

fn value_f64(value: &Value, path: &str) -> Result<f64, String> {
    value
        .as_f64()
        .ok_or_else(|| format!("{path} should be f64"))
}

fn value_str(value: &Value, path: &str) -> Result<String, String> {
    value
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{path} should be string"))
}

fn load_full_table_model(
    model: &Value,
    cutoff_override: Option<f64>,
    coordination_cutoff_override: Option<f64>,
    cutoff_smoothing: Dftd3CutoffSmoothing,
) -> Result<Dftd3BjCoordinationModel, String> {
    let parameters = Dftd3BjParameters::new(
        value_f64(&model["parameters"]["s6"], "s6")?,
        value_f64(&model["parameters"]["s8"], "s8")?,
        value_f64(&model["parameters"]["a1"], "a1")?,
        value_f64(&model["parameters"]["a2_angstrom"], "a2")?,
        cutoff_override.unwrap_or(value_f64(
            &model["parameters"]["cutoff_angstrom"],
            "cutoff",
        )?),
    );
    let coordination = Dftd3CoordinationParameters::new(
        value_f64(&model["coordination"]["k1"], "k1")?,
        value_f64(&model["coordination"]["k3"], "k3")?,
        coordination_cutoff_override.unwrap_or(value_f64(
            &model["coordination"]["cutoff_angstrom"],
            "coordination_cutoff",
        )?),
    );
    let elements = model["elements"]
        .as_array()
        .ok_or_else(|| "elements should be an array".to_owned())?
        .iter()
        .map(|element| {
            Ok(Dftd3ElementCoefficient::new(
                value_str(&element["species"], "species")?,
                value_f64(&element["covalent_radius_angstrom"], "covalent_radius")?,
                value_f64(&element["r2r4_angstrom"], "r2r4")?,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let pair_coefficients = model["pair_coefficients"]
        .as_array()
        .ok_or_else(|| "pair_coefficients should be an array".to_owned())?
        .iter()
        .map(|pair| {
            let references = pair["references"]
                .as_array()
                .ok_or_else(|| "references should be an array".to_owned())?
                .iter()
                .map(|reference| {
                    Ok(Dftd3C6Reference::new(
                        value_f64(&reference["c6_ev_angstrom6"], "c6")?,
                        value_f64(&reference["coordination_a"], "coordination_a")?,
                        value_f64(&reference["coordination_b"], "coordination_b")?,
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(Dftd3CoordinationPairCoefficient::new(
                value_str(&pair["species_a"], "species_a")?,
                value_str(&pair["species_b"], "species_b")?,
                references,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;

    Dftd3BjCoordinationModel::new(parameters, coordination, elements, pair_coefficients)
        .map(|model| model.with_cutoff_smoothing(cutoff_smoothing))
        .map_err(|err| format!("failed to construct DFTD3 model: {err}"))
}

fn diamond_supercell(repeats: [usize; 3], lattice: f64, cell_skew: [f64; 3]) -> Structure {
    let basis = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ];
    let atom_count = repeats[0] * repeats[1] * repeats[2] * basis.len();
    let cell = Cell::from_vectors(
        [repeats[0] as f64 * lattice, 0.0, 0.0],
        [cell_skew[0], repeats[1] as f64 * lattice, 0.0],
        [cell_skew[1], cell_skew[2], repeats[2] as f64 * lattice],
    );
    let mut species = Vec::with_capacity(atom_count);
    let mut positions = Vec::with_capacity(atom_count);
    for ix in 0..repeats[0] {
        for iy in 0..repeats[1] {
            for iz in 0..repeats[2] {
                for offset in basis {
                    species.push("Si".to_owned());
                    positions.push(cell.fractional_to_cartesian([
                        (ix as f64 + offset[0]) / repeats[0] as f64,
                        (iy as f64 + offset[1]) / repeats[1] as f64,
                        (iz as f64 + offset[2]) / repeats[2] as f64,
                    ]));
                }
            }
        }
    }
    Structure::new(species, positions).with_cell(cell, [true, true, true])
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

fn summarize(samples: &[f64]) -> Value {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    json!({
        "mean": mean,
        "median": percentile_sorted(&sorted, 50.0),
        "p95": percentile_sorted(&sorted, 95.0),
        "min": sorted[0],
        "max": sorted[sorted.len() - 1],
        "samples": samples,
    })
}

fn percentile(samples: &[f64], pct: f64) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    percentile_sorted(&sorted, pct)
}

fn percentile_sorted(sorted: &[f64], pct: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (pct / 100.0) * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = rank - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

fn thread_env() -> Value {
    json!({
        "RAYON_NUM_THREADS": env::var("RAYON_NUM_THREADS").ok(),
        "OMP_NUM_THREADS": env::var("OMP_NUM_THREADS").ok(),
        "OPENBLAS_NUM_THREADS": env::var("OPENBLAS_NUM_THREADS").ok(),
        "MKL_NUM_THREADS": env::var("MKL_NUM_THREADS").ok(),
    })
}
