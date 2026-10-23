use std::{
    env,
    error::Error,
    hint::black_box,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use llmlip::{
    all_pairs_multi_image_neighbor_list, all_pairs_neighbor_list, read_extxyz, LlmlipModel,
    Prediction, PredictionOptions, Structure,
};
use serde_json::{json, Value};

struct Args {
    warmup: usize,
    iterations: usize,
}

struct Case {
    label: &'static str,
    path: &'static str,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut warmup = 1_usize;
    let mut iterations = 5_usize;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--warmup" => {
                let Some(value) = args.next() else {
                    return Err("--warmup requires a value".into());
                };
                warmup = value.parse()?;
            }
            "--iterations" => {
                let Some(value) = args.next() else {
                    return Err("--iterations requires a value".into());
                };
                iterations = value.parse()?;
            }
            "--help" | "-h" => {
                println!("Usage: cargo run -p llmlip --example nequip_oam_l_timing -- [--warmup N] [--iterations N]");
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    if iterations == 0 {
        return Err("--iterations must be greater than zero".into());
    }
    Ok(Args { warmup, iterations })
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn stats(samples: &[Duration]) -> Value {
    let mut ms = samples
        .iter()
        .map(|sample| duration_ms(*sample))
        .collect::<Vec<_>>();
    ms.sort_by(f64::total_cmp);
    let len = ms.len();
    let mean = ms.iter().sum::<f64>() / len as f64;
    let median = if len % 2 == 0 {
        (ms[(len / 2) - 1] + ms[len / 2]) * 0.5
    } else {
        ms[len / 2]
    };
    let p95_index = (((len as f64) * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(len - 1);
    json!({
        "iterations": len,
        "mean_ms": mean,
        "median_ms": median,
        "p95_ms": ms[p95_index],
        "min_ms": ms[0],
        "max_ms": ms[len - 1],
        "samples_ms": ms,
    })
}

fn prediction_checksum(prediction: &Prediction) -> f64 {
    let energy = prediction.energy.unwrap_or_default();
    let force_sum = prediction
        .forces
        .as_ref()
        .map(|forces| {
            forces
                .iter()
                .flat_map(|force| force.iter())
                .copied()
                .sum::<f64>()
        })
        .unwrap_or_default();
    energy + force_sum
}

fn time_repeated<F>(
    warmup: usize,
    iterations: usize,
    mut func: F,
) -> Result<(Value, f64), Box<dyn Error>>
where
    F: FnMut() -> Result<f64, Box<dyn Error>>,
{
    let mut checksum = 0.0;
    for _ in 0..warmup {
        checksum += black_box(func()?);
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        checksum += black_box(func()?);
        samples.push(start.elapsed());
    }
    Ok((stats(&samples), checksum))
}

fn neighbor_list_checksum(structure: &Structure) -> Result<f64, Box<dyn Error>> {
    let has_pbc = structure.pbc.iter().any(|enabled| *enabled);
    let list = if has_pbc {
        all_pairs_multi_image_neighbor_list(structure, 6.0)?
    } else {
        all_pairs_neighbor_list(structure, 6.0)?
    };
    Ok(list.len() as f64)
}

fn median_ms(value: &Value) -> Option<f64> {
    value.get("median_ms").and_then(Value::as_f64)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let root = repo_root();
    let model_path = root.join("fixtures/nequip/v0_17_1_604ce73b/models/nequip_oam_l_prefix.e3rs");
    let cases = [
        Case {
            label: "benzo_a_pyrene_32atom",
            path: "fixtures/nequip/structures/benzo_a_pyrene.extxyz",
        },
        Case {
            label: "polycyclic_c32h32_64atom",
            path: "fixtures/nequip/structures/polycyclic_c32h32.extxyz",
        },
        Case {
            label: "polycyclic_c32h32_dimer_128atom",
            path: "fixtures/nequip/structures/polycyclic_c32h32_dimer_separated_128.extxyz",
        },
        Case {
            label: "pubchem_91109058_128atom",
            path: "fixtures/nequip/structures/1_7_3_ethyl_2_methyl_3_1_methylcycloundecyl_cycloheptyl_3_methyl_91109058.extxyz",
        },
        Case {
            label: "pubchem_129755176_128atom",
            path: "fixtures/nequip/structures/5s_8s_9s_10s_13r_14s_17s_17_ethyl_1_2_5s_8s_9s_10s_13r_14s_17s_1_129755176.extxyz",
        },
    ];

    let model_start = Instant::now();
    let model = LlmlipModel::load(&model_path)?;
    let model_load_ms = duration_ms(model_start.elapsed());
    let options = PredictionOptions {
        compute_energy: true,
        compute_forces: true,
        compute_stress: false,
        compute_virial: false,
        ..PredictionOptions::default()
    };

    let mut case_results = Vec::with_capacity(cases.len());
    for case in cases {
        let structure_start = Instant::now();
        let structure = read_extxyz(root.join(case.path))?;
        let structure_load_ms = duration_ms(structure_start.elapsed());

        let (neighbor_list_stats, neighbor_list_checksum) =
            time_repeated(args.warmup, args.iterations, || {
                neighbor_list_checksum(&structure)
            })?;

        let first_predict_start = Instant::now();
        let first_prediction = model.predict(&structure, &options)?;
        let first_predict_ms = duration_ms(first_predict_start.elapsed());
        let first_predict_checksum = prediction_checksum(&first_prediction);

        let (predict_stats, predict_checksum) =
            time_repeated(args.warmup, args.iterations, || {
                let prediction = model.predict(&structure, &options)?;
                Ok(prediction_checksum(&prediction))
            })?;

        let estimated_model_eval_minus_neighbor_list_ms =
            match (median_ms(&predict_stats), median_ms(&neighbor_list_stats)) {
                (Some(predict), Some(neighbor_list)) => Some((predict - neighbor_list).max(0.0)),
                _ => None,
            };

        case_results.push(json!({
            "label": case.label,
            "structure_path": case.path,
            "atom_count": structure.positions.len(),
            "structure_load_ms": structure_load_ms,
            "neighbor_list": {
                "checksum": neighbor_list_checksum,
                "timing": neighbor_list_stats,
            },
            "first_energy_force_predict_ms": first_predict_ms,
            "first_energy_force_checksum": first_predict_checksum,
            "steady_energy_force_predict": {
                "checksum": predict_checksum,
                "timing": predict_stats,
                "scope": "LlmlipModel::predict; includes Rust neighbor-list construction and energy+force runtime, excludes process startup, model load, and structure load",
            },
            "estimated_energy_force_minus_neighbor_list_median_ms": estimated_model_eval_minus_neighbor_list_ms,
        }));
    }

    let result = json!({
        "backend": "Rust NeQUIP-OAM-L .e3rs CPU",
        "model_path": model_path,
        "model_load_ms": model_load_ms,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "cases": case_results,
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
