#![allow(clippy::expect_used, clippy::panic)]

#[path = "../tests/common/mod.rs"]
mod common;

use std::{path::PathBuf, time::Instant};

use common::{
    ase_si_diamond_cubic_8, ase_si_diamond_cubic_8_perturbed, hea_bcc, hea_bcc_perturbed,
};
use llmlip::{load_llmlip_model, LlmlipModel, PredictionComputeDtype, PredictionOptions};

fn main() {
    let mut model_dir =
        PathBuf::from("outputs/reference-artifacts/fairchem/uma_s_1p2_state_export.e3rs");
    let mut warmup = 5usize;
    let mut repeat = 20usize;
    let mut structure_key = "perturbed".to_owned();
    let mut use_prepared = false;
    let mut dtype = PredictionComputeDtype::Float64;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-dir" => model_dir = PathBuf::from(args.next().expect("--model-dir value")),
            "--warmup" => {
                warmup = args
                    .next()
                    .expect("--warmup value")
                    .parse()
                    .expect("warmup usize")
            }
            "--repeat" => {
                repeat = args
                    .next()
                    .expect("--repeat value")
                    .parse()
                    .expect("repeat usize")
            }
            "--structure" => structure_key = args.next().expect("--structure value"),
            "--prepared" => use_prepared = true,
            "--dtype" => {
                dtype = match args.next().expect("--dtype value").as_str() {
                    "f32" | "float32" => PredictionComputeDtype::Float32,
                    "f64" | "float64" => PredictionComputeDtype::Float64,
                    other => panic!("unknown dtype {other}"),
                }
            }
            other => panic!("unknown argument {other}"),
        }
    }

    let model = load_llmlip_model(&model_dir).expect("load UMA-S-1.2 model");
    let LlmlipModel::Uma(model) = model else {
        panic!("expected UMA model");
    };
    let structure = match structure_key.as_str() {
        "builtin" => ase_si_diamond_cubic_8(),
        "perturbed" => ase_si_diamond_cubic_8_perturbed(),
        "hea128" => hea_bcc(4),
        "hea128p" => hea_bcc_perturbed(4),
        "hea432" => hea_bcc(6),
        "hea432p" => hea_bcc_perturbed(6),
        other => panic!("unknown structure {other}"),
    };
    let options = PredictionOptions {
        compute_energy: true,
        compute_forces: true,
        ..PredictionOptions::default()
    };
    #[cfg(feature = "parallel")]
    let rayon_threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let rayon_threads = 1usize;

    let dtype_str = match dtype {
        PredictionComputeDtype::Float32 => "f32",
        PredictionComputeDtype::Float64 => "f64",
    };

    let prepared = if use_prepared {
        Some(
            model
                .prepare_prediction_input(&structure)
                .expect("prepare UMA input"),
        )
    } else {
        None
    };

    let predict_once = |prepared: Option<&_>| match (prepared, dtype) {
        (Some(p), PredictionComputeDtype::Float32) => model
            .predict_prepared_f32(p, &options)
            .expect("predict f32"),
        (Some(p), PredictionComputeDtype::Float64) => {
            model.predict_prepared(p, &options).expect("predict f64")
        }
        (None, PredictionComputeDtype::Float32) => {
            let p = model
                .prepare_prediction_input(&structure)
                .expect("prepare input");
            model
                .predict_prepared_f32(&p, &options)
                .expect("predict f32")
        }
        (None, PredictionComputeDtype::Float64) => {
            model.predict(&structure, &options).expect("predict f64")
        }
    };

    for _ in 0..warmup {
        let _ = predict_once(prepared.as_ref());
    }

    let mut times = Vec::with_capacity(repeat);
    let mut energy = 0.0;
    let mut force_max = 0.0_f64;
    for _ in 0..repeat {
        let start = Instant::now();
        let prediction = predict_once(prepared.as_ref());
        times.push(start.elapsed().as_secs_f64());
        energy = prediction.energy.expect("energy");
        force_max = prediction
            .forces
            .expect("forces")
            .iter()
            .flat_map(|force| force.iter())
            .map(|value| value.abs())
            .fold(0.0, f64::max);
    }
    times.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).expect("finite timing"));
    let median = if repeat.is_multiple_of(2) {
        0.5 * (times[repeat / 2 - 1] + times[repeat / 2])
    } else {
        times[repeat / 2]
    };
    let p95_index = ((repeat as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(repeat - 1);
    let p95 = times[p95_index];
    let min = *times.first().expect("nonempty timing");

    println!(
        "{{\"schema\":\"uma-rust-benchmark-v1\",\"structure\":\"{}\",\"dtype\":\"{}\",\"prepared\":{},\"warmup\":{},\"repeat\":{},\"rayon_threads\":{},\"median_s\":{:.15e},\"p95_s\":{:.15e},\"min_s\":{:.15e},\"energy_ev\":{:.15e},\"force_max_abs_ev_per_angstrom\":{:.15e}}}",
        structure_key, dtype_str, use_prepared, warmup, repeat, rayon_threads, median, p95, min, energy, force_max
    );
}
