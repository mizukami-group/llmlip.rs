mod mace_baked_common;

use clap::{Parser, Subcommand};
use mace_baked_common::{
    run_benchmark_predict, run_predict, ComputeDtype, MACE_OAM_MEDIUM_RUNTIME,
};

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Debug, Parser)]
#[command(name = "llmlip-mace-oam-medium-baked")]
#[command(about = "MACE-OAM Medium model-specific baked inference binary")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Predict energy and/or derivatives with the baked MACE-OAM Medium model.
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
        #[arg(long, value_enum, default_value_t = ComputeDtype::Float64)]
        dtype: ComputeDtype,
    },
    /// Benchmark repeated load-once predictions with the baked model.
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
        #[arg(long, value_enum, default_value_t = ComputeDtype::Float64)]
        dtype: ComputeDtype,
    },
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
            dtype,
        } => run_predict(
            MACE_OAM_MEDIUM_RUNTIME,
            &structure,
            energy,
            forces,
            forces_out,
            stress,
            virial,
            json,
            dtype,
        ),
        Command::BenchmarkPredict {
            structure,
            energy,
            forces,
            stress,
            virial,
            warmup,
            iterations,
            dtype,
        } => run_benchmark_predict(
            MACE_OAM_MEDIUM_RUNTIME,
            &structure,
            energy,
            forces,
            stress,
            virial,
            warmup,
            iterations,
            dtype,
        ),
    }
}
