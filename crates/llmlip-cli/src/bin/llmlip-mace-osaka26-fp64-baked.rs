mod mace_baked_common;

use clap::{Parser, Subcommand};
use mace_baked_common::{run_benchmark_predict, run_predict, ComputeDtype, MACE_OSAKA26_RUNTIME};

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

const DTYPE: ComputeDtype = ComputeDtype::Float64;

#[derive(Debug, Parser)]
#[command(name = "llmlip-mace-osaka26-fp64-baked")]
#[command(about = "MACE-Osaka26 Small FP64-only baked inference binary")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Predict energy and/or derivatives with the baked MACE-Osaka26 model.
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
        } => run_predict(
            MACE_OSAKA26_RUNTIME,
            &structure,
            energy,
            forces,
            forces_out,
            stress,
            virial,
            json,
            DTYPE,
        ),
        Command::BenchmarkPredict {
            structure,
            energy,
            forces,
            stress,
            virial,
            warmup,
            iterations,
        } => run_benchmark_predict(
            MACE_OSAKA26_RUNTIME,
            &structure,
            energy,
            forces,
            stress,
            virial,
            warmup,
            iterations,
            DTYPE,
        ),
    }
}
