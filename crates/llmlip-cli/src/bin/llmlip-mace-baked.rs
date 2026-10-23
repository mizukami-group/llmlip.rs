mod mace_baked_common;

use clap::{Parser, Subcommand, ValueEnum};
use mace_baked_common::{
    run_benchmark_predict, run_predict, BakedRuntime, ComputeDtype, MACE_MP0B_RUNTIME,
    MACE_OAM_MEDIUM_RUNTIME, MACE_OAM_SMALL_RUNTIME, MACE_OSAKA24_LARGE_RUNTIME,
    MACE_OSAKA24_MEDIUM_RUNTIME, MACE_OSAKA24_SMALL_RUNTIME, MACE_OSAKA26_RUNTIME,
};

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Clone, Copy, Debug, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum BakedModel {
    #[value(alias = "oam-medium", alias = "mace-oam0-medium")]
    MaceOamMedium,
    #[value(alias = "oam-small", alias = "mace-oam0-small")]
    MaceOamSmall,
    #[value(alias = "mp0b", alias = "mace-mp-0b3-medium", alias = "medium-0b3")]
    MaceMp0b,
    #[value(alias = "osaka24-small", alias = "mace-osaka24-small")]
    MaceOsaka24Small,
    #[value(alias = "osaka24-medium", alias = "mace-osaka24-medium")]
    MaceOsaka24Medium,
    #[value(alias = "osaka24-large", alias = "mace-osaka24-large")]
    MaceOsaka24Large,
    #[value(
        alias = "osaka26",
        alias = "osaka26-small",
        alias = "mace-osaka26-small"
    )]
    MaceOsaka26,
}

impl BakedModel {
    const fn runtime(self) -> BakedRuntime {
        match self {
            Self::MaceOamMedium => MACE_OAM_MEDIUM_RUNTIME,
            Self::MaceOamSmall => MACE_OAM_SMALL_RUNTIME,
            Self::MaceMp0b => MACE_MP0B_RUNTIME,
            Self::MaceOsaka24Small => MACE_OSAKA24_SMALL_RUNTIME,
            Self::MaceOsaka24Medium => MACE_OSAKA24_MEDIUM_RUNTIME,
            Self::MaceOsaka24Large => MACE_OSAKA24_LARGE_RUNTIME,
            Self::MaceOsaka26 => MACE_OSAKA26_RUNTIME,
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "llmlip-mace-baked")]
#[command(about = "Multi-model baked MACE inference binary")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Predict energy and/or derivatives with a baked MACE model.
    Predict {
        #[arg(long, value_enum)]
        model: BakedModel,
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
    /// Benchmark repeated load-once predictions with a baked MACE model.
    BenchmarkPredict {
        #[arg(long, value_enum)]
        model: BakedModel,
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
            model,
            structure,
            energy,
            forces,
            forces_out,
            stress,
            virial,
            json,
            dtype,
        } => run_predict(
            model.runtime(),
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
            model,
            structure,
            energy,
            forces,
            stress,
            virial,
            warmup,
            iterations,
            dtype,
        } => run_benchmark_predict(
            model.runtime(),
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
