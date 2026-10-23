use std::{env, fs, hint::black_box, time::Instant};

use clap::{Parser, Subcommand, ValueEnum};
use llmlip::{
    pack_uma_model_tensors, read_extxyz, read_extxyz_frames, validate_llmlip_model,
    write_extxyz_with_forces, Cell, EsenOamProfile, ExtXyzFrameMetadata, LlmlipModel,
    MaceMh1WignerMode, ModelError, PetLiteProfile, Prediction, PredictionComputeDtype,
    PredictionInputRequirements, PredictionOptions, PreparedPredictionInput,
    PreparedPredictionWorkspace, Structure, UmaProfile, Vec3,
};
use serde::Serialize;

const BENCHMARK_PREDICT_SCHEMA_VERSION: &str = "llmlip-predict-benchmark-v1.1";
const PREDICTION_SCHEMA_VERSION: &str = "llmlip-prediction-v1";

#[derive(Debug, Parser)]
#[command(name = "llmlip")]
#[command(about = "Rust-native fixed-weight equivariant MLIP inference")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Predict energy and/or forces from an .e3rs model.
    Predict {
        #[arg(long)]
        model: String,
        #[arg(long)]
        structure: String,
        #[arg(long)]
        energy: bool,
        /// Request forces. With an optional PATH, also write legacy force extXYZ output.
        #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "")]
        forces: Option<Option<String>>,
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
        #[arg(long, value_enum, default_value_t = MaceMh1WignerModeOpt::TrueFloat64)]
        mace_mh1_wigner_mode: MaceMh1WignerModeOpt,
        #[arg(long)]
        head: Option<String>,
        #[arg(long)]
        total_charge: Option<f64>,
        #[arg(long)]
        spin_multiplicity: Option<f64>,
    },
    /// Check conservative forces against central finite differences.
    FiniteDifferenceForces {
        #[arg(long)]
        model: String,
        #[arg(long)]
        structure: String,
        #[arg(long, default_value_t = 1.0e-4)]
        fd_eps: f64,
        #[arg(long, default_value_t = 5.0e-3)]
        fd_tol: f64,
        #[arg(long, value_enum, default_value_t = FdMode::MaxForce)]
        fd_mode: FdMode,
        #[arg(long, default_value_t = 0)]
        component_start: usize,
        #[arg(long)]
        component_limit: Option<usize>,
        #[arg(long)]
        include_components: bool,
        #[arg(long, value_enum, default_value_t = MaceMh1WignerModeOpt::TrueFloat64)]
        mace_mh1_wigner_mode: MaceMh1WignerModeOpt,
        #[arg(long)]
        head: Option<String>,
        #[arg(long)]
        total_charge: Option<f64>,
        #[arg(long)]
        spin_multiplicity: Option<f64>,
    },
    /// Benchmark repeated load-once predictions and emit JSON timing stats.
    BenchmarkPredict {
        #[arg(long)]
        model: String,
        #[arg(long)]
        structure: Option<String>,
        #[arg(long)]
        fixture: Option<String>,
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
        /// Prepare supported graph/topology inputs once before timed samples.
        #[arg(long)]
        prepared_graph: bool,
        /// Emit supported model stage-level timing counters in the JSON report.
        #[arg(long)]
        profile_stages: bool,
        #[arg(long)]
        label: Option<String>,
        #[arg(long, value_enum, default_value_t = ComputeDtype::Float64)]
        dtype: ComputeDtype,
        #[arg(long, value_enum, default_value_t = MaceMh1WignerModeOpt::TrueFloat64)]
        mace_mh1_wigner_mode: MaceMh1WignerModeOpt,
        #[arg(long)]
        head: Option<String>,
        #[arg(long)]
        total_charge: Option<f64>,
        #[arg(long)]
        spin_multiplicity: Option<f64>,
    },
    /// Batch-predict energy/forces/stress for all frames in a multi-frame extXYZ.
    ///
    /// Outputs per-structure JSON results with timing, atom count, element
    /// species, and optional graph-setup breakdown. Designed for large-scale
    /// MPTrj verification comparing Rust vs upstream.
    BatchPredict {
        #[arg(long)]
        model: String,
        /// Path to a multi-frame extXYZ file.
        #[arg(long)]
        structures: String,
        #[arg(long)]
        energy: bool,
        #[arg(long)]
        forces: bool,
        #[arg(long)]
        stress: bool,
        #[arg(long)]
        virial: bool,
        #[arg(long, value_enum, default_value_t = ComputeDtype::Float64)]
        dtype: ComputeDtype,
        #[arg(long, value_enum, default_value_t = MaceMh1WignerModeOpt::TrueFloat64)]
        mace_mh1_wigner_mode: MaceMh1WignerModeOpt,
        #[arg(long)]
        head: Option<String>,
        #[arg(long)]
        total_charge: Option<f64>,
        #[arg(long)]
        spin_multiplicity: Option<f64>,
        /// Output JSON file path for the full batch result.
        #[arg(long)]
        out: String,
        /// Print progress every N structures.
        #[arg(long, default_value_t = 100)]
        progress_interval: usize,
        /// Start from this zero-based frame index after reading the extXYZ file.
        #[arg(long, default_value_t = 0)]
        start_index: usize,
        /// Limit the number of frames predicted after --start-index.
        #[arg(long)]
        max_structures: Option<usize>,
        /// Emit supported model stage-level timing counters per structure.
        #[arg(long)]
        profile_stages: bool,
    },
    /// Validate an .e3rs model without running prediction.
    ValidateModel {
        #[arg(long)]
        model: String,
    },
    /// Pack UMA tensor `.bin` files into a single-file `tensors.pack`.
    PackUmaTensors {
        #[arg(long)]
        model: String,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
#[value(rename_all = "snake_case")]
enum FdMode {
    Representative,
    MaxForce,
    #[value(name = "full_3n", alias = "full3n")]
    Full3n,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
#[value(rename_all = "lower")]
enum ComputeDtype {
    #[value(alias = "fp64", alias = "f64")]
    Float64,
    #[value(alias = "fp32", alias = "f32")]
    Float32,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum MaceMh1WignerModeOpt {
    TrueFloat64,
    UpstreamArtifact,
}

impl From<MaceMh1WignerModeOpt> for MaceMh1WignerMode {
    fn from(value: MaceMh1WignerModeOpt) -> Self {
        match value {
            MaceMh1WignerModeOpt::TrueFloat64 => Self::TrueFloat64,
            MaceMh1WignerModeOpt::UpstreamArtifact => Self::UpstreamArtifact,
        }
    }
}

impl From<ComputeDtype> for PredictionComputeDtype {
    fn from(value: ComputeDtype) -> Self {
        match value {
            ComputeDtype::Float64 => Self::Float64,
            ComputeDtype::Float32 => Self::Float32,
        }
    }
}

impl ComputeDtype {
    const fn report_name(self) -> &'static str {
        match self {
            Self::Float64 => "float64",
            Self::Float32 => "float32",
        }
    }
}

#[derive(Clone, Debug)]
struct FiniteDifferenceOptions {
    fd_eps: f64,
    fd_tol: f64,
    fd_mode: FdMode,
    component_start: usize,
    component_limit: Option<usize>,
    include_components: bool,
    mace_mh1_wigner_mode: MaceMh1WignerMode,
    head: Option<String>,
    total_charge: Option<f64>,
    spin_multiplicity: Option<f64>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
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
            mace_mh1_wigner_mode,
            head,
            total_charge,
            spin_multiplicity,
        } => {
            let (compute_forces, forces_out) = resolve_predict_forces(forces, forces_out, json)?;
            if !energy && !compute_forces && !stress && !virial {
                return Err(
                    "predict requires at least one of --energy, --forces, --forces-out <PATH>, --stress, or --virial"
                        .to_owned(),
                );
            }
            let model =
                LlmlipModel::load(&model).map_err(|err| format!("failed to load model: {err}"))?;
            let structure = read_extxyz(&structure)
                .map_err(|err| format!("failed to read structure: {err}"))?;
            let options = PredictionOptions {
                compute_energy: energy,
                compute_forces,
                compute_stress: stress,
                compute_virial: virial,
                total_charge,
                spin_multiplicity,
                selected_head: head,
                mace_mh1_wigner_mode: mace_mh1_wigner_mode.into(),
            };
            let prediction = model
                .predict_with_compute_dtype(&structure, &options, dtype.into())
                .map_err(predict_error)?;
            if !json && energy {
                let value = prediction
                    .energy
                    .ok_or_else(|| "model did not return requested energy".to_owned())?;
                println!("energy = {value:.16e}");
            }
            if let Some(path) = &forces_out {
                let force_values = prediction
                    .forces
                    .as_ref()
                    .ok_or_else(|| "model did not return requested forces".to_owned())?;
                write_extxyz_with_forces(path, &structure, force_values, prediction.energy)
                    .map_err(|err| format!("failed to write forces: {err}"))?;
            }
            if !json && stress {
                let value = prediction
                    .stress
                    .ok_or_else(|| {
                        "model did not return requested stress; the model may not support stress or the structure may lack full PBC".to_owned()
                    })?;
                print_matrix("stress", "eV/angstrom^3", value);
            }
            if !json && virial {
                let value = prediction
                    .virial
                    .ok_or_else(|| {
                        "model did not return requested virial; the model may not support virial or the structure may lack full PBC".to_owned()
                    })?;
                print_matrix("virial", "eV", value);
            }
            if json {
                if energy && prediction.energy.is_none() {
                    return Err("model did not return requested energy".to_owned());
                }
                if compute_forces && prediction.forces.is_none() {
                    return Err("model did not return requested forces".to_owned());
                }
                if stress && prediction.stress.is_none() {
                    return Err(
                        "model did not return requested stress; the model may not support stress or the structure may lack full PBC"
                            .to_owned(),
                    );
                }
                if virial && prediction.virial.is_none() {
                    return Err(
                        "model did not return requested virial; the model may not support virial or the structure may lack full PBC"
                            .to_owned(),
                    );
                }
                let report = PredictionJson {
                    schema_version: PREDICTION_SCHEMA_VERSION,
                    energy: prediction.energy,
                    forces: prediction.forces.as_deref(),
                    stress: prediction.stress,
                    virial: prediction.virial,
                };
                serde_json::to_writer_pretty(std::io::stdout(), &report)
                    .map_err(|err| format!("failed to write prediction JSON: {err}"))?;
                println!();
            }
        }
        Command::FiniteDifferenceForces {
            model,
            structure,
            fd_eps,
            fd_tol,
            fd_mode,
            component_start,
            component_limit,
            include_components,
            mace_mh1_wigner_mode,
            head,
            total_charge,
            spin_multiplicity,
        } => {
            run_finite_difference_forces(
                &model,
                &structure,
                FiniteDifferenceOptions {
                    fd_eps,
                    fd_tol,
                    fd_mode,
                    component_start,
                    component_limit,
                    include_components,
                    mace_mh1_wigner_mode: mace_mh1_wigner_mode.into(),
                    head,
                    total_charge,
                    spin_multiplicity,
                },
            )?;
        }
        Command::BenchmarkPredict {
            model,
            structure,
            fixture,
            energy,
            forces,
            stress,
            virial,
            warmup,
            iterations,
            prepared_graph,
            profile_stages,
            label,
            dtype,
            mace_mh1_wigner_mode,
            head,
            total_charge,
            spin_multiplicity,
        } => {
            run_benchmark_predict(
                &model,
                structure.as_deref(),
                fixture.as_deref(),
                PredictionOptions {
                    compute_energy: energy,
                    compute_forces: forces,
                    compute_stress: stress,
                    compute_virial: virial,
                    total_charge,
                    spin_multiplicity,
                    selected_head: head,
                    mace_mh1_wigner_mode: mace_mh1_wigner_mode.into(),
                },
                BenchmarkPredictOptions {
                    warmup,
                    iterations,
                    prepared_graph,
                    profile_stages,
                    label,
                    dtype,
                },
            )?;
        }
        Command::BatchPredict {
            model,
            structures,
            energy,
            forces,
            stress,
            virial,
            dtype,
            mace_mh1_wigner_mode,
            head,
            total_charge,
            spin_multiplicity,
            out,
            progress_interval,
            start_index,
            max_structures,
            profile_stages,
        } => {
            if !energy && !forces && !stress && !virial {
                return Err(
                    "batch-predict requires at least one of --energy, --forces, --stress, or --virial"
                        .to_owned(),
                );
            }
            run_batch_predict(
                &model,
                &structures,
                PredictionOptions {
                    compute_energy: energy,
                    compute_forces: forces,
                    compute_stress: stress,
                    compute_virial: virial,
                    total_charge,
                    spin_multiplicity,
                    selected_head: head,
                    mace_mh1_wigner_mode: mace_mh1_wigner_mode.into(),
                },
                dtype,
                &out,
                BatchPredictRunOptions {
                    progress_interval,
                    start_index,
                    max_structures,
                    profile_stages,
                },
            )?;
        }
        Command::ValidateModel { model } => {
            validate_llmlip_model(&model)
                .map_err(|err| format!("failed to validate model: {err}"))?;
            println!("model validation ok");
        }
        Command::PackUmaTensors { model } => {
            let output = pack_uma_model_tensors(&model)
                .map_err(|err| format!("failed to pack UMA tensors: {err}"))?;
            println!("wrote {}", output.display());
        }
    }
    Ok(())
}

fn print_matrix(label: &str, unit: &str, matrix: [[f64; 3]; 3]) {
    println!("{label} ({unit}, row-major) =");
    for row in matrix {
        println!("  [{:.16e} {:.16e} {:.16e}]", row[0], row[1], row[2]);
    }
}

fn resolve_predict_forces(
    forces: Option<Option<String>>,
    forces_out: Option<String>,
    json: bool,
) -> Result<(bool, Option<String>), String> {
    let forces_requested = forces.is_some();
    let forces_path = forces.and_then(|path| path).filter(|path| !path.is_empty());
    if forces_path.is_some() && forces_out.is_some() {
        return Err("use only one of --forces [PATH] or --forces-out <PATH>".to_owned());
    }
    let output_path = forces_out.or(forces_path);
    if forces_requested && output_path.is_none() && !json {
        return Err(
            "--forces without PATH is supported only with --json; use --forces-out <PATH> for legacy extXYZ output"
                .to_owned(),
        );
    }
    let compute_forces = forces_requested || output_path.is_some();
    Ok((compute_forces, output_path))
}

#[derive(Serialize)]
struct PredictionJson<'a> {
    schema_version: &'static str,
    energy: Option<f64>,
    forces: Option<&'a [Vec3]>,
    stress: Option<[[f64; 3]; 3]>,
    virial: Option<[[f64; 3]; 3]>,
}

fn predict_error(err: ModelError) -> String {
    format!("prediction failed: {err}")
}

#[derive(Clone, Debug, Serialize)]
struct FiniteDifferenceComponent {
    component_index: usize,
    atom: usize,
    axis: usize,
    analytic_force: f64,
    finite_difference_force: f64,
    abs_delta: f64,
    plus_energy: f64,
    minus_energy: f64,
}

#[derive(Debug, Serialize)]
struct FiniteDifferenceReport {
    schema_version: &'static str,
    mode: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    selected_head: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_charge: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    spin_multiplicity: Option<f64>,
    eps: f64,
    tolerance: f64,
    base_energy: f64,
    total_component_count: usize,
    selected_component_start: usize,
    selected_component_limit: Option<usize>,
    component_count: usize,
    energy_eval_count: usize,
    prediction_runtime_ms: f64,
    energy_runtime_ms: f64,
    passed: bool,
    max_abs_delta: f64,
    rms_delta: f64,
    p95_abs_delta: f64,
    p99_abs_delta: f64,
    p999_abs_delta: f64,
    failed_component_count: usize,
    worst_components: Vec<FiniteDifferenceComponent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    components: Option<Vec<FiniteDifferenceComponent>>,
    abs_delta: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    atom: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    axis: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    analytic_force: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finite_difference_force: Option<f64>,
}

#[derive(Clone, Debug)]
struct BenchmarkPredictOptions {
    warmup: usize,
    iterations: usize,
    prepared_graph: bool,
    profile_stages: bool,
    label: Option<String>,
    dtype: ComputeDtype,
}

#[derive(Debug, Serialize)]
struct BenchmarkRequestedOutputs {
    energy: bool,
    forces: bool,
    stress: bool,
    virial: bool,
}

#[derive(Debug, Serialize)]
struct BenchmarkEnvironment {
    logical_cpus: Option<usize>,
    target_os: &'static str,
    target_arch: &'static str,
    target_family: &'static str,
    enabled_features: Vec<&'static str>,
    rustflags: Option<String>,
    thread_env: Vec<BenchmarkEnvVar>,
}

#[derive(Debug, Serialize)]
struct BenchmarkMeasurementScope {
    model_load_excluded: bool,
    structure_file_read_excluded: bool,
    neighbor_graph_setup_excluded: bool,
    prepared_scope_match: bool,
    note: &'static str,
}

#[derive(Debug, Serialize)]
struct BenchmarkEnvVar {
    key: &'static str,
    value: Option<String>,
}

#[derive(Debug, Serialize)]
struct BenchmarkPredictReport {
    schema_version: &'static str,
    label: String,
    model: String,
    structure: String,
    structure_count: usize,
    atom_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    edge_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    active_edge_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_edge_count: Option<usize>,
    requested_outputs: BenchmarkRequestedOutputs,
    #[serde(skip_serializing_if = "Option::is_none")]
    selected_head: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_charge: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    spin_multiplicity: Option<f64>,
    compute_dtype: &'static str,
    warmup: usize,
    iterations: usize,
    load_runtime_ms: f64,
    preparation_runtime_ms: f64,
    mean_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    min_ms: f64,
    max_ms: f64,
    stdev_ms: f64,
    samples_ms: Vec<f64>,
    checksum: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    esen_oam_profile: Option<EsenOamProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pet_lite_profile: Option<PetLiteProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    uma_profile: Option<UmaProfile>,
    measurement_scope: BenchmarkMeasurementScope,
    environment: BenchmarkEnvironment,
}

fn run_finite_difference_forces(
    model_path: &str,
    structure_path: &str,
    options: FiniteDifferenceOptions,
) -> Result<(), String> {
    if !options.fd_eps.is_finite() || options.fd_eps <= 0.0 {
        return Err(format!(
            "--fd-eps must be finite and positive, got {}",
            options.fd_eps
        ));
    }
    if !options.fd_tol.is_finite() || options.fd_tol < 0.0 {
        return Err(format!(
            "--fd-tol must be finite and non-negative, got {}",
            options.fd_tol
        ));
    }

    let model =
        LlmlipModel::load(model_path).map_err(|err| format!("failed to load model: {err}"))?;
    let structure =
        read_extxyz(structure_path).map_err(|err| format!("failed to read structure: {err}"))?;
    let report = finite_difference_report(&model, &structure, options)?;
    serde_json::to_writer_pretty(std::io::stdout(), &report)
        .map_err(|err| format!("failed to write finite-difference JSON report: {err}"))?;
    println!();
    Ok(())
}

fn run_benchmark_predict(
    model_path: &str,
    structure_path: Option<&str>,
    fixture_path: Option<&str>,
    prediction_options: PredictionOptions,
    options: BenchmarkPredictOptions,
) -> Result<(), String> {
    if !prediction_options.compute_energy
        && !prediction_options.compute_forces
        && !prediction_options.compute_stress
        && !prediction_options.compute_virial
    {
        return Err(
            "benchmark-predict requires at least one of --energy, --forces, --stress, or --virial"
                .to_owned(),
        );
    }
    if options.iterations == 0 {
        return Err("--iterations must be at least 1".to_owned());
    }
    let (input_path, structures) = match (structure_path, fixture_path) {
        (Some(_), Some(_)) => {
            return Err("benchmark-predict accepts only one of --structure or --fixture".to_owned())
        }
        (Some(path), None) => (
            path.to_owned(),
            vec![read_extxyz(path).map_err(|err| format!("failed to read structure: {err}"))?],
        ),
        (None, Some(path)) => (path.to_owned(), read_benchmark_fixture_structures(path)?),
        (None, None) => {
            return Err("benchmark-predict requires --structure or --fixture".to_owned())
        }
    };

    let load_start = Instant::now();
    let model =
        LlmlipModel::load(model_path).map_err(|err| format!("failed to load model: {err}"))?;
    let load_runtime_ms = elapsed_ms(load_start);
    let mut esen_oam_profile = (options.profile_stages
        && matches!(&model, LlmlipModel::EsenOam(_)))
    .then(EsenOamProfile::new);
    let mut pet_lite_profile = (options.profile_stages
        && matches!(&model, LlmlipModel::PetLite(_)))
    .then(PetLiteProfile::new);
    let mut uma_profile =
        (options.profile_stages && matches!(&model, LlmlipModel::Uma(_))).then(UmaProfile::new);
    let prepare_start = Instant::now();
    let prepared_inputs = if options.prepared_graph {
        let mut prepared_inputs = Vec::with_capacity(structures.len());
        for structure in &structures {
            prepared_inputs.push(
                model
                    .prepare_prediction_input_with_options_profiled(
                        structure,
                        &prediction_options,
                        esen_oam_profile.as_mut(),
                    )
                    .map_err(predict_error)?,
            );
        }
        Some(prepared_inputs)
    } else {
        None
    };
    let preparation_runtime_ms = if options.prepared_graph {
        elapsed_ms(prepare_start)
    } else {
        0.0
    };
    let mut prepared_workspace = PreparedPredictionWorkspace::new();
    let mut checksum = 0.0;
    for _ in 0..options.warmup {
        checksum += match prepared_inputs.as_ref() {
            Some(prepared) => benchmark_prepared_prediction_pass(
                &model,
                prepared,
                &prediction_options,
                options.dtype,
                &mut prepared_workspace,
                esen_oam_profile.as_mut(),
                pet_lite_profile.as_mut(),
                uma_profile.as_mut(),
            )?,
            None => benchmark_prediction_pass(
                &model,
                &structures,
                &prediction_options,
                options.dtype,
                &mut prepared_workspace,
                esen_oam_profile.as_mut(),
                pet_lite_profile.as_mut(),
                uma_profile.as_mut(),
            )?,
        };
    }

    let mut samples_ms = Vec::with_capacity(options.iterations);
    for _ in 0..options.iterations {
        let start = Instant::now();
        let pass_checksum = match prepared_inputs.as_ref() {
            Some(prepared) => benchmark_prepared_prediction_pass(
                &model,
                prepared,
                &prediction_options,
                options.dtype,
                &mut prepared_workspace,
                esen_oam_profile.as_mut(),
                pet_lite_profile.as_mut(),
                uma_profile.as_mut(),
            )?,
            None => benchmark_prediction_pass(
                &model,
                &structures,
                &prediction_options,
                options.dtype,
                &mut prepared_workspace,
                esen_oam_profile.as_mut(),
                pet_lite_profile.as_mut(),
                uma_profile.as_mut(),
            )?,
        };
        samples_ms.push(elapsed_ms(start));
        checksum += pass_checksum;
    }
    let mut sorted_samples = samples_ms.clone();
    sorted_samples.sort_by(f64::total_cmp);
    let mean_ms = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;
    let stdev_ms = if samples_ms.len() > 1 {
        let variance = samples_ms
            .iter()
            .map(|sample| {
                let delta = sample - mean_ms;
                delta * delta
            })
            .sum::<f64>()
            / (samples_ms.len() - 1) as f64;
        variance.sqrt()
    } else {
        0.0
    };
    let label = options.label.unwrap_or_else(|| "llmlip_predict".to_owned());
    let active_edge_count = prepared_inputs.as_ref().map(|prepared| {
        prepared
            .iter()
            .map(PreparedPredictionInput::edge_count)
            .sum()
    });
    let raw_edge_count = prepared_inputs.as_ref().and_then(|prepared| {
        prepared
            .iter()
            .map(PreparedPredictionInput::raw_edge_count)
            .try_fold(0usize, |total, raw_edges| {
                raw_edges.and_then(|count| total.checked_add(count))
            })
    });
    let report = BenchmarkPredictReport {
        schema_version: BENCHMARK_PREDICT_SCHEMA_VERSION,
        label,
        model: model_path.to_owned(),
        structure: input_path,
        structure_count: structures.len(),
        atom_count: structures.iter().map(Structure::len).sum(),
        edge_count: active_edge_count,
        active_edge_count,
        raw_edge_count,
        requested_outputs: BenchmarkRequestedOutputs {
            energy: prediction_options.compute_energy,
            forces: prediction_options.compute_forces,
            stress: prediction_options.compute_stress,
            virial: prediction_options.compute_virial,
        },
        selected_head: prediction_options.selected_head.clone(),
        total_charge: prediction_options.total_charge,
        spin_multiplicity: prediction_options.spin_multiplicity,
        compute_dtype: options.dtype.report_name(),
        warmup: options.warmup,
        iterations: options.iterations,
        load_runtime_ms,
        preparation_runtime_ms,
        mean_ms,
        median_ms: percentile(&sorted_samples, 50.0),
        p95_ms: percentile(&sorted_samples, 95.0),
        min_ms: sorted_samples[0],
        max_ms: *sorted_samples
            .last()
            .ok_or_else(|| "benchmark produced no samples".to_owned())?,
        stdev_ms,
        samples_ms,
        checksum,
        esen_oam_profile,
        pet_lite_profile,
        uma_profile,
        measurement_scope: BenchmarkMeasurementScope {
            model_load_excluded: true,
            structure_file_read_excluded: true,
            neighbor_graph_setup_excluded: options.prepared_graph,
            prepared_scope_match: options.prepared_graph,
            note: if options.prepared_graph {
                "Timed samples use prepared graph/topology inputs when supported by the model."
            } else {
                "Timed samples call normal prediction; model load and structure file read are excluded, but graph/topology setup may remain inside predict()."
            },
        },
        environment: benchmark_environment(),
    };
    serde_json::to_writer_pretty(std::io::stdout(), &report)
        .map_err(|err| format!("failed to write benchmark JSON report: {err}"))?;
    println!();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn benchmark_prediction_pass(
    model: &LlmlipModel,
    structures: &[Structure],
    options: &PredictionOptions,
    dtype: ComputeDtype,
    workspace: &mut PreparedPredictionWorkspace,
    mut profile: Option<&mut EsenOamProfile>,
    mut pet_profile: Option<&mut PetLiteProfile>,
    mut uma_profile: Option<&mut UmaProfile>,
) -> Result<f64, String> {
    let mut checksum = 0.0;
    for structure in structures {
        let prediction = if profile.is_some() || pet_profile.is_some() || uma_profile.is_some() {
            match model.prepare_prediction_input_with_options_profiled(
                black_box(structure),
                black_box(options),
                profile.as_deref_mut(),
            ) {
                Ok(prepared) => model
                    .predict_prepared_with_compute_dtype_workspace_profiled(
                        black_box(&prepared),
                        black_box(options),
                        dtype.into(),
                        workspace,
                        profile.as_deref_mut(),
                        pet_profile.as_deref_mut(),
                        uma_profile.as_deref_mut(),
                    )
                    .map_err(predict_error)?,
                Err(ModelError::PreparedPredictionNotSupported(_)) => model
                    .predict_with_compute_dtype(
                        black_box(structure),
                        black_box(options),
                        dtype.into(),
                    )
                    .map_err(predict_error)?,
                Err(err) => return Err(predict_error(err)),
            }
        } else {
            model
                .predict_with_compute_dtype(black_box(structure), black_box(options), dtype.into())
                .map_err(predict_error)?
        };
        checksum += prediction_checksum(&prediction, options)?;
    }
    Ok(black_box(checksum))
}

#[allow(clippy::too_many_arguments)]
fn benchmark_prepared_prediction_pass(
    model: &LlmlipModel,
    prepared_inputs: &[PreparedPredictionInput],
    options: &PredictionOptions,
    dtype: ComputeDtype,
    workspace: &mut PreparedPredictionWorkspace,
    mut profile: Option<&mut EsenOamProfile>,
    mut pet_profile: Option<&mut PetLiteProfile>,
    mut uma_profile: Option<&mut UmaProfile>,
) -> Result<f64, String> {
    let mut checksum = 0.0;
    for prepared in prepared_inputs {
        let prediction = model
            .predict_prepared_with_compute_dtype_workspace_profiled(
                black_box(prepared),
                black_box(options),
                dtype.into(),
                workspace,
                profile.as_deref_mut(),
                pet_profile.as_deref_mut(),
                uma_profile.as_deref_mut(),
            )
            .map_err(predict_error)?;
        checksum += prediction_checksum(&prediction, options)?;
    }
    Ok(black_box(checksum))
}

fn read_benchmark_fixture_structures(path: &str) -> Result<Vec<Structure>, String> {
    let value: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(path).map_err(|err| format!("failed to read fixture: {err}"))?,
    )
    .map_err(|err| format!("failed to parse fixture JSON: {err}"))?;
    let structures = value["structures"]
        .as_array()
        .ok_or_else(|| "fixture must contain structures array".to_owned())?;
    structures
        .iter()
        .map(|case| {
            let structure = &case["structure"];
            let species = structure["symbols"]
                .as_array()
                .ok_or_else(|| "structure must contain symbols".to_owned())?
                .iter()
                .map(|symbol| {
                    symbol
                        .as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| "symbol must be a string".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let positions = structure["positions_angstrom"]
                .as_array()
                .ok_or_else(|| "structure must contain positions_angstrom".to_owned())?
                .iter()
                .map(|row| {
                    let row = row
                        .as_array()
                        .ok_or_else(|| "position row must be an array".to_owned())?;
                    if row.len() != 3 {
                        return Err("position row must have length 3".to_owned());
                    }
                    Ok([
                        row[0]
                            .as_f64()
                            .ok_or_else(|| "position x must be f64".to_owned())?,
                        row[1]
                            .as_f64()
                            .ok_or_else(|| "position y must be f64".to_owned())?,
                        row[2]
                            .as_f64()
                            .ok_or_else(|| "position z must be f64".to_owned())?,
                    ])
                })
                .collect::<Result<Vec<_>, _>>()?;
            let pbc = if structure["pbc"].is_null() {
                [false; 3]
            } else {
                bool3_from_json(&structure["pbc"], "pbc")?
            };
            if pbc.iter().any(|enabled| *enabled) {
                let cell = mat3_from_json(&structure["cell"], "cell")?;
                Ok(Structure::new(species, positions).with_cell(Cell::new(cell), pbc))
            } else {
                Ok(Structure::new(species, positions))
            }
        })
        .collect()
}

fn bool3_from_json(value: &serde_json::Value, field: &str) -> Result<[bool; 3], String> {
    let row = value
        .as_array()
        .ok_or_else(|| format!("{field} must be an array"))?;
    if row.len() != 3 {
        return Err(format!("{field} must have length 3"));
    }
    Ok([
        row[0]
            .as_bool()
            .ok_or_else(|| format!("{field}[0] must be bool"))?,
        row[1]
            .as_bool()
            .ok_or_else(|| format!("{field}[1] must be bool"))?,
        row[2]
            .as_bool()
            .ok_or_else(|| format!("{field}[2] must be bool"))?,
    ])
}

fn mat3_from_json(value: &serde_json::Value, field: &str) -> Result<[Vec3; 3], String> {
    let rows = value
        .as_array()
        .ok_or_else(|| format!("{field} must be an array when pbc is enabled"))?;
    if rows.len() != 3 {
        return Err(format!("{field} must have 3 rows"));
    }
    Ok([
        vec3_from_json(&rows[0], &format!("{field}[0]"))?,
        vec3_from_json(&rows[1], &format!("{field}[1]"))?,
        vec3_from_json(&rows[2], &format!("{field}[2]"))?,
    ])
}

fn vec3_from_json(value: &serde_json::Value, field: &str) -> Result<Vec3, String> {
    let row = value
        .as_array()
        .ok_or_else(|| format!("{field} must be an array"))?;
    if row.len() != 3 {
        return Err(format!("{field} must have length 3"));
    }
    Ok([
        row[0]
            .as_f64()
            .ok_or_else(|| format!("{field}[0] must be f64"))?,
        row[1]
            .as_f64()
            .ok_or_else(|| format!("{field}[1] must be f64"))?,
        row[2]
            .as_f64()
            .ok_or_else(|| format!("{field}[2] must be f64"))?,
    ])
}

fn prediction_checksum(
    prediction: &Prediction,
    options: &PredictionOptions,
) -> Result<f64, String> {
    let mut checksum = 0.0;
    if options.compute_energy {
        checksum += prediction
            .energy
            .ok_or_else(|| "model did not return requested energy".to_owned())?;
    }
    if options.compute_forces {
        checksum += prediction
            .forces
            .as_ref()
            .ok_or_else(|| "model did not return requested forces".to_owned())?
            .iter()
            .flatten()
            .sum::<f64>();
    }
    if options.compute_stress {
        checksum += matrix_sum(
            prediction
                .stress
                .as_ref()
                .ok_or_else(|| "model did not return requested stress".to_owned())?,
        );
    }
    if options.compute_virial {
        checksum += matrix_sum(
            prediction
                .virial
                .as_ref()
                .ok_or_else(|| "model did not return requested virial".to_owned())?,
        );
    }
    Ok(black_box(checksum))
}

fn matrix_sum(matrix: &[[f64; 3]; 3]) -> f64 {
    matrix.iter().flatten().sum()
}

fn benchmark_environment() -> BenchmarkEnvironment {
    const THREAD_ENV_KEYS: [&str; 6] = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "PYTORCH_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ];
    BenchmarkEnvironment {
        logical_cpus: std::thread::available_parallelism().ok().map(usize::from),
        target_os: env::consts::OS,
        target_arch: env::consts::ARCH,
        target_family: env::consts::FAMILY,
        enabled_features: enabled_llmlip_cli_features(),
        rustflags: env::var("RUSTFLAGS").ok(),
        thread_env: THREAD_ENV_KEYS
            .iter()
            .map(|key| BenchmarkEnvVar {
                key,
                value: env::var(key).ok(),
            })
            .collect(),
    }
}

fn enabled_llmlip_cli_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    macro_rules! push_feature {
        ($name:literal) => {
            if cfg!(feature = $name) {
                features.push($name);
            }
        };
    }
    push_feature!("accelerate-blas");
    push_feature!("apple-metal");
    push_feature!("armpl");
    push_feature!("embed-model");

    push_feature!("intel-mkl");
    push_feature!("intel-mkl-threaded");
    push_feature!("libxsmm");
    push_feature!("mimalloc");
    push_feature!("openblas");
    push_feature!("orb-f32-edge-mlp");
    push_feature!("parallel");
    push_feature!("parallel-blas");

    push_feature!("small-gemm-bypass");
    features
}

fn finite_difference_report(
    model: &LlmlipModel,
    structure: &Structure,
    options: FiniteDifferenceOptions,
) -> Result<FiniteDifferenceReport, String> {
    let prediction_start = Instant::now();
    let prediction = model
        .predict(
            structure,
            &PredictionOptions {
                compute_energy: true,
                compute_forces: true,
                total_charge: options.total_charge,
                spin_multiplicity: options.spin_multiplicity,
                selected_head: options.head.clone(),
                mace_mh1_wigner_mode: options.mace_mh1_wigner_mode,
                ..PredictionOptions::default()
            },
        )
        .map_err(predict_error)?;
    let prediction_runtime_ms = elapsed_ms(prediction_start);
    let base_energy = prediction
        .energy
        .ok_or_else(|| "model did not return requested energy".to_owned())?;
    let forces = prediction
        .forces
        .ok_or_else(|| "model did not return requested forces".to_owned())?;
    let all_components = fd_components(&forces, options.fd_mode)?;
    let total_component_count = all_components.len();
    let components = select_components(
        all_components,
        options.component_start,
        options.component_limit,
    )?;
    let selected_component_count = components.len();
    let energy_start = Instant::now();
    let mut records = Vec::with_capacity(selected_component_count);
    for (offset, (atom, axis)) in components.into_iter().enumerate() {
        let component_index = options.component_start + offset;
        let plus_energy =
            predict_perturbed_energy(model, structure, atom, axis, options.fd_eps, &options)?;
        let minus_energy =
            predict_perturbed_energy(model, structure, atom, axis, -options.fd_eps, &options)?;
        let finite_difference_force = -(plus_energy - minus_energy) / (2.0 * options.fd_eps);
        let analytic_force = forces[atom][axis];
        records.push(FiniteDifferenceComponent {
            component_index,
            atom,
            axis,
            analytic_force,
            finite_difference_force,
            abs_delta: (finite_difference_force - analytic_force).abs(),
            plus_energy,
            minus_energy,
        });
    }
    let energy_runtime_ms = elapsed_ms(energy_start);
    let mut sorted_abs = records
        .iter()
        .map(|record| record.abs_delta)
        .collect::<Vec<_>>();
    sorted_abs.sort_by(f64::total_cmp);
    let max_abs_delta = sorted_abs.last().copied().unwrap_or(0.0);
    let rms_delta = if sorted_abs.is_empty() {
        0.0
    } else {
        (sorted_abs.iter().map(|value| value * value).sum::<f64>() / sorted_abs.len() as f64).sqrt()
    };
    let failed_component_count = sorted_abs
        .iter()
        .filter(|value| **value > options.fd_tol)
        .count();
    let component_records = options.include_components.then(|| records.clone());
    records.sort_by(|left, right| right.abs_delta.total_cmp(&left.abs_delta));
    let worst_components = records.into_iter().take(10).collect::<Vec<_>>();
    let (atom, axis, analytic_force, finite_difference_force) =
        if let Some(record) = worst_components.first() {
            (
                Some(record.atom),
                Some(record.axis),
                Some(record.analytic_force),
                Some(record.finite_difference_force),
            )
        } else {
            (None, None, None, None)
        };

    Ok(FiniteDifferenceReport {
        schema_version: "llmlip-force-finite-difference-report-v1",
        mode: options.fd_mode.report_name(),
        selected_head: options.head.clone(),
        total_charge: options.total_charge,
        spin_multiplicity: options.spin_multiplicity,
        eps: options.fd_eps,
        tolerance: options.fd_tol,
        base_energy,
        total_component_count,
        selected_component_start: options.component_start,
        selected_component_limit: options.component_limit,
        component_count: sorted_abs.len(),
        energy_eval_count: 2 * sorted_abs.len(),
        prediction_runtime_ms,
        energy_runtime_ms,
        passed: failed_component_count == 0,
        max_abs_delta,
        rms_delta,
        p95_abs_delta: percentile(&sorted_abs, 95.0),
        p99_abs_delta: percentile(&sorted_abs, 99.0),
        p999_abs_delta: percentile(&sorted_abs, 99.9),
        failed_component_count,
        worst_components,
        components: component_records,
        abs_delta: max_abs_delta,
        atom,
        axis,
        analytic_force,
        finite_difference_force,
    })
}

fn select_components(
    components: Vec<(usize, usize)>,
    component_start: usize,
    component_limit: Option<usize>,
) -> Result<Vec<(usize, usize)>, String> {
    if component_start >= components.len() {
        return Err(format!(
            "--component-start {component_start} selects no components from {} total",
            components.len()
        ));
    }
    let end = match component_limit {
        Some(limit) => component_start.saturating_add(limit).min(components.len()),
        None => components.len(),
    };
    if end == component_start {
        return Err("--component-limit must select at least one component".to_owned());
    }
    Ok(components[component_start..end].to_vec())
}

fn fd_components(forces: &[Vec3], fd_mode: FdMode) -> Result<Vec<(usize, usize)>, String> {
    match fd_mode {
        FdMode::Representative | FdMode::MaxForce => Ok(vec![max_force_component(forces)?]),
        FdMode::Full3n => Ok((0..forces.len())
            .flat_map(|atom| (0..3).map(move |axis| (atom, axis)))
            .collect()),
    }
}

fn max_force_component(forces: &[Vec3]) -> Result<(usize, usize), String> {
    let mut best = None;
    let mut best_value = -1.0;
    for (atom, force) in forces.iter().enumerate() {
        for (axis, value) in force.iter().enumerate() {
            let abs_value = value.abs();
            if abs_value > best_value {
                best = Some((atom, axis));
                best_value = abs_value;
            }
        }
    }
    best.ok_or_else(|| "finite-difference force check requires at least one atom".to_owned())
}

fn predict_perturbed_energy(
    model: &LlmlipModel,
    structure: &Structure,
    atom: usize,
    axis: usize,
    delta: f64,
    options: &FiniteDifferenceOptions,
) -> Result<f64, String> {
    let mut perturbed = structure.clone();
    perturbed.positions[atom][axis] += delta;
    let prediction = model
        .predict(
            &perturbed,
            &PredictionOptions {
                compute_energy: true,
                total_charge: options.total_charge,
                spin_multiplicity: options.spin_multiplicity,
                selected_head: options.head.clone(),
                mace_mh1_wigner_mode: options.mace_mh1_wigner_mode,
                ..PredictionOptions::default()
            },
        )
        .map_err(predict_error)?;
    prediction
        .energy
        .ok_or_else(|| "model did not return requested energy".to_owned())
}

fn percentile(sorted_values: &[f64], pct: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    if sorted_values.len() == 1 {
        return sorted_values[0];
    }
    let rank = (sorted_values.len() - 1) as f64 * pct / 100.0;
    let low = rank.floor() as usize;
    let high = rank.ceil() as usize;
    if low == high {
        sorted_values[low]
    } else {
        let weight = rank - low as f64;
        sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

impl FdMode {
    fn report_name(self) -> &'static str {
        match self {
            Self::Representative => "representative",
            Self::MaxForce => "max_force",
            Self::Full3n => "full_3n",
        }
    }
}

const BATCH_PREDICT_SCHEMA_VERSION: &str = "llmlip-batch-predict-v1";

#[derive(Debug, Serialize)]
struct BatchPredictStructureResult {
    index: usize,
    atom_count: usize,
    unique_element_count: usize,
    elements: Vec<String>,
    pbc: [bool; 3],
    has_cell: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_charge: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    spin_multiplicity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    energy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    forces: Option<Vec<[f64; 3]>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stress: Option<[[f64; 3]; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    virial: Option<[[f64; 3]; 3]>,
    predict_ms: f64,
    graph_setup_ms: f64,
    end_to_end_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    esen_oam_profile: Option<EsenOamProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pet_lite_profile: Option<PetLiteProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct BatchPredictReport {
    schema_version: &'static str,
    model: String,
    structures_file: String,
    structure_count: usize,
    success_count: usize,
    error_count: usize,
    compute_dtype: &'static str,
    requested_outputs: BenchmarkRequestedOutputs,
    input_requirements: BatchPredictInputRequirements,
    #[serde(skip_serializing_if = "Option::is_none")]
    selected_head: Option<String>,
    model_load_ms: f64,
    file_read_ms: f64,
    total_predict_ms: f64,
    total_graph_setup_ms: f64,
    total_end_to_end_ms: f64,
    mean_predict_ms: f64,
    mean_graph_setup_ms: f64,
    mean_end_to_end_ms: f64,
    environment: BenchmarkEnvironment,
    results: Vec<BatchPredictStructureResult>,
}

#[derive(Debug, Serialize)]
struct BatchPredictInputRequirements {
    total_charge: bool,
    spin_multiplicity: bool,
}

#[derive(Clone, Copy, Debug)]
struct BatchPredictRunOptions {
    progress_interval: usize,
    start_index: usize,
    max_structures: Option<usize>,
    profile_stages: bool,
}

impl From<PredictionInputRequirements> for BatchPredictInputRequirements {
    fn from(value: PredictionInputRequirements) -> Self {
        Self {
            total_charge: value.total_charge,
            spin_multiplicity: value.spin_multiplicity,
        }
    }
}

fn unique_elements(species: &[String]) -> Vec<String> {
    let mut seen = Vec::new();
    for s in species {
        if !seen.contains(s) {
            seen.push(s.clone());
        }
    }
    seen.sort();
    seen
}

fn batch_prediction_options_for_frame(
    base: &PredictionOptions,
    metadata: ExtXyzFrameMetadata,
    requirements: PredictionInputRequirements,
) -> Result<PredictionOptions, String> {
    let mut options = base.clone();
    if requirements.total_charge && options.total_charge.is_none() {
        options.total_charge = metadata.total_charge;
    }
    if requirements.spin_multiplicity && options.spin_multiplicity.is_none() {
        options.spin_multiplicity = metadata.spin_multiplicity;
    }

    let mut missing = Vec::new();
    if requirements.total_charge && options.total_charge.is_none() {
        missing.push("total_charge");
    }
    if requirements.spin_multiplicity && options.spin_multiplicity.is_none() {
        missing.push("spin_multiplicity");
    }
    if missing.is_empty() {
        Ok(options)
    } else {
        Err(format!(
            "model requires {}; provide extxyz frame metadata or batch-predict CLI overrides",
            missing.join(" and ")
        ))
    }
}

fn run_batch_predict(
    model_path: &str,
    structures_path: &str,
    options: PredictionOptions,
    dtype: ComputeDtype,
    out_path: &str,
    run_options: BatchPredictRunOptions,
) -> Result<(), String> {
    let file_read_start = Instant::now();
    let frames = read_extxyz_frames(structures_path)
        .map_err(|err| format!("failed to read structures: {err}"))?;
    let file_read_ms = elapsed_ms(file_read_start);
    let total_structure_count = frames.len();
    if run_options.start_index > total_structure_count {
        return Err(format!(
            "--start-index {} exceeds structure count {total_structure_count}",
            run_options.start_index
        ));
    }
    let selected_end = match run_options.max_structures {
        Some(limit) => run_options
            .start_index
            .saturating_add(limit)
            .min(total_structure_count),
        None => total_structure_count,
    };
    let selected_frames = &frames[run_options.start_index..selected_end];
    let structure_count = selected_frames.len();
    eprintln!(
        "read {} structures from {} ({:.1} ms), selected {} starting at {}",
        total_structure_count,
        structures_path,
        file_read_ms,
        structure_count,
        run_options.start_index
    );

    let load_start = Instant::now();
    let model =
        LlmlipModel::load(model_path).map_err(|err| format!("failed to load model: {err}"))?;
    let model_load_ms = elapsed_ms(load_start);
    let profile_stages = run_options.profile_stages
        && matches!(&model, LlmlipModel::EsenOam(_) | LlmlipModel::PetLite(_));
    let input_requirements = model.prediction_input_requirements();
    eprintln!("loaded model from {} ({:.1} ms)", model_path, model_load_ms);

    let mut results = Vec::with_capacity(structure_count);
    let mut success_count = 0usize;
    let mut error_count = 0usize;
    let mut total_predict_ms = 0.0;
    let mut total_graph_setup_ms = 0.0;
    let mut total_end_to_end_ms = 0.0;

    for (offset, frame) in selected_frames.iter().enumerate() {
        let index = run_options.start_index + offset;
        let e2e_start = Instant::now();
        let structure = &frame.structure;
        let elements = unique_elements(&structure.species);
        let unique_element_count = elements.len();
        let frame_options =
            batch_prediction_options_for_frame(&options, frame.metadata, input_requirements);
        let mut esen_oam_profile =
            (profile_stages && matches!(&model, LlmlipModel::EsenOam(_))).then(EsenOamProfile::new);
        let mut pet_lite_profile =
            (profile_stages && matches!(&model, LlmlipModel::PetLite(_))).then(PetLiteProfile::new);
        let mut prepared_workspace = PreparedPredictionWorkspace::new();

        let (energy, forces, stress, virial, predict_ms, graph_setup_ms, error) =
            match frame_options {
                Ok(frame_options) => {
                    let graph_start = Instant::now();
                    let prepared = model.prepare_prediction_input_with_options_profiled(
                        structure,
                        &frame_options,
                        esen_oam_profile.as_mut(),
                    );
                    let graph_setup_ms = elapsed_ms(graph_start);

                    match prepared {
                        Ok(prepared_input) => {
                            let pred_start = Instant::now();
                            let pred_result = model
                                .predict_prepared_with_compute_dtype_workspace_profiled(
                                    &prepared_input,
                                    &frame_options,
                                    dtype.into(),
                                    &mut prepared_workspace,
                                    esen_oam_profile.as_mut(),
                                    pet_lite_profile.as_mut(),
                                    None,
                                );
                            let predict_ms = elapsed_ms(pred_start);
                            match pred_result {
                                Ok(prediction) => (
                                    prediction.energy,
                                    prediction.forces,
                                    prediction.stress,
                                    prediction.virial,
                                    predict_ms,
                                    graph_setup_ms,
                                    None,
                                ),
                                Err(err) => (
                                    None,
                                    None,
                                    None,
                                    None,
                                    predict_ms,
                                    graph_setup_ms,
                                    Some(format!("{err}")),
                                ),
                            }
                        }
                        Err(ModelError::PreparedPredictionNotSupported(_)) => {
                            let pred_start = Instant::now();
                            let pred_result = model.predict_with_compute_dtype(
                                structure,
                                &frame_options,
                                dtype.into(),
                            );
                            let predict_ms = elapsed_ms(pred_start);
                            match pred_result {
                                Ok(prediction) => (
                                    prediction.energy,
                                    prediction.forces,
                                    prediction.stress,
                                    prediction.virial,
                                    predict_ms,
                                    0.0,
                                    None,
                                ),
                                Err(err) => (
                                    None,
                                    None,
                                    None,
                                    None,
                                    predict_ms,
                                    0.0,
                                    Some(format!("{err}")),
                                ),
                            }
                        }
                        Err(err) => (
                            None,
                            None,
                            None,
                            None,
                            0.0,
                            graph_setup_ms,
                            Some(format!("{err}")),
                        ),
                    }
                }
                Err(err) => (None, None, None, None, 0.0, 0.0, Some(err)),
            };

        let end_to_end_ms = elapsed_ms(e2e_start);
        total_predict_ms += predict_ms;
        total_graph_setup_ms += graph_setup_ms;
        total_end_to_end_ms += end_to_end_ms;

        if error.is_some() {
            error_count += 1;
        } else {
            success_count += 1;
        }

        results.push(BatchPredictStructureResult {
            index,
            atom_count: structure.len(),
            unique_element_count,
            elements,
            pbc: structure.pbc,
            has_cell: structure.cell.is_some(),
            total_charge: frame
                .metadata
                .total_charge
                .or(if input_requirements.total_charge {
                    options.total_charge
                } else {
                    None
                }),
            spin_multiplicity: frame.metadata.spin_multiplicity.or(
                if input_requirements.spin_multiplicity {
                    options.spin_multiplicity
                } else {
                    None
                },
            ),
            energy,
            forces: forces.map(|f| f.into_iter().collect()),
            stress,
            virial,
            predict_ms,
            graph_setup_ms,
            end_to_end_ms,
            esen_oam_profile,
            pet_lite_profile,
            error,
        });

        if run_options.progress_interval > 0 && (offset + 1) % run_options.progress_interval == 0 {
            eprintln!(
                "  [{}/{}] {:.0}% done, last structure: {} atoms, {:.2} ms predict",
                offset + 1,
                structure_count,
                (offset + 1) as f64 / structure_count as f64 * 100.0,
                structure.len(),
                predict_ms,
            );
        }
    }

    let mean_predict_ms = if structure_count > 0 {
        total_predict_ms / structure_count as f64
    } else {
        0.0
    };
    let mean_graph_setup_ms = if structure_count > 0 {
        total_graph_setup_ms / structure_count as f64
    } else {
        0.0
    };
    let mean_end_to_end_ms = if structure_count > 0 {
        total_end_to_end_ms / structure_count as f64
    } else {
        0.0
    };

    let report = BatchPredictReport {
        schema_version: BATCH_PREDICT_SCHEMA_VERSION,
        model: model_path.to_owned(),
        structures_file: structures_path.to_owned(),
        structure_count,
        success_count,
        error_count,
        compute_dtype: dtype.report_name(),
        requested_outputs: BenchmarkRequestedOutputs {
            energy: options.compute_energy,
            forces: options.compute_forces,
            stress: options.compute_stress,
            virial: options.compute_virial,
        },
        input_requirements: input_requirements.into(),
        selected_head: options.selected_head,
        model_load_ms,
        file_read_ms,
        total_predict_ms,
        total_graph_setup_ms,
        total_end_to_end_ms,
        mean_predict_ms,
        mean_graph_setup_ms,
        mean_end_to_end_ms,
        environment: benchmark_environment(),
        results,
    };

    let json = serde_json::to_string_pretty(&report)
        .map_err(|err| format!("failed to serialize batch report: {err}"))?;
    fs::write(out_path, &json)
        .map_err(|err| format!("failed to write batch report to {out_path}: {err}"))?;
    eprintln!(
        "batch-predict complete: {}/{} succeeded, {} errors, total predict {:.1} ms, output: {}",
        success_count, structure_count, error_count, total_predict_ms, out_path
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_fixture_reader_preserves_pbc_cell() -> Result<(), String> {
        let unique = std::process::id();
        let path = std::env::temp_dir().join(format!("llmlip-benchmark-fixture-{unique}.json"));
        fs::write(
            &path,
            r#"{
  "schema_version": "nequix-benchmark-structure-v1",
  "structures": [
    {
      "name": "pbc_probe",
      "structure": {
        "symbols": ["Fe", "Co"],
        "positions_angstrom": [[0.0, 0.0, 0.0], [1.0, 1.5, 2.0]],
        "cell": [[6.1, 0.0, 0.0], [0.0, 6.2, 0.0], [0.0, 0.0, 6.3]],
        "pbc": [true, true, true]
      }
    }
  ]
}
"#,
        )
        .map_err(|err| format!("failed to write temporary fixture: {err}"))?;

        let fixture_path = path.to_string_lossy().into_owned();
        let structures = read_benchmark_fixture_structures(&fixture_path)?;
        fs::remove_file(path)
            .map_err(|err| format!("failed to remove temporary fixture: {err}"))?;

        assert_eq!(structures.len(), 1);
        assert_eq!(structures[0].pbc, [true, true, true]);
        let cell = structures[0]
            .cell
            .ok_or_else(|| "PBC fixture should preserve cell".to_owned())?;
        assert_eq!(
            cell.vectors,
            [[6.1, 0.0, 0.0], [0.0, 6.2, 0.0], [0.0, 0.0, 6.3]]
        );
        Ok(())
    }

    #[test]
    fn batch_options_fill_required_charge_spin_from_frame_metadata() -> Result<(), String> {
        let base = PredictionOptions {
            compute_energy: true,
            ..PredictionOptions::default()
        };
        let options = batch_prediction_options_for_frame(
            &base,
            ExtXyzFrameMetadata {
                total_charge: Some(-1.0),
                spin_multiplicity: Some(2.0),
            },
            PredictionInputRequirements::charge_spin(),
        )?;

        assert_eq!(options.total_charge, Some(-1.0));
        assert_eq!(options.spin_multiplicity, Some(2.0));
        Ok(())
    }

    #[test]
    fn batch_options_cli_charge_spin_override_frame_metadata() -> Result<(), String> {
        let base = PredictionOptions {
            total_charge: Some(3.0),
            spin_multiplicity: Some(4.0),
            ..PredictionOptions::default()
        };
        let options = batch_prediction_options_for_frame(
            &base,
            ExtXyzFrameMetadata {
                total_charge: Some(-1.0),
                spin_multiplicity: Some(2.0),
            },
            PredictionInputRequirements::charge_spin(),
        )?;

        assert_eq!(options.total_charge, Some(3.0));
        assert_eq!(options.spin_multiplicity, Some(4.0));
        Ok(())
    }

    #[test]
    fn batch_options_ignore_frame_charge_spin_when_model_does_not_require_them(
    ) -> Result<(), String> {
        let base = PredictionOptions::default();
        let options = batch_prediction_options_for_frame(
            &base,
            ExtXyzFrameMetadata {
                total_charge: Some(-1.0),
                spin_multiplicity: Some(2.0),
            },
            PredictionInputRequirements::none(),
        )?;

        assert_eq!(options.total_charge, None);
        assert_eq!(options.spin_multiplicity, None);
        Ok(())
    }

    #[test]
    fn batch_options_fail_closed_when_required_charge_spin_is_missing() -> Result<(), String> {
        let result = batch_prediction_options_for_frame(
            &PredictionOptions::default(),
            ExtXyzFrameMetadata {
                total_charge: Some(-1.0),
                spin_multiplicity: None,
            },
            PredictionInputRequirements::charge_spin(),
        );

        match result {
            Ok(_) => Err("missing required spin should fail".to_owned()),
            Err(err) => {
                assert!(err.contains("spin_multiplicity"));
                Ok(())
            }
        }
    }
}
