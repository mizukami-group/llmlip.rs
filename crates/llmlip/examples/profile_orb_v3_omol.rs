use llmlip::{LlmlipModel, PredictionComputeDtype, Structure};
use serde::Serialize;
use serde_json::Value;
#[cfg(feature = "orb-profile-breakdown")]
use std::collections::BTreeMap;
#[cfg(feature = "orb-profile-breakdown")]
use std::collections::BTreeSet;
use std::process::Command;
use std::{env, error::Error, fs, hint::black_box, io::Write};

const SCHEMA_VERSION: &str = "orb-v3-omol-profile-v6";
const THREAD_ENV_KEYS: [&str; 7] = [
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MKL_DYNAMIC",
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
];

#[derive(Clone, Debug)]
struct Config {
    label: String,
    model_path: String,
    fixture_path: String,
    case_index: usize,
    warmup: usize,
    iterations: usize,
    total_charge: f64,
    spin_multiplicity: f64,
    dtype: PredictionComputeDtype,
    out_path: Option<String>,
    include_forces: bool,
}

#[derive(Default)]
struct ProfileTotals {
    total_ms: f64,
    input_features_ms: f64,
    encoder_forward_ms: f64,
    gnn_weight_plan_ms: f64,
    gnn_forward_ms: f64,
    energy_tail_vjp_ms: f64,
    final_cotangent_ms: f64,
    gnn_input_vjp_ms: f64,
    encoder_vjp_ms: f64,
    geometry_forces_ms: f64,
    zbl_force_accum_ms: f64,
    force_vjp_ms: f64,
    blas_ms: f64,
    activation_ms: f64,
    scatter_reduction_ms: f64,
    checksum: f64,
    total_energy: f64,
    force_checksum: f64,
    force_l2_norm: f64,
    force_max_abs: f64,
    #[cfg(feature = "orb-profile-breakdown")]
    blas_shapes: BTreeMap<BlasShapeKey, BlasShapeTotals>,
    #[cfg(feature = "orb-profile-breakdown")]
    blas_families: BTreeMap<&'static str, BlasFamilyTotals>,
    #[cfg(feature = "orb-profile-breakdown")]
    kernels: BTreeMap<KernelKey, KernelTotals>,
}

#[cfg(feature = "orb-profile-breakdown")]
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct BlasShapeKey {
    label: String,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    full_input_dim: usize,
    column_offset: usize,
}

#[cfg(feature = "orb-profile-breakdown")]
#[derive(Default)]
struct BlasShapeTotals {
    calls: usize,
    ms: f64,
    samples_ms: Vec<f64>,
}

#[cfg(feature = "orb-profile-breakdown")]
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct KernelKey {
    label: String,
    rows: usize,
    width: usize,
}

#[cfg(feature = "orb-profile-breakdown")]
#[derive(Default)]
struct KernelTotals {
    calls: usize,
    ms: f64,
    samples_ms: Vec<f64>,
}

#[cfg(feature = "orb-profile-breakdown")]
#[derive(Default)]
struct BlasFamilyTotals {
    shape_keys: BTreeSet<BlasShapeKey>,
    calls: usize,
    ms: f64,
    samples_ms: Vec<f64>,
}

#[derive(Serialize)]
struct ProfileReport {
    schema_version: &'static str,
    label: String,
    model_path: String,
    fixture_path: String,
    case_index: usize,
    atom_count: usize,
    edge_count: usize,
    warmup: usize,
    iterations: usize,
    total_charge: f64,
    spin_multiplicity: f64,
    compute_dtype: &'static str,
    checksum: f64,
    mean_total_energy: f64,
    mean_force_checksum: f64,
    mean_force_l2_norm: f64,
    max_force_abs: f64,
    forces: Option<Vec<[f64; 3]>>,
    measurement_scope: MeasurementScopeReport,
    provenance: ProvenanceReport,
    backend: BackendReport,
    thread_env: Vec<EnvReport>,
    samples_ms: Vec<f64>,
    samples: Vec<SampleReport>,
    sample_stats: SampleStatsReport,
    stages: StageReport,
    blas_shapes: Vec<BlasShapeReport>,
    blas_families: Vec<BlasFamilyReport>,
    kernels: Vec<KernelReport>,
}

#[derive(Serialize)]
struct MeasurementScopeReport {
    scope: &'static str,
    model_load_excluded: bool,
    fixture_read_excluded: bool,
    graph_build_excluded: bool,
    timed_operation: &'static str,
}

#[derive(Serialize)]
struct BackendReport {
    target_os: &'static str,
    target_arch: &'static str,
    accelerate_blas: bool,
    intel_mkl: bool,
    intel_mkl_threaded: bool,
    openblas: bool,
    libxsmm: bool,
    orb_fixed_shape_dense: bool,
    orb_fixed_shape_dense_pack_a: bool,
    orb_f32_edge_mlp: bool,
    parallel: bool,
    orb_profile_breakdown: bool,
}

#[derive(Serialize)]
struct ProvenanceReport {
    git_commit: Option<String>,
    git_dirty: Option<bool>,
    rustc_verbose_version: Option<String>,
    enabled_cargo_features: Vec<&'static str>,
}

#[derive(Serialize)]
struct EnvReport {
    key: &'static str,
    value: Option<String>,
}

#[derive(Serialize)]
struct StageReport {
    total_ms: f64,
    input_features_ms: f64,
    encoder_forward_ms: f64,
    gnn_weight_plan_ms: f64,
    gnn_forward_ms: f64,
    energy_tail_vjp_ms: f64,
    final_cotangent_ms: f64,
    gnn_input_vjp_ms: f64,
    encoder_vjp_ms: f64,
    geometry_forces_ms: f64,
    zbl_force_accum_ms: f64,
    force_vjp_ms: f64,
    blas_ms: f64,
    activation_ms: f64,
    scatter_reduction_ms: f64,
    non_bucketed_ms: f64,
}

#[derive(Clone, Serialize)]
struct DistributionReport {
    mean_ms: f64,
    median_ms: f64,
    min_ms: f64,
    p95_ms: f64,
    stdev_ms: f64,
    samples: usize,
}

#[derive(Serialize)]
struct SampleStatsReport {
    total_ms: DistributionReport,
}

#[derive(Serialize)]
struct SampleReport {
    iteration: usize,
    total_ms: f64,
    total_energy: f64,
    force_checksum: f64,
    force_l2_norm: f64,
    force_max_abs: f64,
    stages: StageReport,
}

#[derive(Serialize)]
struct BlasShapeReport {
    label: String,
    calls: f64,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    full_input_dim: usize,
    column_offset: usize,
    mean_ms: f64,
    median_ms: f64,
    min_ms: f64,
    p95_ms: f64,
    stdev_ms: f64,
    sample_count: usize,
    share_pct: f64,
    flop_count: f64,
    gflop: f64,
    gflops_per_s: f64,
}

#[derive(Serialize)]
struct BlasFamilyReport {
    family: String,
    shapes: usize,
    calls: f64,
    mean_ms: f64,
    median_ms: f64,
    min_ms: f64,
    p95_ms: f64,
    stdev_ms: f64,
    sample_count: usize,
    share_pct: f64,
    gflop: f64,
    gflops_per_s: f64,
}

#[derive(Serialize)]
struct KernelReport {
    label: String,
    calls: f64,
    rows: usize,
    width: usize,
    mean_ms: f64,
    median_ms: f64,
    min_ms: f64,
    p95_ms: f64,
    stdev_ms: f64,
    sample_count: usize,
    share_pct: f64,
}

struct ReportPayload {
    atom_count: usize,
    edge_count: usize,
    samples_ms: Vec<f64>,
    samples: Vec<SampleReport>,
    sample_stats: SampleStatsReport,
    stages: StageReport,
    blas_shapes: Vec<BlasShapeReport>,
    blas_families: Vec<BlasFamilyReport>,
    kernels: Vec<KernelReport>,
    forces: Option<Vec<[f64; 3]>>,
}

impl Config {
    fn from_args() -> Result<Self, Box<dyn Error>> {
        let mut cfg = Self {
            label: "rust_orb_v3_omol_profile".to_owned(),
            model_path:
                "outputs/reference-artifacts/orb/orb_v3_conservative_omol_state_export.e3rs"
                    .to_owned(),
            fixture_path:
                "outputs/benchmarks/orb/orb_decane_tetramer_128_atoms_benchmark_fixture.json"
                    .to_owned(),
            case_index: 0,
            warmup: 1,
            iterations: 3,
            // ORB's pinned public default for omitted charge/spin metadata is
            // 0.0/0.0, matching the local reference fixtures and benchmarks.
            total_charge: 0.0,
            spin_multiplicity: 0.0,
            dtype: PredictionComputeDtype::Float64,
            out_path: None,
            include_forces: false,
        };

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--label" => cfg.label = args.next().ok_or("--label requires a value")?,
                "--model" => cfg.model_path = args.next().ok_or("--model requires a path")?,
                "--fixture" => cfg.fixture_path = args.next().ok_or("--fixture requires a path")?,
                "--case-index" => {
                    cfg.case_index = args
                        .next()
                        .ok_or("--case-index requires a value")?
                        .parse()?;
                }
                "--warmup" => {
                    cfg.warmup = args.next().ok_or("--warmup requires a value")?.parse()?
                }
                "--iterations" => {
                    cfg.iterations = args
                        .next()
                        .ok_or("--iterations requires a value")?
                        .parse()?;
                }
                "--total-charge" => {
                    cfg.total_charge = args
                        .next()
                        .ok_or("--total-charge requires a value")?
                        .parse()?;
                }
                "--spin-multiplicity" => {
                    cfg.spin_multiplicity = args
                        .next()
                        .ok_or("--spin-multiplicity requires a value")?
                        .parse()?;
                }
                "--dtype" => {
                    cfg.dtype =
                        parse_compute_dtype(&args.next().ok_or("--dtype requires a value")?)?;
                }
                "--out" => cfg.out_path = Some(args.next().ok_or("--out requires a path")?),
                "--include-forces" => cfg.include_forces = true,
                _ => return Err(format!("unknown argument: {arg}").into()),
            }
        }
        if cfg.iterations == 0 {
            return Err("--iterations must be at least 1".into());
        }
        Ok(cfg)
    }
}

fn parse_compute_dtype(value: &str) -> Result<PredictionComputeDtype, Box<dyn Error>> {
    match value {
        "f64" | "float64" => Ok(PredictionComputeDtype::Float64),
        "f32" | "float32" => Ok(PredictionComputeDtype::Float32),
        _ => Err("--dtype must be one of f64, float64, f32, or float32".into()),
    }
}

fn compute_dtype_label(dtype: PredictionComputeDtype) -> &'static str {
    match dtype {
        PredictionComputeDtype::Float64 => "float64",
        PredictionComputeDtype::Float32 => "float32",
    }
}

impl ProfileTotals {
    fn add(&mut self, profile: &llmlip::orb_v3_omol::OrbV3OmolEnergyForcesProfile) {
        self.total_ms += profile.total_ms;
        self.input_features_ms += profile.input_features_ms;
        self.encoder_forward_ms += profile.encoder_forward_ms;
        self.gnn_weight_plan_ms += profile.gnn_weight_plan_ms;
        self.gnn_forward_ms += profile.gnn_forward_ms;
        self.energy_tail_vjp_ms += profile.energy_tail_vjp_ms;
        self.final_cotangent_ms += profile.final_cotangent_ms;
        self.gnn_input_vjp_ms += profile.gnn_input_vjp_ms;
        self.encoder_vjp_ms += profile.encoder_vjp_ms;
        self.geometry_forces_ms += profile.geometry_forces_ms;
        self.zbl_force_accum_ms += profile.zbl_force_accum_ms;
        self.force_vjp_ms += profile.force_vjp_ms;
        self.blas_ms += profile.blas_ms;
        self.activation_ms += profile.activation_ms;
        self.scatter_reduction_ms += profile.scatter_reduction_ms;
        self.checksum += profile.total_energy + profile.force_checksum;
        self.total_energy += profile.total_energy;
        self.force_checksum += profile.force_checksum;
        self.force_l2_norm += profile.force_l2_norm;
        self.force_max_abs = self.force_max_abs.max(profile.force_max_abs);
        #[cfg(feature = "orb-profile-breakdown")]
        for shape in &profile.blas_shapes {
            let shape_key = BlasShapeKey {
                label: shape.label.clone(),
                rows: shape.rows,
                input_dim: shape.input_dim,
                output_dim: shape.output_dim,
                full_input_dim: shape.full_input_dim,
                column_offset: shape.column_offset,
            };
            let entry = self.blas_shapes.entry(shape_key.clone()).or_default();
            entry.calls += shape.calls;
            entry.ms += shape.ms;
            entry.samples_ms.push(shape.ms);

            let family = blas_family_name_from_parts(
                &shape.label,
                shape.rows,
                shape.input_dim,
                shape.output_dim,
                shape.full_input_dim,
                shape.column_offset,
            );
            let family_entry = self.blas_families.entry(family).or_default();
            family_entry.shape_keys.insert(shape_key);
            family_entry.calls += shape.calls;
            family_entry.ms += shape.ms;
        }
        #[cfg(feature = "orb-profile-breakdown")]
        {
            let mut sample_families = BTreeMap::<&'static str, f64>::new();
            for shape in &profile.blas_shapes {
                let family = blas_family_name_from_parts(
                    &shape.label,
                    shape.rows,
                    shape.input_dim,
                    shape.output_dim,
                    shape.full_input_dim,
                    shape.column_offset,
                );
                *sample_families.entry(family).or_default() += shape.ms;
            }
            for (family, sample_ms) in sample_families {
                self.blas_families
                    .entry(family)
                    .or_default()
                    .samples_ms
                    .push(sample_ms);
            }
        }
        #[cfg(feature = "orb-profile-breakdown")]
        for kernel in &profile.kernels {
            let entry = self
                .kernels
                .entry(KernelKey {
                    label: kernel.label.clone(),
                    rows: kernel.rows,
                    width: kernel.width,
                })
                .or_default();
            entry.calls += kernel.calls;
            entry.ms += kernel.ms;
            entry.samples_ms.push(kernel.ms);
        }
    }

    fn mean(&self, value: f64, iterations: usize) -> f64 {
        value / iterations as f64
    }
}

fn read_fixture_structure(path: &str, case_index: usize) -> Result<Structure, Box<dyn Error>> {
    let value: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    let cases = value["structures"]
        .as_array()
        .ok_or("fixture must contain structures array")?;
    let case = cases
        .get(case_index)
        .ok_or("fixture case index is out of range")?;
    let structure = &case["structure"];
    let species = structure["symbols"]
        .as_array()
        .ok_or("structure must contain symbols")?
        .iter()
        .map(|symbol| {
            symbol
                .as_str()
                .map(str::to_owned)
                .ok_or("symbol must be a string")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let positions = structure["positions_angstrom"]
        .as_array()
        .ok_or("structure must contain positions_angstrom")?
        .iter()
        .map(vec3_from_json)
        .collect::<Result<Vec<_>, _>>()?;
    let pbc = &structure["pbc"];
    if pbc
        .as_array()
        .is_some_and(|values| values.iter().any(|value| value.as_bool() == Some(true)))
    {
        return Err("ORB profiling example supports only non-PBC fixtures".into());
    }
    Ok(Structure::new(species, positions))
}

fn vec3_from_json(value: &Value) -> Result<[f64; 3], &'static str> {
    let row = value.as_array().ok_or("position row must be an array")?;
    if row.len() != 3 {
        return Err("position row must have length 3");
    }
    Ok([
        row.first()
            .and_then(Value::as_f64)
            .ok_or("position x must be f64")?,
        row.get(1)
            .and_then(Value::as_f64)
            .ok_or("position y must be f64")?,
        row.get(2)
            .and_then(Value::as_f64)
            .ok_or("position z must be f64")?,
    ])
}

fn print_stage(name: &str, mean_ms: f64, total_ms: f64) {
    let share = share_pct(mean_ms, total_ms);
    println!("{name:<24} {mean_ms:>12.3} {share:>10.1}");
}

fn share_pct(mean_ms: f64, total_ms: f64) -> f64 {
    let share = if total_ms > 0.0 {
        100.0 * mean_ms / total_ms
    } else {
        0.0
    };
    if share.is_finite() {
        share
    } else {
        0.0
    }
}

fn percentile(sorted_samples: &[f64], percentile: f64) -> f64 {
    if sorted_samples.is_empty() {
        return 0.0;
    }
    if sorted_samples.len() == 1 {
        return sorted_samples[0];
    }
    let rank = (sorted_samples.len() - 1) as f64 * percentile.clamp(0.0, 1.0);
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted_samples[lower]
    } else {
        let fraction = rank - lower as f64;
        sorted_samples[lower] * (1.0 - fraction) + sorted_samples[upper] * fraction
    }
}

fn distribution_report(samples: &[f64]) -> DistributionReport {
    if samples.is_empty() {
        return DistributionReport {
            mean_ms: 0.0,
            median_ms: 0.0,
            min_ms: 0.0,
            p95_ms: 0.0,
            stdev_ms: 0.0,
            samples: 0,
        };
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let mut sorted = samples.to_vec();
    sorted.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.len().is_multiple_of(2) {
        let upper = sorted.len() / 2;
        0.5 * (sorted[upper - 1] + sorted[upper])
    } else {
        sorted[sorted.len() / 2]
    };
    let variance = samples
        .iter()
        .map(|sample| {
            let delta = sample - mean;
            delta * delta
        })
        .sum::<f64>()
        / samples.len() as f64;
    DistributionReport {
        mean_ms: mean,
        median_ms: median,
        min_ms: sorted[0],
        p95_ms: percentile(&sorted, 0.95),
        stdev_ms: variance.sqrt(),
        samples: samples.len(),
    }
}

fn stage_report_from_profile(
    profile: &llmlip::orb_v3_omol::OrbV3OmolEnergyForcesProfile,
) -> StageReport {
    StageReport {
        total_ms: profile.total_ms,
        input_features_ms: profile.input_features_ms,
        encoder_forward_ms: profile.encoder_forward_ms,
        gnn_weight_plan_ms: profile.gnn_weight_plan_ms,
        gnn_forward_ms: profile.gnn_forward_ms,
        energy_tail_vjp_ms: profile.energy_tail_vjp_ms,
        final_cotangent_ms: profile.final_cotangent_ms,
        gnn_input_vjp_ms: profile.gnn_input_vjp_ms,
        encoder_vjp_ms: profile.encoder_vjp_ms,
        geometry_forces_ms: profile.geometry_forces_ms,
        zbl_force_accum_ms: profile.zbl_force_accum_ms,
        force_vjp_ms: profile.force_vjp_ms,
        blas_ms: profile.blas_ms,
        activation_ms: profile.activation_ms,
        scatter_reduction_ms: profile.scatter_reduction_ms,
        non_bucketed_ms: profile.total_ms
            - profile.blas_ms
            - profile.activation_ms
            - profile.scatter_reduction_ms,
    }
}

fn stage_report(totals: &ProfileTotals, iterations: usize) -> StageReport {
    let total_ms = totals.mean(totals.total_ms, iterations);
    let blas_ms = totals.mean(totals.blas_ms, iterations);
    let activation_ms = totals.mean(totals.activation_ms, iterations);
    let scatter_reduction_ms = totals.mean(totals.scatter_reduction_ms, iterations);
    StageReport {
        total_ms,
        input_features_ms: totals.mean(totals.input_features_ms, iterations),
        encoder_forward_ms: totals.mean(totals.encoder_forward_ms, iterations),
        gnn_weight_plan_ms: totals.mean(totals.gnn_weight_plan_ms, iterations),
        gnn_forward_ms: totals.mean(totals.gnn_forward_ms, iterations),
        energy_tail_vjp_ms: totals.mean(totals.energy_tail_vjp_ms, iterations),
        final_cotangent_ms: totals.mean(totals.final_cotangent_ms, iterations),
        gnn_input_vjp_ms: totals.mean(totals.gnn_input_vjp_ms, iterations),
        encoder_vjp_ms: totals.mean(totals.encoder_vjp_ms, iterations),
        geometry_forces_ms: totals.mean(totals.geometry_forces_ms, iterations),
        zbl_force_accum_ms: totals.mean(totals.zbl_force_accum_ms, iterations),
        force_vjp_ms: totals.mean(totals.force_vjp_ms, iterations),
        blas_ms,
        activation_ms,
        scatter_reduction_ms,
        non_bucketed_ms: total_ms - blas_ms - activation_ms - scatter_reduction_ms,
    }
}

fn backend_report() -> BackendReport {
    BackendReport {
        target_os: env::consts::OS,
        target_arch: env::consts::ARCH,
        accelerate_blas: cfg!(feature = "accelerate-blas"),
        intel_mkl: cfg!(feature = "intel-mkl"),
        intel_mkl_threaded: cfg!(feature = "intel-mkl-threaded"),
        openblas: cfg!(feature = "openblas"),
        libxsmm: cfg!(feature = "libxsmm"),
        orb_fixed_shape_dense: cfg!(feature = "orb-fixed-shape-dense"),
        orb_fixed_shape_dense_pack_a: cfg!(feature = "orb-fixed-shape-dense-pack-a"),
        orb_f32_edge_mlp: cfg!(feature = "orb-f32-edge-mlp"),
        parallel: cfg!(feature = "parallel"),
        orb_profile_breakdown: cfg!(feature = "orb-profile-breakdown"),
    }
}

fn command_stdout(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    Some(stdout.trim().to_owned())
}

fn provenance_report() -> ProvenanceReport {
    let git_commit = command_stdout("git", &["rev-parse", "HEAD"]);
    let git_dirty =
        command_stdout("git", &["status", "--porcelain"]).map(|status| !status.trim().is_empty());
    let rustc_verbose_version = command_stdout("rustc", &["-vV"]);
    let mut enabled_cargo_features = Vec::new();
    for (name, enabled) in [
        ("accelerate-blas", cfg!(feature = "accelerate-blas")),
        ("intel-mkl", cfg!(feature = "intel-mkl")),
        ("intel-mkl-threaded", cfg!(feature = "intel-mkl-threaded")),
        ("openblas", cfg!(feature = "openblas")),
        ("libxsmm", cfg!(feature = "libxsmm")),
        (
            "orb-fixed-shape-dense",
            cfg!(feature = "orb-fixed-shape-dense"),
        ),
        (
            "orb-fixed-shape-dense-pack-a",
            cfg!(feature = "orb-fixed-shape-dense-pack-a"),
        ),
        ("orb-f32-edge-mlp", cfg!(feature = "orb-f32-edge-mlp")),
        ("parallel", cfg!(feature = "parallel")),
        (
            "orb-profile-breakdown",
            cfg!(feature = "orb-profile-breakdown"),
        ),
    ] {
        if enabled {
            enabled_cargo_features.push(name);
        }
    }
    ProvenanceReport {
        git_commit,
        git_dirty,
        rustc_verbose_version,
        enabled_cargo_features,
    }
}

fn measurement_scope_report() -> MeasurementScopeReport {
    MeasurementScopeReport {
        scope: "prepared_non_pbc_graph_energy_forces",
        model_load_excluded: true,
        fixture_read_excluded: true,
        graph_build_excluded: true,
        timed_operation: "LlmlipModel::OrbV3Omol::profile_energy_forces_for_prepared_graph",
    }
}

fn thread_env_report() -> Vec<EnvReport> {
    THREAD_ENV_KEYS
        .iter()
        .map(|&key| EnvReport {
            key,
            value: env::var(key).ok(),
        })
        .collect()
}

#[cfg(feature = "orb-profile-breakdown")]
fn dense_shape_flop_count(calls: f64, rows: usize, input_dim: usize, output_dim: usize) -> f64 {
    2.0 * calls * rows as f64 * input_dim as f64 * output_dim as f64
}

#[cfg(feature = "orb-profile-breakdown")]
fn gflops_per_second(flop_count: f64, mean_ms: f64) -> f64 {
    if mean_ms > 0.0 {
        flop_count / (mean_ms * 1.0e6)
    } else {
        0.0
    }
}

fn blas_shape_reports(
    totals: &ProfileTotals,
    iterations: usize,
    total_ms: f64,
) -> Vec<BlasShapeReport> {
    #[cfg(feature = "orb-profile-breakdown")]
    {
        let mut shapes = totals.blas_shapes.iter().collect::<Vec<_>>();
        shapes.sort_by(|(_, lhs), (_, rhs)| {
            rhs.ms
                .partial_cmp(&lhs.ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        shapes
            .into_iter()
            .map(|(key, value)| {
                let mean_ms = value.ms / iterations as f64;
                let stats = distribution_report(&value.samples_ms);
                let calls = value.calls as f64 / iterations as f64;
                let flop_count =
                    dense_shape_flop_count(calls, key.rows, key.input_dim, key.output_dim);
                BlasShapeReport {
                    label: key.label.clone(),
                    calls,
                    rows: key.rows,
                    input_dim: key.input_dim,
                    output_dim: key.output_dim,
                    full_input_dim: key.full_input_dim,
                    column_offset: key.column_offset,
                    mean_ms,
                    median_ms: stats.median_ms,
                    min_ms: stats.min_ms,
                    p95_ms: stats.p95_ms,
                    stdev_ms: stats.stdev_ms,
                    sample_count: stats.samples,
                    share_pct: share_pct(mean_ms, total_ms),
                    flop_count,
                    gflop: flop_count / 1.0e9,
                    gflops_per_s: gflops_per_second(flop_count, mean_ms),
                }
            })
            .collect()
    }
    #[cfg(not(feature = "orb-profile-breakdown"))]
    {
        let _ = (totals, iterations, total_ms);
        Vec::new()
    }
}

#[cfg(feature = "orb-profile-breakdown")]
fn blas_family_name_from_parts(
    label: &str,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    full_input_dim: usize,
    column_offset: usize,
) -> &'static str {
    if label.starts_with("edge_mlp_layer1_forward")
        || label.starts_with("edge_mlp_layer1_input_vjp")
    {
        return "edge_mlp_layer1_full";
    }
    if label.starts_with("edge_mlp_layer2_forward")
        || label.starts_with("edge_mlp_layer2_input_vjp")
    {
        return "edge_mlp_layer2_full";
    }
    if label.starts_with("edge_mlp_layer0_edge_")
        || ((label == "linear_forward_column_block_add" || label == "linear_vjp_column_block_add")
            && rows == 2304
            && input_dim == 256
            && output_dim == 1024
            && full_input_dim == 768
            && column_offset == 0)
    {
        return "edge_mlp_layer0_edge_column";
    }
    if label.starts_with("edge_mlp_layer0_sender_")
        || label.starts_with("edge_mlp_layer0_receiver_")
        || ((label == "linear_forward_column_block_add" || label == "linear_vjp_column_block_add")
            && rows == 128
            && input_dim == 256
            && output_dim == 1024
            && full_input_dim == 768
            && (column_offset == 256 || column_offset == 512))
    {
        return "edge_mlp_layer0_node_column";
    }
    if label.starts_with("encoder_edge_mlp_")
        || (rows == 2304 && full_input_dim == 128)
        || (rows == 2304
            && full_input_dim == 1024
            && (output_dim == 1024 || output_dim == 256)
            && (label == "linear_forward" || label == "linear_input_vjp"))
    {
        return "encoder_edge_dense";
    }
    if rows == 128 {
        return "node_row_dense";
    }
    if rows == 2304 {
        return "edge_row_dense_other";
    }
    "other_dense"
}

#[cfg(feature = "orb-profile-breakdown")]
fn blas_family_name(shape: &BlasShapeReport) -> &'static str {
    blas_family_name_from_parts(
        &shape.label,
        shape.rows,
        shape.input_dim,
        shape.output_dim,
        shape.full_input_dim,
        shape.column_offset,
    )
}

fn blas_family_reports(
    totals: &ProfileTotals,
    shapes: &[BlasShapeReport],
    total_ms: f64,
) -> Vec<BlasFamilyReport> {
    #[cfg(feature = "orb-profile-breakdown")]
    {
        let mut reports = totals
            .blas_families
            .iter()
            .map(|(family, totals)| {
                let sample_count = totals.samples_ms.len().max(1);
                let mean_ms = totals.ms / sample_count as f64;
                let stats = distribution_report(&totals.samples_ms);
                let gflop = shapes
                    .iter()
                    .filter(|shape| blas_family_name(shape) == *family)
                    .map(|shape| shape.gflop)
                    .sum::<f64>();
                BlasFamilyReport {
                    family: (*family).to_owned(),
                    shapes: totals.shape_keys.len(),
                    calls: totals.calls as f64 / sample_count as f64,
                    mean_ms,
                    median_ms: stats.median_ms,
                    min_ms: stats.min_ms,
                    p95_ms: stats.p95_ms,
                    stdev_ms: stats.stdev_ms,
                    sample_count: stats.samples,
                    share_pct: share_pct(mean_ms, total_ms),
                    gflop,
                    gflops_per_s: gflops_per_second(gflop * 1.0e9, mean_ms),
                }
            })
            .collect::<Vec<_>>();
        reports.sort_by(|lhs, rhs| {
            rhs.mean_ms
                .partial_cmp(&lhs.mean_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        reports
    }
    #[cfg(not(feature = "orb-profile-breakdown"))]
    {
        let _ = (totals, shapes, total_ms);
        Vec::new()
    }
}

fn kernel_reports(totals: &ProfileTotals, iterations: usize, total_ms: f64) -> Vec<KernelReport> {
    #[cfg(feature = "orb-profile-breakdown")]
    {
        let mut kernels = totals.kernels.iter().collect::<Vec<_>>();
        kernels.sort_by(|(_, lhs), (_, rhs)| {
            rhs.ms
                .partial_cmp(&lhs.ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        kernels
            .into_iter()
            .map(|(key, value)| {
                let mean_ms = value.ms / iterations as f64;
                let stats = distribution_report(&value.samples_ms);
                KernelReport {
                    label: key.label.clone(),
                    calls: value.calls as f64 / iterations as f64,
                    rows: key.rows,
                    width: key.width,
                    mean_ms,
                    median_ms: stats.median_ms,
                    min_ms: stats.min_ms,
                    p95_ms: stats.p95_ms,
                    stdev_ms: stats.stdev_ms,
                    sample_count: stats.samples,
                    share_pct: share_pct(mean_ms, total_ms),
                }
            })
            .collect()
    }
    #[cfg(not(feature = "orb-profile-breakdown"))]
    {
        let _ = (totals, iterations, total_ms);
        Vec::new()
    }
}

fn write_report(
    cfg: &Config,
    totals: &ProfileTotals,
    payload: ReportPayload,
) -> Result<(), Box<dyn Error>> {
    let Some(path) = &cfg.out_path else {
        return Ok(());
    };
    let report = ProfileReport {
        schema_version: SCHEMA_VERSION,
        label: cfg.label.clone(),
        model_path: cfg.model_path.clone(),
        fixture_path: cfg.fixture_path.clone(),
        case_index: cfg.case_index,
        atom_count: payload.atom_count,
        edge_count: payload.edge_count,
        warmup: cfg.warmup,
        iterations: cfg.iterations,
        total_charge: cfg.total_charge,
        spin_multiplicity: cfg.spin_multiplicity,
        compute_dtype: compute_dtype_label(cfg.dtype),
        checksum: totals.checksum,
        mean_total_energy: totals.mean(totals.total_energy, cfg.iterations),
        mean_force_checksum: totals.mean(totals.force_checksum, cfg.iterations),
        mean_force_l2_norm: totals.mean(totals.force_l2_norm, cfg.iterations),
        max_force_abs: totals.force_max_abs,
        forces: payload.forces,
        measurement_scope: measurement_scope_report(),
        provenance: provenance_report(),
        backend: backend_report(),
        thread_env: thread_env_report(),
        samples_ms: payload.samples_ms,
        samples: payload.samples,
        sample_stats: payload.sample_stats,
        stages: payload.stages,
        blas_shapes: payload.blas_shapes,
        blas_families: payload.blas_families,
        kernels: payload.kernels,
    };
    let mut file = fs::File::create(path)?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    writeln!(file)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let cfg = Config::from_args()?;
    let model = LlmlipModel::load(&cfg.model_path)?;
    let LlmlipModel::OrbV3Omol(model) = model else {
        return Err("--model must be an ORB-v3 Conservative OMol .e3rs bundle".into());
    };
    let structure = read_fixture_structure(&cfg.fixture_path, cfg.case_index)?;
    let graph = model.prepare_non_pbc_graph(&structure, cfg.total_charge, cfg.spin_multiplicity)?;

    for _ in 0..cfg.warmup {
        let profile =
            model.profile_energy_forces_for_prepared_graph_with_compute_dtype(&graph, cfg.dtype)?;
        black_box(profile.total_energy + profile.force_checksum);
    }

    let mut totals = ProfileTotals::default();
    let mut samples_ms = Vec::with_capacity(cfg.iterations);
    let mut samples = Vec::with_capacity(cfg.iterations);
    let mut atom_count = 0usize;
    let mut edge_count = 0usize;
    let mut forces = None;
    for iteration in 0..cfg.iterations {
        let profile =
            model.profile_energy_forces_for_prepared_graph_with_compute_dtype(&graph, cfg.dtype)?;
        atom_count = profile.atom_count;
        edge_count = profile.edge_count;
        black_box(profile.total_energy + profile.force_checksum);
        samples_ms.push(profile.total_ms);
        samples.push(SampleReport {
            iteration,
            total_ms: profile.total_ms,
            total_energy: profile.total_energy,
            force_checksum: profile.force_checksum,
            force_l2_norm: profile.force_l2_norm,
            force_max_abs: profile.force_max_abs,
            stages: stage_report_from_profile(&profile),
        });
        if cfg.include_forces {
            forces = Some(profile.forces.clone());
        }
        totals.add(&profile);
    }
    let stages = stage_report(&totals, cfg.iterations);
    let total_ms = stages.total_ms;
    let blas_shapes = blas_shape_reports(&totals, cfg.iterations, total_ms);
    let blas_families = blas_family_reports(&totals, &blas_shapes, total_ms);
    let kernels = kernel_reports(&totals, cfg.iterations, total_ms);
    let sample_stats = SampleStatsReport {
        total_ms: distribution_report(&samples_ms),
    };

    println!(
        "label={} dtype={} atoms={atom_count} edges={edge_count} warmup={} iterations={} checksum={:.12e}",
        cfg.label,
        compute_dtype_label(cfg.dtype),
        cfg.warmup,
        cfg.iterations,
        totals.checksum
    );
    println!(
        "mean_total_energy={:.12e} mean_force_checksum={:.12e} mean_force_l2_norm={:.12e} max_force_abs={:.12e}",
        totals.mean(totals.total_energy, cfg.iterations),
        totals.mean(totals.force_checksum, cfg.iterations),
        totals.mean(totals.force_l2_norm, cfg.iterations),
        totals.force_max_abs
    );
    println!("{:<24} {:>12} {:>10}", "stage", "mean_ms", "share_pct");
    print_stage("total", total_ms, total_ms);
    print_stage(
        "input_features",
        totals.mean(totals.input_features_ms, cfg.iterations),
        total_ms,
    );
    print_stage(
        "encoder_forward",
        totals.mean(totals.encoder_forward_ms, cfg.iterations),
        total_ms,
    );
    print_stage(
        "gnn_weight_plan",
        totals.mean(totals.gnn_weight_plan_ms, cfg.iterations),
        total_ms,
    );
    print_stage(
        "gnn_forward",
        totals.mean(totals.gnn_forward_ms, cfg.iterations),
        total_ms,
    );
    print_stage(
        "energy_tail_vjp",
        totals.mean(totals.energy_tail_vjp_ms, cfg.iterations),
        total_ms,
    );
    print_stage(
        "final_cotangent",
        totals.mean(totals.final_cotangent_ms, cfg.iterations),
        total_ms,
    );
    print_stage(
        "gnn_input_vjp",
        totals.mean(totals.gnn_input_vjp_ms, cfg.iterations),
        total_ms,
    );
    print_stage(
        "encoder_vjp",
        totals.mean(totals.encoder_vjp_ms, cfg.iterations),
        total_ms,
    );
    print_stage(
        "geometry_forces",
        totals.mean(totals.geometry_forces_ms, cfg.iterations),
        total_ms,
    );
    print_stage(
        "zbl_force_accum",
        totals.mean(totals.zbl_force_accum_ms, cfg.iterations),
        total_ms,
    );
    print_stage(
        "force_vjp_total",
        totals.mean(totals.force_vjp_ms, cfg.iterations),
        total_ms,
    );
    #[cfg(feature = "orb-profile-breakdown")]
    {
        let blas_ms = stages.blas_ms;
        let activation_ms = stages.activation_ms;
        let scatter_reduction_ms = stages.scatter_reduction_ms;
        print_stage("blas", blas_ms, total_ms);
        print_stage("activation", activation_ms, total_ms);
        print_stage("scatter_reduction", scatter_reduction_ms, total_ms);
        print_stage("non_bucketed", stages.non_bucketed_ms, total_ms);
        if !blas_shapes.is_empty() {
            println!();
            println!("BLAS families by mean wall time:");
            println!(
                "{:<30} {:>7} {:>9} {:>9} {:>9} {:>9}",
                "family", "shapes", "calls", "mean_ms", "share_pct", "gflop/s"
            );
            for family in &blas_families {
                println!(
                    "{:<30} {:>7} {:>9.1} {:>9.3} {:>9.1} {:>9.1}",
                    family.family,
                    family.shapes,
                    family.calls,
                    family.mean_ms,
                    family.share_pct,
                    family.gflops_per_s
                );
            }
            println!();
            println!("BLAS shapes by mean wall time:");
            println!(
                "{:<34} {:>9} {:>9} {:>9} {:>9} {:>6} {:>6} {:>6} {:>8} {:>6}",
                "label",
                "calls",
                "mean_ms",
                "share_pct",
                "gflop/s",
                "rows",
                "in",
                "out",
                "full_in",
                "offset"
            );
            for shape in &blas_shapes {
                println!(
                    "{:<34} {:>9.1} {:>9.3} {:>9.1} {:>9.1} {:>6} {:>6} {:>6} {:>8} {:>6}",
                    shape.label,
                    shape.calls,
                    shape.mean_ms,
                    shape.share_pct,
                    shape.gflops_per_s,
                    shape.rows,
                    shape.input_dim,
                    shape.output_dim,
                    shape.full_input_dim,
                    shape.column_offset
                );
            }
        }
        if !kernels.is_empty() {
            println!();
            println!("Profiled kernels by mean wall time:");
            println!(
                "{:<28} {:>9} {:>9} {:>9} {:>8} {:>8}",
                "label", "calls", "mean_ms", "share_pct", "rows", "width"
            );
            for kernel in &kernels {
                println!(
                    "{:<28} {:>9.1} {:>9.3} {:>9.1} {:>8} {:>8}",
                    kernel.label,
                    kernel.calls,
                    kernel.mean_ms,
                    kernel.share_pct,
                    kernel.rows,
                    kernel.width
                );
            }
        }
    }

    write_report(
        &cfg,
        &totals,
        ReportPayload {
            atom_count,
            edge_count,
            samples_ms,
            samples,
            sample_stats,
            stages,
            blas_shapes,
            blas_families,
            kernels,
            forces,
        },
    )?;

    Ok(())
}
