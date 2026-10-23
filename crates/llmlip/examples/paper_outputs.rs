#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use e3nn_core::{
    radial::{
        cosine_cutoff_forward, cosine_cutoff_vjp, cutoff_gaussian_radial_forward,
        cutoff_gaussian_radial_vjp, distance_forward, distance_vjp, gaussian_radial_basis_forward,
        gaussian_radial_basis_vjp,
    },
    Activation, CgEntry, DenseLayer, FixedMlp, Irreps, MlpCache, SphericalHarmonicsPlan,
    TensorProductInstruction, TensorProductPlan, Vec3, VERSION as E3NN_CORE_VERSION,
};
use llmlip::{all_pairs_neighbor_list, LlmlipModel, PredictionOptions, Structure};
use serde::Serialize;
use std::{
    env,
    error::Error,
    fs::{self, File},
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const EPS: f64 = 1.0e-12;
const FD_STEP: f64 = 1.0e-6;
const INV_SQRT_3: f64 = 0.577_350_269_189_625_8;
const DIRECTED_EDGE_ENERGY_SCALE: f64 = 0.5;
const ROTATION_SEED: u64 = 0xE3_00_00_14;

fn main() -> Result<(), Box<dyn Error>> {
    let config = Config::from_args()?;
    fs::create_dir_all(&config.out_dir)?;

    let metadata = collect_metadata(&config);
    write_json(&config.out_dir.join("metadata.json"), &metadata)?;
    write_json(
        &config.out_dir.join("primitive_vjp_validation.json"),
        &primitive_vjp_validation()?,
    )?;
    write_json(
        &config.out_dir.join("rotation_covariance.json"),
        &rotation_covariance(&config.model_path, &metadata)?,
    )?;
    write_json(
        &config.out_dir.join("performance_scaling.json"),
        &performance_scaling(&config.model_path, &metadata)?,
    )?;
    write_json(
        &config.out_dir.join("rust_force_cases.json"),
        &rust_force_cases(&config.model_path, &metadata)?,
    )?;
    write_json(
        &config.out_dir.join("index.json"),
        &OutputIndex {
            schema_version: "llmlip-paper-output-index-v1",
            generated_by: "cargo run -p llmlip --example paper_outputs",
            files: vec![
                "metadata.json",
                "primitive_vjp_validation.json",
                "rotation_covariance.json",
                "performance_scaling.json",
                "rust_force_cases.json",
            ],
        },
    )?;

    println!("wrote raw paper outputs to {}", config.out_dir.display());
    Ok(())
}

#[derive(Clone, Debug)]
struct Config {
    out_dir: PathBuf,
    model_path: PathBuf,
}

impl Config {
    fn from_args() -> Result<Self, Box<dyn Error>> {
        let mut out_dir = PathBuf::from("outputs/paper/current");
        let mut model_path = PathBuf::from("examples/model.e3rs");
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out" => {
                    let value = args.next().ok_or("--out requires a path")?;
                    out_dir = PathBuf::from(value);
                }
                "--model" => {
                    let value = args.next().ok_or("--model requires a path")?;
                    model_path = PathBuf::from(value);
                }
                "--help" | "-h" => {
                    println!(
                        "usage: cargo run -p llmlip --example paper_outputs -- \
                         [--out outputs/paper/current] [--model examples/model.e3rs]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument: {other}").into()),
            }
        }

        Ok(Self {
            out_dir,
            model_path,
        })
    }
}

#[derive(Serialize)]
struct OutputIndex {
    schema_version: &'static str,
    generated_by: &'static str,
    files: Vec<&'static str>,
}

#[derive(Clone, Serialize)]
struct OutputMetadata {
    schema_version: &'static str,
    generated_by: &'static str,
    generated_at_unix_seconds: u64,
    command: Vec<String>,
    cwd: String,
    model_path: String,
    output_dir: String,
    os: &'static str,
    arch: &'static str,
    rustc_version: Option<String>,
    git_commit: Option<String>,
    cpu_model: Option<String>,
    logical_cpus: Option<usize>,
    physical_cpus: Option<usize>,
    total_memory_bytes: Option<u64>,
    rustflags: Option<String>,
    crate_versions: CrateVersions,
    notes: Vec<&'static str>,
}

#[derive(Clone, Serialize)]
struct CrateVersions {
    e3nn_core: &'static str,
    llmlip: &'static str,
}

fn collect_metadata(config: &Config) -> OutputMetadata {
    OutputMetadata {
        schema_version: "llmlip-paper-metadata-v1",
        generated_by: "crates/llmlip/examples/paper_outputs.rs",
        generated_at_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0),
        command: env::args().collect(),
        cwd: env::current_dir()
            .ok()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<unknown>".to_owned()),
        model_path: config.model_path.display().to_string(),
        output_dir: config.out_dir.display().to_string(),
        os: env::consts::OS,
        arch: env::consts::ARCH,
        rustc_version: command_output("rustc", &["--version"]),
        git_commit: command_output("git", &["rev-parse", "HEAD"]),
        cpu_model: cpu_model(),
        logical_cpus: std::thread::available_parallelism()
            .ok()
            .map(|count| count.get()),
        physical_cpus: physical_cpus(),
        total_memory_bytes: total_memory_bytes(),
        rustflags: env::var("RUSTFLAGS").ok(),
        crate_versions: CrateVersions {
            e3nn_core: E3NN_CORE_VERSION,
            llmlip: env!("CARGO_PKG_VERSION"),
        },
        notes: vec![
            "Raw validation data only; plotting is intentionally out of scope for T14.",
            "Performance records are reproducibility data, not performance superiority claims.",
        ],
    }
}

#[derive(Serialize)]
struct PrimitiveVjpDataset {
    schema_version: &'static str,
    dtype: &'static str,
    method: &'static str,
    finite_difference_step: f64,
    records: Vec<VjpRecord>,
}

#[derive(Serialize)]
struct VjpRecord {
    primitive: &'static str,
    samples: usize,
    max_abs_error: f64,
    relative_l2_error: f64,
    tolerance: f64,
    notes: &'static str,
}

fn primitive_vjp_validation() -> Result<PrimitiveVjpDataset, Box<dyn Error>> {
    let records = vec![
        distance_vjp_record(),
        cutoff_vjp_record(),
        radial_basis_vjp_record()?,
        cutoff_radial_vjp_record()?,
        spherical_harmonics_vjp_record()?,
        tensor_product_vjp_record()?,
        mlp_input_vjp_record()?,
    ];

    Ok(PrimitiveVjpDataset {
        schema_version: "llmlip-primitive-vjp-validation-v1",
        dtype: "f64",
        method: "analytic VJP compared to central finite differences",
        finite_difference_step: FD_STEP,
        records,
    })
}

fn distance_vjp_record() -> VjpRecord {
    let r = [1.2, -0.7, 2.5];
    let bar_distance = -1.7;
    let actual = distance_vjp(r, bar_distance, EPS).unwrap().to_vec();
    let expected = central_difference_vec3(r, |trial| {
        bar_distance * distance_forward(trial, EPS).unwrap().distance
    })
    .to_vec();
    make_record("distance_vjp", actual, expected, 1.0e-8, "d||r||/dr")
}

fn cutoff_vjp_record() -> VjpRecord {
    let distance = 1.3;
    let cutoff = 3.5;
    let bar_cutoff = 0.7;
    let actual = vec![cosine_cutoff_vjp(distance, cutoff, bar_cutoff).unwrap()];
    let expected = vec![central_difference_scalar(distance, |trial| {
        bar_cutoff * cosine_cutoff_forward(trial, cutoff).unwrap().value
    })];
    make_record(
        "cosine_cutoff_vjp",
        actual,
        expected,
        1.0e-9,
        "d cutoff(d) / dd",
    )
}

fn radial_basis_vjp_record() -> Result<VjpRecord, Box<dyn Error>> {
    let centers = [0.5, 1.0, 1.5, 2.0];
    let gamma = 0.8;
    let distance = 1.2;
    let bar_basis = [0.3, -0.4, 0.8, 0.1];
    let mut values = [0.0; 4];
    gaussian_radial_basis_forward(distance, &centers, gamma, &mut values)?;
    let actual = vec![gaussian_radial_basis_vjp(
        distance, &centers, gamma, &values, &bar_basis,
    )?];
    let expected = vec![central_difference_scalar(distance, |trial| {
        let mut out = [0.0; 4];
        gaussian_radial_basis_forward(trial, &centers, gamma, &mut out).unwrap();
        dot(&out, &bar_basis)
    })];
    Ok(make_record(
        "gaussian_radial_basis_vjp",
        actual,
        expected,
        1.0e-8,
        "d Gaussian radial basis / dd",
    ))
}

fn cutoff_radial_vjp_record() -> Result<VjpRecord, Box<dyn Error>> {
    let centers = [0.5, 1.0, 1.5, 2.0];
    let gamma = 0.8;
    let cutoff = 3.5;
    let distance = 1.2;
    let bar_weighted = [0.3, -0.4, 0.8, 0.1];
    let mut basis = [0.0; 4];
    gaussian_radial_basis_forward(distance, &centers, gamma, &mut basis)?;
    let actual = vec![cutoff_gaussian_radial_vjp(
        distance,
        cutoff,
        &centers,
        gamma,
        &basis,
        &bar_weighted,
    )?];
    let expected = vec![central_difference_scalar(distance, |trial| {
        let mut out = [0.0; 4];
        cutoff_gaussian_radial_forward(trial, cutoff, &centers, gamma, &mut out).unwrap();
        dot(&out, &bar_weighted)
    })];
    Ok(make_record(
        "cutoff_gaussian_radial_vjp",
        actual,
        expected,
        1.0e-7,
        "d cutoff-weighted radial basis / dd",
    ))
}

fn spherical_harmonics_vjp_record() -> Result<VjpRecord, Box<dyn Error>> {
    let plan = SphericalHarmonicsPlan::component(2, true)?;
    let r = [0.7, -1.1, 2.3];
    let grad_y = (0..plan.output_dim())
        .map(|index| 0.1 + index as f64 * 0.03)
        .collect::<Vec<_>>();
    let mut grad_r = [0.0; 3];
    plan.vjp_edge_vector(r, &grad_y, &mut grad_r)?;
    let expected = central_difference_vec3(r, |trial| {
        let mut out = vec![0.0; plan.output_dim()];
        plan.forward(trial, &mut out).unwrap();
        dot(&out, &grad_y)
    });
    Ok(make_record(
        "spherical_harmonics_lmax2_component_normalized_vjp",
        grad_r.to_vec(),
        expected.to_vec(),
        1.0e-7,
        "d SH(r) / dr for l_max=2, component normalization, normalize=true",
    ))
}

fn tensor_product_vjp_record() -> Result<VjpRecord, Box<dyn Error>> {
    let plan = e3nn_tensor_product_smoke_plan()?;
    let x = sequence(plan.input1_dim(), 0.17);
    let y = sequence(plan.input2_dim(), -0.11);
    let w = sequence(plan.weight_numel, 0.23);
    let grad_out = sequence(plan.output_dim(), 0.31);
    let mut grad_x = vec![0.0; plan.input1_dim()];
    let mut grad_y = vec![0.0; plan.input2_dim()];
    let mut grad_w = vec![0.0; plan.weight_numel];
    plan.vjp(
        &x,
        &y,
        Some(&w),
        &grad_out,
        &mut grad_x,
        &mut grad_y,
        Some(&mut grad_w),
    )?;

    let expected_x = central_difference_slice(&x, |trial| {
        tensor_product_projection(&plan, trial, &y, &w, &grad_out)
    });
    let expected_y = central_difference_slice(&y, |trial| {
        tensor_product_projection(&plan, &x, trial, &w, &grad_out)
    });
    let expected_w = central_difference_slice(&w, |trial| {
        tensor_product_projection(&plan, &x, &y, trial, &grad_out)
    });

    let mut actual = grad_x;
    actual.extend(grad_y);
    actual.extend(grad_w);
    let mut expected = expected_x;
    expected.extend(expected_y);
    expected.extend(expected_w);
    Ok(make_record(
        "tensor_product_smoke_vjp_x_y_w",
        actual,
        expected,
        1.0e-8,
        "VJP wrt first input, second input, and dynamic edge weights",
    ))
}

fn mlp_input_vjp_record() -> Result<VjpRecord, Box<dyn Error>> {
    let mlp = validation_mlp()?;
    let x = vec![0.2, -0.3, 0.4, 0.7];
    let grad_out = [0.6];
    let mut out = [0.0];
    let mut cache = MlpCache::default();
    mlp.forward(&x, &mut out, &mut cache)?;
    let mut grad_x = vec![0.0; mlp.input_dim()];
    mlp.vjp_input_only(&cache, &grad_out, &mut grad_x)?;
    let expected = central_difference_slice(&x, |trial| {
        let mut trial_out = [0.0];
        let mut trial_cache = MlpCache::default();
        mlp.forward(trial, &mut trial_out, &mut trial_cache)
            .unwrap();
        trial_out[0] * grad_out[0]
    });
    Ok(make_record(
        "fixed_mlp_input_only_vjp",
        grad_x,
        expected,
        1.0e-8,
        "fixed-weight MLP input-only VJP",
    ))
}

#[derive(Serialize)]
struct RotationDataset {
    schema_version: &'static str,
    metadata: OutputMetadata,
    model_path: String,
    seed: u64,
    cases: Vec<RotationRecord>,
}

#[derive(Serialize)]
struct RotationRecord {
    structure_id: String,
    rotation_id: String,
    axis: Vec3,
    angle_radians: f64,
    energy_error: f64,
    force_relative_l2_error: f64,
    max_force_abs_error: f64,
}

fn rotation_covariance(
    model_path: &Path,
    metadata: &OutputMetadata,
) -> Result<RotationDataset, Box<dyn Error>> {
    let model = LlmlipModel::load(model_path)?;
    let rotations = deterministic_rotations(ROTATION_SEED, 20);
    let mut records = Vec::new();

    for (structure_id, structure) in validation_structures() {
        let (energy, forces) = predict_energy_forces(&model, &structure)?;
        for rotation in &rotations {
            let matrix = rotation_matrix(rotation.axis, rotation.angle_radians);
            let rotated_structure = rotate_structure(&structure, matrix);
            let (rotated_energy, rotated_forces) =
                predict_energy_forces(&model, &rotated_structure)?;
            let expected_forces = forces
                .iter()
                .map(|force| mat_vec(matrix, *force))
                .collect::<Vec<_>>();
            let (force_relative_l2_error, max_force_abs_error) =
                force_errors(&rotated_forces, &expected_forces);
            records.push(RotationRecord {
                structure_id: structure_id.to_owned(),
                rotation_id: rotation.id.clone(),
                axis: rotation.axis,
                angle_radians: rotation.angle_radians,
                energy_error: (rotated_energy - energy).abs(),
                force_relative_l2_error,
                max_force_abs_error,
            });
        }
    }

    Ok(RotationDataset {
        schema_version: "llmlip-rotation-covariance-v1",
        metadata: metadata.clone(),
        model_path: model_path.display().to_string(),
        seed: ROTATION_SEED,
        cases: records,
    })
}

#[derive(Clone, Debug)]
struct RotationSpec {
    id: String,
    axis: Vec3,
    angle_radians: f64,
}

#[derive(Serialize)]
struct PerformanceDataset {
    schema_version: &'static str,
    metadata: OutputMetadata,
    model_path: String,
    limitations: Vec<&'static str>,
    records: Vec<PerformanceRecord>,
}

#[derive(Serialize)]
struct PerformanceRecord {
    atom_count: usize,
    edge_count: usize,
    l_max: Option<usize>,
    mode: &'static str,
    iterations: usize,
    total_nanoseconds: u128,
    nanoseconds_per_iteration: f64,
    atoms_per_second: f64,
    edges_per_second: f64,
}

fn performance_scaling(
    model_path: &Path,
    metadata: &OutputMetadata,
) -> Result<PerformanceDataset, Box<dyn Error>> {
    let model = LlmlipModel::load(model_path)?;
    let mut records = Vec::new();

    for atom_count in [2_usize, 4, 8, 16, 32, 64] {
        let structure = grid_structure(atom_count);
        let edge_count = all_pairs_neighbor_list(&structure, model_cutoff(&model))?.len();
        let iterations = if atom_count <= 8 {
            200
        } else if atom_count <= 32 {
            50
        } else {
            20
        };

        records.push(time_prediction(
            &model,
            &structure,
            edge_count,
            "energy_only",
            PredictionOptions {
                compute_energy: true,
                compute_forces: false,
                ..PredictionOptions::default()
            },
            iterations,
        )?);
        records.push(time_prediction(
            &model,
            &structure,
            edge_count,
            "energy_force",
            PredictionOptions {
                compute_energy: true,
                compute_forces: true,
                ..PredictionOptions::default()
            },
            iterations,
        )?);
    }

    Ok(PerformanceDataset {
        schema_version: "llmlip-performance-scaling-v1",
        metadata: metadata.clone(),
        model_path: model_path.display().to_string(),
        limitations: vec![
            "synthetic-local-v1 is radial-only, so l_max dependence is not applicable until an SH/TP model is added.",
        ],
        records,
    })
}

fn time_prediction(
    model: &LlmlipModel,
    structure: &Structure,
    edge_count: usize,
    mode: &'static str,
    options: PredictionOptions,
    iterations: usize,
) -> Result<PerformanceRecord, Box<dyn Error>> {
    let _warmup = model.predict(structure, &options)?;
    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..iterations {
        let prediction = model.predict(structure, &options)?;
        if let Some(energy) = prediction.energy {
            checksum += energy;
        }
        if let Some(forces) = prediction.forces {
            checksum += forces.iter().flatten().sum::<f64>();
        }
    }
    std::hint::black_box(checksum);
    let elapsed = start.elapsed();
    let total_nanoseconds = elapsed.as_nanos();
    let seconds = elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
    Ok(PerformanceRecord {
        atom_count: structure.len(),
        edge_count,
        l_max: None,
        mode,
        iterations,
        total_nanoseconds,
        nanoseconds_per_iteration: total_nanoseconds as f64 / iterations as f64,
        atoms_per_second: structure.len() as f64 * iterations as f64 / seconds,
        edges_per_second: edge_count as f64 * iterations as f64 / seconds,
    })
}

#[derive(Serialize)]
struct RustForceDataset {
    schema_version: &'static str,
    metadata: OutputMetadata,
    reference_kind: &'static str,
    model: ModelSnapshot,
    cases: Vec<ForceCase>,
}

#[derive(Serialize)]
struct ModelSnapshot {
    architecture: &'static str,
    source_path: String,
    edge_convention: &'static str,
    force_convention: &'static str,
    directed_edge_energy_scale: f64,
    cutoff: f64,
    centers: Vec<f64>,
    gamma: f64,
    distance_eps: f64,
    edge_mlp_layers: Vec<LayerSnapshot>,
}

#[derive(Serialize)]
struct LayerSnapshot {
    in_dim: usize,
    out_dim: usize,
    weight_row_major: Vec<f64>,
    bias: Vec<f64>,
    activation: &'static str,
}

#[derive(Serialize)]
struct ForceCase {
    id: String,
    species: Vec<String>,
    positions: Vec<Vec3>,
    pbc: [bool; 3],
    edge_count: usize,
    rust_energy: f64,
    rust_forces: Vec<Vec3>,
}

fn rust_force_cases(
    model_path: &Path,
    metadata: &OutputMetadata,
) -> Result<RustForceDataset, Box<dyn Error>> {
    let model = LlmlipModel::load(model_path)?;
    let snapshot = model_snapshot(&model, model_path);
    let mut cases = Vec::new();

    for (id, structure) in validation_structures() {
        let (rust_energy, rust_forces) = predict_energy_forces(&model, &structure)?;
        let edge_count = all_pairs_neighbor_list(&structure, snapshot.cutoff)?.len();
        cases.push(ForceCase {
            id: id.to_owned(),
            species: structure.species,
            positions: structure.positions,
            pbc: structure.pbc,
            edge_count,
            rust_energy,
            rust_forces,
        });
    }

    Ok(RustForceDataset {
        schema_version: "llmlip-rust-force-cases-v1",
        metadata: metadata.clone(),
        reference_kind: "explicit PyTorch autograd mirror of synthetic-local-v1, generated by tools/reference/generate_paper_outputs.py",
        model: snapshot,
        cases,
    })
}

fn model_snapshot(model: &LlmlipModel, model_path: &Path) -> ModelSnapshot {
    let LlmlipModel::SyntheticRadial(synthetic) = model else {
        panic!("paper output snapshot currently supports only synthetic-local-v1")
    };
    let edge_mlp_layers = synthetic
        .edge_mlp()
        .layers()
        .iter()
        .map(|layer| LayerSnapshot {
            in_dim: layer.in_dim(),
            out_dim: layer.out_dim(),
            weight_row_major: layer.weight().to_vec(),
            bias: layer.bias().to_vec(),
            activation: match layer.activation() {
                Activation::Identity => "identity",
                Activation::Tanh => "tanh",
            },
        })
        .collect();

    ModelSnapshot {
        architecture: "synthetic-local-v1",
        source_path: model_path.display().to_string(),
        edge_convention: "r_ij = R_j + shift_ij - R_i",
        force_convention: "F_i += bar_r_ij; F_j += -bar_r_ij",
        directed_edge_energy_scale: DIRECTED_EDGE_ENERGY_SCALE,
        cutoff: synthetic.cutoff(),
        centers: synthetic.centers().to_vec(),
        gamma: synthetic.gamma(),
        distance_eps: synthetic.distance_eps(),
        edge_mlp_layers,
    }
}

fn validation_structures() -> Vec<(&'static str, Structure)> {
    vec![
        (
            "dimer_in_cutoff",
            Structure::new(
                vec!["H".to_owned(), "O".to_owned()],
                vec![[0.0, 0.0, 0.0], [1.15, -0.2, 0.35]],
            ),
        ),
        (
            "trimer_mixed",
            Structure::new(
                vec!["H".to_owned(), "H".to_owned(), "O".to_owned()],
                vec![[0.0, 0.0, 0.0], [1.1, -0.2, 0.4], [-0.6, 1.3, 0.7]],
            ),
        ),
        (
            "four_atom_skew",
            Structure::new(
                vec![
                    "O".to_owned(),
                    "H".to_owned(),
                    "H".to_owned(),
                    "O".to_owned(),
                ],
                vec![
                    [0.0, 0.0, 0.0],
                    [0.9, 0.4, -0.2],
                    [-0.7, 1.1, 0.6],
                    [1.8, -0.5, 1.2],
                ],
            ),
        ),
        ("six_atom_grid", grid_structure(6)),
    ]
}

fn grid_structure(atom_count: usize) -> Structure {
    let mut species = Vec::with_capacity(atom_count);
    let mut positions = Vec::with_capacity(atom_count);
    for index in 0..atom_count {
        species.push(if index % 3 == 0 { "O" } else { "H" }.to_owned());
        let x = (index % 4) as f64 * 0.85;
        let y = ((index / 4) % 4) as f64 * 0.9 + 0.03 * (index % 2) as f64;
        let z = (index / 16) as f64 * 0.95 + 0.02 * (index % 3) as f64;
        positions.push([x, y, z]);
    }
    Structure::new(species, positions)
}

fn predict_energy_forces(
    model: &LlmlipModel,
    structure: &Structure,
) -> Result<(f64, Vec<Vec3>), Box<dyn Error>> {
    let prediction = model.predict(
        structure,
        &PredictionOptions {
            compute_energy: true,
            compute_forces: true,
            ..PredictionOptions::default()
        },
    )?;
    Ok((
        prediction.energy.ok_or("energy was not returned")?,
        prediction.forces.ok_or("forces were not returned")?,
    ))
}

fn model_cutoff(model: &LlmlipModel) -> f64 {
    let LlmlipModel::SyntheticRadial(model) = model else {
        panic!("paper output scaling currently supports only synthetic-local-v1")
    };
    model.cutoff()
}

fn validation_mlp() -> Result<FixedMlp, Box<dyn Error>> {
    Ok(FixedMlp::new(vec![
        DenseLayer::new(
            4,
            3,
            vec![
                0.8, -0.4, 0.2, 0.5, -0.3, 0.7, -0.6, 0.1, 0.25, 0.35, -0.45, 0.9,
            ],
            vec![0.05, -0.1, 0.2],
            Activation::Tanh,
        )?,
        DenseLayer::new(
            3,
            1,
            vec![1.1, -0.75, 0.4],
            vec![0.03],
            Activation::Identity,
        )?,
    ])?)
}

fn e3nn_tensor_product_smoke_plan() -> Result<TensorProductPlan, Box<dyn Error>> {
    let irreps_in1 = "2x0e + 1x1o".parse::<Irreps>()?;
    let irreps_in2 = "1x0e + 1x1o".parse::<Irreps>()?;
    let irreps_out = "2x0e + 1x1o".parse::<Irreps>()?;
    let in1_segment = [0, 0, 1, 1, 0, 1, 2, 2, 2];
    let in2_segment = [0, 0, 0, 0, 1, 1, 0, 1, 1];
    let out_segment = [0, 1, 0, 1, 2, 2, 2, 0, 1];
    let cg_offset = [0, 0, 0, 0, 1, 1, 4, 7, 7];
    let cg_len = [1, 1, 1, 1, 3, 3, 3, 3, 3];
    let path_weight = [
        INV_SQRT_3, INV_SQRT_3, INV_SQRT_3, INV_SQRT_3, 1.0, 1.0, 1.0, INV_SQRT_3, INV_SQRT_3,
    ];
    let instructions = (0..9)
        .map(|index| TensorProductInstruction {
            in1_segment: in1_segment[index],
            in2_segment: in2_segment[index],
            out_segment: out_segment[index],
            cg_offset: cg_offset[index],
            cg_len: cg_len[index],
            weight_offset: Some(index),
            path_weight: path_weight[index],
        })
        .collect();
    let cg_entries = vec![
        CgEntry {
            i_m: 0,
            j_m: 0,
            k_m: 0,
            value: 1.0,
        },
        CgEntry {
            i_m: 0,
            j_m: 0,
            k_m: 0,
            value: INV_SQRT_3,
        },
        CgEntry {
            i_m: 0,
            j_m: 1,
            k_m: 1,
            value: INV_SQRT_3,
        },
        CgEntry {
            i_m: 0,
            j_m: 2,
            k_m: 2,
            value: INV_SQRT_3,
        },
        CgEntry {
            i_m: 0,
            j_m: 0,
            k_m: 0,
            value: INV_SQRT_3,
        },
        CgEntry {
            i_m: 1,
            j_m: 0,
            k_m: 1,
            value: INV_SQRT_3,
        },
        CgEntry {
            i_m: 2,
            j_m: 0,
            k_m: 2,
            value: INV_SQRT_3,
        },
        CgEntry {
            i_m: 0,
            j_m: 0,
            k_m: 0,
            value: INV_SQRT_3,
        },
        CgEntry {
            i_m: 1,
            j_m: 1,
            k_m: 0,
            value: INV_SQRT_3,
        },
        CgEntry {
            i_m: 2,
            j_m: 2,
            k_m: 0,
            value: INV_SQRT_3,
        },
    ];
    Ok(TensorProductPlan::new(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        cg_entries,
        9,
    )?)
}

fn tensor_product_projection(
    plan: &TensorProductPlan,
    x: &[f64],
    y: &[f64],
    w: &[f64],
    grad_out: &[f64],
) -> f64 {
    let mut out = vec![0.0; plan.output_dim()];
    plan.forward(x, y, Some(w), &mut out).unwrap();
    dot(&out, grad_out)
}

fn make_record(
    primitive: &'static str,
    actual: Vec<f64>,
    expected: Vec<f64>,
    tolerance: f64,
    notes: &'static str,
) -> VjpRecord {
    let (max_abs_error, relative_l2_error) = error_metrics(&actual, &expected);
    VjpRecord {
        primitive,
        samples: actual.len(),
        max_abs_error,
        relative_l2_error,
        tolerance,
        notes,
    }
}

fn error_metrics(actual: &[f64], expected: &[f64]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0_f64;
    let mut diff_sq = 0.0_f64;
    let mut expected_sq = 0.0_f64;
    for (a, e) in actual.iter().zip(expected) {
        let diff = a - e;
        max_abs = max_abs.max(diff.abs());
        diff_sq += diff * diff;
        expected_sq += e * e;
    }
    (max_abs, diff_sq.sqrt() / expected_sq.sqrt().max(EPS))
}

fn force_errors(actual: &[Vec3], expected: &[Vec3]) -> (f64, f64) {
    let actual_flat = actual.iter().flatten().copied().collect::<Vec<_>>();
    let expected_flat = expected.iter().flatten().copied().collect::<Vec<_>>();
    let (max_abs, relative_l2) = error_metrics(&actual_flat, &expected_flat);
    (relative_l2, max_abs)
}

fn central_difference_scalar<F>(x: f64, f: F) -> f64
where
    F: Fn(f64) -> f64,
{
    (f(x + FD_STEP) - f(x - FD_STEP)) / (2.0 * FD_STEP)
}

fn central_difference_vec3<F>(r: Vec3, f: F) -> Vec3
where
    F: Fn(Vec3) -> f64,
{
    let mut out = [0.0; 3];
    for axis in 0..3 {
        let mut plus = r;
        plus[axis] += FD_STEP;
        let mut minus = r;
        minus[axis] -= FD_STEP;
        out[axis] = (f(plus) - f(minus)) / (2.0 * FD_STEP);
    }
    out
}

fn central_difference_slice<F>(values: &[f64], f: F) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let mut out = vec![0.0; values.len()];
    for index in 0..values.len() {
        let mut plus = values.to_vec();
        plus[index] += FD_STEP;
        let mut minus = values.to_vec();
        minus[index] -= FD_STEP;
        out[index] = (f(&plus) - f(&minus)) / (2.0 * FD_STEP);
    }
    out
}

fn sequence(len: usize, phase: f64) -> Vec<f64> {
    (0..len)
        .map(|index| ((index + 1) as f64 * 0.19 + phase).sin())
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(left, right)| left * right).sum()
}

fn rotation_matrix(axis: Vec3, angle: f64) -> [[f64; 3]; 3] {
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let [x, y, z] = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
    let c = angle.cos();
    let s = angle.sin();
    let one_c = 1.0 - c;
    [
        [
            c + x * x * one_c,
            x * y * one_c - z * s,
            x * z * one_c + y * s,
        ],
        [
            y * x * one_c + z * s,
            c + y * y * one_c,
            y * z * one_c - x * s,
        ],
        [
            z * x * one_c - y * s,
            z * y * one_c + x * s,
            c + z * z * one_c,
        ],
    ]
}

fn deterministic_rotations(seed: u64, count: usize) -> Vec<RotationSpec> {
    let mut state = seed;
    (0..count)
        .map(|index| {
            let mut axis = [
                2.0 * lcg_unit(&mut state) - 1.0,
                2.0 * lcg_unit(&mut state) - 1.0,
                2.0 * lcg_unit(&mut state) - 1.0,
            ];
            let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
            if norm < EPS {
                axis = [0.0, 0.0, 1.0];
            }
            let angle = 0.05 + lcg_unit(&mut state) * (std::f64::consts::PI - 0.1);
            RotationSpec {
                id: format!("seeded_rotation_{index:02}"),
                axis,
                angle_radians: angle,
            }
        })
        .collect()
}

fn lcg_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1_u64 << 53) as f64)
}

fn rotate_structure(structure: &Structure, matrix: [[f64; 3]; 3]) -> Structure {
    Structure::new(
        structure.species.clone(),
        structure
            .positions
            .iter()
            .map(|position| mat_vec(matrix, *position))
            .collect(),
    )
}

fn mat_vec(matrix: [[f64; 3]; 3], value: Vec3) -> Vec3 {
    [
        matrix[0][0] * value[0] + matrix[0][1] * value[1] + matrix[0][2] * value[2],
        matrix[1][0] * value[0] + matrix[1][1] * value[1] + matrix[1][2] * value[2],
        matrix[2][0] * value[0] + matrix[2][1] * value[1] + matrix[2][2] * value[2],
    ]
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    Some(text.trim().to_owned())
}

fn cpu_model() -> Option<String> {
    if env::consts::OS == "macos" {
        return command_output("sysctl", &["-n", "machdep.cpu.brand_string"]);
    }
    if env::consts::OS == "linux" {
        let text = fs::read_to_string("/proc/cpuinfo").ok()?;
        return text.lines().find_map(|line| {
            line.strip_prefix("model name")
                .and_then(|value| value.split_once(':'))
                .map(|(_, value)| value.trim().to_owned())
        });
    }
    None
}

fn physical_cpus() -> Option<usize> {
    if env::consts::OS == "macos" {
        return command_output("sysctl", &["-n", "hw.physicalcpu"])
            .and_then(|value| value.parse().ok());
    }
    None
}

fn total_memory_bytes() -> Option<u64> {
    if env::consts::OS == "macos" {
        return command_output("sysctl", &["-n", "hw.memsize"])
            .and_then(|value| value.parse().ok());
    }
    if env::consts::OS == "linux" {
        let text = fs::read_to_string("/proc/meminfo").ok()?;
        return text.lines().find_map(|line| {
            line.strip_prefix("MemTotal:")
                .and_then(|value| value.split_whitespace().next())
                .and_then(|kib| kib.parse::<u64>().ok())
                .map(|kib| kib * 1024)
        });
    }
    None
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, value)?;
    Ok(())
}
