use std::{env, hint::black_box, process, thread, time::Instant};

use llmlip::{ExternalGraphView, LlmlipModel, Prediction, PredictionOptions, Vec3};
use serde::Serialize;

const SCHEMA_VERSION: &str = "llmlip-external-graph-benchmark-v1.0";

#[derive(Debug)]
struct Args {
    model: String,
    warmup: usize,
    iterations: usize,
    label: String,
    graph_spec: GraphSpec,
}

#[derive(Clone, Copy, Debug)]
enum GraphSpec {
    TwoAtom,
    CubicGrid { side: usize, spacing: f64 },
    CompactLattice { atom_count: usize, spacing: f64 },
}

struct GraphData {
    name: String,
    atom_count: usize,
    owned_atom_count: usize,
    atomic_numbers: Vec<u8>,
    center_atom_indices: Vec<usize>,
    center_owned: Vec<bool>,
    edge_offsets: Vec<usize>,
    edge_dst: Vec<usize>,
    edge_vectors: Vec<Vec3>,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: &'static str,
    label: String,
    model: String,
    graph: String,
    atom_count: usize,
    owned_atom_count: usize,
    center_count: usize,
    edge_count: usize,
    model_cutoff: f64,
    requested_outputs: RequestedOutputs,
    warmup: usize,
    iterations: usize,
    load_runtime_ms: f64,
    mean_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    min_ms: f64,
    max_ms: f64,
    stdev_ms: f64,
    samples_ms: Vec<f64>,
    checksum: f64,
    measurement_scope: MeasurementScope,
    environment: Environment,
}

#[derive(Debug, Serialize)]
struct RequestedOutputs {
    energy: bool,
    forces: bool,
}

#[derive(Debug, Serialize)]
struct MeasurementScope {
    model_load_excluded: bool,
    lammps_dispatch_excluded: bool,
    lammps_neighbor_graph_build_excluded: bool,
    c_abi_conversion_excluded: bool,
    note: &'static str,
}

#[derive(Debug, Serialize)]
struct Environment {
    logical_cpus: Option<usize>,
    thread_env: Vec<EnvVar>,
}

#[derive(Debug, Serialize)]
struct EnvVar {
    key: &'static str,
    value: Option<String>,
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

    let load_start = Instant::now();
    let model =
        LlmlipModel::load(&args.model).map_err(|err| format!("failed to load model: {err}"))?;
    let load_runtime_ms = elapsed_ms(load_start);
    let capabilities = model.external_graph_capabilities();

    let graph_data = build_graph(args.graph_spec, capabilities.model_cutoff)?;
    let graph = ExternalGraphView {
        atom_count: graph_data.atom_count,
        owned_atom_count: graph_data.owned_atom_count,
        atomic_numbers: &graph_data.atomic_numbers,
        center_atom_indices: &graph_data.center_atom_indices,
        center_owned: &graph_data.center_owned,
        edge_offsets: &graph_data.edge_offsets,
        edge_dst: &graph_data.edge_dst,
        edge_vectors: &graph_data.edge_vectors,
        model_cutoff: capabilities.model_cutoff,
    };
    let options = PredictionOptions {
        compute_energy: true,
        compute_forces: true,
        ..PredictionOptions::default()
    };

    let mut checksum = 0.0;
    for _ in 0..args.warmup {
        let prediction = model
            .predict_external_graph(black_box(&graph), black_box(&options))
            .map_err(|err| format!("external graph warmup failed: {err}"))?;
        checksum += prediction_checksum(&prediction);
    }

    let mut samples_ms = Vec::with_capacity(args.iterations);
    for _ in 0..args.iterations {
        let start = Instant::now();
        let prediction = model
            .predict_external_graph(black_box(&graph), black_box(&options))
            .map_err(|err| format!("external graph prediction failed: {err}"))?;
        samples_ms.push(elapsed_ms(start));
        checksum += prediction_checksum(&prediction);
    }

    let mut sorted_samples = samples_ms.clone();
    sorted_samples.sort_by(f64::total_cmp);
    let mean_ms = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;
    let stdev_ms = sample_stdev(&samples_ms, mean_ms);
    let min_ms = sorted_samples[0];
    let max_ms = sorted_samples[sorted_samples.len() - 1];

    let report = Report {
        schema_version: SCHEMA_VERSION,
        label: args.label,
        model: args.model,
        graph: graph_data.name,
        atom_count: graph.atom_count,
        owned_atom_count: graph.owned_atom_count,
        center_count: graph.center_count(),
        edge_count: graph.edge_count(),
        model_cutoff: graph.model_cutoff,
        requested_outputs: RequestedOutputs {
            energy: options.compute_energy,
            forces: options.compute_forces,
        },
        warmup: args.warmup,
        iterations: args.iterations,
        load_runtime_ms,
        mean_ms,
        median_ms: percentile(&sorted_samples, 50.0),
        p95_ms: percentile(&sorted_samples, 95.0),
        min_ms,
        max_ms,
        stdev_ms,
        samples_ms,
        checksum: black_box(checksum),
        measurement_scope: MeasurementScope {
            model_load_excluded: true,
            lammps_dispatch_excluded: true,
            lammps_neighbor_graph_build_excluded: true,
            c_abi_conversion_excluded: true,
            note: "Timed samples call LlmlipModel::predict_external_graph on a borrowed prebuilt graph; this is the Rust external-graph baseline for LAMMPS bridge overhead comparisons.",
        },
        environment: benchmark_environment(),
    };
    serde_json::to_writer_pretty(std::io::stdout(), &report)
        .map_err(|err| format!("failed to write benchmark JSON report: {err}"))?;
    println!();
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut warmup = 100;
    let mut iterations = 10_000;
    let mut label = "rust_external_graph_synthetic_two_atom".to_owned();
    let mut graph_spec = GraphSpec::TwoAtom;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => {
                model = Some(
                    args.next()
                        .ok_or_else(|| "--model requires a path".to_owned())?,
                );
            }
            "--warmup" => {
                warmup = parse_usize_arg("--warmup", args.next())?;
            }
            "--iterations" => {
                iterations = parse_usize_arg("--iterations", args.next())?;
            }
            "--label" => {
                label = args
                    .next()
                    .ok_or_else(|| "--label requires a value".to_owned())?;
            }
            "--grid-side" => {
                graph_spec = GraphSpec::CubicGrid {
                    side: parse_usize_arg("--grid-side", args.next())?,
                    spacing: match graph_spec {
                        GraphSpec::TwoAtom => 2.4,
                        GraphSpec::CubicGrid { spacing, .. } => spacing,
                        GraphSpec::CompactLattice { spacing, .. } => spacing,
                    },
                };
            }
            "--atom-count" => {
                graph_spec = GraphSpec::CompactLattice {
                    atom_count: parse_usize_arg("--atom-count", args.next())?,
                    spacing: match graph_spec {
                        GraphSpec::TwoAtom => 2.4,
                        GraphSpec::CubicGrid { spacing, .. } => spacing,
                        GraphSpec::CompactLattice { spacing, .. } => spacing,
                    },
                };
            }
            "--spacing" => {
                let spacing = parse_f64_arg("--spacing", args.next())?;
                graph_spec = match graph_spec {
                    GraphSpec::TwoAtom => GraphSpec::CubicGrid { side: 4, spacing },
                    GraphSpec::CubicGrid { side, .. } => GraphSpec::CubicGrid { side, spacing },
                    GraphSpec::CompactLattice { atom_count, .. } => GraphSpec::CompactLattice {
                        atom_count,
                        spacing,
                    },
                };
            }
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        model: model.ok_or_else(|| "--model is required".to_owned())?,
        warmup,
        iterations,
        label,
        graph_spec,
    })
}

fn parse_usize_arg(name: &str, value: Option<String>) -> Result<usize, String> {
    value
        .ok_or_else(|| format!("{name} requires a value"))?
        .parse::<usize>()
        .map_err(|err| format!("invalid {name}: {err}"))
}

fn parse_f64_arg(name: &str, value: Option<String>) -> Result<f64, String> {
    let parsed = value
        .ok_or_else(|| format!("{name} requires a value"))?
        .parse::<f64>()
        .map_err(|err| format!("invalid {name}: {err}"))?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(format!("{name} must be finite and positive, got {parsed}"));
    }
    Ok(parsed)
}

fn print_usage() {
    println!(
        "Usage: cargo run --release -p llmlip --example benchmark_external_graph -- \\
         --model <MODEL.e3rs> [--warmup N] [--iterations N] [--label LABEL] \\
         [--atom-count N | --grid-side N] [--spacing ANGSTROM]"
    );
}

fn build_graph(spec: GraphSpec, cutoff: f64) -> Result<GraphData, String> {
    match spec {
        GraphSpec::TwoAtom => Ok(GraphData {
            name: "synthetic_h_o_two_atom_directed_full".to_owned(),
            atom_count: 2,
            owned_atom_count: 2,
            atomic_numbers: vec![1, 8],
            center_atom_indices: vec![0, 1],
            center_owned: vec![true, true],
            edge_offsets: vec![0, 1, 2],
            edge_dst: vec![1, 0],
            edge_vectors: vec![[0.0, 0.0, 1.2], [0.0, 0.0, -1.2]],
        }),
        GraphSpec::CubicGrid { side, spacing } => build_cubic_grid(side, spacing, cutoff),
        GraphSpec::CompactLattice {
            atom_count,
            spacing,
        } => build_compact_lattice(atom_count, spacing, cutoff),
    }
}

fn build_cubic_grid(side: usize, spacing: f64, cutoff: f64) -> Result<GraphData, String> {
    if side < 2 {
        return Err(format!("--grid-side must be at least 2, got {side}"));
    }
    if !spacing.is_finite() || spacing <= 0.0 {
        return Err(format!(
            "--spacing must be finite and positive, got {spacing}"
        ));
    }
    if !cutoff.is_finite() || cutoff <= 0.0 {
        return Err(format!(
            "model cutoff must be finite and positive, got {cutoff}"
        ));
    }

    let atom_count = side
        .checked_mul(side)
        .and_then(|value| value.checked_mul(side))
        .ok_or_else(|| "--grid-side cubed overflowed".to_owned())?;
    let mut positions = Vec::with_capacity(atom_count);
    let mut atomic_numbers = Vec::with_capacity(atom_count);
    for ix in 0..side {
        for iy in 0..side {
            for iz in 0..side {
                positions.push([
                    ix as f64 * spacing,
                    iy as f64 * spacing,
                    iz as f64 * spacing,
                ]);
                atomic_numbers.push(if (ix + iy + iz) % 2 == 0 { 1 } else { 8 });
            }
        }
    }

    let center_atom_indices = (0..atom_count).collect::<Vec<_>>();
    let center_owned = vec![true; atom_count];
    let mut edge_offsets = Vec::with_capacity(atom_count + 1);
    let mut edge_dst = Vec::new();
    let mut edge_vectors = Vec::new();
    let cutoff_sq = cutoff * cutoff;
    edge_offsets.push(0);
    for src in 0..atom_count {
        let src_pos = positions[src];
        for (dst, dst_pos) in positions.iter().copied().enumerate() {
            if src == dst {
                continue;
            }
            let vector = [
                dst_pos[0] - src_pos[0],
                dst_pos[1] - src_pos[1],
                dst_pos[2] - src_pos[2],
            ];
            let dist_sq = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
            if dist_sq < cutoff_sq {
                edge_dst.push(dst);
                edge_vectors.push(vector);
            }
        }
        edge_offsets.push(edge_dst.len());
    }

    Ok(GraphData {
        name: format!("synthetic_h_o_cubic_grid_side{side}_spacing{spacing}"),
        atom_count,
        owned_atom_count: atom_count,
        atomic_numbers,
        center_atom_indices,
        center_owned,
        edge_offsets,
        edge_dst,
        edge_vectors,
    })
}

fn build_compact_lattice(
    atom_count: usize,
    spacing: f64,
    cutoff: f64,
) -> Result<GraphData, String> {
    if atom_count < 2 {
        return Err(format!("--atom-count must be at least 2, got {atom_count}"));
    }
    if !spacing.is_finite() || spacing <= 0.0 {
        return Err(format!(
            "--spacing must be finite and positive, got {spacing}"
        ));
    }
    if !cutoff.is_finite() || cutoff <= 0.0 {
        return Err(format!(
            "model cutoff must be finite and positive, got {cutoff}"
        ));
    }

    let (nx, ny, nz) = compact_lattice_dims(atom_count);
    let mut positions = Vec::with_capacity(atom_count);
    let mut atomic_numbers = Vec::with_capacity(atom_count);
    'outer: for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                if positions.len() == atom_count {
                    break 'outer;
                }
                positions.push([
                    ix as f64 * spacing,
                    iy as f64 * spacing,
                    iz as f64 * spacing,
                ]);
                atomic_numbers.push(if (ix + iy + iz) % 2 == 0 { 1 } else { 8 });
            }
        }
    }

    graph_from_positions(
        format!("synthetic_h_o_compact_lattice_n{atom_count}_dims{nx}x{ny}x{nz}_spacing{spacing}"),
        positions,
        atomic_numbers,
        cutoff,
    )
}

fn compact_lattice_dims(atom_count: usize) -> (usize, usize, usize) {
    let nx = (atom_count as f64).cbrt().ceil() as usize;
    let yz_plane = atom_count.div_ceil(nx);
    let ny = (yz_plane as f64).sqrt().ceil() as usize;
    let nz = yz_plane.div_ceil(ny);
    (nx, ny, nz)
}

fn graph_from_positions(
    name: String,
    positions: Vec<Vec3>,
    atomic_numbers: Vec<u8>,
    cutoff: f64,
) -> Result<GraphData, String> {
    let atom_count = positions.len();
    if atomic_numbers.len() != atom_count {
        return Err("internal graph generator produced mismatched atomic numbers".to_owned());
    }
    let center_atom_indices = (0..atom_count).collect::<Vec<_>>();
    let center_owned = vec![true; atom_count];
    let mut edge_offsets = Vec::with_capacity(atom_count + 1);
    let mut edge_dst = Vec::new();
    let mut edge_vectors = Vec::new();
    let cutoff_sq = cutoff * cutoff;
    edge_offsets.push(0);
    for src in 0..atom_count {
        let src_pos = positions[src];
        for (dst, dst_pos) in positions.iter().copied().enumerate() {
            if src == dst {
                continue;
            }
            let vector = [
                dst_pos[0] - src_pos[0],
                dst_pos[1] - src_pos[1],
                dst_pos[2] - src_pos[2],
            ];
            let dist_sq = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
            if dist_sq < cutoff_sq {
                edge_dst.push(dst);
                edge_vectors.push(vector);
            }
        }
        edge_offsets.push(edge_dst.len());
    }

    Ok(GraphData {
        name,
        atom_count,
        owned_atom_count: atom_count,
        atomic_numbers,
        center_atom_indices,
        center_owned,
        edge_offsets,
        edge_dst,
        edge_vectors,
    })
}

fn prediction_checksum(prediction: &Prediction) -> f64 {
    let mut checksum = prediction.energy.unwrap_or(0.0);
    if let Some(forces) = &prediction.forces {
        for (atom_index, force) in forces.iter().enumerate() {
            let atom_weight = atom_index as f64 + 1.0;
            checksum += atom_weight * (0.5 * force[0] + 0.7 * force[1] + 0.9 * force[2]);
        }
    }
    black_box(checksum)
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

fn sample_stdev(samples: &[f64], mean: f64) -> f64 {
    if samples.len() <= 1 {
        return 0.0;
    }
    let variance = samples
        .iter()
        .map(|sample| {
            let delta = sample - mean;
            delta * delta
        })
        .sum::<f64>()
        / (samples.len() - 1) as f64;
    variance.sqrt()
}

fn percentile(sorted_samples: &[f64], percentile: f64) -> f64 {
    let rank = (percentile / 100.0) * (sorted_samples.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        return sorted_samples[lower];
    }
    let fraction = rank - lower as f64;
    sorted_samples[lower] * (1.0 - fraction) + sorted_samples[upper] * fraction
}

fn benchmark_environment() -> Environment {
    Environment {
        logical_cpus: thread::available_parallelism().map(usize::from).ok(),
        thread_env: [
            "RAYON_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "MKL_NUM_THREADS",
        ]
        .into_iter()
        .map(|key| EnvVar {
            key,
            value: env::var(key).ok(),
        })
        .collect(),
    }
}
