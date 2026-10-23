#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use llmlip::{LlmlipModel, PredictionOptions, Structure};
use std::path::Path;

fn structure(atom_count: usize) -> Structure {
    let mut species = Vec::with_capacity(atom_count);
    let mut positions = Vec::with_capacity(atom_count);
    for index in 0..atom_count {
        species.push(if index % 3 == 0 { "O" } else { "H" }.to_owned());
        let x = (index % 8) as f64 * 0.7;
        let y = ((index / 8) % 8) as f64 * 0.8;
        let z = (index / 64) as f64 * 0.9;
        positions.push([x, y, z]);
    }
    Structure::new(species, positions)
}

fn bench_synthetic_model(c: &mut Criterion) {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/model.e3rs");
    let model = LlmlipModel::load(&model_path).unwrap_or_else(|err| {
        panic!(
            "failed to load synthetic benchmark model at {}: {err}",
            model_path.display()
        )
    });
    let mut group = c.benchmark_group("synthetic_model");
    for atom_count in [8_usize, 32, 128] {
        let structure = structure(atom_count);
        group.bench_with_input(
            BenchmarkId::new("energy_only", atom_count),
            &structure,
            |b, structure| {
                b.iter(|| {
                    let prediction = model
                        .predict(
                            structure,
                            &PredictionOptions {
                                compute_energy: true,
                                compute_forces: false,
                                ..PredictionOptions::default()
                            },
                        )
                        .unwrap();
                    black_box(prediction.energy.unwrap())
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("energy_force", atom_count),
            &structure,
            |b, structure| {
                b.iter(|| {
                    let prediction = model
                        .predict(
                            structure,
                            &PredictionOptions {
                                compute_energy: true,
                                compute_forces: true,
                                ..PredictionOptions::default()
                            },
                        )
                        .unwrap();
                    black_box((prediction.energy.unwrap(), prediction.forces.unwrap()))
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_synthetic_model);
criterion_main!(benches);
