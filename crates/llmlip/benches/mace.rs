#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use llmlip::mace::MaceScalarSymmetricContraction;

fn sequence(len: usize, start: f64, step: f64) -> Vec<f64> {
    (0..len).map(|index| start + step * index as f64).collect()
}

fn one_hot_node_attrs(num_nodes: usize, num_elements: usize) -> Vec<f64> {
    let mut node_attrs = vec![0.0; num_nodes * num_elements];
    let element_ids = [0_usize, 5, 7, 13];
    for node in 0..num_nodes {
        let element = element_ids[node % element_ids.len()];
        node_attrs[node * num_elements + element] = 1.0;
    }
    node_attrs
}

fn bench_symmetric_contraction(c: &mut Criterion) {
    let num_elements = 89;
    let num_features = 128;
    let num_ell = 16;
    let u1_params = 1;
    let u2_params = 4;
    let u3_params = 23;
    let u1 = sequence(num_ell * u1_params, 0.02, 0.001);
    let u2 = sequence(num_ell * num_ell * u2_params, -0.03, 0.0007);
    let u3 = sequence(num_ell * num_ell * num_ell * u3_params, 0.01, -0.00003);
    let weights_u2 = sequence(num_elements * u2_params * num_features, 0.02, 0.00005);
    let weights_u1 = sequence(num_elements * u1_params * num_features, -0.01, 0.00008);
    let weights_u3 = sequence(num_elements * u3_params * num_features, 0.015, -0.00002);
    let contraction = MaceScalarSymmetricContraction {
        num_elements,
        num_features,
        num_ell,
        u1_params,
        u2_params,
        u3_params,
        u1: &u1,
        u2: &u2,
        u3: &u3,
        weights_u2: &weights_u2,
        weights_u1: &weights_u1,
        weights_u3: &weights_u3,
        entries: None,
    };

    let mut group = c.benchmark_group("mace_symmetric_contraction");
    for num_nodes in [1_usize, 4] {
        let x = sequence(num_nodes * num_features * num_ell, -0.3, 0.0009);
        let node_attrs = one_hot_node_attrs(num_nodes, num_elements);
        let grad_out = sequence(num_nodes * num_features, 0.2, -0.0007);
        group.bench_with_input(
            BenchmarkId::new("forward", num_nodes),
            &num_nodes,
            |b, _| {
                let mut out = vec![0.0; num_nodes * num_features];
                b.iter(|| {
                    contraction.forward(&x, &node_attrs, &mut out).unwrap();
                    black_box(out.iter().sum::<f64>())
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("forward_vjp", num_nodes),
            &num_nodes,
            |b, _| {
                let mut out = vec![0.0; num_nodes * num_features];
                let mut grad_x = vec![0.0; x.len()];
                b.iter(|| {
                    contraction.forward(&x, &node_attrs, &mut out).unwrap();
                    grad_x.fill(0.0);
                    contraction
                        .vjp_x(&x, &node_attrs, &grad_out, &mut grad_x)
                        .unwrap();
                    black_box(out.iter().chain(&grad_x).sum::<f64>())
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_symmetric_contraction);
criterion_main!(benches);
