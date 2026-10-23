#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use llmlip::{
    all_pairs_neighbor_list, read_extxyz, Cell, LlmlipModel, PredictionComputeDtype,
    PredictionOptions, Structure,
};
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn load_model(relative_path: &str) -> LlmlipModel {
    let model_path = repo_root().join(relative_path);
    LlmlipModel::load(&model_path).unwrap_or_else(|err| {
        panic!(
            "failed to load representative benchmark model at {}: {err}",
            model_path.display()
        )
    })
}

fn try_load_model(relative_path: &str) -> Option<LlmlipModel> {
    let model_path = repo_root().join(relative_path);
    if !model_path.exists() {
        eprintln!(
            "skipping optional representative benchmark model at {}",
            model_path.display()
        );
        return None;
    }
    Some(LlmlipModel::load(&model_path).unwrap_or_else(|err| {
        panic!(
            "failed to load representative benchmark model at {}: {err}",
            model_path.display()
        )
    }))
}

fn load_structure(relative_path: &str) -> Structure {
    let structure_path = repo_root().join(relative_path);
    read_extxyz(&structure_path).unwrap_or_else(|err| {
        panic!(
            "failed to load representative benchmark structure at {}: {err}",
            structure_path.display()
        )
    })
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            matches!(
                value.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

fn mace_mp0b_broad_nonperiodic() -> Structure {
    Structure::new(
        ["H", "C", "O", "Si", "Fe", "Cu"]
            .iter()
            .map(|symbol| (*symbol).to_owned())
            .collect(),
        vec![
            [0.0, 0.0, 0.0],
            [1.2, 0.2, 0.1],
            [0.3, 1.4, 0.2],
            [1.7, 1.5, 0.4],
            [2.7, 0.5, 1.3],
            [3.3, 1.8, 1.8],
        ],
    )
}

fn mace_mp0b_water_nonperiodic() -> Structure {
    Structure::new(
        ["O", "H", "H"]
            .iter()
            .map(|symbol| (*symbol).to_owned())
            .collect(),
        vec![
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.239_987_2, 0.927_297, 0.0],
        ],
    )
}

fn mace_mp0b_complex_nonperiodic() -> Structure {
    Structure::new(
        ["C", "C", "O", "H", "H", "H", "H", "H", "H", "H", "C", "H"]
            .iter()
            .map(|symbol| (*symbol).to_owned())
            .collect(),
        vec![
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [2.3, 0.0, 0.0],
            [0.0, 0.9, 0.0],
            [-0.8, -0.7, 0.0],
            [0.0, 0.0, 0.9],
            [1.2, 1.0, 0.0],
            [2.0, -0.7, 0.0],
            [2.3, 0.0, 0.9],
            [3.1, 0.0, 0.0],
            [4.2, 0.0, 0.0],
            [5.0, 0.9, 0.0],
        ],
    )
}

fn mace_mp0b_triclinic_pbc() -> Structure {
    Structure::new(
        ["Li", "O", "Al", "S"]
            .iter()
            .map(|symbol| (*symbol).to_owned())
            .collect(),
        vec![
            [0.4, 0.5, 0.6],
            [7.7, 0.8, 0.9],
            [1.0, 7.9, 1.2],
            [2.2, 2.5, 7.8],
        ],
    )
    .with_cell(
        Cell::new([[8.3, 0.2, 0.1], [0.4, 8.7, 0.3], [0.2, 0.5, 9.1]]),
        [true, true, true],
    )
}

fn nequip_oam_hcosi_pbc() -> Structure {
    Structure::new(
        ["H", "C", "O", "Si"]
            .iter()
            .map(|symbol| (*symbol).to_owned())
            .collect(),
        vec![
            [0.20, 0.30, 0.40],
            [1.40, 0.55, 0.70],
            [0.85, 1.85, 0.90],
            [6.42, 0.30, 0.40],
        ],
    )
    .with_cell(
        Cell::new([[12.2, 0.0, 0.0], [0.0, 12.3, 0.0], [0.0, 0.0, 12.4]]),
        [true, true, true],
    )
}

fn energy_only() -> PredictionOptions {
    PredictionOptions {
        compute_energy: true,
        compute_forces: false,
        compute_stress: false,
        compute_virial: false,
        ..PredictionOptions::default()
    }
}

fn energy_force() -> PredictionOptions {
    PredictionOptions {
        compute_energy: true,
        compute_forces: true,
        compute_stress: false,
        compute_virial: false,
        ..PredictionOptions::default()
    }
}

fn energy_force_head(head: &str) -> PredictionOptions {
    PredictionOptions {
        compute_energy: true,
        compute_forces: true,
        compute_stress: false,
        compute_virial: false,
        selected_head: Some(head.to_owned()),
        ..PredictionOptions::default()
    }
}

fn energy_force_stress_virial() -> PredictionOptions {
    PredictionOptions {
        compute_energy: true,
        compute_forces: true,
        compute_stress: true,
        compute_virial: true,
        ..PredictionOptions::default()
    }
}

fn bench_representative_models(c: &mut Criterion) {
    let mace = load_model("fixtures/mace/v0_3_16_4d2da09/models/mace_mp0b_runtime.e3rs");
    let mace_osaka26 = load_model("fixtures/mace/v0_3_16_4d2da09/models/mace_osaka26_runtime.e3rs");
    let mace_mh1 = try_load_model(
        "outputs/reference-artifacts/mace/mace-mh1/github-release/mace_mh1_state.e3rs",
    );
    let nequip_oam_l =
        load_model("fixtures/nequip/v0_17_1_604ce73b/models/nequip_oam_l_prefix.e3rs");
    let mace_nonperiodic = mace_mp0b_broad_nonperiodic();
    let mace_water = mace_mp0b_water_nonperiodic();
    let mace_complex = mace_mp0b_complex_nonperiodic();
    let mace_pbc = mace_mp0b_triclinic_pbc();
    let mace_32_benzo = load_structure(
        "fixtures/mace/v0_3_16_4d2da09/benchmark_structures/structures_32atom_molecules/benzo_a_pyrene.extxyz",
    );
    let mace_64_polycyclic = load_structure(
        "fixtures/mace/v0_3_16_4d2da09/benchmark_structures/structures_64atom_molecules/polycyclic_c32h32.extxyz",
    );
    let mace_128_dimer = load_structure(
        "fixtures/mace/v0_3_16_4d2da09/benchmark_structures/structures_128atom_molecules/polycyclic_c32h32_dimer_separated_128.extxyz",
    );
    let mace_128_dense = load_structure(
        "fixtures/mace/v0_3_16_4d2da09/benchmark_structures/structures_128atom_molecules/1_7_3_ethyl_2_methyl_3_1_methylcycloundecyl_cycloheptyl_3_methyl_91109058.extxyz",
    );
    let mace_128_dense_steroid = load_structure(
        "fixtures/mace/v0_3_16_4d2da09/benchmark_structures/structures_128atom_molecules/5s_8s_9s_10s_13r_14s_17s_17_ethyl_1_2_5s_8s_9s_10s_13r_14s_17s_1_129755176.extxyz",
    );
    let nequip_32_benzo = load_structure("fixtures/nequip/structures/benzo_a_pyrene.extxyz");
    let nequip_64_polycyclic =
        load_structure("fixtures/nequip/structures/polycyclic_c32h32.extxyz");
    let nequip_128_dimer =
        load_structure("fixtures/nequip/structures/polycyclic_c32h32_dimer_separated_128.extxyz");
    let nequip_256_dimer_pair = load_structure(
        "fixtures/nequip/structures/polycyclic_c32h32_dimer_pair_separated_256.extxyz",
    );
    let nequip_128_dense = load_structure(
        "fixtures/nequip/structures/1_7_3_ethyl_2_methyl_3_1_methylcycloundecyl_cycloheptyl_3_methyl_91109058.extxyz",
    );
    let nequip_128_dense_steroid = load_structure(
        "fixtures/nequip/structures/5s_8s_9s_10s_13r_14s_17s_17_ethyl_1_2_5s_8s_9s_10s_13r_14s_17s_1_129755176.extxyz",
    );
    let nequip_pbc = nequip_oam_hcosi_pbc();

    let mut group = c.benchmark_group("representative_models");

    if let Some(mace_mh1) = mace_mh1.as_ref() {
        for head in [
            "matpes_r2scan",
            "mp_pbe_refit_add",
            "spice_wB97M",
            "oc20_usemppbe",
            "omol",
            "omat_pbe",
        ] {
            for (structure_label, structure) in [
                ("benzo_a_pyrene_32atom", &mace_32_benzo),
                ("polycyclic_c32h32_64atom", &mace_64_polycyclic),
                ("polycyclic_c32h32_dimer_128atom", &mace_128_dimer),
            ] {
                let label = format!("mace_mh1_runtime/{head}/{structure_label}/energy_force");
                let options = energy_force_head(head);
                group.bench_with_input(
                    BenchmarkId::from_parameter(label),
                    structure,
                    |b, structure| {
                        b.iter(|| {
                            black_box(
                                mace_mh1
                                    .predict(black_box(structure), black_box(&options))
                                    .unwrap(),
                            )
                        })
                    },
                );
            }
        }
    }

    for (label, structure, options) in [
        (
            "mace_mp0b_runtime/broad_nonperiodic/energy",
            &mace_nonperiodic,
            energy_only(),
        ),
        (
            "mace_mp0b_runtime/broad_nonperiodic/energy_force",
            &mace_nonperiodic,
            energy_force(),
        ),
        (
            "mace_mp0b_runtime/water_nonperiodic/energy_force",
            &mace_water,
            energy_force(),
        ),
        (
            "mace_mp0b_runtime/complex_nonperiodic/energy_force",
            &mace_complex,
            energy_force(),
        ),
        (
            "mace_mp0b_runtime/benzo_a_pyrene_32atom/energy_force",
            &mace_32_benzo,
            energy_force(),
        ),
        (
            "mace_mp0b_runtime/polycyclic_c32h32_64atom/energy_force",
            &mace_64_polycyclic,
            energy_force(),
        ),
        (
            "mace_mp0b_runtime/polycyclic_c32h32_dimer_128atom/energy_force",
            &mace_128_dimer,
            energy_force(),
        ),
        (
            "mace_mp0b_runtime/pubchem_91109058_128atom/energy_force",
            &mace_128_dense,
            energy_force(),
        ),
        (
            "mace_mp0b_runtime/pubchem_129755176_128atom/energy_force",
            &mace_128_dense_steroid,
            energy_force(),
        ),
        (
            "mace_mp0b_runtime/triclinic_pbc/energy_force_stress_virial",
            &mace_pbc,
            energy_force_stress_virial(),
        ),
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            structure,
            |b, structure| {
                b.iter(|| {
                    black_box(
                        mace.predict(black_box(structure), black_box(&options))
                            .unwrap(),
                    )
                })
            },
        );
    }

    for (label, structure, options) in [
        (
            "mace_osaka26_runtime/broad_nonperiodic/energy",
            &mace_nonperiodic,
            energy_only(),
        ),
        (
            "mace_osaka26_runtime/broad_nonperiodic/energy_force",
            &mace_nonperiodic,
            energy_force(),
        ),
        (
            "mace_osaka26_runtime/water_nonperiodic/energy_force",
            &mace_water,
            energy_force(),
        ),
        (
            "mace_osaka26_runtime/complex_nonperiodic/energy_force",
            &mace_complex,
            energy_force(),
        ),
        (
            "mace_osaka26_runtime/benzo_a_pyrene_32atom/energy_force",
            &mace_32_benzo,
            energy_force(),
        ),
        (
            "mace_osaka26_runtime/polycyclic_c32h32_64atom/energy_force",
            &mace_64_polycyclic,
            energy_force(),
        ),
        (
            "mace_osaka26_runtime/polycyclic_c32h32_dimer_128atom/energy_force",
            &mace_128_dimer,
            energy_force(),
        ),
        (
            "mace_osaka26_runtime/pubchem_91109058_128atom/energy_force",
            &mace_128_dense,
            energy_force(),
        ),
        (
            "mace_osaka26_runtime/pubchem_129755176_128atom/energy_force",
            &mace_128_dense_steroid,
            energy_force(),
        ),
        (
            "mace_osaka26_runtime/triclinic_pbc/energy_force_stress_virial",
            &mace_pbc,
            energy_force_stress_virial(),
        ),
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            structure,
            |b, structure| {
                b.iter(|| {
                    black_box(
                        mace_osaka26
                            .predict(black_box(structure), black_box(&options))
                            .unwrap(),
                    )
                })
            },
        );
    }

    for (label, structure, options) in [
        (
            "mace_osaka26_runtime_fp32/water_nonperiodic/energy_force",
            &mace_water,
            energy_force(),
        ),
        (
            "mace_osaka26_runtime_fp32/benzo_a_pyrene_32atom/energy_force",
            &mace_32_benzo,
            energy_force(),
        ),
        (
            "mace_osaka26_runtime_fp32/polycyclic_c32h32_64atom/energy_force",
            &mace_64_polycyclic,
            energy_force(),
        ),
        (
            "mace_osaka26_runtime_fp32/polycyclic_c32h32_dimer_128atom/energy_force",
            &mace_128_dimer,
            energy_force(),
        ),
        (
            "mace_osaka26_runtime_fp32/pubchem_91109058_128atom/energy_force",
            &mace_128_dense,
            energy_force(),
        ),
        (
            "mace_osaka26_runtime_fp32/pubchem_129755176_128atom/energy_force",
            &mace_128_dense_steroid,
            energy_force(),
        ),
        (
            "mace_osaka26_runtime_fp32/triclinic_pbc/energy_force_stress_virial",
            &mace_pbc,
            energy_force_stress_virial(),
        ),
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            structure,
            |b, structure| {
                b.iter(|| {
                    black_box(
                        mace_osaka26
                            .predict_with_compute_dtype(
                                black_box(structure),
                                black_box(&options),
                                PredictionComputeDtype::Float32,
                            )
                            .unwrap(),
                    )
                })
            },
        );
    }

    for (label, structure, options) in [
        (
            "nequip_oam_l_runtime/benzo_a_pyrene_32atom/energy_force",
            &nequip_32_benzo,
            energy_force(),
        ),
        (
            "nequip_oam_l_runtime/polycyclic_c32h32_64atom/energy_force",
            &nequip_64_polycyclic,
            energy_force(),
        ),
        (
            "nequip_oam_l_runtime/polycyclic_c32h32_dimer_128atom/energy_force",
            &nequip_128_dimer,
            energy_force(),
        ),
        (
            "nequip_oam_l_runtime/polycyclic_c32h32_dimer_pair_256atom/energy_force",
            &nequip_256_dimer_pair,
            energy_force(),
        ),
        (
            "nequip_oam_l_runtime/pubchem_91109058_128atom/energy_force",
            &nequip_128_dense,
            energy_force(),
        ),
        (
            "nequip_oam_l_runtime/pubchem_129755176_128atom/energy_force",
            &nequip_128_dense_steroid,
            energy_force(),
        ),
        (
            "nequip_oam_prefix/hcosi_pbc/energy",
            &nequip_pbc,
            energy_only(),
        ),
        (
            "nequip_oam_prefix/hcosi_pbc/energy_force",
            &nequip_pbc,
            energy_force(),
        ),
        (
            "nequip_oam_prefix/hcosi_pbc/energy_force_stress_virial",
            &nequip_pbc,
            energy_force_stress_virial(),
        ),
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            structure,
            |b, structure| {
                b.iter(|| {
                    black_box(
                        nequip_oam_l
                            .predict(black_box(structure), black_box(&options))
                            .unwrap(),
                    )
                })
            },
        );
    }

    for (label, structure, options) in [
        (
            "nequip_oam_l_runtime_fp32/benzo_a_pyrene_32atom/energy_force",
            &nequip_32_benzo,
            energy_force(),
        ),
        (
            "nequip_oam_l_runtime_fp32/polycyclic_c32h32_64atom/energy_force",
            &nequip_64_polycyclic,
            energy_force(),
        ),
        (
            "nequip_oam_l_runtime_fp32/polycyclic_c32h32_dimer_128atom/energy_force",
            &nequip_128_dimer,
            energy_force(),
        ),
        (
            "nequip_oam_l_runtime_fp32/polycyclic_c32h32_dimer_pair_256atom/energy_force",
            &nequip_256_dimer_pair,
            energy_force(),
        ),
        (
            "nequip_oam_l_runtime_fp32/pubchem_91109058_128atom/energy_force",
            &nequip_128_dense,
            energy_force(),
        ),
        (
            "nequip_oam_l_runtime_fp32/pubchem_129755176_128atom/energy_force",
            &nequip_128_dense_steroid,
            energy_force(),
        ),
        (
            "nequip_oam_prefix_fp32/hcosi_pbc/energy",
            &nequip_pbc,
            energy_only(),
        ),
        (
            "nequip_oam_prefix_fp32/hcosi_pbc/energy_force",
            &nequip_pbc,
            energy_force(),
        ),
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            structure,
            |b, structure| {
                b.iter(|| {
                    black_box(
                        nequip_oam_l
                            .predict_with_compute_dtype(
                                black_box(structure),
                                black_box(&options),
                                PredictionComputeDtype::Float32,
                            )
                            .unwrap(),
                    )
                })
            },
        );
    }

    for (label, structure) in [
        (
            "nequip_oam_l_runtime_overhead/benzo_a_pyrene_32atom/neighbor_list",
            &nequip_32_benzo,
        ),
        (
            "nequip_oam_l_runtime_overhead/polycyclic_c32h32_64atom/neighbor_list",
            &nequip_64_polycyclic,
        ),
        (
            "nequip_oam_l_runtime_overhead/polycyclic_c32h32_dimer_128atom/neighbor_list",
            &nequip_128_dimer,
        ),
        (
            "nequip_oam_l_runtime_overhead/polycyclic_c32h32_dimer_pair_256atom/neighbor_list",
            &nequip_256_dimer_pair,
        ),
        (
            "nequip_oam_l_runtime_overhead/pubchem_91109058_128atom/neighbor_list",
            &nequip_128_dense,
        ),
        (
            "nequip_oam_l_runtime_overhead/pubchem_129755176_128atom/neighbor_list",
            &nequip_128_dense_steroid,
        ),
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            structure,
            |b, structure| {
                b.iter(|| all_pairs_neighbor_list(black_box(structure), black_box(6.0)).unwrap())
            },
        );
    }

    if env_flag("LLMLIP_BENCH_PET_MAD_S_LARGE") {
        let pet_mad_s = match load_model(
            "outputs/reference-artifacts/pet/pet_mad_s_1_5_tensor_payloads.local.e3rs",
        ) {
            LlmlipModel::PetLite(model) => model,
            _ => panic!("expected PET-lite MAD-S model"),
        };
        let water2000 = load_structure(
            "fixtures/pet/v0_2_3_1f0b65c9/benchmark_structures/structures_6000atom_pet_mad_s/pet_mad_s_water2000_full_pbc.extxyz",
        );
        let si6000 = load_structure(
            "fixtures/pet/v0_2_3_1f0b65c9/benchmark_structures/structures_6000atom_pet_mad_s/pet_mad_s_si_diamond_conventional_10x15x5_full_pbc.extxyz",
        );
        for (structure_label, structure) in [
            ("water2000_full_pbc", &water2000),
            ("si6000_full_pbc", &si6000),
        ] {
            let batch = pet_mad_s
                .batch_for_full_pbc_structure(structure)
                .unwrap_or_else(|err| {
                    panic!("failed to prepare PET-MAD-S batch for {structure_label}: {err}")
                });
            let cell = structure.cell.as_ref();
            for (output_label, options, dtype) in [
                (
                    "prebuilt_batch_energy_f64",
                    energy_only(),
                    PredictionComputeDtype::Float64,
                ),
                (
                    "prebuilt_batch_energy_force_f64",
                    energy_force(),
                    PredictionComputeDtype::Float64,
                ),
                (
                    "prebuilt_batch_energy_f32",
                    energy_only(),
                    PredictionComputeDtype::Float32,
                ),
                (
                    "prebuilt_batch_energy_force_f32",
                    energy_force(),
                    PredictionComputeDtype::Float32,
                ),
            ] {
                let label = format!("pet_mad_s_runtime/{structure_label}/{output_label}");
                group.bench_with_input(BenchmarkId::from_parameter(label), &batch, |b, batch| {
                    b.iter(|| {
                        black_box(
                            pet_mad_s
                                .predict_for_batch_with_compute_dtype(
                                    black_box(batch),
                                    black_box(cell),
                                    black_box(&options),
                                    black_box(dtype),
                                )
                                .unwrap(),
                        )
                    })
                });
            }
        }
    }

    group.finish();
}

criterion_group!(benches, bench_representative_models);
criterion_main!(benches);
