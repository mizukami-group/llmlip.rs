#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn temp_output(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "llmlip_cli_{name}_{}_{}.xyz",
        std::process::id(),
        nanos
    ))
}

#[allow(dead_code)]
fn write_temp_pbc_input(name: &str) -> PathBuf {
    let path = temp_output(name);
    fs::write(
        &path,
        concat!(
            "3\n",
            "Lattice=\"8.5 0 0 0 8.7 0 0 0 8.9\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n",
            "H 0.2 0.3 0.4\n",
            "H 8.05 0.35 0.45\n",
            "O 0.25 8.2 0.5\n",
        ),
    )
    .unwrap();
    path
}

#[allow(dead_code)]
fn write_temp_mace_pbc_input(name: &str) -> PathBuf {
    let path = temp_output(name);
    fs::write(
        &path,
        concat!(
            "3\n",
            "Lattice=\"12.5 0 0 0 12.75 0 0 0 13.0\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n",
            "C 0.25 0.5 0.75\n",
            "H 12.0 0.75 1.0\n",
            "O 0.5 12.25 1.25\n",
        ),
    )
    .unwrap();
    path
}

#[allow(dead_code)]
fn write_temp_osaka26_input(name: &str) -> PathBuf {
    let path = temp_output(name);
    fs::write(
        &path,
        concat!(
            "3\n",
            "Properties=species:S:1:pos:R:3\n",
            "H 0.0 0.0 0.0\n",
            "C 1.12 0.15 0.05\n",
            "O 0.25 1.28 0.2\n",
        ),
    )
    .unwrap();
    path
}

#[allow(dead_code)]
fn write_temp_allegro_oam_l_input(name: &str) -> PathBuf {
    let path = temp_output(name);
    fs::write(
        &path,
        concat!(
            "3\n",
            "Properties=species:S:1:pos:R:3\n",
            "H 0.0 0.0 0.0\n",
            "O 0.96 0.0 0.0\n",
            "Si 0.1 1.4 0.2\n",
        ),
    )
    .unwrap();
    path
}

#[allow(dead_code)]
fn write_temp_osaka26_pbc_input(name: &str) -> PathBuf {
    let path = temp_output(name);
    fs::write(
        &path,
        concat!(
            "3\n",
            "Lattice=\"12.0 0 0 0 12.25 0 0 0 12.5\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n",
            "H 0.3 0.4 0.5\n",
            "C 11.7 0.6 0.8\n",
            "O 0.5 11.9 1.0\n",
        ),
    )
    .unwrap();
    path
}

#[allow(dead_code)]
fn write_temp_pet_water_input(name: &str) -> PathBuf {
    let path = temp_output(name);
    fs::write(
        &path,
        concat!(
            "3\n",
            "Properties=species:S:1:pos:R:3\n",
            "O -0.5 0.763239 0.596309\n",
            "H -0.5 1.526478 0.0\n",
            "H -0.5 0.0 0.0\n",
        ),
    )
    .unwrap();
    path
}

#[allow(dead_code)]
fn write_temp_pet_shifted_pbc_input(name: &str) -> PathBuf {
    let path = temp_output(name);
    fs::write(
        &path,
        concat!(
            "4\n",
            "Lattice=\"12.0 0 0 0 12.2 0 0 0 12.4\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n",
            "Li 0.20 0.30 0.40\n",
            "O 1.45 0.55 0.65\n",
            "H 0.60 1.40 0.70\n",
            "H 11.70 0.35 0.45\n",
        ),
    )
    .unwrap();
    path
}

#[cfg(all(feature = "apple-metal", target_os = "macos", target_arch = "aarch64"))]
fn write_temp_polarmace_water_input(name: &str) -> PathBuf {
    let path = temp_output(name);
    fs::write(
        &path,
        concat!(
            "3\n",
            "Properties=species:S:1:pos:R:3\n",
            "O 0.0 0.0 0.0\n",
            "H 0.9584 0.0 0.0\n",
            "H -0.239 0.927 0.0\n",
        ),
    )
    .unwrap();
    path
}

#[cfg(all(feature = "apple-metal", target_os = "macos", target_arch = "aarch64"))]
fn polarmace_polar_1_s_bundle(root: &Path) -> PathBuf {
    root.join("outputs/reference-artifacts/mace/polarmace_polar_1_s_state.e3rs")
}

#[cfg(all(feature = "apple-metal", target_os = "macos", target_arch = "aarch64"))]
fn command_with_polarmace_metal_gate(metal: bool) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_llmlip"));
    command
        .env("RAYON_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("MKL_NUM_THREADS", "1")
        .env("VECLIB_MAXIMUM_THREADS", "1");
    if metal {
        command
            .env("LLMLIP_MACE_POLAR_METAL", "1")
            .env("LLMLIP_MACE_METAL", "1");
    } else {
        command
            .env_remove("LLMLIP_MACE_POLAR_METAL")
            .env_remove("LLMLIP_MACE_METAL");
    }
    command
}

#[cfg(all(feature = "apple-metal", target_os = "macos", target_arch = "aarch64"))]
fn run_polarmace_predict_json(
    root: &Path,
    model: &Path,
    structure: &Path,
    metal: bool,
) -> serde_json::Value {
    let result = command_with_polarmace_metal_gate(metal)
        .current_dir(root)
        .args([
            "predict",
            "--model",
            model.to_str().unwrap(),
            "--structure",
            structure.to_str().unwrap(),
            "--energy",
            "--forces",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    serde_json::from_slice(&result.stdout).unwrap()
}

#[cfg(all(feature = "apple-metal", target_os = "macos", target_arch = "aarch64"))]
fn run_polarmace_prepared_checksum(
    root: &Path,
    model: &Path,
    structure: &Path,
    metal: bool,
) -> f64 {
    let result = command_with_polarmace_metal_gate(metal)
        .current_dir(root)
        .args([
            "benchmark-predict",
            "--model",
            model.to_str().unwrap(),
            "--structure",
            structure.to_str().unwrap(),
            "--energy",
            "--forces",
            "--prepared-graph",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--label",
            if metal {
                "polarmace_metal_smoke"
            } else {
                "polarmace_cpu_smoke"
            },
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    report["checksum"].as_f64().unwrap()
}

#[cfg(all(feature = "apple-metal", target_os = "macos", target_arch = "aarch64"))]
fn assert_prediction_json_close(cpu: &serde_json::Value, metal: &serde_json::Value) {
    let cpu_energy = cpu["energy"].as_f64().unwrap();
    let metal_energy = metal["energy"].as_f64().unwrap();
    let energy_delta = (cpu_energy - metal_energy).abs();
    assert!(
        energy_delta <= 1.0e-5,
        "PolarMACE Metal-vs-CPU energy delta={energy_delta:e} cpu={cpu_energy:e} metal={metal_energy:e}"
    );

    let cpu_forces = cpu["forces"].as_array().unwrap();
    let metal_forces = metal["forces"].as_array().unwrap();
    assert_eq!(cpu_forces.len(), metal_forces.len(), "force row count");
    let mut max_force_delta = 0.0_f64;
    let mut worst_force = String::new();
    for (atom, (cpu_row, metal_row)) in cpu_forces.iter().zip(metal_forces).enumerate() {
        let cpu_components = cpu_row.as_array().unwrap();
        let metal_components = metal_row.as_array().unwrap();
        assert_eq!(
            cpu_components.len(),
            metal_components.len(),
            "force component count for atom {atom}"
        );
        for axis in 0..cpu_components.len() {
            let cpu_value = cpu_components[axis].as_f64().unwrap();
            let metal_value = metal_components[axis].as_f64().unwrap();
            let delta = (cpu_value - metal_value).abs();
            if delta > max_force_delta {
                max_force_delta = delta;
                worst_force =
                    format!("atom={atom} axis={axis} cpu={cpu_value:e} metal={metal_value:e}");
            }
        }
    }
    assert!(
        max_force_delta <= 1.0e-4,
        "PolarMACE Metal-vs-CPU max force delta={max_force_delta:e} {worst_force}"
    );
}

#[cfg(all(feature = "apple-metal", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn polarmace_metal_vs_cpu_energy_forces_smoke_when_bundle_is_present() {
    let root = repo_root();
    let model = polarmace_polar_1_s_bundle(&root);
    if !model.is_dir() {
        eprintln!(
            "skipping PolarMACE Metal-vs-CPU E+F smoke; runtime bundle not found: {}",
            model.display()
        );
        return;
    }
    let structure = write_temp_polarmace_water_input("polarmace_metal_cpu_smoke");

    let cpu = run_polarmace_predict_json(&root, &model, &structure, false);
    let metal = run_polarmace_predict_json(&root, &model, &structure, true);
    assert_prediction_json_close(&cpu, &metal);

    let cpu_prepared_checksum = run_polarmace_prepared_checksum(&root, &model, &structure, false);
    let metal_prepared_checksum = run_polarmace_prepared_checksum(&root, &model, &structure, true);
    let checksum_delta = (cpu_prepared_checksum - metal_prepared_checksum).abs();
    assert!(
        checksum_delta <= 1.0e-4,
        "PolarMACE prepared Metal-vs-CPU checksum delta={checksum_delta:e} cpu={cpu_prepared_checksum:e} metal={metal_prepared_checksum:e}"
    );

    fs::remove_file(structure).unwrap();
}

#[test]
fn predict_prints_energy_and_writes_forces() {
    let root = repo_root();
    let output = temp_output("forces");
    let result = Command::new(env!("CARGO_BIN_EXE_llmlip"))
        .current_dir(&root)
        .args([
            "predict",
            "--model",
            "examples/model.e3rs",
            "--structure",
            "examples/input.extxyz",
            "--energy",
            "--forces-out",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(stdout.contains("energy = "));
    let forces = fs::read_to_string(&output).unwrap();
    assert!(forces.contains("Properties=species:S:1:pos:R:3:force:R:3"));
    assert!(forces.contains("energy="));
    fs::remove_file(output).unwrap();
}

#[test]
fn predict_keeps_forces_alias_for_backwards_compatibility() {
    let root = repo_root();
    let output = temp_output("forces_alias");
    let result = Command::new(env!("CARGO_BIN_EXE_llmlip"))
        .current_dir(&root)
        .args([
            "predict",
            "--model",
            "examples/model.e3rs",
            "--structure",
            "examples/input.extxyz",
            "--forces",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        String::from_utf8(result.stdout).unwrap().is_empty(),
        "force-only prediction should not print energy"
    );
    let forces = fs::read_to_string(&output).unwrap();
    assert!(forces.contains("Properties=species:S:1:pos:R:3:force:R:3"));
    fs::remove_file(output).unwrap();
}

#[test]
fn predict_json_emits_energy_and_forces_without_forces_file() {
    let root = repo_root();
    let result = Command::new(env!("CARGO_BIN_EXE_llmlip"))
        .current_dir(&root)
        .args([
            "predict",
            "--model",
            "examples/model.e3rs",
            "--structure",
            "examples/input.extxyz",
            "--energy",
            "--forces",
            "--json",
        ])
        .output()
        .unwrap();

    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stderr).is_empty());
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["schema_version"], "llmlip-prediction-v1");
    assert!(report["energy"].as_f64().unwrap().is_finite());
    let forces = report["forces"].as_array().unwrap();
    assert_eq!(forces.len(), 3);
    assert_eq!(forces[0].as_array().unwrap().len(), 3);
    assert!(report["stress"].is_null());
    assert!(report["virial"].is_null());
}

#[test]
fn predict_json_forces_flag_does_not_consume_following_energy_flag() {
    let root = repo_root();
    let cwd = temp_output("json_forces_flag_cwd");
    fs::create_dir(&cwd).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_llmlip"))
        .current_dir(&cwd)
        .args([
            "predict",
            "--model",
            root.join("examples/model.e3rs").to_str().unwrap(),
            "--structure",
            root.join("examples/input.extxyz").to_str().unwrap(),
            "--forces",
            "--energy",
            "--json",
        ])
        .output()
        .unwrap();

    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        !cwd.join("--energy").exists(),
        "--energy must be parsed as a flag, not as the optional --forces output path"
    );
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert!(report["energy"].as_f64().unwrap().is_finite());
    assert_eq!(report["forces"].as_array().unwrap().len(), 3);
    fs::remove_dir(&cwd).unwrap();
}

#[test]
fn predict_json_can_also_write_forces_extxyz() {
    let root = repo_root();
    let output = temp_output("json_forces_out");
    let result = Command::new(env!("CARGO_BIN_EXE_llmlip"))
        .current_dir(&root)
        .args([
            "predict",
            "--model",
            "examples/model.e3rs",
            "--structure",
            "examples/input.extxyz",
            "--energy",
            "--forces-out",
            output.to_str().unwrap(),
            "--json",
        ])
        .output()
        .unwrap();

    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["schema_version"], "llmlip-prediction-v1");
    assert!(report["forces"].as_array().unwrap().len() == 3);
    let forces = fs::read_to_string(&output).unwrap();
    assert!(forces.contains("Properties=species:S:1:pos:R:3:force:R:3"));
    assert!(forces.contains("energy="));
    fs::remove_file(output).unwrap();
}

#[test]
fn predict_forces_without_path_requires_json() {
    let root = repo_root();
    let result = Command::new(env!("CARGO_BIN_EXE_llmlip"))
        .current_dir(&root)
        .args([
            "predict",
            "--model",
            "examples/model.e3rs",
            "--structure",
            "examples/input.extxyz",
            "--forces",
        ])
        .output()
        .unwrap();

    assert!(!result.status.success());
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(stderr.contains("--forces without PATH is supported only with --json"));
}

#[test]
fn finite_difference_forces_reports_full_3n_json() {
    let root = repo_root();
    let result = Command::new(env!("CARGO_BIN_EXE_llmlip"))
        .current_dir(&root)
        .args([
            "finite-difference-forces",
            "--model",
            "examples/model.e3rs",
            "--structure",
            "examples/input.extxyz",
            "--fd-mode",
            "full_3n",
            "--fd-eps",
            "1e-5",
            "--fd-tol",
            "1e-3",
        ])
        .output()
        .unwrap();

    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(
        report["schema_version"],
        "llmlip-force-finite-difference-report-v1"
    );
    assert_eq!(report["mode"], "full_3n");
    assert_eq!(report["total_component_count"], 9);
    assert_eq!(report["selected_component_start"], 0);
    assert!(report["selected_component_limit"].is_null());
    assert_eq!(report["component_count"], 9);
    assert_eq!(report["energy_eval_count"], 18);
    assert!(report["passed"].as_bool().unwrap());
    assert!(report["worst_components"].as_array().unwrap().len() <= 10);
}

#[test]
fn finite_difference_forces_supports_component_chunks() {
    let root = repo_root();
    let result = Command::new(env!("CARGO_BIN_EXE_llmlip"))
        .current_dir(&root)
        .args([
            "finite-difference-forces",
            "--model",
            "examples/model.e3rs",
            "--structure",
            "examples/input.extxyz",
            "--fd-mode",
            "full_3n",
            "--component-start",
            "2",
            "--component-limit",
            "3",
        ])
        .output()
        .unwrap();

    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["total_component_count"], 9);
    assert_eq!(report["selected_component_start"], 2);
    assert_eq!(report["selected_component_limit"], 3);
    assert_eq!(report["component_count"], 3);
    assert_eq!(report["energy_eval_count"], 6);
    assert_eq!(report["worst_components"].as_array().unwrap().len(), 3);
}

#[test]
fn benchmark_predict_reports_load_once_timing_json() {
    let root = repo_root();
    let result = Command::new(env!("CARGO_BIN_EXE_llmlip"))
        .current_dir(&root)
        .args([
            "benchmark-predict",
            "--model",
            "examples/model.e3rs",
            "--structure",
            "examples/input.extxyz",
            "--energy",
            "--forces",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--label",
            "cli_smoke",
        ])
        .output()
        .unwrap();

    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["schema_version"], "llmlip-predict-benchmark-v1.1");
    assert_eq!(report["label"], "cli_smoke");
    assert_eq!(report["atom_count"], 3);
    assert_eq!(report["compute_dtype"], "float64");
    assert_eq!(report["requested_outputs"]["energy"], true);
    assert_eq!(report["requested_outputs"]["forces"], true);
    assert_eq!(report["warmup"], 1);
    assert_eq!(report["iterations"], 2);
    assert_eq!(report["preparation_runtime_ms"], 0.0);
    assert!(report["edge_count"].is_null());
    assert_eq!(report["measurement_scope"]["model_load_excluded"], true);
    assert_eq!(
        report["measurement_scope"]["structure_file_read_excluded"],
        true
    );
    assert_eq!(
        report["measurement_scope"]["neighbor_graph_setup_excluded"],
        false
    );
    assert_eq!(report["measurement_scope"]["prepared_scope_match"], false);
    assert_eq!(report["samples_ms"].as_array().unwrap().len(), 2);
    assert!(report["median_ms"].as_f64().unwrap() >= 0.0);
    assert!(report["p95_ms"].as_f64().unwrap() >= 0.0);
    assert!(report["checksum"].as_f64().unwrap().is_finite());
    assert!(
        report["environment"]["thread_env"]
            .as_array()
            .unwrap()
            .len()
            >= 6
    );
    assert!(report["environment"]["logical_cpus"].as_u64().unwrap() >= 1);
    assert_eq!(report["environment"]["target_os"], std::env::consts::OS);
    assert_eq!(report["environment"]["target_arch"], std::env::consts::ARCH);
    assert_eq!(
        report["environment"]["target_family"],
        std::env::consts::FAMILY
    );
    assert!(!report["environment"]["enabled_features"]
        .as_array()
        .unwrap()
        .iter()
        .any(|feature| feature == "reference-fixtures"));
}

#[test]
fn benchmark_predict_prepared_graph_rejects_unsupported_model() {
    let root = repo_root();
    let result = Command::new(env!("CARGO_BIN_EXE_llmlip"))
        .current_dir(&root)
        .args([
            "benchmark-predict",
            "--model",
            "examples/model.e3rs",
            "--structure",
            "examples/input.extxyz",
            "--energy",
            "--forces",
            "--prepared-graph",
            "--warmup",
            "0",
            "--iterations",
            "1",
        ])
        .output()
        .unwrap();

    assert!(!result.status.success());
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(stderr.contains("prepared prediction is not supported"));
}

#[test]
fn batch_predict_falls_back_for_models_without_prepared_prediction() {
    let root = repo_root();
    let output = temp_output("batch_predict_fallback");
    let result = Command::new(env!("CARGO_BIN_EXE_llmlip"))
        .current_dir(&root)
        .args([
            "batch-predict",
            "--model",
            "examples/model.e3rs",
            "--structures",
            "examples/input.extxyz",
            "--energy",
            "--forces",
            "--out",
            output.to_str().unwrap(),
            "--start-index",
            "0",
            "--max-structures",
            "1",
        ])
        .output()
        .unwrap();

    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&output).unwrap()).unwrap();
    assert_eq!(report["schema_version"], "llmlip-batch-predict-v1");
    assert_eq!(report["structure_count"], 1);
    assert_eq!(report["success_count"], 1);
    assert_eq!(report["error_count"], 0);
    assert_eq!(report["environment"]["target_os"], std::env::consts::OS);
    assert_eq!(report["environment"]["target_arch"], std::env::consts::ARCH);
    assert!(!report["environment"]["enabled_features"]
        .as_array()
        .unwrap()
        .iter()
        .any(|feature| feature == "reference-fixtures"));
    assert!(report["results"][0]["energy"].as_f64().unwrap().is_finite());
    assert!(report["results"][0]["forces"].as_array().is_some());
    assert_eq!(report["results"][0]["graph_setup_ms"], 0.0);
    fs::remove_file(output).unwrap();
}

#[test]
fn benchmark_predict_rejects_missing_outputs() {
    let root = repo_root();
    let result = Command::new(env!("CARGO_BIN_EXE_llmlip"))
        .current_dir(&root)
        .args([
            "benchmark-predict",
            "--model",
            "examples/model.e3rs",
            "--structure",
            "examples/input.extxyz",
        ])
        .output()
        .unwrap();

    assert!(!result.status.success());
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(stderr.contains("benchmark-predict requires at least one of"));
}

#[test]
fn predict_rejects_missing_outputs() {
    let root = repo_root();
    let result = Command::new(env!("CARGO_BIN_EXE_llmlip"))
        .current_dir(&root)
        .args([
            "predict",
            "--model",
            "examples/model.e3rs",
            "--structure",
            "examples/input.extxyz",
        ])
        .output()
        .unwrap();

    assert!(!result.status.success());
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(stderr.contains("predict requires at least one of"));
    assert!(stderr.contains("--energy"));
    assert!(stderr.contains("--forces-out"));
    assert!(stderr.contains("--stress"));
    assert!(stderr.contains("--virial"));
}

#[test]
fn predict_rejects_selected_head_for_headless_model() {
    let root = repo_root();
    let result = Command::new(env!("CARGO_BIN_EXE_llmlip"))
        .current_dir(&root)
        .args([
            "predict",
            "--model",
            "examples/model.e3rs",
            "--structure",
            "examples/input.extxyz",
            "--energy",
            "--head",
            "omat_pbe",
        ])
        .output()
        .unwrap();

    assert!(!result.status.success());
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(stderr.contains("unsupported prediction head"));
    assert!(stderr.contains("omat_pbe"));
}

#[test]
fn predict_writes_forces_without_energy_field_when_energy_not_requested() {
    let root = repo_root();
    let output = temp_output("forces_only");
    let result = Command::new(env!("CARGO_BIN_EXE_llmlip"))
        .current_dir(&root)
        .args([
            "predict",
            "--model",
            "examples/model.e3rs",
            "--structure",
            "examples/input.extxyz",
            "--forces-out",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        String::from_utf8(result.stdout).unwrap().is_empty(),
        "force-only prediction should not print energy"
    );
    let forces = fs::read_to_string(&output).unwrap();
    assert!(forces.contains("Properties=species:S:1:pos:R:3:force:R:3"));
    assert!(!forces.contains("energy="));
    fs::remove_file(output).unwrap();
}

#[test]
fn predict_reports_missing_structure() {
    let root = repo_root();
    let result = Command::new(env!("CARGO_BIN_EXE_llmlip"))
        .current_dir(&root)
        .args([
            "predict",
            "--model",
            "examples/model.e3rs",
            "--structure",
            "examples/does-not-exist.extxyz",
            "--energy",
        ])
        .output()
        .unwrap();

    assert!(!result.status.success());
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(stderr.contains("failed to read structure"));
}

#[test]
fn validate_model_reports_missing_model() {
    let root = repo_root();
    let result = Command::new(env!("CARGO_BIN_EXE_llmlip"))
        .current_dir(&root)
        .args(["validate-model", "--model", "fixtures/pet/missing.e3rs"])
        .output()
        .unwrap();

    assert!(!result.status.success());
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(stderr.contains("failed to validate model"));
    assert!(stderr.contains("manifest.toml"));
}

#[test]
fn predict_reports_stress_unsupported_for_non_pbc_model() {
    let root = repo_root();
    let result = Command::new(env!("CARGO_BIN_EXE_llmlip"))
        .current_dir(&root)
        .args([
            "predict",
            "--model",
            "examples/model.e3rs",
            "--structure",
            "examples/input.extxyz",
            "--stress",
        ])
        .output()
        .unwrap();

    assert!(!result.status.success());
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(stderr.contains("prediction failed"));
    assert!(stderr.contains("stress"));
}
