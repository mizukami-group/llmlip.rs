// Build scripts fail during compilation, not in the inference runtime. Panics
// and unwraps here intentionally turn malformed build inputs into hard build
// failures with specific messages.
#![allow(clippy::panic, clippy::unwrap_used)]

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(llmlip_aarch64_sve_kernels)");

    // Intel MKL linking for x86_64 Linux.
    //
    // We use the LP64 interface (32-bit integers). By default, MKL is linked
    // through the sequential threading layer to avoid oversubscription when
    // Rayon is also in use. The opt-in `intel-mkl-threaded` feature switches
    // to `mkl_intel_thread + iomp5` for provider-matrix benchmarking.
    //
    // The library search path defaults to the standard oneAPI install location
    // but can be overridden via the MKLROOT environment variable.
    #[cfg(feature = "intel-mkl")]
    {
        println!("cargo:rerun-if-env-changed=MKLROOT");
        println!("cargo:rerun-if-env-changed=MKL_LIB_DIR");
        println!("cargo:rerun-if-env-changed=CMPLR_ROOT");

        // Use CARGO_CFG_TARGET_* so cross-compilation targets are handled
        // correctly (build scripts run on the host, not the target).
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
        if target_os == "linux" && target_arch == "x86_64" {
            let mkl_root = std::env::var("MKLROOT")
                .unwrap_or_else(|_| "/opt/intel/oneapi/mkl/latest".to_string());
            if let Ok(lib_dir) = std::env::var("MKL_LIB_DIR") {
                println!("cargo:rustc-link-search=native={lib_dir}");
            }
            for candidate in [
                format!("{mkl_root}/lib/intel64"),
                "/usr/lib/x86_64-linux-gnu".to_string(),
            ] {
                if std::path::Path::new(&candidate).is_dir() {
                    println!("cargo:rustc-link-search=native={candidate}");
                }
            }
            #[cfg(feature = "intel-mkl-threaded")]
            {
                let compiler_root = std::env::var("CMPLR_ROOT")
                    .unwrap_or_else(|_| "/opt/intel/oneapi/compiler/latest".to_string());
                for candidate in [
                    format!("{compiler_root}/linux/compiler/lib/intel64_lin"),
                    format!("{compiler_root}/lib/intel64_lin"),
                    format!("{compiler_root}/lib"),
                ] {
                    if std::path::Path::new(&candidate).is_dir() {
                        println!("cargo:rustc-link-search=native={candidate}");
                    }
                }
            }
            // Static MKL linking avoids runtime MKL_LIBRARY_PATH requirements.
            // Link order matters: lp64 interface → threading layer → core.
            println!("cargo:rustc-link-lib=static=mkl_intel_lp64");
            #[cfg(feature = "intel-mkl-threaded")]
            println!("cargo:rustc-link-lib=static=mkl_intel_thread");
            #[cfg(not(feature = "intel-mkl-threaded"))]
            println!("cargo:rustc-link-lib=static=mkl_sequential");
            println!("cargo:rustc-link-lib=static=mkl_core");
            // MKL core has circular dependencies; repeat to resolve them.
            #[cfg(feature = "intel-mkl-threaded")]
            println!("cargo:rustc-link-lib=static=mkl_intel_thread");
            #[cfg(not(feature = "intel-mkl-threaded"))]
            println!("cargo:rustc-link-lib=static=mkl_sequential");
            println!("cargo:rustc-link-lib=static=mkl_core");
            #[cfg(feature = "intel-mkl-threaded")]
            println!("cargo:rustc-link-lib=dylib=iomp5");
            // System libraries required by MKL static archives.
            println!("cargo:rustc-link-lib=dylib=pthread");
            println!("cargo:rustc-link-lib=dylib=m");
            println!("cargo:rustc-link-lib=dylib=dl");
        }
    }

    // Arm Performance Libraries linking for aarch64 Linux.
    //
    // `armpl` enables the same CBLAS call sites as `openblas`, but uses ArmPL's
    // LP64 OpenMP build. Runtime commands should set LD_LIBRARY_PATH to the
    // ArmPL lib directory and pin OMP_NUM_THREADS explicitly.
    #[cfg(feature = "armpl")]
    {
        println!("cargo:rerun-if-env-changed=ARMPL_ROOT");
        println!("cargo:rerun-if-env-changed=ARMPL_LIB_DIR");

        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
        if target_os != "linux" || target_arch != "aarch64" {
            panic!(
                "llmlip feature `armpl` is supported only for Linux/aarch64; \
                 use `openblas` or `intel-mkl` on {target_os}/{target_arch}"
            );
        }

        let lib_dir = std::env::var("ARMPL_LIB_DIR").unwrap_or_else(|_| {
            let root = std::env::var("ARMPL_ROOT")
                .unwrap_or_else(|_| "/opt/arm/arm-performance-libraries".to_string());
            format!("{root}/lib")
        });
        println!("cargo:rustc-link-search=native={lib_dir}");
        println!("cargo:rustc-link-lib=dylib=armpl_lp64_mp");
        // `armpl_lp64_mp` is the OpenMP (multi-threaded) ArmPL build, so it
        // depends on a GNU OpenMP runtime. Link `libgomp` explicitly; rustc
        // does not pass `-fopenmp`, so without this a final link of a binary
        // or test with `--features armpl` fails on undefined `GOMP_*`/`omp_*`
        // symbols.
        println!("cargo:rustc-link-lib=dylib=gomp");
        // ArmPL's vector-math library (libamath) provides the SVE
        // transcendentals used by the f64 SVE kernels
        // (`armpl_svexp_f64_u10_x`, 1-ULP). It is a separate `.so` from
        // `libarmpl_lp64_mp`, so link it explicitly.
        println!("cargo:rustc-link-lib=dylib=amath");
        println!("cargo:rustc-link-lib=dylib=m");
    }

    // OpenBLAS linking for Linux.
    //
    // This uses the system CBLAS interface and leaves threading policy to
    // OPENBLAS_NUM_THREADS, which benchmark commands pin explicitly.
    #[cfg(all(feature = "openblas", not(feature = "armpl")))]
    {
        println!("cargo:rerun-if-env-changed=OPENBLAS_LIB_DIR");

        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
        if target_os == "linux" && (target_arch == "x86_64" || target_arch == "aarch64") {
            if let Ok(lib_dir) = std::env::var("OPENBLAS_LIB_DIR") {
                println!("cargo:rustc-link-search=native={lib_dir}");
            }
            println!("cargo:rustc-link-lib=dylib=openblas");
            if target_arch == "aarch64" {
                println!("cargo:rustc-link-lib=dylib=m");
            }
        }
    }

    compile_aarch64_sve_kernels();

    // libxsmm linking for x86_64 Linux.
    //
    // LIBXSMM provides JIT-compiled small-matrix GEMM kernels that can be
    // faster than general BLAS for the dense layers in MACE inference (typical
    // M,N,K <= 128). The library search path defaults to /opt/libxsmm but can
    // be overridden via the LIBXSMM_ROOT environment variable.
    #[cfg(feature = "libxsmm")]
    {
        println!("cargo:rerun-if-env-changed=LIBXSMM_ROOT");

        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
        if target_os == "linux" && target_arch == "x86_64" {
            let xsmm_root =
                std::env::var("LIBXSMM_ROOT").unwrap_or_else(|_| "/opt/libxsmm".to_string());
            let lib_dir = format!("{xsmm_root}/lib");
            println!("cargo:rustc-link-search=native={lib_dir}");
            println!("cargo:rustc-link-lib=static=xsmm");
            // Some distro libxsmm static archives retain BLAS fallback symbols
            // (dgemm_/sgemm_/gemv_); link the system BLAS shim explicitly so
            // libxsmm-only test binaries link without also selecting a CBLAS
            // provider feature.
            println!("cargo:rustc-link-lib=dylib=blas");
            // System libraries required by libxsmm static archive.
            println!("cargo:rustc-link-lib=dylib=pthread");
            println!("cargo:rustc-link-lib=dylib=m");
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rustc-link-lib=dylib=rt");
        }
    }

    // Optional libmvec C shim for ORB activation probes. Rust stable cannot
    // pass SIMD vector types across FFI, so the AVX2 vector ABI stays inside C
    // and Rust calls only pointer-sized wrapper functions.
    #[cfg(feature = "orb-f32-edge-mlp")]
    compile_orb_mvec_shim();

    // Optional compile-time model embedding (feature `embed-model`).
    //
    // When `LLMLIP_EMBED_MODEL` points at a converted `.e3rs` model directory,
    // every file in it is baked into the binary via `include_bytes!` so the
    // model is loadable at runtime with `LlmlipModel::load_embedded()` and no
    // filesystem model path. When the variable is unset the embedded model is
    // simply absent; the default `LlmlipModel::load(path)` runtime is unchanged.
    #[cfg(feature = "embed-model")]
    generate_embedded_model();

    #[cfg(feature = "baked-mace-oam-medium")]
    generate_baked_mace_oam_medium();
    #[cfg(feature = "baked-mace-oam-small")]
    generate_baked_mace_oam_small();
    #[cfg(feature = "baked-mace-mp0b")]
    generate_baked_mace_mp0b();
    #[cfg(feature = "baked-mace-osaka24-small")]
    generate_baked_mace_osaka24_small();
    #[cfg(feature = "baked-mace-osaka24-medium")]
    generate_baked_mace_osaka24_medium();
    #[cfg(feature = "baked-mace-osaka24-large")]
    generate_baked_mace_osaka24_large();
    #[cfg(feature = "baked-mace-osaka26")]
    generate_baked_mace_osaka26();
}

#[cfg(feature = "orb-f32-edge-mlp")]
fn compile_orb_mvec_shim() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_os != "linux" || target_arch != "x86_64" || target_env != "gnu" {
        return;
    }

    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=AR");
    println!("cargo:rerun-if-env-changed=CFLAGS");
    println!("cargo:rerun-if-env-changed=TARGET_CC");
    println!("cargo:rerun-if-env-changed=TARGET_AR");
    println!("cargo:rerun-if-env-changed=TARGET_CFLAGS");
    println!("cargo:rerun-if-env-changed=LLMLIP_ORB_MVEC_CFLAGS");
    println!("cargo:rerun-if-changed=c/llmlip_orb_f32_mvec.c");

    let out_dir = std::path::PathBuf::from(
        std::env::var("OUT_DIR").expect("OUT_DIR must be set for ORB libmvec shim build"),
    );
    let obj = out_dir.join("llmlip_orb_f32_mvec.o");
    let archive = out_dir.join("libllmlip_orb_f32_mvec.a");
    let cc = std::env::var("TARGET_CC")
        .or_else(|_| std::env::var("CC"))
        .unwrap_or_else(|_| "cc".to_string());
    let ar = std::env::var("TARGET_AR")
        .or_else(|_| std::env::var("AR"))
        .unwrap_or_else(|_| "ar".to_string());

    if !tool_reports_library(&cc, "libmvec.so") && !tool_reports_library(&cc, "libmvec.a") {
        panic!(
            "orb-f32-edge-mlp on x86_64-unknown-linux-gnu requires glibc libmvec; \
             {cc:?} did not report libmvec.so or libmvec.a"
        );
    }

    let mut cc_command = std::process::Command::new(&cc);
    cc_command.args([
        "-O3",
        "-std=c11",
        "-fPIC",
        "-mavx2",
        "-mfma",
        "-fno-math-errno",
        "-c",
        "c/llmlip_orb_f32_mvec.c",
        "-o",
    ]);
    cc_command.arg(&obj);
    for flag in std::env::var("TARGET_CFLAGS")
        .or_else(|_| std::env::var("CFLAGS"))
        .unwrap_or_default()
        .split_whitespace()
    {
        cc_command.arg(flag);
    }
    for flag in std::env::var("LLMLIP_ORB_MVEC_CFLAGS")
        .unwrap_or_default()
        .split_whitespace()
    {
        cc_command.arg(flag);
    }

    let cc_status = cc_command.status();
    match cc_status {
        Ok(status) if status.success() => {}
        Ok(status) => panic!("failed to compile ORB libmvec shim: {cc} exited with {status}"),
        Err(err) => panic!("failed to run ORB libmvec shim compiler {cc}: {err}"),
    }

    let ar_status = std::process::Command::new(&ar)
        .arg("crs")
        .arg(&archive)
        .arg(&obj)
        .status();
    match ar_status {
        Ok(status) if status.success() => {}
        Ok(status) => panic!("failed to archive ORB libmvec shim: {ar} exited with {status}"),
        Err(err) => panic!("failed to run ORB libmvec shim archiver {ar}: {err}"),
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=llmlip_orb_f32_mvec");
    println!("cargo:rustc-link-lib=dylib=mvec");
    println!("cargo:rustc-link-lib=dylib=m");
}

#[cfg(feature = "orb-f32-edge-mlp")]
fn tool_reports_library(tool: &str, library: &str) -> bool {
    let output = std::process::Command::new(tool)
        .arg(format!("-print-file-name={library}"))
        .output();
    let Ok(output) = output else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    let path = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    !path.is_empty() && path != library && std::path::Path::new(&path).exists()
}

fn compile_aarch64_sve_kernels() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_os != "linux" || target_arch != "aarch64" {
        return;
    }

    println!("cargo:rerun-if-env-changed=CC");
    println!("cargo:rerun-if-env-changed=AR");
    println!("cargo:rerun-if-env-changed=LLMLIP_AARCH64_SVE_CFLAGS");
    let sources = [
        "src/arch/aarch64/mace_mp0b_sve.c",
        "src/arch/aarch64/uma_so2_sve.c",
    ];
    for source in sources {
        println!("cargo:rerun-if-changed={source}");
    }

    let out_dir = match std::env::var("OUT_DIR") {
        Ok(out_dir) => out_dir,
        Err(err) => {
            println!("cargo:warning=skipping aarch64 SVE kernels: OUT_DIR unavailable: {err}");
            return;
        }
    };
    let lib = std::path::Path::new(&out_dir).join("libllmlip_mace_mp0b_sve.a");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let ar = std::env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let extra_cflags = std::env::var("LLMLIP_AARCH64_SVE_CFLAGS").unwrap_or_default();
    let armpl_feature_enabled = std::env::var_os("CARGO_FEATURE_ARMPL").is_some();
    let mut objects = Vec::with_capacity(sources.len());

    for source in sources {
        let stem = std::path::Path::new(source)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("sve_kernel");
        let obj = std::path::Path::new(&out_dir).join(format!("{stem}.o"));
        let mut cc_command = std::process::Command::new(&cc);
        cc_command.args([
            "-O3",
            "-std=c11",
            "-fPIC",
            "-march=armv8.2-a+sve",
            "-c",
            source,
            "-o",
        ]);
        cc_command.arg(&obj);
        if armpl_feature_enabled {
            cc_command.arg("-fopenmp");
            cc_command.arg("-DLLMLIP_ENABLE_OMP_SVE=1");
            // ArmPL libamath provides the SVE f64 transcendentals
            // (`armpl_svexp_f64_u10_x`). Only compile the f64 SVE activation
            // kernels that depend on it when ArmPL is actually linked, so
            // non-ArmPL aarch64 builds do not pull an undefined symbol.
            cc_command.arg("-DLLMLIP_ENABLE_ARMPL_AMATH=1");
        }
        for flag in extra_cflags.split_whitespace() {
            cc_command.arg(flag);
        }

        let cc_status = cc_command.status();
        match cc_status {
            Ok(status) if status.success() => {}
            Ok(status) => {
                println!(
                    "cargo:warning=skipping aarch64 SVE kernels: {cc} failed for {source} with {status}"
                );
                return;
            }
            Err(err) => {
                println!(
                    "cargo:warning=skipping aarch64 SVE kernels: failed to run {cc} for {source}: {err}"
                );
                return;
            }
        }
        objects.push(obj);
    }

    let mut ar_command = std::process::Command::new(&ar);
    ar_command.arg("crs").arg(&lib);
    for obj in &objects {
        ar_command.arg(obj);
    }
    let ar_status = ar_command.status();
    match ar_status {
        Ok(status) if status.success() => {
            println!("cargo:rustc-link-search=native={out_dir}");
            println!("cargo:rustc-link-lib=static=llmlip_mace_mp0b_sve");
            println!("cargo:rustc-cfg=llmlip_aarch64_sve_kernels");
        }
        Ok(status) => {
            println!("cargo:warning=skipping aarch64 SVE kernels: {ar} exited with {status}");
        }
        Err(err) => {
            println!("cargo:warning=skipping aarch64 SVE kernels: failed to run {ar}: {err}");
        }
    }
}

#[cfg(feature = "embed-model")]
fn generate_embedded_model() {
    use std::fmt::Write as _;

    println!("cargo:rerun-if-env-changed=LLMLIP_EMBED_MODEL");
    let out_dir = std::env::var("OUT_DIR").unwrap_or_default();
    let dest = std::path::Path::new(&out_dir).join("embedded_model.rs");

    let mut code = String::new();
    match std::env::var("LLMLIP_EMBED_MODEL") {
        Ok(model_dir) if !model_dir.trim().is_empty() => {
            let root = std::path::Path::new(model_dir.trim());
            println!("cargo:rerun-if-changed={}", root.display());
            let mut entries: Vec<(String, String)> = Vec::new();
            collect_embedded_files(root, root, &mut entries);
            entries.sort();
            let _ = writeln!(code, "pub const EMBEDDED_MODEL_PRESENT: bool = true;");
            let _ = writeln!(
                code,
                "pub static EMBEDDED_MODEL_FILES: &[(&str, &[u8])] = &["
            );
            for (rel, abs) in &entries {
                println!("cargo:rerun-if-changed={abs}");
                let _ = writeln!(code, "    ({rel:?}, include_bytes!({abs:?})),");
            }
            let _ = writeln!(code, "];");
        }
        _ => {
            let _ = writeln!(code, "pub const EMBEDDED_MODEL_PRESENT: bool = false;");
            let _ = writeln!(
                code,
                "pub static EMBEDDED_MODEL_FILES: &[(&str, &[u8])] = &[];"
            );
        }
    }

    if let Err(err) = std::fs::write(&dest, code) {
        panic!("failed to write {}: {err}", dest.display());
    }
}

#[cfg(feature = "embed-model")]
fn collect_embedded_files(
    root: &std::path::Path,
    dir: &std::path::Path,
    out: &mut Vec<(String, String)>,
) {
    let read = match std::fs::read_dir(dir) {
        Ok(read) => read,
        Err(err) => panic!("LLMLIP_EMBED_MODEL: cannot read {}: {err}", dir.display()),
    };
    for entry in read.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_embedded_files(root, &path, out);
        } else if let Ok(rel) = path.strip_prefix(root) {
            out.push((
                rel.to_string_lossy().replace('\\', "/"),
                path.to_string_lossy().into_owned(),
            ));
        }
    }
}

#[cfg(feature = "baked-mace-model")]
#[derive(Clone, Debug)]
struct BakedTensorSpec {
    name: String,
    shape: Vec<usize>,
    path: std::path::PathBuf,
    sha256: Option<String>,
}

#[cfg(feature = "baked-mace-oam-medium")]
fn generate_baked_mace_oam_medium() {
    generate_baked_mace_model(BakedMaceModelConfig {
        label: "MACE-OAM Medium",
        env_var: "LLMLIP_BAKED_MACE_OAM_MEDIUM_MODEL",
        default_model_dir: "fixtures/mace/v0_3_16_4d2da09/models/mace_oam_medium_runtime.e3rs",
        selected_model: "mace-oam0-medium",
        generated_file: "mace_oam_medium_baked.rs",
    });
}

#[cfg(feature = "baked-mace-oam-small")]
fn generate_baked_mace_oam_small() {
    generate_baked_mace_model(BakedMaceModelConfig {
        label: "MACE-OAM Small",
        env_var: "LLMLIP_BAKED_MACE_OAM_SMALL_MODEL",
        default_model_dir: "fixtures/mace/v0_3_16_4d2da09/models/mace_oam_small_runtime.e3rs",
        selected_model: "mace-oam0-small",
        generated_file: "mace_oam_small_baked.rs",
    });
}

#[cfg(feature = "baked-mace-mp0b")]
fn generate_baked_mace_mp0b() {
    generate_baked_mace_model(BakedMaceModelConfig {
        label: "MACE-MP-0b",
        env_var: "LLMLIP_BAKED_MACE_MP0B_MODEL",
        default_model_dir: "fixtures/mace/v0_3_16_4d2da09/models/mace_mp0b_runtime.e3rs",
        selected_model: "medium-0b3",
        generated_file: "mace_mp0b_baked.rs",
    });
}

#[cfg(feature = "baked-mace-osaka24-small")]
fn generate_baked_mace_osaka24_small() {
    generate_baked_mace_model(BakedMaceModelConfig {
        label: "MACE-Osaka24 Small",
        env_var: "LLMLIP_BAKED_MACE_OSAKA24_SMALL_MODEL",
        default_model_dir: "fixtures/mace/v0_3_16_4d2da09/models/mace_osaka24_small_runtime.e3rs",
        selected_model: "mace-osaka24-small",
        generated_file: "mace_osaka24_small_baked.rs",
    });
}

#[cfg(feature = "baked-mace-osaka24-medium")]
fn generate_baked_mace_osaka24_medium() {
    generate_baked_mace_model(BakedMaceModelConfig {
        label: "MACE-Osaka24 Medium",
        env_var: "LLMLIP_BAKED_MACE_OSAKA24_MEDIUM_MODEL",
        default_model_dir: "fixtures/mace/v0_3_16_4d2da09/models/mace_osaka24_medium_runtime.e3rs",
        selected_model: "mace-osaka24-medium",
        generated_file: "mace_osaka24_medium_baked.rs",
    });
}

#[cfg(feature = "baked-mace-osaka24-large")]
fn generate_baked_mace_osaka24_large() {
    generate_baked_mace_model(BakedMaceModelConfig {
        label: "MACE-Osaka24 Large",
        env_var: "LLMLIP_BAKED_MACE_OSAKA24_LARGE_MODEL",
        default_model_dir: "fixtures/mace/v0_3_16_4d2da09/models/mace_osaka24_large_runtime.e3rs",
        selected_model: "mace-osaka24-large",
        generated_file: "mace_osaka24_large_baked.rs",
    });
}

#[cfg(feature = "baked-mace-osaka26")]
fn generate_baked_mace_osaka26() {
    generate_baked_mace_model(BakedMaceModelConfig {
        label: "MACE-Osaka26",
        env_var: "LLMLIP_BAKED_MACE_OSAKA26_MODEL",
        default_model_dir: "fixtures/mace/v0_3_16_4d2da09/models/mace_osaka26_runtime.e3rs",
        selected_model: "mace-osaka26-small",
        generated_file: "mace_osaka26_baked.rs",
    });
}

#[cfg(feature = "baked-mace-model")]
#[derive(Clone, Debug)]
struct BakedMaceModelConfig {
    label: &'static str,
    env_var: &'static str,
    default_model_dir: &'static str,
    selected_model: &'static str,
    generated_file: &'static str,
}

#[cfg(feature = "baked-mace-model")]
fn generate_baked_mace_model(config: BakedMaceModelConfig) {
    use std::io::Write as _;

    println!("cargo:rerun-if-env-changed={}", config.env_var);

    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let default_model_dir = manifest_dir.join("../..").join(config.default_model_dir);
    let model_dir = std::env::var(config.env_var)
        .map(std::path::PathBuf::from)
        .unwrap_or(default_model_dir);
    let model_dir = model_dir
        .canonicalize()
        .unwrap_or_else(|err| panic!("cannot canonicalize {}: {err}", model_dir.display()));
    let manifest_path = model_dir.join("manifest.toml");
    println!("cargo:rerun-if-changed={}", manifest_path.display());

    let manifest_text = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|err| panic!("cannot read {}: {err}", manifest_path.display()));
    let manifest: toml::Value = toml::from_str(&manifest_text)
        .unwrap_or_else(|err| panic!("cannot parse {}: {err}", manifest_path.display()));

    let architecture = manifest_string(&manifest, "architecture");
    if architecture != "mace-lite-v1" {
        panic!(
            "baked {} requires architecture \"mace-lite-v1\", got {architecture:?}",
            config.label
        );
    }
    let dtype = manifest_string(&manifest, "dtype");
    if dtype != "f64" {
        panic!(
            "baked {} requires f64 tensors, got dtype {dtype:?}",
            config.label
        );
    }

    let model_rel = manifest
        .get("mace_lite")
        .and_then(|value| value.get("model"))
        .and_then(toml::Value::as_str)
        .unwrap_or_else(|| panic!("{} is missing [mace_lite].model", manifest_path.display()));
    let model_path = model_dir.join(model_rel);
    println!("cargo:rerun-if-changed={}", model_path.display());
    let model_json_text = std::fs::read_to_string(&model_path)
        .unwrap_or_else(|err| panic!("cannot read {}: {err}", model_path.display()));
    let model_json: serde_json::Value = serde_json::from_str(&model_json_text)
        .unwrap_or_else(|err| panic!("cannot parse {}: {err}", model_path.display()));
    let selected_model = model_json
        .pointer("/provenance/selected_model")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if selected_model != config.selected_model {
        panic!(
            "expected {} selected_model {:?}, got {selected_model:?}",
            config.label, config.selected_model
        );
    }

    let tensors = baked_tensor_specs(&manifest, &model_dir);
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dest = std::path::Path::new(&out_dir).join(config.generated_file);
    let file = std::fs::File::create(&dest)
        .unwrap_or_else(|err| panic!("failed to create {}: {err}", dest.display()));
    let mut out = std::io::BufWriter::new(file);

    writeln!(
        out,
        "// @generated by crates/llmlip/build.rs for a {} model-specific binary.",
        config.label
    )
    .unwrap();
    writeln!(
        out,
        "pub(super) fn load() -> Result<super::MaceLiteModel, super::MaceLiteError> {{"
    )
    .unwrap();
    writeln!(out, "    let spec = {};", model_spec_expr(&model_json)).unwrap();
    writeln!(out, "    let external_tensors = external_tensors();").unwrap();
    writeln!(
        out,
        "    super::MaceLiteModel::from_spec(spec, external_tensors)"
    )
    .unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    writeln!(
        out,
        "fn external_tensors() -> std::collections::HashMap<String, super::MaceLiteExternalTensor> {{"
    )
    .unwrap();
    writeln!(
        out,
        "    let mut tensors = std::collections::HashMap::with_capacity({});",
        tensors.len()
    )
    .unwrap();
    for (index, tensor) in tensors.iter().enumerate() {
        writeln!(
            out,
            "    tensors.insert({:?}.to_owned(), super::MaceLiteExternalTensor {{ shape: {}, values: TENSOR_{index}.to_vec() }});",
            tensor.name,
            usize_vec_expr(&tensor.shape)
        )
        .unwrap();
    }
    writeln!(out, "    tensors").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    for (index, tensor) in tensors.iter().enumerate() {
        write_baked_tensor_static(&mut out, index, tensor);
    }
    out.flush()
        .unwrap_or_else(|err| panic!("failed to flush {}: {err}", dest.display()));
}

#[cfg(feature = "baked-mace-model")]
fn manifest_string(manifest: &toml::Value, key: &str) -> String {
    manifest
        .get(key)
        .and_then(toml::Value::as_str)
        .unwrap_or_else(|| panic!("manifest is missing string key {key:?}"))
        .to_owned()
}

#[cfg(feature = "baked-mace-model")]
fn baked_tensor_specs(manifest: &toml::Value, model_dir: &std::path::Path) -> Vec<BakedTensorSpec> {
    let tensors = manifest
        .get("tensors")
        .and_then(toml::Value::as_array)
        .unwrap_or_else(|| panic!("manifest is missing [[tensors]]"));
    tensors
        .iter()
        .map(|tensor| {
            let name = tensor
                .get("name")
                .and_then(toml::Value::as_str)
                .unwrap_or_else(|| panic!("manifest tensor is missing name"))
                .to_owned();
            let dtype = tensor
                .get("dtype")
                .and_then(toml::Value::as_str)
                .unwrap_or_else(|| panic!("manifest tensor {name:?} is missing dtype"));
            if dtype != "f64" {
                panic!("baked tensor {name:?} must be f64, got {dtype:?}");
            }
            let layout = tensor
                .get("layout")
                .and_then(toml::Value::as_str)
                .unwrap_or_else(|| panic!("manifest tensor {name:?} is missing layout"));
            if layout != "row-major" {
                panic!("baked tensor {name:?} must be row-major, got {layout:?}");
            }
            let shape = tensor
                .get("shape")
                .and_then(toml::Value::as_array)
                .unwrap_or_else(|| panic!("manifest tensor {name:?} is missing shape"))
                .iter()
                .map(|dim| {
                    dim.as_integer()
                        .unwrap_or_else(|| panic!("shape for tensor {name:?} must be integers"))
                        as usize
                })
                .collect::<Vec<_>>();
            let rel_path = tensor
                .get("path")
                .and_then(toml::Value::as_str)
                .unwrap_or_else(|| panic!("manifest tensor {name:?} is missing path"));
            let path = model_dir.join(rel_path);
            let sha256 = tensor
                .get("sha256")
                .and_then(toml::Value::as_str)
                .map(|value| {
                    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                        panic!("manifest tensor {name:?} sha256 must be a SHA256 hex digest");
                    }
                    value.to_owned()
                });
            println!("cargo:rerun-if-changed={}", path.display());
            BakedTensorSpec {
                name,
                shape,
                path,
                sha256,
            }
        })
        .collect()
}

#[cfg(feature = "baked-mace-model")]
fn write_baked_tensor_static<W: std::io::Write>(
    out: &mut W,
    index: usize,
    tensor: &BakedTensorSpec,
) {
    let bytes = std::fs::read(&tensor.path)
        .unwrap_or_else(|err| panic!("cannot read tensor {}: {err}", tensor.path.display()));
    let expected_len = tensor_shape_len(&tensor.shape) * std::mem::size_of::<f64>();
    if bytes.len() != expected_len {
        panic!(
            "tensor {:?} byte length mismatch: got {}, expected {}",
            tensor.name,
            bytes.len(),
            expected_len
        );
    }
    if let Some(expected_sha256) = &tensor.sha256 {
        let actual_sha256 = sha256_hex(&bytes);
        if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
            panic!(
                "tensor {:?} sha256 mismatch: got {}, expected {}",
                tensor.name, actual_sha256, expected_sha256
            );
        }
    }
    writeln!(
        out,
        "#[allow(clippy::approx_constant, clippy::excessive_precision)]"
    )
    .unwrap();
    writeln!(
        out,
        "static TENSOR_{index}: [f64; {}] = [",
        bytes.len() / std::mem::size_of::<f64>()
    )
    .unwrap();
    for (value_index, chunk) in bytes.chunks_exact(std::mem::size_of::<f64>()).enumerate() {
        let value = f64::from_le_bytes(chunk.try_into().unwrap());
        if !value.is_finite() {
            panic!(
                "tensor {:?} contains non-finite value at {value_index}",
                tensor.name
            );
        }
        if value_index % 4 == 0 {
            write!(out, "    ").unwrap();
        }
        // `Debug` formatting is round-trip for finite f64 values and keeps
        // production baked sources smaller than `f64::from_bits(...)` arrays.
        write!(out, "{:?}, ", value).unwrap();
        if value_index % 4 == 3 {
            writeln!(out).unwrap();
        }
    }
    if !bytes
        .chunks_exact(std::mem::size_of::<f64>())
        .len()
        .is_multiple_of(4)
    {
        writeln!(out).unwrap();
    }
    writeln!(out, "];").unwrap();
    writeln!(out).unwrap();
}

#[cfg(feature = "baked-mace-model")]
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest as _, Sha256};

    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").unwrap();
    }
    out
}

#[cfg(feature = "baked-mace-model")]
fn tensor_shape_len(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}

#[cfg(feature = "baked-mace-model")]
fn model_spec_expr(value: &serde_json::Value) -> String {
    format!(
        "super::ModelSpec {{ schema_version: {}, name: {}, type_names: {}, atomic_numbers: {}, cutoff: {}, l_max: {}, heads: {}, radial: {}, pair_repulsion: {}, layers: {}, readouts: {}, mp0b_runtime_plan: {}, osaka26_runtime_plan: {}, osaka_runtime_plan: {}, oam_runtime_plan: {}, provenance: {}, conv_tp_plans: {}, state_dict_format: {}, state_dict: Vec::new(), unsupported_features: {} }}",
        string_field(value, "schema_version"),
        string_field(value, "name"),
        string_vec_field(value, "type_names"),
        usize_vec_field(value, "atomic_numbers"),
        f64_field(value, "cutoff"),
        usize_field(value, "l_max"),
        string_vec_field(value, "heads"),
        radial_expr(object_field(value, "radial")),
        optional_object_expr(value, "pair_repulsion", pair_repulsion_expr),
        array_expr(value, "layers", layer_expr),
        array_expr(value, "readouts", readout_expr),
        optional_object_expr(value, "mp0b_runtime_plan", pinned_runtime_plan_expr),
        optional_object_expr(value, "osaka26_runtime_plan", pinned_runtime_plan_expr),
        optional_object_expr(value, "osaka_runtime_plan", pinned_runtime_plan_expr),
        optional_object_expr(value, "oam_runtime_plan", pinned_runtime_plan_expr),
        optional_object_expr(value, "provenance", provenance_expr),
        array_expr(value, "conv_tp_plans", tensor_product_plan_expr),
        optional_string_field(value, "state_dict_format"),
        string_vec_field(value, "unsupported_features"),
    )
}

#[cfg(feature = "baked-mace-model")]
fn provenance_expr(value: &serde_json::Value) -> String {
    format!(
        "super::MaceProvenanceSpec {{ source_family: {}, selected_model: {}, artifact: {}, reference: {}, state_dict_format: {}, upstream_state_tensor_count: {}, runtime_state_tensor_count: {}, archival_only_state_tensor_count: {}, upstream_state_dict_inventory: {} }}",
        string_field(value, "source_family"),
        string_field(value, "selected_model"),
        artifact_provenance_expr(object_field(value, "artifact")),
        reference_provenance_expr(object_field(value, "reference")),
        string_field(value, "state_dict_format"),
        usize_field(value, "upstream_state_tensor_count"),
        usize_field(value, "runtime_state_tensor_count"),
        usize_field(value, "archival_only_state_tensor_count"),
        string_field(value, "upstream_state_dict_inventory"),
    )
}

#[cfg(feature = "baked-mace-model")]
fn artifact_provenance_expr(value: &serde_json::Value) -> String {
    format!(
        "super::MaceArtifactProvenanceSpec {{ filename: {}, url: {}, sha256: {}, size_bytes: {} }}",
        string_field(value, "filename"),
        string_field(value, "url"),
        string_field(value, "sha256"),
        usize_field(value, "size_bytes"),
    )
}

#[cfg(feature = "baked-mace-model")]
fn reference_provenance_expr(value: &serde_json::Value) -> String {
    format!(
        "super::MaceReferenceProvenanceSpec {{ name: {}, repository: {}, commit: {}, package_version: {} }}",
        string_field(value, "name"),
        string_field(value, "repository"),
        string_field(value, "commit"),
        string_field(value, "package_version"),
    )
}

#[cfg(feature = "baked-mace-model")]
fn radial_expr(value: &serde_json::Value) -> String {
    format!(
        "super::RadialSpec {{ kind: {}, num_bessel: {}, polynomial_cutoff_p: {}, r_max: {}, distance_transform: {} }}",
        string_field(value, "type"),
        usize_field(value, "num_bessel"),
        usize_field(value, "polynomial_cutoff_p"),
        f64_field(value, "r_max"),
        optional_string_field(value, "distance_transform"),
    )
}

#[cfg(feature = "baked-mace-model")]
fn pair_repulsion_expr(value: &serde_json::Value) -> String {
    format!(
        "super::PairRepulsionSpec {{ kind: {}, polynomial_cutoff_p: {} }}",
        string_field(value, "type"),
        usize_field(value, "polynomial_cutoff_p"),
    )
}

#[cfg(feature = "baked-mace-model")]
fn layer_expr(value: &serde_json::Value) -> String {
    format!(
        "super::LayerSpec {{ interaction: {}, avg_num_neighbors_norm: {}, product: {} }}",
        string_field(value, "interaction"),
        f64_field(value, "avg_num_neighbors_norm"),
        product_expr(object_field(value, "product")),
    )
}

#[cfg(feature = "baked-mace-model")]
fn product_expr(value: &serde_json::Value) -> String {
    format!(
        "super::ProductSpec {{ correlation: {}, symmetric_contraction: {} }}",
        usize_field(value, "correlation"),
        symmetric_contraction_expr(object_field(value, "symmetric_contraction")),
    )
}

#[cfg(feature = "baked-mace-model")]
fn symmetric_contraction_expr(value: &serde_json::Value) -> String {
    format!(
        "super::SymmetricContractionSpec {{ irreps_in: {}, irreps_out: {}, num_elements: {}, irrep_normalization: {}, path_normalization: {}, use_reduced_cg: {}, u_tensors: {}, weight_tensors: {}, contractions: {} }}",
        string_field(value, "irreps_in"),
        string_field(value, "irreps_out"),
        usize_field(value, "num_elements"),
        string_field(value, "irrep_normalization"),
        string_field(value, "path_normalization"),
        bool_field(value, "use_reduced_cg"),
        string_vec_field(value, "u_tensors"),
        string_vec_field(value, "weight_tensors"),
        array_expr(value, "contractions", contraction_component_expr),
    )
}

#[cfg(feature = "baked-mace-model")]
fn contraction_component_expr(value: &serde_json::Value) -> String {
    format!(
        "super::SymmetricContractionComponentSpec {{ kind: {}, num_equivariance: {}, u_tensors: {}, weight_tensors: {} }}",
        string_field(value, "kind"),
        optional_usize_field(value, "num_equivariance"),
        string_vec_field(value, "u_tensors"),
        string_vec_field(value, "weight_tensors"),
    )
}

#[cfg(feature = "baked-mace-model")]
fn readout_expr(value: &serde_json::Value) -> String {
    format!(
        "super::ReadoutSpec {{ kind: {} }}",
        string_field(value, "kind")
    )
}

#[cfg(feature = "baked-mace-model")]
fn pinned_runtime_plan_expr(value: &serde_json::Value) -> String {
    format!(
        "super::PinnedMaceRuntimePlanSpec {{ schema_version: {}, num_elements: {}, num_features: {}, l_max: {}, sh_dim: {}, num_bessel: {}, hidden_row_len: {}, product0_row_len: {}, product1_row_len: {}, node_embedding: {}, layers: {}, readouts: {} }}",
        string_field(value, "schema_version"),
        usize_field(value, "num_elements"),
        usize_field(value, "num_features"),
        usize_field(value, "l_max"),
        usize_field(value, "sh_dim"),
        usize_field(value, "num_bessel"),
        usize_field(value, "hidden_row_len"),
        usize_field(value, "product0_row_len"),
        usize_field(value, "product1_row_len"),
        node_embedding_plan_expr(object_field(value, "node_embedding")),
        array_expr(value, "layers", pinned_runtime_layer_expr),
        array_expr(value, "readouts", pinned_runtime_readout_expr),
    )
}

#[cfg(feature = "baked-mace-model")]
fn node_embedding_plan_expr(value: &serde_json::Value) -> String {
    format!(
        "super::Mp0bNodeEmbeddingPlanSpec {{ input_row_len: {}, in_features: {}, out_features: {}, path_weight: {} }}",
        usize_field(value, "input_row_len"),
        usize_field(value, "in_features"),
        usize_field(value, "out_features"),
        f64_field(value, "path_weight"),
    )
}

#[cfg(feature = "baked-mace-model")]
fn pinned_runtime_layer_expr(value: &serde_json::Value) -> String {
    format!(
        "super::PinnedMaceRuntimeLayerPlanSpec {{ layer: {}, interaction: {}, linear_up_input_dim: {}, linear_up_output_dim: {}, conv_tp_weight_numel: {}, conv_tp_output_dim: {}, conv_tp_weight_mlp_hidden_dim: {}, density_input_dim: {}, density_output_dim: {}, linear_message_output_dim: {}, skip_output_dim: {}, skip_kind: {}, product_output_dim: {}, product_kind: {} }}",
        usize_field(value, "layer"),
        string_field(value, "interaction"),
        usize_field(value, "linear_up_input_dim"),
        usize_field(value, "linear_up_output_dim"),
        usize_field(value, "conv_tp_weight_numel"),
        usize_field(value, "conv_tp_output_dim"),
        usize_field(value, "conv_tp_weight_mlp_hidden_dim"),
        usize_field(value, "density_input_dim"),
        usize_field(value, "density_output_dim"),
        usize_field(value, "linear_message_output_dim"),
        usize_field(value, "skip_output_dim"),
        string_field(value, "skip_kind"),
        usize_field(value, "product_output_dim"),
        string_field(value, "product_kind"),
    )
}

#[cfg(feature = "baked-mace-model")]
fn pinned_runtime_readout_expr(value: &serde_json::Value) -> String {
    format!(
        "super::PinnedMaceRuntimeReadoutPlanSpec {{ layer: {}, kind: {}, input_dim: {} }}",
        usize_field(value, "layer"),
        string_field(value, "kind"),
        usize_field(value, "input_dim"),
    )
}

#[cfg(feature = "baked-mace-model")]
fn tensor_product_plan_expr(value: &serde_json::Value) -> String {
    format!(
        "super::TensorProductPlanSpec {{ layer: {}, irreps_in1: {}, irreps_in2: {}, irreps_out: {}, weight_numel: {}, instruction_in1_segment: {}, instruction_in2_segment: {}, instruction_out_segment: {}, instruction_cg_offset: {}, instruction_cg_len: {}, instruction_weight_offset: {}, instruction_path_weight: {}, cg_i_m: {}, cg_j_m: {}, cg_k_m: {}, cg_value: {} }}",
        usize_field(value, "layer"),
        string_field(value, "irreps_in1"),
        string_field(value, "irreps_in2"),
        string_field(value, "irreps_out"),
        usize_field(value, "weight_numel"),
        usize_vec_field(value, "instruction_in1_segment"),
        usize_vec_field(value, "instruction_in2_segment"),
        usize_vec_field(value, "instruction_out_segment"),
        usize_vec_field(value, "instruction_cg_offset"),
        usize_vec_field(value, "instruction_cg_len"),
        usize_vec_field(value, "instruction_weight_offset"),
        f64_vec_field(value, "instruction_path_weight"),
        usize_vec_field(value, "cg_i_m"),
        usize_vec_field(value, "cg_j_m"),
        usize_vec_field(value, "cg_k_m"),
        f64_vec_field(value, "cg_value"),
    )
}

#[cfg(feature = "baked-mace-model")]
fn object_field<'a>(value: &'a serde_json::Value, key: &str) -> &'a serde_json::Value {
    value
        .get(key)
        .unwrap_or_else(|| panic!("JSON object is missing key {key:?}"))
}

#[cfg(feature = "baked-mace-model")]
fn string_field(value: &serde_json::Value, key: &str) -> String {
    string_expr(
        value
            .get(key)
            .and_then(serde_json::Value::as_str)
            .unwrap_or_else(|| panic!("JSON field {key:?} must be a string")),
    )
}

#[cfg(feature = "baked-mace-model")]
fn optional_string_field(value: &serde_json::Value, key: &str) -> String {
    value.get(key).map_or_else(
        || "None".to_owned(),
        |field| {
            if field.is_null() {
                "None".to_owned()
            } else {
                format!(
                    "Some({})",
                    string_expr(
                        field
                            .as_str()
                            .unwrap_or_else(|| panic!("JSON field {key:?} must be a string"))
                    )
                )
            }
        },
    )
}

#[cfg(feature = "baked-mace-model")]
fn optional_usize_field(value: &serde_json::Value, key: &str) -> String {
    value.get(key).map_or_else(
        || "None".to_owned(),
        |field| {
            if field.is_null() {
                "None".to_owned()
            } else {
                format!(
                    "Some({})",
                    field
                        .as_u64()
                        .unwrap_or_else(|| panic!("JSON field {key:?} must be an integer"))
                )
            }
        },
    )
}

#[cfg(feature = "baked-mace-model")]
fn optional_object_expr(
    value: &serde_json::Value,
    key: &str,
    build: fn(&serde_json::Value) -> String,
) -> String {
    value.get(key).map_or_else(
        || "None".to_owned(),
        |field| {
            if field.is_null() {
                "None".to_owned()
            } else {
                format!("Some({})", build(field))
            }
        },
    )
}

#[cfg(feature = "baked-mace-model")]
fn string_vec_field(value: &serde_json::Value, key: &str) -> String {
    value.get(key).map_or_else(
        || "Vec::new()".to_owned(),
        |field| {
            let values = field
                .as_array()
                .unwrap_or_else(|| panic!("JSON field {key:?} must be an array"))
                .iter()
                .map(|item| {
                    string_expr(
                        item.as_str()
                            .unwrap_or_else(|| panic!("JSON field {key:?} must contain strings")),
                    )
                })
                .collect::<Vec<_>>();
            format!("vec![{}]", values.join(", "))
        },
    )
}

#[cfg(feature = "baked-mace-model")]
fn usize_field(value: &serde_json::Value, key: &str) -> usize {
    value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_else(|| panic!("JSON field {key:?} must be an integer")) as usize
}

#[cfg(feature = "baked-mace-model")]
fn usize_vec_field(value: &serde_json::Value, key: &str) -> String {
    value.get(key).map_or_else(
        || "Vec::new()".to_owned(),
        |field| {
            let values = field
                .as_array()
                .unwrap_or_else(|| panic!("JSON field {key:?} must be an array"))
                .iter()
                .map(|item| {
                    item.as_u64()
                        .unwrap_or_else(|| panic!("JSON field {key:?} must contain integers"))
                        .to_string()
                })
                .collect::<Vec<_>>();
            format!("vec![{}]", values.join(", "))
        },
    )
}

#[cfg(feature = "baked-mace-model")]
fn usize_vec_expr(values: &[usize]) -> String {
    let values = values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("vec![{values}]")
}

#[cfg(feature = "baked-mace-model")]
fn f64_field(value: &serde_json::Value, key: &str) -> String {
    let value = value
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .unwrap_or_else(|| panic!("JSON field {key:?} must be a number"));
    f64_expr(value)
}

#[cfg(feature = "baked-mace-model")]
fn f64_vec_field(value: &serde_json::Value, key: &str) -> String {
    value.get(key).map_or_else(
        || "Vec::new()".to_owned(),
        |field| {
            let values = field
                .as_array()
                .unwrap_or_else(|| panic!("JSON field {key:?} must be an array"))
                .iter()
                .map(|item| {
                    f64_expr(
                        item.as_f64()
                            .unwrap_or_else(|| panic!("JSON field {key:?} must contain numbers")),
                    )
                })
                .collect::<Vec<_>>();
            format!("vec![{}]", values.join(", "))
        },
    )
}

#[cfg(feature = "baked-mace-model")]
fn bool_field(value: &serde_json::Value, key: &str) -> bool {
    value
        .get(key)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or_else(|| panic!("JSON field {key:?} must be a bool"))
}

#[cfg(feature = "baked-mace-model")]
fn array_expr(
    value: &serde_json::Value,
    key: &str,
    build: fn(&serde_json::Value) -> String,
) -> String {
    value.get(key).map_or_else(
        || "Vec::new()".to_owned(),
        |field| {
            let values = field
                .as_array()
                .unwrap_or_else(|| panic!("JSON field {key:?} must be an array"))
                .iter()
                .map(build)
                .collect::<Vec<_>>();
            format!("vec![{}]", values.join(", "))
        },
    )
}

#[cfg(feature = "baked-mace-model")]
fn string_expr(value: &str) -> String {
    format!("{value:?}.to_owned()")
}

#[cfg(feature = "baked-mace-model")]
fn f64_expr(value: f64) -> String {
    if !value.is_finite() {
        panic!("baked MACE model spec contains non-finite float");
    }
    format!("f64::from_bits(0x{:016x})", value.to_bits())
}
