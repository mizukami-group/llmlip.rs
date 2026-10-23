#![allow(unsafe_code)]

#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
use std::sync::OnceLock;

use std::os::raw::{c_double, c_int};
// `c_float` is only consumed by the f32 CBLAS paths, which have no libxsmm
// provider; keep the import gated identically to avoid unused-import errors
// in libxsmm-only builds.
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
use std::os::raw::c_float;

const CBLAS_ROW_MAJOR: c_int = 101;
const CBLAS_NO_TRANS: c_int = 111;
const CBLAS_TRANS: c_int = 112;
#[cfg(feature = "parallel")]
const UMA_FUSED_WEIGHT_PARALLEL_MIN_LEN: usize = 32_768;
#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
const CBLAS_PACKED: c_int = 151;
#[cfg(all(
    feature = "orb-fixed-shape-dense-pack-a",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
const CBLAS_A_MATRIX: c_int = 161;
#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
const CBLAS_B_MATRIX: c_int = 162;

#[inline(always)]
fn strided_matrix_len(rows: usize, row_stride: usize, cols: usize) -> usize {
    if rows == 0 || cols == 0 {
        0
    } else {
        (rows - 1) * row_stride + cols
    }
}

#[inline(always)]
fn debug_assert_c_int_dims(dimensions: &[(&str, usize)]) {
    for &(name, value) in dimensions {
        debug_assert!(
            value <= c_int::MAX as usize,
            "BLAS dimension {name}={value} exceeds c_int::MAX"
        );
    }
}

fn uninit_f64_buffer(len: usize) -> Vec<std::mem::MaybeUninit<f64>> {
    let mut buffer = Vec::with_capacity(len);
    // SAFETY: `MaybeUninit<f64>` may be uninitialized. Callers must initialize
    // all elements before converting the buffer to `Vec<f64>`.
    unsafe {
        buffer.set_len(len);
    }
    buffer
}

unsafe fn assume_init_f64_vec(mut buffer: Vec<std::mem::MaybeUninit<f64>>) -> Vec<f64> {
    let len = buffer.len();
    let capacity = buffer.capacity();
    let ptr = buffer.as_mut_ptr().cast::<f64>();
    std::mem::forget(buffer);
    // SAFETY: The caller guarantees that every element has been written as an
    // initialized f64 and that `MaybeUninit<f64>` has the same layout as `f64`.
    unsafe { Vec::from_raw_parts(ptr, len, capacity) }
}

/// Writes `0.0` into every slot of an uninitialized f64 buffer.
///
/// Used as the guaranteed-init path when a beta=0 GEMM is skipped because the
/// contraction dimension K is zero. CBLAS/MKL/libxsmm DGEMM early-return
/// WITHOUT writing the output when K == 0, so without this the buffer would be
/// read uninitialized by `assume_init_f64_vec`. `C = 0 * A * B` is the
/// all-zeros matrix, so zero-filling reproduces the GEMM's mathematical result
/// while guaranteeing the buffer is fully initialized.
#[allow(dead_code)]
fn zero_fill_f64_uninit(buffer: &mut [std::mem::MaybeUninit<f64>]) {
    for slot in buffer.iter_mut() {
        slot.write(0.0);
    }
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
fn uma_blas_trace_timings_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("LLMLIP_UMA_TRACE_TIMINGS").ok().as_deref(),
            Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
        )
    })
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
fn uma_blas_trace_start() -> Option<std::time::Instant> {
    uma_blas_trace_timings_enabled().then(std::time::Instant::now)
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
fn trace_uma_so2_f32_matmul(
    kind: &str,
    rows: usize,
    k: usize,
    n: usize,
    accumulate: bool,
    path: &str,
    start: Option<std::time::Instant>,
) {
    if let Some(start) = start {
        eprintln!(
            "uma_trace stage=blas.so2_f32.{} path={} rows={} k={} n={} accumulate={} elapsed_ms={:.6}",
            kind,
            path,
            rows,
            k,
            n,
            accumulate,
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}

#[cfg(any(
    all(
        feature = "accelerate-blas",
        target_os = "macos",
        target_arch = "x86_64"
    ),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        target_arch = "x86_64"
    )
))]
fn trace_uma_f32_row_matmul(
    rows: usize,
    k: usize,
    n: usize,
    accumulate: bool,
    path: &str,
    start: Option<std::time::Instant>,
) {
    if let Some(start) = start {
        eprintln!(
            "uma_trace stage=blas.uma_f32.row_matmul path={} rows={} k={} n={} accumulate={} elapsed_ms={:.6}",
            path,
            rows,
            k,
            n,
            accumulate,
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}

// macOS Accelerate framework link.
#[cfg(all(feature = "accelerate-blas", target_os = "macos"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_dgemm(
        order: c_int,
        trans_a: c_int,
        trans_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_double,
        a: *const c_double,
        lda: c_int,
        b: *const c_double,
        ldb: c_int,
        beta: c_double,
        c: *mut c_double,
        ldc: c_int,
    );
    fn cblas_dgemv(
        order: c_int,
        trans_a: c_int,
        m: c_int,
        n: c_int,
        alpha: c_double,
        a: *const c_double,
        lda: c_int,
        x: *const c_double,
        inc_x: c_int,
        beta: c_double,
        y: *mut c_double,
        inc_y: c_int,
    );
    fn cblas_sgemv(
        order: c_int,
        trans_a: c_int,
        m: c_int,
        n: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        x: *const c_float,
        inc_x: c_int,
        beta: c_float,
        y: *mut c_float,
        inc_y: c_int,
    );
    fn cblas_sgemm(
        order: c_int,
        trans_a: c_int,
        trans_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        b: *const c_float,
        ldb: c_int,
        beta: c_float,
        c: *mut c_float,
        ldc: c_int,
    );
    fn vvexp(y: *mut c_double, x: *const c_double, n: *const c_int);
    fn vvexpf(y: *mut c_float, x: *const c_float, n: *const c_int);
}

// Linux CBLAS link. Declared whenever the `intel-mkl` or `openblas` feature is
// on (independent of `libxsmm`) so the combined backend can still call into
// CBLAS for large matrices.
#[cfg(any(
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
extern "C" {
    fn cblas_dgemm(
        order: c_int,
        trans_a: c_int,
        trans_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_double,
        a: *const c_double,
        lda: c_int,
        b: *const c_double,
        ldb: c_int,
        beta: c_double,
        c: *mut c_double,
        ldc: c_int,
    );
    fn cblas_dgemv(
        order: c_int,
        trans_a: c_int,
        m: c_int,
        n: c_int,
        alpha: c_double,
        a: *const c_double,
        lda: c_int,
        x: *const c_double,
        inc_x: c_int,
        beta: c_double,
        y: *mut c_double,
        inc_y: c_int,
    );
    fn cblas_sgemv(
        order: c_int,
        trans_a: c_int,
        m: c_int,
        n: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        x: *const c_float,
        inc_x: c_int,
        beta: c_float,
        y: *mut c_float,
        inc_y: c_int,
    );
    fn cblas_sgemm(
        order: c_int,
        trans_a: c_int,
        trans_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        b: *const c_float,
        ldb: c_int,
        beta: c_float,
        c: *mut c_float,
        ldc: c_int,
    );
}

#[cfg(all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"))]
#[allow(non_snake_case)]
extern "C" {
    fn vdExp(n: c_int, input: *const c_double, output: *mut c_double);
    fn vsExp(n: c_int, input: *const c_float, output: *mut c_float);
}

#[cfg(all(
    feature = "orb-fixed-shape-dense-pack-a",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
struct MklPackedAThreadScratch {
    ptr: Option<std::ptr::NonNull<c_double>>,
    bytes: usize,
}

#[cfg(all(
    feature = "orb-fixed-shape-dense-pack-a",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
impl MklPackedAThreadScratch {
    const fn new() -> Self {
        Self {
            ptr: None,
            bytes: 0,
        }
    }

    fn ensure(&mut self, bytes: usize) -> Option<*mut c_double> {
        if self.bytes < bytes {
            if let Some(ptr) = self.ptr.take() {
                unsafe {
                    mkl_free(ptr.as_ptr().cast::<std::ffi::c_void>());
                }
            }
            self.bytes = 0;
            let raw = unsafe { mkl_malloc(bytes, 64) };
            let ptr = std::ptr::NonNull::new(raw.cast::<c_double>())?;
            self.ptr = Some(ptr);
            self.bytes = bytes;
        }
        self.ptr.map(std::ptr::NonNull::as_ptr)
    }
}

#[cfg(all(
    feature = "orb-fixed-shape-dense-pack-a",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
impl Drop for MklPackedAThreadScratch {
    fn drop(&mut self) {
        if let Some(ptr) = self.ptr.take() {
            unsafe {
                mkl_free(ptr.as_ptr().cast::<std::ffi::c_void>());
            }
        }
    }
}

#[cfg(all(
    feature = "orb-fixed-shape-dense-pack-a",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
thread_local! {
    static MKL_PACKED_A_SCRATCH: std::cell::RefCell<MklPackedAThreadScratch> =
        const { std::cell::RefCell::new(MklPackedAThreadScratch::new()) };
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
extern "C" {
    #[link_name = "MKL_malloc"]
    fn mkl_malloc(size: usize, align: c_int) -> *mut std::ffi::c_void;
    #[link_name = "MKL_free"]
    fn mkl_free(ptr: *mut std::ffi::c_void);
    fn cblas_dgemm_pack_get_size(identifier: c_int, m: c_int, n: c_int, k: c_int) -> usize;
    fn cblas_dgemm_pack(
        order: c_int,
        identifier: c_int,
        trans: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_double,
        src: *const c_double,
        ld: c_int,
        dest: *mut c_double,
    );
    fn cblas_dgemm_compute(
        order: c_int,
        trans_a: c_int,
        trans_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        a: *const c_double,
        lda: c_int,
        b: *const c_double,
        ldb: c_int,
        beta: c_double,
        c: *mut c_double,
        ldc: c_int,
    );
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
pub(crate) struct MklPackedDgemmBMatrix {
    ptr: std::ptr::NonNull<c_double>,
    bytes: usize,
    m: c_int,
    n: c_int,
    k: c_int,
    ldb: c_int,
    #[cfg(feature = "orb-fixed-shape-dense-pack-a")]
    pack_a_bytes: usize,
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
impl std::fmt::Debug for MklPackedDgemmBMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MklPackedDgemmBMatrix")
            .field("bytes", &self.bytes)
            .field("m", &self.m)
            .field("n", &self.n)
            .field("k", &self.k)
            .field("ldb", &self.ldb)
            .field(
                "pack_a_enabled",
                &cfg!(feature = "orb-fixed-shape-dense-pack-a"),
            )
            .finish_non_exhaustive()
    }
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
// SAFETY: oneMKL packed buffers are immutable after construction in this
// wrapper, and compute calls only read the packed B matrix.
unsafe impl Send for MklPackedDgemmBMatrix {}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
// SAFETY: see `Send`; shared references only expose read-only compute access.
unsafe impl Sync for MklPackedDgemmBMatrix {}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
impl Drop for MklPackedDgemmBMatrix {
    fn drop(&mut self) {
        unsafe {
            mkl_free(self.ptr.as_ptr().cast::<std::ffi::c_void>());
        }
    }
}

#[cfg(all(
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
impl MklPackedDgemmBMatrix {
    pub(crate) fn pack_output_major_weight(
        weight: &[f64],
        m: usize,
        n: usize,
        k: usize,
        trans: bool,
    ) -> Result<Self, String> {
        let ldb = if trans { k } else { n };
        Self::pack_strided_b_matrix(weight, 0, ldb, m, n, k, trans)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pack_strided_b_matrix(
        weight: &[f64],
        column_offset: usize,
        ldb: usize,
        m: usize,
        n: usize,
        k: usize,
        trans: bool,
    ) -> Result<Self, String> {
        if column_offset > ldb {
            return Err(format!(
                "MKL packed DGEMM column offset {column_offset} exceeds ldb {ldb}"
            ));
        }
        let selected_cols = if trans { k } else { n };
        if selected_cols > ldb.saturating_sub(column_offset) {
            return Err(format!(
                "MKL packed DGEMM column offset {column_offset} width {selected_cols} exceeds ldb {ldb}"
            ));
        }
        let source_rows = if trans { n } else { k };
        let required = if source_rows == 0 || selected_cols == 0 {
            column_offset
        } else {
            column_offset + (source_rows - 1) * ldb + selected_cols
        };
        if weight.len() < required {
            return Err(format!(
                "MKL packed DGEMM weight length must be at least {required}, got {}",
                weight.len()
            ));
        }
        let m_i = c_int::try_from(m)
            .map_err(|_| format!("MKL packed DGEMM m dimension overflows c_int: {m}"))?;
        let n_i = c_int::try_from(n)
            .map_err(|_| format!("MKL packed DGEMM n dimension overflows c_int: {n}"))?;
        let k_i = c_int::try_from(k)
            .map_err(|_| format!("MKL packed DGEMM k dimension overflows c_int: {k}"))?;
        let ldb_i = c_int::try_from(ldb)
            .map_err(|_| format!("MKL packed DGEMM ldb dimension overflows c_int: {ldb}"))?;
        let bytes = unsafe { cblas_dgemm_pack_get_size(CBLAS_B_MATRIX, m_i, n_i, k_i) };
        if bytes == 0 {
            return Err("MKL packed DGEMM returned zero pack size".to_owned());
        }
        #[cfg(feature = "orb-fixed-shape-dense-pack-a")]
        let pack_a_bytes = unsafe { cblas_dgemm_pack_get_size(CBLAS_A_MATRIX, m_i, n_i, k_i) };
        #[cfg(feature = "orb-fixed-shape-dense-pack-a")]
        if pack_a_bytes == 0 {
            return Err("MKL packed-A DGEMM returned zero pack size".to_owned());
        }
        let raw = unsafe { mkl_malloc(bytes, 64) };
        let ptr = std::ptr::NonNull::new(raw.cast::<c_double>())
            .ok_or_else(|| format!("MKL packed DGEMM allocation failed for {bytes} bytes"))?;
        let trans_b = if trans { CBLAS_TRANS } else { CBLAS_NO_TRANS };
        unsafe {
            cblas_dgemm_pack(
                CBLAS_ROW_MAJOR,
                CBLAS_B_MATRIX,
                trans_b,
                m_i,
                n_i,
                k_i,
                1.0,
                weight.as_ptr().add(column_offset),
                ldb_i,
                ptr.as_ptr(),
            );
        }
        Ok(Self {
            ptr,
            bytes,
            m: m_i,
            n: n_i,
            k: k_i,
            ldb: ldb_i,
            #[cfg(feature = "orb-fixed-shape-dense-pack-a")]
            pack_a_bytes,
        })
    }

    #[cfg(feature = "orb-fixed-shape-dense-pack-a")]
    fn compute_packed_a_no_trans_into_raw(&self, input: &[f64], output: *mut f64, beta: f64) {
        let packed_a = MKL_PACKED_A_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.ensure(self.pack_a_bytes)
        });
        let Some(packed_a) = packed_a else {
            self.compute_unpacked_a_no_trans_into_raw(input, output, beta);
            return;
        };
        unsafe {
            cblas_dgemm_pack(
                CBLAS_ROW_MAJOR,
                CBLAS_A_MATRIX,
                CBLAS_NO_TRANS,
                self.m,
                self.n,
                self.k,
                1.0,
                input.as_ptr(),
                self.k,
                packed_a,
            );
            cblas_dgemm_compute(
                CBLAS_ROW_MAJOR,
                CBLAS_PACKED,
                CBLAS_PACKED,
                self.m,
                self.n,
                self.k,
                packed_a,
                self.k,
                self.ptr.as_ptr(),
                self.ldb,
                beta,
                output,
                self.n,
            );
        }
    }

    fn compute_unpacked_a_no_trans_into_raw(&self, input: &[f64], output: *mut f64, beta: f64) {
        unsafe {
            cblas_dgemm_compute(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_PACKED,
                self.m,
                self.n,
                self.k,
                input.as_ptr(),
                self.k,
                self.ptr.as_ptr(),
                self.ldb,
                beta,
                output,
                self.n,
            );
        }
    }

    pub(crate) fn compute_no_trans_a_alloc(&self, input: &[f64]) -> Vec<f64> {
        let rows = self.m as usize;
        let input_dim = self.k as usize;
        let output_dim = self.n as usize;
        debug_assert!(input.len() >= rows * input_dim);
        let mut output = uninit_f64_buffer(rows * output_dim);
        if rows != 0 && output_dim != 0 && input_dim != 0 {
            #[cfg(feature = "orb-fixed-shape-dense-pack-a")]
            self.compute_packed_a_no_trans_into_raw(input, output.as_mut_ptr().cast::<f64>(), 0.0);
            #[cfg(not(feature = "orb-fixed-shape-dense-pack-a"))]
            self.compute_unpacked_a_no_trans_into_raw(
                input,
                output.as_mut_ptr().cast::<f64>(),
                0.0,
            );
        } else if rows != 0 && output_dim != 0 {
            // K (input_dim) == 0: DGEMM early-returns without writing the
            // beta=0 output. Zero-fill so the buffer is fully initialized;
            // C = 0 * A * B is the all-zeros matrix.
            zero_fill_f64_uninit(&mut output);
        }
        // SAFETY: every element is now initialized: the beta=0 DGEMM writes the
        // whole buffer when rows, output_dim, and input_dim (K) are all
        // non-zero; the K==0 branch zero-fills it; and a zero-length buffer
        // (rows==0 || output_dim==0) is trivially initialized.
        unsafe { assume_init_f64_vec(output) }
    }

    pub(crate) fn compute_no_trans_a_into(&self, input: &[f64], output: &mut [f64], beta: f64) {
        let rows = self.m as usize;
        let input_dim = self.k as usize;
        let output_dim = self.n as usize;
        debug_assert!(input.len() >= rows * input_dim);
        debug_assert!(output.len() >= rows * output_dim);
        if rows != 0 && output_dim != 0 {
            #[cfg(feature = "orb-fixed-shape-dense-pack-a")]
            self.compute_packed_a_no_trans_into_raw(input, output.as_mut_ptr(), beta);
            #[cfg(not(feature = "orb-fixed-shape-dense-pack-a"))]
            self.compute_unpacked_a_no_trans_into_raw(input, output.as_mut_ptr(), beta);
        }
    }
}

// libxsmm link (LP64, JIT-compiled small GEMM kernels, linked via build.rs).
// Uses Fortran-style LAPACK DGEMM convention (column-major, pass-by-pointer).
#[cfg(all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64"))]
extern "C" {
    fn libxsmm_dgemm(
        transa: *const std::os::raw::c_char,
        transb: *const std::os::raw::c_char,
        m: *const c_int,
        n: *const c_int,
        k: *const c_int,
        alpha: *const c_double,
        a: *const c_double,
        lda: *const c_int,
        b: *const c_double,
        ldb: *const c_int,
        beta: *const c_double,
        c: *mut c_double,
        ldc: *const c_int,
    );
}

// Row-major CBLAS-style DGEMM implemented via libxsmm's Fortran-style API.
//
// Row-major C = alpha * op(A) * op(B) + beta * C is equivalent to
// column-major C' = alpha * op(B)' * op(A)' + beta * C' where X' means
// "X viewed in column-major". For row-major data the column-major view is
// the transpose, so we swap A/B, M/N, and the transpose flags.
//
// libxsmm JIT kernels only handle alpha=1, beta∈{0,1}. When the caller
// supplies other values, we decompose the operation:
//   beta=0:  compute C = op(A)*op(B) then scale C by alpha.
//   beta≠0:  compute temp = op(A)*op(B) into thread-local scratch,
//            then combine C[i] = alpha * temp[i] + beta * C[i].
#[cfg(all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn cblas_dgemm_via_xsmm(
    _order: c_int,
    trans_a: c_int,
    trans_b: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: c_double,
    a: *const c_double,
    lda: c_int,
    b: *const c_double,
    ldb: c_int,
    beta: c_double,
    c: *mut c_double,
    ldc: c_int,
) {
    let fort_trans_a: std::os::raw::c_char = if trans_b == 111 { b'N' as _ } else { b'T' as _ };
    let fort_trans_b: std::os::raw::c_char = if trans_a == 111 { b'N' as _ } else { b'T' as _ };
    let rows = m as usize;
    let cols = n as usize;
    let stride = ldc as usize;

    if beta == 0.0 {
        // C = alpha * op(A) * op(B).
        // Step 1: C = op(A)*op(B) via libxsmm (alpha=1, beta=0).
        let one: c_double = 1.0;
        let zero: c_double = 0.0;
        unsafe {
            libxsmm_dgemm(
                &fort_trans_a,
                &fort_trans_b,
                &n,
                &m,
                &k,
                &one,
                b,
                &ldb,
                a,
                &lda,
                &zero,
                c,
                &ldc,
            );
        }
        // Step 2: scale C by alpha if needed.
        if alpha != 1.0 {
            let c_slice = unsafe {
                std::slice::from_raw_parts_mut(c, strided_matrix_len(rows, stride, cols))
            };
            for row in 0..rows {
                let base = row * stride;
                for col in 0..cols {
                    c_slice[base + col] *= alpha;
                }
            }
        }
    } else if alpha == 1.0 && beta == 1.0 {
        // C += op(A) * op(B). Directly supported by libxsmm.
        let one: c_double = 1.0;
        unsafe {
            libxsmm_dgemm(
                &fort_trans_a,
                &fort_trans_b,
                &n,
                &m,
                &k,
                &one,
                b,
                &ldb,
                a,
                &lda,
                &one,
                c,
                &ldc,
            );
        }
    } else {
        // General: C = alpha * op(A)*op(B) + beta * C.
        // Use a thread-local scratch buffer to avoid per-call allocation.
        use std::cell::RefCell;
        thread_local! {
            static SCRATCH: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
        }
        let needed = rows * cols;
        SCRATCH.with(|cell| {
            let mut buf = cell.borrow_mut();
            if buf.len() < needed {
                buf.resize(needed, 0.0);
            }
            let one: c_double = 1.0;
            let zero: c_double = 0.0;
            let cols_i = n;
            unsafe {
                libxsmm_dgemm(
                    &fort_trans_a,
                    &fort_trans_b,
                    &cols_i,
                    &m,
                    &k,
                    &one,
                    b,
                    &ldb,
                    a,
                    &lda,
                    &zero,
                    buf.as_mut_ptr(),
                    &cols_i,
                );
            }
            let c_slice = unsafe {
                std::slice::from_raw_parts_mut(c, strided_matrix_len(rows, stride, cols))
            };
            for row in 0..rows {
                let c_base = row * stride;
                let t_base = row * cols;
                for col in 0..cols {
                    c_slice[c_base + col] =
                        alpha * buf[t_base + col] + beta * c_slice[c_base + col];
                }
            }
        });
    }
}

// Row count above which the combined libxsmm+CBLAS backend prefers CBLAS over
// libxsmm. The crossover from MKL micro-benchmarks (M×128×128 DGEMM,
// AMD EPYC 7763): libxsmm wins at M <= 12, MKL wins at M >= 32. We pick 16
// as the cleanest dispatch boundary.
#[cfg(all(
    feature = "libxsmm",
    any(feature = "intel-mkl", feature = "openblas"),
    target_os = "linux",
    target_arch = "x86_64"
))]
const LIBXSMM_VS_CBLAS_THRESHOLD: c_int = 16;

// Dispatch helper: routes the call to the appropriate DGEMM backend.
//
// Active-feature matrix (Linux x86_64 unless otherwise noted):
//   - libxsmm only             → libxsmm (`cblas_dgemm_via_xsmm`)
//   - intel-mkl/openblas only  → CBLAS (`cblas_dgemm`)
//   - accelerate-blas (macOS)  → Accelerate (`cblas_dgemm`)
//   - libxsmm + CBLAS          → libxsmm for M < threshold, CBLAS otherwise
//
// The three `cfg` arms below are mutually exclusive: gated on whether
// libxsmm is active, and within that, whether MKL is also active.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[cfg(all(
    feature = "libxsmm",
    any(feature = "intel-mkl", feature = "openblas"),
    target_os = "linux",
    target_arch = "x86_64"
))]
fn dgemm_dispatch(
    order: c_int,
    trans_a: c_int,
    trans_b: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: c_double,
    a: *const c_double,
    lda: c_int,
    b: *const c_double,
    ldb: c_int,
    beta: c_double,
    c: *mut c_double,
    ldc: c_int,
) {
    if m < LIBXSMM_VS_CBLAS_THRESHOLD {
        cblas_dgemm_via_xsmm(
            order, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    } else {
        unsafe {
            cblas_dgemm(
                order, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
            );
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[cfg(all(
    feature = "libxsmm",
    not(any(feature = "intel-mkl", feature = "openblas")),
    target_os = "linux",
    target_arch = "x86_64"
))]
fn dgemm_dispatch(
    order: c_int,
    trans_a: c_int,
    trans_b: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: c_double,
    a: *const c_double,
    lda: c_int,
    b: *const c_double,
    ldb: c_int,
    beta: c_double,
    c: *mut c_double,
    ldc: c_int,
) {
    cblas_dgemm_via_xsmm(
        order, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    );
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[cfg(not(all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")))]
fn dgemm_dispatch(
    order: c_int,
    trans_a: c_int,
    trans_b: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: c_double,
    a: *const c_double,
    lda: c_int,
    b: *const c_double,
    ldb: c_int,
    beta: c_double,
    c: *mut c_double,
    ldc: c_int,
) {
    unsafe {
        cblas_dgemm(
            order, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        );
    }
}

#[inline]
fn blas_dim(value: usize) -> c_int {
    assert!(
        value <= c_int::MAX as usize,
        "BLAS dimension exceeds c_int range"
    );
    value as c_int
}

fn apply_beta_f64(value: &mut f64, beta: f64) {
    if beta == 0.0 {
        *value = 0.0;
    } else if beta != 1.0 {
        *value *= beta;
    }
}

fn apply_beta_f32(value: &mut f32, beta: f32) {
    if beta == 0.0 {
        *value = 0.0;
    } else if beta != 1.0 {
        *value *= beta;
    }
}

fn apply_beta_contiguous_f64(values: &mut [f64], beta: f64) {
    for value in values {
        apply_beta_f64(value, beta);
    }
}

fn apply_beta_contiguous_f32(values: &mut [f32], beta: f32) {
    for value in values {
        apply_beta_f32(value, beta);
    }
}

fn apply_beta_strided_f64(values: &mut [f64], rows: usize, cols: usize, stride: usize, beta: f64) {
    if rows == 0 || cols == 0 {
        return;
    }
    for row in values.chunks_mut(stride).take(rows) {
        for value in &mut row[..cols] {
            apply_beta_f64(value, beta);
        }
    }
}

fn apply_beta_strided_f32(values: &mut [f32], rows: usize, cols: usize, stride: usize, beta: f32) {
    if rows == 0 || cols == 0 {
        return;
    }
    for row in values.chunks_mut(stride).take(rows) {
        for value in &mut row[..cols] {
            apply_beta_f32(value, beta);
        }
    }
}

pub fn dense_forward(
    input: &[f64],
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out: &mut [f64],
    alpha: f64,
) {
    debug_assert!(in_dim > 0);
    debug_assert_eq!(weight.len() % in_dim, 0);
    let out_dim = weight.len() / in_dim;
    debug_assert!(input.len() >= rows * in_dim);
    debug_assert!(out.len() >= rows * out_dim);
    debug_assert_c_int_dims(&[("rows", rows), ("in_dim", in_dim), ("out_dim", out_dim)]);
    if rows == 0 || out_dim == 0 {
        return;
    }
    dgemm_dispatch(
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        CBLAS_NO_TRANS,
        blas_dim(rows),
        blas_dim(out_dim),
        blas_dim(in_dim),
        alpha,
        input.as_ptr(),
        blas_dim(in_dim),
        weight.as_ptr(),
        blas_dim(out_dim),
        0.0,
        out.as_mut_ptr(),
        blas_dim(out_dim),
    );
}

/// Compute `out = coefficients[1, rows] @ values[rows, cols]`.
///
/// `values` is row-major with one contiguous row per expert/source. This is the
/// layout used by UMA MoLE expert weights, so the helper avoids transposing the
/// large expert matrix before mixing.
#[allow(dead_code)]
pub(crate) fn weighted_sum_rows_f64(
    rows: usize,
    cols: usize,
    values: &[f64],
    coefficients: &[f64],
    out: &mut [f64],
) {
    debug_assert!(values.len() >= rows * cols);
    debug_assert!(coefficients.len() >= rows);
    debug_assert!(out.len() >= cols);
    if rows == 0 || cols == 0 {
        return;
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    unsafe {
        cblas_dgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            blas_dim(rows),
            blas_dim(cols),
            1.0,
            values.as_ptr(),
            blas_dim(cols),
            coefficients.as_ptr(),
            1,
            0.0,
            out.as_mut_ptr(),
            1,
        );
    }
    #[cfg(all(
        feature = "libxsmm",
        not(any(
            feature = "accelerate-blas",
            feature = "intel-mkl",
            feature = "openblas"
        )),
        target_os = "linux",
        target_arch = "x86_64"
    ))]
    dgemm_dispatch(
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        CBLAS_NO_TRANS,
        1,
        blas_dim(cols),
        blas_dim(rows),
        1.0,
        coefficients.as_ptr(),
        blas_dim(rows),
        values.as_ptr(),
        blas_dim(cols),
        0.0,
        out.as_mut_ptr(),
        blas_dim(cols),
    );
}

#[cfg(all(feature = "accelerate-blas", target_os = "macos"))]
pub(crate) fn exp_f64(input: &[f64], output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert!(input.len() <= c_int::MAX as usize);
    if input.is_empty() {
        return;
    }
    let n = blas_dim(input.len());
    unsafe {
        // Invariant: input and output are valid contiguous f64 buffers of n
        // elements. vForce vvexp writes one output element per input element.
        vvexp(output.as_mut_ptr(), input.as_ptr(), &n);
    }
}

/// Compute `out = coefficients[1, rows] @ values[rows, cols]` for f32 rows.
///
/// `values` is row-major with one contiguous row per expert/source. This mirrors
/// [`weighted_sum_rows_f64`] for UMA FP32 mixed-weight construction.
#[allow(dead_code)]
pub(crate) fn weighted_sum_rows_f32(
    rows: usize,
    cols: usize,
    values: &[f32],
    coefficients: &[f32],
    out: &mut [f32],
) {
    debug_assert!(values.len() >= rows * cols);
    debug_assert!(coefficients.len() >= rows);
    debug_assert!(out.len() >= cols);
    if rows == 0 || cols == 0 {
        return;
    }
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    unsafe {
        cblas_sgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            blas_dim(rows),
            blas_dim(cols),
            1.0,
            values.as_ptr(),
            blas_dim(cols),
            coefficients.as_ptr(),
            1,
            0.0,
            out.as_mut_ptr(),
            1,
        );
    }
    #[cfg(not(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    {
        for out_value in out.iter_mut().take(cols) {
            *out_value = 0.0;
        }
        for (row_values, &coefficient) in values
            .chunks_exact(cols)
            .zip(coefficients.iter())
            .take(rows)
        {
            for (out_value, &value) in out.iter_mut().take(cols).zip(row_values.iter()) {
                *out_value += coefficient * value;
            }
        }
    }
}

#[cfg(all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn exp_f64(input: &[f64], output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert!(input.len() <= c_int::MAX as usize);
    if input.is_empty() {
        return;
    }
    let n = blas_dim(input.len());
    unsafe {
        // Invariant: input and output are valid contiguous f64 buffers of n
        // elements. MKL vdExp writes one output element per input element.
        vdExp(n, input.as_ptr(), output.as_mut_ptr());
    }
}

#[cfg(all(feature = "openblas", target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn exp_f64(input: &[f64], output: &mut [f64]) {
    debug_assert_eq!(input.len(), output.len());
    for (dst, &src) in output.iter_mut().zip(input) {
        *dst = src.exp();
    }
}

#[cfg(all(feature = "accelerate-blas", target_os = "macos"))]
pub(crate) fn exp_f32(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert!(input.len() <= c_int::MAX as usize);
    if input.is_empty() {
        return;
    }
    let n = blas_dim(input.len());
    unsafe {
        // Invariant: input and output are valid contiguous f32 buffers of n
        // elements. vForce vvexpf writes one output element per input element.
        vvexpf(output.as_mut_ptr(), input.as_ptr(), &n);
    }
}

#[cfg(all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn exp_f32(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert!(input.len() <= c_int::MAX as usize);
    if input.is_empty() {
        return;
    }
    let n = blas_dim(input.len());
    unsafe {
        // Invariant: input and output are valid contiguous f32 buffers of n
        // elements. MKL vsExp writes one output element per input element.
        vsExp(n, input.as_ptr(), output.as_mut_ptr());
    }
}

#[cfg(all(feature = "openblas", target_os = "linux", target_arch = "x86_64"))]
pub(crate) fn exp_f32(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    for (dst, &src) in output.iter_mut().zip(input) {
        *dst = src.exp();
    }
}

#[allow(clippy::too_many_arguments)]
pub fn dense_forward_strided(
    input: &[f64],
    rows: usize,
    in_dim: usize,
    input_row_stride: usize,
    weight: &[f64],
    out: &mut [f64],
    out_dim: usize,
    out_row_stride: usize,
    alpha: f64,
    beta: f64,
) {
    debug_assert!(input_row_stride >= in_dim);
    debug_assert!(out_row_stride >= out_dim);
    debug_assert!(weight.len() >= in_dim * out_dim);
    debug_assert!(input.len() >= strided_matrix_len(rows, input_row_stride, in_dim));
    debug_assert!(out.len() >= strided_matrix_len(rows, out_row_stride, out_dim));
    debug_assert_c_int_dims(&[
        ("rows", rows),
        ("in_dim", in_dim),
        ("out_dim", out_dim),
        ("input_row_stride", input_row_stride),
        ("out_row_stride", out_row_stride),
    ]);
    if rows == 0 || out_dim == 0 {
        return;
    }
    if in_dim == 0 {
        apply_beta_strided_f64(out, rows, out_dim, out_row_stride, beta);
        return;
    }
    dgemm_dispatch(
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        CBLAS_NO_TRANS,
        blas_dim(rows),
        blas_dim(out_dim),
        blas_dim(in_dim),
        alpha,
        input.as_ptr(),
        blas_dim(input_row_stride),
        weight.as_ptr(),
        blas_dim(out_dim),
        beta,
        out.as_mut_ptr(),
        blas_dim(out_row_stride),
    );
}

pub fn dense_vjp_input(
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    bar_out: &[f64],
    bar_input: &mut [f64],
    alpha: f64,
) {
    dense_vjp_input_with_beta(rows, in_dim, weight, bar_out, bar_input, alpha, 1.0);
}

pub fn dense_vjp_input_with_beta(
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    bar_out: &[f64],
    bar_input: &mut [f64],
    alpha: f64,
    beta: f64,
) {
    if rows == 0 || in_dim == 0 {
        return;
    }
    debug_assert_eq!(weight.len() % in_dim, 0);
    let out_dim = weight.len() / in_dim;
    debug_assert!(bar_out.len() >= rows * out_dim);
    debug_assert!(bar_input.len() >= rows * in_dim);
    debug_assert_c_int_dims(&[("rows", rows), ("in_dim", in_dim), ("out_dim", out_dim)]);
    if out_dim == 0 {
        apply_beta_contiguous_f64(&mut bar_input[..rows * in_dim], beta);
        return;
    }
    dgemm_dispatch(
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        CBLAS_TRANS,
        blas_dim(rows),
        blas_dim(in_dim),
        blas_dim(out_dim),
        alpha,
        bar_out.as_ptr(),
        blas_dim(out_dim),
        weight.as_ptr(),
        blas_dim(out_dim),
        beta,
        bar_input.as_mut_ptr(),
        blas_dim(in_dim),
    );
}

#[allow(clippy::too_many_arguments)]
pub fn dense_vjp_input_strided(
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out_dim: usize,
    bar_out: &[f64],
    bar_out_row_stride: usize,
    bar_input: &mut [f64],
    bar_input_row_stride: usize,
    alpha: f64,
    beta: f64,
) {
    debug_assert!(bar_out_row_stride >= out_dim);
    debug_assert!(bar_input_row_stride >= in_dim);
    debug_assert!(weight.len() >= in_dim * out_dim);
    debug_assert!(bar_out.len() >= strided_matrix_len(rows, bar_out_row_stride, out_dim));
    debug_assert!(bar_input.len() >= strided_matrix_len(rows, bar_input_row_stride, in_dim));
    debug_assert_c_int_dims(&[
        ("rows", rows),
        ("in_dim", in_dim),
        ("out_dim", out_dim),
        ("bar_out_row_stride", bar_out_row_stride),
        ("bar_input_row_stride", bar_input_row_stride),
    ]);
    if rows == 0 || in_dim == 0 {
        return;
    }
    if out_dim == 0 {
        apply_beta_strided_f64(bar_input, rows, in_dim, bar_input_row_stride, beta);
        return;
    }
    dgemm_dispatch(
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        CBLAS_TRANS,
        blas_dim(rows),
        blas_dim(in_dim),
        blas_dim(out_dim),
        alpha,
        bar_out.as_ptr(),
        blas_dim(bar_out_row_stride),
        weight.as_ptr(),
        blas_dim(out_dim),
        beta,
        bar_input.as_mut_ptr(),
        blas_dim(bar_input_row_stride),
    );
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub fn dense_forward_f32(
    input: &[f32],
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    out: &mut [f32],
    alpha: f32,
) {
    debug_assert!(in_dim > 0);
    debug_assert_eq!(weight.len() % in_dim, 0);
    let out_dim = weight.len() / in_dim;
    debug_assert!(input.len() >= rows * in_dim);
    debug_assert!(out.len() >= rows * out_dim);
    debug_assert_c_int_dims(&[("rows", rows), ("in_dim", in_dim), ("out_dim", out_dim)]);
    if rows == 0 || out_dim == 0 {
        return;
    }
    unsafe {
        // Invariant: all matrices are row-major contiguous f32 buffers with
        // leading dimensions matching the logical row width.
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            blas_dim(rows),
            blas_dim(out_dim),
            blas_dim(in_dim),
            alpha,
            input.as_ptr(),
            blas_dim(in_dim),
            weight.as_ptr(),
            blas_dim(out_dim),
            0.0,
            out.as_mut_ptr(),
            blas_dim(out_dim),
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub fn dense_forward_strided_f32(
    input: &[f32],
    rows: usize,
    in_dim: usize,
    input_row_stride: usize,
    weight: &[f32],
    out: &mut [f32],
    out_dim: usize,
    out_row_stride: usize,
    alpha: f32,
    beta: f32,
) {
    debug_assert!(input_row_stride >= in_dim);
    debug_assert!(out_row_stride >= out_dim);
    debug_assert!(weight.len() >= in_dim * out_dim);
    debug_assert!(input.len() >= strided_matrix_len(rows, input_row_stride, in_dim));
    debug_assert!(out.len() >= strided_matrix_len(rows, out_row_stride, out_dim));
    debug_assert_c_int_dims(&[
        ("rows", rows),
        ("in_dim", in_dim),
        ("out_dim", out_dim),
        ("input_row_stride", input_row_stride),
        ("out_row_stride", out_row_stride),
    ]);
    if rows == 0 || out_dim == 0 {
        return;
    }
    if in_dim == 0 {
        apply_beta_strided_f32(out, rows, out_dim, out_row_stride, beta);
        return;
    }
    unsafe {
        // Invariant: all matrices are row-major f32 buffers. Strided
        // inputs/outputs must provide a leading dimension at least as wide
        // as the logical row.
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            blas_dim(rows),
            blas_dim(out_dim),
            blas_dim(in_dim),
            alpha,
            input.as_ptr(),
            blas_dim(input_row_stride),
            weight.as_ptr(),
            blas_dim(out_dim),
            beta,
            out.as_mut_ptr(),
            blas_dim(out_row_stride),
        );
    }
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub fn dense_vjp_input_f32(
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    bar_out: &[f32],
    bar_input: &mut [f32],
    alpha: f32,
) {
    dense_vjp_input_f32_with_beta(rows, in_dim, weight, bar_out, bar_input, alpha, 1.0);
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub fn dense_vjp_input_f32_with_beta(
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    bar_out: &[f32],
    bar_input: &mut [f32],
    alpha: f32,
    beta: f32,
) {
    if rows == 0 || in_dim == 0 {
        return;
    }
    debug_assert_eq!(weight.len() % in_dim, 0);
    let out_dim = weight.len() / in_dim;
    debug_assert!(bar_out.len() >= rows * out_dim);
    debug_assert!(bar_input.len() >= rows * in_dim);
    debug_assert_c_int_dims(&[("rows", rows), ("in_dim", in_dim), ("out_dim", out_dim)]);
    if out_dim == 0 {
        apply_beta_contiguous_f32(&mut bar_input[..rows * in_dim], beta);
        return;
    }
    unsafe {
        // Invariant: bar_out is row-major [rows, out_dim], weight is
        // row-major [in_dim, out_dim], and bar_input is row-major
        // [rows, in_dim].
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            blas_dim(rows),
            blas_dim(in_dim),
            blas_dim(out_dim),
            alpha,
            bar_out.as_ptr(),
            blas_dim(out_dim),
            weight.as_ptr(),
            blas_dim(out_dim),
            beta,
            bar_input.as_mut_ptr(),
            blas_dim(in_dim),
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub fn dense_vjp_input_strided_f32(
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    out_dim: usize,
    bar_out: &[f32],
    bar_out_row_stride: usize,
    bar_input: &mut [f32],
    bar_input_row_stride: usize,
    alpha: f32,
    beta: f32,
) {
    debug_assert!(bar_out_row_stride >= out_dim);
    debug_assert!(bar_input_row_stride >= in_dim);
    debug_assert!(weight.len() >= in_dim * out_dim);
    debug_assert!(bar_out.len() >= strided_matrix_len(rows, bar_out_row_stride, out_dim));
    debug_assert!(bar_input.len() >= strided_matrix_len(rows, bar_input_row_stride, in_dim));
    debug_assert_c_int_dims(&[
        ("rows", rows),
        ("in_dim", in_dim),
        ("out_dim", out_dim),
        ("bar_out_row_stride", bar_out_row_stride),
        ("bar_input_row_stride", bar_input_row_stride),
    ]);
    if rows == 0 || in_dim == 0 {
        return;
    }
    if out_dim == 0 {
        apply_beta_strided_f32(bar_input, rows, in_dim, bar_input_row_stride, beta);
        return;
    }
    unsafe {
        // Invariant: strided row-major gradient buffers provide valid
        // leading dimensions for [rows, out_dim] and [rows, in_dim].
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            blas_dim(rows),
            blas_dim(in_dim),
            blas_dim(out_dim),
            alpha,
            bar_out.as_ptr(),
            blas_dim(bar_out_row_stride),
            weight.as_ptr(),
            blas_dim(out_dim),
            beta,
            bar_input.as_mut_ptr(),
            blas_dim(bar_input_row_stride),
        );
    }
}
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn dense_forward_output_major_weight(
    input: &[f64],
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out: &mut [f64],
    out_dim: usize,
    alpha: f64,
    beta: f64,
) {
    debug_assert!(input.len() >= rows * in_dim);
    debug_assert!(weight.len() >= out_dim * in_dim);
    debug_assert!(out.len() >= rows * out_dim);
    if rows == 0 || out_dim == 0 {
        return;
    }
    if in_dim == 0 {
        apply_beta_contiguous_f64(&mut out[..rows * out_dim], beta);
        return;
    }
    // Invariant: input is [rows, in_dim], weight is output-major
    // [out_dim, in_dim], and out is [rows, out_dim], all row-major.
    #[cfg(feature = "small-gemm-bypass")]
    if rows < SMALL_GEMM_BYPASS_MAX_ROWS {
        small_gemm_output_major_weight(input, rows, in_dim, weight, out, out_dim, alpha, beta);
        return;
    }
    dgemm_dispatch(
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        CBLAS_TRANS,
        blas_dim(rows),
        blas_dim(out_dim),
        blas_dim(in_dim),
        alpha,
        input.as_ptr(),
        blas_dim(in_dim),
        weight.as_ptr(),
        blas_dim(in_dim),
        beta,
        out.as_mut_ptr(),
        blas_dim(out_dim),
    );
}

/// Max `rows` (M = node/atom count) for which `dense_forward_output_major_weight`
/// bypasses BLAS under `feature = "small-gemm-bypass"`. For a single small
/// molecule M is the atom count; below this the BLAS dgemm dispatch + packing
/// overhead dominates the actual FLOPs, so a vectorizable pure-Rust dot product
/// is faster. Tunable; start generous since the bypass is opt-in.
#[cfg(feature = "small-gemm-bypass")]
const SMALL_GEMM_BYPASS_MAX_ROWS: usize = 64;

/// Pure-Rust `out = alpha * input @ weightᵀ + beta * out` for output-major
/// weight `[out_dim, in_dim]`. Both `input[r]` and `weight[o]` are contiguous
/// over `in_dim`, so the dot product uses 4 independent accumulators that the
/// compiler vectorizes (and that AVX2/FMA fills under `target-cpu`). Last-ULP
/// vs the BLAS kernel (4-lane partial sums reassociate the reduction); validated
/// within the pinned fixture tolerances.
#[cfg(feature = "small-gemm-bypass")]
fn small_gemm_output_major_weight(
    input: &[f64],
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out: &mut [f64],
    out_dim: usize,
    alpha: f64,
    beta: f64,
) {
    for r in 0..rows {
        let in_row = &input[r * in_dim..r * in_dim + in_dim];
        let out_row = &mut out[r * out_dim..r * out_dim + out_dim];
        for (o, out_slot) in out_row.iter_mut().enumerate() {
            let w_row = &weight[o * in_dim..o * in_dim + in_dim];
            let mut acc = [0.0_f64; 4];
            let chunks = in_dim / 4;
            for c in 0..chunks {
                let base = c * 4;
                acc[0] += in_row[base] * w_row[base];
                acc[1] += in_row[base + 1] * w_row[base + 1];
                acc[2] += in_row[base + 2] * w_row[base + 2];
                acc[3] += in_row[base + 3] * w_row[base + 3];
            }
            let mut dot = (acc[0] + acc[1]) + (acc[2] + acc[3]);
            for i in chunks * 4..in_dim {
                dot += in_row[i] * w_row[i];
            }
            *out_slot = alpha * dot + beta * *out_slot;
        }
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
fn row_parallel_gemm_min_ops() -> usize {
    static MIN_OPS: OnceLock<usize> = OnceLock::new();
    *MIN_OPS.get_or_init(|| {
        std::env::var("LLMLIP_ROW_PARALLEL_GEMM_MIN_OPS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1_000_000)
    })
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
fn row_parallel_gemm_block_rows(rows: usize) -> Option<usize> {
    let threads = rayon::current_num_threads();
    if rows <= 1 || threads <= 1 {
        return None;
    }
    let external_blas_threads = [
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]
    .into_iter()
    .filter_map(|name| std::env::var(name).ok())
    .filter_map(|value| value.parse::<usize>().ok())
    .max()
    .unwrap_or(1);
    if external_blas_threads > 1 {
        return None;
    }
    let target_chunks = std::env::var("LLMLIP_ROW_PARALLEL_GEMM_CHUNKS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(threads);
    Some(rows.div_ceil(target_chunks).max(1))
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
unsafe fn row_parallel_gemm_f64(
    rows: usize,
    cols: usize,
    depth: usize,
    dst: *mut f64,
    dst_row_stride: isize,
    read_dst: bool,
    dst_scale: f64,
    lhs: *const f64,
    lhs_row_stride: isize,
    rhs: *const f64,
    rhs_col_stride: isize,
    rhs_row_stride: isize,
    alpha: f64,
) {
    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        ),
        all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
    ))]
    {
        let rhs_no_trans = rhs_col_stride == 1 && rhs_row_stride == cols as isize;
        let rhs_trans = rhs_col_stride == depth as isize && rhs_row_stride == 1;
        if rhs_no_trans || rhs_trans {
            dgemm_dispatch(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                if rhs_no_trans {
                    CBLAS_NO_TRANS
                } else {
                    CBLAS_TRANS
                },
                rows as c_int,
                cols as c_int,
                depth as c_int,
                alpha,
                lhs,
                lhs_row_stride as c_int,
                rhs,
                if rhs_no_trans {
                    cols as c_int
                } else {
                    depth as c_int
                },
                if read_dst { dst_scale } else { 0.0 },
                dst,
                dst_row_stride as c_int,
            );
            return;
        }
    }

    gemm::gemm::<f64>(
        rows,
        cols,
        depth,
        dst,
        1,
        dst_row_stride,
        read_dst,
        lhs,
        1,
        lhs_row_stride,
        rhs,
        rhs_col_stride,
        rhs_row_stride,
        dst_scale,
        alpha,
        false,
        false,
        false,
        gemm::Parallelism::None,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dense_forward_output_major_weight_row_parallel_f64(
    input: &[f64],
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out: &mut [f64],
    out_dim: usize,
    alpha: f64,
    beta: f64,
) {
    debug_assert!(input.len() >= rows * in_dim);
    debug_assert!(weight.len() >= out_dim * in_dim);
    debug_assert!(out.len() >= rows * out_dim);
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    if rows.saturating_mul(in_dim).saturating_mul(out_dim) >= row_parallel_gemm_min_ops() {
        if let Some(row_block) = row_parallel_gemm_block_rows(rows) {
            out.par_chunks_mut(row_block * out_dim)
                .enumerate()
                .for_each(|(block_index, out_block)| {
                    let row_start = block_index * row_block;
                    let block_rows = (rows - row_start).min(out_block.len() / out_dim);
                    unsafe {
                        // SAFETY: Each Rayon chunk owns a disjoint row block
                        // of `out`; matching input rows are in-bounds and
                        // weight is output-major [out_dim, in_dim].
                        row_parallel_gemm_f64(
                            block_rows,
                            out_dim,
                            in_dim,
                            out_block.as_mut_ptr(),
                            out_dim as isize,
                            beta != 0.0,
                            beta,
                            input.as_ptr().add(row_start * in_dim),
                            in_dim as isize,
                            weight.as_ptr(),
                            in_dim as isize,
                            1,
                            alpha,
                        );
                    }
                });
            return;
        }
    }
    dense_forward_output_major_weight(input, rows, in_dim, weight, out, out_dim, alpha, beta);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dense_vjp_input_output_major_weight_row_parallel_f64(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    weight: &[f64],
    bar_out: &[f64],
    bar_input: &mut [f64],
    alpha: f64,
    beta: f64,
) {
    debug_assert!(weight.len() >= in_dim * out_dim);
    debug_assert!(bar_out.len() >= rows * out_dim);
    debug_assert!(bar_input.len() >= rows * in_dim);
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    if rows.saturating_mul(in_dim).saturating_mul(out_dim) >= row_parallel_gemm_min_ops() {
        if let Some(row_block) = row_parallel_gemm_block_rows(rows) {
            bar_input
                .par_chunks_mut(row_block * in_dim)
                .enumerate()
                .for_each(|(block_index, bar_input_block)| {
                    let row_start = block_index * row_block;
                    let block_rows = (rows - row_start).min(bar_input_block.len() / in_dim);
                    unsafe {
                        // SAFETY: Each Rayon chunk owns a disjoint
                        // bar_input row block; bar_out rows are read-only and
                        // weight is output-major [out_dim, in_dim].
                        row_parallel_gemm_f64(
                            block_rows,
                            in_dim,
                            out_dim,
                            bar_input_block.as_mut_ptr(),
                            in_dim as isize,
                            beta != 0.0,
                            beta,
                            bar_out.as_ptr().add(row_start * out_dim),
                            out_dim as isize,
                            weight.as_ptr(),
                            1,
                            in_dim as isize,
                            alpha,
                        );
                    }
                });
            return;
        }
    }
    dense_vjp_input_output_major_weight(
        rows, in_dim, out_dim, weight, bar_out, bar_input, alpha, beta,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dense_forward_strided_row_parallel_f64(
    input: &[f64],
    rows: usize,
    in_dim: usize,
    input_row_stride: usize,
    weight: &[f64],
    out: &mut [f64],
    out_dim: usize,
    out_row_stride: usize,
    alpha: f64,
    beta: f64,
) {
    debug_assert!(input_row_stride >= in_dim);
    debug_assert!(out_row_stride >= out_dim);
    debug_assert!(weight.len() >= in_dim * out_dim);
    debug_assert!(input.len() >= strided_matrix_len(rows, input_row_stride, in_dim));
    debug_assert!(out.len() >= strided_matrix_len(rows, out_row_stride, out_dim));
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    if rows.saturating_mul(in_dim).saturating_mul(out_dim) >= row_parallel_gemm_min_ops() {
        if let Some(row_block) = row_parallel_gemm_block_rows(rows) {
            out.par_chunks_mut(row_block * out_row_stride)
                .enumerate()
                .for_each(|(block_index, out_block)| {
                    let row_start = block_index * row_block;
                    let block_rows = (rows - row_start).min(row_block);
                    unsafe {
                        // SAFETY: Rayon gives each chunk a disjoint row block
                        // of the strided output. The GEMM only touches each
                        // block's logical columns, and input/weight are
                        // read-only row-major buffers.
                        row_parallel_gemm_f64(
                            block_rows,
                            out_dim,
                            in_dim,
                            out_block.as_mut_ptr(),
                            out_row_stride as isize,
                            beta != 0.0,
                            beta,
                            input.as_ptr().add(row_start * input_row_stride),
                            input_row_stride as isize,
                            weight.as_ptr(),
                            1,
                            out_dim as isize,
                            alpha,
                        );
                    }
                });
            return;
        }
    }
    dense_forward_strided(
        input,
        rows,
        in_dim,
        input_row_stride,
        weight,
        out,
        out_dim,
        out_row_stride,
        alpha,
        beta,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dense_vjp_input_strided_row_parallel_f64(
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out_dim: usize,
    bar_out: &[f64],
    bar_out_row_stride: usize,
    bar_input: &mut [f64],
    bar_input_row_stride: usize,
    alpha: f64,
    beta: f64,
) {
    debug_assert!(bar_out_row_stride >= out_dim);
    debug_assert!(bar_input_row_stride >= in_dim);
    debug_assert!(weight.len() >= in_dim * out_dim);
    debug_assert!(bar_out.len() >= strided_matrix_len(rows, bar_out_row_stride, out_dim));
    debug_assert!(bar_input.len() >= strided_matrix_len(rows, bar_input_row_stride, in_dim));
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    if rows.saturating_mul(in_dim).saturating_mul(out_dim) >= row_parallel_gemm_min_ops() {
        if let Some(row_block) = row_parallel_gemm_block_rows(rows) {
            bar_input
                .par_chunks_mut(row_block * bar_input_row_stride)
                .enumerate()
                .for_each(|(block_index, bar_input_block)| {
                    let row_start = block_index * row_block;
                    let block_rows = (rows - row_start).min(row_block);
                    unsafe {
                        // SAFETY: Rayon gives each chunk a disjoint row block
                        // of the strided input cotangent. The GEMM only
                        // touches each block's logical columns, and
                        // bar_out/weight are read-only row-major buffers.
                        row_parallel_gemm_f64(
                            block_rows,
                            in_dim,
                            out_dim,
                            bar_input_block.as_mut_ptr(),
                            bar_input_row_stride as isize,
                            beta != 0.0,
                            beta,
                            bar_out.as_ptr().add(row_start * bar_out_row_stride),
                            bar_out_row_stride as isize,
                            weight.as_ptr(),
                            out_dim as isize,
                            1,
                            alpha,
                        );
                    }
                });
            return;
        }
    }
    dense_vjp_input_strided(
        rows,
        in_dim,
        weight,
        out_dim,
        bar_out,
        bar_out_row_stride,
        bar_input,
        bar_input_row_stride,
        alpha,
        beta,
    );
}

#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn dense_forward_output_major_weight_strided_output(
    input: &[f64],
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out_dim: usize,
    out: &mut [f64],
    out_row_stride: usize,
    out_feature_offset: usize,
    alpha: f64,
    beta: f64,
) {
    debug_assert!(out_row_stride >= out_dim);
    debug_assert!(input.len() >= rows * in_dim);
    debug_assert!(weight.len() >= out_dim * in_dim);
    debug_assert!(
        out.len() >= out_feature_offset + strided_matrix_len(rows, out_row_stride, out_dim)
    );
    if rows == 0 || out_dim == 0 {
        return;
    }
    if in_dim == 0 {
        let end = out_feature_offset + strided_matrix_len(rows, out_row_stride, out_dim);
        apply_beta_strided_f64(
            &mut out[out_feature_offset..end],
            rows,
            out_dim,
            out_row_stride,
            beta,
        );
        return;
    }
    // Invariant: `out_feature_offset` has been bounds-checked above and
    // `out_row_stride` is the leading dimension for the output rows.
    let out_ptr = unsafe { out.as_mut_ptr().add(out_feature_offset) };
    dgemm_dispatch(
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        CBLAS_TRANS,
        blas_dim(rows),
        blas_dim(out_dim),
        blas_dim(in_dim),
        alpha,
        input.as_ptr(),
        blas_dim(in_dim),
        weight.as_ptr(),
        blas_dim(in_dim),
        beta,
        out_ptr,
        blas_dim(out_row_stride),
    );
}

#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn dense_forward_output_major_weight_alloc(
    input: &[f64],
    rows: usize,
    in_dim: usize,
    weight: &[f64],
    out_dim: usize,
    alpha: f64,
) -> Vec<f64> {
    debug_assert!(input.len() >= rows * in_dim);
    debug_assert!(weight.len() >= out_dim * in_dim);
    let mut out = uninit_f64_buffer(rows * out_dim);
    if rows != 0 && out_dim != 0 && in_dim != 0 {
        // Invariant: input is [rows, in_dim], weight is output-major
        // [out_dim, in_dim], and out is [rows, out_dim], all row-major. DGEMM
        // with beta=0 writes every output element, so the uninitialized buffer
        // is not read before being initialized.
        dgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            blas_dim(rows),
            blas_dim(out_dim),
            blas_dim(in_dim),
            alpha,
            input.as_ptr(),
            blas_dim(in_dim),
            weight.as_ptr(),
            blas_dim(in_dim),
            0.0,
            out.as_mut_ptr().cast::<f64>(),
            blas_dim(out_dim),
        );
    } else if rows != 0 && out_dim != 0 {
        // K (in_dim) == 0: DGEMM early-returns without writing the beta=0
        // output. Zero-fill so the buffer is fully initialized; C = 0 * A * B
        // is the all-zeros matrix.
        zero_fill_f64_uninit(&mut out);
    }
    // SAFETY: every element is now initialized: the beta=0 DGEMM writes the
    // whole buffer when rows, out_dim, and in_dim (K) are all non-zero; the
    // K==0 branch zero-fills it; and a zero-length buffer (rows==0 ||
    // out_dim==0) is trivially initialized.
    unsafe { assume_init_f64_vec(out) }
}

#[cfg(test)]
mod k_zero_init_tests {
    use super::*;

    // M1 regression: when K (in_dim) == 0 the underlying DGEMM early-returns
    // without writing the beta=0 output, so the allocating helper must zero-fill
    // the buffer rather than `assume_init` uninitialized memory (UB). The
    // mathematical result is C = 0 * A * B = the all-zeros matrix.
    #[test]
    fn forward_alloc_with_zero_k_is_zero_filled() {
        let input: [f64; 0] = [];
        let weight: [f64; 0] = [];
        let rows = 3usize;
        let in_dim = 0usize;
        let out_dim = 4usize;
        let out =
            dense_forward_output_major_weight_alloc(&input, rows, in_dim, &weight, out_dim, 1.0);
        assert_eq!(out.len(), rows * out_dim);
        assert!(
            out.iter().all(|x| x.to_bits() == 0),
            "K==0 forward output must be all zeros (was uninitialized before the fix): {out:?}"
        );
    }

    #[test]
    fn forward_into_with_zero_k_applies_beta() {
        let input: [f64; 0] = [];
        let weight: [f64; 0] = [];
        let rows = 2usize;
        let in_dim = 0usize;
        let out_dim = 3usize;

        let mut out = vec![7.0; rows * out_dim];
        dense_forward_output_major_weight(
            &input, rows, in_dim, &weight, &mut out, out_dim, 1.0, 0.0,
        );
        assert!(out.iter().all(|value| value.to_bits() == 0));

        let mut out = vec![8.0; rows * out_dim];
        dense_forward_output_major_weight(
            &input, rows, in_dim, &weight, &mut out, out_dim, 1.0, 0.25,
        );
        assert!(out.iter().all(|&value| value == 2.0));
    }

    #[test]
    fn strided_forward_into_with_zero_k_applies_beta() {
        let input: [f64; 0] = [];
        let weight: [f64; 0] = [];
        let rows = 2usize;
        let in_dim = 0usize;
        let out_dim = 3usize;
        let out_stride = 5usize;
        let out_offset = 1usize;
        let mut out = vec![9.0; out_offset + strided_matrix_len(rows, out_stride, out_dim)];

        dense_forward_output_major_weight_strided_output(
            &input, rows, in_dim, &weight, out_dim, &mut out, out_stride, out_offset, 1.0, 0.5,
        );

        assert_eq!(out[0], 9.0);
        assert_eq!(out[4], 9.0);
        assert_eq!(out[5], 9.0);
        for row in 0..rows {
            let base = out_offset + row * out_stride;
            assert!(out[base..base + out_dim].iter().all(|&value| value == 4.5));
        }
    }

    #[test]
    fn public_strided_forward_and_vjp_with_zero_k_apply_beta() {
        let input: [f64; 0] = [];
        let weight: [f64; 0] = [];
        let rows = 2usize;
        let in_dim = 0usize;
        let out_dim = 3usize;
        let out_stride = 5usize;
        let mut out = vec![9.0; strided_matrix_len(rows, out_stride, out_dim)];

        dense_forward_strided(
            &input, rows, in_dim, 0, &weight, &mut out, out_dim, out_stride, 1.0, 0.5,
        );

        for row in 0..rows {
            let base = row * out_stride;
            assert!(out[base..base + out_dim].iter().all(|&value| value == 4.5));
            if row + 1 < rows {
                assert!(out[base + out_dim..base + out_stride]
                    .iter()
                    .all(|&value| value == 9.0));
            }
        }

        let in_dim = 3usize;
        let out_dim = 0usize;
        let bar_out: [f64; 0] = [];
        let bar_input_stride = 5usize;
        let mut bar_input = vec![8.0; strided_matrix_len(rows, bar_input_stride, in_dim)];
        dense_vjp_input_strided(
            rows,
            in_dim,
            &weight,
            out_dim,
            &bar_out,
            0,
            &mut bar_input,
            bar_input_stride,
            1.0,
            0.25,
        );

        for row in 0..rows {
            let base = row * bar_input_stride;
            assert!(bar_input[base..base + in_dim]
                .iter()
                .all(|&value| value == 2.0));
            if row + 1 < rows {
                assert!(bar_input[base + in_dim..base + bar_input_stride]
                    .iter()
                    .all(|&value| value == 8.0));
            }
        }
    }

    #[test]
    fn public_contiguous_forward_with_zero_live_extent_returns() {
        let input = vec![1.0; 6];
        let weight: [f64; 0] = [];
        let mut out: [f64; 0] = [];
        dense_forward(&input, 2, 3, &weight, &mut out, 1.0);

        let input: [f64; 0] = [];
        let weight = vec![1.0; 6];
        let mut out: [f64; 0] = [];
        dense_forward(&input, 0, 3, &weight, &mut out, 1.0);
    }

    #[test]
    fn vjp_into_with_zero_k_applies_beta() {
        let weight: [f64; 0] = [];
        let bar_out: [f64; 0] = [];
        let rows = 2usize;
        let in_dim = 3usize;
        let out_dim = 0usize;

        let mut bar_input = vec![6.0; rows * in_dim];
        dense_vjp_input_with_beta(rows, in_dim, &weight, &bar_out, &mut bar_input, 1.0, 0.0);
        assert!(bar_input.iter().all(|value| value.to_bits() == 0));

        let mut bar_input = vec![8.0; rows * in_dim];
        dense_vjp_input_output_major_weight(
            rows,
            in_dim,
            out_dim,
            &weight,
            &bar_out,
            &mut bar_input,
            1.0,
            0.25,
        );
        assert!(bar_input.iter().all(|&value| value == 2.0));
    }

    #[test]
    fn column_block_into_with_zero_k_applies_beta() {
        let input: [f64; 0] = [];
        let bar_out: [f64; 0] = [];
        let weight = vec![1.0; 4];
        let rows = 2usize;
        let selected_in_dim = 0usize;
        let full_in_dim = 2usize;
        let column_offset = 1usize;
        let out_dim = 2usize;

        let mut out = vec![7.0; rows * out_dim];
        dense_forward_output_major_weight_column_block_add(
            &input,
            rows,
            selected_in_dim,
            &weight,
            full_in_dim,
            column_offset,
            out_dim,
            &mut out,
            1.0,
            0.0,
        );
        assert!(out.iter().all(|value| value.to_bits() == 0));

        let selected_in_dim = 3usize;
        let out_dim = 0usize;
        let weight: [f64; 0] = [];
        let mut bar_input = vec![10.0; rows * selected_in_dim];
        dense_vjp_input_output_major_weight_column_block_add(
            rows,
            selected_in_dim,
            selected_in_dim,
            out_dim,
            0,
            &weight,
            &bar_out,
            &mut bar_input,
            1.0,
            0.5,
        );
        assert!(bar_input.iter().all(|&value| value == 5.0));
    }

    #[cfg(any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))]
    #[test]
    fn f32_into_with_zero_k_applies_beta() {
        let input: [f32; 0] = [];
        let weight: [f32; 0] = [];
        let bar_out: [f32; 0] = [];
        let rows = 2usize;
        let in_dim = 0usize;
        let out_dim = 3usize;

        let mut out = vec![8.0_f32; rows * out_dim];
        dense_forward_output_major_weight_f32(
            &input, rows, in_dim, &weight, &mut out, out_dim, 1.0, 0.25,
        );
        assert!(out.iter().all(|&value| value == 2.0));

        let in_dim = 3usize;
        let out_dim = 0usize;
        let mut bar_input = vec![8.0_f32; rows * in_dim];
        dense_vjp_input_output_major_weight_f32(
            rows,
            in_dim,
            out_dim,
            &weight,
            &bar_out,
            &mut bar_input,
            1.0,
            0.25,
        );
        assert!(bar_input.iter().all(|&value| value == 2.0));

        let in_dim = 0usize;
        let out_dim = 3usize;
        let out_stride = 5usize;
        let mut out = vec![8.0_f32; strided_matrix_len(rows, out_stride, out_dim)];
        dense_forward_strided_f32(
            &input, rows, in_dim, 0, &weight, &mut out, out_dim, out_stride, 1.0, 0.25,
        );
        for row in 0..rows {
            let base = row * out_stride;
            assert!(out[base..base + out_dim].iter().all(|&value| value == 2.0));
            if row + 1 < rows {
                assert!(out[base + out_dim..base + out_stride]
                    .iter()
                    .all(|&value| value == 8.0));
            }
        }

        let mut out = vec![8.0_f32; rows * out_dim];
        dense_forward_input_major_weight_f32(
            &input, rows, in_dim, &weight, &mut out, out_dim, 1.0, 0.25,
        );
        assert!(out.iter().all(|&value| value == 2.0));

        let input = vec![1.0_f32; 6];
        let weight: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        dense_forward_f32(&input, 2, 3, &weight, &mut out, 1.0);

        let input: [f32; 0] = [];
        let weight = vec![1.0_f32; 6];
        let mut out: [f32; 0] = [];
        dense_forward_f32(&input, 0, 3, &weight, &mut out, 1.0);

        let in_dim = 3usize;
        let out_dim = 0usize;
        let bar_input_stride = 5usize;
        let mut bar_input = vec![8.0_f32; strided_matrix_len(rows, bar_input_stride, in_dim)];
        dense_vjp_input_strided_f32(
            rows,
            in_dim,
            &weight,
            out_dim,
            &bar_out,
            0,
            &mut bar_input,
            bar_input_stride,
            1.0,
            0.25,
        );
        for row in 0..rows {
            let base = row * bar_input_stride;
            assert!(bar_input[base..base + in_dim]
                .iter()
                .all(|&value| value == 2.0));
            if row + 1 < rows {
                assert!(bar_input[base + in_dim..base + bar_input_stride]
                    .iter()
                    .all(|&value| value == 8.0));
            }
        }
    }
}

#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn dense_vjp_input_output_major_weight(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    weight: &[f64],
    bar_out: &[f64],
    bar_input: &mut [f64],
    alpha: f64,
    beta: f64,
) {
    debug_assert!(weight.len() >= out_dim * in_dim);
    debug_assert!(bar_out.len() >= rows * out_dim);
    debug_assert!(bar_input.len() >= rows * in_dim);
    if rows == 0 || in_dim == 0 {
        return;
    }
    if out_dim == 0 {
        apply_beta_contiguous_f64(&mut bar_input[..rows * in_dim], beta);
        return;
    }
    // Invariant: bar_out is [rows, out_dim], weight is output-major
    // [out_dim, in_dim], and bar_input is [rows, in_dim].
    dgemm_dispatch(
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        CBLAS_NO_TRANS,
        blas_dim(rows),
        blas_dim(in_dim),
        blas_dim(out_dim),
        alpha,
        bar_out.as_ptr(),
        blas_dim(out_dim),
        weight.as_ptr(),
        blas_dim(in_dim),
        beta,
        bar_input.as_mut_ptr(),
        blas_dim(in_dim),
    );
}

#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn dense_vjp_input_output_major_weight_alloc(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    weight: &[f64],
    bar_out: &[f64],
    alpha: f64,
) -> Vec<f64> {
    debug_assert!(weight.len() >= out_dim * in_dim);
    debug_assert!(bar_out.len() >= rows * out_dim);
    let mut bar_input = uninit_f64_buffer(rows * in_dim);
    if rows != 0 && in_dim != 0 && out_dim != 0 {
        // Invariant: bar_out is [rows, out_dim], weight is output-major
        // [out_dim, in_dim], and bar_input is [rows, in_dim]. DGEMM with
        // beta=0 writes every output element.
        dgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            blas_dim(rows),
            blas_dim(in_dim),
            blas_dim(out_dim),
            alpha,
            bar_out.as_ptr(),
            blas_dim(out_dim),
            weight.as_ptr(),
            blas_dim(in_dim),
            0.0,
            bar_input.as_mut_ptr().cast::<f64>(),
            blas_dim(in_dim),
        );
    } else if rows != 0 && in_dim != 0 {
        // K (out_dim) == 0: DGEMM early-returns without writing the beta=0
        // output. Zero-fill so the buffer is fully initialized; C = 0 * A * B
        // is the all-zeros matrix.
        zero_fill_f64_uninit(&mut bar_input);
    }
    // SAFETY: every element is now initialized: the beta=0 DGEMM writes the
    // whole buffer when rows, in_dim, and out_dim (K) are all non-zero; the
    // K==0 branch zero-fills it; and a zero-length buffer (rows==0 ||
    // in_dim==0) is trivially initialized.
    unsafe { assume_init_f64_vec(bar_input) }
}

#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn dense_forward_output_major_weight_column_block_add(
    input: &[f64],
    rows: usize,
    selected_in_dim: usize,
    weight: &[f64],
    full_in_dim: usize,
    column_offset: usize,
    out_dim: usize,
    out: &mut [f64],
    alpha: f64,
    beta: f64,
) {
    debug_assert!(column_offset <= full_in_dim);
    debug_assert!(selected_in_dim <= full_in_dim - column_offset);
    debug_assert!(input.len() >= rows * selected_in_dim);
    debug_assert!(weight.len() >= out_dim * full_in_dim);
    debug_assert!(out.len() >= rows * out_dim);
    if rows == 0 || out_dim == 0 {
        return;
    }
    if selected_in_dim == 0 {
        apply_beta_contiguous_f64(&mut out[..rows * out_dim], beta);
        return;
    }
    dgemm_dispatch(
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        CBLAS_TRANS,
        blas_dim(rows),
        blas_dim(out_dim),
        blas_dim(selected_in_dim),
        alpha,
        input.as_ptr(),
        blas_dim(selected_in_dim),
        unsafe { weight.as_ptr().add(column_offset) },
        blas_dim(full_in_dim),
        beta,
        out.as_mut_ptr(),
        blas_dim(out_dim),
    );
}

#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn dense_vjp_input_output_major_weight_column_block_add(
    rows: usize,
    selected_in_dim: usize,
    full_in_dim: usize,
    out_dim: usize,
    column_offset: usize,
    weight: &[f64],
    bar_out: &[f64],
    bar_input: &mut [f64],
    alpha: f64,
    beta: f64,
) {
    debug_assert!(column_offset <= full_in_dim);
    debug_assert!(selected_in_dim <= full_in_dim - column_offset);
    debug_assert!(weight.len() >= out_dim * full_in_dim);
    debug_assert!(bar_out.len() >= rows * out_dim);
    debug_assert!(bar_input.len() >= rows * selected_in_dim);
    if rows == 0 || selected_in_dim == 0 {
        return;
    }
    if out_dim == 0 {
        apply_beta_contiguous_f64(&mut bar_input[..rows * selected_in_dim], beta);
        return;
    }
    dgemm_dispatch(
        CBLAS_ROW_MAJOR,
        CBLAS_NO_TRANS,
        CBLAS_NO_TRANS,
        blas_dim(rows),
        blas_dim(selected_in_dim),
        blas_dim(out_dim),
        alpha,
        bar_out.as_ptr(),
        blas_dim(out_dim),
        unsafe { weight.as_ptr().add(column_offset) },
        blas_dim(full_in_dim),
        beta,
        bar_input.as_mut_ptr(),
        blas_dim(selected_in_dim),
    );
}

#[allow(dead_code, clippy::too_many_arguments)]
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub(crate) fn dense_forward_output_major_weight_f32(
    input: &[f32],
    rows: usize,
    in_dim: usize,
    weight: &[f32],
    out: &mut [f32],
    out_dim: usize,
    alpha: f32,
    beta: f32,
) {
    debug_assert!(input.len() >= rows * in_dim);
    debug_assert!(weight.len() >= out_dim * in_dim);
    debug_assert!(out.len() >= rows * out_dim);
    if rows == 0 || out_dim == 0 {
        return;
    }
    if in_dim == 0 {
        apply_beta_contiguous_f32(&mut out[..rows * out_dim], beta);
        return;
    }
    unsafe {
        // Invariant: input is [rows, in_dim], weight is output-major
        // [out_dim, in_dim], and out is [rows, out_dim], all row-major.
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            blas_dim(rows),
            blas_dim(out_dim),
            blas_dim(in_dim),
            alpha,
            input.as_ptr(),
            blas_dim(in_dim),
            weight.as_ptr(),
            blas_dim(in_dim),
            beta,
            out.as_mut_ptr(),
            blas_dim(out_dim),
        );
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub(crate) fn dense_forward_input_major_weight_f32(
    input: &[f32],
    rows: usize,
    in_dim: usize,
    weight_in_by_out: &[f32],
    out: &mut [f32],
    out_dim: usize,
    alpha: f32,
    beta: f32,
) {
    debug_assert!(input.len() >= rows * in_dim);
    debug_assert!(weight_in_by_out.len() >= in_dim * out_dim);
    debug_assert!(out.len() >= rows * out_dim);
    debug_assert_c_int_dims(&[("rows", rows), ("in_dim", in_dim), ("out_dim", out_dim)]);
    if rows == 0 || out_dim == 0 {
        return;
    }
    if in_dim == 0 {
        apply_beta_contiguous_f32(&mut out[..rows * out_dim], beta);
        return;
    }
    unsafe {
        // Invariant: input is [rows, in_dim], weight is input-major
        // [in_dim, out_dim], and out is [rows, out_dim], all row-major.
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            blas_dim(rows),
            blas_dim(out_dim),
            blas_dim(in_dim),
            alpha,
            input.as_ptr(),
            blas_dim(in_dim),
            weight_in_by_out.as_ptr(),
            blas_dim(out_dim),
            beta,
            out.as_mut_ptr(),
            blas_dim(out_dim),
        );
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub(crate) fn dense_vjp_input_output_major_weight_f32(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    weight: &[f32],
    bar_out: &[f32],
    bar_input: &mut [f32],
    alpha: f32,
    beta: f32,
) {
    debug_assert!(weight.len() >= out_dim * in_dim);
    debug_assert!(bar_out.len() >= rows * out_dim);
    debug_assert!(bar_input.len() >= rows * in_dim);
    if rows == 0 || in_dim == 0 {
        return;
    }
    if out_dim == 0 {
        apply_beta_contiguous_f32(&mut bar_input[..rows * in_dim], beta);
        return;
    }
    unsafe {
        // Invariant: bar_out is [rows, out_dim], weight is output-major
        // [out_dim, in_dim], and bar_input is [rows, in_dim].
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            blas_dim(rows),
            blas_dim(in_dim),
            blas_dim(out_dim),
            alpha,
            bar_out.as_ptr(),
            blas_dim(out_dim),
            weight.as_ptr(),
            blas_dim(in_dim),
            beta,
            bar_input.as_mut_ptr(),
            blas_dim(in_dim),
        );
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub(crate) fn dense_vjp_input_output_major_weight_strided_f32(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    weight: &[f32],
    bar_out: &[f32],
    bar_input: &mut [f32],
    bar_input_row_stride: usize,
    alpha: f32,
    beta: f32,
) {
    debug_assert!(bar_input_row_stride >= in_dim);
    debug_assert!(weight.len() >= out_dim * in_dim);
    debug_assert!(bar_out.len() >= rows * out_dim);
    debug_assert!(
        bar_input.len()
            >= rows
                .saturating_sub(1)
                .saturating_mul(bar_input_row_stride)
                .saturating_add(in_dim)
    );
    if rows == 0 || in_dim == 0 {
        return;
    }
    if out_dim == 0 {
        apply_beta_strided_f32(bar_input, rows, in_dim, bar_input_row_stride, beta);
        return;
    }
    unsafe {
        // Invariant: bar_out is [rows, out_dim], weight is output-major
        // [out_dim, in_dim], and bar_input has row stride
        // `bar_input_row_stride` with at least `in_dim` writable values per row.
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            blas_dim(rows),
            blas_dim(in_dim),
            blas_dim(out_dim),
            alpha,
            bar_out.as_ptr(),
            blas_dim(out_dim),
            weight.as_ptr(),
            blas_dim(in_dim),
            beta,
            bar_input.as_mut_ptr(),
            blas_dim(bar_input_row_stride),
        );
    }
}

// Thread-local scratch buffers for the fused SO2 m-conv weight matrices (f32).
// Avoids per-call allocation when the fused weight is the same size across
// multiple iterations. Both the forward and VJP fused functions reuse these
// buffers to eliminate repeated Vec allocation overhead.
#[allow(clippy::too_many_arguments, dead_code)]
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub(crate) fn dense_vjp_input_input_major_weight_f32(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    input_major_weight: &[f32],
    bar_out: &[f32],
    bar_input: &mut [f32],
    alpha: f32,
    beta: f32,
) {
    debug_assert!(input_major_weight.len() >= in_dim * out_dim);
    debug_assert!(bar_out.len() >= rows * out_dim);
    debug_assert!(bar_input.len() >= rows * in_dim);
    if rows == 0 || in_dim == 0 {
        return;
    }
    if out_dim == 0 {
        apply_beta_contiguous_f32(&mut bar_input[..rows * in_dim], beta);
        return;
    }
    unsafe {
        // Invariant: bar_out is [rows, out_dim], input_major_weight is
        // [in_dim, out_dim], and bar_input is [rows, in_dim].
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            blas_dim(rows),
            blas_dim(in_dim),
            blas_dim(out_dim),
            alpha,
            bar_out.as_ptr(),
            blas_dim(out_dim),
            input_major_weight.as_ptr(),
            blas_dim(out_dim),
            beta,
            bar_input.as_mut_ptr(),
            blas_dim(in_dim),
        );
    }
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
thread_local! {
    static SO2_MCONV_FWD_FUSED_WEIGHT_SCRATCH_F32: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static SO2_MCONV_VJP_FUSED_WEIGHT_SCRATCH_F32: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Fused SO2 m-conv forward for f32: one SGEMM instead of four.
///
/// Computes `out = input @ fused_weight` (set, not accumulate) where `fused_weight`
/// is the `[2*in_dim, 2*out_dim]` block matrix `[[W0^T, W1^T], [-W1^T, W0^T]]` built
/// from the SO2 complex weight `mixed_weight = [W0 | W1]` ([out_dim, in_dim] each).
///
/// Forward complex matmul:
/// - `real_out = real_in @ W0^T - imag_in @ W1^T`
/// - `imag_out = imag_in @ W0^T + real_in @ W1^T`
///
/// Inputs/outputs are block-interleaved: each row holds `[real | imag]` halves.
/// - `input`: `[rows, 2 * in_dim]`
/// - `out`:   `[rows, 2 * out_dim]` (overwritten)
/// - `mixed_weight`: `[2 * out_dim * in_dim]` (W0 then W1)
///
/// Uses a thread-local scratch buffer to avoid per-call allocation.
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub(crate) fn so2_mconv_forward_set_fused_f32(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    mixed_weight: &[f32],
    input: &[f32],
    out: &mut [f32],
) {
    debug_assert!(mixed_weight.len() >= 2 * out_dim * in_dim);
    debug_assert!(input.len() >= rows * 2 * in_dim);
    debug_assert!(out.len() >= rows * 2 * out_dim);

    let fused_in_dim = 2 * in_dim;
    let fused_out_dim = 2 * out_dim;
    let w0 = &mixed_weight[..out_dim * in_dim];
    let w1 = &mixed_weight[out_dim * in_dim..2 * out_dim * in_dim];

    SO2_MCONV_FWD_FUSED_WEIGHT_SCRATCH_F32.with(|cell| {
        let mut fused = cell.borrow_mut();
        let fused_len = fused_in_dim * fused_out_dim;
        fused.clear();
        fused.resize(fused_len, 0.0f32);

        // Build fused weight [[W0^T, W1^T], [-W1^T, W0^T]] as [2*in_dim, 2*out_dim].
        // w0 is stored [out_dim, in_dim], so W0^T[c][r] = w0[r * in_dim + c].
        for r in 0..out_dim {
            for c in 0..in_dim {
                let w0_val = w0[r * in_dim + c];
                let w1_val = w1[r * in_dim + c];
                // W0^T top-left: fused[c, r]
                fused[c * fused_out_dim + r] = w0_val;
                // W1^T top-right: fused[c, out_dim + r]
                fused[c * fused_out_dim + out_dim + r] = w1_val;
                // -W1^T bottom-left: fused[in_dim + c, r]
                fused[(in_dim + c) * fused_out_dim + r] = -w1_val;
                // W0^T bottom-right: fused[in_dim + c, out_dim + r]
                fused[(in_dim + c) * fused_out_dim + out_dim + r] = w0_val;
            }
        }

        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                blas_dim(rows),
                blas_dim(fused_out_dim),
                blas_dim(fused_in_dim),
                1.0_f32,
                input.as_ptr(),
                blas_dim(fused_in_dim),
                fused.as_ptr(),
                blas_dim(fused_out_dim),
                0.0_f32,
                out.as_mut_ptr(),
                blas_dim(fused_out_dim),
            );
        }
    });
}

/// Fused SO2 m-conv VJP for f32: one SGEMM instead of four.
///
/// Computes `grad_input += grad_out @ fused_weight` where `fused_weight` is
/// the `[2*out_dim, 2*in_dim]` block matrix `[[W0, -W1], [W1, W0]]` built from
/// the SO2 complex weight `mixed_weight = [W0 | W1]` ([out_dim, in_dim] each).
///
/// Inputs/outputs are block-interleaved: each row holds `[real | imag]` halves.
/// - `grad_out`:   `[rows, 2 * out_dim]`
/// - `grad_input`: `[rows, 2 * in_dim]` (accumulated in place)
/// - `mixed_weight`: `[2 * out_dim * in_dim]` (W0 then W1)
///
/// Uses a thread-local scratch buffer to avoid per-call allocation.
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub(crate) fn so2_mconv_vjp_input_add_fused_f32(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    mixed_weight: &[f32],
    grad_out: &[f32],
    grad_input: &mut [f32],
) {
    debug_assert!(mixed_weight.len() >= 2 * out_dim * in_dim);
    debug_assert!(grad_out.len() >= rows * 2 * out_dim);
    debug_assert!(grad_input.len() >= rows * 2 * in_dim);

    let fused_in_dim = 2 * in_dim;
    let fused_out_dim = 2 * out_dim;
    let w0 = &mixed_weight[..out_dim * in_dim];
    let w1 = &mixed_weight[out_dim * in_dim..2 * out_dim * in_dim];

    SO2_MCONV_VJP_FUSED_WEIGHT_SCRATCH_F32.with(|cell| {
        let mut fused = cell.borrow_mut();
        let fused_len = fused_out_dim * fused_in_dim;
        fused.clear();
        fused.resize(fused_len, 0.0f32);

        // Build fused weight [[W0, -W1], [W1, W0]] as [2*out_dim, 2*in_dim].
        for r in 0..out_dim {
            let src = r * in_dim;
            let dst_real = r * fused_in_dim;
            let dst_imag = (out_dim + r) * fused_in_dim;
            for c in 0..in_dim {
                fused[dst_real + c] = w0[src + c]; // W0 top-left
                fused[dst_real + in_dim + c] = -w1[src + c]; // -W1 top-right
                fused[dst_imag + c] = w1[src + c]; // W1 bottom-left
                fused[dst_imag + in_dim + c] = w0[src + c]; // W0 bottom-right
            }
        }

        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                blas_dim(rows),
                blas_dim(fused_in_dim),
                blas_dim(fused_out_dim),
                1.0_f32,
                grad_out.as_ptr(),
                blas_dim(fused_out_dim),
                fused.as_ptr(),
                blas_dim(fused_in_dim),
                1.0_f32,
                grad_input.as_mut_ptr(),
                blas_dim(fused_in_dim),
            );
        }
    });
}

/// Build the precomputed fused forward weight for f32 SO2 m-conv.
///
/// Returns the `[2*in_dim, 2*out_dim]` matrix `[[W0^T, W1^T], [-W1^T, W0^T]]`.
/// `mixed_weight` is `[2 * out_dim * in_dim]` (W0 then W1, each `[out_dim, in_dim]`).
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub(crate) fn so2_mconv_build_fused_fwd_weight_f32(
    in_dim: usize,
    out_dim: usize,
    mixed_weight: &[f32],
) -> Vec<f32> {
    debug_assert!(mixed_weight.len() >= 2 * out_dim * in_dim);
    let fused_in_dim = 2 * in_dim;
    let fused_out_dim = 2 * out_dim;
    let w0 = &mixed_weight[..out_dim * in_dim];
    let w1 = &mixed_weight[out_dim * in_dim..2 * out_dim * in_dim];
    let mut fused = vec![0.0f32; fused_in_dim * fused_out_dim];
    #[cfg(feature = "parallel")]
    if fused.len() >= UMA_FUSED_WEIGHT_PARALLEL_MIN_LEN && rayon::current_num_threads() > 1 {
        fused
            .par_chunks_mut(fused_out_dim)
            .enumerate()
            .for_each(|(row_index, row)| {
                let (c, imag_input) = if row_index < in_dim {
                    (row_index, false)
                } else {
                    (row_index - in_dim, true)
                };
                for r in 0..out_dim {
                    let w_index = r * in_dim + c;
                    let w0_val = w0[w_index];
                    let w1_val = w1[w_index];
                    if imag_input {
                        row[r] = -w1_val;
                        row[out_dim + r] = w0_val;
                    } else {
                        row[r] = w0_val;
                        row[out_dim + r] = w1_val;
                    }
                }
            });
        return fused;
    }
    for r in 0..out_dim {
        for c in 0..in_dim {
            let w0_val = w0[r * in_dim + c];
            let w1_val = w1[r * in_dim + c];
            fused[c * fused_out_dim + r] = w0_val;
            fused[c * fused_out_dim + out_dim + r] = w1_val;
            fused[(in_dim + c) * fused_out_dim + r] = -w1_val;
            fused[(in_dim + c) * fused_out_dim + out_dim + r] = w0_val;
        }
    }
    fused
}

/// Build the precomputed fused VJP weight for f32 SO2 m-conv.
///
/// Returns the `[2*out_dim, 2*in_dim]` matrix `[[W0, -W1], [W1, W0]]`.
/// `mixed_weight` is `[2 * out_dim * in_dim]` (W0 then W1, each `[out_dim, in_dim]`).
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub(crate) fn so2_mconv_build_fused_vjp_weight_f32(
    in_dim: usize,
    out_dim: usize,
    mixed_weight: &[f32],
) -> Vec<f32> {
    debug_assert!(mixed_weight.len() >= 2 * out_dim * in_dim);
    let fused_in_dim = 2 * in_dim;
    let fused_out_dim = 2 * out_dim;
    let w0 = &mixed_weight[..out_dim * in_dim];
    let w1 = &mixed_weight[out_dim * in_dim..2 * out_dim * in_dim];
    let mut fused = vec![0.0f32; fused_out_dim * fused_in_dim];
    #[cfg(feature = "parallel")]
    if fused.len() >= UMA_FUSED_WEIGHT_PARALLEL_MIN_LEN && rayon::current_num_threads() > 1 {
        fused
            .par_chunks_mut(fused_in_dim)
            .enumerate()
            .for_each(|(row_index, row)| {
                let (r, imag_output) = if row_index < out_dim {
                    (row_index, false)
                } else {
                    (row_index - out_dim, true)
                };
                let src = r * in_dim;
                for c in 0..in_dim {
                    let w0_val = w0[src + c];
                    let w1_val = w1[src + c];
                    if imag_output {
                        row[c] = w1_val;
                        row[in_dim + c] = w0_val;
                    } else {
                        row[c] = w0_val;
                        row[in_dim + c] = -w1_val;
                    }
                }
            });
        return fused;
    }
    for r in 0..out_dim {
        let dst_real = r * fused_in_dim;
        let dst_imag = (out_dim + r) * fused_in_dim;
        for c in 0..in_dim {
            let w0_val = w0[r * in_dim + c];
            let w1_val = w1[r * in_dim + c];
            fused[dst_real + c] = w0_val;
            fused[dst_real + in_dim + c] = -w1_val;
            fused[dst_imag + c] = w1_val;
            fused[dst_imag + in_dim + c] = w0_val;
        }
    }
    fused
}

/// Build the precomputed fused forward weight for f64 SO2 m-conv.
///
/// Returns the `[2*in_dim, 2*out_dim]` matrix `[[W0^T, W1^T], [-W1^T, W0^T]]`.
/// `mixed_weight` is `[2 * out_dim * in_dim]` (W0 then W1, each `[out_dim, in_dim]`).
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub(crate) fn so2_mconv_build_fused_fwd_weight(
    in_dim: usize,
    out_dim: usize,
    mixed_weight: &[f64],
) -> Vec<f64> {
    debug_assert!(mixed_weight.len() >= 2 * out_dim * in_dim);
    let fused_in_dim = 2 * in_dim;
    let fused_out_dim = 2 * out_dim;
    let w0 = &mixed_weight[..out_dim * in_dim];
    let w1 = &mixed_weight[out_dim * in_dim..2 * out_dim * in_dim];
    let mut fused = vec![0.0f64; fused_in_dim * fused_out_dim];
    #[cfg(feature = "parallel")]
    if fused.len() >= UMA_FUSED_WEIGHT_PARALLEL_MIN_LEN && rayon::current_num_threads() > 1 {
        fused
            .par_chunks_mut(fused_out_dim)
            .enumerate()
            .for_each(|(row_index, row)| {
                let (c, imag_input) = if row_index < in_dim {
                    (row_index, false)
                } else {
                    (row_index - in_dim, true)
                };
                for r in 0..out_dim {
                    let w_index = r * in_dim + c;
                    let w0_val = w0[w_index];
                    let w1_val = w1[w_index];
                    if imag_input {
                        row[r] = -w1_val;
                        row[out_dim + r] = w0_val;
                    } else {
                        row[r] = w0_val;
                        row[out_dim + r] = w1_val;
                    }
                }
            });
        return fused;
    }
    for r in 0..out_dim {
        for c in 0..in_dim {
            let w0_val = w0[r * in_dim + c];
            let w1_val = w1[r * in_dim + c];
            fused[c * fused_out_dim + r] = w0_val;
            fused[c * fused_out_dim + out_dim + r] = w1_val;
            fused[(in_dim + c) * fused_out_dim + r] = -w1_val;
            fused[(in_dim + c) * fused_out_dim + out_dim + r] = w0_val;
        }
    }
    fused
}

/// Build the precomputed fused VJP weight for f64 SO2 m-conv.
///
/// Returns the `[2*out_dim, 2*in_dim]` matrix `[[W0, -W1], [W1, W0]]`.
/// `mixed_weight` is `[2 * out_dim * in_dim]` (W0 then W1, each `[out_dim, in_dim]`).
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub(crate) fn so2_mconv_build_fused_vjp_weight(
    in_dim: usize,
    out_dim: usize,
    mixed_weight: &[f64],
) -> Vec<f64> {
    debug_assert!(mixed_weight.len() >= 2 * out_dim * in_dim);
    let fused_in_dim = 2 * in_dim;
    let fused_out_dim = 2 * out_dim;
    let w0 = &mixed_weight[..out_dim * in_dim];
    let w1 = &mixed_weight[out_dim * in_dim..2 * out_dim * in_dim];
    let mut fused = vec![0.0f64; fused_out_dim * fused_in_dim];
    #[cfg(feature = "parallel")]
    if fused.len() >= UMA_FUSED_WEIGHT_PARALLEL_MIN_LEN && rayon::current_num_threads() > 1 {
        fused
            .par_chunks_mut(fused_in_dim)
            .enumerate()
            .for_each(|(row_index, row)| {
                let (r, imag_output) = if row_index < out_dim {
                    (row_index, false)
                } else {
                    (row_index - out_dim, true)
                };
                let src = r * in_dim;
                for c in 0..in_dim {
                    let w0_val = w0[src + c];
                    let w1_val = w1[src + c];
                    if imag_output {
                        row[c] = w1_val;
                        row[in_dim + c] = w0_val;
                    } else {
                        row[c] = w0_val;
                        row[in_dim + c] = -w1_val;
                    }
                }
            });
        return fused;
    }
    for r in 0..out_dim {
        let dst_real = r * fused_in_dim;
        let dst_imag = (out_dim + r) * fused_in_dim;
        for c in 0..in_dim {
            let w0_val = w0[r * in_dim + c];
            let w1_val = w1[r * in_dim + c];
            fused[dst_real + c] = w0_val;
            fused[dst_real + in_dim + c] = -w1_val;
            fused[dst_imag + c] = w1_val;
            fused[dst_imag + in_dim + c] = w0_val;
        }
    }
    fused
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[inline]
fn uma_so2_f32_simd_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var_os("LLMLIP_DISABLE_SIMD").is_some())
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[inline]
fn uma_so2_f32_avx_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var_os("LLMLIP_DISABLE_AVX").is_some())
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[inline]
fn uma_so2_f32_avx2_fma_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !uma_so2_f32_simd_disabled()
            && !uma_so2_f32_avx_disabled()
            && std::env::var_os("LLMLIP_DISABLE_AVX2").is_none()
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
    })
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[inline]
fn uma_so2_f32_avx512_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !uma_so2_f32_simd_disabled()
            && !uma_so2_f32_avx_disabled()
            && (std::env::var_os("LLMLIP_ENABLE_F32_AVX512").is_some()
                || std::env::var_os("LLMLIP_ENABLE_AVX512").is_some())
            && std::env::var_os("LLMLIP_DISABLE_AVX512").is_none()
            && std::arch::is_x86_feature_detected!("avx512f")
    })
}

/// Opt-in AVX-512 gate for UMA *elementwise* kernels (sigmoid/SiLU/scatter).
/// Keep these kernels behind the same explicit environment knobs as SO2 matmul
/// until AVX-512 CI coverage is available.
#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[inline]
fn uma_elementwise_avx512_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !uma_so2_f32_simd_disabled()
            && !uma_so2_f32_avx_disabled()
            && (std::env::var_os("LLMLIP_ENABLE_F32_AVX512").is_some()
                || std::env::var_os("LLMLIP_ENABLE_AVX512").is_some())
            && std::env::var_os("LLMLIP_DISABLE_AVX512").is_none()
            && std::arch::is_x86_feature_detected!("avx512f")
    })
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[inline]
fn uma_so2_f32_provider_threads_is_one() -> bool {
    static THREADS_ONE: OnceLock<bool> = OnceLock::new();
    *THREADS_ONE.get_or_init(|| {
        #[cfg(feature = "openblas")]
        {
            if std::env::var("OPENBLAS_NUM_THREADS")
                .ok()
                .is_some_and(|value| value.trim() == "1")
            {
                return true;
            }
        }
        #[cfg(feature = "intel-mkl")]
        {
            if std::env::var("MKL_NUM_THREADS")
                .ok()
                .is_some_and(|value| value.trim() == "1")
            {
                return true;
            }
        }
        false
    })
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[inline]
fn uma_so2_f32_row_kernel_should_run(
    rows: usize,
    k: usize,
    n: usize,
    require_provider_threads_one: bool,
) -> bool {
    rayon::current_num_threads() > 1
        && (!require_provider_threads_one || uma_so2_f32_provider_threads_is_one())
        && rows >= 16
        && k >= 128
        && n >= 128
        && k.is_multiple_of(8)
        && n.is_multiple_of(8)
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn uma_f32_try_l2_m2_m0_gate_vjp(
    nodes: usize,
    channels: usize,
    activated_m0_grad: &[f32],
    conv1_out_l_major: &[f32],
    gate_sigmoids: &[f32],
    scalar_silu_derivatives: &[f32],
    conv1_m0_full_grad_out: &mut [f32],
) -> bool {
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    {
        let m0_out_dim = 5 * channels;
        debug_assert!(activated_m0_grad.len() >= nodes * 3 * channels);
        debug_assert!(conv1_out_l_major.len() >= nodes * 9 * channels);
        debug_assert!(gate_sigmoids.len() >= nodes * 2 * channels);
        debug_assert!(scalar_silu_derivatives.len() >= nodes * channels);
        debug_assert!(conv1_m0_full_grad_out.len() >= nodes * m0_out_dim);

        if nodes >= 16 && channels.is_multiple_of(8) {
            if uma_so2_f32_avx512_enabled() && channels.is_multiple_of(16) {
                unsafe {
                    uma_f32_l2_m2_m0_gate_vjp_avx512(
                        nodes,
                        channels,
                        activated_m0_grad,
                        conv1_out_l_major,
                        gate_sigmoids,
                        scalar_silu_derivatives,
                        conv1_m0_full_grad_out,
                    );
                }
                return true;
            }
            if uma_so2_f32_avx2_fma_enabled() {
                unsafe {
                    uma_f32_l2_m2_m0_gate_vjp_avx2(
                        nodes,
                        channels,
                        activated_m0_grad,
                        conv1_out_l_major,
                        gate_sigmoids,
                        scalar_silu_derivatives,
                        conv1_m0_full_grad_out,
                    );
                }
                return true;
            }
        }
    }
    #[cfg(not(all(feature = "parallel", target_arch = "x86_64")))]
    {
        let _ = nodes;
        let _ = channels;
        let _ = activated_m0_grad;
        let _ = conv1_out_l_major;
        let _ = gate_sigmoids;
        let _ = scalar_silu_derivatives;
        let _ = conv1_m0_full_grad_out;
    }
    false
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn uma_f32_l2_m2_m0_gate_vjp_avx512(
    nodes: usize,
    channels: usize,
    activated_m0_grad: &[f32],
    conv1_out_l_major: &[f32],
    gate_sigmoids: &[f32],
    scalar_silu_derivatives: &[f32],
    conv1_m0_full_grad_out: &mut [f32],
) {
    use std::arch::x86_64::{_mm512_fmadd_ps, _mm512_loadu_ps, _mm512_storeu_ps};

    let m0_out_dim = 5 * channels;
    let act_stride = 3 * channels;
    let out_stride = 9 * channels;
    let gate_stride = 2 * channels;
    for node in 0..nodes {
        let act_base = node * act_stride;
        let conv1_base = node * out_stride;
        let gate_base = node * gate_stride;
        let full_base = node * m0_out_dim;
        let mut ch = 0usize;
        while ch + 16 <= channels {
            let grad_scalar =
                unsafe { _mm512_loadu_ps(activated_m0_grad.as_ptr().add(act_base + ch)) };
            let scalar_deriv = unsafe {
                _mm512_loadu_ps(scalar_silu_derivatives.as_ptr().add(node * channels + ch))
            };
            let full_scalar = unsafe {
                _mm512_loadu_ps(
                    conv1_m0_full_grad_out
                        .as_ptr()
                        .add(full_base + 2 * channels + ch),
                )
            };
            unsafe {
                _mm512_storeu_ps(
                    conv1_m0_full_grad_out
                        .as_mut_ptr()
                        .add(full_base + 2 * channels + ch),
                    _mm512_fmadd_ps(grad_scalar, scalar_deriv, full_scalar),
                );
            }

            let grad_l1 = unsafe {
                _mm512_loadu_ps(activated_m0_grad.as_ptr().add(act_base + channels + ch))
            };
            let gate_l1 = unsafe { _mm512_loadu_ps(gate_sigmoids.as_ptr().add(gate_base + ch)) };
            let conv1_l1 = unsafe {
                _mm512_loadu_ps(conv1_out_l_major.as_ptr().add(conv1_base + channels + ch))
            };
            let full_l1 = unsafe {
                _mm512_loadu_ps(
                    conv1_m0_full_grad_out
                        .as_ptr()
                        .add(full_base + 3 * channels + ch),
                )
            };
            let full_gate_l1 =
                unsafe { _mm512_loadu_ps(conv1_m0_full_grad_out.as_ptr().add(full_base + ch)) };
            unsafe {
                _mm512_storeu_ps(
                    conv1_m0_full_grad_out
                        .as_mut_ptr()
                        .add(full_base + 3 * channels + ch),
                    _mm512_fmadd_ps(grad_l1, gate_l1, full_l1),
                );
                _mm512_storeu_ps(
                    conv1_m0_full_grad_out.as_mut_ptr().add(full_base + ch),
                    _mm512_fmadd_ps(grad_l1, conv1_l1, full_gate_l1),
                );
            }

            let grad_l2 = unsafe {
                _mm512_loadu_ps(activated_m0_grad.as_ptr().add(act_base + 2 * channels + ch))
            };
            let gate_l2 =
                unsafe { _mm512_loadu_ps(gate_sigmoids.as_ptr().add(gate_base + channels + ch)) };
            let conv1_l2 = unsafe {
                _mm512_loadu_ps(
                    conv1_out_l_major
                        .as_ptr()
                        .add(conv1_base + 2 * channels + ch),
                )
            };
            let full_l2 = unsafe {
                _mm512_loadu_ps(
                    conv1_m0_full_grad_out
                        .as_ptr()
                        .add(full_base + 4 * channels + ch),
                )
            };
            let full_gate_l2 = unsafe {
                _mm512_loadu_ps(
                    conv1_m0_full_grad_out
                        .as_ptr()
                        .add(full_base + channels + ch),
                )
            };
            unsafe {
                _mm512_storeu_ps(
                    conv1_m0_full_grad_out
                        .as_mut_ptr()
                        .add(full_base + 4 * channels + ch),
                    _mm512_fmadd_ps(grad_l2, gate_l2, full_l2),
                );
                _mm512_storeu_ps(
                    conv1_m0_full_grad_out
                        .as_mut_ptr()
                        .add(full_base + channels + ch),
                    _mm512_fmadd_ps(grad_l2, conv1_l2, full_gate_l2),
                );
            }
            ch += 16;
        }
        while ch < channels {
            let grad_scalar = activated_m0_grad[act_base + ch];
            conv1_m0_full_grad_out[full_base + 2 * channels + ch] +=
                grad_scalar * scalar_silu_derivatives[node * channels + ch];

            let grad_l1 = activated_m0_grad[act_base + channels + ch];
            conv1_m0_full_grad_out[full_base + 3 * channels + ch] +=
                grad_l1 * gate_sigmoids[gate_base + ch];
            conv1_m0_full_grad_out[full_base + ch] +=
                grad_l1 * conv1_out_l_major[conv1_base + channels + ch];

            let grad_l2 = activated_m0_grad[act_base + 2 * channels + ch];
            conv1_m0_full_grad_out[full_base + 4 * channels + ch] +=
                grad_l2 * gate_sigmoids[gate_base + channels + ch];
            conv1_m0_full_grad_out[full_base + channels + ch] +=
                grad_l2 * conv1_out_l_major[conv1_base + 2 * channels + ch];
            ch += 1;
        }
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn uma_f32_l2_m2_m0_gate_vjp_avx2(
    nodes: usize,
    channels: usize,
    activated_m0_grad: &[f32],
    conv1_out_l_major: &[f32],
    gate_sigmoids: &[f32],
    scalar_silu_derivatives: &[f32],
    conv1_m0_full_grad_out: &mut [f32],
) {
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_storeu_ps};

    let m0_out_dim = 5 * channels;
    let act_stride = 3 * channels;
    let out_stride = 9 * channels;
    let gate_stride = 2 * channels;
    for node in 0..nodes {
        let act_base = node * act_stride;
        let conv1_base = node * out_stride;
        let gate_base = node * gate_stride;
        let full_base = node * m0_out_dim;
        let mut ch = 0usize;
        while ch + 8 <= channels {
            let grad_scalar =
                unsafe { _mm256_loadu_ps(activated_m0_grad.as_ptr().add(act_base + ch)) };
            let scalar_deriv = unsafe {
                _mm256_loadu_ps(scalar_silu_derivatives.as_ptr().add(node * channels + ch))
            };
            let full_scalar = unsafe {
                _mm256_loadu_ps(
                    conv1_m0_full_grad_out
                        .as_ptr()
                        .add(full_base + 2 * channels + ch),
                )
            };
            unsafe {
                _mm256_storeu_ps(
                    conv1_m0_full_grad_out
                        .as_mut_ptr()
                        .add(full_base + 2 * channels + ch),
                    _mm256_fmadd_ps(grad_scalar, scalar_deriv, full_scalar),
                );
            }

            let grad_l1 = unsafe {
                _mm256_loadu_ps(activated_m0_grad.as_ptr().add(act_base + channels + ch))
            };
            let gate_l1 = unsafe { _mm256_loadu_ps(gate_sigmoids.as_ptr().add(gate_base + ch)) };
            let conv1_l1 = unsafe {
                _mm256_loadu_ps(conv1_out_l_major.as_ptr().add(conv1_base + channels + ch))
            };
            let full_l1 = unsafe {
                _mm256_loadu_ps(
                    conv1_m0_full_grad_out
                        .as_ptr()
                        .add(full_base + 3 * channels + ch),
                )
            };
            let full_gate_l1 =
                unsafe { _mm256_loadu_ps(conv1_m0_full_grad_out.as_ptr().add(full_base + ch)) };
            unsafe {
                _mm256_storeu_ps(
                    conv1_m0_full_grad_out
                        .as_mut_ptr()
                        .add(full_base + 3 * channels + ch),
                    _mm256_fmadd_ps(grad_l1, gate_l1, full_l1),
                );
                _mm256_storeu_ps(
                    conv1_m0_full_grad_out.as_mut_ptr().add(full_base + ch),
                    _mm256_fmadd_ps(grad_l1, conv1_l1, full_gate_l1),
                );
            }

            let grad_l2 = unsafe {
                _mm256_loadu_ps(activated_m0_grad.as_ptr().add(act_base + 2 * channels + ch))
            };
            let gate_l2 =
                unsafe { _mm256_loadu_ps(gate_sigmoids.as_ptr().add(gate_base + channels + ch)) };
            let conv1_l2 = unsafe {
                _mm256_loadu_ps(
                    conv1_out_l_major
                        .as_ptr()
                        .add(conv1_base + 2 * channels + ch),
                )
            };
            let full_l2 = unsafe {
                _mm256_loadu_ps(
                    conv1_m0_full_grad_out
                        .as_ptr()
                        .add(full_base + 4 * channels + ch),
                )
            };
            let full_gate_l2 = unsafe {
                _mm256_loadu_ps(
                    conv1_m0_full_grad_out
                        .as_ptr()
                        .add(full_base + channels + ch),
                )
            };
            unsafe {
                _mm256_storeu_ps(
                    conv1_m0_full_grad_out
                        .as_mut_ptr()
                        .add(full_base + 4 * channels + ch),
                    _mm256_fmadd_ps(grad_l2, gate_l2, full_l2),
                );
                _mm256_storeu_ps(
                    conv1_m0_full_grad_out
                        .as_mut_ptr()
                        .add(full_base + channels + ch),
                    _mm256_fmadd_ps(grad_l2, conv1_l2, full_gate_l2),
                );
            }
            ch += 8;
        }
        while ch < channels {
            let grad_scalar = activated_m0_grad[act_base + ch];
            conv1_m0_full_grad_out[full_base + 2 * channels + ch] +=
                grad_scalar * scalar_silu_derivatives[node * channels + ch];

            let grad_l1 = activated_m0_grad[act_base + channels + ch];
            conv1_m0_full_grad_out[full_base + 3 * channels + ch] +=
                grad_l1 * gate_sigmoids[gate_base + ch];
            conv1_m0_full_grad_out[full_base + ch] +=
                grad_l1 * conv1_out_l_major[conv1_base + channels + ch];

            let grad_l2 = activated_m0_grad[act_base + 2 * channels + ch];
            conv1_m0_full_grad_out[full_base + 4 * channels + ch] +=
                grad_l2 * gate_sigmoids[gate_base + channels + ch];
            conv1_m0_full_grad_out[full_base + channels + ch] +=
                grad_l2 * conv1_out_l_major[conv1_base + 2 * channels + ch];
            ch += 1;
        }
    }
}

// ── Vectorized UMA l=2,m=2 SO2 m=0 forward gate + scatter ─────────────────────
// Fuses the sigmoid gates, SiLU value+derivative, and conv1/conv2 buffer scatter
// of the SO2 m=0 forward into one pass with a vectorized exp(). The scalar path
// this replaces is dominated by libm exp() (sigmoid/silu); on Linux there is no
// vectorized exp, so this is the hottest non-GEMM SO2 substage.
//
// sigmoid(x) = 1/(1+exp(-x)); silu(x) = x*sigmoid(x);
// silu'(x) = sig*(1 + x*(1 - sig)). Mathematically identical to the scalar
// reference (which only branches for overflow safety); the vectorized exp clamps
// its argument to [-88, 88] so no Inf is produced.

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[inline]
fn scalar_sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
#[allow(clippy::excessive_precision)]
unsafe fn exp_ps_avx512(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::{
        _mm512_add_ps, _mm512_cvtepi32_ps, _mm512_cvtps_epi32, _mm512_fmadd_ps, _mm512_fnmadd_ps,
        _mm512_max_ps, _mm512_min_ps, _mm512_mul_ps, _mm512_scalef_ps, _mm512_set1_ps,
    };
    const LOG2EF: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_38;
    const LN2_LO: f32 = -2.121_944_4e-4;
    // Cephes expf minimax polynomial coefficients.
    const P0: f32 = 1.987_569_15e-4;
    const P1: f32 = 1.398_199_95e-3;
    const P2: f32 = 8.333_452e-3;
    const P3: f32 = 4.166_579_6e-2;
    const P4: f32 = 1.666_666_5e-1;
    const P5: f32 = 5.000_000_1e-1;
    let xc = _mm512_min_ps(
        _mm512_max_ps(x, _mm512_set1_ps(-88.0)),
        _mm512_set1_ps(88.0),
    );
    let n = _mm512_cvtps_epi32(_mm512_mul_ps(xc, _mm512_set1_ps(LOG2EF)));
    let fx = _mm512_cvtepi32_ps(n);
    let mut r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(LN2_HI), xc);
    r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(LN2_LO), r);
    let mut y = _mm512_set1_ps(P0);
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(P1));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(P2));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(P3));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(P4));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(P5));
    let z = _mm512_mul_ps(r, r);
    y = _mm512_fmadd_ps(y, z, r);
    y = _mm512_add_ps(y, _mm512_set1_ps(1.0));
    _mm512_scalef_ps(y, fx)
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn sigmoid_ps_avx512(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::{_mm512_add_ps, _mm512_div_ps, _mm512_set1_ps, _mm512_sub_ps};
    let one = _mm512_set1_ps(1.0);
    let neg = _mm512_sub_ps(_mm512_set1_ps(0.0), x);
    let e = unsafe { exp_ps_avx512(neg) };
    _mm512_div_ps(one, _mm512_add_ps(one, e))
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn uma_f32_l2_m2_m0_gate_scatter_forward_avx512(
    nodes: usize,
    channels: usize,
    conv1_m0_dense_out: &[f32],
    gate_sigmoids: &mut [f32],
    scalar_silu_derivatives: &mut [f32],
    conv1_out_l_major: &mut [f32],
    conv2_m0_input: &mut [f32],
) {
    use std::arch::x86_64::{
        _mm512_add_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_set1_ps, _mm512_storeu_ps,
        _mm512_sub_ps,
    };
    let m0_out_dim = 5 * channels;
    let one = _mm512_set1_ps(1.0);
    for node in 0..nodes {
        let dense_base = node * m0_out_dim;
        let gate_base = node * 2 * channels;
        let deriv_base = node * channels;
        let conv1_base = node * 9 * channels;
        let conv2_base = node * 3 * channels;
        let mut ch = 0usize;
        while ch + 16 <= channels {
            let gate1_raw =
                unsafe { _mm512_loadu_ps(conv1_m0_dense_out.as_ptr().add(dense_base + ch)) };
            let gate2_raw = unsafe {
                _mm512_loadu_ps(conv1_m0_dense_out.as_ptr().add(dense_base + channels + ch))
            };
            let scalar = unsafe {
                _mm512_loadu_ps(
                    conv1_m0_dense_out
                        .as_ptr()
                        .add(dense_base + 2 * channels + ch),
                )
            };
            let l1 = unsafe {
                _mm512_loadu_ps(
                    conv1_m0_dense_out
                        .as_ptr()
                        .add(dense_base + 3 * channels + ch),
                )
            };
            let l2 = unsafe {
                _mm512_loadu_ps(
                    conv1_m0_dense_out
                        .as_ptr()
                        .add(dense_base + 4 * channels + ch),
                )
            };

            let gate1 = unsafe { sigmoid_ps_avx512(gate1_raw) };
            let gate2 = unsafe { sigmoid_ps_avx512(gate2_raw) };
            let sig_s = unsafe { sigmoid_ps_avx512(scalar) };
            let silu = _mm512_mul_ps(scalar, sig_s);
            // deriv = sig_s + scalar*sig_s*(1 - sig_s) = sig_s*(1 + scalar*(1 - sig_s))
            let one_minus = _mm512_sub_ps(one, sig_s);
            let t = _mm512_mul_ps(scalar, one_minus);
            let deriv = _mm512_mul_ps(sig_s, _mm512_add_ps(one, t));

            unsafe {
                _mm512_storeu_ps(gate_sigmoids.as_mut_ptr().add(gate_base + ch), gate1);
                _mm512_storeu_ps(
                    gate_sigmoids.as_mut_ptr().add(gate_base + channels + ch),
                    gate2,
                );
                _mm512_storeu_ps(
                    scalar_silu_derivatives.as_mut_ptr().add(deriv_base + ch),
                    deriv,
                );
                _mm512_storeu_ps(conv1_out_l_major.as_mut_ptr().add(conv1_base + ch), scalar);
                _mm512_storeu_ps(
                    conv1_out_l_major
                        .as_mut_ptr()
                        .add(conv1_base + channels + ch),
                    l1,
                );
                _mm512_storeu_ps(
                    conv1_out_l_major
                        .as_mut_ptr()
                        .add(conv1_base + 2 * channels + ch),
                    l2,
                );
                _mm512_storeu_ps(conv2_m0_input.as_mut_ptr().add(conv2_base + ch), silu);
                _mm512_storeu_ps(
                    conv2_m0_input.as_mut_ptr().add(conv2_base + channels + ch),
                    _mm512_mul_ps(l1, gate1),
                );
                _mm512_storeu_ps(
                    conv2_m0_input
                        .as_mut_ptr()
                        .add(conv2_base + 2 * channels + ch),
                    _mm512_mul_ps(l2, gate2),
                );
            }
            ch += 16;
        }
        while ch < channels {
            let gate1 = scalar_sigmoid_f32(conv1_m0_dense_out[dense_base + ch]);
            let gate2 = scalar_sigmoid_f32(conv1_m0_dense_out[dense_base + channels + ch]);
            let scalar = conv1_m0_dense_out[dense_base + 2 * channels + ch];
            let l1 = conv1_m0_dense_out[dense_base + 3 * channels + ch];
            let l2 = conv1_m0_dense_out[dense_base + 4 * channels + ch];
            let sig_s = scalar_sigmoid_f32(scalar);
            gate_sigmoids[gate_base + ch] = gate1;
            gate_sigmoids[gate_base + channels + ch] = gate2;
            scalar_silu_derivatives[deriv_base + ch] = sig_s * (1.0 + scalar * (1.0 - sig_s));
            conv1_out_l_major[conv1_base + ch] = scalar;
            conv1_out_l_major[conv1_base + channels + ch] = l1;
            conv1_out_l_major[conv1_base + 2 * channels + ch] = l2;
            conv2_m0_input[conv2_base + ch] = scalar * sig_s;
            conv2_m0_input[conv2_base + channels + ch] = l1 * gate1;
            conv2_m0_input[conv2_base + 2 * channels + ch] = l2 * gate2;
            ch += 1;
        }
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
#[allow(clippy::excessive_precision)]
unsafe fn exp_ps_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::{
        _mm256_add_epi32, _mm256_add_ps, _mm256_castsi256_ps, _mm256_cvtepi32_ps,
        _mm256_cvtps_epi32, _mm256_fmadd_ps, _mm256_fnmadd_ps, _mm256_max_ps, _mm256_min_ps,
        _mm256_mul_ps, _mm256_set1_epi32, _mm256_set1_ps, _mm256_slli_epi32,
    };
    const LOG2EF: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_38;
    const LN2_LO: f32 = -2.121_944_4e-4;
    const P0: f32 = 1.987_569_15e-4;
    const P1: f32 = 1.398_199_95e-3;
    const P2: f32 = 8.333_452e-3;
    const P3: f32 = 4.166_579_6e-2;
    const P4: f32 = 1.666_666_5e-1;
    const P5: f32 = 5.000_000_1e-1;
    let xc = _mm256_min_ps(
        _mm256_max_ps(x, _mm256_set1_ps(-88.0)),
        _mm256_set1_ps(88.0),
    );
    let ni = _mm256_cvtps_epi32(_mm256_mul_ps(xc, _mm256_set1_ps(LOG2EF)));
    let fx = _mm256_cvtepi32_ps(ni);
    let mut r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(LN2_HI), xc);
    r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(LN2_LO), r);
    let mut y = _mm256_set1_ps(P0);
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(P1));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(P2));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(P3));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(P4));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(P5));
    let z = _mm256_mul_ps(r, r);
    y = _mm256_fmadd_ps(y, z, r);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0));
    let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
        ni,
        _mm256_set1_epi32(127),
    )));
    _mm256_mul_ps(y, pow2n)
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn sigmoid_ps_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::{_mm256_add_ps, _mm256_div_ps, _mm256_set1_ps, _mm256_sub_ps};
    let one = _mm256_set1_ps(1.0);
    let neg = _mm256_sub_ps(_mm256_set1_ps(0.0), x);
    let e = unsafe { exp_ps_avx2(neg) };
    _mm256_div_ps(one, _mm256_add_ps(one, e))
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn uma_f32_l2_m2_m0_gate_scatter_forward_avx2(
    nodes: usize,
    channels: usize,
    conv1_m0_dense_out: &[f32],
    gate_sigmoids: &mut [f32],
    scalar_silu_derivatives: &mut [f32],
    conv1_out_l_major: &mut [f32],
    conv2_m0_input: &mut [f32],
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
        _mm256_sub_ps,
    };
    let m0_out_dim = 5 * channels;
    let one = _mm256_set1_ps(1.0);
    for node in 0..nodes {
        let dense_base = node * m0_out_dim;
        let gate_base = node * 2 * channels;
        let deriv_base = node * channels;
        let conv1_base = node * 9 * channels;
        let conv2_base = node * 3 * channels;
        let mut ch = 0usize;
        while ch + 8 <= channels {
            let gate1_raw =
                unsafe { _mm256_loadu_ps(conv1_m0_dense_out.as_ptr().add(dense_base + ch)) };
            let gate2_raw = unsafe {
                _mm256_loadu_ps(conv1_m0_dense_out.as_ptr().add(dense_base + channels + ch))
            };
            let scalar = unsafe {
                _mm256_loadu_ps(
                    conv1_m0_dense_out
                        .as_ptr()
                        .add(dense_base + 2 * channels + ch),
                )
            };
            let l1 = unsafe {
                _mm256_loadu_ps(
                    conv1_m0_dense_out
                        .as_ptr()
                        .add(dense_base + 3 * channels + ch),
                )
            };
            let l2 = unsafe {
                _mm256_loadu_ps(
                    conv1_m0_dense_out
                        .as_ptr()
                        .add(dense_base + 4 * channels + ch),
                )
            };
            let gate1 = unsafe { sigmoid_ps_avx2(gate1_raw) };
            let gate2 = unsafe { sigmoid_ps_avx2(gate2_raw) };
            let sig_s = unsafe { sigmoid_ps_avx2(scalar) };
            let silu = _mm256_mul_ps(scalar, sig_s);
            let one_minus = _mm256_sub_ps(one, sig_s);
            let t = _mm256_mul_ps(scalar, one_minus);
            let deriv = _mm256_mul_ps(sig_s, _mm256_add_ps(one, t));
            unsafe {
                _mm256_storeu_ps(gate_sigmoids.as_mut_ptr().add(gate_base + ch), gate1);
                _mm256_storeu_ps(
                    gate_sigmoids.as_mut_ptr().add(gate_base + channels + ch),
                    gate2,
                );
                _mm256_storeu_ps(
                    scalar_silu_derivatives.as_mut_ptr().add(deriv_base + ch),
                    deriv,
                );
                _mm256_storeu_ps(conv1_out_l_major.as_mut_ptr().add(conv1_base + ch), scalar);
                _mm256_storeu_ps(
                    conv1_out_l_major
                        .as_mut_ptr()
                        .add(conv1_base + channels + ch),
                    l1,
                );
                _mm256_storeu_ps(
                    conv1_out_l_major
                        .as_mut_ptr()
                        .add(conv1_base + 2 * channels + ch),
                    l2,
                );
                _mm256_storeu_ps(conv2_m0_input.as_mut_ptr().add(conv2_base + ch), silu);
                _mm256_storeu_ps(
                    conv2_m0_input.as_mut_ptr().add(conv2_base + channels + ch),
                    _mm256_mul_ps(l1, gate1),
                );
                _mm256_storeu_ps(
                    conv2_m0_input
                        .as_mut_ptr()
                        .add(conv2_base + 2 * channels + ch),
                    _mm256_mul_ps(l2, gate2),
                );
            }
            ch += 8;
        }
        while ch < channels {
            let gate1 = scalar_sigmoid_f32(conv1_m0_dense_out[dense_base + ch]);
            let gate2 = scalar_sigmoid_f32(conv1_m0_dense_out[dense_base + channels + ch]);
            let scalar = conv1_m0_dense_out[dense_base + 2 * channels + ch];
            let l1 = conv1_m0_dense_out[dense_base + 3 * channels + ch];
            let l2 = conv1_m0_dense_out[dense_base + 4 * channels + ch];
            let sig_s = scalar_sigmoid_f32(scalar);
            gate_sigmoids[gate_base + ch] = gate1;
            gate_sigmoids[gate_base + channels + ch] = gate2;
            scalar_silu_derivatives[deriv_base + ch] = sig_s * (1.0 + scalar * (1.0 - sig_s));
            conv1_out_l_major[conv1_base + ch] = scalar;
            conv1_out_l_major[conv1_base + channels + ch] = l1;
            conv1_out_l_major[conv1_base + 2 * channels + ch] = l2;
            conv2_m0_input[conv2_base + ch] = scalar * sig_s;
            conv2_m0_input[conv2_base + channels + ch] = l1 * gate1;
            conv2_m0_input[conv2_base + 2 * channels + ch] = l2 * gate2;
            ch += 1;
        }
    }
}

/// Vectorized UMA l=2,m=2 SO2 m=0 forward gate+scatter. Returns true when a SIMD
/// kernel handled the work; false (e.g. non-x86 or SIMD disabled) means the
/// caller must run its scalar fallback.
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn uma_f32_try_l2_m2_m0_gate_scatter_forward(
    nodes: usize,
    channels: usize,
    conv1_m0_dense_out: &[f32],
    gate_sigmoids: &mut [f32],
    scalar_silu_derivatives: &mut [f32],
    conv1_out_l_major: &mut [f32],
    conv2_m0_input: &mut [f32],
) -> bool {
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    {
        let m0_out_dim = 5 * channels;
        debug_assert!(conv1_m0_dense_out.len() >= nodes * m0_out_dim);
        debug_assert!(gate_sigmoids.len() >= nodes * 2 * channels);
        debug_assert!(scalar_silu_derivatives.len() >= nodes * channels);
        debug_assert!(conv1_out_l_major.len() >= nodes * 9 * channels);
        debug_assert!(conv2_m0_input.len() >= nodes * 3 * channels);
        if nodes == 0 {
            return true;
        }
        if channels.is_multiple_of(8) {
            if uma_elementwise_avx512_enabled() && channels.is_multiple_of(16) {
                unsafe {
                    uma_f32_l2_m2_m0_gate_scatter_forward_avx512(
                        nodes,
                        channels,
                        conv1_m0_dense_out,
                        gate_sigmoids,
                        scalar_silu_derivatives,
                        conv1_out_l_major,
                        conv2_m0_input,
                    );
                }
                return true;
            }
            // AVX2 sigmoid is divide-unit-bound and did not beat scalar on the
            // AVX-512 target hardware, so it is only used as a fallback on
            // machines without AVX-512.
            if !std::arch::is_x86_feature_detected!("avx512f") && uma_so2_f32_avx2_fma_enabled() {
                unsafe {
                    uma_f32_l2_m2_m0_gate_scatter_forward_avx2(
                        nodes,
                        channels,
                        conv1_m0_dense_out,
                        gate_sigmoids,
                        scalar_silu_derivatives,
                        conv1_out_l_major,
                        conv2_m0_input,
                    );
                }
                return true;
            }
        }
    }
    #[cfg(not(all(feature = "parallel", target_arch = "x86_64")))]
    {
        let _ = nodes;
        let _ = channels;
        let _ = conv1_m0_dense_out;
        let _ = gate_sigmoids;
        let _ = scalar_silu_derivatives;
        let _ = conv1_out_l_major;
        let _ = conv2_m0_input;
    }
    false
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn uma_f32_try_l2_m2_order1_gate_vjp(
    nodes: usize,
    channels: usize,
    activated_block_grad: &[f32],
    conv1_out_l_major: &[f32],
    gate_sigmoids: &[f32],
    conv1_m0_full_grad_out: &mut [f32],
    conv1_block_grad_out: &mut [f32],
) -> bool {
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    {
        let block_stride = 4 * channels;
        let out_node_stride = 9 * channels;
        let gate_node_stride = 2 * channels;
        let m0_out_dim = 5 * channels;
        debug_assert!(activated_block_grad.len() >= nodes * block_stride);
        debug_assert!(conv1_out_l_major.len() >= nodes * out_node_stride);
        debug_assert!(gate_sigmoids.len() >= nodes * gate_node_stride);
        debug_assert!(conv1_m0_full_grad_out.len() >= nodes * m0_out_dim);
        debug_assert!(conv1_block_grad_out.len() >= nodes * block_stride);

        if nodes >= 16 && channels.is_multiple_of(8) {
            if uma_so2_f32_avx512_enabled() && channels.is_multiple_of(16) {
                unsafe {
                    uma_f32_l2_m2_order1_gate_vjp_avx512(
                        nodes,
                        channels,
                        activated_block_grad,
                        conv1_out_l_major,
                        gate_sigmoids,
                        conv1_m0_full_grad_out,
                        conv1_block_grad_out,
                    );
                }
                return true;
            }
            if uma_so2_f32_avx2_fma_enabled() {
                unsafe {
                    uma_f32_l2_m2_order1_gate_vjp_avx2(
                        nodes,
                        channels,
                        activated_block_grad,
                        conv1_out_l_major,
                        gate_sigmoids,
                        conv1_m0_full_grad_out,
                        conv1_block_grad_out,
                    );
                }
                return true;
            }
        }
    }
    #[cfg(not(all(feature = "parallel", target_arch = "x86_64")))]
    {
        let _ = nodes;
        let _ = channels;
        let _ = activated_block_grad;
        let _ = conv1_out_l_major;
        let _ = gate_sigmoids;
        let _ = conv1_m0_full_grad_out;
        let _ = conv1_block_grad_out;
    }
    false
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn uma_f32_l2_m2_order1_gate_vjp_avx512(
    nodes: usize,
    channels: usize,
    activated_block_grad: &[f32],
    conv1_out_l_major: &[f32],
    gate_sigmoids: &[f32],
    conv1_m0_full_grad_out: &mut [f32],
    conv1_block_grad_out: &mut [f32],
) {
    use std::arch::x86_64::{_mm512_fmadd_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_storeu_ps};

    let block_stride = 4 * channels;
    let out_node_stride = 9 * channels;
    let gate_node_stride = 2 * channels;
    let m0_out_dim = 5 * channels;
    for node in 0..nodes {
        let block_base = node * block_stride;
        let conv1_base = node * out_node_stride + 3 * channels;
        let gate_base = node * gate_node_stride;
        let full_base = node * m0_out_dim;
        let mut ch = 0usize;
        while ch + 16 <= channels {
            let gate1 = unsafe { _mm512_loadu_ps(gate_sigmoids.as_ptr().add(gate_base + ch)) };
            let gate2 =
                unsafe { _mm512_loadu_ps(gate_sigmoids.as_ptr().add(gate_base + channels + ch)) };
            let grad3 =
                unsafe { _mm512_loadu_ps(activated_block_grad.as_ptr().add(block_base + ch)) };
            let grad7 = unsafe {
                _mm512_loadu_ps(
                    activated_block_grad
                        .as_ptr()
                        .add(block_base + channels + ch),
                )
            };
            let grad1 = unsafe {
                _mm512_loadu_ps(
                    activated_block_grad
                        .as_ptr()
                        .add(block_base + 2 * channels + ch),
                )
            };
            let grad5 = unsafe {
                _mm512_loadu_ps(
                    activated_block_grad
                        .as_ptr()
                        .add(block_base + 3 * channels + ch),
                )
            };
            unsafe {
                _mm512_storeu_ps(
                    conv1_block_grad_out.as_mut_ptr().add(block_base + ch),
                    _mm512_mul_ps(grad3, gate1),
                );
                _mm512_storeu_ps(
                    conv1_block_grad_out
                        .as_mut_ptr()
                        .add(block_base + channels + ch),
                    _mm512_mul_ps(grad7, gate2),
                );
                _mm512_storeu_ps(
                    conv1_block_grad_out
                        .as_mut_ptr()
                        .add(block_base + 2 * channels + ch),
                    _mm512_mul_ps(grad1, gate1),
                );
                _mm512_storeu_ps(
                    conv1_block_grad_out
                        .as_mut_ptr()
                        .add(block_base + 3 * channels + ch),
                    _mm512_mul_ps(grad5, gate2),
                );
            }

            let conv3 = unsafe { _mm512_loadu_ps(conv1_out_l_major.as_ptr().add(conv1_base + ch)) };
            let conv1 = unsafe {
                _mm512_loadu_ps(
                    conv1_out_l_major
                        .as_ptr()
                        .add(conv1_base + 2 * channels + ch),
                )
            };
            let full1 =
                unsafe { _mm512_loadu_ps(conv1_m0_full_grad_out.as_ptr().add(full_base + ch)) };
            let acc1 = _mm512_fmadd_ps(grad3, conv3, _mm512_fmadd_ps(grad1, conv1, full1));

            let conv7 = unsafe {
                _mm512_loadu_ps(conv1_out_l_major.as_ptr().add(conv1_base + channels + ch))
            };
            let conv5 = unsafe {
                _mm512_loadu_ps(
                    conv1_out_l_major
                        .as_ptr()
                        .add(conv1_base + 3 * channels + ch),
                )
            };
            let full2 = unsafe {
                _mm512_loadu_ps(
                    conv1_m0_full_grad_out
                        .as_ptr()
                        .add(full_base + channels + ch),
                )
            };
            let acc2 = _mm512_fmadd_ps(grad7, conv7, _mm512_fmadd_ps(grad5, conv5, full2));
            unsafe {
                _mm512_storeu_ps(
                    conv1_m0_full_grad_out.as_mut_ptr().add(full_base + ch),
                    acc1,
                );
                _mm512_storeu_ps(
                    conv1_m0_full_grad_out
                        .as_mut_ptr()
                        .add(full_base + channels + ch),
                    acc2,
                );
            }
            ch += 16;
        }
        while ch < channels {
            let gate1 = gate_sigmoids[gate_base + ch];
            let gate2 = gate_sigmoids[gate_base + channels + ch];
            let grad3 = activated_block_grad[block_base + ch];
            let grad7 = activated_block_grad[block_base + channels + ch];
            let grad1 = activated_block_grad[block_base + 2 * channels + ch];
            let grad5 = activated_block_grad[block_base + 3 * channels + ch];
            conv1_block_grad_out[block_base + ch] = grad3 * gate1;
            conv1_block_grad_out[block_base + channels + ch] = grad7 * gate2;
            conv1_block_grad_out[block_base + 2 * channels + ch] = grad1 * gate1;
            conv1_block_grad_out[block_base + 3 * channels + ch] = grad5 * gate2;
            conv1_m0_full_grad_out[full_base + ch] += grad3 * conv1_out_l_major[conv1_base + ch]
                + grad1 * conv1_out_l_major[conv1_base + 2 * channels + ch];
            conv1_m0_full_grad_out[full_base + channels + ch] += grad7
                * conv1_out_l_major[conv1_base + channels + ch]
                + grad5 * conv1_out_l_major[conv1_base + 3 * channels + ch];
            ch += 1;
        }
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn uma_f32_l2_m2_order1_gate_vjp_avx2(
    nodes: usize,
    channels: usize,
    activated_block_grad: &[f32],
    conv1_out_l_major: &[f32],
    gate_sigmoids: &[f32],
    conv1_m0_full_grad_out: &mut [f32],
    conv1_block_grad_out: &mut [f32],
) {
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps};

    let block_stride = 4 * channels;
    let out_node_stride = 9 * channels;
    let gate_node_stride = 2 * channels;
    let m0_out_dim = 5 * channels;
    for node in 0..nodes {
        let block_base = node * block_stride;
        let conv1_base = node * out_node_stride + 3 * channels;
        let gate_base = node * gate_node_stride;
        let full_base = node * m0_out_dim;
        let mut ch = 0usize;
        while ch + 8 <= channels {
            let gate1 = unsafe { _mm256_loadu_ps(gate_sigmoids.as_ptr().add(gate_base + ch)) };
            let gate2 =
                unsafe { _mm256_loadu_ps(gate_sigmoids.as_ptr().add(gate_base + channels + ch)) };
            let grad3 =
                unsafe { _mm256_loadu_ps(activated_block_grad.as_ptr().add(block_base + ch)) };
            let grad7 = unsafe {
                _mm256_loadu_ps(
                    activated_block_grad
                        .as_ptr()
                        .add(block_base + channels + ch),
                )
            };
            let grad1 = unsafe {
                _mm256_loadu_ps(
                    activated_block_grad
                        .as_ptr()
                        .add(block_base + 2 * channels + ch),
                )
            };
            let grad5 = unsafe {
                _mm256_loadu_ps(
                    activated_block_grad
                        .as_ptr()
                        .add(block_base + 3 * channels + ch),
                )
            };
            unsafe {
                _mm256_storeu_ps(
                    conv1_block_grad_out.as_mut_ptr().add(block_base + ch),
                    _mm256_mul_ps(grad3, gate1),
                );
                _mm256_storeu_ps(
                    conv1_block_grad_out
                        .as_mut_ptr()
                        .add(block_base + channels + ch),
                    _mm256_mul_ps(grad7, gate2),
                );
                _mm256_storeu_ps(
                    conv1_block_grad_out
                        .as_mut_ptr()
                        .add(block_base + 2 * channels + ch),
                    _mm256_mul_ps(grad1, gate1),
                );
                _mm256_storeu_ps(
                    conv1_block_grad_out
                        .as_mut_ptr()
                        .add(block_base + 3 * channels + ch),
                    _mm256_mul_ps(grad5, gate2),
                );
            }

            let conv3 = unsafe { _mm256_loadu_ps(conv1_out_l_major.as_ptr().add(conv1_base + ch)) };
            let conv1 = unsafe {
                _mm256_loadu_ps(
                    conv1_out_l_major
                        .as_ptr()
                        .add(conv1_base + 2 * channels + ch),
                )
            };
            let full1 =
                unsafe { _mm256_loadu_ps(conv1_m0_full_grad_out.as_ptr().add(full_base + ch)) };
            let acc1 = _mm256_fmadd_ps(grad3, conv3, _mm256_fmadd_ps(grad1, conv1, full1));

            let conv7 = unsafe {
                _mm256_loadu_ps(conv1_out_l_major.as_ptr().add(conv1_base + channels + ch))
            };
            let conv5 = unsafe {
                _mm256_loadu_ps(
                    conv1_out_l_major
                        .as_ptr()
                        .add(conv1_base + 3 * channels + ch),
                )
            };
            let full2 = unsafe {
                _mm256_loadu_ps(
                    conv1_m0_full_grad_out
                        .as_ptr()
                        .add(full_base + channels + ch),
                )
            };
            let acc2 = _mm256_fmadd_ps(grad7, conv7, _mm256_fmadd_ps(grad5, conv5, full2));
            unsafe {
                _mm256_storeu_ps(
                    conv1_m0_full_grad_out.as_mut_ptr().add(full_base + ch),
                    acc1,
                );
                _mm256_storeu_ps(
                    conv1_m0_full_grad_out
                        .as_mut_ptr()
                        .add(full_base + channels + ch),
                    acc2,
                );
            }
            ch += 8;
        }
        while ch < channels {
            let gate1 = gate_sigmoids[gate_base + ch];
            let gate2 = gate_sigmoids[gate_base + channels + ch];
            let grad3 = activated_block_grad[block_base + ch];
            let grad7 = activated_block_grad[block_base + channels + ch];
            let grad1 = activated_block_grad[block_base + 2 * channels + ch];
            let grad5 = activated_block_grad[block_base + 3 * channels + ch];
            conv1_block_grad_out[block_base + ch] = grad3 * gate1;
            conv1_block_grad_out[block_base + channels + ch] = grad7 * gate2;
            conv1_block_grad_out[block_base + 2 * channels + ch] = grad1 * gate1;
            conv1_block_grad_out[block_base + 3 * channels + ch] = grad5 * gate2;
            conv1_m0_full_grad_out[full_base + ch] += grad3 * conv1_out_l_major[conv1_base + ch]
                + grad1 * conv1_out_l_major[conv1_base + 2 * channels + ch];
            conv1_m0_full_grad_out[full_base + channels + ch] += grad7
                * conv1_out_l_major[conv1_base + channels + ch]
                + grad5 * conv1_out_l_major[conv1_base + 3 * channels + ch];
            ch += 1;
        }
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn uma_f32_try_l2_m2_order2_gate_vjp(
    nodes: usize,
    channels: usize,
    activated_block_grad: &[f32],
    conv1_out_l_major: &[f32],
    gate_sigmoids: &[f32],
    order2_m0_gate_grad: &mut [f32],
    conv1_block_grad_out: &mut [f32],
) -> bool {
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    {
        let block_stride = 2 * channels;
        let out_node_stride = 9 * channels;
        let gate_node_stride = 2 * channels;
        debug_assert!(activated_block_grad.len() >= nodes * block_stride);
        debug_assert!(conv1_out_l_major.len() >= nodes * out_node_stride);
        debug_assert!(gate_sigmoids.len() >= nodes * gate_node_stride);
        debug_assert!(order2_m0_gate_grad.len() >= nodes * channels);
        debug_assert!(conv1_block_grad_out.len() >= nodes * block_stride);

        if nodes >= 16 && channels.is_multiple_of(8) {
            if uma_so2_f32_avx512_enabled() && channels.is_multiple_of(16) {
                unsafe {
                    uma_f32_l2_m2_order2_gate_vjp_avx512(
                        nodes,
                        channels,
                        activated_block_grad,
                        conv1_out_l_major,
                        gate_sigmoids,
                        order2_m0_gate_grad,
                        conv1_block_grad_out,
                    );
                }
                return true;
            }
            if uma_so2_f32_avx2_fma_enabled() {
                unsafe {
                    uma_f32_l2_m2_order2_gate_vjp_avx2(
                        nodes,
                        channels,
                        activated_block_grad,
                        conv1_out_l_major,
                        gate_sigmoids,
                        order2_m0_gate_grad,
                        conv1_block_grad_out,
                    );
                }
                return true;
            }
        }
    }
    #[cfg(not(all(feature = "parallel", target_arch = "x86_64")))]
    {
        let _ = nodes;
        let _ = channels;
        let _ = activated_block_grad;
        let _ = conv1_out_l_major;
        let _ = gate_sigmoids;
        let _ = order2_m0_gate_grad;
        let _ = conv1_block_grad_out;
    }
    false
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn uma_f32_l2_m2_order2_gate_vjp_avx512(
    nodes: usize,
    channels: usize,
    activated_block_grad: &[f32],
    conv1_out_l_major: &[f32],
    gate_sigmoids: &[f32],
    order2_m0_gate_grad: &mut [f32],
    conv1_block_grad_out: &mut [f32],
) {
    use std::arch::x86_64::{_mm512_fmadd_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_storeu_ps};

    let block_stride = 2 * channels;
    let out_node_stride = 9 * channels;
    let gate_node_stride = 2 * channels;
    for node in 0..nodes {
        let block_base = node * block_stride;
        let conv1_base = node * out_node_stride + 7 * channels;
        let gate_base = node * gate_node_stride;
        let gate_grad_base = node * channels;
        let mut ch = 0usize;
        while ch + 16 <= channels {
            let gate2 =
                unsafe { _mm512_loadu_ps(gate_sigmoids.as_ptr().add(gate_base + channels + ch)) };
            let grad8 =
                unsafe { _mm512_loadu_ps(activated_block_grad.as_ptr().add(block_base + ch)) };
            let grad4 = unsafe {
                _mm512_loadu_ps(
                    activated_block_grad
                        .as_ptr()
                        .add(block_base + channels + ch),
                )
            };
            unsafe {
                _mm512_storeu_ps(
                    conv1_block_grad_out.as_mut_ptr().add(block_base + ch),
                    _mm512_mul_ps(grad8, gate2),
                );
                _mm512_storeu_ps(
                    conv1_block_grad_out
                        .as_mut_ptr()
                        .add(block_base + channels + ch),
                    _mm512_mul_ps(grad4, gate2),
                );
            }

            let conv8 = unsafe { _mm512_loadu_ps(conv1_out_l_major.as_ptr().add(conv1_base + ch)) };
            let conv4 = unsafe {
                _mm512_loadu_ps(conv1_out_l_major.as_ptr().add(conv1_base + channels + ch))
            };
            let acc = _mm512_fmadd_ps(grad8, conv8, _mm512_mul_ps(grad4, conv4));
            unsafe {
                _mm512_storeu_ps(
                    order2_m0_gate_grad.as_mut_ptr().add(gate_grad_base + ch),
                    acc,
                );
            }
            ch += 16;
        }
        while ch < channels {
            let gate2 = gate_sigmoids[gate_base + channels + ch];
            let grad8 = activated_block_grad[block_base + ch];
            let grad4 = activated_block_grad[block_base + channels + ch];
            conv1_block_grad_out[block_base + ch] = grad8 * gate2;
            conv1_block_grad_out[block_base + channels + ch] = grad4 * gate2;
            order2_m0_gate_grad[gate_grad_base + ch] = grad8 * conv1_out_l_major[conv1_base + ch]
                + grad4 * conv1_out_l_major[conv1_base + channels + ch];
            ch += 1;
        }
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn uma_f32_l2_m2_order2_gate_vjp_avx2(
    nodes: usize,
    channels: usize,
    activated_block_grad: &[f32],
    conv1_out_l_major: &[f32],
    gate_sigmoids: &[f32],
    order2_m0_gate_grad: &mut [f32],
    conv1_block_grad_out: &mut [f32],
) {
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps};

    let block_stride = 2 * channels;
    let out_node_stride = 9 * channels;
    let gate_node_stride = 2 * channels;
    for node in 0..nodes {
        let block_base = node * block_stride;
        let conv1_base = node * out_node_stride + 7 * channels;
        let gate_base = node * gate_node_stride;
        let gate_grad_base = node * channels;
        let mut ch = 0usize;
        while ch + 8 <= channels {
            let gate2 =
                unsafe { _mm256_loadu_ps(gate_sigmoids.as_ptr().add(gate_base + channels + ch)) };
            let grad8 =
                unsafe { _mm256_loadu_ps(activated_block_grad.as_ptr().add(block_base + ch)) };
            let grad4 = unsafe {
                _mm256_loadu_ps(
                    activated_block_grad
                        .as_ptr()
                        .add(block_base + channels + ch),
                )
            };
            unsafe {
                _mm256_storeu_ps(
                    conv1_block_grad_out.as_mut_ptr().add(block_base + ch),
                    _mm256_mul_ps(grad8, gate2),
                );
                _mm256_storeu_ps(
                    conv1_block_grad_out
                        .as_mut_ptr()
                        .add(block_base + channels + ch),
                    _mm256_mul_ps(grad4, gate2),
                );
            }

            let conv8 = unsafe { _mm256_loadu_ps(conv1_out_l_major.as_ptr().add(conv1_base + ch)) };
            let conv4 = unsafe {
                _mm256_loadu_ps(conv1_out_l_major.as_ptr().add(conv1_base + channels + ch))
            };
            let acc = _mm256_fmadd_ps(grad8, conv8, _mm256_mul_ps(grad4, conv4));
            unsafe {
                _mm256_storeu_ps(
                    order2_m0_gate_grad.as_mut_ptr().add(gate_grad_base + ch),
                    acc,
                );
            }
            ch += 8;
        }
        while ch < channels {
            let gate2 = gate_sigmoids[gate_base + channels + ch];
            let grad8 = activated_block_grad[block_base + ch];
            let grad4 = activated_block_grad[block_base + channels + ch];
            conv1_block_grad_out[block_base + ch] = grad8 * gate2;
            conv1_block_grad_out[block_base + channels + ch] = grad4 * gate2;
            order2_m0_gate_grad[gate_grad_base + ch] = grad8 * conv1_out_l_major[conv1_base + ch]
                + grad4 * conv1_out_l_major[conv1_base + channels + ch];
            ch += 1;
        }
    }
}

#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn uma_f32_try_l2_m2_order1_gate_conv1_vjp_stack(
    nodes: usize,
    conv1_in_channels: usize,
    channels: usize,
    activated_block_grad: &[f32],
    conv1_out_l_major: &[f32],
    gate_sigmoids: &[f32],
    fused_weight: &[f32],
    conv1_m0_full_grad_out: &mut [f32],
    grad_input: &mut [f32],
) -> bool {
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    {
        const UMA_L2_M2_CHANNELS: usize = 128;
        const UMA_L2_M2_CONV1_IN_CHANNELS: usize = 256;
        const ORDER1_SOURCES: usize = 4;
        const MAX_STACK_ROWS: usize = 8;
        const MAX_STACK_LEN: usize = MAX_STACK_ROWS * ORDER1_SOURCES * UMA_L2_M2_CHANNELS;

        if channels != UMA_L2_M2_CHANNELS || conv1_in_channels != UMA_L2_M2_CONV1_IN_CHANNELS {
            return false;
        }

        let k = ORDER1_SOURCES * channels;
        let n = ORDER1_SOURCES * conv1_in_channels;
        let out_node_stride = 9 * channels;
        let gate_node_stride = 2 * channels;
        let m0_out_dim = 5 * channels;
        debug_assert!(activated_block_grad.len() >= nodes * k);
        debug_assert!(conv1_out_l_major.len() >= nodes * out_node_stride);
        debug_assert!(gate_sigmoids.len() >= nodes * gate_node_stride);
        debug_assert!(fused_weight.len() >= k * n);
        debug_assert!(conv1_m0_full_grad_out.len() >= nodes * m0_out_dim);
        debug_assert!(grad_input.len() >= nodes * n);

        if !uma_so2_f32_row_kernel_should_run(nodes, k, n, true) {
            return false;
        }

        let trace_start = uma_blas_trace_start();
        let m0_addr = conv1_m0_full_grad_out.as_mut_ptr() as usize;
        if uma_so2_f32_avx512_enabled() {
            grad_input
                .par_chunks_mut(n * MAX_STACK_ROWS)
                .enumerate()
                .for_each(|(block, c_block)| {
                    let row_start = block * MAX_STACK_ROWS;
                    let block_rows = c_block.len() / n;
                    let mut a_scratch = std::mem::MaybeUninit::<[f32; MAX_STACK_LEN]>::uninit();
                    let a_scratch_ptr = a_scratch.as_mut_ptr().cast::<f32>();
                    unsafe {
                        // SAFETY: `a_scratch_ptr` points at `MAX_STACK_LEN`
                        // stack slots and the fill helper writes exactly the
                        // `block_rows * k` prefix before the row kernels read it.
                        fill_order1_gate_scaled_stack_uninit(
                            row_start,
                            block_rows,
                            channels,
                            activated_block_grad,
                            conv1_out_l_major,
                            gate_sigmoids,
                            m0_addr,
                            a_scratch_ptr,
                        );
                        let a_scratch = std::slice::from_raw_parts(a_scratch_ptr, block_rows * k);
                        if block_rows == MAX_STACK_ROWS {
                            uma_so2_f32_matmul_rows8x16_avx512(
                                a_scratch,
                                fused_weight,
                                k,
                                n,
                                c_block,
                                false,
                            );
                        } else {
                            let mut row = 0usize;
                            while row + 4 <= block_rows {
                                uma_so2_f32_matmul_rows4_avx512(
                                    &a_scratch[row * k..row * k + 4 * k],
                                    fused_weight,
                                    k,
                                    n,
                                    &mut c_block[row * n..row * n + 4 * n],
                                    false,
                                );
                                row += 4;
                            }
                            while row < block_rows {
                                uma_so2_f32_matmul_row_avx512(
                                    &a_scratch[row * k..row * k + k],
                                    fused_weight,
                                    k,
                                    n,
                                    &mut c_block[row * n..row * n + n],
                                    false,
                                );
                                row += 1;
                            }
                        }
                    }
                });
            trace_uma_so2_f32_matmul(
                "vjp_order1_gate_stack",
                nodes,
                k,
                n,
                false,
                "row_avx512",
                trace_start,
            );
            return true;
        }

        if uma_so2_f32_avx2_fma_enabled() {
            const AVX2_STACK_ROWS: usize = 4;
            grad_input
                .par_chunks_mut(n * AVX2_STACK_ROWS)
                .enumerate()
                .for_each(|(block, c_block)| {
                    let row_start = block * AVX2_STACK_ROWS;
                    let block_rows = c_block.len() / n;
                    let mut a_scratch = std::mem::MaybeUninit::<
                        [f32; AVX2_STACK_ROWS * ORDER1_SOURCES * UMA_L2_M2_CHANNELS],
                    >::uninit();
                    let a_scratch_ptr = a_scratch.as_mut_ptr().cast::<f32>();
                    unsafe {
                        // SAFETY: `a_scratch_ptr` points at the AVX2 stack
                        // buffer and the fill helper initializes the complete
                        // `block_rows * k` prefix before the row kernels read it.
                        fill_order1_gate_scaled_stack_uninit(
                            row_start,
                            block_rows,
                            channels,
                            activated_block_grad,
                            conv1_out_l_major,
                            gate_sigmoids,
                            m0_addr,
                            a_scratch_ptr,
                        );
                        let a_scratch = std::slice::from_raw_parts(a_scratch_ptr, block_rows * k);
                        if block_rows == AVX2_STACK_ROWS {
                            uma_so2_f32_matmul_rows4_avx2(
                                a_scratch,
                                fused_weight,
                                k,
                                n,
                                c_block,
                                false,
                            );
                        } else {
                            for row in 0..block_rows {
                                uma_so2_f32_matmul_row_avx2(
                                    &a_scratch[row * k..row * k + k],
                                    fused_weight,
                                    k,
                                    n,
                                    &mut c_block[row * n..row * n + n],
                                    false,
                                );
                            }
                        }
                    }
                });
            trace_uma_so2_f32_matmul(
                "vjp_order1_gate_stack",
                nodes,
                k,
                n,
                false,
                "row_avx2",
                trace_start,
            );
            return true;
        }
    }
    #[cfg(not(all(feature = "parallel", target_arch = "x86_64")))]
    {
        let _ = nodes;
        let _ = conv1_in_channels;
        let _ = channels;
        let _ = activated_block_grad;
        let _ = conv1_out_l_major;
        let _ = gate_sigmoids;
        let _ = fused_weight;
        let _ = conv1_m0_full_grad_out;
        let _ = grad_input;
    }
    false
}

#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn uma_f32_try_l2_m2_order2_gate_conv1_vjp_stack(
    nodes: usize,
    conv1_in_channels: usize,
    channels: usize,
    activated_block_grad: &[f32],
    conv1_out_l_major: &[f32],
    gate_sigmoids: &[f32],
    fused_weight: &[f32],
    order2_m0_gate_grad: &mut [f32],
    grad_input: &mut [f32],
) -> bool {
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    {
        const UMA_L2_M2_CHANNELS: usize = 128;
        const UMA_L2_M2_CONV1_IN_CHANNELS: usize = 256;
        const ORDER2_SOURCES: usize = 2;
        const MAX_STACK_ROWS: usize = 8;
        const MAX_STACK_LEN: usize = MAX_STACK_ROWS * ORDER2_SOURCES * UMA_L2_M2_CHANNELS;

        if channels != UMA_L2_M2_CHANNELS || conv1_in_channels != UMA_L2_M2_CONV1_IN_CHANNELS {
            return false;
        }

        let k = ORDER2_SOURCES * channels;
        let n = ORDER2_SOURCES * conv1_in_channels;
        let out_node_stride = 9 * channels;
        let gate_node_stride = 2 * channels;
        debug_assert!(activated_block_grad.len() >= nodes * k);
        debug_assert!(conv1_out_l_major.len() >= nodes * out_node_stride);
        debug_assert!(gate_sigmoids.len() >= nodes * gate_node_stride);
        debug_assert!(fused_weight.len() >= k * n);
        debug_assert!(order2_m0_gate_grad.len() >= nodes * channels);
        debug_assert!(grad_input.len() >= nodes * n);

        if !uma_so2_f32_row_kernel_should_run(nodes, k, n, true) {
            return false;
        }

        let trace_start = uma_blas_trace_start();
        let gate_grad_addr = order2_m0_gate_grad.as_mut_ptr() as usize;
        if uma_so2_f32_avx512_enabled() {
            grad_input
                .par_chunks_mut(n * MAX_STACK_ROWS)
                .enumerate()
                .for_each(|(block, c_block)| {
                    let row_start = block * MAX_STACK_ROWS;
                    let block_rows = c_block.len() / n;
                    let mut a_scratch = std::mem::MaybeUninit::<[f32; MAX_STACK_LEN]>::uninit();
                    let a_scratch_ptr = a_scratch.as_mut_ptr().cast::<f32>();
                    unsafe {
                        // SAFETY: `a_scratch_ptr` points at `MAX_STACK_LEN`
                        // stack slots and the fill helper writes exactly the
                        // `block_rows * k` prefix before the row kernels read it.
                        fill_order2_gate_scaled_stack_uninit(
                            row_start,
                            block_rows,
                            channels,
                            activated_block_grad,
                            conv1_out_l_major,
                            gate_sigmoids,
                            gate_grad_addr,
                            a_scratch_ptr,
                        );
                        let a_scratch = std::slice::from_raw_parts(a_scratch_ptr, block_rows * k);
                        if block_rows == MAX_STACK_ROWS {
                            uma_so2_f32_matmul_rows8x16_avx512(
                                a_scratch,
                                fused_weight,
                                k,
                                n,
                                c_block,
                                false,
                            );
                        } else {
                            let mut row = 0usize;
                            while row + 4 <= block_rows {
                                uma_so2_f32_matmul_rows4_avx512(
                                    &a_scratch[row * k..row * k + 4 * k],
                                    fused_weight,
                                    k,
                                    n,
                                    &mut c_block[row * n..row * n + 4 * n],
                                    false,
                                );
                                row += 4;
                            }
                            while row < block_rows {
                                uma_so2_f32_matmul_row_avx512(
                                    &a_scratch[row * k..row * k + k],
                                    fused_weight,
                                    k,
                                    n,
                                    &mut c_block[row * n..row * n + n],
                                    false,
                                );
                                row += 1;
                            }
                        }
                    }
                });
            trace_uma_so2_f32_matmul(
                "vjp_order2_gate_stack",
                nodes,
                k,
                n,
                false,
                "row_avx512",
                trace_start,
            );
            return true;
        }

        if uma_so2_f32_avx2_fma_enabled() {
            const AVX2_STACK_ROWS: usize = 4;
            grad_input
                .par_chunks_mut(n * AVX2_STACK_ROWS)
                .enumerate()
                .for_each(|(block, c_block)| {
                    let row_start = block * AVX2_STACK_ROWS;
                    let block_rows = c_block.len() / n;
                    let mut a_scratch = std::mem::MaybeUninit::<
                        [f32; AVX2_STACK_ROWS * ORDER2_SOURCES * UMA_L2_M2_CHANNELS],
                    >::uninit();
                    let a_scratch_ptr = a_scratch.as_mut_ptr().cast::<f32>();
                    unsafe {
                        // SAFETY: `a_scratch_ptr` points at the AVX2 stack
                        // buffer and the fill helper initializes the complete
                        // `block_rows * k` prefix before the row kernels read it.
                        fill_order2_gate_scaled_stack_uninit(
                            row_start,
                            block_rows,
                            channels,
                            activated_block_grad,
                            conv1_out_l_major,
                            gate_sigmoids,
                            gate_grad_addr,
                            a_scratch_ptr,
                        );
                        let a_scratch = std::slice::from_raw_parts(a_scratch_ptr, block_rows * k);
                        if block_rows == AVX2_STACK_ROWS {
                            uma_so2_f32_matmul_rows4_avx2(
                                a_scratch,
                                fused_weight,
                                k,
                                n,
                                c_block,
                                false,
                            );
                        } else {
                            for row in 0..block_rows {
                                uma_so2_f32_matmul_row_avx2(
                                    &a_scratch[row * k..row * k + k],
                                    fused_weight,
                                    k,
                                    n,
                                    &mut c_block[row * n..row * n + n],
                                    false,
                                );
                            }
                        }
                    }
                });
            trace_uma_so2_f32_matmul(
                "vjp_order2_gate_stack",
                nodes,
                k,
                n,
                false,
                "row_avx2",
                trace_start,
            );
            return true;
        }
    }
    #[cfg(not(all(feature = "parallel", target_arch = "x86_64")))]
    {
        let _ = nodes;
        let _ = conv1_in_channels;
        let _ = channels;
        let _ = activated_block_grad;
        let _ = conv1_out_l_major;
        let _ = gate_sigmoids;
        let _ = fused_weight;
        let _ = order2_m0_gate_grad;
        let _ = grad_input;
    }
    false
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_order1_gate_scaled_stack_uninit(
    row_start: usize,
    block_rows: usize,
    channels: usize,
    activated_block_grad: &[f32],
    conv1_out_l_major: &[f32],
    gate_sigmoids: &[f32],
    conv1_m0_full_grad_addr: usize,
    a_scratch: *mut f32,
) {
    let block_stride = 4 * channels;
    let out_node_stride = 9 * channels;
    let gate_node_stride = 2 * channels;
    let m0_out_dim = 5 * channels;
    let m0_grad = conv1_m0_full_grad_addr as *mut f32;
    for row in 0..block_rows {
        let node = row_start + row;
        let block_base = node * block_stride;
        let conv1_base = node * out_node_stride + 3 * channels;
        let gate_base = node * gate_node_stride;
        let full_base = node * m0_out_dim;
        let scratch_base = row * block_stride;
        for ch in 0..channels {
            let gate1 = gate_sigmoids[gate_base + ch];
            let gate2 = gate_sigmoids[gate_base + channels + ch];
            let grad3 = activated_block_grad[block_base + ch];
            let grad7 = activated_block_grad[block_base + channels + ch];
            let grad1 = activated_block_grad[block_base + 2 * channels + ch];
            let grad5 = activated_block_grad[block_base + 3 * channels + ch];
            unsafe {
                a_scratch.add(scratch_base + ch).write(grad3 * gate1);
                a_scratch
                    .add(scratch_base + channels + ch)
                    .write(grad7 * gate2);
                a_scratch
                    .add(scratch_base + 2 * channels + ch)
                    .write(grad1 * gate1);
                a_scratch
                    .add(scratch_base + 3 * channels + ch)
                    .write(grad5 * gate2);
                *m0_grad.add(full_base + ch) += grad3 * conv1_out_l_major[conv1_base + ch]
                    + grad1 * conv1_out_l_major[conv1_base + 2 * channels + ch];
                *m0_grad.add(full_base + channels + ch) += grad7
                    * conv1_out_l_major[conv1_base + channels + ch]
                    + grad5 * conv1_out_l_major[conv1_base + 3 * channels + ch];
            }
        }
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_order2_gate_scaled_stack_uninit(
    row_start: usize,
    block_rows: usize,
    channels: usize,
    activated_block_grad: &[f32],
    conv1_out_l_major: &[f32],
    gate_sigmoids: &[f32],
    order2_m0_gate_grad_addr: usize,
    a_scratch: *mut f32,
) {
    let block_stride = 2 * channels;
    let out_node_stride = 9 * channels;
    let gate_node_stride = 2 * channels;
    let gate_grad = order2_m0_gate_grad_addr as *mut f32;
    for row in 0..block_rows {
        let node = row_start + row;
        let block_base = node * block_stride;
        let conv1_base = node * out_node_stride + 7 * channels;
        let gate_base = node * gate_node_stride;
        let gate_grad_base = node * channels;
        let scratch_base = row * block_stride;
        for ch in 0..channels {
            let gate2 = gate_sigmoids[gate_base + channels + ch];
            let grad8 = activated_block_grad[block_base + ch];
            let grad4 = activated_block_grad[block_base + channels + ch];
            unsafe {
                a_scratch.add(scratch_base + ch).write(grad8 * gate2);
                a_scratch
                    .add(scratch_base + channels + ch)
                    .write(grad4 * gate2);
                *gate_grad.add(gate_grad_base + ch) = grad8 * conv1_out_l_major[conv1_base + ch]
                    + grad4 * conv1_out_l_major[conv1_base + channels + ch];
            }
        }
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn uma_so2_f32_try_parallel_row_matmul(
    rows: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    accumulate: bool,
    require_provider_threads_one: bool,
    bias: Option<&[f32]>,
) -> bool {
    debug_assert!(a.len() >= rows * k);
    debug_assert!(b.len() >= k * n);
    debug_assert!(c.len() >= rows * n);
    debug_assert!(bias.is_none_or(|bias| bias.len() >= n));
    if !uma_so2_f32_row_kernel_should_run(rows, k, n, require_provider_threads_one) {
        return false;
    }

    if uma_so2_f32_avx512_enabled() {
        c.par_chunks_mut(n * 8)
            .enumerate()
            .for_each(|(block, c_block)| {
                let row_start = block * 8;
                let block_rows = c_block.len() / n;
                let a_block = &a[row_start * k..row_start * k + block_rows * k];
                unsafe {
                    let mut row = 0usize;
                    if block_rows == 8 {
                        uma_so2_f32_matmul_rows8x16_avx512(a_block, b, k, n, c_block, accumulate);
                    } else {
                        while row + 4 <= block_rows {
                            uma_so2_f32_matmul_rows4_avx512(
                                &a_block[row * k..row * k + 4 * k],
                                b,
                                k,
                                n,
                                &mut c_block[row * n..row * n + 4 * n],
                                accumulate,
                            );
                            row += 4;
                        }
                        while row < block_rows {
                            uma_so2_f32_matmul_row_avx512(
                                &a_block[row * k..row * k + k],
                                b,
                                k,
                                n,
                                &mut c_block[row * n..row * n + n],
                                accumulate,
                            );
                            row += 1;
                        }
                    }
                }
                if let Some(bias) = bias {
                    uma_f32_add_bias_to_row_block(block_rows, n, bias, c_block);
                }
            });
        return true;
    }

    if uma_so2_f32_avx2_fma_enabled() {
        c.par_chunks_mut(n * 4)
            .enumerate()
            .for_each(|(block, c_block)| {
                let row_start = block * 4;
                let block_rows = c_block.len() / n;
                let a_block = &a[row_start * k..row_start * k + block_rows * k];
                unsafe {
                    if block_rows == 4 {
                        uma_so2_f32_matmul_rows4_avx2(a_block, b, k, n, c_block, accumulate);
                    } else {
                        for row in 0..block_rows {
                            uma_so2_f32_matmul_row_avx2(
                                &a_block[row * k..row * k + k],
                                b,
                                k,
                                n,
                                &mut c_block[row * n..row * n + n],
                                accumulate,
                            );
                        }
                    }
                }
                if let Some(bias) = bias {
                    uma_f32_add_bias_to_row_block(block_rows, n, bias, c_block);
                }
            });
        return true;
    }

    false
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn uma_so2_f32_try_parallel_lmajor_direct_row_matmul(
    rows: usize,
    channels: usize,
    source_positions: &[usize],
    lmajor_row_stride: usize,
    a_lmajor: &[f32],
    b: &[f32],
    n: usize,
    c: &mut [f32],
    accumulate: bool,
) -> bool {
    let Some(k) = source_positions.len().checked_mul(channels) else {
        return false;
    };
    debug_assert!(a_lmajor.len() >= rows * lmajor_row_stride);
    debug_assert!(b.len() >= k * n);
    debug_assert!(c.len() >= rows * n);
    if k == 0
        || channels == 0
        || n == 0
        || a_lmajor.len() < rows * lmajor_row_stride
        || b.len() < k * n
        || c.len() < rows * n
        || !uma_so2_f32_row_kernel_should_run(rows, k, n, true)
    {
        return false;
    }
    for &source in source_positions {
        let Some(source_base) = source.checked_mul(channels) else {
            return false;
        };
        let Some(source_end) = source_base.checked_add(channels) else {
            return false;
        };
        if source_end > lmajor_row_stride {
            return false;
        }
    }

    if uma_so2_f32_avx512_enabled() {
        c.par_chunks_mut(n * 8)
            .enumerate()
            .for_each(|(block, c_block)| {
                let row_start = block * 8;
                let block_rows = c_block.len() / n;
                let a_block = &a_lmajor[row_start * lmajor_row_stride
                    ..row_start * lmajor_row_stride + block_rows * lmajor_row_stride];
                unsafe {
                    let mut row = 0usize;
                    if block_rows == 8 {
                        uma_so2_f32_lmajor_direct_matmul_rows8x16_avx512(
                            a_block,
                            channels,
                            source_positions,
                            lmajor_row_stride,
                            b,
                            n,
                            c_block,
                            accumulate,
                        );
                    } else {
                        while row + 4 <= block_rows {
                            uma_so2_f32_lmajor_direct_matmul_rows4_avx512(
                                &a_block[row * lmajor_row_stride
                                    ..row * lmajor_row_stride + 4 * lmajor_row_stride],
                                channels,
                                source_positions,
                                lmajor_row_stride,
                                b,
                                n,
                                &mut c_block[row * n..row * n + 4 * n],
                                accumulate,
                            );
                            row += 4;
                        }
                        while row < block_rows {
                            uma_so2_f32_lmajor_direct_matmul_row_avx512(
                                &a_block[row * lmajor_row_stride
                                    ..row * lmajor_row_stride + lmajor_row_stride],
                                channels,
                                source_positions,
                                b,
                                n,
                                &mut c_block[row * n..row * n + n],
                                accumulate,
                            );
                            row += 1;
                        }
                    }
                }
            });
        return true;
    }

    if uma_so2_f32_avx2_fma_enabled() {
        c.par_chunks_mut(n * 4)
            .enumerate()
            .for_each(|(block, c_block)| {
                let row_start = block * 4;
                let block_rows = c_block.len() / n;
                let a_block = &a_lmajor[row_start * lmajor_row_stride
                    ..row_start * lmajor_row_stride + block_rows * lmajor_row_stride];
                unsafe {
                    if block_rows == 4 {
                        uma_so2_f32_lmajor_direct_matmul_rows4_avx2(
                            a_block,
                            channels,
                            source_positions,
                            lmajor_row_stride,
                            b,
                            n,
                            c_block,
                            accumulate,
                        );
                    } else {
                        for row in 0..block_rows {
                            uma_so2_f32_lmajor_direct_matmul_row_avx2(
                                &a_block[row * lmajor_row_stride
                                    ..row * lmajor_row_stride + lmajor_row_stride],
                                channels,
                                source_positions,
                                b,
                                n,
                                &mut c_block[row * n..row * n + n],
                                accumulate,
                            );
                        }
                    }
                }
            });
        return true;
    }

    false
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[inline]
fn uma_f32_add_bias_to_row_block(block_rows: usize, n: usize, bias: &[f32], c_block: &mut [f32]) {
    debug_assert!(bias.len() >= n);
    for row in c_block.chunks_exact_mut(n).take(block_rows) {
        for (value, bias_value) in row.iter_mut().zip(bias.iter()) {
            *value += *bias_value;
        }
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
pub(crate) fn uma_f32_try_parallel_row_matmul(
    rows: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    accumulate: bool,
) -> bool {
    let trace_start = uma_blas_trace_start();
    let ran = uma_so2_f32_try_parallel_row_matmul(rows, k, n, a, b, c, accumulate, true, None);
    if ran {
        let path = if uma_so2_f32_avx512_enabled() {
            "row_avx512"
        } else {
            "row_avx2"
        };
        trace_uma_f32_row_matmul(rows, k, n, accumulate, path, trace_start);
    }
    ran
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
pub(crate) fn uma_f32_try_parallel_row_matmul_bias(
    rows: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
) -> bool {
    debug_assert!(bias.len() >= n);
    let trace_start = uma_blas_trace_start();
    let ran = uma_so2_f32_try_parallel_row_matmul(rows, k, n, a, b, c, false, true, Some(bias));
    if ran {
        let path = if uma_so2_f32_avx512_enabled() {
            "row_avx512_bias"
        } else {
            "row_avx2_bias"
        };
        trace_uma_f32_row_matmul(rows, k, n, false, path, trace_start);
    }
    ran
}

// ── FP64 AVX-512 row-matmul kernel ────────────────────────────────────────────
// Mirrors the f32 row kernels (`uma_so2_f32_matmul_rows8x16_avx512` etc.) but in
// f64 (8-wide zmm). The FP64 UMA path is dgemm-bound: a `perf` profile of the
// FP64 frame attributes ~47% to OpenBLAS dgemm plus ~15% kernel time to its
// thread synchronization on the small per-edge SO2 GEMMs. A rayon-parallel row
// kernel (one BLAS thread, rayon over rows, the weight kept hot in cache and
// reused across 8 rows per inner load) removes that synchronization overhead.
// f64 has ample mantissa; FMA reassociation vs dgemm is ~1e-16 relative, well
// inside the FP64 parity tolerance.

/// Default-on AVX-512 gate for the FP64 row kernels (honors the global disables).
#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[inline]
fn uma_so2_f64_avx512_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !uma_so2_f32_simd_disabled()
            && !uma_so2_f32_avx_disabled()
            && std::env::var_os("LLMLIP_DISABLE_AVX512").is_none()
            && std::arch::is_x86_feature_detected!("avx512f")
    })
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[inline]
fn uma_so2_f64_row_kernel_should_run(
    rows: usize,
    k: usize,
    n: usize,
    require_provider_threads_one: bool,
) -> bool {
    rayon::current_num_threads() > 1
        && (!require_provider_threads_one || uma_so2_f32_provider_threads_is_one())
        && rows >= 16
        && k >= 128
        && n >= 128
        && k.is_multiple_of(8)
        && n.is_multiple_of(8)
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
fn uma_so2_f64_matmul_row_scalar_tail(
    a_row: &[f64],
    b: &[f64],
    k: usize,
    n: usize,
    c_row: &mut [f64],
    col_start: usize,
    accumulate: bool,
) {
    for col in col_start..n {
        let mut acc = if accumulate { c_row[col] } else { 0.0 };
        for kk in 0..k {
            acc += a_row[kk] * b[kk * n + col];
        }
        c_row[col] = acc;
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn uma_so2_f64_matmul_row_avx512(
    a_row: &[f64],
    b: &[f64],
    k: usize,
    n: usize,
    c_row: &mut [f64],
    accumulate: bool,
) {
    use std::arch::x86_64::{
        __m512d, _mm512_fmadd_pd, _mm512_loadu_pd, _mm512_set1_pd, _mm512_setzero_pd,
        _mm512_storeu_pd,
    };
    let mut col = 0usize;
    while col + 32 <= n {
        let mut acc0: __m512d = if accumulate {
            unsafe { _mm512_loadu_pd(c_row.as_ptr().add(col)) }
        } else {
            _mm512_setzero_pd()
        };
        let mut acc1: __m512d = if accumulate {
            unsafe { _mm512_loadu_pd(c_row.as_ptr().add(col + 8)) }
        } else {
            _mm512_setzero_pd()
        };
        let mut acc2: __m512d = if accumulate {
            unsafe { _mm512_loadu_pd(c_row.as_ptr().add(col + 16)) }
        } else {
            _mm512_setzero_pd()
        };
        let mut acc3: __m512d = if accumulate {
            unsafe { _mm512_loadu_pd(c_row.as_ptr().add(col + 24)) }
        } else {
            _mm512_setzero_pd()
        };
        for kk in 0..k {
            let a = unsafe { _mm512_set1_pd(*a_row.get_unchecked(kk)) };
            let bp = unsafe { b.as_ptr().add(kk * n + col) };
            acc0 = _mm512_fmadd_pd(a, unsafe { _mm512_loadu_pd(bp) }, acc0);
            acc1 = _mm512_fmadd_pd(a, unsafe { _mm512_loadu_pd(bp.add(8)) }, acc1);
            acc2 = _mm512_fmadd_pd(a, unsafe { _mm512_loadu_pd(bp.add(16)) }, acc2);
            acc3 = _mm512_fmadd_pd(a, unsafe { _mm512_loadu_pd(bp.add(24)) }, acc3);
        }
        unsafe {
            _mm512_storeu_pd(c_row.as_mut_ptr().add(col), acc0);
            _mm512_storeu_pd(c_row.as_mut_ptr().add(col + 8), acc1);
            _mm512_storeu_pd(c_row.as_mut_ptr().add(col + 16), acc2);
            _mm512_storeu_pd(c_row.as_mut_ptr().add(col + 24), acc3);
        }
        col += 32;
    }
    while col + 8 <= n {
        let mut acc = if accumulate {
            unsafe { _mm512_loadu_pd(c_row.as_ptr().add(col)) }
        } else {
            _mm512_setzero_pd()
        };
        for kk in 0..k {
            let a = unsafe { _mm512_set1_pd(*a_row.get_unchecked(kk)) };
            acc = _mm512_fmadd_pd(
                a,
                unsafe { _mm512_loadu_pd(b.as_ptr().add(kk * n + col)) },
                acc,
            );
        }
        unsafe { _mm512_storeu_pd(c_row.as_mut_ptr().add(col), acc) };
        col += 8;
    }
    if col < n {
        uma_so2_f64_matmul_row_scalar_tail(a_row, b, k, n, c_row, col, accumulate);
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn uma_so2_f64_matmul_rows8_avx512(
    a_block: &[f64],
    b: &[f64],
    k: usize,
    n: usize,
    c_block: &mut [f64],
    accumulate: bool,
) {
    use std::arch::x86_64::{
        __m512d, _mm512_fmadd_pd, _mm512_loadu_pd, _mm512_set1_pd, _mm512_setzero_pd,
        _mm512_storeu_pd,
    };
    let mut col = 0usize;
    while col + 8 <= n {
        let mut r0: __m512d = if accumulate {
            unsafe { _mm512_loadu_pd(c_block.as_ptr().add(col)) }
        } else {
            _mm512_setzero_pd()
        };
        let mut r1: __m512d = if accumulate {
            unsafe { _mm512_loadu_pd(c_block.as_ptr().add(n + col)) }
        } else {
            _mm512_setzero_pd()
        };
        let mut r2: __m512d = if accumulate {
            unsafe { _mm512_loadu_pd(c_block.as_ptr().add(2 * n + col)) }
        } else {
            _mm512_setzero_pd()
        };
        let mut r3: __m512d = if accumulate {
            unsafe { _mm512_loadu_pd(c_block.as_ptr().add(3 * n + col)) }
        } else {
            _mm512_setzero_pd()
        };
        let mut r4: __m512d = if accumulate {
            unsafe { _mm512_loadu_pd(c_block.as_ptr().add(4 * n + col)) }
        } else {
            _mm512_setzero_pd()
        };
        let mut r5: __m512d = if accumulate {
            unsafe { _mm512_loadu_pd(c_block.as_ptr().add(5 * n + col)) }
        } else {
            _mm512_setzero_pd()
        };
        let mut r6: __m512d = if accumulate {
            unsafe { _mm512_loadu_pd(c_block.as_ptr().add(6 * n + col)) }
        } else {
            _mm512_setzero_pd()
        };
        let mut r7: __m512d = if accumulate {
            unsafe { _mm512_loadu_pd(c_block.as_ptr().add(7 * n + col)) }
        } else {
            _mm512_setzero_pd()
        };
        for kk in 0..k {
            let w = unsafe { _mm512_loadu_pd(b.as_ptr().add(kk * n + col)) };
            r0 = _mm512_fmadd_pd(unsafe { _mm512_set1_pd(*a_block.get_unchecked(kk)) }, w, r0);
            r1 = _mm512_fmadd_pd(
                unsafe { _mm512_set1_pd(*a_block.get_unchecked(k + kk)) },
                w,
                r1,
            );
            r2 = _mm512_fmadd_pd(
                unsafe { _mm512_set1_pd(*a_block.get_unchecked(2 * k + kk)) },
                w,
                r2,
            );
            r3 = _mm512_fmadd_pd(
                unsafe { _mm512_set1_pd(*a_block.get_unchecked(3 * k + kk)) },
                w,
                r3,
            );
            r4 = _mm512_fmadd_pd(
                unsafe { _mm512_set1_pd(*a_block.get_unchecked(4 * k + kk)) },
                w,
                r4,
            );
            r5 = _mm512_fmadd_pd(
                unsafe { _mm512_set1_pd(*a_block.get_unchecked(5 * k + kk)) },
                w,
                r5,
            );
            r6 = _mm512_fmadd_pd(
                unsafe { _mm512_set1_pd(*a_block.get_unchecked(6 * k + kk)) },
                w,
                r6,
            );
            r7 = _mm512_fmadd_pd(
                unsafe { _mm512_set1_pd(*a_block.get_unchecked(7 * k + kk)) },
                w,
                r7,
            );
        }
        unsafe {
            _mm512_storeu_pd(c_block.as_mut_ptr().add(col), r0);
            _mm512_storeu_pd(c_block.as_mut_ptr().add(n + col), r1);
            _mm512_storeu_pd(c_block.as_mut_ptr().add(2 * n + col), r2);
            _mm512_storeu_pd(c_block.as_mut_ptr().add(3 * n + col), r3);
            _mm512_storeu_pd(c_block.as_mut_ptr().add(4 * n + col), r4);
            _mm512_storeu_pd(c_block.as_mut_ptr().add(5 * n + col), r5);
            _mm512_storeu_pd(c_block.as_mut_ptr().add(6 * n + col), r6);
            _mm512_storeu_pd(c_block.as_mut_ptr().add(7 * n + col), r7);
        }
        col += 8;
    }
    if col < n {
        for row in 0..8 {
            uma_so2_f64_matmul_row_scalar_tail(
                &a_block[row * k..row * k + k],
                b,
                k,
                n,
                &mut c_block[row * n..row * n + n],
                col,
                accumulate,
            );
        }
    }
}

/// Row-parallel f64 matmul `C[rows,n] (+)= A[rows,k] @ B[k,n]`. Returns true when
/// the AVX-512 row kernel handled it; false means the caller must run dgemm.
#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn uma_so2_f64_try_parallel_row_matmul(
    rows: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    accumulate: bool,
    require_provider_threads_one: bool,
) -> bool {
    if !uma_so2_f64_avx512_enabled()
        || !uma_so2_f64_row_kernel_should_run(rows, k, n, require_provider_threads_one)
        || a.len() < rows * k
        || b.len() < k * n
        || c.len() < rows * n
    {
        return false;
    }
    c.par_chunks_mut(n * 8)
        .enumerate()
        .for_each(|(block, c_block)| {
            let row_start = block * 8;
            let block_rows = c_block.len() / n;
            let a_block = &a[row_start * k..row_start * k + block_rows * k];
            unsafe {
                if block_rows == 8 {
                    uma_so2_f64_matmul_rows8_avx512(a_block, b, k, n, c_block, accumulate);
                } else {
                    for row in 0..block_rows {
                        uma_so2_f64_matmul_row_avx512(
                            &a_block[row * k..row * k + k],
                            b,
                            k,
                            n,
                            &mut c_block[row * n..row * n + n],
                            accumulate,
                        );
                    }
                }
            }
        });
    true
}

#[cfg(all(
    feature = "parallel",
    target_arch = "x86_64",
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn so2_mconv_vjp_input_add_precomputed_fused_f32_lmajor_direct(
    rows: usize,
    channels: usize,
    source_positions: &[usize],
    lmajor_row_stride: usize,
    fused_in_dim: usize,
    fused_weight: &[f32], // precomputed [source_positions.len()*channels, fused_in_dim]
    grad_out_lmajor: &[f32],
    grad_input: &mut [f32],
) -> bool {
    let Some(fused_out_dim) = source_positions.len().checked_mul(channels) else {
        return false;
    };
    debug_assert!(fused_weight.len() >= fused_out_dim * fused_in_dim);
    debug_assert!(grad_out_lmajor.len() >= rows * lmajor_row_stride);
    debug_assert!(grad_input.len() >= rows * fused_in_dim);
    let trace_start = uma_blas_trace_start();
    let ran = uma_so2_f32_try_parallel_lmajor_direct_row_matmul(
        rows,
        channels,
        source_positions,
        lmajor_row_stride,
        grad_out_lmajor,
        fused_weight,
        fused_in_dim,
        grad_input,
        true,
    );
    if ran {
        let path = if uma_so2_f32_avx512_enabled() {
            "row_avx512_lmajor_direct"
        } else {
            "row_avx2_lmajor_direct"
        };
        trace_uma_so2_f32_matmul(
            "vjp_lmajor_direct",
            rows,
            fused_out_dim,
            fused_in_dim,
            true,
            path,
            trace_start,
        );
    }
    ran
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[allow(dead_code)]
fn uma_so2_f32_matmul_row_scalar(
    a_row: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    c_row: &mut [f32],
    accumulate: bool,
) {
    for col in 0..n {
        let mut acc = if accumulate { c_row[col] } else { 0.0_f32 };
        for kk in 0..k {
            acc += a_row[kk] * b[kk * n + col];
        }
        c_row[col] = acc;
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn uma_so2_f32_matmul_row_avx2(
    a_row: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    c_row: &mut [f32],
    accumulate: bool,
) {
    use std::arch::x86_64::{
        __m256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
    };

    let mut col = 0usize;
    while col + 32 <= n {
        let mut acc0: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_row.as_ptr().add(col)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut acc1: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_row.as_ptr().add(col + 8)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut acc2: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_row.as_ptr().add(col + 16)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut acc3: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_row.as_ptr().add(col + 24)) }
        } else {
            _mm256_setzero_ps()
        };
        for kk in 0..k {
            let a = unsafe { _mm256_set1_ps(*a_row.get_unchecked(kk)) };
            let b_base = unsafe { b.as_ptr().add(kk * n + col) };
            let w0 = unsafe { _mm256_loadu_ps(b_base) };
            let w1 = unsafe { _mm256_loadu_ps(b_base.add(8)) };
            let w2 = unsafe { _mm256_loadu_ps(b_base.add(16)) };
            let w3 = unsafe { _mm256_loadu_ps(b_base.add(24)) };
            acc0 = _mm256_fmadd_ps(a, w0, acc0);
            acc1 = _mm256_fmadd_ps(a, w1, acc1);
            acc2 = _mm256_fmadd_ps(a, w2, acc2);
            acc3 = _mm256_fmadd_ps(a, w3, acc3);
        }
        unsafe {
            _mm256_storeu_ps(c_row.as_mut_ptr().add(col), acc0);
            _mm256_storeu_ps(c_row.as_mut_ptr().add(col + 8), acc1);
            _mm256_storeu_ps(c_row.as_mut_ptr().add(col + 16), acc2);
            _mm256_storeu_ps(c_row.as_mut_ptr().add(col + 24), acc3);
        }
        col += 32;
    }
    while col + 8 <= n {
        let mut acc: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_row.as_ptr().add(col)) }
        } else {
            _mm256_setzero_ps()
        };
        for kk in 0..k {
            let a = unsafe { _mm256_set1_ps(*a_row.get_unchecked(kk)) };
            let w = unsafe { _mm256_loadu_ps(b.as_ptr().add(kk * n + col)) };
            acc = _mm256_fmadd_ps(a, w, acc);
        }
        unsafe {
            _mm256_storeu_ps(c_row.as_mut_ptr().add(col), acc);
        }
        col += 8;
    }
    if col < n {
        uma_so2_f32_matmul_row_scalar_tail(a_row, b, k, n, c_row, accumulate, col);
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn uma_so2_f32_matmul_rows4_avx2(
    a_block: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    c_block: &mut [f32],
    accumulate: bool,
) {
    use std::arch::x86_64::{
        __m256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
    };

    let mut col = 0usize;
    while col + 16 <= n {
        let mut r0c0: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(col)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r0c1: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(col + 8)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r1c0: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(n + col)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r1c1: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(n + col + 8)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r2c0: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(2 * n + col)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r2c1: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(2 * n + col + 8)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r3c0: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(3 * n + col)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r3c1: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(3 * n + col + 8)) }
        } else {
            _mm256_setzero_ps()
        };
        for kk in 0..k {
            let b_base = unsafe { b.as_ptr().add(kk * n + col) };
            let w0 = unsafe { _mm256_loadu_ps(b_base) };
            let w1 = unsafe { _mm256_loadu_ps(b_base.add(8)) };
            let a0 = unsafe { _mm256_set1_ps(*a_block.get_unchecked(kk)) };
            let a1 = unsafe { _mm256_set1_ps(*a_block.get_unchecked(k + kk)) };
            let a2 = unsafe { _mm256_set1_ps(*a_block.get_unchecked(2 * k + kk)) };
            let a3 = unsafe { _mm256_set1_ps(*a_block.get_unchecked(3 * k + kk)) };
            r0c0 = _mm256_fmadd_ps(a0, w0, r0c0);
            r0c1 = _mm256_fmadd_ps(a0, w1, r0c1);
            r1c0 = _mm256_fmadd_ps(a1, w0, r1c0);
            r1c1 = _mm256_fmadd_ps(a1, w1, r1c1);
            r2c0 = _mm256_fmadd_ps(a2, w0, r2c0);
            r2c1 = _mm256_fmadd_ps(a2, w1, r2c1);
            r3c0 = _mm256_fmadd_ps(a3, w0, r3c0);
            r3c1 = _mm256_fmadd_ps(a3, w1, r3c1);
        }
        unsafe {
            _mm256_storeu_ps(c_block.as_mut_ptr().add(col), r0c0);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(col + 8), r0c1);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(n + col), r1c0);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(n + col + 8), r1c1);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(2 * n + col), r2c0);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(2 * n + col + 8), r2c1);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(3 * n + col), r3c0);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(3 * n + col + 8), r3c1);
        }
        col += 16;
    }
    while col + 8 <= n {
        let mut r0: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(col)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r1: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(n + col)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r2: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(2 * n + col)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r3: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(3 * n + col)) }
        } else {
            _mm256_setzero_ps()
        };
        for kk in 0..k {
            let w = unsafe { _mm256_loadu_ps(b.as_ptr().add(kk * n + col)) };
            let a0 = unsafe { _mm256_set1_ps(*a_block.get_unchecked(kk)) };
            let a1 = unsafe { _mm256_set1_ps(*a_block.get_unchecked(k + kk)) };
            let a2 = unsafe { _mm256_set1_ps(*a_block.get_unchecked(2 * k + kk)) };
            let a3 = unsafe { _mm256_set1_ps(*a_block.get_unchecked(3 * k + kk)) };
            r0 = _mm256_fmadd_ps(a0, w, r0);
            r1 = _mm256_fmadd_ps(a1, w, r1);
            r2 = _mm256_fmadd_ps(a2, w, r2);
            r3 = _mm256_fmadd_ps(a3, w, r3);
        }
        unsafe {
            _mm256_storeu_ps(c_block.as_mut_ptr().add(col), r0);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(n + col), r1);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(2 * n + col), r2);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(3 * n + col), r3);
        }
        col += 8;
    }
    if col < n {
        for row in 0..4 {
            uma_so2_f32_matmul_row_scalar_tail(
                &a_block[row * k..row * k + k],
                b,
                k,
                n,
                &mut c_block[row * n..row * n + n],
                accumulate,
                col,
            );
        }
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn uma_so2_f32_matmul_row_avx512(
    a_row: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    c_row: &mut [f32],
    accumulate: bool,
) {
    use std::arch::x86_64::{
        __m512, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_setzero_ps,
        _mm512_storeu_ps,
    };

    let mut col = 0usize;
    while col + 64 <= n {
        let mut acc0: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_row.as_ptr().add(col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut acc1: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_row.as_ptr().add(col + 16)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut acc2: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_row.as_ptr().add(col + 32)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut acc3: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_row.as_ptr().add(col + 48)) }
        } else {
            _mm512_setzero_ps()
        };
        for kk in 0..k {
            let a = unsafe { _mm512_set1_ps(*a_row.get_unchecked(kk)) };
            let b_base = unsafe { b.as_ptr().add(kk * n + col) };
            let w0 = unsafe { _mm512_loadu_ps(b_base) };
            let w1 = unsafe { _mm512_loadu_ps(b_base.add(16)) };
            let w2 = unsafe { _mm512_loadu_ps(b_base.add(32)) };
            let w3 = unsafe { _mm512_loadu_ps(b_base.add(48)) };
            acc0 = _mm512_fmadd_ps(a, w0, acc0);
            acc1 = _mm512_fmadd_ps(a, w1, acc1);
            acc2 = _mm512_fmadd_ps(a, w2, acc2);
            acc3 = _mm512_fmadd_ps(a, w3, acc3);
        }
        unsafe {
            _mm512_storeu_ps(c_row.as_mut_ptr().add(col), acc0);
            _mm512_storeu_ps(c_row.as_mut_ptr().add(col + 16), acc1);
            _mm512_storeu_ps(c_row.as_mut_ptr().add(col + 32), acc2);
            _mm512_storeu_ps(c_row.as_mut_ptr().add(col + 48), acc3);
        }
        col += 64;
    }
    while col + 16 <= n {
        let mut acc: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_row.as_ptr().add(col)) }
        } else {
            _mm512_setzero_ps()
        };
        for kk in 0..k {
            let a = unsafe { _mm512_set1_ps(*a_row.get_unchecked(kk)) };
            let w = unsafe { _mm512_loadu_ps(b.as_ptr().add(kk * n + col)) };
            acc = _mm512_fmadd_ps(a, w, acc);
        }
        unsafe {
            _mm512_storeu_ps(c_row.as_mut_ptr().add(col), acc);
        }
        col += 16;
    }
    if col < n {
        uma_so2_f32_matmul_row_scalar_tail(a_row, b, k, n, c_row, accumulate, col);
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn uma_so2_f32_matmul_rows4_avx512(
    a_block: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    c_block: &mut [f32],
    accumulate: bool,
) {
    use std::arch::x86_64::{
        __m512, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_setzero_ps,
        _mm512_storeu_ps,
    };

    let mut col = 0usize;
    while col + 32 <= n {
        let mut r0c0: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r0c1: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(col + 16)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r1c0: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r1c1: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(n + col + 16)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r2c0: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(2 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r2c1: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(2 * n + col + 16)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r3c0: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(3 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r3c1: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(3 * n + col + 16)) }
        } else {
            _mm512_setzero_ps()
        };
        for kk in 0..k {
            let b_base = unsafe { b.as_ptr().add(kk * n + col) };
            let w0 = unsafe { _mm512_loadu_ps(b_base) };
            let w1 = unsafe { _mm512_loadu_ps(b_base.add(16)) };
            let a0 = unsafe { _mm512_set1_ps(*a_block.get_unchecked(kk)) };
            let a1 = unsafe { _mm512_set1_ps(*a_block.get_unchecked(k + kk)) };
            let a2 = unsafe { _mm512_set1_ps(*a_block.get_unchecked(2 * k + kk)) };
            let a3 = unsafe { _mm512_set1_ps(*a_block.get_unchecked(3 * k + kk)) };
            r0c0 = _mm512_fmadd_ps(a0, w0, r0c0);
            r0c1 = _mm512_fmadd_ps(a0, w1, r0c1);
            r1c0 = _mm512_fmadd_ps(a1, w0, r1c0);
            r1c1 = _mm512_fmadd_ps(a1, w1, r1c1);
            r2c0 = _mm512_fmadd_ps(a2, w0, r2c0);
            r2c1 = _mm512_fmadd_ps(a2, w1, r2c1);
            r3c0 = _mm512_fmadd_ps(a3, w0, r3c0);
            r3c1 = _mm512_fmadd_ps(a3, w1, r3c1);
        }
        unsafe {
            _mm512_storeu_ps(c_block.as_mut_ptr().add(col), r0c0);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(col + 16), r0c1);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(n + col), r1c0);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(n + col + 16), r1c1);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(2 * n + col), r2c0);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(2 * n + col + 16), r2c1);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(3 * n + col), r3c0);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(3 * n + col + 16), r3c1);
        }
        col += 32;
    }
    while col + 16 <= n {
        let mut r0: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r1: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r2: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(2 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r3: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(3 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        for kk in 0..k {
            let w = unsafe { _mm512_loadu_ps(b.as_ptr().add(kk * n + col)) };
            let a0 = unsafe { _mm512_set1_ps(*a_block.get_unchecked(kk)) };
            let a1 = unsafe { _mm512_set1_ps(*a_block.get_unchecked(k + kk)) };
            let a2 = unsafe { _mm512_set1_ps(*a_block.get_unchecked(2 * k + kk)) };
            let a3 = unsafe { _mm512_set1_ps(*a_block.get_unchecked(3 * k + kk)) };
            r0 = _mm512_fmadd_ps(a0, w, r0);
            r1 = _mm512_fmadd_ps(a1, w, r1);
            r2 = _mm512_fmadd_ps(a2, w, r2);
            r3 = _mm512_fmadd_ps(a3, w, r3);
        }
        unsafe {
            _mm512_storeu_ps(c_block.as_mut_ptr().add(col), r0);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(n + col), r1);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(2 * n + col), r2);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(3 * n + col), r3);
        }
        col += 16;
    }
    if col < n {
        for row in 0..4 {
            uma_so2_f32_matmul_row_scalar_tail(
                &a_block[row * k..row * k + k],
                b,
                k,
                n,
                &mut c_block[row * n..row * n + n],
                accumulate,
                col,
            );
        }
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn uma_so2_f32_matmul_rows8x16_avx512(
    a_block: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    c_block: &mut [f32],
    accumulate: bool,
) {
    use std::arch::x86_64::{
        __m512, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_setzero_ps,
        _mm512_storeu_ps,
    };

    let mut col = 0usize;
    while col + 16 <= n {
        let mut r0: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r1: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r2: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(2 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r3: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(3 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r4: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(4 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r5: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(5 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r6: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(6 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r7: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(7 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        for kk in 0..k {
            let w = unsafe { _mm512_loadu_ps(b.as_ptr().add(kk * n + col)) };
            let a0 = unsafe { _mm512_set1_ps(*a_block.get_unchecked(kk)) };
            let a1 = unsafe { _mm512_set1_ps(*a_block.get_unchecked(k + kk)) };
            let a2 = unsafe { _mm512_set1_ps(*a_block.get_unchecked(2 * k + kk)) };
            let a3 = unsafe { _mm512_set1_ps(*a_block.get_unchecked(3 * k + kk)) };
            let a4 = unsafe { _mm512_set1_ps(*a_block.get_unchecked(4 * k + kk)) };
            let a5 = unsafe { _mm512_set1_ps(*a_block.get_unchecked(5 * k + kk)) };
            let a6 = unsafe { _mm512_set1_ps(*a_block.get_unchecked(6 * k + kk)) };
            let a7 = unsafe { _mm512_set1_ps(*a_block.get_unchecked(7 * k + kk)) };
            r0 = _mm512_fmadd_ps(a0, w, r0);
            r1 = _mm512_fmadd_ps(a1, w, r1);
            r2 = _mm512_fmadd_ps(a2, w, r2);
            r3 = _mm512_fmadd_ps(a3, w, r3);
            r4 = _mm512_fmadd_ps(a4, w, r4);
            r5 = _mm512_fmadd_ps(a5, w, r5);
            r6 = _mm512_fmadd_ps(a6, w, r6);
            r7 = _mm512_fmadd_ps(a7, w, r7);
        }
        unsafe {
            _mm512_storeu_ps(c_block.as_mut_ptr().add(col), r0);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(n + col), r1);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(2 * n + col), r2);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(3 * n + col), r3);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(4 * n + col), r4);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(5 * n + col), r5);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(6 * n + col), r6);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(7 * n + col), r7);
        }
        col += 16;
    }
    if col < n {
        for row in 0..8 {
            uma_so2_f32_matmul_row_scalar_tail(
                &a_block[row * k..row * k + k],
                b,
                k,
                n,
                &mut c_block[row * n..row * n + n],
                accumulate,
                col,
            );
        }
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
fn uma_so2_f32_matmul_row_scalar_tail(
    a_row: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    c_row: &mut [f32],
    accumulate: bool,
    start_col: usize,
) {
    for col in start_col..n {
        let mut acc = if accumulate { c_row[col] } else { 0.0_f32 };
        for kk in 0..k {
            acc += a_row[kk] * b[kk * n + col];
        }
        c_row[col] = acc;
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn uma_so2_f32_lmajor_direct_matmul_row_scalar_tail(
    a_row_lmajor: &[f32],
    channels: usize,
    source_positions: &[usize],
    b: &[f32],
    n: usize,
    c_row: &mut [f32],
    accumulate: bool,
    start_col: usize,
) {
    for col in start_col..n {
        let mut acc = if accumulate { c_row[col] } else { 0.0_f32 };
        let mut kk = 0usize;
        for &source in source_positions {
            let source_base = source * channels;
            for ch in 0..channels {
                acc += a_row_lmajor[source_base + ch] * b[kk * n + col];
                kk += 1;
            }
        }
        c_row[col] = acc;
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn uma_so2_f32_lmajor_direct_matmul_row_avx2(
    a_row_lmajor: &[f32],
    channels: usize,
    source_positions: &[usize],
    b: &[f32],
    n: usize,
    c_row: &mut [f32],
    accumulate: bool,
) {
    use std::arch::x86_64::{
        __m256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
    };

    let mut col = 0usize;
    while col + 32 <= n {
        let mut acc0: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_row.as_ptr().add(col)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut acc1: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_row.as_ptr().add(col + 8)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut acc2: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_row.as_ptr().add(col + 16)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut acc3: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_row.as_ptr().add(col + 24)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut kk = 0usize;
        for &source in source_positions {
            let source_base = source * channels;
            for ch in 0..channels {
                let a = unsafe { _mm256_set1_ps(*a_row_lmajor.get_unchecked(source_base + ch)) };
                let b_base = unsafe { b.as_ptr().add(kk * n + col) };
                let w0 = unsafe { _mm256_loadu_ps(b_base) };
                let w1 = unsafe { _mm256_loadu_ps(b_base.add(8)) };
                let w2 = unsafe { _mm256_loadu_ps(b_base.add(16)) };
                let w3 = unsafe { _mm256_loadu_ps(b_base.add(24)) };
                acc0 = _mm256_fmadd_ps(a, w0, acc0);
                acc1 = _mm256_fmadd_ps(a, w1, acc1);
                acc2 = _mm256_fmadd_ps(a, w2, acc2);
                acc3 = _mm256_fmadd_ps(a, w3, acc3);
                kk += 1;
            }
        }
        unsafe {
            _mm256_storeu_ps(c_row.as_mut_ptr().add(col), acc0);
            _mm256_storeu_ps(c_row.as_mut_ptr().add(col + 8), acc1);
            _mm256_storeu_ps(c_row.as_mut_ptr().add(col + 16), acc2);
            _mm256_storeu_ps(c_row.as_mut_ptr().add(col + 24), acc3);
        }
        col += 32;
    }
    while col + 8 <= n {
        let mut acc: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_row.as_ptr().add(col)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut kk = 0usize;
        for &source in source_positions {
            let source_base = source * channels;
            for ch in 0..channels {
                let a = unsafe { _mm256_set1_ps(*a_row_lmajor.get_unchecked(source_base + ch)) };
                let w = unsafe { _mm256_loadu_ps(b.as_ptr().add(kk * n + col)) };
                acc = _mm256_fmadd_ps(a, w, acc);
                kk += 1;
            }
        }
        unsafe {
            _mm256_storeu_ps(c_row.as_mut_ptr().add(col), acc);
        }
        col += 8;
    }
    if col < n {
        uma_so2_f32_lmajor_direct_matmul_row_scalar_tail(
            a_row_lmajor,
            channels,
            source_positions,
            b,
            n,
            c_row,
            accumulate,
            col,
        );
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn uma_so2_f32_lmajor_direct_matmul_rows4_avx2(
    a_block_lmajor: &[f32],
    channels: usize,
    source_positions: &[usize],
    lmajor_row_stride: usize,
    b: &[f32],
    n: usize,
    c_block: &mut [f32],
    accumulate: bool,
) {
    use std::arch::x86_64::{
        __m256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
    };

    let mut col = 0usize;
    while col + 16 <= n {
        let mut r0c0: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(col)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r0c1: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(col + 8)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r1c0: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(n + col)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r1c1: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(n + col + 8)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r2c0: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(2 * n + col)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r2c1: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(2 * n + col + 8)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r3c0: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(3 * n + col)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r3c1: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(3 * n + col + 8)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut kk = 0usize;
        for &source in source_positions {
            let source_base = source * channels;
            for ch in 0..channels {
                let b_base = unsafe { b.as_ptr().add(kk * n + col) };
                let w0 = unsafe { _mm256_loadu_ps(b_base) };
                let w1 = unsafe { _mm256_loadu_ps(b_base.add(8)) };
                let a0 = unsafe { _mm256_set1_ps(*a_block_lmajor.get_unchecked(source_base + ch)) };
                let a1 = unsafe {
                    _mm256_set1_ps(
                        *a_block_lmajor.get_unchecked(lmajor_row_stride + source_base + ch),
                    )
                };
                let a2 = unsafe {
                    _mm256_set1_ps(
                        *a_block_lmajor.get_unchecked(2 * lmajor_row_stride + source_base + ch),
                    )
                };
                let a3 = unsafe {
                    _mm256_set1_ps(
                        *a_block_lmajor.get_unchecked(3 * lmajor_row_stride + source_base + ch),
                    )
                };
                r0c0 = _mm256_fmadd_ps(a0, w0, r0c0);
                r0c1 = _mm256_fmadd_ps(a0, w1, r0c1);
                r1c0 = _mm256_fmadd_ps(a1, w0, r1c0);
                r1c1 = _mm256_fmadd_ps(a1, w1, r1c1);
                r2c0 = _mm256_fmadd_ps(a2, w0, r2c0);
                r2c1 = _mm256_fmadd_ps(a2, w1, r2c1);
                r3c0 = _mm256_fmadd_ps(a3, w0, r3c0);
                r3c1 = _mm256_fmadd_ps(a3, w1, r3c1);
                kk += 1;
            }
        }
        unsafe {
            _mm256_storeu_ps(c_block.as_mut_ptr().add(col), r0c0);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(col + 8), r0c1);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(n + col), r1c0);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(n + col + 8), r1c1);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(2 * n + col), r2c0);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(2 * n + col + 8), r2c1);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(3 * n + col), r3c0);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(3 * n + col + 8), r3c1);
        }
        col += 16;
    }
    while col + 8 <= n {
        let mut r0: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(col)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r1: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(n + col)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r2: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(2 * n + col)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut r3: __m256 = if accumulate {
            unsafe { _mm256_loadu_ps(c_block.as_ptr().add(3 * n + col)) }
        } else {
            _mm256_setzero_ps()
        };
        let mut kk = 0usize;
        for &source in source_positions {
            let source_base = source * channels;
            for ch in 0..channels {
                let w = unsafe { _mm256_loadu_ps(b.as_ptr().add(kk * n + col)) };
                let a0 = unsafe { _mm256_set1_ps(*a_block_lmajor.get_unchecked(source_base + ch)) };
                let a1 = unsafe {
                    _mm256_set1_ps(
                        *a_block_lmajor.get_unchecked(lmajor_row_stride + source_base + ch),
                    )
                };
                let a2 = unsafe {
                    _mm256_set1_ps(
                        *a_block_lmajor.get_unchecked(2 * lmajor_row_stride + source_base + ch),
                    )
                };
                let a3 = unsafe {
                    _mm256_set1_ps(
                        *a_block_lmajor.get_unchecked(3 * lmajor_row_stride + source_base + ch),
                    )
                };
                r0 = _mm256_fmadd_ps(a0, w, r0);
                r1 = _mm256_fmadd_ps(a1, w, r1);
                r2 = _mm256_fmadd_ps(a2, w, r2);
                r3 = _mm256_fmadd_ps(a3, w, r3);
                kk += 1;
            }
        }
        unsafe {
            _mm256_storeu_ps(c_block.as_mut_ptr().add(col), r0);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(n + col), r1);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(2 * n + col), r2);
            _mm256_storeu_ps(c_block.as_mut_ptr().add(3 * n + col), r3);
        }
        col += 8;
    }
    if col < n {
        for row in 0..4 {
            uma_so2_f32_lmajor_direct_matmul_row_scalar_tail(
                &a_block_lmajor
                    [row * lmajor_row_stride..row * lmajor_row_stride + lmajor_row_stride],
                channels,
                source_positions,
                b,
                n,
                &mut c_block[row * n..row * n + n],
                accumulate,
                col,
            );
        }
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn uma_so2_f32_lmajor_direct_matmul_row_avx512(
    a_row_lmajor: &[f32],
    channels: usize,
    source_positions: &[usize],
    b: &[f32],
    n: usize,
    c_row: &mut [f32],
    accumulate: bool,
) {
    use std::arch::x86_64::{
        __m512, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_setzero_ps,
        _mm512_storeu_ps,
    };

    let mut col = 0usize;
    while col + 64 <= n {
        let mut acc0: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_row.as_ptr().add(col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut acc1: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_row.as_ptr().add(col + 16)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut acc2: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_row.as_ptr().add(col + 32)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut acc3: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_row.as_ptr().add(col + 48)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut kk = 0usize;
        for &source in source_positions {
            let source_base = source * channels;
            for ch in 0..channels {
                let a = unsafe { _mm512_set1_ps(*a_row_lmajor.get_unchecked(source_base + ch)) };
                let b_base = unsafe { b.as_ptr().add(kk * n + col) };
                let w0 = unsafe { _mm512_loadu_ps(b_base) };
                let w1 = unsafe { _mm512_loadu_ps(b_base.add(16)) };
                let w2 = unsafe { _mm512_loadu_ps(b_base.add(32)) };
                let w3 = unsafe { _mm512_loadu_ps(b_base.add(48)) };
                acc0 = _mm512_fmadd_ps(a, w0, acc0);
                acc1 = _mm512_fmadd_ps(a, w1, acc1);
                acc2 = _mm512_fmadd_ps(a, w2, acc2);
                acc3 = _mm512_fmadd_ps(a, w3, acc3);
                kk += 1;
            }
        }
        unsafe {
            _mm512_storeu_ps(c_row.as_mut_ptr().add(col), acc0);
            _mm512_storeu_ps(c_row.as_mut_ptr().add(col + 16), acc1);
            _mm512_storeu_ps(c_row.as_mut_ptr().add(col + 32), acc2);
            _mm512_storeu_ps(c_row.as_mut_ptr().add(col + 48), acc3);
        }
        col += 64;
    }
    while col + 16 <= n {
        let mut acc: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_row.as_ptr().add(col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut kk = 0usize;
        for &source in source_positions {
            let source_base = source * channels;
            for ch in 0..channels {
                let a = unsafe { _mm512_set1_ps(*a_row_lmajor.get_unchecked(source_base + ch)) };
                let w = unsafe { _mm512_loadu_ps(b.as_ptr().add(kk * n + col)) };
                acc = _mm512_fmadd_ps(a, w, acc);
                kk += 1;
            }
        }
        unsafe {
            _mm512_storeu_ps(c_row.as_mut_ptr().add(col), acc);
        }
        col += 16;
    }
    if col < n {
        uma_so2_f32_lmajor_direct_matmul_row_scalar_tail(
            a_row_lmajor,
            channels,
            source_positions,
            b,
            n,
            c_row,
            accumulate,
            col,
        );
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn uma_so2_f32_lmajor_direct_matmul_rows4_avx512(
    a_block_lmajor: &[f32],
    channels: usize,
    source_positions: &[usize],
    lmajor_row_stride: usize,
    b: &[f32],
    n: usize,
    c_block: &mut [f32],
    accumulate: bool,
) {
    use std::arch::x86_64::{
        __m512, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_setzero_ps,
        _mm512_storeu_ps,
    };

    let mut col = 0usize;
    while col + 32 <= n {
        let mut r0c0: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r0c1: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(col + 16)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r1c0: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r1c1: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(n + col + 16)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r2c0: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(2 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r2c1: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(2 * n + col + 16)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r3c0: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(3 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r3c1: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(3 * n + col + 16)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut kk = 0usize;
        for &source in source_positions {
            let source_base = source * channels;
            for ch in 0..channels {
                let b_base = unsafe { b.as_ptr().add(kk * n + col) };
                let w0 = unsafe { _mm512_loadu_ps(b_base) };
                let w1 = unsafe { _mm512_loadu_ps(b_base.add(16)) };
                let a0 = unsafe { _mm512_set1_ps(*a_block_lmajor.get_unchecked(source_base + ch)) };
                let a1 = unsafe {
                    _mm512_set1_ps(
                        *a_block_lmajor.get_unchecked(lmajor_row_stride + source_base + ch),
                    )
                };
                let a2 = unsafe {
                    _mm512_set1_ps(
                        *a_block_lmajor.get_unchecked(2 * lmajor_row_stride + source_base + ch),
                    )
                };
                let a3 = unsafe {
                    _mm512_set1_ps(
                        *a_block_lmajor.get_unchecked(3 * lmajor_row_stride + source_base + ch),
                    )
                };
                r0c0 = _mm512_fmadd_ps(a0, w0, r0c0);
                r0c1 = _mm512_fmadd_ps(a0, w1, r0c1);
                r1c0 = _mm512_fmadd_ps(a1, w0, r1c0);
                r1c1 = _mm512_fmadd_ps(a1, w1, r1c1);
                r2c0 = _mm512_fmadd_ps(a2, w0, r2c0);
                r2c1 = _mm512_fmadd_ps(a2, w1, r2c1);
                r3c0 = _mm512_fmadd_ps(a3, w0, r3c0);
                r3c1 = _mm512_fmadd_ps(a3, w1, r3c1);
                kk += 1;
            }
        }
        unsafe {
            _mm512_storeu_ps(c_block.as_mut_ptr().add(col), r0c0);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(col + 16), r0c1);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(n + col), r1c0);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(n + col + 16), r1c1);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(2 * n + col), r2c0);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(2 * n + col + 16), r2c1);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(3 * n + col), r3c0);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(3 * n + col + 16), r3c1);
        }
        col += 32;
    }
    while col + 16 <= n {
        let mut r0: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r1: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r2: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(2 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r3: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(3 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut kk = 0usize;
        for &source in source_positions {
            let source_base = source * channels;
            for ch in 0..channels {
                let w = unsafe { _mm512_loadu_ps(b.as_ptr().add(kk * n + col)) };
                let a0 = unsafe { _mm512_set1_ps(*a_block_lmajor.get_unchecked(source_base + ch)) };
                let a1 = unsafe {
                    _mm512_set1_ps(
                        *a_block_lmajor.get_unchecked(lmajor_row_stride + source_base + ch),
                    )
                };
                let a2 = unsafe {
                    _mm512_set1_ps(
                        *a_block_lmajor.get_unchecked(2 * lmajor_row_stride + source_base + ch),
                    )
                };
                let a3 = unsafe {
                    _mm512_set1_ps(
                        *a_block_lmajor.get_unchecked(3 * lmajor_row_stride + source_base + ch),
                    )
                };
                r0 = _mm512_fmadd_ps(a0, w, r0);
                r1 = _mm512_fmadd_ps(a1, w, r1);
                r2 = _mm512_fmadd_ps(a2, w, r2);
                r3 = _mm512_fmadd_ps(a3, w, r3);
                kk += 1;
            }
        }
        unsafe {
            _mm512_storeu_ps(c_block.as_mut_ptr().add(col), r0);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(n + col), r1);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(2 * n + col), r2);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(3 * n + col), r3);
        }
        col += 16;
    }
    if col < n {
        for row in 0..4 {
            uma_so2_f32_lmajor_direct_matmul_row_scalar_tail(
                &a_block_lmajor
                    [row * lmajor_row_stride..row * lmajor_row_stride + lmajor_row_stride],
                channels,
                source_positions,
                b,
                n,
                &mut c_block[row * n..row * n + n],
                accumulate,
                col,
            );
        }
    }
}

#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn uma_so2_f32_lmajor_direct_matmul_rows8x16_avx512(
    a_block_lmajor: &[f32],
    channels: usize,
    source_positions: &[usize],
    lmajor_row_stride: usize,
    b: &[f32],
    n: usize,
    c_block: &mut [f32],
    accumulate: bool,
) {
    use std::arch::x86_64::{
        __m512, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_setzero_ps,
        _mm512_storeu_ps,
    };

    let mut col = 0usize;
    while col + 16 <= n {
        let mut r0: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r1: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r2: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(2 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r3: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(3 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r4: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(4 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r5: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(5 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r6: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(6 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut r7: __m512 = if accumulate {
            unsafe { _mm512_loadu_ps(c_block.as_ptr().add(7 * n + col)) }
        } else {
            _mm512_setzero_ps()
        };
        let mut kk = 0usize;
        for &source in source_positions {
            let source_base = source * channels;
            for ch in 0..channels {
                let w = unsafe { _mm512_loadu_ps(b.as_ptr().add(kk * n + col)) };
                let a0 = unsafe { _mm512_set1_ps(*a_block_lmajor.get_unchecked(source_base + ch)) };
                let a1 = unsafe {
                    _mm512_set1_ps(
                        *a_block_lmajor.get_unchecked(lmajor_row_stride + source_base + ch),
                    )
                };
                let a2 = unsafe {
                    _mm512_set1_ps(
                        *a_block_lmajor.get_unchecked(2 * lmajor_row_stride + source_base + ch),
                    )
                };
                let a3 = unsafe {
                    _mm512_set1_ps(
                        *a_block_lmajor.get_unchecked(3 * lmajor_row_stride + source_base + ch),
                    )
                };
                let a4 = unsafe {
                    _mm512_set1_ps(
                        *a_block_lmajor.get_unchecked(4 * lmajor_row_stride + source_base + ch),
                    )
                };
                let a5 = unsafe {
                    _mm512_set1_ps(
                        *a_block_lmajor.get_unchecked(5 * lmajor_row_stride + source_base + ch),
                    )
                };
                let a6 = unsafe {
                    _mm512_set1_ps(
                        *a_block_lmajor.get_unchecked(6 * lmajor_row_stride + source_base + ch),
                    )
                };
                let a7 = unsafe {
                    _mm512_set1_ps(
                        *a_block_lmajor.get_unchecked(7 * lmajor_row_stride + source_base + ch),
                    )
                };
                r0 = _mm512_fmadd_ps(a0, w, r0);
                r1 = _mm512_fmadd_ps(a1, w, r1);
                r2 = _mm512_fmadd_ps(a2, w, r2);
                r3 = _mm512_fmadd_ps(a3, w, r3);
                r4 = _mm512_fmadd_ps(a4, w, r4);
                r5 = _mm512_fmadd_ps(a5, w, r5);
                r6 = _mm512_fmadd_ps(a6, w, r6);
                r7 = _mm512_fmadd_ps(a7, w, r7);
                kk += 1;
            }
        }
        unsafe {
            _mm512_storeu_ps(c_block.as_mut_ptr().add(col), r0);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(n + col), r1);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(2 * n + col), r2);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(3 * n + col), r3);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(4 * n + col), r4);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(5 * n + col), r5);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(6 * n + col), r6);
            _mm512_storeu_ps(c_block.as_mut_ptr().add(7 * n + col), r7);
        }
        col += 16;
    }
    if col < n {
        for row in 0..8 {
            uma_so2_f32_lmajor_direct_matmul_row_scalar_tail(
                &a_block_lmajor
                    [row * lmajor_row_stride..row * lmajor_row_stride + lmajor_row_stride],
                channels,
                source_positions,
                b,
                n,
                &mut c_block[row * n..row * n + n],
                accumulate,
                col,
            );
        }
    }
}

/// Fused SO2 m-conv forward for f32 using a precomputed fused weight.
///
/// Like `so2_mconv_forward_set_fused_f32` but takes the prebuilt
/// `[2*in_dim, 2*out_dim]` fused weight — no rebuild overhead.
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub(crate) fn so2_mconv_forward_set_precomputed_fused_f32(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    fused_weight: &[f32], // precomputed [2*in_dim, 2*out_dim]
    input: &[f32],
    out: &mut [f32],
) {
    let fused_in_dim = 2 * in_dim;
    let fused_out_dim = 2 * out_dim;
    debug_assert!(fused_weight.len() >= fused_in_dim * fused_out_dim);
    debug_assert!(input.len() >= rows * fused_in_dim);
    debug_assert!(out.len() >= rows * fused_out_dim);
    let trace_start = uma_blas_trace_start();
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    if uma_so2_f32_try_parallel_row_matmul(
        rows,
        fused_in_dim,
        fused_out_dim,
        input,
        fused_weight,
        out,
        false,
        true,
        None,
    ) {
        let path = if uma_so2_f32_avx512_enabled() {
            "row_avx512"
        } else {
            "row_avx2"
        };
        trace_uma_so2_f32_matmul(
            "forward_precomputed",
            rows,
            fused_in_dim,
            fused_out_dim,
            false,
            path,
            trace_start,
        );
        return;
    }
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            blas_dim(rows),
            blas_dim(fused_out_dim),
            blas_dim(fused_in_dim),
            1.0_f32,
            input.as_ptr(),
            blas_dim(fused_in_dim),
            fused_weight.as_ptr(),
            blas_dim(fused_out_dim),
            0.0_f32,
            out.as_mut_ptr(),
            blas_dim(fused_out_dim),
        );
    }
    trace_uma_so2_f32_matmul(
        "forward_precomputed",
        rows,
        fused_in_dim,
        fused_out_dim,
        false,
        "cblas",
        trace_start,
    );
}

/// Fused SO2 m-conv forward for f64 using a precomputed fused weight.
///
/// Like `so2_mconv_forward_set_precomputed_fused_f32` but calls DGEMM.
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(dead_code)]
pub(crate) fn so2_mconv_forward_set_precomputed_fused(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    fused_weight: &[f64],
    input: &[f64],
    out: &mut [f64],
) {
    let fused_in_dim = 2 * in_dim;
    let fused_out_dim = 2 * out_dim;
    debug_assert!(fused_weight.len() >= fused_in_dim * fused_out_dim);
    debug_assert!(input.len() >= rows * fused_in_dim);
    debug_assert!(out.len() >= rows * fused_out_dim);
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    if uma_so2_f64_try_parallel_row_matmul(
        rows,
        fused_in_dim,
        fused_out_dim,
        input,
        fused_weight,
        out,
        false,
        true,
    ) {
        return;
    }
    unsafe {
        cblas_dgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            blas_dim(rows),
            blas_dim(fused_out_dim),
            blas_dim(fused_in_dim),
            1.0_f64,
            input.as_ptr(),
            blas_dim(fused_in_dim),
            fused_weight.as_ptr(),
            blas_dim(fused_out_dim),
            0.0_f64,
            out.as_mut_ptr(),
            blas_dim(fused_out_dim),
        );
    }
}

/// Fused SO2 m-conv VJP for f64 using a precomputed fused weight.
///
/// Like `so2_mconv_vjp_input_add_precomputed_fused_f32` but calls DGEMM.
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(dead_code)]
pub(crate) fn so2_mconv_vjp_input_add_precomputed_fused(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    fused_weight: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
) {
    let fused_in_dim = 2 * in_dim;
    let fused_out_dim = 2 * out_dim;
    debug_assert!(fused_weight.len() >= fused_out_dim * fused_in_dim);
    debug_assert!(grad_out.len() >= rows * fused_out_dim);
    debug_assert!(grad_input.len() >= rows * fused_in_dim);
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    if uma_so2_f64_try_parallel_row_matmul(
        rows,
        fused_out_dim,
        fused_in_dim,
        grad_out,
        fused_weight,
        grad_input,
        true,
        true,
    ) {
        return;
    }
    unsafe {
        cblas_dgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            blas_dim(rows),
            blas_dim(fused_in_dim),
            blas_dim(fused_out_dim),
            1.0_f64,
            grad_out.as_ptr(),
            blas_dim(fused_out_dim),
            fused_weight.as_ptr(),
            blas_dim(fused_in_dim),
            1.0_f64,
            grad_input.as_mut_ptr(),
            blas_dim(fused_in_dim),
        );
    }
}

/// Overwrite-mode fused SO2 m-conv VJP for f64 using a precomputed fused weight.
///
/// This is only for fresh scratch buffers whose complete `[rows, 2*in_dim]`
/// output is written by the GEMM. Callers that need accumulation must use
/// `so2_mconv_vjp_input_add_precomputed_fused`.
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(dead_code)]
pub(crate) fn so2_mconv_vjp_input_write_precomputed_fused(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    fused_weight: &[f64],
    grad_out: &[f64],
    grad_input: &mut [f64],
) {
    let fused_in_dim = 2 * in_dim;
    let fused_out_dim = 2 * out_dim;
    debug_assert!(fused_weight.len() >= fused_out_dim * fused_in_dim);
    debug_assert!(grad_out.len() >= rows * fused_out_dim);
    debug_assert!(grad_input.len() >= rows * fused_in_dim);
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    if uma_so2_f64_try_parallel_row_matmul(
        rows,
        fused_out_dim,
        fused_in_dim,
        grad_out,
        fused_weight,
        grad_input,
        false,
        true,
    ) {
        return;
    }
    unsafe {
        cblas_dgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            blas_dim(rows),
            blas_dim(fused_in_dim),
            blas_dim(fused_out_dim),
            1.0_f64,
            grad_out.as_ptr(),
            blas_dim(fused_out_dim),
            fused_weight.as_ptr(),
            blas_dim(fused_in_dim),
            0.0_f64,
            grad_input.as_mut_ptr(),
            blas_dim(fused_in_dim),
        );
    }
}

/// Fused SO2 m-conv forward for f32 with strided output rows.
///
/// Computes the same `out = input @ fused_weight` as
/// `so2_mconv_forward_set_precomputed_fused_f32`, but writes each output row
/// with caller-provided `out_row_stride`. This lets UMA order conv1 write
/// directly into the final l-major row layout without a compact
/// `conv1_blk_out` followed by scatter.
#[allow(dead_code, unsafe_code)]
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub(crate) unsafe fn so2_mconv_forward_set_precomputed_fused_f32_strided_output_raw(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    fused_weight: &[f32], // precomputed [2*in_dim, 2*out_dim]
    input: &[f32],
    out: *mut f32,
    out_row_stride: usize,
) {
    let fused_in_dim = 2 * in_dim;
    let fused_out_dim = 2 * out_dim;
    debug_assert!(fused_weight.len() >= fused_in_dim * fused_out_dim);
    debug_assert!(input.len() >= rows * fused_in_dim);
    debug_assert!(out_row_stride >= fused_out_dim);
    debug_assert!(rows == 0 || !out.is_null());
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            rows as c_int,
            fused_out_dim as c_int,
            fused_in_dim as c_int,
            1.0_f32,
            input.as_ptr(),
            fused_in_dim as c_int,
            fused_weight.as_ptr(),
            fused_out_dim as c_int,
            0.0_f32,
            out,
            out_row_stride as c_int,
        );
    }
}

/// Fused SO2 m-conv VJP for f32 using a precomputed fused weight.
///
/// Like `so2_mconv_vjp_input_add_fused_f32` but takes the prebuilt
/// `[2*out_dim, 2*in_dim]` fused weight — no rebuild overhead.
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(dead_code)]
pub(crate) fn so2_mconv_vjp_input_add_precomputed_fused_f32(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    fused_weight: &[f32], // precomputed [2*out_dim, 2*in_dim]
    grad_out: &[f32],
    grad_input: &mut [f32],
) {
    so2_mconv_vjp_input_precomputed_fused_f32(
        rows,
        in_dim,
        out_dim,
        fused_weight,
        grad_out,
        grad_input,
        true,
    );
}

/// Overwrite-mode fused SO2 m-conv VJP for f32 using a precomputed fused weight.
///
/// This is only for fresh scratch buffers whose complete `[rows, 2*in_dim]`
/// output is written by the GEMM. Callers that need accumulation must use
/// `so2_mconv_vjp_input_add_precomputed_fused_f32`.
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(dead_code)]
pub(crate) fn so2_mconv_vjp_input_write_precomputed_fused_f32(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    fused_weight: &[f32], // precomputed [2*out_dim, 2*in_dim]
    grad_out: &[f32],
    grad_input: &mut [f32],
) {
    so2_mconv_vjp_input_precomputed_fused_f32(
        rows,
        in_dim,
        out_dim,
        fused_weight,
        grad_out,
        grad_input,
        false,
    );
}

#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(dead_code)]
fn so2_mconv_vjp_input_precomputed_fused_f32(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    fused_weight: &[f32], // precomputed [2*out_dim, 2*in_dim]
    grad_out: &[f32],
    grad_input: &mut [f32],
    accumulate: bool,
) {
    let fused_in_dim = 2 * in_dim;
    let fused_out_dim = 2 * out_dim;
    debug_assert!(fused_weight.len() >= fused_out_dim * fused_in_dim);
    debug_assert!(grad_out.len() >= rows * fused_out_dim);
    debug_assert!(grad_input.len() >= rows * fused_in_dim);
    let trace_start = uma_blas_trace_start();
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    if uma_so2_f32_try_parallel_row_matmul(
        rows,
        fused_out_dim,
        fused_in_dim,
        grad_out,
        fused_weight,
        grad_input,
        accumulate,
        true,
        None,
    ) {
        let path = if uma_so2_f32_avx512_enabled() {
            "row_avx512"
        } else {
            "row_avx2"
        };
        trace_uma_so2_f32_matmul(
            "vjp_precomputed",
            rows,
            fused_out_dim,
            fused_in_dim,
            accumulate,
            path,
            trace_start,
        );
        return;
    }
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            blas_dim(rows),
            blas_dim(fused_in_dim),
            blas_dim(fused_out_dim),
            1.0_f32,
            grad_out.as_ptr(),
            blas_dim(fused_out_dim),
            fused_weight.as_ptr(),
            blas_dim(fused_in_dim),
            if accumulate { 1.0_f32 } else { 0.0_f32 },
            grad_input.as_mut_ptr(),
            blas_dim(fused_in_dim),
        );
    }
    trace_uma_so2_f32_matmul(
        "vjp_precomputed",
        rows,
        fused_out_dim,
        fused_in_dim,
        accumulate,
        "cblas",
        trace_start,
    );
}

/// Fused SO2 m-conv VJP for f32 using a precomputed fused weight.
///
/// Sets `grad_input = grad_out @ fused_weight` with `beta=0`, avoiding a
/// prior zero-fill when the destination is scratch storage.
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub(crate) fn so2_mconv_vjp_input_set_precomputed_fused_f32(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    fused_weight: &[f32], // precomputed [2*out_dim, 2*in_dim]
    grad_out: &[f32],
    grad_input: &mut [f32],
) {
    let fused_in_dim = 2 * in_dim;
    let fused_out_dim = 2 * out_dim;
    debug_assert!(fused_weight.len() >= fused_out_dim * fused_in_dim);
    debug_assert!(grad_out.len() >= rows * fused_out_dim);
    debug_assert!(grad_input.len() >= rows * fused_in_dim);
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            rows as c_int,
            fused_in_dim as c_int,
            fused_out_dim as c_int,
            1.0_f32,
            grad_out.as_ptr(),
            fused_out_dim as c_int,
            fused_weight.as_ptr(),
            fused_in_dim as c_int,
            0.0_f32,
            grad_input.as_mut_ptr(),
            fused_in_dim as c_int,
        );
    }
}

/// Fused SO2 m-conv VJP for f32 with strided `grad_out` rows.
///
/// Computes `grad_input = grad_out @ fused_weight` with `beta=0`, where each
/// `grad_out` row starts `grad_out_row_stride` elements after the previous row.
/// This is used when UMA keeps order conv2 gradients in an order-packed
/// full-row buffer but still wants one ArmPL SGEMM per order.
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(
        any(feature = "intel-mkl", feature = "openblas"),
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
#[allow(dead_code)]
pub(crate) fn so2_mconv_vjp_input_set_precomputed_fused_f32_strided_grad_out(
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    fused_weight: &[f32], // precomputed [2*out_dim, 2*in_dim]
    grad_out: &[f32],
    grad_out_row_stride: usize,
    grad_input: &mut [f32],
) {
    let fused_in_dim = 2 * in_dim;
    let fused_out_dim = 2 * out_dim;
    debug_assert!(fused_weight.len() >= fused_out_dim * fused_in_dim);
    debug_assert!(grad_out_row_stride >= fused_out_dim);
    debug_assert!(grad_out.len() >= strided_matrix_len(rows, grad_out_row_stride, fused_out_dim));
    debug_assert!(grad_input.len() >= rows * fused_in_dim);
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            rows as c_int,
            fused_in_dim as c_int,
            fused_out_dim as c_int,
            1.0_f32,
            grad_out.as_ptr(),
            grad_out_row_stride as c_int,
            fused_weight.as_ptr(),
            fused_in_dim as c_int,
            0.0_f32,
            grad_input.as_mut_ptr(),
            fused_in_dim as c_int,
        );
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub(crate) fn dense_forward_output_major_weight_column_block_add_f32(
    input: &[f32],
    rows: usize,
    selected_in_dim: usize,
    weight: &[f32],
    full_in_dim: usize,
    column_offset: usize,
    out_dim: usize,
    out: &mut [f32],
    alpha: f32,
    beta: f32,
) {
    debug_assert!(column_offset <= full_in_dim);
    debug_assert!(selected_in_dim <= full_in_dim - column_offset);
    debug_assert!(input.len() >= rows * selected_in_dim);
    debug_assert!(weight.len() >= out_dim * full_in_dim);
    debug_assert!(out.len() >= rows * out_dim);
    if rows == 0 || out_dim == 0 {
        return;
    }
    if selected_in_dim == 0 {
        apply_beta_contiguous_f32(&mut out[..rows * out_dim], beta);
        return;
    }
    unsafe {
        // Invariant: input is [rows, selected_in_dim], weight is output-major
        // [out_dim, full_in_dim], and selected weight columns are contiguous
        // from column_offset while preserving full_in_dim row stride.
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            blas_dim(rows),
            blas_dim(out_dim),
            blas_dim(selected_in_dim),
            alpha,
            input.as_ptr(),
            blas_dim(selected_in_dim),
            weight.as_ptr().add(column_offset),
            blas_dim(full_in_dim),
            beta,
            out.as_mut_ptr(),
            blas_dim(out_dim),
        );
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
#[cfg(any(
    all(feature = "accelerate-blas", target_os = "macos"),
    all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
    all(
        feature = "openblas",
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
pub(crate) fn dense_vjp_input_output_major_weight_column_block_add_f32(
    rows: usize,
    selected_in_dim: usize,
    full_in_dim: usize,
    out_dim: usize,
    column_offset: usize,
    weight: &[f32],
    bar_out: &[f32],
    bar_input: &mut [f32],
    alpha: f32,
    beta: f32,
) {
    debug_assert!(column_offset <= full_in_dim);
    debug_assert!(selected_in_dim <= full_in_dim - column_offset);
    debug_assert!(weight.len() >= out_dim * full_in_dim);
    debug_assert!(bar_out.len() >= rows * out_dim);
    debug_assert!(bar_input.len() >= rows * selected_in_dim);
    if rows == 0 || selected_in_dim == 0 {
        return;
    }
    if out_dim == 0 {
        apply_beta_contiguous_f32(&mut bar_input[..rows * selected_in_dim], beta);
        return;
    }
    unsafe {
        // Invariant: bar_out is [rows, out_dim], weight is output-major
        // [out_dim, full_in_dim], and bar_input is [rows, selected_in_dim].
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            blas_dim(rows),
            blas_dim(selected_in_dim),
            blas_dim(out_dim),
            alpha,
            bar_out.as_ptr(),
            blas_dim(out_dim),
            weight.as_ptr().add(column_offset),
            blas_dim(full_in_dim),
            beta,
            bar_input.as_mut_ptr(),
            blas_dim(selected_in_dim),
        );
    }
}

#[cfg(all(
    test,
    feature = "orb-fixed-shape-dense",
    feature = "intel-mkl",
    target_os = "linux",
    target_arch = "x86_64"
))]
mod mkl_packed_tests {
    use super::*;

    fn deterministic_f64(len: usize, scale: f64) -> Vec<f64> {
        (0..len)
            .map(|index| ((index as f64 + 1.0) * 0.173_205_080_756_887_73).sin() * scale)
            .collect()
    }

    fn assert_close_slices(label: &str, actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
        let mut max_delta = 0.0_f64;
        for (&actual_value, &expected_value) in actual.iter().zip(expected) {
            max_delta = max_delta.max((actual_value - expected_value).abs());
        }
        assert!(
            max_delta <= tolerance,
            "{label} max delta {max_delta:.3e} exceeded tolerance {tolerance:.3e}"
        );
    }

    #[test]
    fn mkl_packed_output_major_forward_and_vjp_match_generic_dgemm() -> Result<(), String> {
        let rows = 9;
        let input_dim = 7;
        let output_dim = 5;
        let input = deterministic_f64(rows * input_dim, 0.7);
        let weight = deterministic_f64(output_dim * input_dim, 0.2);

        let packed_forward = MklPackedDgemmBMatrix::pack_output_major_weight(
            &weight, rows, output_dim, input_dim, true,
        )?;
        let actual = packed_forward.compute_no_trans_a_alloc(&input);
        let expected = dense_forward_output_major_weight_alloc(
            &input, rows, input_dim, &weight, output_dim, 1.0,
        );
        assert_close_slices("packed forward", &actual, &expected, 1.0e-10);

        let mut actual_add = deterministic_f64(rows * output_dim, 0.05);
        let mut expected_add = actual_add.clone();
        packed_forward.compute_no_trans_a_into(&input, &mut actual_add, 0.25);
        dense_forward_output_major_weight(
            &input,
            rows,
            input_dim,
            &weight,
            &mut expected_add,
            output_dim,
            1.0,
            0.25,
        );
        assert_close_slices("packed forward beta", &actual_add, &expected_add, 1.0e-10);

        let bar_out = deterministic_f64(rows * output_dim, 0.6);
        let packed_vjp = MklPackedDgemmBMatrix::pack_output_major_weight(
            &weight, rows, input_dim, output_dim, false,
        )?;
        let actual = packed_vjp.compute_no_trans_a_alloc(&bar_out);
        let expected = dense_vjp_input_output_major_weight_alloc(
            rows, input_dim, output_dim, &weight, &bar_out, 1.0,
        );
        assert_close_slices("packed input VJP", &actual, &expected, 1.0e-10);

        let mut actual_add = deterministic_f64(rows * input_dim, 0.03);
        let mut expected_add = actual_add.clone();
        packed_vjp.compute_no_trans_a_into(&bar_out, &mut actual_add, 0.5);
        dense_vjp_input_output_major_weight(
            rows,
            input_dim,
            output_dim,
            &weight,
            &bar_out,
            &mut expected_add,
            1.0,
            0.5,
        );
        assert_close_slices("packed input VJP beta", &actual_add, &expected_add, 1.0e-10);
        Ok(())
    }

    #[test]
    fn mkl_packed_strided_column_block_matches_generic_dgemm() -> Result<(), String> {
        let rows = 8;
        let selected_in_dim = 4;
        let full_in_dim = 11;
        let column_offset = 5;
        let output_dim = 6;
        let input = deterministic_f64(rows * selected_in_dim, 0.4);
        let weight = deterministic_f64(output_dim * full_in_dim, 0.25);

        let packed_forward = MklPackedDgemmBMatrix::pack_strided_b_matrix(
            &weight,
            column_offset,
            full_in_dim,
            rows,
            output_dim,
            selected_in_dim,
            true,
        )?;
        let actual = packed_forward.compute_no_trans_a_alloc(&input);
        let mut expected = vec![0.0; rows * output_dim];
        dense_forward_output_major_weight_column_block_add(
            &input,
            rows,
            selected_in_dim,
            &weight,
            full_in_dim,
            column_offset,
            output_dim,
            &mut expected,
            1.0,
            0.0,
        );
        assert_close_slices("packed strided forward", &actual, &expected, 1.0e-10);

        let bar_out = deterministic_f64(rows * output_dim, 0.3);
        let packed_vjp = MklPackedDgemmBMatrix::pack_strided_b_matrix(
            &weight,
            column_offset,
            full_in_dim,
            rows,
            selected_in_dim,
            output_dim,
            false,
        )?;
        let actual = packed_vjp.compute_no_trans_a_alloc(&bar_out);
        let mut expected = vec![0.0; rows * selected_in_dim];
        dense_vjp_input_output_major_weight_column_block_add(
            rows,
            selected_in_dim,
            full_in_dim,
            output_dim,
            column_offset,
            &weight,
            &bar_out,
            &mut expected,
            1.0,
            0.0,
        );
        assert_close_slices("packed strided input VJP", &actual, &expected, 1.0e-10);
        Ok(())
    }
}

#[cfg(all(test, feature = "parallel", target_arch = "x86_64"))]
mod row_parallel_gemm_tests {
    use super::*;

    fn deterministic_f64(len: usize, scale: f64) -> Vec<f64> {
        (0..len)
            .map(|index| ((index as f64 + 1.0) * 0.141_421_356_237_309_5).cos() * scale)
            .collect()
    }

    fn assert_close_slices(label: &str, actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
        let mut max_delta = 0.0_f64;
        for (&actual_value, &expected_value) in actual.iter().zip(expected) {
            max_delta = max_delta.max((actual_value - expected_value).abs());
        }
        assert!(
            max_delta <= tolerance,
            "{label} max delta {max_delta:.3e} exceeded tolerance {tolerance:.3e}"
        );
    }

    #[test]
    fn row_parallel_output_major_forward_and_vjp_match_generic_dgemm() {
        let rows = 96;
        let input_dim = 128;
        let output_dim = 128;
        let input = deterministic_f64(rows * input_dim, 0.7);
        let weight = deterministic_f64(output_dim * input_dim, 0.2);
        let bar_out = deterministic_f64(rows * output_dim, 0.6);
        let pool = match rayon::ThreadPoolBuilder::new().num_threads(4).build() {
            Ok(pool) => pool,
            Err(_) => return,
        };

        pool.install(|| {
            let mut actual = deterministic_f64(rows * output_dim, 0.03);
            let mut expected = actual.clone();
            dense_forward_output_major_weight_row_parallel_f64(
                &input,
                rows,
                input_dim,
                &weight,
                &mut actual,
                output_dim,
                0.75,
                0.25,
            );
            dense_forward_output_major_weight(
                &input,
                rows,
                input_dim,
                &weight,
                &mut expected,
                output_dim,
                0.75,
                0.25,
            );
            assert_close_slices("row-parallel forward", &actual, &expected, 1.0e-10);

            let mut actual = deterministic_f64(rows * input_dim, 0.04);
            let mut expected = actual.clone();
            dense_vjp_input_output_major_weight_row_parallel_f64(
                rows,
                input_dim,
                output_dim,
                &weight,
                &bar_out,
                &mut actual,
                0.5,
                0.125,
            );
            dense_vjp_input_output_major_weight(
                rows,
                input_dim,
                output_dim,
                &weight,
                &bar_out,
                &mut expected,
                0.5,
                0.125,
            );
            assert_close_slices("row-parallel input VJP", &actual, &expected, 1.0e-10);
        });
    }

    #[test]
    fn row_parallel_strided_forward_and_vjp_match_generic_dgemm() {
        let rows = 96;
        let input_dim = 96;
        let output_dim = 112;
        assert!(rows * input_dim * output_dim >= row_parallel_gemm_min_ops());
        let input_stride = 128;
        let output_stride = 144;
        let mut input = vec![0.0_f64; rows * input_stride];
        let mut bar_out = vec![0.0_f64; rows * output_stride];
        let weight = deterministic_f64(input_dim * output_dim, 0.15);
        for row in 0..rows {
            let input_start = row * input_stride;
            input[input_start..input_start + input_dim]
                .copy_from_slice(&deterministic_f64(input_dim, 0.3 + row as f64 * 0.001));
            let bar_out_start = row * output_stride;
            bar_out[bar_out_start..bar_out_start + output_dim]
                .copy_from_slice(&deterministic_f64(output_dim, 0.2 + row as f64 * 0.001));
        }
        let pool = match rayon::ThreadPoolBuilder::new().num_threads(4).build() {
            Ok(pool) => pool,
            Err(_) => return,
        };

        pool.install(|| {
            let mut actual = deterministic_f64(rows * output_stride, 0.02);
            let mut expected = actual.clone();
            dense_forward_strided_row_parallel_f64(
                &input,
                rows,
                input_dim,
                input_stride,
                &weight,
                &mut actual,
                output_dim,
                output_stride,
                0.75,
                0.25,
            );
            dense_forward_strided(
                &input,
                rows,
                input_dim,
                input_stride,
                &weight,
                &mut expected,
                output_dim,
                output_stride,
                0.75,
                0.25,
            );
            assert_close_slices("row-parallel strided forward", &actual, &expected, 1.0e-10);

            let mut actual = deterministic_f64(rows * input_stride, 0.03);
            let mut expected = actual.clone();
            dense_vjp_input_strided_row_parallel_f64(
                rows,
                input_dim,
                &weight,
                output_dim,
                &bar_out,
                output_stride,
                &mut actual,
                input_stride,
                0.5,
                0.125,
            );
            dense_vjp_input_strided(
                rows,
                input_dim,
                &weight,
                output_dim,
                &bar_out,
                output_stride,
                &mut expected,
                input_stride,
                0.5,
                0.125,
            );
            assert_close_slices(
                "row-parallel strided input VJP",
                &actual,
                &expected,
                1.0e-10,
            );
        });
    }
}

#[cfg(all(
    test,
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        all(feature = "libxsmm", target_os = "linux", target_arch = "x86_64")
    )
))]
mod weighted_sum_tests {
    use super::*;

    fn deterministic_f64(len: usize, scale: f64) -> Vec<f64> {
        (0..len)
            .map(|index| ((index as f64 + 1.0) * 0.271_828_182_845_904_5).cos() * scale)
            .collect()
    }

    fn deterministic_f32(len: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|index| ((index as f32 + 1.0) * 0.271_828_18).cos() * scale)
            .collect()
    }

    #[test]
    fn weighted_sum_rows_f64_matches_scalar_reference() {
        let rows = 7;
        let cols = 257;
        let values = deterministic_f64(rows * cols, 0.35);
        let coefficients = deterministic_f64(rows, 0.6);
        let mut actual = vec![0.0; cols];
        weighted_sum_rows_f64(rows, cols, &values, &coefficients, &mut actual);

        let mut expected = vec![0.0; cols];
        for row in 0..rows {
            let coefficient = coefficients[row];
            let row_values = &values[row * cols..(row + 1) * cols];
            for col in 0..cols {
                expected[col] += coefficient * row_values[col];
            }
        }

        let mut max_delta = 0.0_f64;
        for (&actual_value, &expected_value) in actual.iter().zip(&expected) {
            max_delta = max_delta.max((actual_value - expected_value).abs());
        }
        assert!(
            max_delta <= 1.0e-13,
            "weighted sum max delta {max_delta:.3e} exceeded tolerance"
        );
    }

    #[test]
    fn weighted_sum_rows_f32_matches_scalar_reference() {
        let rows = 7;
        let cols = 257;
        let values = deterministic_f32(rows * cols, 0.35);
        let coefficients = deterministic_f32(rows, 0.6);
        let mut actual = vec![0.0_f32; cols];
        weighted_sum_rows_f32(rows, cols, &values, &coefficients, &mut actual);

        let mut expected = vec![0.0_f32; cols];
        for row in 0..rows {
            let coefficient = coefficients[row];
            let row_values = &values[row * cols..(row + 1) * cols];
            for col in 0..cols {
                expected[col] += coefficient * row_values[col];
            }
        }

        let mut max_delta = 0.0_f32;
        for (&actual_value, &expected_value) in actual.iter().zip(&expected) {
            max_delta = max_delta.max((actual_value - expected_value).abs());
        }
        assert!(
            max_delta <= 1.0e-5,
            "weighted sum f32 max delta {max_delta:.3e} exceeded tolerance"
        );
    }
}

#[cfg(all(
    test,
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(feature = "intel-mkl", target_os = "linux", target_arch = "x86_64"),
        all(
            feature = "openblas",
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )
))]
mod f32_tests {
    use super::*;

    fn deterministic_f32(len: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|index| ((index as f32 + 1.0) * 0.231_840_79).sin() * scale)
            .collect()
    }

    fn assert_close_slices(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
        let mut max_delta = 0.0_f32;
        for (&actual_value, &expected_value) in actual.iter().zip(expected) {
            max_delta = max_delta.max((actual_value - expected_value).abs());
        }
        assert!(
            max_delta <= tolerance,
            "{label} max delta {max_delta:.3e} exceeded tolerance {tolerance:.3e}"
        );
    }

    #[test]
    fn f32_output_major_forward_and_vjp_match_scalar_reference() {
        let rows = 6;
        let input_dim = 7;
        let output_dim = 5;
        let input = deterministic_f32(rows * input_dim, 0.4);
        let weight = deterministic_f32(output_dim * input_dim, 0.2);

        let mut actual = deterministic_f32(rows * output_dim, 0.03);
        let mut expected = actual.clone();
        dense_forward_output_major_weight_f32(
            &input,
            rows,
            input_dim,
            &weight,
            &mut actual,
            output_dim,
            1.0,
            0.25,
        );
        for row in 0..rows {
            for out_channel in 0..output_dim {
                let mut sum = 0.0_f32;
                for in_channel in 0..input_dim {
                    sum += input[row * input_dim + in_channel]
                        * weight[out_channel * input_dim + in_channel];
                }
                expected[row * output_dim + out_channel] =
                    sum + 0.25 * expected[row * output_dim + out_channel];
            }
        }
        assert_close_slices("f32 forward", &actual, &expected, 2.0e-5);

        let bar_out = deterministic_f32(rows * output_dim, 0.35);
        let mut actual = deterministic_f32(rows * input_dim, 0.02);
        let mut expected = actual.clone();
        dense_vjp_input_output_major_weight_f32(
            rows,
            input_dim,
            output_dim,
            &weight,
            &bar_out,
            &mut actual,
            1.0,
            0.5,
        );
        for row in 0..rows {
            for in_channel in 0..input_dim {
                let mut sum = 0.0_f32;
                for out_channel in 0..output_dim {
                    sum += bar_out[row * output_dim + out_channel]
                        * weight[out_channel * input_dim + in_channel];
                }
                expected[row * input_dim + in_channel] =
                    sum + 0.5 * expected[row * input_dim + in_channel];
            }
        }
        assert_close_slices("f32 input VJP", &actual, &expected, 2.0e-5);

        let mut input_major_weight = vec![0.0_f32; weight.len()];
        for out_channel in 0..output_dim {
            for in_channel in 0..input_dim {
                input_major_weight[in_channel * output_dim + out_channel] =
                    weight[out_channel * input_dim + in_channel];
            }
        }
        let mut actual = deterministic_f32(rows * input_dim, 0.02);
        dense_vjp_input_input_major_weight_f32(
            rows,
            input_dim,
            output_dim,
            &input_major_weight,
            &bar_out,
            &mut actual,
            1.0,
            0.5,
        );
        assert_close_slices("f32 input-major input VJP", &actual, &expected, 2.0e-5);
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    fn row_matmul_reference(
        a_row: &[f32],
        b: &[f32],
        k: usize,
        n: usize,
        initial: &[f32],
        accumulate: bool,
    ) -> Vec<f32> {
        let mut out = initial.to_vec();
        for col in 0..n {
            let mut acc = if accumulate { out[col] } else { 0.0_f32 };
            for kk in 0..k {
                acc += a_row[kk] * b[kk * n + col];
            }
            out[col] = acc;
        }
        out
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    fn avx512f_available_or_skip(test_name: &str) -> bool {
        if std::arch::is_x86_feature_detected!("avx512f") {
            true
        } else {
            eprintln!("skipping {test_name}: AVX-512F not detected on this host");
            false
        }
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_so2_f32_row_matmul_scalar_matches_reference() {
        let k = 17;
        let n = 19;
        let a_row = deterministic_f32(k, 0.3);
        let b = deterministic_f32(k * n, 0.2);
        let initial = deterministic_f32(n, 0.05);
        let expected = row_matmul_reference(&a_row, &b, k, n, &initial, true);

        let mut actual = initial.clone();
        uma_so2_f32_matmul_row_scalar(&a_row, &b, k, n, &mut actual, true);
        assert_close_slices("UMA SO2 scalar row matmul", &actual, &expected, 2.0e-6);
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_so2_f32_row_matmul_avx2_matches_reference_when_available() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }

        let k = 33;
        let n = 31;
        let a_row = deterministic_f32(k, 0.3);
        let b = deterministic_f32(k * n, 0.2);
        let initial = deterministic_f32(n, 0.05);
        let expected = row_matmul_reference(&a_row, &b, k, n, &initial, true);

        let mut actual = initial.clone();
        unsafe {
            uma_so2_f32_matmul_row_avx2(&a_row, &b, k, n, &mut actual, true);
        }
        assert_close_slices("UMA SO2 AVX2 row matmul", &actual, &expected, 2.5e-6);
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_so2_f32_rows4_avx2_matches_reference_when_available() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }

        let rows = 4;
        let k = 33;
        let n = 31;
        let a = deterministic_f32(rows * k, 0.3);
        let b = deterministic_f32(k * n, 0.2);
        let mut actual = deterministic_f32(rows * n, 0.05);
        let mut expected = Vec::with_capacity(rows * n);
        for row in 0..rows {
            expected.extend(row_matmul_reference(
                &a[row * k..row * k + k],
                &b,
                k,
                n,
                &actual[row * n..row * n + n],
                true,
            ));
        }

        unsafe {
            uma_so2_f32_matmul_rows4_avx2(&a, &b, k, n, &mut actual, true);
        }
        assert_close_slices("UMA SO2 AVX2 rows4 matmul", &actual, &expected, 2.5e-6);
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_so2_f64_row_matmul_dispatch_matches_reference_when_available() {
        if !avx512f_available_or_skip("uma_so2_f64_row_matmul_dispatch")
            || rayon::current_num_threads() <= 1
        {
            if rayon::current_num_threads() <= 1 {
                eprintln!("skipping uma_so2_f64_row_matmul_dispatch: rayon has one worker");
            }
            return;
        }
        // Shapes that pass the guard and exercise the rows8 block + remainder rows
        // and the 32-col / 8-col / (none here) scalar-tail column paths.
        let rows = 20usize; // 2 full rows8 blocks + 4 remainder rows
        let k = 128usize;
        let n = 136usize; // 4*32 + 8 -> 32-col path then 8-col path
        let a: Vec<f64> = deterministic_f32(rows * k, 0.13)
            .into_iter()
            .map(|x| x as f64)
            .collect();
        let b: Vec<f64> = deterministic_f32(k * n, 0.07)
            .into_iter()
            .map(|x| x as f64)
            .collect();
        for &accumulate in &[false, true] {
            let init: Vec<f64> = deterministic_f32(rows * n, 0.05)
                .into_iter()
                .map(|x| x as f64)
                .collect();
            let mut expected = init.clone();
            for row in 0..rows {
                for col in 0..n {
                    let mut acc = if accumulate { init[row * n + col] } else { 0.0 };
                    for kk in 0..k {
                        acc += a[row * k + kk] * b[kk * n + col];
                    }
                    expected[row * n + col] = acc;
                }
            }
            let mut actual = init.clone();
            let ran = uma_so2_f64_try_parallel_row_matmul(
                rows,
                k,
                n,
                &a,
                &b,
                &mut actual,
                accumulate,
                false,
            );
            assert!(
                ran,
                "f64 row kernel should run when avx512f + rayon available"
            );
            for (i, (x, y)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-9 * (1.0 + y.abs()),
                    "f64 row matmul mismatch (accumulate={accumulate}) at {i}: {x} vs {y}"
                );
            }
        }
    }

    // Build a deterministic f64 vector (mirrors the f32 generator widened to f64)
    // so the f64 AVX-512 kernel tests share the module's reproducible inputs.
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    fn deterministic_f64_vec(len: usize, scale: f32) -> Vec<f64> {
        deterministic_f32(len, scale)
            .into_iter()
            .map(|x| x as f64)
            .collect()
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    fn f64_rows_matmul_scalar_reference(
        a: &[f64],
        b: &[f64],
        rows: usize,
        k: usize,
        n: usize,
        init: &[f64],
        accumulate: bool,
    ) -> Vec<f64> {
        let mut out = init.to_vec();
        for row in 0..rows {
            for col in 0..n {
                let mut acc = if accumulate { init[row * n + col] } else { 0.0 };
                for kk in 0..k {
                    acc += a[row * k + kk] * b[kk * n + col];
                }
                out[row * n + col] = acc;
            }
        }
        out
    }

    // Exercise the f64 single-row AVX-512 kernel across every column path: the
    // 32-wide block, the 8-wide block, and the scalar column tail. The dispatch
    // guard (`uma_so2_f64_row_kernel_should_run`) requires `n % 8 == 0`, so the
    // tail path is unreachable through the wrapper and would otherwise ship
    // unverified. Calling the kernel directly here covers it on AVX-512 hosts.
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_so2_f64_matmul_row_avx512_column_paths_match_reference_when_available() {
        if !avx512f_available_or_skip("uma_so2_f64_matmul_row_avx512_column_paths") {
            return;
        }
        let k = 130usize; // not a multiple of 8, to stress the inner k loop
                          // 7 -> tail only; 8/24 -> 8-col path; 40/72 -> 32-col + 8-col; 43 -> 32 + 8 + tail.
        for &n in &[7usize, 8, 24, 40, 43, 72] {
            let a = deterministic_f64_vec(k, 0.13);
            let b = deterministic_f64_vec(k * n, 0.07);
            for &accumulate in &[false, true] {
                let init = deterministic_f64_vec(n, 0.05);
                let expected = f64_rows_matmul_scalar_reference(&a, &b, 1, k, n, &init, accumulate);
                let mut actual = init.clone();
                unsafe {
                    uma_so2_f64_matmul_row_avx512(&a, &b, k, n, &mut actual, accumulate);
                }
                for (i, (x, y)) in actual.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        (x - y).abs() <= 1e-9 * (1.0 + y.abs()),
                        "f64 row avx512 mismatch (n={n}, accumulate={accumulate}) at {i}: {x} vs {y}"
                    );
                }
            }
        }
    }

    // Exercise the f64 8-row AVX-512 block kernel across its 8-wide column block
    // and scalar column tail (again unreachable through the `n % 8 == 0` dispatch
    // guard), for both accumulate and overwrite modes.
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_so2_f64_matmul_rows8_avx512_column_paths_match_reference_when_available() {
        if !avx512f_available_or_skip("uma_so2_f64_matmul_rows8_avx512_column_paths") {
            return;
        }
        let rows = 8usize;
        let k = 130usize;
        // 7 -> tail only; 8/24/56 -> 8-col path; 43 -> 8-col blocks + scalar tail.
        for &n in &[7usize, 8, 24, 43, 56] {
            let a = deterministic_f64_vec(rows * k, 0.11);
            let b = deterministic_f64_vec(k * n, 0.09);
            for &accumulate in &[false, true] {
                let init = deterministic_f64_vec(rows * n, 0.04);
                let expected =
                    f64_rows_matmul_scalar_reference(&a, &b, rows, k, n, &init, accumulate);
                let mut actual = init.clone();
                unsafe {
                    uma_so2_f64_matmul_rows8_avx512(&a, &b, k, n, &mut actual, accumulate);
                }
                for (i, (x, y)) in actual.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        (x - y).abs() <= 1e-9 * (1.0 + y.abs()),
                        "f64 rows8 avx512 mismatch (n={n}, accumulate={accumulate}) at {i}: {x} vs {y}"
                    );
                }
            }
        }
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_so2_f32_rows4_avx2_overwrite_matches_reference_when_available() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }

        let rows = 4;
        let k = 40;
        let n = 48;
        let a = deterministic_f32(rows * k, 0.31);
        let b = deterministic_f32(k * n, 0.23);
        let mut actual = deterministic_f32(rows * n, 0.07);
        let mut expected = Vec::with_capacity(rows * n);
        for row in 0..rows {
            expected.extend(row_matmul_reference(
                &a[row * k..row * k + k],
                &b,
                k,
                n,
                &actual[row * n..row * n + n],
                false,
            ));
        }

        unsafe {
            uma_so2_f32_matmul_rows4_avx2(&a, &b, k, n, &mut actual, false);
        }
        assert_close_slices(
            "UMA SO2 AVX2 rows4 overwrite matmul",
            &actual,
            &expected,
            2.5e-6,
        );
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_so2_f32_row_matmul_avx512_matches_reference_when_available() {
        if !avx512f_available_or_skip("uma_so2_f32_row_matmul_avx512") {
            return;
        }

        let k = 33;
        let n = 47;
        let a_row = deterministic_f32(k, 0.3);
        let b = deterministic_f32(k * n, 0.2);
        let initial = deterministic_f32(n, 0.05);
        let expected = row_matmul_reference(&a_row, &b, k, n, &initial, true);

        let mut actual = initial.clone();
        unsafe {
            uma_so2_f32_matmul_row_avx512(&a_row, &b, k, n, &mut actual, true);
        }
        assert_close_slices("UMA SO2 AVX512 row matmul", &actual, &expected, 2.5e-6);
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_so2_f32_rows4_avx512_matches_reference_when_available() {
        if !avx512f_available_or_skip("uma_so2_f32_rows4_avx512") {
            return;
        }

        let rows = 4;
        let k = 33;
        let n = 55;
        let a = deterministic_f32(rows * k, 0.3);
        let b = deterministic_f32(k * n, 0.2);
        let mut actual = deterministic_f32(rows * n, 0.05);
        let mut expected = Vec::with_capacity(rows * n);
        for row in 0..rows {
            expected.extend(row_matmul_reference(
                &a[row * k..row * k + k],
                &b,
                k,
                n,
                &actual[row * n..row * n + n],
                true,
            ));
        }

        unsafe {
            uma_so2_f32_matmul_rows4_avx512(&a, &b, k, n, &mut actual, true);
        }
        assert_close_slices("UMA SO2 AVX512 rows4 matmul", &actual, &expected, 2.5e-6);
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_so2_f32_rows4_avx512_overwrite_matches_reference_when_available() {
        if !avx512f_available_or_skip("uma_so2_f32_rows4_avx512_overwrite") {
            return;
        }

        let rows = 4;
        let k = 40;
        let n = 64;
        let a = deterministic_f32(rows * k, 0.31);
        let b = deterministic_f32(k * n, 0.23);
        let mut actual = deterministic_f32(rows * n, 0.07);
        let mut expected = Vec::with_capacity(rows * n);
        for row in 0..rows {
            expected.extend(row_matmul_reference(
                &a[row * k..row * k + k],
                &b,
                k,
                n,
                &actual[row * n..row * n + n],
                false,
            ));
        }

        unsafe {
            uma_so2_f32_matmul_rows4_avx512(&a, &b, k, n, &mut actual, false);
        }
        assert_close_slices(
            "UMA SO2 AVX512 rows4 overwrite matmul",
            &actual,
            &expected,
            2.5e-6,
        );
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_so2_f32_rows8x16_avx512_matches_reference_when_available() {
        if !avx512f_available_or_skip("uma_so2_f32_rows8x16_avx512") {
            return;
        }

        let rows = 8;
        let k = 41;
        let n = 67;
        let a = deterministic_f32(rows * k, 0.29);
        let b = deterministic_f32(k * n, 0.19);
        let mut actual = deterministic_f32(rows * n, 0.04);
        let mut expected = Vec::with_capacity(rows * n);
        for row in 0..rows {
            expected.extend(row_matmul_reference(
                &a[row * k..row * k + k],
                &b,
                k,
                n,
                &actual[row * n..row * n + n],
                true,
            ));
        }

        unsafe {
            uma_so2_f32_matmul_rows8x16_avx512(&a, &b, k, n, &mut actual, true);
        }
        assert_close_slices("UMA SO2 AVX512 rows8x16 matmul", &actual, &expected, 3.0e-6);
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_so2_f32_rows8x16_avx512_overwrite_matches_reference_when_available() {
        if !avx512f_available_or_skip("uma_so2_f32_rows8x16_avx512_overwrite") {
            return;
        }

        let rows = 8;
        let k = 40;
        let n = 80;
        let a = deterministic_f32(rows * k, 0.33);
        let b = deterministic_f32(k * n, 0.21);
        let mut actual = deterministic_f32(rows * n, 0.06);
        let mut expected = Vec::with_capacity(rows * n);
        for row in 0..rows {
            expected.extend(row_matmul_reference(
                &a[row * k..row * k + k],
                &b,
                k,
                n,
                &actual[row * n..row * n + n],
                false,
            ));
        }

        unsafe {
            uma_so2_f32_matmul_rows8x16_avx512(&a, &b, k, n, &mut actual, false);
        }
        assert_close_slices(
            "UMA SO2 AVX512 rows8x16 overwrite matmul",
            &actual,
            &expected,
            3.0e-6,
        );
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    struct M0GateVjpFixture {
        nodes: usize,
        channels: usize,
        activated: Vec<f32>,
        conv1_out: Vec<f32>,
        gates: Vec<f32>,
        derivatives: Vec<f32>,
        initial: Vec<f32>,
        expected: Vec<f32>,
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    fn m0_gate_vjp_fixture() -> M0GateVjpFixture {
        let nodes = 19;
        let channels = 128;
        let m0_out_dim = 5 * channels;
        let activated = deterministic_f32(nodes * 3 * channels, 0.17);
        let conv1_out = deterministic_f32(nodes * 9 * channels, 0.11);
        let gates = deterministic_f32(nodes * 2 * channels, 0.07)
            .into_iter()
            .map(|x| 0.5 + 0.25 * x.tanh())
            .collect::<Vec<_>>();
        let derivatives = deterministic_f32(nodes * channels, 0.23);
        let initial = deterministic_f32(nodes * m0_out_dim, 0.03);
        let mut expected = initial.clone();
        for node in 0..nodes {
            let act_base = node * 3 * channels;
            let conv1_base = node * 9 * channels;
            let gate_base = node * 2 * channels;
            let deriv_base = node * channels;
            let full_base = node * m0_out_dim;
            for ch in 0..channels {
                let grad_scalar = activated[act_base + ch];
                expected[full_base + 2 * channels + ch] +=
                    grad_scalar * derivatives[deriv_base + ch];

                let grad_l1 = activated[act_base + channels + ch];
                expected[full_base + 3 * channels + ch] += grad_l1 * gates[gate_base + ch];
                expected[full_base + ch] += grad_l1 * conv1_out[conv1_base + channels + ch];

                let grad_l2 = activated[act_base + 2 * channels + ch];
                expected[full_base + 4 * channels + ch] +=
                    grad_l2 * gates[gate_base + channels + ch];
                expected[full_base + channels + ch] +=
                    grad_l2 * conv1_out[conv1_base + 2 * channels + ch];
            }
        }
        M0GateVjpFixture {
            nodes,
            channels,
            activated,
            conv1_out,
            gates,
            derivatives,
            initial,
            expected,
        }
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_f32_l2_m2_m0_gate_vjp_dispatch_matches_reference_when_available() {
        if !uma_so2_f32_avx2_fma_enabled() {
            return;
        }

        let fixture = m0_gate_vjp_fixture();
        let mut actual = fixture.initial;
        assert!(uma_f32_try_l2_m2_m0_gate_vjp(
            fixture.nodes,
            fixture.channels,
            &fixture.activated,
            &fixture.conv1_out,
            &fixture.gates,
            &fixture.derivatives,
            &mut actual,
        ));
        assert_close_slices(
            "UMA f32 l2/m2 m0 gate VJP dispatch",
            &actual,
            &fixture.expected,
            2.5e-6,
        );
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_f32_l2_m2_m0_gate_vjp_avx2_matches_reference_when_available() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }

        let fixture = m0_gate_vjp_fixture();
        let mut actual = fixture.initial;
        unsafe {
            uma_f32_l2_m2_m0_gate_vjp_avx2(
                fixture.nodes,
                fixture.channels,
                &fixture.activated,
                &fixture.conv1_out,
                &fixture.gates,
                &fixture.derivatives,
                &mut actual,
            );
        }
        assert_close_slices(
            "UMA f32 l2/m2 m0 gate VJP AVX2",
            &actual,
            &fixture.expected,
            2.5e-6,
        );
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_f32_l2_m2_m0_gate_vjp_avx512_matches_reference_when_available() {
        if !avx512f_available_or_skip("uma_f32_l2_m2_m0_gate_vjp_avx512") {
            return;
        }

        let fixture = m0_gate_vjp_fixture();
        let mut actual = fixture.initial;
        unsafe {
            uma_f32_l2_m2_m0_gate_vjp_avx512(
                fixture.nodes,
                fixture.channels,
                &fixture.activated,
                &fixture.conv1_out,
                &fixture.gates,
                &fixture.derivatives,
                &mut actual,
            );
        }
        assert_close_slices(
            "UMA f32 l2/m2 m0 gate VJP AVX512",
            &actual,
            &fixture.expected,
            2.5e-6,
        );
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    struct M0GateScatterFwdFixture {
        nodes: usize,
        channels: usize,
        dense: Vec<f32>,
        exp_gate: Vec<f32>,
        exp_deriv: Vec<f32>,
        exp_conv1: Vec<f32>,
        exp_conv2: Vec<f32>,
        init_conv1: Vec<f32>,
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    fn m0_gate_scatter_fwd_fixture() -> M0GateScatterFwdFixture {
        let nodes = 17;
        let channels = 128;
        let m0_out_dim = 5 * channels;
        // Spread inputs over a wide range to exercise the sigmoid/exp clamp.
        let dense = deterministic_f32(nodes * m0_out_dim, 0.37)
            .into_iter()
            .map(|x| 6.0 * x)
            .collect::<Vec<_>>();
        let init_conv1 = deterministic_f32(nodes * 9 * channels, 0.021);
        let mut exp_gate = vec![0.0_f32; nodes * 2 * channels];
        let mut exp_deriv = vec![0.0_f32; nodes * channels];
        let mut exp_conv1 = init_conv1.clone();
        let mut exp_conv2 = vec![0.0_f32; nodes * 3 * channels];
        for node in 0..nodes {
            let db = node * m0_out_dim;
            let gb = node * 2 * channels;
            let dvb = node * channels;
            let c1b = node * 9 * channels;
            let c2b = node * 3 * channels;
            for ch in 0..channels {
                let g1 = scalar_sigmoid_f32(dense[db + ch]);
                let g2 = scalar_sigmoid_f32(dense[db + channels + ch]);
                let scalar = dense[db + 2 * channels + ch];
                let l1 = dense[db + 3 * channels + ch];
                let l2 = dense[db + 4 * channels + ch];
                let sig = scalar_sigmoid_f32(scalar);
                exp_gate[gb + ch] = g1;
                exp_gate[gb + channels + ch] = g2;
                exp_deriv[dvb + ch] = sig * (1.0 + scalar * (1.0 - sig));
                exp_conv1[c1b + ch] = scalar;
                exp_conv1[c1b + channels + ch] = l1;
                exp_conv1[c1b + 2 * channels + ch] = l2;
                exp_conv2[c2b + ch] = scalar * sig;
                exp_conv2[c2b + channels + ch] = l1 * g1;
                exp_conv2[c2b + 2 * channels + ch] = l2 * g2;
            }
        }
        M0GateScatterFwdFixture {
            nodes,
            channels,
            dense,
            exp_gate,
            exp_deriv,
            exp_conv1,
            exp_conv2,
            init_conv1,
        }
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_f32_l2_m2_m0_gate_scatter_fwd_avx2_matches_reference_when_available() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        let f = m0_gate_scatter_fwd_fixture();
        let mut gate = vec![0.0_f32; f.nodes * 2 * f.channels];
        let mut deriv = vec![0.0_f32; f.nodes * f.channels];
        let mut conv1 = f.init_conv1.clone();
        let mut conv2 = vec![0.0_f32; f.nodes * 3 * f.channels];
        unsafe {
            uma_f32_l2_m2_m0_gate_scatter_forward_avx2(
                f.nodes, f.channels, &f.dense, &mut gate, &mut deriv, &mut conv1, &mut conv2,
            );
        }
        assert_close_slices("m0 gate-scatter fwd AVX2 gate", &gate, &f.exp_gate, 2.0e-6);
        assert_close_slices(
            "m0 gate-scatter fwd AVX2 deriv",
            &deriv,
            &f.exp_deriv,
            2.0e-6,
        );
        assert_close_slices(
            "m0 gate-scatter fwd AVX2 conv1",
            &conv1,
            &f.exp_conv1,
            2.0e-6,
        );
        assert_close_slices(
            "m0 gate-scatter fwd AVX2 conv2",
            &conv2,
            &f.exp_conv2,
            2.0e-6,
        );
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_f32_l2_m2_m0_gate_scatter_fwd_avx512_matches_reference_when_available() {
        if !avx512f_available_or_skip("uma_f32_l2_m2_m0_gate_scatter_fwd_avx512") {
            return;
        }
        let f = m0_gate_scatter_fwd_fixture();
        let mut gate = vec![0.0_f32; f.nodes * 2 * f.channels];
        let mut deriv = vec![0.0_f32; f.nodes * f.channels];
        let mut conv1 = f.init_conv1.clone();
        let mut conv2 = vec![0.0_f32; f.nodes * 3 * f.channels];
        unsafe {
            uma_f32_l2_m2_m0_gate_scatter_forward_avx512(
                f.nodes, f.channels, &f.dense, &mut gate, &mut deriv, &mut conv1, &mut conv2,
            );
        }
        assert_close_slices(
            "m0 gate-scatter fwd AVX512 gate",
            &gate,
            &f.exp_gate,
            2.0e-6,
        );
        assert_close_slices(
            "m0 gate-scatter fwd AVX512 deriv",
            &deriv,
            &f.exp_deriv,
            2.0e-6,
        );
        assert_close_slices(
            "m0 gate-scatter fwd AVX512 conv1",
            &conv1,
            &f.exp_conv1,
            2.0e-6,
        );
        assert_close_slices(
            "m0 gate-scatter fwd AVX512 conv2",
            &conv2,
            &f.exp_conv2,
            2.0e-6,
        );
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    struct Order1GateVjpFixture {
        nodes: usize,
        channels: usize,
        activated: Vec<f32>,
        conv1_out: Vec<f32>,
        gates: Vec<f32>,
        initial_m0: Vec<f32>,
        expected_m0: Vec<f32>,
        expected_block: Vec<f32>,
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    fn order1_gate_vjp_fixture() -> Order1GateVjpFixture {
        let nodes = 19;
        let channels = 128;
        let block_stride = 4 * channels;
        let out_node_stride = 9 * channels;
        let gate_node_stride = 2 * channels;
        let m0_out_dim = 5 * channels;
        let activated = deterministic_f32(nodes * block_stride, 0.13);
        let conv1_out = deterministic_f32(nodes * out_node_stride, 0.09);
        let gates = deterministic_f32(nodes * gate_node_stride, 0.05)
            .into_iter()
            .map(|x| 0.5 + 0.25 * x.tanh())
            .collect::<Vec<_>>();
        let initial_m0 = deterministic_f32(nodes * m0_out_dim, 0.025);
        let mut expected_m0 = initial_m0.clone();
        let mut expected_block = vec![0.0_f32; nodes * block_stride];
        for node in 0..nodes {
            let block_base = node * block_stride;
            let conv1_base = node * out_node_stride + 3 * channels;
            let gate_base = node * gate_node_stride;
            let full_base = node * m0_out_dim;
            for ch in 0..channels {
                let gate1 = gates[gate_base + ch];
                let gate2 = gates[gate_base + channels + ch];
                let grad3 = activated[block_base + ch];
                let grad7 = activated[block_base + channels + ch];
                let grad1 = activated[block_base + 2 * channels + ch];
                let grad5 = activated[block_base + 3 * channels + ch];
                expected_block[block_base + ch] = grad3 * gate1;
                expected_block[block_base + channels + ch] = grad7 * gate2;
                expected_block[block_base + 2 * channels + ch] = grad1 * gate1;
                expected_block[block_base + 3 * channels + ch] = grad5 * gate2;
                expected_m0[full_base + ch] += grad3 * conv1_out[conv1_base + ch]
                    + grad1 * conv1_out[conv1_base + 2 * channels + ch];
                expected_m0[full_base + channels + ch] += grad7
                    * conv1_out[conv1_base + channels + ch]
                    + grad5 * conv1_out[conv1_base + 3 * channels + ch];
            }
        }
        Order1GateVjpFixture {
            nodes,
            channels,
            activated,
            conv1_out,
            gates,
            initial_m0,
            expected_m0,
            expected_block,
        }
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_f32_l2_m2_order1_gate_vjp_dispatch_matches_reference_when_available() {
        if !uma_so2_f32_avx2_fma_enabled() {
            return;
        }

        let fixture = order1_gate_vjp_fixture();
        let mut actual_m0 = fixture.initial_m0;
        let mut actual_block = deterministic_f32(fixture.nodes * 4 * fixture.channels, 0.01);
        assert!(uma_f32_try_l2_m2_order1_gate_vjp(
            fixture.nodes,
            fixture.channels,
            &fixture.activated,
            &fixture.conv1_out,
            &fixture.gates,
            &mut actual_m0,
            &mut actual_block,
        ));
        assert_close_slices(
            "UMA f32 l2/m2 order1 gate VJP dispatch m0",
            &actual_m0,
            &fixture.expected_m0,
            2.5e-6,
        );
        assert_close_slices(
            "UMA f32 l2/m2 order1 gate VJP dispatch block",
            &actual_block,
            &fixture.expected_block,
            2.5e-6,
        );
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_f32_l2_m2_order1_gate_vjp_avx2_matches_reference_when_available() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }

        let fixture = order1_gate_vjp_fixture();
        let mut actual_m0 = fixture.initial_m0;
        let mut actual_block = deterministic_f32(fixture.nodes * 4 * fixture.channels, 0.01);
        unsafe {
            uma_f32_l2_m2_order1_gate_vjp_avx2(
                fixture.nodes,
                fixture.channels,
                &fixture.activated,
                &fixture.conv1_out,
                &fixture.gates,
                &mut actual_m0,
                &mut actual_block,
            );
        }
        assert_close_slices(
            "UMA f32 l2/m2 order1 gate VJP AVX2 m0",
            &actual_m0,
            &fixture.expected_m0,
            2.5e-6,
        );
        assert_close_slices(
            "UMA f32 l2/m2 order1 gate VJP AVX2 block",
            &actual_block,
            &fixture.expected_block,
            2.5e-6,
        );
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_f32_l2_m2_order1_gate_vjp_avx512_matches_reference_when_available() {
        if !avx512f_available_or_skip("uma_f32_l2_m2_order1_gate_vjp_avx512") {
            return;
        }

        let fixture = order1_gate_vjp_fixture();
        let mut actual_m0 = fixture.initial_m0;
        let mut actual_block = deterministic_f32(fixture.nodes * 4 * fixture.channels, 0.01);
        unsafe {
            uma_f32_l2_m2_order1_gate_vjp_avx512(
                fixture.nodes,
                fixture.channels,
                &fixture.activated,
                &fixture.conv1_out,
                &fixture.gates,
                &mut actual_m0,
                &mut actual_block,
            );
        }
        assert_close_slices(
            "UMA f32 l2/m2 order1 gate VJP AVX512 m0",
            &actual_m0,
            &fixture.expected_m0,
            2.5e-6,
        );
        assert_close_slices(
            "UMA f32 l2/m2 order1 gate VJP AVX512 block",
            &actual_block,
            &fixture.expected_block,
            2.5e-6,
        );
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    struct Order2GateVjpFixture {
        nodes: usize,
        channels: usize,
        activated: Vec<f32>,
        conv1_out: Vec<f32>,
        gates: Vec<f32>,
        expected_gate: Vec<f32>,
        expected_block: Vec<f32>,
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    fn order2_gate_vjp_fixture() -> Order2GateVjpFixture {
        let nodes = 19;
        let channels = 128;
        let block_stride = 2 * channels;
        let out_node_stride = 9 * channels;
        let gate_node_stride = 2 * channels;
        let activated = deterministic_f32(nodes * block_stride, 0.16);
        let conv1_out = deterministic_f32(nodes * out_node_stride, 0.08);
        let gates = deterministic_f32(nodes * gate_node_stride, 0.06)
            .into_iter()
            .map(|x| 0.5 + 0.25 * x.tanh())
            .collect::<Vec<_>>();
        let mut expected_gate = vec![0.0_f32; nodes * channels];
        let mut expected_block = vec![0.0_f32; nodes * block_stride];
        for node in 0..nodes {
            let block_base = node * block_stride;
            let conv1_base = node * out_node_stride + 7 * channels;
            let gate_base = node * gate_node_stride;
            let gate_grad_base = node * channels;
            for ch in 0..channels {
                let gate2 = gates[gate_base + channels + ch];
                let grad8 = activated[block_base + ch];
                let grad4 = activated[block_base + channels + ch];
                expected_block[block_base + ch] = grad8 * gate2;
                expected_block[block_base + channels + ch] = grad4 * gate2;
                expected_gate[gate_grad_base + ch] = grad8 * conv1_out[conv1_base + ch]
                    + grad4 * conv1_out[conv1_base + channels + ch];
            }
        }
        Order2GateVjpFixture {
            nodes,
            channels,
            activated,
            conv1_out,
            gates,
            expected_gate,
            expected_block,
        }
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_f32_l2_m2_order2_gate_vjp_dispatch_matches_reference_when_available() {
        if !uma_so2_f32_avx2_fma_enabled() {
            return;
        }

        let fixture = order2_gate_vjp_fixture();
        let mut actual_gate = deterministic_f32(fixture.nodes * fixture.channels, 0.01);
        let mut actual_block = deterministic_f32(fixture.nodes * 2 * fixture.channels, 0.012);
        assert!(uma_f32_try_l2_m2_order2_gate_vjp(
            fixture.nodes,
            fixture.channels,
            &fixture.activated,
            &fixture.conv1_out,
            &fixture.gates,
            &mut actual_gate,
            &mut actual_block,
        ));
        assert_close_slices(
            "UMA f32 l2/m2 order2 gate VJP dispatch gate",
            &actual_gate,
            &fixture.expected_gate,
            2.5e-6,
        );
        assert_close_slices(
            "UMA f32 l2/m2 order2 gate VJP dispatch block",
            &actual_block,
            &fixture.expected_block,
            2.5e-6,
        );
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_f32_l2_m2_order2_gate_vjp_avx2_matches_reference_when_available() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }

        let fixture = order2_gate_vjp_fixture();
        let mut actual_gate = deterministic_f32(fixture.nodes * fixture.channels, 0.01);
        let mut actual_block = deterministic_f32(fixture.nodes * 2 * fixture.channels, 0.012);
        unsafe {
            uma_f32_l2_m2_order2_gate_vjp_avx2(
                fixture.nodes,
                fixture.channels,
                &fixture.activated,
                &fixture.conv1_out,
                &fixture.gates,
                &mut actual_gate,
                &mut actual_block,
            );
        }
        assert_close_slices(
            "UMA f32 l2/m2 order2 gate VJP AVX2 gate",
            &actual_gate,
            &fixture.expected_gate,
            2.5e-6,
        );
        assert_close_slices(
            "UMA f32 l2/m2 order2 gate VJP AVX2 block",
            &actual_block,
            &fixture.expected_block,
            2.5e-6,
        );
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_f32_l2_m2_order2_gate_vjp_avx512_matches_reference_when_available() {
        if !avx512f_available_or_skip("uma_f32_l2_m2_order2_gate_vjp_avx512") {
            return;
        }

        let fixture = order2_gate_vjp_fixture();
        let mut actual_gate = deterministic_f32(fixture.nodes * fixture.channels, 0.01);
        let mut actual_block = deterministic_f32(fixture.nodes * 2 * fixture.channels, 0.012);
        unsafe {
            uma_f32_l2_m2_order2_gate_vjp_avx512(
                fixture.nodes,
                fixture.channels,
                &fixture.activated,
                &fixture.conv1_out,
                &fixture.gates,
                &mut actual_gate,
                &mut actual_block,
            );
        }
        assert_close_slices(
            "UMA f32 l2/m2 order2 gate VJP AVX512 gate",
            &actual_gate,
            &fixture.expected_gate,
            2.5e-6,
        );
        assert_close_slices(
            "UMA f32 l2/m2 order2 gate VJP AVX512 block",
            &actual_block,
            &fixture.expected_block,
            2.5e-6,
        );
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_f32_l2_m2_order1_gate_conv1_stack_matches_materialized_when_available() {
        if !uma_so2_f32_avx2_fma_enabled() {
            return;
        }

        let fixture = order1_gate_vjp_fixture();
        let conv1_in_channels = 2 * fixture.channels;
        let k = 4 * fixture.channels;
        let n = 4 * conv1_in_channels;
        let fused_weight = deterministic_f32(k * n, 0.045);
        let mut expected_grad = Vec::with_capacity(fixture.nodes * n);
        let zero = vec![0.0_f32; n];
        for row in 0..fixture.nodes {
            expected_grad.extend(row_matmul_reference(
                &fixture.expected_block[row * k..row * k + k],
                &fused_weight,
                k,
                n,
                &zero,
                false,
            ));
        }

        let pool = match rayon::ThreadPoolBuilder::new().num_threads(4).build() {
            Ok(pool) => pool,
            Err(_) => return,
        };
        if !pool.install(|| uma_so2_f32_row_kernel_should_run(fixture.nodes, k, n, true)) {
            return;
        }

        let mut actual_m0 = fixture.initial_m0;
        let mut actual_grad = deterministic_f32(fixture.nodes * n, 0.02);
        let ran = pool.install(|| {
            uma_f32_try_l2_m2_order1_gate_conv1_vjp_stack(
                fixture.nodes,
                conv1_in_channels,
                fixture.channels,
                &fixture.activated,
                &fixture.conv1_out,
                &fixture.gates,
                &fused_weight,
                &mut actual_m0,
                &mut actual_grad,
            )
        });
        assert!(ran);
        assert_close_slices(
            "UMA f32 l2/m2 order1 gate+conv1 stack m0",
            &actual_m0,
            &fixture.expected_m0,
            2.5e-6,
        );
        assert_close_slices(
            "UMA f32 l2/m2 order1 gate+conv1 stack grad",
            &actual_grad,
            &expected_grad,
            8.0e-5,
        );
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_f32_l2_m2_order2_gate_conv1_stack_matches_materialized_when_available() {
        if !uma_so2_f32_avx2_fma_enabled() {
            return;
        }

        let fixture = order2_gate_vjp_fixture();
        let conv1_in_channels = 2 * fixture.channels;
        let k = 2 * fixture.channels;
        let n = 2 * conv1_in_channels;
        let fused_weight = deterministic_f32(k * n, 0.052);
        let mut expected_grad = Vec::with_capacity(fixture.nodes * n);
        let zero = vec![0.0_f32; n];
        for row in 0..fixture.nodes {
            expected_grad.extend(row_matmul_reference(
                &fixture.expected_block[row * k..row * k + k],
                &fused_weight,
                k,
                n,
                &zero,
                false,
            ));
        }

        let pool = match rayon::ThreadPoolBuilder::new().num_threads(4).build() {
            Ok(pool) => pool,
            Err(_) => return,
        };
        if !pool.install(|| uma_so2_f32_row_kernel_should_run(fixture.nodes, k, n, true)) {
            return;
        }

        let mut actual_gate = deterministic_f32(fixture.nodes * fixture.channels, 0.01);
        let mut actual_grad = deterministic_f32(fixture.nodes * n, 0.02);
        let ran = pool.install(|| {
            uma_f32_try_l2_m2_order2_gate_conv1_vjp_stack(
                fixture.nodes,
                conv1_in_channels,
                fixture.channels,
                &fixture.activated,
                &fixture.conv1_out,
                &fixture.gates,
                &fused_weight,
                &mut actual_gate,
                &mut actual_grad,
            )
        });
        assert!(ran);
        assert_close_slices(
            "UMA f32 l2/m2 order2 gate+conv1 stack gate",
            &actual_gate,
            &fixture.expected_gate,
            2.5e-6,
        );
        assert_close_slices(
            "UMA f32 l2/m2 order2 gate+conv1 stack grad",
            &actual_grad,
            &expected_grad,
            6.0e-5,
        );
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_so2_f32_parallel_row_dispatcher_matches_reference_with_tail_rows() {
        if !uma_so2_f32_avx2_fma_enabled() {
            return;
        }

        let rows = 18;
        let k = 128;
        let n = 136;
        let a = deterministic_f32(rows * k, 0.17);
        let b = deterministic_f32(k * n, 0.11);
        let initial = deterministic_f32(rows * n, 0.03);
        let mut expected = Vec::with_capacity(rows * n);
        for row in 0..rows {
            expected.extend(row_matmul_reference(
                &a[row * k..row * k + k],
                &b,
                k,
                n,
                &initial[row * n..row * n + n],
                true,
            ));
        }

        let pool = match rayon::ThreadPoolBuilder::new().num_threads(4).build() {
            Ok(pool) => pool,
            Err(_) => return,
        };
        let mut actual = initial;
        let ran = pool.install(|| {
            uma_so2_f32_try_parallel_row_matmul(rows, k, n, &a, &b, &mut actual, true, false, None)
        });
        assert!(ran);
        assert_close_slices(
            "UMA SO2 parallel row dispatcher tail rows",
            &actual,
            &expected,
            8.0e-5,
        );
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_so2_f32_precomputed_vjp_write_overwrites_initial_buffer() {
        let rows = 18;
        let in_dim = 64;
        let out_dim = 64;
        let fused_in_dim = 2 * in_dim;
        let fused_out_dim = 2 * out_dim;
        let grad_out = deterministic_f32(rows * fused_out_dim, 0.17);
        let fused_weight = deterministic_f32(fused_out_dim * fused_in_dim, 0.11);
        let mut actual = deterministic_f32(rows * fused_in_dim, 0.03);
        let mut expected = Vec::with_capacity(rows * fused_in_dim);
        for row in 0..rows {
            expected.extend(row_matmul_reference(
                &grad_out[row * fused_out_dim..row * fused_out_dim + fused_out_dim],
                &fused_weight,
                fused_out_dim,
                fused_in_dim,
                &actual[row * fused_in_dim..row * fused_in_dim + fused_in_dim],
                false,
            ));
        }

        so2_mconv_vjp_input_write_precomputed_fused_f32(
            rows,
            in_dim,
            out_dim,
            &fused_weight,
            &grad_out,
            &mut actual,
        );
        assert_close_slices("UMA SO2 precomputed VJP write", &actual, &expected, 2.5e-4);
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_so2_f32_lmajor_direct_vjp_dispatch_matches_reference_when_available() {
        if !uma_so2_f32_avx2_fma_enabled() || !uma_so2_f32_provider_threads_is_one() {
            return;
        }

        let rows = 18;
        let channels = 128;
        let source_positions = [3usize, 7, 1, 5];
        let lmajor_row_stride = 9 * channels;
        let k = source_positions.len() * channels;
        let n = 4 * channels;
        let grad_out_lmajor = deterministic_f32(rows * lmajor_row_stride, 0.13);
        let fused_weight = deterministic_f32(k * n, 0.09);
        let initial = deterministic_f32(rows * n, 0.03);
        let mut expected = Vec::with_capacity(rows * n);
        for row in 0..rows {
            let row_base = row * lmajor_row_stride;
            let mut gathered = Vec::with_capacity(k);
            for &source in &source_positions {
                let source_base = row_base + source * channels;
                gathered.extend_from_slice(&grad_out_lmajor[source_base..source_base + channels]);
            }
            expected.extend(row_matmul_reference(
                &gathered,
                &fused_weight,
                k,
                n,
                &initial[row * n..row * n + n],
                true,
            ));
        }

        let pool = match rayon::ThreadPoolBuilder::new().num_threads(4).build() {
            Ok(pool) => pool,
            Err(_) => return,
        };
        let mut actual = initial;
        let ran = pool.install(|| {
            so2_mconv_vjp_input_add_precomputed_fused_f32_lmajor_direct(
                rows,
                channels,
                &source_positions,
                lmajor_row_stride,
                n,
                &fused_weight,
                &grad_out_lmajor,
                &mut actual,
            )
        });
        assert!(ran);
        assert_close_slices(
            "UMA SO2 l-major direct VJP dispatcher",
            &actual,
            &expected,
            2.5e-4,
        );
    }

    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    #[test]
    fn uma_f32_parallel_row_dispatcher_with_bias_matches_reference() {
        if !uma_so2_f32_avx2_fma_enabled() {
            return;
        }

        let rows = 18;
        let k = 128;
        let n = 136;
        let a = deterministic_f32(rows * k, 0.13);
        let b = deterministic_f32(k * n, 0.09);
        let bias = deterministic_f32(n, 0.07);
        let initial = deterministic_f32(rows * n, 0.03);
        let mut expected = Vec::with_capacity(rows * n);
        for row in 0..rows {
            let mut expected_row = row_matmul_reference(
                &a[row * k..row * k + k],
                &b,
                k,
                n,
                &initial[row * n..row * n + n],
                false,
            );
            for (value, bias_value) in expected_row.iter_mut().zip(bias.iter()) {
                *value += *bias_value;
            }
            expected.extend(expected_row);
        }

        let pool = match rayon::ThreadPoolBuilder::new().num_threads(4).build() {
            Ok(pool) => pool,
            Err(_) => return,
        };
        let mut actual = initial;
        let ran = pool.install(|| {
            uma_so2_f32_try_parallel_row_matmul(
                rows,
                k,
                n,
                &a,
                &b,
                &mut actual,
                false,
                false,
                Some(&bias),
            )
        });
        assert!(ran);
        assert_close_slices(
            "UMA f32 parallel row dispatcher with bias",
            &actual,
            &expected,
            8.0e-5,
        );
    }
}

#[cfg(all(
    test,
    any(
        all(feature = "accelerate-blas", target_os = "macos"),
        all(
            any(feature = "intel-mkl", feature = "openblas"),
            target_os = "linux",
            target_arch = "x86_64"
        )
    )
))]
mod exp_tests {
    use super::*;

    #[test]
    fn batched_exp_f64_matches_scalar_reference() {
        let input = [-20.0_f64, -7.5, -1.0, -0.125, 0.0, 0.125, 1.0, 4.25, 10.0];
        let mut actual = vec![0.0; input.len()];
        exp_f64(&input, &mut actual);
        for (&actual, &value) in actual.iter().zip(&input) {
            let expected = value.exp();
            let tolerance = 1.0e-13_f64.max(1.0e-14 * expected.abs());
            assert!(
                (actual - expected).abs() <= tolerance,
                "exp_f64({value}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn batched_exp_f32_matches_scalar_reference() {
        let input = [-12.0_f32, -4.5, -1.0, -0.125, 0.0, 0.125, 1.0, 3.5, 8.0];
        let mut actual = vec![0.0; input.len()];
        exp_f32(&input, &mut actual);
        for (&actual, &value) in actual.iter().zip(&input) {
            let expected = value.exp();
            let tolerance = 1.0e-5_f32.max(2.0e-6 * expected.abs());
            assert!(
                (actual - expected).abs() <= tolerance,
                "exp_f32({value}) = {actual}, expected {expected}"
            );
        }
    }
}
