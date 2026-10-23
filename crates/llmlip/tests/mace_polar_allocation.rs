#![cfg(all(feature = "apple-metal", target_os = "macos", target_arch = "aarch64"))]
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::error::Error;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use llmlip::{LlmlipModel, Prediction, PredictionOptions, Structure};

struct CountingAllocator;

static TRACK_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static REALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static REALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

// Safety: this test allocator forwards every operation to `System` and only
// updates atomics after the system allocator returns. It does not retain,
// inspect, or transform pointers or layouts.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if TRACK_ALLOCATIONS.load(Ordering::Relaxed) && !ptr.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            DEALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if TRACK_ALLOCATIONS.load(Ordering::Relaxed) && !new_ptr.is_null() {
            REALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            REALLOCATED_BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        new_ptr
    }
}

#[derive(Clone, Copy, Debug)]
struct AllocationSnapshot {
    allocations: usize,
    reallocations: usize,
    deallocations: usize,
    allocated_bytes: usize,
    reallocated_bytes: usize,
}

impl AllocationSnapshot {
    fn allocation_events(self) -> usize {
        self.allocations + self.reallocations
    }
}

fn start_allocation_count() {
    TRACK_ALLOCATIONS.store(false, Ordering::SeqCst);
    ALLOCATIONS.store(0, Ordering::SeqCst);
    REALLOCATIONS.store(0, Ordering::SeqCst);
    DEALLOCATIONS.store(0, Ordering::SeqCst);
    ALLOCATED_BYTES.store(0, Ordering::SeqCst);
    REALLOCATED_BYTES.store(0, Ordering::SeqCst);
    TRACK_ALLOCATIONS.store(true, Ordering::SeqCst);
}

fn stop_allocation_count() -> AllocationSnapshot {
    TRACK_ALLOCATIONS.store(false, Ordering::SeqCst);
    AllocationSnapshot {
        allocations: ALLOCATIONS.load(Ordering::SeqCst),
        reallocations: REALLOCATIONS.load(Ordering::SeqCst),
        deallocations: DEALLOCATIONS.load(Ordering::SeqCst),
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::SeqCst),
        reallocated_bytes: REALLOCATED_BYTES.load(Ordering::SeqCst),
    }
}

fn model_bundle_path(suffix: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("outputs")
        .join("reference-artifacts")
        .join("mace")
        .join(format!("polarmace_polar_1_{suffix}_state.e3rs"))
}

fn metal_runtime_gate_enabled() -> bool {
    std::env::var("LLMLIP_MACE_POLAR_METAL").ok().as_deref() == Some("1")
        || std::env::var("LLMLIP_MACE_METAL").ok().as_deref() == Some("1")
}

fn measure_prediction(
    predict: impl FnOnce() -> Result<Prediction, Box<dyn Error>>,
) -> Result<(Prediction, AllocationSnapshot), Box<dyn Error>> {
    start_allocation_count();
    let prediction = predict()?;
    let allocations = stop_allocation_count();
    Ok((prediction, allocations))
}

#[test]
#[ignore = "requires real PolarMACE runtime .e3rs bundles and Metal runtime gate"]
fn polarmace_prepared_predict_reports_stable_rust_heap_allocations() -> Result<(), Box<dyn Error>> {
    if !metal_runtime_gate_enabled() {
        eprintln!(
            "skipping PolarMACE allocation profile; LLMLIP_MACE_POLAR_METAL=1 or LLMLIP_MACE_METAL=1 is not set"
        );
        return Ok(());
    }

    let structure = Structure::new(
        vec!["O".to_owned(), "H".to_owned(), "H".to_owned()],
        vec![[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.239, 0.927, 0.0]],
    );
    let options = PredictionOptions {
        compute_energy: true,
        compute_forces: true,
        ..PredictionOptions::default()
    };

    for suffix in ["s", "m", "l"] {
        let bundle_path = model_bundle_path(suffix);
        if !bundle_path.is_dir() {
            eprintln!(
                "skipping PolarMACE allocation profile; runtime bundle not found: {}",
                bundle_path.display()
            );
            return Ok(());
        }
        let model = match LlmlipModel::load(&bundle_path)? {
            LlmlipModel::MacePolar(model) => model,
            other => {
                return Err(std::io::Error::other(format!(
                    "expected PolarMACE runtime bundle, got {other:?}"
                ))
                .into());
            }
        };
        let prepared = model.prepare_prediction_input(&structure)?;

        let _ = model.predict_prepared(&prepared, &options)?;
        let _ = model.predict_prepared(&prepared, &options)?;
        let (first, first_allocations) =
            measure_prediction(|| Ok(model.predict_prepared(&prepared, &options)?))?;
        let (second, second_allocations) =
            measure_prediction(|| Ok(model.predict_prepared(&prepared, &options)?))?;

        eprintln!(
            "polarmace_polar_1_{suffix}_prepared_allocations first={first_allocations:?} second={second_allocations:?}"
        );
        assert_eq!(first.energy, second.energy);
        assert_eq!(first.forces, second.forces);
        assert!(
            second_allocations.allocation_events() > 0,
            "public Prediction path should still report owned output/internal allocations; this diagnostic is not a zero-allocation proof"
        );
        assert_eq!(
            first_allocations.allocation_events(),
            second_allocations.allocation_events(),
            "same-shape prepared prediction allocation event count should be stable for polar-1-{suffix}"
        );
        assert_eq!(
            first_allocations.allocated_bytes + first_allocations.reallocated_bytes,
            second_allocations.allocated_bytes + second_allocations.reallocated_bytes,
            "same-shape prepared prediction allocated byte count should be stable for polar-1-{suffix}"
        );
        assert_eq!(
            first_allocations.deallocations, second_allocations.deallocations,
            "same-shape prepared prediction deallocation count should be stable for polar-1-{suffix}"
        );
    }

    Ok(())
}
