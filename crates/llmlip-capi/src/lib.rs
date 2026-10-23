//! C ABI boundary for `.e3rs` fixed-weight inference.

use std::{
    cell::RefCell,
    ffi::{CStr, CString},
    os::raw::c_char,
    panic::{catch_unwind, AssertUnwindSafe},
    path::Path,
    ptr, slice,
    sync::Mutex,
};

use llmlip::{
    Cell, ExternalGraphModelCapabilities, ExternalGraphView, LlmlipModel, ModelError, Prediction,
    PredictionComputeDtype, PredictionOptions, Structure,
};
use thiserror::Error;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

const EMPTY_ERROR: &[u8] = b"\0";

pub struct LlmlipModelHandle {
    model: LlmlipModel,
    structure_scratch: Mutex<StructureScratch>,
    external_graph_scratch: Mutex<ExternalGraphScratch>,
}

impl LlmlipModelHandle {
    fn into_raw(model: LlmlipModel) -> *mut Self {
        Box::into_raw(Box::new(Self {
            model,
            structure_scratch: Mutex::default(),
            external_graph_scratch: Mutex::default(),
        }))
    }
}

#[derive(Debug, Default)]
struct StructureScratch {
    species: Vec<String>,
    positions: Vec<[f64; 3]>,
}

#[derive(Debug, Default)]
struct ExternalGraphScratch {
    atomic_numbers: Vec<u8>,
    centers: Vec<usize>,
    center_owned: Vec<bool>,
    edge_offsets: Vec<usize>,
    dst: Vec<usize>,
    edge_vectors: Vec<[f64; 3]>,
}

struct ExternalGraphScratchInput<'a> {
    raw_atomic_numbers: &'a [i32],
    raw_centers: &'a [i64],
    raw_center_owned: &'a [u8],
    raw_edge_offsets: &'a [i64],
    raw_dst: &'a [i64],
    raw_edge_vectors: &'a [f64],
    atom_count: usize,
    owned_atom_count: usize,
    model_cutoff: f64,
}

impl ExternalGraphScratch {
    fn populate<'a>(
        &'a mut self,
        input: ExternalGraphScratchInput<'_>,
    ) -> Result<ExternalGraphView<'a>, CApiError> {
        convert_atomic_numbers_into(input.raw_atomic_numbers, &mut self.atomic_numbers)?;
        convert_i64_indices_into(input.raw_centers, "graph.centers", &mut self.centers)?;
        convert_center_owned_into(input.raw_center_owned, &mut self.center_owned)?;
        convert_i64_indices_into(
            input.raw_edge_offsets,
            "graph.edge_offsets",
            &mut self.edge_offsets,
        )?;
        convert_i64_indices_into(input.raw_dst, "graph.dst", &mut self.dst)?;
        convert_edge_vectors_into(input.raw_edge_vectors, &mut self.edge_vectors);

        Ok(ExternalGraphView {
            atom_count: input.atom_count,
            owned_atom_count: input.owned_atom_count,
            atomic_numbers: &self.atomic_numbers,
            center_atom_indices: &self.centers,
            center_owned: &self.center_owned,
            edge_offsets: &self.edge_offsets,
            edge_dst: &self.dst,
            edge_vectors: &self.edge_vectors,
            model_cutoff: input.model_cutoff,
        })
    }
}

impl StructureScratch {
    fn build_structure(
        &mut self,
        raw_atomic_numbers: &[i32],
        raw_positions: &[f64],
        raw_cell: Option<&[f64]>,
        pbc: [bool; 3],
    ) -> Result<Structure, CApiError> {
        self.species
            .resize_with(raw_atomic_numbers.len(), String::new);
        for (index, (slot, value)) in self.species.iter_mut().zip(raw_atomic_numbers).enumerate() {
            let symbol = element_symbol_from_atomic_number(*value).ok_or_else(|| {
                CApiError::InvalidArgument(format!(
                    "atomic_numbers[{index}] is not a supported element atomic number: {value}"
                ))
            })?;
            slot.clear();
            slot.push_str(symbol);
        }

        self.positions.clear();
        self.positions.reserve(raw_atomic_numbers.len());
        for chunk in raw_positions.chunks_exact(3) {
            self.positions.push([chunk[0], chunk[1], chunk[2]]);
        }

        let species = std::mem::take(&mut self.species);
        let positions = std::mem::take(&mut self.positions);
        let mut structure = Structure::new(species, positions);
        if pbc.iter().any(|is_periodic| *is_periodic) {
            let raw_cell = raw_cell.ok_or(CApiError::NullPointer(
                "structure.cell for periodic structure",
            ))?;
            structure = structure.with_cell(
                Cell::new([
                    [raw_cell[0], raw_cell[1], raw_cell[2]],
                    [raw_cell[3], raw_cell[4], raw_cell[5]],
                    [raw_cell[6], raw_cell[7], raw_cell[8]],
                ]),
                pbc,
            );
        }
        Ok(structure)
    }

    fn reclaim(&mut self, mut structure: Structure) {
        self.species = std::mem::take(&mut structure.species);
        self.positions = std::mem::take(&mut structure.positions);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlmlipStatus {
    Ok = 0,
    NullPointer = 1,
    InvalidArgument = 2,
    ModelLoad = 3,
    Unsupported = 4,
    Prediction = 5,
    Panic = 6,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlmlipDtype {
    F64 = 0,
    F32 = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlmlipModelCapabilities {
    pub model_cutoff: f64,
    pub mpi_halo_cutoff: f64,
    pub message_passing_depth: i64,
    pub supports_non_pbc: i32,
    pub supports_partial_pbc: i32,
    pub supports_full_pbc: i32,
    pub supports_forces: i32,
    pub supports_external_graph: i32,
    pub supports_external_graph_ghost_atoms: i32,
    pub supports_external_graph_global_virial: i32,
    pub supports_external_graph_global_stress: i32,
    pub supports_global_virial: i32,
    pub supports_global_stress: i32,
    pub supports_per_atom_energy: i32,
    pub supports_per_atom_virial: i32,
    pub supports_f64: i32,
    pub supports_f32: i32,
    pub requires_total_charge: i32,
    pub requires_spin_multiplicity: i32,
    pub supports_selected_head: i32,
    pub supported_head_count: i64,
}

impl Default for LlmlipModelCapabilities {
    fn default() -> Self {
        Self {
            model_cutoff: 0.0,
            mpi_halo_cutoff: 0.0,
            message_passing_depth: 0,
            supports_non_pbc: 0,
            supports_partial_pbc: 0,
            supports_full_pbc: 0,
            supports_forces: 0,
            supports_external_graph: 0,
            supports_external_graph_ghost_atoms: 0,
            supports_external_graph_global_virial: 0,
            supports_external_graph_global_stress: 0,
            supports_global_virial: 0,
            supports_global_stress: 0,
            supports_per_atom_energy: 0,
            supports_per_atom_virial: 0,
            supports_f64: 0,
            supports_f32: 0,
            requires_total_charge: 0,
            requires_spin_multiplicity: 0,
            supports_selected_head: 0,
            supported_head_count: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlmlipStructureView {
    pub n_atoms_total: i64,
    pub n_atoms_owned: i64,
    pub atomic_numbers: *const i32,
    pub positions: *const f64,
    pub cell: *const f64,
    pub pbc: [i32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlmlipExternalGraphView {
    pub n_centers: i64,
    pub centers: *const i64,
    pub center_owned: *const u8,
    pub edge_offsets: *const i64,
    pub dst: *const i64,
    pub edge_vector: *const f64,
    pub model_cutoff: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlmlipPredictionOptions {
    pub compute_energy: i32,
    pub compute_forces: i32,
    pub compute_virial: i32,
    pub compute_stress: i32,
    pub dtype: LlmlipDtype,
    pub selected_head: *const c_char,
    pub has_total_charge: i32,
    pub total_charge: f64,
    pub has_spin_multiplicity: i32,
    pub spin_multiplicity: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LlmlipPredictionOut {
    pub has_energy: i32,
    pub energy: f64,
    pub has_forces: i32,
    pub forces: *mut f64,
    pub has_virial: i32,
    pub virial: [f64; 9],
    pub has_stress: i32,
    pub stress: [f64; 9],
}

#[derive(Debug, Error)]
enum CApiError {
    #[error("null pointer: {0}")]
    NullPointer(&'static str),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("model load failed: {0}")]
    ModelLoad(String),

    #[error("unsupported request: {0}")]
    Unsupported(String),

    #[error("prediction failed: {0}")]
    Prediction(String),
}

impl CApiError {
    fn status(&self) -> LlmlipStatus {
        match self {
            Self::NullPointer(_) => LlmlipStatus::NullPointer,
            Self::InvalidArgument(_) => LlmlipStatus::InvalidArgument,
            Self::ModelLoad(_) => LlmlipStatus::ModelLoad,
            Self::Unsupported(_) => LlmlipStatus::Unsupported,
            Self::Prediction(_) => LlmlipStatus::Prediction,
        }
    }
}

#[no_mangle]
pub extern "C" fn llmlip_model_load(
    model_dir: *const c_char,
    out: *mut *mut LlmlipModelHandle,
) -> LlmlipStatus {
    ffi_boundary(|| {
        let out_ref = required_mut(out, "out")?;
        *out_ref = ptr::null_mut();
        let model_dir = c_str_to_str(model_dir, "model_dir")?;
        let model = LlmlipModel::load(Path::new(model_dir))
            .map_err(|err| CApiError::ModelLoad(err.to_string()))?;
        *out_ref = LlmlipModelHandle::into_raw(model);
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn llmlip_model_load_baked_mace_oam_medium(
    out: *mut *mut LlmlipModelHandle,
) -> LlmlipStatus {
    ffi_boundary(|| {
        let out_ref = required_mut(out, "out")?;
        *out_ref = ptr::null_mut();
        #[cfg(feature = "baked-mace-oam-medium")]
        {
            let model = LlmlipModel::load_baked_mace_oam_medium()
                .map_err(|err| CApiError::ModelLoad(err.to_string()))?;
            *out_ref = LlmlipModelHandle::into_raw(model);
            Ok(())
        }
        #[cfg(not(feature = "baked-mace-oam-medium"))]
        {
            Err(CApiError::Unsupported(
                "this llmlip-capi build does not include baked MACE-OAM Medium; rebuild with --features baked-mace-oam-medium"
                    .to_owned(),
            ))
        }
    })
}

#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn llmlip_model_free(model: *mut LlmlipModelHandle) {
    if model.is_null() {
        return;
    }
    // SAFETY: `model` must be a pointer previously returned by
    // `llmlip_model_load` and not yet freed. Null is handled above.
    unsafe {
        drop(Box::from_raw(model));
    }
}

#[no_mangle]
pub extern "C" fn llmlip_model_capabilities(
    model: *const LlmlipModelHandle,
    out: *mut LlmlipModelCapabilities,
) -> LlmlipStatus {
    ffi_boundary(|| {
        let model = required_ref(model, "model")?;
        let out = required_mut(out, "out")?;
        *out = capabilities_to_c(&model.model.external_graph_capabilities())?;
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn llmlip_predict_external_graph(
    model: *const LlmlipModelHandle,
    structure: *const LlmlipStructureView,
    graph: *const LlmlipExternalGraphView,
    options: *const LlmlipPredictionOptions,
    out: *mut LlmlipPredictionOut,
) -> LlmlipStatus {
    ffi_boundary(|| {
        let model = required_ref(model, "model")?;
        let structure = required_ref(structure, "structure")?;
        let graph = required_ref(graph, "graph")?;
        let options = required_ref(options, "options")?;
        let out = required_mut(out, "out")?;
        clear_prediction_out(out);

        let compute_dtype = prediction_compute_dtype_from_c(options.dtype);
        let prediction_options = prediction_options_from_c(options)?;
        let atom_count = checked_i64_to_usize(structure.n_atoms_total, "n_atoms_total")?;
        let owned_atom_count = checked_i64_to_usize(structure.n_atoms_owned, "n_atoms_owned")?;
        let center_count = checked_i64_to_usize(graph.n_centers, "n_centers")?;
        let raw_edge_offsets = slice_from_raw(
            graph.edge_offsets,
            center_count
                .checked_add(1)
                .ok_or_else(|| CApiError::InvalidArgument("n_centers + 1 overflowed".to_owned()))?,
            "graph.edge_offsets",
        )?;
        let edge_count =
            checked_i64_to_usize(*raw_edge_offsets.last().unwrap_or(&0), "edge_offsets.last")?;

        let raw_atomic_numbers = slice_from_raw(
            structure.atomic_numbers,
            atom_count,
            "structure.atomic_numbers",
        )?;
        let raw_centers = slice_from_raw(graph.centers, center_count, "graph.centers")?;
        let raw_center_owned =
            slice_from_raw(graph.center_owned, center_count, "graph.center_owned")?;
        let raw_dst = slice_from_raw(graph.dst, edge_count, "graph.dst")?;
        let raw_edge_vectors = slice_from_raw(
            graph.edge_vector,
            edge_count
                .checked_mul(3)
                .ok_or_else(|| CApiError::InvalidArgument("n_edges * 3 overflowed".to_owned()))?,
            "graph.edge_vector",
        )?;

        let mut scratch = model.external_graph_scratch.lock().map_err(|_| {
            CApiError::Prediction("external graph scratch lock poisoned".to_owned())
        })?;
        let graph_view = scratch.populate(ExternalGraphScratchInput {
            raw_atomic_numbers,
            raw_centers,
            raw_center_owned,
            raw_edge_offsets,
            raw_dst,
            raw_edge_vectors,
            atom_count,
            owned_atom_count,
            model_cutoff: graph.model_cutoff,
        })?;

        let prediction = model
            .model
            .predict_external_graph_with_compute_dtype(
                &graph_view,
                &prediction_options,
                compute_dtype,
            )
            .map_err(map_prediction_error)?;
        write_prediction(out, atom_count, prediction)
    })
}

#[no_mangle]
pub extern "C" fn llmlip_predict_structure(
    model: *const LlmlipModelHandle,
    structure: *const LlmlipStructureView,
    options: *const LlmlipPredictionOptions,
    out: *mut LlmlipPredictionOut,
) -> LlmlipStatus {
    ffi_boundary(|| {
        let model = required_ref(model, "model")?;
        let structure_view = required_ref(structure, "structure")?;
        let options = required_ref(options, "options")?;
        let out = required_mut(out, "out")?;
        clear_prediction_out(out);

        let compute_dtype = prediction_compute_dtype_from_c(options.dtype);
        let prediction_options = prediction_options_from_c(options)?;
        let atom_count = checked_i64_to_usize(structure_view.n_atoms_total, "n_atoms_total")?;
        let owned_atom_count = checked_i64_to_usize(structure_view.n_atoms_owned, "n_atoms_owned")?;
        if owned_atom_count != atom_count {
            return Err(CApiError::InvalidArgument(format!(
                "structure prediction requires n_atoms_owned == n_atoms_total, got {owned_atom_count} and {atom_count}"
            )));
        }
        let raw_atomic_numbers = slice_from_raw(
            structure_view.atomic_numbers,
            atom_count,
            "structure.atomic_numbers",
        )?;
        let raw_positions = slice_from_raw(
            structure_view.positions,
            atom_count.checked_mul(3).ok_or_else(|| {
                CApiError::InvalidArgument("n_atoms_total * 3 overflowed".to_owned())
            })?,
            "structure.positions",
        )?;
        let raw_cell = if structure_view.cell.is_null() {
            None
        } else {
            Some(slice_from_raw(structure_view.cell, 9, "structure.cell")?)
        };
        let pbc = pbc_from_c(structure_view.pbc)?;

        let mut scratch = model
            .structure_scratch
            .lock()
            .map_err(|_| CApiError::Prediction("structure scratch lock poisoned".to_owned()))?;
        let structure =
            scratch.build_structure(raw_atomic_numbers, raw_positions, raw_cell, pbc)?;
        let prediction_result =
            model
                .model
                .predict_with_compute_dtype(&structure, &prediction_options, compute_dtype);
        scratch.reclaim(structure);
        let prediction = prediction_result.map_err(map_prediction_error)?;
        write_prediction(out, atom_count, prediction)
    })
}

#[no_mangle]
pub extern "C" fn llmlip_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| match slot.borrow().as_ref() {
        Some(message) => message.as_ptr(),
        None => EMPTY_ERROR.as_ptr().cast(),
    })
}

#[no_mangle]
pub extern "C" fn llmlip_clear_error() {
    clear_last_error();
}

fn ffi_boundary(action: impl FnOnce() -> Result<(), CApiError>) -> LlmlipStatus {
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(action)) {
        Ok(Ok(())) => LlmlipStatus::Ok,
        Ok(Err(error)) => {
            let status = error.status();
            set_last_error(error.to_string());
            status
        }
        Err(_) => {
            set_last_error("panic crossed llmlip C ABI boundary".to_owned());
            LlmlipStatus::Panic
        }
    }
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

fn set_last_error(message: String) {
    let sanitized = message.replace('\0', "\\0");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(sanitized).ok();
    });
}

fn required_ref<'a, T>(ptr: *const T, name: &'static str) -> Result<&'a T, CApiError> {
    if ptr.is_null() {
        return Err(CApiError::NullPointer(name));
    }
    // SAFETY: caller supplied a non-null pointer and the C ABI contract
    // requires it to reference a valid `T` for the duration of this call.
    Ok(unsafe { &*ptr })
}

fn required_mut<'a, T>(ptr: *mut T, name: &'static str) -> Result<&'a mut T, CApiError> {
    if ptr.is_null() {
        return Err(CApiError::NullPointer(name));
    }
    // SAFETY: caller supplied a non-null pointer and the C ABI contract
    // requires unique mutable access to a valid `T` for this call.
    Ok(unsafe { &mut *ptr })
}

fn slice_from_raw<'a, T>(
    ptr: *const T,
    len: usize,
    name: &'static str,
) -> Result<&'a [T], CApiError> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(CApiError::NullPointer(name));
    }
    // SAFETY: the pointer is non-null and the C ABI contract requires at
    // least `len` initialized contiguous elements for this call.
    Ok(unsafe { slice::from_raw_parts(ptr, len) })
}

fn slice_from_raw_mut<'a, T>(
    ptr: *mut T,
    len: usize,
    name: &'static str,
) -> Result<&'a mut [T], CApiError> {
    if len == 0 {
        return Ok(&mut []);
    }
    if ptr.is_null() {
        return Err(CApiError::NullPointer(name));
    }
    // SAFETY: the pointer is non-null and the C ABI contract requires at
    // least `len` initialized contiguous elements with unique mutable access.
    Ok(unsafe { slice::from_raw_parts_mut(ptr, len) })
}

fn c_str_to_str<'a>(ptr: *const c_char, name: &'static str) -> Result<&'a str, CApiError> {
    if ptr.is_null() {
        return Err(CApiError::NullPointer(name));
    }
    // SAFETY: the pointer is non-null and the C ABI contract requires a
    // NUL-terminated string valid for this call.
    let c_str = unsafe { CStr::from_ptr(ptr) };
    c_str
        .to_str()
        .map_err(|err| CApiError::InvalidArgument(format!("{name} is not UTF-8: {err}")))
}

fn optional_c_str(ptr: *const c_char, name: &'static str) -> Result<Option<String>, CApiError> {
    if ptr.is_null() {
        return Ok(None);
    }
    c_str_to_str(ptr, name).map(|value| Some(value.to_owned()))
}

fn checked_i64_to_usize(value: i64, name: &'static str) -> Result<usize, CApiError> {
    if value < 0 {
        return Err(CApiError::InvalidArgument(format!(
            "{name} must be non-negative, got {value}"
        )));
    }
    usize::try_from(value).map_err(|_| {
        CApiError::InvalidArgument(format!("{name} does not fit in usize, got {value}"))
    })
}

fn usize_to_i64(value: usize, name: &'static str) -> Result<i64, CApiError> {
    i64::try_from(value)
        .map_err(|_| CApiError::InvalidArgument(format!("{name} does not fit in i64")))
}

fn i32_flag(value: bool) -> i32 {
    if value {
        1
    } else {
        0
    }
}

fn capabilities_to_c(
    capabilities: &ExternalGraphModelCapabilities,
) -> Result<LlmlipModelCapabilities, CApiError> {
    Ok(LlmlipModelCapabilities {
        model_cutoff: capabilities.model_cutoff,
        mpi_halo_cutoff: capabilities.mpi_halo_cutoff,
        message_passing_depth: usize_to_i64(
            capabilities.message_passing_depth,
            "message_passing_depth",
        )?,
        supports_non_pbc: i32_flag(capabilities.supports_non_pbc),
        supports_partial_pbc: i32_flag(capabilities.supports_partial_pbc),
        supports_full_pbc: i32_flag(capabilities.supports_full_pbc),
        supports_forces: i32_flag(capabilities.supports_forces),
        supports_external_graph: i32_flag(capabilities.supports_external_graph),
        supports_external_graph_ghost_atoms: i32_flag(
            capabilities.supports_external_graph_ghost_atoms,
        ),
        supports_external_graph_global_virial: i32_flag(
            capabilities.supports_external_graph_global_virial,
        ),
        supports_external_graph_global_stress: i32_flag(
            capabilities.supports_external_graph_global_stress,
        ),
        supports_global_virial: i32_flag(capabilities.supports_global_virial),
        supports_global_stress: i32_flag(capabilities.supports_global_stress),
        supports_per_atom_energy: 0,
        supports_per_atom_virial: 0,
        supports_f64: i32_flag(capabilities.supports_f64),
        supports_f32: i32_flag(capabilities.supports_f32),
        requires_total_charge: i32_flag(capabilities.requires_total_charge),
        requires_spin_multiplicity: i32_flag(capabilities.requires_spin_multiplicity),
        supports_selected_head: i32_flag(!capabilities.supported_heads.is_empty()),
        supported_head_count: usize_to_i64(
            capabilities.supported_heads.len(),
            "supported_head_count",
        )?,
    })
}

fn prediction_options_from_c(
    options: &LlmlipPredictionOptions,
) -> Result<PredictionOptions, CApiError> {
    Ok(PredictionOptions {
        compute_energy: options.compute_energy != 0,
        compute_forces: options.compute_forces != 0,
        compute_stress: options.compute_stress != 0,
        compute_virial: options.compute_virial != 0,
        total_charge: (options.has_total_charge != 0).then_some(options.total_charge),
        spin_multiplicity: (options.has_spin_multiplicity != 0)
            .then_some(options.spin_multiplicity),
        selected_head: optional_c_str(options.selected_head, "options.selected_head")?,
        ..PredictionOptions::default()
    })
}

fn prediction_compute_dtype_from_c(dtype: LlmlipDtype) -> PredictionComputeDtype {
    match dtype {
        LlmlipDtype::F64 => PredictionComputeDtype::Float64,
        LlmlipDtype::F32 => PredictionComputeDtype::Float32,
    }
}

fn pbc_from_c(raw: [i32; 3]) -> Result<[bool; 3], CApiError> {
    let mut pbc = [false; 3];
    for (axis, value) in raw.iter().enumerate() {
        pbc[axis] = match *value {
            0 => false,
            1 => true,
            other => {
                return Err(CApiError::InvalidArgument(format!(
                    "structure.pbc[{axis}] must be 0 or 1, got {other}"
                )))
            }
        };
    }
    Ok(pbc)
}

fn element_symbol_from_atomic_number(atomic_number: i32) -> Option<&'static str> {
    const SYMBOLS: [&str; 119] = [
        "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
        "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
        "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
        "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
        "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
        "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
        "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg",
        "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
    ];
    usize::try_from(atomic_number)
        .ok()
        .and_then(|index| SYMBOLS.get(index))
        .copied()
        .filter(|symbol| !symbol.is_empty())
}

fn convert_atomic_numbers_into(raw: &[i32], out: &mut Vec<u8>) -> Result<(), CApiError> {
    out.clear();
    out.reserve(raw.len());
    for (index, value) in raw.iter().enumerate() {
        out.push(u8::try_from(*value).map_err(|_| {
            CApiError::InvalidArgument(format!(
                "atomic_numbers[{index}] must fit in u8, got {value}"
            ))
        })?);
    }
    Ok(())
}

fn convert_i64_indices_into(
    raw: &[i64],
    name: &'static str,
    out: &mut Vec<usize>,
) -> Result<(), CApiError> {
    out.clear();
    out.reserve(raw.len());
    for (index, value) in raw.iter().enumerate() {
        if *value < 0 {
            return Err(CApiError::InvalidArgument(format!(
                "{name}[{index}] must be non-negative, got {value}"
            )));
        }
        out.push(usize::try_from(*value).map_err(|_| {
            CApiError::InvalidArgument(format!("{name}[{index}] does not fit in usize"))
        })?);
    }
    Ok(())
}

fn convert_center_owned_into(raw: &[u8], out: &mut Vec<bool>) -> Result<(), CApiError> {
    out.clear();
    out.reserve(raw.len());
    for (index, value) in raw.iter().enumerate() {
        out.push(match *value {
            0 => false,
            1 => true,
            other => {
                return Err(CApiError::InvalidArgument(format!(
                    "center_owned[{index}] must be 0 or 1, got {other}"
                )))
            }
        });
    }
    Ok(())
}

fn convert_edge_vectors_into(raw: &[f64], out: &mut Vec<[f64; 3]>) {
    out.clear();
    out.reserve(raw.len() / 3);
    for chunk in raw.chunks_exact(3) {
        out.push([chunk[0], chunk[1], chunk[2]]);
    }
}

fn clear_prediction_out(out: &mut LlmlipPredictionOut) {
    out.has_energy = 0;
    out.energy = 0.0;
    out.has_forces = 0;
    out.has_virial = 0;
    out.has_stress = 0;
    for value in &mut out.virial {
        *value = 0.0;
    }
    for value in &mut out.stress {
        *value = 0.0;
    }
}

fn write_prediction(
    out: &mut LlmlipPredictionOut,
    atom_count: usize,
    prediction: Prediction,
) -> Result<(), CApiError> {
    if let Some(energy) = prediction.energy {
        out.has_energy = 1;
        out.energy = energy;
    }
    if let Some(forces) = prediction.forces {
        if forces.len() != atom_count {
            return Err(CApiError::Prediction(format!(
                "force output length {} does not match atom count {atom_count}",
                forces.len()
            )));
        }
        let force_out = slice_from_raw_mut(
            out.forces,
            atom_count.checked_mul(3).ok_or_else(|| {
                CApiError::InvalidArgument("n_atoms_total * 3 overflowed".to_owned())
            })?,
            "out.forces",
        )?;
        for (chunk, force) in force_out.chunks_exact_mut(3).zip(&forces) {
            chunk.copy_from_slice(force);
        }
        out.has_forces = 1;
    }
    if let Some(virial) = prediction.virial {
        out.has_virial = 1;
        write_mat3(&mut out.virial, virial);
    }
    if let Some(stress) = prediction.stress {
        out.has_stress = 1;
        write_mat3(&mut out.stress, stress);
    }
    Ok(())
}

fn write_mat3(target: &mut [f64; 9], matrix: [[f64; 3]; 3]) {
    for row in 0..3 {
        for col in 0..3 {
            target[row * 3 + col] = matrix[row][col];
        }
    }
}

fn map_prediction_error(error: ModelError) -> CApiError {
    match &error {
        ModelError::UnsupportedComputeDtype(_)
        | ModelError::UnsupportedPredictionHead { .. }
        | ModelError::ExternalGraphPredictionNotSupported(_) => {
            CApiError::Unsupported(error.to_string())
        }
        ModelError::ExternalGraph(_) => CApiError::InvalidArgument(error.to_string()),
        _ => CApiError::Prediction(error.to_string()),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use std::{
        ffi::CString,
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn write_f64(path: &Path, values: &[f64]) {
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        fs::write(path, bytes).unwrap();
    }

    fn temp_model_dir(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("llmlip_capi_{name}_{nanos}"));
        fs::create_dir_all(root.join("weights")).unwrap();
        root
    }

    fn write_valid_model(root: &Path) {
        write_f64(
            &root.join("weights/edge_mlp.layers.0.weight.f64.bin"),
            &[
                0.8, -0.4, 0.2, 0.5, -0.3, 0.7, -0.6, 0.1, 0.25, 0.35, -0.45, 0.9,
            ],
        );
        write_f64(
            &root.join("weights/edge_mlp.layers.0.bias.f64.bin"),
            &[0.05, -0.1, 0.2],
        );
        write_f64(
            &root.join("weights/edge_mlp.layers.1.weight.f64.bin"),
            &[1.1, -0.75, 0.4],
        );
        write_f64(
            &root.join("weights/edge_mlp.layers.1.bias.f64.bin"),
            &[0.03],
        );
        fs::write(
            root.join("manifest.toml"),
            r#"
format_version = "0.1"
architecture = "synthetic-local-v1"
dtype = "f64"
length_unit = "angstrom"
energy_unit = "eV"
force_unit = "eV/angstrom"

[species]
symbols = ["H", "O"]

[geometry]
cutoff = 3.5
edge_vector = "R_j + shift - R_i"
directed_edges = true

[spherical_harmonics]
l_max = 2
normalize_input = true
normalization = "component"
reference_convention = "pinned-fixture"

[synthetic_radial]
centers = [0.5, 1.0, 1.5, 2.0]
gamma = 0.8

[[mlp_layers]]
weight = "edge_mlp.layers.0.weight"
bias = "edge_mlp.layers.0.bias"
activation = "tanh"

[[mlp_layers]]
weight = "edge_mlp.layers.1.weight"
bias = "edge_mlp.layers.1.bias"
activation = "identity"

[[tensors]]
name = "edge_mlp.layers.0.weight"
dtype = "f64"
shape = [3, 4]
layout = "row-major"
path = "weights/edge_mlp.layers.0.weight.f64.bin"

[[tensors]]
name = "edge_mlp.layers.0.bias"
dtype = "f64"
shape = [3]
layout = "row-major"
path = "weights/edge_mlp.layers.0.bias.f64.bin"

[[tensors]]
name = "edge_mlp.layers.1.weight"
dtype = "f64"
shape = [1, 3]
layout = "row-major"
path = "weights/edge_mlp.layers.1.weight.f64.bin"

[[tensors]]
name = "edge_mlp.layers.1.bias"
dtype = "f64"
shape = [1]
layout = "row-major"
path = "weights/edge_mlp.layers.1.bias.f64.bin"
"#,
        )
        .unwrap();
    }

    #[test]
    fn load_free_and_capabilities_smoke() {
        let root = temp_model_dir("load");
        write_valid_model(&root);
        let path = CString::new(root.to_string_lossy().as_bytes()).unwrap();
        let mut model = ptr::null_mut();

        assert_eq!(
            llmlip_model_load(path.as_ptr(), &mut model),
            LlmlipStatus::Ok
        );
        assert!(!model.is_null());

        let mut capabilities = LlmlipModelCapabilities::default();
        assert_eq!(
            llmlip_model_capabilities(model, &mut capabilities),
            LlmlipStatus::Ok
        );
        assert_eq!(capabilities.model_cutoff, 3.5);
        assert_eq!(capabilities.mpi_halo_cutoff, 3.5);
        assert_eq!(capabilities.supports_forces, 1);
        assert_eq!(capabilities.supports_external_graph, 1);
        assert_eq!(capabilities.supports_external_graph_ghost_atoms, 1);
        assert_eq!(capabilities.supports_external_graph_global_virial, 0);

        llmlip_model_free(model);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn null_load_sets_last_error() {
        let mut model = ptr::null_mut();
        assert_eq!(
            llmlip_model_load(ptr::null(), &mut model),
            LlmlipStatus::NullPointer
        );
        // SAFETY: `llmlip_last_error` returns a valid NUL-terminated process-local
        // error string pointer.
        let message = unsafe { CStr::from_ptr(llmlip_last_error()) }
            .to_str()
            .unwrap();
        assert!(message.contains("model_dir"));
    }

    #[test]
    #[cfg(not(feature = "baked-mace-oam-medium"))]
    fn baked_loader_reports_unsupported_without_feature() {
        let mut model = ptr::null_mut();

        assert_eq!(
            llmlip_model_load_baked_mace_oam_medium(&mut model),
            LlmlipStatus::Unsupported
        );
        assert!(model.is_null());
        // SAFETY: `llmlip_last_error` returns a valid NUL-terminated process-local
        // error string pointer.
        let message = unsafe { CStr::from_ptr(llmlip_last_error()) }
            .to_str()
            .unwrap();
        assert!(message.contains("baked MACE-OAM Medium"));
    }

    #[test]
    fn external_graph_predict_smoke() {
        let root = temp_model_dir("predict");
        write_valid_model(&root);
        let path = CString::new(root.to_string_lossy().as_bytes()).unwrap();
        let mut model = ptr::null_mut();
        assert_eq!(
            llmlip_model_load(path.as_ptr(), &mut model),
            LlmlipStatus::Ok
        );

        let atomic_numbers = [1_i32, 8_i32];
        let structure = LlmlipStructureView {
            n_atoms_total: 2,
            n_atoms_owned: 2,
            atomic_numbers: atomic_numbers.as_ptr(),
            positions: ptr::null(),
            cell: ptr::null(),
            pbc: [0, 0, 0],
        };
        let centers = [0_i64, 1_i64];
        let center_owned = [1_u8, 1_u8];
        let edge_offsets = [0_i64, 1_i64, 2_i64];
        let dst = [1_i64, 0_i64];
        let edge_vectors = [1.1_f64, -0.2, 0.4, -1.1, 0.2, -0.4];
        let graph = LlmlipExternalGraphView {
            n_centers: 2,
            centers: centers.as_ptr(),
            center_owned: center_owned.as_ptr(),
            edge_offsets: edge_offsets.as_ptr(),
            dst: dst.as_ptr(),
            edge_vector: edge_vectors.as_ptr(),
            model_cutoff: 3.5,
        };
        let options = LlmlipPredictionOptions {
            compute_energy: 1,
            compute_forces: 1,
            compute_virial: 0,
            compute_stress: 0,
            dtype: LlmlipDtype::F64,
            selected_head: ptr::null(),
            has_total_charge: 0,
            total_charge: 0.0,
            has_spin_multiplicity: 0,
            spin_multiplicity: 0.0,
        };
        let mut forces = [0.0_f64; 6];
        let mut out = LlmlipPredictionOut {
            has_energy: 0,
            energy: 0.0,
            has_forces: 0,
            forces: forces.as_mut_ptr(),
            has_virial: 0,
            virial: [0.0; 9],
            has_stress: 0,
            stress: [0.0; 9],
        };

        assert_eq!(
            llmlip_predict_external_graph(model, &structure, &graph, &options, &mut out),
            LlmlipStatus::Ok
        );
        assert_eq!(out.has_energy, 1);
        assert!(out.energy.is_finite());
        assert_eq!(out.has_forces, 1);
        assert!(forces.iter().any(|value| *value != 0.0));
        // SAFETY: `model` is still the live handle returned by
        // `llmlip_model_load`; the test has exclusive access to it.
        let first_edge_vector_capacity = unsafe {
            (*model)
                .external_graph_scratch
                .lock()
                .unwrap()
                .edge_vectors
                .capacity()
        };

        assert_eq!(
            llmlip_predict_external_graph(model, &structure, &graph, &options, &mut out),
            LlmlipStatus::Ok
        );
        // SAFETY: same live handle and exclusive test access as above.
        let second_edge_vector_capacity = unsafe {
            (*model)
                .external_graph_scratch
                .lock()
                .unwrap()
                .edge_vectors
                .capacity()
        };
        assert_eq!(second_edge_vector_capacity, first_edge_vector_capacity);

        llmlip_model_free(model);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn invalid_external_graph_sets_last_error() {
        let root = temp_model_dir("invalid_graph");
        write_valid_model(&root);
        let path = CString::new(root.to_string_lossy().as_bytes()).unwrap();
        let mut model = ptr::null_mut();
        assert_eq!(
            llmlip_model_load(path.as_ptr(), &mut model),
            LlmlipStatus::Ok
        );

        let atomic_numbers = [1_i32, 8_i32];
        let structure = LlmlipStructureView {
            n_atoms_total: 2,
            n_atoms_owned: 2,
            atomic_numbers: atomic_numbers.as_ptr(),
            positions: ptr::null(),
            cell: ptr::null(),
            pbc: [0, 0, 0],
        };
        let centers = [0_i64, 2_i64];
        let center_owned = [1_u8, 1_u8];
        let edge_offsets = [0_i64, 1_i64, 2_i64];
        let dst = [1_i64, 0_i64];
        let edge_vectors = [1.1_f64, -0.2, 0.4, -1.1, 0.2, -0.4];
        let graph = LlmlipExternalGraphView {
            n_centers: 2,
            centers: centers.as_ptr(),
            center_owned: center_owned.as_ptr(),
            edge_offsets: edge_offsets.as_ptr(),
            dst: dst.as_ptr(),
            edge_vector: edge_vectors.as_ptr(),
            model_cutoff: 3.5,
        };
        let options = LlmlipPredictionOptions {
            compute_energy: 1,
            compute_forces: 1,
            compute_virial: 0,
            compute_stress: 0,
            dtype: LlmlipDtype::F64,
            selected_head: ptr::null(),
            has_total_charge: 0,
            total_charge: 0.0,
            has_spin_multiplicity: 0,
            spin_multiplicity: 0.0,
        };
        let mut forces = [0.0_f64; 6];
        let mut out = LlmlipPredictionOut {
            has_energy: 0,
            energy: 0.0,
            has_forces: 0,
            forces: forces.as_mut_ptr(),
            has_virial: 0,
            virial: [0.0; 9],
            has_stress: 0,
            stress: [0.0; 9],
        };

        assert_eq!(
            llmlip_predict_external_graph(model, &structure, &graph, &options, &mut out),
            LlmlipStatus::InvalidArgument
        );
        // SAFETY: `llmlip_last_error` returns a valid NUL-terminated process-local
        // error string pointer.
        let message = unsafe { CStr::from_ptr(llmlip_last_error()) }
            .to_str()
            .unwrap();
        assert!(message.contains("outside atom count"));

        llmlip_model_free(model);
        fs::remove_dir_all(root).unwrap();
    }
}
