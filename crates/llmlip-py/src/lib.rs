use std::{path::PathBuf, sync::Arc};

use llmlip::{
    Cell, LlmlipModel, MaceMh1WignerMode, ModelCapabilities, ModelError, PredictionComputeDtype,
    PredictionInputRequirements, PredictionOptions, Structure, Vec3,
};
use numpy::{
    ndarray::{Array1, Array2},
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::{
    create_exception,
    exceptions::{PyException, PyValueError},
    prelude::*,
};

create_exception!(_llmlip, LlmlipError, PyException);
create_exception!(_llmlip, ModelLoadError, LlmlipError);
create_exception!(_llmlip, PredictionError, LlmlipError);
create_exception!(_llmlip, UnsupportedCapabilityError, LlmlipError);
create_exception!(_llmlip, InvalidStructureError, LlmlipError);
create_exception!(_llmlip, BackendUnavailableError, LlmlipError);
create_exception!(_llmlip, CliBackendError, LlmlipError);

#[pyclass(name = "PredictionInputRequirements", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyPredictionInputRequirements {
    #[pyo3(get)]
    total_charge: bool,
    #[pyo3(get)]
    spin_multiplicity: bool,
}

#[pymethods]
impl PyPredictionInputRequirements {
    fn __repr__(&self) -> String {
        format!(
            "PredictionInputRequirements(total_charge={}, spin_multiplicity={})",
            py_bool(self.total_charge),
            py_bool(self.spin_multiplicity)
        )
    }
}

#[pyclass(name = "ModelCapabilities", frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyModelCapabilities {
    #[pyo3(get)]
    architecture: String,
    #[pyo3(get)]
    supports_energy: bool,
    #[pyo3(get)]
    supports_forces: bool,
    #[pyo3(get)]
    supports_stress: bool,
    #[pyo3(get)]
    supports_virial: bool,
    #[pyo3(get)]
    supports_float64: bool,
    #[pyo3(get)]
    supports_float32: bool,
    #[pyo3(get)]
    supports_prepared_input: bool,
    #[pyo3(get)]
    stress_requires_full_pbc: bool,
    #[pyo3(get)]
    virial_requires_full_pbc: bool,
    #[pyo3(get)]
    supported_species: Vec<String>,
    #[pyo3(get)]
    supported_heads: Vec<String>,
    #[pyo3(get)]
    input_requirements: PyPredictionInputRequirements,
}

#[pymethods]
impl PyModelCapabilities {
    fn __repr__(&self) -> String {
        format!(
            "ModelCapabilities(architecture={:?}, supports_energy={}, supports_forces={}, supports_stress={}, supports_virial={}, supports_float64={}, supports_float32={}, supports_prepared_input={}, supported_species={:?}, supported_heads={:?})",
            self.architecture,
            py_bool(self.supports_energy),
            py_bool(self.supports_forces),
            py_bool(self.supports_stress),
            py_bool(self.supports_virial),
            py_bool(self.supports_float64),
            py_bool(self.supports_float32),
            py_bool(self.supports_prepared_input),
            self.supported_species,
            self.supported_heads
        )
    }
}

#[pyclass(name = "Prediction")]
struct PyPrediction {
    #[pyo3(get)]
    energy: Option<f64>,
    #[pyo3(get)]
    forces: Option<Py<PyArray2<f64>>>,
    #[pyo3(get)]
    stress: Option<Py<PyArray2<f64>>>,
    #[pyo3(get)]
    virial: Option<Py<PyArray2<f64>>>,
}

type PyExtraOutputs = (Py<PyArray2<f64>>, Py<PyArray1<f64>>);

#[pymethods]
impl PyPrediction {
    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "Prediction(energy={}, forces={}, stress={}, virial={})",
            py_option_f64(self.energy),
            py_array_option_repr(py, &self.forces),
            py_array_option_repr(py, &self.stress),
            py_array_option_repr(py, &self.virial)
        )
    }
}

#[pyclass(name = "Model")]
struct PyModel {
    inner: Arc<LlmlipModel>,
    path: PathBuf,
}

#[pymethods]
impl PyModel {
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        let model = LlmlipModel::load(&path).map_err(model_load_error)?;
        Ok(Self {
            inner: Arc::new(model),
            path,
        })
    }

    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        Self::new(path)
    }

    #[cfg(feature = "baked-mace-oam-medium")]
    #[staticmethod]
    fn load_baked_mace_oam_medium() -> PyResult<Self> {
        let model = LlmlipModel::load_baked_mace_oam_medium().map_err(model_load_error)?;
        Ok(Self {
            inner: Arc::new(model),
            path: PathBuf::from("<baked:mace-oam0-medium>"),
        })
    }

    #[getter]
    fn path(&self) -> String {
        self.path.display().to_string()
    }

    fn architecture(&self) -> String {
        self.inner.architecture().to_owned()
    }

    fn input_requirements(&self) -> PyPredictionInputRequirements {
        input_requirements_to_py(self.inner.prediction_input_requirements())
    }

    fn capabilities(&self) -> PyModelCapabilities {
        capabilities_to_py(self.inner.capabilities())
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(path={:?}, architecture={:?})",
            self.path.display().to_string(),
            self.inner.architecture()
        )
    }

    #[pyo3(signature = (
        *,
        species,
        positions,
        cell=None,
        pbc=(false, false, false),
        energy=true,
        forces=true,
        stress=false,
        virial=false,
        dtype="float64",
        head=None,
        total_charge=None,
        spin_multiplicity=None,
        mace_mh1_wigner_mode="true_float64",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        species: Vec<String>,
        positions: PyReadonlyArray2<'py, f64>,
        cell: Option<PyReadonlyArray2<'py, f64>>,
        pbc: (bool, bool, bool),
        energy: bool,
        forces: bool,
        stress: bool,
        virial: bool,
        dtype: &str,
        head: Option<String>,
        total_charge: Option<f64>,
        spin_multiplicity: Option<f64>,
        mace_mh1_wigner_mode: &str,
    ) -> PyResult<PyPrediction> {
        let structure = structure_from_numpy(species, positions, cell, pbc)?;
        let options = PredictionOptions {
            compute_energy: energy,
            compute_forces: forces,
            compute_stress: stress,
            compute_virial: virial,
            total_charge,
            spin_multiplicity,
            selected_head: head,
            mace_mh1_wigner_mode: parse_wigner_mode(mace_mh1_wigner_mode)?,
        };
        let dtype = parse_dtype(dtype)?;
        let model = Arc::clone(&self.inner);
        // PyO3 0.28's Python::detach is the GIL-release API; allow_threads was
        // the older spelling. Only owned Rust values enter this closure.
        let prediction = py
            .detach(move || model.predict_with_compute_dtype(&structure, &options, dtype))
            .map_err(prediction_error)?;
        prediction_to_py(py, prediction)
    }

    #[pyo3(signature = (
        *,
        species,
        positions,
        cell=None,
        pbc=(false, false, false),
        invariants_only=true,
        num_layers=-1,
        dtype="float64",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn get_descriptors<'py>(
        &self,
        py: Python<'py>,
        species: Vec<String>,
        positions: PyReadonlyArray2<'py, f64>,
        cell: Option<PyReadonlyArray2<'py, f64>>,
        pbc: (bool, bool, bool),
        invariants_only: bool,
        num_layers: isize,
        dtype: &str,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let structure = structure_from_numpy(species, positions, cell, pbc)?;
        let dtype = parse_dtype(dtype)?;
        let model = Arc::clone(&self.inner);
        let descriptors = py
            .detach(move || {
                model.get_descriptors_with_compute_dtype(
                    &structure,
                    invariants_only,
                    num_layers,
                    dtype,
                )
            })
            .map_err(prediction_error)?;
        vecvec_to_py(py, descriptors)
    }

    #[pyo3(signature = (
        *,
        species,
        positions,
        cell=None,
        pbc=(false, false, false),
        dtype="float64",
    ))]
    fn get_atomic_energies<'py>(
        &self,
        py: Python<'py>,
        species: Vec<String>,
        positions: PyReadonlyArray2<'py, f64>,
        cell: Option<PyReadonlyArray2<'py, f64>>,
        pbc: (bool, bool, bool),
        dtype: &str,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let structure = structure_from_numpy(species, positions, cell, pbc)?;
        let dtype = parse_dtype(dtype)?;
        let model = Arc::clone(&self.inner);
        let energies = py
            .detach(move || model.get_atomic_energies_with_compute_dtype(&structure, dtype))
            .map_err(prediction_error)?;
        array1_to_py(py, energies)
    }

    #[pyo3(signature = (
        *,
        species,
        positions,
        cell=None,
        pbc=(false, false, false),
        dtype="float64",
    ))]
    fn get_energies<'py>(
        &self,
        py: Python<'py>,
        species: Vec<String>,
        positions: PyReadonlyArray2<'py, f64>,
        cell: Option<PyReadonlyArray2<'py, f64>>,
        pbc: (bool, bool, bool),
        dtype: &str,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let structure = structure_from_numpy(species, positions, cell, pbc)?;
        let dtype = parse_dtype(dtype)?;
        let model = Arc::clone(&self.inner);
        let energies = py
            .detach(move || model.get_atomic_energies_with_compute_dtype(&structure, dtype))
            .map_err(prediction_error)?;
        array1_to_py(py, energies)
    }

    /// Descriptors and per-atom energies together from a single forward pass,
    /// returned as `(descriptors, atomic_energies)`. Equivalent to calling
    /// `get_descriptors` and `get_atomic_energies` separately but without the
    /// duplicate forward pass.
    #[pyo3(signature = (
        *,
        species,
        positions,
        cell=None,
        pbc=(false, false, false),
        invariants_only=true,
        num_layers=-1,
        dtype="float64",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn get_extra_outputs<'py>(
        &self,
        py: Python<'py>,
        species: Vec<String>,
        positions: PyReadonlyArray2<'py, f64>,
        cell: Option<PyReadonlyArray2<'py, f64>>,
        pbc: (bool, bool, bool),
        invariants_only: bool,
        num_layers: isize,
        dtype: &str,
    ) -> PyResult<PyExtraOutputs> {
        let structure = structure_from_numpy(species, positions, cell, pbc)?;
        let dtype = parse_dtype(dtype)?;
        let model = Arc::clone(&self.inner);
        let (descriptors, energies) = py
            .detach(move || {
                model.get_extra_outputs_with_compute_dtype(
                    &structure,
                    true,
                    true,
                    invariants_only,
                    num_layers,
                    dtype,
                )
            })
            .map_err(prediction_error)?;
        let descriptors =
            descriptors.ok_or_else(|| PyValueError::new_err("model did not return descriptors"))?;
        let energies = energies
            .ok_or_else(|| PyValueError::new_err("model did not return atomic energies"))?;
        Ok((vecvec_to_py(py, descriptors)?, array1_to_py(py, energies)?))
    }
}

fn py_bool(value: bool) -> &'static str {
    if value {
        "True"
    } else {
        "False"
    }
}

fn py_option_f64(value: Option<f64>) -> String {
    value.map_or_else(|| "None".to_owned(), |value| format!("{value:?}"))
}

fn py_array_option_repr(py: Python<'_>, value: &Option<Py<PyArray2<f64>>>) -> String {
    match value {
        Some(array) => {
            let bound = array.bind(py);
            let shape = bound.shape();
            format!("array(shape=({}, {}), dtype=float64)", shape[0], shape[1])
        }
        None => "None".to_owned(),
    }
}

fn structure_from_numpy<'py>(
    species: Vec<String>,
    positions: PyReadonlyArray2<'py, f64>,
    cell: Option<PyReadonlyArray2<'py, f64>>,
    pbc: (bool, bool, bool),
) -> PyResult<Structure> {
    let pos = positions.as_array();
    if pos.ndim() != 2 || pos.shape()[1] != 3 {
        return Err(InvalidStructureError::new_err(
            "positions must have shape (n, 3)",
        ));
    }
    if pos.shape()[0] != species.len() {
        return Err(InvalidStructureError::new_err(
            "species length must match positions.shape[0]",
        ));
    }
    let positions_vec: Vec<Vec3> = pos
        .outer_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();
    let mut structure = Structure::new(species, positions_vec);
    if let Some(cell_array) = cell {
        let cell_view = cell_array.as_array();
        if cell_view.ndim() != 2 || cell_view.shape() != [3_usize, 3_usize] {
            return Err(InvalidStructureError::new_err(
                "cell must have shape (3, 3)",
            ));
        }
        let vectors = [
            [cell_view[[0, 0]], cell_view[[0, 1]], cell_view[[0, 2]]],
            [cell_view[[1, 0]], cell_view[[1, 1]], cell_view[[1, 2]]],
            [cell_view[[2, 0]], cell_view[[2, 1]], cell_view[[2, 2]]],
        ];
        structure = structure.with_cell(Cell::new(vectors), [pbc.0, pbc.1, pbc.2]);
    }
    Ok(structure)
}

fn parse_dtype(dtype: &str) -> PyResult<PredictionComputeDtype> {
    match dtype {
        "float64" | "f64" | "fp64" => Ok(PredictionComputeDtype::Float64),
        "float32" | "f32" | "fp32" => Ok(PredictionComputeDtype::Float32),
        other => Err(UnsupportedCapabilityError::new_err(format!(
            "unsupported dtype: {other}"
        ))),
    }
}

fn parse_wigner_mode(mode: &str) -> PyResult<MaceMh1WignerMode> {
    match mode {
        "true_float64" | "true-float64" => Ok(MaceMh1WignerMode::TrueFloat64),
        "upstream_artifact" | "upstream-artifact" => Ok(MaceMh1WignerMode::UpstreamArtifact),
        other => Err(PyValueError::new_err(format!(
            "unsupported mace_mh1_wigner_mode: {other}"
        ))),
    }
}

fn prediction_to_py(py: Python<'_>, prediction: llmlip::Prediction) -> PyResult<PyPrediction> {
    let forces = match prediction.forces {
        Some(values) => {
            let flat = values
                .iter()
                .flat_map(|value| value.iter().copied())
                .collect::<Vec<_>>();
            Some(array2_to_py(py, values.len(), 3, flat)?)
        }
        None => None,
    };
    let stress = prediction
        .stress
        .map(|matrix| mat3_to_py(py, matrix))
        .transpose()?;
    let virial = prediction
        .virial
        .map(|matrix| mat3_to_py(py, matrix))
        .transpose()?;
    Ok(PyPrediction {
        energy: prediction.energy,
        forces,
        stress,
        virial,
    })
}

fn mat3_to_py(py: Python<'_>, matrix: [[f64; 3]; 3]) -> PyResult<Py<PyArray2<f64>>> {
    let flat = matrix.into_iter().flatten().collect::<Vec<_>>();
    array2_to_py(py, 3, 3, flat)
}

fn array2_to_py(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    values: Vec<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let array = Array2::from_shape_vec((rows, cols), values)
        .map_err(|err| PredictionError::new_err(err.to_string()))?;
    Ok(array.into_pyarray(py).unbind())
}

fn array1_to_py(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyArray1<f64>>> {
    Ok(Array1::from(values).into_pyarray(py).unbind())
}

fn vecvec_to_py(py: Python<'_>, values: Vec<Vec<f64>>) -> PyResult<Py<PyArray2<f64>>> {
    let rows = values.len();
    let cols = values.first().map_or(0, Vec::len);
    if values.iter().any(|row| row.len() != cols) {
        return Err(PredictionError::new_err(
            "descriptor rows must all have the same length",
        ));
    }
    let flat = values.into_iter().flatten().collect::<Vec<_>>();
    array2_to_py(py, rows, cols, flat)
}

fn input_requirements_to_py(
    requirements: PredictionInputRequirements,
) -> PyPredictionInputRequirements {
    PyPredictionInputRequirements {
        total_charge: requirements.total_charge,
        spin_multiplicity: requirements.spin_multiplicity,
    }
}

fn capabilities_to_py(capabilities: ModelCapabilities) -> PyModelCapabilities {
    PyModelCapabilities {
        architecture: capabilities.architecture.to_owned(),
        supports_energy: capabilities.supports_energy,
        supports_forces: capabilities.supports_forces,
        supports_stress: capabilities.supports_stress,
        supports_virial: capabilities.supports_virial,
        supports_float64: capabilities.supports_float64,
        supports_float32: capabilities.supports_float32,
        supports_prepared_input: capabilities.supports_prepared_input,
        stress_requires_full_pbc: capabilities.stress_requires_full_pbc,
        virial_requires_full_pbc: capabilities.virial_requires_full_pbc,
        supported_species: capabilities.supported_species,
        supported_heads: capabilities.supported_heads,
        input_requirements: input_requirements_to_py(capabilities.input_requirements),
    }
}

fn model_load_error(err: ModelError) -> PyErr {
    let message = format!("failed to load llmlip model: {err}");
    match &err {
        ModelError::Io { .. }
        | ModelError::ManifestToml(_)
        | ModelError::DispersionJson(_)
        | ModelError::EsenOamStateJson(_)
        | ModelError::InvalidManifest(_)
        | ModelError::MissingTensor(_)
        | ModelError::TensorSizeMismatch { .. } => ModelLoadError::new_err(message),
        ModelError::UnsupportedArchitecture(_) | ModelError::ArchitectureNotImplemented(_) => {
            UnsupportedCapabilityError::new_err(message)
        }
        _ => ModelLoadError::new_err(message),
    }
}

fn prediction_error(err: ModelError) -> PyErr {
    let message = format!("llmlip prediction failed: {err}");
    match &err {
        ModelError::UnsupportedComputeDtype(_)
        | ModelError::UnsupportedPredictionHead { .. }
        | ModelError::PreparedPredictionNotSupported(_)
        | ModelError::ArchitectureNotImplemented(_) => UnsupportedCapabilityError::new_err(message),
        ModelError::MissingRequestedOutput { .. } => PredictionError::new_err(message),
        _ => PredictionError::new_err(message),
    }
}

#[pymodule]
#[pyo3(name = "_llmlip")]
fn llmlip_native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyPrediction>()?;
    m.add_class::<PyPredictionInputRequirements>()?;
    m.add_class::<PyModelCapabilities>()?;
    m.add("LlmlipError", py.get_type::<LlmlipError>())?;
    m.add("ModelLoadError", py.get_type::<ModelLoadError>())?;
    m.add("PredictionError", py.get_type::<PredictionError>())?;
    m.add(
        "UnsupportedCapabilityError",
        py.get_type::<UnsupportedCapabilityError>(),
    )?;
    m.add(
        "InvalidStructureError",
        py.get_type::<InvalidStructureError>(),
    )?;
    m.add(
        "BackendUnavailableError",
        py.get_type::<BackendUnavailableError>(),
    )?;
    m.add("CliBackendError", py.get_type::<CliBackendError>())?;
    Ok(())
}
