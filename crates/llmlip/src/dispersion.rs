//! Composition helpers for fixed-weight models plus dispersion corrections.
//!
//! The model loader can wrap a base `.e3rs` model with these helpers when a
//! top-level `[dispersion]` section is present. Direct construction remains
//! available for tests and callers that assemble fixed corrections explicitly.

use thiserror::Error;

use crate::{
    dftd3::{Dftd3BjCoordinationModel, Dftd3BjTwoBodyModel, Dftd3Error},
    force_accum::Mat3,
    model::{LlmlipModel, ModelError},
    neighbor::{cell_list_multi_image_neighbor_list, NeighborList, NeighborListError},
    structure::{Prediction, PredictionOptions, Structure},
    Vec3,
};

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum DispersionCorrection {
    Dftd3BjTwoBody(Dftd3BjTwoBodyModel),
    Dftd3BjCoordination(Box<Dftd3BjCoordinationModel>),
}

impl DispersionCorrection {
    pub fn predict(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
    ) -> Result<Prediction, Dftd3Error> {
        match self {
            Self::Dftd3BjTwoBody(model) => model.predict(structure, options),
            Self::Dftd3BjCoordination(model) => model.predict(structure, options),
        }
    }

    fn predict_with_neighbor_list(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
        neighbor_list: &NeighborList,
    ) -> Result<Prediction, Dftd3Error> {
        match self {
            Self::Dftd3BjTwoBody(model) => {
                model.predict_with_neighbor_list(structure, options, neighbor_list)
            }
            Self::Dftd3BjCoordination(model) => {
                model.predict_with_neighbor_list(structure, options, neighbor_list)
            }
        }
    }

    fn required_neighbor_cutoff(&self) -> f64 {
        match self {
            Self::Dftd3BjTwoBody(model) => model.required_neighbor_cutoff(),
            Self::Dftd3BjCoordination(model) => model.required_neighbor_cutoff(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DispersionCorrectedModel {
    base: LlmlipModel,
    correction: DispersionCorrection,
}

impl DispersionCorrectedModel {
    #[must_use]
    pub fn new(base: LlmlipModel, correction: DispersionCorrection) -> Self {
        Self { base, correction }
    }

    #[must_use]
    pub fn base(&self) -> &LlmlipModel {
        &self.base
    }

    #[must_use]
    pub fn correction(&self) -> &DispersionCorrection {
        &self.correction
    }

    pub(crate) fn required_neighbor_cutoff(&self) -> f64 {
        self.correction.required_neighbor_cutoff()
    }

    pub fn predict(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
    ) -> Result<Prediction, DispersionCorrectedError> {
        if !options.compute_energy
            && !options.compute_forces
            && !options.compute_stress
            && !options.compute_virial
        {
            return Ok(Prediction::default());
        }

        let (base, correction) = if let Some(shared_cutoff) = self.shared_neighbor_cutoff() {
            let shared_neighbor_list =
                cell_list_multi_image_neighbor_list(structure, shared_cutoff)
                    .map_err(DispersionCorrectedError::SharedNeighborList)?;
            let base = match &self.base {
                LlmlipModel::MaceLite(model) => model
                    .predict_with_neighbor_list(structure, options, &shared_neighbor_list)
                    .map_err(|err| DispersionCorrectedError::Base(ModelError::from(err)))?,
                _ => self
                    .base
                    .predict(structure, options)
                    .map_err(DispersionCorrectedError::Base)?,
            };
            let correction = self
                .correction
                .predict_with_neighbor_list(structure, options, &shared_neighbor_list)
                .map_err(DispersionCorrectedError::Correction)?;
            (base, correction)
        } else {
            let base = self
                .base
                .predict(structure, options)
                .map_err(DispersionCorrectedError::Base)?;
            let correction = self
                .correction
                .predict(structure, options)
                .map_err(DispersionCorrectedError::Correction)?;
            (base, correction)
        };

        Ok(Prediction {
            energy: if options.compute_energy {
                Some(
                    requested_scalar("energy", base.energy)?
                        + requested_scalar("energy", correction.energy)?,
                )
            } else {
                None
            },
            forces: if options.compute_forces {
                Some(sum_forces(
                    requested_forces("forces", base.forces)?,
                    requested_forces("forces", correction.forces)?,
                )?)
            } else {
                None
            },
            stress: if options.compute_stress {
                Some(sum_matrix(
                    requested_matrix("stress", base.stress)?,
                    requested_matrix("stress", correction.stress)?,
                ))
            } else {
                None
            },
            virial: if options.compute_virial {
                Some(sum_matrix(
                    requested_matrix("virial", base.virial)?,
                    requested_matrix("virial", correction.virial)?,
                ))
            } else {
                None
            },
        })
    }

    fn shared_neighbor_cutoff(&self) -> Option<f64> {
        match &self.base {
            LlmlipModel::MaceLite(model) if model.has_pinned_mace_runtime_plan() => Some(
                model
                    .cutoff()
                    .max(self.correction.required_neighbor_cutoff()),
            ),
            _ => None,
        }
    }
}

fn requested_scalar(
    name: &'static str,
    value: Option<f64>,
) -> Result<f64, DispersionCorrectedError> {
    value.ok_or(DispersionCorrectedError::MissingRequestedOutput { name })
}

fn requested_forces(
    name: &'static str,
    value: Option<Vec<Vec3>>,
) -> Result<Vec<Vec3>, DispersionCorrectedError> {
    value.ok_or(DispersionCorrectedError::MissingRequestedOutput { name })
}

fn requested_matrix(
    name: &'static str,
    value: Option<Mat3>,
) -> Result<Mat3, DispersionCorrectedError> {
    value.ok_or(DispersionCorrectedError::MissingRequestedOutput { name })
}

fn sum_forces(
    mut base: Vec<Vec3>,
    correction: Vec<Vec3>,
) -> Result<Vec<Vec3>, DispersionCorrectedError> {
    if base.len() != correction.len() {
        return Err(DispersionCorrectedError::ForceLengthMismatch {
            base: base.len(),
            correction: correction.len(),
        });
    }
    for (target, contribution) in base.iter_mut().zip(correction) {
        for axis in 0..3 {
            target[axis] += contribution[axis];
        }
    }
    Ok(base)
}

fn sum_matrix(mut base: Mat3, correction: Mat3) -> Mat3 {
    for row in 0..3 {
        for col in 0..3 {
            base[row][col] += correction[row][col];
        }
    }
    base
}

#[derive(Debug, Error)]
pub enum DispersionCorrectedError {
    #[error("base model prediction failed: {0}")]
    Base(ModelError),

    #[error("dispersion correction prediction failed: {0}")]
    Correction(Dftd3Error),

    #[error("shared neighbor-list construction failed: {0}")]
    SharedNeighborList(NeighborListError),

    #[error("base model or correction did not return requested {name}")]
    MissingRequestedOutput { name: &'static str },

    #[error("base force length {base} does not match correction force length {correction}")]
    ForceLengthMismatch { base: usize, correction: usize },
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::{
        dftd3::{Dftd3BjParameters, Dftd3PairCoefficient},
        geometry::ZERO_VEC3,
        SyntheticRadialModel,
    };
    use e3nn_core::{Activation, DenseLayer, FixedMlp};

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {expected:.16e}, got {actual:.16e}, diff {:.3e}",
            (actual - expected).abs()
        );
    }

    fn assert_vec_close(actual: Vec3, expected: Vec3, tol: f64) {
        for axis in 0..3 {
            assert_close(actual[axis], expected[axis], tol);
        }
    }

    fn synthetic_model() -> LlmlipModel {
        let mlp = FixedMlp::new(vec![DenseLayer::new(
            1,
            1,
            vec![0.25],
            vec![0.05],
            Activation::Identity,
        )
        .unwrap()])
        .unwrap();
        LlmlipModel::SyntheticRadial(SyntheticRadialModel::new(5.0, vec![1.0], 0.7, mlp).unwrap())
    }

    fn dftd3_correction() -> DispersionCorrection {
        DispersionCorrection::Dftd3BjTwoBody(
            Dftd3BjTwoBodyModel::new(
                Dftd3BjParameters::new(1.0, 0.5, 0.4, 2.5, 6.0),
                vec![Dftd3PairCoefficient::new("H", "H", 2.0, 4.0)],
            )
            .unwrap(),
        )
    }

    #[allow(dead_code)]
    fn dftd3_hco_correction() -> DispersionCorrection {
        let mut coefficients = Vec::new();
        for (species_a, species_b) in [
            ("H", "H"),
            ("H", "C"),
            ("H", "O"),
            ("C", "C"),
            ("C", "O"),
            ("O", "O"),
        ] {
            coefficients.push(Dftd3PairCoefficient::new(species_a, species_b, 2.0, 4.0));
        }
        DispersionCorrection::Dftd3BjTwoBody(
            Dftd3BjTwoBodyModel::new(
                Dftd3BjParameters::new(1.0, 0.5, 0.4, 2.5, 8.0),
                coefficients,
            )
            .unwrap(),
        )
    }

    fn h2_structure() -> Structure {
        Structure::new(
            vec!["H".to_owned(), "H".to_owned()],
            vec![[0.0, 0.0, 0.0], [1.2, 0.1, 0.0]],
        )
    }

    #[allow(dead_code)]
    fn hco_structure() -> Structure {
        Structure::new(
            vec!["H".to_owned(), "C".to_owned(), "O".to_owned()],
            vec![[0.0, 0.0, 0.0], [1.2, 0.15, 0.05], [0.25, 1.35, 0.2]],
        )
    }

    #[test]
    fn sums_requested_energy_and_forces() {
        let base = synthetic_model();
        let correction = dftd3_correction();
        let model = DispersionCorrectedModel::new(base.clone(), correction.clone());
        let structure = h2_structure();
        let options = PredictionOptions {
            compute_energy: true,
            compute_forces: true,
            compute_stress: false,
            compute_virial: false,
            ..PredictionOptions::default()
        };

        let base_prediction = base.predict(&structure, &options).unwrap();
        let correction_prediction = correction.predict(&structure, &options).unwrap();
        let combined = model.predict(&structure, &options).unwrap();

        assert_close(
            combined.energy.unwrap(),
            base_prediction.energy.unwrap() + correction_prediction.energy.unwrap(),
            1.0e-14,
        );
        for ((actual, base_force), correction_force) in combined
            .forces
            .unwrap()
            .iter()
            .zip(base_prediction.forces.unwrap())
            .zip(correction_prediction.forces.unwrap())
        {
            let expected = [
                base_force[0] + correction_force[0],
                base_force[1] + correction_force[1],
                base_force[2] + correction_force[2],
            ];
            assert_vec_close(*actual, expected, 1.0e-14);
        }
    }

    #[test]
    fn honors_prediction_options() {
        let model = DispersionCorrectedModel::new(synthetic_model(), dftd3_correction());
        let prediction = model
            .predict(
                &h2_structure(),
                &PredictionOptions {
                    compute_energy: true,
                    compute_forces: false,
                    compute_stress: false,
                    compute_virial: false,
                    ..PredictionOptions::default()
                },
            )
            .unwrap();

        assert!(prediction.energy.is_some());
        assert!(prediction.forces.is_none());
        assert!(prediction.stress.is_none());
        assert!(prediction.virial.is_none());
    }

    #[test]
    fn propagates_base_model_errors() {
        let model = DispersionCorrectedModel::new(synthetic_model(), dftd3_correction());
        let error = model
            .predict(
                &h2_structure(),
                &PredictionOptions {
                    compute_energy: false,
                    compute_forces: false,
                    compute_stress: true,
                    compute_virial: false,
                    ..PredictionOptions::default()
                },
            )
            .expect_err("synthetic base model should reject stress");

        assert!(matches!(error, DispersionCorrectedError::Base(_)));
    }

    #[test]
    fn sum_forces_reports_shape_mismatch() {
        let error = sum_forces(vec![ZERO_VEC3], vec![ZERO_VEC3, ZERO_VEC3])
            .expect_err("force lengths should mismatch");

        assert!(matches!(
            error,
            DispersionCorrectedError::ForceLengthMismatch {
                base: 1,
                correction: 2
            }
        ));
    }
}
