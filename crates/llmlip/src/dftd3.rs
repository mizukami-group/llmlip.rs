//! Rust-native DFTD3(BJ) two-body dispersion correction.
//!
//! This module starts with validated two-body BJ damping kernels. The first
//! model uses explicit per-species-pair `C6`/`C8` coefficients. The second
//! model adds coordination-number-dependent `C6` interpolation with caller
//! supplied element and pair grids. The committed full-table fixture packages
//! the torch-dftd PBE D3(BJ) two-body data for Z=1..94; ATM three-body terms
//! remain unsupported. Cutoff smoothing supports `none` and a cosine taper.

use std::collections::{HashMap, HashSet};

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use thiserror::Error;
use wide::f64x2;

use crate::{
    edge::Edge,
    force_accum::{
        accumulate_edge_displacement_gradient, stress_from_displacement_gradient,
        virial_from_displacement_gradient, ForceAccumError, Mat3,
    },
    geometry::{dot, scale, Cell, MinimumImageError, ZERO_VEC3},
    neighbor::{
        cell_list_multi_image_neighbor_list, single_image_cutoff_limit, EdgeMode, NeighborList,
        NeighborListError,
    },
    structure::{Prediction, PredictionOptions, Structure},
    Vec3,
};

const DIRECTED_EDGE_WEIGHT: f64 = 0.5;
const MIN_FORCE_DISTANCE: f64 = 1.0e-12;
const BOHR_TO_ANGSTROM: f64 = 0.529_177_26;
const D3BJ_DEFAULT_CUTOFF_BOHR: f64 = 95.0;
const C6_INTERPOLATION_CACHE_MAX_ENTRIES: usize = 4_096;

#[derive(Clone, Debug, PartialEq)]
pub struct Dftd3BjParameters {
    pub s6: f64,
    pub s8: f64,
    pub a1: f64,
    pub a2: f64,
    pub cutoff: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Dftd3CutoffSmoothing {
    /// Preserve the pinned torch-dftd two-body D3(BJ) values without an extra cutoff taper.
    #[default]
    None,

    /// Opt-in Rust-local cosine taper; this is not the default torch-dftd smoothing convention.
    Cosine,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dftd3BjMethod {
    /// Grimme D3(BJ) PBE parameter set used by the pinned torch-dftd fixture.
    Pbe,
}

impl Dftd3BjParameters {
    #[must_use]
    pub fn new(s6: f64, s8: f64, a1: f64, a2: f64, cutoff: f64) -> Self {
        Self {
            s6,
            s8,
            a1,
            a2,
            cutoff,
        }
    }

    #[must_use]
    pub fn from_method(method: Dftd3BjMethod, cutoff: f64) -> Self {
        match method {
            Dftd3BjMethod::Pbe => Self::new(1.0, 0.7875, 0.4289, 4.4407 * BOHR_TO_ANGSTROM, cutoff),
        }
    }

    #[must_use]
    pub fn pbe_default_cutoff() -> Self {
        Self::from_method(
            Dftd3BjMethod::Pbe,
            D3BJ_DEFAULT_CUTOFF_BOHR * BOHR_TO_ANGSTROM,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dftd3CoordinationParameters {
    pub k1: f64,
    pub k3: f64,
    pub cutoff: f64,
}

impl Dftd3CoordinationParameters {
    #[must_use]
    pub fn new(k1: f64, k3: f64, cutoff: f64) -> Self {
        Self { k1, k3, cutoff }
    }

    #[must_use]
    pub fn grimme_d3_default(cutoff: f64) -> Self {
        Self {
            k1: 16.0,
            k3: -4.0,
            cutoff,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dftd3PairCoefficient {
    pub species_a: String,
    pub species_b: String,
    pub c6: f64,
    pub c8: f64,
}

impl Dftd3PairCoefficient {
    #[must_use]
    pub fn new(
        species_a: impl Into<String>,
        species_b: impl Into<String>,
        c6: f64,
        c8: f64,
    ) -> Self {
        Self {
            species_a: species_a.into(),
            species_b: species_b.into(),
            c6,
            c8,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dftd3ElementCoefficient {
    pub species: String,
    pub covalent_radius: f64,
    pub r2r4: f64,
}

impl Dftd3ElementCoefficient {
    #[must_use]
    pub fn new(species: impl Into<String>, covalent_radius: f64, r2r4: f64) -> Self {
        Self {
            species: species.into(),
            covalent_radius,
            r2r4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dftd3C6Reference {
    pub c6: f64,
    pub coordination_a: f64,
    pub coordination_b: f64,
}

impl Dftd3C6Reference {
    #[must_use]
    pub fn new(c6: f64, coordination_a: f64, coordination_b: f64) -> Self {
        Self {
            c6,
            coordination_a,
            coordination_b,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dftd3CoordinationPairCoefficient {
    pub species_a: String,
    pub species_b: String,
    pub references: Vec<Dftd3C6Reference>,
}

impl Dftd3CoordinationPairCoefficient {
    #[must_use]
    pub fn new(
        species_a: impl Into<String>,
        species_b: impl Into<String>,
        references: Vec<Dftd3C6Reference>,
    ) -> Self {
        Self {
            species_a: species_a.into(),
            species_b: species_b.into(),
            references,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Dftd3BjTwoBodyModel {
    parameters: Dftd3BjParameters,
    pair_coefficients: HashMap<PairKey, PairCoefficient>,
    cutoff_smoothing: Dftd3CutoffSmoothing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dftd3SupportedTerms {
    /// BJ-damped two-body D3 pair terms with explicit fixed pair coefficients.
    TwoBodyBjPairCoefficientsOnly,
    /// BJ-damped two-body D3 pair terms with coordination-number C6 grids.
    TwoBodyBjCoordinationDependentC6,
}

impl Dftd3BjTwoBodyModel {
    pub const SUPPORTED_TERMS: Dftd3SupportedTerms =
        Dftd3SupportedTerms::TwoBodyBjPairCoefficientsOnly;

    pub fn new(
        parameters: Dftd3BjParameters,
        pair_coefficients: Vec<Dftd3PairCoefficient>,
    ) -> Result<Self, Dftd3Error> {
        validate_parameters(&parameters)?;
        let mut coefficient_map = HashMap::new();
        for coefficient in pair_coefficients {
            validate_species(&coefficient.species_a)?;
            validate_species(&coefficient.species_b)?;
            validate_positive_finite("c6", coefficient.c6)?;
            validate_positive_finite("c8", coefficient.c8)?;
            let key = PairKey::new(&coefficient.species_a, &coefficient.species_b);
            if coefficient_map
                .insert(
                    key,
                    PairCoefficient {
                        c6: coefficient.c6,
                        c8: coefficient.c8,
                    },
                )
                .is_some()
            {
                return Err(Dftd3Error::DuplicatePairCoefficient {
                    species_a: coefficient.species_a,
                    species_b: coefficient.species_b,
                });
            }
        }
        if coefficient_map.is_empty() {
            return Err(Dftd3Error::InvalidParameter {
                name: "pair_coefficients",
                value: 0.0,
            });
        }

        Ok(Self {
            parameters,
            pair_coefficients: coefficient_map,
            cutoff_smoothing: Dftd3CutoffSmoothing::None,
        })
    }

    #[must_use]
    pub fn parameters(&self) -> &Dftd3BjParameters {
        &self.parameters
    }

    #[must_use]
    pub fn cutoff_smoothing(&self) -> Dftd3CutoffSmoothing {
        self.cutoff_smoothing
    }

    #[must_use]
    pub fn with_cutoff_smoothing(mut self, smoothing: Dftd3CutoffSmoothing) -> Self {
        self.cutoff_smoothing = smoothing;
        self
    }

    #[must_use]
    pub fn required_neighbor_cutoff(&self) -> f64 {
        self.parameters.cutoff
    }

    #[must_use]
    pub fn supported_terms(&self) -> Dftd3SupportedTerms {
        Self::SUPPORTED_TERMS
    }

    #[must_use]
    pub fn supports_three_body(&self) -> bool {
        false
    }

    pub fn predict(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
    ) -> Result<Prediction, Dftd3Error> {
        if !options.compute_energy
            && !options.compute_forces
            && !options.compute_stress
            && !options.compute_virial
        {
            return Ok(Prediction::default());
        }

        let evaluation = self.evaluate(structure, options)?;
        Ok(Prediction {
            energy: options.compute_energy.then_some(evaluation.energy),
            forces: options.compute_forces.then_some(evaluation.forces),
            stress: options.compute_stress.then_some(evaluation.stress),
            virial: options.compute_virial.then_some(evaluation.virial),
        })
    }

    pub(crate) fn predict_with_neighbor_list(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
        neighbor_list: &NeighborList,
    ) -> Result<Prediction, Dftd3Error> {
        if !options.compute_energy
            && !options.compute_forces
            && !options.compute_stress
            && !options.compute_virial
        {
            return Ok(Prediction::default());
        }

        let evaluation = self.evaluate_with_neighbor_list(structure, options, neighbor_list)?;
        Ok(Prediction {
            energy: options.compute_energy.then_some(evaluation.energy),
            forces: options.compute_forces.then_some(evaluation.forces),
            stress: options.compute_stress.then_some(evaluation.stress),
            virial: options.compute_virial.then_some(evaluation.virial),
        })
    }

    pub fn energy_and_forces(&self, structure: &Structure) -> Result<(f64, Vec<Vec3>), Dftd3Error> {
        let evaluation = self.evaluate(
            structure,
            &PredictionOptions {
                compute_energy: true,
                compute_forces: true,
                compute_stress: false,
                compute_virial: false,
                ..PredictionOptions::default()
            },
        )?;
        Ok((evaluation.energy, evaluation.forces))
    }

    fn evaluate(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
    ) -> Result<Dftd3Evaluation, Dftd3Error> {
        self.validate_structure(structure)?;
        if let Some(context) = general_single_image_pbc_context(structure, self.parameters.cutoff)?
        {
            let neighbor_list =
                general_single_image_neighbor_list(structure, &context, self.parameters.cutoff)?;
            return self.evaluate_validated_neighbor_list(structure, options, &neighbor_list);
        }
        let neighbor_list = cell_list_multi_image_neighbor_list(structure, self.parameters.cutoff)?;
        self.evaluate_validated_neighbor_list(structure, options, &neighbor_list)
    }

    fn evaluate_with_neighbor_list(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
        neighbor_list: &NeighborList,
    ) -> Result<Dftd3Evaluation, Dftd3Error> {
        self.validate_structure(structure)?;
        let neighbor_list = neighbor_list.filtered_by_cutoff(self.parameters.cutoff)?;
        self.evaluate_validated_neighbor_list(structure, options, &neighbor_list)
    }

    fn validate_structure(&self, structure: &Structure) -> Result<(), Dftd3Error> {
        if structure.species.len() != structure.positions.len() {
            return Err(Dftd3Error::SpeciesPositionMismatch {
                species: structure.species.len(),
                positions: structure.positions.len(),
            });
        }
        for species in &structure.species {
            validate_species(species)?;
        }
        validate_supported_pbc(structure)?;
        Ok(())
    }

    fn evaluate_validated_neighbor_list(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
        neighbor_list: &NeighborList,
    ) -> Result<Dftd3Evaluation, Dftd3Error> {
        let mut energy = 0.0;
        let mut forces = vec![ZERO_VEC3; structure.positions.len()];
        let mut displacement_gradient = [[0.0; 3]; 3];

        for edge in &neighbor_list.edges {
            let coefficient =
                self.coefficient_for(&structure.species[edge.src], &structure.species[edge.dst])?;
            let pair = smooth_d3bj_pair_value(
                self.cutoff_smoothing,
                self.parameters.cutoff,
                edge.distance,
                d3bj_pair_energy_derivative(&self.parameters, coefficient, edge.distance),
            );
            let weighted_energy = DIRECTED_EDGE_WEIGHT * pair.energy;
            energy += weighted_energy;

            if options.compute_forces || options.compute_stress || options.compute_virial {
                let bar_r = if edge.distance > MIN_FORCE_DISTANCE {
                    scale(
                        edge.vector,
                        DIRECTED_EDGE_WEIGHT * pair.d_energy_d_distance / edge.distance,
                    )
                } else {
                    ZERO_VEC3
                };
                if options.compute_forces {
                    crate::force_accum::scatter_edge_vector_vjp(
                        edge.src,
                        edge.dst,
                        bar_r,
                        &mut forces,
                    )?;
                }
                if options.compute_stress || options.compute_virial {
                    accumulate_edge_displacement_gradient(
                        edge.vector,
                        bar_r,
                        &mut displacement_gradient,
                    );
                }
            }
        }

        let stress = if options.compute_stress {
            let cell = structure
                .cell
                .as_ref()
                .ok_or(Dftd3Error::MissingCellForStress)?;
            stress_from_displacement_gradient(displacement_gradient, cell)?
        } else {
            [[0.0; 3]; 3]
        };
        let virial = if options.compute_virial {
            virial_from_displacement_gradient(displacement_gradient)
        } else {
            [[0.0; 3]; 3]
        };

        Ok(Dftd3Evaluation {
            energy,
            forces,
            stress,
            virial,
        })
    }

    fn coefficient_for(
        &self,
        species_a: &str,
        species_b: &str,
    ) -> Result<PairCoefficient, Dftd3Error> {
        self.pair_coefficients
            .get(&PairKey::new(species_a, species_b))
            .copied()
            .ok_or_else(|| Dftd3Error::UnknownPairCoefficient {
                species_a: species_a.to_owned(),
                species_b: species_b.to_owned(),
            })
    }
}

#[derive(Clone, Debug)]
pub struct Dftd3BjCoordinationModel {
    parameters: Dftd3BjParameters,
    coordination: Dftd3CoordinationParameters,
    species: Vec<String>,
    elements: Vec<ElementCoefficient>,
    species_indices: HashMap<String, usize>,
    pair_grids: Vec<C6Grid>,
    pair_grid_indices: Vec<Option<PairGridLookup>>,
    cutoff_smoothing: Dftd3CutoffSmoothing,
}

impl Dftd3BjCoordinationModel {
    pub const SUPPORTED_TERMS: Dftd3SupportedTerms =
        Dftd3SupportedTerms::TwoBodyBjCoordinationDependentC6;

    pub fn new(
        parameters: Dftd3BjParameters,
        coordination: Dftd3CoordinationParameters,
        elements: Vec<Dftd3ElementCoefficient>,
        pair_coefficients: Vec<Dftd3CoordinationPairCoefficient>,
    ) -> Result<Self, Dftd3Error> {
        validate_parameters(&parameters)?;
        validate_coordination_parameters(&coordination)?;

        let mut species = Vec::new();
        let mut elements_by_index = Vec::new();
        let mut species_indices = HashMap::new();
        for element in elements {
            validate_species(&element.species)?;
            validate_positive_finite("covalent_radius", element.covalent_radius)?;
            validate_positive_finite("r2r4", element.r2r4)?;
            if species_indices.contains_key(&element.species) {
                return Err(Dftd3Error::DuplicateElementCoefficient {
                    species: element.species,
                });
            }
            let element_index = elements_by_index.len();
            species_indices.insert(element.species.clone(), element_index);
            species.push(element.species);
            elements_by_index.push(ElementCoefficient {
                covalent_radius: element.covalent_radius,
                r2r4: element.r2r4,
            });
        }
        if elements_by_index.is_empty() {
            return Err(Dftd3Error::InvalidParameter {
                name: "elements",
                value: 0.0,
            });
        }

        let element_count = elements_by_index.len();
        let mut seen_pairs = HashSet::new();
        let mut pair_grids = Vec::new();
        let mut pair_grid_indices = vec![None; element_count * element_count];
        for coefficient in pair_coefficients {
            validate_species(&coefficient.species_a)?;
            validate_species(&coefficient.species_b)?;
            let species_a_index =
                *species_indices.get(&coefficient.species_a).ok_or_else(|| {
                    Dftd3Error::UnknownElementCoefficient {
                        species: coefficient.species_a.clone(),
                    }
                })?;
            let species_b_index =
                *species_indices.get(&coefficient.species_b).ok_or_else(|| {
                    Dftd3Error::UnknownElementCoefficient {
                        species: coefficient.species_b.clone(),
                    }
                })?;
            if coefficient.references.is_empty() {
                return Err(Dftd3Error::InvalidParameter {
                    name: "c6_references",
                    value: 0.0,
                });
            }
            for reference in &coefficient.references {
                validate_positive_finite("c6", reference.c6)?;
                validate_nonnegative_finite("coordination_a", reference.coordination_a)?;
                validate_nonnegative_finite("coordination_b", reference.coordination_b)?;
            }
            let key = PairKey::new(&coefficient.species_a, &coefficient.species_b);
            if !seen_pairs.insert(key) {
                return Err(Dftd3Error::DuplicatePairCoefficient {
                    species_a: coefficient.species_a,
                    species_b: coefficient.species_b,
                });
            }
            let grid_index = pair_grids.len();
            pair_grids.push(C6Grid {
                references: coefficient.references,
            });
            pair_grid_indices[species_a_index * element_count + species_b_index] =
                Some(PairGridLookup {
                    grid_index,
                    swapped: false,
                });
            if species_a_index != species_b_index {
                pair_grid_indices[species_b_index * element_count + species_a_index] =
                    Some(PairGridLookup {
                        grid_index,
                        swapped: true,
                    });
            }
        }
        if pair_grids.is_empty() {
            return Err(Dftd3Error::InvalidParameter {
                name: "pair_coefficients",
                value: 0.0,
            });
        }

        Ok(Self {
            parameters,
            coordination,
            species,
            elements: elements_by_index,
            species_indices,
            pair_grids,
            pair_grid_indices,
            cutoff_smoothing: Dftd3CutoffSmoothing::None,
        })
    }

    #[must_use]
    pub fn parameters(&self) -> &Dftd3BjParameters {
        &self.parameters
    }

    #[must_use]
    pub fn cutoff_smoothing(&self) -> Dftd3CutoffSmoothing {
        self.cutoff_smoothing
    }

    #[must_use]
    pub fn with_cutoff_smoothing(mut self, smoothing: Dftd3CutoffSmoothing) -> Self {
        self.cutoff_smoothing = smoothing;
        self
    }

    #[must_use]
    pub fn required_neighbor_cutoff(&self) -> f64 {
        self.parameters.cutoff.max(self.coordination.cutoff)
    }

    #[must_use]
    pub fn coordination_parameters(&self) -> &Dftd3CoordinationParameters {
        &self.coordination
    }

    #[must_use]
    pub fn supported_terms(&self) -> Dftd3SupportedTerms {
        Self::SUPPORTED_TERMS
    }

    #[must_use]
    pub fn supports_three_body(&self) -> bool {
        false
    }

    pub fn predict(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
    ) -> Result<Prediction, Dftd3Error> {
        if !options.compute_energy
            && !options.compute_forces
            && !options.compute_stress
            && !options.compute_virial
        {
            return Ok(Prediction::default());
        }

        let evaluation = self.evaluate(structure, options)?;
        Ok(Prediction {
            energy: options.compute_energy.then_some(evaluation.energy),
            forces: options.compute_forces.then_some(evaluation.forces),
            stress: options.compute_stress.then_some(evaluation.stress),
            virial: options.compute_virial.then_some(evaluation.virial),
        })
    }

    pub(crate) fn predict_with_neighbor_list(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
        neighbor_list: &NeighborList,
    ) -> Result<Prediction, Dftd3Error> {
        if !options.compute_energy
            && !options.compute_forces
            && !options.compute_stress
            && !options.compute_virial
        {
            return Ok(Prediction::default());
        }

        let evaluation = self.evaluate_with_neighbor_list(structure, options, neighbor_list)?;
        Ok(Prediction {
            energy: options.compute_energy.then_some(evaluation.energy),
            forces: options.compute_forces.then_some(evaluation.forces),
            stress: options.compute_stress.then_some(evaluation.stress),
            virial: options.compute_virial.then_some(evaluation.virial),
        })
    }

    pub fn energy_and_forces(&self, structure: &Structure) -> Result<(f64, Vec<Vec3>), Dftd3Error> {
        let evaluation = self.evaluate(
            structure,
            &PredictionOptions {
                compute_energy: true,
                compute_forces: true,
                compute_stress: false,
                compute_virial: false,
                ..PredictionOptions::default()
            },
        )?;
        Ok((evaluation.energy, evaluation.forces))
    }

    fn evaluate(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
    ) -> Result<Dftd3Evaluation, Dftd3Error> {
        validate_supported_pbc(structure)?;
        let atom_species = self.structure_species_indices(structure)?;
        let neighbor_cutoff = self.required_neighbor_cutoff();
        if let Some(context) = orthorhombic_single_image_pbc_context(structure, neighbor_cutoff)? {
            return self.evaluate_orthorhombic_single_image_pbc(
                structure,
                options,
                &atom_species,
                context,
            );
        }
        if let Some(context) = general_single_image_pbc_context(structure, neighbor_cutoff)? {
            #[cfg(feature = "parallel")]
            if rayon::current_num_threads() > 1 && structure.positions.len() >= 512 {
                return self.evaluate_general_single_image_pbc_streaming_parallel(
                    structure,
                    options,
                    &atom_species,
                    &context,
                );
            }

            let edges = collect_general_single_image_edges(structure, &context, neighbor_cutoff)?;
            return self.evaluate_single_image_pbc_edges(structure, options, &atom_species, &edges);
        }
        if let Some(context) = orthorhombic_multi_image_pbc_context(structure, neighbor_cutoff)? {
            return self.evaluate_orthorhombic_multi_image_pbc(
                structure,
                options,
                &atom_species,
                context,
            );
        }
        let neighbor_list = cell_list_multi_image_neighbor_list(structure, neighbor_cutoff)?;
        self.evaluate_validated_neighbor_list(structure, options, &atom_species, &neighbor_list)
    }

    fn evaluate_with_neighbor_list(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
        neighbor_list: &NeighborList,
    ) -> Result<Dftd3Evaluation, Dftd3Error> {
        validate_supported_pbc(structure)?;
        let atom_species = self.structure_species_indices(structure)?;
        let neighbor_list = neighbor_list.filtered_by_cutoff(self.required_neighbor_cutoff())?;
        self.evaluate_validated_neighbor_list(structure, options, &atom_species, &neighbor_list)
    }

    fn evaluate_validated_neighbor_list(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
        atom_species: &[usize],
        neighbor_list: &NeighborList,
    ) -> Result<Dftd3Evaluation, Dftd3Error> {
        let coordination_numbers = self.coordination_numbers(atom_species, neighbor_list);
        let mut coordination_bars = vec![0.0; structure.positions.len()];
        let mut energy = 0.0;
        let mut forces = vec![ZERO_VEC3; structure.positions.len()];
        let mut displacement_gradient = [[0.0; 3]; 3];
        let need_vjp = options.compute_forces || options.compute_stress || options.compute_virial;

        let mut c6_cache = C6InterpolationCache::new();
        for edge in &neighbor_list.edges {
            if edge.distance >= self.parameters.cutoff {
                continue;
            }
            let species_src = atom_species[edge.src];
            let species_dst = atom_species[edge.dst];
            let c6 = self.c6_for_cached(
                species_src,
                species_dst,
                coordination_numbers[edge.src],
                coordination_numbers[edge.dst],
                &mut c6_cache,
            )?;
            let r2r4_src = self.elements[species_src].r2r4;
            let r2r4_dst = self.elements[species_dst].r2r4;
            let value = smooth_d3bj_coordination_pair_value(
                self.cutoff_smoothing,
                self.parameters.cutoff,
                edge.distance,
                d3bj_coordination_pair_energy_derivative(
                    &self.parameters,
                    c6.c6,
                    3.0 * r2r4_src * r2r4_dst,
                    edge.distance,
                ),
            );
            let weighted_energy = DIRECTED_EDGE_WEIGHT * value.energy;
            energy += weighted_energy;

            if need_vjp {
                if edge.distance > MIN_FORCE_DISTANCE {
                    let bar_r = scale(
                        edge.vector,
                        DIRECTED_EDGE_WEIGHT * value.d_energy_d_distance / edge.distance,
                    );
                    if options.compute_forces {
                        crate::force_accum::scatter_edge_vector_vjp(
                            edge.src,
                            edge.dst,
                            bar_r,
                            &mut forces,
                        )?;
                    }
                    if options.compute_stress || options.compute_virial {
                        accumulate_edge_displacement_gradient(
                            edge.vector,
                            bar_r,
                            &mut displacement_gradient,
                        );
                    }
                }
                let weighted_bar_c6 = DIRECTED_EDGE_WEIGHT * value.d_energy_d_c6;
                coordination_bars[edge.src] += weighted_bar_c6 * c6.d_c6_d_coordination_i;
                coordination_bars[edge.dst] += weighted_bar_c6 * c6.d_c6_d_coordination_j;
            }
        }

        if need_vjp {
            self.accumulate_coordination_vjp(
                atom_species,
                neighbor_list,
                &coordination_bars,
                options,
                &mut forces,
                &mut displacement_gradient,
            )?;
        }

        let stress = if options.compute_stress {
            let cell = structure
                .cell
                .as_ref()
                .ok_or(Dftd3Error::MissingCellForStress)?;
            stress_from_displacement_gradient(displacement_gradient, cell)?
        } else {
            [[0.0; 3]; 3]
        };
        let virial = if options.compute_virial {
            virial_from_displacement_gradient(displacement_gradient)
        } else {
            [[0.0; 3]; 3]
        };

        Ok(Dftd3Evaluation {
            energy,
            forces,
            stress,
            virial,
        })
    }

    fn evaluate_orthorhombic_single_image_pbc(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
        atom_species: &[usize],
        context: OrthorhombicSingleImagePbc,
    ) -> Result<Dftd3Evaluation, Dftd3Error> {
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && structure.positions.len() >= 512 {
            return self.evaluate_orthorhombic_single_image_pbc_streaming_parallel(
                structure,
                options,
                atom_species,
                context,
            );
        }

        let edges = collect_orthorhombic_single_image_edges(
            structure,
            context,
            self.required_neighbor_cutoff() * self.required_neighbor_cutoff(),
        );
        self.evaluate_single_image_pbc_edges(structure, options, atom_species, &edges)
    }

    #[cfg(feature = "parallel")]
    fn evaluate_orthorhombic_single_image_pbc_streaming_parallel(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
        atom_species: &[usize],
        context: OrthorhombicSingleImagePbc,
    ) -> Result<Dftd3Evaluation, Dftd3Error> {
        let atom_count = structure.positions.len();
        let axes = OrthorhombicPositionAxes::from_positions(&structure.positions, context.lengths);
        let atom_covalent_radii = self.atom_covalent_radii(atom_species);
        let atom_r2r4 = self.atom_r2r4(atom_species);
        let neighbor_cutoff_squared =
            self.required_neighbor_cutoff() * self.required_neighbor_cutoff();
        let ranges = parallel_unordered_pair_source_ranges(atom_count);
        let range_capacity =
            orthorhombic_edge_capacity_hint(atom_count, context, neighbor_cutoff_squared)
                .checked_div(ranges.len().max(1))
                .unwrap_or(0)
                .saturating_add(1);
        // The hint already estimates unordered stored edges. Each pass rebuilds
        // only its range-local buffer deliberately; retaining all per-range
        // edges across CN, energy, and CN-VJP raises peak memory on large PBC
        // structures more than the repeated enumeration costs.
        let need_vjp = options.compute_forces || options.compute_stress || options.compute_virial;

        let coordination_numbers = ranges
            .par_iter()
            .map(|&(start, end)| {
                let mut local = vec![0.0; atom_count];
                let mut edges = Vec::with_capacity(range_capacity);
                collect_orthorhombic_edges_for_source_range(
                    &axes,
                    context,
                    neighbor_cutoff_squared,
                    start,
                    end,
                    &mut edges,
                );
                for edge in &edges {
                    if edge.distance >= self.coordination.cutoff {
                        continue;
                    }
                    let src = edge.src;
                    let dst = edge.dst;
                    let value = coordination_value(
                        self.coordination.k1,
                        atom_covalent_radii[src] + atom_covalent_radii[dst],
                        edge.distance,
                    )
                    .value;
                    local[src] += value;
                    local[dst] += value;
                }
                local
            })
            .reduce(
                || vec![0.0; atom_count],
                |mut left, right| {
                    for (target, source) in left.iter_mut().zip(right) {
                        *target += source;
                    }
                    left
                },
            );

        let mut accum = ranges
            .par_iter()
            .map(|&(start, end)| {
                let mut local = Dftd3OrthorhombicAccum::new(atom_count);
                let mut c6_cache = C6InterpolationCache::new();
                let mut force_scatter = if options.compute_forces {
                    Some(SourceBufferedForceScatter::new(&mut local.forces))
                } else {
                    None
                };
                let mut edges = Vec::with_capacity(range_capacity);
                collect_orthorhombic_edges_for_source_range(
                    &axes,
                    context,
                    neighbor_cutoff_squared,
                    start,
                    end,
                    &mut edges,
                );
                for edge in &edges {
                    if edge.distance >= self.parameters.cutoff {
                        continue;
                    }
                    let src = edge.src;
                    let dst = edge.dst;
                    let species_src = atom_species[src];
                    let species_dst = atom_species[dst];
                    let c6 = self.c6_for_cached(
                        species_src,
                        species_dst,
                        coordination_numbers[src],
                        coordination_numbers[dst],
                        &mut c6_cache,
                    )?;
                    let value = smooth_d3bj_coordination_pair_value(
                        self.cutoff_smoothing,
                        self.parameters.cutoff,
                        edge.distance,
                        d3bj_coordination_pair_energy_derivative(
                            &self.parameters,
                            c6.c6,
                            3.0 * atom_r2r4[src] * atom_r2r4[dst],
                            edge.distance,
                        ),
                    );
                    local.energy += value.energy;

                    if need_vjp {
                        if edge.distance > MIN_FORCE_DISTANCE {
                            let bar_r =
                                scale(edge.vector, value.d_energy_d_distance / edge.distance);
                            if let Some(scatter) = force_scatter.as_mut() {
                                scatter.scatter(src, dst, bar_r);
                            }
                            if options.compute_stress || options.compute_virial {
                                accumulate_edge_displacement_gradient(
                                    edge.vector,
                                    bar_r,
                                    &mut local.displacement_gradient,
                                );
                            }
                        }
                        let weighted_bar_c6 = DIRECTED_EDGE_WEIGHT * value.d_energy_d_c6;
                        local.coordination_bars[src] +=
                            2.0 * weighted_bar_c6 * c6.d_c6_d_coordination_i;
                        local.coordination_bars[dst] +=
                            2.0 * weighted_bar_c6 * c6.d_c6_d_coordination_j;
                    }
                }
                if let Some(scatter) = force_scatter {
                    scatter.finish();
                }
                Ok::<Dftd3OrthorhombicAccum, Dftd3Error>(local)
            })
            .try_reduce(
                || Dftd3OrthorhombicAccum::new(atom_count),
                |mut left, right| {
                    left.merge(right);
                    Ok::<Dftd3OrthorhombicAccum, Dftd3Error>(left)
                },
            )?;

        if need_vjp {
            let vjp_accum = ranges
                .par_iter()
                .map(|&(start, end)| {
                    let mut local = Dftd3OrthorhombicAccum::new(atom_count);
                    let mut force_scatter = if options.compute_forces {
                        Some(SourceBufferedForceScatter::new(&mut local.forces))
                    } else {
                        None
                    };
                    let mut edges = Vec::with_capacity(range_capacity);
                    collect_orthorhombic_edges_for_source_range(
                        &axes,
                        context,
                        neighbor_cutoff_squared,
                        start,
                        end,
                        &mut edges,
                    );
                    for edge in &edges {
                        if edge.distance >= self.coordination.cutoff {
                            continue;
                        }
                        let src = edge.src;
                        let dst = edge.dst;
                        let bar_coordination =
                            accum.coordination_bars[src] + accum.coordination_bars[dst];
                        if bar_coordination == 0.0 || edge.distance <= MIN_FORCE_DISTANCE {
                            continue;
                        }
                        let value = coordination_value(
                            self.coordination.k1,
                            atom_covalent_radii[src] + atom_covalent_radii[dst],
                            edge.distance,
                        );
                        let bar_r = scale(
                            edge.vector,
                            bar_coordination * value.d_value_d_distance / edge.distance,
                        );
                        if let Some(scatter) = force_scatter.as_mut() {
                            scatter.scatter(src, dst, bar_r);
                        }
                        if options.compute_stress || options.compute_virial {
                            accumulate_edge_displacement_gradient(
                                edge.vector,
                                bar_r,
                                &mut local.displacement_gradient,
                            );
                        }
                    }
                    if let Some(scatter) = force_scatter {
                        scatter.finish();
                    }
                    local
                })
                .reduce(
                    || Dftd3OrthorhombicAccum::new(atom_count),
                    |mut left, right| {
                        left.merge_vjp(right);
                        left
                    },
                );
            accum.merge_vjp(vjp_accum);
        }

        self.finish_orthorhombic_accum(structure, options, accum)
    }

    #[cfg(feature = "parallel")]
    fn evaluate_general_single_image_pbc_streaming_parallel(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
        atom_species: &[usize],
        context: &GeneralSingleImagePbc,
    ) -> Result<Dftd3Evaluation, Dftd3Error> {
        let atom_count = structure.positions.len();
        let atom_covalent_radii = self.atom_covalent_radii(atom_species);
        let atom_r2r4 = self.atom_r2r4(atom_species);
        let neighbor_cutoff = self.required_neighbor_cutoff();
        let neighbor_cutoff_squared = neighbor_cutoff * neighbor_cutoff;
        let ranges = parallel_unordered_pair_source_ranges(atom_count);
        let range_capacity =
            general_single_image_edge_capacity_hint(atom_count, context, neighbor_cutoff)
                .checked_div(2 * ranges.len().max(1))
                .unwrap_or(0)
                .saturating_add(1);
        // The general-cell hint is directed, while the streaming stored-edge
        // passes below use unordered pairs. Like the orthorhombic path, each
        // pass rebuilds only a range-local buffer to avoid materializing the
        // full edge list for CN, energy, and CN-VJP.
        let need_vjp = options.compute_forces || options.compute_stress || options.compute_virial;

        let coordination_numbers = ranges
            .par_iter()
            .map(|&(start, end)| {
                let mut local = vec![0.0; atom_count];
                let mut edges = Vec::with_capacity(range_capacity);
                collect_general_single_image_stored_edges_for_source_range(
                    context,
                    neighbor_cutoff_squared,
                    start,
                    end,
                    &mut edges,
                )?;
                for edge in &edges {
                    if edge.distance >= self.coordination.cutoff {
                        continue;
                    }
                    let src = edge.src;
                    let dst = edge.dst;
                    let value = coordination_value(
                        self.coordination.k1,
                        atom_covalent_radii[src] + atom_covalent_radii[dst],
                        edge.distance,
                    )
                    .value;
                    local[src] += value;
                    local[dst] += value;
                }
                Ok::<Vec<f64>, Dftd3Error>(local)
            })
            .try_reduce(
                || vec![0.0; atom_count],
                |mut left, right| {
                    for (target, source) in left.iter_mut().zip(right) {
                        *target += source;
                    }
                    Ok::<Vec<f64>, Dftd3Error>(left)
                },
            )?;

        let mut accum = ranges
            .par_iter()
            .map(|&(start, end)| {
                let mut local = Dftd3OrthorhombicAccum::new(atom_count);
                let mut c6_cache = C6InterpolationCache::new();
                let mut force_scatter = if options.compute_forces {
                    Some(SourceBufferedForceScatter::new(&mut local.forces))
                } else {
                    None
                };
                let mut edges = Vec::with_capacity(range_capacity);
                collect_general_single_image_stored_edges_for_source_range(
                    context,
                    neighbor_cutoff_squared,
                    start,
                    end,
                    &mut edges,
                )?;
                for edge in &edges {
                    if edge.distance >= self.parameters.cutoff {
                        continue;
                    }
                    let src = edge.src;
                    let dst = edge.dst;
                    let species_src = atom_species[src];
                    let species_dst = atom_species[dst];
                    let c6 = self.c6_for_cached(
                        species_src,
                        species_dst,
                        coordination_numbers[src],
                        coordination_numbers[dst],
                        &mut c6_cache,
                    )?;
                    let value = smooth_d3bj_coordination_pair_value(
                        self.cutoff_smoothing,
                        self.parameters.cutoff,
                        edge.distance,
                        d3bj_coordination_pair_energy_derivative(
                            &self.parameters,
                            c6.c6,
                            3.0 * atom_r2r4[src] * atom_r2r4[dst],
                            edge.distance,
                        ),
                    );
                    local.energy += value.energy;

                    if need_vjp {
                        if edge.distance > MIN_FORCE_DISTANCE {
                            let bar_r =
                                scale(edge.vector, value.d_energy_d_distance / edge.distance);
                            if let Some(scatter) = force_scatter.as_mut() {
                                scatter.scatter(src, dst, bar_r);
                            }
                            if options.compute_stress || options.compute_virial {
                                accumulate_edge_displacement_gradient(
                                    edge.vector,
                                    bar_r,
                                    &mut local.displacement_gradient,
                                );
                            }
                        }
                        let weighted_bar_c6 = DIRECTED_EDGE_WEIGHT * value.d_energy_d_c6;
                        local.coordination_bars[src] +=
                            2.0 * weighted_bar_c6 * c6.d_c6_d_coordination_i;
                        local.coordination_bars[dst] +=
                            2.0 * weighted_bar_c6 * c6.d_c6_d_coordination_j;
                    }
                }
                if let Some(scatter) = force_scatter {
                    scatter.finish();
                }
                Ok::<Dftd3OrthorhombicAccum, Dftd3Error>(local)
            })
            .try_reduce(
                || Dftd3OrthorhombicAccum::new(atom_count),
                |mut left, right| {
                    left.merge(right);
                    Ok::<Dftd3OrthorhombicAccum, Dftd3Error>(left)
                },
            )?;

        if need_vjp {
            let vjp_accum = ranges
                .par_iter()
                .map(|&(start, end)| {
                    let mut local = Dftd3OrthorhombicAccum::new(atom_count);
                    let mut force_scatter = if options.compute_forces {
                        Some(SourceBufferedForceScatter::new(&mut local.forces))
                    } else {
                        None
                    };
                    let mut edges = Vec::with_capacity(range_capacity);
                    collect_general_single_image_stored_edges_for_source_range(
                        context,
                        neighbor_cutoff_squared,
                        start,
                        end,
                        &mut edges,
                    )?;
                    for edge in &edges {
                        if edge.distance >= self.coordination.cutoff {
                            continue;
                        }
                        let src = edge.src;
                        let dst = edge.dst;
                        let bar_coordination =
                            accum.coordination_bars[src] + accum.coordination_bars[dst];
                        if bar_coordination == 0.0 || edge.distance <= MIN_FORCE_DISTANCE {
                            continue;
                        }
                        let value = coordination_value(
                            self.coordination.k1,
                            atom_covalent_radii[src] + atom_covalent_radii[dst],
                            edge.distance,
                        );
                        let bar_r = scale(
                            edge.vector,
                            bar_coordination * value.d_value_d_distance / edge.distance,
                        );
                        if let Some(scatter) = force_scatter.as_mut() {
                            scatter.scatter(src, dst, bar_r);
                        }
                        if options.compute_stress || options.compute_virial {
                            accumulate_edge_displacement_gradient(
                                edge.vector,
                                bar_r,
                                &mut local.displacement_gradient,
                            );
                        }
                    }
                    if let Some(scatter) = force_scatter {
                        scatter.finish();
                    }
                    Ok::<Dftd3OrthorhombicAccum, Dftd3Error>(local)
                })
                .try_reduce(
                    || Dftd3OrthorhombicAccum::new(atom_count),
                    |mut left, right| {
                        left.merge_vjp(right);
                        Ok::<Dftd3OrthorhombicAccum, Dftd3Error>(left)
                    },
                )?;
            accum.merge_vjp(vjp_accum);
        }

        self.finish_orthorhombic_accum(structure, options, accum)
    }

    fn evaluate_single_image_pbc_edges(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
        atom_species: &[usize],
        edges: &[SingleImageStoredEdge],
    ) -> Result<Dftd3Evaluation, Dftd3Error> {
        let atom_covalent_radii = self.atom_covalent_radii(atom_species);
        let atom_r2r4 = self.atom_r2r4(atom_species);
        let coordination_values =
            self.coordination_values_orthorhombic_edges(&atom_covalent_radii, edges);
        let coordination_numbers = &coordination_values.numbers;
        let mut coordination_bars = vec![0.0; structure.positions.len()];
        let mut energy = 0.0;
        let mut forces = vec![ZERO_VEC3; structure.positions.len()];
        let mut displacement_gradient = [[0.0; 3]; 3];
        let need_vjp = options.compute_forces || options.compute_stress || options.compute_virial;
        let mut c6_cache = C6InterpolationCache::new();
        let mut force_scatter = if options.compute_forces {
            Some(SourceBufferedForceScatter::new(&mut forces))
        } else {
            None
        };

        for edge in edges {
            if edge.distance >= self.parameters.cutoff {
                continue;
            }
            let src = edge.src;
            let dst = edge.dst;
            let species_src = atom_species[src];
            let species_dst = atom_species[dst];
            let c6 = self.c6_for_cached(
                species_src,
                species_dst,
                coordination_numbers[src],
                coordination_numbers[dst],
                &mut c6_cache,
            )?;
            let value = smooth_d3bj_coordination_pair_value(
                self.cutoff_smoothing,
                self.parameters.cutoff,
                edge.distance,
                d3bj_coordination_pair_energy_derivative(
                    &self.parameters,
                    c6.c6,
                    3.0 * atom_r2r4[src] * atom_r2r4[dst],
                    edge.distance,
                ),
            );
            energy += value.energy;

            if need_vjp {
                if edge.distance > MIN_FORCE_DISTANCE {
                    let bar_r = scale(edge.vector, value.d_energy_d_distance / edge.distance);
                    if let Some(scatter) = force_scatter.as_mut() {
                        scatter.scatter(src, dst, bar_r);
                    }
                    if options.compute_stress || options.compute_virial {
                        accumulate_edge_displacement_gradient(
                            edge.vector,
                            bar_r,
                            &mut displacement_gradient,
                        );
                    }
                }
                let weighted_bar_c6 = DIRECTED_EDGE_WEIGHT * value.d_energy_d_c6;
                coordination_bars[src] += 2.0 * weighted_bar_c6 * c6.d_c6_d_coordination_i;
                coordination_bars[dst] += 2.0 * weighted_bar_c6 * c6.d_c6_d_coordination_j;
            }
        }
        if let Some(scatter) = force_scatter {
            scatter.finish();
        }

        if need_vjp {
            self.accumulate_coordination_vjp_orthorhombic_edges(
                edges,
                &coordination_values.d_value_d_distances,
                &coordination_bars,
                options,
                &mut forces,
                &mut displacement_gradient,
            )?;
        }

        let stress = if options.compute_stress {
            let cell = structure
                .cell
                .as_ref()
                .ok_or(Dftd3Error::MissingCellForStress)?;
            stress_from_displacement_gradient(displacement_gradient, cell)?
        } else {
            [[0.0; 3]; 3]
        };
        let virial = if options.compute_virial {
            virial_from_displacement_gradient(displacement_gradient)
        } else {
            [[0.0; 3]; 3]
        };

        Ok(Dftd3Evaluation {
            energy,
            forces,
            stress,
            virial,
        })
    }

    fn evaluate_orthorhombic_multi_image_pbc(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
        atom_species: &[usize],
        context: OrthorhombicMultiImagePbc,
    ) -> Result<Dftd3Evaluation, Dftd3Error> {
        #[cfg(feature = "parallel")]
        if rayon::current_num_threads() > 1 && structure.positions.len() >= 64 {
            return self.evaluate_orthorhombic_multi_image_pbc_parallel(
                structure,
                options,
                atom_species,
                context,
            );
        }

        let atom_count = structure.positions.len();
        let axes = OrthorhombicPositionAxes::from_positions(&structure.positions, context.lengths);
        let atom_covalent_radii = self.atom_covalent_radii(atom_species);
        let atom_r2r4 = self.atom_r2r4(atom_species);
        let coordination_edges =
            collect_orthorhombic_multi_image_edges(&axes, context, self.coordination.cutoff);
        let coordination_values =
            self.coordination_values_orthorhombic_edges(&atom_covalent_radii, &coordination_edges);
        let coordination_numbers = &coordination_values.numbers;
        let need_vjp = options.compute_forces || options.compute_stress || options.compute_virial;
        let mut accum = Dftd3OrthorhombicAccum::new(atom_count);

        self.accumulate_orthorhombic_multi_image_pair_range(
            &axes,
            context,
            0,
            atom_count,
            options,
            atom_species,
            &atom_r2r4,
            coordination_numbers,
            need_vjp,
            &mut accum,
        )?;

        if need_vjp {
            self.accumulate_coordination_vjp_orthorhombic_edges(
                &coordination_edges,
                &coordination_values.d_value_d_distances,
                &accum.coordination_bars,
                options,
                &mut accum.forces,
                &mut accum.displacement_gradient,
            )?;
        }

        self.finish_orthorhombic_accum(structure, options, accum)
    }

    #[cfg(feature = "parallel")]
    fn evaluate_orthorhombic_multi_image_pbc_parallel(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
        atom_species: &[usize],
        context: OrthorhombicMultiImagePbc,
    ) -> Result<Dftd3Evaluation, Dftd3Error> {
        let atom_count = structure.positions.len();
        let axes = OrthorhombicPositionAxes::from_positions(&structure.positions, context.lengths);
        let atom_covalent_radii = self.atom_covalent_radii(atom_species);
        let atom_r2r4 = self.atom_r2r4(atom_species);
        let coordination_edges = collect_orthorhombic_multi_image_edges_parallel(
            &axes,
            context,
            self.coordination.cutoff,
        );
        let coordination_values = self.coordination_values_orthorhombic_edges_parallel(
            &atom_covalent_radii,
            &coordination_edges,
        );
        let coordination_numbers = &coordination_values.numbers;
        let need_vjp = options.compute_forces || options.compute_stress || options.compute_virial;
        let ranges = parallel_unordered_pair_source_ranges(atom_count);

        let mut accum = ranges
            .par_iter()
            .map(|&(start, end)| {
                let mut local = Dftd3OrthorhombicAccum::new(atom_count);
                self.accumulate_orthorhombic_multi_image_pair_range(
                    &axes,
                    context,
                    start,
                    end,
                    options,
                    atom_species,
                    &atom_r2r4,
                    coordination_numbers,
                    need_vjp,
                    &mut local,
                )?;
                Ok::<Dftd3OrthorhombicAccum, Dftd3Error>(local)
            })
            .try_reduce(
                || Dftd3OrthorhombicAccum::new(atom_count),
                |mut left, right| {
                    left.merge(right);
                    Ok::<Dftd3OrthorhombicAccum, Dftd3Error>(left)
                },
            )?;

        if need_vjp {
            let vjp_ranges = parallel_edge_work_ranges(coordination_edges.len());
            let vjp_accum = vjp_ranges
                .par_iter()
                .map(|&(start, end)| {
                    let mut local = Dftd3OrthorhombicAccum::new(atom_count);
                    let mut force_scatter = if options.compute_forces {
                        Some(SourceBufferedForceScatter::new(&mut local.forces))
                    } else {
                        None
                    };
                    for (edge_index, edge) in
                        coordination_edges.iter().enumerate().take(end).skip(start)
                    {
                        let d_value_d_distance =
                            coordination_values.d_value_d_distances[edge_index];
                        if d_value_d_distance == 0.0 {
                            continue;
                        }
                        let src = edge.src;
                        let dst = edge.dst;
                        let bar_coordination =
                            accum.coordination_bars[src] + accum.coordination_bars[dst];
                        if bar_coordination == 0.0 || edge.distance <= MIN_FORCE_DISTANCE {
                            continue;
                        }
                        let bar_r = scale(
                            edge.vector,
                            bar_coordination * d_value_d_distance / edge.distance,
                        );
                        if let Some(scatter) = force_scatter.as_mut() {
                            scatter.scatter(src, dst, bar_r);
                        }
                        if options.compute_stress || options.compute_virial {
                            accumulate_edge_displacement_gradient(
                                edge.vector,
                                bar_r,
                                &mut local.displacement_gradient,
                            );
                        }
                    }
                    if let Some(scatter) = force_scatter {
                        scatter.finish();
                    }
                    local
                })
                .reduce(
                    || Dftd3OrthorhombicAccum::new(atom_count),
                    |mut left, right| {
                        left.merge_vjp(right);
                        left
                    },
                );
            accum.merge_vjp(vjp_accum);
        }

        self.finish_orthorhombic_accum(structure, options, accum)
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_orthorhombic_multi_image_pair_range(
        &self,
        axes: &OrthorhombicPositionAxes,
        context: OrthorhombicMultiImagePbc,
        start: usize,
        end: usize,
        options: &PredictionOptions,
        atom_species: &[usize],
        atom_r2r4: &[f64],
        coordination_numbers: &[f64],
        need_vjp: bool,
        accum: &mut Dftd3OrthorhombicAccum,
    ) -> Result<(), Dftd3Error> {
        let cutoff = self.parameters.cutoff;
        let cutoff_squared = cutoff * cutoff;
        let mut c6_cache = C6InterpolationCache::new();
        let energy = &mut accum.energy;
        let coordination_bars = &mut accum.coordination_bars;
        let displacement_gradient = &mut accum.displacement_gradient;
        let mut force_scatter = if options.compute_forces {
            Some(SourceBufferedForceScatter::new(&mut accum.forces))
        } else {
            None
        };

        for src in start..end {
            let src_x = axes.x[src];
            let src_y = axes.y[src];
            let src_z = axes.z[src];
            self.accumulate_orthorhombic_multi_image_self_pairs(
                src,
                cutoff,
                cutoff_squared,
                options,
                atom_species,
                atom_r2r4,
                coordination_numbers,
                need_vjp,
                energy,
                coordination_bars,
                displacement_gradient,
                &mut c6_cache,
                &mut force_scatter,
                context,
            )?;
            for dst in src + 1..axes.x.len() {
                let delta = [
                    axes.x[dst] - src_x,
                    axes.y[dst] - src_y,
                    axes.z[dst] - src_z,
                ];
                self.accumulate_orthorhombic_multi_image_cross_pairs(
                    src,
                    dst,
                    delta,
                    cutoff,
                    cutoff_squared,
                    options,
                    atom_species,
                    atom_r2r4,
                    coordination_numbers,
                    need_vjp,
                    energy,
                    coordination_bars,
                    displacement_gradient,
                    &mut c6_cache,
                    &mut force_scatter,
                    context,
                )?;
            }
        }
        if let Some(scatter) = force_scatter {
            scatter.finish();
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_orthorhombic_multi_image_self_pairs(
        &self,
        atom: usize,
        cutoff: f64,
        cutoff_squared: f64,
        options: &PredictionOptions,
        atom_species: &[usize],
        atom_r2r4: &[f64],
        coordination_numbers: &[f64],
        need_vjp: bool,
        energy: &mut f64,
        coordination_bars: &mut [f64],
        displacement_gradient: &mut Mat3,
        c6_cache: &mut C6InterpolationCache,
        force_scatter: &mut Option<SourceBufferedForceScatter<'_>>,
        context: OrthorhombicMultiImagePbc,
    ) -> Result<(), Dftd3Error> {
        let zero = [0.0, 0.0, 0.0];
        let [range_x, range_y, range_z] =
            orthorhombic_multi_image_shift_ranges(zero, context, cutoff);
        for shift_x in range_x.0..=range_x.1 {
            let x = f64::from(shift_x) * context.lengths[0];
            let x2 = x * x;
            if x2 >= cutoff_squared {
                continue;
            }
            for shift_y in range_y.0..=range_y.1 {
                let y = f64::from(shift_y) * context.lengths[1];
                let xy2 = x2 + y * y;
                if xy2 >= cutoff_squared {
                    continue;
                }
                for shift_z in range_z.0..=range_z.1 {
                    if !is_positive_periodic_shift([shift_x, shift_y, shift_z]) {
                        continue;
                    }
                    let z = f64::from(shift_z) * context.lengths[2];
                    let distance_squared = xy2 + z * z;
                    if distance_squared >= cutoff_squared {
                        continue;
                    }
                    self.accumulate_orthorhombic_multi_image_pair(
                        atom,
                        atom,
                        [x, y, z],
                        distance_squared.sqrt(),
                        options,
                        atom_species,
                        atom_r2r4,
                        coordination_numbers,
                        need_vjp,
                        energy,
                        coordination_bars,
                        displacement_gradient,
                        c6_cache,
                        force_scatter,
                    )?;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_orthorhombic_multi_image_cross_pairs(
        &self,
        src: usize,
        dst: usize,
        delta: Vec3,
        cutoff: f64,
        cutoff_squared: f64,
        options: &PredictionOptions,
        atom_species: &[usize],
        atom_r2r4: &[f64],
        coordination_numbers: &[f64],
        need_vjp: bool,
        energy: &mut f64,
        coordination_bars: &mut [f64],
        displacement_gradient: &mut Mat3,
        c6_cache: &mut C6InterpolationCache,
        force_scatter: &mut Option<SourceBufferedForceScatter<'_>>,
        context: OrthorhombicMultiImagePbc,
    ) -> Result<(), Dftd3Error> {
        let [range_x, range_y, range_z] =
            orthorhombic_multi_image_shift_ranges(delta, context, cutoff);
        for shift_x in range_x.0..=range_x.1 {
            let x = delta[0] + f64::from(shift_x) * context.lengths[0];
            let x2 = x * x;
            if x2 >= cutoff_squared {
                continue;
            }
            for shift_y in range_y.0..=range_y.1 {
                let y = delta[1] + f64::from(shift_y) * context.lengths[1];
                let xy2 = x2 + y * y;
                if xy2 >= cutoff_squared {
                    continue;
                }
                for shift_z in range_z.0..=range_z.1 {
                    let z = delta[2] + f64::from(shift_z) * context.lengths[2];
                    let distance_squared = xy2 + z * z;
                    if distance_squared >= cutoff_squared {
                        continue;
                    }
                    self.accumulate_orthorhombic_multi_image_pair(
                        src,
                        dst,
                        [x, y, z],
                        distance_squared.sqrt(),
                        options,
                        atom_species,
                        atom_r2r4,
                        coordination_numbers,
                        need_vjp,
                        energy,
                        coordination_bars,
                        displacement_gradient,
                        c6_cache,
                        force_scatter,
                    )?;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn accumulate_orthorhombic_multi_image_pair(
        &self,
        src: usize,
        dst: usize,
        vector: Vec3,
        distance: f64,
        options: &PredictionOptions,
        atom_species: &[usize],
        atom_r2r4: &[f64],
        coordination_numbers: &[f64],
        need_vjp: bool,
        energy: &mut f64,
        coordination_bars: &mut [f64],
        displacement_gradient: &mut Mat3,
        c6_cache: &mut C6InterpolationCache,
        force_scatter: &mut Option<SourceBufferedForceScatter<'_>>,
    ) -> Result<(), Dftd3Error> {
        let species_src = atom_species[src];
        let species_dst = atom_species[dst];
        let c6 = self.c6_for_cached(
            species_src,
            species_dst,
            coordination_numbers[src],
            coordination_numbers[dst],
            c6_cache,
        )?;
        let value = smooth_d3bj_coordination_pair_value(
            self.cutoff_smoothing,
            self.parameters.cutoff,
            distance,
            d3bj_coordination_pair_energy_derivative(
                &self.parameters,
                c6.c6,
                3.0 * atom_r2r4[src] * atom_r2r4[dst],
                distance,
            ),
        );
        *energy += value.energy;

        if need_vjp {
            if distance > MIN_FORCE_DISTANCE {
                let bar_r = scale(vector, value.d_energy_d_distance / distance);
                if let Some(scatter) = force_scatter.as_mut() {
                    scatter.scatter(src, dst, bar_r);
                }
                if options.compute_stress || options.compute_virial {
                    accumulate_edge_displacement_gradient(vector, bar_r, displacement_gradient);
                }
            }
            let weighted_bar_c6 = DIRECTED_EDGE_WEIGHT * value.d_energy_d_c6;
            coordination_bars[src] += 2.0 * weighted_bar_c6 * c6.d_c6_d_coordination_i;
            coordination_bars[dst] += 2.0 * weighted_bar_c6 * c6.d_c6_d_coordination_j;
        }
        Ok(())
    }

    fn finish_orthorhombic_accum(
        &self,
        structure: &Structure,
        options: &PredictionOptions,
        accum: Dftd3OrthorhombicAccum,
    ) -> Result<Dftd3Evaluation, Dftd3Error> {
        let stress = if options.compute_stress {
            let cell = structure
                .cell
                .as_ref()
                .ok_or(Dftd3Error::MissingCellForStress)?;
            stress_from_displacement_gradient(accum.displacement_gradient, cell)?
        } else {
            [[0.0; 3]; 3]
        };
        let virial = if options.compute_virial {
            virial_from_displacement_gradient(accum.displacement_gradient)
        } else {
            [[0.0; 3]; 3]
        };

        Ok(Dftd3Evaluation {
            energy: accum.energy,
            forces: accum.forces,
            stress,
            virial,
        })
    }

    fn coordination_numbers(
        &self,
        atom_species: &[usize],
        neighbor_list: &crate::neighbor::NeighborList,
    ) -> Vec<f64> {
        let mut coordination_numbers = vec![0.0; atom_species.len()];
        for edge in &neighbor_list.edges {
            if edge.distance >= self.coordination.cutoff {
                continue;
            }
            let src = self.elements[atom_species[edge.src]];
            let dst = self.elements[atom_species[edge.dst]];
            coordination_numbers[edge.src] += coordination_value(
                self.coordination.k1,
                src.covalent_radius + dst.covalent_radius,
                edge.distance,
            )
            .value;
        }
        coordination_numbers
    }

    fn coordination_values_orthorhombic_edges(
        &self,
        atom_covalent_radii: &[f64],
        edges: &[SingleImageStoredEdge],
    ) -> OrthorhombicCoordinationValues {
        let mut coordination_numbers = vec![0.0; atom_covalent_radii.len()];
        let mut d_value_d_distances = vec![0.0; edges.len()];
        for (edge_index, edge) in edges.iter().enumerate() {
            if edge.distance >= self.coordination.cutoff {
                continue;
            }
            let src = edge.src;
            let dst = edge.dst;
            let value = coordination_value(
                self.coordination.k1,
                atom_covalent_radii[src] + atom_covalent_radii[dst],
                edge.distance,
            );
            let value_scalar = value.value;
            coordination_numbers[src] += value_scalar;
            coordination_numbers[dst] += value_scalar;
            d_value_d_distances[edge_index] = value.d_value_d_distance;
        }
        OrthorhombicCoordinationValues {
            numbers: coordination_numbers,
            d_value_d_distances,
        }
    }

    #[cfg(feature = "parallel")]
    fn coordination_values_orthorhombic_edges_parallel(
        &self,
        atom_covalent_radii: &[f64],
        edges: &[SingleImageStoredEdge],
    ) -> OrthorhombicCoordinationValues {
        let atom_count = atom_covalent_radii.len();
        let ranges = parallel_edge_work_ranges(edges.len());
        let chunks = ranges
            .par_iter()
            .map(|&(start, end)| {
                let mut local = vec![0.0; atom_count];
                let mut local_derivatives = Vec::with_capacity(end - start);
                for edge in &edges[start..end] {
                    if edge.distance >= self.coordination.cutoff {
                        local_derivatives.push(0.0);
                        continue;
                    }
                    let src = edge.src;
                    let dst = edge.dst;
                    let value = coordination_value(
                        self.coordination.k1,
                        atom_covalent_radii[src] + atom_covalent_radii[dst],
                        edge.distance,
                    );
                    local[src] += value.value;
                    local[dst] += value.value;
                    local_derivatives.push(value.d_value_d_distance);
                }
                (start, local, local_derivatives)
            })
            .collect::<Vec<_>>();
        let mut coordination_numbers = vec![0.0; atom_count];
        let mut d_value_d_distances = vec![0.0; edges.len()];
        for (start, local, local_derivatives) in chunks {
            for (target, source) in coordination_numbers.iter_mut().zip(local) {
                *target += source;
            }
            d_value_d_distances[start..start + local_derivatives.len()]
                .copy_from_slice(&local_derivatives);
        }
        OrthorhombicCoordinationValues {
            numbers: coordination_numbers,
            d_value_d_distances,
        }
    }

    fn accumulate_coordination_vjp(
        &self,
        atom_species: &[usize],
        neighbor_list: &crate::neighbor::NeighborList,
        coordination_bars: &[f64],
        options: &PredictionOptions,
        forces: &mut [Vec3],
        displacement_gradient: &mut Mat3,
    ) -> Result<(), Dftd3Error> {
        for edge in &neighbor_list.edges {
            if edge.distance >= self.coordination.cutoff {
                continue;
            }
            let src = self.elements[atom_species[edge.src]];
            let dst = self.elements[atom_species[edge.dst]];
            let value = coordination_value(
                self.coordination.k1,
                src.covalent_radius + dst.covalent_radius,
                edge.distance,
            );
            if edge.distance <= MIN_FORCE_DISTANCE || coordination_bars[edge.src] == 0.0 {
                continue;
            }
            let bar_r = scale(
                edge.vector,
                coordination_bars[edge.src] * value.d_value_d_distance / edge.distance,
            );
            if options.compute_forces {
                crate::force_accum::scatter_edge_vector_vjp(edge.src, edge.dst, bar_r, forces)?;
            }
            if options.compute_stress || options.compute_virial {
                accumulate_edge_displacement_gradient(edge.vector, bar_r, displacement_gradient);
            }
        }
        Ok(())
    }

    fn accumulate_coordination_vjp_orthorhombic_edges(
        &self,
        edges: &[SingleImageStoredEdge],
        coordination_derivatives: &[f64],
        coordination_bars: &[f64],
        options: &PredictionOptions,
        forces: &mut [Vec3],
        displacement_gradient: &mut Mat3,
    ) -> Result<(), Dftd3Error> {
        let mut force_scatter = if options.compute_forces {
            Some(SourceBufferedForceScatter::new(forces))
        } else {
            None
        };
        for (edge_index, edge) in edges.iter().enumerate() {
            let d_value_d_distance = coordination_derivatives[edge_index];
            if d_value_d_distance == 0.0 {
                continue;
            }
            let src = edge.src;
            let dst = edge.dst;
            let bar_coordination = coordination_bars[src] + coordination_bars[dst];
            if bar_coordination == 0.0 {
                continue;
            }
            if edge.distance <= MIN_FORCE_DISTANCE {
                continue;
            }
            let bar_r = scale(
                edge.vector,
                bar_coordination * d_value_d_distance / edge.distance,
            );
            if let Some(scatter) = force_scatter.as_mut() {
                scatter.scatter(src, dst, bar_r);
            }
            if options.compute_stress || options.compute_virial {
                accumulate_edge_displacement_gradient(edge.vector, bar_r, displacement_gradient);
            }
        }
        if let Some(scatter) = force_scatter {
            scatter.finish();
        }
        Ok(())
    }

    fn structure_species_indices(&self, structure: &Structure) -> Result<Vec<usize>, Dftd3Error> {
        if structure.species.len() != structure.positions.len() {
            return Err(Dftd3Error::SpeciesPositionMismatch {
                species: structure.species.len(),
                positions: structure.positions.len(),
            });
        }

        structure
            .species
            .iter()
            .map(|species| {
                validate_species(species)?;
                self.species_index_for(species)
            })
            .collect()
    }

    fn species_index_for(&self, species: &str) -> Result<usize, Dftd3Error> {
        self.species_indices.get(species).copied().ok_or_else(|| {
            Dftd3Error::UnknownElementCoefficient {
                species: species.to_owned(),
            }
        })
    }

    fn atom_covalent_radii(&self, atom_species: &[usize]) -> Vec<f64> {
        atom_species
            .iter()
            .map(|&species| self.elements[species].covalent_radius)
            .collect()
    }

    fn atom_r2r4(&self, atom_species: &[usize]) -> Vec<f64> {
        atom_species
            .iter()
            .map(|&species| self.elements[species].r2r4)
            .collect()
    }

    fn c6_for(
        &self,
        species_i: usize,
        species_j: usize,
        coordination_i: f64,
        coordination_j: f64,
    ) -> Result<C6InterpolationValue, Dftd3Error> {
        let element_count = self.elements.len();
        let lookup =
            self.pair_grid_indices[species_i * element_count + species_j].ok_or_else(|| {
                Dftd3Error::UnknownPairCoefficient {
                    species_a: self.species[species_i].clone(),
                    species_b: self.species[species_j].clone(),
                }
            })?;
        let (coordination_a, coordination_b) = if lookup.swapped {
            (coordination_j, coordination_i)
        } else {
            (coordination_i, coordination_j)
        };
        let value = interpolate_c6_references(
            &self.pair_grids[lookup.grid_index].references,
            coordination_a,
            coordination_b,
            self.coordination.k3,
        );
        if lookup.swapped {
            Ok(C6InterpolationValue {
                c6: value.c6,
                d_c6_d_coordination_i: value.d_c6_d_coordination_b,
                d_c6_d_coordination_j: value.d_c6_d_coordination_a,
            })
        } else {
            Ok(C6InterpolationValue {
                c6: value.c6,
                d_c6_d_coordination_i: value.d_c6_d_coordination_a,
                d_c6_d_coordination_j: value.d_c6_d_coordination_b,
            })
        }
    }

    fn c6_for_cached(
        &self,
        species_i: usize,
        species_j: usize,
        coordination_i: f64,
        coordination_j: f64,
        cache: &mut C6InterpolationCache,
    ) -> Result<C6InterpolationValue, Dftd3Error> {
        let key = C6InterpolationCacheKey {
            species_i,
            species_j,
            coordination_i: coordination_i.to_bits(),
            coordination_j: coordination_j.to_bits(),
        };
        if let Some(value) = cache.get(key) {
            return Ok(value);
        }
        let value = self.c6_for(species_i, species_j, coordination_i, coordination_j)?;
        cache.insert_or_disable(key, value);
        Ok(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PairCoefficient {
    c6: f64,
    c8: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ElementCoefficient {
    covalent_radius: f64,
    r2r4: f64,
}

#[derive(Clone, Debug)]
struct C6Grid {
    references: Vec<Dftd3C6Reference>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PairGridLookup {
    grid_index: usize,
    swapped: bool,
}

#[derive(Clone, Debug)]
struct Dftd3Evaluation {
    energy: f64,
    forces: Vec<Vec3>,
    stress: Mat3,
    virial: Mat3,
}

#[derive(Clone, Copy, Debug)]
struct OrthorhombicSingleImagePbc {
    lengths: Vec3,
    half_lengths: Vec3,
}

#[derive(Clone, Debug)]
struct GeneralSingleImagePbc {
    cell: Cell,
    metric: Mat3,
    fractional_positions: Vec<Vec3>,
    fractional_image_shifts: Vec<[i32; 3]>,
    cartesian_lower_bound_squared: f64,
}

#[derive(Clone, Copy, Debug)]
struct OrthorhombicMultiImagePbc {
    lengths: Vec3,
    max_shifts: [i32; 3],
}

#[derive(Clone, Copy, Debug)]
struct SingleImageStoredEdge {
    src: usize,
    dst: usize,
    vector: Vec3,
    distance: f64,
}

#[derive(Clone, Debug)]
struct OrthorhombicPositionAxes {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
}

impl OrthorhombicPositionAxes {
    fn from_positions(positions: &[Vec3], lengths: Vec3) -> Self {
        let mut x = Vec::with_capacity(positions.len());
        let mut y = Vec::with_capacity(positions.len());
        let mut z = Vec::with_capacity(positions.len());
        for position in positions {
            x.push(wrap_orthorhombic_coordinate(position[0], lengths[0]));
            y.push(wrap_orthorhombic_coordinate(position[1], lengths[1]));
            z.push(wrap_orthorhombic_coordinate(position[2], lengths[2]));
        }
        Self { x, y, z }
    }
}

fn wrap_orthorhombic_coordinate(value: f64, length: f64) -> f64 {
    value.rem_euclid(length)
}

#[derive(Clone, Debug)]
struct OrthorhombicCoordinationValues {
    numbers: Vec<f64>,
    d_value_d_distances: Vec<f64>,
}

#[derive(Clone, Debug)]
struct Dftd3OrthorhombicAccum {
    energy: f64,
    forces: Vec<Vec3>,
    coordination_bars: Vec<f64>,
    displacement_gradient: Mat3,
}

impl Dftd3OrthorhombicAccum {
    fn new(atom_count: usize) -> Self {
        Self {
            energy: 0.0,
            forces: vec![ZERO_VEC3; atom_count],
            coordination_bars: vec![0.0; atom_count],
            displacement_gradient: [[0.0; 3]; 3],
        }
    }

    #[cfg(feature = "parallel")]
    fn merge(&mut self, other: Self) {
        self.energy += other.energy;
        self.merge_vjp(other);
    }

    #[cfg(feature = "parallel")]
    fn merge_vjp(&mut self, other: Self) {
        for (target, source) in self.forces.iter_mut().zip(other.forces) {
            for axis in 0..3 {
                target[axis] += source[axis];
            }
        }
        for (target, source) in self
            .coordination_bars
            .iter_mut()
            .zip(other.coordination_bars)
        {
            *target += source;
        }
        for row in 0..3 {
            for col in 0..3 {
                self.displacement_gradient[row][col] += other.displacement_gradient[row][col];
            }
        }
    }
}

#[cfg(feature = "parallel")]
fn parallel_unordered_pair_source_ranges(atom_count: usize) -> Vec<(usize, usize)> {
    if atom_count == 0 {
        return Vec::new();
    }
    let total_pairs = atom_count.saturating_mul(atom_count.saturating_sub(1)) / 2;
    if total_pairs == 0 {
        return vec![(0, atom_count)];
    }

    let chunk_count = (rayon::current_num_threads() * 16).clamp(1, atom_count);
    let target_pairs = total_pairs.div_ceil(chunk_count).max(1);
    let mut ranges = Vec::with_capacity(chunk_count);
    let mut start = 0;
    while start < atom_count {
        let mut end = start;
        let mut pair_count = 0usize;
        while end < atom_count {
            let source_pairs = atom_count - end - 1;
            if end > start && pair_count + source_pairs > target_pairs {
                break;
            }
            pair_count += source_pairs;
            end += 1;
        }
        if end == start {
            end += 1;
        }
        ranges.push((start, end));
        start = end;
    }
    ranges
}

#[cfg(feature = "parallel")]
fn parallel_edge_work_ranges(item_count: usize) -> Vec<(usize, usize)> {
    let chunk_count = (rayon::current_num_threads() * 2).clamp(1, item_count.max(1));
    let chunk_size = item_count.div_ceil(chunk_count).max(1);
    (0..item_count)
        .step_by(chunk_size)
        .map(|start| (start, (start + chunk_size).min(item_count)))
        .collect()
}

struct SourceBufferedForceScatter<'a> {
    forces: &'a mut [Vec3],
    current_src: usize,
    current_bar: Vec3,
}

impl<'a> SourceBufferedForceScatter<'a> {
    const NO_SOURCE: usize = usize::MAX;

    fn new(forces: &'a mut [Vec3]) -> Self {
        Self {
            forces,
            current_src: Self::NO_SOURCE,
            current_bar: ZERO_VEC3,
        }
    }

    #[inline]
    fn scatter(&mut self, src: usize, dst: usize, bar_r: Vec3) {
        if self.current_src != src {
            self.flush();
            self.current_src = src;
        }
        self.current_bar[0] += bar_r[0];
        self.current_bar[1] += bar_r[1];
        self.current_bar[2] += bar_r[2];
        self.forces[dst][0] -= bar_r[0];
        self.forces[dst][1] -= bar_r[1];
        self.forces[dst][2] -= bar_r[2];
    }

    fn finish(mut self) {
        self.flush();
    }

    #[inline]
    fn flush(&mut self) {
        if self.current_src == Self::NO_SOURCE {
            return;
        }
        self.forces[self.current_src][0] += self.current_bar[0];
        self.forces[self.current_src][1] += self.current_bar[1];
        self.forces[self.current_src][2] += self.current_bar[2];
        self.current_bar = ZERO_VEC3;
        self.current_src = Self::NO_SOURCE;
    }
}

fn orthorhombic_single_image_pbc_context(
    structure: &Structure,
    cutoff: f64,
) -> Result<Option<OrthorhombicSingleImagePbc>, Dftd3Error> {
    if !structure.pbc.iter().all(|enabled| *enabled) {
        return Ok(None);
    }
    let Some(cell) = structure.cell else {
        return Ok(None);
    };
    validate_fast_path_positions_and_cell(structure)?;
    let vectors = cell.vectors;
    let off_diagonal_max = vectors[0][1]
        .abs()
        .max(vectors[0][2].abs())
        .max(vectors[1][0].abs())
        .max(vectors[1][2].abs())
        .max(vectors[2][0].abs())
        .max(vectors[2][1].abs());
    if off_diagonal_max > 1.0e-12 {
        return Ok(None);
    }
    let lengths = [vectors[0][0], vectors[1][1], vectors[2][2]];
    if lengths
        .iter()
        .any(|length| !length.is_finite() || *length <= 0.0)
    {
        return Ok(None);
    }
    let half_lengths = [0.5 * lengths[0], 0.5 * lengths[1], 0.5 * lengths[2]];
    let min_half_length = half_lengths[0].min(half_lengths[1]).min(half_lengths[2]);
    if cutoff >= min_half_length {
        return Ok(None);
    }
    Ok(Some(OrthorhombicSingleImagePbc {
        lengths,
        half_lengths,
    }))
}

fn orthorhombic_multi_image_pbc_context(
    structure: &Structure,
    cutoff: f64,
) -> Result<Option<OrthorhombicMultiImagePbc>, Dftd3Error> {
    if !structure.pbc.iter().all(|enabled| *enabled) {
        return Ok(None);
    }
    let Some(cell) = structure.cell else {
        return Ok(None);
    };
    validate_fast_path_positions_and_cell(structure)?;
    let vectors = cell.vectors;
    let off_diagonal_max = vectors[0][1]
        .abs()
        .max(vectors[0][2].abs())
        .max(vectors[1][0].abs())
        .max(vectors[1][2].abs())
        .max(vectors[2][0].abs())
        .max(vectors[2][1].abs());
    if off_diagonal_max > 1.0e-12 {
        return Ok(None);
    }
    let lengths = [vectors[0][0], vectors[1][1], vectors[2][2]];
    if lengths
        .iter()
        .any(|length| !length.is_finite() || *length <= 0.0)
    {
        return Ok(None);
    }

    let mut max_shifts = [0_i32; 3];
    for axis in 0..3 {
        // The guard image covers wrapped cross-pair deltas near +/-L before
        // the per-delta shift range is narrowed. The final distance check
        // rejects any over-provisioned images.
        let bound = (cutoff / lengths[axis]).ceil() + 1.0;
        if !bound.is_finite() || bound > f64::from(i32::MAX) {
            return Err(Dftd3Error::NeighborList(NeighborListError::MinimumImage(
                MinimumImageError::ShiftOutOfRange { axis, value: bound },
            )));
        }
        max_shifts[axis] = bound.max(0.0) as i32;
    }

    Ok(Some(OrthorhombicMultiImagePbc {
        lengths,
        max_shifts,
    }))
}

fn general_single_image_pbc_context(
    structure: &Structure,
    cutoff: f64,
) -> Result<Option<GeneralSingleImagePbc>, Dftd3Error> {
    if !structure.pbc.iter().all(|enabled| *enabled) {
        return Ok(None);
    }
    let Some(cell) = structure.cell else {
        return Ok(None);
    };
    validate_fast_path_positions_and_cell(structure)?;
    let limit = single_image_cutoff_limit(structure)?;
    if cutoff >= limit {
        return Ok(None);
    }

    let mut fractional_positions = Vec::with_capacity(structure.positions.len());
    let mut fractional_image_shifts = Vec::with_capacity(structure.positions.len());
    for position in &structure.positions {
        let fractional = cell
            .cartesian_to_fractional(*position)
            .ok_or(NeighborListError::SingularCell)?;
        let (wrapped_fractional, image_shift) = wrap_fractional_position(fractional)?;
        fractional_positions.push(wrapped_fractional);
        fractional_image_shifts.push(image_shift);
    }

    let cartesian_lower_bound = 2.0 * limit;
    let metric = cell_metric(cell);
    Ok(Some(GeneralSingleImagePbc {
        cell,
        metric,
        fractional_positions,
        fractional_image_shifts,
        cartesian_lower_bound_squared: cartesian_lower_bound * cartesian_lower_bound,
    }))
}

fn wrap_fractional_position(fractional: Vec3) -> Result<(Vec3, [i32; 3]), Dftd3Error> {
    let mut wrapped = [0.0; 3];
    let mut image_shift = [0_i32; 3];
    for axis in 0..3 {
        let value = fractional[axis];
        if !value.is_finite() {
            return Err(Dftd3Error::NeighborList(NeighborListError::MinimumImage(
                MinimumImageError::NonFiniteFractional { axis, value },
            )));
        }
        let image = fractional[axis].floor();
        if image < f64::from(i32::MIN) || image > f64::from(i32::MAX) {
            return Err(Dftd3Error::NeighborList(NeighborListError::MinimumImage(
                MinimumImageError::ShiftOutOfRange { axis, value: image },
            )));
        }
        image_shift[axis] = image as i32;
        wrapped[axis] = fractional[axis] - image;
    }
    Ok((wrapped, image_shift))
}

fn validate_fast_path_positions_and_cell(structure: &Structure) -> Result<(), Dftd3Error> {
    for (atom, position) in structure.positions.iter().enumerate() {
        for (axis, value) in position.iter().enumerate() {
            if !value.is_finite() {
                return Err(Dftd3Error::NeighborList(
                    NeighborListError::InvalidPosition {
                        atom,
                        axis,
                        value: *value,
                    },
                ));
            }
        }
    }

    let cell = structure.cell.ok_or(NeighborListError::MissingCellForPbc)?;
    for (vector, cell_vector) in cell.vectors.iter().enumerate() {
        for (axis, value) in cell_vector.iter().enumerate() {
            if !value.is_finite() {
                return Err(Dftd3Error::NeighborList(
                    NeighborListError::InvalidCellVector {
                        vector,
                        axis,
                        value: *value,
                    },
                ));
            }
        }
    }
    Ok(())
}

fn cell_metric(cell: Cell) -> Mat3 {
    let vectors = cell.vectors;
    [
        [
            dot(vectors[0], vectors[0]),
            dot(vectors[0], vectors[1]),
            dot(vectors[0], vectors[2]),
        ],
        [
            dot(vectors[1], vectors[0]),
            dot(vectors[1], vectors[1]),
            dot(vectors[1], vectors[2]),
        ],
        [
            dot(vectors[2], vectors[0]),
            dot(vectors[2], vectors[1]),
            dot(vectors[2], vectors[2]),
        ],
    ]
}

fn metric_norm2(metric: Mat3, vector: Vec3) -> f64 {
    metric[0][0] * vector[0] * vector[0]
        + metric[1][1] * vector[1] * vector[1]
        + metric[2][2] * vector[2] * vector[2]
        + 2.0
            * (metric[0][1] * vector[0] * vector[1]
                + metric[0][2] * vector[0] * vector[2]
                + metric[1][2] * vector[1] * vector[2])
}

fn general_single_image_neighbor_list(
    structure: &Structure,
    context: &GeneralSingleImagePbc,
    cutoff: f64,
) -> Result<NeighborList, Dftd3Error> {
    if cutoff == 0.0 || structure.positions.is_empty() {
        return Ok(NeighborList {
            edges: Vec::new(),
            cutoff,
            edge_mode: EdgeMode::Directed,
        });
    }

    let atom_count = structure.positions.len();
    let mut edges = Vec::with_capacity(general_single_image_edge_capacity_hint(
        atom_count, context, cutoff,
    ));
    let cutoff_squared = cutoff * cutoff;

    #[cfg(feature = "parallel")]
    if rayon::current_num_threads() > 1 && atom_count >= 512 {
        let ranges = parallel_unordered_pair_source_ranges(atom_count);
        let range_capacity = edges
            .capacity()
            .checked_div(ranges.len().max(1))
            .unwrap_or(0)
            .saturating_add(1);
        let chunks = ranges
            .par_iter()
            .map(|&(start, end)| {
                let mut local = Vec::with_capacity(range_capacity);
                collect_general_single_image_edges_for_source_range(
                    context,
                    cutoff_squared,
                    start,
                    end,
                    &mut local,
                )?;
                Ok::<Vec<Edge>, Dftd3Error>(local)
            })
            .collect::<Result<Vec<_>, Dftd3Error>>()?;
        let edge_count = chunks.iter().map(Vec::len).sum();
        edges = Vec::with_capacity(edge_count);
        for chunk in chunks {
            edges.extend(chunk);
        }
        return Ok(NeighborList {
            edges,
            cutoff,
            edge_mode: EdgeMode::Directed,
        });
    }

    collect_general_single_image_edges_for_source_range(
        context,
        cutoff_squared,
        0,
        atom_count,
        &mut edges,
    )?;

    Ok(NeighborList {
        edges,
        cutoff,
        edge_mode: EdgeMode::Directed,
    })
}

fn collect_general_single_image_edges(
    structure: &Structure,
    context: &GeneralSingleImagePbc,
    cutoff: f64,
) -> Result<Vec<SingleImageStoredEdge>, Dftd3Error> {
    if cutoff == 0.0 || structure.positions.is_empty() {
        return Ok(Vec::new());
    }
    let atom_count = structure.positions.len();
    let mut edges = Vec::with_capacity(
        general_single_image_edge_capacity_hint(atom_count, context, cutoff) / 2,
    );
    collect_general_single_image_stored_edges_for_source_range(
        context,
        cutoff * cutoff,
        0,
        atom_count,
        &mut edges,
    )?;
    Ok(edges)
}

fn collect_general_single_image_stored_edges_for_source_range(
    context: &GeneralSingleImagePbc,
    cutoff_squared: f64,
    start: usize,
    end: usize,
    edges: &mut Vec<SingleImageStoredEdge>,
) -> Result<(), Dftd3Error> {
    let fractional_positions = context.fractional_positions.as_slice();
    for src in start..end {
        let src_fractional = fractional_positions[src];
        for (dst, dst_fractional) in fractional_positions
            .iter()
            .copied()
            .enumerate()
            .skip(src + 1)
        {
            let delta_fractional = [
                dst_fractional[0] - src_fractional[0],
                dst_fractional[1] - src_fractional[1],
                dst_fractional[2] - src_fractional[2],
            ];
            let Some(image) = general_single_image_pair(context, delta_fractional, cutoff_squared)?
            else {
                continue;
            };
            edges.push(SingleImageStoredEdge {
                src,
                dst,
                vector: image.vector,
                distance: image.distance_squared.sqrt(),
            });
        }
    }
    Ok(())
}

fn collect_general_single_image_edges_for_source_range(
    context: &GeneralSingleImagePbc,
    cutoff_squared: f64,
    start: usize,
    end: usize,
    edges: &mut Vec<Edge>,
) -> Result<(), Dftd3Error> {
    let fractional_positions = context.fractional_positions.as_slice();
    let fractional_image_shifts = context.fractional_image_shifts.as_slice();
    for src in start..end {
        let src_fractional = fractional_positions[src];
        let src_image_shift = fractional_image_shifts[src];
        for (dst, dst_fractional) in fractional_positions
            .iter()
            .copied()
            .enumerate()
            .skip(src + 1)
        {
            let delta_fractional = [
                dst_fractional[0] - src_fractional[0],
                dst_fractional[1] - src_fractional[1],
                dst_fractional[2] - src_fractional[2],
            ];
            let Some(image) = general_single_image_pair(context, delta_fractional, cutoff_squared)?
            else {
                continue;
            };
            let shift = combine_wrapped_shift(
                image.wrapped_shift,
                src_image_shift,
                fractional_image_shifts[dst],
            )?;
            let shift_cart = context.cell.cartesian_shift(shift);
            push_directed_single_image_edges(
                edges,
                src,
                dst,
                shift,
                shift_cart,
                image.vector,
                image.distance_squared.sqrt(),
            )?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct GeneralSingleImagePair {
    wrapped_shift: [i32; 3],
    vector: Vec3,
    distance_squared: f64,
}

fn general_single_image_pair(
    context: &GeneralSingleImagePbc,
    delta_fractional: Vec3,
    cutoff_squared: f64,
) -> Result<Option<GeneralSingleImagePair>, Dftd3Error> {
    let base_shift = wrapped_minimum_image_shift(delta_fractional);
    let wrapped_fractional = [
        delta_fractional[0] + f64::from(base_shift[0]),
        delta_fractional[1] + f64::from(base_shift[1]),
        delta_fractional[2] + f64::from(base_shift[2]),
    ];
    let fractional_norm2 = dot(wrapped_fractional, wrapped_fractional);
    let lower_bound2 = context.cartesian_lower_bound_squared * fractional_norm2;
    if lower_bound2 >= cutoff_squared {
        return Ok(None);
    }

    let base_norm2 = metric_norm2(context.metric, wrapped_fractional);
    if base_norm2 >= cutoff_squared {
        return Ok(None);
    }
    // The context exists only when cutoff is below a conservative half shortest
    // periodic-translation bound, so an alternate image cannot be the only one
    // inside cutoff after the wrapped fractional candidate is outside.
    Ok(Some(GeneralSingleImagePair {
        wrapped_shift: base_shift,
        vector: context.cell.fractional_to_cartesian(wrapped_fractional),
        distance_squared: base_norm2,
    }))
}

fn wrapped_minimum_image_shift(delta_fractional: Vec3) -> [i32; 3] {
    let mut shift = [0_i32; 3];
    for axis in 0..3 {
        let value = delta_fractional[axis];
        shift[axis] = if value >= 0.5 {
            -1
        } else if value <= -0.5 {
            1
        } else {
            0
        };
    }
    shift
}

fn combine_wrapped_shift(
    wrapped_shift: [i32; 3],
    src_image_shift: [i32; 3],
    dst_image_shift: [i32; 3],
) -> Result<[i32; 3], Dftd3Error> {
    let mut shift = [0_i32; 3];
    for axis in 0..3 {
        let image_delta = src_image_shift[axis]
            .checked_sub(dst_image_shift[axis])
            .ok_or(Dftd3Error::NeighborList(NeighborListError::MinimumImage(
                MinimumImageError::ShiftOverflow {
                    axis,
                    base: src_image_shift[axis],
                    delta: dst_image_shift[axis],
                },
            )))?;
        shift[axis] =
            wrapped_shift[axis]
                .checked_add(image_delta)
                .ok_or(Dftd3Error::NeighborList(NeighborListError::MinimumImage(
                    MinimumImageError::ShiftOverflow {
                        axis,
                        base: wrapped_shift[axis],
                        delta: image_delta,
                    },
                )))?;
    }
    Ok(shift)
}

fn push_directed_single_image_edges(
    edges: &mut Vec<Edge>,
    src: usize,
    dst: usize,
    shift: [i32; 3],
    shift_cart: Vec3,
    vector: Vec3,
    distance: f64,
) -> Result<(), Dftd3Error> {
    edges.push(Edge {
        src,
        dst,
        shift,
        shift_cart,
        vector,
        distance,
    });

    let reverse_shift = negate_shift(shift)?;
    let reverse_shift_cart = scale(shift_cart, -1.0);
    edges.push(Edge {
        src: dst,
        dst: src,
        shift: reverse_shift,
        shift_cart: reverse_shift_cart,
        vector: scale(vector, -1.0),
        distance,
    });
    Ok(())
}

fn negate_shift(shift: [i32; 3]) -> Result<[i32; 3], Dftd3Error> {
    let mut reverse = [0_i32; 3];
    for axis in 0..3 {
        reverse[axis] = shift[axis].checked_neg().ok_or(Dftd3Error::NeighborList(
            NeighborListError::MinimumImage(MinimumImageError::ShiftOutOfRange {
                axis,
                value: f64::from(shift[axis]),
            }),
        ))?;
    }
    Ok(reverse)
}

fn general_single_image_edge_capacity_hint(
    atom_count: usize,
    context: &GeneralSingleImagePbc,
    cutoff: f64,
) -> usize {
    if atom_count < 2 {
        return 0;
    }
    let volume = crate::force_accum::cell_volume(&context.cell).abs();
    if !volume.is_finite() || volume <= 0.0 {
        return 0;
    }
    let density = atom_count as f64 / volume;
    let sphere_volume = 4.0 / 3.0 * std::f64::consts::PI * cutoff.powi(3);
    let directed_edges = atom_count as f64 * density * sphere_volume;
    directed_edges.ceil().clamp(
        0.0,
        (atom_count.saturating_mul(atom_count.saturating_sub(1))) as f64,
    ) as usize
}

fn collect_orthorhombic_multi_image_edges(
    axes: &OrthorhombicPositionAxes,
    context: OrthorhombicMultiImagePbc,
    cutoff: f64,
) -> Vec<SingleImageStoredEdge> {
    if cutoff == 0.0 || axes.x.is_empty() {
        return Vec::new();
    }
    let mut edges = Vec::with_capacity(orthorhombic_multi_image_edge_capacity_hint(
        axes.x.len(),
        context,
        cutoff,
    ));
    collect_orthorhombic_multi_image_edges_for_source_range(
        axes,
        context,
        cutoff,
        0,
        axes.x.len(),
        &mut edges,
    );
    edges
}

#[cfg(feature = "parallel")]
fn collect_orthorhombic_multi_image_edges_parallel(
    axes: &OrthorhombicPositionAxes,
    context: OrthorhombicMultiImagePbc,
    cutoff: f64,
) -> Vec<SingleImageStoredEdge> {
    if cutoff == 0.0 || axes.x.is_empty() {
        return Vec::new();
    }
    let ranges = parallel_unordered_pair_source_ranges(axes.x.len());
    let range_capacity = orthorhombic_multi_image_edge_capacity_hint(axes.x.len(), context, cutoff)
        .checked_div(ranges.len().max(1))
        .unwrap_or(0)
        .saturating_add(1);
    let chunks = ranges
        .par_iter()
        .map(|&(start, end)| {
            let mut local = Vec::with_capacity(range_capacity);
            collect_orthorhombic_multi_image_edges_for_source_range(
                axes, context, cutoff, start, end, &mut local,
            );
            local
        })
        .collect::<Vec<_>>();
    let edge_count = chunks.iter().map(Vec::len).sum();
    let mut edges = Vec::with_capacity(edge_count);
    for chunk in chunks {
        edges.extend(chunk);
    }
    edges
}

fn collect_orthorhombic_multi_image_edges_for_source_range(
    axes: &OrthorhombicPositionAxes,
    context: OrthorhombicMultiImagePbc,
    cutoff: f64,
    start: usize,
    end: usize,
    edges: &mut Vec<SingleImageStoredEdge>,
) {
    let cutoff_squared = cutoff * cutoff;
    for src in start..end {
        collect_orthorhombic_multi_image_self_edges(src, context, cutoff, cutoff_squared, edges);
        let src_x = axes.x[src];
        let src_y = axes.y[src];
        let src_z = axes.z[src];
        for dst in src + 1..axes.x.len() {
            let delta = [
                axes.x[dst] - src_x,
                axes.y[dst] - src_y,
                axes.z[dst] - src_z,
            ];
            collect_orthorhombic_multi_image_cross_edges(
                src,
                dst,
                delta,
                context,
                cutoff,
                cutoff_squared,
                edges,
            );
        }
    }
}

fn collect_orthorhombic_multi_image_self_edges(
    atom: usize,
    context: OrthorhombicMultiImagePbc,
    cutoff: f64,
    cutoff_squared: f64,
    edges: &mut Vec<SingleImageStoredEdge>,
) {
    let [range_x, range_y, range_z] =
        orthorhombic_multi_image_shift_ranges([0.0, 0.0, 0.0], context, cutoff);
    for shift_x in range_x.0..=range_x.1 {
        let x = f64::from(shift_x) * context.lengths[0];
        let x2 = x * x;
        if x2 >= cutoff_squared {
            continue;
        }
        for shift_y in range_y.0..=range_y.1 {
            let y = f64::from(shift_y) * context.lengths[1];
            let xy2 = x2 + y * y;
            if xy2 >= cutoff_squared {
                continue;
            }
            for shift_z in range_z.0..=range_z.1 {
                if !is_positive_periodic_shift([shift_x, shift_y, shift_z]) {
                    continue;
                }
                let z = f64::from(shift_z) * context.lengths[2];
                let distance_squared = xy2 + z * z;
                if distance_squared < cutoff_squared {
                    edges.push(SingleImageStoredEdge {
                        src: atom,
                        dst: atom,
                        vector: [x, y, z],
                        distance: distance_squared.sqrt(),
                    });
                }
            }
        }
    }
}

fn collect_orthorhombic_multi_image_cross_edges(
    src: usize,
    dst: usize,
    delta: Vec3,
    context: OrthorhombicMultiImagePbc,
    cutoff: f64,
    cutoff_squared: f64,
    edges: &mut Vec<SingleImageStoredEdge>,
) {
    let [range_x, range_y, range_z] = orthorhombic_multi_image_shift_ranges(delta, context, cutoff);
    for shift_x in range_x.0..=range_x.1 {
        let x = delta[0] + f64::from(shift_x) * context.lengths[0];
        let x2 = x * x;
        if x2 >= cutoff_squared {
            continue;
        }
        for shift_y in range_y.0..=range_y.1 {
            let y = delta[1] + f64::from(shift_y) * context.lengths[1];
            let xy2 = x2 + y * y;
            if xy2 >= cutoff_squared {
                continue;
            }
            for shift_z in range_z.0..=range_z.1 {
                let z = delta[2] + f64::from(shift_z) * context.lengths[2];
                let distance_squared = xy2 + z * z;
                if distance_squared < cutoff_squared {
                    edges.push(SingleImageStoredEdge {
                        src,
                        dst,
                        vector: [x, y, z],
                        distance: distance_squared.sqrt(),
                    });
                }
            }
        }
    }
}

fn orthorhombic_multi_image_shift_ranges(
    delta: Vec3,
    context: OrthorhombicMultiImagePbc,
    cutoff: f64,
) -> [(i32, i32); 3] {
    [
        orthorhombic_multi_image_shift_range(
            delta[0],
            context.lengths[0],
            cutoff,
            context.max_shifts[0],
        ),
        orthorhombic_multi_image_shift_range(
            delta[1],
            context.lengths[1],
            cutoff,
            context.max_shifts[1],
        ),
        orthorhombic_multi_image_shift_range(
            delta[2],
            context.lengths[2],
            cutoff,
            context.max_shifts[2],
        ),
    ]
}

fn orthorhombic_multi_image_shift_range(
    delta: f64,
    length: f64,
    cutoff: f64,
    max_shift: i32,
) -> (i32, i32) {
    let min_shift = ((-cutoff - delta) / length).ceil() as i32;
    let max_shift_for_delta = ((cutoff - delta) / length).floor() as i32;
    (
        min_shift.max(-max_shift),
        max_shift_for_delta.min(max_shift),
    )
}

fn is_positive_periodic_shift(shift: [i32; 3]) -> bool {
    shift[0] > 0 || (shift[0] == 0 && (shift[1] > 0 || (shift[1] == 0 && shift[2] > 0)))
}

fn orthorhombic_multi_image_edge_capacity_hint(
    atom_count: usize,
    context: OrthorhombicMultiImagePbc,
    cutoff: f64,
) -> usize {
    if atom_count == 0 {
        return 0;
    }
    let volume = context.lengths[0] * context.lengths[1] * context.lengths[2];
    if !volume.is_finite() || volume <= 0.0 {
        return 0;
    }
    let density = atom_count as f64 / volume;
    let sphere_volume = 4.0 / 3.0 * std::f64::consts::PI * cutoff.powi(3);
    let unordered_edges = 0.5 * atom_count as f64 * density * sphere_volume;
    unordered_edges.ceil().max(0.0) as usize
}

fn collect_orthorhombic_edges_for_source_range(
    axes: &OrthorhombicPositionAxes,
    context: OrthorhombicSingleImagePbc,
    cutoff_squared: f64,
    start: usize,
    end: usize,
    edges: &mut Vec<SingleImageStoredEdge>,
) {
    let atom_count = axes.x.len();
    let cutoff_squared_x2 = f64x2::splat(cutoff_squared);
    let xs = axes.x.as_slice();
    let ys = axes.y.as_slice();
    let zs = axes.z.as_slice();
    for src in start..end {
        let src_x = xs[src];
        let src_y = ys[src];
        let src_z = zs[src];
        let mut dst = src + 1;
        while dst + 1 < atom_count {
            let dst0 = dst;
            let dst1 = dst + 1;
            let x = orthorhombic_minimum_image_component_x2(
                f64x2::new([xs[dst0] - src_x, xs[dst1] - src_x]),
                context.lengths[0],
                context.half_lengths[0],
            );
            let x2 = x * x;
            if !x2.simd_lt(cutoff_squared_x2).any() {
                dst += 2;
                continue;
            }

            let y = orthorhombic_minimum_image_component_x2(
                f64x2::new([ys[dst0] - src_y, ys[dst1] - src_y]),
                context.lengths[1],
                context.half_lengths[1],
            );
            let xy2 = x2 + y * y;
            if !xy2.simd_lt(cutoff_squared_x2).any() {
                dst += 2;
                continue;
            }

            let z = orthorhombic_minimum_image_component_x2(
                f64x2::new([zs[dst0] - src_z, zs[dst1] - src_z]),
                context.lengths[2],
                context.half_lengths[2],
            );
            let distance_squared = xy2 + z * z;
            if distance_squared.simd_lt(cutoff_squared_x2).any() {
                let x = x.to_array();
                let y = y.to_array();
                let z = z.to_array();
                let distance_squared = distance_squared.to_array();
                if distance_squared[0] < cutoff_squared {
                    edges.push(SingleImageStoredEdge {
                        src,
                        dst: dst0,
                        vector: [x[0], y[0], z[0]],
                        distance: distance_squared[0].sqrt(),
                    });
                }
                if distance_squared[1] < cutoff_squared {
                    edges.push(SingleImageStoredEdge {
                        src,
                        dst: dst1,
                        vector: [x[1], y[1], z[1]],
                        distance: distance_squared[1].sqrt(),
                    });
                }
            }
            dst += 2;
        }
        for dst in dst..atom_count {
            let x = orthorhombic_minimum_image_component(
                xs[dst] - src_x,
                context.lengths[0],
                context.half_lengths[0],
            );
            let x2 = x * x;
            if x2 >= cutoff_squared {
                continue;
            }

            let y = orthorhombic_minimum_image_component(
                ys[dst] - src_y,
                context.lengths[1],
                context.half_lengths[1],
            );
            let xy2 = x2 + y * y;
            if xy2 >= cutoff_squared {
                continue;
            }

            let z = orthorhombic_minimum_image_component(
                zs[dst] - src_z,
                context.lengths[2],
                context.half_lengths[2],
            );
            let distance_squared = xy2 + z * z;
            if distance_squared < cutoff_squared {
                edges.push(SingleImageStoredEdge {
                    src,
                    dst,
                    vector: [x, y, z],
                    distance: distance_squared.sqrt(),
                });
            }
        }
    }
}

fn orthorhombic_minimum_image_component_x2(delta: f64x2, length: f64, half_length: f64) -> f64x2 {
    let negative_half_length = f64x2::splat(-half_length);
    let length = f64x2::splat(length);
    let half_length = f64x2::splat(half_length);
    delta.simd_gt(half_length).blend(
        delta - length,
        delta
            .simd_lt(negative_half_length)
            .blend(delta + length, delta),
    )
}

fn orthorhombic_minimum_image_component(delta: f64, length: f64, half_length: f64) -> f64 {
    if delta > half_length {
        delta - length
    } else if delta < -half_length {
        delta + length
    } else {
        delta
    }
}

fn collect_orthorhombic_single_image_edges(
    structure: &Structure,
    context: OrthorhombicSingleImagePbc,
    cutoff_squared: f64,
) -> Vec<SingleImageStoredEdge> {
    let atom_count = structure.positions.len();
    let axes = OrthorhombicPositionAxes::from_positions(&structure.positions, context.lengths);
    let mut edges = Vec::with_capacity(orthorhombic_edge_capacity_hint(
        atom_count,
        context,
        cutoff_squared,
    ));
    collect_orthorhombic_edges_for_source_range(
        &axes,
        context,
        cutoff_squared,
        0,
        atom_count,
        &mut edges,
    );
    edges
}

fn orthorhombic_edge_capacity_hint(
    atom_count: usize,
    context: OrthorhombicSingleImagePbc,
    cutoff_squared: f64,
) -> usize {
    if atom_count < 2 {
        return 0;
    }
    let volume = context.lengths[0] * context.lengths[1] * context.lengths[2];
    if volume <= 0.0 {
        return 0;
    }
    let cutoff = cutoff_squared.sqrt();
    let sphere_fraction =
        (4.0 * std::f64::consts::PI * cutoff * cutoff * cutoff / (3.0 * volume)).clamp(0.0, 1.0);
    let unordered_pairs = (atom_count as f64) * ((atom_count - 1) as f64) * 0.5;
    (unordered_pairs * sphere_fraction * 1.1).ceil() as usize
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PairKey {
    species_low: String,
    species_high: String,
}

impl PairKey {
    fn new(species_a: &str, species_b: &str) -> Self {
        if species_a <= species_b {
            Self {
                species_low: species_a.to_owned(),
                species_high: species_b.to_owned(),
            }
        } else {
            Self {
                species_low: species_b.to_owned(),
                species_high: species_a.to_owned(),
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dftd3PairValue {
    pub energy: f64,
    pub d_energy_d_distance: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Dftd3CoordinationPairValue {
    energy: f64,
    d_energy_d_distance: f64,
    d_energy_d_c6: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct C6InterpolationValue {
    c6: f64,
    d_c6_d_coordination_i: f64,
    d_c6_d_coordination_j: f64,
}

impl C6InterpolationValue {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct C6InterpolationCacheKey {
    species_i: usize,
    species_j: usize,
    coordination_i: u64,
    coordination_j: u64,
}

#[derive(Clone, Debug)]
struct C6InterpolationCache {
    slots: Vec<Option<C6InterpolationCacheEntry>>,
}

#[derive(Clone, Copy, Debug)]
struct C6InterpolationCacheEntry {
    key: C6InterpolationCacheKey,
    value: C6InterpolationValue,
}

fn c6_cache_mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

impl C6InterpolationCache {
    fn new() -> Self {
        Self {
            slots: vec![None; C6_INTERPOLATION_CACHE_MAX_ENTRIES],
        }
    }

    fn get(&self, key: C6InterpolationCacheKey) -> Option<C6InterpolationValue> {
        let entry = self.slots[c6_cache_slot(key)]?;
        if entry.key == key {
            Some(entry.value)
        } else {
            None
        }
    }

    fn insert_or_disable(&mut self, key: C6InterpolationCacheKey, value: C6InterpolationValue) {
        self.slots[c6_cache_slot(key)] = Some(C6InterpolationCacheEntry { key, value });
    }
}

fn c6_cache_slot(key: C6InterpolationCacheKey) -> usize {
    debug_assert!(C6_INTERPOLATION_CACHE_MAX_ENTRIES.is_power_of_two());
    let mut state = c6_cache_mix(key.species_i as u64);
    state = c6_cache_mix(state ^ key.species_j as u64);
    state = c6_cache_mix(state ^ key.coordination_i);
    state = c6_cache_mix(state ^ key.coordination_j);
    (state as usize) & (C6_INTERPOLATION_CACHE_MAX_ENTRIES - 1)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct C6ReferenceInterpolationValue {
    c6: f64,
    d_c6_d_coordination_a: f64,
    d_c6_d_coordination_b: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CoordinationValue {
    value: f64,
    d_value_d_distance: f64,
}

#[must_use]
fn d3bj_pair_energy_derivative(
    parameters: &Dftd3BjParameters,
    coefficient: PairCoefficient,
    distance: f64,
) -> Dftd3PairValue {
    if distance <= 0.0 {
        let damping_radius = bj_damping_radius(parameters, coefficient);
        return Dftd3PairValue {
            energy: -parameters.s6 * coefficient.c6 / damping_radius.powi(6)
                - parameters.s8 * coefficient.c8 / damping_radius.powi(8),
            d_energy_d_distance: 0.0,
        };
    }

    let damping_radius = bj_damping_radius(parameters, coefficient);
    let r2 = distance * distance;
    let r5 = r2 * r2 * distance;
    let r6 = r5 * distance;
    let r7 = r6 * distance;
    let r8 = r7 * distance;
    let a2 = damping_radius * damping_radius;
    let a6 = a2 * a2 * a2;
    let a8 = a6 * a2;
    let denom6 = r6 + a6;
    let denom8 = r8 + a8;
    let energy = -parameters.s6 * coefficient.c6 / denom6 - parameters.s8 * coefficient.c8 / denom8;
    let d_energy_d_distance = parameters.s6 * coefficient.c6 * 6.0 * r5 / (denom6 * denom6)
        + parameters.s8 * coefficient.c8 * 8.0 * r7 / (denom8 * denom8);

    Dftd3PairValue {
        energy,
        d_energy_d_distance,
    }
}

#[must_use]
fn d3bj_coordination_pair_energy_derivative(
    parameters: &Dftd3BjParameters,
    c6: f64,
    c8_ratio: f64,
    distance: f64,
) -> Dftd3CoordinationPairValue {
    let c8 = c8_ratio * c6;
    if distance <= 0.0 {
        let damping_radius = parameters.a1 * c8_ratio.sqrt() + parameters.a2;
        return Dftd3CoordinationPairValue {
            energy: -parameters.s6 * c6 / damping_radius.powi(6)
                - parameters.s8 * c8 / damping_radius.powi(8),
            d_energy_d_distance: 0.0,
            d_energy_d_c6: -parameters.s6 / damping_radius.powi(6)
                - parameters.s8 * c8_ratio / damping_radius.powi(8),
        };
    }

    let damping_radius = parameters.a1 * c8_ratio.sqrt() + parameters.a2;
    let r2 = distance * distance;
    let r5 = r2 * r2 * distance;
    let r6 = r5 * distance;
    let r7 = r6 * distance;
    let r8 = r7 * distance;
    let a2 = damping_radius * damping_radius;
    let a6 = a2 * a2 * a2;
    let a8 = a6 * a2;
    let denom6 = r6 + a6;
    let denom8 = r8 + a8;
    let energy = -parameters.s6 * c6 / denom6 - parameters.s8 * c8 / denom8;
    let d_energy_d_distance = parameters.s6 * c6 * 6.0 * r5 / (denom6 * denom6)
        + parameters.s8 * c8 * 8.0 * r7 / (denom8 * denom8);
    let d_energy_d_c6 = -parameters.s6 / denom6 - parameters.s8 * c8_ratio / denom8;

    Dftd3CoordinationPairValue {
        energy,
        d_energy_d_distance,
        d_energy_d_c6,
    }
}

#[must_use]
fn smooth_d3bj_pair_value(
    smoothing: Dftd3CutoffSmoothing,
    cutoff: f64,
    distance: f64,
    value: Dftd3PairValue,
) -> Dftd3PairValue {
    let (smooth, d_smooth_d_distance) =
        cutoff_smoothing_value_derivative(smoothing, cutoff, distance);
    Dftd3PairValue {
        energy: smooth * value.energy,
        d_energy_d_distance: smooth * value.d_energy_d_distance
            + d_smooth_d_distance * value.energy,
    }
}

#[must_use]
fn smooth_d3bj_coordination_pair_value(
    smoothing: Dftd3CutoffSmoothing,
    cutoff: f64,
    distance: f64,
    value: Dftd3CoordinationPairValue,
) -> Dftd3CoordinationPairValue {
    let (smooth, d_smooth_d_distance) =
        cutoff_smoothing_value_derivative(smoothing, cutoff, distance);
    Dftd3CoordinationPairValue {
        energy: smooth * value.energy,
        d_energy_d_distance: smooth * value.d_energy_d_distance
            + d_smooth_d_distance * value.energy,
        d_energy_d_c6: smooth * value.d_energy_d_c6,
    }
}

#[must_use]
fn cutoff_smoothing_value_derivative(
    smoothing: Dftd3CutoffSmoothing,
    cutoff: f64,
    distance: f64,
) -> (f64, f64) {
    match smoothing {
        Dftd3CutoffSmoothing::None => (1.0, 0.0),
        Dftd3CutoffSmoothing::Cosine => {
            if distance >= cutoff {
                (0.0, 0.0)
            } else {
                let angle = std::f64::consts::PI * distance / cutoff;
                let (sin_angle, cos_angle) = angle.sin_cos();
                (
                    0.5 * (cos_angle + 1.0),
                    -0.5 * std::f64::consts::PI / cutoff * sin_angle,
                )
            }
        }
    }
}

#[must_use]
fn coordination_value(k1: f64, covalent_radius_sum: f64, distance: f64) -> CoordinationValue {
    if distance <= 0.0 {
        return CoordinationValue {
            value: 1.0,
            d_value_d_distance: 0.0,
        };
    }
    let exponent = -k1 * (covalent_radius_sum / distance - 1.0);
    let value = 1.0 / (1.0 + exponent.exp());
    let d_value_d_distance =
        -value * (1.0 - value) * k1 * covalent_radius_sum / (distance * distance);
    CoordinationValue {
        value,
        d_value_d_distance,
    }
}

#[must_use]
fn interpolate_c6_references(
    references: &[Dftd3C6Reference],
    coordination_a: f64,
    coordination_b: f64,
    k3: f64,
) -> C6ReferenceInterpolationValue {
    debug_assert!(
        !references.is_empty(),
        "DFTD3 C6 interpolation requires at least one reference; model construction validates this"
    );
    let mut max_log_weight = f64::NEG_INFINITY;
    for reference in references {
        let delta_a = coordination_a - reference.coordination_a;
        let delta_b = coordination_b - reference.coordination_b;
        let log_weight = k3 * (delta_a * delta_a + delta_b * delta_b);
        max_log_weight = max_log_weight.max(log_weight);
    }

    let mut weight_sum = 0.0;
    let mut weighted_c6 = 0.0;
    let mut d_weight_sum_d_coordination_a = 0.0;
    let mut d_weight_sum_d_coordination_b = 0.0;
    let mut d_weighted_c6_d_coordination_a = 0.0;
    let mut d_weighted_c6_d_coordination_b = 0.0;
    for reference in references {
        let delta_a = coordination_a - reference.coordination_a;
        let delta_b = coordination_b - reference.coordination_b;
        let log_weight = k3 * (delta_a * delta_a + delta_b * delta_b);
        let weight = (log_weight - max_log_weight).exp();
        let d_log_weight_d_coordination_a = 2.0 * k3 * delta_a;
        let d_log_weight_d_coordination_b = 2.0 * k3 * delta_b;
        let d_weight_d_coordination_a = weight * d_log_weight_d_coordination_a;
        let d_weight_d_coordination_b = weight * d_log_weight_d_coordination_b;
        weight_sum += weight;
        weighted_c6 += weight * reference.c6;
        d_weight_sum_d_coordination_a += d_weight_d_coordination_a;
        d_weight_sum_d_coordination_b += d_weight_d_coordination_b;
        d_weighted_c6_d_coordination_a += d_weight_d_coordination_a * reference.c6;
        d_weighted_c6_d_coordination_b += d_weight_d_coordination_b * reference.c6;
    }
    let c6 = weighted_c6 / weight_sum;
    let d_c6_d_coordination_a =
        (d_weighted_c6_d_coordination_a - c6 * d_weight_sum_d_coordination_a) / weight_sum;
    let d_c6_d_coordination_b =
        (d_weighted_c6_d_coordination_b - c6 * d_weight_sum_d_coordination_b) / weight_sum;

    C6ReferenceInterpolationValue {
        c6,
        d_c6_d_coordination_a,
        d_c6_d_coordination_b,
    }
}

#[must_use]
fn bj_damping_radius(parameters: &Dftd3BjParameters, coefficient: PairCoefficient) -> f64 {
    parameters.a1 * (coefficient.c8 / coefficient.c6).sqrt() + parameters.a2
}

fn validate_coordination_parameters(
    parameters: &Dftd3CoordinationParameters,
) -> Result<(), Dftd3Error> {
    validate_positive_finite("k1", parameters.k1)?;
    validate_negative_finite("k3", parameters.k3)?;
    validate_positive_finite("coordination_cutoff", parameters.cutoff)?;
    Ok(())
}

fn validate_parameters(parameters: &Dftd3BjParameters) -> Result<(), Dftd3Error> {
    validate_positive_finite("s6", parameters.s6)?;
    validate_nonnegative_finite("s8", parameters.s8)?;
    validate_nonnegative_finite("a1", parameters.a1)?;
    validate_positive_finite("a2", parameters.a2)?;
    validate_positive_finite("cutoff", parameters.cutoff)?;
    Ok(())
}

fn validate_supported_pbc(structure: &Structure) -> Result<(), Dftd3Error> {
    let any_pbc = structure.pbc.iter().any(|enabled| *enabled);
    let all_pbc = structure.pbc.iter().all(|enabled| *enabled);
    if any_pbc && !all_pbc {
        return Err(Dftd3Error::UnsupportedPartialPbc { pbc: structure.pbc });
    }
    Ok(())
}

fn validate_species(species: &str) -> Result<(), Dftd3Error> {
    if species.is_empty()
        || !species
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    {
        Err(Dftd3Error::InvalidSpecies(species.to_owned()))
    } else {
        Ok(())
    }
}

fn validate_negative_finite(name: &'static str, value: f64) -> Result<(), Dftd3Error> {
    if value.is_finite() && value < 0.0 {
        Ok(())
    } else {
        Err(Dftd3Error::InvalidParameter { name, value })
    }
}

fn validate_positive_finite(name: &'static str, value: f64) -> Result<(), Dftd3Error> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(Dftd3Error::InvalidParameter { name, value })
    }
}

fn validate_nonnegative_finite(name: &'static str, value: f64) -> Result<(), Dftd3Error> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(Dftd3Error::InvalidParameter { name, value })
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Dftd3Error {
    #[error("DFTD3 species count {species} does not match position count {positions}")]
    SpeciesPositionMismatch { species: usize, positions: usize },

    #[error("invalid DFTD3 species label {0:?}")]
    InvalidSpecies(String),

    #[error("invalid DFTD3 parameter {name}: {value}")]
    InvalidParameter { name: &'static str, value: f64 },

    #[error("duplicate DFTD3 pair coefficient for {species_a:?}/{species_b:?}")]
    DuplicatePairCoefficient {
        species_a: String,
        species_b: String,
    },

    #[error("duplicate DFTD3 element coefficient for {species:?}")]
    DuplicateElementCoefficient { species: String },

    #[error("missing DFTD3 element coefficient for {species:?}")]
    UnknownElementCoefficient { species: String },

    #[error("missing DFTD3 pair coefficient for {species_a:?}/{species_b:?}")]
    UnknownPairCoefficient {
        species_a: String,
        species_b: String,
    },

    #[error("DFTD3 stress or virial output requires a periodic cell")]
    MissingCellForStress,

    #[error("DFTD3 supports only non-PBC or full 3D PBC structures, got pbc={pbc:?}")]
    UnsupportedPartialPbc { pbc: [bool; 3] },

    #[error(transparent)]
    NeighborList(#[from] NeighborListError),

    #[error(transparent)]
    ForceAccum(#[from] ForceAccumError),
}
