#![allow(clippy::expect_used, clippy::panic, clippy::print_stdout)]

use std::{env, fs, path::PathBuf};

use llmlip::{Cell, LlmlipModel, PredictionComputeDtype, PredictionOptions, Structure};
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize)]
struct Fixture {
    structures: Vec<FixtureStructure>,
}

#[derive(Debug, Deserialize)]
struct FixtureStructure {
    name: String,
    atomic_numbers: Vec<usize>,
    positions: Vec<[f64; 3]>,
    cell: Option<[[f64; 3]; 3]>,
    pbc: Option<[bool; 3]>,
}

#[derive(Debug)]
struct Args {
    model: PathBuf,
    fixture: PathBuf,
    structure: String,
    invariants_only: bool,
    num_layers: isize,
    compute_dtype: PredictionComputeDtype,
}

fn main() {
    let args = parse_args();
    let model = LlmlipModel::load(&args.model).unwrap_or_else(|err| {
        panic!("failed to load model {}: {err}", args.model.display());
    });
    let fixture_text = fs::read_to_string(&args.fixture).unwrap_or_else(|err| {
        panic!("failed to read fixture {}: {err}", args.fixture.display());
    });
    let fixture: Fixture = serde_json::from_str(&fixture_text).unwrap_or_else(|err| {
        panic!("failed to parse fixture {}: {err}", args.fixture.display());
    });
    let record = fixture
        .structures
        .iter()
        .find(|record| record.name == args.structure)
        .unwrap_or_else(|| panic!("structure {:?} not found in fixture", args.structure));
    let structure = structure_from_fixture(record);
    let descriptors = model
        .get_descriptors_with_compute_dtype(
            &structure,
            args.invariants_only,
            args.num_layers,
            args.compute_dtype,
        )
        .expect("failed to compute MACE descriptors");
    let atomic_energies = model
        .get_atomic_energies_with_compute_dtype(&structure, args.compute_dtype)
        .expect("failed to compute MACE atomic energies");
    let prediction = model
        .predict_with_compute_dtype(
            &structure,
            &PredictionOptions {
                compute_energy: true,
                ..PredictionOptions::default()
            },
            args.compute_dtype,
        )
        .expect("failed to compute MACE total energy");
    let total_energy = prediction.energy.expect("energy was requested");
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "model": args.model,
            "fixture": args.fixture,
            "structure": record.name,
            "natoms": structure.len(),
            "invariants_only": args.invariants_only,
            "num_layers": args.num_layers,
            "compute_dtype": compute_dtype_name(args.compute_dtype),
            "descriptors": descriptors,
            "atomic_energies": atomic_energies,
            "atomic_energies_sum": atomic_energies.iter().sum::<f64>(),
            "total_energy": total_energy,
        }))
        .expect("failed to serialize output")
    );
}

fn parse_args() -> Args {
    let mut raw = env::args().skip(1);
    let mut model = None;
    let mut fixture = None;
    let mut structure = None;
    let mut invariants_only = true;
    let mut num_layers = -1;
    let mut compute_dtype = PredictionComputeDtype::Float64;
    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "--model" => model = raw.next().map(PathBuf::from),
            "--fixture" => fixture = raw.next().map(PathBuf::from),
            "--structure" => structure = raw.next(),
            "--invariants-only" => invariants_only = parse_bool(next_value(&mut raw, &arg)),
            "--num-layers" => {
                num_layers = next_value(&mut raw, &arg)
                    .parse::<isize>()
                    .expect("--num-layers must be an integer");
            }
            "--compute-dtype" => compute_dtype = parse_compute_dtype(next_value(&mut raw, &arg)),
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => panic!("unknown argument {other:?}"),
        }
    }
    Args {
        model: model.unwrap_or_else(|| required("--model")),
        fixture: fixture.unwrap_or_else(|| required("--fixture")),
        structure: structure.unwrap_or_else(|| required("--structure")),
        invariants_only,
        num_layers,
        compute_dtype,
    }
}

fn next_value(raw: &mut impl Iterator<Item = String>, flag: &str) -> String {
    raw.next()
        .unwrap_or_else(|| panic!("missing value for {flag}"))
}

fn required(flag: &str) -> ! {
    print_usage();
    panic!("missing required argument {flag}");
}

fn print_usage() {
    eprintln!(
        "usage: mace_lite_extra_outputs --model PATH --fixture PATH --structure NAME [--invariants-only true|false] [--num-layers N] [--compute-dtype f64|f32]"
    );
}

fn parse_bool(value: String) -> bool {
    match value.as_str() {
        "1" | "true" | "True" | "TRUE" | "yes" | "on" => true,
        "0" | "false" | "False" | "FALSE" | "no" | "off" => false,
        other => panic!("invalid boolean value {other:?}"),
    }
}

fn parse_compute_dtype(value: String) -> PredictionComputeDtype {
    match value.as_str() {
        "f64" | "fp64" | "float64" | "Float64" => PredictionComputeDtype::Float64,
        "f32" | "fp32" | "float32" | "Float32" => PredictionComputeDtype::Float32,
        other => panic!("invalid compute dtype {other:?}"),
    }
}

fn compute_dtype_name(dtype: PredictionComputeDtype) -> &'static str {
    match dtype {
        PredictionComputeDtype::Float64 => "f64",
        PredictionComputeDtype::Float32 => "f32",
    }
}

fn structure_from_fixture(record: &FixtureStructure) -> Structure {
    let species = record
        .atomic_numbers
        .iter()
        .map(|atomic_number| atomic_symbol(*atomic_number).to_owned())
        .collect::<Vec<_>>();
    let mut structure = Structure::new(species, record.positions.clone());
    if let Some(cell) = record.cell {
        structure = structure.with_cell(
            Cell::from_vectors(cell[0], cell[1], cell[2]),
            record.pbc.unwrap_or([false; 3]),
        );
    } else if let Some(pbc) = record.pbc {
        structure.pbc = pbc;
    }
    structure
}

fn atomic_symbol(atomic_number: usize) -> &'static str {
    const SYMBOLS: [&str; 118] = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
        "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
        "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
        "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
        "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
        "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
        "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg",
        "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
    ];
    SYMBOLS
        .get(
            atomic_number
                .checked_sub(1)
                .expect("atomic number must be positive"),
        )
        .copied()
        .unwrap_or_else(|| panic!("unsupported atomic number {atomic_number}"))
}
