//! Minimal extXYZ I/O for CLI prediction tests and examples.

use std::{fs, io, path::Path};

use thiserror::Error;

use crate::{geometry::Cell, structure::Structure, Vec3};

#[derive(Debug, Error)]
pub enum ExtXyzError {
    #[error("I/O error for {path}: {source}")]
    Io { path: String, source: io::Error },

    #[error("invalid extXYZ: {0}")]
    Invalid(String),
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExtXyzFrameMetadata {
    pub total_charge: Option<f64>,
    pub spin_multiplicity: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct ExtXyzFrame {
    pub structure: Structure,
    pub metadata: ExtXyzFrameMetadata,
}

pub fn read_extxyz(path: impl AsRef<Path>) -> Result<Structure, ExtXyzError> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|source| ExtXyzError::Io {
        path: path.display().to_string(),
        source,
    })?;
    parse_extxyz(&text).map_err(|err| match err {
        ExtXyzError::Invalid(message) => invalid(format!("{}: {message}", path.display())),
        other => other,
    })
}

/// Read all frames from a multi-frame extXYZ file.
pub fn read_extxyz_multi(path: impl AsRef<Path>) -> Result<Vec<Structure>, ExtXyzError> {
    Ok(read_extxyz_frames(path)?
        .into_iter()
        .map(|frame| frame.structure)
        .collect())
}

/// Read all frames plus supported comment metadata from a multi-frame extXYZ file.
pub fn read_extxyz_frames(path: impl AsRef<Path>) -> Result<Vec<ExtXyzFrame>, ExtXyzError> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|source| ExtXyzError::Io {
        path: path.display().to_string(),
        source,
    })?;
    parse_extxyz_frames(&text).map_err(|err| match err {
        ExtXyzError::Invalid(message) => invalid(format!("{}: {message}", path.display())),
        other => other,
    })
}

/// Parse all frames from a multi-frame extXYZ string.
pub fn parse_extxyz_multi(text: &str) -> Result<Vec<Structure>, ExtXyzError> {
    Ok(parse_extxyz_frames(text)?
        .into_iter()
        .map(|frame| frame.structure)
        .collect())
}

/// Parse all frames plus supported comment metadata from a multi-frame extXYZ string.
pub fn parse_extxyz_frames(text: &str) -> Result<Vec<ExtXyzFrame>, ExtXyzError> {
    let mut structures = Vec::new();
    let mut remaining = text;
    while !remaining.trim().is_empty() {
        let (frame, consumed) = parse_extxyz_frame(remaining)?;
        structures.push(frame);
        remaining = &remaining[consumed..];
    }
    Ok(structures)
}

pub fn write_extxyz_with_forces(
    path: impl AsRef<Path>,
    structure: &Structure,
    forces: &[Vec3],
    energy: Option<f64>,
) -> Result<(), ExtXyzError> {
    if forces.len() != structure.len() {
        return Err(invalid(format!(
            "force count {} does not match atom count {}",
            forces.len(),
            structure.len()
        )));
    }

    let mut text = String::new();
    text.push_str(&format!("{}\n", structure.len()));
    let mut comment = String::from("Properties=species:S:1:pos:R:3:force:R:3");
    if let Some(cell) = structure.cell {
        comment.push_str(&format!(
            " Lattice=\"{} {} {} {} {} {} {} {} {}\"",
            cell.vectors[0][0],
            cell.vectors[0][1],
            cell.vectors[0][2],
            cell.vectors[1][0],
            cell.vectors[1][1],
            cell.vectors[1][2],
            cell.vectors[2][0],
            cell.vectors[2][1],
            cell.vectors[2][2],
        ));
    }
    comment.push_str(&format!(
        " pbc=\"{} {} {}\"",
        bool_to_extxyz(structure.pbc[0]),
        bool_to_extxyz(structure.pbc[1]),
        bool_to_extxyz(structure.pbc[2])
    ));
    if let Some(energy) = energy {
        comment.push_str(&format!(" energy={energy:.16e}"));
    }
    text.push_str(&comment);
    text.push('\n');

    for ((species, position), force) in structure
        .species
        .iter()
        .zip(structure.positions.iter())
        .zip(forces.iter())
    {
        text.push_str(&format!(
            "{} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e}\n",
            species, position[0], position[1], position[2], force[0], force[1], force[2]
        ));
    }

    fs::write(path.as_ref(), text).map_err(|source| ExtXyzError::Io {
        path: path.as_ref().display().to_string(),
        source,
    })
}

pub fn parse_extxyz(text: &str) -> Result<Structure, ExtXyzError> {
    let (frame, _consumed) = parse_extxyz_frame(text)?;
    Ok(frame.structure)
}

/// Parse a single extXYZ frame and return the structure plus consumed byte count.
///
/// Byte tracking uses `\n` positions in the raw text to handle both LF and CRLF.
fn parse_extxyz_frame(text: &str) -> Result<(ExtXyzFrame, usize), ExtXyzError> {
    let remaining = text;

    let (atom_count_line, rest) = next_line(remaining, "missing atom count line")?;
    let atom_count = atom_count_line
        .trim()
        .parse::<usize>()
        .map_err(|_| invalid(format!("invalid atom count {atom_count_line:?}")))?;

    let (comment, rest) = next_line(rest, "missing comment line")?;

    let metadata = parse_frame_metadata(comment)?;
    let cell = parse_lattice(comment)?;
    let pbc = parse_pbc(comment)?;
    if pbc.iter().any(|value| *value) && cell.is_none() {
        return Err(invalid("pbc=true requires Lattice metadata"));
    }
    let layout = parse_properties(comment)?;

    let mut species = Vec::with_capacity(atom_count);
    let mut positions = Vec::with_capacity(atom_count);
    let mut rest = rest;
    for atom_index in 0..atom_count {
        let (line, next_rest) = next_line(rest, &format!("missing atom line {atom_index}"))?;
        rest = next_rest;
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() < layout.min_fields {
            return Err(invalid(format!(
                "atom line {atom_index} has {} fields but Properties requires at least {}",
                fields.len(),
                layout.min_fields
            )));
        }
        species.push(fields[layout.species].to_owned());
        positions.push([
            parse_f64(fields[layout.pos[0]], "x", atom_index)?,
            parse_f64(fields[layout.pos[1]], "y", atom_index)?,
            parse_f64(fields[layout.pos[2]], "z", atom_index)?,
        ]);
    }

    let consumed = text.len() - rest.len();
    let mut structure = Structure::new(species, positions);
    structure.cell = cell;
    structure.pbc = pbc;
    Ok((
        ExtXyzFrame {
            structure,
            metadata,
        },
        consumed,
    ))
}

/// Split `text` at the first `\n`, returning the line content (without `\r\n`
/// or `\n`) and the remainder after the newline. If there is no newline, the
/// entire text is consumed as the line.
fn next_line<'a>(text: &'a str, missing_msg: &str) -> Result<(&'a str, &'a str), ExtXyzError> {
    if text.is_empty() {
        return Err(invalid(missing_msg));
    }
    match text.find('\n') {
        Some(pos) => {
            let line = &text[..pos];
            let line = line.strip_suffix('\r').unwrap_or(line);
            Ok((line, &text[pos + 1..]))
        }
        None => {
            let line = text.strip_suffix('\r').unwrap_or(text);
            Ok((line, &text[text.len()..]))
        }
    }
}

fn parse_lattice(comment: &str) -> Result<Option<Cell>, ExtXyzError> {
    let Some(value) = metadata_value(comment, "Lattice") else {
        return Ok(None);
    };
    let values = value
        .split_whitespace()
        .map(|field| parse_finite(field, "Lattice value"))
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() != 9 {
        return Err(invalid(format!(
            "Lattice must contain 9 values, got {}",
            values.len()
        )));
    }
    Ok(Some(Cell::new([
        [values[0], values[1], values[2]],
        [values[3], values[4], values[5]],
        [values[6], values[7], values[8]],
    ])))
}

fn parse_pbc(comment: &str) -> Result<[bool; 3], ExtXyzError> {
    let Some(value) = metadata_value(comment, "pbc") else {
        return Ok([false; 3]);
    };
    let fields = value.split_whitespace().collect::<Vec<_>>();
    if fields.len() != 3 {
        return Err(invalid(format!(
            "pbc must contain 3 values, got {}",
            fields.len()
        )));
    }
    Ok([
        parse_bool(fields[0])?,
        parse_bool(fields[1])?,
        parse_bool(fields[2])?,
    ])
}

fn parse_frame_metadata(comment: &str) -> Result<ExtXyzFrameMetadata, ExtXyzError> {
    Ok(ExtXyzFrameMetadata {
        total_charge: parse_first_metadata_f64(comment, &["total_charge", "charge"])?,
        spin_multiplicity: parse_first_metadata_f64(comment, &["spin_multiplicity", "spin"])?,
    })
}

#[derive(Clone, Copy, Debug)]
struct PropertyLayout {
    species: usize,
    pos: [usize; 3],
    min_fields: usize,
}

fn parse_properties(comment: &str) -> Result<PropertyLayout, ExtXyzError> {
    let Some(value) = metadata_value(comment, "Properties") else {
        return Ok(PropertyLayout {
            species: 0,
            pos: [1, 2, 3],
            min_fields: 4,
        });
    };

    let parts = value.split(':').collect::<Vec<_>>();
    if parts.len() % 3 != 0 {
        return Err(invalid("Properties must be name:type:count triples"));
    }

    let mut species = None;
    let mut pos = None;
    let mut column = 0usize;
    for chunk in parts.chunks_exact(3) {
        let name = chunk[0];
        let kind = chunk[1];
        let count = chunk[2]
            .parse::<usize>()
            .map_err(|_| invalid(format!("invalid Properties count {:?}", chunk[2])))?;
        if count == 0 {
            return Err(invalid(format!("Properties field {name:?} has zero count")));
        }
        match name {
            "species" => {
                if kind != "S" {
                    return Err(invalid("Properties species field must have type S"));
                }
                if count != 1 {
                    return Err(invalid("Properties species field must have count 1"));
                }
                species = Some(column);
            }
            "pos" => {
                if kind != "R" {
                    return Err(invalid("Properties pos field must have type R"));
                }
                if count != 3 {
                    return Err(invalid("Properties pos field must have count 3"));
                }
                pos = Some([column, column + 1, column + 2]);
            }
            _ => {}
        }
        column = column
            .checked_add(count)
            .ok_or_else(|| invalid("Properties column count overflows"))?;
    }

    Ok(PropertyLayout {
        species: species.ok_or_else(|| invalid("Properties must include species:S:1"))?,
        pos: pos.ok_or_else(|| invalid("Properties must include pos:R:3"))?,
        min_fields: column,
    })
}

fn parse_first_metadata_f64(comment: &str, keys: &[&str]) -> Result<Option<f64>, ExtXyzError> {
    for key in keys {
        if let Some(value) = metadata_value(comment, key) {
            return Ok(Some(parse_finite(value, &format!("metadata {key}"))?));
        }
    }
    Ok(None)
}

fn metadata_value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let (rest, quoted) = metadata_value_start(text, key)?;
    if quoted {
        let end = rest.find('"')?;
        Some(&rest[..end])
    } else {
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        Some(&rest[..end])
    }
}

fn metadata_value_start<'a>(text: &'a str, key: &str) -> Option<(&'a str, bool)> {
    let mut search_start = 0usize;
    while search_start < text.len() {
        let relative = text[search_start..].find(key)?;
        let start = search_start + relative;
        let before_ok = start == 0 || text.as_bytes()[start - 1].is_ascii_whitespace();
        let after_key = start + key.len();
        let rest = &text[after_key..];
        if before_ok {
            if let Some(value) = rest.strip_prefix("=\"") {
                return Some((value, true));
            }
            if let Some(value) = rest.strip_prefix('=') {
                return Some((value, false));
            }
        }
        search_start = after_key;
    }
    None
}

fn parse_bool(value: &str) -> Result<bool, ExtXyzError> {
    match value {
        "T" | "t" | "True" | "true" | "1" => Ok(true),
        "F" | "f" | "False" | "false" | "0" => Ok(false),
        other => Err(invalid(format!("invalid pbc boolean {other:?}"))),
    }
}

fn parse_f64(value: &str, axis: &str, atom_index: usize) -> Result<f64, ExtXyzError> {
    parse_finite(value, &format!("atom {atom_index} {axis} coordinate"))
}

fn parse_finite(value: &str, label: &str) -> Result<f64, ExtXyzError> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| invalid(format!("invalid {label} {value:?}")))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(invalid(format!("{label} must be finite, got {value:?}")))
    }
}

fn bool_to_extxyz(value: bool) -> &'static str {
    if value {
        "T"
    } else {
        "F"
    }
}

fn invalid(message: impl Into<String>) -> ExtXyzError {
    ExtXyzError::Invalid(message.into())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_extxyz_with_lattice_and_pbc() {
        let structure = parse_extxyz(
            "2\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3 pbc=\"T F F\"\nH 0 0.5 1\nO 1 2 3\n",
        )
        .unwrap();

        assert_eq!(structure.species, vec!["H", "O"]);
        assert_eq!(structure.positions, vec![[0.0, 0.5, 1.0], [1.0, 2.0, 3.0]]);
        assert_eq!(structure.pbc, [true, false, false]);
        assert_eq!(structure.cell.unwrap().vectors[0], [10.0, 0.0, 0.0]);
    }

    #[test]
    fn parses_non_periodic_extxyz_without_lattice() {
        let structure = parse_extxyz("1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n").unwrap();

        assert_eq!(structure.species, vec!["H"]);
        assert_eq!(structure.positions, vec![[0.0, 0.0, 0.0]]);
        assert_eq!(structure.pbc, [false; 3]);
        assert!(structure.cell.is_none());
    }

    #[test]
    fn parses_reordered_properties_with_extra_columns() {
        let structure =
            parse_extxyz("1\nProperties=charge:R:1:species:S:1:pos:R:3\n0.5 O 1 2 3\n").unwrap();

        assert_eq!(structure.species, vec!["O"]);
        assert_eq!(structure.positions, vec![[1.0, 2.0, 3.0]]);
    }

    #[test]
    fn rejects_non_finite_coordinates() {
        let err = parse_extxyz("1\nProperties=species:S:1:pos:R:3\nH NaN 0 0\n")
            .expect_err("non-finite coordinate should fail");

        assert!(matches!(err, ExtXyzError::Invalid(_)));
    }

    #[test]
    fn rejects_non_finite_lattice_values() {
        let err = parse_extxyz(
            "1\nLattice=\"inf 0 0 0 1 0 0 0 1\" Properties=species:S:1:pos:R:3\nH 0 0 0\n",
        )
        .expect_err("non-finite lattice should fail");

        assert!(matches!(err, ExtXyzError::Invalid(_)));
    }

    #[test]
    fn rejects_properties_without_pos() {
        let err = parse_extxyz("1\nProperties=species:S:1:charge:R:1\nH 0.5\n")
            .expect_err("missing pos property should fail");

        assert!(matches!(err, ExtXyzError::Invalid(_)));
    }

    #[test]
    fn rejects_bad_atom_count() {
        assert!(matches!(
            parse_extxyz("x\ncomment\n"),
            Err(ExtXyzError::Invalid(_))
        ));
    }

    #[test]
    fn parses_multi_frame_extxyz() {
        let text = concat!(
            "1\nProperties=species:S:1:pos:R:3 pbc=\"F F F\"\nH 0 0 0\n",
            "2\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n",
            "O 1 2 3\nFe 4 5 6\n",
        );
        let structures = parse_extxyz_multi(text).unwrap();
        assert_eq!(structures.len(), 2);
        assert_eq!(structures[0].species, vec!["H"]);
        assert_eq!(structures[0].pbc, [false, false, false]);
        assert!(structures[0].cell.is_none());
        assert_eq!(structures[1].species, vec!["O", "Fe"]);
        assert_eq!(structures[1].pbc, [true, true, true]);
        assert!(structures[1].cell.is_some());
    }

    #[test]
    fn parses_frame_charge_spin_metadata() {
        let text = concat!(
            "1\nProperties=species:S:1:pos:R:3 total_charge=-1 spin_multiplicity=2\nH 0 0 0\n",
            "1\nProperties=species:S:1:pos:R:3 charge=\"0\" spin=\"1\"\nO 1 1 1\n",
        );
        let frames = parse_extxyz_frames(text).unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].metadata.total_charge, Some(-1.0));
        assert_eq!(frames[0].metadata.spin_multiplicity, Some(2.0));
        assert_eq!(frames[1].metadata.total_charge, Some(0.0));
        assert_eq!(frames[1].metadata.spin_multiplicity, Some(1.0));
    }

    #[test]
    fn metadata_key_matching_requires_token_boundary() {
        assert_eq!(metadata_value("total_charge=-1", "charge"), None);
        assert_eq!(
            metadata_value("prefix_total_charge=-1 total_charge=2", "total_charge"),
            Some("2")
        );
    }

    #[test]
    fn rejects_non_finite_frame_charge_spin_metadata() {
        let err = parse_extxyz_frames(
            "1\nProperties=species:S:1:pos:R:3 total_charge=NaN spin_multiplicity=1\nH 0 0 0\n",
        )
        .expect_err("non-finite metadata should fail");

        assert!(matches!(err, ExtXyzError::Invalid(_)));
    }

    #[test]
    fn parses_empty_multi_frame_extxyz() {
        let structures = parse_extxyz_multi("").unwrap();
        assert!(structures.is_empty());
    }

    #[test]
    fn parses_single_frame_as_multi() {
        let text = "1\nProperties=species:S:1:pos:R:3\nH 1 2 3\n";
        let structures = parse_extxyz_multi(text).unwrap();
        assert_eq!(structures.len(), 1);
        assert_eq!(structures[0].positions, vec![[1.0, 2.0, 3.0]]);
    }

    #[test]
    fn multi_frame_without_trailing_newline() {
        let text = "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n1\nProperties=species:S:1:pos:R:3\nO 1 1 1";
        let structures = parse_extxyz_multi(text).unwrap();
        assert_eq!(structures.len(), 2);
        assert_eq!(structures[0].species, vec!["H"]);
        assert_eq!(structures[1].species, vec!["O"]);
    }

    #[test]
    fn parses_multi_frame_with_crlf_line_endings() {
        let text = "1\r\nProperties=species:S:1:pos:R:3\r\nH 0 0 0\r\n2\r\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"\r\nO 1 2 3\r\nFe 4 5 6\r\n";
        let structures = parse_extxyz_multi(text).unwrap();
        assert_eq!(structures.len(), 2);
        assert_eq!(structures[0].species, vec!["H"]);
        assert_eq!(structures[0].positions, vec![[0.0, 0.0, 0.0]]);
        assert_eq!(structures[1].species, vec!["O", "Fe"]);
        assert_eq!(structures[1].pbc, [true, true, true]);
    }

    #[test]
    fn parses_single_frame_with_crlf() {
        let structure = parse_extxyz(
            "2\r\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3 pbc=\"T F F\"\r\nH 0 0.5 1\r\nO 1 2 3\r\n",
        )
        .unwrap();
        assert_eq!(structure.species, vec!["H", "O"]);
        assert_eq!(structure.positions, vec![[0.0, 0.5, 1.0], [1.0, 2.0, 3.0]]);
        assert_eq!(structure.pbc, [true, false, false]);
    }
}
