//! UMA テンソル単一ファイルパック形式。
//!
//! ## 概要
//! `.e3rs` 配下のテンソルを 1 つの `tensors.pack` にまとめ、
//! 実行時は安全な aligned word buffer から参照する。pack は UMA export 時に 1 回生成する。
//!
//! ## 主な仕様
//! - ファイル名: モデルルート直下の `tensors.pack`
//! - エンディアン: リトルエンディアン
//! - データ部は manifest の `[[tensors]]` 順に raw LE バイトを連結
//!
//! ## 制限事項
//! - UMA 向け（f64 / i64 のみ）

#[cfg(target_endian = "big")]
compile_error!("UMA tensor pack currently supports little-endian targets only");

use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

const PACK_MAGIC: &[u8; 10] = b"LLMLIPMPK1";
const PACK_VERSION: u32 = 1;
const PACK_HEADER_START: usize = 26;
const MAX_TENSOR_RANK: usize = 16;

const DTYPE_TAG_F64: u8 = 1;
const DTYPE_TAG_I64: u8 = 2;

/// モデルルート直下に置く単一テンソル blob のファイル名。
pub const UMA_TENSOR_PACK_FILENAME: &str = "tensors.pack";

/// pack 生成時に参照する 1 テンソル分のメタデータ。
#[derive(Clone, Debug)]
pub struct UmaTensorPackSourceSpec {
    /// manifest 上のテンソル名。
    pub name: String,
    /// `f64` または `i64`。
    pub dtype: String,
    /// row-major 形状。
    pub shape: Vec<usize>,
    /// モデルルートからの相対パス（個別 `.bin`）。
    pub path: PathBuf,
    /// 期待バイト数（shape × dtype size）。
    pub expected_bytes: usize,
}

/// pack データ部の 1 テンソル位置。
#[derive(Clone, Debug)]
pub struct UmaPackedTensorEntry {
    /// `data_offset` からのバイトオフセット。
    pub byte_offset: usize,
    /// 生バイト長。
    pub byte_len: usize,
}

/// pack 読み込み結果。
#[derive(Clone, Debug)]
pub struct UmaTensorPack {
    /// pack 全体を 8-byte aligned storage に読み込んだもの。
    pub words: Arc<Vec<u64>>,
    /// `words` 中で有効な元ファイルバイト数。
    pub len_bytes: usize,
    /// データ部の先頭バイトオフセット。
    pub data_offset: usize,
    /// テンソル名 → pack エントリ。
    pub entries: std::collections::HashMap<String, UmaPackedTensorEntry>,
    /// データ部合計バイト数。
    pub total_bytes: usize,
}

impl UmaTensorPack {
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &bytemuck::cast_slice(self.words.as_slice())[..self.len_bytes]
    }

    pub fn entry_bytes<'a>(
        &'a self,
        name: &str,
        entry: &UmaPackedTensorEntry,
    ) -> Result<&'a [u8], UmaTensorPackError> {
        let start = self
            .data_offset
            .checked_add(entry.byte_offset)
            .ok_or_else(|| UmaTensorPackError::tensor(name, "absolute byte offset overflows"))?;
        let end = start
            .checked_add(entry.byte_len)
            .ok_or_else(|| UmaTensorPackError::tensor(name, "absolute byte range overflows"))?;
        self.as_bytes()
            .get(start..end)
            .ok_or_else(|| UmaTensorPackError::tensor(name, "byte range is outside pack data"))
    }
}

/// pack 関連エラー。
#[derive(Debug, thiserror::Error)]
pub enum UmaTensorPackError {
    #[error("UMA tensor pack I/O at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("UMA tensor pack format error: {0}")]
    Format(String),
    #[error("UMA tensor pack tensor {name}: {detail}")]
    Tensor { name: String, detail: String },
}

impl UmaTensorPackError {
    fn format(message: impl Into<String>) -> Self {
        Self::Format(message.into())
    }

    fn tensor(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::Tensor {
            name: name.into(),
            detail: detail.into(),
        }
    }
}

/// モデルルートに `tensors.pack` が存在するか。
#[must_use]
pub fn uma_tensor_pack_path(root: &Path) -> PathBuf {
    root.join(UMA_TENSOR_PACK_FILENAME)
}

#[must_use]
pub fn uma_tensor_pack_exists(root: &Path) -> bool {
    uma_tensor_pack_path(root).is_file()
}

/// 個別 `.bin` から `tensors.pack` を生成する。
pub fn write_uma_tensor_pack(
    output_path: &Path,
    specs: &[UmaTensorPackSourceSpec],
) -> Result<(), UmaTensorPackError> {
    if specs.is_empty() {
        return Err(UmaTensorPackError::format(
            "UMA tensor pack requires at least one tensor spec",
        ));
    }
    for spec in specs {
        validate_pack_source_spec(spec)?;
    }

    let mut index_bytes = Vec::new();
    let mut data_bytes = Vec::new();
    for spec in specs {
        let source_path = output_path
            .parent()
            .ok_or_else(|| {
                UmaTensorPackError::format(format!(
                    "UMA tensor pack output {output_path:?} has no parent directory"
                ))
            })?
            .join(&spec.path);
        let mut file_bytes = Vec::new();
        File::open(&source_path)
            .and_then(|mut file| file.read_to_end(&mut file_bytes))
            .map_err(|source| UmaTensorPackError::Io {
                path: source_path.clone(),
                source,
            })?;
        if file_bytes.len() != spec.expected_bytes {
            return Err(UmaTensorPackError::tensor(
                &spec.name,
                format!(
                    "byte size mismatch: expected {}, got {}",
                    spec.expected_bytes,
                    file_bytes.len()
                ),
            ));
        }

        let data_rel_offset = u64::try_from(data_bytes.len()).map_err(|_| {
            UmaTensorPackError::tensor(&spec.name, "data section offset overflows u64".to_owned())
        })?;
        let byte_len = u64::try_from(file_bytes.len()).map_err(|_| {
            UmaTensorPackError::tensor(&spec.name, "byte length overflows u64".to_owned())
        })?;

        write_index_entry(&mut index_bytes, spec, data_rel_offset, byte_len)?;
        data_bytes.extend_from_slice(&file_bytes);
    }

    let index_end = u64::try_from(PACK_HEADER_START)
        .ok()
        .and_then(|start| start.checked_add(u64::try_from(index_bytes.len()).ok()?))
        .ok_or_else(|| UmaTensorPackError::format("UMA tensor pack index end overflows u64"))?;
    let data_offset = align_u64(index_end, 8)?;
    let padding_len = usize::try_from(data_offset)
        .map_err(|_| UmaTensorPackError::format("data offset overflows usize"))?
        - PACK_HEADER_START
        - index_bytes.len();
    let mut file_bytes =
        Vec::with_capacity(PACK_HEADER_START + index_bytes.len() + padding_len + data_bytes.len());
    file_bytes.extend_from_slice(PACK_MAGIC);
    file_bytes.extend_from_slice(&PACK_VERSION.to_le_bytes());
    file_bytes.extend_from_slice(
        &(u32::try_from(specs.len())
            .map_err(|_| UmaTensorPackError::format("tensor count overflows u32"))?)
        .to_le_bytes(),
    );
    file_bytes.extend_from_slice(&data_offset.to_le_bytes());
    file_bytes.extend_from_slice(&index_bytes);
    file_bytes.extend(std::iter::repeat_n(0u8, padding_len));
    debug_assert_eq!(u64::try_from(file_bytes.len()).ok(), Some(data_offset));
    file_bytes.extend_from_slice(&data_bytes);

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|source| UmaTensorPackError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let temp_path = temporary_pack_path(output_path);
    let _ = fs::remove_file(&temp_path);
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)
        .map_err(|source| UmaTensorPackError::Io {
            path: temp_path.clone(),
            source,
        })?;
    output
        .write_all(&file_bytes)
        .map_err(|source| UmaTensorPackError::Io {
            path: temp_path.clone(),
            source,
        })?;
    output.sync_all().map_err(|source| UmaTensorPackError::Io {
        path: temp_path.clone(),
        source,
    })?;
    drop(output);
    fs::rename(&temp_path, output_path).map_err(|source| UmaTensorPackError::Io {
        path: output_path.to_path_buf(),
        source,
    })?;
    sync_parent_dir(output_path)?;
    Ok(())
}

/// `tensors.pack` を安全な aligned storage に読み、manifest 仕様と突合したエントリ表を返す。
pub fn load_uma_tensor_pack(
    pack_path: &Path,
    specs: &[UmaTensorPackSourceSpec],
) -> Result<UmaTensorPack, UmaTensorPackError> {
    let (words, len_bytes) = read_pack_words(pack_path)?;
    parse_uma_tensor_pack(words, len_bytes, specs)
}

fn parse_uma_tensor_pack(
    words: Arc<Vec<u64>>,
    len_bytes: usize,
    specs: &[UmaTensorPackSourceSpec],
) -> Result<UmaTensorPack, UmaTensorPackError> {
    let bytes = &bytemuck::cast_slice(words.as_slice())[..len_bytes];
    if bytes.len() < PACK_HEADER_START {
        return Err(UmaTensorPackError::format(format!(
            "UMA tensor pack {UMA_TENSOR_PACK_FILENAME} is too small ({} bytes)",
            bytes.len()
        )));
    }
    if &bytes[..PACK_MAGIC.len()] != PACK_MAGIC {
        return Err(UmaTensorPackError::format(format!(
            "UMA tensor pack magic mismatch in {UMA_TENSOR_PACK_FILENAME}"
        )));
    }
    let version = read_u32_le(bytes, 10)?;
    if version != PACK_VERSION {
        return Err(UmaTensorPackError::format(format!(
            "unsupported UMA tensor pack version {version}; expected {PACK_VERSION}"
        )));
    }
    let tensor_count = read_u32_le(bytes, 14)? as usize;
    let data_offset = read_u64_le(bytes, 18)?;
    let data_offset_usize = usize::try_from(data_offset)
        .map_err(|_| UmaTensorPackError::format("UMA tensor pack data_offset overflows usize"))?;
    if !data_offset_usize.is_multiple_of(8) {
        return Err(UmaTensorPackError::format(format!(
            "UMA tensor pack data_offset {data_offset_usize} is not 8-byte aligned"
        )));
    }
    if data_offset_usize > bytes.len() {
        return Err(UmaTensorPackError::format(format!(
            "UMA tensor pack data_offset {data_offset_usize} exceeds file size {}",
            bytes.len()
        )));
    }
    if tensor_count != specs.len() {
        return Err(UmaTensorPackError::format(format!(
            "UMA tensor pack tensor count {tensor_count} does not match manifest count {}",
            specs.len()
        )));
    }

    let mut cursor = PACK_HEADER_START;
    let mut entries = std::collections::HashMap::with_capacity(tensor_count);
    let mut total_bytes = 0usize;
    for spec in specs {
        let (entry, next_cursor) = read_index_entry(bytes, cursor, spec, total_bytes)?;
        cursor = next_cursor;
        if entries.insert(spec.name.clone(), entry).is_some() {
            return Err(UmaTensorPackError::tensor(
                &spec.name,
                "duplicate tensor name in pack index".to_owned(),
            ));
        }
        total_bytes = total_bytes
            .checked_add(spec.expected_bytes)
            .ok_or_else(|| UmaTensorPackError::format("UMA tensor pack total bytes overflow"))?;
    }
    if cursor > data_offset_usize {
        return Err(UmaTensorPackError::format(format!(
            "UMA tensor pack index extends past data section (index_end={cursor}, data_offset={data_offset_usize})"
        )));
    }

    let data_end = data_offset_usize
        .checked_add(total_bytes)
        .ok_or_else(|| UmaTensorPackError::format("UMA tensor pack data end overflow"))?;
    if data_end > bytes.len() {
        return Err(UmaTensorPackError::format(format!(
            "UMA tensor pack data section exceeds file size (need {data_end}, have {})",
            bytes.len()
        )));
    }

    Ok(UmaTensorPack {
        words,
        len_bytes,
        data_offset: data_offset_usize,
        entries,
        total_bytes,
    })
}

fn write_index_entry(
    out: &mut Vec<u8>,
    spec: &UmaTensorPackSourceSpec,
    data_rel_offset: u64,
    byte_len: u64,
) -> Result<(), UmaTensorPackError> {
    let name_bytes = spec.name.as_bytes();
    let name_len = u32::try_from(name_bytes.len())
        .map_err(|_| UmaTensorPackError::tensor(&spec.name, "name length overflows u32"))?;
    out.extend_from_slice(&name_len.to_le_bytes());
    out.extend_from_slice(name_bytes);
    out.push(dtype_tag(&spec.dtype)?);
    let shape_rank = u32::try_from(spec.shape.len())
        .map_err(|_| UmaTensorPackError::tensor(&spec.name, "shape rank overflows u32"))?;
    out.extend_from_slice(&shape_rank.to_le_bytes());
    for dim in &spec.shape {
        out.extend_from_slice(
            &u64::try_from(*dim)
                .map_err(|_| {
                    UmaTensorPackError::tensor(&spec.name, "shape dimension overflows u64")
                })?
                .to_le_bytes(),
        );
    }
    out.extend_from_slice(&data_rel_offset.to_le_bytes());
    out.extend_from_slice(&byte_len.to_le_bytes());
    Ok(())
}

fn read_index_entry(
    bytes: &[u8],
    cursor: usize,
    expected: &UmaTensorPackSourceSpec,
    expected_data_rel_offset: usize,
) -> Result<(UmaPackedTensorEntry, usize), UmaTensorPackError> {
    let mut offset = cursor;
    let name_len = read_u32_le(bytes, offset)? as usize;
    offset += 4;
    let name_end = offset
        .checked_add(name_len)
        .ok_or_else(|| UmaTensorPackError::format("UMA tensor pack name end overflow"))?;
    if name_end > bytes.len() {
        return Err(UmaTensorPackError::format(
            "UMA tensor pack index truncated while reading tensor name",
        ));
    }
    let name = std::str::from_utf8(&bytes[offset..name_end])
        .map_err(|err| UmaTensorPackError::format(format!("UMA tensor pack name utf8: {err}")))?
        .to_owned();
    offset = name_end;
    if name != expected.name {
        return Err(UmaTensorPackError::tensor(
            &expected.name,
            format!(
                "pack index name mismatch: expected {:?}, got {name:?}",
                expected.name
            ),
        ));
    }

    let dtype_tag = *bytes.get(offset).ok_or_else(|| {
        UmaTensorPackError::format("UMA tensor pack index truncated while reading dtype tag")
    })?;
    offset += 1;
    let dtype = dtype_from_tag(dtype_tag)?;
    if dtype != expected.dtype {
        return Err(UmaTensorPackError::tensor(
            &name,
            format!(
                "dtype mismatch: manifest {:?}, pack {dtype:?}",
                expected.dtype
            ),
        ));
    }

    let shape_rank = read_u32_le(bytes, offset)? as usize;
    offset += 4;
    if shape_rank > MAX_TENSOR_RANK {
        return Err(UmaTensorPackError::tensor(
            &name,
            format!("shape rank {shape_rank} exceeds maximum {MAX_TENSOR_RANK}"),
        ));
    }
    let mut shape = Vec::with_capacity(shape_rank);
    for _ in 0..shape_rank {
        let dim = read_u64_le(bytes, offset)?;
        offset += 8;
        let dim = usize::try_from(dim)
            .map_err(|_| UmaTensorPackError::tensor(&name, "shape dimension overflows usize"))?;
        shape.push(dim);
    }
    if shape != expected.shape {
        return Err(UmaTensorPackError::tensor(
            &name,
            format!(
                "shape mismatch: manifest {:?}, pack {shape:?}",
                expected.shape
            ),
        ));
    }

    let data_rel_offset = read_u64_le(bytes, offset)?;
    offset += 8;
    let byte_len = read_u64_le(bytes, offset)?;
    offset += 8;
    let data_rel_offset_usize = usize::try_from(data_rel_offset)
        .map_err(|_| UmaTensorPackError::tensor(&name, "data_rel_offset overflows usize"))?;
    if data_rel_offset_usize != expected_data_rel_offset {
        return Err(UmaTensorPackError::tensor(
            &name,
            format!(
                "data_rel_offset must be contiguous at {expected_data_rel_offset}, got {data_rel_offset_usize}"
            ),
        ));
    }
    if !data_rel_offset_usize.is_multiple_of(8) {
        return Err(UmaTensorPackError::tensor(
            &name,
            format!("data_rel_offset {data_rel_offset_usize} is not 8-byte aligned"),
        ));
    }
    let byte_len_usize = usize::try_from(byte_len)
        .map_err(|_| UmaTensorPackError::tensor(&name, "byte_len overflows usize"))?;
    if !byte_len_usize.is_multiple_of(8) {
        return Err(UmaTensorPackError::tensor(
            &name,
            format!("byte_len {byte_len_usize} is not 8-byte aligned"),
        ));
    }
    if byte_len_usize != expected.expected_bytes {
        return Err(UmaTensorPackError::tensor(
            &name,
            format!(
                "byte_len mismatch: manifest {}, pack {byte_len_usize}",
                expected.expected_bytes
            ),
        ));
    }

    Ok((
        UmaPackedTensorEntry {
            byte_offset: data_rel_offset_usize,
            byte_len: byte_len_usize,
        },
        offset,
    ))
}

fn dtype_tag(dtype: &str) -> Result<u8, UmaTensorPackError> {
    match dtype {
        "f64" => Ok(DTYPE_TAG_F64),
        "i64" => Ok(DTYPE_TAG_I64),
        other => Err(UmaTensorPackError::tensor(
            "unknown",
            format!("unsupported dtype {other:?} for UMA tensor pack"),
        )),
    }
}

fn dtype_from_tag(tag: u8) -> Result<String, UmaTensorPackError> {
    match tag {
        DTYPE_TAG_F64 => Ok("f64".to_owned()),
        DTYPE_TAG_I64 => Ok("i64".to_owned()),
        other => Err(UmaTensorPackError::format(format!(
            "unsupported UMA tensor pack dtype tag {other}"
        ))),
    }
}

fn validate_pack_source_spec(spec: &UmaTensorPackSourceSpec) -> Result<(), UmaTensorPackError> {
    let _ = dtype_tag(&spec.dtype)?;
    if spec.shape.len() > MAX_TENSOR_RANK {
        return Err(UmaTensorPackError::tensor(
            &spec.name,
            format!(
                "shape rank {} exceeds maximum {MAX_TENSOR_RANK}",
                spec.shape.len()
            ),
        ));
    }
    if !spec.expected_bytes.is_multiple_of(8) {
        return Err(UmaTensorPackError::tensor(
            &spec.name,
            format!(
                "expected byte length {} is not 8-byte aligned",
                spec.expected_bytes
            ),
        ));
    }
    Ok(())
}

fn temporary_pack_path(output_path: &Path) -> PathBuf {
    let filename = output_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(UMA_TENSOR_PACK_FILENAME);
    output_path.with_file_name(format!(".{filename}.tmp-{}", std::process::id()))
}

fn sync_parent_dir(output_path: &Path) -> Result<(), UmaTensorPackError> {
    #[cfg(unix)]
    {
        if let Some(parent) = output_path.parent() {
            let dir = File::open(parent).map_err(|source| UmaTensorPackError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
            dir.sync_all().map_err(|source| UmaTensorPackError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
    }
    Ok(())
}

fn read_pack_words(pack_path: &Path) -> Result<(Arc<Vec<u64>>, usize), UmaTensorPackError> {
    let len_bytes = fs::metadata(pack_path)
        .map_err(|source| UmaTensorPackError::Io {
            path: pack_path.to_path_buf(),
            source,
        })?
        .len();
    let len_bytes = usize::try_from(len_bytes).map_err(|_| {
        UmaTensorPackError::format(format!(
            "UMA tensor pack {} is too large for this platform",
            pack_path.display()
        ))
    })?;
    let word_count = len_bytes
        .checked_add(7)
        .ok_or_else(|| UmaTensorPackError::format("UMA tensor pack byte length overflow"))?
        / 8;
    let mut words = vec![0_u64; word_count];
    let mut file = File::open(pack_path).map_err(|source| UmaTensorPackError::Io {
        path: pack_path.to_path_buf(),
        source,
    })?;
    file.read_exact(&mut bytemuck::cast_slice_mut(words.as_mut_slice())[..len_bytes])
        .map_err(|source| UmaTensorPackError::Io {
            path: pack_path.to_path_buf(),
            source,
        })?;
    Ok((Arc::new(words), len_bytes))
}

fn align_u64(value: u64, alignment: u64) -> Result<u64, UmaTensorPackError> {
    if alignment == 0 {
        return Err(UmaTensorPackError::format(
            "UMA tensor pack alignment must be nonzero",
        ));
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| UmaTensorPackError::format("UMA tensor pack alignment overflow"))
    }
}

fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32, UmaTensorPackError> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| UmaTensorPackError::format("u32 read offset overflow"))?;
    let chunk = bytes
        .get(offset..end)
        .ok_or_else(|| UmaTensorPackError::format(format!("truncated u32 at offset {offset}")))?;
    let mut array = [0u8; 4];
    array.copy_from_slice(chunk);
    Ok(u32::from_le_bytes(array))
}

fn read_u64_le(bytes: &[u8], offset: usize) -> Result<u64, UmaTensorPackError> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| UmaTensorPackError::format("u64 read offset overflow"))?;
    let chunk = bytes
        .get(offset..end)
        .ok_or_else(|| UmaTensorPackError::format(format!("truncated u64 at offset {offset}")))?;
    let mut array = [0u8; 8];
    array.copy_from_slice(chunk);
    Ok(u64::from_le_bytes(array))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("llmlip-uma-pack-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    #[test]
    fn roundtrip_f64_and_i64_pack() {
        let root = temp_dir("roundtrip");
        let tensor_dir = root.join("tensors");
        fs::create_dir_all(&tensor_dir).expect("tensor dir");
        let f64_path = PathBuf::from("tensors/a.f64.bin");
        let i64_path = PathBuf::from("tensors/b.i64.bin");
        fs::write(
            root.join(&f64_path),
            [1.0f64, 2.0].map(f64::to_le_bytes).concat(),
        )
        .expect("write f64");
        fs::write(
            root.join(&i64_path),
            [3i64, 4].map(i64::to_le_bytes).concat(),
        )
        .expect("write i64");

        let specs = vec![
            UmaTensorPackSourceSpec {
                name: "alpha".to_owned(),
                dtype: "f64".to_owned(),
                shape: vec![2],
                path: f64_path.clone(),
                expected_bytes: 16,
            },
            UmaTensorPackSourceSpec {
                name: "beta".to_owned(),
                dtype: "i64".to_owned(),
                shape: vec![2],
                path: i64_path,
                expected_bytes: 16,
            },
        ];
        let pack_path = root.join(UMA_TENSOR_PACK_FILENAME);
        write_uma_tensor_pack(&pack_path, &specs).expect("write pack");
        let loaded = load_uma_tensor_pack(&pack_path, &specs).expect("load pack");
        assert_eq!(loaded.total_bytes, 32);
        let alpha = loaded.entries.get("alpha").expect("alpha");
        let f64_bytes = loaded.entry_bytes("alpha", alpha).expect("alpha bytes");
        let f64_words: &[u64] = bytemuck::cast_slice(f64_bytes);
        let f64_values: &[f64] = bytemuck::cast_slice(f64_words);
        assert_eq!(f64_values, &[1.0, 2.0]);
    }

    #[test]
    fn rejects_non_contiguous_data_offsets_before_access() {
        let root = temp_dir("bad-offset");
        let tensor_path = PathBuf::from("a.f64.bin");
        fs::write(
            root.join(&tensor_path),
            [1.0f64].map(f64::to_le_bytes).concat(),
        )
        .expect("write f64");
        let specs = vec![UmaTensorPackSourceSpec {
            name: "alpha".to_owned(),
            dtype: "f64".to_owned(),
            shape: vec![1],
            path: tensor_path,
            expected_bytes: 8,
        }];
        let pack_path = root.join(UMA_TENSOR_PACK_FILENAME);
        write_uma_tensor_pack(&pack_path, &specs).expect("write pack");

        let mut bytes = fs::read(&pack_path).expect("read pack");
        let data_rel_offset_field = PACK_HEADER_START + 4 + "alpha".len() + 1 + 4 + 8;
        bytes[data_rel_offset_field..data_rel_offset_field + 8]
            .copy_from_slice(&8_u64.to_le_bytes());
        fs::write(&pack_path, bytes).expect("rewrite corrupt pack");

        let error = load_uma_tensor_pack(&pack_path, &specs).expect_err("bad offset must fail");
        assert!(
            error
                .to_string()
                .contains("data_rel_offset must be contiguous"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_excessive_shape_rank_before_allocation() {
        let root = temp_dir("bad-rank");
        let tensor_path = PathBuf::from("a.f64.bin");
        fs::write(
            root.join(&tensor_path),
            [1.0f64].map(f64::to_le_bytes).concat(),
        )
        .expect("write f64");
        let specs = vec![UmaTensorPackSourceSpec {
            name: "alpha".to_owned(),
            dtype: "f64".to_owned(),
            shape: vec![1],
            path: tensor_path,
            expected_bytes: 8,
        }];
        let pack_path = root.join(UMA_TENSOR_PACK_FILENAME);
        write_uma_tensor_pack(&pack_path, &specs).expect("write pack");

        let mut bytes = fs::read(&pack_path).expect("read pack");
        let shape_rank_field = PACK_HEADER_START + 4 + "alpha".len() + 1;
        bytes[shape_rank_field..shape_rank_field + 4]
            .copy_from_slice(&((MAX_TENSOR_RANK as u32) + 1).to_le_bytes());
        fs::write(&pack_path, bytes).expect("rewrite corrupt pack");

        let error = load_uma_tensor_pack(&pack_path, &specs).expect_err("bad rank must fail");
        assert!(
            error.to_string().contains("shape rank"),
            "unexpected error: {error}"
        );
    }
}
