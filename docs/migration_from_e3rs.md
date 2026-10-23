# `e3rs` から `llmlip` への移行ガイド

公開分割の Phase 1(PR100)で、プロダクトの名前空間を `e3rs` から
**`llmlip`** に改名した。**モデル形式とその拡張子 `.e3rs` は改名対象外**で、
変換済みモデルディレクトリはそのまま動作する。フォーマット仕様書も
`docs/MODEL_FORMAT_E3RS.md` のまま変わらない。

## 変更が必要なもの

### CLI

| 旧 | 新 |
|---|---|
| `e3rs predict --model model.e3rs ...` | `llmlip predict --model model.e3rs ...` |
| `cargo build -p e3rs-cli` | `cargo build -p llmlip-cli`(バイナリ名: `llmlip`) |

### Rust クレート / API

| 旧 | 新 |
|---|---|
| `mlip-force` クレート | `llmlip` クレート |
| `e3rs-cli` / `e3rs-capi` / `e3rs-py` | `llmlip-cli` / `llmlip-capi` / `llmlip-py` |
| `E3rsModel` | `LlmlipModel` |
| `load_e3rs_model` / `validate_e3rs_model` | `load_llmlip_model` / `validate_llmlip_model` |

### Python

| 旧 | 新 |
|---|---|
| pip パッケージ名 `e3rs`(pyproject) | `llmlip` |
| `import e3rs` / ネイティブモジュール `e3rs._e3rs` | `import llmlip` / `llmlip._llmlip` |

### ASE

| 旧 | 新 |
|---|---|
| `from e3rs.ase import E3rsCalculator` | `from llmlip.ase import LlmlipCalculator` |
| `tools/python/e3rs_ase` shim(削除済み) | `from llmlip.ase import LlmlipCalculator` |

### OpenMM / OpenMM-ML

| 旧 | 新 |
|---|---|
| `create_e3rs_force`(`e3rs.openmm`) | `create_llmlip_force`(`llmlip.openmm`) |
| `E3RSPotentialImplFactory`(`e3rs.openmmml`) | `LlmlipPotentialImplFactory`(`llmlip.openmmml`) |
| OpenMM-ML potential 名 / entry point `e3rs` | `llmlip` |

### LAMMPS

| 旧 | 新 |
|---|---|
| `pair_style e3rs` | `pair_style llmlip` |
| `PairE3rs` | `PairLlmlip` |
| `pair_e3rs.{h,cpp}` | `pair_llmlip.{h,cpp}` |
| プラグインライブラリ `lammps_e3rsplugin` | `lammps_llmlipplugin` |
| CMake プロジェクト `lammps_e3rs_plugin` | `lammps_llmlip_plugin` |
| CMake 変数 `E3RS_CAPI_INCLUDE_DIR` / `E3RS_CAPI_LIBRARY` / `E3RS_LAMMPS_USE_STUBS` | `LLMLIP_CAPI_INCLUDE_DIR` / `LLMLIP_CAPI_LIBRARY` / `LLMLIP_LAMMPS_USE_STUBS` |
| `lammps_plugin/e3rs/` | `lammps_plugin/llmlip/` |
| `scripts/*lammps_e3rs*` | `scripts/*lammps_llmlip*` |

### C ABI

| 旧 | 新 |
|---|---|
| ヘッダ `e3rs_capi.h` | `llmlip_capi.h` |
| `e3rs_model_load` / `e3rs_model_free` / `e3rs_model_capabilities` | `llmlip_model_load` / `llmlip_model_free` / `llmlip_model_capabilities` |
| `e3rs_predict_external_graph` / `e3rs_predict_structure` | `llmlip_predict_external_graph` / `llmlip_predict_structure` |
| `e3rs_last_error` / `e3rs_clear_error` | `llmlip_last_error` / `llmlip_clear_error` |
| 型 `e3rs_model_t` / `e3rs_status_t` / `e3rs_dtype_t` | `llmlip_model_t` / `llmlip_status_t` / `llmlip_dtype_t` |
| 定数 `E3RS_OK` / `E3RS_ERROR_*` / `E3RS_DTYPE_*`(その他 `E3RS_*` 全般) | `LLMLIP_OK` / `LLMLIP_ERROR_*` / `LLMLIP_DTYPE_*`(`LLMLIP_*`) |

### 環境変数

| 旧 | 新 |
|---|---|
| `E3RS_BINARY` | `LLMLIP_BINARY` |
| `E3RS_ASE_BACKEND` | `LLMLIP_ASE_BACKEND` |
| `E3RS_MODEL` | `LLMLIP_MODEL` |
| `E3RS_LAMMPS_PLUGIN` | `LLMLIP_LAMMPS_PLUGIN` |
| `E3RS_*`(その他すべて) | `LLMLIP_*` |
| (新規) | `LLMLIP_EMBED_MODEL` — コンパイル時モデル埋め込み(`--features embed-model`) |

## 変更されないもの(改名対象外)

- **モデル形式と拡張子 `.e3rs` はそのまま**(改名対象外)。`.e3rs` モデル
  ディレクトリ(manifest.toml + tensors/state ファイル)は再変換不要。
- manifest 内のアーキテクチャ識別子(`mace-lite-v1`、`uma-v1`、
  `esen-oam-v1` など)。
- **`docs/MODEL_FORMAT_E3RS.md` も名前ごとそのまま維持**(フォーマット仕様書
  の改名はしない)。
- `tools/reference/` 配下のコンバータ exporter インターフェース(新しい
  `converters/convert.py` フロントエンドがこれらをラップする)。
