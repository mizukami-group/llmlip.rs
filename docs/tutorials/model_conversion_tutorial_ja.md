# モデル変換チュートリアル — PyTorch モデルから `.e3rs` 形式への変換

## 概要

`e3nn-core-rs` は Python/PyTorch に依存しない Rust ネイティブの推論ランタイムです。
PyTorch で学習済みの MLIP モデル（MACE, NeQUIP 等）を Rust ランタイムで実行するには、
まず Python 側のエクスポータで **`.e3rs` 形式**に変換する必要があります。

本チュートリアルでは、変換の全体像・具体的な手順・トラブルシューティングを説明します。

---

## 1. `.e3rs` 形式の構造

`.e3rs` モデルはディレクトリ形式で、以下のファイルで構成されます。

```text
model.e3rs/
  manifest.toml          # モデルメタデータ（アーキテクチャ、種、カットオフ等）
  mace_lite.json         # MACE/NeQUIP 用：実行プラン（レイヤー構成、TP プラン等）
  weights/
    tensor_000.f64.bin   # 重みテンソル（リトルエンディアン f64/f32 バイナリ）
    tensor_001.f64.bin
  plans/
    tensor_product_000.json  # （オプション）テンソル積プランファイル
```

### manifest.toml の主要フィールド

```toml
format_version = "0.1"
architecture = "mace-lite-v1"       # 対応: mace-lite-v1, mace-polar-v1,
                                    #       nequip-lite-v1, nequip-oam-l-v1,
                                    #       synthetic-local-v1, synthetic-equivariant-v1
dtype = "f64"
length_unit = "angstrom"
energy_unit = "eV"
force_unit = "eV/angstrom"

[species]
symbols = ["H", "C", "N", "O"]     # モデルが扱う元素記号

[geometry]
cutoff = 5.0                        # カットオフ半径 (Å)
edge_vector = "R_j + shift - R_i"   # エッジベクトル規約
directed_edges = true

[spherical_harmonics]
l_max = 3
normalize_input = true
normalization = "component"
reference_convention = "pinned-fixture"

[[tensors]]
name = "interactions.0.linear_up.weight"
dtype = "f64"
shape = [128, 64]
layout = "row-major"
path = "weights/interactions.0.linear_up.weight.f64.bin"
```

### テンソルバイナリの仕様

- リトルエンディアン、f64 (8 bytes) または f32 (4 bytes)
- Row-major（行優先）連続配列、圧縮なし
- ファイルサイズの検証: `file_bytes == product(shape) * sizeof(dtype)`
- f32 テンソルはローダーが f64 に昇格して使用

---

## 2. 前提条件：参照環境のセットアップ

変換スクリプトは `tools/reference/` にあり、Python 環境が必要です。

```bash
# 参照用 Python 仮想環境を作成
python3 -m venv .venv-reference
source .venv-reference/bin/activate
pip install --upgrade pip

# e3nn 参照サブモジュールの追加（初回のみ）
scripts/add_e3nn_reference_submodule.sh

# MACE 参照サブモジュールの追加（MACE モデル変換時）
scripts/add_mace_reference_submodule.sh

# NeQUIP 参照サブモジュールの追加（NeQUIP モデル変換時）
scripts/add_nequip_reference_submodule.sh

# 依存パッケージのインストール
pip install -r tools/reference/requirements.txt
```

> **重要**: 参照サブモジュールは必ずピン留めされたバージョンを使用してください。
> `third_party/` 配下のチェックアウトが `*.PINNED_COMMIT` マーカーと一致している
> ことがスクリプトにより自動検証されます。

---

## 3. MACE モデルの変換手順

### 3.1 概要

MACE モデルの変換は `tools/reference/generate_mace_fixtures.py` が担います。
このスクリプトは以下を行います：

1. ピン留めされた MACE チェックアウト（`third_party/mace`）を検証
2. upstream の PyTorch モデル（`.model` / `.pt`）をロード
3. モデル構成を検査し、対応するサブセットかどうかを判定
4. 重みテンソルを f64 バイナリに書き出し
5. `manifest.toml` と `mace_lite.json`（実行プラン）を生成
6. 参照出力 fixture（エネルギー/力/応力/ヴィリアル）を生成

### 3.2 対応しているモデルバリアント

| アーキテクチャ名 | 対応モデル |
|---|---|
| `mace-lite-v1` | `ScaleShiftMACE` (MACE-MP0b, Osaka26 等) |
| `mace-polar-v1` | `PolarMACE` (PolarMACE foundation 等) |

### 3.3 実行例：MACE-MP0b ランタイムエクスポート

```bash
source .venv-reference/bin/activate

PYTHONPATH=tools/reference:third_party/mace \
  python tools/reference/generate_mace_fixtures.py \
    --mp0b-runtime \
    --out fixtures/mace/v0_3_16_4d2da09
```

### 3.4 実行例：MACE-Osaka24 バリアントのエクスポート

```bash
PYTHONPATH=tools/reference:third_party/mace \
  python tools/reference/generate_mace_osaka24_fixtures.py \
    --out fixtures/mace/v0_3_16_4d2da09
```

### 3.5 エクスポート後の確認

```bash
# Rust 側でロードテスト
cargo test -p llmlip --test mace_lite_reference

# upstream との比較検証
PYTHONPATH=tools/reference:third_party/mace \
  python tools/reference/compare_against_mace.py \
    --fixture fixtures/mace/v0_3_16_4d2da09
```

---

## 4. NeQUIP モデルの変換手順

### 4.1 Nequip-lite（決定論的小規模モデル）

```bash
source .venv-reference/bin/activate

PYTHONPATH=tools/reference:third_party/nequip:third_party/e3nn-torch \
  python tools/reference/generate_nequip_fixtures.py \
    --out fixtures/nequip/current
```

このスクリプトは以下を生成します：

- 決定論的な `NequIPGNNModel` インスタンスを構築
- 対応するサブセットを `.e3rs` にエクスポート
- Python autograd による参照エネルギー/力の fixture を出力

### 4.2 NeQUIP-OAM-L（本番アーティファクト）

NeQUIP-OAM-L パッケージアーティファクトのエクスポートは多段階です：

```bash
# Step 1: アーティファクトのダウンロードとマニフェストロック
PYTHONPATH=tools/reference:third_party/nequip:third_party/e3nn-torch \
  python tools/reference/nequip_oam_artifact.py \
    --download \
    --out fixtures/nequip/current/nequip_oam_l_artifact_manifest.json

# Step 2: アーティファクトの検査
PYTHONPATH=tools/reference:third_party/nequip:third_party/e3nn-torch \
  python tools/reference/inspect_nequip_oam_artifact.py \
    --manifest fixtures/nequip/v0_17_1_604ce73b/nequip_oam_l_artifact_manifest.json \
    --out fixtures/nequip/current/nequip_oam_l_inspection.json

# Step 3: .e3rs へのエクスポートと fixture 生成
PYTHONPATH=tools/reference:third_party/nequip:third_party/e3nn-torch \
  python tools/reference/generate_nequip_oam_prefix_fixture.py \
    --out fixtures/nequip/v0_17_1_604ce73b
```

---

## 5. Synthetic モデルの変換

Synthetic モデル（`synthetic-local-v1`, `synthetic-equivariant-v1`）は
`tools/reference/generate_e3nn_fixtures.py` で生成されます。
これは主にテスト・検証用であり、e3nn-torch の参照実装との parity 確認に使用されます。

```bash
source .venv-reference/bin/activate
python tools/reference/generate_e3nn_fixtures.py \
  --out fixtures/e3nn_torch/current

# JSON に変換（Rust テストが消費する形式）
python tools/reference/convert_e3nn_fixtures_to_json.py \
  --fixtures fixtures/e3nn_torch/current \
  --out fixtures/e3nn_torch/v0_6_0_2aa7f584
```

---

## 6. DFTD3 分散補正付きモデルの生成

MACE モデルに DFTD3(BJ) 二体分散補正を付加する場合：

```bash
PYTHONPATH=tools/reference:third_party/mace \
  python tools/reference/create_dftd3_corrected_llmlip.py \
    --base-model fixtures/mace/v0_3_16_4d2da09/models/mace_osaka24_medium_runtime.e3rs \
    --out fixtures/mace/v0_3_16_4d2da09/models/mace_osaka24_medium_dftd3_runtime.e3rs
```

DFTD3 パラメータは `manifest.toml` の `[dispersion]` セクションと
`dftd3.json` ペイロードとして `.e3rs` に埋め込まれます。

---

## 7. Rust 側でのモデルロードと推論

変換後の `.e3rs` モデルは `llmlip` クレートの `LlmlipModel::load()` でロードできます。

```rust
use llmlip::{LlmlipModel, PredictionOptions, Structure};

// モデルのロード
let model = LlmlipModel::load("path/to/model.e3rs")?;

// 構造の作成
let structure = Structure::new(
    vec!["H".into(), "O".into(), "H".into()],
    vec![
        [0.0, 0.0, 0.0],
        [0.9572, 0.0, 0.0],
        [-0.2400, 0.9273, 0.0],
    ],
);

// エネルギーと力の予測
let prediction = model.predict(
    &structure,
    &PredictionOptions {
        compute_energy: true,
        compute_forces: true,
        compute_stress: false,
        compute_virial: false,
    },
)?;

println!("Energy: {:.6} eV", prediction.energy.unwrap());
for (i, force) in prediction.forces.unwrap().iter().enumerate() {
    println!("Force[{i}]: [{:.6}, {:.6}, {:.6}] eV/Å", force[0], force[1], force[2]);
}
```

CLI からの実行：

```bash
cargo run -p llmlip-cli --release -- predict \
  --model path/to/model.e3rs \
  --structure input.extxyz \
  --energy --forces
```

---

## 8. トラブルシューティング

### 「imported mace does not come from the pinned checkout」エラー

ピン留めされた `third_party/mace` 以外の MACE がインポートされています。
`PYTHONPATH` の設定を確認し、システムにインストールされた MACE と衝突していないか
確認してください。

### テンソルサイズ不一致エラー

`manifest.toml` の `[[tensors]]` に記載された `shape` とバイナリファイルサイズが
合致しない場合に発生します。エクスポータのバージョンとモデルの一貫性を確認してください。

### 「unsupported architecture」エラー

Rust ローダーが認識しないアーキテクチャ名が `manifest.toml` に記載されています。
対応アーキテクチャは以下のみです：
- `synthetic-local-v1`
- `synthetic-equivariant-v1`
- `nequip-lite-v1`
- `nequip-oam-l-v1`（旧名 `nequip-oam-prefix-v1` も受理）
- `mace-lite-v1`
- `mace-polar-v1`

### f32 テンソルの扱い

upstream アーティファクトが f32 で保存されている場合、`[[tensors]]` の
`dtype = "f32"` を明示的に宣言できます。ローダーは f64 に昇格して使用します。

---

## 9. 変換プロセスのまとめ

```text
PyTorch モデル (.model / .pt)
  │
  ├─ tools/reference/generate_mace_fixtures.py      (MACE)
  ├─ tools/reference/generate_nequip_fixtures.py     (NeQUIP)
  ├─ tools/reference/generate_e3nn_fixtures.py       (Synthetic)
  │
  ▼
.e3rs ディレクトリ
  ├── manifest.toml      ← モデルメタデータ
  ├── mace_lite.json     ← 実行プラン (MACE/NeQUIP)
  └── weights/           ← バイナリテンソル
        ├── *.f64.bin
        └── *.f32.bin
  │
  ▼
Rust ランタイム (LlmlipModel::load → predict)
  ├── llmlip-cli predict   ← CLI
  └── llmlip API     ← ライブラリ
```

> **参考ドキュメント**:
> - `docs/MODEL_FORMAT_E3RS.md` — `.e3rs` 形式の完全な仕様
> - `docs/REFERENCE_IMPLEMENTATION_SPEC.md` — 参照実装ポリシー
> - `docs/MACE_LITE_SPEC.md` — MACE-lite サブセットの詳細
> - `docs/NEQUIP_LITE_SPEC.md` — Nequip-lite サブセットの詳細
