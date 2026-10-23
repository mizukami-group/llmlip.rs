# アーキテクチャ概要

本ドキュメントでは、`e3nn-core-rs` のワークスペース構成、
各 crate の役割、モジュール間の依存関係、そしてデータフロー全体を
日本語で解説します。

---

## 目次

1. [プロジェクトの目的](#プロジェクトの目的)
2. [ワークスペース構成](#ワークスペース構成)
3. [e3nn-core — 数学プリミティブ層](#e3nn-core--数学プリミティブ層)
4. [llmlip — MLIP 推論ランタイム層](#llmlip--mlip-推論ランタイム層)
5. [llmlip-cli — コマンドラインインターフェース](#llmlip-cli--コマンドラインインターフェース)
6. [エンドツーエンドのデータフロー](#エンドツーエンドのデータフロー)
7. [モデル形式と読み込み](#モデル形式と読み込み)
8. [サポートするアーキテクチャ](#サポートするアーキテクチャ)
9. [補助ディレクトリ](#補助ディレクトリ)

---

## プロジェクトの目的

`e3nn-core-rs` は、**固定重みの等変機械学習原子間ポテンシャル（MLIP）** から
Python/PyTorch なしで **エネルギー** と **保存力** を推論するための
Rust ネイティブランタイムです。

```text
入力: 原子種, 原子位置, 単位格子, 周期境界条件
  → 固定 MLIP モデル
  → 全エネルギー E
  → 保存力 F_i = −∂E/∂R_i
  → 応力 / ビリアル（PBC の場合）
```

**設計上の非目標**：
- 学習（training）・オプティマイザ
- パラメータ勾配の計算
- 汎用自動微分エンジン
- GPU カーネル（v0.1 時点）
- 二階微分（ヘッセ行列・フォノン）

---

## ワークスペース構成

```text
e3nn-core-rs/
├── crates/
│   ├── e3nn-core/          数学プリミティブ（フレームワーク非依存）
│   ├── llmlip/         MLIP 推論ランタイム（構造体・近傍リスト・モデル実行）
│   └── llmlip-cli/           コマンドラインインターフェース
├── docs/                   仕様書・規約・チュートリアル
├── fixtures/               参照フィクスチャ（テスト用固定データ）
├── tools/reference/        Python 参照実装・フィクスチャ生成スクリプト
├── third_party/            外部参照（e3nn-torch, NequIP, MACE）の pinned submodule
└── prompts/development/          development 用タスクプロンプト
```

依存関係は一方向です：

```text
llmlip-cli  →  llmlip  →  e3nn-core
```

`e3nn-core` は外部の ML フレームワークに一切依存しません。
Python は `tools/reference/` でのフィクスチャ生成にのみ使用されます。

---

## e3nn-core — 数学プリミティブ層

**目的**: フレームワーク非依存の等変数学カーネルを提供します。
MLIP 固有のロジック（近傍リスト・モデル読み込み等）は含みません。

### モジュール一覧

| モジュール | ファイル | 役割 |
|:--|:--|:--|
| `error` | `error.rs` | 型付きエラー定義 |
| `geometry` | `geometry.rs` | `Vec3` 型、ベクトル演算（`add`, `sub`, `dot`, `norm`） |
| `irreps` | `irreps.rs` | `Irrep`, `IrrepMul`, `Irreps`, `LayoutSegment` — O(3) 既約表現とメモリレイアウト |
| `radial` | `radial.rs` | 距離計算、カットオフ関数、放射基底（Bessel, Gaussian, Agnesi, ZBL）の順方向・VJP |
| `sh` | `sh.rs` | 実球面調和関数の順方向・VJP (`l_max ≤ 4`, component 正規化) |
| `tp` | `tp.rs` | テンソル積 — `TensorProductPlan`、CG 係数、順方向・VJP |
| `mlp` | `mlp.rs` | 固定スカラー MLP — `FixedMlp`、`DenseLayer`、input-only VJP |
| `mace` | `mace.rs` | MACE 対称縮約カーネル（スカラー・ベクトル・混合） |

### 設計原則

- **Plan オブジェクト** を事前構築し、繰り返し使用する
- 順方向・VJP 関数はスライスと出力バッファを受け取る
- 内部カーネルでの隠れた動的アロケーションは最小限
- 形状検証は Plan 構築時または外部 API 境界で実施

```rust
// 使用パターンの例
let plan = SphericalHarmonicsPlan::component(l_max, normalize_input)?;
plan.forward(r, &mut out)?;
plan.vjp_edge_vector(r, &grad_y, &mut grad_r)?;
```

---

## llmlip — MLIP 推論ランタイム層

**目的**: 原子構造の入力から、固定モデルを用いた
エネルギー・力・応力の予測までを組み立てます。

### モジュール一覧

| モジュール | ファイル | 役割 |
|:--|:--|:--|
| `structure` | `structure.rs` | `Structure`（原子種・位置・格子・PBC）、`Prediction`（E/F/σ/virial） |
| `geometry` | `geometry.rs` | `Cell`（格子ベクトル）、PBC 用幾何ユーティリティ |
| `neighbor` | `neighbor.rs` | `NeighborList` — カットオフ内の原子ペア列挙（全ペア・セルリスト） |
| `edge` | `edge.rs` | `Edge`（src, dst, shift, vector, distance）、力蓄積ヘルパー |
| `force_accum` | `force_accum.rs` | エッジ余接 → 原子力 scatter、変位勾配 → 応力/ビリアル |
| `model` | `model.rs` | `.e3rs` モデルローダー、`LlmlipModel` enum ディスパッチ |
| `extxyz` | `extxyz.rs` | Extended XYZ 形式の読み書き |
| `synthetic` | `synthetic.rs` | 放射基底のみの合成テストモデル |
| `synthetic_equivariant` | `synthetic_equivariant.rs` | SH/TP を使う合成等変テストモデル |
| `nequip_lite` | `nequip_lite.rs` | NequIP-lite サブセットランタイム |
| `nequip_oam_prefix` | `nequip_oam_prefix.rs` | NequIP OAM prefix 変種ランタイム |
| `mace_lite` | `mace_lite.rs` | MACE-lite サブセットランタイム（BLAS 統合含む） |
| `mace_polar` | `mace_polar.rs` | PolarMACE ランタイム（極性テンソル出力） |
| `dftd3` | `dftd3.rs` | DFT-D3 分散補正 |
| `dispersion` | `dispersion.rs` | 分散補正付きモデルのラッパー |

### キャッシュ戦略

力の計算には、逆伝播時に順方向の中間値が必要です。
明示的なキャッシュ構造体で管理します：

```text
MlpCache          — MLP 各層の活性化値を保持
ModelForwardCache  — モデル全体の中間値を保持（モデル内部で管理）
```

v0.1 では明瞭さのために中間値を保持します。
メモリ削減のための再計算モードは将来のオプションです。

---

## llmlip-cli — コマンドラインインターフェース

**目的**: `llmlip` の薄いラッパー。数学ロジックは一切含みません。

### コマンド

```bash
# エネルギー・力の予測
llmlip predict --model model.e3rs --structure input.extxyz --energy --forces out.xyz

# PBC 系の応力・ビリアル
llmlip predict --model model.e3rs --structure pbc.extxyz --energy --stress --virial

# 保存力の有限差分検証
llmlip finite-difference-forces --model model.e3rs --structure input.extxyz

# ベンチマーク（JSON 統計出力）
llmlip benchmark-predict --model model.e3rs --structure input.extxyz --energy --forces
```

`predict` は `.e3rs` モデルと extXYZ 構造を読み込み、
`llmlip` の `LlmlipModel::predict` を呼び出します。

---

## エンドツーエンドのデータフロー

以下は NequIP/MACE 系モデルの推論における典型的なデータフローです：

```text
┌─────────────────────────────────────────────────────────────┐
│                        入力                                  │
│  Structure { species, positions, cell, pbc }                │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  近傍リスト構築  (llmlip/neighbor.rs)                     │
│  NeighborList { edges: Vec<Edge>, cutoff, edge_mode }       │
│                                                             │
│  各 Edge: { src, dst, shift, shift_cart, vector, distance } │
│  r_ij = R_j + shift_cart − R_i                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                    ┌─────┴─────┐
                    ▼           ▼
    ┌───────────────────┐  ┌──────────────────────┐
    │ 放射パイプライン     │  │ 角度パイプライン        │
    │ (e3nn-core/radial) │  │ (e3nn-core/sh)       │
    │                   │  │                      │
    │ d_ij → cutoff     │  │ r_ij → Y_ij          │
    │ d_ij → basis      │  │ (l_max ≤ 4)          │
    │ basis → MLP → w_ij│  │                      │
    └────────┬──────────┘  └──────────┬───────────┘
             │                        │
             └────────┬───────────────┘
                      ▼
    ┌─────────────────────────────────────────────┐
    │ テンソル積  (e3nn-core/tp)                     │
    │ m_ij = TP(x_i, Y_ij, w_ij)                  │
    │ (CG 係数による等変メッセージ生成)                │
    └────────────────────┬────────────────────────┘
                         ▼
    ┌─────────────────────────────────────────────┐
    │ メッセージ集約                                 │
    │ h_i = Σ_{j ∈ neighbors(i)} m_ij             │
    └────────────────────┬────────────────────────┘
                         │
                    ┌────┴────┐
                    ▼         ▼
            NequIP 系     MACE 系
         (直接 readout)  (対称縮約 → readout)
                    │         │
                    └────┬────┘
                         ▼
    ┌─────────────────────────────────────────────┐
    │ エネルギー集約                                 │
    │ E = Σ_i (scale_i × ε_i + shift_i)           │
    └────────────────────┬────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────┐
    │ 逆伝播（Force-only VJP）                      │
    │                                             │
    │ bar_E = 1                                   │
    │   → bar_ε → bar_h → bar_m                  │
    │   → bar_x, bar_Y, bar_w                    │
    │   → bar_basis, bar_cutoff                   │
    │   → bar_d → bar_r_ij                        │
    │                                             │
    │ 力の蓄積:                                    │
    │   F_i += bar_r_ij                           │
    │   F_j −= bar_r_ij                           │
    │                                             │
    │ 応力（PBC の場合）:                            │
    │   G[row][col] += r_ij[row] × bar_r_ij[col] │
    │   stress = G / |det(cell)|                  │
    └─────────────────────────────────────────────┘
```

---

## モデル形式と読み込み

### `.e3rs` 形式

`.e3rs` は本プロジェクト独自のシリアライズ形式です。
ディレクトリ構造を持ち、以下を含みます：

```text
model.e3rs/
├── manifest.toml          メタデータ（アーキテクチャ・種類・カットオフ等）
├── plan.json              実行計画（TP パス・MLP 構成・irreps 等）
└── tensors/               バイナリ f64 重みファイル群
    ├── layer0_weight.bin
    ├── layer0_bias.bin
    └── ...
```

`manifest.toml` は以下を記録します：
- フォーマットバージョン (`0.1`)
- アーキテクチャ名 (`mace-lite-v1`, `nequip-lite-v1` 等)
- データ型 (`f64`)
- 単位系 (`angstrom`, `eV`)
- エッジ規約 (`R_j + shift - R_i`)
- カットオフ半径
- 対応する原子種

### モデルローダー

`model.rs` の `load_llmlip_model` が manifest を読み、
アーキテクチャに応じて適切なモデル構造体を構築します：

```rust
pub enum LlmlipModel {
    SyntheticRadial(SyntheticRadialModel),
    SyntheticEquivariant(SyntheticEquivariantModel),
    NequipLite(NequipLiteModel),
    NequipOamPrefix(NequipOamPrefixModel),
    MaceLite(Box<MaceLiteModel>),
    MacePolar(Box<MacePolarModel>),
    DispersionCorrected(Box<DispersionCorrectedModel>),
}
```

各バリアントは同じインターフェースで `predict(structure, options)` を提供します。

---

## サポートするアーキテクチャ

### NequIP-lite

NequIP の固定サブセット。メッセージパッシング層を直列に接続し、
各層で TP + 放射 MLP + 集約を行います。

```text
node_embedding → [Layer 1] → [Layer 2] → ... → readout → E
                    ↑
              TP(h_i, Y_ij, w_ij) + 集約
```

### MACE-lite

MACE の固定サブセット。NequIP に加えて、メッセージ集約後に
**対称縮約（Symmetric Contraction）** を行います。

```text
node_embedding → [Interaction 1] → 対称縮約 → [Interaction 2] → 対称縮約 → readout → E
                      ↑                                  ↑
                 TP + radial + 集約                  TP + radial + 集約
```

対称縮約は、固定の U テンソルと元素条件付き重みを用いた多体特徴量展開です。
スカラー（`0e`）とベクトル（`1o`）の出力を連結して次の層に渡します。

### DFT-D3 分散補正

MACE 等のベースモデルに DFT-D3(BJ) 分散補正を加えるラッパーです。
配位数依存の C6 係数を使った二体相互作用を追加します。

---

## 補助ディレクトリ

| ディレクトリ | 用途 |
|:--|:--|
| `fixtures/` | 参照フィクスチャ（JSON/バイナリ）。Rust テストで数値パリティを検証 |
| `tools/reference/` | Python スクリプト（PyTorch/e3nn/NequIP/MACE）によるフィクスチャ生成 |
| `third_party/` | 外部参照実装の pinned submodule（読み取り専用、コード流用禁止） |
| `docs/` | 仕様書（MATH_CONVENTIONS, FORCE_ONLY_VJP_SPEC 等）・チュートリアル |
| `prompts/development/` | development 向けタスクプロンプト |

### テストとフィクスチャの方針

- すべての数学プリミティブは順方向テスト + VJP テスト + 有限差分テストを持つ
- 規約依存の操作は pinned 参照フィクスチャで検証する
- 通常の `cargo test` は Python/PyTorch/e3nn を必要としない
- 参照フィクスチャは JSON として `fixtures/` に格納される

---

## まとめ

```text
┌─────────────────────────────────────────────────────────────────┐
│  llmlip-cli           薄い CLI。数学ロジックなし                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  llmlip       構造体・近傍リスト・モデル実行・力蓄積        │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  e3nn-core     Irreps・SH・TP・放射基底・MLP・MACE SC │  │  │
│  │  │                (Python 非依存・フレームワーク非依存)      │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

3つの crate は明確に責務を分離しています：

1. **`e3nn-core`** — 等変数学の純粋なカーネル群
2. **`llmlip`** — 原子構造から力までの推論パイプライン
3. **`llmlip-cli`** — ユーザー向け CLI ラッパー

この階層設計により、数学プリミティブの検証・MLIP ランタイムの組み立て・
ユーザーインターフェースがそれぞれ独立してテスト・保守できます。
