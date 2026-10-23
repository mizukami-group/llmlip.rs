# モデル別プロダクション適性分類（FP64・CPU）

分離元で2026-06-10/11に実施した系統検証で取得した
実測速度・メモリ挙動に基づき、各モデルを**プロダクション利用可否**で分類する。

この表は分離前の測定記録です。今回の公開候補で全条件を再測定したものではありません。
現在の検証範囲は[VALIDATION.md](VALIDATION.md)を参照してください。

## 測定条件と判定基準

- 環境: Apple M4 Max 16コア / 128 GB、`--release --features parallel,accelerate-blas`、
  **FP64**（f32 対応モデルはさらに高速になるが、本表は f64 基準）。
- 指標:
  - **MPtrj ms/frame** — バルク結晶 10,000 構造（平均 ~32 原子）の平均予測時間
  - **Si6000 ms/atom** — 6000 原子 full-PBC Si での原子あたり予測時間（MD 規模の代表）
  - **Mindless ms/frame** — 小分子（5–25 原子）平均
- 判定基準（明示）:
  - **Tier A（プロダクション可）**: Si6000 ≤ 0.7 ms/atom かつ MPtrj ≤ 50 ms/frame かつ
    6000 原子系がメモリ内で完走
  - **Tier B（条件付き可）**: 上記を満たさないが、特定用途（小〜中規模系、単発計算、
    非 PBC 分子、固有機能）では実用になる
  - **Tier C（プロダクション非推奨・CPU）**: 大規模系がメモリ超過する、または
    速度が用途を強く制限する
- 全モデルとも**正しさは検証済み**（エネルギー/力/ビリアル全件成功・非有限値ゼロ）。
  本分類は純粋に**速度・メモリの実用性**の区分である。

## Tier A — プロダクション可（MD・大規模系・ハイスループット）

| モデル | MPtrj ms/frame | Si6000 ms/atom | water6000 | ビリアル | 備考 |
|---|---:|---:|:---:|:---:|---|
| MACE-Osaka24 small | 7.3 | 0.112 | ✅ | ✓ | 最速。r_max 4.5 Å |
| MACE-Osaka26 small | 9.7 | 0.158 | ✅ | ✓ | 97 元素 |
| PET-MAD-XS | 10.3 | 0.179 | ✅ | ✓ | 最軽量級(f32 経路は未対応 — 対応は PET-MAD-S) |
| MACE-OAM0 small | 11.6 | 0.176 | ✅ | ✓ | OMat24 系列 |
| MACE-MP-0b3 medium | 15.7 | 0.327 | ✅ | ✓ | 実績豊富な基準モデル |
| MACE-Osaka24 medium | 17.0 | 0.239 | ✅ | ✓ | |
| MACE-OAM0 medium | 22.4 | 0.351 | ✅ | ✓ | OMat24 系列 |
| Nequix-MP-1 / OAM-1 | 29.3 | 0.52–0.53 | ✅ | — | ビリアル未実装に注意 |
| MACE-Osaka24 large | 33.9 | 0.521 | ✅ | ✓ | |
| PET-MAD-S | 40.5 | 0.63 | ✅ | ✓ | f32 経路あり(E/F のみ)で更に高速化可 |

## Tier B — 条件付き可（用途を選べば実用）

| モデル | MPtrj ms/frame | Si6000 ms/atom | 制約・適用条件 |
|---|---:|---:|---|
| NequIP-OAM-L | 49.6 | 1.21 | 中規模系まで。精度は良好（MPtrj E/atom MAE 9.8 meV） |
| MACE-MH-1 | 125.7 | 1.83 | 遅いが**6 ヘッド切替**（omat/omol/spice 等）が唯一の機能。単発・小規模向け |
| UMA-S-1.2 | 197.5 | 5.48 | 重い（MoLE 4.3 GB）。water6000 は OOM。OMAT24 で E/atom MAE 3.8–20 meV と最高精度級 — 単発高精度計算向け |
| ORB-v3-OMOL | 15.3（小分子） | n/a（非 PBC 専用） | 分子系専用・電荷/スピン必須。分子なら高速で実用的 |
| MACE-Polar-1 S | 13.8（小分子） | n/a（非 PBC 専用） | 分極系の小分子なら実用的 |
| MACE-Polar-1 M | 32.4（小分子） | n/a（非 PBC 専用） | 同上（中型） |

## Tier C — プロダクション非推奨（CPU・本クラスのハードウェア）

| モデル | MPtrj ms/frame | 6000 原子系 | 理由 |
|---|---:|:---:|---|
| eSEN-30M-OAM | 617.9 | **OOM**（water/Si 両方、128 GB 超） | 精度は最高（MPtrj E/atom MAE 7.9 meV）だが速度・メモリが CPU 実用域外。精度基準が必要な単発・小規模計算に限定 |
| PET-OAM-XL | 1016 | **OOM**（両方） | 730M パラメータ。CPU では研究検証用途のみ |
| Allegro-OAM-L | 290.4 | water **OOM** / Si 18 ms/atom | エッジ数比例メモリ。孤立原子も設計上拒否 |
| MACE-Polar-1 L | 88.5（小分子） | n/a（非 PBC） | 小分子 1 個に ~0.1 s。スループット用途には不向き |

## 補足

- **OOM はアーキテクチャ特性**（エッジ／活性化メモリがエッジ数・パラメータ数に比例）で
  あり実装バグではない。より大きいメモリの環境では Tier が変わり得る。
- f32 経路（MACE-lite 系・PET・MH-1・NequIP-OAM の E+F）を使うと Tier A/B の
  ほとんどはさらに 1.5–2 倍程度高速になる（過去ベンチ
  `docs/benchmark_summaries/` 参照）。
- AMD EPYC + OpenBLAS/MKL 環境の絶対値は本表と異なる（過去の AMD 実測は
  `docs/benchmark_summaries/2026-05-25-*amd*` 等）。Tier の相対関係は概ね保たれる
  見込みだが、境界上のモデル（PET-S / Osaka24-L / NequIP-OAM-L）は再測定を推奨。
- ビリアル列「—」は stress/virial 未実装（NPT MD には使えない）ことを示す。
