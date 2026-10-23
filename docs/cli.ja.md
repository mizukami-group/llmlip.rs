# llmlip CLI

[English](cli.md) | 日本語

`llmlip` は Rust ネイティブの固定重み等変 MLIP 推論エンジンのコマンドライン
フロントエンドである。`.e3rs` モデルディレクトリを読み込み、extXYZ 構造に対する
エネルギー・力・応力・ビリアルを CPU 上で予測する。既定の計算 dtype は FP64。

## 前提とビルド

```sh
cargo build --release -p llmlip-cli
./target/release/llmlip --help
```

バイナリ名は `llmlip` で、`target/release/llmlip` に生成される。

## 最小の動く例

リポジトリには synthetic デモモデル(`examples/model.e3rs`、H/O 種、FP64 専用)と
3 原子の水様構造(`examples/input.extxyz`)が同梱されている:

```sh
$ llmlip validate-model --model examples/model.e3rs
model validation ok

$ llmlip predict --model examples/model.e3rs --structure examples/input.extxyz --energy
energy = 2.8927585729719403e0

$ llmlip predict --model examples/model.e3rs --structure examples/input.extxyz \
    --energy --forces --json
{
  "schema_version": "llmlip-prediction-v1",
  "energy": 2.8927585729719403,
  "forces": [
    [-0.0137316644530655, -0.017634105420898622, 0.0],
    [0.05699796467573395, -0.029977567698224745, 0.0],
    [-0.043266300222668454, 0.04761167311912337, 0.0]
  ],
  "stress": null,
  "virial": null
}
```

(上の JSON は紙面の都合で整形しています。実際の出力は各力成分が 1 行ずつの pretty-print です。数値は実出力そのままです。)

## サブコマンド

llmlip は 5 つのサブコマンド(と clap 生成の `help`)を提供する: `predict`、
`finite-difference-forces`、`benchmark-predict`、`batch-predict`、`validate-model`。

### 共通フラグ

以下は `predict`・`finite-difference-forces`・`benchmark-predict`・`batch-predict` で
共通(`--dtype` は `finite-difference-forces` には存在しない):

| フラグ | 既定値 | 説明 |
| --- | --- | --- |
| `--model <MODEL>` | 必須 | `.e3rs` モデルディレクトリへのパス。 |
| `--dtype <DTYPE>` | `float64` | 計算 dtype: `float64` または `float32`(別名 `fp64`/`f64`、`fp32`/`f32`)。float32 非対応モデルは明示エラーを返す。 |
| `--head <HEAD>` | – | マルチヘッドアーキテクチャ向けのモデルヘッド名。ヘッドを持たないモデルは黙って無視せず明示的に拒否する。 |
| `--total-charge <X>` | – | 全体電荷条件付けを必要とするモデル向けの総電荷。 |
| `--spin-multiplicity <X>` | – | 全体スピン条件付けを必要とするモデル向けのスピン多重度。 |
| `--mace-mh1-wigner-mode <MODE>` | `true-float64` | MACE-MH-1 専用。`true-float64` は Rust ランタイムに組み込まれた f64 Clebsch-Gordan 定数を使う。`upstream-artifact` は upstream TorchScript artifact に一致させるもので、参照比較用。 |

### predict

`--energy`、`--forces`、`--forces-out <PATH>`、`--stress`、`--virial` のうち
少なくとも 1 つが必須。

| フラグ | 説明 |
| --- | --- |
| `--structure <STRUCTURE>` | 入力 extXYZ ファイル(単一フレーム)。 |
| `--energy` | 全エネルギーを計算する(テキストモードでは `energy = ...` を表示)。 |
| `--forces [<PATH>]` | 力を要求する。`PATH` なしの場合は `--json` 併用時のみサポート。`PATH` ありの場合はレガシー力 extXYZ 出力も書き出す。 |
| `--forces-out <PATH>` | 力を要求し、`force` 列(およびエネルギーを要求した場合はエネルギー)付き extXYZ ファイルを書き出す。`--forces <PATH>` との併用は不可。 |
| `--stress` | 3x3 応力テンソルを計算する(テキストモードでは eV/angstrom^3、row-major で表示)。 |
| `--virial` | 3x3 ビリアルテンソルを計算する(テキストモードでは eV、row-major で表示)。 |
| `--json` | テキスト出力の代わりに `llmlip-prediction-v1` JSON ドキュメントを stdout に出力する。 |

### finite-difference-forces

解析的な力をエネルギーの中心差分と比較し、
`llmlip-force-finite-difference-report-v1` JSON レポートを stdout に出力する。

| フラグ | 既定値 | 説明 |
| --- | --- | --- |
| `--structure <STRUCTURE>` | 必須 | 入力 extXYZ ファイル(単一フレーム)。 |
| `--fd-eps <FD_EPS>` | `0.0001` | 中心差分の変位。有限かつ正であること。 |
| `--fd-tol <FD_TOL>` | `0.005` | 解析値と差分値の絶対差に対する成分ごとの許容値。有限かつ非負であること。 |
| `--fd-mode <FD_MODE>` | `max_force` | `representative` と `max_force` は絶対値最大の力成分 1 つを検査する。`full_3n` は全 3N 成分を検査する。 |
| `--component-start <N>` | `0` | `full_3n` 実行をスライスする際の先頭成分インデックス。 |
| `--component-limit <N>` | 全て | `--component-start` から検査する成分数の上限。 |
| `--include-components` | off | 全成分のレコードを JSON レポートに含める(`components` 配列)。 |

レポートには `base_energy`、差分統計(`max_abs_delta`、`rms_delta`、各
パーセンタイル)、`failed_component_count`、上位最大 10 件の `worst_components`、
真偽値 `passed` が含まれる。`passed` が `false` でも終了コードは 0 であり、
利用側は `passed` フィールドを読む必要がある。

### benchmark-predict

`--structure <PATH>`(単一フレーム extXYZ)または `--fixture <PATH>`(`structures`
配列を持つ JSON fixture)のいずれか一方が必須で、かつ `--energy`、`--forces`、
`--stress`、`--virial` の少なくとも 1 つが必要。タイミング統計(`mean_ms`、
`median_ms`、`p95_ms`、`min_ms`、`max_ms`、`stdev_ms`、`samples_ms`)、`checksum`、
`measurement_scope` ブロック、`environment` ブロック(論理 CPU 数とスレッド関連
環境変数)を持つ `llmlip-predict-benchmark-v1.1` JSON レポートを stdout に出力する。

| フラグ | 既定値 | 説明 |
| --- | --- | --- |
| `--warmup <N>` | `3` | 計測しないウォームアップ回数。 |
| `--iterations <N>` | `10` | 計測回数。1 以上であること。 |
| `--prepared-graph` | off | 計測前にサポートされるグラフ/トポロジー入力を 1 回だけ準備する。 |
| `--profile-stages` | off | eSEN-OAM のステージ別タイミングカウンタを JSON レポートに出力する。 |
| `--label <LABEL>` | `llmlip_predict` | レポートに記録されるラベル。 |

### batch-predict

`--energy`、`--forces`、`--stress`、`--virial` の少なくとも 1 つが必須。構造ごとの
結果(エネルギー/力/応力/ビリアル、タイミング、原子数、元素種)を持つ
`llmlip-batch-predict-v1` JSON レポートを `--out` に書き出し、進捗を stderr に
表示する。構造単位の失敗は各結果の `error` フィールドに記録され `error_count` に
集計されるだけで、実行全体は中断されない。

| フラグ | 既定値 | 説明 |
| --- | --- | --- |
| `--structures <STRUCTURES>` | 必須 | 複数フレーム extXYZ ファイルへのパス。 |
| `--out <OUT>` | 必須 | バッチ結果全体を書き出す JSON ファイルパス。 |
| `--progress-interval <N>` | `100` | N 構造ごとに進捗を表示する(0 で進捗行を無効化)。 |
| `--profile-stages` | off | eSEN-OAM のステージ別タイミングカウンタを構造ごとに出力する。 |

モデルが `total_charge`/`spin_multiplicity` を必要とする場合、値は extXYZ フレーム
メタデータからフレームごとに読み取られる。CLI フラグ `--total-charge` と
`--spin-multiplicity` が優先され、必須値が欠けたフレームは明示エラーで失敗する。

### validate-model

```sh
$ llmlip validate-model --model examples/model.e3rs
model validation ok
```

`--model` のみを取り、モデルディレクトリが不正な場合は
`error: failed to validate model: ...` を出力して非ゼロ終了する。

## JSON 出力スキーマ(`predict --json`)

`predict --json` のドキュメントは常に次の 5 キーちょうどを含む:

| キー | 型 | 備考 |
| --- | --- | --- |
| `schema_version` | string | `"llmlip-prediction-v1"`。 |
| `energy` | number または null | 全エネルギー。`--energy` を要求しなかった場合は `null`。 |
| `forces` | `[fx, fy, fz]` の配列 または null | 原子 1 つにつき 1 行、入力順。力を要求しなかった場合は `null`。 |
| `stress` | 3x3 配列 または null | row-major。`--stress` を要求しなかった場合は `null`。 |
| `virial` | 3x3 配列 または null | row-major。`--virial` を要求しなかった場合は `null`。 |

要求した量が `null` として出力されることはない。モデルが提供できない場合は
ドキュメントを出力せずエラーで失敗する。

## エラー挙動

全てのエラーは `error: <message>` として stderr に出力され、プロセスは
終了コード 1 で終了する。同梱デモモデルで確認したメッセージ:

- 出力フラグなし: `predict requires at least one of --energy, --forces, --forces-out <PATH>, --stress, or --virial`。
- `PATH` なし `--forces` を `--json` なしで指定: `--forces without PATH is supported only with --json; use --forces-out <PATH> for legacy extXYZ output`。
- `--forces <PATH>` と `--forces-out` の併用: `use only one of --forces [PATH] or --forces-out <PATH>`。
- stress/virial 非対応モデルでは黙ってゼロを返さず明示エラーになる: `prediction failed: synthetic radial model does not support stress or virial prediction`。stress/virial 対応モデルでも完全な PBC が必要。
- FP64 専用モデルへの `--dtype float32`: `prediction failed: prediction compute dtype Float32 is unsupported for this model`。

## 関連ドキュメント

- [docs/MODEL_FORMAT_E3RS.md](MODEL_FORMAT_E3RS.md) — `.e3rs` モデルディレクトリ形式。
- [converters/README.md](../converters/README.md) — upstream チェックポイントの `.e3rs` への変換。
- [docs/PYTHON_ASE_INTERFACE_SPEC.md](PYTHON_ASE_INTERFACE_SPEC.md) — Python/ASE インターフェース。
- [docs/c_abi.ja.md](c_abi.ja.md) — C ABI(`llmlip_capi.h`)。
- [docs/openmm.ja.md](openmm.ja.md) — OpenMM 連携。
