# llmlip C ABI(`libllmlip_capi`)

[English](c_abi.md) | 日本語

`llmlip-capi` クレートは llmlip 推論エンジンをプレーンな C ABI として公開します。
`.e3rs` モデルディレクトリを一度ロードし、構造または外部構築した近接グラフに
対してエネルギー / 力 / ビリアル / 応力を要求できます。LAMMPS プラグインや
C/C++/Fortran ホストコードからの統合レイヤです。

## 提供されるもの

| 成果物 | パス(ビルド後) | 備考 |
|---|---|---|
| C ヘッダ | `crates/llmlip-capi/include/llmlip_capi.h` | 単一ヘッダ、C / C++(`extern "C"`)両対応 |
| 静的ライブラリ | `target/release/libllmlip_capi.a` | `crate-type = "staticlib"` |
| 共有ライブラリ | `target/release/libllmlip_capi.dylib`(macOS)/ `.so`(Linux) | `crate-type = "cdylib"` |
| Rust rlib | `target/release/libllmlip_capi.rlib` | Rust から使う場合のみ |

エクスポートされるシンボルはすべて `llmlip_` プレフィックス付きです。
baked MACE-OAM Medium ローダは常に C ABI に存在します。
`baked-mace-oam-medium` を有効にしていないビルドでは、そのシンボルは
消えず、`LLMLIP_ERROR_UNSUPPORTED` を返します。`reference-fixtures` は
内部テスト専用のゲートです。

## ビルドとリンク

```sh
cargo build -p llmlip-capi --release
```

MACE-OAM Medium の重みを埋め込み、Apple Silicon の Metal 経路も有効にする
再配布用ビルドでは、llmlip へ転送される feature を指定します:

```sh
cargo build -p llmlip-capi --release \
  --features baked-mace-oam-medium,apple-metal,accelerate-blas,parallel
```

どちらかの成果物をプログラムにリンクします:

```sh
# 静的リンク(実行時のライブラリパス設定が不要)
cc -std=c11 -Icrates/llmlip-capi/include demo.c target/release/libllmlip_capi.a -o demo

# 動的リンク(実行時に LD_LIBRARY_PATH / DYLD_LIBRARY_PATH を target/release に設定)
cc -std=c11 -Icrates/llmlip-capi/include demo.c -Ltarget/release -lllmlip_capi -o demo
```

静的リンクでプラットフォーム固有のシステムライブラリが追加で必要になる場合、
正確なリストは Rust ツールチェインに問い合わせられます:
`RUSTFLAGS="--print native-static-libs" cargo rustc -p llmlip-capi --release --lib`

## 最小の動く例

`demo.c` として保存し、上記の手順でビルドして、リポジトリルートから実行します
(同梱の synthetic デモモデルは `examples/model.e3rs` にあります):

```c
#include <stdio.h>
#include <llmlip_capi.h>

static int check(llmlip_status_t status, const char *what) {
  if (status != LLMLIP_OK) {
    fprintf(stderr, "%s failed (status %d): %s\n", what, (int)status, llmlip_last_error());
    return 1;
  }
  return 0;
}

int main(void) {
  llmlip_model_t *model = NULL;
  if (check(llmlip_model_load("examples/model.e3rs", &model), "llmlip_model_load")) return 1;
  /* baked 再配布ビルドでは代わりに次を呼べます:
   * llmlip_model_load_baked_mace_oam_medium(&model)
   */

  llmlip_model_capabilities_t caps;
  if (check(llmlip_model_capabilities(model, &caps), "llmlip_model_capabilities")) return 1;
  printf("model cutoff: %.2f\n", caps.model_cutoff);

  /* Water molecule, no periodic boundary conditions. */
  int32_t atomic_numbers[3] = {8, 1, 1};
  double positions[9] = {
      0.000000, 0.000000, 0.0,  /* O */
      0.957200, 0.000000, 0.0,  /* H */
      -0.239987, 0.927297, 0.0, /* H */
  };
  llmlip_structure_view_t structure = {
      .n_atoms_total = 3,
      .n_atoms_owned = 3,
      .atomic_numbers = atomic_numbers,
      .positions = positions,
      .cell = NULL, /* may be NULL when all pbc flags are 0 */
      .pbc = {0, 0, 0},
  };

  llmlip_prediction_options_t options = {
      .compute_energy = 1,
      .compute_forces = 1,
      .dtype = LLMLIP_DTYPE_F64,
      .selected_head = NULL, /* NULL selects the default head */
  };

  double forces[9] = {0}; /* caller-owned: n_atoms_total * 3 doubles */
  llmlip_prediction_out_t out = {0};
  out.forces = forces;

  if (check(llmlip_predict_structure(model, &structure, &options, &out),
            "llmlip_predict_structure")) {
    llmlip_model_free(model);
    return 1;
  }

  if (out.has_energy) printf("energy: %.6f eV\n", out.energy);
  if (out.has_forces) {
    for (int atom = 0; atom < 3; ++atom) {
      printf("force[%d] = (%.6f, %.6f, %.6f)\n", atom, forces[3 * atom + 0],
             forces[3 * atom + 1], forces[3 * atom + 2]);
    }
  }

  llmlip_model_free(model);
  return 0;
}
```

期待される出力: `model cutoff: 5.00`、有限のエネルギー、原子ごとに 1 行の力。

## API リファレンス

| 関数 | 引数 | 戻り値 | エラー時 |
|---|---|---|---|
| `llmlip_model_load` | `model_dir`(`.e3rs` ディレクトリへの NUL 終端 UTF-8 パス)、`out`(ハンドルを受け取る) | `llmlip_status_t` | 非 OK ステータス、`*out` は `NULL` のまま |
| `llmlip_model_load_baked_mace_oam_medium` | `out`(ハンドルを受け取る) | `llmlip_status_t` | 非 OK ステータス、`*out` は `NULL` のまま。`baked-mace-oam-medium` なしでビルドされたライブラリでは `LLMLIP_ERROR_UNSUPPORTED` |
| `llmlip_model_free` | `model` ハンドル | `void` | `NULL` は無視。同じハンドルを二度 free しないこと |
| `llmlip_model_capabilities` | `model`、`out`(`llmlip_model_capabilities_t *`) | `llmlip_status_t` | 非 OK ステータス(`NULL` 引数は `LLMLIP_ERROR_NULL_POINTER`) |
| `llmlip_predict_structure` | `model`、`structure`、`options`、`out` | `llmlip_status_t` | 非 OK ステータス。`out` のフラグは 0 にリセット |
| `llmlip_predict_external_graph` | `model`、`structure`、`graph`、`options`、`out` | `llmlip_status_t` | 非 OK ステータス。`out` のフラグは 0 にリセット |
| `llmlip_last_error` | — | `const char *`(スレッドローカルなメッセージ。なければ空文字列) | — |
| `llmlip_clear_error` | — | `void` | — |

入力の契約(ライブラリが検査し、違反はエラーステータスを返す):

- 数値の入出力はすべて `double`(FP64)。`LLMLIP_DTYPE_F32` は内部の計算精度を
  切り替えるだけで、結果は常に `double` で書き込まれる。
- 力を要求する場合、`out.forces` は呼び出し側所有の `n_atoms_total * 3` 個の
  double バッファ(atom-major xyz)を指すこと。要求しない場合のみ `NULL` 可。
  力が返されるのにバッファが `NULL` ならステータスコードで失敗する。
- `llmlip_predict_structure` は `n_atoms_owned == n_atoms_total` を要求する。
  `cell`(9 個の double、各行が格子ベクトル)はいずれかの `pbc` フラグが 1 の
  とき必須、そうでなければ `NULL` 可。`pbc` の各要素は 0 か 1。
- 外部グラフ: `edge_offsets` は `n_centers + 1` 要素を持ち、エッジ数はその最終
  要素。`edge_vector` は `3 * n_edges` 個の double。`center_owned` の各要素は
  0 か 1。
- `virial` / `stress` は 9 個の double の行優先フィールドで、`has_virial` /
  `has_stress` が 1 のときのみ有効。応力・ビリアルを計算できないモデルは黙って
  ゼロを返さず、明示的なエラーステータスで失敗する。

## ステータスコードとエラー

| コード | 値 | 意味 |
|---|---|---|
| `LLMLIP_OK` | 0 | 成功 |
| `LLMLIP_ERROR_NULL_POINTER` | 1 | 必須ポインタが `NULL`(周期構造で `cell` がない、力バッファがない場合を含む) |
| `LLMLIP_ERROR_INVALID_ARGUMENT` | 2 | 不正な入力データ: 負のカウント、`pbc`/`center_owned` が 0/1 以外、未知の原子番号、不整合な外部グラフ |
| `LLMLIP_ERROR_MODEL_LOAD` | 3 | `.e3rs` ディレクトリをロードできなかった |
| `LLMLIP_ERROR_UNSUPPORTED` | 4 | 要求された計算 dtype・予測ヘッド・外部グラフ予測をモデルがサポートしない |
| `LLMLIP_ERROR_PREDICTION` | 5 | 予測中にエンジンが失敗(例: 応力/ビリアルを計算できないモデルへの要求) |
| `LLMLIP_ERROR_PANIC` | 6 | Rust の panic を ABI 境界で捕捉。プロセスは継続する |

非 OK ステータスの後、`llmlip_last_error()` は人間可読なメッセージを返します。
メッセージはスレッドローカルに保存されるため、ステータスを受け取ったのと同じ
スレッドで、そのスレッドの次の llmlip C ABI 呼び出しより前に読んでください。
ポインタはライブラリ所有で、同一スレッドでの次の llmlip C ABI 呼び出しまたは
`llmlip_clear_error()` まで有効です。free してはいけません。

## スレッド安全性と所有権

- ロード済みモデルは読み取り専用です。ライブラリはハンドルごとのスクラッチ
  バッファをミューテックスで保護するため、同一ハンドルへの複数スレッドからの
  予測呼び出しは予測関数ごとに直列化されます。最も堅実なパターンはスレッドごと
  に 1 ハンドル、または共有ハンドルへの呼び出しを外部で直列化することです。
- 入力配列と力バッファはすべて呼び出し側所有で、各呼び出しの間だけ有効であれば
  十分です。ライブラリが呼び出し側のメモリを free することはありません。
- モデルハンドルはライブラリ所有で、解放は `llmlip_model_free` のみで行います
  (ちょうど 1 回。`NULL` は no-op)。他スレッドが使用中のハンドルを free しない
  でください。

## 関連ドキュメント

- [`.e3rs` モデル形式](MODEL_FORMAT_E3RS.md)
- [公開モデルの `.e3rs` への変換](../converters/README.md)
- [LAMMPS `pair_style llmlip` プラグイン](../lammps_plugin/llmlip/README.md) — この ABI の完全な利用例
- [OpenMM native C ABI bridge](../openmm_plugin/llmlip/README.md)
