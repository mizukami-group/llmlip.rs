# llmlip.rs

[English](README.md)

固定重みの機械学習原子間ポテンシャルを実行するRust推論エンジンです。
エネルギーと保存力、一部のモデルでは応力・ビリアルを計算します。
Rustライブラリ・CLI・C ABIの実行にはPython/PyTorchは不要です。
Python/ASE/OpenMMアダプターと上流チェックポイントの変換にはPythonが必要です。

このソースはソースのみの公開に向けた準備版です。ビルド済みCLI、wheel、Python sdist、
ネイティブプラグイン、コンテナ、基盤モデル重みは配布対象外です。
ローカルビルド手順は利用者自身がビルドするためのものです。[配布方針](docs/distribution_policy.md)を参照してください。
付属の合成モデルは動作確認用であり、科学計算用ではありません。

## ビルドと実行

Rust 1.89以上とC/C++ビルド環境を使用し、`Cargo.lock`を保持してください。

```bash
cargo build --locked -p llmlip-cli --release
./target/release/llmlip predict \
  --model examples/model.e3rs --structure examples/input.extxyz \
  --energy --forces --json
```

等変演算は[e3nn-core-rs](https://github.com/mizukami-group/e3nn-core-rs)の
v0.1.0コミット`9da369cf1d854da6af1a46928bc953c7966e9382`に固定しています。
初回ビルドには依存クレート取得のためのネットワーク接続が必要です。

## 利用方法

- [CLI](docs/cli.ja.md)
- [Python / ASE](docs/PYTHON_ASE_INTERFACE_SPEC.md)
- [OpenMM / OpenMM-ML](docs/openmm.ja.md)
- [LAMMPS](docs/tutorials/lammps_e3rs_mace_tutorial.md)
- [C ABI](docs/c_abi.ja.md)

MACE、NequIP、Allegro、PET、Nequix、ORB、eSEN、UMAの限定された形式に対応します。
精度モード・周期境界・応力の対応はモデルごとに異なり、非対応の要求はエラーになります。
任意の上流チェックポイントをそのまま読み込むものではありません。
[変換手順](converters/README.md)と[モデル形式](docs/MODEL_FORMAT_E3RS.md)を参照してください。

今回の分離後検証では、合成モデル、各インターフェース、実際に変換したMACE-MP-0b3を確認しました。
全モデル系列の上流比較を再実行したわけではありません。[検証範囲](docs/VALIDATION.md)を参照してください。
上流ライブラリと重みは別途取得し、[重みの取り扱い](docs/model_weight_policy.md)に従ってください。

## ローカル検証

CIは設定していません。

```bash
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
```

Pythonテストは仮想環境で`maturin develop --manifest-path crates/llmlip-py/Cargo.toml`を実行後、
`pytest crates/llmlip-py/tests`で実行します。別途取得する参照モデルが必要なテストは、
モデル不在時に明示的にスキップします。

既定のCargo featureは空です。`parallel`、macOSの`accelerate-blas`などは明示的に指定してください。
埋め込みモデル、Apple Metal、Linux向けバックエンドなどの条件は[英語版](README.md)に記載しています。

個別の表示がないプロジェクトコードは[MIT](LICENSE-MIT)または[Apache-2.0](LICENSE-APACHE)です。
LAMMPSへ直接接続するC++インターフェースファイルはGPL-2.0-onlyです。
[連携コードの配布条件](lammps_plugin/llmlip/README.md#license-and-distribution)を参照してください。
この例外によって、単独のRustライブラリ全体をGPLへ変更するものではありません。
モデルにはそれぞれの上流ライセンスが適用されます。[第三者ライセンス記録](THIRD_PARTY_LICENSES.md)を参照してください。

開発者: Wataru Mizukami、Do Tuan Minh、Kenji Ishihara。
JST COI-NEXT「量子ソフトウェア研究拠点」JPMJPF2014の支援を受けています。

自作部分はMIT／Apache-2.0の選択を維持しますが、ORB実装モジュールはApache-2.0、
LAMMPSの3つのC++連携ファイルはGPL-2.0-onlyの例外です。
LAMMPSにはORBを除外した専用ビルドを使用します。
[ライセンス範囲とビルド方法](docs/ORB_LICENSE_SCOPE.md)、
[実装の出典](docs/SOURCE_PROVENANCE.md)、[第三者通知](THIRD_PARTY_LICENSES.md)を参照してください。
