# OpenMM から llmlip モデルを使う

[English](openmm.md) | 日本語

`llmlip` Python パッケージは、固定重み `.e3rs` モデルを力場として
[OpenMM](https://openmm.org) シミュレーションを駆動できます。統合レイヤは
2 つ同梱されています。[OpenMM-ML](https://github.com/openmm/openmm-ml)
アダプタ(`llmlip.openmmml`、ポテンシャル名 `llmlip` で登録)と、
`llmlip.Model.predict()` を直接呼ぶコールバックを持つ
`openmm.PythonForce` を構築する低レベルヘルパ
(`llmlip.openmm.create_llmlip_force`)です。
`PythonForce` を避けたいホストコード向けには、
[openmm_plugin/llmlip](../openmm_plugin/llmlip/README.md) にネイティブ C++
C ABI bridge も用意しています。

## 前提

パッケージはまだ PyPI に公開されていません。リポジトリルートで
[maturin](https://www.maturin.rs) を使ってビルドし、
[crates/llmlip-py/pyproject.toml](../crates/llmlip-py/pyproject.toml) の
optional-dependency グループ `openmm` が宣言する OpenMM 依存を
インストールしてください:

```bash
pip install maturin numpy
maturin develop --release --manifest-path crates/llmlip-py/Cargo.toml
pip install "openmm>=8.5" "openmmml>=1.6"
```

Apple Silicon で ASE/OpenMM の既定経路から baked MACE-OAM Medium を使う
Python extension は、baked と Metal の feature を有効にしてビルドします:

```bash
maturin develop --release \
  --manifest-path crates/llmlip-py/Cargo.toml \
  --features baked-mace-oam-medium,apple-metal,accelerate-blas,parallel
```

実行時は、利用者が import 前に別の値を設定しない限り、`llmlip` が
`LLMLIP_MACE_METAL=1` を既定値として設定します。

force 生成時に検査される要件:

- `openmm.PythonForce` を提供する OpenMM ビルドであること。なければ即座に
  `ImportError` を送出します。
- OpenMM-ML が必要なのは `llmlip.openmmml` アダプタのみです。ファクトリは
  `openmmml.potentials` entry point
  (`llmlip = "llmlip.openmmml:LlmlipPotentialImplFactory"`)経由で公開され、
  `openmmml.MLPotential.registerImplFactory("llmlip", LlmlipPotentialImplFactory())`
  による明示登録もできます。

## 最小の動く例

同梱の synthetic デモモデル
[examples/model.e3rs](../examples/model.e3rs)(species は H と O、float64
のみ)を使います:

```python
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
from openmmml import MLPotential

topology = app.Topology()
chain = topology.addChain()
residue = topology.addResidue("MOL", chain)
for i, symbol in enumerate(["O", "H", "H"]):
    topology.addAtom(f"{symbol}{i}", app.Element.getBySymbol(symbol), residue)

potential = MLPotential("llmlip", modelPath="examples/model.e3rs")
system = potential.createSystem(topology, dtype="float64", removeCMMotion=False)

positions_ang = np.array(
    [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.927297, 0.0]]
)
context = openmm.Context(
    system,
    openmm.VerletIntegrator(0.001),
    openmm.Platform.getPlatformByName("Reference"),
)
context.setPositions((positions_ang / 10.0) * unit.nanometer)
state = context.getState(getEnergy=True, getForces=True)
print(state.getPotentialEnergy())  # kJ/mol
```

低レベルヘルパは force を system に追加せずに返します。system は
topology の原子数と同数の粒子をあらかじめ持っている必要があります:

```python
from llmlip.openmm import create_llmlip_force

force = create_llmlip_force(topology, system, "examples/model.e3rs")
system.addForce(force)
```

baked MACE-OAM Medium ビルドでは、`modelPath` / `model_path` を省略すると
コンパイル時に埋め込まれたモデルを使います:

```python
potential = MLPotential("llmlip")
system = potential.createSystem(topology, dtype="float32", removeCMMotion=False)

force = create_llmlip_force(topology, system, dtype="float32")
```

## リファレンス

`create_llmlip_force(topology, system, model_path=None, *, ...)` のキーワード
引数。OpenMM-ML の `createSystem`/`createMixedSystem` 経路が同じ名前で
転送するのは `dtype`・`precision`・`head`・`total_charge`・
`spin_multiplicity`・`mace_mh1_wigner_mode` です。force group は OpenMM-ML
自身の `forceGroup` 引数で指定し、`MLPotential` 構築時に渡せるのは
`modelPath`/`model_path` のみ(ロード済み `model` は不可)です:

| 引数 | 既定値 | 意味 |
|---|---|---|
| `model_path` / `model` | 両方省略時は baked MACE-OAM Medium | `.e3rs` モデルへのパス、またはロード済み `llmlip.Model`。両方を渡すとエラー。両方省略時は `llmlip.load_baked_model()` を呼び、`baked-mace-oam-medium` 付きビルドが必要。 |
| `atoms` | `None`(全原子) | ML サブセットのインデックス。非 ML 原子はこの force からゼロの力を受け取る。 |
| `force_group` | `0` | 返される force の OpenMM force group。 |
| `dtype` | `"float64"` | `"float64"`/`"f64"`/`"fp64"` または `"float32"`/`"f32"`/`"fp32"`。 |
| `precision` | `None` | `"double"` または `"single"`(OpenMM 流のエイリアス)。`dtype` と矛盾してはならない。 |
| `head` | `None` | 予測 head。モデルの capabilities に対して検証される。 |
| `total_charge` | `None` | モデルが入力要件として宣言する場合は必須。OpenMM-ML エイリアス: `charge`。 |
| `spin_multiplicity` | `None` | モデルが入力要件として宣言する場合は必須。OpenMM-ML エイリアス: `multiplicity`、`spin`。 |
| `mace_mh1_wigner_mode` | `"true_float64"` | MACE-MH1 専用。代替値: `"upstream_artifact"`。 |

アダプタが行う単位変換(llmlip モデルは Angstrom / eV / eV/Angstrom、
OpenMM は nm / kJ/mol を使用):

| 量 | 方向 | 係数 |
|---|---|---|
| 座標 | nm → Angstrom | × 10 |
| エネルギー | eV → kJ/mol | × 96.4853321233100184 |
| 力 | eV/Angstrom → kJ/mol/nm | × 964.853321233100184 |

## エラー挙動と制約

- エネルギーと力のみ: アダプタは `stress=False, virial=False` で
  `predict()` を呼びます。OpenMM への virial/stress 経路はありません。
- Python アダプタの入口は引き続き `PythonForce` コールバックです。
  extension を `apple-metal` 付きでビルドし、`LLMLIP_MACE_METAL=1` の場合は、
  underlying llmlip runtime が baked Metal 経路を使えます。それ以外は通常の
  CPU 経路です。同梱テストは OpenMM の `Reference` プラットフォームを使用します。
- force 生成時のフェイルファストな capability 検証
  (`UnsupportedCapabilityError`): エネルギーまたは力に非対応のモデル、
  非対応の `dtype`、モデルの対応 species 集合外の topology species、未知の
  `head`、必須の `total_charge`/`spin_multiplicity` の欠落。
- force 生成時の `ValueError`: `model_path` と `model` の両方指定、
  `dtype`/`precision` の矛盾、未知の dtype エイリアス、topology の
  原子数 ≠ system の粒子数、範囲外/重複/非 1 次元の `atoms`、element を
  持たない ML 原子、OpenMM-ML アダプタへの非対応の追加キーワード引数。
- 周期性: topology が周期 box vectors を持つか system が PBC を使う場合、
  box は完全 3D PBC 契約として扱われます。部分 PBC は意図的に非対応です。
  座標は毎回の `predict()` 呼び出し前に primary cell へ wrap されます。
  特異な box や非有限の分数座標は `PredictionError` を送出します。
- OpenMM-ML の mixed system は補間なし
  (`createMixedSystem(..., interpolate=False)`)で動作します。
  `interpolate=True` は `openmm.OpenMMException` を送出します。OpenMM-ML が
  `PythonForce` コールバックを pickle しようとし、コールバックが pickle
  不可能なネイティブ `llmlip.Model` を束縛しているためです。

上記すべてを網羅するテスト:
[crates/llmlip-py/tests/test_openmm.py](../crates/llmlip-py/tests/test_openmm.py)。

## 関連ドキュメント

- ASE calculator と Python API: [docs/PYTHON_ASE_INTERFACE_SPEC.md](PYTHON_ASE_INTERFACE_SPEC.md)
- Python パッケージ概要: [crates/llmlip-py/README.md](../crates/llmlip-py/README.md)
- `.e3rs` モデル形式: [docs/MODEL_FORMAT_E3RS.md](MODEL_FORMAT_E3RS.md)
