# llmlip Model Converters / モデル変換ツール

公開版の `llmlip` はソースのみを配布する方針であり、基盤モデルの重みは同梱しません。
これは本プロジェクトの配布方針であり、各モデルの上流ライセンス上の再配布権限とは別です。**UMA / eSEN / MACE-MH-1 / MACE-Polar-1** を使う
場合は、各自でチェックポイントをダウンロードし、ここにあるツールで `.e3rs` 形式に
変換してから、実行時ロードまたはコンパイル時埋め込みで利用します。

```
upstream checkpoint  ──convert──▶  model.e3rs  ──▶  llmlip predict --model model.e3rs ...
(.pt / .model, etc.)                 (manifest.toml + weights/)   または LLMLIP_EMBED_MODEL で焼き込み
```

## What this directory provides

| file | purpose |
|---|---|
| `models.json` | 変換対象モデルの provenance（取得元・ライセンス・SHA256・使用するエクスポータ）レジストリ |
| `convert.py` | 統一フロントエンド（ライセンス表示＋同意・SHA256検証・出力検証つきで各エクスポータを呼ぶ） |

実際の重み変換は `tools/reference/export_*_state.py` が行います（PyTorch と各上流ライブラリが必要）。
`convert.py` はその前後に「ライセンス同意」「チェックポイント同一性検証」「出力検証」を足します。

## Prerequisites

- Python 3.10+
- 変換時のみ: PyTorch と対象モデルの上流ライブラリ
  - UMA / eSEN: `fairchem`（`third_party/fairchem` または `pip install fairchem-core`）
  - MACE-MH-1 / MACE-Polar: `mace`（`third_party/mace` または `pip install mace-torch`）
- 注: `llmlip` の **Rust ランタイム自体は Python 非依存**。Python が要るのは変換ステップだけです。

## Workflow

### 1. モデルを確認
```bash
python converters/convert.py list
python converters/convert.py info uma-s-1p2
```

### 2. チェックポイントをダウンロード（各自・ライセンス同意）
`models.json` の `source.url` から取得します。gated（`facebook/UMA`, `facebook/OMAT24` 等）は
Hugging Face 等でアクセス申請・ライセンス同意が必要です。

### 3. 同一性を検証（任意だが推奨）
```bash
python converters/convert.py verify uma-s-1p2 /path/to/uma-s-1p2.pt
```
SHA256/サイズが不一致なら非ゼロ終了します。警告だけにしたい場合は
`--warn-only` を明示してください。

### 4. `.e3rs` に変換
```bash
python converters/convert.py convert uma-s-1p2 \
    --checkpoint /path/to/uma-s-1p2.pt \
    --out out/uma-s-1p2.e3rs \
    --accept-license
```
`--accept-license` を付けないと変換は実行されません（ライセンスを表示して停止します）。
SHA256/サイズが不一致なら停止します（`--skip-verify` で警告のみに変更可）。
`--dry-run` で実行コマンドだけ確認できます。エクスポータ固有の追加引数は末尾にそのまま渡せます。

### 5. 使う

**実行時ロード（既定）:**
```bash
llmlip predict --model out/uma-s-1p2.e3rs --structure input.extxyz --energy --forces --json
```

**コンパイル時埋め込み（自己完結バイナリ／プラグイン）:**
```bash
LLMLIP_EMBED_MODEL="$PWD/out/uma-s-1p2.e3rs" \
    cargo build -p llmlip-cli --release --features embed-model
# 以後、実行時にモデルパス不要（LlmlipModel::load_embedded() が使えます）
```

## Per-model notes

| model key | architecture | source | gated | exporter |
|---|---|---|---|---|
| `uma-s-1p2` | `uma-v1` | HF `facebook/UMA` | ✅ | `export_uma_state.py` |
| `esen-30m-oam` | `esen-oam-v1` | HF `facebook/OMAT24` | ✅ | `export_esen_oam_state.py` |
| `mace-mh-1` | `mace-mh1-v1` | GitHub `ACEsuit/mace-foundations` release | — | `export_mace_mh1_state.py` |
| `mace-polar-1-s` | `mace-polar-v1` | GitHub release `mace_polar_1` (`MACE-POLAR-1-S.model`) | — | `export_polarmace_state.py` |
| `mace-polar-1-m` | `mace-polar-v1` | GitHub release `mace_polar_1` (`MACE-POLAR-1-M.model`) | — | `export_polarmace_state.py` |
| `mace-polar-1-l` | `mace-polar-v1` | GitHub release `mace_polar_1` (`MACE-POLAR-1-L.model`) | — | `export_polarmace_state.py` |

- **MACE-MH-1**: 取得元として GitHub release（レジストリで SHA256 固定、こちらが
  パリティ検証済みの正）と Hugging Face（**別ペイロード**、`alternate_sources` に
  SHA256 記録）が存在します。HF 版を変換する場合は `--skip-verify` が必要ですが、
  パリティは未検証です。エクスポータは `third_party/mace`（pinned commit、
  `scripts/add_mace_reference_submodule.sh` で取得）を要求します。
- **MACE-Polar-1 (s/m/l)**: エクスポータは checkpoint そのものではなく
  artifact-manifest JSON を取ります。`convert.py` がレジストリの
  `manifest_template`（コミット済みの pinned manifest）にあなたの checkpoint
  パスを差し込んだ一時 manifest を自動生成して渡すので、使い方は他モデルと
  同じ `--checkpoint /path/to/MACE-POLAR-1-S.model` で構いません。
  pinned SHA256 はエクスポータ側でも再検証されます。
- **eSEN-30M-OAM**: FP64 CPU の energy/forces はモノレポ内で上流パリティ検証済み
  です。stress/virial・FP32 は未対応です。

詳細は各 `tools/reference/export_*_state.py` の `--help` と、
`docs/MACE_MH1_ARTIFACT.md` / `docs/POLARMACE_ARTIFACT.md` / `docs/UMA_S_1P2_SUPPORT_PLAN.md`
を参照してください。必要に応じて末尾の passthrough 引数で渡せます。

## Output validation

変換後、`convert.py` は次の順で出力を検証します。

1. ビルド済みの `llmlip` CLI（`LLMLIP_BINARY`、`target/{release,debug}/llmlip`、
   または `PATH` 上）があれば `llmlip validate-model --model <out>` を実行。
2. なければ最小検証: `manifest.toml` の `architecture`、参照される state JSON、
   `[[tensors]]` の各 `path` の実在を確認します。

## Licensing

各チェックポイントのライセンスは取得元で確認し、同意のうえ各自取得してください。
本リポジトリは重みを再配布しません。`models.json` の `license` フィールドは参考情報であり、
`redistribution.policy` は本プロジェクトの非配布方針を表します。上流が条件付きで与える
再配布権限の有無を意味しません。PETを含め、今回はモデルassetを配布しません。モデル重みの扱いは
[重みの取り扱い](../docs/model_weight_policy.md)を参照してください。

## 分離後の変換確認

登録モデルの一覧とコマンド生成に加え、MACE-MP-0b3の専用ツールで
実チェックポイントからの変換を確認しました。MACE v0.3.16、e3nn 0.4.4、
PyTorch 2.8を使った参照環境での例です。各パスは別途取得したデータを指定します。

```bash
python tools/reference/generate_mace_fixtures.py \
  --mace-dir /path/to/mace \
  --e3nn-compat-path /path/to/e3nn-0.4.4-install \
  --artifact /path/to/mace-mp-0b3-medium.model \
  --emit-mp0b-runtime-model \
  --mp0b-runtime-model-out outputs/mp0b.e3rs
```

MACE-MP-0b3の参照JSONを別途用意した場合、Pythonバインディングを
インストールした環境で以下を実行できます。

```bash
python tools/python/check_model_parity.py \
  --model outputs/mp0b.e3rs --reference /path/to/mace_mp0b_layer_flow_reference_f64.json
```

他のモデル系列の実変換は、この分離作業では再実行していません。
上流の依存ライブラリやチェックアウトはソースに同梱しません。
