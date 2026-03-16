# フェーズ0: プロジェクト基盤構築 -- 詳細マイルストーン

> **対象プロジェクト**: StyleStream 再現実装
> **フェーズ目的**: フェーズ1以降の Destylizer / Stylizer / Vocoder 実装に必要なリポジトリ基盤・共通インフラを整備する
> **推定総作業量**: 8〜11日
> **前提**: Python 3.13、パッケージ管理は `uv`、GPU環境は NVIDIA A6000 想定（開発時は単一GPU可）

---

## 目次

1. [M0.1 リポジトリ構造設計](#m01-リポジトリ構造設計)
2. [M0.2 Python環境・依存パッケージ整備](#m02-python環境依存パッケージ整備)
3. [M0.3 設定管理システム](#m03-設定管理システム)
4. [M0.4 共通ユーティリティ](#m04-共通ユーティリティ)
5. [M0.5 学習基盤](#m05-学習基盤)
6. [M0.6 外部モデルのダウンロード・キャッシュ戦略](#m06-外部モデルのダウンロードキャッシュ戦略)
7. [依存関係グラフ](#依存関係グラフ)
8. [全体スケジュール](#全体スケジュール)

---

## M0.1 リポジトリ構造設計

### タスク名
リポジトリディレクトリ構成の策定とモジュール分割

### 背景・設計方針

StyleStream は3段階パイプライン（Destylizer / Stylizer / Vocoder）で構成される。各コンポーネントは独立して学習されるが、共通のメルスペクトログラム計算・オーディオI/O・学習ループを共有する。参考実装である F5-TTS（DiT+CFM）や Vocos の構造を踏まえ、コンポーネント単位でモジュールを分割しつつ、共通基盤を `stylestream/utils/` に集約する設計とする。

### 具体的な成果物

以下のディレクトリ構造を作成する（初期段階では `__init__.py` とモジュールのスタブのみ）。

```
StyleStream/
├── configs/                          # 設定ファイル群（M0.3で詳細化）
│   ├── destylizer/
│   │   ├── offline.yaml
│   │   └── streaming.yaml
│   ├── stylizer/
│   │   ├── offline.yaml
│   │   └── streaming.yaml
│   ├── vocoder/
│   │   └── causal_vocos.yaml
│   ├── data/
│   │   ├── libritts.yaml
│   │   ├── emilia.yaml
│   │   └── lmg.yaml
│   └── eval/
│       └── stylestream_test.yaml
│
├── stylestream/                      # メインパッケージ
│   ├── __init__.py
│   ├── destylizer/                   # Destylizer モジュール
│   │   ├── __init__.py
│   │   ├── model.py                  # Conformer + FSQ + ASRデコーダ
│   │   ├── conformer.py              # Conformerブロック実装
│   │   ├── fsq.py                    # 有限スカラー量子化
│   │   └── streaming.py              # ストリーミング版（チャンク因果的注意）
│   │
│   ├── stylizer/                     # Stylizer モジュール
│   │   ├── __init__.py
│   │   ├── model.py                  # DiT + CFM 統合モデル
│   │   ├── dit.py                    # Diffusion Transformer
│   │   ├── cfm.py                    # 条件付きフローマッチング
│   │   ├── style_encoder.py          # WavLM-TDNN スタイルエンコーダ
│   │   └── streaming.py              # ストリーミング版
│   │
│   ├── vocoder/                      # Vocoder モジュール
│   │   ├── __init__.py
│   │   ├── model.py                  # 因果的Vocos
│   │   └── causal_conv.py            # 因果的畳み込みレイヤー
│   │
│   ├── data/                         # データ前処理・ローダー
│   │   ├── __init__.py
│   │   ├── dataset.py                # 共通Datasetクラス
│   │   ├── collator.py               # バッチコレーター
│   │   ├── preprocessing.py          # 前処理パイプライン
│   │   └── augmentation.py           # データ拡張（必要時）
│   │
│   ├── utils/                        # 共通ユーティリティ（M0.4で詳細化）
│   │   ├── __init__.py
│   │   ├── audio.py                  # オーディオI/O
│   │   ├── mel.py                    # メルスペクトログラム計算
│   │   ├── logging.py                # ロギング設定
│   │   ├── checkpoint.py             # チェックポイント管理
│   │   └── hub.py                    # 外部モデルダウンロード・キャッシュ
│   │
│   ├── training/                     # 学習基盤（M0.5で詳細化）
│   │   ├── __init__.py
│   │   ├── trainer.py                # 共通トレーナー
│   │   ├── scheduler.py              # 学習率スケジューラ
│   │   └── distributed.py            # 分散学習ユーティリティ
│   │
│   ├── eval/                         # 評価パイプライン
│   │   ├── __init__.py
│   │   ├── metrics.py                # WER, S-SIM, A-SIM, E-SIM, UTMOS
│   │   └── evaluator.py              # 評価ランナー
│   │
│   └── inference/                    # 推論パイプライン
│       ├── __init__.py
│       ├── offline.py                # オフライン推論
│       └── streaming.py              # ストリーミング推論
│
├── scripts/                          # エントリーポイントスクリプト
│   ├── train_destylizer.py
│   ├── train_stylizer.py
│   ├── train_vocoder.py
│   ├── evaluate.py
│   ├── inference.py
│   └── preprocess_data.py
│
├── tests/                            # テスト群
│   ├── __init__.py
│   ├── test_mel.py
│   ├── test_audio.py
│   ├── test_fsq.py
│   └── test_config.py
│
├── docs/                             # 既存のデモサイト + ドキュメント
├── pyproject.toml                    # 既存（M0.2で拡張）
├── CLAUDE.md                         # 既存
├── README.md                         # 既存
└── LICENSE                           # 既存
```

### 設計判断の根拠

| 判断事項 | 選択 | 根拠 |
|---|---|---|
| パッケージ名 | `stylestream/` | 論文名と一致、pip installable にする |
| コンポーネント分割 | `destylizer/`, `stylizer/`, `vocoder/` | 学習が独立しており、各々で閉じたモジュールにすべき |
| 学習基盤の分離 | `training/` | 3コンポーネント共通の学習ループ・分散設定を集約 |
| `scripts/` の分離 | エントリーポイントをパッケージ外に配置 | `stylestream/` パッケージはライブラリとして import 可能に保つ |
| `configs/` のコンポーネント別分割 | 各コンポーネントにサブディレクトリ | 論文の学習設定が各コンポーネントで大きく異なるため |
| `data/` をパッケージ内に配置 | `stylestream/data/` | 前処理ロジックはモデル仕様（50Hz、メル100ビン等）と密結合 |

### 受け入れ基準

- [ ] 上記ディレクトリ構造が作成され、全 `__init__.py` が配置されている
- [ ] `pip install -e .`（`uv pip install -e .`）でパッケージとしてインストール可能
- [ ] `import stylestream` および各サブモジュールの import がエラーなく成功する
- [ ] `scripts/` 配下のスクリプトがスタブとして存在し、`--help` で引数一覧を表示できる

### 推定作業量

**1日**

### 依存関係

なし（最初に着手するタスク）

---

## M0.2 Python環境・依存パッケージ整備

### タスク名
uv による依存パッケージの定義と仮想環境の構築

### 背景・設計方針

現在の `pyproject.toml` は `pymupdf` のみが依存に含まれている。StyleStream の再現実装には PyTorch エコシステムを中心とした多数のパッケージが必要になる。`uv` で管理し、開発用・学習用・評価用の依存グループを分離することで、環境の再現性と軽量性を両立する。

### 具体的な成果物

**ファイル**: `pyproject.toml`（既存ファイルを拡張）

以下の依存パッケージ群を定義する。

#### コア依存（`dependencies`）

| パッケージ | バージョン目安 | 用途 |
|---|---|---|
| `torch` | `>=2.1` | 深層学習フレームワーク |
| `torchaudio` | `>=2.1` | オーディオ処理、メルスペクトログラム |
| `transformers` | `>=4.36` | HuBERT, WavLM のロード |
| `accelerate` | `>=0.25` | 分散学習、mixed precision |
| `einops` | `>=0.7` | テンソル操作の簡潔な記述 |
| `safetensors` | `>=0.4` | 安全なモデル重み保存・読込 |
| `soundfile` | `>=0.12` | オーディオファイルI/O |
| `librosa` | `>=0.10` | オーディオ前処理（リサンプリング等） |
| `omegaconf` | `>=2.3` | YAML設定管理 |
| `hydra-core` | `>=1.3` | 設定管理フレームワーク |
| `numpy` | `>=1.24` | 数値計算 |
| `scipy` | `>=1.11` | 信号処理 |

#### 学習用依存（`optional-dependencies.train`）

| パッケージ | 用途 |
|---|---|
| `wandb` | 実験追跡 |
| `tensorboard` | 学習曲線可視化 |
| `datasets` | HuggingFace Datasets（Emilia等のロード） |

#### 評価用依存（`optional-dependencies.eval`）

| パッケージ | 用途 |
|---|---|
| `resemblyzer` | 話者類似度（S-SIM） |
| `jiwer` | 単語誤り率（WER） |

#### 開発用依存（`optional-dependencies.dev`）

| パッケージ | 用途 |
|---|---|
| `pytest` | テストフレームワーク |
| `pytest-cov` | テストカバレッジ |
| `ruff` | リンター・フォーマッター |

### Python バージョンに関する注意事項

現在 `.python-version` は `3.13` に設定されている。PyTorch の `3.13` サポート状況を確認し、必要に応じて `3.11` または `3.12` に変更する。`pyproject.toml` の `requires-python` も合わせて調整する。PyTorch 公式の CUDA 対応ホイールが確実に利用できるバージョンを優先する。

### 受け入れ基準

- [ ] `pyproject.toml` に全依存パッケージが定義されている
- [ ] `uv sync` で仮想環境が構築できる
- [ ] `uv sync --extra train --extra eval --extra dev` で全依存を含む環境が構築できる
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` が `True` を返す（GPU環境）
- [ ] `python -c "import transformers, accelerate, einops, torchaudio"` がエラーなく成功する
- [ ] `.python-version` が PyTorch と互換性のあるバージョンに設定されている

### 推定作業量

**1日**

### 依存関係

- M0.1（リポジトリ構造設計）-- パッケージ名・構造が確定していること

---

## M0.3 設定管理システム

### タスク名
Hydra + OmegaConf による階層的ハイパーパラメータ管理システムの構築

### 背景・設計方針

StyleStream は3つの独立したコンポーネントを持ち、各々が異なるハイパーパラメータ（モデル構成、学習設定、データ設定）を持つ。論文セクション10.8に全パラメータが整理されており、これを構造化された YAML 設定として管理する。Hydra を採用することで、コマンドラインからのオーバーライド、設定の合成（compose）、マルチラン実験が可能になる。

### 具体的な成果物

#### 設定ファイル群

**ファイル**: `configs/destylizer/offline.yaml`
```
内容: HuBERTモデルID、レイヤー番号、凍結フラグ、Conformer層数・隠れ層サイズ・FFNサイズ、
      位置エンコーディング種別、FSQレベル、ASRデコーダ設定、
      学習パラメータ（ステップ数、バッチサイズ、最適化器、学習率、ウォームアップ、スケジューラ）
```

**ファイル**: `configs/stylizer/offline.yaml`
```
内容: DiT層数・隠れ層・FFN、メルビン数・ホップサイズ・サンプルレート、
      マスク率範囲、CFGドロップ率・強度、NFE・サンプリング方式、
      スタイルエンコーダ設定（WavLMモデルID、TDNN、プーリング）、adaLN-Zero設定、
      学習パラメータ（ステップ数、バッチサイズ、セグメント長等）
```

**ファイル**: `configs/vocoder/causal_vocos.yaml`
```
内容: ベースアーキテクチャ、初期チェックポイント、学習パラメータ、損失関数設定
```

**ファイル**: `configs/data/libritts.yaml`, `configs/data/emilia.yaml`, `configs/data/lmg.yaml`
```
内容: データパス、サンプルレート、前処理パラメータ、セグメント長、データ分割
```

**ファイル**: `configs/eval/stylestream_test.yaml`
```
内容: 評価データセットパス、使用メトリクス、評価モデルID
```

#### 設定スキーマ（dataclass）

**ファイル**: `stylestream/config.py`
```
内容: 各設定ファイルに対応する構造化 dataclass の定義。
      AudioConfig, MelConfig, DestylizerConfig, StylizerConfig, VocoderConfig,
      TrainingConfig, DataConfig, EvalConfig を定義し、型安全な設定アクセスを保証。
```

### 設計判断の根拠

| 判断事項 | 選択 | 根拠 |
|---|---|---|
| 設定管理ツール | Hydra + OmegaConf | MLプロジェクトの標準。CLI オーバーライド、合成、マルチランをサポート |
| 設定の粒度 | コンポーネント別 + データ別 + 評価 | 学習パイプラインごとに異なる組合せが必要（例: Destylizer は LMG データ、Stylizer は Emilia） |
| 型安全 | structured configs（dataclass） | 設定ミスの早期検出。IDE補完が効く |
| パラメータ値 | 論文セクション10.8 の値をデフォルトとして設定 | 再現性の基準点 |

### 論文パラメータとの対応表

以下を `configs/` 内の YAML に正確に反映する。

| 設定項目 | YAML キー例 | 論文値 |
|---|---|---|
| HuBERTモデル | `destylizer.hubert.model_id` | `facebook/hubert-large-ls960-ft` |
| HuBERT抽出層 | `destylizer.hubert.layer` | `18` |
| FSQレベル | `destylizer.fsq.levels` | `[5, 3, 3]` |
| DiT層数 | `stylizer.dit.num_layers` | `16` |
| DiT隠れ層 | `stylizer.dit.hidden_size` | `768` |
| メルビン数 | `audio.mel.n_mels` | `100` |
| ホップサイズ | `audio.mel.hop_length` | `320` |
| サンプルレート | `audio.sample_rate` | `16000` |
| CFG強度 | `stylizer.cfg.strength` | `2` |
| NFE | `stylizer.cfm.nfe` | `16` |

### 受け入れ基準

- [ ] 全設定ファイル（YAML）が `configs/` 配下に作成されている
- [ ] `stylestream/config.py` に structured config の dataclass が定義されている
- [ ] Hydra で設定をロードし、CLI からオーバーライドできる（例: `python scripts/train_destylizer.py destylizer.fsq.levels=[7,5,5] training.batch_size=16`）
- [ ] 設定値が論文セクション10.8のパラメータ一覧と一致している
- [ ] 不正な設定値（型不一致、範囲外）に対してバリデーションエラーが発生する
- [ ] `tests/test_config.py` で設定のロード・オーバーライド・バリデーションのテストが通る

### 推定作業量

**2日**

### 依存関係

- M0.1（リポジトリ構造設計）-- ディレクトリ構造が確定していること
- M0.2（依存パッケージ）-- `hydra-core`, `omegaconf` がインストールされていること

---

## M0.4 共通ユーティリティ

### タスク名
メルスペクトログラム計算、オーディオI/O、ロギングの共通モジュール実装

### 背景・設計方針

StyleStream の全コンポーネントは50Hzフレームレートのメルスペクトログラム（100ビン、ホップサイズ320、16kHz）を共有する。メル計算のパラメータ不一致は品質劣化の主要リスク（論文セクション10.7）であるため、単一の実装を全コンポーネントで共有し、Vocos 公式設定との整合性を保証する必要がある。

### 具体的な成果物

#### 4-a. メルスペクトログラム計算

**ファイル**: `stylestream/utils/mel.py`

| 機能 | 仕様 |
|---|---|
| `MelSpectrogramTransform` クラス | `torch.nn.Module` として実装。学習・推論で同一インスタンスを使用 |
| パラメータ | `n_mels=100`, `hop_length=320`, `n_fft=1024`, `sample_rate=16000`, `f_min=0`, `f_max=8000` |
| 出力形状 | `(batch, 100, T)` -- 50Hz |
| 正規化 | 対数メル（log-mel）。Vocos の正規化方式に準拠 |
| 逆変換 | 不要（Vocoder がメルから波形を生成するため） |

**設計上の注意点**:
- `torchaudio.transforms.MelSpectrogram` をベースに構築
- `n_fft=1024` は論文に明記されていないが、Vocos デフォルトに準拠（論文セクション10.2）
- `f_max=8000` は16kHzサンプリングのナイキスト周波数（8000Hz）に合わせる
- パラメータは `configs/` の YAML から注入可能にするが、デフォルト値は上記に固定

#### 4-b. オーディオI/O

**ファイル**: `stylestream/utils/audio.py`

| 機能 | 仕様 |
|---|---|
| `load_audio(path, sr=16000)` | 任意フォーマットを読み込み、モノラル・指定サンプルレートに変換して返す |
| `save_audio(path, waveform, sr=16000)` | テンソルを WAV ファイルとして保存 |
| `resample(waveform, orig_sr, target_sr)` | リサンプリング（`torchaudio.functional.resample`） |
| `segment_audio(waveform, sr, segment_sec)` | 固定長セグメントに分割（Stylizer: 6秒、Vocoder: 2秒） |
| `pad_or_trim(waveform, target_length)` | 指定長へのパディングまたはトリミング |

**設計上の注意点**:
- `torchaudio.load` と `soundfile` をバックエンドとして使用
- 全ての入出力は `torch.Tensor` 型。NumPy 変換は内部で吸収
- ファイルパスは `pathlib.Path` で統一

#### 4-c. ロギング設定

**ファイル**: `stylestream/utils/logging.py`

| 機能 | 仕様 |
|---|---|
| `setup_logger(name, level)` | Python 標準 `logging` の設定。ファイル出力とコンソール出力を併用 |
| フォーマット | `[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s` |
| 分散学習対応 | rank 0 のみコンソール出力、全ランクでファイル出力 |
| wandb/tensorboard 統合 | M0.5 の学習基盤で統合。ここではロガーインターフェースのみ定義 |

### 受け入れ基準

- [ ] `MelSpectrogramTransform` が `(batch, waveform_length)` 入力に対して `(batch, 100, T)` を返す
- [ ] 50Hz であることを検証: `T == waveform_length // 320` が成立する
- [ ] `load_audio` が WAV/FLAC/MP3 を読み込み、16kHz モノラルの `torch.Tensor` を返す
- [ ] `save_audio` → `load_audio` のラウンドトリップでデータが保持される
- [ ] メルスペクトログラムが Vocos の公式実装と同一パラメータで計算される
- [ ] `tests/test_mel.py` でメル計算の形状・値の正当性テストが通る
- [ ] `tests/test_audio.py` でオーディオI/Oのラウンドトリップテストが通る
- [ ] ロガーが設定通りに動作し、分散学習時に rank 0 のみコンソール出力される

### 推定作業量

**2日**

### 依存関係

- M0.1（リポジトリ構造設計）-- `stylestream/utils/` の構造が確定していること
- M0.2（依存パッケージ）-- `torch`, `torchaudio`, `soundfile`, `librosa` がインストールされていること
- M0.3（設定管理）-- メルパラメータを設定から注入する仕組みが確定していること

---

## M0.5 学習基盤

### タスク名
分散学習、チェックポイント管理、実験追跡の共通学習インフラ構築

### 背景・設計方針

StyleStream の3コンポーネントはいずれも AdamW + コサインアニーリング + マルチGPU で学習される。共通の学習ループを構築し、コンポーネント固有のロジック（損失関数、データローダー構成）のみを差し替える設計とする。`accelerate` を分散学習バックエンドとして採用し、DDP/FSDP の切替をコード変更なしで実現する。

### 具体的な成果物

#### 5-a. 共通トレーナー

**ファイル**: `stylestream/training/trainer.py`

| 機能 | 仕様 |
|---|---|
| `BaseTrainer` クラス | 共通学習ループの抽象基底クラス |
| 学習ループ | ステップベース（エポックベースではない）。論文の学習設定に準拠 |
| 抽象メソッド | `compute_loss(batch)`, `build_model()`, `build_dataloader()` |
| mixed precision | `accelerate` の `mixed_precision="bf16"` をデフォルトに |
| gradient accumulation | 設定から指定可能（実効バッチサイズの調整用） |
| gradient clipping | `max_grad_norm` を設定から指定可能 |
| ログ間隔 | ステップごとの損失、定期的なバリデーション |

#### 5-b. 学習率スケジューラ

**ファイル**: `stylestream/training/scheduler.py`

| 機能 | 仕様 |
|---|---|
| コサインアニーリング | 線形ウォームアップ + コサイン減衰。論文の全コンポーネント共通 |
| パラメータ | `peak_lr`, `warmup_steps`, `total_steps`, `min_lr` |
| Destylizer 用デフォルト | `peak_lr=1e-4`, `warmup_steps=4000`, `total_steps=100000` |
| Stylizer 用デフォルト | `peak_lr=1e-4`, `warmup_steps=2000`, `total_steps=400000` |

#### 5-c. チェックポイント管理

**ファイル**: `stylestream/utils/checkpoint.py`

| 機能 | 仕様 |
|---|---|
| 保存内容 | モデル重み、オプティマイザ状態、スケジューラ状態、ステップ数、設定 |
| 保存形式 | `safetensors`（モデル重み）+ `torch.save`（オプティマイザ・スケジューラ） |
| 保存戦略 | 定期保存（N ステップごと）+ ベストモデル保存（バリデーション損失基準） |
| 保持数 | 直近 K 個を保持、古いものは自動削除（設定可能） |
| 再開 | `resume_from` 指定でチェックポイントから学習再開 |
| ストリーミング版への引継ぎ | オフラインチェックポイントからストリーミング版を初期化するユーティリティ |

**チェックポイントディレクトリ構造**:
```
outputs/
└── {experiment_name}/
    └── {timestamp}/
        ├── config.yaml            # 実験設定のスナップショット
        ├── checkpoints/
        │   ├── step_10000/
        │   │   ├── model.safetensors
        │   │   ├── optimizer.pt
        │   │   └── scheduler.pt
        │   ├── step_20000/
        │   └── best/
        └── logs/
            ├── train.log
            └── events.out.tfevents.*
```

#### 5-d. 分散学習ユーティリティ

**ファイル**: `stylestream/training/distributed.py`

| 機能 | 仕様 |
|---|---|
| 初期化 | `accelerate.Accelerator` のラッパー |
| GPU 割当 | 設定またはコマンドラインから GPU 数を指定 |
| 起動方法 | `accelerate launch scripts/train_*.py` |
| 対応構成 | 単一GPU、マルチGPU DDP、マルチGPU FSDP（大規模モデル用） |

**論文の学習構成との対応**:

| コンポーネント | GPU数 | 対応方法 |
|---|---|---|
| Destylizer | 8台 A6000 | DDP + gradient accumulation（少数GPU時） |
| Stylizer | 8台 A6000 | DDP + gradient accumulation（少数GPU時） |
| Vocoder | 2台 A6000 | DDP |

#### 5-e. 実験追跡

**ファイル**: `stylestream/training/trainer.py` 内に統合

| 機能 | 仕様 |
|---|---|
| wandb | プロジェクト名、実験名、設定のログ。損失・学習率・勾配ノルムを自動記録 |
| tensorboard | wandb が使えない環境のフォールバック |
| 切替 | 設定ファイルで `logger: wandb` または `logger: tensorboard` を指定 |
| 音声サンプル | バリデーション時に変換音声サンプルを wandb/tensorboard に記録（フェーズ1以降で実装） |

### 受け入れ基準

- [ ] `BaseTrainer` を継承したダミートレーナーで、単一GPU での学習ループが動作する
- [ ] `accelerate launch --num_processes=2` で2GPU DDP 学習が動作する
- [ ] チェックポイントの保存・読込・学習再開が正常に動作する
- [ ] コサインアニーリングスケジューラが論文の設定（ウォームアップ + コサイン減衰）通りに動作する
- [ ] wandb にログが記録される（wandb がインストールされている場合）
- [ ] tensorboard にイベントファイルが出力される
- [ ] gradient accumulation により、少数GPU でも大バッチサイズを模擬できる
- [ ] `outputs/` ディレクトリに期待通りの構造でチェックポイントと設定が保存される

### 推定作業量

**3日**

### 依存関係

- M0.1（リポジトリ構造設計）-- `stylestream/training/` の構造が確定していること
- M0.2（依存パッケージ）-- `torch`, `accelerate`, `wandb`, `tensorboard`, `safetensors` がインストールされていること
- M0.3（設定管理）-- 学習パラメータを設定から注入する仕組みが確定していること
- M0.4（共通ユーティリティ）-- ロギング、チェックポイント管理の基盤が確定していること

---

## M0.6 外部モデルのダウンロード・キャッシュ戦略

### タスク名
HuggingFace Hub からの事前学習モデルのダウンロード・キャッシュ・ロードの統一管理

### 背景・設計方針

StyleStream は学習・推論・評価の各段階で計7つの外部モデルを使用する。これらを統一されたインターフェースで管理し、ダウンロードの冪等性、キャッシュの再利用、オフライン環境への対応を保証する。

### 外部モデル一覧と使用段階

| モデル | HuggingFace ID | 使用段階 | 用途 | 推定サイズ |
|---|---|---|---|---|
| HuBERT-Large-ASR | `facebook/hubert-large-ls960-ft` | 学習・推論 | Destylizer 入力（第18層抽出） | 約1.2GB |
| WavLM-Base-Plus-SV | `microsoft/wavlm-base-plus-sv` | 学習・推論 | スタイルエンコーダ | 約360MB |
| Vocos | `gemelo-ai/vocos` (charfi/vocos-mel-24khz) | 学習 | Vocoder ウォームスタート | 約55MB |
| Whisper-large-v3 | `openai/whisper-large-v3` | 評価 | WER 計算 | 約3GB |
| Resemblyzer | `resemble-ai/Resemblyzer` | 評価 | S-SIM 計算 | 約20MB |
| accent-id ECAPA | `Jzuluaga/accent-id-commonaccent-ecapa` | 評価 | A-SIM 計算 | 約80MB |
| emotion2vec | `ddlBoJack/emotion2vec` | 評価 | E-SIM 計算 | 約300MB |

### 具体的な成果物

**ファイル**: `stylestream/utils/hub.py`

| 機能 | 仕様 |
|---|---|
| `ModelRegistry` クラス | 全外部モデルのID・用途・ロード方法を一元管理する辞書 |
| `download_model(model_key)` | 指定モデルをダウンロードしてキャッシュパスを返す |
| `load_hubert(device, layer=18)` | HuBERT-Large をロードし、中間層抽出フックを設定して返す |
| `load_wavlm(device)` | WavLM-Base-Plus-SV をロードして返す（凍結済み） |
| `load_vocos_checkpoint()` | Vocos の重みを辞書として返す（因果的Vocos の初期化用） |
| `download_all(stage)` | 指定段階（`train`, `eval`, `all`）に必要な全モデルを一括ダウンロード |
| `verify_cache()` | キャッシュ済みモデルの一覧と整合性チェック |

### キャッシュ戦略

| 項目 | 方針 |
|---|---|
| キャッシュディレクトリ | HuggingFace デフォルト（`~/.cache/huggingface/hub/`） |
| 環境変数 | `HF_HOME` で変更可能（共有ストレージ環境向け） |
| オフライン対応 | `HF_HUB_OFFLINE=1` 設定時はキャッシュのみ使用 |
| 事前ダウンロード | `scripts/download_models.py` スクリプトで一括取得。学習開始前にすべてのモデルが利用可能であることを保証 |
| バージョン固定 | 各モデルの `revision`（コミットハッシュ）を `configs/` 内で固定し、再現性を保証 |

### 具体的な成果物（追加）

**ファイル**: `scripts/download_models.py`

| 機能 | 仕様 |
|---|---|
| コマンド | `python scripts/download_models.py --stage train` |
| 引数 | `--stage {train, eval, all}` で必要なモデルのみをダウンロード |
| 出力 | ダウンロード状況のプログレスバー、キャッシュパスの表示 |
| 検証 | ダウンロード後にモデルのロード可能性を検証 |

### HuBERT 中間層抽出の設計メモ

Destylizer は HuBERT-Large の**第18層**の隠れ状態を入力とする。`transformers` の `HubertModel` は `output_hidden_states=True` で全層の隠れ状態を返す。抽出のための実装方針:

- `HubertModel.from_pretrained()` でロード
- `forward()` で `output_hidden_states=True` を指定
- `hidden_states[18]` を取得（0-indexed で CNN 特徴量抽出層 + トランスフォーマー12〜24層のうち第18層）
- 凍結状態で使用（`model.eval()`, `torch.no_grad()`、ストリーミング版では凍結解除）

### 受け入れ基準

- [ ] `ModelRegistry` に全7モデルが登録されている
- [ ] `scripts/download_models.py --stage train` で学習用モデル（HuBERT, WavLM, Vocos）がダウンロードされる
- [ ] `scripts/download_models.py --stage eval` で評価用モデル（Whisper, Resemblyzer, accent-id, emotion2vec）がダウンロードされる
- [ ] `load_hubert(device, layer=18)` が HuBERT の第18層隠れ状態を抽出する関数を返す
- [ ] `load_wavlm(device)` が凍結済み WavLM モデルを返す
- [ ] `HF_HUB_OFFLINE=1` 設定時にキャッシュ済みモデルがロードできる
- [ ] `verify_cache()` が未ダウンロードモデルを検出してレポートする
- [ ] ダウンロード済みモデルのリビジョン（コミットハッシュ）が設定ファイルで管理されている

### 推定作業量

**2日**

### 依存関係

- M0.1（リポジトリ構造設計）-- `stylestream/utils/hub.py` のパスが確定していること
- M0.2（依存パッケージ）-- `transformers`, `safetensors` がインストールされていること
- M0.3（設定管理）-- モデルIDとリビジョンを設定で管理する仕組みが確定していること

---

## 依存関係グラフ

```
M0.1 リポジトリ構造設計
 │
 ├──→ M0.2 Python環境・依存パッケージ
 │     │
 │     ├──→ M0.3 設定管理システム
 │     │     │
 │     │     ├──→ M0.4 共通ユーティリティ
 │     │     │     │
 │     │     │     └──→ M0.5 学習基盤
 │     │     │
 │     │     └──→ M0.6 外部モデル管理
 │     │
 │     └──→ M0.6 外部モデル管理
 │
 └──→ M0.3 設定管理システム
```

**クリティカルパス**: M0.1 → M0.2 → M0.3 → M0.4 → M0.5

**並列可能なタスク**:
- M0.4（共通ユーティリティ）と M0.6（外部モデル管理）は M0.3 完了後に並列着手可能
- M0.5（学習基盤）は M0.4 完了後に着手（M0.6 とは独立）

---

## 全体スケジュール

| 日程 | タスク | 並列度 |
|---|---|---|
| Day 1 | M0.1 リポジトリ構造設計 | 単独 |
| Day 2 | M0.2 Python環境・依存パッケージ | 単独 |
| Day 3-4 | M0.3 設定管理システム | 単独 |
| Day 5-6 | M0.4 共通ユーティリティ + M0.6 外部モデル管理（並列） | 2タスク並列 |
| Day 7-8 | M0.4 完了 + M0.6 完了 | -- |
| Day 9-11 | M0.5 学習基盤 | 単独 |

**合計: 8〜11日**（1人作業の場合）

### フェーズ0完了の定義

以下の全てが達成された状態をフェーズ0完了とする:

1. `uv sync --extra train --extra dev` で環境が構築できる
2. `import stylestream` が成功し、全サブモジュールが import 可能
3. 設定ファイルが Hydra でロード・オーバーライドできる
4. メルスペクトログラム計算が50Hz・100ビンで正しく動作する
5. ダミーモデルで単一GPU/マルチGPU学習ループが完走する
6. チェックポイントの保存・再開が動作する
7. HuBERT / WavLM / Vocos がダウンロード・ロードできる
8. 全テスト（`pytest tests/`）がパスする

### フェーズ1への接続

フェーズ0完了後、フェーズ1（データ前処理パイプライン）では以下の基盤を直接活用する:

| フェーズ0の成果物 | フェーズ1での活用 |
|---|---|
| `stylestream/utils/mel.py` | 全データセットのメルスペクトログラム事前計算 |
| `stylestream/utils/audio.py` | 音声ファイルの読込・リサンプリング・セグメント分割 |
| `stylestream/utils/hub.py` | HuBERT 特徴量の事前抽出（Destylizer 学習データ用） |
| `stylestream/data/` | Dataset / DataLoader の実装 |
| `configs/data/` | データセット固有の設定 |
