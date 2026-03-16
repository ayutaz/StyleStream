# CLAUDE.md

このファイルはClaude Code (claude.ai/code) がこのリポジトリで作業する際のガイダンスを提供します。

## プロジェクト概要

StyleStreamはUC Berkeley Speech Groupによるリアルタイムゼロショット音声スタイル（音色・アクセント・感情）変換システムです。
- 論文: http://arxiv.org/abs/2602.20113
- デモ: https://berkeley-speech-group.github.io/StyleStream/
- ライセンス: 研究目的のみ、商用利用不可

## 現在の状態

フェーズ0（プロジェクト基盤構築）完了。メインパッケージ `stylestream/` にモデル・学習・評価の基盤コードが整備済み。フェーズ1（データ前処理パイプライン）に着手可能。

## アーキテクチャ（論文より）

StyleStreamは3段階パイプラインを使用: **Destylizer → Stylizer → Vocoder**

- **Destylizer**: HuBERT-Large第18層 → Conformerブロック×6 → FSQ [5,3,3]（コードブック45）。ASR損失で学習。50Hzで連続pre-quantization特徴量を出力。
- **Stylizer**: 16層DiT（hidden 768, FFN 3072）とConditional Flow Matching (CFM)。WavLM-TDNNエンコーダ + adaLN-Zeroによるスタイル条件付け。スペクトログラムインペインティング目的関数。
- **Vocoder**: Causal Vocos（ConvNextブロックをcausal convolutionに変更）。公式Vocosチェックポイントからwarm start。

全コンポーネントが50Hzフレームレートで統一。メルスペクトログラム: 100ビン, ホップサイズ320, 16kHz。

主な機能:
- ゼロショット変換（新しい話者/スタイルへのファインチューニング不要）
- リアルタイムストリーミング（エンドツーエンドレイテンシ約1秒、chunked-causal attention、600msチャンク）
- 音色・アクセント・感情の変換に対応

## 論文分析ドキュメント

`docs/paper_analysis.md` に10人の専門家エージェントによる詳細分析があります:
- 全アーキテクチャ仕様とハイパーパラメータ
- 学習パイプライン（データセット、依存関係、計算コスト）
- 全アブレーション実験の結果と知見
- 再現実装計画（リスクと対策を含む）

## リポジトリ構造

- `stylestream/` — メインPythonパッケージ
  - `config.py` — 全構造化設定dataclass（AudioConfig, MelConfig, Destylizer/Stylizer/VocoderConfig等）
  - `destylizer/` — Destylizerモジュール（Conformer + FSQ + ASRデコーダ）
  - `stylizer/` — Stylizerモジュール（DiT + CFM + スタイルエンコーダ）
  - `vocoder/` — ボコーダモジュール（Causal Vocos）
  - `data/` — データ前処理・ローダー
  - `utils/` — 共通ユーティリティ
    - `mel.py` — MelSpectrogramTransform（100ビン, hop 320, 50Hz）
    - `audio.py` — オーディオI/O（load/save/resample/segment）
    - `logging.py` — ロギング設定（分散学習対応）
    - `checkpoint.py` — CheckpointManager（safetensors + torch）
    - `hub.py` — 外部モデル管理（HuBERT, WavLM, Vocos等7モデル）
  - `training/` — 学習基盤
    - `trainer.py` — BaseTrainer（accelerate, step-based, DDP/FSDP）
    - `scheduler.py` — CosineAnnealingWarmup
    - `distributed.py` — 分散学習ユーティリティ
  - `eval/` — 評価パイプライン
  - `inference/` — 推論パイプライン
- `configs/` — Hydra YAML設定ファイル
  - `destylizer/` — offline.yaml, streaming.yaml
  - `stylizer/` — offline.yaml, streaming.yaml
  - `vocoder/` — causal_vocos.yaml
  - `data/` — libritts.yaml, emilia.yaml, lmg.yaml
  - `eval/` — stylestream_test.yaml
- `scripts/` — エントリーポイントスクリプト
- `tests/` — テスト（pytest）
- `docs/` — 静的デモWebサイト + 論文分析 + マイルストーン

## 開発環境

- Python 3.12、パッケージ管理: `uv`
- 依存パッケージ: `uv sync` でコア依存をインストール
- 全依存（学習+評価+開発）: `uv sync --extra train --extra eval --extra dev`
- テスト実行: `uv run pytest tests/`
- コア依存: torch, torchaudio, transformers, accelerate, einops, hydra-core, omegaconf

## コマンド

```bash
# 環境構築
uv sync --extra train --extra eval --extra dev

# テスト
uv run pytest tests/ -v

# モデルダウンロード
uv run python scripts/download_models.py --stage train
uv run python scripts/download_models.py --list

# 学習（未実装、スタブのみ）
uv run python scripts/train_destylizer.py --help
```
