# CLAUDE.md

このファイルはClaude Code (claude.ai/code) がこのリポジトリで作業する際のガイダンスを提供します。

## プロジェクト概要

StyleStreamはUC Berkeley Speech Groupによるリアルタイムゼロショット音声スタイル（音色・アクセント・感情）変換システムです。
- 論文: http://arxiv.org/abs/2602.20113
- デモ: https://berkeley-speech-group.github.io/StyleStream/
- ライセンス: 研究目的のみ、商用利用不可

## 現在の状態

フェーズ0（プロジェクト基盤構築）・フェーズ1（データ前処理パイプライン）・フェーズ2（Destylizer実装）・フェーズ3（Stylizer / DiT実装）完了。ALiBi付きConformer×6、FSQ [5,3,3]、CTC/seq2seq ASRデコーダ、学習パイプライン、推論API実装済み。16層DiT、CFM、adaLN-Zero、WavLM-TDNNスタイルエンコーダ、CFG実装済み。次はフェーズ4（Vocoder: Causal Vocos実装）。

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
  - `config.py` — 全構造化設定dataclass
  - `destylizer/` — Destylizerモジュール（実装済み）
    - `alibi.py` — ALiBi位置エンコーディング
    - `conformer.py` — Conformerブロック×6（マカロン構造, ALiBi, 深さ方向分離畳み込み）
    - `fsq.py` — FSQ [5,3,3]（コードブック45, STE勾配伝搬）
    - `asr_head.py` — CTC + seq2seq ASRデコーダ
    - `model.py` — Destylizer統合モデル
    - `trainer.py` — DestylizerTrainer（BaseTrainer拡張）
    - `feature_extractor.py` — 推論時コンテンツ特徴量抽出API
  - `stylizer/` — Stylizerモジュール（実装済み）
    - `rope.py` — RoPE回転位置埋め込み
    - `timestep_embedding.py` — 正弦波+MLP時間ステップ埋め込み
    - `adaln_zero.py` — adaLN-Zero条件付き正規化
    - `style_encoder.py` — WavLM-TDNNスタイルエンコーダ
    - `dit.py` — 16層Diffusion Transformer（adaLN-Zero, RoPE）
    - `cfm.py` — Conditional Flow Matching（OTパス, マスク損失, オイラーサンプリング）
    - `cfg.py` — Classifier-Free Guidance（3条件ドロップ, ガイダンスα=2）
    - `model.py` — Stylizer統合モデル
    - `trainer.py` — StylizerTrainer（BaseTrainer拡張）
  - `vocoder/__init__.py` — ボコーダモジュール（モデル実装はPhase4）
  - `data/` — データ前処理・ローダー
    - `manifest.py` — Manifest/Utterance、LibriTTS/ESD/GLOBE対応
    - `preprocessing.py` — リサンプリング+メル計算パイプライン
    - `hubert_extractor.py` — HuBERT L18特徴量抽出（GPU）
    - `text.py` — CTC用CharTokenizer（30トークン）
    - `destylizer_dataset.py` — DestylizerDataset+BucketBatchSampler
    - `stylizer_dataset.py` — StylizerDataset（6秒,マスク,CFG）
    - `vocoder_dataset.py` — VocoderDataset（2秒,アライメント）
  - `utils/` — 共通ユーティリティ（mel.py, audio.py, logging.py, checkpoint.py, hub.py）
  - `training/` — 学習基盤（trainer.py, scheduler.py, distributed.py）
  - `eval/` — 評価パイプライン
  - `inference/` — 推論パイプライン
- `configs/` — YAML設定ファイル（destylizer, stylizer, vocoder, data, eval）
- `scripts/` — エントリーポイントスクリプト
  - `download_libritts.py`, `download_esd.py`, `download_globe.py` — データセットダウンロード
  - `download_models.py` — 事前学習モデルダウンロード
  - `preprocess_data.py` — 前処理CLI（実装済み）
  - `validate_features.py` — 特徴量検証（実装済み）
  - `train_destylizer.py` — Destylizer学習CLI（実装済み）
  - `train_stylizer.py` — Stylizer学習CLI（実装済み）
  - `train_vocoder.py` — 学習スクリプト（スタブ）
  - `evaluate.py`, `inference.py` — 評価・推論（スタブ）
- `tests/` — 355テスト（mel, audio, text, manifest, datasets, conformer, fsq, asr_head, destylizer_model, rope, timestep_embedding, adaln_zero, dit, style_encoder, cfm, cfg, stylizer_model）
- `docs/` — 静的デモWebサイト + 論文分析 + マイルストーン
- `pyproject.toml`, `CLAUDE.md`, `README.md`, `LICENSE`, `.gitignore`

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

# データセットダウンロード
uv run python scripts/download_libritts.py --output-dir data/raw/libritts
uv run python scripts/download_esd.py --output-dir data/raw/esd

# 前処理
uv run python scripts/preprocess_data.py --manifest data/manifests/libritts.csv --output-dir data/processed

# 特徴量検証
uv run python scripts/validate_features.py --manifest data/manifests/libritts.csv --processed-dir data/processed

# テスト (355件)
uv run pytest tests/ -v

# Destylizer学習
uv run python scripts/train_destylizer.py --config configs/destylizer/offline.yaml

# Stylizer学習
uv run python scripts/train_stylizer.py --config configs/stylizer/offline.yaml

# モデルダウンロード
uv run python scripts/download_models.py --stage train
uv run python scripts/download_models.py --list
```
