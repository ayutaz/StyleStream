# CLAUDE.md

このファイルはClaude Code (claude.ai/code) がこのリポジトリで作業する際のガイダンスを提供します。

## プロジェクト概要

StyleStreamはUC Berkeley Speech Groupによるリアルタイムゼロショット音声スタイル（音色・アクセント・感情）変換システムです。
- 論文: http://arxiv.org/abs/2602.20113
- デモ: https://berkeley-speech-group.github.io/StyleStream/
- ライセンス: 研究目的のみ、商用利用不可

## 現在の状態

このリポジトリには現在**デモWebサイト**（`docs/`ディレクトリ）の音声サンプルと比較結果のみが含まれています。モデルの推論コードと重みは原著者からまだ公開されていません。このリポジトリで再現実装を構築中です。

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

- `docs/` — 静的デモWebサイト（GitHub Pages）+ 論文分析
  - `docs/paper_analysis.md` — 再現実装のための包括的な論文分析
  - `docs/static/audio/` — 音声サンプル（アクセント、感情、ソース、ターゲット、多言語）
  - `docs/static/images/` — システムアーキテクチャ図
  - `docs/index.html` — 音声比較付きデモページ
  - `docs/milestones.md` — 開発マイルストーン統合ドキュメント
  - `docs/phase0_milestones.md` — フェーズ0: プロジェクト基盤構築
  - `docs/phase1_data_milestones.md` — フェーズ1: データ前処理パイプライン
  - `docs/phase2_destylizer_milestones.md` — フェーズ2: Destylizer実装
  - `docs/phase3_stylizer_milestones.md` — フェーズ3: Stylizer/DiT実装
  - `docs/phase4_5_milestones.md` — フェーズ4-5: ボコーダ・ストリーミング
  - `docs/phase6_eval_milestones.md` — フェーズ6: 評価パイプライン
- `README.md` — プロジェクト説明と引用情報
- `LICENSE` — 研究目的のみのライセンス

## 開発環境

- Python依存関係管理: `uv`（新パッケージは `uv add` で追加）
- 現在の依存関係: `pymupdf`（PDF解析）
