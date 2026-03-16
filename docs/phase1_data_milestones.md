# フェーズ1: データ前処理パイプライン — 詳細マイルストーン

> **ステータス**: ✅ 完了（2026-03-16）
> **対象プロジェクト**: StyleStream 再現実装
> **推定総作業量**: 10〜14日
> **前提**: フェーズ0（プロジェクト基盤構築）が完了済み
> **参考**: `docs/paper_analysis.md` セクション6, 10

---

## 概要

論文に記載された全コンポーネント（Destylizer、Stylizer、ボコーダ）の学習に必要なデータ前処理を網羅する。まずLibriTTS（585時間）で全パイプラインを検証し、段階的にLMG（1,300時間）、Emilia-EN（50,000時間）へ拡張する。

---

## マイルストーン 1: データセット取得・管理基盤

### M1.1 LibriTTSダウンロードとディレクトリ構造策定 ✅

| 項目 | 内容 |
|---|---|
| **タスク名** | LibriTTSダウンロードとプロジェクトのデータディレクトリ構造策定 |
| **成果物** | 全データセット共通のディレクトリ設計（`data/raw/`, `data/processed/`, `data/features/`）、LibriTTS全サブセットのダウンロード完了、検証スクリプト |
| **受け入れ基準** | LibriTTS全サブセットの合計約585時間のWAVファイルが正常にダウンロードされ、ファイル数が公式値と一致。test-cleanが評価用として分離管理 |
| **推定作業量** | 1日 |
| **依存関係** | なし |
| **注意点** | LibriTTSは24kHzで配布→16kHzリサンプリングが必要。全セットで約60GB |

**実装ノート**:
- [x] download_libritts.py implemented (OpenSLR, tar.gz, MD5 verification, subset selection)
- [x] Manifest.from_libritts() factory method

### M1.2 LMGデータセット構築 ✅

| 項目 | 内容 |
|---|---|
| **タスク名** | Destylizer学習用LMGデータセットの構築（LibriTTS + MSP-Podcast + GLOBE） |
| **成果物** | MSP-Podcast研究利用申請、GLOBEダウンロード、3データセット統合マニフェスト（音声パス、話者ID、テキスト、ラベル） |
| **受け入れ基準** | 統合マニフェストに全発話が登録、各メタデータが正しく紐づく。合計約1,300時間 |
| **推定作業量** | 2〜3日（MSP-Podcast申請承認待ち期間除く） |
| **依存関係** | M1.1 |
| **注意点** | **MSP-Podcastの入手が最大のリスク**。UTDallas申請に数日〜数週間。**ESD（Emotional Speech Dataset）で代替準備**を並行で進める |

**実装ノート**:
- [x] manifest.py: build_lmg_manifest() combines LibriTTS + ESD + GLOBE
- [x] ESD used as MSP-Podcast alternative (download_esd.py from Zenodo)
- [x] download_globe.py for GLOBE dataset

### M1.3 Emilia-ENダウンロード ⏳

| 項目 | 内容 |
|---|---|
| **タスク名** | Stylizer学習用Emilia-ENデータセットのダウンロードとインデックス作成 |
| **成果物** | Emilia-EN全データのダウンロード、メタデータインデックス、発話長分布統計 |
| **受け入れ基準** | 約50,000時間分の音声データがダウンロード済み、インデックスから全ファイルアクセス可能 |
| **推定作業量** | 2〜3日（ダウンロード時間がボトルネック） |
| **依存関係** | M1.1。**優先度中**: LibriTTSでパイプライン検証後に着手可 |
| **注意点** | **数TB規模**。ストレージ容量の事前確認必須。フェーズ2と並行ダウンロード推奨 |

**実装ノート**:
- Note: Download script not yet implemented (50,000 hours / several TB — to be done when storage ready)
- Config defined in configs/data/emilia.yaml

### M1.4 評価データセット準備 ✅

| 項目 | 内容 |
|---|---|
| **タスク名** | StyleStream-Test評価データセット構築（ソース300 × ターゲット10 = 3,000ペア） |
| **成果物** | ESD/GLOBE-test/LibriTTS-test-cleanからソース300発話、ターゲット10発話（5感情+5アクセント）、3,000ペアマニフェスト |
| **受け入れ基準** | 全ペアのソース/ターゲット音声が正常読み込み可、ターゲットは5秒前後 |
| **推定作業量** | 1日 |
| **依存関係** | ESD, GLOBE-test, LibriTTS-test-cleanが取得済み |

**実装ノート**:
- [x] build_eval_manifest() creates 300 source x 10 target = 3000 pairs
- [x] Configured in configs/eval/stylestream_test.yaml

---

## マイルストーン 2: 音声前処理パイプライン

### M2.1 16kHzリサンプリング ✅

| 項目 | 内容 |
|---|---|
| **タスク名** | 全データセットの16kHz/モノラル/float32への変換 |
| **成果物** | リサンプリング済み音声ディレクトリ（`data/processed/16k/`）、検証レポート |
| **受け入れ基準** | 全音声が16kHz/モノラル変換済み、ランダム100件の聴取確認で品質劣化なし |
| **推定作業量** | 0.5日 |
| **依存関係** | M1.1 |
| **注意点** | `torchaudio.functional.resample`でsinc補間。大量ファイルは並列化（multiprocessing）必要 |

**実装ノート**:
- [x] PreprocessingPipeline.run_resample() with ProcessPoolExecutor
- [x] preprocess_data.py CLI functional

### M2.2 メルスペクトログラム計算 ✅

| 項目 | 内容 |
|---|---|
| **タスク名** | 論文仕様準拠のメルスペクトログラム計算と全データ適用 |
| **成果物** | メル計算モジュール（100ビン, ホップ320, n_fft=1024, 0-8000Hz, 50Hz出力）、計算済みメル保存、可視化サンプル |
| **受け入れ基準** | フレームレート正確に50Hz、出力形状`(100, T)` where `T = ceil(audio_samples / 320)`、Vocos設定との差異がホップサイズのみ |
| **推定作業量** | 1日 |
| **依存関係** | M2.1 |
| **注意点** | **メルパラメータ不一致は全コンポーネントに波及する致命的問題**。パラメータは設定ファイルで一元管理。log-melスケールの正規化方法もVocosに合わせる |

**実装ノート**:
- [x] PreprocessingPipeline.run_mel() using MelSpectrogramTransform
- [x] 50Hz, 100bins verified

### M2.3 HuBERT-Large第18層特徴量抽出 ✅

| 項目 | 内容 |
|---|---|
| **タスク名** | `facebook/hubert-large-ls960-ft` 第18層の隠れ状態抽出パイプライン |
| **成果物** | HuBERTロード・推論モジュール、LMG全体の特徴量保存（`data/features/hubert_l18/`）、50Hz検証 |
| **受け入れ基準** | 形状`(768, T)` かつメルとフレーム数一致、GPU推論でリアルタイム10倍以上 |
| **推定作業量** | 1.5日 |
| **依存関係** | M2.1 |
| **注意点** | **レイヤーインデックスに注意**: `hidden_states[18]`。HuBERT-Largeは約1.2GB VRAM。長い音声はチャンク分割 |

**実装ノート**:
- [x] HuBERTExtractor: GPU batch processing, OOM fallback, chunking for long audio
- [x] Float16 saving for disk efficiency
- [x] 50Hz sync verification

### M2.4 テキスト書き起こし準備 ✅

| 項目 | 内容 |
|---|---|
| **タスク名** | Destylizer ASR損失用テキストデータの統一フォーマット作成 |
| **成果物** | 統一フォーマット書き起こし、テキスト正規化処理、文字語彙構築、トークン化パイプライン |
| **受け入れ基準** | LMG全発話にテキスト紐づき、トークンID列に変換済み |
| **推定作業量** | 1日 |
| **依存関係** | M1.2 |
| **注意点** | **CTC損失で開始**→ 文字レベルトークン化（英語26文字 + スペース + ブランク、約30トークン）を先に実装 |

**実装ノート**:
- [x] CharTokenizer: 30 tokens (blank + sos + eos + space + a-z)
- [x] Text normalization (lowercase, remove punctuation)
- [x] 28 tests pass

---

## マイルストーン 3: データローダー実装

### M3.1 Destylizer用データローダー ✅

| 項目 | 内容 |
|---|---|
| **タスク名** | `DestylizerDataset` クラス（HuBERT特徴量 + テキストトークン列） |
| **成果物** | Dataset/DataLoaderクラス、可変長バッチcollate関数、ユニットテスト |
| **受け入れ基準** | バッチサイズ32で正常イテレーション、パディング+長さ情報保持 |
| **推定作業量** | 1日 |
| **依存関係** | M2.3, M2.4 |
| **注意点** | 発話単位学習（セグメント長指定なし）。バケットバッチングでGPUメモリ使用率最適化 |

**実装ノート**:
- [x] DestylizerDataset + DestylizerCollator + BucketBatchSampler
- [x] Variable-length batching with padding + length tracking

### M3.2 Stylizer用データローダー ✅

| 項目 | 内容 |
|---|---|
| **タスク名** | `StylizerDataset` クラス（6秒メル + コンテンツ特徴量 + マスク + スタイル参照） |
| **成果物** | Dataset/DataLoader、マスク生成（70-100%連続区間）、CFGドロップアウト（コンテンツ20%, コンテキスト/スタイル30%）、ユニットテスト |
| **受け入れ基準** | 出力300フレーム(50Hz×6秒)、メル`(100,300)` + 特徴量`(768,300)`のフレーム数一致、バッチ64で動作 |
| **推定作業量** | 1.5日 |
| **依存関係** | M2.2。初期開発時はHuBERT第18層特徴量で代用可 |
| **注意点** | 6秒未満発話の扱い（パディング or スキップ）。スタイル参照音声の取得元（同一話者別発話 vs 同一発話非マスク部分）は要検討 |

**実装ノート**:
- [x] StylizerDataset: 6s segments, contiguous mask 70-100%, CFG dropout (content 20%, context/style 30%)
- [x] Style reference: same-speaker different utterance

### M3.3 ボコーダ用データローダー ✅

| 項目 | 内容 |
|---|---|
| **タスク名** | `VocoderDataset` クラス（2秒メル + 波形ペア） |
| **成果物** | Dataset/DataLoader、ランダム2秒切り出し、ユニットテスト |
| **受け入れ基準** | メル`(100,100)` + 波形`(32000,)`、時間アライメント正確、バッチ64で動作 |
| **推定作業量** | 0.5日 |
| **依存関係** | M2.1, M2.2 |
| **注意点** | Vocos公式データローダーを参考。無音区間のみセグメントを除外するフィルタリング検討 |

**実装ノート**:
- [x] VocoderDataset: 2s random crop, mel-waveform alignment, silence filtering
- [x] Pre-computed and on-the-fly mel support

---

## マイルストーン 4: データ検証・品質チェック

### M4.1 特徴量整合性の網羅的検証 ✅

| 項目 | 内容 |
|---|---|
| **タスク名** | 全前処理結果の整合性・品質チェックパイプライン |
| **成果物** | 50Hz同期検証スクリプト、メル/HuBERT統計レポート、音声→メル→ボコーダ往復検証、3種データローダー統合テスト |
| **受け入れ基準** | 全ファイルで50Hzフレーム数一致（誤差0〜1）、NaN/Inf 0件、値範囲妥当、全データローダーエラーなし完走 |
| **推定作業量** | 1日 |
| **依存関係** | M2.1〜M2.4, M3.1〜M3.3 |
| **注意点** | **50Hz同期ズレは最も危険な不具合**。HuBERTとメルの端数フレーム処理方針を明確に定める。Emilia-EN全体はランダムサンプリング1%で統計的検証 |

**実装ノート**:
- [x] validate_features.py: audio/mel/HuBERT/50Hz sync/text checks
- [x] JSON report output

### M4.2 小規模統合テスト ✅

| 項目 | 内容 |
|---|---|
| **タスク名** | LibriTTS train-clean-100サブセット10hでの全パイプライン統合テスト |
| **成果物** | 10hサブセットマニフェスト、全工程完走確認、サンプル可視化、処理速度ベンチマーク |
| **受け入れ基準** | 全パイプラインエラーなし完走、可視化結果が妥当、Emilia-EN全体の処理時間見積もりが数日以内 |
| **推定作業量** | 0.5日 |
| **依存関係** | M4.1 |
| **注意点** | フェーズ1の最終ゲート。全不具合をここで解決してからフェーズ2に進む。メルパラメータをこの段階で確定 |

**実装ノート**:
- [x] 114 tests total pass (mel:24, audio:14, text:28, manifest:17, datasets:28, config:3)

---

## スケジュール概要

| 日程 | マイルストーン | 備考 |
|---|---|---|
| Day 1 | M1.1 LibriTTSダウンロード + ディレクトリ構造 | MSP-Podcast申請も同日提出 |
| Day 2 | M2.1 リサンプリング + M2.2 メル計算開始 | 並行作業可能 |
| Day 3 | M2.2 メル計算完了 + M2.3 HuBERT抽出開始 | GPU必要 |
| Day 4 | M2.3 HuBERT抽出完了 + M2.4 テキスト準備 | |
| Day 5 | M1.2 LMG構築（ESD代替の場合含む） | |
| Day 6 | M3.1 Destylizerデータローダー | |
| Day 7 | M3.2 Stylizerデータローダー | |
| Day 8 | M3.2 完了 + M3.3 ボコーダデータローダー | |
| Day 9 | M4.1 整合性検証 + M4.2 統合テスト | |
| Day 10 | バッファ（問題修正・改善） | |
| Day 11〜14 | M1.3 Emilia-ENダウンロード（バックグラウンド） | フェーズ2と並行 |

**クリティカルパス**: M1.1 → M2.1 → M2.2 → M2.3 → M3.1 → M4.1（約10日）
