# StyleStream

[![Paper](https://img.shields.io/badge/Paper-ArXiv-b31b1b?style=for-the-badge)](http://arxiv.org/abs/2602.20113)
[![Demo](https://img.shields.io/badge/Demo-Page-4c4c4c?style=for-the-badge)](https://berkeley-speech-group.github.io/StyleStream/)
[![License](https://img.shields.io/badge/License-Research--Only-blue?style=for-the-badge)](./LICENSE)
[![Tests](https://img.shields.io/badge/Tests-568%20passing-brightgreen?style=for-the-badge)](#implementation-status)

A complete PyTorch reimplementation of **StyleStream: Real-Time Zero-Shot Voice Style Conversion** ([arXiv:2602.20113](http://arxiv.org/abs/2602.20113)).

## Overview

StyleStream is a real-time zero-shot voice style conversion system that transforms the timbre, accent, and emotion of speech without any fine-tuning on the target speaker or style. It achieves state-of-the-art conversion quality with an end-to-end streaming latency of approximately 1 second, using a three-stage pipeline: content extraction (Destylizer), style-conditioned synthesis (Stylizer), and waveform generation (Vocoder). All components operate at a unified 50 Hz frame rate on 16 kHz audio.

## Architecture

```
                         Style Reference
                              |
                              v
                       [Style Encoder]
                        (WavLM-TDNN)
                              |
                              v
Source Audio ---> [Destylizer] ---> Content ---> [Stylizer] ---> Mel ---> [Vocoder] ---> Converted Audio
                  HuBERT L18        Features      DiT x16     Spectrogram  Causal        (16 kHz)
                  Conformer x6      (50 Hz)       CFM + CFG   (100 bins)   Vocos
                  FSQ [5,3,3]                     adaLN-Zero               ConvNeXt x8
                                                                           ISTFT
```

**Destylizer** -- Extracts style-invariant content features. HuBERT-Large layer 18 feeds into 6 Conformer blocks with ALiBi positional encoding, quantized through FSQ with codebook size 45. Trained with CTC + seq2seq ASR losses.

**Stylizer** -- Generates mel spectrograms conditioned on content and style. 16-layer Diffusion Transformer with RoPE, Conditional Flow Matching (OT path + Euler sampling), adaLN-Zero conditioning from a WavLM-TDNN style encoder, and Classifier-Free Guidance (alpha=2).

**Vocoder** -- Converts mel spectrograms to waveforms. Causal Vocos architecture with 8 ConvNeXt V2 blocks using causal depthwise separable convolutions, ISTFT head for waveform synthesis, and GAN training with multi-scale discriminator.

## Features

- **Zero-shot conversion** -- No fine-tuning required for new speakers or styles
- **Real-time streaming** -- End-to-end latency of ~1 second using chunked causal attention with 600ms chunks
- **Multi-style transfer** -- Supports timbre, accent, and emotion conversion
- **Streaming-optimized** -- KV caching, StreamingHuBERT, ring buffer pipeline, and MSE distillation for efficient inference
- **Training optimizations** -- Flash Attention (SDPA), torch.compile, Lion optimizer, Grouped Query Attention (GQA), gradient checkpointing, progressive training, Min-SNR loss weighting
- **Fast experimentation** -- Reduced-size configs (`configs/*/fast.yaml`) for 2-3x faster training; `--micro N` for rapid prototyping on small subsets
- **Preprocessing acceleration** -- Merged resample+mel pipeline (80% I/O reduction), GPU-batched HuBERT extraction with duration sorting, FP16 mel/feature storage, pipeline parallelism

## Project Structure

```
stylestream/
  config.py              # Structured configuration dataclasses
  destylizer/            # ALiBi, Conformer x6, FSQ, ASR decoder, trainer
  stylizer/              # RoPE, DiT x16, CFM, adaLN-Zero, style encoder, CFG, trainer
  vocoder/               # Causal ConvNeXt, ISTFT head, discriminator, GAN trainer
  streaming/             # Chunked attention, KV cache, StreamingHuBERT, distillation, pipeline
  data/                  # Manifests, preprocessing, HuBERT extraction, datasets
  eval/                  # Whisper WER, Resemblyzer S-SIM, ECAPA A-SIM, emotion2vec E-SIM, UTMOS
  training/              # Base trainer, scheduler, distributed training
  utils/                 # Mel, audio, logging, checkpointing utilities
configs/                 # YAML configs (destylizer, stylizer, vocoder, streaming, eval, + fast.yaml variants)
scripts/                 # CLI entry points for training, inference, evaluation
tests/                   # 568 tests across all modules
```

## Installation

Requires Python 3.12+. Uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
git clone https://github.com/berkeley-speech-group/StyleStream.git
cd StyleStream

# Core dependencies only
uv sync

# Full installation (training + evaluation + development)
uv sync --extra train --extra eval --extra dev
```

### Dependencies

Core: `torch`, `torchaudio`, `transformers`, `accelerate`, `einops`, `hydra-core`, `omegaconf`
Training: `wandb`, `tensorboard`, `datasets`
Evaluation: `resemblyzer`, `jiwer`, `matplotlib`

## Quick Start

### Data Preprocessing

```bash
# Download datasets
uv run python scripts/download_libritts.py --output-dir data/raw/libritts
uv run python scripts/download_esd.py --output-dir data/raw/esd

# Download pretrained feature extractors
uv run python scripts/download_models.py --stage train

# Preprocess (merged resample + mel computation + HuBERT feature extraction)
uv run python scripts/preprocess_data.py --manifest data/manifests/libritts.csv --output-dir data/processed

# Preprocess with pipeline parallelism (overlap resample and mel stages)
uv run python scripts/preprocess_data.py --manifest data/manifests/libritts.csv --output-dir data/processed --pipelined

# Micro-dataset for rapid prototyping (stratified sample of N utterances)
uv run python scripts/preprocess_data.py --manifest data/manifests/libritts.csv --output-dir data/processed --micro 1000

# Validate extracted features
uv run python scripts/validate_features.py --manifest data/manifests/libritts.csv --processed-dir data/processed
```

### Training

```bash
# Stage 1: Destylizer (Conformer + FSQ with ASR loss)
uv run python scripts/train_destylizer.py --config configs/destylizer/offline.yaml

# Stage 2: Stylizer (DiT + CFM with spectral inpainting)
uv run python scripts/train_stylizer.py --config configs/stylizer/offline.yaml

# Stage 3: Vocoder (Causal Vocos with GAN training)
uv run python scripts/train_vocoder.py --config configs/vocoder/causal_vocos.yaml

# Stage 4: Streaming adaptation (MSE distillation + fine-tuning)
uv run python scripts/train_streaming_destylizer.py --config configs/streaming/distillation.yaml
uv run python scripts/train_streaming_stylizer.py --config configs/streaming/stylizer.yaml
```

### Fast Training

Fast configs (`configs/*/fast.yaml`) use reduced model sizes and optimizations for 2-3x faster experimentation at ~85-90% of full quality. Changes include fewer layers, smaller FFN, Lion optimizer, torch.compile, and progressive training.

```bash
# Fast Destylizer (Conformer x4, FFN 2048, 50k steps)
uv run python scripts/train_destylizer.py --config configs/destylizer/fast.yaml

# Fast Stylizer (DiT x10, FFN 2048, Lion optimizer, progressive training, 200k steps)
uv run python scripts/train_stylizer.py --config configs/stylizer/fast.yaml

# Fast Vocoder (intermediate 1024, 50k steps)
uv run python scripts/train_vocoder.py --config configs/vocoder/fast.yaml
```

### Inference

```bash
# Offline (full-utterance) conversion
uv run python scripts/inference.py \
    --source source.wav --reference target_style.wav -o converted.wav

# Streaming conversion (~1s latency)
uv run python scripts/inference.py \
    --source source.wav --reference target_style.wav --streaming

# Streaming inference demo with ring buffer pipeline
uv run python scripts/streaming_inference.py \
    --source source.wav --target target_style.wav --output converted.wav

# Batch conversion from evaluation pairs
uv run python scripts/inference.py \
    --batch pairs.csv --output-dir converted/
```

### Evaluation

```bash
# Run full evaluation (WER, S-SIM, A-SIM, E-SIM, UTMOS)
uv run python scripts/evaluate.py \
    --converted-dir eval_results/converted --pairs pairs.csv

# Evaluate specific metrics only
uv run python scripts/evaluate.py \
    --converted-dir eval_results/converted --pairs pairs.csv \
    --metrics wer,s_sim

# With paper baselines for comparison
uv run python scripts/evaluate.py \
    --converted-dir eval_results/converted --pairs pairs.csv \
    --config configs/eval/stylestream_test.yaml --output-dir eval_results
```

### Testing

```bash
# Run all 568 tests
uv run pytest tests/ -v

# Run tests for a specific module
uv run pytest tests/test_conformer.py -v
uv run pytest tests/test_cfm.py -v
uv run pytest tests/test_streaming_models.py -v
```

## Optimizations

### Training

| Optimization | Description |
|---|---|
| Flash Attention (SDPA) | `F.scaled_dot_product_attention` in Conformer and DiT for automatic Flash/memory-efficient kernel dispatch |
| `torch.compile` | Optional compilation with `reduce-overhead` mode (enabled in fast configs via `compile_model: true`) |
| Lion optimizer | Momentum-only optimizer with 2x memory efficiency vs AdamW (Chen et al., 2023) |
| Grouped Query Attention | GQA in DiT blocks -- configurable `num_kv_heads` reduces KV memory while preserving quality |
| Gradient checkpointing | Optional per-block checkpointing in DiT to reduce activation memory |
| Progressive training | 3-stage curriculum: gradually increases segment length (3s -> 4.5s -> 6s) and mask ratio |
| Style embedding cache | Pre-computed style embeddings to skip WavLM forward pass during Stylizer training |
| ALiBi/RoPE caching | Positional encoding tensors cached and reused across forward passes |
| Mixed precision (bf16) | All components trained in bf16; mel spectrograms and HuBERT features stored as float16 |
| CUDA optimizations | `cudnn.benchmark`, TF32 matmul, `set_float32_matmul_precision("high")` enabled by default |

### Preprocessing

| Optimization | Description |
|---|---|
| Merged resample+mel | Single-pass pipeline eliminates intermediate WAV I/O (~80% I/O reduction) |
| GPU-batched HuBERT | Duration-sorted batching (`batch_size=16`) minimizes padding waste on GPU |
| FP16 storage | Mel spectrograms and HuBERT features stored as float16 (halves disk usage and I/O) |
| Pipeline parallelism | `--pipelined` flag overlaps resample and mel stages across chunks |
| Micro-dataset | `--micro N` creates stratified subsets for rapid architecture validation |
| Manifest API | `sort_by_duration`, `shard`, `filter_valid`, `stratified_sample` for flexible data management |

### Fast Configs

| Config | Key Changes | Steps |
|---|---|---|
| `configs/destylizer/fast.yaml` | Conformer x4, FFN 2048, ASR decoder x2 | 50k (vs 80k) |
| `configs/stylizer/fast.yaml` | DiT x10, FFN 2048, Lion, torch.compile, progressive | 200k (vs 320k) |
| `configs/vocoder/fast.yaml` | intermediate_size 1024 | 50k (vs 80k) |

## Implementation Status

| Phase | Component | Description | Status |
|-------|-----------|-------------|--------|
| P0 | Infrastructure | Config dataclasses, training base, utilities, checkpoint management | Done |
| P1 | Data Pipeline | Manifests (LibriTTS/ESD/GLOBE), mel preprocessing, HuBERT extraction, datasets | Done |
| P2 | Destylizer | Conformer x6 with ALiBi, FSQ [5,3,3], CTC + seq2seq ASR decoder | Done |
| P3 | Stylizer | 16-layer DiT, CFM (OT path), adaLN-Zero, WavLM-TDNN style encoder, CFG | Done |
| P4 | Vocoder | Causal Vocos (ConvNeXt x8, ISTFT), multi-scale discriminator, GAN training | Done |
| P5 | Streaming | Chunked causal attention, KV cache, StreamingHuBERT, MSE distillation, ring buffer | Done |
| P6 | Evaluation | Whisper WER/CER, Resemblyzer S-SIM, ECAPA A-SIM, emotion2vec E-SIM, UTMOS, visualization | Done |

**568 tests** covering all modules -- passing.

## Paper Target Metrics

Reference baselines from Table 1 of the paper (StyleStream-Test, 3000 pairs):

| Mode | WER (%) | S-SIM | A-SIM | E-SIM |
|------|---------|-------|-------|-------|
| Ground Truth | 3.8 | -- | -- | -- |
| **Offline** | **9.2** | **0.852** | **0.640** | **0.827** |
| Streaming | 10.7 | 0.837 | 0.626 | 0.733 |

- **WER**: Word Error Rate via Whisper-large-v3 (lower is better)
- **S-SIM**: Speaker/timbre similarity via Resemblyzer (higher is better)
- **A-SIM**: Accent similarity via ECAPA-TDNN accent-ID (higher is better)
- **E-SIM**: Emotion similarity via emotion2vec (higher is better)

## Citation

If you find this repository useful, please consider giving a star and citation:

```bibtex
@article{liu2026stylestream,
  title={StyleStream: Real-Time Zero-Shot Voice Style Conversion},
  author={Yisi Liu, Nicholas Lee, Gopala Anumanchipalli},
  journal={arXiv preprint arXiv:2602.20113},
  year={2026}
}
```

## License

This code is released under a **research-only, non-commercial license**.

Commercial use is **not permitted** without explicit permission.
