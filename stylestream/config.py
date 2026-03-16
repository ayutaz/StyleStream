from dataclasses import dataclass, field
from typing import List

from omegaconf import MISSING


# ---------------------------------------------------------------------------
# Audio & feature extraction
# ---------------------------------------------------------------------------

@dataclass
class AudioConfig:
    sample_rate: int = 16000


@dataclass
class MelConfig:
    n_mels: int = 100
    hop_length: int = 320
    n_fft: int = 1024
    f_min: float = 0.0
    f_max: float = 8000.0
    sample_rate: int = 16000

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length  # 50 Hz


# ---------------------------------------------------------------------------
# Destylizer sub-configs
# ---------------------------------------------------------------------------

@dataclass
class HuBERTConfig:
    model_id: str = "facebook/hubert-large-ls960-ft"
    layer: int = 18
    frozen: bool = True  # unfrozen for streaming variant


@dataclass
class ConformerConfig:
    num_layers: int = 6
    hidden_size: int = 768
    ffn_size: int = 3072
    positional_encoding: str = "ALiBi"
    kernel_size: int = 31
    num_heads: int = 12


@dataclass
class FSQConfig:
    levels: List[int] = field(default_factory=lambda: [5, 3, 3])
    down_dim: int = 3  # projects hidden_size -> down_dim before quantization

    @property
    def codebook_size(self) -> int:
        result = 1
        for lv in self.levels:
            result *= lv
        return result  # 45


@dataclass
class ASRDecoderConfig:
    num_layers: int = 4
    hidden_size: int = 768
    ffn_size: int = 3072


@dataclass
class DestylizerConfig:
    hubert: HuBERTConfig = field(default_factory=HuBERTConfig)
    conformer: ConformerConfig = field(default_factory=ConformerConfig)
    fsq: FSQConfig = field(default_factory=FSQConfig)
    asr_decoder: ASRDecoderConfig = field(default_factory=ASRDecoderConfig)
    content_feature_rate: int = 50  # Hz


# ---------------------------------------------------------------------------
# Stylizer sub-configs
# ---------------------------------------------------------------------------

@dataclass
class StyleEncoderConfig:
    model_id: str = "microsoft/wavlm-base-plus-sv"
    hidden_size: int = 768
    num_layers: int = 13
    pooling: str = "attentive_statistics"


@dataclass
class CFMConfig:
    nfe: int = 16  # number of function evaluations (ODE steps)
    sampling: str = "euler"


@dataclass
class CFGConfig:
    strength: float = 2.0
    content_drop: float = 0.2
    context_drop: float = 0.3
    style_drop: float = 0.3


@dataclass
class DiTConfig:
    num_layers: int = 16
    hidden_size: int = 768
    ffn_size: int = 3072


@dataclass
class StylizerConfig:
    dit: DiTConfig = field(default_factory=DiTConfig)
    style_encoder: StyleEncoderConfig = field(default_factory=StyleEncoderConfig)
    cfm: CFMConfig = field(default_factory=CFMConfig)
    cfg: CFGConfig = field(default_factory=CFGConfig)
    mel: MelConfig = field(default_factory=MelConfig)
    mask_ratio_min: float = 0.7  # U[min, max] during training
    mask_ratio_max: float = 1.0
    adaln_zero: bool = True


# ---------------------------------------------------------------------------
# Vocoder sub-configs
# ---------------------------------------------------------------------------

@dataclass
class VocoderBackboneConfig:
    hidden_size: int = 512
    num_layers: int = 8
    intermediate_size: int = 1536
    kernel_size: int = 7
    causal: bool = True


@dataclass
class VocoderDiscriminatorConfig:
    type: str = "multi_scale"
    scales: List[int] = field(default_factory=lambda: [1, 2, 4])
    channels: int = 64


@dataclass
class VocoderLossConfig:
    reconstruction: float = 45.0
    gan_generator: float = 1.0
    gan_discriminator: float = 1.0
    feature_matching: float = 2.0


@dataclass
class VocoderConfig:
    backbone: VocoderBackboneConfig = field(default_factory=VocoderBackboneConfig)
    discriminator: VocoderDiscriminatorConfig = field(
        default_factory=VocoderDiscriminatorConfig,
    )
    loss: VocoderLossConfig = field(default_factory=VocoderLossConfig)
    init_checkpoint: str = "charactr/vocos-mel-24khz"
    causal: bool = True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    steps: int = MISSING
    batch_size: int = MISSING
    num_gpus: int = 8
    optimizer: str = "AdamW"
    peak_lr: float = 1e-4
    warmup_steps: int = 4000
    scheduler: str = "cosine_annealing"
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    log_interval: int = 100
    save_interval: int = 5000
    val_interval: int = 1000


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

@dataclass
class StreamingConfig:
    chunk_size_ms: int = 600
    target_length_s: float = 5.0  # causal attention target window
    ring_buffer_s: float = 5.0  # ring-buffer context for vocoder


# ---------------------------------------------------------------------------
# Data & evaluation
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    name: str = MISSING  # e.g. "libritts", "emilia_en"
    root_dir: str = MISSING
    sample_rate: int = 16000
    segment_length: float = 6.0  # seconds, cropped during training


@dataclass
class EvalConfig:
    metrics: List[str] = field(
        default_factory=lambda: ["wer", "cer", "speaker_sim", "mos"]
    )
    batch_size: int = 16


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    name: str = MISSING
    seed: int = 42
    audio: AudioConfig = field(default_factory=AudioConfig)
    mel: MelConfig = field(default_factory=MelConfig)
    destylizer: DestylizerConfig = field(default_factory=DestylizerConfig)
    stylizer: StylizerConfig = field(default_factory=StylizerConfig)
    vocoder: VocoderConfig = field(default_factory=VocoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
