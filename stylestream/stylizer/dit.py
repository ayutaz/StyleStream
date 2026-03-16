"""Diffusion Transformer (DiT) for the StyleStream Stylizer.

Core neural network that predicts velocity fields for Conditional Flow Matching.
The DiT takes noisy mel spectrograms, context mel, content features from the
Destylizer, a flow timestep t, and a style embedding, and predicts the velocity
field v(x_t, t) in mel spectrogram space.

Architecture::

    Inputs:
        x_t             (B, T, 100)   -- noisy mel spectrogram
        context_mel     (B, T, 100)   -- unmasked context mel
        content_features(B, T, 768)   -- Destylizer pre-FSQ output
        t               (B,)          -- flow timestep in [0, 1]
        style_emb       (B, 768)      -- style embedding from WavLM-TDNN

    Pipeline:
        1. Concatenate [x_t, context_mel, content_features]   -> (B, T, 968)
        2. Input projection: Linear(968, 768)                 -> (B, T, 768)
        3. Conditioning: c = timestep_emb(t) + style_emb      -> (B, 768)
        4. RoPE: compute cos/sin once for all layers
        5. 16x DiTBlock(hidden=768, heads=12, ffn=3072)       -> (B, T, 768)
        6. FinalAdaLN + Linear(768, 100)                      -> (B, T, 100)

    DiTBlock (adaLN-Zero):
        c -> SiLU -> Linear -> (gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2)

        x -> (1+gamma_1)*LN(x)+beta_1 -> MultiHeadSelfAttn(RoPE) -> alpha_1*x -> +residual
          -> (1+gamma_2)*LN(x)+beta_2 -> FFN(GELU)               -> alpha_2*x -> +residual

StyleStream Stylizer spec:
    - 16 DiT layers, hidden_size 768, FFN 3072, 12 heads (head_dim 64)
    - RoPE positional encoding (not ALiBi)
    - adaLN-Zero conditioning with zero-initialized gates
    - Input: mel_dim (100) + mel_dim (100) + content_dim (768) = 968
    - Output: mel_dim (100) velocity field
    - 50 Hz frame rate, typical 300 frames (6 seconds)

Reference:
    Peebles & Xie.  "Scalable Diffusion Models with Transformers."
    ICCV 2023.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from stylestream.stylizer.rope import RotaryPositionEmbedding, apply_rotary_pos_emb
from stylestream.stylizer.timestep_embedding import TimestepEmbedding
from stylestream.stylizer.adaln_zero import AdaLNZero, AdaLNModulation, FinalAdaLN

logger = logging.getLogger(__name__)


# ======================================================================
# DiT Block
# ======================================================================


class DiTBlock(nn.Module):
    """Single Diffusion Transformer block with adaLN-Zero conditioning.

    Architecture::

        c (B, 768) -> AdaLNZero -> (gamma_1, beta_1, alpha_1,
                                     gamma_2, beta_2, alpha_2)

        Self-Attention sub-layer:
            residual = x
            x = (1 + gamma_1) * LayerNorm(x) + beta_1
            x = MultiHeadSelfAttention(x, rope)
            x = alpha_1 * x
            x = residual + x

        FFN sub-layer:
            residual = x
            x = (1 + gamma_2) * LayerNorm(x) + beta_2
            x = Linear(768, 3072) -> GELU -> Dropout -> Linear(3072, 768) -> Dropout
            x = alpha_2 * x
            x = residual + x

    Parameters
    ----------
    hidden_size : int
        Hidden dimension.  Default 768.
    num_heads : int
        Number of attention heads.  Default 12.
    ffn_size : int
        FFN intermediate dimension.  Default 3072.
    dropout : float
        Dropout rate.  Default 0.0.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        ffn_size: int = 3072,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by "
            f"num_heads ({num_heads})"
        )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.ffn_size = ffn_size

        # -- adaLN-Zero: generates 6 modulation vectors from c --
        self.adaln = AdaLNZero(hidden_size)

        # -- Self-Attention sub-layer --
        self.attn_modulation = AdaLNModulation(hidden_size)

        # Fused Q/K/V projection
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

        # -- FFN sub-layer --
        self.ffn_modulation = AdaLNModulation(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_size),
            nn.Dropout(dropout),
        )

    def _self_attention(
        self,
        x: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
    ) -> Tensor:
        """Multi-head self-attention with RoPE.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, T, hidden_size)``.  Already modulated by adaLN.
        rope_cos, rope_sin : Tensor
            Shape ``(1, 1, T, head_dim)`` -- precomputed RoPE embeddings.

        Returns
        -------
        Tensor
            Shape ``(B, T, hidden_size)``.
        """
        B, T, _ = x.shape
        H = self.num_heads
        D = self.head_dim

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3 * hidden_size)
        qkv = qkv.reshape(B, T, 3, H, D)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv.unbind(dim=0)  # each (B, H, T, D)

        # Apply RoPE to Q and K only (not V)
        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        # Scaled dot-product attention (PyTorch 2.0+ flash/efficient kernels)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,
        )  # (B, H, T, D)

        # Reshape back to (B, T, hidden_size)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.hidden_size)

        return self.out_proj(attn_output)

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
    ) -> Tensor:
        """Forward pass of one DiT block.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, T, hidden_size)``.
        c : Tensor
            Shape ``(B, hidden_size)`` -- conditioning vector
            (timestep_emb + style_emb).
        rope_cos, rope_sin : Tensor
            Shape ``(1, 1, T, head_dim)`` -- precomputed RoPE embeddings.

        Returns
        -------
        Tensor
            Shape ``(B, T, hidden_size)``.
        """
        # Generate 6 modulation vectors from conditioning
        gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = self.adaln(c)
        # Each: (B, 1, hidden_size)

        # -- Self-Attention sub-layer --
        residual = x
        x = self.attn_modulation(x, gamma_1, beta_1)  # adaLN: (1+gamma)*LN(x)+beta
        x = self._self_attention(x, rope_cos, rope_sin)
        x = alpha_1 * x  # gated (zero-init -> identity at start)
        x = residual + x

        # -- FFN sub-layer --
        residual = x
        x = self.ffn_modulation(x, gamma_2, beta_2)  # adaLN: (1+gamma)*LN(x)+beta
        x = self.ffn(x)
        x = alpha_2 * x  # gated (zero-init -> identity at start)
        x = residual + x

        return x


# ======================================================================
# Full DiT Model
# ======================================================================


class DiT(nn.Module):
    """Full Diffusion Transformer for Stylizer velocity prediction.

    Takes noisy mel, context mel, content features, timestep, and style
    embedding as input.  Outputs predicted velocity field in mel space.

    The input features are concatenated along the feature dimension and
    projected to the hidden dimension.  A shared conditioning vector
    ``c = timestep_emb(t) + style_emb`` is fed to every DiT block via
    adaLN-Zero.  RoPE cos/sin tables are computed once per forward pass
    and shared across all blocks.

    Parameters
    ----------
    num_layers : int
        Number of DiT blocks.  Default 16.
    hidden_size : int
        Hidden dimension.  Default 768.
    ffn_size : int
        FFN intermediate dimension.  Default 3072.
    num_heads : int
        Number of attention heads.  Default 12.
    mel_dim : int
        Mel spectrogram dimension.  Default 100.
    content_dim : int
        Content feature dimension (from Destylizer).  Default 768.
    dropout : float
        Dropout rate.  Default 0.0.
    gradient_checkpointing : bool
        If ``True``, use gradient checkpointing on DiT blocks to save
        memory at the cost of extra computation.  Default ``False``.
    """

    def __init__(
        self,
        num_layers: int = 16,
        hidden_size: int = 768,
        ffn_size: int = 3072,
        num_heads: int = 12,
        mel_dim: int = 100,
        content_dim: int = 768,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.mel_dim = mel_dim
        self.content_dim = content_dim
        self.gradient_checkpointing = gradient_checkpointing

        # -- Input projection --
        # Concatenation: [x_t (mel_dim), context_mel (mel_dim), content (content_dim)]
        input_dim = mel_dim + mel_dim + content_dim  # 100 + 100 + 768 = 968
        self.input_proj = nn.Linear(input_dim, hidden_size)

        # -- Timestep embedding --
        self.timestep_emb = TimestepEmbedding(hidden_size=hidden_size)

        # -- RoPE (shared across all blocks) --
        self.rope = RotaryPositionEmbedding(dim=self.head_dim)

        # -- DiT blocks --
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_size=ffn_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # -- Final layer: adaLN + projection to mel_dim --
        self.final_layer = FinalAdaLN(
            hidden_size=hidden_size,
            output_size=mel_dim,
        )

        # Initialize input projection with Xavier uniform
        self._init_weights()

        logger.info(
            "DiT: %d layers, hidden=%d, heads=%d, ffn=%d, "
            "input=%d->%d, output=%d, params=%.2fM",
            num_layers, hidden_size, num_heads, ffn_size,
            input_dim, hidden_size, mel_dim,
            self.num_parameters() / 1e6,
        )

    def _init_weights(self) -> None:
        """Initialize input projection with Xavier uniform, biases to zero."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        content_features: Tensor,
        context_mel: Tensor,
        style_emb: Tensor,
    ) -> Tensor:
        """Predict velocity field for Conditional Flow Matching.

        Parameters
        ----------
        x_t : Tensor
            Shape ``(B, T, mel_dim)`` -- noisy mel spectrogram at time t.
        t : Tensor
            Shape ``(B,)`` -- flow timestep values in [0, 1].
        content_features : Tensor
            Shape ``(B, T, content_dim)`` -- Destylizer output (pre-FSQ).
        context_mel : Tensor
            Shape ``(B, T, mel_dim)`` -- unmasked context mel spectrogram.
        style_emb : Tensor
            Shape ``(B, hidden_size)`` -- style embedding from WavLM-TDNN.

        Returns
        -------
        Tensor
            Shape ``(B, T, mel_dim)`` -- predicted velocity field v(x_t, t).
        """
        # -- Channel concatenation --
        # (B, T, mel_dim) + (B, T, mel_dim) + (B, T, content_dim) -> (B, T, 968)
        x = torch.cat([x_t, context_mel, content_features], dim=-1)

        # -- Input projection --
        x = self.input_proj(x)  # (B, T, hidden_size)

        # -- Conditioning vector --
        c = self.timestep_emb(t) + style_emb  # (B, hidden_size)

        # -- RoPE: compute cos/sin once, shared by all blocks --
        # RoPE.forward reads shape[-2] as seq_len, so pass (B, H, T, D)
        B, T, _ = x.shape
        rope_cos, rope_sin = self.rope(
            x.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        )  # each (1, 1, T, head_dim)

        # -- DiT blocks --
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(
                    block,
                    x, c, rope_cos, rope_sin,
                    use_reentrant=False,
                )
            else:
                x = block(x, c, rope_cos, rope_sin)

        # -- Final layer: adaLN + projection to mel_dim --
        v = self.final_layer(x, c)  # (B, T, mel_dim)

        return v

    @classmethod
    def from_config(cls, config: Any) -> DiT:
        """Build a DiT from configuration objects.

        Accepts either a ``DiTConfig`` (with additional keyword overrides)
        or a ``StylizerConfig`` which contains a nested ``dit`` field.

        Parameters
        ----------
        config : DiTConfig or StylizerConfig
            Configuration object.  If a ``StylizerConfig`` is provided,
            the ``dit`` sub-config is used for layer/hidden/ffn dimensions
            and ``mel`` sub-config for mel_dim.

        Returns
        -------
        DiT
            Initialized model (on CPU, with random weights).
        """
        # Handle StylizerConfig (has .dit and .mel attributes)
        if hasattr(config, "dit"):
            dit_cfg = config.dit
            mel_dim = getattr(config.mel, "n_mels", 100) if hasattr(config, "mel") else 100
        else:
            # Direct DiTConfig
            dit_cfg = config
            mel_dim = getattr(config, "mel_dim", 100)

        return cls(
            num_layers=dit_cfg.num_layers,
            hidden_size=dit_cfg.hidden_size,
            ffn_size=dit_cfg.ffn_size,
            num_heads=getattr(dit_cfg, "num_heads", 12),
            mel_dim=mel_dim,
            content_dim=getattr(dit_cfg, "content_dim", 768),
            dropout=getattr(dit_cfg, "dropout", 0.0),
            gradient_checkpointing=getattr(dit_cfg, "gradient_checkpointing", False),
        )

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters.

        Parameters
        ----------
        trainable_only : bool
            If ``True`` (default), count only parameters with
            ``requires_grad=True``.

        Returns
        -------
        int
            Total number of (trainable) parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        n_params = self.num_parameters(trainable_only=True) / 1e6
        return (
            f"{self.__class__.__name__}(\n"
            f"  num_layers={self.num_layers},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_heads={self.num_heads},\n"
            f"  ffn_size={self.ffn_size},\n"
            f"  mel_dim={self.mel_dim},\n"
            f"  content_dim={self.content_dim},\n"
            f"  head_dim={self.head_dim},\n"
            f"  gradient_checkpointing={self.gradient_checkpointing},\n"
            f"  trainable_params={n_params:.2f}M\n"
            f")"
        )
