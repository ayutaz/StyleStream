"""Conformer encoder for StyleStream Destylizer.

Implements the Conformer architecture (Gulati et al., 2020) with ALiBi
positional encoding for the StyleStream Destylizer pipeline:

    HuBERT L18 features (B, T, 768) -> Conformer x6 -> FSQ -> ASR decoder

Key design choices matching the StyleStream paper:
    - ALiBi positional bias (no learned positional embeddings)
    - Macaron-style feed-forward sandwich (half-step residual)
    - Depthwise separable convolution with causal/non-causal modes
    - 6 Conformer blocks, hidden_size 768, 12 heads, kernel 31

Reference:
    Gulati et al.  "Conformer: Convolution-augmented Transformer for
    Speech Recognition."  Interspeech 2020.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from stylestream.destylizer.alibi import build_alibi_bias


# ======================================================================
# Feed-Forward Module
# ======================================================================


class FeedForwardModule(nn.Module):
    """Position-wise feed-forward with Swish activation.

    ``Linear(hidden, ffn) -> Swish -> Dropout -> Linear(ffn, hidden) -> Dropout``

    In the Conformer macaron structure the output of each FFN sub-layer
    is scaled by 0.5 **outside** this module (handled by
    :class:`ConformerBlock`).

    Parameters
    ----------
    hidden_size : int
        Input and output dimensionality.
    ffn_size : int
        Inner (expanded) dimensionality.
    dropout : float
        Dropout probability applied after each linear projection.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, ffn_size)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, T, hidden_size)``.

        Returns
        -------
        Tensor
            Shape ``(B, T, hidden_size)``.
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


# ======================================================================
# Convolution Module
# ======================================================================


class ConvolutionModule(nn.Module):
    """Conformer convolution module with depthwise separable convolution.

    Architecture::

        LayerNorm -> Pointwise Conv (expand 1x -> 2x) -> GLU gate
        -> Depthwise Conv (kernel_size) -> BatchNorm -> Swish
        -> Pointwise Conv (project 1x -> 1x) -> Dropout

    The depthwise convolution uses ``padding = (kernel_size - 1) // 2``
    for non-causal (bidirectional) mode.  When ``causal=True``, left-only
    padding ``(kernel_size - 1, 0)`` is applied instead, ensuring no
    future information leaks.

    Parameters
    ----------
    hidden_size : int
        Channel dimensionality (input and output).
    kernel_size : int
        Kernel size for the depthwise convolution.
    dropout : float
        Dropout probability after the final pointwise projection.
    causal : bool
        If ``True``, use causal (left-only) padding for streaming.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size

        self.layer_norm = nn.LayerNorm(hidden_size)

        # Pointwise expansion: hidden -> 2*hidden (GLU will halve it back)
        self.pointwise_conv1 = nn.Conv1d(
            hidden_size, 2 * hidden_size, kernel_size=1, bias=True,
        )

        # Depthwise convolution operates on the hidden_size channels.
        # For non-causal mode, use symmetric padding so output length = input.
        # For causal mode, we pad manually with left-only padding.
        if causal:
            self.depthwise_conv = nn.Conv1d(
                hidden_size,
                hidden_size,
                kernel_size=kernel_size,
                groups=hidden_size,
                padding=0,  # manual padding via F.pad
                bias=True,
            )
        else:
            self.depthwise_conv = nn.Conv1d(
                hidden_size,
                hidden_size,
                kernel_size=kernel_size,
                groups=hidden_size,
                padding=(kernel_size - 1) // 2,
                bias=True,
            )

        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.activation = nn.SiLU()

        # Pointwise projection: hidden -> hidden
        self.pointwise_conv2 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=1, bias=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, T, hidden_size)``.
        padding_mask : Tensor or None
            Shape ``(B, T)``.  ``True`` = padded position to ignore.
            Padded positions are zeroed before convolution to prevent
            information leaking through padding tokens.

        Returns
        -------
        Tensor
            Shape ``(B, T, hidden_size)``.
        """
        x = self.layer_norm(x)

        # Zero out padded positions before convolution
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # (B, T, C) -> (B, C, T) for Conv1d
        x = x.transpose(1, 2)

        # Pointwise expand + GLU
        x = self.pointwise_conv1(x)  # (B, 2C, T)
        x = x.chunk(2, dim=1)  # two (B, C, T) tensors
        x = x[0] * torch.sigmoid(x[1])  # GLU gate -> (B, C, T)

        # Depthwise convolution
        if self.causal:
            # Left-only padding: (kernel_size - 1) on left, 0 on right
            x = torch.nn.functional.pad(
                x, (self.kernel_size - 1, 0), mode="constant", value=0.0,
            )
        x = self.depthwise_conv(x)  # (B, C, T)

        # BatchNorm + Swish
        x = self.batch_norm(x)
        x = self.activation(x)

        # Pointwise project
        x = self.pointwise_conv2(x)  # (B, C, T)

        # Dropout
        x = self.dropout(x)

        # (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)

        return x


# ======================================================================
# Multi-Head Self-Attention (with ALiBi)
# ======================================================================


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with ALiBi positional bias.

    Standard scaled dot-product attention augmented with ALiBi
    (Press et al., 2022) positional bias.  No learned positional
    embeddings are used.

    Parameters
    ----------
    hidden_size : int
        Model dimensionality.
    num_heads : int
        Number of attention heads.  ``hidden_size`` must be divisible by
        ``num_heads``, giving ``head_dim = hidden_size // num_heads``.
    dropout : float
        Dropout probability on attention weights.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by "
            f"num_heads ({num_heads})"
        )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # Fused Q/K/V projection
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
        causal: bool = False,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, T, hidden_size)``.
        padding_mask : Tensor or None
            Shape ``(B, T)``.  ``True`` = padded position to mask out.
        causal : bool
            If ``True``, apply causal ALiBi bias (mask future positions).

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

        # Build ALiBi positional bias
        alibi_bias = build_alibi_bias(
            seq_len=T,
            num_heads=H,
            device=x.device,
            dtype=q.dtype,
            causal=causal,
        )  # (1, H, T, T)

        # Combine ALiBi bias with padding mask into a single attention mask.
        # F.scaled_dot_product_attention adds attn_mask to the attention
        # scores before softmax, so ALiBi bias can be passed directly.
        if padding_mask is not None:
            # padding_mask: (B, T) bool, True = ignore -> convert to float
            # with 0.0 for valid positions and -inf for padded positions.
            # Shape: (B, 1, 1, T) to broadcast over (B, H, T, T).
            padding_float = torch.zeros_like(
                padding_mask, dtype=q.dtype,
            ).masked_fill_(padding_mask, float("-inf"))
            attn_mask = alibi_bias + padding_float.unsqueeze(1).unsqueeze(2)
        else:
            attn_mask = alibi_bias

        # Scaled dot-product attention with Flash/memory-efficient kernels
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,
        )  # (B, H, T, D)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.hidden_size)

        return self.out_proj(attn_output)


# ======================================================================
# Conformer Block (Macaron structure)
# ======================================================================


class ConformerBlock(nn.Module):
    """Single Conformer block with macaron-style feed-forward sandwich.

    Architecture::

        x -> LayerNorm -> FFN (x0.5) -> + residual
          -> LayerNorm -> MHSA(ALiBi) -> + residual
          -> LayerNorm -> ConvModule  -> + residual
          -> LayerNorm -> FFN (x0.5) -> + residual
          -> LayerNorm (final)

    Parameters
    ----------
    hidden_size : int
        Model dimensionality.
    ffn_size : int
        Inner dimensionality of the feed-forward modules.
    num_heads : int
        Number of attention heads.
    kernel_size : int
        Kernel size for the depthwise convolution.
    dropout : float
        Dropout probability.
    causal : bool
        If ``True``, use causal convolution and causal ALiBi bias.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        ffn_size: int = 3072,
        num_heads: int = 12,
        kernel_size: int = 31,
        dropout: float = 0.1,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.causal = causal

        # --- First FFN (macaron half-step) ---
        self.ffn1_norm = nn.LayerNorm(hidden_size)
        self.ffn1 = FeedForwardModule(hidden_size, ffn_size, dropout)

        # --- Multi-Head Self-Attention ---
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.self_attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout)

        # --- Convolution Module (has its own internal LayerNorm) ---
        self.conv_norm = nn.LayerNorm(hidden_size)
        self.conv_module = ConvolutionModule(
            hidden_size, kernel_size, dropout, causal,
        )

        # --- Second FFN (macaron half-step) ---
        self.ffn2_norm = nn.LayerNorm(hidden_size)
        self.ffn2 = FeedForwardModule(hidden_size, ffn_size, dropout)

        # --- Final LayerNorm ---
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, T, hidden_size)``.
        padding_mask : Tensor or None
            Shape ``(B, T)``.  ``True`` = padded position.

        Returns
        -------
        Tensor
            Shape ``(B, T, hidden_size)``.
        """
        # First FFN half-step
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))

        # Multi-Head Self-Attention
        x = x + self.self_attn(self.attn_norm(x), padding_mask, causal=self.causal)

        # Convolution Module
        x = x + self.conv_module(self.conv_norm(x), padding_mask)

        # Second FFN half-step
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))

        # Final LayerNorm
        x = self.final_norm(x)

        return x


# ======================================================================
# Conformer Encoder (stacked blocks)
# ======================================================================


class ConformerEncoder(nn.Module):
    """Stacked Conformer encoder for the StyleStream Destylizer.

    Processes pre-extracted HuBERT layer-18 features (or any ``(B, T, 768)``
    input) through ``num_layers`` Conformer blocks with ALiBi positional
    encoding.

    Can be instantiated either with explicit keyword arguments::

        encoder = ConformerEncoder(num_layers=6, hidden_size=768, ...)

    or from an existing :class:`stylestream.config.ConformerConfig`::

        encoder = ConformerEncoder.from_config(config)

    Parameters
    ----------
    num_layers : int
        Number of Conformer blocks.
    hidden_size : int
        Model dimensionality.
    ffn_size : int
        Inner dimensionality of the feed-forward modules.
    num_heads : int
        Number of attention heads.
    kernel_size : int
        Kernel size for depthwise convolutions.
    dropout : float
        Dropout probability.
    causal : bool
        If ``True``, use causal convolution and causal ALiBi bias.
    """

    def __init__(
        self,
        num_layers: int = 6,
        hidden_size: int = 768,
        ffn_size: int = 3072,
        num_heads: int = 12,
        kernel_size: int = 31,
        dropout: float = 0.1,
        causal: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gradient_checkpointing = gradient_checkpointing

        self.layers = nn.ModuleList([
            ConformerBlock(
                hidden_size=hidden_size,
                ffn_size=ffn_size,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dropout=dropout,
                causal=causal,
            )
            for _ in range(num_layers)
        ])

    @classmethod
    def from_config(
        cls,
        config,
        dropout: float = 0.1,
        causal: bool = False,
    ) -> ConformerEncoder:
        """Construct a :class:`ConformerEncoder` from a ``ConformerConfig``.

        Parameters
        ----------
        config : ConformerConfig
            Configuration dataclass with ``num_layers``, ``hidden_size``,
            ``ffn_size``, ``num_heads``, and ``kernel_size`` fields.
        dropout : float
            Dropout probability (not stored in ``ConformerConfig``).
        causal : bool
            Whether to use causal mode.

        Returns
        -------
        ConformerEncoder
        """
        return cls(
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            ffn_size=config.ffn_size,
            num_heads=config.num_heads,
            kernel_size=config.kernel_size,
            dropout=dropout,
            causal=causal,
            gradient_checkpointing=getattr(config, "gradient_checkpointing", False),
        )

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through all Conformer blocks.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, T, hidden_size)``.  Typically pre-extracted HuBERT
            layer-18 features.
        padding_mask : Tensor or None
            Shape ``(B, T)``.  ``True`` = padded position to ignore.

        Returns
        -------
        Tensor
            Shape ``(B, T, hidden_size)``.
        """
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(
                    layer,
                    x, padding_mask,
                    use_reentrant=False,
                )
            else:
                x = layer(x, padding_mask)
        return x
