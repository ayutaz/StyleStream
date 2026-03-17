"""Streaming Stylizer (DiT) for StyleStream streaming inference.

Streaming variant of the Stylizer that replaces full self-attention
with chunked causal attention in all 16 DiT layers. Supports both
training (full sequence with chunked mask) and chunk-by-chunk
inference with KV caching.

StyleStream spec:
    - 16 StreamingDiTBlocks with chunked causal attention
    - RoPE with global position offsets (via KV cache length)
    - adaLN-Zero conditioning unchanged from offline
    - CFM + CFG unchanged from offline
    - Chunk size: 30 frames (600ms @ 50Hz)
    - KV cache: max 250 frames per layer
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from stylestream.streaming.attention_mask import (
    build_chunked_causal_mask,
    chunked_causal_mask_to_attn_bias,
)
from stylestream.streaming.kv_cache import MultiLayerKVCache
from stylestream.stylizer.adaln_zero import AdaLNZero, AdaLNModulation, FinalAdaLN
from stylestream.stylizer.cfm import ConditionalFlowMatching
from stylestream.stylizer.cfg import ClassifierFreeGuidance
from stylestream.stylizer.dit import DiT, DiTBlock
from stylestream.stylizer.rope import RotaryPositionEmbedding, apply_rotary_pos_emb
from stylestream.stylizer.style_encoder import StyleEncoder
from stylestream.stylizer.timestep_embedding import TimestepEmbedding

logger = logging.getLogger(__name__)


# ======================================================================
# Streaming DiT Block
# ======================================================================


class StreamingDiTBlock(nn.Module):
    """DiT block with chunked causal self-attention.

    Same as DiTBlock but uses chunked causal attention for streaming.
    adaLN-Zero conditioning is unchanged.

    During training, a chunked causal mask is applied so that each
    chunk of frames can attend to itself and all past chunks, but
    not to future chunks.  During incremental inference with KV cache,
    the mask is implicit: the current chunk's queries attend to all
    cached keys/values (past chunks) plus the current chunk's own
    keys/values.

    Parameters
    ----------
    hidden_size : int
        Hidden dimension.  Default 768.
    num_heads : int
        Number of attention heads.  Default 12.
    ffn_size : int
        FFN intermediate dimension.  Default 3072.
    dropout : float
        Dropout rate.  Default 0.1.
    chunk_size : int
        Number of frames per chunk (30 = 600ms @ 50Hz).  Default 30.
    num_kv_heads : int
        Number of key/value heads for Grouped Query Attention (GQA).
        When ``0`` (default) or equal to ``num_heads``, standard
        multi-head attention is used.  When ``0 < num_kv_heads < num_heads``,
        K/V projections produce fewer heads and the results are
        repeated to match the number of Q heads.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        ffn_size: int = 3072,
        dropout: float = 0.1,
        chunk_size: int = 30,
        num_kv_heads: int = 0,
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
        self.chunk_size = chunk_size

        # -- GQA configuration --
        self.num_kv_heads = num_kv_heads if num_kv_heads > 0 else num_heads
        self.use_gqa = self.num_kv_heads < self.num_heads

        if self.use_gqa:
            assert self.num_heads % self.num_kv_heads == 0, (
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
            self.kv_repeat_factor = self.num_heads // self.num_kv_heads

        # -- adaLN-Zero: generates 6 modulation vectors from c --
        self.adaln = AdaLNZero(hidden_size)

        # -- Self-Attention sub-layer --
        self.attn_modulation = AdaLNModulation(hidden_size)

        # Q/K/V projections: separate when using GQA, fused otherwise
        if self.use_gqa:
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            self.kv_proj = nn.Linear(
                hidden_size, 2 * self.head_dim * self.num_kv_heads, bias=True
            )
        else:
            # Fused Q/K/V projection (standard MHA)
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
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Multi-head self-attention with RoPE, optional GQA, and chunked causal mask.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, T, hidden_size)``.  Already modulated by adaLN.
        rope_cos, rope_sin : Tensor
            Shape ``(1, 1, T, head_dim)`` -- precomputed RoPE embeddings.
        attn_mask : Tensor or None
            Shape ``(1, 1, T, T)`` additive attention bias where
            blocked positions have ``-inf``.  Used during training.
            ``None`` during incremental inference (no masking needed).

        Returns
        -------
        Tensor
            Shape ``(B, T, hidden_size)``.
        """
        B, T, _ = x.shape
        H = self.num_heads
        D = self.head_dim

        if self.use_gqa:
            # -- GQA path --
            H_kv = self.num_kv_heads

            q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)  # (B, H, T, D)

            kv = self.kv_proj(x).reshape(B, T, 2, H_kv, D)
            kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, H_kv, T, D)
            k, v = kv.unbind(0)

            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

            k = k.repeat_interleave(self.kv_repeat_factor, dim=1)  # (B, H, T, D)
            v = v.repeat_interleave(self.kv_repeat_factor, dim=1)
        else:
            # -- Standard MHA path --
            qkv = self.qkv_proj(x)  # (B, T, 3 * hidden_size)
            qkv = qkv.reshape(B, T, 3, H, D)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
            q, k, v = qkv.unbind(dim=0)  # each (B, H, T, D)

            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        # Scaled dot-product attention with chunked causal mask
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,
        )  # (B, H, T, D)

        # Reshape back to (B, T, hidden_size)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.hidden_size)

        return self.out_proj(attn_output)

    def _self_attention_cached(
        self,
        x: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
        kv_cache: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Multi-head self-attention with KV cache for incremental inference.

        The current chunk's queries attend to all cached keys/values
        plus the current chunk.  RoPE embeddings must be pre-offset to
        the correct global positions.

        When GQA is enabled, the KV cache stores the *expanded* K/V
        (with heads already repeated to match Q), so that cache
        concatenation works without extra bookkeeping.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, T_chunk, hidden_size)``.
        rope_cos, rope_sin : Tensor
            Shape ``(1, 1, T_chunk, head_dim)`` -- RoPE for the current
            chunk's positions (already offset by cache length).
        kv_cache : tuple[Tensor, Tensor] or None
            Past ``(K, V)`` each ``(B, H, T_past, D)`` where H is
            ``num_heads`` (already expanded for GQA).

        Returns
        -------
        output : Tensor
            Shape ``(B, T_chunk, hidden_size)``.
        new_kv : tuple[Tensor, Tensor]
            Updated ``(K, V)`` including the current chunk.
        """
        B, T, _ = x.shape
        H = self.num_heads
        D = self.head_dim

        if self.use_gqa:
            # -- GQA path --
            H_kv = self.num_kv_heads

            q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)  # (B, H, T, D)

            kv = self.kv_proj(x).reshape(B, T, 2, H_kv, D)
            kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, H_kv, T, D)
            k, v = kv.unbind(0)

            # Apply RoPE before expansion
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

            # Expand KV heads to match Q heads before caching
            k = k.repeat_interleave(self.kv_repeat_factor, dim=1)  # (B, H, T, D)
            v = v.repeat_interleave(self.kv_repeat_factor, dim=1)
        else:
            # -- Standard MHA path --
            qkv = self.qkv_proj(x)  # (B, T, 3 * hidden_size)
            qkv = qkv.reshape(B, T, 3, H, D)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
            q, k, v = qkv.unbind(dim=0)  # each (B, H, T, D)

            # Apply RoPE with global position offsets
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        # Concatenate with cached K, V (both stored with full num_heads)
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k_full = torch.cat([past_k, k], dim=2)  # (B, H, T_past + T, D)
            v_full = torch.cat([past_v, v], dim=2)
        else:
            k_full = k
            v_full = v

        new_kv = (k_full, v_full)

        # Attention: current queries attend to all past + current K/V.
        # No explicit mask needed -- causality is enforced by the cache
        # structure (only past + current frames are present).
        attn_output = F.scaled_dot_product_attention(
            q, k_full, v_full,
            attn_mask=None,
            dropout_p=0.0,  # no dropout during inference
            is_causal=False,
        )  # (B, H, T, D)

        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.hidden_size)
        output = self.out_proj(attn_output)

        return output, new_kv

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Training forward pass with chunked causal mask.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, T, hidden_size)``.
        c : Tensor
            Shape ``(B, hidden_size)`` -- conditioning vector.
        rope_cos, rope_sin : Tensor
            Shape ``(1, 1, T, head_dim)``.
        attn_mask : Tensor or None
            Shape ``(1, 1, T, T)`` additive bias.

        Returns
        -------
        Tensor
            Shape ``(B, T, hidden_size)``.
        """
        # Generate 6 modulation vectors from conditioning
        gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = self.adaln(c)

        # -- Self-Attention sub-layer --
        residual = x
        x = self.attn_modulation(x, gamma_1, beta_1)
        x = self._self_attention(x, rope_cos, rope_sin, attn_mask=attn_mask)
        x = alpha_1 * x
        x = residual + x

        # -- FFN sub-layer --
        residual = x
        x = self.ffn_modulation(x, gamma_2, beta_2)
        x = self.ffn(x)
        x = alpha_2 * x
        x = residual + x

        return x

    def forward_cached(
        self,
        x: Tensor,
        c: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
        kv_cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Incremental inference forward with KV cache.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, T_chunk, hidden_size)``.
        c : Tensor
            Shape ``(B, hidden_size)`` -- conditioning vector.
        rope_cos, rope_sin : Tensor
            Shape ``(1, 1, T_chunk, head_dim)`` -- offset RoPE.
        kv_cache : tuple[Tensor, Tensor] or None
            Past ``(K, V)`` from this layer.

        Returns
        -------
        output : Tensor
            Shape ``(B, T_chunk, hidden_size)``.
        new_kv : tuple[Tensor, Tensor]
            Updated cache.
        """
        gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = self.adaln(c)

        # -- Self-Attention sub-layer with KV cache --
        residual = x
        x = self.attn_modulation(x, gamma_1, beta_1)
        x, new_kv = self._self_attention_cached(x, rope_cos, rope_sin, kv_cache)
        x = alpha_1 * x
        x = residual + x

        # -- FFN sub-layer --
        residual = x
        x = self.ffn_modulation(x, gamma_2, beta_2)
        x = self.ffn(x)
        x = alpha_2 * x
        x = residual + x

        return x, new_kv


# ======================================================================
# Streaming DiT Model
# ======================================================================


class StreamingDiT(nn.Module):
    """Streaming Diffusion Transformer with chunked causal attention.

    Same architecture as the offline DiT but with chunked causal
    attention for streaming training and inference.

    The input pipeline is identical to the offline DiT:

    1. Concatenate ``[x_t, context_mel, content_features]`` -> ``(B, T, 968)``
    2. Input projection ``Linear(968, 768)`` -> ``(B, T, 768)``
    3. Conditioning ``c = timestep_emb(t) + style_emb`` -> ``(B, 768)``
    4. RoPE cos/sin tables (with global position offsets for caching)
    5. 16x ``StreamingDiTBlock`` with chunked causal attention
    6. ``FinalAdaLN`` + ``Linear(768, 100)`` -> velocity ``(B, T, 100)``

    For training, the full sequence is processed with a chunked causal
    attention mask.  For inference, chunks are processed incrementally
    with a KV cache.

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
        Content feature dimension.  Default 768.
    dropout : float
        Dropout rate.  Default 0.0.
    num_kv_heads : int
        Number of key/value heads for Grouped Query Attention (GQA).
        When ``0`` (default) or equal to ``num_heads``, standard MHA
        is used.
    chunk_size : int
        Chunk size in frames (30 = 600ms @ 50Hz).  Default 30.
    max_cache_frames : int
        Maximum frames to cache per layer.  Default 250 (5s @ 50Hz).
    gradient_checkpointing : bool
        Use gradient checkpointing on DiT blocks.  Default False.
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
        num_kv_heads: int = 0,
        chunk_size: int = 30,
        max_cache_frames: int = 250,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.mel_dim = mel_dim
        self.content_dim = content_dim
        self.chunk_size = chunk_size
        self.max_cache_frames = max_cache_frames
        self.gradient_checkpointing = gradient_checkpointing

        # -- Input projection --
        input_dim = mel_dim + mel_dim + content_dim  # 100 + 100 + 768 = 968
        self.input_proj = nn.Linear(input_dim, hidden_size)

        # -- Timestep embedding --
        self.timestep_emb = TimestepEmbedding(hidden_size=hidden_size)

        # -- RoPE (shared across all blocks) --
        self.rope = RotaryPositionEmbedding(dim=self.head_dim)

        # -- Streaming DiT blocks --
        self.blocks = nn.ModuleList([
            StreamingDiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_size=ffn_size,
                dropout=dropout,
                chunk_size=chunk_size,
                num_kv_heads=num_kv_heads,
            )
            for _ in range(num_layers)
        ])

        # -- Final layer: adaLN + projection to mel_dim --
        self.final_layer = FinalAdaLN(
            hidden_size=hidden_size,
            output_size=mel_dim,
        )

        # Initialize input projection
        self._init_weights()

        logger.info(
            "StreamingDiT: %d layers, hidden=%d, heads=%d, ffn=%d, "
            "chunk_size=%d, max_cache=%d, input=%d->%d, output=%d, "
            "params=%.2fM",
            num_layers, hidden_size, num_heads, ffn_size,
            chunk_size, max_cache_frames,
            input_dim, hidden_size, mel_dim,
            self.num_parameters() / 1e6,
        )

    def _init_weights(self) -> None:
        """Initialize input projection with Xavier uniform, biases to zero."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    # ------------------------------------------------------------------
    # Training forward (full sequence)
    # ------------------------------------------------------------------

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        content_features: Tensor,
        context_mel: Tensor,
        style_emb: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Training forward pass with chunked causal attention.

        Processes the full sequence with a chunked causal mask so that
        each chunk attends only to itself and past chunks.

        Parameters
        ----------
        x_t : Tensor
            Shape ``(B, T, mel_dim)`` -- noisy mel spectrogram at time t.
        t : Tensor
            Shape ``(B,)`` -- flow timestep values in [0, 1].
        content_features : Tensor
            Shape ``(B, T, content_dim)`` -- Destylizer output.
        context_mel : Tensor
            Shape ``(B, T, mel_dim)`` -- unmasked context mel spectrogram.
        style_emb : Tensor
            Shape ``(B, hidden_size)`` -- style embedding from WavLM-TDNN.
        mask : Tensor or None
            Shape ``(B, T)`` -- inpainting mask (unused here, accepted
            for API compatibility with the offline DiT).

        Returns
        -------
        Tensor
            Shape ``(B, T, mel_dim)`` -- predicted velocity field.
        """
        # -- Channel concatenation --
        x = torch.cat([x_t, context_mel, content_features], dim=-1)

        # -- Input projection --
        x = self.input_proj(x)  # (B, T, hidden_size)

        # -- Conditioning vector --
        c = self.timestep_emb(t) + style_emb  # (B, hidden_size)

        # -- RoPE: compute cos/sin once, shared by all blocks --
        B, T, _ = x.shape
        rope_cos, rope_sin = self.rope(
            x.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        )  # each (1, 1, T, head_dim)

        # -- Build chunked causal attention mask --
        attn_mask: Tensor | None = None
        if self.chunk_size < T:
            bool_mask = build_chunked_causal_mask(
                seq_len=T,
                chunk_size=self.chunk_size,
                device=x.device,
            )
            attn_mask = chunked_causal_mask_to_attn_bias(bool_mask)
            attn_mask = attn_mask.to(dtype=x.dtype)

        # -- Streaming DiT blocks --
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(
                    block,
                    x, c, rope_cos, rope_sin, attn_mask,
                    use_reentrant=False,
                )
            else:
                x = block(x, c, rope_cos, rope_sin, attn_mask=attn_mask)

        # -- Final layer --
        v = self.final_layer(x, c)  # (B, T, mel_dim)

        return v

    # ------------------------------------------------------------------
    # Incremental inference (chunk-by-chunk)
    # ------------------------------------------------------------------

    def forward_chunk(
        self,
        x_t_chunk: Tensor,
        t: Tensor,
        content_features_chunk: Tensor,
        context_mel_chunk: Tensor,
        style_emb: Tensor,
        kv_cache: MultiLayerKVCache | None = None,
    ) -> tuple[Tensor, MultiLayerKVCache]:
        """Incremental inference for a single chunk.

        Processes one chunk of frames, using a KV cache to store past
        keys and values.  RoPE positions are offset by the cache length
        to maintain global position awareness.

        Parameters
        ----------
        x_t_chunk : Tensor
            Shape ``(B, T_chunk, mel_dim)``.
        t : Tensor
            Shape ``(B,)`` -- flow timestep.
        content_features_chunk : Tensor
            Shape ``(B, T_chunk, content_dim)``.
        context_mel_chunk : Tensor
            Shape ``(B, T_chunk, mel_dim)``.
        style_emb : Tensor
            Shape ``(B, hidden_size)``.
        kv_cache : MultiLayerKVCache or None
            Cache from previous chunks.  If None, a new cache is created.

        Returns
        -------
        velocity_chunk : Tensor
            Shape ``(B, T_chunk, mel_dim)``.
        kv_cache : MultiLayerKVCache
            Updated cache including the current chunk.
        """
        if kv_cache is None:
            kv_cache = MultiLayerKVCache(
                num_layers=self.num_layers,
                max_frames=self.max_cache_frames,
            )

        # -- Channel concatenation --
        x = torch.cat(
            [x_t_chunk, context_mel_chunk, content_features_chunk], dim=-1
        )

        # -- Input projection --
        x = self.input_proj(x)  # (B, T_chunk, hidden_size)

        # -- Conditioning vector --
        c = self.timestep_emb(t) + style_emb  # (B, hidden_size)

        # -- RoPE with global position offset from cache --
        B, T_chunk, _ = x.shape
        cache_len = kv_cache.length
        rope_cos, rope_sin = self.rope(
            x.view(B, T_chunk, self.num_heads, self.head_dim).permute(0, 2, 1, 3),
            offset=cache_len,
        )  # each (1, 1, T_chunk, head_dim)

        # -- Process through all blocks with KV caching --
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i]
            past_kv = layer_cache.get()
            # Convert (None, None) to None for the block
            past_kv_tuple = (
                (past_kv[0], past_kv[1])
                if past_kv[0] is not None
                else None
            )

            x, new_kv = block.forward_cached(
                x, c, rope_cos, rope_sin, kv_cache=past_kv_tuple
            )

            # Update the cache for this layer (append handles trim)
            # We need to store only the new K, V (the block already
            # concatenated past + new).  Since LayerKVCache.append
            # concatenates, we store the full result directly.
            layer_cache._k = new_kv[0]
            layer_cache._v = new_kv[1]
            layer_cache.trim()

        # -- Final layer --
        v = self.final_layer(x, c)  # (B, T_chunk, mel_dim)

        return v, kv_cache

    # ------------------------------------------------------------------
    # Weight loading from offline DiT
    # ------------------------------------------------------------------

    @classmethod
    def from_offline(
        cls,
        source: DiT | nn.Module,
        chunk_size: int = 30,
        max_cache_frames: int = 250,
    ) -> StreamingDiT:
        """Create a StreamingDiT by loading weights from an offline DiT.

        Since the architecture is identical except for attention masking,
        all weights transfer directly.  The offline DiT's ``DiTBlock``
        layers map to ``StreamingDiTBlock`` layers with identical
        parameter names.

        Parameters
        ----------
        source : DiT or nn.Module
            An offline ``DiT`` instance, or a ``Stylizer`` (in which
            case ``source.dit`` is used).
        chunk_size : int
            Chunk size for streaming.  Default 30.
        max_cache_frames : int
            Maximum cache frames per layer.  Default 250.

        Returns
        -------
        StreamingDiT
            Streaming model with weights loaded from the offline model.
        """
        # If given a Stylizer, extract the DiT.
        if hasattr(source, "dit"):
            source = source.dit

        streaming = cls(
            num_layers=source.num_layers,
            hidden_size=source.hidden_size,
            ffn_size=source.ffn_size,
            num_heads=source.num_heads,
            mel_dim=source.mel_dim,
            content_dim=source.content_dim,
            dropout=0.0,
            num_kv_heads=getattr(source, "num_kv_heads", 0),
            chunk_size=chunk_size,
            max_cache_frames=max_cache_frames,
            gradient_checkpointing=False,
        )

        # Build weight mapping: offline DiTBlock -> StreamingDiTBlock
        # Both have identical parameter names within each block.
        source_sd = source.state_dict()
        target_sd = streaming.state_dict()

        # Verify key compatibility
        missing = set(target_sd.keys()) - set(source_sd.keys())
        unexpected = set(source_sd.keys()) - set(target_sd.keys())

        if missing:
            logger.warning(
                "Keys in StreamingDiT but not in source: %s", missing
            )
        if unexpected:
            logger.warning(
                "Keys in source but not in StreamingDiT: %s", unexpected
            )

        # Load matching weights (strict=False to allow new parameters)
        streaming.load_state_dict(source_sd, strict=False)

        logger.info(
            "Loaded StreamingDiT weights from offline DiT "
            "(%d matching keys, %d missing, %d unexpected)",
            len(set(target_sd.keys()) & set(source_sd.keys())),
            len(missing),
            len(unexpected),
        )

        return streaming

    # ------------------------------------------------------------------
    # Config constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> StreamingDiT:
        """Build a StreamingDiT from configuration objects.

        Parameters
        ----------
        config : StylizerConfig or ExperimentConfig
            Configuration object.

        Returns
        -------
        StreamingDiT
        """
        # Handle ExperimentConfig (has .stylizer and .streaming)
        if hasattr(config, "stylizer"):
            dit_cfg = config.stylizer.dit
            mel_dim = (
                getattr(config.mel, "n_mels", 100)
                if hasattr(config, "mel")
                else 100
            )
            streaming_cfg = getattr(config, "streaming", None)
            if streaming_cfg is not None:
                chunk_ms = getattr(streaming_cfg, "chunk_size_ms", 600)
                # Convert ms to frames at 50Hz: ms / 20
                chunk_size = chunk_ms // 20
            else:
                chunk_size = 30
        elif hasattr(config, "dit"):
            dit_cfg = config.dit
            mel_dim = (
                getattr(config.mel, "n_mels", 100)
                if hasattr(config, "mel")
                else 100
            )
            chunk_size = 30
        else:
            dit_cfg = config
            mel_dim = getattr(config, "mel_dim", 100)
            chunk_size = getattr(config, "chunk_size", 30)

        return cls(
            num_layers=dit_cfg.num_layers,
            hidden_size=dit_cfg.hidden_size,
            ffn_size=dit_cfg.ffn_size,
            num_heads=getattr(dit_cfg, "num_heads", 12),
            mel_dim=mel_dim,
            content_dim=getattr(dit_cfg, "content_dim", 768),
            dropout=getattr(dit_cfg, "dropout", 0.0),
            num_kv_heads=getattr(dit_cfg, "num_kv_heads", 0),
            chunk_size=chunk_size,
            max_cache_frames=getattr(config, "max_cache_frames", 250),
            gradient_checkpointing=getattr(
                dit_cfg, "gradient_checkpointing", False
            ),
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
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
            f"  chunk_size={self.chunk_size},\n"
            f"  max_cache_frames={self.max_cache_frames},\n"
            f"  head_dim={self.head_dim},\n"
            f"  gradient_checkpointing={self.gradient_checkpointing},\n"
            f"  trainable_params={n_params:.2f}M\n"
            f")"
        )


# ======================================================================
# Streaming Stylizer
# ======================================================================


_DEFAULT_MEL_DIM: int = 100
_DEFAULT_HIDDEN_SIZE: int = 768
_DEFAULT_CONTENT_DIM: int = 768


class StreamingStylizer(nn.Module):
    """Streaming Stylizer: StreamingDiT + StyleEncoder + CFM + CFG.

    Streaming variant of the Stylizer.  Uses chunked causal attention
    in the DiT for training and chunk-by-chunk inference.  All other
    components (StyleEncoder, CFM, CFG) are identical to the offline
    Stylizer.

    At **training** time, :meth:`forward` computes the CFM loss on the
    masked region using the full sequence with a chunked causal mask.

    At **inference** time, :meth:`sample_chunk` generates mel for one
    chunk using CFM Euler sampling with KV caching.

    Parameters
    ----------
    config : StylizerConfig, optional
        Structured configuration.
    **kwargs
        Override individual settings when config is not provided.
        Accepted keys include all DiT, style encoder, CFM, and CFG
        parameters, plus ``chunk_size`` and ``max_cache_frames``.
    """

    def __init__(self, config=None, **kwargs: Any) -> None:
        super().__init__()

        if config is not None:
            self._init_from_config(config)
        else:
            self._init_from_kwargs(kwargs)

        logger.info(
            "StreamingStylizer: dit=%d layers, chunk_size=%d, "
            "style_encoder=%s, cfm sigma_min=%.1e, "
            "cfg alpha=%.1f, mel_dim=%d",
            self.dit.num_layers,
            self.dit.chunk_size,
            self.style_encoder.wavlm_model_id,
            self.cfm.sigma_min,
            self.cfg.guidance_strength,
            self.mel_dim,
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_from_config(self, config) -> None:
        """Build sub-modules from configuration."""
        dit_cfg = config.dit
        se_cfg = config.style_encoder
        cfm_cfg = config.cfm
        cfg_cfg = config.cfg
        mel_cfg = config.mel

        self.mel_dim = mel_cfg.n_mels
        self.nfe = cfm_cfg.nfe

        # Determine chunk_size from streaming config if available
        streaming_cfg = getattr(config, "streaming", None)
        if streaming_cfg is not None:
            chunk_ms = getattr(streaming_cfg, "chunk_size_ms", 600)
            chunk_size = chunk_ms // 20  # convert ms to frames at 50Hz
        else:
            chunk_size = 30

        max_cache_frames = 250
        if streaming_cfg is not None:
            target_s = getattr(streaming_cfg, "target_length_s", 5.0)
            max_cache_frames = int(target_s * 50)  # 50Hz frame rate

        # --- StreamingDiT ---
        self.dit = StreamingDiT(
            num_layers=dit_cfg.num_layers,
            hidden_size=dit_cfg.hidden_size,
            ffn_size=dit_cfg.ffn_size,
            num_heads=getattr(dit_cfg, "num_heads", 12),
            mel_dim=self.mel_dim,
            content_dim=getattr(dit_cfg, "content_dim", _DEFAULT_CONTENT_DIM),
            dropout=getattr(dit_cfg, "dropout", 0.0),
            num_kv_heads=getattr(dit_cfg, "num_kv_heads", 0),
            chunk_size=chunk_size,
            max_cache_frames=max_cache_frames,
            gradient_checkpointing=getattr(
                dit_cfg, "gradient_checkpointing", False
            ),
        )

        # --- Style encoder ---
        self.style_encoder = StyleEncoder(
            wavlm_model_id=se_cfg.model_id,
            hidden_size=se_cfg.hidden_size,
            num_wavlm_layers=se_cfg.num_layers,
            tdnn_channels=getattr(se_cfg, "tdnn_channels", 512),
            output_size=getattr(se_cfg, "output_size", _DEFAULT_HIDDEN_SIZE),
            freeze_wavlm=getattr(se_cfg, "freeze_wavlm", True),
        )

        # --- CFM ---
        self.cfm = ConditionalFlowMatching(
            sigma_min=getattr(cfm_cfg, "sigma_min", 1e-5),
        )

        # --- CFG ---
        self.cfg = ClassifierFreeGuidance(
            content_drop_prob=cfg_cfg.content_drop,
            context_drop_prob=cfg_cfg.context_drop,
            style_drop_prob=cfg_cfg.style_drop,
            guidance_strength=cfg_cfg.strength,
        )

    def _init_from_kwargs(self, kw: dict[str, Any]) -> None:
        """Build sub-modules from flat keyword arguments."""
        hidden = kw.get("hidden_size", _DEFAULT_HIDDEN_SIZE)
        mel_dim = kw.get("mel_dim", _DEFAULT_MEL_DIM)

        self.mel_dim = mel_dim
        self.nfe = kw.get("nfe", 16)

        # --- StreamingDiT ---
        self.dit = StreamingDiT(
            num_layers=kw.get("num_layers", 16),
            hidden_size=hidden,
            ffn_size=kw.get("ffn_size", 3072),
            num_heads=kw.get("num_heads", 12),
            mel_dim=mel_dim,
            content_dim=kw.get("content_dim", _DEFAULT_CONTENT_DIM),
            dropout=kw.get("dropout", 0.0),
            chunk_size=kw.get("chunk_size", 30),
            max_cache_frames=kw.get("max_cache_frames", 250),
            gradient_checkpointing=kw.get("gradient_checkpointing", False),
        )

        # --- Style encoder ---
        self.style_encoder = StyleEncoder(
            wavlm_model_id=kw.get(
                "wavlm_model_id", "microsoft/wavlm-base-plus-sv"
            ),
            hidden_size=kw.get("style_hidden_size", hidden),
            num_wavlm_layers=kw.get("num_wavlm_layers", 13),
            tdnn_channels=kw.get("tdnn_channels", 512),
            output_size=kw.get("output_size", hidden),
            freeze_wavlm=kw.get("freeze_wavlm", True),
        )

        # --- CFM ---
        self.cfm = ConditionalFlowMatching(
            sigma_min=kw.get("sigma_min", 1e-5),
        )

        # --- CFG ---
        self.cfg = ClassifierFreeGuidance(
            content_drop_prob=kw.get("content_drop_prob", 0.2),
            context_drop_prob=kw.get("context_drop_prob", 0.3),
            style_drop_prob=kw.get("style_drop_prob", 0.3),
            guidance_strength=kw.get("guidance_strength", 2.0),
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config) -> StreamingStylizer:
        """Build a StreamingStylizer from configuration.

        Parameters
        ----------
        config : StylizerConfig or ExperimentConfig
            Configuration object.

        Returns
        -------
        StreamingStylizer
        """
        return cls(config=config)

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        mel: Tensor,
        content_features: Tensor,
        mask: Tensor,
        style_waveform: Tensor,
        cfg_drop_content: Tensor | None = None,
        cfg_drop_context: Tensor | None = None,
        cfg_drop_style: Tensor | None = None,
    ) -> dict[str, Any]:
        """Training forward pass (same interface as offline Stylizer).

        Computes the masked CFM loss for spectrogram inpainting using
        chunked causal attention in the DiT.

        Parameters
        ----------
        mel : Tensor
            ``(B, T, 100)`` ground-truth mel spectrogram.
        content_features : Tensor
            ``(B, T, 768)`` content features from the Destylizer.
        mask : Tensor
            ``(B, T)`` binary mask where 1 = masked (to generate).
        style_waveform : Tensor
            ``(B, num_samples)`` raw 16kHz audio for style extraction.
        cfg_drop_content : Tensor or None
            ``(B,)`` bool -- pre-sampled content dropout decisions.
        cfg_drop_context : Tensor or None
            ``(B,)`` bool -- pre-sampled context dropout decisions.
        cfg_drop_style : Tensor or None
            ``(B,)`` bool -- pre-sampled style dropout decisions.

        Returns
        -------
        dict
            ``'loss'``: scalar CFM loss.
            ``'velocity_pred'``: ``(B, T, 100)`` predicted velocity.
        """
        B = mel.shape[0]
        device = mel.device
        dtype = mel.dtype

        # 1. Context mel: zero out masked positions.
        context_mel = mel * (1.0 - mask.unsqueeze(-1))

        # 2. Extract style embedding.
        style_emb = self.style_encoder(style_waveform)

        # 3. Apply CFG training dropout.
        content_dropped, context_dropped, style_dropped = (
            self.cfg.apply_training_dropout(
                content_features,
                context_mel,
                style_emb,
                cfg_drop_content=cfg_drop_content,
                cfg_drop_context=cfg_drop_context,
                cfg_drop_style=cfg_drop_style,
            )
        )

        # 4. Sample CFM noise and timestep.
        x_0 = self.cfm.sample_noise(mel.shape, device=device, dtype=dtype)
        t = self.cfm.sample_timestep(B, device=device)

        # 5. OT interpolation.
        x_t = self.cfm.interpolate(x_0, mel, t)

        # 6. Predict velocity with StreamingDiT.
        velocity_pred = self.dit(
            x_t, t, content_dropped, context_dropped, style_dropped
        )

        # 7. Compute masked CFM loss (with Min-SNR weighting when enabled).
        loss = self.cfm.compute_loss(velocity_pred, mel, x_0, mask, t=t)

        return {
            "loss": loss,
            "velocity_pred": velocity_pred,
        }

    # ------------------------------------------------------------------
    # Chunk-by-chunk inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_chunk(
        self,
        content_chunk: Tensor,
        context_mel_chunk: Tensor,
        style_emb: Tensor,
        mask_chunk: Tensor | None = None,
        kv_cache: MultiLayerKVCache | None = None,
        nfe: int | None = None,
        guidance_strength: float | None = None,
    ) -> tuple[Tensor, MultiLayerKVCache]:
        """Generate mel for one chunk using CFM Euler sampling with KV cache.

        This method performs the full Euler ODE integration for a single
        chunk of frames.  The KV cache accumulates across chunks (not
        across Euler steps within a chunk -- each Euler step reuses the
        same cache from prior chunks).

        For each Euler step, the DiT processes only the current chunk
        with the KV cache providing context from prior chunks.  After
        all Euler steps, the cache is updated with the final hidden
        representations of the current chunk.

        Parameters
        ----------
        content_chunk : Tensor
            ``(B, T_chunk, 768)`` content features for this chunk.
        context_mel_chunk : Tensor
            ``(B, T_chunk, 100)`` context mel for this chunk.
        style_emb : Tensor
            ``(B, hidden_size)`` style embedding (pre-computed, shared
            across all chunks).
        mask_chunk : Tensor or None
            ``(B, T_chunk)`` inpainting mask for this chunk.  If None,
            all frames are generated.
        kv_cache : MultiLayerKVCache or None
            Cache from prior chunks.  None for the first chunk.
        nfe : int or None
            Number of Euler steps.  Default uses ``self.nfe``.
        guidance_strength : float or None
            CFG strength override.

        Returns
        -------
        mel_chunk : Tensor
            ``(B, T_chunk, 100)`` generated mel for this chunk.
        kv_cache : MultiLayerKVCache
            Updated cache including the current chunk.
        """
        B, T_chunk, _ = content_chunk.shape
        device = content_chunk.device
        dtype = content_chunk.dtype

        nfe = nfe if nfe is not None else self.nfe
        alpha = (
            guidance_strength
            if guidance_strength is not None
            else self.cfg.guidance_strength
        )

        if mask_chunk is None:
            mask_chunk = torch.ones(B, T_chunk, device=device, dtype=dtype)
        mask_chunk = mask_chunk.to(dtype=dtype)

        # Start from noise for this chunk
        x_t = self.cfm.sample_noise(
            (B, T_chunk, self.mel_dim), device=device, dtype=dtype
        )
        dt = 1.0 / nfe

        # We need a temporary cache for Euler steps: each step sees the
        # same past context (from prior chunks), not the iterating x_t.
        # The cache is only updated once after sampling completes.
        #
        # Save a snapshot of the incoming cache state so that each Euler
        # step starts from the same past context.
        snapshot_cache = self._snapshot_cache(kv_cache)

        final_kv_cache: MultiLayerKVCache | None = None

        for step in range(nfe):
            t_i = torch.full((B,), step * dt, device=device, dtype=dtype)

            # Restore cache from snapshot for each step (except the last
            # one, whose output cache we keep).
            if step < nfe - 1:
                step_cache = self._snapshot_cache(snapshot_cache)
            else:
                step_cache = self._snapshot_cache(snapshot_cache)

            if alpha == 0.0:
                # No guidance
                v, step_cache = self.dit.forward_chunk(
                    x_t, t_i, content_chunk, context_mel_chunk,
                    style_emb, kv_cache=step_cache,
                )
            else:
                # CFG: double-batch
                x_double = torch.cat([x_t, x_t], dim=0)
                t_double = torch.cat([t_i, t_i], dim=0)
                content_double = torch.cat(
                    [content_chunk, torch.zeros_like(content_chunk)], dim=0
                )
                context_double = torch.cat(
                    [context_mel_chunk, torch.zeros_like(context_mel_chunk)],
                    dim=0,
                )
                style_double = torch.cat(
                    [style_emb, torch.zeros_like(style_emb)], dim=0
                )

                # Double the cache for the batched call
                double_cache = self._double_cache(step_cache)

                v_double, double_cache_out = self.dit.forward_chunk(
                    x_double, t_double, content_double,
                    context_double, style_double,
                    kv_cache=double_cache,
                )

                v_cond, v_uncond = v_double.chunk(2, dim=0)
                v = (1.0 + alpha) * v_cond - alpha * v_uncond

                # Extract the conditioned half of the cache for the
                # final update.
                step_cache = self._extract_half_cache(double_cache_out)

            # Euler step
            x_t = x_t + v * dt

            # Keep the cache from the last Euler step
            if step == nfe - 1:
                final_kv_cache = step_cache

        # Inpainting blend
        mask_expanded = mask_chunk.unsqueeze(-1)
        output = mask_expanded * x_t + (1.0 - mask_expanded) * context_mel_chunk

        return output, final_kv_cache

    # ------------------------------------------------------------------
    # Full inference (non-streaming, using chunked attention)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        content_features: Tensor,
        style_waveform: Tensor,
        context_mel: Tensor | None = None,
        mask: Tensor | None = None,
        nfe: int | None = None,
        guidance_strength: float | None = None,
    ) -> Tensor:
        """Generate mel spectrogram (non-streaming, full sequence).

        Uses the streaming DiT's training forward (chunked causal mask)
        for each Euler step.  This is useful for evaluation where you
        want to match streaming behavior without chunk-by-chunk calls.

        Parameters
        ----------
        content_features : Tensor
            ``(B, T, 768)``
        style_waveform : Tensor
            ``(B, num_samples)``
        context_mel : Tensor or None
            ``(B, T, 100)``
        mask : Tensor or None
            ``(B, T)``
        nfe : int or None
        guidance_strength : float or None

        Returns
        -------
        Tensor
            ``(B, T, 100)``
        """
        B, T, _ = content_features.shape
        device = content_features.device
        dtype = content_features.dtype

        nfe = nfe if nfe is not None else self.nfe

        if context_mel is None:
            context_mel = torch.zeros(
                B, T, self.mel_dim, device=device, dtype=dtype
            )
            mask = torch.ones(B, T, device=device, dtype=dtype)
        elif mask is None:
            mask = torch.ones(B, T, device=device, dtype=dtype)
        mask = mask.to(dtype=dtype)

        # Extract style embedding
        style_emb = self.style_encoder(style_waveform)

        def velocity_fn(
            x_t: Tensor, t_step: Tensor
        ) -> Tensor:
            return self.cfg.guided_velocity(
                velocity_fn=lambda xt, ti, c, ctx, s: self.dit(
                    xt, ti, c, ctx, s
                ),
                x_t=x_t,
                t=t_step,
                content_features=content_features,
                context_mel=context_mel,
                style_emb=style_emb,
                guidance_strength=guidance_strength,
            )

        shape = (B, T, self.mel_dim)
        generated = self.cfm.euler_sample(
            velocity_fn=velocity_fn,
            shape=shape,
            nfe=nfe,
            device=device,
            dtype=dtype,
        )

        mask_expanded = mask.unsqueeze(-1)
        output = mask_expanded * generated + (1.0 - mask_expanded) * context_mel

        return output

    # ------------------------------------------------------------------
    # Offline weight loading
    # ------------------------------------------------------------------

    def load_from_offline(
        self, offline_checkpoint_path: str
    ) -> None:
        """Load weights from an offline Stylizer checkpoint.

        Loads the checkpoint, extracts DiT weights and style encoder
        weights, and maps them to the streaming model.  Since the
        parameter names match (the only difference is attention masking,
        not parameter structure), loading is straightforward.

        Parameters
        ----------
        offline_checkpoint_path : str
            Path to the offline Stylizer checkpoint (.pt file).
        """
        checkpoint = torch.load(
            offline_checkpoint_path, map_location="cpu", weights_only=True
        )

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Try to load directly first
        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        logger.info(
            "Loaded StreamingStylizer from offline checkpoint: "
            "%d missing keys, %d unexpected keys",
            len(missing),
            len(unexpected),
        )

        if missing:
            logger.debug("Missing keys: %s", missing)
        if unexpected:
            logger.debug("Unexpected keys: %s", unexpected)

    # ------------------------------------------------------------------
    # Cache helpers for CFG double-batching
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_cache(
        cache: MultiLayerKVCache | None,
    ) -> MultiLayerKVCache | None:
        """Create a deep copy of a KV cache.

        Clones all cached tensors so modifications do not affect the
        original.
        """
        if cache is None:
            return None

        snapshot = MultiLayerKVCache(
            num_layers=len(cache),
            max_frames=cache.caches[0].max_frames,
        )
        for i, layer_cache in enumerate(cache.caches):
            k, v = layer_cache.get()
            if k is not None:
                snapshot.caches[i]._k = k.clone()
                snapshot.caches[i]._v = v.clone()
        return snapshot

    @staticmethod
    def _double_cache(
        cache: MultiLayerKVCache | None,
    ) -> MultiLayerKVCache | None:
        """Double a KV cache along the batch dimension for CFG.

        Creates a new cache where each layer's K and V are repeated
        along the batch dimension: ``(B, H, T, D) -> (2B, H, T, D)``.
        """
        if cache is None:
            return None

        doubled = MultiLayerKVCache(
            num_layers=len(cache),
            max_frames=cache.caches[0].max_frames,
        )
        for i, layer_cache in enumerate(cache.caches):
            k, v = layer_cache.get()
            if k is not None:
                doubled.caches[i]._k = k.repeat(2, 1, 1, 1)
                doubled.caches[i]._v = v.repeat(2, 1, 1, 1)
        return doubled

    @staticmethod
    def _extract_half_cache(
        cache: MultiLayerKVCache | None,
    ) -> MultiLayerKVCache | None:
        """Extract the first half (conditioned) of a doubled cache.

        After a CFG double-batch forward, the cache has batch size 2B.
        This extracts the first B entries.
        """
        if cache is None:
            return None

        half = MultiLayerKVCache(
            num_layers=len(cache),
            max_frames=cache.caches[0].max_frames,
        )
        for i, layer_cache in enumerate(cache.caches):
            k, v = layer_cache.get()
            if k is not None:
                B_half = k.shape[0] // 2
                half.caches[i]._k = k[:B_half]
                half.caches[i]._v = v[:B_half]
        return half

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        n_trainable = self.num_parameters(trainable_only=True) / 1e6
        n_total = self.num_parameters(trainable_only=False) / 1e6
        return (
            f"{self.__class__.__name__}(\n"
            f"  (dit): {self.dit.num_layers} layers, "
            f"hidden={self.dit.hidden_size}, "
            f"heads={self.dit.num_heads}, "
            f"chunk_size={self.dit.chunk_size}\n"
            f"  (style_encoder): {self.style_encoder.wavlm_model_id}\n"
            f"  (cfm): sigma_min={self.cfm.sigma_min}\n"
            f"  (cfg): alpha={self.cfg.guidance_strength}, "
            f"drops=({self.cfg.content_drop_prob}, "
            f"{self.cfg.context_drop_prob}, "
            f"{self.cfg.style_drop_prob})\n"
            f"  mel_dim={self.mel_dim}, nfe={self.nfe}\n"
            f"  trainable_params={n_trainable:.2f}M, "
            f"total_params={n_total:.2f}M\n"
            f")"
        )
