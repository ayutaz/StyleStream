"""Streaming HuBERT wrapper for the StyleStream streaming Destylizer.

Wraps HuggingFace's HuBERT-Large with causal modifications for streaming:
    - CNN feature extractor: left-only padding for causality
    - Transformer layers: chunked causal attention mask
    - Layer 18 extraction for content features
    - Unfreezing support for MSE distillation fine-tuning

StyleStream spec:
    - HuBERT-Large (facebook/hubert-large-ls960-ft)
    - 24 Transformer layers, hidden 1024, 16 heads
    - CNN: 7 layers, total stride 320 (16kHz -> 50Hz)
    - Extract layer 18 output -> 1024-dim features
    - Chunk size: 30 frames (600ms)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from stylestream.streaming.attention_mask import (
    build_chunked_causal_mask,
    chunked_causal_mask_to_attn_bias,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_ID: str = "facebook/hubert-large-ls960-ft"
_HUBERT_HIDDEN: int = 1024  # HuBERT-Large hidden dimension
_HUBERT_NUM_LAYERS: int = 24  # Transformer layers in HuBERT-Large
_HUBERT_NUM_HEADS: int = 16  # Attention heads in HuBERT-Large
_SAMPLE_RATE: int = 16_000
_HOP_LENGTH: int = 320  # Total CNN stride -> 50Hz at 16kHz

# HuBERT CNN feature extractor specs (7 layers).
_CNN_KERNEL_SIZES: list[int] = [10, 3, 3, 3, 3, 2, 2]
_CNN_STRIDES: list[int] = [5, 2, 2, 2, 2, 2, 2]


# ---------------------------------------------------------------------------
# Causal CNN wrapper
# ---------------------------------------------------------------------------


class _CausalConv1d(nn.Module):
    """Wrapper that converts a Conv1d to causal (left-only padding).

    Replaces symmetric padding with ``(kernel_size - 1, 0)`` left-padding
    so the convolution never sees future samples.

    Parameters
    ----------
    conv : nn.Conv1d
        Original convolution layer (weights are shared, not copied).
    """

    def __init__(self, conv: nn.Conv1d) -> None:
        super().__init__()
        self.conv = conv
        # Store the amount of left-padding needed for causal behaviour.
        # For stride-1 convolutions: pad = (kernel_size - 1) * dilation.
        # HuBERT CNN uses dilation=1 everywhere and stride >= 1.
        kernel_size = conv.kernel_size[0]
        dilation = conv.dilation[0]
        self._left_pad = (kernel_size - 1) * dilation

        # Zero out any existing padding so we control it ourselves.
        self.conv.padding = (0,)

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal (left-only) padding then convolution.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, C, T)``.

        Returns
        -------
        Tensor
            Convolution output with causal padding applied.
        """
        if self._left_pad > 0:
            x = F.pad(x, (self._left_pad, 0))
        return self.conv(x)


# ---------------------------------------------------------------------------
# StreamingHuBERT
# ---------------------------------------------------------------------------


class StreamingHuBERT(nn.Module):
    """Streaming-capable HuBERT-Large wrapper.

    Wraps the HuggingFace HuBERT model with:
    1. Causal CNN feature extraction (left-only padding)
    2. Chunked causal self-attention in Transformer layers
    3. Layer 18 output extraction
    4. Optional unfreezing for fine-tuning

    The HuBERT model is loaded lazily on first forward pass (not at
    construction) so that importing this module does not trigger a
    multi-GB download.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID (default ``"facebook/hubert-large-ls960-ft"``).
    layer : int
        Which transformer layer output to extract (default 18).
    chunk_size : int
        Chunk size in frames for chunked causal attention (default 30).
    max_cache_frames : int
        Max KV cache frames (default 250).
    frozen : bool
        If True, freeze all parameters. Default False (for distillation).
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        layer: int = 18,
        chunk_size: int = 30,
        max_cache_frames: int = 250,
        frozen: bool = False,
    ) -> None:
        super().__init__()

        if layer < 1 or layer > _HUBERT_NUM_LAYERS:
            raise ValueError(
                f"layer must be in [1, {_HUBERT_NUM_LAYERS}], got {layer}. "
                f"Layer indexing: 0 = CNN output, 1..{_HUBERT_NUM_LAYERS} = "
                f"Transformer layers."
            )

        self._model_id = model_id
        self._layer = layer
        self._chunk_size = chunk_size
        self._max_cache_frames = max_cache_frames
        self._frozen = frozen

        # The underlying HuBERT model is loaded lazily via _ensure_loaded().
        self.hubert: nn.Module | None = None
        self._loaded = False

        # CNN buffer for streaming: stores the tail of the previous chunk's
        # raw audio so that the CNN feature extractor has valid left context.
        # Total receptive field of the 7-layer CNN.
        self._cnn_receptive_field = self._compute_cnn_receptive_field()

        # Buffer for streaming CNN context (registered as non-persistent
        # buffer once the model is loaded and device is known).
        self._cnn_context_buffer: Tensor | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load and patch the HuBERT model on first use."""
        if self._loaded:
            return

        try:
            from transformers import HubertModel
        except ImportError as exc:
            raise ImportError(
                "The `transformers` library is required for StreamingHuBERT. "
                "Install it with: pip install transformers\n"
                "Or: uv sync --extra train"
            ) from exc

        logger.info(
            "Loading HuBERT from '%s' (layer %d, chunk_size=%d) ...",
            self._model_id,
            self._layer,
            self._chunk_size,
        )
        self.hubert = self._load_and_patch_model(self._model_id)
        self._loaded = True

        if self._frozen:
            self.freeze()

        n_params = sum(p.numel() for p in self.hubert.parameters()) / 1e6
        logger.info(
            "StreamingHuBERT ready: %.1fM params, frozen=%s",
            n_params,
            self._frozen,
        )

    def _load_and_patch_model(self, model_id: str) -> nn.Module:
        """Load the HuggingFace HuBERT model and apply causal patches.

        1. Load pre-trained HuBERT-Large.
        2. Causalize CNN layers (left-only padding).
        3. Configure ``output_hidden_states=True`` for layer extraction.

        Parameters
        ----------
        model_id : str
            HuggingFace model identifier.

        Returns
        -------
        nn.Module
            The patched HuBERT model.
        """
        from transformers import HubertModel

        model = HubertModel.from_pretrained(model_id)

        # Enable hidden state output so we can extract a specific layer.
        model.config.output_hidden_states = True

        # Causalize CNN feature extractor.
        self._causalize_cnn(model)

        return model

    # ------------------------------------------------------------------
    # CNN causalization
    # ------------------------------------------------------------------

    def _causalize_cnn(self, model: nn.Module) -> None:
        """Replace symmetric CNN padding with left-only (causal) padding.

        HuBERT's CNN feature extractor consists of 7 ``Conv1d`` layers
        wrapped in ``Wav2Vec2GroupNormConvLayer`` or
        ``Wav2Vec2NoLayerNormConvLayer`` modules.  Each contains a ``.conv``
        attribute that we wrap with :class:`_CausalConv1d`.

        Parameters
        ----------
        model : nn.Module
            The HuBERT model to patch in-place.
        """
        feature_extractor = model.feature_extractor

        for i, conv_layer in enumerate(feature_extractor.conv_layers):
            original_conv = conv_layer.conv
            causal_conv = _CausalConv1d(original_conv)

            # Replace the Conv1d with our causal wrapper.
            conv_layer.conv = causal_conv

            logger.debug(
                "Causalized CNN layer %d: kernel=%d, stride=%d, left_pad=%d",
                i,
                original_conv.kernel_size[0],
                original_conv.stride[0],
                causal_conv._left_pad,
            )

    # ------------------------------------------------------------------
    # Chunked causal attention mask
    # ------------------------------------------------------------------

    def _build_attention_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        """Build a chunked causal attention mask for the Transformer layers.

        The mask is a 4-D float tensor with shape
        ``(1, 1, seq_len, seq_len)`` where allowed positions have value 0
        and blocked positions have value ``-inf``.  This is passed via
        HuBERT's ``attention_mask`` mechanism.

        HuggingFace's ``Wav2Vec2Model`` / ``HubertModel`` uses the
        attention mask differently from standard ``attention_mask`` in
        BERT-like models.  The Wav2Vec2 encoder's ``_get_feature_vector_attention_mask``
        method computes a 1-D mask from input lengths.  For our chunked
        causal masking we bypass this and inject the mask directly.

        Parameters
        ----------
        seq_len : int
            Number of frames in the feature sequence.
        device : torch.device
            Target device.

        Returns
        -------
        Tensor
            ``(1, 1, seq_len, seq_len)`` additive attention bias.
        """
        # Build block lower-triangular bool mask.
        mask = build_chunked_causal_mask(
            seq_len=seq_len,
            chunk_size=self._chunk_size,
            device=device,
        )  # (seq_len, seq_len) bool

        # Convert to additive bias: 0 for allowed, -inf for blocked.
        attn_bias = chunked_causal_mask_to_attn_bias(mask)  # (1, 1, S, S)

        return attn_bias

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        waveform: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Extract layer-N features with chunked causal attention.

        On the first call, the HuBERT model is lazily loaded and patched.

        Parameters
        ----------
        waveform : Tensor
            Raw audio at 16kHz, shape ``(B, T_samples)``.
        attention_mask : Tensor or None
            Optional attention mask for the transformer layers.  If
            ``None``, a chunked causal mask is built automatically.
            Shape should be ``(B, T_frames)`` (1 = valid, 0 = padding)
            or ``(1, 1, T_frames, T_frames)`` (additive bias).

        Returns
        -------
        Tensor
            Layer-N features, shape ``(B, T_frames, 1024)``.
        """
        self._ensure_loaded()
        assert self.hubert is not None

        # Run the HuBERT model.  We pass `output_hidden_states=True` to
        # obtain all intermediate Transformer layer outputs.
        #
        # HuBERT's encoder architecture (inheriting from Wav2Vec2):
        #   1. CNN feature extractor -> (B, T_frames, CNN_out_dim)
        #   2. Feature projection -> (B, T_frames, hidden_size)
        #   3. Transformer encoder layers (x24)
        #
        # outputs.hidden_states is a tuple of (num_layers + 1) tensors:
        #   index 0 = post-projection (pre-Transformer) features
        #   index 1..24 = Transformer layer outputs
        # Each has shape (B, T_frames, hidden_size).

        # We need to apply the chunked causal mask to the Transformer
        # attention layers.  HuBERT (Wav2Vec2) accepts `attention_mask`
        # which it converts internally.  However, the standard interface
        # expects a simple (B, T) padding mask, not a 2-D attention bias.
        #
        # Strategy: Use forward hooks on the encoder to inject the causal
        # mask, or use the model with a custom approach.
        #
        # The simplest correct approach for HuggingFace HuBERT:
        # - Run the CNN feature extractor manually
        # - Run the Transformer encoder with our custom attention mask
        #
        # We use the model's internal methods for correct behaviour.

        outputs = self._forward_with_causal_mask(waveform, attention_mask)

        # Extract the requested layer's hidden states.
        # hidden_states tuple: index 0 = CNN/projection output,
        # indices 1..24 = Transformer layers.
        hidden = outputs.hidden_states[self._layer]  # (B, T_frames, 1024)

        return hidden

    def _forward_with_causal_mask(
        self,
        waveform: Tensor,
        attention_mask: Tensor | None,
    ) -> Any:
        """Run HuBERT with chunked causal attention masking.

        This method handles the internal mechanics of injecting a
        chunked causal mask into HuBERT's Transformer encoder.

        HuggingFace's Wav2Vec2/HuBERT encoder accepts an ``attention_mask``
        of shape ``(B, T_samples)`` which it internally converts to the
        correct shape for the Transformer layers.  We hook into this
        pipeline to replace the final mask with our chunked causal version.

        Parameters
        ----------
        waveform : Tensor
            Raw audio, shape ``(B, T_samples)``.
        attention_mask : Tensor or None
            Caller-provided mask, or None for auto-generated chunked
            causal mask.

        Returns
        -------
        model output
            HuBERT model output with ``hidden_states`` attribute.
        """
        assert self.hubert is not None

        # Step 1: Run CNN feature extractor to get frame count.
        # We do this in two stages to know T_frames for mask construction.
        extract_features = self.hubert.feature_extractor(waveform)
        # extract_features: (B, CNN_out_dim, T_frames) -- channels first
        extract_features = extract_features.transpose(1, 2)
        # -> (B, T_frames, CNN_out_dim)

        T_frames = extract_features.shape[1]

        # Step 2: Feature projection (LayerNorm + Linear + Dropout).
        hidden_states, _ = self.hubert.feature_projection(extract_features)
        # hidden_states: (B, T_frames, 1024)

        # Step 3: Build the chunked causal attention mask.
        if attention_mask is None:
            # Build a 4-D chunked causal bias: (1, 1, T, T).
            causal_bias = self._build_attention_mask(
                T_frames, device=waveform.device
            )
        elif attention_mask.dim() == 4:
            # Caller provided a pre-built 4-D mask.
            causal_bias = attention_mask
        else:
            # Caller provided a (B, T_frames) padding mask.  Combine with
            # our chunked causal mask.
            causal_bias = self._build_attention_mask(
                T_frames, device=waveform.device
            )
            # Also mask padded positions: set blocked columns to -inf.
            # attention_mask: (B, T_frames), 1 = valid, 0 = pad.
            pad_bias = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * -1e9
            # pad_bias: (B, 1, 1, T_frames) -- broadcasts over query dim
            causal_bias = causal_bias + pad_bias

        # Step 4: Run the Transformer encoder with our custom mask.
        # HuBERT's encoder (Wav2Vec2Encoder / Wav2Vec2EncoderStableLayerNorm)
        # calls each layer with `attention_mask` as a 4-D float tensor
        # (the "extended attention mask").  We provide our chunked causal
        # bias directly.
        encoder_outputs = self.hubert.encoder(
            hidden_states,
            attention_mask=causal_bias,
            output_hidden_states=True,
            return_dict=True,
        )

        # Build a result-like object with hidden_states that includes
        # the pre-Transformer features at index 0.
        class _HuBERTOutput:
            pass

        result = _HuBERTOutput()
        # Prepend the post-projection features (layer 0) to match
        # HuBERT's standard hidden_states indexing.
        result.hidden_states = (hidden_states,) + encoder_outputs.hidden_states
        result.last_hidden_state = encoder_outputs.last_hidden_state

        return result

    # ------------------------------------------------------------------
    # Streaming inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_features_streaming(
        self,
        waveform_chunk: Tensor,
    ) -> Tensor:
        """Extract features for a single streaming chunk.

        Uses internal CNN context buffer to ensure the CNN feature
        extractor has valid left context at chunk boundaries.

        This method is designed for incremental streaming: call it
        repeatedly with successive audio chunks.  Use
        :meth:`reset_streaming_state` between utterances.

        Parameters
        ----------
        waveform_chunk : Tensor
            Audio chunk at 16kHz, shape ``(B, chunk_samples)``.

        Returns
        -------
        Tensor
            ``(B, chunk_frames, 1024)`` features for this chunk.
        """
        self._ensure_loaded()
        assert self.hubert is not None

        device = waveform_chunk.device
        has_context = self._cnn_context_buffer is not None

        # Prepend CNN context from previous chunk for valid left context.
        if has_context:
            # Ensure buffer is on the same device.
            if self._cnn_context_buffer.device != device:
                self._cnn_context_buffer = self._cnn_context_buffer.to(device)
            waveform_with_context = torch.cat(
                [self._cnn_context_buffer, waveform_chunk], dim=1
            )
        else:
            waveform_with_context = waveform_chunk

        # Update CNN context buffer: keep the last receptive_field samples.
        ctx_len = min(self._cnn_receptive_field, waveform_chunk.shape[1])
        self._cnn_context_buffer = waveform_chunk[:, -ctx_len:].detach().clone()

        # Run the full forward pass on the context-extended waveform.
        # The causal CNN + chunked attention will produce valid output.
        features = self.forward(waveform_with_context)
        # features: (B, T_total_frames, 1024)

        # We only want the frames corresponding to the current chunk,
        # not the context prefix.  Calculate how many frames the context
        # prefix produced and trim them.
        if has_context:
            context_samples = waveform_with_context.shape[1] - waveform_chunk.shape[1]
            context_frames = self._samples_to_frames(context_samples)
            features = features[:, context_frames:, :]

        return features

    def reset_streaming_state(self) -> None:
        """Reset all streaming state between utterances.

        Clears the CNN context buffer.  Must be called when starting
        a new utterance in streaming mode.
        """
        self._cnn_context_buffer = None

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze all parameters (disable gradient computation)."""
        self._ensure_loaded()
        assert self.hubert is not None

        for param in self.hubert.parameters():
            param.requires_grad_(False)
        self._frozen = True
        logger.info("StreamingHuBERT: all parameters frozen.")

    def unfreeze(self, deep_layers_only: bool = True) -> None:
        """Unfreeze parameters for fine-tuning.

        Parameters
        ----------
        deep_layers_only : bool
            If True (default), only unfreeze Transformer layers >= 13
            (the upper half of the 24-layer stack).  The CNN feature
            extractor and early Transformer layers remain frozen.
            If False, unfreeze all parameters.
        """
        self._ensure_loaded()
        assert self.hubert is not None

        if not deep_layers_only:
            # Unfreeze everything.
            for param in self.hubert.parameters():
                param.requires_grad_(True)
            self._frozen = False
            logger.info("StreamingHuBERT: all parameters unfrozen.")
            return

        # First freeze everything, then selectively unfreeze.
        for param in self.hubert.parameters():
            param.requires_grad_(False)

        # Unfreeze deep Transformer layers (>= 13).
        # HuBERT's encoder has a `layers` ModuleList of Transformer layers.
        encoder = self.hubert.encoder
        layers = encoder.layers

        unfrozen_count = 0
        for i, layer in enumerate(layers):
            if i >= 13:
                for param in layer.parameters():
                    param.requires_grad_(True)
                    unfrozen_count += 1

        # Also unfreeze the encoder's final LayerNorm if present.
        if hasattr(encoder, "layer_norm") and encoder.layer_norm is not None:
            for param in encoder.layer_norm.parameters():
                param.requires_grad_(True)
                unfrozen_count += 1

        self._frozen = False
        total_unfrozen = sum(
            p.numel() for p in self.hubert.parameters() if p.requires_grad
        )
        logger.info(
            "StreamingHuBERT: unfrozen layers >= 13 (%d params, %.1fM total).",
            unfrozen_count,
            total_unfrozen / 1e6,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def layer(self) -> int:
        """Which Transformer layer's output is extracted."""
        return self._layer

    @property
    def chunk_size(self) -> int:
        """Chunk size in frames for chunked causal attention."""
        return self._chunk_size

    @property
    def hidden_size(self) -> int:
        """Output feature dimension (1024 for HuBERT-Large)."""
        return _HUBERT_HIDDEN

    @property
    def is_frozen(self) -> bool:
        """Whether all parameters are frozen."""
        return self._frozen

    @property
    def is_loaded(self) -> bool:
        """Whether the HuBERT model has been loaded."""
        return self._loaded

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cnn_receptive_field() -> int:
        """Compute the receptive field of HuBERT's CNN feature extractor.

        The receptive field determines how many past audio samples are
        needed for the CNN to produce valid output at a given position.

        For a stack of strided convolutions::

            receptive_field = 1
            for each layer (from last to first):
                receptive_field = (receptive_field - 1) * stride + kernel_size

        Returns
        -------
        int
            Receptive field in samples.
        """
        rf = 1
        # Iterate from last layer to first.
        for kernel, stride in zip(
            reversed(_CNN_KERNEL_SIZES), reversed(_CNN_STRIDES)
        ):
            rf = (rf - 1) * stride + kernel
        return rf

    @staticmethod
    def _samples_to_frames(num_samples: int) -> int:
        """Convert a sample count to the expected number of HuBERT frames.

        HuBERT's CNN feature extractor has a total stride of 320,
        producing one frame per 320 input samples.

        Parameters
        ----------
        num_samples : int
            Number of audio samples at 16kHz.

        Returns
        -------
        int
            Number of output frames.
        """
        # With causal (left-only) padding of (kernel - 1), the output
        # length for each Conv1d layer is:
        #   floor((length + (kernel - 1) - kernel) / stride) + 1
        #   = floor((length - 1) / stride) + 1
        # We compute layer-by-layer for accuracy.
        length = num_samples
        for kernel, stride in zip(_CNN_KERNEL_SIZES, _CNN_STRIDES):
            length = (length - 1) // stride + 1
        return max(length, 0)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters.

        Parameters
        ----------
        trainable_only : bool
            If True (default), count only trainable parameters.

        Returns
        -------
        int
            Number of (trainable) parameters.
        """
        if self.hubert is None:
            return 0
        if trainable_only:
            return sum(p.numel() for p in self.hubert.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.hubert.parameters())

    def __repr__(self) -> str:
        loaded = "loaded" if self._loaded else "not loaded"
        n_params = self.num_parameters(trainable_only=False) / 1e6
        return (
            f"{self.__class__.__name__}(\n"
            f"  model_id='{self._model_id}',\n"
            f"  layer={self._layer},\n"
            f"  chunk_size={self._chunk_size},\n"
            f"  max_cache_frames={self._max_cache_frames},\n"
            f"  frozen={self._frozen},\n"
            f"  status={loaded},\n"
            f"  cnn_receptive_field={self._cnn_receptive_field},\n"
            f"  params={n_params:.1f}M,\n"
            f")"
        )
