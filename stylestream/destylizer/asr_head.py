"""ASR heads for StyleStream Destylizer training.

The Destylizer is trained with ASR loss to ensure that the content features
(extracted by HuBERT -> Conformer -> FSQ) preserve linguistic information.
At inference time the ASR head is discarded entirely -- only the continuous
pre-quantization features ``fc`` are forwarded to the Stylizer.

Two modes are supported:

* **CTC** (``loss_type="ctc"``): A simple linear projection from the FSQ
  output to log-probabilities over the character vocabulary.  This is the
  recommended starting point -- it is simpler and faster to train.

* **Seq2seq cross-entropy** (``loss_type="seq2seq_ce"``): A 4-layer
  Transformer decoder with cross-attention into the encoder output.
  Teacher forcing is used during training; the target is shifted right
  by one position and prepended with ``<sos>``.

Architecture (from ``configs/destylizer/offline.yaml``)::

    asr_decoder:
      num_layers: 4
      hidden_size: 768
      ffn_size: 3072
      num_heads: 12
      dropout: 0.1
      vocab_size: 30
      loss_type: "seq2seq_ce"
      label_smoothing: 0.1

Vocabulary layout (30 tokens, see :mod:`stylestream.data.text`)::

    0  <blank>   CTC blank
    1  <sos>     Start of sequence
    2  <eos>     End of sequence
    3  <space>   Word separator
    4-29  a-z    Lowercase English letters

Reference: StyleStream paper (arXiv:2602.20113), Section 10.8.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Token constants (mirrored from stylestream.data.text for decoupling)
_BLANK_ID = 0
_SOS_ID = 1
_EOS_ID = 2


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding (for the seq2seq decoder)
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.

    This is the standard formulation from Vaswani et al. (2017).  It is used
    for the Transformer ASR decoder because the decoder sequences are short
    (character-level transcripts) and do not need learned or ALiBi-style
    encodings.

    Parameters
    ----------
    d_model : int
        Embedding / hidden dimension.
    max_len : int
        Maximum sequence length supported.
    dropout : float
        Dropout applied after adding the positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1, max_len, d_model) for easy broadcast over the batch dimension
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, S, D)``.

        Returns
        -------
        torch.Tensor
            Same shape as *x*, with positional encoding added and dropout
            applied.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# CTCHead
# ---------------------------------------------------------------------------

class CTCHead(nn.Module):
    """Simple linear projection for CTC loss.

    This is the recommended first approach (simpler, faster training).
    Projects encoder output directly to vocabulary logits, then applies
    ``log_softmax`` to produce log-probabilities for CTC.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the FSQ / encoder output (default 768).
    vocab_size : int
        Number of tokens in the character vocabulary (default 30).
    dropout : float
        Dropout rate applied before the linear projection.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        vocab_size: int = 30,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project encoder output to log-probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, hidden_size)`` -- FSQ output.

        Returns
        -------
        torch.Tensor
            Shape ``(B, T, vocab_size)`` -- log-probabilities for CTC.
        """
        x = self.dropout(x)
        logits = self.proj(x)
        return F.log_softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# TransformerASRDecoder
# ---------------------------------------------------------------------------

class TransformerASRDecoder(nn.Module):
    """4-layer Transformer decoder with cross-attention for seq2seq ASR.

    Used when ``loss_type="seq2seq_ce"`` in the Destylizer config.  The
    decoder attends to the FSQ encoder output via cross-attention and
    predicts the next character in the transcript using teacher forcing.

    Parameters
    ----------
    hidden_size : int
        Model / embedding dimension.
    ffn_size : int
        Feed-forward inner dimension.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of Transformer decoder layers.
    vocab_size : int
        Character vocabulary size.
    dropout : float
        Dropout rate used throughout.
    label_smoothing : float
        Label smoothing for the cross-entropy loss (stored but not used
        directly -- the :class:`ASRHead` wrapper handles loss computation).
    """

    def __init__(
        self,
        hidden_size: int = 768,
        ffn_size: int = 3072,
        num_heads: int = 12,
        num_layers: int = 4,
        vocab_size: int = 30,
        dropout: float = 0.1,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Token embedding + sinusoidal positional encoding
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_enc = SinusoidalPositionalEncoding(hidden_size, dropout=dropout)

        # Transformer decoder stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Output projection to vocabulary
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        encoder_output: torch.Tensor,
        target_ids: torch.Tensor,
        encoder_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the decoder with teacher forcing.

        The target sequence is shifted right by one position: the first
        input token is always ``<sos>`` and the last target token is
        dropped from the input (it becomes the final prediction target).

        Parameters
        ----------
        encoder_output : torch.Tensor
            Shape ``(B, T, hidden_size)`` -- FSQ output.
        target_ids : torch.Tensor
            Shape ``(B, S)`` -- ground-truth token IDs including ``<sos>``
            at position 0 and ``<eos>`` at the end.  The decoder input is
            ``target_ids[:, :-1]`` and the prediction target is
            ``target_ids[:, 1:]``.
        encoder_padding_mask : torch.Tensor or None
            Shape ``(B, T)``.  ``True`` for padded encoder positions.

        Returns
        -------
        torch.Tensor
            Shape ``(B, S-1, vocab_size)`` -- raw logits (no softmax).
        """
        # Shift right: decoder input = all tokens except the last
        decoder_input = target_ids[:, :-1]  # (B, S-1)
        seq_len = decoder_input.size(1)

        # Embed + positional encoding
        x = self.embedding(decoder_input) * math.sqrt(self.hidden_size)
        x = self.pos_enc(x)  # (B, S-1, hidden_size)

        # Causal mask for decoder self-attention
        causal_mask = self._build_causal_mask(seq_len, x.device)

        # Run through decoder stack
        x = self.decoder(
            tgt=x,
            memory=encoder_output,
            tgt_mask=causal_mask,
            memory_key_padding_mask=encoder_padding_mask,
        )
        x = self.layer_norm(x)

        # Project to vocabulary
        logits = self.output_proj(x)  # (B, S-1, vocab_size)
        return logits

    def _build_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Build a causal (upper-triangular) attention mask for decoder self-attention.

        Parameters
        ----------
        size : int
            Sequence length.
        device : torch.device
            Target device.

        Returns
        -------
        torch.Tensor
            Shape ``(size, size)``.  Positions that should be masked (future)
            are ``-inf``; valid positions are ``0.0``.
        """
        return torch.triu(
            torch.full((size, size), float("-inf"), device=device),
            diagonal=1,
        )


# ---------------------------------------------------------------------------
# ASRHead (unified factory / wrapper)
# ---------------------------------------------------------------------------

class ASRHead(nn.Module):
    """Unified ASR head that delegates to CTC or seq2seq decoder.

    This class wraps either a :class:`CTCHead` (for CTC loss) or a
    :class:`TransformerASRDecoder` (for seq2seq cross-entropy loss) and
    provides a common ``forward`` / ``compute_loss`` interface so the
    Destylizer training loop does not need to branch on ``loss_type``.

    Parameters
    ----------
    loss_type : str
        ``"ctc"`` for CTC loss, ``"seq2seq_ce"`` for cross-entropy with
        a Transformer decoder.
    hidden_size : int
        Encoder / FSQ output dimension.
    vocab_size : int
        Character vocabulary size.
    num_layers : int
        Number of Transformer decoder layers (only for ``seq2seq_ce``).
    ffn_size : int
        Feed-forward inner dimension (only for ``seq2seq_ce``).
    num_heads : int
        Attention heads (only for ``seq2seq_ce``).
    dropout : float
        Dropout rate.
    label_smoothing : float
        Label smoothing for cross-entropy loss (only for ``seq2seq_ce``).
    """

    VALID_LOSS_TYPES = ("ctc", "seq2seq_ce")

    def __init__(
        self,
        loss_type: str = "ctc",
        hidden_size: int = 768,
        vocab_size: int = 30,
        num_layers: int = 4,
        ffn_size: int = 3072,
        num_heads: int = 12,
        dropout: float = 0.1,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()

        if loss_type not in self.VALID_LOSS_TYPES:
            raise ValueError(
                f"Unknown loss_type={loss_type!r}. "
                f"Must be one of {self.VALID_LOSS_TYPES}."
            )

        self.loss_type = loss_type
        self.vocab_size = vocab_size

        if loss_type == "ctc":
            self.head = CTCHead(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                dropout=dropout,
            )
            self.ctc_loss_fn = nn.CTCLoss(
                blank=_BLANK_ID,
                reduction="mean",
                zero_infinity=True,
            )
        else:
            # seq2seq_ce
            self.head = TransformerASRDecoder(
                hidden_size=hidden_size,
                ffn_size=ffn_size,
                num_heads=num_heads,
                num_layers=num_layers,
                vocab_size=vocab_size,
                dropout=dropout,
                label_smoothing=label_smoothing,
            )
            self.ce_loss_fn = nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=label_smoothing,
            )

    def forward(
        self,
        encoder_output: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        encoder_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ASR logits.

        Parameters
        ----------
        encoder_output : torch.Tensor
            Shape ``(B, T, hidden_size)`` -- FSQ output.
        target_ids : torch.Tensor or None
            Shape ``(B, S)`` -- token IDs (required for ``seq2seq_ce``,
            ignored for ``ctc``).
        encoder_padding_mask : torch.Tensor or None
            Shape ``(B, T)`` -- ``True`` for padded encoder positions
            (used by ``seq2seq_ce``; ignored by ``ctc``).

        Returns
        -------
        torch.Tensor
            * CTC: ``(B, T, vocab_size)`` log-probabilities.
            * seq2seq: ``(B, S-1, vocab_size)`` raw logits.
        """
        if self.loss_type == "ctc":
            return self.head(encoder_output)

        # seq2seq_ce
        if target_ids is None:
            raise ValueError(
                "target_ids is required for loss_type='seq2seq_ce'."
            )
        return self.head(
            encoder_output=encoder_output,
            target_ids=target_ids,
            encoder_padding_mask=encoder_padding_mask,
        )

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        encoder_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the appropriate loss (CTC or CE with label smoothing).

        Parameters
        ----------
        logits : torch.Tensor
            Output of :meth:`forward`.
            * CTC: ``(B, T, vocab_size)`` log-probabilities.
            * seq2seq: ``(B, S-1, vocab_size)`` raw logits.
        targets : torch.Tensor
            Shape ``(B, S)`` -- ground-truth token IDs.
            * CTC: raw character IDs without ``<sos>`` / ``<eos>``.
            * seq2seq: includes ``<sos>`` at position 0 and ``<eos>`` at
              the end.  The loss is computed against ``targets[:, 1:]``
              (shifted by one to align with decoder output).
        encoder_lengths : torch.Tensor
            Shape ``(B,)`` -- valid (non-padded) length for each encoder
            output in frames.  Used by CTC.
        target_lengths : torch.Tensor
            Shape ``(B,)`` -- valid length of each target sequence.
            * CTC: length of the raw character sequence (no special tokens).
            * seq2seq: length including ``<sos>`` / ``<eos>`` (the loss
              function handles the offset internally).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        if self.loss_type == "ctc":
            return self._ctc_loss(logits, targets, encoder_lengths, target_lengths)
        return self._ce_loss(logits, targets, target_lengths)

    # -- private loss helpers -----------------------------------------------

    def _ctc_loss(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        encoder_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.

        ``nn.CTCLoss`` expects inputs in ``(T, B, C)`` layout, so we
        transpose here.
        """
        # log_probs: (B, T, V) -> (T, B, V) as required by CTCLoss
        log_probs = log_probs.transpose(0, 1)  # (T, B, V)

        # CTCLoss expects targets as a 1-D concatenation or (B, S) with
        # target_lengths.  The (B, S) form is more convenient.
        return self.ctc_loss_fn(
            log_probs,
            targets,
            encoder_lengths,
            target_lengths,
        )

    def _ce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss with label smoothing.

        The decoder output at position *i* predicts ``targets[:, i+1]``
        (teacher-forced, shifted by one).  We mask out positions beyond
        each sequence's valid length to avoid penalizing padding.
        """
        # Prediction targets are the shifted-right ground truth:
        # decoder predicts targets[:, 1], targets[:, 2], ..., targets[:, S-1]
        shifted_targets = targets[:, 1:]  # (B, S-1)
        B, S_minus_1 = shifted_targets.shape

        # Build a padding mask: positions beyond (target_length - 1) are padded
        # (target_lengths includes <sos> and <eos>, so valid output length
        # is target_lengths - 1).
        valid_output_lengths = target_lengths - 1  # (B,)
        position_indices = torch.arange(S_minus_1, device=targets.device).unsqueeze(0)  # (1, S-1)
        padding_mask = position_indices >= valid_output_lengths.unsqueeze(1)  # (B, S-1)

        # Replace padded positions with ignore_index so they don't contribute
        loss_targets = shifted_targets.clone()
        loss_targets[padding_mask] = -100

        # Reshape for CrossEntropyLoss: (B*(S-1), V) and (B*(S-1),)
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = loss_targets.reshape(-1)

        return self.ce_loss_fn(logits_flat, targets_flat)
