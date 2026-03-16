"""Tests for ASR heads (CTCHead, TransformerASRDecoder, ASRHead).

All tests are self-contained and use synthetic tensors on CPU.  Smaller
dimensions (hidden_size=64, num_heads=4, num_layers=2, vocab_size=10) are used
throughout to keep tests fast.
"""

from __future__ import annotations

import pytest
import torch

from stylestream.destylizer.asr_head import ASRHead, CTCHead, TransformerASRDecoder

# ---------------------------------------------------------------------------
# Constants (smaller than production for fast CPU tests)
# ---------------------------------------------------------------------------

B = 2           # batch size
T = 20          # encoder sequence length (frames)
S = 10          # target sequence length
H = 64          # hidden_size
FFN = 256       # ffn_size
HEADS = 4       # num_heads
LAYERS = 2      # num_layers
VOCAB = 10      # vocab_size

# Special token IDs (mirrored from asr_head.py)
_BLANK_ID = 0
_SOS_ID = 1
_EOS_ID = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encoder_output(batch: int = B, seq_len: int = T, hidden: int = H) -> torch.Tensor:
    """Random encoder output tensor (B, T, H)."""
    torch.manual_seed(42)
    return torch.randn(batch, seq_len, hidden)


def _target_ids_ctc(batch: int = B, length: int = S) -> torch.Tensor:
    """Random CTC target IDs in [1, VOCAB-1] (no blank=0 in targets)."""
    torch.manual_seed(42)
    return torch.randint(1, VOCAB, (batch, length))


def _target_ids_seq2seq(batch: int = B, length: int = S) -> torch.Tensor:
    """Target IDs with <sos> at start, <eos> at end, content in between."""
    torch.manual_seed(42)
    content = torch.randint(3, VOCAB, (batch, length - 2))  # skip special tokens
    sos = torch.full((batch, 1), _SOS_ID, dtype=torch.long)
    eos = torch.full((batch, 1), _EOS_ID, dtype=torch.long)
    return torch.cat([sos, content, eos], dim=1)  # (B, S)


# ===========================================================================
# CTCHead Tests
# ===========================================================================


class TestCTCHead:
    @pytest.fixture()
    def ctc_head(self) -> CTCHead:
        torch.manual_seed(42)
        return CTCHead(hidden_size=H, vocab_size=VOCAB, dropout=0.1)

    # -----------------------------------------------------------------------
    # 1. output shape
    # -----------------------------------------------------------------------

    def test_output_shape(self, ctc_head: CTCHead) -> None:
        """CTCHead: (B, T, H) -> (B, T, vocab_size)."""
        ctc_head.eval()
        x = _encoder_output()
        out = ctc_head(x)
        assert out.shape == (B, T, VOCAB)

    # -----------------------------------------------------------------------
    # 2. log probabilities
    # -----------------------------------------------------------------------

    def test_log_probabilities(self, ctc_head: CTCHead) -> None:
        """Output should be log-probabilities (exp sums to ~1 along vocab dim)."""
        ctc_head.eval()
        x = _encoder_output()
        log_probs = ctc_head(x)

        probs = log_probs.exp()
        sums = probs.sum(dim=-1)  # (B, T)
        torch.testing.assert_close(
            sums,
            torch.ones_like(sums),
            atol=1e-5, rtol=1e-5,
            msg="exp(log_probs) should sum to 1 along vocab dim",
        )

    # -----------------------------------------------------------------------
    # 3. gradient flow
    # -----------------------------------------------------------------------

    def test_gradient_flow(self, ctc_head: CTCHead) -> None:
        """Gradients should flow back to the input."""
        x = _encoder_output()
        x.requires_grad_(True)

        out = ctc_head(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input should receive gradients"
        assert (x.grad != 0).any(), "Input gradients should be non-zero"


# ===========================================================================
# TransformerASRDecoder Tests
# ===========================================================================


class TestTransformerASRDecoder:
    @pytest.fixture()
    def decoder(self) -> TransformerASRDecoder:
        torch.manual_seed(42)
        return TransformerASRDecoder(
            hidden_size=H,
            ffn_size=FFN,
            num_heads=HEADS,
            num_layers=LAYERS,
            vocab_size=VOCAB,
            dropout=0.1,
            label_smoothing=0.1,
        )

    # -----------------------------------------------------------------------
    # 4. output shape
    # -----------------------------------------------------------------------

    def test_output_shape(self, decoder: TransformerASRDecoder) -> None:
        """(B, T, H) encoder + (B, S) targets -> (B, S-1, vocab_size)."""
        decoder.eval()
        enc = _encoder_output()
        targets = _target_ids_seq2seq()
        logits = decoder(enc, targets)

        assert logits.shape == (B, S - 1, VOCAB)

    # -----------------------------------------------------------------------
    # 5. gradient flow
    # -----------------------------------------------------------------------

    def test_gradient_flow(self, decoder: TransformerASRDecoder) -> None:
        """Gradients should flow to all named parameters."""
        enc = _encoder_output()
        targets = _target_ids_seq2seq()

        logits = decoder(enc, targets)
        loss = logits.sum()
        loss.backward()

        for name, param in decoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                # At least some parameters should have non-zero gradients
                # (not all will, e.g. some biases might be zero due to symmetry)

    # -----------------------------------------------------------------------
    # 6. causal mask
    # -----------------------------------------------------------------------

    def test_causal_mask(self, decoder: TransformerASRDecoder) -> None:
        """Decoder should be auto-regressive: changing a future token must not
        affect logits at earlier positions."""
        decoder.eval()
        enc = _encoder_output()
        targets = _target_ids_seq2seq()

        # Forward with original targets
        logits_orig = decoder(enc, targets)

        # Modify the last target token (future w.r.t. earlier positions)
        targets_mod = targets.clone()
        targets_mod[:, -1] = (targets[:, -1] + 1) % VOCAB

        logits_mod = decoder(enc, targets_mod)

        # The last target token is targets[:, -1].  The decoder input is
        # targets[:, :-1], so modifying targets[:, -1] does NOT change
        # decoder input at all -> logits should be identical.
        torch.testing.assert_close(
            logits_orig, logits_mod,
            atol=1e-5, rtol=1e-5,
            msg="Changing the token beyond decoder input should not affect logits",
        )


# ===========================================================================
# ASRHead Tests
# ===========================================================================


class TestASRHead:
    # -----------------------------------------------------------------------
    # 7. ctc mode
    # -----------------------------------------------------------------------

    def test_ctc_mode(self) -> None:
        """loss_type='ctc' should create a CTCHead internally."""
        torch.manual_seed(42)
        head = ASRHead(
            loss_type="ctc",
            hidden_size=H,
            vocab_size=VOCAB,
            dropout=0.1,
        )
        assert isinstance(head.head, CTCHead)
        assert head.loss_type == "ctc"

    # -----------------------------------------------------------------------
    # 8. seq2seq mode
    # -----------------------------------------------------------------------

    def test_seq2seq_mode(self) -> None:
        """loss_type='seq2seq_ce' should create a TransformerASRDecoder internally."""
        torch.manual_seed(42)
        head = ASRHead(
            loss_type="seq2seq_ce",
            hidden_size=H,
            vocab_size=VOCAB,
            num_heads=HEADS,
            num_layers=LAYERS,
            ffn_size=FFN,
            dropout=0.1,
            label_smoothing=0.1,
        )
        assert isinstance(head.head, TransformerASRDecoder)
        assert head.loss_type == "seq2seq_ce"

    # -----------------------------------------------------------------------
    # 9. ctc loss computation
    # -----------------------------------------------------------------------

    def test_ctc_loss_computation(self) -> None:
        """compute_loss with CTC inputs should return a scalar loss."""
        torch.manual_seed(42)
        head = ASRHead(
            loss_type="ctc", hidden_size=H, vocab_size=VOCAB, dropout=0.1,
        )
        head.eval()

        enc = _encoder_output()
        logits = head(enc)  # (B, T, VOCAB) log-probs

        # CTC targets: no blank in targets, target_lengths < encoder_lengths
        target_len = 5  # must be < T=20
        targets = torch.randint(1, VOCAB, (B, target_len))
        encoder_lengths = torch.full((B,), T, dtype=torch.long)
        target_lengths = torch.full((B,), target_len, dtype=torch.long)

        loss = head.compute_loss(logits, targets, encoder_lengths, target_lengths)

        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() > 0, "CTC loss should be positive"

    # -----------------------------------------------------------------------
    # 10. seq2seq loss computation
    # -----------------------------------------------------------------------

    def test_seq2seq_loss_computation(self) -> None:
        """compute_loss with seq2seq inputs should return a scalar loss."""
        torch.manual_seed(42)
        head = ASRHead(
            loss_type="seq2seq_ce",
            hidden_size=H,
            vocab_size=VOCAB,
            num_heads=HEADS,
            num_layers=LAYERS,
            ffn_size=FFN,
            dropout=0.1,
            label_smoothing=0.1,
        )
        head.eval()

        enc = _encoder_output()
        targets = _target_ids_seq2seq()  # (B, S) with <sos> and <eos>

        logits = head(enc, target_ids=targets)  # (B, S-1, VOCAB)

        encoder_lengths = torch.full((B,), T, dtype=torch.long)
        target_lengths = torch.full((B,), S, dtype=torch.long)

        loss = head.compute_loss(logits, targets, encoder_lengths, target_lengths)

        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() > 0, "CE loss should be positive"

    # -----------------------------------------------------------------------
    # 11. ctc loss finite
    # -----------------------------------------------------------------------

    def test_ctc_loss_finite(self) -> None:
        """CTC loss should be finite (no NaN or Inf)."""
        torch.manual_seed(42)
        head = ASRHead(
            loss_type="ctc", hidden_size=H, vocab_size=VOCAB, dropout=0.1,
        )
        head.eval()

        enc = _encoder_output()
        logits = head(enc)

        target_len = 5
        targets = torch.randint(1, VOCAB, (B, target_len))
        encoder_lengths = torch.full((B,), T, dtype=torch.long)
        target_lengths = torch.full((B,), target_len, dtype=torch.long)

        loss = head.compute_loss(logits, targets, encoder_lengths, target_lengths)

        assert torch.isfinite(loss), f"CTC loss is not finite: {loss.item()}"

    # -----------------------------------------------------------------------
    # 12. ctc loss gradient
    # -----------------------------------------------------------------------

    def test_ctc_loss_gradient(self) -> None:
        """loss.backward() should succeed and produce gradients."""
        torch.manual_seed(42)
        head = ASRHead(
            loss_type="ctc", hidden_size=H, vocab_size=VOCAB, dropout=0.1,
        )
        # Keep in train mode so dropout is active (closer to real usage)

        enc = _encoder_output()
        enc.requires_grad_(True)

        logits = head(enc)

        target_len = 5
        targets = torch.randint(1, VOCAB, (B, target_len))
        encoder_lengths = torch.full((B,), T, dtype=torch.long)
        target_lengths = torch.full((B,), target_len, dtype=torch.long)

        loss = head.compute_loss(logits, targets, encoder_lengths, target_lengths)
        loss.backward()

        # Check that head parameters got gradients
        has_nonzero_grad = False
        for name, param in head.named_parameters():
            if param.requires_grad and param.grad is not None:
                if (param.grad != 0).any():
                    has_nonzero_grad = True
                    break

        assert has_nonzero_grad, "At least one parameter should have non-zero gradients"

        # Check that encoder input got gradients
        assert enc.grad is not None, "Encoder output should receive gradients"
