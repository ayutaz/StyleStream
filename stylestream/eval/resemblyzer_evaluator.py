"""Resemblyzer-based speaker similarity (S-SIM) evaluator.

Computes speaker/timbre similarity between converted and target speech
using Resemblyzer d-vector embeddings (cosine similarity).

The Resemblyzer VoiceEncoder produces a 256-dimensional speaker embedding.
S-SIM is the cosine similarity between the converted and target embeddings.

Paper reference:
    - S-SIM: Speaker similarity using Resemblyzer
    - Same-speaker pairs should score > 0.85
    - Different-speaker pairs should score < 0.5
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from stylestream.eval.base import SimilarityEvaluator

logger = logging.getLogger(__name__)


class ResemblyzerEvaluator(SimilarityEvaluator):
    """Speaker similarity evaluator using Resemblyzer d-vectors.

    Wraps the Resemblyzer ``VoiceEncoder`` to extract 256-dim speaker
    embeddings, then delegates cosine-similarity computation to the
    parent :class:`SimilarityEvaluator`.

    Parameters
    ----------
    device : str
        Device string.  Resemblyzer supports ``"cpu"`` and ``"cuda"``.
        Passed through so the interface stays consistent with other
        evaluators.  Default ``"cuda"``.
    sample_rate : int
        Expected input sample rate in Hz.  Default 16000.
    """

    def __init__(
        self,
        device: str = "cuda",
        sample_rate: int = 16000,
    ) -> None:
        super().__init__(device=device, sample_rate=sample_rate)

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def metric_name(self) -> str:  # noqa: D401
        """Short metric identifier."""
        return "s_sim"

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load Resemblyzer VoiceEncoder."""
        from resemblyzer import VoiceEncoder

        # Resemblyzer accepts "cpu" or "cuda" as device strings.
        encoder_device = "cpu"
        if "cuda" in self.device and torch.cuda.is_available():
            encoder_device = "cuda"

        self._model = VoiceEncoder(device=encoder_device)
        self._model.eval()
        self.logger.info(
            "Resemblyzer VoiceEncoder loaded on %s.", encoder_device
        )

    def _unload_model(self) -> None:
        """Release the VoiceEncoder.

        VoiceEncoder is not a standard ``nn.Module`` with ``.cpu()``,
        so we simply drop the reference.
        """
        self._model = None

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def _to_numpy(self, audio: torch.Tensor) -> np.ndarray:
        """Convert an audio tensor to a float32 numpy array.

        Parameters
        ----------
        audio : Tensor
            Waveform of shape ``(samples,)`` at :attr:`sample_rate`.

        Returns
        -------
        np.ndarray
            Float32 numpy array suitable for Resemblyzer.
        """
        return audio.detach().cpu().numpy().astype(np.float32)

    def _resample_if_needed(self, wav: np.ndarray) -> np.ndarray:
        """Resample *wav* to 16 kHz if :attr:`sample_rate` differs.

        Resemblyzer internally expects 16 kHz audio.  Our default
        pipeline already operates at 16 kHz, so this is a defensive
        fallback.

        Parameters
        ----------
        wav : np.ndarray
            Waveform at :attr:`sample_rate`.

        Returns
        -------
        np.ndarray
            Waveform at 16 kHz.
        """
        if self.sample_rate == 16000:
            return wav

        # Prefer Resemblyzer's own preprocessing when available.
        try:
            from resemblyzer import preprocess_wav

            return preprocess_wav(wav, source_sr=self.sample_rate)
        except (ImportError, TypeError):
            pass

        # Fallback: torchaudio resample.
        import torchaudio

        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        wav_tensor = torchaudio.functional.resample(
            wav_tensor, self.sample_rate, 16000
        )
        return wav_tensor.squeeze(0).numpy()

    def extract_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract a 256-dim speaker embedding from a waveform.

        Parameters
        ----------
        audio : Tensor
            Waveform of shape ``(samples,)`` at :attr:`sample_rate`.

        Returns
        -------
        Tensor
            L2-normalised embedding of shape ``(256,)``.
        """
        self._ensure_loaded()

        wav = self._to_numpy(audio)
        wav = self._resample_if_needed(wav)

        # VoiceEncoder.embed_utterance returns a numpy array of shape (256,).
        embedding: np.ndarray = self._model.embed_utterance(wav)
        emb_tensor = torch.from_numpy(embedding).float()

        # L2 normalise
        emb_tensor = emb_tensor / (emb_tensor.norm() + 1e-8)
        return emb_tensor
