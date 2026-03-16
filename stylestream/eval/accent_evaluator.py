"""Accent similarity (A-SIM) evaluator using ECAPA-TDNN.

Computes accent similarity between converted and target speech using
the accent-id-commonaccent-ecapa model from SpeechBrain Hub.

A-SIM is the cosine similarity between accent embeddings.

Paper reference:
    - A-SIM: Accent similarity using accent-id ECAPA-TDNN
    - Target: A-SIM ≈ 0.640 (offline), 0.635 (streaming)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from stylestream.eval.base import SimilarityEvaluator

logger = logging.getLogger(__name__)


class AccentEvaluator(SimilarityEvaluator):
    """Accent similarity evaluator using SpeechBrain ECAPA-TDNN.

    Uses the accent-id-commonaccent-ecapa model to extract accent
    embeddings and compute cosine similarity.

    Parameters
    ----------
    device : str
        Device for model inference.
    sample_rate : int
        Expected audio sample rate. Default 16000.
    model_id : str
        SpeechBrain Hub model ID.
    cache_dir : str or None
        Directory to cache the downloaded model. None uses default.
    """

    def __init__(
        self,
        device: str = "cuda",
        sample_rate: int = 16000,
        model_id: str = "Jzuluaga/accent-id-commonaccent-ecapa",
        cache_dir: str | None = None,
    ) -> None:
        super().__init__(device=device, sample_rate=sample_rate)
        self.model_id = model_id
        self.cache_dir = cache_dir or str(
            Path.home() / ".cache" / "speechbrain" / "accent_ecapa"
        )

    @property
    def metric_name(self) -> str:
        return "a_sim"

    def _load_model(self) -> None:
        """Load SpeechBrain EncoderClassifier from HuggingFace Hub."""
        from speechbrain.inference.classifiers import EncoderClassifier

        self._model = EncoderClassifier.from_hparams(
            source=self.model_id,
            savedir=self.cache_dir,
            run_opts={"device": self.device},
        )

    def _unload_model(self) -> None:
        """Release SpeechBrain model.

        SpeechBrain models don't follow the standard ``nn.Module.cpu()``
        pattern reliably, so we simply discard the reference and let the
        garbage collector free the memory.
        """
        self._model = None

    def extract_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract accent embedding from waveform.

        Parameters
        ----------
        audio : Tensor
            Waveform shape ``(samples,)`` at :attr:`sample_rate`.

        Returns
        -------
        Tensor
            L2-normalized accent embedding on CPU.
        """
        self._ensure_loaded()

        # SpeechBrain expects (batch, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)

        with torch.no_grad():
            # encode_batch returns (batch, 1, embed_dim)
            embeddings = self._model.encode_batch(audio)
            # Squeeze to (embed_dim,)
            embedding = embeddings.squeeze(0).squeeze(0)

        # L2 normalize
        embedding = embedding / (embedding.norm() + 1e-8)
        return embedding.cpu()
