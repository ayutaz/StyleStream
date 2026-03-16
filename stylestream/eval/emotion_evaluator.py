"""Emotion similarity (E-SIM) evaluator using emotion2vec.

Computes emotion similarity between converted and target speech using
emotion2vec embeddings (cosine similarity).

Paper reference:
    - E-SIM: Emotion similarity using emotion2vec
    - Target: E-SIM ~ 0.827 (offline), 0.803 (streaming)
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from stylestream.eval.base import SimilarityEvaluator

logger = logging.getLogger(__name__)


class EmotionEvaluator(SimilarityEvaluator):
    """Emotion similarity evaluator using emotion2vec.

    Uses emotion2vec_plus_large to extract emotion embeddings and
    compute cosine similarity between converted and target speech.

    Two backends are supported:

    1. **funasr** (official): Uses ``funasr.AutoModel`` with the ModelScope
       model ID.  This is the recommended backend.
    2. **transformers** (fallback): Uses HuggingFace ``AutoModel`` with
       mean-pooled hidden states when funasr is not installed.

    Parameters
    ----------
    device : str
        Device for model inference.
    sample_rate : int
        Expected audio sample rate. Default 16000.
    model_id : str
        Model identifier for emotion2vec.
    """

    def __init__(
        self,
        device: str = "cuda",
        sample_rate: int = 16000,
        model_id: str = "iic/emotion2vec_plus_large",
    ) -> None:
        super().__init__(device=device, sample_rate=sample_rate)
        self.model_id = model_id
        self._backend: str = ""
        self._feature_extractor = None

    @property
    def metric_name(self) -> str:
        return "e_sim"

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load emotion2vec model.

        Tries funasr first (official), falls back to transformers.
        """
        try:
            self._load_funasr()
            self._backend = "funasr"
        except (ImportError, Exception) as e:
            self.logger.warning(
                "funasr not available (%s), falling back to transformers.", e
            )
            self._load_transformers()
            self._backend = "transformers"

    def _load_funasr(self) -> None:
        """Load via the official funasr package."""
        from funasr import AutoModel as FunASRAutoModel

        self._model = FunASRAutoModel(
            model=self.model_id,
            device=self.device,
        )

    def _load_transformers(self) -> None:
        """Load via HuggingFace transformers (fallback)."""
        from transformers import AutoModel as HFAutoModel
        from transformers import Wav2Vec2FeatureExtractor

        hf_model_id = self.model_id
        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            hf_model_id
        )
        self._model = HFAutoModel.from_pretrained(hf_model_id).to(self.device)
        self._model.eval()

    def _unload_model(self) -> None:
        """Release model and feature extractor.

        For the funasr backend the model object does not follow the
        standard ``nn.Module`` pattern, so we simply discard it.
        For the transformers backend we move the model to CPU first.
        """
        if self._backend == "transformers" and self._model is not None:
            self._model.cpu()
        self._model = None
        self._feature_extractor = None

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def _extract_funasr(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract embedding using funasr backend.

        Parameters
        ----------
        audio : Tensor
            Waveform shape ``(samples,)`` on any device.

        Returns
        -------
        Tensor
            Raw (un-normalised) embedding on CPU.

        Raises
        ------
        RuntimeError
            If the model did not return embeddings.
        """
        wav = audio.cpu().numpy().astype(np.float32)

        res = self._model.generate(
            input=wav,
            granularity="utterance",
            extract_embedding=True,
        )

        # Result: list of dicts, each with 'feats' key
        if isinstance(res, list) and len(res) > 0:
            feats = res[0].get("feats", None)
            if feats is not None:
                if isinstance(feats, np.ndarray):
                    return torch.from_numpy(feats).float()
                return feats.float()

        raise RuntimeError("emotion2vec (funasr) did not return embeddings.")

    def _extract_transformers(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract embedding using transformers backend.

        Uses the last hidden state mean-pooled over the time axis as the
        utterance-level emotion embedding.

        Parameters
        ----------
        audio : Tensor
            Waveform shape ``(samples,)`` on any device.

        Returns
        -------
        Tensor
            Raw (un-normalised) embedding on CPU.
        """
        wav = audio.cpu().numpy().astype(np.float32)

        inputs = self._feature_extractor(
            wav,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self._model(input_values, output_hidden_states=True)
            # Use last hidden state, mean pool over time
            hidden = outputs.last_hidden_state  # (1, T, D)
            embedding = hidden.mean(dim=1).squeeze(0)  # (D,)

        return embedding.cpu().float()

    def extract_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract emotion embedding from waveform.

        Parameters
        ----------
        audio : Tensor
            Waveform shape ``(samples,)`` at :attr:`sample_rate`.

        Returns
        -------
        Tensor
            L2-normalized emotion embedding on CPU.
        """
        self._ensure_loaded()

        if self._backend == "funasr":
            embedding = self._extract_funasr(audio)
        else:
            embedding = self._extract_transformers(audio)

        # L2 normalize
        embedding = embedding.flatten()
        embedding = embedding / (embedding.norm() + 1e-8)
        return embedding
