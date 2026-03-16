"""UTMOS-based MOS prediction evaluator for StyleStream.

Predicts Mean Opinion Score (MOS) for speech quality using the
UTMOS model. Does not require target/reference audio.

UTMOS predicts a score from 1.0 (worst) to 5.0 (best).

Paper reference:
    - UTMOS used for speech quality evaluation in ablation studies
    - Model: sarulab-speech/UTMOS-demo (UTMOS22 strong learner)
"""

from __future__ import annotations

import logging

import torch

from stylestream.eval.base import BaseEvaluator, EvalResult

logger = logging.getLogger(__name__)


class UTMOSEvaluator(BaseEvaluator):
    """Speech quality evaluator using UTMOS MOS prediction.

    Parameters
    ----------
    device : str
        Device for model inference.
    sample_rate : int
        Expected audio sample rate. Default 16000.
    """

    def __init__(
        self,
        device: str = "cuda",
        sample_rate: int = 16000,
    ) -> None:
        super().__init__(device=device, sample_rate=sample_rate)

    @property
    def metric_name(self) -> str:
        return "utmos"

    @property
    def direction(self) -> str:
        return "higher_is_better"

    @property
    def unit(self) -> str:
        return "mos"

    def _load_model(self) -> None:
        """Load UTMOS model via torch.hub."""
        self._model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True,
        ).to(self.device)
        self._model.eval()

    def predict_mos(self, audio: torch.Tensor) -> float:
        """Predict MOS for a single waveform.

        Parameters
        ----------
        audio : Tensor
            Waveform shape (samples,) at 16kHz.

        Returns
        -------
        float
            Predicted MOS score (1.0-5.0).
        """
        self._ensure_loaded()

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (1, samples)
        audio = audio.to(self.device)

        with torch.no_grad():
            score = self._model(audio, self.sample_rate)

        # score may be tensor or float
        if isinstance(score, torch.Tensor):
            score = score.item()

        # Clamp to valid MOS range
        return max(1.0, min(5.0, score))

    def evaluate_pair(
        self,
        converted_audio: torch.Tensor,
        target_audio: torch.Tensor | None = None,
        source_text: str | None = None,
    ) -> EvalResult:
        """Evaluate MOS for a converted utterance.

        Parameters
        ----------
        converted_audio : Tensor
            Converted waveform, shape (samples,).
        target_audio : Tensor or None
            Not used for UTMOS.
        source_text : str or None
            Not used for UTMOS.

        Returns
        -------
        EvalResult
            Predicted MOS score.
        """
        mos = self.predict_mos(converted_audio)
        return EvalResult(
            metric_name=self.metric_name,
            value=mos,
            direction=self.direction,
            unit=self.unit,
        )

    def evaluate_batch(
        self,
        converted_audios: list[torch.Tensor],
        target_audios: list[torch.Tensor | None] | None = None,
        source_texts: list[str | None] | None = None,
    ) -> list[EvalResult]:
        """Batch MOS evaluation."""
        self._ensure_loaded()
        results = []
        for audio in converted_audios:
            results.append(self.evaluate_pair(audio))
        return results
