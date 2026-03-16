"""Whisper-based WER/CER evaluator for StyleStream.

Uses Whisper-large-v3 to transcribe converted speech and compute
Word Error Rate (WER) or Character Error Rate (CER) against the
source transcript.

Paper spec:
    - Whisper-large-v3 for all WER evaluations
    - English language, greedy decoding
    - Text normalization before jiwer computation

Usage::

    from stylestream.eval.whisper_evaluator import WhisperEvaluator

    evaluator = WhisperEvaluator(device="cuda")

    # Single pair
    result = evaluator.evaluate_pair(
        converted_audio=waveform,   # (samples,) @ 16kHz
        source_text="the quick brown fox",
    )
    print(result.value)  # WER in percent

    # Batch
    results = evaluator.evaluate_batch(
        converted_audios=[wav1, wav2],
        source_texts=["hello world", "foo bar"],
    )

    # Standalone transcription
    text = evaluator.transcribe(waveform)

    # CER mode
    cer_evaluator = WhisperEvaluator(use_cer=True)
"""

from __future__ import annotations

import logging
import re

import torch

from stylestream.eval.base import BaseEvaluator, EvalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Normalize text for WER/CER computation.

    Applies the following transformations:

    1. Lowercase
    2. Remove punctuation except apostrophes in contractions
       (e.g. *don't*, *it's* are preserved)
    3. Collapse consecutive whitespace to a single space
    4. Strip leading / trailing whitespace

    Parameters
    ----------
    text : str
        Raw text (e.g. from Whisper transcription or ground truth).

    Returns
    -------
    str
        Cleaned text suitable for ``jiwer`` comparison.
    """
    text = text.lower()
    # Keep apostrophes in contractions (e.g., don't, it's).
    # Remove all other punctuation by replacing with space.
    text = re.sub(r"[^\w\s']", " ", text)
    # Remove standalone apostrophes (not part of a contraction).
    text = re.sub(r"\s'|'\s", " ", text)
    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# WhisperEvaluator
# ---------------------------------------------------------------------------


class WhisperEvaluator(BaseEvaluator):
    """WER/CER evaluator using Whisper-large-v3.

    Transcribes converted audio with Whisper and computes Word Error Rate
    (or Character Error Rate) against a ground-truth transcript using
    ``jiwer``.  The Whisper model is loaded lazily on first use and can be
    explicitly released via :meth:`unload_model`.

    Parameters
    ----------
    device : str
        Device for model inference.  Default ``"cuda"``.
    sample_rate : int
        Expected audio sample rate.  Must be 16000 for Whisper.
    model_id : str
        HuggingFace model ID.  Default ``"openai/whisper-large-v3"``.
    use_cer : bool
        If *True*, compute CER instead of WER.  Default *False*.
    language : str
        Language for forced decoding.  Default ``"en"``.
    """

    def __init__(
        self,
        device: str = "cuda",
        sample_rate: int = 16000,
        model_id: str = "openai/whisper-large-v3",
        use_cer: bool = False,
        language: str = "en",
    ) -> None:
        super().__init__(device=device, sample_rate=sample_rate)
        self.model_id = model_id
        self.use_cer = use_cer
        self.language = language

        # Populated by _load_model; set to None so hasattr checks work.
        self._processor = None
        self._forced_decoder_ids = None

    # ------------------------------------------------------------------
    # BaseEvaluator interface
    # ------------------------------------------------------------------

    @property
    def metric_name(self) -> str:  # noqa: D401
        """Name of the primary metric (``"wer"`` or ``"cer"``)."""
        return "cer" if self.use_cer else "wer"

    @property
    def direction(self) -> str:  # noqa: D401
        """Optimisation direction -- lower error rate is better."""
        return "lower_is_better"

    @property
    def unit(self) -> str:  # noqa: D401
        """Metric unit."""
        return "percent"

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load Whisper model and processor from HuggingFace Hub.

        Uses ``float16`` on CUDA devices for faster inference and lower
        memory.  The processor and forced decoder IDs (language / task)
        are cached alongside the model.
        """
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        logger.info("Loading Whisper model '%s' on %s ...", self.model_id, self.device)

        self._processor = WhisperProcessor.from_pretrained(self.model_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
        ).to(self.device)
        self._model.eval()

        # Set forced decoder IDs for language and task.
        self._forced_decoder_ids = self._processor.get_decoder_prompt_ids(
            language=self.language, task="transcribe"
        )

        logger.info(
            "Whisper loaded: model=%s, device=%s, language=%s",
            self.model_id,
            self.device,
            self.language,
        )

    def _unload_model(self) -> None:
        """Release Whisper model, processor, and decoder IDs."""
        super()._unload_model()
        self._processor = None
        self._forced_decoder_ids = None

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(self, audio: torch.Tensor) -> str:
        """Transcribe a single waveform.

        Parameters
        ----------
        audio : Tensor
            Waveform of shape ``(samples,)`` at 16 kHz.

        Returns
        -------
        str
            Transcribed text.

        Raises
        ------
        ValueError
            If *audio* is empty (zero samples).
        """
        self._ensure_loaded()

        if audio.numel() == 0:
            return ""

        audio_np = audio.cpu().numpy()

        inputs = self._processor(
            audio_np,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(
            self.device, dtype=self._model.dtype
        )

        with torch.no_grad():
            predicted_ids = self._model.generate(
                input_features,
                forced_decoder_ids=self._forced_decoder_ids,
                max_new_tokens=448,
            )

        transcription = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        return transcription.strip()

    def transcribe_batch(self, audios: list[torch.Tensor]) -> list[str]:
        """Transcribe a batch of waveforms efficiently.

        All waveforms are padded to the same length and processed in a
        single forward pass through Whisper.

        Parameters
        ----------
        audios : list[Tensor]
            List of waveforms, each of shape ``(samples,)``.

        Returns
        -------
        list[str]
            List of transcriptions (one per input waveform).
        """
        self._ensure_loaded()

        if not audios:
            return []

        # Filter out empty tensors, keeping track of original indices.
        non_empty_indices: list[int] = []
        non_empty_arrays: list = []
        for i, a in enumerate(audios):
            if a.numel() > 0:
                non_empty_indices.append(i)
                non_empty_arrays.append(a.cpu().numpy())

        # If all audios are empty, return empty strings.
        if not non_empty_arrays:
            return [""] * len(audios)

        inputs = self._processor(
            non_empty_arrays,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs.input_features.to(
            self.device, dtype=self._model.dtype
        )

        with torch.no_grad():
            predicted_ids = self._model.generate(
                input_features,
                forced_decoder_ids=self._forced_decoder_ids,
                max_new_tokens=448,
            )

        decoded = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )

        # Reconstruct full results list, inserting empty strings for
        # originally-empty inputs.
        results: list[str] = [""] * len(audios)
        for idx, text in zip(non_empty_indices, decoded):
            results[idx] = text.strip()

        return results

    # ------------------------------------------------------------------
    # Error rate computation
    # ------------------------------------------------------------------

    def compute_error_rate(self, hypothesis: str, reference: str) -> float:
        """Compute WER or CER between hypothesis and reference.

        Both strings are normalised with :func:`normalize_text` before
        comparison.  Uses ``jiwer`` for the actual metric computation.

        Parameters
        ----------
        hypothesis : str
            Predicted transcription.
        reference : str
            Ground truth text.

        Returns
        -------
        float
            Error rate as a percentage (0--100+).  Can exceed 100 when
            there are more errors than reference words/characters.
        """
        import jiwer

        hyp = normalize_text(hypothesis)
        ref = normalize_text(reference)

        # Edge cases: empty reference / hypothesis.
        if not ref:
            return 0.0 if not hyp else 100.0

        if self.use_cer:
            return jiwer.cer(ref, hyp) * 100.0
        else:
            return jiwer.wer(ref, hyp) * 100.0

    # ------------------------------------------------------------------
    # Evaluation interface
    # ------------------------------------------------------------------

    def evaluate_pair(
        self,
        converted_audio: torch.Tensor,
        target_audio: torch.Tensor | None = None,
        source_text: str | None = None,
    ) -> EvalResult:
        """Evaluate WER/CER for a converted utterance.

        Transcribes *converted_audio* with Whisper, then computes the
        error rate against *source_text*.

        Parameters
        ----------
        converted_audio : Tensor
            Converted waveform, shape ``(samples,)`` at 16 kHz.
        target_audio : Tensor or None
            Not used for WER/CER (accepted for API compatibility).
        source_text : str or None
            Ground truth transcription.  **Required** -- raises
            :class:`ValueError` if not provided.

        Returns
        -------
        EvalResult
            Result with ``metric_name`` set to ``"wer"`` or ``"cer"``
            and ``value`` in percent.  ``metadata`` contains the raw
            transcription and reference text.

        Raises
        ------
        ValueError
            If *source_text* is not provided.
        """
        if source_text is None:
            raise ValueError(
                "WhisperEvaluator requires source_text for WER/CER computation."
            )

        transcription = self.transcribe(converted_audio)
        error_rate = self.compute_error_rate(transcription, source_text)

        return EvalResult(
            metric_name=self.metric_name,
            value=error_rate,
            direction=self.direction,
            unit=self.unit,
            metadata={"transcription": transcription, "reference": source_text},
        )

    def evaluate_batch(
        self,
        converted_audios: list[torch.Tensor],
        target_audios: list[torch.Tensor | None] | None = None,
        source_texts: list[str | None] | None = None,
    ) -> list[EvalResult]:
        """Batch WER/CER evaluation with efficient batch transcription.

        All *converted_audios* are transcribed in a single batched
        Whisper forward pass, then compared to the corresponding
        *source_texts*.

        Parameters
        ----------
        converted_audios : list[Tensor]
            List of converted waveforms, each shape ``(samples,)``.
        target_audios : list or None
            Not used for WER/CER (accepted for API compatibility).
        source_texts : list[str | None] or None
            List of ground-truth transcriptions.  **Required** -- every
            element must be a non-None string.

        Returns
        -------
        list[EvalResult]
            One :class:`EvalResult` per input pair.

        Raises
        ------
        ValueError
            If *source_texts* is None, or if any element is None.
        """
        if source_texts is None:
            raise ValueError(
                "WhisperEvaluator requires source_texts for batch evaluation."
            )

        if len(converted_audios) != len(source_texts):
            raise ValueError(
                f"Length mismatch: {len(converted_audios)} audios vs "
                f"{len(source_texts)} texts."
            )

        transcriptions = self.transcribe_batch(converted_audios)

        results: list[EvalResult] = []
        for trans, ref_text in zip(transcriptions, source_texts):
            if ref_text is None:
                raise ValueError(
                    "All source_texts must be provided for WER/CER."
                )
            error_rate = self.compute_error_rate(trans, ref_text)
            results.append(
                EvalResult(
                    metric_name=self.metric_name,
                    value=error_rate,
                    direction=self.direction,
                    unit=self.unit,
                    metadata={
                        "transcription": trans,
                        "reference": ref_text,
                    },
                )
            )

        return results
