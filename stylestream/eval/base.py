"""Base evaluator interface for StyleStream evaluation metrics.

All evaluators follow a consistent lifecycle:

1. ``__init__``: stores configuration (device, sample_rate) but does **not**
   load any model.
2. ``load()``: explicitly loads the evaluation model to ``self.device``.
   Also triggered automatically on first ``evaluate_pair`` call (lazy loading).
3. ``evaluate_pair(converted_audio, target_audio, source_text)`` -> ``EvalResult``:
   computes the metric for a single converted-target pair.
4. ``evaluate_batch(pairs)`` -> ``list[EvalResult]``: sequential by default,
   subclasses may override for batched GPU inference.
5. ``unload()``: frees GPU memory.

Context-manager protocol is supported::

    with SomeEvaluator(device="cuda") as evaluator:
        result = evaluator.evaluate_pair(conv, tgt)
    # model automatically unloaded here

Metric convention
-----------------
- **WER / CER**: ``lower_is_better``, unit ``"percent"``.
- **S-SIM, A-SIM, E-SIM**: ``higher_is_better``, unit ``"cosine_similarity"``.
- **UTMOS**: ``higher_is_better``, unit ``"mos"``.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Result from a single evaluation metric computation.

    Parameters
    ----------
    metric_name : str
        Short identifier for the metric (e.g. ``"wer"``, ``"s_sim"``).
    value : float
        Numeric metric value.
    direction : str
        ``"higher_is_better"`` or ``"lower_is_better"``.
    unit : str
        Human-readable unit (e.g. ``"percent"``, ``"cosine_similarity"``).
    metadata : dict
        Arbitrary extra information (e.g. transcript text for WER).
    """

    metric_name: str
    value: float
    direction: str  # "higher_is_better" or "lower_is_better"
    unit: str = ""
    metadata: dict = field(default_factory=dict)

    def is_better_than(self, other: float) -> bool:
        """Return ``True`` if ``self.value`` is better than *other*.

        Parameters
        ----------
        other : float
            Reference value to compare against.

        Returns
        -------
        bool
            ``True`` when this result improves on *other* according to
            :attr:`direction`.
        """
        if self.direction == "higher_is_better":
            return self.value > other
        return self.value < other


@dataclass
class PairResult:
    """Aggregated result for a single source-target evaluation pair.

    Parameters
    ----------
    source_id : str
        Identifier for the source utterance.
    target_id : str
        Identifier for the target / reference utterance.
    metrics : dict[str, float]
        Mapping of metric name -> value.
    metadata : dict[str, Any]
        Arbitrary extra information (e.g. file paths, durations).
    """

    source_id: str
    target_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BaseEvaluator
# ---------------------------------------------------------------------------


class BaseEvaluator(abc.ABC):
    """Abstract base class for all evaluation metric computers.

    Subclasses must implement:

    - :attr:`metric_name` (property) -- short metric identifier.
    - :attr:`direction` (property) -- ``"higher_is_better"`` or
      ``"lower_is_better"``.
    - :meth:`_load_model` -- internal model loading logic.
    - :meth:`evaluate_pair` -- compute metric for one pair.

    Parameters
    ----------
    device : str
        Device to run the evaluation model on.  Default ``"cuda"``.
    sample_rate : int
        Expected audio sample rate in Hz.  Default 16000.
    """

    def __init__(self, device: str = "cuda", sample_rate: int = 16000) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self._model: nn.Module | None = None
        self._loaded: bool = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # ------------------------------------------------------------------
    # Abstract properties
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def metric_name(self) -> str:
        """Short name for the metric (e.g. ``'wer'``, ``'s_sim'``)."""
        ...

    @property
    @abc.abstractmethod
    def direction(self) -> str:
        """``'higher_is_better'`` or ``'lower_is_better'``."""
        ...

    @property
    def unit(self) -> str:
        """Unit of measurement (e.g. ``'percent'``, ``'cosine_similarity'``)."""
        return ""

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether the evaluation model is currently loaded."""
        return self._loaded

    def load(self) -> None:
        """Load the evaluation model to :attr:`device`.  Idempotent."""
        if self._loaded:
            return
        self.logger.info("Loading %s evaluator model...", self.metric_name)
        self._load_model()
        self._loaded = True
        self.logger.info(
            "%s evaluator ready on %s.", self.metric_name, self.device
        )

    @abc.abstractmethod
    def _load_model(self) -> None:
        """Internal model loading (implemented by subclasses).

        After this method returns the model must be ready on
        ``self.device`` and ``self._model`` should reference it.
        """
        ...

    def unload(self) -> None:
        """Free model from GPU memory."""
        if not self._loaded:
            return
        self._unload_model()
        self._model = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("%s evaluator unloaded.", self.metric_name)

    def _unload_model(self) -> None:
        """Override for custom unload logic.

        Default implementation moves ``self._model`` to CPU if it is an
        ``nn.Module``.
        """
        if self._model is not None and isinstance(self._model, nn.Module):
            self._model.cpu()

    def _ensure_loaded(self) -> None:
        """Lazy-load the model if not already loaded."""
        if not self._loaded:
            self.load()

    # ------------------------------------------------------------------
    # Evaluation interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def evaluate_pair(
        self,
        converted_audio: torch.Tensor,
        target_audio: torch.Tensor | None = None,
        source_text: str | None = None,
    ) -> EvalResult:
        """Evaluate a single converted-target pair.

        Parameters
        ----------
        converted_audio : Tensor
            Converted waveform, shape ``(samples,)`` at
            :attr:`sample_rate`.
        target_audio : Tensor or None
            Target / reference waveform for similarity metrics.
            ``None`` is acceptable for metrics that do not require a
            reference (e.g. WER with ground-truth text, UTMOS).
        source_text : str or None
            Ground-truth transcript for WER / CER computation.

        Returns
        -------
        EvalResult
            Metric result for this pair.
        """
        ...

    def evaluate_batch(
        self,
        converted_audios: list[torch.Tensor],
        target_audios: list[torch.Tensor | None] | None = None,
        source_texts: list[str | None] | None = None,
    ) -> list[EvalResult]:
        """Evaluate a batch of pairs.

        Default implementation iterates sequentially.  Subclasses may
        override for batched GPU inference.

        Parameters
        ----------
        converted_audios : list[Tensor]
            List of converted waveforms.
        target_audios : list[Tensor | None] or None
            List of reference waveforms (same length as
            *converted_audios*), or ``None`` for metrics that do not
            need references.
        source_texts : list[str | None] or None
            Ground-truth transcripts for WER / CER.

        Returns
        -------
        list[EvalResult]
            One result per input pair.
        """
        self._ensure_loaded()
        results: list[EvalResult] = []
        n = len(converted_audios)
        targets = target_audios if target_audios is not None else [None] * n
        texts = source_texts if source_texts is not None else [None] * n
        for conv, tgt, txt in zip(converted_audios, targets, texts):
            results.append(self.evaluate_pair(conv, tgt, txt))
        return results

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> BaseEvaluator:
        self.load()
        return self

    def __exit__(self, *args: object) -> None:
        self.unload()


# ---------------------------------------------------------------------------
# SimilarityEvaluator
# ---------------------------------------------------------------------------


class SimilarityEvaluator(BaseEvaluator):
    """Base class for embedding-based similarity evaluators.

    Used for S-SIM (speaker), A-SIM (accent), and E-SIM (emotion).
    Subclasses only need to implement:

    - :meth:`_load_model` -- load the embedding model.
    - :meth:`extract_embedding` -- extract a fixed-length embedding from
      a waveform.

    The cosine-similarity computation and ``evaluate_pair`` logic are
    provided by this base class.
    """

    @property
    def direction(self) -> str:
        return "higher_is_better"

    @property
    def unit(self) -> str:
        return "cosine_similarity"

    @abc.abstractmethod
    def extract_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract a fixed-length embedding from a waveform.

        Parameters
        ----------
        audio : Tensor
            Waveform of shape ``(samples,)`` at :attr:`sample_rate`.

        Returns
        -------
        Tensor
            L2-normalised embedding of shape ``(D,)``.
        """
        ...

    def cosine_similarity(
        self, emb1: torch.Tensor, emb2: torch.Tensor
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Both embeddings are L2-normalised before computing the dot
        product, so the result lies in ``[-1, 1]``.

        Parameters
        ----------
        emb1 : Tensor
            First embedding (any shape; will be flattened).
        emb2 : Tensor
            Second embedding (any shape; will be flattened).

        Returns
        -------
        float
            Cosine similarity value.
        """
        emb1 = emb1.flatten().float()
        emb2 = emb2.flatten().float()
        # L2 normalise
        emb1 = emb1 / (emb1.norm() + 1e-8)
        emb2 = emb2 / (emb2.norm() + 1e-8)
        return float(torch.dot(emb1, emb2))

    def evaluate_pair(
        self,
        converted_audio: torch.Tensor,
        target_audio: torch.Tensor | None = None,
        source_text: str | None = None,
    ) -> EvalResult:
        """Compute embedding cosine similarity between converted and target.

        Parameters
        ----------
        converted_audio : Tensor
            Converted waveform, shape ``(samples,)``.
        target_audio : Tensor
            Target / reference waveform, shape ``(samples,)``.
            **Required** for similarity metrics.
        source_text : str or None
            Unused; accepted for interface compatibility.

        Returns
        -------
        EvalResult
            Cosine similarity result.

        Raises
        ------
        ValueError
            If *target_audio* is ``None``.
        """
        self._ensure_loaded()
        if target_audio is None:
            raise ValueError(
                f"{self.metric_name} requires target_audio for similarity "
                f"computation."
            )

        with torch.no_grad():
            emb_conv = self.extract_embedding(converted_audio)
            emb_tgt = self.extract_embedding(target_audio)

        sim = self.cosine_similarity(emb_conv, emb_tgt)
        return EvalResult(
            metric_name=self.metric_name,
            value=sim,
            direction=self.direction,
            unit=self.unit,
        )
