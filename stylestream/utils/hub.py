"""External model registry and download management for StyleStream.

Centralises all references to pre-trained models (HuggingFace Hub or pip
packages) so that every component resolves weights through a single API.

Usage::

    from stylestream.utils.hub import download_model, load_hubert

    path = download_model("hubert")
    model, extract_fn = load_hubert(device="cuda")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

# ===================================================================
# Model registry
# ===================================================================
# Each entry describes one external model used somewhere in the
# StyleStream pipeline.  ``hf_id`` is a HuggingFace Hub model ID
# (or a pip package name when the model is not hosted on HF).
# ``stage`` marks whether it is needed at training, evaluation, or both.

MODEL_REGISTRY: dict[str, dict[str, str]] = {
    # ------ Training models ------
    "hubert": {
        "hf_id": "facebook/hubert-large-ls960-ft",
        "stage": "train",
        "description": "HuBERT-Large ASR - Destylizer input (layer 18)",
    },
    "wavlm": {
        "hf_id": "microsoft/wavlm-base-plus-sv",
        "stage": "train",
        "description": "WavLM-Base-Plus-SV - Style encoder backbone",
    },
    "vocos": {
        "hf_id": "charactr/vocos-mel-24khz",
        "stage": "train",
        "description": "Vocos - Vocoder warm start",
    },
    # ------ Evaluation models ------
    "whisper": {
        "hf_id": "openai/whisper-large-v3",
        "stage": "eval",
        "description": "Whisper-large-v3 - WER evaluation",
    },
    "resemblyzer": {
        "hf_id": "resemblyzer",  # pip package, not HF
        "stage": "eval",
        "description": "Resemblyzer - Speaker similarity (S-SIM)",
    },
    "accent_ecapa": {
        "hf_id": "Jzuluaga/accent-id-commonaccent-ecapa",
        "stage": "eval",
        "description": "Accent-ID ECAPA - Accent similarity (A-SIM)",
    },
    "emotion2vec": {
        "hf_id": "emotion2vec/emotion2vec_base",
        "stage": "eval",
        "description": "emotion2vec - Emotion similarity (E-SIM)",
    },
}

# ===================================================================
# Internal helpers
# ===================================================================

_DEFAULT_CACHE_DIR: str | None = None  # None -> HF default (~/.cache/huggingface/hub)


def _resolve_cache_dir(cache_dir: str | None) -> str | None:
    """Return *cache_dir* or the module-level default."""
    return cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR


def _validate_key(model_key: str) -> dict[str, str]:
    """Look up *model_key* in the registry or raise ``KeyError``."""
    if model_key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(
            f"Unknown model key '{model_key}'. Available: {available}"
        )
    return MODEL_REGISTRY[model_key]


# ===================================================================
# Download helpers
# ===================================================================

def download_model(model_key: str, cache_dir: str | None = None) -> Path:
    """Download a model and return its local cache path.

    Parameters
    ----------
    model_key:
        Key in :data:`MODEL_REGISTRY` (e.g. ``"hubert"``, ``"wavlm"``).
    cache_dir:
        Override for the HuggingFace cache directory.

    Returns
    -------
    Path
        Local directory containing the downloaded model files.

    Raises
    ------
    KeyError
        If *model_key* is not in the registry.
    RuntimeError
        If the model is a pip package (not hosted on HuggingFace Hub).
    """
    entry = _validate_key(model_key)
    hf_id = entry["hf_id"]

    # Models that are pip packages cannot be downloaded from HF Hub.
    if hf_id == "resemblyzer":
        raise RuntimeError(
            f"Model '{model_key}' is a pip package, not a HuggingFace model.  "
            "Install it with: pip install resemblyzer"
        )

    from huggingface_hub import snapshot_download

    resolved_cache = _resolve_cache_dir(cache_dir)
    logger.info("Downloading %s (%s) ...", model_key, hf_id)

    local_dir = snapshot_download(
        repo_id=hf_id,
        cache_dir=resolved_cache,
    )
    logger.info("Downloaded %s -> %s", model_key, local_dir)
    return Path(local_dir)


def download_all(
    stage: str = "all",
    cache_dir: str | None = None,
) -> dict[str, Path]:
    """Download all models for a given *stage*.

    Parameters
    ----------
    stage:
        One of ``"train"``, ``"eval"``, or ``"all"``.
    cache_dir:
        Override for the HuggingFace cache directory.

    Returns
    -------
    dict[str, Path]
        Mapping from model key to local cache path.  Models that are pip
        packages are skipped with a warning.
    """
    if stage not in ("train", "eval", "all"):
        raise ValueError(f"stage must be 'train', 'eval', or 'all', got '{stage}'")

    results: dict[str, Path] = {}
    for key, entry in MODEL_REGISTRY.items():
        if stage != "all" and entry["stage"] != stage:
            continue

        # Skip pip-only packages
        if entry["hf_id"] == "resemblyzer":
            logger.warning(
                "Skipping '%s' (pip package). Install with: pip install resemblyzer",
                key,
            )
            continue

        results[key] = download_model(key, cache_dir=cache_dir)

    return results


# ===================================================================
# Model loaders
# ===================================================================

def load_hubert(
    device: str | torch.device = "cpu",
    layer: int = 18,
) -> tuple[Any, Callable[[torch.Tensor], torch.Tensor]]:
    """Load HuBERT-Large and return ``(model, extract_fn)``.

    The returned *extract_fn* takes a waveform tensor of shape
    ``(batch, samples)`` and returns the hidden states at the requested
    layer with shape ``(batch, 768, T)`` at 50 Hz.

    Parameters
    ----------
    device:
        Device to place the model on.
    layer:
        Transformer layer whose hidden states to extract (0-indexed where
        index 0 is the CNN feature-extractor output, 1..N are the
        transformer layers).  Default **18** per the paper.

    Returns
    -------
    tuple[HubertModel, Callable]
        The frozen model and a convenience extraction function.
    """
    from transformers import HubertModel

    hf_id = MODEL_REGISTRY["hubert"]["hf_id"]
    logger.info("Loading HuBERT from %s (layer %d) ...", hf_id, layer)

    model = HubertModel.from_pretrained(hf_id)
    model = model.to(device).eval().half()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    @torch.inference_mode()
    def extract_fn(waveform: torch.Tensor) -> torch.Tensor:
        """Extract hidden states from *waveform*.

        Parameters
        ----------
        waveform:
            Shape ``(batch, samples)`` at 16 kHz.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 768, T)`` – hidden states at 50 Hz.
        """
        waveform = waveform.to(device=device, dtype=torch.float16)
        outputs = model(waveform, output_hidden_states=True)
        # outputs.hidden_states is a tuple of (num_layers + 1) tensors,
        # each of shape (batch, T, 768).
        # Index 0 = CNN output, 1..N = transformer layers.
        hidden = outputs.hidden_states[layer]  # (batch, T, 768)
        return hidden.transpose(1, 2)  # (batch, 768, T)

    return model, extract_fn


def load_wavlm(
    device: str | torch.device = "cpu",
) -> Any:
    """Load frozen WavLM-Base-Plus-SV.

    All parameters are frozen and the model is set to eval mode.

    Parameters
    ----------
    device:
        Device to place the model on.

    Returns
    -------
    WavLMModel
        The frozen model in eval mode.
    """
    from transformers import WavLMModel

    hf_id = MODEL_REGISTRY["wavlm"]["hf_id"]
    logger.info("Loading WavLM from %s ...", hf_id)

    model = WavLMModel.from_pretrained(hf_id)
    model = model.to(device).eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


# ===================================================================
# Inspection helpers
# ===================================================================

def verify_cache(cache_dir: str | None = None) -> dict[str, bool]:
    """Check which models are already cached locally.

    Parameters
    ----------
    cache_dir:
        Override for the HuggingFace cache directory.

    Returns
    -------
    dict[str, bool]
        Mapping from model key to whether its files exist in the cache.
    """
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.utils import LocalEntryNotFoundError

    resolved_cache = _resolve_cache_dir(cache_dir)
    status: dict[str, bool] = {}

    for key, entry in MODEL_REGISTRY.items():
        hf_id = entry["hf_id"]

        # pip-only packages: check import availability
        if hf_id == "resemblyzer":
            try:
                import importlib

                importlib.import_module("resemblyzer")
                status[key] = True
            except ImportError:
                status[key] = False
            continue

        # HuggingFace models: probe config.json in the cache
        try:
            result = try_to_load_from_cache(
                repo_id=hf_id,
                filename="config.json",
                cache_dir=resolved_cache,
            )
            # Returns a filepath string if cached, False if not,
            # or raises LocalEntryNotFoundError.
            status[key] = isinstance(result, str)
        except (LocalEntryNotFoundError, Exception):
            status[key] = False

    return status


def list_models(stage: str | None = None) -> list[dict[str, str]]:
    """List registered models, optionally filtered by stage.

    Parameters
    ----------
    stage:
        If given, only return models matching this stage
        (``"train"`` or ``"eval"``).

    Returns
    -------
    list[dict[str, str]]
        Each dict contains ``key``, ``hf_id``, ``stage``, and
        ``description``.
    """
    results: list[dict[str, str]] = []
    for key, entry in MODEL_REGISTRY.items():
        if stage is not None and entry["stage"] != stage:
            continue
        results.append({"key": key, **entry})
    return results
