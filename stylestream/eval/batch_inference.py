"""Batch inference for StyleStream evaluation.

Converts all source-target pairs using a trained StyleStream pipeline
and saves the converted waveforms to disk for metric computation.

Supports:
    - Offline (full-utterance) inference
    - Streaming (chunked) inference
    - Resume from interruption (skips already-converted files)
    - Progress logging
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import torchaudio

logger = logging.getLogger(__name__)


class BatchInference:
    """Batch inference pipeline for evaluation.

    Loads a trained StyleStream pipeline (Destylizer + Stylizer + Vocoder)
    and converts all pairs in an evaluation dataset.

    Parameters
    ----------
    destylizer_checkpoint : str
        Path to Destylizer checkpoint directory.
    stylizer_checkpoint : str
        Path to Stylizer checkpoint directory.
    vocoder_checkpoint : str
        Path to Vocoder checkpoint directory.
    output_dir : str
        Directory to save converted audio files.
    device : str
        Device for inference. Default "cuda".
    sample_rate : int
        Output sample rate. Default 16000.
    use_streaming : bool
        Use streaming inference mode. Default False.
    nfe : int
        Number of function evaluations for CFM. Default 16.
    cfg_strength : float
        CFG guidance strength. Default 2.0.
    """

    def __init__(
        self,
        destylizer_checkpoint: str,
        stylizer_checkpoint: str,
        vocoder_checkpoint: str,
        output_dir: str,
        device: str = "cuda",
        sample_rate: int = 16000,
        use_streaming: bool = False,
        nfe: int = 16,
        cfg_strength: float = 2.0,
    ) -> None:
        self.destylizer_checkpoint = destylizer_checkpoint
        self.stylizer_checkpoint = stylizer_checkpoint
        self.vocoder_checkpoint = vocoder_checkpoint
        self.output_dir = Path(output_dir)
        self.device = device
        self.sample_rate = sample_rate
        self.use_streaming = use_streaming
        self.nfe = nfe
        self.cfg_strength = cfg_strength

        self._pipeline = None
        self._destylizer = None
        self._stylizer = None
        self._vocoder = None
        self._loaded = False

    def load_pipeline(self) -> None:
        """Load the full StyleStream pipeline."""
        if self._loaded:
            return

        logger.info("Loading StyleStream pipeline...")

        if self.use_streaming:
            self._load_streaming_pipeline()
        else:
            self._load_offline_pipeline()

        self._loaded = True
        logger.info("Pipeline loaded on %s.", self.device)

    def _load_offline_pipeline(self) -> None:
        """Load offline (full-utterance) pipeline.

        Loads each component from its checkpoint directory using the
        CheckpointManager format (model.safetensors + optional config.yaml).
        """
        from stylestream.destylizer.feature_extractor import ContentFeatureExtractor
        from stylestream.stylizer.model import Stylizer
        from stylestream.vocoder.model import CausalVocos

        # Load Destylizer (feature extractor mode)
        self._destylizer = ContentFeatureExtractor.from_checkpoint(
            self.destylizer_checkpoint,
            device=self.device,
        )

        # Load Stylizer
        self._stylizer = self._load_model_from_checkpoint(
            Stylizer, self.stylizer_checkpoint,
        )
        self._stylizer.to(self.device).eval()

        # Load Vocoder
        self._vocoder = self._load_model_from_checkpoint(
            CausalVocos, self.vocoder_checkpoint,
        )
        self._vocoder.to(self.device).eval()

    def _load_model_from_checkpoint(
        self,
        model_cls: type,
        checkpoint_path: str,
    ) -> torch.nn.Module:
        """Load a model from a checkpoint directory or file.

        Handles CheckpointManager directory format (model.safetensors)
        and plain state-dict files (.pt, .bin, .safetensors).
        """
        path = Path(checkpoint_path)

        # Try to load config from checkpoint directory
        config = self._try_load_config(path, model_cls)

        # Build model with config or defaults
        if config is not None:
            model = model_cls.from_config(config)
        else:
            model = model_cls()

        # Load state dict
        state_dict = self._load_state_dict(path)
        model.load_state_dict(state_dict, strict=False)

        logger.info(
            "Loaded %s from %s (%d tensors)",
            model_cls.__name__,
            checkpoint_path,
            len(state_dict),
        )
        return model

    @staticmethod
    def _try_load_config(checkpoint_path: Path, model_cls: type):
        """Try to load a config from the checkpoint directory."""
        for candidate in [
            checkpoint_path / "config.yaml",
            checkpoint_path.parent / "config.yaml",
            checkpoint_path.parent.parent / "config.yaml",
        ]:
            if candidate.exists():
                try:
                    from omegaconf import OmegaConf
                    return OmegaConf.load(candidate)
                except Exception as exc:
                    logger.warning(
                        "Could not parse config from %s: %s",
                        candidate, exc,
                    )
        return None

    @staticmethod
    def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
        """Load model weights from a checkpoint path.

        Supports:
        - Directory with ``model.safetensors`` (CheckpointManager format)
        - Single ``.safetensors`` file
        - Single ``.pt`` / ``.bin`` file (torch.load)
        """
        # Case 1: directory containing model.safetensors
        if checkpoint_path.is_dir():
            safetensors_path = checkpoint_path / "model.safetensors"
            if safetensors_path.exists():
                from safetensors.torch import load_file
                return load_file(str(safetensors_path))

            # Fallback: look for a .pt file in the directory
            pt_files = list(checkpoint_path.glob("*.pt"))
            if pt_files:
                state = torch.load(
                    pt_files[0], map_location="cpu", weights_only=True
                )
                if isinstance(state, dict) and "model" in state:
                    return state["model"]
                return state

            raise FileNotFoundError(
                f"No model weights found in checkpoint directory: "
                f"{checkpoint_path}"
            )

        # Case 2: single safetensors file
        if checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file
            return load_file(str(checkpoint_path))

        # Case 3: single .pt / .bin file
        state = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=True
        )
        if isinstance(state, dict) and "model" in state:
            return state["model"]
        return state

    def _load_streaming_pipeline(self) -> None:
        """Load streaming (chunked) pipeline."""
        from stylestream.streaming.destylizer import StreamingDestylizer
        from stylestream.streaming.stylizer import StreamingStylizer
        from stylestream.streaming.pipeline import StreamingInferencePipeline
        from stylestream.vocoder.model import CausalVocos

        # Load streaming components
        destylizer = self._load_model_from_checkpoint(
            StreamingDestylizer, self.destylizer_checkpoint,
        )
        destylizer.to(self.device).eval()

        stylizer = self._load_model_from_checkpoint(
            StreamingStylizer, self.stylizer_checkpoint,
        )
        stylizer.to(self.device).eval()

        vocoder = self._load_model_from_checkpoint(
            CausalVocos, self.vocoder_checkpoint,
        )
        vocoder.to(self.device).eval()

        self._pipeline = StreamingInferencePipeline(
            destylizer=destylizer,
            stylizer=stylizer,
            vocoder=vocoder,
            nfe=self.nfe,
            cfg_strength=self.cfg_strength,
            device=self.device,
        )

    def _get_output_path(self, source_id: str, target_id: str) -> Path:
        """Get output path for a converted pair."""
        return self.output_dir / f"{source_id}__{target_id}.wav"

    def _is_already_converted(self, source_id: str, target_id: str) -> bool:
        """Check if this pair has already been converted (for resume)."""
        return self._get_output_path(source_id, target_id).exists()

    @torch.no_grad()
    def convert_pair(
        self,
        source_audio: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a single source-target pair.

        Parameters
        ----------
        source_audio : Tensor
            Source waveform shape (samples,) at 16kHz.
        target_audio : Tensor
            Target/reference waveform for style, shape (samples,) at 16kHz.

        Returns
        -------
        Tensor
            Converted waveform shape (samples,).
        """
        if not self._loaded:
            self.load_pipeline()

        if self.use_streaming:
            return self._convert_streaming(source_audio, target_audio)
        else:
            return self._convert_offline(source_audio, target_audio)

    def _convert_offline(
        self,
        source_audio: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> torch.Tensor:
        """Offline conversion: full utterance at once.

        Pipeline: Destylizer (content) -> Stylizer (mel generation) -> Vocoder (waveform)
        """
        from stylestream.utils.mel import MelSpectrogramTransform

        mel_transform = MelSpectrogramTransform().to(self.device)

        source = source_audio.to(self.device)
        target = target_audio.to(self.device)

        # Ensure (1, samples) for batched processing
        if source.dim() == 1:
            source = source.unsqueeze(0)
        if target.dim() == 1:
            target = target.unsqueeze(0)

        # 1. Extract content features from source via Destylizer.
        #    ContentFeatureExtractor.extract_from_hubert_features expects
        #    HuBERT features, but we need the full pipeline (HuBERT -> Conformer).
        #    Use the extract() method which takes an audio path, or use the
        #    internal _hubert_single + _destylizer_forward for tensor input.
        self._destylizer._ensure_hubert_loaded()
        hubert_feat = self._destylizer._hubert_single(
            source.squeeze(0).cpu()
        )  # (768, T)
        content_features = self._destylizer._destylizer_forward(
            hubert_feat.unsqueeze(0)
        )  # (1, T, 768)

        # 2. Generate mel using Stylizer with full inpainting (mask all frames).
        #    Stylizer.sample() handles context_mel=None as full generation.
        generated_mel = self._stylizer.sample(
            content_features=content_features,
            style_waveform=target,
            context_mel=None,  # full generation, no context
            mask=None,  # defaults to all-ones
            nfe=self.nfe,
            guidance_strength=self.cfg_strength,
        )  # (1, T, 100)

        # 3. Vocoder: mel -> waveform
        #    CausalVocos expects (B, n_mels, T) channels-first layout.
        generated_mel_ct = generated_mel.transpose(1, 2)  # (1, 100, T)
        waveform = self._vocoder(generated_mel_ct)  # (1, T_samples)

        return waveform.squeeze(0).cpu()

    def _convert_streaming(
        self,
        source_audio: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> torch.Tensor:
        """Streaming conversion: chunk by chunk.

        Uses StreamingInferencePipeline.convert_file() which returns
        a (waveform, stats) tuple.
        """
        waveform, _stats = self._pipeline.convert_file(
            source_waveform=source_audio,
            target_waveform=target_audio,
        )
        return waveform.cpu()

    def run(
        self,
        pairs: list,
        load_audio_fn=None,
    ) -> dict:
        """Run batch inference on all pairs.

        Parameters
        ----------
        pairs : list[EvalPair]
            Evaluation pairs to convert. Each pair must have
            ``source_id``, ``target_id``, ``source_path``, and
            ``target_path`` attributes.
        load_audio_fn : callable or None
            Function to load audio: path -> Tensor (samples,).
            Uses torchaudio if None.

        Returns
        -------
        dict
            Statistics with keys: total, converted, skipped, failed,
            time_seconds.
        """
        self.load_pipeline()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if load_audio_fn is None:
            load_audio_fn = self._default_load_audio

        stats = {
            "total": len(pairs),
            "converted": 0,
            "skipped": 0,
            "failed": 0,
        }
        start_time = time.time()

        for i, pair in enumerate(pairs):
            # Resume support: skip already converted
            if self._is_already_converted(pair.source_id, pair.target_id):
                stats["skipped"] += 1
                continue

            try:
                source = load_audio_fn(pair.source_path)
                target = load_audio_fn(pair.target_path)

                converted = self.convert_pair(source, target)

                output_path = self._get_output_path(
                    pair.source_id, pair.target_id
                )
                torchaudio.save(
                    str(output_path),
                    converted.unsqueeze(0),
                    self.sample_rate,
                )

                stats["converted"] += 1

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        "Progress: %d/%d (%.1f%%), elapsed: %.1fs",
                        i + 1,
                        len(pairs),
                        (i + 1) / len(pairs) * 100,
                        elapsed,
                    )
            except Exception as e:
                logger.error(
                    "Failed to convert %s -> %s: %s",
                    pair.source_id,
                    pair.target_id,
                    e,
                )
                stats["failed"] += 1

        stats["time_seconds"] = time.time() - start_time

        # Save stats
        stats_path = self.output_dir / "inference_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(
            "Batch inference complete: %d converted, %d skipped, "
            "%d failed in %.1fs",
            stats["converted"],
            stats["skipped"],
            stats["failed"],
            stats["time_seconds"],
        )
        return stats

    def _default_load_audio(self, path: str) -> torch.Tensor:
        """Default audio loader: loads, converts to mono, resamples.

        Returns
        -------
        Tensor
            Waveform shape (samples,) at self.sample_rate.
        """
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )
        return waveform.squeeze(0)
