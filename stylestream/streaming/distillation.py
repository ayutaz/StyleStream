"""MSE distillation trainer for the StyleStream streaming Destylizer.

Trains the streaming Destylizer (student) to match the offline
Destylizer's (teacher) content features via MSE loss.

Training spec (paper):
    - Teacher: offline Destylizer (frozen)
    - Student: streaming Destylizer (causal HuBERT + causal Conformer)
    - Loss: MSE(fc_streaming, fc_offline.detach())
    - Optional: auxiliary ASR loss (lambda * L_ASR)
    - HuBERT LR: 1/10 of main LR (prevent catastrophic forgetting)
    - 100k steps, batch 32, 8 GPUs
    - Dataset: LMG (~1300 hours)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from stylestream.data.manifest import Manifest, Utterance
from stylestream.streaming.destylizer import StreamingDestylizer
from stylestream.training.trainer import BaseTrainer
from stylestream.utils.audio import load_audio

logger = logging.getLogger(__name__)

# Feature rate (50 Hz) and sample rate (16 kHz).
_FEATURE_RATE: int = 50
_SAMPLE_RATE: int = 16_000
_HOP_LENGTH: int = 320
_HUBERT_DIM: int = 768
_CONTENT_DIM: int = 768


# ======================================================================
# Dataset
# ======================================================================


class DistillationDataset(Dataset):
    """Dataset for MSE distillation: provides raw audio + HuBERT features.

    Each item returns a dict with:

    - ``waveform`` -- ``(T_samples,)`` raw 16 kHz audio for the student.
    - ``hubert_features`` -- ``(768, T_frames)`` pre-extracted HuBERT
      layer-18 features for the teacher (stored channels-first on disk).
    - ``feature_length`` -- int, number of frames.
    - ``waveform_length`` -- int, number of audio samples.

    The student (StreamingDestylizer) takes raw waveforms and runs its
    own causal HuBERT internally.  The teacher (offline Destylizer)
    takes pre-extracted HuBERT features.  Pre-extracting teacher inputs
    avoids running HuBERT twice per sample.

    Parameters
    ----------
    manifest :
        Utterance manifest.
    audio_dir :
        Root directory containing resampled 16 kHz audio files.
        Audio is located at ``audio_dir / utterance.audio_path``.
    hubert_features_dir :
        Root directory with pre-extracted HuBERT layer-18 ``.pt`` files.
        Layout: ``hubert_features_dir/hubert_l18/{dataset}/{subset}/{stem}.pt``
    max_frames :
        Skip utterances longer than this many frames (default 3000 = 60s).
    max_duration :
        Maximum utterance duration in seconds.  Utterances exceeding
        this are silently skipped.  Default 30.0.
    """

    def __init__(
        self,
        manifest: Manifest,
        audio_dir: str | Path,
        hubert_features_dir: str | Path,
        max_frames: int = 3000,
        max_duration: float = 30.0,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.hubert_features_dir = Path(hubert_features_dir)
        self.max_frames = max_frames

        # Filter utterances: keep only those with both audio and features.
        self.utterances: list[Utterance] = []
        self.audio_paths: list[Path] = []
        self.feature_paths: list[Path] = []
        n_skipped_length = 0
        n_skipped_missing_audio = 0
        n_skipped_missing_feat = 0

        for utt in manifest:
            # Skip overly long utterances.
            estimated_frames = int(utt.duration * _FEATURE_RATE)
            if estimated_frames > max_frames or utt.duration > max_duration:
                n_skipped_length += 1
                continue

            audio_path = self.audio_dir / utt.audio_path
            feat_path = self._feature_path(utt)

            if not audio_path.exists():
                n_skipped_missing_audio += 1
                continue
            if not feat_path.exists():
                n_skipped_missing_feat += 1
                continue

            self.utterances.append(utt)
            self.audio_paths.append(audio_path)
            self.feature_paths.append(feat_path)

        # Cache estimated lengths for the bucket sampler.
        self._estimated_lengths: list[int] = [
            max(1, int(u.duration * _FEATURE_RATE)) for u in self.utterances
        ]

        logger.info(
            "DistillationDataset: %d utterances kept "
            "(skipped: %d too long, %d missing audio, %d missing features)",
            len(self.utterances),
            n_skipped_length,
            n_skipped_missing_audio,
            n_skipped_missing_feat,
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _feature_path(self, utt: Utterance) -> Path:
        """Resolve the on-disk path to the HuBERT feature file."""
        return (
            self.hubert_features_dir
            / "hubert_l18"
            / utt.dataset
            / utt.subset
            / f"{utt.stem}.pt"
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> dict:
        """Return a single training sample.

        Keys
        ----
        waveform         : Tensor (T_samples,)     -- raw 16kHz audio
        hubert_features  : Tensor (768, T_frames)   -- pre-extracted for teacher
        feature_length   : int                       -- actual T_frames
        waveform_length  : int                       -- actual T_samples
        """
        audio_path = self.audio_paths[idx]
        feat_path = self.feature_paths[idx]

        # --- Raw audio for student ---
        waveform = load_audio(audio_path, sr=_SAMPLE_RATE)  # (T_samples,)

        # --- Pre-extracted HuBERT features for teacher ---
        features: torch.Tensor = torch.load(
            feat_path, map_location="cpu", weights_only=True,
        )
        # Normalise to (768, T) layout.
        if features.dim() == 2 and features.shape[0] != _HUBERT_DIM and features.shape[1] == _HUBERT_DIM:
            features = features.t()  # (T, 768) -> (768, T)

        feature_length = features.shape[-1]
        waveform_length = waveform.shape[0]

        return {
            "waveform": waveform,
            "hubert_features": features,
            "feature_length": feature_length,
            "waveform_length": waveform_length,
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def estimated_lengths(self) -> list[int]:
        """Per-utterance estimated frame counts (for bucket sampling)."""
        return self._estimated_lengths


# ======================================================================
# Collator
# ======================================================================


class DistillationCollator:
    """Collate variable-length distillation samples into a padded batch.

    Waveforms are zero-padded along the sample axis.  HuBERT features
    are zero-padded along the time axis.  Padding masks are computed
    for the feature-level positions (50 Hz).

    Returns
    -------
    dict
        waveform             : (B, T_max_samples)  zero-padded
        hubert_features      : (B, 768, T_max)     zero-padded
        feature_lengths      : (B,)
        waveform_lengths     : (B,)
        feature_padding_mask : (B, T_max)           True = padded
    """

    def __call__(self, batch: list[dict]) -> dict:
        feat_lengths = torch.tensor(
            [item["feature_length"] for item in batch], dtype=torch.long,
        )
        wave_lengths = torch.tensor(
            [item["waveform_length"] for item in batch], dtype=torch.long,
        )

        t_max = int(feat_lengths.max().item())
        w_max = int(wave_lengths.max().item())
        bsz = len(batch)

        # --- Pad waveforms ---
        padded_waveforms = torch.zeros(bsz, w_max, dtype=torch.float32)
        for i, item in enumerate(batch):
            w = item["waveform_length"]
            padded_waveforms[i, :w] = item["waveform"]

        # --- Pad HuBERT features ---
        padded_features = torch.zeros(bsz, _HUBERT_DIM, t_max, dtype=torch.float32)
        for i, item in enumerate(batch):
            t = item["feature_length"]
            padded_features[i, :, :t] = item["hubert_features"]

        # --- Feature-level padding mask (True = padded) ---
        arange = torch.arange(t_max).unsqueeze(0)  # (1, T_max)
        feature_padding_mask = arange >= feat_lengths.unsqueeze(1)  # (B, T_max)

        return {
            "waveform": padded_waveforms,
            "hubert_features": padded_features,
            "feature_lengths": feat_lengths,
            "waveform_lengths": wave_lengths,
            "feature_padding_mask": feature_padding_mask,
        }


# ======================================================================
# Bucket batch sampler
# ======================================================================


class DistillationBucketSampler(Sampler[list[int]]):
    """Group utterances by length for efficient batching.

    Same algorithm as :class:`~stylestream.data.destylizer_dataset.BucketBatchSampler`:
    sort by length, partition into consecutive buckets of *batch_size*,
    optionally shuffle bucket order.

    Parameters
    ----------
    lengths :
        Per-utterance estimated frame counts.
    batch_size :
        Utterances per batch.
    drop_last :
        Drop the last incomplete bucket.
    shuffle :
        Shuffle bucket order each epoch.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.buckets: list[list[int]] = []
        for start in range(0, len(sorted_indices), batch_size):
            bucket = sorted_indices[start : start + batch_size]
            if self.drop_last and len(bucket) < batch_size:
                continue
            self.buckets.append(bucket)

    def __iter__(self) -> Iterator[list[int]]:
        bucket_order = list(range(len(self.buckets)))
        if self.shuffle:
            random.shuffle(bucket_order)
        for idx in bucket_order:
            yield self.buckets[idx]

    def __len__(self) -> int:
        return len(self.buckets)


# ======================================================================
# DataLoader builder
# ======================================================================


def build_distillation_dataloader(
    manifest: Manifest,
    audio_dir: str | Path,
    hubert_features_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    max_frames: int = 3000,
    max_duration: float = 30.0,
) -> DataLoader:
    """Build a :class:`DataLoader` for distillation training.

    Uses :class:`DistillationBucketSampler` to group similar-length
    utterances, and :class:`DistillationCollator` to pad each batch.

    Parameters
    ----------
    manifest :
        Utterance manifest.
    audio_dir :
        Root directory with resampled 16 kHz audio.
    hubert_features_dir :
        Root directory with pre-extracted HuBERT ``.pt`` files.
    batch_size :
        Utterances per batch (default 32).
    num_workers :
        DataLoader worker processes.
    shuffle :
        Shuffle bucket order.
    max_frames :
        Skip utterances exceeding this frame count.
    max_duration :
        Skip utterances exceeding this duration in seconds.

    Returns
    -------
    DataLoader
        Yields dicts with ``waveform``, ``hubert_features``,
        ``feature_lengths``, ``waveform_lengths``, and
        ``feature_padding_mask``.
    """
    dataset = DistillationDataset(
        manifest=manifest,
        audio_dir=audio_dir,
        hubert_features_dir=hubert_features_dir,
        max_frames=max_frames,
        max_duration=max_duration,
    )

    sampler = DistillationBucketSampler(
        lengths=dataset.estimated_lengths,
        batch_size=batch_size,
        drop_last=False,
        shuffle=shuffle,
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=DistillationCollator(),
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


# ======================================================================
# Distillation trainer
# ======================================================================


class DistillationTrainer(BaseTrainer):
    """MSE distillation trainer for the streaming Destylizer.

    Trains the streaming Destylizer (student) to match the offline
    Destylizer's (teacher) content feature output via MSE loss.

    The teacher model is frozen and produces target features.
    The student uses chunked causal attention and causal CNN.

    Config structure::

        config.training.steps = 100000
        config.training.batch_size = 32
        config.training.peak_lr = 1e-4
        config.training.warmup_steps = 4000

        config.distillation.teacher_checkpoint = "checkpoints/destylizer/best"
        config.distillation.hubert_lr_scale = 0.1  # HuBERT gets 1/10 LR
        config.distillation.aux_asr_weight = 0.0    # optional ASR loss weight
        config.distillation.chunk_size = 30

        config.data.manifest_path = "data/manifests/lmg.csv"
        config.data.audio_dir = "data/processed/audio"
        config.data.hubert_features_dir = "data/processed/hubert"

    Parameters
    ----------
    config :
        OmegaConf DictConfig with training, distillation, and data sub-configs.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self._teacher: nn.Module | None = None

    # ------------------------------------------------------------------
    # Model (student)
    # ------------------------------------------------------------------

    def build_model(self) -> nn.Module:
        """Build the StreamingDestylizer (student).

        Optionally initialises weights from an offline Destylizer
        checkpoint via ``config.distillation.student_init_checkpoint``.

        Returns
        -------
        nn.Module
            StreamingDestylizer ready for distillation training.
        """
        distil_cfg = self.config.distillation

        # Conformer hyper-parameters (fall back to paper defaults).
        chunk_size = getattr(distil_cfg, "chunk_size", 30)
        hidden_size = getattr(distil_cfg, "hidden_size", _CONTENT_DIM)
        num_layers = getattr(distil_cfg, "num_layers", 6)
        ffn_size = getattr(distil_cfg, "ffn_size", 3072)
        num_heads = getattr(distil_cfg, "num_heads", 12)
        kernel_size = getattr(distil_cfg, "kernel_size", 31)
        hubert_model_id = getattr(
            distil_cfg, "hubert_model_id", "facebook/hubert-large-ls960-ft",
        )
        hubert_layer = getattr(distil_cfg, "hubert_layer", 18)

        student = StreamingDestylizer(
            chunk_size=chunk_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            ffn_size=ffn_size,
            num_heads=num_heads,
            kernel_size=kernel_size,
            hubert_model_id=hubert_model_id,
            hubert_layer=hubert_layer,
        )

        # Optionally warm-start from an offline Destylizer checkpoint.
        offline_ckpt = getattr(distil_cfg, "student_init_checkpoint", None)
        if offline_ckpt is not None:
            self.logger.info(
                "Initialising student from offline checkpoint: %s", offline_ckpt,
            )
            student.load_from_offline(offline_ckpt)

        n_params = sum(p.numel() for p in student.parameters())
        n_trainable = sum(
            p.numel() for p in student.parameters() if p.requires_grad
        )
        self.logger.info(
            "StreamingDestylizer (student) built: %s total params, %s trainable",
            f"{n_params:,}",
            f"{n_trainable:,}",
        )

        return student

    # ------------------------------------------------------------------
    # Teacher
    # ------------------------------------------------------------------

    def _build_teacher(self) -> nn.Module:
        """Load and freeze the offline Destylizer (teacher).

        The teacher checkpoint path is read from
        ``config.distillation.teacher_checkpoint``.  The model is set
        to eval mode with all parameters frozen.

        Returns
        -------
        nn.Module
            Frozen offline Destylizer.
        """
        from stylestream.destylizer.model import Destylizer

        teacher = Destylizer()  # default architecture (paper spec)

        teacher_ckpt = self.config.distillation.teacher_checkpoint
        self.logger.info("Loading teacher from: %s", teacher_ckpt)

        state = torch.load(
            str(teacher_ckpt), map_location="cpu", weights_only=False,
        )
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        teacher.load_state_dict(state)

        # Freeze teacher completely.
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        n_params = sum(p.numel() for p in teacher.parameters())
        self.logger.info("Teacher loaded and frozen: %s params", f"{n_params:,}")

        return teacher

    # ------------------------------------------------------------------
    # Optimizer (two param groups)
    # ------------------------------------------------------------------

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Build AdamW with separate LR for HuBERT parameters.

        HuBERT parameters receive ``peak_lr * hubert_lr_scale`` (default
        1/10) to prevent catastrophic forgetting of the pre-trained
        acoustic features.

        Parameters
        ----------
        model : nn.Module
            The StreamingDestylizer student model.

        Returns
        -------
        torch.optim.Optimizer
            AdamW with two parameter groups.
        """
        hubert_lr_scale = getattr(
            self.config.distillation, "hubert_lr_scale", 0.1,
        )
        peak_lr = self.config.training.peak_lr

        hubert_params: list[nn.Parameter] = []
        other_params: list[nn.Parameter] = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "hubert" in name:
                hubert_params.append(param)
            else:
                other_params.append(param)

        self.logger.info(
            "Optimizer: %d other params (lr=%.2e), %d HuBERT params (lr=%.2e)",
            len(other_params),
            peak_lr,
            len(hubert_params),
            peak_lr * hubert_lr_scale,
        )

        return torch.optim.AdamW(
            [
                {"params": other_params, "lr": peak_lr},
                {"params": hubert_params, "lr": peak_lr * hubert_lr_scale},
            ],
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def build_dataloader(self) -> DataLoader:
        """Build the training DataLoader with audio + HuBERT features.

        Reads ``config.data.manifest_path``, ``config.data.audio_dir``,
        and ``config.data.hubert_features_dir`` to construct a
        :func:`build_distillation_dataloader`.

        Returns
        -------
        DataLoader
            Yields dicts with ``waveform``, ``hubert_features``,
            ``feature_lengths``, ``waveform_lengths``, and
            ``feature_padding_mask``.
        """
        manifest_path = self.config.data.manifest_path
        audio_dir = self.config.data.audio_dir
        hubert_features_dir = self.config.data.hubert_features_dir
        batch_size = self.config.training.batch_size
        num_workers = getattr(self.config.data, "num_workers", 4)
        max_frames = getattr(self.config.data, "max_frames", 3000)
        max_duration = getattr(self.config.data, "max_duration", 30.0)

        manifest = Manifest.load(manifest_path)
        self.logger.info(
            "Training manifest loaded: %d utterances from %s",
            len(manifest),
            manifest_path,
        )

        return build_distillation_dataloader(
            manifest=manifest,
            audio_dir=audio_dir,
            hubert_features_dir=hubert_features_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            max_frames=max_frames,
            max_duration=max_duration,
        )

    def build_val_dataloader(self) -> DataLoader | None:
        """Build a validation DataLoader if ``val_manifest_path`` is set.

        Returns
        -------
        DataLoader or None
            *None* if no validation manifest is configured.
        """
        val_path = getattr(self.config.data, "val_manifest_path", None)
        if val_path is None:
            self.logger.info(
                "No val_manifest_path configured, skipping validation.",
            )
            return None

        audio_dir = self.config.data.audio_dir
        hubert_features_dir = self.config.data.hubert_features_dir
        batch_size = self.config.training.batch_size
        num_workers = getattr(self.config.data, "num_workers", 4)
        max_frames = getattr(self.config.data, "max_frames", 3000)
        max_duration = getattr(self.config.data, "max_duration", 30.0)

        manifest = Manifest.load(val_path)
        self.logger.info(
            "Validation manifest loaded: %d utterances from %s",
            len(manifest),
            val_path,
        )

        self._val_dataloader = build_distillation_dataloader(
            manifest=manifest,
            audio_dir=audio_dir,
            hubert_features_dir=hubert_features_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            max_frames=max_frames,
            max_duration=max_duration,
        )
        return self._val_dataloader

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        """Compute MSE distillation loss for a single batch.

        Steps:
            1. Run the student (StreamingDestylizer) on raw waveforms.
            2. Run the frozen teacher (offline Destylizer) on pre-extracted
               HuBERT features.
            3. Align output lengths and compute MSE loss.
            4. Optionally add auxiliary ASR loss.

        Parameters
        ----------
        batch : dict
            From :class:`DistillationCollator` with keys ``waveform``,
            ``hubert_features``, ``feature_lengths``, ``waveform_lengths``,
            ``feature_padding_mask``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``loss``               : scalar total loss.
            ``mse``                : MSE distillation loss (detached).
            ``cosine_similarity``  : mean cosine similarity (detached).
        """
        waveform = batch["waveform"]                      # (B, T_samples)
        hubert_features = batch["hubert_features"]        # (B, 768, T)
        padding_mask = batch["feature_padding_mask"]      # (B, T)

        # --- Student forward (streaming) ---
        student_output = self.model(waveform, padding_mask=None)
        fc_student = student_output["content_features"]  # (B, T_s, 768)

        # --- Teacher forward (frozen, offline) ---
        # Lazily build teacher on first call (after Accelerate has prepared
        # the student model, so we know which device to target).
        if self._teacher is None:
            self._teacher = self._build_teacher()
            self._teacher = self._teacher.to(self.accelerator.device)

        # Teacher expects (B, T, 768); dataset stores (B, 768, T).
        hubert_features_bt = hubert_features.transpose(1, 2)  # (B, T, 768)

        with torch.inference_mode():
            fc_teacher = self._teacher.extract_content_features(
                hubert_features_bt, padding_mask=padding_mask,
            )  # (B, T_t, 768)

        # --- Align temporal lengths ---
        # Student and teacher may produce slightly different lengths due
        # to HuBERT CNN boundary effects vs. pre-extracted features.
        min_t = min(fc_student.shape[1], fc_teacher.shape[1])
        fc_student = fc_student[:, :min_t]
        fc_teacher = fc_teacher[:, :min_t]

        # Build a valid-frame mask for the aligned length.  The original
        # padding mask may be longer, so truncate it too.
        if padding_mask is not None and padding_mask.shape[1] >= min_t:
            valid_mask = ~padding_mask[:, :min_t]  # True = valid
        else:
            valid_mask = torch.ones(
                fc_student.shape[0], min_t,
                dtype=torch.bool, device=fc_student.device,
            )

        # --- MSE loss (masked) ---
        # Compute MSE only on valid (non-padded) frames to avoid wasted
        # backward computation.  Previous approach computed diff**2 for
        # ALL frames then masked; this gathers valid frames first.
        diff = fc_student - fc_teacher.detach()
        valid_mask_f = valid_mask.float()  # compute once, reuse below
        valid_count = valid_mask.sum()
        if valid_count > 0:
            # Zero out padded frames so they contribute nothing to sum
            # or gradient, then normalise by valid element count.
            masked_diff = diff * valid_mask_f.unsqueeze(-1)  # (B, T, D)
            mse_loss = masked_diff.pow(2).sum() / (valid_count * diff.shape[-1])
        else:
            mse_loss = diff.pow(2).mean()

        total_loss = mse_loss

        # --- Optional auxiliary ASR loss ---
        aux_asr_weight = getattr(
            self.config.distillation, "aux_asr_weight", 0.0,
        )
        metrics: dict[str, torch.Tensor] = {
            "mse": mse_loss.detach(),
        }

        if aux_asr_weight > 0.0 and hasattr(batch, "token_ids"):
            # This path is only active if the dataset provides text
            # targets and the student model has an ASR head.
            unwrapped = self.accelerator.unwrap_model(self.model)
            if hasattr(unwrapped, "asr_head"):
                asr_loss = self._compute_aux_asr_loss(
                    fc_student, batch, valid_mask,
                )
                total_loss = mse_loss + aux_asr_weight * asr_loss
                metrics["asr_loss"] = asr_loss.detach()

        # --- Cosine similarity (for monitoring convergence) ---
        with torch.no_grad():
            flat_student = fc_student.reshape(-1, _CONTENT_DIM)
            flat_teacher = fc_teacher.reshape(-1, _CONTENT_DIM)
            cos_sim = F.cosine_similarity(
                flat_student, flat_teacher, dim=-1,
            ).mean()
            metrics["cosine_similarity"] = cos_sim

        metrics["loss"] = total_loss

        return metrics

    def _compute_aux_asr_loss(
        self,
        content_features: torch.Tensor,
        batch: dict,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute auxiliary ASR loss on student content features.

        This is only called when ``aux_asr_weight > 0`` and the
        batch contains ``token_ids``.  The student model must have an
        ``asr_head`` attribute (not standard for StreamingDestylizer,
        but can be added for regularisation).

        Parameters
        ----------
        content_features : Tensor
            ``(B, T, 768)`` student content features.
        batch : dict
            Must contain ``token_ids`` and ``token_lengths``.
        valid_mask : Tensor
            ``(B, T)`` boolean mask (True = valid frame).

        Returns
        -------
        Tensor
            Scalar ASR loss.
        """
        unwrapped = self.accelerator.unwrap_model(self.model)
        token_ids = batch["token_ids"]
        token_lengths = batch["token_lengths"]
        feature_lengths = valid_mask.long().sum(dim=-1)

        logits = unwrapped.asr_head(
            encoder_output=content_features,
            target_ids=token_ids,
        )
        asr_loss = unwrapped.asr_head.compute_loss(
            logits=logits,
            targets=token_ids,
            encoder_lengths=feature_lengths,
            target_lengths=token_lengths,
        )
        return asr_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> dict[str, float]:
        """Run validation and return average MSE and cosine similarity.

        Returns
        -------
        dict[str, float]
            ``loss``, ``mse``, ``cosine_similarity``.
            Empty dict if no validation dataloader is available.
        """
        val_dl = getattr(self, "_val_dataloader", None)
        if val_dl is None:
            return {}

        total_loss = 0.0
        total_mse = 0.0
        total_cos = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_dl:
                batch = _move_batch(batch, self.accelerator.device)
                loss_dict = self.compute_loss(batch)
                total_loss += loss_dict["loss"].item()
                total_mse += loss_dict["mse"].item()
                total_cos += loss_dict["cosine_similarity"].item()
                n_batches += 1

        if n_batches == 0:
            return {}

        return {
            "loss": total_loss / n_batches,
            "mse": total_mse / n_batches,
            "cosine_similarity": total_cos / n_batches,
        }

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        """Build the teacher model and log config at training start.

        The teacher is built here (rather than in ``__init__``) so that
        it is placed on the correct device after Accelerate preparation.
        """
        self.logger.info("Config: %s", _config_summary(self.config))

        # Pre-build the teacher so it is ready for the first step.
        if self._teacher is None:
            self._teacher = self._build_teacher()
            self._teacher = self._teacher.to(self.accelerator.device)
            self.logger.info("Teacher model placed on %s", self.accelerator.device)


# ======================================================================
# Helpers
# ======================================================================


def _move_batch(batch: dict, device: torch.device) -> dict:
    """Move all tensor values in *batch* to *device*."""
    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def _config_summary(config) -> str:
    """Return a concise string summary of the config for logging."""
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_yaml(config, resolve=True)
    except Exception:
        return str(config)
