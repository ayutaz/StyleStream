"""Loss functions for the StyleStream Causal Vocos vocoder GAN training.

Implements all loss components:
    - LS-GAN adversarial losses (generator and discriminator)
    - Feature matching loss (L1 on discriminator intermediate features)
    - Mel spectrogram reconstruction loss (L1)
    - VocoderLoss: combined manager with configurable weights

StyleStream spec:
    - Reconstruction weight: 45.0
    - GAN generator weight: 1.0
    - GAN discriminator weight: 1.0
    - Feature matching weight: 2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from stylestream.utils.mel import MelSpectrogramTransform


# ======================================================================
# Standalone loss functions
# ======================================================================


def generator_adversarial_loss(disc_outputs: list[Tensor]) -> Tensor:
    """Least-squares GAN generator loss.

    The generator wants discriminator outputs to be close to 1 (i.e. to
    fool the discriminator into classifying generated audio as real).

    .. math::

        L_G = \\frac{1}{K} \\sum_{k=1}^{K} \\mathrm{mean}\\bigl(
              (D_k(\\text{fake}) - 1)^2 \\bigr)

    Parameters
    ----------
    disc_outputs : list[Tensor]
        Per-scale discriminator logits for generated (fake) audio.
        Each element has shape ``(B, ...)``.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    loss = torch.zeros(1, device=disc_outputs[0].device, dtype=disc_outputs[0].dtype)

    for d_out in disc_outputs:
        loss = loss + torch.mean((d_out - 1.0) ** 2)

    return loss / len(disc_outputs)


def discriminator_adversarial_loss(
    real_outputs: list[Tensor],
    fake_outputs: list[Tensor],
) -> Tensor:
    """Least-squares GAN discriminator loss.

    The discriminator should output 1 for real audio and 0 for generated
    (fake) audio.

    .. math::

        L_D = \\frac{1}{K} \\sum_{k=1}^{K} \\bigl[
              \\mathrm{mean}\\bigl((D_k(\\text{real}) - 1)^2\\bigr)
              + \\mathrm{mean}\\bigl(D_k(\\text{fake})^2\\bigr) \\bigr]

    Parameters
    ----------
    real_outputs : list[Tensor]
        Per-scale discriminator logits for real audio.
    fake_outputs : list[Tensor]
        Per-scale discriminator logits for generated audio.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    loss = torch.zeros(1, device=real_outputs[0].device, dtype=real_outputs[0].dtype)

    for r_out, f_out in zip(real_outputs, fake_outputs):
        loss = loss + torch.mean((r_out - 1.0) ** 2) + torch.mean(f_out ** 2)

    return loss / len(real_outputs)


def feature_matching_loss(
    real_features: list[list[Tensor]],
    fake_features: list[list[Tensor]],
) -> Tensor:
    """Feature matching loss across all discriminator scales and layers.

    Computes the L1 distance between discriminator intermediate features
    for real and fake audio.  Real features are detached so that
    gradients flow only through the generator path.

    .. math::

        L_{fm} = \\frac{1}{K} \\sum_{k=1}^{K} \\frac{1}{L_k}
                 \\sum_{l=1}^{L_k} \\lVert
                 \\text{feat}^{\\text{real}}_{k,l}
                 - \\text{feat}^{\\text{fake}}_{k,l}
                 \\rVert_1

    Parameters
    ----------
    real_features : list[list[Tensor]]
        Per-scale, per-layer intermediate features for real audio.
        ``real_features[k][l]`` has shape ``(B, C, T)``.
    fake_features : list[list[Tensor]]
        Per-scale, per-layer intermediate features for fake audio.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    loss = torch.zeros(
        1, device=fake_features[0][0].device, dtype=fake_features[0][0].dtype
    )

    num_scales = len(real_features)

    for scale_real, scale_fake in zip(real_features, fake_features):
        num_layers = len(scale_real)
        scale_loss = torch.zeros(
            1, device=fake_features[0][0].device, dtype=fake_features[0][0].dtype
        )

        for feat_real, feat_fake in zip(scale_real, scale_fake):
            # Detach real features — gradients flow only through the generator
            scale_loss = scale_loss + torch.mean(
                torch.abs(feat_real.detach() - feat_fake)
            )

        loss = loss + scale_loss / num_layers

    return loss / num_scales


def mel_reconstruction_loss(
    pred_waveform: Tensor,
    target_waveform: Tensor,
    mel_transform: nn.Module,
) -> Tensor:
    """L1 mel spectrogram reconstruction loss.

    Computes mel spectrograms for both predicted and target waveforms,
    then returns the L1 distance.

    Parameters
    ----------
    pred_waveform : Tensor
        Generated waveform of shape ``(B, T)``.
    target_waveform : Tensor
        Target waveform of shape ``(B, T)``.
    mel_transform : nn.Module
        :class:`~stylestream.utils.mel.MelSpectrogramTransform` that
        converts waveform to log-mel spectrogram.

    Returns
    -------
    Tensor
        Scalar L1 loss.
    """
    pred_mel = mel_transform(pred_waveform)      # (B, n_mels, T)
    target_mel = mel_transform(target_waveform)   # (B, n_mels, T)

    # Align lengths in case of minor frame-count differences
    min_t = min(pred_mel.shape[-1], target_mel.shape[-1])
    pred_mel = pred_mel[:, :, :min_t]
    target_mel = target_mel[:, :, :min_t]

    return torch.nn.functional.l1_loss(pred_mel, target_mel)


# ======================================================================
# Combined loss manager
# ======================================================================


class VocoderLoss(nn.Module):
    """Combined vocoder loss with configurable weights.

    Manages all loss components and returns both individual and total losses.
    The :class:`MelSpectrogramTransform` used for reconstruction loss is
    lazily initialised on first call so that its internal buffers are
    created on the correct device.

    Parameters
    ----------
    reconstruction_weight : float
        Weight for mel reconstruction loss (default 45.0).
    gan_generator_weight : float
        Weight for generator adversarial loss (default 1.0).
    gan_discriminator_weight : float
        Weight for discriminator adversarial loss (default 1.0).
    feature_matching_weight : float
        Weight for feature matching loss (default 2.0).
    n_mels : int
        Mel bins for reconstruction loss (default 100).
    hop_length : int
        Hop length for mel computation (default 320).
    sample_rate : int
        Sample rate (default 16000).
    """

    def __init__(
        self,
        reconstruction_weight: float = 45.0,
        gan_generator_weight: float = 1.0,
        gan_discriminator_weight: float = 1.0,
        feature_matching_weight: float = 2.0,
        n_mels: int = 100,
        hop_length: int = 320,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.gan_generator_weight = gan_generator_weight
        self.gan_discriminator_weight = gan_discriminator_weight
        self.feature_matching_weight = feature_matching_weight

        # Mel transform parameters (lazy init)
        self._n_mels = n_mels
        self._hop_length = hop_length
        self._sample_rate = sample_rate
        self._mel_transform: MelSpectrogramTransform | None = None

    # ------------------------------------------------------------------
    # Lazy mel transform
    # ------------------------------------------------------------------

    @property
    def mel_transform(self) -> MelSpectrogramTransform:
        """Lazily create :class:`MelSpectrogramTransform` on first access.

        This avoids creating torchaudio buffers at construction time,
        allowing the module to be moved to a device before first use.
        """
        if self._mel_transform is None:
            self._mel_transform = MelSpectrogramTransform(
                n_mels=self._n_mels,
                hop_length=self._hop_length,
                sample_rate=self._sample_rate,
            )
        return self._mel_transform

    # ------------------------------------------------------------------
    # Generator loss
    # ------------------------------------------------------------------

    def generator_loss(
        self,
        pred_waveform: Tensor,
        target_waveform: Tensor,
        disc_fake_outputs: list[Tensor],
        disc_real_features: list[list[Tensor]],
        disc_fake_features: list[list[Tensor]],
    ) -> dict[str, Tensor]:
        """Compute total generator loss.

        Parameters
        ----------
        pred_waveform : Tensor
            Generated waveform ``(B, T)``.
        target_waveform : Tensor
            Target waveform ``(B, T)``.
        disc_fake_outputs : list[Tensor]
            Per-scale discriminator logits for generated audio.
        disc_real_features : list[list[Tensor]]
            Per-scale, per-layer intermediate features for real audio.
        disc_fake_features : list[list[Tensor]]
            Per-scale, per-layer intermediate features for generated audio.

        Returns
        -------
        dict[str, Tensor]
            ``'loss'``: total generator loss (scalar).
            ``'mel_loss'``: mel reconstruction loss.
            ``'gan_loss'``: adversarial loss.
            ``'fm_loss'``: feature matching loss.
        """
        # Ensure mel transform is on the same device as the waveforms
        mel_xform = self.mel_transform.to(pred_waveform.device)

        mel_loss = mel_reconstruction_loss(pred_waveform, target_waveform, mel_xform)
        gan_loss = generator_adversarial_loss(disc_fake_outputs)
        fm_loss = feature_matching_loss(disc_real_features, disc_fake_features)

        total = (
            self.reconstruction_weight * mel_loss
            + self.gan_generator_weight * gan_loss
            + self.feature_matching_weight * fm_loss
        )

        return {
            "loss": total,
            "mel_loss": mel_loss,
            "gan_loss": gan_loss,
            "fm_loss": fm_loss,
        }

    # ------------------------------------------------------------------
    # Discriminator loss
    # ------------------------------------------------------------------

    def discriminator_loss(
        self,
        disc_real_outputs: list[Tensor],
        disc_fake_outputs: list[Tensor],
    ) -> dict[str, Tensor]:
        """Compute discriminator loss.

        Parameters
        ----------
        disc_real_outputs : list[Tensor]
            Per-scale discriminator logits for real audio.
        disc_fake_outputs : list[Tensor]
            Per-scale discriminator logits for generated audio.

        Returns
        -------
        dict[str, Tensor]
            ``'loss'``: total discriminator loss (scalar).
        """
        d_loss = discriminator_adversarial_loss(disc_real_outputs, disc_fake_outputs)
        total = self.gan_discriminator_weight * d_loss

        return {"loss": total}
