"""Multi-scale discriminator for the StyleStream Causal Vocos vocoder.

Implements the multi-scale discriminator (MSD) for GAN-based vocoder
training. Multiple SubDiscriminators operate on different temporal
resolutions of the input waveform via average pooling.

StyleStream spec:
    - Scales: [1, 2, 4]
    - Base channels: 64
    - Weight-normalized Conv1d layers
    - Returns intermediate features for feature matching loss
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torch.nn.utils.parametrizations import weight_norm


class SubDiscriminator(nn.Module):
    """Single-scale waveform discriminator.

    Architecture::

        Conv1d(1, ch, 15, 1, 7)          -> LeakyReLU
        Conv1d(ch, 2*ch, 41, 4, 20, g=4) -> LeakyReLU
        Conv1d(2*ch, 4*ch, 41, 4, 20, g=16) -> LeakyReLU
        Conv1d(4*ch, 8*ch, 41, 4, 20, g=64) -> LeakyReLU
        Conv1d(8*ch, 8*ch, 5, 1, 2)      -> LeakyReLU
        Conv1d(8*ch, 1, 3, 1, 1)

    All convolutions use weight normalization. LeakyReLU slope is 0.1.

    Parameters
    ----------
    channels : int
        Base channel count (default 64).
    """

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        ch = channels

        self.layers = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(1, ch, 15, 1, 7)),
                weight_norm(nn.Conv1d(ch, 2 * ch, 41, 4, 20, groups=4)),
                weight_norm(nn.Conv1d(2 * ch, 4 * ch, 41, 4, 20, groups=16)),
                weight_norm(nn.Conv1d(4 * ch, 8 * ch, 41, 4, 20, groups=64)),
                weight_norm(nn.Conv1d(8 * ch, 8 * ch, 5, 1, 2)),
            ]
        )
        self.final_conv = weight_norm(nn.Conv1d(8 * ch, 1, 3, 1, 1))
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Run the sub-discriminator on a (possibly pooled) waveform.

        Parameters
        ----------
        x : Tensor
            Waveform of shape ``(B, 1, T)``.

        Returns
        -------
        logits : Tensor
            Discriminator output of shape ``(B, 1, T')``.
        features : list[Tensor]
            Intermediate feature maps from each layer (before the final
            convolution), used for feature matching loss.
        """
        features: list[Tensor] = []
        for layer in self.layers:
            x = self.activation(layer(x))
            features.append(x)
        logits = self.final_conv(x)
        return logits, features


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for Vocos GAN training.

    Applies average pooling at multiple scales and runs a
    :class:`SubDiscriminator` at each scale.  Returns per-scale logits and
    intermediate features for adversarial and feature matching losses.

    Parameters
    ----------
    scales : list[int]
        Pooling factors for each scale (default ``[1, 2, 4]``).
    channels : int
        Base channel count for :class:`SubDiscriminator` (default 64).
    """

    def __init__(
        self,
        scales: list[int] | None = None,
        channels: int = 64,
    ) -> None:
        super().__init__()
        if scales is None:
            scales = [1, 2, 4]

        self.discriminators = nn.ModuleList(
            [SubDiscriminator(channels) for _ in scales]
        )
        self.pooling = nn.ModuleList(
            [
                nn.Identity() if s == 1 else nn.AvgPool1d(s, s)
                for s in scales
            ]
        )

    def forward(
        self, x: Tensor
    ) -> tuple[list[Tensor], list[list[Tensor]]]:
        """Discriminate waveform at multiple scales.

        Parameters
        ----------
        x : Tensor
            Waveform of shape ``(B, 1, T)`` or ``(B, T)``.
            If ``(B, T)``, unsqueezes to ``(B, 1, T)``.

        Returns
        -------
        logits_list : list[Tensor]
            Per-scale logits, each ``(B, 1, T_i)``.
        features_list : list[list[Tensor]]
            Per-scale intermediate features for feature matching.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        logits_list: list[Tensor] = []
        features_list: list[list[Tensor]] = []

        for pool, disc in zip(self.pooling, self.discriminators):
            x_scaled = pool(x)
            logits, features = disc(x_scaled)
            logits_list.append(logits)
            features_list.append(features)

        return logits_list, features_list

    @classmethod
    def from_config(cls, config) -> MultiScaleDiscriminator:
        """Build from config with ``discriminator.scales`` and ``discriminator.channels``.

        Parameters
        ----------
        config
            Configuration object (OmegaConf DictConfig or similar) with
            ``discriminator.scales`` and ``discriminator.channels`` fields.

        Returns
        -------
        MultiScaleDiscriminator
            Instantiated multi-scale discriminator.
        """
        disc_cfg = config.discriminator
        scales = list(disc_cfg.scales)
        channels = disc_cfg.channels
        return cls(scales=scales, channels=channels)
