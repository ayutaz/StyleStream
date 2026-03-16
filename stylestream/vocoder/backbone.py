"""Vocos backbone for the StyleStream Causal Vocos vocoder.

Implements the ConvNeXt-based backbone that transforms mel spectrograms
into hidden feature representations. The backbone consists of an input
embedding layer followed by a stack of ConvNeXt V2 blocks.

StyleStream spec:
    - hidden_size: 512
    - num_layers: 8 ConvNeXt blocks
    - intermediate_size: 1536
    - Causal convolutions for streaming
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from stylestream.vocoder.causal_conv import CausalConv1d
from stylestream.vocoder.convnext import ConvNeXtBlock


class VocosBackbone(nn.Module):
    """Vocos backbone: mel input projection + ConvNeXt stack.

    Transforms mel spectrograms into hidden feature representations
    that the ISTFT head converts to waveform.

    Parameters
    ----------
    n_mels : int
        Number of mel bins (100 for StyleStream).
    hidden_size : int
        Hidden dimension throughout the backbone (512).
    intermediate_size : int
        Intermediate dimension in ConvNeXt blocks (1536).
    num_layers : int
        Number of ConvNeXt blocks (8).
    kernel_size : int
        Kernel size for depthwise convolutions (default 7).
    causal : bool
        If True, use causal convolutions throughout. Default True.
    """

    def __init__(
        self,
        n_mels: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        kernel_size: int = 7,
        causal: bool = True,
    ) -> None:
        super().__init__()

        # Input embedding: project mel bins to hidden dimension
        if causal:
            self.embed = CausalConv1d(n_mels, hidden_size, kernel_size)
        else:
            self.embed = nn.Conv1d(
                n_mels, hidden_size, kernel_size, padding=kernel_size // 2
            )

        # Stack of ConvNeXt V2 blocks
        self.layers = nn.ModuleList(
            [
                ConvNeXtBlock(hidden_size, intermediate_size, kernel_size, causal)
                for _ in range(num_layers)
            ]
        )

        # Final layer normalization (applied channel-wise)
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, mel: Tensor) -> Tensor:
        """Transform mel spectrogram to hidden features.

        Parameters
        ----------
        mel : Tensor
            Mel spectrogram of shape ``(B, n_mels, T)``.

        Returns
        -------
        Tensor
            Hidden features of shape ``(B, hidden_size, T)``.
        """
        x = self.embed(mel)

        for layer in self.layers:
            x = layer(x)

        # Final LayerNorm (channel-wise): permute to (B, T, C), norm, permute back
        x = x.transpose(1, 2)
        x = self.final_norm(x)
        x = x.transpose(1, 2)

        return x
