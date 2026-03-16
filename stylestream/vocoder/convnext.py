"""ConvNeXt V2 block for the StyleStream Causal Vocos vocoder.

Implements the ConvNeXt V2 architecture adapted for 1D audio processing.
Each block consists of a depthwise convolution, layer normalization,
pointwise convolution with GELU activation, and a residual connection.

Supports both causal (streaming) and non-causal (offline) modes.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from stylestream.vocoder.causal_conv import CausalConv1d


class ConvNeXtBlock(nn.Module):
    """ConvNeXt V2 block for Vocos backbone.

    Depthwise separable convolution with inverted bottleneck and residual
    connection. Supports both causal and non-causal modes.

    Architecture::

        x -> DepthwiseConv1d(dim, dim, kernel=7, groups=dim)
          -> LayerNorm (channel-wise)
          -> PointwiseConv1d(dim, intermediate_dim, kernel=1)
          -> GELU
          -> PointwiseConv1d(intermediate_dim, dim, kernel=1)
          -> + x (residual)

    Parameters
    ----------
    dim : int
        Number of input/output channels (e.g. 512).
    intermediate_dim : int
        Intermediate (expanded) channels in the pointwise layers (e.g. 1536).
    kernel_size : int
        Kernel size for the depthwise convolution (default 7).
    causal : bool
        If True, use causal (left-only) padding. Default True.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel_size: int = 7,
        causal: bool = True,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.intermediate_dim = intermediate_dim
        self.kernel_size = kernel_size
        self.causal = causal

        # Depthwise convolution: groups=dim for channel-wise filtering
        if causal:
            self.dwconv = CausalConv1d(
                dim, dim, kernel_size, groups=dim,
            )
        else:
            self.dwconv = nn.Conv1d(
                dim, dim, kernel_size, padding=kernel_size // 2, groups=dim,
            )

        # Channel-wise layer normalization (applied over the channel dim)
        self.norm = nn.LayerNorm(dim)

        # Pointwise up-projection
        self.pwconv1 = nn.Conv1d(dim, intermediate_dim, 1)

        # Activation
        self.act = nn.GELU()

        # Pointwise down-projection
        self.pwconv2 = nn.Conv1d(intermediate_dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, C, T)``.

        Returns
        -------
        Tensor
            Shape ``(B, C, T)``.
        """
        residual = x

        x = self.dwconv(x)

        # LayerNorm over channel dimension: (B, C, T) -> (B, T, C) -> norm -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        return x + residual

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dim={self.dim}, "
            f"intermediate_dim={self.intermediate_dim}, "
            f"kernel_size={self.kernel_size}, "
            f"causal={self.causal})"
        )
