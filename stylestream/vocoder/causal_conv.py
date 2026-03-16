"""Causal convolution primitives for the StyleStream Causal Vocos vocoder.

Provides causal variants of Conv1d and ConvTranspose1d that ensure no future
information leaks, enabling streaming inference.

Key property: output at time t depends only on inputs at times <= t.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CausalConv1d(nn.Module):
    """Causal 1D convolution with left-only padding.

    For kernel_size K and dilation D, applies ``(K - 1) * D`` left padding and
    zero right padding.  This ensures that the output at time *t* depends only
    on inputs at times ``<= t``.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolution kernel.
    stride : int
        Stride of the convolution (default ``1``).
    dilation : int
        Dilation factor (default ``1``).
    groups : int
        Number of blocked connections from input to output channels
        (default ``1``).
    bias : bool
        If ``True``, add a learnable bias (default ``True``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.pad_left = (kernel_size - 1) * dilation
        self.stride = stride

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal convolution.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(B, C_in, T)``.

        Returns
        -------
        Tensor
            Output tensor of shape ``(B, C_out, T_out)``.  When ``stride == 1``
            the output length equals the input length.  When ``stride > 1`` the
            output length is ``ceil(T / stride)``.
        """
        # Left-only padding preserves causality
        x = F.pad(x, (self.pad_left, 0))
        x = self.conv(x)

        return x


class CausalConvTranspose1d(nn.Module):
    """Causal transposed 1D convolution.

    Standard ``ConvTranspose1d`` with right-side trimming to remove samples
    that would depend on future inputs, restoring causal behaviour.

    For an upsampling transposed convolution with kernel_size K and stride S,
    the right trim removes ``K - S`` trailing samples so that no future
    information leaks into the output.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the transposed convolution kernel.
    stride : int
        Stride (upsampling factor) of the transposed convolution
        (default ``1``).
    bias : bool
        If ``True``, add a learnable bias (default ``True``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.trim_right = kernel_size - stride

        self.conv_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal transposed convolution.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(B, C_in, T)``.

        Returns
        -------
        Tensor
            Output tensor of shape ``(B, C_out, T * stride)``.
        """
        x = self.conv_transpose(x)

        # Trim the right side to remove future-leaking samples
        if self.trim_right > 0:
            x = x[:, :, : -self.trim_right]

        return x
