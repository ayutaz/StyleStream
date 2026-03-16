"""Adaptive Layer Norm Zero (adaLN-Zero) for StyleStream Stylizer DiT.

Implements the adaLN-Zero conditioning mechanism from DiT-original
(Peebles & Xie, 2023) for the StyleStream Stylizer's 16-layer DiT.
Each DiT block applies adaLN-Zero twice (self-attention + FFN), with
a single conditioning MLP generating all 6 modulation vectors.

Architecture per DiT block::

    c (768) ── SiLU ── Linear(768, 768*6) ── split ──┐
                                                       │
              ┌──── gamma_1, beta_1, alpha_1 ──────────┤
              │     gamma_2, beta_2, alpha_2 ──────────┘
              │
    x ── LayerNorm ── (1 + gamma_1) * x + beta_1 ── SelfAttn ── alpha_1 * x ── + residual
      ── LayerNorm ── (1 + gamma_2) * x + beta_2 ── FFN ─────── alpha_2 * x ── + residual

The conditioning vector c = timestep_embedding(t) + style_embedding(e),
both 768-dimensional.  The alpha (gate) portions are zero-initialized,
so each DiT block acts as an identity function at the start of training,
stabilizing the gradient flow.

StyleStream Stylizer spec:
    - hidden_size = 768, 16 DiT layers
    - 6 modulation vectors per block: gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2
    - Entire projection layer zero-initialized (DiT-original approach)
    - FinalAdaLN: 2-parameter adaLN + linear projection to mel dim (100)

Reference:
    Peebles & Xie.  "Scalable Diffusion Models with Transformers."
    ICCV 2023.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaLNZero(nn.Module):
    """Adaptive Layer Norm Zero for DiT conditioning.

    Generates 6 modulation vectors from a conditioning signal:
    ``(gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2)``.

    The entire projection layer is zero-initialized (DiT-original approach)
    so that at initialization:
    - ``gamma = 0`` and ``beta = 0`` → adaLN outputs ``LayerNorm(x)``
    - ``alpha = 0`` → gated sub-layer output is zero → residual passes through

    Parameters
    ----------
    hidden_size : int
        Hidden dimension (768 for StyleStream DiT).
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        self.linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6),
        )

        # Zero-initialize the entire projection (DiT-original):
        # gamma=0 → scale is 1+0=1, beta=0 → no shift, alpha=0 → gate blocks.
        nn.init.zeros_(self.linear[1].weight)
        nn.init.zeros_(self.linear[1].bias)

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Compute 6 modulation vectors from conditioning signal.

        Parameters
        ----------
        c : torch.Tensor
            Shape ``(B, hidden_size)``.  Conditioning vector
            (timestep_emb + style_emb).

        Returns
        -------
        tuple of 6 Tensors, each ``(B, 1, hidden_size)``
            ``(gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2)``.
            Unsqueezed at dim=1 for broadcasting over sequence length T.
        """
        # (B, hidden_size) -> (B, hidden_size * 6)
        params = self.linear(c)

        # Split into 6 vectors of hidden_size each and unsqueeze for (B, 1, H)
        return tuple(p.unsqueeze(1) for p in params.chunk(6, dim=-1))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(hidden_size={self.hidden_size})"


class AdaLNModulation(nn.Module):
    """Apply adaLN modulation: scale and shift after LayerNorm.

    Computes ``(1 + gamma) * LayerNorm(x) + beta``, so that when
    ``gamma = 0`` and ``beta = 0`` the output equals ``LayerNorm(x)``.

    Parameters
    ----------
    hidden_size : int
        Feature dimension for the LayerNorm.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive layer normalization.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, hidden_size)``.
        gamma : torch.Tensor
            Shape ``(B, 1, hidden_size)`` — scale modulation.
        beta : torch.Tensor
            Shape ``(B, 1, hidden_size)`` — shift modulation.

        Returns
        -------
        torch.Tensor
            Shape ``(B, T, hidden_size)``.
        """
        return (1 + gamma) * self.norm(x) + beta


class FinalAdaLN(nn.Module):
    """Final adaLN + linear projection for DiT output.

    DiT-original applies one more adaLN before the final linear projection
    to the output dimension (mel spectrogram).  The output linear layer is
    zero-initialized so that the initial model prediction is zero.

    Parameters
    ----------
    hidden_size : int
        DiT hidden dimension (768).
    output_size : int
        Output dimension (100 for mel spectrogram).
    """

    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        self.norm = nn.LayerNorm(hidden_size)

        # 2-parameter adaLN: c -> (gamma, beta)
        self.adaln_linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )

        # Output projection to mel dimension
        self.output_linear = nn.Linear(hidden_size, output_size)

        # Zero-initialize both the adaLN and output projections
        nn.init.zeros_(self.adaln_linear[1].weight)
        nn.init.zeros_(self.adaln_linear[1].bias)
        nn.init.zeros_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply final adaLN and project to output dimension.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, hidden_size)``.
        c : torch.Tensor
            Shape ``(B, hidden_size)`` — conditioning vector.

        Returns
        -------
        torch.Tensor
            Shape ``(B, T, output_size)`` — velocity field prediction.
        """
        # Compute 2 modulation vectors: (B, hidden_size * 2)
        params = self.adaln_linear(c)
        gamma, beta = params.chunk(2, dim=-1)

        # Unsqueeze for broadcasting: (B, 1, hidden_size)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # Adaptive layer norm + output projection
        x = (1 + gamma) * self.norm(x) + beta
        return self.output_linear(x)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hidden_size={self.hidden_size}, "
            f"output_size={self.output_size})"
        )
