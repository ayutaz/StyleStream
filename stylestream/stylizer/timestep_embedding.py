"""Sinusoidal timestep embedding for the StyleStream Stylizer DiT.

Embeds a scalar flow timestep t in [0, 1] (from Conditional Flow Matching)
into a dense vector suitable for adaLN-Zero conditioning.  The design follows
DiT-original (Peebles & Xie, 2023):

1. Sinusoidal positional encoding of the scalar timestep, with frequencies
   arranged on a log scale (identical to the DDPM / Transformer PE scheme
   but applied to a single scalar rather than sequence positions).
2. Two-layer MLP with SiLU activation to project the encoding to the DiT
   hidden dimension.

StyleStream Stylizer spec:
    - hidden_size = 768 (DiT hidden dimension)
    - frequency_dim = 256 (sinusoidal encoding half-dimension)
    - MLP: Linear(512, 768) -> SiLU -> Linear(768, 768)
    - Output is combined with the style embedding for adaLN-Zero

Reference:
    Peebles & Xie.  "Scalable Diffusion Models with Transformers."
    ICCV 2023.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimestepEmbedding(nn.Module):
    """Embed a scalar flow timestep t in [0, 1] into a vector.

    Uses sinusoidal encoding followed by a 2-layer MLP, following the
    DiT-original design (Peebles & Xie, 2023).

    Parameters
    ----------
    hidden_size : int
        Output dimension (768 for StyleStream DiT).
    frequency_dim : int
        Dimension of the sinusoidal encoding.  Default 256.
        The MLP input is ``2 * frequency_dim`` (sin + cos concatenated).
    max_period : float
        Maximum period for sinusoidal frequencies.  Default 10000.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        frequency_dim: int = 256,
        max_period: float = 10000.0,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.frequency_dim = frequency_dim

        # Pre-compute log-spaced frequencies as a non-learnable buffer.
        # freqs[i] = exp(-log(max_period) * i / frequency_dim)
        #          = max_period^{-i / frequency_dim}
        # for i = 0, 1, ..., frequency_dim - 1
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(frequency_dim, dtype=torch.float32)
            / frequency_dim
        )
        self.register_buffer("freqs", freqs)  # (frequency_dim,)

        # 2-layer MLP: [sin, cos] -> hidden_size
        mlp_input_dim = 2 * frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize MLP weights with Xavier uniform, biases to zero."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timestep(s).

        Parameters
        ----------
        t : torch.Tensor
            Shape ``(B,)`` or scalar.  Flow timestep values in [0, 1].

        Returns
        -------
        torch.Tensor
            Shape ``(B, hidden_size)``.  Timestep embeddings.
        """
        # Handle scalar input: reshape to (1,)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # t: (B,) -> (B, 1) for broadcasting against freqs (frequency_dim,)
        args = t.unsqueeze(-1).float() * self.freqs  # (B, frequency_dim)

        # Sinusoidal encoding: concatenate sin and cos
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        # (B, 2 * frequency_dim)

        # MLP projection to hidden_size
        return self.mlp(embedding)  # (B, hidden_size)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hidden_size={self.hidden_size}, "
            f"frequency_dim={self.frequency_dim})"
        )
