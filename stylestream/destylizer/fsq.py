"""Finite Scalar Quantization (FSQ) for StyleStream Destylizer.

Implements FSQ (Mentzer et al., 2024) as an information bottleneck in the
Destylizer pipeline.  FSQ quantizes each dimension of a low-dimensional
projection independently using a fixed, codebook-free scheme.  The
straight-through estimator (STE) makes the operation differentiable.

StyleStream Destylizer spec:
    - Input: Conformer output, shape (B, T, 768)
    - Down projection: Linear(768, 3)
    - Quantization levels: [5, 3, 3]  (codebook size 45, ~5.49 bits/frame)
    - Up projection: Linear(3, 768)
    - Content feature fc is the Conformer output BEFORE FSQ (continuous)
    - FSQ exists only to create a training bottleneck; at inference fc is used

Reference:
    Mentzer, Minnen, Agustsson & Tschannen.  "Finite Scalar Quantization:
    VQ-VAE Made Simple."  ICLR 2024.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


class FSQ(nn.Module):
    """Finite Scalar Quantization with configurable levels.

    Each quantization dimension *d* has ``levels[d]`` discrete values
    (must be odd so they are symmetric around zero).  The half-width for
    dimension *d* is ``K_d = (levels[d] - 1) // 2``, and the quantized
    value is ``round(clamp(z, -K_d, K_d))``.

    Parameters
    ----------
    levels : list[int]
        Quantization levels per dimension, e.g. ``[5, 3, 3]``.
        Every entry must be an odd positive integer.
    hidden_size : int
        Input / output feature dimension (typically 768 for the Destylizer).
    """

    def __init__(self, levels: List[int], hidden_size: int = 768) -> None:
        super().__init__()

        for lv in levels:
            if lv < 1 or lv % 2 == 0:
                raise ValueError(
                    f"Each level must be an odd positive integer, got {lv}"
                )

        self._levels = list(levels)
        self._hidden_size = hidden_size
        D = len(levels)

        # Half-widths per dimension: K_d = (V_d - 1) / 2
        # e.g. levels [5, 3, 3] -> half_levels [2, 1, 1]
        half_levels = torch.tensor(
            [(lv - 1) / 2 for lv in levels], dtype=torch.float32
        )
        self.register_buffer("_half_levels", half_levels)  # (D,)

        # Basis for converting D-dim codes to flat indices (big-endian):
        #   index = codes[0] * (L1 * L2) + codes[1] * L2 + codes[2]
        # We store cumulative products from the right.
        basis = torch.ones(D, dtype=torch.long)
        for d in range(D - 2, -1, -1):
            basis[d] = basis[d + 1] * levels[d + 1]
        self.register_buffer("_basis", basis)  # (D,)

        # Store levels as a long tensor for index conversion
        self.register_buffer(
            "_levels_tensor", torch.tensor(levels, dtype=torch.long)
        )

        # Down / up linear projections
        self.down_proj = nn.Linear(hidden_size, D)
        self.up_proj = nn.Linear(D, hidden_size)

        # Xavier uniform initialization for the projections
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def codebook_size(self) -> int:
        """Total codebook size (product of levels)."""
        result = 1
        for lv in self._levels:
            result *= lv
        return result

    @property
    def num_dimensions(self) -> int:
        """Number of quantization dimensions D."""
        return len(self._levels)

    # -----------------------------------------------------------------
    # Core operations
    # -----------------------------------------------------------------

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Apply per-dimension quantization with STE.

        Parameters
        ----------
        z : torch.Tensor
            Shape ``(B, T, D)`` where ``D = len(levels)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, T, D)`` — quantized values, each entry an integer
            in ``[-K_d, K_d]`` (as float for gradient flow).
        """
        # _half_levels: (D,)  -> broadcasts against (B, T, D)
        K = self._half_levels
        z_q = z + (torch.round(torch.clamp(z, -K, K)) - z).detach()
        return z_q

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert D-dimensional quantized codes to flat codebook indices.

        The codes are in ``[-K_d, K_d]``; we first shift them to
        ``[0, levels_d - 1]`` then apply a mixed-radix encoding.

        Parameters
        ----------
        codes : torch.Tensor
            Shape ``(B, T, D)`` — quantized values (integer-valued floats).

        Returns
        -------
        torch.Tensor
            Shape ``(B, T)`` — indices in ``[0, codebook_size)``.
        """
        # Shift to non-negative: add K_d so range becomes [0, V_d - 1]
        shifted = (codes + self._half_levels).long()  # (B, T, D)
        # Mixed-radix dot product with basis
        indices = (shifted * self._basis).sum(dim=-1)  # (B, T)
        return indices

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert flat indices back to D-dimensional codes.

        Parameters
        ----------
        indices : torch.Tensor
            Shape ``(B, T)`` — integer indices in ``[0, codebook_size)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, T, D)`` — quantized codes in ``[-K_d, K_d]``.
        """
        codes = []
        remainder = indices
        for d in range(self.num_dimensions):
            codes.append(remainder // self._basis[d])
            remainder = remainder % self._basis[d]
        # Stack: (B, T, D)  — shift back to signed range
        codes_tensor = torch.stack(codes, dim=-1).float()
        codes_tensor = codes_tensor - self._half_levels
        return codes_tensor

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """Run the full FSQ pipeline: down-project, quantize, up-project.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, T, hidden_size)`` — Conformer output.  This tensor
            **is** the content feature ``fc`` (continuous, 768-dim) that the
            Stylizer will consume; FSQ is applied only to create a training
            bottleneck.

        Returns
        -------
        quantized : torch.Tensor
            Shape ``(B, T, hidden_size)`` — FSQ-processed features.
        info : dict
            Diagnostic information:

            - ``'indices'``  ``(B, T)`` — flat codebook indices.
            - ``'codebook_usage'`` ``float`` — fraction of the 45 codes
              observed in this batch.
            - ``'perplexity'`` ``float`` — ``exp(entropy)`` of the index
              distribution over the batch.
            - ``'pre_quant'`` ``(B, T, D)`` — values after the down
              projection but before quantization.
        """
        # --- Down projection ---
        z = self.down_proj(x)  # (B, T, D)

        # --- Quantize with STE ---
        z_q = self.quantize(z)  # (B, T, D)

        # --- Up projection ---
        quantized = self.up_proj(z_q)  # (B, T, hidden_size)

        # --- Diagnostics ---
        indices = self.codes_to_indices(z_q)  # (B, T)
        usage, perplexity = self._compute_utilization(indices)

        info = {
            "indices": indices,
            "codebook_usage": usage,
            "perplexity": perplexity,
            "pre_quant": z,
        }

        return quantized, info

    # -----------------------------------------------------------------
    # Utilization metrics
    # -----------------------------------------------------------------

    def _compute_utilization(
        self, indices: torch.Tensor
    ) -> tuple[float, float]:
        """Compute codebook utilization and perplexity over a batch.

        Parameters
        ----------
        indices : torch.Tensor
            Shape ``(B, T)`` — flat codebook indices.

        Returns
        -------
        usage : float
            Fraction of the codebook entries that appear at least once.
        perplexity : float
            ``exp(entropy)`` of the usage distribution.  Maximum possible
            value equals ``codebook_size`` (uniform usage).
        """
        flat = indices.reshape(-1)  # (B*T,)
        counts = torch.bincount(flat, minlength=self.codebook_size).float()

        # Usage: fraction of codes with count > 0
        usage = (counts > 0).float().sum().item() / self.codebook_size

        # Perplexity: exp(H) where H = -sum(p * log(p))
        probs = counts / counts.sum()
        # Avoid log(0) by filtering zero entries
        nonzero_mask = probs > 0
        log_probs = torch.zeros_like(probs)
        log_probs[nonzero_mask] = torch.log(probs[nonzero_mask])
        entropy = -(probs * log_probs).sum().item()
        perplexity = math.exp(entropy)

        return usage, perplexity

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"levels={self._levels}, "
            f"codebook_size={self.codebook_size}, "
            f"hidden_size={self._hidden_size})"
        )
