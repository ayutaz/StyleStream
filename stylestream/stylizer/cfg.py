"""Classifier-Free Guidance (CFG) for StyleStream Stylizer.

Implements training-time condition dropout and inference-time guided velocity
computation for the Stylizer's Conditional Flow Matching (CFM) framework.

During training, three conditions -- content features, context mel, and style
embedding -- are independently dropped (zeroed) with per-sample Bernoulli
draws.  This teaches the model to generate plausible velocities even when
some or all conditions are absent.

At inference, the guided velocity is computed as:

    v_cfg = (1 + alpha) * v_cond - alpha * v_uncond

where ``v_cond`` is the fully-conditioned velocity and ``v_uncond`` is the
velocity with all conditions zeroed.  A batched formulation doubles the input
to compute both in a single forward pass for efficiency.

Paper reference (Section 3.3):
    - Content dropout: 20 %
    - Context dropout: 30 %
    - Style dropout: 30 %
    - Guidance strength alpha = 2.0
"""

from __future__ import annotations

from typing import Callable

import torch


class ClassifierFreeGuidance:
    """Classifier-Free Guidance for StyleStream Stylizer.

    Handles both training-time condition dropout and inference-time
    guided velocity computation.

    This is **not** an ``nn.Module`` -- it has no learnable parameters.
    It is a stateless utility that operates on tensors directly.

    Parameters
    ----------
    content_drop_prob : float
        Probability of dropping content features during training.
        Default 0.2 (paper value).
    context_drop_prob : float
        Probability of dropping context mel during training.
        Default 0.3 (paper value).
    style_drop_prob : float
        Probability of dropping style embedding during training.
        Default 0.3 (paper value).
    guidance_strength : float
        CFG strength ``alpha`` used at inference time.  Default 2.0
        (paper value).
    """

    def __init__(
        self,
        content_drop_prob: float = 0.2,
        context_drop_prob: float = 0.3,
        style_drop_prob: float = 0.3,
        guidance_strength: float = 2.0,
    ) -> None:
        if not 0.0 <= content_drop_prob <= 1.0:
            raise ValueError(
                f"content_drop_prob must be in [0, 1], got {content_drop_prob}"
            )
        if not 0.0 <= context_drop_prob <= 1.0:
            raise ValueError(
                f"context_drop_prob must be in [0, 1], got {context_drop_prob}"
            )
        if not 0.0 <= style_drop_prob <= 1.0:
            raise ValueError(
                f"style_drop_prob must be in [0, 1], got {style_drop_prob}"
            )

        self.content_drop_prob = content_drop_prob
        self.context_drop_prob = context_drop_prob
        self.style_drop_prob = style_drop_prob
        self.guidance_strength = guidance_strength

    # -----------------------------------------------------------------
    # Training: condition dropout
    # -----------------------------------------------------------------

    def apply_training_dropout(
        self,
        content_features: torch.Tensor,
        context_mel: torch.Tensor,
        style_emb: torch.Tensor,
        cfg_drop_content: torch.Tensor | None = None,
        cfg_drop_context: torch.Tensor | None = None,
        cfg_drop_style: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply independent CFG dropout during training.

        Each condition is independently zeroed on a per-sample basis.
        If pre-sampled dropout masks are provided (from the dataloader),
        they are used directly.  Otherwise, new Bernoulli masks are sampled.

        Parameters
        ----------
        content_features : Tensor
            Shape ``(B, T, 768)`` -- content features from Destylizer.
        context_mel : Tensor
            Shape ``(B, T, 100)`` -- unmasked context mel spectrogram.
        style_emb : Tensor
            Shape ``(B, 768)`` -- style embedding from WavLM-TDNN encoder.
        cfg_drop_content : Tensor or None
            Shape ``(B,)`` bool -- pre-sampled content dropout decisions.
        cfg_drop_context : Tensor or None
            Shape ``(B,)`` bool -- pre-sampled context dropout decisions.
        cfg_drop_style : Tensor or None
            Shape ``(B,)`` bool -- pre-sampled style dropout decisions.

        Returns
        -------
        tuple of (content_features, context_mel, style_emb)
            Copies of the inputs with dropped conditions zeroed out.
        """
        B = content_features.shape[0]
        device = content_features.device

        # Sample dropout masks if not provided
        if cfg_drop_content is None:
            cfg_drop_content = (
                torch.rand(B, device=device) < self.content_drop_prob
            )
        else:
            cfg_drop_content = cfg_drop_content.to(device=device)

        if cfg_drop_context is None:
            cfg_drop_context = (
                torch.rand(B, device=device) < self.context_drop_prob
            )
        else:
            cfg_drop_context = cfg_drop_context.to(device=device)

        if cfg_drop_style is None:
            cfg_drop_style = (
                torch.rand(B, device=device) < self.style_drop_prob
            )
        else:
            cfg_drop_style = cfg_drop_style.to(device=device)

        # Apply per-sample zeroing
        content_features = self._zero_like_condition(
            content_features, cfg_drop_content
        )
        context_mel = self._zero_like_condition(context_mel, cfg_drop_context)
        style_emb = self._zero_like_condition(style_emb, cfg_drop_style)

        return content_features, context_mel, style_emb

    # -----------------------------------------------------------------
    # Inference: guided velocity
    # -----------------------------------------------------------------

    def guided_velocity(
        self,
        velocity_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
        x_t: torch.Tensor,
        t: torch.Tensor,
        content_features: torch.Tensor,
        context_mel: torch.Tensor,
        style_emb: torch.Tensor,
        guidance_strength: float | None = None,
    ) -> torch.Tensor:
        """Compute CFG-guided velocity at inference time.

        Uses a batched formulation: the batch is doubled so that the
        conditional and unconditional velocities are computed in a single
        forward pass through the DiT.

        Parameters
        ----------
        velocity_fn : callable
            ``fn(x_t, t, content, context, style) -> velocity``
            where all inputs and the output have batch dimension ``B``
            (or ``2B`` for the batched call).
        x_t : Tensor
            Shape ``(B, T, mel_dim)`` -- noisy mel at time *t*.
        t : Tensor
            Shape ``(B,)`` -- diffusion time step.
        content_features : Tensor
            Shape ``(B, T, 768)`` -- content features.
        context_mel : Tensor
            Shape ``(B, T, 100)`` -- context mel.
        style_emb : Tensor
            Shape ``(B, 768)`` -- style embedding.
        guidance_strength : float or None
            Override the default ``alpha``.  When ``None``, the instance
            default (``self.guidance_strength``) is used.

        Returns
        -------
        Tensor
            Shape ``(B, T, mel_dim)`` -- CFG-guided velocity.
        """
        alpha = (
            guidance_strength
            if guidance_strength is not None
            else self.guidance_strength
        )

        # Fast path: no guidance -- skip unconditional computation
        if alpha == 0.0:
            return velocity_fn(
                x_t, t, content_features, context_mel, style_emb
            )

        # Batched computation: double the batch
        x_double = torch.cat([x_t, x_t], dim=0)  # (2B, T, mel_dim)
        t_double = torch.cat([t, t], dim=0)  # (2B,)

        zeros_content = torch.zeros_like(content_features)  # (B, T, 768)
        zeros_context = torch.zeros_like(context_mel)  # (B, T, 100)
        zeros_style = torch.zeros_like(style_emb)  # (B, 768)

        fc_double = torch.cat(
            [content_features, zeros_content], dim=0
        )  # (2B, T, 768)
        ctx_double = torch.cat(
            [context_mel, zeros_context], dim=0
        )  # (2B, T, 100)
        e_double = torch.cat(
            [style_emb, zeros_style], dim=0
        )  # (2B, 768)

        # Single forward pass
        v_double = velocity_fn(
            x_double, t_double, fc_double, ctx_double, e_double
        )  # (2B, T, mel_dim)

        v_cond, v_uncond = v_double.chunk(2, dim=0)  # each (B, T, mel_dim)

        # CFG formula: v_cfg = (1 + alpha) * v_cond - alpha * v_uncond
        v_cfg = (1.0 + alpha) * v_cond - alpha * v_uncond

        return v_cfg

    # -----------------------------------------------------------------
    # Static helper
    # -----------------------------------------------------------------

    @staticmethod
    def _zero_like_condition(
        tensor: torch.Tensor, drop_mask: torch.Tensor
    ) -> torch.Tensor:
        """Zero out tensor entries where ``drop_mask`` is True.

        Parameters
        ----------
        tensor : Tensor
            Arbitrary shape ``(B, ...)`` -- the condition tensor.
        drop_mask : Tensor
            Shape ``(B,)`` bool -- ``True`` means drop (zero out).

        Returns
        -------
        Tensor
            Same shape as *tensor*, with dropped entries zeroed.
        """
        # Build a broadcastable keep-mask: (B,) -> (B, 1, ..., 1)
        # with as many trailing 1s as tensor has non-batch dimensions.
        keep = (~drop_mask).float()
        for _ in range(tensor.dim() - 1):
            keep = keep.unsqueeze(-1)

        return tensor * keep

    # -----------------------------------------------------------------
    # Repr
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"content_drop_prob={self.content_drop_prob}, "
            f"context_drop_prob={self.context_drop_prob}, "
            f"style_drop_prob={self.style_drop_prob}, "
            f"guidance_strength={self.guidance_strength})"
        )
