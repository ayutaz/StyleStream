"""Conditional Flow Matching (CFM) for the StyleStream Stylizer.

Implements the Optimal Transport (OT) conditional flow path, masked training
loss, and Euler ODE sampling for mel spectrogram generation.  This module is
self-contained and does not depend on other Stylizer sub-modules.

Training workflow::

    x_0 ~ N(0, I)                               # noise
    x_1 = target mel                             # ground truth
    t ~ U[0, 1]                                  # flow timestep
    x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1   # OT interpolation
    u_t = x_1 - (1 - sigma_min) * x_0           # target velocity

    v_hat = DiT(x_t, t, ...)                     # predicted velocity
    loss = masked_mse(v_hat, u_t, mask)          # loss on inpainting region

Inference (Euler sampling)::

    x_t = N(0, I)
    dt = 1 / NFE
    for i in range(NFE):
        t_i = i / NFE
        v = velocity_fn(x_t, t_i)
        x_t = x_t + v * dt
    return x_t  # generated mel

StyleStream Stylizer spec:
    - Mel spectrogram: 100 bins, 50 Hz frame rate (hop 320, 16 kHz)
    - NFE = 16 Euler steps at inference
    - OT path with sigma_min = 1e-5 for numerical stability
    - Loss masked to inpainting region (1 = generate, 0 = context)

References:
    Lipman, Chen & Ben-Hamu et al.  "Flow Matching for Generative Modeling."
    ICLR 2023.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class ConditionalFlowMatching(nn.Module):
    """Conditional Flow Matching for mel spectrogram generation.

    Implements the OT path, masked loss, and Euler sampling for the
    StyleStream Stylizer.

    Parameters
    ----------
    sigma_min : float
        Minimum noise level.  Default ``1e-5``.  At ``t = 0`` the
        interpolation uses ``sigma_min * x_0`` rather than ``0 * x_0``,
        preventing the ODE from being degenerate at ``t = 0``.  When
        ``sigma_min`` is very small this is nearly identical to the
        simple OT path.
    """

    def __init__(self, sigma_min: float = 1e-5, snr_gamma: float = 0.0) -> None:
        super().__init__()
        if sigma_min < 0:
            raise ValueError(f"sigma_min must be non-negative, got {sigma_min}")
        if snr_gamma < 0:
            raise ValueError(f"snr_gamma must be non-negative, got {snr_gamma}")
        self.sigma_min = sigma_min
        self.snr_gamma = snr_gamma

    # ------------------------------------------------------------------
    # Noise sampling
    # ------------------------------------------------------------------

    def sample_noise(
        self,
        shape: tuple,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sample initial noise x_0 ~ N(0, I).

        Parameters
        ----------
        shape : tuple
            Desired output shape, typically ``(B, T, mel_dim)``.
        device : torch.device
            Device for the output tensor.
        dtype : torch.dtype
            Data type for the output tensor.  Default ``torch.float32``.

        Returns
        -------
        torch.Tensor
            Gaussian noise of the given shape.
        """
        return torch.randn(shape, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Timestep sampling
    # ------------------------------------------------------------------

    def sample_timestep(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Sample flow timesteps t ~ U[0, 1].

        Parameters
        ----------
        batch_size : int
            Number of timesteps to sample.
        device : torch.device
            Device for the output tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(B,)`` with values in ``[0, 1]``.
        """
        return torch.rand(batch_size, device=device)

    # ------------------------------------------------------------------
    # OT path interpolation
    # ------------------------------------------------------------------

    def interpolate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute OT path interpolation with sigma_min stabilisation.

        .. math::

            x_t = (1 - (1 - \\sigma_{\\min}) \\cdot t) \\cdot x_0
                  + t \\cdot x_1

        When ``sigma_min`` is near zero this simplifies to the standard
        linear interpolation ``(1 - t) * x_0 + t * x_1``.

        Parameters
        ----------
        x_0 : torch.Tensor
            Shape ``(B, T, mel_dim)`` -- noise.
        x_1 : torch.Tensor
            Shape ``(B, T, mel_dim)`` -- target mel spectrogram.
        t : torch.Tensor
            Shape ``(B,)`` or ``(B, 1, 1)`` -- flow timestep.  If ``(B,)``
            it is automatically reshaped to ``(B, 1, 1)`` for broadcasting.

        Returns
        -------
        torch.Tensor
            Shape ``(B, T, mel_dim)`` -- interpolated noisy mel.
        """
        # Reshape t for broadcasting: (B,) -> (B, 1, 1)
        if t.dim() == 1:
            t = t.view(-1, 1, 1)

        # x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
        return (1.0 - (1.0 - self.sigma_min) * t) * x_0 + t * x_1

    # ------------------------------------------------------------------
    # Target velocity
    # ------------------------------------------------------------------

    def target_velocity(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the target velocity field for the OT path.

        .. math::

            u_t = x_1 - (1 - \\sigma_{\\min}) \\cdot x_0

        This velocity is constant (independent of t) along the OT path.

        Parameters
        ----------
        x_0 : torch.Tensor
            Shape ``(B, T, mel_dim)`` -- noise.
        x_1 : torch.Tensor
            Shape ``(B, T, mel_dim)`` -- target mel spectrogram.

        Returns
        -------
        torch.Tensor
            Shape ``(B, T, mel_dim)`` -- target velocity.
        """
        return x_1 - (1.0 - self.sigma_min) * x_0

    # ------------------------------------------------------------------
    # Masked loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        velocity_pred: torch.Tensor,
        x_1: torch.Tensor,
        x_0: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute masked CFM loss with optional Min-SNR weighting.

        The loss is the mean squared error between the predicted velocity
        and the target velocity, restricted to the masked (inpainting)
        region only.  When ``snr_gamma > 0`` and ``t`` is provided,
        Min-SNR weighting (Hang et al., 2023) is applied per sample to
        down-weight high-noise timesteps and accelerate convergence.

        .. math::

            L = \\frac{\\sum_{b,t,d} w_b \\cdot m_{b,t} \\cdot (\\hat{v}_{b,t,d}
                - u_{b,t,d})^2}{\\sum_{b,t} m_{b,t} \\cdot D + \\epsilon}

        where :math:`w_b = \\min(\\text{SNR}(t_b), \\gamma) / \\text{SNR}(t_b)`
        and :math:`\\text{SNR}(t) = t^2 / (1 - t)^2` for the OT path.

        Parameters
        ----------
        velocity_pred : torch.Tensor
            Shape ``(B, T, mel_dim)`` -- predicted velocity from the DiT.
        x_1 : torch.Tensor
            Shape ``(B, T, mel_dim)`` -- ground truth mel spectrogram.
        x_0 : torch.Tensor
            Shape ``(B, T, mel_dim)`` -- sampled noise.
        mask : torch.Tensor
            Shape ``(B, T)`` -- binary mask where ``1`` = masked (generate)
            and ``0`` = context.
        t : torch.Tensor or None
            Shape ``(B,)`` -- flow timestep used to generate ``x_t``.
            Required when ``snr_gamma > 0`` for Min-SNR weighting.
            When *None*, Min-SNR weighting is skipped regardless of
            ``snr_gamma``.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        # Target velocity: u_t = x_1 - (1 - sigma_min) * x_0
        u_t = self.target_velocity(x_0, x_1)  # (B, T, mel_dim)

        # Squared difference
        diff = velocity_pred - u_t  # (B, T, mel_dim)
        sq_diff = diff.pow(2)  # (B, T, mel_dim)

        # Expand mask from (B, T) to (B, T, 1) for broadcasting over mel_dim
        mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)

        # Zero out context positions
        masked_sq_diff = sq_diff * mask_expanded  # (B, T, mel_dim)

        # Apply Min-SNR weighting (Hang et al., 2023) when enabled.
        # For the OT path x_t = (1-t)*x_0 + t*x_1, the signal-to-noise
        # ratio is SNR(t) = t^2 / (1-t)^2.  The weight clamps high-SNR
        # timesteps and down-weights low-SNR (high-noise) timesteps that
        # would otherwise dominate the loss.
        if self.snr_gamma > 0 and t is not None:
            eps_snr = 1e-8
            snr = (t / (1.0 - t + eps_snr)).pow(2)  # (B,)
            weights = torch.clamp(snr, max=self.snr_gamma) / (snr + eps_snr)  # (B,)
            # Broadcast to (B, T, mel_dim)
            weights = weights.view(-1, 1, 1)
            masked_sq_diff = masked_sq_diff * weights

        # Normalise by the number of masked elements (frames * mel_dim)
        mel_dim = velocity_pred.shape[-1]
        num_masked = mask.sum() * mel_dim  # scalar

        eps = 1e-8
        loss = masked_sq_diff.sum() / (num_masked + eps)

        return loss

    # ------------------------------------------------------------------
    # Euler ODE sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def euler_sample(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: tuple,
        nfe: int = 16,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Generate mel spectrogram via Euler ODE integration.

        Integrates from ``t = 0`` (noise) to ``t = 1`` (mel) using ``nfe``
        Euler steps with uniform step size ``dt = 1 / nfe``.

        Parameters
        ----------
        velocity_fn : callable
            Function ``(x_t, t) -> velocity`` where ``x_t`` has shape
            ``(B, T, mel_dim)`` and ``t`` has shape ``(B,)``.  Returns
            predicted velocity ``(B, T, mel_dim)``.  This should include
            classifier-free guidance (CFG) if desired.
        shape : tuple
            Output shape ``(B, T, mel_dim)``.
        nfe : int
            Number of function evaluations (Euler steps).  Default 16.
        device : torch.device
            Device for the computation.  If ``None``, uses ``'cpu'``.
        dtype : torch.dtype
            Data type.  Default ``torch.float32``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, T, mel_dim)`` -- generated mel spectrogram.
        """
        if nfe < 1:
            raise ValueError(f"nfe must be >= 1, got {nfe}")

        if device is None:
            device = torch.device("cpu")

        batch_size = shape[0]
        dt = 1.0 / nfe

        # Start from pure noise
        x_t = self.sample_noise(shape, device=device, dtype=dtype)

        for i in range(nfe):
            # Current timestep for each sample in the batch
            t_i = torch.full(
                (batch_size,), i * dt, device=device, dtype=dtype
            )

            # Predict velocity
            v = velocity_fn(x_t, t_i)

            # Euler step
            x_t = x_t + v * dt

        return x_t

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sigma_min={self.sigma_min}, snr_gamma={self.snr_gamma})"
        )


# ======================================================================
# Convenience function for training
# ======================================================================


def prepare_training_data(
    x_1: torch.Tensor,
    mask: torch.Tensor,
    sigma_min: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """Prepare all CFM training data for one step.

    Convenience function that samples noise, a timestep, and computes the
    interpolated ``x_t`` and target velocity.  Useful for concise training
    loops.

    Parameters
    ----------
    x_1 : torch.Tensor
        Shape ``(B, T, mel_dim)`` -- ground truth mel spectrogram.
    mask : torch.Tensor
        Shape ``(B, T)`` -- binary mask (``1`` = masked / generate,
        ``0`` = context).  Passed through for reference but not used
        in the interpolation itself.
    sigma_min : float
        Minimum noise level.  Default ``1e-5``.

    Returns
    -------
    dict
        Keys:

        - ``'x_0'``: noise ``(B, T, mel_dim)``
        - ``'x_t'``: interpolated noisy mel ``(B, T, mel_dim)``
        - ``'t'``: sampled timestep ``(B,)``
        - ``'target_velocity'``: ``x_1 - (1 - sigma_min) * x_0``
          ``(B, T, mel_dim)``
    """
    cfm = ConditionalFlowMatching(sigma_min=sigma_min)

    batch_size = x_1.shape[0]
    device = x_1.device
    dtype = x_1.dtype

    # Sample noise and timestep
    x_0 = cfm.sample_noise(x_1.shape, device=device, dtype=dtype)
    t = cfm.sample_timestep(batch_size, device=device)

    # Compute interpolation and target velocity
    x_t = cfm.interpolate(x_0, x_1, t)
    target_velocity = cfm.target_velocity(x_0, x_1)

    return {
        "x_0": x_0,
        "x_t": x_t,
        "t": t,
        "target_velocity": target_velocity,
    }
