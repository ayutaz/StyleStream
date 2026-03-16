"""Full Stylizer model: DiT + Style Encoder + CFM + CFG.

Integrates all Stylizer sub-components into a single module that handles
both training (CFM loss computation) and inference (Euler ODE sampling
with classifier-free guidance).

Architecture::

    Training:
        mel (B,T,100), content (B,T,768), mask (B,T), style_waveform (B,S)
          -> Style Encoder: style_waveform -> style_emb (B, 768)
          -> CFG dropout on (content, context_mel, style_emb)
          -> Sample t ~ U[0,1], x_0 ~ N(0,I)
          -> x_t = (1-(1-sigma_min)*t) * x_0 + t * mel    (OT interpolation)
          -> DiT(x_t, t, content, context_mel, style_emb)  -> velocity_pred
          -> CFM masked loss: ||mask * (v_pred - u_t)||^2

    Inference:
        content_features (B,T,768), style_waveform (B,S),
        context_mel (B,T_total,100), mask (B,T_total)
          -> Style Encoder: style_waveform -> style_emb (B, 768)
          -> Euler sampling with CFG: x_0 -> x_1 over NFE=16 steps
          -> Replace non-masked region with context mel
          -> output mel (B, T_total, 100)

Sub-components
--------------
- **DiT**: 16-layer Diffusion Transformer with adaLN-Zero conditioning,
  RoPE positional encoding, 768 hidden, 3072 FFN, 12 heads.
- **Style Encoder**: Frozen WavLM-Base-Plus-SV with learned layer
  aggregation, TDNN stack, and attentive statistics pooling.
- **CFM**: Conditional Flow Matching with OT path and Euler sampling.
- **CFG**: Classifier-Free Guidance with per-condition training dropout
  and batched inference guidance.

Spectrogram inpainting
----------------------
The Stylizer operates in an inpainting paradigm:

- A binary mask ``(B, T)`` indicates which frames to generate (1) and
  which to keep as context (0).
- During training, the loss is computed only on masked frames.
- During inference, the generated mel is blended with the context mel
  using the mask.

StyleStream spec
----------------
- DiT: 16 layers, hidden 768, FFN 3072, 12 heads, head_dim 64
- Style encoder: WavLM-Base-Plus-SV, 13 hidden states, TDNN 512ch
- CFM: OT path, sigma_min 1e-5, NFE 16, Euler sampling
- CFG: content_drop 0.2, context_drop 0.3, style_drop 0.3, alpha 2.0
- Mel: 100 bins, hop 320, 16 kHz, 50 Hz frame rate
- Mask ratio: U[0.7, 1.0] during training
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from stylestream.stylizer.dit import DiT
from stylestream.stylizer.style_encoder import StyleEncoder
from stylestream.stylizer.cfm import ConditionalFlowMatching
from stylestream.stylizer.cfg import ClassifierFreeGuidance

logger = logging.getLogger(__name__)

# Default dimensions matching the paper spec.
_DEFAULT_MEL_DIM: int = 100
_DEFAULT_HIDDEN_SIZE: int = 768
_DEFAULT_CONTENT_DIM: int = 768


class Stylizer(nn.Module):
    """Full Stylizer model: DiT + Style Encoder + CFM + CFG.

    The Stylizer takes content features from the Destylizer and a style
    reference audio, then generates a mel spectrogram via Conditional
    Flow Matching with spectrogram inpainting.

    At **training** time, :meth:`forward` computes the CFM loss on the
    masked region.  At **inference** time, :meth:`sample` generates a
    mel spectrogram via Euler ODE integration with classifier-free
    guidance.

    Parameters
    ----------
    config : StylizerConfig, optional
        Structured configuration from :mod:`stylestream.config`.
        When *None*, keyword arguments are used to build each
        sub-component with default or overridden values.
    **kwargs
        Override individual settings when *config* is not provided.
        Accepted keys:

        DiT:
            ``num_layers``, ``hidden_size``, ``ffn_size``, ``num_heads``,
            ``mel_dim``, ``content_dim``, ``dropout``,
            ``gradient_checkpointing``

        Style encoder:
            ``wavlm_model_id``, ``num_wavlm_layers``, ``tdnn_channels``,
            ``output_size``, ``freeze_wavlm``

        CFM:
            ``sigma_min``

        CFG:
            ``content_drop_prob``, ``context_drop_prob``,
            ``style_drop_prob``, ``guidance_strength``

        Inference defaults:
            ``nfe`` (default Euler steps)
    """

    def __init__(self, config=None, **kwargs: Any) -> None:
        super().__init__()

        if config is not None:
            self._init_from_config(config)
        else:
            self._init_from_kwargs(kwargs)

        logger.info(
            "Stylizer: dit=%d layers, style_encoder=%s, "
            "cfm sigma_min=%.1e, cfg alpha=%.1f, mel_dim=%d",
            self.dit.num_layers,
            self.style_encoder.wavlm_model_id,
            self.cfm.sigma_min,
            self.cfg.guidance_strength,
            self.mel_dim,
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_from_config(self, config) -> None:
        """Build sub-modules from a :class:`StylizerConfig`."""
        dit_cfg = config.dit
        se_cfg = config.style_encoder
        cfm_cfg = config.cfm
        cfg_cfg = config.cfg
        mel_cfg = config.mel

        self.mel_dim = mel_cfg.n_mels
        self.nfe = cfm_cfg.nfe

        # --- DiT ---
        self.dit = DiT(
            num_layers=dit_cfg.num_layers,
            hidden_size=dit_cfg.hidden_size,
            ffn_size=dit_cfg.ffn_size,
            num_heads=getattr(dit_cfg, "num_heads", 12),
            mel_dim=self.mel_dim,
            content_dim=getattr(dit_cfg, "content_dim", _DEFAULT_CONTENT_DIM),
            dropout=getattr(dit_cfg, "dropout", 0.0),
            gradient_checkpointing=getattr(
                dit_cfg, "gradient_checkpointing", False
            ),
        )

        # --- Style encoder ---
        self.style_encoder = StyleEncoder(
            wavlm_model_id=se_cfg.model_id,
            hidden_size=se_cfg.hidden_size,
            num_wavlm_layers=se_cfg.num_layers,
            tdnn_channels=getattr(se_cfg, "tdnn_channels", 512),
            output_size=getattr(se_cfg, "output_size", _DEFAULT_HIDDEN_SIZE),
            freeze_wavlm=getattr(se_cfg, "freeze_wavlm", True),
        )

        # --- CFM (no learnable parameters) ---
        self.cfm = ConditionalFlowMatching(
            sigma_min=getattr(cfm_cfg, "sigma_min", 1e-5),
        )

        # --- CFG (not an nn.Module, no learnable parameters) ---
        self.cfg = ClassifierFreeGuidance(
            content_drop_prob=cfg_cfg.content_drop,
            context_drop_prob=cfg_cfg.context_drop,
            style_drop_prob=cfg_cfg.style_drop,
            guidance_strength=cfg_cfg.strength,
        )

    def _init_from_kwargs(self, kw: dict[str, Any]) -> None:
        """Build sub-modules from flat keyword arguments."""
        hidden = kw.get("hidden_size", _DEFAULT_HIDDEN_SIZE)
        mel_dim = kw.get("mel_dim", _DEFAULT_MEL_DIM)

        self.mel_dim = mel_dim
        self.nfe = kw.get("nfe", 16)

        # --- DiT ---
        self.dit = DiT(
            num_layers=kw.get("num_layers", 16),
            hidden_size=hidden,
            ffn_size=kw.get("ffn_size", 3072),
            num_heads=kw.get("num_heads", 12),
            mel_dim=mel_dim,
            content_dim=kw.get("content_dim", _DEFAULT_CONTENT_DIM),
            dropout=kw.get("dropout", 0.0),
            gradient_checkpointing=kw.get("gradient_checkpointing", False),
        )

        # --- Style encoder ---
        self.style_encoder = StyleEncoder(
            wavlm_model_id=kw.get(
                "wavlm_model_id", "microsoft/wavlm-base-plus-sv"
            ),
            hidden_size=kw.get("style_hidden_size", hidden),
            num_wavlm_layers=kw.get("num_wavlm_layers", 13),
            tdnn_channels=kw.get("tdnn_channels", 512),
            output_size=kw.get("output_size", hidden),
            freeze_wavlm=kw.get("freeze_wavlm", True),
        )

        # --- CFM ---
        self.cfm = ConditionalFlowMatching(
            sigma_min=kw.get("sigma_min", 1e-5),
        )

        # --- CFG ---
        self.cfg = ClassifierFreeGuidance(
            content_drop_prob=kw.get("content_drop_prob", 0.2),
            context_drop_prob=kw.get("context_drop_prob", 0.3),
            style_drop_prob=kw.get("style_drop_prob", 0.3),
            guidance_strength=kw.get("guidance_strength", 2.0),
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config) -> "Stylizer":
        """Build a Stylizer from a :class:`StylizerConfig`.

        Parameters
        ----------
        config : StylizerConfig
            Full stylizer configuration from :mod:`stylestream.config`.

        Returns
        -------
        Stylizer
            Initialised model (on CPU, with random trainable weights
            and pretrained frozen WavLM weights).
        """
        return cls(config=config)

    # ------------------------------------------------------------------
    # Training forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        mel: torch.Tensor,
        content_features: torch.Tensor,
        mask: torch.Tensor,
        style_waveform: torch.Tensor,
        cfg_drop_content: torch.Tensor | None = None,
        cfg_drop_context: torch.Tensor | None = None,
        cfg_drop_style: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Training forward pass.

        Computes the masked CFM loss for spectrogram inpainting.

        Steps:

        1. Compute context mel from the ground-truth mel and mask.
        2. Extract style embedding from the target waveform.
        3. Apply CFG training dropout to content, context, and style.
        4. Sample CFM noise and timestep.
        5. Interpolate to get noisy mel ``x_t``.
        6. Predict velocity with the DiT.
        7. Compute masked loss.

        Parameters
        ----------
        mel : Tensor
            ``(B, T, 100)`` ground-truth mel spectrogram.
        content_features : Tensor
            ``(B, T, 768)`` content features from the Destylizer.
        mask : Tensor
            ``(B, T)`` binary mask where ``1`` = masked (to generate)
            and ``0`` = context (to keep).
        style_waveform : Tensor
            ``(B, num_samples)`` raw 16 kHz audio of the target speaker
            for style extraction.
        cfg_drop_content : Tensor or None
            ``(B,)`` bool -- pre-sampled per-sample content dropout
            decisions.  If *None*, sampled internally by CFG.
        cfg_drop_context : Tensor or None
            ``(B,)`` bool -- pre-sampled context dropout decisions.
        cfg_drop_style : Tensor or None
            ``(B,)`` bool -- pre-sampled style dropout decisions.

        Returns
        -------
        dict
            ``'loss'``
                Scalar CFM loss (masked MSE between predicted and
                target velocity).
            ``'velocity_pred'``
                ``(B, T, 100)`` predicted velocity field.
        """
        B = mel.shape[0]
        device = mel.device
        dtype = mel.dtype

        # 1. Context mel: zero out masked positions.
        # mask (B, T) -> (B, T, 1) for broadcasting over mel_dim.
        context_mel = mel * (1.0 - mask.unsqueeze(-1))  # (B, T, mel_dim)

        # 2. Extract style embedding.
        style_emb = self.style_encoder(style_waveform)  # (B, hidden_size)

        # 3. Apply CFG training dropout.
        content_dropped, context_dropped, style_dropped = (
            self.cfg.apply_training_dropout(
                content_features,
                context_mel,
                style_emb,
                cfg_drop_content=cfg_drop_content,
                cfg_drop_context=cfg_drop_context,
                cfg_drop_style=cfg_drop_style,
            )
        )

        # 4. Sample CFM data: noise x_0 and timestep t.
        x_0 = self.cfm.sample_noise(
            mel.shape, device=device, dtype=dtype
        )  # (B, T, mel_dim)
        t = self.cfm.sample_timestep(B, device=device)  # (B,)

        # 5. OT interpolation: x_t = (1 - (1-sigma_min)*t)*x_0 + t*mel.
        x_t = self.cfm.interpolate(x_0, mel, t)  # (B, T, mel_dim)

        # 6. Predict velocity with DiT.
        velocity_pred = self.dit(
            x_t, t, content_dropped, context_dropped, style_dropped
        )  # (B, T, mel_dim)

        # 7. Compute masked CFM loss.
        loss = self.cfm.compute_loss(
            velocity_pred, mel, x_0, mask
        )  # scalar

        return {
            "loss": loss,
            "velocity_pred": velocity_pred,
        }

    # ------------------------------------------------------------------
    # Inference sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        content_features: torch.Tensor,
        style_waveform: torch.Tensor,
        context_mel: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        nfe: int | None = None,
        guidance_strength: float | None = None,
    ) -> torch.Tensor:
        """Generate mel spectrogram via CFM sampling with CFG.

        For voice conversion:

        - ``content_features``: from source audio via Destylizer.
        - ``style_waveform``: target speaker's reference audio.
        - ``context_mel``: target speaker's mel (for inpainting context).
        - ``mask``: ``1`` = generate, ``0`` = keep context.

        If ``context_mel`` is *None*, full generation is performed
        (equivalent to 100% mask with zero context).

        Parameters
        ----------
        content_features : Tensor
            ``(B, T, 768)`` content features from the Destylizer.
        style_waveform : Tensor
            ``(B, num_samples)`` raw 16 kHz audio for style extraction.
        context_mel : Tensor or None
            ``(B, T, 100)`` mel spectrogram of context frames.  When
            *None*, a zero tensor is used and the mask is set to all 1s.
        mask : Tensor or None
            ``(B, T)`` binary mask.  Required when ``context_mel`` is
            provided.  When *None*, defaults to all 1s (full generation).
        nfe : int or None
            Number of Euler steps.  Overrides the instance default when
            provided.  When *None*, uses ``self.nfe`` (typically 16).
        guidance_strength : float or None
            CFG guidance strength ``alpha``.  Overrides the instance
            default when provided.  When *None*, uses
            ``self.cfg.guidance_strength``.

        Returns
        -------
        Tensor
            ``(B, T, 100)`` generated mel spectrogram, with context
            frames replaced from ``context_mel`` in non-masked regions.
        """
        B, T, _ = content_features.shape
        device = content_features.device
        dtype = content_features.dtype

        nfe = nfe if nfe is not None else self.nfe

        # Handle missing context: full generation mode.
        if context_mel is None:
            context_mel = torch.zeros(
                B, T, self.mel_dim, device=device, dtype=dtype
            )
            mask = torch.ones(B, T, device=device, dtype=dtype)
        elif mask is None:
            mask = torch.ones(B, T, device=device, dtype=dtype)

        # Ensure mask dtype matches for arithmetic.
        mask = mask.to(dtype=dtype)

        # 1. Extract style embedding.
        style_emb = self.style_encoder(style_waveform)  # (B, hidden_size)

        # 2. Build velocity function with CFG guidance.
        #    The velocity_fn wraps the DiT and CFG together.
        def velocity_fn(x_t: torch.Tensor, t_step: torch.Tensor) -> torch.Tensor:
            return self.cfg.guided_velocity(
                velocity_fn=lambda xt, ti, c, ctx, s: self.dit(
                    xt, ti, c, ctx, s
                ),
                x_t=x_t,
                t=t_step,
                content_features=content_features,
                context_mel=context_mel,
                style_emb=style_emb,
                guidance_strength=guidance_strength,
            )

        # 3. Euler ODE sampling: x_0 (noise) -> x_1 (mel).
        shape = (B, T, self.mel_dim)
        generated = self.cfm.euler_sample(
            velocity_fn=velocity_fn,
            shape=shape,
            nfe=nfe,
            device=device,
            dtype=dtype,
        )  # (B, T, mel_dim)

        # 4. Inpainting blend: keep context in non-masked regions.
        mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)
        output = mask_expanded * generated + (1.0 - mask_expanded) * context_mel

        return output

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters.

        Parameters
        ----------
        trainable_only : bool
            If *True* (default), count only parameters with
            ``requires_grad=True``.

        Returns
        -------
        int
            Total number of (trainable) parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        n_trainable = self.num_parameters(trainable_only=True) / 1e6
        n_total = self.num_parameters(trainable_only=False) / 1e6
        return (
            f"{self.__class__.__name__}(\n"
            f"  (dit): {self.dit.num_layers} layers, "
            f"hidden={self.dit.hidden_size}, "
            f"heads={self.dit.num_heads}\n"
            f"  (style_encoder): {self.style_encoder.wavlm_model_id}\n"
            f"  (cfm): sigma_min={self.cfm.sigma_min}\n"
            f"  (cfg): alpha={self.cfg.guidance_strength}, "
            f"drops=({self.cfg.content_drop_prob}, "
            f"{self.cfg.context_drop_prob}, "
            f"{self.cfg.style_drop_prob})\n"
            f"  mel_dim={self.mel_dim}, nfe={self.nfe}\n"
            f"  trainable_params={n_trainable:.2f}M, "
            f"total_params={n_total:.2f}M\n"
            f")"
        )
