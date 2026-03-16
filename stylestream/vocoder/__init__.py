"""Vocoder module: Causal Vocos mel-to-waveform synthesis.

Exports
-------
CausalVocos
    Full vocoder model (ConvNeXt backbone + ISTFT head).
VocosBackbone
    ConvNeXt V2 backbone.
ISTFTHead
    ISTFT-based waveform generation head.
ConvNeXtBlock
    Single ConvNeXt V2 block.
CausalConv1d
    Causal 1D convolution primitive.
MultiScaleDiscriminator
    Multi-scale discriminator for GAN training.
VocoderLoss
    Combined loss manager.
"""

from stylestream.vocoder.model import CausalVocos
from stylestream.vocoder.backbone import VocosBackbone
from stylestream.vocoder.istft_head import ISTFTHead
from stylestream.vocoder.convnext import ConvNeXtBlock
from stylestream.vocoder.causal_conv import CausalConv1d
from stylestream.vocoder.discriminator import MultiScaleDiscriminator
from stylestream.vocoder.losses import VocoderLoss

__all__ = [
    "CausalVocos",
    "VocosBackbone",
    "ISTFTHead",
    "ConvNeXtBlock",
    "CausalConv1d",
    "MultiScaleDiscriminator",
    "VocoderLoss",
]
