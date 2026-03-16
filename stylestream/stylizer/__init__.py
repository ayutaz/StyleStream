"""Stylizer module: DiT + CFM + Style Encoder for mel generation.

Exports
-------
Stylizer
    Full model (DiT + Style Encoder + CFM + CFG).
DiT
    Diffusion Transformer backbone.
DiTBlock
    Single DiT block with adaLN-Zero.
StyleEncoder
    WavLM-TDNN style embedding extractor.
ConditionalFlowMatching
    CFM training loss and Euler sampling.
ClassifierFreeGuidance
    CFG dropout and guided velocity.
"""

from stylestream.stylizer.model import Stylizer
from stylestream.stylizer.dit import DiT, DiTBlock
from stylestream.stylizer.style_encoder import StyleEncoder
from stylestream.stylizer.cfm import ConditionalFlowMatching
from stylestream.stylizer.cfg import ClassifierFreeGuidance

__all__ = [
    "Stylizer",
    "DiT",
    "DiTBlock",
    "StyleEncoder",
    "ConditionalFlowMatching",
    "ClassifierFreeGuidance",
]
