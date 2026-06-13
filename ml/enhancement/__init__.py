"""Mel-domain speech-enhancement module ``E`` (Phase 4).

``E`` maps a noisy Whisper log-Mel to an estimated clean log-Mel,
``[B, 80, T] -> [B, 80, T]``. It is trained for recognition (end-to-end in the
fusion stack) with an auxiliary log-Mel L1 reconstruction loss, not as a frozen
perceptual denoiser. See ``docs/speech-enhancement-and-fusion-decisions.md``.
"""

from __future__ import annotations

from ml.enhancement.enhancer import (
    ResidualUNetEnhancer,
    build_enhancer,
    enhancement_l1_loss,
)

__all__ = [
    "ResidualUNetEnhancer",
    "build_enhancer",
    "enhancement_l1_loss",
]
