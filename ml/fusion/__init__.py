"""Dual-view fusion stack and the single staged training orchestrator (Phase 5).

The curriculum (D8/D10) lives in one config-driven script, ``train_fusion.py``,
which runs Stage 0 (enhancer warm-up) -> Stage 1 (enhancer + fusion, Whisper
frozen) -> Stage 2 (joint end-to-end) in one invocation, checkpointing at every
stage boundary and supporting resume-from-stage.

The fusion model itself (``model.py``) performs **encoder-feature-space** fusion
(D4): the noisy and enhanced log-Mels are each encoded by the shared Whisper
encoder and their hidden states are blended by a gated ``GatedFusion`` block
before the decoder.
"""

from ml.fusion.model import (
    DualViewFusionModel,
    GatedFusion,
    build_fusion,
    build_fusion_model,
    load_whisper_backbone,
)

__all__ = [
    "DualViewFusionModel",
    "GatedFusion",
    "build_fusion",
    "build_fusion_model",
    "load_whisper_backbone",
]
