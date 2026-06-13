"""Encoder-feature-space dual-view fusion stack (D4).

The thesis contribution fuses two *views* of the same utterance in the Whisper
encoder's hidden-state space, deliberately **not** at the Mel input (early
Mel-channel fusion is the design D4 rejects as "lost completely"):

1. the **noisy** log-Mel ``[B, 80, T]`` straight from the channel, and
2. the **enhanced** log-Mel produced by the enhancer ``E`` (``ml/enhancement``).

Both are pushed through the *same* (shared-weight) Whisper encoder, yielding two
hidden-state streams ``[B, T_enc, D]``. A lightweight ``FusionBlock`` combines
them into one fused stream that is handed to the Whisper decoder via
``encoder_outputs`` for the ASR objective. Because the encoder is shared and the
fusion is a small gated mixer, the only genuinely new parameters are the enhancer
and the fusion block; the backbone is reused.

The fusion block is zero-initialised to a *balanced blend* (gate ``= 0.5``) so the
model starts as the mean of the two encodings and never destabilises the frozen
backbone when Stage 1 begins — mirroring the enhancer's identity-init property.

Staged use (D8), driven by ``ml/fusion/train_fusion.py``:

- Stage 1 (fusion): train ``E`` + fusion, Whisper frozen  -> ``freeze_backbone()``.
- Stage 2 (joint):  train ``E`` + fusion + Whisper         -> ``unfreeze_backbone()``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class GatedFusion(nn.Module):
    """Per-channel gated blend of the noisy and enhanced encoder streams.

    A gate ``g = sigmoid(MLP([noisy_h, enhanced_h]))`` of shape ``[B, T, D]``
    selects, per time-step and per feature, how much of the enhanced encoding to
    trust versus the noisy one::

        fused = g * enhanced_h + (1 - g) * noisy_h

    The gate MLP's output layer is zero-initialised so the gate starts at exactly
    ``0.5`` everywhere (``sigmoid(0)``), i.e. a balanced average of the two views.
    Training then learns where the enhanced stream is reliable and where the raw
    noisy stream carries detail the enhancer over-smoothed.
    """

    def __init__(self, d_model: int, hidden_ratio: float = 1.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = max(1, int(round(d_model * hidden_ratio)))
        self.proj_in = nn.Linear(2 * d_model, hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(hidden, d_model)
        # Zero-init the gate output -> sigmoid(0) = 0.5 -> balanced blend at start.
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, noisy_h: torch.Tensor, enhanced_h: torch.Tensor) -> torch.Tensor:
        gate_logits = self.proj_out(self.dropout(self.act(self.proj_in(torch.cat([noisy_h, enhanced_h], dim=-1)))))
        gate = torch.sigmoid(gate_logits)
        return gate * enhanced_h + (1.0 - gate) * noisy_h


_FUSIONS: dict[str, type[nn.Module]] = {
    "gated": GatedFusion,
}


def build_fusion(d_model: int, config: dict[str, Any] | None = None) -> nn.Module:
    """Build a fusion block from a config mapping.

    ``config["type"]`` selects the architecture (default ``gated``); remaining
    keys are passed to the constructor. Factory-shaped like ``build_enhancer`` so
    the staged trainer can swap fusion strategies by config alone.
    """
    config = dict(config or {})
    arch = str(config.pop("type", "gated"))
    if arch not in _FUSIONS:
        raise ValueError(f"unknown fusion type {arch!r}; available: {sorted(_FUSIONS)}")
    return _FUSIONS[arch](d_model=d_model, **config)


class DualViewFusionModel(nn.Module):
    """Enhancer + shared Whisper encoder + fusion + Whisper decoder, end to end.

    ``forward(noisy_mel, labels)`` returns a dict with the ASR loss (``loss``),
    the decoder ``logits``, the enhanced log-Mel (so the trainer can add the
    auxiliary ``L_enh`` against the clean target), and the fused encoder stream.
    """

    def __init__(self, enhancer: nn.Module, whisper: Any, fusion: nn.Module) -> None:
        super().__init__()
        self.enhancer = enhancer
        self.whisper = whisper
        self.fusion = fusion

    @property
    def encoder(self) -> Any:
        return self.whisper.get_encoder()

    def freeze_backbone(self) -> None:
        """Stage 1: freeze the whole Whisper backbone (encoder + decoder)."""
        for param in self.whisper.parameters():
            param.requires_grad_(False)
        self.whisper.eval()

    def unfreeze_backbone(self) -> None:
        """Stage 2: make the Whisper backbone trainable again."""
        for param in self.whisper.parameters():
            param.requires_grad_(True)
        self.whisper.train()

    def encode_views(self, noisy_mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(enhanced_mel, fused_hidden_states)`` for a noisy log-Mel batch."""
        enhanced_mel = self.enhancer(noisy_mel)
        noisy_h = self.encoder(noisy_mel).last_hidden_state
        enhanced_h = self.encoder(enhanced_mel).last_hidden_state
        fused = self.fusion(noisy_h, enhanced_h)
        return enhanced_mel, fused

    def forward(self, noisy_mel: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, Any]:
        from transformers.modeling_outputs import BaseModelOutput

        enhanced_mel, fused = self.encode_views(noisy_mel)
        outputs = self.whisper(
            encoder_outputs=BaseModelOutput(last_hidden_state=fused),
            labels=labels,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "enhanced_mel": enhanced_mel,
            "encoder_hidden_states": fused,
        }


def load_whisper_backbone(checkpoint: str, model_name: str = "openai/whisper-small") -> Any:
    """Load the fine-tuned Persian Whisper backbone (Phase 1), falling back to base.

    ``checkpoint`` is normally the fine-tuned run dir (e.g.
    ``models/asr/whisper-small/runs/best``); if it is missing or unset we fall
    back to ``model_name`` so the stack can still be exercised.
    """
    from pathlib import Path

    from transformers import WhisperForConditionalGeneration

    source = checkpoint if (checkpoint and Path(checkpoint).exists()) else model_name
    return WhisperForConditionalGeneration.from_pretrained(source)


def build_fusion_model(
    config: dict[str, Any],
    *,
    enhancer: nn.Module | None = None,
    whisper: Any = None,
) -> DualViewFusionModel:
    """Assemble a :class:`DualViewFusionModel` from a fusion-training config.

    Reuses an already-built ``enhancer`` (e.g. warmed up in Stage 0) and/or a
    preloaded ``whisper`` backbone when provided; otherwise builds them from
    ``config`` (``enhancer`` block / ``base_asr_checkpoint`` + ``model_name``).
    The fusion block is sized from the backbone's ``d_model``.
    """
    from ml.enhancement.enhancer import build_enhancer

    if enhancer is None:
        enhancer = build_enhancer(config.get("enhancer"))
    if whisper is None:
        whisper = load_whisper_backbone(
            str(config.get("base_asr_checkpoint") or ""),
            model_name=str(config.get("model_name", "openai/whisper-small")),
        )
    fusion = build_fusion(int(whisper.config.d_model), config.get("fusion"))
    return DualViewFusionModel(enhancer=enhancer, whisper=whisper, fusion=fusion)
