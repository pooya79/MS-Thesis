from __future__ import annotations

import torch

from ml.enhancement.enhancer import build_enhancer
from ml.fusion.model import (
    DualViewFusionModel,
    GatedFusion,
    build_fusion,
    build_fusion_model,
)


def _tiny_whisper():
    """A randomly-initialised Whisper just big enough to exercise the stack offline."""
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    config = WhisperConfig(
        vocab_size=64,
        num_mel_bins=80,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_source_positions=50,  # encoder conv halves the 100-frame input
        max_target_positions=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    return WhisperForConditionalGeneration(config)


def test_gated_fusion_starts_as_balanced_blend() -> None:
    fusion = GatedFusion(d_model=16)
    noisy = torch.randn(2, 5, 16)
    enhanced = torch.randn(2, 5, 16)
    fused = fusion(noisy, enhanced)
    # Zero-init gate -> sigmoid(0)=0.5 -> exact average of the two views.
    assert torch.allclose(fused, 0.5 * (noisy + enhanced), atol=1e-6)
    assert fused.shape == noisy.shape


def test_gated_fusion_can_learn_away_from_balance() -> None:
    fusion = GatedFusion(d_model=8)
    with torch.no_grad():
        fusion.proj_out.bias.fill_(20.0)  # gate -> ~1 -> pass enhanced through
    noisy = torch.randn(1, 3, 8)
    enhanced = torch.randn(1, 3, 8)
    fused = fusion(noisy, enhanced)
    assert torch.allclose(fused, enhanced, atol=1e-3)


def test_build_fusion_factory_rejects_unknown_type() -> None:
    import pytest

    with pytest.raises(ValueError):
        build_fusion(16, {"type": "does-not-exist"})


def test_dual_view_model_forward_produces_loss_and_enhanced_mel() -> None:
    whisper = _tiny_whisper()
    enhancer = build_enhancer({"base_channels": 8, "depth": 2})
    fusion = build_fusion(int(whisper.config.d_model))
    model = DualViewFusionModel(enhancer=enhancer, whisper=whisper, fusion=fusion)

    noisy_mel = torch.randn(2, 80, 100)
    labels = torch.tensor([[1, 5, 2, -100], [1, 7, 8, 2]])
    out = model(noisy_mel, labels=labels)

    assert out["enhanced_mel"].shape == noisy_mel.shape
    assert out["loss"].requires_grad
    assert torch.isfinite(out["loss"])
    # The fused stream is in encoder feature space: [B, T_enc, d_model].
    assert out["encoder_hidden_states"].shape[0] == 2
    assert out["encoder_hidden_states"].shape[-1] == whisper.config.d_model


def test_freeze_and_unfreeze_backbone_toggles_whisper_grads() -> None:
    whisper = _tiny_whisper()
    model = DualViewFusionModel(
        enhancer=build_enhancer({"base_channels": 8, "depth": 2}),
        whisper=whisper,
        fusion=build_fusion(int(whisper.config.d_model)),
    )

    model.freeze_backbone()
    assert all(not p.requires_grad for p in model.whisper.parameters())
    # Enhancer + fusion stay trainable (Stage 1 trains the front end only).
    assert any(p.requires_grad for p in model.enhancer.parameters())
    assert any(p.requires_grad for p in model.fusion.parameters())

    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.whisper.parameters())


def test_build_fusion_model_reuses_passed_components() -> None:
    whisper = _tiny_whisper()
    enhancer = build_enhancer({"base_channels": 8, "depth": 2})
    model = build_fusion_model({}, enhancer=enhancer, whisper=whisper)
    assert isinstance(model, DualViewFusionModel)
    assert model.enhancer is enhancer
    assert model.whisper is whisper
