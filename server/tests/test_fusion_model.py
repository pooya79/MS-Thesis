from __future__ import annotations

import torch

from ml.enhancement.enhancer import build_enhancer
from ml.fusion.model import (
    CrossAttentionFusion,
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


def test_cross_attention_fusion_starts_as_balanced_blend() -> None:
    fusion = CrossAttentionFusion(d_model=16, num_layers=2, num_heads=4)
    fusion.eval()  # disable dropout for the exact-identity check
    noisy = torch.randn(2, 5, 16)
    enhanced = torch.randn(2, 5, 16)
    fused = fusion(noisy, enhanced)
    # Identity-init cross-attn layers + zero-init gate -> exact average at start.
    assert torch.allclose(fused, 0.5 * (noisy + enhanced), atol=1e-5)
    assert fused.shape == noisy.shape


def test_cross_attention_fusion_is_default_and_learns_away_from_blend() -> None:
    fusion = build_fusion(8)
    assert isinstance(fusion, CrossAttentionFusion)
    # Push the gate hard toward the enhanced stream; with identity-init layers
    # the output should then track the enhanced view.
    with torch.no_grad():
        fusion.combine.proj_out.bias.fill_(20.0)
    noisy = torch.randn(1, 3, 8)
    enhanced = torch.randn(1, 3, 8)
    fusion.eval()
    fused = fusion(noisy, enhanced)
    assert torch.allclose(fused, enhanced, atol=1e-3)


def test_cross_attention_fusion_picks_valid_head_count() -> None:
    # 8 heads do not divide d_model=12; the block must fall back to a divisor.
    fusion = CrossAttentionFusion(d_model=12, num_heads=8)
    out = fusion(torch.randn(1, 4, 12), torch.randn(1, 4, 12))
    assert out.shape == (1, 4, 12)


def test_combine_gate_exposes_blend_weight() -> None:
    # The shared gate reports the per-channel enhanced-view weight that eval
    # measures, and the gate reproduces the forward blend exactly.
    fusion = CrossAttentionFusion(d_model=16, num_layers=2, num_heads=4)
    fusion.eval()
    noisy = torch.randn(2, 5, 16)
    enhanced = torch.randn(2, 5, 16)
    # Refined streams are what the gate actually sees inside forward; at the
    # balanced init the cross-attn layers are the identity, so feed the raw views.
    gate = fusion.combine.gate(noisy, enhanced)
    assert gate.shape == noisy.shape
    assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)
    # Zero-init gate -> 0.5 everywhere -> balanced average of the two views.
    assert torch.allclose(gate, torch.full_like(gate, 0.5), atol=1e-6)
    blended = gate * enhanced + (1.0 - gate) * noisy
    assert torch.allclose(blended, fusion.combine(noisy, enhanced), atol=1e-6)


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


def test_gate_override_pins_the_blend_to_a_constant() -> None:
    # The ablation knob replaces the learned gate with a fixed weight so 0.0 yields
    # the noisy stream, 1.0 the enhanced stream, and 0.5 the balanced average —
    # exactly what eval_fusion's --gate-override relies on to isolate the gate.
    fusion = CrossAttentionFusion(d_model=16)
    noisy = torch.randn(2, 5, 16)
    enhanced = torch.randn(2, 5, 16)
    # Identity-init cross-attn layers leave the streams untouched, so the override
    # acts directly on (noisy, enhanced).
    assert torch.allclose(fusion(noisy, enhanced, gate_override=0.0), noisy, atol=1e-5)
    assert torch.allclose(fusion(noisy, enhanced, gate_override=1.0), enhanced, atol=1e-5)
    assert torch.allclose(fusion(noisy, enhanced, gate_override=0.5), 0.5 * (noisy + enhanced), atol=1e-5)


def test_encode_views_modes_select_the_expected_stream() -> None:
    whisper = _tiny_whisper()
    model = DualViewFusionModel(
        enhancer=build_enhancer({"base_channels": 8, "depth": 2}),
        whisper=whisper,
        fusion=build_fusion(int(whisper.config.d_model)),
    ).eval()
    noisy_mel = torch.randn(2, 80, 100)

    # "noisy" skips the enhancer entirely (no enhanced_mel) and decodes the raw view.
    enhanced_mel, noisy_h = model.encode_views(noisy_mel, view_mode="noisy")
    assert enhanced_mel is None
    expected_noisy = model.encoder(noisy_mel).last_hidden_state
    assert torch.allclose(noisy_h, expected_noisy, atol=1e-5)

    # "enhanced" is the enhancer-alone path: encoder hidden states of the enhanced mel.
    enhanced_mel, enhanced_h = model.encode_views(noisy_mel, view_mode="enhanced")
    assert enhanced_mel is not None and enhanced_mel.shape == noisy_mel.shape
    expected_enhanced = model.encoder(model.enhancer(noisy_mel)).last_hidden_state
    assert torch.allclose(enhanced_h, expected_enhanced, atol=1e-5)


def test_encode_views_rejects_gate_override_outside_fusion() -> None:
    import pytest

    whisper = _tiny_whisper()
    model = DualViewFusionModel(
        enhancer=build_enhancer({"base_channels": 8, "depth": 2}),
        whisper=whisper,
        fusion=build_fusion(int(whisper.config.d_model)),
    )
    with pytest.raises(ValueError):
        model.encode_views(torch.randn(1, 80, 100), view_mode="noisy", gate_override=0.0)


def _eval_config(**eval_overrides):
    import copy

    from ml.fusion.eval_fusion import DEFAULT_EVAL_CONFIG

    config = copy.deepcopy(DEFAULT_EVAL_CONFIG)
    config["model"]["checkpoint"] = "fusion_model.pt"
    config["eval"].update(eval_overrides)
    return config


def test_eval_config_accepts_ablation_knobs() -> None:
    from ml.fusion.eval_fusion import validate_eval_config

    validate_eval_config(_eval_config(view_mode="noisy"))
    validate_eval_config(_eval_config(view_mode="enhanced"))
    validate_eval_config(_eval_config(view_mode="fusion", gate_override=0.0))
    validate_eval_config(_eval_config(view_mode="fusion", gate_override=1.0))


def test_eval_config_rejects_bad_ablation_knobs() -> None:
    import pytest

    from ml.fusion.eval_fusion import validate_eval_config

    with pytest.raises(ValueError):
        validate_eval_config(_eval_config(view_mode="bogus"))
    with pytest.raises(ValueError):  # gate override only valid in fusion mode
        validate_eval_config(_eval_config(view_mode="noisy", gate_override=0.5))
    with pytest.raises(ValueError):  # outside [0, 1]
        validate_eval_config(_eval_config(gate_override=1.5))
