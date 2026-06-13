from __future__ import annotations

import pytest
import torch

from ml.enhancement.enhancer import (
    ResidualUNetEnhancer,
    build_enhancer,
    enhancement_l1_loss,
)


@pytest.mark.parametrize("time_frames", [3000, 400, 401])
def test_enhancer_preserves_shape(time_frames: int) -> None:
    model = build_enhancer({"type": "residual_unet", "base_channels": 16, "depth": 3})
    mel = torch.randn(2, 80, time_frames)
    out = model(mel)
    assert out.shape == mel.shape
    assert torch.isfinite(out).all()


def test_enhancer_starts_as_identity() -> None:
    # out_proj is zero-initialised, so a fresh residual enhancer returns input.
    model = ResidualUNetEnhancer(base_channels=16, depth=2)
    model.eval()
    mel = torch.randn(1, 80, 256)
    with torch.no_grad():
        out = model(mel)
    assert torch.allclose(out, mel, atol=1e-6)


def test_enhancer_non_residual_changes_input() -> None:
    model = ResidualUNetEnhancer(base_channels=8, depth=2, residual=False)
    mel = torch.randn(1, 80, 128)
    out = model(mel)
    assert out.shape == mel.shape
    assert not torch.allclose(out, mel)


def test_enhancer_rejects_wrong_rank() -> None:
    model = build_enhancer()
    with pytest.raises(ValueError):
        model(torch.randn(2, 80))


def test_build_enhancer_unknown_type() -> None:
    with pytest.raises(ValueError):
        build_enhancer({"type": "does_not_exist"})


def test_enhancement_l1_loss_finite_and_zero_on_match() -> None:
    clean = torch.randn(2, 80, 200)
    assert enhancement_l1_loss(clean, clean).item() == pytest.approx(0.0)
    loss = enhancement_l1_loss(clean + 1.0, clean)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(1.0, abs=1e-5)


def test_enhancer_overfits_tiny_batch() -> None:
    # A few steps should drive L_enh down on a single fixed pair (Milestone 3).
    torch.manual_seed(0)
    model = ResidualUNetEnhancer(base_channels=16, depth=2)
    noisy = torch.randn(1, 80, 128)
    clean = torch.randn(1, 80, 128)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = enhancement_l1_loss(model(noisy), clean).item()
    for _ in range(30):
        opt.zero_grad()
        loss = enhancement_l1_loss(model(noisy), clean)
        loss.backward()
        opt.step()
    assert loss.item() < first
