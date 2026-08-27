from __future__ import annotations

import pytest
import torch

from ml.fa_fastconformer.conformer import ConvSubsampling
from ml.fa_fastconformer.features import MelSpectrogramPreprocessor, normalize_batch


def test_conv_subsampling_supports_nemo_causal_input_dimension() -> None:
    regular = ConvSubsampling(
        subsampling_factor=8,
        feat_in=80,
        feat_out=512,
        conv_channels=256,
    )
    causal = ConvSubsampling(
        subsampling_factor=8,
        feat_in=80,
        feat_out=512,
        conv_channels=256,
        causal_downsampling=True,
    )

    assert regular.out.in_features == 2560
    assert causal.out.in_features == 2816


def test_causal_conv_subsampling_forward_matches_projection_shape() -> None:
    subsampling = ConvSubsampling(
        subsampling_factor=8,
        feat_in=80,
        feat_out=512,
        conv_channels=256,
        causal_downsampling=True,
    )
    features = torch.randn(2, 1, 101, 80)
    lengths = torch.tensor([101, 79], dtype=torch.long)

    output, output_lengths = subsampling(features, lengths)

    assert output.shape == (2, 14, 512)
    assert output_lengths.tolist() == [14, 11]


def test_nemo_na_feature_normalization_sentinel_disables_normalization() -> None:
    preprocessor = MelSpectrogramPreprocessor(features=8, normalize="NA", pad_to=0)
    waveforms = torch.randn(2, 3200)
    lengths = torch.tensor([3200, 2400], dtype=torch.long)

    features, feature_lengths = preprocessor(waveforms, lengths)

    assert preprocessor.normalize is None
    assert features.shape[:2] == (2, 8)
    assert feature_lengths.tolist() == [20, 15]
    assert torch.isfinite(features).all()


def test_all_features_normalization_matches_nemo_mode() -> None:
    features = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    lengths = torch.tensor([4, 3], dtype=torch.long)

    normalized, _, _ = normalize_batch(features, lengths, "all_features")

    assert normalized[0, :, :4].mean().item() == pytest.approx(0.0, abs=1e-6)
    assert normalized[1, :, :3].mean().item() == pytest.approx(0.0, abs=1e-6)
