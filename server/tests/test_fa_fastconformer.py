from __future__ import annotations

import torch

from ml.fa_fastconformer.conformer import ConvSubsampling


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
