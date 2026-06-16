"""Residual 2D-conv U-Net speech-enhancement module ``E``.

``E`` treats the noisy log-Mel ``[B, 80, T]`` as a single-channel image and
predicts a *residual* to the clean log-Mel, so it initialises close to the
identity and never feeds garbage into the backbone during early training
(D8 Stage 0). The architecture is deliberately lightweight to fit alongside
Whisper-small on an RTX 3090 (D3); depth and width are config-driven so the
parameter budget stays tunable while the interface and objective are fixed.

Interface contract (Phase 4): input and output are both ``[B, 80, T]`` log-Mel
tensors with identical shape. ``T`` is arbitrary (4 s crops for the standalone
warm-up, 3000 for the full Whisper window), so the net pads internally to keep
the output time/mel dimensions exactly equal to the input.

Auxiliary objective (D5): L1 in the log-Mel domain against the
bandwidth-aligned clean target.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def enhancement_l1_loss(
    enhanced_mel: torch.Tensor,
    clean_mel: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """L_enh: L1 loss in the log-Mel domain (D5).

    L1 (not MSE) is more robust to outliers and over-smooths less, preserving
    the low-energy phonetic detail that matters for recognition. ``clean_mel``
    must be the bandwidth-aligned clean reference from the pair manifest.
    """
    return F.l1_loss(enhanced_mel, clean_mel, reduction=reduction)


class _ConvBlock(nn.Module):
    """Two 3x3 convs with GroupNorm + GELU; preserves spatial size."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__()
        norm_groups = _safe_groups(groups, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _safe_groups(groups: int, channels: int) -> int:
    """Largest divisor of ``channels`` that is <= ``groups`` (>=1)."""
    g = min(groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(1, g)


def _valid_heads(dim: int, requested: int) -> int:
    """Largest divisor of ``dim`` that is ``<= requested`` (``>= 1``)."""
    heads = min(max(1, requested), dim)
    while heads > 1 and dim % heads != 0:
        heads -= 1
    return heads


def _downsampled_size(size: int, depth: int) -> int:
    """Size after ``depth`` stride-2, kernel-3, pad-1 convs (the U-Net down path)."""
    for _ in range(depth):
        size = (size + 2 * 1 - 3) // 2 + 1
    return size


class TemporalBottleneck(nn.Module):
    """Sequence model over the time axis at the U-Net bottleneck.

    The conv U-Net only sees a few hundred ms of temporal context, which is too
    little for noise/reverb suppression. This block adds long-range temporal
    modelling where the feature map is smallest (cheapest): the ``[B, C, F, T]``
    bottleneck is flattened to a ``[B, T, C*F]`` sequence, projected to ``dim``,
    passed through ``layers`` of a transformer encoder (or a bidirectional GRU),
    projected back, and added as a **residual**. The output projection is
    zero-initialised so the block starts as the exact identity — preserving the
    enhancer's identity-init property (D8) so it never destabilises early training.
    """

    def __init__(
        self,
        channels: int,
        freq: int,
        *,
        kind: str = "transformer",
        layers: int = 2,
        heads: int = 4,
        dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kind not in {"transformer", "gru"}:
            raise ValueError(f"unknown bottleneck kind {kind!r}; expected 'transformer' or 'gru'")
        if layers < 1:
            raise ValueError("bottleneck layers must be >= 1")
        self.kind = kind
        self.in_dim = int(channels) * int(freq)
        self.proj_in = nn.Linear(self.in_dim, dim)
        if kind == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=_valid_heads(dim, heads),
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            # norm_first layers can't use the nested-tensor fast path; disable it
            # explicitly to skip the spurious warning.
            self.seq = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
            seq_out = dim
        else:
            self.seq = nn.GRU(
                dim,
                dim,
                num_layers=layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            seq_out = dim * 2
        # Zero-init the output projection -> residual is 0 at start -> identity.
        self.proj_out = nn.Linear(seq_out, self.in_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, C, F, T]`` -> ``[B, C, F, T]`` with temporal context mixed in."""
        b, c, f, t = x.shape
        if c * f != self.in_dim:
            raise ValueError(
                f"TemporalBottleneck expected C*F={self.in_dim} at the bottleneck, got C={c} F={f}; "
                "check base_channels/depth match the configured bottleneck."
            )
        seq = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # [B, T, C*F]
        h = self.proj_in(seq)
        if self.kind == "gru":
            h, _ = self.seq(h)
        else:
            h = self.seq(h)
        seq = seq + self.proj_out(h)  # residual; proj_out is zero-init -> identity at start
        return seq.reshape(b, t, c, f).permute(0, 2, 3, 1).contiguous()


class ResidualUNetEnhancer(nn.Module):
    """Lightweight residual 2D-conv U-Net mapping noisy -> clean log-Mel.

    The net operates on ``[B, 1, n_mels, T]`` (the log-Mel as an image),
    down/upsamples ``depth`` times with stride-2 convs and skip connections,
    and adds its output as a residual to the input log-Mel.

    ``bottleneck`` optionally inserts a :class:`TemporalBottleneck` (``"transformer"``
    or ``"gru"``) at the deepest feature map to add the long-range temporal context
    the purely-convolutional path lacks; ``"none"`` (default) keeps the original
    lightweight net. The bottleneck is identity-initialised, so enabling it does not
    change the identity-init behaviour Stage 0 relies on.
    """

    def __init__(
        self,
        n_mels: int = 80,
        base_channels: int = 32,
        depth: int = 3,
        groups: int = 8,
        residual: bool = True,
        bottleneck: str = "none",
        bottleneck_layers: int = 2,
        bottleneck_heads: int = 4,
        bottleneck_dim: int = 256,
        bottleneck_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.n_mels = n_mels
        self.residual = residual

        self.in_proj = _ConvBlock(1, base_channels, groups)

        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        channels = base_channels
        for _ in range(depth):
            out_channels = channels * 2
            self.downsamplers.append(
                nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=1)
            )
            self.down_blocks.append(_ConvBlock(out_channels, out_channels, groups))
            channels = out_channels

        # Optional long-range temporal model at the bottleneck (identity at init).
        # `channels` is now the bottleneck width; the freq axis is downsampled
        # `depth` times by the stride-2 convs above.
        if bottleneck in {None, "none"}:
            self.bottleneck: nn.Module = nn.Identity()
        else:
            self.bottleneck = TemporalBottleneck(
                channels,
                _downsampled_size(n_mels, depth),
                kind=str(bottleneck),
                layers=int(bottleneck_layers),
                heads=int(bottleneck_heads),
                dim=int(bottleneck_dim),
                dropout=float(bottleneck_dropout),
            )

        self.up_samplers = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for _ in range(depth):
            out_channels = channels // 2
            self.up_samplers.append(
                nn.ConvTranspose2d(channels, out_channels, kernel_size=2, stride=2)
            )
            # skip concat doubles the channel count entering the block
            self.up_blocks.append(_ConvBlock(out_channels * 2, out_channels, groups))
            channels = out_channels

        # Zero-init the output conv so E starts as the identity (residual=0).
        self.out_proj = nn.Conv2d(base_channels, 1, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """``mel``: ``[B, n_mels, T]`` -> enhanced ``[B, n_mels, T]``."""
        if mel.dim() != 3:
            raise ValueError(f"expected [B, n_mels, T], got shape {tuple(mel.shape)}")
        x = mel.unsqueeze(1)  # [B, 1, n_mels, T]

        x = self.in_proj(x)
        skips: list[torch.Tensor] = []
        for downsampler, block in zip(self.downsamplers, self.down_blocks):
            skips.append(x)
            x = block(downsampler(x))

        x = self.bottleneck(x)

        for upsampler, block, skip in zip(
            self.up_samplers, self.up_blocks, reversed(skips)
        ):
            x = upsampler(x)
            x = _match_size(x, skip)
            x = block(torch.cat([x, skip], dim=1))

        delta = self.out_proj(x).squeeze(1)  # [B, n_mels, T]
        delta = _match_size_2d(delta, mel)
        return mel + delta if self.residual else delta


def _match_size(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Crop/pad ``x`` (``[B, C, H, W]``) to ``reference``'s spatial size."""
    return _resize_to(x, reference.shape[-2], reference.shape[-1])


def _match_size_2d(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Crop/pad ``x`` (``[B, H, W]``) to ``reference``'s last two dims."""
    expanded = _resize_to(x.unsqueeze(1), reference.shape[-2], reference.shape[-1])
    return expanded.squeeze(1)


def _resize_to(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Center-agnostic crop/zero-pad of ``x``'s last two dims to (H, W)."""
    h, w = x.shape[-2], x.shape[-1]
    if h > height:
        x = x[..., :height, :]
    if w > width:
        x = x[..., :width]
    pad_h = max(0, height - x.shape[-2])
    pad_w = max(0, width - x.shape[-1])
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x


_ENHANCERS = {
    "residual_unet": ResidualUNetEnhancer,
}


def build_enhancer(config: dict[str, Any] | None = None) -> nn.Module:
    """Build an enhancer from a config mapping.

    ``config["type"]`` selects the architecture (default ``residual_unet``);
    remaining keys are passed to the module constructor. Keeping this a factory
    lets the staged trainer swap architectures by config without touching code.
    """
    config = dict(config or {})
    arch = str(config.pop("type", "residual_unet"))
    if arch not in _ENHANCERS:
        raise ValueError(
            f"unknown enhancer type {arch!r}; available: {sorted(_ENHANCERS)}"
        )
    return _ENHANCERS[arch](**config)
