"""Independent implementation of Cui et al. (ICASSP 2025), arXiv:2501.02452.

The bridge consumes filterbanks but mixes *waveforms*. See the script guide for
paper details that are underspecified and our explicit implementation choices.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


OA_COEFFICIENTS = tuple(i / 10 for i in range(10, -1, -1))


def observation_addition(noisy: torch.Tensor, enhanced: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Eq. 3: omega weights the ORIGINAL waveform (opposite to our fusion gate)."""
    if noisy.shape != enhanced.shape:
        raise ValueError("waveforms must have identical shapes and be time aligned")
    if not torch.isfinite(omega).all() or ((omega < 0) | (omega > 1)).any():
        raise ValueError("omega must be finite and in [0, 1]")
    while omega.ndim < noisy.ndim:
        omega = omega.unsqueeze(-1)
    return omega * noisy + (1 - omega) * enhanced


def recognition_information_loss(logits: torch.Tensor, wers: torch.Tensor) -> torch.Tensor:
    """Eq. 6, including both sigmoid operations; WER uses ratios, not percent."""
    if logits.shape != wers.shape or not torch.isfinite(wers).all() or (wers < 0).any():
        raise ValueError("finite nonnegative WER targets must match logits")
    similarity = F.cosine_similarity(logits.sigmoid(), wers.detach().sigmoid(), dim=-1)
    return -F.logsigmoid(similarity).mean()


def perceptual_target(sig: torch.Tensor, bak: torch.Tensor) -> torch.Tensor:
    """Eq. 5: DNSMOS SIG/BAK of original noisy audio, in the range [1, 5]."""
    for value in (sig, bak):
        if not torch.isfinite(value).all() or ((value < 1) | (value > 5)).any():
            raise ValueError("DNSMOS scores must be finite and in [1, 5]")
    return ((sig - 1) + (bak - 1)) / 8


def bridge_loss(outputs: dict[str, torch.Tensor], wers: torch.Tensor,
                sig: torch.Tensor, bak: torch.Tensor, *, recognition: bool = True) -> torch.Tensor:
    pq = F.mse_loss(outputs["omega"], perceptual_target(sig, bak).detach())
    return (pq + recognition_information_loss(outputs["logits"], wers)) / 2 if recognition else pq


def conv(channels_in: int, channels_out: int, kernel: int, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv1d(channels_in, channels_out, kernel,
                                  padding=dilation * (kernel - 1) // 2, dilation=dilation),
                         nn.ReLU(), nn.BatchNorm1d(channels_out))


class SERes2Block(nn.Module):
    def __init__(self, channels: int, dilation: int, scale: int = 8, bottleneck: int = 256):
        super().__init__()
        if channels % scale:
            raise ValueError("channels must be divisible by Res2 scale")
        width = channels // scale
        self.pre = conv(channels, channels, 1)
        self.parts = nn.ModuleList(conv(width, width, 3, dilation) for _ in range(scale - 1))
        self.post = conv(channels, channels, 1)
        self.se = nn.Sequential(nn.Conv1d(channels, bottleneck, 1), nn.ReLU(),
                                nn.Conv1d(bottleneck, channels, 1), nn.Sigmoid())

    @staticmethod
    def _mask(x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        if lengths is None:
            return x
        valid = torch.arange(x.shape[-1], device=x.device)[None, :] < lengths[:, None]
        return x.masked_fill(~valid[:, None, :], 0)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        pieces = self._mask(self.pre(x), lengths).chunk(len(self.parts) + 1, dim=1)
        values = [pieces[0]]
        for i, layer in enumerate(self.parts, start=1):
            value = layer(pieces[i] if i == 1 else pieces[i] + values[-1])
            values.append(self._mask(value, lengths))
        y = self._mask(self.post(torch.cat(values, dim=1)), lengths)
        if lengths is None:
            pooled = y.mean(dim=-1, keepdim=True)
        else:
            pooled = y.sum(dim=-1, keepdim=True) / lengths[:, None, None]
        return self._mask(x + y * self.se(pooled), lengths)


class UtteranceEncoder(nn.Module):
    """Fig. 1c; process unpadded utterances to exclude padding from BN and ASP."""
    def __init__(self, channels: int = 256, bottleneck: int = 256):
        super().__init__()
        self.front = conv(80, channels, 5)
        self.blocks = nn.ModuleList(SERes2Block(channels, d, bottleneck=bottleneck) for d in (2, 3, 4))
        self.aggregate = conv(3 * channels, channels, 1)
        self.attention = nn.Sequential(nn.Conv1d(channels, bottleneck, 1), nn.Tanh(),
                                       nn.Conv1d(bottleneck, channels, 1))

    @staticmethod
    def _mask(x: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        if lengths is None:
            return x
        valid = torch.arange(x.shape[-1], device=x.device)[None, :] < lengths[:, None]
        return x.masked_fill(~valid[:, None, :], 0)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is not None:
            if (lengths.ndim != 1 or len(lengths) != len(x) or (lengths < 2).any()
                    or (lengths > x.shape[-1]).any()):
                raise ValueError("lengths must contain one valid frame count per utterance")
        x = self.front(x)
        x = self._mask(x, lengths)
        stages = []
        for block in self.blocks:
            x = block(x, lengths)
            stages.append(x)
        x = self.aggregate(torch.cat(stages, dim=1))
        x = self._mask(x, lengths)
        attention_logits = self.attention(x)
        if lengths is not None:
            valid = torch.arange(x.shape[-1], device=x.device)[None, :] < lengths[:, None]
            attention_logits = attention_logits.masked_fill(~valid[:, None, :], -torch.inf)
        a = attention_logits.softmax(dim=-1)
        mean = (a * x).sum(dim=-1)
        std = ((a * x.square()).sum(dim=-1) - mean.square()).clamp_min(1e-6).sqrt()
        return torch.cat((mean, std), dim=-1)


class BridgingModule(nn.Module):
    """Shared utterance encoder and recognition/PQ heads; no SE/ASR parameters."""
    def __init__(self, channels: int = 256, bottleneck: int = 256, hidden: int = 384):
        super().__init__()
        self.encoder = UtteranceEncoder(channels, bottleneck)
        self.recognition = nn.Sequential(nn.Linear(4 * channels, hidden), nn.ReLU(), nn.Linear(hidden, 11))
        self.quality = nn.Linear(11, 1)

    def forward(self, noisy: torch.Tensor, enhanced: torch.Tensor,
                lengths: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if noisy.shape != enhanced.shape or noisy.ndim != 3 or noisy.shape[1] != 80 or noisy.shape[-1] < 2:
            raise ValueError("expected aligned [batch, 80, frames>=2] filterbanks")
        logits = self.recognition(torch.cat(
            (self.encoder(noisy, lengths), self.encoder(enhanced, lengths)), dim=-1,
        ))
        return {"logits": logits, "omega": self.quality(logits).sigmoid().squeeze(-1)}
