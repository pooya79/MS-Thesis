"""Canonical Whisper log-Mel feature extraction.

This is the single feature path shared by the ASR backbone, the enhancement
module ``E`` (``ml/enhancement``), and the dual-view fusion stack
(``ml/fusion``). All three must see identical log-Mel statistics, so feature
computation lives here rather than being re-derived per module.

Whisper-small consumes an 80-bin log-Mel padded to a fixed 30 s window
(``T = 3000`` frames at 16 kHz). The enhancer and fusion operate on exactly
this ``[B, 80, 3000]`` tensor, since the fused result is fed straight to the
encoder. A shorter ``segment_seconds`` crop is only meaningful for the
standalone enhancer warm-up (Stage 0); anything that reaches Whisper must keep
the full padded width.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import torch

WHISPER_SAMPLE_RATE = 16000
WHISPER_NUM_MELS = 80
WHISPER_NUM_FRAMES = 3000


@lru_cache(maxsize=4)
def get_feature_extractor(model_name: str = "openai/whisper-small") -> Any:
    """Return a cached Whisper feature extractor for ``model_name``.

    ``from_pretrained`` revalidates the HF cache over the network in every
    fresh process (the lru_cache above is per-process, so each DataLoader worker
    pays this on first use). We prefer the local cache (``local_files_only``) to
    avoid those repeated HEAD requests, falling back to a networked fetch only
    when the model is not cached yet.
    """
    from transformers import WhisperFeatureExtractor

    try:
        return WhisperFeatureExtractor.from_pretrained(model_name, local_files_only=True)
    except OSError:
        return WhisperFeatureExtractor.from_pretrained(model_name)


def waveform_to_log_mel(
    waveform: np.ndarray | torch.Tensor,
    *,
    sample_rate: int = WHISPER_SAMPLE_RATE,
    model_name: str = "openai/whisper-small",
) -> torch.Tensor:
    """Convert a mono waveform to a Whisper log-Mel of shape ``[80, 3000]``.

    The input is expected to already be mono at ``sample_rate``; resampling and
    channel-mixing are the caller's responsibility (see ``ml.utils.audio``).
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError(f"expected a mono 1-D waveform, got shape {waveform.shape}")
    extractor = get_feature_extractor(model_name)
    features = extractor(
        waveform,
        sampling_rate=int(sample_rate),
        return_tensors="pt",
    ).input_features[0]
    return features.to(torch.float32)


def batch_waveforms_to_log_mel(
    waveforms: list[np.ndarray | torch.Tensor],
    *,
    sample_rate: int = WHISPER_SAMPLE_RATE,
    model_name: str = "openai/whisper-small",
) -> torch.Tensor:
    """Stack several mono waveforms into a ``[B, 80, 3000]`` log-Mel batch."""
    return torch.stack(
        [
            waveform_to_log_mel(waveform, sample_rate=sample_rate, model_name=model_name)
            for waveform in waveforms
        ]
    )
