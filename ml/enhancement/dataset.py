"""Dataset over a degraded-dataset directory for log-Mel enhancement/fusion.

Consumes the output of ``ml.speech_data.generate_degraded_dataset``: a directory
containing degraded WAV clips plus a ``degraded_to_clean.jsonl`` mapping with one
row per degraded variant. Each row carries the degraded clip path, the *original*
(full-band) clean source path, the transcript, and the full per-variant
degradation metadata.

This single dataset feeds all three curriculum stages (D8):

- Stage 0 (enhancer warm-up): needs ``noisy_mel`` + ``clean_mel`` only.
- Stages 1-2 (fusion / joint): additionally need transcript ``labels``.

**Bandwidth-aligned clean target (D5).** The dataset script only saves degraded
audio and points back at the full-band clean source, so the bandwidth-aligned
clean target is *reconstructed* here from the recorded degradation metadata: for
narrowband / wideband-filtered paths we re-apply the recorded band-pass at the
channel rate and the recorded peak-safety ``normalization_scale``, mirroring what
``generate_degraded_pairs.degrade_item`` produced. This avoids penalising the
enhancer for failing to hallucinate frequencies the channel genuinely removed.
Set ``clean_target: full_band`` to skip alignment and target the raw clean.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ml.asr.whisper_features import WHISPER_SAMPLE_RATE, waveform_to_log_mel
from ml.utils.audio import bandpass_filter, load_audio, match_length, resample_audio, to_mono

MAPPING_FILENAME = "degraded_to_clean.jsonl"
FRAMES_PER_SECOND = 100  # Whisper hop: 160 samples @ 16 kHz
_BANDWIDTH_ALIGNED = {"narrowband", "wideband_filtered"}


@dataclass(frozen=True)
class DegradedPair:
    pair_id: str
    split: str
    degraded_path: Path
    clean_path: Path
    transcript: str
    degradation: dict[str, Any]


def read_mapping(dataset_dir: Path, split: str | None = None) -> list[DegradedPair]:
    """Read ``degraded_to_clean.jsonl`` into pairs, optionally filtered by split."""
    mapping_path = dataset_dir / MAPPING_FILENAME
    if not mapping_path.is_file():
        raise FileNotFoundError(
            f"{mapping_path} not found; run ml.speech_data.generate_degraded_dataset first"
        )
    pairs: list[DegradedPair] = []
    with mapping_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_split = str(row.get("split", ""))
            if split is not None and row_split != split:
                continue
            degradation = dict(row.get("degradation") or {})
            pairs.append(
                DegradedPair(
                    pair_id=str(row.get("degraded_id") or f"{row_split}-{line_number}"),
                    split=row_split,
                    degraded_path=_resolve(dataset_dir, str(row["degraded_path"])),
                    clean_path=_resolve(dataset_dir, str(row["clean_path"])),
                    transcript=str(row.get("sentence") or "").strip(),
                    degradation=degradation,
                )
            )
    if not pairs:
        scope = f" for split={split!r}" if split is not None else ""
        raise ValueError(f"no degraded pairs found in {mapping_path}{scope}")
    return pairs


def _resolve(dataset_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (dataset_dir / path)


def reconstruct_clean_target(
    clean_audio: np.ndarray,
    source_rate: int,
    degradation: dict[str, Any],
    target_length: int,
    *,
    mode: str = "bandwidth_aligned",
    model_rate: int = WHISPER_SAMPLE_RATE,
) -> np.ndarray:
    """Rebuild the D5 bandwidth-aligned clean waveform target at ``model_rate``.

    ``target_length`` is the degraded waveform length (samples at ``model_rate``);
    the target is length-matched to it so their log-Mels align frame-for-frame.
    """
    clean = to_mono(np.asarray(clean_audio, dtype=np.float32))
    target_bandwidth = str(degradation.get("target_bandwidth", "wideband"))
    if mode == "bandwidth_aligned" and target_bandwidth in _BANDWIDTH_ALIGNED:
        channel_rate = int(degradation["channel_sample_rate"])
        low_hz, high_hz = degradation["channel_bandpass_hz"]
        channel = resample_audio(clean, source_rate, channel_rate)
        channel = bandpass_filter(channel, channel_rate, float(low_hz), float(high_hz))
        target = resample_audio(channel, channel_rate, model_rate)
    else:
        target = resample_audio(clean, source_rate, model_rate)
    target = match_length(target, target_length)
    scale = degradation.get("normalization_scale")
    if scale is not None:
        target = target * float(scale)
    return np.asarray(target, dtype=np.float32)


class DegradedMelDataset(Dataset):
    """Yields noisy/clean log-Mels (and optional transcript labels) per variant."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        split: str | None = None,
        clean_target: str = "bandwidth_aligned",
        segment_seconds: float | None = None,
        model_name: str = "openai/whisper-small",
        return_labels: bool = False,
        tokenizer: Any = None,
        seed: int = 1337,
    ) -> None:
        if clean_target not in {"bandwidth_aligned", "full_band"}:
            raise ValueError("clean_target must be 'bandwidth_aligned' or 'full_band'")
        if return_labels and tokenizer is None:
            raise ValueError("return_labels=True requires a tokenizer")
        self.dataset_dir = Path(dataset_dir)
        self.pairs = read_mapping(self.dataset_dir, split)
        self.clean_target = clean_target
        self.model_name = model_name
        self.return_labels = return_labels
        self.tokenizer = tokenizer
        self.segment_frames = (
            int(round(segment_seconds * FRAMES_PER_SECOND)) if segment_seconds else None
        )
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        degraded_audio, degraded_rate = load_audio(pair.degraded_path)
        degraded_audio = to_mono(np.asarray(degraded_audio, dtype=np.float32))
        model_rate = int(pair.degradation.get("model_sample_rate", degraded_rate))
        if degraded_rate != model_rate:
            degraded_audio = resample_audio(degraded_audio, degraded_rate, model_rate)

        clean_source, clean_rate = load_audio(pair.clean_path)
        clean_audio = reconstruct_clean_target(
            clean_source,
            clean_rate,
            pair.degradation,
            target_length=len(degraded_audio),
            mode=self.clean_target,
            model_rate=model_rate,
        )

        noisy_mel = waveform_to_log_mel(degraded_audio, sample_rate=model_rate, model_name=self.model_name)
        clean_mel = waveform_to_log_mel(clean_audio, sample_rate=model_rate, model_name=self.model_name)
        noisy_mel, clean_mel = self._maybe_crop(noisy_mel, clean_mel)

        item: dict[str, Any] = {
            "pair_id": pair.pair_id,
            "noisy_mel": noisy_mel,
            "clean_mel": clean_mel,
        }
        if self.return_labels:
            item["labels"] = self.tokenizer(pair.transcript).input_ids
        return item

    def _maybe_crop(
        self, noisy_mel: torch.Tensor, clean_mel: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.segment_frames is None:
            return noisy_mel, clean_mel
        total = noisy_mel.shape[-1]
        if total <= self.segment_frames:
            return noisy_mel, clean_mel
        start = int(self._rng.integers(0, total - self.segment_frames + 1))
        end = start + self.segment_frames
        return noisy_mel[..., start:end], clean_mel[..., start:end]


def collate_mels(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate fixed-width log-Mels; pads ``labels`` with -100 when present."""
    collated: dict[str, Any] = {
        "pair_id": [item["pair_id"] for item in batch],
        "noisy_mel": torch.stack([item["noisy_mel"] for item in batch]),
        "clean_mel": torch.stack([item["clean_mel"] for item in batch]),
    }
    if "labels" in batch[0]:
        max_len = max(len(item["labels"]) for item in batch)
        padded = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for row, item in enumerate(batch):
            label = torch.as_tensor(item["labels"], dtype=torch.long)
            padded[row, : label.numel()] = label
        collated["labels"] = padded
    return collated
