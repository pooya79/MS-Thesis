"""Dataset over a degraded-dataset directory for log-Mel enhancement/fusion.

Consumes the output of ``ml.speech_data.generate_degraded_dataset``: a directory
containing degraded WAV clips plus a ``degraded_to_clean.jsonl`` mapping with one
row per degraded variant. Each row carries the degraded clip path, the *original*
(full-band) clean source path, the transcript, and the full per-variant
degradation metadata.

This single dataset feeds all three curriculum stages (D8):

- Stage 0 (enhancer warm-up): needs ``noisy_mel`` + ``clean_mel`` only.
- Stages 1-2 (fusion / joint): additionally need transcript ``labels``.

A second dataset type, :class:`CleanMelDataset`, reads a plain (non-degraded) ASR
dataset (the project split-TSV + ``clips/`` contract) and yields the *same*
item layout with the noisy and clean views set to the same clean log-Mel. It is
mixed into the joint stage only, to keep the full stack strong on clean speech.
Use :func:`detect_dataset_kind` to classify a directory as degraded or clean.

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

import csv
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
_CLEAN_SPLITS = ("train", "dev", "test")


def detect_dataset_kind(dataset_dir: str | Path) -> str:
    """Classify a dataset directory as ``"degraded"`` or ``"clean"``.

    A *degraded* dataset is a ``generate_degraded_dataset`` output (it carries a
    ``degraded_to_clean.jsonl`` mapping). A *clean* dataset is a plain ASR dataset
    that follows the project contract (split ``*.tsv`` with ``path`` + ``sentence``
    columns plus a ``clips/`` dir). The degraded mapping is checked first so a
    directory that happens to hold both is treated as degraded.
    """
    path = Path(dataset_dir)
    if (path / MAPPING_FILENAME).is_file():
        return "degraded"
    if any((path / f"{split}.tsv").is_file() for split in _CLEAN_SPLITS):
        return "clean"
    raise FileNotFoundError(
        f"{path} is neither a degraded dataset ({MAPPING_FILENAME}) nor a clean ASR "
        f"dataset (a {' / '.join(f'{s}.tsv' for s in _CLEAN_SPLITS)})"
    )


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


@dataclass(frozen=True)
class CleanClip:
    clip_id: str
    audio_path: Path
    transcript: str


def read_clean_rows(dataset_dir: Path, split: str) -> list[CleanClip]:
    """Read one split TSV of a clean ASR dataset (project ASR contract).

    The TSV needs at least ``path`` + ``sentence`` columns; audio resolves as
    ``<dataset>/clips/<path>`` then ``<dataset>/<path>`` (mirroring the ASR
    trainer). Rows with an empty path or sentence are skipped.
    """
    split_path = dataset_dir / f"{split}.tsv"
    if not split_path.is_file():
        raise FileNotFoundError(f"{split_path} not found; expected a clean ASR dataset split TSV")
    with split_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"path", "sentence"}.issubset(reader.fieldnames):
            raise ValueError(f"{split_path} must contain path and sentence columns")
        rows = list(reader)
    clips: list[CleanClip] = []
    for line_number, row in enumerate(rows, start=1):
        raw_path = str(row.get("path", "")).strip()
        transcript = str(row.get("sentence", "")).strip()
        if not raw_path or not transcript:
            continue
        clips.append(
            CleanClip(
                clip_id=f"{split}-{line_number}",
                audio_path=_resolve_clip(dataset_dir, raw_path),
                transcript=transcript,
            )
        )
    if not clips:
        raise ValueError(f"no usable rows found in {split_path}")
    return clips


def _resolve_clip(dataset_dir: Path, value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw
    clips_candidate = dataset_dir / "clips" / raw
    return clips_candidate if clips_candidate.exists() else dataset_dir / raw


class CleanMelDataset(Dataset):
    """Yields identical noisy/clean log-Mels (+ optional labels) from a clean ASR dataset.

    A non-degraded ASR dataset has no channel degradation, so the noisy and clean
    *views* are the same clean log-Mel. This feeds the joint stage only, where it
    keeps the full enhancement+fusion+Whisper stack strong on clean speech: the
    enhancer/fusion see clean input against an identity target (``L_enh`` -> 0)
    while ``L_ASR`` fine-tunes the backbone on undegraded audio. The item layout
    matches :class:`DegradedMelDataset` so both concatenate into one loader.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        split: str = "train",
        model_name: str = "openai/whisper-small",
        return_labels: bool = False,
        tokenizer: Any = None,
        sample_rate: int = WHISPER_SAMPLE_RATE,
    ) -> None:
        if return_labels and tokenizer is None:
            raise ValueError("return_labels=True requires a tokenizer")
        self.dataset_dir = Path(dataset_dir)
        self.clips = read_clean_rows(self.dataset_dir, split)
        self.model_name = model_name
        self.return_labels = return_labels
        self.tokenizer = tokenizer
        self.sample_rate = int(sample_rate)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip = self.clips[index]
        audio, source_rate = load_audio(clip.audio_path)
        audio = to_mono(np.asarray(audio, dtype=np.float32))
        if int(source_rate) != self.sample_rate:
            audio = resample_audio(audio, int(source_rate), self.sample_rate)
        mel = waveform_to_log_mel(audio, sample_rate=self.sample_rate, model_name=self.model_name)
        item: dict[str, Any] = {
            "pair_id": clip.clip_id,
            "noisy_mel": mel,
            "clean_mel": mel.clone(),
        }
        if self.return_labels:
            item["labels"] = self.tokenizer(clip.transcript).input_ids
        return item


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
