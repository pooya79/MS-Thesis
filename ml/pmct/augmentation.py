"""Waveform patching and pair-manifest loading for pMCT."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.utils.seed import stable_seed


@dataclass(frozen=True)
class PMCTExample:
    degraded_path: Path
    clean_path: Path
    transcript: str
    dataset_dir: Path
    clean_scale: float = 1.0

    @property
    def audio_path(self) -> Path:
        """Compatibility with the shared Whisper filtering helpers."""
        return self.degraded_path


def _resolve_recorded_path(value: str, dataset_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    if path.exists():
        return path.resolve()
    return (dataset_dir / path).resolve()


def _resolve_tsv_audio_path(dataset_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidates = (dataset_dir / "clips" / path, dataset_dir / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _load_split_index(dataset_dir: Path, split: str) -> dict[Path, str]:
    split_path = dataset_dir / f"{split}.tsv"
    if not split_path.is_file():
        raise FileNotFoundError(f"paired dataset is missing {split_path.name}: {dataset_dir}")
    index: dict[Path, str] = {}
    with split_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"path", "sentence"}.issubset(reader.fieldnames):
            raise ValueError(f"{split_path} must contain path and sentence columns")
        for line_number, row in enumerate(reader, start=2):
            path = _resolve_tsv_audio_path(dataset_dir, str(row.get("path", "")).strip())
            transcript = str(row.get("sentence", "")).strip()
            if path in index:
                raise ValueError(f"{split_path}:{line_number} duplicates audio path {path}")
            index[path] = transcript
    return index


def load_pmct_examples(
    dataset_dirs: list[Path],
    split: str = "train",
    mapping_filename: str = "degraded_to_clean.jsonl",
    expected_sample_rate: int | None = None,
) -> list[PMCTExample]:
    """Load pairs and require exact parity with the baseline split TSV."""
    examples: list[PMCTExample] = []
    for dataset_dir in dataset_dirs:
        split_index = _load_split_index(dataset_dir, split)
        mapping_path = dataset_dir / mapping_filename
        if not mapping_path.is_file():
            raise FileNotFoundError(f"pMCT mapping does not exist: {mapping_path}")
        mapped_paths: set[Path] = set()
        with mapping_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if str(row.get("split", "")) != split:
                        continue
                    degraded_tsv_path = row.get("degraded_tsv_path")
                    degraded_path = (
                        _resolve_tsv_audio_path(dataset_dir, str(degraded_tsv_path))
                        if degraded_tsv_path
                        else _resolve_recorded_path(str(row["degraded_path"]), dataset_dir)
                    )
                    clean_path = _resolve_recorded_path(str(row["clean_path"]), dataset_dir)
                    degradation = row.get("degradation")
                    if not isinstance(degradation, dict):
                        raise TypeError("degradation must be a mapping")
                    transcript = str(row.get("sentence") or degradation.get("transcript") or "").strip()
                    clean_scale = float(degradation["normalization_scale"])
                    model_sample_rate_value = float(degradation["model_sample_rate"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{mapping_path}:{line_number} is not a valid pMCT pair") from exc
                if not transcript:
                    raise ValueError(f"{mapping_path}:{line_number} has no transcript")
                if not math.isfinite(clean_scale) or not 0 < clean_scale <= 1:
                    raise ValueError(
                        f"{mapping_path}:{line_number} normalization_scale must be finite and in (0, 1]"
                    )
                if (
                    not math.isfinite(model_sample_rate_value)
                    or model_sample_rate_value <= 0
                    or not model_sample_rate_value.is_integer()
                ):
                    raise ValueError(
                        f"{mapping_path}:{line_number} model_sample_rate must be a positive integer"
                    )
                model_sample_rate = int(model_sample_rate_value)
                if expected_sample_rate is not None:
                    if model_sample_rate != expected_sample_rate:
                        raise ValueError(
                            f"{mapping_path}:{line_number} model_sample_rate={model_sample_rate} "
                            f"does not match training sample_rate={expected_sample_rate}"
                        )
                if degraded_path in mapped_paths:
                    raise ValueError(f"{mapping_path}:{line_number} duplicates degraded path {degraded_path}")
                if degraded_path not in split_index:
                    raise ValueError(
                        f"{mapping_path}:{line_number} degraded path is not present in {split}.tsv: {degraded_path}"
                    )
                if transcript != split_index[degraded_path]:
                    raise ValueError(
                        f"{mapping_path}:{line_number} transcript does not match {split}.tsv for {degraded_path}"
                    )
                if not degraded_path.is_file():
                    raise FileNotFoundError(f"{mapping_path}:{line_number} missing degraded audio: {degraded_path}")
                if not clean_path.is_file():
                    raise FileNotFoundError(f"{mapping_path}:{line_number} missing clean audio: {clean_path}")
                mapped_paths.add(degraded_path)
                examples.append(
                    PMCTExample(
                        degraded_path=degraded_path,
                        clean_path=clean_path,
                        transcript=transcript,
                        dataset_dir=dataset_dir.resolve(),
                        clean_scale=clean_scale,
                    )
                )
        missing = set(split_index) - mapped_paths
        if missing:
            preview = ", ".join(str(path) for path in sorted(missing)[:3])
            raise ValueError(
                f"{mapping_path} is missing {len(missing)} {split}.tsv row(s); first: {preview}"
            )
    if not examples:
        raise ValueError(f"no pMCT examples found for split {split!r}")
    return examples


def patch_seed(global_seed: int, epoch: int, degraded_path: Path) -> int:
    """Return a process-stable patch seed that changes between epochs."""
    return stable_seed(global_seed, "pmct", str(degraded_path.resolve()), epoch)


def mix_aligned_patches(
    clean: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int,
    patch_seconds: float,
    clean_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose each aligned patch from clean audio with probability ``pi``."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if patch_seconds <= 0:
        raise ValueError("patch_seconds must be > 0")
    if not 0 <= clean_probability <= 1:
        raise ValueError("clean_probability must be in [0, 1]")

    clean = np.asarray(clean, dtype=np.float32)
    degraded = np.asarray(degraded, dtype=np.float32)
    if clean.ndim != 1 or degraded.ndim != 1:
        raise ValueError("pMCT expects mono waveforms")
    if len(clean) != len(degraded):
        raise ValueError(
            f"pMCT requires aligned equal-length audio, got clean={len(clean)} degraded={len(degraded)}"
        )

    patch_samples = max(1, int(round(patch_seconds * sample_rate)))
    output = degraded.copy()
    for start in range(0, len(output), patch_samples):
        end = min(start + patch_samples, len(output))
        if rng.random() < clean_probability:
            output[start:end] = clean[start:end]
    return output
