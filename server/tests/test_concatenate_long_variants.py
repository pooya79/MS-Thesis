from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from ml.speech_data.concatenate_long_variants import concatenate_long_variants


SAMPLE_RATE = 16000


def _write_clip(path: Path, seconds: float, freq: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(seconds * SAMPLE_RATE), endpoint=False, dtype=np.float32)
    sf.write(str(path), 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32), SAMPLE_RATE, subtype="PCM_16")


def _build_dataset(root: Path, splits: dict[str, int]) -> dict[str, set[str]]:
    """Create a tiny dataset; return the set of clip paths per split stem."""
    paths_by_split: dict[str, set[str]] = {}
    for split, count in splits.items():
        stem = Path(split).stem
        rows = []
        names = set()
        for i in range(count):
            name = f"{stem}_{i:03d}.wav"
            _write_clip(root / "clips" / name, seconds=1.5, freq=200 + i)
            rows.append({"path": name, "sentence": f"{stem} sentence {i}"})
            names.add(name)
        with (root / split).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "sentence"], delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        paths_by_split[stem] = names
    return paths_by_split


def _read_manifest(output_root: Path) -> list[dict]:
    lines = (output_root / "long_variants_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_generates_long_variants_for_each_split(tmp_path: Path) -> None:
    source = tmp_path / "src"
    paths_by_split = _build_dataset(source, {"train.tsv": 8, "dev.tsv": 6, "test.tsv": 5})
    output = tmp_path / "out"

    report = concatenate_long_variants(
        source,
        output,
        seed=7,
        variants_per_split=5,
        sample_rate=SAMPLE_RATE,
        min_clips=2,
        max_clips=4,
        target_min_sec=3.0,
        max_duration_sec=10.0,
        gap_sec=0.1,
    )

    for split in ("train.tsv", "dev.tsv", "test.tsv"):
        assert (output / split).exists(), f"missing output split {split}"
        assert report["splits"][split]["variants_written"] == 5
        # Every variant clears the target minimum duration.
        assert report["splits"][split]["min_duration_sec"] >= 3.0


def test_concatenation_stays_within_each_split(tmp_path: Path) -> None:
    source = tmp_path / "src"
    paths_by_split = _build_dataset(source, {"train.tsv": 8, "dev.tsv": 6, "test.tsv": 5})
    output = tmp_path / "out"

    concatenate_long_variants(
        source,
        output,
        seed=7,
        variants_per_split=5,
        sample_rate=SAMPLE_RATE,
        min_clips=2,
        max_clips=4,
        target_min_sec=3.0,
        max_duration_sec=10.0,
        gap_sec=0.1,
    )

    for entry in _read_manifest(output):
        split = entry["split"]
        sources = set(entry["source_paths"])
        assert len(entry["source_paths"]) >= 2
        # No source clip may come from a different split.
        assert sources.issubset(paths_by_split[split]), f"{split} variant pulled cross-split clips"


def test_deterministic_for_same_seed(tmp_path: Path) -> None:
    source = tmp_path / "src"
    _build_dataset(source, {"train.tsv": 8})

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    common = dict(
        seed=42,
        variants_per_split=4,
        sample_rate=SAMPLE_RATE,
        min_clips=2,
        max_clips=4,
        target_min_sec=3.0,
        max_duration_sec=10.0,
        gap_sec=0.1,
    )
    concatenate_long_variants(source, out_a, **common)
    concatenate_long_variants(source, out_b, **common)

    man_a = [{k: v for k, v in row.items() if k != "seed"} for row in _read_manifest(out_a)]
    man_b = [{k: v for k, v in row.items() if k != "seed"} for row in _read_manifest(out_b)]
    assert man_a == man_b

    # Audio bytes match too.
    for clip in (out_a / "clips").glob("*.wav"):
        a, _ = sf.read(str(clip))
        b, _ = sf.read(str(out_b / "clips" / clip.name))
        assert np.array_equal(a, b)
