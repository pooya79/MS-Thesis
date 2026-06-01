from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ml.speech_data.generate_degraded_pairs import load_asset_index
from ml.speech_data.scripts.prepare_degradation_assets import prepare_degradation_assets


def write_tone(path: Path, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 0.05, sample_rate // 20, endpoint=False, dtype=np.float32)
    audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(path, audio, sample_rate)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_prepare_degradation_assets_writes_loadable_indexes(tmp_path: Path) -> None:
    noise_root = tmp_path / "assets" / "noise" / "DEMAND"
    manifest_dir = tmp_path / "manifests"

    write_tone(noise_root / "DKITCHEN_16k" / "ch01.wav")

    audit = prepare_degradation_assets(
        noise_root,
        manifest_dir,
        extract=False,
        show_progress=False,
    )

    assert audit.noise_candidates == 1
    assert audit.noise_indexed == 1

    noise_rows = read_jsonl(manifest_dir / "demand_noise_index.jsonl")

    assert noise_rows[0]["scene"] == "dkitchen"
    assert noise_rows[0]["id"] == "demand-DKITCHEN_16k_ch01"
    assert not str(noise_rows[0]["path"]).startswith("/")

    noise_assets = load_asset_index(manifest_dir / "demand_noise_index.jsonl")
    assert Path(str(noise_assets[0]["path"])).exists()


def test_prepare_degradation_assets_extracts_local_archives(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    noise_source = staging / "DKITCHEN_16k" / "ch01.wav"
    write_tone(noise_source)

    noise_root = tmp_path / "assets" / "noise" / "DEMAND"
    noise_root.mkdir(parents=True)

    with zipfile.ZipFile(noise_root / "DKITCHEN_16k.zip", "w") as archive:
        archive.write(noise_source, arcname="DKITCHEN_16k/ch01.wav")

    audit = prepare_degradation_assets(
        noise_root,
        tmp_path / "manifests",
        show_progress=False,
    )

    assert audit.archives_extracted == 1
    assert audit.noise_indexed == 1
