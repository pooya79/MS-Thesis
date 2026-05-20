from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ml.speech_data.generate_degraded_pairs import codec_roundtrip, generate_from_config
from ml.speech_data.inspect_manifest import inspect_manifest
from ml.utils.audio import bandpass_filter, load_audio, peak_safety_normalize, resample_audio
from ml.utils.seed import stable_seed


def test_stable_seed_is_process_stable() -> None:
    expected = stable_seed(1337, "train", "clip-1", 0)
    code = "from ml.utils.seed import stable_seed; print(stable_seed(1337, 'train', 'clip-1', 0))"
    output = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert int(output) == expected


def test_audio_helpers_preserve_shape_and_peak() -> None:
    sample_rate = 16000
    t = np.linspace(0, 1, sample_rate, endpoint=False, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    resampled = resample_audio(audio, sample_rate, 8000)
    filtered = bandpass_filter(resampled, 8000, 300, 3400)
    normalized = peak_safety_normalize(filtered * 4, peak=0.5)

    assert len(resampled) == 8000
    assert len(filtered) == len(resampled)
    assert np.isfinite(filtered).all()
    assert np.max(np.abs(normalized)) <= 0.5001


def test_generate_degraded_pairs_smoke(tmp_path: Path) -> None:
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    sample_rate = 16000
    t = np.linspace(0, 0.5, sample_rate // 2, endpoint=False, dtype=np.float32)
    for split in ("train", "valid"):
        audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(clean_dir / f"{split}.wav", audio, sample_rate)

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    for split in ("train", "valid"):
        row = {"id": f"{split}-clip", "split": split, "clean_path": str(clean_dir / f"{split}.wav"), "transcript": "سلام"}
        (manifest_dir / f"{split}.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    config = {
        "seed": 7,
        "variants_per_clip": 2,
        "output_dir": str(tmp_path / "out"),
        "manifests": {"train": str(manifest_dir / "train.jsonl"), "valid": str(manifest_dir / "valid.jsonl")},
        "rir_index": None,
        "noise_index": None,
        "reverb": {
            "severe": {"probability": 0.0, "wet_mix": [0.6, 0.8], "dr_db": [6, 10]},
            "mild": {"probability": 0.0, "wet_mix": [0.3, 0.5], "dr_db": [12, 18]},
        },
        "noise": {"probability": 0.0, "second_scene_probability": 0.0, "snr_buckets": [[0, 1]]},
        "level": {"gain_db": [0, 0], "clipping": {"enabled": False}, "agc": {"enabled": False}},
        "codec_distribution": [{"codec": "pass_through", "weight": 1.0}],
        "channel": {
            "narrowband": {"bandpass_hz": [300, 3400]},
            "wideband": {"bandpass_hz": [50, 7000], "filter_target": False},
            "pass_through_path_distribution": [{"path": "wideband", "weight": 1.0}],
        },
        "network_impairment": {"enabled": False, "probability": 0.0, "loss_rate_buckets": [[0.0, 0.0]], "burst_length": [1, 1], "frame_ms": 20},
    }

    report = generate_from_config(config)
    assert report["splits"]["train"]["pairs"] == 2
    assert report["splits"]["valid"]["pairs"] == 2

    train_manifest = tmp_path / "out" / "manifests" / "se_train_pairs.jsonl"
    rows = [json.loads(line) for line in train_manifest.read_text(encoding="utf-8").splitlines()]
    pair_ids = {row["pair_id"] for row in rows}
    assert len(pair_ids) == 2

    for row in rows:
        clean, clean_sr = load_audio(row["clean_path"])
        degraded, degraded_sr = load_audio(row["degraded_path"])
        assert clean_sr == degraded_sr == 16000
        assert len(clean) == len(degraded)
        assert row["channel_path"] == "wideband"
        assert row["codec"] == "pass_through"

    inspection = inspect_manifest(train_manifest)
    assert inspection["pairs"] == 2
    assert inspection["missing_files"] == 0
    assert inspection["length_mismatches"] == 0


def test_ffmpeg_available_for_configured_codecs() -> None:
    ffmpeg = subprocess.run(["ffmpeg", "-hide_banner", "-codecs"], text=True, capture_output=True, check=True)
    output = ffmpeg.stdout + ffmpeg.stderr
    for codec in ("libopencore_amrnb", "libvo_amrwbenc", "libgsm", "libopus", "pcm_alaw"):
        if codec not in output:
            pytest.skip(f"ffmpeg codec {codec} is not available in this environment")


@pytest.mark.parametrize(
    ("codec", "sample_rate"),
    [
        ("g711_alaw", 8000),
        ("g711_mulaw", 8000),
        ("gsm", 8000),
        ("amr_nb_12k2", 8000),
        ("amr_wb_12k65", 16000),
        ("opus_nb", 8000),
        ("opus_wb", 16000),
    ],
)
def test_configured_codec_roundtrips(codec: str, sample_rate: int) -> None:
    t = np.linspace(0, 0.5, int(sample_rate * 0.5), endpoint=False, dtype=np.float32)
    audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    decoded = codec_roundtrip(audio, sample_rate, codec)

    assert len(decoded) > 0
    assert np.isfinite(decoded).all()
    assert np.max(np.abs(decoded)) <= 1.0
