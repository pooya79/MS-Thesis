from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ml.speech_data.generate_degraded_dataset import generate_degraded_dataset
from ml.speech_data.generate_degraded_pairs import (
    apply_decoded_waveform_dropout,
    codec_roundtrip,
    ffmpeg_encoder_candidates,
    generate_from_config,
    resolve_ffmpeg_encoder,
)
from ml.speech_data.scripts.generate_random_degraded_clip import generate_random_degraded_clip
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
        "noise_index": None,
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
        assert "rir_id" not in row
        assert "reverb_mode" not in row

    inspection = inspect_manifest(train_manifest)
    assert inspection["pairs"] == 2
    assert inspection["missing_files"] == 0
    assert inspection["length_mismatches"] == 0
    assert inspection["distributions"]["profile"] == {"legacy": 2}


def test_generate_degraded_pairs_records_selected_profile(tmp_path: Path) -> None:
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    sample_rate = 16000
    t = np.linspace(0, 0.25, sample_rate // 4, endpoint=False, dtype=np.float32)
    audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(clean_dir / "train.wav", audio, sample_rate)
    sf.write(clean_dir / "valid.wav", audio, sample_rate)

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    for split in ("train", "valid"):
        row = {"id": f"{split}-clip", "split": split, "clean_path": str(clean_dir / f"{split}.wav")}
        (manifest_dir / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    config = {
        "seed": 11,
        "variants_per_clip": 1,
        "output_dir": str(tmp_path / "out"),
        "manifests": {"train": str(manifest_dir / "train.jsonl"), "valid": str(manifest_dir / "valid.jsonl")},
        "noise_index": None,
        "noise": {"probability": 0.0, "second_scene_probability": 0.0, "snr_buckets": [[0, 1]]},
        "level": {"gain_db": [0, 0], "clipping": {"enabled": False}, "agc": {"enabled": False}},
        "codec_distribution": [{"codec": "pass_through", "weight": 1.0}],
        "channel": {
            "narrowband": {"bandpass_hz": [300, 3400]},
            "wideband": {"bandpass_hz": [50, 7000], "filter_target": False},
            "pass_through_path_distribution": [{"path": "wideband", "weight": 1.0}],
        },
        "network_impairment": {"enabled": False, "probability": 0.0, "loss_rate_buckets": [[0.0, 0.0]], "burst_length": [1, 1], "frame_ms": 20},
        "profiles": [
            {
                "name": "unit_profile",
                "weight": 1.0,
                "noise": {"probability": 0.0},
                "network_impairment": {"probability": 0.0},
                "codec_distribution": [{"codec": "pass_through", "weight": 1.0}],
            }
        ],
    }

    generate_from_config(config)

    train_manifest = tmp_path / "out" / "manifests" / "se_train_pairs.jsonl"
    rows = [json.loads(line) for line in train_manifest.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["profile"] == "unit_profile"
    assert rows[0]["codec_bitrate"] is None
    assert rows[0]["codec_frame_duration_ms"] is None

    inspection = inspect_manifest(train_manifest)
    assert inspection["distributions"]["profile"] == {"unit_profile": 1}


def test_generate_degraded_dataset_writes_tsvs_and_mapping(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "data" / "cv-corpus-25.0"
    clips_dir = dataset_dir / "clips"
    clips_dir.mkdir(parents=True)
    sample_rate = 16000
    t = np.linspace(0, 0.25, sample_rate // 4, endpoint=False, dtype=np.float32)
    audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(clips_dir / "sample.wav", audio, sample_rate)
    for split in ("train", "eval"):
        with (dataset_dir / f"{split}.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "sentence"], delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerow({"path": "sample.wav", "sentence": "سلام"})

    config = {
        "dataset": {
            "source_dir": str(dataset_dir),
            "output_dir": str(tmp_path / "data" / "cv-corpus-25.0-degraded"),
            "splits": ["train.tsv", "eval.tsv"],
            "variations_per_sample": 2,
            "mapping_filename": "degraded_to_clean.jsonl",
            "metadata_filename": "degradation_metadata.jsonl",
            "report_filename": "generation_report.json",
        },
        "degradation": {
            "seed": 3,
            "model_sample_rate": 16000,
            "working_sample_rate": 16000,
            "noise_index": None,
            "noise": {"probability": 0.0, "second_scene_probability": 0.0, "snr_buckets": [[0, 1]]},
            "level": {"gain_db": [0, 0], "clipping": {"enabled": False}, "agc": {"enabled": False}},
            "codec_distribution": [{"codec": "pass_through", "weight": 1.0}],
            "channel": {
                "narrowband": {"bandpass_hz": [300, 3400]},
                "wideband": {"bandpass_hz": [50, 7000], "filter_target": False},
                "pass_through_path_distribution": [{"path": "wideband", "weight": 1.0}],
            },
            "network_impairment": {
                "enabled": False,
                "probability": 0.0,
                "loss_rate_buckets": [[0.0, 0.0]],
                "burst_length": [1, 1],
                "frame_ms": 20,
            },
        },
    }

    report = generate_degraded_dataset(config)

    output_dir = tmp_path / "data" / "cv-corpus-25.0-degraded"
    assert report["splits"]["train"]["degraded_rows"] == 2
    assert report["splits"]["eval"]["degraded_rows"] == 2
    assert not (output_dir / "pairs").exists()

    train_rows = list(csv.DictReader((output_dir / "train.tsv").open(encoding="utf-8"), delimiter="\t"))
    assert len(train_rows) == 2
    assert train_rows[0]["sentence"] == "سلام"
    assert train_rows[0]["path"].startswith("train/")

    mapping_rows = [json.loads(line) for line in (output_dir / "degraded_to_clean.jsonl").read_text(encoding="utf-8").splitlines()]
    metadata_rows = [json.loads(line) for line in (output_dir / "degradation_metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(mapping_rows) == 4
    assert len(metadata_rows) == 4
    assert {row["split"] for row in mapping_rows} == {"train", "eval"}
    assert {row["pair_id"] for row in metadata_rows} == {row["degraded_id"] for row in mapping_rows}
    assert {row["variant_index"] for row in mapping_rows if row["split"] == "train"} == {0, 1}
    for row in mapping_rows:
        assert Path(row["degraded_path"]).is_file()
        assert row["clean_path"] == str((clips_dir / "sample.wav").resolve())
        degraded, degraded_sr = load_audio(row["degraded_path"])
        assert degraded_sr == 16000
        assert len(degraded) == len(audio)
        assert row["degradation"]["codec"] == "pass_through"


def test_generate_random_degraded_clip_writes_demo_variants(tmp_path: Path) -> None:
    input_root = tmp_path / "data"
    clips_dir = input_root / "clips"
    clips_dir.mkdir(parents=True)
    sample_rate = 16000
    t = np.linspace(0, 0.25, sample_rate // 4, endpoint=False, dtype=np.float32)
    audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(clips_dir / "source.wav", audio, sample_rate)

    config_path = tmp_path / "degradation.yaml"
    config_path.write_text(
        """
seed: 1
model_sample_rate: 16000
working_sample_rate: 16000
noise_index: null
noise:
  probability: 0.0
  second_scene_probability: 0.0
  snr_buckets:
    - [0, 1]
level:
  gain_db: [0, 0]
  clipping:
    enabled: false
  agc:
    enabled: false
channel:
  narrowband:
    bandpass_hz: [300, 3400]
  wideband:
    bandpass_hz: [50, 7000]
    filter_target: false
  pass_through_path_distribution:
    - path: wideband
      weight: 1.0
codec_distribution:
  - codec: pass_through
    weight: 1.0
network_impairment:
  enabled: false
  probability: 0.0
  loss_rate_buckets:
    - [0.0, 0.0]
  burst_length: [1, 1]
  frame_ms: 20
""",
        encoding="utf-8",
    )

    report = generate_random_degraded_clip(
        config_path=config_path,
        input_root=input_root,
        output_dir=tmp_path / "demo_out",
        variants=3,
        seed=42,
    )

    assert report["selected_audio"] == str(clips_dir / "source.wav")
    manifest_path = Path(str(report["manifest"]))
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert len(report["degraded_paths"]) == 3
    assert {row["split"] for row in rows} == {"demo"}
    assert {row["codec"] for row in rows} == {"pass_through"}

    for row in rows:
        clean, clean_sr = load_audio(row["clean_path"])
        degraded, degraded_sr = load_audio(row["degraded_path"])
        assert clean_sr == degraded_sr == 16000
        assert len(clean) == len(degraded)


def test_decoded_waveform_dropout_reports_observed_loss() -> None:
    rng = np.random.default_rng(123)
    audio = np.ones(16000, dtype=np.float32)

    degraded, dropped_frames, total_frames = apply_decoded_waveform_dropout(
        audio,
        sample_rate=16000,
        rng=rng,
        loss_rate=0.20,
        burst_length=4,
        frame_ms=20,
    )

    assert len(degraded) == len(audio)
    assert total_frames == 50
    assert 0 < dropped_frames < total_frames
    assert np.count_nonzero(degraded == 0) > 0


def test_ffmpeg_available_for_configured_codecs() -> None:
    ffmpeg = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], text=True, capture_output=True, check=True)
    output = ffmpeg.stdout + ffmpeg.stderr
    expected_encoders = {
        "amr_nb_12k2": ["libopencore_amrnb"],
        "amr_wb_12k65": ["libvo_amrwbenc"],
        "gsm": ffmpeg_encoder_candidates("gsm"),
        "opus": ["libopus"],
        "g711_alaw": ["pcm_alaw"],
    }
    available_names = set(re.findall(r"[A-Za-z0-9_]+", output))
    for codec, encoders in expected_encoders.items():
        if not any(encoder in available_names for encoder in encoders):
            pytest.skip(f"ffmpeg codec {codec} ({' or '.join(encoders)}) is not available in this environment")


def test_gsm_accepts_libgsm_or_native_encoder_name() -> None:
    assert ffmpeg_encoder_candidates("gsm") == ["libgsm", "gsm"]
    assert resolve_ffmpeg_encoder("gsm", "encoders: gsm") == "gsm"
    assert resolve_ffmpeg_encoder("gsm", "encoders: libgsm gsm") == "libgsm"
    assert resolve_ffmpeg_encoder("gsm", "encoders: libgsm_ms") is None
    assert resolve_ffmpeg_encoder("gsm", "encoders: pcm_alaw") is None


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
