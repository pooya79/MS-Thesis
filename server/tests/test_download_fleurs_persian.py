from __future__ import annotations

import json
from pathlib import Path

import pytest
import soundfile as sf

from ml.speech_data.scripts.download_fleurs_persian import download_fleurs_persian


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fake_row(row_id: str, transcript: str) -> dict[str, object]:
    return {
        "id": row_id,
        "audio": {
            "array": [0.0, 0.1, -0.1],
            "sampling_rate": 8000,
            "path": f"/hf/cache/{row_id}.wav",
        },
        "transcription": transcript,
        "raw_transcription": f"raw {transcript}",
        "language": "Persian",
        "lang_id": 22,
        "lang_group_id": 2,
        "gender": 1,
        "num_samples": 3,
    }


def fake_dataset() -> dict[str, list[dict[str, object]]]:
    return {
        "train": [fake_row("1", "سلام")],
        "validation": [fake_row("1", "اعتبارسنجی")],
        "test": [fake_row("1", "آزمون")],
    }


def test_download_fleurs_persian_exports_jsonl_and_audio(tmp_path: Path) -> None:
    output_root = tmp_path / "source"

    audit = download_fleurs_persian(output_root, dataset=fake_dataset())

    assert audit.train_rows == 1
    assert audit.validation_rows == 1
    assert audit.test_rows == 1
    assert audit.audio_written == 3

    train_records = read_jsonl(output_root / "train.jsonl")
    assert train_records == [
        {
            "audio_path": "audio/train/train-1.wav",
            "config_name": "fa_ir",
            "dataset_name": "google/fleurs",
            "gender": 1,
            "hf_id": "1",
            "id": "train-1",
            "lang_group_id": 2,
            "lang_id": 22,
            "language": "Persian",
            "num_samples": 3,
            "raw_transcription": "raw سلام",
            "sample_rate": 8000,
            "source_audio_path": "/hf/cache/1.wav",
            "split": "train",
            "transcription": "سلام",
        }
    ]

    audio, sample_rate = sf.read(output_root / "audio" / "train" / "train-1.wav")
    assert sample_rate == 8000
    assert len(audio) == 3


def test_download_fleurs_persian_exports_decode_false_audio_bytes(tmp_path: Path) -> None:
    output_root = tmp_path / "source"
    dataset = {
        "train": [{"id": "1", "audio": {"bytes": b"wav-bytes", "path": "train.wav"}, "transcription": "سلام"}],
        "validation": [{"id": "1", "audio": {"bytes": b"dev-bytes", "path": "dev.wav"}, "transcription": "درود"}],
        "test": [{"id": "1", "audio": {"bytes": b"test-bytes", "path": "test.wav"}, "transcription": "آزمون"}],
    }

    audit = download_fleurs_persian(output_root, dataset=dataset)

    assert audit.audio_written == 3
    assert (output_root / "audio" / "train" / "train-1.wav").read_bytes() == b"wav-bytes"
    train_records = read_jsonl(output_root / "train.jsonl")
    assert train_records[0]["sample_rate"] == 16000
    assert train_records[0]["source_audio_path"] == "train.wav"


def test_download_fleurs_persian_requires_force_for_existing_export(tmp_path: Path) -> None:
    output_root = tmp_path / "source"
    download_fleurs_persian(output_root, dataset=fake_dataset())

    with pytest.raises(FileExistsError, match="pass --force"):
        download_fleurs_persian(output_root, dataset=fake_dataset())

    audit = download_fleurs_persian(output_root, dataset=fake_dataset(), force=True)

    assert audit.audio_written == 3


def test_download_fleurs_persian_validates_required_splits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required splits"):
        download_fleurs_persian(tmp_path, dataset={"train": [], "validation": []})
