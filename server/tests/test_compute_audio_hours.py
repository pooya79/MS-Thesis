from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ml.speech_data.scripts.compute_audio_hours import (
    compute_audio_hours,
    discover_audio_files,
    main,
    read_audio_duration,
)


def write_silence(path: Path, seconds: float, sample_rate: int = 8_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(round(seconds * sample_rate), dtype=np.float32), sample_rate)


def test_discovers_supported_audio_recursively_and_case_insensitively(tmp_path: Path) -> None:
    write_silence(tmp_path / "one.wav", 0.1)
    write_silence(tmp_path / "nested" / "two.FLAC", 0.1)
    (tmp_path / "ignored.txt").write_text("not audio", encoding="utf-8")

    paths = discover_audio_files(tmp_path)

    assert paths == [tmp_path / "nested" / "two.FLAC", tmp_path / "one.wav"]


def test_reads_mp3_duration_without_decoding_samples(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "output.mp3"
    write_silence(source, 1.0)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-codec:a",
            "libmp3lame",
            str(output),
        ],
        check=True,
    )

    result = read_audio_duration(output)

    assert result.error is None
    # MP3 encoder delay/padding makes container duration slightly longer.
    assert result.seconds == pytest.approx(1.0, abs=0.2)


def test_computes_total_with_multiple_workers_and_logs_progress(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    write_silence(tmp_path / "one.wav", 1.0)
    write_silence(tmp_path / "nested" / "two.flac", 2.0)

    with caplog.at_level(logging.INFO):
        summary = compute_audio_hours(tmp_path, workers=2)

    assert summary.discovered_files == 2
    assert summary.processed_files == 2
    assert summary.failed_files == 0
    assert summary.total_seconds == pytest.approx(3.0)
    assert summary.total_hours == pytest.approx(3.0 / 3_600)
    assert "Discovered 2 audio files" in caplog.text
    assert "[1/2]" in caplog.text
    assert "[2/2]" in caplog.text
    assert "running total" in caplog.text


def test_reports_bad_audio_without_discarding_valid_total(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    write_silence(tmp_path / "valid.wav", 1.0)
    (tmp_path / "broken.mp3").write_bytes(b"not an mp3")

    with caplog.at_level(logging.INFO):
        summary = compute_audio_hours(tmp_path, workers=2)

    assert summary.processed_files == 1
    assert summary.failed_files == 1
    assert summary.total_seconds == pytest.approx(1.0)
    assert "Failed" in caplog.text
    assert "broken.mp3" in caplog.text


def test_cli_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_silence(tmp_path / "one.wav", 1.0)

    exit_code = main([str(tmp_path), "--workers", "1"])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Total hours: 0.000278" in stdout
    assert "Files processed: 1/1" in stdout


def test_rejects_invalid_worker_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workers must be at least 1"):
        compute_audio_hours(tmp_path, workers=0)
