from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ml.speech_data.scripts.convert_dataset_to_flac import convert_dataset_to_flac
from ml.speech_data.scripts.verify_flac_conversion import (
    FlacVerificationError,
    verify_flac_conversion,
)


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "sentence", "speaker"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def make_dataset(root: Path) -> None:
    (root / "clips" / "nested").mkdir(parents=True)
    waveform = np.linspace(-0.75, 0.75, 3200, dtype=np.float32)
    sf.write(root / "clips" / "nested" / "one.wav", waveform, 16000, subtype="PCM_16")
    sf.write(root / "clips" / "orphan.wav", waveform, 16000, subtype="PCM_16")
    (root / "metadata.json").write_text('{"name": "example"}\n', encoding="utf-8")
    write_tsv(
        root / "train.tsv",
        [{"path": "nested/one.wav", "sentence": "سلام", "speaker": "speaker-1"}],
    )


def test_verifies_all_expected_files_and_logs_each_check(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source"
    converted = tmp_path / "converted"
    make_dataset(source)
    convert_dataset_to_flac(source, converted, log_progress=False)

    audit = verify_flac_conversion(source, converted)

    assert audit.checked_files == 3
    assert audit.split_files == 1
    assert audit.audio_files == 1
    assert audit.metadata_files == 1
    stdout = capsys.readouterr().out
    assert "[1/3] checking split" in stdout
    assert "[2/3] checking audio" in stdout
    assert "[3/3] checking metadata" in stdout
    assert "nested/one.wav" in stdout
    assert "nested/one.flac" in stdout


def test_reports_changed_audio_tsv_metadata_and_unexpected_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    converted = tmp_path / "converted"
    make_dataset(source)
    convert_dataset_to_flac(source, converted, log_progress=False)
    sf.write(converted / "clips" / "nested" / "one.flac", np.zeros(3200), 16000)
    write_tsv(
        converted / "train.tsv",
        [{"path": "nested/one.flac", "sentence": "تغییر", "speaker": "speaker-1"}],
    )
    (converted / "metadata.json").write_text("changed\n", encoding="utf-8")
    (converted / "extra.txt").write_text("extra\n", encoding="utf-8")

    with pytest.raises(FlacVerificationError) as error:
        verify_flac_conversion(source, converted, log_progress=False)

    message = str(error.value)
    assert "rows differ" in message
    assert "decoded samples differ" in message
    assert "metadata file contents differ" in message
    assert "unexpected file" in message
    assert error.value.audit.checked_files == 3


def test_cli_help_describes_roots_and_split_selection() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ml.speech_data.scripts.verify_flac_conversion", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--source-root" in result.stdout
    assert "--converted-root" in result.stdout
    assert "--splits" in result.stdout
    assert "decoded audio" in result.stdout
