from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ml.speech_data.scripts.convert_dataset_to_flac import convert_dataset_to_flac


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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def make_dataset(root: Path) -> np.ndarray:
    (root / "clips" / "nested").mkdir(parents=True)
    waveform = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    sf.write(root / "clips" / "nested" / "one.wav", waveform, 16000, subtype="PCM_16")
    sf.write(root / "clips" / "orphan.wav", waveform, 16000, subtype="PCM_16")
    (root / "metadata.json").write_text('{"dataset": "test"}\n', encoding="utf-8")
    rows = [{"path": "nested/one.wav", "sentence": "سلام", "speaker": "speaker-1"}]
    write_tsv(root / "train.tsv", rows)
    write_tsv(root / "dev.tsv", rows)
    return waveform


def test_converts_unique_referenced_clips_and_rewrites_all_split_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "flac"
    make_dataset(source)

    audit = convert_dataset_to_flac(source, output)

    expected_row = {"path": "nested/one.flac", "sentence": "سلام", "speaker": "speaker-1"}
    assert read_tsv(output / "train.tsv") == [expected_row]
    assert read_tsv(output / "dev.tsv") == [expected_row]
    assert sf.info(output / "clips" / "nested" / "one.flac").format == "FLAC"
    decoded, sample_rate = sf.read(output / "clips" / "nested" / "one.flac", dtype="float32")
    original, _ = sf.read(source / "clips" / "nested" / "one.wav", dtype="float32")
    assert sample_rate == 16000
    assert np.array_equal(decoded, original)
    assert not (output / "clips" / "orphan.flac").exists()
    assert (output / "metadata.json").read_text(encoding="utf-8") == '{"dataset": "test"}\n'
    assert audit.split_rows == 2
    assert audit.unique_clips == 1
    stdout = capsys.readouterr().out
    assert "[1/1] converting" in stdout
    assert "nested/one.wav -> clips/nested/one.flac" in stdout
    assert "[1/1] complete:" in stdout
    assert "saved" in stdout
    assert "cumulative" in stdout


def test_supports_paths_that_include_clips_prefix_and_selected_splits(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "flac"
    make_dataset(source)
    rows = [{"path": "clips/nested/one.wav", "sentence": "آزمایش", "speaker": "speaker-2"}]
    write_tsv(source / "test.tsv", rows)

    convert_dataset_to_flac(source, output, splits=["test"])

    assert read_tsv(output / "test.tsv")[0]["path"] == "nested/one.flac"
    assert not (output / "train.tsv").exists()
    assert not (output / "dev.tsv").exists()


def test_rejects_different_sources_that_collapse_to_same_flac_name(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "flac"
    (source / "clips").mkdir(parents=True)
    audio = np.zeros(160, dtype=np.float32)
    sf.write(source / "clips" / "same.wav", audio, 16000)
    sf.write(source / "clips" / "same.ogg", audio, 16000)
    write_tsv(
        source / "train.tsv",
        [
            {"path": "same.wav", "sentence": "یک", "speaker": "a"},
            {"path": "same.ogg", "sentence": "دو", "speaker": "b"},
        ],
    )

    with pytest.raises(ValueError, match="same FLAC path"):
        convert_dataset_to_flac(source, output)

    assert not output.exists()


def test_rejects_existing_output_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "flac"
    make_dataset(source)
    output.mkdir()

    with pytest.raises(FileExistsError, match="output root already exists"):
        convert_dataset_to_flac(source, output)


def test_failed_overwrite_keeps_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "flac"
    (source / "clips").mkdir(parents=True)
    (source / "clips" / "broken.wav").write_bytes(b"not audio")
    write_tsv(
        source / "train.tsv",
        [{"path": "broken.wav", "sentence": "خراب", "speaker": "a"}],
    )
    output.mkdir()
    (output / "keep.txt").write_text("old output", encoding="utf-8")

    with pytest.raises(ValueError, match="could not decode"):
        convert_dataset_to_flac(source, output, overwrite=True)

    assert (output / "keep.txt").read_text(encoding="utf-8") == "old output"


def test_cli_help_describes_arguments_and_defaults() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ml.speech_data.scripts.convert_dataset_to_flac", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--source-root" in result.stdout
    assert "--output-root" in result.stdout
    assert "--subtype {PCM_16,PCM_24}" in result.stdout
    assert "default: PCM_16" in result.stdout
