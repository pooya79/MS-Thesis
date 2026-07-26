from __future__ import annotations

import csv
import io
import subprocess
import tarfile
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable

import pytest

from ml.speech_data.scripts.download_filimo_persian_asr import download_filimo_persian_asr
from ml.speech_data.scripts.download_youtube_persian_asr import download_youtube_persian_asr
from ml.speech_data.scripts.persets_asr import (
    ConversionResult,
    FILIMO_DATASET,
    YOUTUBE_DATASET,
    PreparationAudit,
    _finish_futures,
    read_metadata,
)
from ml.speech_data.scripts.prepare_filimo_persian_asr import prepare_filimo_persian_asr
from ml.speech_data.scripts.prepare_youtube_persian_asr import prepare_youtube_persian_asr


def write_tar(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


@pytest.mark.parametrize(
    ("download", "repo_id"),
    [
        (download_youtube_persian_asr, "PerSets/youtube-persian-asr"),
        (download_filimo_persian_asr, "PerSets/filimo-persian-asr"),
    ],
)
def test_download_scripts_fetch_only_metadata_and_tar_shards(
    tmp_path: Path,
    download: Callable[..., object],
    repo_id: str,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_downloader(**kwargs: Any) -> str:
        calls.append(kwargs)
        output_root = Path(kwargs["local_dir"])
        output_root.mkdir(parents=True)
        (output_root / "unvalidated.csv").write_text("file_name,sentence\n", encoding="utf-8")
        write_tar(output_root / "data" / "unvalidated_001.tar", {})
        return str(output_root)

    output_root = tmp_path / "source"
    audit = download(
        output_root,
        revision="commit-sha",
        workers=3,
        force=True,
        downloader=fake_downloader,
    )

    assert calls == [
        {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "revision": "commit-sha",
            "local_dir": output_root,
            "allow_patterns": ["unvalidated.csv", "data/*.tar"],
            "force_download": True,
            "max_workers": 3,
        }
    ]
    assert audit.repo_id == repo_id
    assert audit.tar_shards == 1
    assert audit.metadata_files == 1
    assert audit.downloaded_bytes == sum(
        path.stat().st_size
        for path in [output_root / "unvalidated.csv", output_root / "data" / "unvalidated_001.tar"]
    )


def test_download_rejects_invalid_workers_before_network_access(tmp_path: Path) -> None:
    called = False

    def fake_downloader(**_: Any) -> str:
        nonlocal called
        called = True
        return ""

    with pytest.raises(ValueError, match="workers must be >= 1"):
        download_youtube_persian_asr(tmp_path, workers=0, downloader=fake_downloader)

    assert called is False


def test_prepare_youtube_normalizes_drops_rejected_and_excludes_missing_audio(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "normalized"
    source_root.mkdir()
    (source_root / "unvalidated.csv").write_text(
        "file_name,sentence\n"
        "0053700001.mp3,خب ، تو چیكار می كنی؟\n"
        "0053700002.mp3,hello سلام\n"
        "0053700003.mp3,این فایل موجود نیست\n",
        encoding="utf-8",
    )
    write_tar(
        source_root / "data" / "unvalidated_1.tar",
        {
            "nested/0053700001.mp3": b"first-mp3",
            "nested/0053700002.mp3": b"rejected-mp3",
        },
    )
    conversions: list[tuple[bytes, Path]] = []

    def fake_converter(audio_bytes: bytes, output_path: Path) -> None:
        conversions.append((audio_bytes, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"wav:" + audio_bytes)

    audit = prepare_youtube_persian_asr(
        source_root,
        output_root,
        workers=4,
        converter=fake_converter,
        show_progress=False,
    )

    assert read_tsv(output_root / "train.tsv") == [
        {"path": "0053700001.wav", "sentence": "خب تو چیکار می کنی"}
    ]
    assert conversions == [(b"first-mp3", output_root / "clips" / "0053700001.wav")]
    assert (output_root / "clips" / "0053700001.wav").read_bytes() == b"wav:first-mp3"
    assert audit == PreparationAudit(
        source_rows=3,
        normalized_rows=2,
        changed_rows=1,
        discarded_rows=1,
        missing_audio_rows=1,
        wav_converted=1,
        wav_skipped_existing=0,
        final_train_rows=1,
    )


def test_prepare_filimo_accepts_tab_metadata_with_unnamed_index_and_resumes(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "normalized"
    source_root.mkdir()
    (source_root / "unvalidated.csv").write_text(
        "\tfile_name\tsentence\n"
        "0\t0036100001.mp3\tسلام! «دوست»؛\n",
        encoding="utf-8",
    )
    write_tar(source_root / "data" / "unvalidated_001.tar", {"clips/0036100001.mp3": b"mp3"})
    existing = output_root / "clips" / "0036100001.wav"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"existing")
    conversions: list[bytes] = []

    def fake_converter(audio_bytes: bytes, output_path: Path) -> None:
        conversions.append(audio_bytes)
        output_path.write_bytes(b"converted")

    first_audit = prepare_filimo_persian_asr(
        source_root,
        output_root,
        converter=fake_converter,
        show_progress=False,
    )
    forced_audit = prepare_filimo_persian_asr(
        source_root,
        output_root,
        force=True,
        converter=fake_converter,
        show_progress=False,
    )

    assert read_tsv(output_root / "train.tsv") == [{"path": "0036100001.wav", "sentence": "سلام دوست"}]
    assert first_audit.wav_skipped_existing == 1
    assert first_audit.wav_converted == 0
    assert forced_audit.wav_skipped_existing == 0
    assert forced_audit.wav_converted == 1
    assert conversions == [b"mp3"]
    assert existing.read_bytes() == b"converted"


@pytest.mark.parametrize(
    ("prepare", "delimiter"),
    [
        (prepare_youtube_persian_asr, ","),
        (prepare_filimo_persian_asr, "\t"),
    ],
)
def test_prepare_skips_failed_audio_and_records_ffmpeg_error(
    tmp_path: Path,
    prepare: Callable[..., PreparationAudit],
    delimiter: str,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "normalized"
    source_root.mkdir()
    (source_root / "unvalidated.csv").write_text(
        delimiter.join(["file_name", "sentence"])
        + "\n"
        + delimiter.join(["good.mp3", "صدای سالم"])
        + "\n"
        + delimiter.join(["corrupt.mp3", "صدای خراب"])
        + "\n",
        encoding="utf-8",
    )
    write_tar(
        source_root / "data" / "unvalidated_001.tar",
        {
            "clips/good.mp3": b"good",
            "clips/corrupt.mp3": b"corrupt",
        },
    )

    def fake_converter(audio_bytes: bytes, output_path: Path) -> None:
        if audio_bytes == b"corrupt":
            raise subprocess.CalledProcessError(
                69,
                ["ffmpeg"],
                stderr=b"Decode error rate 1 exceeds maximum 0.666667",
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"wav")

    audit = prepare(
        source_root,
        output_root,
        converter=fake_converter,
        show_progress=False,
    )

    assert read_tsv(output_root / "train.tsv") == [
        {"path": "good.wav", "sentence": "صدای سالم"}
    ]
    assert read_tsv(output_root / "failed_audio.tsv") == [
        {
            "path": "corrupt.mp3",
            "error": "Decode error rate 1 exceeds maximum 0.666667",
        }
    ]
    assert not (output_root / "clips" / "corrupt.wav").exists()
    assert audit.wav_converted == 1
    assert audit.wav_failed == 1
    assert audit.final_train_rows == 1


def test_prepare_does_not_hide_non_ffmpeg_conversion_errors(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "unvalidated.csv").write_text(
        "file_name,sentence\nclip.mp3,سلام\n",
        encoding="utf-8",
    )
    write_tar(source_root / "data" / "unvalidated_001.tar", {"clips/clip.mp3": b"audio"})

    def broken_converter(_audio_bytes: bytes, _output_path: Path) -> None:
        raise OSError("output filesystem unavailable")

    with pytest.raises(OSError, match="output filesystem unavailable"):
        prepare_youtube_persian_asr(
            source_root,
            tmp_path / "normalized",
            converter=broken_converter,
            show_progress=False,
        )


def test_parallel_conversion_results_audit_failures_without_raising() -> None:
    converted: Future[ConversionResult] = Future()
    converted.set_result(ConversionResult("good.mp3"))
    failed: Future[ConversionResult] = Future()
    failed.set_result(ConversionResult("corrupt.mp3", "Decode error rate exceeded"))
    successful: set[str] = set()
    failures: dict[str, str] = {}
    audit = PreparationAudit()

    pending = _finish_futures(
        {converted, failed},
        successful,
        failures,
        audit,
        wait_for_all=True,
    )

    assert pending == set()
    assert successful == {"good.mp3"}
    assert failures == {"corrupt.mp3": "Decode error rate exceeded"}
    assert audit.wav_converted == 1
    assert audit.wav_failed == 1


def test_read_metadata_rejects_duplicate_filenames(tmp_path: Path) -> None:
    metadata_path = tmp_path / "unvalidated.csv"
    metadata_path.write_text(
        "file_name,sentence\n"
        "one.mp3,سلام\n"
        "one.mp3,درود\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate file_name"):
        read_metadata(metadata_path, delimiter=YOUTUBE_DATASET.metadata_delimiter, audit=PreparationAudit())


def test_read_metadata_adds_mp3_extension_and_preserves_delimiters_in_sentence(tmp_path: Path) -> None:
    metadata_path = tmp_path / "unvalidated.csv"
    metadata_path.write_text("file_name,sentence\none,سلام, دوست\n", encoding="utf-8")
    audit = PreparationAudit()

    rows = read_metadata(metadata_path, delimiter=YOUTUBE_DATASET.metadata_delimiter, audit=audit)

    assert rows[0].source_name == "one.mp3"
    assert rows[0].sentence == "سلام دوست"
    assert audit.changed_rows == 1


def test_prepare_rejects_missing_source_files_and_invalid_workers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workers must be >= 1"):
        prepare_filimo_persian_asr(tmp_path, tmp_path / "output", workers=0, show_progress=False)

    with pytest.raises(FileNotFoundError, match="metadata"):
        prepare_youtube_persian_asr(
            tmp_path,
            tmp_path / "output",
            converter=lambda _audio, _path: None,
            show_progress=False,
        )


def test_dataset_defaults_are_separate() -> None:
    assert YOUTUBE_DATASET.default_source_root != FILIMO_DATASET.default_source_root
    assert YOUTUBE_DATASET.default_output_root != FILIMO_DATASET.default_output_root
