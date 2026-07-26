from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tarfile
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

from huggingface_hub import snapshot_download
from tqdm import tqdm

from ml.speech_data.text_normalization import normalize_persian_asr_text


@dataclass(frozen=True)
class PerSetsDataset:
    label: str
    repo_id: str
    metadata_delimiter: str
    default_source_root: Path
    default_output_root: Path


YOUTUBE_DATASET = PerSetsDataset(
    label="YouTube Persian ASR",
    repo_id="PerSets/youtube-persian-asr",
    metadata_delimiter=",",
    default_source_root=Path("data/youtube-persian-asr/source"),
    default_output_root=Path("data/youtube-persian-asr/normalized"),
)

FILIMO_DATASET = PerSetsDataset(
    label="Filimo Persian ASR",
    repo_id="PerSets/filimo-persian-asr",
    metadata_delimiter="\t",
    default_source_root=Path("data/filimo-persian-asr/source"),
    default_output_root=Path("data/filimo-persian-asr/normalized"),
)


@dataclass
class DownloadAudit:
    repo_id: str
    revision: str
    metadata_files: int = 0
    tar_shards: int = 0
    downloaded_bytes: int = 0


@dataclass
class PreparationAudit:
    source_rows: int = 0
    normalized_rows: int = 0
    changed_rows: int = 0
    discarded_rows: int = 0
    missing_audio_rows: int = 0
    wav_converted: int = 0
    wav_failed: int = 0
    wav_skipped_existing: int = 0
    final_train_rows: int = 0


@dataclass(frozen=True)
class PreparedRow:
    source_name: str
    wav_name: str
    sentence: str


@dataclass(frozen=True)
class ConversionResult:
    source_name: str
    error: str | None = None


SnapshotDownloader = Callable[..., str]
AudioConverter = Callable[[bytes, Path], None]


def _natural_shard_key(path: Path) -> tuple[str, tuple[int, int | str]]:
    suffix = path.stem.rsplit("_", maxsplit=1)[-1]
    ordered_suffix: tuple[int, int | str] = (0, int(suffix)) if suffix.isdigit() else (1, suffix)
    return (path.stem[: -len(suffix)], ordered_suffix)


def source_files(source_root: Path) -> tuple[Path, list[Path]]:
    metadata_path = source_root / "unvalidated.csv"
    tar_shards = sorted((source_root / "data").glob("*.tar"), key=_natural_shard_key)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing PerSets metadata file: {metadata_path}")
    if not tar_shards:
        raise FileNotFoundError(f"no PerSets tar shards found under {source_root / 'data'}")
    return metadata_path, tar_shards


def download_persets_source(
    dataset: PerSetsDataset,
    output_root: Path,
    *,
    revision: str = "main",
    workers: int = 8,
    force: bool = False,
    downloader: SnapshotDownloader = snapshot_download,
) -> DownloadAudit:
    if workers < 1:
        raise ValueError("workers must be >= 1")

    downloader(
        repo_id=dataset.repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=output_root,
        allow_patterns=["unvalidated.csv", "data/*.tar"],
        force_download=force,
        max_workers=workers,
    )
    metadata_path, tar_shards = source_files(output_root)
    downloaded_bytes = metadata_path.stat().st_size + sum(path.stat().st_size for path in tar_shards)
    return DownloadAudit(
        repo_id=dataset.repo_id,
        revision=revision,
        metadata_files=1,
        tar_shards=len(tar_shards),
        downloaded_bytes=downloaded_bytes,
    )


def read_metadata(path: Path, *, delimiter: str, audit: PreparationAudit) -> list[PreparedRow]:
    rows: list[PreparedRow] = []
    source_names: set[str] = set()
    wav_names: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        required = {"file_name", "sentence"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain file_name and sentence columns")

        for line_number, record in enumerate(reader, start=2):
            audit.source_rows += 1
            raw_source_name = str(record.get("file_name") or "").strip()
            source_name = Path(raw_source_name).name
            if not source_name:
                raise ValueError(f"{path}:{line_number} has an empty file_name")
            if not source_name.lower().endswith(".mp3"):
                source_name = f"{source_name}.mp3"
            if source_name in source_names:
                raise ValueError(f"{path}:{line_number} has duplicate file_name {source_name!r}")
            source_names.add(source_name)

            raw_sentence = str(record.get("sentence") or "")
            extra_sentence_fields = record.get(None)
            if isinstance(extra_sentence_fields, list):
                raw_sentence = delimiter.join([raw_sentence, *extra_sentence_fields])
            normalized = normalize_persian_asr_text(raw_sentence)
            if not normalized:
                audit.discarded_rows += 1
                continue

            wav_name = f"{Path(source_name).stem}.wav"
            if wav_name in wav_names:
                raise ValueError(f"{path}:{line_number} maps multiple rows to {wav_name!r}")
            wav_names.add(wav_name)
            audit.normalized_rows += 1
            if normalized != raw_sentence:
                audit.changed_rows += 1
            rows.append(PreparedRow(source_name=source_name, wav_name=wav_name, sentence=normalized))
    return rows


def convert_mp3_bytes(audio_bytes: bytes, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f".{output_path.stem}.part.wav")
    partial_path.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                "pipe:0",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(partial_path),
            ],
            input=audio_bytes,
            stderr=subprocess.PIPE,
            check=True,
        )
        partial_path.replace(output_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise


def _conversion_error_message(error: subprocess.CalledProcessError) -> str:
    stderr = error.stderr
    if isinstance(stderr, bytes):
        detail = stderr.decode("utf-8", errors="replace").strip()
    else:
        detail = str(stderr or "").strip()
    return detail or f"ffmpeg exited with status {error.returncode}"


def _convert_job(job: tuple[str, bytes, str]) -> ConversionResult:
    source_name, audio_bytes, output_path = job
    try:
        convert_mp3_bytes(audio_bytes, Path(output_path))
    except subprocess.CalledProcessError as error:
        return ConversionResult(source_name, _conversion_error_message(error))
    return ConversionResult(source_name)


def _finish_futures(
    futures: set[Future[ConversionResult]],
    successful: set[str],
    failures: dict[str, str],
    audit: PreparationAudit,
    *,
    wait_for_all: bool,
) -> set[Future[str]]:
    if not futures:
        return futures
    done, pending = wait(futures, return_when=ALL_COMPLETED if wait_for_all else FIRST_COMPLETED)
    for future in done:
        result = future.result()
        if result.error is None:
            successful.add(result.source_name)
            audit.wav_converted += 1
        else:
            failures[result.source_name] = result.error
            audit.wav_failed += 1
    return set(pending)


def _write_train_tsv(path: Path, rows: Iterable[PreparedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tsv.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sentence"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({"path": row.wav_name, "sentence": row.sentence})
    temporary_path.replace(path)


def _write_failed_audio_tsv(path: Path, failures: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tsv.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "error"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for source_name in sorted(failures):
            writer.writerow({"path": source_name, "error": failures[source_name]})
    temporary_path.replace(path)


def prepare_persets_dataset(
    dataset: PerSetsDataset,
    source_root: Path,
    output_root: Path,
    *,
    workers: int = 1,
    force: bool = False,
    converter: AudioConverter = convert_mp3_bytes,
    show_progress: bool = True,
) -> PreparationAudit:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if converter is convert_mp3_bytes and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to convert PerSets MP3 clips to WAV")

    metadata_path, tar_shards = source_files(source_root)
    audit = PreparationAudit()
    rows = read_metadata(metadata_path, delimiter=dataset.metadata_delimiter, audit=audit)
    by_source_name = {row.source_name: row for row in rows}
    output_root.mkdir(parents=True, exist_ok=True)

    successful: set[str] = set()
    if not force:
        for row in rows:
            output_path = output_root / "clips" / row.wav_name
            if output_path.is_file() and output_path.stat().st_size > 0:
                successful.add(row.source_name)
                audit.wav_skipped_existing += 1

    seen_in_archives: set[str] = set()
    failures: dict[str, str] = {}
    pending: set[Future[ConversionResult]] = set()
    executor = ProcessPoolExecutor(max_workers=workers) if converter is convert_mp3_bytes and workers > 1 else None
    try:
        shard_iterator = tqdm(
            tar_shards,
            desc=f"Preparing {dataset.label}",
            unit="shard",
            disable=not show_progress,
        )
        for tar_path in shard_iterator:
            with tarfile.open(tar_path, mode="r:*") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    source_name = Path(member.name).name
                    row = by_source_name.get(source_name)
                    if row is None:
                        continue
                    if source_name in seen_in_archives:
                        raise ValueError(f"duplicate audio member {source_name!r} in source tar shards")
                    seen_in_archives.add(source_name)
                    if source_name in successful:
                        continue

                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise ValueError(f"could not read {member.name!r} from {tar_path}")
                    audio_bytes = extracted.read()
                    output_path = output_root / "clips" / row.wav_name
                    if executor is None:
                        try:
                            converter(audio_bytes, output_path)
                        except subprocess.CalledProcessError as error:
                            failures[source_name] = _conversion_error_message(error)
                            audit.wav_failed += 1
                        else:
                            successful.add(source_name)
                            audit.wav_converted += 1
                    else:
                        pending.add(executor.submit(_convert_job, (source_name, audio_bytes, str(output_path))))
                        if len(pending) >= workers * 2:
                            pending = _finish_futures(
                                pending,
                                successful,
                                failures,
                                audit,
                                wait_for_all=False,
                            )

        pending = _finish_futures(
            pending,
            successful,
            failures,
            audit,
            wait_for_all=True,
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    audit.missing_audio_rows = len(set(by_source_name) - seen_in_archives)
    final_rows = [row for row in rows if row.source_name in successful]
    audit.final_train_rows = len(final_rows)
    _write_train_tsv(output_root / "train.tsv", final_rows)
    _write_failed_audio_tsv(output_root / "failed_audio.tsv", failures)
    return audit


def default_worker_count() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def print_download_audit(dataset: PerSetsDataset, audit: DownloadAudit) -> None:
    print(f"{dataset.label} download summary")
    print(f"  repository: {audit.repo_id}")
    print(f"  revision: {audit.revision}")
    print(f"  metadata files: {audit.metadata_files}")
    print(f"  tar shards: {audit.tar_shards}")
    print(f"  downloaded bytes: {audit.downloaded_bytes}")


def print_preparation_audit(dataset: PerSetsDataset, audit: PreparationAudit) -> None:
    print(f"{dataset.label} preparation summary")
    for key, value in asdict(audit).items():
        print(f"  {key.replace('_', ' ')}: {value}")
