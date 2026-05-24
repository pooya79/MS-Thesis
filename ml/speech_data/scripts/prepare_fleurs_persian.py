from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from tqdm import tqdm

from ml.speech_data.scripts.prepare_common_voice_25 import maybe_normalize, write_split_tsv


SOURCE_TO_OUTPUT_SPLIT = {
    "train": "train",
    "validation": "dev",
    "test": "test",
}


@dataclass(frozen=True)
class FleursRow:
    path: str
    sentence: str
    source_audio_path: Path


@dataclass(frozen=True)
class PreparedRow:
    path: str
    sentence: str
    source_audio_path: Path


@dataclass
class Audit:
    source_train_rows: int = 0
    source_validation_rows: int = 0
    source_test_rows: int = 0
    final_train_rows: int = 0
    final_dev_rows: int = 0
    final_test_rows: int = 0
    normalized_rows: int = 0
    changed_rows: int = 0
    discarded_rows: int = 0
    test_fallback_rows: int = 0
    wav_converted: int = 0
    wav_skipped_existing: int = 0


def wav_name(path: str) -> str:
    return f"{Path(path).stem}.wav"


def read_source_jsonl(source_root: Path, split: str) -> list[FleursRow]:
    path = source_root / f"{split}.jsonl"
    rows: list[FleursRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            audio_path = record.get("audio_path")
            transcript = record.get("transcription")
            if not audio_path or transcript is None:
                raise ValueError(f"{path}:{line_number} must contain audio_path and transcription")
            rows.append(
                FleursRow(
                    path=wav_name(str(audio_path)),
                    sentence=str(transcript),
                    source_audio_path=source_root / str(audio_path),
                )
            )
    return rows


def normalize_rows(
    rows: Iterable[FleursRow],
    audit: Audit,
    *,
    keep_rejected_with_raw_text: bool = False,
) -> list[PreparedRow]:
    normalized_rows: list[PreparedRow] = []
    for row in rows:
        normalized = maybe_normalize(row.sentence)
        if normalized is None or not normalized:
            if keep_rejected_with_raw_text:
                audit.test_fallback_rows += 1
                normalized_rows.append(
                    PreparedRow(path=wav_name(row.path), sentence=row.sentence, source_audio_path=row.source_audio_path)
                )
            else:
                audit.discarded_rows += 1
            continue

        audit.normalized_rows += 1
        if normalized != row.sentence:
            audit.changed_rows += 1
        normalized_rows.append(PreparedRow(path=wav_name(row.path), sentence=normalized, source_audio_path=row.source_audio_path))
    return normalized_rows


def build_splits(source_root: Path) -> tuple[dict[str, list[PreparedRow]], Audit]:
    source_rows = {split: read_source_jsonl(source_root, split) for split in SOURCE_TO_OUTPUT_SPLIT}
    audit = Audit(
        source_train_rows=len(source_rows["train"]),
        source_validation_rows=len(source_rows["validation"]),
        source_test_rows=len(source_rows["test"]),
    )

    splits = {
        "train": normalize_rows(source_rows["train"], audit),
        "dev": normalize_rows(source_rows["validation"], audit),
        "test": normalize_rows(source_rows["test"], audit, keep_rejected_with_raw_text=True),
    }
    audit.final_train_rows = len(splits["train"])
    audit.final_dev_rows = len(splits["dev"])
    audit.final_test_rows = len(splits["test"])
    return splits, audit


def write_prepared_split_tsv(path: Path, rows: Iterable[PreparedRow]) -> None:
    write_split_tsv(path, rows)


def convert_clip(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        check=True,
    )


def convert_clip_job(paths: tuple[str, str]) -> None:
    source_path, output_path = paths
    convert_clip(Path(source_path), Path(output_path))


def convert_required_clips(
    output_root: Path,
    rows: Iterable[PreparedRow],
    audit: Audit,
    *,
    converter: Callable[[Path, Path], None] = convert_clip,
    show_progress: bool = True,
    workers: int = 1,
) -> None:
    if converter is convert_clip and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to convert FLEURS clips to WAV")
    if workers < 1:
        raise ValueError("workers must be >= 1")

    unique_rows: dict[str, Path] = {}
    for row in rows:
        unique_rows.setdefault(row.path, row.source_audio_path)

    jobs: list[tuple[Path, Path]] = []
    for wav_path, source_path in unique_rows.items():
        output_path = output_root / "clips" / wav_path
        if output_path.exists():
            audit.wav_skipped_existing += 1
            continue
        if not source_path.exists():
            raise FileNotFoundError(f"missing source clip: {source_path}")
        jobs.append((source_path, output_path))

    if not jobs:
        return

    if workers == 1 or converter is not convert_clip:
        iterator = tqdm(jobs, desc="Converting clips", unit="clip", disable=not show_progress)
        for source_path, output_path in iterator:
            converter(source_path, output_path)
            audit.wav_converted += 1
        return

    serialized_jobs = [(str(source_path), str(output_path)) for source_path, output_path in jobs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(convert_clip_job, job) for job in serialized_jobs]
        iterator = tqdm(as_completed(futures), total=len(futures), desc="Converting clips", unit="clip", disable=not show_progress)
        for future in iterator:
            future.result()
            audit.wav_converted += 1


def print_audit(audit: Audit) -> None:
    print("FLEURS Persian preparation summary")
    print(f"  source train rows: {audit.source_train_rows}")
    print(f"  source validation rows: {audit.source_validation_rows}")
    print(f"  source test rows: {audit.source_test_rows}")
    print(f"  final train rows: {audit.final_train_rows}")
    print(f"  final dev rows: {audit.final_dev_rows}")
    print(f"  final test rows: {audit.final_test_rows}")
    print(f"  normalized rows: {audit.normalized_rows}")
    print(f"  changed rows: {audit.changed_rows}")
    print(f"  discarded rows: {audit.discarded_rows}")
    print(f"  test fallback rows: {audit.test_fallback_rows}")
    print(f"  wav converted: {audit.wav_converted}")
    print(f"  wav skipped existing: {audit.wav_skipped_existing}")


def prepare_fleurs_persian(source_root: Path, output_root: Path, *, workers: int = 1) -> Audit:
    splits, audit = build_splits(source_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_prepared_split_tsv(output_root / f"{split}.tsv", rows)

    all_rows = [row for rows in splits.values() for row in rows]
    convert_required_clips(output_root, all_rows, audit, workers=workers)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare normalized Persian FLEURS data with train/dev/test TSVs "
            "and mono 16 kHz WAV clips."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/fleurs/fa_ir/source"),
        help="FLEURS Persian source directory containing train.jsonl, validation.jsonl, test.jsonl, and audio/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/fleurs/fa_ir/normalized"),
        help="Output directory for train.tsv, dev.tsv, test.tsv, and converted WAV clips.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of parallel ffmpeg conversion worker processes. Use 1 for single-process conversion.",
    )
    args = parser.parse_args(argv)

    audit = prepare_fleurs_persian(args.source_root, args.output_root, workers=args.workers)
    print_audit(audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
