from __future__ import annotations

import argparse
import csv
import os
import shutil
import string
import subprocess
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from tqdm import tqdm


csv.field_size_limit(sys.maxsize)

SKIP = set(
    list(string.ascii_letters)
    + [
        "=",  # occurs only 2x in utterance (transl.): "twenty = xx"
        "ā",  # occurs only 4x together with "š"
        "š",
        # Arabic letters
        "ة",  # TEH MARBUTA
    ]
)

DISCARD = [
    # "(laughter)" in Farsi
    "(خنده)",
    # ASCII
    "!",
    '"',
    "#",
    "&",
    "'",
    "(",
    ")",
    ",",
    "-",
    ".",
    ":",
    ";",
    # Unicode punctuation?
    "–",
    "¬",
    "“",
    "”",
    "…",
    "؟",
    "،",
    "؛",
    "ـ",
    # Unicode whitespace?
    "ً",
    "ٌ",
    "َ",
    "ُ",
    "ِ",
    "ّ",
    "ْ",
    "ٔ",
    # Other
    "«",
    "»",
]

REPLACEMENTS = {
    "أ": "ا",
    "ۀ": "ە",
    "ك": "ک",
    "ي": "ی",
    "ى": "ی",
    "ﯽ": "ی",
    "ﻮ": "و",
    "ے": "ی",
    "ﺒ": "ب",
    "ﻢ": "ﻡ",
    "٬": " ",
    "ە": "ه",
}


@dataclass(frozen=True)
class CommonVoiceRow:
    path: str
    sentence: str


@dataclass
class Audit:
    source_validated_rows: int = 0
    source_dev_rows: int = 0
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


def maybe_normalize(text: str) -> str | None:
    if set(text) & SKIP:
        return None

    text = " ".join(w for w in text.split() if not w.startswith("#"))

    for lhs, rhs in REPLACEMENTS.items():
        text = text.replace(lhs, rhs)

    for tok in DISCARD:
        text = text.replace(tok, "")

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ء", "")
    text = remove_punctuation(text)

    return " ".join(t for t in text.split() if t)


def remove_punctuation(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = "".join(char for char in text if not unicodedata.category(char).startswith("P"))
    return " ".join(t for t in text.split() if t)


def read_cv_tsv(path: Path) -> list[CommonVoiceRow]:
    rows: list[CommonVoiceRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"path", "sentence"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain path and sentence columns")
        for row in reader:
            rows.append(CommonVoiceRow(path=row["path"], sentence=row["sentence"]))
    return rows


def wav_name(path: str) -> str:
    return f"{Path(path).stem}.wav"


def normalize_rows(
    rows: Iterable[CommonVoiceRow],
    audit: Audit,
    *,
    keep_rejected_with_raw_text: bool = False,
) -> list[CommonVoiceRow]:
    normalized_rows: list[CommonVoiceRow] = []
    for row in rows:
        normalized = maybe_normalize(row.sentence)
        if normalized is None or not normalized:
            if keep_rejected_with_raw_text:
                audit.test_fallback_rows += 1
                normalized_rows.append(
                    CommonVoiceRow(path=wav_name(row.path), sentence=remove_punctuation(row.sentence))
                )
            else:
                audit.discarded_rows += 1
            continue

        audit.normalized_rows += 1
        if normalized != row.sentence:
            audit.changed_rows += 1
        normalized_rows.append(CommonVoiceRow(path=wav_name(row.path), sentence=normalized))
    return normalized_rows


def build_splits(source_root: Path) -> tuple[dict[str, list[CommonVoiceRow]], Audit]:
    validated_rows = read_cv_tsv(source_root / "validated.tsv")
    official_dev_rows = read_cv_tsv(source_root / "dev.tsv")
    official_test_rows = read_cv_tsv(source_root / "test.tsv")

    audit = Audit(
        source_validated_rows=len(validated_rows),
        source_dev_rows=len(official_dev_rows),
        source_test_rows=len(official_test_rows),
    )

    validated_paths = {row.path for row in validated_rows}
    test_paths = {row.path for row in official_test_rows}
    dev_source_rows = [row for row in official_dev_rows if row.path in validated_paths and row.path not in test_paths]
    dev_paths = {row.path for row in dev_source_rows}
    train_source_rows = [row for row in validated_rows if row.path not in test_paths and row.path not in dev_paths]

    splits = {
        "train": normalize_rows(train_source_rows, audit),
        "dev": normalize_rows(dev_source_rows, audit),
        "test": normalize_rows(official_test_rows, audit, keep_rejected_with_raw_text=True),
    }
    audit.final_train_rows = len(splits["train"])
    audit.final_dev_rows = len(splits["dev"])
    audit.final_test_rows = len(splits["test"])
    return splits, audit


def write_split_tsv(path: Path, rows: Iterable[CommonVoiceRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sentence"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({"path": row.path, "sentence": row.sentence})


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
    source_root: Path,
    output_root: Path,
    rows: Iterable[CommonVoiceRow],
    audit: Audit,
    *,
    converter: Callable[[Path, Path], None] = convert_clip,
    show_progress: bool = True,
    workers: int = 1,
) -> None:
    if converter is convert_clip and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to convert Common Voice MP3 clips to WAV")
    if workers < 1:
        raise ValueError("workers must be >= 1")

    unique_rows = list(dict.fromkeys(row.path for row in rows))
    jobs: list[tuple[Path, Path]] = []
    for wav_path in unique_rows:
        output_path = output_root / "clips" / wav_path
        if output_path.exists():
            audit.wav_skipped_existing += 1
            continue
        source_path = source_root / "clips" / f"{Path(wav_path).stem}.mp3"
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
    print("Common Voice 25 preparation summary")
    print(f"  source validated rows: {audit.source_validated_rows}")
    print(f"  source dev rows: {audit.source_dev_rows}")
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


def prepare_common_voice_25(source_root: Path, output_root: Path, *, workers: int = 1) -> Audit:
    splits, audit = build_splits(source_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_split_tsv(output_root / f"{split}.tsv", rows)

    all_rows = [row for rows in splits.values() for row in rows]
    convert_required_clips(source_root, output_root, all_rows, audit, workers=workers)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare normalized Common Voice 25 Persian data with train/dev/test TSVs "
            "and mono 16 kHz WAV clips."
        )
    )
    parser.add_argument(
        "--source-root",
        default="data/cv-corpus-25.0-2026-03-09/fa",
        help="Common Voice Persian source directory containing validated.tsv, dev.tsv, test.tsv, and clips/.",
    )
    parser.add_argument(
        "--output-root",
        default="data/cv-corpus-25.0",
        help="Output directory for train.tsv, dev.tsv, test.tsv, and converted WAV clips.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of parallel ffmpeg conversion worker processes. Use 1 for single-process conversion.",
    )
    args = parser.parse_args(argv)

    audit = prepare_common_voice_25(Path(args.source_root), Path(args.output_root), workers=args.workers)
    print_audit(audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
