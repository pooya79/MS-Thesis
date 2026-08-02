from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

from ml.speech_data.scripts.convert_dataset_to_flac import (
    DEFAULT_SPLITS,
    DEFAULT_SPLIT_STEMS,
    SplitTable,
    discover_splits,
    prepare_conversion,
    read_split,
)


READ_FRAMES = 65_536
FLAC_QUANTIZATION_TOLERANCE = {
    "PCM_16": 1.0 / (2**15),
    "PCM_24": 1.0 / (2**23),
}


@dataclass(frozen=True)
class FlacVerificationAudit:
    checked_files: int
    split_files: int
    audio_files: int
    metadata_files: int


class FlacVerificationError(ValueError):
    def __init__(self, failures: list[str], audit: FlacVerificationAudit) -> None:
        self.failures = failures
        self.audit = audit
        details = "\n".join(f"  - {failure}" for failure in failures)
        super().__init__(f"FLAC conversion verification failed:\n{details}")


def regular_files(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


def source_metadata_files(source_root: Path) -> set[Path]:
    files: set[Path] = set()
    for relative in regular_files(source_root):
        if relative.parts[0] == "clips":
            continue
        if len(relative.parts) == 1 and relative.name in DEFAULT_SPLITS:
            continue
        files.add(relative)
    return files


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def compare_split(expected: SplitTable, output_path: Path) -> str | None:
    if not output_path.is_file():
        return f"missing split file: {output_path}"
    try:
        actual = read_split(output_path)
    except (OSError, ValueError) as exc:
        return str(exc)
    if actual.fieldnames != expected.fieldnames:
        return (
            f"{output_path} columns differ: expected {expected.fieldnames}, "
            f"found {actual.fieldnames}"
        )
    if actual.rows != expected.rows:
        return f"{output_path} rows differ from the converted source rows"
    return None


def compare_audio(source_path: Path, output_path: Path) -> str | None:
    if not output_path.is_file():
        return f"missing converted audio file: {output_path}"
    try:
        source_info = sf.info(source_path)
        output_info = sf.info(output_path)
    except sf.LibsndfileError as exc:
        return f"could not decode audio pair {source_path} -> {output_path}: {exc}"

    if output_info.format != "FLAC":
        return f"{output_path} is {output_info.format}, not FLAC"
    tolerance = FLAC_QUANTIZATION_TOLERANCE.get(output_info.subtype)
    if tolerance is None:
        supported = ", ".join(FLAC_QUANTIZATION_TOLERANCE)
        return f"{output_path} has unsupported FLAC subtype {output_info.subtype}; expected {supported}"

    source_shape = (source_info.frames, source_info.channels, source_info.samplerate)
    output_shape = (output_info.frames, output_info.channels, output_info.samplerate)
    if source_shape != output_shape:
        return (
            f"audio properties differ for {source_path} -> {output_path}: "
            f"expected frames/channels/rate {source_shape}, found {output_shape}"
        )

    try:
        with sf.SoundFile(source_path) as source, sf.SoundFile(output_path) as output:
            offset = 0
            while offset < source_info.frames:
                source_audio = source.read(READ_FRAMES, dtype="float32", always_2d=True)
                output_audio = output.read(READ_FRAMES, dtype="float32", always_2d=True)
                if source_audio.shape != output_audio.shape:
                    return f"decoded block shapes differ at frame {offset}: {source_path} -> {output_path}"
                differences = np.abs(source_audio - output_audio)
                if not np.all(np.isfinite(differences)):
                    return f"non-finite decoded samples at frame {offset}: {source_path} -> {output_path}"
                if differences.size and float(np.max(differences)) > tolerance:
                    maximum = float(np.max(differences))
                    return (
                        f"decoded samples differ at frames {offset}-{offset + len(source_audio) - 1}: "
                        f"{source_path} -> {output_path}; max difference {maximum:.9g}, "
                        f"allowed {tolerance:.9g}"
                    )
                offset += len(source_audio)
    except sf.LibsndfileError as exc:
        return f"could not decode audio pair {source_path} -> {output_path}: {exc}"
    return None


def verify_flac_conversion(
    source_root: Path,
    converted_root: Path,
    *,
    splits: Iterable[str] | None = None,
    log_progress: bool = True,
) -> FlacVerificationAudit:
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist or is not a directory: {source_root}")
    if not (source_root / "clips").is_dir():
        raise FileNotFoundError(f"source dataset is missing clips/: {source_root}")
    if not converted_root.is_dir():
        raise FileNotFoundError(
            f"converted root does not exist or is not a directory: {converted_root}"
        )
    if not (converted_root / "clips").is_dir():
        raise FileNotFoundError(f"converted dataset is missing clips/: {converted_root}")

    split_names = discover_splits(source_root, splits)
    expected_tables, conversions = prepare_conversion(source_root, split_names)
    metadata_files = source_metadata_files(source_root)
    expected_files = {Path(table.name) for table in expected_tables}
    expected_files.update(metadata_files)
    expected_files.update(Path("clips") / relative for relative in conversions.values())

    failures: list[str] = []
    checked_files = 0
    total_files = len(expected_files)

    def log_check(label: str) -> None:
        nonlocal checked_files
        checked_files += 1
        if log_progress:
            print(f"[{checked_files}/{total_files}] checking {label}", flush=True)

    for table in expected_tables:
        output_path = converted_root / table.name
        log_check(f"split {source_root / table.name} -> {output_path}")
        if failure := compare_split(table, output_path):
            failures.append(failure)

    for source_audio, relative_output in conversions.items():
        output_audio = converted_root / "clips" / relative_output
        log_check(f"audio {source_audio} -> {output_audio}")
        if failure := compare_audio(source_audio, output_audio):
            failures.append(failure)

    for relative in sorted(metadata_files):
        source_path = source_root / relative
        output_path = converted_root / relative
        log_check(f"metadata {source_path} -> {output_path}")
        if not output_path.is_file():
            failures.append(f"missing metadata file: {output_path}")
        elif file_digest(source_path) != file_digest(output_path):
            failures.append(f"metadata file contents differ: {source_path} -> {output_path}")

    actual_files = regular_files(converted_root)
    for relative in sorted(actual_files - expected_files):
        failures.append(f"unexpected file in converted dataset: {converted_root / relative}")
    for relative in sorted(expected_files - actual_files):
        missing = f"missing expected file from converted dataset: {converted_root / relative}"
        if not any(str(converted_root / relative) in failure for failure in failures):
            failures.append(missing)

    audit = FlacVerificationAudit(
        checked_files=checked_files,
        split_files=len(expected_tables),
        audio_files=len(conversions),
        metadata_files=len(metadata_files),
    )
    if failures:
        raise FlacVerificationError(failures, audit)
    return audit


def print_audit(audit: FlacVerificationAudit, converted_root: Path) -> None:
    print("FLAC conversion verification passed")
    print(f"  converted root: {converted_root}")
    print(f"  checked files: {audit.checked_files}")
    print(f"  split files: {audit.split_files}")
    print(f"  audio files: {audit.audio_files}")
    print(f"  metadata files: {audit.metadata_files}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a dataset produced by convert_dataset_to_flac has the same "
            "TSV data, copied metadata, and decoded audio as its source dataset."
        )
    )
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Original dataset directory containing split TSV files and clips/.",
    )
    parser.add_argument(
        "--converted-root",
        required=True,
        type=Path,
        help="Converted dataset directory containing FLAC clips and rewritten split TSVs.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=DEFAULT_SPLITS + DEFAULT_SPLIT_STEMS,
        help=(
            "Split TSVs to verify. Defaults to whichever of train.tsv, dev.tsv, and "
            "test.tsv exist in the source dataset."
        ),
    )
    args = parser.parse_args(argv)

    try:
        audit = verify_flac_conversion(
            args.source_root,
            args.converted_root,
            splits=args.splits,
        )
    except FlacVerificationError as exc:
        print(exc)
        return 1
    print_audit(audit, args.converted_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
