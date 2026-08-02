from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import soundfile as sf


csv.field_size_limit(sys.maxsize)

DEFAULT_SPLITS = ("train.tsv", "dev.tsv", "test.tsv")
DEFAULT_SPLIT_STEMS = tuple(Path(split).stem for split in DEFAULT_SPLITS)
FLAC_SUBTYPES = ("PCM_16", "PCM_24")


@dataclass(frozen=True)
class FlacConversionAudit:
    split_rows: int
    unique_clips: int
    source_bytes: int
    output_bytes: int


@dataclass(frozen=True)
class SplitTable:
    name: str
    fieldnames: list[str]
    rows: list[dict[str, str]]


def split_name(value: str) -> str:
    return value if value.endswith(".tsv") else f"{value}.tsv"


def validate_roots(source_root: Path, output_root: Path) -> None:
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist or is not a directory: {source_root}")
    if not (source_root / "clips").is_dir():
        raise FileNotFoundError(f"source dataset is missing clips/: {source_root}")

    source_resolved = source_root.resolve()
    output_resolved = output_root.resolve()
    if source_resolved == output_resolved:
        raise ValueError("output root must be different from source root")
    try:
        output_resolved.relative_to(source_resolved)
    except ValueError:
        return
    raise ValueError("output root must not be inside source root")


def discover_splits(source_root: Path, splits: Iterable[str] | None) -> list[str]:
    if splits is None:
        names = [name for name in DEFAULT_SPLITS if (source_root / name).is_file()]
        if not names:
            expected = ", ".join(DEFAULT_SPLITS)
            raise FileNotFoundError(f"missing split TSV: expected one of {expected} under {source_root}")
        return names

    names = [split_name(split) for split in splits]
    for name in names:
        path = source_root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing split TSV: {path}")
    return names


def read_split(path: Path) -> SplitTable:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"path", "sentence"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain path and sentence columns")
        return SplitTable(path.name, list(reader.fieldnames), list(reader))


def resolve_audio_path(source_root: Path, value: str, *, split: str, row_number: int) -> Path:
    raw_path = Path(value)
    if raw_path.is_absolute():
        candidates = [raw_path]
    else:
        candidates = [source_root / "clips" / raw_path, source_root / raw_path]
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            try:
                resolved.relative_to(source_root.resolve())
            except ValueError as exc:
                raise ValueError(f"{split}:{row_number} audio path is outside the dataset: {resolved}") from exc
            return resolved
    raise FileNotFoundError(f"{split}:{row_number} missing audio file: {candidates[0]}")


def output_clip_path(source_root: Path, source_audio: Path) -> Path:
    clips_root = (source_root / "clips").resolve()
    try:
        relative = source_audio.relative_to(clips_root)
    except ValueError:
        relative = source_audio.relative_to(source_root.resolve())
    return relative.with_suffix(".flac")


def prepare_conversion(
    source_root: Path,
    split_names: list[str],
) -> tuple[list[SplitTable], dict[Path, Path]]:
    tables = [read_split(source_root / name) for name in split_names]
    conversions: dict[Path, Path] = {}
    destinations: dict[Path, Path] = {}

    for table in tables:
        for row_number, row in enumerate(table.rows, start=2):
            raw_path = str(row.get("path", "")).strip()
            if not raw_path:
                raise ValueError(f"{table.name}:{row_number} has an empty path")
            source_audio = resolve_audio_path(
                source_root,
                raw_path,
                split=table.name,
                row_number=row_number,
            )
            relative_output = output_clip_path(source_root, source_audio)
            previous_source = destinations.get(relative_output)
            if previous_source is not None and previous_source != source_audio:
                raise ValueError(
                    "multiple source clips would map to the same FLAC path: "
                    f"{previous_source} and {source_audio} -> clips/{relative_output}"
                )
            destinations[relative_output] = source_audio
            conversions[source_audio] = relative_output
            row["path"] = relative_output.as_posix()
    return tables, conversions


def copy_dataset_metadata(source_root: Path, output_root: Path) -> None:
    ignored = {"clips", *DEFAULT_SPLITS}
    source_resolved = source_root.resolve()

    def ignore(directory: str, names: list[str]) -> set[str]:
        if Path(directory).resolve() != source_resolved:
            return set()
        return {name for name in names if name in ignored}

    shutil.copytree(source_root, output_root, ignore=ignore)
    (output_root / "clips").mkdir(parents=True, exist_ok=True)


def convert_clip(source_audio: Path, output_audio: Path, subtype: str) -> None:
    try:
        audio, sample_rate = sf.read(source_audio, dtype="float32", always_2d=False)
    except sf.LibsndfileError as exc:
        raise ValueError(f"could not decode audio file: {source_audio}") from exc
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_audio.with_name(f".{output_audio.name}.part")
    try:
        sf.write(temporary, audio, sample_rate, format="FLAC", subtype=subtype)
        temporary.replace(output_audio)
    finally:
        temporary.unlink(missing_ok=True)


def write_split(output_root: Path, table: SplitTable) -> None:
    path = output_root / table.name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=table.fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(table.rows)


def convert_dataset_to_flac(
    source_root: Path,
    output_root: Path,
    *,
    splits: Iterable[str] | None = None,
    subtype: str = "PCM_16",
    overwrite: bool = False,
    log_progress: bool = True,
) -> FlacConversionAudit:
    if subtype not in FLAC_SUBTYPES:
        raise ValueError(f"subtype must be one of: {', '.join(FLAC_SUBTYPES)}")
    validate_roots(source_root, output_root)
    split_names = discover_splits(source_root, splits)
    tables, conversions = prepare_conversion(source_root, split_names)

    if output_root.exists() and not overwrite:
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    staging_root = staging_parent / "dataset"
    source_bytes = 0
    output_bytes = 0
    try:
        copy_dataset_metadata(source_root, staging_root)
        total_clips = len(conversions)
        for index, (source_audio, relative_output) in enumerate(conversions.items(), start=1):
            output_audio = staging_root / "clips" / relative_output
            if log_progress:
                print(
                    f"[{index}/{total_clips}] converting {source_audio} -> clips/{relative_output.as_posix()}",
                    flush=True,
                )
            convert_clip(source_audio, output_audio, subtype)
            clip_source_bytes = source_audio.stat().st_size
            clip_output_bytes = output_audio.stat().st_size
            source_bytes += clip_source_bytes
            output_bytes += clip_output_bytes
            if log_progress:
                clip_saved = clip_source_bytes - clip_output_bytes
                total_saved = source_bytes - output_bytes
                print(
                    f"[{index}/{total_clips}] complete: {format_bytes(clip_source_bytes)} -> "
                    f"{format_bytes(clip_output_bytes)}; saved {format_bytes(clip_saved)} "
                    f"(cumulative {format_bytes(total_saved)})",
                    flush=True,
                )
        for table in tables:
            write_split(staging_root, table)
        if output_root.exists():
            shutil.rmtree(output_root)
        staging_root.replace(output_root)
    except Exception:
        raise
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)

    return FlacConversionAudit(
        split_rows=sum(len(table.rows) for table in tables),
        unique_clips=len(conversions),
        source_bytes=source_bytes,
        output_bytes=output_bytes,
    )


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


def print_audit(audit: FlacConversionAudit, output_root: Path) -> None:
    saved = audit.source_bytes - audit.output_bytes
    percent = (saved / audit.source_bytes * 100) if audit.source_bytes else 0.0
    print("FLAC dataset conversion summary")
    print(f"  output root: {output_root}")
    print(f"  split rows: {audit.split_rows}")
    print(f"  unique clips: {audit.unique_clips}")
    print(f"  source audio size: {format_bytes(audit.source_bytes)}")
    print(f"  FLAC audio size: {format_bytes(audit.output_bytes)}")
    print(f"  space saved: {format_bytes(saved)} ({percent:.1f}%)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a TSV ASR dataset to a new directory, encode every referenced "
            "audio clip as FLAC, and update TSV path values. The source dataset is unchanged."
        )
    )
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Input dataset directory containing split TSV files and clips/.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="New output dataset directory. It must not be inside the source directory.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=DEFAULT_SPLITS + DEFAULT_SPLIT_STEMS,
        help=(
            "Split TSVs to include. Defaults to whichever of train.tsv, dev.tsv, and "
            "test.tsv exist. Unselected split TSVs are not copied."
        ),
    )
    parser.add_argument(
        "--subtype",
        choices=FLAC_SUBTYPES,
        default="PCM_16",
        help="FLAC PCM bit depth (default: %(default)s). Use PCM_24 for sources above 16-bit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    args = parser.parse_args(argv)

    audit = convert_dataset_to_flac(
        args.source_root,
        args.output_root,
        splits=args.splits,
        subtype=args.subtype,
        overwrite=args.overwrite,
    )
    print_audit(audit, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
