from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ml.speech_data.scripts.prepare_common_voice_25 import maybe_normalize, remove_punctuation


csv.field_size_limit(sys.maxsize)

DEFAULT_SPLITS = ("train.tsv", "dev.tsv", "test.tsv")


@dataclass
class NormalizeAudit:
    source_rows: int = 0
    final_rows: int = 0
    normalized_rows: int = 0
    changed_rows: int = 0
    discarded_rows: int = 0


def split_name(value: str) -> str:
    return value if value.endswith(".tsv") else f"{value}.tsv"


def validate_output_root(source_root: Path, output_root: Path) -> None:
    source_resolved = source_root.resolve()
    output_resolved = output_root.resolve()
    if source_resolved == output_resolved:
        raise ValueError("output root must be different from source root")
    try:
        output_resolved.relative_to(source_resolved)
    except ValueError:
        return
    raise ValueError("output root must not be inside source root")


def copy_dataset_tree(source_root: Path, output_root: Path, *, overwrite: bool = False) -> None:
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist or is not a directory: {source_root}")
    validate_output_root(source_root, output_root)
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output root already exists: {output_root}")
        shutil.rmtree(output_root)
    shutil.copytree(source_root, output_root)


def normalize_tsv_file(path: Path, *, discard_rejected: bool = True) -> NormalizeAudit:
    audit = NormalizeAudit()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"path", "sentence"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain path and sentence columns")
        fieldnames = reader.fieldnames
        rows = list(reader)

    normalized_rows: list[dict[str, str]] = []
    audit.source_rows = len(rows)
    for row in rows:
        normalized = maybe_normalize(row.get("sentence", ""))
        if normalized is None or not normalized:
            if discard_rejected:
                audit.discarded_rows += 1
                continue
            row["sentence"] = remove_punctuation(row.get("sentence", ""))
            normalized_rows.append(row)
            continue

        audit.normalized_rows += 1
        if normalized != row.get("sentence", ""):
            audit.changed_rows += 1
        row["sentence"] = normalized
        normalized_rows.append(row)

    audit.final_rows = len(normalized_rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(normalized_rows)
    return audit


def normalize_dataset(
    source_root: Path,
    output_root: Path,
    *,
    splits: Iterable[str] = DEFAULT_SPLITS,
    overwrite: bool = False,
) -> dict[str, NormalizeAudit]:
    split_paths = [split_name(split) for split in splits]
    copy_dataset_tree(source_root, output_root, overwrite=overwrite)

    audits: dict[str, NormalizeAudit] = {}
    for split in split_paths:
        path = output_root / split
        if not path.exists():
            raise FileNotFoundError(f"missing split TSV: {path}")
        audits[split] = normalize_tsv_file(path)
    return audits


def print_audit(audits: dict[str, NormalizeAudit], output_root: Path) -> None:
    print("TSV dataset normalization summary")
    print(f"  output root: {output_root}")
    for split, audit in audits.items():
        print(f"  {split}:")
        print(f"    source rows: {audit.source_rows}")
        print(f"    final rows: {audit.final_rows}")
        print(f"    normalized rows: {audit.normalized_rows}")
        print(f"    changed rows: {audit.changed_rows}")
        print(f"    discarded rows: {audit.discarded_rows}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a TSV ASR dataset to a new output directory and normalize the "
            "sentence column using the same Persian text rules as Common Voice 25 preparation, "
            "including Unicode punctuation removal."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Input dataset directory containing train.tsv, dev.tsv, test.tsv, and related assets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New output directory where the copied dataset and normalized TSVs will be written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split TSV filenames to normalize. Defaults to train.tsv dev.tsv test.tsv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    args = parser.parse_args(argv)

    audits = normalize_dataset(
        args.source_root,
        args.output_root,
        splits=args.splits,
        overwrite=args.overwrite,
    )
    print_audit(audits, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
