from __future__ import annotations

import argparse
import csv
import math
import random
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    proportion: float


@dataclass(frozen=True)
class SourceRow:
    dataset_name: str
    sentence: str
    source_audio: Path
    relative_audio: Path


@dataclass(frozen=True)
class MixSummary:
    total_rows: int
    selected_per_dataset: dict[str, int]


def parse_dataset_spec(values: Sequence[str]) -> DatasetSpec:
    name, raw_root, raw_proportion = values
    if name in {".", ".."} or DATASET_NAME_PATTERN.fullmatch(name) is None:
        raise argparse.ArgumentTypeError(
            "dataset NAME must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_', or '-'"
        )
    try:
        proportion = float(raw_proportion)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid proportion for {name}: {raw_proportion}") from exc
    if not math.isfinite(proportion) or proportion <= 0:
        raise argparse.ArgumentTypeError(f"proportion for {name} must be a finite number greater than zero")
    return DatasetSpec(name=name, root=Path(raw_root), proportion=proportion)


def validate_specs(specs: Sequence[DatasetSpec]) -> None:
    if not specs:
        raise ValueError("at least one dataset is required")
    names = [spec.name for spec in specs]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"dataset names must be unique; duplicates: {', '.join(duplicate_names)}")
    for spec in specs:
        if spec.name in {".", ".."} or DATASET_NAME_PATTERN.fullmatch(spec.name) is None:
            raise ValueError(
                "dataset names must start with an alphanumeric character and contain only "
                "letters, numbers, '.', '_', or '-'"
            )
        if not math.isfinite(spec.proportion) or spec.proportion <= 0:
            raise ValueError(f"proportion for {spec.name} must be a finite number greater than zero")


def allocate_counts(total_count: int, specs: Sequence[DatasetSpec]) -> dict[str, int]:
    if total_count <= 0:
        raise ValueError("count must be greater than zero")
    validate_specs(specs)
    total_weight = sum(spec.proportion for spec in specs)
    exact_counts = [total_count * spec.proportion / total_weight for spec in specs]
    counts = [math.floor(value) for value in exact_counts]
    remaining = total_count - sum(counts)
    remainder_order = sorted(
        range(len(specs)),
        key=lambda index: (-(exact_counts[index] - counts[index]), index),
    )
    for index in remainder_order[:remaining]:
        counts[index] += 1
    return {spec.name: count for spec, count in zip(specs, counts, strict=True)}


def resolve_audio_path(dataset_root: Path, raw_path: str) -> tuple[Path, Path]:
    relative_path = Path(raw_path)
    if relative_path.is_absolute():
        raise ValueError(f"audio path must be relative to its dataset: {raw_path}")

    root = dataset_root.resolve()
    clips_root = (dataset_root / "clips").resolve()
    candidates = [dataset_root / "clips" / relative_path, dataset_root / relative_path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            continue
        if not resolved.is_file():
            continue
        try:
            output_relative = resolved.relative_to(clips_root)
        except ValueError:
            output_relative = resolved.relative_to(root)
        return resolved, output_relative
    raise FileNotFoundError(f"audio referenced by test.tsv was not found under {dataset_root}: {raw_path}")


def read_test_rows(spec: DatasetSpec) -> list[SourceRow]:
    tsv_path = spec.root / "test.tsv"
    if not tsv_path.is_file():
        raise FileNotFoundError(f"missing test.tsv for dataset {spec.name}: {tsv_path}")

    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"path", "sentence"}.issubset(reader.fieldnames):
            raise ValueError(f"{tsv_path} must contain path and sentence columns")
        rows: list[SourceRow] = []
        for line_number, row in enumerate(reader, start=2):
            raw_path = (row.get("path") or "").strip()
            if not raw_path:
                raise ValueError(f"empty audio path at {tsv_path}:{line_number}")
            source_audio, relative_audio = resolve_audio_path(spec.root, raw_path)
            rows.append(
                SourceRow(
                    dataset_name=spec.name,
                    sentence=row.get("sentence") or "",
                    source_audio=source_audio,
                    relative_audio=relative_audio,
                )
            )
    return rows


def validate_output_root(output_root: Path, specs: Sequence[DatasetSpec]) -> None:
    output = output_root.resolve()
    for spec in specs:
        source = spec.root.resolve()
        if output == source or output.is_relative_to(source):
            raise ValueError(f"output root must not be the same as or inside source dataset {spec.name}")
        if source.is_relative_to(output):
            raise ValueError(f"output root must not contain source dataset {spec.name}")


def select_rows(
    specs: Sequence[DatasetSpec],
    rows_by_dataset: dict[str, list[SourceRow]],
    counts: dict[str, int],
    *,
    seed: int,
) -> list[SourceRow]:
    selected: list[SourceRow] = []
    for spec in specs:
        rows = rows_by_dataset[spec.name]
        count = counts[spec.name]
        if count > len(rows):
            raise ValueError(
                f"dataset {spec.name} needs {count} rows for the requested proportion "
                f"but test.tsv contains only {len(rows)}"
            )
        dataset_rng = random.Random(f"{seed}:{spec.name}")
        selected.extend(dataset_rng.sample(rows, count))
    random.Random(seed).shuffle(selected)
    return selected


def write_dataset(staging_root: Path, selected_rows: Sequence[SourceRow]) -> None:
    output_rows: list[dict[str, str]] = []
    destinations: dict[Path, Path] = {}
    for row in selected_rows:
        relative_destination = Path("clips") / row.dataset_name / row.relative_audio
        previous_source = destinations.get(relative_destination)
        if previous_source is not None and previous_source != row.source_audio:
            raise ValueError(
                f"two source clips map to the same output path {relative_destination}: "
                f"{previous_source} and {row.source_audio}"
            )
        destinations[relative_destination] = row.source_audio
        destination = staging_root / relative_destination
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row.source_audio, destination)
        output_rows.append(
            {
                "path": relative_destination.as_posix(),
                "sentence": row.sentence,
                "source_dataset": row.dataset_name,
            }
        )

    with (staging_root / "test.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "sentence", "source_dataset"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)


def create_mixed_test_dataset(
    specs: Sequence[DatasetSpec],
    output_root: Path,
    *,
    count: int,
    seed: int = 0,
    overwrite: bool = False,
) -> MixSummary:
    validate_specs(specs)
    validate_output_root(output_root, specs)
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"output root already exists: {output_root}; pass --overwrite")

    counts = allocate_counts(count, specs)
    rows_by_dataset = {spec.name: read_test_rows(spec) for spec in specs}
    selected_rows = select_rows(specs, rows_by_dataset, counts, seed=seed)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_root.name}-", dir=output_root.parent) as temp_dir:
        staging_root = Path(temp_dir)
        write_dataset(staging_root, selected_rows)
        if output_root.exists():
            shutil.rmtree(output_root)
        staging_root.rename(output_root)

    return MixSummary(total_rows=len(selected_rows), selected_per_dataset=counts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic mixed ASR test dataset by randomly sampling test.tsv "
            "rows from multiple source datasets in user-defined proportions."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=3,
        required=True,
        metavar=("NAME", "PATH", "PROPORTION"),
        help=(
            "Source dataset name, root path, and positive relative weight. Repeat for every "
            "dataset; weights such as 70/30 and 0.7/0.3 are equivalent."
        ),
    )
    parser.add_argument("--count", type=int, required=True, help="Exact total number of test clips to select.")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New dataset directory in which test.tsv and copied clips/ will be created.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible selection (default: 0).")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    args = parser.parse_args(argv)
    specs = [parse_dataset_spec(values) for values in args.dataset]
    summary = create_mixed_test_dataset(
        specs,
        args.output_root,
        count=args.count,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    print(f"Created mixed test dataset with {summary.total_rows} clips at {args.output_root}")
    for spec in specs:
        print(f"  {spec.name}: {summary.selected_per_dataset[spec.name]}")
    print(f"  seed: {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
