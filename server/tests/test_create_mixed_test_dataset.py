from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ml.speech_data.scripts.create_mixed_test_dataset import (
    DatasetSpec,
    allocate_counts,
    create_mixed_test_dataset,
)


def make_dataset(root: Path, prefix: str, row_count: int, *, paths_include_clips: bool = False) -> None:
    (root / "clips").mkdir(parents=True)
    rows: list[dict[str, str]] = []
    for index in range(row_count):
        relative = Path("nested") / f"{prefix}-{index}.wav"
        audio_path = root / "clips" / relative
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(f"audio-{prefix}-{index}".encode())
        tsv_path = Path("clips") / relative if paths_include_clips else relative
        rows.append({"path": tsv_path.as_posix(), "sentence": f"sentence {prefix} {index}"})
    with (root / "test.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sentence"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_allocate_counts_uses_relative_weights_and_largest_remainders() -> None:
    specs = [
        DatasetSpec("a", Path("a"), 70),
        DatasetSpec("b", Path("b"), 20),
        DatasetSpec("c", Path("c"), 10),
    ]

    assert allocate_counts(11, specs) == {"a": 8, "b": 2, "c": 1}


def test_allocate_counts_rejects_unsafe_dataset_name() -> None:
    with pytest.raises(ValueError, match="dataset names must start"):
        allocate_counts(1, [DatasetSpec("../outside", Path("source"), 1)])


def test_create_mixed_test_dataset_copies_exact_reproducible_mix(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    make_dataset(first_root, "a", 10)
    make_dataset(second_root, "b", 10, paths_include_clips=True)
    specs = [DatasetSpec("first", first_root, 3), DatasetSpec("second", second_root, 1)]

    first_output = tmp_path / "mixed-one"
    second_output = tmp_path / "mixed-two"
    summary = create_mixed_test_dataset(specs, first_output, count=8, seed=42)
    create_mixed_test_dataset(specs, second_output, count=8, seed=42)

    rows = read_rows(first_output / "test.tsv")
    assert summary.selected_per_dataset == {"first": 6, "second": 2}
    assert len(rows) == 8
    assert rows == read_rows(second_output / "test.tsv")
    assert {row["source_dataset"] for row in rows} == {"first", "second"}
    for row in rows:
        assert row["path"].startswith(f"clips/{row['source_dataset']}/nested/")
        assert (first_output / row["path"]).is_file()


def test_create_mixed_test_dataset_rejects_insufficient_source_rows(tmp_path: Path) -> None:
    small_root = tmp_path / "small"
    large_root = tmp_path / "large"
    make_dataset(small_root, "small", 1)
    make_dataset(large_root, "large", 10)
    specs = [DatasetSpec("small", small_root, 1), DatasetSpec("large", large_root, 1)]

    with pytest.raises(ValueError, match="small needs 3 rows.*only 1"):
        create_mixed_test_dataset(specs, tmp_path / "mixed", count=6)

    assert not (tmp_path / "mixed").exists()


def test_create_mixed_test_dataset_rejects_existing_output(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "mixed"
    make_dataset(source_root, "sample", 2)
    output_root.mkdir()

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        create_mixed_test_dataset([DatasetSpec("source", source_root, 1)], output_root, count=1)


def test_create_mixed_test_dataset_rejects_output_containing_source(tmp_path: Path) -> None:
    source_root = tmp_path / "container" / "source"
    make_dataset(source_root, "sample", 2)

    with pytest.raises(ValueError, match="must not contain source dataset source"):
        create_mixed_test_dataset(
            [DatasetSpec("source", source_root, 1)],
            tmp_path / "container",
            count=1,
            overwrite=True,
        )

    assert source_root.is_dir()
