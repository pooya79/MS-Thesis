from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ml.speech_data.scripts.normalize_tsv_dataset import normalize_dataset


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["path", "sentence", "speaker"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def make_dataset(root: Path) -> None:
    (root / "clips").mkdir(parents=True)
    (root / "clips" / "sample.wav").write_bytes(b"fake-audio")
    rows = [
        {"path": "clips/sample.wav", "sentence": "خب ، تو چیكار می كنی؟", "speaker": "a"},
        {"path": "clips/rejected.wav", "sentence": "hello سلام", "speaker": "b"},
    ]
    for split in ("train.tsv", "dev.tsv", "test.tsv"):
        write_tsv(root / split, rows)


def test_normalize_dataset_copies_tree_and_normalizes_default_splits(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "normalized"
    make_dataset(source_root)

    audits = normalize_dataset(source_root, output_root)

    assert (output_root / "clips" / "sample.wav").read_bytes() == b"fake-audio"
    for split in ("train.tsv", "dev.tsv", "test.tsv"):
        rows = read_tsv(output_root / split)
        assert rows == [{"path": "clips/sample.wav", "sentence": "خب تو چیکار می کنی", "speaker": "a"}]
        assert audits[split].source_rows == 2
        assert audits[split].final_rows == 1
        assert audits[split].changed_rows == 1
        assert audits[split].discarded_rows == 1

    assert read_tsv(source_root / "train.tsv")[0]["sentence"] == "خب ، تو چیكار می كنی؟"


def test_normalize_dataset_rejects_existing_output_without_overwrite(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "normalized"
    make_dataset(source_root)
    output_root.mkdir()

    with pytest.raises(FileExistsError, match="output root already exists"):
        normalize_dataset(source_root, output_root)


def test_normalize_dataset_rejects_output_inside_source(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    make_dataset(source_root)

    with pytest.raises(ValueError, match="must not be inside source root"):
        normalize_dataset(source_root, source_root / "normalized")
