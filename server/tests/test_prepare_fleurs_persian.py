from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ml.speech_data.scripts.prepare_fleurs_persian import (
    Audit,
    PreparedRow,
    build_splits,
    convert_required_clips,
    prepare_fleurs_persian,
)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_source_split(source_root: Path, split: str, records: list[tuple[str, str]]) -> None:
    jsonl_records: list[dict[str, object]] = []
    for row_id, sentence in records:
        audio_path = Path("audio") / split / f"{row_id}.wav"
        full_audio_path = source_root / audio_path
        full_audio_path.parent.mkdir(parents=True, exist_ok=True)
        full_audio_path.write_bytes(f"audio-{row_id}".encode())
        jsonl_records.append(
            {
                "id": row_id,
                "split": split,
                "audio_path": str(audio_path),
                "sample_rate": 16000,
                "transcription": sentence,
            }
        )
    write_jsonl(source_root / f"{split}.jsonl", jsonl_records)


def read_simple_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def build_fake_source(source_root: Path) -> None:
    write_source_split(
        source_root,
        "train",
        [
            ("train-a", "سلام! «دوست»؛"),
            ("train-skip", "hello سلام"),
        ],
    )
    write_source_split(source_root, "validation", [("valid-a", "خب ، تو چیكار می كنی؟")])
    write_source_split(
        source_root,
        "test",
        [
            ("test-a", "این تست است."),
            ("test-reject", "hello تست؟."),
        ],
    )


def test_build_splits_maps_validation_to_dev_and_normalizes_rows(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    build_fake_source(source_root)

    splits, audit = build_splits(source_root)

    assert [row.path for row in splits["train"]] == ["train-a.wav"]
    assert splits["train"][0].sentence == "سلام دوست"
    assert [row.path for row in splits["dev"]] == ["valid-a.wav"]
    assert splits["dev"][0].sentence == "خب تو چیکار می کنی"
    assert [row.path for row in splits["test"]] == ["test-a.wav", "test-reject.wav"]
    assert splits["test"][0].sentence == "این تست است"
    assert splits["test"][1].sentence == "hello تست"
    assert audit.source_validation_rows == 1
    assert audit.discarded_rows == 1
    assert audit.test_fallback_rows == 1


def test_prepare_fleurs_persian_writes_common_voice_style_tsvs_and_clips(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "normalized"
    build_fake_source(source_root)
    converted: list[tuple[Path, Path]] = []

    def fake_converter(source: Path, output: Path) -> None:
        converted.append((source, output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source.read_bytes())

    splits, audit = build_splits(source_root)
    output_root.mkdir(parents=True)
    for split, rows in splits.items():
        from ml.speech_data.scripts.prepare_fleurs_persian import write_prepared_split_tsv

        write_prepared_split_tsv(output_root / f"{split}.tsv", rows)
    all_rows = [row for rows in splits.values() for row in rows]
    convert_required_clips(output_root, all_rows, audit, converter=fake_converter, show_progress=False)

    assert read_simple_tsv(output_root / "train.tsv") == [{"path": "train-a.wav", "sentence": "سلام دوست"}]
    assert read_simple_tsv(output_root / "dev.tsv") == [{"path": "valid-a.wav", "sentence": "خب تو چیکار می کنی"}]
    assert read_simple_tsv(output_root / "test.tsv") == [
        {"path": "test-a.wav", "sentence": "این تست است"},
        {"path": "test-reject.wav", "sentence": "hello تست"},
    ]
    assert (output_root / "clips" / "train-a.wav").read_bytes() == b"audio-train-a"
    assert len(converted) == 4


def test_prepare_fleurs_persian_runs_end_to_end_with_injected_converter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "normalized"
    build_fake_source(source_root)

    def fake_convert_required_clips(output_root: Path, rows: object, audit: Audit, *, workers: int = 1) -> None:
        for row in rows:
            assert isinstance(row, PreparedRow)
            output_path = output_root / "clips" / row.path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(row.source_audio_path.read_bytes())
            audit.wav_converted += 1

    monkeypatch.setattr("ml.speech_data.scripts.prepare_fleurs_persian.convert_required_clips", fake_convert_required_clips)

    audit = prepare_fleurs_persian(source_root, output_root, workers=3)

    assert audit.final_train_rows == 1
    assert audit.final_dev_rows == 1
    assert audit.final_test_rows == 2
    assert audit.wav_converted == 4
    assert (output_root / "dev.tsv").exists()


def test_convert_required_clips_converts_unique_targets_and_skips_existing(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    (source_root / "a.wav").parent.mkdir(parents=True, exist_ok=True)
    (source_root / "a.wav").write_bytes(b"a")
    (source_root / "b.wav").write_bytes(b"b")
    (output_root / "clips").mkdir(parents=True)
    (output_root / "clips" / "b.wav").write_bytes(b"existing")
    converted: list[tuple[Path, Path]] = []

    def fake_converter(source: Path, output: Path) -> None:
        converted.append((source, output))
        output.write_bytes(source.read_bytes())

    audit = Audit()
    rows = [
        PreparedRow("a.wav", "یک", source_root / "a.wav"),
        PreparedRow("a.wav", "یک دوباره", source_root / "a.wav"),
        PreparedRow("b.wav", "دو", source_root / "b.wav"),
    ]

    convert_required_clips(output_root, rows, audit, converter=fake_converter, show_progress=False)

    assert converted == [(source_root / "a.wav", output_root / "clips" / "a.wav")]
    assert audit.wav_converted == 1
    assert audit.wav_skipped_existing == 1


def test_convert_required_clips_rejects_invalid_worker_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workers must be >= 1"):
        convert_required_clips(tmp_path, [], Audit(), show_progress=False, workers=0)
