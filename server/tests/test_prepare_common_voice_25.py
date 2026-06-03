from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ml.speech_data.scripts.prepare_common_voice_25 import (
    Audit,
    CommonVoiceRow,
    build_splits,
    convert_required_clips,
    maybe_normalize,
    write_split_tsv,
)


def write_cv_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "client_id",
        "path",
        "sentence_id",
        "sentence",
        "sentence_domain",
        "up_votes",
        "down_votes",
        "age",
        "gender",
        "accents",
        "variant",
        "locale",
        "segment",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            complete = {key: "" for key in fieldnames}
            complete.update(
                {
                    "client_id": f"client-{index}",
                    "sentence_id": f"sentence-{index}",
                    "up_votes": "2",
                    "down_votes": "0",
                    "locale": "fa",
                }
            )
            complete.update(row)
            writer.writerow(complete)


def read_simple_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_maybe_normalize_matches_nvidia_card_rules() -> None:
    assert maybe_normalize("خب ، تو چیكار می كنی؟") == "خب تو چیکار می کنی"
    assert maybe_normalize("أۀك ي ى ﯽ ﻮ ے ﺒ ﻢ ٬") == "اهک ی ی ی و ی ب م"
    assert maybe_normalize("سلام! «دوست»؛") == "سلام دوست"
    assert maybe_normalize("سلام [دوست] / امروز") == "سلام دوست امروز"
    assert maybe_normalize("سلام #خوانده_نمیشود دنیا") == "سلام دنیا"
    assert maybe_normalize("  سلام    دنیا  ") == "سلام دنیا"
    assert maybe_normalize("hello سلام") is None


def test_build_splits_preserves_test_and_excludes_it_from_train_dev(tmp_path: Path) -> None:
    source_root = tmp_path / "fa"
    source_root.mkdir()

    validated = [
        {"path": "train-a.mp3", "sentence": "آه، سلام!"},
        {"path": "train-skip.mp3", "sentence": "hello سلام"},
        {"path": "dev-a.mp3", "sentence": "خب ، تو چیكار می كنی؟"},
        {"path": "test-a.mp3", "sentence": "این تست است."},
        {"path": "test-reject.mp3", "sentence": "hello تست"},
    ]
    dev = [
        {"path": "dev-a.mp3", "sentence": "خب ، تو چیكار می كنی؟"},
        {"path": "dev-not-validated.mp3", "sentence": "این نباید بیاید"},
        {"path": "test-a.mp3", "sentence": "این تست است."},
    ]
    test = [
        {"path": "test-a.mp3", "sentence": "این تست است."},
        {"path": "test-reject.mp3", "sentence": "hello تست"},
    ]
    write_cv_tsv(source_root / "validated.tsv", validated)
    write_cv_tsv(source_root / "dev.tsv", dev)
    write_cv_tsv(source_root / "test.tsv", test)

    splits, audit = build_splits(source_root)

    assert [row.path for row in splits["test"]] == ["test-a.wav", "test-reject.wav"]
    assert splits["test"][0].sentence == "این تست است"
    assert splits["test"][1].sentence == "hello تست"
    assert [row.path for row in splits["dev"]] == ["dev-a.wav"]
    assert splits["dev"][0].sentence == "خب تو چیکار می کنی"
    assert [row.path for row in splits["train"]] == ["train-a.wav"]
    assert splits["train"][0].sentence == "آه سلام"
    assert audit.discarded_rows == 1
    assert audit.test_fallback_rows == 1
    assert audit.final_test_rows == 2


def test_write_split_tsv_uses_only_path_and_sentence(tmp_path: Path) -> None:
    output = tmp_path / "train.tsv"
    write_split_tsv(output, [CommonVoiceRow(path="clip.wav", sentence="سلام")])

    rows = read_simple_tsv(output)

    assert rows == [{"path": "clip.wav", "sentence": "سلام"}]
    assert output.read_text(encoding="utf-8").splitlines()[0] == "path\tsentence"


def test_convert_required_clips_converts_unique_wav_targets(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    (source_root / "clips").mkdir(parents=True)
    (source_root / "clips" / "a.mp3").write_bytes(b"fake-a")
    (source_root / "clips" / "b.mp3").write_bytes(b"fake-b")
    (output_root / "clips").mkdir(parents=True)
    (output_root / "clips" / "b.wav").write_bytes(b"existing")
    converted: list[tuple[Path, Path]] = []

    def fake_converter(source: Path, output: Path) -> None:
        converted.append((source, output))
        output.write_bytes(source.read_bytes())

    audit = Audit()
    rows = [CommonVoiceRow("a.wav", "یک"), CommonVoiceRow("a.wav", "یک دوباره"), CommonVoiceRow("b.wav", "دو")]

    convert_required_clips(source_root, output_root, rows, audit, converter=fake_converter, show_progress=False)

    assert converted == [(source_root / "clips" / "a.mp3", output_root / "clips" / "a.wav")]
    assert (output_root / "clips" / "a.wav").read_bytes() == b"fake-a"
    assert audit.wav_converted == 1
    assert audit.wav_skipped_existing == 1


def test_convert_required_clips_rejects_invalid_worker_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workers must be >= 1"):
        convert_required_clips(tmp_path, tmp_path, [], Audit(), show_progress=False, workers=0)
