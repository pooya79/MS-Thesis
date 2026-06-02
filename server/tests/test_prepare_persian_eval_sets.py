from __future__ import annotations

import csv
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from ml.speech_data.scripts.prepare_persian_eval_sets import (
    DatasetAudit,
    SourceRow,
    extract_archive,
    normalize_eval_rows,
    prepare_dataset,
    prepare_persian_eval_sets,
    read_persian_speech_corpus_rows,
    read_persian_speech_rows,
)


def read_simple_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_minimal_xlsx(path: Path, rows: list[list[str]]) -> None:
    shared_strings: list[str] = []
    string_indexes: dict[str, int] = {}

    def shared_index(value: str) -> int:
        if value not in string_indexes:
            string_indexes[value] = len(shared_strings)
            shared_strings.append(value)
        return string_indexes[value]

    row_xml: list[str] = []
    for row_number, row in enumerate(rows, start=1):
        cells = []
        for column_number, value in enumerate(row):
            column_name = chr(ord("A") + column_number)
            cells.append(f'<c r="{column_name}{row_number}" t="s"><v>{shared_index(value)}</v></c>')
        row_xml.append(f'<row r="{row_number}">{"".join(cells)}</row>')

    shared_xml = "".join(f"<si><t>{value}</t></si>" for value in shared_strings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "")
        archive.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="worksheet" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<sheetData>{"".join(row_xml)}</sheetData></worksheet>',
        )
        archive.writestr(
            "xl/sharedStrings.xml",
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f"{shared_xml}</sst>",
        )


def write_persian_speech_corpus_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("orthographic-transcript.txt", '"001-A.wav" "خب ، تو چیكار می كنی؟"\n')
        archive.writestr("wav/001-A.wav", b"psc-audio")


def write_persian_speech_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        content = b"ps-audio"
        info = tarfile.TarInfo("myaudio_tiny/myaudio/450001.wav")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    path.write_bytes(payload.getvalue())


def test_read_persian_speech_corpus_rows_maps_transcript_to_audio(tmp_path: Path) -> None:
    source_root = tmp_path / "psc"
    (source_root / "wav").mkdir(parents=True)
    (source_root / "wav" / "001-A.wav").write_bytes(b"audio")
    (source_root / "orthographic-transcript.txt").write_text('"001-A.wav" "خب ، تو چیكار می كنی؟"\n', encoding="utf-8")

    rows = read_persian_speech_corpus_rows(source_root)

    assert rows == [SourceRow(path="001-A.wav", sentence="خب ، تو چیكار می كنی؟", source_audio_path=source_root / "wav" / "001-A.wav")]


def test_read_persian_speech_corpus_rows_can_collect_missing_audio(tmp_path: Path) -> None:
    source_root = tmp_path / "psc"
    source_root.mkdir()
    (source_root / "orthographic-transcript.txt").write_text('"106-A.wav" "متن بدون صوت"\n', encoding="utf-8")
    missing_audio: list[str] = []

    rows = read_persian_speech_corpus_rows(source_root, missing_audio=missing_audio)

    assert rows == []
    assert missing_audio == ["106-A.wav"]


def test_read_persian_speech_rows_maps_xlsx_audio_column(tmp_path: Path) -> None:
    source_root = tmp_path / "ps"
    (source_root / "myaudio_tiny" / "myaudio").mkdir(parents=True)
    (source_root / "myaudio_tiny" / "myaudio" / "450001.wav").write_bytes(b"audio")
    metadata_path = tmp_path / "myaudio_tiny.xlsx"
    write_minimal_xlsx(metadata_path, [["audio", "text"], ["myaudio/450001.wav", "سلام! «دوست»؛"]])

    rows = read_persian_speech_rows(source_root, metadata_path)

    assert rows == [
        SourceRow(
            path="450001.wav",
            sentence="سلام! «دوست»؛",
            source_audio_path=source_root / "myaudio_tiny" / "myaudio" / "450001.wav",
        )
    ]


def test_normalize_eval_rows_keeps_rejected_text_as_test_fallback(tmp_path: Path) -> None:
    source_audio = tmp_path / "a.wav"
    source_audio.write_bytes(b"audio")
    audit = DatasetAudit()
    rows = [SourceRow(path="a.wav", sentence="hello سلام", source_audio_path=source_audio)]

    normalized = normalize_eval_rows(rows, audit)

    assert normalized[0].sentence == "hello سلام"
    assert audit.test_fallback_rows == 1
    assert audit.discarded_rows == 0


def test_prepare_dataset_writes_test_tsv_and_converts_wavs(tmp_path: Path) -> None:
    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"audio")
    output_root = tmp_path / "dataset"
    converted: list[tuple[Path, Path]] = []

    def fake_converter(source: Path, output: Path) -> None:
        converted.append((source, output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source.read_bytes())

    audit = prepare_dataset(
        output_root,
        [SourceRow(path="source.wav", sentence="سلام! «دوست»؛", source_audio_path=source_audio)],
        converter=fake_converter,
        show_progress=False,
    )

    assert read_simple_tsv(output_root / "test.tsv") == [{"path": "source.wav", "sentence": "سلام دوست"}]
    assert (output_root / "clips" / "source.wav").read_bytes() == b"audio"
    assert converted == [(source_audio, output_root / "clips" / "source.wav")]
    assert audit.wav_converted == 1


def test_prepare_persian_eval_sets_extracts_archives_and_prepares_outputs(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    write_persian_speech_corpus_archive(cache_dir / "persian-speech-corpus.zip")
    write_persian_speech_archive(cache_dir / "myaudio_tiny.tar.gz")
    write_minimal_xlsx(cache_dir / "myaudio_tiny.xlsx", [["audio", "text"], ["myaudio/450001.wav", "سلام! «دوست»؛"]])
    converted: list[tuple[Path, Path]] = []

    def fake_converter(source: Path, output: Path) -> None:
        converted.append((source, output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source.read_bytes())

    audit = prepare_persian_eval_sets(
        cache_dir=cache_dir,
        source_root=tmp_path / "source",
        persian_speech_corpus_output_root=tmp_path / "psc-out",
        persian_speech_output_root=tmp_path / "ps-out",
        converter=fake_converter,
        show_progress=False,
    )

    assert audit.persian_speech_corpus.final_rows == 1
    assert audit.persian_speech.final_rows == 1
    assert read_simple_tsv(tmp_path / "psc-out" / "test.tsv") == [{"path": "001-A.wav", "sentence": "خب تو چیکار می کنی"}]
    assert read_simple_tsv(tmp_path / "ps-out" / "test.tsv") == [{"path": "450001.wav", "sentence": "سلام دوست"}]
    assert len(converted) == 2


def test_prepare_persian_eval_sets_skips_missing_audio_rows(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    with zipfile.ZipFile(cache_dir / "persian-speech-corpus.zip", "w") as archive:
        archive.writestr(
            "orthographic-transcript.txt",
            '"001-A.wav" "خب ، تو چیكار می كنی؟"\n"106-A.wav" "متن بدون صوت"\n',
        )
        archive.writestr("wav/001-A.wav", b"psc-audio")
    write_persian_speech_archive(cache_dir / "myaudio_tiny.tar.gz")
    write_minimal_xlsx(cache_dir / "myaudio_tiny.xlsx", [["audio", "text"], ["myaudio/450001.wav", "سلام! «دوست»؛"]])

    def fake_converter(source: Path, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source.read_bytes())

    audit = prepare_persian_eval_sets(
        cache_dir=cache_dir,
        source_root=tmp_path / "source",
        persian_speech_corpus_output_root=tmp_path / "psc-out",
        persian_speech_output_root=tmp_path / "ps-out",
        converter=fake_converter,
        show_progress=False,
    )

    assert audit.persian_speech_corpus.source_rows == 2
    assert audit.persian_speech_corpus.final_rows == 1
    assert audit.persian_speech_corpus.missing_audio_rows == 1
    assert read_simple_tsv(tmp_path / "psc-out" / "test.tsv") == [{"path": "001-A.wav", "sentence": "خب تو چیکار می کنی"}]


def test_extract_archive_rejects_unsafe_zip_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "bad")

    with pytest.raises(RuntimeError, match="unsafe archive member"):
        extract_archive(archive_path, tmp_path / "out")
