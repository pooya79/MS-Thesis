from __future__ import annotations

import csv
import zipfile
from pathlib import Path

from ml.speech_data.scripts.download_persian_eval_sets import (
    DatasetAudit,
    PreparedRow,
    build_persian_speech_corpus_rows,
    build_persian_speech_rows,
    convert_required_clips,
    extract_archive,
    normalize_prepared_rows,
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

    row_xml = []
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


def test_build_persian_speech_corpus_rows_reads_orthographic_transcript(tmp_path: Path) -> None:
    source_root = tmp_path / "psc"
    (source_root / "wav").mkdir(parents=True)
    (source_root / "wav" / "utt001.wav").write_bytes(b"audio")
    (source_root / "orthographic-transcript.txt").write_text('"utt001" "خب ، تو چیكار می كنی؟"\n', encoding="utf-8")

    rows = build_persian_speech_corpus_rows(source_root)
    audit = DatasetAudit(source_rows=len(rows))
    normalized = normalize_prepared_rows(rows, audit)

    assert normalized == [PreparedRow(path="psc-utt001.wav", sentence="خب تو چیکار می کنی", source_audio_path=source_root / "wav" / "utt001.wav")]
    assert audit.normalized_rows == 1


def test_build_persian_speech_rows_reads_xlsx_metadata(tmp_path: Path) -> None:
    source_root = tmp_path / "persian-speech"
    (source_root / "clips").mkdir(parents=True)
    (source_root / "clips" / "a.wav").write_bytes(b"audio-a")
    metadata_path = tmp_path / "myaudio_tiny.xlsx"
    write_minimal_xlsx(metadata_path, [["file", "text"], ["a.wav", "سلام! «دوست»؛"]])

    rows = build_persian_speech_rows(source_root, metadata_path)
    audit = DatasetAudit(source_rows=len(rows))
    normalized = normalize_prepared_rows(rows, audit)

    assert normalized == [PreparedRow(path="ps-a.wav", sentence="سلام دوست", source_audio_path=source_root / "clips" / "a.wav")]


def test_convert_required_clips_writes_clips_directory_and_tsv_shape(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()
    (source_root / "a.wav").write_bytes(b"a")
    rows = [PreparedRow(path="a.wav", sentence="سلام", source_audio_path=source_root / "a.wav")]
    converted: list[tuple[Path, Path]] = []

    def fake_converter(source: Path, output: Path) -> None:
        converted.append((source, output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source.read_bytes())

    audit = DatasetAudit()
    from ml.speech_data.scripts.prepare_common_voice_25 import write_split_tsv

    output_root.mkdir()
    write_split_tsv(output_root / "test.tsv", rows)
    convert_required_clips(output_root, rows, audit, converter=fake_converter, show_progress=False)

    assert read_simple_tsv(output_root / "test.tsv") == [{"path": "a.wav", "sentence": "سلام"}]
    assert converted == [(source_root / "a.wav", output_root / "clips" / "a.wav")]
    assert (output_root / "clips" / "a.wav").read_bytes() == b"a"
    assert audit.wav_converted == 1


def test_extract_archive_rejects_unsafe_zip_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "bad")

    import pytest

    with pytest.raises(RuntimeError, match="unsafe archive member"):
        extract_archive(archive_path, tmp_path / "out")
