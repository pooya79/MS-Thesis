from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from xml.etree import ElementTree

from tqdm import tqdm

from ml.speech_data.scripts.prepare_common_voice_25 import write_split_tsv
from ml.speech_data.text_normalization import normalize_persian_asr_text as maybe_normalize
from ml.speech_data.text_normalization import remove_punctuation


@dataclass(frozen=True)
class SourceRow:
    path: str
    sentence: str
    source_audio_path: Path


@dataclass(frozen=True)
class PreparedRow:
    path: str
    sentence: str
    source_audio_path: Path


@dataclass
class DatasetAudit:
    source_rows: int = 0
    final_rows: int = 0
    normalized_rows: int = 0
    changed_rows: int = 0
    discarded_rows: int = 0
    missing_audio_rows: int = 0
    test_fallback_rows: int = 0
    wav_converted: int = 0
    wav_skipped_existing: int = 0


@dataclass
class PrepareAudit:
    persian_speech_corpus: DatasetAudit
    persian_speech: DatasetAudit


def wav_name(path: str) -> str:
    return f"{Path(path).stem}.wav"


def validate_output_root(output_root: Path, source_root: Path) -> None:
    output_resolved = output_root.resolve()
    source_resolved = source_root.resolve()
    try:
        output_resolved.relative_to(source_resolved)
    except ValueError:
        return
    raise ValueError("output roots must not be inside the extraction source root")


def ensure_new_output_root(output_root: Path, source_root: Path, *, force: bool) -> None:
    validate_output_root(output_root, source_root)
    if output_root.exists():
        if not force:
            raise FileExistsError(f"output root already exists: {output_root}; pass --force")
        shutil.rmtree(output_root)


def safe_members_zip(archive: zipfile.ZipFile, output_dir: Path) -> None:
    output_resolved = output_dir.resolve()
    for member in archive.infolist():
        target = (output_dir / member.filename).resolve()
        if not target.is_relative_to(output_resolved):
            raise RuntimeError(f"unsafe archive member path: {member.filename}")


def safe_members_tar(archive: tarfile.TarFile, output_dir: Path) -> None:
    output_resolved = output_dir.resolve()
    for member in archive.getmembers():
        target = (output_dir / member.name).resolve()
        if not target.is_relative_to(output_resolved):
            raise RuntimeError(f"unsafe archive member path: {member.name}")


def extract_archive(archive_path: Path, output_dir: Path, *, force: bool = False) -> None:
    if output_dir.exists():
        if not force:
            return
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            safe_members_zip(archive, output_dir)
            archive.extractall(output_dir)
        return
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            safe_members_tar(archive, output_dir)
            archive.extractall(output_dir, filter="data")
        return
    raise ValueError(f"unsupported archive type: {archive_path}")


def parse_orthographic_transcript(path: Path) -> list[tuple[str, str]]:
    pattern = re.compile(r'^\s*"?(?P<audio>[^"\s]+)"?\s+"(?P<sentence>.+?)"\s*$')
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            match = pattern.match(line)
            if match is None:
                raise ValueError(f"could not parse transcript row {path}:{line_number}")
            rows.append((match.group("audio"), match.group("sentence")))
    return rows


def audio_stem_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            index.setdefault(path.stem, path)
    return index


def read_persian_speech_corpus_rows(source_root: Path, *, missing_audio: list[str] | None = None) -> list[SourceRow]:
    transcript_paths = sorted(source_root.rglob("orthographic-transcript.txt"))
    if not transcript_paths:
        raise FileNotFoundError(f"missing orthographic-transcript.txt under {source_root}")
    transcript_path = transcript_paths[0]
    indexed_audio = audio_stem_index(source_root)
    rows: list[SourceRow] = []
    for raw_path, sentence in parse_orthographic_transcript(transcript_path):
        stem = Path(raw_path).stem
        source_audio_path = indexed_audio.get(stem)
        if source_audio_path is None:
            if missing_audio is not None:
                missing_audio.append(raw_path)
                continue
            raise FileNotFoundError(f"missing Persian Speech Corpus audio for transcript item {raw_path}")
        rows.append(SourceRow(path=wav_name(raw_path), sentence=sentence, source_audio_path=source_audio_path))
    return rows


def xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        payload = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ElementTree.fromstring(payload)
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings: list[str] = []
    for item in root.findall("x:si", namespace):
        strings.append("".join(node.text or "" for node in item.findall(".//x:t", namespace)))
    return strings


def first_worksheet_path(archive: zipfile.ZipFile) -> str:
    namespace = {
        "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    sheet = workbook.find("x:sheets/x:sheet", namespace)
    if sheet is None:
        raise ValueError("xlsx workbook has no worksheets")
    relationship_id = sheet.attrib[f"{{{namespace['r']}}}id"]

    relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_namespace = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
    for relationship in relationships.findall("rel:Relationship", rel_namespace):
        if relationship.attrib.get("Id") == relationship_id:
            target = relationship.attrib["Target"]
            return f"xl/{target}" if not target.startswith("/") else target.lstrip("/")
    raise ValueError("xlsx worksheet relationship was not found")


def cell_column_index(cell_ref: str) -> int:
    letters = "".join(char for char in cell_ref if char.isalpha()).upper()
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return max(0, index - 1)


def cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    value = cell.find("x:v", namespace)
    if value is None or value.text is None:
        inline = cell.find(".//x:t", namespace)
        return inline.text if inline is not None and inline.text is not None else ""
    raw = value.text
    if cell.attrib.get("t") == "s":
        return shared_strings[int(raw)]
    return raw


def read_xlsx_rows(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = xlsx_shared_strings(archive)
        worksheet = ElementTree.fromstring(archive.read(first_worksheet_path(archive)))
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows: list[list[str]] = []
    for row in worksheet.findall("x:sheetData/x:row", namespace):
        values: list[str] = []
        for fallback_index, cell in enumerate(row.findall("x:c", namespace)):
            ref = cell.attrib.get("r", "")
            column_index = cell_column_index(ref) if ref else fallback_index
            while len(values) <= column_index:
                values.append("")
            values[column_index] = cell_value(cell, shared_strings).strip()
        if any(values):
            rows.append(values)
    return rows


def choose_column(headers: list[str], names: set[str], *, default: int) -> int:
    lowered = [header.strip().lower().replace(" ", "_") for header in headers]
    for index, header in enumerate(lowered):
        if header in names or any(name in header for name in names):
            return index
    return default


def find_persian_speech_data_root(source_root: Path) -> Path:
    candidates = sorted(path.parent for path in source_root.rglob("myaudio") if path.is_dir())
    return candidates[0] if candidates else source_root


def read_persian_speech_rows(source_root: Path, metadata_path: Path, *, missing_audio: list[str] | None = None) -> list[SourceRow]:
    rows = read_xlsx_rows(metadata_path)
    if not rows:
        raise ValueError(f"empty PersianSpeech metadata workbook: {metadata_path}")
    headers = rows[0]
    body = rows[1:]
    audio_index = choose_column(headers, {"audio", "path", "file", "filename", "wav"}, default=0)
    sentence_index = choose_column(headers, {"text", "sentence", "transcript", "transcription", "label"}, default=1)
    data_root = find_persian_speech_data_root(source_root)
    indexed_audio = audio_stem_index(source_root)

    source_rows: list[SourceRow] = []
    for line_number, values in enumerate(body, start=2):
        if max(audio_index, sentence_index) >= len(values):
            continue
        raw_audio_path = values[audio_index]
        sentence = values[sentence_index]
        if not raw_audio_path or not sentence:
            continue
        source_audio_path = data_root / raw_audio_path
        if not source_audio_path.exists():
            source_audio_path = indexed_audio.get(Path(raw_audio_path).stem, source_audio_path)
        if not source_audio_path.exists():
            if missing_audio is not None:
                missing_audio.append(raw_audio_path)
                continue
            raise FileNotFoundError(f"missing PersianSpeech audio for metadata row {line_number}: {raw_audio_path}")
        source_rows.append(SourceRow(path=wav_name(raw_audio_path), sentence=sentence, source_audio_path=source_audio_path))
    return source_rows


def normalize_eval_rows(rows: Iterable[SourceRow], audit: DatasetAudit) -> list[PreparedRow]:
    prepared_rows: list[PreparedRow] = []
    for row in rows:
        normalized = maybe_normalize(row.sentence)
        if normalized is None or not normalized:
            if row.sentence.strip():
                audit.test_fallback_rows += 1
                prepared_rows.append(
                    PreparedRow(
                        path=wav_name(row.path),
                        sentence=remove_punctuation(row.sentence),
                        source_audio_path=row.source_audio_path,
                    )
                )
            else:
                audit.discarded_rows += 1
            continue
        audit.normalized_rows += 1
        if normalized != row.sentence:
            audit.changed_rows += 1
        prepared_rows.append(PreparedRow(path=wav_name(row.path), sentence=normalized, source_audio_path=row.source_audio_path))
    return prepared_rows


def convert_clip(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        check=True,
    )


def convert_clip_job(paths: tuple[str, str]) -> None:
    convert_clip(Path(paths[0]), Path(paths[1]))


def convert_required_clips(
    output_root: Path,
    rows: Iterable[PreparedRow],
    audit: DatasetAudit,
    *,
    converter: Callable[[Path, Path], None] = convert_clip,
    show_progress: bool = True,
    workers: int = 1,
) -> None:
    if converter is convert_clip and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to prepare Persian evaluation WAV clips")
    if workers < 1:
        raise ValueError("workers must be >= 1")

    unique_rows: dict[str, Path] = {}
    for row in rows:
        unique_rows.setdefault(row.path, row.source_audio_path)

    jobs: list[tuple[Path, Path]] = []
    for output_name, source_path in unique_rows.items():
        output_path = output_root / "clips" / output_name
        if output_path.exists():
            audit.wav_skipped_existing += 1
            continue
        jobs.append((source_path, output_path))

    if not jobs:
        return
    if workers == 1 or converter is not convert_clip:
        iterator = tqdm(jobs, desc="Converting clips", unit="clip", disable=not show_progress)
        for source_path, output_path in iterator:
            converter(source_path, output_path)
            audit.wav_converted += 1
        return

    serialized_jobs = [(str(source_path), str(output_path)) for source_path, output_path in jobs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(convert_clip_job, job) for job in serialized_jobs]
        iterator = tqdm(as_completed(futures), total=len(futures), desc="Converting clips", unit="clip", disable=not show_progress)
        for future in iterator:
            future.result()
            audit.wav_converted += 1


def prepare_dataset(
    output_root: Path,
    source_rows: list[SourceRow],
    *,
    converter: Callable[[Path, Path], None] = convert_clip,
    workers: int = 1,
    show_progress: bool = True,
) -> DatasetAudit:
    audit = DatasetAudit(source_rows=len(source_rows))
    rows = normalize_eval_rows(source_rows, audit)
    audit.final_rows = len(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    write_split_tsv(output_root / "test.tsv", rows)
    convert_required_clips(output_root, rows, audit, converter=converter, workers=workers, show_progress=show_progress)
    return audit


def prepare_persian_eval_sets(
    *,
    cache_dir: Path = Path("data/downloads/persian_eval_sets"),
    source_root: Path = Path("data/persian_eval_sets/source"),
    persian_speech_corpus_output_root: Path = Path("data/persian-speech-corpus-test"),
    persian_speech_output_root: Path = Path("data/PersianSpeech_test"),
    force: bool = False,
    workers: int = 1,
    show_progress: bool = True,
    converter: Callable[[Path, Path], None] = convert_clip,
) -> PrepareAudit:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    ensure_new_output_root(persian_speech_corpus_output_root, source_root, force=force)
    ensure_new_output_root(persian_speech_output_root, source_root, force=force)

    psc_archive = cache_dir / "persian-speech-corpus.zip"
    ps_archive = cache_dir / "myaudio_tiny.tar.gz"
    ps_metadata = cache_dir / "myaudio_tiny.xlsx"
    for path in (psc_archive, ps_archive, ps_metadata):
        if not path.exists():
            raise FileNotFoundError(f"missing downloaded file: {path}")

    psc_source_root = source_root / "persian-speech-corpus"
    ps_source_root = source_root / "PersianSpeech"
    extract_archive(psc_archive, psc_source_root, force=force)
    extract_archive(ps_archive, ps_source_root, force=force)

    psc_missing_audio: list[str] = []
    ps_missing_audio: list[str] = []
    psc_rows = read_persian_speech_corpus_rows(psc_source_root, missing_audio=psc_missing_audio)
    ps_rows = read_persian_speech_rows(ps_source_root, ps_metadata, missing_audio=ps_missing_audio)
    psc_audit = prepare_dataset(
        persian_speech_corpus_output_root,
        psc_rows,
        converter=converter,
        workers=workers,
        show_progress=show_progress,
    )
    psc_audit.source_rows += len(psc_missing_audio)
    psc_audit.missing_audio_rows = len(psc_missing_audio)
    ps_audit = prepare_dataset(
        persian_speech_output_root,
        ps_rows,
        converter=converter,
        workers=workers,
        show_progress=show_progress,
    )
    ps_audit.source_rows += len(ps_missing_audio)
    ps_audit.missing_audio_rows = len(ps_missing_audio)
    return PrepareAudit(persian_speech_corpus=psc_audit, persian_speech=ps_audit)


def print_dataset_audit(name: str, output_root: Path, audit: DatasetAudit) -> None:
    print(f"{name} preparation summary")
    print(f"  output root: {output_root}")
    print(f"  source rows: {audit.source_rows}")
    print(f"  final rows: {audit.final_rows}")
    print(f"  normalized rows: {audit.normalized_rows}")
    print(f"  changed rows: {audit.changed_rows}")
    print(f"  discarded rows: {audit.discarded_rows}")
    print(f"  missing audio rows: {audit.missing_audio_rows}")
    print(f"  test fallback rows: {audit.test_fallback_rows}")
    print(f"  wav converted: {audit.wav_converted}")
    print(f"  wav skipped existing: {audit.wav_skipped_existing}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract downloaded Persian evaluation archives and prepare repo-style "
            "test.tsv datasets with mono 16 kHz WAV clips."
        )
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/downloads/persian_eval_sets"), help="Directory containing persian-speech-corpus.zip, myaudio_tiny.tar.gz, and myaudio_tiny.xlsx.")
    parser.add_argument("--source-root", type=Path, default=Path("data/persian_eval_sets/source"), help="Directory where archives will be extracted.")
    parser.add_argument("--persian-speech-corpus-output-root", type=Path, default=Path("data/persian-speech-corpus-test"), help="Output dataset directory for Persian Speech Corpus test.tsv and clips/.")
    parser.add_argument("--persian-speech-output-root", type=Path, default=Path("data/PersianSpeech_test"), help="Output dataset directory for PersianSpeech test.tsv and clips/.")
    parser.add_argument("--force", action="store_true", help="Replace output directories and re-extract source archives.")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)), help="Number of parallel ffmpeg conversion worker processes. Use 1 for single-process conversion.")
    args = parser.parse_args(argv)

    audit = prepare_persian_eval_sets(
        cache_dir=args.cache_dir,
        source_root=args.source_root,
        persian_speech_corpus_output_root=args.persian_speech_corpus_output_root,
        persian_speech_output_root=args.persian_speech_output_root,
        force=args.force,
        workers=args.workers,
    )
    print_dataset_audit("Persian Speech Corpus", args.persian_speech_corpus_output_root, audit.persian_speech_corpus)
    print_dataset_audit("PersianSpeech", args.persian_speech_output_root, audit.persian_speech)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
