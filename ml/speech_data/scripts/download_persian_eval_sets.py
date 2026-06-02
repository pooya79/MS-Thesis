from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from xml.etree import ElementTree

from tqdm import tqdm

from ml.speech_data.scripts.download_common_voice_fa import download_url_to_file
from ml.speech_data.scripts.prepare_common_voice_25 import maybe_normalize, write_split_tsv


PERSIAN_SPEECH_CORPUS_URL = "https://en.persianspeechcorpus.com/persian-speech-corpus.zip"
PERSIAN_SPEECH_DRIVE_FILE_ID = "1cCWH_eoa4Nq17XDHn6e1WIfHomdGWPKO"
PERSIAN_SPEECH_URL = f"https://drive.google.com/uc?export=download&id={PERSIAN_SPEECH_DRIVE_FILE_ID}"
PERSIAN_SPEECH_METADATA_URL = "https://raw.githubusercontent.com/persiandataset/PersianSpeech/main/myaudio_tiny.xlsx"


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
    raw_fallback_rows: int = 0
    wav_converted: int = 0
    wav_skipped_existing: int = 0


@dataclass
class Audit:
    persian_speech_corpus: DatasetAudit
    persian_speech: DatasetAudit


def safe_name(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return normalized or fallback


def wav_name(path: str, *, prefix: str = "") -> str:
    stem = safe_name(Path(path).stem, fallback="clip")
    return f"{prefix}{stem}.wav"


def normalize_prepared_rows(rows: Iterable[PreparedRow], audit: DatasetAudit, *, keep_rejected_with_raw_text: bool = True) -> list[PreparedRow]:
    normalized_rows: list[PreparedRow] = []
    for row in rows:
        normalized = maybe_normalize(row.sentence)
        if normalized is None or not normalized:
            if keep_rejected_with_raw_text and row.sentence.strip():
                audit.raw_fallback_rows += 1
                normalized_rows.append(row)
            else:
                audit.discarded_rows += 1
            continue

        audit.normalized_rows += 1
        if normalized != row.sentence:
            audit.changed_rows += 1
        normalized_rows.append(PreparedRow(path=row.path, sentence=normalized, source_audio_path=row.source_audio_path))
    return normalized_rows


def download_google_drive_file(
    url: str,
    output_path: Path,
    *,
    force: bool = False,
    show_progress: bool = True,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path.stat().st_size

    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    with opener.open(url) as response:
        html = response.read()
        final_url = response.geturl()

    token_match = re.search(rb"confirm=([0-9A-Za-z_%-]+)", html)
    if "drive.google.com" in final_url and token_match:
        parsed = urllib.parse.urlparse(final_url)
        query = urllib.parse.parse_qs(parsed.query)
        query["confirm"] = [token_match.group(1).decode("ascii")]
        confirm_url = urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query, doseq=True)))
        with opener.open(confirm_url) as response, output_path.open("wb") as handle:
            with tqdm.wrapattr(
                response,
                "read",
                desc=f"Downloading {output_path.name}",
                unit="B",
                unit_scale=True,
                disable=not show_progress,
            ) as reader:
                shutil.copyfileobj(reader, handle)
    else:
        output_path.write_bytes(html)
    return output_path.stat().st_size


def download_public_url(url: str, output_path: Path, *, force: bool = False, show_progress: bool = True) -> int:
    if output_path.exists() and not force:
        return output_path.stat().st_size
    return download_url_to_file(url, output_path, force=force, resume=False, show_progress=show_progress)


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (output_dir / member.filename).resolve()
                if not target.is_relative_to(output_dir.resolve()):
                    raise RuntimeError(f"unsafe archive member path: {member.filename}")
            archive.extractall(output_dir)
        return

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                target = (output_dir / member.name).resolve()
                if not target.is_relative_to(output_dir.resolve()):
                    raise RuntimeError(f"unsafe archive member path: {member.name}")
            archive.extractall(output_dir)
        return

    raise ValueError(f"unsupported archive type: {archive_path}")


def parse_orthographic_transcript(path: Path) -> list[tuple[str, str]]:
    pattern = re.compile(r'^\s*"?(?P<name>[^"\s]+)"?\s+"(?P<text>.+?)"\s*$')
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            match = pattern.match(line)
            if match is None:
                raise ValueError(f"could not parse transcript row {path}:{line_number}")
            rows.append((match.group("name"), match.group("text")))
    return rows


def find_audio_by_stem(root: Path) -> dict[str, Path]:
    audio_by_stem: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            audio_by_stem.setdefault(path.stem, path)
    return audio_by_stem


def build_persian_speech_corpus_rows(source_root: Path) -> list[PreparedRow]:
    transcript_candidates = sorted(source_root.rglob("orthographic-transcript.txt"))
    if not transcript_candidates:
        raise FileNotFoundError(f"missing orthographic-transcript.txt under {source_root}")
    transcript_path = transcript_candidates[0]
    audio_by_stem = find_audio_by_stem(source_root)

    rows: list[PreparedRow] = []
    for index, (name, sentence) in enumerate(parse_orthographic_transcript(transcript_path), start=1):
        stem = Path(name).stem
        source_audio_path = audio_by_stem.get(stem)
        if source_audio_path is None:
            raise FileNotFoundError(f"missing audio for transcript item {name}")
        rows.append(PreparedRow(path=wav_name(name, prefix="psc-"), sentence=sentence, source_audio_path=source_audio_path))
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
        parts = [node.text or "" for node in item.findall(".//x:t", namespace)]
        strings.append("".join(parts))
    return strings


def first_worksheet_path(archive: zipfile.ZipFile) -> str:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    namespace = {
        "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    sheet = workbook.find("x:sheets/x:sheet", namespace)
    if sheet is None:
        raise ValueError("xlsx workbook has no worksheets")
    relationship_id = sheet.attrib[f"{{{namespace['r']}}}id"]

    relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_namespace = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
    for rel in relationships.findall("rel:Relationship", rel_namespace):
        if rel.attrib.get("Id") == relationship_id:
            target = rel.attrib["Target"]
            return f"xl/{target}" if not target.startswith("/") else target.lstrip("/")
    raise ValueError("xlsx worksheet relationship was not found")


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


def cell_column_index(cell_ref: str) -> int:
    letters = "".join(char for char in cell_ref if char.isalpha()).upper()
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return max(0, index - 1)


def read_xlsx_rows(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = xlsx_shared_strings(archive)
        worksheet = ElementTree.fromstring(archive.read(first_worksheet_path(archive)))
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows: list[list[str]] = []
    for row in worksheet.findall("x:sheetData/x:row", namespace):
        values: list[str] = []
        for fallback_index, cell in enumerate(row.findall("x:c", namespace)):
            column_index = cell_column_index(cell.attrib.get("r", "")) if cell.attrib.get("r") else fallback_index
            while len(values) <= column_index:
                values.append("")
            values[column_index] = cell_value(cell, shared_strings).strip()
        if any(values):
            rows.append(values)
    return rows


def choose_column(headers: list[str], names: set[str], *, default: int) -> int:
    lowered = [header.strip().lower().replace(" ", "_") for header in headers]
    for index, header in enumerate(lowered):
        if header in names or any(token in header for token in names):
            return index
    return default


def build_persian_speech_rows(source_root: Path, metadata_path: Path) -> list[PreparedRow]:
    xlsx_rows = read_xlsx_rows(metadata_path)
    if not xlsx_rows:
        raise ValueError(f"empty metadata workbook: {metadata_path}")

    headers = xlsx_rows[0]
    body = xlsx_rows[1:]
    path_index = choose_column(headers, {"path", "file", "filename", "audio", "wav", "name"}, default=0)
    sentence_index = choose_column(headers, {"sentence", "text", "transcript", "transcription", "label"}, default=1)
    audio_by_stem = find_audio_by_stem(source_root)

    rows: list[PreparedRow] = []
    for index, values in enumerate(body, start=1):
        if max(path_index, sentence_index) >= len(values):
            continue
        raw_path = values[path_index]
        sentence = values[sentence_index]
        if not raw_path or not sentence:
            continue
        stem = Path(raw_path).stem
        source_audio_path = audio_by_stem.get(stem)
        if source_audio_path is None:
            raise FileNotFoundError(f"missing audio for PersianSpeech metadata item {raw_path}")
        rows.append(PreparedRow(path=wav_name(raw_path, prefix="ps-"), sentence=sentence, source_audio_path=source_audio_path))
    return rows


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
        raise RuntimeError("ffmpeg is required to convert evaluation clips to WAV")
    if workers < 1:
        raise ValueError("workers must be >= 1")

    unique_rows: dict[str, Path] = {}
    for row in rows:
        unique_rows.setdefault(row.path, row.source_audio_path)

    jobs: list[tuple[Path, Path]] = []
    for wav_path, source_path in unique_rows.items():
        output_path = output_root / "clips" / wav_path
        if output_path.exists():
            audit.wav_skipped_existing += 1
            continue
        if not source_path.exists():
            raise FileNotFoundError(f"missing source clip: {source_path}")
        jobs.append((source_path, output_path))

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


def write_dataset(output_root: Path, split_name: str, rows: list[PreparedRow], audit: DatasetAudit, *, workers: int = 1) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    write_split_tsv(output_root / f"{split_name}.tsv", rows)
    convert_required_clips(output_root, rows, audit, workers=workers)


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    split_name: str,
    row_builder: Callable[[Path], list[PreparedRow]],
    *,
    workers: int = 1,
) -> DatasetAudit:
    raw_rows = row_builder(source_root)
    audit = DatasetAudit(source_rows=len(raw_rows))
    rows = normalize_prepared_rows(raw_rows, audit)
    audit.final_rows = len(rows)
    write_dataset(output_root, split_name, rows, audit, workers=workers)
    return audit


def prepare_persian_speech(
    source_root: Path,
    metadata_path: Path,
    output_root: Path,
    split_name: str,
    *,
    workers: int = 1,
) -> DatasetAudit:
    return prepare_dataset(
        source_root,
        output_root,
        split_name,
        lambda root: build_persian_speech_rows(root, metadata_path),
        workers=workers,
    )


def print_dataset_audit(name: str, output_root: Path, audit: DatasetAudit) -> None:
    print(f"{name} summary")
    print(f"  output root: {output_root}")
    print(f"  source rows: {audit.source_rows}")
    print(f"  final rows: {audit.final_rows}")
    print(f"  normalized rows: {audit.normalized_rows}")
    print(f"  changed rows: {audit.changed_rows}")
    print(f"  discarded rows: {audit.discarded_rows}")
    print(f"  raw fallback rows: {audit.raw_fallback_rows}")
    print(f"  wav converted: {audit.wav_converted}")
    print(f"  wav skipped existing: {audit.wav_skipped_existing}")


def download_and_prepare_persian_eval_sets(
    *,
    cache_dir: Path = Path("data/downloads/persian_eval_sets"),
    persian_speech_corpus_output_root: Path = Path("data/persian-speech-corpus-test"),
    persian_speech_output_root: Path = Path("data/PersianSpeech_test"),
    split_name: str = "test",
    persian_speech_corpus_url: str = PERSIAN_SPEECH_CORPUS_URL,
    persian_speech_url: str = PERSIAN_SPEECH_URL,
    persian_speech_metadata_url: str = PERSIAN_SPEECH_METADATA_URL,
    force: bool = False,
    workers: int = 1,
    show_progress: bool = True,
) -> Audit:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if persian_speech_corpus_output_root.exists() and not force:
        raise FileExistsError(f"{persian_speech_corpus_output_root} already exists; pass --force")
    if persian_speech_output_root.exists() and not force:
        raise FileExistsError(f"{persian_speech_output_root} already exists; pass --force")
    if force:
        shutil.rmtree(persian_speech_corpus_output_root, ignore_errors=True)
        shutil.rmtree(persian_speech_output_root, ignore_errors=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    psc_archive = cache_dir / "persian-speech-corpus.zip"
    ps_archive = cache_dir / "myaudio_tiny.tar.gz"
    ps_metadata = cache_dir / "myaudio_tiny.xlsx"

    download_public_url(persian_speech_corpus_url, psc_archive, force=force, show_progress=show_progress)
    download_google_drive_file(persian_speech_url, ps_archive, force=force, show_progress=show_progress)
    download_public_url(persian_speech_metadata_url, ps_metadata, force=force, show_progress=show_progress)

    with tempfile.TemporaryDirectory(prefix="persian-eval-sets-") as temp_name:
        temp_root = Path(temp_name)
        psc_source = temp_root / "persian-speech-corpus"
        ps_source = temp_root / "PersianSpeech"
        extract_archive(psc_archive, psc_source)
        extract_archive(ps_archive, ps_source)

        psc_audit = prepare_dataset(
            psc_source,
            persian_speech_corpus_output_root,
            split_name,
            build_persian_speech_corpus_rows,
            workers=workers,
        )
        ps_audit = prepare_persian_speech(
            ps_source,
            ps_metadata,
            persian_speech_output_root,
            split_name,
            workers=workers,
        )
    return Audit(persian_speech_corpus=psc_audit, persian_speech=ps_audit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download Persian Speech Corpus and PersianSpeech tiny, normalize transcripts, "
            "and export dataset directories with a split TSV plus mono 16 kHz WAV clips."
        )
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/downloads/persian_eval_sets"), help="Directory for downloaded archives and metadata.")
    parser.add_argument(
        "--persian-speech-corpus-output-root",
        type=Path,
        default=Path("data/persian-speech-corpus-test"),
        help="Output directory for Nawar Halabi's Persian Speech Corpus test dataset.",
    )
    parser.add_argument(
        "--persian-speech-output-root",
        type=Path,
        default=Path("data/PersianSpeech_test"),
        help="Output directory for the persiandataset/PersianSpeech tiny test dataset.",
    )
    parser.add_argument("--split-name", default="test", help="Output split TSV basename. Defaults to test.")
    parser.add_argument("--persian-speech-corpus-url", default=PERSIAN_SPEECH_CORPUS_URL, help="Download URL for persian-speech-corpus.zip.")
    parser.add_argument("--persian-speech-url", default=PERSIAN_SPEECH_URL, help="Download URL for PersianSpeech myaudio_tiny archive.")
    parser.add_argument("--persian-speech-metadata-url", default=PERSIAN_SPEECH_METADATA_URL, help="Download URL for PersianSpeech XLSX metadata.")
    parser.add_argument("--force", action="store_true", help="Replace existing outputs and re-download archives.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of parallel ffmpeg conversion worker processes. Use 1 for single-process conversion.",
    )
    args = parser.parse_args(argv)

    audit = download_and_prepare_persian_eval_sets(
        cache_dir=args.cache_dir,
        persian_speech_corpus_output_root=args.persian_speech_corpus_output_root,
        persian_speech_output_root=args.persian_speech_output_root,
        split_name=args.split_name,
        persian_speech_corpus_url=args.persian_speech_corpus_url,
        persian_speech_url=args.persian_speech_url,
        persian_speech_metadata_url=args.persian_speech_metadata_url,
        force=args.force,
        workers=args.workers,
    )
    print_dataset_audit("Persian Speech Corpus", args.persian_speech_corpus_output_root, audit.persian_speech_corpus)
    print_dataset_audit("PersianSpeech", args.persian_speech_output_root, audit.persian_speech)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
