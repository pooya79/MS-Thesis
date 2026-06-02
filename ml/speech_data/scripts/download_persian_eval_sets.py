from __future__ import annotations

import argparse
import html
import re
import shutil
import tarfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from ml.speech_data.scripts.download_common_voice_fa import download_url_to_file


PERSIAN_SPEECH_CORPUS_URL = "https://en.persianspeechcorpus.com/persian-speech-corpus.zip"
PERSIAN_SPEECH_DRIVE_FILE_ID = "1cCWH_eoa4Nq17XDHn6e1WIfHomdGWPKO"
PERSIAN_SPEECH_URL = f"https://drive.google.com/uc?export=download&id={PERSIAN_SPEECH_DRIVE_FILE_ID}"
PERSIAN_SPEECH_METADATA_URL = "https://raw.githubusercontent.com/persiandataset/PersianSpeech/main/myaudio_tiny.xlsx"


@dataclass(frozen=True)
class DownloadedFile:
    name: str
    path: Path
    bytes_written: int
    reused_existing: bool


@dataclass(frozen=True)
class DownloadAudit:
    persian_speech_corpus_archive: DownloadedFile
    persian_speech_archive: DownloadedFile
    persian_speech_metadata: DownloadedFile


def is_valid_zip(path: Path) -> bool:
    return path.exists() and zipfile.is_zipfile(path)


def is_valid_tar(path: Path) -> bool:
    return path.exists() and tarfile.is_tarfile(path)


def is_valid_xlsx(path: Path) -> bool:
    return is_valid_zip(path)


def first_bytes(path: Path, limit: int = 512) -> bytes:
    with path.open("rb") as handle:
        return handle.read(limit)


def invalid_download_hint(path: Path) -> str:
    prefix = first_bytes(path).lstrip().lower()
    if prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html"):
        return " The response was HTML, usually a download confirmation or access page."
    return ""


def write_response_to_file(response: Any, output_path: Path, *, show_progress: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        with tqdm.wrapattr(
            response,
            "read",
            desc=f"Downloading {output_path.name}",
            unit="B",
            unit_scale=True,
            disable=not show_progress,
        ) as reader:
            shutil.copyfileobj(reader, handle)


def google_drive_confirm_url(final_url: str, payload: bytes) -> str | None:
    if "drive.google.com" not in final_url and "drive.usercontent.google.com" not in final_url:
        return None

    token_match = re.search(rb"confirm=([0-9A-Za-z_%-]+)", payload)
    if token_match:
        parsed = urllib.parse.urlparse(final_url)
        query = urllib.parse.parse_qs(parsed.query)
        query["confirm"] = [token_match.group(1).decode("ascii")]
        return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query, doseq=True)))

    form_match = re.search(rb"<form(?=[^>]+id=\"download-form\")[^>]+action=\"([^\"]+)\"", payload)
    if form_match is None:
        return None

    action = html.unescape(form_match.group(1).decode("utf-8"))
    inputs = {
        html.unescape(name.decode("utf-8")): html.unescape(value.decode("utf-8"))
        for name, value in re.findall(rb"<input[^>]+name=\"([^\"]+)\"[^>]+value=\"([^\"]*)\"", payload)
    }
    parsed = urllib.parse.urlparse(action)
    query = urllib.parse.parse_qs(parsed.query)
    for key, value in inputs.items():
        query[key] = [value]
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query, doseq=True)))


def download_validated_public_file(
    *,
    url: str,
    output_path: Path,
    validator: Callable[[Path], bool],
    force: bool = False,
    show_progress: bool = True,
) -> DownloadedFile:
    if output_path.exists() and not force and validator(output_path):
        return DownloadedFile(name=output_path.name, path=output_path, bytes_written=output_path.stat().st_size, reused_existing=True)
    if output_path.exists():
        output_path.unlink()

    bytes_written = download_url_to_file(url, output_path, force=True, resume=False, show_progress=show_progress)
    if not validator(output_path):
        hint = invalid_download_hint(output_path)
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"download did not produce the expected file type for {output_path}.{hint}")
    return DownloadedFile(name=output_path.name, path=output_path, bytes_written=bytes_written, reused_existing=False)


def download_validated_google_drive_file(
    *,
    url: str,
    output_path: Path,
    validator: Callable[[Path], bool],
    force: bool = False,
    show_progress: bool = True,
    opener: Any | None = None,
) -> DownloadedFile:
    if output_path.exists() and not force and validator(output_path):
        return DownloadedFile(name=output_path.name, path=output_path, bytes_written=output_path.stat().st_size, reused_existing=True)
    if output_path.exists():
        output_path.unlink()

    opener = opener or urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    with opener.open(url) as response:
        payload = response.read()
        final_url = response.geturl()

    confirm_url = google_drive_confirm_url(final_url, payload)
    if confirm_url is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)
    else:
        with opener.open(confirm_url) as response:
            write_response_to_file(response, output_path, show_progress=show_progress)

    if not validator(output_path):
        hint = invalid_download_hint(output_path)
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Google Drive download did not produce the expected file type for {output_path}.{hint}")

    return DownloadedFile(name=output_path.name, path=output_path, bytes_written=output_path.stat().st_size, reused_existing=False)


def download_persian_eval_sets(
    *,
    cache_dir: Path = Path("data/downloads/persian_eval_sets"),
    persian_speech_corpus_url: str = PERSIAN_SPEECH_CORPUS_URL,
    persian_speech_url: str = PERSIAN_SPEECH_URL,
    persian_speech_metadata_url: str = PERSIAN_SPEECH_METADATA_URL,
    force: bool = False,
    show_progress: bool = True,
) -> DownloadAudit:
    cache_dir.mkdir(parents=True, exist_ok=True)
    persian_speech_corpus_archive = download_validated_public_file(
        url=persian_speech_corpus_url,
        output_path=cache_dir / "persian-speech-corpus.zip",
        validator=is_valid_zip,
        force=force,
        show_progress=show_progress,
    )
    persian_speech_archive = download_validated_google_drive_file(
        url=persian_speech_url,
        output_path=cache_dir / "myaudio_tiny.tar.gz",
        validator=is_valid_tar,
        force=force,
        show_progress=show_progress,
    )
    persian_speech_metadata = download_validated_public_file(
        url=persian_speech_metadata_url,
        output_path=cache_dir / "myaudio_tiny.xlsx",
        validator=is_valid_xlsx,
        force=force,
        show_progress=show_progress,
    )
    return DownloadAudit(
        persian_speech_corpus_archive=persian_speech_corpus_archive,
        persian_speech_archive=persian_speech_archive,
        persian_speech_metadata=persian_speech_metadata,
    )


def print_downloaded_file(file: DownloadedFile) -> None:
    status = "reused" if file.reused_existing else "downloaded"
    print(f"  {file.name}: {file.path} ({file.bytes_written} bytes, {status})")


def print_audit(audit: DownloadAudit) -> None:
    print("Persian evaluation set download summary")
    print_downloaded_file(audit.persian_speech_corpus_archive)
    print_downloaded_file(audit.persian_speech_archive)
    print_downloaded_file(audit.persian_speech_metadata)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download only the upstream Persian Speech Corpus archive, PersianSpeech "
            "myaudio_tiny archive, and PersianSpeech XLSX metadata into a local cache."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/downloads/persian_eval_sets"),
        help="Directory where downloaded archives and metadata will be stored.",
    )
    parser.add_argument(
        "--persian-speech-corpus-url",
        default=PERSIAN_SPEECH_CORPUS_URL,
        help="Download URL for Nawar Halabi's persian-speech-corpus.zip.",
    )
    parser.add_argument(
        "--persian-speech-url",
        default=PERSIAN_SPEECH_URL,
        help="Download URL for the PersianSpeech myaudio_tiny archive.",
    )
    parser.add_argument(
        "--persian-speech-metadata-url",
        default=PERSIAN_SPEECH_METADATA_URL,
        help="Download URL for the PersianSpeech myaudio_tiny XLSX metadata.",
    )
    parser.add_argument("--force", action="store_true", help="Redownload files even when valid cached files already exist.")
    args = parser.parse_args(argv)

    audit = download_persian_eval_sets(
        cache_dir=args.cache_dir,
        persian_speech_corpus_url=args.persian_speech_corpus_url,
        persian_speech_url=args.persian_speech_url,
        persian_speech_metadata_url=args.persian_speech_metadata_url,
        force=args.force,
    )
    print_audit(audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
