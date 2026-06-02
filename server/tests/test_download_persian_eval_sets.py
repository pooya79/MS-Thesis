from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from ml.speech_data.scripts import download_persian_eval_sets as script


def zip_bytes(filename: str = "file.txt") -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(filename, "content")
    return payload.getvalue()


def tar_gz_bytes(filename: str = "clip.wav") -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        content = b"audio"
        info = tarfile.TarInfo(filename)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    return payload.getvalue()


class FakeResponse:
    def __init__(self, payload: bytes, url: str = "https://drive.google.com/uc?export=download&id=file") -> None:
        self.payload = io.BytesIO(payload)
        self.url = url

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.payload.read(size)

    def geturl(self) -> str:
        return self.url


class FakeOpener:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    def open(self, url: str) -> FakeResponse:
        self.urls.append(url)
        return self.responses.pop(0)


def test_download_public_file_reuses_valid_cached_file(tmp_path: Path) -> None:
    output_path = tmp_path / "persian-speech-corpus.zip"
    output_path.write_bytes(zip_bytes())

    downloaded = script.download_validated_public_file(
        url="https://example.test/archive.zip",
        output_path=output_path,
        validator=script.is_valid_zip,
        show_progress=False,
    )

    assert downloaded.reused_existing is True
    assert downloaded.bytes_written == output_path.stat().st_size


def test_download_public_file_replaces_invalid_cached_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "myaudio_tiny.xlsx"
    output_path.write_text("<html>bad cache</html>", encoding="utf-8")

    def fake_download_url_to_file(
        url: str,
        output_path: Path,
        *,
        force: bool = False,
        resume: bool = False,
        show_progress: bool = True,
    ) -> int:
        output_path.write_bytes(zip_bytes("[Content_Types].xml"))
        return output_path.stat().st_size

    monkeypatch.setattr(script, "download_url_to_file", fake_download_url_to_file)

    downloaded = script.download_validated_public_file(
        url="https://example.test/myaudio_tiny.xlsx",
        output_path=output_path,
        validator=script.is_valid_xlsx,
        show_progress=False,
    )

    assert downloaded.reused_existing is False
    assert zipfile.is_zipfile(output_path)


def test_download_google_drive_file_follows_confirmation_form(tmp_path: Path) -> None:
    output_path = tmp_path / "myaudio_tiny.tar.gz"
    html = (
        '<html><form action="https://drive.usercontent.google.com/download" id="download-form">'
        '<input type="hidden" name="id" value="file">'
        '<input type="hidden" name="export" value="download">'
        '<input type="hidden" name="confirm" value="token">'
        "</form></html>"
    ).encode()
    opener = FakeOpener([FakeResponse(html), FakeResponse(tar_gz_bytes(), url="https://drive.usercontent.google.com/download")])

    downloaded = script.download_validated_google_drive_file(
        url="https://drive.google.com/uc?export=download&id=file",
        output_path=output_path,
        validator=script.is_valid_tar,
        opener=opener,
        show_progress=False,
    )

    assert downloaded.reused_existing is False
    assert tarfile.is_tarfile(output_path)
    assert opener.urls[1] == "https://drive.usercontent.google.com/download?id=file&export=download&confirm=token"


def test_download_google_drive_file_rejects_html_response(tmp_path: Path) -> None:
    output_path = tmp_path / "myaudio_tiny.tar.gz"
    opener = FakeOpener([FakeResponse(b"<html>access denied</html>")])

    with pytest.raises(RuntimeError, match="Google Drive download did not produce the expected file type"):
        script.download_validated_google_drive_file(
            url="https://drive.google.com/uc?export=download&id=file",
            output_path=output_path,
            validator=script.is_valid_tar,
            opener=opener,
            show_progress=False,
        )

    assert not output_path.exists()


def test_download_persian_eval_sets_downloads_three_cached_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    public_payloads = {
        "https://example.test/persian-speech-corpus.zip": zip_bytes("orthographic-transcript.txt"),
        "https://example.test/myaudio_tiny.xlsx": zip_bytes("[Content_Types].xml"),
    }

    def fake_public_download(
        *,
        url: str,
        output_path: Path,
        validator,
        force: bool = False,
        show_progress: bool = True,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(public_payloads[url])
        return script.DownloadedFile(output_path.name, output_path, output_path.stat().st_size, False)

    def fake_drive_download(
        *,
        url: str,
        output_path: Path,
        validator,
        force: bool = False,
        show_progress: bool = True,
        opener=None,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(tar_gz_bytes())
        return script.DownloadedFile(output_path.name, output_path, output_path.stat().st_size, False)

    monkeypatch.setattr(script, "download_validated_public_file", fake_public_download)
    monkeypatch.setattr(script, "download_validated_google_drive_file", fake_drive_download)

    audit = script.download_persian_eval_sets(
        cache_dir=tmp_path / "cache",
        persian_speech_corpus_url="https://example.test/persian-speech-corpus.zip",
        persian_speech_url="https://example.test/myaudio_tiny.tar.gz",
        persian_speech_metadata_url="https://example.test/myaudio_tiny.xlsx",
        show_progress=False,
    )

    assert audit.persian_speech_corpus_archive.path.name == "persian-speech-corpus.zip"
    assert audit.persian_speech_archive.path.name == "myaudio_tiny.tar.gz"
    assert audit.persian_speech_metadata.path.name == "myaudio_tiny.xlsx"
