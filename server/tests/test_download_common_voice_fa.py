from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from ml.speech_data.scripts.download_common_voice_fa import (
    create_download_session,
    download_url_to_file,
    expected_sha256,
    load_api_key,
    parse_dotenv,
    verify_checksum,
)


class FakeResponse(io.BytesIO):
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_parse_dotenv_reads_quoted_values(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# ignored",
                "MOZILLA_DATA_COLLECTIVE_API_KEY='secret-token'",
                'APP_NAME="MS Thesis"',
                "EMPTY=",
            ]
        ),
        encoding="utf-8",
    )

    assert parse_dotenv(env_path) == {
        "MOZILLA_DATA_COLLECTIVE_API_KEY": "secret-token",
        "APP_NAME": "MS Thesis",
        "EMPTY": "",
    }


def test_load_api_key_uses_dotenv_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MOZILLA_DATA_COLLECTIVE_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("MOZILLA_DATA_COLLECTIVE_API_KEY=from-env-file\n", encoding="utf-8")

    assert load_api_key(env_path) == "from-env-file"


def test_load_api_key_prefers_process_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOZILLA_DATA_COLLECTIVE_API_KEY", "from-process-env")
    env_path = tmp_path / ".env"
    env_path.write_text("MOZILLA_DATA_COLLECTIVE_API_KEY=from-env-file\n", encoding="utf-8")

    assert load_api_key(env_path) == "from-process-env"


def test_create_download_session_posts_to_dataset_download_endpoint() -> None:
    requests: list[Any] = []

    def fake_opener(request: Any) -> FakeResponse:
        requests.append(request)
        payload = {
            "downloadUrl": "https://storage.example/archive.tar.gz",
            "filename": "common-voice-fa.tar.gz",
            "sizeBytes": "12",
            "checksum": "sha256:abc",
            "expiresAt": "2026-01-01T00:00:00Z",
        }
        return FakeResponse(json.dumps(payload).encode("utf-8"))

    session = create_download_session(
        api_key="api-token",
        dataset_id="dataset-1",
        api_base_url="https://mozilladatacollective.com/api",
        opener=fake_opener,
    )

    request = requests[0]
    assert request.full_url == "https://mozilladatacollective.com/api/datasets/dataset-1/download"
    assert request.get_method() == "POST"
    assert request.headers["Authorization"] == "Bearer api-token"
    assert session.download_url == "https://storage.example/archive.tar.gz"
    assert session.filename == "common-voice-fa.tar.gz"
    assert session.size_bytes == 12
    assert session.checksum == "sha256:abc"


def test_download_url_to_file_resumes_existing_archive(tmp_path: Path) -> None:
    requests: list[Any] = []
    output_path = tmp_path / "archive.tar.gz"
    output_path.write_bytes(b"abc")

    def fake_opener(request: Any) -> FakeResponse:
        requests.append(request)
        return FakeResponse(b"def")

    bytes_written = download_url_to_file(
        "https://storage.example/archive.tar.gz",
        output_path,
        expected_size=6,
        opener=fake_opener,
        show_progress=False,
    )

    assert bytes_written == 6
    assert output_path.read_bytes() == b"abcdef"
    assert requests[0].headers["Range"] == "bytes=3-"


def test_download_url_to_file_rejects_existing_archive_without_resume_or_force(tmp_path: Path) -> None:
    output_path = tmp_path / "archive.tar.gz"
    output_path.write_bytes(b"abc")

    with pytest.raises(FileExistsError, match="--force or --resume"):
        download_url_to_file("https://storage.example/archive.tar.gz", output_path, resume=False, show_progress=False)


def test_verify_checksum_accepts_sha256_prefix(tmp_path: Path) -> None:
    output_path = tmp_path / "archive.tar.gz"
    output_path.write_bytes(b"archive")
    checksum = hashlib.sha256(b"archive").hexdigest()

    assert expected_sha256(f"sha256:{checksum}") == checksum
    assert verify_checksum(output_path, f"sha256:{checksum}") is True


def test_verify_checksum_raises_on_mismatch(tmp_path: Path) -> None:
    output_path = tmp_path / "archive.tar.gz"
    output_path.write_bytes(b"archive")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_checksum(output_path, "sha256:" + ("0" * 64))
