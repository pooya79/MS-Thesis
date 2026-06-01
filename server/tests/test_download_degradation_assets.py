from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from ml.speech_data.scripts.download_degradation_assets import (
    demand_16k_files,
    download_degradation_assets,
    expected_md5,
    verify_md5,
)


class FakeResponse(io.BytesIO):
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def md5(value: bytes) -> str:
    return hashlib.md5(value, usedforsecurity=False).hexdigest()


def test_demand_16k_files_selects_only_16k_zip_downloads() -> None:
    record = {
        "files": [
            {
                "key": "DKITCHEN_16k.zip",
                "size": 3,
                "checksum": "md5:abc",
                "links": {"self": "https://zenodo.example/api/files/dkitchen"},
            },
            {
                "key": "DKITCHEN_48k.zip",
                "size": 4,
                "checksum": "md5:def",
                "links": {"self": "https://zenodo.example/api/files/dkitchen48"},
            },
            {"key": "DEMAND.pdf", "links": {"self": "https://zenodo.example/api/files/pdf"}},
        ]
    }

    specs = demand_16k_files(record, record_api_url="https://zenodo.example/api/records/1227121")

    assert [spec.name for spec in specs] == ["DKITCHEN_16k.zip"]
    assert specs[0].url == "https://zenodo.example/api/files/dkitchen"
    assert specs[0].size_bytes == 3
    assert specs[0].checksum == "md5:abc"


def test_verify_md5_accepts_md5_prefix(tmp_path: Path) -> None:
    path = tmp_path / "archive.zip"
    path.write_bytes(b"archive")
    checksum = md5(b"archive")

    assert expected_md5(f"md5:{checksum}") == checksum
    assert verify_md5(path, f"md5:{checksum}") is True


def test_verify_md5_raises_on_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "archive.zip"
    path.write_bytes(b"archive")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_md5(path, "md5:" + ("0" * 32))


def test_download_degradation_assets_downloads_demand(tmp_path: Path) -> None:
    kitchen_bytes = b"kitchen"
    living_bytes = b"living"
    requests: list[Any] = []

    record = {
        "files": [
            {
                "key": "DLIVING_16k.zip",
                "size": len(living_bytes),
                "checksum": f"md5:{md5(living_bytes)}",
                "links": {"self": "https://zenodo.example/files/living"},
            },
            {
                "key": "DKITCHEN_16k.zip",
                "size": len(kitchen_bytes),
                "checksum": f"md5:{md5(kitchen_bytes)}",
                "links": {"self": "https://zenodo.example/files/kitchen"},
            },
            {
                "key": "DKITCHEN_48k.zip",
                "size": 3,
                "checksum": f"md5:{md5(b'48k')}",
                "links": {"self": "https://zenodo.example/files/kitchen48"},
            },
        ]
    }

    def fake_opener(request: Any) -> FakeResponse:
        requests.append(request)
        if request.full_url == "https://zenodo.example/api/records/1227121":
            return FakeResponse(json.dumps(record).encode("utf-8"))
        if request.full_url == "https://zenodo.example/files/kitchen":
            return FakeResponse(kitchen_bytes)
        if request.full_url == "https://zenodo.example/files/living":
            return FakeResponse(living_bytes)
        raise AssertionError(f"unexpected URL: {request.full_url}")

    audit = download_degradation_assets(
        noise_root=tmp_path / "noise",
        zenodo_record_api="https://zenodo.example/api/records/1227121",
        show_progress=False,
        opener=fake_opener,
    )

    assert [item.name for item in audit.demand_archives] == ["DKITCHEN_16k.zip", "DLIVING_16k.zip"]
    assert all(item.checksum_verified for item in audit.demand_archives)
    assert (tmp_path / "noise" / "DKITCHEN_16k.zip").read_bytes() == kitchen_bytes
    assert (tmp_path / "noise" / "DLIVING_16k.zip").read_bytes() == living_bytes
    assert [request.full_url for request in requests] == [
        "https://zenodo.example/api/records/1227121",
        "https://zenodo.example/files/kitchen",
        "https://zenodo.example/files/living",
    ]
