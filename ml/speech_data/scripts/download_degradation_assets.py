from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ml.speech_data.scripts.download_common_voice_fa import api_error_message, download_url_to_file
from ml.speech_data.scripts.prepare_degradation_assets import prepare_degradation_assets


DEFAULT_ZENODO_RECORD_API = "https://zenodo.org/api/records/1227121"


@dataclass(frozen=True)
class FileSpec:
    name: str
    url: str
    size_bytes: int | None = None
    checksum: str | None = None


@dataclass(frozen=True)
class DownloadedFile:
    name: str
    path: Path
    bytes_written: int
    checksum_verified: bool = False


@dataclass
class AssetDownloadAudit:
    demand_archives: list[DownloadedFile] = field(default_factory=list)
    prepared_indexes: bool = False


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_md5(checksum: str | None) -> str | None:
    if not checksum:
        return None
    if checksum.startswith("md5:"):
        return checksum.split(":", 1)[1].lower()
    return checksum.lower() if len(checksum) == 32 else None


def verify_md5(path: Path, checksum: str | None) -> bool:
    expected = expected_md5(checksum)
    if expected is None:
        return False
    actual = md5_file(path)
    if actual != expected:
        raise RuntimeError(f"checksum mismatch for {path}: expected {expected}, got {actual}")
    return True


def load_zenodo_record(
    api_url: str,
    *,
    opener: Callable[[urllib.request.Request], Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    request = urllib.request.Request(api_url, headers={"Accept": "application/json"}, method="GET")
    try:
        with opener(request) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        message = api_error_message(error)
        raise RuntimeError(f"Zenodo API returned {error.code}: {message}") from error

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError("Zenodo API returned invalid JSON") from error
    if not isinstance(parsed, dict):
        raise RuntimeError("Zenodo API returned a non-object JSON payload")
    return parsed


def zenodo_download_url(record_api_url: str, name: str, links: dict[str, Any]) -> str:
    direct = links.get("self") or links.get("content")
    if isinstance(direct, str) and direct:
        return direct

    parsed = urllib.parse.urlparse(record_api_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    record_id = Path(parsed.path).name
    quoted_name = urllib.parse.quote(name)
    return f"{base}/records/{record_id}/files/{quoted_name}?download=1"


def demand_16k_files(record: dict[str, Any], *, record_api_url: str) -> list[FileSpec]:
    raw_files = record.get("files")
    if not isinstance(raw_files, list):
        raise RuntimeError("Zenodo record did not include a files list")

    specs: list[FileSpec] = []
    for item in raw_files:
        if not isinstance(item, dict):
            continue
        name = item.get("key")
        if not isinstance(name, str) or not name.endswith("_16k.zip"):
            continue

        links = item.get("links") if isinstance(item.get("links"), dict) else {}
        raw_size = item.get("size")
        specs.append(
            FileSpec(
                name=name,
                url=zenodo_download_url(record_api_url, name, links),
                size_bytes=int(raw_size) if raw_size not in (None, "") else None,
                checksum=item.get("checksum") if isinstance(item.get("checksum"), str) else None,
            )
        )
    if not specs:
        raise RuntimeError("Zenodo record did not contain any *_16k.zip DEMAND files")
    return sorted(specs, key=lambda spec: spec.name)


def download_file(
    spec: FileSpec,
    output_path: Path,
    *,
    force: bool,
    resume: bool,
    show_progress: bool,
    opener: Callable[[urllib.request.Request], Any] = urllib.request.urlopen,
) -> DownloadedFile:
    if output_path.exists() and not force:
        if verify_md5(output_path, spec.checksum):
            return DownloadedFile(spec.name, output_path, output_path.stat().st_size, checksum_verified=True)
        if spec.size_bytes is not None and output_path.stat().st_size == spec.size_bytes and spec.checksum is None:
            return DownloadedFile(spec.name, output_path, output_path.stat().st_size, checksum_verified=False)

    bytes_written = download_url_to_file(
        spec.url,
        output_path,
        expected_size=spec.size_bytes,
        force=force,
        resume=resume,
        opener=opener,
        show_progress=show_progress,
    )
    checksum_verified = verify_md5(output_path, spec.checksum)
    return DownloadedFile(spec.name, output_path, bytes_written, checksum_verified=checksum_verified)


def download_degradation_assets(
    *,
    noise_root: Path = Path("data/speech_enhancement/assets/noise/DEMAND"),
    manifest_dir: Path = Path("data/speech_enhancement/manifests"),
    zenodo_record_api: str = DEFAULT_ZENODO_RECORD_API,
    include_noise: bool = True,
    force: bool = False,
    resume: bool = True,
    prepare: bool = False,
    show_progress: bool = True,
    opener: Callable[[urllib.request.Request], Any] = urllib.request.urlopen,
) -> AssetDownloadAudit:
    audit = AssetDownloadAudit()

    if include_noise:
        record = load_zenodo_record(zenodo_record_api, opener=opener)
        for spec in demand_16k_files(record, record_api_url=zenodo_record_api):
            audit.demand_archives.append(
                download_file(
                    spec,
                    noise_root / spec.name,
                    force=force,
                    resume=resume,
                    show_progress=show_progress,
                    opener=opener,
                )
            )

    if prepare:
        prepare_degradation_assets(
            noise_root,
            manifest_dir,
            extract=True,
            show_progress=show_progress,
        )
        audit.prepared_indexes = True

    return audit


def print_audit(audit: AssetDownloadAudit) -> None:
    print("Degradation asset download summary")
    print(f"  DEMAND 16 kHz archives: {len(audit.demand_archives)}")
    for item in audit.demand_archives:
        verified = "verified" if item.checksum_verified else "not verified"
        print(f"    {item.path.name}: {item.bytes_written} bytes, checksum {verified}")
    print(f"  prepared indexes: {audit.prepared_indexes}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download DEMAND 16 kHz noise archives for the speech degradation pipeline."
        )
    )
    parser.add_argument(
        "--noise-root",
        type=Path,
        default=Path("data/speech_enhancement/assets/noise/DEMAND"),
        help="Output directory for DEMAND *_16k.zip archives.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/speech_enhancement/manifests"),
        help="Manifest directory used when --prepare-indexes is enabled.",
    )
    parser.add_argument(
        "--zenodo-record-api",
        default=DEFAULT_ZENODO_RECORD_API,
        help="Zenodo record API URL for DEMAND.",
    )
    parser.add_argument("--skip-noise", action="store_true", help="Do not download DEMAND.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing archives.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume partially downloaded archives when possible.",
    )
    parser.add_argument(
        "--prepare-indexes",
        action="store_true",
        help="After download, extract archives and write the DEMAND noise JSONL index.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars.")
    args = parser.parse_args(argv)

    audit = download_degradation_assets(
        noise_root=args.noise_root,
        manifest_dir=args.manifest_dir,
        zenodo_record_api=args.zenodo_record_api,
        include_noise=not args.skip_noise,
        force=args.force,
        resume=args.resume,
        prepare=args.prepare_indexes,
        show_progress=not args.quiet,
    )
    print_audit(audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
