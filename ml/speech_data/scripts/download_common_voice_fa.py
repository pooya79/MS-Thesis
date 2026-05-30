from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm


API_BASE_URL = "https://mozilladatacollective.com/api"
DATASET_ID = "cmn2gho8i01gio107ckfuqzxo"
DEFAULT_API_KEY_ENV = "MOZILLA_DATA_COLLECTIVE_API_KEY"
FALLBACK_API_KEY_ENVS = ("MOZILLA_API_KEY", "COMMON_VOICE_API_KEY")


@dataclass(frozen=True)
class DownloadSession:
    download_url: str
    filename: str
    size_bytes: int | None
    checksum: str | None
    expires_at: str | None


@dataclass(frozen=True)
class DownloadAudit:
    dataset_id: str
    output_path: Path
    bytes_written: int
    checksum_verified: bool
    expires_at: str | None


def parse_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            values[key] = value
    return values


def load_api_key(env_path: Path, *, env_name: str = DEFAULT_API_KEY_ENV) -> str:
    env_values = parse_dotenv(env_path)
    candidate_names = (env_name, *FALLBACK_API_KEY_ENVS)
    for name in candidate_names:
        value = os.environ.get(name) or env_values.get(name)
        if value:
            return value

    names = ", ".join(dict.fromkeys(candidate_names))
    raise RuntimeError(f"missing Mozilla Data Collective API key; set one of: {names}")


def api_error_message(error: urllib.error.HTTPError) -> str:
    try:
        body = error.read().decode("utf-8")
    except Exception:
        body = ""

    if body:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return body
        message = parsed.get("error")
        if isinstance(message, str):
            return message
    return error.reason


def post_json(url: str, *, api_key: str, opener: Callable[[urllib.request.Request], Any] = urllib.request.urlopen) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=b"{}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with opener(request) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        message = api_error_message(error)
        raise RuntimeError(f"Mozilla Data Collective API returned {error.code}: {message}") from error

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError("Mozilla Data Collective API returned invalid JSON") from error
    if not isinstance(parsed, dict):
        raise RuntimeError("Mozilla Data Collective API returned a non-object JSON payload")
    return parsed


def create_download_session(
    *,
    api_key: str,
    dataset_id: str = DATASET_ID,
    api_base_url: str = API_BASE_URL,
    opener: Callable[[urllib.request.Request], Any] = urllib.request.urlopen,
) -> DownloadSession:
    payload = post_json(f"{api_base_url.rstrip('/')}/datasets/{dataset_id}/download", api_key=api_key, opener=opener)
    download_url = payload.get("downloadUrl")
    filename = payload.get("filename")
    if not isinstance(download_url, str) or not download_url:
        raise RuntimeError("download response did not include downloadUrl")
    if not isinstance(filename, str) or not filename:
        filename = f"common-voice-fa-{dataset_id}.tar.gz"

    raw_size = payload.get("sizeBytes")
    size_bytes = int(raw_size) if raw_size not in (None, "") else None
    checksum = payload.get("checksum")
    expires_at = payload.get("expiresAt")
    return DownloadSession(
        download_url=download_url,
        filename=filename,
        size_bytes=size_bytes,
        checksum=checksum if isinstance(checksum, str) else None,
        expires_at=expires_at if isinstance(expires_at, str) else None,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_sha256(checksum: str | None) -> str | None:
    if not checksum:
        return None
    if checksum.startswith("sha256:"):
        return checksum.split(":", 1)[1].lower()
    return checksum.lower() if len(checksum) == 64 else None


def download_url_to_file(
    url: str,
    output_path: Path,
    *,
    expected_size: int | None = None,
    force: bool = False,
    resume: bool = True,
    opener: Callable[[urllib.request.Request], Any] = urllib.request.urlopen,
    show_progress: bool = True,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force and not resume:
        raise FileExistsError(f"{output_path} already exists; pass --force or --resume")

    existing_size = output_path.stat().st_size if output_path.exists() and resume and not force else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
    request = urllib.request.Request(url, headers=headers, method="GET")
    mode = "ab" if existing_size else "wb"

    try:
        with opener(request) as response, output_path.open(mode) as handle:
            with tqdm.wrapattr(
                response,
                "read",
                total=expected_size,
                initial=existing_size,
                desc=f"Downloading {output_path.name}",
                unit="B",
                unit_scale=True,
                disable=not show_progress,
            ) as reader:
                shutil.copyfileobj(reader, handle)
    except urllib.error.HTTPError as error:
        message = api_error_message(error)
        raise RuntimeError(f"download URL returned {error.code}: {message}") from error

    return output_path.stat().st_size


def verify_checksum(path: Path, checksum: str | None) -> bool:
    expected = expected_sha256(checksum)
    if expected is None:
        return False
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"checksum mismatch for {path}: expected {expected}, got {actual}")
    return True


def download_common_voice_fa(
    *,
    output_dir: Path = Path("data"),
    env_path: Path = Path(".env"),
    api_key_env: str = DEFAULT_API_KEY_ENV,
    dataset_id: str = DATASET_ID,
    api_base_url: str = API_BASE_URL,
    output_name: str | None = None,
    force: bool = False,
    resume: bool = True,
    show_progress: bool = True,
) -> DownloadAudit:
    api_key = load_api_key(env_path, env_name=api_key_env)
    session = create_download_session(api_key=api_key, dataset_id=dataset_id, api_base_url=api_base_url)
    output_path = output_dir / (output_name or session.filename)
    bytes_written = download_url_to_file(
        session.download_url,
        output_path,
        expected_size=session.size_bytes,
        force=force,
        resume=resume,
        show_progress=show_progress,
    )
    checksum_verified = verify_checksum(output_path, session.checksum)
    return DownloadAudit(
        dataset_id=dataset_id,
        output_path=output_path,
        bytes_written=bytes_written,
        checksum_verified=checksum_verified,
        expires_at=session.expires_at,
    )


def print_audit(audit: DownloadAudit) -> None:
    print("Common Voice Persian download summary")
    print(f"  dataset id: {audit.dataset_id}")
    print(f"  output path: {audit.output_path}")
    print(f"  bytes written: {audit.bytes_written}")
    print(f"  checksum verified: {audit.checksum_verified}")
    if audit.expires_at:
        print(f"  presigned URL expires at: {audit.expires_at}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch the Mozilla Data Collective Common Voice Persian archive using "
            "an API key loaded from .env and save it under data/."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where the downloaded Common Voice archive will be saved.",
    )
    parser.add_argument(
        "--output-name",
        help="Optional filename override. By default, the API-provided filename is used.",
    )
    parser.add_argument(
        "--env-path",
        type=Path,
        default=Path(".env"),
        help="Path to the .env file containing the Mozilla Data Collective API key.",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help="Environment/.env variable name for the API key.",
    )
    parser.add_argument(
        "--dataset-id",
        default=DATASET_ID,
        help="Mozilla Data Collective dataset ID to download.",
    )
    parser.add_argument(
        "--api-base-url",
        default=API_BASE_URL,
        help="Mozilla Data Collective API base URL.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing archive instead of resuming it.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Fail if the archive already exists instead of resuming with a Range request.",
    )
    args = parser.parse_args(argv)

    audit = download_common_voice_fa(
        output_dir=args.output_dir,
        env_path=args.env_path,
        api_key_env=args.api_key_env,
        dataset_id=args.dataset_id,
        api_base_url=args.api_base_url,
        output_name=args.output_name,
        force=args.force,
        resume=not args.no_resume,
    )
    print_audit(audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
