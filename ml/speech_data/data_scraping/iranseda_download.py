from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

from .iranseda_common import (
    DEFAULT_DELAY_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    DownloadTrafficLimitError,
    PoliteHttpClient,
    ScrapeError,
    download_audio,
    write_jsonl_atomic,
)


SourceKind = Literal["audiobooks", "radio"]


@dataclass(frozen=True)
class DownloadCandidate:
    id: str
    source_kind: SourceKind
    url: str
    relative_path: Path
    source_record: dict[str, Any]


@dataclass(frozen=True)
class DownloadAudit:
    selected: int
    downloaded: int
    reused: int
    failed: int
    bytes_downloaded: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ScrapeError(f"invalid_json:{path.name}:{line_number}") from error
                if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                    raise ScrapeError(f"invalid_record:{path.name}:{line_number}:missing_string_id")
                if record["id"] in seen:
                    raise ScrapeError(f"duplicate_id:{path.name}:{record['id']}")
                seen.add(record["id"])
                records.append(record)
    except OSError as error:
        raise ScrapeError(f"could_not_read_manifest:{path}") from error
    return records


def _read_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {record["id"]: record for record in _read_jsonl(path)}


def detect_source(source_root: Path) -> tuple[SourceKind, Path]:
    tracks = source_root / "tracks.jsonl"
    episodes = source_root / "episodes.jsonl"
    present = [path for path in (tracks, episodes) if path.is_file()]
    if len(present) != 1:
        raise ScrapeError(
            "source root must contain exactly one of tracks.jsonl or episodes.jsonl"
        )
    return ("audiobooks", tracks) if present[0] == tracks else ("radio", episodes)


def _numeric_component(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not re.fullmatch(r"\d+", value):
        raise ScrapeError(f"invalid_{field}")
    return value


def _audio_url(record: dict[str, Any]) -> str | None:
    value = record.get("mp3_url")
    return value if isinstance(value, str) and value.strip() else None


def _book_candidates(records: Iterable[dict[str, Any]]) -> list[DownloadCandidate]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        book_id = record.get("book_id")
        if not isinstance(book_id, str):
            raise ScrapeError(f"invalid_book_id:{record['id']}")
        grouped[book_id].append(record)

    candidates: list[DownloadCandidate] = []
    for tracks in grouped.values():
        full_tracks = [
            track
            for track in tracks
            if track.get("is_full_book") is True and _audio_url(track)
        ]
        selected = full_tracks[:1] or [
            track for track in tracks if track.get("is_sample") is not True and _audio_url(track)
        ]
        for track in selected:
            safe_book_id = _numeric_component(track, "book_id")
            attachment_id = _numeric_component(track, "attachment_id")
            candidates.append(
                DownloadCandidate(
                    id=track["id"],
                    source_kind="audiobooks",
                    url=_audio_url(track) or "",
                    relative_path=Path("clips") / safe_book_id / f"{attachment_id}.mp3",
                    source_record=track,
                )
            )
    return candidates


def _radio_candidates(
    records: Iterable[dict[str, Any]],
) -> tuple[list[DownloadCandidate], list[dict[str, Any]]]:
    candidates: list[DownloadCandidate] = []
    failures: list[dict[str, Any]] = []
    for episode in records:
        if episode.get("eligible") is not True:
            continue
        url = _audio_url(episode)
        if url is None:
            failures.append(
                {"id": episode["id"], "url": None, "reason": "missing_explicit_mp3"}
            )
            continue
        try:
            station_id = _numeric_component(episode, "station_id")
            episode_id = _numeric_component(episode, "id")
            day = episode.get("date")
            if not isinstance(day, str):
                raise ScrapeError("invalid_date")
            try:
                date.fromisoformat(day)
            except ValueError as error:
                raise ScrapeError("invalid_date") from error
        except ScrapeError as error:
            failures.append({"id": episode["id"], "url": url, "reason": str(error)})
            continue
        candidates.append(
            DownloadCandidate(
                id=episode["id"],
                source_kind="radio",
                url=url,
                relative_path=Path("clips") / station_id / day / f"{episode_id}.mp3",
                source_record=episode,
            )
        )
    return candidates, failures


def discover_candidates(
    source_kind: SourceKind, records: Iterable[dict[str, Any]]
) -> tuple[list[DownloadCandidate], list[dict[str, Any]]]:
    if source_kind == "audiobooks":
        return _book_candidates(records), []
    return _radio_candidates(records)


def _expected_checksum(
    candidate: DownloadCandidate, existing_state: dict[str, Any] | None
) -> str | None:
    relative = candidate.relative_path.as_posix()
    for record in (existing_state, candidate.source_record):
        if not record or record.get("path") != relative:
            continue
        checksum = record.get("checksum")
        if isinstance(checksum, str) and checksum.startswith("sha256:"):
            return checksum
    return None


def _ordered(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: str(record["id"]))


def _gib_to_bytes(value: str) -> int:
    try:
        gibibytes = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if not gibibytes.is_finite() or gibibytes <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    byte_count = int(gibibytes * (1024**3))
    if byte_count < 1:
        raise argparse.ArgumentTypeError("must represent at least one byte")
    return byte_count


def download_discovered(
    source_root: Path,
    *,
    force: bool = False,
    max_download_bytes: int | None = None,
    client: PoliteHttpClient | None = None,
    retrieved_at: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> DownloadAudit:
    if max_download_bytes is not None and max_download_bytes <= 0:
        raise ValueError("max_download_bytes must be greater than zero")
    source_kind, manifest_path = detect_source(source_root)
    manifest_records = _read_jsonl(manifest_path)
    candidates, selection_failures = discover_candidates(source_kind, manifest_records)
    print(
        f"[download] source={source_kind} manifest={manifest_path} "
        f"records={len(manifest_records)} selected={len(candidates) + len(selection_failures)}",
        flush=True,
    )
    state_path = source_root / "downloads.jsonl"
    failure_path = source_root / "download_skipped.jsonl"
    states = _read_state(state_path)
    failures: list[dict[str, Any]] = []
    write_jsonl_atomic(state_path, _ordered(states.values()))
    write_jsonl_atomic(failure_path, failures)

    downloaded = reused = bytes_downloaded = 0
    timestamp = lambda: retrieved_at().astimezone(timezone.utc).isoformat()
    for failure in selection_failures:
        failures.append({**failure, "source_kind": source_kind, "failed_at": timestamp()})
        write_jsonl_atomic(failure_path, _ordered(failures))
        print(f"[download] skipped id={failure['id']}: {failure['reason']}", flush=True)

    owns_client = client is None
    client = client or PoliteHttpClient()
    try:
        for index, candidate in enumerate(candidates, start=1):
            remaining_bytes = (
                None
                if max_download_bytes is None
                else max_download_bytes - bytes_downloaded
            )
            print(
                f"[download] item {index}/{len(candidates)} id={candidate.id} "
                f"path={candidate.relative_path.as_posix()}"
                + (
                    ""
                    if remaining_bytes is None
                    else f" remaining_bytes={remaining_bytes}"
                ),
                flush=True,
            )
            old = states.get(candidate.id)
            expected_checksum = _expected_checksum(candidate, old)
            try:
                result = download_audio(
                    client,
                    url=candidate.url,
                    output_path=source_root / candidate.relative_path,
                    expected_checksum=expected_checksum,
                    force=force,
                    max_download_bytes=remaining_bytes,
                )
            except DownloadTrafficLimitError as error:
                failures.append(
                    {
                        "id": candidate.id,
                        "source_kind": source_kind,
                        "url": candidate.url,
                        "reason": str(error),
                        "failed_at": timestamp(),
                    }
                )
                write_jsonl_atomic(failure_path, _ordered(failures))
                print(f"[download] stopped id={candidate.id}: {error}", flush=True)
                break
            except (ScrapeError, FileExistsError, OSError) as error:
                failures.append(
                    {
                        "id": candidate.id,
                        "source_kind": source_kind,
                        "url": candidate.url,
                        "reason": str(error),
                        "failed_at": timestamp(),
                    }
                )
                write_jsonl_atomic(failure_path, _ordered(failures))
                print(f"[download] failed id={candidate.id}: {error}", flush=True)
                continue

            now = timestamp()
            states[candidate.id] = {
                "id": candidate.id,
                "source_kind": source_kind,
                "source_manifest": manifest_path.name,
                "url": candidate.url,
                "path": candidate.relative_path.as_posix(),
                "checksum": result.checksum,
                "bytes": result.bytes,
                "media_type": result.media_type,
                "downloaded_at": (
                    old.get("downloaded_at")
                    if result.reused and old and isinstance(old.get("downloaded_at"), str)
                    else now
                ),
                "verified_at": now,
            }
            write_jsonl_atomic(state_path, _ordered(states.values()))
            if result.reused:
                reused += 1
                print(f"[download] reused id={candidate.id} bytes={result.bytes}", flush=True)
            else:
                downloaded += 1
                bytes_downloaded += result.bytes
                print(f"[download] downloaded id={candidate.id} bytes={result.bytes}", flush=True)
    finally:
        if owns_client:
            client.close()

    return DownloadAudit(
        selected=len(candidates) + len(selection_failures),
        downloaded=downloaded,
        reused=reused,
        failed=len(failures),
        bytes_downloaded=bytes_downloaded,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download dataset-eligible IranSeda MP3s from a saved discovery output root."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Discovery directory containing exactly one of tracks.jsonl or episodes.jsonl.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help="Minimum per-origin delay in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-request timeout in seconds (default: 30.0).",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help=f"HTTP User-Agent (default: {DEFAULT_USER_AGENT}).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace selected audio instead of checksum-based reuse."
    )
    parser.add_argument(
        "--max-download-gib",
        dest="max_download_bytes",
        type=_gib_to_bytes,
        default=None,
        metavar="GIB",
        help="Maximum audio GiB to transfer in this run; accepts decimals and is unlimited by default.",
    )
    args = parser.parse_args(argv)
    if args.delay_seconds < 0:
        parser.error("--delay-seconds must be non-negative")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be greater than zero")
    try:
        with PoliteHttpClient(
            user_agent=args.user_agent,
            delay_seconds=args.delay_seconds,
            timeout_seconds=args.timeout_seconds,
        ) as client:
            audit = download_discovered(
                args.source_root,
                force=args.force,
                max_download_bytes=args.max_download_bytes,
                client=client,
            )
    except ScrapeError as error:
        parser.error(str(error))
    print("IranSeda download summary")
    for key, value in asdict(audit).items():
        print(f"  {key.replace('_', ' ')}: {value}")
    return 1 if audit.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
