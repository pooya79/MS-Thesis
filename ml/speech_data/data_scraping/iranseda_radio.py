from __future__ import annotations

import argparse
import html
import re
import urllib.parse
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .iranseda_common import (
    DEFAULT_DELAY_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    PoliteHttpClient,
    ScrapeError,
    normalize_space,
    normalized_url,
    read_jsonl_by_id,
    write_jsonl_atomic,
)


DEFAULT_OUTPUT_ROOT = Path("data/iranseda/radio/raw")
RADIO_BASE_URL = "http://radio.iranseda.ir/"
TEHRAN = ZoneInfo("Asia/Tehran")
DEFAULT_CHECKPOINT_EVERY = 1

# IDs and labels explicitly identify formats/languages unsuitable for Persian ASR.
DEFAULT_EXCLUDED_CHANNELS: dict[str, str] = {
    "15": "explicit_recitation_station",
    "21": "explicit_music_station",
    "101": "explicit_recitation_station",
    "201": "non_persian_station",
    "202": "non_persian_station",
    "203": "explicit_recitation_station",
    "901": "non_persian_station",
    "902": "non_persian_station",
}
RADIO_VISUAL_IDS = {str(value) for value in range(50, 67)}


@dataclass(frozen=True)
class Station:
    id: str
    name: str
    kind: str
    epg_url: str
    included: bool
    classification_reason: str


@dataclass(frozen=True)
class EpgEntry:
    id: str
    station_id: str
    title: str
    date: str
    start_time: str
    duration_minutes: int | None
    archive_url: str
    player_url: str | None


@dataclass(frozen=True)
class ArchiveEpisode:
    id: str
    program_id: str | None
    program_url: str | None
    title: str
    description: str
    participant_producer_text: str
    mp3_url: str | None


@dataclass(frozen=True)
class ScrapeAudit:
    stations: int
    programs: int
    episodes: int
    eligible: int
    resumed: int
    skipped: int


def _strip_tags(value: str) -> str:
    return normalize_space(html.unescape(re.sub(r"<[^>]+>", " ", value)))


def _query_value(url: str, key: str) -> str | None:
    values = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get(key)
    return values[0] if values else None


def parse_iso_date(value: str, *, today: date | None = None) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("date must use Gregorian ISO format YYYY-MM-DD") from error
    if parsed > (today or datetime.now(TEHRAN).date()):
        raise ValueError("date range cannot include future dates")
    return parsed


def validate_date_range(start: str, end: str, *, today: date | None = None) -> tuple[date, date]:
    start_date = parse_iso_date(start, today=today)
    end_date = parse_iso_date(end, today=today)
    if start_date > end_date:
        raise ValueError("start date must not be after end date")
    return start_date, end_date


def inclusive_dates(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def parse_station_list(page_html: str, *, overrides: Iterable[str] | None = None) -> list[Station]:
    found: dict[str, str] = {}
    for match in re.finditer(
        r"""<a\b[^>]*href=["'][^"']*/live/\?[^"']*\bch=(\d+)[^"']*["'][^>]*>(.*?)</a>""",
        page_html,
        re.I | re.S,
    ):
        station_id, label_html = match.groups()
        name = _strip_tags(label_html)
        if name and station_id not in found:
            found[station_id] = name
    override_set = set(overrides or ())
    if override_set:
        missing = sorted(override_set - found.keys())
        if missing:
            raise ScrapeError(f"unknown_channel_id:{','.join(missing)}")
        selected = override_set
    else:
        selected = {
            station_id
            for station_id in found
            if (
                11 <= int(station_id) <= 29
                or 50 <= int(station_id) <= 66
                or 101 <= int(station_id) <= 203
                or 501 <= int(station_id) <= 538
            )
        }
    stations: list[Station] = []
    for station_id in sorted(selected, key=int):
        name = found[station_id]
        kind = "provincial" if 500 <= int(station_id) < 600 else "national"
        reason = "included_override" if override_set else "included_default"
        included = True
        if not override_set:
            if station_id in RADIO_VISUAL_IDS or re.search(r"رادیو\s*نما(?:\s|[-–—]|$)", name):
                included, reason = False, "radio_visual_station"
            elif station_id in DEFAULT_EXCLUDED_CHANNELS:
                included, reason = False, DEFAULT_EXCLUDED_CHANNELS[station_id]
        stations.append(
            Station(
                id=station_id,
                name=name,
                kind=kind,
                epg_url=normalized_url(f"{RADIO_BASE_URL}epglist/?VALID=TRUE&ch={station_id}"),
                included=included,
                classification_reason=reason,
            )
        )
    return stations


def dated_epg_url(station_id: str, day: date) -> str:
    # IranSeda's public schedule uses unpadded Gregorian M/D/YYYY.
    return normalized_url(
        f"{RADIO_BASE_URL}epglist/?VALID=TRUE&ch={station_id}&d={day.month}/{day.day}/{day.year}"
    )


def parse_epg_page(page_html: str, *, station_id: str, day: date) -> list[EpgEntry]:
    matches = list(
        re.finditer(
            r"""href=["']([^"']*/epgarchivePart/\?[^"']*\be=(\d+)[^"']*)["']""",
            page_html,
            re.I,
        )
    )
    entries: list[EpgEntry] = []
    seen: set[str] = set()
    for index, match in enumerate(matches):
        episode_id = match.group(2)
        if episode_id in seen:
            continue
        seen.add(episode_id)
        end = matches[index + 1].start() if index + 1 < len(matches) else min(len(page_html), match.end() + 5000)
        block = page_html[match.start() : end]
        title_match = re.search(r"<h4\b[^>]*itemprop=[\"']name[\"'][^>]*>(.*?)</h4>", block, re.I | re.S)
        if not title_match:
            title_match = re.search(r"<h4\b[^>]*>(.*?)</h4>", block, re.I | re.S)
        title = _strip_tags(title_match.group(1)) if title_match else ""
        start_match = re.search(r"زمان\s*شروع.*?<span\b[^>]*>\s*([\d:]+)", block, re.I | re.S)
        duration_match = re.search(r"مدت.*?<span\b[^>]*>\s*(\d+)\s*دقیقه", block, re.I | re.S)
        player_match = re.search(r"""openPlayer\(["']([^"']*/epg-player/[^"']+)["']\)""", block, re.I)
        entries.append(
            EpgEntry(
                id=episode_id,
                station_id=station_id,
                title=title,
                date=day.isoformat(),
                start_time=start_match.group(1) if start_match else "",
                duration_minutes=int(duration_match.group(1)) if duration_match else None,
                archive_url=normalized_url(html.unescape(match.group(1)), base_url=RADIO_BASE_URL),
                player_url=(
                    normalized_url(html.unescape(player_match.group(1)), base_url=RADIO_BASE_URL)
                    if player_match
                    else None
                ),
            )
        )
    return entries


def entry_completed(entry: EpgEntry, *, now: datetime | None = None) -> bool:
    current = (now or datetime.now(TEHRAN)).astimezone(TEHRAN)
    day = date.fromisoformat(entry.date)
    if day < current.date():
        return True
    if day > current.date() or not entry.start_time:
        return False
    try:
        hour, minute = (int(part) for part in entry.start_time.split(":", 1))
    except ValueError:
        return False
    end_at = datetime.combine(day, time(hour, minute), tzinfo=TEHRAN) + timedelta(
        minutes=entry.duration_minutes or 0
    )
    return end_at <= current


def _first_text(page_html: str, pattern: str) -> str:
    match = re.search(pattern, page_html, re.I | re.S)
    return _strip_tags(match.group(1)) if match else ""


def parse_archive_page(page_html: str, entry: EpgEntry) -> ArchiveEpisode:
    title = _first_text(page_html, r"<h1\b[^>]*itemprop=[\"']name[\"'][^>]*>(.*?)</h1>") or entry.title
    program_match = re.search(r"""href=["']([^"']*/Program/\?[^"']*\bm=(\d+)[^"']*)["']""", page_html, re.I)
    mp3_url: str | None = None
    for match in re.finditer(r"""<a\b[^>]*href=["']([^"']+)["'][^>]*>(.*?)</a>""", page_html, re.I | re.S):
        if "mp3" in _strip_tags(match.group(2)).lower():
            mp3_url = normalized_url(html.unescape(match.group(1)), base_url=entry.archive_url, audio=True)
            break
    descriptions = [
        _strip_tags(value)
        for value in re.findall(r"<p\b[^>]*itemprop=[\"']description[\"'][^>]*>(.*?)</p>", page_html, re.I | re.S)
    ]
    description = max(descriptions, key=len, default="")
    participant_parts: list[str] = []
    for match in re.finditer(
        r"(تهیه\s*کننده|تهیه‌کننده|گوینده|کارشناس|مهمان|مجری|سردبیر|نویسنده)\s*:?\s*</?[^>]*>?\s*([^<]{2,200})",
        page_html,
        re.I,
    ):
        participant_parts.append(normalize_space(f"{match.group(1)}: {html.unescape(match.group(2))}"))
    return ArchiveEpisode(
        id=entry.id,
        program_id=program_match.group(2) if program_match else None,
        program_url=(
            normalized_url(html.unescape(program_match.group(1)), base_url=entry.archive_url)
            if program_match
            else None
        ),
        title=title,
        description=description,
        participant_producer_text=" | ".join(dict.fromkeys(participant_parts)),
        mp3_url=mp3_url,
    )


def parse_program_page(page_html: str, *, program_id: str, program_url: str, station_id: str) -> dict[str, Any]:
    title = _first_text(page_html, r"<h1\b[^>]*itemprop=[\"']name[\"'][^>]*>(.*?)</h1>")
    if not title:
        title = _first_text(page_html, r"<title\b[^>]*>(.*?)</title>")
    descriptions = [
        _strip_tags(value)
        for value in re.findall(r"<p\b[^>]*itemprop=[\"']description[\"'][^>]*>(.*?)</p>", page_html, re.I | re.S)
    ]
    description = max(descriptions, key=len, default="")
    return {
        "id": program_id,
        "station_id": station_id,
        "title": title,
        "description": description,
        "program_url": program_url,
    }


EXCLUSION_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("explicit_music", ("موسیقی", "ترانه", "آهنگ", "تصنیف", "کنسرت")),
    ("explicit_recitation", ("تلاوت", "ترتیل", "قرائت قرآن", "جزء خوانی", "جزءخوانی")),
    ("advertisement_or_promo", ("پیام بازرگانی", "آگهی", "تبلیغ", "تیزر", "پرومو")),
    ("station_filler", ("میان برنامه", "میان‌برنامه", "شناسه شبکه", "اعلام برنامه", "پخش برنامه های")),
    ("non_persian", ("به زبان عربی", "به زبان انگلیسی", "به زبان کردی", "به زبان ترکی", "به زبان آذری")),
)


def classify_episode(*parts: str) -> tuple[bool, str]:
    text = normalize_space(" ".join(parts))
    for reason, keywords in EXCLUSION_RULES:
        if any(keyword in text for keyword in keywords):
            return False, reason
    return True, "broad_speech_default"


def scrape_radio(
    start_date: date,
    end_date: date,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    channel_ids: Iterable[str] | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    refresh: bool = False,
    client: PoliteHttpClient | None = None,
    retrieved_at: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    now: Callable[[], datetime] = lambda: datetime.now(TEHRAN),
) -> ScrapeAudit:
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    if end_date > now().astimezone(TEHRAN).date():
        raise ValueError("date range cannot include future dates")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be greater than zero")
    output_root.mkdir(parents=True, exist_ok=True)
    existing_stations = read_jsonl_by_id(output_root / "stations.jsonl")
    existing_programs = read_jsonl_by_id(output_root / "programs.jsonl")
    existing_episodes = read_jsonl_by_id(output_root / "episodes.jsonl")
    discovery_checkpoints = read_jsonl_by_id(output_root / "discovery_checkpoints.jsonl")
    station_records: dict[str, dict[str, Any]] = dict(existing_stations)
    programs: dict[str, dict[str, Any]] = dict(existing_programs)
    episode_records: dict[str, dict[str, Any]] = dict(existing_episodes)
    skipped: list[dict[str, Any]] = []
    eligible_count = 0
    programs_seen: set[str] = set()
    episodes_seen = 0
    resumed = 0
    work_units = 0
    owns_client = client is None
    client = client or PoliteHttpClient()

    def checkpoint(label: str) -> None:
        write_jsonl_atomic(
            output_root / "stations.jsonl",
            sorted(station_records.values(), key=lambda item: int(item["id"])),
        )
        write_jsonl_atomic(
            output_root / "programs.jsonl",
            sorted(programs.values(), key=lambda item: item["id"]),
        )
        write_jsonl_atomic(
            output_root / "episodes.jsonl",
            sorted(
                episode_records.values(),
                key=lambda item: (
                    item["date"],
                    item["station_id"],
                    item["start_time"],
                    item["id"],
                ),
            ),
        )
        write_jsonl_atomic(
            output_root / "skipped.jsonl",
            sorted(skipped, key=lambda item: (item["id"], item["reason"])),
        )
        write_jsonl_atomic(
            output_root / "discovery_checkpoints.jsonl",
            sorted(discovery_checkpoints.values(), key=lambda item: item["id"]),
        )
        print(
            f"[radio] checkpoint {label}: stations={len(station_records)} "
            f"programs={len(programs)} episodes={len(episode_records)} "
            f"skipped={len(skipped)}",
            flush=True,
        )

    try:
        print("[radio] discovering stations", flush=True)
        station_page = client.get_text(normalized_url(f"{RADIO_BASE_URL}radiolist/?VALID=TRUE"))
        stations = parse_station_list(station_page, overrides=channel_ids)
        for station in stations:
            station_records[station.id] = {**existing_stations.get(station.id, {}), **asdict(station)}
        print(f"[radio] discovered {len(stations)} selected station records", flush=True)
        checkpoint("stations discovered")
        for station_index, station in enumerate(stations, start=1):
            print(
                f"[radio] station {station_index}/{len(stations)} "
                f"id={station.id} name={station.name!r}",
                flush=True,
            )
            if not station.included:
                skipped.append({"id": station.id, "source_url": station.epg_url, "reason": station.classification_reason})
                print(
                    f"[radio] excluded station id={station.id}: {station.classification_reason}",
                    flush=True,
                )
                checkpoint(f"excluded station {station.id}")
                continue
            for day in inclusive_dates(start_date, end_date):
                checkpoint_id = f"{station.id}:{day.isoformat()}"
                if not refresh and checkpoint_id in discovery_checkpoints:
                    resumed += 1
                    work_units += 1
                    print(
                        f"[radio] resume skip station={station.id} "
                        f"date={day.isoformat()} already processed",
                        flush=True,
                    )
                    if work_units % checkpoint_every == 0:
                        checkpoint(f"station={station.id} date={day.isoformat()}")
                    continue
                print(f"[radio] fetching station={station.id} date={day.isoformat()}", flush=True)
                epg_url = dated_epg_url(station.id, day)
                try:
                    entries = parse_epg_page(client.get_text(epg_url), station_id=station.id, day=day)
                except ScrapeError as error:
                    skipped.append({"id": f"{station.id}:{day}", "source_url": epg_url, "reason": str(error)})
                    work_units += 1
                    print(
                        f"[radio] skipped station={station.id} date={day.isoformat()}: {error}",
                        flush=True,
                    )
                    if work_units % checkpoint_every == 0:
                        checkpoint(f"station={station.id} date={day.isoformat()}")
                    continue
                day_had_errors = False
                for entry in entries:
                    if not entry_completed(entry, now=now()):
                        skipped.append({"id": entry.id, "source_url": entry.archive_url, "reason": "episode_not_completed"})
                        day_had_errors = True
                        continue
                    try:
                        archive = parse_archive_page(client.get_text(entry.archive_url), entry)
                    except ScrapeError as error:
                        skipped.append({"id": entry.id, "source_url": entry.archive_url, "reason": str(error)})
                        day_had_errors = True
                        continue
                    program: dict[str, Any] | None = None
                    if archive.program_id and archive.program_url:
                        programs_seen.add(archive.program_id)
                        if archive.program_id not in programs:
                            try:
                                programs[archive.program_id] = parse_program_page(
                                    client.get_text(archive.program_url),
                                    program_id=archive.program_id,
                                    program_url=archive.program_url,
                                    station_id=station.id,
                                )
                            except ScrapeError as error:
                                day_had_errors = True
                                skipped.append(
                                    {"id": archive.program_id, "source_url": archive.program_url, "reason": str(error)}
                                )
                        program = programs.get(archive.program_id)
                    eligible, classification_reason = classify_episode(
                        entry.title,
                        archive.title,
                        archive.description,
                        archive.participant_producer_text,
                        str(program.get("title", "")) if program else "",
                        str(program.get("description", "")) if program else "",
                    )
                    if eligible:
                        eligible_count += 1
                    episodes_seen += 1
                    timestamp = retrieved_at().astimezone(timezone.utc).isoformat()
                    record: dict[str, Any] = {
                        **existing_episodes.get(entry.id, {}),
                        **asdict(entry),
                        "station": station.name,
                        "program_id": archive.program_id,
                        "program_url": archive.program_url,
                        "title": archive.title or entry.title,
                        "description": archive.description,
                        "participant_producer_text": archive.participant_producer_text,
                        "mp3_url": archive.mp3_url,
                        "eligible": eligible,
                        "classification_reason": classification_reason,
                        "retrieved_at": timestamp,
                    }
                    if not eligible:
                        skipped.append({"id": entry.id, "source_url": entry.archive_url, "reason": classification_reason})
                    elif archive.mp3_url is None:
                        skipped.append({"id": entry.id, "source_url": entry.archive_url, "reason": "missing_explicit_mp3"})
                    episode_records[entry.id] = record
                if not day_had_errors:
                    discovery_checkpoints[checkpoint_id] = {
                        "id": checkpoint_id,
                        "status": "discovered",
                        "epg_entries": len(entries),
                        "completed_at": retrieved_at().astimezone(timezone.utc).isoformat(),
                    }
                work_units += 1
                print(
                    f"[radio] processed station={station.id} date={day.isoformat()} "
                    f"epg_entries={len(entries)} total_episodes={episodes_seen}",
                    flush=True,
                )
                if work_units % checkpoint_every == 0:
                    checkpoint(f"station={station.id} date={day.isoformat()}")
    except BaseException:
        checkpoint("interrupted")
        raise
    finally:
        if owns_client:
            client.close()
    checkpoint("complete")
    return ScrapeAudit(
        len(stations),
        len(programs_seen),
        episodes_seen,
        eligible_count,
        resumed,
        len(skipped),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Discover completed IranSeda radio archive metadata and explicit MP3 links without downloading audio."
    )
    parser.add_argument("--start-date", required=True, help="Inclusive Gregorian start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", required=True, help="Inclusive Gregorian end date in YYYY-MM-DD format.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help=f"Output directory (default: {DEFAULT_OUTPUT_ROOT}).")
    parser.add_argument("--channel-id", action="append", dest="channel_ids", help="IranSeda channel ID; repeat to override default station discovery.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="Write manifests after this many processed station-days (default: 1).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Revisit station-days already recorded in discovery checkpoints.",
    )
    parser.add_argument("--delay-seconds", type=float, default=DEFAULT_DELAY_SECONDS, help="Minimum per-origin delay in seconds (default: 1.0).")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="Per-request timeout in seconds (default: 120.0).")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help=f"HTTP User-Agent (default: {DEFAULT_USER_AGENT}).")
    args = parser.parse_args(argv)
    try:
        start_date, end_date = validate_date_range(args.start_date, args.end_date)
    except ValueError as error:
        parser.error(str(error))
    if args.delay_seconds < 0:
        parser.error("--delay-seconds must be non-negative")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be greater than zero")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be greater than zero")
    with PoliteHttpClient(
        user_agent=args.user_agent,
        delay_seconds=args.delay_seconds,
        timeout_seconds=args.timeout_seconds,
    ) as client:
        audit = scrape_radio(
            start_date,
            end_date,
            args.output_root,
            channel_ids=args.channel_ids,
            checkpoint_every=args.checkpoint_every,
            refresh=args.refresh,
            client=client,
        )
    print("IranSeda radio scrape summary")
    for key, value in asdict(audit).items():
        print(f"  {key.replace('_', ' ')}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
