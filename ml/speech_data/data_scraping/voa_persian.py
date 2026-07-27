from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.robotparser
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx


DEFAULT_FEED_URLS = (
    "https://ir.voanews.com/api/zrkqorl-vomx-tpeoppoq",  # Tafsir-e Khabar audio
    "https://ir.voanews.com/api/zyv-qql-vomx-tpetqqoo",  # YadAr
    "https://ir.voanews.com/api/zmyrorl-vomx-tpeyrior",  # Taboo audio
)
DEFAULT_OUTPUT_ROOT = Path("data/voa-persian/raw")
DEFAULT_USER_AGENT = "MS-Thesis-VOA-Persian-Audio/1.0"
RIGHTS_POLICY_URL = "https://ir.voanews.com/p/6154.html"
ALLOWED_PAGE_HOSTS = {"ir.voanews.com"}
ALLOWED_AUDIO_HOSTS = {
    "voa-audio-ns.akamaized.net",
    "voa-audio.voanews.eu",
}
ALLOWED_AUDIO_TYPES = {"audio/mpeg", "audio/mp3", "audio/x-mpeg"}
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
MAX_REDIRECTS = 5


class ScrapeError(RuntimeError):
    """Base error for a fetch or parsing failure."""


class RobotsDeniedError(ScrapeError):
    """Raised when robots.txt disallows a URL."""


@dataclass(frozen=True)
class FeedEpisode:
    feed_url: str
    feed_title: str
    title: str
    published_at: str
    source_url: str


@dataclass(frozen=True)
class PageAudio:
    audio_url: str
    media_type: str
    entity: str
    copied: str
    language_service: str
    content_type: str


@dataclass(frozen=True)
class DownloadAudit:
    feeds_read: int
    episodes_seen: int
    downloaded: int
    reused: int
    skipped: int
    bytes_downloaded: int


@dataclass(frozen=True)
class RobotsRules:
    parser: urllib.robotparser.RobotFileParser
    crawl_delay: float


class AudioPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.audio_candidates: list[tuple[str, str, int]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() == "meta" and values.get("name", "").lower() == "twitter:player:stream":
            url = values.get("content")
            if url:
                self.audio_candidates.append((url, "audio/mpeg", bitrate_from_url(url)))
            return
        if tag.lower() != "audio":
            return

        source = values.get("src")
        media_type = values.get("data-type") or "audio/mpeg"
        if source:
            self.audio_candidates.append(
                (source, media_type, bitrate_from_label(values.get("data-info", "")) or bitrate_from_url(source))
            )

        raw_sources = values.get("data-sources")
        if not raw_sources:
            return
        try:
            sources = json.loads(html.unescape(raw_sources))
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(sources, list):
            return
        for item in sources:
            if not isinstance(item, dict):
                continue
            url = item.get("Src") or item.get("AmpSrc")
            if not isinstance(url, str) or not url:
                continue
            item_type = item.get("Type") if isinstance(item.get("Type"), str) else media_type
            label = item.get("DataInfo") if isinstance(item.get("DataInfo"), str) else ""
            self.audio_candidates.append((url, item_type, bitrate_from_label(label) or bitrate_from_url(url)))


def bitrate_from_label(value: str) -> int:
    match = re.search(r"(\d+)\s*kbps", value, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def bitrate_from_url(url: str) -> int:
    return 128 if Path(urllib.parse.urlparse(url).path).stem.endswith("_hq") else 64


def analytics_value(page_html: str, key: str) -> str | None:
    match = re.search(rf'(?:^|[,{{])\s*{re.escape(key)}\s*:\s*"([^"]*)"', page_html)
    return html.unescape(match.group(1)) if match else None


def normalized_https_url(url: str, *, allowed_hosts: set[str]) -> str:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme.lower() != "https" or host not in allowed_hosts:
        raise ScrapeError(f"URL is not an allowed HTTPS origin: {url}")
    return urllib.parse.urlunparse(parsed._replace(fragment=""))


def parse_feed(feed_url: str, payload: str, *, max_items: int | None = None) -> list[FeedEpisode]:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as error:
        raise ScrapeError(f"invalid RSS XML from {feed_url}") from error

    channel = root.find("channel")
    if channel is None:
        raise ScrapeError(f"RSS feed has no channel: {feed_url}")
    language = (channel.findtext("language") or "").strip().lower()
    generator = (channel.findtext("generator") or "").strip().lower()
    if not language.startswith("fa") or "voa" not in generator:
        raise ScrapeError(f"RSS feed is not identified as an official Persian VOA feed: {feed_url}")

    feed_title = (channel.findtext("title") or "").strip()
    episodes: list[FeedEpisode] = []
    for item in channel.findall("item"):
        source_url = (item.findtext("link") or item.findtext("guid") or "").strip()
        title = (item.findtext("title") or "").strip()
        published = (item.findtext("pubDate") or "").strip()
        if not source_url or not title or not published:
            continue
        try:
            published_at = parsedate_to_datetime(published).isoformat()
        except (TypeError, ValueError):
            continue
        episodes.append(
            FeedEpisode(
                feed_url=feed_url,
                feed_title=feed_title,
                title=title,
                published_at=published_at,
                source_url=source_url,
            )
        )
        if max_items is not None and len(episodes) >= max_items:
            break
    return episodes


def parse_audio_page(page_html: str) -> PageAudio:
    provenance = {
        key: analytics_value(page_html, key)
        for key in ("entity", "copied", "language_service", "content_type")
    }
    expected = {
        "entity": "VOA",
        "copied": "no",
        "language_service": "VOA Persian",
        "content_type": "audio",
    }
    for key, expected_value in expected.items():
        if provenance[key] != expected_value:
            raise ScrapeError(f"unverified_provenance:{key}")

    parser = AudioPageParser()
    parser.feed(page_html)
    candidates: list[tuple[str, str, int]] = []
    for url, media_type, bitrate in parser.audio_candidates:
        try:
            normalized = normalized_https_url(url, allowed_hosts=ALLOWED_AUDIO_HOSTS)
        except ScrapeError:
            continue
        candidates.append((normalized, media_type.split(";", 1)[0].strip().lower(), bitrate))
    if not candidates:
        raise ScrapeError("missing_official_audio")

    audio_url, media_type, _ = max(candidates, key=lambda candidate: (candidate[2], candidate[0]))
    return PageAudio(
        audio_url=audio_url,
        media_type=media_type,
        entity=expected["entity"],
        copied=expected["copied"],
        language_service=expected["language_service"],
        content_type=expected["content_type"],
    )


def article_id(source_url: str) -> str:
    path = urllib.parse.urlparse(source_url).path
    match = re.search(r"/(\d+)\.html$", path)
    if not match:
        raise ScrapeError("source URL does not contain a VOA article ID")
    return match.group(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retry_after_seconds(value: str | None, *, now: Callable[[], float]) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, retry_at.timestamp() - now())


class PoliteHttpClient:
    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        delay_seconds: float = 1.0,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.user_agent = user_agent
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self.clock = clock
        self.wall_clock = wall_clock
        self.sleeper = sleeper
        self.client = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout_seconds,
            follow_redirects=False,
            transport=transport,
        )
        self.last_request_at: dict[str, float] = {}
        self.robots_cache: dict[str, RobotsRules] = {}

    def __enter__(self) -> PoliteHttpClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self.client.close()

    @staticmethod
    def origin(url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"

    def _pace(self, url: str, extra_delay: float = 0.0) -> None:
        origin = self.origin(url)
        delay = max(self.delay_seconds, extra_delay)
        last_request = self.last_request_at.get(origin)
        if last_request is not None:
            remaining = delay - (self.clock() - last_request)
            if remaining > 0:
                self.sleeper(remaining)
        self.last_request_at[origin] = self.clock()

    def _send_once(self, url: str, *, stream: bool) -> httpx.Response:
        self._pace(url)
        request = self.client.build_request("GET", url, headers={"Accept": "*/*"})
        return self.client.send(request, stream=stream)

    def _raw_request(self, url: str, *, stream: bool = False) -> httpx.Response:
        current_url = url
        redirects = 0
        while True:
            for attempt in range(self.max_retries + 1):
                try:
                    response = self._send_once(current_url, stream=stream)
                except httpx.HTTPError as error:
                    if attempt >= self.max_retries:
                        raise ScrapeError(f"request failed for {current_url}: {error}") from error
                    self.sleeper(2.0**attempt)
                    continue
                if response.status_code not in TRANSIENT_STATUS_CODES:
                    break
                if attempt >= self.max_retries:
                    response.close()
                    raise ScrapeError(f"request returned {response.status_code}: {current_url}")
                retry_after = retry_after_seconds(response.headers.get("Retry-After"), now=self.wall_clock)
                response.close()
                self.sleeper(retry_after if retry_after is not None else 2.0**attempt)

            if response.status_code not in REDIRECT_STATUS_CODES:
                if response.status_code >= 400:
                    status = response.status_code
                    response.close()
                    raise ScrapeError(f"request returned {status}: {current_url}")
                return response
            location = response.headers.get("Location")
            response.close()
            if not location or redirects >= MAX_REDIRECTS:
                raise ScrapeError(f"invalid or excessive redirect from {current_url}")
            current_url = urllib.parse.urljoin(current_url, location)
            self._assert_robots_allowed(current_url)
            redirects += 1

    def _robots_rules(self, url: str) -> RobotsRules:
        origin = self.origin(url)
        cached = self.robots_cache.get(origin)
        if cached is not None:
            return cached

        robots_url = f"{origin}/robots.txt"
        try:
            response = self._raw_request(robots_url)
        except ScrapeError as error:
            if "returned 404" not in str(error):
                raise ScrapeError(f"could not verify robots.txt for {origin}") from error
            lines = ["User-agent: *", "Allow: /"]
        else:
            try:
                lines = response.text.splitlines()
            finally:
                response.close()

        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(lines)
        delay = parser.crawl_delay(self.user_agent)
        if delay is None:
            delay = parser.crawl_delay("*")
        rules = RobotsRules(parser=parser, crawl_delay=float(delay or 0.0))
        self.robots_cache[origin] = rules
        return rules

    def _assert_robots_allowed(self, url: str) -> None:
        rules = self._robots_rules(url)
        if not rules.parser.can_fetch(self.user_agent, url):
            raise RobotsDeniedError(f"robots.txt disallows {url}")
        origin = self.origin(url)
        if rules.crawl_delay > self.delay_seconds:
            last_request = self.last_request_at.get(origin)
            if last_request is not None:
                remaining = rules.crawl_delay - (self.clock() - last_request)
                if remaining > 0:
                    self.sleeper(remaining)

    def get_text(self, url: str) -> str:
        self._assert_robots_allowed(url)
        response = self._raw_request(url)
        try:
            return response.text
        finally:
            response.close()

    @contextmanager
    def stream(self, url: str) -> Iterator[httpx.Response]:
        self._assert_robots_allowed(url)
        response = self._raw_request(url, stream=True)
        try:
            yield response
        finally:
            response.close()


def read_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = record.get("id")
            if isinstance(record_id, str):
                records[record_id] = record
    return records


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def skip_record(episode: FeedEpisode, reason: str) -> dict[str, Any]:
    return {
        "feed_url": episode.feed_url,
        "feed_title": episode.feed_title,
        "title": episode.title,
        "published_at": episode.published_at,
        "source_url": episode.source_url,
        "reason": reason,
    }


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def download_audio(
    client: PoliteHttpClient,
    *,
    url: str,
    output_path: Path,
) -> tuple[int, str, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    bytes_written = 0
    try:
        with client.stream(url) as response, temporary.open("wb") as handle:
            normalized_https_url(str(response.url), allowed_hosts=ALLOWED_AUDIO_HOSTS)
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type not in ALLOWED_AUDIO_TYPES:
                raise ScrapeError(f"unexpected_audio_content_type:{content_type or 'missing'}")
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
                bytes_written += len(chunk)
        if bytes_written == 0:
            raise ScrapeError("empty_audio_response")
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return bytes_written, f"sha256:{digest.hexdigest()}", content_type


def download_voa_persian(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    feed_urls: Iterable[str] = DEFAULT_FEED_URLS,
    max_items: int | None = None,
    force: bool = False,
    client: PoliteHttpClient | None = None,
    retrieved_at: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    log: Callable[[str], None] = print,
) -> DownloadAudit:
    if max_items is not None and max_items <= 0:
        raise ValueError("max_items must be greater than zero")

    feed_urls = tuple(feed_urls)
    output_root.mkdir(parents=True, exist_ok=True)
    clips_root = output_root / "clips"
    metadata_path = output_root / "metadata.jsonl"
    existing = read_jsonl_by_id(metadata_path)
    metadata = dict(existing)
    skipped_records: list[dict[str, Any]] = []
    downloaded = reused = bytes_downloaded = episodes_seen = feeds_read = 0
    owns_client = client is None
    client = client or PoliteHttpClient()

    try:
        for raw_feed_url in feed_urls:
            log(f"Reading feed: {raw_feed_url}")
            try:
                feed_url = normalized_https_url(raw_feed_url, allowed_hosts=ALLOWED_PAGE_HOSTS)
                episodes = parse_feed(feed_url, client.get_text(feed_url), max_items=max_items)
            except ScrapeError as error:
                log(f"  feed skipped: {error}")
                skipped_records.append({"feed_url": raw_feed_url, "reason": str(error)})
                continue
            feeds_read += 1
            episodes_seen += len(episodes)
            log(f"  found {len(episodes)} item(s) in {episodes[0].feed_title if episodes else feed_url}")

            for item_index, episode in enumerate(episodes, start=1):
                prefix = f"  [{item_index}/{len(episodes)}]"
                log(f"{prefix} Inspecting: {episode.title}")
                try:
                    source_url = normalized_https_url(episode.source_url, allowed_hosts=ALLOWED_PAGE_HOSTS)
                    page_audio = parse_audio_page(client.get_text(source_url))
                    audio_url = normalized_https_url(page_audio.audio_url, allowed_hosts=ALLOWED_AUDIO_HOSTS)
                    record_id = article_id(source_url)
                except ScrapeError as error:
                    log(f"{prefix} Skipped: {error}")
                    skipped_records.append(skip_record(episode, str(error)))
                    continue

                relative_path = Path("clips") / f"{record_id}.mp3"
                output_path = output_root / relative_path
                old_record = existing.get(record_id)
                if output_path.exists() and not force:
                    expected = old_record.get("checksum") if old_record else None
                    actual = f"sha256:{sha256_file(output_path)}"
                    if expected == actual:
                        reused += 1
                        log(
                            f"{prefix} Reused: {relative_path.as_posix()} "
                            f"({human_bytes(output_path.stat().st_size)})"
                        )
                        continue
                    raise FileExistsError(
                        f"{output_path} conflicts with its recorded checksum; pass --force to replace it"
                    )

                log(f"{prefix} Downloading: {relative_path.as_posix()}")
                try:
                    size, checksum, response_type = download_audio(client, url=audio_url, output_path=output_path)
                except ScrapeError as error:
                    log(f"{prefix} Skipped during download: {error}")
                    skipped_records.append(skip_record(episode, str(error)))
                    continue

                downloaded += 1
                bytes_downloaded += size
                log(
                    f"{prefix} Downloaded: {relative_path.as_posix()} "
                    f"({human_bytes(size)}; cumulative audio {human_bytes(bytes_downloaded)})"
                )
                metadata[record_id] = {
                    "id": record_id,
                    "path": relative_path.as_posix(),
                    "source_url": source_url,
                    "feed_url": episode.feed_url,
                    "feed_title": episode.feed_title,
                    "title": episode.title,
                    "published_at": episode.published_at,
                    "audio_url": audio_url,
                    "checksum": checksum,
                    "bytes": size,
                    "media_type": response_type or page_audio.media_type,
                    "language": "fa",
                    "entity": page_audio.entity,
                    "copied": page_audio.copied,
                    "language_service": page_audio.language_service,
                    "content_type": page_audio.content_type,
                    "rights_policy_url": RIGHTS_POLICY_URL,
                    "retrieved_at": retrieved_at().astimezone(timezone.utc).isoformat(),
                }
    finally:
        if owns_client:
            client.close()

    ordered_metadata = sorted(
        metadata.values(),
        key=lambda record: (str(record.get("published_at", "")), str(record.get("id", ""))),
    )
    ordered_skips = sorted(
        skipped_records,
        key=lambda record: (str(record.get("published_at", "")), str(record.get("source_url", ""))),
    )
    write_jsonl_atomic(metadata_path, ordered_metadata)
    write_jsonl_atomic(output_root / "skipped.jsonl", ordered_skips)
    return DownloadAudit(
        feeds_read=feeds_read,
        episodes_seen=episodes_seen,
        downloaded=downloaded,
        reused=reused,
        skipped=len(skipped_records),
        bytes_downloaded=bytes_downloaded,
    )


def print_audit(audit: DownloadAudit) -> None:
    print("VOA Persian raw audio download summary")
    for key, value in asdict(audit).items():
        print(f"  {key.replace('_', ' ')}: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download raw, publicly accessible, VOA-produced Persian podcast audio "
            "from official VOA RSS feeds into clips/ with provenance JSONL metadata."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Raw dataset directory (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--feed-url",
        action="append",
        dest="feed_urls",
        help="Official VOA Persian RSS URL. Repeat to override the built-in audio feed list.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        help="Maximum number of RSS items to inspect per feed (default: all exposed items).",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Minimum delay between requests to the same origin (default: 1.0).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds (default: 30.0).",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help=f"HTTP User-Agent header (default: {DEFAULT_USER_AGENT}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload selected audio and replace existing matching article IDs.",
    )
    args = parser.parse_args(argv)
    if args.delay_seconds < 0:
        parser.error("--delay-seconds must be non-negative")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be greater than zero")

    with PoliteHttpClient(
        user_agent=args.user_agent,
        delay_seconds=args.delay_seconds,
        timeout_seconds=args.timeout_seconds,
    ) as client:
        audit = download_voa_persian(
            args.output_root,
            feed_urls=args.feed_urls or DEFAULT_FEED_URLS,
            max_items=args.max_items,
            force=args.force,
            client=client,
        )
    print_audit(audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
