from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.robotparser
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx


DEFAULT_DELAY_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_USER_AGENT = "MS-Thesis-IranSeda-Research/1.0"
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
ALLOWED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/x-mpeg",
}
MAX_REDIRECTS = 5
INCOMPLETE_HTML_RETRY_SECONDS = 60.0

# These are the public routes linked by the IranSeda catalogue/archive pages.
# Matching is case-insensitive; unknown routes and hosts fail closed.
ALLOWED_ROUTES: dict[str, set[str]] = {
    "book.iranseda.ir": {
        "/",
        "/category/",
        "/categorylist/",
        "/detailsalbum/",
        "/serialhome/",
        "/download",
        "/download/",
        "/robots.txt",
    },
    "radio.iranseda.ir": {
        "/radiolist/",
        "/epglist/",
        "/program/",
        "/epgarchivepart/",
        "/robots.txt",
    },
    "player.iranseda.ir": {
        "/book-player/",
        "/epg-player/",
        "/downloadfile",
        "/robots.txt",
    },
    "headend2.iranseda.ir": {"/dlfile/", "/robots.txt"},
}
AUDIO_ROUTES = {
    ("player.iranseda.ir", "/downloadfile"),
    ("headend2.iranseda.ir", "/dlfile/"),
}


class ScrapeError(RuntimeError):
    """Base error for a safe IranSeda discovery or download failure."""


class RobotsDeniedError(ScrapeError):
    """Raised when an origin's verified robots policy denies a request."""


class TransientResponseError(ScrapeError):
    """Raised when an origin returns a successful but incomplete response."""


@dataclass(frozen=True)
class RobotsRules:
    parser: urllib.robotparser.RobotFileParser
    crawl_delay: float


@dataclass(frozen=True)
class DownloadResult:
    bytes: int
    checksum: str
    media_type: str
    reused: bool


def normalize_space(value: str) -> str:
    return " ".join(value.replace("\u200c", " ").split())


def normalized_url(url: str, *, base_url: str | None = None, audio: bool = False) -> str:
    absolute = urllib.parse.urljoin(base_url or "", url)
    parsed = urllib.parse.urlparse(absolute)
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower() or "/"
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or host not in ALLOWED_ROUTES
        or path not in ALLOWED_ROUTES[host]
    ):
        raise ScrapeError(f"URL is not an allowed IranSeda route: {absolute}")
    if audio and (host, path) not in AUDIO_ROUTES:
        raise ScrapeError(f"URL is not an explicit IranSeda audio route: {absolute}")
    return urllib.parse.urlunparse(parsed._replace(fragment=""))


def is_allowed_url(url: str, *, base_url: str | None = None, audio: bool = False) -> bool:
    try:
        normalized_url(url, base_url=base_url, audio=audio)
    except (ScrapeError, ValueError):
        return False
    return True


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
    """Allowlisted, robots-aware, per-origin-paced HTTP client."""

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        delay_seconds: float = DEFAULT_DELAY_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
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
        self.last_request_at: dict[str, float] = {}
        self.robots_cache: dict[str, RobotsRules] = {}
        self.client = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout_seconds,
            follow_redirects=False,
            transport=transport,
        )

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

    def _pace(self, url: str, minimum: float = 0.0) -> None:
        origin = self.origin(url)
        delay = max(self.delay_seconds, minimum)
        previous = self.last_request_at.get(origin)
        if previous is not None:
            remaining = delay - (self.clock() - previous)
            if remaining > 0:
                self.sleeper(remaining)
        self.last_request_at[origin] = self.clock()

    def _send_once(self, url: str, *, stream: bool, minimum_delay: float = 0.0) -> httpx.Response:
        self._pace(url, minimum_delay)
        request = self.client.build_request("GET", url, headers={"Accept": "*/*"})
        return self.client.send(request, stream=stream)

    def _request(
        self,
        url: str,
        *,
        stream: bool = False,
        check_robots: bool = True,
        accept_client_error: bool = False,
    ) -> httpx.Response:
        current = normalized_url(url)
        crawl_delay = 0.0
        if check_robots:
            crawl_delay = self._assert_robots_allowed(current)
        redirects = 0
        while True:
            response: httpx.Response | None = None
            for attempt in range(self.max_retries + 1):
                try:
                    response = self._send_once(current, stream=stream, minimum_delay=crawl_delay)
                except httpx.HTTPError as error:
                    if attempt == self.max_retries:
                        raise ScrapeError(f"request failed for {current}: {error}") from error
                    self.sleeper(2.0**attempt)
                    continue
                if response.status_code not in TRANSIENT_STATUS_CODES:
                    break
                if attempt == self.max_retries:
                    status = response.status_code
                    response.close()
                    raise ScrapeError(f"request returned {status}: {current}")
                delay = retry_after_seconds(response.headers.get("Retry-After"), now=self.wall_clock)
                response.close()
                self.sleeper(delay if delay is not None else 2.0**attempt)
            assert response is not None
            if response.status_code not in REDIRECT_STATUS_CODES:
                if response.status_code >= 400:
                    status = response.status_code
                    if accept_client_error and 400 <= status < 500:
                        return response
                    response.close()
                    raise ScrapeError(f"request returned {status}: {current}")
                return response
            location = response.headers.get("Location")
            response.close()
            if not location or redirects >= MAX_REDIRECTS:
                raise ScrapeError(f"invalid or excessive redirect from {current}")
            current = normalized_url(location, base_url=current)
            if check_robots:
                crawl_delay = self._assert_robots_allowed(current)
            redirects += 1

    def _robots_rules(self, url: str) -> RobotsRules:
        origin = self.origin(url)
        if origin in self.robots_cache:
            return self.robots_cache[origin]
        robots_url = normalized_url(f"{origin}/robots.txt")
        try:
            response = self._request(
                robots_url,
                check_robots=False,
                accept_client_error=True,
            )
        except ScrapeError as error:
            raise ScrapeError(f"could not verify robots.txt for {origin}") from error
        if 400 <= response.status_code < 500:
            response.close()
            # RFC 9309 section 2.3.1.3 defines 4xx robots responses as
            # unavailable and permits crawlers to access server resources.
            parser = urllib.robotparser.RobotFileParser(robots_url)
            parser.parse(["User-agent: *", "Allow: /"])
            rules = RobotsRules(parser, 0.0)
            self.robots_cache[origin] = rules
            return rules
        try:
            lines = response.text.splitlines()
        finally:
            response.close()
        if not any(line.strip().lower().startswith("user-agent:") for line in lines):
            raise ScrapeError(f"could not verify robots.txt for {origin}: invalid policy")
        parser = urllib.robotparser.RobotFileParser(robots_url)
        parser.parse(lines)
        delay = parser.crawl_delay(self.user_agent)
        if delay is None:
            delay = parser.crawl_delay("*")
        rules = RobotsRules(parser, float(delay or 0.0))
        self.robots_cache[origin] = rules
        return rules

    def _assert_robots_allowed(self, url: str) -> float:
        rules = self._robots_rules(url)
        if not rules.parser.can_fetch(self.user_agent, url):
            raise RobotsDeniedError(f"robots.txt disallows {url}")
        return rules.crawl_delay

    def get_text(self, url: str) -> str:
        for attempt in range(self.max_retries + 1):
            response = self._request(url)
            try:
                text = response.text
                media_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            finally:
                response.close()
            normalized = text.strip().lower()
            incomplete_html = (
                media_type == "text/html"
                and len(text) < 256
                and normalized.startswith(("<!doctype html", "<html"))
                and "</html>" not in normalized
            )
            if not incomplete_html:
                return text
            if attempt == self.max_retries:
                raise TransientResponseError(f"incomplete HTML response after retries: {url}")
            # IranSeda sometimes answers a burst with a 200 response containing
            # only the document declaration. Give its server-side throttle a
            # real cooldown instead of immediately extending the block.
            self.sleeper(INCOMPLETE_HTML_RETRY_SECONDS)
        raise AssertionError("unreachable")

    @contextmanager
    def stream(self, url: str) -> Iterator[httpx.Response]:
        response = self._request(url, stream=True)
        try:
            yield response
        finally:
            response.close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def read_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if isinstance(record.get("id"), str):
                    records[record["id"]] = record
    return records


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def download_audio(
    client: PoliteHttpClient,
    *,
    url: str,
    output_path: Path,
    expected_checksum: str | None = None,
    force: bool = False,
) -> DownloadResult:
    audio_url = normalized_url(url, audio=True)
    if output_path.exists() and not force:
        checksum = sha256_file(output_path)
        if expected_checksum == checksum:
            return DownloadResult(output_path.stat().st_size, checksum, "audio/mpeg", True)
        raise FileExistsError(f"{output_path} does not match its manifest checksum; pass --force to replace it")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with client.stream(audio_url) as response, temporary.open("wb") as handle:
            normalized_url(str(response.url), audio=True)
            media_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if media_type not in ALLOWED_AUDIO_TYPES:
                raise ScrapeError(f"unexpected_audio_content_type:{media_type or 'missing'}")
            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        if size == 0:
            raise ScrapeError("empty_audio_response")
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return DownloadResult(size, f"sha256:{digest.hexdigest()}", media_type, False)
