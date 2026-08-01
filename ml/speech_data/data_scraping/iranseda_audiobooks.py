from __future__ import annotations

import argparse
import html
import re
import urllib.parse
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


DEFAULT_OUTPUT_ROOT = Path("data/iranseda/audiobooks/raw")
BOOK_BASE_URL = "http://book.iranseda.ir/"
DEFAULT_CHECKPOINT_EVERY = 10


@dataclass(frozen=True)
class BookLink:
    id: str
    url: str
    serial_parent_id: str | None = None
    serial_parent_url: str | None = None


@dataclass(frozen=True)
class Track:
    id: str
    book_id: str
    title: str
    duration: str
    duration_seconds: int | None
    attachment_id: str
    player_url: str
    download_modal_url: str
    mp3_url: str | None = None
    declared_size: str | None = None
    is_sample: bool = False
    is_full_book: bool = False


@dataclass(frozen=True)
class Book:
    id: str
    title: str
    authors: tuple[str, ...]
    narrators: tuple[str, ...]
    description: str
    categories: tuple[str, ...]
    total_duration: str | None
    source_url: str
    serial_parent_id: str | None
    serial_parent_url: str | None
    tracks: tuple[Track, ...]


@dataclass(frozen=True)
class ScrapeAudit:
    books: int
    tracks: int
    resumed: int
    skipped: int


def _strip_tags(value: str) -> str:
    return normalize_space(html.unescape(re.sub(r"<[^>]+>", " ", value)))


def _query_id(url: str, key: str) -> str | None:
    values = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get(key)
    return values[0] if values and re.fullmatch(r"\d+", values[0]) else None


def _duration_seconds(value: str) -> int | None:
    parts = value.strip().split(":")
    if not parts or not all(part.isdigit() for part in parts):
        return None
    numbers = [int(part) for part in parts]
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    if len(numbers) == 3:
        return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    return None


def discover_category_codes(page_html: str) -> list[str]:
    codes = re.findall(r"""href=["'][^"']*/?category(?:list)?/\?[^"']*\bc=([a-z]+)""", page_html, re.I)
    # Leaf codes are preferable; parent codes are kept if a page exposes no leaves.
    leaf_codes = [code.lower() for code in codes if len(code) >= 3]
    return list(dict.fromkeys(leaf_codes or (code.lower() for code in codes)))


def parse_catalogue_links(page_html: str, page_url: str = BOOK_BASE_URL) -> list[BookLink]:
    links: list[BookLink] = []
    for href in re.findall(r"""href=["']([^"']+)["']""", page_html, re.I):
        absolute = urllib.parse.urljoin(page_url, html.unescape(href))
        path = urllib.parse.urlparse(absolute).path.lower()
        if path.endswith("/detailsalbum/"):
            book_id = _query_id(absolute, "g")
            if book_id:
                links.append(BookLink(book_id, normalized_url(absolute)))
        elif path.endswith("/serialhome/"):
            serial_id = _query_id(absolute, "p")
            if serial_id:
                links.append(BookLink(f"serial:{serial_id}", normalized_url(absolute)))
    return list({(link.id, link.url): link for link in links}.values())


def parse_serial_page(page_html: str, serial_url: str) -> list[BookLink]:
    serial_id = _query_id(serial_url, "p")
    result: list[BookLink] = []
    for link in parse_catalogue_links(page_html, serial_url):
        if not link.id.startswith("serial:"):
            result.append(replace(link, serial_parent_id=serial_id, serial_parent_url=serial_url))
    return result


def _field_values(page_html: str, labels: tuple[str, ...]) -> tuple[str, ...]:
    label_pattern = "|".join(re.escape(label) for label in labels)
    match = re.search(
        rf"<dd\b[^>]*>.*?<strong\b[^>]*>\s*(?:{label_pattern})\s*:?\s*</strong>(.*?)</dd>",
        page_html,
        re.I | re.S,
    )
    if not match:
        return ()
    anchors = [_strip_tags(value) for value in re.findall(r"<a\b[^>]*>(.*?)</a>", match.group(1), re.I | re.S)]
    values = [value for value in anchors if value]
    if not values:
        plain = _strip_tags(match.group(1))
        values = [part.strip() for part in re.split(r"[،,]", plain) if part.strip()]
    return tuple(dict.fromkeys(values))


def _meta_content(page_html: str, property_name: str) -> str:
    for tag in re.findall(r"<meta\b[^>]*>", page_html, re.I):
        attributes = {
            name.lower(): html.unescape(value)
            for name, _, value in re.findall(
                r"([^\s=/>]+)\s*=\s*([\"'])(.*?)\2",
                tag,
                re.S,
            )
        }
        if attributes.get("property", "").lower() == property_name.lower():
            return normalize_space(attributes.get("content", ""))
    return ""


def _book_title(page_html: str) -> str:
    title_match = re.search(
        r"<h1\b(?=[^>]*\bitemprop=[\"']name[\"'])[^>]*>(.*?)</h1>",
        page_html,
        re.I | re.S,
    )
    if title_match:
        title = _strip_tags(title_match.group(1))
        if title:
            return title
    title = _meta_content(page_html, "og:title")
    if title:
        return title
    document_title = re.search(r"<title\b[^>]*>(.*?)</title>", page_html, re.I | re.S)
    return _strip_tags(document_title.group(1)) if document_title else ""


def parse_book_page(page_html: str, link: BookLink) -> Book:
    title = _book_title(page_html)
    if not title:
        raise ScrapeError("missing_book_title")
    total_duration: str | None = None
    for item in re.findall(r"<li\b[^>]*>(.*?)</li>", page_html, re.I | re.S):
        if "مدت کتاب" not in _strip_tags(item):
            continue
        duration_match = re.search(r"<strong\b[^>]*>(.*?)</strong>", item, re.I | re.S)
        if duration_match:
            total_duration = _strip_tags(duration_match.group(1))
            break
    tracks: list[Track] = []
    for item in re.findall(r"<li\b[^>]*>(.*?)</li>", page_html, re.I | re.S):
        player_match = re.search(
            r"""href=["']([^"']*/book-player/\?[^"']*\battid=(\d+)[^"']*)["']""",
            item,
            re.I,
        )
        if not player_match:
            continue
        player_url = normalized_url(html.unescape(player_match.group(1)), base_url=link.url)
        attachment_id = player_match.group(2)
        title_match = re.search(r"<a\b[^>]*book-player/[^>]*>\s*<span\b[^>]*>(.*?)</span>", item, re.I | re.S)
        track_title = _strip_tags(title_match.group(1)) if title_match else f"attachment {attachment_id}"
        duration_match = re.search(r"""song-duration[^>]*>\s*<span[^>]*>\s*([\d:]+)""", item, re.I | re.S)
        duration = duration_match.group(1) if duration_match else ""
        modal_match = re.search(r"""ajaxModalLoad\(\s*["']([^"']+\battid=\d+[^"']*)["']""", item, re.I)
        if not modal_match:
            raise ScrapeError(f"missing_download_modal:{attachment_id}")
        modal_url = normalized_url(html.unescape(modal_match.group(1)), base_url=link.url)
        normalized_title = normalize_space(track_title).lower()
        tracks.append(
            Track(
                id=f"{link.id}:{attachment_id}",
                book_id=link.id,
                title=track_title,
                duration=duration,
                duration_seconds=_duration_seconds(duration),
                attachment_id=attachment_id,
                player_url=player_url,
                download_modal_url=modal_url,
                is_sample="نمونه" in normalized_title or "sample" in normalized_title,
                is_full_book=("کل کتاب" in normalized_title or "کامل" in normalized_title or "full book" in normalized_title),
            )
        )
    if not tracks:
        raise ScrapeError("missing_book_tracks")
    return Book(
        id=link.id,
        title=title,
        authors=_field_values(page_html, ("نویسنده", "مولف", "مؤلف")),
        narrators=_field_values(page_html, ("راوی", "گوینده")),
        description=_meta_content(page_html, "og:description"),
        categories=_field_values(page_html, ("دسته‌بندی", "طبقه‌بندی", "موضوع")),
        total_duration=total_duration,
        source_url=link.url,
        serial_parent_id=link.serial_parent_id,
        serial_parent_url=link.serial_parent_url,
        tracks=tuple(tracks),
    )


def parse_download_modal(page_html: str, modal_url: str) -> tuple[str | None, str | None]:
    for match in re.finditer(r"""<a\b[^>]*href=["']([^"']+)["'][^>]*>(.*?)</a>""", page_html, re.I | re.S):
        label = _strip_tags(match.group(2))
        if "mp3" not in label.lower():
            continue
        url = normalized_url(html.unescape(match.group(1)), base_url=modal_url, audio=True)
        size_match = re.search(r"([\d.]+\s*(?:KB|MB|GB|کیلوبایت|مگابایت|گیگابایت))", label, re.I)
        return url, size_match.group(1) if size_match else None
    return None, None


def _category_url(code: str, page: int) -> str:
    route = "categorylist" if len(code) >= 3 else "category"
    return normalized_url(f"{BOOK_BASE_URL}{route}/?VALID=TRUE&c={urllib.parse.quote(code)}&pn={page}")


def _discover_book_links(
    client: PoliteHttpClient,
    *,
    book_ids: Iterable[str] | None,
    category_codes: Iterable[str] | None,
    max_pages: int | None,
    max_books: int | None,
) -> list[BookLink]:
    if book_ids:
        book_ids = tuple(book_ids)
        invalid = [book_id for book_id in book_ids if not re.fullmatch(r"\d+", book_id)]
        if invalid:
            raise ValueError("book_ids must contain numeric IranSeda IDs")
        return [
            BookLink(book_id, normalized_url(f"{BOOK_BASE_URL}DetailsAlbum/?VALID=TRUE&g={book_id}"))
            for book_id in dict.fromkeys(book_ids)
        ]
    print("[audiobooks] discovering catalogue categories", flush=True)
    codes = list(dict.fromkeys(category_codes or discover_category_codes(client.get_text(BOOK_BASE_URL))))
    print(f"[audiobooks] discovered {len(codes)} categories", flush=True)
    links: dict[str, BookLink] = {}
    for category_index, code in enumerate(codes, start=1):
        if not re.fullmatch(r"[a-z]+", code, re.I):
            raise ValueError("category_codes must contain only ASCII letters")
        page = 1
        page_signatures: set[tuple[tuple[str, str], ...]] = set()
        while max_pages is None or page <= max_pages:
            print(
                f"[audiobooks] category {category_index}/{len(codes)} code={code} page={page}",
                flush=True,
            )
            page_links = parse_catalogue_links(client.get_text(_category_url(code, page)), _category_url(code, page))
            signature = tuple(sorted((link.id, link.url) for link in page_links))
            if not page_links or signature in page_signatures:
                break
            page_signatures.add(signature)
            for link in page_links:
                if link.id not in links:
                    links[link.id] = link
            if max_books is not None and sum(not item.id.startswith("serial:") for item in links.values()) >= max_books:
                break
            page += 1
        if max_books is not None and sum(not item.id.startswith("serial:") for item in links.values()) >= max_books:
            break
    expanded: dict[str, BookLink] = {}
    for link in links.values():
        if link.id.startswith("serial:"):
            print(f"[audiobooks] expanding {link.id}", flush=True)
            for child in parse_serial_page(client.get_text(link.url), link.url):
                expanded.setdefault(child.id, child)
        else:
            expanded.setdefault(link.id, link)
    return list(expanded.values())


def scrape_audiobooks(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    book_ids: Iterable[str] | None = None,
    category_codes: Iterable[str] | None = None,
    max_pages: int | None = None,
    max_books: int | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
    refresh: bool = False,
    client: PoliteHttpClient | None = None,
    retrieved_at: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> ScrapeAudit:
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be greater than zero")
    if max_books is not None and max_books <= 0:
        raise ValueError("max_books must be greater than zero")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be greater than zero")
    output_root.mkdir(parents=True, exist_ok=True)
    existing_books = read_jsonl_by_id(output_root / "books.jsonl")
    existing_tracks = read_jsonl_by_id(output_root / "tracks.jsonl")
    discovery_checkpoints = read_jsonl_by_id(output_root / "discovery_checkpoints.jsonl")
    book_records: dict[str, dict[str, Any]] = dict(existing_books)
    track_records: dict[str, dict[str, Any]] = dict(existing_tracks)
    skipped: list[dict[str, Any]] = []
    books_seen = tracks_seen = resumed = 0
    owns_client = client is None
    client = client or PoliteHttpClient()

    def checkpoint(label: str) -> None:
        write_jsonl_atomic(
            output_root / "books.jsonl",
            sorted(book_records.values(), key=lambda item: item["id"]),
        )
        write_jsonl_atomic(
            output_root / "tracks.jsonl",
            sorted(track_records.values(), key=lambda item: item["id"]),
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
            f"[audiobooks] checkpoint {label}: books={len(book_records)} "
            f"tracks={len(track_records)} skipped={len(skipped)}",
            flush=True,
        )

    try:
        links = _discover_book_links(
            client,
            book_ids=book_ids,
            category_codes=category_codes,
            max_pages=max_pages,
            max_books=max_books,
        )
        if max_books is not None:
            links = links[:max_books]
        print(f"[audiobooks] processing {len(links)} book detail pages", flush=True)
        for index, link in enumerate(links, start=1):
            legacy_book = existing_books.get(link.id)
            legacy_tracks = legacy_book.get("tracks", []) if legacy_book else []
            legacy_complete = bool(legacy_book) and isinstance(legacy_tracks, list) and all(
                isinstance(track_id, str) and track_id in existing_tracks for track_id in legacy_tracks
            )
            checkpoint_record = discovery_checkpoints.get(link.id, {})
            checkpoint_complete = checkpoint_record.get("status") == "discovered"
            if not refresh and (checkpoint_complete or legacy_complete):
                resumed += 1
                print(
                    f"[audiobooks] resume skip {index}/{len(links)} id={link.id} already processed",
                    flush=True,
                )
                discovery_checkpoints.setdefault(
                    link.id,
                    {
                        "id": link.id,
                        "status": "discovered",
                        "completed_at": retrieved_at().astimezone(timezone.utc).isoformat(),
                        "inferred_from_legacy_manifest": True,
                    },
                )
                if index % checkpoint_every == 0:
                    checkpoint(f"book {index}/{len(links)}")
                continue
            print(f"[audiobooks] book {index}/{len(links)} id={link.id}", flush=True)
            page_html = client.get_text(link.url)
            try:
                book = parse_book_page(page_html, link)
            except ScrapeError as error:
                skipped.append({"id": link.id, "source_url": link.url, "reason": str(error)})
                discovery_checkpoints[link.id] = {
                    "id": link.id,
                    "status": "skipped",
                    "reason": str(error),
                    "completed_at": retrieved_at().astimezone(timezone.utc).isoformat(),
                }
                print(f"[audiobooks] skipped id={link.id}: {error}", flush=True)
                if index % checkpoint_every == 0:
                    checkpoint(f"book {index}/{len(links)}")
                continue
            enriched_tracks: list[Track] = []
            for track in book.tracks:
                try:
                    mp3_url, declared_size = parse_download_modal(
                        client.get_text(track.download_modal_url), track.download_modal_url
                    )
                    enriched_tracks.append(replace(track, mp3_url=mp3_url, declared_size=declared_size))
                    if mp3_url is None:
                        skipped.append(
                            {"id": track.id, "source_url": track.download_modal_url, "reason": "missing_explicit_mp3"}
                        )
                except ScrapeError as error:
                    enriched_tracks.append(track)
                    skipped.append({"id": track.id, "source_url": track.download_modal_url, "reason": str(error)})
            book = replace(book, tracks=tuple(enriched_tracks))
            books_seen += 1
            tracks_seen += len(book.tracks)
            timestamp = retrieved_at().astimezone(timezone.utc).isoformat()
            book_records[book.id] = {
                **existing_books.get(book.id, {}),
                **asdict(book),
                "tracks": [track.id for track in book.tracks],
                "retrieved_at": timestamp,
            }
            for track in book.tracks:
                record: dict[str, Any] = {
                    **existing_tracks.get(track.id, {}),
                    **asdict(track),
                    "retrieved_at": timestamp,
                }
                track_records[track.id] = record
            discovery_checkpoints[link.id] = {
                "id": link.id,
                "status": "discovered",
                "tracks": len(book.tracks),
                "completed_at": timestamp,
            }
            print(
                f"[audiobooks] discovered id={book.id} title={book.title!r} "
                f"tracks={len(book.tracks)}",
                flush=True,
            )
            if index % checkpoint_every == 0:
                checkpoint(f"book {index}/{len(links)}")
    except BaseException:
        checkpoint("interrupted")
        raise
    finally:
        if owns_client:
            client.close()
    checkpoint("complete")
    return ScrapeAudit(books_seen, tracks_seen, resumed, len(skipped))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Discover public IranSeda audiobook metadata and explicit MP3 links without downloading audio."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help=f"Output directory (default: {DEFAULT_OUTPUT_ROOT}).")
    parser.add_argument("--book-id", action="append", dest="book_ids", help="Book/album ID to inspect; repeat to override category discovery.")
    parser.add_argument("--category-code", action="append", dest="category_codes", help="IranSeda category code to crawl; repeat for multiple categories.")
    parser.add_argument("--max-pages", type=int, help="Maximum category pages per category (default: all exposed pages).")
    parser.add_argument("--max-books", type=int, help="Maximum unique detail pages to inspect (default: all discovered books).")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help=f"Write manifests after this many processed books (default: {DEFAULT_CHECKPOINT_EVERY}).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Revisit books already recorded in discovery checkpoints.",
    )
    parser.add_argument("--delay-seconds", type=float, default=DEFAULT_DELAY_SECONDS, help="Minimum per-origin delay in seconds (default: 1.0).")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="Per-request timeout in seconds (default: 30.0).")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help=f"HTTP User-Agent (default: {DEFAULT_USER_AGENT}).")
    args = parser.parse_args(argv)
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
        audit = scrape_audiobooks(
            args.output_root,
            book_ids=args.book_ids,
            category_codes=args.category_codes,
            max_pages=args.max_pages,
            max_books=args.max_books,
            checkpoint_every=args.checkpoint_every,
            refresh=args.refresh,
            client=client,
        )
    print("IranSeda audiobook scrape summary")
    for key, value in asdict(audit).items():
        print(f"  {key.replace('_', ' ')}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
