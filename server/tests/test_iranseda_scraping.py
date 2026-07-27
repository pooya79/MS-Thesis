from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from ml.speech_data.data_scraping import iranseda_audiobooks as books
from ml.speech_data.data_scraping import iranseda_common as common
from ml.speech_data.data_scraping import iranseda_radio as radio


BOOK_ID = "706538"
BOOK_URL = f"http://book.iranseda.ir/DetailsAlbum/?VALID=TRUE&g={BOOK_ID}"
SAMPLE_MODAL = "http://book.iranseda.ir/download?attid=100"
FULL_MODAL = "http://book.iranseda.ir/download?attid=101"
CHAPTER_MODAL = "http://book.iranseda.ir/download?attid=102"
FULL_MP3 = "http://player.iranseda.ir/downloadfile?attid=101&VALID=TRUE&q=11"


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def book_page() -> str:
    def track(attachment_id: str, title: str, duration: str) -> str:
        return (
            "<li>"
            f'<div class="album-detail"><a href="http://player.iranseda.ir/book-player/'
            f'?VALID=TRUE&g={BOOK_ID}&attid={attachment_id}&t=1&w=43"><span>{title}</span></a></div>'
            f'<div class="song-duration"><span>{duration}</span></div>'
            f"""<a onclick="ajaxModalLoad('../download?attid={attachment_id}')">download</a>"""
            "</li>"
        )

    return (
        '<meta property="og:description" content="شرح کامل کتاب">'
        '<h1 itemprop="name">کتاب آزمایشی</h1>'
        "<ul><li><span><strong>259 دقیقه</strong></span><span>مدت کتاب</span></li></ul>"
        '<dd class="field"><strong>نویسنده:</strong><a>نویسنده یک</a><a>نویسنده دو</a></dd>'
        '<dd class="field"><strong>راوی:</strong><a>راوی کتاب</a></dd>'
        '<dd class="field"><strong>دسته‌بندی:</strong><a>رمان</a></dd>'
        + track("100", "صدای نمونه کتاب", "2:46")
        + track("101", "صدای کل کتاب", "259:49")
        + track("102", "فصل اول", "32:49")
    )


def modal(attachment_id: str, *, mp3: bool = True) -> str:
    body = (
        f'<a href="http://player.iranseda.ir/downloadfile?attid={attachment_id}&VALID=TRUE&q=9">'
        "mp4 با حجم : 20 MB</a>"
    )
    if mp3:
        body += (
            f'<a href="http://player.iranseda.ir/downloadfile?attid={attachment_id}&VALID=TRUE&q=11">'
            "mp3 با حجم : 10.5 MB</a>"
        )
    return body


class FakeResponse:
    def __init__(self, url: str, payload: bytes, media_type: str = "audio/mpeg") -> None:
        self.url = httpx.URL(url)
        self.headers = {"Content-Type": media_type}
        self.payload = payload

    def iter_bytes(self, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        yield self.payload


class FakeClient:
    def __init__(self, texts: dict[str, str], audio: dict[str, tuple[bytes, str]] | None = None) -> None:
        self.texts = texts
        self.audio = audio or {}
        self.text_requests: list[str] = []
        self.audio_requests: list[str] = []

    def get_text(self, url: str) -> str:
        self.text_requests.append(url)
        if url not in self.texts:
            raise common.ScrapeError(f"unexpected URL: {url}")
        return self.texts[url]

    @contextmanager
    def stream(self, url: str) -> Iterator[FakeResponse]:
        self.audio_requests.append(url)
        payload, media_type = self.audio[url]
        yield FakeResponse(url, payload, media_type)

    def close(self) -> None:
        return None


def test_audiobook_parses_details_tracks_and_explicit_modal() -> None:
    parsed = books.parse_book_page(book_page(), books.BookLink(BOOK_ID, BOOK_URL))
    mp3_url, size = books.parse_download_modal(modal("101"), FULL_MODAL)

    assert parsed.title == "کتاب آزمایشی"
    assert parsed.authors == ("نویسنده یک", "نویسنده دو")
    assert parsed.narrators == ("راوی کتاب",)
    assert parsed.categories == ("رمان",)
    assert parsed.total_duration == "259 دقیقه"
    assert parsed.tracks[0].is_sample
    assert parsed.tracks[1].is_full_book
    assert parsed.tracks[2].duration_seconds == 1969
    assert mp3_url == FULL_MP3
    assert size == "10.5 MB"


def test_serial_expansion_and_catalogue_deduplication() -> None:
    serial_url = "http://book.iranseda.ir/SerialHome/?VALID=TRUE&p=55&g=60"
    page = (
        '<a href="../DetailsAlbum/?VALID=TRUE&g=10">one</a>'
        '<a href="../DetailsAlbum/?VALID=TRUE&g=10">duplicate</a>'
        '<a href="../DetailsAlbum/?VALID=TRUE&g=11">two</a>'
    )
    links = books.parse_serial_page(page, serial_url)

    assert [link.id for link in links] == ["10", "11"]
    assert all(link.serial_parent_id == "55" for link in links)


def test_audiobook_metadata_only_reads_modals_but_never_audio(tmp_path: Path) -> None:
    client = FakeClient(
        {
            BOOK_URL: book_page(),
            SAMPLE_MODAL: modal("100"),
            FULL_MODAL: modal("101"),
            CHAPTER_MODAL: modal("102"),
        }
    )
    audit = books.scrape_audiobooks(tmp_path, book_ids=[BOOK_ID], client=client)  # type: ignore[arg-type]

    assert audit.books == 1
    assert audit.tracks == 3
    assert audit.downloaded == 0
    assert client.audio_requests == []
    assert len(jsonl(tmp_path / "books.jsonl")) == 1
    assert all("path" not in track for track in jsonl(tmp_path / "tracks.jsonl"))
    assert not (tmp_path / "train.tsv").exists()


def test_audiobook_download_prefers_full_book_and_reuses_checksum(tmp_path: Path) -> None:
    payload = b"ID3-full-book"
    texts = {
        BOOK_URL: book_page(),
        SAMPLE_MODAL: modal("100"),
        FULL_MODAL: modal("101"),
        CHAPTER_MODAL: modal("102"),
    }
    client = FakeClient(texts, {FULL_MP3: (payload, "audio/mpeg")})
    audit = books.scrape_audiobooks(
        tmp_path, book_ids=[BOOK_ID], download=True, client=client  # type: ignore[arg-type]
    )

    assert audit.downloaded == 1
    assert client.audio_requests == [FULL_MP3]
    clip = tmp_path / "clips" / BOOK_ID / "101.mp3"
    assert clip.read_bytes() == payload
    record = next(item for item in jsonl(tmp_path / "tracks.jsonl") if item["attachment_id"] == "101")
    assert record["checksum"] == f"sha256:{hashlib.sha256(payload).hexdigest()}"

    second = FakeClient(texts)
    audit = books.scrape_audiobooks(
        tmp_path, book_ids=[BOOK_ID], download=True, client=second  # type: ignore[arg-type]
    )
    assert audit.reused == 1
    assert second.audio_requests == []


def test_audiobook_selection_falls_back_to_non_sample_chapters() -> None:
    parsed = books.parse_book_page(book_page(), books.BookLink(BOOK_ID, BOOK_URL))
    enriched = [
        replace(track, mp3_url=f"http://player.iranseda.ir/downloadfile?attid={track.attachment_id}")
        for track in parsed.tracks
        if not track.is_full_book
    ]

    assert [track.attachment_id for track in books.selected_download_tracks(enriched)] == ["102"]


def station_list() -> str:
    return (
        '<a href="../live/?VALID=TRUE&ch=11">رادیو ایران</a>'
        '<a href="../live/?VALID=TRUE&ch=21">رادیو آوا</a>'
        '<a href="../live/?VALID=TRUE&ch=50">رادیو نما - ایران</a>'
        '<a href="../live/?VALID=TRUE&ch=202">رادیو عربی</a>'
        '<a href="../live/?VALID=TRUE&ch=501">رادیو البرز</a>'
    )


def epg_page(title: str = "گفت و گوی روز") -> str:
    return (
        '<a href="../epgarchivePart/?VALID=TRUE&ch=11&e=700">'
        f'<h4 itemprop="name">{title}</h4>'
        '<div>زمان شروع <span>9:05</span> مدت <span>30 دقیقه</span></div>'
        """<a onclick="openPlayer('http://player.iranseda.ir/epg-player/?VALID=TRUE&e=700')"></a>"""
    )


ARCHIVE_URL = "http://radio.iranseda.ir/epgarchivePart/?VALID=TRUE&ch=11&e=700"
PROGRAM_URL = "http://radio.iranseda.ir/Program/?VALID=TRUE&ch=11&m=123"
RADIO_MP3 = "http://headend2.iranseda.ir/DLFile/?VALID=TRUE&vid=300_1"


def archive_page(*, title: str = "گفت و گوی روز") -> str:
    return (
        f'<h1 itemprop="name">{title}</h1>'
        '<p itemprop="description">گفت‌وگو با پژوهشگران درباره اقتصاد</p>'
        f'<a href="{PROGRAM_URL}">نمایش تمام قسمت ها</a>'
        f'<a href="{RADIO_MP3}"><span>دانلود از سرور 13 (.mp3)</span></a>'
    )


def program_page() -> str:
    return (
        '<h1 itemprop="name">گفت و گوی روز</h1>'
        '<p itemprop="description">یک برنامه تحلیلی و خبری</p>'
    )


def test_radio_station_defaults_and_overrides() -> None:
    defaults = radio.parse_station_list(station_list())
    by_id = {station.id: station for station in defaults}

    assert by_id["11"].included
    assert by_id["501"].included
    assert not by_id["21"].included
    assert not by_id["202"].included
    override = radio.parse_station_list(station_list(), overrides=["21"])
    assert override[0].included
    assert override[0].classification_reason == "included_override"

    spoken_station = radio.parse_station_list(
        station_list() + '<a href="../live/?VALID=TRUE&ch=22">رادیو نمایش</a>'
    )
    assert next(station for station in spoken_station if station.id == "22").included


def test_radio_dates_epg_archive_and_broad_classification() -> None:
    start, end = radio.validate_date_range("2026-07-01", "2026-07-07", today=date(2026, 7, 27))
    entry = radio.parse_epg_page(epg_page(), station_id="11", day=start)[0]
    archive = radio.parse_archive_page(archive_page(), entry)

    assert (start, end) == (date(2026, 7, 1), date(2026, 7, 7))
    assert entry.start_time == "9:05"
    assert entry.duration_minutes == 30
    assert archive.program_id == "123"
    assert archive.mp3_url == RADIO_MP3
    assert radio.classify_episode("مصاحبه خبری")[0]
    assert radio.classify_episode("میزگرد علمی")[0]
    assert not radio.classify_episode("گلچین موسیقی")[0]
    assert not radio.classify_episode("پیام بازرگانی")[0]
    assert not radio.classify_episode("میان برنامه")[0]


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        ("2026/07/01", "2026-07-02", "Gregorian ISO"),
        ("2026-07-03", "2026-07-02", "after end"),
        ("2026-07-01", "2026-07-28", "future"),
    ],
)
def test_radio_rejects_invalid_date_ranges(start: str, end: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        radio.validate_date_range(start, end, today=date(2026, 7, 27))


def test_radio_metadata_only_keeps_eligible_episode_without_audio_request(tmp_path: Path) -> None:
    station_url = "http://radio.iranseda.ir/radiolist/?VALID=TRUE"
    epg_url = "http://radio.iranseda.ir/epglist/?VALID=TRUE&ch=11&d=7/1/2026"
    client = FakeClient(
        {
            station_url: station_list(),
            epg_url: epg_page(),
            ARCHIVE_URL: archive_page(),
            PROGRAM_URL: program_page(),
        }
    )
    audit = radio.scrape_radio(
        date(2026, 7, 1),
        date(2026, 7, 1),
        tmp_path,
        channel_ids=["11"],
        client=client,  # type: ignore[arg-type]
        now=lambda: datetime(2026, 7, 27, 12, tzinfo=radio.TEHRAN),
    )

    assert audit.episodes == 1
    assert audit.eligible == 1
    assert client.audio_requests == []
    assert jsonl(tmp_path / "episodes.jsonl")[0]["mp3_url"] == RADIO_MP3
    assert not (tmp_path / "train.tsv").exists()


def test_common_client_fails_closed_on_unverifiable_robots() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request)

    with common.PoliteHttpClient(delay_seconds=0, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(common.ScrapeError, match="could not verify robots"):
            client.get_text("http://book.iranseda.ir/DetailsAlbum/?g=1")


def test_common_client_rejects_cross_host_redirect() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n", request=request)
        return httpx.Response(302, headers={"Location": "http://evil.example/audio.mp3"}, request=request)

    with common.PoliteHttpClient(delay_seconds=0, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(common.ScrapeError, match="not an allowed"):
            client.get_text("http://book.iranseda.ir/DetailsAlbum/?g=1")


def test_common_client_honors_retry_after() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n", request=request)
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, request=request)
        return httpx.Response(200, text="ok", request=request)

    with common.PoliteHttpClient(
        delay_seconds=0,
        transport=httpx.MockTransport(handler),
        sleeper=sleeps.append,
    ) as client:
        assert client.get_text("http://book.iranseda.ir/DetailsAlbum/?g=1") == "ok"

    assert attempts == 2
    assert 2.0 in sleeps


def test_download_rejects_invalid_mime_and_removes_partial(tmp_path: Path) -> None:
    client = FakeClient({}, {FULL_MP3: (b"<html>blocked</html>", "text/html")})
    output = tmp_path / "clip.mp3"

    with pytest.raises(common.ScrapeError, match="content_type"):
        common.download_audio(client, url=FULL_MP3, output_path=output)  # type: ignore[arg-type]

    assert not output.exists()
    assert not (tmp_path / "clip.mp3.part").exists()
