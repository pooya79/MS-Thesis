from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from ml.speech_data.data_scraping import iranseda_audiobooks as books
from ml.speech_data.data_scraping import iranseda_common as common
from ml.speech_data.data_scraping import iranseda_download as downloader
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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


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


def test_audiobook_title_falls_back_to_order_independent_open_graph_metadata() -> None:
    page = book_page().replace(
        '<h1 itemprop="name">کتاب آزمایشی</h1>',
        '<meta content="عنوان از فراداده" data-source="catalogue" property="og:title">',
    )

    parsed = books.parse_book_page(page, books.BookLink(BOOK_ID, BOOK_URL))

    assert parsed.title == "عنوان از فراداده"


def test_audiobook_title_falls_back_to_document_title() -> None:
    page = book_page().replace(
        '<h1 itemprop="name">کتاب آزمایشی</h1>',
        "<title>عنوان صفحه</title>",
    )

    parsed = books.parse_book_page(page, books.BookLink(BOOK_ID, BOOK_URL))

    assert parsed.title == "عنوان صفحه"


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


def test_audiobook_metadata_only_reads_modals_but_never_audio(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = FakeClient(
        {
            BOOK_URL: book_page(),
            SAMPLE_MODAL: modal("100"),
            FULL_MODAL: modal("101"),
            CHAPTER_MODAL: modal("102"),
        }
    )
    audit = books.scrape_audiobooks(
        tmp_path,
        book_ids=[BOOK_ID],
        checkpoint_every=1,
        client=client,  # type: ignore[arg-type]
    )

    assert audit.books == 1
    assert audit.tracks == 3
    assert client.audio_requests == []
    assert len(jsonl(tmp_path / "books.jsonl")) == 1
    assert all("path" not in track for track in jsonl(tmp_path / "tracks.jsonl"))
    assert jsonl(tmp_path / "discovery_checkpoints.jsonl")[0]["status"] == "discovered"
    assert not (tmp_path / "train.tsv").exists()
    output = capsys.readouterr().out
    assert "[audiobooks] book 1/1" in output
    assert "[audiobooks] checkpoint book 1/1" in output

    resumed_client = FakeClient({})
    resumed = books.scrape_audiobooks(
        tmp_path,
        book_ids=[BOOK_ID],
        client=resumed_client,  # type: ignore[arg-type]
    )
    assert resumed.resumed == 1
    assert resumed_client.text_requests == []


def test_audiobook_interruption_checkpoints_completed_books(tmp_path: Path) -> None:
    interrupted_url = "http://book.iranseda.ir/DetailsAlbum/?VALID=TRUE&g=2"

    class InterruptingClient(FakeClient):
        def get_text(self, url: str) -> str:
            if url == interrupted_url:
                raise KeyboardInterrupt
            return super().get_text(url)

    client = InterruptingClient(
        {
            BOOK_URL: book_page(),
            SAMPLE_MODAL: modal("100"),
            FULL_MODAL: modal("101"),
            CHAPTER_MODAL: modal("102"),
        }
    )

    with pytest.raises(KeyboardInterrupt):
        books.scrape_audiobooks(
            tmp_path,
            book_ids=[BOOK_ID, "2"],
            checkpoint_every=10,
            client=client,  # type: ignore[arg-type]
        )

    assert [record["id"] for record in jsonl(tmp_path / "books.jsonl")] == [BOOK_ID]
    assert len(jsonl(tmp_path / "tracks.jsonl")) == 3


def test_audiobook_retries_skipped_checkpoint(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "discovery_checkpoints.jsonl",
        [{"id": BOOK_ID, "status": "skipped", "reason": "missing_book_title"}],
    )
    client = FakeClient(
        {
            BOOK_URL: book_page(),
            SAMPLE_MODAL: modal("100"),
            FULL_MODAL: modal("101"),
            CHAPTER_MODAL: modal("102"),
        }
    )

    audit = books.scrape_audiobooks(
        tmp_path,
        book_ids=[BOOK_ID],
        client=client,  # type: ignore[arg-type]
    )

    assert audit.books == 1
    assert audit.resumed == 0
    assert jsonl(tmp_path / "discovery_checkpoints.jsonl")[0]["status"] == "discovered"


def test_audiobook_downloader_prefers_full_book_and_reuses_checksum(tmp_path: Path) -> None:
    payload = b"ID3-full-book"
    texts = {
        BOOK_URL: book_page(),
        SAMPLE_MODAL: modal("100"),
        FULL_MODAL: modal("101"),
        CHAPTER_MODAL: modal("102"),
    }
    discovery_client = FakeClient(texts)
    books.scrape_audiobooks(tmp_path, book_ids=[BOOK_ID], client=discovery_client)  # type: ignore[arg-type]
    tracks_before = jsonl(tmp_path / "tracks.jsonl")

    client = FakeClient({}, {FULL_MP3: (payload, "audio/mpeg")})
    audit = downloader.download_discovered(tmp_path, client=client)  # type: ignore[arg-type]

    assert audit.downloaded == 1
    assert audit.failed == 0
    assert client.audio_requests == [FULL_MP3]
    clip = tmp_path / "clips" / BOOK_ID / "101.mp3"
    assert clip.read_bytes() == payload
    assert jsonl(tmp_path / "tracks.jsonl") == tracks_before
    record = jsonl(tmp_path / "downloads.jsonl")[0]
    assert record["checksum"] == f"sha256:{hashlib.sha256(payload).hexdigest()}"

    second = FakeClient({})
    audit = downloader.download_discovered(tmp_path, client=second)  # type: ignore[arg-type]
    assert audit.reused == 1
    assert second.audio_requests == []


def test_audiobook_downloader_falls_back_to_non_sample_chapters(tmp_path: Path) -> None:
    sample_url = "http://player.iranseda.ir/downloadfile?attid=100"
    chapter_url = "http://player.iranseda.ir/downloadfile?attid=102"
    write_jsonl(
        tmp_path / "tracks.jsonl",
        [
            {"id": "7:100", "book_id": "7", "attachment_id": "100", "is_sample": True, "is_full_book": False, "mp3_url": sample_url},
            {"id": "7:102", "book_id": "7", "attachment_id": "102", "is_sample": False, "is_full_book": False, "mp3_url": chapter_url},
        ],
    )
    client = FakeClient({}, {chapter_url: (b"chapter", "audio/mpeg")})

    audit = downloader.download_discovered(tmp_path, client=client)  # type: ignore[arg-type]

    assert audit.selected == 1
    assert client.audio_requests == [chapter_url]
    assert (tmp_path / "clips" / "7" / "102.mp3").read_bytes() == b"chapter"


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


def test_radio_metadata_only_keeps_eligible_episode_without_audio_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
    assert jsonl(tmp_path / "discovery_checkpoints.jsonl")[0]["id"] == "11:2026-07-01"
    assert not (tmp_path / "train.tsv").exists()
    output = capsys.readouterr().out
    assert "[radio] fetching station=11 date=2026-07-01" in output
    assert "[radio] checkpoint station=11 date=2026-07-01" in output

    resumed_client = FakeClient({station_url: station_list()})
    resumed = radio.scrape_radio(
        date(2026, 7, 1),
        date(2026, 7, 1),
        tmp_path,
        channel_ids=["11"],
        client=resumed_client,  # type: ignore[arg-type]
        now=lambda: datetime(2026, 7, 27, 12, tzinfo=radio.TEHRAN),
    )
    assert resumed.resumed == 1
    assert resumed_client.text_requests == [station_url]


def test_radio_downloader_selects_only_eligible_and_reports_missing_mp3(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "episodes.jsonl",
        [
            {"id": "700", "station_id": "11", "date": "2026-07-01", "eligible": True, "mp3_url": RADIO_MP3},
            {"id": "701", "station_id": "11", "date": "2026-07-01", "eligible": False, "mp3_url": "http://headend2.iranseda.ir/DLFile/?vid=301"},
            {"id": "702", "station_id": "11", "date": "2026-07-01", "eligible": True, "mp3_url": None},
        ],
    )
    client = FakeClient({}, {RADIO_MP3: (b"radio", "audio/mpeg")})

    audit = downloader.download_discovered(tmp_path, client=client)  # type: ignore[arg-type]

    assert audit.selected == 2
    assert audit.downloaded == 1
    assert audit.failed == 1
    assert client.audio_requests == [RADIO_MP3]
    assert jsonl(tmp_path / "download_skipped.jsonl")[0]["reason"] == "missing_explicit_mp3"


def test_downloader_rejects_missing_or_ambiguous_source_root(tmp_path: Path) -> None:
    with pytest.raises(common.ScrapeError, match="exactly one"):
        downloader.detect_source(tmp_path)

    write_jsonl(tmp_path / "tracks.jsonl", [])
    write_jsonl(tmp_path / "episodes.jsonl", [])
    with pytest.raises(common.ScrapeError, match="exactly one"):
        downloader.detect_source(tmp_path)


def test_downloader_continues_after_failure_and_checkpoints_success(tmp_path: Path) -> None:
    bad_url = "http://player.iranseda.ir/downloadfile?attid=201"
    good_url = "http://player.iranseda.ir/downloadfile?attid=202"
    write_jsonl(
        tmp_path / "tracks.jsonl",
        [
            {"id": "8:201", "book_id": "8", "attachment_id": "201", "is_sample": False, "is_full_book": False, "mp3_url": bad_url},
            {"id": "8:202", "book_id": "8", "attachment_id": "202", "is_sample": False, "is_full_book": False, "mp3_url": good_url},
        ],
    )
    client = FakeClient(
        {},
        {
            bad_url: (b"not audio", "text/html"),
            good_url: (b"good", "audio/mpeg"),
        },
    )

    audit = downloader.download_discovered(tmp_path, client=client)  # type: ignore[arg-type]

    assert audit.downloaded == 1
    assert audit.failed == 1
    assert client.audio_requests == [bad_url, good_url]
    assert jsonl(tmp_path / "downloads.jsonl")[0]["id"] == "8:202"
    assert jsonl(tmp_path / "download_skipped.jsonl")[0]["id"] == "8:201"


def test_downloader_imports_matching_legacy_checksum(tmp_path: Path) -> None:
    payload = b"legacy"
    checksum = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    relative = "clips/9/301.mp3"
    write_jsonl(
        tmp_path / "tracks.jsonl",
        [
            {
                "id": "9:301",
                "book_id": "9",
                "attachment_id": "301",
                "is_sample": False,
                "is_full_book": True,
                "mp3_url": "http://player.iranseda.ir/downloadfile?attid=301",
                "path": relative,
                "checksum": checksum,
            }
        ],
    )
    clip = tmp_path / relative
    clip.parent.mkdir(parents=True)
    clip.write_bytes(payload)
    client = FakeClient({})

    audit = downloader.download_discovered(tmp_path, client=client)  # type: ignore[arg-type]

    assert audit.reused == 1
    assert client.audio_requests == []
    assert jsonl(tmp_path / "downloads.jsonl")[0]["checksum"] == checksum


def test_downloader_requires_force_for_untracked_existing_file(tmp_path: Path) -> None:
    url = "http://player.iranseda.ir/downloadfile?attid=401"
    write_jsonl(
        tmp_path / "tracks.jsonl",
        [
            {
                "id": "10:401",
                "book_id": "10",
                "attachment_id": "401",
                "is_sample": False,
                "is_full_book": True,
                "mp3_url": url,
            }
        ],
    )
    clip = tmp_path / "clips" / "10" / "401.mp3"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"untracked")

    failed = downloader.download_discovered(tmp_path, client=FakeClient({}))  # type: ignore[arg-type]
    assert failed.failed == 1
    assert clip.read_bytes() == b"untracked"

    client = FakeClient({}, {url: (b"replacement", "audio/mpeg")})
    replaced = downloader.download_discovered(tmp_path, force=True, client=client)  # type: ignore[arg-type]
    assert replaced.downloaded == 1
    assert replaced.failed == 0
    assert clip.read_bytes() == b"replacement"


def test_downloader_rejects_malformed_manifest(tmp_path: Path) -> None:
    (tmp_path / "tracks.jsonl").write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(common.ScrapeError, match="invalid_json:tracks.jsonl:1"):
        downloader.download_discovered(tmp_path, client=FakeClient({}))  # type: ignore[arg-type]


def test_downloader_checkpoints_before_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_jsonl(
        tmp_path / "tracks.jsonl",
        [
            {
                "id": f"11:{attachment_id}",
                "book_id": "11",
                "attachment_id": attachment_id,
                "is_sample": False,
                "is_full_book": False,
                "mp3_url": f"http://player.iranseda.ir/downloadfile?attid={attachment_id}",
            }
            for attachment_id in ("501", "502")
        ],
    )
    calls = 0

    def interrupted_download(*args: object, **kwargs: object) -> common.DownloadResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return common.DownloadResult(5, "sha256:first", "audio/mpeg", False)

    monkeypatch.setattr(downloader, "download_audio", interrupted_download)

    with pytest.raises(KeyboardInterrupt):
        downloader.download_discovered(tmp_path, client=FakeClient({}))  # type: ignore[arg-type]

    assert [record["id"] for record in jsonl(tmp_path / "downloads.jsonl")] == ["11:501"]


def test_downloader_cli_returns_nonzero_for_partial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        downloader,
        "download_discovered",
        lambda *args, **kwargs: downloader.DownloadAudit(1, 0, 0, 1, 0),
    )

    assert downloader.main(["--source-root", str(tmp_path)]) == 1


def test_discovery_cli_help_has_no_download_option(capsys: pytest.CaptureFixture[str]) -> None:
    for module in (books, radio):
        with pytest.raises(SystemExit) as exc_info:
            module.main(["--help"])
        assert exc_info.value.code == 0
        assert "--download" not in capsys.readouterr().out


def test_common_client_allows_resources_when_robots_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404, request=request)
        return httpx.Response(200, text="ok", request=request)

    with common.PoliteHttpClient(delay_seconds=0, transport=httpx.MockTransport(handler)) as client:
        assert client.get_text("http://book.iranseda.ir/DetailsAlbum/?g=1") == "ok"


@pytest.mark.parametrize("status", [429, 500, 503])
def test_common_client_fails_closed_on_unreachable_robots(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, request=request)

    with common.PoliteHttpClient(
        delay_seconds=0,
        max_retries=0,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(common.ScrapeError, match="could not verify robots"):
            client.get_text("http://book.iranseda.ir/DetailsAlbum/?g=1")


def test_common_client_fails_closed_on_invalid_successful_robots() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not a robots policy", request=request)

    with common.PoliteHttpClient(delay_seconds=0, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(common.ScrapeError, match="invalid policy"):
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


def test_common_client_retries_truncated_successful_html() -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        if request.url.path == "/robots.txt":
            return httpx.Response(404, request=request)
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                200,
                text="<!DOCTYPE html>\r\n",
                headers={"Content-Type": "text/html"},
                request=request,
            )
        return httpx.Response(
            200,
            text="<!DOCTYPE html><html><title>book</title></html>",
            headers={"Content-Type": "text/html"},
            request=request,
        )

    with common.PoliteHttpClient(
        delay_seconds=0,
        transport=httpx.MockTransport(handler),
        sleeper=sleeps.append,
    ) as client:
        page = client.get_text("http://book.iranseda.ir/DetailsAlbum/?g=1")

    assert "<title>book</title>" in page
    assert attempts == 2
    assert sleeps == [common.INCOMPLETE_HTML_RETRY_SECONDS]


def test_common_client_rejects_persistently_truncated_successful_html() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404, request=request)
        return httpx.Response(
            200,
            text="<!DOCTYPE html>\r\n",
            headers={"Content-Type": "text/html"},
            request=request,
        )

    with common.PoliteHttpClient(
        delay_seconds=0,
        max_retries=1,
        transport=httpx.MockTransport(handler),
        sleeper=lambda _: None,
    ) as client:
        with pytest.raises(common.TransientResponseError, match="incomplete HTML"):
            client.get_text("http://book.iranseda.ir/DetailsAlbum/?g=1")


def test_download_rejects_invalid_mime_and_removes_partial(tmp_path: Path) -> None:
    client = FakeClient({}, {FULL_MP3: (b"<html>blocked</html>", "text/html")})
    output = tmp_path / "clip.mp3"

    with pytest.raises(common.ScrapeError, match="content_type"):
        common.download_audio(client, url=FULL_MP3, output_path=output)  # type: ignore[arg-type]

    assert not output.exists()
    assert not (tmp_path / "clip.mp3.part").exists()
