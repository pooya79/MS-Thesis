from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from ml.speech_data.data_scraping import voa_persian as scraper


FEED_URL = "https://ir.voanews.com/api/test-audio-feed"
SOURCE_URL = "https://ir.voanews.com/a/example/8171694.html"
AUDIO_URL = "https://voa-audio-ns.akamaized.net/vpe/episode_hq.mp3"


def rss_payload(*items: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<rss><channel>"
        "<title>پادکست صوتی آزمایشی - صدای آمریکا</title>"
        "<language>fa</language>"
        "<generator>Pangea CMS – VOA</generator>"
        f"{''.join(items)}"
        "</channel></rss>"
    )


def rss_item(
    *,
    title: str = "برنامه آزمایشی",
    source_url: str = SOURCE_URL,
    published: str = "Tue, 14 Jul 2026 19:00:00 +0330",
) -> str:
    return (
        "<item>"
        f"<title>{title}</title>"
        f"<link>{source_url}</link>"
        f"<guid>{source_url}</guid>"
        f"<pubDate>{published}</pubDate>"
        '<enclosure url="https://gdb.voanews.com/cover.jpg" type="image/jpeg"/>'
        "</item>"
    )


def audio_page(*, copied: str = "no", entity: str = "VOA", content_type: str = "audio") -> str:
    sources = (
        "[{&quot;Src&quot;:&quot;https://voa-audio-ns.akamaized.net/vpe/episode_hq.mp3&quot;,"
        "&quot;Type&quot;:&quot;audio/mp3&quot;,&quot;DataInfo&quot;:&quot;128 kbps&quot;}]"
    )
    return (
        "<html><head>"
        '<meta name="twitter:player:stream" '
        'content="https://voa-audio-ns.akamaized.net/vpe/episode.mp3">'
        "</head><body>"
        f'<script>embedProperties, {{entity:"{entity}",copied:"{copied}",'
        f'language_service:"VOA Persian",content_type:"{content_type}"}});</script>'
        f'<audio src="https://voa-audio-ns.akamaized.net/vpe/episode.mp3" '
        f'data-type="audio/mp3" data-info="64 kbps" data-sources="{sources}"></audio>'
        "</body></html>"
    )


class FakeAudioResponse:
    def __init__(self, payload: bytes, content_type: str = "audio/mpeg", url: str = AUDIO_URL) -> None:
        self.payload = payload
        self.headers = {"Content-Type": content_type}
        self.url = httpx.URL(url)

    def iter_bytes(self, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        for start in range(0, len(self.payload), chunk_size):
            yield self.payload[start : start + chunk_size]


class FakeClient:
    def __init__(
        self,
        texts: dict[str, str],
        audio: dict[str, tuple[bytes, str]],
    ) -> None:
        self.texts = texts
        self.audio = audio
        self.text_requests: list[str] = []
        self.audio_requests: list[str] = []

    def get_text(self, url: str) -> str:
        self.text_requests.append(url)
        if url not in self.texts:
            raise scraper.ScrapeError(f"unexpected URL: {url}")
        return self.texts[url]

    @contextmanager
    def stream(self, url: str) -> Iterator[FakeAudioResponse]:
        self.audio_requests.append(url)
        payload, content_type = self.audio[url]
        yield FakeAudioResponse(payload, content_type)

    def close(self) -> None:
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_parse_feed_uses_episode_page_not_image_enclosure() -> None:
    episodes = scraper.parse_feed(FEED_URL, rss_payload(rss_item()))

    assert episodes == [
        scraper.FeedEpisode(
            feed_url=FEED_URL,
            feed_title="پادکست صوتی آزمایشی - صدای آمریکا",
            title="برنامه آزمایشی",
            published_at="2026-07-14T19:00:00+03:30",
            source_url=SOURCE_URL,
        )
    ]


def test_parse_feed_rejects_non_persian_or_non_voa_channel() -> None:
    payload = "<rss><channel><language>en</language><generator>Other</generator></channel></rss>"

    with pytest.raises(scraper.ScrapeError, match="official Persian VOA"):
        scraper.parse_feed(FEED_URL, payload)


def test_parse_audio_page_selects_highest_bitrate_official_mp3() -> None:
    parsed = scraper.parse_audio_page(audio_page())

    assert parsed.audio_url == AUDIO_URL
    assert parsed.media_type == "audio/mp3"
    assert parsed.entity == "VOA"
    assert parsed.copied == "no"


@pytest.mark.parametrize(
    ("page", "reason"),
    [
        (audio_page(copied="yes"), "copied"),
        (audio_page(entity="Reuters"), "entity"),
        (audio_page(content_type="video"), "content_type"),
        (
            audio_page().replace("voa-audio-ns.akamaized.net", "third-party.example"),
            "missing_official_audio",
        ),
    ],
)
def test_parse_audio_page_fails_closed_for_unverified_material(page: str, reason: str) -> None:
    with pytest.raises(scraper.ScrapeError, match=reason):
        scraper.parse_audio_page(page)


def test_download_writes_raw_clip_and_complete_provenance_manifest(tmp_path: Path) -> None:
    audio_bytes = b"ID3-raw-voa-audio"
    client = FakeClient(
        {FEED_URL: rss_payload(rss_item()), SOURCE_URL: audio_page()},
        {AUDIO_URL: (audio_bytes, "audio/mpeg")},
    )

    audit = scraper.download_voa_persian(
        tmp_path,
        feed_urls=[FEED_URL],
        client=client,  # type: ignore[arg-type]
        retrieved_at=lambda: datetime(2026, 7, 27, 8, 0, tzinfo=timezone.utc),
    )

    assert audit == scraper.DownloadAudit(
        feeds_read=1,
        episodes_seen=1,
        downloaded=1,
        reused=0,
        skipped=0,
        bytes_downloaded=len(audio_bytes),
    )
    assert (tmp_path / "clips" / "8171694.mp3").read_bytes() == audio_bytes
    assert not (tmp_path / "train.tsv").exists()
    assert read_jsonl(tmp_path / "metadata.jsonl") == [
        {
            "audio_url": AUDIO_URL,
            "bytes": len(audio_bytes),
            "checksum": f"sha256:{hashlib.sha256(audio_bytes).hexdigest()}",
            "content_type": "audio",
            "copied": "no",
            "entity": "VOA",
            "feed_title": "پادکست صوتی آزمایشی - صدای آمریکا",
            "feed_url": FEED_URL,
            "id": "8171694",
            "language": "fa",
            "language_service": "VOA Persian",
            "media_type": "audio/mpeg",
            "path": "clips/8171694.mp3",
            "published_at": "2026-07-14T19:00:00+03:30",
            "retrieved_at": "2026-07-27T08:00:00+00:00",
            "rights_policy_url": scraper.RIGHTS_POLICY_URL,
            "source_url": SOURCE_URL,
            "title": "برنامه آزمایشی",
        }
    ]
    assert read_jsonl(tmp_path / "skipped.jsonl") == []


def test_download_logs_live_file_progress_before_final_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_bytes = b"x" * 2048
    client = FakeClient(
        {FEED_URL: rss_payload(rss_item()), SOURCE_URL: audio_page()},
        {AUDIO_URL: (audio_bytes, "audio/mpeg")},
    )

    audit = scraper.download_voa_persian(
        tmp_path,
        feed_urls=[FEED_URL],
        client=client,  # type: ignore[arg-type]
    )
    scraper.print_audit(audit)

    output = capsys.readouterr().out
    assert "Reading feed:" in output
    assert "[1/1] Inspecting: برنامه آزمایشی" in output
    assert "[1/1] Downloading: clips/8171694.mp3" in output
    assert "[1/1] Downloaded: clips/8171694.mp3 (2.00 KiB; cumulative audio 2.00 KiB)" in output
    assert "VOA Persian raw audio download summary" in output
    assert output.index("[1/1] Downloaded:") < output.index("VOA Persian raw audio download summary")


def test_download_records_rejected_episode_without_fetching_audio(tmp_path: Path) -> None:
    client = FakeClient(
        {FEED_URL: rss_payload(rss_item()), SOURCE_URL: audio_page(copied="yes")},
        {},
    )

    audit = scraper.download_voa_persian(
        tmp_path,
        feed_urls=[FEED_URL],
        client=client,  # type: ignore[arg-type]
    )

    assert audit.skipped == 1
    assert audit.downloaded == 0
    assert client.audio_requests == []
    assert read_jsonl(tmp_path / "metadata.jsonl") == []
    assert read_jsonl(tmp_path / "skipped.jsonl")[0]["reason"] == "unverified_provenance:copied"


def test_download_reuses_only_checksum_matching_clip(tmp_path: Path) -> None:
    audio_bytes = b"ID3-cached"
    client = FakeClient(
        {FEED_URL: rss_payload(rss_item()), SOURCE_URL: audio_page()},
        {AUDIO_URL: (audio_bytes, "audio/mpeg")},
    )
    scraper.download_voa_persian(tmp_path, feed_urls=[FEED_URL], client=client)  # type: ignore[arg-type]

    second_client = FakeClient(
        {FEED_URL: rss_payload(rss_item()), SOURCE_URL: audio_page()},
        {},
    )
    audit = scraper.download_voa_persian(
        tmp_path,
        feed_urls=[FEED_URL],
        client=second_client,  # type: ignore[arg-type]
    )

    assert audit.reused == 1
    assert audit.downloaded == 0
    assert second_client.audio_requests == []


def test_download_requires_force_for_checksum_conflict(tmp_path: Path) -> None:
    client = FakeClient(
        {FEED_URL: rss_payload(rss_item()), SOURCE_URL: audio_page()},
        {AUDIO_URL: (b"original", "audio/mpeg")},
    )
    scraper.download_voa_persian(tmp_path, feed_urls=[FEED_URL], client=client)  # type: ignore[arg-type]
    (tmp_path / "clips" / "8171694.mp3").write_bytes(b"modified")

    with pytest.raises(FileExistsError, match="--force"):
        scraper.download_voa_persian(
            tmp_path,
            feed_urls=[FEED_URL],
            client=client,  # type: ignore[arg-type]
        )


def test_download_removes_partial_file_after_invalid_media_response(tmp_path: Path) -> None:
    client = FakeClient(
        {FEED_URL: rss_payload(rss_item()), SOURCE_URL: audio_page()},
        {AUDIO_URL: (b"<html>denied</html>", "text/html")},
    )

    audit = scraper.download_voa_persian(
        tmp_path,
        feed_urls=[FEED_URL],
        client=client,  # type: ignore[arg-type]
    )

    assert audit.skipped == 1
    assert not (tmp_path / "clips" / "8171694.mp3").exists()
    assert not (tmp_path / "clips" / "8171694.mp3.part").exists()


def test_polite_client_respects_robots_disallow() -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200,
                text="User-agent: *\nDisallow: /podcast/sublink/\n",
                request=request,
            )
        raise AssertionError("disallowed URL should not be requested")

    with scraper.PoliteHttpClient(
        delay_seconds=0,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(scraper.RobotsDeniedError):
            client.get_text("https://ir.voanews.com/podcast/sublink/8470")

    assert requests == ["https://ir.voanews.com/robots.txt"]


def test_polite_client_honors_retry_after() -> None:
    target_attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal target_attempts
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n", request=request)
        target_attempts += 1
        if target_attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, request=request)
        return httpx.Response(200, text="ok", request=request)

    with scraper.PoliteHttpClient(
        delay_seconds=0,
        transport=httpx.MockTransport(handler),
        sleeper=sleeps.append,
    ) as client:
        assert client.get_text("https://ir.voanews.com/api/test") == "ok"

    assert target_attempts == 2
    assert 2.0 in sleeps
