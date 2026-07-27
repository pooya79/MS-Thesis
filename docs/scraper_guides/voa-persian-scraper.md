# VOA Persian Raw Audio Scraper

Download publicly accessible, VOA-produced Persian podcast audio from official
VOA Persian RSS feeds. This is a raw acquisition step: downloaded MP3 bytes are
kept unchanged for later transcription and normalization.

## Command

Inspect all options and defaults:

```bash
uv run python -m ml.speech_data.data_scraping.voa_persian --help
```

Run against the built-in official audio feed list:

```bash
uv run python -m ml.speech_data.data_scraping.voa_persian \
  --output-root data/voa-persian/raw
```

Start with a small trial that inspects at most one RSS item per feed:

```bash
uv run python -m ml.speech_data.data_scraping.voa_persian \
  --output-root data/voa-persian/raw \
  --max-items 1
```

## Output Layout

```text
data/voa-persian/raw/
├── clips/
│   └── <voa-article-id>.mp3
├── metadata.jsonl
└── skipped.jsonl
```

Each `metadata.jsonl` record includes the local path, episode page URL, feed,
title, publication date, direct audio URL, SHA-256 checksum, byte size, media
type, retrieval time, VOA provenance fields, and the VOA rights-policy URL.
`skipped.jsonl` records rejected candidates and a machine-readable reason.

The scraper does not create `train.tsv`. Episode titles and descriptions are
metadata rather than transcripts of the speech, so ASR split files must wait
until a later transcription and normalization step.

## Source and Copyright Safeguards

The built-in feed list contains the official VOA Persian audio feeds for
`تفسیر خبر`, `یادآر`, and `تابو`. RSS image enclosures are ignored; the scraper
visits each official episode page and selects its highest-bitrate official MP3.

An item is downloaded only when page metadata identifies it as Persian audio
produced by VOA and not copied. The episode page and audio must use explicitly
allowed HTTPS hosts. Missing or conflicting provenance, non-audio pages,
third-party media hosts, and invalid audio responses are rejected.

The scraper checks `robots.txt` for every origin, never requests VOA's
robots-disallowed `/podcast/sublink/*` paths, waits at least one second between
requests to the same origin by default, honors longer crawl delays, and honors
`Retry-After` during bounded retries.

VOA's rights notice says material produced exclusively by VOA is public domain,
while licensed third-party material is not. Keep `source_url`,
`rights_policy_url`, and the other provenance fields with every downloaded
clip.

## Options

- `--feed-url URL`: repeat to replace the built-in feed list with selected
  official VOA Persian feeds. Custom feeds still undergo the same origin and
  provenance validation.
- `--max-items N`: inspect no more than `N` RSS items per feed.
- `--delay-seconds N`: set the minimum per-origin request interval; the default
  is `1.0`, and a larger server-declared crawl delay takes precedence.
- `--timeout-seconds N`: set the per-request timeout; the default is `30.0`.
- `--user-agent TEXT`: override the descriptive research user agent.
- `--force`: redownload selected article IDs and atomically replace their clips.

## Restart and Validation Behavior

Normal reruns recompute the SHA-256 checksum of an existing clip. A clip is
reused only when that value matches its `metadata.jsonl` record. A mismatch
stops the run and requires explicit `--force`; local content is never silently
overwritten.

Downloads are streamed to `.part` files. Empty responses, unexpected media
types, failed requests, and interrupted transfers remove the partial file.
Completed clips and JSONL manifests are moved into place atomically.

While running, the command logs each feed and episode to stdout. It reports
inspection, skip, reuse, and download events; every completed download includes
the file size and cumulative successfully downloaded audio bytes in human-readable
units. The existing end-of-run summary still reports total feeds, episodes,
downloaded files, reused files, skipped items, and downloaded audio bytes.
