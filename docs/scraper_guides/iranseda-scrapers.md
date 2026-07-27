# IranSeda Audiobook and Radio Scrapers

The IranSeda commands collect reproducible raw metadata and, only when
explicitly requested, publicly linked MP3 audio. They do not log in, send
credentials, infer media URLs, access live streams as dataset audio, or produce
an ASR `train.tsv`.

## Audiobooks

Metadata-only discovery:

```bash
uv run python -m ml.speech_data.data_scraping.iranseda_audiobooks \
  --output-root data/iranseda/audiobooks/raw
```

For a small reproducible run:

```bash
uv run python -m ml.speech_data.data_scraping.iranseda_audiobooks \
  --category-code daa \
  --max-pages 1 \
  --max-books 10 \
  --output-root data/iranseda/audiobooks/raw
```

`--book-id` may be repeated to inspect exact public detail-page IDs and bypass
category discovery. Serial containers are expanded into their linked book
detail pages. The command writes:

- `books.jsonl`: title, authors, narrators, description, categories, duration,
  source page, serial parent, track IDs, and retrieval time.
- `tracks.jsonl`: track title and duration, attachment ID, player URL, modal
  URL, explicit MP3 URL, declared size, sample/full-book flags, and any local
  download audit fields.
- `skipped.jsonl`: page, parsing, policy, classification, and download failures.

With `--download`, one explicit full-book MP3 is preferred. If no full-book MP3
is exposed, all explicit non-sample chapter MP3s are selected. Sample and
alternative links remain in metadata. Files are stored at
`clips/<book-id>/<attachment-id>.mp3`.

## Radio archives

Both dates are required, Gregorian, ISO-formatted, and inclusive:

```bash
uv run python -m ml.speech_data.data_scraping.iranseda_radio \
  --start-date 2026-07-01 \
  --end-date 2026-07-07 \
  --output-root data/iranseda/radio/raw
```

Future and reversed ranges are rejected. National and provincial stations are
discovered by default. Explicit music/recitation, radio-visual, and explicitly
non-Persian stations are retained in `stations.jsonl` but excluded. Repeating
`--channel-id` overrides the default station selection.

The command traverses dated EPG pages, completed archive details, and linked
program pages. Live player links may be retained for provenance but are never
downloaded. It writes `stations.jsonl`, `programs.jsonl`, `episodes.jsonl`, and
`skipped.jsonl`.

Classification is deliberately broad: an episode stays eligible unless its
metadata explicitly identifies music, recitation, advertising/promotion,
station filler, or non-Persian material. Excluded episodes and explicit links
remain in metadata. With `--download`, only eligible completed episodes with an
explicit archive MP3 are stored at
`clips/<station-id>/<YYYY-MM-DD>/<episode-id>.mp3`.

## Network and file safeguards

Only the public IranSeda catalogue, archive, player-download, and headend
hosts/routes linked by the inspected pages are allowed. HTTP and HTTPS are both
accepted because the site redirects HTTPS to HTTP; every redirect is validated
again. The client checks `robots.txt` separately for every origin and stops if
the policy cannot be verified or denies a path. It honors crawl delay,
`Retry-After`, bounded retries, and a one-second default per-origin delay.

Metadata-only mode may read an explicit book download modal or completed radio
archive detail but issues no audio request. Audio streams through a `.part`
file, must have a non-empty accepted audio response, receives a SHA-256
checksum, and atomically replaces its destination. An existing clip is reused
only when its manifest checksum matches; otherwise use `--force` deliberately.

Review the site's current terms and robots policy before a real crawl. In
particular, an unavailable `robots.txt` intentionally stops the commands rather
than assuming permission.
