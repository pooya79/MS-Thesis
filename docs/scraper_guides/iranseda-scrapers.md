# IranSeda Audiobook and Radio Scrapers

The IranSeda discovery commands collect reproducible raw metadata and explicit
public MP3 URLs without requesting audio. A separate downloader consumes those
saved manifests. The commands do not log in, send credentials, infer media
URLs, access live streams as dataset audio, or produce an ASR `train.tsv`.

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
  URL, explicit MP3 URL, declared size, and sample/full-book flags.
- `discovery_checkpoints.jsonl`: processed book IDs used for safe resume.
- `skipped.jsonl`: page, parsing, policy, and discovery failures.

The command logs category pages, book progress, parsing skips, and checkpoint
counts to the terminal. It atomically checkpoints all three manifests every 10
processed books by default, on interruption, and at completion. Use
`--checkpoint-every N` to choose a different interval; `1` gives maximum
durability at the cost of more manifest rewrites.
On rerun, checkpointed books are skipped. Pass `--refresh` to revisit them.

Discovery never requests the MP3 URLs. During the separate download phase, one
explicit full-book MP3 is preferred. If no full-book MP3 is exposed, all
explicit non-sample chapter MP3s are selected. Sample and alternative links
remain metadata-only.

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
`skipped.jsonl`, plus `discovery_checkpoints.jsonl` for completed station-days.

Station, date, episode, skip, and checkpoint progress is printed to the
terminal. Manifests are atomically checkpointed after every processed
station-day by default, on interruption, and at completion. Increase
`--checkpoint-every N` to checkpoint every N station-days instead.
Checkpointed station-days are skipped on rerun; `--refresh` revisits them.
Days containing incomplete episodes or request/parsing failures are not marked
complete, so a later run retries them.

Classification is deliberately broad: an episode stays eligible unless its
metadata explicitly identifies music, recitation, advertising/promotion,
station filler, or non-Persian material. Excluded episodes and explicit links
remain in metadata.

## Download discovered audio

Run the downloader once for each discovery root:

```bash
uv run python -m ml.speech_data.data_scraping.iranseda_download \
  --source-root data/iranseda/audiobooks/raw

uv run python -m ml.speech_data.data_scraping.iranseda_download \
  --source-root data/iranseda/radio/raw
```

Use `--max-download-gib N` to cap newly transferred audio in one invocation.
Checksum-verified files reused from disk do not count. If the next response
would cross the remaining allowance, its partial file is removed, the event is
recorded in `download_skipped.jsonl`, and the batch stops with a nonzero exit.
For example, `--max-download-gib 10` limits a run to 10 GiB; decimal values such
as `0.5` are accepted.

The source root must contain exactly one of `tracks.jsonl` or `episodes.jsonl`.
The downloader does not revisit catalogue, EPG, archive, or program pages. It
selects the audiobook and eligible Persian-speech radio records described
above, then writes:

- `downloads.jsonl`: source type and manifest, URL, local path, SHA-256,
  response size/type, and download/verification times.
- `download_skipped.jsonl`: failures from the current run.

Audiobooks are stored at `clips/<book-id>/<attachment-id>.mp3`; radio episodes
are stored at `clips/<station-id>/<YYYY-MM-DD>/<episode-id>.mp3`. State is
checkpointed atomically after each item. A failed item does not stop the rest of
the batch, but the command exits nonzero when any selected item fails. Rerun the
same command to retry. Existing audio is reused only when its recorded checksum
matches; `--force` replaces selected files. Legacy checksum/path fields written
by the former combined workflow are verified and imported into
`downloads.jsonl`. Selection, download, reuse, and failure progress is printed
as each item is processed.

## Network and file safeguards

Only the public IranSeda catalogue, archive, player-download, and headend
hosts/routes linked by the inspected pages are allowed. HTTP and HTTPS are both
accepted because the site redirects HTTPS to HTTP; every redirect is validated
again. The client checks `robots.txt` separately for every origin and stops if
the policy denies a path. Following RFC 9309, a 4xx response means the robots
file is unavailable and permits access; network errors and 5xx responses remain
fail-closed, as does a successful response that cannot be parsed as a robots
policy. The client honors crawl delay, `Retry-After`, bounded retries, and a
one-second default per-origin delay.

Discovery may read an explicit book download modal or completed radio archive
detail but issues no audio request. The downloader still verifies the current
`robots.txt` for every audio origin. Audio streams through a `.part` file, must
have a non-empty accepted audio response, receives a SHA-256 checksum, and
atomically replaces its destination.

Saved MP3 URLs may become stale. If IranSeda changes or invalidates them, rerun
discovery before retrying the downloader.

Review the site's current terms and robots policy before a real crawl. In
particular, an unreachable `robots.txt` still stops the commands rather than
assuming permission.
