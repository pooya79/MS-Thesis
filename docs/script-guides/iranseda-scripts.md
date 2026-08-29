# IranSeda Scripts

## IranSeda Raw Audio Discovery

Discover public audiobook metadata without requesting audio:

```bash
uv run python -m ml.speech_data.data_scraping.iranseda_audiobooks \
  --output-root data/iranseda/audiobooks/raw
```

Discover completed radio archive metadata for an inclusive Gregorian date
range:

```bash
uv run python -m ml.speech_data.data_scraping.iranseda_radio \
  --start-date 2026-07-01 \
  --end-date 2026-07-07 \
  --output-root data/iranseda/radio/raw
```

Both commands are strictly metadata-only. Use
`--book-id`, `--category-code`, `--max-pages`, and `--max-books` to bound an
audiobook run, or repeat `--channel-id` to override radio station discovery.
They save explicit public MP3 URLs without requesting the audio. Download the
dataset-eligible records afterward, once per discovery root:

```bash
uv run python -m ml.speech_data.data_scraping.iranseda_download \
  --source-root data/iranseda/audiobooks/raw \
  --max-download-gib 10

uv run python -m ml.speech_data.data_scraping.iranseda_download \
  --source-root data/iranseda/radio/raw
```

The downloader records paths and SHA-256 checksums in `downloads.jsonl` and
current-run failures in `download_skipped.jsonl`. Existing audio is reused only
when its recorded checksum matches; use `--force` to replace selected files.
Requests use a 120-second timeout by default. If an audio stream times out, its
partial file is removed, the timeout is recorded in `download_skipped.jsonl`,
and the downloader continues with the next selected item. Use
`--timeout-seconds` to override the default.
`--max-download-gib` limits newly transferred audio per run; exceeding the
remaining allowance removes the partial file and stops the batch.
None of these commands creates `train.tsv`. See
[`scraper_guides/iranseda-scrapers.md`](../scraper_guides/iranseda-scrapers.md)
for routes, manifests, classification, and safety behavior.
Run the implemented long-audio segmentation stage with lossless FLAC output:

```bash
uv run python -m ml.speech_data.long_audio_asr_pipeline.segment_audio \
  --config configs/long_audio_asr_pipeline/segmentation.yaml \
  --manifest data/iranseda/radio/raw/downloads.jsonl \
  --source-root data/iranseda/radio/raw \
  --output-root data/iranseda/segmented/flac-v1 \
  --workers 4
```

`--workers` processes independent source cohorts concurrently. The checked
configuration keeps `execution.vad_batch_size: 1` for scientifically identical
Silero probability records, uses `execution.vad_device: cpu`, and sets
`execution.torch_threads: 1` to avoid oversubscribing tiny recurrent inference
calls. Each worker owns its own model,
while audit-manifest writes remain serialized. Start with 2–4 workers and
adjust for available CPU, temporary disk, and memory. Batch sizes above one are
experimental because PyTorch batching can change recorded probabilities at
approximately `1e-7`; execution settings are included in the configuration
digest. On GPU servers, decoded cohort waveforms remain GPU-resident during VAD,
so each waveform is uploaded once and its probabilities are downloaded once.
Use `vad_device: cuda` with batch size 8–32 for short or medium recordings and
start with one or two workers. For multi-hour recordings, begin with batch size
1–2 because device and host staging memory scale with the padded cohort; CUDA
inputs are limited to five hours per source. CUDA accelerates only VAD, while
decoding and export remain on CPU. See
`docs/long-audio-asr-pipeline-guide.md` for profiling and tuning.

For smaller lossy outputs, follow the inline comments in
`configs/long_audio_asr_pipeline/segmentation.yaml`: change the `audio` block
to `format: MP3`, remove `subtype`, and add `bitrate_kbps: 48`. Format, bitrate,
VAD, and boundary settings are included in the run's configuration digest.

Transcription can run before normal segmentation finishes:

```bash
uv run python -m ml.speech_data.long_audio_asr_pipeline.transcribe_segments \
  --config configs/long_audio_asr_pipeline/transcription.yaml \
  --input-root data/iranseda/segmented/flac-v1
```

The command freezes the currently available, atomically completed clips in
`transcription_pending_snapshot.jsonl`, including clips exported before
`segments.jsonl` receives their source-level records. Temporary `.part` files
are ignored. Results are checkpointed after every batch and accumulated into a
deduplicated TSV/JSONL result. Run the same command again to transcribe later
clips and retry operational failures; successful results and normalization
rejections are reused. Later official manifest metadata is added to reused
fallback transcripts automatically. Do not run forced segmentation on the
shared root at the same time. Flushed `[transcribe]` stdout messages report clip
indexing, discovery/checksumming, snapshot writing, per-batch validation, model
initialization, inference batches, and saved checkpoints.

To refine an approximately hour-limited random subset, first select complete
original-audio groups from the finished segmented/transcribed root:

```bash
uv run python -m ml.speech_data.long_audio_asr_pipeline.select_refinement_subset \
  --input-root data/iranseda/segmented/flac-v1 \
  --hours 100 \
  --seed 0 \
  --output data/iranseda/refinement-selection-100h.json
```

The selector includes a source only when every segment has a usable matching
transcription. It shuffles whole `source_id` groups deterministically and picks
the prefix immediately below or above the requested duration that is closer;
therefore the selected duration can differ from `--hours`. The JSON manifest
records selected source IDs, durations, segment counts, seed, and exact upstream
checksums. Regenerate it whenever `segments.jsonl` or `transcriptions.jsonl`
changes. Flushed `[select-refinement]` messages report manifest loading,
validation progress every 1,000 records, source completeness, selection,
checksumming, and output writing.

Refine completed normalized transcripts through vLLM without altering the
Whisper artifacts:

```bash
uv run python -m ml.speech_data.long_audio_asr_pipeline.refine_transcriptions \
  --config configs/long_audio_asr_pipeline/refinement.yaml \
  --input-root data/iranseda/segmented/flac-v1 \
  --books-manifest data/iranseda/audiobooks/raw/books.jsonl
```

To apply the subset, set `selection.manifest` in the refinement YAML. Relative
paths are resolved from the YAML file's directory:

```yaml
selection:
  manifest: ../../data/iranseda/refinement-selection-100h.json
```

Leave the value `null` to refine every usable transcription. A configured
selection filters by exact `source_id`, so all short clips from each chosen
original recording remain available as preceding and following context.

This command requires vLLM's non-streaming native
`/v1/chat/completions/batch` route and checks for it before inference. It batches
at most one target from each original source, uses accepted refinements only as
past context, uses normalized Whisper text as future context, and checkpoints
after every native batch. Strict JSON, Persian normalization, unchanged numeric
tokens, uncertainty, and edit-distance checks decide whether a row is published
to `refined_transcription.tsv`. Full prompt/response and validation audits are
kept in `refinements.jsonl` or `refinement_rejected.jsonl`. Identical safe
records resume; operational failures retry; changed upstream/context/title/description data
invalidates reuse. `--force` stages and atomically replaces only refinement
artifacts. See [`long-audio-asr-pipeline-guide.md`](../long-audio-asr-pipeline-guide.md)
for the complete artifact and dataset-safety contract.

Discovery prints live category/station/item progress and writes atomic
checkpoints during the crawl. Audiobooks checkpoint every 10 processed books
by default; radio checkpoints every station-day. Set `--checkpoint-every 1` for
the most frequent audiobook checkpoints, or increase the value to reduce disk
writes. Reruns skip completed work recorded in `discovery_checkpoints.jsonl`;
use `--refresh` to deliberately revisit it. The downloader checkpoints and logs
every selected audio item.
