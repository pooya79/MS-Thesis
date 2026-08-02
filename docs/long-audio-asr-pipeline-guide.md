# Long-Audio ASR Pipeline: VAD Segmentation Guide

Every maintained script in this pipeline exposes `--help`. Inspect the command
before running it with custom paths or configuration:

```bash
uv run python -m ml.speech_data.long_audio_asr_pipeline.segment_audio --help
```

## Segment Long Audio with VAD

The first reusable stage of the long-audio ASR pipeline detects speech and
exports deterministic, non-overlapping WAV clips. It is dataset-independent:
it accepts individual audio files, recursively scanned directories, or a JSONL
source manifest. It does not transcribe clips or create train/dev/test TSVs.

The stage uses the packaged Silero VAD model, decodes every input through
FFmpeg to mono 16 kHz working audio, and exports mono 16 kHz PCM-16 WAV. Source
audio is never modified. The default policy targets 20 seconds, prefers 15–25
seconds, and retains natural speech clips shorter than 15 seconds when they
contain at least two seconds of detected speech.

## Installation

Install the locked project dependencies and confirm FFmpeg is available:

```bash
uv sync
ffmpeg -version
```

Silero is bundled by the `silero-vad` package. The command does not require a
Hugging Face token or download a model at runtime.

## Input Modes

Process one file, several files, or directories recursively. Repeat `--input`
as needed:

```bash
uv run python -m ml.speech_data.long_audio_asr_pipeline.segment_audio \
  --config configs/long_audio_asr_pipeline/segmentation.yaml \
  --input data/recordings/episode-001.mp3 \
  --input data/more-recordings \
  --output-root data/long-audio-segments/v1
```

Supported discovery extensions are AAC, FLAC, M4A, MP3, OGG, Opus, WAV, and
WMA. Directory inputs receive deterministic IDs derived from their relative
paths. Use a manifest when IDs must remain stable after files are moved.

For manifest input, each non-empty JSONL row must contain string `id` and
`path` fields. An optional `checksum` must use `sha256:<64 lowercase hex>`.
Other fields are preserved in `sources.jsonl` as source metadata:

```json
{"id":"episode-42","path":"clips/station/42.mp3","checksum":"sha256:...","source_kind":"radio"}
```

Relative paths resolve against the manifest directory by default. Override the
base explicitly with `--source-root`:

```bash
uv run python -m ml.speech_data.long_audio_asr_pipeline.segment_audio \
  --config configs/long_audio_asr_pipeline/segmentation.yaml \
  --manifest data/iranseda/radio/raw/downloads.jsonl \
  --source-root data/iranseda/radio/raw \
  --output-root data/iranseda/segmented/v1
```

IranSeda `downloads.jsonl` files already satisfy this interface; the command
does not import IranSeda code or interpret its source-specific metadata.

## Boundary Policy and Configuration

Edit a copied configuration rather than changing an existing experiment in
place. Important defaults in
`configs/long_audio_asr_pipeline/segmentation.yaml` are:

- Silero speech threshold: `0.5`.
- Minimum VAD speech event: `0.25` seconds.
- Merge speech across gaps shorter than `0.5` seconds.
- Useful silence boundary: `0.3` seconds.
- Target/preferred duration: `20` seconds within `15–25` seconds.
- Minimum accepted detected speech: `2` seconds.
- Boundary padding: `0.15` seconds.
- Energy fallback: lowest-RMS 200 ms window from 18–25 seconds, accepted when
  it is at least 6 dB below the search-region median.

Within the preferred window, the command chooses the useful silence midpoint
nearest 20 seconds. With no silence, it uses the energy fallback. If no strong
energy dip exists, it cuts at 25 seconds and records `hard_cut`. Final clips do
not overlap. A separated short utterance is emitted as `natural_short` rather
than being padded with silence or joined to unrelated speech.

Tune VAD thresholds on an audited sample from the target domain. A changed
configuration produces a new digest and is intentionally treated as a
different preparation run.

## Output and Resume Behavior

The output has this layout:

```text
output-root/
├── clips/
│   └── episode-42_000000.wav
├── effective_config.yaml
├── run.json
├── sources.jsonl
├── vad_intervals.jsonl
├── segments.jsonl
├── rejected.jsonl
└── summary.json
```

`segments.jsonl` records source and clip checksums, exact source timestamps,
duration, detected-speech seconds and ratio, boundary type, silence length or
energy dip, and configuration digest. `vad_intervals.jsonl` records detected
speech intervals and probability summaries. `rejected.jsonl` uses stable
reasons such as `checksum_mismatch`, `audio_decode_failed`, and
`no_usable_speech`.

Rerunning with the same effective configuration reuses a source only when its
checksum matches and every recorded clip still has the expected checksum,
sample rate, channel count, and PCM subtype. Missing or corrupt clips are
regenerated. A different configuration is rejected unless `--force` is used.
Forced generation happens in a staging directory and replaces the existing
output only after all sources finish without operational failures.

Quality-only rejection such as `no_usable_speech` is reported but does not make
the command fail. Missing files, checksum failures, decoding failures, VAD
errors, and export errors are operational failures: processing continues, but
the command exits nonzero after writing its audit manifests.

## Next Pipeline Stages

The generated clips and `segments.jsonl` are the inputs for future Whisper
pseudo-transcription. Transcript normalization, constrained LLM cleanup,
content filtering, leakage-safe source grouping, TSV generation, and human
audit remain separate stages described in
[`iranseda-whisper-dataset-pipeline.md`](iranseda-whisper-dataset-pipeline.md).
