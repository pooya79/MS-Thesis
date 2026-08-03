# Long-Audio ASR Pipeline Guide

Every maintained script in this pipeline exposes `--help`. Inspect the command
before running it with custom paths or configuration:

```bash
uv run python -m ml.speech_data.long_audio_asr_pipeline.segment_audio --help
uv run python -m ml.speech_data.long_audio_asr_pipeline.transcribe_segments --help
```

## Segment Long Audio with VAD

The first reusable stage of the long-audio ASR pipeline detects speech and
exports deterministic, non-overlapping audio clips. It is dataset-independent:
it accepts individual audio files, recursively scanned directories, or a JSONL
source manifest. It does not transcribe clips or create train/dev/test TSVs.

The stage uses the packaged Silero VAD model, decodes every input through
FFmpeg to mono 16 kHz working audio, and exports mono 16 kHz PCM-16 FLAC by
default. Source audio is never modified. The default policy targets 20 seconds,
prefers 15–25 seconds, and retains natural speech clips shorter than 15 seconds
when they contain at least two seconds of detected speech.

FLAC is the default because it is smaller than WAV without introducing another
lossy codec pass. The comments in the segmentation configuration show every
supported output option. When disk space is the overriding constraint, edit a
working copy of its `audio` block to export 48-kbps MP3:

```yaml
audio:
  bitrate_kbps: 48
  channels: 1
  format: MP3
  sample_rate: 16000
```

MP3 clips are accurately cut and re-encoded from the decoded working audio;
they are not copied at approximate MP3 frame boundaries. To choose another
rate, set `audio.bitrate_kbps` to an integer from 8 through 160. Do not include
`subtype` for MP3; that setting applies only to FLAC and WAV. A 48-kbps mono
output occupies about 22 MB per hour, versus about 115 MB per hour for PCM WAV.
The temporary working WAV still requires about 115 MB per hour of the one
source currently being processed and is deleted when that source finishes.

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
│   └── episode-42_000000.flac
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
format, sample rate, channel count, and codec subtype. Missing or corrupt clips
are regenerated. A different configuration, including an output format or MP3
bitrate change, is rejected unless `--force` is used.
Forced generation happens in a staging directory and replaces the existing
output only after all sources finish without operational failures.

Quality-only rejection such as `no_usable_speech` is reported but does not make
the command fail. Missing files, checksum failures, decoding failures, VAD
errors, and export errors are operational failures: processing continues, but
the command exits nonzero after writing its audit manifests.

## Whisper Transcription and Normalization

After segmentation, transcribe every manifest-backed clip with the fine-tuned
Whisper Medium model and apply the repository's shared Persian ASR text rules:

```bash
uv run python -m ml.speech_data.long_audio_asr_pipeline.transcribe_segments \
  --config configs/long_audio_asr_pipeline/transcription.yaml \
  --input-root data/iranseda/segmented
```

Set `model.checkpoint` in the YAML to the trained final model or Trainer
checkpoint. The processor defaults to `openai/whisper-medium`, allowing a
weights-only Trainer checkpoint to be used. The inference section controls
device selection, mixed precision, batch size, and maximum generation length;
decoding is deterministic and explicitly uses Persian transcription mode.

The input root must be an unchanged `segment_audio` output containing
`run.json`, `segments.jsonl`, and `clips/`. The stage verifies clip paths,
checksums, readability, channel count, and 16 kHz sample rate before inference.
It writes:

```text
transcription.tsv
transcriptions.jsonl
transcription_rejected.jsonl
transcription_summary.json
transcription_run.json
transcription_effective_config.yaml
```

`transcription.tsv` contains `path` and `sentence`, with paths relative to
`clips/`. The accepted JSONL preserves raw and normalized transcripts plus the
model and decoding identity. Rejected normalization and operational failures
are recorded separately; operational failures make the command exit nonzero.

An identical rerun reuses records whose clip checksum and transcription digest
still match. Changed settings require `--force`. A forced run stages its output
and preserves the previous transcription artifacts if an operational failure
occurs. `--force` never changes clips or segmentation manifests.

Constrained LLM cleanup, content filtering, leakage-safe source grouping,
train/dev/test publication, and human audit remain later stages described in
[`iranseda-whisper-dataset-pipeline.md`](iranseda-whisper-dataset-pipeline.md).
