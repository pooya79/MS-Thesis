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
Each in-flight source also needs about 115 MB of temporary WAV space per hour
of source audio. Working WAVs are deleted when their source cohort finishes.

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
  --output-root data/long-audio-segments/v1 \
  --workers 4
```

Supported discovery extensions are AAC, FLAC, M4A, MP3, OGG, Opus, WAV, and
WMA. Directory inputs receive deterministic IDs derived from their relative
paths. Use a manifest when IDs must remain stable after files are moved.
`--workers` controls how many source cohorts are decoded, analyzed, and
exported concurrently. With the reproducible default `vad_batch_size: 1`, a
cohort is one source and behavior is equivalent to processing that many source
files concurrently. Each worker owns a Silero VAD model instance; start with
2–4 workers and reduce the value if memory, CPU, or disk contention becomes
excessive. Manifest updates remain serialized and deterministic. The default
is one worker.

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
  --output-root data/iranseda/segmented/v1 \
  --workers 4
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
- PyTorch VAD runtime with batch size `1` and one Torch compute thread.
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

## VAD Execution and Performance

Execution settings live in the segmentation YAML so every run records the
runtime choices in `effective_config.yaml` and its configuration digest:

```yaml
execution:
  vad_engine: pytorch
  vad_device: cpu
  vad_batch_size: 1
  torch_threads: 1
```

The detector reads 2,048 VAD frames per audio read, but still sends the model
the required 512 samples in temporal order. Inference mode covers the complete
recording, FFmpeg version discovery is cached once per run, and interval
construction advances through speech intervals instead of restarting its
search for every output clip. These changes preserve single-source VAD
probabilities, boundaries, and exported audio.

Keep `vad_device: cpu` and `vad_batch_size: 1` when reproducing an existing CPU
dataset run. PyTorch batching
of independent recordings is implemented for controlled experiments, but a
representative Persian sample produced mean/max VAD probability differences
around `1e-7`. Start/end decisions were unchanged in that check, but the audit
records were not scientifically identical. A batch size above one therefore
requires a new output root and an explicit quality audit. The maximum number
of in-flight decoded sources is approximately `--workers × vad_batch_size`, so
temporary disk demand grows by the same factor.

On a GPU server, move batched Silero inference to CUDA explicitly:

```yaml
execution:
  vad_engine: pytorch
  vad_device: cuda
  vad_batch_size: 16
  torch_threads: 1
```

Start with `--workers 2` on a four-core host, then compare batch sizes 8, 16,
and 32 using a fixed manifest. CUDA accelerated the packaged model only once
several independent recordings were batched; batch size one remained faster on
CPU. The model and batched 512-sample tensors run on CUDA, while FFmpeg decode,
energy-boundary analysis, clip export, checksums, and manifest work remain on
CPU. CUDA availability is checked before the output directory is initialized;
the command fails instead of silently falling back to CPU. The selected device
is recorded in each source's `vad_model` metadata.

CPU and CUDA can differ slightly in floating-point probability summaries, so a
device change requires a new output root and a quality comparison. Large VRAM
does not remove the temporary-disk cost: each cohort retains decoded working
WAVs until its longest source finishes VAD.

Small Silero calls can become slower with large Torch thread pools. The checked
configuration uses one Torch thread so file/cohort workers provide the outer
parallelism. Benchmark `torch_threads` together with `--workers`; changing
any YAML execution value changes the configuration digest.

For a reproducible wall-time profile, use a fixed source and a new temporary
output root, record the source duration with `ffprobe`, and time the complete
command:

```bash
ffprobe -v error -show_entries format=duration \
  -of default=nw=1:nk=1 data/recordings/profile.mp3
/usr/bin/time -v uv run python \
  -m ml.speech_data.long_audio_asr_pipeline.segment_audio \
  --config configs/long_audio_asr_pipeline/segmentation.yaml \
  --input data/recordings/profile.mp3 \
  --output-root /tmp/long-audio-profile \
  --workers 1
```

ONNX Runtime is not a supported engine. It should be added only if a
representative comparison is at least 20% faster end-to-end, produces
identical VAD interval records and clip checksums, and does not materially
regress memory use or per-source failure isolation.

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

Transcription can start while a normal, non-forced segmentation run is still
publishing clips. Each invocation transcribes the manifest-backed clips that
are available when it starts and applies the repository's shared Persian ASR
text rules:

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

The input root must contain the `segment_audio` artifacts `run.json`,
`segments.jsonl`, and `clips/`. The stage reads one atomic view of
`segments.jsonl`, verifies clip paths,
checksums, readability, channel count, and 16 kHz sample rate before inference.
It writes:

```text
transcription.tsv
transcriptions.jsonl
transcription_rejected.jsonl
transcription_pending_snapshot.jsonl
transcription_summary.json
transcription_run.json
transcription_effective_config.yaml
```

`transcription_pending_snapshot.jsonl` is atomically replaced at startup and
contains the exact worklist selected for that invocation. Clips published by
the segmenter afterward wait for the next invocation. `transcription.tsv`
contains `path` and `sentence`, with paths relative to `clips/`. The accepted
JSONL preserves raw and normalized transcripts plus the model and decoding
identity. Rejected normalization and operational failures are recorded
separately; operational failures make the command exit nonzero.

Results are checkpointed atomically after every inference batch. An identical
rerun reuses records whose clip checksum and transcription digest still match,
processes newly published clips, and retries operational failures. Normalization
rejections are reused. The canonical TSV and JSONL outputs are rebuilt as a
sorted, deduplicated union rather than byte-appended. Changed settings require
`--force`. A forced transcription stages its output and preserves the previous
transcription artifacts if an operational failure occurs. Do not run forced
segmentation against the shared root during transcription because forced
segmentation replaces that directory. Multiple transcription processes must
not use the same input root simultaneously.

Constrained LLM cleanup, content filtering, leakage-safe source grouping,
train/dev/test publication, and human audit remain later stages described in
[`iranseda-whisper-dataset-pipeline.md`](iranseda-whisper-dataset-pipeline.md).
