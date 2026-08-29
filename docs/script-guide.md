# Script Guide

Every maintained Python script exposes `--help`. Use the help output before running a script with custom paths:

```bash
uv run python -m ml.speech_data.data_scraping.iranseda_audiobooks --help
uv run python -m ml.speech_data.data_scraping.iranseda_download --help
uv run python -m ml.speech_data.data_scraping.iranseda_radio --help
uv run python -m ml.speech_data.scripts.download_common_voice_fa --help
uv run python -m ml.speech_data.scripts.download_degradation_assets --help
uv run python -m ml.speech_data.scripts.download_filimo_persian_asr --help
uv run python -m ml.speech_data.scripts.download_fleurs_persian --help
uv run python -m ml.speech_data.scripts.download_persian_eval_sets --help
uv run python -m ml.speech_data.scripts.download_youtube_persian_asr --help
uv run python -m ml.speech_data.scripts.compute_audio_hours --help
uv run python -m ml.speech_data.scripts.create_mixed_test_dataset --help
uv run python -m ml.speech_data.scripts.summarize_hf_audio_dataset --help
uv run python -m ml.speech_data.scripts.upload_hf_audio_dataset --help
bash ml/speech_data/scripts/upload_persian_audiobook_subset.sh --help
uv run python -m ml.speech_data.scripts.prepare_common_voice_25 --help
uv run python -m ml.speech_data.scripts.prepare_degradation_assets --help
uv run python -m ml.speech_data.scripts.prepare_filimo_persian_asr --help
uv run python -m ml.speech_data.scripts.prepare_fleurs_persian --help
uv run python -m ml.speech_data.scripts.prepare_persian_eval_sets --help
uv run python -m ml.speech_data.scripts.prepare_youtube_persian_asr --help
uv run python -m ml.speech_data.scripts.convert_dataset_to_flac --help
uv run python -m ml.speech_data.scripts.verify_flac_conversion --help
uv run python -m ml.speech_data.scripts.generate_random_degraded_clip --help
uv run python -m ml.speech_data.generate_degraded_dataset --help
uv run python -m ml.speech_data.generate_degraded_pairs --help
uv run python -m ml.speech_data.generate_noise_added_dataset --help
uv run python -m ml.speech_data.inspect_manifest --help
uv run python -m ml.speech_data.long_audio_asr_pipeline.segment_audio --help
uv run python -m ml.speech_data.long_audio_asr_pipeline.transcribe_segments --help
uv run python -m ml.speech_data.long_audio_asr_pipeline.select_refinement_subset --help
uv run python -m ml.speech_data.long_audio_asr_pipeline.refine_transcriptions --help
uv run python -m ml.speech_data.validate_degraded_dataset --help
uv run python -m ml.asr.train_whisper_small --help
uv run python -m ml.pmct.train_whisper_small --help
uv run python -m ml.asr.eval_whisper_small --help
uv run python -m ml.asr.train_fastconformer --help
uv run python -m ml.asr.eval_fastconformer --help
uv run python -m ml.asr.eval_mixed_dataset --help
uv run python -m ml.asr.eval_openrouter_stt --help
uv run python -m ml.asr.rescore_openrouter_stt --help
uv run python -m ml.asr.eval_elevenlabs_scribe --help
uv run python -m ml.asr.eval_ivira_avanegar --help
uv run python -m ml.fusion.train_fusion --help
uv run python -m ml.fusion.eval_fusion --help
uv run python -m ml.enhancement.diagnose_enhancement --help
```

## Evaluate OpenRouter Speech-to-Text Models

Evaluate multiple OpenRouter STT models against the mixed test dataset while
retaining every exact prediction and tracking actual API cost:

```bash
export OPENROUTER_API_KEY='YOUR_DEDICATED_EVALUATION_KEY'

uv run python -m ml.asr.eval_openrouter_stt \
  --dataset-root data/mixed-persian-test \
  --model openai/whisper-large-v3 \
  --model openai/gpt-4o-mini-transcribe \
  --max-run-cost-usd 5.00 \
  --min-key-remaining-usd 1.00 \
  --output-dir artifacts/openrouter-stt/mixed-persian-test
```

Use OpenRouter model slugs that advertise the `transcription` output modality.
The API key is read only from `OPENROUTER_API_KEY`; do not put it in a config,
command argument, or committed file. For the strongest cost protection, create
a dedicated OpenRouter key with its own server-side spending limit. The required
`--max-run-cost-usd` is also checked locally before every sequential request,
and `--min-key-remaining-usd` stops when the key's reported `limit_remaining`
reaches that reserve. Because OpenRouter reports exact cost only after a request,
the local run cap alone can be exceeded by one in-flight request; the dedicated
key limit is the hard remote guard.

The output is checkpointed after each successful request. `predictions.jsonl`
is the append-only source of truth and includes the exact model response,
reference, normalized scoring strings, per-clip WER/CER, source dataset, full
usage object, and request cost. `predictions.tsv` is a convenient tabular view;
`metrics.json` reports corpus-level WER/CER and cost for every model, overall and
for every `source_dataset`. `events.jsonl` records each budget check and request
transition, while `logs/openrouter_stt.log` is the human-readable live log.
Interrupted or budget-stopped evaluations can continue with `--resume` and the
same dataset, ordered model list, language, and output directory. A completed
run exits 0, API failures/incomplete results exit 1, and a safe budget stop exits
2. WER/CER use the repository's Persian ASR normalization, but raw references
and predictions are always preserved for auditability.

```bash
uv run python -m ml.asr.eval_openrouter_stt \
  --dataset-root data/mixed-persian-test \
  --model openai/whisper-large-v3 \
  --model openai/gpt-4o-mini-transcribe \
  --max-run-cost-usd 5.00 \
  --min-key-remaining-usd 1.00 \
  --output-dir artifacts/openrouter-stt/mixed-persian-test \
  --resume
```

OpenRouter's current-key endpoint is queried before each transcription. The log
therefore shows the key's cumulative usage and remaining key limit alongside
this evaluator's independently checkpointed run cost.

### Re-score OpenRouter output with strict normalization

Recompute WER/CER from an existing OpenRouter evaluation without making API
requests:

```bash
uv run python -m ml.asr.rescore_openrouter_stt \
  --output-dir artifacts/openrouter-stt/mixed-persian-test
```

The command always reads the raw `reference` and `prediction` values from
`predictions.jsonl`; it does not reuse the evaluator's existing normalized
fields. Both sides receive NFKC and Persian character normalization, followed
by removal of every Unicode punctuation (`P*`) and format (`Cf`) character.
The latter covers zero-width non-joiner, zero-width joiner, zero-width space,
word joiner, and legacy zero-width no-break-space representations of
half-spaces. Whitespace is then collapsed.

The original evaluation artifacts remain unchanged. In
`predictions_strict_normalized.jsonl`, both the `reference`/`prediction` fields
and their `*_normalized` aliases contain only the strictly normalized text;
the raw values remain available in the original `predictions.jsonl`.
Per-example WER/CER is included alongside the text, while corpus WER/CER grouped
by model and `source_dataset` is written to `metrics_strict_normalized.json`.

## Evaluate ElevenLabs Scribe v2

Evaluate ElevenLabs Scribe v2 on the same mixed Persian test dataset:

```bash
export ELEVENLABS_API_KEY='YOUR_DEDICATED_EVALUATION_KEY'

uv run python -m ml.asr.eval_elevenlabs_scribe \
  --dataset-root data/mixed-persian-test \
  --max-run-credits 10000 \
  --min-account-remaining-credits 1000 \
  --max-estimated-cost-usd 5.00 \
  --output-dir artifacts/elevenlabs-scribe/mixed-persian-test
```

The evaluator uses the synchronous `POST /v1/speech-to-text` endpoint with
`scribe_v2`, Persian (`fa`), temperature 0, seed 0, and audio-event tagging and
diarization disabled. This keeps the output focused on ASR text and makes the
run as reproducible as the service permits. The key is read only from
`ELEVENLABS_API_KEY`.

ElevenLabs' subscription endpoint is queried before every request and after
every successful transcription. The logs and prediction records retain the
exact account credit counters (`character_count` is the API's legacy field name
for credits), included-credit limit and remaining balance, observed per-request
credit change, tier, and current overage. `--max-run-credits` stops when account
credit growth since the run began reaches the cap; one in-flight request can
cross this local cap. `--min-account-remaining-credits` is an optional reserve.
For the strongest protection, create a dedicated ElevenLabs API key with its
own credit quota in the ElevenLabs dashboard; that quota is enforced remotely.

The STT response does not report exact request cost in USD. The evaluator
therefore computes and clearly labels an estimate from each clip's duration.
`--price-per-hour-usd` defaults to the current public Scribe v2 API price of
`0.22`, and should be overridden if your contract or current price differs.
Before sending a clip, `--max-estimated-cost-usd` checks its projected cost, so
this estimate-based cap is not crossed by an in-flight request.

`predictions.jsonl` preserves the exact raw prediction, raw provider response,
response request IDs, reference, normalized scoring strings, source dataset,
per-clip WER/CER, duration, estimated cost, and credit snapshots.
`predictions.tsv` provides the main fields in tabular form. `metrics.json`
reports corpus-level WER/CER, duration, estimated cost, and observed credit
change overall and for every `source_dataset`; `events.jsonl` and
`logs/elevenlabs_scribe.log` provide machine-readable and live operational
logs. Exit codes are 0 for complete, 1 for incomplete/API failures, and 2 for a
safe budget stop.

Resume an interrupted or budget-stopped run after raising either local cap:

```bash
uv run python -m ml.asr.eval_elevenlabs_scribe \
  --dataset-root data/mixed-persian-test \
  --max-run-credits 20000 \
  --min-account-remaining-credits 1000 \
  --max-estimated-cost-usd 10.00 \
  --output-dir artifacts/elevenlabs-scribe/mixed-persian-test \
  --resume
```

The dataset, manifest, model, language, seed, and price assumption must match
the original run. Completed clips are skipped safely. Local caps and the
remaining-credit reserve may be changed when resuming.

## Evaluate Ivira Avanegar

Evaluate Avanegar on the same mixed Persian test dataset using its
[documented synchronous short-audio API](https://api.ivira.ai/partai/avanegar?type=document):

```bash
export IVIRA_GATEWAY_TOKEN='YOUR_DEDICATED_EVALUATION_TOKEN'

uv run python -m ml.asr.eval_ivira_avanegar \
  --dataset-root data/mixed-persian-test \
  --model default \
  --max-run-units 10000 \
  --output-dir artifacts/ivira-avanegar/mixed-persian-test
```

The token is read only from `IVIRA_GATEWAY_TOKEN`. The evaluator sends clips
sequentially to `POST /avanegar/avanegar/request`, and rejects clips of 60
seconds or longer before upload because the documented synchronous endpoint is
for audio below one minute. The request always sends both `punctuation=false`
and `spokenPunctuation=false`; these settings are fixed and cannot accidentally
be enabled from the CLI. SRT, timestamps, inverse normalization, diarization,
and speaker separation are also disabled.

Avanegar reports `units` in each successful API response. The exact response
and per-request units are checkpointed in `predictions.jsonl`, and cumulative
units are written to `metrics.json`, `events.jsonl`, and
`logs/ivira_avanegar.log`. `--max-run-units` is checked before every request.
Because the next request's units are unknown until it succeeds, one in-flight
request can cross the local cap. The public API document does not expose an
account-balance endpoint or a conversion from units to currency, so the script
does not invent a remaining-credit or USD figure. Use a dedicated gateway token
with a provider-side quota, if your Ivira account supports one, for a remote
hard cap.

`predictions.jsonl` preserves the exact raw prediction, complete raw provider
response, response IDs, reference, normalized scoring strings, source dataset,
per-clip WER/CER, duration, request options, and billed units.
`predictions.tsv` is the tabular view. `metrics.json` reports total and
per-`source_dataset` WER/CER, duration, and units. Exit codes are 0 for complete,
1 for incomplete/API failures, and 2 for a safe unit-cap stop.

Resume after interruption or after raising the local unit cap:

```bash
uv run python -m ml.asr.eval_ivira_avanegar \
  --dataset-root data/mixed-persian-test \
  --model default \
  --max-run-units 20000 \
  --output-dir artifacts/ivira-avanegar/mixed-persian-test \
  --resume
```

The dataset, manifest, model, duration limit, and fixed processing options must
match the original run. Completed clips are skipped; `--max-run-units` may be
raised when resuming.

## Create a Mixed Test Dataset

Randomly select an exact number of clips from multiple datasets' `test.tsv`
files, using a reproducible seed and relative proportions:

```bash
uv run python -m ml.speech_data.scripts.create_mixed_test_dataset \
  --dataset AGFarsdat_test_normalized data/AGFarsdat_test_normalized 1 \
  --dataset cv-corpus-25.0 data/cv-corpus-25.0 1 \
  --dataset fleurs-normalized data/fleurs-normalized 1 \
  --dataset PersianSpeech_test data/PersianSpeech_test 1 \
  --dataset persian-speech-corpus-test data/persian-speech-corpus-test 1 \
  --count 1000 \
  --seed 42 \
  --output-root data/mixed-persian-test
```

Each `--dataset` takes a unique safe name, its dataset directory, and a
positive relative weight. Weights are normalized, so `70 20 10` is equivalent
to `0.7 0.2 0.1`; the example gives all five datasets equal weight. Integer
sample counts use the largest-remainder method and always sum to `--count`; the
command fails before creating output if a source does not contain its allocated
number of rows.

The output contains `test.tsv` with `path`, `sentence`, and `source_dataset`
columns. Selected audio is copied under `clips/NAME/`, preserving its path
within the source `clips/` directory. Both `path` styles accepted by the project
dataset contract (`file.wav` and `clips/file.wav`) are supported. Existing
output is rejected unless `--overwrite` is passed.

## Compute Total Audio Hours

Recursively total the duration of all FLAC, WAV, and MP3 files in a directory:

```bash
uv run python -m ml.speech_data.scripts.compute_audio_hours \
  data/my-audio-directory \
  --workers 8
```

The command uses Mutagen to read duration metadata without decoding audio and
uses the requested number of worker processes. It logs each completed file and
the running total. The final summary is printed in hours and seconds. Corrupt or
unreadable supported files are logged, omitted from the total, and cause a
non-zero exit status after all other files have been inspected.

## Convert an ASR Dataset to FLAC

Create a new copy of a convention-style ASR dataset in which every clip
referenced by the selected split TSVs is stored as lossless FLAC:

```bash
uv run python -m ml.speech_data.scripts.convert_dataset_to_flac \
  --source-root data/my-dataset \
  --output-root data/my-dataset-flac
```

The source dataset is not changed. By default, the converter includes every
existing standard split (`train.tsv`, `dev.tsv`, and `test.tsv`), preserves all
TSV columns, rewrites only the `path` values, and copies non-clip dataset
metadata. Only referenced clips are copied, so orphaned audio does not consume
space in the output. Shared clips are encoded once. During conversion, stdout
reports the current clip plus its input size, FLAC size, per-file savings, and
cumulative savings. The default `PCM_16` FLAC
subtype is appropriate for typical speech datasets; use `--subtype PCM_24` if
the source contains audio with more than 16 bits of meaningful precision. Use
`--overwrite` to deliberately replace an existing output directory.

Verify a completed conversion against its original dataset:

```bash
uv run python -m ml.speech_data.scripts.verify_flac_conversion \
  --source-root data/my-dataset \
  --converted-root data/my-dataset-flac
```

The verifier logs every selected split file, referenced audio pair, and copied
metadata file as it checks it. It requires matching TSV columns and rows (apart
from the expected FLAC path rewrite), byte-identical metadata, FLAC output, and
matching audio frame count, channel count, sample rate, and decoded samples.
Decoded samples may differ only by one least-significant bit at the output
FLAC's PCM bit depth. Missing and unexpected output files fail verification.
Pass the same `--splits` selection used for conversion when only a subset was
converted. A successful command exits with status 0; a mismatch exits with
status 1 and lists all detected failures.

## Publish a refined audio dataset to Hugging Face

See [`persian-audiobook-hugging-face-publication.md`](persian-audiobook-hugging-face-publication.md)
for the recorded initial release, access policy, resumable-upload operations,
and the checklist for publishing the later complete configuration.

The uploader publishes a `path` / `sentence` TSV as standard Parquet audio
shards. Each Parquet row contains a typed Hugging Face `Audio` value plus `id`,
`sentence`, and available segmentation metadata. This avoids the Hub's
recommended 100,000-file repository and 10,000-entry directory thresholds when
the complete IranSeda collection is published.

Only one bounded shard is assembled in the temporary directory at a time. It
is uploaded, checkpointed, and immediately deleted, so the command never needs
a second full copy of the dataset. The default 512 MiB source-audio target uses
roughly that amount of temporary disk plus Parquet overhead.

First extract exact statistics and a Markdown table for the dataset card:

```bash
uv run python -m ml.speech_data.scripts.summarize_hf_audio_dataset \
  --manifest data/iranseda/250h-refinement-reports/refined_transcription.tsv \
  --audio-root data/iranseda/audiobooks/segmented \
  --segments-manifest data/iranseda/audiobooks/segmented/segments.jsonl \
  --refinement-summary data/iranseda/250h-refinement-reports/refinement_summary.json \
  --output data/iranseda/250h-refinement-reports/hf_dataset_summary.json \
  --card-snippet-output data/iranseda/250h-refinement-reports/hf_card_statistics.md \
  --workers 8
```

Validate the upload without contacting Hugging Face:

```bash
uv run python -m ml.speech_data.scripts.upload_hf_audio_dataset \
  --manifest data/iranseda/250h-refinement-reports/refined_transcription.tsv \
  --audio-root data/iranseda/audiobooks/segmented \
  --segments-manifest data/iranseda/audiobooks/segmented/segments.jsonl \
  --repo-id USER_OR_ORG/REPOSITORY \
  --config-name refined-subset \
  --state-file data/iranseda/hf-upload/refined-subset.state.json \
  --temp-dir data/iranseda/hf-upload/tmp \
  --dry-run
```

For a deliberately slow resumable upload, put a write-scoped token in
`HF_TOKEN` without adding it to the command line, then upload a bounded number
of shards per invocation:

```bash
uv run python -m ml.speech_data.scripts.upload_hf_audio_dataset \
  --manifest data/iranseda/250h-refinement-reports/refined_transcription.tsv \
  --audio-root data/iranseda/audiobooks/segmented \
  --segments-manifest data/iranseda/audiobooks/segmented/segments.jsonl \
  --repo-id USER_OR_ORG/REPOSITORY \
  --config-name refined-subset \
  --state-file data/iranseda/hf-upload/refined-subset.state.json \
  --temp-dir data/iranseda/hf-upload/tmp \
  --target-shard-mib 512 \
  --row-group-size 100 \
  --max-workers 1 \
  --sleep-seconds 10 \
  --max-shards 1 \
  --card docs/huggingface/PersianAudiobook/README.md \
  --card-asset Thesis/figs/long-audio-data-creation-pipeline.png
```

The checkpoint is written atomically after every successful commit. Rerunning
the same command skips completed shards. `--max-gib` can bound transferred
source audio instead of, or together with, `--max-shards`. Use `--create-repo --private` for
the first run if the target does not yet exist; publish privately until the
redistribution license and dataset card have been reviewed. Add `--card
path/to/README.md` once the card is ready.

The manifest checksum, segmentation-manifest checksum, shard size, row-group
size, config, and split form an immutable shard plan. Keep the state file for
resumes. When the complete refined TSV is available, use the same repository
but a new config and state file, for example `--config-name refined-full` and
`--state-file data/iranseda/hf-upload/refined-full.state.json`. The dataset card
YAML should list both configs and their `CONFIG/train-*.parquet` patterns. A
random 250-hour subset is not a prefix of the full manifest, so trying to append
to its existing Parquet shards would silently duplicate or reorder examples;
the uploader rejects that unsafe reuse.

On the current IranSeda server, prepend `/home/user01/.local/bin` to `PATH` or
invoke `/home/user01/.local/bin/uv` directly because non-interactive SSH shells
do not include that directory.

For the initial manually gated `PersianAudiobook` research upload, authenticate
on the server first. The uploader makes the repository public but configures
manual gating before committing any card or data file. Then start the prepared
launcher in a named, logged `screen` session:

The standard Hugging Face gate collects each requester's username and email.
One additional single-line field asks for their intended use and research
purpose. Keep approval in manual mode, contact each requester using that
information, verify their permission, and only then accept or reject the
pending request.

```bash
cd /home/user01/MS-Thesis
/home/user01/MS-Thesis/.venv/bin/hf auth login
mkdir -p data/iranseda/hf-upload
screen -L -Logfile data/iranseda/hf-upload/screen.log \
  -S persian-audiobook-upload \
  bash ml/speech_data/scripts/upload_persian_audiobook_subset.sh \
  OWNER/PersianAudiobook
```

Detach with `Ctrl-A`, then `D`. Reattach with
`screen -r persian-audiobook-upload`. If the process is interrupted after a
successful shard commit, rerun the same launcher: its state file skips every
checkpointed shard. The launcher deliberately refuses a different repository
name or a card missing the gated-research restriction. The generic Python uploader remains
available for the later complete manifest with a new config and state file.

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
[`scraper_guides/iranseda-scrapers.md`](scraper_guides/iranseda-scrapers.md)
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
artifacts. See [`long-audio-asr-pipeline-guide.md`](long-audio-asr-pipeline-guide.md)
for the complete artifact and dataset-safety contract.

Discovery prints live category/station/item progress and writes atomic
checkpoints during the crawl. Audiobooks checkpoint every 10 processed books
by default; radio checkpoints every station-day. Set `--checkpoint-every 1` for
the most frequent audiobook checkpoints, or increase the value to reduce disk
writes. Reruns skip completed work recorded in `discovery_checkpoints.jsonl`;
use `--refresh` to deliberately revisit it. The downloader checkpoints and logs
every selected audio item.

## Common Voice Persian Download

Download the Mozilla Data Collective Common Voice Persian archive. Set `MOZILLA_DATA_COLLECTIVE_API_KEY` in the environment or `.env` first:

```bash
uv run python -m ml.speech_data.scripts.download_common_voice_fa \
  --output-dir data
```

## FLEURS Persian Download

Download and export the Persian FLEURS subset from Hugging Face:

```bash
uv run python -m ml.speech_data.scripts.download_fleurs_persian \
  --output-root data/fleurs/fa_ir/source
```

## PerSets YouTube and Filimo Downloads

Download the public PerSets Persian ASR metadata and tar shards from Hugging
Face. The download commands keep the upstream tar files intact and resume from
valid locally cached files by default:

```bash
uv run python -m ml.speech_data.scripts.download_youtube_persian_asr \
  --output-root data/youtube-persian-asr/source \
  --workers 4

uv run python -m ml.speech_data.scripts.download_filimo_persian_asr \
  --output-root data/filimo-persian-asr/source \
  --workers 4
```

Use `--force` to redownload cached artifacts and `--revision` to pin a Hugging
Face revision. The upstream repositories are large (approximately 23 GB for
YouTube and 36.7 GB for Filimo), and both contain raw, unvalidated
transcriptions.

## Persian Evaluation Set Download

Download Nawar Halabi's Persian Speech Corpus and the free `myaudio_tiny`
PersianSpeech release into a local cache. This step only downloads and validates
the upstream archive/metadata files; preparation into TSVs and WAV clips is a
separate step.

```bash
uv run python -m ml.speech_data.scripts.download_persian_eval_sets \
  --cache-dir data/downloads/persian_eval_sets
```

The script caches the upstream archives under `data/downloads/persian_eval_sets/`.
Use `--force` to redownload valid cached files. The default URLs point to the
public Persian Speech Corpus package, the public Google Drive `myaudio_tiny.tar.gz`
archive, and the PersianSpeech GitHub XLSX metadata file.

## Persian Evaluation Set Preparation

Extract the downloaded Persian evaluation archives and prepare both sources as
repo-style ASR test datasets with `test.tsv` and mono 16 kHz WAV clips under
`clips/`:

```bash
uv run python -m ml.speech_data.scripts.prepare_persian_eval_sets \
  --cache-dir data/downloads/persian_eval_sets \
  --source-root data/persian_eval_sets/source \
  --persian-speech-corpus-output-root data/persian-speech-corpus-test \
  --persian-speech-output-root data/PersianSpeech_test \
  --workers 4
```

The script parses `orthographic-transcript.txt` from Persian Speech Corpus and
the `audio`/`text` columns from PersianSpeech `myaudio_tiny.xlsx`. It normalizes
transcripts with the same Persian text rules as the other ASR preparation
scripts, but keeps rejected rows with raw text because these are test/evaluation
sets. Transcript rows whose referenced audio is absent from the downloaded
archive are skipped and reported as `missing audio rows` in the summary. Use
`--force` to replace prepared outputs and re-extract the source archives.

## Common Voice Preparation

Prepare Common Voice 25 Persian into normalized TSV files and mono 16 kHz WAV clips:

```bash
uv run python -m ml.speech_data.scripts.prepare_common_voice_25 \
  --source-root data/cv-corpus-25.0-2026-03-09/fa \
  --output-root data/cv-corpus-25.0 \
  --workers 4
```

## FLEURS Preparation

Prepare exported FLEURS Persian into normalized TSV files and mono 16 kHz WAV clips:

```bash
uv run python -m ml.speech_data.scripts.prepare_fleurs_persian \
  --source-root data/fleurs/fa_ir/source \
  --output-root data/fleurs/fa_ir/normalized \
  --workers 4
```

To retain every FLEURS row and transcription exactly as exported, and copy the
source WAV files without re-encoding them, pass `--no-normalize`. This mode only
maps `validation` to `dev` and creates the standard TSV/`clips/` layout:

```bash
uv run python -m ml.speech_data.scripts.prepare_fleurs_persian \
  --source-root data/fleurs/fa_ir/source \
  --output-root data/fleurs/fa_ir/structured \
  --no-normalize
```

## PerSets YouTube and Filimo Preparation

Prepare each downloaded PerSets corpus as a train-only project ASR dataset. The
commands normalize Persian transcripts with the shared project rules, discard
rejected or empty text, stream MP3 members from the source tar shards, and write
mono 16 kHz PCM-16 WAV files with a `train.tsv` containing `path` and
`sentence` columns:

```bash
uv run python -m ml.speech_data.scripts.prepare_youtube_persian_asr \
  --source-root data/youtube-persian-asr/source \
  --output-root data/youtube-persian-asr/normalized \
  --workers 4

uv run python -m ml.speech_data.scripts.prepare_filimo_persian_asr \
  --source-root data/filimo-persian-asr/source \
  --output-root data/filimo-persian-asr/normalized \
  --workers 4
```

Preparation requires `ffmpeg`. It skips existing non-empty WAV files so an
interrupted run can resume; pass `--force` to reconvert them. No `dev.tsv` or
`test.tsv` is produced, so use the project's existing evaluation datasets for
validation and testing. MP3 files that ffmpeg cannot decode are excluded from
`train.tsv` and recorded with the ffmpeg diagnostic in `failed_audio.tsv`.
Keep enough free disk space for the downloaded tar cache and the converted WAV
dataset.

## TSV Dataset Transcript Normalization

Copy an existing ASR dataset to a new directory and normalize the `sentence`
column in `train.tsv`, `dev.tsv`, and `test.tsv` with the same Persian text
rules used by the Common Voice 25 preparation script, including Unicode
punctuation removal:

```bash
uv run python -m ml.speech_data.scripts.normalize_tsv_dataset \
  --source-root data/my_dataset/raw \
  --output-root data/my_dataset/normalized
```

The output directory must be new unless `--overwrite` is passed. The script
copies the full source tree first, preserves TSV columns, rewrites normalized
transcriptions in place under the output directory, and discards rows whose
sentences are rejected by the Common Voice 25 normalization rules. By default it
normalizes whichever of `train.tsv`, `dev.tsv`, and `test.tsv` exist, so
test-only evaluation directories are supported.

## Long-Audio Variant Concatenation

Build a new ASR dataset of long utterances by concatenating short clips, to
correct the short-utterance length/emission prior that degrades FastConformer on
audio longer than the training clips. Concatenation happens **independently
within each split**, so no train/dev/test leakage is introduced. All parameters
come from a YAML config:

```bash
uv run python -m ml.speech_data.concatenate_long_variants \
  --config configs/long_variants.yaml
```

`variants_per_split` in the config is a mapping of split name to count, so each
split gets a **different number** of variants and only the listed splits are
processed:

```yaml
variants_per_split:
  train.tsv: 3000
  dev.tsv: 300
  test.tsv: 300
```

Each variant joins `min_clips`–`max_clips` short clips (until `target_min_sec`
is reached, capped by `max_duration_sec`), loudness-normalizes every segment,
inserts a `gap_sec` silence between them, and joins the transcripts. Clips are
drawn across speakers by default; set `speaker_column: client_id` to force
same-speaker joins. Generation is deterministic per `seed`, with full provenance
written to `long_variants_manifest.jsonl` and a `generation_report.json`
summary. Pass `--overwrite` to write into an existing output directory. The
output is a long-only dataset: combine or oversample it alongside the original
short dataset at train time rather than using it as a replacement.

Generation is parallelizable: set `workers` in the config (or `--workers N` to
override) to fan variant generation across processes. Output is byte-identical
regardless of worker count, since each variant is seeded independently by its
index. A per-split progress bar reports throughput.

## Degradation Asset Download

Download all DEMAND `*_16k.zip` noise archives:

```bash
uv run python -m ml.speech_data.scripts.download_degradation_assets \
  --noise-root data/speech_enhancement/assets/noise/DEMAND
```

To download, extract, validate, and write indexes in one step:

```bash
uv run python -m ml.speech_data.scripts.download_degradation_assets \
  --noise-root data/speech_enhancement/assets/noise/DEMAND \
  --manifest-dir data/speech_enhancement/manifests \
  --prepare-indexes
```

## Degradation Asset Preparation

Prepare DEMAND 16 kHz noise assets after downloading the archives. Place the DEMAND
`*_16k.zip` files under `data/speech_enhancement/assets/noise/DEMAND/`, then run:

```bash
uv run python -m ml.speech_data.scripts.prepare_degradation_assets \
  --noise-root data/speech_enhancement/assets/noise/DEMAND \
  --manifest-dir data/speech_enhancement/manifests
```

The script extracts local archives by default, validates readable audio, and writes:

```text
data/speech_enhancement/manifests/demand_noise_index.jsonl
```

## Speech Degradation Generation

Generate paired clean/degraded speech-enhancement data from a YAML config:

```bash
uv run python -m ml.speech_data.generate_degraded_pairs \
  --config configs/speech_enhancement/degradation.yaml
```

See `docs/speech-degradation-pipeline.md` for the full degradation chain, profile
semantics, metadata fields, and known limitations.

## Degraded-only ASR Dataset Generation

Generate a dataset-shaped directory with degraded-only clips and TSVs from an existing
TSV-based ASR dataset such as Common Voice 25:

```bash
uv run python -m ml.speech_data.generate_degraded_dataset \
  --config configs/speech_enhancement/cv25_degraded_dataset.yaml \
  --workers 4
```

The config selects the source dataset directory, output dataset directory, included
split TSVs, variations per sample, and worker count. `--workers` overrides
`dataset.workers` for the current run. The output keeps `train.tsv`, `dev.tsv`,
`test.tsv`, or any selected TSV names, writes degraded WAV files under `clips/`, and
records clean-to-degraded traceability in `degraded_to_clean.jsonl`. Full per-variant
degradation metadata is also written to `degradation_metadata.jsonl`.

## Noise-only ASR Dataset Generation

Create two noise-added variants of every `train.tsv` and `dev.tsv` sample without
codec simulation, packet loss, filtering, random gain, clipping, or AGC:

```bash
uv run python -m ml.speech_data.generate_noise_added_dataset \
  --config configs/speech_enhancement/noise_added_dataset.yaml \
  --workers 4
```

Every output contains exactly one DEMAND noise scene. The default config selects
the 0–5, 5–10, 10–15, and 15–20 dB SNR buckets with equal probability, then
samples the exact SNR uniformly within the selected bucket. Output TSVs preserve
the source columns, while `degraded_to_clean.jsonl` and `noise_metadata.jsonl`
record the source clip, noise asset, scene, seed, SNR bucket, and exact SNR.

## Random Degraded Clip Demo

Generate several degraded variants of one random readable audio clip found under `data/`.
The output folder contains the selected clean target, degraded WAV files, a JSONL manifest,
and a JSON report:

```bash
uv run python -m ml.speech_data.scripts.generate_random_degraded_clip \
  --input-root data \
  --output-dir data/speech_enhancement/random_clip_degradations \
  --variants 8 \
  --seed 1337
```

## Manifest Inspection

Inspect a generated speech-enhancement manifest:

```bash
uv run python -m ml.speech_data.inspect_manifest \
  data/speech_enhancement/manifests/se_train_pairs.jsonl
```

## Degraded Dataset Validation

Check that a degraded dataset's noisy/clean pairs are actually trainable — the enhancer is trained to map the noisy clip to the *reconstructed* bandwidth-aligned clean target, so misaligned, mislabeled, or no-op pairs silently corrupt `L_enh`. This validates pair **consistency** (it does not re-run the codec/network degradation, which is not bit-reproducible) across four axes: **alignment** (cross-correlation lag between noisy and reconstructed clean), **degradation magnitude** (waveform SNR + mel L1, by `target_bandwidth`/channel/codec), **bandwidth consistency** (fraction of the noisy clip's energy above the recorded channel cutoff — ~0 for narrowband / wideband-filtered), and **metadata completeness** (the fields the target reconstruction needs). It flags clips whose lag reaches `--max-lag-ms`, whose degradation is a near no-op (`--noop-rel-l2`), whose band-limiting is violated (`--hf-tolerance`), or with missing metadata:

```bash
uv run python -m ml.speech_data.validate_degraded_dataset \
  --dataset data/cv-corpus-25.0-degraded-v2 \
  --sample 300 --output-dir artifacts/degraded_validation
```

`--dataset` is a degraded `generate_degraded_dataset` directory (repeatable). `--split` defaults to all splits; `--sample N` randomly checks N pairs per dataset (0 = all) under `--seed`. `--clean-target` matches the training target mode. `--output-dir` writes `validation.json` (per-dataset flag counts, overall + per-`target_bandwidth` distributions, and up to 50 flagged examples for inspection). A high `misaligned`/`bandwidth_mismatch` count means the data, not the model, is the problem; a high `near_noop` count or low SNR means the degradation is too weak for enhancement to have any headroom.

## Whisper-small Training

Fine-tune Whisper-small from the training config. Outputs go under the configured run directory unless `--run-dir` overrides it:

```bash
uv run python -m ml.asr.train_whisper_small \
  --config configs/whisper_small_train.yaml \
  --resume auto
```

Set `model.pretrained_model` to start from an existing local model directory, such as a previous run's `final` or `best` directory. Leave it empty to start from `model.name`, which defaults to `openai/whisper-small`.

Each configured dataset contributes `train.tsv` to training and `dev.tsv` to
evaluation when those splits are present. A missing split is skipped for that
dataset, and `test.tsv` is not used during training. Across the configured
datasets, at least one must provide usable training rows and at least one must
provide usable development rows.

Audio files with unreadable headers are skipped before training and recorded in
`manifests/skipped_unreadable_train.jsonl` or
`manifests/skipped_unreadable_dev.jsonl`. If an audio file passes that header
check but fails during full decoding, its path is logged and the loader
substitutes the next readable example instead of stopping the run.

## Whisper-small Evaluation

Run a saved Whisper-small checkpoint on the configured dataset `test.tsv` files. Outputs include `metrics.json`, `predictions.jsonl`, the effective config, logs, and a source manifest. `metrics.json` reports aggregate WER/CER and a `dataset_metrics` list with WER/CER per dataset directory:

```bash
uv run python -m ml.asr.eval_whisper_small \
  --config configs/whisper_small_eval.yaml
```

Set `model.checkpoint` to the local model/checkpoint path to evaluate. `model.processor` defaults to `openai/whisper-small`; point it at a saved `final`/`best` model directory only if you intentionally changed processor/tokenizer files. Set `data.datasets` to the dataset directories whose `test.tsv` files should be evaluated. Samples whose transcript token count exceeds `eval.max_label_tokens` are skipped before prediction; by default this should match Whisper-small's 448-token decoder limit. Keep `eval.eval_accumulation_steps` low, such as `1`, so generated prediction tensors are moved off GPU during long evaluations instead of accumulating until the end.

## FastConformer-CTC Training

Fine-tune the standalone FastConformer-CTC Persian model (the CTC branch of `nvidia/stt_fa_fastconformer_hybrid_large`, reimplemented under `ml/fa_fastconformer/` with no NeMo dependency) on the configured dataset `train.tsv` / `dev.tsv` files. Because the standalone model is a plain `nn.Module` rather than a Hugging Face model, training runs through a small hand-written PyTorch loop (CTC loss, AdamW, linear warmup schedule, gradient accumulation, optional AMP) instead of `transformers.Trainer`. The run layout mirrors the Whisper trainer — `status.json`, `logs/train.log`, `logs/train_metrics.jsonl`, the effective config, source manifests, rolling `checkpoints/checkpoint-<step>.pt` bundles, plus `final.pt` and `best.pt`:

```bash
uv run python -m ml.asr.train_fastconformer \
  --config configs/fastconformer_train.yaml \
  --resume auto
```

Set `model.checkpoint` to either the original `.nemo` archive or a converted `.pt` bundle to fine-tune from — the format is chosen by file extension (use `ml/fa_fastconformer/convert.py` to produce the `.pt` bundle; see the evaluation section below). Every checkpoint and the `final`/`best` models are written as the same self-contained `.pt` bundle that `eval_fastconformer` loads, so a trained checkpoint can be evaluated directly by pointing `fastconformer_eval.yaml`'s `model.checkpoint` at it. Resume state (optimizer, scheduler, AMP scaler, step) is stashed inside each rolling checkpoint bundle, so `--resume auto` (or `run.resume: auto`) continues from the latest one. Set `training.freeze_encoder: true` to train only the CTC head. Stop with Ctrl+C after a checkpoint exists, then re-run with `--resume auto` to continue.

Clips outside `data.min_duration_sec` / `data.max_duration_sec` (default `0.1`–`20.0`) are dropped from both the train and dev splits before batching — durations come from the audio header only (no decoding). Conformer self-attention costs O(T²) memory per layer, so without an upper cap a single multi-minute utterance (common in spontaneous-speech corpora) can OOM the GPU even when typical fixed-size batches fit comfortably. Raise `data.max_duration_sec` to keep longer clips (watch GPU memory), or set it to `null` to disable the cap. The trainer also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (unless already set) to reduce allocator fragmentation across the variable-length batches.

## Mixed-Dataset Local ASR Comparison

Evaluate several local model families in one run against a dataset created by
`ml.speech_data.scripts.create_mixed_test_dataset`:

```bash
uv run python -m ml.asr.eval_mixed_dataset \
  --config configs/mixed_asr_eval.yaml
```

The top-level config contains `dataset_root`, `output_dir`, and a repeatable
`models` list. Every model entry has a unique filesystem-safe `name`, a `type`,
and the path to that model's normal evaluator config:

```yaml
dataset_root: data/mixed-persian-test
output_dir: artifacts/asr-mixed-eval/mixed-persian-test
models:
  - name: whisper-small
    type: whisper_small
    config: configs/whisper_small_eval.yaml
  - name: whisper-medium
    type: whisper_medium
    config: configs/whisper_medium_eval.yaml
  - name: fastconformer
    type: fastconformer
    config: configs/fastconformer_eval.yaml
  - name: fusion
    type: fusion
    config: configs/speech_enhancement/fusion_eval.yaml
```

Supported types are `whisper_small`, `whisper_medium`,
`whisper_large_v3_turbo`, `fastconformer`, and `fusion` (hyphenated aliases are
also accepted). The referenced model config remains the source of checkpoint,
processor, device, batch-size, and generation settings; only its `data.root_dir`,
`data.datasets`, and `data.split` fields are adapted to the mixed test set. This
means checkpoints do not need to be copied and each underlying evaluator still
writes its usual logs, effective config, manifest, metrics, and predictions.
After every model finishes (including a failed evaluation), the runner collects
unreachable Python objects and clears PyTorch's CUDA cache and IPC allocations
before loading the next model.

The runner joins predictions back to the mixed manifest by resolved audio path,
not by row position. Repeated rows for the same audio are supported when their
source and reference agree; conflicting labels for one audio are rejected as
ambiguous. It adds `source_dataset`, the original and normalized reference, and
the normalized hypothesis to every model's `predictions.jsonl`.
For fair cross-model comparison, the new scores consistently use the repository's
Persian ASR normalization. `<output_dir>/summary.json` and `summary.tsv` contain
overall and per-source-dataset WER/CER for every model. Each model directory also
gets `source_metrics.json`, and its original `metrics.json` gains a
`source_dataset_metrics` block without losing the evaluator-specific metrics.
The command returns a nonzero status if an underlying evaluator did not produce a
prediction for every mixed-dataset row (for example, because a Whisper label was
over its configured token limit).

## FastConformer-CTC Evaluation

Evaluate the standalone FastConformer-CTC Persian model (the CTC branch of `nvidia/stt_fa_fastconformer_hybrid_large`, reimplemented under `ml/fa_fastconformer/` with no NeMo dependency) on the configured dataset `test.tsv` files. Outputs match the Whisper eval layout: `metrics.json` (aggregate WER/CER plus a `dataset_metrics` list per dataset directory), `predictions.jsonl`, the effective config, logs, and a source manifest:

```bash
uv run python -m ml.asr.eval_fastconformer \
  --config configs/fastconformer_eval.yaml
```

Set `model.checkpoint` to either the original `.nemo` archive or a converted `.pt` bundle — the format is chosen by file extension. To produce the `.pt` bundle (CTC weights + config + tokenizer, repacked from the `.nemo` so loading needs neither a tar unpack nor NeMo), run the standalone converter from inside the package directory:

```bash
cd ml/fa_fastconformer
python convert.py /path/to/stt_fa_fastconformer_hybrid_large.nemo models/stt_fa_fastconformer_ctc.pt --verify
```

Greedy CTC decoding has no decoder token limit, so there is no `max_label_tokens` skipping. Batching is duration-aware: clips are sorted by length and each batch is capped by both `eval.batch_size` and `eval.max_batch_seconds`, so the heaviest batch costs about one clip of that many seconds and a few long clips cannot exhaust GPU memory. Raise `eval.batch_size` to speed up short-clip throughput; lower `eval.max_batch_seconds` if you still hit out-of-memory on long clips (set it to `null` to disable the cap and use fixed-size batches).

## Enhancement + Fusion Curriculum Training

Run the 3-stage enhancement+fusion curriculum (Stage 0 enhancer warm-up → Stage 1 enhancer+fusion with Whisper frozen → Stage 2 joint end-to-end) from a single YAML config, writing every artifact to one run directory:

```bash
uv run python -m ml.fusion.train_fusion \
  --config configs/speech_enhancement/fusion_train.yaml \
  --resume-from-stage 0
```

The trainer consumes one or more datasets listed under `datasets` (each entry is a path string with its kind auto-detected, or a `{path, kind}` mapping with `kind` in `degraded`/`clean`; the legacy single `dataset_dir` is still accepted as one degraded dataset when `datasets` is null). A **degraded** dataset is a `ml.speech_data.generate_degraded_dataset` directory (`degraded_to_clean.jsonl`): each degraded clip becomes a noisy Whisper log-Mel and the bandwidth-aligned clean log-Mel target is reconstructed from the recorded degradation metadata (`clean_target: bandwidth_aligned`, or `full_band` to target the raw clean). At least one degraded dataset is required, and degraded datasets drive every stage. A **clean** dataset is a plain (non-degraded) ASR dataset following the project split-TSV + `clips/` contract; it is folded into the **joint stage only**, where the noisy and clean views are the same clean log-Mel (`L_enh` → 0) so its undegraded audio fine-tunes the full stack and keeps it strong on clean speech without regressing the degraded result. Dev WER/CER is measured on the degraded datasets' `valid_split` (the metric the fusion system must beat and selects `best.pt` on); the joint stage additionally evaluates the clean datasets' `valid_split` and logs those as `clean_wer`/`clean_cer`/… so clean-speech regression is visible without driving checkpoint selection. Clean audio (resampled to `sample_rate`) contributes to joint-stage training and joint-stage eval only. The enhancer architecture is selected by the `enhancer` block (default `residual_unet`, a lightweight residual 2D-conv U-Net that starts as the identity). Stage 0 trains the enhancer alone on the log-Mel L1 loss `L_enh`. Stages 1–2 build the encoder-feature-space fusion model (`ml/fusion/model.py`): the noisy and enhanced log-Mels are each run through the shared Whisper encoder and the two hidden-state streams are merged by the fusion block (`fusion` config block, default `cross_attention` — bidirectional cross-attention so the streams exchange context before a gated merge; `gated` is the lightweight element-wise baseline) before the decoder, optimising `L_ASR + lambda * L_enh` — Stage 1 with the backbone frozen, Stage 2 end to end (the backbone is initialised from the fine-tuned Persian Whisper at `base_asr_checkpoint`). The enhancer can optionally add a long-range **temporal bottleneck** via the `enhancer` block (`bottleneck: transformer` or `gru`, sized by `bottleneck_layers`/`bottleneck_heads`/`bottleneck_dim`/`bottleneck_dropout`; default `none`) — a sequence model at the U-Net bottleneck that supplies the temporal context the purely-convolutional path lacks. It is identity-initialised, so enabling it does not perturb the Stage 0 identity-init. **ASR-aware warm-up:** set the warm-up stage's `feature_match_weight` > 0 to add `feature_match_weight * L1(encoder(enhanced), encoder(clean))` to the Stage 0 objective — an encoder-feature-space (perceptual) loss far more correlated with WER than raw mel L1 — with `lambda` then weighting `L_enh`. This needs `base_asr_checkpoint` (the same Whisper encoder Stages 1–2 use) and forces full 30 s windows (the Whisper encoder cannot encode the cheap short crops), so `segment_seconds` is ignored when it is on; `L_feat` and the combined `L_warmup` are logged and `L_warmup` drives Stage 0 best-checkpoint selection. Each stage validates on `valid_split` every `eval_every` steps — Stage 0 by dev `L_enh` (or `L_warmup` when feature matching is on), Stages 1–2 by dev WER/CER decoded through the fused encoder (capped at `eval_max_batches`, decoder steered by `language`/`task`/`generation_max_length`) — logging dev metrics to `logs/eval_metrics.jsonl` and keeping the best-scoring weights as `best.pt`; eval is skipped automatically when no usable dev split exists. Each stage writes a checkpoint under `checkpoints/stage{0,1,2}_*/` (`enhancer.pt`, plus `fusion_model.pt` for Stages 1–2) and a rolling `last.pt` that carries optimizer/scaler state and the step, so re-invoking resumes an interrupted stage mid-way; `--resume-from-stage` (or `resume_from_stage` in the config; `0`/`1`/`2` or `warmup`/`fusion`/`joint`) restarts the curriculum at a later stage by loading the prior stage's checkpoint, so a Stage 2 crash never forces re-running the earlier stages. Stages 1–2 clip gradients at `grad_clip`; seeding (`transformers.set_seed` + seeded dataloaders) makes runs reproducible.

## Enhancement + Fusion Evaluation

Evaluate a trained dual-view fusion model as a whole ASR system on the configured dataset `test.tsv` files. Each clip's Whisper log-Mel is fed as the *noisy* view through the trained `DualViewFusionModel` (enhancer → shared Whisper encoder → cross-attention fusion → Whisper decoder) and `model.generate` decodes token ids from the fused encoder stream exactly as the trainer's dev eval does. Outputs match the Whisper/FastConformer eval layout: `metrics.json` (aggregate WER/CER plus a `dataset_metrics` list per dataset directory), `predictions.jsonl`, the effective config, logs, and a source manifest:

```bash
uv run python -m ml.fusion.eval_fusion \
  --config configs/speech_enhancement/fusion_eval.yaml
```

**Ablation modes.** `eval.view_mode` / `eval.gate_override` (or the `--view-mode {fusion,noisy,enhanced}` / `--gate-override FLOAT` CLI flags, which win over the config) re-route which encoder stream feeds the decoder so the fusion contribution can be isolated against the *same* checkpoint and test sets: `--view-mode noisy` runs the backbone encoder on the raw audio with the enhancer and fusion skipped (the "is the front end net overhead?" baseline), `--view-mode enhanced` is the enhancer-alone path (enhancer → encoder → decoder, fusion skipped, measuring whether the enhancer alone helps or hurts ASR), and `--gate-override g` (only with `view-mode fusion`) pins the fusion blend gate to a constant `g` in `[0, 1]` instead of the learned per-channel gate — `0.0` disables the enhanced stream, `1.0` the noisy stream — to test what the gate would contribute if it committed. The `noisy`/`enhanced` modes bypass the fusion block so `view_usage` is empty; under `gate_override` the reported view weight is the forced constant. Both knobs are echoed into `metrics.json` (`view_mode`, `gate_override`).

Set `model.checkpoint` to a fusion training checkpoint — normally the joint-stage final model `artifacts/…/checkpoints/stage2_joint/fusion_model.pt`, though `best.pt` or a Stage 1 checkpoint also load. The enhancer/fusion architecture is read back from the checkpoint, so it need not be repeated in the config. `model.base_asr_checkpoint` supplies the Whisper backbone *architecture* and `generation_config`. **Whether its weights matter depends on the checkpoint:** Stage 2 (joint) checkpoints carry the **jointly-trained backbone**, so it is loaded straight from the checkpoint and `base_asr_checkpoint`'s weights are overwritten — point `model.checkpoint` at the joint checkpoint and the trained backbone comes with it (keep `base_asr_checkpoint: openai/whisper-small` as a bare architecture skeleton). A Stage 1 rolling `last.pt` is **backbone-free** (its frozen backbone equals the base), so the backbone is taken wholly from `base_asr_checkpoint`; there you must set it to the fine-tuned Persian Whisper run dir used during training, not a vanilla baseline. The script logs which case applied and records `backbone_from_checkpoint` in `metrics.json`, warning when a backbone-free checkpoint falls back to `base_asr_checkpoint`. `model.processor` is the tokenizer used to decode (defaults to `model_name`), and `model.language`/`model.task` steer the decoder prompt on multilingual backbones. Set `data.datasets` to the dataset directories whose `test.tsv` files should be evaluated — a clean (non-degraded) set measures the fused stack as a drop-in ASR model, a degraded dataset's clip dirs measure robustness. Decoding is greedy with a `generation_max_length` token cap; raise `eval.batch_size` to speed up throughput, and `eval.mixed_precision` (`auto`/`true`/`false`) controls CUDA autocast.

## Enhancement Diagnosis

Measure how much an enhancer actually denoises, to settle whether a small `L_enh` means "good enhancer" or "the noisy mel was already close to clean". Reports the **headroom**: `identity_L_enh` (mel L1 of the do-nothing identity, i.e. noisy↔clean), `trained_L_enh` (mel L1 of enhanced↔clean for a checkpoint), and `captured` = `(identity − trained) / identity` (fraction of headroom removed; ~0 means the enhancer is inert). Results break down by degradation `target_bandwidth` (telephone/narrowband vs wideband) and by dataset:

```bash
uv run python -m ml.enhancement.diagnose_enhancement \
  --dataset data/cv-corpus-25.0-degraded-v2 --split dev \
  --enhancer-checkpoint artifacts/.../checkpoints/stage0_warmup/enhancer.pt \
  --feature-encoder models/asr/whisper-small/runs/best \
  --output-dir artifacts/enhancement_diagnosis
```

`--dataset` is a degraded `generate_degraded_dataset` directory (repeatable). Omit `--enhancer-checkpoint` to report the identity baseline only; pass an `enhancer.pt` or a `fusion_model.pt` (the `enhancer.*` weights are extracted) to evaluate a trained enhancer. `--feature-encoder` (a Whisper run dir or Hub id) additionally reports the same identity-vs-trained comparison in the **encoder feature space** (`identity_L_feat`/`trained_L_feat`/`captured_feat`) — the distance that actually tracks WER and the target of the warm-up feature-matching loss. `--dump-mels N` writes the first N clips' noisy/clean/enhanced log-Mels as `.npy` under `<output-dir>/mels/` for offline plotting (no plotting dependency). `--output-dir` writes `diagnosis.json` (overall + per-bandwidth + per-dataset); `--max-batches` caps work per dataset and `--batch-size`/`--device` control throughput.
