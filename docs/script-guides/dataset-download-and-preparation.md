# Dataset Download and Preparation

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
