# Dataset Management

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

See [`persian-audiobook-hugging-face-publication.md`](../persian-audiobook-hugging-face-publication.md)
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
