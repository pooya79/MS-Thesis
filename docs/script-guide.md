# Script Guide

Every maintained Python script exposes `--help`. Use the help output before running a script with custom paths:

```bash
uv run python -m ml.speech_data.scripts.download_common_voice_fa --help
uv run python -m ml.speech_data.scripts.download_degradation_assets --help
uv run python -m ml.speech_data.scripts.download_fleurs_persian --help
uv run python -m ml.speech_data.scripts.prepare_common_voice_25 --help
uv run python -m ml.speech_data.scripts.prepare_degradation_assets --help
uv run python -m ml.speech_data.scripts.prepare_fleurs_persian --help
uv run python -m ml.speech_data.scripts.generate_random_degraded_clip --help
uv run python -m ml.speech_data.generate_degraded_dataset --help
uv run python -m ml.speech_data.generate_degraded_pairs --help
uv run python -m ml.speech_data.inspect_manifest --help
uv run python -m ml.asr.train_whisper_small --help
```

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
  --config configs/speech_enhancement/cv25_degraded_dataset.yaml
```

The config selects the source dataset directory, output dataset directory, included
split TSVs, and variations per sample. The output keeps `train.tsv`, `dev.tsv`,
`test.tsv`, or any selected TSV names, writes degraded WAV files under `clips/`, and
records clean-to-degraded traceability in `degraded_to_clean.jsonl`. Full per-variant
degradation metadata is also written to `degradation_metadata.jsonl`.

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

## Whisper-small Training

Fine-tune Whisper-small from the training config. Outputs go under the configured run directory unless `--run-dir` overrides it:

```bash
uv run python -m ml.asr.train_whisper_small \
  --config configs/whisper_small_train.yaml \
  --resume auto
```

Set `model.pretrained_model` to start from an existing local model directory, such as a previous run's `final` or `best` directory. Leave it empty to start from `model.name`, which defaults to `openai/whisper-small`.
