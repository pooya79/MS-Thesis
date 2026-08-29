# Speech Degradation

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
