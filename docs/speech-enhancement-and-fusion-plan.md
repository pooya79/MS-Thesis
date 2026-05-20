# Speech Enhancement and Fusion Implementation Plan

This document translates the thesis method chapter into an implementation plan for this repository. It intentionally excludes Whisper ASR fine-tuning because the Persian Whisper checkpoint already exists. The remaining work is:

1. Build the paired clean/degraded speech-enhancement dataset.
2. Train or domain-adapt PrimeK-Net for Persian telecommunication-style degradation.
3. Train the log-Mel fusion network that combines noisy and enhanced features before the frozen Whisper model.

## Goals

The implementation should produce a reproducible pipeline that starts from clean Persian speech clips and ends with trained speech-enhancement and fusion checkpoints. The system should keep all generated artifacts traceable to source audio, augmentation parameters, and model configuration.

The target behavior is not to make enhancement universally better for every dataset. The expected finding from the thesis is more specific:

- Enhancement should help real or simulated telecommunication audio.
- Direct Whisper should remain strong on clean general-purpose speech.
- Fusion should reduce the failure mode where a pure enhancement cascade damages clean or out-of-domain speech, even if it does not always beat the best conditional choice.

## Proposed Repository Layout

Add the machine-learning implementation under a separate training package rather than mixing it with the FastAPI app.

```text
ml/
  speech_data/
    prepare_common_voice.py
    download_noise_assets.py
    generate_degraded_pairs.py
    inspect_manifest.py
  enhancement/
    primek_adapter.py
    train_primek.py
    eval_primek.py
  fusion/
    model.py
    dataset.py
    train_fusion.py
    eval_fusion.py
  asr/
    whisper_features.py
    whisper_scoring.py
  utils/
    audio.py
    manifests.py
    seed.py
    text_normalization.py
configs/
  speech_enhancement/
    data.yaml
    primek_train.yaml
    fusion_train.yaml
artifacts/
  .gitkeep
data/
  .gitkeep
```

`data/` and `artifacts/` should be ignored by Git except for placeholder files. The code and config files should be committed; generated audio, manifests, and checkpoints should not.

## External Inputs

Required input assets:

- Clean Persian speech: Common Voice Persian v21 train and validation splits.
- Room impulse responses: BUT ReverbDB.
- Background noise: DEMAND 16 kHz release.
- Frozen ASR checkpoint: the already-trained Persian Whisper checkpoint.
- Optional baseline enhancement checkpoint: official or compatible PrimeK-Net checkpoint.

The exact local paths should live in `configs/speech_enhancement/data.yaml`, not hard-coded in scripts.

Example configuration fields:

```yaml
common_voice_root: /path/to/common_voice/fa
but_reverbdb_root: /path/to/BUT_ReverbDB
demand_root: /path/to/DEMAND
work_dir: data/speech_enhancement
artifact_dir: artifacts/speech_enhancement
sample_rate: 16000
degraded_variants_per_clip: 2
seed: 1337
```

## Phase 1: Dataset Preparation

### 1. Normalize Common Voice Input

Implement `ml/speech_data/prepare_common_voice.py`.

Responsibilities:

- Read Common Voice Persian v21 metadata for train and validation splits.
- Keep source clip ID, source path, sentence, speaker/client ID when available, duration, and split.
- Convert all audio to mono 16 kHz PCM WAV or FLAC.
- Normalize transcript text using the same Persian normalization rules used for ASR scoring.
- Reject only objectively invalid clips: unreadable audio, zero duration, impossible sample rate conversion, or empty transcript.

Output:

```text
data/speech_enhancement/clean/
  train/
    <clip_id>.wav
  valid/
    <clip_id>.wav
data/speech_enhancement/manifests/
  clean_train.jsonl
  clean_valid.jsonl
```

Manifest row schema:

```json
{
  "id": "cv-fa-train-000001",
  "split": "train",
  "clean_path": "data/speech_enhancement/clean/train/cv-fa-train-000001.wav",
  "source_path": "/raw/common_voice/fa/clips/...",
  "duration_sec": 4.18,
  "sample_rate": 16000,
  "transcript": "..."
}
```

### 2. Index Noise and RIR Assets

Implement `ml/speech_data/download_noise_assets.py` only if automated download is desired. Otherwise implement an indexing mode that assumes the archives were downloaded manually.

Responsibilities:

- Scan BUT ReverbDB and build a manifest of RIR files.
- Scan DEMAND and build a manifest of noise files with scene labels.
- Validate that assets are readable and can be resampled to 16 kHz.

Output:

```text
data/speech_enhancement/manifests/
  rir_index.jsonl
  demand_noise_index.jsonl
```

## Phase 2: Degraded Pair Generation

Implement `ml/speech_data/generate_degraded_pairs.py`.

For every clean clip in train and validation, generate two degraded variants. Each output must remain time-aligned with the clean target and must record every random choice in metadata.

### Degradation Chain

Apply degradations in this order:

1. Load clean audio.
2. Convert to mono and choose the working source sample rate, usually 16 kHz or higher.
3. Optional RIR convolution.
4. Optional environmental/background noise mixing.
5. Optional talker/device level variation: gain shift, clipping, AGC, or other level effects.
6. Telephone channel simulation:
   - Narrowband path: resample to 8 kHz, band-limit around 300 to 3400 Hz, then encode/decode with G.711, GSM, AMR-NB, or a similar narrowband codec.
   - Wideband path: keep or resample to 16 kHz, band-limit roughly 50 to 7000 Hz, then encode/decode with AMR-WB, Opus wideband, or a similar wideband codec.
7. Optional network impairment:
   - Packet loss or burst loss before decoding when the selected codec path supports proper packet-level simulation.
   - Waveform dropout as a simpler approximation when packet-level simulation is unavailable.
8. Resample the final degraded input to the model input rate, normally 16 kHz.
9. Apply loudness or peak safety normalization.
10. Write the degraded input and clean target.

### RIR Simulation

Use total RIR probability `0.35`.

- Severe reverb: probability `0.10`, wet mix `0.6` to `0.8`, D/R `6` to `10` dB.
- Mild reverb: probability `0.25`, wet mix `0.3` to `0.5`, D/R `12` to `18` dB.

Prefer RIRs with speaker/session labels when available. Normalize the RIR before convolution and keep output length aligned to the clean clip.

### Noise Mixing

Use noise probability `0.90`.

- Choose one DEMAND scene by default.
- Add a second scene with probability `0.10`.
- Sample SNR from configured buckets spanning `+15` dB to `-5` dB.
- Loop or crop noise to match clean length.

The config should make SNR buckets explicit, for example:

```yaml
snr_buckets:
  - [10, 15]
  - [5, 10]
  - [0, 5]
  - [-5, 0]
```

### Level and Device Simulation

Apply level and device effects after acoustic noise and before the telephone channel.

Recommended first implementation:

- Gain shift sampled from `-6` to `+6` dB.
- Soft clipping or hard clipping with a low probability, such as `0.10` to `0.20`.
- Optional AGC with conservative settings and complete metadata logging.

Keep these effects configurable because aggressive clipping or AGC can easily dominate the training signal.

### Telephone Channel Simulation

Sample channel path and codec type from a configurable distribution. A reasonable starting distribution is:

- `30%` G.711 A-law.
- `30%` AMR-WB at 12.65 kb/s.
- `20%` AMR-NB at 12.2 kb/s or GSM.
- `10%` Opus, split between narrowband and wideband modes.
- `10%` pass-through.

The channel path controls the sample rate and bandwidth:

- Narrowband path:
  - Resample channel input to 8 kHz.
  - Band-limit around `300` to `3400` Hz.
  - Apply a narrowband codec such as G.711, GSM, or AMR-NB when selected.
- Wideband path:
  - Use 16 kHz channel audio.
  - Band-limit roughly `50` to `7000` Hz.
  - Apply a wideband codec such as AMR-WB or Opus wideband when selected.

For pass-through samples, still allow a channel filter with configurable probability so the model sees bandwidth limitation without codec artifacts.

After the channel simulation, resample the degraded input back to the model input rate, usually 16 kHz.

For narrowband paths, produce a bandwidth-aligned clean target by applying the same narrowband filtering to the clean reference and returning it to the model target rate. For wideband paths, keep the clean target at 16 kHz unless the wideband filter is intentionally part of the supervised target. This avoids training the enhancement model to hallucinate high-frequency content that the simulated channel removed.

### Network Impairment

If packet-level simulation is available for the selected codec, apply Gilbert-Elliott packet or burst loss before decoding with probability `0.60`.

Sample packet loss rate from:

- Light: `0.3%` to `2%`.
- Medium: `2%` to `5%`.
- Heavy: `5%` to `10%`.

If packet-level simulation is not available, use waveform dropout as a simpler approximation. The implementation should clearly record which mode was used, plus loss rate, burst parameters, frame size, and dropout duration.

### Deterministic Seeding

Each degraded variant must be reproducible from:

- Global seed.
- Split name.
- Clean clip ID.
- Variant index.

Implement this in `ml/utils/seed.py` using a stable hash, not Python's process-randomized `hash()`.

### Output Layout

```text
data/speech_enhancement/pairs/
  train/
    clean/
      <pair_id>.wav
    degraded/
      <pair_id>.wav
  valid/
    clean/
      <pair_id>.wav
    degraded/
      <pair_id>.wav
data/speech_enhancement/manifests/
  se_train_pairs.jsonl
  se_valid_pairs.jsonl
```

Pair manifest row schema:

```json
{
  "pair_id": "train_cv-fa-000001_v0",
  "split": "train",
  "source_clean_id": "cv-fa-000001",
  "clean_path": "data/speech_enhancement/pairs/train/clean/train_cv-fa-000001_v0.wav",
  "degraded_path": "data/speech_enhancement/pairs/train/degraded/train_cv-fa-000001_v0.wav",
  "target_bandwidth": "wideband",
  "model_sample_rate": 16000,
  "duration_sec": 4.18,
  "rir_id": "rir_...",
  "reverb_mode": "mild",
  "noise_scenes": ["cafeteria"],
  "snr_db": 4.6,
  "gain_db": -1.2,
  "clipping": {
    "enabled": false,
    "mode": null
  },
  "agc": {
    "enabled": false
  },
  "channel_path": "wideband",
  "channel_sample_rate": 16000,
  "channel_bandpass_hz": [50, 7000],
  "codec": "amr_wb_12k65",
  "network_impairment": {
    "enabled": true,
    "mode": "packet_loss",
    "model": "gilbert_elliott",
    "loss_rate": 0.031,
    "burst_length": 3,
    "frame_ms": 20,
    "dropout_ms": null
  },
  "normalization": "peak_safety",
  "seed": 123456789
}
```

### Validation Gates

Add a lightweight inspection command in `ml/speech_data/inspect_manifest.py` that reports:

- Number of pairs per split.
- Total hours per split.
- Duration distribution.
- Codec distribution.
- SNR distribution.
- Packet-loss distribution.
- Count of missing/unreadable files.
- Count of length mismatches.

This must run before training.

## Phase 3: PrimeK-Net Training

Implement `ml/enhancement/train_primek.py`.

The preferred path is to wrap the official PrimeK-Net implementation with a thin adapter instead of reimplementing the architecture from scratch. Keep local code responsible for data loading, configuration, checkpointing, logging, and evaluation.

### Training Data

Use:

- Input: degraded waveform.
- Target: clean waveform from the pair manifest.
- Train split: `se_train_pairs.jsonl`.
- Validation split: `se_valid_pairs.jsonl`.

The dataset loader should:

- Load both waveforms.
- Assert equal sample rate and length.
- Randomly crop or pad to the configured training segment length.
- Return tensors shaped as expected by PrimeK-Net.

### Training Configuration

Start with PrimeK-Net's published defaults. Avoid broad hyperparameter search during the first implementation.

Configuration should include:

```yaml
checkpoint_init: /path/to/primek_pretrained.pt
train_manifest: data/speech_enhancement/manifests/se_train_pairs.jsonl
valid_manifest: data/speech_enhancement/manifests/se_valid_pairs.jsonl
sample_rate: 16000
segment_seconds: 4.0
batch_size: 8
gradient_accumulation_steps: 1
epochs: 100
learning_rate: 0.0002
num_workers: 4
mixed_precision: true
save_every_epochs: 5
```

Exact values can change after confirming the official PrimeK-Net recipe and available GPU memory.

### Checkpoints and Logs

Write outputs to:

```text
artifacts/speech_enhancement/primek/
  checkpoints/
    epoch_001.pt
    best.pt
  logs/
    train_metrics.jsonl
  eval/
    valid_metrics.json
```

Each checkpoint should include:

- Model state.
- Optimizer state.
- Scheduler state if used.
- Config snapshot.
- Git commit hash when available.
- Manifest paths and manifest content hashes.

### PrimeK-Net Evaluation

Implement `ml/enhancement/eval_primek.py`.

At minimum, compute:

- PESQ where legally and technically available.
- SI-SDR or SDR.
- Optional STOI if dependency is acceptable.
- Runtime per second of audio.

Evaluation should write enhanced validation audio for a small fixed subset:

```text
artifacts/speech_enhancement/primek/samples/
  <pair_id>_degraded.wav
  <pair_id>_enhanced.wav
  <pair_id>_clean.wav
```

This makes auditory inspection reproducible.

## Phase 4: Fusion Network Training

Implement the fusion model after the enhancement checkpoint is usable.

The fusion stage has three frozen components:

- Frozen Persian Whisper checkpoint.
- Frozen Whisper feature extractor.
- Frozen trained PrimeK-Net enhancement checkpoint.

Only the fusion network parameters are trained.

### Fusion Dataset

Implement `ml/fusion/dataset.py`.

Each item should provide:

- Degraded waveform.
- Enhanced waveform generated by frozen PrimeK-Net.
- Transcript tokens for Whisper supervision.
- Optional metadata from the pair manifest.

For performance, support two modes:

1. `online_enhancement`: run PrimeK-Net inside the training loop.
2. `cached_enhancement`: precompute enhanced waveforms or log-Mel features before fusion training.

The first implementation should prefer cached enhancement because it reduces GPU memory pressure and makes fusion training easier to debug on an RTX 3090.

Suggested cached layout:

```text
data/fusion_cache/
  train/
    noisy_mel/
      <pair_id>.pt
    enhanced_mel/
      <pair_id>.pt
  valid/
    noisy_mel/
      <pair_id>.pt
    enhanced_mel/
      <pair_id>.pt
  manifests/
    fusion_train.jsonl
    fusion_valid.jsonl
```

### Fusion Model

Implement `ml/fusion/model.py`.

Inputs:

- `M_n`: noisy Whisper log-Mel, shape `[B, 80, T]`.
- `M_e`: enhanced Whisper log-Mel, shape `[B, 80, T]`.

Architecture:

1. Rearrange both inputs to `[B, 1, T, 80]`.
2. Project each stream to `C=64` channels with `1x1 Conv + BatchNorm + PReLU`.
3. Apply `N=4` ladder stages.
4. In each stage:
   - Apply residual attention block independently to enhanced and noisy streams.
   - Exchange information with gated interaction in both directions.
5. Down-project both streams to one channel.
6. Concatenate down-projected streams and original streams.
7. Produce a time-frequency fusion mask with local convolution and chunked temporal attention.
8. Compute convex fusion:

```text
X_f = X_e' * mask + X_n' * (1 - mask)
```

9. Return fused log-Mel as `[B, 80, T]`.

Implementation constraints:

- Keep attention chunk size configurable, default `256`.
- Keep channel count configurable, default `64`.
- Avoid changing Whisper internals.
- Include shape assertions in debug mode.
- Unit-test the model with random tensors before connecting Whisper.

### Fusion Training Objective

Implement `ml/fusion/train_fusion.py`.

Use Whisper's autoregressive cross-entropy loss:

```text
L_ASR = -sum_t log p(y_t | y_<t, M_f)
```

Whisper remains frozen. The optimizer updates only fusion network parameters.

Training loop responsibilities:

- Load cached noisy/enhanced log-Mel features.
- Run fusion network.
- Feed fused features to frozen Whisper.
- Compute token-level cross-entropy against transcript labels.
- Backpropagate only through fusion.
- Save checkpoints and validation metrics.

### Fusion Evaluation

Implement `ml/fusion/eval_fusion.py`.

Compare three inference modes on the same evaluation manifests:

1. Baseline: noisy audio directly to frozen Persian Whisper.
2. Cascade: PrimeK-Net enhanced audio to frozen Persian Whisper.
3. Fusion: noisy and enhanced log-Mels through fusion network, then frozen Persian Whisper.

Report:

- WER.
- CER.
- Per-dataset averages.
- Runtime and memory if feasible.

Keep text normalization identical across all modes.

Expected evaluation sets:

- Common Voice Persian v21 test.
- FLEURS Persian test.
- Persian Speech Corpus test.
- PersianSpeech test.
- Real telephone test set.

## Phase 5: Reproducibility and Tests

Add tests for the code paths that can be tested without full GPU training.

Recommended tests:

- Manifest schema validation.
- Stable deterministic seed generation.
- Audio pair length alignment.
- Degradation metadata completeness.
- Fusion model forward pass shape.
- Frozen-parameter check for Whisper and PrimeK-Net during fusion training.
- Text normalization consistency between training and scoring.

For long-running GPU jobs, add smoke-test commands that run on a tiny manifest with 2 to 4 clips.

Example commands to add later:

```make
prepare-se-data:
	uv run python -m ml.speech_data.prepare_common_voice --config configs/speech_enhancement/data.yaml

generate-se-pairs:
	uv run python -m ml.speech_data.generate_degraded_pairs --config configs/speech_enhancement/data.yaml

train-primek:
	uv run python -m ml.enhancement.train_primek --config configs/speech_enhancement/primek_train.yaml

train-fusion:
	uv run python -m ml.fusion.train_fusion --config configs/speech_enhancement/fusion_train.yaml
```

## Milestones

### Milestone 1: Data Pipeline

Deliverables:

- Clean Common Voice manifests.
- RIR and DEMAND indexes.
- Degraded pair generator.
- Pair manifests with complete metadata.
- Inspection report for generated pairs.

Exit criteria:

- No missing files.
- Clean and degraded lengths match.
- Codec, SNR, RIR, and packet-loss distributions match the configured probabilities within reasonable sampling variance.

### Milestone 2: PrimeK-Net Domain Adaptation

Deliverables:

- Training wrapper.
- Best PrimeK-Net checkpoint.
- Validation PESQ/SI-SDR report.
- Fixed sample audio exports for inspection.

Exit criteria:

- Validation metrics improve over degraded input.
- No obvious speech deletion or severe artifacting on the fixed sample subset.

### Milestone 3: Fusion Model

Deliverables:

- Cached noisy/enhanced log-Mel generation.
- Fusion model implementation.
- Fusion training loop with frozen Whisper and PrimeK-Net.
- WER/CER comparison against baseline and cascade.

Exit criteria:

- Fusion training loss decreases on a tiny overfit run.
- Frozen models remain unchanged.
- Evaluation script produces comparable WER/CER tables for all three inference modes.

### Milestone 4: Thesis-Ready Result Reproduction

Deliverables:

- Full evaluation tables.
- Config snapshots.
- Checkpoint hashes.
- Manifest hashes.
- Short written analysis of where enhancement helps, where it hurts, and whether fusion reduces cascade failures.

Exit criteria:

- The implementation can reproduce the thesis claims with traceable artifacts.
- Any mismatch from the thesis chapter is documented with the exact config, data, or checkpoint difference.

## Open Decisions

These should be resolved before implementation starts:

- Exact path and format of the trained Persian Whisper checkpoint.
- Whether PrimeK-Net will be vendored, installed as a dependency, or used as a Git submodule.
- Whether codec simulation will rely on `ffmpeg`/external binaries or a pure Python implementation where possible.
- Whether enhanced audio/log-Mel features should be cached before fusion training. The recommended default is yes.
- Exact WER/CER normalization rules for Persian punctuation, digits, whitespace, and Arabic/Persian character variants.
