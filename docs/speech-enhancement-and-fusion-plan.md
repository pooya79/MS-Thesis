# Speech Enhancement and Fusion Implementation Plan

This document translates the thesis method chapter into an implementation plan for this repository. The ASR backbone is a fine-tuned Whisper-small checkpoint. The work is:

1. Fine-tune Whisper-small on the general Persian ASR corpus.
2. Build the paired clean/degraded speech-enhancement dataset.
3. Train or domain-adapt PrimeK-Net for Persian telecommunication-style degradation.
4. Train the log-Mel fusion network that combines noisy and enhanced features before the frozen Whisper-small model.

## Goals

The implementation should produce a reproducible pipeline that starts from clean Persian speech clips and ends with trained speech-enhancement and fusion checkpoints. The system should keep all generated artifacts traceable to source audio, augmentation parameters, and model configuration.

The target behavior is not to make enhancement universally better for every dataset. The expected finding from the thesis is more specific:

- Enhancement should help real or simulated telecommunication audio.
- Direct Whisper-small should remain strong on clean general-purpose speech.
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
    prepare_asr_manifest.py
    train_whisper_small.py
    eval_whisper.py
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
    whisper_small_train.yaml
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

- General Persian ASR audio-text corpus from the thesis data table, excluding the dedicated telephone data and excluding held-out test/evaluation splits.
- Clean Persian speech: Common Voice Persian v21 train and validation splits.
- Room impulse responses: BUT ReverbDB.
- Background noise: DEMAND 16 kHz release.
- Base ASR checkpoint: OpenAI Whisper-small.
- Fine-tuned ASR checkpoint: the Persian-adapted Whisper-small checkpoint produced by Phase 1.
- Optional baseline enhancement checkpoint: official or compatible PrimeK-Net checkpoint.

The exact local paths should live in `configs/speech_enhancement/data.yaml`, not hard-coded in scripts.

Example configuration fields:

```yaml
common_voice_root: /path/to/common_voice/fa
general_persian_asr_manifest: /path/to/general_persian_asr_train.jsonl
asr_validation_manifest: /path/to/general_persian_asr_valid.jsonl
exclude_datasets:
  - telephone
but_reverbdb_root: /path/to/BUT_ReverbDB
demand_root: /path/to/DEMAND
work_dir: data/speech_enhancement
artifact_dir: artifacts/speech_enhancement
sample_rate: 16000
degraded_variants_per_clip: 2
seed: 1337
```

## Phase 1: Whisper-Small Fine-Tuning

Fine-tune Whisper-small before training enhancement or fusion components. This stage produces the frozen ASR backbone used everywhere downstream.

### 1. Prepare General Persian ASR Manifests

Implement `ml/asr/prepare_asr_manifest.py`.

Responsibilities:

- Read the thesis general Persian ASR sources and their audio-text metadata.
- Include public and private general Persian training data.
- Exclude the dedicated telephone data from ASR fine-tuning.
- Exclude held-out test/evaluation splits from training.
- Normalize all audio references to mono 16 kHz input expectations.
- Normalize transcripts with the same Persian text rules used for ASR scoring.
- Record dataset name, split, source path, duration, transcript, and stable utterance ID.

Output:

```text
data/asr/manifests/
  whisper_small_train.jsonl
  whisper_small_valid.jsonl
```

### 2. Fine-Tune Whisper-Small

Implement `ml/asr/train_whisper_small.py`.

Training should start from OpenAI Whisper-small and optimize the standard autoregressive sequence-to-sequence cross-entropy objective on the general Persian ASR training manifest. The dedicated telephone set remains evaluation-only and must not be included in training.

Suggested configuration:

```yaml
base_model: openai/whisper-small
train_manifest: data/asr/manifests/whisper_small_train.jsonl
valid_manifest: data/asr/manifests/whisper_small_valid.jsonl
artifact_dir: artifacts/asr/whisper_small
sample_rate: 16000
seed: 1337
mixed_precision: true
```

### Outputs

Write outputs to:

```text
artifacts/asr/whisper_small/
  checkpoints/
    best/
  logs/
    train_metrics.jsonl
    valid_metrics.json
  config/
    text_normalization.json
    tokenizer_config.json
    training_config.yaml
    manifest_hashes.json
```

The saved checkpoint is the single ASR backbone for later phases. It should be identified as the fine-tuned Persian Whisper-small checkpoint in configs, logs, and thesis artifacts.

## Phase 2: Speech-Enhancement Dataset Preparation

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

## Phase 3: Degraded Pair Generation

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

Use total RIR probability `0.18`.

- Severe reverb: probability `0.03`, wet mix `0.6` to `0.8`, D/R `6` to `10` dB.
- Mild reverb: probability `0.15`, wet mix `0.3` to `0.5`, D/R `12` to `18` dB.

Prefer RIRs with speaker/session labels when available. Normalize the RIR before convolution and keep output length aligned to the clean clip.

### Noise Mixing

Use noise probability `0.60`.

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

Sample a named degradation profile first, then sample channel path and codec type from that profile. A reasonable profile set is:

- `telephone_clean`: narrowband codec/channel degradation without additive noise or RIR.
- `telephone_noisy`: narrowband telephone-style speech with environmental noise and rare room coloration.
- `voip_lossy`: Opus/AMR/G.711-style degradation with frequent bursty decoded-waveform loss approximation.
- `mobile_wideband`: wideband mobile or app-call speech with moderate acoustic contamination.

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

If packet-level simulation is available for the selected codec in a future implementation, apply packet or burst loss before decoding.

Sample packet loss rate from:

- Light: `0.3%` to `2%`.
- Medium: `2%` to `5%`.
- Heavy: `5%` to `10%`.

The current implementation uses decoded waveform dropout as a simpler approximation. It records `mode: decoded_waveform_dropout`, `model: two_state_burst`, target loss rate, observed loss rate, burst parameters, frame size, and dropout duration.

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
  "profile": "voip_lossy",
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
  "codec_bitrate": null,
  "codec_frame_duration_ms": null,
  "network_impairment": {
    "enabled": true,
    "mode": "decoded_waveform_dropout",
    "model": "two_state_burst",
    "loss_rate": 0.031,
    "observed_loss_rate": 0.028,
    "burst_length": 3,
    "frame_ms": 20,
    "dropout_ms": 120
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
- Profile distribution.
- Codec distribution.
- SNR distribution.
- Decoded-dropout loss distribution.
- Count of missing/unreadable files.
- Count of length mismatches.

This must run before training.

## Phase 4: PrimeK-Net Training

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

## Phase 5: Fusion Network Training

Implement the fusion model after the enhancement checkpoint is usable.

The fusion stage has three frozen components:

- Frozen fine-tuned Persian Whisper-small checkpoint.
- Frozen Whisper-small feature extractor.
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

Use Whisper-small's autoregressive cross-entropy loss:

```text
L_ASR = -sum_t log p(y_t | y_<t, M_f)
```

Whisper-small remains frozen. The optimizer updates only fusion network parameters.

Training loop responsibilities:

- Load cached noisy/enhanced log-Mel features.
- Run fusion network.
- Feed fused features to frozen Whisper-small.
- Compute token-level cross-entropy against transcript labels.
- Backpropagate only through fusion.
- Save checkpoints and validation metrics.

### Fusion Evaluation

Implement `ml/fusion/eval_fusion.py`.

Compare three inference modes on the same evaluation manifests:

1. Baseline: noisy audio directly to frozen fine-tuned Persian Whisper-small.
2. Cascade: PrimeK-Net enhanced audio to frozen fine-tuned Persian Whisper-small.
3. Fusion: noisy and enhanced log-Mels through fusion network, then frozen fine-tuned Persian Whisper-small.

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

## Phase 6: Reproducibility and Tests

Add tests for the code paths that can be tested without full GPU training.

Recommended tests:

- ASR manifest excludes dedicated telephone data and held-out test splits.
- Whisper-small checkpoint metadata records base model, config snapshot, and manifest hashes.
- Manifest schema validation.
- Stable deterministic seed generation.
- Audio pair length alignment.
- Degradation metadata completeness.
- Fusion model forward pass shape.
- Frozen-parameter check for Whisper-small and PrimeK-Net during fusion training.
- Text normalization consistency between training and scoring.

For long-running GPU jobs, add smoke-test commands that run on a tiny manifest with 2 to 4 clips.

Example commands to add later:

```make
prepare-asr-data:
	uv run python -m ml.asr.prepare_asr_manifest --config configs/speech_enhancement/data.yaml

train-whisper-small:
	uv run python -m ml.asr.train_whisper_small --config configs/speech_enhancement/whisper_small_train.yaml

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

### Milestone 1: Whisper-Small ASR Backbone

Deliverables:

- General Persian ASR train/validation manifests.
- Fine-tuned Persian Whisper-small checkpoint.
- Text normalization and tokenizer config snapshots.
- Training config, validation metrics, and manifest hashes.

Exit criteria:

- Dedicated telephone data is absent from ASR training manifests.
- Held-out evaluation splits are absent from ASR training manifests.
- The checkpoint can transcribe a small validation subset through the shared scoring path.

### Milestone 2: Data Pipeline

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

### Milestone 3: PrimeK-Net Domain Adaptation

Deliverables:

- Training wrapper.
- Best PrimeK-Net checkpoint.
- Validation PESQ/SI-SDR report.
- Fixed sample audio exports for inspection.

Exit criteria:

- Validation metrics improve over degraded input.
- No obvious speech deletion or severe artifacting on the fixed sample subset.

### Milestone 4: Fusion Model

Deliverables:

- Cached noisy/enhanced log-Mel generation.
- Fusion model implementation.
- Fusion training loop with frozen Whisper-small and PrimeK-Net.
- WER/CER comparison against baseline and cascade.

Exit criteria:

- Fusion training loss decreases on a tiny overfit run.
- Frozen models remain unchanged.
- Evaluation script produces comparable WER/CER tables for all three inference modes.

### Milestone 5: Thesis-Ready Result Reproduction

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

- Exact path and format of the fine-tuned Persian Whisper-small checkpoint.
- Exact location and source metadata for each non-telephone general Persian ASR training source.
- Whether PrimeK-Net will be vendored, installed as a dependency, or used as a Git submodule.
- Whether codec simulation will rely on `ffmpeg`/external binaries or a pure Python implementation where possible.
- Whether enhanced audio/log-Mel features should be cached before fusion training. The recommended default is yes.
- Exact WER/CER normalization rules for Persian punctuation, digits, whitespace, and Arabic/Persian character variants.
