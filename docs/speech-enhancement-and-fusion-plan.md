# Speech Enhancement and Fusion Implementation Plan

This document translates the thesis method chapter into an implementation plan for this repository. The ASR backbone is a fine-tuned Whisper-small checkpoint. The work is:

1. Fine-tune Whisper-small on the general Persian ASR corpus.
2. Build the paired clean/degraded speech-enhancement dataset.
3. Train a log-Mel speech-enhancement module that maps noisy log-Mel features to clean log-Mel features.
4. Train the dual-view fusion model that combines noisy and enhanced log-Mel features, and fine-tune the full stack (enhancement + fusion + Whisper-small) end-to-end on the ASR objective.

This revision covers the **general overview** and the **loss function**. The exact enhancement and fusion architectures are intentionally left open and will be specified in a later pass; the relevant phases below mark architecture as deferred.

## Goals

The implementation should produce a reproducible pipeline that starts from clean Persian speech clips and ends with a trained dual-view enhancement+fusion ASR system. The system should keep all generated artifacts traceable to source audio, augmentation parameters, and model configuration.

The target behavior is not to make enhancement universally better for every dataset. The expected finding from the thesis is more specific:

- The dual-view fusion system should outperform a single-stream Whisper-small fine-tuned on the same degraded data, especially on the most degraded conditions (codec artifacts, packet loss, low SNR).
- The system should remain strong on clean general-purpose speech and not regress relative to the noisy single-stream fine-tune.
- Fusion should reduce the failure mode where a pure enhancement path damages clean or out-of-domain speech, by keeping the noisy view available as a complementary signal.

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
    enhancer.py
    train_enhancer.py
    eval_enhancer.py
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
    enhancer_train.yaml
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
- Background noise: DEMAND 16 kHz release.
- Base ASR checkpoint: OpenAI Whisper-small.
- Fine-tuned ASR checkpoint: the Persian-adapted Whisper-small checkpoint produced by Phase 1.

The exact local paths should live in `configs/speech_enhancement/data.yaml`, not hard-coded in scripts.

Example configuration fields:

```yaml
common_voice_root: /path/to/common_voice/fa
general_persian_asr_manifest: /path/to/general_persian_asr_train.jsonl
asr_validation_manifest: /path/to/general_persian_asr_valid.jsonl
exclude_datasets:
  - telephone
demand_root: /path/to/DEMAND
work_dir: data/speech_enhancement
artifact_dir: artifacts/speech_enhancement
sample_rate: 16000
degraded_variants_per_clip: 2
seed: 1337
```

## Phase 1: Whisper-Small Fine-Tuning

Fine-tune Whisper-small before training enhancement or fusion components. This stage produces the Persian-adapted ASR backbone used to initialize the downstream system.

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

The saved checkpoint is the Persian ASR backbone used to initialize later phases. It should be identified as the fine-tuned Persian Whisper-small checkpoint in configs, logs, and thesis artifacts.

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

### 2. Index Noise Assets

Implement `ml/speech_data/download_noise_assets.py` only if automated download is desired. Otherwise implement an indexing mode that assumes the archives were downloaded manually.

Responsibilities:

- Scan DEMAND and build a manifest of noise files with scene labels.
- Validate that assets are readable and can be resampled to 16 kHz.

Output:

```text
data/speech_enhancement/manifests/
  demand_noise_index.jsonl
```

## Phase 3: Degraded Pair Generation

Implement `ml/speech_data/generate_degraded_pairs.py`.

For every clean clip in train and validation, generate two degraded variants. Each output must remain time-aligned with the clean target and must record every random choice in metadata.

### Degradation Chain

Apply degradations in this order:

1. Load clean audio.
2. Convert to mono and choose the working source sample rate, usually 16 kHz or higher.
3. Optional environmental/background noise mixing.
4. Optional talker/device level variation: gain shift, clipping, AGC, or other level effects.
5. Telephone channel simulation:
   - Narrowband path: resample to 8 kHz, band-limit around 300 to 3400 Hz, then encode/decode with G.711, GSM, AMR-NB, or a similar narrowband codec.
   - Wideband path: keep or resample to 16 kHz, band-limit roughly 50 to 7000 Hz, then encode/decode with AMR-WB, Opus wideband, or a similar wideband codec.
6. Optional network impairment:
   - Packet loss or burst loss before decoding when the selected codec path supports proper packet-level simulation.
   - Waveform dropout as a simpler approximation when packet-level simulation is unavailable.
7. Resample the final degraded input to the model input rate, normally 16 kHz.
8. Apply loudness or peak safety normalization.
9. Write the degraded input and clean target.

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

- `telephone_clean`: narrowband codec/channel degradation without additive noise.
- `telephone_noisy`: narrowband telephone-style speech with environmental noise.
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

## Phase 4: Speech-Enhancement Module

Implement a lightweight speech-enhancement module `E` that operates directly in the log-Mel domain consumed by Whisper: it maps a noisy Whisper log-Mel to an estimated clean log-Mel. There is no waveform output and no STFT round-trip. Unlike a perceptual denoiser, this module is optimized for recognition rather than perceptual quality, and it is trained as part of the end-to-end system in Phase 5 (optionally with a standalone warm-up).

### Design Intent

- Input: noisy Whisper log-Mel, shape `[B, 80, T]`.
- Output: enhanced log-Mel, shape `[B, 80, T]`.
- Lightweight enough to train alongside Whisper-small on an RTX 3090.
- Optimized for downstream recognition, with an auxiliary reconstruction loss keeping the output a faithful "clean view."

The exact architecture (block types, depth, parameter budget) is **deferred** and will be specified in a later pass. This phase fixes the interface, training data, loss, and outputs only.

### Training Data

Use:

- Input: degraded waveform converted to noisy log-Mel with the Whisper feature extractor.
- Target: bandwidth-aligned clean log-Mel from the pair manifest.
- Train split: `se_train_pairs.jsonl`.
- Validation split: `se_valid_pairs.jsonl`.

The dataset loader should:

- Load both waveforms and assert equal sample rate and length.
- Compute log-Mel features with the same Whisper feature extractor used downstream.
- Randomly crop or pad to the configured training segment length.

### Auxiliary Enhancement Loss

The enhancement objective is an **L1 loss in the log-Mel domain** between the predicted log-Mel and the clean log-Mel target:

```text
L_enh = || E(noisy_mel) - clean_mel ||_1
```

Key points:

- The loss lives in the **log-Mel domain** — the exact representation Whisper consumes — not in the linear magnitude spectrum. It is not magnitude-spectrum MSE.
- **L1, not L2:** more robust to outliers and over-smooths less, preserving the low-energy phonetic detail that matters for recognition.
- The target is **log-compressed** (Whisper-style), so errors are weighted perceptually rather than dominated by high-energy bins.
- The `clean_mel` target must be the **bandwidth-aligned** reference produced by Phase 3 (e.g., narrowband-filtered clean for narrowband paths), so the module is not penalized for failing to reconstruct frequencies the channel genuinely removed.

If a future variant produces a waveform instead of a log-Mel, the auxiliary loss should move to a multi-resolution STFT and/or SI-SDR loss; for the current Mel-domain module, log-Mel L1 is the correct match.

Standalone warm-up of `E` on `L_enh` is performed as **Stage 0 of the single staged training script** (Phase 5), not as a separate job.

### Configuration

```yaml
train_manifest: data/speech_enhancement/manifests/se_train_pairs.jsonl
valid_manifest: data/speech_enhancement/manifests/se_valid_pairs.jsonl
sample_rate: 16000
segment_seconds: 4.0
batch_size: 8
gradient_accumulation_steps: 1
learning_rate: 0.0002
num_workers: 4
mixed_precision: true
```

### Checkpoints and Logs

Write outputs to:

```text
artifacts/speech_enhancement/enhancer/
  checkpoints/
    epoch_001.pt
    best.pt
  logs/
    train_metrics.jsonl
  eval/
    valid_metrics.json
```

Each checkpoint should include model state, optimizer/scheduler state, a config snapshot, the Git commit hash when available, and manifest paths with content hashes.

### Enhancement Evaluation

Implement `ml/enhancement/eval_enhancer.py`.

Because the module is recognition-oriented and Mel-domain, evaluation should focus on log-Mel reconstruction quality and downstream effect rather than perceptual codec metrics:

- Log-Mel L1/L2 against the bandwidth-aligned clean target on the validation split.
- Optional: downstream WER/CER of `E` + Whisper as a sanity check (the enhanced-only configuration of Phase 5).

Export fixed-subset before/after log-Mel visualizations for inspection.

## Phase 5: Fusion Network and End-to-End Training

Combine the two complementary views — noisy and enhanced — and fine-tune the full stack on the ASR objective. This is the core of the proposed method.

The stack has three trainable components, optimized jointly end-to-end:

- The speech-enhancement module `E` (Phase 4).
- The fusion mechanism.
- The Whisper-small backbone, initialized from the Persian-adapted checkpoint (Phase 1) and fine-tuned (not frozen).

### Dual-View Inputs

For each utterance:

- `M_n`: noisy Whisper log-Mel.
- `M_e = E(M_n)`: enhanced log-Mel from the Phase 4 module.

The fusion mechanism combines `M_n` and `M_e` into the representation consumed by Whisper. The **exact fusion mechanism and its location in the stack** (for example, early channel-level fusion at the encoder input vs. fusion in the encoder feature space) is **deferred** and will be specified in a later pass. This phase fixes the dual-view interface, the training objective, the dataset, and the evaluation protocol only.

### Fusion Dataset

Implement `ml/fusion/dataset.py`.

Each item should provide:

- Noisy log-Mel (or the degraded waveform plus the Whisper feature extractor).
- Bandwidth-aligned clean log-Mel (target for the auxiliary enhancement loss).
- Transcript tokens for Whisper supervision.
- Optional metadata from the pair manifest.

Because `E` is trainable, its enhanced output **cannot be cached**; only the noisy and clean log-Mel features (which do not depend on model weights) may be precomputed and cached:

```text
data/fusion_cache/
  train/
    noisy_mel/
      <pair_id>.pt
    clean_mel/
      <pair_id>.pt
  valid/
    noisy_mel/
      <pair_id>.pt
    clean_mel/
      <pair_id>.pt
  manifests/
    fusion_train.jsonl
    fusion_valid.jsonl
```

### Training Objective

The primary objective is Whisper-small's autoregressive cross-entropy on the fused input, with the auxiliary log-Mel enhancement loss from Phase 4 keeping the enhanced view faithful:

```text
L_ASR = - sum_t log p(y_t | y_<t, M_f)
L_enh = || E(noisy_mel) - clean_mel ||_1
L     = L_ASR + lambda * L_enh
```

where `M_f` is the fused representation and `lambda` is a small weight (around `0.1` to `0.3`) so enhancement regularizes the clean view without dominating recognition.

### Staged Training in a Single Script

Implement `ml/fusion/train_fusion.py` as a **single, config-driven orchestrator** that runs the full curriculum in one invocation and writes all artifacts to one run directory. It must not be three separate scripts.

The script runs three stages in sequence:

| Stage | What trains | Loss | Purpose |
|---|---|---|---|
| 0 — warm-up | enhancer `E` only | `L_enh` | Give `E` a sane init so it does not emit garbage into the backbone |
| 1 — protect backbone | `E` + fusion, Whisper frozen | `L_ASR + lambda * L_enh` | Front-end learns to produce in-distribution Mels Whisper already accepts |
| 2 — joint | `E` + fusion + Whisper | `L_ASR + lambda * L_enh` | End-to-end gradients make `E` recognition-aware (the core result) |

Orchestration requirements:

- A single config file defines all three stages (steps/epochs, learning rates, frozen flags, `lambda` schedule, and the Whisper adaptation mode for Stage 2).
- The stages run automatically in order within one process; the output of each stage initializes the next.
- **Checkpoint at every stage boundary and support resume-from-stage** so a Stage 2 crash or OOM does not force re-running Stages 0 and 1. Default behavior is still a single end-to-end run; resume is for recovery.
- **Discriminative learning rates** in Stage 2: a much lower learning rate for Whisper than for `E`/fusion (for example, 10x lower).
- **`lambda` annealing:** higher in Stage 1 to anchor the clean view, decayed in Stage 2 so recognition dominates.
- Stage 2 does full fine-tuning of Whisper: the whole backbone is unfrozen and trained at its own lower learning rate. Parameter-efficient adaptation (LoRA/adapters) was considered as a memory fallback but is not implemented; the pipeline always fully fine-tunes the unfrozen backbone.

Per-stage loop responsibilities:

- Load cached noisy/clean log-Mel features (or compute them online).
- Run `E` to produce the enhanced view, then fuse with the noisy view (Stages 1 and 2).
- Feed the fused representation to Whisper-small.
- Compute the active stage's loss and backpropagate only through the parameters that stage trains.
- Log metrics and write checkpoints.

### Configuration

A single config drives the whole run:

```yaml
run_dir: artifacts/speech_enhancement/fusion/run_001
base_asr_checkpoint: artifacts/asr/whisper_small/checkpoints/best
train_manifest: data/fusion_cache/manifests/fusion_train.jsonl
valid_manifest: data/fusion_cache/manifests/fusion_valid.jsonl
sample_rate: 16000
mixed_precision: true
resume_from_stage: null   # null | 0 | 1 | 2

stages:
  warmup:        # Stage 0
    max_steps: 5000
    lr_enhancer: 0.0002
    lambda: 1.0
  fusion:        # Stage 1 (Whisper frozen)
    max_steps: 20000
    lr_frontend: 0.0002
    lambda: 0.3
  joint:         # Stage 2 (end-to-end)
    max_steps: 40000
    lr_frontend: 0.0001
    lr_whisper: 0.00001
    lambda: 0.1
```

### Outputs

All stages write under one run directory:

```text
artifacts/speech_enhancement/fusion/run_001/
  checkpoints/
    stage0_warmup/
    stage1_fusion/
    stage2_joint/
    best/
  logs/
    train_metrics.jsonl
    valid_metrics.json
  config/
    training_config.yaml
    manifest_hashes.json
    git_commit.txt
```

The `best/` checkpoint holds the final enhancement + fusion + Whisper weights for evaluation.

### Fusion Evaluation

Implement `ml/fusion/eval_fusion.py`.

Compare inference modes on the same evaluation manifests, with the noisy single-stream fine-tune as the baseline to beat:

1. Baseline: Whisper-small fine-tuned on the degraded data, evaluated directly on noisy audio (single stream).
2. Enhanced-only: `E` + Whisper-small (no noisy view).
3. Fusion: noisy and enhanced log-Mels combined by the fusion mechanism, then Whisper-small.

Report:

- WER.
- CER.
- Per-dataset averages, with attention to the most degraded subsets (codec artifacts, packet loss, low SNR).
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
- Enhancement module forward pass shape: `[B, 80, T]` in and out.
- Fusion model forward pass shape produces a valid Whisper input.
- Combined loss assembles `L_ASR` and `L_enh` and is finite on a tiny batch.
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

train-enhancer:
	uv run python -m ml.enhancement.train_enhancer --config configs/speech_enhancement/enhancer_train.yaml

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
- DEMAND noise index.
- Degraded pair generator.
- Pair manifests with complete metadata.
- Inspection report for generated pairs.

Exit criteria:

- No missing files.
- Clean and degraded lengths match.
- Codec, SNR, and packet-loss distributions match the configured probabilities within reasonable sampling variance.

### Milestone 3: Speech-Enhancement Module

Deliverables:

- Mel-domain enhancement module implementation.
- Optional standalone warm-up checkpoint.
- Log-Mel reconstruction report on the validation split.
- Fixed-subset before/after log-Mel visualizations.

Exit criteria:

- Validation log-Mel reconstruction improves over the noisy input.
- The module forward pass is shape-correct and trains without instability on a tiny overfit run.

### Milestone 4: Fusion Model and End-to-End Training

Deliverables:

- Cached noisy/clean log-Mel generation.
- Fusion model implementation.
- Single staged training script (`train_fusion.py`) running Stage 0 -> 1 -> 2 in one invocation, with per-stage checkpoints, resume-from-stage, and one run directory.
- WER/CER comparison against the noisy single-stream baseline and the enhanced-only configuration.

Exit criteria:

- The script runs all three stages end-to-end from a single config and writes all artifacts to one run directory.
- Resume-from-stage restores Stage 2 from the Stage 1 checkpoint without re-running Stages 0 and 1.
- The combined training loss decreases on a tiny overfit run.
- The evaluation script produces comparable WER/CER tables for all inference modes.
- The fusion system beats the noisy single-stream fine-tune on at least the most degraded subsets.

### Milestone 5: Thesis-Ready Result Reproduction

Deliverables:

- Full evaluation tables.
- Config snapshots.
- Checkpoint hashes.
- Manifest hashes.
- Short written analysis of where the dual-view helps, where it does not, and how it compares to the noisy single-stream fine-tune.

Exit criteria:

- The implementation can reproduce the thesis claims with traceable artifacts.
- Any mismatch from the thesis chapter is documented with the exact config, data, or checkpoint difference.

## Open Decisions

These should be resolved before or during implementation:

- Exact architecture and parameter budget of the Mel-domain enhancement module.
- Exact fusion mechanism and its location in the stack (early channel-level vs. encoder-feature-space fusion).
- Full fine-tuning vs. parameter-efficient adaptation (LoRA/adapters) for Whisper-small, given RTX 3090 memory.
- The auxiliary-loss weight `lambda`.
- Whether to warm up the enhancement module standalone before end-to-end training.
- Exact path and format of the fine-tuned Persian Whisper-small checkpoint.
- Exact location and source metadata for each non-telephone general Persian ASR training source.
- Whether codec simulation will rely on `ffmpeg`/external binaries or a pure Python implementation where possible.
- Exact WER/CER normalization rules for Persian punctuation, digits, whitespace, and Arabic/Persian character variants.
