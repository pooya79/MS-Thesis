# MS Thesis Progress Report

## 1. Work completed

The work so far has focused on building a reproducible experimental pipeline for robust Persian automatic speech recognition (ASR), particularly under telephone and VoIP degradation.

I first prepared and normalized the Persian training and evaluation datasets. Audio was converted to mono 16 kHz, transcripts were normalized consistently, and all datasets were organized using the same TSV-based interface. Because much of the training material consists of short utterances of approximately five seconds, I also implemented long-audio augmentation. This creates additional utterances by concatenating two to four clips, with a target duration above 5 seconds and a maximum duration of 20 seconds.

I selected Whisper-small and FastConformer as the ASR backbones. Both are comparatively compact models, which makes repeated training, evaluation, and architecture experiments practical with the available hardware.

The completed model experiments are:

1. Fine-tuning Whisper-small and FastConformer on the normal Persian datasets plus the generated long-utterance variants.
2. Fine-tuning both models on the same data with additional telephone/VoIP-degraded Common Voice 25 data.
3. Implementing and training a dual-view Whisper fusion model that combines the original and enhanced representations of an utterance.
4. Evaluating all five trained systems on the same five Persian test datasets using word error rate (WER) and character error rate (CER).

All reported training runs used seed 1337 and one epoch for the standalone Whisper and FastConformer experiments.

## 2. Data used

### Training data

The normal-data Whisper and FastConformer runs used:

- `cv-corpus-25.0`: Persian Common Voice 25.
- `FarsSpon_train_dev`: the available Persian spontaneous/private training collection.
- `fleurs-normalized`: Persian FLEURS.
- `cv-corpus-25.0-long`: long variants of Common Voice.
- `FarsSpon_train_dev-long`: long variants of FarsSpon.

The local normalized source datasets contain:

| Dataset | Train utterances | Development utterances |
|---|---:|---:|
| Common Voice 25 | 299,430 | 10,006 |
| FarsSpon | 600,672 | 6,044 |
| FLEURS | 3,019 | 362 |

The degraded-data runs added:

- `cv-corpus-25.0-degraded-v2`: two independently degraded variants of Common Voice training/development speech.
- `cv-corpus-25.0-degraded-long-v2`: long-utterance variants of the degraded Common Voice data.

Fusion v1 used the same normal, long, and degraded dataset groups. Its saved configuration initialized the ASR backbone from the original `openai/whisper-small` checkpoint rather than the Persian fine-tuned checkpoint.

### Evaluation data

The models were evaluated on held-out test splits from:

- AGFarsdat normalized telephone speech: 10,044 evaluated utterances.
- Common Voice 25: approximately 10,519 utterances.
- Persian FLEURS: 871 utterances.
- PersianSpeech: 24 evaluated utterances in the saved runs.
- Persian Speech Corpus: 391–396 utterances, depending on model-side filtering.

## 3. Telephone and VoIP simulation

I implemented a deterministic, config-driven degradation pipeline that produces time-aligned clean/degraded speech pairs and records every random decision in JSONL metadata. Each variant is seeded from the global seed, split, source clip identifier, and variant index.

The pipeline samples one of four conditions: clean telephone, noisy telephone, lossy VoIP, or mobile wideband. It can then apply environmental DEMAND noise at SNRs from +15 dB down to −5 dB, random gain from −6 to +6 dB, channel bandwidth limitation, codec distortion, and network loss.

Narrowband speech is filtered to approximately 300–3400 Hz and supports G.711 A-law, G.711 μ-law, GSM, and AMR-NB. Wideband speech is filtered to approximately 50–7000 Hz and supports AMR-WB and Opus. Codec effects are produced through real FFmpeg encode/decode round trips rather than numerical approximations.

For lossy Opus, packets are dropped before decoding so that the Opus decoder performs packet-loss concealment. Loss rates are sampled from 0.3–2%, 2–5%, and 5–10%, with burst lengths of one to five frames. A labeled waveform-dropout fallback is available for codecs without packet-level simulation. Codec delay is compensated by cross-correlation, and final clean/degraded pairs are length-aligned and normalized with a shared peak-safety scale.

## 4. Fusion model

The fusion system receives two log-Mel views of the same input:

- The original noisy/channel-degraded log-Mel representation.
- An enhanced log-Mel representation produced by a residual U-Net.

Both views pass through the shared Whisper encoder. Their encoder states are refined using bidirectional cross-attention and then combined by a learned per-time-step, per-feature gate. This preserves the original view when enhancement removes useful phonetic information, while allowing the model to use the enhanced view in corrupted regions.

Fusion v1 used a three-level residual U-Net with 48 base channels and a three-layer, 12-head cross-attention module. Training followed three stages:

1. 20,000-step enhancer warm-up using log-Mel L1 reconstruction loss.
2. 30,000 steps training the enhancer and fusion module while Whisper remained frozen.
3. 50,000 steps of joint end-to-end training with ASR cross-entropy plus a smaller enhancement loss.

## 5. Evaluation results

Lower WER and CER are better.

| Model | Aggregate WER | Aggregate CER |
|---|---:|---:|
| Whisper-small, normal + long data | 23.27% | 10.84% |
| Whisper-small, normal + long + degraded data | **17.83%** | **7.93%** |
| FastConformer, normal + long data | 22.73% | 14.79% |
| FastConformer, normal + long + degraded data | 21.81% | 14.45% |
| Whisper fusion v1 | 24.35% | 15.46% |

Per-dataset WER:

| Model | AGFarsdat | Common Voice | FLEURS | PersianSpeech | Persian Speech Corpus |
|---|---:|---:|---:|---:|---:|
| Whisper normal | 35.63% | 11.52% | 21.05% | 34.53% | **30.17%** |
| Whisper degraded | **25.80%** | **7.84%** | **19.82%** | 35.25% | 30.90% |
| FastConformer normal | 31.07% | 11.68% | 29.22% | 30.70% | 33.69% |
| FastConformer degraded | 30.48% | 10.01% | 29.40% | 31.41% | 33.59% |
| Fusion v1 | 29.45% | 18.37% | 21.55% | **28.78%** | 37.26% |

Adding degraded training data reduced aggregate WER from 23.27% to 17.83% for Whisper and from 22.73% to 21.81% for FastConformer. The degraded-data Whisper model is therefore the current best system.

## 6. Current status and next work

Fusion v1 currently underperforms the augmented degraded-data Whisper baseline: 24.35% versus 17.83% aggregate WER. Its learned average gate also remained close to an equal mixture of the two views (51.5% enhanced and 48.5% original), suggesting that the model has not yet learned strong condition-dependent view selection.

I am currently working on improving this result. The first priority is to initialize fusion from the best Persian Whisper model—especially the degraded-data checkpoint—instead of vanilla Whisper-small. I will also compare noisy-only, enhanced-only, and fused inference from the same checkpoint, improve the enhancer's ASR-aware training objective, and tune the fusion architecture and training schedule. The objective is for fusion to outperform the strong single-stream degraded-data Whisper baseline rather than only the normal-data baseline.
