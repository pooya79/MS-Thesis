# Speech Degradation Pipeline

This document explains the paired clean/degraded speech generation pipeline implemented in
`ml/speech_data/generate_degraded_pairs.py` and configured by
`configs/speech_enhancement/degradation.yaml`.

The pipeline is designed for thesis experiments on Persian speech enhancement and ASR
robustness under telephone and VoIP-style degradation. It does not claim to be a perfect
telephony network simulator. It creates reproducible, metadata-rich synthetic pairs that
separate channel bandwidth, codec artifacts, environmental noise, and approximate network
loss well enough for controlled training and ablation.

## Inputs And Outputs

Inputs:

- Clean train and validation manifests in JSONL format.
- Optional noise index JSONL for environmental noise recordings.
- A YAML degradation config.

Download the optional DEMAND noise archives with:

```bash
uv run python -m ml.speech_data.scripts.download_degradation_assets \
  --noise-root data/speech_enhancement/assets/noise/DEMAND
```

Prepare the optional DEMAND noise index from downloaded DEMAND assets with:

```bash
uv run python -m ml.speech_data.scripts.prepare_degradation_assets \
  --noise-root data/speech_enhancement/assets/noise/DEMAND \
  --manifest-dir data/speech_enhancement/manifests
```

The script extracts local `.zip` archives by default, validates readable audio, and
writes `demand_noise_index.jsonl`.

Each clean manifest row must include:

```json
{
  "id": "cv-fa-train-000001",
  "split": "train",
  "clean_path": "data/speech_enhancement/clean/train/cv-fa-train-000001.wav",
  "transcript": "optional transcript"
}
```

Outputs:

- `data/speech_enhancement/pairs/<split>/clean/*.wav`
- `data/speech_enhancement/pairs/<split>/degraded/*.wav`
- `data/speech_enhancement/manifests/se_train_pairs.jsonl`
- `data/speech_enhancement/manifests/se_valid_pairs.jsonl`
- `data/speech_enhancement/manifests/generation_report.json`

The clean and degraded files in each pair are time-aligned and written at the configured
model sample rate, normally 16 kHz.

## Reproducibility

Every generated variant is seeded from:

- Global config seed.
- Split name.
- Clean clip ID.
- Variant index.

This means the same config and input manifests produce the same random choices for each
pair. The output manifest records the selected random choices so each degradation can be
audited later.

## Profile-Based Generation

The generator first samples a named `profile` from the config. A profile is a weighted set
of overrides for codec distribution, noise probability, and network impairment. This
prevents all degradation types from being blended into one opaque global distribution.

Current profiles:

- `telephone_clean`: narrowband telephone codec/channel degradation without additive
  noise.
- `telephone_noisy`: narrowband telephone-style speech with environmental noise.
- `voip_lossy`: Opus/AMR/G.711-style VoIP degradation with frequent bursty decoded
  waveform loss approximation.
- `mobile_wideband`: wideband mobile or app-call speech with moderate acoustic
  contamination.

Every manifest row records the selected profile:

```json
{
  "profile": "voip_lossy"
}
```

Profiles make the dataset easier to inspect and support thesis ablations such as
`codec_only`, `codec_noise`, and `voip_lossy`.

If a config does not define `profiles`, the generator uses a single `legacy` profile and
keeps the older global behavior.

## Degradation Chain

For each clean clip and variant, the generator applies these stages:

1. Load clean audio as mono.
2. Resample to the working sample rate.
3. Sample a profile and merge its overrides into the base config.
4. Optionally mix environmental noise at a sampled SNR.
5. Apply level variation and optional clipping.
6. Select channel path and codec.
7. Resample and band-limit for narrowband or wideband channel simulation.
8. Apply an ffmpeg codec round-trip.
9. Optionally apply decoded waveform dropout as a network-loss approximation.
10. Resample degraded input to the model sample rate.
11. Normalize peaks for safety.
12. Create the clean target, with bandwidth alignment for narrowband samples.
13. Write WAV files and JSONL metadata.

## Noise Stage

Noise is selected from the configured noise index when enabled. The generator can mix one
or two noise scenes:

- One scene by default.
- A second scene with `second_scene_probability`.
- SNR sampled from explicit buckets.

Manifest fields:

- `noise_scenes`
- `noise_ids`
- `snr_db`

Noise probability is profile-specific. This keeps pure codec/channel examples in the
training set while still exposing the model to realistic call backgrounds.

## Level And Device Stage

The level stage currently supports:

- Random gain in dB.
- Optional hard clipping.
- AGC metadata placeholder.

These effects run after optional noise and before channel simulation. That order treats
them as part of the talker/device front end rather than the network channel.

Future device-front-end improvements can be added here:

- Conservative AGC.
- Dynamic range compression.
- Spectral tilt.
- Mobile noise suppression artifacts.
- DC removal or high-pass filtering.

## Channel And Codec Stage

The selected codec determines the channel path:

- Narrowband codecs use 8 kHz channel audio and 300-3400 Hz band-pass filtering.
- Wideband codecs use 16 kHz channel audio and 50-7000 Hz band-pass filtering.
- Pass-through samples choose narrowband or wideband from
  `pass_through_path_distribution`.

Supported codecs:

- `g711_alaw`
- `g711_mulaw`
- `gsm`
- `amr_nb_12k2`
- `amr_wb_12k65`
- `opus_nb`
- `opus_wb`
- `pass_through`

Codec simulation uses real ffmpeg encode/decode round-trips. Opus config entries can
sample `bitrate` and `frame_duration_ms`; those values are recorded as:

- `codec_bitrate`
- `codec_frame_duration_ms`

This improves VoIP diversity without requiring a full packet-level network simulator.

## Network Impairment Stage

The current network stage is intentionally labeled as an approximation:

```json
{
  "mode": "decoded_waveform_dropout",
  "model": "two_state_burst"
}
```

It runs after codec decoding, so it does not model codec packet loss, decoder packet-loss
concealment, jitter buffers, retransmission, or late packets. It does create short bursty
dropouts using a two-state frame process:

- Good state: frames pass through.
- Bad state: frames are zeroed.
- `loss_rate` controls the target long-run loss rate.
- `burst_length` controls expected bad-state duration.

Recorded fields:

- `loss_rate`
- `burst_length`
- `frame_ms`
- `dropout_ms`
- `dropped_frames`
- `total_frames`
- `observed_loss_rate`

This is more honest and more auditable than calling the effect true packet loss.

## Clean Target Policy

The degraded input and clean target have the same duration and sample rate. The target is
not always the original fullband clean signal:

- Narrowband degraded inputs get a narrowband-filtered clean target.
- Wideband degraded inputs keep the normal 16 kHz clean target unless
  `channel.wideband.filter_target` is enabled.

This avoids training the model to hallucinate high-frequency content that the simulated
narrowband channel removed.

## Manifest Metadata

Each output row records the important choices:

```json
{
  "pair_id": "train_cv-fa-train-000001_v0",
  "split": "train",
  "profile": "telephone_noisy",
  "source_clean_id": "cv-fa-train-000001",
  "clean_path": "data/speech_enhancement/pairs/train/clean/train_cv-fa-train-000001_v0.wav",
  "degraded_path": "data/speech_enhancement/pairs/train/degraded/train_cv-fa-train-000001_v0.wav",
  "target_bandwidth": "narrowband",
  "noise_scenes": ["cafeteria"],
  "noise_ids": ["demand-cafeteria-001"],
  "snr_db": 7.4,
  "gain_db": -1.2,
  "channel_path": "narrowband",
  "channel_sample_rate": 8000,
  "channel_bandpass_hz": [300.0, 3400.0],
  "codec": "g711_alaw",
  "codec_bitrate": null,
  "codec_frame_duration_ms": null,
  "network_impairment": {
    "enabled": false,
    "mode": null,
    "model": null
  },
  "normalization": "peak_safety",
  "seed": 123456789
}
```

## Inspection

Use the manifest inspector before training:

```bash
uv run python -m ml.speech_data.inspect_manifest \
  data/speech_enhancement/manifests/se_train_pairs.jsonl
```

The inspector reports:

- Pair count.
- Total hours.
- Missing and unreadable files.
- Length mismatches.
- Profile distribution.
- Channel and codec distribution.
- Codec bitrate distribution.
- Network impairment mode distribution.
- SNR summary.
- Observed decoded-dropout loss summary.

## Known Limitations

The current pipeline is useful for controlled synthetic training, but it is not a complete
telephone or VoIP emulator.

Known limitations:

- Network loss is applied after decoding, not as packet corruption before decoding.
- There is no codec-specific packet-loss concealment simulation.
- There is no jitter buffer, variable delay, duplicate packet, or late-packet model.
- There is no DTX, comfort noise, sidetone, echo, or acoustic echo cancellation model.
- AGC is currently metadata-only unless implemented in a future level stage.
- Noise assets only reflect the quality and diversity of the indexed corpora.

The thesis should describe the generated data as telephone/VoIP-inspired synthetic
degradation, not as real network capture.

## Recommended Experiment Reporting

When reporting results, break metrics down by:

- Profile.
- Codec.
- Narrowband vs wideband.
- Noise present vs absent.
- Network impairment enabled vs disabled.
- Target bandwidth.

This makes it possible to explain whether a model improves because it handles codec
coloration, bandwidth limitation, noise, or bursty loss.
