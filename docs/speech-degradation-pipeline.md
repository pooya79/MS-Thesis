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
- `voip_lossy`: Opus VoIP degradation with bursty packet loss and decoder packet-loss
  concealment.
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
9. Cross-correlate against the pre-codec channel waveform and compensate codec delay.
10. Optionally apply packet loss during Opus encode/decode so decoder PLC handles
    missing packets. Non-Opus codecs receive a clearly labeled decoded-waveform fallback
    impairment.
11. Resample degraded input to the model sample rate.
12. Create the clean target, with bandwidth alignment for narrowband and default wideband samples.
13. Apply one shared peak-safety scale to the clean/degraded pair.
14. Write WAV files and JSONL metadata.

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

After decoding, the generator estimates codec delay with bounded cross-correlation
against the pre-codec channel waveform, shifts the decoded waveform, and records the
measured `codec_alignment_lag_samples`.

- `codec_bitrate`
- `codec_frame_duration_ms`

This improves VoIP diversity without requiring a full network simulator.

## Network Impairment Stage

The primary VoIP network stage applies loss before decoding for Opus codecs:

```json
{
  "mode": "packet_loss_plc",
  "model": "opus_decoder_plc"
}
```

The generator encodes each Opus frame, samples a two-state burst loss process over the
encoded packet sequence, and passes missing packets to the Opus decoder. The degraded
waveform therefore contains the decoder's packet-loss concealment output rather than hard
zeroed gaps.

For non-Opus codecs, the pipeline uses a clearly labeled fallback mode:

```json
{
  "mode": "decoded_waveform_dropout_fallback",
  "model": "two_state_burst"
}
```

That fallback runs after codec decoding and should not be used as evidence for true VoIP
packet-loss robustness. The checked-in `voip_lossy` profile is Opus-only so packet-loss
examples use decoder PLC by default. Explicit ablations can also request
`mode: decoded_waveform_dropout`, which uses the same waveform dropout model.

Both modes use a two-state frame process:

- Good state: frames pass through.
- Bad state: encoded Opus packets are dropped, or fallback waveform frames are zeroed.
- `loss_rate` controls the target long-run loss rate.
- `burst_length` controls expected bad-state duration.

Recorded fields:

- `loss_rate`
- `burst_length`
- `frame_ms` records the effective impairment frame size. For Opus PLC this is the
  selected codec frame duration when present.
- `dropout_ms`
- `dropped_frames`
- `total_frames`
- `observed_loss_rate`

Metadata records which path was used, making packet-loss PLC and fallback examples
auditable.

## Clean Target Policy

The degraded input and clean target have the same duration and sample rate. The target is
not always the original fullband clean signal:

- Narrowband degraded inputs get a narrowband-filtered clean target.
- Wideband degraded inputs get a wideband-filtered clean target by default because
  `channel.wideband.filter_target` is enabled in the checked-in config.
- If `channel.wideband.filter_target` is disabled, wideband degraded inputs keep the
  normal 16 kHz clean target. That setting intentionally trains bandwidth extension.

This avoids training the model to hallucinate high-frequency content that the simulated
narrowband or wideband channel removed.

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
  "codec_alignment_lag_samples": 0,
  "network_impairment": {
    "enabled": false,
    "mode": null,
    "model": null
  },
  "normalization": "shared_pair_peak_safety",
  "normalization_scale": 1.0,
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

- Packet loss with decoder PLC is currently implemented for Opus codecs only.
- Non-Opus packet-loss requests use decoded waveform dropout as a labeled fallback.
- There is no jitter buffer, variable delay, duplicate packet, or late-packet model.
- There is no DTX, comfort noise, sidetone, echo, or acoustic echo cancellation model.
- AGC is currently metadata-only unless implemented in a future level stage.
- Noise assets only reflect the quality and diversity of the indexed corpora.

The thesis should describe the generated data as telephone/VoIP-inspired synthetic
degradation with Opus packet-loss PLC, not as real network capture.

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
