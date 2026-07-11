# Degradation Pipeline Architecture Diagram

This document explains the blocks and branches in the accompanying
[degradation pipeline diagram](index.html). The figure is a general view of the
implemented pipeline in
[`generate_degraded_pairs.py`](../../ml/speech_data/generate_degraded_pairs.py),
not a snapshot of profile weights or degradation probabilities. Those values belong to
[`degradation.yaml`](../../configs/speech_enhancement/degradation.yaml).

The pipeline creates time-aligned clean/degraded speech pairs for speech-enhancement and
robust-ASR experiments. It is intended to simulate controlled telephone and VoIP-style
degradation. It is not a complete telephone-network simulator: room impulse responses,
echo, jitter, overlapping speakers, and similar effects are outside its current scope.

## How to read the diagram

- Solid boxes and connectors represent stages that always run once their branch is
  selected.
- Dashed boxes and connectors represent optional or conditional operations.
- Orange identifies the native Opus packet-loss/PLC branch.
- Blue identifies ffmpeg codec processing and the decoded-waveform fallback.
- Purple identifies pass-through processing.
- The top row shows the complete pair-generation flow. The large inset expands the
  “Channel, codec + network” block.

Configuration probabilities are intentionally omitted. This keeps the figure valid when
profile weights, SNR buckets, codec distributions, or loss rates change.

## Top-level pair flow

### 1. Clean audio

The input is one clean manifest item containing a stable source ID, split, path, and
optional transcript. The audio loader returns a mono waveform and its source sample rate.
The original clean waveform is never degraded in place; a working copy becomes the
degraded side of the pair.

The source ID and path are retained in the output metadata so every generated pair can be
traced back to its input.

### 2. Mono + resampling

The clean waveform is resampled from its source rate to the configured working sample
rate. The checked-in configuration uses 16 kHz, but the diagram treats this as a
configuration value rather than an architectural constant.

All acoustic front-end operations—noise mixing, gain, and optional clipping—operate on
this working-rate waveform.

### 3. Profile sampling

The generator derives a stable seed from four values:

1. the global configuration seed;
2. the dataset split;
3. the clean source ID; and
4. the variant index.

That seed initializes a local random-number generator for the variant. A named profile is
then sampled, and its overrides are merged into the base configuration. Profiles constrain
the effective codec distribution, noise probability, and network-impairment probability.

This makes each variant reproducible independently of manifest processing order. The
output row records both the derived `seed` and selected `profile`.

For example, the checked-in `telephone_noisy` profile is defined as:

```yaml
- name: telephone_noisy
  weight: 0.30
  description: Narrowband telephone-style speech with environmental noise.
  noise:
    probability: 0.70
  network_impairment:
    probability: 0.20
  codec_distribution:
    - codec: g711_alaw
      weight: 0.35
    - codec: g711_mulaw
      weight: 0.15
    - codec: gsm
      weight: 0.20
    - codec: amr_nb_12k2
      weight: 0.30
```

If this profile is selected, its noise and network probabilities and narrowband codec
distribution override the corresponding base settings for that variant. The profile does
not itself select a particular codec or guarantee that noise/network impairment fires;
those choices are sampled later from the effective configuration. The numeric values above
are a configuration example and are not architectural constants.

### 4. Optional noise

When noise is selected and indexed noise assets are available, the generator chooses one
or optionally two noise recordings. Each recording is resampled to the working rate,
cropped or repeated to match the speech duration, and combined with the other selected
scene when necessary. The resulting noise is mixed with speech at an SNR sampled from the
effective profile configuration.

The noise-probability random draw is consumed even when no noise index is available. This
prevents the presence or absence of noise assets from shifting later random choices.

Recorded fields include `noise_scenes`, `noise_ids`, and `snr_db`. Empty lists and a null
SNR explicitly identify variants for which no noise was mixed.

### 5. Gain + optional hard clipping

A gain value is sampled in decibels and applied to the working waveform. Hard clipping can
also be enabled and sampled conditionally. When clipping fires, the implementation records
the threshold and the fraction of samples that actually exceeded it.

The diagram deliberately does not call this a general “device effects” block. At present,
the implemented signal operations are gain and optional hard clipping. AGC is represented
only by an `agc.enabled` metadata placeholder and does not process the waveform.

Recorded fields include `gain_db`, `clipping`, and `agc`.

### 6. Channel, codec + network

This block is expanded in the central inset. It selects a codec, derives the channel path,
applies bandwidth simulation and peak protection, and then runs exactly one codec/network
branch. Every branch rejoins at delay estimation and alignment.

### 7. Model-rate resampling

After codec/network processing and delay alignment, the degraded channel waveform is
resampled to the configured model sample rate. Its length is matched to the duration
expected from the clean working waveform.

The output metadata records `model_sample_rate` and `duration_sec`.

### 8. Clean target: bandwidth + length aligned

The clean target is constructed separately from the unmodified clean working waveform.
Its bandwidth policy follows the selected channel:

- Narrowband inputs receive a narrowband-filtered clean target.
- Wideband inputs receive a wideband-filtered target when
  `channel.wideband.filter_target` is enabled.
- If wideband target filtering is disabled, the target remains normal wideband audio,
  intentionally making bandwidth extension part of the learning objective.

The target is resampled to the model rate and length-matched to the degraded waveform.
`target_bandwidth` records which target policy was used.

This block is distinct from codec-delay alignment. Codec delay is compensated on the
degraded channel waveform first; target length matching occurs later at model rate.

### 9. Shared pair peak safety

A single normalization scale is computed from the largest absolute sample across both the
clean target and degraded input. If either waveform exceeds the configured safe peak, the
same scale is applied to both.

Using one shared scale preserves the relative amplitude relationship between the two sides
of the pair. The output row records the normalization mode and `normalization_scale`.

### 10. WAV pairs + JSONL metadata

The final clean and degraded waveforms are saved as separate WAV files at the model sample
rate. Their paths are added to the JSONL pair manifest.

Each row provides an audit trail containing source identity, transcript, seed, profile,
noise choices, level effects, channel bandwidth, codec parameters, network impairment,
alignment lag, normalization, duration, and output paths. This supports reproducibility,
sample-level inspection, and evaluation grouped by degradation condition.

## Expanded channel, codec, and network stage

### Select codec

The codec is sampled first from the effective profile's `codec_distribution`. This order
matters: for every actual codec, the codec determines whether the channel is narrowband or
wideband.

Supported selections are:

- `g711_alaw`;
- `g711_mulaw`;
- `gsm`;
- `amr_nb_12k2`;
- `amr_wb_12k65`;
- `opus_nb`;
- `opus_wb`; and
- `pass_through`.

Optional bitrate and frame-duration values are sampled from the selected codec entry and
recorded as `codec_bitrate` and `codec_frame_duration_ms`.

### Channel path

Non-pass-through codecs declare their channel path directly:

- G.711, GSM, AMR-NB, and Opus-NB select narrowband.
- AMR-WB and Opus-WB select wideband.

`pass_through` has no codec-defined path, so it samples narrowband or wideband from
`channel.pass_through_path_distribution`.

The selected `channel_path`, `channel_sample_rate`, and `channel_bandpass_hz` are written
to metadata.

### Resample + band-limit

The degraded working waveform is resampled to the channel rate and passed through the
configured band-pass filter. In the checked-in configuration, narrowband processing uses
8 kHz channel audio and a 300–3400 Hz passband, while wideband processing uses 16 kHz and a
50–7000 Hz passband.

These values are configuration settings and therefore are described in this guide but not
printed as constants inside the diagram.

### Pre-codec peak guard

Before any codec branch, the channel waveform receives a peak guard. This prevents the
conversion to PCM16 at a codec boundary from introducing unrecorded hard clipping that
could be mistaken for a codec artifact.

Both the normal ffmpeg path and native Opus PLC path receive this same guarded waveform.
The measured peak and applied scale are recorded as `pre_codec_peak` and
`pre_codec_guard_scale`.

## Codec and network branches

### Opus with network impairment enabled

When the selected codec is Opus, network impairment is enabled for the variant, and the
configured mode is `packet_loss_plc`, the implementation uses the native Opus library
rather than an ffmpeg container round-trip.

The guarded waveform is divided into frames and encoded into individual Opus packets. A
two-state burst model marks some packets as lost. Lost packets are withheld from the
decoder, which is invoked without packet data so that its packet-loss concealment (PLC)
generates replacement audio. Received packets are decoded normally.

This branch models loss at the packet boundary and therefore is the pipeline's primary
VoIP loss simulation. The effective frame duration uses the selected Opus frame duration
when present, otherwise the network frame setting.

### Opus without network impairment

When Opus is selected but network impairment does not fire, it follows the ffmpeg
encode/decode branch. No packets are dropped and no decoded-waveform fallback is applied.

This distinction is why the ffmpeg branch in the diagram is labeled “non-Opus; or Opus
without loss.”

### Other codecs through ffmpeg

G.711, GSM, AMR-NB, and AMR-WB are encoded and decoded through actual ffmpeg codec
implementations. The generator writes temporary PCM16 input, encodes it using the selected
codec, decodes it back to mono waveform audio, and returns it at the channel sample rate.

When network impairment is disabled, this decoded waveform proceeds directly to delay
estimation and alignment.

### Decoded-waveform frame-dropout fallback

When network impairment is enabled for a codec that cannot use the native Opus PLC path,
the generator first performs the normal codec round-trip and then zeros frames in the
decoded waveform using the same two-state burst process.

This branch is recorded as `decoded_waveform_dropout_fallback` unless the configuration
explicitly requests `decoded_waveform_dropout`. It is an approximation, not true encoded
packet loss, and should not be interpreted as evidence of realistic PLC behavior for those
codecs.

The same fallback also applies when `pass_through` is selected and network impairment is
enabled. In that case there is no codec round-trip, but the channel-filtered waveform can
still receive frame dropout.

### Pass-through

Pass-through retains channel-rate resampling, band limiting, and the pre-codec peak guard,
but bypasses codec encoding and decoding. It therefore isolates channel bandwidth and
front-end effects from codec artifacts.

Two exits are possible:

- With network impairment disabled, the waveform goes directly to delay estimation and
  alignment.
- With network impairment enabled, it first receives decoded-waveform frame dropout.

Pass-through does not mean “skip the complete channel stage.” It only bypasses the codec
round-trip.

## Delay estimation + alignment

Every branch rejoins at alignment. The processed waveform is cross-correlated against the
guarded pre-codec channel waveform within a bounded lag window. The estimated lag is
compensated, and the result is matched to the reference length.

For a true codec path, this removes encoder/decoder delay that would otherwise cause the
enhancement loss to compare shifted signals. Pass-through still uses the same common
alignment function; its estimated lag will normally be zero unless another branch effect
changes the correlation result.

The measured value is stored as `codec_alignment_lag_samples` even for pass-through,
because all branches share one metadata contract.

## Network metadata

The `network_impairment` object always exists. When impairment is disabled, it records
`enabled: false` and null detail fields. When enabled, it records:

- `mode` and `model`;
- requested `loss_rate`;
- sampled `burst_length`;
- effective `frame_ms`;
- `dropped_frames` and `total_frames`;
- `observed_loss_rate`; and
- total `dropout_ms`.

This makes native Opus packet loss distinguishable from decoded-waveform fallback during
analysis.

## Diagram and thesis assets

The standalone figure consists of:

- [`index.html`](index.html), which provides the accessible page and export controls;
- [`diagram.js`](diagram.js), which declares and renders the SVG scene; and
- [`styles.css`](styles.css), which provides responsive, print, and SVG presentation
  styles.

The thesis uses a 3× raster export at
[`Thesis/figs/degradation-pipeline-architecture.png`](../../Thesis/figs/degradation-pipeline-architecture.png).
It is placed as a dedicated landscape figure page in
[`Thesis/chapters/work.tex`](../../Thesis/chapters/work.tex) so that branch labels remain
legible in print.

For a broader methodological discussion, including current profile probabilities and
parameter ranges, see
[`docs/speech-degradation-pipeline.md`](../speech-degradation-pipeline.md).
