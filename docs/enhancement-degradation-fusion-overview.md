# Log-Mel Enhancement, Speech Degradation, and Fusion Architecture

This note answers three implementation questions about the final dual-view fusion system.
Unless stated otherwise, architecture hyperparameters refer to the latest recorded experiment,
**fusion v2 / run 004**, rather than the smaller defaults in the reusable training config.

## Why enhance log-Mel features instead of raw audio?

The main reason is that the enhancement module is an **ASR front end**, not a general-purpose
audio restoration model. Whisper consumes an 80-bin log-Mel spectrogram, so enhancing that
representation directly makes the enhancer optimize the same information that the recognizer
uses.

More specifically:

1. **It matches Whisper's input domain.** The degraded waveform is converted once to Whisper's
   canonical log-Mel representation, and the enhancer maps noisy log-Mel to estimated clean
   log-Mel. Its output can therefore go directly into the shared Whisper encoder.
2. **It removes an unnecessary reconstruction loop.** A waveform-domain or STFT-domain
   enhancer would have to estimate a waveform (including phase), synthesize it, and then have
   Whisper compute another log-Mel spectrogram. Direct log-Mel enhancement avoids this
   waveform/STFT/Mel round-trip and the artifacts or information loss it can introduce.
3. **It makes the objective recognition-oriented.** The enhancer is trained with log-Mel L1
   reconstruction and, in fusion v2, Whisper-encoder feature matching. During joint training,
   ASR cross-entropy also sends gradients through the fusion block and enhancer. This is more
   closely aligned with word recognition than optimizing a perceptual waveform metric such as
   PESQ or a generic magnitude-spectrum MSE.
4. **It is computationally practical.** An 80-bin feature map is much smaller than a 16 kHz
   waveform or a high-resolution complex spectrogram. The lightweight enhancer can therefore
   be trained beside Whisper-small on the available RTX 3090.
5. **It fits the dual-view design.** The fusion model needs two complementary views: the
   original noisy log-Mel, which may retain weak phonetic details, and an enhanced log-Mel,
   which suppresses corruption but may over-smooth speech. Both views have the same shape and
   statistics and can be processed by the same Whisper encoder.

The exact interface is

```text
16 kHz degraded waveform
        |
        v
Whisper feature extractor -> noisy log-Mel [B, 80, 3000]
        |                                  |
        |                                  v
        |                         enhancer E(noisy log-Mel)
        |                                  |
        v                                  v
 shared Whisper encoder            shared Whisper encoder
        |                                  |
        +---------- feature-space fusion --+
                           |
                     Whisper decoder
```

This choice also has a limitation: log-Mel features discard phase and fine waveform detail.
The enhancer is therefore not intended to synthesize high-quality restored audio, and it
cannot truly reconstruct speech removed by packet loss. That trade-off is acceptable here
because transcription accuracy, not waveform playback quality, is the target.

## Exact degradation pipeline

The following is the execution order in `degrade_item`, including the branch used for real
Opus packet-loss concealment. Each variant receives a stable seed derived from the global
seed, split, source clip ID, and variant index.

1. **Load and standardize the source.** Load the clean clip as mono and resample it from its
   source rate to the working rate, currently 16 kHz. Copy it to create the degraded branch.
2. **Select a degradation profile.** Sample one weighted profile and merge its overrides into
   the base configuration. The checked-in profiles are `telephone_clean` (weight 0.25),
   `telephone_noisy` (0.30), `voip_lossy` (0.25), and `mobile_wideband` (0.20).
3. **Optionally mix environmental noise.** If the selected profile's noise draw succeeds and
   indexed noise is available, select one noise scene or two scenes (10% second-scene
   probability in the base config). Resample and length-match the noise, average two selected
   scenes when applicable, sample an SNR from one of `[10,15]`, `[5,10]`, `[0,5]`, or
   `[-5,0]` dB, and mix it with the speech at that SNR.
4. **Apply source/device level effects.** Sample a gain uniformly from -6 to +6 dB and apply
   it. Optional hard clipping can then be applied at a sampled threshold of 0.80-0.98, but it
   is disabled in the checked-in config. AGC is currently only recorded as disabled metadata;
   no AGC signal processing is implemented.
5. **Select codec and channel path.** Sample a codec from the active profile. A codec fixes
   the channel as narrowband or wideband. `pass_through` instead samples a path from the
   configured 50/50 narrowband/wideband distribution.
6. **Simulate channel bandwidth.** Resample to 8 kHz and band-pass 300-3400 Hz for the
   narrowband path, or use 16 kHz and band-pass 50-7000 Hz for the wideband path.
7. **Guard the codec input peak.** Measure the post-filter waveform peak. If it exceeds 0.99,
   scale the entire waveform to a 0.99 peak before codec processing. This prevents an
   unrecorded PCM-16 hard clip and records both the original peak and applied scale.
8. **Perform codec and optional network processing.** There are three exact branches:

   - **Network disabled:** run the selected codec through a real ffmpeg encode/decode
     round-trip. `pass_through` returns the channel-filtered signal unchanged.
   - **Network enabled, Opus, `packet_loss_plc`:** encode the waveform frame by frame with
     libopus, sample a two-state burst-loss mask over encoded packets, and call the Opus
     decoder with a missing packet for every dropped frame. The decoder's packet-loss
     concealment (PLC) produces the replacement audio. The configured Opus bitrate and frame
     duration are honored.
   - **Network enabled, non-Opus (or explicit waveform-dropout mode):** first run the ffmpeg
     codec round-trip, then zero decoded waveform frames using the same two-state burst model.
     This is recorded as `decoded_waveform_dropout_fallback` for a non-Opus PLC request, or
     `decoded_waveform_dropout` when explicitly requested. It is not true packet-level PLC.

   The configured target loss-rate ranges are 0.3-2%, 2-5%, and 5-10%; the sampled expected
   burst length is 1-5 frames. The default fallback frame size is 20 ms, while Opus PLC uses
   the selected codec frame duration when one is present.
9. **Compensate codec delay.** Cross-correlate the decoded/degraded channel waveform against
   the pre-codec channel waveform within a +/-50 ms search window, shift it by the estimated
   lag, and match its length to the reference.
10. **Return to the model rate.** Resample the degraded channel waveform to 16 kHz and force
    its sample count to the clean working waveform's duration.
11. **Construct the clean training target.** Start from the uncorrupted clean working
    waveform. For narrowband examples, apply the same 8 kHz and 300-3400 Hz channel filtering;
    for wideband examples, the checked-in config also applies the 50-7000 Hz filter because
    `filter_target: true`. Resample the target to 16 kHz and length-match it to the degraded
    waveform. This bandwidth alignment avoids asking the enhancer to hallucinate frequencies
    that the selected channel removed.
12. **Normalize the pair together.** Replace non-finite values safely, find the largest
    absolute sample across both clean and degraded signals, and—only if it exceeds 0.99—apply
    the same scale to both. Using a shared scale preserves their relative levels.
13. **Write outputs and metadata.** Save the aligned 16 kHz WAV data and record the seed,
    profile, noises, SNR, gain, clipping, channel, codec, codec parameters and delay, peak
    guard, loss statistics, target bandwidth, and final normalization scale in JSONL.

The supported codec set is G.711 A-law, G.711 mu-law, GSM, AMR-NB 12.2 kb/s, AMR-WB
12.65 kb/s, Opus narrowband, Opus wideband, and pass-through. The pipeline does **not**
currently model reverberation, jitter-buffer timing, late or duplicate packets, echo, DTX,
or comfort noise.

## Model architectures

### Enhancer: residual 2D U-Net with a temporal Transformer bottleneck

The fusion-v2 enhancer treats a log-Mel spectrogram as a single-channel image with shape
`[B, 1, 80, T]`. Its recorded configuration is:

| Component | Fusion-v2 setting |
| --- | --- |
| Architecture | Residual 2D convolutional U-Net |
| Base channels | 48 |
| Down/up levels | 3 |
| Channel widths | 48 -> 96 -> 192 -> 384 -> 192 -> 96 -> 48 |
| Convolution block | Two 3x3 convolutions, each followed by GroupNorm and GELU |
| Downsampling | 3x3 stride-2 convolution |
| Upsampling | 2x2 stride-2 transposed convolution, skip concatenation, then convolution block |
| Temporal bottleneck | 2-layer Transformer encoder, 4 heads, dimension 256, FFN dimension 1024, dropout 0 |
| Output | Zero-initialized 1x1 convolution predicts `delta`; result is `noisy_mel + delta` |

After three downsampling levels, the 80-bin frequency axis becomes 10 bins and the feature
width becomes 384. The temporal bottleneck reshapes `[B, 384, 10, T/8]` into a sequence,
projects each time step from `384 x 10 = 3840` features to 256, processes it with the
Transformer, projects back to 3840, and adds the result residually. The projection back is
zero-initialized, as is the U-Net's final 1x1 convolution, so the complete enhancer starts as
an identity mapping. This protects Whisper from arbitrary enhanced features at the beginning
of training.

Fusion v1 used the same three-level, 48-channel residual U-Net but **without** the temporal
bottleneck. Fusion v2 added the Transformer because the purely convolutional model had only
short local context, which was weak for non-stationary noise and longer corruption patterns.
The implementation also supports a bidirectional-GRU bottleneck, but the recorded v2 run used
the Transformer.

### Fusion: shared Whisper encoding, bidirectional cross-attention, and a learned gate

The fusion module does not merge the two spectrograms at the input. Instead:

1. The original noisy log-Mel and the enhanced log-Mel are encoded separately by the **same
   shared-weight Whisper-small encoder**.
2. The resulting noisy and enhanced hidden-state sequences pass through **three layers of
   bidirectional cross-attention**. At every layer, the noisy stream queries the enhanced
   stream while the enhanced stream queries the noisy stream; both refinements use the
   pre-update pair of streams, making the exchange symmetric.
3. Each directional block is pre-normalized multi-head attention with a residual connection,
   followed by a pre-normalized position-wise FFN and another residual connection. Fusion v2
   uses 12 attention heads and an FFN expansion ratio of 4.
4. A feature-wise sigmoid gate merges the refined streams:

   ```text
   g = sigmoid(MLP(concat(noisy_hidden, enhanced_hidden)))
   fused = g * enhanced_hidden + (1 - g) * noisy_hidden
   ```

   The gate is learned for every encoder time step and feature channel, allowing the model to
   retain noisy-stream detail in one region and prefer enhancement in another.
5. The fused hidden-state sequence is passed directly to the Whisper decoder for
   autoregressive transcription.

The attention output projections, FFN output projections, and final gate output layer are
zero-initialized. Consequently, every cross-attention block initially acts as the identity
and `g = 0.5`, so the fusion block starts as the balanced mean of the two encoder streams.
The code also contains a simpler position-wise gated-fusion baseline, but both recorded fusion
v1 and v2 experiments used the cross-attention architecture; v2 used three layers, 12 heads,
and an FFN ratio of 4.

## Implementation sources

- Enhancer implementation: [`ml/enhancement/enhancer.py`](../ml/enhancement/enhancer.py)
- Fusion implementation: [`ml/fusion/model.py`](../ml/fusion/model.py)
- Whisper log-Mel extraction: [`ml/asr/whisper_features.py`](../ml/asr/whisper_features.py)
- Degradation implementation: [`ml/speech_data/generate_degraded_pairs.py`](../ml/speech_data/generate_degraded_pairs.py)
- Degradation configuration: [`configs/speech_enhancement/degradation.yaml`](../configs/speech_enhancement/degradation.yaml)
- Fusion-v2 experiment configuration: [`report/whisper-fusion-v2/fusion_train_v4.yaml`](../report/whisper-fusion-v2/fusion_train_v4.yaml)
- Fusion-v2 architecture and results report: [`report/progress-report-2.md`](../report/progress-report-2.md)

