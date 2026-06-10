# Degradation Pipeline Fixes (June 2026)

This document records a set of correctness fixes applied to the telephone/VoIP
degradation pipeline (`ml/speech_data/generate_degraded_pairs.py`). All fixes share one
goal: the JSONL manifest must never disagree with what actually happened to the audio.

All changes live in `degrade_item` and its helpers, so they apply equally to
`generate_degraded_pairs`, `generate_degraded_dataset`, the random-clip demo script, and
the web demo page.

## 1. Pre-codec peak guard (unrecorded hard clipping)

**Problem.** The signal entering the codec stage could exceed ±1.0 (low-SNR noise mixing
plus up to +6 dB gain), because the shared pair normalization only runs at the very end.
The ffmpeg round-trip wrote the intermediate WAV as `PCM_16`, which silently hard-clips
anything above full scale — an unrecorded distortion. The Opus packet-loss PLC path fed
floats directly to `opus_encode_float`, so the two paths treated overload differently.

**Fix.** A `pre_codec_peak_guard` step now runs right after the channel band-pass filter,
before either codec path. If the waveform peak exceeds 0.99 (`PRE_CODEC_GUARD_PEAK`), the
whole waveform is scaled down so the codec input is never clipped, and both codec paths
receive the same guarded signal. Two new manifest fields record it:

- `pre_codec_peak` — the peak amplitude measured before the guard.
- `pre_codec_guard_scale` — the scale applied (`1.0` means the guard was a no-op).

## 2. `normalization.mode` is now validated

**Problem.** The config knob `normalization.mode` was copied into the manifest but never
controlled behavior — the code always applies shared-pair peak-safety normalization. The
checked-in `degradation.yaml` said `peak_safety`, so manifests claimed a mode that
disagreed with the docs and with the actual behavior.

**Fix.** `validate_config` rejects any mode not in `SUPPORTED_NORMALIZATION_MODES`
(currently only `shared_pair_peak_safety`). The stale value was corrected in
`configs/speech_enhancement/degradation.yaml` and in the web demo service
(`server/app/services/speech_degradation_demo.py`).

## 3. Missing ffmpeg encoder now raises instead of passing through

**Problem.** If `codec_roundtrip` could not resolve an ffmpeg encoder it returned the
input audio unchanged, while the manifest still recorded the codec as applied — clean
audio labeled as codec-degraded.

**Fix.** `codec_roundtrip` raises `RuntimeError` when a codec has encoder candidates but
none is available. `pass_through` (which has no candidates by design) still returns the
input unchanged. The startup check `require_ffmpeg_codecs` makes the error unreachable in
normal runs; the raise is a guarantee, not a new failure mode.

## 4. Clipping metadata records the actual clipped fraction

**Problem.** When the clipping stage fired, the manifest recorded `enabled: true` even if
no sample exceeded the sampled threshold (common after a negative gain draw), making
clipping ablations unreliable.

**Fix.** The clipping metadata gains a `clipped_fraction` field — the fraction of samples
whose magnitude exceeded the threshold before `np.clip` was applied (`null` when the
stage did not fire, `0.0` when it fired but clipped nothing).

## 5. Noise probability draw is now unconditional

**Problem.** The noise-probability `rng.random()` draw was only consumed when a noise
index was configured. Running the same config with and without a noise index therefore
changed *every* downstream random choice (codec, loss rate, gain, …), not just the noise
fields — undermining seed-comparable ablations.

**Fix.** The draw is consumed unconditionally; the noise stage still only applies when
noise assets exist and the draw passes the probability threshold.

## 6. Gilbert burst-loss chain starts from its stationary distribution

**Problem.** `sample_burst_loss_mask` always started the two-state chain in the good
state with no burn-in, so short clips systematically under-shot the target loss rate.

**Fix.** The initial state is sampled from the stationary distribution
(`P(bad) = target_loss_rate`), consuming one extra RNG draw. `observed_loss_rate` in the
manifest remains the ground truth for what was actually dropped.

## Reproducibility impact

Fixes 5 and 6 change the RNG draw order, and fix 1 can change waveform amplitudes near
full scale. **Datasets generated before these fixes are not sample-identical to datasets
generated after them, even with the same seed and config.** Regenerate any existing pairs
rather than mixing pre- and post-fix outputs in one training set. Determinism itself is
unchanged: the same config, manifests, and assets still reproduce the same dataset.

## New manifest fields

| Field | Meaning |
| --- | --- |
| `pre_codec_peak` | Peak amplitude of the channel waveform before the codec-input guard. |
| `pre_codec_guard_scale` | Scale applied by the guard (`1.0` = no-op). |
| `clipping.clipped_fraction` | Fraction of samples actually clipped (`null` if the stage did not fire). |

## Verification

The full test suite (`make test`, 132 tests) passes, including the degradation
determinism, codec round-trip, and web demo tests. The demo-page test was updated for the
corrected normalization mode string.
