# Speech Enhancement and Fusion: Design Decisions and Rationale

This document records *why* the speech-enhancement + fusion approach is designed the way it
is, as a companion to `speech-enhancement-and-fusion-plan.md` (which records *what* to build).
It exists so the reasoning behind each choice can be recovered later without re-deriving it.

Context: the goal is robust Persian telecommunication ASR — transcribing speech corrupted by
codec artifacts, packet loss, jitter, reverberation, and additive noise. The ASR backbone is
Whisper-small, adapted on ~1000 h of general Persian speech. The core idea under
investigation is **dual-view fusion**: give the recognizer two complementary views of each
utterance — a *noisy* view and an *enhanced/clean* view — and let it combine them.

---

## D1. Drop the standalone PrimeK-Net speech-enhancement model

**Decision.** Do not use PrimeK-Net (or any frozen, perceptually-optimized denoiser) as the
enhancement stage.

**Why.**
- PrimeK-Net is trained for perceptual quality (PESQ on VoiceBank+DEMAND). Perceptual quality
  is a *proxy* for, and not aligned with, downstream word error rate. The well-known result in
  robust-ASR literature is that perceptually-optimized front-ends frequently fail to help, and
  sometimes hurt, ASR — they over-suppress low-energy phonetic cues and add processing
  artifacts.
- Using it frozen meant its output could never adapt to what the recognizer actually needs.
- It forced a waveform -> Mel round-trip, adding avoidable loss.

**Alternatives considered and rejected.**
- Swapping in a different off-the-shelf SE model (MP-SENet, DeepFilterNet3, GTCRN,
  TF-GridNet). Rejected: because the fusion gates the enhanced stream, the SE model is nearly
  interchangeable; no drop-in replacement meaningfully moves WER, so the SE *choice* was the
  wrong lever.
- Generative restoration models (Miipher, VoiceFixer, diffusion SE). Rejected as a front-end:
  they hallucinate plausible-but-wrong content, which is poison for an ASR front-end.

**Consequence.** The enhancement stage becomes a lightweight module trained *for recognition*
(see D3), not a frozen perceptual denoiser.

---

## D2. Keep the dual-view fusion idea (noisy + enhanced)

**Decision.** Retain the central thesis idea: feed the recognizer two complementary views.

**Why.**
- The noisy view preserves fragile phonetic detail (e.g., weak consonants) but is corrupted.
- The enhanced view is cleaner but prone to over-suppression and enhancement artifacts.
- These failure modes are complementary, so a model that can trust the right view per
  time-frequency region should beat either alone — particularly under non-stationary,
  channel-dependent telecom degradation.

This is the actual contribution of the work; the enhancement model is a supporting component,
not the novelty.

---

## D3. Enhancement operates in the log-Mel domain and is trained for recognition

**Decision.** Replace the waveform denoiser with a lightweight module `E` that maps a noisy
Whisper log-Mel to an estimated clean log-Mel, optimized end-to-end for ASR (D7), with an
auxiliary reconstruction loss (D5).

**Why.**
- Whisper consumes log-Mel; producing log-Mel directly removes the waveform/STFT round-trip.
- Optimizing in the same representation the recognizer sees keeps the enhancement
  ASR-relevant rather than perceptually-tuned.
- Lightweight is required to fit alongside Whisper-small on a single RTX 3090.

**Status.** The exact architecture (block types, depth, parameter budget) is deliberately
left open; this decision fixes only the interface and objective.

---

## D4. Move fusion off the frozen-Mel-input bottleneck

**Decision.** Abandon the previous design that fused two streams into a single log-Mel and fed
it to a frozen Whisper via a late convex mask.

**Why the previous design failed ("loses completely").**
- It fused at the rawest possible point (the Mel input), forcing all complementary information
  to be reconciled before the encoder did any contextual processing.
- Critically, the fused tensor was a convex combination of *learned* single-channel feature
  maps (Conv+BN+PReLU), not real log-Mels. So the claimed "in-distribution" guarantee was
  false: a frozen Whisper received input whose scale and statistics did not match log-Mel.
  This is the most likely cause of the catastrophic result.

**Caveat.** The old fusion code is not in this repository, so the out-of-distribution
diagnosis is inferred from the method chapter's math, not confirmed by reading the code. If
that code resurfaces, the diagnosis should be verified.

**Consequence.** Fusion will combine the two views in a form the recognizer can exploit, and
the recognizer is no longer frozen (D7). The exact fusion mechanism and its location (early
channel-level vs. encoder-feature-space) is left open.

---

## D5. Auxiliary loss is log-Mel L1, not magnitude-spectrum MSE

**Decision.** The enhancement auxiliary loss is `L_enh = || E(noisy_mel) - clean_mel ||_1`,
computed in the log-Mel domain.

**Why each part.**
- **Log-Mel domain, not linear magnitude:** the loss is computed on the 80-bin log-Mel that
  Whisper actually consumes. A loss on the linear STFT magnitude would optimize a
  representation the recognizer never sees, and the Mel module has no linear magnitude to
  measure anyway.
- **L1, not L2/MSE:** L1 is more robust to outliers and over-smooths less. MSE produces
  blurry, mean-reverting spectra, which erases the low-energy phonetic detail that matters for
  recognition.
- **Log compression:** errors are weighted perceptually rather than dominated by high-energy
  bins.
- **Bandwidth-aligned target:** because the data includes narrowband paths (8 kHz, ~300-3400
  Hz), the clean target must be the band-aligned reference per channel path. Otherwise the
  module is penalized for not hallucinating frequencies the channel genuinely removed.

**Future note.** If a later variant outputs a waveform, this loss should become a
multi-resolution STFT and/or SI-SDR loss instead.

---

## D6. Combined objective = ASR cross-entropy + small auxiliary enhancement term

**Decision.** `L = L_ASR + lambda * L_enh`, with `lambda` small (~0.1-0.3).

**Why.**
- `L_ASR` (Whisper autoregressive cross-entropy) is the real objective and the source of
  recognition-aware gradients for the enhancer.
- `L_enh` regularizes the enhanced stream so it stays a genuine "clean view" and does not
  collapse into a copy of the noisy stream.
- `lambda` is kept small so enhancement supports, not dominates, recognition.

---

## D7. Allow full fine-tuning (do not keep everything frozen)

**Decision.** Fine-tune the full stack (enhancer + fusion + Whisper) rather than freezing
Whisper and PrimeK-Net and training only a fusion module.

**Why.**
- The original frozen design existed to give a clean attribution argument ("any gain is due to
  fusion alone"), but it capped the achievable ceiling and, combined with D4, produced a
  system that lost completely.
- Unfreezing removes the out-of-distribution failure mode (Whisper adapts to whatever the
  front-end emits) and lifts the performance ceiling.

**Trade-off accepted.** The pure-attribution story weakens. It is recovered partially by the
controlled ablation in D9 and optionally by using LoRA/adapters (D8) so that most weights stay
fixed.

---

## D8. Train in three staged steps that converge to joint end-to-end

**Decision.** Train with a curriculum: Stage 0 enhancer warm-up (`L_enh` only) -> Stage 1
enhancer + fusion with Whisper frozen (`L`) -> Stage 2 joint end-to-end (`L`).

**Why not fully separate.** Training the enhancer to convergence on `L_enh` alone, then
freezing it, reintroduces exactly the proxy-objective problem that killed PrimeK-Net: the
enhancer would be optimized for reconstruction, not recognition. The final stage must be
joint.

**Why not cold-start fully joint.** Random enhancer + fusion would feed garbage Mels into the
most valuable pretrained component (Whisper) and damage it before the front-end stabilizes;
optimization across three random interacting modules is also needlessly hard.

**Why the curriculum.** Stage 0 gives the enhancer a sane init; Stage 1 lets the front-end
learn to emit in-distribution Mels while protecting the frozen backbone; Stage 2 delivers the
payoff — recognition-aware gradients reaching the enhancer end-to-end.

**Supporting knobs.**
- Discriminative learning rates in Stage 2 (Whisper much lower than front-end) for gentle
  backbone adaptation.
- `lambda` annealed higher in Stage 1, lower in Stage 2.
- LoRA/adapters on Whisper in Stage 2 as the fallback if full fine-tuning does not fit the
  RTX 3090 or trains unstably; this also reduces catastrophic forgetting and partially
  restores the attribution story.

---

## D9. Required baseline and ablation

**Decision.** The baseline the method must beat is **Whisper-small fine-tuned directly on the
degraded data (single noisy stream)** — not the frozen baseline. Minimal ablation: noisy-only
fine-tune vs. enhanced-only vs. dual-view fusion.

**Why.**
- Once the stack is fully fine-tuned (D7), simple multi-condition fine-tuning on noisy data is
  a strong and simple competitor that a reviewer will ask about first. If the dual-view does
  not beat it, the contribution collapses.
- The expected win is concentrated on the most degraded subsets (codec artifacts, packet loss,
  low SNR); the ablation should report per-condition, not just averages.

---

## D10. One config-driven training script, one output location, resumable per stage

**Decision.** Implement the curriculum (D8) as a single orchestrator script driven by one
config that runs all three stages in one invocation and writes every artifact to one run
directory. The script checkpoints at each stage boundary and supports resume-from-stage.

**Why.**
- One script + one config + one output directory maximizes reproducibility and makes a run
  self-contained and traceable.
- Resume-from-stage is insurance: Stage 2 (joint) is the most likely to OOM or diverge on a
  3090, and a crash there should not force re-running the cheap Stage 0/1 warm-ups. Default
  behavior remains a single end-to-end run.

---

## Open decisions (intentionally deferred)

These were consciously left unresolved and should be decided before/while implementing:

- Exact architecture and parameter budget of the Mel-domain enhancer (D3).
- Exact fusion mechanism and its location in the stack: early channel-level fusion vs.
  encoder-feature-space fusion (D4).
- Full fine-tuning vs. LoRA/adapters for Whisper in Stage 2 (D7, D8), pending observed Stage 2
  memory on the RTX 3090.
- The auxiliary-loss weight `lambda` and its annealing schedule (D6, D8).

---

## Known limitations to acknowledge

- **Packet loss is a reconstruction problem, not a denoising one.** A discriminative
  Mel-domain enhancer cannot regenerate audio lost to packet drops; it can only suppress
  corruption. The fusion design mitigates this (the model can lean on the noisy view in
  dropped regions), but full reconstruction would need a generative packet-loss-concealment
  model, noted as future work.
- The out-of-distribution diagnosis for the previous design (D4) is inferred, not verified
  against the original code.
