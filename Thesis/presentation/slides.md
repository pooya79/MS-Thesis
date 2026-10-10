# Robust Speech Recognition in Persian Language

Replacement defense presentation based on the current thesis chapters. English, 16:9. **20 main slides plus 10 hidden backup slides.** Main speaking budget: 19:05, leaving 55 seconds for transitions in a 20-minute slot.

## Recommended organization

Organize the defense around the experiments, rather than repeating the thesis chapter order. Introduce only the background needed to explain each comparison. Present each method immediately before its evidence. Spend the most time on the real-call transfer result and the domain trade-off after adding IranSeda.

| Slides | Topic | Why it belongs here |
|---|---|---|
| 1–5 | Problem, comparisons, data and evaluation | Establish the question and the controls before results |
| 6–8 | Telecom augmentation and real-call transfer | Present the largest controlled robustness gain first |
| 9–12 | Fusion, training, results and Tiny ablations | Explain the architecture and bound its small-model evidence |
| 13–15 | Audiobook preparation, curation and domain effects | Connect label quality and data relevance to the final experiment |
| 16–17 | Main comparison and Medium context | Separate Whisper-small comparisons from capacity changes |
| 18–20 | Contributions, limits and conclusion | State what the evidence establishes and what remains open |
| 21–30 | Hidden backup material | Keep full tables and detailed figures available for questions |

## Presentation choices

- Keep the official thesis title. Do not invent a new defense date.
- Explain WER once. Use CER as corroborating evidence rather than a second full narration.
- Make real AGFarsdat speech the central robustness test.
- Fusion and IranSeda are separate experiments from the multi-condition reference. No fusion-plus-IranSeda result exists in the current thesis.
- Use “numerical improvement” for small margins. The thesis does not establish statistical significance.
- End on the conclusion slide. Slides 21–30 are hidden during slideshow playback and can be opened for questions.
- Use the detailed architecture figures in the appendix when the committee asks about implementation. Main slides keep the mechanism in readable text and editable tables.

## Source precedence and issues to resolve before defense

The current `.tex` chapters and English abstract are the sources for this presentation. Outdated claims about unsuccessful fusion, a collapsed gate and a pending IranSeda experiment are excluded. The thesis source itself has not been changed.

- Current fusion: aggregate WER/CER **17.64% / 7.85%**. Enhanced-stream gate mean **0.6785**, median **0.6930**. Do not repeat 18.44% WER, 14.04% CER or 0.0039 gate weight from old material.
- IranSeda training and evaluation are complete: **17.22% WER / 7.70% CER**. The accepted corpus is **218.564 hours**, distinct from the **250-hour raw subset** selected for preparation.
- Differences calculated from displayed scores: 17.83 − 17.22 = **0.61** WER points, 23.27 − 17.22 = **6.05** points, 12.96 − 12.62 = **0.34** CER points, and 3.29 − 3.22 = **0.07** CER points. Thesis prose/table deltas sometimes report 0.60, 6.04, 0.35 and 0.08 respectively. This deck preserves endpoint scores and uses arithmetic from the displayed values. Raw unrounded counts would be needed to reconcile the original deltas.
- The thesis explicitly establishes **21,848 common eligible examples** for the paired single-stream comparisons. Its dataset table gives **391–396** Persian Speech Corpus examples across systems. Confirm exact paired eligibility for fusion/bridge comparisons before treating small ranking differences as controlled effects. Do not carry forward the old deck’s unverified six-extra-samples claim.
- The selected IranSeda subset is 250 raw hours. Do not claim that 6,300 hours or all 3,777 books entered training.
- Qwen permits conservative, context-supported ASR substitutions as well as orthographic cleanup. It is text-only. Do not describe its changes as human-verified transcription.

## Slide-by-slide rehearsal guide

### 1. Robust Speech Recognition in Persian Language

Timing: 25 seconds.

Pouya Fallah Sichani

Supervisor: Dr. Hosein Sameti

Sharif University of Technology

**Speaker notes**

This thesis studies Persian ASR under telephone and VoIP conditions. I compare improvements from matching training conditions, adding an enhancement-fusion architecture, and expanding training data with pseudo-labelled audiobooks. The largest controlled robustness gain comes from telecom-oriented multi-condition training. The later experiments show smaller and domain-dependent gains.

**Sources:** `Thesis/front/info.tex`, `Thesis/front/abstract-eng.tex`

### 2. The mismatch in Persian telephone ASR

Timing: 50 seconds.

Available speech: General Persian corpora, many short clips, varied recording conditions

Target speech: Telephone and VoIP with noise, limited bandwidth, codecs and packet loss

Research objective: Improve real-call recognition using existing speech and unlabelled audio

The test must include real calls that the simulator never generated.

**Speaker notes**

Persian training data includes read and spontaneous speech, but does not fully cover real telecom channels. A call can combine environmental noise, bandwidth restriction, codec distortion, gain changes and packet loss. Collecting large quantities of transcribed calls is expensive and privacy-sensitive. The goal is to improve recognition on real calls while monitoring performance on general Persian speech.

**Sources:** `Thesis/chapters/introduction.tex`, `Thesis/chapters/preliminaries.tex`

### 3. Research design

Timing: 55 seconds.

| Comparison | What changes | Reference |
|---|---|---|
| Telecom augmentation | Add degraded speech | Ordinary Whisper-small |
| Enhancement and fusion | Add dual streams and staged training | Multi-condition Whisper-small |
| Audiobook data | Add accepted IranSeda clips | Multi-condition Whisper-small |

Whisper-small remains the recognizer for the main comparisons.
Whisper Medium and Fast Conformer provide additional context.

**Speaker notes**

The ordinary Whisper-small baseline includes original and constructed long examples. The multi-condition baseline adds degraded speech. It becomes the reference for both later comparisons. Fusion changes the system architecture and training schedule. The IranSeda experiment retains the complete multi-condition mixture and adds accepted audiobook clips. Fusion and IranSeda are separate branches. I did not train a combined fusion-plus-IranSeda system.

**Sources:** `Thesis/chapters/work.tex`

### 4. A shared Persian training foundation

Timing: 55 seconds.

| Base corpus | Training clips | Validation clips |
|---|---|---|
| Common Voice Persian 25 | 299,430 | 10,006 |
| FarsSpon, limited subset | 600,672 | 6,044 |
| FLEURS Persian | 3,019 | 362 |

Mono 16 kHz audio and one Persian text normalizer

Long variants join 2–4 clips within a split, with durations above 5 and at most 20 seconds.

**Speaker notes**

The base corpora are Common Voice Persian 25, FarsSpon and FLEURS Persian. All audio uses mono 16 kHz and transcripts share deterministic Persian normalization. Constructed long examples concatenate two to four clips within the same split, with final duration above five and at most twenty seconds. Every main model uses these long variants. Multi-condition training additionally uses two independent degraded variants for each Common Voice clip and its long variant. These counts describe base examples before derived data and IranSeda.

**Sources:** `Thesis/chapters/work.tex`

### 5. Evaluation on real calls and general speech

Timing: 60 seconds.

| Test set | Examples | Role |
|---|---|---|
| AGFarsdat | 10,044 | Real telephone and VoIP |
| Common Voice Persian 25 | ~10,518 | Crowdsourced general speech |
| FLEURS Persian | 871 | Read speech across domains |
| PersianSpeech | 24 | Small supplementary test |
| Persian Speech Corpus | 391–396 | Studio speech outside training domain |

WER = (substitutions + deletions + insertions) / reference words
CER counts characters. Aggregate scores pool counts. Lower is better.

**Speaker notes**

WER is substitutions plus deletions plus insertions divided by reference words. CER uses the same calculation over characters. Lower is better. All main systems use greedy decoding and identical Persian normalization. Aggregate metrics pool error counts and reference units rather than average dataset percentages. The paired single-stream comparisons use 21,848 common eligible examples. The thesis lists a 391 to 396 range for Persian Speech Corpus across models, so model-wide rankings require care about eligibility. AGFarsdat is held out from training and model selection.

**Sources:** `Thesis/chapters/work.tex`, `Thesis/chapters/results.tex`, `Thesis/chapters/conclusions.tex`

### 6. Reproducible telecom augmentation

Timing: 75 seconds.

| Stage | Implementation |
|---|---|
| Noise and gain | DEMAND noise, SNR from −5 to 15 dB, gain ±6 dB |
| Channel and codec | Narrowband / wideband, real codec encode–decode |
| Network effects | Opus packet loss + PLC, waveform-dropout approximation elsewhere |
| Aligned output | 16 kHz pair, matched bandwidth target, saved seed and metadata |

G.711, GSM, AMR-NB, AMR-WB and Opus cover the implemented codec paths.

**Speaker notes**

Four sampled profiles cover clean telephone, noisy telephone, lossy VoIP and mobile wideband. Their weights describe a designed training distribution, not measured operator traffic. Noise and gain changes precede channel filtering and codec processing. FFmpeg round-trips model codec distortion. Native Opus supports packet loss and decoder PLC. Other paths approximate loss through waveform dropouts. The final aligned pair has a shared transcript, compensated delay and a common peak scale. The clean enhancement target has the same bandwidth as its degraded input. Seeds and settings make each example traceable.

**Sources:** `Thesis/chapters/work.tex`

### 7. Synthetic training transfers to real calls

Timing: 70 seconds.

WER (%) on the aggregate evaluation and held-out real calls. Lower is better.

| Category | Ordinary Whisper-small | Multi-condition Whisper-small |
|---|---|---|
| Aggregate | 23.27% | 17.83% |
| AGFarsdat | 35.63% | 25.80% |

5.44 points lower overall                         9.83 points lower on real calls

**Speaker notes**

This is the main robustness result. Multi-condition training lowers aggregate WER by 5.44 percentage points, a 23.4 percent relative reduction. On AGFarsdat, WER falls by 9.83 points, a 27.6 percent relative reduction. AGFarsdat contains real telephone and VoIP recordings and did not participate in training or simulator construction. This is evidence of transfer to the target channel. It does not establish uniform improvement across every general speech dataset.

**Sources:** `Thesis/chapters/results.tex`

### 8. Noise alone leaves part of the gap

Timing: 55 seconds.

| Whisper-small training | Aggregate WER | AGFarsdat WER |
|---|---|---|
| Ordinary | 23.27% | 35.63% |
| Noise only | 20.38% | 32.67% |
| pMCT, clean/noisy segment mixing | 19.21% | 30.01% |
| Full telecom multi-condition | 17.83% | 25.80% |

Lower is better. The full pipeline adds bandwidth, codec and network effects.

**Speaker notes**

Both noise-only augmentation and pMCT improve over the ordinary baseline. In the implemented pMCT comparison, aligned clean and noisy examples are divided into one-second pieces, then each piece is selected from either view with equal probability. The full telecom pipeline gives the lowest aggregate and AGFarsdat WER of these augmentation experiments. This supports modelling more than noise. It does not isolate the contribution of each codec or loss mechanism, since those components were not individually ablated.

**Sources:** `Thesis/chapters/results.tex`

### 9. Dual-stream enhancement and fusion

Timing: 80 seconds.

Two views: Original log-Mel and a residual U-Net enhanced view

Shared encoding: One Whisper encoder, two passes with shared weights

Learned combination: Bidirectional cross-attention, then a gate per time and feature

Fused = g ⊙ Enhanced′ + (1 − g) ⊙ Original′

**Speaker notes**

The hypothesis is that a learned enhanced view may complement the original observation. A residual U-Net modifies 80-band log-Mel features without reconstructing a waveform. Original and enhanced features pass through the same Whisper encoder in two passes with shared parameters. One layer of bidirectional cross-attention with four heads exchanges information between streams. A time-feature gate then combines the outputs before the Whisper decoder. Starting from the multi-condition checkpoint makes the comparison more demanding than starting from ordinary Whisper. The diagram in the appendix contains the complete architecture.

**Sources:** `Thesis/chapters/work.tex`

### 10. Three-stage fusion training

Timing: 50 seconds.

| Stage | Trainable components | Objective |
|---|---|---|
| 1. Enhancer warm-up | Enhancer only | 0.5 enhancement + 0.5 feature |
| 2. Frozen recognizer | Enhancer and fusion | ASR + 0.3 enhancement |
| 3. Joint training | All components | ASR + 0.1 enhancement |

One epoch per stage. Final learning rates: added modules 10⁻⁴, Whisper 5 × 10⁻⁶.
Fusion changes both architecture and the training schedule.

**Speaker notes**

Every stage runs for one epoch. Warm-up trains the enhancer using equal-weight log-Mel reconstruction and encoder-feature matching, while Whisper remains frozen. Stage two trains enhancer and fusion with ASR loss plus 0.3 enhancement loss, again with frozen Whisper. Stage three unfreezes the whole model with ASR loss plus 0.1 enhancement loss. The additional modules use a higher learning rate than Whisper. The last stage includes degraded examples, long examples and a randomly selected 25 percent of clean data. Fusion therefore has extra recognizer training relative to the single-stream baseline.

**Sources:** `Thesis/chapters/work.tex`

### 11. Fusion gives a small numerical improvement

Timing: 65 seconds.

| System | Aggregate WER | Aggregate CER | AGFarsdat WER |
|---|---|---|---|
| Multi-condition | 17.83% | 7.93% | 25.80% |
| Dual-stream fusion | 17.64% | 7.85% | 25.37% |
| Adaptive bridge, independent | 17.74% | 8.00% | 25.84% |

Mean enhanced-stream gate weight: 0.6785

Gate weights do not measure independent causal contribution.
Small margins lack confidence intervals and repeated-seed evidence.

**Speaker notes**

Fusion lowers aggregate WER from 17.83 to 17.64 percent, a 0.19-point gain, and AGFarsdat WER from 25.80 to 25.37 percent. The independently implemented adaptive bridge combines original and FRCRN-enhanced waveforms before frozen Whisper, rather than encoder representations. It improves aggregate WER slightly but increases CER. Fusion’s mean gate weight for the enhanced stream is 0.6785, with median 0.6930. Because cross-attention already exchanged information, this weight is not a causal contribution estimate. Different enhancers and training schedules prevent isolating the fusion mechanism from the bridge comparison. No significance test establishes reliability of the small margins.

**Sources:** `Thesis/chapters/results.tex`, `Thesis/chapters/work.tex`

### 12. Tiny-model ablations support using both views

Timing: 65 seconds.

| Whisper Tiny configuration | CV25 WER | AGFarsdat WER |
|---|---|---|
| Multi-condition baseline | 74.92% | 76.85% |
| Bidirectional cross-attention fusion | 57.82% | 63.50% |
| Gate only, without cross-attention | 62.35% | 65.88% |
| Original representation only¹ | 64.10% | 69.23% |
| Enhanced representation only¹ | 63.81% | 69.69% |

¹ Inference-time removal from the trained fusion checkpoint. Lower is better.
Separate Tiny experiment. Extra training and unequal capacity limit attribution.

**Speaker notes**

These are separate experiments with Whisper Tiny and Common Voice training only. They are not additional rows in the Whisper-small benchmark. Full bidirectional fusion performs best among these variants on both tests. Removing attention or using either stream alone raises WER. The single-view conditions bypass fusion at inference using the same trained checkpoint. They do not represent separately trained single-stream controls. Fusion also receives an extra joint ASR training stage, so the large gap from Tiny’s multi-condition baseline cannot be attributed to architecture alone. Full WER and CER results are in the appendix.

**Sources:** `Thesis/chapters/results.tex`

### 13. IranSeda audiobook preparation

Timing: 65 seconds.

Source preparation: 3,777 books, about 6,300 raw hours, 250 hours selected

Segmentation and transcription: Silero VAD, 20-second target, Persian-adapted Whisper Medium teacher

Text preparation: Persian normalization, conservative Qwen refinement, quality filtering

The new Whisper-small run retains ordinary, long and degraded speech.

**Speaker notes**

The second data direction asks whether additional real Persian speech helps after multi-condition training. The acquisition contains 3,777 audiobooks and approximately 6,300 raw hours. Compute limits motivate processing a selected 250-hour subset. Silero VAD segments audio around natural boundaries, with a twenty-second target, preferred fifteen to twenty-five seconds and a hard twenty-eight-second maximum. Persian-adapted Whisper Medium generates each segment’s transcript. Deterministic normalization precedes conservative Qwen refinement and quality filtering. Accepted clips supplement all original, long and degraded data, rather than replacing any baseline component.

**Sources:** `Thesis/chapters/work.tex`

### 14. 218.564 hours remain after filtering

Timing: 60 seconds.

| Reviewed clips | Accepted | Rejected |
|---|---|---|
| 44,820 | 39,454 | 5,366 |

Label safeguards: Retain raw text, preserve numbers, reject uncertain or excessive edits

Leakage control: Split by original book and preserve source identity, times and checksums

Accepted corpus: 218.564 hours from 136 files. Forty operational failures are separate.

**Speaker notes**

The selected 250 raw hours must be distinguished from the final accepted corpus. Of 44,820 reviewed clips, 39,454 were accepted and 5,366 rejected. Forty operational failures were tracked separately. Accepted audio totals 218.564 hours from 136 audiobook files. Qwen receives text and context only. It may correct orthography and clearly supported substitutions, but cannot add speech or complete segments. Numeric tokens must remain unchanged. Uncertain or excessive edits are rejected. Raw teacher text and refined text remain separate. Splits operate at book level, not clip level, and provenance preserves source identity, timestamps, hashes and processing settings. There is no human transcription audit that quantifies pseudo-label or LLM correction accuracy.

**Sources:** `Thesis/chapters/work.tex`, `Thesis/chapters/conclusions.tex`

### 15. Audiobook gains depend on the test domain

Timing: 85 seconds.

Change in WER versus multi-condition Whisper-small, in percentage points

| Category | With IranSeda − baseline |
|---|---|
| AGFarsdat | 0.86 |
| Common Voice | 0.35 |
| FLEURS | -6.04 |
| PersianSpeech¹ | -8.39 |
| Persian Corpus | -5.03 |

Negative means improvement. Aggregate WER: 17.83% to 17.22%.
¹ Only 24 test examples. Domain effects require further analysis.

**Speaker notes**

Adding IranSeda lowers aggregate WER from 17.83 to 17.22 percent and CER from 7.93 to 7.70 percent. That is a 0.61-point WER reduction when calculated from displayed values. Per-domain changes are more revealing: FLEURS improves by 6.04 points, PersianSpeech by 8.39 points and Persian Speech Corpus by 5.03 points. AGFarsdat worsens by 0.86 points and Common Voice by 0.35 points. PersianSpeech has only 24 examples, so its large change is preliminary. The data improves several general domains but does not add telephone robustness. The experiment does not isolate whether style, recording conditions, label quality or sampling mixture causes this pattern.

**Sources:** `Thesis/chapters/results.tex`

### 16. Results of the main Whisper-small comparisons

Timing: 60 seconds.

| Whisper-small system | Aggregate WER | Aggregate CER | AGFarsdat WER |
|---|---|---|---|
| Ordinary | 23.27% | 10.84% | 35.63% |
| Multi-condition | 17.83% | 7.93% | 25.80% |
| Dual-stream fusion | 17.64% | 7.85% | 25.37% |
| Multi-condition + IranSeda | 17.22% | 7.70% | 26.66% |

Lower is better. The lowest aggregate score and the lowest real-call score differ.

**Speaker notes**

This table brings the primary comparisons together. Ordinary Whisper-small provides the initial reference. Multi-condition training gives the large reduction in aggregate and real-call errors. Fusion adds a small numerical improvement. Additional IranSeda speech produces the lowest aggregate WER among these Whisper-small systems but a higher AGFarsdat WER than multi-condition or fusion. The ranking depends on the application domain. These are observed values from the completed experiments rather than statistically established model rankings. The complete augmentation and bridge comparisons are in backup slides.

**Sources:** `Thesis/chapters/results.tex`

### 17. Whisper Medium provides capacity context

Timing: 40 seconds.

| Model | Aggregate WER | Aggregate CER | AGFarsdat WER |
|---|---|---|---|
| Small, multi-condition | 17.83% | 7.93% | 25.80% |
| Small, + IranSeda | 17.22% | 7.70% | 26.66% |
| Medium, Persian-adapted | 14.36% | 6.68% | 21.99% |

Medium is also the pseudo-label teacher.
Changing model capacity prevents a data-only interpretation.

**Speaker notes**

Whisper Medium achieved the lowest overall WER and CER of the completed main evaluations. It also served as the pseudo-label teacher. Its aggregate WER is 3.47 percentage points below multi-condition Whisper-small. Its AGFarsdat WER is 21.99 percent. The single-stream models share the same 21,848 eligible examples, but model capacity and training details differ. These numbers give a useful higher-capacity reference and do not identify a data-only or architecture-only causal effect.

**Sources:** `Thesis/chapters/results.tex`, `Thesis/chapters/work.tex`

### 18. Research contributions

Timing: 40 seconds.

Reproducible telecom data: Aligned pairs with real codec paths and recorded augmentation settings

Controlled robustness evidence: Large Whisper-small gain on held-out real telephone and VoIP speech

Architecture and data studies: Fusion evaluation, Tiny ablations and a curated audiobook corpus

Domain-specific interpretation: Separate general-speech gains from real-call robustness

**Speaker notes**

The contribution is a reproducible experimental framework, alongside the implemented systems and data preparation. It includes deterministic telecom degradation with provenance, evidence of synthetic-to-real transfer, a task-trained dual-stream system evaluated against a strong multi-condition baseline and an independently implemented waveform bridge, plus a curated audiobook corpus. The domain-specific analysis prevents treating lower aggregate WER as universal robustness. Claims of a new universal architecture or state-of-the-art Persian ASR are outside the evidence.

**Sources:** `Thesis/chapters/introduction.tex`, `Thesis/chapters/work.tex`, `Thesis/chapters/conclusions.tex`

### 19. Limits and the next experiments

Timing: 45 seconds.

| Current limitation | Next experiment |
|---|---|
| Small margins, single runs | Repeated seeds and paired confidence intervals |
| Fusion training differs | Match training budget and ablate modules in Whisper-small |
| Pseudo-labels lack human audit | Audit teacher, Qwen edits and accepted text separately |
| Domain and deployment uncertainty | Sweep data mixture, report codec/SNR errors and runtime cost |

AGFarsdat is private. PersianSpeech has 24 examples. Simulator coverage is incomplete.

**Speaker notes**

The most urgent follow-ups test reliability and explain mechanism. Repeated seeds, exact paired eligibility and bootstrap confidence intervals are necessary for small differences. Matched-budget ablations should isolate attention and enhancement at Whisper-small scale. A human audit should compare raw teacher text, Qwen refinements and retained labels. Mixture sweeps and more telephone-like unlabelled speech would test the domain trade-off. Codec- and SNR-specific analyses and measurements of latency, memory and real-time factor remain necessary before deployment decisions. AGFarsdat is private, and PersianSpeech is very small.

**Sources:** `Thesis/chapters/results.tex`, `Thesis/chapters/conclusions.tex`

### 20. Telecom-matched training gives the largest gain

Timing: 45 seconds.

Real-call robustness: 35.63% to 25.80% WER with multi-condition training

Added architecture: Fusion reaches 25.37% real-call WER, a small numerical gain

Added audiobook speech: 17.22% aggregate WER, with a 0.86-point real-call regression

Questions

**Speaker notes**

The central answer is that matching the training conditions to the target channel yields the largest controlled improvement. Multi-condition training cuts real-call WER from 35.63 to 25.80 percent without changing the Whisper-small architecture. Fusion improves it slightly further, but statistical reliability and runtime value remain open. Audiobooks lower aggregate WER to 17.22 percent while slightly worsening AGFarsdat, showing why data relevance matters alongside volume. Thank you. I am happy to discuss the experimental controls and the domain-specific results.

**Sources:** `Thesis/chapters/results.tex`, `Thesis/chapters/conclusions.tex`

### Backup 21. Complete aggregate results

Timing: questions only.

| System | WER | CER |
|---|---|---|
| Ordinary | 23.27% | 10.84% |
| Noise only | 20.38% | 9.59% |
| pMCT | 19.21% | 8.91% |
| Multi-condition | 17.83% | 7.93% |
| Dual-stream fusion | 17.64% | 7.85% |
| Adaptive bridge | 17.74% | 8.00% |
| Whisper Medium | 14.36% | 6.68% |
| Multi-condition + IranSeda | 17.22% | 7.70% |

Lower is better. Confidence intervals and repeated-seed estimates are unavailable.

**Speaker notes**

Reported aggregate scores from the current results table. All methods except Medium use Whisper-small. Adaptive bridge is an independent implementation. Lower is better. Aggregate values pool error counts and reference units.

**Sources:** `Thesis/chapters/results.tex`

### Backup 22. WER by test domain

Timing: questions only.

| System | AG | CV25 | FLEURS | PS¹ | PSC |
|---|---|---|---|---|---|
| Ordinary | 35.63 | 11.52 | 21.05 | 34.53 | 30.17 |
| Noise only | 32.67 | 7.86 | 19.26 | 33.57 | 30.47 |
| pMCT | 30.01 | 7.25 | 20.23 | 31.65 | 30.50 |
| Multi-condition | 25.80 | 7.84 | 19.82 | 35.25 | 30.90 |
| Dual-stream fusion | 25.37 | 7.53 | 19.67 | 31.81 | 33.82 |
| Adaptive bridge | 25.84 | 7.63 | 19.73 | 35.09 | 32.32 |
| Whisper Medium | 21.99 | 3.84 | 17.14 | 31.65 | 31.17 |
| Multi-condition + IranSeda | 26.66 | 8.19 | 13.78 | 26.86 | 25.87 |

All values are %. AG: AGFarsdat. PS: PersianSpeech (24 examples). PSC: Persian Speech Corpus.

**Speaker notes**

All WER values are percentages. Lower is better. CV25 means Common Voice Persian 25, PS means PersianSpeech and PSC means Persian Speech Corpus. PersianSpeech has 24 examples. Cross-model eligibility counts vary slightly in PSC. These are the thesis table values, not simple averages of the aggregate scores.

**Sources:** `Thesis/chapters/results.tex`

### Backup 23. CER by test domain

Timing: questions only.

| System | AG | CV25 | FLEURS | PS¹ | PSC |
|---|---|---|---|---|---|
| Ordinary | 19.73 | 4.06 | 6.13 | 16.21 | 15.48 |
| Noise only | 17.49 | 3.17 | 5.72 | 14.64 | 15.40 |
| pMCT | 15.98 | 2.79 | 6.15 | 10.57 | 15.13 |
| Multi-condition | 12.62 | 3.22 | 6.36 | 15.62 | 15.22 |
| Dual-stream fusion | 12.39 | 3.13 | 6.29 | 12.95 | 19.95 |
| Adaptive bridge | 12.59 | 3.26 | 6.39 | 15.53 | 17.02 |
| Whisper Medium | 10.92 | 1.65 | 5.42 | 13.99 | 17.29 |
| Multi-condition + IranSeda | 12.96 | 3.29 | 4.09 | 9.82 | 14.16 |

All values are %. AG: AGFarsdat. PS: PersianSpeech (24 examples). PSC: Persian Speech Corpus.

**Speaker notes**

All CER values are percentages. Lower is better. CV25 means Common Voice Persian 25, PS means PersianSpeech and PSC means Persian Speech Corpus. PersianSpeech has 24 examples. Cross-model eligibility counts vary slightly in PSC. These are the thesis table values, not simple averages of the aggregate scores.

**Sources:** `Thesis/chapters/results.tex`

### Backup 24. Complete Whisper Tiny ablations

Timing: questions only.

| Tiny system | CV WER | CV CER | AG WER | AG CER |
|---|---|---|---|---|
| Multi-condition | 74.92 | 38.03 | 76.85 | 35.04 |
| Bidirectional fusion | 57.82 | 30.57 | 63.50 | 27.02 |
| Gate only | 62.35 | 32.87 | 65.88 | 28.42 |
| Residual cross-attention | 60.26 | 31.90 | 65.48 | 28.35 |
| Original only, inference | 64.10 | 33.21 | 69.23 | 30.39 |
| Enhanced only, inference | 63.81 | 33.05 | 69.69 | 30.73 |
| Bridge, PQ + RI | 74.40 | 37.81 | 76.81 | 34.94 |
| Bridge, PQ only | 74.48 | 37.81 | 76.63 | 34.83 |
| FRCRN enhanced only | 75.90 | 38.79 | 77.67 | 35.53 |

All values are %. Lower is better. This study has a different scale and training mix.

**Speaker notes**

Separate Common Voice-only Tiny study. Original-only and enhanced-only bypass fusion at inference using one final checkpoint. Residual cross-attention, gate-only and full attention have unequal parameter counts. Fusion gets an extra ASR training stage. Adaptive bridge uses frozen FRCRN and frozen Whisper, with either perceptual quality (PQ) or PQ plus recognition information (RI). This is an independent implementation, not an author-code reproduction.

**Sources:** `Thesis/chapters/results.tex`

### Backup 25. Fast Conformer corroborates a smaller gain

Timing: questions only.

| Test set | Ord. WER | MCT WER | Ord. CER | MCT CER |
|---|---|---|---|---|
| Aggregate | 22.73 | 21.81 | 14.79 | 14.45 |
| AGFarsdat | 31.07 | 30.48 | 14.81 | 14.09 |
| Common Voice | 11.68 | 10.01 | 16.46 | 16.10 |
| FLEURS | 29.22 | 29.40 | 9.94 | 10.22 |
| PersianSpeech | 30.70 | 31.41 | 13.50 | 13.56 |
| Persian Speech Corpus | 33.69 | 33.59 | 12.22 | 12.68 |

All values are %. Lower is better. Aggregate WER improves by 0.92 points.

**Speaker notes**

The within-model comparison is the appropriate control. Ordinary and multi-condition Fast Conformer use the same number of steps and learning settings. Their absolute scores cannot isolate architecture differences relative to Whisper because pretraining, decoder and optimization differ. CER values below are transcribed from the thesis table.

**Sources:** `Thesis/chapters/results.tex`

### Backup 26. External-model comparison on 100 samples

Timing: questions only.

| Model | Aggregate WER | Aggregate CER | AG WER |
|---|---|---|---|
| Chirp 3 | 16.44 | 5.14 | 22.94 |
| GPT-4o Transcribe | 18.55 | 6.94 | 48.62 |
| Small, + IranSeda | 20.72 | 10.72 | 26.61 |
| Whisper Medium | 23.13 | 11.24 | 24.77 |
| Voxtral Small 24B | 26.92 | 10.75 | 45.87 |
| Qwen3-ASR 1.7B | 36.66 | 15.68 | 63.30 |
| Grok STT 1.0 | 40.88 | 23.68 | 67.89 |
| Nemotron 3.5 ASR 0.6B | 98.70 | 80.86 | 104.59 |

All values are %. Lower is better. Twenty examples per domain. Separate benchmark.

**Speaker notes**

This is a completed thesis experiment on 20 randomly selected samples from each of the five datasets, with common normalization. These scores apply only to this fixed 100-sample subset and are not directly comparable to full-test results. Chirp 3 leads this subset. The evidence is supplementary and does not support a global model ranking. Names and scores reproduce the archived experiment rather than current product specifications.

**Sources:** `Thesis/chapters/results.tex`

### Backup 27. Controls and attribution boundaries

Timing: questions only.

| Comparison | Main controls | Remaining boundary |
|---|---|---|
| Ordinary vs MCT | Small model, LR, step budget, tests | Composite telecom treatment |
| MCT vs + IranSeda | Small model, full base mix, steps, tests | Data domain and labels both change |
| MCT vs fusion | Starting checkpoint and test domains | Extra modules and ASR training |
| Fusion vs bridge | Test domains, training data family | Enhancer and freezing differ |
| Small vs Medium | Common single-stream eligible tests | Capacity and training differ |

Small runs: learning rate 10⁻⁵, batch 4, gradient accumulation 8.

**Speaker notes**

Ordinary, multi-condition and multi-condition plus IranSeda use Whisper-small with the reported same main training settings and step budget. Fusion starts from multi-condition and receives an additional staged training schedule. Bridge freezes both FRCRN and Whisper, so its comparison with fusion is a system-level comparison. Medium changes capacity. Human-labelled tests and common normalization remain fixed. Exact per-model sample eligibility should be reconciled for small cross-system differences.

**Sources:** `Thesis/chapters/work.tex`, `Thesis/chapters/results.tex`

### Backup 28. Telecom simulator architecture

Timing: questions only.

Original thesis figure: figs/degradation-pipeline-architecture.png

**Speaker notes**

Complete source diagram. Native Opus applies packet-level loss and PLC. Other codec paths use real FFmpeg round-trips and optionally approximate network loss through waveform dropout. The aligned clean target has matched bandwidth.

**Sources:** `Thesis/chapters/work.tex`, `Thesis/figs/degradation-pipeline-architecture.png`

### Backup 29. Fusion architecture and training

Timing: questions only.

Original thesis figure: figs/fusion-model-architecture-left-to-right.png

**Speaker notes**

Complete source diagram. Residual U-Net, shared encoder with two passes, bidirectional attention and a time-feature gate. The lower row gives the three training stages.

**Sources:** `Thesis/chapters/work.tex`, `Thesis/figs/fusion-model-architecture-left-to-right.png`

### Backup 30. Long-audio construction pipeline

Timing: questions only.

Original thesis figure: figs/long-audio-data-creation-pipeline.png

**Speaker notes**

Complete source diagram. Source validation, VAD segmentation, teacher transcription, normalization, constrained Qwen refinement and filtering. Accepted speech adds to the complete baseline mixture.

**Sources:** `Thesis/chapters/work.tex`, `Thesis/figs/long-audio-data-creation-pipeline.png`

## Likely defense questions

1. **How do you know synthetic training transfers to calls?** AGFarsdat is real telecom audio, held out from training and model selection. WER falls from 35.63% to 25.80%.
2. **Does the gate prove enhancement helps?** No. Cross-attention mixes the streams before gating. Use ablations, and distinguish Tiny results from Whisper-small.
3. **Why compare fusion to multi-condition Whisper?** It is the relevant stronger reference after matching training to the channel. Ordinary Whisper alone would overstate architectural benefit.
4. **Why does IranSeda help general speech but worsen calls?** The observed effect is domain-dependent. Book style, channel, label quality and mixture proportions are plausible factors, not established causes.
5. **Does Qwen make labels correct?** No human audit quantifies that. It performs constrained text correction, while raw labels and rejected edits remain traceable.
6. **Why not select Medium as the only final answer?** It gives the lowest main aggregate error, but changes model capacity and does not answer the controlled data and architecture questions.
7. **Is fusion worth deploying?** Current results lack runtime, memory and latency measurements, and its small gain lacks confidence intervals.
8. **Does the 100-sample comparison establish model superiority?** It is a supplementary fixed-subset experiment with only 20 examples per domain. It cannot replace the full evaluation.

## Source assets

The original thesis figures are unchanged and appear once each on backup slides 28–30. All main tables, charts and text are editable. Existing presentation assets remain available.
