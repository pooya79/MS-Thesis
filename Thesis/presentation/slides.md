# Robust Speech Recognition in Persian Language

## Presentation plan

Updated: 2026-10-13. This file is the text plan for the next PowerPoint revision. The current redesigned PPTX has not yet been rebuilt to match it.

**17 slides total, including the title and conclusion. No hidden table appendix.** English, 16:9, for a 20-minute defense. Suggested speaking time: **19:05**, leaving 55 seconds for transitions.

The flow follows the implemented methods and their evidence: problem and experimental setup, telecom augmentation, enhancement and fusion with the bridge comparator, audiobook data, then interpretation and conclusion.

The proposed generated clean/degraded example slide is removed. The bridge explanation comes immediately before the fusion/bridge results, followed by the ablation study. This introduces both systems before comparing them.

## Content and visual decisions

- Put the existing thesis figures in the main talk. Preserve original source files and enlarge relevant sections when a complete diagram would make labels too small.
- Explain fusion architecture and training on separate slides using the upper architecture and lower training sections of the same thesis figure.
- Use the original architecture figure from the cited bridge paper, with visible attribution and a paper link in the eventual speaker notes.
- Use charts for augmentation, fusion/bridge, ablation and audiobook-domain results. The data lists and tables in this Markdown file are authoring references, not instructions to display more tables.
- Keep at most one compact results table, on the synthesis slide; two aligned plots are also acceptable.
- Keep detailed corpus inventories, full WER/CER matrices, complete ablations, Fast Conformer details and the external-model subset benchmark in the thesis. Do not append them as presentation slides.
- Preserve the official thesis title and university identity. Do not invent a defense date.
- State important qualifications near the result they concern; use the limitations slide for the overall discussion.

## Source precedence and interpretation

Use the current Thesis/chapters/work.tex, Thesis/chapters/results.tex, Thesis/chapters/conclusions.tex and Thesis/front/abstract-eng.tex as the sources of truth. Older decks and notes do not override the current thesis.

- The ordinary baseline already includes constructed long speech. Multi-condition training adds degraded speech.
- Fusion and IranSeda are separate branches from the multi-condition reference. There is no evaluated fusion-plus-IranSeda system.
- Current fusion aggregate WER/CER is **17.64% / 7.85%**. Enhanced-stream gate mean: **0.6785**; median: **0.6930**. Earlier failed-fusion/collapsed-gate claims are obsolete.
- IranSeda is a completed experiment: **17.22% WER / 7.70% CER**. **250 hours** is the selected raw subset; **218.564 hours** is the accepted corpus. The approximately 6,300 acquired hours did not all enter training.
- Calculate differences from displayed endpoints: IranSeda improves aggregate WER by **0.61 percentage points**, and worsens AGFarsdat CER by **0.34 points** and Common Voice CER by **0.07 points**.
- The bridge is an independent implementation of the cited approach trained and evaluated for this thesis, not a verified author-code reproduction. Its Persian results are separate from the paper's CHiME-4 results.
- Small fusion/bridge differences are numerical observations. Repeated-seed estimates and confidence intervals are unavailable.
- Paired single-stream comparisons use **21,848** common eligible examples. Persian Speech Corpus counts vary across systems; do not assert exact pairing for every cross-system comparison without reconciling eligibility.
- Tiny ablations are a separate Common Voice-only training study. Original-only and enhanced-only conditions bypass fusion at inference from the trained checkpoint; they are not separately trained single-stream controls.
- Qwen receives text and context, not audio. Conservative context-supported substitutions are permitted alongside orthographic correction. No quantitative human transcription audit establishes label accuracy.
- Aggregate WER/CER pool errors and reference units, rather than averaging dataset percentages. Lower is better; report absolute differences in percentage points.

## Slide-by-slide plan

### 1. Robust Speech Recognition in Persian Language

**Time:** 0:20.

**Purpose:** Introduce the thesis and target application.

**On the slide:** Official title, Pouya Fallah Sichani, supervisor Dr. Hosein Sameti, Sharif University of Technology and degree. Keep the cover minimal.

**Visual:** Original university branding from Thesis/front/template/images/logo.pdf.

**What to say:**

> This thesis investigates how to improve Persian speech recognition when audio comes through telephone and VoIP channels. I study telecom-oriented training data, enhancement and fusion, and additional pseudo-labelled Persian speech.

**Transition:** Explain the mismatch between training speech and calls.

**Sources:** Thesis/front/info.tex; Thesis/front/abstract-eng.tex.

### 2. The mismatch between training speech and real calls

**Time:** 0:50.

**Purpose:** Establish the problem and motivate the research.

**On the slide:** General Persian training speech versus telephone/VoIP conditions: noise, restricted bandwidth, codec distortion and packet loss. Mention the difficulty of obtaining enough transcribed calls.

**Visual:** A simple comparison of available and target conditions. Avoid a generic ASR history.

**What to say:**

> Persian corpora cover different speaking and recording styles, but they do not fully represent telecom channels. Calls can combine several distortions at once. Transcribed real calls are expensive to collect and privacy-sensitive, so I investigate how existing labelled speech and unlabelled audio can help. Success must be tested on real calls outside the simulator.

**Transition:** Introduce the experiments.

**Sources:** Thesis/chapters/introduction.tex; Thesis/chapters/preliminaries.tex.

### 3. Research approach and comparison structure

**Time:** 0:55.

**Purpose:** Establish the baselines and separate the experimental branches.

**On the slide:** Ordinary Whisper-small becomes multi-condition Whisper-small through telecom augmentation. From that reference, one branch adds enhancement/fusion and another adds IranSeda. Identify the adaptive bridge as a fusion comparator.

**Visual:** A compact branching overview with the shared reference clearly identified.

**What to say:**

> The multi-condition model becomes the reference for the later experiments. Fusion changes the system and training schedule. IranSeda adds data while retaining the full baseline mixture. These are separate branches; I did not evaluate fusion trained with IranSeda. A waveform bridge provides another comparison for fusion.

**Transition:** Establish the shared data and evaluation foundation.

**Sources:** Thesis/chapters/work.tex.

### 4. Training data and evaluation

**Time:** 1:05.

**Purpose:** Provide enough context to assess the comparisons.

**On the slide:**

- Training: Common Voice Persian 25, limited FarsSpon, FLEURS Persian and constructed long examples.
- Testing: AGFarsdat for real calls, plus Common Voice, FLEURS, PersianSpeech and Persian Speech Corpus.
- Shared preprocessing, Persian normalization and greedy decoding; define WER once.

**Visual:** Separate training and evaluation groups. Highlight AGFarsdat's **10,044 real telephone/VoIP examples**. Avoid an inventory table.

**What to say:**

> The base mixture covers read and spontaneous Persian speech. Long examples join two to four clips within a split, with durations above five and at most twenty seconds. They already belong to the ordinary baseline. WER counts substitutions, deletions and insertions per reference word; CER counts characters. Aggregate scores pool counts. AGFarsdat is held out from training and model selection.

**Qualification:** PersianSpeech has only 24 examples; eligibility varies slightly across some systems. Audio processing uses mono 16 kHz.

**Transition:** Explain how telecom-degraded speech is generated.

**Sources:** Thesis/chapters/work.tex; Thesis/chapters/results.tex.

### 5. Reproducible telecom augmentation

**Time:** 1:30.

**Purpose:** Explain the simulator as an implemented research contribution.

**On the slide:** Preparation, channel conditioning, codec/network processing and aligned output pairs.

**Visual:** Make the existing [thesis degradation figure](../figs/degradation-pipeline-architecture.png) the main content. Use the thesis original rather than an older copied presentation asset. Reveal or enlarge sections if needed.

**What to say:**

> The pipeline samples telephone and VoIP profiles, adds noise and gain variation, applies bandwidth restrictions, and processes real codec paths. Native Opus handles packet loss and decoder concealment. Other codecs use FFmpeg encode–decode processing and optional waveform-dropout approximations. The outputs remain aligned, and the clean enhancement target has matching bandwidth. Seeds and metadata make each pair traceable.

**Narration details:** DEMAND noise, SNR −5 to 15 dB, gain ±6 dB; G.711, GSM, AMR-NB, AMR-WB and Opus. Profile weights describe a designed distribution, not operator measurements.

**Transition:** Move directly to results. There is no separate generated-pair demonstration slide.

**Sources:** Thesis/chapters/work.tex; Thesis/figs/degradation-pipeline-architecture.png.

### 6. Telecom augmentation transfers to real calls

**Time:** 1:25.

**Purpose:** Present the strongest robustness finding and compare augmentation approaches.

**On the slide:** A real-call WER chart for four approaches. The aggregate improvement is a smaller supporting result.

**Chart data — AGFarsdat WER:**

- Ordinary Whisper-small: **35.63%**.
- Noise only: **32.67%**.
- pMCT: **30.01%**.
- Full telecom multi-condition: **25.80%**.

**Supporting result:** Aggregate WER **23.27% to 17.83%**. Real-call improvement: **9.83 percentage points**, approximately **27.6% relative**.

**What to say:**

> Noise-only augmentation and pMCT improve over ordinary training. The complete telecom pipeline gives the lowest real-call WER among these approaches. The reduction from 35.63 to 25.80 percent is measured on held-out real calls, providing evidence of synthetic-to-real transfer. It supports modelling more than noise, although individual codec and network effects were not separately ablated.

**Transition:** Ask whether enhancement and fusion can improve the stronger reference further.

**Sources:** Thesis/chapters/results.tex.

### 7. Dual-stream enhancement and fusion

**Time:** 1:25.

**Purpose:** Explain the architecture and why both views are retained.

**On the slide:** Original log-Mel input, residual U-Net enhanced view, shared encoder passes, bidirectional cross-attention, time-feature gate and decoder.

**Visual:** Enlarge the upper architecture section of the [thesis fusion figure](../figs/fusion-model-architecture-left-to-right.png). Preserve the original; its lower training strip belongs to slide 8.

**What to say:**

> An enhanced view may reduce corruption while losing useful information. I therefore retain the original view. A residual U-Net modifies 80-band log-Mel features. Both views pass through the same Whisper encoder with shared weights. Bidirectional attention exchanges information, and a gate combines the representations before decoding. The system starts from the multi-condition checkpoint.

**Narration details:** One cross-attention layer, four heads; this enhancer does not reconstruct a waveform.

**Transition:** Explain how the components learn to work with Whisper.

**Sources:** Thesis/chapters/work.tex; Thesis/figs/fusion-model-architecture-left-to-right.png.

### 8. Three-stage fusion training

**Time:** 1:05.

**Purpose:** Explain the training progression and its interpretation boundary.

**On the slide:** Enhancer warm-up, enhancer/fusion training with frozen Whisper, then joint fine-tuning.

**Visual:** Enlarge the lower training strip from the preceding thesis figure. Keep loss coefficients secondary to which components train.

**Authoring details:**

- Stage 1: enhancer only; 0.5 L_enh + 0.5 L_feat; Whisper frozen.
- Stage 2: enhancer and fusion; L_ASR + 0.3 L_enh; Whisper frozen.
- Stage 3: all components; L_ASR + 0.1 L_enh.
- One epoch per stage. Final learning rates: added modules 1e-4, Whisper 5e-6.

**What to say:**

> Training gradually connects enhancement to recognition. First the enhancer learns from reconstruction and feature matching. Then enhancer and fusion learn against frozen Whisper. Finally the whole model trains, with a lower learning rate for Whisper. The final stage includes degraded and long examples plus a random 25-percent subset of clean data. Fusion changes architecture and adds recognizer training.

**Transition:** Introduce the bridge before comparing the systems' results.

**Sources:** Thesis/chapters/work.tex; Thesis/figs/fusion-model-architecture-left-to-right.png.

### 9. Adaptive bridge: combining speech before recognition

**Time:** 1:25.

**Purpose:** Explain the prior-work comparator before its scores appear.

**On the slide:** Original Figure 1 from Cui et al., ICASSP 2025. Emphasize panel (a)'s signal flow and panel (b)'s coefficient prediction. Keep panel (c)'s internal layers secondary. Visible attribution: **Cui et al., ICASSP 2025, Fig. 1 (CC BY 4.0)**.

**Visual:** [Downloaded bridge architecture](assets/cui2025-bridge-figure1.png). Original source, version, attribution and checksum are recorded in [figure provenance](assets/cui2025-bridge-figure1.source.md).

**What to say:**

> The bridge combines original and enhanced waveforms before recognition. In my independent implementation, FRCRN and Persian multi-condition Whisper remain frozen; only the bridge trains. It predicts one coefficient per utterance from both views, using perceptual-quality targets and precomputed recognition information during training. Our fusion system combines encoded representations and later fine-tunes Whisper jointly.

**Qualification:** The figure illustrates the authors' method. The next slide reports this thesis's independent Persian implementation, not the paper's CHiME-4 scores. Different enhancers and training prevent isolating the effect of fusion location.

**Transition:** Compare the previous multi-condition baseline, bridge and proposed fusion together.

**Sources:** Thesis/chapters/work.tex, adaptive bridge section; Thesis/bibs/refs.bib, cui2025realbridge; [Cui et al. paper](https://arxiv.org/abs/2501.02452), [versioned Figure 1](https://arxiv.org/html/2501.02452v1#S1.F1).

### 10. What fusion and the bridge add

**Time:** 1:25.

**Purpose:** Compare both systems against the appropriate previous baseline.

**On the slide:** Two small plots: aggregate WER and AGFarsdat WER, with matching system order and colors. Put aggregate CER in a short secondary annotation.

**Authoring data:**

| System | Aggregate WER | Aggregate CER | AGFarsdat WER |
|---|---:|---:|---:|
| Multi-condition Whisper-small | 17.83% | 7.93% | 25.80% |
| Adaptive bridge, independent implementation | 17.74% | 8.00% | 25.84% |
| Proposed dual-stream fusion | 17.64% | 7.85% | 25.37% |

**What to say:**

> Fusion lowers aggregate WER by 0.19 points and real-call WER by 0.43 points relative to multi-condition training. The bridge slightly improves aggregate WER but increases CER, with real-call WER close to baseline. Fusion has the lowest displayed WER here, but the margins lack confidence intervals. The comparison evaluates complete systems with different enhancers and training schedules.

**Optional spoken detail:** Gate mean **0.6785**, median **0.6930**. These do not measure independent causal contribution because attention has already mixed the streams. Do not add a gate-statistics slide.

**Transition:** Examine the fusion components through ablations.

**Sources:** Thesis/chapters/results.tex; Thesis/chapters/work.tex.

### 11. Ablations: using both views and attention

**Time:** 1:15.

**Purpose:** Explain what the component study supports.

**On the slide:** Label **Separate Whisper Tiny experiment**. Show a compact AGFarsdat WER chart and mention the corresponding pattern on Common Voice.

**Chart data — AGFarsdat WER:**

- Full bidirectional fusion: **63.50%**.
- Gate only, without cross-attention: **65.88%**.
- Original representation only, inference: **69.23%**.
- Enhanced representation only, inference: **69.69%**.

**Supporting Common Voice WER, same order:** **57.82%, 62.35%, 64.10%, 63.81%**.

**What to say:**

> This is a separate Tiny study using Common Voice training. Full bidirectional fusion performs best among these configurations on both tests. Removing attention or bypassing fusion with either representation raises WER. The single-view cases use the trained fusion checkpoint at inference. Unequal parameter counts and extra training prevent assigning all improvement to one module.

**Transition:** Move from added architecture to added Persian data.

**Sources:** Thesis/chapters/results.tex, Tiny ablation table; experiment source report/cv25-tiny-official/final-tests/summary.json.

### 12. Turning IranSeda audiobooks into training data

**Time:** 1:25.

**Purpose:** Explain the additional-data pipeline as an implemented contribution.

**On the slide:** Source validation, VAD segmentation, teacher transcription, normalization, Qwen refinement, filtering and training.

**Visual:** Give the [thesis long-audio pipeline figure](../figs/long-audio-data-creation-pipeline.png) most of the slide; explain it by phases.

**What to say:**

> The acquisition contains 3,777 audiobooks and approximately 6,300 raw hours. Compute limits led to selecting 250 hours for processing. Silero VAD segments the audio, and Persian-adapted Whisper Medium supplies transcripts. Normalization, constrained Qwen refinement and filtering follow. Accepted clips supplement all original, long and degraded speech. The new Small run retains the main training settings and step budget.

**Narration details:** Target 20-second segments, preferred 15–25 seconds, hard maximum 28 seconds. Medium also provides a higher-capacity evaluation reference.

**Transition:** Explain the resulting corpus and safeguards.

**Sources:** Thesis/chapters/work.tex; Thesis/figs/long-audio-data-creation-pipeline.png.

### 13. Accepted data and label safeguards

**Time:** 1:00.

**Purpose:** Explain what enters training and the limits of automatic refinement.

**On the slide:** **39,454 accepted clips**, **218.564 hours**; retain raw text, preserve numbers, reject uncertainty/excessive edits, split by original book.

**Visual:** Choose one genuine saved raw/refined or rejected transcript pair with provenance. Do not invent a correction. If no suitable record is available, use a short excerpt of the refinement rules already shown in the thesis. A new example is not a prerequisite for the deck.

**What to say:**

> Of 44,820 reviewed clips, 39,454 were accepted and 5,366 rejected. Accepted audio totals 218.564 hours from 136 files; forty operational failures are separate. Qwen receives text and context but no audio. Numeric tokens must remain unchanged, and uncertain or excessive edits are rejected. Book-level splitting reduces leakage. These safeguards do not replace a human accuracy audit.

**Transition:** Test whether the added speech improves the target domain.

**Sources:** Thesis/chapters/work.tex, refinement prompt and filtering sections; Thesis/chapters/conclusions.tex.

### 14. Audiobook gains depend on the domain

**Time:** 1:25.

**Purpose:** Show the trade-off behind the improved aggregate score.

**On the slide:** WER-change chart with a clear zero line. Define change as **with IranSeda minus multi-condition baseline**, in percentage points; negative means improvement.

**Chart data:**

- AGFarsdat: **+0.86**.
- Common Voice: **+0.35**.
- FLEURS: **−6.04**.
- PersianSpeech: **−8.39**; mark **24 examples**.
- Persian Speech Corpus: **−5.03**.

**Supporting result:** Aggregate WER **17.83% to 17.22%**, CER **7.93% to 7.70%**.

**What to say:**

> Audiobook data improves aggregate performance, but the effect differs by domain. FLEURS and Persian Speech Corpus improve substantially. Real-call WER worsens by 0.86 points, and Common Voice also worsens slightly. The large PersianSpeech change is preliminary because it has only 24 examples. This experiment does not isolate whether style, channel, labels or sampling proportions cause the pattern.

**Transition:** Bring the main findings together.

**Sources:** Thesis/chapters/results.tex.

### 15. What the main experiments establish

**Time:** 1:05.

**Purpose:** Synthesize the results without another complete inventory.

**On the slide:** Four main Small systems and two primary metrics, using one compact table or two aligned plots.

**Authoring data:**

| Whisper-small system | Aggregate WER | AGFarsdat WER |
|---|---:|---:|
| Ordinary | 23.27% | 35.63% |
| Multi-condition | 17.83% | 25.80% |
| Proposed fusion | 17.64% | 25.37% |
| Multi-condition + IranSeda | 17.22% | 26.66% |

**What to say:**

> Multi-condition training provides the largest controlled robustness improvement. Fusion adds a small numerical gain. IranSeda gives the lowest aggregate Small-model WER, while fusion gives the lowest real-call WER. The intended domain matters. Persian-adapted Medium reaches 14.36 percent aggregate WER and 21.99 percent real-call WER, but also changes model capacity and training.

**Presentation choice:** The bridge was explained and compared on slides 9–10; do not repeat it here. Medium gets a contextual mention rather than another slide.

**Transition:** State the unresolved questions and next experiments.

**Sources:** Thesis/chapters/results.tex; Thesis/chapters/conclusions.tex.

### 16. Limitations and next experiments

**Time:** 0:50.

**Purpose:** Identify the most consequential remaining uncertainty.

**On the slide:**

- Reliability: repeated seeds, exact pairing and confidence intervals.
- Fusion attribution and value: matched-budget Small ablations and inference-cost measurements.
- Data quality and relevance: human pseudo-label audit and telephone-oriented mixture experiments.

**Visual:** Three short priorities, without a large limitation/solution table.

**What to say:**

> Small differences need repeated runs and paired confidence intervals. Fusion needs a single-stream control with comparable additional training, Small-scale ablations, and runtime and memory measurements. The audiobook pipeline needs a human audit of teacher text and Qwen changes. Mixture experiments could explain the domain trade-off. Simulator coverage and the private real-call test also limit the evidence.

**Transition:** Return to the research question.

**Sources:** Thesis/chapters/results.tex; Thesis/chapters/conclusions.tex.

### 17. Conclusion and questions

**Time:** 0:40.

**Purpose:** End with the answer and leave the essential findings visible.

**On the slide:**

- Telecom-matched training: real-call WER **35.63% to 25.80%**.
- Fusion: **25.37%** real-call WER, a small numerical gain.
- Audiobook data: **17.22%** aggregate WER, with a domain trade-off.

**Visual:** Restrained conclusion using the essential numbers. Include “Questions”; do not add another closing slide.

**What to say:**

> Telecom-matched training substantially improves Persian recognition on real calls. Fusion provides a smaller numerical gain. Additional audiobook speech improves aggregate performance, but its usefulness depends on the target domain. Together, these experiments provide reproducible systems and evidence for adapting Persian ASR when transcribed telecom speech is limited. Thank you.

**Sources:** Thesis/chapters/results.tex; Thesis/chapters/conclusions.tex.

## Figure inventory for PowerPoint authoring

| Slide | Asset | Use |
|---|---|---|
| 1 | Thesis/front/template/images/logo.pdf | Original university branding |
| 5 | Thesis/figs/degradation-pipeline-architecture.png | Telecom simulator explanation |
| 7 | Thesis/figs/fusion-model-architecture-left-to-right.png | Enlarged upper architecture section |
| 8 | Same fusion figure | Enlarged lower training strip |
| 9 | Thesis/presentation/assets/cui2025-bridge-figure1.png | Original paper figure, visibly attributed |
| 12 | Thesis/figs/long-audio-data-creation-pipeline.png | Audiobook method explanation |
| 13 | Traceable transcript record or thesis prompt excerpt | Label safeguards |

## What stays in the thesis

Full corpus counts, complete aggregate and per-domain WER/CER matrices, the full Tiny ablation/bridge variants, Fast Conformer tables, the separate 100-sample external-model benchmark, and detailed optimization settings remain available in the thesis and research archive. They do not become hidden slides. This plan ends at slide 17.
