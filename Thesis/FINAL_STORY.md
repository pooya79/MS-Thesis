# Final Thesis Story

## Working title

**A Data-Centric Approach to Robust Persian Speech Recognition: From
Multi-Condition Training and Enhancement Fusion to Pseudo-Labelled Long
Audio**

## Central thesis

This thesis argues that, under the data and computational constraints of
Persian automatic speech recognition (ASR), increasing the amount and acoustic
diversity of training data is more effective than adding a complex speech
enhancement and feature-fusion architecture. The argument is built through a
sequence of controlled experiments rather than assumed in advance. A
reproducible degradation pipeline and multi-condition training substantially
improved Whisper-small, whereas the proposed enhancer–fusion model did not
surpass that stronger data-trained baseline. This negative result motivated the
final, data-centric stage of the thesis: a completed pseudo-labelled Persian
speech dataset from long-form IranSeda audio, followed by the pending test of
whether the same Whisper-small architecture improves when trained with more
real Persian speech.

## The story in one page

Persian ASR is limited not only by model architecture but also by the amount,
diversity, and acoustic coverage of available labelled speech. This problem is
especially visible in telephone and VoIP audio, where noise, gain variation,
bandwidth limitation, codec distortion, and packet loss create a mismatch
between relatively clean training speech and real deployment conditions. The
research began by gathering approximately 1,000 hours of open-source and
proprietary Persian speech and converting the sources to a common,
traceable training format. Because many public corpora contain short clips,
longer training examples were also constructed to expose the recognizer to
longer temporal contexts.

The first proposed solution was to improve robustness without requiring a
large corpus of manually transcribed telephone calls. A deterministic speech
degradation pipeline created aligned clean–degraded pairs using environmental
noise, gain changes, narrowband and wideband filtering, real FFmpeg codec
round-trips, and packet-loss simulation. Random seeds and augmentation metadata
were saved for every generated example. Whisper-small was then fine-tuned in
two controlled settings: once with the ordinary Persian training mixture and
once with ordinary, long-form, and synthetically degraded data. On the shared
evaluation set, adding degraded data reduced aggregate WER from **23.27% to
17.83%** and CER from **10.84% to 7.93%**. On the real telephone/VoIP
AGFarsdat set, WER fell from **35.63% to 25.80%**. These results established a
strong data-based baseline and showed that simulated channel variability could
transfer to real telecommunication speech.

The next question was whether a more elaborate architecture could improve on
this baseline. A learnable enhancer was placed before the Whisper encoder. It
produced an enhanced log-Mel view while preserving the original degraded view.
Both views passed through a shared Whisper encoder, after which a fusion module
used bidirectional cross-attention and a learned time–feature gate to combine
their encoder representations. The model was trained in stages: enhancer
warm-up, enhancer-and-fusion training with Whisper frozen, and joint
end-to-end training. The final fusion model was competitive and improved over
Whisper trained only on ordinary data, but it did not beat the multi-condition
Whisper-small baseline. Its aggregate WER was **18.44%**, compared with
**17.83%** for multi-condition Whisper-small, and its aggregate CER was
**14.04%**, compared with **7.93%**. The learned gate assigned an average
weight of only **0.0039** to the enhanced stream and **0.9961** to the original
stream. Therefore, the proposed fusion architecture failed in its main goal:
it did not provide stable complementary information beyond what Whisper had
already learned directly from diverse data.

This failure is a central result, not a discarded experiment. It suggests that
the main bottleneck in the studied setting is better addressed through data
than through additional architectural complexity. The final stage of the
thesis therefore follows the evidence and asks a broader question: if exposing
Whisper-small to more varied training data produced the strongest controlled
small-model result, can its performance be improved further with substantially
more real Persian speech data?

To answer this question, a higher-capacity Whisper model was adapted to Persian
and used as a teacher. Whisper Medium was selected instead of a larger Whisper
variant because of the available computational resources. Its completed
evaluation on the same 21,848 eligible examples as the single-stream
Whisper-small models produced **14.36% WER** and **6.68% CER**, the strongest
completed aggregate result. Relative to multi-condition Whisper-small, these
scores are lower by **3.47 WER percentage points** and **1.25 CER percentage
points**. This is not a data-only controlled comparison because model capacity
also changed.

In parallel, **3,777 audiobooks** were collected from the public IranSeda
catalogue, representing approximately **6,300 raw hours**. A **250-hour subset**
was selected for the final experiment in accordance with the available thesis
schedule and compute budget.

The selected long recordings were converted into training examples through a
reproducible pipeline. Silero VAD detected speech regions, and clips were cut at
natural boundaries with a target duration of 20 seconds and a preferred range
of 15–25 seconds. The upper bound leaves a safety margin below Whisper's
30-second input window. Each clip was independently transcribed by the
Persian-adapted Whisper Medium teacher. The raw hypothesis was retained, and a
deterministic Persian normalizer standardized characters, whitespace, and the
project's transcription conventions. A constrained LLM refinement stage then
performed orthographic cleanup only using the configured `Qwen/Qwen3.6-27B`
model, under rules prohibiting the addition, removal, inference, reordering,
summarization, or paraphrasing of spoken content. Quality filtering and
source-level preparation were also completed. All derived clips retain their
IranSeda source identity, timestamps, checksums, processing settings, teacher
identity, raw transcript, normalized transcript, and refined text.

The accepted IranSeda clips are ready to be added to the complete training
mixture used by the strongest Whisper-small baseline. The final model retains the
ordinary Persian speech, constructed long examples, and synthetically degraded
examples, and adds the selected pseudo-labelled IranSeda speech. It is not
trained on IranSeda plus ordinary data alone. The prepared splits are made at
the original audiobook level so that clips from one source cannot appear in
multiple splits. The final evaluation will reuse the same human-labelled test
sets, Persian normalization, WER/CER implementation, and comparison rules used
for the earlier Whisper-small baselines. This gives the thesis a controlled
final comparison: multi-condition Whisper-small trained with ordinary, long,
and degraded data versus the same architecture trained with that complete
mixture plus the selected IranSeda data.

The final numerical outcome of this experiment is not yet known and must not
be predicted. Its result will complete the thesis argument in either direction.
An improvement would support the conclusion that scalable, carefully filtered
pseudo-labelling is an effective response to Persian ASR data scarcity. A null
or negative result would show that raw data volume is insufficient without
better pseudo-label quality, domain balance, or training-mixture control. In
both cases, the scientific contribution is the controlled progression from
data augmentation, through an unsuccessful architectural hypothesis, to a
reproducible test of real-data scaling.

## Research questions

1. Does adding synthetically degraded speech to training improve
   Whisper-small over fine-tuning on ordinary Persian speech alone?
2. Does the improvement from synthetic multi-condition training transfer to
   real telephone and VoIP speech without unacceptable regression on general
   Persian speech?
3. Can an enhancer before the Whisper encoder and a fusion module after the
   encoder outperform a strong multi-condition Whisper-small baseline?
4. Does the learned fusion gate make meaningful use of both original and
   enhanced representations?
5. Does adding the selected pseudo-labelled IranSeda speech to the existing
   ordinary, long, and degraded training mixture improve Whisper-small on the
   same fixed, human-labelled evaluation sets?
6. How do any gains or regressions from the added IranSeda data vary between
   general Persian speech and real telephone/VoIP speech?

The first four questions are answered by the existing experiments. Questions
five and six are answered only after the final Whisper-small training and
evaluation run.

## Contributions

1. **A unified Persian ASR data foundation.** Approximately 1,000 hours of
   open-source and proprietary speech were standardized, normalized, and made
   usable through a common manifest and dataset interface.
2. **A reproducible speech-degradation pipeline.** The pipeline generates
   aligned clean–degraded pairs and records seeds, noise, gain, bandwidth,
   codec, packet-loss, alignment, and normalization metadata.
3. **Controlled evidence for multi-condition training.** Training
   Whisper-small with degraded examples produced the strongest controlled
   small-model baseline and improved performance on real telephone/VoIP speech.
4. **An honestly evaluated enhancer–fusion architecture.** The work includes
   the pre-encoder enhancer, shared encoder, post-encoder fusion module, and
   multistage training procedure, together with the negative result and gate
   collapse that prevented it from beating the data-based baseline.
5. **A completed reproducible long-audio dataset-construction pipeline.** The
   final phase covers IranSeda acquisition, Silero-VAD segmentation, Whisper-Medium
   pseudo-labelling, deterministic Persian normalization, constrained LLM
   refinement, quality filtering, provenance, and leakage-safe publication.
6. **A completed higher-capacity teacher evaluation and a pending controlled
   data-scaling experiment.** Whisper Medium achieved 14.36% WER and 6.68% CER;
   Whisper-small will be retrained
   with ordinary, long, degraded, and selected IranSeda data, then compared
   with the existing multi-condition baseline on unchanged human-labelled test
   sets.

## Evidence and claim boundaries

### Completed and safe to state as findings

- Approximately 1,000 hours of existing open-source and proprietary Persian
  speech were gathered for the earlier experiments.
- The long-example and deterministic degradation pipelines were built.
- Whisper-small was trained with and without synthetically degraded data.
- The enhancer–fusion architecture and multistage training were implemented
  and evaluated.
- Multi-condition Whisper-small achieved 17.83% aggregate WER and 7.93%
  aggregate CER, outperforming the completed fusion model overall.
- The final fusion model achieved 18.44% aggregate WER and 14.04% aggregate
  CER, and its gate relied almost entirely on the original stream.
- Whisper Medium training and evaluation are complete. On the same 21,848
  eligible examples as the single-stream Whisper-small models, it achieved
  14.36% aggregate WER and 6.68% aggregate CER. It reduced WER by 3.47 points
  and CER by 1.25 points relative to multi-condition Whisper-small, although
  the capacity change prevents a data-only interpretation. Its per-dataset
  WER/CER scores were 21.99%/10.92% on AGFarsdat, 3.84%/1.65% on Common Voice,
  17.14%/5.42% on FLEURS, 31.65%/13.99% on PersianSpeech, and 31.17%/17.29% on
  Persian Speech Corpus.
- The IranSeda pipeline is complete: 3,777 books and approximately 6,300 raw
  hours were collected, and a 250-hour subset was segmented, transcribed,
  normalized, refined, filtered, and prepared with source-level provenance.

### Remaining experimental work

- Final Whisper-small training with ordinary, long, degraded, and selected
  IranSeda data.
- Final WER/CER and per-dataset comparison against the existing baseline.
- `TODO:` State the answer to research questions five and six only after the
  fixed evaluation is complete.

## Final comparison to report

| Model | Training data | Aggregate WER | Aggregate CER |
| --- | --- | ---: | ---: |
| Whisper-small baseline | Original ordinary + long data | 23.27% | 10.84% |
| Whisper-small multi-condition baseline | Original ordinary + long + degraded data | **17.83%** | **7.93%** |
| Whisper enhancer–fusion | Original ordinary + long + degraded data | 18.44% | 14.04% |
| Whisper-medium Persian | Persian adaptation described in the method chapter | **14.36%** | **6.68%** |
| Whisper-small multi-condition + IranSeda | Original ordinary + long + degraded data + selected pseudo-labelled IranSeda data | `TODO:` | `TODO:` |

The final row must be computed on the same eligible evaluation examples as the
baseline rows. If that is impossible, both the paired subset and the full-set
results must be reported separately rather than compared as if they were
identical evaluations.

## Chapter-level narrative

- **Chapter 1 — Introduction:** Present Persian ASR robustness and data
  scarcity as the connected problem. Introduce the architectural hypothesis
  and the later data-scaling experiment without predicting either outcome.
- **Chapter 2 — Background:** Cover ASR metrics, Whisper, domain mismatch,
  multi-condition training, enhancement for ASR, VAD segmentation,
  pseudo-labelling, and constrained transcript correction.
- **Chapter 3 — Related work:** Connect robust ASR, Persian corpora,
  weakly-supervised speech learning, web-sourced long audio, and LLM-assisted
  data cleaning.
- **Chapter 4 — Method:** Preserve the current data preparation, degradation,
  Whisper baselines, fusion architecture, and multistage training. Add the
  IranSeda acquisition-to-publication pipeline and the final controlled
  Whisper-small experiment.
- **Chapter 5 — Results:** Preserve the completed degradation and fusion
  results, including the negative finding, and report the completed
  Whisper-medium evaluation. Keep only the fixed comparison with the
  IranSeda-trained Whisper-small model pending.
- **Chapter 6 — Conclusion:** Conclude from the complete evidence. If the final
  experiment improves the baseline, emphasize the value of carefully curated
  data scaling; otherwise, emphasize the limits of pseudo-labelled volume and
  the importance of label quality and mixture design.
