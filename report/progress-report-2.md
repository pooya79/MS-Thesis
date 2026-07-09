# MS Thesis Progress Report 2

## 1. Scope of this report

Since the first progress report, I have revised the enhancement component of the Whisper fusion system, introduced an ASR-aware loss for enhancer warm-up, and trained and evaluated a new fusion model. This report describes the motivation for these changes, the revised architecture and training objective, and the resulting recognition performance.

The motivation for this revision was the behavior of the first fusion model. Its enhancer was trained during warm-up only with an L1 distance between enhanced and clean log-Mel spectrograms. Although this objective encourages local spectrogram reconstruction, a lower log-Mel error does not necessarily preserve the representations that Whisper needs for transcription. In practice, the enhanced features could become less useful for ASR even when their spectrogram-level L1 loss improved. Therefore, the new design explicitly optimizes both signal-level similarity and similarity in the feature space of the ASR encoder.

## 2. Architecture changes

The overall system remains a dual-view Whisper model. It receives the original log-Mel representation and an enhanced version of the same input, encodes both with a shared Whisper encoder, exchanges information between the two streams through bidirectional cross-attention, and combines them with a learned gate before decoding.

The main architectural change is inside the residual U-Net enhancer. The original convolutional U-Net had a limited temporal receptive field. This is undesirable for suppressing non-stationary noise and reverberation, whose effects may extend over a much longer interval. I therefore added an optional long-range temporal module at the deepest U-Net bottleneck. It reshapes the bottleneck feature map into a sequence along time, processes it using either a Transformer encoder or a bidirectional GRU, and then projects it back to the convolutional feature map.

For fusion v2, the selected bottleneck was a two-layer Transformer with four attention heads, a hidden dimension of 256, and zero dropout. Its output projection is initialized to zero and applied as a residual connection. Consequently, the new module starts as an exact identity mapping and does not disturb the identity initialization of the residual enhancer at the beginning of training.

The rest of the fusion block was retained from v1: three layers of bidirectional cross-attention, 12 attention heads, an FFN expansion ratio of 4, and a learned feature-wise gate. The enhancer also retained three U-Net levels and 48 base channels.

## 3. ASR-aware enhancer loss

The second change was to replace the L1-only warm-up objective with a combination of log-Mel reconstruction and Whisper encoder feature matching. Let \(E\) denote the enhancer, \(H\) the frozen Whisper encoder, \(X_n\) the noisy log-Mel input, and \(X_c\) its clean, bandwidth-aligned target. The losses are

\[
L_{\mathrm{enh}} = \lVert E(X_n)-X_c \rVert_1,
\]

\[
L_{\mathrm{feat}} = \lVert H(E(X_n))-H(X_c) \rVert_1,
\]

and the new warm-up objective is

\[
L_{\mathrm{warmup}} = \lambda L_{\mathrm{enh}} + \beta L_{\mathrm{feat}}.
\]

In the new run, both \(\lambda\) and the feature-matching weight \(\beta\) were set to 0.5. The Whisper encoder is frozen while computing this loss, and the clean-side encoder representation is treated as a fixed target; gradients therefore update only the enhancer. Because the Whisper encoder expects a full 30-second log-Mel window, feature-aware warm-up uses full windows rather than the four-second crops used by the original L1-only warm-up. The combined validation loss, rather than raw log-Mel L1 alone, is used to select the best enhancer obtained during warm-up.

This objective does not require the enhanced spectrogram to reproduce every clean log-Mel value perfectly. Instead, it also penalizes changes that move the enhanced signal away from the clean signal in the representation space actually used by the ASR model.

## 4. Training of fusion v2

The new model used the same normal, long-utterance, and telephone/VoIP-degraded Persian training datasets as fusion v1. The run used seed 1337 and the following three-stage curriculum:

1. **Enhancer warm-up:** 20,000 steps, batch size 8, using the combined log-Mel and encoder-feature loss.
2. **Fusion training:** 30,000 steps, batch size 8, training the enhancer and fusion layers while keeping Whisper frozen.
3. **Joint training:** 120,000 steps, batch size 4, training the entire enhancer, fusion module, and Whisper backbone end to end.

The frontend learning rates were \(2\times10^{-4}\) in Stage 1 and \(1\times10^{-4}\) in Stage 2. The Whisper learning rate during joint training was reduced to \(5\times10^{-6}\). Every stage used a linear warm-up followed by cosine decay. The joint ASR stages continued to optimize ASR cross-entropy together with a smaller log-Mel enhancement term.

Unlike fusion v1, which started from the original pretrained Whisper-small model, fusion v2 was initialized from the Persian Whisper model previously fine-tuned with degraded data. The entire model was then jointly optimized during the final training stage, and the best jointly trained version was used for evaluation.

## 5. Evaluation results

The new model was evaluated on the same five test sets used previously. Fusion v1 and v2 each contain 21,854 evaluated examples. Lower WER and CER are better.

### Aggregate comparison

| Model | Aggregate WER | Aggregate CER |
|---|---:|---:|
| Whisper trained with degraded data | **17.83%** | **7.93%** |
| Whisper fusion v1 | 24.35% | 15.46% |
| Whisper fusion v2 | 18.44% | 14.04% |

Compared with fusion v1, fusion v2 reduced aggregate WER by **5.90 percentage points**, from 24.35% to 18.44%. This is a **24.25% relative WER reduction**. Aggregate CER decreased by **1.42 percentage points**, from 15.46% to 14.04%, corresponding to a **9.20% relative reduction**.

Fusion v2 is substantially closer to the strongest standalone Whisper model, but it has not yet surpassed it. Its aggregate WER is 0.62 percentage points higher than that baseline. The standalone Whisper evaluation contains 21,848 examples because six over-length labels were skipped, so this aggregate comparison has a small sample-count difference.

### Per-dataset comparison

| Test dataset | Fusion v1 WER | Fusion v2 WER | Change | Fusion v1 CER | Fusion v2 CER | Change |
|---|---:|---:|---:|---:|---:|---:|
| AGFarsdat | 29.45% | **25.38%** | −4.07 pp | 13.79% | **12.38%** | −1.41 pp |
| Common Voice 25 | 18.37% | **9.06%** | −9.30 pp | 17.87% | **16.08%** | −1.79 pp |
| FLEURS | 21.55% | **19.61%** | −1.93 pp | 6.68% | **6.23%** | −0.46 pp |
| PersianSpeech | **28.78%** | 31.18% | +2.40 pp | **11.33%** | 12.91% | +1.57 pp |
| Persian Speech Corpus | 37.26% | **35.17%** | −2.09 pp | 21.91% | **21.07%** | −0.85 pp |

Fusion v2 improved WER and CER on four of the five datasets. The largest WER reduction was on Common Voice, where WER fell by 9.30 percentage points. Performance regressed on PersianSpeech; however, this test set contains only 24 examples, so its estimate has much higher uncertainty than the results for AGFarsdat and Common Voice.

## 6. Gate behavior and interpretation

The gate behavior changed strongly between the two fusion runs. Fusion v1 assigned an average weight of 51.54% to the enhanced view and 48.46% to the original view. Fusion v2 assigned only **0.39%** to the enhanced view and **99.61%** to the original/noisy view. The corresponding median weights were 0.39% and 99.61%.

This shows that the new system learned to preserve almost all of the original representation and to suppress the enhanced stream. That behavior is consistent with the original concern that enhancement can remove ASR-relevant information, and it prevents the enhancer from strongly damaging the decoder input. However, it also means that the current accuracy gain should not be interpreted as evidence that the enhanced view is yet contributing substantially to recognition. Part of the improvement is likely due to the stronger Persian degraded-data initialization, the longer joint-training stage, and the gate learning to fall back to the original stream.

Because several factors changed together, this run does not isolate the individual causal contribution of the temporal bottleneck or feature-matching loss. A controlled ablation is required before attributing the full improvement to either change.

## 7. Conclusion and next work

The second fusion model resolves much of the performance loss observed in v1. It reduces aggregate WER by 5.90 percentage points and improves four of five evaluation datasets, bringing the dual-view system close to the strongest standalone Whisper baseline. The ASR-aware feature loss provides a more task-aligned training target than log-Mel L1 alone, while the temporal bottleneck gives the enhancer access to long-range context.

The remaining issue is that the learned gate almost completely rejects the enhanced view. The next experiments should therefore:

1. Evaluate noisy-only, enhanced-only, and fused inference using the same trained v2 model.
2. Run controlled ablations for the temporal bottleneck, feature-matching loss, and fine-tuned backbone initialization.
3. Measure enhancement headroom in both log-Mel and Whisper encoder feature space, separated by degradation and bandwidth condition.
4. Investigate gate regularization or staged gate training so that the enhanced stream is used only where it provides measurable ASR benefit.
5. Re-evaluate on matched degraded test sets in addition to the current mixed clean/telephone evaluation sets.

At this stage, fusion v2 is a clear improvement over fusion v1, but the best overall system remains the standalone Whisper model trained with degraded data.
