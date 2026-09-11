# CV25 / Whisper Tiny bridging baseline and fusion experiments

## Status and scope

Independent reconstruction of Cui et al., ICASSP 2025,
[arXiv:2501.02452v1](https://arxiv.org/html/2501.02452v1).
This is **not an author-code reproduction or a validated replication**. No
training result or novelty claim follows from the implementation tests.
No remote files were changed and no GPU experiments were launched.

The published baseline mixes original and enhanced **waveforms**, uses frozen
SE/Whisper, and learns from DNSMOS and precomputed WER profiles. Our Mel enhancer
cannot supply its waveform input. Do not invert its Mels or silently substitute
Mel mixing and label that the paper's method.

The first data preparation prerequisite is aligned enhanced WAVs produced by a
named, frozen waveform SE checkpoint (prefer the paper's FRCRN or DCCRN), plus
DNSMOS SIG/BAK from a named scorer. These external models/weights and their
execution are not bundled here. The scripts consume their cached outputs.
Local CUDA is unavailable; real CV25 runtime and results remain unmeasured.

## Where to start: fixed training / dev / test protocol

The final ablation tests are now **original** `data/cv-corpus-25.0/test.tsv`
and `data/AGFarsdat_test_normalized/test.tsv`. No synthetic degradation is
applied to either test set. AGFarsdat is used only for final reporting, never
for training, checkpoint selection, or tuning. All training remains CV25-only.

During runs, baseline ASR validates on original + degraded CV25 dev; fusion
warm-up selects by dev enhancement/feature loss, fusion/joint selects by
degraded CV25 dev WER, and joint also reports original CV25 dev metrics.
The pilot fusion configs now use **full dev**, with `eval_max_batches: null`.
The paper reconstruction selects on its train/dev cache's dev objective.
Final-test evaluation is a separate command and never updates any checkpoint.

Start on the GPU server after syncing the new repository files there:

```bash
cd ~/MS-Thesis
uv run python -m ml.asr.train_whisper_small \
  --config configs/speech_enhancement/cv25_tiny/baseline.yaml
```

Despite its historical script name, this config trains **Whisper Tiny**. It
produces `models/asr/cv25-tiny/baseline/best`, which initializes the other runs.
Next run the three fusion commands in the Commands section below. The paper
baseline can be prepared/trained after the same baseline exists and waveform
SE/DNSMOS outputs have been prepared. To train its PQ-only ablation:

```bash
uv run python -m ml.fusion.bridging_experiment train \
  --cache artifacts/cv25-tiny/bridge-cache \
  --output models/asr/cv25-tiny/bridge-pq --pq-only --device cuda
```

### Final evaluation of all methods

Use the shared final-test runner, rather than combining scores from the older
separate ASR/fusion evaluators. It uses the same original TSV rows, raw stripped
references, 225-new-token greedy decoding, and FP32 inference for all methods.
No reference-token-length filtering is applied. This avoids the older ASR
evaluator's reference filtering and tokenizer round-trip mismatch with fusion.
Results therefore need not numerically match historical evaluation files.
Preflight refuses missing audio, empty/duplicate rows, nonfinite audio, or
clips outside 35 ms–30 s; it never silently drops different clips per method.
If any duration fails, define a common segmentation policy before proceeding.

```bash
uv run python -m ml.fusion.evaluate_ablation --help

# After designs/checkpoints are fixed on dev: baseline and existing-model variants.
uv run python -m ml.fusion.evaluate_ablation \
  --config configs/speech_enhancement/cv25_tiny/final_tests.yaml \
  --methods baseline cross_attention gated residual_cross_attention \
    cross_attention_noisy cross_attention_enhanced \
  --output artifacts/cv25-tiny/final-tests-fusion --device cuda

# Once both bridge checkpoints and enhanced test waveforms exist: all methods.
uv run python -m ml.fusion.evaluate_ablation \
  --config configs/speech_enhancement/cv25_tiny/final_tests.yaml \
  --output artifacts/cv25-tiny/final-tests-all --device cuda
```

All final methods are registered in `final_tests.yaml`: standard Tiny, three
fusion architectures, current fusion's noisy/enhanced inference ablations,
bridge PQ+RI, bridge PQ-only, and the bridge's enhancement-only endpoint.
Select subsets with `--methods` to avoid repeating completed inference; use a
new output directory for every invocation. This is sequential offline evaluation,
one utterance at a time; no training or test-based model selection happens here.

For bridge tests, `noisy` means the **original test waveform**, not an artificially
degraded copy. Pass that original audio through the same frozen waveform enhancer
used in bridge training. Store its aligned outputs under:

```text
artifacts/cv25-tiny/bridge-test-enhanced/cv-corpus-25.0/<TSV path with .wav suffix>
artifacts/cv25-tiny/bridge-test-enhanced/AGFarsdat_test_normalized/<TSV path with .wav suffix>
```

The path is relative to `clips/`; a leading `clips/` in the TSV is stripped.
Set `bridge_enhancer_id` in `final_tests.yaml` to the exact ID recorded when
preparing the bridge cache. The runner verifies it and the ASR checkpoint
against the bridge checkpoint provenance. DNSMOS and test WER targets are
**not** needed for final inference. Generating enhanced audio does not change
which original dataset is under evaluation.

Outputs: `test_manifest.jsonl`, effective config, `<method>.predictions.jsonl`,
`<method>.metrics.json`, and `summary.json` with **separate WER/CER for each
dataset**, aggregate scores, and a common test-manifest hash. When running methods
separately, compare the hashes to verify the cohort is identical. These original
test sets measure transfer/original-speech behavior; they cannot alone establish
robustness on synthetically degraded test conditions.

## What matches, and what is reconstructed

- Eq. 3 waveform OA with **omega weighting original audio**; omega=1 is noisy-only.
- Eleven WER targets, coefficients **1.0, 0.9, ..., 0.0**, as the descending-OA
  ordering stated in Eq. 6. Never sort WER values by magnitude.
- Eqs. 5–7: normalized SIG/BAK target; sigmoid/cosine/log-sigmoid RI loss;
  combined objective `(PQ + RI) / 2`. WER is an unbounded nonnegative ratio,
  not percentage points. No gradients through ASR or SE.
- 80-bin Kaldi Fbank, 25 ms windows / 10 ms hop; these are distinct from Whisper
  log-Mels. Shared utterance encoder, SE-Res2 blocks with dilations 2/3/4,
  scale 8, attention statistics pooling, 256 or 384 channels, hidden size 384.
- Time/frequency masks of up to 5/4; LR 0.0005, 45 epochs, gradient norm 10.

Underspecified details are fixed explicitly: channel-wise attentive mean/std
pooling, concatenation of three block outputs before projection, concatenation
of the two utterance embeddings, a 384-unit ReLU layer then 11 recognition
logits, and a linear sigmoid coefficient head on those logits. Review Fig. 1
against any subsequently obtained author code before claiming architectural
fidelity. These head/aggregation choices are reconstruction assumptions.

Other declared choices: Adam, no unspecified paper warm-up schedule, one
unpadded utterance per forward with gradient accumulation (avoids padding in
batch normalization and pooling), zero Fbank dither, dev-loss checkpoint
selection, and strip-only transcript handling. The published RI-only variant
is not exposed: without PQ the separate coefficient head would lack direct
supervision in this reconstruction. `--pq-only` is a supported loss ablation.

CV25 Persian, Tiny, our adapted ASR checkpoint, and synthetic telecom conditions
are protocol changes from the paper. They do not reproduce its CHiME-4 results.

## Input contract

UTF-8 JSONL, unique `id`, paths absolute or relative to the manifest directory:

```json
{"id":"cv25-train-clip1-v1","source_id":"cv25-clip1","speaker_id":"speaker1","split":"train","sentence":"متن فارسی","noisy_path":"noisy/clip1.wav","enhanced_path":"enhanced/clip1.wav","dnsmos_sig":3.5,"dnsmos_bak":2.8}
```

`source_id` identifies the original CV25 clip, shared across all its variants.
Include `speaker_id` wherever available. Train/dev preparation rejects source
or supplied-speaker overlap and rejects all test rows. Evaluation needs the
same fields except DNSMOS; only the requested dev/test split is accepted.
Audit test IDs against training IDs separately before final evaluation.
Normalize the reference transcripts identically for baseline and fusion.
Audio must be aligned, finite, mono after channel averaging, and no longer than
30 seconds. Resampling is to 16 kHz; unequal lengths are refused rather than
silently repaired. Equal lengths alone do not establish alignment: verify SE
delay upstream. No automatic normalization changes the waveform mixing ratio.

## Commands

Run from the repository root. Every subcommand supports `--help` with defaults.
Output directories must be new; an existing directory is refused to prevent
accidental replacement. This initial baseline trainer does not automatically
resume a partial run. Preserve completed caches to avoid repeated ASR inference.

```bash
uv run python -m ml.fusion.bridging_experiment prepare --help
uv run python -m ml.fusion.bridging_experiment train --help
uv run python -m ml.fusion.bridging_experiment evaluate --help

uv run python -m ml.asr.train_whisper_small --config configs/speech_enhancement/cv25_tiny/baseline.yaml

uv run python -m ml.fusion.bridging_experiment prepare \
  --manifest data/speech_enhancement/bridge_train_dev.jsonl \
  --asr-checkpoint models/asr/cv25-tiny/baseline/best \
  --enhancer-id FRCRN_CHECKPOINT_REVISION --dnsmos-id DNSMOS_SCORER_REVISION \
  --output artifacts/cv25-tiny/bridge-cache --device cuda

uv run python -m ml.fusion.bridging_experiment train \
  --cache artifacts/cv25-tiny/bridge-cache \
  --output models/asr/cv25-tiny/bridge --device cuda

uv run python -m ml.fusion.bridging_experiment evaluate \
  --manifest data/speech_enhancement/bridge_dev.jsonl \
  --checkpoint models/asr/cv25-tiny/bridge/best.pt \
  --output artifacts/cv25-tiny/bridge-dev --split dev --device cuda

uv run python -m ml.fusion.train_fusion --config configs/speech_enhancement/cv25_tiny/cross_attention.yaml
uv run python -m ml.fusion.train_fusion --config configs/speech_enhancement/cv25_tiny/gated.yaml
uv run python -m ml.fusion.train_fusion --config configs/speech_enhancement/cv25_tiny/residual_cross_attention.yaml
```

Use `--omega 1` / `--omega 0` during bridge evaluation for original / enhanced
endpoints. Tune fixed OA on **dev only**, then fix it before test. Preparation
decodes eleven mixtures per clip, so first profile a deterministic train/dev
subset, then cache all selected targets once. Do not use test to choose a gate,
coefficient, epoch, loss, or architecture. Cache metadata records the manifest
hash, backbone, enhancer/scorer IDs, coefficient order and decoding limit.
Regenerate the cache when any upstream checkpoint or decoding policy changes.

## Experiment order and fair claims

1. Audit CV25 splits and matched scoring; profile Tiny on a fixed small subset.
2. Establish the CV25 clean+degraded adapted Tiny. All branches start there.
3. Run paper reconstruction with fixed waveform SE; compare its own noisy-only,
   enhanced-only, fixed-OA, PQ-only and PQ+RI variants with the same frozen ASR.
4. Run current fusion, simple gating, and residual correction with the common
   Mel enhancer configuration; then use same-checkpoint inference ablations.
5. Only after diagnosing useful complementary errors, implement and test explicit
   recognition-benefit gate supervision. **That objective is not implemented in
   this change**. The residual variant currently uses ordinary ASR + Mel loss.
6. Confirm selected Tiny systems on Small; repeat decisive seeds as runtime allows.

The Tiny YAMLs are compact pilot budgets, not v2 reproductions or finalized
recipes. Selecting a different `fusion.type` replaces the default architecture
options instead of inheriting cross-attention-only constructor arguments.
The existing trainer jointly updates the enhancer; its
three stages are retained. Paired variants must share warm-up initialization;
check this before attributing differences to fusion. Periodic dev evaluation
now uses full CV25 dev splits. Reserve Small/writing time before enlarging runs.

The paper baseline freezes ASR and uses a different enhancer from our full
system. A system-level comparison therefore does **not** isolate fusion alone.
Report frozen-ASR results separately from joint-finetuning results and retain a
matched continued noisy-only training control. Within-method ablations isolate
the contributions of the gate and attention; report parameter count, GPU time,
WER/CER and original-speech regression. A matched waveform-view experiment
would require feeding the same external enhanced audio into our encoder-fusion
path; it is a later extension, not present in the Mel-enhancer YAMLs.

For residual fusion, zero gate is an exact bypass of the correction at the
fusion module. Its gate measures correction strength, not information fraction.
For the previous cross-attention design, zero final gate is **not** noisy-only:
enhanced information can already enter through cross-attention. `--view-mode
noisy` bypasses the entire frontend in the existing evaluator. Gates are not
causal explanations; use pathway interventions and decoded errors.
