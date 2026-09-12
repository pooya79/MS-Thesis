# CV25 / Whisper Tiny bridging baseline and fusion experiments

For one complete ordered command sequence, use
[Run all Tiny experiments](cv25-tiny-run-all.md). It includes input preparation,
the PQ-only run, shared warm-up initialization, and final scoring of all methods.

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

The missing waveform/DNSMOS stage is implemented in
`ml.fusion.prepare_bridge_inputs`. It downloads the authors' FRCRN checkpoint
and Microsoft's DNSMOS P.835 model, then generates aligned float WAVs, scores,
manifest files and provenance. No hand-built manifest or model-ID placeholders
are needed. Model weights remain generated artifacts outside Git.

### Prepare the inputs (can run before Tiny training)

Sync the project's `.venv`. The project dependency configuration installs
ClearVoice while overriding its outdated NumPy, librosa, and soundfile pins with
the versions used by this project. The direct FRCRN adapter has been smoke-tested
with the project runtime; other ClearVoice pipelines are not used here.

```bash
cd ~/MS-Thesis
uv sync

# Validate metadata and report counts without loading models or changing data.
uv run python -m ml.fusion.prepare_bridge_inputs --dry-run

# Download models once, then prepare all train/dev inputs and original test views.
uv run python -m ml.fusion.prepare_bridge_inputs --device cuda
```

Default sources: degraded CV25 train/dev mapping and original CV25 + AGFarsdat
`test.tsv` files. No new degradation is generated and clean reference audio is
never used as the enhanced view. DNSMOS is computed on original noisy train/dev
inputs only. Test receives enhancement, but no DNSMOS or recognition targets.
`--scope train-dev` or `--scope test` prepares only that phase. Train/dev source
identities are audited even in test scope. CV25 client IDs, when present, are
used to detect cross-split speaker leakage. Inconsistent clip rows are skipped
before selection and reported by reason; structural dataset errors or a required
split with no usable clips still stop the command.

For a small environment pilot, use a separate output directory:

```bash
uv run python -m ml.fusion.prepare_bridge_inputs \
  --scope train-dev --max-per-split 10 \
  --output artifacts/cv25-tiny/bridge-inputs-pilot --device cuda
```

Rerun the same command to resume: completed records are verified against source
and output hashes and the model identities, then reused. Do not change the
selection limit/seed in an existing output directory. Final manifests are written
only after the selected scope completes. Keep pilot manifests out of full runs.

The default output is `artifacts/cv25-tiny/bridge-inputs/`:
- `bridge_train_dev.jsonl` and `bridge_dev.jsonl`, with scored waveform pairs;
- matching `.provenance.json` sidecars with automatic model IDs;
- `bridge_test.jsonl` and `test-enhanced/` for the two original test sets;
- `final_tests.yaml`, prefilled with the correct enhanced root and enhancer ID;
- `skipped_inputs.jsonl`, containing every omitted clip and its reason;
- `completed/` records for resume, and per-scope completion reports.

Only trailing model-input padding is removed; no shifts or gain normalization
are applied. Equal length is checked. An actual CPU adapter smoke test and a
synthetic four-clip preparation/resume test verify mechanics, not speech quality
or research performance. FRCRN may learn intrinsic processing effects that need
examining on real speech; equal lengths alone do not prove phonetic alignment.

Pretrained sources:
[FRCRN authors](https://github.com/alibabasglab/FRCRN),
[ClearVoice model code](https://github.com/modelscope/ClearerVoice-Studio/tree/main/clearvoice),
[FRCRN weights](https://huggingface.co/alibabasglab/FRCRN_SE_16K), and
[Microsoft DNSMOS](https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/dnsmos_local.py).
FRCRN revision and DNSMOS checksum are pinned in `bridge_pretrained.py`.
The ClearVoice FRCRN release is identified explicitly; its identity is not proof
that it is the exact checkpoint used in the bridging paper. DNSMOS uses the
non-personalized P.835 model, repetition for clips under 9.01 seconds, 1-second
hops and Microsoft's polynomial calibration. Calibrated scores outside [1,5]
are clipped for supervision with unclipped values also retained in the manifest.
This bounding is a declared implementation choice. P.808 is not computed.

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
Next run the three fusion commands in the Commands section below. The bridge WER cache and training require the same ASR baseline. Its waveform
inputs can be prepared beforehand using the commands above. To train its PQ-only ablation:

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
Preflight omits missing, malformed, duplicate, nonfinite, unaligned, or
out-of-duration clips before loading models. The omissions are written to
`skipped_inputs.jsonl`, and every method receives the same filtered cohort.
An empty required dataset still stops evaluation.

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
  --config artifacts/cv25-tiny/bridge-inputs/final_tests.yaml \
  --output artifacts/cv25-tiny/final-tests-all --device cuda
```

All final methods are registered in `final_tests.yaml`: standard Tiny, three
fusion architectures, current fusion's noisy/enhanced inference ablations,
bridge PQ+RI, bridge PQ-only, and the bridge's enhancement-only endpoint.
Select subsets with `--methods` to avoid repeating completed inference; use a
new output directory for every invocation. This is sequential offline evaluation,
one utterance at a time; no training or test-based model selection happens here.

For bridge tests, `noisy` means the original test waveform. Use the generated
`artifacts/cv25-tiny/bridge-inputs/final_tests.yaml` after input preparation,
which supplies the enhanced paths and verified model ID automatically. There
are no DNSMOS or WER training targets for test clips. Generating enhanced audio
does not change which original dataset is under evaluation.

Outputs: `test_manifest.jsonl`, `skipped_inputs.jsonl`, effective config, `<method>.predictions.jsonl`,
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
Include `speaker_id` wherever available. Train/dev preparation omits every row
involved in source or supplied-speaker overlap and rejects all test rows. Evaluation needs the
same fields except DNSMOS; only the requested dev/test split is accepted.
Audit test IDs against training IDs separately before final evaluation.
Normalize the reference transcripts identically for baseline and fusion.
Audio must be aligned, finite, mono after channel averaging, and no longer than
30 seconds. Resampling is to 16 kHz; invalid or unequal pairs are omitted and
reported rather than repaired. Equal lengths alone do not establish alignment: verify SE
delay upstream. No automatic normalization changes the waveform mixing ratio.

## Commands

Run from the repository root. Every subcommand supports `--help` with defaults.
The bridge WER-cache/training/evaluation output directories must be new.
The separate waveform preparation command is resumable in its existing output. This initial baseline trainer does not automatically
resume a partial run. Preserve completed caches to avoid repeated ASR inference.

```bash
uv run python -m ml.fusion.bridging_experiment prepare --help
uv run python -m ml.fusion.bridging_experiment train --help
uv run python -m ml.fusion.bridging_experiment evaluate --help

uv run python -m ml.asr.train_whisper_small --config configs/speech_enhancement/cv25_tiny/baseline.yaml

uv run python -m ml.fusion.bridging_experiment prepare \
  --manifest artifacts/cv25-tiny/bridge-inputs/bridge_train_dev.jsonl \
  --asr-checkpoint models/asr/cv25-tiny/baseline/best \
  --output artifacts/cv25-tiny/bridge-cache --device cuda

uv run python -m ml.fusion.bridging_experiment train \
  --cache artifacts/cv25-tiny/bridge-cache \
  --output models/asr/cv25-tiny/bridge --device cuda

uv run python -m ml.fusion.bridging_experiment evaluate \
  --manifest artifacts/cv25-tiny/bridge-inputs/bridge_dev.jsonl \
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
