# Complete CV25 / Whisper Tiny experiment sequence

Run these blocks **in order**, from `~/MS-Thesis` on the GPU server after syncing
this commit. Wait for each command to succeed before continuing. These commands
cover all nine methods currently registered in the final-test configuration.
They do not launch Whisper Small or the proposed recognition-benefit gate loss:
those extensions are not configured/implemented yet.

Training uses CV25 only. During training, use dev for checkpoint selection and
diagnosis. Final reporting uses the **original**, undegraded CV25 test and
`AGFarsdat_test_normalized/test.tsv`. Freeze design decisions on dev before
running step 8; do not tune against the final test results.

## 1. Environment and dataset preflight

The server must already have the project's working CUDA-enabled `.venv`, `uv`,
and these prepared datasets under `data/`:

- `cv-corpus-25.0`: train/dev/test TSVs and audio.
- `cv-corpus-25.0-degraded-v2`: existing train/dev degraded audio and mappings.
- `AGFarsdat_test_normalized`: test TSV and audio.

No new degradation is generated. Model downloads require internet access.
Use `uv run python` throughout so every command uses the locked project
environment.

Long-running bridge commands print elapsed time and ETA and atomically maintain
`progress.json` in their output directory. This remains available under
`nohup`; `estimated_finish_at` is a throughput-based estimate, not a deadline.

```bash
cd ~/MS-Thesis

# Stop a pasted block at the first failed command.
set -euo pipefail

uv sync

uv run python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'

uv run python -m ml.fusion.prepare_bridge_inputs --dry-run
```

Review the printed skip counts before proceeding. Row-level mismatches are
omitted; structural errors and required splits with no usable clips still fail.

## 2. Generate the paper baseline's waveform inputs and DNSMOS scores

```bash
uv run python -m ml.fusion.prepare_bridge_inputs --device cuda
```

This downloads frozen FRCRN and DNSMOS models, enhances degraded CV25 train/dev
inputs, scores train/dev inputs, and enhances both original test sets. It creates
`artifacts/cv25-tiny/bridge-inputs/bridge_train_dev.jsonl`, `bridge_dev.jsonl`,
provenance, and a ready-to-use `final_tests.yaml`. No test supervision is created.
Rerun this same command to resume completed-clip preparation. Preparation defaults
to `--batch-size 4 --workers 4`; tune those options as described in the bridging
baseline script guide, using a new output directory if the batch size changes.

## 3. Train the shared Whisper Tiny ASR baseline

```bash
uv run python -m ml.asr.train_whisper_small \
  --config configs/speech_enhancement/cv25_tiny/baseline.yaml
```

The historical module name says Small; the configuration selects **Tiny**.
The resulting `models/asr/cv25-tiny/baseline/best` initializes all branches.
Keep this checkpoint fixed once you create the bridge cache or start fusion.

## 4. Cache recognition targets for the published-method reconstruction

```bash
uv run python -m ml.fusion.bridging_experiment prepare \
  --manifest artifacts/cv25-tiny/bridge-inputs/bridge_train_dev.jsonl \
  --asr-checkpoint models/asr/cv25-tiny/baseline/best \
  --output artifacts/cv25-tiny/bridge-cache \
  --device cuda
```

This decodes eleven original/enhanced mixtures per train/dev clip. It can be
expensive; preserve the completed cache. Model IDs are read automatically from
the generated provenance. This cache command requires a new output directory
and does not resume a partial cache.

## 5. Train the paper baseline and its loss ablation

```bash
# Paper baseline reconstruction: perceptual quality + recognition information.
uv run python -m ml.fusion.bridging_experiment train \
  --cache artifacts/cv25-tiny/bridge-cache \
  --output models/asr/cv25-tiny/bridge \
  --device cuda

# Loss ablation: perceptual quality only, with the same cached inputs.
uv run python -m ml.fusion.bridging_experiment train \
  --cache artifacts/cv25-tiny/bridge-cache \
  --output models/asr/cv25-tiny/bridge-pq \
  --pq-only --device cuda
```

Both default to seed 1337 and 45 epochs, selecting `best.pt` by dev objective.
FRCRN and Whisper remain frozen. These trainers require new output directories
and do not resume interrupted training.

## 6. Train the three fusion variants with a shared warm-up enhancer

First run the existing cross-attention architecture through all three stages:

```bash
uv run python -m ml.fusion.train_fusion \
  --config configs/speech_enhancement/cv25_tiny/cross_attention.yaml
```

Reuse its **dev-selected stage-0 enhancer** for the other variants. The stage-0
`enhancer.pt` is saved before later stages and contains the best warm-up weights.
These copies avoid repeating warm-up and give the variants identical enhancer
initialization. Their enhancer configurations match. Run the copy commands once,
before the corresponding variant's first training invocation.

```bash
mkdir -p models/asr/cv25-tiny/gated/checkpoints/stage0_warmup
cp -n models/asr/cv25-tiny/cross_attention/checkpoints/stage0_warmup/enhancer.pt \
  models/asr/cv25-tiny/gated/checkpoints/stage0_warmup/enhancer.pt

uv run python -m ml.fusion.train_fusion \
  --config configs/speech_enhancement/cv25_tiny/gated.yaml \
  --resume-from-stage fusion

mkdir -p models/asr/cv25-tiny/residual_cross_attention/checkpoints/stage0_warmup
cp -n models/asr/cv25-tiny/cross_attention/checkpoints/stage0_warmup/enhancer.pt \
  models/asr/cv25-tiny/residual_cross_attention/checkpoints/stage0_warmup/enhancer.pt

uv run python -m ml.fusion.train_fusion \
  --config configs/speech_enhancement/cv25_tiny/residual_cross_attention.yaml \
  --resume-from-stage fusion
```

The curriculum is warm-up (1,000 steps), fusion (2,000), joint training (4,000).
The latter two variants skip only the shared warm-up computation. Each saves
its final dev-selected checkpoint at `checkpoints/stage2_joint/best.pt`.
The residual variant uses ordinary ASR + Mel loss, without recognition-benefit
gate supervision. These are the committed pilot budgets, not a converged recipe.

## 7. Decode the bridge variants on dev

Fusion and ASR training already perform dev evaluation. The following commands
provide decoded dev WER/CER for both learned bridge variants and its two endpoints.
Each output directory must be new.

```bash
uv run python -m ml.fusion.bridging_experiment evaluate \
  --manifest artifacts/cv25-tiny/bridge-inputs/bridge_dev.jsonl \
  --checkpoint models/asr/cv25-tiny/bridge/best.pt \
  --output artifacts/cv25-tiny/bridge-dev --split dev --device cuda

uv run python -m ml.fusion.bridging_experiment evaluate \
  --manifest artifacts/cv25-tiny/bridge-inputs/bridge_dev.jsonl \
  --checkpoint models/asr/cv25-tiny/bridge-pq/best.pt \
  --output artifacts/cv25-tiny/bridge-pq-dev --split dev --device cuda

uv run python -m ml.fusion.bridging_experiment evaluate \
  --manifest artifacts/cv25-tiny/bridge-inputs/bridge_dev.jsonl \
  --checkpoint models/asr/cv25-tiny/bridge/best.pt \
  --omega 1 --output artifacts/cv25-tiny/bridge-original-dev \
  --split dev --device cuda

uv run python -m ml.fusion.bridging_experiment evaluate \
  --manifest artifacts/cv25-tiny/bridge-inputs/bridge_dev.jsonl \
  --checkpoint models/asr/cv25-tiny/bridge/best.pt \
  --omega 0 --output artifacts/cv25-tiny/bridge-enhanced-dev \
  --split dev --device cuda
```

Review the dev results and training logs now. Any recipe changes require new
consistent runs before final testing. A dev-tuned fixed mixture is a possible
additional experiment, but is not one of the nine registered final methods.

## 8. Evaluate all nine methods on both original test sets

Run this **once the designs and checkpoints are fixed**. Use the generated config,
which has real enhancer identities and paths; the static template has placeholders.

```bash
uv run python -m ml.fusion.evaluate_ablation \
  --config artifacts/cv25-tiny/bridge-inputs/final_tests.yaml \
  --output artifacts/cv25-tiny/final-tests-all \
  --device cuda
```

Omitting `--methods` runs all registered methods in sequence:

| Method | What is evaluated |
| --- | --- |
| `baseline` | Adapted Tiny on original audio |
| `cross_attention` | Existing cross-attention fusion |
| `gated` | Simple gated fusion |
| `residual_cross_attention` | Residual cross-attention fusion |
| `cross_attention_noisy` | Same cross-attention checkpoint, original-view bypass |
| `cross_attention_enhanced` | Same cross-attention checkpoint, enhanced-view bypass |
| `bridge` | Published-method reconstruction, PQ + RI |
| `bridge_pq` | Published-method reconstruction, PQ only |
| `bridge_enhanced_only` | Frozen FRCRN → adapted Tiny, omega = 0 |

Bridge omega = 1 is already represented by `baseline`. View bypass ablations
require no extra training. The runner scores the same original utterances for
every method, with separate CV25 and AGFarsdat WER/CER. Its preflight refuses
empty datasets and writes invalid or over-30-second clip omissions to
`skipped_inputs.jsonl`; every method uses the same filtered cohort.

Results are in `artifacts/cv25-tiny/final-tests-all/summary.json`, with per-method
metrics/predictions and the common `test_manifest.jsonl`. This output directory
must be new. For a failed partial final evaluation, preserve the outputs and run
only missing methods with `--methods` into a new directory; compare manifest
hashes before combining results.

## What this sequence completes

This completes the implemented Tiny experiment matrix and shared final scoring.
It does not establish novelty or reproduce the paper's original dataset results.
The paper reconstruction freezes ASR and uses FRCRN, while the fusion systems
jointly train ASR and a Mel enhancer, so the between-system comparison alone does
not isolate fusion. Small confirmation runs, additional seeds, a matched continued
ASR training control, and recognition-benefit supervision remain later work.
See [method details and reconstruction assumptions](bridging-baseline.md).
