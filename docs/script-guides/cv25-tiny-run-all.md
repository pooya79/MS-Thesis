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
and these source datasets under `data/`:

- `cv-corpus-25.0`: train/dev/test TSVs and audio.
- `cv-corpus-25.0-degraded-v2`: existing train/dev degraded audio and mappings.
- `AGFarsdat_test_normalized`: test TSV and audio.

Create the isolated official cohort once before preflight:

```bash
uv run python -m ml.speech_data.prepare_cv25_official
```

See [official split preparation](prepare-cv25-official.md). All configs use
`data/cv25-official` and fresh `cv25-tiny-official` outputs. Never resume old
expanded-data checkpoints. The archive's original test split is also restored,
so no official test clip can enter training through the old expanded split.

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

uv run python -m ml.fusion.prepare_bridge_inputs --data-root data/cv25-official --output artifacts/cv25-tiny-official/bridge-inputs --dry-run
```

Review the printed skip counts before proceeding. Row-level mismatches are
omitted; structural errors and required splits with no usable clips still fail.

For the fixed recipe below, the complete sequence can also run unattended after
reviewing `selection.json`:

```bash
bash ml/speech_data/scripts/run_cv25_tiny_official.sh --help
nohup bash ml/speech_data/scripts/run_cv25_tiny_official.sh > /tmp/cv25-tiny-official.log 2>&1 < /dev/null &
```

The launcher accepts no arguments, uses CUDA and the committed configs, and
runs preflight followed by steps 2–8 in order. It makes no recipe changes based
on dev or test scores. Logs, PID, start/finish times and exit code are saved in
`artifacts/cv25-tiny-official/run-all`. It refuses an existing launcher directory;
resume failed work manually using the individual commands and their documented
resume rules. Do not start a second launcher alongside an active run.

## 2. Generate the paper baseline's waveform inputs and DNSMOS scores

```bash
uv run python -m ml.fusion.prepare_bridge_inputs --data-root data/cv25-official --output artifacts/cv25-tiny-official/bridge-inputs --device cuda
```

This downloads frozen FRCRN and DNSMOS models, enhances degraded CV25 train/dev
inputs, scores train/dev inputs, and enhances both original test sets. It creates
`artifacts/cv25-tiny-official/bridge-inputs/bridge_train_dev.jsonl`, `bridge_dev.jsonl`,
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
The resulting `models/asr/cv25-tiny-official/baseline/best` initializes all branches.
Keep this checkpoint fixed once you create the bridge cache or start fusion.

The baseline trains for **1 epoch** over usable original + degraded train rows,
with microbatch 8, gradient accumulation 2 (effective batch 16 on one GPU), and
evaluation batch 128. Evaluation and saving happen every **1,000 optimizer
updates**, plus every epoch end, so a shorter run still selects a checkpoint.
These settings are in `baseline.yaml`; `eval_save_at_epoch_end: true` supplements
the requested `eval_steps: 1000` and `save_steps: 1000` cadence.

This is a changed training recipe. Use fresh outputs for the entire sequence;
do not auto-resume the old baseline or reuse its dependent caches/checkpoints.
If prior outputs exist, preserve them under a separate experiment directory
before starting this recipe at the configured paths.

## 4. Cache recognition targets for the published-method reconstruction

```bash
uv run python -m ml.fusion.bridging_experiment prepare \
  --manifest artifacts/cv25-tiny-official/bridge-inputs/bridge_train_dev.jsonl \
  --asr-checkpoint models/asr/cv25-tiny-official/baseline/best \
  --output artifacts/cv25-tiny-official/bridge-cache \
  --device cuda --batch-size 8
```

This decodes eleven original/enhanced mixtures per train/dev clip. It can be
expensive; preserve the completed cache. Model IDs are read automatically from
the generated provenance. Mixtures are decoded in GPU batches; `--batch-size`
counts clips (8 means 88 waveforms). For a 48 GB GPU, try 16 or 32 and reduce it
if CUDA runs out of memory. Continue a compatible partial cache by rerunning the
same command with `--resume`; preparation validates the completed prefix before
starting at the first missing or invalid item.

## 5. Train the paper baseline and its loss ablation

```bash
# Paper baseline reconstruction: perceptual quality + recognition information.
uv run python -m ml.fusion.bridging_experiment train \
  --cache artifacts/cv25-tiny-official/bridge-cache \
  --output models/asr/cv25-tiny-official/bridge \
  --device cuda

# Loss ablation: perceptual quality only, with the same cached inputs.
uv run python -m ml.fusion.bridging_experiment train \
  --cache artifacts/cv25-tiny-official/bridge-cache \
  --output models/asr/cv25-tiny-official/bridge-pq \
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
mkdir -p models/asr/cv25-tiny-official/gated/checkpoints/stage0_warmup
cp -n models/asr/cv25-tiny-official/cross_attention/checkpoints/stage0_warmup/enhancer.pt \
  models/asr/cv25-tiny-official/gated/checkpoints/stage0_warmup/enhancer.pt

uv run python -m ml.fusion.train_fusion \
  --config configs/speech_enhancement/cv25_tiny/gated.yaml \
  --resume-from-stage fusion

mkdir -p models/asr/cv25-tiny-official/residual_cross_attention/checkpoints/stage0_warmup
cp -n models/asr/cv25-tiny-official/cross_attention/checkpoints/stage0_warmup/enhancer.pt \
  models/asr/cv25-tiny-official/residual_cross_attention/checkpoints/stage0_warmup/enhancer.pt

uv run python -m ml.fusion.train_fusion \
  --config configs/speech_enhancement/cv25_tiny/residual_cross_attention.yaml \
  --resume-from-stage fusion
```

The curriculum uses **1 epoch per stage** as an initial pilot budget, shared
across all three variants. Warm-up and fusion epochs cover degraded pairs only;
joint epochs cover degraded + original examples. These counts are independent
pilot choices, not a conversion of the former 1,000/2,000/4,000-step budgets.
Increase them consistently across variants only after reviewing dev results.

All stages use microbatch 8, gradient accumulation 2 (effective batch 16), and
evaluation batch 128, matching the ASR baseline's batch settings. An incomplete
accumulation group is flushed at each epoch end. Evaluation and saving run every
1,000 optimizer updates and at every epoch end. Full dev splits remain enabled.
If fusion evaluation does not fit GPU memory, reduce `eval_batch_size` identically
across variants; this changes evaluation throughput, not the training budget.

The cosine schedule derives its update count from each stage's actual loader;
`warmup_ratio: 0.05` uses 5% of that budget (rounded down to whole updates).
`config/{warmup,fusion,joint}_budget.json` records usable examples, microbatches,
optimizer updates per epoch, and the resolved total. Metrics record both step
and epoch. See [training budget semantics](enhancement-and-fusion.md#epoch-budgets-and-gradient-accumulation).

The latter two variants skip only the shared warm-up computation. Each saves
its final dev-selected checkpoint at `checkpoints/stage2_joint/best.pt`.
The residual variant uses ordinary ASR + Mel loss, without recognition-benefit
gate supervision. The bridge keeps its published 45-epoch frozen-ASR recipe;
matching epoch units does not make its compute or trainable parameters equal to
fusion. These remain pilot budgets, not a converged recipe.

## 7. Decode the bridge variants on dev

Fusion and ASR training already perform dev evaluation. The following commands
provide decoded dev WER/CER for both learned bridge variants and its two endpoints.
Each output directory must be new.

```bash
uv run python -m ml.fusion.bridging_experiment evaluate \
  --manifest artifacts/cv25-tiny-official/bridge-inputs/bridge_dev.jsonl \
  --checkpoint models/asr/cv25-tiny-official/bridge/best.pt \
  --output artifacts/cv25-tiny-official/bridge-dev --split dev --device cuda

uv run python -m ml.fusion.bridging_experiment evaluate \
  --manifest artifacts/cv25-tiny-official/bridge-inputs/bridge_dev.jsonl \
  --checkpoint models/asr/cv25-tiny-official/bridge-pq/best.pt \
  --output artifacts/cv25-tiny-official/bridge-pq-dev --split dev --device cuda

uv run python -m ml.fusion.bridging_experiment evaluate \
  --manifest artifacts/cv25-tiny-official/bridge-inputs/bridge_dev.jsonl \
  --checkpoint models/asr/cv25-tiny-official/bridge/best.pt \
  --omega 1 --output artifacts/cv25-tiny-official/bridge-original-dev \
  --split dev --device cuda

uv run python -m ml.fusion.bridging_experiment evaluate \
  --manifest artifacts/cv25-tiny-official/bridge-inputs/bridge_dev.jsonl \
  --checkpoint models/asr/cv25-tiny-official/bridge/best.pt \
  --omega 0 --output artifacts/cv25-tiny-official/bridge-enhanced-dev \
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
  --config artifacts/cv25-tiny-official/bridge-inputs/final_tests.yaml \
  --output artifacts/cv25-tiny-official/final-tests-all \
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

Results are in `artifacts/cv25-tiny-official/final-tests-all/summary.json`, with per-method
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
