# Whisper Small compact cross-attention fusion

This configuration starts a new Whisper Small fusion run derived from the
reported fusion v2 experiment. It does not train a new ASR baseline and does not
use the isolated CV25-official Tiny cohort. It initializes from the existing
Persian Whisper Small model trained on the report's clean and degraded mixture:

```text
models/asr/whisper-small-deg-v2/runs/whisper-small-fa/best
```

The dataset mixture, ASR-aware warm-up, and learning rates come from
`report/whisper-fusion-v2/fusion_train_v4.yaml`. The frontend capacity matches
the Tiny cross-attention experiment:

- residual U-Net: 32 base channels, depth 3, no temporal bottleneck;
- cross-attention: one layer, four heads, FFN ratio 2.

During the joint stage, every usable degraded training row is used exactly once
per epoch. `joint_sampling.clean_fraction: 0.25` adds a random 25% of the pooled
usable clean rows, rounded down, without replacement. This is 25% of available
clean data, not 25% of the final training mixture. Larger datasets contribute
according to their row counts; there is no equal-dataset weighting. The selected
rows are shuffled together. Seed 1337 plus the epoch number makes selection and
order reproducible, including on resume; a fresh clean subset is selected each
epoch. Warm-up and frozen-backbone fusion remain degraded-only, and dev
evaluation is unchanged and unweighted.

Each stage runs for one epoch and evaluates every 10,000 optimizer steps as well
as at epoch end. Every stage saves at epoch end, and the cosine scheduler uses a
5% warm-up ratio calculated from that stage's resolved optimizer-update count.
A joint epoch contains `degraded_rows + floor(0.25 * clean_rows)` examples.
Existing weighted joint checkpoints cannot resume this changed recipe in place;
use a new run directory if you already started the weighted experiment.

Start the new run with:

```bash
uv run python -m ml.fusion.train_fusion \
  --config configs/speech_enhancement/fusion_train_compact.yaml
```

The run writes only to `models/asr/fusion/run_005_compact`; the reported v1/v2
configs and their output directories remain unchanged. The trainer resumes an
interrupted stage from that new directory when the same command is run again.
