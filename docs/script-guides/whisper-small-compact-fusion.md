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

During the joint stage, the new recipe samples 75% degraded and 25% clean
examples. It also assigns equal probability mass to each dataset within those
two groups, rather than letting the roughly 600,000-row FarsSpon corpus dominate
the clean share. Sampling is deterministic from seed 1337 and uses replacement.
Warm-up and frozen-backbone fusion remain degraded-only, and dev evaluation is
unchanged and unweighted.

Training is epoch-based rather than using fusion v2's fixed step ceilings. Each
of warm-up, frozen-backbone fusion, and joint fine-tuning runs for one epoch.
Every stage evaluates and saves at its epoch end, and the cosine scheduler uses
a 5% warm-up ratio calculated from that stage's resolved optimizer-update count.
For the joint stage, one epoch contains the sampler's configured number of draws;
with `samples_per_epoch: null`, that equals the concatenated training-row count.

Start the new run with:

```bash
uv run python -m ml.fusion.train_fusion \
  --config configs/speech_enhancement/fusion_train_compact.yaml
```

The run writes only to `models/asr/fusion/run_005_compact`; the reported v1/v2
configs and their output directories remain unchanged. The trainer resumes an
interrupted stage from that new directory when the same command is run again.
