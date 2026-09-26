# CV25 / Whisper Tiny official experiment results

Compact archive copied from the GPU server's `~/MS-Thesis` on 2026-09-26. The
server repository was at commit `5ce9a80672e48eb624f8216edd8a0d5fb6d0c293`,
with no local changes to the experiment configs, `ml/fusion`, or `ml/asr`.
The final evaluation completed on 2026-09-20.

## Final test results

All nine methods evaluated the same 20,563 original test utterances: 10,519
from CV25 and 10,044 from AGFarsdat. No inputs were skipped. WER and CER below
are percentages; lower is better. See [summary.json](final-tests/summary.json)
and the nine corresponding `*.metrics.json` files for full precision.

| Method | CV25 WER | CV25 CER | AGFarsdat WER | AGFarsdat CER |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 74.92 | 38.03 | 76.85 | 35.04 |
| Cross-attention | 57.82 | 30.57 | 63.50 | 27.02 |
| Gated | 62.35 | 32.87 | 65.88 | 28.42 |
| Residual cross-attention | 60.26 | 31.90 | 65.48 | 28.35 |
| Cross-attention, original-view bypass | 64.10 | 33.21 | 69.23 | 30.39 |
| Cross-attention, enhanced-view bypass | 63.81 | 33.05 | 69.69 | 30.73 |
| Bridge, PQ + RI | 74.40 | 37.81 | 76.81 | 34.94 |
| Bridge, PQ only | 74.48 | 37.81 | 76.63 | 34.83 |
| Bridge, enhanced only | 75.90 | 38.79 | 77.67 | 35.53 |

The common test manifest SHA-256 recorded in every method result is
`5997ce573bea7139c216ab48fff2ef35d3f5d67ab8038c777eaf34def3a6ec22`.
The official cohort selection file on the server has SHA-256
`864d8287a08507083aa6105b6176db1dcdf98fb437634e08c67e55d840b21c13`.

## Configurations and provenance

- [Final evaluation config](final-tests/config.yaml) is the exact generated
  config saved in `artifacts/cv25-tiny-official/final-tests-all/config.yaml`.
  It names all nine methods, their checkpoints, and the FRCRN identity.
- [Baseline effective config](configs/baseline.yaml) was saved by training at
  `models/asr/cv25-tiny-official/baseline/config/training.yaml`.
- The [cross-attention](configs/cross_attention/training_config.yaml),
  [gated](configs/gated/training_config.yaml), and
  [residual cross-attention](configs/residual_cross_attention/training_config.yaml)
  directories contain the trainers' saved effective configs, stage budgets,
  manifest hashes, and recorded Git commits.
- [Bridge training metadata](configs/bridge-training.json) was extracted from
  the two `best.pt` checkpoints without copying model weights. It records each
  checkpoint's selected epoch, seed, loss, model config, provenance, and actual
  training settings. Both checkpoints record batch size 256, accumulation 256,
  and 20,012 dev examples for checkpoint selection.
- [Bridge preparation request](provenance/bridge-input-request.json),
  [bridge cache provenance](provenance/bridge-cache.json), and
  [test enhancement provenance](provenance/bridge-test.json) record the input
  preparation and model identities. [Baseline status](provenance/baseline-status.json)
  records training completion and the selected checkpoint.
- `dev/*.metrics.json` contains the four compact bridge dev evaluations from
  step 7 of the [run guide](../../docs/script-guides/cv25-tiny-run-all.md).

The server did not contain `artifacts/cv25-tiny-official/run-all`, so no launcher
snapshot was available. This archive uses the effective configs and checkpoint
metadata saved by the individual runs. Predictions, test manifests, JSONL logs,
audio, and checkpoints were left on the server.
