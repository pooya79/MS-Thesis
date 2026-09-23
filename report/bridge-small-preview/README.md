# Preview of Whisper Small bridge evaluation output

**Synthetic preview only. No model was trained or evaluated to produce these scores.**
The five `PREVIEW_ONLY` rows and their hypotheses are invented solely to show the
exact filenames and JSON/JSONL structure produced by `eval_bridge_small.py`.
The archived rows in `comparison.md` and `comparison.json` are copied from the
existing `report/*/metrics.json` files; the `current/*` rows are synthetic.

A real server run writes the same file types to the configured directory:
`artifacts/bridge-small/evals/bridge-small-test/`. It uses every valid clip in
all five test datasets, so its example counts, hashes, predictions, WER, CER, and
progress timings will differ. The actual scores cannot be known before training
and evaluation finish.

## Files

- `config.yaml`: effective evaluation suite and model paths.
- `test_manifest.jsonl`: the common input/reference rows for all methods.
- `skipped_inputs.jsonl`: rejected clips and reasons (empty in this preview).
- `{baseline,bridge,enhanced_only}.predictions.jsonl`: one hypothesis per clip.
- `{baseline,bridge,enhanced_only}.metrics.json`: aggregate and per-dataset WER/CER.
- `summary.json`: all three metric objects keyed by method.
- `progress.json`: completion state and timing.
- `comparison.md` and `comparison.json`: current scores alongside archived metrics.
