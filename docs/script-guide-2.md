# Script Guide 2

This guide continues `docs/script-guide.md` and covers the Whisper
large-v3-turbo training and evaluation commands.

## Whisper Large-v3-turbo Training

Fine-tune `openai/whisper-large-v3-turbo` for Persian ASR with:

```bash
uv run python -m ml.asr.train_whisper_large_v3_turbo \
  --config configs/whisper_large_v3_turbo_train.yaml \
  --resume auto
```

Each configured dataset must contain `train.tsv`, `dev.tsv`, and `clips/`.
The TSV files need `path` and `sentence` columns. Relative audio paths are
looked up under `<dataset>/clips/` first and then under `<dataset>/`.

The supplied config starts from `openai/whisper-large-v3-turbo`, uses Persian
transcription decoder prompts, and writes the run to
`models/asr/whisper-large-v3-turbo/runs/whisper-large-v3-turbo-fa/`. The run
contains the effective config, source and skipped-example manifests, JSONL
metrics, logs, rolling checkpoints, `status.json`, and the final model.

`training.gradient_checkpointing` is enabled because large-v3-turbo needs much
more GPU memory than Whisper-small. The default device batch size is 1 with 8
gradient-accumulation steps. If memory is still exhausted, keep the device batch
size at 1 and reduce evaluation pressure (for example, evaluate less often or
evaluate separately after training). If memory allows, increase the device batch
size before changing accumulation so the intended effective batch size remains
explicit.

Set `model.pretrained_model` to a local Hugging Face model directory to continue
fine-tuning from saved weights. Leave it empty to use `model.name`. Set
`run.resume` or `--resume` to `auto` to resume the latest rolling Trainer
checkpoint in the same run directory; use `false` to start without resuming, or
pass an explicit checkpoint directory. `--run-dir` overrides the configured run
directory.

Inspect every option without loading the model or data:

```bash
uv run python -m ml.asr.train_whisper_large_v3_turbo --help
```

## Whisper Large-v3-turbo Evaluation

Evaluate the saved model on the configured `test.tsv` files with:

```bash
uv run python -m ml.asr.eval_whisper_large_v3_turbo \
  --config configs/whisper_large_v3_turbo_eval.yaml
```

Set `model.checkpoint` to a local `final`, `best`, or Trainer checkpoint
directory. The default processor is `openai/whisper-large-v3-turbo`. When a
Trainer checkpoint does not contain processor files, keep that Hub processor
setting; when a run intentionally changed its tokenizer or processor, point
`model.processor` to the saved `final` or `best` directory instead.

Evaluation writes an effective config, logs, source and skipped-example
manifests, `predictions.jsonl`, and `metrics.json`. Metrics include aggregate
WER/CER and WER/CER grouped by dataset directory. `data.split` may be changed to
another available TSV split such as `dev`. Transcripts over
`eval.max_label_tokens` are recorded in the skipped manifest and excluded.
Keep `eval.eval_accumulation_steps` low (normally 1) so generated tensors are
moved off the GPU during long evaluations. Use `--output-dir` to override the
configured output location.

Inspect every option without loading the model or data:

```bash
uv run python -m ml.asr.eval_whisper_large_v3_turbo --help
```
