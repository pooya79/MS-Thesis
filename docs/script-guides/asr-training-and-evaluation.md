# ASR Training and Evaluation

## Whisper-small Training

Fine-tune Whisper-small from the training config. Outputs go under the configured run directory unless `--run-dir` overrides it:

```bash
uv run python -m ml.asr.train_whisper_small \
  --config configs/whisper_small_train.yaml \
  --resume auto
```

Set `model.pretrained_model` to start from an existing local model directory, such as a previous run's `final` or `best` directory. Leave it empty to start from `model.name`, which defaults to `openai/whisper-small`.

Each configured dataset contributes `train.tsv` to training and `dev.tsv` to
evaluation when those splits are present. A missing split is skipped for that
dataset, and `test.tsv` is not used during training. Across the configured
datasets, at least one must provide usable training rows and at least one must
provide usable development rows.

Audio files with unreadable headers are skipped before training and recorded in
`manifests/skipped_unreadable_train.jsonl` or
`manifests/skipped_unreadable_dev.jsonl`. If an audio file passes that header
check but fails during full decoding, its path is logged and the loader
substitutes the next readable example instead of stopping the run.

## Whisper-small Evaluation

Run a saved Whisper-small checkpoint on the configured dataset `test.tsv` files. Outputs include `metrics.json`, `predictions.jsonl`, the effective config, logs, and a source manifest. `metrics.json` reports aggregate WER/CER and a `dataset_metrics` list with WER/CER per dataset directory:

```bash
uv run python -m ml.asr.eval_whisper_small \
  --config configs/whisper_small_eval.yaml
```

Set `model.checkpoint` to the local model/checkpoint path to evaluate. `model.processor` defaults to `openai/whisper-small`; point it at a saved `final`/`best` model directory only if you intentionally changed processor/tokenizer files. Set `data.datasets` to the dataset directories whose `test.tsv` files should be evaluated. Samples whose transcript token count exceeds `eval.max_label_tokens` are skipped before prediction; by default this should match Whisper-small's 448-token decoder limit. Keep `eval.eval_accumulation_steps` low, such as `1`, so generated prediction tensors are moved off GPU during long evaluations instead of accumulating until the end.

## FastConformer-CTC Training

Fine-tune the standalone FastConformer-CTC Persian model (the CTC branch of `nvidia/stt_fa_fastconformer_hybrid_large`, reimplemented under `ml/fa_fastconformer/` with no NeMo dependency) on the configured dataset `train.tsv` / `dev.tsv` files. Because the standalone model is a plain `nn.Module` rather than a Hugging Face model, training runs through a small hand-written PyTorch loop (CTC loss, AdamW, linear warmup schedule, gradient accumulation, optional AMP) instead of `transformers.Trainer`. The run layout mirrors the Whisper trainer — `status.json`, `logs/train.log`, `logs/train_metrics.jsonl`, the effective config, source manifests, rolling `checkpoints/checkpoint-<step>.pt` bundles, plus `final.pt` and `best.pt`:

```bash
uv run python -m ml.asr.train_fastconformer \
  --config configs/fastconformer_train.yaml \
  --resume auto
```

Set `model.checkpoint` to either the original `.nemo` archive or a converted `.pt` bundle to fine-tune from — the format is chosen by file extension (use `ml/fa_fastconformer/convert.py` to produce the `.pt` bundle; see the evaluation section below). Every checkpoint and the `final`/`best` models are written as the same self-contained `.pt` bundle that `eval_fastconformer` loads, so a trained checkpoint can be evaluated directly by pointing `fastconformer_eval.yaml`'s `model.checkpoint` at it. Resume state (optimizer, scheduler, AMP scaler, step) is stashed inside each rolling checkpoint bundle, so `--resume auto` (or `run.resume: auto`) continues from the latest one. Set `training.freeze_encoder: true` to train only the CTC head. Stop with Ctrl+C after a checkpoint exists, then re-run with `--resume auto` to continue.

Clips outside `data.min_duration_sec` / `data.max_duration_sec` (default `0.1`–`20.0`) are dropped from both the train and dev splits before batching — durations come from the audio header only (no decoding). Conformer self-attention costs O(T²) memory per layer, so without an upper cap a single multi-minute utterance (common in spontaneous-speech corpora) can OOM the GPU even when typical fixed-size batches fit comfortably. Raise `data.max_duration_sec` to keep longer clips (watch GPU memory), or set it to `null` to disable the cap. The trainer also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (unless already set) to reduce allocator fragmentation across the variable-length batches.

## Mixed-Dataset Local ASR Comparison

Evaluate several local model families in one run against a dataset created by
`ml.speech_data.scripts.create_mixed_test_dataset`:

```bash
uv run python -m ml.asr.eval_mixed_dataset \
  --config configs/mixed_asr_eval.yaml
```

The top-level config contains `dataset_root`, `output_dir`, and a repeatable
`models` list. Every model entry has a unique filesystem-safe `name`, a `type`,
and the path to that model's normal evaluator config:

```yaml
dataset_root: data/mixed-persian-test
output_dir: artifacts/asr-mixed-eval/mixed-persian-test
models:
  - name: whisper-small
    type: whisper_small
    config: configs/whisper_small_eval.yaml
  - name: whisper-medium
    type: whisper_medium
    config: configs/whisper_medium_eval.yaml
  - name: fastconformer
    type: fastconformer
    config: configs/fastconformer_eval.yaml
  - name: fusion
    type: fusion
    config: configs/speech_enhancement/fusion_eval.yaml
```

Supported types are `whisper_small`, `whisper_medium`,
`whisper_large_v3_turbo`, `fastconformer`, and `fusion` (hyphenated aliases are
also accepted). The referenced model config remains the source of checkpoint,
processor, device, batch-size, and generation settings; only its `data.root_dir`,
`data.datasets`, and `data.split` fields are adapted to the mixed test set. This
means checkpoints do not need to be copied and each underlying evaluator still
writes its usual logs, effective config, manifest, metrics, and predictions.
After every model finishes (including a failed evaluation), the runner collects
unreachable Python objects and clears PyTorch's CUDA cache and IPC allocations
before loading the next model.

The runner joins predictions back to the mixed manifest by resolved audio path,
not by row position. Repeated rows for the same audio are supported when their
source and reference agree; conflicting labels for one audio are rejected as
ambiguous. It adds `source_dataset`, the original and normalized reference, and
the normalized hypothesis to every model's `predictions.jsonl`.
For fair cross-model comparison, the new scores consistently use the repository's
Persian ASR normalization. `<output_dir>/summary.json` and `summary.tsv` contain
overall and per-source-dataset WER/CER for every model. Each model directory also
gets `source_metrics.json`, and its original `metrics.json` gains a
`source_dataset_metrics` block without losing the evaluator-specific metrics.
The command returns a nonzero status if an underlying evaluator did not produce a
prediction for every mixed-dataset row (for example, because a Whisper label was
over its configured token limit).

## FastConformer-CTC Evaluation

Evaluate the standalone FastConformer-CTC Persian model (the CTC branch of `nvidia/stt_fa_fastconformer_hybrid_large`, reimplemented under `ml/fa_fastconformer/` with no NeMo dependency) on the configured dataset `test.tsv` files. Outputs match the Whisper eval layout: `metrics.json` (aggregate WER/CER plus a `dataset_metrics` list per dataset directory), `predictions.jsonl`, the effective config, logs, and a source manifest:

```bash
uv run python -m ml.asr.eval_fastconformer \
  --config configs/fastconformer_eval.yaml
```

Set `model.checkpoint` to either the original `.nemo` archive or a converted `.pt` bundle — the format is chosen by file extension. To produce the `.pt` bundle (CTC weights + config + tokenizer, repacked from the `.nemo` so loading needs neither a tar unpack nor NeMo), run the standalone converter from inside the package directory:

```bash
cd ml/fa_fastconformer
python convert.py /path/to/stt_fa_fastconformer_hybrid_large.nemo models/stt_fa_fastconformer_ctc.pt --verify
```

Greedy CTC decoding has no decoder token limit, so there is no `max_label_tokens` skipping. Batching is duration-aware: clips are sorted by length and each batch is capped by both `eval.batch_size` and `eval.max_batch_seconds`, so the heaviest batch costs about one clip of that many seconds and a few long clips cannot exhaust GPU memory. Raise `eval.batch_size` to speed up short-clip throughput; lower `eval.max_batch_seconds` if you still hit out-of-memory on long clips (set it to `null` to disable the cap and use fixed-size batches).


## Whisper Large-v3-turbo Training

Fine-tune `openai/whisper-large-v3-turbo` for Persian ASR with:

```bash
uv run python -m ml.asr.train_whisper_large_v3_turbo \
  --config configs/whisper_large_v3_turbo_train.yaml \
  --resume auto
```

Each configured dataset must contain `clips/`. Its `train.tsv` and `dev.tsv`
are loaded independently when present, so a missing split does not reject that
dataset. `test.tsv` is not used during training. Across all configured
datasets, at least one must provide usable training rows and at least one must
provide usable development rows.
The TSV files need `path` and `sentence` columns. Relative audio paths are
looked up under `<dataset>/clips/` first and then under `<dataset>/`.

The supplied config starts from `openai/whisper-large-v3-turbo`, uses Persian
transcription decoder prompts, and writes the run to
`models/asr/whisper-large-v3-turbo/runs/whisper-large-v3-turbo-fa/`. The run
contains the effective config, source and skipped-example manifests, JSONL
metrics, logs, rolling checkpoints, `status.json`, and the final model.
Audio files with unreadable headers are excluded before training and recorded
in `manifests/skipped_unreadable_train.jsonl` or
`manifests/skipped_unreadable_dev.jsonl`. If decoding still fails at batch-load
time, the loader logs the path and substitutes the next readable example rather
than terminating the run.

`training.gradient_checkpointing` is enabled because large-v3-turbo needs much
more GPU memory than Whisper-small. The default device batch size is 1 with 8
gradient-accumulation steps. If memory is still exhausted, keep the device batch
size at 1 and reduce evaluation pressure (for example, evaluate less often or
evaluate separately after training). If memory allows, increase the device batch
size before changing accumulation so the intended effective batch size remains
explicit.

Trainable model parameters are normalized to FP32 when loaded, including from a
local FP16 checkpoint. `training.mixed_precision` only controls Trainer
autocasting; it does not change the stored parameter dtype.

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

## Whisper Medium Training

Fine-tune `openai/whisper-medium` for Persian ASR with:

```bash
uv run python -m ml.asr.train_whisper_medium \
  --config configs/whisper_medium_train.yaml \
  --resume auto
```

The dataset layout, generated artifacts, metrics, and resume behavior match the
large-v3-turbo workflow described above. The supplied config writes the run to
`models/asr/whisper-medium/runs/whisper-medium-fa/` and uses Persian
transcription decoder prompts.

The conservative defaults use a device batch size of 1, 8 gradient-accumulation
steps, and gradient checkpointing. If GPU memory allows, increase the device
batch size before reducing accumulation so the effective batch size remains
explicit. Set `model.pretrained_model` to a local Hugging Face model directory
to continue from saved weights. `--run-dir` and `--resume` override their YAML
counterparts.

Inspect every option without loading the model or data:

```bash
uv run python -m ml.asr.train_whisper_medium --help
```

## Whisper Medium Evaluation

Evaluate the saved model on the configured `test.tsv` files with:

```bash
uv run python -m ml.asr.eval_whisper_medium \
  --config configs/whisper_medium_eval.yaml
```

The default checkpoint is the training run's `final` directory and the default
processor is `openai/whisper-medium`. Keep the Hub processor when evaluating a
Trainer checkpoint without processor files; point `model.processor` to a saved
model directory when its tokenizer or processor was changed intentionally.
Evaluation writes the same manifests, predictions, aggregate and per-dataset
WER/CER metrics, effective config, and logs as the large-v3-turbo command. Use
`--output-dir` to override the configured destination.

Inspect every option without loading the model or data:

```bash
uv run python -m ml.asr.eval_whisper_medium --help
```
