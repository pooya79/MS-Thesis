# Script Guide

Every maintained Python script exposes `--help`. Use the help output before running a script with custom paths:

```bash
uv run python -m ml.speech_data.scripts.download_common_voice_fa --help
uv run python -m ml.speech_data.scripts.download_degradation_assets --help
uv run python -m ml.speech_data.scripts.download_fleurs_persian --help
uv run python -m ml.speech_data.scripts.download_persian_eval_sets --help
uv run python -m ml.speech_data.scripts.prepare_common_voice_25 --help
uv run python -m ml.speech_data.scripts.prepare_degradation_assets --help
uv run python -m ml.speech_data.scripts.prepare_fleurs_persian --help
uv run python -m ml.speech_data.scripts.prepare_persian_eval_sets --help
uv run python -m ml.speech_data.scripts.generate_random_degraded_clip --help
uv run python -m ml.speech_data.generate_degraded_dataset --help
uv run python -m ml.speech_data.generate_degraded_pairs --help
uv run python -m ml.speech_data.inspect_manifest --help
uv run python -m ml.asr.train_whisper_small --help
uv run python -m ml.asr.eval_whisper_small --help
uv run python -m ml.asr.train_fastconformer --help
uv run python -m ml.asr.eval_fastconformer --help
uv run python -m ml.fusion.train_fusion --help
uv run python -m ml.fusion.eval_fusion --help
```

## Common Voice Persian Download

Download the Mozilla Data Collective Common Voice Persian archive. Set `MOZILLA_DATA_COLLECTIVE_API_KEY` in the environment or `.env` first:

```bash
uv run python -m ml.speech_data.scripts.download_common_voice_fa \
  --output-dir data
```

## FLEURS Persian Download

Download and export the Persian FLEURS subset from Hugging Face:

```bash
uv run python -m ml.speech_data.scripts.download_fleurs_persian \
  --output-root data/fleurs/fa_ir/source
```

## Persian Evaluation Set Download

Download Nawar Halabi's Persian Speech Corpus and the free `myaudio_tiny`
PersianSpeech release into a local cache. This step only downloads and validates
the upstream archive/metadata files; preparation into TSVs and WAV clips is a
separate step.

```bash
uv run python -m ml.speech_data.scripts.download_persian_eval_sets \
  --cache-dir data/downloads/persian_eval_sets
```

The script caches the upstream archives under `data/downloads/persian_eval_sets/`.
Use `--force` to redownload valid cached files. The default URLs point to the
public Persian Speech Corpus package, the public Google Drive `myaudio_tiny.tar.gz`
archive, and the PersianSpeech GitHub XLSX metadata file.

## Persian Evaluation Set Preparation

Extract the downloaded Persian evaluation archives and prepare both sources as
repo-style ASR test datasets with `test.tsv` and mono 16 kHz WAV clips under
`clips/`:

```bash
uv run python -m ml.speech_data.scripts.prepare_persian_eval_sets \
  --cache-dir data/downloads/persian_eval_sets \
  --source-root data/persian_eval_sets/source \
  --persian-speech-corpus-output-root data/persian-speech-corpus-test \
  --persian-speech-output-root data/PersianSpeech_test \
  --workers 4
```

The script parses `orthographic-transcript.txt` from Persian Speech Corpus and
the `audio`/`text` columns from PersianSpeech `myaudio_tiny.xlsx`. It normalizes
transcripts with the same Persian text rules as the other ASR preparation
scripts, but keeps rejected rows with raw text because these are test/evaluation
sets. Transcript rows whose referenced audio is absent from the downloaded
archive are skipped and reported as `missing audio rows` in the summary. Use
`--force` to replace prepared outputs and re-extract the source archives.

## Common Voice Preparation

Prepare Common Voice 25 Persian into normalized TSV files and mono 16 kHz WAV clips:

```bash
uv run python -m ml.speech_data.scripts.prepare_common_voice_25 \
  --source-root data/cv-corpus-25.0-2026-03-09/fa \
  --output-root data/cv-corpus-25.0 \
  --workers 4
```

## FLEURS Preparation

Prepare exported FLEURS Persian into normalized TSV files and mono 16 kHz WAV clips:

```bash
uv run python -m ml.speech_data.scripts.prepare_fleurs_persian \
  --source-root data/fleurs/fa_ir/source \
  --output-root data/fleurs/fa_ir/normalized \
  --workers 4
```

## TSV Dataset Transcript Normalization

Copy an existing ASR dataset to a new directory and normalize the `sentence`
column in `train.tsv`, `dev.tsv`, and `test.tsv` with the same Persian text
rules used by the Common Voice 25 preparation script, including Unicode
punctuation removal:

```bash
uv run python -m ml.speech_data.scripts.normalize_tsv_dataset \
  --source-root data/my_dataset/raw \
  --output-root data/my_dataset/normalized
```

The output directory must be new unless `--overwrite` is passed. The script
copies the full source tree first, preserves TSV columns, rewrites normalized
transcriptions in place under the output directory, and discards rows whose
sentences are rejected by the Common Voice 25 normalization rules. By default it
normalizes whichever of `train.tsv`, `dev.tsv`, and `test.tsv` exist, so
test-only evaluation directories are supported.

## Long-Audio Variant Concatenation

Build a new ASR dataset of long utterances by concatenating short clips, to
correct the short-utterance length/emission prior that degrades FastConformer on
audio longer than the training clips. Concatenation happens **independently
within each split**, so no train/dev/test leakage is introduced. All parameters
come from a YAML config:

```bash
uv run python -m ml.speech_data.concatenate_long_variants \
  --config configs/long_variants.yaml
```

`variants_per_split` in the config is a mapping of split name to count, so each
split gets a **different number** of variants and only the listed splits are
processed:

```yaml
variants_per_split:
  train.tsv: 3000
  dev.tsv: 300
  test.tsv: 300
```

Each variant joins `min_clips`–`max_clips` short clips (until `target_min_sec`
is reached, capped by `max_duration_sec`), loudness-normalizes every segment,
inserts a `gap_sec` silence between them, and joins the transcripts. Clips are
drawn across speakers by default; set `speaker_column: client_id` to force
same-speaker joins. Generation is deterministic per `seed`, with full provenance
written to `long_variants_manifest.jsonl` and a `generation_report.json`
summary. Pass `--overwrite` to write into an existing output directory. The
output is a long-only dataset: combine or oversample it alongside the original
short dataset at train time rather than using it as a replacement.

Generation is parallelizable: set `workers` in the config (or `--workers N` to
override) to fan variant generation across processes. Output is byte-identical
regardless of worker count, since each variant is seeded independently by its
index. A per-split progress bar reports throughput.

## Degradation Asset Download

Download all DEMAND `*_16k.zip` noise archives:

```bash
uv run python -m ml.speech_data.scripts.download_degradation_assets \
  --noise-root data/speech_enhancement/assets/noise/DEMAND
```

To download, extract, validate, and write indexes in one step:

```bash
uv run python -m ml.speech_data.scripts.download_degradation_assets \
  --noise-root data/speech_enhancement/assets/noise/DEMAND \
  --manifest-dir data/speech_enhancement/manifests \
  --prepare-indexes
```

## Degradation Asset Preparation

Prepare DEMAND 16 kHz noise assets after downloading the archives. Place the DEMAND
`*_16k.zip` files under `data/speech_enhancement/assets/noise/DEMAND/`, then run:

```bash
uv run python -m ml.speech_data.scripts.prepare_degradation_assets \
  --noise-root data/speech_enhancement/assets/noise/DEMAND \
  --manifest-dir data/speech_enhancement/manifests
```

The script extracts local archives by default, validates readable audio, and writes:

```text
data/speech_enhancement/manifests/demand_noise_index.jsonl
```

## Speech Degradation Generation

Generate paired clean/degraded speech-enhancement data from a YAML config:

```bash
uv run python -m ml.speech_data.generate_degraded_pairs \
  --config configs/speech_enhancement/degradation.yaml
```

See `docs/speech-degradation-pipeline.md` for the full degradation chain, profile
semantics, metadata fields, and known limitations.

## Degraded-only ASR Dataset Generation

Generate a dataset-shaped directory with degraded-only clips and TSVs from an existing
TSV-based ASR dataset such as Common Voice 25:

```bash
uv run python -m ml.speech_data.generate_degraded_dataset \
  --config configs/speech_enhancement/cv25_degraded_dataset.yaml \
  --workers 4
```

The config selects the source dataset directory, output dataset directory, included
split TSVs, variations per sample, and worker count. `--workers` overrides
`dataset.workers` for the current run. The output keeps `train.tsv`, `dev.tsv`,
`test.tsv`, or any selected TSV names, writes degraded WAV files under `clips/`, and
records clean-to-degraded traceability in `degraded_to_clean.jsonl`. Full per-variant
degradation metadata is also written to `degradation_metadata.jsonl`.

## Random Degraded Clip Demo

Generate several degraded variants of one random readable audio clip found under `data/`.
The output folder contains the selected clean target, degraded WAV files, a JSONL manifest,
and a JSON report:

```bash
uv run python -m ml.speech_data.scripts.generate_random_degraded_clip \
  --input-root data \
  --output-dir data/speech_enhancement/random_clip_degradations \
  --variants 8 \
  --seed 1337
```

## Manifest Inspection

Inspect a generated speech-enhancement manifest:

```bash
uv run python -m ml.speech_data.inspect_manifest \
  data/speech_enhancement/manifests/se_train_pairs.jsonl
```

## Whisper-small Training

Fine-tune Whisper-small from the training config. Outputs go under the configured run directory unless `--run-dir` overrides it:

```bash
uv run python -m ml.asr.train_whisper_small \
  --config configs/whisper_small_train.yaml \
  --resume auto
```

Set `model.pretrained_model` to start from an existing local model directory, such as a previous run's `final` or `best` directory. Leave it empty to start from `model.name`, which defaults to `openai/whisper-small`.

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

## Enhancement + Fusion Curriculum Training

Run the 3-stage enhancement+fusion curriculum (Stage 0 enhancer warm-up → Stage 1 enhancer+fusion with Whisper frozen → Stage 2 joint end-to-end) from a single YAML config, writing every artifact to one run directory:

```bash
uv run python -m ml.fusion.train_fusion \
  --config configs/speech_enhancement/fusion_train.yaml \
  --resume-from-stage 0
```

The trainer consumes one or more datasets listed under `datasets` (each entry is a path string with its kind auto-detected, or a `{path, kind}` mapping with `kind` in `degraded`/`clean`; the legacy single `dataset_dir` is still accepted as one degraded dataset when `datasets` is null). A **degraded** dataset is a `ml.speech_data.generate_degraded_dataset` directory (`degraded_to_clean.jsonl`): each degraded clip becomes a noisy Whisper log-Mel and the bandwidth-aligned clean log-Mel target is reconstructed from the recorded degradation metadata (`clean_target: bandwidth_aligned`, or `full_band` to target the raw clean). At least one degraded dataset is required, and degraded datasets drive every stage. A **clean** dataset is a plain (non-degraded) ASR dataset following the project split-TSV + `clips/` contract; it is folded into the **joint stage only**, where the noisy and clean views are the same clean log-Mel (`L_enh` → 0) so its undegraded audio fine-tunes the full stack and keeps it strong on clean speech without regressing the degraded result. Dev WER/CER is measured on the degraded datasets' `valid_split` (the metric the fusion system must beat and selects `best.pt` on); the joint stage additionally evaluates the clean datasets' `valid_split` and logs those as `clean_wer`/`clean_cer`/… so clean-speech regression is visible without driving checkpoint selection. Clean audio (resampled to `sample_rate`) contributes to joint-stage training and joint-stage eval only. The enhancer architecture is selected by the `enhancer` block (default `residual_unet`, a lightweight residual 2D-conv U-Net that starts as the identity). Stage 0 trains the enhancer alone on the log-Mel L1 loss `L_enh`. Stages 1–2 build the encoder-feature-space fusion model (`ml/fusion/model.py`): the noisy and enhanced log-Mels are each run through the shared Whisper encoder and the two hidden-state streams are merged by the fusion block (`fusion` config block, default `cross_attention` — bidirectional cross-attention so the streams exchange context before a gated merge; `gated` is the lightweight element-wise baseline) before the decoder, optimising `L_ASR + lambda * L_enh` — Stage 1 with the backbone frozen, Stage 2 end to end (the backbone is initialised from the fine-tuned Persian Whisper at `base_asr_checkpoint`). Each stage validates on `valid_split` every `eval_every` steps — Stage 0 by dev `L_enh`, Stages 1–2 by dev WER/CER decoded through the fused encoder (capped at `eval_max_batches`, decoder steered by `language`/`task`/`generation_max_length`) — logging dev metrics to `logs/eval_metrics.jsonl` and keeping the best-scoring weights as `best.pt`; eval is skipped automatically when no usable dev split exists. Each stage writes a checkpoint under `checkpoints/stage{0,1,2}_*/` (`enhancer.pt`, plus `fusion_model.pt` for Stages 1–2) and a rolling `last.pt` that carries optimizer/scaler state and the step, so re-invoking resumes an interrupted stage mid-way; `--resume-from-stage` (or `resume_from_stage` in the config; `0`/`1`/`2` or `warmup`/`fusion`/`joint`) restarts the curriculum at a later stage by loading the prior stage's checkpoint, so a Stage 2 crash never forces re-running the earlier stages. Stages 1–2 clip gradients at `grad_clip`; seeding (`transformers.set_seed` + seeded dataloaders) makes runs reproducible.

## Enhancement + Fusion Evaluation

Evaluate a trained dual-view fusion model as a whole ASR system on the configured dataset `test.tsv` files. Each clip's Whisper log-Mel is fed as the *noisy* view through the trained `DualViewFusionModel` (enhancer → shared Whisper encoder → cross-attention fusion → Whisper decoder) and `model.generate` decodes token ids from the fused encoder stream exactly as the trainer's dev eval does. Outputs match the Whisper/FastConformer eval layout: `metrics.json` (aggregate WER/CER plus a `dataset_metrics` list per dataset directory), `predictions.jsonl`, the effective config, logs, and a source manifest:

```bash
uv run python -m ml.fusion.eval_fusion \
  --config configs/speech_enhancement/fusion_eval.yaml
```

Set `model.checkpoint` to a fusion training checkpoint — normally the joint-stage final model `artifacts/…/checkpoints/stage2_joint/fusion_model.pt`, though `best.pt` or a Stage 1 checkpoint also load. The enhancer/fusion architecture is read back from the checkpoint, so it need not be repeated in the config. `model.base_asr_checkpoint` supplies the Whisper backbone *architecture* and `generation_config`. **Whether its weights matter depends on the checkpoint:** Stage 2 (joint) checkpoints carry the **jointly-trained backbone**, so it is loaded straight from the checkpoint and `base_asr_checkpoint`'s weights are overwritten — point `model.checkpoint` at the joint checkpoint and the trained backbone comes with it (keep `base_asr_checkpoint: openai/whisper-small` as a bare architecture skeleton). A Stage 1 rolling `last.pt` is **backbone-free** (its frozen backbone equals the base), so the backbone is taken wholly from `base_asr_checkpoint`; there you must set it to the fine-tuned Persian Whisper run dir used during training, not a vanilla baseline. The script logs which case applied and records `backbone_from_checkpoint` in `metrics.json`, warning when a backbone-free checkpoint falls back to `base_asr_checkpoint`. `model.processor` is the tokenizer used to decode (defaults to `model_name`), and `model.language`/`model.task` steer the decoder prompt on multilingual backbones. Set `data.datasets` to the dataset directories whose `test.tsv` files should be evaluated — a clean (non-degraded) set measures the fused stack as a drop-in ASR model, a degraded dataset's clip dirs measure robustness. Decoding is greedy with a `generation_max_length` token cap; raise `eval.batch_size` to speed up throughput, and `eval.mixed_precision` (`auto`/`true`/`false`) controls CUDA autocast.
