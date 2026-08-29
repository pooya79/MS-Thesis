# Whisper-small pMCT experiment

This directory is an isolated implementation of Patched Multi-Condition
Training (pMCT). It does not modify the existing ASR trainers or their configs.

## Data flow

1. Generate the aligned additive-noise dataset and its clean/degraded mapping:

   ```bash
   uv run python -m ml.speech_data.generate_noise_added_dataset \
     --config configs/speech_enhancement/noise_added_dataset.yaml
   ```

2. List any number of paired and clean datasets in `whisper_small.yaml`:

   ```yaml
   data:
     root_dir: data
     datasets:
       - path: cv-corpus-25.0
         kind: clean
       - path: cv-corpus-25.0-noise-added
         kind: paired
       - path: FarsSpon_train_dev
         kind: clean
   ```

   A paired dataset must contain `degraded_to_clean.jsonl`. Clean datasets use
   the normal `train.tsv` audio unchanged. A plain string entry is also accepted
   and auto-detected as paired when that mapping file exists, but explicit kinds
   make experiment configs easier to audit.

3. Train:

   ```bash
   uv run python -m ml.pmct.train_whisper_small \
     --config ml/pmct/whisper_small.yaml
   ```

No patched audio is written to disk. For each training example, the loader
reads its aligned clean and noisy waveforms and chooses every patch from clean
audio with probability `clean_probability` (the paper's pi). The patch mask is
reproducible from the global seed and changes at each epoch. Development audio
is loaded normally and is never patch-mixed.

Before training, every paired mapping is checked one-for-one against its split
TSV, including unique paths and matching transcripts. Clean audio is rebuilt
with the same SoXR resampler used by the generator, the recorded shared peak
scale is applied, and a possible one-sample rounding difference is aligned to
the degraded waveform. Larger length differences are rejected.

The current upstream generator is noise-only, so this experiment implements
the additive-noise form of pMCT. Reproducing the paper's reverberant condition
also requires aligned RIR simulation before patching.
