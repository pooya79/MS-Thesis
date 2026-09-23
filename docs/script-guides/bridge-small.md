# Whisper Small bridge training and evaluation

This experiment trains Cui et al.'s perceptual-quality-plus-recognition-information
(PQ+RI) waveform bridge with a frozen Persian Whisper Small ASR checkpoint and a
frozen FRCRN enhancer. Only the bridge parameters change. The seven training
and five test dataset names came from the archived Whisper Small configurations
in `report/`; they are now explicit in the two experiment configs:

- `configs/speech_enhancement/bridge_small_train.yaml` lists train/dev datasets,
  the frozen ASR checkpoint, preparation and cache settings, and bridge training
  settings.
- `configs/speech_enhancement/bridge_small_eval.yaml` lists held-out test
  datasets, model paths, decoding settings, and the evaluation output directory.

Clean train/dev audio supplies the original view directly. For degraded datasets,
the mapped degraded audio supplies that view; clean references are never fed to
the bridge. Frozen FRCRN creates the enhanced view of each utterance. DNSMOS and
11-coefficient WER profiles are computed for train/dev only. Test references are
used only for final scoring.

Run from the repository root on the machine containing the datasets and the
configured Whisper Small checkpoint:

```bash
uv run python -m ml.fusion.train_bridge_small --help
uv run python -m ml.fusion.eval_bridge_small --help

# Check all paths, audio, and selected split counts before loading models.
uv run python -m ml.fusion.train_bridge_small prepare --dry-run

# Complete each stage before starting the next one.
uv run python -m ml.fusion.train_bridge_small prepare
uv run python -m ml.fusion.train_bridge_small cache
uv run python -m ml.fusion.train_bridge_small train
uv run python -m ml.fusion.eval_bridge_small
```

`prepare` enhances the configured train/dev and test audio and scores train/dev
original audio with DNSMOS. It writes resumable WAVs, bridge manifests,
provenance, and a generated `final_tests.yaml` under
`artifacts/bridge-small/inputs`. Rerun it unchanged after interruption. Invalid
or conflicting rows are listed in `skipped_inputs.jsonl`.

`cache` runs the frozen Whisper Small checkpoint on eleven waveform mixtures per
train/dev utterance (original-audio weights 1.0 through 0.0 in steps of 0.1).
It writes WER targets and filterbanks under `artifacts/bridge-small/cache`.
After interruption, use `cache --resume` with the same inputs and settings.

`train` optimizes `(PQ + RI) / 2` for five epochs with the Tiny pilot's bridge
settings, saving `models/asr/bridge-small/best.pt`. It refuses an existing output
directory; preserve an interrupted run before restarting. The paper reports up
to 45 epochs, so this is a five-epoch experimental budget rather than a claim to
reproduce its training duration.

`eval_bridge_small` loads its own YAML, checks that its test datasets and model
paths match the prepared audio, and scores frozen Whisper Small on original
audio, the learned bridge, and FRCRN-enhanced audio on one common test manifest.
It writes predictions, per-dataset WER/CER, `comparison.md`, and
`comparison.json` under `artifacts/bridge-small/evals/bridge-small-test`. The
comparison files include archived metrics as context. Archived runs may have
filtered different clips, so use the newly scored baseline and bridge for the
paired comparison. Evaluation refuses an existing result directory.

Both CLIs accept `--config` to select another YAML file. The training CLI also
accepts `--device`, `prepare --dry-run`, and `cache --resume`; the evaluation CLI
accepts `--device` and `--output`. A changed training config requires fresh input,
cache, and checkpoint directories: preparation records the config hash and
refuses incompatible resume. Do not commit prepared audio, caches, or weights.
