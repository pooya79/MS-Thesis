# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope note

`AGENTS.md` holds the authoritative conventions (project structure, frontend/template/CSS layering, auth rules, test expectations, ASR dataset layout, working rules). Read it. This file adds the big-picture architecture and the commands you'll actually run, without repeating those conventions.

## Commands

- `make run` — start the FastAPI dev server (`uvicorn server.app.main:app --reload` on `:8001`).
- `make test` — full pytest suite (`uv run pytest -q`).
- `uv sync` — install/refresh dependencies from `pyproject.toml` + `uv.lock`.
- Run one test file: `uv run pytest server/tests/test_degradation_pipeline.py -q`
- Run one test: `uv run pytest server/tests/test_health.py::<name> -q`

Everything runs through `uv` (Python 3.13). `torch`/`torchvision` come from the `pytorch-cu130` CUDA index pinned in `pyproject.toml`. `pythonpath = ["."]` is set for pytest, so modules import as `server.*` / `ml.*` from the repo root.

For every data/training/inspection script and its exact CLI flags, see `docs/script-guides/README.md` — it is the canonical index for the topic-based script guides. All maintained scripts expose `--help`; when adding or changing a script, update the relevant guide, the command index, and its `--help` test (this is an enforced rule, see AGENTS.md).

## Architecture

Two largely independent halves share one repo and one `uv` environment:

### 1. `server/` — FastAPI ASR playground
- Entry point `server/app/main.py` serves the model picker and `POST /api/transcriptions`.
- `configs/server.yaml` is the source of truth for available models, checkpoint/processor paths, device choice, and upload limits. Set `ASR_SERVER_CONFIG` to use another config file.
- `server/app/services/transcription.py` lazily loads and caches Whisper, Fusion, and FastConformer backends. Per-model locks prevent concurrent GPU use of the same model.
- Uploaded or browser-recorded audio is bounded, converted by FFmpeg to mono 16 kHz WAV in a temporary directory, and deleted after inference.
- The app has no authentication, database, or migration layer; it is intended to be run as a local research tool.

### 2. `ml/` — reproducible thesis ML pipelines
All run as modules: `uv run python -m ml.<...>`. Determinism is a hard requirement (seeds + augmentation metadata recorded in JSONL manifests).
- `ml/speech_data/` — the speech-degradation / dataset pipeline. `scripts/` = download + prepare per corpus (Common Voice 25 fa, FLEURS fa, Persian eval sets, DEMAND noise). Top-level modules generate degraded pairs / degraded-only datasets and inspect manifests. The data flow is: **download → prepare (normalize transcripts, mono 16 kHz WAV) → generate degraded audio → JSONL manifests**.
- `ml/asr/` — Whisper-small fine-tuning (`train_whisper_small.py`) and evaluation (`eval_whisper_small.py`, reports WER/CER per dataset). Driven entirely by YAML in `configs/`.
- `ml/utils/` — shared `seed.py` and `audio.py`.
- `configs/` — YAML for ASR training/eval (`whisper_small_*.yaml`) and degradation/dataset generation (`speech_enhancement/`).

**ASR dataset contract** (used across `ml/` and tests): a dataset dir has split TSVs (`train.tsv`/`dev.tsv`/`test.tsv`) with at least `path` + `sentence` columns; audio resolves as `<dataset>/clips/<path>` then `<dataset>/<path>`. See AGENTS.md "ASR Dataset Layout".

### Tests
`server/tests/` covers both halves: the ASR web UI/API (model listing, uploads, limits, responses) and ML utilities (script `--help` text, dataset prep, degradation determinism/codec round-trips/manifest fields). Template or style changes must validate both HTML content and asset links.

## Data & artifacts

Generated audio, checkpoints, manifests, and reports live under `data/` or `artifacts/` and are git-ignored — never commit them. `Thesis/` is the LaTeX thesis document; ignore it for development/testing unless explicitly asked.
