# AGENTS

## Project Overview
This repository is my research archive for my MS thesis.

## Project Structure
- `server/`: FastAPI application, templates, database setup, and server-side tests.
- `server/app/`: application package (`core`, `db`, `models`, `routers`, `templates`).
- `server/app/static/`: frontend assets (`css/`, `js/`) served at `/static`.
- `server/tests/`: automated tests for API and page behavior.
- `ml/`: reproducible machine-learning utilities for thesis experiments, including speech degradation and future enhancement/fusion training code.
- `configs/`: configuration files for ML/data-generation workflows.
- `docs/`: implementation notes and experiment plans.
- `Thesis`: thesis document and related research notes are there, unless explicitly mentioned, you don't need to worry about it for development or testing.

## Commands
- Run server: `make run`
- Run tests: `make test`
- Generate degraded speech pairs: `uv run python -m ml.speech_data.generate_degraded_pairs --config configs/speech_enhancement/degradation.yaml`
- Inspect generated pair manifest: `uv run python -m ml.speech_data.inspect_manifest data/speech_enhancement/manifests/se_train_pairs.jsonl`

## Working Rules
- Keep server code minimal, typed, and modular.
- Add or update tests for every behavior change.
- Keep archived findings and blog/demo content reproducible and traceable to experiments.
- Keep ML/data-generation code deterministic where possible; record seeds and augmentation metadata in JSONL manifests.
- Do not commit generated audio, checkpoints, or large experiment artifacts under `data/` or `artifacts/`.
- Use the configured `ffmpeg` codec round-trips for speech degradation unless explicitly changing the experiment design.

## Frontend and Template Conventions
- Use `server/app/templates/base.html` as the global document skeleton.
- Use `server/app/templates/shell.html` for authenticated app pages that need top nav + left sidebar + right sidebar.
- Keep page templates thin: extend base/shell and place page-specific markup in blocks.
- Keep shared styles in tokenized CSS under `server/app/static/css/`.
- CSS layering: `tokens.css` for design variables.
- CSS layering: `base.css` for reset and global primitives.
- CSS layering: `shell.css` for app shell layout and responsive behavior.
- CSS layering: `home.css` for home/dashboard-specific styles.
- Keep shell interaction JS in `server/app/static/js/shell.js` (mobile sidebar toggle and TOC highlighting).

## Auth and Static Rules
- Static assets must remain mounted at `/static` from `server/app/static`.
- Unauthenticated users must be allowed to read `/static/*` with `GET/HEAD`.
- Keep `/login` public for `GET/POST`; all other routes remain protected by password session middleware unless explicitly designed otherwise.

## Test Expectations
- For template/style changes, validate both content and asset links in HTML responses.
- Keep coverage for unauthenticated redirect to `/login`.
- Keep coverage for successful login + authenticated homepage render.
- Keep coverage for static asset public accessibility.
- Keep coverage for protected non-HTML endpoint behavior (e.g. `/health` returns `401` when unauthenticated).
- For speech degradation changes, keep coverage for deterministic seeds, audio shape/range safety, codec round-trips, generated manifest fields, and clean/degraded length alignment.


## Important Notes
- When writing a script make sure to add a --help description and argument types for clarity and reproducibility.