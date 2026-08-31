# AGENTS

## Project Overview
This repository is my research archive for my MS thesis.

## Project Structure
- `server/`: FastAPI ASR playground, model adapters, templates, static assets, and tests.
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

## ASR Dataset Layout
- Each dataset directory should contain split TSV files such as `train.tsv`, `dev.tsv`, and optionally `test.tsv`, plus a `clips/` directory containing the referenced audio.
- Split TSV files must include at least `path` and `sentence` columns.
- Audio paths in TSV rows are resolved first as `<dataset>/clips/<path>`, then as `<dataset>/<path>`.

## Frontend and Template Conventions
- Keep the single-page ASR UI in `server/app/templates/index.html`.
- Keep its styles in `server/app/static/css/app.css` and browser recording/upload behavior in `server/app/static/js/app.js`.
- Model choices and checkpoint paths belong in `configs/server.yaml`, not in templates or JavaScript.
- Keep templates thin and keep model inference behind `server/app/services/transcription.py`.

## Diagram Conventions
- Use `docs/diagram-style-guide.md` as the visual design system for thesis and research diagrams.
- Prefer one self-contained HTML file with inline SVG and CSS for each new diagram.
- Keep source diagrams intact when producing redesigns; add the new artifact beside the original.

## Server and Static Rules
- Static assets must remain mounted at `/static` from `server/app/static`.
- The server has no account or database layer; access uses the shared `APP_PASSWORD` environment variable and a signed HTTP-only session cookie.
- Keep `/login` and `/static/*` public; all application pages and APIs must require authentication.
- Audio uploads must be bounded, decoded into a temporary mono WAV, and removed after each request.
- Models must load lazily and remain cached across requests.

## Test Expectations
- For template/style changes, validate both content and asset links in HTML responses.
- Keep coverage for model metadata, upload validation, API transcription results, and duration limits.
- For speech degradation changes, keep coverage for deterministic seeds, audio shape/range safety, codec round-trips, generated manifest fields, and clean/degraded length alignment.


## Important Notes
- When writing or changing a script, expose clear `--help` text with argument types/defaults, add or update its command and workflow documentation under `docs/script-guides/`, and keep CLI help covered by tests.
