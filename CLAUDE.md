# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope note

`AGENTS.md` holds the authoritative conventions (project structure, frontend/template/CSS layering, auth rules, test expectations, ASR dataset layout, working rules). Read it. This file adds the big-picture architecture and the commands you'll actually run, without repeating those conventions.

## Commands

- `make run` — start the FastAPI dev server (`uvicorn server.app.main:app --reload` on `:8001`).
- `make test` — full pytest suite (`uv run pytest -q`).
- `make migrate-up` / `make migrate-down` — apply / revert one Alembic migration. DB is SQLite at `server/data/app.db`.
- `uv sync` — install/refresh dependencies from `pyproject.toml` + `uv.lock`.
- Run one test file: `uv run pytest server/tests/test_degradation_pipeline.py -q`
- Run one test: `uv run pytest server/tests/test_health.py::<name> -q`

Everything runs through `uv` (Python 3.13). `torch`/`torchvision` come from the `pytorch-cu130` CUDA index pinned in `pyproject.toml`. `pythonpath = ["."]` is set for pytest, so modules import as `server.*` / `ml.*` from the repo root.

For every data/training/inspection script and its exact CLI flags, see `docs/script-guide.md` — it is the canonical, test-covered list. All maintained scripts expose `--help`; when adding or changing a script, update both `docs/script-guide.md` and its `--help` test (this is an enforced rule, see AGENTS.md).

## Architecture

Two largely independent halves share one repo and one `uv` environment:

### 1. `server/` — FastAPI web app
- Entry point `server/app/main.py` wires three layers of middleware in order: `PasswordAuthMiddleware` → `SessionMiddleware` → static mount + `web` router.
- **Auth is global-by-default**: `PasswordAuthMiddleware` (`server/app/core/auth.py`) protects every route. Only `is_public_path` exceptions pass through unauthenticated — `GET/POST /login` and `GET/HEAD /static/*`. Single shared password from `APP_PASSWORD` (`secrets.compare_digest`), session-cookie backed. HTML requests get a redirect to `/login?next=`; non-HTML requests get `401`. Don't add public routes without going through `is_public_path`.
- Settings come from `server/app/core/config.py` (`pydantic-settings`, `.env`-backed, `lru_cache`d `get_settings()`). `APP_PASSWORD` and `APP_AUTH_SECRET` are required env vars.
- Persistence: SQLModel + SQLite, sessions via `server/app/db/session.py`, models under `server/app/models/`, migrations via Alembic (`alembic.ini`, `sqlite:///server/data/app.db`).
- Templates/CSS/JS follow a strict layering convention — see AGENTS.md before touching `templates/` or `static/css/`.

### 2. `ml/` — reproducible thesis ML pipelines
All run as modules: `uv run python -m ml.<...>`. Determinism is a hard requirement (seeds + augmentation metadata recorded in JSONL manifests).
- `ml/speech_data/` — the speech-degradation / dataset pipeline. `scripts/` = download + prepare per corpus (Common Voice 25 fa, FLEURS fa, Persian eval sets, DEMAND noise). Top-level modules generate degraded pairs / degraded-only datasets and inspect manifests. The data flow is: **download → prepare (normalize transcripts, mono 16 kHz WAV) → generate degraded audio → JSONL manifests**.
- `ml/asr/` — Whisper-small fine-tuning (`train_whisper_small.py`) and evaluation (`eval_whisper_small.py`, reports WER/CER per dataset). Driven entirely by YAML in `configs/`.
- `ml/utils/` — shared `seed.py` and `audio.py`.
- `configs/` — YAML for ASR training/eval (`whisper_small_*.yaml`) and degradation/dataset generation (`speech_enhancement/`).

**ASR dataset contract** (used across `ml/` and tests): a dataset dir has split TSVs (`train.tsv`/`dev.tsv`/`test.tsv`) with at least `path` + `sentence` columns; audio resolves as `<dataset>/clips/<path>` then `<dataset>/<path>`. See AGENTS.md "ASR Dataset Layout".

### Tests
`server/tests/` covers both halves: web (auth redirect, login flow, static accessibility, protected `/health`) and ML (script `--help` text, dataset prep, degradation determinism/codec round-trips/manifest fields). Template or style changes must validate both HTML content and asset links.

## Data & artifacts

Generated audio, checkpoints, manifests, and reports live under `data/` or `artifacts/` and are git-ignored — never commit them. `Thesis/` is the LaTeX thesis document; ignore it for development/testing unless explicitly asked.
