# ASR playground

Run from the repository root. Initial setup requires `uv sync --locked` and
FFmpeg/FFprobe on `PATH`. Model paths are configured in `configs/server.yaml`.

Set `APP_PASSWORD` in the repository-root `.env`. If that file does not exist,
create it from `.env.example` first:

```dotenv
APP_PASSWORD=replace-this-with-a-long-random-password
```

Start the demo:

```bash
make run
```

Open http://localhost:8001 and enter the password from `.env`. Stop with Ctrl+C.
Use `make help` for target help or `uv run --no-sync uvicorn --help` for server options.

`make run` executes:

```bash
uv run --no-sync --env-file .env uvicorn server.app.main:app --reload --host 0.0.0.0 --port 8001
```

The default host is `0.0.0.0` (string; all interfaces), the port is `8001`
(integer), and reload is enabled. To use localhost only or another port, run
the command directly with `--host 127.0.0.1` or a different `--port`.
`APP_PASSWORD` is a required string; `ASR_SERVER_CONFIG` is an optional path
defaulting to `configs/server.yaml`. `uv --env-file .env` loads these settings
for the server; exporting them manually is unnecessary. Already exported variables
take precedence over `.env`; unset an old exported `APP_PASSWORD` if you want to
use the file's value. The `.env` file is ignored by Git. If starting Uvicorn
without `uv --env-file`, supply the variables in its process environment.

Starting the server skips dependency synchronization. A plain `uv run` can try
to install missing research dependencies, including ClearVoice's `scenedetect`,
before starting Uvicorn. If that download times out, use `make run` or the
`--no-sync` command above with the already prepared environment. If a required
demo package is actually missing, complete `uv sync --locked` when package
downloads are available. `--no-sync` does not install missing packages.

Models load lazily on first use. The configured local checkpoints and processors
do not need Hub downloads. Upload conversion requires FFmpeg; on Ubuntu/Debian,
install it with `sudo apt install ffmpeg`. Models use CUDA when available and
otherwise fall back to CPU with `device: auto`.
