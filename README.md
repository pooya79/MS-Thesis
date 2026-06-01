# MS-Thesis

Research archive and implementation workspace for my MS thesis.

## Requirements

- Python `3.13`
- [`uv`](https://docs.astral.sh/uv/) for Python dependency management
- `ffmpeg` with audio codec support for G.711, GSM, AMR-NB, AMR-WB, and Opus

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg
```

The degradation pipeline checks encoder availability before generation. The current environment is expected to expose these ffmpeg encoders/codecs:

- `pcm_alaw`
- `pcm_mulaw`
- `libgsm` for GSM 06.10 encoding (`gsm` is also accepted if exposed by your ffmpeg build)
- `libopencore_amrnb`
- `libvo_amrwbenc`
- `libopus`

You can inspect your local build with:

```bash
ffmpeg -hide_banner -codecs | grep -Ei 'amr|gsm|opus|pcm_alaw|pcm_mulaw'
```

## Install

Install Python dependencies from `pyproject.toml` and `uv.lock`:

```bash
uv sync
```

The project currently uses packages for the FastAPI app and speech degradation, including `numpy`, `scipy`, `soundfile`, `soxr`, `pyyaml`, and `tqdm`.

## Development Commands

Run the FastAPI app:

```bash
make run
```

Run the full test suite:

```bash
make test
```

Run only the speech degradation tests:

```bash
uv run pytest server/tests/test_degradation_pipeline.py -q
```

For data, training, and inspection script usage, see [docs/script-guide.md](docs/script-guide.md).

## Speech Degradation Pipeline

The degradation pipeline reads clean-audio JSONL manifests and writes paired clean/degraded audio for speech-enhancement training.

Default config:

```text
configs/speech_enhancement/degradation.yaml
```

Expected clean manifest rows:

```json
{"id": "clip-001", "split": "train", "clean_path": "/path/to/audio.wav", "transcript": "optional text"}
```

Generate degraded pairs:

```bash
uv run python -m ml.speech_data.generate_degraded_pairs --config configs/speech_enhancement/degradation.yaml
```

Inspect a generated manifest:

```bash
uv run python -m ml.speech_data.inspect_manifest data/speech_enhancement/manifests/se_train_pairs.jsonl
```

Generated audio, manifests, checkpoints, and reports belong under `data/` or `artifacts/` and are ignored by Git.
