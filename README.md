# MS Thesis Research Workspace

Research code and experiment archive for Persian automatic speech recognition
(ASR), speech degradation, speech enhancement, and noisy/clean feature fusion.

The useful parts of this repository currently live in the command-line tools
under `ml/`. The FastAPI server is an early experimental interface: most of its
dashboard content is placeholder material, and it should not be treated as a
finished research application.

## What Is Implemented

- Download and prepare Persian Common Voice, FLEURS, and evaluation datasets.
- Normalize Persian transcripts and convert audio to the repository dataset format.
- Generate deterministic telephony and environmental speech degradations.
- Inspect and validate generated clean/degraded pairs.
- Build long-utterance dataset variants.
- Train and evaluate Whisper-small and a standalone FastConformer-CTC model.
- Train and evaluate an enhancement plus dual-view fusion pipeline.
- Diagnose whether a trained enhancer improves over the identity baseline.
- Run a password-protected web demo for generating and comparing degraded speech.

## Repository Layout

```text
configs/        YAML configurations for data generation, training, and evaluation
docs/           Script reference, pipeline notes, and experiment decisions
ml/             Dataset, degradation, ASR, enhancement, and fusion code
server/         Experimental FastAPI UI and automated tests
data/           Local datasets and generated audio (not committed)
artifacts/      Local checkpoints, metrics, and reports (not committed)
Thesis/         Thesis documents and research notes
```

## Requirements and Installation

- Python 3.13
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- `ffmpeg` with the codecs needed by the degradation pipeline
- A CUDA-capable environment for practical model training

Install the Python dependencies:

```bash
uv sync
```

Install `ffmpeg` on Ubuntu or Debian:

```bash
sudo apt update
sudo apt install ffmpeg
```

The complete degradation setup expects these encoders/codecs:

- `pcm_alaw`
- `pcm_mulaw`
- `libgsm` (or `gsm`)
- `libopencore_amrnb`
- `libvo_amrwbenc`
- `libopus`

Check the local `ffmpeg` build with:

```bash
ffmpeg -hide_banner -codecs | grep -Ei 'amr|gsm|opus|pcm_alaw|pcm_mulaw'
```

Some distribution builds do not include the AMR encoders. Scripts check the
required codecs before starting generation and fail with a clear error when a
selected codec is unavailable.

## Script-First Workflow

Every maintained script provides `--help`. The full argument and workflow
reference is in [docs/script-guide.md](docs/script-guide.md).

### 1. Download and Prepare Datasets

Download Common Voice Persian. This requires
`MOZILLA_DATA_COLLECTIVE_API_KEY` in the environment or `.env`:

```bash
uv run python -m ml.speech_data.scripts.download_common_voice_fa \
  --output-dir data
```

Prepare Common Voice as normalized TSV splits with mono 16 kHz WAV clips:

```bash
uv run python -m ml.speech_data.scripts.prepare_common_voice_25 \
  --source-root data/cv-corpus-25.0-2026-03-09/fa \
  --output-root data/cv-corpus-25.0 \
  --workers 4
```

Equivalent download and preparation scripts are available for FLEURS and the
Persian evaluation sets:

```bash
uv run python -m ml.speech_data.scripts.download_fleurs_persian --help
uv run python -m ml.speech_data.scripts.prepare_fleurs_persian --help
uv run python -m ml.speech_data.scripts.download_persian_eval_sets --help
uv run python -m ml.speech_data.scripts.prepare_persian_eval_sets --help
```

Prepared datasets use this layout:

```text
dataset/
├── train.tsv
├── dev.tsv
├── test.tsv
└── clips/
```

TSV files must contain at least `path` and `sentence` columns. A referenced
audio path is resolved as `dataset/clips/<path>` first, then `dataset/<path>`.

To normalize an existing TSV dataset with the same Persian text rules:

```bash
uv run python -m ml.speech_data.scripts.normalize_tsv_dataset \
  --source-root data/my_dataset/raw \
  --output-root data/my_dataset/normalized
```

### 2. Prepare Degradation Assets

Download, extract, validate, and index the DEMAND noise data:

```bash
uv run python -m ml.speech_data.scripts.download_degradation_assets \
  --noise-root data/speech_enhancement/assets/noise/DEMAND \
  --manifest-dir data/speech_enhancement/manifests \
  --prepare-indexes
```

### 3. Generate and Validate Degraded Speech

Generate clean/degraded training pairs:

```bash
uv run python -m ml.speech_data.generate_degraded_pairs \
  --config configs/speech_enhancement/degradation.yaml
```

Generate a complete degraded ASR dataset with split TSV files:

```bash
uv run python -m ml.speech_data.generate_degraded_dataset \
  --config configs/speech_enhancement/cv25_degraded_dataset.yaml \
  --workers 4
```

Inspect a pair manifest:

```bash
uv run python -m ml.speech_data.inspect_manifest \
  data/speech_enhancement/manifests/se_train_pairs.jsonl
```

Validate pair alignment, degradation strength, bandwidth consistency, and
required metadata before training:

```bash
uv run python -m ml.speech_data.validate_degraded_dataset \
  --dataset data/cv-corpus-25.0-degraded-v2 \
  --sample 300 \
  --output-dir artifacts/degraded_validation
```

For a quick audible check, generate several degradation variants from one
random clip under `data/`:

```bash
uv run python -m ml.speech_data.scripts.generate_random_degraded_clip \
  --input-root data \
  --output-dir data/speech_enhancement/random_clip_degradations \
  --variants 8 \
  --seed 1337
```

The degradation design and metadata contract are documented in
[docs/speech-degradation-pipeline.md](docs/speech-degradation-pipeline.md).

### 4. Train and Evaluate ASR Models

Whisper-small:

```bash
uv run python -m ml.asr.train_whisper_small \
  --config configs/whisper_small_train.yaml \
  --resume auto

uv run python -m ml.asr.eval_whisper_small \
  --config configs/whisper_small_eval.yaml
```

Standalone FastConformer-CTC:

```bash
uv run python -m ml.asr.train_fastconformer \
  --config configs/fastconformer_train.yaml \
  --resume auto

uv run python -m ml.asr.eval_fastconformer \
  --config configs/fastconformer_eval.yaml
```

Training and evaluation outputs are written to the run directories configured
in YAML. They include effective configs, logs, metrics, predictions, source
manifests, and checkpoints where applicable.

### 5. Train and Evaluate Enhancement/Fusion

Run the enhancement and dual-view fusion curriculum:

```bash
uv run python -m ml.fusion.train_fusion \
  --config configs/speech_enhancement/fusion_train.yaml \
  --resume-from-stage 0
```

Evaluate the complete system or its noisy/enhanced ablations:

```bash
uv run python -m ml.fusion.eval_fusion \
  --config configs/speech_enhancement/fusion_eval.yaml
```

Measure how much denoising headroom an enhancer captures:

```bash
uv run python -m ml.enhancement.diagnose_enhancement \
  --dataset data/cv-corpus-25.0-degraded-v2 \
  --split dev \
  --enhancer-checkpoint artifacts/.../checkpoints/stage0_warmup/enhancer.pt \
  --output-dir artifacts/enhancement_diagnosis
```

## FastAPI Server

The server is currently a development scaffold, not a finished thesis product.
The home dashboard contains placeholder metrics and inactive links. Its only
meaningful experiment interface is the speech-degradation demo, which reuses
the real degradation pipeline to:

- accept an audio upload;
- apply selected channel, codec, noise, level, and network effects;
- return the input, bandwidth-aligned clean target, and degraded WAV;
- display the generated metadata.

The app also includes password/session middleware, templates, static assets,
and a protected health endpoint. Generated demo files are temporary local
artifacts under `server/data/`.

Configure the server:

```bash
cp .env.example .env
```

At minimum, replace these values in `.env`:

```dotenv
APP_PASSWORD=change-me
APP_AUTH_SECRET=change-this-secret
```

Run the development server:

```bash
make run
```

Open `http://localhost:8001`, sign in, and navigate to
`/experiments/speech-degradation`. Noise generation requires the DEMAND index
created during degradation asset preparation. Codec options depend on the local
`ffmpeg` build.

## Tests

Run the complete test suite:

```bash
make test
```

Run a focused test module:

```bash
uv run pytest server/tests/test_degradation_pipeline.py -q
```

Tests under `server/tests/` cover both the FastAPI behavior and the ML/data
utilities, including script help, deterministic generation, audio safety,
manifest fields, dataset preparation, training helpers, and model components.

## Reproducibility and Generated Files

- YAML files under `configs/` define the main experiment inputs.
- Data-generation code records seeds and augmentation metadata in JSONL.
- Dataset preparation and degradation scripts are designed to be deterministic
  where the underlying codec/tooling permits it.
- Do not commit generated audio, downloaded datasets, checkpoints, or large
  reports under `data/` or `artifacts/`.
