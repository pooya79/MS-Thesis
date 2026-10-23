# Persian Speech Recognition — MS Thesis Workspace

Research code and experiment archive for an MS thesis on Persian automatic
speech recognition (ASR), with a focus on recognizing noisy and degraded speech.
The workspace brings together dataset preparation, model training and evaluation,
enhancement/fusion experiments, and a FastAPI playground called **Persian Speech Lab**.

The main research question is whether combining the original and enhanced views
of speech improves transcription. The repository also contains baseline models,
waveform-bridging experiments, and tools for creating Persian speech datasets.
Experiments share utilities, but have separate inputs, configs, and checkpoints.

## Start here

| I want to… | Start with… |
| --- | --- |
| Download a published Persian Whisper checkpoint | [Published models](#published-persian-whisper-models) |
| Try a model on uploaded or recorded audio | [Run the playground](#run-the-playground) |
| Download and prepare speech datasets | [Prepare-data workflow](#1-prepare-speech-datasets) and [dataset guide](docs/script-guides/dataset-download-and-preparation.md) |
| Create noisy or telephone-style speech | [Degradation workflow](#2-generate-and-validate-degraded-speech) |
| Fine-tune or evaluate Whisper / FastConformer | [ASR workflow](#3-train-and-evaluate-asr-models) |
| Understand or train enhancement and fusion | [Fusion workflow](#4-train-and-evaluate-enhancementfusion) |
| Run the official CV25 / Whisper Tiny comparison | [Complete experiment sequence](docs/script-guides/cv25-tiny-run-all.md) |
| Explore waveform bridging or pMCT | [Other experiment workflows](#other-experiment-workflows) |
| Turn long recordings into labeled clips | [Long-audio pipeline](#long-audio-dataset-creation) |
| Find recorded results | [Experiment archive](#experiment-archive-and-reproducibility) |

## Published Persian Whisper models

The project's Persian Whisper checkpoints are available on Hugging Face:

| Model | Repository ID and model card |
| --- | --- |
| Whisper Small — Persian | [Pooya-Fallah/whisper-small-fa](https://huggingface.co/Pooya-Fallah/whisper-small-fa) |
| Whisper Medium — Persian | [Pooya-Fallah/whisper-medium-fa](https://huggingface.co/Pooya-Fallah/whisper-medium-fa) |

Use the repository ID as `checkpoint` in a `whisper` entry in
[`configs/server.yaml`](configs/server.yaml). Consult each model card for access
requirements, inference instructions, and training provenance. For gated downloads,
request access on the model page and authenticate with an approved Hugging Face
account on the machine running inference.

## How the pieces fit together

```text
Speech datasets / long recordings
    |
    +-- prepare audio + normalize Persian transcripts --> TSV datasets
    |                                                        |
    |                   +------------------------------------+
    |                   |
    |                   +-- train/evaluate ASR baselines
    |                   |
    |                   +-- generate aligned degraded speech
    |                           |
    |                           +-- enhancement + feature fusion
    |                           +-- waveform bridge experiments
    |
    +-- segment --> Whisper transcribe --> contextual text refinement

Trained checkpoints --> evaluation reports / local FastAPI playground
```

The two enhancement approaches operate in different domains:

- **Feature fusion:** a learned enhancer estimates cleaner Whisper log-Mel
  features. Original and enhanced features pass through a shared Whisper encoder;
  a fusion block combines their hidden states before the decoder transcribes them.
  This enhancer produces features for ASR, rather than restored playback audio.
- **Waveform bridging:** a frozen FRCRN speech enhancer produces an enhanced
  waveform. A learned bridge mixes original and enhanced audio before frozen
  Whisper recognition. DNSMOS quality scores and recognition targets supervise
  bridge training. This is a separate reconstruction of a published method.

**WER** (word error rate) and **CER** (character error rate) are the main ASR
metrics; lower is better. An *ablation* evaluates a reduced system, such as using
only the original view, to measure what a component contributes.

## Repository map

| Path | Purpose |
| --- | --- |
| [`ml/speech_data/`](ml/speech_data/) | Downloads, preparation, normalization, degradation, validation, and long-audio tools |
| [`ml/asr/`](ml/asr/) | Whisper and FastConformer training, local/external evaluation, and model publication |
| [`ml/enhancement/`](ml/enhancement/) | Log-Mel enhancer, paired-data loading, and enhancement diagnosis |
| [`ml/fusion/`](ml/fusion/) | Feature fusion, waveform bridging, and ablation experiments |
| [`ml/fa_fastconformer/`](ml/fa_fastconformer/) | Standalone Persian FastConformer-CTC implementation and checkpoint conversion |
| [`ml/pmct/`](ml/pmct/) | Isolated Patched Multi-Condition Training experiment |
| [`configs/`](configs/) | Editable YAML recipes for datasets, training, evaluation, and the server |
| [`server/app/`](server/app/) | FastAPI application, authentication, inference service, templates, and static assets |
| [`server/tests/`](server/tests/) | Tests for both the server and most ML/data utilities |
| [`docs/`](docs/) | Script guides, architecture explanations, design decisions, and diagrams |
| [`notebooks/`](notebooks/) | Project, ASR, and IranSeda pipeline demonstrations |
| [`report/`](report/) | Archived experiment configs, metrics, and progress reports |
| [`Thesis/`](Thesis/) | Thesis documents and research notes |
| `data/`, `models/`, `artifacts/` | Local datasets, checkpoints, caches, and generated outputs; ignored by Git |

## Environment setup

Run commands from the repository root unless a linked guide says otherwise.

- **Python:** the project requires Python **3.13 or newer**; `.python-version`
  selects 3.13.
- **Dependency manager:** `uv`; dependencies are declared in
  [`pyproject.toml`](pyproject.toml) and pinned in [`uv.lock`](uv.lock).
- **Audio tools:** `ffmpeg` and `ffprobe` on `PATH`.
- **Compute:** a CUDA-capable GPU is practical for training and large evaluations.
  Several inference tools support CPU; the official Tiny launcher requires CUDA.
- **Storage and downloads:** datasets, model weights, and generated audio are
  separate from the checkout. Allow space for the inputs and outputs of your recipe.

Install the locked Python environment:

```bash
uv sync --locked
```

The dependency configuration selects the PyTorch CUDA 13.0 wheel index for Torch
and Torchvision on Linux/Windows. GPU use needs a compatible NVIDIA driver.
The environment includes substantial ML dependencies even if you only run the UI.

On Ubuntu/Debian, install the system audio tools with:

```bash
sudo apt update
sudo apt install ffmpeg
ffmpeg -version
ffprobe -version
```

Full telephone/VoIP degradation additionally needs the FFmpeg encoders
`pcm_alaw`, `pcm_mulaw`, `libgsm` (or `gsm`), `libopencore_amrnb`,
`libvo_amrwbenc`, and `libopus`. Inspect **encoders**, since decoder support alone
is insufficient:

```bash
ffmpeg -hide_banner -encoders | grep -Ei 'amr|gsm|opus|pcm_alaw|pcm_mulaw'
```

Distribution FFmpeg builds may omit AMR encoders. The generators check the codecs
selected by the config before generating data. Opus packet-loss concealment also
requires the system `libopus` shared library. See the
[degradation guide](docs/script-guides/speech-degradation.md) for the workflow;
changing the codec distribution changes the experiment recipe.

[`.env.example`](.env.example) lists the basic environment variables. `make run`
loads the repository-root `.env` through `uv --env-file`; set `APP_PASSWORD`
there before starting the server. Already exported variables take precedence.
The Common Voice downloader also supports reading its API key
from `.env`. Other services have their own credentials, described in their guides.

## Run the playground

The UI lets you choose a configured model, upload audio or record from a microphone,
preview it, and copy the transcript. It also shows duration and inference time.

1. Install the environment and FFmpeg as above.
2. Edit [`configs/server.yaml`](configs/server.yaml). Point each model you want
   to use at an available checkpoint. The checked-in config uses the repository's
   `models/` directory: Whisper Small (Normal, Degraded v2, IranSeda),
   Fusion (Run 004 and Run 005 Compact), and FastConformer (Pretrained, Normal,
   Degraded v2). Fine-tuned entries use `best` / `best.pt` validation checkpoints;
   Fusion entries use the joint-stage checkpoints. Processors, tokenizers, and
   Fusion backbones also use local paths, so no Hub download is required.
   The weights are **not bundled**. On this machine, inference files were copied
   from `/home/pouya/models` into the same relative layout under `models/`;
   training-only files such as Whisper optimizer state were omitted. Copies were
   verified with SHA-256, recorded in the ignored `models/demo-copy-manifest.json`.
   Run the server from the repository root so these relative paths resolve.
   For a basic Whisper trial, a `whisper` entry can use
   `checkpoint: openai/whisper-small` and `processor: openai/whisper-small`;
   this downloads the upstream model on first use.
3. Set a shared password in the repository-root `.env` (create it from
   `.env.example` if needed):

   ```dotenv
   APP_PASSWORD=replace-this-with-a-long-random-password
   ```

   Start the development server:

   ```bash
   make run
   ```

`make run` loads `.env` and uses the installed environment with
`uv run --no-sync --env-file .env`, so starting
the UI does not trigger package downloads. Run `uv sync --locked` explicitly
when installing or updating dependencies. See the
[playground guide](docs/script-guides/asr-playground.md) for startup troubleshooting.

Open [http://localhost:8001](http://localhost:8001) and enter that password.
A configured model can appear in the UI before its weights are available:
loading happens on first use, so check checkpoint paths before transcribing.

Useful server settings:

| Setting | Default / behavior |
| --- | --- |
| `ASR_SERVER_CONFIG` | Optional environment variable selecting another YAML file; defaults to `configs/server.yaml` |
| `APP_PASSWORD` | Required; one shared password, with no account or database layer |
| `app.max_upload_mb` | 25 MB |
| `app.max_audio_seconds` | 30 seconds |
| `app.sample_rate` | 16,000 Hz |
| Model `device` | `auto`; configurable per model |

Local checkpoint paths resolve from the directory where the server starts.
Models load lazily and remain cached. Uploads are size-bounded, converted by
FFmpeg into temporary mono WAV files, and removed after the request.

Authentication uses a signed HTTP-only session cookie lasting seven days;
changing the password and restarting invalidates existing sessions. `/login`
and `/static/*` are public; application pages and APIs require authentication.
The APIs include `GET /api/models`, `GET /api/models/{model_id}/status`, and
`POST /api/transcriptions` (multipart fields `model_id` and `audio`).

`make run` starts Uvicorn with reload on `0.0.0.0:8001`, so it listens on all
interfaces. Use HTTPS when exposing the playground beyond your local machine;
browser microphone recording needs localhost or a secure context.

## Core research workflows

**Choose a recipe before running commands.** The configs contain concrete dataset
names, checkpoint paths, and output directories from different experiments.
Review those fields and the linked guide first. The examples below are entry
points, rather than a single sequence that reproduces every archived experiment.
Every maintained Python CLI provides `--help`; the full catalog is in
[Script Guides](docs/script-guides/README.md).

### 1. Prepare speech datasets

For Common Voice Persian, set `MOZILLA_DATA_COLLECTIVE_API_KEY` in your environment
or `.env`, download the archive, and prepare its extracted Persian directory:

```bash
uv run python -m ml.speech_data.scripts.download_common_voice_fa \
  --output-dir data

uv run python -m ml.speech_data.scripts.prepare_common_voice_25 \
  --source-root data/cv-corpus-25.0-2026-03-09/fa \
  --output-root data/cv-corpus-25.0 \
  --workers 4
```

Adjust `--source-root` to match the downloaded archive's extracted layout.
Download/preparation tools also cover FLEURS Persian, PerSets YouTube/Filimo,
and Persian evaluation sets. See the
[dataset download and preparation guide](docs/script-guides/dataset-download-and-preparation.md).

The shared ASR dataset contract is:

```text
dataset/
├── train.tsv          # required for training
├── dev.tsv            # used for validation/checkpoint selection
├── test.tsv           # required for held-out evaluation
└── clips/
    └── example.wav    # supported formats depend on the consuming tool
```

Only include the splits needed by the workflow. TSVs must have `path` and
`sentence` columns; additional metadata columns are allowed. Audio paths resolve
first as `<dataset>/clips/<path>`, then as `<dataset>/<path>`. Preparation tools
normally standardize audio to mono 16 kHz and apply shared Persian text rules.

For normalization, mixed test sets, duration summaries, FLAC conversion, and
Hugging Face dataset publication, use the
[dataset management guide](docs/script-guides/dataset-management.md).

### 2. Generate and validate degraded speech

First download and index DEMAND environmental noise:

```bash
uv run python -m ml.speech_data.scripts.download_degradation_assets \
  --noise-root data/speech_enhancement/assets/noise/DEMAND \
  --manifest-dir data/speech_enhancement/manifests \
  --prepare-indexes
```

Then generate a TSV-shaped dataset from prepared Common Voice audio:

```bash
uv run python -m ml.speech_data.generate_degraded_dataset \
  --config configs/speech_enhancement/cv25_degraded_dataset.yaml \
  --workers 4

uv run python -m ml.speech_data.validate_degraded_dataset \
  --dataset data/cv-corpus-25.0-degraded \
  --sample 300 \
  --output-dir artifacts/degraded_validation
```

The checked-in recipe generates train/dev variants under
`data/cv-corpus-25.0-degraded`, including `degraded_to_clean.jsonl`,
`degradation_metadata.jsonl`, and a generation report. Keep these mappings:
paired training uses them to locate clean sources and reconstruct aligned targets.

The pipeline combines environmental noise, level changes, channel bandwidth,
codec round-trips, and configured network impairments. Random choices are seeded
per variant and recorded. Clean targets are aligned in length and bandwidth;
Opus supports packet-level loss concealment, while other codecs use a documented
waveform-dropout fallback. See the
[degradation design and metadata contract](docs/speech-degradation-pipeline.md).

A separate `generate_degraded_pairs` workflow consumes **clean JSONL manifests**
configured in `configs/speech_enhancement/degradation.yaml`. Create those manifests
first using the schema in the design document; downloading noise does not create
them. It writes clean/degraded pairs and `se_*_pairs.jsonl` manifests:

```bash
uv run python -m ml.speech_data.generate_degraded_pairs \
  --config configs/speech_enhancement/degradation.yaml

uv run python -m ml.speech_data.inspect_manifest \
  data/speech_enhancement/manifests/se_train_pairs.jsonl
```

Noise-only generation and single-clip audible demos are covered in the
[speech degradation guide](docs/script-guides/speech-degradation.md).

### 3. Train and evaluate ASR models

Whisper Small is a useful starting point. Set the dataset list and run directory
in its training config, then point evaluation at the resulting checkpoint:

```bash
uv run python -m ml.asr.train_whisper_small \
  --config configs/whisper_small_train.yaml \
  --resume auto

uv run python -m ml.asr.eval_whisper_small \
  --config configs/whisper_small_eval.yaml
```

The default Small training run exports to
`models/asr/whisper-small/runs/whisper-small-fa/final`, matching the default Small
evaluation and server entry. Whisper Medium and Large-v3-turbo have separate
train/eval modules and configs. The historical `train_whisper_small` module also
runs the Tiny baseline when given the Tiny configuration.

FastConformer has its own training/evaluation workflow:

```bash
uv run python -m ml.asr.train_fastconformer \
  --config configs/fastconformer_train.yaml \
  --resume auto

uv run python -m ml.asr.eval_fastconformer \
  --config configs/fastconformer_eval.yaml
```

The standalone implementation uses the **CTC branch** of the Persian hybrid
FastConformer checkpoint and loads `.nemo` or converted `.pt` weights without
NeMo. See the [model package](ml/fa_fastconformer/README.md) and
[ASR training/evaluation guide](docs/script-guides/asr-training-and-evaluation.md)
for conversion, model-specific settings, and outputs. Evaluation records
predictions and aggregate/per-dataset WER/CER.

### 4. Train and evaluate enhancement/fusion

The reusable curriculum has three stages: train the log-Mel enhancer, train fusion
with the Whisper backbone frozen, then train the system jointly. Warm-up can also
use Whisper encoder feature matching. Fusion supports gated and cross-attention
variants; the YAML selects the architecture and training budget.

Before starting, set `datasets`, `base_asr_checkpoint`, and `run_dir` in the
training config. In particular, its example backbone path
`models/asr/whisper-small/runs/best` differs from the default Small trainer's
`whisper-small-fa` run directory; replace it with your intended saved checkpoint.

```bash
uv run python -m ml.fusion.train_fusion \
  --config configs/speech_enhancement/fusion_train.yaml

uv run python -m ml.fusion.eval_fusion \
  --config configs/speech_enhancement/fusion_eval.yaml
```

Evaluation must select the intended fusion checkpoint and test datasets.
Original-view, enhanced-view, and fixed-gate ablations help isolate the fusion
contribution. `ml.enhancement.diagnose_enhancement` compares a trained enhancer
with the identity baseline to measure how much denoising headroom it captures.
See [training, evaluation, and diagnosis](docs/script-guides/enhancement-and-fusion.md)
and the [architecture explanation](docs/enhancement-degradation-fusion-overview.md).
The architecture note distinguishes recorded fusion-v2 settings from reusable defaults.

## Other experiment workflows

| Workflow | What it does / where to start |
| --- | --- |
| **Official CV25 / Whisper Tiny** | Isolates official splits, trains a shared baseline, reconstructs the waveform bridge, compares fusion variants, and scores nine methods on original CV25 + AGFarsdat test audio. Start with [cohort preparation](docs/script-guides/prepare-cv25-official.md), then the [ordered run guide](docs/script-guides/cv25-tiny-run-all.md). |
| **Whisper Small waveform bridge** | Runs `prepare` → `cache` → `train`, followed by separate evaluation, using frozen FRCRN and Whisper. See [bridge-small](docs/script-guides/bridge-small.md). |
| **Compact Whisper Small fusion** | Uses an existing Persian Small checkpoint and a smaller fusion frontend. This has its own data mixture and output directory; see [compact fusion](docs/script-guides/whisper-small-compact-fusion.md). |
| **pMCT** | Mixes aligned clean/noisy waveform patches during training without saving patched audio. The current implementation uses additive noise; see [pMCT](ml/pmct/README.md). |
| **External ASR comparisons** | Evaluates OpenRouter STT, ElevenLabs Scribe, and Ivira Avanegar, with prediction/cost records. Requires provider credentials and sends audio to the selected service; see [external evaluation](docs/script-guides/external-asr-evaluation.md). |
| **Model publication** | Previews and uploads Whisper exports with model cards and access settings; see [Hugging Face model publication](docs/script-guides/huggingface-model-publication.md). |

The Tiny launcher at `ml/speech_data/scripts/run_cv25_tiny_official.sh` requires
an already prepared official cohort and CUDA. Read the ordered guide before
launching it: it runs a fixed recipe and refuses an existing launcher directory.
Resume rules differ between preparation, caching, training, and evaluation.

## Long-audio dataset creation

This separate pipeline turns long recordings into candidate labeled clips:

1. Discover/download source recordings, including IranSeda audiobooks/radio or
   VOA Persian, using the [IranSeda guide](docs/script-guides/iranseda-scripts.md),
   [IranSeda scraper notes](docs/scraper_guides/iranseda-scrapers.md), and
   [VOA scraper guide](docs/scraper_guides/voa-persian-scraper.md).
2. Segment speech with Silero VAD. Default outputs are mono 16 kHz FLAC clips,
   with source timestamps, checksums, and audit manifests.
3. Transcribe with Whisper and normalize Persian text.
4. Optionally select a reproducible subset and refine transcripts with contextual
   text cleanup through a separately running vLLM server.

Use the [long-audio CLI guide](docs/long-audio-asr-pipeline-guide.md) for commands,
configuration, and resume behavior. Refinement requires vLLM's native synchronous
`/v1/chat/completions/batch` endpoint; a generic chat endpoint is insufficient.
Segmentation/transcription/refinement do not automatically publish train/dev/test
splits. Filtering, human audit, and leakage-safe publication are described in the
[dataset pipeline plan](docs/iranseda-whisper-dataset-pipeline.md).

## Tests

```bash
make test

# Focus on one module while developing:
uv run pytest server/tests/test_degradation_pipeline.py -q
```

`make test` runs `uv run pytest -q`. Most tests live under `server/tests/`,
including ML/data tests; pMCT also has tests under `ml/pmct/tests/`. Coverage
includes authentication, HTML/static assets, uploads and duration limits,
model adapters, CLI help, dataset preparation, deterministic seeds, codec
round-trips, pair alignment, model components, and training helpers.
Tests validate implementation behavior; experiment quality is assessed with
held-out evaluation and the recorded results.

## Experiment archive and reproducibility

Start with the [official Tiny results archive](report/cv25-tiny-official/README.md)
for the nine-method comparison, metrics, effective configs, and provenance.
Earlier progress reports are in [`report/`](report/), including
[progress report 2](report/progress-report-2.md) and its
[Persian version](report/progress-report-2-fa.md).

Use the **saved effective config and provenance for a recorded run** when
interpreting results. Editable recipes under `configs/` can evolve; they need
not reproduce older runs unchanged. Architecture notes and plans also distinguish
implemented features from proposed extensions.

For a traceable experiment:

- Record the Git revision, input dataset identities/splits, effective config,
  seed, model/checkpoint identity, and relevant runtime/codec versions.
- Keep JSONL mappings, source manifests, skipped-input audits, predictions,
  metrics, and training logs alongside the run. Stable random choices do not
  guarantee identical codec output across tool versions or hardware.
- Select checkpoints and design choices on dev; reserve test splits for final
  scoring. Follow the official-cohort guide when reproducing the Tiny study.
- Use a new run directory when changing a recipe incompatibly. Check each CLI's
  resume rules before reusing caches or checkpoints.
- Keep downloaded/generated audio, model weights, and large outputs out of Git.
  `data/`, `models/`, and `artifacts/` are ignored; `report/` holds selected compact
  research records.
