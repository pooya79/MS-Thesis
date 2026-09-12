# Publish the Persian Whisper models

`ml.asr.upload_hf_models` uploads the medium and small exports
directly from the server to two Hugging Face model repositories. Run it from the
checkout after transferring these code changes with your usual Git push/pull
workflow. It requires no GPU, does not load weights into memory, and does not
modify the source checkpoints.

## Choose the exports

Edit `configs/huggingface_models.yaml` before publication. Each entry has a model
key, repository name (without an owner), title, base model, checkpoint path,
saved training-config path, and a provenance note. Relative paths resolve against
the project root, not the configuration file's directory. Absolute paths work too.

The defaults are:

| Key | Export selected from the configured run | Destination suffix |
| --- | --- | --- |
| `medium` | `final` | `whisper-medium-fa` |
| `small` | `final` | `whisper-small-fa` |

`final` is an explicit candidate, not an automatic best-checkpoint choice. Change
`checkpoint` to the corresponding `best` directory if that is the export you
intend to release. Compare validation results and confirm the training manifests
and logs before claiming full dataset coverage or an exact multi-condition recipe.
The generated cards preserve that uncertainty. Edit the provenance note and title
once the actual composition is established. Dataset names in the card come from
the selected saved training configuration. Local checkpoint paths remain in the
publication configuration and console output; they are not included in the
uploaded card or metadata. Published metadata uses the model key and export name.

## Server commands

After pulling the changes into `~/MS-Thesis`:

```bash
cd ~/MS-Thesis
uv run python -m ml.asr.upload_hf_models --help
uv run hf auth login
```

Use your own Hugging Face write token, scoped to the intended repositories with
permission to create them. An organization namespace must already exist and your
account must have appropriate permissions there. `HF_TOKEN` also works; never
commit it to Git. Readers authenticate with their own accounts and read tokens.

First create a local preview. Replace `YOUR_HF_USERNAME` with your actual username:

```bash
uv run python -m ml.asr.upload_hf_models \
  --namespace YOUR_HF_USERNAME \
  --access gated
```

This validates both exports and writes `README.md` and `publication.json` under
`artifacts/huggingface/model-upload-preview/{medium,small}/`. It makes no Hub
requests and copies no weights. Read the generated cards before uploading.

For **shareable model-page links with manual download approval**:

```bash
uv run python -m ml.asr.upload_hf_models \
  --namespace YOUR_HF_USERNAME \
  --access gated \
  --upload
```

The script prints the model URLs and uploaded commit SHAs. Send the model URLs to
the people you want to share with. They sign in and request access on each page;
approve their accounts in each model's access-request settings. You do not need
their emails in advance. In this mode, the card and model description are public.
The script does not invite people, send messages, or approve requests automatically.

For **fully private repositories** owned by an organization:

```bash
uv run python -m ml.asr.upload_hf_models \
  --namespace YOUR_HF_ORGANIZATION \
  --access private \
  --upload
```

`private` is the default. Invite the readers by Hugging Face username to the
owning organization with the `read` role, then share the links. A private repo
under your personal account does not provide link-based access to other people.
Standard organization membership grants access across its repositories; use a
dedicated organization for these models or paid resource groups for finer access.

To publish just one model, add `--only medium` or `--only small`. For alternate
storage, use `--project-root /path/to/MS-Thesis`. `--config /path/to/models.yaml`
selects another publication configuration and `--preview-dir /path/to/previews`
changes the preview destination. Run `--help` for argument types and defaults.

## Upload contents and failure behavior

Only known inference files are selected: weights (preferring safetensors), weight
indexes and their shards, model/generation configuration, and processor/tokenizer
assets. Missing/empty required files, missing shards, mismatched small/medium
architecture, and incompatible sample rates stop publication. Trainer state,
optimizer/scheduler state, logs, training audio, and raw manifests are excluded.
These are structural checks; they do not run inference or prove model quality.

Every selected local export is validated before any Hub request. New repositories
are created private. Gated mode configures and verifies manual gating before
uploading, then makes the repository public only after the upload commits.
Existing public ungated/auto-gated destinations are refused. Private mode also
refuses existing public repositories. This script never silently changes an
existing public repository into a private one.

Each model's selected files and card are committed together. The two repositories
are independent: if the second upload fails, the first remains uploaded. Rerun
the same command, optionally with `--only` for the unfinished model. Hub storage
deduplicates unchanged content; this script has no separate resume-state file.
A failed new gated upload leaves its repository private. A rerun of an existing
public/manual-gated model keeps its existing access policy during the update.
An interrupted final visibility update can be completed by rerunning.

Existing unrelated files are retained. Obsolete remote inference files from
another serialization/sharding or tokenizer layout cause an error instead of
mixing exports; choose a fresh repository name or remove obsolete files manually.
Stop training or choose an immutable completed export before upload, so weights
cannot change during hashing or transfer.

The generated model cards contain authenticated CPU/GPU inference examples for
short and long audio, provenance, and limitations. Metrics and a fine-tuned-model
license are not invented. To change the shared guide, edit
`docs/huggingface/whisper-model-card.template.md`; reruns regenerate and replace
the remote card. Generated previews and all weights stay outside Git.

## Verification

```bash
uv run pytest -q server/tests/test_upload_hf_models.py
```

Tests use synthetic exports and a fake Hub API; no account, network, GPU, or real
weights are needed. After publication, log in as an approved reader and run the
model-card example on a known recording to verify end-to-end access and inference.

References: [Hub repository management](https://huggingface.co/docs/huggingface_hub/guides/repository),
[manual gating](https://huggingface.co/docs/hub/models-gated),
[organization access](https://huggingface.co/docs/hub/organizations-security),
and [Whisper inference](https://huggingface.co/docs/transformers/model_doc/whisper).
