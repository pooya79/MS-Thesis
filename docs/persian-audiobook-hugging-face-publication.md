# PersianAudiobook Hugging Face Publication

This document records the initial PersianAudiobook publication and the procedure
for adding a complete refined release later. The live dataset repository is
[`Pooya-Fallah/PersianAudiobook`](https://huggingface.co/datasets/Pooya-Fallah/PersianAudiobook).

## Initial release

The initial `refined-subset` configuration was prepared from:

- accepted rows in
  `data/iranseda/250h-refinement-reports/refined_transcription.tsv`;
- audio clips under `data/iranseda/audiobooks/segmented`;
- segmentation provenance in
  `data/iranseda/audiobooks/segmented/segments.jsonl`; and
- summary records under `data/iranseda/250h-refinement-reports`.

Despite the working name "250h refinement," filtering left 39,454 accepted
examples and 218.564 hours of mono 16 kHz PCM-16 FLAC audio. The accepted source
audio occupies 13.161 GiB and comes from 136 source recordings. Rejected,
uncertain, invalid, and operationally failed targets are not published.

The uploader packages the accepted clips as 27 Parquet files under
`refined-subset/train-*.parquet`. Audio bytes are embedded in a typed Hugging
Face `Audio` column. Each row also contains the stable clip identifier, refined
pseudo-transcription, source identifier, timing and speech measurements,
segmentation boundary type, and clip checksum. Complete refinement prompts,
rejected rows, server-local paths, and internal audit artifacts remain outside
the Hub dataset.

The upload is intentionally sequential. It builds one approximately 512 MiB
source-audio shard, uploads it with one worker, checkpoints the successful Hub
commit, deletes the temporary Parquet file, waits ten seconds, and proceeds to
the next shard. Its state file is:

```text
data/iranseda/hf-upload/refined-subset.state.json
```

This file records the immutable input hashes, shard-plan settings, and every
successfully committed shard. Keep it on the server for safe resumes, but do not
commit it because `data/` contains generated and machine-specific artifacts.

## Dataset card and access policy

The Hub `README.md` is maintained in
`docs/huggingface/PersianAudiobook/README.md`. The pipeline figure is maintained
in `Thesis/figs/long-audio-data-creation-pipeline.png` and is uploaded to the
dataset repository as `assets/long-audio-data-creation-pipeline.png`.

The repository is public so its description and metadata are visible, but its
files use Hugging Face manual gating. The standard form collects the requester's
Hugging Face username and email, and the card adds one text field asking for the
intended use and research purpose. The publisher reviews each request and grants
access only after verifying the requester's permissions.

The card uses `license: other`. It does not apply the MIT license to the audio:
the publisher cannot grant a permissive redistribution license for third-party
recordings or literary works. The card states that access approval does not
grant ownership or redistribution rights.

## Repository-maintained publication tools

- `ml/speech_data/scripts/hf_audio_dataset.py` validates TSV rows and joins
  segmentation metadata.
- `ml/speech_data/scripts/summarize_hf_audio_dataset.py` computes exact dataset
  statistics and an optional Markdown card snippet.
- `ml/speech_data/scripts/upload_hf_audio_dataset.py` implements generic,
  resumable, one-shard-at-a-time Hugging Face publication.
- `ml/speech_data/scripts/upload_persian_audiobook_subset.sh` fixes the paths and
  safety settings for the initial `refined-subset` release.
- `docs/huggingface/PersianAudiobook/README.md` is the published dataset card.

Do not commit an access token, shell history containing a token, generated
Parquet files, upload logs, or upload state. Authenticate with `hf auth login`
or place a write-scoped token in `HF_TOKEN`. Rotate any token that has been
shared in chat, a command line, or another non-secret channel.

## Resume the initial upload

If the initial upload is interrupted, rerun the same launcher from the server:

```bash
cd /home/user01/MS-Thesis
bash ml/speech_data/scripts/upload_persian_audiobook_subset.sh \
  Pooya-Fallah/PersianAudiobook
```

Run that command inside a persistent terminal such as `screen`. Do not delete or
edit `refined-subset.state.json`; the uploader uses it to skip committed shards.
Changing the manifest, config name, split, shard size, row-group size, or
segmentation manifest requires a different state file.

## Add the complete refined collection later

Publish the complete collection as a **new configuration**, recommended name
`refined-full`. Do not append its rows to `refined-subset`: the subset is not
guaranteed to be a prefix of the complete manifest, and rebuilding its existing
shards could duplicate, reorder, or invalidate checkpointed examples. Keeping
both configs also preserves the reproducibility of the initial release. A clip
may consequently appear in both configurations, but a consumer selects one
configuration at load time.

### 1. Freeze and summarize the complete manifest

Create a final accepted TSV with `path` and `sentence` columns, retain its
matching `segments.jsonl`, and generate exact statistics before editing the
card. For example, after replacing the placeholder paths:

```bash
uv run python -m ml.speech_data.scripts.summarize_hf_audio_dataset \
  --manifest COMPLETE_REPORT_DIR/refined_transcription.tsv \
  --audio-root data/iranseda/audiobooks/segmented \
  --segments-manifest data/iranseda/audiobooks/segmented/segments.jsonl \
  --refinement-summary COMPLETE_REPORT_DIR/refinement_summary.json \
  --output COMPLETE_REPORT_DIR/hf_dataset_summary.json \
  --card-snippet-output COMPLETE_REPORT_DIR/hf_card_statistics.md \
  --workers 8
```

Inspect the accepted count, duration, missing-file checks, transcript statistics,
and refinement totals. Do not proceed if the summary does not match the intended
release.

### 2. Update the dataset card

Update the release statistics and limitations in the card. Preserve the manual
gate, intended-use field, rights statement, and source attribution. Extend the
YAML `configs` list without removing `refined-subset`:

```yaml
configs:
  - config_name: refined-subset
    data_files:
      - split: train
        path: refined-subset/train-*.parquet
  - config_name: refined-full
    data_files:
      - split: train
        path: refined-full/train-*.parquet
```

If the complete release changes the processing pipeline, schema, limitations,
or intended uses, describe those differences explicitly rather than presenting
both configurations as equivalent.

### 3. Validate a new immutable shard plan

Use a new config name, state file, and temporary directory:

```bash
uv run python -m ml.speech_data.scripts.upload_hf_audio_dataset \
  --manifest COMPLETE_REPORT_DIR/refined_transcription.tsv \
  --audio-root data/iranseda/audiobooks/segmented \
  --segments-manifest data/iranseda/audiobooks/segmented/segments.jsonl \
  --repo-id Pooya-Fallah/PersianAudiobook \
  --config-name refined-full \
  --state-file data/iranseda/hf-upload/refined-full.state.json \
  --temp-dir data/iranseda/hf-upload/refined-full-tmp \
  --target-shard-mib 512 \
  --row-group-size 100 \
  --max-workers 1 \
  --sleep-seconds 10 \
  --card docs/huggingface/PersianAudiobook/README.md \
  --card-asset Thesis/figs/long-audio-data-creation-pipeline.png \
  --public \
  --gated-manual \
  --dry-run
```

The dry run does not contact Hugging Face or create Parquet shards. Record its
row count, total source size, number of planned shards, and largest planned
shard in the release notes.

### 4. Start or resume the complete upload

After reviewing the dry-run output, repeat the command without `--dry-run`
inside `screen`. The repository already exists, so `--create-repo` is
unnecessary. Keep `--public --gated-manual` so the script verifies the intended
visibility and reapplies manual gating before uploading the updated card or any
new data shard.

For an intentionally bounded batch, add either `--max-shards N` or `--max-gib
N`. Rerun the identical command for later batches. Never reuse
`refined-subset.state.json` for the full release.

### 5. Verify completion

After the command reports `"complete": true`:

1. Confirm that all planned `refined-full/train-*.parquet` files appear on the
   Hub and that the local state records the same count.
2. Confirm that the repository remains public and gated in `manual` mode.
3. Review the rendered card, pipeline image, statistics, form field, and both
   configuration names.
4. Approve a test account, load several rows from `refined-full`, and inspect
   audio decoding, Persian text, identifiers, timing metadata, and checksums.
5. Test metadata-only and streaming access so the complete dataset does not
   need to be downloaded to one disk before use.
6. Retain the final manifest, refinement summary, segmentation manifest, and
   state file together as the internal release record.
7. Revoke or rotate the upload token when publication is complete.

Consumers can then select the complete release explicitly:

```python
from datasets import load_dataset

dataset = load_dataset(
    "Pooya-Fallah/PersianAudiobook",
    "refined-full",
    split="train",
    token=True,
)
```
