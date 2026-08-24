#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: upload_persian_audiobook_subset.sh OWNER/PersianAudiobook

Upload the initial refined Persian audiobook subset from this server to a
manually gated repository in sequential, resumable Parquet shards. Authenticate
first with `hf auth login` or export HF_TOKEN. Run inside screen for a long-lived upload.

The command uses one Hub upload worker and stages only one approximately
512 MiB shard at a time. Successful shards are checkpointed and deleted from
the temporary directory. Rerun the identical command after interruption.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

repo_id=$1
if [[ ! "$repo_id" =~ ^[A-Za-z0-9._-]+/PersianAudiobook$ ]]; then
    echo "error: repository must be OWNER/PersianAudiobook" >&2
    exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
card_path="$repo_root/docs/huggingface/PersianAudiobook/README.md"
uv_bin=${UV_BIN:-/home/user01/.local/bin/uv}

if [[ ! -x "$uv_bin" ]]; then
    echo "error: uv executable not found: $uv_bin" >&2
    exit 2
fi
if ! grep -q "This is a restricted-access research dataset" "$card_path"; then
    echo "error: dataset card must retain the gated research restriction" >&2
    exit 2
fi

cd "$repo_root"
exec "$uv_bin" run python -m ml.speech_data.scripts.upload_hf_audio_dataset \
    --manifest data/iranseda/250h-refinement-reports/refined_transcription.tsv \
    --audio-root data/iranseda/audiobooks/segmented \
    --segments-manifest data/iranseda/audiobooks/segmented/segments.jsonl \
    --repo-id "$repo_id" \
    --config-name refined-subset \
    --state-file data/iranseda/hf-upload/refined-subset.state.json \
    --temp-dir data/iranseda/hf-upload/tmp \
    --target-shard-mib 512 \
    --row-group-size 100 \
    --max-workers 1 \
    --sleep-seconds 10 \
    --card "$card_path" \
    --card-asset Thesis/figs/long-audio-data-creation-pipeline.png \
    --create-repo \
    --public \
    --gated-manual
