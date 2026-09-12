#!/usr/bin/env bash
# Execute the fixed official-split Tiny recipe from the repository root.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: bash ml/speech_data/scripts/run_cv25_tiny_official.sh"
  echo "No arguments. Defaults: official CV25 root, CUDA, fixed Tiny configs."
  echo "Requires prepared data/cv25-official/selection.json; stops on first failure."
  echo "Logs/status: artifacts/cv25-tiny-official/run-all/; existing run refused."
  exit 0
fi
[[ $# == 0 ]] || { echo "Unexpected argument; use --help" >&2; exit 2; }
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "uv is required on PATH" >&2; exit 1; }
run=artifacts/cv25-tiny-official/run-all
mkdir -p "$(dirname "$run")"
mkdir "$run"
exec > >(tee -a "$run/run.log") 2>&1
trap 'result=$?; echo "$result" > "$run/exit-code"; date -Is > "$run/finished-at"' EXIT
cp -r configs/speech_enhancement/cv25_tiny "$run/configs"
cp data/cv25-official/selection.json "$run/selection.json"
git rev-parse HEAD > "$run/git-head"
git diff -- configs/speech_enhancement/cv25_tiny ml/fusion ml/asr > "$run/code.diff"
sha256sum ml/speech_data/prepare_cv25_official.py ml/speech_data/scripts/run_cv25_tiny_official.sh > "$run/scripts.sha256"
date -Is > "$run/started-at"
echo "$$" > "$run/pid"
uv run python -c 'import json, torch; assert json.load(open("data/cv25-official/selection.json"))["complete"]; assert torch.cuda.is_available()'
uv run python -m ml.fusion.prepare_bridge_inputs --data-root data/cv25-official --output artifacts/cv25-tiny-official/bridge-inputs --dry-run

uv run python -m ml.fusion.prepare_bridge_inputs --data-root data/cv25-official --output artifacts/cv25-tiny-official/bridge-inputs --device cuda


uv run python -m ml.asr.train_whisper_small \
  --config configs/speech_enhancement/cv25_tiny/baseline.yaml


uv run python -m ml.fusion.bridging_experiment prepare \
  --manifest artifacts/cv25-tiny-official/bridge-inputs/bridge_train_dev.jsonl \
  --asr-checkpoint models/asr/cv25-tiny-official/baseline/best \
  --output artifacts/cv25-tiny-official/bridge-cache \
  --device cuda


# Paper baseline reconstruction: perceptual quality + recognition information.
uv run python -m ml.fusion.bridging_experiment train \
  --cache artifacts/cv25-tiny-official/bridge-cache \
  --output models/asr/cv25-tiny-official/bridge \
  --device cuda

# Loss ablation: perceptual quality only, with the same cached inputs.
uv run python -m ml.fusion.bridging_experiment train \
  --cache artifacts/cv25-tiny-official/bridge-cache \
  --output models/asr/cv25-tiny-official/bridge-pq \
  --pq-only --device cuda


uv run python -m ml.fusion.train_fusion \
  --config configs/speech_enhancement/cv25_tiny/cross_attention.yaml


mkdir -p models/asr/cv25-tiny-official/gated/checkpoints/stage0_warmup
cp -n models/asr/cv25-tiny-official/cross_attention/checkpoints/stage0_warmup/enhancer.pt \
  models/asr/cv25-tiny-official/gated/checkpoints/stage0_warmup/enhancer.pt

uv run python -m ml.fusion.train_fusion \
  --config configs/speech_enhancement/cv25_tiny/gated.yaml \
  --resume-from-stage fusion

mkdir -p models/asr/cv25-tiny-official/residual_cross_attention/checkpoints/stage0_warmup
cp -n models/asr/cv25-tiny-official/cross_attention/checkpoints/stage0_warmup/enhancer.pt \
  models/asr/cv25-tiny-official/residual_cross_attention/checkpoints/stage0_warmup/enhancer.pt

uv run python -m ml.fusion.train_fusion \
  --config configs/speech_enhancement/cv25_tiny/residual_cross_attention.yaml \
  --resume-from-stage fusion


uv run python -m ml.fusion.bridging_experiment evaluate \
  --manifest artifacts/cv25-tiny-official/bridge-inputs/bridge_dev.jsonl \
  --checkpoint models/asr/cv25-tiny-official/bridge/best.pt \
  --output artifacts/cv25-tiny-official/bridge-dev --split dev --device cuda

uv run python -m ml.fusion.bridging_experiment evaluate \
  --manifest artifacts/cv25-tiny-official/bridge-inputs/bridge_dev.jsonl \
  --checkpoint models/asr/cv25-tiny-official/bridge-pq/best.pt \
  --output artifacts/cv25-tiny-official/bridge-pq-dev --split dev --device cuda

uv run python -m ml.fusion.bridging_experiment evaluate \
  --manifest artifacts/cv25-tiny-official/bridge-inputs/bridge_dev.jsonl \
  --checkpoint models/asr/cv25-tiny-official/bridge/best.pt \
  --omega 1 --output artifacts/cv25-tiny-official/bridge-original-dev \
  --split dev --device cuda

uv run python -m ml.fusion.bridging_experiment evaluate \
  --manifest artifacts/cv25-tiny-official/bridge-inputs/bridge_dev.jsonl \
  --checkpoint models/asr/cv25-tiny-official/bridge/best.pt \
  --omega 0 --output artifacts/cv25-tiny-official/bridge-enhanced-dev \
  --split dev --device cuda


uv run python -m ml.fusion.evaluate_ablation \
  --config artifacts/cv25-tiny-official/bridge-inputs/final_tests.yaml \
  --output artifacts/cv25-tiny-official/final-tests-all \
  --device cuda
