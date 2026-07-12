"""Evaluate a fine-tuned ``openai/whisper-large-v3-turbo`` checkpoint."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from ml.asr.eval_whisper_small import DEFAULT_EVAL_CONFIG, deep_merge
from ml.asr.eval_whisper_small import load_eval_config as _load_eval_config
from ml.asr.eval_whisper_small import run_evaluation as _run_evaluation


DEFAULT_EVAL_CONFIG_LARGE_V3_TURBO: dict[str, Any] = deep_merge(
    deepcopy(DEFAULT_EVAL_CONFIG),
    {
        "model": {"processor": "openai/whisper-large-v3-turbo"},
        "eval": {
            "output_dir": "models/asr/whisper-large-v3-turbo/evals",
            "batch_size": 1,
        },
    },
)


def load_eval_config(config_path: Path) -> dict[str, Any]:
    return _load_eval_config(config_path, DEFAULT_EVAL_CONFIG_LARGE_V3_TURBO)


def run_evaluation(config_path: Path, output_dir_override: Path | None = None) -> int:
    return _run_evaluation(
        config_path,
        output_dir_override,
        defaults=DEFAULT_EVAL_CONFIG_LARGE_V3_TURBO,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a Whisper large-v3-turbo checkpoint on configured TSV datasets."
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML evaluation config path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional evaluation output directory override.")
    args = parser.parse_args(argv)
    return run_evaluation(args.config, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
