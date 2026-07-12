"""Fine-tune ``openai/whisper-large-v3-turbo`` for Persian ASR.

The data pipeline, artifacts, metrics, and resume behavior intentionally share
the well-tested Whisper-small implementation. This entry point supplies
large-v3-turbo defaults, including gradient checkpointing for lower VRAM use.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from ml.asr.train_whisper_small import DEFAULT_CONFIG, deep_merge
from ml.asr.train_whisper_small import load_training_config as _load_training_config
from ml.asr.train_whisper_small import run_training as _run_training


DEFAULT_CONFIG_LARGE_V3_TURBO: dict[str, Any] = deep_merge(
    deepcopy(DEFAULT_CONFIG),
    {
        "model": {"name": "openai/whisper-large-v3-turbo"},
        "run": {"output_dir": "models/asr/whisper-large-v3-turbo/runs"},
        "training": {
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "gradient_checkpointing": True,
        },
    },
)


def load_training_config(config_path: Path) -> dict[str, Any]:
    return _load_training_config(config_path, DEFAULT_CONFIG_LARGE_V3_TURBO)


def run_training(
    config_path: Path,
    run_dir_override: Path | None = None,
    resume_override: str | None = None,
) -> int:
    return _run_training(
        config_path,
        run_dir_override,
        resume_override,
        defaults=DEFAULT_CONFIG_LARGE_V3_TURBO,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune openai/whisper-large-v3-turbo from a YAML config. "
            "Checkpoints can be resumed with run.resume: auto or --resume auto."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML training config path.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Optional run directory override.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume mode: auto, false, or an explicit checkpoint directory. Overrides run.resume.",
    )
    args = parser.parse_args(argv)
    return run_training(args.config, args.run_dir, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
