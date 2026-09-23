"""Prepare inputs, cache WER profiles, and train the Whisper Small PQ+RI bridge."""
from __future__ import annotations

import argparse
from pathlib import Path

from ml.fusion import bridging_experiment, prepare_bridge_inputs
from ml.fusion.bridge_small_data import load_training_config

DEFAULT_CONFIG = Path("configs/speech_enhancement/bridge_small_train.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("stage", choices=("prepare", "cache", "train"),
                        help="enhance configured audio, cache 11 WERs, or train the PQ+RI bridge")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="training YAML with datasets, checkpoints, and stage settings")
    parser.add_argument("--device", default="cuda", help="torch device for FRCRN, Whisper, and bridge")
    parser.add_argument("--dry-run", action="store_true", help="prepare: validate and count without models or output")
    parser.add_argument("--resume", action="store_true", help="cache: resume a matching partial WER cache")
    args = parser.parse_args(argv)
    if args.dry_run and args.stage != "prepare":
        parser.error("--dry-run applies only to prepare")
    if args.resume and args.stage != "cache":
        parser.error("--resume applies only to cache")
    config = load_training_config(args.config)
    inputs = Path(config["run"]["inputs_dir"])
    cache = Path(config["run"]["cache_dir"])
    if args.stage == "prepare":
        settings = config["preparation"]
        return prepare_bridge_inputs.main([
            "--train-config", str(args.config), "--output", str(inputs), "--device", args.device,
            "--batch-size", str(settings["batch_size"]), "--workers", str(settings["workers"]),
            "--seed", str(config["training"]["seed"]),
            *(["--dry-run"] if args.dry_run else []),
        ])
    if args.stage == "cache":
        settings = config["cache"]
        return bridging_experiment.main([
            "prepare", "--manifest", str(inputs / "bridge_train_dev.jsonl"),
            "--asr-checkpoint", str(config["model"]["asr_checkpoint"]),
            "--output", str(cache), "--device", args.device,
            "--batch-size", str(settings["batch_size"]), "--workers", str(settings["workers"]),
            *(["--resume"] if args.resume else []),
        ])
    settings = config["training"]
    return bridging_experiment.main([
        "train", "--cache", str(cache), "--output", str(config["run"]["output_dir"]),
        "--device", args.device, "--epochs", str(settings["epochs"]),
        "--lr", str(settings["learning_rate"]), "--batch-size", str(settings["batch_size"]),
        "--accumulation", str(settings["accumulation"]), "--workers", str(settings["workers"]),
        "--eval-max-batches", str(settings["eval_max_batches"]), "--seed", str(settings["seed"]),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
