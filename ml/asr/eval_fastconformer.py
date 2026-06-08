"""Evaluate the standalone FastConformer-CTC (Persian) model on dataset test splits.

Mirrors ``ml.asr.eval_whisper_small``: it reads a YAML config, loads the configured
dataset ``test.tsv`` files, greedily transcribes every clip with the CTC branch of
``nvidia/stt_fa_fastconformer_hybrid_large`` (no NeMo), and writes ``metrics.json``
(aggregate WER/CER plus per-dataset metrics), ``predictions.jsonl``, the effective
config, logs, and a source manifest.

``model.checkpoint`` may point at either the original ``.nemo`` archive or a
converted ``.pt`` bundle (see ``ml/fa_fastconformer/convert.py``); the format is
chosen from the file extension.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ml.asr.train_whisper_small import (
    WhisperExample,
    character_error_rate,
    deep_merge,
    load_split_examples,
    resolve_dataset_dirs,
    word_error_rate,
    write_examples_manifest,
)


DEFAULT_EVAL_CONFIG: dict[str, Any] = {
    "model": {
        "checkpoint": "models/stt_fa_fastconformer_hybrid_large.nemo",
    },
    "data": {
        "root_dir": "data",
        "datasets": ["cv-corpus-25.0"],
        "sample_rate": 16000,
        "split": "test",
    },
    "eval": {
        "output_dir": "models/asr/fastconformer/evals",
        "name": None,
        "batch_size": 8,
        "max_batch_seconds": 30,
        "device": "auto",
    },
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def load_eval_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    config = deep_merge(DEFAULT_EVAL_CONFIG, loaded)
    validate_eval_config(config)
    return config


def validate_eval_config(config: dict[str, Any]) -> None:
    model = config["model"]
    data = config["data"]
    eval_config = config["eval"]
    if not str(model.get("checkpoint") or "").strip():
        raise ValueError("model.checkpoint must be a non-empty .nemo or .pt path")
    datasets = data.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("data.datasets must be a non-empty list of dataset directory names")
    if any(not str(dataset).strip() for dataset in datasets):
        raise ValueError("data.datasets cannot contain empty values")
    split = str(data.get("split", "")).strip()
    if not split or split.endswith(".tsv"):
        raise ValueError("data.split must be a split name such as test, not a TSV filename")
    minimums = {
        "sample_rate": (data, 8000),
        "batch_size": (eval_config, 1),
    }
    if eval_config.get("max_batch_seconds") is not None:
        minimums["max_batch_seconds"] = (eval_config, 1)
    for key, (section, minimum) in minimums.items():
        if float(section[key]) < minimum:
            raise ValueError(f"{key} must be >= {minimum:g}")
    if eval_config["device"] not in {"auto", "cuda", "cpu"}:
        raise ValueError("eval.device must be auto, cuda, or cpu")


def resolve_existing_path(raw_path: str | Path, config_path: Path | None = None) -> Path:
    source_path = Path(str(raw_path)).expanduser()
    candidates = [source_path]
    if config_path is not None and not source_path.is_absolute():
        candidates.append(config_path.parent / source_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"path does not exist: {raw_path}")


def resolve_output_dir(config: dict[str, Any], override: Path | None = None) -> Path:
    if override is not None:
        return override
    eval_config = config["eval"]
    name = eval_config.get("name") or run_id()
    return Path(str(eval_config["output_dir"])) / str(name)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def configure_logging(output_dir: Path) -> None:
    log_path = output_dir / "logs" / "eval.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    for noisy_logger in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def error_metrics(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    return {
        "wer": word_error_rate(references, hypotheses),
        "cer": character_error_rate(references, hypotheses),
    }


def dataset_error_metrics(
    examples: list[WhisperExample],
    references: list[str],
    hypotheses: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[str]]] = {}
    for example, reference, hypothesis in zip(examples, references, hypotheses, strict=True):
        dataset = str(example.dataset_dir or example.audio_path.parent)
        group = grouped.setdefault(dataset, {"references": [], "hypotheses": []})
        group["references"].append(reference)
        group["hypotheses"].append(hypothesis)

    return [
        {
            "dataset": dataset,
            "examples": len(group["references"]),
            **error_metrics(group["references"], group["hypotheses"]),
        }
        for dataset, group in grouped.items()
    ]


def resolve_device(requested: str) -> str:
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("eval.device is cuda, but CUDA is not available")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_fastconformer(checkpoint: Path, device: str):
    """Load the standalone FastConformer-CTC model from a .nemo or .pt checkpoint."""
    package_dir = Path(__file__).resolve().parents[1] / "fa_fastconformer"
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
    from model import FastConformerCTC  # standalone package (no NeMo)

    if checkpoint.suffix == ".nemo":
        model = FastConformerCTC.from_nemo(str(checkpoint), map_location="cpu")
    else:
        model = FastConformerCTC.from_pretrained(str(checkpoint), map_location="cpu")
    return model.to(device)


def run_evaluation(config_path: Path, output_dir_override: Path | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_eval_config(config_path)
    output_dir = resolve_output_dir(config, output_dir_override)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir)

    effective_config_path = output_dir / "config" / "eval.yaml"
    effective_config_path.parent.mkdir(parents=True, exist_ok=True)
    effective_config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    model_config = config["model"]
    data_config = config["data"]
    eval_config = config["eval"]

    checkpoint = resolve_existing_path(str(model_config["checkpoint"]), config_path)
    device = resolve_device(str(eval_config["device"]))

    dataset_dirs = resolve_dataset_dirs(config)
    split = str(data_config["split"])
    examples = load_split_examples(dataset_dirs, split)
    write_examples_manifest(output_dir / "manifests" / f"{split}.jsonl", examples)

    logging.info("loading checkpoint=%s device=%s", checkpoint, device)
    model = load_fastconformer(checkpoint, device)

    logging.info("transcribing %s %s examples", len(examples), split)
    hypotheses = model.transcribe(
        [str(example.audio_path) for example in examples],
        batch_size=int(eval_config["batch_size"]),
        device=device,
        target_sr=int(data_config["sample_rate"]),
        progress=True,
        max_batch_seconds=(
            float(eval_config["max_batch_seconds"])
            if eval_config.get("max_batch_seconds") is not None
            else None
        ),
    )
    references = [example.transcript for example in examples]

    aggregate_metrics = error_metrics(references, hypotheses)
    metrics = {
        "created_at": utc_now(),
        "config_path": str(config_path),
        "effective_config_path": str(effective_config_path),
        "checkpoint": str(checkpoint),
        "device": device,
        "datasets": [str(path) for path in dataset_dirs],
        "split": split,
        "examples": len(examples),
        **aggregate_metrics,
        "dataset_metrics": dataset_error_metrics(examples, references, hypotheses),
    }
    write_json(output_dir / "metrics.json", metrics)

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for index, (example, reference, hypothesis) in enumerate(zip(examples, references, hypotheses, strict=True), start=1):
            handle.write(
                json.dumps(
                    {
                        "id": index,
                        "audio_path": str(example.audio_path),
                        "dataset": str(example.dataset_dir or example.audio_path.parent),
                        "reference": reference,
                        "hypothesis": hypothesis,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    logging.info("wrote metrics=%s predictions=%s", output_dir / "metrics.json", predictions_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the FastConformer-CTC Persian model on configured dataset test.tsv files."
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML evaluation config path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional evaluation output directory override.")
    args = parser.parse_args(argv)
    return run_evaluation(args.config, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
