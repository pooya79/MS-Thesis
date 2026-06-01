from __future__ import annotations

import argparse
import inspect
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ml.asr.train_whisper_small import (
    WhisperDataCollator,
    WhisperDataset,
    deep_merge,
    load_split_examples,
    resolve_dataset_dirs,
    word_error_rate,
    write_examples_manifest,
)


DEFAULT_EVAL_CONFIG: dict[str, Any] = {
    "model": {
        "checkpoint": None,
        "processor": "openai/whisper-small",
        "language": "Persian",
        "task": "transcribe",
    },
    "data": {
        "root_dir": "data",
        "datasets": ["cv-corpus-25.0"],
        "sample_rate": 16000,
        "split": "test",
    },
    "eval": {
        "output_dir": "models/asr/whisper-small/evals",
        "name": None,
        "batch_size": 4,
        "num_workers": 2,
        "device": "auto",
        "generation_max_length": 225,
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
        raise ValueError("model.checkpoint must be a non-empty local model or checkpoint path")
    processor = model.get("processor")
    if processor is not None and not str(processor).strip():
        raise ValueError("model.processor must be a non-empty model id or local path when set")
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
        "num_workers": (eval_config, 0),
        "generation_max_length": (eval_config, 1),
    }
    for key, (section, minimum) in minimums.items():
        if float(section[key]) < minimum:
            raise ValueError(f"{key} must be >= {minimum:g}")
    if eval_config["device"] not in {"auto", "cuda", "cpu"}:
        raise ValueError("eval.device must be auto, cuda, or cpu")


def resolve_existing_path(raw_path: str | Path, config_path: Path | None = None) -> Path:
    source_path = Path(str(raw_path)).expanduser()
    candidates = [source_path] if source_path.is_absolute() else [source_path]
    if config_path is not None and not source_path.is_absolute():
        candidates.append(config_path.parent / source_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"path does not exist: {raw_path}")


def resolve_processor_source(raw_source: str | Path, config_path: Path | None = None) -> str:
    source = str(raw_source).strip()
    source_path = Path(source).expanduser()
    candidates = [source_path] if source_path.is_absolute() else [source_path]
    if config_path is not None and not source_path.is_absolute():
        candidates.append(config_path.parent / source_path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return source


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


def build_eval_arguments(config: dict[str, Any], output_dir: Path) -> Any:
    from transformers import Seq2SeqTrainingArguments

    eval_config = config["eval"]
    device = str(eval_config["device"])
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("eval.device is cuda, but CUDA is not available")
    common_kwargs = {
        "output_dir": str(output_dir / "trainer"),
        "per_device_eval_batch_size": int(eval_config["batch_size"]),
        "predict_with_generate": True,
        "generation_max_length": int(eval_config["generation_max_length"]),
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": int(eval_config["num_workers"]),
    }
    argument_names = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if device == "cpu":
        common_kwargs["use_cpu" if "use_cpu" in argument_names else "no_cuda"] = True
    return Seq2SeqTrainingArguments(**common_kwargs)


def run_evaluation(config_path: Path, output_dir_override: Path | None = None) -> int:
    from transformers import Seq2SeqTrainer, WhisperForConditionalGeneration, WhisperProcessor

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
    checkpoint = resolve_existing_path(str(model_config["checkpoint"]), config_path)
    processor_source = model_config.get("processor")
    processor_name = resolve_processor_source(
        str(processor_source or DEFAULT_EVAL_CONFIG["model"]["processor"]),
        config_path,
    )

    logging.info("loading processor=%s checkpoint=%s", processor_name, checkpoint)
    try:
        processor = WhisperProcessor.from_pretrained(
            processor_name,
            language=str(model_config.get("language", "Persian")),
            task=str(model_config.get("task", "transcribe")),
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Could not load the Whisper processor. If model.checkpoint points to a "
            "Trainer checkpoint such as checkpoints/checkpoint-20000, set "
            "model.processor to openai/whisper-small or to a saved final/best model "
            "directory that includes tokenizer and processor files."
        ) from exc
    model = WhisperForConditionalGeneration.from_pretrained(str(checkpoint))
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    dataset_dirs = resolve_dataset_dirs(config)
    split = str(data_config["split"])
    examples = load_split_examples(dataset_dirs, split)
    write_examples_manifest(output_dir / "manifests" / f"{split}.jsonl", examples)

    dataset = WhisperDataset(examples, processor, int(data_config["sample_rate"]))
    args = build_eval_arguments(config, output_dir)
    trainer_kwargs = {
        "args": args,
        "model": model,
        "eval_dataset": dataset,
        "data_collator": WhisperDataCollator(processor),
    }
    processor_arg = "processing_class" if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters else "tokenizer"
    trainer_kwargs[processor_arg] = processor.feature_extractor
    trainer = Seq2SeqTrainer(**trainer_kwargs)
    prediction = trainer.predict(dataset)

    label_ids = prediction.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    hypotheses = processor.tokenizer.batch_decode(prediction.predictions, skip_special_tokens=True)
    references = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    metrics = {
        "created_at": utc_now(),
        "config_path": str(config_path),
        "effective_config_path": str(effective_config_path),
        "checkpoint": str(checkpoint),
        "processor": processor_name,
        "datasets": [str(path) for path in dataset_dirs],
        "split": split,
        "examples": len(examples),
        "wer": word_error_rate(references, hypotheses),
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
        description="Evaluate a Whisper-small checkpoint on configured dataset test.tsv files."
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML evaluation config path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional evaluation output directory override.")
    args = parser.parse_args(argv)
    return run_evaluation(args.config, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
