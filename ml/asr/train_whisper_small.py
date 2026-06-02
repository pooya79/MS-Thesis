from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import re
import shutil
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jiwer
import soundfile as sf
import yaml
from transformers import TrainerCallback


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "name": "openai/whisper-small",
        "pretrained_model": None,
        "language": "Persian",
        "task": "transcribe",
    },
    "data": {
        "root_dir": "data",
        "datasets": ["cv-corpus-25.0"],
        "sample_rate": 16000,
    },
    "run": {
        "output_dir": "models/asr/whisper-small/runs",
        "name": None,
        "resume": "auto",
    },
    "training": {
        "seed": 1337,
        "num_train_epochs": 3,
        "learning_rate": 1e-5,
        "warmup_steps": 500,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "eval_steps": 500,
        "save_steps": 500,
        "logging_steps": 25,
        "save_total_limit": 3,
        "num_workers": 2,
        "device": "auto",
        "mixed_precision": "auto",
        "load_best_model_at_end": False,
        "generation_max_length": 225,
    },
}

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


@dataclass(frozen=True)
class WhisperExample:
    audio_path: Path
    transcript: str
    dataset_dir: Path | None = None


@dataclass(frozen=True)
class SkippedWhisperExample:
    example: WhisperExample
    token_count: int
    max_label_tokens: int
    reason: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_training_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    config = deep_merge(DEFAULT_CONFIG, loaded)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    model = config["model"]
    data = config["data"]
    training = config["training"]
    if not str(model["name"]).strip():
        raise ValueError("model.name must be a non-empty model id")
    pretrained_model = model.get("pretrained_model")
    if pretrained_model is not None and not str(pretrained_model).strip():
        raise ValueError("model.pretrained_model must be a non-empty model id or local path when set")
    datasets = data.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("data.datasets must be a non-empty list of dataset directory names")
    if any(not str(dataset).strip() for dataset in datasets):
        raise ValueError("data.datasets cannot contain empty values")
    minimums = {
        "sample_rate": (data, 8000),
        "num_train_epochs": (training, 0.01),
        "learning_rate": (training, 1e-8),
        "per_device_train_batch_size": (training, 1),
        "per_device_eval_batch_size": (training, 1),
        "gradient_accumulation_steps": (training, 1),
        "eval_steps": (training, 1),
        "save_steps": (training, 1),
        "logging_steps": (training, 1),
        "save_total_limit": (training, 1),
        "num_workers": (training, 0),
        "generation_max_length": (training, 1),
    }
    for key, (section, minimum) in minimums.items():
        if float(section[key]) < minimum:
            raise ValueError(f"{key} must be >= {minimum:g}")
    if training["mixed_precision"] not in {"auto", "true", "false", True, False}:
        raise ValueError("training.mixed_precision must be auto, true, or false")
    if training["device"] not in {"auto", "cuda", "cpu"}:
        raise ValueError("training.device must be auto, cuda, or cpu")
    if not isinstance(training["load_best_model_at_end"], bool):
        raise ValueError("training.load_best_model_at_end must be true or false")


def resolve_run_dir(config: dict[str, Any], override: Path | None = None) -> Path:
    if override is not None:
        return override
    run_config = config["run"]
    output_dir = Path(str(run_config["output_dir"]))
    name = run_config.get("name") or run_id()
    return output_dir / str(name)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def update_status(run_dir: Path, **updates: Any) -> None:
    status_path = run_dir / "status.json"
    status: dict[str, Any] = {}
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(updates)
    status["updated_at"] = utc_now()
    write_json(status_path, status)


def configure_logging(run_dir: Path) -> None:
    log_path = run_dir / "logs" / "train.log"
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


def resolve_dataset_dir(root_dir: Path, dataset_name: str) -> Path:
    dataset_path = Path(dataset_name)
    if dataset_path.is_absolute():
        return dataset_path
    direct_path = dataset_path
    if direct_path.exists():
        return direct_path
    return root_dir / dataset_path


def validate_dataset_dir(dataset_dir: Path, required_splits: tuple[str, ...] = ("train", "dev")) -> None:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_dir}")
    for split in required_splits:
        split_name = split if split.endswith(".tsv") else f"{split}.tsv"
        if not (dataset_dir / split_name).is_file():
            raise FileNotFoundError(f"dataset is missing {split_name}: {dataset_dir}")
    if not (dataset_dir / "clips").is_dir():
        raise FileNotFoundError(f"dataset is missing clips/: {dataset_dir}")


def resolve_audio_path(dataset_dir: Path, value: str) -> Path:
    raw_path = Path(value)
    if raw_path.is_absolute():
        return raw_path
    candidates = [
        dataset_dir / "clips" / raw_path,
        dataset_dir / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def iter_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"path", "sentence"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain path and sentence columns")
        return list(reader)


def load_split_examples(dataset_dirs: list[Path], split: str) -> list[WhisperExample]:
    examples: list[WhisperExample] = []
    for dataset_dir in dataset_dirs:
        logging.info("validating dataset=%s split=%s", dataset_dir, split)
        validate_dataset_dir(dataset_dir, (split,))
        split_path = dataset_dir / f"{split}.tsv"
        rows = iter_tsv_rows(split_path)
        logging.info("loading %s rows from %s", len(rows), split_path)
        for index, row in enumerate(rows, start=1):
            raw_audio = str(row.get("path", "")).strip()
            transcript = str(row.get("sentence", "")).strip()
            if not raw_audio or not transcript:
                logging.warning("%s:%s skipped empty path or sentence", split_path, index)
                continue
            audio_path = resolve_audio_path(dataset_dir, raw_audio)
            if not audio_path.exists():
                raise FileNotFoundError(f"{split_path}:{index} missing audio file: {audio_path}")
            examples.append(WhisperExample(audio_path=audio_path, transcript=transcript, dataset_dir=dataset_dir.resolve()))
    if not examples:
        names = ", ".join(str(path) for path in dataset_dirs)
        raise ValueError(f"no usable {split} examples found in: {names}")
    logging.info("loaded %s usable %s examples", len(examples), split)
    return examples


def resolve_dataset_dirs(config: dict[str, Any]) -> list[Path]:
    data = config["data"]
    root_dir = Path(str(data["root_dir"]))
    return [resolve_dataset_dir(root_dir, str(dataset)) for dataset in data["datasets"]]


def write_examples_manifest(path: Path, examples: list[WhisperExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(examples, start=1):
            handle.write(
                json.dumps(
                    {
                        "id": index,
                        "audio_path": str(example.audio_path),
                        "transcript": example.transcript,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def write_skipped_examples_manifest(path: Path, skipped: list[SkippedWhisperExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, skipped_example in enumerate(skipped, start=1):
            example = skipped_example.example
            handle.write(
                json.dumps(
                    {
                        "id": index,
                        "audio_path": str(example.audio_path),
                        "dataset": str(example.dataset_dir or example.audio_path.parent),
                        "transcript": example.transcript,
                        "token_count": skipped_example.token_count,
                        "max_label_tokens": skipped_example.max_label_tokens,
                        "reason": skipped_example.reason,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def model_max_target_positions(model: Any) -> int | None:
    value = getattr(getattr(model, "config", None), "max_target_positions", None)
    return int(value) if value is not None else None


def filter_examples_by_label_length(
    examples: list[WhisperExample],
    tokenizer: Any,
    max_label_tokens: int | None,
) -> tuple[list[WhisperExample], list[SkippedWhisperExample]]:
    if max_label_tokens is None:
        return examples, []
    kept: list[WhisperExample] = []
    skipped: list[SkippedWhisperExample] = []
    for example in examples:
        token_count = len(tokenizer(example.transcript).input_ids)
        if token_count > max_label_tokens:
            skipped.append(
                SkippedWhisperExample(
                    example=example,
                    token_count=token_count,
                    max_label_tokens=max_label_tokens,
                    reason="label_token_length_exceeds_model_limit",
                )
            )
            continue
        kept.append(example)
    return kept, skipped


def word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    return float(jiwer.wer(references, hypotheses))


def character_error_rate(references: list[str], hypotheses: list[str]) -> float:
    return float(jiwer.cer(references, hypotheses))


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.exists():
        return None
    checkpoints = [
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir() and CHECKPOINT_RE.match(path.name)
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=checkpoint_step)


def resolve_resume_checkpoint(run_dir: Path, resume: str | Path | bool | None) -> Path | None:
    if resume in {None, False, "false", "none", "off"}:
        return None
    if resume in {True, "true", "auto"}:
        return latest_checkpoint(run_dir)
    checkpoint = Path(str(resume))
    if not checkpoint.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
    return checkpoint


def resolve_pretrained_model(config: dict[str, Any], config_path: Path | None = None) -> str:
    model = config["model"]
    raw_source = model.get("pretrained_model") or model["name"]
    source = str(raw_source).strip()
    source_path = Path(source).expanduser()
    candidates: list[Path] = []
    if source_path.is_absolute():
        candidates.append(source_path)
    else:
        candidates.append(source_path)
        if config_path is not None:
            candidates.append(config_path.parent / source_path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return source


class JsonMetricsCallback(TrainerCallback):
    def __init__(self, run_dir: Path, metrics_path: Path) -> None:
        self.run_dir = run_dir
        self.metrics_path = metrics_path

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not logs:
            return
        row = {
            "timestamp": utc_now(),
            "step": int(getattr(state, "global_step", 0)),
            "epoch": getattr(state, "epoch", None),
        }
        row.update({key: value for key, value in logs.items() if isinstance(value, int | float | str | bool) or value is None})
        append_jsonl(self.metrics_path, row)

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        checkpoint = Path(str(args.output_dir)) / f"checkpoint-{int(getattr(state, 'global_step', 0))}"
        if checkpoint.exists():
            update_status(self.run_dir, latest_checkpoint=str(checkpoint))


def build_training_arguments(config: dict[str, Any], run_dir: Path) -> Any:
    from transformers import Seq2SeqTrainingArguments

    training = config["training"]
    output_dir = run_dir / "checkpoints"
    device = str(training["device"])
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("training.device is cuda, but CUDA is not available")
    mixed_precision = training["mixed_precision"]
    if mixed_precision == "auto":
        import torch

        fp16 = torch.cuda.is_available() and device != "cpu"
    else:
        fp16 = mixed_precision in {True, "true"} and device != "cpu"
    common_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(training["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "learning_rate": float(training["learning_rate"]),
        "warmup_steps": int(training["warmup_steps"]),
        "num_train_epochs": float(training["num_train_epochs"]),
        "logging_steps": int(training["logging_steps"]),
        "eval_steps": int(training["eval_steps"]),
        "save_steps": int(training["save_steps"]),
        "save_total_limit": int(training["save_total_limit"]),
        "predict_with_generate": True,
        "generation_max_length": int(training["generation_max_length"]),
        "fp16": fp16,
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": int(training["num_workers"]),
        "load_best_model_at_end": bool(training["load_best_model_at_end"]),
        "metric_for_best_model": "wer",
        "greater_is_better": False,
    }
    argument_names = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if device == "cpu":
        common_kwargs["use_cpu" if "use_cpu" in argument_names else "no_cuda"] = True
    try:
        return Seq2SeqTrainingArguments(
            **common_kwargs,
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
        )
    except TypeError:
        return Seq2SeqTrainingArguments(
            **common_kwargs,
            evaluation_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
        )


class WhisperDataset:
    def __init__(self, examples: list[WhisperExample], processor: Any, sample_rate: int) -> None:
        self.examples = examples
        self.processor = processor
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        import torchaudio.functional as F

        example = self.examples[index]
        audio, source_rate = sf.read(str(example.audio_path), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        waveform = torch.as_tensor(audio, dtype=torch.float32)
        if int(source_rate) != self.sample_rate:
            waveform = F.resample(waveform, int(source_rate), self.sample_rate)
        features = self.processor.feature_extractor(
            waveform.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        ).input_features[0]
        labels = self.processor.tokenizer(example.transcript).input_ids
        return {"input_features": features, "labels": labels}


class WhisperDataCollator:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def run_training(config_path: Path, run_dir_override: Path | None = None, resume_override: str | None = None) -> int:
    from transformers import Seq2SeqTrainer, WhisperForConditionalGeneration, WhisperProcessor, set_seed

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("loading config=%s", config_path)
    config = load_training_config(config_path)
    run_dir = resolve_run_dir(config, run_dir_override)
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(run_dir)
    logging.info("configured logging file=%s", run_dir / "logs" / "train.log")
    effective_config_path = run_dir / "config" / "training.yaml"
    effective_config_path.parent.mkdir(parents=True, exist_ok=True)
    effective_config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    logging.info("wrote effective config=%s", effective_config_path)

    metrics_path = run_dir / "logs" / "train_metrics.jsonl"
    resume_value = resume_override if resume_override is not None else config["run"].get("resume")
    resume_checkpoint = resolve_resume_checkpoint(run_dir, resume_value)
    update_status(
        run_dir,
        run_id=run_dir.name,
        status="running",
        started_at=utc_now(),
        config_path=str(config_path),
        effective_config_path=str(effective_config_path),
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,
        error=None,
    )

    logging.info("run_dir=%s", run_dir)
    logging.info("resume_from_checkpoint=%s", resume_checkpoint or "none")
    logging.info("setting seed=%s", config["training"]["seed"])
    set_seed(int(config["training"]["seed"]))

    try:
        model_config = config["model"]
        data_config = config["data"]
        pretrained_model = resolve_pretrained_model(config, config_path)
        logging.info(
            "loading processor model=%s pretrained_model=%s language=%s task=%s",
            model_config["name"],
            pretrained_model,
            model_config.get("language", "Persian"),
            model_config.get("task", "transcribe"),
        )
        processor = WhisperProcessor.from_pretrained(
            pretrained_model,
            language=str(model_config.get("language", "Persian")),
            task=str(model_config.get("task", "transcribe")),
        )

        logging.info("loading model=%s", pretrained_model)
        model = WhisperForConditionalGeneration.from_pretrained(pretrained_model)
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        max_label_tokens = model_max_target_positions(model)
        logging.info("max_label_tokens=%s", max_label_tokens or "unbounded")

        logging.info("resolving dataset directories root=%s datasets=%s", data_config["root_dir"], data_config["datasets"])
        dataset_dirs = resolve_dataset_dirs(config)
        logging.info("resolved dataset directories=%s", ", ".join(str(path) for path in dataset_dirs))
        logging.info("loading training examples")
        train_examples = load_split_examples(dataset_dirs, "train")
        logging.info("loading evaluation examples")
        eval_examples = load_split_examples(dataset_dirs, "dev")
        train_examples, skipped_train_examples = filter_examples_by_label_length(train_examples, processor.tokenizer, max_label_tokens)
        eval_examples, skipped_eval_examples = filter_examples_by_label_length(eval_examples, processor.tokenizer, max_label_tokens)
        if not train_examples:
            raise ValueError("no train examples remain after label length filtering")
        if not eval_examples:
            raise ValueError("no dev examples remain after label length filtering")
        skipped_count = len(skipped_train_examples) + len(skipped_eval_examples)
        if skipped_count:
            logging.warning(
                "skipped %s examples with label token length above %s",
                skipped_count,
                max_label_tokens,
            )
        logging.info("writing source manifests")
        write_examples_manifest(run_dir / "manifests" / "train.jsonl", train_examples)
        write_examples_manifest(run_dir / "manifests" / "dev.jsonl", eval_examples)
        write_skipped_examples_manifest(run_dir / "manifests" / "skipped_train.jsonl", skipped_train_examples)
        write_skipped_examples_manifest(run_dir / "manifests" / "skipped_dev.jsonl", skipped_eval_examples)
        update_status(
            run_dir,
            datasets=[str(path) for path in dataset_dirs],
            train_examples=len(train_examples),
            eval_examples=len(eval_examples),
            skipped_train_examples=len(skipped_train_examples),
            skipped_eval_examples=len(skipped_eval_examples),
            max_label_tokens=max_label_tokens,
        )
        logging.info("datasets=%s", ", ".join(str(path) for path in dataset_dirs))
        logging.info("train_examples=%s eval_examples=%s", len(train_examples), len(eval_examples))
        logging.info("building datasets with on-demand feature computation")
        train_dataset = WhisperDataset(train_examples, processor, int(data_config["sample_rate"]))
        eval_dataset = WhisperDataset(eval_examples, processor, int(data_config["sample_rate"]))
        update_status(run_dir, pretrained_model=pretrained_model)
        logging.info("building training arguments")
        args = build_training_arguments(config, run_dir)

        def compute_metrics(pred: Any) -> dict[str, float]:
            pred_ids = pred.predictions
            label_ids = pred.label_ids
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            predictions = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            references = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            return {"wer": word_error_rate(references, predictions)}

        trainer_kwargs = {
            "args": args,
            "model": model,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "data_collator": WhisperDataCollator(processor),
            "compute_metrics": compute_metrics,
            "callbacks": [JsonMetricsCallback(run_dir, metrics_path)],
        }
        processor_arg = "processing_class" if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters else "tokenizer"
        trainer_kwargs[processor_arg] = processor.feature_extractor
        logging.info("initializing trainer train_items=%s eval_items=%s", len(train_dataset), len(eval_dataset))
        trainer = Seq2SeqTrainer(**trainer_kwargs)
        logging.info("starting training resume_from_checkpoint=%s", resume_checkpoint or "none")
        train_result = trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
        logging.info(
            "training finished global_step=%s epoch=%s training_loss=%s",
            trainer.state.global_step,
            trainer.state.epoch,
            train_result.training_loss,
        )
        append_jsonl(
            metrics_path,
            {
                "timestamp": utc_now(),
                "step": int(trainer.state.global_step),
                "epoch": trainer.state.epoch,
                "train_loss": float(train_result.training_loss),
            },
        )

        final_dir = run_dir / "final"
        best_dir = run_dir / "best"
        logging.info("saving final model=%s", final_dir)
        trainer.save_model(str(final_dir))
        processor.save_pretrained(str(final_dir))
        if trainer.state.best_model_checkpoint:
            best_checkpoint = Path(str(trainer.state.best_model_checkpoint))
            logging.info("copying best checkpoint=%s to %s", best_checkpoint, best_dir)
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(best_checkpoint, best_dir)
            processor.save_pretrained(str(best_dir))
        update_status(
            run_dir,
            status="completed",
            completed_at=utc_now(),
            latest_checkpoint=str(latest_checkpoint(run_dir)) if latest_checkpoint(run_dir) else None,
            best_checkpoint=str(best_dir) if best_dir.exists() else trainer.state.best_model_checkpoint,
            final_model=str(final_dir),
            error=None,
        )
        logging.info("run completed final_model=%s best_model=%s", final_dir, best_dir if best_dir.exists() else trainer.state.best_model_checkpoint)
        return 0
    except KeyboardInterrupt:
        logging.warning("training interrupted by user")
        update_status(
            run_dir,
            status="interrupted",
            interrupted_at=utc_now(),
            latest_checkpoint=str(latest_checkpoint(run_dir)) if latest_checkpoint(run_dir) else None,
            error="Interrupted by user. Re-run with run.resume: auto or --resume auto to continue.",
        )
        raise
    except Exception as exc:
        logging.exception("training failed: %s", exc)
        update_status(run_dir, status="failed", failed_at=utc_now(), latest_checkpoint=str(latest_checkpoint(run_dir)) if latest_checkpoint(run_dir) else None, error=str(exc))
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Whisper-small from a YAML config. Stop with Ctrl+C after checkpoints "
            "exist, then resume by re-running with run.resume: auto or --resume auto."
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
