"""Fine-tune Whisper-small with Patched Multi-Condition Training (pMCT)."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from transformers import TrainerCallback

from ml.asr.train_whisper_small import (
    DEFAULT_CONFIG as BASE_DEFAULT_CONFIG,
    JsonMetricsCallback,
    WhisperDataCollator,
    WhisperDataset,
    WhisperExample,
    append_jsonl,
    build_training_arguments,
    configure_logging,
    deep_merge,
    filter_examples_by_label_length,
    latest_checkpoint,
    load_split_examples,
    model_max_target_positions,
    prepare_model_for_training,
    resolve_dataset_dir,
    resolve_pretrained_model,
    resolve_resume_checkpoint,
    resolve_run_dir,
    update_status,
    utc_now,
    validate_config as validate_base_config,
    word_error_rate,
    write_examples_manifest,
)
from ml.pmct.augmentation import PMCTExample, load_pmct_examples, mix_aligned_patches, patch_seed
from ml.utils.audio import load_audio, match_length, resample_audio


DEFAULT_CONFIG: dict[str, Any] = deep_merge(
    deepcopy(BASE_DEFAULT_CONFIG),
    {
        "data": {
            "datasets": [{"path": "cv-corpus-25.0-noise-added", "kind": "paired"}],
            "mapping_filename": "degraded_to_clean.jsonl",
        },
        "pmct": {
            "patch_seconds": 1.0,
            "clean_probability": 0.5,
        },
        "run": {
            "output_dir": "models/asr/pmct-whisper-small/runs",
            "name": "pmct-whisper-small-fa",
        },
    },
)


def validate_config(config: dict[str, Any]) -> None:
    data = config["data"]
    pmct = config["pmct"]
    if not isinstance(data.get("datasets"), list) or not data["datasets"]:
        raise ValueError("data.datasets must be a non-empty list")
    for entry in data["datasets"]:
        if isinstance(entry, str):
            if not entry.strip():
                raise ValueError("data.datasets cannot contain empty paths")
            continue
        if not isinstance(entry, dict) or not str(entry.get("path", "")).strip():
            raise ValueError("each data.datasets entry must be a path or {path, kind} mapping")
        if entry.get("kind", "auto") not in {"auto", "clean", "paired"}:
            raise ValueError("dataset kind must be auto, clean, or paired")
    # The shared validation covers model/training settings. Give it string paths
    # because the regular trainer intentionally knows nothing about typed specs.
    base_compatible = deepcopy(config)
    base_compatible["data"]["datasets"] = [
        str(entry["path"]) if isinstance(entry, dict) else str(entry)
        for entry in data["datasets"]
    ]
    validate_base_config(base_compatible)
    if not str(data.get("mapping_filename", "")).strip():
        raise ValueError("data.mapping_filename must be non-empty")
    if float(pmct.get("patch_seconds", 0)) <= 0:
        raise ValueError("pmct.patch_seconds must be > 0")
    probability = float(pmct.get("clean_probability", -1))
    if not 0 <= probability <= 1:
        raise ValueError("pmct.clean_probability must be in [0, 1]")


def load_training_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    config = deep_merge(DEFAULT_CONFIG, loaded)
    validate_config(config)
    return config


def resolve_dataset_specs(config: dict[str, Any]) -> list[tuple[Path, str]]:
    """Resolve dataset paths and classify them as clean or paired."""
    data = config["data"]
    root_dir = Path(str(data["root_dir"]))
    mapping_filename = str(data["mapping_filename"])
    specs: list[tuple[Path, str]] = []
    for entry in data["datasets"]:
        if isinstance(entry, dict):
            value = str(entry["path"])
            kind = str(entry.get("kind", "auto"))
        else:
            value = str(entry)
            kind = "auto"
        dataset_dir = resolve_dataset_dir(root_dir, value)
        if kind == "auto":
            kind = "paired" if (dataset_dir / mapping_filename).is_file() else "clean"
        specs.append((dataset_dir, kind))
    return specs


def _mono_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, source_rate = load_audio(path)
    return resample_audio(audio, source_rate, sample_rate)


class PMCTWhisperDataset:
    """Load clean examples normally and patch paired examples each epoch."""

    def __init__(
        self,
        examples: list[PMCTExample | WhisperExample],
        processor: Any,
        sample_rate: int,
        patch_seconds: float,
        clean_probability: float,
        seed: int,
    ) -> None:
        self.examples = examples
        self.processor = processor
        self.sample_rate = sample_rate
        self.patch_seconds = patch_seconds
        self.clean_probability = clean_probability
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.examples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        if isinstance(example, PMCTExample):
            degraded = _mono_audio(example.degraded_path, self.sample_rate)
            clean = _mono_audio(example.clean_path, self.sample_rate) * example.clean_scale
            length_delta = abs(len(clean) - len(degraded))
            if length_delta > 1:
                raise ValueError(
                    f"unaligned pMCT pair differs by {length_delta} samples: "
                    f"clean={example.clean_path} degraded={example.degraded_path}"
                )
            clean = match_length(clean, len(degraded))
            rng = np.random.default_rng(patch_seed(self.seed, self.epoch, example.degraded_path))
            waveform = mix_aligned_patches(
                clean,
                degraded,
                self.sample_rate,
                self.patch_seconds,
                self.clean_probability,
                rng,
            )
        else:
            waveform = _mono_audio(example.audio_path, self.sample_rate)
        features = self.processor.feature_extractor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        ).input_features[0]
        labels = self.processor.tokenizer(example.transcript).input_ids
        return {"input_features": features, "labels": labels}


class PMCTEpochCallback(TrainerCallback):
    def __init__(self, dataset: PMCTWhisperDataset) -> None:
        self.dataset = dataset

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.dataset.set_epoch(int(state.epoch or 0))


def write_pmct_manifest(path: Path, examples: list[PMCTExample | WhisperExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(examples, start=1):
            if isinstance(example, PMCTExample):
                payload = {
                    "id": index,
                    "kind": "paired",
                    "clean_path": str(example.clean_path),
                    "clean_scale": example.clean_scale,
                    "degraded_path": str(example.degraded_path),
                    "transcript": example.transcript,
                }
            else:
                payload = {
                    "id": index,
                    "kind": "clean",
                    "audio_path": str(example.audio_path),
                    "transcript": example.transcript,
                }
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def run_training(
    config_path: Path,
    run_dir_override: Path | None = None,
    resume_override: str | None = None,
) -> int:
    from transformers import Seq2SeqTrainer, WhisperForConditionalGeneration, WhisperProcessor, set_seed

    config = load_training_config(config_path)
    run_dir = resolve_run_dir(config, run_dir_override)
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(run_dir)
    effective_config_path = run_dir / "config" / "training.yaml"
    effective_config_path.parent.mkdir(parents=True, exist_ok=True)
    effective_config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
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
    set_seed(int(config["training"]["seed"]))

    try:
        model_config = config["model"]
        data_config = config["data"]
        training = config["training"]
        pretrained_model = resolve_pretrained_model(config, config_path)
        processor = WhisperProcessor.from_pretrained(
            pretrained_model,
            language=str(model_config.get("language", "Persian")),
            task=str(model_config.get("task", "transcribe")),
        )
        model = prepare_model_for_training(WhisperForConditionalGeneration.from_pretrained(pretrained_model))
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        max_label_tokens = model_max_target_positions(model)

        dataset_specs = resolve_dataset_specs(config)
        dataset_dirs = [path for path, _kind in dataset_specs]
        train_examples: list[PMCTExample | WhisperExample] = []
        eval_examples = []
        eval_splits: dict[str, str] = {}
        for dataset_dir, kind in dataset_specs:
            if (dataset_dir / "train.tsv").is_file():
                if kind == "paired":
                    loaded = load_pmct_examples(
                        [dataset_dir],
                        split="train",
                        mapping_filename=str(data_config["mapping_filename"]),
                        expected_sample_rate=int(data_config["sample_rate"]),
                    )
                else:
                    loaded = load_split_examples([dataset_dir], "train")
                train_examples.extend(loaded)
            if (dataset_dir / "dev.tsv").is_file():
                eval_examples.extend(load_split_examples([dataset_dir], "dev"))
                eval_splits[str(dataset_dir.resolve())] = "dev"
        train_examples, skipped_train = filter_examples_by_label_length(
            train_examples, processor.tokenizer, max_label_tokens
        )
        eval_examples, skipped_eval = filter_examples_by_label_length(
            eval_examples, processor.tokenizer, max_label_tokens
        )
        if not train_examples:
            raise ValueError("no pMCT training examples remain after label filtering")
        if not eval_examples:
            raise ValueError("no development examples found in the configured degraded datasets")
        paired_count = sum(isinstance(example, PMCTExample) for example in train_examples)
        clean_count = len(train_examples) - paired_count

        write_pmct_manifest(run_dir / "manifests" / "train_pmct_pairs.jsonl", train_examples)
        write_examples_manifest(run_dir / "manifests" / "dev.jsonl", eval_examples)
        update_status(
            run_dir,
            datasets=[str(path) for path in dataset_dirs],
            eval_splits=eval_splits,
            train_examples=len(train_examples),
            paired_train_examples=paired_count,
            clean_train_examples=clean_count,
            eval_examples=len(eval_examples),
            skipped_train_examples=len(skipped_train),
            skipped_eval_examples=len(skipped_eval),
            max_label_tokens=max_label_tokens,
            pmct=config["pmct"],
            pretrained_model=pretrained_model,
        )

        train_dataset = PMCTWhisperDataset(
            train_examples,
            processor,
            int(data_config["sample_rate"]),
            float(config["pmct"]["patch_seconds"]),
            float(config["pmct"]["clean_probability"]),
            int(training["seed"]),
        )
        eval_dataset = WhisperDataset(eval_examples, processor, int(data_config["sample_rate"]))
        args = build_training_arguments(config, run_dir)

        def compute_metrics(pred: Any) -> dict[str, float]:
            label_ids = pred.label_ids
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            predictions = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
            references = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            return {"wer": word_error_rate(references, predictions)}

        trainer_kwargs: dict[str, Any] = {
            "args": args,
            "model": model,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "data_collator": WhisperDataCollator(processor),
            "compute_metrics": compute_metrics,
            "callbacks": [JsonMetricsCallback(run_dir, metrics_path), PMCTEpochCallback(train_dataset)],
        }
        processor_arg = (
            "processing_class"
            if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters
            else "tokenizer"
        )
        trainer_kwargs[processor_arg] = processor.feature_extractor
        trainer = Seq2SeqTrainer(**trainer_kwargs)
        train_result = trainer.train(
            resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
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
        trainer.save_model(str(final_dir))
        processor.save_pretrained(str(final_dir))
        if trainer.state.best_model_checkpoint:
            best_checkpoint = Path(str(trainer.state.best_model_checkpoint))
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(best_checkpoint, best_dir)
            processor.save_pretrained(str(best_dir))
        latest = latest_checkpoint(run_dir)
        update_status(
            run_dir,
            status="completed",
            completed_at=utc_now(),
            latest_checkpoint=str(latest) if latest else None,
            best_checkpoint=str(best_dir) if best_dir.exists() else trainer.state.best_model_checkpoint,
            final_model=str(final_dir),
            error=None,
        )
        return 0
    except KeyboardInterrupt:
        latest = latest_checkpoint(run_dir)
        update_status(
            run_dir,
            status="interrupted",
            interrupted_at=utc_now(),
            latest_checkpoint=str(latest) if latest else None,
            error="Interrupted by user. Re-run with --resume auto to continue.",
        )
        raise
    except Exception as exc:
        logging.exception("pMCT training failed: %s", exc)
        latest = latest_checkpoint(run_dir)
        update_status(
            run_dir,
            status="failed",
            failed_at=utc_now(),
            latest_checkpoint=str(latest) if latest else None,
            error=str(exc),
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune Whisper-small with on-the-fly pMCT patches from a "
            "generate_noise_added_dataset clean/degraded mapping."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="pMCT training YAML path.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Optional run directory override.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume mode: auto, false, or an explicit checkpoint directory.",
    )
    args = parser.parse_args(argv)
    return run_training(args.config, args.run_dir, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
