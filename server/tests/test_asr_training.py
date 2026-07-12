from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from ml.asr.train_whisper_small import (
    JsonMetricsCallback,
    WhisperDataset,
    WhisperExample,
    character_error_rate,
    filter_examples_by_label_length,
    latest_checkpoint,
    load_split_examples,
    load_training_config,
    prepare_model_for_training,
    resolve_pretrained_model,
    resolve_resume_checkpoint,
    resolve_run_dir,
    resolve_dataset_dirs,
    word_error_rate,
)
from ml.asr.eval_whisper_small import (
    build_eval_arguments,
    dataset_error_metrics,
    error_metrics,
    load_eval_config,
    resolve_output_dir,
    resolve_processor_source,
)
from ml.asr.train_whisper_large_v3_turbo import (
    load_training_config as load_large_v3_turbo_training_config,
)
from ml.asr.eval_whisper_large_v3_turbo import (
    load_eval_config as load_large_v3_turbo_eval_config,
)


def write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def write_dataset(root: Path, name: str) -> Path:
    dataset_dir = root / name
    clips_dir = dataset_dir / "clips"
    clips_dir.mkdir(parents=True)
    (clips_dir / "train.wav").write_bytes(b"audio")
    (clips_dir / "dev.wav").write_bytes(b"audio")
    for split in ("train", "dev"):
        with (dataset_dir / f"{split}.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "sentence"], delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerow({"path": f"{split}.wav", "sentence": f"{split} text"})
    return dataset_dir


def write_audio_dataset(root: Path, name: str) -> Path:
    dataset_dir = root / name
    clips_dir = dataset_dir / "clips"
    clips_dir.mkdir(parents=True)
    sf.write(clips_dir / "train.wav", np.zeros(160, dtype=np.float32), 16000)
    sf.write(clips_dir / "dev.wav", np.zeros(160, dtype=np.float32), 16000)
    for split in ("train", "dev"):
        with (dataset_dir / f"{split}.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "sentence"], delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerow({"path": f"{split}.wav", "sentence": f"{split} text"})
    return dataset_dir


def write_test_only_dataset(root: Path, name: str) -> Path:
    dataset_dir = root / name
    clips_dir = dataset_dir / "clips"
    clips_dir.mkdir(parents=True)
    (clips_dir / "test.wav").write_bytes(b"audio")
    with (dataset_dir / "test.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sentence"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow({"path": "test.wav", "sentence": "test text"})
    return dataset_dir


class FakeFeatureExtractor:
    def __call__(self, audio, *, sampling_rate: int, return_tensors: str):
        assert sampling_rate == 16000
        assert return_tensors == "pt"
        return SimpleNamespace(input_features=torch.ones((1, 2, 3), dtype=torch.float32))


class FakeTokenizer:
    def __call__(self, text: str):
        return SimpleNamespace(input_ids=list(range(len(text))))


class FakeProcessor:
    feature_extractor = FakeFeatureExtractor()
    tokenizer = FakeTokenizer()


def test_load_training_config_merges_yaml_with_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    write_yaml(
        config_path,
        {
            "data": {
                "root_dir": str(tmp_path / "data"),
                "datasets": ["cv-corpus-25.0"],
            },
            "run": {
                "output_dir": str(tmp_path / "runs"),
                "name": "smoke",
            },
            "training": {
                "num_train_epochs": 1,
                "per_device_train_batch_size": 2,
            },
        },
    )

    config = load_training_config(config_path)

    assert config["model"]["name"] == "openai/whisper-small"
    assert config["model"]["pretrained_model"] is None
    assert resolve_pretrained_model(config, config_path) == "openai/whisper-small"
    assert config["data"]["sample_rate"] == 16000
    assert config["data"]["datasets"] == ["cv-corpus-25.0"]
    assert config["training"]["num_train_epochs"] == 1
    assert config["training"]["learning_rate"] == 1e-5
    assert config["training"]["device"] == "auto"
    assert config["training"]["load_best_model_at_end"] is False
    assert resolve_run_dir(config) == tmp_path / "runs" / "smoke"
    assert resolve_dataset_dirs(config) == [tmp_path / "data" / "cv-corpus-25.0"]


def test_large_v3_turbo_training_config_uses_model_specific_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    write_yaml(config_path, {"data": {"datasets": ["cv-corpus-25.0"]}})

    config = load_large_v3_turbo_training_config(config_path)

    assert config["model"]["name"] == "openai/whisper-large-v3-turbo"
    assert config["run"]["output_dir"] == "models/asr/whisper-large-v3-turbo/runs"
    assert config["training"]["gradient_checkpointing"] is True
    assert config["training"]["per_device_train_batch_size"] == 1


def test_prepare_model_for_training_converts_half_parameters_to_float() -> None:
    model = torch.nn.Linear(3, 2).half()

    prepared = prepare_model_for_training(model)

    assert prepared is model
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_resolve_pretrained_model_uses_existing_local_path(tmp_path: Path) -> None:
    local_model = tmp_path / "models" / "asr" / "whisper-small" / "runs" / "run-1" / "final"
    local_model.mkdir(parents=True)
    config_path = tmp_path / "configs" / "train.yaml"
    config_path.parent.mkdir()
    write_yaml(
        config_path,
        {
            "model": {
                "pretrained_model": "../models/asr/whisper-small/runs/run-1/final",
            },
            "data": {
                "datasets": ["cv-corpus-25.0"],
            },
        },
    )
    config = load_training_config(config_path)

    assert resolve_pretrained_model(config, config_path) == str(local_model.resolve())


def test_load_training_config_rejects_empty_pretrained_model(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    write_yaml(
        config_path,
        {
            "model": {
                "pretrained_model": "",
            },
            "data": {
                "datasets": ["cv-corpus-25.0"],
            },
        },
    )

    with pytest.raises(ValueError, match="model.pretrained_model"):
        load_training_config(config_path)


def test_load_training_config_validates_minimum_values(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    write_yaml(
        config_path,
        {
            "data": {
                "datasets": ["cv-corpus-25.0"],
            },
            "training": {
                "save_steps": 0,
            },
        },
    )

    with pytest.raises(ValueError, match="save_steps"):
        load_training_config(config_path)


def test_word_error_rate_uses_jiwer_for_insertions_deletions_and_substitutions() -> None:
    assert word_error_rate(["سلام دنیا"], ["سلام"]) == 0.5
    assert word_error_rate(["سلام دنیا"], ["سلام امروز"]) == 0.5
    assert word_error_rate(["سلام دنیا"], ["سلام دنیا اضافه"]) == 0.5


def test_character_error_rate_uses_jiwer_for_character_level_edits() -> None:
    assert character_error_rate(["ab"], ["ac"]) == 0.5
    assert error_metrics(["ab"], ["ac"]) == {"wer": 1.0, "cer": 0.5}


def test_dataset_error_metrics_groups_results_by_dataset_directory(tmp_path: Path) -> None:
    first_dataset = tmp_path / "data" / "first"
    second_dataset = tmp_path / "data" / "second"
    examples = [
        WhisperExample(audio_path=first_dataset / "clips" / "one.wav", transcript="ab", dataset_dir=first_dataset),
        WhisperExample(audio_path=first_dataset / "clips" / "two.wav", transcript="cd", dataset_dir=first_dataset),
        WhisperExample(audio_path=second_dataset / "clips" / "three.wav", transcript="ef", dataset_dir=second_dataset),
    ]

    metrics = dataset_error_metrics(
        examples,
        references=["ab", "cd", "ef"],
        hypotheses=["ac", "cd", "eg"],
    )

    assert metrics == [
        {
            "dataset": str(first_dataset),
            "examples": 2,
            "wer": 0.5,
            "cer": 0.25,
        },
        {
            "dataset": str(second_dataset),
            "examples": 1,
            "wer": 1.0,
            "cer": 0.5,
        },
    ]


def test_latest_checkpoint_uses_largest_checkpoint_step(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints" / "checkpoint-2").mkdir(parents=True)
    (run_dir / "checkpoints" / "checkpoint-10").mkdir()
    (run_dir / "checkpoints" / "notes").mkdir()

    assert latest_checkpoint(run_dir) == run_dir / "checkpoints" / "checkpoint-10"
    assert resolve_resume_checkpoint(run_dir, "auto") == run_dir / "checkpoints" / "checkpoint-10"
    assert resolve_resume_checkpoint(run_dir, "false") is None

    with pytest.raises(FileNotFoundError, match="resume checkpoint"):
        resolve_resume_checkpoint(run_dir, run_dir / "checkpoints" / "missing")


def test_json_metrics_callback_supports_trainer_lifecycle_hooks(tmp_path: Path) -> None:
    callback = JsonMetricsCallback(tmp_path, tmp_path / "logs" / "train_metrics.jsonl")
    control = SimpleNamespace()

    callback.on_init_end(None, SimpleNamespace(), control)

    callback.on_log(
        SimpleNamespace(),
        SimpleNamespace(global_step=12, epoch=0.5),
        control,
        logs={"loss": 1.25, "ignored": {"nested": True}},
    )

    rows = (tmp_path / "logs" / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    row = yaml.safe_load(rows[0])
    assert row["step"] == 12
    assert row["epoch"] == 0.5
    assert row["loss"] == 1.25
    assert "ignored" not in row


def test_load_split_examples_reads_train_and_dev_tsv_from_dataset_dirs(tmp_path: Path) -> None:
    dataset_dir = write_dataset(tmp_path / "data", "cv-corpus-25.0")

    train_examples = load_split_examples([dataset_dir], "train")
    dev_examples = load_split_examples([dataset_dir], "dev")

    assert train_examples[0].audio_path == (dataset_dir / "clips" / "train.wav").resolve()
    assert train_examples[0].transcript == "train text"
    assert dev_examples[0].audio_path == (dataset_dir / "clips" / "dev.wav").resolve()
    assert dev_examples[0].transcript == "dev text"


def test_load_split_examples_reads_test_tsv_without_train_or_dev(tmp_path: Path) -> None:
    dataset_dir = write_test_only_dataset(tmp_path / "data", "cv-test")

    test_examples = load_split_examples([dataset_dir], "test")

    assert test_examples[0].audio_path == (dataset_dir / "clips" / "test.wav").resolve()
    assert test_examples[0].transcript == "test text"


def test_filter_examples_by_label_length_skips_overlong_transcripts(tmp_path: Path) -> None:
    examples = [
        WhisperExample(audio_path=tmp_path / "clips" / "short.wav", transcript="short", dataset_dir=tmp_path),
        WhisperExample(audio_path=tmp_path / "clips" / "long.wav", transcript="too long", dataset_dir=tmp_path),
    ]

    kept, skipped = filter_examples_by_label_length(examples, FakeTokenizer(), max_label_tokens=5)

    assert kept == [examples[0]]
    assert skipped[0].example == examples[1]
    assert skipped[0].token_count == 8
    assert skipped[0].max_label_tokens == 5
    assert skipped[0].reason == "label_token_length_exceeds_model_limit"


def test_load_eval_config_merges_yaml_with_defaults_and_resolves_processor(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "run-1" / "final"
    processor = tmp_path / "runs" / "run-1" / "processor"
    checkpoint.mkdir(parents=True)
    processor.mkdir(parents=True)
    config_path = tmp_path / "configs" / "eval.yaml"
    config_path.parent.mkdir()
    write_yaml(
        config_path,
        {
            "model": {
                "checkpoint": "../runs/run-1/final",
                "processor": "../runs/run-1/processor",
            },
            "data": {
                "root_dir": str(tmp_path / "data"),
                "datasets": ["cv-test"],
            },
            "eval": {
                "output_dir": str(tmp_path / "evals"),
                "name": "smoke",
                "batch_size": 2,
                "max_label_tokens": 448,
                "eval_accumulation_steps": 2,
            },
        },
    )

    config = load_eval_config(config_path)

    assert config["data"]["split"] == "test"
    assert config["eval"]["generation_max_length"] == 225
    assert config["eval"]["max_label_tokens"] == 448
    assert config["eval"]["eval_accumulation_steps"] == 2
    assert resolve_output_dir(config) == tmp_path / "evals" / "smoke"
    assert resolve_processor_source(config["model"]["processor"], config_path) == str(processor.resolve())
    assert resolve_processor_source("openai/whisper-small", config_path) == "openai/whisper-small"


def test_load_eval_config_requires_checkpoint(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.yaml"
    write_yaml(config_path, {"data": {"datasets": ["cv-test"]}})

    with pytest.raises(ValueError, match="model.checkpoint"):
        load_eval_config(config_path)


def test_large_v3_turbo_eval_config_uses_model_specific_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.yaml"
    write_yaml(
        config_path,
        {
            "model": {"checkpoint": "a-model"},
            "data": {"datasets": ["cv-test"]},
        },
    )

    config = load_large_v3_turbo_eval_config(config_path)

    assert config["model"]["processor"] == "openai/whisper-large-v3-turbo"
    assert config["eval"]["output_dir"] == "models/asr/whisper-large-v3-turbo/evals"
    assert config["eval"]["batch_size"] == 1


def test_build_eval_arguments_flushes_predictions_during_eval(tmp_path: Path) -> None:
    config = {
        "eval": {
            "batch_size": 1,
            "num_workers": 0,
            "device": "cpu",
            "generation_max_length": 225,
            "eval_accumulation_steps": 1,
        }
    }

    args = build_eval_arguments(config, tmp_path)

    assert args.eval_accumulation_steps == 1


def test_whisper_dataset_computes_features_in_getitem(tmp_path: Path) -> None:
    dataset_dir = write_audio_dataset(tmp_path / "data", "cv-corpus-25.0")
    examples = load_split_examples([dataset_dir], "train")

    item = WhisperDataset(examples, FakeProcessor(), 16000)[0]

    assert torch.equal(item["input_features"], torch.ones((2, 3), dtype=torch.float32))
    assert item["labels"] == list(range(10))


def test_load_split_examples_rejects_missing_audio(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "data" / "cv-corpus-25.0"
    (dataset_dir / "clips").mkdir(parents=True)
    for split in ("train", "dev"):
        with (dataset_dir / f"{split}.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "sentence"], delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerow({"path": "missing.wav", "sentence": "missing"})

    with pytest.raises(FileNotFoundError, match="missing audio file"):
        load_split_examples([dataset_dir], "train")


def test_load_training_config_requires_dataset_list(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    write_yaml(config_path, {"data": {"datasets": []}})

    with pytest.raises(ValueError, match="data.datasets"):
        load_training_config(config_path)


def test_load_training_config_validates_device(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    write_yaml(
        config_path,
        {
            "data": {
                "datasets": ["cv-corpus-25.0"],
            },
            "training": {
                "device": "tpu",
            },
        },
    )

    with pytest.raises(ValueError, match="training.device"):
        load_training_config(config_path)


def test_load_training_config_validates_load_best_model_flag(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    write_yaml(
        config_path,
        {
            "data": {
                "datasets": ["cv-corpus-25.0"],
            },
            "training": {
                "load_best_model_at_end": "yes",
            },
        },
    )

    with pytest.raises(ValueError, match="load_best_model_at_end"):
        load_training_config(config_path)


def test_resumable_sampler_order_depends_only_on_seed_and_epoch() -> None:
    from ml.asr.train_fastconformer import ResumableRandomSampler

    sampler = ResumableRandomSampler(num_samples=20, seed=1337)

    sampler.set_epoch(0)
    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)

    # A full permutation each epoch, and consecutive epochs differ in order.
    assert sorted(epoch0) == list(range(20))
    assert sorted(epoch1) == list(range(20))
    assert epoch0 != epoch1

    # Re-deriving the same epoch reproduces the exact order — this is what lets a
    # resumed run replay the order the crashed run was iterating.
    other = ResumableRandomSampler(num_samples=20, seed=1337)
    other.set_epoch(0)
    assert list(other) == epoch0


def test_resumable_sampler_skip_drops_already_seen_prefix() -> None:
    from ml.asr.train_fastconformer import ResumableRandomSampler

    sampler = ResumableRandomSampler(num_samples=20, seed=7)
    sampler.set_epoch(3)
    full = list(sampler)

    # Resuming after 5 samples skips exactly that prefix; the tail is identical.
    sampler.set_epoch(3, skip=5)
    assert list(sampler) == full[5:]
    assert len(sampler) == 15

    # Skip is clamped so an over-large offset just yields an empty (finished) epoch.
    sampler.set_epoch(3, skip=999)
    assert list(sampler) == []
    assert len(sampler) == 0
