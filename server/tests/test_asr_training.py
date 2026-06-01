from __future__ import annotations

import csv
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from ml.asr.train_whisper_small import (
    JsonMetricsCallback,
    WhisperDataset,
    latest_checkpoint,
    load_split_examples,
    load_training_config,
    resolve_pretrained_model,
    resolve_resume_checkpoint,
    resolve_run_dir,
    resolve_dataset_dirs,
    word_error_rate,
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


class FakeFeatureExtractor:
    def __call__(self, audio, *, sampling_rate: int, return_tensors: str):
        assert sampling_rate == 16000
        assert return_tensors == "pt"
        return SimpleNamespace(input_features=torch.ones((1, 2, 3), dtype=torch.float32))


class FakeTokenizer:
    def __call__(self, text: str):
        return SimpleNamespace(input_ids=[len(text), 1])


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


def test_whisper_dataset_computes_features_in_getitem(tmp_path: Path) -> None:
    dataset_dir = write_audio_dataset(tmp_path / "data", "cv-corpus-25.0")
    examples = load_split_examples([dataset_dir], "train")

    item = WhisperDataset(examples, FakeProcessor(), 16000)[0]

    assert torch.equal(item["input_features"], torch.ones((2, 3), dtype=torch.float32))
    assert item["labels"] == [10, 1]


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
