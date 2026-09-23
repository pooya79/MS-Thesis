"""Collect configured clean/degraded training and held-out test datasets."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import yaml

from ml.asr.train_whisper_small import resolve_audio_path
from ml.enhancement.dataset import detect_dataset_kind, read_mapping
from ml.fusion.bridging_experiment import filter_split_conflicts, skipped_record
from ml.fusion.evaluate_ablation import test_rows


def load_training_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Small bridge training config must be a mapping")
    for section in ("model", "data", "run", "preparation", "cache", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{section} must be a mapping")
    data = config["data"]
    for key in ("datasets", "test_datasets"):
        value = data.get(key)
        if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x.strip() for x in value):
            raise ValueError(f"data.{key} must be a nonempty list of dataset names")
        if len(value) != len(set(value)):
            raise ValueError(f"data.{key} contains duplicate names")
    if not data.get("root_dir"):
        raise ValueError("data.root_dir is required")
    for key in ("asr_checkpoint", "bridge_checkpoint"):
        if not isinstance(config["model"].get(key), str) or not config["model"][key].strip():
            raise ValueError(f"model.{key} is required")
    if Path(config["model"]["bridge_checkpoint"]).name != "best.pt":
        raise ValueError("model.bridge_checkpoint must name best.pt")
    if Path(config["run"]["output_dir"]) != Path(config["model"]["bridge_checkpoint"]).parent:
        raise ValueError("run.output_dir must be the bridge checkpoint's parent")
    return config


def input_config(config: dict) -> dict:
    """Expose dataset and model choices to the shared waveform preparer."""
    return {"data_root": config["data"]["root_dir"],
            "train_datasets": config["data"]["datasets"],
            "test_datasets": config["data"]["test_datasets"],
            "asr_checkpoint": config["model"]["asr_checkpoint"],
            "bridge_checkpoint": config["model"]["bridge_checkpoint"]}


def _clean_rows(directory: Path, name: str, split: str, skipped: list[dict]) -> list[dict]:
    path = directory / f"{split}.tsv"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"path", "sentence"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} needs path and sentence columns")
        records = list(reader)
    counts = Counter(str(row.get("path") or "").strip() for row in records)
    result = []
    for number, row in enumerate(records, start=2):
        relative = str(row.get("path") or "").strip()
        sentence = str(row.get("sentence") or "").strip()
        identity = {"id": f"{name}/{relative or f'line-{number}'}", "split": split}
        if not relative or not sentence:
            skipped.append(skipped_record(identity, "empty_train_field", f"{path}:{number}"))
            continue
        if counts[relative] > 1:
            skipped.append(skipped_record(identity, "duplicate_train_path", f"{path}:{number}"))
            continue
        audio = resolve_audio_path(directory, relative).resolve()
        if not audio.is_file():
            skipped.append(skipped_record(identity, "missing_train_audio", str(audio)))
            continue
        speaker = str(row.get("client_id") or "").strip()
        result.append({**identity, "source_id": str(audio),
                       "speaker_id": f"{name}/{speaker}" if speaker else None,
                       "sentence": sentence, "noisy_path": str(audio), "dataset": name})
    return result


def collect_small_rows(config: dict, scope: str, skipped: list[dict]) -> list[dict]:
    """Use all configured train/dev rows and held-out test TSVs."""
    root = Path(config["data_root"]).resolve()
    rows: list[dict] = []
    clean_sources: dict[str, set[str]] = {}
    speakers: dict[str, str | None] = {}
    for name in config["train_datasets"]:
        directory = root / name
        if detect_dataset_kind(directory) != "clean":
            continue
        for split in ("train", "dev"):
            for row in _clean_rows(directory, name, split, skipped):
                clean_sources.setdefault(row["source_id"], set()).add(split)
                speakers[row["source_id"]] = row["speaker_id"]
                rows.append(row)
    for name in config["train_datasets"]:
        directory = root / name
        if detect_dataset_kind(directory) != "degraded":
            continue
        for split in ("train", "dev"):
            for pair in read_mapping(directory, split):
                source = str(pair.clean_path.resolve())
                identity = {"id": f"{name}/{pair.pair_id}", "split": split}
                if clean_sources.get(source) != {split}:
                    skipped.append(skipped_record(identity, "source_missing_or_wrong_split", source))
                    continue
                if not pair.degraded_path.is_file():
                    skipped.append(skipped_record(identity, "missing_noisy_audio", str(pair.degraded_path)))
                    continue
                rows.append({**identity, "source_id": source, "speaker_id": speakers.get(source),
                             "sentence": pair.transcript, "noisy_path": str(pair.degraded_path.resolve()),
                             "dataset": name, "degradation": pair.degradation})
    if scope in {"all", "test"}:
        test_config = {"data": {"root_dir": str(root), "datasets": config["test_datasets"], "split": "test"}}
        for row in test_rows(test_config, skipped):
            rows.append({"id": row["id"], "source_id": row["audio_path"],
                         "speaker_id": None, "split": "test", "sentence": row["reference"],
                         "noisy_path": row["audio_path"], "dataset": row["dataset"],
                         "relative_path": row["relative_path"]})
    counts = Counter(row["id"] for row in rows)
    unique = []
    for row in rows:
        if counts[row["id"]] > 1:
            skipped.append(skipped_record(row, "duplicate_id"))
        else:
            unique.append(row)
    rows = unique
    rows = filter_split_conflicts(rows, skipped)
    if scope == "test":
        rows = [row for row in rows if row["split"] == "test"]
    return rows
