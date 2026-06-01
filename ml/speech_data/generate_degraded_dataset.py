from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import yaml
from tqdm import tqdm

from ml.speech_data.generate_degraded_pairs import (
    ManifestItem,
    default_config,
    degrade_item,
    load_asset_index,
    load_config as load_degradation_config,
    resolve_path,
    safe_pair_id,
    validate_config,
    write_jsonl,
)
from ml.utils.audio import save_audio


@dataclass(frozen=True)
class DatasetRow:
    source_index: int
    source_tsv: Path
    split: str
    values: dict[str, str]
    clean_audio_path: Path


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "source_dir": "data/cv-corpus-25.0",
        "output_dir": "data/cv-corpus-25.0-degraded",
        "splits": ["train.tsv", "dev.tsv", "test.tsv"],
        "variations_per_sample": 2,
        "mapping_filename": "degraded_to_clean.jsonl",
        "metadata_filename": "degradation_metadata.jsonl",
        "report_filename": "generation_report.json",
    },
    "degradation_config": "configs/speech_enhancement/degradation.yaml",
    "degradation": {},
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def load_generation_config(path: Path) -> dict[str, Any]:
    raw_config = deep_merge(DEFAULT_CONFIG, load_yaml(path))
    base_dir = Path.cwd()
    degradation_path = raw_config.get("degradation_config")
    base_degradation: dict[str, Any] = {}
    if degradation_path:
        resolved = resolve_path(str(degradation_path), base_dir)
        if resolved is None:
            raise ValueError("degradation_config must be a path when provided")
        base_degradation = load_degradation_config(resolved)
    raw_config["degradation"] = deep_merge(base_degradation, raw_config.get("degradation") or {})
    return raw_config


def normalize_split_name(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("dataset.splits cannot contain empty values")
    return value if value.endswith(".tsv") else f"{value}.tsv"


def resolve_audio_path(dataset_dir: Path, value: str) -> Path:
    raw_path = Path(value)
    if raw_path.is_absolute():
        return raw_path
    candidates = [dataset_dir / "clips" / raw_path, dataset_dir / raw_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def read_split_rows(dataset_dir: Path, split_tsv: str) -> tuple[list[str], list[DatasetRow]]:
    split_path = dataset_dir / split_tsv
    split = Path(split_tsv).stem
    rows: list[DatasetRow] = []
    with split_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"path", "sentence"}.issubset(reader.fieldnames):
            raise ValueError(f"{split_path} must contain path and sentence columns")
        fieldnames = list(reader.fieldnames)
        for index, row in enumerate(reader, start=1):
            clean_audio_path = resolve_audio_path(dataset_dir, str(row.get("path", "")).strip())
            if not clean_audio_path.exists():
                raise FileNotFoundError(f"{split_path}:{index} missing audio file: {clean_audio_path}")
            rows.append(
                DatasetRow(
                    source_index=index,
                    source_tsv=split_path,
                    split=split,
                    values={key: str(value or "") for key, value in row.items()},
                    clean_audio_path=clean_audio_path,
                )
            )
    return fieldnames, rows


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def dataset_item(row: DatasetRow) -> ManifestItem:
    source_path = str(row.values["path"])
    clean_id = f"{row.split}-{Path(source_path).stem}-{row.source_index:06d}"
    return ManifestItem(
        id=clean_id,
        split=row.split,
        clean_path=row.clean_audio_path,
        transcript=row.values.get("sentence"),
    )


def validate_generation_config(config: dict[str, Any]) -> None:
    dataset = config["dataset"]
    source_dir = Path(str(dataset["source_dir"]))
    if not source_dir.is_dir():
        raise FileNotFoundError(f"dataset.source_dir does not exist: {source_dir}")
    if not (source_dir / "clips").is_dir():
        raise FileNotFoundError(f"dataset.source_dir is missing clips/: {source_dir}")
    split_names = [normalize_split_name(str(split)) for split in dataset["splits"]]
    if not split_names:
        raise ValueError("dataset.splits must contain at least one TSV")
    for split_name in split_names:
        if not (source_dir / split_name).is_file():
            raise FileNotFoundError(f"configured split TSV does not exist: {source_dir / split_name}")
    if int(dataset["variations_per_sample"]) < 1:
        raise ValueError("dataset.variations_per_sample must be >= 1")
    degradation_config = default_config(config["degradation"])
    degradation_config.setdefault("manifests", {})
    degradation_config["manifests"].setdefault("train", "__unused_train__.jsonl")
    degradation_config["manifests"].setdefault("valid", "__unused_valid__.jsonl")
    validate_config(degradation_config)


def generate_degraded_dataset(config: dict[str, Any]) -> dict[str, Any]:
    validate_generation_config(config)
    dataset_config = config["dataset"]
    source_dir = Path(str(dataset_config["source_dir"]))
    output_dir = Path(str(dataset_config["output_dir"]))
    output_clips_dir = output_dir / "clips"
    split_names = [normalize_split_name(str(split)) for split in dataset_config["splits"]]
    variations = int(dataset_config["variations_per_sample"])

    degradation_config = default_config(config["degradation"])
    degradation_config["output_dir"] = str(output_dir)
    degradation_config.setdefault("manifests", {})
    degradation_config["manifests"].setdefault("train", "__unused_train__.jsonl")
    degradation_config["manifests"].setdefault("valid", "__unused_valid__.jsonl")

    config_base = Path.cwd()
    rir_assets = load_asset_index(resolve_path(degradation_config.get("rir_index"), config_base))
    noise_assets = load_asset_index(resolve_path(degradation_config.get("noise_index"), config_base))

    mapping_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "splits": {},
        "mapping": str(output_dir / str(dataset_config["mapping_filename"])),
        "metadata": str(output_dir / str(dataset_config["metadata_filename"])),
        "skipped": [],
    }
    for split_tsv in split_names:
        fieldnames, source_rows = read_split_rows(source_dir, split_tsv)
        degraded_rows: list[dict[str, str]] = []
        iterator = tqdm(source_rows, desc=f"degrading {split_tsv}", unit="clip")
        for row in iterator:
            item = dataset_item(row)
            for variant_index in range(variations):
                try:
                    metadata, _clean_target, degraded_audio, model_rate = degrade_item(
                        item,
                        variant_index,
                        degradation_config,
                        rir_assets,
                        noise_assets,
                    )
                except sf.LibsndfileError as exc:
                    report["skipped"].append(
                        {
                            "source_tsv": str(row.source_tsv),
                            "source_index": row.source_index,
                            "variant_index": variant_index,
                            "error": str(exc),
                        }
                    )
                    continue

                degraded_relative_path = Path(row.split) / f"{safe_pair_id(row.split, item.id, variant_index)}.wav"
                degraded_path = output_clips_dir / degraded_relative_path
                save_audio(degraded_path, degraded_audio, model_rate)
                output_values = dict(row.values)
                output_values["path"] = degraded_relative_path.as_posix()
                degraded_rows.append(output_values)
                degradation_metadata = dict(metadata)
                degradation_metadata.update({"clean_path": str(row.clean_audio_path), "degraded_path": str(degraded_path)})
                metadata_rows.append(degradation_metadata)

                mapping_row = {
                    "degraded_id": metadata["pair_id"],
                    "split": row.split,
                    "source_tsv": str(row.source_tsv),
                    "source_row_index": row.source_index,
                    "variant_index": variant_index,
                    "clean_path": str(row.clean_audio_path),
                    "source_path": row.values["path"],
                    "degraded_path": str(degraded_path),
                    "degraded_tsv_path": degraded_relative_path.as_posix(),
                    "sentence": row.values.get("sentence"),
                    "degradation": degradation_metadata,
                }
                mapping_rows.append(mapping_row)

        output_tsv = output_dir / split_tsv
        write_tsv(output_tsv, fieldnames, degraded_rows)
        report["splits"][Path(split_tsv).stem] = {
            "source_rows": len(source_rows),
            "degraded_rows": len(degraded_rows),
            "tsv": str(output_tsv),
        }

    mapping_path = output_dir / str(dataset_config["mapping_filename"])
    write_jsonl(mapping_path, mapping_rows)
    metadata_path = output_dir / str(dataset_config["metadata_filename"])
    write_jsonl(metadata_path, metadata_rows)
    report_path = output_dir / str(dataset_config["report_filename"])
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a degraded-only dataset directory from TSV-based ASR data, "
            "preserving split TSVs and writing degraded-to-clean mapping metadata."
        )
    )
    parser.add_argument("--config", required=True, help="Path to degraded dataset generation YAML config.")
    args = parser.parse_args(argv)
    report = generate_degraded_dataset(load_generation_config(Path(args.config)))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
