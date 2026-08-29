from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

from ml.speech_data.generate_degraded_dataset import (
    DatasetRow,
    dataset_item,
    normalize_split_name,
    read_split_rows,
    write_tsv,
)
from ml.speech_data.generate_degraded_pairs import (
    ManifestItem,
    choose_noise_segment,
    load_asset_index,
    pair_peak_safety_normalize,
    resolve_path,
    safe_pair_id,
    sample_uniform,
    write_jsonl,
)
from ml.utils.audio import load_audio, match_length, mix_at_snr, resample_audio, save_audio
from ml.utils.seed import stable_seed


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "source_dir": "data/cv-corpus-25.0",
        "output_dir": "data/cv-corpus-25.0-noise-added",
        "splits": ["train.tsv", "dev.tsv"],
        "variations_per_sample": 2,
        "workers": 1,
        "mapping_filename": "degraded_to_clean.jsonl",
        "metadata_filename": "noise_metadata.jsonl",
        "report_filename": "generation_report.json",
    },
    "seed": 1337,
    "model_sample_rate": 16000,
    "noise_index": "data/speech_enhancement/manifests/demand_noise_index.jsonl",
    "noise": {
        "snr_buckets_db": [[0, 5], [5, 10], [10, 15], [15, 20]],
    },
    "normalization": {"peak": 0.99},
}


@dataclass(frozen=True)
class NoiseJob:
    row: DatasetRow
    item: ManifestItem
    variant_index: int
    config: dict[str, Any]
    noise_assets: list[dict[str, Any]]


@dataclass(frozen=True)
class NoiseJobResult:
    row: DatasetRow
    variant_index: int
    metadata: dict[str, Any] | None
    noisy_audio: Any
    model_rate: int | None
    error: str | None = None


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return deep_merge(DEFAULT_CONFIG, loaded)


def validate_config(config: dict[str, Any]) -> None:
    dataset = config["dataset"]
    source_dir = Path(str(dataset["source_dir"]))
    if not source_dir.is_dir():
        raise FileNotFoundError(f"dataset.source_dir does not exist: {source_dir}")
    if not (source_dir / "clips").is_dir():
        raise FileNotFoundError(f"dataset.source_dir is missing clips/: {source_dir}")

    splits = [normalize_split_name(str(value)) for value in dataset["splits"]]
    if not splits:
        raise ValueError("dataset.splits must contain at least one TSV")
    for split in splits:
        if not (source_dir / split).is_file():
            raise FileNotFoundError(f"configured split TSV does not exist: {source_dir / split}")
    if int(dataset["variations_per_sample"]) < 1:
        raise ValueError("dataset.variations_per_sample must be >= 1")
    if int(dataset.get("workers", 1)) < 1:
        raise ValueError("dataset.workers must be >= 1")
    if int(config["model_sample_rate"]) < 1:
        raise ValueError("model_sample_rate must be >= 1")

    buckets = config["noise"]["snr_buckets_db"]
    if not buckets:
        raise ValueError("noise.snr_buckets_db must contain at least one [min, max] bucket")
    for bucket in buckets:
        if not isinstance(bucket, (list, tuple)) or len(bucket) != 2:
            raise ValueError(f"each SNR bucket must be [min, max], got {bucket!r}")
        if float(bucket[0]) > float(bucket[1]):
            raise ValueError(f"SNR bucket minimum exceeds maximum: {bucket!r}")
    peak = float(config["normalization"]["peak"])
    if not 0 < peak <= 1:
        raise ValueError("normalization.peak must be in (0, 1]")


def add_noise(
    item: ManifestItem,
    variant_index: int,
    config: dict[str, Any],
    noise_assets: list[dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray, int]:
    if not noise_assets:
        raise ValueError("noise_index must contain at least one noise asset")

    seed = stable_seed(int(config["seed"]), item.split, item.id, variant_index)
    rng = np.random.default_rng(seed)
    model_rate = int(config["model_sample_rate"])
    clean, source_rate = load_audio(item.clean_path)
    clean = resample_audio(clean, source_rate, model_rate)

    buckets = config["noise"]["snr_buckets_db"]
    bucket_index = int(rng.integers(0, len(buckets)))
    snr_db = sample_uniform(rng, buckets[bucket_index])
    noise_asset = noise_assets[int(rng.integers(0, len(noise_assets)))]
    noise, noise_rate = load_audio(noise_asset["path"])
    noise = resample_audio(noise, noise_rate, model_rate)
    noise = choose_noise_segment(noise, len(clean), rng)
    noisy = mix_at_snr(clean, noise, snr_db)
    clean = match_length(clean, len(noisy))
    _clean_target, noisy, normalization_scale = pair_peak_safety_normalize(
        clean,
        noisy,
        peak=float(config["normalization"]["peak"]),
    )

    metadata: dict[str, Any] = {
        "pair_id": safe_pair_id(item.split, item.id, variant_index),
        "split": item.split,
        "profile": "noise_only",
        "degradation_type": "additive_noise",
        "source_clean_id": item.id,
        "source_clean_path": str(item.clean_path),
        "model_sample_rate": model_rate,
        "seed": seed,
        "transcript": item.transcript,
        "noise_id": noise_asset.get("id"),
        "noise_scene": noise_asset.get("scene", noise_asset.get("id")),
        "noise_source_path": str(noise_asset["path"]),
        "snr_bucket_index": bucket_index,
        "snr_bucket_db": [float(buckets[bucket_index][0]), float(buckets[bucket_index][1])],
        "snr_db": snr_db,
        "duration_sec": len(noisy) / model_rate,
        "target_bandwidth": "wideband",
        "normalization": "shared_pair_peak_safety",
        "normalization_scale": normalization_scale,
        "codec": None,
        "network_impairment": {"enabled": False},
        "filtering": {"enabled": False},
        "random_gain": {"enabled": False},
        "clipping": {"enabled": False},
        "agc": {"enabled": False},
    }
    return metadata, noisy, model_rate


def run_noise_job(job: NoiseJob) -> NoiseJobResult:
    try:
        metadata, noisy_audio, model_rate = add_noise(
            job.item, job.variant_index, job.config, job.noise_assets
        )
    except sf.LibsndfileError as exc:
        return NoiseJobResult(
            row=job.row,
            variant_index=job.variant_index,
            metadata=None,
            noisy_audio=None,
            model_rate=None,
            error=str(exc),
        )
    return NoiseJobResult(
        row=job.row,
        variant_index=job.variant_index,
        metadata=metadata,
        noisy_audio=noisy_audio,
        model_rate=model_rate,
    )


def iter_noise_jobs(jobs: list[NoiseJob], workers: int) -> Iterator[NoiseJobResult]:
    if workers < 1:
        raise ValueError("dataset.workers must be >= 1")
    if workers == 1:
        for job in jobs:
            yield run_noise_job(job)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        yield from executor.map(run_noise_job, jobs)


def generate_noise_added_dataset(config: dict[str, Any]) -> dict[str, Any]:
    config = deep_merge(DEFAULT_CONFIG, config)
    validate_config(config)
    dataset = config["dataset"]
    source_dir = Path(str(dataset["source_dir"]))
    output_dir = Path(str(dataset["output_dir"]))
    output_clips_dir = output_dir / "clips"
    split_names = [normalize_split_name(str(value)) for value in dataset["splits"]]
    variations = int(dataset["variations_per_sample"])
    workers = int(dataset.get("workers", 1))

    noise_index = resolve_path(str(config["noise_index"]), Path.cwd())
    noise_assets = load_asset_index(noise_index)
    if not noise_assets:
        raise ValueError("noise_index must contain at least one noise asset")

    mapping_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "splits": {},
        "mapping": str(output_dir / str(dataset["mapping_filename"])),
        "metadata": str(output_dir / str(dataset["metadata_filename"])),
        "skipped": [],
    }

    for split_tsv in split_names:
        fieldnames, source_rows = read_split_rows(source_dir, split_tsv)
        output_rows: list[dict[str, str]] = []
        jobs = [
            NoiseJob(row, dataset_item(row), variant_index, config, noise_assets)
            for row in source_rows
            for variant_index in range(variations)
        ]
        iterator = tqdm(
            iter_noise_jobs(jobs, workers),
            desc=f"adding noise to {split_tsv}",
            unit="variant",
            total=len(jobs),
        )
        for result in iterator:
            row = result.row
            if result.error is not None:
                report["skipped"].append(
                    {
                        "source_tsv": str(row.source_tsv),
                        "source_index": row.source_index,
                        "variant_index": result.variant_index,
                        "error": result.error,
                    }
                )
                continue
            if result.metadata is None or result.model_rate is None:
                raise RuntimeError("noise job succeeded without metadata or sample rate")

            relative_path = Path(row.split) / f"{result.metadata['pair_id']}.wav"
            noisy_path = output_clips_dir / relative_path
            save_audio(noisy_path, result.noisy_audio, result.model_rate)
            output_values = dict(row.values)
            output_values["path"] = relative_path.as_posix()
            output_rows.append(output_values)

            metadata = dict(result.metadata)
            metadata.update({"clean_path": str(row.clean_audio_path), "degraded_path": str(noisy_path)})
            metadata_rows.append(metadata)
            mapping_rows.append(
                {
                    "degraded_id": metadata["pair_id"],
                    "split": row.split,
                    "source_tsv": str(row.source_tsv),
                    "source_row_index": row.source_index,
                    "variant_index": result.variant_index,
                    "clean_path": str(row.clean_audio_path),
                    "source_path": row.values["path"],
                    "degraded_path": str(noisy_path),
                    "degraded_tsv_path": relative_path.as_posix(),
                    "sentence": row.values.get("sentence"),
                    "degradation": metadata,
                }
            )

        output_tsv = output_dir / split_tsv
        write_tsv(output_tsv, fieldnames, output_rows)
        report["splits"][Path(split_tsv).stem] = {
            "source_rows": len(source_rows),
            "degraded_rows": len(output_rows),
            "tsv": str(output_tsv),
        }

    mapping_path = output_dir / str(dataset["mapping_filename"])
    metadata_path = output_dir / str(dataset["metadata_filename"])
    report_path = output_dir / str(dataset["report_filename"])
    write_jsonl(mapping_path, mapping_rows)
    write_jsonl(metadata_path, metadata_rows)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a TSV-based ASR dataset whose only degradation is one additive "
            "noise scene mixed at a configured SNR."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the noise-added dataset YAML config.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes; overrides dataset.workers. Defaults to the YAML value.",
    )
    args = parser.parse_args(argv)
    config = load_config(Path(args.config))
    if args.workers is not None:
        config["dataset"]["workers"] = args.workers
    report = generate_noise_added_dataset(config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
