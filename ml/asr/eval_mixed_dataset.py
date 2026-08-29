"""Evaluate multiple local ASR model types on one mixed test dataset.

This runner adapts the existing Whisper, FastConformer, and fusion evaluators to
the ``create_mixed_test_dataset`` output contract.  Each model is transcribed
once; predictions are then joined back to ``test.tsv`` by resolved audio path so
aggregate and per-``source_dataset`` WER/CER can be computed consistently.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import jiwer
import yaml

from ml.asr.eval_openrouter_stt import EvalRow, normalize_for_scoring, read_mixed_test_rows


MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MODEL_RUNNERS: dict[str, tuple[str, str]] = {
    "whisper_small": ("ml.asr.eval_whisper_small", "run_evaluation"),
    "whisper_medium": ("ml.asr.eval_whisper_medium", "run_evaluation"),
    "whisper_large_v3_turbo": ("ml.asr.eval_whisper_large_v3_turbo", "run_evaluation"),
    "fastconformer": ("ml.asr.eval_fastconformer", "run_evaluation"),
    "fusion": ("ml.fusion.eval_fusion", "run_evaluation"),
}
MODEL_TYPE_ALIASES = {
    "whisper-small": "whisper_small",
    "whisper-medium": "whisper_medium",
    "whisper-large-v3-turbo": "whisper_large_v3_turbo",
    "fastconformer-ctc": "fastconformer",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    type: str
    config: Path


@dataclass(frozen=True)
class MixedEvalConfig:
    dataset_root: Path
    output_dir: Path
    models: tuple[ModelSpec, ...]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def resolve_from_config(raw_path: str | Path, config_path: Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    candidates = [path] if path.is_absolute() else [path, config_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (config_path.parent / path).resolve() if not path.is_absolute() else path


def canonical_model_type(raw_type: str) -> str:
    normalized = raw_type.strip().lower()
    return MODEL_TYPE_ALIASES.get(normalized, normalized)


def load_config(config_path: Path, output_dir_override: Path | None = None) -> MixedEvalConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")

    dataset_value = payload.get("dataset_root")
    if not str(dataset_value or "").strip():
        raise ValueError("dataset_root must be a non-empty mixed dataset path")
    dataset_root = resolve_from_config(str(dataset_value), config_path)

    raw_output = output_dir_override or payload.get("output_dir")
    if raw_output is None:
        output_dir = Path("artifacts/asr-mixed-eval") / run_id()
    elif output_dir_override is not None:
        output_dir = output_dir_override.expanduser()
    else:
        output_dir = Path(str(raw_output)).expanduser()

    raw_models = payload.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("models must be a non-empty list")
    models: list[ModelSpec] = []
    for index, raw_model in enumerate(raw_models):
        if not isinstance(raw_model, dict):
            raise ValueError(f"models[{index}] must be a mapping")
        name = str(raw_model.get("name") or "").strip()
        model_type = canonical_model_type(str(raw_model.get("type") or ""))
        raw_model_config = raw_model.get("config")
        if MODEL_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError(
                f"models[{index}].name must start with an alphanumeric character and contain only "
                "letters, numbers, '.', '_', or '-'"
            )
        if model_type not in MODEL_RUNNERS:
            supported = ", ".join(sorted(MODEL_RUNNERS))
            raise ValueError(f"unsupported model type {model_type!r}; expected one of: {supported}")
        if not str(raw_model_config or "").strip():
            raise ValueError(f"models[{index}].config must be a non-empty YAML path")
        models.append(
            ModelSpec(
                name=name,
                type=model_type,
                config=resolve_from_config(str(raw_model_config), config_path),
            )
        )
    names = [model.name for model in models]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"model names must be unique; duplicates: {', '.join(duplicates)}")
    return MixedEvalConfig(dataset_root=dataset_root, output_dir=output_dir, models=tuple(models))


def load_model_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"model config does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def resolve_local_model_sources(config: dict[str, Any], original_config_path: Path) -> None:
    """Keep config-relative local model paths valid after writing an adapted config."""
    model = config.get("model")
    if not isinstance(model, dict):
        return
    for key in ("checkpoint", "processor", "base_asr_checkpoint"):
        raw_value = model.get(key)
        if not isinstance(raw_value, str) or not raw_value.strip():
            continue
        path = Path(raw_value).expanduser()
        candidates = [path] if path.is_absolute() else [path, original_config_path.parent / path]
        for candidate in candidates:
            if candidate.exists():
                model[key] = str(candidate.resolve())
                break


def write_adapted_config(spec: ModelSpec, dataset_root: Path, path: Path) -> None:
    config = load_model_config(spec.config)
    resolve_local_model_sources(config, spec.config)
    data = config.setdefault("data", {})
    if not isinstance(data, dict):
        raise ValueError(f"{spec.config}: data must be a mapping")
    data.update({"root_dir": str(dataset_root.parent), "datasets": [str(dataset_root)], "split": "test"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def load_runner(model_type: str) -> Callable[[Path, Path | None], int]:
    module_name, function_name = MODEL_RUNNERS[model_type]
    return getattr(importlib.import_module(module_name), function_name)


def release_model_memory() -> None:
    """Release unreachable model objects and PyTorch's cached CUDA allocations."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(record)
    return records


def score_predictions(
    predictions: Sequence[dict[str, Any]], rows: Sequence[EvalRow]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows_by_audio: dict[str, list[EvalRow]] = defaultdict(list)
    for row in rows:
        rows_by_audio[str(row.audio_path.resolve())].append(row)
    for audio_path, matching_rows in rows_by_audio.items():
        labels = {(row.source_dataset, row.reference) for row in matching_rows}
        if len(labels) > 1:
            raise ValueError(
                "mixed test.tsv contains the same resolved audio path with conflicting "
                f"source/reference values: {audio_path}"
            )
    enriched: list[dict[str, Any]] = []
    matched_occurrences: dict[str, int] = defaultdict(int)
    for index, prediction in enumerate(predictions, start=1):
        raw_audio = prediction.get("audio_path")
        if not isinstance(raw_audio, str) or not raw_audio.strip():
            raise ValueError(f"prediction {index} has no audio_path")
        audio_key = str(Path(raw_audio).resolve())
        matching_rows = rows_by_audio.get(audio_key)
        if matching_rows is None:
            raise ValueError(f"prediction audio is not present in mixed test.tsv: {raw_audio}")
        occurrence = matched_occurrences[audio_key]
        if occurrence >= len(matching_rows):
            raise ValueError(f"more predictions than mixed test.tsv rows for audio: {raw_audio}")
        row = matching_rows[occurrence]
        matched_occurrences[audio_key] += 1
        hypothesis = str(prediction.get("hypothesis") or "")
        reference_normalized = normalize_for_scoring(row.reference)
        hypothesis_normalized = normalize_for_scoring(hypothesis)
        enriched.append(
            {
                **prediction,
                "path": row.path,
                "source_dataset": row.source_dataset,
                "reference_original": row.reference,
                "reference_normalized": reference_normalized,
                "hypothesis_normalized": hypothesis_normalized,
            }
        )

    def score(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        references = [str(record["reference_normalized"]) for record in records]
        hypotheses = [str(record["hypothesis_normalized"]) for record in records]
        return {
            "examples": len(records),
            "wer": float(jiwer.wer(references, hypotheses)),
            "cer": float(jiwer.cer(references, hypotheses)),
        }

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in enriched:
        by_dataset[str(record["source_dataset"])].append(record)
    metrics = {
        "complete": len(enriched) == len(rows),
        "expected_examples": len(rows),
        "completed_examples": len(enriched),
        "scoring_normalization": "normalize_persian_asr_text",
        "overall": score(enriched),
        "datasets": [
            {"source_dataset": name, **score(records)}
            for name, records in sorted(by_dataset.items())
        ],
    }
    return metrics, enriched


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_tsv(path: Path, models: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "type", "source_dataset", "examples", "wer", "cer"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for model in models:
            for source, metrics in [("__overall__", model["overall"])] + [
                (item["source_dataset"], item) for item in model["datasets"]
            ]:
                writer.writerow(
                    {
                        "model": model["name"],
                        "type": model["type"],
                        "source_dataset": source,
                        "examples": metrics["examples"],
                        "wer": metrics["wer"],
                        "cer": metrics["cer"],
                    }
                )


def run_evaluation(config_path: Path, output_dir_override: Path | None = None) -> int:
    config = load_config(config_path, output_dir_override)
    rows = read_mixed_test_rows(config.dataset_root)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_summaries: list[dict[str, Any]] = []

    for spec in config.models:
        model_output = config.output_dir / "models" / spec.name
        adapted_config = config.output_dir / "configs" / f"{spec.name}.yaml"
        write_adapted_config(spec, config.dataset_root, adapted_config)
        try:
            result = load_runner(spec.type)(adapted_config, model_output)
        finally:
            release_model_memory()
        if result != 0:
            raise RuntimeError(f"model {spec.name} ({spec.type}) evaluator exited with status {result}")
        predictions_path = model_output / "predictions.jsonl"
        if not predictions_path.is_file():
            raise FileNotFoundError(f"model {spec.name} did not write {predictions_path}")
        source_metrics, enriched = score_predictions(read_jsonl(predictions_path), rows)
        write_jsonl(predictions_path, enriched)
        write_json(model_output / "source_metrics.json", source_metrics)

        existing_metrics_path = model_output / "metrics.json"
        existing_metrics = json.loads(existing_metrics_path.read_text(encoding="utf-8"))
        existing_metrics["source_dataset_metrics"] = source_metrics
        write_json(existing_metrics_path, existing_metrics)
        model_summaries.append(
            {
                "name": spec.name,
                "type": spec.type,
                "config": str(spec.config),
                "output_dir": str(model_output),
                **source_metrics,
            }
        )

    summary = {
        "created_at": utc_now(),
        "dataset_root": str(config.dataset_root),
        "complete": all(model["complete"] for model in model_summaries),
        "models": model_summaries,
    }
    write_json(config.output_dir / "summary.json", summary)
    write_summary_tsv(config.output_dir / "summary.tsv", model_summaries)
    return 0 if summary["complete"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a configured list of Whisper, FastConformer, and fusion models on a mixed "
            "test dataset, reporting normalized overall and per-source-dataset WER/CER."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="Mixed-model evaluation YAML path.")
    parser.add_argument("--output-dir", type=Path, help="Optional output directory override.")
    args = parser.parse_args(argv)
    try:
        return run_evaluation(args.config, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
