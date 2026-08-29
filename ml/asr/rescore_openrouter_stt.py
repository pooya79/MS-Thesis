"""Re-score an OpenRouter STT output directory with stricter text normalization."""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import jiwer

from ml.asr.eval_openrouter_stt import read_predictions, utc_now, write_json_atomic
from ml.speech_data.text_normalization import REPLACEMENTS


NORMALIZED_PREDICTIONS_FILENAME = "predictions_strict_normalized.jsonl"
NORMALIZED_METRICS_FILENAME = "metrics_strict_normalized.json"


def normalize_strictly(text: str) -> str:
    """Normalize Persian variants and remove every Unicode punctuation/format code point.

    Unicode format characters (category ``Cf``) include the zero-width non-joiner,
    zero-width joiner, zero-width space, word joiner, and legacy zero-width no-break
    space representations commonly encountered as Persian half-spaces.
    """

    normalized = unicodedata.normalize("NFKC", text)
    for source, replacement in REPLACEMENTS.items():
        normalized = normalized.replace(source, replacement)
    normalized = "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith("P")
        and unicodedata.category(character) != "Cf"
    )
    return " ".join(normalized.split())


def score_records(records: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    references = [str(record["reference_normalized"]) for record in records]
    predictions = [str(record["prediction_normalized"]) for record in records]
    return {
        "examples": len(records),
        "wer": float(jiwer.wer(references, predictions)),
        "cer": float(jiwer.cer(references, predictions)),
    }


def normalize_record(record: dict[str, Any], *, line_number: int) -> dict[str, Any]:
    for field in ("model", "row_index", "path", "source_dataset", "reference", "prediction"):
        if field not in record:
            raise ValueError(f"predictions.jsonl:{line_number} is missing {field!r}")
    if not isinstance(record["reference"], str) or not isinstance(record["prediction"], str):
        raise ValueError(
            f"predictions.jsonl:{line_number} reference and prediction must be strings"
        )

    reference_normalized = normalize_strictly(record["reference"])
    prediction_normalized = normalize_strictly(record["prediction"])
    return {
        "model": str(record["model"]),
        "row_index": int(record["row_index"]),
        "path": str(record["path"]),
        "source_dataset": str(record["source_dataset"]),
        "reference": reference_normalized,
        "prediction": prediction_normalized,
        "reference_normalized": reference_normalized,
        "prediction_normalized": prediction_normalized,
        "wer": float(jiwer.wer(reference_normalized, prediction_normalized)),
        "cer": float(jiwer.cer(reference_normalized, prediction_normalized)),
    }


def build_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_model[str(record["model"])].append(record)

    models: list[dict[str, Any]] = []
    for model, model_records in by_model.items():
        by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in model_records:
            by_dataset[str(record["source_dataset"])].append(record)
        models.append(
            {
                "model": model,
                "overall": score_records(model_records),
                "datasets": [
                    {"source_dataset": dataset, **score_records(dataset_records)}
                    for dataset, dataset_records in sorted(by_dataset.items())
                ],
            }
        )

    return {
        "updated_at": utc_now(),
        "examples": len(records),
        "normalization": {
            "unicode_form": "NFKC",
            "persian_character_replacements": True,
            "removed_unicode_categories": ["P*", "Cf"],
            "applied_to": ["reference", "prediction"],
        },
        "models": models,
    }


def write_jsonl_atomic(path: Path, records: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def rescore_output_dir(output_dir: Path) -> dict[str, Any]:
    predictions_path = output_dir / "predictions.jsonl"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"missing OpenRouter predictions: {predictions_path}")
    source_records = read_predictions(predictions_path)
    if not source_records:
        raise ValueError(f"{predictions_path} contains no predictions")

    records = [
        normalize_record(record, line_number=line_number)
        for line_number, record in enumerate(source_records, start=1)
    ]
    metrics = build_metrics(records)
    write_jsonl_atomic(output_dir / NORMALIZED_PREDICTIONS_FILENAME, records)
    write_json_atomic(output_dir / NORMALIZED_METRICS_FILENAME, metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-score an ml.asr.eval_openrouter_stt output directory after applying "
            "NFKC, Persian character normalization, and removal of all Unicode "
            "punctuation and format/half-space characters to references and predictions."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Existing eval_openrouter_stt output directory containing predictions.jsonl.",
    )
    args = parser.parse_args(argv)
    try:
        rescore_output_dir(args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
