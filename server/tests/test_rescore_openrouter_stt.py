from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.asr.rescore_openrouter_stt import normalize_strictly, rescore_output_dir


def test_strict_normalization_removes_unicode_punctuation_and_half_space_variants() -> None:
    text = "كتاب\u200cها، (کتاب\u200bها)! کتاب\u200dها؛ کتاب\u2060ها؟ کتاب\ufeffها"

    assert normalize_strictly(text) == "کتابها کتابها کتابها کتابها کتابها"


def test_rescore_normalizes_raw_reference_and_prediction_and_groups_metrics(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "openrouter-output"
    output_dir.mkdir()
    source_records = [
        {
            "model": "provider/model",
            "row_index": 0,
            "path": "one.wav",
            "source_dataset": "dataset-a",
            "reference": "می\u200cروم.",
            "prediction": "میروم!",
            "reference_normalized": "stale reference",
            "prediction_normalized": "stale prediction",
        },
        {
            "model": "provider/model",
            "row_index": 1,
            "path": "two.wav",
            "source_dataset": "dataset-b",
            "reference": "كتاب خوب؟",
            "prediction": "کتاب خوب",
        },
    ]
    (output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in source_records),
        encoding="utf-8",
    )

    metrics = rescore_output_dir(output_dir)

    rescored = [
        json.loads(line)
        for line in (output_dir / "predictions_strict_normalized.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rescored[0]["reference_normalized"] == "میروم"
    assert rescored[0]["prediction_normalized"] == "میروم"
    assert rescored[0]["reference"] == "میروم"
    assert rescored[0]["prediction"] == "میروم"
    assert rescored[1]["reference_normalized"] == "کتاب خوب"
    assert all(
        punctuation not in record[field]
        for record in rescored
        for field in (
            "reference",
            "prediction",
            "reference_normalized",
            "prediction_normalized",
        )
        for punctuation in "؟،؛.!()"
    )
    assert all(record["wer"] == 0 and record["cer"] == 0 for record in rescored)
    assert metrics["models"][0]["overall"] == {"examples": 2, "wer": 0.0, "cer": 0.0}
    assert {item["source_dataset"] for item in metrics["models"][0]["datasets"]} == {
        "dataset-a",
        "dataset-b",
    }
    written_metrics = json.loads(
        (output_dir / "metrics_strict_normalized.json").read_text(encoding="utf-8")
    )
    assert written_metrics["normalization"]["applied_to"] == ["reference", "prediction"]


def test_rescore_requires_nonempty_predictions_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "empty-output"
    output_dir.mkdir()
    (output_dir / "predictions.jsonl").touch()

    with pytest.raises(ValueError, match="contains no predictions"):
        rescore_output_dir(output_dir)
