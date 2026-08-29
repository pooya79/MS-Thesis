from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from ml.asr.eval_mixed_dataset import (
    ModelSpec,
    load_config,
    run_evaluation,
    score_predictions,
    write_adapted_config,
)
from ml.asr.eval_openrouter_stt import read_mixed_test_rows


def make_mixed_dataset(root: Path) -> None:
    rows = [
        {"path": "clips/first/a.wav", "sentence": "سلام دنیا", "source_dataset": "first"},
        {"path": "clips/second/b.wav", "sentence": "حال شما", "source_dataset": "second"},
    ]
    for row in rows:
        audio = root / row["path"]
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"fake audio")
    with (root / "test.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "sentence", "source_dataset"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_load_config_accepts_aliases_and_rejects_duplicate_names(tmp_path: Path) -> None:
    dataset = tmp_path / "mixed"
    make_mixed_dataset(dataset)
    model_config = tmp_path / "model.yaml"
    model_config.write_text("model: {}\n", encoding="utf-8")
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset_root": "mixed",
                "models": [
                    {"name": "small", "type": "whisper-small", "config": "model.yaml"},
                    {"name": "small", "type": "fusion", "config": "model.yaml"},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model names must be unique"):
        load_config(config_path)

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["models"] = payload["models"][:1]
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_config(config_path)
    assert config.dataset_root == dataset.resolve()
    assert config.models[0].type == "whisper_small"


def test_write_adapted_config_replaces_only_dataset_selection(tmp_path: Path) -> None:
    dataset = tmp_path / "mixed"
    make_mixed_dataset(dataset)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    source_config = tmp_path / "configs" / "model.yaml"
    source_config.parent.mkdir()
    source_config.write_text(
        yaml.safe_dump(
            {
                "model": {"checkpoint": "../checkpoint"},
                "data": {"root_dir": "old", "datasets": ["old"], "sample_rate": 16000},
                "eval": {"batch_size": 3},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "adapted.yaml"

    write_adapted_config(ModelSpec("small", "whisper_small", source_config), dataset, output)

    adapted = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert adapted["model"]["checkpoint"] == str(checkpoint.resolve())
    assert adapted["data"]["datasets"] == [str(dataset)]
    assert adapted["data"]["split"] == "test"
    assert adapted["data"]["sample_rate"] == 16000
    assert adapted["eval"]["batch_size"] == 3


def test_score_predictions_reports_overall_and_source_metrics(tmp_path: Path) -> None:
    dataset = tmp_path / "mixed"
    make_mixed_dataset(dataset)
    rows = read_mixed_test_rows(dataset)
    predictions = [
        {"audio_path": str(rows[0].audio_path), "reference": "ignored", "hypothesis": "سلام، دنیا!"},
        {"audio_path": str(rows[1].audio_path), "reference": "ignored", "hypothesis": "حال من"},
    ]

    metrics, enriched = score_predictions(predictions, rows)

    assert metrics["complete"] is True
    assert metrics["overall"]["examples"] == 2
    assert [item["source_dataset"] for item in metrics["datasets"]] == ["first", "second"]
    assert metrics["datasets"][0]["wer"] == 0.0
    assert metrics["datasets"][1]["wer"] == 0.5
    assert enriched[0]["source_dataset"] == "first"
    assert enriched[0]["reference_original"] == "سلام دنیا"


def test_run_evaluation_dispatches_models_and_writes_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "mixed"
    make_mixed_dataset(dataset)
    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        yaml.safe_dump(
            {
                "model": {"checkpoint": "checkpoint.pt"},
                "data": {"sample_rate": 16000},
                "eval": {"batch_size": 1},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    config_path = tmp_path / "mixed-eval.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset_root": str(dataset),
                "output_dir": str(output),
                "models": [{"name": "test-model", "type": "fastconformer", "config": str(model_config)}],
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(adapted_config: Path, model_output: Path | None) -> int:
        assert model_output is not None
        adapted = yaml.safe_load(adapted_config.read_text(encoding="utf-8"))
        assert adapted["data"]["datasets"] == [str(dataset.resolve())]
        rows = read_mixed_test_rows(dataset)
        model_output.mkdir(parents=True)
        with (model_output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        {"audio_path": str(row.audio_path), "reference": row.reference, "hypothesis": row.reference},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        (model_output / "metrics.json").write_text('{"wer": 0.0}\n', encoding="utf-8")
        return 0

    cleanup_calls: list[None] = []
    monkeypatch.setattr("ml.asr.eval_mixed_dataset.load_runner", lambda _model_type: fake_runner)
    monkeypatch.setattr("ml.asr.eval_mixed_dataset.release_model_memory", lambda: cleanup_calls.append(None))

    assert run_evaluation(config_path) == 0
    assert cleanup_calls == [None]
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["models"][0]["overall"] == {"examples": 2, "wer": 0.0, "cer": 0.0}
    enriched = [json.loads(line) for line in (output / "models/test-model/predictions.jsonl").read_text().splitlines()]
    assert [row["source_dataset"] for row in enriched] == ["first", "second"]
    model_metrics = json.loads((output / "models/test-model/metrics.json").read_text(encoding="utf-8"))
    assert model_metrics["source_dataset_metrics"]["complete"] is True


def test_run_evaluation_releases_model_memory_when_runner_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "mixed"
    make_mixed_dataset(dataset)
    model_config = tmp_path / "model.yaml"
    model_config.write_text("model: {}\n", encoding="utf-8")
    config_path = tmp_path / "mixed-eval.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset_root": str(dataset),
                "output_dir": str(tmp_path / "output"),
                "models": [
                    {"name": "test-model", "type": "fastconformer", "config": str(model_config)}
                ],
            }
        ),
        encoding="utf-8",
    )
    cleanup_calls: list[None] = []

    def failing_runner(_config: Path, _output: Path | None) -> int:
        raise RuntimeError("inference failed")

    monkeypatch.setattr("ml.asr.eval_mixed_dataset.load_runner", lambda _model_type: failing_runner)
    monkeypatch.setattr("ml.asr.eval_mixed_dataset.release_model_memory", lambda: cleanup_calls.append(None))

    with pytest.raises(RuntimeError, match="inference failed"):
        run_evaluation(config_path)

    assert cleanup_calls == [None]
