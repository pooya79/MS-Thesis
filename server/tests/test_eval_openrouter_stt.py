from __future__ import annotations

import base64
import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from ml.asr.eval_openrouter_stt import EvalRow, OpenRouterClient, evaluate, read_mixed_test_rows


def make_dataset(root: Path) -> Path:
    clips = root / "clips"
    clips.mkdir(parents=True)
    (clips / "one.wav").write_bytes(b"first-audio")
    (clips / "two.flac").write_bytes(b"second-audio")
    with (root / "test.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "sentence", "source_dataset"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            [
                {"path": "one.wav", "sentence": "سلام دنیا", "source_dataset": "dataset-a"},
                {"path": "two.flac", "sentence": "كتاب خوب", "source_dataset": "dataset-b"},
            ]
        )
    return root


class FakeOpenRouterClient:
    predictions = {
        ("model/a", 0): "سلام، دنیا!",
        ("model/a", 1): "کتاب خوب",
        ("model/b", 0): "سلام",
        ("model/b", 1): "کتاب بسیار خوب",
    }
    calls: list[tuple[str, int]] = []
    remaining = 10.0

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        assert api_key == "test-key"

    def __enter__(self) -> FakeOpenRouterClient:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def key_status(self) -> dict[str, Any]:
        return {"usage": 3.0, "limit_remaining": self.remaining}

    def transcribe(self, model: str, row: EvalRow, language: str) -> dict[str, Any]:
        assert language == "fa"
        self.calls.append((model, row.index))
        return {
            "text": self.predictions[(model, row.index)],
            "usage": {"seconds": 1.0, "input_tokens": 2, "output_tokens": 1, "cost": 0.1},
        }


def test_reads_source_dataset_and_resolves_mixed_clip_paths(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "mixed")

    rows = read_mixed_test_rows(dataset)

    assert [row.source_dataset for row in rows] == ["dataset-a", "dataset-b"]
    assert [row.audio_path.name for row in rows] == ["one.wav", "two.flac"]


def test_evaluation_writes_exact_predictions_grouped_metrics_cost_and_events(tmp_path: Path) -> None:
    FakeOpenRouterClient.calls = []
    dataset = make_dataset(tmp_path / "mixed")
    output = tmp_path / "results"

    exit_code = evaluate(
        dataset_root=dataset,
        models=["model/a", "model/b"],
        output_dir=output,
        max_run_cost=Decimal("5"),
        api_key="test-key",
        client_factory=FakeOpenRouterClient,
    )

    assert exit_code == 0
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [record["prediction"] for record in predictions] == [
        "سلام، دنیا!",
        "کتاب خوب",
        "سلام",
        "کتاب بسیار خوب",
    ]
    assert predictions[0]["prediction_normalized"] == "سلام دنیا"
    assert predictions[1]["reference"] == "كتاب خوب"
    assert predictions[1]["reference_normalized"] == "کتاب خوب"
    assert all(record["source_dataset"] in {"dataset-a", "dataset-b"} for record in predictions)

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["complete"] is True
    assert metrics["total_cost_usd"] == pytest.approx(0.4)
    assert [entry["model"] for entry in metrics["models"]] == ["model/a", "model/b"]
    assert metrics["models"][0]["overall"]["wer"] == 0
    assert {item["source_dataset"] for item in metrics["models"][0]["datasets"]} == {
        "dataset-a",
        "dataset-b",
    }
    assert metrics["models"][1]["overall"]["wer"] > 0
    assert (output / "predictions.tsv").is_file()
    events = [json.loads(line)["event"] for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events.count("budget_checked") == 4
    assert events.count("request_succeeded") == 4
    assert events[-1] == "run_finished"
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert "test-key" not in json.dumps(config)


def test_budget_stop_checkpoints_and_resume_skips_completed_requests(tmp_path: Path) -> None:
    FakeOpenRouterClient.calls = []
    dataset = make_dataset(tmp_path / "mixed")
    output_two = tmp_path / "results-two"
    stopped = evaluate(
        dataset_root=dataset,
        models=["model/a", "model/b"],
        output_dir=output_two,
        max_run_cost=Decimal("0.15"),
        api_key="test-key",
        client_factory=FakeOpenRouterClient,
    )
    assert stopped == 2
    assert FakeOpenRouterClient.calls == [("model/a", 0), ("model/a", 1)]
    # The actual cost is known only after a request, so this demonstrates the
    # documented one-request local-cap overshoot before model/b is attempted.
    assert json.loads((output_two / "metrics.json").read_text(encoding="utf-8"))["total_cost_usd"] == pytest.approx(0.2)

    # Budget may be raised on resume; dataset, models, language, and manifest
    # remain immutable so completed requests can be identified safely.
    resumed = evaluate(
        dataset_root=dataset,
        models=["model/a", "model/b"],
        output_dir=output_two,
        max_run_cost=Decimal("1"),
        resume=True,
        api_key="test-key",
        client_factory=FakeOpenRouterClient,
    )
    assert resumed == 0
    assert FakeOpenRouterClient.calls == [
        ("model/a", 0),
        ("model/a", 1),
        ("model/b", 0),
        ("model/b", 1),
    ]
    assert len((output_two / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 4


def test_minimum_key_remaining_stops_before_spending(tmp_path: Path) -> None:
    FakeOpenRouterClient.calls = []
    FakeOpenRouterClient.remaining = 0.5
    try:
        exit_code = evaluate(
            dataset_root=make_dataset(tmp_path / "mixed"),
            models=["model/a"],
            output_dir=tmp_path / "results",
            max_run_cost=Decimal("5"),
            min_key_remaining=Decimal("1"),
            api_key="test-key",
            client_factory=FakeOpenRouterClient,
        )
    finally:
        FakeOpenRouterClient.remaining = 10.0

    assert exit_code == 2
    assert FakeOpenRouterClient.calls == []


def test_openrouter_client_uses_current_key_and_base64_stt_endpoints(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio-bytes")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer secret"
        if request.url.path.endswith("/key"):
            return httpx.Response(200, json={"data": {"usage": 2, "limit_remaining": 8}})
        body = json.loads(request.content)
        assert body["model"] == "provider/model"
        assert body["language"] == "fa"
        assert body["temperature"] == 0
        assert body["input_audio"] == {
            "data": base64.b64encode(b"audio-bytes").decode("ascii"),
            "format": "wav",
        }
        return httpx.Response(200, json={"text": "متن", "usage": {"cost": 0.02}})

    row = EvalRow(0, "clip.wav", audio, "متن", "source")
    with OpenRouterClient(
        "secret",
        timeout=5,
        attempts=1,
        retry_delay=0,
        transport=httpx.MockTransport(handler),
    ) as client:
        assert client.key_status()["usage"] == 2
        assert client.transcribe("provider/model", row, "fa")["text"] == "متن"

    assert [request.url.path for request in requests] == [
        "/api/v1/key",
        "/api/v1/audio/transcriptions",
    ]


def test_resume_rejects_changed_manifest_or_model_list(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path / "mixed")
    output = tmp_path / "results"
    assert evaluate(
        dataset_root=dataset,
        models=["model/a"],
        output_dir=output,
        max_run_cost=Decimal("5"),
        api_key="test-key",
        client_factory=FakeOpenRouterClient,
    ) == 0

    with pytest.raises(ValueError, match="differs"):
        evaluate(
            dataset_root=dataset,
            models=["model/b"],
            output_dir=output,
            max_run_cost=Decimal("5"),
            resume=True,
            api_key="test-key",
            client_factory=FakeOpenRouterClient,
        )
