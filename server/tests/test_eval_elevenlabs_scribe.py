from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from ml.asr.eval_elevenlabs_scribe import ElevenLabsClient, evaluate
from ml.asr.eval_openrouter_stt import EvalRow


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


class FakeElevenLabsClient:
    predictions = {0: "سلام، دنیا!", 1: "کتاب خوب"}
    calls: list[int] = []

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        assert api_key == "test-key"

    def __enter__(self) -> FakeElevenLabsClient:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def transcribe(self, model: str, row: EvalRow, language: str, seed: int) -> dict[str, Any]:
        assert model == "scribe_v2"
        assert language == "fa"
        assert seed == 0
        self.calls.append(row.index)
        return {
            "text": self.predictions[row.index],
            "language_code": "fa",
            "language_probability": 0.99,
            "words": [],
            "_response_headers": {"request-id": f"req-{row.index}"},
        }


def reset_fake() -> None:
    FakeElevenLabsClient.calls = []


def test_evaluation_writes_predictions_metrics_cost_and_provider_response(
    tmp_path: Path,
) -> None:
    reset_fake()
    output = tmp_path / "results"

    exit_code = evaluate(
        dataset_root=make_dataset(tmp_path / "mixed"),
        output_dir=output,
        max_estimated_cost=Decimal("1"),
        api_key="test-key",
        client_factory=FakeElevenLabsClient,
        duration_reader=lambda _: 60.0,
    )

    assert exit_code == 0
    records = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["prediction"] for record in records] == ["سلام، دنیا!", "کتاب خوب"]
    assert [record["source_dataset"] for record in records] == ["dataset-a", "dataset-b"]
    assert records[0]["prediction_normalized"] == "سلام دنیا"
    assert records[0]["estimated_cost_usd"] == pytest.approx(0.22 / 60)
    assert records[0]["provider_response"]["language_code"] == "fa"
    assert records[0]["response_headers"]["request-id"] == "req-0"

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["complete"] is True
    assert metrics["overall"]["wer"] == 0
    assert {item["source_dataset"] for item in metrics["datasets"]} == {
        "dataset-a",
        "dataset-b",
    }
    assert (output / "predictions.tsv").is_file()
    config_text = (output / "run_config.json").read_text(encoding="utf-8")
    assert "test-key" not in config_text


def test_cost_guard_stops_and_resume_skips_checkpointed_rows(tmp_path: Path) -> None:
    reset_fake()
    dataset = make_dataset(tmp_path / "mixed")
    output = tmp_path / "results"

    stopped = evaluate(
        dataset_root=dataset,
        output_dir=output,
        max_estimated_cost=Decimal("0.004"),
        api_key="test-key",
        client_factory=FakeElevenLabsClient,
        duration_reader=lambda _: 60.0,
    )
    assert stopped == 2
    assert FakeElevenLabsClient.calls == [0]

    resumed = evaluate(
        dataset_root=dataset,
        output_dir=output,
        max_estimated_cost=Decimal("1"),
        resume=True,
        api_key="test-key",
        client_factory=FakeElevenLabsClient,
        duration_reader=lambda _: 60.0,
    )
    assert resumed == 0
    assert FakeElevenLabsClient.calls == [0, 1]


def test_estimated_cost_guard_stops_before_sending_clip(tmp_path: Path) -> None:
    reset_fake()
    exit_code = evaluate(
        dataset_root=make_dataset(tmp_path / "mixed"),
        output_dir=tmp_path / "results",
        max_estimated_cost=Decimal("0.001"),
        api_key="test-key",
        client_factory=FakeElevenLabsClient,
        duration_reader=lambda _: 60.0,
    )
    assert exit_code == 2
    assert FakeElevenLabsClient.calls == []


def test_client_uses_only_multipart_scribe_v2_endpoint(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio-bytes")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["xi-api-key"] == "secret"
        body = request.content
        assert b'name="model_id"' in body and b"scribe_v2" in body
        assert b'name="language_code"' in body and b"fa" in body
        assert b'name="tag_audio_events"' in body and b"false" in body
        assert b'name="seed"' in body and b"42" in body
        assert b"audio-bytes" in body
        return httpx.Response(
            200,
            headers={"request-id": "request-123"},
            json={"text": "متن", "language_code": "fa", "words": []},
        )

    row = EvalRow(0, "clip.wav", audio, "متن", "source")
    with ElevenLabsClient(
        "secret", timeout=5, attempts=1, retry_delay=0,
        transport=httpx.MockTransport(handler),
    ) as client:
        response = client.transcribe("scribe_v2", row, "fa", 42)
        assert response["text"] == "متن"
        assert response["_response_headers"]["request-id"] == "request-123"

    assert [request.url.path for request in requests] == ["/v1/speech-to-text"]


def test_client_reopens_audio_stream_for_retry(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"complete-audio")
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        if len(bodies) == 1:
            return httpx.Response(500, json={"detail": "temporary"})
        return httpx.Response(200, json={"text": "متن", "words": []})

    row = EvalRow(0, "clip.wav", audio, "متن", "source")
    with ElevenLabsClient(
        "secret", timeout=5, attempts=2, retry_delay=0,
        transport=httpx.MockTransport(handler),
    ) as client:
        assert client.transcribe("scribe_v2", row, "fa", 0)["text"] == "متن"

    assert len(bodies) == 2
    assert all(b"complete-audio" in body for body in bodies)
