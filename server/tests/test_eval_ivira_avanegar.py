from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from ml.asr.eval_ivira_avanegar import AvanegarClient, evaluate
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


class FakeAvanegarClient:
    predictions = {0: "سلام، دنیا!", 1: "کتاب خوب"}
    calls: list[int] = []

    def __init__(self, gateway_token: str, **kwargs: Any) -> None:
        assert gateway_token == "test-token"

    def __enter__(self) -> FakeAvanegarClient:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def transcribe(self, model: str, row: EvalRow) -> dict[str, Any]:
        assert model == "default"
        self.calls.append(row.index)
        return {
            "text": self.predictions[row.index],
            "units": "5",
            "response_headers": {"x-trace-id": f"trace-{row.index}"},
            "provider_response": {
                "data": {
                    "data": {
                        "aiResponse": {
                            "result": {"text": self.predictions[row.index], "rtf": 0.1},
                            "meta": {"units": 5},
                        }
                    }
                }
            },
        }


def reset_fake() -> None:
    FakeAvanegarClient.calls = []


def test_evaluation_writes_predictions_metrics_units_and_provenance(tmp_path: Path) -> None:
    reset_fake()
    output = tmp_path / "results"

    exit_code = evaluate(
        dataset_root=make_dataset(tmp_path / "mixed"),
        output_dir=output,
        max_run_units=Decimal("100"),
        gateway_token="test-token",
        client_factory=FakeAvanegarClient,
        duration_reader=lambda _: 30.0,
    )

    assert exit_code == 0
    records = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["prediction"] for record in records] == ["سلام، دنیا!", "کتاب خوب"]
    assert [record["source_dataset"] for record in records] == ["dataset-a", "dataset-b"]
    assert records[0]["prediction_normalized"] == "سلام دنیا"
    assert records[0]["provider_units"] == 5
    assert records[0]["request_options"]["punctuation"] is False
    assert records[0]["request_options"]["spokenPunctuation"] is False
    assert records[0]["provider_response"]["data"]["data"]["aiResponse"]["meta"]["units"] == 5

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["complete"] is True
    assert metrics["overall"]["wer"] == 0
    assert metrics["overall"]["provider_units"] == 10
    assert metrics["provider_units_used"] == 10
    assert {item["source_dataset"] for item in metrics["datasets"]} == {
        "dataset-a",
        "dataset-b",
    }
    assert (output / "predictions.tsv").is_file()
    assert "test-token" not in (output / "run_config.json").read_text(encoding="utf-8")


def test_unit_guard_stops_and_resume_skips_checkpointed_rows(tmp_path: Path) -> None:
    reset_fake()
    dataset = make_dataset(tmp_path / "mixed")
    output = tmp_path / "results"

    stopped = evaluate(
        dataset_root=dataset,
        output_dir=output,
        max_run_units=Decimal("5"),
        gateway_token="test-token",
        client_factory=FakeAvanegarClient,
        duration_reader=lambda _: 30.0,
    )
    assert stopped == 2
    assert FakeAvanegarClient.calls == [0]

    resumed = evaluate(
        dataset_root=dataset,
        output_dir=output,
        max_run_units=Decimal("10"),
        resume=True,
        gateway_token="test-token",
        client_factory=FakeAvanegarClient,
        duration_reader=lambda _: 30.0,
    )
    assert resumed == 0
    assert FakeAvanegarClient.calls == [0, 1]


def test_duration_limit_rejects_clip_without_sending_it(tmp_path: Path) -> None:
    reset_fake()
    exit_code = evaluate(
        dataset_root=make_dataset(tmp_path / "mixed"),
        output_dir=tmp_path / "results",
        max_run_units=Decimal("100"),
        gateway_token="test-token",
        client_factory=FakeAvanegarClient,
        duration_reader=lambda _: 60.0,
    )
    assert exit_code == 1
    assert FakeAvanegarClient.calls == []


def test_client_uses_documented_endpoint_and_disables_punctuation(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio-bytes")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["gateway-token"] == "secret"
        body = request.content
        assert b'name="model"' in body and b"default" in body
        for field in (
            b"srt",
            b"inverseNormalizer",
            b"timestamp",
            b"spokenPunctuation",
            b"punctuation",
            b"diarize",
        ):
            assert b'name="' + field + b'"' in body
        assert body.count(b"false") >= 6
        assert b'name="numSpeakers"' in body and b"0" in body
        assert b'name="audio"' in body and b"audio-bytes" in body
        return httpx.Response(
            201,
            headers={"x-trace-id": "trace-123"},
            json={
                "data": {
                    "status": "success",
                    "data": {
                        "aiResponse": {
                            "result": {"text": "متن", "rtf": 0.1},
                            "meta": {"units": 7},
                        }
                    },
                }
            },
        )

    row = EvalRow(0, "clip.wav", audio, "متن", "source")
    with AvanegarClient(
        "secret",
        timeout=5,
        attempts=1,
        retry_delay=0,
        transport=httpx.MockTransport(handler),
    ) as client:
        response = client.transcribe("default", row)

    assert response["text"] == "متن"
    assert response["units"] == "7"
    assert response["response_headers"]["x-trace-id"] == "trace-123"
    assert [request.url.path for request in requests] == ["/avanegar/avanegar/request"]


def test_client_reopens_audio_stream_for_retry(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"complete-audio")
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        if len(bodies) == 1:
            return httpx.Response(500, json={"message": "temporary"})
        return httpx.Response(
            201,
            json={
                "data": {
                    "data": {
                        "aiResponse": {
                            "result": {"text": "متن"},
                            "meta": {"units": 1},
                        }
                    }
                }
            },
        )

    row = EvalRow(0, "clip.wav", audio, "متن", "source")
    with AvanegarClient(
        "secret",
        timeout=5,
        attempts=2,
        retry_delay=0,
        transport=httpx.MockTransport(handler),
    ) as client:
        assert client.transcribe("default", row)["text"] == "متن"

    assert len(bodies) == 2
    assert all(b"complete-audio" in body for body in bodies)
