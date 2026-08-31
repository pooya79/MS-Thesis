from __future__ import annotations

import io
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from server.app import main
from server.app.services.transcription import ModelStatus, TranscriptionResult


class FakeRegistry:
    def status(self, model_id: str) -> ModelStatus:
        return ModelStatus(model_id, "Whisper Small", "ready", "cuda")

    def transcribe(self, model_id: str, audio_path: Path) -> TranscriptionResult:
        assert model_id == "whisper-small"
        assert audio_path.suffix == ".wav"
        return TranscriptionResult("سلام دنیا", model_id, "cuda", 1.0, 0.12)


def wav_bytes(seconds: float = 1.0) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(int(16_000 * seconds), dtype=np.float32), 16_000, format="WAV")
    return buffer.getvalue()


def test_homepage_exposes_model_picker_and_audio_sources() -> None:
    with TestClient(main.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Whisper Small" in response.text
    assert "FastConformer" in response.text
    assert "Upload file" in response.text
    assert "Record voice" in response.text
    assert "/static/css/app.css" in response.text
    assert "/static/js/app.js" in response.text
    assert 'id="loading-panel"' in response.text
    assert "loading-spinner" in response.text


def test_models_endpoint_returns_safe_public_metadata() -> None:
    with TestClient(main.app) as client:
        response = client.get("/api/models")

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["models"]] == [
        "whisper-small", "whisper-medium", "fusion", "fastconformer"
    ]
    assert "checkpoint" not in response.text


def test_transcription_endpoint_returns_result(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_registry", lambda: FakeRegistry())
    monkeypatch.setattr(main, "_convert_to_wav", lambda source, destination: shutil.copyfile(source, destination))
    with TestClient(main.app) as client:
        response = client.post(
            "/api/transcriptions",
            data={"model_id": "whisper-small"},
            files={"audio": ("voice.wav", wav_bytes(), "audio/wav")},
        )

    assert response.status_code == 200
    assert response.json() == {
        "text": "سلام دنیا",
        "model_id": "whisper-small",
        "device": "cuda",
        "duration_seconds": 1.0,
        "processing_seconds": 0.12,
    }


def test_model_status_endpoint_reports_runtime_device(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_registry", lambda: FakeRegistry())
    with TestClient(main.app) as client:
        response = client.get("/api/models/whisper-small/status")

    assert response.status_code == 200
    assert response.json() == {
        "model_id": "whisper-small",
        "label": "Whisper Small",
        "state": "ready",
        "device": "cuda",
    }


def test_transcription_endpoint_rejects_oversized_upload(monkeypatch) -> None:
    monkeypatch.setattr(main, "settings", replace(main.settings, max_upload_mb=0))
    with TestClient(main.app) as client:
        response = client.post(
            "/api/transcriptions",
            data={"model_id": "whisper-small"},
            files={"audio": ("voice.wav", b"audio", "audio/wav")},
        )

    assert response.status_code == 413
