from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from server.app.core.config import AppConfig, ModelConfig
from server.app.services.transcription import ModelRegistry, resolve_device


def settings(max_seconds: float = 2.0) -> AppConfig:
    return AppConfig(
        title="Test",
        max_upload_mb=1,
        max_audio_seconds=max_seconds,
        sample_rate=16_000,
        models=(ModelConfig("test", "Test", "whisper", "checkpoint"),),
    )


def test_registry_rejects_unknown_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown model"):
        ModelRegistry(settings()).transcribe("missing", tmp_path / "missing.wav")


def test_registry_rejects_audio_over_duration_limit(tmp_path: Path) -> None:
    audio = tmp_path / "long.wav"
    sf.write(audio, np.zeros(48_000, dtype=np.float32), 16_000)

    with pytest.raises(ValueError, match="2 seconds or shorter"):
        ModelRegistry(settings()).transcribe("test", audio)


def test_auto_device_prefers_cuda_when_available(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device("auto") == "cuda"


def test_registry_status_changes_after_model_load(monkeypatch) -> None:
    registry = ModelRegistry(settings())
    backend = type("Backend", (), {"device": "cpu"})()
    monkeypatch.setattr(registry, "_load", lambda _config: backend)

    assert registry.status("test").state == "cold"
    registry._backend(settings().models[0])

    status = registry.status("test")
    assert status.state == "ready"
    assert status.device in {"cpu", "cuda"}
