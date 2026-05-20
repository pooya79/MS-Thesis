from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from server.app.main import app


def login(client: TestClient) -> None:
    response = client.post(
        "/login",
        data={"password": "test-password", "next": "/experiments/speech-degradation"},
        follow_redirects=False,
    )
    assert response.status_code == 303


def wav_bytes(duration_seconds: float = 0.5, sample_rate: int = 16000) -> bytes:
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False, dtype=np.float32)
    audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def test_speech_degradation_page_requires_auth() -> None:
    with TestClient(app) as client:
        response = client.get("/experiments/speech-degradation", headers={"accept": "text/html"}, follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"].startswith("/login?next=%2Fexperiments%2Fspeech-degradation")


def test_speech_degradation_page_renders_for_authenticated_user() -> None:
    with TestClient(app) as client:
        login(client)
        response = client.get("/experiments/speech-degradation")

    assert response.status_code == 200
    assert "/static/css/speech_degradation.css" in response.text
    assert "/static/js/speech_degradation.js" in response.text
    assert "Generate degraded pair" in response.text
    assert "RIR convolution" in response.text
    assert "DEMAND noise" in response.text
    assert "narrowband" in response.text
    assert "wideband" in response.text
    assert "ffmpeg" in response.text
    assert "waveform dropout" in response.text
    assert "JSONL-style metadata" in response.text
    assert 'class="nav-item active">Experiments</a>' in response.text


def test_generated_file_route_requires_auth() -> None:
    with TestClient(app) as client:
        response = client.get("/experiments/speech-degradation/files/demo/input")

    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required"}


def test_speech_degradation_generation_creates_session_files(tmp_path: Path, monkeypatch) -> None:
    from server.app.services import speech_degradation_demo as demo_service

    monkeypatch.setattr(demo_service, "DEMO_ROOT", tmp_path / "demos")

    with TestClient(app) as client:
        login(client)
        response = client.post(
            "/experiments/speech-degradation/generate",
            data={
                "reverb_mode": "disabled",
                "snr_bucket": "5:10",
                "gain_db": "0",
                "channel_path": "wideband",
                "codec": "pass_through",
            },
            files={"audio_file": ("clean.wav", wav_bytes(), "audio/wav")},
        )

        assert response.status_code == 200
        assert "Uploaded input" in response.text
        assert "Degraded input" in response.text
        demo_ids = client.cookies

        metadata_files = list((tmp_path / "demos").glob("*/metadata.json"))
        assert len(metadata_files) == 1
        metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
        assert metadata["channel_path"] == "wideband"
        assert metadata["codec"] == "pass_through"
        assert metadata["normalization"] == "peak_safety"
        assert metadata["seed"] is not None
        assert Path(metadata["clean_path"]).exists()
        assert Path(metadata["degraded_path"]).exists()

        clean_info = sf.info(metadata["clean_path"])
        degraded_info = sf.info(metadata["degraded_path"])
        assert clean_info.samplerate == degraded_info.samplerate == 16000
        assert clean_info.frames == degraded_info.frames

        file_response = client.get(f"/experiments/speech-degradation/files/{metadata['demo_id']}/degraded")

    assert demo_ids is not None
    assert file_response.status_code == 200
    assert file_response.headers["content-type"].startswith("audio/wav")


def test_rir_enabled_without_assets_returns_helpful_error(tmp_path: Path, monkeypatch) -> None:
    from server.app.services import speech_degradation_demo as demo_service

    monkeypatch.setattr(demo_service, "DEMO_ROOT", tmp_path / "demos")

    with TestClient(app) as client:
        login(client)
        response = client.post(
            "/experiments/speech-degradation/generate",
            data={
                "reverb_mode": "mild",
                "snr_bucket": "5:10",
                "gain_db": "0",
                "channel_path": "wideband",
                "codec": "pass_through",
            },
            files={"audio_file": ("clean.wav", wav_bytes(), "audio/wav")},
        )

    assert response.status_code == 400
    assert "RIR is enabled" in response.text
    assert "Disable" in response.text


def test_noise_enabled_without_assets_returns_helpful_error(tmp_path: Path, monkeypatch) -> None:
    from server.app.services import speech_degradation_demo as demo_service

    monkeypatch.setattr(demo_service, "DEMO_ROOT", tmp_path / "demos")

    with TestClient(app) as client:
        login(client)
        response = client.post(
            "/experiments/speech-degradation/generate",
            data={
                "reverb_mode": "disabled",
                "noise_enabled": "1",
                "snr_bucket": "5:10",
                "gain_db": "0",
                "channel_path": "wideband",
                "codec": "pass_through",
            },
            files={"audio_file": ("clean.wav", wav_bytes(), "audio/wav")},
        )

    assert response.status_code == 400
    assert "Noise is enabled" in response.text
    assert "Disable" in response.text


def test_unsupported_upload_extension_returns_error(tmp_path: Path, monkeypatch) -> None:
    from server.app.services import speech_degradation_demo as demo_service

    monkeypatch.setattr(demo_service, "DEMO_ROOT", tmp_path / "demos")

    with TestClient(app) as client:
        login(client)
        response = client.post(
            "/experiments/speech-degradation/generate",
            data={
                "reverb_mode": "disabled",
                "snr_bucket": "5:10",
                "gain_db": "0",
                "channel_path": "wideband",
                "codec": "pass_through",
            },
            files={"audio_file": ("clean.txt", b"not-audio", "text/plain")},
        )

    assert response.status_code == 400
    assert "Unsupported audio file type" in response.text


def test_too_long_upload_returns_error(tmp_path: Path, monkeypatch) -> None:
    from server.app.services import speech_degradation_demo as demo_service

    monkeypatch.setattr(demo_service, "DEMO_ROOT", tmp_path / "demos")

    with TestClient(app) as client:
        login(client)
        response = client.post(
            "/experiments/speech-degradation/generate",
            data={
                "reverb_mode": "disabled",
                "snr_bucket": "5:10",
                "gain_db": "0",
                "channel_path": "wideband",
                "codec": "pass_through",
            },
            files={"audio_file": ("long.wav", wav_bytes(duration_seconds=61.0), "audio/wav")},
        )

    assert response.status_code == 400
    assert "longer than 60 seconds" in response.text


def test_speech_degradation_static_assets_are_public() -> None:
    with TestClient(app) as client:
        css = client.get("/static/css/speech_degradation.css")
        js = client.get("/static/js/speech_degradation.js")

    assert css.status_code == 200
    assert js.status_code == 200
    assert "tab-panel" in css.text
    assert "data-tab-button" in js.text
