from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ml.utils.audio import load_audio, resample_audio
from server.app.core.config import AppConfig, ModelConfig


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    model_id: str
    device: str
    duration_seconds: float
    processing_seconds: float


@dataclass(frozen=True)
class ModelStatus:
    model_id: str
    label: str
    state: str
    device: str


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This model is configured for CUDA, but CUDA is unavailable")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"invalid device {requested!r}")
    return requested


class _WhisperBackend:
    def __init__(self, config: ModelConfig) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = resolve_device(config.device)
        checkpoint = Path(config.checkpoint)
        if (
            checkpoint.is_absolute()
            or config.checkpoint.startswith(("./", "../", "models/", "artifacts/"))
        ) and not checkpoint.exists():
            raise FileNotFoundError(f"model checkpoint not found: {checkpoint}")
        self.processor = WhisperProcessor.from_pretrained(
            config.processor or config.checkpoint,
            language=config.language,
            task=config.task,
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(config.checkpoint).to(self.device).eval()
        self.generation_max_length = config.generation_max_length

    def transcribe(self, audio: np.ndarray, sample_rate: int, _path: Path) -> str:
        import torch

        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        with torch.inference_mode():
            predicted = self.model.generate(
                inputs.input_features.to(self.device),
                max_new_tokens=self.generation_max_length,
            )
        return str(self.processor.batch_decode(predicted, skip_special_tokens=True)[0]).strip()


class _FastConformerBackend:
    def __init__(self, config: ModelConfig) -> None:
        package_dir = Path(__file__).resolve().parents[3] / "ml" / "fa_fastconformer"
        if str(package_dir) not in sys.path:
            sys.path.insert(0, str(package_dir))
        from model import FastConformerCTC

        self.device = resolve_device(config.device)
        checkpoint = Path(config.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"model checkpoint not found: {checkpoint}")
        if checkpoint.suffix == ".nemo":
            self.model = FastConformerCTC.from_nemo(str(checkpoint), map_location="cpu")
        else:
            self.model = FastConformerCTC.from_pretrained(str(checkpoint), map_location="cpu")
        self.model.to(self.device).eval()

    def transcribe(self, _audio: np.ndarray, sample_rate: int, path: Path) -> str:
        result = self.model.transcribe(
            [str(path)], batch_size=1, device=self.device, target_sr=sample_rate, progress=False
        )
        return str(result[0]).strip()


class _FusionBackend:
    def __init__(self, config: ModelConfig) -> None:
        from transformers import WhisperTokenizer

        from ml.fusion.eval_fusion import configure_generation, load_fusion_model

        checkpoint = Path(config.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"model checkpoint not found: {checkpoint}")
        self.model_name = config.model_name or "openai/whisper-small"
        self.device = resolve_device(config.device)
        self.max_tokens = config.generation_max_length
        self.tokenizer = WhisperTokenizer.from_pretrained(config.processor or self.model_name)
        self.model, _ = load_fusion_model(
            checkpoint,
            base_asr_checkpoint=config.base_asr_checkpoint or self.model_name,
            model_name=self.model_name,
        )
        configure_generation(self.model, config.language, config.task)
        self.model.to(self.device).eval()

    def transcribe(self, audio: np.ndarray, sample_rate: int, _path: Path) -> str:
        import torch

        from ml.asr.whisper_features import waveform_to_log_mel

        mel = waveform_to_log_mel(audio, sample_rate=sample_rate, model_name=self.model_name)
        with torch.inference_mode():
            predicted = self.model.generate(
                mel.unsqueeze(0).to(self.device), max_new_tokens=self.max_tokens
            )
        return str(self.tokenizer.batch_decode(predicted, skip_special_tokens=True)[0]).strip()


class ModelRegistry:
    def __init__(self, settings: AppConfig) -> None:
        self.settings = settings
        self._models: dict[str, Any] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._states: dict[str, str] = {}
        self._registry_lock = threading.Lock()

    def _load(self, config: ModelConfig) -> Any:
        factories = {
            "whisper": _WhisperBackend,
            "fastconformer": _FastConformerBackend,
            "fusion": _FusionBackend,
        }
        return factories[config.backend](config)

    def _backend(self, config: ModelConfig) -> tuple[Any, threading.Lock]:
        with self._registry_lock:
            lock = self._locks.setdefault(config.id, threading.Lock())
            if config.id not in self._models:
                self._states[config.id] = "loading"
                try:
                    self._models[config.id] = self._load(config)
                except Exception:
                    self._states[config.id] = "error"
                    raise
                self._states[config.id] = "ready"
            return self._models[config.id], lock

    def status(self, model_id: str) -> ModelStatus:
        try:
            config = self.settings.model(model_id)
        except KeyError as exc:
            raise ValueError(f"unknown model: {model_id}") from exc
        return ModelStatus(
            model_id=model_id,
            label=config.label,
            state=self._states.get(model_id, "cold"),
            device=resolve_device(config.device),
        )

    def transcribe(self, model_id: str, audio_path: Path) -> TranscriptionResult:
        try:
            config = self.settings.model(model_id)
        except KeyError as exc:
            raise ValueError(f"unknown model: {model_id}") from exc
        audio, source_rate = load_audio(audio_path)
        duration = len(audio) / source_rate
        if duration <= 0:
            raise ValueError("the audio file is empty")
        if duration > self.settings.max_audio_seconds:
            raise ValueError(f"audio must be {self.settings.max_audio_seconds:g} seconds or shorter")
        audio = resample_audio(audio, source_rate, self.settings.sample_rate)
        backend, lock = self._backend(config)
        started = time.perf_counter()
        with lock:
            text = backend.transcribe(audio, self.settings.sample_rate, audio_path)
        return TranscriptionResult(
            text=text,
            model_id=model_id,
            device=backend.device,
            duration_seconds=round(duration, 2),
            processing_seconds=round(time.perf_counter() - started, 2),
        )


@lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    from server.app.core.config import get_settings

    return ModelRegistry(get_settings())
