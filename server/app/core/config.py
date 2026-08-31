from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    id: str
    label: str
    backend: str
    checkpoint: str
    description: str = ""
    processor: str | None = None
    base_asr_checkpoint: str | None = None
    model_name: str | None = None
    language: str = "Persian"
    task: str = "transcribe"
    device: str = "auto"
    generation_max_length: int = 225


@dataclass(frozen=True)
class AppConfig:
    title: str
    max_upload_mb: int
    max_audio_seconds: float
    sample_rate: int
    models: tuple[ModelConfig, ...]

    def model(self, model_id: str) -> ModelConfig:
        for item in self.models:
            if item.id == model_id:
                return item
        raise KeyError(model_id)


def _config_path() -> Path:
    return Path(os.environ.get("ASR_SERVER_CONFIG", "configs/server.yaml")).expanduser()


def _required_text(data: dict[str, Any], key: str, context: str) -> str:
    value = str(data.get(key, "")).strip()
    if not value:
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    path = _config_path()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    raw_models = raw.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("models must be a non-empty list")
    models: list[ModelConfig] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_models):
        if not isinstance(item, dict):
            raise ValueError(f"models[{index}] must be a mapping")
        context = f"models[{index}]"
        model_id = _required_text(item, "id", context)
        if model_id in seen:
            raise ValueError(f"duplicate model id: {model_id}")
        seen.add(model_id)
        backend = _required_text(item, "backend", context)
        if backend not in {"whisper", "fastconformer", "fusion"}:
            raise ValueError(f"unsupported backend {backend!r} for {model_id}")
        models.append(
            ModelConfig(
                id=model_id,
                label=_required_text(item, "label", context),
                backend=backend,
                checkpoint=_required_text(item, "checkpoint", context),
                description=str(item.get("description", "")).strip(),
                processor=item.get("processor"),
                base_asr_checkpoint=item.get("base_asr_checkpoint"),
                model_name=item.get("model_name"),
                language=str(item.get("language", "Persian")),
                task=str(item.get("task", "transcribe")),
                device=str(item.get("device", "auto")),
                generation_max_length=int(item.get("generation_max_length", 225)),
            )
        )

    app = raw.get("app") or {}
    return AppConfig(
        title=str(app.get("title", "Persian speech recognition")),
        max_upload_mb=int(app.get("max_upload_mb", 25)),
        max_audio_seconds=float(app.get("max_audio_seconds", 30)),
        sample_rate=int(app.get("sample_rate", 16_000)),
        models=tuple(models),
    )
