"""Transcribe segmented long audio with Whisper and normalize Persian labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf
import yaml

from ml.asr.eval_whisper_small import resolve_processor_source
from ml.speech_data.long_audio_asr_pipeline.segment_audio import (
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from ml.speech_data.text_normalization import normalize_persian_asr_text


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "checkpoint": None,
        "processor": "openai/whisper-medium",
        "language": "Persian",
        "task": "transcribe",
    },
    "inference": {
        "device": "auto",
        "mixed_precision": "auto",
        "batch_size": 1,
        "generation_max_length": 225,
    },
}

TRANSCRIPTION_PIPELINE_VERSION = 1
PENDING_SNAPSHOT_NAME = "transcription_pending_snapshot.jsonl"

OPERATIONAL_REASONS = {
    "audio_read_failed",
    "checksum_mismatch",
    "inference_failed",
    "invalid_audio",
    "invalid_segment_path",
    "missing_audio",
}

ARTIFACT_NAMES = (
    "transcription.tsv",
    "transcriptions.jsonl",
    "transcription_rejected.jsonl",
    PENDING_SNAPSHOT_NAME,
    "transcription_summary.json",
    "transcription_run.json",
    "transcription_effective_config.yaml",
)


@dataclass(frozen=True)
class TranscriptionAudit:
    clips_total: int
    clips_processed: int
    clips_reused: int
    clips_accepted: int
    clips_rejected: int
    operational_failures: int


class SegmentTranscriber(Protocol):
    def transcribe(self, paths: Sequence[Path]) -> list[str]: ...


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_local_path(raw_path: str | Path, config_path: Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    candidates = [path] if path.is_absolute() else [path, config_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"path does not exist: {raw_path}")


def checkpoint_fingerprint(path: Path) -> str:
    """Hash the model files that can affect ``from_pretrained`` inference."""
    digest = hashlib.sha256()
    if path.is_file():
        files = [path]
        root = path.parent
    else:
        fixed_files = (path / "config.json", path / "generation_config.json")
        files = sorted(
            {
                *[item for item in fixed_files if item.is_file()],
                *[item for item in path.glob("*.safetensors") if item.is_file()],
                *[item for item in path.glob("pytorch_model*.bin") if item.is_file()],
            }
        )
        root = path
    if not files:
        raise ValueError(f"checkpoint contains no model files: {path}")
    for file_path in files:
        digest.update(file_path.relative_to(root).as_posix().encode("utf-8"))
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("configuration must be a YAML mapping")
    config = deep_merge(DEFAULT_CONFIG, loaded)
    model = config.get("model")
    inference = config.get("inference")
    if not isinstance(model, dict) or not isinstance(inference, dict):
        raise ValueError("configuration must contain model and inference mappings")
    checkpoint_value = model.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value.strip():
        raise ValueError("model.checkpoint must be a non-empty local path")
    checkpoint = _resolve_local_path(checkpoint_value, path)
    processor = model.get("processor")
    if not isinstance(processor, str) or not processor.strip():
        raise ValueError("model.processor must be a non-empty model id or local path")
    if not isinstance(model.get("language"), str) or not str(model["language"]).strip():
        raise ValueError("model.language must be a non-empty string")
    if model.get("task") != "transcribe":
        raise ValueError("model.task must be transcribe")
    if inference.get("device") not in {"auto", "cuda", "cpu"}:
        raise ValueError("inference.device must be auto, cuda, or cpu")
    if inference.get("mixed_precision") not in {"auto", "true", "false", True, False}:
        raise ValueError("inference.mixed_precision must be auto, true, or false")
    for field in ("batch_size", "generation_max_length"):
        value = inference.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"inference.{field} must be an integer greater than zero")

    model["checkpoint"] = str(checkpoint)
    model["processor"] = resolve_processor_source(processor, path)
    model["checkpoint_fingerprint"] = checkpoint_fingerprint(checkpoint)
    config["pipeline_version"] = TRANSCRIPTION_PIPELINE_VERSION
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
    return config, digest


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("inference.device is cuda, but CUDA is not available")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


class WhisperSegmentTranscriber:
    def __init__(self, config: dict[str, Any]) -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_config = config["model"]
        inference = config["inference"]
        self.processor = WhisperProcessor.from_pretrained(
            str(model_config["processor"]),
            language=str(model_config["language"]),
            task="transcribe",
        )
        self.device = _resolve_device(str(inference["device"]))
        mixed_precision = inference["mixed_precision"]
        use_fp16 = self.device == "cuda" and mixed_precision in {"auto", "true", True}
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.model = WhisperForConditionalGeneration.from_pretrained(
            str(model_config["checkpoint"]),
            dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        self.language = str(model_config["language"])
        self.maximum_length = int(inference["generation_max_length"])

    def transcribe(self, paths: Sequence[Path]) -> list[str]:
        import torch

        audio: list[np.ndarray] = []
        for path in paths:
            samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
            if sample_rate != 16000 or samples.ndim != 1:
                raise ValueError(f"audio must be mono 16 kHz: {path}")
            audio.append(np.asarray(samples, dtype=np.float32))
        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        generation_inputs: dict[str, Any] = {
            "input_features": inputs.input_features.to(device=self.device, dtype=self.dtype),
        }
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            generation_inputs["attention_mask"] = attention_mask.to(self.device)
        with torch.inference_mode():
            predicted = self.model.generate(
                **generation_inputs,
                language=self.language,
                task="transcribe",
                max_length=self.maximum_length,
                num_beams=1,
                do_sample=False,
            )
        return list(self.processor.batch_decode(predicted, skip_special_tokens=True))


def _validate_input_root(input_root: Path) -> list[dict[str, Any]]:
    if not (input_root / "run.json").is_file() or not (input_root / "segments.jsonl").is_file():
        raise ValueError("input root must contain run.json and segments.jsonl from segment_audio")
    segments = read_jsonl(input_root / "segments.jsonl")
    if not segments:
        raise ValueError("segments.jsonl contains no clips")
    seen: set[str] = set()
    for index, segment in enumerate(segments, start=1):
        segment_id = segment.get("id")
        if not isinstance(segment_id, str) or not segment_id.strip():
            raise ValueError(f"segments.jsonl:{index} requires a non-empty string id")
        if segment_id in seen:
            raise ValueError(f"segments.jsonl contains duplicate id: {segment_id}")
        seen.add(segment_id)
    return sorted(segments, key=lambda item: str(item["id"]))


def _clip_path(input_root: Path, segment: dict[str, Any]) -> tuple[Path | None, str | None, str | None]:
    raw_path = segment.get("path")
    if not isinstance(raw_path, str):
        return None, "invalid_segment_path", "segment path must be a string"
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "clips":
        return None, "invalid_segment_path", f"unsafe segment path: {raw_path}"
    path = input_root / relative
    if not path.is_file():
        return None, "missing_audio", str(path)
    expected_checksum = segment.get("clip_checksum")
    actual_checksum = sha256_file(path)
    if not isinstance(expected_checksum, str) or expected_checksum != actual_checksum:
        return None, "checksum_mismatch", f"expected {expected_checksum}, got {actual_checksum}"
    try:
        info = sf.info(path)
    except Exception as error:
        return None, "audio_read_failed", f"{type(error).__name__}: {error}"
    if info.frames <= 0 or info.samplerate != 16000 or info.channels != 1 or not math.isfinite(info.duration):
        return None, "invalid_audio", "clip must be non-empty, finite, mono, and 16 kHz"
    return path, None, None


def _write_tsv_atomic(path: Path, accepted: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "sentence"], delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for record in sorted(accepted, key=lambda item: str(item["id"])):
                relative = Path(str(record["path"])).relative_to("clips").as_posix()
                writer.writerow({"path": relative, "sentence": record["normalized_transcript"]})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _persist(
    output_root: Path,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    audit: TranscriptionAudit,
) -> None:
    accepted.sort(key=lambda item: str(item["id"]))
    rejected.sort(key=lambda item: str(item["id"]))
    write_jsonl_atomic(output_root / "transcriptions.jsonl", accepted)
    write_jsonl_atomic(output_root / "transcription_rejected.jsonl", rejected)
    _write_tsv_atomic(output_root / "transcription.tsv", accepted)
    write_json_atomic(output_root / "transcription_summary.json", asdict(audit))


def _make_rejection(
    segment: dict[str, Any], config_digest: str, reason: str, detail: str, raw: str | None = None
) -> dict[str, Any]:
    return {
        "id": segment["id"],
        "path": segment.get("path"),
        "clip_checksum": segment.get("clip_checksum"),
        "config_digest": config_digest,
        "reason": reason,
        "detail": detail,
        "raw_transcript": raw,
        "operational": reason in OPERATIONAL_REASONS,
    }


def process_transcription(
    input_root: Path,
    output_root: Path,
    config: dict[str, Any],
    config_digest: str,
    transcriber_factory: Callable[[dict[str, Any]], SegmentTranscriber] = WhisperSegmentTranscriber,
) -> TranscriptionAudit:
    segments = _validate_input_root(input_root)
    output_root.mkdir(parents=True, exist_ok=True)
    prior_accepted = {item["id"]: item for item in read_jsonl(output_root / "transcriptions.jsonl")} if (output_root / "transcriptions.jsonl").is_file() else {}
    prior_rejected = {item["id"]: item for item in read_jsonl(output_root / "transcription_rejected.jsonl")} if (output_root / "transcription_rejected.jsonl").is_file() else {}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    worklist: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], Path]] = []
    reused = 0
    processed = 0

    for segment in segments:
        prior = prior_accepted.get(segment["id"]) or prior_rejected.get(segment["id"])
        if (
            prior
            and not prior.get("operational", False)
            and prior.get("config_digest") == config_digest
            and prior.get("clip_checksum") == segment.get("clip_checksum")
        ):
            (rejected if "reason" in prior else accepted).append(prior)
            reused += 1
            continue
        worklist.append(segment)

    # Freeze the exact set selected from the startup view of segments.jsonl.
    # The segmenter may publish more records while inference is running; those
    # records intentionally wait for the next transcription invocation.
    write_jsonl_atomic(output_root / PENDING_SNAPSHOT_NAME, worklist)

    for segment in worklist:
        path, reason, detail = _clip_path(input_root, segment)
        if reason is not None:
            rejected.append(_make_rejection(segment, config_digest, reason, detail or reason))
            processed += 1
        else:
            pending.append((segment, path))  # type: ignore[arg-type]

    try:
        transcriber = transcriber_factory(config) if pending else None
    except Exception as error:
        raise RuntimeError(f"could not initialize Whisper transcription: {type(error).__name__}: {error}") from error
    batch_size = int(config["inference"]["batch_size"])
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        try:
            raw_texts = transcriber.transcribe([path for _, path in batch])  # type: ignore[union-attr]
            if len(raw_texts) != len(batch):
                raise RuntimeError("transcriber returned a different number of results than inputs")
            results = list(zip(batch, raw_texts, strict=True))
        except Exception:
            results = []
            for item in batch:
                try:
                    raw = transcriber.transcribe([item[1]])[0]  # type: ignore[union-attr]
                    results.append((item, raw))
                except Exception as error:
                    segment = item[0]
                    rejected.append(
                        _make_rejection(segment, config_digest, "inference_failed", f"{type(error).__name__}: {error}")
                    )
                    processed += 1

        for (segment, _), raw in results:
            raw = str(raw).strip()
            normalized = normalize_persian_asr_text(raw)
            if not normalized:
                rejected.append(
                    _make_rejection(segment, config_digest, "normalization_rejected", "shared Persian normalization rejected or emptied the transcript", raw)
                )
            else:
                accepted.append(
                    {
                        "id": segment["id"],
                        "source_id": segment.get("source_id"),
                        "path": segment["path"],
                        "clip_checksum": segment["clip_checksum"],
                        "raw_transcript": raw,
                        "normalized_transcript": normalized,
                        "checkpoint": config["model"]["checkpoint"],
                        "checkpoint_fingerprint": config["model"]["checkpoint_fingerprint"],
                        "processor": config["model"]["processor"],
                        "language": config["model"]["language"],
                        "task": "transcribe",
                        "generation": {
                            "generation_max_length": config["inference"]["generation_max_length"],
                            "num_beams": 1,
                            "do_sample": False,
                        },
                        "config_digest": config_digest,
                    }
                )
            processed += 1

        audit = TranscriptionAudit(
            len(segments), processed, reused, len(accepted), len(rejected), sum(bool(item["operational"]) for item in rejected)
        )
        _persist(output_root, accepted, rejected, audit)

    if not pending:
        audit = TranscriptionAudit(
            len(segments), processed, reused, len(accepted), len(rejected), sum(bool(item["operational"]) for item in rejected)
        )
        _persist(output_root, accepted, rejected, audit)
    return audit


def run_transcription(
    input_root: Path,
    config: dict[str, Any],
    config_digest: str,
    *,
    force: bool = False,
    transcriber_factory: Callable[[dict[str, Any]], SegmentTranscriber] = WhisperSegmentTranscriber,
) -> TranscriptionAudit:
    input_root = input_root.resolve()
    run_path = input_root / "transcription_run.json"
    if run_path.is_file():
        prior = json.loads(run_path.read_text(encoding="utf-8"))
        if prior.get("config_digest") != config_digest and not force:
            raise ValueError("transcription uses a different configuration digest; pass --force to replace it")
    elif any((input_root / name).exists() for name in ARTIFACT_NAMES if name != "transcription_run.json") and not force:
        raise FileExistsError("input root contains untracked transcription artifacts; pass --force to replace them")

    if force:
        staging = Path(tempfile.mkdtemp(prefix=".transcription-", dir=input_root))
        try:
            audit = process_transcription(input_root, staging, config, config_digest, transcriber_factory)
            if audit.operational_failures:
                raise RuntimeError("forced transcription had operational failures; existing artifacts were preserved")
            write_json_atomic(staging / "transcription_run.json", {"config_digest": config_digest})
            (staging / "transcription_effective_config.yaml").write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=True), encoding="utf-8"
            )
            for name in ARTIFACT_NAMES:
                (staging / name).replace(input_root / name)
            return audit
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    write_json_atomic(run_path, {"config_digest": config_digest})
    (input_root / "transcription_effective_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=True), encoding="utf-8"
    )
    return process_transcription(input_root, input_root, config, config_digest, transcriber_factory)


def print_audit(audit: TranscriptionAudit, input_root: Path) -> None:
    print("Whisper segment transcription summary")
    print(f"  input root: {input_root}")
    for key, value in asdict(audit).items():
        print(f"  {key.replace('_', ' ')}: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe a fixed startup snapshot of segment_audio output with Whisper Medium, "
            "checkpoint results, and normalize Persian labels."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML transcription configuration file.")
    parser.add_argument(
        "--input-root",
        required=True,
        type=Path,
        help=(
            "Directory created by segment_audio, containing clips/ and segments.jsonl; "
            "identical reruns process only newly published or retryable segments."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace transcription artifacts created with different settings.",
    )
    args = parser.parse_args(argv)
    try:
        config, digest = load_config(args.config)
        audit = run_transcription(args.input_root, config, digest, force=args.force)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print_audit(audit, args.input_root)
    return 1 if audit.operational_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
