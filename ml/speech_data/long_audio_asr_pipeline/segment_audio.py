from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock, local
from typing import Any, Protocol

import numpy as np
import soundfile as sf
import torch
import yaml

from .segmentation import SegmentationSettings, SpeechInterval, construct_segments


SUPPORTED_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}
OUTPUT_FORMATS = {
    "FLAC": {"extension": ".flac", "subtype": "PCM_16"},
    "MP3": {"extension": ".mp3", "subtype": "MPEG_LAYER_III"},
    "WAV": {"extension": ".wav", "subtype": "PCM_16"},
}
OPERATIONAL_REASONS = {
    "audio_decode_failed",
    "checksum_mismatch",
    "missing_audio",
    "invalid_audio",
    "vad_failed",
    "clip_export_failed",
}


class AudioDecodeError(RuntimeError):
    """The source could not be decoded into valid working audio."""


class ClipExportError(RuntimeError):
    """A proposed interval could not be constructed or exported."""


@dataclass(frozen=True)
class SourceRecord:
    id: str
    path: Path
    expected_checksum: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PipelineAudit:
    sources_total: int
    sources_processed: int
    sources_reused: int
    sources_rejected: int
    clips_written: int
    duration_seconds: float
    operational_failures: int


@dataclass(frozen=True)
class ExecutionSettings:
    vad_engine: str = "pytorch"
    vad_batch_size: int = 1
    torch_threads: int | None = None


@dataclass(frozen=True)
class SourceProcessingResult:
    source: SourceRecord
    source_checksum: str
    source_state: dict[str, Any]
    segments: list[dict[str, Any]]
    vad_records: list[dict[str, Any]]
    rejection: dict[str, Any] | None
    processed: int
    reused: int
    clips_written: int
    duration_seconds: float
    operational_failures: int


class VadDetector(Protocol):
    @property
    def metadata(self) -> dict[str, str]: ...

    def detect(self, path: Path, settings: dict[str, Any]) -> list[SpeechInterval]: ...

    def detect_many(
        self,
        paths: Sequence[Path | None],
        settings: dict[str, Any],
    ) -> list[list[SpeechInterval]]: ...


class UnavailableVadDetector:
    def __init__(self, error: Exception) -> None:
        self.error = error

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "backend": "silero",
            "model": "silero_vad",
            "package": "silero-vad",
            "package_version": importlib.metadata.version("silero-vad"),
            "runtime": "unavailable",
        }

    def detect(self, path: Path, settings: dict[str, Any]) -> list[SpeechInterval]:
        raise self.error

    def detect_many(
        self,
        paths: Sequence[Path | None],
        settings: dict[str, Any],
    ) -> list[list[SpeechInterval]]:
        raise self.error


class SileroVadDetector:
    _FRAMES_PER_READ = 2048

    def __init__(self, *, engine: str = "pytorch", model: Any | None = None) -> None:
        from silero_vad import load_silero_vad

        if engine != "pytorch":
            raise ValueError("Silero VAD engine must be pytorch")
        self.engine = engine
        self.model = model if model is not None else load_silero_vad(onnx=False)
        self.model.eval()

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "backend": "silero",
            "model": "silero_vad",
            "package": "silero-vad",
            "package_version": importlib.metadata.version("silero-vad"),
            "runtime": self.engine,
        }

    def detect(self, path: Path, settings: dict[str, Any]) -> list[SpeechInterval]:
        return self.detect_many([path], settings)[0]

    @staticmethod
    def _frames(path: Path, frame_samples: int) -> Iterable[np.ndarray]:
        block_samples = frame_samples * SileroVadDetector._FRAMES_PER_READ
        with sf.SoundFile(path) as audio:
            while True:
                block = audio.read(block_samples, dtype="float32", always_2d=False)
                if not len(block):
                    return
                valid = len(block)
                padded = math.ceil(valid / frame_samples) * frame_samples
                if padded != valid:
                    block = np.pad(block, (0, padded - valid))
                frames = np.asarray(block, dtype=np.float32).reshape(-1, frame_samples)
                yield from frames
                if valid < block_samples:
                    return

    @staticmethod
    def _intervals(
        probabilities: list[float],
        duration: float,
        settings: dict[str, Any],
    ) -> list[SpeechInterval]:
        sample_rate = int(settings["sample_rate"])
        frame_samples = int(settings["frame_samples"])
        threshold = float(settings["threshold"])
        minimum_speech = float(settings["minimum_speech_seconds"])
        merge_silence = float(settings["merge_silence_seconds"])

        active: list[SpeechInterval] = []
        start_index: int | None = None
        for index, probability in enumerate([*probabilities, 0.0]):
            if probability >= threshold and start_index is None:
                start_index = index
            elif probability < threshold and start_index is not None:
                start = start_index * frame_samples / sample_rate
                end = min(index * frame_samples / sample_rate, duration)
                if end - start >= minimum_speech:
                    values = probabilities[start_index:index]
                    active.append(SpeechInterval(start, end, float(np.mean(values)), max(values)))
                start_index = None

        merged: list[SpeechInterval] = []
        for interval in active:
            if merged and interval.start_sec - merged[-1].end_sec < merge_silence:
                previous = merged[-1]
                merged[-1] = SpeechInterval(
                    previous.start_sec,
                    interval.end_sec,
                    (previous.mean_probability + interval.mean_probability) / 2.0,
                    max(previous.max_probability, interval.max_probability),
                )
            else:
                merged.append(interval)
        return merged

    def detect_many(
        self,
        paths: Sequence[Path | None],
        settings: dict[str, Any],
    ) -> list[list[SpeechInterval]]:
        if not paths:
            return []
        sample_rate = int(settings["sample_rate"])
        frame_samples = int(settings["frame_samples"])
        durations = [
            sf.info(path).duration if path is not None else 0.0
            for path in paths
        ]
        frame_iterators = [
            iter(self._frames(path, frame_samples)) if path is not None else iter(())
            for path in paths
        ]
        probabilities: list[list[float]] = [[] for _ in paths]
        finished = [path is None for path in paths]
        self.model.reset_states()
        with torch.inference_mode():
            while not all(finished):
                frames = np.zeros((len(paths), frame_samples), dtype=np.float32)
                active = [False] * len(paths)
                for index, iterator in enumerate(frame_iterators):
                    if finished[index]:
                        continue
                    try:
                        frames[index] = next(iterator)
                        active[index] = True
                    except StopIteration:
                        finished[index] = True
                if not any(active):
                    break
                tensor = torch.from_numpy(frames)
                model_input = tensor[0] if len(paths) == 1 else tensor
                output = self.model(model_input, sample_rate)
                values = np.asarray(output.detach().cpu()).reshape(-1)
                for index, is_active in enumerate(active):
                    if is_active:
                        probabilities[index].append(float(values[index]))

        return [
            self._intervals(values, duration, settings)
            for values, duration in zip(probabilities, durations, strict=True)
        ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(record)
    return records


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration must be a YAML mapping")
    required = {"audio", "vad", "segmentation"}
    if not required.issubset(value):
        raise ValueError(f"configuration must contain: {', '.join(sorted(required))}")
    audio = value["audio"]
    if not isinstance(audio, dict):
        raise ValueError("audio configuration must be a mapping")
    if audio.get("sample_rate") != 16000 or audio.get("channels") != 1:
        raise ValueError("audio output must use one channel at 16000 Hz")
    output_format = audio.get("format")
    if output_format not in OUTPUT_FORMATS:
        raise ValueError("audio.format must be one of: FLAC, MP3, WAV")
    if output_format in {"FLAC", "WAV"}:
        if audio.get("subtype") != "PCM_16":
            raise ValueError(f"{output_format} audio output must use subtype PCM_16")
        if "bitrate_kbps" in audio:
            raise ValueError(f"{output_format} audio output must not set bitrate_kbps")
    else:
        if "subtype" in audio:
            raise ValueError("MP3 audio output must not set subtype")
        bitrate = audio.get("bitrate_kbps")
        if isinstance(bitrate, bool) or not isinstance(bitrate, int) or not 8 <= bitrate <= 160:
            raise ValueError("MP3 audio output requires integer bitrate_kbps between 8 and 160")
    vad = value["vad"]
    if vad.get("backend") != "silero":
        raise ValueError("vad.backend must be silero")
    if vad.get("sample_rate") != 16000 or vad.get("frame_samples") != 512:
        raise ValueError("Silero VAD must use 16000 Hz and 512-sample frames")
    threshold = vad.get("threshold")
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ValueError("vad.threshold must be between zero and one")
    for field in ("minimum_speech_seconds", "merge_silence_seconds"):
        setting = vad.get(field)
        if not isinstance(setting, (int, float)) or not math.isfinite(setting) or setting <= 0:
            raise ValueError(f"vad.{field} must be finite and greater than zero")
    execution = value.get("execution", {})
    if not isinstance(execution, dict):
        raise ValueError("execution configuration must be a mapping")
    allowed_execution = {"vad_engine", "vad_batch_size", "torch_threads"}
    unknown_execution = set(execution) - allowed_execution
    if unknown_execution:
        raise ValueError(
            f"unknown execution setting(s): {', '.join(sorted(unknown_execution))}"
        )
    if execution.get("vad_engine", "pytorch") != "pytorch":
        raise ValueError("execution.vad_engine must be pytorch")
    for field in ("vad_batch_size", "torch_threads"):
        setting = execution.get(field)
        if setting is not None and (
            isinstance(setting, bool) or not isinstance(setting, int) or setting < 1
        ):
            raise ValueError(
                f"execution.{field} must be an integer greater than or equal to one"
            )
    SegmentationSettings(**value["segmentation"])
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return value, f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def execution_settings(config: dict[str, Any]) -> ExecutionSettings:
    values = config.get("execution", {})
    return ExecutionSettings(
        vad_engine=str(values.get("vad_engine", "pytorch")),
        vad_batch_size=int(values.get("vad_batch_size", 1)),
        torch_threads=(
            int(values["torch_threads"])
            if values.get("torch_threads") is not None
            else None
        ),
    )


def safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    if not normalized:
        normalized = "source"
    return normalized[:96]


def _generated_id(label: str, stem: str) -> str:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:10]
    return f"{safe_id(stem)}-{digest}"


def discover_inputs(inputs: Sequence[Path]) -> list[SourceRecord]:
    sources: list[SourceRecord] = []
    for supplied in inputs:
        if supplied.is_file():
            sources.append(SourceRecord(_generated_id(str(supplied.resolve()), supplied.stem), supplied, None, {}))
        elif supplied.is_dir():
            for path in sorted(item for item in supplied.rglob("*") if item.suffix.lower() in SUPPORTED_EXTENSIONS):
                relative = path.relative_to(supplied).as_posix()
                identity = f"{supplied.resolve()}::{relative}"
                sources.append(
                    SourceRecord(
                        _generated_id(identity, path.stem),
                        path,
                        None,
                        {"input_root": str(supplied.resolve()), "input_relative_path": relative},
                    )
                )
        else:
            raise FileNotFoundError(f"input does not exist: {supplied}")
    return validate_sources(sources)


def sources_from_manifest(path: Path, source_root: Path | None = None) -> list[SourceRecord]:
    root = (source_root or path.parent).resolve()
    sources: list[SourceRecord] = []
    for line_number, record in enumerate(read_jsonl(path), start=1):
        source_id = record.get("id")
        raw_path = record.get("path")
        checksum = record.get("checksum")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"{path}:{line_number} requires a non-empty string id")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"{path}:{line_number} requires a non-empty string path")
        if checksum is not None and (not isinstance(checksum, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", checksum)):
            raise ValueError(f"{path}:{line_number} has an invalid checksum")
        candidate = Path(raw_path)
        resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        if not candidate.is_absolute():
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number} path escapes the source root") from error
        sources.append(SourceRecord(source_id, resolved, checksum, dict(record)))
    return validate_sources(sources)


def validate_sources(sources: list[SourceRecord]) -> list[SourceRecord]:
    if not sources:
        raise ValueError("no audio sources were found")
    seen: set[str] = set()
    safe_seen: set[str] = set()
    for source in sources:
        if source.id in seen:
            raise ValueError(f"duplicate source id: {source.id}")
        output_id = safe_id(source.id)
        if output_id in safe_seen:
            raise ValueError(f"source IDs collide after filename normalization: {source.id}")
        seen.add(source.id)
        safe_seen.add(output_id)
    return sources


def validate_output_location(inputs: Sequence[Path], output_root: Path) -> None:
    output = output_root.resolve()
    for supplied in inputs:
        if not supplied.is_dir():
            continue
        source_root = supplied.resolve()
        try:
            output.relative_to(source_root)
        except ValueError:
            continue
        raise ValueError(f"output root must not be inside an input directory: {source_root}")


def decode_audio(source: Path, destination: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise AudioDecodeError("ffmpeg is required but was not found on PATH")
    result = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(destination),
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else f"exit {result.returncode}"
        raise AudioDecodeError(f"ffmpeg could not decode {source}: {detail}")


def ffmpeg_version() -> str:
    result = subprocess.run(
        ["ffmpeg", "-version"],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.splitlines()[0].strip()


def energy_boundary(path: Path, start: float, end: float, window_seconds: float) -> tuple[float, float]:
    if end <= start:
        return end, 0.0
    with sf.SoundFile(path) as audio:
        sample_rate = audio.samplerate
        audio.seek(max(0, int(start * sample_rate)))
        samples = audio.read(max(1, int((end - start) * sample_rate)), dtype="float32")
    window = max(1, int(window_seconds * sample_rate))
    if len(samples) < window:
        return (start + end) / 2.0, 0.0
    count = len(samples) - window + 1
    step = max(1, window // 4)
    offsets = list(range(0, count, step))
    if offsets[-1] != count - 1:
        offsets.append(count - 1)
    rms = np.asarray([
        math.sqrt(float(np.mean(np.square(samples[offset:offset + window], dtype=np.float64))))
        for offset in offsets
    ])
    minimum_index = int(np.argmin(rms))
    median = max(float(np.median(rms)), 1e-12)
    minimum = max(float(rms[minimum_index]), 1e-12)
    dip_db = max(0.0, 20.0 * math.log10(median / minimum))
    midpoint = start + (offsets[minimum_index] + window / 2.0) / sample_rate
    return min(end, midpoint), dip_db


def export_clip(
    working_audio: Path,
    output: Path,
    start_sec: float,
    end_sec: float,
    audio_config: dict[str, Any],
) -> None:
    output_format = str(audio_config["format"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.part{output.suffix}")
    if output_format == "MP3":
        duration = end_sec - start_sec
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(working_audio), "-ss", f"{start_sec:.9f}", "-t", f"{duration:.9f}",
                    "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame",
                    "-b:a", f"{audio_config['bitrate_kbps']}k", str(temporary),
                ],
                text=True,
                capture_output=True,
            )
            if result.returncode != 0:
                detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else f"exit {result.returncode}"
                raise RuntimeError(f"ffmpeg MP3 export failed: {detail}")
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        return

    with sf.SoundFile(working_audio) as source:
        source.seek(int(round(start_sec * source.samplerate)))
        samples = source.read(int(round((end_sec - start_sec) * source.samplerate)), dtype="float32")
        sample_rate = source.samplerate
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("clip is empty or contains non-finite samples")
    try:
        sf.write(temporary, samples, sample_rate, format=output_format, subtype="PCM_16")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _load_existing(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() else []


def _verified_reusable(
    record: dict[str, Any],
    source_checksum: str,
    config_digest: str,
    output_root: Path,
    audio_config: dict[str, Any],
) -> bool:
    if record.get("status") != "complete" or record.get("source_checksum") != source_checksum or record.get("config_digest") != config_digest:
        return False
    clips = record.get("clips")
    if not isinstance(clips, list):
        return False
    for clip in clips:
        if not isinstance(clip, dict) or not isinstance(clip.get("path"), str):
            return False
        path = output_root / clip["path"]
        if not path.is_file() or sha256_file(path) != clip.get("checksum"):
            return False
        info = sf.info(path)
        expected = OUTPUT_FORMATS[str(audio_config["format"])]
        if (
            info.samplerate != audio_config["sample_rate"]
            or info.channels != audio_config["channels"]
            or info.format != audio_config["format"]
            or info.subtype != expected["subtype"]
        ):
            return False
    return True


def _settings(config: dict[str, Any]) -> SegmentationSettings:
    values = config["segmentation"]
    return SegmentationSettings(**values)


def _reused_result(
    source: SourceRecord,
    source_checksum: str,
    previous: dict[str, Any],
) -> SourceProcessingResult:
    clips = previous["clips"]
    print(f"[segment] reused id={source.id} clips={len(clips)}", flush=True)
    return SourceProcessingResult(
        source=source,
        source_checksum=source_checksum,
        source_state=previous,
        segments=[],
        vad_records=[],
        rejection=None,
        processed=0,
        reused=1,
        clips_written=len(clips),
        duration_seconds=sum(float(clip["duration_sec"]) for clip in clips),
        operational_failures=0,
    )


def _source_result(
    source: SourceRecord,
    source_checksum: str,
    config_digest: str,
    *,
    reason: str | None,
    detail: str | None,
    decoded_audio: dict[str, Any] | None,
    detector_metadata: dict[str, str] | None,
    segments: list[dict[str, Any]] | None = None,
    vad_records: list[dict[str, Any]] | None = None,
    clips: list[dict[str, Any]] | None = None,
) -> SourceProcessingResult:
    source_segments = segments or []
    source_vad = vad_records or []
    clip_states = clips or []
    status = "complete" if reason is None else "rejected"
    rejection = None
    if reason is not None:
        rejection = {
            "source_id": source.id,
            "source_path": str(source.path),
            "source_checksum": source_checksum,
            "reason": reason,
            "detail": detail,
            "config_digest": config_digest,
        }
        print(f"[segment] rejected id={source.id} reason={reason} detail={detail}", flush=True)
    else:
        print(f"[segment] completed id={source.id} clips={len(clip_states)}", flush=True)

    source_state = {
        "id": source.id,
        "path": str(source.path),
        "source_checksum": source_checksum,
        "expected_checksum": source.expected_checksum,
        "source_metadata": source.metadata,
        "config_digest": config_digest,
        "status": status,
        "clips": clip_states,
        "decoded_audio": decoded_audio,
        "vad_model": detector_metadata,
    }
    return SourceProcessingResult(
        source=source,
        source_checksum=source_checksum,
        source_state=source_state,
        segments=source_segments,
        vad_records=source_vad,
        rejection=rejection,
        processed=int(reason is None),
        reused=0,
        clips_written=len(clip_states) if reason is None else 0,
        duration_seconds=(
            sum(float(clip["duration_sec"]) for clip in clip_states)
            if reason is None
            else 0.0
        ),
        operational_failures=int(reason in OPERATIONAL_REASONS),
    )


def _finish_source(
    source: SourceRecord,
    source_checksum: str,
    working: Path,
    decoded_audio: dict[str, Any],
    intervals: list[SpeechInterval],
    detector_metadata: dict[str, str],
    output_root: Path,
    config: dict[str, Any],
    config_digest: str,
) -> SourceProcessingResult:
    source_segments: list[dict[str, Any]] = []
    source_vad = [
        {"source_id": source.id, "config_digest": config_digest, **asdict(interval)}
        for interval in intervals
    ]
    clip_states: list[dict[str, Any]] = []
    reason: str | None = None
    detail: str | None = None
    try:
        boundaries = construct_segments(
            float(decoded_audio["duration_sec"]),
            intervals,
            _settings(config),
            lambda start, end, window: energy_boundary(working, start, end, window),
        )
        if not boundaries:
            reason, detail = (
                "no_usable_speech",
                "no segment met the configured minimum speech duration",
            )
        for segment_index, boundary in enumerate(boundaries):
            clip_id = f"{safe_id(source.id)}_{segment_index:06d}"
            extension = str(OUTPUT_FORMATS[str(config["audio"]["format"])]["extension"])
            relative = Path("clips") / f"{clip_id}{extension}"
            clip_path = output_root / relative
            try:
                export_clip(
                    working,
                    clip_path,
                    boundary.start_sec,
                    boundary.end_sec,
                    config["audio"],
                )
            except Exception as error:
                raise ClipExportError(f"could not export {clip_id}: {error}") from error
            checksum = sha256_file(clip_path)
            duration = sf.info(clip_path).duration
            clip_state = {
                "path": relative.as_posix(),
                "checksum": checksum,
                "duration_sec": duration,
            }
            clip_states.append(clip_state)
            source_segments.append(
                {
                    "id": clip_id,
                    "source_id": source.id,
                    "source_path": str(source.path),
                    "source_checksum": source_checksum,
                    "path": relative.as_posix(),
                    "clip_checksum": checksum,
                    "start_sec": boundary.start_sec,
                    "end_sec": boundary.end_sec,
                    "duration_sec": duration,
                    "speech_seconds": boundary.speech_seconds,
                    "speech_ratio": boundary.speech_ratio,
                    "boundary_type": boundary.boundary_type,
                    "boundary_silence_sec": boundary.boundary_silence_sec,
                    "energy_dip_db": boundary.energy_dip_db,
                    "config_digest": config_digest,
                }
            )
    except ClipExportError as error:
        reason, detail = "clip_export_failed", str(error)
    except Exception as error:
        reason, detail = "clip_export_failed", f"could not construct segments: {error}"

    return _source_result(
        source,
        source_checksum,
        config_digest,
        reason=reason,
        detail=detail,
        decoded_audio=decoded_audio,
        detector_metadata=detector_metadata,
        segments=source_segments,
        vad_records=source_vad,
        clips=clip_states,
    )


def _process_cohort(
    cohort: Sequence[tuple[int, SourceRecord]],
    total: int,
    output_root: Path,
    config: dict[str, Any],
    config_digest: str,
    previous_sources: dict[str, dict[str, Any]],
    detector_provider: Callable[[], VadDetector],
    ffmpeg_version_provider: Callable[[], str],
) -> list[SourceProcessingResult]:
    results: list[SourceProcessingResult | None] = [None] * len(cohort)
    source_checksums = [""] * len(cohort)
    decoded: dict[int, tuple[Path, dict[str, Any]]] = {}

    with tempfile.TemporaryDirectory(prefix="long-audio-cohort-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        for position, (index, source) in enumerate(cohort):
            print(
                f"[segment] source {index}/{total} id={source.id} path={source.path}",
                flush=True,
            )
            if not source.path.is_file():
                source_checksum = source.expected_checksum or ""
                source_checksums[position] = source_checksum
                results[position] = _source_result(
                    source,
                    source_checksum,
                    config_digest,
                    reason="missing_audio",
                    detail=str(source.path),
                    decoded_audio=None,
                    detector_metadata=None,
                )
                continue

            source_checksum = sha256_file(source.path)
            source_checksums[position] = source_checksum
            if source.expected_checksum and source.expected_checksum != source_checksum:
                results[position] = _source_result(
                    source,
                    source_checksum,
                    config_digest,
                    reason="checksum_mismatch",
                    detail=f"expected {source.expected_checksum}, got {source_checksum}",
                    decoded_audio=None,
                    detector_metadata=None,
                )
                continue

            previous = previous_sources.get(source.id)
            if previous and _verified_reusable(
                previous,
                source_checksum,
                config_digest,
                output_root,
                config["audio"],
            ):
                results[position] = _reused_result(source, source_checksum, previous)
                continue

            working = temporary_root / f"{position:04d}-{safe_id(source.id)}.wav"
            try:
                decode_audio(source.path, working)
                info = sf.info(working)
                if info.frames <= 0 or info.samplerate != 16000 or info.channels != 1:
                    raise AudioDecodeError("decoded audio is empty or not mono 16 kHz")
                decoded[position] = (
                    working,
                    {
                        "sample_rate": info.samplerate,
                        "channels": info.channels,
                        "frames": info.frames,
                        "duration_sec": info.duration,
                        "format": info.format,
                        "subtype": info.subtype,
                        "ffmpeg_version": ffmpeg_version_provider(),
                    },
                )
            except Exception as error:
                detail = (
                    str(error)
                    if isinstance(error, AudioDecodeError)
                    else f"{type(error).__name__}: {error}"
                )
                results[position] = _source_result(
                    source,
                    source_checksum,
                    config_digest,
                    reason="audio_decode_failed",
                    detail=detail,
                    decoded_audio=None,
                    detector_metadata=None,
                )

        if decoded:
            try:
                detector = detector_provider()
            except Exception as error:
                detector = UnavailableVadDetector(error)
            paths = [
                decoded[position][0] if position in decoded else None
                for position in range(len(cohort))
            ]
            detected: dict[int, list[SpeechInterval]] = {}
            batch_error: Exception | None = None
            try:
                detect_many = getattr(detector, "detect_many", None)
                if detect_many is None:
                    batch_values = [
                        detector.detect(path, config["vad"]) if path is not None else []
                        for path in paths
                    ]
                else:
                    batch_values = detect_many(paths, config["vad"])
                if len(batch_values) != len(cohort):
                    raise RuntimeError("batched VAD returned the wrong number of results")
                detected = {
                    position: batch_values[position]
                    for position in decoded
                }
            except Exception as error:
                batch_error = error

            if batch_error is not None:
                for position, (working, decoded_audio) in decoded.items():
                    source = cohort[position][1]
                    try:
                        detected[position] = detector.detect(working, config["vad"])
                    except Exception as error:
                        results[position] = _source_result(
                            source,
                            source_checksums[position],
                            config_digest,
                            reason="vad_failed",
                            detail=f"{type(error).__name__}: {error}",
                            decoded_audio=decoded_audio,
                            detector_metadata=detector.metadata,
                        )

            for position, intervals in detected.items():
                if results[position] is not None:
                    continue
                source = cohort[position][1]
                working, decoded_audio = decoded[position]
                results[position] = _finish_source(
                    source,
                    source_checksums[position],
                    working,
                    decoded_audio,
                    intervals,
                    detector.metadata,
                    output_root,
                    config,
                    config_digest,
                )

    if any(result is None for result in results):
        raise RuntimeError("source cohort did not produce a result for every source")
    return [result for result in results if result is not None]


def process_pipeline(
    sources: list[SourceRecord],
    output_root: Path,
    config: dict[str, Any],
    config_digest: str,
    detector_factory: Callable[[], VadDetector] | None = None,
    *,
    workers: int = 1,
) -> PipelineAudit:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    runtime = execution_settings(config)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "clips").mkdir(exist_ok=True)
    existing_sources = {
        record["id"]: record
        for record in _load_existing(output_root / "sources.jsonl")
    }
    previous_sources = dict(existing_sources)
    segments = _load_existing(output_root / "segments.jsonl")
    vad_records = _load_existing(output_root / "vad_intervals.jsonl")
    rejected = _load_existing(output_root / "rejected.jsonl")
    worker_state = local()
    version_lock = Lock()
    cached_ffmpeg_version: str | None = None

    def ffmpeg_version_provider() -> str:
        nonlocal cached_ffmpeg_version
        if cached_ffmpeg_version is None:
            with version_lock:
                if cached_ffmpeg_version is None:
                    cached_ffmpeg_version = ffmpeg_version()
        return cached_ffmpeg_version

    def detector_provider() -> VadDetector:
        detector = getattr(worker_state, "detector", None)
        if detector is None:
            detector = (
                detector_factory()
                if detector_factory is not None
                else SileroVadDetector(engine=runtime.vad_engine)
            )
            worker_state.detector = detector
        return detector

    def process(cohort: Sequence[tuple[int, SourceRecord]]) -> list[SourceProcessingResult]:
        return _process_cohort(
            cohort,
            len(sources),
            output_root,
            config,
            config_digest,
            previous_sources,
            detector_provider,
            ffmpeg_version_provider,
        )

    indexed_sources = list(enumerate(sources, start=1))
    cohorts = [
        indexed_sources[start:start + runtime.vad_batch_size]
        for start in range(0, len(indexed_sources), runtime.vad_batch_size)
    ]
    original_torch_threads = torch.get_num_threads()
    if runtime.torch_threads is not None:
        torch.set_num_threads(runtime.torch_threads)
    if workers == 1:
        cohort_results: Iterable[list[SourceProcessingResult]] = map(process, cohorts)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="segment-audio")
        cohort_results = executor.map(process, cohorts)
    results = (result for cohort_result in cohort_results for result in cohort_result)

    processed = reused = rejected_count = clips_written = operational = 0
    total_duration = 0.0
    try:
        for result in results:
            processed += result.processed
            reused += result.reused
            clips_written += result.clips_written
            total_duration += result.duration_seconds
            operational += result.operational_failures
            if result.reused:
                continue

            source_id = result.source.id
            rejected = [record for record in rejected if record.get("source_id") != source_id]
            if result.rejection is not None:
                rejected_count += 1
                rejected.append(result.rejection)
            segments = [record for record in segments if record.get("source_id") != source_id] + result.segments
            vad_records = [
                record for record in vad_records
                if record.get("source_id") != source_id
            ] + result.vad_records
            existing_sources[source_id] = result.source_state
            write_jsonl_atomic(
                output_root / "sources.jsonl",
                sorted(existing_sources.values(), key=lambda item: item["id"]),
            )
            write_jsonl_atomic(
                output_root / "segments.jsonl",
                sorted(segments, key=lambda item: item["id"]),
            )
            write_jsonl_atomic(
                output_root / "vad_intervals.jsonl",
                sorted(
                    vad_records,
                    key=lambda item: (item["source_id"], item["start_sec"]),
                ),
            )
            write_jsonl_atomic(
                output_root / "rejected.jsonl",
                sorted(rejected, key=lambda item: item["source_id"]),
            )
    finally:
        if executor is not None:
            executor.shutdown()
        if runtime.torch_threads is not None:
            torch.set_num_threads(original_torch_threads)

    audit = PipelineAudit(
        len(sources),
        processed,
        reused,
        rejected_count,
        clips_written,
        total_duration,
        operational,
    )
    write_json_atomic(output_root / "summary.json", asdict(audit))
    return audit


def run_pipeline(
    sources: list[SourceRecord],
    output_root: Path,
    config: dict[str, Any],
    config_digest: str,
    *,
    force: bool = False,
    detector_factory: Callable[[], VadDetector] | None = None,
    workers: int = 1,
) -> PipelineAudit:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    run_path = output_root / "run.json"
    if output_root.exists() and run_path.is_file():
        prior = json.loads(run_path.read_text(encoding="utf-8"))
        if prior.get("config_digest") != config_digest and not force:
            raise ValueError("output uses a different configuration digest; pass --force to replace it")
    elif output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"output root is not an initialized segmentation run: {output_root}")

    if force and output_root.exists():
        output_root.parent.mkdir(parents=True, exist_ok=True)
        staging_parent = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
        staging = staging_parent / "output"
        try:
            audit = process_pipeline(
                sources,
                staging,
                config,
                config_digest,
                detector_factory,
                workers=workers,
            )
            if audit.operational_failures:
                raise RuntimeError("forced run had operational failures; existing output was preserved")
            write_json_atomic(staging / "run.json", {"config_digest": config_digest})
            (staging / "effective_config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
            shutil.rmtree(output_root)
            staging.replace(output_root)
            return audit
        finally:
            shutil.rmtree(staging_parent, ignore_errors=True)

    output_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(run_path, {"config_digest": config_digest})
    (output_root / "effective_config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return process_pipeline(
        sources,
        output_root,
        config,
        config_digest,
        detector_factory,
        workers=workers,
    )


def print_audit(audit: PipelineAudit, output_root: Path) -> None:
    print("Long-audio segmentation summary")
    print(f"  output root: {output_root}")
    for key, value in asdict(audit).items():
        print(f"  {key.replace('_', ' ')}: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Use Silero VAD to split generic long audio into deterministic non-overlapping audio clips."
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML segmentation configuration file.")
    parser.add_argument("--output-root", required=True, type=Path, help="Directory for clips and audit manifests.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, action="append", help="Audio file or recursively scanned directory; repeat as needed.")
    inputs.add_argument("--manifest", type=Path, help="JSONL manifest containing string id/path and optional sha256 checksum.")
    parser.add_argument("--source-root", type=Path, help="Base for relative manifest paths (default: manifest directory).")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of source files to process concurrently (default: 1).",
    )
    parser.add_argument("--force", action="store_true", help="Stage and replace an existing run, including one with a different config digest.")
    args = parser.parse_args(argv)
    if args.source_root is not None and args.manifest is None:
        parser.error("--source-root requires --manifest")
    try:
        config, digest = load_config(args.config)
        if args.input:
            validate_output_location(args.input, args.output_root)
            sources = discover_inputs(args.input)
        else:
            sources = sources_from_manifest(args.manifest, args.source_root)
        audit = run_pipeline(
            sources,
            args.output_root,
            config,
            digest,
            force=args.force,
            workers=args.workers,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print_audit(audit, args.output_root)
    return 1 if audit.operational_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
