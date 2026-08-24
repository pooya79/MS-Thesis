"""Shared validation and metadata helpers for Hugging Face audio publishing."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class AudioDatasetRow:
    """One validated ASR manifest row and its resolved audio file."""

    relative_path: str
    audio_path: Path
    sentence: str
    example_id: str


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _normalise_relative_audio_path(value: str, *, line_number: int) -> str:
    raw = value.strip().replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"manifest line {line_number} has an unsafe relative audio path: {value!r}"
        )
    parts = list(path.parts)
    if parts and parts[0] == "clips":
        parts = parts[1:]
    if not parts or any(part in {"", "."} for part in parts):
        raise ValueError(f"manifest line {line_number} has an invalid audio path: {value!r}")
    return PurePosixPath(*parts).as_posix()


def _resolve_audio(audio_root: Path, relative_path: str) -> Path:
    candidates = (audio_root / relative_path, audio_root / "clips" / relative_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"audio file not found for {relative_path!r}; checked "
        + " and ".join(str(path) for path in candidates)
    )


def load_audio_dataset_rows(manifest: Path, audio_root: Path) -> list[AudioDatasetRow]:
    """Load a path/sentence TSV and resolve every referenced audio file."""

    if not manifest.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest}")
    if not audio_root.is_dir():
        raise NotADirectoryError(f"audio root does not exist or is not a directory: {audio_root}")

    rows: list[AudioDatasetRow] = []
    seen_paths: set[str] = set()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"path", "sentence"}.issubset(reader.fieldnames):
            raise ValueError("manifest must contain path and sentence columns")
        for line_number, source in enumerate(reader, start=2):
            relative_path = _normalise_relative_audio_path(
                source.get("path", ""), line_number=line_number
            )
            if relative_path in seen_paths:
                raise ValueError(
                    f"manifest line {line_number} repeats audio path {relative_path!r}"
                )
            sentence = source.get("sentence", "").strip()
            if not sentence:
                raise ValueError(f"manifest line {line_number} has an empty sentence")
            seen_paths.add(relative_path)
            rows.append(
                AudioDatasetRow(
                    relative_path=relative_path,
                    audio_path=_resolve_audio(audio_root, relative_path),
                    sentence=sentence,
                    example_id=PurePosixPath(relative_path).stem,
                )
            )
    if not rows:
        raise ValueError("manifest contains no data rows")
    return rows


def load_segment_metadata(
    path: Path | None, *, include_ids: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Index optional segmentation JSONL by clip id without retaining unsafe paths."""

    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"segments manifest does not exist: {path}")
    result: dict[str, dict[str, Any]] = {}
    allowed = {
        "source_id",
        "duration_sec",
        "start_sec",
        "end_sec",
        "speech_seconds",
        "speech_ratio",
        "boundary_type",
        "clip_checksum",
    }
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            example_id = record.get("id")
            if not isinstance(example_id, str) or not example_id:
                raise ValueError(f"{path}:{line_number} requires a non-empty string id")
            if include_ids is not None and example_id not in include_ids:
                continue
            if example_id in result:
                raise ValueError(f"{path}:{line_number} repeats id {example_id!r}")
            result[example_id] = {key: record[key] for key in allowed if key in record}
            if include_ids is not None and len(result) == len(include_ids):
                break
    return result


def iter_hf_metadata(
    rows: Iterable[AudioDatasetRow],
    *,
    segment_metadata: dict[str, dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Yield AudioFolder-compatible metadata records."""

    for row in rows:
        record: dict[str, Any] = {
            "file_name": f"audio/{row.relative_path}",
            "id": row.example_id,
            "sentence": row.sentence,
        }
        record.update(segment_metadata.get(row.example_id, {}))
        yield record


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
