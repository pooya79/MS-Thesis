"""Extract reproducible statistics for an ASR dataset card."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import soundfile as sf

from ml.speech_data.scripts.hf_audio_dataset import (
    AudioDatasetRow,
    load_audio_dataset_rows,
    load_segment_metadata,
    sha256_file,
)


SUMMARY_SCHEMA_VERSION = "hf-audio-dataset-summary-v1"


@dataclass(frozen=True)
class AudioFileInfo:
    relative_path: str
    size_bytes: int
    duration_seconds: float
    sample_rate_hz: int
    channels: int
    format: str
    subtype: str


def inspect_audio_file(row: AudioDatasetRow) -> AudioFileInfo:
    info = sf.info(row.audio_path)
    duration = float(info.duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid duration for {row.audio_path}: {duration}")
    return AudioFileInfo(
        relative_path=row.relative_path,
        size_bytes=row.audio_path.stat().st_size,
        duration_seconds=duration,
        sample_rate_hz=int(info.samplerate),
        channels=int(info.channels),
        format=str(info.format),
        subtype=str(info.subtype),
    )


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _distribution(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "min": ordered[0],
        "p05": _percentile(ordered, 0.05),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _counter_dict(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted((str(key), count) for key, count in Counter(values).items()))


def build_summary(
    *,
    manifest: Path,
    audio_root: Path,
    segments_manifest: Path | None,
    refinement_summary: Path | None,
    workers: int,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    rows = load_audio_dataset_rows(manifest.resolve(), audio_root.resolve())
    segment_metadata = load_segment_metadata(
        segments_manifest.resolve() if segments_manifest is not None else None,
        include_ids={row.example_id for row in rows},
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        audio_info = list(executor.map(inspect_audio_file, rows))

    durations = [info.duration_seconds for info in audio_info]
    sizes = [info.size_bytes for info in audio_info]
    sentences = [row.sentence for row in rows]
    source_ids = {
        str(segment_metadata.get(row.example_id, {}).get("source_id"))
        if segment_metadata.get(row.example_id, {}).get("source_id")
        else row.example_id.rsplit("_", 1)[0]
        for row in rows
    }
    described_segments = sum(row.example_id in segment_metadata for row in rows)
    refinement: dict[str, Any] | None = None
    if refinement_summary is not None:
        if not refinement_summary.is_file():
            raise FileNotFoundError(
                f"refinement summary does not exist: {refinement_summary}"
            )
        refinement = json.loads(refinement_summary.read_text(encoding="utf-8"))

    total_seconds = sum(durations)
    total_bytes = sum(sizes)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "manifest": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
            "audio_root": str(audio_root.resolve()),
            "segments_manifest": (
                str(segments_manifest.resolve()) if segments_manifest is not None else None
            ),
            "segments_manifest_sha256": (
                sha256_file(segments_manifest) if segments_manifest is not None else None
            ),
            "refinement_summary": (
                str(refinement_summary.resolve()) if refinement_summary is not None else None
            ),
            "refinement_summary_sha256": (
                sha256_file(refinement_summary) if refinement_summary is not None else None
            ),
        },
        "examples": len(rows),
        "source_recordings": len(source_ids),
        "segment_metadata_coverage": {
            "described": described_segments,
            "total": len(rows),
        },
        "audio": {
            "total_duration_seconds": total_seconds,
            "total_duration_hours": total_seconds / 3600,
            "total_size_bytes": total_bytes,
            "total_size_gib": total_bytes / 2**30,
            "duration_seconds": _distribution(durations),
            "file_size_bytes": _distribution(sizes),
            "sample_rate_hz": _counter_dict(info.sample_rate_hz for info in audio_info),
            "channels": _counter_dict(info.channels for info in audio_info),
            "formats": _counter_dict(info.format for info in audio_info),
            "subtypes": _counter_dict(info.subtype for info in audio_info),
        },
        "transcriptions": {
            "total_characters": sum(len(sentence) for sentence in sentences),
            "total_whitespace_tokens": sum(len(sentence.split()) for sentence in sentences),
            "unique_sentences": len(set(sentences)),
            "character_count": _distribution([len(sentence) for sentence in sentences]),
            "whitespace_token_count": _distribution(
                [len(sentence.split()) for sentence in sentences]
            ),
        },
        "refinement": refinement,
    }


def render_card_statistics(summary: dict[str, Any]) -> str:
    audio = summary["audio"]
    transcripts = summary["transcriptions"]
    rates = ", ".join(
        f"{rate} Hz ({count:,})" for rate, count in audio["sample_rate_hz"].items()
    )
    channels = ", ".join(
        f"{value} ({count:,})" for value, count in audio["channels"].items()
    )
    lines = [
        "## Dataset statistics",
        "",
        "The statistics below were generated from the published manifest and audio headers.",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Examples | {summary['examples']:,} |",
        f"| Source recordings | {summary['source_recordings']:,} |",
        f"| Total duration | {audio['total_duration_hours']:,.3f} hours |",
        f"| Total FLAC size | {audio['total_size_gib']:,.3f} GiB |",
        f"| Mean clip duration | {audio['duration_seconds']['mean']:,.3f} s |",
        f"| Median clip duration | {audio['duration_seconds']['median']:,.3f} s |",
        f"| 5th–95th percentile duration | {audio['duration_seconds']['p05']:,.3f}–{audio['duration_seconds']['p95']:,.3f} s |",
        f"| Total whitespace-delimited transcript tokens | {transcripts['total_whitespace_tokens']:,} |",
        f"| Unique transcriptions | {transcripts['unique_sentences']:,} |",
        f"| Sample rates | {rates} |",
        f"| Channels | {channels} |",
        "",
        "These counts describe the accepted refined subset only; rejected or uncertain refinements are not included.",
        "",
    ]
    return "\n".join(lines)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect every clip referenced by a path/sentence TSV and write reproducible "
            "JSON statistics plus an optional Markdown snippet for a Hugging Face dataset card."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Input TSV with path and sentence columns.")
    parser.add_argument("--audio-root", required=True, type=Path, help="Directory containing clips directly or in a clips/ child directory.")
    parser.add_argument("--segments-manifest", type=Path, help="Optional segments.jsonl for source counts and provenance coverage.")
    parser.add_argument("--refinement-summary", type=Path, help="Optional refinement_summary.json to preserve acceptance/rejection counts.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON statistics path.")
    parser.add_argument("--card-snippet-output", type=Path, help="Optional output Markdown table for inclusion in README.md.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent audio-header readers (default: 8).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = build_summary(
            manifest=args.manifest,
            audio_root=args.audio_root,
            segments_manifest=args.segments_manifest,
            refinement_summary=args.refinement_summary,
            workers=args.workers,
        )
        _write_text_atomic(
            args.output,
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        if args.card_snippet_output is not None:
            _write_text_atomic(args.card_snippet_output, render_card_statistics(summary))
    except Exception as exc:
        print(f"error: {exc}")
        return 1
    print(
        f"Examples: {summary['examples']:,}\n"
        f"Hours: {summary['audio']['total_duration_hours']:.6f}\n"
        f"Size GiB: {summary['audio']['total_size_gib']:.6f}\n"
        f"Summary: {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
