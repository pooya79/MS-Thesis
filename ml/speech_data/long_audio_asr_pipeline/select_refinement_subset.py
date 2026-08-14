"""Select a reproducible hour-limited subset for contextual refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ml.speech_data.long_audio_asr_pipeline.segment_audio import (
    read_jsonl,
    sha256_file,
    write_json_atomic,
)
from ml.speech_data.text_normalization import normalize_persian_asr_text


SELECTION_SCHEMA_VERSION = "refinement-source-selection-v1"
PROGRESS_INTERVAL = 1000


@dataclass(frozen=True)
class EligibleSource:
    source_id: str
    duration_seconds: float
    segment_count: int


def _load_eligible_sources(input_root: Path) -> list[EligibleSource]:
    segment_path = input_root / "segments.jsonl"
    transcription_path = input_root / "transcriptions.jsonl"
    if not segment_path.is_file() or not transcription_path.is_file():
        raise ValueError("input root must contain segments.jsonl and transcriptions.jsonl")

    print(f"[select-refinement] loading segments path={segment_path}", flush=True)
    segments = read_jsonl(segment_path)
    print(f"[select-refinement] segments loaded records={len(segments)}", flush=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    segment_ids: set[str] = set()
    for line_number, segment in enumerate(segments, start=1):
        if (
            line_number == 1
            or line_number % PROGRESS_INTERVAL == 0
            or line_number == len(segments)
        ):
            print(
                f"[select-refinement] validating segment {line_number}/{len(segments)}",
                flush=True,
            )
        segment_id = segment.get("id")
        source_id = segment.get("source_id")
        duration = segment.get("duration_sec")
        if not isinstance(segment_id, str) or not segment_id or segment_id in segment_ids:
            raise ValueError(f"segments.jsonl:{line_number} requires a unique non-empty string id")
        segment_ids.add(segment_id)
        if not isinstance(source_id, str) or not source_id:
            continue
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError(f"segments.jsonl:{line_number} requires a positive finite duration_sec")
        grouped.setdefault(source_id, []).append(segment)

    print(
        f"[select-refinement] segment validation complete "
        f"records={len(segments)} source_groups={len(grouped)}",
        flush=True,
    )
    print(f"[select-refinement] loading transcriptions path={transcription_path}", flush=True)
    transcriptions = read_jsonl(transcription_path)
    print(
        f"[select-refinement] transcriptions loaded records={len(transcriptions)}",
        flush=True,
    )
    usable_transcriptions: set[str] = set()
    seen_transcriptions: set[str] = set()
    for line_number, transcript in enumerate(transcriptions, start=1):
        if (
            line_number == 1
            or line_number % PROGRESS_INTERVAL == 0
            or line_number == len(transcriptions)
        ):
            print(
                f"[select-refinement] validating transcription "
                f"{line_number}/{len(transcriptions)}",
                flush=True,
            )
        target_id = transcript.get("id")
        text = transcript.get("normalized_transcript")
        if not isinstance(target_id, str) or not target_id or target_id in seen_transcriptions:
            raise ValueError(
                f"transcriptions.jsonl:{line_number} requires a unique non-empty string id"
            )
        seen_transcriptions.add(target_id)
        if target_id not in segment_ids:
            raise ValueError(f"transcription has no matching segment: {target_id}")
        if not isinstance(text, str) or not normalize_persian_asr_text(text):
            raise ValueError(f"transcription has no usable normalized text: {target_id}")
        usable_transcriptions.add(target_id)

    print(
        f"[select-refinement] transcription validation complete "
        f"usable={len(usable_transcriptions)}",
        flush=True,
    )
    print(
        f"[select-refinement] evaluating source completeness groups={len(grouped)}",
        flush=True,
    )
    eligible: list[EligibleSource] = []
    grouped_items = sorted(grouped.items())
    for group_number, (source_id, source_segments) in enumerate(grouped_items, start=1):
        if (
            group_number == 1
            or group_number % PROGRESS_INTERVAL == 0
            or group_number == len(grouped_items)
        ):
            print(
                f"[select-refinement] evaluating source group "
                f"{group_number}/{len(grouped_items)} eligible={len(eligible)}",
                flush=True,
            )
        if all(
            str(segment["id"]) in usable_transcriptions for segment in source_segments
        ):
            eligible.append(
                EligibleSource(
                    source_id=source_id,
                    duration_seconds=sum(
                        float(segment["duration_sec"]) for segment in source_segments
                    ),
                    segment_count=len(source_segments),
                )
            )
    print(
        f"[select-refinement] eligibility complete total_groups={len(grouped_items)} "
        f"eligible={len(eligible)} incomplete={len(grouped_items) - len(eligible)}",
        flush=True,
    )
    if not eligible:
        raise ValueError("no completely transcribed source groups are eligible for selection")
    return eligible


def select_sources(
    sources: list[EligibleSource], requested_seconds: float, seed: int
) -> list[EligibleSource]:
    if not math.isfinite(requested_seconds) or requested_seconds <= 0:
        raise ValueError("requested duration must be a positive finite number")
    shuffled = sorted(
        sources,
        key=lambda source: (
            hashlib.sha256(f"{seed}\0{source.source_id}".encode()).digest(),
            source.source_id,
        ),
    )

    selected: list[EligibleSource] = []
    selected_seconds = 0.0
    for source in shuffled:
        before = list(selected)
        before_seconds = selected_seconds
        selected.append(source)
        selected_seconds += source.duration_seconds
        if selected_seconds >= requested_seconds:
            if before and abs(before_seconds - requested_seconds) <= abs(
                selected_seconds - requested_seconds
            ):
                return before
            return selected
    return selected


def create_selection_manifest(
    input_root: Path, requested_hours: float, seed: int
) -> dict[str, Any]:
    if not math.isfinite(requested_hours) or requested_hours <= 0:
        raise ValueError("hours must be a positive finite number")
    input_root = input_root.resolve()
    print(
        f"[select-refinement] selection begin input_root={input_root} "
        f"requested_hours={requested_hours:.6f} seed={seed}",
        flush=True,
    )
    selected = select_sources(_load_eligible_sources(input_root), requested_hours * 3600, seed)
    selected_seconds = sum(source.duration_seconds for source in selected)
    print(
        f"[select-refinement] selection complete sources={len(selected)} "
        f"segments={sum(source.segment_count for source in selected)} "
        f"hours={selected_seconds / 3600:.6f}",
        flush=True,
    )
    print("[select-refinement] hashing upstream manifests", flush=True)
    upstream_checksums = {
        "segments": sha256_file(input_root / "segments.jsonl"),
        "transcriptions": sha256_file(input_root / "transcriptions.jsonl"),
    }
    print("[select-refinement] upstream manifest checksums ready", flush=True)
    selection_parameters: dict[str, Any] = {
        "requested_hours": requested_hours,
        "seed": seed,
        "policy": "seeded-random-closest-prefix",
    }
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        **selection_parameters,
        "selected_duration_seconds": selected_seconds,
        "selected_hours": selected_seconds / 3600,
        "upstream_checksums": upstream_checksums,
        "sources": [asdict(source) for source in selected],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly select complete original-audio groups near a requested number of hours "
            "for contextual transcription refinement."
        )
    )
    parser.add_argument(
        "--input-root",
        required=True,
        type=Path,
        help="Segmented root containing segments.jsonl and transcriptions.jsonl.",
    )
    parser.add_argument(
        "--hours",
        required=True,
        type=float,
        help="Positive target duration in hours; whole source groups may make the result differ.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="JSON selection manifest to write atomically.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic random seed (default: 0).",
    )
    args = parser.parse_args(argv)
    try:
        manifest = create_selection_manifest(args.input_root, args.hours, args.seed)
        print(f"[select-refinement] writing selection manifest path={args.output}", flush=True)
        write_json_atomic(args.output, manifest)
        print(f"[select-refinement] selection manifest ready path={args.output}", flush=True)
    except (FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))
    print("Refinement source selection summary")
    print(f"  requested hours: {manifest['requested_hours']:.6f}")
    print(f"  selected hours: {manifest['selected_hours']:.6f}")
    print(f"  selected sources: {len(manifest['sources'])}")
    print(f"  selected segments: {sum(item['segment_count'] for item in manifest['sources'])}")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
