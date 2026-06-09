"""Build long-audio training variants by concatenating short clips per split.

FastConformer (and CTC ASR generally) learns an implicit utterance-length /
emission prior from the length distribution it is trained on. A corpus made
almost entirely of short (2-5 s) clips therefore degrades on longer utterances:
the self-attention softmax is only ever tuned over short key sequences and the
CTC head learns to stop emitting early. When collecting real long audio is not
feasible, the standard remedy is to synthesise long utterances by concatenating
existing short clips.

This script reads an ASR dataset (``train.tsv`` / ``dev.tsv`` / ``eval.tsv`` /
``test.tsv`` with at least ``path`` + ``sentence`` columns, audio under
``clips/``) and writes a NEW dataset directory whose splits contain long
concatenated variants. Concatenation happens **independently within each split**
- a ``test.tsv`` variant is built only from ``test.tsv`` rows, never mixing
splits - so train/dev/test stay disjoint and no leakage is introduced.

Each variant joins 2-N short clips (until a target duration is reached, capped
by ``--max-duration-sec``), loudness-normalises every segment and inserts a
short silence gap between them, then concatenates the transcripts with a space.
By default clips are drawn regardless of speaker (matching real long-form audio,
which changes speakers); pass ``--speaker-column`` to force same-speaker joins.

The output is a self-contained, long-only ASR dataset. Combine it with the
original short dataset at train time (e.g. point the trainer at both, or
oversample this one) rather than treating it as a replacement.

Determinism: every variant is seeded from ``stable_seed(seed, split, ...)`` so
re-running with the same arguments reproduces the same dataset, and full
provenance (source clips, seed, durations) is recorded in a JSONL manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from ml.utils.audio import load_audio, resample_audio, save_audio
from ml.utils.seed import stable_seed


csv.field_size_limit(sys.maxsize)

DEFAULT_SPLITS = ("train.tsv", "dev.tsv", "eval.tsv", "test.tsv")
DEFAULT_SPLIT_STEMS = tuple(Path(split).stem for split in DEFAULT_SPLITS)

FloatArray = np.ndarray


@dataclass
class SplitAudit:
    input_rows: int = 0
    variants_written: int = 0
    variants_skipped: int = 0
    mean_duration_sec: float = 0.0
    min_duration_sec: float = 0.0
    max_duration_sec: float = 0.0
    durations: list[float] = field(default_factory=list)


def split_name(value: str) -> str:
    return value if value.endswith(".tsv") else f"{value}.tsv"


def validate_output_root(source_root: Path, output_root: Path) -> None:
    source_resolved = source_root.resolve()
    output_resolved = output_root.resolve()
    if source_resolved == output_resolved:
        raise ValueError("output root must be different from source root")
    try:
        output_resolved.relative_to(source_resolved)
    except ValueError:
        return
    raise ValueError("output root must not be inside source root")


def resolve_audio_path(dataset_dir: Path, value: str) -> Path:
    """Resolve a TSV ``path`` value: ``clips/<path>`` first, then ``<path>``."""
    raw_path = Path(value)
    if raw_path.is_absolute():
        return raw_path
    candidates = [dataset_dir / "clips" / raw_path, dataset_dir / raw_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def read_split_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"path", "sentence"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain path and sentence columns")
        return [dict(row) for row in reader]


def write_tsv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sentence"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def rms_normalize(audio: FloatArray, target_rms: float, max_gain: float = 20.0) -> FloatArray:
    """Scale ``audio`` toward ``target_rms`` so segment loudness matches at joins."""
    arr = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0
    if rms <= 1e-8:
        return arr
    gain = min(target_rms / rms, max_gain)
    return (arr * gain).astype(np.float32)


def peak_normalize(audio: FloatArray, peak: float = 0.99) -> FloatArray:
    """Scale down so the absolute peak never clips after concatenation."""
    arr = np.asarray(audio, dtype=np.float32)
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    if max_abs > peak:
        arr = (arr * (peak / max_abs)).astype(np.float32)
    return arr


def _load_resampled(dataset_dir: Path, value: str, target_rate: int) -> FloatArray:
    audio, source_rate = load_audio(resolve_audio_path(dataset_dir, value))
    return resample_audio(audio, source_rate, target_rate)


@dataclass(frozen=True)
class VariantPlan:
    sources: list[dict[str, str]]
    audio: FloatArray
    duration_sec: float


def build_variant(
    rng: np.random.RandomState,
    rows: list[dict[str, str]],
    candidate_indices: list[int],
    dataset_dir: Path,
    *,
    sample_rate: int,
    min_clips: int,
    max_clips: int,
    target_min_sec: float,
    max_duration_sec: float,
    gap_sec: float,
    target_rms: float,
) -> VariantPlan | None:
    """Sample short clips from ``candidate_indices`` and concatenate them.

    Keeps adding randomly-drawn clips until the running duration reaches
    ``target_min_sec`` (or ``max_clips`` is hit), never exceeding
    ``max_duration_sec``. Returns ``None`` if a usable variant (>= ``min_clips``
    segments) could not be assembled.
    """
    if len(candidate_indices) < min_clips:
        return None

    gap = np.zeros(int(round(gap_sec * sample_rate)), dtype=np.float32)
    selected_rows: list[dict[str, str]] = []
    parts: list[FloatArray] = []
    total_sec = 0.0
    guard = 0
    guard_limit = max(max_clips * 8, 16)

    while guard < guard_limit:
        guard += 1
        if len(selected_rows) >= max_clips:
            break
        if len(selected_rows) >= min_clips and total_sec >= target_min_sec:
            break

        idx = candidate_indices[rng.randint(0, len(candidate_indices))]
        row = rows[idx]
        try:
            audio = _load_resampled(dataset_dir, row["path"], sample_rate)
        except Exception:
            # Missing/unreadable clip: draw a different one.
            continue
        if audio.size == 0:
            continue
        clip_sec = len(audio) / sample_rate
        added_sec = clip_sec + (gap_sec if selected_rows else 0.0)

        if selected_rows and total_sec + added_sec > max_duration_sec:
            # Adding this clip overflows the cap; stop if we already have enough.
            if len(selected_rows) >= min_clips:
                break
            continue  # try to find a shorter clip to reach min_clips

        if selected_rows:
            parts.append(gap)
        parts.append(rms_normalize(audio, target_rms))
        selected_rows.append(row)
        total_sec += added_sec

    if len(selected_rows) < min_clips:
        return None

    merged = peak_normalize(np.concatenate(parts).astype(np.float32))
    return VariantPlan(sources=selected_rows, audio=merged, duration_sec=len(merged) / sample_rate)


def _candidate_groups(
    rows: list[dict[str, str]], speaker_column: str | None, min_clips: int
) -> list[list[int]]:
    """All-rows as one pool, or one pool per speaker when grouping by speaker."""
    if speaker_column is None:
        return [list(range(len(rows)))]
    if rows and speaker_column not in rows[0]:
        raise ValueError(f"speaker column '{speaker_column}' not found in split rows")
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(row.get(speaker_column, ""), []).append(i)
    return [idxs for idxs in groups.values() if len(idxs) >= min_clips]


def joined_sentence(rows: list[dict[str, str]]) -> str:
    return " ".join(" ".join(row.get("sentence", "").split()) for row in rows).strip()


def generate_split(
    source_root: Path,
    output_root: Path,
    split: str,
    *,
    seed: int,
    variants: int,
    sample_rate: int,
    min_clips: int,
    max_clips: int,
    target_min_sec: float,
    max_duration_sec: float,
    gap_sec: float,
    target_rms: float,
    speaker_column: str | None,
) -> tuple[list[dict[str, str]], list[dict[str, object]], SplitAudit]:
    stem = Path(split).stem
    rows = read_split_rows(source_root / split)
    audit = SplitAudit(input_rows=len(rows))
    groups = _candidate_groups(rows, speaker_column, min_clips)

    tsv_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, object]] = []
    output_clips = output_root / "clips"

    for variant_index in range(variants):
        variant_seed = stable_seed(seed, stem, "concat", variant_index)
        rng = np.random.RandomState(variant_seed)
        if not groups:
            audit.variants_skipped += 1
            continue
        group = groups[rng.randint(0, len(groups))]
        plan = build_variant(
            rng,
            rows,
            group,
            source_root,
            sample_rate=sample_rate,
            min_clips=min_clips,
            max_clips=max_clips,
            target_min_sec=target_min_sec,
            max_duration_sec=max_duration_sec,
            gap_sec=gap_sec,
            target_rms=target_rms,
        )
        if plan is None:
            audit.variants_skipped += 1
            continue

        rel_path = f"long_{stem}_{variant_index:06d}.wav"
        save_audio(output_clips / rel_path, plan.audio, sample_rate)
        sentence = joined_sentence(plan.sources)
        tsv_rows.append({"path": rel_path, "sentence": sentence})
        manifest_rows.append(
            {
                "split": stem,
                "variant_index": variant_index,
                "output_path": rel_path,
                "sentence": sentence,
                "num_source_clips": len(plan.sources),
                "source_paths": [row["path"] for row in plan.sources],
                "duration_sec": round(plan.duration_sec, 4),
                "sample_rate": sample_rate,
                "gap_sec": gap_sec,
                "speaker_column": speaker_column,
                "seed": variant_seed,
            }
        )
        audit.durations.append(plan.duration_sec)
        audit.variants_written += 1

    if audit.durations:
        audit.mean_duration_sec = float(np.mean(audit.durations))
        audit.min_duration_sec = float(np.min(audit.durations))
        audit.max_duration_sec = float(np.max(audit.durations))
    return tsv_rows, manifest_rows, audit


def concatenate_long_variants(
    source_root: Path,
    output_root: Path,
    *,
    splits: Iterable[str] | None = None,
    seed: int = 1234,
    variants_per_split: int = 2000,
    sample_rate: int = 16000,
    min_clips: int = 2,
    max_clips: int = 4,
    target_min_sec: float = 5.0,
    max_duration_sec: float = 20.0,
    gap_sec: float = 0.2,
    target_rms: float = 0.05,
    speaker_column: str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist or is not a directory: {source_root}")
    validate_output_root(source_root, output_root)
    if min_clips < 2:
        raise ValueError("min_clips must be >= 2 (a concatenation needs at least two clips)")
    if max_clips < min_clips:
        raise ValueError("max_clips must be >= min_clips")
    if target_min_sec > max_duration_sec:
        raise ValueError("target_min_sec must be <= max_duration_sec")

    if splits is None:
        resolved_splits = [s for s in DEFAULT_SPLITS if (source_root / s).exists()]
        if not resolved_splits:
            raise FileNotFoundError(
                f"missing split TSV: expected one of {', '.join(DEFAULT_SPLITS)} under {source_root}"
            )
    else:
        resolved_splits = [split_name(s) for s in splits]
        for s in resolved_splits:
            if not (source_root / s).exists():
                raise FileNotFoundError(f"missing split TSV: {source_root / s}")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    all_manifest: list[dict[str, object]] = []
    audits: dict[str, SplitAudit] = {}
    for split in resolved_splits:
        tsv_rows, manifest_rows, audit = generate_split(
            source_root,
            output_root,
            split,
            seed=seed,
            variants=variants_per_split,
            sample_rate=sample_rate,
            min_clips=min_clips,
            max_clips=max_clips,
            target_min_sec=target_min_sec,
            max_duration_sec=max_duration_sec,
            gap_sec=gap_sec,
            target_rms=target_rms,
            speaker_column=speaker_column,
        )
        write_tsv(output_root / split, tsv_rows)
        all_manifest.extend(manifest_rows)
        audits[split] = audit

    write_jsonl(output_root / "long_variants_manifest.jsonl", all_manifest)
    report = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "params": {
            "seed": seed,
            "variants_per_split": variants_per_split,
            "sample_rate": sample_rate,
            "min_clips": min_clips,
            "max_clips": max_clips,
            "target_min_sec": target_min_sec,
            "max_duration_sec": max_duration_sec,
            "gap_sec": gap_sec,
            "target_rms": target_rms,
            "speaker_column": speaker_column,
        },
        "splits": {
            split: {
                "input_rows": audit.input_rows,
                "variants_written": audit.variants_written,
                "variants_skipped": audit.variants_skipped,
                "mean_duration_sec": round(audit.mean_duration_sec, 4),
                "min_duration_sec": round(audit.min_duration_sec, 4),
                "max_duration_sec": round(audit.max_duration_sec, 4),
            }
            for split, audit in audits.items()
        },
    }
    (output_root / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def print_report(report: dict[str, object]) -> None:
    print("Long-variant concatenation summary")
    print(f"  source root: {report['source_root']}")
    print(f"  output root: {report['output_root']}")
    for split, stats in report["splits"].items():  # type: ignore[union-attr]
        print(f"  {split}:")
        print(f"    input rows: {stats['input_rows']}")
        print(f"    variants written: {stats['variants_written']}")
        print(f"    variants skipped: {stats['variants_skipped']}")
        print(
            "    duration sec (min/mean/max): "
            f"{stats['min_duration_sec']}/{stats['mean_duration_sec']}/{stats['max_duration_sec']}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new ASR dataset of long audio variants by concatenating short clips "
            "within each split (train.tsv, dev.tsv, eval.tsv, test.tsv). Concatenation never "
            "crosses split boundaries, so no train/dev/test leakage is introduced. Segments are "
            "loudness-normalized and separated by a short silence gap; transcripts are joined."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Input ASR dataset directory with split TSVs (path + sentence columns) and a clips/ dir.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New output dataset directory for the generated long variants (must differ from source).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=DEFAULT_SPLITS + DEFAULT_SPLIT_STEMS,
        help=(
            "Split TSV filenames to process. Defaults to whichever of "
            "train.tsv, dev.tsv, eval.tsv, test.tsv exist in the source."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234, help="Global seed for deterministic variant generation (default: 1234).")
    parser.add_argument(
        "--variants-per-split",
        type=int,
        default=2000,
        help="Number of long variants to generate for each processed split (default: 2000).",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output sample rate in Hz; inputs are resampled to match (default: 16000).")
    parser.add_argument("--min-clips", type=int, default=2, help="Minimum short clips per variant; must be >= 2 (default: 2).")
    parser.add_argument("--max-clips", type=int, default=4, help="Maximum short clips per variant (default: 4).")
    parser.add_argument(
        "--target-min-sec",
        type=float,
        default=5.0,
        help="Keep adding clips until the variant reaches at least this many seconds (default: 5.0).",
    )
    parser.add_argument(
        "--max-duration-sec",
        type=float,
        default=20.0,
        help="Hard upper cap on a variant's duration in seconds (default: 20.0).",
    )
    parser.add_argument("--gap-sec", type=float, default=0.2, help="Silence inserted between concatenated clips, in seconds (default: 0.2).")
    parser.add_argument("--target-rms", type=float, default=0.05, help="Per-segment RMS loudness target before concatenation (default: 0.05).")
    parser.add_argument(
        "--speaker-column",
        type=str,
        default=None,
        help=(
            "Optional TSV column (e.g. client_id) to restrict each variant to a single speaker. "
            "Omit to concatenate across speakers, which better matches real long-form audio."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    args = parser.parse_args(argv)

    report = concatenate_long_variants(
        args.source_root,
        args.output_root,
        splits=args.splits,
        seed=args.seed,
        variants_per_split=args.variants_per_split,
        sample_rate=args.sample_rate,
        min_clips=args.min_clips,
        max_clips=args.max_clips,
        target_min_sec=args.target_min_sec,
        max_duration_sec=args.max_duration_sec,
        gap_sec=args.gap_sec,
        target_rms=args.target_rms,
        speaker_column=args.speaker_column,
        overwrite=args.overwrite,
    )
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
