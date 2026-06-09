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
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from tqdm import tqdm

from ml.utils.audio import load_audio, resample_audio, save_audio
from ml.utils.seed import stable_seed


csv.field_size_limit(sys.maxsize)

DEFAULT_SPLITS = ("train.tsv", "dev.tsv", "eval.tsv", "test.tsv")

# All generation parameters live in the YAML config; CLI only points at it.
# ``variants_per_split`` is a mapping of split name -> count, so each split gets
# its own number of variants and only the listed splits are processed.
DEFAULT_CONFIG: dict[str, Any] = {
    "source_root": None,
    "output_root": None,
    "seed": 1234,
    "sample_rate": 16000,
    "min_clips": 2,
    "max_clips": 4,
    "target_min_sec": 5.0,
    "max_duration_sec": 20.0,
    "gap_sec": 0.2,
    "target_rms": 0.05,
    "speaker_column": None,
    "overwrite": False,
    "workers": 1,
    "variants_per_split": {},
}

FloatArray = np.ndarray


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def load_config(path: Path) -> dict[str, Any]:
    return deep_merge(DEFAULT_CONFIG, load_yaml(path))


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


@dataclass(frozen=True)
class _VariantContext:
    """Read-only state needed to build one variant; shared with worker processes."""

    source_root: Path
    output_clips: Path
    rows: list[dict[str, str]]
    groups: list[list[int]]
    seed: int
    stem: str
    speaker_column: str | None
    sample_rate: int
    min_clips: int
    max_clips: int
    target_min_sec: float
    max_duration_sec: float
    gap_sec: float
    target_rms: float


def build_variant_record(ctx: _VariantContext, variant_index: int) -> dict[str, object] | None:
    """Build, save, and describe one variant. Returns ``None`` if it was skipped.

    Deterministic in ``variant_index`` alone (via ``stable_seed``), so variants
    can be generated in any order or in parallel without changing the output.
    """
    variant_seed = stable_seed(ctx.seed, ctx.stem, "concat", variant_index)
    rng = np.random.RandomState(variant_seed)
    if not ctx.groups:
        return None
    group = ctx.groups[rng.randint(0, len(ctx.groups))]
    plan = build_variant(
        rng,
        ctx.rows,
        group,
        ctx.source_root,
        sample_rate=ctx.sample_rate,
        min_clips=ctx.min_clips,
        max_clips=ctx.max_clips,
        target_min_sec=ctx.target_min_sec,
        max_duration_sec=ctx.max_duration_sec,
        gap_sec=ctx.gap_sec,
        target_rms=ctx.target_rms,
    )
    if plan is None:
        return None

    rel_path = f"long_{ctx.stem}_{variant_index:06d}.wav"
    save_audio(ctx.output_clips / rel_path, plan.audio, ctx.sample_rate)
    return {
        "split": ctx.stem,
        "variant_index": variant_index,
        "output_path": rel_path,
        "sentence": joined_sentence(plan.sources),
        "num_source_clips": len(plan.sources),
        "source_paths": [row["path"] for row in plan.sources],
        "duration_sec": round(plan.duration_sec, 4),
        "sample_rate": ctx.sample_rate,
        "gap_sec": ctx.gap_sec,
        "speaker_column": ctx.speaker_column,
        "seed": variant_seed,
    }


# Per-worker handle to the (large, read-only) variant context, set once by the
# pool initializer so the rows list is pickled once per worker, not per task.
_WORKER_CTX: _VariantContext | None = None


def _init_worker(ctx: _VariantContext) -> None:
    global _WORKER_CTX
    _WORKER_CTX = ctx


def _worker_build(variant_index: int) -> dict[str, object] | None:
    assert _WORKER_CTX is not None, "worker context not initialized"
    return build_variant_record(_WORKER_CTX, variant_index)


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
    workers: int = 1,
) -> tuple[list[dict[str, str]], list[dict[str, object]], SplitAudit]:
    stem = Path(split).stem
    rows = read_split_rows(source_root / split)
    audit = SplitAudit(input_rows=len(rows))
    groups = _candidate_groups(rows, speaker_column, min_clips)
    output_clips = output_root / "clips"
    output_clips.mkdir(parents=True, exist_ok=True)

    ctx = _VariantContext(
        source_root=source_root,
        output_clips=output_clips,
        rows=rows,
        groups=groups,
        seed=seed,
        stem=stem,
        speaker_column=speaker_column,
        sample_rate=sample_rate,
        min_clips=min_clips,
        max_clips=max_clips,
        target_min_sec=target_min_sec,
        max_duration_sec=max_duration_sec,
        gap_sec=gap_sec,
        target_rms=target_rms,
    )

    desc = f"{split} variants"
    # executor.map preserves input order, so results align with variant_index
    # regardless of worker scheduling -> output stays deterministic.
    records: list[dict[str, object] | None]
    if workers <= 1:
        records = [
            build_variant_record(ctx, i)
            for i in tqdm(range(variants), desc=desc, unit="clip")
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_init_worker,
            initargs=(ctx,),
        ) as executor:
            records = list(
                tqdm(
                    executor.map(_worker_build, range(variants)),
                    total=variants,
                    desc=desc,
                    unit="clip",
                )
            )

    tsv_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, object]] = []
    for record in records:
        if record is None:
            audit.variants_skipped += 1
            continue
        tsv_rows.append({"path": str(record["output_path"]), "sentence": str(record["sentence"])})
        manifest_rows.append(record)
        audit.durations.append(float(record["duration_sec"]))
        audit.variants_written += 1

    if audit.durations:
        audit.mean_duration_sec = float(np.mean(audit.durations))
        audit.min_duration_sec = float(np.min(audit.durations))
        audit.max_duration_sec = float(np.max(audit.durations))
    return tsv_rows, manifest_rows, audit


def _resolve_variants_per_split(value: Any) -> dict[str, int]:
    """Normalize the ``variants_per_split`` config into ``{split.tsv: count}``."""
    if not isinstance(value, dict) or not value:
        raise ValueError(
            "config 'variants_per_split' must be a non-empty mapping of split name to count, "
            "e.g. {train.tsv: 3000, dev.tsv: 300, test.tsv: 300}"
        )
    resolved: dict[str, int] = {}
    for split, count in value.items():
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"variants_per_split['{split}'] must be a non-negative integer, got {count!r}")
        resolved[split_name(str(split))] = count
    return resolved


def concatenate_long_variants(config: dict[str, Any]) -> dict[str, object]:
    """Generate long concatenated variants for an ASR dataset from a merged config dict.

    ``config['variants_per_split']`` is a mapping of split name to the number of
    variants to generate for that split; only those splits are processed, each
    with its own count.
    """
    merged = deep_merge(DEFAULT_CONFIG, config)
    if not merged.get("source_root"):
        raise ValueError("config 'source_root' is required")
    if not merged.get("output_root"):
        raise ValueError("config 'output_root' is required")

    source_root = Path(str(merged["source_root"]))
    output_root = Path(str(merged["output_root"]))
    seed = int(merged["seed"])
    sample_rate = int(merged["sample_rate"])
    min_clips = int(merged["min_clips"])
    max_clips = int(merged["max_clips"])
    target_min_sec = float(merged["target_min_sec"])
    max_duration_sec = float(merged["max_duration_sec"])
    gap_sec = float(merged["gap_sec"])
    target_rms = float(merged["target_rms"])
    speaker_column = merged["speaker_column"]
    overwrite = bool(merged["overwrite"])
    workers = int(merged["workers"])
    variants_per_split = _resolve_variants_per_split(merged["variants_per_split"])

    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist or is not a directory: {source_root}")
    validate_output_root(source_root, output_root)
    if min_clips < 2:
        raise ValueError("min_clips must be >= 2 (a concatenation needs at least two clips)")
    if max_clips < min_clips:
        raise ValueError("max_clips must be >= min_clips")
    if target_min_sec > max_duration_sec:
        raise ValueError("target_min_sec must be <= max_duration_sec")
    if workers < 1:
        raise ValueError("workers must be >= 1")

    for split in variants_per_split:
        if not (source_root / split).exists():
            raise FileNotFoundError(f"missing split TSV: {source_root / split}")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    all_manifest: list[dict[str, object]] = []
    audits: dict[str, SplitAudit] = {}
    for split, variants in variants_per_split.items():
        tsv_rows, manifest_rows, audit = generate_split(
            source_root,
            output_root,
            split,
            seed=seed,
            variants=variants,
            sample_rate=sample_rate,
            min_clips=min_clips,
            max_clips=max_clips,
            target_min_sec=target_min_sec,
            max_duration_sec=max_duration_sec,
            gap_sec=gap_sec,
            target_rms=target_rms,
            speaker_column=speaker_column,
            workers=workers,
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
            "workers": workers,
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
            "within each split. All generation parameters come from a YAML config; "
            "'variants_per_split' is a mapping of split name to count, so each split gets its own "
            "number of variants and only the listed splits are processed. Concatenation never "
            "crosses split boundaries, so no train/dev/test leakage is introduced. Segments are "
            "loudness-normalized and separated by a short silence gap; transcripts are joined."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the long-variant concatenation YAML config (see configs/long_variants.yaml).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Override the config and allow writing into an existing output directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes. Overrides the config 'workers' value when provided.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.overwrite:
        config["overwrite"] = True
    if args.workers is not None:
        config["workers"] = args.workers
    report = concatenate_long_variants(config)
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
