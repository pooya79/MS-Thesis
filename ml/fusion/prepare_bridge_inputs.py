"""Generate bridge inputs while reporting and skipping inconsistent clip rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml

from ml.fusion.bridge_pretrained import DNSMOS, FRCRN, frcrn_padded_length, sha256
from ml.fusion.bridging_experiment import (
    check_splits,
    filter_split_conflicts,
    skipped_record,
    write_skipped,
)
from ml.fusion.evaluate_ablation import TEST_DATASETS, read_wave, test_rows
from ml.utils.progress import ProgressReporter


def cv25_clip_id(path: str | Path, dataset_root: Path | None = None) -> str:
    """Return a stable CV25 identity that does not depend on the audio suffix."""
    value = Path(path)
    if dataset_root is not None:
        value = value.resolve().relative_to(dataset_root.resolve())
    if value.parts and value.parts[0] == "clips":
        value = Path(*value.parts[1:])
    return value.with_suffix("").as_posix()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temporary.replace(path)


def collect_rows(data_root: Path, scope: str, skipped: list[dict] | None = None) -> list[dict[str, Any]]:
    from ml.enhancement.dataset import read_mapping
    from ml.asr.train_whisper_small import resolve_audio_path
    skipped = skipped if skipped is not None else []
    original = (data_root / "cv-corpus-25.0").resolve()
    speakers = {}
    source_splits: dict[str, set[str]] = {}
    for split in ("train", "dev", "test"):
        with (original / f"{split}.tsv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                path = str(row.get("path", "")).strip()
                if not path:
                    skipped.append(skipped_record({"split": split}, "empty_original_path"))
                    continue
                source = resolve_audio_path(original, path).resolve()
                clip_id = cv25_clip_id(source, original)
                source_splits.setdefault(clip_id, set()).add(split)
                speakers[clip_id] = f"cv25/{row['client_id']}" if row.get("client_id") else None
    rows = []
    # Audit train/dev sources even for a test-only request. Never load clean audio
    # as the enhanced view: clean_path supplies identity/leakage metadata only.
    for split in ("train", "dev"):
        for pair in read_mapping(data_root / "cv-corpus-25.0-degraded-v2", split):
            source = pair.clean_path.resolve()
            if not source.is_relative_to(original):
                skipped.append(skipped_record({"id": pair.pair_id, "split": split},
                                              "non_cv25_source", str(source)))
                continue
            clip_id = cv25_clip_id(source, original)
            if source_splits.get(clip_id) != {split}:
                skipped.append(skipped_record({"id": pair.pair_id, "split": split},
                                              "source_missing_or_wrong_split", str(source)))
                continue
            candidate = {"id": f"cv25-degraded/{pair.pair_id}", "source_id": f"cv25/{clip_id}",
                         "speaker_id": speakers.get(clip_id), "split": split,
                         "sentence": pair.transcript, "noisy_path": str(pair.degraded_path.resolve()),
                         "dataset": "cv-corpus-25.0-degraded-v2", "degradation": pair.degradation}
            if not pair.degraded_path.is_file():
                skipped.append(skipped_record(candidate, "missing_noisy_audio", str(pair.degraded_path)))
                continue
            rows.append(candidate)
    if scope in {"all", "test"}:
        for row in test_rows({"data": {"root_dir": str(data_root), "datasets": list(TEST_DATASETS), "split": "test"}}, skipped):
            if row["dataset"] == "cv-corpus-25.0":
                clip_id = cv25_clip_id(row["relative_path"])
                source_id = f"cv25/{clip_id}"
                speaker_id = speakers.get(clip_id)
            else:
                source_id = f"{row['dataset']}/{Path(row['relative_path']).with_suffix('').as_posix()}"
                speaker_id = None
            rows.append({"id": row["id"], "source_id": source_id,
                         "speaker_id": speaker_id, "split": "test",
                         "sentence": row["reference"], "noisy_path": row["audio_path"],
                         "dataset": row["dataset"], "relative_path": row["relative_path"]})
    id_counts = Counter(row["id"] for row in rows)
    kept = []
    for row in rows:
        if id_counts[row["id"]] > 1:
            skipped.append(skipped_record(row, "duplicate_id"))
        elif not row["sentence"].strip():
            skipped.append(skipped_record(row, "empty_transcript"))
        else:
            kept.append(row)
    rows = filter_split_conflicts(kept, skipped)
    check_splits(rows)
    if scope == "test":
        rows = [r for r in rows if r["split"] == "test"]
    return rows


def select_rows(rows: list[dict], maximum: int | None, seed: int) -> list[dict]:
    key = lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest()
    selected = []
    for split in ("train", "dev", "test"):
        group = sorted([r for r in rows if r["split"] == split], key=key)
        selected.extend(group if maximum is None else group[:maximum])
    return selected


def _cached_or_wave(row: dict, root: Path, identity: dict) -> tuple[dict | None, str, np.ndarray | None]:
    key = hashlib.sha256(row["id"].encode()).hexdigest()
    record = root / "completed" / f"{key}.json"
    source_hash = sha256(Path(row["noisy_path"]))
    if record.is_file():
        saved = json.loads(record.read_text())
        if saved["input_row"] != row or saved["identity"] != identity or saved["source_sha256"] != source_hash:
            raise ValueError(f"cached input changed: {row['id']}; use a new output directory")
        output = Path(saved["result"]["enhanced_path"])
        if not output.is_file() or sha256(output) != saved["enhanced_sha256"]:
            raise ValueError(f"missing/changed cached enhancement: {output}")
        return saved["result"], source_hash, None
    return None, source_hash, read_wave(row["noisy_path"]).numpy()


def _write_result(row: dict, root: Path, wave: np.ndarray, enhanced: np.ndarray,
                  scores: dict[str, float], source_hash: str, identity: dict) -> dict:
    key = hashlib.sha256(row["id"].encode()).hexdigest()
    record = root / "completed" / f"{key}.json"
    if row["split"] == "test":
        relative = Path(row["relative_path"]).with_suffix(".wav")
        target = root / "test-enhanced" / row["dataset"] / relative
    else:
        target = root / "enhanced" / row["split"] / f"{key}.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.wav")
    sf.write(temporary, enhanced, 16000, subtype="FLOAT")
    temporary.replace(target)
    result = {**row, "enhanced_path": str(target.resolve()), "source_sha256": source_hash,
              "sample_rate": 16000, "num_samples": len(wave), **scores}
    atomic_json(record, {"input_row": row, "identity": identity, "source_sha256": source_hash,
                         "enhanced_sha256": sha256(target), "result": result})
    return result


def process_batch(rows: list[dict], root: Path, enhancer: Any, scorer: Any,
                  identity: dict, executor: ThreadPoolExecutor, target_length: int,
                  skipped: list[dict]) -> list[dict]:
    identities = [{**identity, "dnsmos_id": None} if row["split"] == "test" else identity for row in rows]
    loaded = list(executor.map(
        lambda pair: _cached_or_wave(pair[0], root, pair[1]),
        zip(rows, identities, strict=True),
    ))
    pending = [index for index, (cached, _, _) in enumerate(loaded) if cached is None]
    waves = [loaded[index][2] for index in pending]
    enhanced_by_index: dict[int, np.ndarray] = {}

    def keep(index: int, item: Any) -> None:
        value = np.asarray(item, dtype=np.float32)
        wave = loaded[index][2]
        if value.shape != wave.shape or not np.isfinite(value).all():
            raise ValueError("enhancer returned unaligned/nonfinite audio")
        enhanced_by_index[index] = value

    if hasattr(enhancer, "enhance_batch"):
        try:
            enhanced = enhancer.enhance_batch(waves, target_length=target_length)
            if len(enhanced) != len(waves):
                raise ValueError("enhancer returned the wrong batch size")
            for index, item in zip(pending, enhanced, strict=True):
                keep(index, item)
        except ValueError as batch_error:
            print(f"Batch enhancement failed ({batch_error}); retrying clips individually", flush=True)
            enhanced_by_index.clear()
            for index in pending:
                wave = loaded[index][2]
                try:
                    values = enhancer.enhance_batch(
                        [wave], target_length=frcrn_padded_length(len(wave))
                    )
                    if len(values) != 1:
                        raise ValueError("enhancer returned the wrong batch size")
                    keep(index, values[0])
                except ValueError as exc:
                    skipped.append(skipped_record(rows[index], "enhancement_failed", str(exc)))
    else:  # Small test doubles and third-party adapters may expose only scalar inference.
        for index, wave in zip(pending, waves, strict=True):
            try:
                keep(index, enhancer(wave))
            except ValueError as exc:
                skipped.append(skipped_record(rows[index], "enhancement_failed", str(exc)))

    score_indexes = [index for index in enhanced_by_index if rows[index]["split"] != "test"]
    score_values = list(executor.map(
        scorer, (loaded[index][2] for index in score_indexes)
    )) if score_indexes else []
    scores = {index: value for index, value in zip(score_indexes, score_values, strict=True)}
    results = []
    for index, row in enumerate(rows):
        cached, source_hash, wave = loaded[index]
        if cached is not None:
            results.append(cached)
        elif index in enhanced_by_index:
            results.append(_write_result(row, root, wave, enhanced_by_index[index],
                                         scores.get(index, {}), source_hash, identities[index]))
    return results


def _validate_row(row: dict) -> tuple[dict, int | None, str | None]:
    try:
        length = read_wave(row["noisy_path"]).numel()
    except (OSError, RuntimeError, ValueError) as exc:
        return row, None, str(exc)
    return row, length, None


def write_manifests(root: Path, rows: list[dict], identity: dict) -> None:
    for name, splits in (("bridge_train_dev", {"train", "dev"}), ("bridge_dev", {"dev"}), ("bridge_test", {"test"})):
        group = [r for r in rows if r["split"] in splits]
        if not group:
            continue
        path = root / f"{name}.jsonl"
        temporary = path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in group))
        temporary.replace(path)
        atomic_json(path.with_suffix(".provenance.json"), {**identity, "manifest_sha256": sha256(path)})


def run(args: argparse.Namespace) -> None:
    data_root, output = args.data_root.resolve(), args.output.resolve()
    skipped: list[dict] = []
    rows = select_rows(collect_rows(data_root, args.scope, skipped), args.max_per_split, args.seed)
    if not rows:
        raise ValueError("no selected inputs")
    print(f"Collected {len(rows)} inputs; validating with {args.workers} workers", flush=True)
    progress_path = None if args.dry_run else output / "progress.json"
    usable = []
    lengths: dict[str, int] = {}
    with ProgressReporter("bridge-preflight", len(rows), progress_path) as progress:
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="bridge-preflight") as executor:
            for index, (row, length, error) in enumerate(executor.map(_validate_row, rows), start=1):
                if error is not None:
                    skipped.append(skipped_record(row, "unreadable_or_invalid_audio", error))
                else:
                    usable.append(row)
                    lengths[row["id"]] = int(length)
                progress.update(index, row["id"])
    rows = usable
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "dev", "test")}
    print(f"Selected inputs: {counts}", flush=True)
    if skipped:
        reasons = {reason: sum(row["reason"] == reason for row in skipped)
                   for reason in sorted({row["reason"] for row in skipped})}
        print(f"Skipped inputs: {reasons}", flush=True)
    if not rows:
        raise ValueError("no selected inputs")
    required_splits = ({"test"} if args.scope == "test" else
                       {"train", "dev"} if args.scope == "train-dev" else
                       {"train", "dev", "test"})
    available_splits = {str(row["split"]) for row in rows}
    if not required_splits.issubset(available_splits):
        missing = sorted(required_splits - available_splits)
        raise ValueError(f"no usable clips remain in required splits: {missing}")
    if "test" in required_splits:
        missing_datasets = [name for name in TEST_DATASETS
                            if not any(row["split"] == "test" and row["dataset"] == name for row in rows)]
        if missing_datasets:
            raise ValueError(f"no usable test clips remain in required datasets: {missing_datasets}")
    targets = [str(Path(r["dataset"]) / Path(r["relative_path"]).with_suffix(".wav"))
               for r in rows if r["split"] == "test"]
    if len(targets) != len(set(targets)):
        raise ValueError("test filenames collide after replacing suffix with .wav")
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    request = {"data_root": str(data_root), "seed": args.seed, "max_per_split": args.max_per_split,
               "batch_size": args.batch_size, "preparation_version": 2}
    request_path = output / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError("selection or batching changed; use a new output directory (keep pilots separate)")
    atomic_json(request_path, request)
    write_skipped(output / "skipped_inputs.jsonl", skipped)
    with ProgressReporter("bridge-models", 2, output / "progress.json") as progress:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print("Loading/downloading FRCRN...", flush=True)
        enhancer = FRCRN(args.model_root.resolve() / "frcrn", args.device)
        progress.update(1, "FRCRN ready")
        print("Loading/downloading DNSMOS...", flush=True)
        scorer = DNSMOS(args.model_root.resolve() / "dnsmos") if counts["train"] + counts["dev"] else None
        progress.update(2, "DNSMOS ready" if scorer else "DNSMOS not required")
    enhancer_id = f"{enhancer.identity}:batch-pad-v1:size-{args.batch_size}"
    identity = {"enhancer_id": enhancer_id, "dnsmos_id": scorer.identity if scorer else None,
                "preparation_version": 2, "batch_size": args.batch_size}
    order = {row["id"]: index for index, row in enumerate(rows)}
    ordered_rows = sorted(rows, key=lambda row: lengths[row["id"]], reverse=True)
    with ProgressReporter("bridge-inputs", len(rows), output / "progress.json") as progress:
        prepared = []
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="bridge-input") as executor:
            for start in range(0, len(ordered_rows), args.batch_size):
                batch = ordered_rows[start:start + args.batch_size]
                batch_target = max(frcrn_padded_length(lengths[row["id"]]) for row in batch)
                prepared.extend(process_batch(batch, output, enhancer, scorer, identity,
                                              executor, batch_target, skipped))
                for offset, row in enumerate(batch, start=1):
                    progress.update(start + offset, row["id"])
    prepared.sort(key=lambda row: order[row["id"]])
    prepared_splits = {str(row["split"]) for row in prepared}
    if not required_splits.issubset(prepared_splits):
        missing = sorted(required_splits - prepared_splits)
        raise ValueError(f"no successfully enhanced clips remain in required splits: {missing}")
    if "test" in required_splits:
        missing_datasets = [name for name in TEST_DATASETS
                            if not any(row["split"] == "test" and row["dataset"] == name
                                       for row in prepared)]
        if missing_datasets:
            raise ValueError(
                f"no successfully enhanced test clips remain in required datasets: {missing_datasets}"
            )
    counts = {s: sum(r["split"] == s for r in prepared) for s in ("train", "dev", "test")}
    write_skipped(output / "skipped_inputs.jsonl", skipped)
    write_manifests(output, prepared, identity)
    if counts["test"]:
        config = yaml.safe_load(args.test_config.read_text())
        config["data"]["root_dir"] = str(data_root)
        config["bridge_enhanced_root"] = str(output / "test-enhanced")
        config["bridge_enhancer_id"] = enhancer_id
        (output / "final_tests.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    skip_counts = {reason: sum(row["reason"] == reason for row in skipped)
                   for reason in sorted({row["reason"] for row in skipped})}
    atomic_json(output / f"{args.scope}-status.json", {"complete": True, "counts": counts,
                 "skipped": len(skipped), "skip_counts": skip_counts,
                 "pilot": args.max_per_split is not None, "identity": identity})
    print(f"Preparation complete: {output}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="root containing CV25 and AGFarsdat directories")
    parser.add_argument("--output", type=Path, default=Path("artifacts/cv25-tiny/bridge-inputs"), help="resumable output directory")
    parser.add_argument("--model-root", type=Path, default=Path("artifacts/pretrained/bridge"), help="frozen-model identity cache; weights download automatically")
    parser.add_argument("--scope", choices=("all", "train-dev", "test"), default="all", help="which inputs to prepare; test never gets DNSMOS targets")
    parser.add_argument("--device", default="cpu", help="FRCRN torch device, e.g. cuda:0; DNSMOS runs on CPU")
    parser.add_argument("--batch-size", type=int, default=4, help="FRCRN inference batch size; reduce if GPU memory is insufficient")
    parser.add_argument("--workers", type=int, default=4, help="parallel audio I/O, validation, and DNSMOS worker threads")
    parser.add_argument("--max-per-split", type=int, default=None, help="deterministic pilot limit; default all, use a separate pilot output")
    parser.add_argument("--seed", type=int, default=1337, help="deterministic selection/model seed")
    parser.add_argument("--dry-run", action="store_true", help="report usable/skipped counts without downloading or enhancing")
    parser.add_argument("--test-config", type=Path, default=Path("configs/speech_enhancement/cv25_tiny/final_tests.yaml"), help="template for generated final-test config with automatic enhancer identity")
    args = parser.parse_args(argv)
    if args.max_per_split is not None and args.max_per_split < 1:
        parser.error("--max-per-split must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
