"""Generate FRCRN waveforms, DNSMOS supervision and manifests from existing CV25 data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml

from ml.fusion.bridge_pretrained import DNSMOS, FRCRN, sha256
from ml.fusion.bridging_experiment import check_splits
from ml.fusion.evaluate_ablation import TEST_DATASETS, read_wave, test_rows


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temporary.replace(path)


def collect_rows(data_root: Path, scope: str) -> list[dict[str, Any]]:
    from ml.enhancement.dataset import read_mapping
    from ml.asr.train_whisper_small import resolve_audio_path
    original = (data_root / "cv-corpus-25.0").resolve()
    speakers = {}
    source_splits = {}
    for split in ("train", "dev", "test"):
        with (original / f"{split}.tsv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                source = str(resolve_audio_path(original, row["path"]).resolve())
                if source in source_splits and source_splits[source] != split:
                    raise ValueError(f"original CV25 source crosses splits: {source}")
                source_splits[source] = split
                speakers[source] = f"cv25/{row['client_id']}" if row.get("client_id") else None
    rows = []
    # Audit train/dev sources even for a test-only request. Never load clean audio
    # as the enhanced view: clean_path supplies identity/leakage metadata only.
    for split in ("train", "dev"):
        for pair in read_mapping(data_root / "cv-corpus-25.0-degraded-v2", split):
            source = pair.clean_path.resolve()
            if not source.is_relative_to(original):
                raise ValueError(f"non-CV25 source in degraded mapping: {source}")
            if source_splits.get(str(source)) != split:
                raise ValueError(f"degraded source is missing or in a different original CV25 split: {source}")
            rows.append({"id": f"cv25-degraded/{pair.pair_id}", "source_id": str(source),
                         "speaker_id": speakers.get(str(source)), "split": split,
                         "sentence": pair.transcript, "noisy_path": str(pair.degraded_path.resolve()),
                         "dataset": "cv-corpus-25.0-degraded-v2", "degradation": pair.degradation})
    if scope in {"all", "test"}:
        for row in test_rows({"data": {"root_dir": str(data_root), "datasets": list(TEST_DATASETS), "split": "test"}}):
            rows.append({"id": row["id"], "source_id": row["audio_path"],
                         "speaker_id": speakers.get(row["audio_path"]), "split": "test",
                         "sentence": row["reference"], "noisy_path": row["audio_path"],
                         "dataset": row["dataset"], "relative_path": row["relative_path"]})
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate input IDs")
    if any(not row["sentence"].strip() for row in rows):
        raise ValueError("empty transcript in input metadata")
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


def process_row(row: dict, root: Path, enhancer: Any, scorer: Any, identity: dict) -> dict:
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
        return saved["result"]
    wave = read_wave(row["noisy_path"]).numpy()
    enhanced = np.asarray(enhancer(wave), dtype=np.float32)
    if enhanced.shape != wave.shape or not np.isfinite(enhanced).all():
        raise ValueError(f"unaligned/nonfinite enhancement: {row['id']}")
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
              "sample_rate": 16000, "num_samples": len(wave)}
    if row["split"] != "test":
        result.update(scorer(wave))  # noisy signal, never clean reference or test scores
    atomic_json(record, {"input_row": row, "identity": identity, "source_sha256": source_hash,
                         "enhanced_sha256": sha256(target), "result": result})
    return result


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
    rows = select_rows(collect_rows(data_root, args.scope), args.max_per_split, args.seed)
    if not rows:
        raise ValueError("no selected inputs")
    targets = [str(Path(r["dataset"]) / Path(r["relative_path"]).with_suffix(".wav"))
               for r in rows if r["split"] == "test"]
    if len(targets) != len(set(targets)):
        raise ValueError("test filenames collide after replacing suffix with .wav")
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "dev", "test")}
    print(f"Selected inputs: {counts}", flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    request = {"data_root": str(data_root), "seed": args.seed, "max_per_split": args.max_per_split}
    request_path = output / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError("selection changed; use a new output directory (keep pilots separate)")
    atomic_json(request_path, request)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    enhancer = FRCRN(args.model_root.resolve() / "frcrn", args.device)
    scorer = DNSMOS(args.model_root.resolve() / "dnsmos") if counts["train"] + counts["dev"] else None
    identity = {"enhancer_id": enhancer.identity, "dnsmos_id": scorer.identity if scorer else None,
                "preparation_version": 1}
    prepared = []
    for i, row in enumerate(rows, start=1):
        # Test records do not depend on the DNSMOS model, so train-dev/test/all
        # commands can reuse the same completed test files.
        row_identity = {**identity, "dnsmos_id": None} if row["split"] == "test" else identity
        prepared.append(process_row(row, output, enhancer, scorer, row_identity))
        print(f"[{i}/{len(rows)}] ready: {row['id']}", flush=True)
    write_manifests(output, prepared, identity)
    if counts["test"]:
        config = yaml.safe_load(args.test_config.read_text())
        config["data"]["root_dir"] = str(data_root)
        config["bridge_enhanced_root"] = str(output / "test-enhanced")
        config["bridge_enhancer_id"] = enhancer.identity
        (output / "final_tests.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    atomic_json(output / f"{args.scope}-status.json", {"complete": True, "counts": counts,
                 "pilot": args.max_per_split is not None, "identity": identity})
    print(f"Preparation complete: {output}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="root containing CV25 and AGFarsdat directories")
    parser.add_argument("--output", type=Path, default=Path("artifacts/cv25-tiny/bridge-inputs"), help="resumable output directory")
    parser.add_argument("--model-root", type=Path, default=Path("artifacts/pretrained/bridge"), help="frozen-model identity cache; weights download automatically")
    parser.add_argument("--scope", choices=("all", "train-dev", "test"), default="all", help="which inputs to prepare; test never gets DNSMOS targets")
    parser.add_argument("--device", default="cpu", help="FRCRN torch device, e.g. cuda:0; DNSMOS runs on CPU")
    parser.add_argument("--max-per-split", type=int, default=None, help="deterministic pilot limit; default all, use a separate pilot output")
    parser.add_argument("--seed", type=int, default=1337, help="deterministic selection/model seed")
    parser.add_argument("--dry-run", action="store_true", help="validate metadata/split identities and report counts without downloading or enhancing")
    parser.add_argument("--test-config", type=Path, default=Path("configs/speech_enhancement/cv25_tiny/final_tests.yaml"), help="template for generated final-test config with automatic enhancer identity")
    args = parser.parse_args(argv)
    if args.max_per_split is not None and args.max_per_split < 1:
        parser.error("--max-per-split must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
