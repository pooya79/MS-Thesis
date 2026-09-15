"""Run the waveform bridge baseline, reporting and skipping invalid clip rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch

from ml.fusion.bridging import BridgingModule, OA_COEFFICIENTS, bridge_loss, observation_addition
from ml.utils.audio import load_audio, resample_audio
from ml.utils.progress import ProgressReporter


def skipped_record(row: dict, reason: str, detail: str | None = None) -> dict:
    return {"id": str(row.get("id", "")), "split": str(row.get("split", "")),
            "reason": reason, **({"detail": detail} if detail else {})}


def write_skipped(path: Path, skipped: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in skipped))


def read_rows(path: Path, skipped: list[dict] | None = None) -> list[dict]:
    skipped = skipped if skipped is not None else []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, TypeError) as exc:
            skipped.append(skipped_record({"id": f"line-{line_number}"}, "invalid_json", str(exc)))
            continue
        if not isinstance(row, dict) or not row.get("id") or row.get("split") not in {"train", "dev", "test"} or not row.get("source_id"):
            skipped.append(skipped_record(row if isinstance(row, dict) else {"id": f"line-{line_number}"},
                                          "invalid_manifest_row"))
            continue
        rows.append(row)
    duplicates = {key for key, count in Counter(str(row["id"]) for row in rows).items() if count > 1}
    if duplicates:
        kept = []
        for row in rows:
            if str(row["id"]) in duplicates:
                skipped.append(skipped_record(row, "duplicate_id"))
            else:
                kept.append(row)
        rows = kept
    if not rows:
        raise ValueError(f"manifest has no usable rows: {path}")
    return rows


def check_splits(rows: list[dict]) -> None:
    seen = {}
    for row in rows:
        for key in ("source_id", "speaker_id"):
            value = row.get(key)
            if value:
                identity = (key, value)
                if identity in seen and seen[identity] != row["split"]:
                    raise ValueError(f"cross-split leakage: {identity}")
                seen[identity] = row["split"]


def filter_split_conflicts(rows: list[dict], skipped: list[dict]) -> list[dict]:
    identity_splits: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        for key in ("source_id", "speaker_id"):
            if row.get(key):
                identity_splits.setdefault((key, str(row[key])), set()).add(str(row["split"]))
    conflicts = {identity for identity, splits in identity_splits.items() if len(splits) > 1}
    kept = []
    for row in rows:
        row_conflicts = [(key, str(row[key])) for key in ("source_id", "speaker_id")
                         if row.get(key) and (key, str(row[key])) in conflicts]
        if row_conflicts:
            skipped.append(skipped_record(row, "cross_split_identity", repr(row_conflicts)))
        else:
            kept.append(row)
    return kept


def _validate_paired_row(row: dict, root: Path,
                         require_scores: bool) -> tuple[dict, str | None, str | None]:
    if not str(row.get("sentence", "")).strip():
        return row, "empty_transcript", None
    try:
        paired_audio(row, root)
        if require_scores:
            from ml.fusion.bridging import perceptual_target
            perceptual_target(torch.tensor(float(row["dnsmos_sig"])),
                              torch.tensor(float(row["dnsmos_bak"])))
    except (KeyError, TypeError, OSError, RuntimeError, ValueError) as exc:
        return row, "invalid_paired_audio", str(exc)
    return row, None, None


def filter_paired_rows(rows: list[dict], root: Path, skipped: list[dict], *,
                       require_scores: bool = False, workers: int = 1,
                       report_progress: bool = False) -> list[dict]:
    kept = []
    validate = partial(_validate_paired_row, root=root, require_scores=require_scores)
    progress = ProgressReporter("bridge-validate", len(rows), None) if report_progress and rows else None
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bridge-validate") as executor:
        results = executor.map(validate, rows)
        for number, (row, reason, detail) in enumerate(results, start=1):
            if reason is not None:
                skipped.append(skipped_record(row, reason, detail))
            else:
                kept.append(row)
            if progress is not None:
                progress.update(number, str(row.get("id", "")))
    return kept


def paired_audio(row: dict, root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    waves = []
    for key in ("noisy_path", "enhanced_path"):
        wave, rate = load_audio(root / row[key])
        wave = resample_audio(wave, rate, 16000)
        if not 400 <= len(wave) <= 480000 or not np.isfinite(wave).all():
            raise ValueError(f"{row['id']}: require finite audio of 25 ms to 30 s; no silent truncation")
        waves.append(torch.from_numpy(wave))
    if waves[0].shape != waves[1].shape:
        raise ValueError(f"{row['id']}: unequal waveform lengths; correct SE alignment upstream")
    return waves[0], waves[1]


def filterbank(wave: torch.Tensor) -> torch.Tensor:
    from torchaudio.compliance.kaldi import fbank
    return fbank(wave.unsqueeze(0), sample_frequency=16000, num_mel_bins=80,
                 frame_length=25, frame_shift=10, dither=0).T.contiguous()


class Recognizer:
    def __init__(self, checkpoint: str, device: str, max_tokens: int):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.processor = WhisperProcessor.from_pretrained(checkpoint, language="Persian", task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained(checkpoint).to(device).eval()
        self.model.requires_grad_(False)
        self.device, self.max_tokens = device, max_tokens

    @torch.inference_mode()
    def transcribe_batch(self, waves: list[torch.Tensor]) -> list[str]:
        """Decode a waveform batch in one Whisper generation call."""
        features = self.processor.feature_extractor(
            [wave.numpy() for wave in waves], sampling_rate=16000, return_tensors="pt",
        ).input_features.to(self.device)
        tokens = self.model.generate(features, language="Persian", task="transcribe", do_sample=False,
                                     num_beams=1, max_new_tokens=self.max_tokens)
        return [text.strip() for text in self.processor.batch_decode(tokens, skip_special_tokens=True)]

    def __call__(self, wave: torch.Tensor) -> str:
        return self.transcribe_batch([wave])[0]


def _cache_frame_count(path: Path) -> int | None:
    try:
        item = torch.load(path, weights_only=True, map_location="cpu")
        required = {"noisy", "enhanced", "wers", "sig", "bak"}
        valid = (isinstance(item, dict) and required.issubset(item)
                 and item["noisy"].ndim == 2 and item["enhanced"].shape == item["noisy"].shape
                 and item["wers"].numel() == len(OA_COEFFICIENTS)
                 and all(torch.is_tensor(item[key]) and torch.isfinite(item[key]).all()
                         for key in required))
        return int(item["noisy"].shape[-1]) if valid else None
    except (EOFError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _valid_cache(path: Path) -> bool:
    return _cache_frame_count(path) is not None


def _resume_prefix(output: Path, rows: list[dict]) -> list[dict]:
    """Return the valid, contiguous cache prefix and discard an interrupted tail."""
    index_path = output / "index.jsonl"
    if not index_path.is_file():
        return []
    records: list[dict] = []
    for line_number, line in enumerate(index_path.read_text().splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            print(f"[bridge-cache] ignoring interrupted index tail at line {line_number}", flush=True)
            break
        number = len(records)
        expected_name = f"{number:08d}.pt"
        if (number >= len(rows) or not isinstance(record, dict)
                or str(record.get("id")) != str(rows[number]["id"])
                or record.get("cache") != expected_name
                or not _valid_cache(output / expected_name)):
            print(f"[bridge-cache] rebuilding invalid cache tail from item {number + 1}", flush=True)
            break
        records.append(record)
    index_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
    return records


def prepare(args: argparse.Namespace) -> None:
    from jiwer import wer
    skipped: list[dict] = []
    rows = read_rows(args.manifest, skipped)
    if any(row["split"] == "test" for row in rows):
        raise ValueError("prepare accepts train/dev only; test references must not generate supervision")
    rows = filter_split_conflicts(rows, skipped)
    check_splits(rows)
    sidecar = args.manifest.with_suffix(".provenance.json")
    identity = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if identity and identity.get("manifest_sha256") != manifest_hash:
        raise ValueError("input manifest changed after preparation; regenerate its provenance")
    for name in ("enhancer_id", "dnsmos_id"):
        explicit = getattr(args, name)
        if explicit and identity.get(name) and explicit != identity[name]:
            raise ValueError(f"--{name.replace('_', '-')} conflicts with generated provenance")
        identity[name] = explicit or identity.get(name)
        if not identity[name]:
            raise ValueError(f"missing {name}; run ml.fusion.prepare_bridge_inputs first or supply an explicit ID")
    print(f"Validating {len(rows)} paired clips with {args.workers} CPU workers...", flush=True)
    rows = filter_paired_rows(rows, args.manifest.parent, skipped, require_scores=True,
                              workers=args.workers, report_progress=True)
    if not rows or not {"train", "dev"}.issubset({str(row["split"]) for row in rows}):
        raise ValueError("usable paired audio is required in both train and dev splits")
    output = args.output
    output.mkdir(parents=True, exist_ok=args.resume)
    write_skipped(output / "skipped_inputs.jsonl", skipped)
    meta = {"paper": "2501.02452v1", "asr_checkpoint": args.asr_checkpoint,
            "enhancer_id": identity["enhancer_id"], "dnsmos_id": identity["dnsmos_id"],
            "coefficients": list(OA_COEFFICIENTS), "max_tokens": args.max_tokens,
            "manifest_sha256": manifest_hash,
            "skipped_inputs": len(skipped),
            "text_policy": "strip_only; normalize all input references upstream identically",
            "implementation": "independent reconstruction; see docs/script-guides/bridging-baseline.md"}
    provenance_path = output / "provenance.json"
    if args.resume:
        if not provenance_path.is_file():
            raise ValueError("cannot resume: output has no provenance.json")
        existing_meta = json.loads(provenance_path.read_text())
        mismatches = [key for key, value in meta.items() if existing_meta.get(key) != value]
        if mismatches:
            raise ValueError(f"cannot resume cache with different provenance fields: {', '.join(mismatches)}")
        records = _resume_prefix(output, rows)
    else:
        records = []
        provenance_path.write_text(json.dumps(meta, indent=2))
    completed = len(records)
    if completed == len(rows):
        print(f"[bridge-cache] already complete: {completed}/{len(rows)}", flush=True)
        return
    with ProgressReporter("bridge-cache", len(rows), output / "progress.json", initial=completed) as progress:
        asr = Recognizer(args.asr_checkpoint, args.device, args.max_tokens)
        load_pair = partial(paired_audio, root=args.manifest.parent)
        with (output / "index.jsonl").open("a") as index, \
                ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="bridge-load") as audio_pool:
            for start in range(completed, len(rows), args.batch_size):
                batch_rows = rows[start:start + args.batch_size]
                pairs = list(audio_pool.map(load_pair, batch_rows))
                mixtures = [observation_addition(noisy, enhanced, torch.tensor(weight))
                            for noisy, enhanced in pairs for weight in OA_COEFFICIENTS]
                decoded = asr.transcribe_batch(mixtures)
                if len(decoded) != len(mixtures):
                    raise RuntimeError("ASR returned the wrong number of batch hypotheses")
                for offset, (row, (noisy, enhanced)) in enumerate(zip(batch_rows, pairs)):
                    number = start + offset
                    first = offset * len(OA_COEFFICIENTS)
                    hypotheses = decoded[first:first + len(OA_COEFFICIENTS)]
                    wers = [wer(row["sentence"].strip(), hypothesis) for hypothesis in hypotheses]
                    cache = {"noisy": filterbank(noisy), "enhanced": filterbank(enhanced),
                             "wers": torch.tensor(wers), "sig": torch.tensor(float(row["dnsmos_sig"])),
                             "bak": torch.tensor(float(row["dnsmos_bak"]))}
                    name = f"{number:08d}.pt"
                    temporary = output / f".{name}.tmp"
                    torch.save(cache, temporary)
                    temporary.replace(output / name)
                    record = {**row, "cache": name, "wers": wers, "hypotheses": hypotheses}
                    index.write(json.dumps(record, ensure_ascii=False) + "\n")
                    index.flush()
                    progress.update(number + 1, row["id"])


def augment(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    for axis, maximum in ((1, 4), (2, 5)):
        width = random.randint(0, min(maximum, x.shape[axis]))
        start = random.randint(0, x.shape[axis] - width)
        slices = [slice(None)] * x.ndim
        slices[axis] = slice(start, start + width)
        x[tuple(slices)] = 0
    return x


def _load_cache_item(cache: Path, row: dict) -> dict[str, torch.Tensor]:
    return torch.load(cache / row["cache"], weights_only=True, map_location="cpu")


def _load_cache_batch(cache: Path, rows: list[dict], device: str,
                      pool: ThreadPoolExecutor) -> dict[str, torch.Tensor]:
    """Load cached examples concurrently and pad only the filterbank time axis."""
    load = partial(_load_cache_item, cache)
    items = list(pool.map(load, rows))
    lengths = torch.tensor([item["noisy"].shape[-1] for item in items])
    frames = int(lengths.max().item())

    def pad(key: str) -> torch.Tensor:
        return torch.stack([
            torch.nn.functional.pad(item[key], (0, frames - item[key].shape[-1]))
            for item in items
        ])

    batch = {
        "noisy": pad("noisy"),
        "enhanced": pad("enhanced"),
        "wers": torch.stack([item["wers"] for item in items]),
        "sig": torch.stack([item["sig"].reshape(()) for item in items]),
        "bak": torch.stack([item["bak"].reshape(()) for item in items]),
        "lengths": lengths,
    }
    return {key: value.to(device) for key, value in batch.items()}


def _augment_batch(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Apply independent masks without selecting padded time steps."""
    augmented = x.clone()
    for index, length in enumerate(lengths.tolist()):
        augmented[index:index + 1, :, :length] = augment(augmented[index:index + 1, :, :length])
    return augmented


def _chunks(rows: list[dict], size: int):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    skipped: list[dict] = []
    rows = read_rows(args.cache / "index.jsonl", skipped)
    if any(r["split"] == "test" for r in rows):
        raise ValueError("test cache cannot be used in training")
    rows = filter_split_conflicts(rows, skipped)
    check_splits(rows)
    print(f"[bridge-train] validating {len(rows)} cache entries with {args.workers} workers", flush=True)
    usable_rows = []
    with ProgressReporter("bridge-cache-check", len(rows), None) as cache_progress, \
            ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="bridge-cache-check") as pool:
        paths = [args.cache / str(row.get("cache", "")) for row in rows]
        for number, (row, cache_path, frames) in enumerate(
                zip(rows, paths, pool.map(_cache_frame_count, paths)), start=1):
            if frames is not None:
                row["_frames"] = frames
                usable_rows.append(row)
            else:
                skipped.append(skipped_record(row, "invalid_cache", str(cache_path)))
            cache_progress.update(number, str(row["id"]))
    train_rows = [r for r in usable_rows if r["split"] == "train"]
    dev_rows = [r for r in usable_rows if r["split"] == "dev"]
    if not train_rows or not dev_rows:
        raise ValueError("both train and dev caches are required")
    args.output.mkdir(parents=True, exist_ok=False)
    write_skipped(args.output / "skipped_inputs.jsonl", skipped)
    model_config = {"channels": args.channels}
    model = BridgingModule(**model_config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    provenance = json.loads((args.cache / "provenance.json").read_text())
    print(
        f"[bridge-train] starting device={args.device} epochs={args.epochs} "
        f"train={len(train_rows)} dev={len(dev_rows)} batch_size={args.batch_size} "
        f"utterances_per_update={args.accumulation} workers={args.workers} "
        f"loss={'PQ' if args.pq_only else 'PQ+RI'}",
        flush=True,
    )
    best = float("inf")
    total_work = args.epochs * (len(train_rows) + len(dev_rows))
    completed_work = 0
    with ProgressReporter("bridge-train", total_work, args.output / "progress.json",
                          report_every_seconds=args.log_every_seconds) as progress, \
            ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="bridge-train-load") as pool:
        for epoch in range(args.epochs):
            print(f"[bridge-train] epoch {epoch + 1}/{args.epochs}: training", flush=True)
            model.train()
            random.shuffle(train_rows)
            optimizer.zero_grad()
            training_loss = 0.0
            for update_rows in _chunks(train_rows, args.accumulation):
                # Sorting only within an optimizer update limits padding without changing update membership.
                update_rows.sort(key=lambda row: row["_frames"])
                group_size = len(update_rows)
                for batch_rows in _chunks(update_rows, args.batch_size):
                    item = _load_cache_batch(args.cache, batch_rows, args.device, pool)
                    outputs = model(_augment_batch(item["noisy"], item["lengths"]),
                                    _augment_batch(item["enhanced"], item["lengths"]),
                                    item["lengths"])
                    loss = bridge_loss(outputs, item["wers"], item["sig"], item["bak"],
                                       recognition=not args.pq_only)
                    if not torch.isfinite(loss):
                        raise ValueError(f"nonfinite loss near {batch_rows[0]['id']}")
                    (loss * (len(batch_rows) / group_size)).backward()
                    training_loss += loss.item() * len(batch_rows)
                    completed_work += len(batch_rows)
                    progress.update(completed_work, f"epoch={epoch + 1}/{args.epochs} train")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                optimizer.zero_grad()
            print(f"[bridge-train] epoch {epoch + 1}/{args.epochs}: validation", flush=True)
            model.eval()
            dev_loss = 0.0
            with torch.inference_mode():
                ordered_dev = sorted(dev_rows, key=lambda row: row["_frames"])
                for batch_rows in _chunks(ordered_dev, args.batch_size):
                    item = _load_cache_batch(args.cache, batch_rows, args.device, pool)
                    out = model(item["noisy"], item["enhanced"], item["lengths"])
                    loss = bridge_loss(out, item["wers"], item["sig"], item["bak"],
                                       recognition=not args.pq_only).item()
                    dev_loss += loss * len(batch_rows)
                    completed_work += len(batch_rows)
                    progress.update(completed_work, f"epoch={epoch + 1}/{args.epochs} dev")
            dev_loss /= len(dev_rows)
            payload = {"state_dict": model.state_dict(), "model_config": model_config,
                       "provenance": provenance, "epoch": epoch + 1, "seed": args.seed,
                       "dev_loss": dev_loss, "optimizer": optimizer.state_dict(),
                       "training": {"lr": args.lr, "epochs": args.epochs,
                                    "batch_size": args.batch_size, "accumulation": args.accumulation,
                                    "workers": args.workers, "pq_only": args.pq_only,
                                    "selection": "dev combined loss"}}
            torch.save(payload, args.output / "last.pt")
            if dev_loss < best:
                best = dev_loss
                torch.save(payload, args.output / "best.pt")
            metrics = {"epoch": epoch + 1, "train_loss": training_loss / len(train_rows),
                       "dev_loss": dev_loss}
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            print(f"[bridge-train] epoch {epoch + 1}/{args.epochs} complete: {metrics}", flush=True)


def evaluate(args: argparse.Namespace) -> None:
    from jiwer import wer, cer
    skipped: list[dict] = []
    rows = read_rows(args.manifest, skipped)
    if any(row["split"] != args.split for row in rows):
        raise ValueError("evaluation manifest must contain exactly the requested split")
    rows = filter_split_conflicts(rows, skipped)
    check_splits(rows)
    rows = filter_paired_rows(rows, args.manifest.parent, skipped)
    if not rows:
        raise ValueError("evaluation manifest has no usable paired audio")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = BridgingModule(**checkpoint["model_config"]).to(args.device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    provenance = checkpoint["provenance"]
    asr = Recognizer(provenance["asr_checkpoint"], args.device, provenance["max_tokens"])
    args.output.mkdir(parents=True, exist_ok=False)
    write_skipped(args.output / "skipped_inputs.jsonl", skipped)
    references, hypotheses = [], []
    with ProgressReporter("bridge-evaluate", len(rows), args.output / "progress.json") as progress, \
            (args.output / "predictions.jsonl").open("w") as stream, torch.inference_mode():
        for number, row in enumerate(rows, start=1):
            noisy, enhanced = paired_audio(row, args.manifest.parent)
            if args.omega is None:
                omega = model(filterbank(noisy)[None].to(args.device),
                              filterbank(enhanced)[None].to(args.device))["omega"].cpu().squeeze(0)
            else:
                omega = torch.tensor(args.omega)
            hypothesis = asr(observation_addition(noisy, enhanced, omega))
            references.append(row["sentence"].strip())
            hypotheses.append(hypothesis)
            stream.write(json.dumps({"id": row["id"], "source_id": row["source_id"],
                                     "reference": references[-1], "hypothesis": hypothesis,
                                     "omega": omega.item()}, ensure_ascii=False) + "\n")
            progress.update(number, row["id"])
    (args.output / "metrics.json").write_text(json.dumps({"wer": wer(references, hypotheses),
        "cer": cer(references, hypotheses), "examples": len(rows), "split": args.split,
        "skipped_inputs": len(skipped),
        "omega_override": args.omega, "provenance": provenance,
        "checkpoint": str(args.checkpoint),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest()}, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subs = parser.add_subparsers(dest="command", required=True)
    fmt = argparse.ArgumentDefaultsHelpFormatter
    p = subs.add_parser("prepare", help="cache paired filterbanks and eleven decoded WER targets", formatter_class=fmt)
    p.add_argument("--manifest", type=Path, required=True, help="train/dev paired-waveform JSONL")
    p.add_argument("--asr-checkpoint", required=True, help="frozen Whisper checkpoint with processor files")
    p.add_argument("--enhancer-id", default=None, help="frozen enhancer ID; default generated manifest provenance")
    p.add_argument("--dnsmos-id", default=None, help="SIG/BAK scorer ID; default generated manifest provenance")
    p.add_argument("--max-tokens", type=int, default=225, help="maximum generated new tokens")
    p.add_argument("--batch-size", type=int, default=8,
                   help="clips per ASR batch (eleven waveform mixtures per clip)")
    p.add_argument("--workers", type=int, default=4,
                   help="CPU worker threads for parallel audio validation and loading")
    p.add_argument("--resume", action="store_true",
                   help="validate and continue an interrupted existing output directory")
    p.set_defaults(func=prepare)
    t = subs.add_parser("train", help="train only the bridging module from cached supervision", formatter_class=fmt)
    t.add_argument("--cache", type=Path, required=True, help="prepared cache directory")
    t.add_argument("--epochs", type=int, default=5, help="training epochs")
    t.add_argument("--lr", type=float, default=0.0005, help="Adam learning rate")
    t.add_argument("--channels", type=int, choices=(256, 384), default=256, help="frame-layer channels")
    t.add_argument("--accumulation", type=int, default=8, help="utterances per optimizer update")
    t.add_argument("--batch-size", type=int, default=4,
                   help="utterances per padded GPU micro-batch; reduce after CUDA out-of-memory")
    t.add_argument("--workers", type=int, default=4,
                   help="CPU worker threads for parallel cache validation and loading")
    t.add_argument("--log-every-seconds", type=float, default=30,
                   help="maximum interval between progress/ETA log messages")
    t.add_argument("--seed", type=int, default=1337, help="random seed")
    t.add_argument("--pq-only", action="store_true", help="ablate recognition-information loss")
    t.set_defaults(func=train)
    e = subs.add_parser("evaluate", help="decode learned or fixed waveform mixtures", formatter_class=fmt)
    e.add_argument("--manifest", type=Path, required=True, help="paired-waveform evaluation JSONL")
    e.add_argument("--checkpoint", type=Path, required=True, help="trained bridge checkpoint")
    e.add_argument("--split", choices=("dev", "test"), default="dev", help="requested evaluation split")
    e.add_argument("--omega", type=float, default=None, help="fixed original-waveform weight; 0=SE, 1=original")
    e.set_defaults(func=evaluate)
    for sub in (p, t, e):
        output_help = ("cache output directory; existing directory allowed only with --resume" if sub is p
                       else "new output directory; existing directories refused")
        sub.add_argument("--output", type=Path, required=True, help=output_help)
        sub.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda:0")
    args = parser.parse_args(argv)
    for key in ("epochs", "accumulation", "max_tokens", "batch_size", "workers", "lr",
                "log_every_seconds"):
        if hasattr(args, key) and getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if getattr(args, "omega", None) is not None and not 0 <= args.omega <= 1:
        parser.error("--omega must be in [0, 1]")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
