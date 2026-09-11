"""Prepare, train and evaluate the waveform bridging baseline; --help for workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from ml.fusion.bridging import BridgingModule, OA_COEFFICIENTS, bridge_loss, observation_addition
from ml.utils.audio import load_audio, resample_audio


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("manifest must be nonempty with unique IDs")
    for row in rows:
        if row["split"] not in {"train", "dev", "test"} or not row["source_id"]:
            raise ValueError("every row needs split=train/dev/test and original source_id")
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
    def __call__(self, wave: torch.Tensor) -> str:
        features = self.processor.feature_extractor(wave.numpy(), sampling_rate=16000,
                                                   return_tensors="pt").input_features.to(self.device)
        tokens = self.model.generate(features, language="Persian", task="transcribe", do_sample=False,
                                     num_beams=1, max_new_tokens=self.max_tokens)
        return self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


def prepare(args: argparse.Namespace) -> None:
    from jiwer import wer
    rows = read_rows(args.manifest)
    check_splits(rows)
    if any(row["split"] == "test" for row in rows):
        raise ValueError("prepare accepts train/dev only; test references must not generate supervision")
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
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    asr = Recognizer(args.asr_checkpoint, args.device, args.max_tokens)
    meta = {"paper": "2501.02452v1", "asr_checkpoint": args.asr_checkpoint,
            "enhancer_id": identity["enhancer_id"], "dnsmos_id": identity["dnsmos_id"],
            "coefficients": OA_COEFFICIENTS, "max_tokens": args.max_tokens,
            "manifest_sha256": manifest_hash,
            "text_policy": "strip_only; normalize all input references upstream identically",
            "implementation": "independent reconstruction; see docs/script-guides/bridging-baseline.md"}
    (output / "provenance.json").write_text(json.dumps(meta, indent=2))
    with (output / "index.jsonl").open("w") as index:
        for number, row in enumerate(rows):
            if not row["sentence"].strip():
                raise ValueError("empty reference transcript")
            noisy, enhanced = paired_audio(row, args.manifest.parent)
            hypotheses = [asr(observation_addition(noisy, enhanced, torch.tensor(w))) for w in OA_COEFFICIENTS]
            wers = [wer(row["sentence"].strip(), h) for h in hypotheses]
            # DNSMOS comes from a named frozen scorer, not reconstruction error or invented labels.
            from ml.fusion.bridging import perceptual_target
            perceptual_target(torch.tensor(row["dnsmos_sig"]), torch.tensor(row["dnsmos_bak"]))
            cache = {"noisy": filterbank(noisy), "enhanced": filterbank(enhanced),
                     "wers": torch.tensor(wers), "sig": torch.tensor(float(row["dnsmos_sig"])),
                     "bak": torch.tensor(float(row["dnsmos_bak"]))}
            name = f"{number:08d}.pt"
            torch.save(cache, output / name)
            record = {**row, "cache": name, "wers": wers, "hypotheses": hypotheses}
            index.write(json.dumps(record, ensure_ascii=False) + "\n")
            index.flush()
            print(f"prepared {number + 1}/{len(rows)}: {row['id']}", flush=True)


def augment(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    for axis, maximum in ((1, 4), (2, 5)):
        width = random.randint(0, min(maximum, x.shape[axis]))
        start = random.randint(0, x.shape[axis] - width)
        slices = [slice(None)] * x.ndim
        slices[axis] = slice(start, start + width)
        x[tuple(slices)] = 0
    return x


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_rows(args.cache / "index.jsonl")
    check_splits(rows)
    if any(r["split"] == "test" for r in rows):
        raise ValueError("test cache cannot be used in training")
    train_rows = [r for r in rows if r["split"] == "train"]
    dev_rows = [r for r in rows if r["split"] == "dev"]
    if not train_rows or not dev_rows:
        raise ValueError("both train and dev caches are required")
    args.output.mkdir(parents=True, exist_ok=False)
    model_config = {"channels": args.channels}
    model = BridgingModule(**model_config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    provenance = json.loads((args.cache / "provenance.json").read_text())
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train_rows)
        optimizer.zero_grad()
        training_loss = 0.0
        for i, row in enumerate(train_rows):
            item = torch.load(args.cache / row["cache"], weights_only=True, map_location=args.device)
            outputs = model(augment(item["noisy"][None]), augment(item["enhanced"][None]))
            loss = bridge_loss(outputs, item["wers"][None], item["sig"][None], item["bak"][None],
                               recognition=not args.pq_only)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite loss on {row['id']}")
            group_start = (i // args.accumulation) * args.accumulation
            group_size = min(args.accumulation, len(train_rows) - group_start)
            (loss / group_size).backward()
            training_loss += loss.item()
            if (i + 1) % args.accumulation == 0 or i + 1 == len(train_rows):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                optimizer.zero_grad()
        model.eval()
        dev_loss = 0.0
        with torch.inference_mode():
            for row in dev_rows:
                item = torch.load(args.cache / row["cache"], weights_only=True, map_location=args.device)
                out = model(item["noisy"][None], item["enhanced"][None])
                dev_loss += bridge_loss(out, item["wers"][None], item["sig"][None], item["bak"][None],
                                        recognition=not args.pq_only).item()
        dev_loss /= len(dev_rows)
        payload = {"state_dict": model.state_dict(), "model_config": model_config,
                   "provenance": provenance, "epoch": epoch + 1, "seed": args.seed,
                   "dev_loss": dev_loss, "optimizer": optimizer.state_dict(),
                   "training": {"lr": args.lr, "epochs": args.epochs, "accumulation": args.accumulation,
                                "pq_only": args.pq_only, "selection": "dev combined loss"}}
        torch.save(payload, args.output / "last.pt")
        if dev_loss < best:
            best = dev_loss
            torch.save(payload, args.output / "best.pt")
        metrics = {"epoch": epoch + 1, "train_loss": training_loss / len(train_rows), "dev_loss": dev_loss}
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print(metrics, flush=True)


def evaluate(args: argparse.Namespace) -> None:
    from jiwer import wer, cer
    rows = read_rows(args.manifest)
    check_splits(rows)
    if any(row["split"] != args.split for row in rows):
        raise ValueError("evaluation manifest must contain exactly the requested split")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = BridgingModule(**checkpoint["model_config"]).to(args.device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    provenance = checkpoint["provenance"]
    asr = Recognizer(provenance["asr_checkpoint"], args.device, provenance["max_tokens"])
    args.output.mkdir(parents=True, exist_ok=False)
    references, hypotheses = [], []
    with (args.output / "predictions.jsonl").open("w") as stream, torch.inference_mode():
        for row in rows:
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
    (args.output / "metrics.json").write_text(json.dumps({"wer": wer(references, hypotheses),
        "cer": cer(references, hypotheses), "examples": len(rows), "split": args.split,
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
    p.set_defaults(func=prepare)
    t = subs.add_parser("train", help="train only the bridging module from cached supervision", formatter_class=fmt)
    t.add_argument("--cache", type=Path, required=True, help="prepared cache directory")
    t.add_argument("--epochs", type=int, default=45, help="training epochs")
    t.add_argument("--lr", type=float, default=0.0005, help="Adam learning rate")
    t.add_argument("--channels", type=int, choices=(256, 384), default=256, help="frame-layer channels")
    t.add_argument("--accumulation", type=int, default=8, help="utterances per optimizer update")
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
        sub.add_argument("--output", type=Path, required=True, help="new output directory; existing directories refused")
        sub.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda:0")
    args = parser.parse_args(argv)
    for key in ("epochs", "accumulation", "max_tokens", "lr"):
        if hasattr(args, key) and getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if getattr(args, "omega", None) is not None and not 0 <= args.omega <= 1:
        parser.error("--omega must be in [0, 1]")
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
