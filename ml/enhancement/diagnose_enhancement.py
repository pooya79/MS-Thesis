"""Measure how much an enhancer actually denoises, in log-Mel (and encoder) space.

The warm-up objective is ``L_enh`` = mean L1 between the enhanced log-Mel and the
bandwidth-aligned clean log-Mel. On its own that number is hard to read: a small
``L_enh`` can mean "great enhancer" *or* "the noisy mel was already close to clean,
so the identity scores about the same". This script settles that by reporting the
**headroom**:

- ``identity_L_enh`` — L1 between the *noisy* and clean mel, i.e. what an enhancer
  that does nothing (the identity) scores. This is the ceiling on achievable
  improvement in this metric.
- ``trained_L_enh`` — L1 between the *enhanced* and clean mel for a given enhancer
  checkpoint (omit ``--enhancer-checkpoint`` to report the identity baseline only).
- ``captured`` — the fraction of the headroom the enhancer removed,
  ``(identity - trained) / identity``. Near 0 means the enhancer is ~inert.

Results are broken down by degradation ``target_bandwidth`` (e.g. narrowband /
telephone vs wideband), since an enhancer can help on one channel type and not
another. With ``--feature-encoder`` it additionally reports the same identity-vs-
trained comparison in the **Whisper encoder feature space** (L1 between
``encoder(mel)`` outputs) — the distance that actually correlates with WER, and the
target of the warm-up feature-matching loss. ``--dump-mels N`` saves the first N
clips' noisy/clean/enhanced log-Mels as ``.npy`` arrays for offline plotting.

Example::

    uv run python -m ml.enhancement.diagnose_enhancement \\
      --dataset data/cv-corpus-25.0-degraded-v2 \\
      --split dev \\
      --enhancer-checkpoint artifacts/.../checkpoints/stage0_warmup/enhancer.pt \\
      --feature-encoder models/asr/whisper-small/runs/best \\
      --output-dir artifacts/enhancement_diagnosis
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.asr.whisper_features import WHISPER_SAMPLE_RATE
from ml.enhancement.dataset import DegradedMelDataset, collate_mels
from ml.enhancement.enhancer import build_enhancer


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class GroupStats:
    """Running per-item L1 sums for one (dataset, degradation-type) bucket."""

    examples: int = 0
    identity_mel_sum: float = 0.0
    trained_mel_sum: float = 0.0
    identity_feat_sum: float = 0.0
    trained_feat_sum: float = 0.0
    have_trained: bool = False
    have_feat: bool = False

    def add(
        self,
        identity_mel: float,
        trained_mel: float | None,
        identity_feat: float | None,
        trained_feat: float | None,
    ) -> None:
        self.examples += 1
        self.identity_mel_sum += identity_mel
        if trained_mel is not None:
            self.trained_mel_sum += trained_mel
            self.have_trained = True
        if identity_feat is not None:
            self.identity_feat_sum += identity_feat
            self.have_feat = True
        if trained_feat is not None:
            self.trained_feat_sum += trained_feat

    def summary(self) -> dict[str, Any]:
        n = max(1, self.examples)
        identity_mel = self.identity_mel_sum / n
        out: dict[str, Any] = {"examples": self.examples, "identity_L_enh": identity_mel}
        if self.have_trained:
            trained_mel = self.trained_mel_sum / n
            out["trained_L_enh"] = trained_mel
            out["captured"] = _captured(identity_mel, trained_mel)
        if self.have_feat:
            identity_feat = self.identity_feat_sum / n
            out["identity_L_feat"] = identity_feat
            if self.have_trained:
                trained_feat = self.trained_feat_sum / n
                out["trained_L_feat"] = trained_feat
                out["captured_feat"] = _captured(identity_feat, trained_feat)
        return out


def _captured(identity: float, trained: float) -> float:
    """Fraction of the headroom removed: (identity - trained) / identity."""
    if identity <= 0:
        return 0.0
    return (identity - trained) / identity


@dataclass
class Accumulator:
    overall: GroupStats = field(default_factory=GroupStats)
    by_dataset: dict[str, GroupStats] = field(default_factory=dict)
    by_bandwidth: dict[str, GroupStats] = field(default_factory=dict)

    def add(self, dataset: str, bandwidth: str, *values: Any) -> None:
        self.overall.add(*values)
        self.by_dataset.setdefault(dataset, GroupStats()).add(*values)
        self.by_bandwidth.setdefault(bandwidth, GroupStats()).add(*values)


def load_enhancer_from_checkpoint(checkpoint: Path, device: str) -> Any:
    """Rebuild an enhancer from an ``enhancer.pt`` or a ``fusion_model.pt`` payload.

    The architecture is read back from ``enhancer_config``; enhancer weights load
    from a bare enhancer state dict or, for a fusion checkpoint, from the
    ``enhancer.*`` subset of the combined state.
    """
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    enhancer = build_enhancer(payload.get("enhancer_config"))
    state = payload["model_state"]
    enhancer_state = {
        key[len("enhancer.") :]: value for key, value in state.items() if key.startswith("enhancer.")
    }
    enhancer.load_state_dict(enhancer_state or state)
    enhancer.eval().to(device)
    return enhancer


def load_feature_encoder(checkpoint: str, model_name: str, device: str) -> Any:
    """Load a frozen Whisper encoder (for the encoder-feature-space distance)."""
    from ml.fusion.model import load_whisper_backbone

    whisper = load_whisper_backbone(checkpoint, model_name=model_name)
    encoder = whisper.get_encoder()
    for param in encoder.parameters():
        param.requires_grad_(False)
    encoder.eval().to(device)
    return encoder


def resolve_device(requested: str) -> str:
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def make_progress_bar(iterable: Any, desc: str, total: int | None = None) -> Any:
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, total=total, unit="batch", dynamic_ncols=True, leave=False, disable=None)
    except ImportError:
        return iterable


def _per_item_l1(a: Any, b: Any) -> Any:
    """Mean absolute error per batch item over all non-batch dims -> [B]."""
    diff = (a - b).abs()
    return diff.flatten(start_dim=1).mean(dim=1)


def diagnose_dataset(
    dataset_dir: Path,
    *,
    split: str,
    clean_target: str,
    model_name: str,
    batch_size: int,
    device: str,
    max_batches: int | None,
    enhancer: Any,
    feat_encoder: Any,
    accumulator: Accumulator,
    dump_mels: int,
    dump_dir: Path | None,
) -> None:
    """Accumulate identity/trained L1 (mel and optional encoder space) for one dataset."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    dataset = DegradedMelDataset(
        dataset_dir,
        split=split,
        clean_target=clean_target,
        segment_seconds=None,
        model_name=model_name,
        return_labels=False,
    )
    # shuffle=False so item order matches dataset.pairs -> we can label each item
    # with its degradation target_bandwidth.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_mels)
    bandwidths = [str(pair.degradation.get("target_bandwidth", "unknown")) for pair in dataset.pairs]
    name = dataset_dir.name
    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)

    dumped = 0
    index = 0
    with torch.no_grad():
        for batch_no, batch in enumerate(make_progress_bar(loader, f"diagnose {name}", total=total_batches)):
            if max_batches is not None and batch_no >= max_batches:
                break
            noisy = batch["noisy_mel"].to(device)
            clean = batch["clean_mel"].to(device)
            identity_mel = _per_item_l1(noisy, clean).cpu().tolist()

            enhanced = enhancer(noisy) if enhancer is not None else None
            trained_mel = _per_item_l1(enhanced, clean).cpu().tolist() if enhanced is not None else None

            identity_feat = trained_feat = None
            if feat_encoder is not None:
                clean_feat = feat_encoder(clean).last_hidden_state
                identity_feat = _per_item_l1(feat_encoder(noisy).last_hidden_state, clean_feat).cpu().tolist()
                if enhanced is not None:
                    trained_feat = _per_item_l1(feat_encoder(enhanced).last_hidden_state, clean_feat).cpu().tolist()

            for row in range(noisy.shape[0]):
                bandwidth = bandwidths[index] if index < len(bandwidths) else "unknown"
                accumulator.add(
                    name,
                    bandwidth,
                    identity_mel[row],
                    None if trained_mel is None else trained_mel[row],
                    None if identity_feat is None else identity_feat[row],
                    None if trained_feat is None else trained_feat[row],
                )
                index += 1

            if dump_dir is not None and dumped < dump_mels:
                for row in range(noisy.shape[0]):
                    if dumped >= dump_mels:
                        break
                    stem = dump_dir / f"{name}_{split}_{dumped:03d}"
                    np.save(f"{stem}_noisy.npy", noisy[row].float().cpu().numpy())
                    np.save(f"{stem}_clean.npy", clean[row].float().cpu().numpy())
                    if enhanced is not None:
                        np.save(f"{stem}_enhanced.npy", enhanced[row].float().cpu().numpy())
                    dumped += 1


def _format_group(label: str, summary: dict[str, Any]) -> str:
    parts = [f"identity_L_enh={summary['identity_L_enh']:.4f}"]
    if "trained_L_enh" in summary:
        parts.append(f"trained_L_enh={summary['trained_L_enh']:.4f}")
        parts.append(f"captured={summary['captured'] * 100:.1f}%")
    if "identity_L_feat" in summary:
        parts.append(f"identity_L_feat={summary['identity_L_feat']:.4f}")
    if "trained_L_feat" in summary:
        parts.append(f"trained_L_feat={summary['trained_L_feat']:.4f}")
        parts.append(f"captured_feat={summary['captured_feat'] * 100:.1f}%")
    return f"  {label} (n={summary['examples']}): " + " ".join(parts)


def run_diagnosis(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    device = resolve_device(args.device)

    enhancer = None
    if args.enhancer_checkpoint is not None:
        logging.info("loading enhancer checkpoint=%s", args.enhancer_checkpoint)
        enhancer = load_enhancer_from_checkpoint(args.enhancer_checkpoint, device)
    else:
        logging.info("no --enhancer-checkpoint: reporting the identity (do-nothing) baseline only")

    feat_encoder = None
    if args.feature_encoder is not None:
        logging.info("loading feature encoder=%s", args.feature_encoder)
        feat_encoder = load_feature_encoder(args.feature_encoder, args.model_name, device)

    dump_dir = None
    if args.output_dir is not None and args.dump_mels > 0:
        dump_dir = Path(args.output_dir) / "mels"
        dump_dir.mkdir(parents=True, exist_ok=True)

    accumulator = Accumulator()
    for dataset_dir in args.dataset:
        diagnose_dataset(
            Path(dataset_dir),
            split=args.split,
            clean_target=args.clean_target,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=device,
            max_batches=args.max_batches,
            enhancer=enhancer,
            feat_encoder=feat_encoder,
            accumulator=accumulator,
            dump_mels=args.dump_mels,
            dump_dir=dump_dir,
        )

    report = {
        "created_at": utc_now(),
        "split": args.split,
        "clean_target": args.clean_target,
        "datasets": [str(d) for d in args.dataset],
        "enhancer_checkpoint": None if args.enhancer_checkpoint is None else str(args.enhancer_checkpoint),
        "feature_encoder": args.feature_encoder,
        "overall": accumulator.overall.summary(),
        "by_dataset": {name: stats.summary() for name, stats in accumulator.by_dataset.items()},
        "by_bandwidth": {name: stats.summary() for name, stats in accumulator.by_bandwidth.items()},
    }

    logging.info("=== enhancement diagnosis (split=%s) ===", args.split)
    logging.info(_format_group("overall", report["overall"]))
    logging.info("by degradation target_bandwidth:")
    for name in sorted(report["by_bandwidth"]):
        logging.info(_format_group(name, report["by_bandwidth"][name]))
    logging.info("by dataset:")
    for name in sorted(report["by_dataset"]):
        logging.info(_format_group(name, report["by_dataset"][name]))

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "diagnosis.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logging.info("wrote report=%s", report_path)
        if dump_dir is not None:
            logging.info("dumped up to %s mels per dataset to %s", args.dump_mels, dump_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report identity-vs-trained enhancement loss (the L_enh headroom) on a degraded "
            "dataset, broken down by degradation type, with an optional Whisper-encoder-space "
            "distance. Omit --enhancer-checkpoint for the identity baseline only."
        )
    )
    parser.add_argument("--dataset", action="append", required=True, help="Degraded dataset dir (repeatable).")
    parser.add_argument("--split", default="dev", help="Split to evaluate (default: dev).")
    parser.add_argument("--enhancer-checkpoint", type=Path, default=None, help="enhancer.pt or fusion_model.pt to evaluate.")
    parser.add_argument(
        "--feature-encoder",
        default=None,
        help="Whisper backbone (run dir or Hub id) for the encoder-feature-space distance (e.g. the fine-tuned Persian checkpoint).",
    )
    parser.add_argument("--clean-target", choices=["bandwidth_aligned", "full_band"], default="bandwidth_aligned")
    parser.add_argument("--model-name", default="openai/whisper-small", help="Whisper processor for the log-Mel front end.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-batches", type=int, default=None, help="Cap batches per dataset (default: whole split).")
    parser.add_argument("--dump-mels", type=int, default=0, help="Save the first N clips' noisy/clean/enhanced mels as .npy (needs --output-dir).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Write diagnosis.json (and dumped mels) here.")
    parser.add_argument("--sample-rate", type=int, default=WHISPER_SAMPLE_RATE, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    return run_diagnosis(args)


if __name__ == "__main__":
    raise SystemExit(main())
