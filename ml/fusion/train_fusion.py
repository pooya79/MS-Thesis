"""Single config-driven orchestrator for the 3-stage enhancement+fusion curriculum.

One invocation runs the whole curriculum (D8/D10) and writes every artifact to one
run directory:

| Stage | Key   | Trains             | Loss                      |
|-------|-------|--------------------|---------------------------|
| 0     | warmup| enhancer E         | L_enh                     |
| 1     | fusion| E + fusion (Whisper frozen) | L_ASR + lambda*L_enh |
| 2     | joint | E + fusion + Whisper        | L_ASR + lambda*L_enh |

It consumes one or more datasets listed under ``datasets`` (see
``ml.enhancement.dataset``). Each entry is classified automatically: a
*degraded* dataset (a ``ml.speech_data.generate_degraded_dataset`` output with a
``degraded_to_clean.jsonl`` mapping) drives every stage, while a *clean*
(non-degraded) ASR dataset — the project split-TSV + ``clips/`` contract — is
folded into the **joint stage only**, where its undegraded audio keeps the full
stack from regressing on clean speech (the enhancer/fusion see clean input
against an identity target). At least one degraded dataset is required. The
legacy single ``dataset_dir`` is still accepted as one degraded dataset.

All three stages are implemented. Stage 0 trains the enhancer alone on ``L_enh``.
Stages 1-2 build the encoder-feature-space fusion model (``ml/fusion/model.py``)
on top of the warmed enhancer and the fine-tuned Persian Whisper backbone, and
optimise ``L_ASR + lambda * L_enh`` — Stage 1 with the backbone frozen, Stage 2
end to end.

Each stage validates on ``valid_split`` every ``eval_every`` steps — Stage 0 by
dev ``L_enh``, Stages 1-2 by dev WER/CER decoded through the fused encoder — and
keeps the best-scoring weights as ``best.pt`` (dev metrics logged to
``logs/eval_metrics.jsonl``). Eval is skipped automatically when no usable dev
split exists. ``last.pt`` is a rolling checkpoint that carries optimizer/scaler
state and the step, so a stage resumes mid-way on the next invocation. At each
stage boundary the canonical ``enhancer.pt`` / ``fusion_model.pt`` are written
from the stage's **best dev** weights (``best.pt`` reloaded before the final
save; falling back to the final-step weights when no dev split produced one), so
both the next stage and the ``resume_from_stage`` path start from the best model
rather than the last step. Seeding goes through
``transformers.set_seed`` plus seeded dataloaders for run-to-run determinism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ml.asr.train_whisper_small import character_error_rate, word_error_rate
from ml.asr.whisper_features import WHISPER_SAMPLE_RATE
from ml.enhancement.dataset import (
    CleanMelDataset,
    DegradedMelDataset,
    collate_mels,
    detect_dataset_kind,
)
from ml.enhancement.enhancer import build_enhancer, enhancement_l1_loss
from ml.fusion.model import build_fusion_model, is_loadable_checkpoint

STAGE_ORDER = ["warmup", "fusion", "joint"]
STAGE_DIRS = {"warmup": "stage0_warmup", "fusion": "stage1_fusion", "joint": "stage2_joint"}

DEFAULT_CONFIG: dict[str, Any] = {
    "run_dir": "artifacts/speech_enhancement/fusion/run_001",
    "base_asr_checkpoint": None,
    # `datasets` is the multi-dataset list (each: a path string, or {path, kind}
    # with kind in {degraded, clean}; kind auto-detected when omitted). When left
    # null the legacy single `dataset_dir` is used as one degraded dataset.
    "datasets": None,
    "dataset_dir": "data/cv-corpus-25.0-degraded",
    "sample_rate": WHISPER_SAMPLE_RATE,
    "train_split": "train",
    "valid_split": "dev",
    "clean_target": "bandwidth_aligned",
    "model_name": "openai/whisper-small",
    # Language/task drive Whisper's decoder prompt during dev-set generation.
    "language": "Persian",
    "task": "transcribe",
    "generation_max_length": 225,
    "mixed_precision": "auto",
    "device": "auto",
    "seed": 1337,
    "resume_from_stage": None,
    "enhancer": {"type": "residual_unet", "base_channels": 32, "depth": 3},
    "fusion": {"type": "cross_attention", "num_layers": 2, "num_heads": 8, "ffn_ratio": 2.0},
    "stages": {
        "warmup": {
            "max_steps": 5000,
            "batch_size": 8,
            "segment_seconds": 4.0,
            "lr_enhancer": 2e-4,
            # Cosine LR decay to 0 with a linear warm-up; "none"/"constant" keeps a flat LR.
            "lr_scheduler": "cosine",
            "warmup_steps": 500,
            "lambda": 1.0,
            # Encoder-feature-matching loss (D5+): when > 0, add
            # feature_match_weight * L1(encoder(enhanced), encoder(clean)) to the
            # warm-up objective so the enhancer is optimised to look clean to the
            # Whisper encoder, not just to minimise raw mel L1 (a weak ASR proxy).
            # 0.0 keeps the original mel-L1-only warm-up. lambda weights L_enh when
            # feature matching is on. Needs base_asr_checkpoint for the encoder.
            "feature_match_weight": 0.0,
            "num_workers": 4,
            "log_every": 50,
            "eval_every": 500,
            "save_every": 1000,
            # Cap dev batches so periodic eval stays cheap (None = whole dev split).
            "eval_max_batches": None,
        },
        "fusion": {
            "max_steps": 20000,
            "batch_size": 8,
            "lr_frontend": 2e-4,
            "lr_scheduler": "cosine",
            "warmup_steps": 1000,
            "lambda": 0.3,
            "eval_every": 1000,
            "save_every": 1000,
            "eval_max_batches": 50,
        },
        "joint": {
            "max_steps": 40000,
            "batch_size": 4,
            "lr_frontend": 1e-4,
            "lr_whisper": 1e-5,
            "lr_scheduler": "cosine",
            "warmup_steps": 2000,
            "lambda": 0.1,
            "eval_every": 2000,
            "save_every": 1000,
            "eval_max_batches": 50,
        },
    },
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_fusion_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    config = deep_merge(DEFAULT_CONFIG, loaded)
    validate_fusion_config(config)
    return config


def validate_fusion_config(config: dict[str, Any]) -> None:
    datasets = config.get("datasets")
    if datasets is not None:
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("datasets must be a non-empty list when set")
        for entry in datasets:
            path, kind = _dataset_entry_fields(entry)
            if not path:
                raise ValueError(f"each datasets entry needs a non-empty path; got {entry!r}")
            if kind is not None and kind not in {"degraded", "clean"}:
                raise ValueError(f"datasets entry kind must be 'degraded' or 'clean'; got {kind!r}")
    elif not str(config.get("dataset_dir", "")).strip():
        raise ValueError("set datasets (or the legacy dataset_dir) to at least one dataset directory")
    if config["clean_target"] not in {"bandwidth_aligned", "full_band"}:
        raise ValueError("clean_target must be 'bandwidth_aligned' or 'full_band'")
    resume = config.get("resume_from_stage")
    if resume is not None and resume not in STAGE_ORDER and resume not in {0, 1, 2}:
        raise ValueError(f"resume_from_stage must be null, 0-2, or one of {STAGE_ORDER}")
    for name, stage in config["stages"].items():
        if name not in STAGE_ORDER:
            raise ValueError(f"unknown stage {name!r}; expected one of {STAGE_ORDER}")
        if int(stage.get("max_steps", 0)) < 1:
            raise ValueError(f"stage {name}.max_steps must be >= 1")
        if int(stage.get("batch_size", 0)) < 1:
            raise ValueError(f"stage {name}.batch_size must be >= 1")


def _dataset_entry_fields(entry: Any) -> tuple[str, str | None]:
    """Pull ``(path, kind)`` out of a ``datasets`` entry (string or mapping)."""
    if isinstance(entry, str):
        return entry.strip(), None
    if isinstance(entry, dict):
        path = str(entry.get("path", "")).strip()
        kind = entry.get("kind")
        return path, (str(kind).strip() if kind is not None else None)
    raise ValueError(f"datasets entry must be a path string or a mapping; got {entry!r}")


def resolve_dataset_specs(config: dict[str, Any]) -> list[tuple[Path, str]]:
    """Normalise the config into ``[(path, kind), ...]`` with kinds resolved.

    Honours an explicit ``kind`` and otherwise auto-detects it from the directory
    (:func:`detect_dataset_kind`). Falls back to the legacy single ``dataset_dir``
    (always degraded) when ``datasets`` is unset.
    """
    datasets = config.get("datasets")
    if not datasets:
        return [(Path(str(config["dataset_dir"])), "degraded")]
    specs: list[tuple[Path, str]] = []
    for entry in datasets:
        raw_path, kind = _dataset_entry_fields(entry)
        path = Path(raw_path)
        specs.append((path, kind or detect_dataset_kind(path)))
    return specs


def degraded_dataset_dirs(config: dict[str, Any]) -> list[Path]:
    return [path for path, kind in resolve_dataset_specs(config) if kind == "degraded"]


def clean_dataset_dirs(config: dict[str, Any]) -> list[Path]:
    return [path for path, kind in resolve_dataset_specs(config) if kind == "clean"]


def require_base_checkpoint(config: dict[str, Any]) -> None:
    """Fail loudly, up front, if the fine-tuned Whisper checkpoint is absent.

    Stages 1-2 build on the Persian-fine-tuned backbone at ``base_asr_checkpoint``.
    Without it the only previous behaviour was a silent fall-back to vanilla
    Whisper-small, which would quietly invalidate every fusion result. We check
    before any stage runs so Stage 0 never burns time ahead of an inevitable
    Stage 1 failure.
    """
    checkpoint = str(config.get("base_asr_checkpoint") or "").strip()
    if not checkpoint:
        raise FileNotFoundError(
            "base_asr_checkpoint must be set to the fine-tuned Persian Whisper-small "
            "checkpoint (e.g. models/asr/whisper-small/runs/best), or a Hugging Face "
            "Hub id (e.g. openai/whisper-small) to baseline on vanilla Whisper."
        )
    if not is_loadable_checkpoint(checkpoint):
        raise FileNotFoundError(
            f"base_asr_checkpoint does not exist: {checkpoint}. Point it at the "
            "fine-tuned Persian Whisper-small checkpoint, or a Hugging Face Hub id "
            "(e.g. openai/whisper-small) to baseline on vanilla Whisper."
        )


def resolve_start_index(resume_from_stage: Any) -> int:
    if resume_from_stage is None:
        return 0
    if isinstance(resume_from_stage, int):
        return int(resume_from_stage)
    return STAGE_ORDER.index(str(resume_from_stage))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def manifest_hashes(dataset_dirs: list[Path]) -> dict[str, str]:
    """SHA-256 of each degraded dataset's mapping, keyed by ``<dir>/<mapping>``."""
    hashes: dict[str, str] = {}
    for dataset_dir in dataset_dirs:
        mapping = dataset_dir / "degraded_to_clean.jsonl"
        if not mapping.is_file():
            continue
        digest = hashlib.sha256()
        with mapping.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        hashes[f"{dataset_dir.name}/{mapping.name}"] = digest.hexdigest()
    return hashes


def resolve_device(device: str) -> str:
    import torch

    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("device is cuda, but CUDA is not available")
        return "cuda"
    return "cpu"


def use_amp(mixed_precision: Any, device: str) -> bool:
    if device != "cuda":
        return False
    if mixed_precision == "auto":
        return True
    return mixed_precision in {True, "true"}


def seed_everything(seed: int) -> None:
    """Seed Python/NumPy/Torch/CUDA via ``transformers.set_seed`` (matches the ASR trainer)."""
    from transformers import set_seed

    set_seed(int(seed))


def _seeded_generator(seed: int) -> Any:
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _worker_init_fn(worker_id: int) -> None:
    """Give each dataloader worker a distinct, run-reproducible RNG state."""
    import random

    import numpy as np
    import torch

    base = torch.initial_seed() % (2**32)
    seed = (base + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> Any:
    """Build a deterministic ``DataLoader`` (seeded shuffle generator + workers)."""
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        collate_fn=collate_mels,
        drop_last=False,
        generator=_seeded_generator(seed) if shuffle else None,
        worker_init_fn=_worker_init_fn if num_workers else None,
        persistent_workers=bool(num_workers),
    )


def concat_datasets(datasets: list[Any]) -> Any | None:
    """Return the single dataset, a ``ConcatDataset`` of several, or ``None``."""
    from torch.utils.data import ConcatDataset

    if not datasets:
        return None
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def build_train_dataset(
    config: dict[str, Any],
    *,
    segment_seconds: float | None,
    return_labels: bool,
    tokenizer: Any,
    include_clean: bool,
) -> Any:
    """Combined training dataset for one stage.

    Degraded datasets always participate; clean ASR datasets join only when
    ``include_clean`` (the joint stage), where their undegraded audio fine-tunes
    the full stack on clean speech. Raises when no degraded dataset is configured,
    since every stage's enhancement objective needs degraded/clean pairs.
    """
    degraded_dirs = degraded_dataset_dirs(config)
    if not degraded_dirs:
        raise ValueError(
            "no degraded dataset configured; the curriculum needs at least one "
            "generate_degraded_dataset directory (set datasets or dataset_dir)"
        )
    datasets = [
        DegradedMelDataset(
            dataset_dir,
            split=config["train_split"],
            clean_target=config["clean_target"],
            segment_seconds=segment_seconds,
            model_name=config["model_name"],
            return_labels=return_labels,
            tokenizer=tokenizer,
            seed=int(config["seed"]),
        )
        for dataset_dir in degraded_dirs
    ]
    if include_clean:
        for dataset_dir in clean_dataset_dirs(config):
            datasets.append(
                CleanMelDataset(
                    dataset_dir,
                    split=config["train_split"],
                    model_name=config["model_name"],
                    return_labels=return_labels,
                    tokenizer=tokenizer,
                    sample_rate=int(config.get("sample_rate", WHISPER_SAMPLE_RATE)),
                )
            )
    return concat_datasets(datasets)


def build_dev_loader(
    config: dict[str, Any],
    stage: dict[str, Any],
    *,
    return_labels: bool,
    tokenizer: Any = None,
) -> Any | None:
    """Build a dev-split loader, or ``None`` when no usable validation split exists.

    Best-checkpoint selection is measured on the **degraded** datasets — that
    degraded WER is the metric the fusion system must beat. The joint stage
    additionally reports clean-dataset dev metrics (see ``build_clean_dev_loader``)
    so clean-speech regression is visible, but does not select on them.
    Returning ``None`` (rather than raising) lets a run proceed without a ``dev``
    split — periodic eval is simply skipped, keeping small/smoke datasets usable.
    """
    valid_split = config.get("valid_split")
    if not valid_split:
        return None
    dev_datasets: list[Any] = []
    for dataset_dir in degraded_dataset_dirs(config):
        try:
            dev_datasets.append(
                DegradedMelDataset(
                    dataset_dir,
                    split=str(valid_split),
                    clean_target=config["clean_target"],
                    segment_seconds=None,
                    model_name=config["model_name"],
                    return_labels=return_labels,
                    tokenizer=tokenizer,
                    seed=int(config["seed"]),
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            logging.warning("no dev eval for %s: %s", dataset_dir, exc)
    dev_dataset = concat_datasets(dev_datasets)
    if dev_dataset is None:
        return None
    return make_dataloader(
        dev_dataset,
        batch_size=int(stage.get("eval_batch_size", stage["batch_size"])),
        shuffle=False,
        num_workers=int(stage.get("num_workers", 0)),
        seed=int(config["seed"]),
    )


def build_clean_dev_loader(config: dict[str, Any], stage: dict[str, Any], *, tokenizer: Any) -> Any | None:
    """Dev loader over the clean datasets' ``valid_split`` (joint-stage clean eval).

    Clean datasets carry no degradation, so this reports WER/CER on undegraded
    speech — how the joint fine-tune holds up on clean audio. ``None`` when no
    clean dataset has a usable validation split.
    """
    valid_split = config.get("valid_split")
    if not valid_split:
        return None
    dev_datasets: list[Any] = []
    for dataset_dir in clean_dataset_dirs(config):
        try:
            dev_datasets.append(
                CleanMelDataset(
                    dataset_dir,
                    split=str(valid_split),
                    model_name=config["model_name"],
                    return_labels=True,
                    tokenizer=tokenizer,
                    sample_rate=int(config.get("sample_rate", WHISPER_SAMPLE_RATE)),
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            logging.warning("no clean dev eval for %s: %s", dataset_dir, exc)
    dev_dataset = concat_datasets(dev_datasets)
    if dev_dataset is None:
        return None
    return make_dataloader(
        dev_dataset,
        batch_size=int(stage.get("eval_batch_size", stage["batch_size"])),
        shuffle=False,
        num_workers=int(stage.get("num_workers", 0)),
        seed=int(config["seed"]),
    )


def configure_whisper_generation(whisper: Any, language: str | None, task: str | None) -> None:
    """Configure Whisper generation for eval without conflicting length caps."""
    generation_config = whisper.generation_config
    # Transformers may inherit max_length from either generation_config or the
    # legacy model config. Clear both because eval uses max_new_tokens explicitly.
    generation_config.max_length = None
    if hasattr(whisper.config, "max_length"):
        whisper.config.max_length = None
    if getattr(generation_config, "lang_to_id", None):
        if language:
            generation_config.language = str(language)
        if task:
            generation_config.task = str(task)


def _tensor_debug_stats(tensor: Any) -> dict[str, Any]:
    """Small JSON-safe summary for diagnosing non-finite training batches."""
    import torch

    detached = tensor.detach()
    finite = torch.isfinite(detached)
    finite_count = int(finite.sum().item())
    total = int(detached.numel())
    stats: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype).replace("torch.", ""),
        "all_finite": finite_count == total,
        "finite_fraction": finite_count / max(1, total),
    }
    if finite_count:
        finite_values = detached[finite].float()
        stats.update(
            {
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
                "abs_max": float(finite_values.abs().max().item()),
            }
        )
    else:
        stats.update({"min": None, "max": None, "mean": None, "abs_max": None})
    return stats


def _label_lengths(labels: Any) -> list[int]:
    return (labels != -100).sum(dim=1).detach().cpu().tolist()


def _debug_step_window(stage: dict[str, Any]) -> tuple[int | None, int | None]:
    """Return the inclusive debug step window; false/null disables window logging."""
    debug_steps = stage.get("debug_steps", {"start": 5880, "end": 6020})
    if debug_steps is False or debug_steps is None:
        return None, None
    if isinstance(debug_steps, dict):
        return int(debug_steps.get("start", 5880)), int(debug_steps.get("end", 6020))
    raise ValueError("stage.debug_steps must be a mapping with start/end, false, or null")


def make_progress_bar(iterable: Any, desc: str, total: int | None = None) -> Any:
    """Wrap an iterable in a tqdm bar, auto-disabled off-TTY (``disable=None``).

    Mirrors the FastConformer trainer: interactive terminals get a live bar while
    nohup/redirected runs stay quiet, and a missing tqdm degrades to the bare
    iterable. Used for the eval loops, whose Whisper generation is slow enough to
    warrant a bar.
    """
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, total=total, unit="batch", dynamic_ncols=True, leave=False, disable=None)
    except ImportError:
        return iterable


def make_step_bar(desc: str, total: int, initial: int = 0) -> Any:
    """Manual tqdm bar over training steps (auto-disabled off-TTY via ``disable=None``).

    The step loops re-enter the dataloader across the ``while step < max_steps``
    epochs, so we drive a manual bar with ``.update(1)`` rather than wrapping an
    iterable; ``initial`` carries the resumed step count. Returns ``None`` when
    tqdm is missing so callers simply skip the updates.
    """
    try:
        from tqdm.auto import tqdm

        return tqdm(total=total, initial=initial, desc=desc, unit="step", dynamic_ncols=True, leave=False, disable=None)
    except ImportError:
        return None


def evaluate_enhancer(
    enhancer: Any,
    loader: Any,
    device: str,
    amp_enabled: bool,
    *,
    feat_encoder: Any = None,
    feat_weight: float = 0.0,
    lam: float = 1.0,
) -> dict[str, float]:
    """Mean L_enh over the dev loader (Stage 0 has no ASR objective to score).

    When ``feat_encoder`` is given (warm-up feature matching enabled), also reports
    the encoder-feature-matching loss ``L_feat`` and the combined warm-up objective
    ``L_warmup = lam*L_enh + feat_weight*L_feat`` — the latter is what best-checkpoint
    selection minimises in that mode (see :func:`eval_score`).
    """
    import torch

    was_training = enhancer.training
    enhancer.eval()
    total_enh, total_feat, count = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in make_progress_bar(loader, "stage0 eval"):
            noisy = batch["noisy_mel"].to(device)
            clean = batch["clean_mel"].to(device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                enhanced = enhancer(noisy)
                total_enh += float(enhancement_l1_loss(enhanced, clean).detach())
                if feat_encoder is not None:
                    total_feat += float(feature_match_loss(feat_encoder, enhanced, clean).detach())
            count += 1
    enhancer.train(was_training)
    n = max(1, count)
    metrics = {"L_enh": total_enh / n}
    if feat_encoder is not None:
        metrics["L_feat"] = total_feat / n
        metrics["L_warmup"] = lam * metrics["L_enh"] + feat_weight * metrics["L_feat"]
    return metrics


def evaluate_fusion(
    model: Any,
    loader: Any,
    tokenizer: Any,
    device: str,
    amp_enabled: bool,
    *,
    config: dict[str, Any],
    max_batches: int | None,
) -> dict[str, float]:
    """Dev WER/CER (generation) plus teacher-forced ASR loss for Stages 1-2."""
    import torch

    # Snapshot every submodule's train/eval flag, not just the root's: Stage 1
    # puts the frozen Whisper backbone in eval() while the enhancer/fusion stay in
    # train(), so a blanket model.train(root_flag) afterwards would wrongly flip
    # the frozen backbone back into train mode (re-enabling its dropout). Restore
    # each module to exactly the mode it had.
    prior_modes = {name: module.training for name, module in model.named_modules()}
    model.eval()
    references: list[str] = []
    hypotheses: list[str] = []
    total_loss, total_enh, count = 0.0, 0.0, 0
    gen_max_length = int(config.get("generation_max_length", 225))
    language = config.get("language")
    task = config.get("task")
    configure_whisper_generation(model.whisper, language, task)
    # Bar length is the dev loader unless ``max_batches`` caps it shorter.
    bar_total = len(loader) if hasattr(loader, "__len__") else None
    if max_batches is not None:
        bar_total = max_batches if bar_total is None else min(bar_total, max_batches)
    with torch.no_grad():
        for index, batch in enumerate(make_progress_bar(loader, "eval", total=bar_total)):
            if max_batches is not None and index >= max_batches:
                break
            noisy = batch["noisy_mel"].to(device)
            clean = batch["clean_mel"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(noisy, labels=labels)
                total_loss += float(out["loss"].detach())
                total_enh += float(enhancement_l1_loss(out["enhanced_mel"], clean).detach())
                pred_ids = model.generate(noisy, max_new_tokens=gen_max_length)
            hypotheses.extend(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
            label_ids = labels.clone()
            label_ids[label_ids == -100] = tokenizer.pad_token_id
            references.extend(tokenizer.batch_decode(label_ids, skip_special_tokens=True))
            count += 1
    for name, module in model.named_modules():
        module.train(prior_modes[name])
    n = max(1, count)
    return {
        "wer": word_error_rate(references, hypotheses),
        "cer": character_error_rate(references, hypotheses),
        "loss": total_loss / n,
        "L_enh": total_enh / n,
    }


def eval_score(stage_name: str, metrics: dict[str, float]) -> float:
    """Scalar to MINIMISE for best-checkpoint selection: WER for ASR stages, warm-up loss otherwise.

    For warm-up the score is the combined ``L_warmup`` (mel L1 + feature matching)
    when feature matching is enabled, falling back to plain ``L_enh`` otherwise.
    """
    if stage_name != "warmup":
        return float(metrics["wer"])
    return float(metrics.get("L_warmup", metrics["L_enh"]))


def build_lr_scheduler(optimizer: Any, stage: dict[str, Any], max_steps: int) -> Any:
    """Cosine-with-warmup LR schedule for a stage, or ``None`` for a flat LR.

    Per-stage keys: ``lr_scheduler`` (``"cosine"`` default, or ``"none"`` /
    ``"constant"`` to keep the optimiser's flat LR) and ``warmup_steps`` — a linear
    ramp from 0 to the peak LR (default 0), after which the cosine arm decays the
    peak LR to 0 by ``max_steps``. ``warmup_ratio`` (a fraction of ``max_steps``)
    is an alternative to an absolute ``warmup_steps``. With multiple param groups
    (Stage 2's frontend + backbone at different peak LRs) the schedule applies the
    same multiplier to every group, so each group's peak LR and their ratio are
    preserved.
    """
    kind = str(stage.get("lr_scheduler", "cosine")).lower()
    if kind in {"none", "constant", "off"}:
        return None
    if kind != "cosine":
        raise ValueError(f"unknown lr_scheduler {kind!r}; expected 'cosine' or 'none'")
    warmup_steps = stage.get("warmup_steps")
    if warmup_steps is None and stage.get("warmup_ratio") is not None:
        warmup_steps = int(float(stage["warmup_ratio"]) * max_steps)
    warmup_steps = int(warmup_steps or 0)
    from transformers import get_cosine_schedule_with_warmup

    return get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
    )


def load_resume_state(path: Path, module: Any, optimizer: Any, scaler: Any, scheduler: Any = None) -> int:
    """Restore model/optimizer/scaler/scheduler from a ``last.pt`` and return the next step.

    Returns ``0`` when the checkpoint predates mid-stage resume support (no
    ``optimizer_state``), so such a checkpoint only seeds weights, not the step.
    """
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    strict = bool(checkpoint.get("backbone_included", True))
    module.load_state_dict(checkpoint["model_state"], strict=strict)
    if checkpoint.get("optimizer_state") is None:
        return 0
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    # Adam moments load on CPU; move each onto its param's device so the next
    # step doesn't mix CPU state with CUDA grads.
    for param, state in optimizer.state.items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(param.device)
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return int(checkpoint.get("step", 0))


def run_stage_warmup(
    config: dict[str, Any],
    run_dir: Path,
    enhancer: Any,
    device: str,
) -> Path:
    """Stage 0: warm up the enhancer on L_enh only. Returns the checkpoint path."""
    import torch

    stage = config["stages"]["warmup"]
    metrics_path = run_dir / "logs" / "train_metrics.jsonl"
    eval_path = run_dir / "logs" / "eval_metrics.jsonl"
    checkpoint_dir = run_dir / "checkpoints" / STAGE_DIRS["warmup"]

    # ASR-aware warm-up: when feature_match_weight > 0, also pull the enhanced mel's
    # Whisper-encoder features toward the clean mel's. lambda then weights L_enh.
    feat_weight = float(stage.get("feature_match_weight", 0.0) or 0.0)
    lam = float(stage.get("lambda", 1.0))
    # The Whisper encoder only accepts the full [80, 3000] window, so feature
    # matching forces full-window crops (the cheap short crops can't be encoded).
    segment_seconds = stage.get("segment_seconds")
    if feat_weight > 0 and segment_seconds is not None:
        logging.info("stage0 warmup: feature matching needs full 30s windows; ignoring segment_seconds=%s", segment_seconds)
        segment_seconds = None

    train_dataset = build_train_dataset(
        config,
        segment_seconds=segment_seconds,
        return_labels=False,
        tokenizer=None,
        include_clean=False,  # warm-up trains the enhancer on degraded/clean pairs only
    )
    loader = make_dataloader(
        train_dataset,
        batch_size=int(stage["batch_size"]),
        shuffle=True,
        num_workers=int(stage.get("num_workers", 0)),
        seed=int(config["seed"]),
    )
    dev_loader = build_dev_loader(config, stage, return_labels=False)
    enhancer.to(device).train()
    optimizer = torch.optim.Adam(enhancer.parameters(), lr=float(stage["lr_enhancer"]))
    amp_enabled = use_amp(config["mixed_precision"], device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    feat_encoder = load_feature_encoder(config, device) if feat_weight > 0 else None
    if feat_encoder is not None:
        logging.info("stage0 warmup: feature matching on (weight=%s, lambda=%s)", feat_weight, lam)

    max_steps = int(stage["max_steps"])
    log_every = int(stage.get("log_every", 50))
    save_every = int(stage.get("save_every", 1000))
    eval_every = int(stage.get("eval_every", 0) or 0)
    scheduler = build_lr_scheduler(optimizer, stage, max_steps)

    start_step = 0
    resume_ckpt = checkpoint_dir / "last.pt"
    if resume_ckpt.is_file():
        start_step = load_resume_state(resume_ckpt, enhancer, optimizer, scaler, scheduler)
        logging.info("stage0 warmup: resuming from step %s (%s)", start_step, resume_ckpt)
    best_score = float("inf")
    step = start_step
    logging.info("stage0 warmup: max_steps=%s batch_size=%s amp=%s", max_steps, stage["batch_size"], amp_enabled)
    progress = make_step_bar("stage0 warmup", max_steps, initial=start_step)
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            noisy = batch["noisy_mel"].to(device)
            clean = batch["clean_mel"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                enhanced = enhancer(noisy)
                l_enh = enhancement_l1_loss(enhanced, clean)
                if feat_encoder is not None:
                    l_feat = feature_match_loss(feat_encoder, enhanced, clean)
                    loss = lam * l_enh + feat_weight * l_feat
                else:
                    loss = l_enh
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            step += 1
            if progress is not None:
                progress.update(1)
            if step % log_every == 0 or step == max_steps:
                l_enh_value = float(l_enh.detach())
                lr = optimizer.param_groups[0]["lr"]
                record = {"timestamp": utc_now(), "stage": "warmup", "step": step, "L_enh": l_enh_value, "lr": lr}
                postfix = {"L_enh": f"{l_enh_value:.4f}", "lr": f"{lr:.2e}"}
                if feat_encoder is not None:
                    record["L_feat"] = float(l_feat.detach())
                    record["loss"] = float(loss.detach())
                    postfix["L_feat"] = f"{record['L_feat']:.4f}"
                    logging.info("stage0 step=%s loss=%.4f L_enh=%.4f L_feat=%.4f lr=%.2e", step, record["loss"], l_enh_value, record["L_feat"], lr)
                else:
                    logging.info("stage0 step=%s L_enh=%.4f lr=%.2e", step, l_enh_value, lr)
                if progress is not None:
                    progress.set_postfix(**postfix)
                append_jsonl(metrics_path, record)
            if dev_loader is not None and eval_every and (step % eval_every == 0 or step == max_steps):
                metrics = evaluate_enhancer(
                    enhancer, dev_loader, device, amp_enabled,
                    feat_encoder=feat_encoder, feat_weight=feat_weight, lam=lam,
                )
                if "L_feat" in metrics:
                    logging.info("stage0 eval step=%s L_enh=%.4f L_feat=%.4f L_warmup=%.4f", step, metrics["L_enh"], metrics["L_feat"], metrics["L_warmup"])
                else:
                    logging.info("stage0 eval step=%s L_enh=%.4f", step, metrics["L_enh"])
                append_jsonl(eval_path, {"timestamp": utc_now(), "stage": "warmup", "step": step, **metrics})
                score = eval_score("warmup", metrics)
                if score < best_score:
                    best_score = score
                    save_enhancer_checkpoint(checkpoint_dir / "best.pt", enhancer, config, step)
                    logging.info("stage0 new best score=%.4f -> best.pt", score)
            if save_every and step % save_every == 0:
                save_enhancer_checkpoint(checkpoint_dir / "last.pt", enhancer, config, step, optimizer=optimizer, scaler=scaler, scheduler=scheduler)
    if progress is not None:
        progress.close()

    # Hand the *best dev* enhancer to the next stage, not the final-step weights:
    # reload best.pt (when dev eval produced one) into the in-memory enhancer so
    # both the canonical enhancer.pt and the threaded object carry the best model.
    best_path = checkpoint_dir / "best.pt"
    if best_path.is_file():
        load_enhancer_state(enhancer, best_path)
        logging.info("stage0 warmup: handing off best dev enhancer from %s", best_path)
    final_path = checkpoint_dir / "enhancer.pt"
    save_enhancer_checkpoint(final_path, enhancer, config, step)
    logging.info("stage0 warmup complete: checkpoint=%s", final_path)
    return final_path


def save_enhancer_checkpoint(
    path: Path,
    enhancer: Any,
    config: dict[str, Any],
    step: int,
    *,
    optimizer: Any = None,
    scaler: Any = None,
    scheduler: Any = None,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": enhancer.state_dict(),
            "enhancer_config": config["enhancer"],
            "step": step,
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "git_commit": git_commit(),
            "saved_at": utc_now(),
        },
        path,
    )


def load_enhancer_state(enhancer: Any, checkpoint_path: Path) -> None:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    enhancer.load_state_dict(checkpoint["model_state"])


def load_feature_encoder(config: dict[str, Any], device: str) -> Any:
    """Load the frozen Whisper *encoder* for the warm-up feature-matching loss.

    The enhancer is trained to make ``encoder(enhanced_mel)`` match
    ``encoder(clean_mel)`` — an ASR-aware perceptual target far more correlated
    with WER than raw mel L1. The encoder comes from the same fine-tuned backbone
    Stages 1-2 use (``base_asr_checkpoint``), is frozen, and put in eval() so its
    dropout/BN are inert; only the enhancer receives gradients through it.
    """
    from ml.fusion.model import load_whisper_backbone

    whisper = load_whisper_backbone(
        str(config.get("base_asr_checkpoint") or ""),
        model_name=str(config.get("model_name", "openai/whisper-small")),
    )
    encoder = whisper.get_encoder()
    for param in encoder.parameters():
        param.requires_grad_(False)
    encoder.eval().to(device)
    return encoder


def feature_match_loss(encoder: Any, enhanced_mel: Any, clean_mel: Any) -> Any:
    """L1 between the Whisper-encoder features of the enhanced vs clean log-Mel.

    The clean-side features are detached (no_grad): they are a fixed target, so
    only the enhancer is pulled toward producing encoder-clean features.
    """
    import torch
    from torch.nn import functional as F

    enhanced_features = encoder(enhanced_mel).last_hidden_state
    with torch.no_grad():
        clean_features = encoder(clean_mel).last_hidden_state
    return F.l1_loss(enhanced_features, clean_features)


def load_tokenizer(config: dict[str, Any]) -> Any:
    """Load the fine-tuned checkpoint's Whisper tokenizer (Persian lang/task prefix).

    Uses the ``base_asr_checkpoint`` tokenizer so labels carry the exact
    language/task prefix the backbone was trained with (a local run dir or a Hub
    id; see ``require_base_checkpoint``). Factored out as a seam so tests can
    inject a lightweight stub.
    """
    from transformers import WhisperTokenizer

    checkpoint = str(config.get("base_asr_checkpoint") or "")
    if not is_loadable_checkpoint(checkpoint):
        raise FileNotFoundError(
            f"base_asr_checkpoint does not exist: {checkpoint!r}; cannot load tokenizer."
        )
    return WhisperTokenizer.from_pretrained(checkpoint)


def save_fusion_checkpoint(
    path: Path,
    model: Any,
    config: dict[str, Any],
    step: int,
    *,
    optimizer: Any = None,
    scaler: Any = None,
    scheduler: Any = None,
    include_backbone: bool = True,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    if not include_backbone:
        # Stage 1 freezes Whisper, so its weights equal base_asr_checkpoint and
        # need not be re-saved on every rolling checkpoint — keep only the parts
        # that actually change (enhancer + fusion).
        state = {k: v for k, v in state.items() if not k.startswith("whisper.")}
    torch.save(
        {
            "model_state": state,
            "backbone_included": include_backbone,
            "enhancer_config": config["enhancer"],
            "fusion_config": config.get("fusion"),
            "step": step,
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "git_commit": git_commit(),
            "saved_at": utc_now(),
        },
        path,
    )


def load_fusion_checkpoint(model: Any, path: Path) -> None:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    # Backbone-free checkpoints (Stage 1 rolling saves) load non-strictly: the
    # frozen Whisper weights are already in place from base_asr_checkpoint.
    strict = bool(checkpoint.get("backbone_included", True))
    model.load_state_dict(checkpoint["model_state"], strict=strict)


def _run_fusion_stage(
    config: dict[str, Any],
    run_dir: Path,
    enhancer: Any,
    device: str,
    *,
    stage_name: str,
    train_backbone: bool,
) -> Path:
    """Shared loop for Stages 1-2: optimise ``L_ASR + lambda * L_enh``.

    Stage 1 (``train_backbone=False``) trains the enhancer + fusion block with the
    Whisper backbone frozen; Stage 2 (``train_backbone=True``) unfreezes the
    backbone and adds it to the optimiser at its own learning rate. The fusion
    block and backbone state from the preceding fusion stage are loaded from disk
    so weights carry forward both within one run and across resume.
    """
    import torch

    stage = config["stages"][stage_name]
    metrics_path = run_dir / "logs" / "train_metrics.jsonl"
    eval_path = run_dir / "logs" / "eval_metrics.jsonl"
    debug_path = run_dir / "logs" / "fusion_debug_metrics.jsonl"
    checkpoint_dir = run_dir / "checkpoints" / STAGE_DIRS[stage_name]

    tokenizer = load_tokenizer(config)
    # Stage 2 (joint) folds in any clean ASR datasets so the end-to-end fine-tune
    # also sees undegraded speech; Stage 1 stays degraded-only.
    train_dataset = build_train_dataset(
        config,
        segment_seconds=None,  # full [80, 3000] window — the fused result feeds Whisper
        return_labels=True,
        tokenizer=tokenizer,
        include_clean=(stage_name == "joint"),
    )
    loader = make_dataloader(
        train_dataset,
        batch_size=int(stage["batch_size"]),
        shuffle=True,
        num_workers=int(stage.get("num_workers", 0)),
        seed=int(config["seed"]),
    )
    dev_loader = build_dev_loader(config, stage, return_labels=True, tokenizer=tokenizer)
    # The joint (final) stage also reports dev metrics on the clean datasets, so
    # clean-speech regression is visible alongside the degraded WER it selects on.
    clean_dev_loader = (
        build_clean_dev_loader(config, stage, tokenizer=tokenizer) if stage_name == "joint" else None
    )

    model = build_fusion_model(config, enhancer=enhancer)
    prior_ckpt = run_dir / "checkpoints" / STAGE_DIRS["fusion"] / "fusion_model.pt"
    if stage_name == "joint" and prior_ckpt.is_file():
        logging.info("stage2 joint: loading fusion model from %s", prior_ckpt)
        load_fusion_checkpoint(model, prior_ckpt)
    model.to(device)

    if train_backbone:
        model.unfreeze_backbone()
    else:
        model.freeze_backbone()
    model.enhancer.train()
    model.fusion.train()

    frontend_params = list(model.enhancer.parameters()) + list(model.fusion.parameters())
    param_groups = [{"params": frontend_params, "lr": float(stage["lr_frontend"])}]
    if train_backbone:
        param_groups.append({"params": list(model.whisper.parameters()), "lr": float(stage.get("lr_whisper", stage["lr_frontend"]))})
    optimizer = torch.optim.Adam(param_groups)

    amp_enabled = use_amp(config["mixed_precision"], device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    lam = float(stage.get("lambda", 0.3))
    max_steps = int(stage["max_steps"])
    log_every = int(stage.get("log_every", 50))
    save_every = int(stage.get("save_every", 1000))
    eval_every = int(stage.get("eval_every", 0) or 0)
    eval_max_batches = stage.get("eval_max_batches")
    eval_max_batches = int(eval_max_batches) if eval_max_batches is not None else None
    grad_clip = float(stage.get("grad_clip", 1.0))
    debug_start, debug_end = _debug_step_window(stage)
    scheduler = build_lr_scheduler(optimizer, stage, max_steps)
    # Stage 1 freezes the backbone, so rolling last.pt can drop it (it equals the
    # base checkpoint); Stage 2 trains it and must keep it.
    include_backbone = train_backbone

    start_step = 0
    resume_ckpt = checkpoint_dir / "last.pt"
    if resume_ckpt.is_file():
        start_step = load_resume_state(resume_ckpt, model, optimizer, scaler, scheduler)
        logging.info("%s: resuming from step %s (%s)", stage_name, start_step, resume_ckpt)
    best_score = float("inf")
    step = start_step
    logging.info(
        "%s: max_steps=%s batch_size=%s lambda=%s train_backbone=%s amp=%s",
        stage_name, max_steps, stage["batch_size"], lam, train_backbone, amp_enabled,
    )
    progress = make_step_bar(stage_name, max_steps, initial=start_step)
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            noisy = batch["noisy_mel"].to(device)
            clean = batch["clean_mel"].to(device)
            labels = batch["labels"].to(device)
            next_step = step + 1
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(noisy, labels=labels)
                l_asr = out["loss"]
                l_enh = enhancement_l1_loss(out["enhanced_mel"], clean)
                loss = l_asr + lam * l_enh
            debug_in_window = (
                debug_start is not None
                and debug_end is not None
                and debug_start <= next_step <= debug_end
            )
            scalar_nonfinite = not all(
                bool(torch.isfinite(value.detach()).all().item())
                for value in (loss, l_asr, l_enh)
            )
            if debug_in_window or scalar_nonfinite:
                tensors = {
                    "loss": loss,
                    "L_ASR": l_asr,
                    "L_enh": l_enh,
                    "noisy_mel": noisy,
                    "clean_mel": clean,
                    "enhanced_mel": out["enhanced_mel"],
                    "encoder_hidden_states": out["encoder_hidden_states"],
                    "logits": out["logits"],
                }
                tensor_stats = {name: _tensor_debug_stats(value) for name, value in tensors.items()}
                has_nonfinite = any(not stats["all_finite"] for stats in tensor_stats.values())
                valid_labels = labels[labels != -100]
                label_stats = {
                    "shape": list(labels.shape),
                    "lengths": _label_lengths(labels),
                    "min": int(valid_labels.min().item()) if int(valid_labels.numel()) else None,
                    "max": int(valid_labels.max().item()) if int(valid_labels.numel()) else None,
                }
                append_jsonl(
                    debug_path,
                    {
                        "timestamp": utc_now(),
                        "stage": stage_name,
                        "step": next_step,
                        "pair_id": batch.get("pair_id", []),
                        "label_stats": label_stats,
                        "tensor_stats": tensor_stats,
                        "has_nonfinite": has_nonfinite,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                )
                if has_nonfinite:
                    logging.warning(
                        "%s step=%s non-finite debug batch pair_id=%s -> %s",
                        stage_name,
                        next_step,
                        batch.get("pair_id", []),
                        debug_path,
                    )
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_((p for g in param_groups for p in g["params"]), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            step += 1
            if progress is not None:
                progress.update(1)
            if step % log_every == 0 or step == max_steps:
                lr = optimizer.param_groups[0]["lr"]
                logging.info("%s step=%s loss=%.4f L_ASR=%.4f L_enh=%.4f lr=%.2e", stage_name, step, float(loss.detach()), float(l_asr.detach()), float(l_enh.detach()), lr)
                if progress is not None:
                    progress.set_postfix(loss=f"{float(loss.detach()):.4f}", L_ASR=f"{float(l_asr.detach()):.4f}", L_enh=f"{float(l_enh.detach()):.4f}", lr=f"{lr:.2e}")
                append_jsonl(metrics_path, {
                    "timestamp": utc_now(), "stage": stage_name, "step": step,
                    "loss": float(loss.detach()), "L_ASR": float(l_asr.detach()), "L_enh": float(l_enh.detach()), "lr": lr,
                })
            should_eval = bool(eval_every) and (step % eval_every == 0 or step == max_steps)
            if should_eval and (dev_loader is not None or clean_dev_loader is not None):
                record = {"timestamp": utc_now(), "stage": stage_name, "step": step}
                degraded_metrics = None
                if dev_loader is not None:
                    degraded_metrics = evaluate_fusion(
                        model, dev_loader, tokenizer, device, amp_enabled,
                        config=config, max_batches=eval_max_batches,
                    )
                    record.update(degraded_metrics)
                    logging.info("%s eval step=%s wer=%.4f cer=%.4f loss=%.4f", stage_name, step, degraded_metrics["wer"], degraded_metrics["cer"], degraded_metrics["loss"])
                clean_metrics = None
                if clean_dev_loader is not None:
                    clean_metrics = evaluate_fusion(
                        model, clean_dev_loader, tokenizer, device, amp_enabled,
                        config=config, max_batches=eval_max_batches,
                    )
                    record.update({f"clean_{key}": value for key, value in clean_metrics.items()})
                    logging.info("%s clean eval step=%s wer=%.4f cer=%.4f loss=%.4f", stage_name, step, clean_metrics["wer"], clean_metrics["cer"], clean_metrics["loss"])
                append_jsonl(eval_path, record)
                # Select on degraded WER (the metric to beat); fall back to clean
                # only when no degraded dev split exists.
                score = eval_score(stage_name, degraded_metrics if degraded_metrics is not None else clean_metrics)
                if score < best_score:
                    best_score = score
                    save_fusion_checkpoint(checkpoint_dir / "best.pt", model, config, step)
                    logging.info("%s new best wer=%.4f -> best.pt", stage_name, score)
            if save_every and step % save_every == 0:
                save_fusion_checkpoint(
                    checkpoint_dir / "last.pt", model, config, step,
                    optimizer=optimizer, scaler=scaler, scheduler=scheduler, include_backbone=include_backbone,
                )
    if progress is not None:
        progress.close()

    # Hand the *best dev* model to the next stage / resume scaffold, not the
    # final-step weights: reload best.pt (when dev eval produced one) so the
    # canonical fusion_model.pt + enhancer.pt and the threaded enhancer object all
    # carry the best-WER model.
    best_path = checkpoint_dir / "best.pt"
    if best_path.is_file():
        load_fusion_checkpoint(model, best_path)
        logging.info("%s: handing off best dev model from %s", stage_name, best_path)
    save_fusion_checkpoint(checkpoint_dir / "fusion_model.pt", model, config, step)
    # Mirror the enhancer state under the name the resume scaffold expects.
    save_enhancer_checkpoint(checkpoint_dir / "enhancer.pt", model.enhancer, config, step)
    logging.info("%s complete: checkpoint=%s", stage_name, checkpoint_dir / "fusion_model.pt")
    return checkpoint_dir / "fusion_model.pt"


def run_stage_fusion(config: dict[str, Any], run_dir: Path, enhancer: Any, device: str) -> Path:
    """Stage 1: train enhancer + fusion with the Whisper backbone frozen."""
    return _run_fusion_stage(config, run_dir, enhancer, device, stage_name="fusion", train_backbone=False)


def run_stage_joint(config: dict[str, Any], run_dir: Path, enhancer: Any, device: str) -> Path:
    """Stage 2: train enhancer + fusion + Whisper backbone end to end."""
    return _run_fusion_stage(config, run_dir, enhancer, device, stage_name="joint", train_backbone=True)


STAGE_RUNNERS = {
    "warmup": run_stage_warmup,
    "fusion": run_stage_fusion,
    "joint": run_stage_joint,
}


def run_training(config_path: Path, run_dir_override: Path | None = None, resume_from_stage: Any = "__unset__") -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    config = load_fusion_config(config_path)
    if resume_from_stage != "__unset__":
        config["resume_from_stage"] = resume_from_stage
    require_base_checkpoint(config)
    seed_everything(int(config["seed"]))

    run_dir = run_dir_override or Path(str(config["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    (run_dir / "config" / "training_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    write_json(run_dir / "config" / "manifest_hashes.json", manifest_hashes(degraded_dataset_dirs(config)))
    (run_dir / "config" / "git_commit.txt").write_text((git_commit() or "unknown") + "\n", encoding="utf-8")

    device = resolve_device(str(config["device"]))
    logging.info("run_dir=%s device=%s", run_dir, device)
    enhancer = build_enhancer(config["enhancer"])

    start_index = resolve_start_index(config["resume_from_stage"])
    if start_index > 0:
        prior_dir = STAGE_DIRS[STAGE_ORDER[start_index - 1]]
        prior_ckpt = run_dir / "checkpoints" / prior_dir / "enhancer.pt"
        if not prior_ckpt.is_file():
            raise FileNotFoundError(
                f"resume_from_stage={config['resume_from_stage']} needs a prior checkpoint at {prior_ckpt}"
            )
        logging.info("resuming: loading enhancer init from %s", prior_ckpt)
        load_enhancer_state(enhancer, prior_ckpt)

    for stage_name in STAGE_ORDER[start_index:]:
        logging.info("=== running stage: %s ===", stage_name)
        STAGE_RUNNERS[stage_name](config, run_dir, enhancer, device)
    logging.info("curriculum complete")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 3-stage enhancement+fusion curriculum (Stage 0 warm-up -> Stage 1 "
            "fusion -> Stage 2 joint) from one YAML config, writing all artifacts to one "
            "run directory. Consumes a degraded dataset from generate_degraded_dataset."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML fusion training config path.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Optional run directory override.")
    parser.add_argument(
        "--resume-from-stage",
        default="__unset__",
        help="Resume the curriculum at a stage: 0/1/2 or warmup/fusion/joint. Overrides config.",
    )
    args = parser.parse_args(argv)
    resume = args.resume_from_stage
    if resume not in {"__unset__", None} and str(resume).isdigit():
        resume = int(resume)
    return run_training(args.config, args.run_dir, resume)


if __name__ == "__main__":
    raise SystemExit(main())
