"""Single config-driven orchestrator for the 3-stage enhancement+fusion curriculum.

One invocation runs the whole curriculum (D8/D10) and writes every artifact to one
run directory:

| Stage | Key   | Trains             | Loss                      |
|-------|-------|--------------------|---------------------------|
| 0     | warmup| enhancer E         | L_enh                     |
| 1     | fusion| E + fusion (Whisper frozen) | L_ASR + lambda*L_enh |
| 2     | joint | E + fusion + Whisper        | L_ASR + lambda*L_enh |

It consumes a degraded-dataset directory produced by
``ml.speech_data.generate_degraded_dataset`` (see ``ml.enhancement.dataset``).

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
state and the step, so a stage resumes mid-way on the next invocation; the final
``enhancer.pt`` / ``fusion_model.pt`` also hand state to the next stage and the
stage-boundary ``resume_from_stage`` path. Seeding goes through
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
from ml.enhancement.dataset import DegradedMelDataset, collate_mels
from ml.enhancement.enhancer import build_enhancer, enhancement_l1_loss
from ml.fusion.model import build_fusion_model

STAGE_ORDER = ["warmup", "fusion", "joint"]
STAGE_DIRS = {"warmup": "stage0_warmup", "fusion": "stage1_fusion", "joint": "stage2_joint"}

DEFAULT_CONFIG: dict[str, Any] = {
    "run_dir": "artifacts/speech_enhancement/fusion/run_001",
    "base_asr_checkpoint": None,
    "dataset_dir": "data/cv-corpus-25.0-degraded",
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
            "lambda": 1.0,
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
            "lambda": 0.1,
            "whisper_adaptation": "full",
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
    if not str(config.get("dataset_dir", "")).strip():
        raise ValueError("dataset_dir must be set to a degraded-dataset directory")
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


def manifest_hashes(dataset_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    mapping = dataset_dir / "degraded_to_clean.jsonl"
    if mapping.is_file():
        digest = hashlib.sha256()
        with mapping.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        hashes[mapping.name] = digest.hexdigest()
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
    )


def build_dev_loader(
    config: dict[str, Any],
    stage: dict[str, Any],
    *,
    return_labels: bool,
    tokenizer: Any = None,
) -> Any | None:
    """Build a dev-split loader, or ``None`` when no usable validation split exists.

    Returning ``None`` (rather than raising) lets a run proceed without a ``dev``
    split — periodic eval and best-checkpoint selection are simply skipped, which
    keeps small/smoke datasets usable.
    """
    valid_split = config.get("valid_split")
    if not valid_split:
        return None
    try:
        dev_dataset = DegradedMelDataset(
            config["dataset_dir"],
            split=str(valid_split),
            clean_target=config["clean_target"],
            segment_seconds=None,
            model_name=config["model_name"],
            return_labels=return_labels,
            tokenizer=tokenizer,
            seed=int(config["seed"]),
        )
    except (FileNotFoundError, ValueError) as exc:
        logging.warning("no dev eval: %s", exc)
        return None
    return make_dataloader(
        dev_dataset,
        batch_size=int(stage.get("eval_batch_size", stage["batch_size"])),
        shuffle=False,
        num_workers=int(stage.get("num_workers", 0)),
        seed=int(config["seed"]),
    )


def evaluate_enhancer(enhancer: Any, loader: Any, device: str, amp_enabled: bool) -> dict[str, float]:
    """Mean L_enh over the dev loader (Stage 0 has no ASR objective to score)."""
    import torch

    was_training = enhancer.training
    enhancer.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            noisy = batch["noisy_mel"].to(device)
            clean = batch["clean_mel"].to(device)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss = enhancement_l1_loss(enhancer(noisy), clean)
            total += float(loss.detach())
            count += 1
    enhancer.train(was_training)
    return {"L_enh": total / max(1, count)}


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

    was_training = model.training
    model.eval()
    references: list[str] = []
    hypotheses: list[str] = []
    total_loss, total_enh, count = 0.0, 0.0, 0
    gen_max_length = int(config.get("generation_max_length", 225))
    language = config.get("language")
    task = config.get("task")
    # Only steer the decoder prompt on a genuinely multilingual Whisper (the
    # fine-tuned backbone). Models without the language token map — e.g. tiny
    # English-only/test configs — decode from decoder_start_token_id alone.
    generation_config = model.whisper.generation_config
    if getattr(generation_config, "lang_to_id", None):
        if language:
            generation_config.language = str(language)
        if task:
            generation_config.task = str(task)
    with torch.no_grad():
        for index, batch in enumerate(loader):
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
    model.train(was_training)
    n = max(1, count)
    return {
        "wer": word_error_rate(references, hypotheses),
        "cer": character_error_rate(references, hypotheses),
        "loss": total_loss / n,
        "L_enh": total_enh / n,
    }


def eval_score(stage_name: str, metrics: dict[str, float]) -> float:
    """Scalar to MINIMISE for best-checkpoint selection: WER for ASR stages, L_enh for warm-up."""
    return float(metrics["L_enh"] if stage_name == "warmup" else metrics["wer"])


def load_resume_state(path: Path, module: Any, optimizer: Any, scaler: Any) -> int:
    """Restore model/optimizer/scaler from a ``last.pt`` and return the next step.

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

    train_dataset = DegradedMelDataset(
        config["dataset_dir"],
        split=config["train_split"],
        clean_target=config["clean_target"],
        segment_seconds=stage.get("segment_seconds"),
        model_name=config["model_name"],
        seed=int(config["seed"]),
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

    max_steps = int(stage["max_steps"])
    log_every = int(stage.get("log_every", 50))
    save_every = int(stage.get("save_every", 1000))
    eval_every = int(stage.get("eval_every", 0) or 0)

    start_step = 0
    resume_ckpt = checkpoint_dir / "last.pt"
    if resume_ckpt.is_file():
        start_step = load_resume_state(resume_ckpt, enhancer, optimizer, scaler)
        logging.info("stage0 warmup: resuming from step %s (%s)", start_step, resume_ckpt)
    best_score = float("inf")
    step = start_step
    logging.info("stage0 warmup: max_steps=%s batch_size=%s amp=%s", max_steps, stage["batch_size"], amp_enabled)
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            noisy = batch["noisy_mel"].to(device)
            clean = batch["clean_mel"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                enhanced = enhancer(noisy)
                loss = enhancement_l1_loss(enhanced, clean)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step % log_every == 0 or step == max_steps:
                loss_value = float(loss.detach())
                logging.info("stage0 step=%s L_enh=%.4f", step, loss_value)
                append_jsonl(metrics_path, {"timestamp": utc_now(), "stage": "warmup", "step": step, "L_enh": loss_value})
            if dev_loader is not None and eval_every and (step % eval_every == 0 or step == max_steps):
                metrics = evaluate_enhancer(enhancer, dev_loader, device, amp_enabled)
                logging.info("stage0 eval step=%s L_enh=%.4f", step, metrics["L_enh"])
                append_jsonl(eval_path, {"timestamp": utc_now(), "stage": "warmup", "step": step, **metrics})
                score = eval_score("warmup", metrics)
                if score < best_score:
                    best_score = score
                    save_enhancer_checkpoint(checkpoint_dir / "best.pt", enhancer, config, step)
                    logging.info("stage0 new best L_enh=%.4f -> best.pt", score)
            if save_every and step % save_every == 0:
                save_enhancer_checkpoint(checkpoint_dir / "last.pt", enhancer, config, step, optimizer=optimizer, scaler=scaler)

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
            "git_commit": git_commit(),
            "saved_at": utc_now(),
        },
        path,
    )


def load_enhancer_state(enhancer: Any, checkpoint_path: Path) -> None:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    enhancer.load_state_dict(checkpoint["model_state"])


def load_tokenizer(config: dict[str, Any]) -> Any:
    """Load the Whisper tokenizer used to turn transcripts into label ids.

    Prefers the fine-tuned checkpoint's tokenizer (it carries the Persian
    language/task prefix the backbone was trained with) and falls back to the
    base model. Factored out as a seam so tests can inject a lightweight stub.
    """
    from transformers import WhisperTokenizer

    checkpoint = str(config.get("base_asr_checkpoint") or "")
    source = checkpoint if (checkpoint and Path(checkpoint).exists()) else str(config.get("model_name", "openai/whisper-small"))
    return WhisperTokenizer.from_pretrained(source)


def save_fusion_checkpoint(
    path: Path,
    model: Any,
    config: dict[str, Any],
    step: int,
    *,
    optimizer: Any = None,
    scaler: Any = None,
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
    checkpoint_dir = run_dir / "checkpoints" / STAGE_DIRS[stage_name]

    tokenizer = load_tokenizer(config)
    train_dataset = DegradedMelDataset(
        config["dataset_dir"],
        split=config["train_split"],
        clean_target=config["clean_target"],
        segment_seconds=None,  # full [80, 3000] window — the fused result feeds Whisper
        model_name=config["model_name"],
        return_labels=True,
        tokenizer=tokenizer,
        seed=int(config["seed"]),
    )
    loader = make_dataloader(
        train_dataset,
        batch_size=int(stage["batch_size"]),
        shuffle=True,
        num_workers=int(stage.get("num_workers", 0)),
        seed=int(config["seed"]),
    )
    dev_loader = build_dev_loader(config, stage, return_labels=True, tokenizer=tokenizer)

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
    # Stage 1 freezes the backbone, so rolling last.pt can drop it (it equals the
    # base checkpoint); Stage 2 trains it and must keep it.
    include_backbone = train_backbone

    start_step = 0
    resume_ckpt = checkpoint_dir / "last.pt"
    if resume_ckpt.is_file():
        start_step = load_resume_state(resume_ckpt, model, optimizer, scaler)
        logging.info("%s: resuming from step %s (%s)", stage_name, start_step, resume_ckpt)
    best_score = float("inf")
    step = start_step
    logging.info(
        "%s: max_steps=%s batch_size=%s lambda=%s train_backbone=%s amp=%s",
        stage_name, max_steps, stage["batch_size"], lam, train_backbone, amp_enabled,
    )
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            noisy = batch["noisy_mel"].to(device)
            clean = batch["clean_mel"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(noisy, labels=labels)
                l_asr = out["loss"]
                l_enh = enhancement_l1_loss(out["enhanced_mel"], clean)
                loss = l_asr + lam * l_enh
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_((p for g in param_groups for p in g["params"]), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step % log_every == 0 or step == max_steps:
                logging.info("%s step=%s loss=%.4f L_ASR=%.4f L_enh=%.4f", stage_name, step, float(loss.detach()), float(l_asr.detach()), float(l_enh.detach()))
                append_jsonl(metrics_path, {
                    "timestamp": utc_now(), "stage": stage_name, "step": step,
                    "loss": float(loss.detach()), "L_ASR": float(l_asr.detach()), "L_enh": float(l_enh.detach()),
                })
            if dev_loader is not None and eval_every and (step % eval_every == 0 or step == max_steps):
                metrics = evaluate_fusion(
                    model, dev_loader, tokenizer, device, amp_enabled,
                    config=config, max_batches=eval_max_batches,
                )
                logging.info("%s eval step=%s wer=%.4f cer=%.4f loss=%.4f", stage_name, step, metrics["wer"], metrics["cer"], metrics["loss"])
                append_jsonl(eval_path, {"timestamp": utc_now(), "stage": stage_name, "step": step, **metrics})
                score = eval_score(stage_name, metrics)
                if score < best_score:
                    best_score = score
                    save_fusion_checkpoint(checkpoint_dir / "best.pt", model, config, step)
                    logging.info("%s new best wer=%.4f -> best.pt", stage_name, score)
            if save_every and step % save_every == 0:
                save_fusion_checkpoint(
                    checkpoint_dir / "last.pt", model, config, step,
                    optimizer=optimizer, scaler=scaler, include_backbone=include_backbone,
                )

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
    seed_everything(int(config["seed"]))

    run_dir = run_dir_override or Path(str(config["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    (run_dir / "config" / "training_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    write_json(run_dir / "config" / "manifest_hashes.json", manifest_hashes(Path(str(config["dataset_dir"]))))
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
