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
end to end. State is handed from one stage to the next through the run's
per-stage checkpoints, so the curriculum also resumes correctly mid-way.
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
    "mixed_precision": "auto",
    "device": "auto",
    "seed": 1337,
    "resume_from_stage": None,
    "enhancer": {"type": "residual_unet", "base_channels": 32, "depth": 3},
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
        },
        "fusion": {"max_steps": 20000, "batch_size": 8, "lr_frontend": 2e-4, "lambda": 0.3},
        "joint": {
            "max_steps": 40000,
            "batch_size": 4,
            "lr_frontend": 1e-4,
            "lr_whisper": 1e-5,
            "lambda": 0.1,
            "whisper_adaptation": "full",
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


def run_stage_warmup(
    config: dict[str, Any],
    run_dir: Path,
    enhancer: Any,
    device: str,
) -> Path:
    """Stage 0: warm up the enhancer on L_enh only. Returns the checkpoint path."""
    import torch
    from torch.utils.data import DataLoader

    stage = config["stages"]["warmup"]
    metrics_path = run_dir / "logs" / "train_metrics.jsonl"
    checkpoint_dir = run_dir / "checkpoints" / STAGE_DIRS["warmup"]

    train_dataset = DegradedMelDataset(
        config["dataset_dir"],
        split=config["train_split"],
        clean_target=config["clean_target"],
        segment_seconds=stage.get("segment_seconds"),
        model_name=config["model_name"],
        seed=int(config["seed"]),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(stage["batch_size"]),
        shuffle=True,
        num_workers=int(stage.get("num_workers", 0)),
        collate_fn=collate_mels,
        drop_last=False,
    )
    enhancer.to(device).train()
    optimizer = torch.optim.Adam(enhancer.parameters(), lr=float(stage["lr_enhancer"]))
    amp_enabled = use_amp(config["mixed_precision"], device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    max_steps = int(stage["max_steps"])
    log_every = int(stage.get("log_every", 50))
    save_every = int(stage.get("save_every", 1000))
    step = 0
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
            if save_every and step % save_every == 0:
                save_enhancer_checkpoint(checkpoint_dir / "last.pt", enhancer, config, step)

    final_path = checkpoint_dir / "enhancer.pt"
    save_enhancer_checkpoint(final_path, enhancer, config, step)
    logging.info("stage0 warmup complete: checkpoint=%s", final_path)
    return final_path


def save_enhancer_checkpoint(path: Path, enhancer: Any, config: dict[str, Any], step: int) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": enhancer.state_dict(),
            "enhancer_config": config["enhancer"],
            "step": step,
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


def save_fusion_checkpoint(path: Path, model: Any, config: dict[str, Any], step: int) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "enhancer_config": config["enhancer"],
            "fusion_config": config.get("fusion"),
            "step": step,
            "git_commit": git_commit(),
            "saved_at": utc_now(),
        },
        path,
    )


def load_fusion_checkpoint(model: Any, path: Path) -> None:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])


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
    from torch.utils.data import DataLoader

    stage = config["stages"][stage_name]
    metrics_path = run_dir / "logs" / "train_metrics.jsonl"
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
    loader = DataLoader(
        train_dataset,
        batch_size=int(stage["batch_size"]),
        shuffle=True,
        num_workers=int(stage.get("num_workers", 0)),
        collate_fn=collate_mels,
        drop_last=False,
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
    step = 0
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
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step % log_every == 0 or step == max_steps:
                logging.info("%s step=%s loss=%.4f L_ASR=%.4f L_enh=%.4f", stage_name, step, float(loss.detach()), float(l_asr.detach()), float(l_enh.detach()))
                append_jsonl(metrics_path, {
                    "timestamp": utc_now(), "stage": stage_name, "step": step,
                    "loss": float(loss.detach()), "L_ASR": float(l_asr.detach()), "L_enh": float(l_enh.detach()),
                })
            if save_every and step % save_every == 0:
                save_fusion_checkpoint(checkpoint_dir / "last.pt", model, config, step)

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
    import torch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    config = load_fusion_config(config_path)
    if resume_from_stage != "__unset__":
        config["resume_from_stage"] = resume_from_stage
    torch.manual_seed(int(config["seed"]))

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
