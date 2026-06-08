"""Fine-tune the standalone FastConformer-CTC (Persian) model on dataset splits.

Mirrors ``ml.asr.train_whisper_small``: it reads a YAML config, loads the
configured dataset ``train.tsv`` / ``dev.tsv`` files, and fine-tunes the CTC
branch of ``nvidia/stt_fa_fastconformer_hybrid_large`` (reimplemented under
``ml/fa_fastconformer/`` with no NeMo dependency). Because the standalone model
is a plain ``nn.Module`` rather than a Hugging Face model, training runs through
a small hand-written PyTorch loop (CTC loss, AdamW, linear warmup schedule,
gradient accumulation, optional AMP) instead of ``transformers.Trainer``.

``model.checkpoint`` may point at either the original ``.nemo`` archive or a
converted ``.pt`` bundle (see ``ml/fa_fastconformer/convert.py``); the format is
chosen from the file extension. Checkpoints, the ``final``/``best`` models, and
resume work all use the same self-contained ``.pt`` bundle layout that
``ml.asr.eval_fastconformer`` loads via ``FastConformerCTC.from_pretrained``, so
a trained checkpoint can be evaluated directly.

Run layout matches the Whisper trainer: ``status.json``, ``logs/train.log``,
``logs/train_metrics.jsonl``, the effective config under ``config/``, source
manifests under ``manifests/``, rolling ``checkpoints/checkpoint-<step>.pt``
bundles, and ``final.pt`` / ``best.pt``.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from ml.asr.train_whisper_small import (
    WhisperExample,
    append_jsonl,
    character_error_rate,
    configure_logging,
    deep_merge,
    load_split_examples,
    resolve_dataset_dirs,
    run_id,
    update_status,
    utc_now,
    word_error_rate,
    write_examples_manifest,
    write_json,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        # Either the original .nemo archive or a converted .pt bundle to start
        # from (see ml/fa_fastconformer/convert.py). Format chosen by extension.
        "checkpoint": "models/stt_fa_fastconformer_ctc.pt",
    },
    "data": {
        "root_dir": "data",
        "datasets": ["cv-corpus-25.0"],
        "sample_rate": 16000,
    },
    "run": {
        "output_dir": "models/asr/fastconformer/runs",
        "name": None,
        "resume": "auto",
    },
    "training": {
        "seed": 1337,
        "num_train_epochs": 3,
        "learning_rate": 1e-4,
        "warmup_steps": 500,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "eval_steps": 500,
        "save_steps": 500,
        "logging_steps": 25,
        "save_total_limit": 3,
        "num_workers": 2,
        "device": "auto",
        "mixed_precision": "auto",
        "freeze_encoder": False,
        # Duration-aware eval batching cap (seconds); see model.transcribe.
        "eval_max_batch_seconds": 30,
    },
}

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)\.pt$")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_training_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    config = deep_merge(DEFAULT_CONFIG, loaded)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    model = config["model"]
    data = config["data"]
    training = config["training"]
    if not str(model.get("checkpoint") or "").strip():
        raise ValueError("model.checkpoint must be a non-empty .nemo or .pt path")
    datasets = data.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("data.datasets must be a non-empty list of dataset directory names")
    if any(not str(dataset).strip() for dataset in datasets):
        raise ValueError("data.datasets cannot contain empty values")
    minimums = {
        "sample_rate": (data, 8000),
        "num_train_epochs": (training, 0.01),
        "learning_rate": (training, 1e-8),
        "per_device_train_batch_size": (training, 1),
        "per_device_eval_batch_size": (training, 1),
        "gradient_accumulation_steps": (training, 1),
        "eval_steps": (training, 1),
        "save_steps": (training, 1),
        "logging_steps": (training, 1),
        "save_total_limit": (training, 1),
        "num_workers": (training, 0),
    }
    for key, (section, minimum) in minimums.items():
        if float(section[key]) < minimum:
            raise ValueError(f"{key} must be >= {minimum:g}")
    if float(training["warmup_steps"]) < 0:
        raise ValueError("warmup_steps must be >= 0")
    if float(training["weight_decay"]) < 0:
        raise ValueError("weight_decay must be >= 0")
    if float(training["max_grad_norm"]) <= 0:
        raise ValueError("max_grad_norm must be > 0")
    if training.get("eval_max_batch_seconds") is not None and float(training["eval_max_batch_seconds"]) < 1:
        raise ValueError("eval_max_batch_seconds must be >= 1 when set")
    if training["mixed_precision"] not in {"auto", "true", "false", True, False}:
        raise ValueError("training.mixed_precision must be auto, true, or false")
    if training["device"] not in {"auto", "cuda", "cpu"}:
        raise ValueError("training.device must be auto, cuda, or cpu")
    if not isinstance(training["freeze_encoder"], bool):
        raise ValueError("training.freeze_encoder must be true or false")


def resolve_run_dir(config: dict[str, Any], override: Path | None = None) -> Path:
    if override is not None:
        return override
    run_config = config["run"]
    output_dir = Path(str(run_config["output_dir"]))
    name = run_config.get("name") or run_id()
    return output_dir / str(name)


def resolve_existing_path(raw_path: str | Path, config_path: Path | None = None) -> Path:
    source_path = Path(str(raw_path)).expanduser()
    candidates = [source_path]
    if config_path is not None and not source_path.is_absolute():
        candidates.append(config_path.parent / source_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"path does not exist: {raw_path}")


# --------------------------------------------------------------------------- #
# Checkpoint discovery / resume
# --------------------------------------------------------------------------- #
def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.exists():
        return None
    checkpoints = [path for path in checkpoint_root.iterdir() if CHECKPOINT_RE.match(path.name)]
    if not checkpoints:
        return None
    return max(checkpoints, key=checkpoint_step)


def resolve_resume_checkpoint(run_dir: Path, resume: str | Path | bool | None) -> Path | None:
    if resume in {None, False, "false", "none", "off"}:
        return None
    if resume in {True, "true", "auto"}:
        return latest_checkpoint(run_dir)
    checkpoint = Path(str(resume))
    if not checkpoint.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
    return checkpoint


def prune_checkpoints(run_dir: Path, keep: int) -> None:
    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.exists():
        return
    checkpoints = sorted(
        (path for path in checkpoint_root.iterdir() if CHECKPOINT_RE.match(path.name)),
        key=checkpoint_step,
    )
    for stale in checkpoints[:-keep] if keep > 0 else []:
        stale.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Model loading + bundle (re)saving
# --------------------------------------------------------------------------- #
def _fa_package_dir() -> Path:
    package_dir = Path(__file__).resolve().parents[1] / "fa_fastconformer"
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
    return package_dir


def load_fastconformer_for_training(checkpoint: Path, map_location: str = "cpu"):
    """Load the standalone FastConformer-CTC model and the pieces needed to re-save it.

    Returns ``(model, save_config, tokenizer_proto)`` where ``save_config`` is the
    trimmed config and ``tokenizer_proto`` the serialized SentencePiece model, so
    trained weights can be written back into a ``.pt`` bundle that
    ``FastConformerCTC.from_pretrained`` (and ``eval_fastconformer``) can load.
    """
    _fa_package_dir()
    import torch  # noqa: F401  (ensures torch import errors surface here)
    from model import (  # standalone package (no NeMo)
        FastConformerCTC,
        _read_nemo_parts,
        unpack_nemo,
    )

    if checkpoint.suffix == ".nemo":
        import tempfile

        tmp = tempfile.mkdtemp(prefix="nemo_fa_train_")
        unpack_nemo(str(checkpoint), tmp)
        cfg, weights_path, tok_path = _read_nemo_parts(tmp)
        model = FastConformerCTC.from_extracted(cfg, weights_path, tok_path, map_location=map_location)
        save_config = {
            "preprocessor": cfg["preprocessor"],
            "encoder": cfg["encoder"],
            "aux_ctc": {"decoder": cfg["aux_ctc"]["decoder"]},
        }
        with open(tok_path, "rb") as handle:
            tokenizer_proto = handle.read()
    else:
        import torch

        bundle = torch.load(str(checkpoint), map_location=map_location, weights_only=False)
        model = FastConformerCTC.from_pretrained(str(checkpoint), map_location=map_location)
        save_config = bundle["config"]
        tokenizer_proto = bundle["tokenizer_proto"]
    return model, save_config, tokenizer_proto


def save_bundle(
    path: Path,
    model: Any,
    save_config: dict[str, Any],
    tokenizer_proto: bytes,
    training_state: dict[str, Any] | None = None,
) -> None:
    """Write a CTC-only ``.pt`` bundle (optionally with extra training state).

    The ``format``/``config``/``state_dict``/``tokenizer_proto`` keys match what
    ``FastConformerCTC.from_pretrained`` reads, so eval loads the file regardless
    of any extra ``training_state`` we stash for resume.
    """
    import torch

    _fa_package_dir()
    from model import BUNDLE_FORMAT

    state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith("encoder.") or key.startswith("ctc_decoder.")
    }
    bundle: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "config": save_config,
        "state_dict": state_dict,
        "tokenizer_proto": tokenizer_proto,
    }
    if training_state is not None:
        bundle["training_state"] = training_state
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, str(path))


# --------------------------------------------------------------------------- #
# Dataset / collation
# --------------------------------------------------------------------------- #
def encode_target(tokenizer: Any, transcript: str) -> list[int]:
    return list(tokenizer.EncodeAsIds(transcript))


def filter_examples_with_tokens(
    examples: list[WhisperExample],
    tokenizer: Any,
) -> tuple[list[tuple[WhisperExample, list[int]]], int]:
    """Drop examples that tokenize to an empty target (CTC needs length >= 1)."""
    kept: list[tuple[WhisperExample, list[int]]] = []
    skipped = 0
    for example in examples:
        tokens = encode_target(tokenizer, example.transcript)
        if not tokens:
            skipped += 1
            continue
        kept.append((example, tokens))
    return kept, skipped


class FastConformerDataset:
    def __init__(self, items: list[tuple[WhisperExample, list[int]]], sample_rate: int) -> None:
        self.items = items
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import soundfile as sf
        import torch
        import torchaudio.functional as F

        example, tokens = self.items[index]
        audio, source_rate = sf.read(str(example.audio_path), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        waveform = torch.as_tensor(audio, dtype=torch.float32)
        if int(source_rate) != self.sample_rate:
            waveform = F.resample(waveform, int(source_rate), self.sample_rate)
        return {"waveform": waveform, "tokens": torch.as_tensor(tokens, dtype=torch.long)}


class FastConformerCollator:
    """Pad waveforms to the batch max and stack token targets for CTC loss."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        waveforms = [feature["waveform"] for feature in features]
        token_seqs = [feature["tokens"] for feature in features]
        wave_lengths = torch.tensor([w.numel() for w in waveforms], dtype=torch.long)
        target_lengths = torch.tensor([t.numel() for t in token_seqs], dtype=torch.long)
        max_wave = int(wave_lengths.max())
        padded_waveforms = torch.zeros(len(waveforms), max_wave, dtype=torch.float32)
        for i, w in enumerate(waveforms):
            padded_waveforms[i, : w.numel()] = w
        max_target = int(target_lengths.max())
        padded_targets = torch.zeros(len(token_seqs), max_target, dtype=torch.long)
        for i, t in enumerate(token_seqs):
            padded_targets[i, : t.numel()] = t
        return {
            "waveforms": padded_waveforms,
            "wave_lengths": wave_lengths,
            "targets": padded_targets,
            "target_lengths": target_lengths,
        }


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #
def resolve_device(requested: str) -> str:
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device is cuda, but CUDA is not available")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def resolve_fp16(mixed_precision: Any, device: str) -> bool:
    if device == "cpu":
        return False
    import torch

    if mixed_precision == "auto":
        return torch.cuda.is_available()
    return mixed_precision in {True, "true"}


def ctc_loss_step(model: Any, batch: dict[str, Any], device: str) -> Any:
    """Run encoder + CTC head and return the (mean) CTC loss for one batch."""
    import torch
    import torch.nn.functional as F

    waveforms = batch["waveforms"].to(device)
    wave_lengths = batch["wave_lengths"].to(device)
    targets = batch["targets"].to(device)
    target_lengths = batch["target_lengths"].to(device)

    feats, feat_len = model.preprocessor(waveforms, wave_lengths)
    enc, enc_len = model.encoder(feats, feat_len)
    log_probs = model.ctc_decoder(enc)  # (B, T, V+1), already log_softmax
    # CTC expects (T, B, V); compute in float32 for numerical stability under AMP.
    log_probs = log_probs.transpose(0, 1).float()
    return F.ctc_loss(
        log_probs,
        targets,
        enc_len.to(torch.long),
        target_lengths,
        blank=model.blank_id,
        reduction="mean",
        zero_infinity=True,
    )


def evaluate_wer(
    model: Any,
    eval_items: list[tuple[WhisperExample, list[int]]],
    batch_size: int,
    device: str,
    sample_rate: int,
    max_batch_seconds: float | None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    try:
        audio_paths = [str(example.audio_path) for example, _ in eval_items]
        hypotheses = model.transcribe(
            audio_paths,
            batch_size=batch_size,
            device=device,
            target_sr=sample_rate,
            progress=True,
            max_batch_seconds=max_batch_seconds,
        )
        references = [example.transcript for example, _ in eval_items]
        metrics = {
            "wer": word_error_rate(references, hypotheses),
            "cer": character_error_rate(references, hypotheses),
        }
    finally:
        if was_training:
            model.train()
    return metrics


def make_progress_bar(iterable: Any, desc: str, total: int) -> Any:
    """Wrap an iterable in a tqdm bar (auto-disabled when not attached to a TTY).

    ``disable=None`` lets tqdm silence itself for non-interactive runs (nohup,
    redirected logs), so the file log stays clean while interactive terminals
    still get a live bar. Falls back to the bare iterable if tqdm is missing.
    """
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, total=total, unit="batch", dynamic_ncols=True, leave=False, disable=None)
    except ImportError:
        return iterable


def build_dataloader(dataset: Any, batch_size: int, num_workers: int, shuffle: bool, generator: Any) -> Any:
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=FastConformerCollator(),
        drop_last=False,
        generator=generator,
        pin_memory=False,
    )


# --------------------------------------------------------------------------- #
# Training entry point
# --------------------------------------------------------------------------- #
def run_training(config_path: Path, run_dir_override: Path | None = None, resume_override: str | None = None) -> int:
    import math

    import torch
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup, set_seed

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("loading config=%s", config_path)
    config = load_training_config(config_path)
    run_dir = resolve_run_dir(config, run_dir_override)
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(run_dir)
    logging.info("configured logging file=%s", run_dir / "logs" / "train.log")

    effective_config_path = run_dir / "config" / "training.yaml"
    effective_config_path.parent.mkdir(parents=True, exist_ok=True)
    effective_config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    logging.info("wrote effective config=%s", effective_config_path)

    metrics_path = run_dir / "logs" / "train_metrics.jsonl"
    resume_value = resume_override if resume_override is not None else config["run"].get("resume")
    resume_checkpoint = resolve_resume_checkpoint(run_dir, resume_value)
    update_status(
        run_dir,
        run_id=run_dir.name,
        status="running",
        started_at=utc_now(),
        config_path=str(config_path),
        effective_config_path=str(effective_config_path),
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,
        error=None,
    )

    model_config = config["model"]
    data_config = config["data"]
    training = config["training"]
    sample_rate = int(data_config["sample_rate"])

    logging.info("run_dir=%s", run_dir)
    logging.info("resume_from_checkpoint=%s", resume_checkpoint or "none")
    logging.info("setting seed=%s", training["seed"])
    set_seed(int(training["seed"]))

    try:
        device = resolve_device(str(training["device"]))
        fp16 = resolve_fp16(training["mixed_precision"], device)
        logging.info("device=%s fp16=%s", device, fp16)

        # Start either from the configured pretrained checkpoint, or — when
        # resuming — from the resume checkpoint's weights (the resume bundle is a
        # full model bundle plus stashed training_state).
        load_source = resume_checkpoint or resolve_existing_path(str(model_config["checkpoint"]), config_path)
        logging.info("loading model from=%s", load_source)
        model, save_config, tokenizer_proto = load_fastconformer_for_training(load_source)
        model.to(device)

        if bool(training["freeze_encoder"]):
            logging.info("freezing encoder parameters (training CTC head only)")
            for parameter in model.encoder.parameters():
                parameter.requires_grad_(False)

        logging.info("resolving dataset directories root=%s datasets=%s", data_config["root_dir"], data_config["datasets"])
        dataset_dirs = resolve_dataset_dirs(config)
        logging.info("resolved dataset directories=%s", ", ".join(str(path) for path in dataset_dirs))
        logging.info("loading training examples")
        train_examples = load_split_examples(dataset_dirs, "train")
        logging.info("loading evaluation examples")
        eval_examples = load_split_examples(dataset_dirs, "dev")

        train_items, skipped_train = filter_examples_with_tokens(train_examples, model.tokenizer)
        eval_items, skipped_eval = filter_examples_with_tokens(eval_examples, model.tokenizer)
        if not train_items:
            raise ValueError("no train examples remain after empty-target filtering")
        if not eval_items:
            raise ValueError("no dev examples remain after empty-target filtering")
        if skipped_train or skipped_eval:
            logging.warning("skipped %s train / %s dev examples with empty token targets", skipped_train, skipped_eval)

        logging.info("writing source manifests")
        write_examples_manifest(run_dir / "manifests" / "train.jsonl", [example for example, _ in train_items])
        write_examples_manifest(run_dir / "manifests" / "dev.jsonl", [example for example, _ in eval_items])
        update_status(
            run_dir,
            datasets=[str(path) for path in dataset_dirs],
            train_examples=len(train_items),
            eval_examples=len(eval_items),
            skipped_train_examples=skipped_train,
            skipped_eval_examples=skipped_eval,
            pretrained_model=str(load_source),
        )
        logging.info("train_examples=%s eval_examples=%s", len(train_items), len(eval_items))

        train_dataset = FastConformerDataset(train_items, sample_rate)
        generator = torch.Generator()
        generator.manual_seed(int(training["seed"]))
        train_batch_size = int(training["per_device_train_batch_size"])
        grad_accum = int(training["gradient_accumulation_steps"])
        train_loader = build_dataloader(
            train_dataset, train_batch_size, int(training["num_workers"]), shuffle=True, generator=generator
        )

        steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
        # honour fractional epochs for the optimizer-step budget
        max_steps = max(1, int(steps_per_epoch * float(training["num_train_epochs"])))
        logging.info("steps_per_epoch=%s max_optimizer_steps=%s", steps_per_epoch, max_steps)

        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = AdamW(
            trainable,
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(training["warmup_steps"]),
            num_training_steps=max_steps,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=fp16)

        global_step = 0
        start_epoch = 0
        best_wer: float | None = None
        if resume_checkpoint is not None:
            resume_state = torch.load(str(resume_checkpoint), map_location="cpu", weights_only=False).get("training_state")
            if resume_state:
                optimizer.load_state_dict(resume_state["optimizer"])
                scheduler.load_state_dict(resume_state["scheduler"])
                if resume_state.get("scaler") and fp16:
                    scaler.load_state_dict(resume_state["scaler"])
                global_step = int(resume_state.get("global_step", 0))
                start_epoch = int(resume_state.get("epoch", 0))
                best_wer = resume_state.get("best_wer")
                logging.info("resumed at global_step=%s epoch=%s best_wer=%s", global_step, start_epoch, best_wer)

        save_total_limit = int(training["save_total_limit"])
        logging_steps = int(training["logging_steps"])
        eval_steps = int(training["eval_steps"])
        save_steps = int(training["save_steps"])
        max_grad_norm = float(training["max_grad_norm"])
        eval_batch_size = int(training["per_device_eval_batch_size"])
        eval_max_batch_seconds = (
            float(training["eval_max_batch_seconds"]) if training.get("eval_max_batch_seconds") is not None else None
        )

        def training_state() -> dict[str, Any]:
            return {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if fp16 else None,
                "global_step": global_step,
                "epoch": current_epoch,
                "best_wer": best_wer,
            }

        def save_checkpoint() -> Path:
            checkpoint_path = run_dir / "checkpoints" / f"checkpoint-{global_step}.pt"
            save_bundle(checkpoint_path, model, save_config, tokenizer_proto, training_state())
            prune_checkpoints(run_dir, save_total_limit)
            update_status(run_dir, latest_checkpoint=str(checkpoint_path), global_step=global_step)
            logging.info("saved checkpoint=%s", checkpoint_path)
            return checkpoint_path

        num_epochs = int(math.ceil(float(training["num_train_epochs"])))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_count = 0
        current_epoch = start_epoch
        stop = False

        logging.info("starting training global_step=%s -> max_steps=%s", global_step, max_steps)
        for epoch in range(start_epoch, num_epochs):
            current_epoch = epoch
            progress = make_progress_bar(train_loader, desc=f"epoch {epoch + 1}/{num_epochs}", total=len(train_loader))
            set_postfix = getattr(progress, "set_postfix", None)
            for micro_step, batch in enumerate(progress):
                with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", enabled=fp16):
                    loss = ctc_loss_step(model, batch, device)
                loss_value = float(loss.detach().item())
                running_loss += loss_value
                running_count += 1
                scaler.scale(loss / grad_accum).backward()

                is_update_step = (micro_step + 1) % grad_accum == 0 or (micro_step + 1) == len(train_loader)
                if not is_update_step:
                    continue

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if set_postfix is not None:
                    set_postfix(step=global_step, loss=f"{loss_value:.3f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

                if global_step % logging_steps == 0:
                    avg_loss = running_loss / max(1, running_count)
                    current_lr = scheduler.get_last_lr()[0]
                    logging.info(
                        "step=%s epoch=%.2f loss=%.4f lr=%.2e",
                        global_step,
                        epoch + (micro_step + 1) / len(train_loader),
                        avg_loss,
                        current_lr,
                    )
                    append_jsonl(
                        metrics_path,
                        {
                            "timestamp": utc_now(),
                            "step": global_step,
                            "epoch": epoch + (micro_step + 1) / len(train_loader),
                            "loss": avg_loss,
                            "learning_rate": current_lr,
                        },
                    )
                    running_loss = 0.0
                    running_count = 0

                if global_step % eval_steps == 0:
                    logging.info("evaluating at step=%s on %s dev examples", global_step, len(eval_items))
                    eval_metrics = evaluate_wer(
                        model, eval_items, eval_batch_size, device, sample_rate, eval_max_batch_seconds
                    )
                    logging.info("eval step=%s wer=%.4f cer=%.4f", global_step, eval_metrics["wer"], eval_metrics["cer"])
                    append_jsonl(
                        metrics_path,
                        {"timestamp": utc_now(), "step": global_step, "epoch": float(epoch), "eval_wer": eval_metrics["wer"], "eval_cer": eval_metrics["cer"]},
                    )
                    if best_wer is None or eval_metrics["wer"] < best_wer:
                        best_wer = eval_metrics["wer"]
                        save_bundle(run_dir / "best.pt", model, save_config, tokenizer_proto)
                        update_status(run_dir, best_wer=best_wer, best_model=str(run_dir / "best.pt"))
                        logging.info("new best wer=%.4f saved best.pt", best_wer)

                if global_step % save_steps == 0:
                    save_checkpoint()

                if global_step >= max_steps:
                    stop = True
                    break
            if stop:
                break

        logging.info("training loop finished at global_step=%s", global_step)
        save_checkpoint()
        final_path = run_dir / "final.pt"
        save_bundle(final_path, model, save_config, tokenizer_proto)
        logging.info("saved final model=%s", final_path)

        final_metrics = evaluate_wer(model, eval_items, eval_batch_size, device, sample_rate, eval_max_batch_seconds)
        logging.info("final eval wer=%.4f cer=%.4f", final_metrics["wer"], final_metrics["cer"])
        append_jsonl(
            metrics_path,
            {"timestamp": utc_now(), "step": global_step, "final_eval_wer": final_metrics["wer"], "final_eval_cer": final_metrics["cer"]},
        )
        if best_wer is None or final_metrics["wer"] < best_wer:
            best_wer = final_metrics["wer"]
            save_bundle(run_dir / "best.pt", model, save_config, tokenizer_proto)

        write_json(
            run_dir / "train_summary.json",
            {
                "created_at": utc_now(),
                "global_step": global_step,
                "final_wer": final_metrics["wer"],
                "final_cer": final_metrics["cer"],
                "best_wer": best_wer,
                "final_model": str(final_path),
                "best_model": str(run_dir / "best.pt"),
            },
        )
        update_status(
            run_dir,
            status="completed",
            completed_at=utc_now(),
            global_step=global_step,
            latest_checkpoint=str(latest_checkpoint(run_dir)) if latest_checkpoint(run_dir) else None,
            final_model=str(final_path),
            best_model=str(run_dir / "best.pt"),
            best_wer=best_wer,
            final_wer=final_metrics["wer"],
            error=None,
        )
        logging.info("run completed final_model=%s best_model=%s", final_path, run_dir / "best.pt")
        return 0
    except KeyboardInterrupt:
        logging.warning("training interrupted by user")
        update_status(
            run_dir,
            status="interrupted",
            interrupted_at=utc_now(),
            latest_checkpoint=str(latest_checkpoint(run_dir)) if latest_checkpoint(run_dir) else None,
            error="Interrupted by user. Re-run with run.resume: auto or --resume auto to continue.",
        )
        raise
    except Exception as exc:
        logging.exception("training failed: %s", exc)
        update_status(
            run_dir,
            status="failed",
            failed_at=utc_now(),
            latest_checkpoint=str(latest_checkpoint(run_dir)) if latest_checkpoint(run_dir) else None,
            error=str(exc),
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the standalone FastConformer-CTC Persian model from a YAML config. "
            "Stop with Ctrl+C after checkpoints exist, then resume by re-running with "
            "run.resume: auto or --resume auto."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML training config path.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Optional run directory override.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume mode: auto, false, or an explicit checkpoint .pt file. Overrides run.resume.",
    )
    args = parser.parse_args(argv)
    return run_training(args.config, args.run_dir, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
