"""Evaluate a trained dual-view fusion model as a whole ASR system on test splits.

Mirrors ``ml.asr.eval_whisper_small`` / ``ml.asr.eval_fastconformer``: it reads a
YAML config, loads the configured dataset ``test.tsv`` files (the project ASR
split-TSV + ``clips/`` contract), transcribes every clip through the trained
:class:`~ml.fusion.model.DualViewFusionModel` (enhancer -> shared Whisper encoder
-> cross-attention fusion -> Whisper decoder), and writes ``metrics.json``
(aggregate WER/CER plus per-dataset metrics), ``predictions.jsonl``, the effective
config, logs, and a source manifest.

The fusion model is the thesis ASR system end to end: each test clip's Whisper
log-Mel is the *noisy* view fed to the model, and ``model.generate`` decodes token
ids from the fused encoder stream exactly as training's dev eval does. A
``view_usage`` block in ``metrics.json`` additionally reports how much of each
encoded view the fusion gate used — the mean/median over clips of the enhanced
(clean) weight ``g`` and the noisy weight ``1 - g`` — with the per-clip weights
also written to ``predictions.jsonl``. Evaluating
on a clean (non-degraded) ``test.tsv`` therefore measures how the fused stack
transcribes that audio as a drop-in ASR model; point ``data.datasets`` at a
degraded dataset's clip dirs to measure robustness instead.

``model.checkpoint`` is a fusion training checkpoint (``fusion_model.pt`` from a
run's ``checkpoints/stage2_joint/``; ``best.pt`` or a Stage 1 checkpoint also
work). The enhancer/fusion architecture is read back from the checkpoint, so it
need not be repeated in the config. ``model.base_asr_checkpoint`` supplies the
Whisper backbone *architecture* and ``generation_config`` (its weights are
overwritten by the checkpoint when the checkpoint carries the backbone); default
``openai/whisper-small`` keeps eval self-contained without the original
fine-tuned run dir.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ml.asr.train_whisper_small import (
    WhisperExample,
    character_error_rate,
    deep_merge,
    load_split_examples,
    resolve_dataset_dirs,
    word_error_rate,
    write_examples_manifest,
)
from ml.asr.whisper_features import WHISPER_SAMPLE_RATE, waveform_to_log_mel
from ml.fusion.model import build_fusion_model, is_loadable_checkpoint
from ml.fusion.train_fusion import load_fusion_checkpoint
from ml.utils.audio import load_audio, resample_audio, to_mono


DEFAULT_EVAL_CONFIG: dict[str, Any] = {
    "model": {
        "checkpoint": None,
        # Whisper backbone architecture + generation_config source. A local
        # fine-tuned run dir or a Hub id; the checkpoint overwrites its weights.
        "base_asr_checkpoint": "openai/whisper-small",
        "model_name": "openai/whisper-small",
        # Tokenizer for decoding; defaults to model_name when unset.
        "processor": None,
        "language": "Persian",
        "task": "transcribe",
    },
    "data": {
        "root_dir": "data",
        "datasets": ["cv-corpus-25.0"],
        "sample_rate": WHISPER_SAMPLE_RATE,
        "split": "test",
    },
    "eval": {
        "output_dir": "models/asr/fusion/evals",
        "name": None,
        "batch_size": 8,
        "device": "auto",
        "mixed_precision": "auto",
        "generation_max_length": 225,
    },
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def load_eval_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    config = deep_merge(DEFAULT_EVAL_CONFIG, loaded)
    validate_eval_config(config)
    return config


def validate_eval_config(config: dict[str, Any]) -> None:
    model = config["model"]
    data = config["data"]
    eval_config = config["eval"]
    if not str(model.get("checkpoint") or "").strip():
        raise ValueError("model.checkpoint must be a non-empty fusion checkpoint path (e.g. fusion_model.pt)")
    if not str(model.get("base_asr_checkpoint") or "").strip():
        raise ValueError("model.base_asr_checkpoint must be a non-empty local path or Hugging Face Hub id")
    datasets = data.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("data.datasets must be a non-empty list of dataset directory names")
    if any(not str(dataset).strip() for dataset in datasets):
        raise ValueError("data.datasets cannot contain empty values")
    split = str(data.get("split", "")).strip()
    if not split or split.endswith(".tsv"):
        raise ValueError("data.split must be a split name such as test, not a TSV filename")
    minimums = {
        "sample_rate": (data, 8000),
        "batch_size": (eval_config, 1),
        "generation_max_length": (eval_config, 1),
    }
    for key, (section, minimum) in minimums.items():
        if float(section[key]) < minimum:
            raise ValueError(f"{key} must be >= {minimum:g}")
    if eval_config["device"] not in {"auto", "cuda", "cpu"}:
        raise ValueError("eval.device must be auto, cuda, or cpu")


def resolve_existing_path(raw_path: str | Path, config_path: Path | None = None) -> Path:
    source_path = Path(str(raw_path)).expanduser()
    candidates = [source_path]
    if config_path is not None and not source_path.is_absolute():
        candidates.append(config_path.parent / source_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"path does not exist: {raw_path}")


def resolve_source(raw_source: str | Path, config_path: Path | None = None) -> str:
    """Resolve a local path, or return a Hugging Face Hub id unchanged.

    Backbone/processor sources may be a local fine-tuned run dir or a Hub id such
    as ``openai/whisper-small``; an existing path is resolved (also relative to the
    config dir), and anything else is returned as-is for ``from_pretrained``.
    """
    source = str(raw_source).strip()
    source_path = Path(source).expanduser()
    candidates = [source_path]
    if config_path is not None and not source_path.is_absolute():
        candidates.append(config_path.parent / source_path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return source


def resolve_output_dir(config: dict[str, Any], override: Path | None = None) -> Path:
    if override is not None:
        return override
    eval_config = config["eval"]
    name = eval_config.get("name") or run_id()
    return Path(str(eval_config["output_dir"])) / str(name)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def configure_logging(output_dir: Path) -> None:
    log_path = output_dir / "logs" / "eval.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    for noisy_logger in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def error_metrics(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    return {
        "wer": word_error_rate(references, hypotheses),
        "cer": character_error_rate(references, hypotheses),
    }


def dataset_error_metrics(
    examples: list[WhisperExample],
    references: list[str],
    hypotheses: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[str]]] = {}
    for example, reference, hypothesis in zip(examples, references, hypotheses, strict=True):
        dataset = str(example.dataset_dir or example.audio_path.parent)
        group = grouped.setdefault(dataset, {"references": [], "hypotheses": []})
        group["references"].append(reference)
        group["hypotheses"].append(hypothesis)

    return [
        {
            "dataset": dataset,
            "examples": len(group["references"]),
            **error_metrics(group["references"], group["hypotheses"]),
        }
        for dataset, group in grouped.items()
    ]


def view_usage_metrics(clean_view_weights: list[float]) -> dict[str, Any]:
    """Summarise how much of each encoded view the fusion gate used across clips.

    ``clean_view_weights`` is the per-example mean fusion gate ``g`` (weight on the
    enhanced/clean encoder stream); the noisy weight is ``1 - g``. Reports the mean
    and median of both fractions over the evaluated clips, so a number near 1 means
    the fusion leaned on the enhanced view and near 0 means it fell back on the raw
    noisy view.
    """
    if not clean_view_weights:
        return {"examples": 0}
    clean = np.asarray(clean_view_weights, dtype=np.float64)
    return {
        "examples": int(clean.size),
        "clean_view_weight_mean": float(clean.mean()),
        "clean_view_weight_median": float(np.median(clean)),
        "noisy_view_weight_mean": float((1.0 - clean).mean()),
        "noisy_view_weight_median": float(np.median(1.0 - clean)),
    }


def resolve_device(requested: str) -> str:
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("eval.device is cuda, but CUDA is not available")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def use_amp(mixed_precision: Any, device: str) -> bool:
    if device != "cuda":
        return False
    if mixed_precision == "auto":
        return True
    return mixed_precision in {True, "true"}


def make_progress_bar(iterable: Any, desc: str, total: int | None = None) -> Any:
    """Wrap an iterable in a tqdm bar, auto-disabled off-TTY (``disable=None``)."""
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, total=total, unit="batch", dynamic_ncols=True, leave=False, disable=None)
    except ImportError:
        return iterable


def load_fusion_model(
    checkpoint: Path,
    *,
    base_asr_checkpoint: str,
    model_name: str,
) -> tuple[Any, bool]:
    """Rebuild a :class:`DualViewFusionModel` from a fusion training checkpoint.

    The enhancer/fusion architecture is read back from the checkpoint payload, the
    Whisper backbone is instantiated from ``base_asr_checkpoint`` (architecture +
    ``generation_config``), and the trained weights are then loaded — strictly when
    the checkpoint carries the backbone, non-strictly (Stage 1's backbone-free
    rolling save) when it does not.

    Returns ``(model, backbone_included)``. When ``backbone_included`` is true the
    backbone weights came from the checkpoint itself — i.e. the jointly-trained
    Stage 2 backbone — and ``base_asr_checkpoint`` only supplied the architecture
    skeleton. When false the backbone comes wholly from ``base_asr_checkpoint``
    (a Stage 1 backbone-free save), so that must point at the frozen fine-tuned
    backbone used during training, not a vanilla baseline.
    """
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    backbone_included = bool(payload.get("backbone_included", True))
    build_config = {
        "enhancer": payload.get("enhancer_config"),
        "fusion": payload.get("fusion_config"),
        "base_asr_checkpoint": base_asr_checkpoint,
        "model_name": model_name,
    }
    model = build_fusion_model(build_config)
    load_fusion_checkpoint(model, checkpoint)
    return model, backbone_included


def configure_generation(model: Any, language: str | None, task: str | None) -> None:
    """Steer the Whisper decoder prompt for generation (multilingual backbones only).

    Mirrors the trainer's dev eval: drop Whisper's preset ``max_length`` so the
    ``max_new_tokens`` cap applies cleanly, and set language/task only when the
    backbone actually carries the language-token map.
    """
    generation_config = model.whisper.generation_config
    generation_config.max_length = None
    if getattr(generation_config, "lang_to_id", None):
        if language:
            generation_config.language = str(language)
        if task:
            generation_config.task = str(task)


def transcribe_examples(
    model: Any,
    examples: list[WhisperExample],
    tokenizer: Any,
    *,
    device: str,
    sample_rate: int,
    model_name: str,
    batch_size: int,
    generation_max_length: int,
    amp_enabled: bool,
) -> tuple[list[str], list[float]]:
    """Greedily transcribe every clip through the fused ASR stack, in batches.

    Returns ``(hypotheses, clean_view_weights)``: the decoded transcripts and, per
    example (same order), the mean fusion gate ``g`` — the fraction of the
    *enhanced/clean* encoded view the fusion used (``1 - g`` is the noisy view).
    The gate is captured by a forward hook on ``model.fusion.combine`` (the shared
    sigmoid gate of both fusion blocks), which fires once per batch when
    ``encode_views`` runs inside ``generate``.
    """
    import torch

    model.to(device).eval()
    clean_view_weights: list[float] = []

    def capture_gate(module: Any, args: Any, _output: Any) -> None:
        noisy_h, enhanced_h = args[0], args[1]
        gate = module.gate(noisy_h, enhanced_h)  # [B, T, D] weight on the enhanced view
        clean_view_weights.extend(gate.float().mean(dim=(1, 2)).tolist())

    handle = model.fusion.combine.register_forward_hook(capture_gate)
    hypotheses: list[str] = []
    total_batches = (len(examples) + batch_size - 1) // batch_size
    batch_starts = range(0, len(examples), batch_size)
    try:
        for start in make_progress_bar(batch_starts, "fusion eval", total=total_batches):
            batch = examples[start : start + batch_size]
            mels = []
            for example in batch:
                audio, source_rate = load_audio(example.audio_path)
                audio = to_mono(np.asarray(audio, dtype=np.float32))
                if int(source_rate) != sample_rate:
                    audio = resample_audio(audio, int(source_rate), sample_rate)
                mels.append(waveform_to_log_mel(audio, sample_rate=sample_rate, model_name=model_name))
            noisy_mel = torch.stack(mels).to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
                pred_ids = model.generate(noisy_mel, max_new_tokens=int(generation_max_length))
            hypotheses.extend(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
    finally:
        handle.remove()
    return hypotheses, clean_view_weights


def run_evaluation(config_path: Path, output_dir_override: Path | None = None) -> int:
    from transformers import WhisperTokenizer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_eval_config(config_path)
    output_dir = resolve_output_dir(config, output_dir_override)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir)

    effective_config_path = output_dir / "config" / "eval.yaml"
    effective_config_path.parent.mkdir(parents=True, exist_ok=True)
    effective_config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    model_config = config["model"]
    data_config = config["data"]
    eval_config = config["eval"]

    checkpoint = resolve_existing_path(str(model_config["checkpoint"]), config_path)
    base_asr_checkpoint = resolve_source(str(model_config["base_asr_checkpoint"]), config_path)
    if not is_loadable_checkpoint(base_asr_checkpoint):
        raise FileNotFoundError(
            f"model.base_asr_checkpoint is not a loadable source: {base_asr_checkpoint!r}. "
            "Point it at a local Whisper run dir or a Hub id (e.g. openai/whisper-small)."
        )
    model_name = str(model_config["model_name"])
    processor_name = resolve_source(str(model_config.get("processor") or model_name), config_path)
    device = resolve_device(str(eval_config["device"]))
    amp_enabled = use_amp(eval_config.get("mixed_precision", "auto"), device)

    dataset_dirs = resolve_dataset_dirs(config)
    split = str(data_config["split"])
    examples = load_split_examples(dataset_dirs, split)
    write_examples_manifest(output_dir / "manifests" / f"{split}.jsonl", examples)

    logging.info("loading tokenizer=%s", processor_name)
    tokenizer = WhisperTokenizer.from_pretrained(processor_name)
    logging.info("loading fusion checkpoint=%s base_asr=%s device=%s", checkpoint, base_asr_checkpoint, device)
    model, backbone_included = load_fusion_model(
        checkpoint, base_asr_checkpoint=base_asr_checkpoint, model_name=model_name
    )
    if backbone_included:
        logging.info("backbone weights loaded from the checkpoint (jointly-trained backbone)")
    else:
        # A backbone-free checkpoint (Stage 1 last.pt): the backbone is taken
        # wholly from base_asr_checkpoint, so a vanilla baseline there would splice
        # an un-fine-tuned backbone under the trained front end.
        logging.warning(
            "checkpoint carries no backbone; using base_asr_checkpoint=%s as the backbone. "
            "Point it at the fine-tuned Whisper used during training, not a vanilla baseline.",
            base_asr_checkpoint,
        )
    configure_generation(model, model_config.get("language"), model_config.get("task"))

    logging.info("transcribing %s %s examples (amp=%s)", len(examples), split, amp_enabled)
    hypotheses, clean_view_weights = transcribe_examples(
        model,
        examples,
        tokenizer,
        device=device,
        sample_rate=int(data_config["sample_rate"]),
        model_name=model_name,
        batch_size=int(eval_config["batch_size"]),
        generation_max_length=int(eval_config["generation_max_length"]),
        amp_enabled=amp_enabled,
    )
    references = [example.transcript for example in examples]

    aggregate_metrics = error_metrics(references, hypotheses)
    view_usage = view_usage_metrics(clean_view_weights)
    logging.info(
        "fusion view usage: clean(enhanced) mean=%.3f median=%.3f | noisy mean=%.3f median=%.3f",
        view_usage.get("clean_view_weight_mean", float("nan")),
        view_usage.get("clean_view_weight_median", float("nan")),
        view_usage.get("noisy_view_weight_mean", float("nan")),
        view_usage.get("noisy_view_weight_median", float("nan")),
    )
    metrics = {
        "created_at": utc_now(),
        "config_path": str(config_path),
        "effective_config_path": str(effective_config_path),
        "checkpoint": str(checkpoint),
        "base_asr_checkpoint": base_asr_checkpoint,
        "backbone_from_checkpoint": backbone_included,
        "processor": processor_name,
        "device": device,
        "datasets": [str(path) for path in dataset_dirs],
        "split": split,
        "examples": len(examples),
        **aggregate_metrics,
        "view_usage": view_usage,
        "dataset_metrics": dataset_error_metrics(examples, references, hypotheses),
    }
    write_json(output_dir / "metrics.json", metrics)

    # Align per-clip clean-view weights with examples; tolerate a length mismatch
    # (e.g. hook never fired) by falling back to None rather than failing the run.
    clip_weights = clean_view_weights if len(clean_view_weights) == len(examples) else [None] * len(examples)
    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for index, (example, reference, hypothesis, clean_weight) in enumerate(
            zip(examples, references, hypotheses, clip_weights, strict=True), start=1
        ):
            handle.write(
                json.dumps(
                    {
                        "id": index,
                        "audio_path": str(example.audio_path),
                        "dataset": str(example.dataset_dir or example.audio_path.parent),
                        "reference": reference,
                        "hypothesis": hypothesis,
                        "clean_view_weight": clean_weight,
                        "noisy_view_weight": None if clean_weight is None else 1.0 - clean_weight,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    logging.info("wrote metrics=%s predictions=%s", output_dir / "metrics.json", predictions_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained dual-view fusion model as an ASR system on configured dataset test.tsv files."
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML evaluation config path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional evaluation output directory override.")
    args = parser.parse_args(argv)
    return run_evaluation(args.config, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
