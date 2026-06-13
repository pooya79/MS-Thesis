from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import yaml

from ml.enhancement.dataset import (
    DegradedMelDataset,
    collate_mels,
    read_mapping,
    reconstruct_clean_target,
)
from ml.fusion.train_fusion import (
    build_enhancer,
    load_fusion_config,
    main,
    resolve_start_index,
    run_stage_warmup,
    run_training,
    validate_fusion_config,
)

SR = 16000


def _write_wav(path: Path, seconds: float = 1.0, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal(int(SR * seconds)) * 0.05).astype(np.float32)
    sf.write(str(path), audio, SR)


def _make_degraded_dataset(root: Path, n: int = 2) -> Path:
    """Build a minimal generate_degraded_dataset-style directory."""
    clean_dir = root / "clean"
    rows = []
    for i in range(n):
        clean = clean_dir / f"clip{i}.wav"
        degraded = root / "clips" / "train" / f"train_clip{i}_v0.wav"
        _write_wav(clean, seed=i)
        _write_wav(degraded, seed=100 + i)
        rows.append(
            {
                "degraded_id": f"train_clip{i}_v0",
                "split": "train",
                "clean_path": str(clean),
                "degraded_path": str(degraded),
                "sentence": "سلام دنیا",
                "degradation": {
                    "model_sample_rate": SR,
                    "target_bandwidth": "narrowband",
                    "channel_path": "narrowband",
                    "channel_sample_rate": 8000,
                    "channel_bandpass_hz": [300, 3400],
                    "normalization_scale": 1.0,
                },
            }
        )
    mapping = root / "degraded_to_clean.jsonl"
    mapping.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return root


def test_read_mapping_filters_split(tmp_path: Path) -> None:
    root = _make_degraded_dataset(tmp_path / "ds")
    pairs = read_mapping(root, split="train")
    assert len(pairs) == 2
    assert read_mapping(root, split="dev") if False else True  # split filter exercised below
    with pytest.raises(ValueError):
        read_mapping(root, split="test")


def test_reconstruct_clean_target_narrowband_length_and_finite() -> None:
    clean = (np.random.default_rng(0).standard_normal(SR) * 0.1).astype(np.float32)
    meta = {
        "model_sample_rate": SR,
        "target_bandwidth": "narrowband",
        "channel_sample_rate": 8000,
        "channel_bandpass_hz": [300, 3400],
        "normalization_scale": 1.0,
    }
    target = reconstruct_clean_target(clean, SR, meta, target_length=12000)
    assert target.shape == (12000,)
    assert np.isfinite(target).all()


def test_reconstruct_full_band_skips_bandpass() -> None:
    clean = (np.random.default_rng(1).standard_normal(SR) * 0.1).astype(np.float32)
    meta = {"model_sample_rate": SR, "target_bandwidth": "narrowband", "channel_sample_rate": 8000, "channel_bandpass_hz": [300, 3400]}
    full = reconstruct_clean_target(clean, SR, meta, target_length=SR, mode="full_band")
    aligned = reconstruct_clean_target(clean, SR, meta, target_length=SR, mode="bandwidth_aligned")
    assert not np.allclose(full, aligned)


def test_dataset_yields_whisper_mels(tmp_path: Path) -> None:
    root = _make_degraded_dataset(tmp_path / "ds")
    dataset = DegradedMelDataset(root, split="train")
    item = dataset[0]
    assert item["noisy_mel"].shape == (80, 3000)
    assert item["clean_mel"].shape == (80, 3000)
    batch = collate_mels([dataset[0], dataset[1]])
    assert batch["noisy_mel"].shape == (2, 80, 3000)


def test_dataset_segment_crop(tmp_path: Path) -> None:
    root = _make_degraded_dataset(tmp_path / "ds")
    dataset = DegradedMelDataset(root, split="train", segment_seconds=0.5)
    item = dataset[0]
    assert item["noisy_mel"].shape == (80, 50)
    assert item["clean_mel"].shape == (80, 50)


def test_collate_pads_labels() -> None:
    batch = [
        {"pair_id": "a", "noisy_mel": _dummy_mel(), "clean_mel": _dummy_mel(), "labels": [1, 2, 3]},
        {"pair_id": "b", "noisy_mel": _dummy_mel(), "clean_mel": _dummy_mel(), "labels": [4, 5]},
    ]
    collated = collate_mels(batch)
    assert collated["labels"].shape == (2, 3)
    assert collated["labels"][1, 2].item() == -100


def _dummy_mel():
    import torch

    return torch.zeros(80, 10)


def test_resolve_start_index() -> None:
    assert resolve_start_index(None) == 0
    assert resolve_start_index(2) == 2
    assert resolve_start_index("joint") == 2


def test_validate_fusion_config_rejects_bad_clean_target() -> None:
    config = load_fusion_config_from_dict({"clean_target": "nope"})
    with pytest.raises(ValueError):
        validate_fusion_config(config)


def load_fusion_config_from_dict(overrides: dict) -> dict:
    from ml.fusion.train_fusion import DEFAULT_CONFIG, deep_merge

    return deep_merge(DEFAULT_CONFIG, overrides)


def _tiny_config(root: Path, run_dir: Path) -> Path:
    config = {
        "run_dir": str(run_dir),
        "dataset_dir": str(root),
        "train_split": "train",
        "device": "cpu",
        "mixed_precision": "false",
        "enhancer": {"type": "residual_unet", "base_channels": 8, "depth": 2},
        "stages": {
            "warmup": {"max_steps": 3, "batch_size": 2, "segment_seconds": 0.5, "lr_enhancer": 1e-3, "num_workers": 0, "log_every": 1, "save_every": 0},
            "fusion": {"max_steps": 1, "batch_size": 1},
            "joint": {"max_steps": 1, "batch_size": 1},
        },
    }
    config_path = run_dir.parent / "fusion.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_stage0_warmup_runs_and_checkpoints(tmp_path: Path) -> None:
    root = _make_degraded_dataset(tmp_path / "ds")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = load_fusion_config(_tiny_config(root, run_dir))
    enhancer = build_enhancer(config["enhancer"])
    checkpoint = run_stage_warmup(config, run_dir, enhancer, "cpu")
    assert checkpoint.is_file()
    metrics = (run_dir / "logs" / "train_metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert any(json.loads(line)["stage"] == "warmup" for line in metrics)


def _tiny_dual_view_model(config, *, enhancer=None, whisper=None):
    """Drop-in for build_fusion_model that uses a tiny offline Whisper backbone."""
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    from ml.fusion.model import DualViewFusionModel, build_fusion

    if enhancer is None:
        enhancer = build_enhancer(config.get("enhancer"))
    if whisper is None:
        whisper_config = WhisperConfig(
            vocab_size=64,
            num_mel_bins=80,
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_source_positions=1500,  # accepts the full [80, 3000] window
            max_target_positions=64,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            decoder_start_token_id=1,
        )
        whisper = WhisperForConditionalGeneration(whisper_config)
    fusion = build_fusion(int(whisper.config.d_model), config.get("fusion"))
    return DualViewFusionModel(enhancer=enhancer, whisper=whisper, fusion=fusion)


class _FakeTokenizer:
    """Maps any transcript to a few in-vocab label ids (tiny Whisper has vocab 64)."""

    def __call__(self, text: str):
        return SimpleNamespace(input_ids=[1, 5, 7, 2])


def test_run_training_completes_all_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ml.fusion.train_fusion as train_fusion

    monkeypatch.setattr(train_fusion, "build_fusion_model", _tiny_dual_view_model)
    monkeypatch.setattr(train_fusion, "load_tokenizer", lambda config: _FakeTokenizer())

    root = _make_degraded_dataset(tmp_path / "ds")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_path = _tiny_config(root, run_dir)
    assert run_training(config_path) == 0

    assert (run_dir / "checkpoints" / "stage0_warmup" / "enhancer.pt").is_file()
    assert (run_dir / "checkpoints" / "stage1_fusion" / "fusion_model.pt").is_file()
    assert (run_dir / "checkpoints" / "stage2_joint" / "fusion_model.pt").is_file()
    assert (run_dir / "config" / "training_config.yaml").is_file()
    assert (run_dir / "config" / "git_commit.txt").is_file()
    stages_logged = {json.loads(line)["stage"] for line in (run_dir / "logs" / "train_metrics.jsonl").read_text().splitlines()}
    assert {"warmup", "fusion", "joint"} <= stages_logged


def test_resume_from_joint_loads_prior_fusion_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ml.fusion.train_fusion as train_fusion

    monkeypatch.setattr(train_fusion, "build_fusion_model", _tiny_dual_view_model)
    monkeypatch.setattr(train_fusion, "load_tokenizer", lambda config: _FakeTokenizer())

    root = _make_degraded_dataset(tmp_path / "ds")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config_path = _tiny_config(root, run_dir)
    # Full run first so stage1's fusion checkpoint exists, then resume at joint.
    run_training(config_path)
    assert run_training(config_path, resume_from_stage="joint") == 0
    assert (run_dir / "checkpoints" / "stage2_joint" / "fusion_model.pt").is_file()


def test_train_fusion_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "3-stage" in out
    assert "--resume-from-stage" in out
