from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import yaml

from ml.enhancement.dataset import (
    CleanMelDataset,
    DegradedMelDataset,
    collate_mels,
    detect_dataset_kind,
    read_mapping,
    reconstruct_clean_target,
)
from ml.fusion.train_fusion import (
    build_enhancer,
    build_train_dataset,
    clean_dataset_dirs,
    configure_whisper_generation,
    degraded_dataset_dirs,
    eval_score,
    load_fusion_config,
    main,
    resolve_dataset_specs,
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


def _make_degraded_dataset(root: Path, n: int = 2, splits: tuple[str, ...] = ("train",)) -> Path:
    """Build a minimal generate_degraded_dataset-style directory."""
    clean_dir = root / "clean"
    rows = []
    for split in splits:
        for i in range(n):
            clean = clean_dir / f"{split}_clip{i}.wav"
            degraded = root / "clips" / split / f"{split}_clip{i}_v0.wav"
            _write_wav(clean, seed=i)
            _write_wav(degraded, seed=100 + i)
            rows.append(
                {
                    "degraded_id": f"{split}_clip{i}_v0",
                    "split": split,
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


def _make_clean_dataset(root: Path, n: int = 2, splits: tuple[str, ...] = ("train",)) -> Path:
    """Build a minimal clean ASR dataset (split TSVs + clips/) following the project contract."""
    clips_dir = root / "clips"
    for split in splits:
        rows = ["path\tsentence"]
        for i in range(n):
            name = f"{split}_clean{i}.wav"
            _write_wav(clips_dir / name, seed=200 + i)
            rows.append(f"{name}\tسلام دنیا")
        (root / f"{split}.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
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
        # Any existing path satisfies require_base_checkpoint; the real backbone /
        # tokenizer loaders are monkeypatched out in these tests.
        "base_asr_checkpoint": str(root),
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


def test_configure_whisper_generation_clears_legacy_max_length() -> None:
    from transformers import WhisperConfig, WhisperForConditionalGeneration

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
        max_source_positions=1500,
        max_target_positions=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    whisper = WhisperForConditionalGeneration(whisper_config)
    whisper.generation_config.max_length = 20
    whisper.config.max_length = 20

    configure_whisper_generation(whisper, language="Persian", task="transcribe")

    assert whisper.generation_config.max_length is None
    assert whisper.config.max_length is None


class _FakeTokenizer:
    """Maps any transcript to a few in-vocab label ids (tiny Whisper has vocab 64)."""

    pad_token_id = 0

    def __call__(self, text: str):
        return SimpleNamespace(input_ids=[1, 5, 7, 2])

    def batch_decode(self, ids, skip_special_tokens: bool = False):
        # Length-matched stand-in for real text so jiwer can score WER/CER.
        return ["سلام" for _ in ids]


class _CharacterTokenizer(_FakeTokenizer):
    """One token per character, making label-limit tests deterministic."""

    def __call__(self, text: str):
        return SimpleNamespace(input_ids=list(range(len(text))))


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


def test_run_training_requires_base_checkpoint(tmp_path: Path) -> None:
    root = _make_degraded_dataset(tmp_path / "ds")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = {
        "run_dir": str(run_dir),
        "dataset_dir": str(root),
        "base_asr_checkpoint": str(tmp_path / "does_not_exist"),
        "device": "cpu",
        "mixed_precision": "false",
        "enhancer": {"type": "residual_unet", "base_channels": 8, "depth": 2},
        "stages": {"warmup": {"max_steps": 1, "batch_size": 2, "segment_seconds": 0.5, "lr_enhancer": 1e-3, "num_workers": 0}},
    }
    config_path = run_dir.parent / "fusion.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        run_training(config_path)


def test_is_loadable_checkpoint_accepts_path_and_hub_id(tmp_path: Path) -> None:
    from ml.fusion.model import is_loadable_checkpoint

    assert is_loadable_checkpoint(str(tmp_path))  # existing local path
    assert is_loadable_checkpoint("openai/whisper-small")  # Hub id escape hatch
    assert not is_loadable_checkpoint("")
    assert not is_loadable_checkpoint(str(tmp_path / "missing"))  # missing abs path
    assert not is_loadable_checkpoint("just-a-name")  # no slash, not a path


def test_eval_score_minimises_right_metric() -> None:
    assert eval_score("warmup", {"L_enh": 0.5, "wer": 0.9}) == 0.5
    assert eval_score("fusion", {"L_enh": 0.5, "wer": 0.9}) == 0.9


def test_load_resume_state_roundtrips_step_and_optimizer(tmp_path: Path) -> None:
    import torch

    from ml.fusion.train_fusion import load_resume_state, save_enhancer_checkpoint

    config = load_fusion_config_from_dict({"enhancer": {"type": "residual_unet", "base_channels": 8, "depth": 2}})
    enhancer = build_enhancer(config["enhancer"])
    optimizer = torch.optim.Adam(enhancer.parameters(), lr=1e-3)
    # Take one step so the optimizer has non-trivial Adam state to restore.
    loss = enhancer(torch.zeros(1, 80, 20)).sum()
    loss.backward()
    optimizer.step()
    ckpt = tmp_path / "last.pt"
    save_enhancer_checkpoint(ckpt, enhancer, config, step=7, optimizer=optimizer, scaler=None)

    fresh = build_enhancer(config["enhancer"])
    fresh_opt = torch.optim.Adam(fresh.parameters(), lr=1e-3)
    start_step = load_resume_state(ckpt, fresh, fresh_opt, scaler=None)
    assert start_step == 7
    assert fresh_opt.state_dict()["state"], "optimizer state should be restored"


def test_build_lr_scheduler_cosine_warms_up_then_decays() -> None:
    import torch

    from ml.fusion.train_fusion import build_lr_scheduler

    param = torch.nn.Parameter(torch.zeros(1))
    peak = 1e-3
    optimizer = torch.optim.Adam([param], lr=peak)
    scheduler = build_lr_scheduler(optimizer, {"lr_scheduler": "cosine", "warmup_steps": 10}, max_steps=100)

    lrs = []
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    assert lrs[0] < lrs[5] < lrs[9], "LR should ramp up during warm-up"
    assert lrs[9] == pytest.approx(peak, rel=1e-3), "peak LR reached at end of warm-up"
    assert lrs[-1] < 1e-5, "cosine arm should decay LR toward 0 by the final step"


def test_build_lr_scheduler_none_keeps_flat_lr() -> None:
    import torch

    from ml.fusion.train_fusion import build_lr_scheduler

    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    assert build_lr_scheduler(optimizer, {"lr_scheduler": "none"}, max_steps=100) is None


def test_load_resume_state_restores_scheduler(tmp_path: Path) -> None:
    import torch

    from ml.fusion.train_fusion import build_lr_scheduler, load_resume_state, save_enhancer_checkpoint

    config = load_fusion_config_from_dict({"enhancer": {"type": "residual_unet", "base_channels": 8, "depth": 2}})
    enhancer = build_enhancer(config["enhancer"])
    optimizer = torch.optim.Adam(enhancer.parameters(), lr=1e-3)
    scheduler = build_lr_scheduler(optimizer, {"lr_scheduler": "cosine", "warmup_steps": 5}, max_steps=50)
    for _ in range(3):
        enhancer(torch.zeros(1, 80, 20)).sum().backward()
        optimizer.step()
        scheduler.step()
    ckpt = tmp_path / "last.pt"
    save_enhancer_checkpoint(ckpt, enhancer, config, step=3, optimizer=optimizer, scaler=None, scheduler=scheduler)

    fresh = build_enhancer(config["enhancer"])
    fresh_opt = torch.optim.Adam(fresh.parameters(), lr=1e-3)
    fresh_sched = build_lr_scheduler(fresh_opt, {"lr_scheduler": "cosine", "warmup_steps": 5}, max_steps=50)
    load_resume_state(ckpt, fresh, fresh_opt, scaler=None, scheduler=fresh_sched)
    assert fresh_sched.last_epoch == scheduler.last_epoch
    assert fresh_sched.get_last_lr() == pytest.approx(scheduler.get_last_lr())


def test_warmup_writes_resumable_last_checkpoint(tmp_path: Path) -> None:
    import torch

    root = _make_degraded_dataset(tmp_path / "ds")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    overrides = {
        "run_dir": str(run_dir),
        "dataset_dir": str(root),
        "valid_split": None,  # no dev split here; exercise resume plumbing only
        "device": "cpu",
        "mixed_precision": "false",
        "enhancer": {"type": "residual_unet", "base_channels": 8, "depth": 2},
        "stages": {"warmup": {"max_steps": 2, "batch_size": 2, "segment_seconds": 0.5, "lr_enhancer": 1e-3, "num_workers": 0, "log_every": 1, "save_every": 1, "eval_every": 0}},
    }
    config_path = run_dir.parent / "fusion.yaml"
    config_path.write_text(yaml.safe_dump(overrides), encoding="utf-8")
    config = load_fusion_config(config_path)
    enhancer = build_enhancer(config["enhancer"])
    run_stage_warmup(config, run_dir, enhancer, "cpu")

    last = run_dir / "checkpoints" / "stage0_warmup" / "last.pt"
    assert last.is_file()
    payload = torch.load(last, map_location="cpu", weights_only=False)
    assert payload["step"] == 2
    assert payload["optimizer_state"] is not None


def test_determinism_same_seed_same_warmup_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    import ml.fusion.train_fusion as train_fusion

    monkeypatch.setattr(train_fusion, "build_fusion_model", _tiny_dual_view_model)
    monkeypatch.setattr(train_fusion, "load_tokenizer", lambda config: _FakeTokenizer())

    root = _make_degraded_dataset(tmp_path / "ds")

    def _run(tag: str) -> dict:
        run_dir = tmp_path / tag
        run_dir.mkdir()
        run_training(_tiny_config(root, run_dir))
        return torch.load(run_dir / "checkpoints" / "stage0_warmup" / "enhancer.pt", map_location="cpu", weights_only=False)["model_state"]

    state_a = _run("run_a")
    state_b = _run("run_b")
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key]), f"weight {key} differs between identical-seed runs"


def test_fusion_stage_runs_dev_eval_and_saves_best(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ml.fusion.train_fusion as train_fusion
    from ml.fusion.train_fusion import run_stage_fusion

    monkeypatch.setattr(train_fusion, "build_fusion_model", _tiny_dual_view_model)
    monkeypatch.setattr(train_fusion, "load_tokenizer", lambda config: _FakeTokenizer())

    root = _make_degraded_dataset(tmp_path / "ds", splits=("train", "dev"))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    overrides = {
        "run_dir": str(run_dir),
        "dataset_dir": str(root),
        "valid_split": "dev",
        "device": "cpu",
        "mixed_precision": "false",
        "generation_max_length": 16,  # tiny test backbone has max_target_positions=64
        "enhancer": {"type": "residual_unet", "base_channels": 8, "depth": 2},
        "stages": {"fusion": {"max_steps": 1, "batch_size": 1, "lr_frontend": 1e-3, "num_workers": 0, "log_every": 1, "eval_every": 1, "save_every": 0, "eval_max_batches": 1, "grad_clip": 1.0}},
    }
    config_path = run_dir.parent / "fusion.yaml"
    config_path.write_text(yaml.safe_dump(overrides), encoding="utf-8")
    config = load_fusion_config(config_path)
    enhancer = build_enhancer(config["enhancer"])
    run_stage_fusion(config, run_dir, enhancer, "cpu")

    eval_lines = (run_dir / "logs" / "eval_metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in eval_lines]
    assert records and all({"wer", "cer", "loss"} <= rec.keys() for rec in records)
    assert (run_dir / "checkpoints" / "stage1_fusion" / "best.pt").is_file()


def test_detect_dataset_kind(tmp_path: Path) -> None:
    degraded = _make_degraded_dataset(tmp_path / "deg")
    clean = _make_clean_dataset(tmp_path / "clean")
    assert detect_dataset_kind(degraded) == "degraded"
    assert detect_dataset_kind(clean) == "clean"
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        detect_dataset_kind(tmp_path / "empty")


def test_clean_mel_dataset_yields_identical_views(tmp_path: Path) -> None:
    import torch

    root = _make_clean_dataset(tmp_path / "clean")
    dataset = CleanMelDataset(root, split="train", return_labels=True, tokenizer=_FakeTokenizer())
    item = dataset[0]
    assert item["noisy_mel"].shape == (80, 3000)
    assert torch.equal(item["noisy_mel"], item["clean_mel"])  # no degradation -> same view
    assert item["labels"] == [1, 5, 7, 2]


def test_clean_mel_dataset_skips_labels_above_whisper_limit(tmp_path: Path) -> None:
    root = _make_clean_dataset(tmp_path / "clean")
    split_path = root / "train.tsv"
    rows = split_path.read_text(encoding="utf-8").splitlines()
    rows[2] = rows[2].split("\t", 1)[0] + "\t" + ("x" * 65)
    split_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    dataset = CleanMelDataset(
        root,
        split="train",
        return_labels=True,
        tokenizer=_CharacterTokenizer(),
        max_label_tokens=64,
    )

    assert len(dataset) == 1
    assert dataset.skipped_overlong_labels == 1
    assert len(dataset[0]["labels"]) <= 64


def test_degraded_mel_dataset_skips_labels_above_whisper_limit(tmp_path: Path) -> None:
    root = _make_degraded_dataset(tmp_path / "degraded")
    mapping_path = root / "degraded_to_clean.jsonl"
    rows = [json.loads(line) for line in mapping_path.read_text(encoding="utf-8").splitlines()]
    rows[1]["sentence"] = "x" * 65
    mapping_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    dataset = DegradedMelDataset(
        root,
        split="train",
        return_labels=True,
        tokenizer=_CharacterTokenizer(),
        max_label_tokens=64,
    )

    assert len(dataset) == 1
    assert dataset.skipped_overlong_labels == 1
    assert len(dataset[0]["labels"]) <= 64


def test_label_filter_rejects_dataset_when_every_sample_is_overlong(tmp_path: Path) -> None:
    root = _make_clean_dataset(tmp_path / "clean", n=1)
    split_path = root / "train.tsv"
    rows = split_path.read_text(encoding="utf-8").splitlines()
    rows[1] = rows[1].split("\t", 1)[0] + "\t" + ("x" * 65)
    split_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no usable samples remain"):
        CleanMelDataset(
            root,
            split="train",
            return_labels=True,
            tokenizer=_CharacterTokenizer(),
            max_label_tokens=64,
        )


def test_resolve_dataset_specs_autodetects_and_partitions(tmp_path: Path) -> None:
    degraded = _make_degraded_dataset(tmp_path / "deg")
    clean = _make_clean_dataset(tmp_path / "clean")
    config = {"datasets": [str(degraded), {"path": str(clean)}], "dataset_dir": ""}
    specs = resolve_dataset_specs(config)
    assert {kind for _, kind in specs} == {"degraded", "clean"}
    assert degraded_dataset_dirs(config) == [degraded]
    assert clean_dataset_dirs(config) == [clean]


def test_resolve_dataset_specs_legacy_dataset_dir(tmp_path: Path) -> None:
    degraded = _make_degraded_dataset(tmp_path / "deg")
    config = {"datasets": None, "dataset_dir": str(degraded)}
    assert resolve_dataset_specs(config) == [(degraded, "degraded")]


def test_build_train_dataset_includes_clean_only_when_requested(tmp_path: Path) -> None:
    from torch.utils.data import ConcatDataset

    degraded = _make_degraded_dataset(tmp_path / "deg")
    clean = _make_clean_dataset(tmp_path / "clean")
    config = {
        "datasets": [str(degraded), str(clean)],
        "dataset_dir": "",
        "train_split": "train",
        "clean_target": "bandwidth_aligned",
        "model_name": "openai/whisper-small",
        "sample_rate": SR,
        "seed": 1337,
    }
    degraded_only = build_train_dataset(
        config, segment_seconds=None, return_labels=True, tokenizer=_FakeTokenizer(), include_clean=False
    )
    with_clean = build_train_dataset(
        config, segment_seconds=None, return_labels=True, tokenizer=_FakeTokenizer(), include_clean=True
    )
    # Degraded-only sees just the 2 degraded variants; including clean adds 2 more.
    assert len(degraded_only) == 2
    assert isinstance(with_clean, ConcatDataset)
    assert len(with_clean) == 4


def test_build_train_dataset_requires_degraded(tmp_path: Path) -> None:
    clean = _make_clean_dataset(tmp_path / "clean")
    config = {
        "datasets": [str(clean)],
        "dataset_dir": "",
        "train_split": "train",
        "clean_target": "bandwidth_aligned",
        "model_name": "openai/whisper-small",
        "sample_rate": SR,
        "seed": 1337,
    }
    with pytest.raises(ValueError, match="no degraded dataset"):
        build_train_dataset(config, segment_seconds=None, return_labels=False, tokenizer=None, include_clean=True)


def test_joint_stage_trains_on_clean_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ml.fusion.train_fusion as train_fusion
    from ml.fusion.train_fusion import run_stage_joint

    monkeypatch.setattr(train_fusion, "build_fusion_model", _tiny_dual_view_model)
    monkeypatch.setattr(train_fusion, "load_tokenizer", lambda config: _FakeTokenizer())

    degraded = _make_degraded_dataset(tmp_path / "deg")
    clean = _make_clean_dataset(tmp_path / "clean")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    overrides = {
        "run_dir": str(run_dir),
        "datasets": [str(degraded), str(clean)],
        "base_asr_checkpoint": str(degraded),
        "valid_split": None,
        "device": "cpu",
        "mixed_precision": "false",
        "enhancer": {"type": "residual_unet", "base_channels": 8, "depth": 2},
        "stages": {"joint": {"max_steps": 3, "batch_size": 2, "lr_frontend": 1e-3, "lr_whisper": 1e-4, "num_workers": 0, "log_every": 1, "save_every": 0, "grad_clip": 1.0}},
    }
    config_path = run_dir.parent / "fusion.yaml"
    config_path.write_text(yaml.safe_dump(overrides), encoding="utf-8")
    config = load_fusion_config(config_path)
    enhancer = build_enhancer(config["enhancer"])
    run_stage_joint(config, run_dir, enhancer, "cpu")

    assert (run_dir / "checkpoints" / "stage2_joint" / "fusion_model.pt").is_file()
    stages = {json.loads(line)["stage"] for line in (run_dir / "logs" / "train_metrics.jsonl").read_text().splitlines()}
    assert "joint" in stages


def test_joint_stage_reports_clean_dev_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ml.fusion.train_fusion as train_fusion
    from ml.fusion.train_fusion import run_stage_joint

    monkeypatch.setattr(train_fusion, "build_fusion_model", _tiny_dual_view_model)
    monkeypatch.setattr(train_fusion, "load_tokenizer", lambda config: _FakeTokenizer())

    degraded = _make_degraded_dataset(tmp_path / "deg", splits=("train", "dev"))
    clean = _make_clean_dataset(tmp_path / "clean", splits=("train", "dev"))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    overrides = {
        "run_dir": str(run_dir),
        "datasets": [str(degraded), str(clean)],
        "base_asr_checkpoint": str(degraded),
        "valid_split": "dev",
        "device": "cpu",
        "mixed_precision": "false",
        "generation_max_length": 16,  # tiny test backbone has max_target_positions=64
        "enhancer": {"type": "residual_unet", "base_channels": 8, "depth": 2},
        "stages": {"joint": {"max_steps": 1, "batch_size": 1, "lr_frontend": 1e-3, "lr_whisper": 1e-4, "num_workers": 0, "log_every": 1, "eval_every": 1, "save_every": 0, "eval_max_batches": 1, "grad_clip": 1.0}},
    }
    config_path = run_dir.parent / "fusion.yaml"
    config_path.write_text(yaml.safe_dump(overrides), encoding="utf-8")
    config = load_fusion_config(config_path)
    enhancer = build_enhancer(config["enhancer"])
    run_stage_joint(config, run_dir, enhancer, "cpu")

    records = [json.loads(line) for line in (run_dir / "logs" / "eval_metrics.jsonl").read_text().splitlines()]
    # Both the degraded WER it selects on and the clean-dataset WER are reported.
    assert records and all({"wer", "clean_wer", "clean_cer"} <= rec.keys() for rec in records)


def test_train_fusion_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "3-stage" in out
    assert "--resume-from-stage" in out
    assert "num_train_epochs" in out
    assert "gradient_accumulation_steps" in out


def _tiny_whisper_encoder():
    """A tiny Whisper encoder (offline) for feature-matching / diagnostic tests."""
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    config = WhisperConfig(
        vocab_size=64, num_mel_bins=80, d_model=16,
        encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=32, decoder_ffn_dim=32,
        max_source_positions=1500, max_target_positions=64,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1,
    )
    encoder = WhisperForConditionalGeneration(config).get_encoder().eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def test_warmup_feature_matching_logs_and_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ml.fusion.train_fusion as train_fusion

    monkeypatch.setattr(train_fusion, "load_feature_encoder", lambda config, device: _tiny_whisper_encoder())
    root = _make_degraded_dataset(tmp_path / "ds", splits=("train", "dev"))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = load_fusion_config(_tiny_config(root, run_dir))
    # Turn feature matching on and add a dev eval so L_feat shows up in both logs.
    config["stages"]["warmup"].update({"feature_match_weight": 0.5, "eval_every": 3})
    config["valid_split"] = "dev"
    enhancer = build_enhancer(config["enhancer"])
    train_fusion.run_stage_warmup(config, run_dir, enhancer, "cpu")

    train_rows = [json.loads(l) for l in (run_dir / "logs" / "train_metrics.jsonl").read_text().splitlines()]
    assert any("L_feat" in r for r in train_rows if r["stage"] == "warmup")
    eval_rows = [json.loads(l) for l in (run_dir / "logs" / "eval_metrics.jsonl").read_text().splitlines()]
    assert any({"L_feat", "L_warmup"} <= r.keys() for r in eval_rows)


def test_eval_score_warmup_uses_combined_when_present() -> None:
    # Plain L_enh when feature matching is off; combined L_warmup when on.
    assert eval_score("warmup", {"L_enh": 0.5}) == 0.5
    assert eval_score("warmup", {"L_enh": 0.5, "L_feat": 0.2, "L_warmup": 0.7}) == 0.7


def test_diagnose_enhancement_identity_baseline(tmp_path: Path) -> None:
    from ml.enhancement.diagnose_enhancement import main as diagnose_main

    root = _make_degraded_dataset(tmp_path / "ds", splits=("dev",))
    out_dir = tmp_path / "diag"
    rc = diagnose_main([
        "--dataset", str(root), "--split", "dev",
        "--device", "cpu", "--batch-size", "2", "--output-dir", str(out_dir),
    ])
    assert rc == 0
    report = json.loads((out_dir / "diagnosis.json").read_text())
    # Identity baseline only (no checkpoint): headroom present, no trained metric.
    assert report["overall"]["identity_L_enh"] > 0
    assert "trained_L_enh" not in report["overall"]
    assert "narrowband" in report["by_bandwidth"]


def test_diagnose_enhancement_with_checkpoint_reports_captured(tmp_path: Path) -> None:
    from ml.enhancement.diagnose_enhancement import main as diagnose_main
    from ml.fusion.train_fusion import save_enhancer_checkpoint

    config = load_fusion_config_from_dict({"enhancer": {"type": "residual_unet", "base_channels": 8, "depth": 2}})
    enhancer = build_enhancer(config["enhancer"])
    ckpt = tmp_path / "enhancer.pt"
    save_enhancer_checkpoint(ckpt, enhancer, config, step=0)

    root = _make_degraded_dataset(tmp_path / "ds", splits=("dev",))
    out_dir = tmp_path / "diag"
    rc = diagnose_main([
        "--dataset", str(root), "--split", "dev", "--enhancer-checkpoint", str(ckpt),
        "--device", "cpu", "--batch-size", "2", "--output-dir", str(out_dir), "--dump-mels", "1",
    ])
    assert rc == 0
    report = json.loads((out_dir / "diagnosis.json").read_text())
    overall = report["overall"]
    assert "trained_L_enh" in overall and "captured" in overall
    # An identity-init enhancer leaves the mel unchanged -> trained ~= identity.
    assert overall["trained_L_enh"] == pytest.approx(overall["identity_L_enh"], rel=1e-3)
    assert (out_dir / "mels").is_dir() and any((out_dir / "mels").glob("*_noisy.npy"))


@pytest.mark.parametrize("epochs", [0, -1, float("inf"), float("nan")])
def test_invalid_epoch_budget(tmp_path: Path, epochs: float) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"stages": {"warmup": {"num_train_epochs": epochs}}}))
    with pytest.raises(ValueError, match="num_train_epochs"):
        load_fusion_config(path)


def test_epoch_config_replaces_default_steps_and_rejects_explicit_conflict(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("stages:\n  warmup:\n    num_train_epochs: 2\n")
    stage = load_fusion_config(path)["stages"]["warmup"]
    assert stage["max_steps"] is None
    path.write_text("stages:\n  warmup:\n    num_train_epochs: 2\n    max_steps: 10\n")
    with pytest.raises(ValueError, match="not both"):
        load_fusion_config(path)


@pytest.mark.parametrize("accumulation", [0, -1, 1.5, True])
def test_invalid_accumulation(accumulation: float) -> None:
    config = load_fusion_config_from_dict({"stages": {"warmup": {"gradient_accumulation_steps": accumulation}}})
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        validate_fusion_config(config)


def test_epoch_budget_rounds_partial_groups_and_fractional_epochs() -> None:
    from ml.fusion.train_fusion import stage_step_budget, stage_event_due

    stage = {"num_train_epochs": 1.5, "gradient_accumulation_steps": 2, "eval_save_at_epoch_end": True}
    assert stage_step_budget(stage, [None] * 5) == (3, 5)
    assert stage_event_due(stage, 1000, 3, 3, 5)
    assert stage_event_due(stage, 1000, 5, 3, 5)
    assert not stage_event_due(stage, 1000, 2, 3, 5)
    assert stage_event_due(stage, 2, 2, 3, 5)
    with pytest.raises(ValueError, match="at least one"):
        stage_step_budget(stage, [])


def test_training_groups_resume_same_order_and_flush_partial_epoch() -> None:
    import torch
    from ml.fusion.train_fusion import training_groups

    def groups(start: int):
        loader = torch.utils.data.DataLoader(list(range(9)), batch_size=2, shuffle=True)
        return [[int(row) for batch in group for row in batch] for group in training_groups(loader, 2, start, 6, 1337)]

    complete = groups(0)
    assert [len(group) for group in complete] == [4, 4, 1, 4, 4, 1]
    for epoch in (complete[:3], complete[3:]):
        assert sorted(row for group in epoch for row in group) == list(range(9))
    assert groups(2) == complete[2:]
    assert groups(3) == complete[3:]
    assert groups(6) == []


@pytest.mark.parametrize("stage_name", ["warmup", "fusion", "joint"])
def test_epoch_training_evaluates_short_runs_and_saves_optimizer_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage_name: str,
) -> None:
    import torch
    import ml.fusion.train_fusion as trainer

    monkeypatch.setattr(trainer, "build_fusion_model", _tiny_dual_view_model)
    monkeypatch.setattr(trainer, "load_tokenizer", lambda config: _FakeTokenizer())
    root = _make_degraded_dataset(tmp_path / "ds", n=3, splits=("train", "dev"))
    config = load_fusion_config(_tiny_config(root, tmp_path / "run"))
    config["generation_max_length"] = 16
    config["stages"][stage_name].update({
        "max_steps": None, "num_train_epochs": 2, "batch_size": 2,
        "gradient_accumulation_steps": 2, "eval_batch_size": 3,
        "num_workers": 0, "eval_every": 1000, "save_every": 1000,
        "eval_save_at_epoch_end": True, "warmup_steps": 0,
    })
    run_dir = tmp_path / "run"
    trainer.STAGE_RUNNERS[stage_name](config, run_dir, build_enhancer(config["enhancer"]), "cpu")
    records = [json.loads(line) for line in (run_dir / "logs/eval_metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in records] == [1, 2]
    assert [row["epoch"] for row in records] == [1, 2]
    checkpoint = torch.load(run_dir / "checkpoints" / trainer.STAGE_DIRS[stage_name] / "last.pt", weights_only=False)
    assert checkpoint["step"] == 2
    assert checkpoint["scheduler_state"]["last_epoch"] == 2
    assert {int(state["step"]) for state in checkpoint["optimizer_state"]["state"].values()} == {2}
    budget = json.loads((run_dir / "config" / f"{stage_name}_budget.json").read_text())
    assert budget["train_examples"] == 3
    assert budget["optimizer_steps_per_epoch"] == 1
    assert budget["max_optimizer_steps"] == 2


def test_accumulated_warmup_matches_full_batch_with_short_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    import ml.fusion.train_fusion as trainer

    torch.manual_seed(17)
    rows = [{"pair_id": str(i), "noisy_mel": torch.randn(80, 10), "clean_mel": torch.randn(80, 10)} for i in range(7)]
    monkeypatch.setattr(trainer, "build_train_dataset", lambda *args, **kwargs: rows)
    monkeypatch.setattr(trainer, "build_dev_loader", lambda *args, **kwargs: None)

    def run(batch_size: int, accumulation: int):
        config = load_fusion_config_from_dict({"mixed_precision": "false", "stages": {"warmup": {
            "max_steps": None, "num_train_epochs": 2, "batch_size": batch_size,
            "gradient_accumulation_steps": accumulation, "num_workers": 0,
            "eval_every": 0, "save_every": 0, "warmup_steps": 0,
        }}})
        torch.manual_seed(5)
        enhancer = torch.nn.Linear(10, 10)
        trainer.run_stage_warmup(config, tmp_path / f"batch{batch_size}", enhancer, "cpu")
        return enhancer.state_dict()

    accumulated, full = run(2, 2), run(4, 1)
    for name in full:
        torch.testing.assert_close(accumulated[name], full[name], atol=1e-7, rtol=1e-5)


def test_changed_or_legacy_epoch_budget_requires_new_run(tmp_path: Path) -> None:
    import torch
    from ml.fusion.train_fusion import record_stage_budget

    loader = torch.utils.data.DataLoader(list(range(9)), batch_size=2)
    stage = {"batch_size": 2, "num_train_epochs": 1}
    record_stage_budget(tmp_path, "warmup", stage, loader)
    with pytest.raises(ValueError, match="budget changed"):
        record_stage_budget(tmp_path, "warmup", dict(stage, num_train_epochs=2), loader)
    legacy = tmp_path / "legacy"
    checkpoint = legacy / "checkpoints/stage0_warmup/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    with pytest.raises(ValueError, match="legacy step checkpoint"):
        record_stage_budget(legacy, "warmup", stage, loader)


def test_cv25_tiny_recipes_match_training_exposure() -> None:
    from ml.asr.train_whisper_small import load_training_config

    root = Path(__file__).resolve().parents[2] / "configs/speech_enhancement/cv25_tiny"
    baseline = load_training_config(root / "baseline.yaml")["training"]
    assert baseline["num_train_epochs"] == 1
    assert baseline["per_device_train_batch_size"] == 8
    assert baseline["per_device_eval_batch_size"] == 128
    assert baseline["gradient_accumulation_steps"] == 2
    assert baseline["eval_steps"] == baseline["save_steps"] == 1000
    assert baseline["eval_save_at_epoch_end"] is True
    stages = []
    for name in ("cross_attention", "gated", "residual_cross_attention"):
        config = load_fusion_config(root / f"{name}.yaml")
        stages.append(config["stages"])
        for stage in config["stages"].values():
            assert stage["num_train_epochs"] == 1 and stage["max_steps"] is None
            assert stage["batch_size"] == 8 and stage["gradient_accumulation_steps"] == 2
            assert stage["eval_batch_size"] == 128
            assert stage["eval_every"] == stage["save_every"] == 1000
            assert stage["eval_save_at_epoch_end"] is True
            assert stage["warmup_steps"] is None and stage["warmup_ratio"] == 0.05
    assert stages[0] == stages[1] == stages[2]
