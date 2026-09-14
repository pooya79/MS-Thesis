from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

from ml.fusion.bridging import (
    BridgingModule, OA_COEFFICIENTS, bridge_loss, observation_addition,
    perceptual_target, recognition_information_loss,
)
from ml.fusion.bridging_experiment import check_splits, main
from ml.fusion.model import build_fusion


def test_waveform_mixing_endpoints_and_direction():
    noisy, enhanced = torch.randn(2, 100), torch.randn(2, 100)
    assert torch.equal(observation_addition(noisy, enhanced, torch.tensor(1.)), noisy)
    assert torch.equal(observation_addition(noisy, enhanced, torch.tensor(0.)), enhanced)
    result = observation_addition(noisy, enhanced, torch.tensor([0., 1.]))
    assert torch.equal(result[0], enhanced[0]) and torch.equal(result[1], noisy[1])
    with pytest.raises(ValueError):
        observation_addition(noisy, enhanced[:, :-1], torch.tensor(.5))
    with pytest.raises(ValueError):
        observation_addition(noisy, enhanced, torch.tensor(float("nan")))
    assert OA_COEFFICIENTS == tuple(i / 10 for i in range(10, -1, -1))


def test_paper_losses_and_detached_targets():
    assert torch.equal(perceptual_target(torch.tensor([1., 5.]), torch.tensor([1., 5.])), torch.tensor([0., 1.]))
    logits = torch.randn(2, 11, requires_grad=True)
    wers = torch.rand(2, 11, requires_grad=True)
    loss = recognition_information_loss(logits, wers)
    expected = -torch.log(torch.sigmoid(torch.nn.functional.cosine_similarity(logits.sigmoid(), wers.sigmoid(), dim=-1))).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None and wers.grad is None
    with pytest.raises(ValueError):
        perceptual_target(torch.tensor(0.), torch.tensor(4.))


def test_bridge_both_heads_receive_gradients():
    model = BridgingModule(channels=16, bottleneck=8, hidden=16)
    out = model(torch.randn(2, 80, 20), torch.randn(2, 80, 20))
    loss = bridge_loss(out, torch.rand(2, 11), torch.tensor([3., 4.]), torch.tensor([2., 3.]))
    loss.backward()
    assert out["omega"].shape == (2,)
    assert model.quality.weight.grad.abs().sum() > 0
    assert model.recognition[0].weight.grad.abs().sum() > 0


def test_residual_bypass_survives_trained_layers_and_invalid_enhancement():
    model = build_fusion(16, {"type": "residual_cross_attention", "num_layers": 1, "num_heads": 2})
    noisy, enhanced = torch.randn(2, 8, 16), torch.randn(2, 8, 16)
    assert torch.equal(model(noisy, enhanced), noisy)  # identity residual initialization
    with torch.no_grad():
        model.layers[0].attn.out_proj.weight.normal_()
    assert not torch.allclose(model(noisy, enhanced, gate_override=1), noisy)
    captured = []
    handle = model.combine.register_forward_hook(lambda *args: captured.append(True))
    assert torch.equal(model(noisy, enhanced * float("nan"), gate_override=0), noisy)
    handle.remove()
    assert captured
    assert model.combine.gate(noisy, enhanced).shape == (2, 1, 1)


def test_split_leakage_rejected():
    with pytest.raises(ValueError, match="leakage"):
        check_splits([{"source_id": "same", "split": "train"}, {"source_id": "same", "split": "dev"}])
    with pytest.raises(ValueError, match="leakage"):
        check_splits([{"source_id": "a", "speaker_id": "x", "split": "train"},
                      {"source_id": "b", "speaker_id": "x", "split": "dev"}])


@pytest.mark.parametrize("command", [[], ["prepare"], ["train"], ["evaluate"]])
def test_cli_help(command):
    result = subprocess.run([sys.executable, "-m", "ml.fusion.bridging_experiment", *command, "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--help" in result.stdout


def test_cached_training_and_checkpoint_reload(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "provenance.json").write_text(json.dumps({"asr_checkpoint": "offline-test", "max_tokens": 20}))
    rows = []
    for i, split in enumerate(("train", "dev")):
        torch.save({"noisy": torch.randn(80, 12), "enhanced": torch.randn(80, 12),
                    "wers": torch.rand(11), "sig": torch.tensor(3.), "bak": torch.tensor(2.)}, cache / f"{i}.pt")
        rows.append({"id": str(i), "source_id": str(i), "split": split, "cache": f"{i}.pt"})
    rows.append({"id": "missing", "source_id": "missing", "split": "train", "cache": "missing.pt"})
    (cache / "index.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    output = tmp_path / "run"
    assert main(["train", "--cache", str(cache), "--output", str(output), "--epochs", "1"]) == 0
    saved = torch.load(output / "best.pt", weights_only=True)
    model = BridgingModule(**saved["model_config"])
    model.load_state_dict(saved["state_dict"])
    assert saved["epoch"] == 1
    skipped = [json.loads(line) for line in (output / "skipped_inputs.jsonl").read_text().splitlines()]
    assert [(row["id"], row["reason"]) for row in skipped] == [("missing", "invalid_cache")]


def test_manifest_reader_skips_malformed_and_duplicate_rows(tmp_path):
    from ml.fusion.bridging_experiment import read_rows
    path = tmp_path / "rows.jsonl"
    row = {"id": "duplicate", "source_id": "source", "split": "train"}
    valid = {"id": "valid", "source_id": "other", "split": "dev"}
    path.write_text("not-json\n" + json.dumps(row) + "\n" + json.dumps(row) + "\n" + json.dumps(valid))
    skipped = []
    assert read_rows(path, skipped) == [valid]
    assert [item["reason"] for item in skipped] == ["invalid_json", "duplicate_id", "duplicate_id"]


def test_prepare_and_evaluate_waveform_workflow(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    import ml.fusion.bridging_experiment as experiment

    calls = []
    class OfflineRecognizer:
        def __init__(self, *args):
            pass

        def transcribe_batch(self, waves):
            calls.extend(wave.clone() for wave in waves)
            return ["hello"] * len(waves)

        def __call__(self, wave):
            return self.transcribe_batch([wave])[0]

    monkeypatch.setattr(experiment, "Recognizer", OfflineRecognizer)
    sf.write(tmp_path / "noisy.wav", np.sin(np.arange(1600) * .1).astype(np.float32) * .1, 16000)
    sf.write(tmp_path / "enhanced.wav", np.zeros(1600, dtype=np.float32), 16000)
    rows = [{"id": str(i), "source_id": str(i), "split": split, "sentence": "hello",
             "noisy_path": "noisy.wav", "enhanced_path": "enhanced.wav", "dnsmos_sig": 3., "dnsmos_bak": 2.}
            for i, split in enumerate(("train", "dev"))]
    rows.append({"id": "missing", "source_id": "missing", "split": "train", "sentence": "hello",
                 "noisy_path": "missing-noisy.wav", "enhanced_path": "missing-enhanced.wav",
                 "dnsmos_sig": 3., "dnsmos_bak": 2.})
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows))
    import hashlib
    manifest.with_suffix(".provenance.json").write_text(json.dumps({
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "enhancer_id": "automatic-frcrn", "dnsmos_id": "automatic-dnsmos"}))
    cache = tmp_path / "cache"
    main(["prepare", "--manifest", str(manifest), "--output", str(cache),
          "--asr-checkpoint", "offline"])
    assert json.loads((cache / "provenance.json").read_text())["enhancer_id"] == "automatic-frcrn"
    skipped = [json.loads(line) for line in (cache / "skipped_inputs.jsonl").read_text().splitlines()]
    assert [row["id"] for row in skipped] == ["missing"]
    assert skipped[0]["reason"] == "invalid_paired_audio"
    assert len(calls) == 22
    assert calls[0].abs().sum() > 0 and calls[10].abs().sum() == 0
    item = torch.load(cache / "00000000.pt", weights_only=True)
    assert item["noisy"].shape[0] == 80 and item["wers"].sum() == 0
    first_record = (cache / "index.jsonl").read_text().splitlines()[0]
    (cache / "index.jsonl").write_text(first_record + "\n")
    (cache / "00000001.pt").unlink()
    calls.clear()
    main(["prepare", "--manifest", str(manifest), "--output", str(cache),
          "--asr-checkpoint", "offline", "--resume", "--batch-size", "1"])
    assert len(calls) == 11
    assert len((cache / "index.jsonl").read_text().splitlines()) == 2
    checkpoint = tmp_path / "model.pt"
    model = BridgingModule(channels=16)
    torch.save({"state_dict": model.state_dict(), "model_config": {"channels": 16},
                "provenance": {"asr_checkpoint": "offline", "max_tokens": 20}}, checkpoint)
    manifest.write_text(json.dumps(rows[1]))
    output = tmp_path / "eval"
    main(["evaluate", "--manifest", str(manifest), "--output", str(output),
          "--checkpoint", str(checkpoint), "--omega", "1"])
    assert json.loads((output / "metrics.json").read_text())["wer"] == 0


def test_tiny_configs_are_cv25_only():
    from pathlib import Path
    import yaml
    from ml.fusion.train_fusion import load_fusion_config
    root = Path("configs/speech_enhancement/cv25_tiny")
    for name in ("cross_attention", "gated", "residual_cross_attention"):
        config = load_fusion_config(root / f"{name}.yaml")
        assert config["model_name"] == "openai/whisper-tiny"
        assert set(config["datasets"]) == {"data/cv25-official/cv-corpus-25.0", "data/cv25-official/cv-corpus-25.0-degraded-v2"}
        build_fusion(384, config["fusion"])
    baseline = yaml.safe_load((root / "baseline.yaml").read_text())
    assert baseline["data"]["root_dir"] == "data/cv25-official"
    assert baseline["run"]["output_dir"] == "models/asr/cv25-tiny-official"
    assert baseline["model"]["name"] == "openai/whisper-tiny"
    assert set(baseline["data"]["datasets"]) == {"cv-corpus-25.0", "cv-corpus-25.0-degraded-v2"}
