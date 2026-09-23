"""Whisper Small bridge dataset, config, and CLI contracts without model downloads."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from ml.fusion.bridge_small_data import collect_small_rows, input_config, load_training_config
from ml.fusion.eval_bridge_small import load_eval_config, suite_config, write_comparison


def _audio(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.full(1600, 0.1, dtype=np.float32), 16000)


def _configs(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "data"
    for name in ("clean", "other"):
        for split in ("train", "dev", "test"):
            _audio(root / name / "clips" / f"{split}.wav")
            (root / name / f"{split}.tsv").write_text(
                f"path\tsentence\n{split}.wav\thello {split}\n")
    degraded = root / "degraded"
    degraded.mkdir()
    rows = []
    for split in ("train", "dev"):
        _audio(degraded / "clips" / f"{split}.wav")
        rows.append({"degraded_id": split, "split": split, "sentence": f"hello {split}",
                     "degraded_path": f"clips/{split}.wav",
                     "clean_path": str(root / "clean" / "clips" / f"{split}.wav")})
    (degraded / "degraded_to_clean.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    train = {
        "model": {"asr_checkpoint": "small/best", "bridge_checkpoint": "bridge/best.pt"},
        "data": {"root_dir": str(root), "datasets": ["clean", "degraded", "other"],
                 "test_datasets": ["clean", "other"]},
        "run": {"inputs_dir": str(tmp_path / "inputs"), "cache_dir": str(tmp_path / "cache"),
                "output_dir": "bridge"},
        "preparation": {"batch_size": 4, "workers": 1},
        "cache": {"batch_size": 8, "workers": 2},
        "training": {"seed": 1337, "epochs": 5, "learning_rate": .0005,
                     "batch_size": 4, "accumulation": 8, "workers": 1, "eval_max_batches": 4},
    }
    train_path = tmp_path / "train.yaml"
    train_path.write_text(yaml.safe_dump(train))
    evaluate = {
        "model": {**train["model"], "processor": "openai/whisper-small"},
        "data": {"root_dir": str(root), "datasets": ["clean", "other"], "split": "test"},
        "eval": {"inputs_dir": str(tmp_path / "inputs"), "output_dir": str(tmp_path / "evals"),
                 "name": "small-test", "device": "cpu", "max_new_tokens": 225},
    }
    eval_path = tmp_path / "eval.yaml"
    eval_path.write_text(yaml.safe_dump(evaluate))
    return train_path, eval_path, root


def test_small_datasets_use_clean_and_degraded_train_dev_and_test(tmp_path):
    train_path, _, root = _configs(tmp_path)
    skipped = []
    rows = collect_small_rows(input_config(load_training_config(train_path)), "all", skipped)
    assert not skipped
    assert {split: sum(row["split"] == split for row in rows) for split in ("train", "dev", "test")} == {
        "train": 3, "dev": 3, "test": 2}
    degraded = next(row for row in rows if row["dataset"] == "degraded" and row["split"] == "train")
    assert degraded["source_id"] == str((root / "clean/clips/train.wav").resolve())
    assert degraded["noisy_path"] != degraded["source_id"]
    assert all(row["dataset"] != "degraded" for row in rows if row["split"] == "test")


def test_small_datasets_reject_wrong_split_source(tmp_path):
    train_path, _, root = _configs(tmp_path)
    mapping = root / "degraded/degraded_to_clean.jsonl"
    rows = [json.loads(line) for line in mapping.read_text().splitlines()]
    rows[0]["clean_path"] = str(root / "clean/clips/dev.wav")
    mapping.write_text("\n".join(json.dumps(row) for row in rows))
    skipped = []
    result = collect_small_rows(input_config(load_training_config(train_path)), "train-dev", skipped)
    assert [row["reason"] for row in skipped] == ["source_missing_or_wrong_split"]
    assert not any(row["id"] == "degraded/train" for row in result)


def test_small_eval_rejects_incomplete_config(tmp_path):
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump({"data": {"datasets": ["clean"], "split": "test"}}))
    with pytest.raises(ValueError, match="model"):
        load_eval_config(path)


@pytest.mark.parametrize("module", ["ml.fusion.train_bridge_small", "ml.fusion.eval_bridge_small"])
def test_small_cli_help(module):
    result = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_small_input_preparation_and_eval_config_binding(tmp_path, monkeypatch):
    from ml.fusion import prepare_bridge_inputs as prep
    train_path, eval_path, _ = _configs(tmp_path)
    class Enhancer:
        identity = "offline-frcrn"
        def __init__(self, *args): pass
        def __call__(self, wave): return wave * 0.5
    class Scorer:
        identity = "offline-dnsmos"
        def __init__(self, *args): pass
        def __call__(self, wave): return {"dnsmos_sig": 3., "dnsmos_bak": 2.}
    monkeypatch.setattr(prep, "FRCRN", Enhancer)
    monkeypatch.setattr(prep, "DNSMOS", Scorer)
    output = tmp_path / "inputs"
    assert prep.main(["--train-config", str(train_path), "--output", str(output)]) == 0
    train = [json.loads(line) for line in (output / "bridge_train_dev.jsonl").read_text().splitlines()]
    assert len(train) == 6 and all("dnsmos_sig" in row for row in train)
    test = [json.loads(line) for line in (output / "bridge_test.jsonl").read_text().splitlines()]
    assert len(test) == 2 and all("dnsmos_sig" not in row for row in test)
    suite = suite_config(load_eval_config(eval_path))
    assert suite["data"]["datasets"] == ["clean", "other"]
    assert suite["methods"]["bridge"]["checkpoint"] == "bridge/best.pt"
    assert suite["bridge_enhancer_id"] == "offline-frcrn:batch-pad-v1:size-4"
    changed = yaml.safe_load(eval_path.read_text())
    changed["data"]["datasets"] = ["other", "clean"]
    eval_path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="differ"):
        suite_config(load_eval_config(eval_path))


def test_comparison_marks_archived_scores_unmatched(tmp_path):
    output = tmp_path / "results"
    output.mkdir()
    current = {"examples": 2, "wer": .2, "cer": .1,
               "dataset_metrics": {"clean": {"examples": 2, "wer": .2, "cer": .1}}}
    (output / "summary.json").write_text(json.dumps({
        "baseline": current, "bridge": current, "enhanced_only": current}))
    archived = tmp_path / "archive" / "whisper-deg"
    archived.mkdir(parents=True)
    (archived / "metrics.json").write_text(json.dumps({
        "examples": 3, "wer": .3, "cer": .15,
        "dataset_metrics": [{"dataset": "/server/data/clean", "examples": 3, "wer": .3, "cer": .15}]}))
    write_comparison(output, tmp_path / "archive")
    rows = json.loads((output / "comparison.json").read_text())["methods"]
    assert len(rows) == 4
    assert rows[1]["model"] == "current/bridge" and rows[1]["matched_current_test_set"]
    assert rows[-1]["model"] == "archive/whisper-deg" and not rows[-1]["matched_current_test_set"]
    assert "archive/whisper-deg" in (output / "comparison.md").read_text()


def test_small_training_dispatch_is_pq_plus_ri(tmp_path, monkeypatch):
    from ml.fusion import train_bridge_small as experiment
    train_path, _, _ = _configs(tmp_path)
    calls = []
    monkeypatch.setattr(experiment.bridging_experiment, "main", lambda argv: calls.append(argv) or 0)
    assert experiment.main(["train", "--config", str(train_path), "--device", "cpu"]) == 0
    assert calls[0][0] == "train"
    assert "--pq-only" not in calls[0]
    assert calls[0][calls[0].index("--epochs") + 1] == "5"
    assert calls[0][calls[0].index("--output") + 1] == "bridge"
