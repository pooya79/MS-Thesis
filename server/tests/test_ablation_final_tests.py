import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import ml.fusion.evaluate_ablation as evaluation


def test_suite_has_original_test_sets_and_every_method():
    config = evaluation.load_config(Path("configs/speech_enhancement/cv25_tiny/final_tests.yaml"))
    assert config["data"]["root_dir"] == "data/cv25-official"
    assert config["data"]["datasets"] == list(evaluation.TEST_DATASETS)
    assert {"baseline", "cross_attention", "gated", "residual_cross_attention", "bridge", "bridge_pq"} <= config["methods"].keys()
    import yaml
    for name in ("cross_attention", "gated", "residual_cross_attention"):
        train = yaml.safe_load(Path(f"configs/speech_enhancement/cv25_tiny/{name}.yaml").read_text())
        assert train["valid_split"] == "dev"
        assert not any("AGFarsdat" in d for d in train["datasets"])
        assert all(s["eval_max_batches"] is None for s in train["stages"].values())


def make_data(tmp_path):
    for dataset in evaluation.TEST_DATASETS:
        directory = tmp_path / dataset
        (directory / "clips").mkdir(parents=True)
        sf.write(directory / "clips" / "clip.wav", np.zeros(1600), 16000)
        (directory / "test.tsv").write_text("path\tsentence\nclip.wav\thello world\n")
    return {"data": {"root_dir": str(tmp_path), "datasets": list(evaluation.TEST_DATASETS), "split": "test"},
            "methods": {"baseline": {"kind": "asr"}, "fusion": {"kind": "fusion"}, "bridge": {"kind": "bridge"}},
            "bridge_enhanced_root": str(tmp_path / "enhanced")}


def test_all_methods_receive_identical_original_cohort(tmp_path, monkeypatch):
    config = make_data(tmp_path)
    for dataset in evaluation.TEST_DATASETS:
        directory = tmp_path / "enhanced" / dataset
        directory.mkdir(parents=True)
        sf.write(directory / "clip.wav", np.ones(1600) * .1, 16000)
    calls = []
    def build(spec, _config, _device):
        def decode(wave, enhanced):
            calls.append((spec["kind"], wave.abs().sum().item(), enhanced is not None))
            return "hello world"
        return decode
    monkeypatch.setattr(evaluation, "build_decoder", build)
    output = tmp_path / "results"
    evaluation.run(config, output, None, "cpu")
    summary = json.loads((output / "summary.json").read_text())
    assert len({m["test_manifest_sha256"] for m in summary.values()}) == 1
    assert all(m["wer"] == 0 and m["examples"] == 2 for m in summary.values())
    assert all(original == 0 for _, original, _ in calls)
    assert [has_enhanced for kind, _, has_enhanced in calls if kind == "bridge"] == [True, True]
    assert set(summary["baseline"]["dataset_metrics"]) == set(evaluation.TEST_DATASETS)


def test_all_missing_bridge_audio_fails_before_model_loading(tmp_path, monkeypatch):
    config = make_data(tmp_path)
    monkeypatch.setattr(evaluation, "build_decoder", lambda *a: pytest.fail("loaded model before preflight"))
    with pytest.raises(ValueError, match="no usable common-cohort"):
        evaluation.run(config, tmp_path / "results", ["bridge"], "cpu")
    assert not (tmp_path / "results").exists()


def test_skips_duplicate_test_rows_but_requires_each_dataset(tmp_path):
    config = make_data(tmp_path)
    path = tmp_path / evaluation.TEST_DATASETS[0] / "test.tsv"
    path.write_text("path\tsentence\nclip.wav\tx\nclip.wav\ty\n")
    skipped = []
    with pytest.raises(ValueError, match="no usable test clips"):
        evaluation.test_rows(config, skipped)
    assert len(skipped) == 2
    assert {row["reason"] for row in skipped} == {"duplicate_test_path"}


def test_final_evaluation_skips_invalid_clips_from_common_cohort(tmp_path, monkeypatch):
    config = make_data(tmp_path)
    for dataset in evaluation.TEST_DATASETS:
        source = tmp_path / dataset / "test.tsv"
        source.write_text(source.read_text() + "missing.wav\tbad row\n")
        enhanced = tmp_path / "enhanced" / dataset
        enhanced.mkdir(parents=True)
        sf.write(enhanced / "clip.wav", np.ones(1600) * .1, 16000)
    monkeypatch.setattr(evaluation, "build_decoder", lambda *_: lambda _wave, _enhanced: "hello world")
    output = tmp_path / "results"
    evaluation.run(config, output, None, "cpu")
    summary = json.loads((output / "summary.json").read_text())
    assert all(item["examples"] == 2 for item in summary.values())
    skipped = [json.loads(line) for line in (output / "skipped_inputs.jsonl").read_text().splitlines()]
    assert len(skipped) == 2
    assert {row["reason"] for row in skipped} == {"missing_test_audio"}


def test_cli_help():
    proc = subprocess.run([sys.executable, "-m", "ml.fusion.evaluate_ablation", "--help"], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert "--methods" in proc.stdout and "--output" in proc.stdout
