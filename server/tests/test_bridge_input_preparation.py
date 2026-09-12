import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ml.fusion import prepare_bridge_inputs as prep
from ml.fusion.bridge_pretrained import FRCRN, dnsmos_segments


def make_dataset(root):
    clean = root / "cv-corpus-25.0"
    degraded = root / "cv-corpus-25.0-degraded-v2"
    (clean / "clips").mkdir(parents=True)
    (degraded / "clips").mkdir(parents=True)
    rows = []
    for split in ("train", "dev", "test"):
        sf.write(clean / "clips" / f"{split}.wav", np.ones(1600) * .1, 16000, subtype="FLOAT")
        (clean / f"{split}.tsv").write_text(f"path\tsentence\tclient_id\n{split}.wav\thello\t{split}-speaker\n")
        if split != "test":
            sf.write(degraded / "clips" / f"{split}.wav", np.ones(1600) * .2, 16000, subtype="FLOAT")
            rows.append({"degraded_id": split, "split": split, "sentence": "hello",
                         "degraded_path": f"clips/{split}.wav", "clean_path": str(clean / "clips" / f"{split}.wav")})
    (degraded / "degraded_to_clean.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    ag = root / "AGFarsdat_test_normalized"
    (ag / "clips").mkdir(parents=True)
    sf.write(ag / "clips" / "a.wav", np.ones(1600) * .3, 16000, subtype="FLOAT")
    (ag / "test.tsv").write_text("path\tsentence\na.wav\thello\n")


def test_dns_short_clip_repetition_matches_reference():
    wave = np.arange(1600, dtype=np.float32)
    segments = dnsmos_segments(wave)
    repeated = wave.copy()
    while len(repeated) < 144160:
        repeated = np.concatenate((repeated, repeated))
    count = int(np.floor(len(repeated) / 16000) - 9.01) + 1
    assert len(segments) == count
    for i, segment in enumerate(segments):
        np.testing.assert_array_equal(segment, repeated[i * 16000:i * 16000 + 144160])
    with pytest.raises(ValueError):
        dnsmos_segments(np.array([]))


def test_frcrn_batches_variable_length_waves_and_restores_lengths():
    class Model:
        def __init__(self):
            self.shapes = []

        def inference_batch(self, inputs):
            self.shapes.append(tuple(inputs.shape))
            return inputs * .5

    enhancer = FRCRN.__new__(FRCRN)
    enhancer.model = Model()
    enhancer.device = "cpu"
    waves = [np.ones(1600, dtype=np.float32), np.ones(18000, dtype=np.float32)]

    outputs = enhancer.enhance_batch(waves)

    assert enhancer.model.shapes == [(2, 28000)]
    assert [len(output) for output in outputs] == [1600, 18000]
    np.testing.assert_allclose(outputs[0], .5)
    np.testing.assert_allclose(outputs[1], .5)


def test_preparation_generates_all_inputs_and_resumes(tmp_path, monkeypatch):
    data = tmp_path / "data"
    make_dataset(data)
    enhanced_calls, score_calls = [], []
    class Enhancer:
        identity = "verified-frcrn"
        def __init__(self, *args): pass
        def __call__(self, wave):
            enhanced_calls.append(wave.mean())
            return wave * .5
    class Scorer:
        identity = "verified-dnsmos"
        def __init__(self, *args): pass
        def __call__(self, wave):
            score_calls.append(wave.mean())
            return {"dnsmos_sig": 3., "dnsmos_bak": 2.}
    monkeypatch.setattr(prep, "FRCRN", Enhancer)
    monkeypatch.setattr(prep, "DNSMOS", Scorer)
    output = tmp_path / "prepared"
    command = ["--data-root", str(data), "--output", str(output), "--model-root", str(tmp_path / "models")]
    prep.main(command)
    assert len(enhanced_calls) == 4 and len(score_calls) == 2
    assert all(abs(x - .2) < 1e-6 for x in score_calls)  # score noisy, not clean or enhanced
    train_dev = [json.loads(x) for x in (output / "bridge_train_dev.jsonl").read_text().splitlines()]
    test = [json.loads(x) for x in (output / "bridge_test.jsonl").read_text().splitlines()]
    assert {r["split"] for r in train_dev} == {"train", "dev"}
    assert all("dnsmos_sig" not in r for r in test)
    assert (output / "test-enhanced/cv-corpus-25.0/test.wav").is_file()
    assert (output / "test-enhanced/AGFarsdat_test_normalized/a.wav").is_file()
    import yaml
    config = yaml.safe_load((output / "final_tests.yaml").read_text())
    assert config["bridge_enhancer_id"] == f"{Enhancer.identity}:batch-pad-v1:size-4"
    prep.main(command)
    assert len(enhanced_calls) == 4 and len(score_calls) == 2
    sf.write(data / "cv-corpus-25.0-degraded-v2/clips/train.wav", np.zeros(1600), 16000)
    with pytest.raises(ValueError, match="changed"):
        prep.main(command)


def test_preparation_uses_configured_batches_and_workers(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    make_dataset(data)
    batch_sizes = []

    class Enhancer:
        identity = "batch-frcrn"

        def __init__(self, *args): pass

        def enhance_batch(self, waves, *, target_length):
            batch_sizes.append(len(waves))
            assert target_length == 16000
            return [wave * .5 for wave in waves]

    class Scorer:
        identity = "threaded-dnsmos"

        def __init__(self, *args): pass

        def __call__(self, wave):
            return {"dnsmos_sig": 3., "dnsmos_bak": 2.}

    monkeypatch.setattr(prep, "FRCRN", Enhancer)
    monkeypatch.setattr(prep, "DNSMOS", Scorer)
    output = tmp_path / "prepared"

    prep.main(["--data-root", str(data), "--output", str(output),
               "--batch-size", "2", "--workers", "2"])

    assert batch_sizes == [2, 2]
    assert json.loads((output / "progress.json").read_text())["state"] == "completed"
    console = capsys.readouterr().out
    assert "validating with 2 workers" in console
    assert "Loading/downloading FRCRN" in console


def test_selection_reproducible_and_no_clean_reference_leakage(tmp_path):
    make_dataset(tmp_path)
    rows = prep.collect_rows(tmp_path, "all")
    assert prep.select_rows(rows, 1, 1337) == prep.select_rows(list(reversed(rows)), 1, 1337)
    for row in rows:
        if row["split"] != "test":
            assert row["noisy_path"] != row["source_id"]


def test_dry_run_does_not_load_models(tmp_path, monkeypatch):
    make_dataset(tmp_path / "data")
    monkeypatch.setattr(prep, "FRCRN", lambda *a: pytest.fail("model loaded"))
    output = tmp_path / "output"
    prep.main(["--data-root", str(tmp_path / "data"), "--output", str(output), "--dry-run"])
    assert not output.exists()


def test_preparation_skips_degraded_rows_missing_from_original_split(tmp_path, capsys):
    data = tmp_path / "data"
    make_dataset(data)
    mapping = data / "cv-corpus-25.0-degraded-v2/degraded_to_clean.jsonl"
    stale = {"degraded_id": "stale", "split": "train", "sentence": "hello",
             "degraded_path": "clips/train.wav",
             "clean_path": str(data / "cv-corpus-25.0/clips/test.wav")}
    mapping.write_text(mapping.read_text() + "\n" + json.dumps(stale))
    rows = prep.collect_rows(data, "all", skipped := [])
    assert len(rows) == 4
    assert [row["reason"] for row in skipped] == ["source_missing_or_wrong_split"]
    prep.main(["--data-root", str(data), "--output", str(tmp_path / "output"), "--dry-run"])
    assert "Skipped inputs: {'source_missing_or_wrong_split': 1}" in capsys.readouterr().out


def test_preparation_matches_legacy_wav_mapping_to_current_flac_clips(tmp_path):
    data = tmp_path / "data"
    make_dataset(data)
    clean = data / "cv-corpus-25.0"
    for split in ("train", "dev", "test"):
        wav = clean / "clips" / f"{split}.wav"
        wave, rate = sf.read(wav, dtype="float32")
        sf.write(wav.with_suffix(".flac"), wave, rate)
        wav.unlink()
        (clean / f"{split}.tsv").write_text(
            f"path\tsentence\tclient_id\n{split}.flac\thello\t{split}-speaker\n"
        )
        if split != "test":
            degraded_wav = data / "cv-corpus-25.0-degraded-v2/clips" / f"{split}.wav"
            degraded_wave, degraded_rate = sf.read(degraded_wav, dtype="float32")
            sf.write(degraded_wav.with_suffix(".flac"), degraded_wave, degraded_rate)
            degraded_wav.unlink()

    rows = prep.collect_rows(data, "all", skipped := [])

    assert skipped == []
    assert {row["split"] for row in rows} == {"train", "dev", "test"}
    assert {row["source_id"] for row in rows if row["dataset"].startswith("cv-corpus")} == {
        "cv25/train", "cv25/dev", "cv25/test"
    }
    assert all(Path(row["noisy_path"]).suffix == ".flac" for row in rows[:2])


def test_preparation_help():
    result = subprocess.run([sys.executable, "-m", "ml.fusion.prepare_bridge_inputs", "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--scope" in result.stdout and "--dry-run" in result.stdout
    assert "--batch-size" in result.stdout and "--workers" in result.stdout
