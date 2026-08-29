from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ml.pmct.augmentation import PMCTExample, load_pmct_examples, mix_aligned_patches, patch_seed
from ml.pmct.train_whisper_small import (
    PMCTWhisperDataset,
    load_training_config,
    resolve_dataset_specs,
)
from ml.utils.audio import resample_audio


class _FeatureExtractor:
    def __init__(self) -> None:
        self.waveform: np.ndarray | None = None

    def __call__(self, waveform, **kwargs):
        import torch
        from types import SimpleNamespace

        self.waveform = np.asarray(waveform)
        return SimpleNamespace(input_features=torch.ones((1, 2, 3)))


class _Tokenizer:
    def __call__(self, text: str):
        from types import SimpleNamespace

        return SimpleNamespace(input_ids=list(range(len(text))))


class _Processor:
    def __init__(self) -> None:
        self.feature_extractor = _FeatureExtractor()
        self.tokenizer = _Tokenizer()


def _write_split(dataset: Path, relative_path: str, transcript: str = "سلام") -> None:
    (dataset / "train.tsv").write_text(
        f"path\tsentence\n{relative_path}\t{transcript}\n",
        encoding="utf-8",
    )


def _make_mapping_row(clean: Path, degraded: Path, relative_path: str, transcript: str) -> dict:
    return {
        "split": "train",
        "clean_path": str(clean),
        "degraded_path": str(degraded),
        "degraded_tsv_path": relative_path,
        "sentence": transcript,
        "degradation": {"normalization_scale": 1.0, "model_sample_rate": 16000},
    }


def test_mix_aligned_patches_honors_probability_extremes() -> None:
    clean = np.ones(12, dtype=np.float32)
    degraded = np.zeros(12, dtype=np.float32)

    all_clean = mix_aligned_patches(clean, degraded, 4, 0.5, 1.0, np.random.default_rng(1))
    all_degraded = mix_aligned_patches(clean, degraded, 4, 0.5, 0.0, np.random.default_rng(1))

    np.testing.assert_array_equal(all_clean, clean)
    np.testing.assert_array_equal(all_degraded, degraded)


def test_mix_aligned_patches_rejects_unaligned_audio() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        mix_aligned_patches(
            np.zeros(3, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            16000,
            1.0,
            0.5,
            np.random.default_rng(0),
        )


def test_patch_seed_is_stable_and_changes_by_epoch(tmp_path: Path) -> None:
    path = tmp_path / "noisy.wav"
    assert patch_seed(1337, 0, path) == patch_seed(1337, 0, path)
    assert patch_seed(1337, 0, path) != patch_seed(1337, 1, path)


def test_load_pmct_examples_reads_generator_mapping(tmp_path: Path) -> None:
    dataset = tmp_path / "noise-added"
    clean = tmp_path / "clean.wav"
    degraded = dataset / "clips" / "train" / "noisy.wav"
    degraded.parent.mkdir(parents=True)
    sf.write(clean, np.zeros(16, dtype=np.float32), 16000)
    sf.write(degraded, np.ones(16, dtype=np.float32) * 0.1, 16000)
    _write_split(dataset, "train/noisy.wav")
    row = {
        "split": "train",
        "clean_path": str(clean),
        "degraded_path": str(degraded),
        "degraded_tsv_path": "train/noisy.wav",
        "sentence": "سلام",
        "degradation": {"normalization_scale": 0.75, "model_sample_rate": 16000},
    }
    (dataset / "degraded_to_clean.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    examples = load_pmct_examples([dataset], expected_sample_rate=16000)

    assert len(examples) == 1
    assert examples[0].clean_path == clean.resolve()
    assert examples[0].degraded_path == degraded.resolve()
    assert examples[0].clean_scale == 0.75


def test_training_dataset_patches_before_feature_extraction(tmp_path: Path) -> None:
    clean = tmp_path / "clean.wav"
    degraded = tmp_path / "degraded.wav"
    sf.write(clean, np.ones(16, dtype=np.float32) * 0.5, 16000, subtype="FLOAT")
    sf.write(degraded, np.zeros(16, dtype=np.float32), 16000, subtype="FLOAT")
    example = PMCTExample(degraded, clean, "متن", tmp_path, clean_scale=0.5)
    processor = _Processor()
    dataset = PMCTWhisperDataset([example], processor, 16000, 0.0005, 1.0, seed=7)

    item = dataset[0]

    np.testing.assert_allclose(processor.feature_extractor.waveform, 0.25)
    assert item["labels"] == [0, 1, 2]


def test_training_dataset_uses_generator_soxr_resampling_for_odd_length_audio(tmp_path: Path) -> None:
    clean = tmp_path / "clean-48k.wav"
    degraded = tmp_path / "degraded-16k.wav"
    source = np.linspace(-0.5, 0.5, 48_001, dtype=np.float32)
    generated_clean = resample_audio(source, 48000, 16000)
    sf.write(clean, source, 48000, subtype="FLOAT")
    sf.write(degraded, np.zeros(len(generated_clean), dtype=np.float32), 16000, subtype="FLOAT")
    example = PMCTExample(degraded, clean, "متن", tmp_path)
    processor = _Processor()
    dataset = PMCTWhisperDataset([example], processor, 16000, 1.0, 1.0, seed=7)

    dataset[0]

    np.testing.assert_allclose(processor.feature_extractor.waveform, generated_clean, atol=1e-6)


def test_mapping_must_have_exact_tsv_parity(tmp_path: Path) -> None:
    dataset = tmp_path / "noise-added"
    clean = tmp_path / "clean.wav"
    degraded = dataset / "clips" / "train" / "noisy.wav"
    degraded.parent.mkdir(parents=True)
    sf.write(clean, np.zeros(16, dtype=np.float32), 16000)
    sf.write(degraded, np.zeros(16, dtype=np.float32), 16000)
    _write_split(dataset, "train/noisy.wav", transcript="TSV text")
    row = {
        "split": "train",
        "clean_path": str(clean),
        "degraded_path": str(degraded),
        "degraded_tsv_path": "train/noisy.wav",
        "sentence": "different mapping text",
        "degradation": {"normalization_scale": 1.0, "model_sample_rate": 16000},
    }
    (dataset / "degraded_to_clean.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="transcript does not match"):
        load_pmct_examples([dataset], expected_sample_rate=16000)


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), 0.0, -1.0, 1.1])
def test_mapping_rejects_invalid_normalization_scale(tmp_path: Path, scale: float) -> None:
    dataset = tmp_path / "noise-added"
    clean = tmp_path / "clean.wav"
    degraded = dataset / "clips" / "train" / "noisy.wav"
    degraded.parent.mkdir(parents=True)
    sf.write(clean, np.zeros(16, dtype=np.float32), 16000)
    sf.write(degraded, np.zeros(16, dtype=np.float32), 16000)
    _write_split(dataset, "train/noisy.wav")
    row = {
        "split": "train",
        "clean_path": str(clean),
        "degraded_path": str(degraded),
        "degraded_tsv_path": "train/noisy.wav",
        "sentence": "سلام",
        "degradation": {"normalization_scale": scale, "model_sample_rate": 16000},
    }
    (dataset / "degraded_to_clean.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="normalization_scale"):
        load_pmct_examples([dataset], expected_sample_rate=16000)


@pytest.mark.parametrize("sample_rate", [0, -1, 16000.5, float("nan"), float("inf")])
def test_mapping_rejects_invalid_model_sample_rate(tmp_path: Path, sample_rate: float) -> None:
    dataset = tmp_path / "noise-added"
    clean = tmp_path / "clean.wav"
    degraded = dataset / "clips" / "train" / "noisy.wav"
    degraded.parent.mkdir(parents=True)
    sf.write(clean, np.zeros(16, dtype=np.float32), 16000)
    sf.write(degraded, np.zeros(16, dtype=np.float32), 16000)
    _write_split(dataset, "train/noisy.wav")
    row = {
        "split": "train",
        "clean_path": str(clean),
        "degraded_path": str(degraded),
        "degraded_tsv_path": "train/noisy.wav",
        "sentence": "سلام",
        "degradation": {"normalization_scale": 1.0, "model_sample_rate": sample_rate},
    }
    (dataset / "degraded_to_clean.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="model_sample_rate"):
        load_pmct_examples([dataset], expected_sample_rate=16000)


@pytest.mark.parametrize("missing_field", ["normalization_scale", "model_sample_rate"])
def test_mapping_requires_generator_metadata(tmp_path: Path, missing_field: str) -> None:
    dataset = tmp_path / "noise-added"
    clean = tmp_path / "clean.wav"
    degraded = dataset / "clips" / "train" / "noisy.wav"
    degraded.parent.mkdir(parents=True)
    sf.write(clean, np.zeros(16, dtype=np.float32), 16000)
    sf.write(degraded, np.zeros(16, dtype=np.float32), 16000)
    _write_split(dataset, "train/noisy.wav")
    metadata = {"normalization_scale": 1.0, "model_sample_rate": 16000}
    metadata.pop(missing_field)
    row = {
        "split": "train",
        "clean_path": str(clean),
        "degraded_path": str(degraded),
        "degraded_tsv_path": "train/noisy.wav",
        "sentence": "سلام",
        "degradation": metadata,
    }
    (dataset / "degraded_to_clean.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a valid pMCT pair"):
        load_pmct_examples([dataset], expected_sample_rate=16000)


def test_mapping_parity_supports_multiple_variants(tmp_path: Path) -> None:
    dataset = tmp_path / "noise-added"
    clips = dataset / "clips" / "train"
    clips.mkdir(parents=True)
    clean = tmp_path / "clean.wav"
    sf.write(clean, np.zeros(16, dtype=np.float32), 16000)
    rows = []
    mapping_rows = []
    for variant in range(2):
        relative = f"train/noisy-v{variant}.wav"
        degraded = clips / f"noisy-v{variant}.wav"
        sf.write(degraded, np.zeros(16, dtype=np.float32), 16000)
        rows.append(f"{relative}\tسلام")
        mapping_rows.append(_make_mapping_row(clean, degraded, relative, "سلام"))
    (dataset / "train.tsv").write_text(
        "path\tsentence\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    (dataset / "degraded_to_clean.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mapping_rows),
        encoding="utf-8",
    )

    assert len(load_pmct_examples([dataset], expected_sample_rate=16000)) == 2


def test_mapping_parity_rejects_missing_mapping_row(tmp_path: Path) -> None:
    dataset = tmp_path / "noise-added"
    clips = dataset / "clips" / "train"
    clips.mkdir(parents=True)
    clean = tmp_path / "clean.wav"
    first = clips / "first.wav"
    second = clips / "second.wav"
    for path in (clean, first, second):
        sf.write(path, np.zeros(16, dtype=np.float32), 16000)
    (dataset / "train.tsv").write_text(
        "path\tsentence\ntrain/first.wav\tسلام\ntrain/second.wav\tسلام\n",
        encoding="utf-8",
    )
    row = _make_mapping_row(clean, first, "train/first.wav", "سلام")
    (dataset / "degraded_to_clean.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="missing 1 train.tsv row"):
        load_pmct_examples([dataset], expected_sample_rate=16000)


def test_mapping_parity_rejects_extra_or_duplicate_mapping_rows(tmp_path: Path) -> None:
    dataset = tmp_path / "noise-added"
    clips = dataset / "clips" / "train"
    clips.mkdir(parents=True)
    clean = tmp_path / "clean.wav"
    degraded = clips / "noisy.wav"
    extra = clips / "extra.wav"
    for path in (clean, degraded, extra):
        sf.write(path, np.zeros(16, dtype=np.float32), 16000)
    _write_split(dataset, "train/noisy.wav")
    row = _make_mapping_row(clean, degraded, "train/noisy.wav", "سلام")
    extra_row = _make_mapping_row(clean, extra, "train/extra.wav", "سلام")
    mapping = dataset / "degraded_to_clean.jsonl"
    mapping.write_text(
        json.dumps(row, ensure_ascii=False) + "\n" + json.dumps(extra_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not present in train.tsv"):
        load_pmct_examples([dataset], expected_sample_rate=16000)

    mapping.write_text(
        json.dumps(row, ensure_ascii=False) + "\n" + json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicates degraded path"):
        load_pmct_examples([dataset], expected_sample_rate=16000)


def test_mapping_parity_rejects_duplicate_tsv_path(tmp_path: Path) -> None:
    dataset = tmp_path / "noise-added"
    clips = dataset / "clips" / "train"
    clips.mkdir(parents=True)
    clean = tmp_path / "clean.wav"
    degraded = clips / "noisy.wav"
    sf.write(clean, np.zeros(16, dtype=np.float32), 16000)
    sf.write(degraded, np.zeros(16, dtype=np.float32), 16000)
    (dataset / "train.tsv").write_text(
        "path\tsentence\ntrain/noisy.wav\tسلام\ntrain/noisy.wav\tسلام\n",
        encoding="utf-8",
    )
    row = _make_mapping_row(clean, degraded, "train/noisy.wav", "سلام")
    (dataset / "degraded_to_clean.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicates audio path"):
        load_pmct_examples([dataset], expected_sample_rate=16000)


def test_training_config_validates_pmct_values(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("pmct:\n  clean_probability: 1.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="clean_probability"):
        load_training_config(config)


def test_dataset_specs_support_clean_paired_and_auto(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    paired = tmp_path / "paired"
    automatic = tmp_path / "automatic"
    for path in (clean, paired, automatic):
        path.mkdir()
    (automatic / "degraded_to_clean.jsonl").touch()
    config = {
        "data": {
            "root_dir": str(tmp_path),
            "mapping_filename": "degraded_to_clean.jsonl",
            "datasets": [
                {"path": "clean", "kind": "clean"},
                {"path": "paired", "kind": "paired"},
                "automatic",
            ],
        }
    }

    assert resolve_dataset_specs(config) == [
        (clean, "clean"),
        (paired, "paired"),
        (automatic, "paired"),
    ]
