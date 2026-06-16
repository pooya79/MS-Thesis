from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from ml.speech_data.validate_degraded_dataset import (
    best_lag,
    hf_energy_fraction,
    main,
    missing_metadata_fields,
)

SR = 16000


def _write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), SR)


def _make_aligned_dataset(root: Path, n: int = 3) -> Path:
    """A degraded dataset whose degraded clip is clean + mild noise (well aligned)."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        t = np.linspace(0, 1, SR, endpoint=False)
        clean = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
        degraded = (clean + 0.02 * rng.standard_normal(SR)).astype(np.float32)
        clean_path = root / "clean" / f"dev_clip{i}.wav"
        degraded_path = root / "clips" / "dev" / f"dev_clip{i}_v0.wav"
        _write_wav(clean_path, clean)
        _write_wav(degraded_path, degraded)
        rows.append(
            {
                "degraded_id": f"dev_clip{i}_v0",
                "split": "dev",
                "clean_path": str(clean_path),
                "degraded_path": str(degraded_path),
                "sentence": "سلام",
                "degradation": {
                    "model_sample_rate": SR,
                    "target_bandwidth": "wideband",
                    "channel_path": "wideband",
                    "channel_sample_rate": SR,
                    "channel_bandpass_hz": [50, 7000],
                    "normalization_scale": 1.0,
                    "snr_db": 20.0,
                    "codec": "pcm",
                },
            }
        )
    (root / "degraded_to_clean.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return root


def test_best_lag_recovers_known_shift() -> None:
    rng = np.random.default_rng(1)
    base = rng.standard_normal(4000)
    shifted = np.roll(base, 7)
    # Convention: np.roll(second, lag) re-aligns it onto the first (what the SNR
    # alignment relies on), so recovering `shifted` back to `base` needs lag = -7.
    lag, corr = best_lag(base, shifted, max_lag=50)
    assert lag == -7
    assert corr > 0.9
    assert np.allclose(np.roll(shifted, lag), base)


def test_hf_energy_fraction_low_for_lowpass_tone() -> None:
    t = np.linspace(0, 1, SR, endpoint=False)
    tone = np.sin(2 * np.pi * 300 * t)  # well below a 3400 Hz cutoff
    assert hf_energy_fraction(tone, SR, 3400.0) < 0.01


def test_missing_metadata_fields_flags_aligned_channel() -> None:
    assert missing_metadata_fields({"target_bandwidth": "wideband"}) == []
    missing = missing_metadata_fields({"target_bandwidth": "narrowband"})
    assert {"channel_sample_rate", "channel_bandpass_hz", "normalization_scale"} <= set(missing)


def test_validate_dataset_runs_and_does_not_flag_aligned_pairs(tmp_path: Path) -> None:
    root = _make_aligned_dataset(tmp_path / "ds")
    out_dir = tmp_path / "out"
    rc = main(["--dataset", str(root), "--sample", "0", "--output-dir", str(out_dir)])
    assert rc == 0
    report = json.loads((out_dir / "validation.json").read_text())
    block = report["datasets"][0]
    assert block["evaluated"] == 3
    # Aligned clean+noise pairs: lag ~0 and no misalignment / no-op / bandwidth flags.
    assert block["overall"]["lag_ms"]["median"] < 5.0
    assert "misaligned" not in block["flag_counts"]
    assert "near_noop" not in block["flag_counts"]
    assert np.isfinite(block["overall"]["waveform_snr_db"]["median"])
