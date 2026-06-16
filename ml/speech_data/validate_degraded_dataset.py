"""Validate that a degraded dataset's noisy/clean pairs are trainable.

The enhancement/fusion stack trains the enhancer to map the *noisy* clip to the
*reconstructed* bandwidth-aligned clean target (the target is rebuilt from the
recorded degradation metadata — see ``ml.enhancement.dataset``). If those pairs are
misaligned, the metadata does not match the saved clip, or the degradation is a
no-op, the enhancer is trained against garbage and ``L_enh`` plateaus — exactly the
failure this script hunts for. It does **not** re-run the codec/network degradation
(those round-trips are not bit-reproducible); instead it checks the pair for
internal consistency, which catches the same problems more robustly:

1. **Alignment** — best cross-correlation lag between the noisy clip and the
   reconstructed clean target. A lag of more than a few ms means their log-Mels do
   not line up frame-for-frame, so every ``L_enh`` gradient is computed against a
   shifted target.
2. **Degradation magnitude** — waveform SNR (clean vs noisy-minus-clean) and the
   mel L1 ``identity_L_enh`` per clip, grouped by ``target_bandwidth`` / channel /
   codec. Near-zero magnitude means the degradation barely changed the audio (no
   headroom for enhancement); pathologically large means a likely normalization or
   alignment bug.
3. **Bandwidth consistency** — for narrowband / wideband-filtered channels, the
   fraction of the noisy clip's (and the target's) energy *above* the recorded
   channel cutoff. It should be ~0; a large value means the recorded
   ``channel_bandpass_hz`` does not match the actual clip.
4. **Metadata completeness** — every pair must carry the fields the target
   reconstruction needs; missing fields silently fall back to ``full_band`` and
   change what the enhancer is trained against.

A clip is *flagged* when its lag exceeds ``--max-lag-ms``, the degradation is a
near no-op, the band-limiting is violated, or required metadata is missing. The
summary reports the flagged counts plus per-group distributions, and ``--output-dir``
writes the full ``validation.json``.

Example::

    uv run python -m ml.speech_data.validate_degraded_dataset \\
      --dataset data/cv-corpus-25.0-degraded-v2 \\
      --sample 300 --output-dir artifacts/degraded_validation
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ml.asr.whisper_features import WHISPER_SAMPLE_RATE, waveform_to_log_mel
from ml.enhancement.dataset import read_mapping, reconstruct_clean_target
from ml.utils.audio import load_audio, resample_audio, to_mono

_ALIGNED_BANDWIDTHS = {"narrowband", "wideband_filtered"}
# Fields the bandwidth-aligned target reconstruction needs for an aligned channel.
_REQUIRED_ALIGNED_FIELDS = ("channel_sample_rate", "channel_bandpass_hz", "normalization_scale")
_EPS = 1e-12


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def best_lag(noisy: np.ndarray, clean: np.ndarray, max_lag: int) -> tuple[int, float]:
    """FFT cross-correlation lag (samples) and peak correlation in ``[-max_lag, max_lag]``.

    A positive lag means ``clean`` is delayed relative to ``noisy``. The peak value
    is a normalised correlation coefficient in roughly ``[-1, 1]``.
    """
    n = int(min(len(noisy), len(clean)))
    if n == 0:
        return 0, 0.0
    a = np.asarray(noisy[:n], dtype=np.float64)
    b = np.asarray(clean[:n], dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    norm = float(np.sqrt((a @ a) * (b @ b))) + _EPS
    size = 1 << (2 * n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    # Lags 0..max_lag are corr[0..], negative lags wrap to the tail.
    max_lag = min(max_lag, n - 1)
    pos = corr[: max_lag + 1]
    neg = corr[size - max_lag :] if max_lag > 0 else np.empty(0)
    lags = np.concatenate([np.arange(-max_lag, 0), np.arange(0, max_lag + 1)])
    values = np.concatenate([neg, pos])
    best = int(np.argmax(np.abs(values)))
    return int(lags[best]), float(values[best] / norm)


def waveform_snr_db(noisy: np.ndarray, clean: np.ndarray, lag: int) -> float:
    """SNR (dB) of clean vs the residual noisy-minus-clean, after applying ``lag``."""
    clean_shifted = np.roll(clean, lag)
    n = min(len(noisy), len(clean_shifted))
    signal = clean_shifted[:n].astype(np.float64)
    residual = noisy[:n].astype(np.float64) - signal
    sig_power = float(signal @ signal)
    res_power = float(residual @ residual) + _EPS
    if sig_power <= _EPS:
        return 0.0
    return 10.0 * np.log10(sig_power / res_power)


def hf_energy_fraction(audio: np.ndarray, rate: int, cutoff_hz: float) -> float:
    """Fraction of spectral energy above ``cutoff_hz`` (0 = fully band-limited)."""
    if len(audio) == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float64))) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1.0 / rate)
    total = float(spectrum.sum()) + _EPS
    return float(spectrum[freqs > cutoff_hz].sum() / total)


def relative_l2(noisy: np.ndarray, clean: np.ndarray) -> float:
    """``||noisy - clean|| / ||clean||`` — a scale-aware "did anything change?" measure."""
    n = min(len(noisy), len(clean))
    diff = noisy[:n].astype(np.float64) - clean[:n].astype(np.float64)
    denom = float(np.linalg.norm(clean[:n].astype(np.float64))) + _EPS
    return float(np.linalg.norm(diff) / denom)


def missing_metadata_fields(degradation: dict[str, Any]) -> list[str]:
    """Required reconstruction fields absent from a pair's degradation metadata."""
    target_bandwidth = str(degradation.get("target_bandwidth", "wideband"))
    missing = [field for field in ("target_bandwidth",) if field not in degradation]
    if target_bandwidth in _ALIGNED_BANDWIDTHS:
        missing += [field for field in _REQUIRED_ALIGNED_FIELDS if degradation.get(field) is None]
    return missing


class RunningStats:
    """Accumulates a numeric series for mean / median / p10 / p90 reporting."""

    def __init__(self) -> None:
        self.values: list[float] = []

    def add(self, value: float) -> None:
        self.values.append(float(value))

    def summary(self) -> dict[str, Any]:
        if not self.values:
            return {"count": 0}
        arr = np.asarray(self.values, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }


def _group_summaries(groups: dict[str, dict[str, RunningStats]]) -> dict[str, Any]:
    return {
        name: {metric: stats.summary() for metric, stats in metrics.items()}
        for name, metrics in groups.items()
    }


def validate_pair(
    pair: Any,
    *,
    clean_target: str,
    model_name: str,
    max_lag_ms: float,
    noop_rel_l2: float,
    hf_tolerance: float,
) -> dict[str, Any] | None:
    """Compute all consistency metrics for one degraded/clean pair (None if unloadable)."""
    degradation = dict(pair.degradation)
    missing = missing_metadata_fields(degradation)

    try:
        degraded_audio, degraded_rate = load_audio(pair.degraded_path)
        clean_source, clean_rate = load_audio(pair.clean_path)
    except (FileNotFoundError, RuntimeError) as exc:
        logging.warning("skip %s: cannot load audio (%s)", pair.pair_id, exc)
        return None

    degraded_audio = to_mono(np.asarray(degraded_audio, dtype=np.float32))
    model_rate = int(degradation.get("model_sample_rate", degraded_rate))
    if degraded_rate != model_rate:
        degraded_audio = resample_audio(degraded_audio, degraded_rate, model_rate)
    clean_audio = reconstruct_clean_target(
        clean_source, clean_rate, degradation, target_length=len(degraded_audio),
        mode=clean_target, model_rate=model_rate,
    )

    max_lag = int(round(max_lag_ms / 1000.0 * model_rate))
    lag, corr = best_lag(degraded_audio, clean_audio, max_lag)
    snr = waveform_snr_db(degraded_audio, clean_audio, lag)
    rel = relative_l2(degraded_audio, clean_audio)
    mel_l1 = float(
        (waveform_to_log_mel(degraded_audio, sample_rate=model_rate, model_name=model_name)
         - waveform_to_log_mel(clean_audio, sample_rate=model_rate, model_name=model_name)).abs().mean()
    )

    target_bandwidth = str(degradation.get("target_bandwidth", "wideband"))
    cutoff = None
    degraded_hf = clean_hf = None
    if target_bandwidth in _ALIGNED_BANDWIDTHS and degradation.get("channel_bandpass_hz"):
        cutoff = float(degradation["channel_bandpass_hz"][1])
        degraded_hf = hf_energy_fraction(degraded_audio, model_rate, cutoff)
        clean_hf = hf_energy_fraction(clean_audio, model_rate, cutoff)

    flags: list[str] = []
    if missing:
        flags.append("missing_metadata")
    if abs(lag) >= max_lag:  # hit the search ceiling -> alignment is at least this bad
        flags.append("misaligned")
    if rel < noop_rel_l2:
        flags.append("near_noop")
    if degraded_hf is not None and degraded_hf > hf_tolerance:
        flags.append("bandwidth_mismatch")

    return {
        "pair_id": pair.pair_id,
        "split": pair.split,
        "target_bandwidth": target_bandwidth,
        "channel_path": str(degradation.get("channel_path", "unknown")),
        "codec": str(degradation.get("codec", "unknown")),
        "snr_db_meta": degradation.get("snr_db"),
        "lag_samples": lag,
        "lag_ms": lag / model_rate * 1000.0,
        "alignment_corr": corr,
        "waveform_snr_db": snr,
        "relative_l2": rel,
        "mel_l1": mel_l1,
        "degraded_hf_above_cutoff": degraded_hf,
        "clean_hf_above_cutoff": clean_hf,
        "cutoff_hz": cutoff,
        "missing_metadata": missing,
        "flags": flags,
    }


def validate_dataset(
    dataset_dir: Path,
    *,
    split: str | None,
    sample: int,
    rng: random.Random,
    clean_target: str,
    model_name: str,
    max_lag_ms: float,
    noop_rel_l2: float,
    hf_tolerance: float,
) -> dict[str, Any]:
    """Validate a sample of pairs from one degraded dataset; return its report block."""
    pairs = read_mapping(dataset_dir, split)
    if sample and len(pairs) > sample:
        pairs = rng.sample(pairs, sample)

    overall = {metric: RunningStats() for metric in ("lag_ms", "waveform_snr_db", "relative_l2", "mel_l1")}
    by_group: dict[str, dict[str, RunningStats]] = defaultdict(
        lambda: {metric: RunningStats() for metric in ("lag_ms", "waveform_snr_db", "mel_l1", "degraded_hf_above_cutoff")}
    )
    flag_counts: dict[str, int] = defaultdict(int)
    flagged_examples: list[dict[str, Any]] = []
    evaluated = 0

    from tqdm.auto import tqdm

    for pair in tqdm(pairs, desc=f"validate {dataset_dir.name}", unit="pair", dynamic_ncols=True, leave=False, disable=None):
        result = validate_pair(
            pair, clean_target=clean_target, model_name=model_name,
            max_lag_ms=max_lag_ms, noop_rel_l2=noop_rel_l2, hf_tolerance=hf_tolerance,
        )
        if result is None:
            flag_counts["unloadable"] += 1
            continue
        evaluated += 1
        overall["lag_ms"].add(abs(result["lag_ms"]))
        overall["waveform_snr_db"].add(result["waveform_snr_db"])
        overall["relative_l2"].add(result["relative_l2"])
        overall["mel_l1"].add(result["mel_l1"])
        group = by_group[result["target_bandwidth"]]
        group["lag_ms"].add(abs(result["lag_ms"]))
        group["waveform_snr_db"].add(result["waveform_snr_db"])
        group["mel_l1"].add(result["mel_l1"])
        if result["degraded_hf_above_cutoff"] is not None:
            group["degraded_hf_above_cutoff"].add(result["degraded_hf_above_cutoff"])
        for flag in result["flags"]:
            flag_counts[flag] += 1
        if result["flags"] and len(flagged_examples) < 50:
            flagged_examples.append(result)

    return {
        "dataset": str(dataset_dir),
        "evaluated": evaluated,
        "flag_counts": dict(flag_counts),
        "overall": {metric: stats.summary() for metric, stats in overall.items()},
        "by_target_bandwidth": _group_summaries(by_group),
        "flagged_examples": flagged_examples,
    }


def run_validation(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    rng = random.Random(args.seed)
    split = None if args.split in (None, "", "all") else args.split

    report = {
        "created_at": utc_now(),
        "split": split or "all",
        "clean_target": args.clean_target,
        "sample": args.sample,
        "max_lag_ms": args.max_lag_ms,
        "datasets": [],
    }
    total_flags: dict[str, int] = defaultdict(int)
    total_evaluated = 0
    for dataset_dir in args.dataset:
        block = validate_dataset(
            Path(dataset_dir), split=split, sample=args.sample, rng=rng,
            clean_target=args.clean_target, model_name=args.model_name,
            max_lag_ms=args.max_lag_ms, noop_rel_l2=args.noop_rel_l2, hf_tolerance=args.hf_tolerance,
        )
        report["datasets"].append(block)
        total_evaluated += block["evaluated"]
        for flag, count in block["flag_counts"].items():
            total_flags[flag] += count

        logging.info("=== %s (evaluated=%s) ===", dataset_dir, block["evaluated"])
        overall = block["overall"]
        logging.info(
            "  |lag| ms: median=%.1f p90=%.1f | waveform SNR dB: median=%.1f p10=%.1f | mel_l1: median=%.4f",
            overall["lag_ms"].get("median", float("nan")),
            overall["lag_ms"].get("p90", float("nan")),
            overall["waveform_snr_db"].get("median", float("nan")),
            overall["waveform_snr_db"].get("p10", float("nan")),
            overall["mel_l1"].get("median", float("nan")),
        )
        if block["flag_counts"]:
            logging.info("  flags: %s", ", ".join(f"{k}={v}" for k, v in sorted(block["flag_counts"].items())))
        else:
            logging.info("  flags: none")
        for name in sorted(block["by_target_bandwidth"]):
            grp = block["by_target_bandwidth"][name]
            logging.info(
                "    %s (n=%s): SNR median=%.1f dB, mel_l1 median=%.4f, degraded HF>cutoff median=%.3f",
                name, grp["mel_l1"].get("count", 0),
                grp["waveform_snr_db"].get("median", float("nan")),
                grp["mel_l1"].get("median", float("nan")),
                grp["degraded_hf_above_cutoff"].get("median", float("nan")),
            )

    report["total_evaluated"] = total_evaluated
    report["total_flag_counts"] = dict(total_flags)
    flagged = sum(total_flags.values())
    logging.info(
        "validation done: %s pairs evaluated, %s flags raised (%.1f%%)",
        total_evaluated, flagged, 100.0 * flagged / max(1, total_evaluated),
    )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "validation.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logging.info("wrote report=%s", report_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a degraded dataset's noisy/clean pairs for trainability: alignment lag, "
            "degradation magnitude (SNR / mel L1), bandwidth consistency, and metadata completeness. "
            "Does not re-run the codec/network degradation (not bit-reproducible)."
        )
    )
    parser.add_argument("--dataset", action="append", required=True, help="Degraded dataset dir (repeatable).")
    parser.add_argument("--split", default=None, help="Split to validate (default: all splits).")
    parser.add_argument("--sample", type=int, default=200, help="Random pairs per dataset to check (0 = all).")
    parser.add_argument("--seed", type=int, default=1337, help="Sampling seed.")
    parser.add_argument("--clean-target", choices=["bandwidth_aligned", "full_band"], default="bandwidth_aligned")
    parser.add_argument("--model-name", default="openai/whisper-small", help="Whisper processor for the log-Mel front end.")
    parser.add_argument("--max-lag-ms", type=float, default=20.0, help="Flag pairs whose |alignment lag| reaches this (default 20 ms).")
    parser.add_argument("--noop-rel-l2", type=float, default=0.05, help="Flag pairs whose ||noisy-clean||/||clean|| is below this (near no-op).")
    parser.add_argument("--hf-tolerance", type=float, default=0.05, help="Flag band-limited clips with more than this energy fraction above the channel cutoff.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Write validation.json here.")
    parser.add_argument("--sample-rate", type=int, default=WHISPER_SAMPLE_RATE, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    return run_validation(args)


if __name__ == "__main__":
    raise SystemExit(main())
