from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from scipy import signal


FloatArray = np.ndarray


def load_audio(path: str | Path) -> tuple[FloatArray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    return np.asarray(mono, dtype=np.float32), int(sample_rate)


def save_audio(path: str | Path, audio: FloatArray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = np.asarray(audio, dtype=np.float32)
    sf.write(str(path), safe, sample_rate, subtype="PCM_16")


def to_mono(audio: FloatArray) -> FloatArray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.mean(axis=1)
    raise ValueError(f"expected 1D or 2D audio, got shape {arr.shape}")


def resample_audio(audio: FloatArray, source_rate: int, target_rate: int) -> FloatArray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    return np.asarray(soxr.resample(audio, source_rate, target_rate), dtype=np.float32)


def match_length(audio: FloatArray, length: int) -> FloatArray:
    arr = np.asarray(audio, dtype=np.float32)
    if len(arr) == length:
        return arr
    if len(arr) > length:
        return arr[:length]
    return np.pad(arr, (0, length - len(arr))).astype(np.float32)


def bandpass_filter(audio: FloatArray, sample_rate: int, low_hz: float, high_hz: float) -> FloatArray:
    if low_hz <= 0 or high_hz <= 0:
        raise ValueError("band-pass cutoffs must be positive")
    nyquist = sample_rate / 2
    high = min(high_hz, nyquist * 0.98)
    low = min(low_hz, high * 0.8)
    if low >= high:
        raise ValueError(f"invalid band-pass range {low_hz}-{high_hz} for {sample_rate} Hz")
    sos = signal.butter(6, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    if len(audio) < 64:
        return np.asarray(signal.sosfilt(sos, audio), dtype=np.float32)
    return np.asarray(signal.sosfiltfilt(sos, audio), dtype=np.float32)


def peak_safety_normalize(audio: FloatArray, peak: float = 0.99) -> FloatArray:
    if not 0 < peak <= 1:
        raise ValueError("peak must be in (0, 1]")
    arr = np.asarray(audio, dtype=np.float32)
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    if max_abs == 0 or not math.isfinite(max_abs):
        return np.nan_to_num(arr, nan=0.0, posinf=peak, neginf=-peak).astype(np.float32)
    if max_abs <= peak:
        return arr
    return np.asarray(arr * (peak / max_abs), dtype=np.float32)


def rms(audio: FloatArray) -> float:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr)) + 1e-12))


def mix_at_snr(clean: FloatArray, noise: FloatArray, snr_db: float) -> FloatArray:
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    if noise_rms == 0:
        return np.asarray(clean, dtype=np.float32)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    return np.asarray(clean + scaled_noise, dtype=np.float32)


def repeat_or_crop(audio: FloatArray, length: int, start: int = 0) -> FloatArray:
    arr = np.asarray(audio, dtype=np.float32)
    if len(arr) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(arr) < length:
        repeats = int(np.ceil(length / len(arr)))
        arr = np.tile(arr, repeats)
    if len(arr) == length:
        return arr
    start = max(0, min(start, len(arr) - length))
    return np.asarray(arr[start : start + length], dtype=np.float32)


def convolve_rir(audio: FloatArray, rir: FloatArray, wet_mix: float) -> FloatArray:
    if len(rir) == 0:
        return np.asarray(audio, dtype=np.float32)
    rir = np.asarray(rir, dtype=np.float32)
    rir_peak = float(np.max(np.abs(rir)))
    if rir_peak > 0:
        rir = rir / rir_peak
    reverbed = signal.fftconvolve(audio, rir, mode="full")[: len(audio)]
    reverbed = peak_safety_normalize(reverbed, peak=1.0)
    return np.asarray((1 - wet_mix) * audio + wet_mix * reverbed, dtype=np.float32)
