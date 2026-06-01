from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

from ml.utils.audio import (
    bandpass_filter,
    convolve_rir,
    load_audio,
    match_length,
    mix_at_snr,
    peak_safety_normalize,
    repeat_or_crop,
    resample_audio,
    save_audio,
)
from ml.utils.seed import stable_seed


CODECS: dict[str, dict[str, Any]] = {
    "pass_through": {"ffmpeg": None, "extension": None},
    "g711_alaw": {"ffmpeg": "pcm_alaw", "extension": ".wav"},
    "g711_mulaw": {"ffmpeg": "pcm_mulaw", "extension": ".wav"},
    "gsm": {"ffmpeg": "libgsm", "extension": ".gsm"},
    "amr_nb_12k2": {"ffmpeg": "libopencore_amrnb", "extension": ".amr", "bitrate": "12.2k"},
    "amr_wb_12k65": {"ffmpeg": "libvo_amrwbenc", "extension": ".amr", "bitrate": "12.65k"},
    "opus_nb": {"ffmpeg": "libopus", "extension": ".ogg", "bitrate": "16k"},
    "opus_wb": {"ffmpeg": "libopus", "extension": ".ogg", "bitrate": "24k"},
}

NARROWBAND_CODECS = {"g711_alaw", "g711_mulaw", "gsm", "amr_nb_12k2", "opus_nb"}
WIDEBAND_CODECS = {"amr_wb_12k65", "opus_wb"}
SAFE_ID_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ManifestItem:
    id: str
    split: str
    clean_path: Path
    transcript: str | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_path(value: str | None, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_clean_manifest(path: Path, expected_split: str) -> list[ManifestItem]:
    rows = read_jsonl(path)
    items: list[ManifestItem] = []
    for index, row in enumerate(rows, start=1):
        missing = {"id", "split", "clean_path"} - row.keys()
        if missing:
            raise ValueError(f"{path}:{index} missing required keys: {sorted(missing)}")
        split = str(row["split"])
        if split != expected_split:
            raise ValueError(f"{path}:{index} has split {split!r}, expected {expected_split!r}")
        clean_path = resolve_path(str(row["clean_path"]), path.parent)
        if clean_path is None or not clean_path.exists():
            raise FileNotFoundError(f"missing clean audio for {row['id']}: {row['clean_path']}")
        items.append(
            ManifestItem(
                id=str(row["id"]),
                split=split,
                clean_path=clean_path,
                transcript=row.get("transcript"),
            )
        )
    return items


def load_asset_index(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows = read_jsonl(path)
    assets: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if "path" not in row:
            raise ValueError(f"{path}:{index} missing required key: path")
        asset_path = resolve_path(str(row["path"]), path.parent)
        if asset_path is None or not asset_path.exists():
            raise FileNotFoundError(f"missing indexed asset in {path}:{index}: {row['path']}")
        copy = dict(row)
        copy["path"] = str(asset_path)
        copy.setdefault("id", asset_path.stem)
        assets.append(copy)
    return assets


def validate_distribution(name: str, entries: list[dict[str, Any]]) -> None:
    if not entries:
        raise ValueError(f"{name} distribution must not be empty")
    total = sum(float(entry.get("weight", 0)) for entry in entries)
    if total <= 0:
        raise ValueError(f"{name} distribution weights must sum to a positive number")
    for entry in entries:
        if float(entry.get("weight", 0)) < 0:
            raise ValueError(f"{name} distribution contains a negative weight")


def weighted_choice(rng: np.random.Generator, entries: list[dict[str, Any]]) -> dict[str, Any]:
    weights = np.asarray([float(entry["weight"]) for entry in entries], dtype=np.float64)
    probabilities = weights / weights.sum()
    index = int(rng.choice(len(entries), p=probabilities))
    return entries[index]


def sample_uniform(rng: np.random.Generator, bounds: list[float] | tuple[float, float]) -> float:
    if len(bounds) != 2:
        raise ValueError(f"expected [min, max] bounds, got {bounds}")
    low, high = float(bounds[0]), float(bounds[1])
    return float(rng.uniform(low, high))


def configured_codec_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    entries = list(config["codec_distribution"])
    for profile in config.get("profiles") or []:
        entries.extend(profile.get("codec_distribution", []))
    return entries


def require_ffmpeg_codecs(config: dict[str, Any]) -> None:
    selected_codecs = {entry["codec"] for entry in configured_codec_entries(config) if entry["codec"] != "pass_through"}
    if not selected_codecs:
        return
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for codec simulation but was not found on PATH")
    missing = sorted(codec for codec in selected_codecs if codec not in CODECS)
    if missing:
        raise ValueError(f"unsupported codec names in config: {missing}")
    codec_output = subprocess.run(["ffmpeg", "-hide_banner", "-codecs"], text=True, capture_output=True, check=True)
    available = codec_output.stdout + codec_output.stderr
    missing_encoders = sorted(
        f"{codec} ({CODECS[codec]['ffmpeg']})"
        for codec in selected_codecs
        if str(CODECS[codec]["ffmpeg"]) not in available
    )
    if missing_encoders:
        raise RuntimeError(f"ffmpeg is missing required encoders: {missing_encoders}")


def safe_pair_id(split: str, clip_id: str, variant_index: int) -> str:
    safe_clip_id = SAFE_ID_PATTERN.sub("_", clip_id).strip("._")
    if not safe_clip_id:
        safe_clip_id = "clip"
    return f"{split}_{safe_clip_id}_v{variant_index}"


def codec_roundtrip(
    audio: np.ndarray,
    sample_rate: int,
    codec: str,
    bitrate: str | None = None,
    frame_duration_ms: int | None = None,
) -> np.ndarray:
    spec = CODECS[codec]
    if spec["ffmpeg"] is None:
        return np.asarray(audio, dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="degrade_codec_") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "input.wav"
        encoded_path = tmp_path / f"encoded{spec['extension']}"
        output_path = tmp_path / "output.wav"
        sf.write(str(input_path), audio, sample_rate, subtype="PCM_16")
        encode_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-c:a",
            str(spec["ffmpeg"]),
        ]
        selected_bitrate = bitrate or spec.get("bitrate")
        if selected_bitrate:
            encode_cmd.extend(["-b:a", str(selected_bitrate)])
        if spec["ffmpeg"] == "libopus" and frame_duration_ms is not None:
            encode_cmd.extend(["-frame_duration", str(frame_duration_ms)])
        encode_cmd.append(str(encoded_path))
        decode_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(encoded_path),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            str(output_path),
        ]
        subprocess.run(encode_cmd, check=True)
        subprocess.run(decode_cmd, check=True)
        decoded, decoded_rate = load_audio(output_path)
    if decoded_rate != sample_rate:
        decoded = resample_audio(decoded, decoded_rate, sample_rate)
    return np.asarray(decoded, dtype=np.float32)


def apply_decoded_waveform_dropout(
    audio: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    loss_rate: float,
    burst_length: int,
    frame_ms: int,
) -> tuple[np.ndarray, int, int]:
    frame_len = max(1, int(round(sample_rate * frame_ms / 1000)))
    output = np.asarray(audio, dtype=np.float32).copy()
    frame_count = int(np.ceil(len(output) / frame_len))
    target_loss_rate = min(1.0, max(0.0, loss_rate))
    if frame_count == 0 or target_loss_rate == 0:
        return output, 0, frame_count
    bad_to_good = min(1.0, 1.0 / max(1, burst_length))
    good_to_bad = min(1.0, (target_loss_rate * bad_to_good) / max(1e-6, 1.0 - target_loss_rate))
    frame = 0
    in_bad_state = False
    dropped_frames = 0
    while frame < frame_count:
        if in_bad_state:
            start = frame * frame_len
            end = min(len(output), (frame + 1) * frame_len)
            output[start:end] = 0
            dropped_frames += 1
            in_bad_state = rng.random() >= bad_to_good
        else:
            in_bad_state = rng.random() < good_to_bad
        frame += 1
    return output, dropped_frames, frame_count


def choose_profile(config: dict[str, Any], rng: np.random.Generator) -> tuple[str, dict[str, Any]]:
    profiles = config.get("profiles") or [{"name": "legacy", "weight": 1.0}]
    profile = weighted_choice(rng, profiles)
    profile_name = str(profile.get("name", "unnamed"))
    overrides = {key: value for key, value in profile.items() if key not in {"name", "weight", "description"}}
    effective_config = deep_merge({key: value for key, value in config.items() if key != "profiles"}, overrides)
    return profile_name, effective_config


def sample_codec_parameter(rng: np.random.Generator, value: Any) -> Any:
    if isinstance(value, list):
        if not value:
            return None
        return value[int(rng.integers(0, len(value)))]
    return value


def choose_reverb(config: dict[str, Any], rng: np.random.Generator) -> tuple[str, float, float | None]:
    reverb_cfg = config["reverb"]
    roll = rng.random()
    severe_probability = float(reverb_cfg["severe"]["probability"])
    mild_probability = float(reverb_cfg["mild"]["probability"])
    if roll < severe_probability:
        wet_mix = sample_uniform(rng, reverb_cfg["severe"]["wet_mix"])
        return "severe", wet_mix, sample_uniform(rng, reverb_cfg["severe"]["dr_db"])
    if roll < severe_probability + mild_probability:
        wet_mix = sample_uniform(rng, reverb_cfg["mild"]["wet_mix"])
        return "mild", wet_mix, sample_uniform(rng, reverb_cfg["mild"]["dr_db"])
    return "none", 0.0, None


def choose_noise_segment(noise: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if len(noise) <= length:
        return repeat_or_crop(noise, length)
    start = int(rng.integers(0, len(noise) - length + 1))
    return repeat_or_crop(noise, length, start=start)


def process_item(
    item: ManifestItem,
    variant_index: int,
    config: dict[str, Any],
    rir_assets: list[dict[str, Any]],
    noise_assets: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata, clean_target, degraded_model, model_rate = degrade_item(item, variant_index, config, rir_assets, noise_assets)
    pair_dir = Path(config["output_dir"]) / "pairs" / item.split
    clean_out = pair_dir / "clean" / f"{metadata['pair_id']}.wav"
    degraded_out = pair_dir / "degraded" / f"{metadata['pair_id']}.wav"
    save_audio(clean_out, clean_target, model_rate)
    save_audio(degraded_out, degraded_model, model_rate)
    metadata.update({"clean_path": str(clean_out), "degraded_path": str(degraded_out)})
    return metadata


def degrade_item(
    item: ManifestItem,
    variant_index: int,
    config: dict[str, Any],
    rir_assets: list[dict[str, Any]],
    noise_assets: list[dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, int]:
    seed = stable_seed(int(config["seed"]), item.split, item.id, variant_index)
    rng = np.random.default_rng(seed)
    model_rate = int(config["model_sample_rate"])
    working_rate = int(config.get("working_sample_rate", model_rate))
    clean_original, source_rate = load_audio(item.clean_path)
    clean_working = resample_audio(clean_original, source_rate, working_rate)
    degraded = clean_working.copy()
    profile_name, degradation_config = choose_profile(config, rng)
    metadata: dict[str, Any] = {
        "pair_id": safe_pair_id(item.split, item.id, variant_index),
        "split": item.split,
        "profile": profile_name,
        "source_clean_id": item.id,
        "source_clean_path": str(item.clean_path),
        "model_sample_rate": model_rate,
        "seed": seed,
        "transcript": item.transcript,
    }

    reverb_mode, wet_mix, dr_db = choose_reverb(degradation_config, rng)
    metadata.update({"rir_id": None, "reverb_mode": reverb_mode, "reverb_wet_mix": wet_mix, "reverb_dr_db": dr_db})
    if reverb_mode != "none" and rir_assets:
        rir_asset = rir_assets[int(rng.integers(0, len(rir_assets)))]
        rir_audio, rir_rate = load_audio(rir_asset["path"])
        rir_audio = resample_audio(rir_audio, rir_rate, working_rate)
        degraded = convolve_rir(degraded, rir_audio, wet_mix=wet_mix)
        metadata["rir_id"] = rir_asset.get("id")

    noise_cfg = degradation_config["noise"]
    metadata.update({"noise_scenes": [], "noise_ids": [], "snr_db": None})
    if noise_assets and rng.random() < float(noise_cfg["probability"]):
        scene_count = 2 if rng.random() < float(noise_cfg["second_scene_probability"]) else 1
        snr_bucket = noise_cfg["snr_buckets"][int(rng.integers(0, len(noise_cfg["snr_buckets"])))]
        snr_db = sample_uniform(rng, snr_bucket)
        combined_noise = np.zeros_like(degraded)
        selected_assets = [noise_assets[int(rng.integers(0, len(noise_assets)))] for _ in range(scene_count)]
        for noise_asset in selected_assets:
            noise_audio, noise_rate = load_audio(noise_asset["path"])
            noise_audio = resample_audio(noise_audio, noise_rate, working_rate)
            combined_noise += choose_noise_segment(noise_audio, len(degraded), rng)
            metadata["noise_scenes"].append(noise_asset.get("scene", noise_asset.get("id")))
            metadata["noise_ids"].append(noise_asset.get("id"))
        combined_noise /= max(1, scene_count)
        degraded = mix_at_snr(degraded, combined_noise, snr_db=snr_db)
        metadata["snr_db"] = snr_db

    level_cfg = degradation_config["level"]
    gain_db = sample_uniform(rng, level_cfg["gain_db"])
    degraded = np.asarray(degraded * (10 ** (gain_db / 20)), dtype=np.float32)
    metadata["gain_db"] = gain_db
    clipping_cfg = level_cfg.get("clipping", {})
    clipping_enabled = bool(clipping_cfg.get("enabled", False)) and rng.random() < float(clipping_cfg.get("probability", 0))
    metadata["clipping"] = {"enabled": clipping_enabled, "mode": None, "threshold": None}
    if clipping_enabled:
        threshold = sample_uniform(rng, clipping_cfg.get("threshold", [0.8, 0.98]))
        degraded = np.clip(degraded, -threshold, threshold).astype(np.float32)
        metadata["clipping"] = {"enabled": True, "mode": clipping_cfg.get("mode", "hard"), "threshold": threshold}
    metadata["agc"] = {"enabled": bool(level_cfg.get("agc", {}).get("enabled", False))}

    codec_entry = weighted_choice(rng, degradation_config["codec_distribution"])
    codec = str(codec_entry["codec"])
    codec_bitrate = sample_codec_parameter(rng, codec_entry.get("bitrate"))
    codec_frame_duration_ms = sample_codec_parameter(rng, codec_entry.get("frame_duration_ms"))
    if codec_frame_duration_ms is not None:
        codec_frame_duration_ms = int(codec_frame_duration_ms)
    if codec in NARROWBAND_CODECS:
        channel_path = "narrowband"
        channel_rate = 8000
        bandpass_hz = degradation_config["channel"]["narrowband"]["bandpass_hz"]
    elif codec in WIDEBAND_CODECS:
        channel_path = "wideband"
        channel_rate = 16000
        bandpass_hz = degradation_config["channel"]["wideband"]["bandpass_hz"]
    else:
        channel_entry = weighted_choice(rng, degradation_config["channel"]["pass_through_path_distribution"])
        channel_path = str(channel_entry["path"])
        channel_rate = 8000 if channel_path == "narrowband" else 16000
        bandpass_hz = degradation_config["channel"][channel_path]["bandpass_hz"]

    degraded_channel = resample_audio(degraded, working_rate, channel_rate)
    degraded_channel = bandpass_filter(degraded_channel, channel_rate, float(bandpass_hz[0]), float(bandpass_hz[1]))
    degraded_channel = codec_roundtrip(
        degraded_channel,
        channel_rate,
        codec,
        bitrate=str(codec_bitrate) if codec_bitrate is not None else None,
        frame_duration_ms=codec_frame_duration_ms,
    )
    degraded_channel = match_length(degraded_channel, len(resample_audio(degraded, working_rate, channel_rate)))

    network_cfg = degradation_config["network_impairment"]
    network_enabled = bool(network_cfg.get("enabled", True)) and rng.random() < float(network_cfg["probability"])
    network_metadata = {
        "enabled": network_enabled,
        "mode": None,
        "model": None,
        "loss_rate": None,
        "burst_length": None,
        "frame_ms": None,
        "dropout_ms": None,
        "dropped_frames": None,
        "total_frames": None,
        "observed_loss_rate": None,
    }
    if network_enabled:
        loss_bucket = network_cfg["loss_rate_buckets"][int(rng.integers(0, len(network_cfg["loss_rate_buckets"])))]
        loss_rate = sample_uniform(rng, loss_bucket)
        burst_length = int(rng.integers(int(network_cfg["burst_length"][0]), int(network_cfg["burst_length"][1]) + 1))
        frame_ms = int(network_cfg["frame_ms"])
        degraded_channel, dropped_frames, total_frames = apply_decoded_waveform_dropout(
            degraded_channel, channel_rate, rng, loss_rate, burst_length, frame_ms
        )
        observed_loss_rate = dropped_frames / max(1, total_frames)
        network_metadata = {
            "enabled": True,
            "mode": "decoded_waveform_dropout",
            "model": "two_state_burst",
            "loss_rate": loss_rate,
            "burst_length": burst_length,
            "frame_ms": frame_ms,
            "dropout_ms": dropped_frames * frame_ms,
            "dropped_frames": dropped_frames,
            "total_frames": total_frames,
            "observed_loss_rate": observed_loss_rate,
        }

    degraded_model = resample_audio(degraded_channel, channel_rate, model_rate)
    degraded_model = match_length(degraded_model, int(round(len(clean_working) * model_rate / working_rate)))
    degraded_model = peak_safety_normalize(degraded_model, peak=float(degradation_config["normalization"]["peak"]))

    clean_target = resample_audio(clean_working, working_rate, model_rate)
    if channel_path == "narrowband":
        target_channel = resample_audio(clean_working, working_rate, channel_rate)
        target_channel = bandpass_filter(target_channel, channel_rate, float(bandpass_hz[0]), float(bandpass_hz[1]))
        clean_target = resample_audio(target_channel, channel_rate, model_rate)
        target_bandwidth = "narrowband"
    elif bool(degradation_config["channel"]["wideband"].get("filter_target", False)):
        target_channel = resample_audio(clean_working, working_rate, channel_rate)
        target_channel = bandpass_filter(target_channel, channel_rate, float(bandpass_hz[0]), float(bandpass_hz[1]))
        clean_target = resample_audio(target_channel, channel_rate, model_rate)
        target_bandwidth = "wideband_filtered"
    else:
        target_bandwidth = "wideband"
    clean_target = peak_safety_normalize(match_length(clean_target, len(degraded_model)), peak=float(degradation_config["normalization"]["peak"]))

    metadata.update(
        {
            "target_bandwidth": target_bandwidth,
            "duration_sec": len(degraded_model) / model_rate,
            "channel_path": channel_path,
            "channel_sample_rate": channel_rate,
            "channel_bandpass_hz": [float(bandpass_hz[0]), float(bandpass_hz[1])],
            "codec": codec,
            "codec_bitrate": codec_bitrate,
            "codec_frame_duration_ms": codec_frame_duration_ms,
            "network_impairment": network_metadata,
            "normalization": degradation_config["normalization"]["mode"],
        }
    )
    return metadata, clean_target, degraded_model, model_rate


def default_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "seed": 1337,
        "model_sample_rate": 16000,
        "working_sample_rate": 16000,
        "variants_per_clip": 2,
        "output_dir": "data/speech_enhancement",
        "manifests": {},
        "rir_index": None,
        "noise_index": None,
        "reverb": {
            "severe": {"probability": 0.03, "wet_mix": [0.6, 0.8], "dr_db": [6, 10]},
            "mild": {"probability": 0.15, "wet_mix": [0.3, 0.5], "dr_db": [12, 18]},
        },
        "noise": {"probability": 0.60, "second_scene_probability": 0.10, "snr_buckets": [[10, 15], [5, 10], [0, 5], [-5, 0]]},
        "level": {"gain_db": [-6, 6], "clipping": {"enabled": False, "probability": 0.1, "mode": "hard", "threshold": [0.8, 0.98]}, "agc": {"enabled": False}},
        "channel": {
            "narrowband": {"bandpass_hz": [300, 3400]},
            "wideband": {"bandpass_hz": [50, 7000], "filter_target": False},
            "pass_through_path_distribution": [{"path": "narrowband", "weight": 0.5}, {"path": "wideband", "weight": 0.5}],
        },
        "codec_distribution": [
            {"codec": "g711_alaw", "weight": 0.30},
            {"codec": "g711_mulaw", "weight": 0.10},
            {"codec": "gsm", "weight": 0.10},
            {"codec": "amr_wb_12k65", "weight": 0.25},
            {"codec": "amr_nb_12k2", "weight": 0.15},
            {"codec": "opus_wb", "weight": 0.05},
            {"codec": "opus_nb", "weight": 0.05},
            {"codec": "pass_through", "weight": 0.10},
        ],
        "profiles": None,
        "network_impairment": {"enabled": True, "probability": 0.60, "loss_rate_buckets": [[0.003, 0.02], [0.02, 0.05], [0.05, 0.10]], "burst_length": [1, 5], "frame_ms": 20},
        "normalization": {"mode": "peak_safety", "peak": 0.99},
    }
    return deep_merge(merged, config)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_config(config: dict[str, Any]) -> None:
    manifests = config["manifests"]
    for split in ("train", "valid"):
        if split not in manifests:
            raise ValueError(f"config.manifests.{split} is required")
    validate_distribution("codec", config["codec_distribution"])
    validate_distribution("pass_through_path", config["channel"]["pass_through_path_distribution"])
    profiles = config.get("profiles") or []
    if profiles:
        validate_distribution("profile", profiles)
    for index, profile in enumerate(profiles, start=1):
        if "name" not in profile:
            raise ValueError(f"profile {index} is missing required key: name")
        if "codec_distribution" in profile:
            validate_distribution(f"profile {profile['name']} codec", profile["codec_distribution"])
        if "channel" in profile and "pass_through_path_distribution" in profile["channel"]:
            validate_distribution(
                f"profile {profile['name']} pass_through_path",
                profile["channel"]["pass_through_path_distribution"],
            )
    severe = float(config["reverb"]["severe"]["probability"])
    mild = float(config["reverb"]["mild"]["probability"])
    if severe + mild > 1:
        raise ValueError("reverb severe + mild probabilities must be <= 1")
    require_ffmpeg_codecs(config)


def generate_from_config(config: dict[str, Any]) -> dict[str, Any]:
    config = default_config(config)
    validate_config(config)
    config_base = Path.cwd()
    rir_assets = load_asset_index(resolve_path(config.get("rir_index"), config_base))
    noise_assets = load_asset_index(resolve_path(config.get("noise_index"), config_base))
    report: dict[str, Any] = {"splits": {}, "skipped": []}
    output_dir = Path(config["output_dir"])
    manifest_dir = output_dir / "manifests"

    for split in ("train", "valid"):
        manifest_path = resolve_path(config["manifests"][split], config_base)
        if manifest_path is None:
            raise ValueError(f"manifest path for {split} is required")
        items = load_clean_manifest(manifest_path, expected_split=split)
        rows: list[dict[str, Any]] = []
        iterator = tqdm(items, desc=f"degrading {split}", unit="clip")
        for item in iterator:
            for variant_index in range(int(config["variants_per_clip"])):
                try:
                    rows.append(process_item(item, variant_index, config, rir_assets, noise_assets))
                except sf.LibsndfileError as exc:
                    report["skipped"].append({"id": item.id, "split": split, "variant_index": variant_index, "error": str(exc)})
        out_manifest = manifest_dir / f"se_{split}_pairs.jsonl"
        write_jsonl(out_manifest, rows)
        report["splits"][split] = {"input_clips": len(items), "pairs": len(rows), "manifest": str(out_manifest)}

    report_path = manifest_dir / "generation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config root must be a mapping")
    return loaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate paired clean/degraded speech data.")
    parser.add_argument("--config", required=True, help="Path to degradation YAML config.")
    args = parser.parse_args(argv)
    report = generate_from_config(load_config(Path(args.config)))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
