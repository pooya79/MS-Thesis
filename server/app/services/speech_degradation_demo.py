from __future__ import annotations

import json
import secrets
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from ml.speech_data.generate_degraded_pairs import (
    ManifestItem,
    default_config,
    load_asset_index,
    load_config,
    process_item,
    require_ffmpeg_codecs,
    resolve_path,
)
from ml.utils.audio import load_audio, save_audio


DEMO_ROOT = Path("server/data/degradation_demos")
DEFAULT_CONFIG_PATH = Path("configs/speech_enhancement/degradation.yaml")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DURATION_SECONDS = 60.0
ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".webm"}
FILE_KINDS = {
    "input": "input.wav",
    "clean_target": "clean_target.wav",
    "degraded": "degraded.wav",
}
NARROWBAND_CODECS = {"pass_through", "g711_alaw", "g711_mulaw", "gsm", "amr_nb_12k2", "opus_nb"}
WIDEBAND_CODECS = {"pass_through", "amr_wb_12k65", "opus_wb"}


class DemoValidationError(ValueError):
    """A user-facing validation error for the speech degradation demo."""


@dataclass(frozen=True)
class DemoResult:
    demo_id: str
    demo_dir: Path
    metadata: dict[str, Any]


def create_demo_id() -> str:
    return secrets.token_urlsafe(12)


def demo_dir(demo_id: str) -> Path:
    return DEMO_ROOT / demo_id


def demo_file_path(demo_id: str, kind: str) -> Path:
    if kind not in FILE_KINDS:
        raise DemoValidationError("Unknown generated file kind.")
    return demo_dir(demo_id) / FILE_KINDS[kind]


def validate_upload(filename: str, content: bytes) -> str:
    extension = Path(filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise DemoValidationError(f"Unsupported audio file type. Use one of: {allowed}.")
    if not content:
        raise DemoValidationError("Upload an audio file before generating a demo.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise DemoValidationError("Uploaded audio is larger than 25 MB.")
    return extension


def convert_upload_to_wav(content: bytes, extension: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="degradation_upload_") as tmp:
        input_path = Path(tmp) / f"upload{extension}"
        input_path.write_bytes(content)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise DemoValidationError("Could not decode the uploaded audio with ffmpeg.") from exc


def assert_duration_limit(path: Path) -> None:
    try:
        info = sf.info(str(path))
    except sf.LibsndfileError as exc:
        raise DemoValidationError("Could not inspect the decoded audio file.") from exc
    if info.duration <= 0:
        raise DemoValidationError("Uploaded audio has no usable duration.")
    if info.duration > MAX_DURATION_SECONDS:
        raise DemoValidationError("Uploaded audio is longer than 60 seconds.")


def load_base_config() -> dict[str, Any]:
    if DEFAULT_CONFIG_PATH.exists():
        return default_config(load_config(DEFAULT_CONFIG_PATH))
    return default_config({})


def validate_channel_codec(channel_path: str, codec: str) -> None:
    if channel_path == "narrowband" and codec not in NARROWBAND_CODECS:
        raise DemoValidationError("The selected codec is not valid for a narrowband channel.")
    if channel_path == "wideband" and codec not in WIDEBAND_CODECS:
        raise DemoValidationError("The selected codec is not valid for a wideband channel.")
    if channel_path not in {"narrowband", "wideband"}:
        raise DemoValidationError("Choose a valid channel path.")


def require_assets(config: dict[str, Any], key: str, effect_name: str) -> list[dict[str, Any]]:
    path_value = config.get(key)
    path = resolve_path(path_value, Path.cwd())
    if path is None or not path.exists():
        raise DemoValidationError(
            f"{effect_name} is enabled, but its asset index does not exist. Disable {effect_name.lower()} or install/index the assets."
        )
    try:
        assets = load_asset_index(path)
    except (FileNotFoundError, ValueError) as exc:
        raise DemoValidationError(
            f"{effect_name} is enabled, but its asset index is invalid. Disable {effect_name.lower()} or install/index the assets."
        ) from exc
    if not assets:
        raise DemoValidationError(f"{effect_name} is enabled, but the asset index is empty.")
    return assets


def build_demo_config(form: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_config = load_base_config()
    channel_path = str(form["channel_path"])
    codec = str(form["codec"])
    validate_channel_codec(channel_path, codec)

    noise_enabled = bool(form.get("noise_enabled"))
    network_enabled = bool(form.get("network_enabled"))
    clipping_enabled = bool(form.get("clipping_enabled"))
    gain_db = float(form["gain_db"])
    snr_bounds = [float(value) for value in str(form["snr_bucket"]).split(":")]
    if len(snr_bounds) != 2:
        raise DemoValidationError("Choose a valid SNR bucket.")

    config = default_config(
        {
            "seed": int(form.get("seed", 1337)),
            "variants_per_clip": 1,
            "output_dir": str(output_dir),
            "noise": {
                "probability": 1.0 if noise_enabled else 0.0,
                "second_scene_probability": 0.0,
                "snr_buckets": [snr_bounds],
            },
            "level": {
                "gain_db": [gain_db, gain_db],
                "clipping": {
                    "enabled": clipping_enabled,
                    "probability": 1.0 if clipping_enabled else 0.0,
                    "mode": "hard",
                    "threshold": [0.9, 0.9],
                },
                "agc": {"enabled": False},
            },
            "channel": {
                "narrowband": base_config["channel"]["narrowband"],
                "wideband": base_config["channel"]["wideband"],
                "pass_through_path_distribution": [{"path": channel_path, "weight": 1.0}],
            },
            "codec_distribution": [{"codec": codec, "weight": 1.0}],
            "network_impairment": {
                "enabled": network_enabled,
                "probability": 1.0 if network_enabled else 0.0,
                "loss_rate_buckets": [[0.02, 0.05]],
                "burst_length": [2, 4],
                "frame_ms": 20,
            },
            "normalization": {"mode": "peak_safety", "peak": 0.99},
            "noise_index": base_config.get("noise_index"),
        }
    )
    require_ffmpeg_codecs(config)
    noise_assets = require_assets(base_config, "noise_index", "Noise") if noise_enabled else []
    return config, noise_assets


def generate_demo(content: bytes, filename: str, form: dict[str, Any], session_demo_ids: list[str]) -> DemoResult:
    extension = validate_upload(filename, content)
    demo_id = create_demo_id()
    root = demo_dir(demo_id)
    root.mkdir(parents=True, exist_ok=True)
    try:
        input_path = root / "input.wav"
        convert_upload_to_wav(content, extension, input_path)
        assert_duration_limit(input_path)

        config, noise_assets = build_demo_config(form, root / "generated")
        item = ManifestItem(id="upload", split="demo", clean_path=input_path, transcript=None)
        metadata = process_item(item, 0, config, noise_assets)

        clean_target = root / "clean_target.wav"
        degraded = root / "degraded.wav"
        shutil.move(metadata["clean_path"], clean_target)
        shutil.move(metadata["degraded_path"], degraded)
        shutil.rmtree(root / "generated", ignore_errors=True)

        input_audio, input_rate = load_audio(input_path)
        save_audio(input_path, input_audio, input_rate)

        metadata.update(
            {
                "demo_id": demo_id,
                "input_path": str(input_path),
                "clean_path": str(clean_target),
                "degraded_path": str(degraded),
                "ui_parameters": {
                    "noise_enabled": bool(form.get("noise_enabled")),
                    "gain_db": float(form["gain_db"]),
                    "clipping_enabled": bool(form.get("clipping_enabled")),
                    "channel_path": form["channel_path"],
                    "codec": form["codec"],
                    "network_enabled": bool(form.get("network_enabled")),
                },
            }
        )
        (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        cleanup_session_demos([demo_id, *session_demo_ids])
        return DemoResult(demo_id=demo_id, demo_dir=root, metadata=metadata)
    except DemoValidationError:
        shutil.rmtree(root, ignore_errors=True)
        raise
    except (sf.LibsndfileError, subprocess.CalledProcessError, ValueError, RuntimeError) as exc:
        shutil.rmtree(root, ignore_errors=True)
        raise DemoValidationError(f"Could not generate degraded audio: {exc}") from exc


def cleanup_session_demos(session_demo_ids: list[str], keep: int = 10) -> list[str]:
    kept = session_demo_ids[:keep]
    for old_demo_id in session_demo_ids[keep:]:
        shutil.rmtree(demo_dir(old_demo_id), ignore_errors=True)
    return kept
