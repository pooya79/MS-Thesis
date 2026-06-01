from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ml.speech_data.generate_degraded_pairs import (
    ManifestItem,
    default_config,
    load_asset_index,
    load_config,
    process_item,
    require_ffmpeg_codecs,
    resolve_path,
    write_jsonl,
)


DEFAULT_EXTENSIONS = (".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a")


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def find_audio_candidates(input_root: Path, output_dir: Path, extensions: tuple[str, ...]) -> list[Path]:
    normalized_extensions = {extension.lower() for extension in extensions}
    candidates: list[Path] = []
    for path in input_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in normalized_extensions:
            continue
        if is_relative_to(path, output_dir):
            continue
        candidates.append(path)
    return sorted(candidates)


def choose_random_readable_audio(
    input_root: Path,
    output_dir: Path,
    seed: int,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
) -> Path:
    candidates = find_audio_candidates(input_root, output_dir, extensions)
    if not candidates:
        raise FileNotFoundError(f"no audio files found under {input_root}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))
    for index in order:
        candidate = candidates[int(index)]
        try:
            sf.info(str(candidate))
        except sf.LibsndfileError:
            continue
        return candidate
    raise FileNotFoundError(f"no readable audio files found under {input_root}")


def load_optional_assets(config: dict[str, Any], key: str, base: Path) -> list[dict[str, Any]]:
    value = config.get(key)
    path = resolve_path(value if isinstance(value, str) else None, base)
    if path is None or not path.exists():
        return []
    return load_asset_index(path)


def generate_random_degraded_clip(
    *,
    config_path: Path,
    input_root: Path,
    output_dir: Path,
    variants: int,
    seed: int,
) -> dict[str, object]:
    if variants < 1:
        raise ValueError("variants must be at least 1")

    config = default_config(load_config(config_path))
    config["seed"] = seed

    selected_audio = choose_random_readable_audio(input_root, output_dir, seed)
    run_dir = output_dir / f"{selected_audio.stem}_seed{seed}"
    config["output_dir"] = str(run_dir)
    config["variants_per_clip"] = variants

    require_ffmpeg_codecs(config)
    config_base = Path.cwd()
    rir_assets = load_optional_assets(config, "rir_index", config_base)
    noise_assets = load_optional_assets(config, "noise_index", config_base)

    item = ManifestItem(id=selected_audio.stem, split="demo", clean_path=selected_audio)
    rows = [process_item(item, variant_index, config, rir_assets, noise_assets) for variant_index in range(variants)]

    manifest_dir = run_dir / "manifests"
    manifest_path = manifest_dir / "random_demo_pairs.jsonl"
    write_jsonl(manifest_path, rows)

    report = {
        "selected_audio": str(selected_audio),
        "output_dir": str(run_dir),
        "manifest": str(manifest_path),
        "variants": variants,
        "seed": seed,
        "degraded_paths": [row["degraded_path"] for row in rows],
        "clean_paths": [row["clean_path"] for row in rows],
    }
    report_path = manifest_dir / "random_demo_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate several degraded variants of one random audio clip from a data folder."
    )
    parser.add_argument(
        "--config",
        default="configs/speech_enhancement/degradation.yaml",
        help="Path to the degradation YAML config. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--input-root",
        default="data",
        help="Folder to recursively scan for source audio. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/speech_enhancement/random_clip_degradations",
        help="Folder where the selected clip run folder is written. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=8,
        help="Number of degraded variants to generate. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed used for clip selection and degradation. Defaults to %(default)s.",
    )
    args = parser.parse_args(argv)

    report = generate_random_degraded_clip(
        config_path=Path(args.config),
        input_root=Path(args.input_root),
        output_dir=Path(args.output_dir),
        variants=args.variants,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
