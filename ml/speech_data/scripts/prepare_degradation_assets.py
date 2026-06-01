from __future__ import annotations

import argparse
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from tqdm import tqdm

from ml.utils.audio import resample_audio, to_mono


@dataclass
class AssetAudit:
    archives_extracted: int = 0
    noise_candidates: int = 0
    noise_indexed: int = 0
    unreadable: int = 0


def safe_extract_zip(path: Path, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            target = (output_dir / member).resolve()
            if output_root != target and output_root not in target.parents:
                raise ValueError(f"refusing to extract unsafe archive member: {member}")
        archive.extractall(output_dir)


def extract_archives(root: Path) -> int:
    if not root.exists():
        return 0

    count = 0
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() == ".zip":
            safe_extract_zip(path, path.parent)
            count += 1
    return count


def scene_from_noise_path(path: Path, noise_root: Path) -> str:
    try:
        relative_parts = path.relative_to(noise_root).parts
    except ValueError:
        relative_parts = path.parts

    for part in relative_parts:
        if part.upper().endswith("_16K"):
            return part[:-4].lower()
    return path.parent.name.lower()


def stable_asset_id(prefix: str, path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    safe = "_".join(relative.with_suffix("").parts)
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in safe)
    return f"{prefix}-{safe}"


def path_for_manifest(path: Path, manifest_dir: Path, *, absolute_paths: bool) -> str:
    if absolute_paths:
        return str(path.resolve())
    return os.path.relpath(path.resolve(), manifest_dir.resolve())


def validate_audio(path: Path, sample_rate: int) -> bool:
    try:
        with sf.SoundFile(path) as handle:
            frames = min(handle.frames, max(handle.samplerate, 1))
            audio = handle.read(frames=frames, dtype="float32", always_2d=True)
            source_rate = int(handle.samplerate)
    except (RuntimeError, OSError, ValueError):
        return False

    if source_rate <= 0 or audio.size == 0:
        return False
    mono = to_mono(np.asarray(audio, dtype=np.float32))
    try:
        resample_audio(mono, source_rate, sample_rate)
    except Exception:
        return False
    return True


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_noise_rows(
    noise_root: Path,
    manifest_dir: Path,
    audit: AssetAudit,
    *,
    sample_rate: int,
    absolute_paths: bool,
    show_progress: bool,
) -> list[dict[str, object]]:
    candidates = sorted(noise_root.rglob("*.wav"))
    audit.noise_candidates = len(candidates)
    rows: list[dict[str, object]] = []
    for path in tqdm(candidates, desc="Indexing noise", unit="file", disable=not show_progress):
        if not validate_audio(path, sample_rate):
            audit.unreadable += 1
            continue
        scene = scene_from_noise_path(path, noise_root)
        rows.append(
            {
                "id": stable_asset_id("demand", path, noise_root),
                "scene": scene,
                "path": path_for_manifest(path, manifest_dir, absolute_paths=absolute_paths),
            }
        )
    audit.noise_indexed = len(rows)
    return rows


def prepare_degradation_assets(
    noise_root: Path,
    manifest_dir: Path,
    *,
    sample_rate: int = 16000,
    extract: bool = True,
    absolute_paths: bool = False,
    show_progress: bool = True,
) -> AssetAudit:
    audit = AssetAudit()
    if extract:
        audit.archives_extracted += extract_archives(noise_root)

    manifest_dir.mkdir(parents=True, exist_ok=True)
    noise_rows = build_noise_rows(
        noise_root,
        manifest_dir,
        audit,
        sample_rate=sample_rate,
        absolute_paths=absolute_paths,
        show_progress=show_progress,
    )

    write_jsonl(manifest_dir / "demand_noise_index.jsonl", noise_rows)
    return audit


def print_audit(audit: AssetAudit) -> None:
    print("Degradation asset preparation summary")
    print(f"  archives extracted: {audit.archives_extracted}")
    print(f"  noise candidates: {audit.noise_candidates}")
    print(f"  noise indexed: {audit.noise_indexed}")
    print(f"  unreadable or invalid files: {audit.unreadable}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a speech-degradation DEMAND noise asset index from "
            "already-downloaded DEMAND archives or extracted WAV files."
        )
    )
    parser.add_argument(
        "--noise-root",
        type=Path,
        default=Path("data/speech_enhancement/assets/noise/DEMAND"),
        help="Directory containing DEMAND *_16k.zip archives or extracted DEMAND WAV files.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/speech_enhancement/manifests"),
        help="Output directory for demand_noise_index.jsonl.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate used for readability/resampling validation.",
    )
    parser.add_argument(
        "--extract-archives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract .zip archives found under the noise root before indexing.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute paths in the indexes. By default paths are relative when possible.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress bars.",
    )
    args = parser.parse_args(argv)

    audit = prepare_degradation_assets(
        args.noise_root,
        args.manifest_dir,
        sample_rate=args.sample_rate,
        extract=args.extract_archives,
        absolute_paths=args.absolute_paths,
        show_progress=not args.quiet,
    )
    print_audit(audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
