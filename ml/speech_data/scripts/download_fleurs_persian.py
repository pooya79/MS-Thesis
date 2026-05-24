from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf
from tqdm import tqdm


DEFAULT_DATASET_NAME = "google/fleurs"
DEFAULT_CONFIG_NAME = "fa_ir"
SPLITS = ("train", "validation", "test")


@dataclass
class DownloadAudit:
    train_rows: int = 0
    validation_rows: int = 0
    test_rows: int = 0
    audio_written: int = 0


def stable_id(value: object, *, split: str, index: int, config_name: str) -> str:
    raw = str(value) if value not in (None, "") else f"fleurs-{config_name}-{split}-{index:06d}"
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    normalized = normalized or f"fleurs-{config_name}-{split}-{index:06d}"
    if normalized.startswith(f"{split}-"):
        return normalized
    return f"{split}-{normalized}"


def load_fleurs_dataset(dataset_name: str, config_name: str) -> Any:
    from datasets import Audio, load_dataset

    dataset = load_dataset(dataset_name, config_name)
    return dataset.cast_column("audio", Audio(decode=False))


def row_text(row: dict[str, Any]) -> tuple[str, str]:
    transcription = str(row.get("transcription") or row.get("sentence") or "")
    raw_transcription = str(row.get("raw_transcription") or transcription)
    return transcription, raw_transcription


def source_audio_path(audio: object) -> str | None:
    if isinstance(audio, dict):
        path = audio.get("path")
        return str(path) if path else None
    return None


def write_audio(audio: object, output_path: Path) -> int:
    if not isinstance(audio, dict):
        raise ValueError("FLEURS row audio must be a dictionary")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = int(audio.get("sampling_rate") or 16000)

    audio_bytes = audio.get("bytes")
    if isinstance(audio_bytes, bytes):
        output_path.write_bytes(audio_bytes)
        return sample_rate

    path = audio.get("path")
    if path:
        source_path = Path(str(path))
        if source_path.exists():
            shutil.copyfile(source_path, output_path)
            return sample_rate

    if "array" in audio and "sampling_rate" in audio:
        sample_rate = int(audio["sampling_rate"])
        sf.write(output_path, audio["array"], sample_rate)
        return sample_rate

    raise ValueError("FLEURS row audio must contain bytes, an existing path, or array and sampling_rate")


def export_split(
    rows: Iterable[dict[str, Any]],
    *,
    split: str,
    output_root: Path,
    dataset_name: str,
    config_name: str,
    force: bool,
) -> int:
    jsonl_path = output_root / f"{split}.jsonl"
    if jsonl_path.exists() and not force:
        raise FileExistsError(f"{jsonl_path} already exists; pass --force to overwrite")

    output_root.mkdir(parents=True, exist_ok=True)
    count = 0
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(tqdm(rows, desc=f"Exporting {split}", unit="row")):
            row_id = stable_id(row.get("id"), split=split, index=index, config_name=config_name)
            audio = row.get("audio")
            audio_path = output_root / "audio" / split / f"{row_id}.wav"
            if audio_path.exists() and not force:
                raise FileExistsError(f"{audio_path} already exists; pass --force to overwrite")

            sample_rate = write_audio(audio, audio_path)
            transcription, raw_transcription = row_text(row)
            record = {
                "id": row_id,
                "hf_id": row.get("id"),
                "dataset_name": dataset_name,
                "config_name": config_name,
                "split": split,
                "audio_path": str(audio_path.relative_to(output_root)),
                "source_audio_path": source_audio_path(audio),
                "sample_rate": sample_rate,
                "transcription": transcription,
                "raw_transcription": raw_transcription,
                "language": row.get("language"),
                "lang_id": row.get("lang_id"),
                "lang_group_id": row.get("lang_group_id"),
                "gender": row.get("gender"),
                "num_samples": row.get("num_samples"),
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def download_fleurs_persian(
    output_root: Path,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    config_name: str = DEFAULT_CONFIG_NAME,
    force: bool = False,
    dataset: Any | None = None,
) -> DownloadAudit:
    dataset = dataset if dataset is not None else load_fleurs_dataset(dataset_name, config_name)
    missing = [split for split in SPLITS if split not in dataset]
    if missing:
        raise ValueError(f"dataset is missing required splits: {', '.join(missing)}")

    counts: dict[str, int] = {}
    for split in SPLITS:
        counts[split] = export_split(
            dataset[split],
            split=split,
            output_root=output_root,
            dataset_name=dataset_name,
            config_name=config_name,
            force=force,
        )

    total = sum(counts.values())
    return DownloadAudit(
        train_rows=counts["train"],
        validation_rows=counts["validation"],
        test_rows=counts["test"],
        audio_written=total,
    )


def print_audit(audit: DownloadAudit) -> None:
    print("FLEURS Persian download summary")
    print(f"  train rows: {audit.train_rows}")
    print(f"  validation rows: {audit.validation_rows}")
    print(f"  test rows: {audit.test_rows}")
    print(f"  audio written: {audit.audio_written}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download and export the Persian FLEURS subset from Hugging Face into "
            "local JSONL manifests and WAV audio files."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/fleurs/fa_ir/source"),
        help="Directory where train/validation/test JSONL files and audio/ will be written.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset name to load.",
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Hugging Face FLEURS language/config name.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing exported JSONL and audio files.",
    )
    args = parser.parse_args(argv)

    audit = download_fleurs_persian(
        args.output_root,
        dataset_name=args.dataset_name,
        config_name=args.config_name,
        force=args.force,
    )
    print_audit(audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
