from __future__ import annotations

import argparse
from pathlib import Path

from ml.speech_data.scripts.persets_asr import (
    AudioConverter,
    FILIMO_DATASET,
    PreparationAudit,
    convert_mp3_bytes,
    default_worker_count,
    prepare_persets_dataset,
    print_preparation_audit,
)


def prepare_filimo_persian_asr(
    source_root: Path,
    output_root: Path,
    *,
    workers: int = 1,
    force: bool = False,
    converter: AudioConverter = convert_mp3_bytes,
    show_progress: bool = True,
) -> PreparationAudit:
    return prepare_persets_dataset(
        FILIMO_DATASET,
        source_root,
        output_root,
        workers=workers,
        force=force,
        converter=converter,
        show_progress=show_progress,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare PerSets Filimo Persian ASR as normalized train.tsv data "
            "with mono 16 kHz PCM-16 WAV clips."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=FILIMO_DATASET.default_source_root,
        help="Downloaded source directory containing unvalidated.csv and data/*.tar.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=FILIMO_DATASET.default_output_root,
        help="Output directory for train.tsv and clips/*.wav.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Number of parallel ffmpeg conversion worker processes.",
    )
    parser.add_argument("--force", action="store_true", help="Reconvert WAV files that already exist.")
    args = parser.parse_args(argv)

    audit = prepare_filimo_persian_asr(
        args.source_root,
        args.output_root,
        workers=args.workers,
        force=args.force,
    )
    print_preparation_audit(FILIMO_DATASET, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
