from __future__ import annotations

import argparse
import logging
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

from mutagen import File as MutagenFile


AUDIO_EXTENSIONS = frozenset({".flac", ".mp3", ".wav"})
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioDurationSummary:
    discovered_files: int
    processed_files: int
    failed_files: int
    total_seconds: float

    @property
    def total_hours(self) -> float:
        return self.total_seconds / 3_600


@dataclass(frozen=True)
class AudioDurationResult:
    path: Path
    seconds: float | None
    error: str | None = None


def discover_audio_files(directory: Path) -> list[Path]:
    """Return supported audio files below directory in deterministic order."""
    if not directory.is_dir():
        raise NotADirectoryError(f"audio directory does not exist or is not a directory: {directory}")
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def read_audio_duration(path: Path) -> AudioDurationResult:
    """Read duration from an audio container without decoding its samples."""
    try:
        audio = MutagenFile(path)
        if audio is None or getattr(audio, "info", None) is None:
            raise ValueError("unsupported or unreadable audio metadata")
        seconds = float(audio.info.length)
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"invalid duration: {seconds}")
        return AudioDurationResult(path=path, seconds=seconds)
    except Exception as exc:  # Mutagen exposes format-specific parsing exceptions.
        return AudioDurationResult(path=path, seconds=None, error=str(exc))


def compute_audio_hours(
    directory: Path,
    *,
    workers: int,
    logger: logging.Logger = LOGGER,
) -> AudioDurationSummary:
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")

    paths = discover_audio_files(directory)
    total_files = len(paths)
    logger.info("Discovered %d audio files under %s", total_files, directory)

    total_seconds = 0.0
    processed_files = 0
    failed_files = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
    ) as executor:
        futures = {executor.submit(read_audio_duration, path): path for path in paths}
        for completed, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # Protect the scan if a worker exits unexpectedly.
                result = AudioDurationResult(path=path, seconds=None, error=str(exc))

            if result.seconds is None:
                failed_files += 1
                logger.error(
                    "[%d/%d] Failed %s: %s",
                    completed,
                    total_files,
                    result.path,
                    result.error,
                )
                continue

            processed_files += 1
            total_seconds += result.seconds
            logger.info(
                "[%d/%d] %s: %.3f s (running total: %.6f h)",
                completed,
                total_files,
                result.path,
                result.seconds,
                total_seconds / 3_600,
            )

    return AudioDurationSummary(
        discovered_files=total_files,
        processed_files=processed_files,
        failed_files=failed_files,
        total_seconds=total_seconds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively compute the total duration of FLAC, WAV, and MP3 files. "
            "Mutagen reads metadata without decoding audio, and worker processes "
            "inspect files concurrently."
        )
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory to scan recursively for .flac, .wav, and .mp3 files.",
    )
    parser.add_argument(
        "--workers",
        required=True,
        type=int,
        help="Number of worker processes (must be at least 1).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        summary = compute_audio_hours(args.directory, workers=args.workers)
    except (NotADirectoryError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2

    print(f"Total hours: {summary.total_hours:.6f}")
    print(f"Total seconds: {summary.total_seconds:.3f}")
    print(f"Files processed: {summary.processed_files}/{summary.discovered_files}")
    print(f"Files failed: {summary.failed_files}")
    return 1 if summary.failed_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
