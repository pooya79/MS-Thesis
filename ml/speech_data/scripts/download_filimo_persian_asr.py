from __future__ import annotations

import argparse
from pathlib import Path

from ml.speech_data.scripts.persets_asr import (
    DownloadAudit,
    FILIMO_DATASET,
    SnapshotDownloader,
    default_worker_count,
    download_persets_source,
    print_download_audit,
    snapshot_download,
)


def download_filimo_persian_asr(
    output_root: Path,
    *,
    revision: str = "main",
    workers: int = 8,
    force: bool = False,
    downloader: SnapshotDownloader = snapshot_download,
) -> DownloadAudit:
    return download_persets_source(
        FILIMO_DATASET,
        output_root,
        revision=revision,
        workers=workers,
        force=force,
        downloader=downloader,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the PerSets Filimo Persian ASR metadata and tar shards from Hugging Face."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=FILIMO_DATASET.default_source_root,
        help="Directory where unvalidated.csv and data/*.tar will be cached.",
    )
    parser.add_argument("--revision", default="main", help="Hugging Face dataset revision to download.")
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Number of parallel Hugging Face download workers.",
    )
    parser.add_argument("--force", action="store_true", help="Redownload files even when cached copies exist.")
    args = parser.parse_args(argv)

    audit = download_filimo_persian_asr(
        args.output_root,
        revision=args.revision,
        workers=args.workers,
        force=args.force,
    )
    print_download_audit(FILIMO_DATASET, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
