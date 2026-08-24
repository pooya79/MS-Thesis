"""Build and slowly upload resumable Parquet shards for an audio dataset."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Audio, Features, Value
from huggingface_hub import CommitOperationAdd, HfApi

from ml.speech_data.scripts.hf_audio_dataset import (
    AudioDatasetRow,
    load_audio_dataset_rows,
    load_segment_metadata,
    sha256_file,
)


LOGGER = logging.getLogger(__name__)
STATE_SCHEMA_VERSION = "hf-parquet-audio-upload-state-v1"
SEGMENT_FIELDS: dict[str, str] = {
    "source_id": "string",
    "duration_sec": "float64",
    "start_sec": "float64",
    "end_sec": "float64",
    "speech_seconds": "float64",
    "speech_ratio": "float64",
    "boundary_type": "string",
    "clip_checksum": "string",
}


@dataclass(frozen=True)
class UploadSettings:
    repo_id: str
    revision: str
    config_name: str
    split: str
    target_shard_bytes: int
    row_group_size: int
    max_workers: int
    sleep_seconds: float
    max_shards: int | None
    max_bytes: int | None
    retries: int


@dataclass(frozen=True)
class ShardSpec:
    index: int
    total: int
    start_row: int
    stop_row: int
    source_bytes: int
    path_in_repo: str


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _features() -> Features:
    values: dict[str, Any] = {
        "audio": Audio(),
        "id": Value("string"),
        "sentence": Value("string"),
    }
    values.update({name: Value(dtype) for name, dtype in SEGMENT_FIELDS.items()})
    return Features(values)


def plan_shards(
    rows: Sequence[AudioDatasetRow], *, target_shard_bytes: int, config_name: str, split: str
) -> list[ShardSpec]:
    if target_shard_bytes < 1:
        raise ValueError("target shard size must be at least 1 byte")
    boundaries: list[tuple[int, int, int]] = []
    start = 0
    running_bytes = 0
    for index, row in enumerate(rows):
        size = row.audio_path.stat().st_size
        if running_bytes and running_bytes + size > target_shard_bytes:
            boundaries.append((start, index, running_bytes))
            start = index
            running_bytes = 0
        running_bytes += size
    if start < len(rows):
        boundaries.append((start, len(rows), running_bytes))
    total = len(boundaries)
    width = max(5, len(str(max(total - 1, 0))))
    return [
        ShardSpec(
            index=index,
            total=total,
            start_row=start_row,
            stop_row=stop_row,
            source_bytes=source_bytes,
            path_in_repo=(
                f"{config_name}/{split}-{index:0{width}d}-of-{total:0{width}d}.parquet"
            ),
        )
        for index, (start_row, stop_row, source_bytes) in enumerate(boundaries)
    ]


def write_parquet_shard(
    path: Path,
    *,
    rows: Sequence[AudioDatasetRow],
    segment_metadata: dict[str, dict[str, Any]],
    row_group_size: int,
) -> None:
    if row_group_size < 1:
        raise ValueError("row group size must be at least 1")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    schema = _features().arrow_schema
    try:
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            for offset in range(0, len(rows), row_group_size):
                batch: list[dict[str, Any]] = []
                for row in rows[offset : offset + row_group_size]:
                    record: dict[str, Any] = {
                        "audio": {
                            "bytes": row.audio_path.read_bytes(),
                            "path": row.relative_path,
                        },
                        "id": row.example_id,
                        "sentence": row.sentence,
                    }
                    metadata = segment_metadata.get(row.example_id, {})
                    record.update({name: metadata.get(name) for name in SEGMENT_FIELDS})
                    batch.append(record)
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _retry(
    action: Callable[[], Any], *, retries: int, description: str, sleep: Callable[[float], None]
) -> Any:
    for attempt in range(retries + 1):
        try:
            return action()
        except Exception:
            if attempt >= retries:
                raise
            delay = min(60.0, 2.0**attempt)
            LOGGER.warning(
                "%s failed (attempt %d/%d); retrying in %.1f seconds",
                description,
                attempt + 1,
                retries + 1,
                delay,
                exc_info=True,
            )
            sleep(delay)
    raise AssertionError("unreachable")


def _state_identity(
    *,
    manifest: Path,
    segments_manifest: Path | None,
    settings: UploadSettings,
) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "repo_id": settings.repo_id,
        "revision": settings.revision,
        "config_name": settings.config_name,
        "split": settings.split,
        "target_shard_bytes": settings.target_shard_bytes,
        "row_group_size": settings.row_group_size,
        "manifest_sha256": sha256_file(manifest),
        "segments_manifest_sha256": (
            sha256_file(segments_manifest) if segments_manifest is not None else None
        ),
    }


def _load_state(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {**identity, "uploaded_shards": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    mismatches = [key for key, value in identity.items() if state.get(key) != value]
    if mismatches:
        raise ValueError(
            f"state file {path} belongs to a different immutable shard plan: "
            + ", ".join(mismatches)
            + ". Use a new config name and state file for a changed or larger manifest."
        )
    if not isinstance(state.get("uploaded_shards"), dict):
        raise ValueError(f"state file {path} has no valid uploaded_shards mapping")
    return state


def _choose_run_shards(
    pending: Sequence[ShardSpec], *, max_shards: int | None, max_bytes: int | None
) -> list[ShardSpec]:
    selected: list[ShardSpec] = []
    selected_bytes = 0
    for shard in pending:
        if max_shards is not None and len(selected) >= max_shards:
            break
        if max_bytes is not None and selected and selected_bytes + shard.source_bytes > max_bytes:
            break
        if max_bytes is not None and not selected and shard.source_bytes > max_bytes:
            raise ValueError(
                "max-gib is smaller than the first pending shard; increase max-gib or "
                "decrease target-shard-mib"
            )
        selected.append(shard)
        selected_bytes += shard.source_bytes
    return selected


def upload_dataset(
    *,
    manifest: Path,
    audio_root: Path,
    segments_manifest: Path | None,
    state_path: Path,
    temp_dir: Path,
    settings: UploadSettings,
    api: HfApi,
    card_path: Path | None = None,
    card_assets: Sequence[Path] = (),
    create_repo: bool = False,
    private: bool = True,
    gated_manual: bool = False,
    dry_run: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, int | bool | float]:
    if settings.row_group_size < 1:
        raise ValueError("row group size must be at least 1")
    if settings.max_workers < 1:
        raise ValueError("max workers must be at least 1")
    if settings.sleep_seconds < 0 or not math.isfinite(settings.sleep_seconds):
        raise ValueError("sleep seconds must be a finite non-negative number")
    if settings.retries < 0:
        raise ValueError("retries cannot be negative")
    if settings.max_shards is not None and settings.max_shards < 1:
        raise ValueError("max shards must be at least 1")
    if settings.max_bytes is not None and settings.max_bytes < 1:
        raise ValueError("max bytes must be at least 1")
    if gated_manual and private:
        raise ValueError("manual gating requires a public repository, not --private")
    if card_path is not None and not card_path.is_file():
        raise FileNotFoundError(f"dataset card does not exist: {card_path}")
    asset_names: set[str] = set()
    for asset in card_assets:
        if not asset.is_file():
            raise FileNotFoundError(f"dataset card asset does not exist: {asset}")
        if asset.name in asset_names:
            raise ValueError(f"dataset card assets repeat basename {asset.name!r}")
        asset_names.add(asset.name)

    manifest = manifest.resolve()
    segments_manifest = segments_manifest.resolve() if segments_manifest is not None else None
    rows = load_audio_dataset_rows(manifest, audio_root.resolve())
    plans = plan_shards(
        rows,
        target_shard_bytes=settings.target_shard_bytes,
        config_name=settings.config_name,
        split=settings.split,
    )
    identity = _state_identity(
        manifest=manifest, segments_manifest=segments_manifest, settings=settings
    )
    state = _load_state(state_path, identity)
    uploaded: dict[str, Any] = state["uploaded_shards"]
    pending = [shard for shard in plans if shard.path_in_repo not in uploaded]
    selected = _choose_run_shards(
        pending, max_shards=settings.max_shards, max_bytes=settings.max_bytes
    )
    LOGGER.info(
        "Validated %d rows (%.3f GiB); planned %d shards, %d checkpointed, "
        "%d pending, %d selected",
        len(rows),
        sum(row.audio_path.stat().st_size for row in rows) / 2**30,
        len(plans),
        len(plans) - len(pending),
        len(pending),
        len(selected),
    )
    if dry_run:
        return {
            "total_rows": len(rows),
            "total_shards": len(plans),
            "pending_shards": len(pending),
            "selected_shards": len(selected),
            "uploaded_this_run": 0,
            "complete": not pending,
            "largest_selected_source_gib": (
                max((shard.source_bytes for shard in selected), default=0) / 2**30
            ),
        }

    if create_repo:
        api.create_repo(
            repo_id=settings.repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
    if gated_manual:
        _retry(
            lambda: api.update_repo_settings(
                repo_id=settings.repo_id,
                repo_type="dataset",
                gated="manual",
                private=False,
            ),
            retries=settings.retries,
            description="manual access-gate configuration",
            sleep=sleep,
        )

    if card_path is not None or card_assets:
        card_digest = sha256_file(card_path) if card_path is not None else None
        asset_digests = {asset.name: sha256_file(asset) for asset in card_assets}
        operations: list[CommitOperationAdd] = []
        if card_path is not None and state.get("card_sha256") != card_digest:
            operations.append(
                CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card_path)
            )
        previous_asset_digests = state.get("card_asset_sha256", {})
        for asset in card_assets:
            if previous_asset_digests.get(asset.name) != asset_digests[asset.name]:
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=f"assets/{asset.name}", path_or_fileobj=asset
                    )
                )
        if operations:
            _retry(
                lambda: api.create_commit(
                    repo_id=settings.repo_id,
                    repo_type="dataset",
                    revision=settings.revision,
                    operations=operations,
                    commit_message="Update dataset card and assets",
                    num_threads=settings.max_workers,
                ),
                retries=settings.retries,
                description="dataset card commit",
                sleep=sleep,
            )
            if card_digest is not None:
                state["card_sha256"] = card_digest
            state["card_asset_sha256"] = asset_digests
            _write_json_atomic(state_path, state)

    selected_ids = {
        rows[index].example_id
        for shard in selected
        for index in range(shard.start_row, shard.stop_row)
    }
    segment_metadata = (
        load_segment_metadata(segments_manifest, include_ids=selected_ids)
        if selected_ids
        else {}
    )
    temp_dir.mkdir(parents=True, exist_ok=True)
    uploaded_this_run = 0
    for position, shard in enumerate(selected, start=1):
        local_path = temp_dir / Path(shard.path_in_repo).name
        try:
            LOGGER.info(
                "Building shard %d/%d: rows %d:%d (%.3f GiB source audio)",
                position,
                len(selected),
                shard.start_row,
                shard.stop_row,
                shard.source_bytes / 2**30,
            )
            write_parquet_shard(
                local_path,
                rows=rows[shard.start_row : shard.stop_row],
                segment_metadata=segment_metadata,
                row_group_size=settings.row_group_size,
            )
            shard_digest = sha256_file(local_path)
            _retry(
                lambda: api.create_commit(
                    repo_id=settings.repo_id,
                    repo_type="dataset",
                    revision=settings.revision,
                    operations=[
                        CommitOperationAdd(
                            path_in_repo=shard.path_in_repo,
                            path_or_fileobj=local_path,
                        )
                    ],
                    commit_message=(
                        f"Upload {settings.config_name}/{settings.split} shard "
                        f"{shard.index + 1}/{shard.total}"
                    ),
                    num_threads=settings.max_workers,
                ),
                retries=settings.retries,
                description=f"shard {shard.index + 1}/{shard.total}",
                sleep=sleep,
            )
            uploaded[shard.path_in_repo] = {
                "sha256": shard_digest,
                "rows": shard.stop_row - shard.start_row,
                "source_bytes": shard.source_bytes,
                "parquet_bytes": local_path.stat().st_size,
            }
            uploaded_this_run += 1
            _write_json_atomic(state_path, state)
            LOGGER.info(
                "Uploaded shard %d/%d; temporary file removed after checkpoint",
                position,
                len(selected),
            )
        finally:
            local_path.unlink(missing_ok=True)
        if position < len(selected) and settings.sleep_seconds:
            sleep(settings.sleep_seconds)

    complete = all(shard.path_in_repo in uploaded for shard in plans)
    return {
        "total_rows": len(rows),
        "total_shards": len(plans),
        "pending_shards": len(pending),
        "selected_shards": len(selected),
        "uploaded_this_run": uploaded_this_run,
        "complete": complete,
        "largest_selected_source_gib": (
            max((shard.source_bytes for shard in selected), default=0) / 2**30
        ),
    }


def _safe_name(value: str, *, label: str) -> str:
    if not value or "/" in value or value in {".", ".."}:
        raise ValueError(f"{label} must be one safe directory/name component")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Package a path/sentence TSV into standard Hugging Face Parquet audio "
            "shards. Only one bounded shard is staged at a time, uploaded, checkpointed, "
            "and deleted, so large datasets need little free disk and can resume safely."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Input TSV with path and sentence columns.")
    parser.add_argument("--audio-root", required=True, type=Path, help="Directory containing clips directly or in a clips/ child directory.")
    parser.add_argument("--repo-id", required=True, help="Target dataset repository, for example username/PersianAudiobook.")
    parser.add_argument("--segments-manifest", type=Path, help="Optional segments.jsonl used to add duration and provenance columns.")
    parser.add_argument("--config-name", default="refined-subset", help="Hub dataset config/subdirectory (default: refined-subset). Use a new name for a later larger release.")
    parser.add_argument("--split", default="train", help="Dataset split name (default: train).")
    parser.add_argument("--revision", default="main", help="Target Hub branch or revision (default: main).")
    parser.add_argument("--state-file", required=True, type=Path, help="Small local checkpoint JSON. The shard plan is immutable for this state file.")
    parser.add_argument("--temp-dir", required=True, type=Path, help="Temporary shard directory; successful shards are deleted immediately.")
    parser.add_argument("--card", type=Path, help="Optional dataset-card README.md to upload.")
    parser.add_argument("--card-asset", action="append", default=[], type=Path, help="Optional card asset uploaded under assets/; repeat for multiple files.")
    parser.add_argument("--target-shard-mib", type=float, default=512.0, help="Approximate source-audio MiB per Parquet shard (default: 512).")
    parser.add_argument("--row-group-size", type=int, default=100, help="Parquet rows per row group (default: 100, recommended for audio).")
    parser.add_argument("--max-workers", type=int, default=1, help="Hub upload threads for each shard (default: 1).")
    parser.add_argument("--sleep-seconds", type=float, default=10.0, help="Delay between successful shard uploads (default: 10).")
    parser.add_argument("--max-shards", type=int, help="Upload at most this many pending shards in this invocation.")
    parser.add_argument("--max-gib", type=float, help="Upload at most approximately this many GiB of source audio in this invocation.")
    parser.add_argument("--retries", type=int, default=5, help="Retries per failed Hub commit with exponential backoff (default: 5).")
    parser.add_argument("--create-repo", action="store_true", help="Create the dataset repository if it does not exist.")
    parser.add_argument("--gated-manual", action="store_true", help="Make the public dataset manually gated before committing any card, asset, or data file.")
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--public", action="store_true", help="When creating, make the repository public.")
    visibility.add_argument("--private", action="store_true", help="When creating, make the repository private (default).")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable containing a write token (default: HF_TOKEN); cached login is the fallback.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the immutable shard plan without writing shards, state, or Hub data.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config_name = _safe_name(args.config_name, label="config-name")
        split = _safe_name(args.split, label="split")
        if not math.isfinite(args.target_shard_mib) or args.target_shard_mib <= 0:
            raise ValueError("target-shard-mib must be a positive finite number")
        if args.max_gib is not None and (
            not math.isfinite(args.max_gib) or args.max_gib <= 0
        ):
            raise ValueError("max-gib must be a positive finite number")
        settings = UploadSettings(
            repo_id=args.repo_id,
            revision=args.revision,
            config_name=config_name,
            split=split,
            target_shard_bytes=int(args.target_shard_mib * 2**20),
            row_group_size=args.row_group_size,
            max_workers=args.max_workers,
            sleep_seconds=args.sleep_seconds,
            max_shards=args.max_shards,
            max_bytes=None if args.max_gib is None else int(args.max_gib * 2**30),
            retries=args.retries,
        )
        token = os.environ.get(args.token_env) or None
        summary = upload_dataset(
            manifest=args.manifest,
            audio_root=args.audio_root,
            segments_manifest=args.segments_manifest,
            state_path=args.state_file,
            temp_dir=args.temp_dir,
            settings=settings,
            api=HfApi(token=token),
            card_path=args.card,
            card_assets=args.card_asset,
            create_repo=args.create_repo,
            private=not args.public,
            gated_manual=args.gated_manual,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
