from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, Features

from ml.speech_data.scripts.hf_audio_dataset import load_audio_dataset_rows
from ml.speech_data.scripts.upload_hf_audio_dataset import (
    UploadSettings,
    plan_shards,
    upload_dataset,
    write_parquet_shard,
)


class FakeHfApi:
    def __init__(self) -> None:
        self.commits: list[list[str]] = []
        self.created: list[dict[str, Any]] = []
        self.settings: list[dict[str, Any]] = []

    def create_repo(self, **kwargs: Any) -> None:
        self.created.append(kwargs)

    def create_commit(self, **kwargs: Any) -> object:
        self.commits.append([operation.path_in_repo for operation in kwargs["operations"]])
        return object()

    def update_repo_settings(self, **kwargs: Any) -> None:
        self.settings.append(kwargs)


def _write_audio(path: Path, *, frames: int = 800) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(frames, dtype=np.float32), 8_000)


def _write_manifest(path: Path, names: list[str]) -> None:
    path.write_text(
        "path\tsentence\n"
        + "".join(f"{name}\ttranscript for {name}\n" for name in names),
        encoding="utf-8",
    )


def _settings(*, max_shards: int | None = None, target_bytes: int = 10_000) -> UploadSettings:
    return UploadSettings(
        repo_id="owner/dataset",
        revision="main",
        config_name="refined-subset",
        split="train",
        target_shard_bytes=target_bytes,
        row_group_size=1,
        max_workers=1,
        sleep_seconds=0,
        max_shards=max_shards,
        max_bytes=None,
        retries=0,
    )


def test_plans_deterministic_bounded_shards(tmp_path: Path) -> None:
    audio_root = tmp_path / "dataset"
    names = ["one.flac", "two.flac", "three.flac"]
    for name in names:
        _write_audio(audio_root / "clips" / name)
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(manifest, names)
    rows = load_audio_dataset_rows(manifest, audio_root)
    one_file_size = rows[0].audio_path.stat().st_size

    shards = plan_shards(
        rows,
        target_shard_bytes=one_file_size + 1,
        config_name="refined-subset",
        split="train",
    )

    assert [shard.stop_row - shard.start_row for shard in shards] == [1, 1, 1]
    assert shards[0].path_in_repo == "refined-subset/train-00000-of-00003.parquet"


def test_parquet_shard_has_standard_audio_feature_and_metadata(tmp_path: Path) -> None:
    audio_root = tmp_path / "dataset"
    _write_audio(audio_root / "clips" / "source_000001.flac")
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(manifest, ["source_000001.flac"])
    rows = load_audio_dataset_rows(manifest, audio_root)
    output = tmp_path / "shard.parquet"

    write_parquet_shard(
        output,
        rows=rows,
        segment_metadata={
            "source_000001": {
                "source_id": "source",
                "duration_sec": 0.1,
                "source_path": "/private/server/path.mp3",
            }
        },
        row_group_size=1,
    )

    table = pq.read_table(output)
    features = Features.from_arrow_schema(table.schema)
    assert isinstance(features["audio"], Audio)
    record = table.to_pylist()[0]
    assert record["audio"]["bytes"].startswith(b"fLaC")
    assert record["sentence"] == "transcript for source_000001.flac"
    assert record["source_id"] == "source"
    assert record["duration_sec"] == 0.1
    assert "source_path" not in record


def test_dry_run_does_not_write_shards_state_or_call_hub(tmp_path: Path) -> None:
    audio_root = tmp_path / "dataset"
    _write_audio(audio_root / "clips" / "one.flac")
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(manifest, ["one.flac"])
    api = FakeHfApi()

    result = upload_dataset(
        manifest=manifest,
        audio_root=audio_root,
        segments_manifest=None,
        state_path=tmp_path / "state.json",
        temp_dir=tmp_path / "temp",
        settings=_settings(),
        api=api,  # type: ignore[arg-type]
        dry_run=True,
    )

    assert result["total_rows"] == 1
    assert result["total_shards"] == 1
    assert result["selected_shards"] == 1
    assert not (tmp_path / "state.json").exists()
    assert not (tmp_path / "temp").exists()
    assert api.commits == []


def test_upload_checkpoints_each_shard_and_removes_temporary_files(tmp_path: Path) -> None:
    audio_root = tmp_path / "dataset"
    names = ["one.flac", "two.flac", "three.flac"]
    for name in names:
        _write_audio(audio_root / "clips" / name)
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(manifest, names)
    rows = load_audio_dataset_rows(manifest, audio_root)
    state = tmp_path / "state.json"
    temp_dir = tmp_path / "temp"
    api = FakeHfApi()
    settings = _settings(max_shards=1, target_bytes=rows[0].audio_path.stat().st_size + 1)

    first = upload_dataset(
        manifest=manifest,
        audio_root=audio_root,
        segments_manifest=None,
        state_path=state,
        temp_dir=temp_dir,
        settings=settings,
        api=api,  # type: ignore[arg-type]
        sleep=lambda _: None,
    )
    assert first["uploaded_this_run"] == 1
    assert first["complete"] is False
    assert list(temp_dir.iterdir()) == []

    second = upload_dataset(
        manifest=manifest,
        audio_root=audio_root,
        segments_manifest=None,
        state_path=state,
        temp_dir=temp_dir,
        settings=_settings(target_bytes=rows[0].audio_path.stat().st_size + 1),
        api=api,  # type: ignore[arg-type]
        sleep=lambda _: None,
    )
    assert second["uploaded_this_run"] == 2
    assert second["complete"] is True
    assert [path for commit in api.commits for path in commit] == [
        "refined-subset/train-00000-of-00003.parquet",
        "refined-subset/train-00001-of-00003.parquet",
        "refined-subset/train-00002-of-00003.parquet",
    ]
    checkpoint = json.loads(state.read_text(encoding="utf-8"))
    assert len(checkpoint["uploaded_shards"]) == 3


def test_changed_manifest_requires_a_new_config_and_state_file(tmp_path: Path) -> None:
    audio_root = tmp_path / "dataset"
    for name in ["one.flac", "two.flac"]:
        _write_audio(audio_root / "clips" / name)
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(manifest, ["one.flac"])
    state = tmp_path / "state.json"
    api = FakeHfApi()

    upload_dataset(
        manifest=manifest,
        audio_root=audio_root,
        segments_manifest=None,
        state_path=state,
        temp_dir=tmp_path / "temp",
        settings=_settings(),
        api=api,  # type: ignore[arg-type]
        sleep=lambda _: None,
    )
    _write_manifest(manifest, ["one.flac", "two.flac"])

    try:
        upload_dataset(
            manifest=manifest,
            audio_root=audio_root,
            segments_manifest=None,
            state_path=state,
            temp_dir=tmp_path / "temp",
            settings=_settings(),
            api=api,  # type: ignore[arg-type]
            dry_run=True,
        )
    except ValueError as exc:
        assert "new config name and state file" in str(exc)
    else:
        raise AssertionError("changed manifest unexpectedly reused an immutable shard plan")


def test_uploads_card_assets_once_and_places_them_under_assets(tmp_path: Path) -> None:
    audio_root = tmp_path / "dataset"
    _write_audio(audio_root / "clips" / "one.flac")
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(manifest, ["one.flac"])
    card = tmp_path / "README.md"
    card.write_text("# Dataset\n", encoding="utf-8")
    image = tmp_path / "pipeline.png"
    image.write_bytes(b"png bytes")
    state = tmp_path / "state.json"
    api = FakeHfApi()

    upload_dataset(
        manifest=manifest,
        audio_root=audio_root,
        segments_manifest=None,
        state_path=state,
        temp_dir=tmp_path / "temp",
        settings=_settings(),
        api=api,  # type: ignore[arg-type]
        card_path=card,
        card_assets=[image],
        sleep=lambda _: None,
    )
    upload_dataset(
        manifest=manifest,
        audio_root=audio_root,
        segments_manifest=None,
        state_path=state,
        temp_dir=tmp_path / "temp",
        settings=_settings(),
        api=api,  # type: ignore[arg-type]
        card_path=card,
        card_assets=[image],
        sleep=lambda _: None,
    )

    committed_paths = [path for commit in api.commits for path in commit]
    assert committed_paths.count("README.md") == 1
    assert committed_paths.count("assets/pipeline.png") == 1


def test_manual_gate_is_enabled_before_any_file_commit(tmp_path: Path) -> None:
    audio_root = tmp_path / "dataset"
    _write_audio(audio_root / "clips" / "one.flac")
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(manifest, ["one.flac"])

    class OrderedApi(FakeHfApi):
        def __init__(self) -> None:
            super().__init__()
            self.events: list[str] = []

        def update_repo_settings(self, **kwargs: Any) -> None:
            self.events.append("gate")
            super().update_repo_settings(**kwargs)

        def create_commit(self, **kwargs: Any) -> object:
            self.events.append("commit")
            return super().create_commit(**kwargs)

    api = OrderedApi()
    upload_dataset(
        manifest=manifest,
        audio_root=audio_root,
        segments_manifest=None,
        state_path=tmp_path / "state.json",
        temp_dir=tmp_path / "temp",
        settings=_settings(),
        api=api,  # type: ignore[arg-type]
        create_repo=True,
        private=False,
        gated_manual=True,
        sleep=lambda _: None,
    )

    assert api.events == ["gate", "commit"]
    assert api.settings == [
        {
            "repo_id": "owner/dataset",
            "repo_type": "dataset",
            "gated": "manual",
            "private": False,
        }
    ]


def test_manual_gate_rejects_private_visibility(tmp_path: Path) -> None:
    audio_root = tmp_path / "dataset"
    _write_audio(audio_root / "clips" / "one.flac")
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(manifest, ["one.flac"])

    try:
        upload_dataset(
            manifest=manifest,
            audio_root=audio_root,
            segments_manifest=None,
            state_path=tmp_path / "state.json",
            temp_dir=tmp_path / "temp",
            settings=_settings(),
            api=FakeHfApi(),  # type: ignore[arg-type]
            private=True,
            gated_manual=True,
            dry_run=True,
        )
    except ValueError as exc:
        assert "manual gating requires a public repository" in str(exc)
    else:
        raise AssertionError("private manually gated upload unexpectedly passed validation")
