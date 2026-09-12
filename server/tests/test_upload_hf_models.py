from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from ml.asr.upload_hf_models import ROOT, build_plans, collect_files, main, upload_plan


@pytest.fixture
def export(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "run/final"
    checkpoint.mkdir(parents=True)
    values = {
        "config.json": {"model_type": "whisper", "d_model": 768},
        "generation_config.json": {"language": "fa"},
        "preprocessor_config.json": {"sampling_rate": 16000},
        "tokenizer_config.json": {"tokenizer_class": "WhisperTokenizer"},
        "vocab.json": {"hello": 0},
        "merges.txt": "#version: 0.2",
        "model.safetensors": "synthetic weights",
        "optimizer.pt": "DO NOT UPLOAD",
        "training_args.bin": "DO NOT UPLOAD",
        "private.wav": "DO NOT UPLOAD",
        ".env": "DO NOT UPLOAD",
    }
    for name, value in values.items():
        (checkpoint / name).write_text(json.dumps(value) if isinstance(value, dict) else value)
    (tmp_path / "training.yaml").write_text(yaml.safe_dump({
        "model": {"name": "openai/whisper-small"},
        "data": {"datasets": ["cv-corpus-25.0", "fleurs-normalized"]},
        "secret": "DO NOT UPLOAD",
    }))
    config = tmp_path / "publication.yaml"
    config.write_text(yaml.safe_dump({"models": [{
        "key": "small", "repo_name": "whisper-small-fa", "title": "Persian Whisper Small",
        "base_model": "openai/whisper-small", "checkpoint": "run/final",
        "training_config": "training.yaml", "provenance_note": "Training-data composition is unverified.",
    }]}))
    return checkpoint, config


def plan_for(export: tuple[Path, Path], access: str = "private") -> Any:
    _, config = export
    return build_plans(config, config.parent, "test-owner", access)[0]


class FakeHub:
    def __init__(self, *, private: bool = True, gated: str | bool = False,
                 files: tuple[str, ...] = (), fail_commit: bool = False,
                 ignore_gating: bool = False) -> None:
        self.private, self.gated = private, gated
        self.files = files
        self.fail_commit, self.ignore_gating = fail_commit, ignore_gating
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.operations: list[Any] = []

    def create_repo(self, **kwargs: Any) -> None:
        self.events.append(("create", kwargs))

    def model_info(self, repo_id: str) -> Any:
        return SimpleNamespace(private=self.private, gated=self.gated, sha="old-sha",
                               siblings=[SimpleNamespace(rfilename=name) for name in self.files])

    def update_repo_settings(self, **kwargs: Any) -> None:
        self.events.append(("settings", kwargs))
        if "private" in kwargs:
            self.private = kwargs["private"]
        if "gated" in kwargs and not self.ignore_gating:
            self.gated = kwargs["gated"]

    def create_commit(self, **kwargs: Any) -> Any:
        self.events.append(("commit", kwargs))
        if self.fail_commit:
            raise RuntimeError("simulated transfer failure")
        self.operations = kwargs["operations"]
        return SimpleNamespace(oid="new-sha")


def test_cli_help() -> None:
    result = subprocess.run([sys.executable, "-m", "ml.asr.upload_hf_models", "--help"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    for text in ("--namespace", "--upload", "--access", "--only", "--project-root", "default: private", "--preview-dir"):
        assert text in result.stdout


def test_preview_is_offline_and_contains_no_weights(export: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Preview must not instantiate a Hub API client")

    monkeypatch.setattr(huggingface_hub, "HfApi", forbidden)
    _, config = export
    preview = config.parent / "preview"
    assert main(["--namespace", "test-owner", "--config", str(config),
                 "--project-root", str(config.parent), "--preview-dir", str(preview)]) == 0
    assert sorted(p.name for p in (preview / "small").iterdir()) == ["README.md", "publication.json"]
    assert "DO NOT UPLOAD" not in (preview / "small/publication.json").read_text()


def test_card_has_real_provenance_and_valid_examples(export: tuple[Path, Path]) -> None:
    plan = plan_for(export, "gated")
    assert "Training-data composition is unverified" in plan.card
    assert "cv-corpus-25.0" in plan.card
    assert "No WER/CER scores have been verified" in plan.card
    assert "request access" in plan.card
    assert "test-owner/whisper-small-fa" in plan.card
    frontmatter = yaml.safe_load(plan.card.split("---", 2)[1])
    assert frontmatter["base_model"] == "openai/whisper-small"
    for snippet in re.findall(r"```python\n(.*?)```", plan.card, re.S):
        ast.parse(snippet)
    assert "${" not in plan.card
    metadata = json.loads(plan.metadata)
    assert len(metadata["training_config_sha256"]) == 64
    assert "DO NOT UPLOAD" not in plan.metadata


@pytest.mark.parametrize("key", ["medium", "small"])
def test_default_publication_uses_neutral_names_without_local_paths(
    export: tuple[Path, Path], key: str,
) -> None:
    checkpoint, fixture_config = export
    project_root = fixture_config.parent
    config = ROOT / "configs/huggingface_models.yaml"
    entries = yaml.safe_load(config.read_text())["models"]
    entry = next(item for item in entries if item["key"] == key)
    target = project_root / entry["checkpoint"]
    shutil.copytree(checkpoint, target)
    (target / "config.json").write_text(json.dumps({
        "model_type": "whisper", "d_model": 1024 if key == "medium" else 768,
    }))
    training = project_root / entry["training_config"]
    training.parent.mkdir(parents=True, exist_ok=True)
    training.write_text(yaml.safe_dump({
        "model": {"name": entry["base_model"]},
        "data": {"datasets": ["cv-corpus-25.0", "fleurs-normalized"]},
    }))
    plan = build_plans(config, project_root, "owner", "gated", [key])[0]
    assert plan.repo_id == f"owner/whisper-{key}-fa"
    assert "iranseda" not in (plan.card + plan.metadata).lower()
    assert entry["checkpoint"] not in plan.card + plan.metadata
    metadata = json.loads(plan.metadata)
    assert metadata["model_key"] == key
    assert metadata["export_name"] == "final"
    assert plan.files["config.json"] == target / "config.json"


def test_allowlist_and_safetensors_preference(export: tuple[Path, Path]) -> None:
    checkpoint, _ = export
    (checkpoint / "pytorch_model.bin").write_bytes(b"older format")
    files = collect_files(checkpoint, "openai/whisper-small")
    assert "model.safetensors" in files
    assert not {"pytorch_model.bin", "optimizer.pt", "training_args.bin", "private.wav", ".env"} & files.keys()


@pytest.mark.parametrize("missing", ["config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json", "merges.txt", "model.safetensors"])
def test_missing_required_assets_fail(export: tuple[Path, Path], missing: str) -> None:
    checkpoint, _ = export
    (checkpoint / missing).unlink()
    with pytest.raises(ValueError):
        plan_for(export)


def test_architecture_mismatch_and_processor_rate(export: tuple[Path, Path]) -> None:
    checkpoint, _ = export
    with pytest.raises(ValueError, match="architecture"):
        collect_files(checkpoint, "openai/whisper-medium")
    (checkpoint / "preprocessor_config.json").write_text('{"sampling_rate": 8000}')
    with pytest.raises(ValueError, match="16000"):
        plan_for(export)


@pytest.mark.parametrize("serialization", ["safetensors", "bin"])
def test_sharded_weights_require_every_shard(export: tuple[Path, Path], serialization: str) -> None:
    checkpoint, _ = export
    (checkpoint / "model.safetensors").unlink()
    prefix = "model" if serialization == "safetensors" else "pytorch_model"
    shard = f"{prefix}-00001-of-00001.{serialization}"
    index = checkpoint / f"{prefix}.{serialization}.index.json"
    index.write_text(json.dumps({"weight_map": {"encoder.weight": shard}}))
    with pytest.raises(ValueError, match="Missing or empty"):
        plan_for(export)
    (checkpoint / shard).write_bytes(b"synthetic")
    assert shard in plan_for(export).files
    index.write_text(json.dumps({"weight_map": {"encoder.weight": "../model.safetensors"}}))
    with pytest.raises(ValueError, match="Invalid weight shard"):
        plan_for(export)


def test_symlink_outside_checkpoint_is_rejected(export: tuple[Path, Path]) -> None:
    checkpoint, config = export
    (checkpoint / "model.safetensors").unlink()
    (checkpoint / "model.safetensors").symlink_to(config)
    with pytest.raises(ValueError, match="directly inside"):
        plan_for(export)


def test_private_upload_keeps_visibility_and_commits_only_inference_files(export: tuple[Path, Path]) -> None:
    api = FakeHub()
    plan = plan_for(export)
    assert upload_plan(plan, "private", api) == "new-sha"
    assert api.private
    assert [event for event, _ in api.events] == ["create", "commit"]
    assert api.events[0][1]["private"] is True
    assert api.events[1][1]["parent_commit"] == "old-sha"
    assert {op.path_in_repo for op in api.operations} == set(plan.files) | {"README.md", "publication.json"}


def test_gated_upload_enables_gate_before_commit_and_public_after(export: tuple[Path, Path]) -> None:
    api = FakeHub()
    upload_plan(plan_for(export, "gated"), "gated", api)
    assert [event for event, _ in api.events] == ["create", "settings", "commit", "settings"]
    assert api.events[1][1]["gated"] == "manual"
    assert api.events[-1][1]["private"] is False
    assert not api.private and api.gated == "manual"


@pytest.mark.parametrize("access,gated", [("private", False), ("private", "manual"), ("gated", False), ("gated", "auto")])
def test_existing_incompatible_public_repo_is_refused(export: tuple[Path, Path], access: str, gated: str | bool) -> None:
    api = FakeHub(private=False, gated=gated)
    with pytest.raises(ValueError, match="Refusing existing"):
        upload_plan(plan_for(export, access), access, api)
    assert [event for event, _ in api.events] == ["create"]


def test_existing_gated_repo_can_be_updated(export: tuple[Path, Path]) -> None:
    api = FakeHub(private=False, gated="manual", files=("model.safetensors", "notes.md"))
    upload_plan(plan_for(export, "gated"), "gated", api)
    assert not api.private and api.gated == "manual"
    assert "notes.md" not in {op.path_in_repo for op in api.operations}


@pytest.mark.parametrize("stale_file", ["pytorch_model.bin", "tokenizer.json"])
def test_stale_remote_inference_files_are_refused(export: tuple[Path, Path], stale_file: str) -> None:
    api = FakeHub(files=(stale_file,))
    with pytest.raises(ValueError, match="incompatible old inference files"):
        upload_plan(plan_for(export), "private", api)
    assert not api.operations


@pytest.mark.parametrize("failure", ["commit", "gating"])
def test_failed_gated_upload_never_publishes_new_repo(export: tuple[Path, Path], failure: str) -> None:
    api = FakeHub(fail_commit=failure == "commit", ignore_gating=failure == "gating")
    with pytest.raises(RuntimeError):
        upload_plan(plan_for(export, "gated"), "gated", api)
    assert api.private
    assert not any(kwargs.get("private") is False for _, kwargs in api.events)


def test_config_rejects_unknown_keys_and_duplicate_destinations(export: tuple[Path, Path]) -> None:
    _, config = export
    with pytest.raises(ValueError, match="Unknown --only"):
        build_plans(config, config.parent, "owner", "private", ["medium"])
    values = yaml.safe_load(config.read_text())
    values["models"].append({**values["models"][0], "key": "second"})
    config.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="Duplicate destination"):
        plan_for(export)


def test_all_exports_validate_before_upload(export: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    _, config = export
    values = yaml.safe_load(config.read_text())
    values["models"].append({**values["models"][0], "key": "second", "repo_name": "second", "checkpoint": "missing"})
    config.write_text(yaml.safe_dump(values))

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Invalid local export must stop before remote requests")

    monkeypatch.setattr(huggingface_hub, "HfApi", forbidden)
    with pytest.raises(SystemExit) as result:
        main(["--namespace", "owner", "--config", str(config), "--project-root", str(config.parent), "--upload"])
    assert result.value.code == 1


def test_only_skips_unselected_missing_checkpoint(export: tuple[Path, Path]) -> None:
    _, config = export
    values = yaml.safe_load(config.read_text())
    values["models"].append({**values["models"][0], "key": "second", "repo_name": "second", "checkpoint": "missing"})
    config.write_text(yaml.safe_dump(values))
    plans = build_plans(config, config.parent, "owner", "private", ["small"])
    assert [plan.key for plan in plans] == ["small"]
