"""Validate and publish saved Whisper exports with model cards on Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "docs/huggingface/whisper-model-card.template.md"
INFERENCE_FILES = {
    "config.json", "generation_config.json", "preprocessor_config.json",
    "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json",
    "vocab.json", "merges.txt", "added_tokens.json", "normalizer.json",
}
WEIGHT_NAME = re.compile(r"(?:model(?:-\d+-of-\d+)?\.safetensors|pytorch_model(?:-\d+-of-\d+)?\.bin|model\.safetensors\.index\.json|pytorch_model\.bin\.index\.json)")


@dataclass(frozen=True)
class UploadPlan:
    key: str
    repo_id: str
    files: dict[str, Path]
    card: str
    metadata: str


def read_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def required_file(folder: Path, name: str) -> Path:
    path = folder / name
    if path.parent != folder or path.resolve().parent != folder.resolve():
        raise ValueError(f"File must be directly inside checkpoint: {name}")
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty inference file: {path}")
    return path


def collect_files(checkpoint: Path, base_model: str) -> dict[str, Path]:
    """Allowlist inference assets; never include trainer state, logs or audio."""
    required = {
        "config.json", "generation_config.json", "preprocessor_config.json",
        "tokenizer_config.json",
    }
    required.update(
        {"tokenizer.json"} if (checkpoint / "tokenizer.json").is_file()
        else {"vocab.json", "merges.txt"}
    )
    for name in required:
        required_file(checkpoint, name)
    config = read_mapping(checkpoint / "config.json")
    expected_width = {"openai/whisper-small": 768, "openai/whisper-medium": 1024}
    if base_model not in expected_width:
        raise ValueError(f"Unsupported base_model: {base_model}")
    if config.get("model_type") != "whisper" or config.get("d_model") != expected_width[base_model]:
        raise ValueError(f"Checkpoint architecture does not match {base_model}: {checkpoint}")
    processor = read_mapping(checkpoint / "preprocessor_config.json")
    if processor.get("sampling_rate") != 16000:
        raise ValueError(f"Expected a 16000 Hz Whisper processor: {checkpoint}")
    for name in ("generation_config.json", "tokenizer_config.json"):
        read_mapping(checkpoint / name)
    files = {
        name: required_file(checkpoint, name)
        for name in sorted(INFERENCE_FILES) if (checkpoint / name).exists()
    }
    # Prefer safetensors when both serialization formats exist.
    for single, index in (
        ("model.safetensors", "model.safetensors.index.json"),
        ("pytorch_model.bin", "pytorch_model.bin.index.json"),
    ):
        if (checkpoint / single).exists():
            files[single] = required_file(checkpoint, single)
            break
        if (checkpoint / index).exists():
            index_path = required_file(checkpoint, index)
            weight_map = read_mapping(index_path).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Empty or invalid weight_map: {index_path}")
            for shard in weight_map.values():
                if not isinstance(shard, str) or not WEIGHT_NAME.fullmatch(shard) or not shard.endswith(Path(single).suffix):
                    raise ValueError(f"Invalid weight shard in {index_path}: {shard!r}")
                files[shard] = required_file(checkpoint, shard)
            files[index] = index_path
            break
    else:
        raise ValueError(f"No full-model weights found in {checkpoint}; use a saved best/final export")
    return dict(sorted(files.items()))


def build_plans(config_path: Path, project_root: Path, namespace: str,
                access: str, only: Sequence[str] | None = None) -> list[UploadPlan]:
    from huggingface_hub.utils import validate_repo_id

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", namespace):
        raise ValueError("--namespace must be one Hugging Face username or organization name")
    entries = read_mapping(config_path).get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Config must contain a nonempty models list")
    keys: set[str] = set()
    repos: set[str] = set()
    plans = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each model entry must be a mapping")
        for field in ("key", "repo_name", "title", "base_model", "checkpoint", "training_config", "provenance_note"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise ValueError(f"Model entry requires a nonempty string: {field}")
        key = entry["key"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key) or key in keys:
            raise ValueError(f"Invalid or duplicate model key: {key}")
        keys.add(key)
        repo_id = f"{namespace}/{entry['repo_name']}"
        validate_repo_id(repo_id)
        if repo_id in repos:
            raise ValueError(f"Duplicate destination: {repo_id}")
        repos.add(repo_id)
        if only and key not in only:
            continue
        checkpoint = (project_root / entry["checkpoint"]).resolve()
        files = collect_files(checkpoint, entry["base_model"])
        training_path = (project_root / entry["training_config"]).resolve()
        training = read_mapping(training_path)
        if training.get("model", {}).get("name") != entry["base_model"]:
            raise ValueError(f"Training config base model mismatch: {training_path}")
        datasets = training.get("data", {}).get("datasets", [])
        if not isinstance(datasets, list) or not all(isinstance(item, str) for item in datasets):
            raise ValueError(f"Invalid training dataset list: {training_path}")
        # Include selected provenance only, not raw manifests or arbitrary config values.
        metadata = {
            "repo_id": repo_id, "base_model": entry["base_model"],
            "model_key": key,
            "export_name": checkpoint.name,
            "training_config_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
            "configured_datasets": datasets,
            "inference_files": {name: path.stat().st_size for name, path in files.items()},
            "provenance_note": entry["provenance_note"],
        }
        access_text = (
            "This is a private repository. The owner must grant access through the "
            "owning organization; a link alone does not grant access."
            if access == "private" else
            "The model card is public and model downloads require manual approval. "
            "Sign in on this model page, request access, and wait for the owner to "
            "approve your account before downloading."
        )
        details = f"Selected export: `{checkpoint.name}`.\n\nDatasets listed in the saved configuration:\n\n"
        details += "\n".join(f"- `{item}`" for item in datasets) or "No datasets recorded."
        card = Template(TEMPLATE.read_text(encoding="utf-8")).substitute(
            title=entry["title"], repo_id=repo_id, base_model=entry["base_model"],
            base_model_url=f"https://huggingface.co/{entry['base_model']}",
            provenance_note=entry["provenance_note"], training_details=details,
            access_instructions=access_text,
        )
        plans.append(UploadPlan(key, repo_id, files, card, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"))
    if only and set(only) - keys:
        raise ValueError(f"Unknown --only keys: {', '.join(sorted(set(only) - keys))}")
    return plans


def upload_plan(plan: UploadPlan, access: str, api: Any) -> str:
    from huggingface_hub import CommitOperationAdd

    # New repositories stay private until their complete upload is committed.
    api.create_repo(repo_id=plan.repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.model_info(plan.repo_id)
    if not info.private and (access == "private" or info.gated != "manual"):
        raise ValueError(f"Refusing existing public or differently gated destination: {plan.repo_id}. Use a new repository name.")
    remote_files = {sibling.rfilename for sibling in info.siblings or []}
    stale_files = {
        name for name in remote_files if WEIGHT_NAME.fullmatch(name) or name in INFERENCE_FILES
    } - plan.files.keys()
    if stale_files:
        raise ValueError(f"Destination has incompatible old inference files: {sorted(stale_files)}. Use a new repository name or remove obsolete inference files manually.")
    if access == "gated":
        api.update_repo_settings(repo_id=plan.repo_id, repo_type="model", gated="manual")
        if api.model_info(plan.repo_id).gated != "manual":
            raise RuntimeError("Manual gating was not confirmed; upload stopped")
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
        for name, path in plan.files.items()
    ]
    operations.extend([
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=plan.card.encode()),
        CommitOperationAdd(path_in_repo="publication.json", path_or_fileobj=plan.metadata.encode()),
    ])
    result = api.create_commit(
        repo_id=plan.repo_id, repo_type="model", operations=operations,
        commit_message=f"Publish {plan.key} Whisper export and usage guide",
        parent_commit=info.sha,
    )
    if access == "gated":
        api.update_repo_settings(repo_id=plan.repo_id, repo_type="model", private=False)
    final = api.model_info(plan.repo_id)
    if access == "private" and not final.private:
        raise RuntimeError(f"Private visibility verification failed for {plan.repo_id}")
    if access == "gated" and (final.private or final.gated != "manual"):
        raise RuntimeError(f"Public/manual-gating verification failed for {plan.repo_id}")
    return str(result.oid)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description="Preview or upload saved Whisper models and usage cards to Hugging Face. No GPU required.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cli.add_argument("--namespace", required=True, help="Hugging Face username or organization (string; required)")
    cli.add_argument("--config", type=Path, default=ROOT / "configs/huggingface_models.yaml", help="Publication YAML file (path)")
    cli.add_argument("--project-root", type=Path, default=ROOT, help="Root for checkpoint and training-config paths (path)")
    cli.add_argument("--access", choices=("private", "gated"), default="private", help="gated makes the card public and requires manual approval for downloads")
    cli.add_argument("--only", nargs="+", default=[], metavar="KEY", help="Upload only these model keys, e.g. medium small; an empty list selects all")
    cli.add_argument("--preview-dir", type=Path, default=ROOT / "artifacts/huggingface/model-upload-preview", help="Write generated cards and metadata here, without copying weights (path)")
    cli.add_argument("--upload", action="store_true", help="Perform the upload; without this flag only validate and write local previews")
    return cli


def main(argv: Sequence[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    try:
        # Validate every selected local export before making any Hub requests.
        plans = build_plans(args.config, args.project_root, args.namespace, args.access, args.only)
        for plan in plans:
            folder = args.preview_dir / plan.key
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "README.md").write_text(plan.card, encoding="utf-8")
            (folder / "publication.json").write_text(plan.metadata, encoding="utf-8")
            size = sum(path.stat().st_size for path in plan.files.values()) / 1024**3
            print(f"{plan.key}: https://huggingface.co/{plan.repo_id} ({args.access}, {size:.2f} GiB)")
            print(f"  Source: {next(iter(plan.files.values())).parent}")
            print(f"  Preview: {folder / 'README.md'}")
        if not args.upload:
            print("Preview complete. No Hub requests made. Add --upload to publish.")
            return 0
        from huggingface_hub import HfApi, get_token

        token = get_token()
        if not token:
            raise ValueError("Authenticate with hf auth login or set HF_TOKEN before uploading")
        api = HfApi(token=token)
        api.whoami()
        for plan in plans:
            sha = upload_plan(plan, args.access, api)
            print(f"Uploaded https://huggingface.co/{plan.repo_id} at commit {sha}")
        return 0
    except (ValueError, OSError, yaml.YAMLError) as error:
        cli.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
