"""Conservatively refine Persian Whisper transcripts with causal context."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from ml.speech_data.long_audio_asr_pipeline.segment_audio import (
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from ml.speech_data.text_normalization import normalize_persian_asr_text


REFINEMENT_PIPELINE_VERSION = 1
PROMPT_VERSION = "persian-transcript-refinement-v1"
SCHEMA_VERSION = "persian-transcript-refinement-schema-v1"
PENDING_SNAPSHOT_NAME = "refinement_pending_snapshot.jsonl"
CHANGE_CATEGORIES = (
    "none",
    "orthography",
    "spacing",
    "asr_substitution",
)
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["cleaned_text", "uncertain", "change_categories"],
    "properties": {
        "cleaned_text": {"type": "string"},
        "uncertain": {"type": "boolean"},
        "change_categories": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "enum": list(CHANGE_CATEGORIES)},
        },
    },
}

DEFAULT_CONFIG: dict[str, Any] = {
    "server": {
        "base_url": "http://127.0.0.1:8000",
        "model": None,
        "timeout_seconds": 120,
        "retry_count": 2,
        "api_key_env": None,
    },
    "context": {"size": 3},
    "batch": {"size": 16},
    "generation": {
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "seed": 0,
        "max_tokens": 256,
    },
    "validation": {"maximum_normalized_edit_distance": 0.35},
}

ARTIFACT_NAMES = (
    "refined_transcription.tsv",
    "refinements.jsonl",
    "refinement_rejected.jsonl",
    PENDING_SNAPSHOT_NAME,
    "refinement_summary.json",
    "refinement_run.json",
    "refinement_effective_config.yaml",
)


@dataclass(frozen=True)
class RefinementAudit:
    targets_total: int
    targets_processed: int
    targets_reused: int
    targets_accepted: int
    targets_rejected: int
    operational_failures: int


class BatchClient(Protocol):
    def preflight(self) -> None: ...
    def complete(self, request: dict[str, Any]) -> dict[str, Any]: ...


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("configuration must be a YAML mapping")
    config = _deep_merge(DEFAULT_CONFIG, loaded)
    server = config.get("server")
    generation = config.get("generation")
    if not isinstance(server, dict) or not isinstance(generation, dict):
        raise ValueError("configuration must contain server and generation mappings")
    base_url = server.get("base_url")
    if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
        raise ValueError("server.base_url must be an HTTP(S) URL")
    server["base_url"] = base_url.rstrip("/")
    if not isinstance(server.get("model"), str) or not server["model"].strip():
        raise ValueError("server.model must be a non-empty served model name")
    api_env = server.get("api_key_env")
    if api_env is not None and (not isinstance(api_env, str) or not api_env.strip()):
        raise ValueError("server.api_key_env must be null or a non-empty environment variable name")
    for section, field, minimum in (
        (server, "retry_count", 0),
        (config["context"], "size", 0),
        (config["batch"], "size", 1),
        (generation, "n", 1),
        (generation, "seed", 0),
        (generation, "max_tokens", 1),
    ):
        value = section.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{field} must be an integer greater than or equal to {minimum}")
    timeout = server.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("server.timeout_seconds must be greater than zero")
    if generation.get("temperature") != 0 or generation.get("top_p") != 1 or generation.get("n") != 1:
        raise ValueError("generation must use temperature: 0, top_p: 1, and n: 1")
    threshold = config["validation"].get("maximum_normalized_edit_distance")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("validation.maximum_normalized_edit_distance must be between 0 and 1")
    config["pipeline_version"] = REFINEMENT_PIPELINE_VERSION
    config["prompt_version"] = PROMPT_VERSION
    config["schema_version"] = SCHEMA_VERSION
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return config, f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


class VLLMBatchClient:
    def __init__(self, config: dict[str, Any]) -> None:
        server = config["server"]
        self.url = f"{server['base_url']}/v1/chat/completions/batch"
        self.timeout = float(server["timeout_seconds"])
        self.retry_count = int(server["retry_count"])
        self.headers = {"Content-Type": "application/json"}
        api_env = server.get("api_key_env")
        if api_env:
            api_key = os.environ.get(str(api_env))
            if not api_key:
                raise ValueError(f"API key environment variable is not set: {api_env}")
            self.headers["Authorization"] = f"Bearer {api_key}"

    def preflight(self) -> None:
        # An existing POST-only route answers GET with 405; an older vLLM answers 404.
        request = urllib.request.Request(self.url, headers=self.headers, method="GET")
        try:
            urllib.request.urlopen(request, timeout=self.timeout).close()
        except urllib.error.HTTPError as error:
            if error.code == 405:
                return
            if error.code == 404:
                raise RuntimeError(
                    "vLLM does not expose required /v1/chat/completions/batch endpoint"
                ) from error
            if error.code in {401, 403}:
                raise RuntimeError(f"vLLM batch endpoint authorization failed (HTTP {error.code})") from error
            raise RuntimeError(f"vLLM batch endpoint preflight failed (HTTP {error.code})") from error
        except OSError as error:
            raise RuntimeError(f"vLLM batch endpoint preflight failed: {error}") from error

    def complete(self, request_body: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retry_count + 1):
            request = urllib.request.Request(self.url, data=payload, headers=self.headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise RuntimeError("vLLM batch response is not a JSON object")
                return parsed
            except (OSError, urllib.error.HTTPError, json.JSONDecodeError, RuntimeError) as error:
                last_error = error
                if attempt < self.retry_count:
                    time.sleep(min(2**attempt, 4))
        raise RuntimeError(f"vLLM batch request failed after retries: {last_error}")


def _load_inputs(input_root: Path) -> list[dict[str, Any]]:
    segment_path = input_root / "segments.jsonl"
    transcription_path = input_root / "transcriptions.jsonl"
    if not segment_path.is_file() or not transcription_path.is_file():
        raise ValueError("input root must contain segments.jsonl and transcriptions.jsonl")
    segments = read_jsonl(segment_path)
    transcripts = read_jsonl(transcription_path)
    segment_by_id: dict[str, dict[str, Any]] = {}
    for record in segments:
        segment_id = record.get("id")
        if not isinstance(segment_id, str) or segment_id in segment_by_id:
            raise ValueError("segments.jsonl must contain unique string ids")
        segment_by_id[segment_id] = record
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for transcript in transcripts:
        target_id = transcript.get("id")
        text = transcript.get("normalized_transcript")
        if not isinstance(target_id, str) or target_id in seen:
            raise ValueError("transcriptions.jsonl must contain unique string ids")
        seen.add(target_id)
        if target_id not in segment_by_id:
            raise ValueError(f"transcription has no matching segment: {target_id}")
        if not isinstance(text, str) or not normalize_persian_asr_text(text):
            raise ValueError(f"transcription has no usable normalized text: {target_id}")
        segment = segment_by_id[target_id]
        targets.append({**transcript, "segment": segment, "source_id": segment.get("source_id")})
    return targets


def load_book_titles(path: Path | None) -> tuple[dict[str, str], str | None]:
    if path is None:
        return {}, None
    if not path.is_file():
        raise FileNotFoundError(f"books manifest does not exist: {path}")
    titles: dict[str, str] = {}
    for index, record in enumerate(read_jsonl(path), start=1):
        book_id, title = record.get("id"), record.get("title")
        if not isinstance(book_id, str) or not book_id.isdigit() or not isinstance(title, str) or not title.strip():
            continue
        if book_id in titles:
            raise ValueError(f"books manifest contains duplicate usable id: {book_id}")
        normalized = normalize_persian_asr_text(title.strip())
        if normalized:
            titles[book_id] = title.strip()
    return titles, sha256_file(path)


def title_for_source(source_id: Any, titles: dict[str, str]) -> str | None:
    if not isinstance(source_id, str):
        return None
    book_id = source_id.split(":", 1)[0]
    return titles.get(book_id) if book_id.isdigit() else None


def group_targets(targets: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for target in targets:
        source = target.get("source_id")
        key = f"source:{source}" if isinstance(source, str) and source else f"orphan:{target['id']}"
        groups.setdefault(key, []).append(target)
    for items in groups.values():
        items.sort(
            key=lambda item: (
                float(item["segment"].get("start_sec", float("inf")))
                if isinstance(item["segment"].get("start_sec"), (int, float))
                else float("inf"),
                str(item["id"]),
            )
        )
    return [groups[key] for key in sorted(groups)]


def build_prompt(
    target: dict[str, Any],
    preceding: Sequence[dict[str, str]],
    following: Sequence[dict[str, str]],
    title: str | None,
) -> str:
    def lines(items: Sequence[dict[str, str]]) -> str:
        return "\n".join(f"- [{item['id']}] {item['text']}" for item in items) or "(none)"

    return (
        "You refine one Persian ASR segment conservatively. Correct only orthography, spacing, "
        "and clear ASR word substitutions. Context is evidence for disambiguation only: "
        "never copy context into the target, complete a boundary fragment, or add speech. Preserve every "
        "numeric token exactly. If unsure, set uncertain=true. Return only the required JSON object.\n\n"
        f"[AUDIOBOOK TITLE]\n{title or '(omitted)'}\n\n"
        f"[REFINED PRECEDING CONTEXT]\n{lines(preceding)}\n\n"
        f"[TARGET WHISPER TEXT]\n[{target['id']}] {target['normalized_transcript']}\n\n"
        f"[FOLLOWING WHISPER CONTEXT]\n{lines(following)}"
    )


def _numeric_tokens(text: str) -> list[str]:
    current = ""
    tokens: list[str] = []
    for character in text:
        if character.isdigit():
            current += character
        elif current:
            tokens.append(current)
            current = ""
    if current:
        tokens.append(current)
    return tokens


def normalized_edit_distance(left: str, right: str) -> float:
    if left == right:
        return 0.0
    if not left or not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right))


def validate_response(raw_text: str, target_text: str, threshold: float) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
    metrics: dict[str, Any] = {"maximum_normalized_edit_distance": threshold}
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as error:
        return None, "invalid_json", {**metrics, "detail": str(error)}
    if not isinstance(parsed, dict) or set(parsed) != {"cleaned_text", "uncertain", "change_categories"}:
        return None, "invalid_schema", metrics
    cleaned = parsed["cleaned_text"]
    uncertain = parsed["uncertain"]
    categories = parsed["change_categories"]
    if not isinstance(cleaned, str) or not isinstance(uncertain, bool) or not isinstance(categories, list):
        return None, "invalid_schema", metrics
    if not categories or any(not isinstance(item, str) or item not in CHANGE_CATEGORIES for item in categories) or len(categories) != len(set(categories)):
        return None, "invalid_schema", metrics
    if "none" in categories and (len(categories) != 1 or cleaned.strip() != target_text.strip()):
        return None, "invalid_schema", metrics
    if uncertain:
        return parsed, "model_uncertain", metrics
    normalized = normalize_persian_asr_text(cleaned)
    if not normalized:
        return parsed, "persian_normalization_rejected", metrics
    target_normalized = normalize_persian_asr_text(target_text)
    if not target_normalized:
        return parsed, "target_normalization_rejected", metrics
    metrics["normalized_cleaned_text"] = normalized
    metrics["target_numeric_tokens"] = _numeric_tokens(target_normalized)
    metrics["cleaned_numeric_tokens"] = _numeric_tokens(normalized)
    if metrics["target_numeric_tokens"] != metrics["cleaned_numeric_tokens"]:
        return parsed, "numeric_tokens_changed", metrics
    distance = normalized_edit_distance(target_normalized, normalized)
    metrics["normalized_edit_distance"] = distance
    if distance > threshold:
        return parsed, "edit_distance_exceeded", metrics
    return parsed, None, metrics


def _input_fingerprint(
    target: dict[str, Any], preceding: Sequence[dict[str, str]], following: Sequence[dict[str, str]],
    title: str | None, config_digest: str, upstream_checksums: dict[str, str | None],
) -> str:
    payload = {
        "config_digest": config_digest,
        "id": target["id"],
        "target_text": target["normalized_transcript"],
        "segment": target["segment"],
        "preceding": list(preceding),
        "following": list(following),
        "title": title,
        "upstream_checksums": upstream_checksums,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


def _request(config: dict[str, Any], prompts: Sequence[str]) -> dict[str, Any]:
    generation = config["generation"]
    return {
        "model": config["server"]["model"],
        "messages": [[{"role": "user", "content": prompt}] for prompt in prompts],
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "n": generation["n"],
        "seed": generation["seed"],
        "max_tokens": generation["max_tokens"],
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "transcript_refinement", "strict": True, "schema": RESPONSE_SCHEMA},
        },
    }


def _write_tsv_atomic(path: Path, accepted: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "sentence"], delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for record in sorted(accepted, key=lambda item: str(item["id"])):
                raw_path = Path(str(record["path"]))
                relative = raw_path.relative_to("clips").as_posix() if raw_path.parts[:1] == ("clips",) else raw_path.as_posix()
                writer.writerow({"path": relative, "sentence": record["cleaned_text"]})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _persist(output_root: Path, accepted: list[dict[str, Any]], rejected: list[dict[str, Any]], audit: RefinementAudit) -> None:
    accepted.sort(key=lambda item: str(item["id"]))
    rejected.sort(key=lambda item: str(item["id"]))
    write_jsonl_atomic(output_root / "refinements.jsonl", accepted)
    write_jsonl_atomic(output_root / "refinement_rejected.jsonl", rejected)
    _write_tsv_atomic(output_root / "refined_transcription.tsv", accepted)
    write_json_atomic(output_root / "refinement_summary.json", asdict(audit))


def process_refinement(
    input_root: Path, output_root: Path, config: dict[str, Any], config_digest: str,
    books_manifest: Path | None = None,
    client_factory: Callable[[dict[str, Any]], BatchClient] = VLLMBatchClient,
) -> RefinementAudit:
    targets = _load_inputs(input_root)
    titles, books_checksum = load_book_titles(books_manifest)
    upstream = {
        "segments": sha256_file(input_root / "segments.jsonl"),
        "transcriptions": sha256_file(input_root / "transcriptions.jsonl"),
        "books": books_checksum,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    prior_accepted = {item["id"]: item for item in read_jsonl(output_root / "refinements.jsonl")} if (output_root / "refinements.jsonl").is_file() else {}
    prior_rejected = {item["id"]: item for item in read_jsonl(output_root / "refinement_rejected.jsonl")} if (output_root / "refinement_rejected.jsonl").is_file() else {}
    groups = group_targets(targets)
    context_size = int(config["context"]["size"])
    group_state = [{"items": group, "index": 0, "accepted": []} for group in groups]
    queue = deque(range(len(group_state)))
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    pending_snapshot: list[dict[str, Any]] = []
    processed = reused = 0
    client: BatchClient | None = None

    def prepare(group_index: int) -> tuple[dict[str, Any], dict[str, Any]] | None:
        nonlocal reused
        state = group_state[group_index]
        while state["index"] < len(state["items"]):
            index = state["index"]
            target = state["items"][index]
            preceding = [{"id": item["id"], "text": item["cleaned_text"]} for item in state["accepted"][-context_size:]] if context_size else []
            following = [
                {"id": item["id"], "text": item["normalized_transcript"]}
                for item in state["items"][index + 1 : index + 1 + context_size]
            ]
            title = title_for_source(target.get("source_id"), titles)
            prompt = build_prompt(target, preceding, following, title)
            fingerprint = _input_fingerprint(target, preceding, following, title, config_digest, upstream)
            prior = prior_accepted.get(target["id"]) or prior_rejected.get(target["id"])
            if prior and not prior.get("operational", False) and prior.get("input_fingerprint") == fingerprint:
                record = dict(prior)
                (rejected if "reason" in record else accepted).append(record)
                if "reason" not in record:
                    state["accepted"].append(record)
                state["index"] += 1
                reused += 1
                continue
            metadata = {
                "target": target,
                "preceding_context": preceding,
                "following_context": following,
                "title": title,
                "rendered_prompt": prompt,
                "input_fingerprint": fingerprint,
            }
            return target, metadata
        return None

    while queue:
        batch: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        while queue and len(batch) < int(config["batch"]["size"]):
            group_index = queue.popleft()
            prepared = prepare(group_index)
            if prepared:
                target, metadata = prepared
                batch.append((group_index, target, metadata))
        if not batch:
            continue
        pending_snapshot.extend(
            {"id": target["id"], "source_id": target.get("source_id"), "input_fingerprint": metadata["input_fingerprint"]}
            for _, target, metadata in batch
        )
        write_jsonl_atomic(output_root / PENDING_SNAPSHOT_NAME, pending_snapshot)
        if client is None:
            client = client_factory(config)
            client.preflight()
        request_body = _request(config, [metadata["rendered_prompt"] for _, _, metadata in batch])
        operational_detail: str | None = None
        choices: dict[int, dict[str, Any]] = {}
        try:
            response = client.complete(request_body)
            raw_choices = response.get("choices")
            if not isinstance(raw_choices, list) or len(raw_choices) != len(batch):
                raise RuntimeError("incomplete native batch response")
            for choice in raw_choices:
                if not isinstance(choice, dict) or not isinstance(choice.get("index"), int) or choice["index"] in choices:
                    raise RuntimeError("malformed or duplicate choice index in native batch response")
                choices[choice["index"]] = choice
            if set(choices) != set(range(len(batch))):
                raise RuntimeError("native batch response choice indexes are incomplete")
        except Exception as error:
            operational_detail = f"{type(error).__name__}: {error}"

        for choice_index, (group_index, target, metadata) in enumerate(batch):
            common = {
                "id": target["id"],
                "source_id": target.get("source_id"),
                "path": target.get("path"),
                "target_whisper_text": target["normalized_transcript"],
                **metadata,
                "target": {"id": target["id"], "text": target["normalized_transcript"]},
                "model": config["server"]["model"],
                "model_parameters": config["generation"],
                "prompt_version": PROMPT_VERSION,
                "schema_version": SCHEMA_VERSION,
                "response_schema": RESPONSE_SCHEMA,
                "config_digest": config_digest,
                "upstream_checksums": upstream,
            }
            if operational_detail is not None:
                rejected.append({**common, "reason": "batch_request_failed", "detail": operational_detail, "operational": True, "raw_response": None, "parsed_response": None, "validation_metrics": {}})
            else:
                choice = choices[choice_index]
                message = choice.get("message")
                raw_text = message.get("content") if isinstance(message, dict) else None
                if choice.get("finish_reason") not in {"stop", None} or not isinstance(raw_text, str):
                    rejected.append({**common, "reason": "incomplete_batch_output", "detail": f"finish_reason={choice.get('finish_reason')}", "operational": True, "raw_response": choice, "parsed_response": None, "validation_metrics": {}})
                else:
                    parsed, reason, metrics = validate_response(raw_text, target["normalized_transcript"], float(config["validation"]["maximum_normalized_edit_distance"]))
                    record = {**common, "raw_response": choice, "raw_response_text": raw_text, "parsed_response": parsed, "validation_metrics": metrics, "operational": False}
                    if reason:
                        rejected.append({**record, "reason": reason, "detail": reason})
                    else:
                        accepted_record = {**record, "cleaned_text": parsed["cleaned_text"].strip(), "uncertain": False, "change_categories": parsed["change_categories"]}  # type: ignore[index,union-attr]
                        accepted.append(accepted_record)
                        group_state[group_index]["accepted"].append(accepted_record)
            group_state[group_index]["index"] += 1
            if group_state[group_index]["index"] < len(group_state[group_index]["items"]):
                queue.append(group_index)
            processed += 1
        audit = RefinementAudit(len(targets), processed, reused, len(accepted), len(rejected), sum(bool(item.get("operational")) for item in rejected))
        _persist(output_root, accepted, rejected, audit)

    write_jsonl_atomic(output_root / PENDING_SNAPSHOT_NAME, pending_snapshot)
    audit = RefinementAudit(len(targets), processed, reused, len(accepted), len(rejected), sum(bool(item.get("operational")) for item in rejected))
    _persist(output_root, accepted, rejected, audit)
    return audit


def run_refinement(
    input_root: Path, config: dict[str, Any], config_digest: str, *, books_manifest: Path | None = None,
    force: bool = False, client_factory: Callable[[dict[str, Any]], BatchClient] = VLLMBatchClient,
) -> RefinementAudit:
    input_root = input_root.resolve()
    run_path = input_root / "refinement_run.json"
    if run_path.is_file():
        prior = json.loads(run_path.read_text(encoding="utf-8"))
        if prior.get("config_digest") != config_digest and not force:
            raise ValueError("refinement uses a different configuration digest; pass --force to replace it")
    elif any((input_root / name).exists() for name in ARTIFACT_NAMES if name != "refinement_run.json") and not force:
        raise FileExistsError("input root contains untracked refinement artifacts; pass --force to replace them")

    if force:
        staging = Path(tempfile.mkdtemp(prefix=".refinement-", dir=input_root))
        try:
            audit = process_refinement(input_root, staging, config, config_digest, books_manifest, client_factory)
            if audit.operational_failures:
                raise RuntimeError("forced refinement had operational failures; existing artifacts were preserved")
            write_json_atomic(staging / "refinement_run.json", {"config_digest": config_digest})
            (staging / "refinement_effective_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=True), encoding="utf-8")
            for name in ARTIFACT_NAMES:
                (staging / name).replace(input_root / name)
            return audit
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    write_json_atomic(run_path, {"config_digest": config_digest})
    (input_root / "refinement_effective_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=True), encoding="utf-8")
    return process_refinement(input_root, input_root, config, config_digest, books_manifest, client_factory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refine normalized Persian Whisper transcripts with causal context through vLLM's native synchronous batch chat endpoint.")
    parser.add_argument("--config", required=True, type=Path, help="YAML refinement configuration file with deterministic generation settings.")
    parser.add_argument("--input-root", required=True, type=Path, help="Long-audio root containing segments.jsonl and transcriptions.jsonl; refinement artifacts are written here.")
    parser.add_argument("--books-manifest", type=Path, help="Optional IranSeda books.jsonl used to join a safe audiobook title.")
    parser.add_argument("--force", action="store_true", help="Atomically replace only refinement artifacts created with incompatible settings.")
    args = parser.parse_args(argv)
    try:
        config, digest = load_config(args.config)
        audit = run_refinement(args.input_root, config, digest, books_manifest=args.books_manifest, force=args.force)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print("Contextual transcription refinement summary")
    for key, value in asdict(audit).items():
        print(f"  {key.replace('_', ' ')}: {value}")
    return 1 if audit.operational_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
