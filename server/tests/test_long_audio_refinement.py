from __future__ import annotations

import csv
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest
import yaml

from ml.speech_data.long_audio_asr_pipeline.refine_transcriptions import (
    PROMPT_VERSION,
    RESPONSE_SCHEMA,
    SCHEMA_VERSION,
    VLLMBatchClient,
    build_prompt,
    group_targets,
    load_config,
    normalized_edit_distance,
    run_refinement,
    title_for_source,
    validate_response,
)


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def make_root(root: Path, sources: dict[str, list[str]]) -> None:
    segments: list[dict[str, Any]] = []
    transcriptions: list[dict[str, Any]] = []
    for source, texts in sources.items():
        for index, text in enumerate(texts):
            target_id = f"{source}-{index}"
            segments.append(
                {
                    "id": target_id,
                    "source_id": source,
                    "start_sec": index * 20.0,
                    "path": f"clips/{target_id}.flac",
                    "clip_checksum": f"sha256:{target_id}",
                }
            )
            transcriptions.append(
                {
                    "id": target_id,
                    "source_id": source,
                    "path": f"clips/{target_id}.flac",
                    "normalized_transcript": text,
                    "clip_checksum": f"sha256:{target_id}",
                    "config_digest": "sha256:whisper",
                }
            )
    (root / "segments.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in reversed(segments)),
        encoding="utf-8",
    )
    (root / "transcriptions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in transcriptions),
        encoding="utf-8",
    )


def config(*, batch_size: int = 16, context_size: int = 3, threshold: float = 0.35) -> dict[str, Any]:
    return {
        "server": {
            "base_url": "http://vllm.test",
            "model": "test-model",
            "timeout_seconds": 10,
            "retry_count": 1,
            "api_key_env": None,
        },
        "context": {"size": context_size},
        "batch": {"size": batch_size},
        "generation": {"temperature": 0, "top_p": 1, "n": 1, "seed": 7, "max_tokens": 100},
        "validation": {"maximum_normalized_edit_distance": threshold},
        "pipeline_version": 1,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
    }


class FakeClient:
    def __init__(self, outputs: dict[str, str] | None = None, *, malformed: bool = False) -> None:
        self.outputs = outputs or {}
        self.requests: list[dict[str, Any]] = []
        self.preflight_calls = 0
        self.malformed = malformed

    def preflight(self) -> None:
        self.preflight_calls += 1

    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        if self.malformed:
            return {"choices": []}
        choices = []
        for index, messages in enumerate(request["messages"]):
            prompt = messages[0]["content"]
            target_line = prompt.split("[TARGET WHISPER TEXT]\n", 1)[1].split("\n", 1)[0]
            target_id, text = target_line[1:].split("] ", 1)
            cleaned = self.outputs.get(target_id, text)
            choices.append(
                {
                    "index": index,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "cleaned_text": cleaned,
                                "uncertain": False,
                                "change_categories": ["none"] if cleaned == text else ["asr_substitution"],
                            },
                            ensure_ascii=False,
                        ),
                    },
                }
            )
        return {"choices": list(reversed(choices))}


def test_config_validation_digest_and_secret_name_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "refinement.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "server": {"model": "served", "api_key_env": "VLLM_KEY"},
                "context": {"size": 3},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_KEY", "super-secret")
    first, first_digest = load_config(path)
    second, second_digest = load_config(path)
    assert first_digest == second_digest
    assert first["server"]["api_key_env"] == "VLLM_KEY"
    assert "super-secret" not in json.dumps(first)
    path.write_text("server:\n  model: served\ngeneration:\n  temperature: 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="temperature"):
        load_config(path)


def test_vllm_client_preflight_and_identical_payload_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    client = VLLMBatchClient(config())
    calls: list[tuple[str, bytes | None]] = []

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def close(self) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices":[]}'

    attempts = 0

    def urlopen(request: Any, timeout: float) -> Response:
        nonlocal attempts
        calls.append((request.get_method(), request.data))
        if request.get_method() == "GET":
            raise urllib.error.HTTPError(request.full_url, 405, "method", {}, None)
        attempts += 1
        if attempts == 1:
            raise TimeoutError("temporary timeout")
        return Response()

    monkeypatch.setattr("ml.speech_data.long_audio_asr_pipeline.refine_transcriptions.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("ml.speech_data.long_audio_asr_pipeline.refine_transcriptions.time.sleep", lambda _: None)
    client.preflight()
    body = {"messages": [[{"role": "user", "content": "سلام"}]]}
    assert client.complete(body) == {"choices": []}
    assert calls[1][1] == calls[2][1]


def test_vllm_preflight_rejects_missing_native_batch_route(monkeypatch: pytest.MonkeyPatch) -> None:
    client = VLLMBatchClient(config())

    def missing(request: Any, timeout: float) -> None:
        raise urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)

    monkeypatch.setattr("ml.speech_data.long_audio_asr_pipeline.refine_transcriptions.urllib.request.urlopen", missing)
    with pytest.raises(RuntimeError, match="required /v1/chat/completions/batch"):
        client.preflight()


def test_grouping_orders_by_source_time_and_orphans_are_isolated() -> None:
    targets = [
        {"id": "b", "source_id": "s", "segment": {"start_sec": 2}},
        {"id": "a", "source_id": "s", "segment": {"start_sec": 1}},
        {"id": "x", "source_id": None, "segment": {"start_sec": 0}},
        {"id": "y", "source_id": None, "segment": {"start_sec": 0}},
    ]
    groups = group_targets(targets)
    assert sorted([item["id"] for item in group] for group in groups) == [["a", "b"], ["x"], ["y"]]


def test_prompt_labels_context_and_title_without_boundary_completion() -> None:
    target = {"id": "t", "normalized_transcript": "متن هدف"}
    prompt = build_prompt(target, [{"id": "p", "text": "متن پیش"}], [{"id": "f", "text": "متن پس"}], "نام کتاب")
    assert "[AUDIOBOOK TITLE]\nنام کتاب" in prompt
    assert "[REFINED PRECEDING CONTEXT]\n- [p] متن پیش" in prompt
    assert "[TARGET WHISPER TEXT]\n[t] متن هدف" in prompt
    assert "[FOLLOWING WHISPER CONTEXT]\n- [f] متن پس" in prompt
    assert "never copy context" in prompt
    assert title_for_source("123:456", {"123": "کتاب"}) == "کتاب"
    assert title_for_source("unsafe", {"123": "کتاب"}) is None


def test_validation_schema_persian_digits_uncertainty_and_distance() -> None:
    valid = json.dumps({"change_categories": ["orthography"], "uncertain": False, "cleaned_text": "سلام، ۱۲!"}, ensure_ascii=False)
    parsed, reason, metrics = validate_response(valid, "سلام ۱۲", 0.35)
    assert reason is None
    assert parsed and parsed["cleaned_text"] == "سلام، ۱۲!"
    assert metrics["normalized_cleaned_text"] == "سلام ۱۲"
    assert metrics["normalized_edit_distance"] == 0
    uncertain = json.dumps({"cleaned_text": "سلام", "uncertain": True, "change_categories": ["none"]}, ensure_ascii=False)
    assert validate_response(uncertain, "سلام", 0.35)[1] == "model_uncertain"
    latin = json.dumps({"cleaned_text": "hello", "uncertain": False, "change_categories": ["asr_substitution"]})
    assert validate_response(latin, "سلام", 1)[1] == "persian_normalization_rejected"
    digits = json.dumps({"cleaned_text": "سلام ۱۳", "uncertain": False, "change_categories": ["asr_substitution"]}, ensure_ascii=False)
    assert validate_response(digits, "سلام ۱۲", 1)[1] == "numeric_tokens_changed"
    changed = json.dumps({"cleaned_text": "کاملا متفاوت", "uncertain": False, "change_categories": ["asr_substitution"]}, ensure_ascii=False)
    assert validate_response(changed, "سلام دنیا", 0.1)[1] == "edit_distance_exceeded"
    assert normalized_edit_distance("abc", "adc") == pytest.approx(1 / 3)


def test_native_batch_scheduling_response_mapping_and_context_are_causal(tmp_path: Path) -> None:
    make_root(tmp_path, {"10:track": ["من اول", "من دوم", "من سوم"], "20:track": ["تو اول", "تو دوم"]})
    books = tmp_path / "books.jsonl"
    books.write_text('{"id":"10","title":"کتاب من"}\n', encoding="utf-8")
    client = FakeClient({"10:track-0": "من یک", "20:track-0": "تو یک"})
    audit = run_refinement(
        tmp_path, config(batch_size=2, context_size=1, threshold=1), "sha256:config",
        books_manifest=books, client_factory=lambda _: client,
    )
    assert [len(request["messages"]) for request in client.requests] == [2, 2, 1]
    for request in client.requests:
        target_ids = [messages[0]["content"].split("[TARGET WHISPER TEXT]\n[")[1].split("]", 1)[0] for messages in request["messages"]]
        assert len({target_id.rsplit("-", 1)[0] for target_id in target_ids}) == len(target_ids)
        assert request["stream"] is False
        assert request["response_format"]["json_schema"]["schema"] == RESPONSE_SCHEMA
    second_prompt = client.requests[1]["messages"][0][0]["content"]
    assert "[10:track-0] من یک" in second_prompt
    assert "[10:track-2] من سوم" in second_prompt
    accepted = jsonl(tmp_path / "refinements.jsonl")
    assert {item["id"]: item["cleaned_text"] for item in accepted}["10:track-0"] == "من یک"
    assert audit.targets_accepted == 5
    with (tmp_path / "refined_transcription.tsv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle, delimiter="\t"))) == 5
    assert (tmp_path / "transcriptions.jsonl").is_file()


def test_identical_run_reuses_without_client_and_context_change_cascades(tmp_path: Path) -> None:
    make_root(tmp_path, {"s": ["متن اول", "متن دوم"]})
    first = FakeClient()
    run_refinement(tmp_path, config(), "sha256:config", client_factory=lambda _: first)

    def no_client(_: dict[str, Any]) -> FakeClient:
        raise AssertionError("identical results must be reused")

    reused = run_refinement(tmp_path, config(), "sha256:config", client_factory=no_client)
    assert reused.targets_reused == 2
    transcripts = jsonl(tmp_path / "transcriptions.jsonl")
    transcripts[0]["normalized_transcript"] = "متن نخست"
    (tmp_path / "transcriptions.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in transcripts), encoding="utf-8")
    changed = FakeClient()
    rerun = run_refinement(tmp_path, config(), "sha256:config", client_factory=lambda _: changed)
    assert rerun.targets_processed == 2
    assert rerun.targets_reused == 0


def test_nonoperational_rejection_is_reused_but_malformed_batch_is_retried(tmp_path: Path) -> None:
    make_root(tmp_path, {"s": ["درود"]})

    class RejectingClient(FakeClient):
        def complete(self, request: dict[str, Any]) -> dict[str, Any]:
            result = super().complete(request)
            result["choices"][0]["message"]["content"] = json.dumps(
                    {"cleaned_text": "درود", "uncertain": True, "change_categories": ["none"]}, ensure_ascii=False
            )
            return result

    run_refinement(tmp_path, config(), "sha256:config", client_factory=lambda _: RejectingClient())
    assert jsonl(tmp_path / "refinement_rejected.jsonl")[0]["reason"] == "model_uncertain"
    reused = run_refinement(tmp_path, config(), "sha256:config", client_factory=lambda _: (_ for _ in ()).throw(AssertionError()))
    assert reused.targets_reused == 1

    make_root(tmp_path, {"s": ["سلام"]})
    malformed = FakeClient(malformed=True)
    failed = run_refinement(tmp_path, config(), "sha256:config", client_factory=lambda _: malformed)
    assert failed.operational_failures == 1
    retry = FakeClient()
    recovered = run_refinement(tmp_path, config(), "sha256:config", client_factory=lambda _: retry)
    assert recovered.operational_failures == 0
    assert recovered.targets_processed == 1


def test_incompatible_config_requires_force_and_force_preserves_on_failure(tmp_path: Path) -> None:
    make_root(tmp_path, {"s": ["سلام"]})
    run_refinement(tmp_path, config(), "sha256:one", client_factory=lambda _: FakeClient())
    original = (tmp_path / "refinements.jsonl").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="--force"):
        run_refinement(tmp_path, config(), "sha256:two", client_factory=lambda _: FakeClient())
    with pytest.raises(RuntimeError, match="preserved"):
        run_refinement(tmp_path, config(), "sha256:two", force=True, client_factory=lambda _: FakeClient(malformed=True))
    assert (tmp_path / "refinements.jsonl").read_text(encoding="utf-8") == original
