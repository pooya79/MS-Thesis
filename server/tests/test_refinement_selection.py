from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ml.speech_data.long_audio_asr_pipeline.refine_transcriptions import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    load_config,
    run_refinement,
)
from ml.speech_data.long_audio_asr_pipeline.segment_audio import write_json_atomic
from ml.speech_data.long_audio_asr_pipeline.select_refinement_subset import (
    EligibleSource,
    create_selection_manifest,
    main,
    select_sources,
)
from server.tests.test_long_audio_refinement import FakeClient, make_root


def refinement_config(selection_manifest: Path | None = None) -> dict[str, Any]:
    return {
        "server": {
            "base_url": "http://vllm.test",
            "model": "test-model",
            "timeout_seconds": 10,
            "retry_count": 1,
            "api_key_env": None,
        },
        "context": {"size": 1},
        "batch": {"size": 16},
        "generation": {
            "temperature": 0,
            "top_p": 1,
            "n": 1,
            "seed": 7,
            "max_tokens": 100,
        },
        "validation": {"maximum_normalized_edit_distance": 0.35},
        "selection": {
            "manifest": str(selection_manifest) if selection_manifest is not None else None
        },
        "pipeline_version": 1,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
    }


def write_selection_root(root: Path) -> None:
    segments = [
        {"id": "a-0", "source_id": "a", "duration_sec": 40.0},
        {"id": "a-1", "source_id": "a", "duration_sec": 20.0},
        {"id": "b-0", "source_id": "b", "duration_sec": 50.0},
        {"id": "c-0", "source_id": "c", "duration_sec": 70.0},
        {"id": "partial-0", "source_id": "partial", "duration_sec": 30.0},
        {"id": "partial-1", "source_id": "partial", "duration_sec": 30.0},
    ]
    transcriptions = [
        {"id": segment["id"], "normalized_transcript": "متن"}
        for segment in segments
        if segment["id"] != "partial-1"
    ]
    (root / "segments.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in segments), encoding="utf-8"
    )
    (root / "transcriptions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in transcriptions),
        encoding="utf-8",
    )


def test_closest_prefix_policy_prefers_under_target_tie_and_never_returns_empty() -> None:
    sources = [
        EligibleSource("a", 60.0, 2),
        EligibleSource("b", 60.0, 2),
    ]
    assert len(select_sources(sources, 100.0, seed=0)) == 2
    assert len(select_sources(sources, 90.0, seed=0)) == 1
    assert len(select_sources(sources, 1.0, seed=0)) == 1
    with pytest.raises(ValueError, match="positive finite"):
        select_sources(sources, 0, seed=0)


def test_selection_is_seeded_complete_and_excludes_partial_sources(tmp_path: Path) -> None:
    write_selection_root(tmp_path)
    first = create_selection_manifest(tmp_path, requested_hours=90 / 3600, seed=4)
    second = create_selection_manifest(tmp_path, requested_hours=90 / 3600, seed=4)
    assert first == second
    assert all(item["source_id"] != "partial" for item in first["sources"])
    assert first["selected_duration_seconds"] == pytest.approx(
        sum(item["duration_seconds"] for item in first["sources"])
    )
    counts = {"a": 2, "b": 1, "c": 1}
    assert all(item["segment_count"] == counts[item["source_id"]] for item in first["sources"])
    alternatives = [
        create_selection_manifest(tmp_path, requested_hours=90 / 3600, seed=seed)["sources"]
        for seed in range(5, 15)
    ]
    assert any(candidate != first["sources"] for candidate in alternatives)


def test_selector_validates_hours_and_exposes_cli_help(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_selection_root(tmp_path)
    with pytest.raises(ValueError, match="hours must be"):
        create_selection_manifest(tmp_path, requested_hours=float("nan"), seed=0)
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--hours" in help_text
    assert "--seed" in help_text
    assert "--output" in help_text
    output = tmp_path / "selected.json"
    assert main(
        [
            "--input-root",
            str(tmp_path),
            "--hours",
            str(90 / 3600),
            "--seed",
            "4",
            "--output",
            str(output),
        ]
    ) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["schema_version"] == "refinement-source-selection-v1"
    assert written["sources"]
    output_text = capsys.readouterr().out
    assert "[select-refinement] segments loaded records=6" in output_text
    assert "[select-refinement] validating segment 6/6" in output_text
    assert "[select-refinement] transcriptions loaded records=5" in output_text
    assert "[select-refinement] validating transcription 5/5" in output_text
    assert (
        "[select-refinement] eligibility complete total_groups=4 eligible=3 incomplete=1"
        in output_text
    )
    assert "[select-refinement] selection manifest ready" in output_text
    assert "selected sources:" in output_text


def test_refinement_filters_to_selected_complete_source_and_checks_staleness(
    tmp_path: Path,
) -> None:
    make_root(tmp_path, {"a": ["الف اول", "الف دوم"], "b": ["ب اول", "ب دوم"]})
    selection_path = tmp_path / "selection.json"
    manifest = create_selection_manifest(tmp_path, requested_hours=40 / 3600, seed=0)
    write_json_atomic(selection_path, manifest)
    selected_source = manifest["sources"][0]["source_id"]
    client = FakeClient()
    audit = run_refinement(
        tmp_path,
        refinement_config(selection_path),
        "sha256:selection-config",
        client_factory=lambda _: client,
    )
    assert audit.targets_total == 2
    prompts = [
        messages[0]["content"]
        for request in client.requests
        for messages in request["messages"]
    ]
    assert len(prompts) == 2
    selected_prefix = "الف" if selected_source == "a" else "ب"
    assert all(selected_prefix in prompt for prompt in prompts)
    assert f"- {selected_prefix} دوم" in prompts[0]
    assert all(f"[{selected_source}-" not in prompt for prompt in prompts)
    assert all("clips/" not in prompt and ".flac" not in prompt for prompt in prompts)

    transcript_path = tmp_path / "transcriptions.jsonl"
    transcript_path.write_text(
        transcript_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="stale"):
        run_refinement(
            tmp_path,
            refinement_config(selection_path),
            "sha256:selection-config",
            client_factory=lambda _: (_ for _ in ()).throw(AssertionError()),
        )


def test_refinement_rejects_unknown_and_duplicate_selected_sources_before_client(
    tmp_path: Path,
) -> None:
    make_root(tmp_path, {"a": ["متن اول"]})
    selection_path = tmp_path / "selection.json"
    manifest = create_selection_manifest(tmp_path, requested_hours=1, seed=0)
    manifest["sources"][0]["source_id"] = "missing"
    write_json_atomic(selection_path, manifest)
    with pytest.raises(ValueError, match="unknown or unusable"):
        run_refinement(
            tmp_path,
            refinement_config(selection_path),
            "sha256:unknown",
            client_factory=lambda _: (_ for _ in ()).throw(AssertionError()),
        )

    manifest["sources"].append(dict(manifest["sources"][0]))
    write_json_atomic(selection_path, manifest)
    with pytest.raises(ValueError, match="duplicate source_id"):
        run_refinement(
            tmp_path,
            refinement_config(selection_path),
            "sha256:unknown",
            client_factory=lambda _: (_ for _ in ()).throw(AssertionError()),
        )


def test_config_resolves_optional_selection_relative_to_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "refinement.yaml"
    path.write_text(
        "server:\n  model: served\nselection:\n  manifest: ../selection.json\n",
        encoding="utf-8",
    )
    loaded, _ = load_config(path)
    assert loaded["selection"]["manifest"] == str((tmp_path / "selection.json").resolve())
