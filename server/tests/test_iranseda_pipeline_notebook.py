from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import nbformat


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = ROOT / "notebooks" / "iranseda_pipeline_demo.ipynb"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_artifacts(root: Path) -> None:
    root.mkdir(parents=True)
    segments = [
        {
            "id": "book-1_000000",
            "source_id": "1:track",
            "path": "clips/book-1_000000.flac",
            "clip_checksum": "sha256:clip-0",
            "start_sec": 0.0,
            "end_sec": 18.0,
            "duration_sec": 18.0,
            "speech_seconds": 16.0,
            "speech_ratio": 0.889,
            "boundary_type": "silence",
            "boundary_silence_sec": 0.4,
            "energy_dip_db": None,
            "config_digest": "sha256:segment",
        },
        {
            "id": "book-1_000001",
            "source_id": "1:track",
            "path": "clips/book-1_000001.flac",
            "clip_checksum": "sha256:clip-1",
            "start_sec": 18.0,
            "end_sec": 38.0,
            "duration_sec": 20.0,
            "speech_seconds": 19.0,
            "speech_ratio": 0.95,
            "boundary_type": "energy_fallback",
            "boundary_silence_sec": None,
            "energy_dip_db": 8.1,
            "config_digest": "sha256:segment",
        },
    ]
    transcriptions = [
        {
            "id": "book-1_000000",
            "source_id": "1:track",
            "path": "clips/book-1_000000.flac",
            "raw_transcript": " سلام، دنيا! ",
            "normalized_transcript": "سلام دنیا",
            "generation": {"do_sample": False, "num_beams": 1},
            "config_digest": "sha256:whisper",
        },
        {
            "id": "book-1_000001",
            "source_id": "1:track",
            "path": "clips/book-1_000001.flac",
            "raw_transcript": "این يك آزمون است.",
            "normalized_transcript": "این یک آزمون است",
            "generation": {"do_sample": False, "num_beams": 1},
            "config_digest": "sha256:whisper",
        },
    ]
    common = {
        "source_id": "1:track",
        "title": "کتاب نمونه",
        "description": "شرح نمونه",
        "preceding_context": [],
        "following_context": [{"id": "book-1_000001", "text": "این یک آزمون است"}],
        "model": "test-model",
        "model_parameters": {"temperature": 0, "seed": 0},
        "prompt_version": "persian-transcript-refinement-v1",
        "schema_version": "persian-transcript-refinement-schema-v2",
        "response_schema": {"type": "object"},
        "operational": False,
    }
    accepted = {
        **common,
        "id": "book-1_000000",
        "path": "clips/book-1_000000.flac",
        "target_whisper_text": "سلام دنیا",
        "target": {"id": "book-1_000000", "text": "سلام دنیا"},
        "rendered_prompt": "[TARGET WHISPER TEXT]\nسلام دنیا",
        "raw_response_text": '{"cleaned_text":"سلام دنیا","uncertain":false}',
        "parsed_response": {"cleaned_text": "سلام دنیا", "uncertain": False},
        "cleaned_text": "سلام دنیا",
        "validation_metrics": {"normalized_edit_distance": 0.0},
    }
    rejected = {
        **common,
        "id": "book-1_000001",
        "path": "clips/book-1_000001.flac",
        "target_whisper_text": "این یک آزمون است",
        "target": {"id": "book-1_000001", "text": "این یک آزمون است"},
        "preceding_context": [{"id": "book-1_000000", "text": "سلام دنیا"}],
        "following_context": [],
        "rendered_prompt": "[TARGET WHISPER TEXT]\nاین یک آزمون است",
        "raw_response_text": '{"cleaned_text":"این یک آزمایش است","uncertain":true}',
        "parsed_response": {"cleaned_text": "این یک آزمایش است", "uncertain": True},
        "validation_metrics": {"maximum_normalized_edit_distance": 0.35},
        "reason": "model_uncertain",
        "detail": "model_uncertain",
    }

    _write_json(root / "summary.json", {"clips_written": 2, "duration_seconds": 38.0})
    _write_jsonl(root / "segments.jsonl", segments)
    _write_jsonl(
        root / "vad_intervals.jsonl",
        [{"source_id": "1:track", "start_sec": 0.0, "end_sec": 38.0}],
    )
    _write_json(root / "transcription_summary.json", {"clips_accepted": 2})
    _write_jsonl(root / "transcriptions.jsonl", transcriptions)
    _write_jsonl(root / "transcription_rejected.jsonl", [])
    _write_json(root / "refinement_summary.json", {"targets_accepted": 1, "targets_rejected": 1})
    _write_jsonl(root / "refinements.jsonl", [accepted])
    _write_jsonl(root / "refinement_rejected.jsonl", [rejected])
    with (root / "refined_transcription.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sentence"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"path": "book-1_000000.flac", "sentence": "سلام دنیا"})


def _load_notebook() -> Any:
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def test_notebook_is_artifact_only_and_exposes_expected_sections() -> None:
    notebook = _load_notebook()
    markdown = "\n".join(
        "".join(cell.source) if isinstance(cell.source, list) else cell.source
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    )
    code = "\n".join(
        "".join(cell.source) if isinstance(cell.source, list) else cell.source
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    for heading in (
        "Segmentation and VAD outputs",
        "Whisper and deterministic normalization outputs",
        "Exact contextual LLM audit",
        "Multiple LLM inputs and outputs",
        "Refinement decisions and final labels",
    ):
        assert heading in markdown
    for forbidden in ("from_pretrained", ".generate(", "import torch", "import transformers", "urllib", "requests"):
        assert forbidden not in code
    assert "rendered_prompt" in code
    assert "ARTIFACT_OVERRIDES" in code


def test_helpers_prioritize_overrides_and_sample_deterministically(tmp_path: Path) -> None:
    notebook = _load_notebook()
    namespace: dict[str, Any] = {}
    for cell in notebook.cells:
        if cell.cell_type == "code" and set(cell.metadata.get("tags", [])) & {
            "iranseda-demo-configuration",
            "iranseda-demo-helpers",
        }:
            exec(compile(cell.source, f"{NOTEBOOK_PATH}:{cell.id}", "exec"), namespace)

    root = tmp_path / "root"
    override = tmp_path / "elsewhere" / "segments.jsonl"
    resolved = namespace["resolve_artifact_paths"](root, {"segments": override})
    assert resolved["segments"] == override.resolve()
    assert resolved["transcriptions"] == root.resolve() / "transcriptions.jsonl"

    rows = [{"id": str(index)} for index in range(10)]
    jsonl_path = tmp_path / "large.jsonl"
    _write_jsonl(jsonl_path, rows)
    assert namespace["read_jsonl"](jsonl_path, 3) == rows[:3]
    assert len(namespace["ARTIFACT_RECORD_LIMITS"]) == 7

    assert namespace["sample_rows"](rows, 4, 42) == namespace["sample_rows"](rows, 4, 42)
    mixed = namespace["mixed_llm_sample"](rows[:5], rows[5:], 6, 42)
    assert len(mixed) == 6
    assert any(row in rows[:5] for row in mixed)
    assert any(row in rows[5:] for row in mixed)
    assert namespace["mixed_llm_sample"]([], rows[5:], 3, 42) == namespace["mixed_llm_sample"]([], rows[5:], 3, 42)


def test_notebook_executes_against_saved_synthetic_artifacts(
    tmp_path: Path, monkeypatch: Any
) -> None:
    artifact_root = tmp_path / "iranseda-run"
    _make_artifacts(artifact_root)
    monkeypatch.setenv("IRANSEDA_PIPELINE_ROOT", str(artifact_root))
    monkeypatch.setenv("IRANSEDA_DEMO_AUDIO", "false")
    monkeypatch.chdir(ROOT)

    notebook = _load_notebook()
    namespace: dict[str, Any] = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, f"{NOTEBOOK_PATH}:{cell.id}", "exec"), namespace)

    assert namespace["audit_record"]["rendered_prompt"].startswith("[TARGET WHISPER TEXT]")
    assert {row.get("reason") for row in namespace["gallery"]} == {None, "model_uncertain"}
    assert namespace["refined_rows"] == [
        {"path": "book-1_000000.flac", "sentence": "سلام دنیا"}
    ]
