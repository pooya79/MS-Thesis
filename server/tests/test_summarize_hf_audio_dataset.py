from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ml.speech_data.scripts.summarize_hf_audio_dataset import (
    build_summary,
    main,
    render_card_statistics,
)


def _write_audio(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(round(seconds * 16_000), dtype=np.float32), 16_000)


def test_builds_audio_and_transcription_statistics(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio-root"
    _write_audio(audio_root / "clips" / "source_000000.flac", 1.0)
    _write_audio(audio_root / "clips" / "source_000001.flac", 2.0)
    manifest = tmp_path / "refined.tsv"
    manifest.write_text(
        "path\tsentence\n"
        "source_000000.flac\tسلام دنیا\n"
        "source_000001.flac\tمتن آزمایشی دوم\n",
        encoding="utf-8",
    )
    segments = tmp_path / "segments.jsonl"
    segments.write_text(
        json.dumps({"id": "source_000000", "source_id": "source", "duration_sec": 1.0})
        + "\n"
        + json.dumps({"id": "source_000001", "source_id": "source", "duration_sec": 2.0})
        + "\n",
        encoding="utf-8",
    )

    summary = build_summary(
        manifest=manifest,
        audio_root=audio_root,
        segments_manifest=segments,
        refinement_summary=None,
        workers=2,
    )

    assert summary["examples"] == 2
    assert summary["source_recordings"] == 1
    assert summary["audio"]["total_duration_seconds"] == pytest.approx(3.0)
    assert summary["audio"]["sample_rate_hz"] == {"16000": 2}
    assert summary["audio"]["channels"] == {"1": 2}
    assert summary["transcriptions"]["total_whitespace_tokens"] == 5
    snippet = render_card_statistics(summary)
    assert "Dataset statistics" in snippet
    assert "3 whitespace" not in snippet
    assert "16,000 Hz" not in snippet


def test_cli_writes_json_and_card_snippet(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio-root"
    _write_audio(audio_root / "one.flac", 0.5)
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("path\tsentence\none.flac\tسلام\n", encoding="utf-8")
    output = tmp_path / "summary.json"
    snippet = tmp_path / "statistics.md"

    exit_code = main(
        [
            "--manifest",
            str(manifest),
            "--audio-root",
            str(audio_root),
            "--output",
            str(output),
            "--card-snippet-output",
            str(snippet),
            "--workers",
            "1",
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["examples"] == 1
    assert "| Examples | 1 |" in snippet.read_text(encoding="utf-8")
