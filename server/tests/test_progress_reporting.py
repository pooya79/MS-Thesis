from __future__ import annotations

import json

import pytest

from ml.utils.progress import ProgressReporter


def test_progress_reporter_writes_eta_and_completion(tmp_path, capsys):
    path = tmp_path / "progress.json"
    with ProgressReporter("work", 2, path, report_every_seconds=0) as progress:
        assert json.loads(path.read_text())["state"] == "running"
        progress.update(1, "first")
        progress.update(2, "second")
    payload = json.loads(path.read_text())
    assert payload["state"] == "completed"
    assert payload["completed"] == payload["total"] == 2
    assert payload["eta_seconds"] == 0
    assert payload["estimated_finish_at"] is not None
    output = capsys.readouterr().out
    assert "eta=" in output and "finish=" in output


def test_progress_reporter_persists_failure(tmp_path):
    path = tmp_path / "progress.json"
    with pytest.raises(RuntimeError, match="broken"):
        with ProgressReporter("work", 2, path) as progress:
            progress.update(1)
            raise RuntimeError("broken")
    payload = json.loads(path.read_text())
    assert payload["state"] == "failed"
    assert payload["completed"] == 1
    assert payload["error"] == "RuntimeError: broken"


def test_progress_reporter_can_report_to_console_without_a_file(capsys):
    with ProgressReporter("dry-run", 1, None) as progress:
        progress.update(1)
    assert "[dry-run] 1/1" in capsys.readouterr().out
