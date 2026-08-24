from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "ml"
    / "speech_data"
    / "scripts"
    / "upload_persian_audiobook_subset.sh"
)


def test_launcher_help_documents_screen_safe_resumable_behavior() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "OWNER/PersianAudiobook" in result.stdout
    assert "screen" in result.stdout
    assert "resumable" in result.stdout
    assert "512 MiB" in result.stdout
    assert "manually gated repository" in result.stdout


def test_launcher_creates_a_public_manually_gated_repository() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--public" in text
    assert "--gated-manual" in text
    assert "--private" not in text


def test_launcher_rejects_a_different_repository_name() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "owner/different-name"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "repository must be OWNER/PersianAudiobook" in result.stderr
