from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_make_run_loads_dotenv_without_dependency_installation(tmp_path: Path) -> None:
    uv = tmp_path / "uv"
    uv.write_text(
        '#!/bin/sh\n'
        '[ "$1" = run ] && [ "$2" = --no-sync ] || exit 1\n'
        'shift 2\n'
        '[ "$1" = --env-file ] && [ "$2" = .env ] || exit 2\n'
        'shift 2\n'
        '[ "$1" = uvicorn ] && [ "$2" = server.app.main:app ] || exit 3\n'
    )
    uv.chmod(0o755)
    subprocess.run(
        ["make", "run"], cwd=ROOT, check=True, capture_output=True, timeout=10,
        env={**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"},
    )


def test_make_help_explains_installed_environment() -> None:
    result = subprocess.run(
        ["make", "help"], cwd=ROOT, check=True, capture_output=True, text=True, timeout=10,
    )
    assert "make run" in result.stdout
    assert ".env" in result.stdout
    assert "no dependency sync" in result.stdout
