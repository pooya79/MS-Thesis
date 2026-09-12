"""Persistent, terminal-independent progress and ETA reporting."""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def _now() -> datetime:
    return datetime.now().astimezone()


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ProgressReporter:
    """Track average throughput, ETA, and completion in an atomic JSON file."""

    def __init__(self, label: str, total: int, path: Path, *, initial: int = 0,
                 report_every_seconds: float = 30.0) -> None:
        if total < 1 or not 0 <= initial <= total:
            raise ValueError("progress requires total >= 1 and 0 <= initial <= total")
        self.label = label
        self.total = total
        self.path = path
        self.initial = initial
        self.completed = initial
        self.report_every_seconds = max(0.0, report_every_seconds)
        self.started_at = _now()
        self.started_clock = time.monotonic()
        self.last_report_clock = self.started_clock
        self._write("running")

    def _payload(self, state: str, *, error: str | None = None) -> dict[str, Any]:
        now, clock = _now(), time.monotonic()
        elapsed = max(0.0, clock - self.started_clock)
        processed = self.completed - self.initial
        rate = processed / elapsed if processed and elapsed > 0 else None
        remaining = self.total - self.completed
        eta = remaining / rate if rate else None
        finish = now + timedelta(seconds=eta) if eta is not None else None
        return {
            "label": self.label,
            "state": state,
            "started_at": self.started_at.isoformat(),
            "updated_at": now.isoformat(),
            "completed": self.completed,
            "total": self.total,
            "remaining": remaining,
            "elapsed_seconds": round(elapsed, 3),
            "items_per_second": round(rate, 6) if rate is not None else None,
            "eta_seconds": round(eta, 3) if eta is not None else None,
            "estimated_finish_at": finish.isoformat() if finish is not None else None,
            **({"error": error} if error else {}),
        }

    def _write(self, state: str, *, error: str | None = None) -> dict[str, Any]:
        payload = self._payload(state, error=error)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)
        return payload

    def update(self, completed: int, message: str | None = None) -> None:
        if not self.completed <= completed <= self.total:
            raise ValueError("progress completion must be monotonic and no greater than total")
        self.completed = completed
        clock = time.monotonic()
        if completed < self.total and completed > self.initial + 1 and clock - self.last_report_clock < self.report_every_seconds:
            return
        payload = self._write("completed" if completed == self.total else "running")
        self.last_report_clock = clock
        suffix = f" {message}" if message else ""
        print(
            f"[{self.label}] {completed}/{self.total} "
            f"elapsed={_duration(payload['elapsed_seconds'])} "
            f"eta={_duration(payload['eta_seconds'])} "
            f"finish={payload['estimated_finish_at'] or 'unknown'}{suffix}",
            flush=True,
        )

    def fail(self, error: BaseException) -> None:
        payload = self._write("failed", error=f"{type(error).__name__}: {error}")
        print(f"[{self.label}] failed after {_duration(payload['elapsed_seconds'])}: {error}", flush=True)

    def __enter__(self) -> "ProgressReporter":
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> bool:
        if exc is not None:
            self.fail(exc)
        elif self.completed < self.total:
            self._write("incomplete")
        return False
