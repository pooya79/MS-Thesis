from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechInterval:
    start_sec: float
    end_sec: float
    mean_probability: float = 1.0
    max_probability: float = 1.0


@dataclass(frozen=True)
class SegmentBoundary:
    start_sec: float
    end_sec: float
    speech_seconds: float
    speech_ratio: float
    boundary_type: str
    boundary_silence_sec: float | None
    energy_dip_db: float | None


@dataclass(frozen=True)
class SegmentationSettings:
    target_seconds: float = 20.0
    preferred_min_seconds: float = 15.0
    preferred_max_seconds: float = 25.0
    hard_max_seconds: float = 28.0
    minimum_clip_seconds: float = 2.0
    useful_boundary_silence_seconds: float = 0.3
    boundary_padding_seconds: float = 0.15
    energy_search_start_seconds: float = 18.0
    energy_window_seconds: float = 0.2
    minimum_energy_dip_db: float = 6.0

    def __post_init__(self) -> None:
        durations = (
            self.target_seconds,
            self.preferred_min_seconds,
            self.preferred_max_seconds,
            self.hard_max_seconds,
            self.minimum_clip_seconds,
            self.useful_boundary_silence_seconds,
            self.boundary_padding_seconds,
            self.energy_search_start_seconds,
            self.energy_window_seconds,
        )
        if any(not math.isfinite(value) or value <= 0 for value in durations):
            raise ValueError("segmentation durations must be finite and greater than zero")
        if not self.preferred_min_seconds <= self.target_seconds <= self.preferred_max_seconds:
            raise ValueError("target_seconds must be inside the preferred duration range")
        if self.preferred_max_seconds > self.hard_max_seconds:
            raise ValueError("preferred_max_seconds must not exceed hard_max_seconds")
        if self.minimum_clip_seconds > self.preferred_min_seconds:
            raise ValueError("minimum_clip_seconds must not exceed preferred_min_seconds")
        if not self.energy_search_start_seconds < self.preferred_max_seconds:
            raise ValueError("energy search must begin before preferred_max_seconds")
        if not math.isfinite(self.minimum_energy_dip_db) or self.minimum_energy_dip_db < 0:
            raise ValueError("minimum_energy_dip_db must be finite and non-negative")


EnergyBoundary = Callable[[float, float, float], tuple[float, float]]


def _speech_seconds(intervals: Iterable[SpeechInterval], start: float, end: float) -> float:
    return sum(
        max(0.0, min(end, interval.end_sec) - max(start, interval.start_sec))
        for interval in intervals
    )


def _next_speech(intervals: list[SpeechInterval], at_or_after: float) -> SpeechInterval | None:
    return next((interval for interval in intervals if interval.end_sec > at_or_after), None)


def construct_segments(
    duration_sec: float,
    intervals: Iterable[SpeechInterval],
    settings: SegmentationSettings,
    energy_boundary: EnergyBoundary,
) -> list[SegmentBoundary]:
    """Construct deterministic non-overlapping clips from VAD speech intervals."""
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError("duration_sec must be finite and greater than zero")
    speech = sorted(intervals, key=lambda item: (item.start_sec, item.end_sec))
    if not speech:
        return []
    if any(item.start_sec < 0 or item.end_sec <= item.start_sec for item in speech):
        raise ValueError("speech intervals must have valid increasing timestamps")

    results: list[SegmentBoundary] = []
    cursor = max(0.0, speech[0].start_sec - settings.boundary_padding_seconds)
    final_speech_end = min(duration_sec, speech[-1].end_sec)

    while cursor < final_speech_end:
        active = _next_speech(speech, cursor)
        if active is None:
            break
        if active.start_sec - cursor > settings.preferred_max_seconds:
            cursor = max(cursor, active.start_sec - settings.boundary_padding_seconds)

        remaining_end = min(duration_sec, final_speech_end + settings.boundary_padding_seconds)
        if remaining_end - cursor <= settings.preferred_max_seconds:
            end = remaining_end
            boundary_type = "source_end"
            silence_duration = None
            energy_dip = None
        else:
            minimum = cursor + settings.preferred_min_seconds
            maximum = min(cursor + settings.preferred_max_seconds, duration_sec)
            speech_before_min = [
                item for item in speech if item.start_sec < minimum and item.end_sec > cursor
            ]
            next_after = next((item for item in speech if item.start_sec >= minimum), None)
            natural_end = (
                min(duration_sec, speech_before_min[-1].end_sec + settings.boundary_padding_seconds)
                if speech_before_min
                and speech_before_min[-1].end_sec <= minimum
                and (next_after is None or next_after.start_sec > maximum)
                else None
            )
            candidates: list[tuple[float, float]] = []
            for left, right in zip(speech, speech[1:]):
                gap = right.start_sec - left.end_sec
                midpoint = left.end_sec + gap / 2.0
                if gap >= settings.useful_boundary_silence_seconds and minimum <= midpoint <= maximum:
                    candidates.append((midpoint, gap))

            if natural_end is not None:
                end = natural_end
                boundary_type = "natural_short"
                silence_duration = None
                energy_dip = None
            elif candidates:
                end, silence_duration = min(
                    candidates,
                    key=lambda item: (abs(item[0] - (cursor + settings.target_seconds)), item[0]),
                )
                boundary_type = "silence"
                energy_dip = None
            else:
                search_start = min(maximum, cursor + settings.energy_search_start_seconds)
                candidate, energy_dip = energy_boundary(
                    search_start,
                    maximum,
                    settings.energy_window_seconds,
                )
                if energy_dip >= settings.minimum_energy_dip_db:
                    end = candidate
                    boundary_type = "energy_fallback"
                else:
                    end = maximum
                    boundary_type = "hard_cut"
                silence_duration = None

        end = min(end, duration_sec, cursor + settings.hard_max_seconds)
        if end <= cursor:
            raise RuntimeError("segment construction did not advance")
        speech_seconds = _speech_seconds(speech, cursor, end)
        duration = end - cursor
        if speech_seconds >= settings.minimum_clip_seconds:
            results.append(
                SegmentBoundary(
                    start_sec=cursor,
                    end_sec=end,
                    speech_seconds=speech_seconds,
                    speech_ratio=speech_seconds / duration,
                    boundary_type=boundary_type,
                    boundary_silence_sec=silence_duration,
                    energy_dip_db=energy_dip,
                )
            )

        next_interval = _next_speech(speech, end)
        if next_interval is None:
            break
        cursor = end
        if next_interval.start_sec - cursor > settings.preferred_max_seconds:
            cursor = max(cursor, next_interval.start_sec - settings.boundary_padding_seconds)

    return results
