from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest
import soundfile as sf
import yaml

from ml.speech_data.long_audio_asr_pipeline.segment_audio import (
    SileroVadDetector,
    SourceRecord,
    discover_inputs,
    load_config,
    run_pipeline,
    sha256_file,
    sources_from_manifest,
    validate_output_location,
)
from ml.speech_data.long_audio_asr_pipeline.segmentation import (
    SegmentationSettings,
    SpeechInterval,
    construct_segments,
)


CONFIG_PATH = Path("configs/long_audio_asr_pipeline/segmentation.yaml")


class FakeVad:
    def __init__(self, intervals: list[SpeechInterval]) -> None:
        self.intervals = intervals

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "backend": "fake",
            "model": "fixture",
            "package": "tests",
            "package_version": "1",
            "runtime": "python",
        }

    def detect(self, path: Path, settings: dict[str, object]) -> list[SpeechInterval]:
        assert sf.info(path).samplerate == 16000
        return self.intervals


def jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def make_audio(path: Path, duration: float = 42.0, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(duration * sample_rate)
    time = np.arange(frames, dtype=np.float32) / sample_rate
    audio = 0.1 * np.sin(2 * np.pi * 220 * time)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def test_selects_silence_nearest_target_and_keeps_segments_non_overlapping() -> None:
    intervals = [
        SpeechInterval(0.2, 18.0),
        SpeechInterval(18.6, 39.0),
    ]

    segments = construct_segments(
        40.0,
        intervals,
        SegmentationSettings(),
        lambda start, end, window: (21.0, 8.0),
    )

    assert len(segments) == 2
    assert segments[0].boundary_type == "silence"
    assert segments[0].end_sec == pytest.approx(18.3)
    assert segments[1].start_sec == segments[0].end_sec
    assert segments[1].end_sec == pytest.approx(39.15)
    assert all(segment.end_sec - segment.start_sec <= 28.0 for segment in segments)


def test_uses_energy_fallback_then_hard_cut() -> None:
    speech = [SpeechInterval(0.0, 60.0)]

    energy = construct_segments(
        60.0,
        speech,
        SegmentationSettings(),
        lambda start, end, window: (start + 1.5, 9.0),
    )
    hard = construct_segments(
        60.0,
        speech,
        SegmentationSettings(),
        lambda start, end, window: (start + 1.5, 2.0),
    )

    assert energy[0].boundary_type == "energy_fallback"
    assert energy[0].end_sec == 19.5
    assert hard[0].boundary_type == "hard_cut"
    assert hard[0].end_sec == 25.0


def test_keeps_natural_short_clip_without_padding_to_fifteen_seconds() -> None:
    segments = construct_segments(
        8.0,
        [SpeechInterval(0.5, 7.0)],
        SegmentationSettings(),
        lambda start, end, window: (end, 0.0),
    )

    assert len(segments) == 1
    assert segments[0].start_sec == pytest.approx(0.35)
    assert segments[0].end_sec == pytest.approx(7.15)
    assert segments[0].end_sec - segments[0].start_sec < 15.0


def test_rejects_tail_with_less_than_minimum_speech() -> None:
    segments = construct_segments(
        30.0,
        [SpeechInterval(0.0, 20.0), SpeechInterval(28.5, 29.0)],
        SegmentationSettings(),
        lambda start, end, window: (20.0, 8.0),
    )

    assert all(segment.speech_seconds >= 2.0 for segment in segments)


def test_manifest_is_generic_and_preserves_iranseda_shaped_metadata(tmp_path: Path) -> None:
    audio = tmp_path / "clips" / "station" / "episode.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"audio fixture")
    checksum = f"sha256:{hashlib.sha256(audio.read_bytes()).hexdigest()}"
    manifest = tmp_path / "downloads.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "episode:42",
                "source_kind": "radio",
                "path": "clips/station/episode.mp3",
                "checksum": checksum,
                "url": "https://example.invalid/audio",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sources = sources_from_manifest(manifest)

    assert sources == [
        SourceRecord(
            "episode:42",
            audio.resolve(),
            checksum,
            {
                "id": "episode:42",
                "source_kind": "radio",
                "path": "clips/station/episode.mp3",
                "checksum": checksum,
                "url": "https://example.invalid/audio",
            },
        )
    ]


def test_manifest_rejects_path_escape_and_duplicate_ids(tmp_path: Path) -> None:
    escaped = tmp_path / "escaped.jsonl"
    escaped.write_text('{"id":"one","path":"../outside.wav"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        sources_from_manifest(escaped)

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        '{"id":"one","path":"one.wav"}\n{"id":"one","path":"two.wav"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate source id"):
        sources_from_manifest(duplicate)


def test_directory_discovery_is_recursive_deterministic_and_audio_only(tmp_path: Path) -> None:
    make_audio(tmp_path / "nested" / "b.wav", duration=1.0)
    make_audio(tmp_path / "a.flac", duration=1.0)
    (tmp_path / "ignore.txt").write_text("no", encoding="utf-8")

    first = discover_inputs([tmp_path])
    second = discover_inputs([tmp_path])

    assert [source.path.name for source in first] == ["a.flac", "b.wav"]
    assert [source.id for source in first] == [source.id for source in second]
    assert len({source.id for source in first}) == 2


def test_directory_inputs_reject_empty_discovery_and_nested_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no audio sources"):
        discover_inputs([tmp_path])

    with pytest.raises(ValueError, match="must not be inside"):
        validate_output_location([tmp_path], tmp_path / "prepared")


def test_pipeline_exports_flac_manifests_and_resumes_verified_clips(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "prepared"
    make_audio(source)
    config, digest = load_config(CONFIG_PATH)
    factory_calls = 0

    def factory() -> FakeVad:
        nonlocal factory_calls
        factory_calls += 1
        return FakeVad([SpeechInterval(0.5, 18.0), SpeechInterval(18.7, 40.0)])

    sources = [SourceRecord("recording-1", source, sha256_file(source), {"kind": "fixture"})]
    first = run_pipeline(sources, output, config, digest, detector_factory=factory)
    second = run_pipeline(sources, output, config, digest, detector_factory=factory)

    assert first.sources_processed == 1
    assert first.clips_written == 2
    assert second.sources_reused == 1
    assert factory_calls == 1
    records = jsonl(output / "segments.jsonl")
    assert len(records) == 2
    assert records[0]["source_checksum"] == sha256_file(source)
    assert records[0]["boundary_type"] == "silence"
    assert records[0]["clip_checksum"].startswith("sha256:")
    assert jsonl(output / "sources.jsonl")[0]["source_metadata"] == {"kind": "fixture"}
    for record in records:
        info = sf.info(output / str(record["path"]))
        assert info.format == "FLAC"
        assert info.subtype == "PCM_16"
        assert info.channels == 1
        assert info.samplerate == 16000


def test_multiple_workers_process_sources_and_keep_manifests_deterministic(tmp_path: Path) -> None:
    sources_root = tmp_path / "sources"
    first_source = sources_root / "first.wav"
    second_source = sources_root / "second.wav"
    make_audio(first_source, duration=8.0)
    make_audio(second_source, duration=8.0)
    output = tmp_path / "prepared"
    config, digest = load_config(CONFIG_PATH)
    barrier = Barrier(2)

    class ConcurrentFakeVad(FakeVad):
        def detect(self, path: Path, settings: dict[str, object]) -> list[SpeechInterval]:
            barrier.wait(timeout=5)
            return super().detect(path, settings)

    sources = [
        SourceRecord("second", second_source, None, {}),
        SourceRecord("first", first_source, None, {}),
    ]
    audit = run_pipeline(
        sources,
        output,
        config,
        digest,
        detector_factory=lambda: ConcurrentFakeVad([SpeechInterval(0.5, 7.0)]),
        workers=2,
    )

    assert audit.sources_processed == 2
    assert audit.clips_written == 2
    assert [record["id"] for record in jsonl(output / "sources.jsonl")] == ["first", "second"]
    assert [record["id"] for record in jsonl(output / "segments.jsonl")] == [
        "first_000000",
        "second_000000",
    ]


def test_pipeline_rejects_invalid_worker_count(tmp_path: Path) -> None:
    config, digest = load_config(CONFIG_PATH)

    with pytest.raises(ValueError, match="workers must be >= 1"):
        run_pipeline([], tmp_path / "prepared", config, digest, workers=0)


def test_corrupt_clip_is_regenerated_instead_of_reused(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "prepared"
    make_audio(source, duration=8.0)
    config, digest = load_config(CONFIG_PATH)
    calls = 0

    def factory() -> FakeVad:
        nonlocal calls
        calls += 1
        return FakeVad([SpeechInterval(0.5, 7.0)])

    sources = [SourceRecord("one", source, None, {})]
    run_pipeline(sources, output, config, digest, detector_factory=factory)
    clip = next((output / "clips").glob("*.flac"))
    clip.write_bytes(b"corrupt")

    audit = run_pipeline(sources, output, config, digest, detector_factory=factory)

    assert audit.sources_processed == 1
    assert calls == 2
    assert sf.info(clip).format == "FLAC"


def test_pipeline_exports_configured_mp3_and_reuses_it(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "prepared"
    make_audio(source, duration=8.0)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["audio"] = {
        "bitrate_kbps": 48,
        "channels": 1,
        "format": "MP3",
        "sample_rate": 16000,
    }
    mp3_config_path = tmp_path / "segmentation-mp3.yaml"
    mp3_config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    config, digest = load_config(mp3_config_path)
    calls = 0

    def factory() -> FakeVad:
        nonlocal calls
        calls += 1
        return FakeVad([SpeechInterval(0.5, 7.0)])

    sources = [SourceRecord("one", source, None, {})]
    first = run_pipeline(sources, output, config, digest, detector_factory=factory)
    second = run_pipeline(sources, output, config, digest, detector_factory=factory)

    clip = next((output / "clips").glob("*.mp3"))
    info = sf.info(clip)
    assert first.sources_processed == 1
    assert second.sources_reused == 1
    assert calls == 1
    assert info.format == "MP3"
    assert info.subtype == "MPEG_LAYER_III"
    assert info.channels == 1
    assert info.samplerate == 16000
    assert jsonl(output / "segments.jsonl")[0]["path"].endswith(".mp3")


def test_config_change_requires_force_and_failed_force_preserves_output(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "prepared"
    make_audio(source, duration=8.0)
    config, digest = load_config(CONFIG_PATH)
    sources = [SourceRecord("one", source, None, {})]
    run_pipeline(
        sources,
        output,
        config,
        digest,
        detector_factory=lambda: FakeVad([SpeechInterval(0.5, 7.0)]),
    )
    original_run = (output / "run.json").read_text(encoding="utf-8")
    changed = json.loads(json.dumps(config))
    changed["vad"]["threshold"] = 0.6
    changed_digest = "sha256:" + "1" * 64

    with pytest.raises(ValueError, match="different configuration"):
        run_pipeline(sources, output, changed, changed_digest, detector_factory=lambda: FakeVad([]))

    missing = [SourceRecord("missing", tmp_path / "missing.wav", None, {})]
    with pytest.raises(RuntimeError, match="existing output was preserved"):
        run_pipeline(missing, output, changed, changed_digest, force=True, detector_factory=lambda: FakeVad([]))

    assert (output / "run.json").read_text(encoding="utf-8") == original_run


def test_checksum_mismatch_is_recorded_as_operational_failure(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "prepared"
    make_audio(source, duration=3.0)
    config, digest = load_config(CONFIG_PATH)

    audit = run_pipeline(
        [SourceRecord("one", source, "sha256:" + "0" * 64, {})],
        output,
        config,
        digest,
        detector_factory=lambda: FakeVad([]),
    )

    assert audit.operational_failures == 1
    assert jsonl(output / "rejected.jsonl")[0]["reason"] == "checksum_mismatch"


def test_packaged_silero_model_loads_without_network() -> None:
    detector = SileroVadDetector()

    assert detector.metadata["model"] == "silero_vad"
    assert detector.metadata["package_version"].startswith("6.2.")
