from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

import ml.speech_data.long_audio_asr_pipeline.segment_audio as segment_audio_module
from ml.speech_data.long_audio_asr_pipeline.segment_audio import (
    SileroVadDetector,
    SourceRecord,
    discover_inputs,
    execution_settings,
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


class BatchFakeVad(FakeVad):
    def __init__(self, intervals: list[SpeechInterval]) -> None:
        super().__init__(intervals)
        self.batches: list[list[bool]] = []

    def detect_many(
        self,
        paths: list[Path | None],
        settings: dict[str, object],
    ) -> list[list[SpeechInterval]]:
        self.batches.append([path is not None for path in paths])
        return [self.detect(path, settings) if path is not None else [] for path in paths]


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


def test_execution_settings_are_validated_and_loaded(tmp_path: Path) -> None:
    config, _ = load_config(CONFIG_PATH)

    assert execution_settings(config).vad_engine == "pytorch"
    assert execution_settings(config).vad_device == "cpu"
    assert execution_settings(config).vad_batch_size == 1
    assert execution_settings(config).torch_threads == 1

    legacy = json.loads(json.dumps(config))
    legacy.pop("execution")
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(legacy), encoding="utf-8")
    loaded_legacy, _ = load_config(path)
    assert execution_settings(loaded_legacy).vad_batch_size == 1
    assert execution_settings(loaded_legacy).vad_device == "cpu"
    assert execution_settings(loaded_legacy).torch_threads is None

    for field, value in (("vad_batch_size", 0), ("torch_threads", True)):
        invalid = json.loads(json.dumps(config))
        invalid["execution"][field] = value
        path = tmp_path / f"invalid-{field}.yaml"
        path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            load_config(path)

    unknown = json.loads(json.dumps(config))
    unknown["execution"]["surprise"] = 1
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml.safe_dump(unknown), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown execution"):
        load_config(path)

    unsupported = json.loads(json.dumps(config))
    unsupported["execution"]["vad_engine"] = "onnx"
    path = tmp_path / "unsupported-engine.yaml"
    path.write_text(yaml.safe_dump(unsupported), encoding="utf-8")
    with pytest.raises(ValueError, match="vad_engine must be pytorch"):
        load_config(path)

    unsupported = json.loads(json.dumps(config))
    unsupported["execution"]["vad_device"] = "mps"
    path = tmp_path / "unsupported-device.yaml"
    path.write_text(yaml.safe_dump(unsupported), encoding="utf-8")
    with pytest.raises(ValueError, match="vad_device must be cpu or cuda"):
        load_config(path)


def test_silero_frame_reader_uses_blocks_and_pads_only_the_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_samples = 512
    samples = np.linspace(-0.5, 0.5, frame_samples * 5 + 100, dtype=np.float32)
    path = tmp_path / "frames.wav"
    sf.write(path, samples, 16000, subtype="FLOAT")
    real_sound_file = sf.SoundFile
    read_calls = 0

    class CountingSoundFile:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.inner = real_sound_file(*args, **kwargs)

        def __enter__(self) -> CountingSoundFile:
            self.inner.__enter__()
            return self

        def __exit__(self, *args: object) -> None:
            self.inner.__exit__(*args)

        def read(self, *args: object, **kwargs: object) -> np.ndarray:
            nonlocal read_calls
            read_calls += 1
            return self.inner.read(*args, **kwargs)

    monkeypatch.setattr(SileroVadDetector, "_FRAMES_PER_READ", 2)
    monkeypatch.setattr(segment_audio_module.sf, "SoundFile", CountingSoundFile)

    frames = list(SileroVadDetector._frames(path, frame_samples))

    assert read_calls == 3
    assert len(frames) == 6
    np.testing.assert_array_equal(np.concatenate(frames)[: len(samples)], samples)
    np.testing.assert_array_equal(
        frames[-1][100:],
        np.zeros(frame_samples - 100, dtype=np.float32),
    )


def test_single_source_silero_inference_keeps_frame_shape_and_inference_mode(
    tmp_path: Path,
) -> None:
    frame_samples = 512
    frame_values = [0.1, 0.8, 0.9, 0.1, 0.8]
    samples = np.concatenate(
        [np.full(frame_samples, value, dtype=np.float32) for value in frame_values]
        + [np.full(100, 0.8, dtype=np.float32)]
    )
    path = tmp_path / "probabilities.wav"
    sf.write(path, samples, 16000, subtype="FLOAT")

    class ProbabilityModel:
        def __init__(self) -> None:
            self.shapes: list[tuple[int, ...]] = []
            self.inference_modes: list[bool] = []
            self.reset_calls = 0
            self.device: torch.device | None = None

        def to(self, device: torch.device) -> ProbabilityModel:
            self.device = device
            return self

        def eval(self) -> None:
            return None

        def reset_states(self) -> None:
            self.reset_calls += 1

        def __call__(self, value: torch.Tensor, sample_rate: int) -> torch.Tensor:
            assert sample_rate == 16000
            self.shapes.append(tuple(value.shape))
            self.inference_modes.append(torch.is_inference_mode_enabled())
            return value.mean().reshape(1)

    model = ProbabilityModel()
    detector = SileroVadDetector(model=model)
    settings = {
        "sample_rate": 16000,
        "frame_samples": frame_samples,
        "threshold": 0.5,
        "minimum_speech_seconds": 0.02,
        "merge_silence_seconds": 0.01,
    }

    intervals = detector.detect(path, settings)

    assert model.reset_calls == 1
    assert model.device == torch.device("cpu")
    assert model.shapes == [(frame_samples,)] * 6
    assert all(model.inference_modes)
    assert [(item.start_sec, item.end_sec) for item in intervals] == [
        (0.032, 0.096),
        (0.128, 0.16),
    ]


def test_device_resident_silero_reads_once_and_preserves_probabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_samples = 512
    frame_values = [0.1, 0.8, 0.9, 0.1, 0.8]
    samples = np.concatenate(
        [np.full(frame_samples, value, dtype=np.float32) for value in frame_values]
        + [np.full(100, 0.8, dtype=np.float32)]
    )
    path = tmp_path / "resident.wav"
    sf.write(path, samples, 16000, subtype="FLOAT")
    real_sound_file = sf.SoundFile
    read_calls = 0

    class CountingSoundFile:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.inner = real_sound_file(*args, **kwargs)

        def __enter__(self) -> CountingSoundFile:
            self.inner.__enter__()
            return self

        def __exit__(self, *args: object) -> None:
            self.inner.__exit__(*args)

        @property
        def frames(self) -> int:
            return self.inner.frames

        @property
        def samplerate(self) -> int:
            return self.inner.samplerate

        def read(self, *args: object, **kwargs: object) -> np.ndarray:
            nonlocal read_calls
            read_calls += 1
            return self.inner.read(*args, **kwargs)

    class ProbabilityModel:
        def __init__(self) -> None:
            self.reset_calls = 0

        def to(self, device: torch.device) -> ProbabilityModel:
            return self

        def eval(self) -> None:
            return None

        def reset_states(self) -> None:
            self.reset_calls += 1

        def __call__(self, value: torch.Tensor, sample_rate: int) -> torch.Tensor:
            assert torch.is_inference_mode_enabled()
            assert sample_rate == 16000
            return value.mean(dim=-1).reshape(-1)

    monkeypatch.setattr(segment_audio_module.sf, "SoundFile", CountingSoundFile)
    model = ProbabilityModel()
    detector = SileroVadDetector(model=model)
    settings = {
        "sample_rate": 16000,
        "frame_samples": frame_samples,
        "threshold": 0.5,
        "minimum_speech_seconds": 0.02,
        "merge_silence_seconds": 0.01,
    }

    intervals = detector._detect_device_resident([path], settings)

    assert read_calls == 1
    assert model.reset_calls == 1
    assert [(item.start_sec, item.end_sec) for item in intervals[0]] == [
        (0.032, 0.096),
        (0.128, 0.16),
    ]


def test_cuda_audio_length_limit_is_five_hours() -> None:
    sample_rate = 16000
    maximum_frames = 5 * 60 * 60 * sample_rate

    segment_audio_module.validate_cuda_audio_length(
        maximum_frames,
        sample_rate,
        Path("five-hours.wav"),
    )
    with pytest.raises(ValueError, match=r"at most 18000 seconds \(5 hours\)"):
        segment_audio_module.validate_cuda_audio_length(
            maximum_frames + 1,
            sample_rate,
            Path("too-long.wav"),
        )


def test_pipeline_caches_ffmpeg_version_and_restores_torch_threads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = []
    for name in ("one", "two"):
        path = tmp_path / f"{name}.wav"
        make_audio(path, duration=4.0)
        sources.append(SourceRecord(name, path, None, {}))
    config, digest = load_config(CONFIG_PATH)
    version_calls = 0
    factory_threads: list[int] = []
    original_threads = torch.get_num_threads()

    def fake_ffmpeg_version() -> str:
        nonlocal version_calls
        version_calls += 1
        return "ffmpeg fixture"

    def factory() -> FakeVad:
        factory_threads.append(torch.get_num_threads())
        return FakeVad([SpeechInterval(0.25, 3.5)])

    monkeypatch.setattr(segment_audio_module, "ffmpeg_version", fake_ffmpeg_version)

    run_pipeline(
        sources,
        tmp_path / "output",
        config,
        digest,
        detector_factory=factory,
        workers=2,
    )

    assert version_calls == 1
    assert factory_threads == [1, 1]
    assert torch.get_num_threads() == original_threads


def test_batched_failure_retries_and_rejects_only_the_bad_source(tmp_path: Path) -> None:
    good = tmp_path / "good.wav"
    bad = tmp_path / "bad.wav"
    make_audio(good, duration=4.0)
    make_audio(bad, duration=4.0)
    config, _ = load_config(CONFIG_PATH)
    config["execution"]["vad_batch_size"] = 2

    class FailingBatchVad(FakeVad):
        def detect_many(
            self,
            paths: list[Path | None],
            settings: dict[str, object],
        ) -> list[list[SpeechInterval]]:
            raise RuntimeError("batch fixture failure")

        def detect(self, path: Path, settings: dict[str, object]) -> list[SpeechInterval]:
            if "bad" in path.name:
                raise RuntimeError("bad source fixture")
            return super().detect(path, settings)

    audit = run_pipeline(
        [SourceRecord("good", good, None, {}), SourceRecord("bad", bad, None, {})],
        tmp_path / "batched",
        config,
        "sha256:" + "2" * 64,
        detector_factory=lambda: FailingBatchVad([SpeechInterval(0.25, 3.5)]),
    )

    assert audit.sources_processed == 1
    assert audit.operational_failures == 1
    assert jsonl(tmp_path / "batched" / "rejected.jsonl")[0]["source_id"] == "bad"


def test_batch_orchestration_preserves_scientific_outputs_with_equivalent_detector(
    tmp_path: Path,
) -> None:
    sources = []
    for name, duration in (("one", 4.0), ("two", 5.0)):
        path = tmp_path / f"{name}.wav"
        make_audio(path, duration=duration)
        sources.append(SourceRecord(name, path, None, {}))
    config, _ = load_config(CONFIG_PATH)
    digest = "sha256:" + "3" * 64
    outputs = []

    for batch_size in (1, 2):
        selected = json.loads(json.dumps(config))
        selected["execution"]["vad_batch_size"] = batch_size
        output = tmp_path / f"batch-{batch_size}"
        run_pipeline(
            sources,
            output,
            selected,
            digest,
            detector_factory=lambda: BatchFakeVad([SpeechInterval(0.25, 3.5)]),
        )
        outputs.append(output)

    for manifest in ("sources.jsonl", "segments.jsonl", "vad_intervals.jsonl"):
        assert (outputs[0] / manifest).read_bytes() == (outputs[1] / manifest).read_bytes()
    assert {
        path.name: path.read_bytes() for path in (outputs[0] / "clips").iterdir()
    } == {
        path.name: path.read_bytes() for path in (outputs[1] / "clips").iterdir()
    }


def test_resume_keeps_empty_slots_in_a_stable_source_cohort(tmp_path: Path) -> None:
    sources = []
    for name in ("one", "two"):
        path = tmp_path / f"{name}.wav"
        make_audio(path, duration=4.0)
        sources.append(SourceRecord(name, path, None, {}))
    config, digest = load_config(CONFIG_PATH)
    config["execution"]["vad_batch_size"] = 2
    output = tmp_path / "stable-cohort"
    run_pipeline(
        sources,
        output,
        config,
        digest,
        detector_factory=lambda: BatchFakeVad([SpeechInterval(0.25, 3.5)]),
    )
    (output / "clips" / "two_000000.flac").write_bytes(b"corrupt")
    detector = BatchFakeVad([SpeechInterval(0.25, 3.5)])

    audit = run_pipeline(
        sources,
        output,
        config,
        digest,
        detector_factory=lambda: detector,
    )

    assert audit.sources_reused == 1
    assert audit.sources_processed == 1
    assert detector.batches == [[False, True]]


def test_pipeline_rejects_unavailable_cuda_before_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, digest = load_config(CONFIG_PATH)
    config["execution"]["vad_device"] = "cuda"
    output = tmp_path / "cuda-output"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA is not available"):
        run_pipeline([], output, config, digest)

    assert not output.exists()


def test_pipeline_rejects_cuda_audio_over_five_hour_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.wav"
    make_audio(source, duration=1.0)
    config, digest = load_config(CONFIG_PATH)
    config["execution"]["vad_device"] = "cuda"
    output = tmp_path / "cuda-too-long"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(segment_audio_module, "CUDA_MAX_AUDIO_SECONDS", 0.5)
    monkeypatch.setattr(
        segment_audio_module,
        "decode_audio",
        lambda source_path, destination: sf.write(
            destination,
            sf.read(source_path, dtype="float32")[0],
            16000,
            subtype="PCM_16",
        ),
    )
    monkeypatch.setattr(segment_audio_module, "ffmpeg_version", lambda: "ffmpeg fixture")

    audit = run_pipeline(
        [SourceRecord("too-long", source, None, {})],
        output,
        config,
        digest,
        detector_factory=lambda: pytest.fail("VAD must not run for oversized audio"),
    )

    assert audit.operational_failures == 1
    rejection = jsonl(output / "rejected.jsonl")[0]
    assert rejection["reason"] == "invalid_audio"
    assert "CUDA VAD supports at most" in str(rejection["detail"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_packaged_silero_model_runs_on_cuda(tmp_path: Path) -> None:
    path = tmp_path / "cuda.wav"
    make_audio(path, duration=1.0)
    config, _ = load_config(CONFIG_PATH)
    detector = SileroVadDetector(device="cuda")

    intervals = detector.detect(path, config["vad"])

    assert isinstance(intervals, list)
    assert detector.metadata["device"] == "cuda"


def test_packaged_silero_model_loads_without_network() -> None:
    detector = SileroVadDetector()

    assert detector.metadata["model"] == "silero_vad"
    assert detector.metadata["device"] == "cpu"
    assert detector.metadata["package_version"].startswith("6.2.")
