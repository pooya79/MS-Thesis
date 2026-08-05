from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from ml.speech_data.long_audio_asr_pipeline.segment_audio import sha256_file, write_jsonl_atomic
from ml.speech_data.long_audio_asr_pipeline.transcribe_segments import (
    PENDING_SNAPSHOT_NAME,
    SegmentTranscriber,
    load_config,
    run_transcription,
)


def make_audio(path: Path, duration: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(int(duration * 16000), dtype=np.float32)
    sf.write(path, samples, 16000, subtype="PCM_16")


def make_segmentation_root(root: Path, names: tuple[str, ...] = ("one.flac", "two.flac")) -> None:
    (root / "clips").mkdir(parents=True)
    (root / "run.json").write_text('{"config_digest":"segmentation"}\n', encoding="utf-8")
    records = []
    for index, name in enumerate(names):
        path = root / "clips" / name
        make_audio(path)
        records.append(
            {
                "id": f"clip-{index}",
                "source_id": "source",
                "path": f"clips/{name}",
                "clip_checksum": sha256_file(path),
            }
        )
    (root / "segments.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def publish_segment(root: Path, name: str, index: int) -> None:
    path = root / "clips" / name
    make_audio(path)
    records = read_jsonl(root / "segments.jsonl")
    records.append(
        {
            "id": f"clip-{index}",
            "source_id": "source",
            "path": f"clips/{name}",
            "clip_checksum": sha256_file(path),
        }
    )
    write_jsonl_atomic(root / "segments.jsonl", records)


def config(checkpoint: str = "/model", batch_size: int = 2) -> dict[str, object]:
    return {
        "model": {
            "checkpoint": checkpoint,
            "checkpoint_fingerprint": "sha256:model",
            "processor": "openai/whisper-medium",
            "language": "Persian",
            "task": "transcribe",
        },
        "inference": {
            "device": "cpu",
            "mixed_precision": False,
            "batch_size": batch_size,
            "generation_max_length": 225,
        },
    }


class FakeTranscriber:
    def __init__(self, outputs: dict[str, str], calls: list[list[str]]) -> None:
        self.outputs = outputs
        self.calls = calls

    def transcribe(self, paths: list[Path]) -> list[str]:
        self.calls.append([path.name for path in paths])
        return [self.outputs[path.name] for path in paths]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_transcribes_normalizes_and_writes_paths_relative_to_clips(tmp_path: Path) -> None:
    make_segmentation_root(tmp_path)
    calls: list[list[str]] = []

    audit = run_transcription(
        tmp_path,
        config(),
        "sha256:config",
        transcriber_factory=lambda _: FakeTranscriber(
            {"one.flac": " سلام، دنيا! ", "two.flac": "این یک آزمون است."}, calls
        ),
    )

    with (tmp_path / "transcription.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows == [
        {"path": "one.flac", "sentence": "سلام دنیا"},
        {"path": "two.flac", "sentence": "این یک آزمون است"},
    ]
    assert calls == [["one.flac", "two.flac"]]
    assert audit.clips_accepted == 2
    accepted = read_jsonl(tmp_path / "transcriptions.jsonl")
    assert accepted[0]["raw_transcript"] == "سلام، دنيا!"
    assert accepted[0]["normalized_transcript"] == "سلام دنیا"
    assert accepted[0]["generation"] == {
        "do_sample": False,
        "generation_max_length": 225,
        "num_beams": 1,
    }


def test_normalization_rejection_is_audited_and_not_an_operational_failure(tmp_path: Path) -> None:
    make_segmentation_root(tmp_path, ("one.flac",))

    audit = run_transcription(
        tmp_path,
        config(),
        "sha256:config",
        transcriber_factory=lambda _: FakeTranscriber({"one.flac": "hello سلام"}, []),
    )

    rows = (tmp_path / "transcription.tsv").read_text(encoding="utf-8").splitlines()
    rejected = read_jsonl(tmp_path / "transcription_rejected.jsonl")
    assert rows == ["path\tsentence"]
    assert rejected[0]["reason"] == "normalization_rejected"
    assert rejected[0]["raw_transcript"] == "hello سلام"
    assert audit.operational_failures == 0


def test_matching_run_reuses_transcript_without_loading_model(tmp_path: Path) -> None:
    make_segmentation_root(tmp_path, ("one.flac",))
    first_calls: list[list[str]] = []
    run_transcription(
        tmp_path,
        config(),
        "sha256:config",
        transcriber_factory=lambda _: FakeTranscriber({"one.flac": "سلام"}, first_calls),
    )

    def must_not_load(_: dict[str, object]) -> SegmentTranscriber:
        raise AssertionError("matching result should have been reused")

    audit = run_transcription(
        tmp_path,
        config(),
        "sha256:config",
        transcriber_factory=must_not_load,
    )

    assert first_calls == [["one.flac"]]
    assert audit.clips_reused == 1
    assert audit.clips_processed == 0
    assert read_jsonl(tmp_path / PENDING_SNAPSHOT_NAME) == []


def test_run_uses_fixed_snapshot_and_next_run_transcribes_newly_published_segment(
    tmp_path: Path,
) -> None:
    make_segmentation_root(tmp_path, ("one.flac",))
    first_calls: list[list[str]] = []

    class PublishingTranscriber:
        def transcribe(self, paths: list[Path]) -> list[str]:
            first_calls.append([path.name for path in paths])
            publish_segment(tmp_path, "two.flac", 1)
            return ["سلام"]

    first_audit = run_transcription(
        tmp_path,
        config(),
        "sha256:config",
        transcriber_factory=lambda _: PublishingTranscriber(),
    )

    assert first_calls == [["one.flac"]]
    assert first_audit.clips_total == 1
    assert [record["id"] for record in read_jsonl(tmp_path / PENDING_SNAPSHOT_NAME)] == ["clip-0"]
    assert [record["id"] for record in read_jsonl(tmp_path / "transcriptions.jsonl")] == ["clip-0"]

    second_calls: list[list[str]] = []
    second_audit = run_transcription(
        tmp_path,
        config(),
        "sha256:config",
        transcriber_factory=lambda _: FakeTranscriber({"two.flac": "درود"}, second_calls),
    )

    assert second_calls == [["two.flac"]]
    assert second_audit.clips_total == 2
    assert second_audit.clips_reused == 1
    assert second_audit.clips_processed == 1
    assert [record["id"] for record in read_jsonl(tmp_path / PENDING_SNAPSHOT_NAME)] == ["clip-1"]
    assert [record["id"] for record in read_jsonl(tmp_path / "transcriptions.jsonl")] == [
        "clip-0",
        "clip-1",
    ]


def test_completed_batches_survive_interruption_and_are_reused_on_restart(tmp_path: Path) -> None:
    make_segmentation_root(tmp_path)

    class StopRun(BaseException):
        pass

    class InterruptingTranscriber:
        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, paths: list[Path]) -> list[str]:
            self.calls += 1
            if self.calls == 2:
                raise StopRun
            return ["سلام"]

    with pytest.raises(StopRun):
        run_transcription(
            tmp_path,
            config(batch_size=1),
            "sha256:config",
            transcriber_factory=lambda _: InterruptingTranscriber(),
        )

    assert [record["id"] for record in read_jsonl(tmp_path / "transcriptions.jsonl")] == ["clip-0"]
    assert [record["id"] for record in read_jsonl(tmp_path / PENDING_SNAPSHOT_NAME)] == [
        "clip-0",
        "clip-1",
    ]

    restart_calls: list[list[str]] = []
    audit = run_transcription(
        tmp_path,
        config(batch_size=1),
        "sha256:config",
        transcriber_factory=lambda _: FakeTranscriber({"two.flac": "درود"}, restart_calls),
    )

    assert restart_calls == [["two.flac"]]
    assert audit.clips_reused == 1
    assert audit.clips_processed == 1
    assert [record["id"] for record in read_jsonl(tmp_path / "transcriptions.jsonl")] == [
        "clip-0",
        "clip-1",
    ]


def test_inference_failure_is_recorded_after_individual_retry(tmp_path: Path) -> None:
    make_segmentation_root(tmp_path, ("good.flac", "bad.flac"))

    class PartlyFailingTranscriber:
        def transcribe(self, paths: list[Path]) -> list[str]:
            if len(paths) > 1 or paths[0].name == "bad.flac":
                raise RuntimeError("fixture failure")
            return ["سلام"]

    audit = run_transcription(
        tmp_path,
        config(),
        "sha256:config",
        transcriber_factory=lambda _: PartlyFailingTranscriber(),
    )

    assert audit.clips_accepted == 1
    assert audit.operational_failures == 1
    assert read_jsonl(tmp_path / "transcription_rejected.jsonl")[0]["reason"] == "inference_failed"
    assert [record["id"] for record in read_jsonl(tmp_path / PENDING_SNAPSHOT_NAME)] == [
        "clip-0",
        "clip-1",
    ]

    retry_calls: list[list[str]] = []
    retried = run_transcription(
        tmp_path,
        config(),
        "sha256:config",
        transcriber_factory=lambda _: FakeTranscriber({"bad.flac": "درود"}, retry_calls),
    )

    assert retry_calls == [["bad.flac"]]
    assert retried.clips_reused == 1
    assert retried.clips_accepted == 2
    assert retried.operational_failures == 0
    assert read_jsonl(tmp_path / "transcription_rejected.jsonl") == []
    assert [record["id"] for record in read_jsonl(tmp_path / PENDING_SNAPSHOT_NAME)] == ["clip-1"]


def test_changed_config_requires_force_and_failed_force_preserves_artifacts(tmp_path: Path) -> None:
    make_segmentation_root(tmp_path, ("one.flac",))
    run_transcription(
        tmp_path,
        config(),
        "sha256:one",
        transcriber_factory=lambda _: FakeTranscriber({"one.flac": "سلام"}, []),
    )
    original = (tmp_path / "transcription.tsv").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="different configuration"):
        run_transcription(tmp_path, config(), "sha256:two")

    class FailingTranscriber:
        def transcribe(self, paths: list[Path]) -> list[str]:
            raise RuntimeError("fixture failure")

    with pytest.raises(RuntimeError, match="existing artifacts were preserved"):
        run_transcription(
            tmp_path,
            config(),
            "sha256:two",
            force=True,
            transcriber_factory=lambda _: FailingTranscriber(),
        )

    assert (tmp_path / "transcription.tsv").read_text(encoding="utf-8") == original


def test_load_config_resolves_checkpoint_and_validates_inference_settings(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"whisper"}\n', encoding="utf-8")
    config_path = tmp_path / "transcription.yaml"
    config_path.write_text(
        yaml.safe_dump({"model": {"checkpoint": "model"}, "inference": {"batch_size": 2}}),
        encoding="utf-8",
    )

    loaded, digest = load_config(config_path)

    assert loaded["model"]["checkpoint"] == str(checkpoint.resolve())
    assert loaded["model"]["processor"] == "openai/whisper-medium"
    assert loaded["inference"]["batch_size"] == 2
    assert str(loaded["model"]["checkpoint_fingerprint"]).startswith("sha256:")
    assert digest.startswith("sha256:")

    config_path.write_text(
        yaml.safe_dump({"model": {"checkpoint": "model"}, "inference": {"batch_size": 0}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="batch_size"):
        load_config(config_path)
