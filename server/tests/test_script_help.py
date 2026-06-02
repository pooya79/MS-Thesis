from __future__ import annotations

import importlib

import pytest


SCRIPT_MODULES = [
    "ml.speech_data.scripts.download_common_voice_fa",
    "ml.speech_data.scripts.download_degradation_assets",
    "ml.speech_data.scripts.download_fleurs_persian",
    "ml.speech_data.scripts.download_persian_eval_sets",
    "ml.speech_data.scripts.normalize_tsv_dataset",
    "ml.speech_data.scripts.prepare_common_voice_25",
    "ml.speech_data.scripts.prepare_degradation_assets",
    "ml.speech_data.scripts.prepare_fleurs_persian",
    "ml.speech_data.scripts.generate_random_degraded_clip",
    "ml.speech_data.generate_degraded_dataset",
    "ml.speech_data.generate_degraded_pairs",
    "ml.speech_data.inspect_manifest",
    "ml.asr.train_whisper_small",
    "ml.asr.eval_whisper_small",
]


@pytest.mark.parametrize("module_name", SCRIPT_MODULES)
def test_script_entrypoint_prints_help(module_name: str, capsys: pytest.CaptureFixture[str]) -> None:
    module = importlib.import_module(module_name)

    with pytest.raises(SystemExit) as exc_info:
        module.main(["--help"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "usage:" in captured.out
    assert "--help" in captured.out


def test_generate_degraded_dataset_help_documents_workers(capsys: pytest.CaptureFixture[str]) -> None:
    module = importlib.import_module("ml.speech_data.generate_degraded_dataset")

    with pytest.raises(SystemExit) as exc_info:
        module.main(["--help"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "--workers" in captured.out
