from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs" / "huggingface" / "PersianAudiobook" / "README.md"
PIPELINE_IMAGE = ROOT / "Thesis" / "figs" / "long-audio-data-creation-pipeline.png"


def _front_matter(text: str) -> dict[str, object]:
    assert text.startswith("---\n")
    _, yaml_text, _ = text.split("---", 2)
    loaded = yaml.safe_load(yaml_text)
    assert isinstance(loaded, dict)
    return loaded


def test_card_has_valid_config_and_expected_tasks() -> None:
    text = CARD.read_text(encoding="utf-8")
    metadata = _front_matter(text)

    assert metadata["pretty_name"] == "PersianAudiobook"
    assert metadata["language"] == ["fa"]
    assert metadata["task_categories"] == [
        "automatic-speech-recognition",
        "text-to-speech",
    ]
    assert metadata["configs"] == [
        {
            "config_name": "refined-subset",
            "data_files": [
                {"split": "train", "path": "refined-subset/train-*.parquet"}
            ],
        }
    ]


def test_card_mentions_source_once_and_links_existing_pipeline_image() -> None:
    text = CARD.read_text(encoding="utf-8")

    assert text.lower().count("iranseda") == 1
    assert "./assets/long-audio-data-creation-pipeline.png" in text
    assert PIPELINE_IMAGE.is_file()


def test_card_configures_manual_research_gate_and_forbids_redistribution() -> None:
    text = CARD.read_text(encoding="utf-8")
    metadata = _front_matter(text)

    assert metadata["license"] == "other"
    assert metadata["gated"] is True
    assert metadata["extra_gated_fields"] == {
        "Briefly describe your intended use and research purpose": "text"
    }
    assert not any(
        key.startswith("extra_gated_") and key != "extra_gated_fields"
        for key in metadata
    )
    assert "This is a restricted-access research dataset" in text
    assert "username, email address, and brief intended use" in text
    assert "manually approves access" in text
    assert "No license to redistribute the underlying audio is granted" in text
    assert "Do not bypass access controls" in text
