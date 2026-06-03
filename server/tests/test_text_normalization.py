from __future__ import annotations

from ml.speech_data.text_normalization import normalize_persian_asr_text


def test_normalize_persian_asr_text_matches_nvidia_card_rules() -> None:
    assert normalize_persian_asr_text("خب ، تو چیكار می كنی؟") == "خب تو چیکار می کنی"
    assert normalize_persian_asr_text("أۀك ي ى ﯽ ﻮ ے ﺒ ﻢ ٬") == "اهک ی ی ی و ی ب م"
    assert normalize_persian_asr_text("سلام! «دوست»؛") == "سلام دوست"
    assert normalize_persian_asr_text("سلام [دوست] / امروز") == "سلام دوست امروز"
    assert normalize_persian_asr_text("سلام #خوانده_نمیشود دنیا") == "سلام دنیا"
    assert normalize_persian_asr_text("  سلام    دنیا  ") == "سلام دنیا"
    assert normalize_persian_asr_text("hello سلام") is None
