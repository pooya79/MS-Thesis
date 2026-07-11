from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "docs" / "degradation-pipeline-architecture"


class _AssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stylesheets: list[str] = []
        self.scripts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "link" and values.get("rel") == "stylesheet" and values.get("href"):
            self.stylesheets.append(values["href"])
        if tag == "script" and values.get("src"):
            self.scripts.append(values["src"])


def _read(name: str) -> str:
    return (FIGURE_DIR / name).read_text(encoding="utf-8")


def test_standalone_figure_assets_exist() -> None:
    for name in ("index.html", "styles.css", "diagram.js", "README.md"):
        path = FIGURE_DIR / name
        assert path.is_file()
        assert path.stat().st_size > 0


def test_html_references_local_assets_and_pinned_d3() -> None:
    parser = _AssetParser()
    parser.feed(_read("index.html"))
    assert parser.stylesheets == ["styles.css"]
    assert "diagram.js" in parser.scripts
    assert "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js" in parser.scripts
    assert all("@latest" not in source for source in parser.scripts)


def test_figure_contains_pipeline_and_codec_branches() -> None:
    source = _read("index.html") + _read("diagram.js")
    for label in (
        "Clean audio", "Mono +", "Profile", "telephone_noisy", "Optional noise", "Gain + optional",
        "Select codec", "Channel path", "Pre-codec", "Opus packets",
        "Burst packet loss", "Decode with PLC", "ffmpeg round-trip",
        "Decoded-waveform", "Pass-through", "Delay estimation",
        "Clean target", "Shared pair", "WAV pairs +", "JSONL metadata",
        "Deterministic variants", "Recorded metadata",
    ):
        assert label in source, f"figure is missing required label: {label}"


def test_accessibility_error_print_and_export_behavior() -> None:
    html, css, script = _read("index.html"), _read("styles.css"), _read("diagram.js")
    assert '<noscript>' in html
    assert 'role="alert"' in html
    assert '<figcaption id="figure-caption">' in html
    assert 'aria-label="Figure export controls"' in html
    assert '.attr("role","img")' in script
    assert '.append("title")' in script and '.append("desc")' in script
    assert "inlinePresentationStyles(source,clone)" in script
    assert 'downloadBlob(new Blob([xml]' in script
    assert '"degradation-pipeline-architecture.svg"' in script
    assert "const scale = 3" in script
    assert '"degradation-pipeline-architecture@3x.png"' in script
    assert "@media print" in css and ".controls" in css


def test_caption_links_implementation_configuration_and_documentation() -> None:
    html = _read("index.html")
    assert "ml/speech_data/generate_degraded_pairs.py" in html
    assert "configs/speech_enhancement/degradation.yaml" in html
    assert "docs/speech-degradation-pipeline.md" in html


def test_thesis_uses_three_x_export_on_a_landscape_figure_page() -> None:
    image_path = ROOT / "Thesis" / "figs" / "degradation-pipeline-architecture.png"
    assert image_path.is_file()
    with image_path.open("rb") as image:
        assert image.read(8) == b"\x89PNG\r\n\x1a\n"
        chunk_length = struct.unpack(">I", image.read(4))[0]
        assert image.read(4) == b"IHDR"
        width, height = struct.unpack(">II", image.read(chunk_length)[:8])
    assert (width, height) == (5400, 3120)

    chapter = (ROOT / "Thesis" / "chapters" / "work.tex").read_text(encoding="utf-8")
    common = (ROOT / "Thesis" / "styles" / "common.tex").read_text(encoding="utf-8")
    assert r"\usepackage{pdflscape}" in common
    assert r"\begin{landscape}" in chapter
    assert r"figs/degradation-pipeline-architecture.png" in chapter
    assert r"\برچسب{شکل:معماری-خط-لوله-تخریب}" in chapter
