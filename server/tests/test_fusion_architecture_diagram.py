from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "docs" / "fusion-model-architecture"


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
    assert FIGURE_DIR.is_dir()
    for name in ("index.html", "styles.css", "diagram.js"):
        path = FIGURE_DIR / name
        assert path.is_file(), f"missing standalone figure asset: {path}"
        assert path.stat().st_size > 0


def test_html_references_local_assets_and_version_pinned_cdns() -> None:
    parser = _AssetParser()
    parser.feed(_read("index.html"))

    assert parser.stylesheets == ["styles.css"]
    assert "diagram.js" in parser.scripts
    assert "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js" in parser.scripts
    assert (
        "https://cdn.jsdelivr.net/npm/@mathjax/mathjax-tex-font@4.1.2/"
        "tex-mml-svg-mathjax-tex.js"
    ) in parser.scripts
    assert all("@latest" not in source for source in parser.scripts)


def test_figure_contains_required_architecture_and_training_labels() -> None:
    source = _read("index.html") + _read("diagram.js")
    required_labels = (
        "Noisy log-Mel",
        "Residual U-Net",
        "depth 3 · base channels 48",
        "Transformer: 2L · 4H · d256",
        "parallel noisy path",
        "Whisper encoder",
        "ONE SHARED WEIGHT SET",
        "Cross-attention fusion · 3 layers · 12 heads · FFN ×4",
        "Layer 1",
        "Layer 2",
        "Layer 3",
        "N ← Attn(N,E)",
        "E ← Attn(E,N)",
        "Sigmoid gate",
        "learned per time",
        "Whisper",
        "Persian tokens",
        "Enhancement loss",
        "Autoregressive ASR loss",
        "Warm-up feature loss",
        "ENABLED · feature_match_weight = 0.5",
        "Fusion / joint objective",
        "Stage 0",
        "Stage 1",
        "Stage 2",
        "current config: λ = 0.5 · feature weight = 0.5",
        "current config: λ = 0.3",
        "current config: λ = 0.1",
        "report/whisper-fusion-v2/fusion_train_v4.yaml",
    )
    for label in required_labels:
        assert label in source, f"figure is missing required label: {label}"


def test_equations_and_export_behavior_are_declared() -> None:
    html = _read("index.html")
    script = _read("diagram.js")

    assert 'svg: {fontCache: "none"}' in html
    assert "loadDynamicFiles" in script
    assert "tex2svg(item.tex" in script
    assert "h_f=g\\\\odot h'_e+(1-g)\\\\odot h'_n" in script
    assert "L_{\\\\mathrm{enh}}=\\\\lVert E(x_n)-x_c\\\\rVert_1" in script
    assert "L_{\\\\mathrm{ASR}}=-\\\\sum_t" in script
    assert "L=L_{\\\\mathrm{ASR}}+\\\\lambda L_{\\\\mathrm{enh}}" in script
    assert "inlinePresentationStyles(source, clone)" in script
    assert 'downloadBlob(new Blob([xml]' in script
    assert '"fusion-model-architecture.svg"' in script
    assert "const scale = 3" in script
    assert '"fusion-model-architecture@3x.png"' in script


def test_accessibility_error_state_and_print_rules_are_present() -> None:
    html = _read("index.html")
    css = _read("styles.css")
    script = _read("diagram.js")

    assert '<noscript>' in html
    assert 'role="alert"' in html
    assert '<figcaption id="figure-caption">' in html
    assert 'aria-label="Figure export controls"' in html
    assert '.attr("role", "img")' in script
    assert '.append("title")' in script
    assert '.append("desc")' in script
    assert "@media print" in css
    assert ".controls" in css
