# Persian thesis figures

The six `Thesis/figs/*-fa.html` files are the editable, self-contained sources for
the diagrams included in `Thesis/chapters/work.tex`. The original English HTML
and PNG sources remain unchanged. The Persian sources follow
`docs/diagram-style-guide.md`, use a 1120 × 792 canvas, and embed the repository's
Vazirmatn Regular font (license: `Thesis/presentation/assets/fonts/vazirmatn/OFL.txt`).
No network or installed Persian font is required to view or export them.

The three overview figures cover degradation, fusion, and long audio data
creation. The new detail figures cover the residual U-Net, both cross-attention
directions, and the time-feature gate. The fusion overview includes the three
training objectives alongside the architecture. All descriptive labels are Persian; model names,
tools and mathematical notation remain in English where appropriate. Added text
and figure labels contain no hamza characters.

The Persian degradation figure follows the English paper figure's hierarchy,
mirrored for right-to-left reading: three numbered preparation steps, a shared
codec/network band, three codec paths, alignment, and the durable training pair.
Dashed insets mark conditional waveform dropout; a separate dashed route carries
the clean reference into alignment. `Opus`, `ffmpeg`, `PLC`, and output format
names retain their Latin spelling. The HTML source and its vector PDF are updated
together; the original English figure remains the layout reference.

The Persian IranSeda figure follows the three phase columns in
`docs/long-audio-data-creation-pipeline/index.html`, mirrored for right-to-left
reading. The seven numbered stages retain source validation, speech segmentation,
Whisper-Medium transcription, Persian normalization, limited Qwen refinement,
quality filtering, and Whisper-small training. The final training stage and its
incoming data connector carry the accent, matching the English figure. Model
names and `FLAC` use explicit left-to-right text runs inside the Persian labels.

The Persian fusion overview follows
`Thesis/figs/fusion-model-architecture-left-to-right-cross-attention.html`,
mirrored and fitted to the same 1120 × 792 paper canvas. It preserves the two
acoustic paths, a residual U-Net thumbnail, two passes through one shared encoder,
and the fusion panel containing bidirectional cross-attention followed by a
time-feature gate. The lower strip shows the three one-epoch training stages and
their objectives, checked against `fusion_train_compact.yaml` and the thesis text.
Formulas keep left-to-right ordering and typographic loss subscripts.

## Technical provenance

- `ml/enhancement/enhancer.py`: `_ConvBlock`, `ResidualUNetEnhancer`, size matching,
  concatenation skips, and zero-initialized residual output.
- `ml/fusion/model.py`: `_CrossAttnLayer`, `CrossAttentionFusion`, `_GatedCombine`;
  independent directions use pre-update states, followed by element-wise gating.
- `configs/speech_enhancement/fusion_train_compact.yaml`: depth 3, width 32, no
  temporal bottleneck, one fusion layer and four heads. The generic larger
  configuration is not the architecture described by these thesis figures.
- The existing thesis references `ronneberger2015unet` and
  `vaswani2017attention` describe the underlying U-Net and attention principles:
  <https://arxiv.org/abs/1505.04597> and <https://arxiv.org/abs/1706.03762>.

The U-Net detail shows `T` divisible by eight for readability. The prose explains
crop/zero-padding for other lengths. Dimensions inside U-Net omit batch size;
encoded time is `S`, distinct from log-Mel time `T`. The gate weights the
**refined** streams, so its mean is not an independent contribution attribution.

## Export and verify

Discover the installed bundled runtime with `load_workspace_dependencies`.
Use Node.js and the bundled `playwright` package with an installed Chromium.
For this workstation:

```bash
node Thesis/figs/export_persian_figures.mjs --help
node Thesis/figs/export_persian_figures.mjs \
  --module-dir /home/pouya/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules \
  --preview-dir /tmp/persian-thesis-figures
node --test Thesis/figs/tests/persian_figures.test.mjs
```

All path arguments are strings. `--source-dir` defaults to the script directory,
`--output-dir` to that source directory, `--browser-path` to
`/usr/bin/google-chrome`, and `--module-dir` to normal Node package resolution.
`--preview-dir` is disabled by default. Existing PDF exports are overwritten;
HTML sources and original English figures are never modified by the exporter.
The exporter blocks network requests, waits for embedded fonts, checks text
against the canvas and its enclosing node, and emits vector PDFs. Optional
previews include PNGs and `geometry.json`; inspect all six PNGs after changes.
Only files ending in `-fa.html` are exported.

Compile the complete thesis from `Thesis/` with its established XeLaTeX/BibTeX
workflow. A successful figure export does not verify thesis pagination or cross
references. The built-in compiler currently fails before reading the document
because its Tectonic resource bundle is unavailable; no updated full-thesis PDF
is claimed or substituted for the existing `Thesis/thesis.pdf`.

The diagram skill's stock `self_check.py` flags the embedded font as a
non-fragment CSS URL (it also rejects data URLs). This is a deliberate offline
font exception, not a remote dependency. Browser geometry checks passed for all
six sources, and each exported PDF was verified as one page and visually reviewed
after Poppler rendering. CLI help and source/LaTeX linkage tests passed.
