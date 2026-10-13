# Persian thesis defense presentation

`Thesis/presentation/build_persian_deck.mjs` creates a separate, editable,
20-slide Persian defense deck and a PDF with the same slides. It preserves the
English presentations and original thesis figures.

The content follows `robust_persian_asr_thesis_defense_updated.pptx` and the current
`Thesis/chapters/work.tex`, `results.tex`, and `conclusions.tex`. The older 17-slide
plan in `slides.md` does not determine this deck's slide count. Slides include
Persian speaker notes and source references. Charts, the comparison overview and
the summary table remain native editable objects. The implementation figures use
the exact English crops embedded in the updated deck, without translation or
redrawing. The original licensed bridge figure remains unchanged with visible
attribution.

## Dependencies and command

Use the Codex bundled Node.js, Python, `@oai/artifact-tool`, LibreOffice and Poppler
runtime, discovered with `load_workspace_dependencies`. No project ML dependency
or network access is needed to rebuild. Fonts and original image crops are bundled
in the repository. The installed presentation skill supplies validators.
The default paths target bundle `26.1011.11113` on this workstation. Supply
`--runtime-dir` and `--skill-dir` to use a different installation.

```bash
/home/pouya/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node \
  Thesis/presentation/build_persian_deck.mjs --help

/home/pouya/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node \
  Thesis/presentation/build_persian_deck.mjs
```

Options accept paths as strings. `--output-dir` defaults to
`Thesis/presentation/persian`. `--build-dir` defaults to a newly created directory
under `/tmp`. `--basename` defaults to `robust_persian_asr_thesis_defense_fa`.
Existing outputs are never overwritten. Use a new basename for a revision:

```bash
node Thesis/presentation/build_persian_deck.mjs \
  --basename robust_persian_asr_thesis_defense_fa_v2
```

`--no-pdf` skips PDF conversion and PNG previews. `--runtime-dir` and
`--skill-dir` default to the paths encoded near the top of the script. `--help`
works without loading the authoring libraries or creating output files.

## Design and Persian text

The deck uses the paper, ink and blue-teal colors in
`docs/diagram-style-guide.md`, with **Vazirmatn Regular and Bold** for Persian text
and headings. The unmodified fonts, SIL Open Font License, pinned version and
checksums are in `Thesis/presentation/assets/fonts/vazirmatn`. Official source:
[Vazirmatn v33.003](https://github.com/rastikerdar/vazirmatn/tree/v33.003).

The builder registers this font directory through a private Fontconfig file for
validation and PDF rendering. It does not install or alter system fonts. Each
build also delivers `<basename>_fonts.zip`, including installation instructions.
Install its two TTF files on another computer used to edit or present the PPTX.
The PDF embeds its fonts and is the stable viewing/printing version.
The university mark is a raster export of the original
`Thesis/front/template/images/logo.pdf`, saved as
`Thesis/presentation/assets/sharif-logo-presentation.png`.

The builder uses logical Persian Unicode text and sets native OOXML paragraph
direction, Persian language tags and complex-script font mappings after export.
It never reverses text into visual order. English model/dataset identifiers and
the original implementation and paper figures remain in English.

## Implementation image provenance

`Thesis/presentation/assets/english-implementation-crops/provenance.json` records
the source deck checksum, source ZIP member, original slide, dimensions and SHA-256
for each of nine images. These are byte-for-byte copies of already cropped PNGs
embedded in `robust_persian_asr_thesis_defense_updated.pptx`:

- Slides 6–7: degradation preparation, codec/network paths and alignment.
- Slides 9–10: fusion architecture and training strip.
- Slide 14: source validation and segmentation.
- Slide 15: teacher transcription, normalization, Qwen refinement and filtering.

The builder preserves each crop and its aspect ratio. Surrounding titles,
explanations and speaker notes are Persian. If the English deck changes, refresh
the extracted assets and provenance before rebuilding. The tests compare each
asset directly with the recorded source ZIP member.

## Validation and review

The builder validates the 16:9 dimensions, 20-slide count, package structure,
fonts, required native chart/table locations and chart data. Literal chart data
are materialized in embedded workbooks for portability. The private build
directory contains `staging/validation.json`, original drafts and, when exporting
PDF, a `previews/` directory with every slide rendered at 1600 pixels wide.

Review all previews after changing copy or geometry. Check Persian word order,
mixed Persian/Latin text, wrapping, arrow direction, signed domain changes and
chart labels. Small fusion differences must remain qualified, acquired and
accepted IranSeda hours must remain distinct, and the two experimental branches
must remain separate.

CLI help coverage:

```bash
node --test Thesis/presentation/tests/build_persian_deck.test.mjs
```
