# Persian thesis defense presentation

`Thesis/presentation/build_persian_deck.mjs` creates a separate, editable,
22-slide Persian defense deck and a PDF with the same slides. It preserves the
English presentations and original thesis figures.

The content follows `robust_persian_asr_thesis_defense_updated.pptx` and the current
`Thesis/chapters/literature.tex`, `work.tex`, `results.tex`, and `conclusions.tex`. The older 17-slide
plan in `slides.md` does not determine this deck's slide count. Slides include
Persian speaker notes and source references. Charts, the comparison overview and
the summary table remain native editable objects. The implementation figures use
the exact English crops embedded in the updated deck, without translation or
redrawing. The original licensed bridge figure remains unchanged with visible
attribution.

## Literature review

Slides 3–4 survey eight families of modern ASR methods before the experiment
overview. Both use editable tables with **method family**, **main idea** and
**example studies**. They present the field generally. The presenter can state
orally that their main focus is combining original and enhanced audio.

Slide 3, **آموزش، سازگارسازی و اصلاح خروجی**, covers:

- **Multi-condition training and augmentation:** pMCT (2022) and
  [acoustic-aware augmentation (2025, preprint)](https://arxiv.org/abs/2505.20606).
- **Foundation-model adaptation:** [MCAT–LoRA (2025)](https://doi.org/10.1186/s13636-025-00406-5),
  [Whisper–SPS (2026)](https://aclanthology.org/2026.sigul-1.26/) and the
  [Neyshekar Persian corpus and fine-tuning study (2026, preprint)](https://arxiv.org/abs/2609.14542).
- **Self-supervised and semi-supervised learning:**
  [Bhogale et al. (2024)](https://www.isca-archive.org/interspeech_2024/bhogale24_interspeech.html),
  [cross-lingual continual pretraining (2025)](https://aclanthology.org/2025.findings-ijcnlp.99/)
  and [DASH self-distillation (2026)](https://www.isca-archive.org/interspeech_2026/baik26_interspeech.html).
- **Language-model correction:**
  [ClozeGER / Listen Again (2024)](https://aclanthology.org/2024.findings-acl.37/)
  and [conservative filtering (2024)](https://aclanthology.org/2024.emnlp-industry.20/).

It also names Speech Robust Bench, URGENT and PSRB (2025) as evaluation resources.
SPS and Neyshekar are data/fine-tuning studies, not new ASR architectures.
Unlabeled-data pretraining, self-distillation and pseudo-labeling remain distinct.

Slide 4, **بهسازی گفتار و ادغام نماها**, covers:

- **Enhancement before recognition:**
  [Schrödinger bridge enhancement (2025)](https://arxiv.org/abs/2505.04237) and
  [decoupled enhancement/recognition (2026)](https://doi.org/10.1016/j.csl.2025.101821).
- **Task-oriented enhancement:**
  [Dissen et al. (2025)](https://keshet.technion.ac.il/a-front-end-adaptation-network-for-improving-speech-recognition-performance-in-packet-loss-and-noisy-environments/)
  and [SSL-MSE (2025)](https://doi.org/10.3389/frsip.2025.1587969).
- **Representation fusion:** [FAT-HuBERT (2023)](https://arxiv.org/abs/2311.17790)
  and [Wav2code (2024)](https://doi.org/10.1109/TASLP.2023.3332545).
- **Waveform combination:** adaptive bridges from
  [Wang et al. (2024)](https://doi.org/10.23919/EUSIPCO63174.2024.10715447) and
  [Cui et al. (2025)](https://arxiv.org/abs/2501.02452),
  [Conf-OA (Interspeech 2026)](https://www.isca-archive.org/interspeech_2026/li26s_interspeech.html)
  and [Rethink-and-Refine (MLSP 2026)](https://arxiv.org/abs/2608.07781).

The two slides cite 23 studies/resources in their Persian speaker notes, with
full titles, venues, source links and stable reference keys. Sources and publication
status were checked on 15 October 2026. Stars identify the cited preprints
(acoustic-aware augmentation, Neyshekar and PSRB). Conf-OA is cited using its
published Interspeech 2026 record. Additional 2026 sources are documented in the
presentation notes without changing the thesis chapter or bibliography.

The slides distinguish method families without ranking WER across incompatible
languages, corpora and recognizers. Their descriptions and speaker notes explain
the methods without a thesis-question column or a statement of the presenter's
research focus. Allow about 2½ minutes for both slides.

The original 20 slides retain their content and move forward two positions after
slide 2. The current revision is delivered as
`persian/robust_persian_asr_thesis_defense_fa_literature_methods.{pptx,pdf}` with a
matching font ZIP. Existing decks remain available.

## Dependencies and command

Use the Codex bundled Node.js, Python, `@oai/artifact-tool`, LibreOffice and Poppler
runtime, discovered with `load_workspace_dependencies`. No project ML dependency
or network access is needed to rebuild. Fonts and original image crops are bundled
in the repository. The installed presentation skill supplies validators.
The default paths target bundle `26.1013.11914` on this workstation. Supply
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
  --basename robust_persian_asr_thesis_defense_fa_literature_methods
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

The builder validates the 16:9 dimensions, 22-slide count, package structure,
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

The tests also inspect the delivered literature revision for both review slides'
citations, editable table and Persian text direction, and compare all retained
slide text against the earlier revised deck, excluding updated page numbers.
After rebuilding, review the new deliverables before replacing that regression
fixture; the builder itself refuses to overwrite existing outputs.
