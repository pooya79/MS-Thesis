# Review the current Persian defense slides

`Thesis/presentation/revise_persian_deck.mjs` applies the presentation feedback
to the current 39-slide `references_qwen_compact` deck. It creates a separate
40-slide PowerPoint and PDF in `Thesis/presentation/persian`, preserving the
source, references, appendix, charts, embedded workbooks and equation formatting.
It does not rebuild from the older 28-slide generator.

```bash
node Thesis/presentation/revise_persian_deck.mjs --help
node Thesis/presentation/revise_persian_deck.mjs
node --test Thesis/presentation/tests/revise_persian_deck.test.mjs
```

The script uses the bundled Codex Node/Python runtime and presentation library.
If `node` is unavailable on PATH, use
`/home/pouya/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node`.
It accepts `--source PATH`, `--output-dir PATH`, `--basename NAME`,
`--build-dir PATH`, `--pdf-renderer PATH`, and `--no-pdf`. Existing output files are never overwritten.
Use another basename for further revisions. Source layout and expected slide
count are checked so an unrelated deck cannot silently receive these patches.

Counts in the training/validation and test tables are labeled as speech samples
(utterances). WER and CER use percent; differences use percentage points.
Bitrates, segmentation durations and dimensionless parameters are explicit.
Selected technical terms include English equivalents. Slide 13 uses editable
Persian text and decimal learning rates without Unicode direction-isolation
controls, preserving the original training figure. The original summary becomes
textual conclusion bullets (slide 23), followed by contribution bullets (slide
24), supported by the thesis results and conclusions. Reference slides and the
research appendix follow the main talk.

Private staging contains the candidate, authored slide parts and validation
receipt. The finalizer checks package relationships, slide size, native objects,
bullet geometry and import. Render the output PDF and inspect all pages after
building; automated checks cannot establish Persian line wrapping or visual fit.

The current source charts have complete cached values but only placeholder
references (`categories`, `0`) and no embedded workbooks. The revision explicitly
reauthors those complete caches as literal data and creates new editable workbook
snapshots. These snapshots preserve the plotted categories and values; they do
not claim to recover source formulas or workbook provenance. Chart-label XML
ordering and missing complex-script font metadata are also repaired.

PDF export defaults to the installed stable `/usr/bin/soffice` (LibreOffice
24.2). The bundled LibreOfficeDev 26.8 preview incorrectly draws negative
columns upward. The stable renderer displays the original signed values
correctly. `--pdf-renderer PATH` permits another independently verified renderer.
