# Long-Audio Data-Creation Pipeline Diagram

This directory contains the paper figure for the long-audio pseudo-labelled
dataset pipeline described in `Thesis/chapters/work.tex`.

- `index.html` is the editable, self-contained source with inline SVG and CSS;
  its **Download PNG** button downloads the `3360 × 2376` print image.
- `long-audio-data-creation-pipeline.png` is the print raster used by the thesis.

The figure follows `docs/diagram-style-guide.md` and uses a `1120 × 792`
A4-landscape viewBox. It contains English labels only and shows the seven
executed stages from source preparation through Whisper-small training. Source
discovery and source validation share one node. There is no split or publication
stage: all accepted pseudo-labelled clips were used as training data.

To inspect the source, open `index.html` in a modern browser. The checked-in PNG
is rendered at 3× (`3360 × 2376`) from the `1120 × 792` design frame for print.
The thesis copy is stored at
`Thesis/figs/long-audio-data-creation-pipeline.png`. If the diagram is edited,
preserve the frame and scale and replace both PNG copies so they remain in sync.
