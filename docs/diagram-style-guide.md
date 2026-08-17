# MS Thesis Diagram Style Guide

This file is the reusable visual profile for research diagrams in this repository. It is
derived from the existing speech-degradation figure, but reduces its palette and visual
grammar so diagrams remain legible in documentation, presentation, and print contexts.

## Design intent

- Editorial and technical, without looking like an application dashboard.
- Low visual density: show the main argument first and move implementation detail into a
  second figure or nearby prose.
- Shape, line style, and position carry meaning. Color identifies only the focal stage.
- Static, accessible, and useful without JavaScript or network access.

## Tokens

| Role | Value | Use |
|---|---|---|
| `paper` | `#f7f6f2` | Page and SVG background |
| `surface` | `#ffffff` | Standard process nodes |
| `ink` | `#1f2933` | Primary text and strong outlines |
| `muted` | `#52606d` | Connectors and secondary text |
| `soft` | `#7b8794` | Tags, captions, and quiet metadata |
| `rule` | `rgba(31, 41, 51, 0.14)` | Dividers and hairlines |
| `accent` | `#1d6f8a` | One focal stage or decision path |
| `accent-tint` | `#e8f3f6` | Fill behind the focal stage |
| `input-tint` | `#edf1f3` | External input nodes |
| `output-tint` | `#eef4ef` | Durable output nodes |

The accent is a blue-teal consolidation of the earlier blue and green roles. Orange and
purple branch colors are intentionally retired: they made readers learn a palette before
they could follow the pipeline.

## Typography

| Role | Family | Weight | Typical size |
|---|---|---|---|
| Page and figure title | Instrument Serif, Georgia, serif | 400 | 28px |
| Node name | Geist, Inter, system sans-serif | 600 | 12px |
| Supporting prose | Geist, Inter, system sans-serif | 400 | 12–16px |
| Technical labels | Geist Mono, ui-monospace, monospace | 500 | 8px |

Use monospace only for step numbers, branch labels, paths, commands, field names, and
similar technical content. Do not use it for human-readable component names.

## Geometry and hierarchy

- Build on a 4px grid. Use 40px outer SVG margins and at least 24px between nodes.
- Standard node radius is 8px; decision diamonds and start/end capsules are the only
  exceptions.
- Use 1px borders and no shadows, gradients, glow, or decorative background patterns.
- Use rounded orthogonal connectors. Straight lines are allowed only for nodes sharing an
  axis; never use diagonal connectors.
- Draw connectors before nodes. Fan multiple connections across distinct attachment
  points, and never route a connector behind an unrelated node.
- Put connector labels in open canvas with an opaque paper mask and 8–12px clearance from
  the line.

## Information budget

- Overview: at most 7 nodes and 9 connectors.
- Detail: at most 9 nodes and 12 connectors.
- Use the accent on at most one node and its immediately relevant connector.
- Split a system into overview and detail figures instead of shrinking labels.
- Keep changing values such as probabilities, weights, SNR ranges, and codec parameters
  in configuration or prose rather than embedding them in architecture diagrams.

## Accessibility and delivery

- Prefer a single standalone `.html` file with embedded CSS and inline SVG.
- Every meaningful SVG must use `role="img"`, a unique `aria-labelledby`, a first-child
  `<title>`, and a one-sentence `<desc>`.
- The full meaning must remain visible without JavaScript. Remote fonts may enhance the
  rendering, but the fallback stack must remain usable offline.
- For full-width documentation, use a `1280 × 720` viewBox. For paper figures, use one
  self-contained `1120 × 792` A4-landscape SVG, omit webpage-style explanation around the
  figure, and let the paper provide the caption.
- Export raster or standalone SVG files only when the destination requires them.
