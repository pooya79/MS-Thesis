(function () {
  "use strict";

  const VIEWBOX = {width: 1800, height: 1040};

  // The scene is intentionally data-driven: layout or copy changes belong here,
  // not in hand-authored SVG markup.
  const groups = [
    {
      id: "fusion",
      className: "fusion-group",
      x: 900,
      y: 135,
      width: 505,
      height: 455,
      label: "Bidirectional cross-attention fusion · 2 layers"
    },
    {
      id: "training",
      className: "training-group",
      x: 20,
      y: 615,
      width: 1760,
      height: 220,
      label: "Training-only supervision"
    }
  ];

  const nodes = [
    {id: "noisy", x: 30, y: 305, width: 150, height: 100, kind: "input", title: ["Noisy log-Mel"], subtitle: ["input view"], symbol: "x_n"},
    {id: "enhancer", x: 220, y: 290, width: 190, height: 130, kind: "enhancer", title: ["Residual U-Net", "enhancer E"], subtitle: ["log-Mel → log-Mel", "no temporal bottleneck"]},
    {id: "enhanced", x: 450, y: 305, width: 150, height: 100, kind: "enhancer", title: ["Enhanced", "log-Mel"], symbol: "x_e"},
    {id: "encoder-noisy", x: 660, y: 185, width: 180, height: 110, kind: "encoder", title: ["Whisper encoder"], subtitle: ["noisy view"], symbol: "h_n"},
    {id: "encoder-enhanced", x: 660, y: 420, width: 180, height: 110, kind: "encoder", title: ["Whisper encoder"], subtitle: ["enhanced view"], symbol: "h_e"},
    {id: "decoder", x: 1450, y: 310, width: 160, height: 115, kind: "decoder", title: ["Whisper", "decoder"], subtitle: ["autoregressive"]},
    {id: "tokens", x: 1645, y: 310, width: 125, height: 115, kind: "output", title: ["Predicted", "Persian tokens"], symbol: "\\hat{y}_{1:T}"},
    {id: "clean", x: 40, y: 675, width: 170, height: 100, kind: "input", title: ["Clean log-Mel"], subtitle: ["bandwidth-aligned", "training target"], symbol: "x_c"},
    {id: "loss-enh", x: 270, y: 655, width: 330, height: 140, kind: "loss", title: ["Enhancement loss"], equation: "L_{\\mathrm{enh}}=\\lVert E(x_n)-x_c\\rVert_1"},
    {id: "loss-feat", x: 655, y: 655, width: 390, height: 140, kind: "loss disabled", title: ["Optional warm-up feature loss"], equation: "L_{\\mathrm{feat}}=\\lVert \\operatorname{Enc}(E(x_n))-\\operatorname{Enc}(x_c)\\rVert_1", subtitle: ["DISABLED · feature_match_weight = 0.0"]},
    {id: "targets", x: 1055, y: 710, width: 115, height: 85, kind: "input", title: ["Target tokens"], symbol: "y_{1:T}"},
    {id: "loss-asr", x: 1190, y: 645, width: 285, height: 140, kind: "loss", title: ["Autoregressive ASR loss"], equation: "L_{\\mathrm{ASR}}=-\\sum_t \\log p(y_t\\mid y_{<t},h_f)"},
    {id: "loss-total", x: 1515, y: 655, width: 245, height: 140, kind: "total-loss", title: ["Fusion / joint objective"], equation: "L=L_{\\mathrm{ASR}}+\\lambda L_{\\mathrm{enh}}", subtitle: ["λ is stage-configured"]}
  ];

  const edges = [
    {id: "noisy-enhancer", className: "", path: "M180 355 H220"},
    {id: "enhancer-enhanced", className: "enhanced", path: "M410 355 H450"},
    {id: "noisy-encoder", className: "noisy", path: "M105 305 V240 H660", label: "parallel noisy path", labelX: 380, labelY: 230},
    {id: "enhanced-encoder", className: "enhanced", path: "M600 355 H625 V475 H660"},
    {id: "encoder-noisy-fusion", className: "noisy", path: "M840 240 H930", label: "hₙ", labelX: 875, labelY: 227},
    {id: "encoder-enhanced-fusion", className: "enhanced", path: "M840 475 H930", label: "hₑ", labelX: 875, labelY: 462},
    {id: "fusion-decoder", className: "", path: "M1380 355 H1450", label: "h_f", labelX: 1416, labelY: 342},
    {id: "decoder-tokens", className: "", path: "M1610 367.5 H1645"},
    {id: "clean-enh-loss", className: "training", path: "M210 725 H270"},
    {id: "enhanced-enh-loss", className: "training", path: "M525 405 V610 H435 V655"},
    {id: "enhanced-feat-loss", className: "training disabled-edge", path: "M555 405 V595 H850 V655"},
    {id: "clean-feat-loss", className: "training disabled-edge", path: "M125 775 V815 H850 V795"},
    {id: "decoder-asr-loss", className: "training", path: "M1530 425 V605 H1332 V645"},
    {id: "target-asr-loss", className: "training", path: "M1170 752 H1190"},
    {id: "enh-total", className: "training", path: "M600 725 V815 H1490 V760 H1515", label: "λL_enh", labelX: 1435, labelY: 804},
    {id: "asr-total", className: "training", path: "M1475 700 H1515", label: "L_ASR", labelX: 1494, labelY: 687}
  ];

  const attentionLayers = [
    {id: "attn-1", x: 955, y: 205, width: 130, height: 270, label: "Layer 1"},
    {id: "attn-2", x: 1110, y: 205, width: 130, height: 270, label: "Layer 2"}
  ];

  const stageLegend = [
    {stage: "Stage 0", color: "#0072b2", title: "Enhancer warm-up", copy: "Train E with L_enh; optional L_feat is off.", lambda: "current config: λ = 1.0"},
    {stage: "Stage 1", color: "#d55e00", title: "Frozen-backbone fusion", copy: "Train E + fusion; freeze Whisper.", lambda: "current config: λ = 0.3"},
    {stage: "Stage 2", color: "#12805c", title: "End-to-end fine-tuning", copy: "Train E + fusion + complete Whisper stack.", lambda: "current config: λ = 0.1"}
  ];

  const equationPlacements = [
    {id: "noisy-symbol", tex: "x_n", x: 82, y: 350, width: 46, height: 22},
    {id: "enhanced-symbol", tex: "x_e=E(x_n)", x: 482, y: 365, width: 86, height: 26},
    {id: "encoder-noisy-symbol", tex: "h_n=\\operatorname{Enc}(x_n)", x: 685, y: 250, width: 130, height: 25},
    {id: "encoder-enhanced-symbol", tex: "h_e=\\operatorname{Enc}(x_e)", x: 685, y: 485, width: 130, height: 25},
    {id: "gate-equation", tex: "g=\\sigma(\\operatorname{MLP}([h'_n;h'_e]))", x: 1252, y: 325, width: 120, height: 35},
    {id: "fusion-equation", tex: "h_f=g\\odot h'_e+(1-g)\\odot h'_n", x: 1010, y: 510, width: 320, height: 40},
    {id: "token-symbol", tex: "\\hat{y}_{1:T}", x: 1681, y: 380, width: 54, height: 25},
    {id: "clean-symbol", tex: "x_c", x: 96, y: 712, width: 54, height: 22},
    {id: "target-symbol", tex: "y_{1:T}", x: 1088, y: 755, width: 50, height: 20},
    {id: "loss-enh-equation", tex: "L_{\\mathrm{enh}}=\\lVert E(x_n)-x_c\\rVert_1", x: 318, y: 710, width: 235, height: 38},
    {id: "loss-feat-equation", tex: "L_{\\mathrm{feat}}=\\lVert \\operatorname{Enc}(E(x_n))-\\operatorname{Enc}(x_c)\\rVert_1", x: 700, y: 710, width: 300, height: 38},
    {id: "loss-asr-equation", tex: "L_{\\mathrm{ASR}}=-\\sum_t \\log p(y_t\\mid y_{<t},h_f)", x: 1205, y: 690, width: 255, height: 40},
    {id: "loss-total-equation", tex: "L=L_{\\mathrm{ASR}}+\\lambda L_{\\mathrm{enh}}", x: 1540, y: 715, width: 195, height: 38}
  ];

  function addDefinitions(svg) {
    const defs = svg.append("defs");

    const hatch = defs.append("pattern")
      .attr("id", "disabled-hatch")
      .attr("width", 10)
      .attr("height", 10)
      .attr("patternUnits", "userSpaceOnUse")
      .attr("patternTransform", "rotate(45)");
    hatch.append("rect").attr("width", 10).attr("height", 10).attr("fill", "#f6f6f6");
    hatch.append("line").attr("x1", 0).attr("y1", 0).attr("x2", 0).attr("y2", 10).attr("stroke", "#c6cbd0").attr("stroke-width", 3);

    function marker(id, color, size) {
      const result = defs.append("marker")
        .attr("id", id)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 9)
        .attr("refY", 0)
        .attr("markerWidth", size)
        .attr("markerHeight", size)
        .attr("orient", "auto");
      result.append("path").attr("d", "M0,-5L10,0L0,5Z").attr("fill", color);
    }

    marker("arrow-solid", "#38434c", 7);
    marker("arrow-training", "#69737a", 7);
    marker("arrow-small", "#59646d", 5.5);
  }

  function drawGroups(root) {
    const selection = root.selectAll("g.arch-group")
      .data(groups)
      .join("g")
      .attr("class", "arch-group")
      .attr("data-group", (d) => d.id);

    selection.append("rect")
      .attr("class", (d) => `group-box ${d.className}`)
      .attr("x", (d) => d.x)
      .attr("y", (d) => d.y)
      .attr("width", (d) => d.width)
      .attr("height", (d) => d.height)
      .attr("rx", 8);

    selection.append("text")
      .attr("class", "section-label")
      .attr("x", (d) => d.x + 16)
      .attr("y", (d) => d.y + 26)
      .text((d) => d.label);
  }

  function drawEdges(root) {
    const selection = root.selectAll("g.arch-edge")
      .data(edges)
      .join("g")
      .attr("class", "arch-edge")
      .attr("data-edge", (d) => d.id);

    selection.append("path")
      .attr("class", (d) => `edge ${d.className}`)
      .attr("d", (d) => d.path);

    const labelled = selection.filter((d) => d.label);
    labelled.append("rect")
      .attr("class", "edge-label-bg")
      .attr("x", (d) => d.labelX - Math.max(18, d.label.length * 4.5))
      .attr("y", (d) => d.labelY - 16)
      .attr("width", (d) => Math.max(36, d.label.length * 9))
      .attr("height", 21)
      .attr("rx", 3);
    labelled.append("text")
      .attr("class", "edge-label")
      .attr("x", (d) => d.labelX)
      .attr("y", (d) => d.labelY)
      .text((d) => d.label);
  }

  function drawNodes(root) {
    const selection = root.selectAll("g.arch-node")
      .data(nodes)
      .join("g")
      .attr("class", "arch-node")
      .attr("data-node", (d) => d.id);

    selection.append("rect")
      .attr("class", (d) => `node-box ${d.kind}`)
      .attr("x", (d) => d.x)
      .attr("y", (d) => d.y)
      .attr("width", (d) => d.width)
      .attr("height", (d) => d.height)
      .attr("rx", 7);

    selection.each(function (d) {
      const node = d3.select(this);
      const titleStart = d.y + (d.id === "targets" ? 25 : 30);
      d.title.forEach((line, index) => {
        node.append("text")
          .attr("class", "node-title")
          .attr("x", d.x + d.width / 2)
          .attr("y", titleStart + index * 22)
          .style("font-size", d.id === "tokens" ? "16px" : d.id === "targets" ? "14px" : null)
          .text(line);
      });

      if (d.subtitle) {
        const bottom = d.y + d.height - 14;
        d.subtitle.forEach((line, index) => {
          node.append("text")
            .attr("class", d.id === "loss-feat" ? "badge-text" : "node-subtitle")
            .attr("x", d.x + d.width / 2)
            .attr("y", bottom - (d.subtitle.length - 1 - index) * 17)
            .text(line);
        });
      }
    });
  }

  function drawSharedEncoderMark(root) {
    root.append("path")
      .attr("class", "shared-bracket")
      .attr("d", "M850 205 H865 V510 H850");
    root.append("text")
      .attr("class", "shared-label")
      .attr("x", 875)
      .attr("y", 350)
      .attr("transform", "rotate(-90 875 350)")
      .text("ONE SHARED WEIGHT SET");
  }

  function drawFusionDetail(root) {
    root.append("text").attr("class", "lane-label noisy").attr("x", 935).attr("y", 245).text("noisy");
    root.append("text").attr("class", "lane-label enhanced").attr("x", 935).attr("y", 445).text("enhanced");

    const layers = root.selectAll("g.attention")
      .data(attentionLayers)
      .join("g")
      .attr("class", "attention");

    layers.append("rect")
      .attr("class", "attention-layer")
      .attr("x", (d) => d.x)
      .attr("y", (d) => d.y)
      .attr("width", (d) => d.width)
      .attr("height", (d) => d.height)
      .attr("rx", 5);
    layers.append("text")
      .attr("class", "attention-label")
      .attr("x", (d) => d.x + d.width / 2)
      .attr("y", (d) => d.y + 24)
      .text((d) => d.label);

    layers.each(function (d) {
      const layer = d3.select(this);
      layer.append("line").attr("x1", d.x + 12).attr("x2", d.x + d.width - 12).attr("y1", 270).attr("y2", 270).attr("stroke", "#d1d7dc");
      layer.append("line").attr("x1", d.x + 12).attr("x2", d.x + d.width - 12).attr("y1", 405).attr("y2", 405).attr("stroke", "#d1d7dc");
      layer.append("path").attr("class", "cross-edge noisy").attr("d", `M${d.x + 18} 270 C${d.x + 50} 285 ${d.x + 82} 385 ${d.x + 112} 405`);
      layer.append("path").attr("class", "cross-edge enhanced").attr("d", `M${d.x + 18} 405 C${d.x + 50} 385 ${d.x + 82} 285 ${d.x + 112} 270`);
      layer.append("text").attr("class", "lane-label noisy").attr("x", d.x + d.width / 2).attr("y", 255).text("N ← Attn(N,E)");
      layer.append("text").attr("class", "lane-label enhanced").attr("x", d.x + d.width / 2).attr("y", 435).text("E ← Attn(E,N)");
      layer.append("text").attr("class", "small-label").attr("x", d.x + d.width / 2).attr("y", 345).text("cross-view");
      layer.append("text").attr("class", "small-label").attr("x", d.x + d.width / 2).attr("y", 362).text("refinement");
    });

    root.append("path").attr("class", "edge noisy").attr("d", "M930 240 H955");
    root.append("path").attr("class", "edge enhanced").attr("d", "M930 475 V440 H955");
    root.append("path").attr("class", "edge noisy").attr("d", "M1085 270 H1110");
    root.append("path").attr("class", "edge enhanced").attr("d", "M1085 405 H1110");

    root.append("rect")
      .attr("class", "gate-box")
      .attr("x", 1260)
      .attr("y", 235)
      .attr("width", 120)
      .attr("height", 240)
      .attr("rx", 7);
    root.append("text").attr("class", "node-title").attr("x", 1320).attr("y", 270).text("Sigmoid gate");
    root.append("text").attr("class", "node-subtitle").attr("x", 1320).attr("y", 292).text("learned per time");
    root.append("text").attr("class", "node-subtitle").attr("x", 1320).attr("y", 310).text("and per channel");
    root.append("path").attr("class", "edge noisy").attr("d", "M1240 270 H1260");
    root.append("path").attr("class", "edge enhanced").attr("d", "M1240 405 H1260");
    root.append("path").attr("class", "edge").attr("d", "M1380 355 H1405");
    root.append("text").attr("class", "small-label").attr("x", 1160).attr("y", 493).text("gated merge of refined views");
  }

  function drawLegend(root) {
    const x = 20;
    const y = 860;
    const width = 1760;
    const height = 160;
    root.append("rect").attr("class", "legend-panel").attr("x", x).attr("y", y).attr("width", width).attr("height", height).attr("rx", 7);
    root.append("text").attr("class", "section-label").attr("x", x + 16).attr("y", y + 27).text("Training curriculum · configuration snapshot");

    const columns = root.selectAll("g.stage-column")
      .data(stageLegend)
      .join("g")
      .attr("class", "stage-column")
      .attr("transform", (d, index) => `translate(${x + 22 + index * 575},${y + 48})`);

    columns.append("rect").attr("width", 68).attr("height", 24).attr("rx", 4).attr("fill", (d) => d.color);
    columns.append("text").attr("class", "legend-stage").attr("x", 34).attr("y", 17).attr("fill", "#ffffff").attr("text-anchor", "middle").text((d) => d.stage);
    columns.append("text").attr("class", "legend-stage").attr("x", 82).attr("y", 17).attr("fill", (d) => d.color).text((d) => d.title);
    columns.append("text").attr("class", "legend-copy").attr("x", 0).attr("y", 51).text((d) => d.copy);
    columns.append("text").attr("class", "legend-lambda").attr("x", 0).attr("y", 76).text((d) => d.lambda);

    root.append("text")
      .attr("class", "config-note")
      .attr("x", 1760)
      .attr("y", 1004)
      .attr("text-anchor", "end")
      .text("λ values annotate configs/speech_enhancement/fusion_train.yaml; they are not architectural constants.");
  }

  async function renderEquations(root) {
    // Preload the complete original TeX path data once. This keeps conversion
    // deterministic and avoids partial synchronous output at dynamic ranges.
    await window.MathJax.startup.document.outputJax.font.loadDynamicFiles();
    for (const item of equationPlacements) {
      const container = window.MathJax.tex2svg(item.tex, {display: false});
      const source = container.querySelector("svg");
      if (!source) {
        throw new Error(`MathJax did not return SVG for ${item.id}`);
      }
      const mathSvg = source.cloneNode(true);
      mathSvg.setAttribute("class", "equation-host");
      mathSvg.setAttribute("data-equation", item.id);
      mathSvg.setAttribute("x", item.x);
      mathSvg.setAttribute("y", item.y);
      mathSvg.setAttribute("width", item.width);
      mathSvg.setAttribute("height", item.height);
      mathSvg.setAttribute("preserveAspectRatio", "xMidYMid meet");
      mathSvg.removeAttribute("style");
      root.node().appendChild(mathSvg);
    }
  }

  function localizeSvgUrl(value) {
    return value.replace(/url\((?:["'])?.*?(#[\w-]+)(?:["'])?\)/g, "url($1)");
  }

  function inlinePresentationStyles(source, clone) {
    const properties = [
      "color",
      "fill",
      "stroke",
      "stroke-width",
      "stroke-dasharray",
      "stroke-linecap",
      "stroke-linejoin",
      "marker-end",
      "opacity",
      "font-family",
      "font-size",
      "font-style",
      "font-weight",
      "letter-spacing",
      "text-anchor",
      "text-transform",
      "overflow",
      "text-rendering"
    ];
    const sourceElements = [source, ...source.querySelectorAll("*")];
    const cloneElements = [clone, ...clone.querySelectorAll("*")];

    sourceElements.forEach((sourceElement, index) => {
      const equationRoot = sourceElement.closest?.(".equation-host");
      if (equationRoot && sourceElement !== equationRoot) {
        return; // MathJax paths already carry self-contained presentation attributes.
      }
      const computed = window.getComputedStyle(sourceElement);
      const target = cloneElements[index];
      for (const property of properties) {
        const value = computed.getPropertyValue(property);
        if (value) {
          target.style.setProperty(property, localizeSvgUrl(value));
        }
      }
    });
  }

  function serializeFigure() {
    const source = document.querySelector("#fusion-architecture-svg");
    const clone = source.cloneNode(true);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
    clone.setAttribute("width", VIEWBOX.width);
    clone.setAttribute("height", VIEWBOX.height);
    inlinePresentationStyles(source, clone);

    const background = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    background.setAttribute("x", "0");
    background.setAttribute("y", "0");
    background.setAttribute("width", String(VIEWBOX.width));
    background.setAttribute("height", String(VIEWBOX.height));
    background.setAttribute("fill", "#ffffff");
    const firstGraphic = clone.querySelector("g[data-layer='scene']");
    clone.insertBefore(background, firstGraphic);

    return `<?xml version="1.0" encoding="UTF-8"?>\n${new XMLSerializer().serializeToString(clone)}`;
  }

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function exportSvg() {
    const xml = serializeFigure();
    downloadBlob(new Blob([xml], {type: "image/svg+xml;charset=utf-8"}), "fusion-model-architecture.svg");
  }

  function exportPng() {
    const xml = serializeFigure();
    const blob = new Blob([xml], {type: "image/svg+xml;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = function () {
      const scale = 3;
      const canvas = document.createElement("canvas");
      canvas.width = VIEWBOX.width * scale;
      canvas.height = VIEWBOX.height * scale;
      const context = canvas.getContext("2d");
      context.fillStyle = "#ffffff";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      canvas.toBlob((png) => {
        if (png) {
          downloadBlob(png, "fusion-model-architecture@3x.png");
        }
      }, "image/png");
    };
    image.onerror = function () {
      URL.revokeObjectURL(url);
      showError(new Error("The SVG could not be rasterized."));
    };
    image.src = url;
  }

  function showError(error) {
    const status = document.querySelector("#render-status");
    const panel = document.querySelector("#dependency-error");
    status.textContent = "Figure rendering failed.";
    panel.hidden = false;
    console.error(error);
  }

  async function render() {
    if (!window.d3 || !window.MathJax || !window.MathJax.startup) {
      throw new Error("Pinned D3 or MathJax dependency is unavailable.");
    }

    const svg = d3.select("#diagram")
      .append("svg")
      .attr("id", "fusion-architecture-svg")
      .attr("class", "figure-root")
      .attr("viewBox", `0 0 ${VIEWBOX.width} ${VIEWBOX.height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .attr("role", "img")
      .attr("aria-labelledby", "svg-title svg-description");

    svg.append("title")
      .attr("id", "svg-title")
      .text("Dual-view speech enhancement and fusion model for robust Persian ASR");
    svg.append("desc")
      .attr("id", "svg-description")
      .text("Noisy log-Mel features pass through a residual U-Net while a parallel noisy path is preserved. A shared Whisper encoder processes both views, two bidirectional cross-attention layers refine them, a sigmoid gate merges them, and a Whisper decoder predicts Persian tokens. Dashed training-only paths show enhancement, ASR, optional disabled feature, and combined losses, followed by a three-stage training legend.");
    addDefinitions(svg);

    const root = svg.append("g").attr("data-layer", "scene");
    root.append("text").attr("class", "figure-kicker").attr("x", 24).attr("y", 42).text("INFERENCE PATH");
    root.append("line").attr("x1", 24).attr("x2", 1776).attr("y1", 59).attr("y2", 59).attr("stroke", "#c2c9ce").attr("stroke-width", 1);
    drawGroups(root);
    drawEdges(root);
    drawNodes(root);
    drawSharedEncoderMark(root);
    drawFusionDetail(root);
    drawLegend(root);
    await renderEquations(root);

    document.querySelector("#export-svg").disabled = false;
    document.querySelector("#export-png").disabled = false;
    document.querySelector("#export-svg").addEventListener("click", exportSvg);
    document.querySelector("#export-png").addEventListener("click", exportPng);
    const status = document.querySelector("#render-status");
    status.textContent = "Figure ready. Vector equations are embedded in the SVG.";
    status.dataset.ready = "true";
  }

  function beginRender() {
    render().catch(showError);
  }

  // MathJax's pageReady phase is tied to the document load lifecycle. Starting
  // after `load` avoids racing that phase when all three scripts use `defer`.
  if (document.readyState === "complete") {
    beginRender();
  } else {
    window.addEventListener("load", beginRender, {once: true});
  }
})();
