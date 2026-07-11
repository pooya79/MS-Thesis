(function () {
  "use strict";

  const VIEWBOX = {width: 1800, height: 1040};
  const groups = [{id: "codec-stage", x: 30, y: 330, width: 1740, height: 420, label: "CHANNEL, CODEC, AND NETWORK IMPAIRMENT · expanded", className: "codec-group"}];
  const nodes = [
    {id:"clean",x:25,y:105,w:125,h:82,kind:"input",title:["Clean audio"],sub:["source clip"]},
    {id:"normalize",x:175,y:105,w:145,h:82,kind:"process",title:["Mono +","resampling"],sub:["working rate"]},
    {id:"profile",x:345,y:105,w:145,h:82,kind:"process",title:["Profile","sampling"],sub:["e.g. telephone_noisy"]},
    {id:"noise",x:515,y:105,w:145,h:82,kind:"process optional",title:["Optional noise"],sub:["mix at sampled SNR"]},
    {id:"effects",x:685,y:105,w:165,h:82,kind:"level",title:["Gain + optional","hard clipping"]},
    {id:"channel",x:875,y:105,w:160,h:82,kind:"codec",title:["Channel, codec","+ network"],sub:["expanded below"]},
    {id:"modelrate",x:1060,y:105,w:155,h:82,kind:"process",title:["Model-rate","resampling"]},
    {id:"target",x:1240,y:105,w:155,h:82,kind:"process",title:["Clean target"],sub:["bandwidth + length aligned"]},
    {id:"peak",x:1420,y:105,w:165,h:82,kind:"level",title:["Shared pair","peak safety"]},
    {id:"outputs",x:1610,y:105,w:165,h:82,kind:"output",title:["WAV pairs +","JSONL metadata"]},
    {id:"select",x:60,y:405,w:165,h:78,kind:"process",title:["Select codec"]},
    {id:"path",x:255,y:405,w:180,h:78,kind:"process",title:["Channel path"],sub:["codec-defined / sampled for pass-through"]},
    {id:"band",x:465,y:405,w:160,h:78,kind:"process",title:["Resample +","band-limit"]},
    {id:"guard",x:655,y:405,w:150,h:78,kind:"level",title:["Pre-codec","peak guard"]},
    {id:"opus",x:850,y:350,w:175,h:85,kind:"codec",title:["Opus packets"],sub:["packet encode"]},
    {id:"loss",x:1060,y:350,w:185,h:85,kind:"codec optional",title:["Burst packet loss"],sub:["network enabled only"]},
    {id:"plc",x:1280,y:350,w:170,h:85,kind:"codec",title:["Decode with PLC"]},
    {id:"ffmpeg",x:850,y:505,w:190,h:85,kind:"process",title:["ffmpeg round-trip"],sub:["non-Opus; or Opus without loss"]},
    {id:"pass",x:850,y:650,w:190,h:70,kind:"passthrough",title:["Pass-through"],sub:["no codec round-trip"]},
    {id:"dropout",x:1100,y:555,w:220,h:90,kind:"process optional",title:["Decoded-waveform","frame dropout"],sub:["optional packet-loss fallback"]},
    {id:"align",x:1500,y:490,w:220,h:100,kind:"level",title:["Delay estimation","+ alignment"],sub:["rejoin every path"]}
  ];
  const edges = [
    ["clean-normalize","M150 146H175",""],["normalize-profile","M320 146H345",""],["profile-noise","M490 146H515","optional"],["noise-effects","M660 146H685",""],["effects-channel","M850 146H875",""],["channel-model","M1035 146H1060",""],["model-target","M1215 146H1240",""],["target-peak","M1395 146H1420",""],["peak-output","M1585 146H1610",""],
    ["channel-select","M955 187V280H142V405",""],["select-path","M225 444H255",""],["path-band","M435 444H465",""],["band-guard","M625 444H655",""],
    ["guard-opus","M805 444H825V392H850","opus","Opus + network enabled",900,330],["opus-loss","M1025 392H1060","opus optional"],["loss-plc","M1245 392H1280","opus"],
    ["guard-ffmpeg","M805 444H825V547H850","other","other codecs; or Opus without loss",1010,492],["ffmpeg-direct","M1040 547H1080V520H1500","other","network off",1270,510],["ffmpeg-dropout","M1040 565H1100","other optional"],["dropout-align","M1320 600H1460V550H1500","other"],
    ["guard-pass","M730 483V685H850","pass","pass-through",790,675],["pass-direct","M1040 685H1460V570H1500","pass","network off",1270,674],["pass-dropout","M1040 675H1070V620H1100","pass optional"],["plc-align","M1450 392H1475V510H1500","opus"],
    ["align-return","M1610 490V275H955V187","","aligned degraded audio",1285,265]
  ].map(([id,path,className,label,labelX,labelY]) => ({id,path,className,label,labelX,labelY}));

  function definitions(svg) {
    const defs = svg.append("defs");
    [["arrow-solid","#38434c"],["arrow-optional","#69737a"]].forEach(([id,color]) => {
      const marker = defs.append("marker").attr("id",id).attr("viewBox","0 -5 10 10").attr("refX",9).attr("markerWidth",7).attr("markerHeight",7).attr("orient","auto");
      marker.append("path").attr("d","M0,-5L10,0L0,5Z").attr("fill",color);
    });
  }

  function draw(root) {
    const group = root.selectAll("g.arch-group").data(groups).join("g");
    group.append("rect").attr("class",d => `group-box ${d.className}`).attr("x",d => d.x).attr("y",d => d.y).attr("width",d => d.width).attr("height",d => d.height).attr("rx",8);
    group.append("text").attr("class","section-label").attr("x",d => d.x + 18).attr("y",d => d.y + 28).text(d => d.label);

    const edge = root.selectAll("g.arch-edge").data(edges).join("g");
    edge.append("path").attr("class",d => `edge ${d.className}`).attr("d",d => d.path);
    const labelled = edge.filter(d => d.label);
    labelled.append("rect").attr("class","edge-label-bg").attr("x",d => d.labelX - d.label.length * 4.2).attr("y",d => d.labelY - 15).attr("width",d => d.label.length * 8.4).attr("height",19).attr("rx",3);
    labelled.append("text").attr("class","edge-label").attr("x",d => d.labelX).attr("y",d => d.labelY).text(d => d.label);

    const node = root.selectAll("g.arch-node").data(nodes).join("g");
    node.append("rect").attr("class",d => `node-box ${d.kind}`).attr("x",d => d.x).attr("y",d => d.y).attr("width",d => d.w).attr("height",d => d.h).attr("rx",7);
    node.each(function (d) {
      const selection = d3.select(this);
      const start = d.y + 29 - (d.title.length - 1) * 8;
      d.title.forEach((line,index) => selection.append("text").attr("class","node-title").attr("x",d.x + d.w / 2).attr("y",start + index * 20).text(line));
      (d.sub || []).forEach((line,index) => selection.append("text").attr("class","node-subtitle").attr("x",d.x + d.w / 2).attr("y",d.y + d.h - 13 - index * 15).text(line));
    });

    const legend = root.append("g");
    legend.append("rect").attr("class","legend-panel").attr("x",30).attr("y",785).attr("width",1740).attr("height",220).attr("rx",7);
    legend.append("text").attr("class","section-label").attr("x",48).attr("y",815).text("READING THE FIGURE");
    const items = [
      {x:55,title:"Deterministic variants",copy:["A stable seed drives profile and parameter sampling per variant.","Configuration probabilities are intentionally omitted."]},
      {x:620,title:"Conditional paths",copy:["Dashed boxes and connectors run only when selected or enabled.","Codec and network branches remain visually distinct."]},
      {x:1190,title:"Recorded metadata",copy:["JSONL records the profile, effects, codec/network details, alignment,", "seed, source identity, and output paths."]}
    ];
    const columns = legend.selectAll("g.legend-item").data(items).join("g").attr("transform",d => `translate(${d.x},845)`);
    columns.append("text").attr("class","legend-title").text(d => d.title);
    columns.each(function (d) { d.copy.forEach((line,index) => d3.select(this).append("text").attr("class","legend-copy").attr("y",30 + index * 20).text(line)); });
    legend.append("line").attr("x1",625).attr("x2",700).attr("y1",955).attr("y2",955).attr("class","edge optional");
    legend.append("text").attr("class","legend-copy").attr("x",715).attr("y",960).text("optional / conditional operation");
  }

  function localizeSvgUrl(value) { return value.replace(/url\((?:["'])?.*?(#[\w-]+)(?:["'])?\)/g,"url($1)"); }
  function inlinePresentationStyles(source, clone) {
    const properties = ["color","direction","fill","stroke","stroke-width","stroke-dasharray","stroke-linecap","stroke-linejoin","marker-end","opacity","font-family","font-size","font-style","font-weight","letter-spacing","text-anchor","text-rendering","unicode-bidi"];
    const sources = [source,...source.querySelectorAll("*")];
    const targets = [clone,...clone.querySelectorAll("*")];
    sources.forEach((element,index) => { const computed = window.getComputedStyle(element); properties.forEach(property => { const value = computed.getPropertyValue(property); if (value) targets[index].style.setProperty(property,localizeSvgUrl(value)); }); });
  }
  function serializeFigure() {
    const source = document.querySelector("#degradation-pipeline-svg");
    const clone = source.cloneNode(true);
    clone.setAttribute("xmlns","http://www.w3.org/2000/svg"); clone.setAttribute("width",VIEWBOX.width); clone.setAttribute("height",VIEWBOX.height); inlinePresentationStyles(source,clone);
    const background = document.createElementNS("http://www.w3.org/2000/svg","rect");
    [["x","0"],["y","0"],["width",String(VIEWBOX.width)],["height",String(VIEWBOX.height)],["fill","#ffffff"]].forEach(([key,value]) => background.setAttribute(key,value));
    clone.insertBefore(background,clone.querySelector("g[data-layer='scene']"));
    return `<?xml version="1.0" encoding="UTF-8"?>\n${new XMLSerializer().serializeToString(clone)}`;
  }
  function downloadBlob(blob,filename) { const url=URL.createObjectURL(blob),link=document.createElement("a"); link.href=url; link.download=filename; document.body.appendChild(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url),1000); }
  function exportSvg() { const xml=serializeFigure(); downloadBlob(new Blob([xml],{type:"image/svg+xml;charset=utf-8"}),"degradation-pipeline-architecture.svg"); }
  function exportPng() {
    const xml=serializeFigure(),blob=new Blob([xml],{type:"image/svg+xml;charset=utf-8"}),url=URL.createObjectURL(blob),image=new Image();
    image.onload=() => { const scale = 3,canvas=document.createElement("canvas"); canvas.width=VIEWBOX.width*scale; canvas.height=VIEWBOX.height*scale; const context=canvas.getContext("2d"); context.fillStyle="#ffffff"; context.fillRect(0,0,canvas.width,canvas.height); context.drawImage(image,0,0,canvas.width,canvas.height); URL.revokeObjectURL(url); canvas.toBlob(png => { if(png) downloadBlob(png,"degradation-pipeline-architecture@3x.png"); },"image/png"); };
    image.onerror=() => { URL.revokeObjectURL(url); showError(new Error("The SVG could not be rasterized.")); }; image.src=url;
  }
  function showError(error) { document.querySelector("#render-status").textContent="Figure rendering failed."; document.querySelector("#dependency-error").hidden=false; console.error(error); }
  function render() {
    if (!window.d3) throw new Error("Pinned D3 dependency is unavailable.");
    const svg=d3.select("#diagram").append("svg").attr("id","degradation-pipeline-svg").attr("class","figure-root").attr("viewBox",`0 0 ${VIEWBOX.width} ${VIEWBOX.height}`).attr("preserveAspectRatio","xMidYMid meet").attr("role","img").attr("aria-labelledby","svg-title svg-description");
    svg.append("title").attr("id","svg-title").text("Speech degradation pair-generation pipeline architecture");
    svg.append("desc").attr("id","svg-description").text("Clean audio is resampled and deterministically assigned a profile. After optional noise and gain, codec selection determines the channel path. With network impairment, Opus uses packet burst loss and PLC; other codecs use an ffmpeg round-trip plus decoded-waveform dropout. Pass-through bypasses codecs but can still receive dropout. Every path is delay-aligned, paired with a bandwidth-matched clean target, peak-safe, and written as WAV pairs with JSONL metadata.");
    definitions(svg); const root=svg.append("g").attr("data-layer","scene"); root.append("text").attr("class","figure-kicker").attr("x",25).attr("y",43).text("CLEAN-TO-DEGRADED PAIR FLOW"); root.append("line").attr("x1",25).attr("x2",1775).attr("y1",60).attr("y2",60).attr("stroke","#c2c9ce"); draw(root);
    ["export-svg","export-png"].forEach(id => document.querySelector(`#${id}`).disabled=false); document.querySelector("#export-svg").addEventListener("click",exportSvg); document.querySelector("#export-png").addEventListener("click",exportPng); const status=document.querySelector("#render-status"); status.textContent="Figure ready. Exports include all figure styles."; status.dataset.ready="true";
  }
  function beginRender() { try { render(); } catch(error) { showError(error); } }
  if (new URLSearchParams(window.location.search).has("capture")) document.body.classList.add("capture-mode");
  if(document.readyState === "complete") beginRender(); else window.addEventListener("load",beginRender,{once:true});
})();
