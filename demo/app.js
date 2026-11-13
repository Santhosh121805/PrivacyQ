"use strict";
// Architecture walkthrough. Reads manifest.json and POST /api/trace (streamed). Only controls: Run, Back, Next, the box list.

const BOXES = [
  // number, exact diagram title, subtitle, colour, position on the 1536 x 1024 diagram [x0, y0, x1, y1]
  [1, "Data Sources (Adolescent Mental Health)", "", "#2f6db5", [5, 128, 270, 486]],
  [2, "Data Preprocessing & Anonymization", "", "#2e8b74", [279, 128, 517, 486]],
  [3, "Feature Representation", "", "#c77a1c", [526, 128, 777, 486]],
  [4, "Privacy-Preserving Mechanisms", "", "#c0392b", [786, 128, 1041, 486]],
  [5, "Quantum Encoding", "Classical Features → Quantum States", "#5b4bb7", [1049, 128, 1269, 486]],
  [6, "Quantum Model (Simulation)", "Variational Quantum Circuit (VQC / QNN)", "#1f5fa8", [1279, 128, 1533, 486]],
  [7, "Hybrid Quantum-Classical Learning", "", "#b0304a", [5, 512, 396, 756]],
  [8, "Simulation Environment", "", "#2e8b57", [404, 512, 796, 756]],
  [9, "Mental Health Analysis Tasks", "", "#3949ab", [804, 512, 1148, 756]],
  [10, "Evaluation & Comparison", "Compare with classical models and analyze the benefits of quantum AI.", "#1f6fb2", [1156, 512, 1533, 756]],
  [11, "Performance Metrics", "", "#2e7d5b", [5, 762, 362, 968]],
  [12, "Privacy Evaluation", "", "#c46a1a", [369, 762, 637, 968]],
  [13, "Scalability Analysis (Simulation)", "", "#c2185b", [644, 762, 953, 968]],
  [14, "Output & Visualization", "", "#00838f", [961, 762, 1243, 968]],
  [15, "Key Outcomes", "", "#4a3a9a", [1249, 762, 1533, 968]],
];
const STAGES = [1, 2, 3, 4, 5, 6, 7, 8, 9];          // boxes the live trace passes through, in diagram order
const PACE_MS = 350;                                   // display pacing between stage ticks; compute times shown are real
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
let manifest = null;
let current = 0;
let trace = {};
let running = false;
let traceError = null;

const src = (s) => (s ? `<div class="source">Source: ${esc(s)}</div>` : "");
const kv = (rows) => `<table class="kv">${rows.map((r) => `<tr><td>${esc(r[0])}</td><td>${esc(r[1])}</td></tr>`).join("")}</table>`;
const tbl = (cols, rows) => `<table><thead><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
  <tbody>${rows.map((r) => `<tr>${r.map((v) => `<td class="${/^[-+–≥<]?[\d.]/.test(String(v)) ? "num" : ""}">${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
const range = () => `indices ${manifest.trace.start} to ${manifest.trace.start + manifest.trace.count - 1}`;

function render(it, wide) {
  const cls = `card${wide ? " wide" : ""}`;
  switch (it.kind) {
    case "na":
      return `<section class="${cls} na"><h2>${esc(it.title)}</h2><div class="na-badge">Not available on this dataset</div><div>${esc(it.reason)}</div></section>`;
    case "group":
      if (it.items.every((x) => x.kind === "na")) return naCard(it.title, it.items, true);
      if (it.extra) {                                          // box 3: the next diagram item sits in the group's free columns
        const ex = it.extra;
        return `<section class="card wide"><h2>${esc(it.title)}</h2><div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px">
          ${it.items.map((x) => render(x, x.kind === "list" && x.entries.length > 12)).join("")}
          <section class="card" style="grid-column: span 2"><h2>${esc(ex.title)}</h2><div style="display:grid;grid-template-columns:minmax(0,2fr) minmax(0,3fr);gap:10px">
            <div><img class="fig" src="${esc(ex.src)}" alt="" style="width:100%;max-height:none">${src(ex.source)}</div>${ex.rows ? kv(ex.rows) : ""}</div></section></div></section>`;
      }
      return `<section class="card wide"><h2>${esc(it.title)}</h2><div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px">
        ${it.items.map((x) => render(x, x.kind === "list" && x.entries.length > 12)).join("")}</div></section>`;
    case "kv":
      return `<section class="${cls}"><h2>${esc(it.title)}</h2>${kv(it.rows)}${it.note ? `<div class="note">${esc(it.note)}</div>` : ""}${src(it.source)}</section>`;
    case "table":
      return `<section class="${cls}"><h2>${esc(it.title)}</h2>${tbl(it.columns, it.table)}${it.note ? `<div class="note">${esc(it.note)}</div>` : ""}${src(it.source)}</section>`;
    case "list":
      return `<section class="${cls}"><h2>${esc(it.title)}</h2><ul class="cols">${it.entries.map((e) => `<li>${esc(e[0])} <span class="code">${esc(e[1])}</span></li>`).join("")}</ul>${src(it.source)}</section>`;
    case "figure":
      return `<section class="${cls}${it.rows ? " span2" : ""}"><h2>${esc(it.title)}</h2><img class="fig" src="${esc(it.src)}" alt="${esc(it.title)}">${src(it.source)}${it.rows ? kv(it.rows) : ""}</section>`;
    case "figures":
      return `<section class="${cls}"><h2>${esc(it.title)}</h2><div style="display:grid;grid-template-columns:repeat(${it.figures.length},minmax(0,1fr));gap:10px">
        ${it.figures.map((f) => `<figure style="margin:0"><img class="fig" src="${esc(f.src)}" alt=""><div class="source">Source: ${esc(f.source)}</div></figure>`).join("")}</div></section>`;
    case "circuits":
      return `<section class="card wide"><h2>${esc(it.title)}</h2><div class="equation">|ψ⟩ = U(x) |0⟩</div><div class="circuits">
        ${it.circuits.map((c) => `<figure><img src="${esc(c.src)}" alt="${esc(c.encoding)} circuit"><figcaption><b>${esc(c.encoding)}</b>${c.gates !== null ? ` · ${c.gates} gates, depth ${c.depth}` : ""}</figcaption></figure>`).join("")}
        </div><div class="note">${esc(it.note)}</div>${src(it.source)}</section>`;
    case "loop":
      return `<section class="card wide"><h2>${esc(it.title)}</h2><div class="loop">
        <div class="node"><h3>Classical Optimizer (e.g., Adam, COBYLA)</h3>${it.optimizer.map((o) => `<div>${esc(o)}</div>`).join("")}</div>
        <div class="arrows"><div>Evaluate Loss →</div><div>← Update Parameters</div></div>
        <div class="node"><h3>Loss Function (Classification / Regression)</h3>${it.loss.map((o) => `<div>${esc(o)}</div>`).join("")}</div></div>
        <div class="note"><b>${esc(it.note)}</b></div>${src(it.source)}</section>`;
    case "twocol":
      return `<section class="card wide"><h2>${esc(it.title)}</h2><div class="twocol">
        <div><h3>Classical Baselines</h3>${kv(it.left)}</div><div><h3>Quantum Models</h3>${kv(it.right)}</div></div>
        <div class="note">${esc(it.note)}</div>${src(it.source)}</section>`;
    default:
      return `<section class="${cls}"><h2>${esc(it.title)}</h2></section>`;
  }
}

function naCard(title, items, wide) {
  return `<section class="card na${wide ? " wide" : ""}"><h2>${esc(title)}</h2><div class="na-badge">Not available on this dataset</div>
    <table class="kv">${items.map((x) => `<tr><td>${esc(x.title)}</td><td>${esc(x.reason)}</td></tr>`).join("")}</table></section>`;
}

// ---------------------------------------------------------------- live trace
function stageBody(st) {
  return `${st.detail ? `<div>${esc(st.detail)}</div>` : ""}${st.kv ? kv(st.kv) : ""}
    ${st.img ? `<img class="fig circuit-img" src="${esc(st.img)}" alt="circuit">` : ""}
    ${st.table ? `<div class="scroll">${tbl(st.table.columns, st.table.rows)}</div>` : ""}
    <div class="source">Computed live in ${st.ms} ms · ${esc(st.source)}</div>`;
}

function stageCard(b) {
  const [, title, , colour] = BOXES[b - 1];
  const st = trace[b];
  const state = st ? "done" : running ? "waiting" : "idle";
  const span = b === 1 ? " wide" : b === 9 ? " span2" : "";      // the two 50-row tables get room
  return `<section class="card stage ${state}${span}" id="stage-${b}" style="--box:${colour}">
    <h2><span class="tick">${st ? "✓" : "○"}</span> ${b}. ${esc(title)}</h2>
    ${st ? stageBody(st) : `<div class="muted">${running ? "Waiting for this stage…" : "Not run yet."}</div>`}</section>`;
}

function overview() {
  const t = manifest.trace;
  return `<section class="card wide intro"><div class="statement">${t.count} real held-out records, ${range()}, never seen during training</div>
      <div>Each architecture box lights up as its stage completes, in diagram order. Inference only: nothing is trained, and no metric is computed from these ${t.count} records.</div>
      <div class="run-row"><button class="primary" id="run" type="button"${running ? " disabled" : ""}>Run</button>
      <span id="run-status">${traceError ? `<b class="err">Failed: ${esc(traceError)}</b>` : trace[9] ? `All ${STAGES.length} stages completed in ${trace[9].total_ms} ms of compute (display paced at ${PACE_MS / 1000} s per stage).` : running ? "Running…" : ""}</span></div></section>
    <div class="stages">${STAGES.map(stageCard).join("")}</div>`;
}

async function runTrace() {
  if (running) return;
  running = true; trace = {}; traceError = null;
  show(0); markDone();
  const queue = [];
  let finished = false;
  const reveal = (async () => {                                  // tick stages one by one, in order
    while (!finished || queue.length) {
      if (queue.length) {
        const st = queue.shift();
        if (st.error) { traceError = st.error; break; }
        trace[st.box] = st;
        markDone(st.box);
        if (current === 0) { const el = $(`stage-${st.box}`); if (el) el.outerHTML = stageCard(st.box); }
        await new Promise((r) => setTimeout(r, PACE_MS));
      } else await new Promise((r) => setTimeout(r, 50));
    }
  })();
  try {
    const res = await fetch("/api/trace", { method: "POST" });
    if (!res.ok) throw new Error((await res.json()).error || `HTTP ${res.status}`);
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) { const line = buf.slice(0, i); buf = buf.slice(i + 1); if (line.trim()) queue.push(JSON.parse(line)); }
    }
  } catch (e) { queue.push({ error: e.message }); }
  finished = true;
  await reveal;
  if (!traceError && !trace[9]) traceError = "the stream ended before box 9";
  running = false;
  markDone();
  if (current === 0) show(0);
}

function markDone(latest) {
  document.querySelectorAll("#diagram .done").forEach((d) => d.remove());
  const k = $("diagram-img").clientWidth / 1536;
  for (const b of Object.keys(trace).map(Number)) {
    const pos = BOXES[b - 1][4], d = document.createElement("div");
    d.className = "done" + (b === latest ? " latest" : "");
    Object.assign(d.style, { left: `${pos[0] * k}px`, top: `${pos[1] * k}px`, width: `${(pos[2] - pos[0]) * k}px`, height: `${(pos[3] - pos[1]) * k}px` });
    d.textContent = "✓";
    $("diagram").appendChild(d);
  }
}

function livePanel(n) {
  const head = `<h2 class="panel-h">Live run · ${manifest.trace.count} real held-out records (${range()})</h2>`;
  let body;
  if (STAGES.includes(n)) body = trace[n] ? stageBody(trace[n]) : `<div class="note">Not run yet. Press <b>Run</b> on the Overview screen (top of the box list).</div>`;
  else if (n === 14) body = `<div class="note">No live stage in this box: the trace's outputs are shown in boxes 1–9. The figures below come from results/.</div>`;
  else body = `<div class="note">No live stage in this box. The ${manifest.trace.count} live predictions (box 9) are mechanism only: only ${manifest.trace.n_positive} of the ${manifest.trace.count} have a current depression diagnosis, so no accuracy, precision, recall, F1 or AUC is computed from them. Every metric below is read from results/.</div>`;
  return `<section class="panel live">${head}${body}</section>`;
}

// ---------------------------------------------------------------- navigation
function show(n) {
  current = Math.max(0, Math.min(15, n));
  $("back").disabled = current === 0;
  $("next").disabled = current === 15;
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("on", Number(a.dataset.n) === current));
  const mark = $("diagram-mark");
  if (current === 0) {
    document.documentElement.style.setProperty("--box", "#111318");
    $("box-count").textContent = "Overview";
    $("box-title").textContent = "Live trace through the architecture";
    $("box-subtitle").textContent = `${manifest.trace.count} real held-out records, ${range()}`;
    mark.style.display = "none";
    $("content").innerHTML = overview();
    $("run").addEventListener("click", runTrace);
    if (location.hash !== "#overview") history.replaceState(null, "", "#overview");
    window.scrollTo(0, 0);
    return;
  }
  const [num, title, sub, colour, pos] = BOXES[current - 1];
  document.documentElement.style.setProperty("--box", colour);
  $("box-count").textContent = `Box ${num} of 15`;
  $("box-title").textContent = `${num}. ${title}`;
  $("box-subtitle").textContent = sub;
  const k = $("diagram-img").clientWidth / 1536;
  mark.style.display = "block";
  Object.assign(mark.style, { left: `${pos[0] * k}px`, top: `${pos[1] * k}px`, width: `${(pos[2] - pos[0]) * k}px`, height: `${(pos[3] - pos[1]) * k}px` });
  const sec = manifest.sections.find((s) => s.box === current);
  let items = sec.items;
  const gi = items.findIndex((it, i) => it.kind === "group" && items[i + 1] && items[i + 1].kind === "figure");
  if (current === 3 && gi >= 0) items = [...items.slice(0, gi), { ...items[gi], extra: items[gi + 1] }, ...items.slice(gi + 2)];
  const wideKinds = new Set(["circuits", "loop", "twocol", "group"]);
  const nas = items.filter((it) => it.kind === "na");
  const collapse = current === 9 && nas.length >= 3;          // box 9: the three text-only tasks in one card
  let cards = items.filter((it) => !(collapse && it.kind === "na"))
    .map((it) => render(it, wideKinds.has(it.kind) || (it.kind === "list" && it.entries.length > 12) || (it.kind === "table" && it.columns.length >= 7))).join("");
  if (collapse) cards += naCard("Topic and trend analysis · Emotion and sentiment analysis · Personalized insights (simulated)", nas, false);
  $("content").innerHTML = `${livePanel(current)}<section class="panel full"><h2 class="panel-h">${esc(sec.full_heading)}</h2><div class="grid">${cards}</div></section>`;
  if (location.hash !== `#box-${current}`) history.replaceState(null, "", `#box-${current}`);
  window.scrollTo(0, 0);
}

(async function init() {
  $("nav").innerHTML = `<a href="#overview" data-n="0"><b>▶</b><span>Overview — live trace</span></a>`
    + BOXES.map(([n, t]) => `<a href="#box-${n}" data-n="${n}"><b>${n}</b><span>${esc(t)}</span></a>`).join("");
  document.querySelectorAll("#nav a").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); show(Number(a.dataset.n)); }));
  $("back").addEventListener("click", () => show(current - 1));
  $("next").addEventListener("click", () => show(current + 1));
  try {
    const res = await fetch("manifest.json");
    if (!res.ok) throw new Error(`manifest.json not found (HTTP ${res.status}) — run: python scripts/build_demo_manifest.py`);
    manifest = await res.json();
  } catch (e) { $("error").hidden = false; $("error").textContent = e.message; return; }
  const img = $("diagram-img");
  const start = () => show(Number((location.hash.match(/box-(\d+)/) || [])[1]) || 0);
  if (img.complete) start(); else img.addEventListener("load", start);
})();
