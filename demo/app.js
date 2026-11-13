"use strict";
// Architecture walkthrough. Reads only manifest.json and POST /api/live. No configuration controls.

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
const FLOW = "Flow: 1 → 2 → 3 → 4 → 5 → 6, then 7 → 8 → 9 → 10, then 11 · 12 · 13 · 14 · 15";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
let manifest = null;
let current = 1;

const src = (s) => (s ? `<div class="source">Source: ${esc(s)}</div>` : "");
const kv = (rows) => `<table class="kv">${rows.map((r) => `<tr><td>${esc(r[0])}</td><td>${esc(r[1])}</td></tr>`).join("")}</table>`;
const tbl = (cols, rows) => `<table><thead><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
  <tbody>${rows.map((r) => `<tr>${r.map((v) => `<td class="${/^[-+–≥<]?[\d.]/.test(String(v)) ? "num" : ""}">${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;

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

function liveCard() {
  const l = manifest.live;
  return `<section class="card wide" id="live"><h2>Live prediction on one real record</h2>
    <div>Dataset row <span class="num">${l.row_index}</span> — ${esc(l.split)}. No model has ever seen this record.</div>
    <div style="margin-top:8px"><button class="primary" id="run" type="button">Run live prediction</button></div>
    <div id="live-out"></div><div class="source">Source: demo_app.py POST /api/live · models/config_models.pkl · models/screening_models.pkl</div></section>`;
}

async function runLive() {
  const btn = $("run");
  btn.disabled = true;
  $("live-out").innerHTML = "<div class='note'>Running…</div>";
  try {
    const res = await fetch("/api/live", { method: "POST" });
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || `HTTP ${res.status}`);
    $("live-out").innerHTML = `<div class="note">Dataset row <b class="num">${d.row_index}</b> · risk band <span class="band">${esc(d.band.toUpperCase())}</span>
      · survey answer: ${esc(d.survey_label)} · computed in <span class="num">${d.ms}</span> ms</div>
      <div class="live-steps">${d.steps.map((s) => `<div class="step"><h3>${esc(s.title)}</h3><div class="muted">${esc(s.detail)}</div><table class="kv stack">${s.rows.map((r) => `<tr><td>${esc(r[0])}</td><td class="num">${esc(r[1])}</td></tr>`).join("")}</table></div>`).join("")}</div>`;
  } catch (e) {
    $("live-out").innerHTML = `<div id="error">${esc(e.message)}</div>`;
  }
  btn.disabled = false;
}

function show(n) {
  current = Math.max(1, Math.min(15, n));
  const [num, title, sub, colour, pos] = BOXES[current - 1];
  document.documentElement.style.setProperty("--box", colour);
  $("box-count").textContent = `Box ${num} of 15`;
  $("box-title").textContent = `${num}. ${title}`;
  $("box-subtitle").textContent = sub;
  $("flow").textContent = FLOW;
  const img = $("diagram-img"), k = img.clientWidth / 1536;
  Object.assign($("diagram-mark").style, { left: `${pos[0] * k}px`, top: `${pos[1] * k}px`, width: `${(pos[2] - pos[0]) * k}px`, height: `${(pos[3] - pos[1]) * k}px` });
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("on", Number(a.dataset.n) === current));
  $("back").disabled = current === 1;
  $("next").disabled = current === 15;
  const sec = manifest.sections.find((s) => s.box === current);
  let items = sec.items;
  const gi = items.findIndex((it, i) => it.kind === "group" && items[i + 1] && items[i + 1].kind === "figure");
  if (current === 3 && gi >= 0 && items[gi + 1] && items[gi + 1].kind === "figure") {   // Dimensionality Reduction beside Structured Features
    items = [...items.slice(0, gi), { ...items[gi], extra: items[gi + 1] }, ...items.slice(gi + 2)];
  }
  const wideKinds = new Set(["circuits", "loop", "twocol", "group"]);
  const nas = items.filter((it) => it.kind === "na");
  const collapse = current === 9 && nas.length >= 3;          // box 9: the three text-only tasks in one card
  let html = items.filter((it) => !(collapse && it.kind === "na"))
    .map((it) => render(it, wideKinds.has(it.kind) || (it.kind === "list" && it.entries.length > 12))).join("");
  if (collapse) html += naCard("Topic and trend analysis · Emotion and sentiment analysis · Personalized insights (simulated)", nas, false);
  if (current === 9) html += liveCard();
  $("content").innerHTML = html;
  if (current === 9) $("run").addEventListener("click", runLive);
  if (location.hash !== `#box-${current}`) history.replaceState(null, "", `#box-${current}`);
  window.scrollTo(0, 0);
}

(async function init() {
  $("nav").innerHTML = BOXES.map(([n, t]) => `<a href="#box-${n}" data-n="${n}"><b>${n}</b><span>${esc(t)}</span></a>`).join("");
  document.querySelectorAll("#nav a").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); show(Number(a.dataset.n)); }));
  $("back").addEventListener("click", () => show(current - 1));
  $("next").addEventListener("click", () => show(current + 1));
  try {
    const res = await fetch("manifest.json");
    if (!res.ok) throw new Error(`manifest.json not found (HTTP ${res.status}) — run: python scripts/build_demo_manifest.py`);
    manifest = await res.json();
  } catch (e) { $("error").hidden = false; $("error").textContent = e.message; return; }
  const img = $("diagram-img");
  const start = () => show(Number((location.hash.match(/box-(\d+)/) || [])[1]) || 1);
  if (img.complete) start(); else img.addEventListener("load", start);
})();
