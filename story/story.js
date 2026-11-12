"use strict";
// Replays the recorded Experiment 4 run. All numbers come from /api/story (results/story/story.json).

const COLOR = { accent: "#4269d0", quantum: "#6cc5b0", safe: "#3ca951", ink: "#1b1e23", muted: "#6b6f76", line: "#e4e4e7" };
const SCHOOLS = ["Northside High", "Riverside Academy", "Eastgate School"];
const SVGNS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);
const f = (v, d = 3) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "–" : Number(v).toFixed(d));
const fmtInt = (n) => Number(n).toLocaleString("en-US");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let data = null;
let screen = 1;
let trained = false;
let running = false;

function el(tag, attrs = {}, text) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}
const lock = `<svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true" class="shrink-0"><rect x="4" y="9" width="12" height="9" rx="2" fill="none" stroke="#1b1e23" stroke-width="1.5"/><path d="M7 9V6.5a3 3 0 0 1 6 0V9" fill="none" stroke="#1b1e23" stroke-width="1.5"/></svg>`;

// ------------------------------------------------------------------ screens
function show(n) {
  screen = n;
  document.querySelectorAll("[data-screen]").forEach((s) => { s.hidden = Number(s.dataset.screen) !== n; });
  $("step-label").textContent = `${n} / 4`;
  $("btn-back").style.visibility = n === 1 ? "hidden" : "visible";
  $("btn-next").textContent = n === 4 ? "Start over" : "Next";
  $("btn-next").disabled = n === 2 && !trained;
  $("btn-next").style.opacity = $("btn-next").disabled ? "0.4" : "1";
}
$("btn-next").addEventListener("click", () => { if (!$("btn-next").disabled) show(screen === 4 ? 1 : screen + 1); });
$("btn-back").addEventListener("click", () => { if (!running) show(Math.max(1, screen - 1)); });

function renderStatic() {
  const r = data && data.replay;
  // Screen 1
  $("s1-schools").innerHTML = SCHOOLS.map((name, i) => `
    <div>
      <div class="card">
        <div class="text-[24px] font-semibold">${name}</div>
        <div class="mt-3"><span class="num text-[32px]">${r ? fmtInt(r.schools[i].rows) : "–"}</span> <span class="text-[18px] text-muted">student records</span></div>
        <div class="flex items-center gap-2 mt-5 pt-4 border-t border-line text-[18px]">${lock}<span>Data cannot be shared</span></div>
      </div>
      <div class="mt-4 text-[20px] text-muted">Model trained alone: <span class="num">AUC ${data ? f(data.replay_alone_auc_eps4[i], 3) : "–"}</span></div>
    </div>`).join("");
  if (r) {
    const st = r.schools.map((s) => s.n_states);
    $("s1-note").textContent = `Simulation on real data: each “school” holds one third of the NSCH 2024 survey records, grouped by U.S. state `
      + `(${Math.min(...st)}–${Math.max(...st)} states each). “Trained alone” uses the same privacy protection (ε = 4) as the shared model, `
      + `and is scored on the same ${fmtInt(r.n_val)} test records.`;
  }
  // Screen 3
  if (data) {
    const alone = data.replay_alone_auc_eps4;
    const big = (label, value, sub, color) => `<div class="card"><div class="text-[20px] text-muted">${label}</div>
      <div class="num text-[56px] leading-tight mt-2 whitespace-nowrap" style="color:${color}">${value}</div><div class="small mt-2">${sub}</div></div>`;
    $("s3-numbers").innerHTML =
      big("Alone", `${f(Math.min(...alone), 2)}–${f(Math.max(...alone), 2)}`, `AUC of each school's own model (${alone.map((a) => f(a, 3)).join(", ")})`, COLOR.muted) +
      big("Together", f(r.final_auc, 3), "AUC of the shared model after 30 rounds", COLOR.accent) +
      big("Privacy guarantee", `ε = ${f(r.logged_epsilon_spent, 1)}`, `differential privacy for every school (δ = 10⁻⁵)`, COLOR.ink);
    const t = data.ten_folds;
    $("s3-verified").textContent = `Verified on ${fmtInt(r.n_val)} student records that were never used for training. `
      + `Repeated on all ${t.k} test splits: alone ${f(t.alone_eps4_mean, 2)} on average, together ${f(t.federated_eps4_mean, 2)}; `
      + `together was better in ${t.federated_beats_alone_folds} of ${t.k} (corrected t-test p = ${f(t.p_nb_t, 3)}).`;
    // Screen 4
    const q = data.quantum_vs_classical;
    const bar = (label, v, color) => `<div><div class="flex items-baseline justify-between text-[20px]"><span>${label}</span><span class="num text-[48px]">${f(v, 3)}</span></div>
      <div class="h-6 bg-[#f4f4f5] rounded mt-2"><div class="h-6 rounded" style="width:${v * 100}%;background:${color}"></div></div></div>`;
    $("s4-bars").innerHTML = bar("Classical model", q.classical_auc, COLOR.accent) + bar("Quantum model", q.quantum_auc, COLOR.quantum)
      + `<p class="text-[22px] mt-4">Difference not statistically significant <span class="small">(corrected t-test, Holm p = ${f(q.holm_p, 2)}, ${q.k} folds)</span></p>`;
  }
}

// ------------------------------------------------------------------ screen 2: diagram
const SERVER = { x: 350, y: 64 };
const SPOTS = [117, 350, 583];
let dots = [], statusText = [], schoolRects = [], serverStatus;

function drawDiagram() {
  const svg = $("diagram");
  svg.innerHTML = "";
  SPOTS.forEach((x) => svg.appendChild(el("line", { x1: x, y1: 238, x2: SERVER.x, y2: 96, stroke: COLOR.line, "stroke-width": 2 })));
  // labels on the outer links: update size and records sent
  const label = (x, anchor) => {
    const g = el("g", { "font-family": "JetBrains Mono, monospace", "font-size": 15, "text-anchor": anchor });
    g.appendChild(el("text", { x, y: 140, fill: COLOR.accent }, `model update · ${f((data ? data.replay.bytes_per_update : 768) / 1000, 1)} KB`));
    g.appendChild(el("text", { x, y: 164, fill: COLOR.safe, "font-weight": 500 }, "student records sent: 0"));
    return g;
  };
  svg.appendChild(label(14, "start"));
  svg.appendChild(label(686, "end"));
  // server
  svg.appendChild(el("rect", { x: SERVER.x - 110, y: 20, width: 220, height: 76, rx: 8, fill: "white", stroke: COLOR.ink, "stroke-width": 1.5 }));
  svg.appendChild(el("text", { x: SERVER.x, y: 52, "text-anchor": "middle", "font-size": 20, "font-weight": 600, fill: COLOR.ink }, "Server"));
  serverStatus = el("text", { x: SERVER.x, y: 78, "text-anchor": "middle", "font-size": 15, fill: COLOR.muted }, "waiting for updates");
  svg.appendChild(serverStatus);
  // schools
  schoolRects = []; statusText = []; dots = [];
  SPOTS.forEach((x, i) => {
    const rect = el("rect", { x: x - 105, y: 238, width: 210, height: 86, rx: 8, fill: "white", stroke: COLOR.line, "stroke-width": 1.5 });
    rect.style.transition = "stroke 150ms";
    svg.appendChild(rect); schoolRects.push(rect);
    svg.appendChild(el("text", { x, y: 266, "text-anchor": "middle", "font-size": 17, "font-weight": 600, fill: COLOR.ink }, SCHOOLS[i]));
    const g = el("g");
    g.innerHTML = `<rect x="${x - 72}" y="279" width="9" height="7" rx="1.5" fill="none" stroke="${COLOR.ink}" stroke-width="1.2"/><path d="M${x - 70} 279v-2.5a2.5 2.5 0 0 1 5 0V279" fill="none" stroke="${COLOR.ink}" stroke-width="1.2"/>`;
    svg.appendChild(g);
    svg.appendChild(el("text", { x: x - 58, y: 287, "font-size": 15, fill: COLOR.ink }, "records stay here"));
    const st = el("text", { x, y: 311, "text-anchor": "middle", "font-size": 15, fill: COLOR.muted }, "ready");
    svg.appendChild(st); statusText.push(st);
    const up = el("circle", { r: 7, fill: COLOR.accent, cx: x, cy: 238, opacity: 0 });
    const down = el("circle", { r: 7, fill: COLOR.quantum, cx: SERVER.x, cy: 96, opacity: 0 });
    svg.appendChild(up); svg.appendChild(down); dots.push({ up, down, x });
  });
}

function animate(ms, step) {
  return new Promise((resolve) => {
    const t0 = performance.now();
    const tick = (now) => {
      const p = Math.min(1, (now - t0) / ms);
      step(p);
      if (p < 1) requestAnimationFrame(tick); else resolve();
    };
    requestAnimationFrame(tick);
  });
}

// ------------------------------------------------------------------ screen 2: chart, meter, log
const CH = { w: 460, h: 165, l: 48, r: 16, t: 14, b: 32, y0: 0.8, y1: 0.89 };
const cx = (round) => CH.l + ((round - 1) / 29) * (CH.w - CH.l - CH.r);
const cy = (auc) => CH.t + (1 - (auc - CH.y0) / (CH.y1 - CH.y0)) * (CH.h - CH.t - CH.b);
let line, lastDot;

function drawChart() {
  const svg = $("chart");
  svg.innerHTML = "";
  for (let v = 0.8; v <= 0.8901; v += 0.02) {
    svg.appendChild(el("line", { x1: CH.l, x2: CH.w - CH.r, y1: cy(v), y2: cy(v), stroke: COLOR.line }));
    svg.appendChild(el("text", { x: CH.l - 8, y: cy(v) + 5, "text-anchor": "end", "font-size": 14, fill: COLOR.muted, "font-family": "JetBrains Mono, monospace" }, v.toFixed(2)));
  }
  [1, 15, 30].forEach((r) => svg.appendChild(el("text", { x: cx(r), y: CH.h - 8, "text-anchor": r === 1 ? "start" : r === 30 ? "end" : "middle", "font-size": 14, fill: COLOR.muted, "font-family": "JetBrains Mono, monospace" }, `round ${r}`)));
  if (data) {
    const best = Math.max(...data.replay_alone_auc_eps4);
    svg.appendChild(el("line", { x1: CH.l, x2: CH.w - CH.r, y1: cy(best), y2: cy(best), stroke: COLOR.muted, "stroke-dasharray": "5 4" }));
    svg.appendChild(el("text", { x: CH.w - CH.r, y: cy(best) + 18, "text-anchor": "end", "font-size": 14, fill: COLOR.muted }, `best school alone ${f(best, 3)}`));
  }
  line = el("polyline", { fill: "none", stroke: COLOR.accent, "stroke-width": 2.5, points: "" });
  lastDot = el("circle", { r: 5, fill: COLOR.accent, opacity: 0 });
  svg.appendChild(line); svg.appendChild(lastDot);
}

function resetTraining() {
  drawDiagram(); drawChart();
  $("log").innerHTML = "";
  $("eps-bar").style.width = "0%";
  $("eps-text").textContent = "0.0 / 4.0";
  $("auc-now").textContent = "–";
  $("records-sent").textContent = "0";
}

async function runTraining() {
  if (running || !data) return;
  running = true;
  $("btn-start").disabled = true; $("btn-start").style.opacity = "0.4";
  $("btn-next").disabled = true; $("btn-next").style.opacity = "0.4";
  resetTraining();
  const r = data.replay;
  const sigma = r.sigma_per_school.reduce((a, b) => a + b, 0) / r.sigma_per_school.length;
  const pts = [];
  const started = performance.now();
  for (const round of r.rounds) {
    const dur = round.round <= 3 ? 1600 : 520;
    // 1. local training
    schoolRects.forEach((s) => s.setAttribute("stroke", COLOR.accent));
    statusText.forEach((s) => { s.textContent = "training locally…"; s.setAttribute("fill", COLOR.accent); });
    serverStatus.textContent = "waiting for updates";
    await sleep(dur * 0.3);
    // 2. updates travel to the server
    dots.forEach((d) => d.up.setAttribute("opacity", 1));
    await animate(dur * 0.25, (p) => dots.forEach((d) => {
      d.up.setAttribute("cx", d.x + (SERVER.x - d.x) * p); d.up.setAttribute("cy", 238 + (96 - 238) * p);
    }));
    dots.forEach((d) => d.up.setAttribute("opacity", 0));
    // 3. aggregate + noise
    serverStatus.textContent = "aggregating + adding privacy noise";
    serverStatus.setAttribute("fill", COLOR.ink);
    await sleep(dur * 0.2);
    // 4. improved model returns
    serverStatus.textContent = "sending improved model";
    statusText.forEach((s) => { s.textContent = "receiving improved model"; s.setAttribute("fill", COLOR.quantum); });
    dots.forEach((d) => d.down.setAttribute("opacity", 1));
    await animate(dur * 0.25, (p) => dots.forEach((d) => {
      d.down.setAttribute("cx", SERVER.x + (d.x - SERVER.x) * p); d.down.setAttribute("cy", 96 + (238 - 96) * p);
    }));
    dots.forEach((d) => d.down.setAttribute("opacity", 0));
    schoolRects.forEach((s) => s.setAttribute("stroke", COLOR.line));
    // live updates from the recorded run
    pts.push(`${cx(round.round)},${cy(round.val_auc)}`);
    line.setAttribute("points", pts.join(" "));
    lastDot.setAttribute("cx", cx(round.round)); lastDot.setAttribute("cy", cy(round.val_auc)); lastDot.setAttribute("opacity", 1);
    $("auc-now").textContent = `round ${round.round}: ${f(round.val_auc, 3)}`;
    $("eps-bar").style.width = `${Math.min(100, (round.eps_spent_max / 4) * 100)}%`;
    $("eps-text").textContent = `${f(round.eps_spent_max, 1)} / 4.0`;
    const row = document.createElement("div");
    row.innerHTML = `Round ${round.round} · aggregated 3 updates · noise σ=${f(sigma, 2)} · raw records transmitted: <span style="color:${COLOR.safe};font-weight:500">0</span>`;
    $("log").appendChild(row);
    $("log").scrollTop = $("log").scrollHeight;
  }
  statusText.forEach((s) => { s.textContent = "has the shared model"; s.setAttribute("fill", COLOR.ink); });
  serverStatus.textContent = "done · 30 rounds";
  window.__replaySeconds = (performance.now() - started) / 1000;
  serverStatus.setAttribute("fill", COLOR.muted);
  $("auc-now").textContent = `final model: ${f(r.final_auc, 3)}`;
  running = false; trained = true;
  window.__trainingDone = true;
  $("btn-start").disabled = false; $("btn-start").style.opacity = "1"; $("btn-start").textContent = "Replay training";
  $("btn-next").disabled = false; $("btn-next").style.opacity = "1";
}
$("btn-start").addEventListener("click", runTraining);

// ------------------------------------------------------------------ start
(async function init() {
  try {
    const res = await fetch("/api/story");
    const body = await res.json().catch(() => null);
    if (!res.ok) throw new Error((body && body.error) || `HTTP ${res.status}`);
    data = body;
  } catch (e) {
    $("data-error").hidden = false;
    $("data-error").textContent = `Replay data unavailable: ${e.message} The screens below show their text without numbers.`;
    trained = true;  // let the presenter move through the screens anyway
  }
  renderStatic();
  resetTraining();
  if (!data) { $("btn-start").disabled = true; $("btn-start").style.opacity = "0.4"; }
  show(1);
})();
