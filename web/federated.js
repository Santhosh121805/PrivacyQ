"use strict";
// Overview › How it works — replays the recorded Experiment 4 federated run (adapted from the three-schools story,
// story/story.js). Every number comes from /api/federated: results/story/story.json (per-round ROC-AUC and ε, school
// sizes, noise multiplier) and the measured serialised update size from Experiment 7. Nothing is simulated.
(function () {
  const C = { accent: "#4269d0", quantum: "#6cc5b0", ink: "#1b1e23", muted: "#6b6f76", line: "#e4e4e7", surface: "#f9f9f9" };
  const SCHOOLS = ["Northside High", "Riverside Academy", "Eastgate School"];
  const NS = "http://www.w3.org/2000/svg";
  const TARGET_EPS = 4;
  const ROUND_MS = 660;                                   // 30 recorded rounds in about 20 s
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (id) => document.getElementById(id);
  const f = (v, d = 3) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "–" : Number(v).toFixed(d));
  const fmtInt = (n) => Number(n).toLocaleString("en-US");
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const lock = `<svg width="14" height="14" viewBox="0 0 20 20" aria-hidden="true" class="shrink-0"><rect x="4" y="9" width="12" height="9" rx="2" fill="none" stroke="${C.ink}" stroke-width="1.6"/><path d="M7 9V6.5a3 3 0 0 1 6 0V9" fill="none" stroke="${C.ink}" stroke-width="1.6"/></svg>`;

  function el(tag, attrs = {}, text) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function animate(ms, step) {
    if (reduceMotion || ms <= 0) { step(1); return Promise.resolve(); }
    return new Promise((resolve) => {
      const t0 = performance.now();
      const tick = (now) => {
        const p = Math.min(1, (now - t0) / ms);
        step(p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2);   // ease-in-out
        if (p < 1) requestAnimationFrame(tick); else resolve();
      };
      requestAnimationFrame(tick);
    });
  }

  let data = null, running = false;
  let links = [], dots = [], schoolStatus = [], serverBox, serverStatus, chartLine, chartDot;
  const SERVER = { x: 280, top: 24, bottom: 88 };
  const SPOTS = [100, 280, 460];
  const SCHOOL_TOP = 214;

  // ---------------------------------------------------------------- school cards
  function renderSchools(state) {
    const r = data.replay;
    $("fed-schools").innerHTML = SCHOOLS.map((name, i) => `
      <div class="card py-4 transition-colors ${state === "training" ? "border-accent" : ""}" data-school="${i}">
        <div class="text-t15 font-semibold">${name}</div>
        <div class="mt-1"><span class="num text-t20">${fmtInt(r.schools[i].rows)}</span> <span class="text-t13 text-muted">student records</span></div>
        <div class="flex items-center gap-2 mt-3 pt-3 border-t border-line text-t13">${lock}<span>Data cannot be shared</span></div>
      </div>`).join("");
  }
  function setSchoolsTraining(on) {
    document.querySelectorAll("#fed-schools [data-school]").forEach((c) => c.classList.toggle("border-accent", on));
  }

  // ---------------------------------------------------------------- diagram
  function drawDiagram() {
    const svg = $("fed-diagram");
    svg.innerHTML = "";
    const defs = el("defs");
    for (const [id, color] of [["fed-arrow-up", C.accent], ["fed-arrow-down", C.quantum]]) {
      const m = el("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" });
      m.appendChild(el("path", { d: "M0,1 L9,5 L0,9 z", fill: color }));
      defs.appendChild(m);
    }
    svg.appendChild(defs);
    links = []; dots = []; schoolStatus = [];
    SPOTS.forEach((x) => {
      const base = el("line", { x1: x, y1: SCHOOL_TOP, x2: SERVER.x, y2: SERVER.bottom, stroke: C.line, "stroke-width": 1.5 });
      const up = el("line", { x1: x, y1: SCHOOL_TOP, x2: SERVER.x, y2: SERVER.bottom + 4, stroke: C.accent, "stroke-width": 1.5, "marker-end": "url(#fed-arrow-up)", opacity: 0 });
      const down = el("line", { x1: SERVER.x, y1: SERVER.bottom, x2: x, y2: SCHOOL_TOP - 4, stroke: C.quantum, "stroke-width": 1.5, "marker-end": "url(#fed-arrow-down)", opacity: 0 });
      svg.append(base, up, down);
      links.push({ up, down });
    });
    // labels: what travels, and what does not
    const kb = `${f(data.update_bytes / 1000, 1)} KB`;
    svg.appendChild(el("text", { x: 12, y: 126, fill: C.accent, "font-size": 13, "font-family": "JetBrains Mono, monospace" }, `model update · ${kb}`));
    svg.appendChild(el("rect", { x: 10, y: 136, width: 196, height: 24, rx: 12, fill: "white", stroke: C.quantum, "stroke-width": 1.5 }));
    svg.appendChild(el("text", { x: 22, y: 152.5, fill: C.ink, "font-size": 13, "font-weight": 600, "font-family": "JetBrains Mono, monospace" }, "student records sent: 0"));
    const back = el("g", { "font-size": 13, "font-family": "Inter, sans-serif" });
    back.appendChild(el("circle", { cx: 438, cy: 131, r: 4, fill: C.quantum }));
    back.appendChild(el("text", { x: 448, y: 135, fill: C.ink }, "improved model"));
    svg.appendChild(back);
    // server
    serverBox = el("rect", { x: SERVER.x - 90, y: SERVER.top, width: 180, height: SERVER.bottom - SERVER.top, rx: 8, fill: "white", stroke: C.ink, "stroke-width": 1.2 });
    serverBox.style.transition = "stroke 150ms";
    svg.appendChild(serverBox);
    svg.appendChild(el("text", { x: SERVER.x, y: 52, "text-anchor": "middle", "font-size": 16, "font-weight": 600, fill: C.ink }, "Server"));
    serverStatus = el("text", { x: SERVER.x, y: 74, "text-anchor": "middle", "font-size": 12, fill: C.muted }, "waiting for updates");
    svg.appendChild(serverStatus);
    // schools
    SPOTS.forEach((x, i) => {
      svg.appendChild(el("rect", { x: x - 82, y: SCHOOL_TOP, width: 164, height: 72, rx: 8, fill: "white", stroke: C.line, "stroke-width": 1.2 }));
      svg.appendChild(el("text", { x, y: SCHOOL_TOP + 26, "text-anchor": "middle", "font-size": 14, "font-weight": 600, fill: C.ink }, SCHOOLS[i]));
      const st = el("text", { x, y: SCHOOL_TOP + 52, "text-anchor": "middle", "font-size": 12, fill: C.muted }, "records stay here");
      svg.appendChild(st); schoolStatus.push(st);
      const up = el("circle", { r: 4, fill: C.accent, cx: x, cy: SCHOOL_TOP, opacity: 0 });
      const down = el("circle", { r: 4, fill: C.quantum, cx: SERVER.x, cy: SERVER.bottom, opacity: 0 });
      svg.append(up, down);
      dots.push({ up, down, x });
    });
  }

  // ---------------------------------------------------------------- chart
  let CH;
  function drawChart() {
    const svg = $("fed-chart");
    svg.innerHTML = "";
    const r = data.replay, best = Math.max(...data.alone_auc_eps4);
    const vals = r.rounds.map((d) => d.val_auc).concat([best]);
    const y0 = Math.floor((Math.min(...vals) - 0.005) * 100) / 100, y1 = Math.ceil((Math.max(...vals) + 0.005) * 100) / 100;
    const n = r.rounds.length;
    CH = { w: 360, h: 150, l: 40, r: 10, t: 10, b: 26, y0, y1, n };
    const cx = (round) => CH.l + ((round - 1) / (n - 1)) * (CH.w - CH.l - CH.r);
    const cy = (v) => CH.t + (1 - (v - y0) / (y1 - y0)) * (CH.h - CH.t - CH.b);
    CH.cx = cx; CH.cy = cy;
    const mono = { "font-size": 11, fill: C.muted, "font-family": "JetBrains Mono, monospace" };
    for (let v = y0; v <= y1 + 1e-9; v += 0.02) {
      svg.appendChild(el("line", { x1: CH.l, x2: CH.w - CH.r, y1: cy(v), y2: cy(v), stroke: C.line }));
      svg.appendChild(el("text", { x: CH.l - 6, y: cy(v) + 4, "text-anchor": "end", ...mono }, v.toFixed(2)));
    }
    [1, Math.round(n / 2), n].forEach((rd, k) => svg.appendChild(el("text", { x: cx(rd), y: CH.h - 6, "text-anchor": ["start", "middle", "end"][k], ...mono }, `round ${rd}`)));
    svg.appendChild(el("line", { x1: CH.l, x2: CH.w - CH.r, y1: cy(best), y2: cy(best), stroke: C.muted, "stroke-dasharray": "4 4" }));
    svg.appendChild(el("text", { x: cx(4), y: cy(best) + 13, "text-anchor": "start", ...mono }, `best school alone, ε = 4: ${f(best, 3)}`));
    chartLine = el("polyline", { fill: "none", stroke: C.accent, "stroke-width": 2, points: "" });
    chartDot = el("circle", { r: 3.5, fill: C.accent, opacity: 0 });
    svg.append(chartLine, chartDot);
  }

  function reset() {
    drawDiagram(); drawChart();
    $("fed-log").innerHTML = "";
    $("fed-eps-bar").style.width = "100%";
    $("fed-eps-text").textContent = `ε ${f(TARGET_EPS, 1)} of ${f(TARGET_EPS, 1)}`;
    $("fed-auc").textContent = "–";
    $("fed-records").textContent = "0";
  }

  // ---------------------------------------------------------------- run
  async function run() {
    if (running || !data) return;
    running = true;
    const btn = $("fed-start");
    btn.disabled = true;
    reset();
    const r = data.replay;
    const sigma = r.sigma_per_school.reduce((a, b) => a + b, 0) / r.sigma_per_school.length;
    const pts = [];
    for (const round of r.rounds) {
      // 1. local training
      setSchoolsTraining(true);
      schoolStatus.forEach((s) => { s.textContent = "training locally"; s.setAttribute("fill", C.accent); });
      serverStatus.textContent = "waiting for updates"; serverStatus.setAttribute("fill", C.muted);
      await sleep(ROUND_MS * 0.22);
      // 2. updates travel up
      links.forEach((l) => l.up.setAttribute("opacity", 1));
      dots.forEach((d) => d.up.setAttribute("opacity", 1));
      await animate(ROUND_MS * 0.24, (p) => dots.forEach((d) => {
        d.up.setAttribute("cx", d.x + (SERVER.x - d.x) * p); d.up.setAttribute("cy", SCHOOL_TOP + (SERVER.bottom - SCHOOL_TOP) * p);
      }));
      dots.forEach((d) => d.up.setAttribute("opacity", 0));
      links.forEach((l) => l.up.setAttribute("opacity", 0));
      setSchoolsTraining(false);
      // 3. aggregation
      serverBox.setAttribute("stroke", C.accent);
      serverStatus.textContent = "aggregating 3 updates"; serverStatus.setAttribute("fill", C.ink);
      await sleep(ROUND_MS * 0.14);
      serverBox.setAttribute("stroke", C.ink);
      // 4. improved model travels down
      serverStatus.textContent = "sending improved model";
      links.forEach((l) => l.down.setAttribute("opacity", 1));
      dots.forEach((d) => d.down.setAttribute("opacity", 1));
      await animate(ROUND_MS * 0.24, (p) => dots.forEach((d) => {
        d.down.setAttribute("cx", SERVER.x + (d.x - SERVER.x) * p); d.down.setAttribute("cy", SERVER.bottom + (SCHOOL_TOP - SERVER.bottom) * p);
      }));
      dots.forEach((d) => d.down.setAttribute("opacity", 0));
      links.forEach((l) => l.down.setAttribute("opacity", 0));
      schoolStatus.forEach((s) => { s.textContent = "records stay here"; s.setAttribute("fill", C.muted); });
      // recorded values for this round
      pts.push(`${CH.cx(round.round)},${CH.cy(round.val_auc)}`);
      chartLine.setAttribute("points", pts.join(" "));
      chartDot.setAttribute("cx", CH.cx(round.round)); chartDot.setAttribute("cy", CH.cy(round.val_auc)); chartDot.setAttribute("opacity", 1);
      $("fed-auc").textContent = `round ${round.round}: ${f(round.val_auc, 3)}`;
      const left = Math.max(0, TARGET_EPS - round.eps_spent_max);
      $("fed-eps-bar").style.width = `${(left / TARGET_EPS) * 100}%`;
      $("fed-eps-text").textContent = `ε ${f(left, 2)} of ${f(TARGET_EPS, 1)} left`;
      const row = document.createElement("div");
      row.innerHTML = `Round ${round.round} · aggregated 3 updates · noise σ=${f(sigma, 2)} · raw records transmitted: <span class="font-semibold text-ink">0</span>`;
      $("fed-log").appendChild(row);
      $("fed-log").scrollTop = $("fed-log").scrollHeight;
      await sleep(ROUND_MS * 0.16);
    }
    serverStatus.textContent = `done · ${r.rounds.length} rounds`;
    serverStatus.setAttribute("fill", C.muted);
    schoolStatus.forEach((s) => { s.textContent = "has the shared model"; s.setAttribute("fill", C.ink); });
    $("fed-auc").textContent = `final: ${f(r.final_auc, 3)}`;
    $("fed-eps-text").textContent = `ε spent ${f(r.logged_epsilon_spent, 2)} of ${f(TARGET_EPS, 1)}`;
    running = false;
    btn.disabled = false;
    btn.textContent = "Replay training";
    window.__fedDone = true;
  }

  // ---------------------------------------------------------------- init
  async function init() {
    const btn = $("fed-start");
    if (!btn) return;
    try {
      const res = await fetch("/api/federated");
      const body = await res.json().catch(() => null);
      if (!res.ok) throw new Error((body && body.error) || `HTTP ${res.status}`);
      data = body;
    } catch (e) {
      $("fed-error").hidden = false;
      $("fed-error").textContent = `Federated replay unavailable: ${e.message}`;
      return;
    }
    const r = data.replay;
    const st = r.schools.map((s) => s.n_states);
    renderSchools();
    reset();
    $("fed-caption").textContent = `Replay of a recorded run (Experiment 4, fold ${r.fold}): logistic regression trained by FedAvg with `
      + `client-side DP-SGD (target ε = ${TARGET_EPS}, δ = 10⁻⁵, noise multiplier σ = ${f(r.logged_noise_multiplier_mean, 2)}) for `
      + `${r.rounds.length} rounds, scored after each round on ${fmtInt(r.n_val)} validation records. The school names are illustrative: `
      + `each school holds the NSCH 2024 records of ${Math.min(...st)}–${Math.max(...st)} U.S. states. Update size ${fmtInt(data.update_bytes)} B, `
      + `measured in Experiment 7 for the same setup. All ${r.rounds.length} recorded rounds are shown, in about 20 seconds.`;
    btn.disabled = false;
    btn.addEventListener("click", run);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
