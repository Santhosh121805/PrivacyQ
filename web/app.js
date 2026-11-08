"use strict";

/* DP-FedQ notebook UI — vanilla JS + Observable Plot (global `Plot`, local UMD build). */

const COLOR = { classical: "#4269d0", classicalLight: "#97bbf5", quantum: "#6cc5b0", risk: "#ff725c", muted: "#6b6f76", line: "#e4e4e7", ink: "#1b1e23" };
const ENC_NAMES = { angle: "Angle", amplitude: "Amplitude", zz: "ZZ feature map", reupload: "Re-uploading" };

const state = { spec: null, record: null, encoding: "angle", prediction: null, privacy: null, epsIndex: 3 };
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const f = (v, d = 3) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "–" : Number(v).toFixed(d));
const sgn = (v, d = 3) => (Number(v) >= 0 ? "+" : "−") + Math.abs(Number(v)).toFixed(d);
const num = (v, d = 3, cls = "") => `<span class="num ${cls}">${f(v, d)}</span>`;

async function api(path, options = {}) {
  let res, body = null;
  try { res = await fetch(path, options); } catch (e) { throw new Error("Cannot reach the demo server — is `uvicorn api:app --reload` running?"); }
  try { body = await res.json(); } catch (e) { /* not JSON */ }
  if (!res.ok) throw new Error((body && body.error) || `Request failed (HTTP ${res.status})`);
  return body;
}
function showError(id, err) {
  const el = $(id);
  el.hidden = !err;
  el.textContent = err ? (err.message || String(err)) : "";
}
function postPredict() {
  return api("/api/predict", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ record_id: state.record ? state.record.record_id : null, edits: edits(), encoding: state.encoding }),
  });
}

// ---------------------------------------------------------------- 1. Try it
function renderInputs() {
  const form = $("inputs");
  form.innerHTML = state.spec.map((inp) => {
    const opts = [`<option value="">${state.record ? "Missing in record (imputed)" : "Typical (training median)"}</option>`]
      .concat(inp.options.map((o) => `<option value="${o.value}">${esc(o.text)}</option>`)).join("");
    return `<label class="block"><span class="block text-small font-medium mb-1">${esc(inp.label)}</span>
      <select data-key="${inp.key}">${opts}</select>
      <span class="block num text-caption text-muted mt-1">${inp.key}</span></label>`;
  }).join("");
  form.querySelectorAll("select").forEach((s) => {
    const v = state.record ? state.record.inputs.find((i) => i.key === s.dataset.key).value : null;
    s.value = v === null || v === undefined ? "" : String(Math.round(v));
  });
}
function edits() {
  const out = {};
  document.querySelectorAll("#inputs select").forEach((s) => { out[s.dataset.key] = s.value === "" ? null : Number(s.value); });
  return out;
}
function recordInfo() {
  $("record-info").innerHTML = state.record
    ? `Record <span class="num">#${state.record.record_id}</span> · survey label: <span class="text-ink">${state.record.true_label ? "diagnosed (currently has depression)" : "not diagnosed"}</span> · a dev record neither model was trained on.`
    : "No record loaded: answers not shown use training medians.";
}
function clearPrediction() {
  state.prediction = null;
  $("prediction").innerHTML = "";
  $("disagree").hidden = true;
  $("latency").textContent = "";
}
async function loadRecord(query = {}) {
  showError("try-error", null);
  try {
    const s = await api(`/api/sample?${new URLSearchParams(query)}`);
    state.spec = s.inputs.map(({ key, label, options }) => ({ key, label, options }));
    state.record = s;
    renderInputs(); recordInfo(); clearPrediction();
    return true;
  } catch (e) { showError("try-error", e); return false; }
}
async function typicalAnswers() {
  if (!state.spec && !(await loadRecord())) return;
  state.record = null;
  renderInputs(); recordInfo(); clearPrediction();
}

function bar(value, threshold, lo, hi, color) {
  const pct = (v) => Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100));
  return `<div class="relative h-5 mt-4">
      <div class="absolute top-[9px] inset-x-0 h-px bg-line"></div>
      <div class="absolute top-[3px] w-px h-[14px] bg-ink" style="left:${pct(threshold)}%"></div>
      <div class="absolute top-[5px] w-[10px] h-[10px] -ml-[5px] rounded-full transition-all" style="left:${pct(value)}%;background:${color}"></div>
    </div>
    <div class="flex justify-between num text-caption text-muted"><span>${f(lo, 2)}</span><span>threshold ${f(threshold, 3)}</span><span>${f(hi, 2)}</span></div>`;
}
function modelBlock(title, sub, m, value, kind, lo, hi, color) {
  return `<div>
    <div class="text-small font-medium" style="color:${color}">${esc(title)}</div>
    <div class="text-caption text-muted">${esc(sub)}</div>
    <div class="num text-[44px] leading-tight mt-3 tracking-tight">${f(value, 3)}</div>
    <div class="text-caption text-muted">${esc(kind)}</div>
    ${bar(value, m.threshold, lo, hi, color)}
    <table class="mt-4"><tr><td class="text-muted">Decision</td><td class="n ${m.at_risk ? "text-[#b4462f]" : ""}">${m.at_risk ? "flagged as at risk" : "not flagged"}</td></tr>
    <tr><td class="text-muted">Threshold</td><td class="n">${f(m.threshold, 4)}</td></tr>
    <tr><td class="text-muted">Distance from threshold</td><td class="n">${sgn(m.distance, 4)}</td></tr></table></div>`;
}
function renderPrediction(p) {
  const lr = p.logreg, q = p.qsvm;
  const span = Math.max(1, Math.abs(q.distance) * 1.25);
  $("prediction").innerHTML =
    modelBlock("Logistic regression · classical", `all 95 features · C = ${lr.C}`, lr, lr.probability,
      "probability of current diagnosed depression", 0, 1, COLOR.classical) +
    modelBlock("QSVM · quantum kernel", `angle encoding, 8 qubits, PCA-8 · C = ${q.C}`, q, q.score,
      "SVM decision score (not a probability)", q.threshold - span, q.threshold + span, COLOR.quantum);
  const d = $("disagree");
  d.hidden = false;
  const truth = state.record ? ` The survey label for this adolescent is <span class="font-medium">${state.record.true_label ? "diagnosed" : "not diagnosed"}</span>.` : "";
  if (lr.at_risk !== q.at_risk) {
    d.innerHTML = `<span class="font-semibold">The models disagree.</span> Logistic regression ${lr.at_risk ? "flags" : "does not flag"} this record; the QSVM ${q.at_risk ? "flags" : "does not flag"} it.${truth}`;
    d.className = "mt-6 text-body border-l-2 pl-3 border-risk";
  } else {
    d.innerHTML = `Both models agree: ${lr.at_risk ? "flagged as at risk" : "not flagged"}.${truth}`;
    d.className = "mt-6 text-body border-l-2 pl-3 border-line";
  }
  const t = p.timings_ms;
  $("latency").textContent = `server: QSVM ${f(t.qsvm_final, 0)} ms · LogReg ${f(t.logreg, 0)} ms · total ${f(t.total, 0)} ms`;
}
async function predict() {
  if (!state.spec && !(await loadRecord())) return;
  const btn = $("btn-predict");
  btn.disabled = true;
  showError("try-error", null);
  try {
    state.prediction = await postPredict();
    renderPrediction(state.prediction);
    renderPipeline(state.prediction);
  } catch (e) { showError("try-error", e); } finally { btn.disabled = false; }
}

// ---------------------------------------------------------------- 2. Pipeline
async function setEncoding(enc) {
  state.encoding = enc;
  document.querySelectorAll("#enc-switch button").forEach((b) => b.classList.toggle("on", b.dataset.enc === enc));
  if (!state.spec && !(await loadRecord())) return;
  showError("pipe-error", null);
  $("enc-summary").style.opacity = ".5";
  try {
    state.prediction = await postPredict();
    renderPipeline(state.prediction);
    renderPrediction(state.prediction);
  } catch (e) { showError("pipe-error", e); } finally { $("enc-summary").style.opacity = "1"; }
}
function kvTable(rows, cols = ["Value"]) {
  return `<table><tr><th>Name</th>${cols.map((c) => `<th class="n">${esc(c)}</th>`).join("")}</tr>${rows.join("")}</table>`;
}
function valueTable(values, shown = 10, digits = 3) {
  const row = (v) => `<tr><td class="num text-caption">${esc(v.name)}</td><td class="n">${typeof v.value === "number" ? f(v.value, digits) : esc(v.value ?? "missing")}</td></tr>`;
  const head = kvTable(values.slice(0, shown).map(row));
  return values.length > shown
    ? head + `<details><summary>Show all ${values.length}</summary>${kvTable(values.slice(shown).map(row))}</details>` : head;
}
function step(n, title, detail, body) {
  return `<li><div class="flex items-baseline gap-3"><span class="num text-caption text-muted">${String(n).padStart(2, "0")}</span>
    <h2>${esc(title)}</h2></div><p class="text-small text-muted mt-1 mb-3">${esc(detail)}</p>${body}</li>`;
}
function renderPipeline(p) {
  const s = p.selected_encoding;
  $("enc-summary").innerHTML = `<table class="max-w-xl"><tr><th>${ENC_NAMES[s.encoding]} · QSVM</th><th class="n">Value</th></tr>
    <tr><td class="text-muted">Decision score for this record</td><td class="n">${f(s.score, 4)}</td></tr>
    <tr><td class="text-muted">Threshold (inner-validation)</td><td class="n">${f(s.threshold, 4)}</td></tr>
    <tr><td class="text-muted">Decision</td><td class="n ${s.at_risk ? "text-[#b4462f]" : ""}">${s.at_risk ? "flagged as at risk" : "not flagged"}</td></tr>
    <tr><td class="text-muted">Cross-validation ROC-AUC (Exp. 1, PCA, 8 qubits)</td><td class="n">${f(s.cv_auc.mean, 3)} ± ${f(s.cv_auc.std, 3)}</td></tr>
    <tr><td class="text-muted">Status</td><td class="n">${s.encoding === "angle" ? "final evaluated model (hold-out 0.854)" : "comparison only"}</td></tr></table>`;
  $("circuit").innerHTML = p.circuit_svg;
  const svg = $("circuit").querySelector("svg");
  if (svg) {  // fixed, readable height; wide circuits scroll sideways instead of shrinking
    svg.removeAttribute("width"); svg.removeAttribute("height");
    svg.style.height = "360px"; svg.style.width = "auto"; svg.style.maxWidth = "none";
  }
  const wide = svg && svg.viewBox.baseVal && svg.viewBox.baseVal.width / svg.viewBox.baseVal.height * 360 > 780;
  $("circuit-caption").innerHTML = `${ENC_NAMES[s.encoding]} circuit on 8 qubits for this record${s.encoding === "amplitude" ? " — state preparation of 256 amplitudes, drawn as one block" : ", angles in radians rounded to 2 decimals"}.${wide ? " Scroll sideways to see the whole circuit." : ""} Drawn server-side with PennyLane <span class="num">qml.draw_mpl()</span>.`;

  const t = p.trace;
  const red = t[3], st = t[4], me = t[5], sc = t[6];
  const stateRows = st.top.map((r) => `<tr><td class="num">|${r.basis}⟩</td><td class="n">${sgn(r.amplitude_re, 4)}${Math.abs(r.amplitude_im) > 1e-9 ? ` ${sgn(r.amplitude_im, 4)}i` : ""}</td><td class="n">${f(r.probability, 4)}</td></tr>`);
  const nsv = me.nearest_support_vectors.map((r) => `<tr><td>${r.label}</td><td class="n">${f(r.fidelity, 4)}</td><td class="n">${sgn(r.weight, 4)}</td></tr>`);
  $("stages").innerHTML = [
    step(1, t[0].stage, t[0].detail, valueTable(t[0].values, 10, 1)),
    step(2, "Imputed and scaled", `${t[1].detail}; then each column standardized on the 1,200 training rows.`,
      `<table><tr><th>Column</th><th class="n">Imputed</th><th class="n">Standardized</th></tr>${t[1].values.slice(0, 10).map((v, i) =>
        `<tr><td class="num text-caption">${esc(v.name)}</td><td class="n">${f(v.value, 2)}</td><td class="n">${f(t[2].values[i].value, 3)}</td></tr>`).join("")}</table>
       <details><summary>Show all ${t[1].values.length}</summary><table>${t[1].values.slice(10).map((v, i) =>
        `<tr><td class="num text-caption">${esc(v.name)}</td><td class="n">${f(v.value, 2)}</td><td class="n">${f(t[2].values[i + 10].value, 3)}</td></tr>`).join("")}</table></details>`),
    step(3, "Reduced to 8 features", red.detail, red.angles
      ? `<table><tr><th>Component</th><th class="n">PCA value</th><th class="n">Angle (rad)</th></tr>${red.values.map((v, i) =>
          `<tr><td class="num text-caption">${v.name} → qubit ${i}</td><td class="n">${f(v.value, 3)}</td><td class="n">${f(red.angles[i], 3)}</td></tr>`).join("")}</table>`
      : valueTable(red.values, 8, 3)),
    step(4, "Encoded to a quantum state", st.detail, `<div id="state-chart"></div>
      <p class="caption">Probability of each of the 256 basis states. The eight largest:</p>
      <table class="mt-2"><tr><th>Basis state</th><th class="n">Amplitude</th><th class="n">Probability</th></tr>${stateRows.join("")}</table>`),
    step(5, "Measured", me.detail, `<table><tr><th>Quantity</th><th class="n">Value</th></tr>
      <tr><td class="text-muted">Training states compared</td><td class="n">${me.kernel.n_train.toLocaleString()}</td></tr>
      <tr><td class="text-muted">Fidelity min / median / max</td><td class="n">${f(me.kernel.min, 3)} / ${f(me.kernel.median, 3)} / ${f(me.kernel.max, 3)}</td></tr>
      <tr><td class="text-muted">Support vectors</td><td class="n">${me.kernel.n_support}</td></tr></table>
      <p class="caption">Nearest support vectors:</p><table class="mt-2"><tr><th>Their label</th><th class="n">Fidelity</th><th class="n">Weight</th></tr>${nsv.join("")}</table>`),
    step(6, "Score and decision", `${sc.detail}. The QSVM produces a decision score, not a probability.`,
      `<table><tr><th>Quantity</th><th class="n">Value</th></tr>
      <tr><td class="text-muted">Pull towards "at risk"</td><td class="n">${sgn(sc.positive_part, 4)}</td></tr>
      <tr><td class="text-muted">Pull towards "not at risk"</td><td class="n">${sgn(sc.negative_part, 4)}</td></tr>
      <tr><td class="text-muted">Intercept</td><td class="n">${sgn(sc.intercept, 4)}</td></tr>
      <tr><td class="font-medium">Score</td><td class="n font-medium">${f(sc.score, 4)}</td></tr>
      <tr><td class="text-muted">Threshold</td><td class="n">${f(sc.threshold, 4)}</td></tr>
      <tr><td class="text-muted">Decision</td><td class="n ${sc.at_risk ? "text-[#b4462f]" : ""}">${sc.at_risk ? "flagged as at risk" : "not flagged"}</td></tr>
      <tr><td class="text-muted">Logistic regression on the same record</td><td class="n">p = ${f(p.logreg.probability, 3)}</td></tr></table>`),
  ].join("");
  const probs = st.probabilities.map((v, i) => ({ i, v }));
  $("state-chart").replaceChildren(Plot.plot({
    width: 760, height: 150, marginLeft: 48,
    style: { fontFamily: "JetBrains Mono, monospace", fontSize: "11px", color: COLOR.muted },
    x: { label: "basis state index (0–255)", ticks: [0, 64, 128, 192, 255] },
    y: { label: "probability", grid: true },
    marks: [Plot.ruleY([0], { stroke: COLOR.line }), Plot.rectY(probs, { x1: "i", x2: (d) => d.i + 0.8, y: "v", fill: COLOR.quantum })],
  }));
}

// ---------------------------------------------------------------- 3. Privacy
const SERIES = {
  vqc_sub: { color: COLOR.quantum, dash: null, symbol: "circle", short: "VQC (quantum), 1.2k rows" },
  logreg_full: { color: COLOR.classical, dash: null, symbol: "square", short: "LogReg, full data" },
  logreg_sub_clip10: { color: COLOR.classical, dash: "5,4", symbol: "diamond", short: "LogReg, same rows, clip 10" },
  logreg_sub: { color: COLOR.classicalLight, dash: null, symbol: "triangle", short: "LogReg, same rows, clip 1" },
};
async function loadPrivacy() {
  showError("priv-error", null);
  try { state.privacy = await api("/api/privacy"); renderPrivacy(); } catch (e) { showError("priv-error", e); }
}
function renderPrivacy() {
  const P = state.privacy;
  if (!P) return;
  const eps = P.epsilons, e = eps[state.epsIndex];
  const label = (x) => (x === "inf" ? "∞" : x);
  $("eps-label").textContent = e === "inf" ? "ε = ∞  (no privacy)" : `ε = ${e}`;
  const find = (arm, x) => { const s = P.series.find((z) => z.arm === arm); return s && s.points.find((p) => p.eps === x); };
  const readout = (arm, title, sub, color) => {
    const p = find(arm, e), base = find(arm, "inf");
    if (!p) return `<div class="text-small text-muted">No data for ${esc(title)}.</div>`;
    return `<div><div class="text-small font-medium" style="color:${color}">${title}</div><div class="text-caption text-muted">${sub}</div>
      <div class="num text-[44px] leading-tight mt-3 tracking-tight">${f(p.auc, 3)}</div>
      <div class="text-caption text-muted">ROC-AUC <span class="num">± ${f(p.auc_std, 3)}</span>${e === "inf" ? "" : ` · <span class="num">${sgn(p.auc - base.auc, 3)}</span> vs non-private`}</div>
      <table class="mt-3"><tr><td class="text-muted">Balanced accuracy</td><td class="n">${f(p.balanced_accuracy, 3)}</td></tr>
      <tr><td class="text-muted">ε spent (Opacus)</td><td class="n">${p.epsilon_spent === null ? "– (no DP)" : f(p.epsilon_spent, 3)}</td></tr></table></div>`;
  };
  $("priv-readout").innerHTML = readout("logreg_full", "Logistic regression · classical", "full data, all 95 features", COLOR.classical) +
    readout("vqc_sub", "VQC · quantum", "re-uploading, 4 qubits, 1.2k rows, PCA-4", COLOR.quantum);

  const rows = P.series.flatMap((s) => s.points.map((p) => ({ ...p, arm: s.arm, x: label(p.eps), lo: p.auc - p.auc_std, hi: p.auc + p.auc_std })));
  const finite = rows.filter((r) => r.eps !== "inf");
  const sel = rows.filter((r) => r.eps === e);
  const domain = eps.map(label);
  const marks = [Plot.ruleX([label(e)], { stroke: COLOR.classical, strokeOpacity: 0.08, strokeWidth: 28 }),
    Plot.ruleX([label(e)], { stroke: COLOR.classical, strokeOpacity: 0.5, strokeWidth: 1 })];
  for (const [arm, st] of Object.entries(SERIES)) {
    const a = finite.filter((r) => r.arm === arm), all = rows.filter((r) => r.arm === arm);
    if (!all.length) continue;
    marks.push(Plot.line(a, { x: "x", y: "auc", stroke: st.color, strokeWidth: 2, strokeDasharray: st.dash || undefined }));
    const last = a[a.length - 1], inf = all.find((r) => r.eps === "inf");
    if (last && inf) marks.push(Plot.line([last, inf], { x: "x", y: "auc", stroke: st.color, strokeWidth: 1, strokeDasharray: "2,3" }));
    marks.push(Plot.ruleX(all, { x: "x", y1: "lo", y2: "hi", stroke: st.color, strokeOpacity: 0.5 }));
    marks.push(Plot.dot(all, { x: "x", y: "auc", fill: st.color, symbol: st.symbol, r: 4, stroke: "white", strokeWidth: 1 }));
  }
  marks.push(Plot.dot(sel, { x: "x", y: "auc", r: 7, stroke: COLOR.ink, strokeWidth: 1.5, fill: "none" }));
  marks.push(Plot.text(sel.filter((r) => r.arm === "vqc_sub" || r.arm === "logreg_full"),
    { x: "x", y: "auc", text: (d) => d.auc.toFixed(3), dx: 14, textAnchor: "start", fill: COLOR.ink, stroke: "white", strokeWidth: 4,
      paintOrder: "stroke", fontFamily: "JetBrains Mono, monospace", fontSize: 12 }));
  $("priv-chart").replaceChildren(Plot.plot({
    width: 780, height: 380, marginLeft: 52, marginRight: 52,
    style: { fontFamily: "JetBrains Mono, monospace", fontSize: "12px", color: COLOR.muted, background: "white" },
    x: { type: "point", domain, label: "privacy budget ε  (smaller = stronger privacy)", padding: 0.4 },
    y: { label: "ROC-AUC", grid: true, domain: [Math.floor(Math.min(...rows.map((r) => r.lo)) * 20) / 20, Math.ceil(Math.max(...rows.map((r) => r.hi)) * 20) / 20] },
    marks,
  }));
  $("priv-legend").innerHTML = Object.entries(SERIES).filter(([arm]) => P.series.some((s) => s.arm === arm)).map(([arm, st]) =>
    `<span class="inline-flex items-center gap-2"><svg width="24" height="8"><line x1="0" x2="24" y1="4" y2="4" stroke="${st.color}" stroke-width="2" ${st.dash ? `stroke-dasharray="${st.dash}"` : ""}/></svg>${esc(st.short)}</span>`).join("");
  const h = P.headline;
  $("priv-headline").innerHTML = `<div class="border-t border-line pt-6">
    <p class="text-caption text-muted uppercase tracking-wide">Headline · ${esc(h.source)}</p>
    <p class="text-h2 mt-2">Privacy at ε ≈ 4 costs about <span class="num">${f(h.cost, 2)}</span> ROC-AUC on the hold-out
      (<span class="num">${f(h.nonprivate_auc, 4)}</span> → <span class="num">${f(h.dp_eps4_auc, 4)}</span>, a drop of <span class="num">${f(h.cost, 4)}</span>).</p>
    <p class="caption">Measured once on the locked 4,358-adolescent hold-out; ε spent <span class="num">${f(h.epsilon_spent, 3)}</span>, δ = 10⁻⁵. The curve above is cross-validation.
    With a shared clipping norm of 1, LogReg on the quantum model's 1.2k rows looked far worse under DP than the VQC; the dashed clip-10 line shows that gap was a hyperparameter artefact, not a quantum advantage.</p></div>`;
}
$("eps").addEventListener("input", (ev) => { state.epsIndex = Number(ev.target.value); renderPrivacy(); });

// ---------------------------------------------------------------- 4. Results
function table(rows, opts = {}) {
  if (!rows || !rows.length) return `<p class="caption">No rows.</p>`;
  const cols = Object.keys(rows[0]).filter((c) => !(opts.drop || []).includes(c));
  const isNum = (c) => rows.every((r) => r[c] === null || /^[−+\-<\d.]/.test(String(r[c])));
  return `<div class="overflow-x-auto"><table><tr>${cols.map((c) => `<th class="${isNum(c) ? "n" : ""}">${esc(c)}</th>`).join("")}</tr>${rows.map((r) =>
    `<tr>${cols.map((c) => `<td class="${isNum(c) ? "n" : ""}">${esc(r[c] ?? "")}</td>`).join("")}</tr>`).join("")}</table></div>`;
}
async function loadResults() {
  showError("res-error", null);
  try {
    const r = await api("/api/results");
    $("tbl-qvc").innerHTML = table(r.quantum_vs_classical, { drop: ["Metric"] });
    $("tbl-holdout").innerHTML = table(r.holdout);
    const cap = {
      "privacy_utility_curve.png": "Privacy–utility curve (headline figure of the report).",
      "roc_curves.png": "Mean ROC curves over 15 folds: three quantum models, LogReg on the same inputs, LogReg on full data.",
      "encoding_comparison.png": "Experiment 1: ROC-AUC by encoding, qubit count and feature selection; black bar = LogReg on the same inputs.",
      "federated_convergence.png": "Experiment 4: FedAvg with client-side DP at ε = 4 for 3, 5 and 10 clients. The VQC curves are still rising at round 30.",
    };
    $("figures").innerHTML = r.figures.map((fg) => fg.exists
      ? `<figure><img src="${fg.url}" alt="${esc(cap[fg.file] || fg.file)}" class="w-full border border-line rounded-md" loading="lazy"><figcaption class="caption">${esc(cap[fg.file] || fg.file)}</figcaption></figure>`
      : `<p class="error">Figure missing: results/figures/${esc(fg.file)}</p>`).join("");
  } catch (e) { showError("res-error", e); }
}

// ---------------------------------------------------------------- navigation
function highlightNav() {
  let current = "try";
  for (const id of ["try", "pipeline", "privacy", "results"]) {
    if ($(id).getBoundingClientRect().top < 160) current = id;
  }
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === current));
}
window.addEventListener("scroll", highlightNav, { passive: true });

$("btn-random").addEventListener("click", () => loadRecord());
$("btn-random-pos").addEventListener("click", () => loadRecord({ positive: true }));
$("btn-reset").addEventListener("click", typicalAnswers);
$("btn-predict").addEventListener("click", predict);
$("inputs").addEventListener("submit", (e) => e.preventDefault());
document.querySelectorAll("#enc-switch button").forEach((b) => b.addEventListener("click", () => setEncoding(b.dataset.enc)));

// Deep links: #section?record=9517&enc=zz&eps=0  (used for rehearsal and testing)
(async function init() {
  const [section, query] = (location.hash.replace(/^#/, "") || "try").split("?");
  const params = new URLSearchParams(query || "");
  if (params.get("eps") !== null) { state.epsIndex = Number(params.get("eps")); $("eps").value = state.epsIndex; }
  if (params.get("enc")) {
    state.encoding = params.get("enc");
    document.querySelectorAll("#enc-switch button").forEach((b) => b.classList.toggle("on", b.dataset.enc === state.encoding));
  }
  $("record-info").textContent = "Loading the cached models (first start only takes a few seconds)…";
  const t0 = performance.now();
  const ok = params.get("record") ? await loadRecord({ record_id: params.get("record") }) : await loadRecord();
  console.info(`models ready after ${Math.round(performance.now() - t0)} ms`);
  if (ok) await predict();
  await Promise.all([loadPrivacy(), loadResults()]);
  if (section && $(section)) $(section).scrollIntoView();
  highlightNav();
})();
