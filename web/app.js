"use strict";

// ------------------------------------------------------------------ state
const state = {
  spec: null,          // the 8 editable inputs (from /api/sample)
  record: null,        // {record_id, true_label, inputs} or null = typical answers
  encoding: "angle",
  prediction: null,
  privacy: null,
  epsIndex: 3,
};
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const signed = (v, d = 3) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function api(path, options = {}) {
  let res;
  try {
    res = await fetch(path, options);
  } catch (e) {
    throw new Error("Cannot reach the demo server. Is `uvicorn api:app` running?");
  }
  let body = null;
  try { body = await res.json(); } catch (e) { /* non-JSON */ }
  if (!res.ok) throw new Error((body && body.error) || `Request failed (${res.status})`);
  return body;
}
function showError(id, err) {
  const el = $(id);
  if (!err) { el.hidden = true; el.textContent = ""; return; }
  el.hidden = false;
  el.textContent = err.message || String(err);
}

// ------------------------------------------------------------------ routing
const SECTIONS = ["try", "pipeline", "privacy", "results"];
function parseHash() {
  const raw = location.hash.replace(/^#/, "") || "try";
  const [name, query] = raw.split("?");
  return { name: SECTIONS.includes(name) ? name : "try", params: new URLSearchParams(query || "") };
}
async function route() {
  const { name, params } = parseHash();
  SECTIONS.forEach((s) => { $(s).hidden = s !== name; });
  document.querySelectorAll(".sidenav a").forEach((a) => a.classList.toggle("active", a.dataset.section === name));
  if (params.get("record")) await loadRecord({ record_id: params.get("record") }, true);
  if (params.get("enc") && params.get("enc") !== state.encoding) setEncoding(params.get("enc"), false);
  if (name === "pipeline") await ensurePrediction("pipe-error");
  if (name === "try" && params.get("record")) await runPredict();
  if (name === "privacy") await loadPrivacy();
  if (name === "results") await loadResults();
}
window.addEventListener("hashchange", route);

// ------------------------------------------------------------------ 1. Try it
function renderInputs() {
  const form = $("inputs");
  form.innerHTML = "";
  for (const inp of state.spec) {
    const current = state.record ? state.record.inputs.find((i) => i.key === inp.key).value : null;
    const field = document.createElement("div");
    field.className = "field";
    const id = `in-${inp.key}`;
    const opts = [`<option value="">${state.record ? "Missing in record (imputed)" : "Typical (training median)"}</option>`]
      .concat(inp.options.map((o) => `<option value="${o.value}">${esc(o.text)}</option>`)).join("");
    field.innerHTML = `<label for="${id}">${esc(inp.label)}</label><select id="${id}" data-key="${inp.key}">${opts}</select><small>${inp.key}</small>`;
    form.appendChild(field);
    const sel = field.querySelector("select");
    sel.value = current === null || current === undefined ? "" : String(Math.round(current));
    sel.dataset.original = sel.value;
    sel.addEventListener("change", () => field.classList.toggle("changed", sel.value !== sel.dataset.original));
  }
}
function edits() {
  const out = {};
  document.querySelectorAll("#inputs select").forEach((s) => { out[s.dataset.key] = s.value === "" ? null : Number(s.value); });
  return out;
}
function setRecordInfo() {
  const el = $("record-info");
  if (!state.record) { el.textContent = "No record loaded: answers not shown use training medians."; return; }
  const lab = state.record.true_label ? "diagnosed (currently has depression)" : "not diagnosed";
  el.innerHTML = `Record <b>#${state.record.record_id}</b> · survey label: <b>${lab}</b> · a dev record neither model was trained on`;
}
async function loadRecord(query, silent = false) {
  showError("try-error", null);
  try {
    const qs = new URLSearchParams(query).toString();
    const s = await api(`/api/sample${qs ? "?" + qs : ""}`);
    state.spec = s.inputs.map(({ key, label, options }) => ({ key, label, options }));
    state.record = s;
    state.prediction = null;
    renderInputs();
    setRecordInfo();
    $("prediction-cards").innerHTML = "";
    $("disagree").hidden = true;
  } catch (e) {
    showError("try-error", e);
    if (!silent) throw e;
  }
}
async function resetTypical() {
  if (!state.spec) await loadRecord({}, true);
  state.record = null;
  state.prediction = null;
  renderInputs();
  setRecordInfo();
  $("prediction-cards").innerHTML = "";
  $("disagree").hidden = true;
}

function gauge(value, threshold, lo, hi, atRisk) {
  const pct = (v) => Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100));
  return `<div class="gauge"><div class="track"></div><div class="thr" style="left:${pct(threshold)}%"></div>
    <div class="dot ${atRisk ? "yes" : ""}" style="left:${pct(value)}%"></div></div>
    <div class="gauge-labels"><span>${fmt(lo, 2)}</span><span>threshold ${fmt(threshold, 3)}</span><span>${fmt(hi, 2)}</span></div>`;
}
function modelCard(title, sub, m, value, valueLabel, lo, hi) {
  return `<div class="card ${m.at_risk ? "risk" : ""}">
    <div class="kicker">${esc(title)}</div>
    <div class="muted">${esc(sub)}</div>
    <div class="big">${fmt(value, 3)}</div>
    <div class="muted">${esc(valueLabel)}</div>
    ${gauge(value, m.threshold, lo, hi, m.at_risk)}
    <p><span class="verdict ${m.at_risk ? "yes" : "no"}">${m.at_risk ? "Flagged: at risk" : "Not flagged"}</span></p>
    <dl class="kv"><dt>Threshold</dt><dd>${fmt(m.threshold, 4)}</dd>
    <dt>Distance from threshold</dt><dd>${signed(m.distance, 4)}</dd></dl></div>`;
}
function renderPrediction(p) {
  const lr = p.logreg, q = p.qsvm;
  const span = Math.max(1, Math.abs(q.distance) * 1.25);
  $("prediction-cards").innerHTML =
    modelCard("Classical · Logistic regression", `all 95 features · trained on ${lr.trained_rows.toLocaleString()} rows · C = ${lr.C}`,
      lr, lr.probability, "probability of current diagnosed depression", 0, 1) +
    modelCard("Quantum · QSVM (angle encoding, 8 qubits)", `PCA-8 · trained on ${q.trained_rows.toLocaleString()} rows · C = ${q.C}`,
      q, q.score, "SVM decision score (not a probability)", q.threshold - span, q.threshold + span);
  const b = $("disagree");
  b.hidden = false;
  if (lr.at_risk !== q.at_risk) {
    b.className = "banner disagree";
    b.textContent = `The models disagree: logistic regression says ${lr.at_risk ? "at risk" : "not at risk"}, the QSVM says ${q.at_risk ? "at risk" : "not at risk"}.`;
  } else {
    b.className = "banner agree";
    b.textContent = `Both models agree: ${lr.at_risk ? "flagged as at risk" : "not flagged"}.`;
  }
  if (state.record) {
    const truth = state.record.true_label === 1;
    b.textContent += ` Survey label: ${truth ? "diagnosed" : "not diagnosed"}.`;
  }
}
async function runPredict() {
  if (!state.spec) await loadRecord({}, true).catch(() => {});
  if (!state.spec) return;
  const btn = $("btn-predict");
  btn.disabled = true;
  $("predict-status").textContent = "Running the pipeline…";
  showError("try-error", null);
  try {
    state.prediction = await api("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ record_id: state.record ? state.record.record_id : null, edits: edits(), encoding: state.encoding }),
    });
    renderPrediction(state.prediction);
    renderPipeline(state.prediction);
    $("predict-status").textContent = "";
  } catch (e) {
    showError("try-error", e);
    $("predict-status").textContent = "";
  } finally {
    btn.disabled = false;
  }
}

// ------------------------------------------------------------------ 2. Pipeline
function setEncoding(enc, run = true) {
  state.encoding = enc;
  document.querySelectorAll("#enc-switch button").forEach((b) => b.classList.toggle("on", b.dataset.enc === enc));
  if (run) pipelinePredict();
}
async function ensurePrediction(errId) {
  if (state.prediction && state.prediction.selected_encoding.encoding === state.encoding) { renderPipeline(state.prediction); return; }
  await pipelinePredict(errId);
}
async function pipelinePredict(errId = "pipe-error") {
  if (!state.spec) { try { await loadRecord({}); } catch (e) { showError(errId, e); return; } }
  $("pipe-status").textContent = "Encoding and classifying…";
  showError(errId, null);
  try {
    state.prediction = await api("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ record_id: state.record ? state.record.record_id : null, edits: edits(), encoding: state.encoding }),
    });
    renderPipeline(state.prediction);
    renderPrediction(state.prediction);
  } catch (e) {
    showError(errId, e);
  } finally {
    $("pipe-status").textContent = "";
  }
}
function vec(values, n = 12, digits = 3) {
  const cell = (v) => `<span><b>${esc(v.name)}</b>${typeof v.value === "number" ? fmt(v.value, digits) : esc(v.value ?? "missing")}</span>`;
  const head = values.slice(0, n).map(cell).join("");
  const rest = values.length > n ? `<details><summary>Show all ${values.length}</summary><div class="vec">${values.slice(n).map(cell).join("")}</div></details>` : "";
  return `<div class="vec">${head}</div>${rest}`;
}
function renderPipeline(p) {
  const s = p.selected_encoding;
  const names = { angle: "Angle", amplitude: "Amplitude", zz: "ZZ feature map", reupload: "Re-uploading" };
  $("enc-summary").innerHTML = `
    <div class="card ${s.at_risk ? "risk" : ""}"><div class="kicker">${names[s.encoding]} · QSVM score</div><div class="big mid">${fmt(s.score, 3)}</div>
      <span class="verdict ${s.at_risk ? "yes" : "no"}">${s.at_risk ? "Flagged: at risk" : "Not flagged"}</span>
      <dl class="kv"><dt>Threshold</dt><dd>${fmt(s.threshold, 4)}</dd><dt>Distance</dt><dd>${signed(s.distance, 4)}</dd></dl></div>
    <div class="card"><div class="kicker">Cross-validation ROC-AUC</div><div class="big mid">${fmt(s.cv_auc.mean, 3)}</div>
      <div class="muted">± ${fmt(s.cv_auc.std, 3)} over ${s.cv_auc.folds} folds (Experiment 1, PCA, 8 qubits)</div></div>
    <div class="card"><div class="kicker">Status</div><p>${s.encoding === "angle"
      ? "<b>Final evaluated model.</b> Hold-out ROC-AUC 0.854."
      : "Comparison encoding, fitted on the same 1,200 rows. Not the final model."}</p>
      <p class="muted">Logistic regression on this record: ${fmt(p.logreg.probability, 3)} (${p.logreg.at_risk ? "flagged" : "not flagged"}).</p></div>`;
  $("circuit").innerHTML = p.circuit_svg;
  $("circuit-caption").textContent = s.encoding === "amplitude"
    ? "(state preparation of 256 amplitudes; shown as one block)"
    : "(this record's angles, rounded to 2 decimals)";
  const t = p.trace;
  const stage = (title, detail, body) => `<li><h4>${esc(title)}</h4><div class="detail">${esc(detail)}</div>${body}</li>`;
  const red = t[3];
  const st = t[4];
  const maxP = Math.max(...st.probabilities);
  const bars = `<div class="bars" title="probability of each of the 256 basis states">${st.probabilities.map((v) => `<div style="height:${(v / maxP) * 100}%"></div>`).join("")}</div>`;
  const topRows = st.top.map((r) => `<tr><td><code>|${r.basis}⟩</code></td><td>${signed(r.amplitude_re, 4)}${Math.abs(r.amplitude_im) > 1e-9 ? ` ${signed(r.amplitude_im, 4)}i` : ""}</td><td>${fmt(r.probability, 4)}</td></tr>`).join("");
  const m = t[5];
  const nsv = m.nearest_support_vectors.map((r) => `<tr><td>${fmt(r.fidelity, 4)}</td><td>${r.label}</td><td>${signed(r.weight, 4)}</td></tr>`).join("");
  const sc = t[6];
  $("stages").innerHTML = [
    stage(t[0].stage, t[0].detail, vec(t[0].values, 12, 1)),
    stage(t[1].stage, t[1].detail, vec(t[1].values, 12, 2)),
    stage(t[2].stage, t[2].detail, vec(t[2].values, 12, 3)),
    stage(red.stage, red.detail, vec(red.values, 8, 3) + (red.angles ? `<p class="muted" style="margin:10px 0 4px">Rotation angles (radians), one per qubit:</p>` +
      vec(red.angles.map((v, i) => ({ name: `q${i}`, value: v })), 8, 3) : "")),
    stage(st.stage, st.detail, bars + `<table style="margin-top:12px"><tr><th>Basis state</th><th>Amplitude</th><th>Probability</th></tr>${topRows}</table>`),
    stage(m.stage, m.detail, `<dl class="kv"><dt>Training states compared</dt><dd>${m.kernel.n_train.toLocaleString()}</dd>
      <dt>Fidelity min / median / max</dt><dd>${fmt(m.kernel.min, 3)} / ${fmt(m.kernel.median, 3)} / ${fmt(m.kernel.max, 3)}</dd>
      <dt>Support vectors</dt><dd>${m.kernel.n_support}</dd></dl>
      <p class="muted" style="margin:12px 0 4px">Nearest support vectors:</p>
      <table><tr><th>Fidelity</th><th>Their label</th><th>Weight</th></tr>${nsv}</table>`),
    stage(sc.stage, sc.detail, `<dl class="kv"><dt>Pull towards "at risk"</dt><dd>${signed(sc.positive_part, 4)}</dd>
      <dt>Pull towards "not at risk"</dt><dd>${signed(sc.negative_part, 4)}</dd><dt>Intercept</dt><dd>${signed(sc.intercept, 4)}</dd>
      <dt>Score</dt><dd><b>${fmt(sc.score, 4)}</b></dd><dt>Threshold</dt><dd>${fmt(sc.threshold, 4)}</dd>
      <dt>Decision</dt><dd><span class="verdict ${sc.at_risk ? "yes" : "no"}">${sc.at_risk ? "Flagged: at risk" : "Not flagged"}</span></dd></dl>`),
  ].join("");
}

// ------------------------------------------------------------------ 3. Privacy
const SERIES_STYLE = {
  vqc_sub: { color: "#2E7D8C", dash: "", marker: "circle" },
  logreg_full: { color: "#16233D", dash: "", marker: "square" },
  logreg_sub_clip10: { color: "#16233D", dash: "6 4", marker: "diamond" },
  logreg_sub: { color: "#9AA1AE", dash: "", marker: "triangle" },
};
async function loadPrivacy() {
  showError("priv-error", null);
  if (!state.privacy) {
    try { state.privacy = await api("/api/privacy"); } catch (e) { showError("priv-error", e); return; }
  }
  renderPrivacy();
}
function marker(kind, x, y, color) {
  const r = 6;
  if (kind === "square") return `<rect x="${x - r}" y="${y - r}" width="${2 * r}" height="${2 * r}" fill="${color}" stroke="#fff" stroke-width="1.5"/>`;
  if (kind === "diamond") return `<path d="M${x} ${y - r - 1}L${x + r + 1} ${y}L${x} ${y + r + 1}L${x - r - 1} ${y}Z" fill="${color}" stroke="#fff" stroke-width="1.5"/>`;
  if (kind === "triangle") return `<path d="M${x} ${y - r - 1}L${x + r + 1} ${y + r}L${x - r - 1} ${y + r}Z" fill="${color}" stroke="#fff" stroke-width="1.5"/>`;
  return `<circle cx="${x}" cy="${y}" r="${r}" fill="${color}" stroke="#fff" stroke-width="1.5"/>`;
}
function renderPrivacy() {
  const P = state.privacy;
  const eps = P.epsilons;
  const idx = state.epsIndex;
  const e = eps[idx];
  $("eps-value").textContent = e === "inf" ? "ε = ∞ (no privacy)" : `ε = ${e}`;
  const at = (arm) => { const s = P.series.find((x) => x.arm === arm); return s ? s.points.find((p) => p.eps === e) : null; };
  const inf = (arm) => { const s = P.series.find((x) => x.arm === arm); return s ? s.points.find((p) => p.eps === "inf") : null; };
  const card = (title, sub, arm) => {
    const p = at(arm), base = inf(arm);
    if (!p) return `<div class="card"><div class="kicker">${title}</div><p class="muted">No data.</p></div>`;
    const drop = e === "inf" ? "" : `<div class="muted">${signed(p.auc - base.auc, 3)} vs non-private (${fmt(base.auc, 3)})</div>`;
    return `<div class="card"><div class="kicker">${title}</div><div class="muted">${sub}</div><div class="big">${fmt(p.auc, 3)}</div>
      <div class="muted">ROC-AUC ± ${fmt(p.auc_std, 3)}</div>${drop}
      <dl class="kv"><dt>Balanced accuracy</dt><dd>${fmt(p.balanced_accuracy, 3)}</dd><dt>ε spent (Opacus)</dt><dd>${p.epsilon_spent === null ? "– (no DP)" : fmt(p.epsilon_spent, 3)}</dd></dl></div>`;
  };
  $("priv-cards").innerHTML = card("Quantum · VQC", "re-uploading, 4 qubits, 1.2k rows, PCA-4", "vqc_sub") +
    card("Classical · Logistic regression", "full data, all 95 features", "logreg_full");

  // chart
  const W = 900, H = 380, L = 64, R = 24, T = 18, B = 52;
  const xs = (i) => (i < 5 ? L + i * ((W - L - R - 90) / 4) : W - R - 30);
  const all = P.series.flatMap((s) => s.points.flatMap((p) => [p.auc - p.auc_std, p.auc + p.auc_std]));
  const lo = Math.floor(Math.min(...all) * 20) / 20, hi = Math.ceil(Math.max(...all) * 20) / 20;
  const ys = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="ROC-AUC against privacy budget">`;
  for (let v = lo; v <= hi + 1e-9; v += 0.05) {
    svg += `<line x1="${L}" x2="${W - R}" y1="${ys(v)}" y2="${ys(v)}" stroke="#ECE9E0"/>`;
    svg += `<text x="${L - 10}" y="${ys(v) + 4}" text-anchor="end" font-size="13" fill="#6F7686">${v.toFixed(2)}</text>`;
  }
  svg += `<line x1="${(xs(4) + xs(5)) / 2}" x2="${(xs(4) + xs(5)) / 2}" y1="${T}" y2="${H - B}" stroke="#B9B5A8" stroke-dasharray="2 4"/>`;
  svg += `<rect x="${xs(idx) - 26}" y="${T}" width="52" height="${H - T - B}" fill="#E3EFF1" opacity=".7"/>`;
  eps.forEach((v, i) => {
    svg += `<text x="${xs(i)}" y="${H - B + 22}" text-anchor="middle" font-size="14" fill="${i === idx ? "#16233D" : "#6F7686"}" font-weight="${i === idx ? 700 : 400}">${v === "inf" ? "∞" : v}</text>`;
  });
  svg += `<text x="${(L + W - R) / 2}" y="${H - 8}" text-anchor="middle" font-size="13" fill="#6F7686">privacy budget ε (smaller = stronger privacy) · δ = ${P.delta}</text>`;
  for (const s of P.series) {
    const st = SERIES_STYLE[s.arm] || { color: "#888", dash: "", marker: "circle" };
    const pts = s.points.map((p) => ({ i: eps.indexOf(p.eps), ...p })).filter((p) => p.i >= 0);
    const finite = pts.filter((p) => p.eps !== "inf");
    svg += `<polyline fill="none" stroke="${st.color}" stroke-width="2.5" stroke-dasharray="${st.dash}" points="${finite.map((p) => `${xs(p.i)},${ys(p.auc)}`).join(" ")}"/>`;
    const last = finite[finite.length - 1], infp = pts.find((p) => p.eps === "inf");
    if (last && infp) svg += `<line x1="${xs(last.i)}" y1="${ys(last.auc)}" x2="${xs(infp.i)}" y2="${ys(infp.auc)}" stroke="${st.color}" stroke-width="1.2" stroke-dasharray="2 4"/>`;
    for (const p of pts) {
      svg += `<line x1="${xs(p.i)}" x2="${xs(p.i)}" y1="${ys(p.auc - p.auc_std)}" y2="${ys(p.auc + p.auc_std)}" stroke="${st.color}" stroke-width="1.2" opacity=".6"/>`;
      svg += marker(st.marker, xs(p.i), ys(p.auc), st.color);
    }
  }
  svg += "</svg>";
  $("priv-chart").innerHTML = svg;
  $("priv-legend").innerHTML = P.series.map((s) => {
    const st = SERIES_STYLE[s.arm] || { color: "#888", dash: "" };
    return `<span><i class="${st.dash ? "dash" : ""}" style="border-color:${st.color}"></i>${esc(s.label)}</span>`;
  }).join("");

  const h = P.headline;
  $("priv-headline").innerHTML = `<div class="kicker">Headline · ${esc(h.source)}</div>
    <div class="big">−${fmt(h.cost, 3)} AUC</div>
    <p>Privacy at ε ≈ 4 costs about <b>${fmt(h.cost, 4)}</b> ROC-AUC on the locked hold-out: ${fmt(h.nonprivate_auc, 4)} without privacy → ${fmt(h.dp_eps4_auc, 4)} with DP-SGD (ε spent ${fmt(h.epsilon_spent, 2)}, δ = ${P.delta}).
    The curve above is cross-validation (${P.folds} folds); the headline is the one-shot hold-out.</p>
    <p class="muted">Note: with a shared clipping norm of 1, LogReg on the quantum model's 1.2k rows looked far worse under DP than the VQC. The dashed clip-10 line shows that gap was a hyperparameter artefact, not a quantum advantage.</p>`;
  $("priv-table").innerHTML = `<table><tr><th>Arm</th><th>ROC-AUC</th><th>± std</th><th>Balanced acc.</th><th>ε spent</th></tr>${P.series.map((s) => {
    const p = s.points.find((x) => x.eps === e);
    return p ? `<tr class="${s.family === "quantum" ? "hl" : ""}"><td>${esc(s.label)}</td><td>${fmt(p.auc, 3)}</td><td>${fmt(p.auc_std, 3)}</td><td>${fmt(p.balanced_accuracy, 3)}</td><td>${p.epsilon_spent === null ? "–" : fmt(p.epsilon_spent, 3)}</td></tr>` : "";
  }).join("")}</table>`;
}
$("eps-slider").addEventListener("input", (ev) => { state.epsIndex = Number(ev.target.value); if (state.privacy) renderPrivacy(); });

// ------------------------------------------------------------------ 4. Results
function table(rows, highlight) {
  if (!rows || !rows.length) return `<p class="muted">No rows.</p>`;
  const cols = Object.keys(rows[0]);
  return `<table><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>${rows.map((r) =>
    `<tr class="${highlight && highlight(r) ? "hl" : ""}">${cols.map((c) => `<td>${r[c] === null ? "" : esc(r[c])}</td>`).join("")}</tr>`).join("")}</table>`;
}
let resultsLoaded = false;
async function loadResults() {
  if (resultsLoaded) return;
  showError("res-error", null);
  try {
    const r = await api("/api/results");
    $("tbl-qvc").innerHTML = table(r.quantum_vs_classical);
    $("tbl-holdout").innerHTML = table(r.holdout);
    $("tbl-classical").innerHTML = table(r.classical);
    const captions = {
      "privacy_utility_curve.png": "Privacy–utility curve (headline): ROC-AUC against ε for the quantum and classical models.",
      "roc_curves.png": "Mean ROC curves over 15 folds: the three quantum models, LogReg on the same inputs, and LogReg on full data.",
      "encoding_comparison.png": "Experiment 1: ROC-AUC by encoding, qubit count and feature selection; black bar = LogReg on the same inputs.",
      "federated_convergence.png": "Experiment 4: federated training with DP at ε = 4 across 3, 5 and 10 clients.",
    };
    $("figures").innerHTML = r.figures.map((f, i) => f.exists
      ? `<figure class="${i === 0 || i === 2 ? "wide" : ""}"><img src="${f.url}" alt="${esc(captions[f.file] || f.file)}" loading="lazy"><figcaption>${esc(captions[f.file] || f.file)}</figcaption></figure>`
      : `<figure><div class="error">Figure missing: results/figures/${esc(f.file)}</div></figure>`).join("");
    resultsLoaded = true;
  } catch (e) {
    showError("res-error", e);
  }
}

// ------------------------------------------------------------------ wiring
$("btn-random").addEventListener("click", () => loadRecord({}).catch(() => {}));
$("btn-random-pos").addEventListener("click", () => loadRecord({ positive: true }).catch(() => {}));
$("btn-reset").addEventListener("click", () => resetTypical().catch((e) => showError("try-error", e)));
$("btn-predict").addEventListener("click", runPredict);
$("inputs").addEventListener("submit", (e) => e.preventDefault());
document.querySelectorAll("#enc-switch button").forEach((b) => b.addEventListener("click", () => setEncoding(b.dataset.enc)));

(async function init() {
  try { await resetTypical(); } catch (e) { showError("try-error", e); }
  await route();
})();
