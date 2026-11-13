// PrivacyQ — research configuration tool. Vanilla JS module; Observable Plot via ESM CDN with local fallback.

const C = { classical: "#4269d0", quantum: "#6cc5b0", ink: "#1b1e23", muted: "#6b6f76", line: "#e4e4e7", surface: "#f9f9f9" };
const EPS = ["0.5", "1", "2", "4", "8", "inf"];
const SITES = [1, 3, 5, 10];
const MODEL_NAMES = { logreg: "Logistic Regression", qsvm: "QSVM", vqc: "VQC", hybrid: "Hybrid QNN" };
const ENC_NAMES = { angle: "Angle", amplitude: "Amplitude", zz: "ZZ", reupload: "Re-uploading", none: "none" };
const QUANTUM = new Set(["qsvm", "vqc", "hybrid"]);

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const f = (v, d = 3) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "–" : Number(v).toFixed(d));
const epsLabel = (e) => (e === "inf" ? "∞" : e);

// ------------------------------------------------------------- Plot loader
let PlotLib = null;
async function getPlot() {
  if (PlotLib) return PlotLib;
  const timeout = new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), 4000));
  try {
    PlotLib = await Promise.race([import("https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.16/+esm"), timeout]);
  } catch (e) { // offline: identical local build
    const load = (src) => new Promise((res, rej) => { const s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = rej; document.head.appendChild(s); });
    await load("vendor/d3-7.9.0.min.js");
    await load("vendor/plot-0.6.16.umd.min.js");
    PlotLib = window.Plot;
  }
  return PlotLib;
}

async function api(path, options) {
  let res, body = null;
  try { res = await fetch(path, options); } catch (e) { throw new Error("Cannot reach the PrivacyQ server — is `uvicorn api:app --reload` running?"); }
  try { body = await res.json(); } catch (e) { /* non-JSON */ }
  if (!res.ok) throw new Error((body && body.error) || `Request failed (HTTP ${res.status})`);
  return body;
}
function showError(id, err) { const el = $(id); el.hidden = !err; el.textContent = err ? (err.message || String(err)) : ""; }

function table(rows, headers, numericCols = []) {
  return `<table><thead><tr>${headers.map((h, i) => `<th class="${numericCols.includes(i) ? "n" : ""}">${esc(h)}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${r.map((c, i) => `<td class="${numericCols.includes(i) ? "n" : ""}">${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

// ------------------------------------------------------------- state + routing
const cfg = { model: "qsvm", encoding: "angle", features: 8, eps: "inf", sites: 1 };
const PAGES = ["home", "overview", "configure", "inspect", "evidence"];
const loaded = new Set();

function readHash() {
  const [page, query] = (location.hash.replace(/^#/, "") || "home").split("?");
  const p = new URLSearchParams(query || "");
  if (p.get("model") in MODEL_NAMES) cfg.model = p.get("model");
  if (p.get("enc") in ENC_NAMES) cfg.encoding = p.get("enc");
  if (["4", "6", "8"].includes(p.get("f"))) cfg.features = Number(p.get("f"));
  if (EPS.includes(p.get("eps"))) cfg.eps = p.get("eps");
  if (SITES.includes(Number(p.get("sites")))) cfg.sites = Number(p.get("sites"));
  return PAGES.includes(page) ? page : "overview";
}
function cfgQuery() { return `model=${cfg.model}&enc=${cfg.encoding}&f=${cfg.features}&eps=${cfg.eps}&sites=${cfg.sites}`; }
async function route() {
  const page = readHash();
  document.querySelectorAll("[data-view]").forEach((s) => { s.hidden = s.dataset.view !== page; });
  document.body.classList.toggle("home", page === "home");
  if (page === "home" && !loaded.has(page)) { loaded.add(page); renderHome(); }
  document.querySelectorAll(".navitem").forEach((a) => a.classList.toggle("active", a.dataset.page === page));
  window.scrollTo(0, 0);
  if (page === "overview" && !loaded.has(page)) { loaded.add(page); await renderOverview(); }
  if (page === "configure") { syncControls(); await updateConfig(); }
  if (page === "inspect") await renderInspect();
  if (page === "evidence" && !loaded.has(page)) { loaded.add(page); await renderEvidence(); }
}
window.addEventListener("hashchange", route);
document.querySelectorAll(".navitem").forEach((a) => a.addEventListener("click", (e) => {
  e.preventDefault();
  const page = a.dataset.page;
  location.hash = page === "configure" || page === "inspect" ? `${page}?${cfgQuery()}` : page;
}));

// ------------------------------------------------------------- Home (landing page)
async function renderHome() {
  try {
    const o = await api("/api/overview");
    $("home-auc").textContent = f(o.holdout_auc, 3);
    $("home-qvc").textContent = `${f(o.qsvm_auc, 3)}/${f(o.logreg_matched_auc, 3)}`;
    $("home-tests").textContent = o.tests ? `${o.tests.passed} tests passing` : "tests not recorded";
  } catch (e) { /* the pills keep their placeholders; the page still works */ }
}

// ------------------------------------------------------------- Inspect › screening indication (box 9)
const CLASS_LABELS = { none: "None", anxiety_only: "Anxiety only", depression: "Depression (with or without anxiety)" };
function renderScreening(s) {
  const el = $("ins-screening");
  el.hidden = false;
  if (!s || s.error) { el.innerHTML = `<h2>Screening indication</h2><p class="error mt-3">${esc(s ? s.error : "not available")}</p>`; return; }
  const t = s.three_class, r = s.risk, cuts = r.cuts, b = r.inner_val_bands;
  const bar = (k) => `<div class="mt-3"><div class="flex justify-between text-t13"><span>${esc(CLASS_LABELS[k])}${k === t.predicted ? ' <span class="pill ml-1">highest</span>' : ""}</span>
      <span class="num">${f(t.probabilities[k], 3)}</span></div>
      <div class="h-2 bg-surface border border-line rounded-full mt-1 overflow-hidden"><div class="h-full ${k === "none" ? "bg-[#9498a0]" : "bg-accent"}" style="width:${(100 * t.probabilities[k]).toFixed(1)}%"></div></div></div>`;
  const band = (name, lo, hi) => `<div class="flex-1 border rounded-md px-3 py-2 ${r.band === name ? "border-ink" : "border-line"}">
      <div class="text-t13 ${r.band === name ? "font-semibold" : "text-muted"}">${name[0].toUpperCase() + name.slice(1)}${r.band === name ? " ← this record" : ""}</div>
      <div class="num text-t12 text-muted mt-1">${lo} – ${hi}</div>
      <div class="text-t12 text-muted mt-1">inner-val: ${b[name].rows} rows, ${b[name].positives} with depression</div></div>`;
  el.innerHTML = `
    <h2>Screening indication</h2>
    <p class="help mt-2">Not a diagnosis. A research screening indication from logistic regression on all 95 columns, trained on the same
      1,200 training rows as the models above (${t.n_train.toLocaleString("en-US")} for three classes: the rest have undetermined anxiety);
      this record was not used to train either model.</p>
    <div class="grid grid-cols-2 gap-8 mt-5">
      <div>
        <div class="label">Three classes</div>
        <p class="help mt-1">Class-weighted model probabilities (not calibrated to how common each class is). Survey answer for this record:
          <span class="text-ink">${esc(t.survey_answer ? CLASS_LABELS[t.survey_answer] : "undetermined")}</span>.</p>
        ${["none", "anxiety_only", "depression"].map(bar).join("")}
        <p class="help mt-3">Inner-validation macro one-vs-rest ROC-AUC <span class="num">${f(t.inner_val_macro_auc, 3)}</span>.
          Cross-validated and hold-out results: RESULTS.md section 8.</p>
      </div>
      <div>
        <div class="label">Risk level</div>
        <p class="help mt-1">Depression probability <span class="num text-ink">${f(r.probability, 3)}</span> →
          <span class="text-ink font-semibold">${esc(r.band)}</span></p>
        <div class="flex gap-2 mt-3">${band("low", "0", f(cuts.low_moderate, 3))}${band("moderate", f(cuts.low_moderate, 3), f(cuts.moderate_high, 3))}${band("high", f(cuts.moderate_high, 3), "1")}</div>
        <p class="help mt-3">Cut-points chosen on the ${r.n_inner_val} inner-validation rows (${r.n_inner_val_pos} with depression):
          95 % sensitivity for the low/moderate cut (rule-out), 95 % specificity for the moderate/high cut (rule-in).
          A 90 % / 90 % rule was tried first and left only ${r.rule_tried_first ? r.rule_tried_first.bands.moderate.rows : "–"} rows in the moderate band.
          With so few validation positives the cut-points are approximate.</p>
      </div>
    </div>
    <p class="text-t13 mt-6 pt-4 border-t border-line">Box 9 status: state classification (binary and three classes) and risk-level prediction are built.
      Topic and trend analysis, emotion and sentiment analysis, and personalised insights are not, because they need text data the project
      does not have. There is no stress class: NSCH 2024 records no stress variable.</p>`;
}

// ------------------------------------------------------------- Overview
async function renderOverview() {
  showError("ov-error", null);
  let o;
  try { o = await api("/api/overview"); } catch (e) { showError("ov-error", e); return; }
  const tests = o.tests;
  const cards = [
    ["Hold-out ROC-AUC", f(o.holdout_auc, 3), "best classical, untouched test set"],
    ["Best model → DP model (ε ≈ 4)", f(o.privacy_cost, 3), `hold-out AUC gap (${f(o.holdout_auc, 3)} → ${f(o.dp4_holdout_auc, 3)}); in cross-validation about half is the DP noise (0.016, not significant) and half the switch to SGD training`],
    ["Quantum vs classical", `${f(o.qsvm_auc, 3)}/${f(o.logreg_matched_auc, 3)}`, `matched conditions, not significant (Holm p = ${f(o.qvc_holm_p, 2)})`],
    ["Tests passing", tests ? String(tests.passed) : "–",
      tests ? (tests.failed === 0 ? `full suite green · recorded ${tests.recorded.slice(0, 10)}` : `${tests.failed} failing · recorded ${tests.recorded.slice(0, 10)}`)
            : "run scripts/record_test_status.py"],
  ];
  $("ov-cards").innerHTML = cards.map(([label, value, cap]) => `<div class="card">
      <div class="text-t12 uppercase tracking-wide text-muted">${esc(label)}</div>
      <div class="num text-[32px] leading-tight tracking-tight mt-2">${esc(value)}</div>
      <div class="text-t13 text-muted mt-2">${esc(cap)}</div></div>`).join("");

  const Plot = await getPlot();
  const rows = o.curve.series.flatMap((s) => s.points.map((p) => ({ ...p, arm: s.arm, x: epsLabel(p.eps), lo: p.auc - p.std, hi: p.auc + p.std })));
  const marks = [Plot.gridY({ stroke: C.line, strokeOpacity: 1 })];
  for (const s of o.curve.series) {
    const color = s.family === "quantum" ? C.quantum : C.classical;
    const pts = rows.filter((r) => r.arm === s.arm);
    const finite = pts.filter((r) => r.eps !== "inf");
    marks.push(Plot.line(finite, { x: "x", y: "auc", stroke: color, strokeWidth: 2, strokeDasharray: s.dashed ? "5,4" : undefined }));
    marks.push(Plot.line([finite[finite.length - 1], pts[pts.length - 1]], { x: "x", y: "auc", stroke: color, strokeWidth: 1, strokeDasharray: "2,3" }));
    marks.push(Plot.ruleX(pts, { x: "x", y1: "lo", y2: "hi", stroke: color, strokeOpacity: 0.45 }));
    marks.push(Plot.dot(pts, { x: "x", y: "auc", fill: color, r: 3.5, stroke: "white", strokeWidth: 1 }));
  }
  $("ov-chart").replaceChildren(Plot.plot({
    width: 918, height: 300, marginLeft: 44, marginBottom: 36,
    style: { fontFamily: "JetBrains Mono, monospace", fontSize: "12px", color: C.muted, background: "white" },
    x: { type: "point", domain: o.curve.epsilons.map(epsLabel), label: "privacy budget ε →", labelAnchor: "right", padding: 0.3 },
    y: { label: "↑ ROC-AUC", domain: [0.55, 0.9] },
    marks,
  }));
  $("ov-legend").innerHTML = o.curve.series.map((s) => {
    const color = s.family === "quantum" ? C.quantum : C.classical;
    return `<span class="inline-flex items-center gap-1.5"><svg width="20" height="6"><line x1="0" x2="20" y1="3" y2="3" stroke="${color}" stroke-width="2" ${s.dashed ? 'stroke-dasharray="5,4"' : ""}/></svg>${esc(s.label)}</span>`;
  }).join("");
  $("ov-caption").textContent = `Mean ROC-AUC ± std over ${o.curve.folds} cross-validation folds under DP-SGD (Opacus RDP accounting, δ = 10⁻⁵). `
    + "The VQC and the dashed logistic regression use the same 1.2k training rows and 4 features; the solid line uses all rows and 95 features. "
    + "The quantum circuit provides no privacy: the guarantee comes from DP-SGD.";
  $("ov-does").textContent = "PrivacyQ lets a research lead choose how to model sensitive adolescent survey data — encoding, model, feature count, privacy budget and number of participating sites — and see what each choice cost when it was actually measured. "
    + "Every number comes from experiments already run on the NSCH 2024 public-use file; combinations that were never run are flagged and replaced by the nearest measured one. "
    + "It is a planning aid for model builders, not a screening or diagnostic tool.";
  $("ov-found").textContent = `No quantum model beat logistic regression: the best, a quantum-kernel SVM, tied it on identical rows and features (${f(o.qsvm_auc, 3)} vs ${f(o.logreg_matched_auc, 3)}, not significant). `
    + `On the untouched hold-out the private model (DP-SGD, ε ≈ 4) scores ${f(o.privacy_cost, 3)} ROC-AUC below the best non-private model. Cross-validation splits that gap about evenly: 0.016 from training the model by SGD instead of the tuned solver, and 0.016 from the DP noise itself, which is not statistically significant. Federation across up to 10 sites costs nothing further. `
    + "Small quantum models degrade sharply when their 1.2k training rows are split across sites, because each site has too few records for the privacy noise.";
}

// ------------------------------------------------------------- Configure
const configCache = new Map();
function syncControls() {
  document.querySelectorAll("#c-encoding button").forEach((b) => b.classList.toggle("on", b.dataset.v === cfg.encoding));
  document.querySelectorAll("#c-features button").forEach((b) => b.classList.toggle("on", Number(b.dataset.v) === cfg.features));
  document.querySelectorAll('#c-model input').forEach((r) => { r.checked = r.value === cfg.model; });
  $("c-eps").value = EPS.indexOf(cfg.eps);
  $("c-sites").value = SITES.indexOf(cfg.sites);
  $("c-eps-v").textContent = `ε = ${epsLabel(cfg.eps)}`;
  $("c-sites-v").textContent = `${cfg.sites} site${cfg.sites > 1 ? "s" : ""}`;
  const isClassical = cfg.model === "logreg";
  $("c-encoding").classList.toggle("disabled", isClassical);
  $("enc-help").textContent = isClassical ? "Not used by logistic regression, which takes the reduced features directly."
                                          : "How survey values are loaded into qubits.";
}
function onControl(update) {
  update();
  syncControls();
  history.replaceState(null, "", `#configure?${cfgQuery()}`);
  updateConfig();
}
document.querySelectorAll("#c-encoding button").forEach((b) => b.addEventListener("click", () => onControl(() => { cfg.encoding = b.dataset.v; })));
document.querySelectorAll("#c-features button").forEach((b) => b.addEventListener("click", () => onControl(() => { cfg.features = Number(b.dataset.v); })));
document.querySelectorAll("#c-model input").forEach((r) => r.addEventListener("change", () => onControl(() => { cfg.model = r.value; })));
$("c-eps").addEventListener("input", (e) => onControl(() => { cfg.eps = EPS[Number(e.target.value)]; }));
$("c-sites").addEventListener("input", (e) => onControl(() => { cfg.sites = SITES[Number(e.target.value)]; }));

let configSeq = 0;
async function updateConfig() {
  const key = cfgQuery();
  const seq = ++configSeq;
  showError("cfg-error", null);
  let r = configCache.get(key);
  if (!r) {
    try { r = await api(`/api/configure?model=${cfg.model}&encoding=${cfg.encoding}&features=${cfg.features}&eps=${cfg.eps}&sites=${cfg.sites}`); }
    catch (e) { if (seq === configSeq) showError("cfg-error", e); return; }
    configCache.set(key, r);
  }
  if (seq !== configSeq) return; // a newer change arrived
  const body = $("result-body");
  body.style.opacity = "0.35";
  setTimeout(() => { renderConfig(r); body.style.opacity = "1"; }, 150);
}
function renderConfig(r) {
  const m = r.measured, b = r.baseline;
  const color = QUANTUM.has(m.model) ? C.quantum : C.classical;
  $("r-auc").textContent = f(m.auc, 3);
  $("r-std").textContent = `± ${f(m.auc_std, 3)}`;
  $("r-folds").textContent = `${m.folds} folds`;
  const match = r.exact
    ? `<span class="text-accent font-medium">Measured configuration.</span> This exact combination was run.`
    : `<span class="font-medium">Not measured.</span> Showing the nearest measured configuration, which differs in: ${r.differences.map(esc).join("; ")}.`;
  $("r-match").innerHTML = (r.notice ? `<div class="mb-2">${esc(r.notice)}</div>` : "") + match;

  const lo = 0.5, hi = 0.9, pct = (v) => Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100));
  $("r-bar").innerHTML = `
    <div class="absolute inset-x-0 top-[14px] h-2 rounded-full bg-surface border border-line"></div>
    <div class="absolute left-0 top-[14px] h-2 rounded-full transition-all duration-150" style="width:${pct(m.auc)}%;background:${color}"></div>
    <div class="absolute top-[4px] h-7 w-px bg-ink" style="left:${pct(b.cv_auc)}%"></div>
    <div class="absolute -top-[14px] text-t12 text-muted num whitespace-nowrap -translate-x-full pr-1" style="left:${pct(b.cv_auc)}%">baseline ${f(b.cv_auc, 3)}</div>`;
  const gap = r.gap;
  $("r-gap").innerHTML = gap >= 0
    ? `<span class="num">${f(gap, 3)}</span> below the unprivatised classical baseline (logistic regression, all 95 features, full data: <span class="num">${f(b.cv_auc, 3)}</span> cross-validated, <span class="num">${f(b.holdout_auc, 3)}</span> on the locked hold-out).`
    : `<span class="num">${f(-gap, 3)}</span> above the unprivatised classical baseline (<span class="num">${f(b.cv_auc, 3)}</span>).`;
  $("r-bacc").textContent = f(m.bal_acc, 3);
  $("r-recall").textContent = f(m.recall, 3);
  const cost = [];
  cost.push(m.qubits ? `${m.qubits} qubits` : "no circuit");
  if (m.depth) cost.push(`depth ${m.depth}${m.model === "qsvm" ? " (kernel ≈2×)" : ""}`);
  cost.push(`${f(m.fit_s, m.fit_s < 1 ? 2 : 1)} s per fit`);
  if (m.comm_kb) cost.push(`${f(m.comm_kb, 1)} kB communicated`);
  $("r-cost").innerHTML = cost.map(esc).join("<br>");
  $("r-verdict").textContent = r.verdict;
  $("r-sens").textContent = r.sensitivity || "";
  const where = m.sites > 1 ? `${m.sites} sites (${m.partition === "by_state" ? "split by U.S. state" : "IID split"})` : "centralised";
  const priv = m.eps === "inf" ? "no differential privacy" : `ε = ${m.eps}, δ = 10⁻⁵ (ε spent ${f(m.epsilon_spent, 2)})`;
  $("r-source").textContent = `${m.source} · ${m.folds} folds · ${MODEL_NAMES[m.model]}${m.encoding !== "none" ? `, ${ENC_NAMES[m.encoding]} encoding` : ""} · ${m.feature_note} · ${m.rows} training rows · ${priv} · ${where}.`;
}

// ------------------------------------------------------------- Inspect
let inspectRecord = null;
let inspectSeq = 0;
async function loadRandomRecord() {
  showError("ins-error", null);
  try { inspectRecord = await api("/api/sample"); } catch (e) { showError("ins-error", e); return false; }
  return true;
}
$("btn-load").addEventListener("click", async () => { if (await loadRandomRecord()) await renderInspect(); });

async function renderInspect() {
  $("ins-config").innerHTML = `Configured pipeline: <span class="text-ink">${MODEL_NAMES[cfg.model]}${cfg.model !== "logreg" ? ` · ${ENC_NAMES[cfg.encoding]}` : ""} · <span class="num">${cfg.features}</span> features</span> · <a class="text-accent" href="#configure?${cfgQuery()}">change in Configure</a>`;
  if (!inspectRecord && !(await loadRandomRecord())) return;
  const seq = ++inspectSeq;
  const steps = $("ins-steps"), circ = $("ins-circuit-card");
  steps.style.opacity = "0.35"; circ.style.opacity = "0.35";   // old trace stays visible but dimmed while loading
  showError("ins-error", null);
  let d;
  try {
    d = await api("/api/inspect", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ record_id: inspectRecord.record_id, model: cfg.model, encoding: cfg.encoding, features: cfg.features }) });
  } catch (e) { if (seq === inspectSeq) { showError("ins-error", e); steps.style.opacity = "1"; } return; }
  if (seq !== inspectSeq) return;   // a newer request superseded this one
  steps.style.opacity = "1";
  $("ins-record").innerHTML = `Record <span class="num">#${d.record_id}</span> · survey label: ${d.survey_label ? "diagnosed" : "not diagnosed"} · a dev record none of the cached models was trained on · server <span class="num">${f(d.latency_ms, 0)} ms</span>`;
  $("ins-note").hidden = !d.note;
  $("ins-note").textContent = d.note || "";
  $("ins-steps").innerHTML = d.steps.map((s, i) => {
    const rows = s.rows.map((r) => [r[0], r[1] === null ? "missing" : (typeof r[1] === "number" ? f(r[1], Number.isInteger(r[1]) ? 0 : 3) : r[1])]);
    const head = rows.slice(0, 10), rest = rows.slice(10);
    const tbl = rows.length ? table(head, ["", "Value"], [1]) + (rest.length ? `<details><summary>Show all ${rows.length}</summary>${table(rest, ["", "Value"], [1])}</details>` : "") : "";
    return `<li class="card"><div class="flex items-baseline gap-3"><span class="num text-t12 text-muted">${String(i + 1).padStart(2, "0")}</span><span class="text-t15 font-semibold">${esc(s.title)}</span></div>
      <p class="help mt-2 mb-3">${esc(s.detail)}</p>${s.probabilities ? '<div class="state-chart mb-3"></div>' : ""}${tbl}</li>`;
  }).join("");
  renderScreening(d.screening);
  const stateStep = d.steps.find((s) => s.probabilities);
  if (stateStep) {
    const Plot = await getPlot();
    const data = stateStep.probabilities.map((v, i) => ({ i, v }));
    document.querySelector("#ins-steps .state-chart").replaceChildren(Plot.plot({
      width: 870, height: 120, marginLeft: 44,
      style: { fontFamily: "JetBrains Mono, monospace", fontSize: "11px", color: C.muted },
      x: { label: `basis state (0–${data.length - 1}) →`, labelAnchor: "right", ticks: 5 },
      y: { label: "↑ probability", grid: true },
      marks: [Plot.rectY(data, { x1: "i", x2: (q) => q.i + 0.85, y: "v", fill: C.quantum }), Plot.ruleY([0], { stroke: C.line })],
    }));
  }
  const card = $("ins-circuit-card");
  card.hidden = !d.circuit_url;
  if (d.circuit_url) {
    try {
      const res = await fetch(d.circuit_url);
      if (!res.ok) throw new Error((await res.json()).error || `HTTP ${res.status}`);
      const svgText = await res.text();
      if (seq !== inspectSeq) return;
      $("ins-circuit").innerHTML = svgText;
      $("ins-circuit").dataset.key = `${d.encoding}-${d.features}-${d.record_id}`;
      circ.style.opacity = "1";
      const svg = $("ins-circuit").querySelector("svg");
      svg.removeAttribute("width"); svg.removeAttribute("height");
      svg.style.height = `${Math.max(180, 44 * d.features)}px`; svg.style.width = "auto"; svg.style.maxWidth = "none";
      $("ins-circuit-caption").innerHTML = `${ENC_NAMES[d.encoding]} feature map on <span class="num">${d.features}</span> qubits for this record, drawn server-side by PennyLane <span class="num">qml.draw_mpl()</span>${d.encoding === "amplitude" ? "; amplitude state preparation is drawn as one block" : "; angles in radians"}. Scroll sideways for wide circuits.`;
    } catch (e) { $("ins-circuit").innerHTML = `<p class="error">Circuit unavailable: ${esc(e.message)}</p>`; }
  }
}

// ------------------------------------------------------------- Evidence
async function renderEvidence() {
  showError("ev-error", null);
  let e;
  try { e = await api("/api/evidence"); } catch (err) { showError("ev-error", err); return; }
  const generic = (rows) => {
    if (!rows.length) return `<p class="help">No rows.</p>`;
    const cols = Object.keys(rows[0]);
    const numeric = cols.map((c, i) => (rows.every((r) => r[c] === null || /^[−+\-<\d.\s]/.test(String(r[c]))) ? i : -1)).filter((i) => i >= 0);
    return table(rows.map((r) => cols.map((c) => r[c] ?? "")), cols, numeric);
  };
  $("ev-matched").innerHTML = generic(e.matched);
  $("ev-holdout").innerHTML = generic(e.holdout);
  $("ev-encodings").innerHTML = generic(e.encodings);
  const cap = {
    "privacy_utility_curve.png": "Privacy–utility curve: ROC-AUC against ε for the VQC and logistic regression under DP-SGD (10 folds).",
    "encoding_comparison.png": "Encoding sweep: ROC-AUC by encoding, qubit count and feature selection; black bar = logistic regression on the same inputs.",
    "roc_curves.png": "Mean ROC curves over 15 folds for the three quantum models, logistic regression on the same inputs, and logistic regression on full data.",
    "federated_convergence.png": "Federated training with DP at ε = 4 across 3, 5 and 10 sites; dashed line = centralised DP.",
  };
  $("ev-figures").innerHTML = e.figures.map((fg) => fg.exists
    ? `<figure><img src="${fg.url}" alt="${esc(cap[fg.file])}" class="max-w-full max-h-[520px] w-auto border border-line rounded-lg"><figcaption class="caption mt-2">${esc(cap[fg.file])}</figcaption></figure>`
    : `<p class="error">Figure missing: results/figures/${esc(fg.file)}</p>`).join("");
}

route();
