"""Phase 5 report: privacy on the Mental Health Text dataset beside the NSCH privacy results.

Reads results/mhtext/phase5/{reid,mia,dp,ldp}/ and the NSCH result files named in configs/mhtext_phase5.yaml
(nsch_results). Nothing from NSCH is recomputed. A part that has not run yet is reported as missing.

python -m src.m10_exp.mhtext_phase5_report
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.m9_audit.disclosure import RULE, SUPPRESSED, is_suppressed, numeric
from src.m10_exp.common import load_config
from src.m10_exp.exp3_report import cost_of_privacy, quantum_vs_classical

ROOT = Path(__file__).resolve().parents[2]
EPS = ["0.5", "1", "2", "4", "8", "inf"]


def pct(x) -> str:
    if is_suppressed(x):
        return SUPPRESSED
    return f"{100 * float(x):.1f} %"


def f3(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def eps_str(v) -> str:
    return "inf" if str(v) in ("inf", "nan", "None") else f"{float(v):g}"


def section_reid(cfg, d: Path, L: list, facts: dict) -> None:
    r = d / "reid"
    if not (r / "linkage.csv").exists():
        L += ["## Re-identification — not run yet", ""]
        return
    fr, sd, lk = (pd.read_csv(r / f) for f in ("fragment_risk.csv", "self_disclosures.csv", "linkage.csv"))
    ns = pd.read_csv(ROOT / cfg["nsch_results"]["reidentification"])
    L += ["## 1. Re-identification risk (box 12) beside NSCH", "",
          "Same prosecutor-risk scale as the NSCH table (results/reidentification/risk.csv): the attacker knows "
          "something about a person and that they are in the released table; risk = 1 / (rows that match what the "
          "attacker knows). NSCH: the attacker knows quasi-identifier values. Text: the attacker knows a fragment of "
          "the person's post (k consecutive words, one random fragment per post, seed 2026). Development rows only; the "
          "locked hold-out is not read (as for NSCH).", "",
          "| Released table | What the attacker knows | Rows | Unique (risk = 1) | Rows violating k = 5 | Average risk | Maximum risk |",
          "|---|---|---|---|---|---|---|"]
    for _, x in ns.iterrows():
        L.append(f"| NSCH 2024 survey | {x.table} | {int(x.n_rows):,} | {pct(x.uniqueness)} | {pct(x.k_violation_rate)} | "
                 f"{f3(x.avg_prosecutor_risk)} | {f3(x.max_prosecutor_risk)} |")
    dev = fr[(fr.corpus == "dev") & fr.attacker_knows.str.startswith("a ")]
    for _, x in dev.iterrows():
        L.append(f"| Mental Health Text posts, {x.text} | {x.attacker_knows} | {int(x.n_rows):,} | {pct(x.uniqueness)} | "
                 f"{pct(x.k_violation_rate)} | {f3(x.avg_prosecutor_risk)} | {f3(x.max_prosecutor_risk)} |")
    any_u = fr[fr.attacker_knows.str.startswith("any ")]
    L += ["", "Share of posts that contain at least one span found in no other post (an attacker who knows any such "
          "span singles the post out with certainty):", "",
          "| Corpus | Text | Span | Posts | Posts with a unique span |", "|---|---|---|---|---|"]
    for _, x in any_u.iterrows():
        L.append(f"| {x.corpus} | {x.text} | {x.attacker_knows} | {int(x.n_rows):,} | {pct(x.uniqueness)} |")
    big = fr[(fr.corpus == "dev+quarantine") & fr.attacker_knows.str.startswith("a ") & (fr.text == "as released")]
    L += ["", "Sensitivity, larger corpus (development rows + quarantined block, "
          f"{int(big.n_rows.iloc[0]):,} posts, as released): uniqueness "
          + ", ".join(f"{x.attacker_knows.split('a ')[1]} {pct(x.uniqueness)}" for _, x in big.iterrows()) + ".", ""]

    s = sd.set_index("corpus").loc["dev"]
    L += ["### Quasi-identifiers written into the text", "",
          "NSCH's quasi-identifiers (age, sex, state) are columns that can be generalised or suppressed. In free text "
          "the author writes them into the post. Regex counts on the development posts (a lower bound: patterns such "
          "as '(16F)', 'I am 15', '17 years old', a U.S. state name; counted, never read or printed):", "",
          "| Disclosure | Posts |", "|---|---|",
          f"| States an age | {pct(s.states_age)} |", f"| States an age of 12-17 | {pct(s.states_age_12_17)} |",
          f"| States sex/gender | {pct(s.states_sex)} |", f"| Names a U.S. state | {pct(s.names_us_state)} |",
          f"| Mentions a school level | {pct(s.mentions_school_level)} |", f"| Age and sex | {pct(s.age_and_sex)} |",
          f"| Age, sex and state | {pct(s.age_sex_and_state)} |", f"| Any of age, sex, state | {pct(s.any_of_age_sex_state)} |", ""]

    L += ["### Linking a fragment to the released vectors", "",
          "If only model inputs are released (not text), an attacker embeds the known fragment with the public MiniLM "
          "model and looks for the nearest released vector. Rank 1 = the post is re-identified. Released vectors: the "
          "384-dim MiniLM embedding, or the 8 PCA encoder inputs the quantum circuit receives (worst case: the PCA "
          "projection is public), with Laplace local DP on the released encoder inputs at the exp5 budgets.", "",
          "| Released | Fragment words | ε_input | Linked (top-1) | Top-10 | Median rank | Chance (top-1) |",
          "|---|---|---|---|---|---|---|"]
    for _, x in lk.iterrows():
        L.append(f"| {x.released} | {x.fragment_words} | {'∞' if eps_str(x.epsilon_input) == 'inf' else eps_str(x.epsilon_input)} | "
                 f"{pct(x.top1)} | {pct(x.top10)} | {x.median_rank:,.0f} | {x.chance_top1:.5f} |")
    L.append("")
    n = ns.set_index("table")
    facts.update(
        nsch_unique_7=n.loc["7 demographic QIs, raw values", "uniqueness"],
        nsch_unique_state=n.loc["7 demographic QIs, raw values + U.S. state", "uniqueness"],
        nsch_kanon=n.loc[[t for t in n.index if t.startswith("7 demographic QIs after k-anonymity") and "state" not in t][0]],
        text_unique={int(x.span_words): x.uniqueness for _, x in dev[dev.text == "as released"].iterrows()},
        text_unique_scrub={int(x.span_words): x.uniqueness for _, x in dev[dev.text != "as released"].iterrows()},
        text_avg={int(x.span_words): x.avg_prosecutor_risk for _, x in dev[dev.text == "as released"].iterrows()},
        any5=any_u[(any_u.corpus == "dev") & (any_u.text == "as released") & (any_u.span_words == 5)].uniqueness.iloc[0],
        disclose=s, link=lk)


def section_mia(cfg, d: Path, L: list, facts: dict) -> None:
    f = d / "mia" / "fold_metrics.csv"
    if not f.exists():
        L += ["## 2. Membership inference — not run yet", ""]
        return
    res = pd.read_csv(f)
    res["target_epsilon"] = res.target_epsilon.map(eps_str)
    g = res.groupby(["arm", "target_epsilon"])
    s = g[["attack_auc", "tpr_at_1pct_fpr", "advantage"]].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s["n_members"] = g.n_members.first()
    s = s.reset_index()
    s.to_csv(d / "mia" / "summary.csv", index=False)
    ns = pd.read_csv(ROOT / cfg["nsch_results"]["mia"])
    ns["target_epsilon"] = ns.target_epsilon.map(eps_str)
    nsi = ns.set_index(["arm", "target_epsilon"])
    L += ["## 2. Membership inference (box 12) beside NSCH", "",
          "exp6 method unchanged: loss-threshold attack on equal numbers of members (the model's training rows) and "
          "non-members (unseen validation rows), 5 folds; attack ROC AUC 0.5 = no membership signal; mlp_overfit is the "
          "positive control (no early stopping, 200 epochs). Text inputs: all 384 MiniLM dims (LogReg, MLP), PCA-4 "
          "(VQC), PCA-8 (QSVM).", "",
          "| Arm | ε | Text: attack AUC | Text: TPR @ 1 % FPR | Text: advantage | Text members | NSCH: attack AUC | NSCH: advantage |",
          "|---|---|---|---|---|---|---|---|"]
    order = {"dp_logreg_full": 0, "dp_vqc": 1, "qsvm": 2, "mlp_overfit": 3}
    s = s.sort_values(["arm", "target_epsilon"], key=lambda c: c.map(order) if c.name == "arm" else c.map(EPS.index))
    for _, x in s.iterrows():
        k = (x.arm, x.target_epsilon)
        na, nadv = (nsi.loc[k, "attack_auc_mean"], nsi.loc[k, "advantage_mean"]) if k in nsi.index else (np.nan, np.nan)
        L.append(f"| {x.arm} | {'∞' if x.target_epsilon == 'inf' else x.target_epsilon} | {f3(x.attack_auc_mean)} ± "
                 f"{f3(x.attack_auc_std)} | {f3(x.tpr_at_1pct_fpr_mean)} | {f3(x.advantage_mean)} | {int(x.n_members):,} | "
                 f"{f3(na)} | {f3(nadv)} |")
    L.append("")
    facts["mia"], facts["mia_nsch"] = s.set_index(["arm", "target_epsilon"]), nsi


def section_dp(cfg, d: Path, L: list, facts: dict) -> None:
    f = d / "dp" / "fold_metrics.csv"
    if not f.exists():
        L += ["## 3. DP-SGD privacy-utility — not run yet", ""]
        return
    res = pd.read_csv(f)
    res["target_epsilon"] = res.target_epsilon.map(eps_str)
    cost = cost_of_privacy(res)
    cost.to_csv(d / "dp" / "tests_private_vs_nonprivate.csv", index=False)
    run = json.loads((d / "dp" / "run.json").read_text(encoding="utf-8"))
    pairs = run["method_experiment_section"].get("comparisons", [])
    qvc = quantum_vs_classical(res, pairs) if pairs else None
    if qvc is not None:
        qvc.to_csv(d / "dp" / "tests_quantum_vs_classical.csv", index=False)
    nsd = ROOT / cfg["nsch_results"]["dp"]
    ncost = pd.read_csv(nsd / "tests_private_vs_nonprivate.csv")
    ncost["epsilon"] = ncost.epsilon.map(eps_str)
    nci = ncost[ncost.metric == "roc_auc"].set_index(["arm", "epsilon"])
    tab = res.groupby(["arm", "target_epsilon"]).roc_auc.agg(["mean", "std"]).reset_index()
    tab.to_csv(d / "dp" / "summary.csv", index=False)
    ti = tab.set_index(["arm", "target_epsilon"])
    c = cost[cost.metric == "roc_auc"].set_index(["arm", "epsilon"])
    L += ["## 3. DP-SGD privacy-utility (box 4 mechanism, box 12 cost) beside NSCH", "",
          "exp3 method unchanged (Opacus DP-SGD, δ = 1e-5, fixed hyperparameters, 5x2 folds). Arms: VQC re-upload 4 "
          "qubits and LogReg on the same 1,200 rows and PCA-4 inputs; LogReg on every development training row and all "
          "384 dims. Cell: ROC AUC (change vs the same model without DP; NB-t Holm p within arm). NSCH: change, from "
          "its tests_private_vs_nonprivate.csv.", "",
          "| Arm | " + " | ".join(f"ε = {'∞' if e == 'inf' else e}" for e in EPS) + " |", "|---|" + "---|" * len(EPS)]
    for arm in ["vqc_sub", "logreg_sub", "logreg_full"]:
        cells = []
        for e in EPS:
            if (arm, e) not in ti.index:
                cells.append("–")
            elif e == "inf":
                cells.append(f"{f3(ti.loc[(arm, e), 'mean'])}")
            else:
                ch, p = c.loc[(arm, e), "change"], c.loc[(arm, e), "p_nb_t_holm"]
                nch = nci.loc[(arm, e), "change"] if (arm, e) in nci.index else np.nan
                cells.append(f"{f3(ti.loc[(arm, e), 'mean'])} ({ch:+.3f}, p {f3(p)}; NSCH {nch:+.3f})")
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    if qvc is not None:
        L += ["", "VQC vs LogReg on the same rows and inputs at each ε (NB-t, Holm within the pair):", "",
              "| ε | VQC AUC | LogReg AUC | Diff | VQC wins | Holm p | AUC drop VQC / LogReg |", "|---|---|---|---|---|---|---|"]
        for _, x in qvc.iterrows():
            drop = "" if pd.isna(x.get("quantum_auc_drop")) else f"{x.quantum_auc_drop:.3f} / {x.classical_auc_drop:.3f}"
            L.append(f"| {'∞' if x.epsilon == 'inf' else x.epsilon} | {f3(x.quantum_auc)} | {f3(x.classical_auc)} | "
                     f"{x.auc_diff:+.3f} | {x.quantum_wins}/{x.k} | {f3(x.p_nb_t_holm)} | {drop} |")
    L.append("")
    facts["dp"], facts["dp_cost"], facts["dp_nsch"] = ti, c, nci


def section_ldp(cfg, d: Path, L: list, facts: dict) -> None:
    f = d / "ldp" / "fold_metrics.csv"
    if not f.exists():
        L += ["## 4. Input-stage local DP — not run yet", ""]
        return
    res = pd.read_csv(f)
    res["epsilon_input"] = res.epsilon_input.map(eps_str)
    tab = res.groupby(["arm", "regime", "epsilon_input"]).roc_auc.agg(["mean", "std"]).reset_index()
    tab.to_csv(d / "ldp" / "summary.csv", index=False)
    ns = pd.read_csv(ROOT / cfg["nsch_results"]["ldp"])
    ns["epsilon_input"] = ns.epsilon_input.map(eps_str)
    nsi = ns[ns.k_anon == "off"].set_index(["arm", "regime", "epsilon_input"])
    ti = tab.set_index(["arm", "regime", "epsilon_input"])
    order = ["inf", "32", "16", "8", "4"]
    L += ["## 4. Input-stage local DP before encoding (box 4) beside NSCH", "",
          "exp5 method unchanged: Laplace pure ε-LDP on the 8 PCA encoder inputs of training rows, validation rows clean; "
          "dp_logreg adds DP-SGD at ε_training = 4. k-anonymity: NOT APPLICABLE to free text (there are no "
          "quasi-identifier columns to generalise; the identifying content is the text itself - see section 1). "
          "Cell: text ROC AUC (NSCH, k-anonymity off).", "",
          "| Arm | Regime | " + " | ".join(f"ε_input = {'∞' if e == 'inf' else e}" for e in order) + " |",
          "|---|---|" + "---|" * len(order)]
    for (arm, regime) in [("qsvm", "subsample"), ("logreg", "subsample"), ("dp_logreg", "subsample"), ("logreg", "full")]:
        cells = []
        for e in order:
            t = ti.loc[(arm, regime, e), "mean"] if (arm, regime, e) in ti.index else np.nan
            n = nsi.loc[(arm, regime, e), "roc_auc_mean"] if (arm, regime, e) in nsi.index else np.nan
            cells.append(f"{f3(t)} ({f3(n)})")
        L.append(f"| {arm} | {regime} | " + " | ".join(cells) + " |")
    L.append("")
    facts["ldp"] = ti


def main() -> None:
    cfg = load_config(ROOT / "configs/mhtext_phase5.yaml")
    d = ROOT / cfg["project"]["output_dir"]
    facts: dict = {}
    L = ["# Phase 5 — privacy on Mental Health Text (binary Depression vs Suicidal view), beside NSCH", "",
         "Boxes 4 (privacy-preserving mechanisms) and 12 (privacy evaluation). Source REAL:MHText. Every model-based "
         "privacy parameter is read from the NSCH experiment configs (exp3, exp5, exp6) and the NSCH numbers are read "
         "from its results/ files; no NSCH experiment was run. Inputs: frozen MiniLM sentence encoder (BERT family) "
         "embeddings. The locked hold-out is not used. Quantum circuits are simulated; a quantum model is not a privacy "
         "mechanism.", "",
         f"**Disclosure control.** {RULE} Applied in this report to 10 cells: 8 in the linkage table (posts linked under "
         "local DP at ε_input 16, 8 and 4) and 2 in the self-disclosure table (posts stating age, sex and state together, "
         "in both corpora). Secondary suppression was checked and not needed: the age-sex-state cell cannot be "
         "rebuilt from the published cells (that would need the age-state and sex-state pairs, which are not published, "
         "and both corpus rows are suppressed), and a top-1 linkage cell cannot be derived from top-5, top-10, median "
         "rank or MRR, which only bound it. Zero cells are kept; they identify no one.", ""]
    body: list = []
    for sec in (section_reid, section_mia, section_dp, section_ldp):
        sec(cfg, d, body, facts)
    L += summary(facts) + body + commentary(facts)
    (d / "phase5.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {d / 'phase5.md'}")


def summary(facts: dict) -> list[str]:
    if not all(k in facts for k in ("text_unique", "mia", "dp", "ldp")):
        return []
    lk, m, mn = facts["link"], facts["mia"], facts["mia_nsch"]
    pca12 = lk[(lk.released.str.startswith("PCA")) & (lk.fragment_words == 12) & (lk.epsilon_input.map(eps_str) == "inf")].iloc[0]
    ldp = facts["ldp"]
    cost32 = ldp.loc[("qsvm", "subsample", "inf"), "mean"] - ldp.loc[("qsvm", "subsample", "32"), "mean"]
    c, dn = facts["dp_cost"], facts["dp_nsch"]
    return [
        "## Summary", "",
        f"- **The 8 inputs fed to the quantum circuit (step 5) still identify the post {pct(pca12.top1)} of the time** "
        "from a 12-word fragment, when the PCA projection is known. 8 PCA components are not anonymous. Keeping the "
        "projection secret would lower linkage, but secrecy of a projection is not a privacy guarantee and is not claimed.",
        f"- **A post is far more identifying than a survey row:** a 5-word fragment singles out {pct(facts['text_unique'][5])} "
        f"of posts, against {pct(facts['nsch_unique_state'])} of NSCH adolescents on 7 demographics + state (raw) and "
        f"{pct(facts['nsch_kanon'].uniqueness)} on the 7 demographics after k-anonymity (k = 5). Scrubbing identifiers leaves it at "
        f"{pct(facts['text_unique_scrub'][5])}; k-anonymity has no columns to act on.",
        f"- **The small quantum models leak little, and that comes from their low capacity, not from anything quantum.** "
        f"Membership-attack AUC: QSVM {f3(m.loc[('qsvm', 'inf'), 'attack_auc_mean'])}, VQC "
        f"{f3(m.loc[('dp_vqc', 'inf'), 'attack_auc_mean'])}; the overfitted MLP control reaches "
        f"{f3(m.loc[('mlp_overfit', 'inf'), 'attack_auc_mean'])} (NSCH {f3(mn.loc[('mlp_overfit', 'inf'), 'attack_auc_mean'])}).",
        f"- **Local DP at ε = 32 defeats the linkage attack we implemented, at a QSVM cost of {cost32:.3f} AUC. The formal "
        "guarantee at this budget is weak and must not be read as a general privacy claim.**",
        f"- DP-SGD at ε = 0.5 costs the same-rows LogReg {-c.loc[('logreg_sub', '0.5'), 'change']:.3f} AUC on text against "
        f"{-dn.loc[('logreg_sub', '0.5'), 'change']:.3f} on NSCH, about tenfold less; the cause is not established.",
        ""]


def commentary(facts: dict) -> list[str]:
    if not all(k in facts for k in ("text_unique", "mia", "dp", "ldp")):
        return ["## What differs from NSCH — written once every part has run", ""]
    u, us, k = facts["text_unique"], facts["text_unique_scrub"], facts["nsch_kanon"]
    m, mn = facts["mia"], facts["mia_nsch"]
    lk = facts["link"]
    lk_emb = lk[(lk.released.str.startswith("MiniLM")) & (lk.fragment_words == 12)].iloc[0]
    lk_pca = lk[(lk.released.str.startswith("PCA")) & (lk.fragment_words == 12) & (lk.epsilon_input.map(eps_str) == "inf")].iloc[0]
    lk_32 = numeric(lk[(lk.released.str.startswith("PCA")) & (lk.epsilon_input.map(eps_str) == "32")].top1).max()
    ldp = facts["ldp"]
    q_inf, q_32 = ldp.loc[("qsvm", "subsample", "inf"), "mean"], ldp.loc[("qsvm", "subsample", "32"), "mean"]
    d, c, dn = facts["dp"], facts["dp_cost"], facts["dp_nsch"]
    s = facts["disclose"]
    return [
        "## What differs from NSCH (explicit comparison)", "",
        f"**1. A post is far more identifying than a row of survey codes.** On NSCH the most identifying table "
        f"(7 demographic quasi-identifiers + U.S. state, raw values) leaves {pct(facts['nsch_unique_state'])} of "
        f"adolescents unique, and the pipeline's k-anonymity (k = 5) brings that to {pct(k.uniqueness)} with average "
        f"prosecutor risk {f3(k.avg_prosecutor_risk)}. On this dataset an attacker who knows just 5 consecutive words of "
        f"a post singles it out in {pct(u[5])} of cases (average risk {f3(facts['text_avg'][5])}); 8 words: {pct(u[8])}. "
        f"Even a 3-word fragment ({pct(u[3])} unique) is more identifying than NSCH's raw 10-column demographic table. "
        f"{pct(facts['any5'])} of posts contain a 5-word span that appears in no other post.",
        "",
        f"**2. Removing direct identifiers does not help, and k-anonymity has nothing to act on.** Scrubbing URLs, "
        f"e-mails, handles, community names and every number leaves 5-word uniqueness at {pct(us[5])} (from "
        f"{pct(u[5])}). Explicit quasi-identifiers are rare in the text (an age in {pct(s.states_age)} of posts, a U.S. "
        f"state in {pct(s.names_us_state)}); the identifying content is the wording itself, which no column-level "
        "generalisation or suppression can reach. On NSCH the same threat was answered by k-anonymity; for text there is "
        "no equivalent, so the only safe release is no release of text. The source corpora are public social-media posts "
        "(drawn largely from Reddit), so a fragment can also be searched for on the open web, where the original post "
        "may still be online with its author's account (not checked here: no web search was run on any post). This "
        "re-identification risk is a property of the data, not of any model trained on it.",
        "",
        f"**3. The 8 numbers fed to the quantum circuit still identify the post about a third of the time.** These are "
        f"the PCA encoder inputs of architecture step 5 (quantum encoding). Most readers would assume 8 PCA components "
        f"are anonymous; they are not. An attacker who knows a 12-word fragment, embeds it with the public MiniLM model "
        f"and projects it with the same PCA finds the right post among {int(lk_pca.n_queries):,} as the nearest released "
        f"8-number vector {pct(lk_pca.top1)} of the time (chance {lk_pca.chance_top1:.5f}); with a 20-word fragment, "
        f"{pct(lk[(lk.released.str.startswith('PCA')) & (lk.fragment_words == 20) & (lk.epsilon_input.map(eps_str) == 'inf')].top1.iloc[0])}. "
        f"Through the full 384-dim MiniLM embedding it is {pct(lk_emb.top1)}. Encoding into a quantum state does not "
        "change this: the circuit's input is these 8 numbers. This assumes the projection is known to the attacker. If "
        "the projection were kept secret, linkage would fall; but secrecy of a projection is not a privacy guarantee, "
        "and we do not claim it as one.",
        "",
        f"**3b. Local DP at ε = 32 defeats the linkage attack we implemented, at a QSVM cost of {q_inf - q_32:.3f} AUC** "
        f"(top-1 linkage at most {pct(lk_32)}; QSVM ROC AUC {f3(q_32)} from {f3(q_inf)}; on NSCH the same ε_input cost "
        "the QSVM 0.027). The formal guarantee at this budget is weak and must not be read as a general privacy claim: "
        "e^32 is astronomically large, so the bound is close to vacuous. It does not bound a stronger attack.",
        "",
        f"**4. Text models memorise individuals more readily.** The deliberately overfitted MLP (positive control) "
        f"reaches membership-attack AUC {f3(m.loc[('mlp_overfit', 'inf'), 'attack_auc_mean'])} with advantage "
        f"{f3(m.loc[('mlp_overfit', 'inf'), 'advantage_mean'])} on text, against "
        f"{f3(mn.loc[('mlp_overfit', 'inf'), 'attack_auc_mean'])} / {f3(mn.loc[('mlp_overfit', 'inf'), 'advantage_mean'])} "
        f"on NSCH with the same training-row count (1,200): unique, high-dimensional posts are easy to memorise. The "
        f"properly trained models leak little on both datasets, but the non-private LogReg on all 384 dims is "
        f"measurably above chance on text ({f3(m.loc[('dp_logreg_full', 'inf'), 'attack_auc_mean'])} vs NSCH "
        f"{f3(mn.loc[('dp_logreg_full', 'inf'), 'attack_auc_mean'])}), and DP-SGD lowers it to "
        f"{f3(m.loc[('dp_logreg_full', '0.5'), 'attack_auc_mean'])} at ε = 0.5. The VQC and QSVM stay at "
        f"{f3(m.loc[('dp_vqc', 'inf'), 'attack_auc_mean'])} and {f3(m.loc[('qsvm', 'inf'), 'attack_auc_mean'])}, as on "
        "NSCH. The small quantum models leak little, and that comes from their low capacity, not from anything quantum.",
        "",
        f"**5. DP-SGD utility cost at ε = 0.5.** Same-rows LogReg: {-c.loc[('logreg_sub', '0.5'), 'change']:.3f} ROC AUC "
        f"on text vs {-dn.loc[('logreg_sub', '0.5'), 'change']:.3f} on NSCH, about a tenfold difference. VQC: "
        f"{-c.loc[('vqc_sub', '0.5'), 'change']:.3f} vs {-dn.loc[('vqc_sub', '0.5'), 'change']:.3f}. LogReg on all inputs "
        f"and every development row: {-c.loc[('logreg_full', '0.5'), 'change']:.3f} vs "
        f"{-dn.loc[('logreg_full', '0.5'), 'change']:.3f}. The cause of the difference is not established. The VQC stays "
        "below LogReg at every ε, as on NSCH. DP-SGD bounds what a trained model reveals about its training rows; it "
        "does nothing for points 1-3, which concern the released data itself.",
        "",
        "**Bottom line.** On NSCH, privacy is mostly about the model, because the released table can be made "
        "k-anonymous. On free text, the data is the dominant risk: posts are effectively unique and stay unique after "
        "scrubbing, and even their embeddings link back. Text must stay where it was collected; only protected model "
        "inputs (local DP) or protected models (DP-SGD, federated - Phase 6) should leave it.",
        ""]


if __name__ == "__main__":
    main()
