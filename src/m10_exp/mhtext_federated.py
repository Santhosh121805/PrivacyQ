"""Phase 6 (box 13): SIMULATED federated learning on the Mental Health Text dataset, NSCH method held fixed.

Task function: exp4_federated.run_task, unchanged (FedAvg, client-side Opacus DP-SGD at the NSCH epsilon, 30
rounds x 1 local epoch, 3 / 5 / 10 clients). Arms, rounds, epsilon, clients: read from configs/exp4_nsch2024.yaml.
Binary Depression vs Suicidal view, MiniLM embedding; folds are the Phase 5 DP-SGD folds (5x2), whose centralised
DP-SGD runs (results/mhtext/phase5/dp) are the centralised references, exactly as exp4 used exp3.

Clients are SIMULATED: partitions of one dataset on one machine, not separate institutions.
Partitions: iid (stratified random, as NSCH) and by_length - the non-IID substitute for NSCH's by_state, since the
posts carry no site or state: whole post-length deciles (fitted on the fold's inner-train rows) are assigned to
clients, so clients differ in what kind of post they hold.

Replication rule, declared before the run (primary partition iid, the only one both datasets have): the NSCH
pattern "federated VQC falls as clients increase while LogReg holds flat" replicates if (a) the VQC's ROC AUC
drop from 3 to 10 clients is significant, (b) LogReg's change from 3 to 10 clients is not significant or below
0.01 in size, and (c) the VQC's drop exceeds LogReg's drop significantly (paired over folds). NB-t, Holm over
the family {VQC drop, LogReg drop, difference}. The same statistics are computed for NSCH from its stored fold
metrics (no NSCH model is run).

python -m src.m10_exp.mhtext_federated --config configs/mhtext_phase6.yaml [--report-only]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.m8_eval.stats import add_holm, corrected_resampled_ttest, test_train_ratio, wilcoxon_p
from src.m10_exp.common import environment_record, limit_worker_threads, load_config
from src.m10_exp.mhtext_privacy import _all384, folds_for
from src.m10_exp.mhtext_quantum import view_data
from src.m10_exp.tuning import inner_split, stratified_subsample

ROOT = Path(__file__).resolve().parents[2]


def length_groups(X_len: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """Post-length decile of each inner-train row (cut points from those rows only)."""
    w = X_len[tr, 0]
    cuts = np.quantile(w, np.linspace(0, 1, 11)[1:-1])
    return np.searchsorted(cuts, w, side="right")


def build_tasks(cfg, p4, y, X, X_len, dev, source):
    nm = _all384(load_config(ROOT / cfg["nsch_method_config"]))
    exp, seed0 = nm["experiment"], int(p4["project"]["seed"])
    exp = {**exp, "partitions": cfg["partitions"]}
    frac, n_sub = float(nm["tuning"]["inner_val_fraction"]), int(nm["subsample"]["n"])
    tasks = []
    for f in folds_for(p4, dev, y, int(nm["cv"]["n_repeats"])):
        seed = seed0 + 1000 * f["repeat"] + f["fold"]
        regimes = {"subsample": stratified_subsample(f["train"], y, n_sub, seed), "full": f["train"]}
        prepared = {}
        for name, arm in exp["arms"].items():
            regime = arm["regime"]
            if regime not in prepared:
                tr, iv = inner_split(regimes[regime], y, frac, seed)
                prepared[regime] = dict(X_in=X[tr], X_iv=X[iv], X_ov=X[f["val"]], y_in=y[tr], y_iv=y[iv],
                                        y_ov=y[f["val"]], groups_in=length_groups(X_len, tr), n_fold_train=len(f["train"]))
            for n_clients in exp["clients"]:
                for partition in exp["partitions"]:
                    for eps in exp["epsilons"]:
                        e = "inf" if eps is None else f"{float(eps):g}"
                        tasks.append({**prepared[regime], "arm_name": name, "arm": arm, "exp": exp,
                                      "feature_spec": exp["feature_sets"][arm["feature_set"]], "n_clients": n_clients,
                                      "partition": "by_state" if partition == "by_length" else partition,
                                      "partition_label": partition, "epsilon": eps, "seed": seed,
                                      "repeat": f["repeat"], "fold": f["fold"], "source": source,
                                      "key": f"{name}_{partition}_c{n_clients}_eps{e}_r{f['repeat']}f{f['fold']}"})
    heavy = {"hybrid": 0, "vqc": 1, "mlp": 2, "logreg": 3}
    return sorted(tasks, key=lambda t: (heavy.get(t["arm"]["kind"], 9), -t["n_clients"])), nm


def run_keyed(task: dict):
    from src.m10_exp.exp4_federated import run_task
    out = run_task(task)           # partition "by_state" = group_partition on the supplied groups (length deciles)
    out["row"]["partition"] = task["partition_label"]
    hist = out["history"].assign(partition=task["partition_label"])
    return task["key"], out["row"], hist.to_dict(orient="list")


def run(cfg: dict, out: Path) -> None:
    p4 = load_config(ROOT / cfg["phase4_config"])
    _, y, feats, dev, _ = view_data(p4, cfg["view"])
    source = f"REAL:MHText-{cfg['view']}"
    tasks, nm = build_tasks(cfg, p4, y, feats["embedding"], feats["length"], dev, source)
    ckpt = out / "tasks"
    ckpt.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if not (ckpt / f"{t['key']}.json").exists()]
    print(f"phase 6: {len(tasks)} tasks, {len(tasks) - len(todo)} already done (source {source}; clients SIMULATED)", flush=True)
    from joblib import Parallel, delayed
    t0 = time.time()
    for i, (key, row, hist) in enumerate(Parallel(n_jobs=int(cfg["project"]["n_jobs"]), return_as="generator_unordered")(
            delayed(run_keyed)(t) for t in todo), 1):
        tmp = ckpt / f"{key}.json.tmp"
        tmp.write_text(json.dumps({"row": row, "history": hist}, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                       encoding="utf-8")
        tmp.replace(ckpt / f"{key}.json")
        print(f"[{i}/{len(todo)} {time.time() - t0:6.0f}s] {key}: AUC {row['roc_auc']:.3f}", flush=True)
    recs = [json.loads((ckpt / f"{t['key']}.json").read_text(encoding="utf-8")) for t in tasks]
    pd.DataFrame([r["row"] for r in recs]).to_csv(out / "fold_metrics.csv", index=False)
    pd.concat([pd.DataFrame(r["history"]) for r in recs]).to_csv(out / "round_history.csv", index=False)
    (out / "run.json").write_text(json.dumps({"tasks": len(tasks), "wall_seconds_this_launch": round(time.time() - t0, 1),
                                              "nsch_method_config": cfg["nsch_method_config"],
                                              "method_experiment_section": nm["experiment"], "partitions": cfg["partitions"],
                                              "environment": environment_record(cfg)}, indent=2, default=str), encoding="utf-8")
    print(f"done: wall {time.time() - t0:.0f}s this launch", flush=True)


# ---------------------------------------------------------------- report
def _auc(df, arm, part, n):
    s = df[(df.arm == arm) & (df.partition == part) & (df.n_clients == n)].set_index(["repeat", "fold"]).roc_auc
    return s.sort_index()


def replication(df: pd.DataFrame, part: str, arms=("vqc_sub", "logreg_full")) -> pd.DataFrame:
    ratio = test_train_ratio(df)
    q, c = arms
    dq = _auc(df, q, part, 10) - _auc(df, q, part, 3)
    dc = _auc(df, c, part, 10) - _auc(df, c, part, 3)
    idx = dq.dropna().index.intersection(dc.dropna().index)
    rows = []
    for name, x in (("VQC: 10 clients minus 3", dq[idx]), ("LogReg: 10 clients minus 3", dc[idx]),
                    ("VQC change minus LogReg change", dq[idx] - dc[idx])):
        t = corrected_resampled_ttest(x.to_numpy(), np.zeros(len(x)), ratio)
        rows.append({"partition": part, "quantity": name, "mean": t["mean_diff"], "ci95_low": t["ci95_low"],
                     "ci95_high": t["ci95_high"], "k": len(x), "p_nb_t": t["p"],
                     "wilcoxon_p": wilcoxon_p(x.to_numpy(), np.zeros(len(x)))})
    return add_holm(pd.DataFrame(rows), None)


def verdict(r: pd.DataFrame) -> tuple[bool, str]:
    v, l, d = (r.iloc[i] for i in range(3))
    a = v.p_nb_t_holm < 0.05 and v["mean"] < 0
    b = l.p_nb_t_holm >= 0.05 or abs(l["mean"]) < 0.01
    c = d.p_nb_t_holm < 0.05 and d["mean"] < 0
    return a and b and c, f"(a) VQC drop significant: {'yes' if a else 'no'}; (b) LogReg flat: {'yes' if b else 'no'}; " \
                          f"(c) VQC drop larger than LogReg's: {'yes' if c else 'no'}"


def report(cfg: dict, out: Path) -> None:
    df = pd.read_csv(out / "fold_metrics.csv")
    ns = pd.read_csv(ROOT / cfg["nsch_results"]["fold_metrics"])
    ns_sum = pd.read_csv(ROOT / cfg["nsch_results"]["summary"])
    ref = pd.read_csv(ROOT / cfg["centralised_reference"])
    ref["target_epsilon"] = ref.target_epsilon.map(lambda v: "inf" if str(v) in ("inf", "nan") else f"{float(v):g}")
    ratio = test_train_ratio(df)
    eps = "4"
    g = df.groupby(["arm", "partition", "n_clients"])
    summ = g[["roc_auc", "balanced_accuracy", "recall"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.join(g.size().rename("n_folds")).reset_index().assign(setting="federated")
    cen = ref[ref.target_epsilon == eps].groupby("arm")[["roc_auc", "balanced_accuracy", "recall"]].agg(["mean", "std"])
    cen.columns = [f"{a}_{b}" for a, b in cen.columns]
    cen = cen.reset_index().assign(setting="centralised", partition="-", n_clients=1)
    pd.concat([summ, cen[cen.arm.isin(summ.arm.unique())]], ignore_index=True).to_csv(out / "summary.csv", index=False)
    L = ["# Phase 6 — SIMULATED federated learning on Mental Health Text (binary view), beside NSCH", "",
         "Box 13. Clients are SIMULATED: partitions of one dataset on one machine, not separate institutions. exp4 method "
         "unchanged (FedAvg, 30 rounds x 1 local epoch, client-side DP-SGD at ε = 4, δ = 1e-5; arms VQC re-upload 4 qubits "
         "on 1,200 rows / PCA-4 and LogReg on every development training row / all 384 MiniLM dims), read from "
         "configs/exp4_nsch2024.yaml. 5x2 folds = the Phase 5 DP-SGD folds; centralised references are those Phase 5 runs "
         "(as exp4 used exp3). Partitions: iid (as NSCH) and by_length (non-IID substitute for NSCH's by_state: whole "
         "post-length deciles per client; the posts carry no site). NSCH numbers read from "
         f"{cfg['nsch_results']['summary']} and its fold_metrics.csv; no NSCH model was run.", "",
         "## ROC AUC by number of clients (ε = 4)", "",
         "| Arm | Partition | Centralised | 3 clients | 5 clients | 10 clients | NSCH partition | NSCH centralised | NSCH 3 | NSCH 5 | NSCH 10 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ("vqc_sub", "logreg_full"):
        cen = ref[(ref.arm == arm) & (ref.target_epsilon == eps)].roc_auc.mean()
        ncen = ns_sum[(ns_sum.arm == arm) & (ns_sum.setting == "centralised")].roc_auc_mean.iloc[0]
        for part, npart in (("iid", "iid"), ("by_length", "by_state")):
            cells = [f"{df[(df.arm == arm) & (df.partition == part) & (df.n_clients == n)].roc_auc.mean():.3f}" for n in (3, 5, 10)]
            ncells = [f"{ns_sum[(ns_sum.arm == arm) & (ns_sum.partition == npart) & (ns_sum.n_clients == n)].roc_auc_mean.iloc[0]:.3f}"
                      for n in (3, 5, 10)]
            L.append(f"| {arm} | {part} | {cen:.3f} | " + " | ".join(cells) + f" | {npart} | {ncen:.3f} | " + " | ".join(ncells) + " |")

    tests = []
    for arm in ("vqc_sub", "logreg_full"):
        base = ref[(ref.arm == arm) & (ref.target_epsilon == eps)].set_index(["repeat", "fold"]).roc_auc
        for part in ("iid", "by_length"):
            for n in (3, 5, 10):
                fed = _auc(df, arm, part, n)
                idx = fed.index.intersection(base.index)
                tests.append({"arm": arm, "partition": part, "n_clients": n, "federated": fed[idx].mean(),
                              "centralised_dp": base[idx].mean(), "diff": (fed[idx] - base[idx]).mean(),
                              "p_nb_t": corrected_resampled_ttest(fed[idx].to_numpy(), base[idx].to_numpy(), ratio)["p"],
                              "wilcoxon_p": wilcoxon_p(fed[idx].to_numpy(), base[idx].to_numpy())})
    tests = add_holm(pd.DataFrame(tests), ["arm"])
    tests.to_csv(out / "tests_federated_vs_centralised.csv", index=False)
    L += ["", "## Federated vs centralised DP-SGD at the same ε (NB-t, Holm within arm)", "",
          "| Arm | Partition | Clients | Federated | Centralised DP | Diff | Holm p |", "|---|---|---|---|---|---|---|"]
    for _, t in tests.iterrows():
        L.append(f"| {t.arm} | {t.partition} | {t.n_clients} | {t.federated:.3f} | {t.centralised_dp:.3f} | {t['diff']:+.3f} | {t.p_nb_t_holm:.3f} |")

    L += ["", "## Replication test: does the federated VQC fall with more clients while LogReg holds flat?", "",
          "Rule declared before the run (module docstring): primary partition iid. Replicates if (a) the VQC's AUC change "
          "from 3 to 10 clients is a significant drop, (b) LogReg's change is not significant or below 0.01, and (c) the "
          "VQC's drop is significantly larger than LogReg's (paired over folds). NB-t, Holm within each block. The NSCH "
          "rows apply the same statistics to its stored fold metrics.", "",
          "| Dataset | Partition | Quantity | Mean change in AUC [95% CI] | Holm p |", "|---|---|---|---|---|"]
    allrep, verdicts = [], {}
    for label, data, parts in (("Mental Health Text", df, ("iid", "by_length")), ("NSCH 2024", ns, ("iid", "by_state"))):
        for part in parts:
            r = replication(data, part).assign(dataset=label)
            allrep.append(r)
            verdicts[(label, part)] = verdict(r)
            for _, x in r.iterrows():
                L.append(f"| {label} | {part} | {x.quantity} | {x['mean']:+.3f} [{x.ci95_low:+.3f}, {x.ci95_high:+.3f}] | {x.p_nb_t_holm:.3f} |")
    pd.concat(allrep).to_csv(out / "tests_replication_clients.csv", index=False)
    L += [""]
    for (label, part), (ok, why) in verdicts.items():
        L.append(f"- **{label}, {part}: {'pattern present' if ok else 'pattern not present'}** — {why}.")
    prim = verdicts[("Mental Health Text", "iid")][0] and verdicts[("NSCH 2024", "iid")][0]
    L += ["", f"**Verdict (primary, iid on both datasets): {'REPLICATES' if prim else 'DOES NOT REPLICATE'}.**", "",
          "Communication per task is in fold_metrics.csv (bytes_per_round, total_bytes); convergence per round in "
          "round_history.csv (logged for plots only, never used for a choice).", ""]
    (out / "phase6.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out / 'phase6.md'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()
    cfg = load_config(ROOT / a.config)
    out = ROOT / cfg["project"]["output_dir"]
    limit_worker_threads()
    if not a.report_only:
        run(cfg, out)
    report(cfg, out)


if __name__ == "__main__":
    main()
