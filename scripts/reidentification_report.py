"""Re-identification risk of the NSCH 2024 adolescent table (box 12).

Measured on the 17,431 dev rows (the locked hold-out is not touched) for several
quasi-identifier sets, before and after the k-anonymity transform used in the pipeline.

Usage: python scripts/reidentification_report.py  -> results/reidentification/{risk.csv, risk.md}
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.m7_privacy.input_privacy import KAnonymizer, NSCH_QUASI_IDENTIFIERS  # noqa: E402
from src.m9_audit.reidentification import reidentification_risk  # noqa: E402
from src.m10_exp.common import holdout_split, load_config, load_dataset  # noqa: E402

RAW10 = ["sc_age_years", "sc_sex", "sc_racer", "sc_hispanic_r", "fpl_i1", "family_r", "hhlanguage", "house_gen",
         "totkids_r", "higrade"]
DEMO7 = list(NSCH_QUASI_IDENTIFIERS)


def main() -> None:
    cfg = load_config(ROOT / "configs/exp5_nsch2024.yaml")
    loader, frame = load_dataset(cfg)
    labels = frame[loader.label_column].to_numpy()
    dev, _ = holdout_split(labels, 0.2, int(cfg["project"]["seed"]))
    d = frame.iloc[dev].reset_index(drop=True)
    anon = KAnonymizer(k=5).fit(d)
    gen = anon._generalise(d)                  # generalisation only (bands), before suppression
    kanon = anon.transform(d)                  # generalisation + local suppression (pipeline output)
    rows = []
    for name, table, qis in [
        ("10 demographic/household QIs, raw values", d, RAW10),
        ("7 demographic QIs, raw values", d, DEMO7),
        ("7 demographic QIs, raw values + U.S. state", d, DEMO7 + ["state_fips"]),
        ("7 demographic QIs, generalised (age and poverty bands)", gen, DEMO7),
        ("7 demographic QIs, generalised + state", gen, DEMO7 + ["state_fips"]),
        ("7 demographic QIs after k-anonymity (k = 5, as used in the pipeline)", kanon, DEMO7),
        ("7 demographic QIs after k-anonymity + state", kanon.assign(state_fips=d["state_fips"]), DEMO7 + ["state_fips"]),
    ]:
        rows.append({"table": name, **reidentification_risk(table, qis, k=5)})
    out = pd.DataFrame(rows)
    outdir = ROOT / "results" / "reidentification"
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "risk.csv", index=False)
    lines = [f"# Re-identification risk — {loader.source_tag()}", "",
             f"{len(d):,} dev rows (locked hold-out excluded). k = 5. Suppression applied by the k-anonymity step: "
             f"{anon.report_['n_suppressed']:,} rows ({anon.report_['suppression_rate']:.1%}) had their quasi-identifiers blanked.", "",
             "| Quasi-identifier set | Classes | Unique rows | Rows in classes < 5 | Avg risk | Max risk |", "|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        lines.append(f"| {r.table} | {r.n_classes:,} | {r.uniqueness:.1%} | {r.k_violation_rate:.1%} | "
                     f"{r.avg_prosecutor_risk:.3f} | {r.max_prosecutor_risk:.2f} |")
    (outdir / "risk.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
