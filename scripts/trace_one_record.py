"""Follow one adolescent record through the QSVM pipeline of Experiment 2, fold r0f0.

Usage: python scripts/trace_one_record.py
Prints the record's state at each stage (public-use NSCH data; no identifiers).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.m1_data.nsch import FEATURES  # noqa: E402
from src.m2_preprocess.preprocessing import TabularPreprocessor  # noqa: E402
from src.m3_features.features import FeatureTransformer  # noqa: E402
from src.m4_encoding import make_encoding  # noqa: E402
from src.m5_models.quantum import QuantumKernelSVC  # noqa: E402
from src.m10_exp.common import cv_folds, holdout_split, load_config, load_dataset  # noqa: E402
from src.m10_exp.tuning import inner_split, select_and_fit, stratified_subsample  # noqa: E402

np.set_printoptions(precision=3, suppress=True, linewidth=110)
cfg = load_config("configs/exp2_nsch2024.yaml")
loader, frame = load_dataset(cfg)
labels = frame.label.to_numpy()
dev, _ = holdout_split(labels, 0.2, 2026)
f = cv_folds(dev, labels, 5, 3, 2026)[0]
seed = 2026
sub = stratified_subsample(f["train"], labels, 1500, seed)
tr, iv = inner_split(sub, labels, 0.2, seed)

# the first depressed adolescent in this fold's validation rows
row = int(f["val"][labels[f["val"]] == 1][0])
print(f"== 0. Record: loader row {row} (outer validation fold r0f0, never used for training here); true label = {labels[row]}")
raw = pd.read_stata(cfg["data"]["paths"][0], convert_categoricals=False)
raw.columns = raw.columns.str.lower()
raw = raw[raw.sc_age_years.between(12, 17)]
print(f"   raw NSCH file: 457 columns. Diagnosis fields for this child: K2Q32A (ever told depression) / K2Q32B (currently): "
      "reconstructed as label only, never used as features")
print("\n== 1. Whitelisted (66 features + 3 metadata), first 12 shown:")
rec = frame.iloc[row]
print(rec[list(FEATURES)[:12]].to_string())
print(f"   missing answers in this record: {int(rec[list(FEATURES)].isna().sum())}; metadata (not features): "
      f"state_fips={rec.state_fips}, survey_weight={rec.survey_weight:.1f}")

prep = TabularPreprocessor(identifier_columns=loader.metadata_columns)
X_tr = prep.fit_transform(frame.iloc[tr], "label")
x = prep.transform(frame.iloc[[row]], "label")
print(f"\n== 2. Imputed + one-hot (fitted on {len(tr)} inner-train rows only): shape {x.shape}")
print("   first 10 values:", x.to_numpy()[0][:10])

ft = FeatureTransformer(d_prime=8, method="pca", output="angle", seed=seed).fit(X_tr.to_numpy(), labels[tr])
z_std = ft.scaler.transform(x.to_numpy())
z_pca = ft.pca.transform(z_std)
z = ft.transform(x.to_numpy())
print(f"\n== 3. Standardised (95 values, mean 0 / sd 1 on inner-train): first 10: {z_std[0][:10]}")
print(f"   PCA to 8 components (explain {ft.pca.explained_variance_ratio_.sum():.1%} of variance): {z_pca[0]}")
print(f"   scaled to rotation angles in [0, pi]:                  {z[0]}")

enc = make_encoding("angle", 8)
model_probe = QuantumKernelSVC(encoding="angle", n_qubits=8)
state = model_probe._states(z)[0]
top = np.argsort(-np.abs(state))[:5]
print(f"\n== 4. Angle encoding: qubit i rotated by RY(angle_i) -> 8-qubit state with {state.size} amplitudes "
      f"(norm {np.linalg.norm(state):.6f})")
for k in top:
    print(f"   |{k:08b}>  amplitude {state[k].real:+.4f}  probability {abs(state[k])**2:.4f}")

X_iv = ft.transform(prep.transform(frame.iloc[iv], "label").to_numpy())
best = select_and_fit(lambda C: QuantumKernelSVC(encoding="angle", n_qubits=8, C=C, seed=seed),
                      {"C": [0.1, 1.0, 10.0]}, ft.transform(X_tr.to_numpy()), labels[tr], X_iv, labels[iv])
m = best["model"]
k_row = m._kernel(model_probe._states(z), m.train_states_)[0]
print(f"\n== 5. QSVM (C={best['params']['C']} chosen on inner-val): kernel = |<phi(x)|phi(x_j)>|^2 against "
      f"{len(m.train_states_)} training states")
print(f"   kernel values vs training rows: min {k_row.min():.3f}, median {np.median(k_row):.3f}, max {k_row.max():.3f}; "
      f"{len(m.svc_.support_)} support vectors")
score = float(m.decision_scores(z)[0])
print(f"\n== 6. Decision score {score:+.3f}; threshold {best['threshold']:+.3f} (Youden J on inner-val) -> "
      f"predicted {'AT RISK (1)' if score >= best['threshold'] else 'not at risk (0)'}; true label {labels[row]}")
