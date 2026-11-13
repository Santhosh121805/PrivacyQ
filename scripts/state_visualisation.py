"""Box 6 state visualisation: Bloch vectors of real encoded records (angle vs ZZ, 8 qubits, PCA-8),
and the full statevector of one real training record with its 8 rotation angles.

Rows: the inner-train rows (1,200) of Experiment 2's first fold (repeat 0, fold 0, 1,500-row
subsample), preprocessed and PCA-8 angle-scaled exactly as in Experiment 2 (fitted on those rows
only). The hold-out and validation folds are not used. The single record is the first inner-train row
(position 0, no selection). Writes results/state_visualisation/: bloch_angle_vs_zz.png (colour),
bloch_angle_vs_zz_bw.png (classes by marker; prints in black and white), bloch_summary.csv/.md,
state_vector_record_input.csv (the record's 8 angles), state_vector_record.csv (256 amplitudes per
encoding) and state_vector_record.png.

Usage: python scripts/state_visualisation.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.m2_preprocess.preprocessing import TabularPreprocessor  # noqa: E402
from src.m8_eval.state_viz import bloch_vectors, class_separation, statevector  # noqa: E402
from src.m10_exp.common import cv_folds, holdout_split, load_config, load_dataset  # noqa: E402
from src.m10_exp.exp2_compare import feature_matrices  # noqa: E402
from src.m10_exp.tuning import inner_split, stratified_subsample  # noqa: E402

INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d4"
COLORS = {0: "#2a78d6", 1: "#eb6834"}            # categorical slots 1-2, validated (CVD dE 24.7)
NAMES = {0: "No current depression", 1: "Current depression"}


def fold_rows():
    config = load_config(ROOT / "configs/exp2_nsch2024.yaml")
    seed = int(config["project"]["seed"])
    loader, frame = load_dataset(config)
    labels = frame[loader.label_column].to_numpy(dtype=int)
    dev, _ = holdout_split(labels, float(config["split"]["holdout_fraction"]), seed)
    f = cv_folds(dev, labels, int(config["cv"]["n_splits"]), int(config["cv"]["n_repeats"]), seed)[0]
    s = seed + 1000 * f["repeat"] + f["fold"]
    rows = stratified_subsample(f["train"], labels, int(config["subsample"]["n"]), s)
    tr, _ = inner_split(rows, labels, float(config["tuning"]["inner_val_fraction"]), s)
    prep = TabularPreprocessor(identifier_columns=loader.metadata_columns)
    X = prep.fit_transform(frame.iloc[tr], loader.label_column).to_numpy()
    (A,) = feature_matrices({"method": "pca", "k": 8}, X, labels[tr], [], s)
    return A, labels[tr], tr


def sphere(ax):
    u, v = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20)
    ax.plot_wireframe(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
                      np.outer(np.ones_like(u), np.cos(v)), color=GRID, linewidth=0.4)
    for axis in ("x", "y", "z"):
        getattr(ax, f"set_{axis}lim")(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.text(0, 0, 1.18, "|0⟩", color=INK2, ha="center", fontsize=8)
    ax.text(0, 0, -1.3, "|1⟩", color=INK2, ha="center", fontsize=8)
    ax.view_init(elev=8, azim=-90)                 # look along -Y: the X-Z plane faces the reader
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1, 1, 1, 0)); pane.line.set_color((1, 1, 1, 0))
    ax.text(1.25, 0, 0, "+X", color=INK2, ha="center", fontsize=7)


def main():
    out = ROOT / "results/state_visualisation"
    out.mkdir(parents=True, exist_ok=True)
    X, y, rows = fold_rows()
    vecs = {enc: bloch_vectors(enc, 8, X) for enc in ("angle", "zz")}
    plt.rcParams.update({"text.color": INK, "axes.labelcolor": INK2, "font.size": 9})
    fig = plt.figure(figsize=(12, 6.4), facecolor=SURFACE)
    for row, enc in enumerate(("angle", "zz")):
        for q in range(4):
            ax = fig.add_subplot(2, 4, row * 4 + q + 1, projection="3d", facecolor=SURFACE)
            sphere(ax)
            for cls in (0, 1):                                   # minority class drawn on top
                v = vecs[enc][y == cls, q]
                ax.scatter(v[:, 0], v[:, 1], v[:, 2], s=6, color=COLORS[cls], alpha=0.35 if cls == 0 else 0.8,
                           edgecolors="none", label=NAMES[cls])
            ax.set_title(f"{'Angle' if enc == 'angle' else 'ZZ'} · qubit {q + 1} (PC{q + 1})", color=INK, fontsize=9)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=COLORS[c], label=NAMES[c]) for c in (0, 1)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False)
    fig.suptitle("Single-qubit states of 1,200 training records (Exp 2 fold 1), qubits 1–4 of 8", color=INK, y=0.99)
    fig.text(0.5, 0.02, "Angle: pure product states on the X–Z circle (|r| = 1). ZZ: entangled, so each qubit's "
             "reduced state lies inside the sphere (|r| < 1).", ha="center", color=INK2, fontsize=8)
    fig.savefig(out / "bloch_angle_vs_zz.png", dpi=160, bbox_inches="tight", facecolor=SURFACE)

    # black-and-white version: classes by marker shape, not colour
    fig = plt.figure(figsize=(12, 6.4), facecolor="white")
    for row, enc in enumerate(("angle", "zz")):
        for q in range(4):
            ax = fig.add_subplot(2, 4, row * 4 + q + 1, projection="3d", facecolor="white")
            sphere(ax)
            v0, v1 = vecs[enc][y == 0, q], vecs[enc][y == 1, q]
            ax.scatter(v0[:, 0], v0[:, 1], v0[:, 2], s=5, marker="o", color="#9a9a9a", alpha=0.35, edgecolors="none")
            ax.scatter(v1[:, 0], v1[:, 1], v1[:, 2], s=14, marker="x", color="black", linewidths=0.8)
            ax.set_title(f"{'Angle' if enc == 'angle' else 'ZZ'} · qubit {q + 1} (PC{q + 1})", color="black", fontsize=9)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color="#9a9a9a", label=NAMES[0]),
               plt.Line2D([], [], marker="x", linestyle="", color="black", label=NAMES[1])]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False)
    fig.savefig(out / "bloch_angle_vs_zz_bw.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close("all")

    # one real training record: its 8 rotation angles and full statevector under angle and ZZ encoding
    x0, row0 = X[0], int(rows[0])
    pd.DataFrame({"dataset_row": row0, "rows_from": "Experiment 2 repeat 0 fold 0, inner-train position 0",
                  "qubit": np.arange(1, 9), "angle_rad": x0}).to_csv(out / "state_vector_record_input.csv", index=False)
    amp = []
    for enc in ("angle", "zz"):
        psi = statevector(enc, 8, x0)
        amp += [{"encoding": enc, "basis_index": i, "basis_state": format(i, "08b"), "amplitude_real": a.real,
                 "amplitude_imag": a.imag, "probability": abs(a) ** 2} for i, a in enumerate(psi)]
    amp = pd.DataFrame(amp)
    amp.to_csv(out / "state_vector_record.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, enc, style in zip(axes, ("angle", "zz"), ({"color": "white", "hatch": "////"}, {"color": "#bdbdbd", "hatch": None})):
        top = amp[amp.encoding == enc].nlargest(16, "probability")
        ax.bar(range(16), top.probability, edgecolor="black", lw=0.7, **style)
        ax.set_xticks(range(16), top.basis_state, rotation=90, fontfamily="monospace", fontsize=7)
        ax.set_title(f"{'Angle' if enc == 'angle' else 'ZZ'} encoding: 16 most probable of 256 basis states", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Probability |amplitude|²")
    fig.tight_layout()
    fig.savefig(out / "state_vector_record.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    rows = []
    for enc, v in vecs.items():
        length, sep = np.linalg.norm(v, axis=-1), class_separation(v, y)
        for q in range(8):
            rows.append({"encoding": enc, "qubit": q + 1, "mean_bloch_length": length[:, q].mean(),
                         "min_bloch_length": length[:, q].min(), "class_mean_distance": sep[q]})
    table = pd.DataFrame(rows)
    table.to_csv(out / "bloch_summary.csv", index=False)
    lines = ["# State visualisation (box 6): Bloch vectors of encoded records", "",
             f"Rows: {len(y):,} inner-train rows of Experiment 2 fold 1 (repeat 0, fold 0), {int(y.sum())} with current "
             "depression; PCA-8, angle-scaled; fitted on these rows only. Exact statevectors (`default.qubit`).", "",
             "Bloch length 1 = the qubit is unentangled; < 1 = entangled with the others. Class-mean distance = "
             "distance between the mean Bloch vectors of the two classes (0 to 2). Descriptive only.", "",
             "| Encoding | Qubit | Mean Bloch length | Min Bloch length | Class-mean distance |", "|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r.encoding} | {r.qubit} | {r.mean_bloch_length:.3f} | {r.min_bloch_length:.3f} | {r.class_mean_distance:.3f} |")
    lines += ["", f"## One record: dataset row {row0} (first inner-train row of the same fold)", "",
              "Rotation angles (rad): " + ", ".join(f"{a:.4f}" for a in x0), ""]
    for enc in ("angle", "zz"):
        a = amp[amp.encoding == enc]
        top = a.nlargest(4, "probability")
        lines.append(f"- {enc}: norm {a.probability.sum():.6f}; most probable basis states "
                     + ", ".join(f"|{r.basis_state}⟩ {r.probability:.4f}" for _, r in top.iterrows()))
    (out / "bloch_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
