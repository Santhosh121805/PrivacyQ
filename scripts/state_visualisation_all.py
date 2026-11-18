"""Quantum state visualisation of ALL 21,789 records (descriptive only; no model is trained or scored).

Every record of the NSCH 2024 file (ages 12-17) is preprocessed, reduced to 8 PCA components and scaled to rotation
angles in [0, pi], then encoded on 8 qubits with the angle and ZZ feature maps. Imputation, one-hot encoding,
standardisation, PCA and the angle scaling are fitted on the 17,431 development rows only (the locked hold-out rows
are transformed with those fitted statistics, never used to fit them). States are exact (PennyLane default.qubit).

Writes results/state_visualisation_all/:
  bloch_all_angle.png, bloch_all_zz.png     8 Bloch spheres each (one per qubit), every record, class by marker
  bloch_length_all.png                      |r| per qubit (1 = unentangled, < 1 = entangled), ZZ vs angle
  angles_all.png                            rotation-angle distribution per qubit, by class
  statevector_mean_all.png                  mean probability of the 256 basis states, by class, both encodings
  class_separation_all.png                  distance between the two classes' mean Bloch vectors, per qubit
  summary.csv, summary.md                   every number shown in the figures

Usage: python scripts/state_visualisation_all.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pennylane as qml  # noqa: E402

from src.m2_preprocess.preprocessing import TabularPreprocessor  # noqa: E402
from src.m3_features.features import FeatureTransformer  # noqa: E402
from src.m4_encoding import make_encoding  # noqa: E402
from src.m8_eval.state_viz import bloch_vectors, class_separation  # noqa: E402
from src.m10_exp.common import holdout_split, load_config, load_dataset  # noqa: E402

OUT = ROOT / "results/state_visualisation_all"
NAMES = {0: "No current depression", 1: "Current depression"}
ENC = {"angle": "Angle encoding", "zz": "ZZ feature map"}
NQ = 8


def encoded_angles():
    cfg = load_config(ROOT / "configs/exp2_nsch2024.yaml")
    seed = int(cfg["project"]["seed"])
    loader, frame = load_dataset(cfg)
    y = frame[loader.label_column].to_numpy(dtype=int)
    dev, hold = holdout_split(y, float(cfg["split"]["holdout_fraction"]), seed)
    prep = TabularPreprocessor(identifier_columns=loader.metadata_columns)
    X_dev = prep.fit_transform(frame.iloc[dev], loader.label_column).to_numpy()
    X_all = prep.transform(frame, loader.label_column).to_numpy()
    ft = FeatureTransformer(d_prime=NQ, method="pca", output="angle", seed=seed).fit(X_dev, y[dev])
    return ft.transform(X_all), y, len(dev), len(hold), float(ft.pca.explained_variance_ratio_.sum())


def statevectors(encoding: str, X: np.ndarray, chunk: int = 512) -> np.ndarray:
    enc = make_encoding(encoding, NQ)

    @qml.qnode(qml.device("default.qubit", wires=NQ))
    def state(x):
        enc.feature_map(x)
        return qml.state()

    x = np.asarray(enc.prepare(X), dtype=float)
    return np.concatenate([np.asarray(state(x[i:i + chunk])).reshape(-1, 2 ** NQ) for i in range(0, len(x), chunk)])


def sphere(ax):
    u, v = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20)
    ax.plot_wireframe(np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)),
                      np.outer(np.ones_like(u), np.cos(v)), color="#d9d8d4", linewidth=0.4)
    for axis in ("x", "y", "z"):
        getattr(ax, f"set_{axis}lim")(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.text(0, 0, 1.18, "|0⟩", ha="center", fontsize=8, color="#52514e")
    ax.text(0, 0, -1.32, "|1⟩", ha="center", fontsize=8, color="#52514e")
    ax.text(1.28, 0, 0, "+X", ha="center", fontsize=7, color="#52514e")
    ax.view_init(elev=8, azim=-90)                 # look along -Y: the X-Z plane faces the reader
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1, 1, 1, 0)); pane.line.set_color((1, 1, 1, 0))


def bloch_figure(vec, y, enc, n, path):
    fig = plt.figure(figsize=(13, 7.2), facecolor="white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.06, wspace=0.02, hspace=0.08)
    for q in range(NQ):
        ax = fig.add_subplot(2, 4, q + 1, projection="3d", facecolor="white")
        sphere(ax)
        v0, v1 = vec[y == 0, q], vec[y == 1, q]
        ax.scatter(v0[:, 0], v0[:, 1], v0[:, 2], s=1.5, marker="o", color="#9a9a9a", alpha=0.25, edgecolors="none")
        ax.scatter(v1[:, 0], v1[:, 1], v1[:, 2], s=6, marker="x", color="black", linewidths=0.5, alpha=0.6)
        for cls, ls in ((0, "-"), (1, "--")):                      # class-mean Bloch vectors
            m = vec[y == cls, q].mean(0)
            ax.plot([0, m[0]], [0, m[1]], [0, m[2]], color="black", lw=2, ls=ls)
        ax.set_title(f"Qubit {q + 1} (PC{q + 1})", fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", color="#9a9a9a", label=f"{NAMES[0]} ({(y == 0).sum():,})"),
               plt.Line2D([], [], marker="x", ls="", color="black", label=f"{NAMES[1]} ({(y == 1).sum():,})"),
               plt.Line2D([], [], color="black", lw=2, label="class mean vector: no depression"),
               plt.Line2D([], [], color="black", lw=2, ls="--", label="class mean vector: depression")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"{ENC[enc]}: single-qubit Bloch vectors of all {n:,} records, 8 qubits", fontsize=13, y=0.995)
    note = ("Angle encoding gives unentangled product states: every point lies on the sphere's surface (|r| = 1)."
            if enc == "angle" else
            "The ZZ feature map entangles the qubits, so each qubit's reduced state lies inside the sphere (|r| < 1).")
    fig.text(0.5, 0.015, note, ha="center", fontsize=9, color="#33363c")
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, n_dev, n_hold, evr = encoded_angles()
    n = len(y)
    print(f"{n:,} records encoded ({y.sum():,} with current depression); fitted on {n_dev:,} development rows; "
          f"PCA-8 keeps {evr:.1%} of variance  ({time.perf_counter() - t0:.0f} s)", flush=True)
    vec = {}
    for enc in ("angle", "zz"):
        vec[enc] = bloch_vectors(enc, NQ, X)
        print(f"Bloch vectors, {enc}: {vec[enc].shape}  ({time.perf_counter() - t0:.0f} s)", flush=True)
        bloch_figure(vec[enc], y, enc, n, OUT / f"bloch_all_{enc}.png")
    probs = {}
    for enc in ("angle", "zz"):
        psi = statevectors(enc, X)
        probs[enc] = np.abs(psi) ** 2
        print(f"statevectors, {enc}: {psi.shape}, max |norm - 1| = {np.abs(probs[enc].sum(1) - 1).max():.2e}  "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)

    # Bloch length per qubit
    length = {e: np.linalg.norm(v, axis=-1) for e, v in vec.items()}
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True)
    bins = np.linspace(0, 1, 41)
    for q, ax in enumerate(axes.ravel()):
        for cls, style in ((0, {"color": "#9a9a9a", "alpha": 0.8}), (1, {"histtype": "step", "color": "black", "lw": 1.4})):
            ax.hist(length["zz"][y == cls, q], bins=bins, density=True, **style)
        ax.axvline(1.0, color="black", ls=":", lw=1)
        ax.set_title(f"Qubit {q + 1}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Bloch vector length |r|")
    for ax in axes[:, 0]:
        ax.set_ylabel("Density")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#9a9a9a", label=f"ZZ, {NAMES[0]}"),
                        plt.Line2D([], [], color="black", lw=1.4, label=f"ZZ, {NAMES[1]}"),
                        plt.Line2D([], [], color="black", ls=":", label="angle encoding: every record at |r| = 1")],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"How entangled each qubit is: Bloch length of all {n:,} records under the ZZ feature map", y=1.07)
    fig.tight_layout()
    fig.savefig(OUT / "bloch_length_all.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # rotation angles per qubit
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True)
    bins = np.linspace(0, np.pi, 41)
    for q, ax in enumerate(axes.ravel()):
        ax.hist(X[y == 0, q], bins=bins, density=True, color="#9a9a9a", alpha=0.8)
        ax.hist(X[y == 1, q], bins=bins, density=True, histtype="step", color="black", lw=1.4)
        ax.set_title(f"Qubit {q + 1} (PC{q + 1})", fontsize=10)
        ax.set_xticks([0, np.pi / 2, np.pi], ["0", "π/2", "π"])
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Rotation angle θ (rad)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Density")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#9a9a9a", label=NAMES[0]),
                        plt.Line2D([], [], color="black", lw=1.4, label=NAMES[1])],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Rotation angles fed to the encoder: all {n:,} records, by class", y=1.07)
    fig.tight_layout()
    fig.savefig(OUT / "angles_all.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # mean basis-state probabilities
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True)
    idx = np.arange(2 ** NQ)
    for ax, enc in zip(axes, ("angle", "zz")):
        ax.fill_between(idx, probs[enc][y == 0].mean(0), step="mid", color="#bdbdbd", label=NAMES[0])
        ax.step(idx, probs[enc][y == 1].mean(0), where="mid", color="black", lw=1.1, label=NAMES[1])
        ax.set_ylabel("Mean probability")
        ax.set_title(f"{ENC[enc]}: mean probability of each of the 256 basis states, all {n:,} records", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    axes[1].set_xticks(range(0, 256, 32), [format(i, "08b") for i in range(0, 256, 32)], fontfamily="monospace", fontsize=8)
    axes[1].set_xlabel("Basis state |q1 … q8⟩")
    fig.tight_layout()
    fig.savefig(OUT / "statevector_mean_all.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # class separation per qubit
    sep = {e: class_separation(v, y) for e, v in vec.items()}
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.38
    ax.bar(np.arange(NQ) - w / 2, sep["angle"], w, color="white", edgecolor="black", hatch="////", label=ENC["angle"])
    ax.bar(np.arange(NQ) + w / 2, sep["zz"], w, color="#9a9a9a", edgecolor="black", label=ENC["zz"])
    for i in range(NQ):
        ax.text(i - w / 2, sep["angle"][i] + 0.01, f"{sep['angle'][i]:.3f}", ha="center", fontsize=7)
        ax.text(i + w / 2, sep["zz"][i] + 0.01, f"{sep['zz'][i]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(range(NQ), [f"Qubit {q + 1}" for q in range(NQ)])
    ax.set_ylabel("Distance between class means")
    ax.set_title(f"Class separation in Bloch space, all {n:,} records (0 = none, 2 = maximum; descriptive)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "class_separation_all.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    rows = []
    for enc in ("angle", "zz"):
        for q in range(NQ):
            rows.append({"encoding": enc, "qubit": q + 1, "n_records": n, "mean_bloch_length": length[enc][:, q].mean(),
                         "min_bloch_length": length[enc][:, q].min(), "class_mean_distance": sep[enc][q],
                         "mean_angle_no_depression": X[y == 0, q].mean(), "mean_angle_depression": X[y == 1, q].mean()})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "summary.csv", index=False)
    lines = ["# Quantum state visualisation: all records", "",
             f"Records: {n:,} (all ages 12-17 in the NSCH 2024 file), {int(y.sum()):,} with current depression. "
             f"Preprocessing, PCA-8 ({evr:.1%} of variance) and angle scaling fitted on the {n_dev:,} development rows; "
             f"the {n_hold:,} hold-out rows are transformed only. Exact states (PennyLane default.qubit). "
             "Descriptive only: no model is trained or scored.", "",
             "| Encoding | Qubit | Mean Bloch length | Min Bloch length | Class-mean distance | Mean angle (no dep.) | Mean angle (dep.) |",
             "|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r.encoding} | {r.qubit} | {r.mean_bloch_length:.3f} | {r.min_bloch_length:.3f} | "
                     f"{r.class_mean_distance:.3f} | {r.mean_angle_no_depression:.3f} | {r.mean_angle_depression:.3f} |")
    for enc in ("angle", "zz"):
        p = probs[enc].mean(0)
        top = np.argsort(-p)[:4]
        lines.append(f"\n{enc}: most probable basis states on average: "
                     + ", ".join(f"|{k:08b}⟩ {p[k]:.4f}" for k in top))
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
