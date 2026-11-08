"""Shared experiment plumbing: configs, run directories, provenance, splits."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from src.m1_data import DatasetLoader, make_loader

PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "torch", "PennyLane",
    "PennyLane_Lightning", "opacus", "matplotlib", "PyYAML",
)

SYNTHETIC_BANNER = (
    "SYNTHETIC DATA - these numbers come from a generator and say nothing about "
    "real adolescents. Do not report them as findings."
)


LABEL_LIMITATION = (
    "Limitation: the NSCH target is caregiver-reported, provider-diagnosed depression. "
    "Much depression is never diagnosed (~63.6% of cases per the WHO estimate cited in "
    "the literature survey), so undiagnosed adolescents sit in the negative class. Models "
    "identify adolescents resembling those already diagnosed, which under-represents groups "
    "with poorer access to care. The survey is cross-sectional and some predictors may be "
    "consequences rather than causes: results identify who is at risk now; they do not "
    "forecast who will become depressed."
)


def report_header(title: str, loader) -> list[str]:
    """Standard header for every results markdown file."""
    lines = [f"# {title} — {loader.source_tag()}", ""]
    if loader.is_synthetic:
        lines += [f"> **{SYNTHETIC_BANNER}**", ""]
    elif loader.name.startswith("NSCH"):
        lines += [f"> {LABEL_LIMITATION}", ""]
    return lines


def limit_worker_threads() -> None:
    """One BLAS/OpenMP thread per worker process. Must run before workers spawn.

    On the 12-core (4P+8E) development laptop, 13 workers with default thread
    pools ran ~8x slower per task than a single process; 6 single-threaded
    workers gave the best throughput.
    """
    import os

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def make_run_dir(config: dict, loader: DatasetLoader) -> Path:
    project = config.get("project", {})
    root = Path(project.get("output_dir", "results"))
    prefix = "SYNTHETIC-" if loader.is_synthetic else ""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{prefix}{project.get('name', 'run')}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    if loader.is_synthetic:
        (run_dir / "SYNTHETIC_DATA_WARNING.txt").write_text(SYNTHETIC_BANNER + "\n", encoding="utf-8")
    return run_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_record(config: dict) -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    data_files = {}
    for path in config.get("data", {}).get("paths", []) or [config.get("data", {}).get("path")]:
        if path and Path(path).exists():
            data_files[str(path)] = _sha256(Path(path))
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": commit,
        "git_dirty": dirty,
        "data_file_sha256": data_files,
        "started": datetime.now().isoformat(timespec="seconds"),
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def load_dataset(config: dict) -> tuple[DatasetLoader, pd.DataFrame]:
    seed = int(config.get("project", {}).get("seed", 0))
    loader = make_loader(config["data"], seed=seed)
    return loader, loader.load()


def holdout_split(labels: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified (dev, holdout) row positions. The holdout is locked: it is
    neither used for model selection nor touched until the final evaluation."""
    positions = np.arange(len(labels))
    dev, holdout = train_test_split(
        positions, test_size=fraction, stratify=labels, random_state=seed
    )
    return np.sort(dev), np.sort(holdout)


def cv_folds(
    dev_positions: np.ndarray, labels: np.ndarray, n_splits: int, n_repeats: int, seed: int
) -> list[dict]:
    """Repeated stratified K-fold over the dev rows. Returns row positions into
    the full frame so every experiment can reuse the exact same folds."""
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    folds = []
    for index, (tr, va) in enumerate(splitter.split(dev_positions, labels[dev_positions])):
        folds.append(
            {
                "repeat": index // n_splits,
                "fold": index % n_splits,
                "train": dev_positions[tr],
                "val": dev_positions[va],
            }
        )
    return folds


def save_splits(run_dir: Path, holdout: np.ndarray, folds: list[dict]) -> None:
    pd.DataFrame({"row": holdout}).to_csv(run_dir / "holdout_rows.csv", index=False)
    rows = [
        {"repeat": f["repeat"], "fold": f["fold"], "row": int(r)}
        for f in folds for r in f["val"]
    ]
    pd.DataFrame(rows).to_csv(run_dir / "cv_val_rows.csv", index=False)


def summarise(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys, sort=False)[metrics]
    mean, std = grouped.mean(), grouped.std(ddof=1)
    out = mean.add_suffix("_mean").join(std.add_suffix("_std"))
    out["n_folds"] = grouped.size()
    return out.reset_index()


def markdown_table(summary: pd.DataFrame, keys: list[str], metrics: list[str]) -> str:
    header = "| " + " | ".join(keys + metrics) + " |"
    rule = "|" + "|".join(["---"] * (len(keys) + len(metrics))) + "|"
    lines = [header, rule]
    for _, row in summary.iterrows():
        cells = [str(row[k]) for k in keys]
        cells += [f"{row[m + '_mean']:.3f} ± {row[m + '_std']:.3f}" for m in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
