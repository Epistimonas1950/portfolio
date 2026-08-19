#!/usr/bin/env python3
"""Draw the figures BRIEF.md asks for, from the CSVs the other scripts already wrote.

Deliberately the only file in this repo that imports matplotlib. Every number it draws
is produced and committed by oracle/*.py and analysis/*.py regardless of whether
anything can be plotted, so a machine without matplotlib -- such as the one this repo
was built on -- still reproduces every result. It just does not draw it.

    python3 -m venv .venv && .venv/bin/pip install matplotlib
    .venv/bin/python analysis/plot_results.py

Three figures, the three BRIEF.md names:
    roc.png                    ROC curves on one axis
    orthogonality_drift.png    || U^T U - I ||_F over the stream, with and without repair
    spectrum.png               the singular-value spectrum that justifies the rank
"""

from __future__ import annotations

import csv
import pathlib
import sys

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is not installed. It is intentionally not required to "
             "reproduce the results -- see results/*.csv, which are committed.\n"
             "  python3 -m venv .venv && .venv/bin/pip install matplotlib")


def read(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        sys.exit(f"{path} missing -- run `make results` first.")
    with path.open() as fh:
        return list(csv.DictReader(fh))


def plot_roc() -> None:
    rows = read("roc.csv")
    aucs = {(r["scenario"], r["anomaly_kind"], r["method"]): float(r["auc"])
            for r in read("auc.csv")}
    scenarios = [("unimodal", "spike"), ("multimode", "spike"), ("manymode", "spike")]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(13, 4.2), sharey=True)
    for ax, (scenario, kind) in zip(axes, scenarios):
        sub = [r for r in rows if r["scenario"] == scenario
               and r["anomaly_kind"] == kind]
        for method in dict.fromkeys(r["method"] for r in sub):
            pts = [r for r in sub if r["method"] == method]
            value = aucs.get((scenario, kind, method), float("nan"))
            ax.plot([float(r["fpr"]) for r in pts], [float(r["tpr"]) for r in pts],
                    label=f"{method} ({value:.3f})", lw=1.4)
        ax.plot([0, 1], [0, 1], ls="--", c="k", lw=0.7)
        ax.set_title(f"{scenario} / {kind}")
        ax.set_xlabel("false positive rate")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("true positive rate")
    axes[-1].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "roc.png", dpi=150)


def plot_drift() -> None:
    rows = read("orthogonality_drift.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method in dict.fromkeys(r["method"] for r in rows):
        pts = [r for r in rows if r["method"] == method]
        ax.plot([int(r["update"]) for r in pts], [float(r["drift"]) for r in pts],
                lw=1.1, ls="--" if "no reorth" in method else "-", label=method)
    ax.set_yscale("log")
    ax.set_xlabel("rank-one updates")
    ax.set_ylabel(r"$\|U^\top U - I\|_F$")
    ax.set_title("Orthogonality drift, with and without periodic repair")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "orthogonality_drift.png", dpi=150)


def plot_spectrum() -> None:
    rows = read("spectrum.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for stream in dict.fromkeys(r["stream"] for r in rows):
        pts = [r for r in rows if r["stream"] == stream]
        ax1.semilogy([int(r["index"]) for r in pts], [float(r["sigma"]) for r in pts],
                     marker="o", ms=3, label=stream)
        ax2.plot([int(r["index"]) for r in pts],
                 [float(r["cumulative_energy"]) for r in pts], marker="o", ms=3,
                 label=stream)
    ax2.axhline(0.95, ls="--", c="k", lw=0.8)
    ax2.annotate("energy threshold 0.95", (1, 0.955), fontsize=8)
    for ax, ylabel in ((ax1, r"$\sigma_i$"), (ax2, "cumulative energy")):
        ax.set_xlabel("index")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Warm-up spectrum, and the rank each criterion selects")
    fig.tight_layout()
    fig.savefig(RESULTS / "spectrum.png", dpi=150)


def plot_rangefinder() -> None:
    rows = read("rangefinder.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for q in dict.fromkeys(r["power_iters"] for r in rows):
        pts = sorted((r for r in rows if r["power_iters"] == q),
                     key=lambda r: int(r["oversampling"]))
        ax.plot([int(r["oversampling"]) for r in pts],
                [float(r["truncated_over_optimal"]) for r in pts],
                marker="o", label=f"q = {q} power iterations")
    ax.axhline(1.0, ls="--", c="k", lw=0.8)
    ax.annotate("Eckart-Young optimum", (0.2, 1.01), fontsize=8)
    ax.set_xlabel("oversampling p")
    ax.set_ylabel(r"$\|A - U_kU_k^\top A\|_F$ / optimal rank-$k$")
    ax.set_title(r"What $p$ and $q$ buy on a $\sigma_j = j^{-1}$ spectrum")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "rangefinder.png", dpi=150)


if __name__ == "__main__":
    plot_roc()
    plot_drift()
    plot_spectrum()
    plot_rangefinder()
    print(f"wrote figures to {RESULTS}")
