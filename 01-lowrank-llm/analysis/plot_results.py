#!/usr/bin/env python3
"""Draw the figures from the CSVs the other analysis scripts wrote.

Deliberately the only file in the repo that needs matplotlib. Every number is produced
and committed by pareto.py / spectra.py / allocation_compare.py regardless of whether
anything can be plotted, so a machine without matplotlib still reproduces every result
-- it just does not draw it.

    python3 -m venv .venv && .venv/bin/pip install matplotlib
    .venv/bin/python analysis/plot_results.py
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

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
        sys.exit(f"{path} missing -- run the matching analysis script first.")
    with path.open() as fh:
        return list(csv.DictReader(fh))


def plot_pareto() -> None:
    rows = read("pareto.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, regime in zip(axes, ("anisotropic", "isotropic")):
        sub = [r for r in rows if r["regime"] == regime]
        series = [("plain SVD", "")] + [("whitened SVD", r)
                                        for r in sorted({s["ridge_ratio"] for s in sub
                                                         if s["ridge_ratio"]},
                                                        key=float)]
        for method, ridge in series:
            pts = sorted((r for r in sub if r["method"] == method
                          and r["ridge_ratio"] == ridge),
                         key=lambda r: float(r["compression_ratio"]))
            label = method if not ridge else f"whitened, ridge {float(ridge):g}"
            ax.plot([float(r["compression_ratio"]) for r in pts],
                    [float(r["rel_error_holdout_mean"]) for r in pts],
                    marker="o", label=label)
        ax.set_xscale("log")
        ax.set_xlabel("compression ratio (dense / factored)")
        ax.set_title(f"{regime} activations")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"held-out $\|(W-\hat W)X\|_F / \|WX\|_F$")
    axes[0].legend(fontsize=8)
    fig.suptitle("Whitening wins only where the activations are anisotropic")
    fig.tight_layout()
    fig.savefig(RESULTS / "pareto.png", dpi=150)


def plot_spectra() -> None:
    rows = read("spectra.csv")
    layers = sorted({r["layer"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, name in enumerate(layers):
        sub = sorted((r for r in rows if r["layer"] == name),
                     key=lambda r: int(r["index"]))
        colour = f"C{i}"
        ax.semilogy([int(r["index"]) for r in sub],
                    [float(r["sigma_w_relative"]) for r in sub],
                    color=colour, ls="--", lw=1.0,
                    label=f"{name}  W" if i == 0 else None)
        ax.semilogy([int(r["index"]) for r in sub],
                    [float(r["sigma_ws_relative"]) for r in sub],
                    color=colour, ls="-", lw=1.4,
                    label=f"{name}  WS" if i == 0 else None)
    ax.set_xlabel("singular value index")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.set_title("Weight spectra before (dashed) and after (solid) whitening")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "spectra.png", dpi=150)


def plot_allocation() -> None:
    rows = read("allocation.csv")
    strategies = ["uniform", "greedy", "lagrangian", "knapsack_dp"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in strategies:
        pts = sorted((r for r in rows if r["strategy"] == name),
                     key=lambda r: float(r["budget_fraction"]))
        ax.plot([float(r["budget_fraction"]) for r in pts],
                [float(r["gap_vs_optimum_pct"]) for r in pts],
                marker="o", label=name)
    ax.set_xlabel("parameter budget, as a fraction of dense")
    ax.set_ylabel("excess loss over the knapsack optimum (%)")
    ax.set_title("Uniform rank allocation is the only one that is badly wrong")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "allocation.png", dpi=150)


def plot_crossover() -> None:
    """The crossover, one panel per activation spectrum.

    Both arms on the same x-axis (achieved compression), which is the whole point --
    quantization's rungs are discrete and low-rank is handed exactly those rungs.
    """
    rows = read("composition.csv")
    spectra = list(dict.fromkeys(r["spectrum"] for r in rows))
    fig, axes = plt.subplots(1, len(spectra), figsize=(4 * len(spectra), 4),
                             sharey=True)
    for ax, name in zip(np.atleast_1d(axes), spectra):
        sub = sorted((r for r in rows if r["spectrum"] == name),
                     key=lambda r: float(r["achieved_compression"]))
        x = [float(r["achieved_compression"]) for r in sub]
        q = [float(r["quantize_only_error"]) for r in sub]
        l = [float(r["best_lowrank_family_error"]) for r in sub]
        ax.plot(x, q, marker="o", label="quantize only")
        ax.plot(x, l, marker="s", label="best low-rank family")
        # Shade the bracket the crossover falls in, rather than drawing a false
        # single crossing point: the measurement only resolves it to one rung.
        flip = next((i for i in range(len(sub) - 1)
                     if sub[i]["winner"] == "quantize"
                     and sub[i + 1]["winner"] == "lowrank"), None)
        if flip is not None:
            ax.axvspan(x[flip], x[flip + 1], color="0.85", zorder=0)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("achieved compression")
        ax.set_title(name, fontsize=9)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel(r"relative $\|(W-\hat W)X\|_F / \|WX\|_F$")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "crossover.png", dpi=150)


if __name__ == "__main__":
    plot_pareto()
    plot_spectra()
    plot_allocation()
    plot_crossover()
    print(f"wrote figures to {RESULTS}")
