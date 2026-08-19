#!/usr/bin/env python3
"""Draw the figures from the CSVs the other analysis scripts wrote.

Deliberately the only file in the repo that needs matplotlib. The numbers are produced
and committed by bits_vs_error.py / damping_sweep.py / error_bound.py regardless of
whether anything can be plotted, so a machine without matplotlib still reproduces
every result -- it just does not draw them.

    python3 -m venv .venv && .venv/bin/pip install matplotlib
    .venv/bin/python analysis/plot_results.py
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
        sys.exit(f"{path} missing -- run the matching analysis script first.")
    with path.open() as fh:
        return list(csv.DictReader(fh))


def plot_bits_vs_error() -> None:
    rows = read("bits_vs_error.csv")
    methods = sorted({r["method"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in methods:
        sub = sorted((r for r in rows if r["method"] == m),
                     key=lambda r: int(r["nominal_bits"]))
        ax.plot([int(r["nominal_bits"]) for r in sub],
                [float(r["relative_error_mean"]) for r in sub],
                marker="o", label=m)
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("bits per weight")
    ax.set_ylabel(r"relative $\|(W-\hat W)X\|_F / \|WX\|_F$")
    ax.set_title("Activation-weighted error vs. bit-width")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "bits_vs_error.png", dpi=150)


def plot_damping() -> None:
    rows = [r for r in read("damping_sweep.csv") if r["output_error_mean"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([float(r["damping_ratio"]) for r in rows],
            [float(r["improvement_over_rtn"]) for r in rows], marker="o")
    ax.axhline(1.0, ls="--", c="k", lw=0.8)
    ax.annotate("no better than RTN", (1e-7, 1.05), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"damping ratio $\lambda$ / mean diag $H$")
    ax.set_ylabel("improvement over RTN")
    ax.set_title("Too little damping fails to factor; too much becomes RTN")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "damping_sweep.png", dpi=150)


def plot_propagation() -> None:
    rows = [r for r in read("error_propagation.csv")
            if r["bits"] == "3" and r["method"] == "sequential"]
    layers = [int(r["layer"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(layers, [float(r["predicted"]) for r in rows], marker="s",
            label="predicted (bound)")
    ax.plot(layers, [float(r["measured"]) for r in rows], marker="o",
            label="measured")
    ax.set_yscale("log")
    ax.set_xlabel("layer")
    ax.set_ylabel(r"$\|E_\ell\|_F$")
    ax.set_title("Cross-layer error: bound vs. measurement (3-bit)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "predicted_vs_measured.png", dpi=150)


if __name__ == "__main__":
    plot_bits_vs_error()
    plot_damping()
    plot_propagation()
    print(f"wrote figures to {RESULTS}")
