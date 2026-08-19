#!/usr/bin/env python3
"""Draw the figures from the CSVs the other analysis scripts wrote.

Deliberately the only file in the repo that needs matplotlib. Every number these
figures show is already in results/*.csv, produced by scripts that need numpy alone,
so a machine without matplotlib still reproduces every result -- it just does not
draw it.

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


def plot_convergence() -> None:
    """The credibility anchor: log-log error against step size, with fitted slopes."""
    rows = [r for r in read("convergence.csv")
            if r["mode"] == "ode_trajectory" and r["prior"] == "canonical"
            and r["grid"] == "uniform_logsnr"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in sorted({r["sampler"] for r in rows}):
        sub = sorted((r for r in rows if r["sampler"] == name),
                     key=lambda r: float(r["h_max"]))
        ax.loglog([float(r["h_max"]) for r in sub], [float(r["error"]) for r in sub],
                  marker="o", label=f"{name} (slope {sub[0]['fitted_slope']})")
    for p, style in ((1, ":"), (2, "--")):
        h = [1e-2, 1.0]
        ax.loglog(h, [1e-2 * (x / h[1]) ** p for x in h], style, c="k", lw=0.8,
                  label=f"$h^{p}$ reference")
    ax.set_xlabel(r"step size $h$ (in log-SNR $\lambda$)")
    ax.set_ylabel("RMS trajectory error against the exact quantile map")
    ax.set_title("Empirical order of convergence, probability-flow ODE")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "convergence.png", dpi=150)


def plot_nfe_frontier() -> None:
    rows = read("nfe_quality.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for key, ax, title in (("w1", axes[0], r"Wasserstein-1 vs. exact $p_{t_\epsilon}$"),
                           ("traj_rmse", axes[1], "trajectory RMSE (discretization only)")):
        for name in sorted({r["sampler"] for r in rows if r["sampler"] != "exact_map"}):
            sub = sorted((r for r in rows if r["sampler"] == name and r[key] != ""),
                         key=lambda r: int(r["nfe"]))
            if not sub:
                continue
            ax.loglog([int(r["nfe"]) for r in sub], [float(r[key]) for r in sub],
                      marker="o", ms=3, label=name)
        ax.set_xlabel("NFE (network evaluations)")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    floor = [r for r in rows if r["sampler"] == "exact_map"][0]
    axes[0].axhline(float(floor["w1"]), ls="--", c="k", lw=0.8)
    axes[0].annotate("exact-map floor", (10, float(floor["w1"]) * 1.3), fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "nfe_frontier.png", dpi=150)


def plot_sde_vs_ode() -> None:
    rows = read("sde_vs_ode.csv")
    div = [r for r in rows if r["section"] == "diversity"
           and r["sampler"] == "euler_maruyama" and r["prob_level"] == "0.5"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([float(r["conditioning_time"]) for r in div],
            [float(r["mode_entropy_bits"]) for r in div], marker="o",
            label="reverse SDE")
    ax.axhline(0.0, ls="--", c="k", lw=0.8)
    ax.annotate("probability-flow ODE: 0 bits at every $t_c$", (0.3, 0.05), fontsize=8)
    ax.set_xlabel(r"conditioning time $t_c$")
    ax.set_ylabel("conditional mode entropy (bits)")
    ax.set_title("How much of the sample the initial state decides")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "sde_vs_ode.png", dpi=150)


def plot_stability() -> None:
    rows = [r for r in read("stability.csv") if r["section"] == "sharpness"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    v = [float(r["prior_variance"]) for r in rows if float(r["prior_variance"]) > 0]
    for key, label in (("err_euler", "Euler"), ("err_heun", "Heun"),
                       ("err_dpm1", "DPM-Solver-1"), ("err_dpm2", "DPM-Solver-2")):
        y = [float(r[key]) for r in rows if float(r["prior_variance"]) > 0]
        ax.loglog(v, y, marker="o", label=label)
    ax.invert_xaxis()
    ax.set_xlabel("prior variance $v$ (sharper $\\rightarrow$)")
    ax.set_ylabel("error at 16 steps against the exact solution")
    ax.set_title("The exponential integrator wins as the data approaches a point mass")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "stability.png", dpi=150)


if __name__ == "__main__":
    plot_convergence()
    plot_nfe_frontier()
    plot_sde_vs_ode()
    plot_stability()
    print(f"wrote figures to {RESULTS}")
