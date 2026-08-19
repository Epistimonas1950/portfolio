#!/usr/bin/env python3
"""Draw the figures from the CSVs that eval/*.py wrote.

Deliberately the only file in the repo that imports matplotlib. Every number is
produced and committed by the eval scripts whether or not anything can be plotted, so a
machine without matplotlib still reproduces every result -- it just does not draw it.

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
    import numpy as np
except ImportError:
    sys.exit("matplotlib is not installed. It is intentionally not required to "
             "reproduce the results -- see results/*.csv, which are committed.\n"
             "  python3 -m venv .venv && .venv/bin/pip install matplotlib\n"
             "  .venv/bin/python analysis/plot_results.py")


def read(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        sys.exit(f"{path} missing -- run `make results` first.")
    with path.open() as fh:
        return list(csv.DictReader(fh))


def plot_regret() -> None:
    rows = [r for r in read("regret.csv") if r["experiment"] == "minimax"]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for policy in ("LinUCB", "Thompson", "LinGreedy", "random"):
        sub = sorted((r for r in rows if r["policy"] == policy),
                     key=lambda r: int(r["t"]))
        if not sub:
            continue
        t = np.array([int(r["t"]) for r in sub])
        y = np.array([float(r["cum_regret"]) for r in sub])
        slope = float(sub[0]["fitted_slope"])
        ax.plot(t, y, marker="o", ms=3, label=f"{policy}  (slope {slope:.3f})")
    ref = sorted((r for r in rows if r["policy"] == "LinUCB"), key=lambda r: int(r["t"]))
    t = np.array([int(r["t"]) for r in ref])
    pref = float(ref[0]["fit_prefactor"])
    ax.plot(t, pref * np.sqrt(t), "k--", lw=0.9, label=r"$c\sqrt{T}$ reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("T")
    ax.set_ylabel("sup over instance family of cumulative regret")
    ax.set_title("Minimax regret envelope (synthetic linear-reward bandit)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "regret_loglog.png", dpi=150)


def plot_coverage() -> None:
    rows = [r for r in read("coverage.csv") if r["experiment"] in ("marginal", "shift")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for exp, ax, title in ((("marginal"), axes[0], "Exchangeable (as calibrated)"),
                           (("shift"), axes[1], "Calibrated on easy, tested on hard")):
        for arm in ("small", "mid", "large"):
            sub = sorted((r for r in rows if r["experiment"] == exp
                          and r["unit"] == arm), key=lambda r: float(r["alpha"]))
            ax.plot([float(r["target_coverage"]) for r in sub],
                    [float(r["empirical_coverage"]) for r in sub],
                    marker="o", ms=4, label=arm)
        lims = [0.78, 1.0]
        ax.plot(lims, lims, "k--", lw=0.9, label="diagonal")
        ax.set_xlim(lims)
        ax.set_ylim(0.3, 1.02)
        ax.set_xlabel(r"target $1-\alpha$")
        ax.set_ylabel("empirical coverage")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "coverage.png", dpi=150)


def plot_pareto() -> None:
    rows = [r for r in read("pareto.csv") if r["block"] == "policy"]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for r in rows:
        x, y = float(r["mean_cost"]), float(r["task_success"])
        oracle = r["policy"].startswith("oracle")
        ax.scatter(x, y, s=64 if oracle else 40,
                   marker="*" if oracle else "o", zorder=3)
        ax.annotate(r["policy"], (x, y), textcoords="offset points",
                    xytext=(5, 4), fontsize=7)
    ax.set_xlabel("mean cost per query  (simulated GPU-seconds)")
    ax.set_ylabel("task success")
    ax.set_title("Cost-quality plane, simulated fleet")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "pareto.png", dpi=150)


def plot_budget() -> None:
    rows = [r for r in read("pareto.csv") if r["block"] == "budget_trace"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    t = np.array([int(r["t"]) for r in rows])
    ax.plot(t, [float(r["spend"]) for r in rows], label="realized spend")
    ax.plot(t, [float(r["budget_line"]) for r in rows], "k--", lw=0.9,
            label=r"budget line $(B/T)\,t$")
    ax2 = ax.twinx()
    ax2.plot(t, [float(r["dual_price"]) for r in rows], color="tab:red", lw=0.8,
             alpha=0.7, label=r"dual price $p_t$")
    ax2.set_ylabel(r"dual price $p_t$", color="tab:red")
    ax.set_xlabel("t")
    ax.set_ylabel("cumulative cost")
    ax.set_title("Single-price router: realized spend against the budget line")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "budget_tracking.png", dpi=150)


def plot_surrogate() -> None:
    rows = read("surrogate_bias.csv")
    biases = sorted({float(r["bias_b"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for b in biases:
        sub = sorted((r for r in rows if float(r["bias_b"]) == b),
                     key=lambda r: int(r["t"]))
        t = [int(r["t"]) for r in sub]
        axes[0].plot(t, [float(r["regret_vs_surrogate"]) for r in sub], label=f"b={b}")
        axes[1].plot(t, [float(r["regret_vs_true"]) for r in sub], label=f"b={b}")
    for ax, title in zip(axes, ("regret vs the surrogate optimum\n(what the policy sees)",
                                "regret vs the TRUE optimum\n(what actually happens)")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel("cumulative regret")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "surrogate_bias.png", dpi=150)


def plot_compounding() -> None:
    rows = read("compounding.csv")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for policy in sorted({r["policy"] for r in rows}):
        for regime, style in (("independent", "--"), ("correlated", "-")):
            sub = sorted((r for r in rows if r["policy"] == policy
                          and r["regime"] == regime), key=lambda r: int(r["n_steps"]))
            if not sub:
                continue
            ax.plot([int(r["n_steps"]) for r in sub],
                    [float(r["measured_end_to_end"]) for r in sub], style,
                    marker="o", ms=4, label=f"{policy} ({regime}) measured")
            ax.plot([int(r["n_steps"]) for r in sub],
                    [float(r["predicted_end_to_end"]) for r in sub], ":",
                    lw=0.8, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("steps per episode")
    ax.set_ylabel("end-to-end success")
    ax.set_title(r"Measured end-to-end vs the $p^n$ prediction (dotted)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "compounding.png", dpi=150)


if __name__ == "__main__":
    plot_regret()
    plot_coverage()
    plot_pareto()
    plot_budget()
    plot_surrogate()
    plot_compounding()
    print(f"wrote figures to {RESULTS}")
