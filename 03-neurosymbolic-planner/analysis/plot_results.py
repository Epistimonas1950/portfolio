#!/usr/bin/env python3
"""Draw the figures from the CSVs the benchmarks wrote.

Deliberately the only file in this repo that imports matplotlib, which is not installed
on this machine.  Every number these figures show is already committed under results/,
so a reader without matplotlib reproduces every result -- they just do not get pictures.

    python3 -m venv .venv && .venv/bin/pip install matplotlib
    .venv/bin/python analysis/plot_results.py

One panel of the headline figure is deliberately empty: BRIEF.md asks for A* optimality
plotted against a pure-LLM chain-of-thought baseline on the identical instances, and
there is no language model on this machine, so that curve has not been measured.  The
axis is drawn with the note in place rather than filled with a plausible line.
"""

from __future__ import annotations

import csv
import pathlib
import sys

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
FIGURES = RESULTS

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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


def plot_optimality() -> None:
    rows = [r for r in read("optimality.csv") if r["ground_truth"] == "brute-force"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in sorted({r["heuristic"] for r in rows}):
        sub = sorted((r for r in rows if r["heuristic"] == name),
                     key=lambda r: int(r["horizon"]))
        ax.plot([int(r["horizon"]) for r in sub],
                [float(r["optimal_found_pct"]) for r in sub], "o-", label=name)
    ax.set_xlabel("horizon (stops)")
    ax.set_ylabel("plans equal to the exhaustive optimum (%)")
    ax.set_ylim(0, 105)
    ax.set_title("A* optimality, verified against brute force")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2.set_xlim(4, 22)
    ax2.set_ylim(0, 105)
    ax2.axhline(100, color="C0", lw=2, label="A* (proved optimal)")
    ax2.set_xlabel("horizon (stops)")
    ax2.set_ylabel("optimality (%)")
    ax2.set_title("vs a pure-LLM planner")
    ax2.text(13, 50, "no language model on this machine:\nthis curve is NOT measured",
             ha="center", va="center", fontsize=9, style="italic",
             bbox=dict(boxstyle="round", fc="0.92", ec="0.6"))
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "optimality_vs_horizon.png", dpi=150)
    plt.close(fig)


def plot_dominance() -> None:
    rows = read("dominance.csv")
    horizons = sorted({int(r["horizon"]) for r in rows})
    names = sorted({r["heuristic"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name in names:
        ys = []
        for n in horizons:
            sub = [r for r in rows if int(r["horizon"]) == n and r["heuristic"] == name]
            ys.append(sum(int(r["expansions"]) for r in sub) / len(sub))
        ax.semilogy(horizons, ys, "o-", label=name)
    ax.set_xlabel("horizon (stops)")
    ax.set_ylabel("mean node expansions")
    ax.set_title("Heuristic dominance: expansions per heuristic")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "dominance.png", dpi=150)
    plt.close(fig)


def plot_adversarial() -> None:
    rows = read("adversarial.csv")
    depths = sorted({int(r["depth"]) for r in rows})
    variants = ["minimax", "alpha-beta", "alpha-beta+ordering",
                "alpha-beta+ordering+tt"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for v in variants:
        nodes, ebfs = [], []
        for d in depths:
            sub = [r for r in rows if int(r["depth"]) == d and r["variant"] == v]
            nodes.append(sum(int(r["expansions"]) for r in sub) / len(sub))
            ebfs.append(sum(float(r["ebf"]) for r in sub) / len(sub))
        ax.semilogy(depths, nodes, "o-", label=v)
        ax2.plot(depths, ebfs, "o-", label=v)
    b = float(rows[0]["nominal_b"])
    ax2.axhline(b ** 0.5, ls="--", color="k", lw=1, label=f"sqrt(b) = {b ** 0.5:.2f}")
    ax.set_xlabel("search depth (plies)")
    ax.set_ylabel("mean nodes expanded")
    ax.set_title("Alpha-beta pruning")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax2.set_xlabel("search depth (plies)")
    ax2.set_ylabel("effective branching factor b*")
    ax2.set_title("b* against the Knuth-Moore square root")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "adversarial.png", dpi=150)
    plt.close(fig)


def plot_translation() -> None:
    rows = read("translation.csv")
    modes = read("translation_modes.csv")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    groups = ["plain", "hard"]
    vals = []
    for g in groups:
        sub = [r for r in rows if r["difficulty"] == g]
        vals.append(100.0 * sum(int(r["exact"]) for r in sub) / len(sub))
    vals.append(100.0 * sum(int(r["exact"]) for r in rows) / len(rows))
    ax.bar(["in-grammar", "out-of-grammar", "whole corpus"], vals,
           color=["C0", "C3", "C7"])
    ax.set_ylabel("exact-match instance extraction (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Translation accuracy (rule backend, offline)")
    ax.grid(alpha=0.3, axis="y")

    labels = [m["failure_mode"] for m in modes]
    wrong = [int(m["wrong_parses"]) for m in modes]
    lost = [int(m["exact_but_information_lost"]) for m in modes]
    ypos = range(len(labels))
    ax2.barh(list(ypos), wrong, color="C3", label="wrong parse")
    ax2.barh(list(ypos), lost, left=wrong, color="C1",
             label="exact match, information silently lost")
    ax2.set_yticks(list(ypos))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("requests")
    ax2.set_title("Failure taxonomy")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIGURES / "translation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    plot_optimality()
    plot_dominance()
    plot_adversarial()
    plot_translation()
    print(f"wrote 4 figures to {FIGURES}")


if __name__ == "__main__":
    main()
