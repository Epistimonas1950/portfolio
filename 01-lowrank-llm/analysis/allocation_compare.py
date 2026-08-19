#!/usr/bin/env python3
"""Four rank allocators at several budgets, scored against the true optimum.

The stack (src/synth.py, STACK_SHAPES) is deliberately heterogeneous: six layers whose
shapes differ, so a unit of rank costs a different number of parameters in each, and
whose activation conditioning spans four decades, so a unit of rank buys a different
amount of error reduction in each. Uniform compression cannot be right on a stack like
that, and this script measures by how much.

Two numbers are reported for every allocation:

  proxy_loss     sum_l sum_{i > r_l} sigma_{l,i}^2, the objective the allocators
                 actually minimize, read off the whitened spectra
  measured_sq_error   sum_l ||(W_l - W_hat_l) X_l||_F^2, obtained by actually
                 factorizing every layer at its allocated rank and multiplying out

At zero ridge these are the same quantity -- that is the whitening identity -- so the
two columns agreeing to the ridge's precision is the check that the allocation
objective is the real error and not a stand-in for it. They are printed side by side
rather than asserted in prose.

Writes results/allocation.csv.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.allocate import (STRATEGIES, LayerSpec, allocate_knapsack_dp,  # noqa: E402
                          total_dense)
from src.factorize import (activation_error, whitened_spectrum,          # noqa: E402
                           whitened_svd)
from src.rebuild import report_layer, stack_compression                 # noqa: E402
from src.synth import make_stack                                        # noqa: E402
from src.whiten import whiten                                           # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
N_SAMPLES = 512
RIDGE = 1e-6            # near-zero, so the allocation objective and the measured
                        # squared error are the same number up to lambda ||E||^2
SEED = 0
BUDGET_FRACTIONS = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def run() -> list[dict]:
    stack = make_stack(n_samples=N_SAMPLES, seed=SEED)
    specs, whitenings = [], []
    for layer in stack:
        sigma, wh = whitened_spectrum(layer.w, layer.x, ridge=RIDGE)
        specs.append(LayerSpec(name=layer.name, m=layer.w.shape[0],
                               n=layer.w.shape[1], sigma=sigma))
        whitenings.append(wh)

    dense = total_dense(specs)
    denom_sq = sum(float(np.linalg.norm(l.w @ l.x)) ** 2 for l in stack)
    print(f"stack: {len(specs)} layers, {dense} dense parameters\n")

    rows: list[dict] = []
    for frac in BUDGET_FRACTIONS:
        budget = int(frac * dense)
        optimum = allocate_knapsack_dp(specs, budget).loss
        for name, solve in STRATEGIES.items():
            alloc = solve(specs, budget)

            # Actually build every layer at its allocated rank and measure, rather
            # than trusting the spectral tail that chose the rank.
            measured_sq, reports = 0.0, []
            for layer, spec, wh, r in zip(stack, specs, whitenings, alloc.ranks):
                fac = whitened_svd(layer.w, layer.x, r, whitening=wh)
                err = activation_error(layer.w, fac.w_hat, layer.x)
                measured_sq += err ** 2
                reports.append(report_layer(spec.name, spec.m, spec.n, r,
                                            err / float(np.linalg.norm(layer.w @ layer.x))))
            rows.append({
                "budget_fraction": frac,
                "budget_params": budget,
                "dense_params": dense,
                "strategy": name,
                "params_used": alloc.params,
                "leftover_params": alloc.leftover_params,
                "achieved_compression": round(stack_compression(reports), 4),
                "proxy_loss": f"{alloc.loss:.8e}",
                "measured_sq_error": f"{measured_sq:.8e}",
                "proxy_over_measured": round(alloc.loss / measured_sq, 6),
                "rel_error_stack": round(float(np.sqrt(measured_sq / denom_sq)), 6),
                "gap_vs_optimum_pct": round(100.0 * (alloc.loss - optimum) / optimum, 4),
                "ranks": "|".join(str(r) for r in alloc.ranks),
            })
            print(f"  B={frac:.2f}  {name:12s} loss={alloc.loss:.6g} "
                  f"gap={rows[-1]['gap_vs_optimum_pct']:+7.3f}%  "
                  f"rel_err={rows[-1]['rel_error_stack']:.4f}  "
                  f"leftover={alloc.leftover_params:5d}  ranks={rows[-1]['ranks']}")
        print()
    return rows


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = run()
    out = RESULTS / "allocation.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
