#!/usr/bin/env python3
"""The cross-layer error bound, executable.

The derivation is in src/propagation.py; this script runs it. A quantized stack and
an exact stack are propagated side by side, and for each depth we record

  local        ||(W_l - W_hat_l) a_{l-1}||_F   the per-layer objective, on clean input
  predicted    the unrolled recursion            an upper bound, provable
  measured     ||a_L - a_hat_L||_F               what actually happens

Writes results/error_propagation.csv. The interesting column is predicted/measured:
it must be >= 1 (or the bound is not a bound) and it should grow with depth, because
the triangle inequality assumes each layer's fresh rounding error points the same way
as the error arriving from below, and independent rounding decisions do not.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.grid import Grid                                       # noqa: E402
from src.propagation import propagate                           # noqa: E402
from src.rtn import rtn_quantize                                # noqa: E402
from src.sequential import sequential_quantize                  # noqa: E402
from src.synth import make_stack                                # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
DEPTH = 10
BITS = (4, 3)


def quantize_stack(weights, x0, grid, method):
    """Quantize layer by layer, feeding each layer the activations it truly sees."""
    hats, activations = [], x0
    for w in weights:
        if method == "rtn":
            hats.append(rtn_quantize(w, grid))
        else:
            hats.append(sequential_quantize(w, activations, grid, damping=1e-3).w_hat)
        activations = w @ activations          # exact stack drives the calibration
    return hats


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    weights, x0 = make_stack(n_layers=DEPTH, width=128, n_samples=256, cond=1e4,
                             seed=1)
    rows = []
    for bits in BITS:
        grid = Grid(bits)
        for method in ("rtn", "sequential"):
            hats = quantize_stack(weights, x0, grid, method)
            for row in propagate(weights, hats, x0):
                rows.append({
                    "bits": bits,
                    "method": method,
                    "layer": row.layer,
                    "local_error": round(row.local_error, 6),
                    "predicted": round(row.predicted, 6),
                    "measured": round(row.measured, 6),
                    "predicted_over_measured": round(row.ratio, 4),
                })
            last = rows[-1]
            print(f"{bits}b {method:11s} depth {DEPTH}: measured={last['measured']:.4f}"
                  f"  bound={last['predicted']:.4f}"
                  f"  looseness={last['predicted_over_measured']:.2f}x")

    out = RESULTS / "error_propagation.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")

    # The compounding statement, in one line, for the README.
    seq = [r for r in rows if r["bits"] == 3 and r["method"] == "sequential"]
    rtn = [r for r in rows if r["bits"] == 3 and r["method"] == "rtn"]
    print(f"\n3-bit, {DEPTH} layers: end-to-end error "
          f"{rtn[-1]['measured'] / seq[-1]['measured']:.2f}x larger with RTN than with "
          "sequential compensation")


if __name__ == "__main__":
    main()
