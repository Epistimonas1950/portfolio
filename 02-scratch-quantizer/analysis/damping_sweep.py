#!/usr/bin/env python3
"""The damping tradeoff, measured rather than asserted.

H = 2 X X^T is singular whenever an input channel is not excited by the calibration
set -- and in a real layer several always are not. The ridge H + lambda*mean(diag H)*I
buys a factorization, and the sweep below shows what it costs:

  lambda too small   Cholesky fails outright, or succeeds on a matrix whose condition
                     number makes the compensation direction numerically meaningless
  lambda too large   H is pushed toward a multiple of the identity, H^{-1} toward a
                     multiple of the identity too, and the compensation term toward
                     zero -- the method degrades continuously into round-to-nearest

Writes results/damping_sweep.csv, including the failures, which are the informative
end of the range.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.grid import Grid                            # noqa: E402
from src.hessian import damp, hessian                # noqa: E402
from src.rtn import rtn_quantize                     # noqa: E402
from src.sequential import output_error, sequential_quantize  # noqa: E402
from src.synth import make_layer                     # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
RATIOS = [0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    grid = Grid(3)
    rows = []
    for ratio in RATIOS:
        errs, conds, failures = [], [], 0
        for seed in range(5):
            # 192 calibration samples for 256 input channels: X X^T has rank at most 192 and
            # is therefore singular by construction -- the undamped case must fail, and does.
            layer = make_layer(n_out=64, n_in=256, n_samples=192, cond=1e8,
                               n_outliers=0, seed=seed)
            try:
                res = sequential_quantize(layer.w, layer.x, grid, damping=ratio)
            except np.linalg.LinAlgError:
                failures += 1
                continue
            errs.append(res.output_error)
            conds.append(res.condition_number)
        rtn_ref = []
        for seed in range(5):
            layer = make_layer(n_out=64, n_in=256, n_samples=192, cond=1e8,
                               n_outliers=0, seed=seed)
            rtn_ref.append(output_error(layer.w, rtn_quantize(layer.w, grid), layer.x))
        rows.append({
            "damping_ratio": ratio,
            "cholesky_failures": failures,
            "condition_number_mean": f"{np.mean(conds):.4e}" if conds else "",
            "output_error_mean": round(float(np.mean(errs)), 6) if errs else "",
            "rtn_reference": round(float(np.mean(rtn_ref)), 6),
            "improvement_over_rtn": round(float(np.mean(rtn_ref) / np.mean(errs)), 4)
            if errs else "",
        })
        print(f"lambda_ratio={ratio:<8g} failures={failures}  "
              f"err={rows[-1]['output_error_mean']}  "
              f"x_over_rtn={rows[-1]['improvement_over_rtn']}")

    out = RESULTS / "damping_sweep.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
