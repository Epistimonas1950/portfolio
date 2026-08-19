#!/usr/bin/env python3
"""What oversampling and power iterations actually buy, measured against the optimum.

Two different errors are reported for each (p, q), and confusing them is the standard
way this experiment is misread:

`sketch_error`     || A - Q Q^T A ||_F with Q having ell = k + p columns. This is the
                   quantity the Halko-Martinsson-Tropp bound is stated for, so it is the
                   one to compare against `frobenius_bound`. It can legitimately fall
                   BELOW the optimal rank-k error, because a rank-(k+p) projection is
                   being compared against the best rank-k one.
`truncated_error`  || A - U_k U_k^T A ||_F after truncating the sketch's SVD back to k
                   columns. This is apples to apples: it is bounded below by the
                   Eckart-Young optimum and approaching 1.0 is the thing to watch.

The test matrix is 64 x 800 with sigma_j = j^-1. A slow decay is essential: on a sharply
decaying spectrum even p = 0 is near-optimal, and the experiment would "pass" while
measuring nothing. The tail is what oversampling and power iteration are for.

The C range finder is run on the same matrix at each setting and its sketch error is
recorded alongside. It uses a different random stream (PCG32 vs numpy's PCG64), so it is
a different draw of the same algorithm, not a bit-for-bit check -- see oracle/compare.py.

Writes results/rangefinder.csv.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from oracle.chost import BUILD, TRACKER, require, run_c_rangefinder      # noqa: E402
from oracle.batch_svd import RESULTS                                     # noqa: E402
from oracle.rangefinder import (frobenius_bound, optimal_error,          # noqa: E402
                                projection_error, randomized_range_finder,
                                randomized_svd)

RANK = 8
ROWS, COLS = 64, 800
SEEDS = range(20)
OVERSAMPLING = (0, 2, 6, 16)
POWER_ITERS = (0, 1, 2)


def test_matrix(decay: float = 1.0, seed: int = 0) -> np.ndarray:
    """64 x 800 with sigma_j = j^-decay and random orthonormal factors."""
    rng = np.random.default_rng(seed)
    left, _ = np.linalg.qr(rng.normal(size=(ROWS, ROWS)))
    right, _ = np.linalg.qr(rng.normal(size=(COLS, ROWS)))
    sigma = np.arange(1, ROWS + 1, dtype=float) ** (-decay)
    return (left * sigma) @ right.T


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    BUILD.mkdir(exist_ok=True)
    a = test_matrix()
    opt = optimal_error(a, RANK)

    # The C reads one sample per row, so the file is A transposed. Written to build/
    # rather than data/ because it is an intermediate this script regenerates, not an
    # input anyone needs to keep.
    matrix_csv = BUILD / "rangefinder_matrix.csv"
    header = ",".join(f"ch{i:02d}" for i in range(ROWS))
    np.savetxt(matrix_csv, a.T, delimiter=",", header=header, comments="", fmt="%.10g")
    have_c = TRACKER.exists()
    if not have_c:
        print("note: build/tracker not found -- C column omitted. Run `make host`.")

    rows = []
    for q in POWER_ITERS:
        for p in OVERSAMPLING:
            sketch, trunc = [], []
            for seed in SEEDS:
                rng = np.random.default_rng(1000 + seed)
                qmat = randomized_range_finder(a, RANK, p, q, rng)
                sketch.append(projection_error(a, qmat))
                rng = np.random.default_rng(1000 + seed)
                u, _ = randomized_svd(a, RANK, p, q, rng)
                trunc.append(projection_error(a, u[:, :RANK]))
            bound = frobenius_bound(a, RANK, p)
            c_err = (run_c_rangefinder(matrix_csv, RANK, p, q, seed=0)
                     if have_c else float("nan"))
            rows.append({
                "rank": RANK, "oversampling": p, "power_iters": q,
                "sketch_width": min(RANK + p, ROWS, COLS),
                "n_seeds": len(SEEDS),
                "optimal_rank_k": round(opt, 8),
                "sketch_error_mean": round(float(np.mean(sketch)), 8),
                "sketch_error_over_optimal": round(float(np.mean(sketch)) / opt, 5),
                "truncated_error_mean": round(float(np.mean(trunc)), 8),
                "truncated_over_optimal": round(float(np.mean(trunc)) / opt, 5),
                # The bound is an EXPECTATION bound and requires p >= 2; it is reported
                # as inf for p < 2 rather than extrapolated outside its hypothesis.
                "hmt_frobenius_bound": (round(bound, 8) if np.isfinite(bound) else ""),
                "bound_over_optimal": (round(bound / opt, 5) if np.isfinite(bound) else ""),
                "sketch_error_c": (round(c_err, 8) if have_c else ""),
            })
            print(f"q={q} p={p:2d}  sketch/opt={rows[-1]['sketch_error_over_optimal']:.4f}"
                  f"  truncated/opt={rows[-1]['truncated_over_optimal']:.4f}"
                  f"  bound/opt={rows[-1]['bound_over_optimal'] or 'n/a (p<2)'}"
                  f"  C sketch={rows[-1]['sketch_error_c']}")

    out = RESULTS / "rangefinder.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
