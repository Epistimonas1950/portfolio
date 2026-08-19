#!/usr/bin/env python3
"""Does exponential forgetting actually track a moving subspace? Sweep lambda and see.

data/rotating.csv turns its subspace through 90 degrees over 2 000 samples, so the final
subspace is ORTHOGONAL to the initial one. A tracker with lambda = 1 weights the first
sample as heavily as the last and cannot be right at both ends; a tracker with a finite
effective window can. That is the whole claim of BRIEF.md section 4, and it is either
true of this code or it is not.

Two metrics, because they fail in different directions:

`angle_to_truth`   the largest principal angle between the tracked subspace and the
                   TRUE subspace at the last sample, which oracle/generate_data.py can
                   return exactly because it built it. This is the honest measure of
                   tracking, and it is basis-independent.
`mean_residual`    the mean normalized residual energy over the last 200 samples. This
                   is what the detector actually sees: a stale subspace inflates the
                   normal score distribution, which is how a tracking failure turns into
                   a detection failure.

The sweep is over the EFFECTIVE WINDOW N_eff, not over lambda, because N_eff is the
quantity with a meaning -- lambda = sqrt(1 - 1/N_eff), derived in src/forget.h. Reporting
lambda = 0.99875 without saying it means "400 samples" is the usual way this parameter
becomes folklore.

Both the numpy oracle and the compiled C are run. Writes results/forgetting.csv.
"""

from __future__ import annotations

import csv
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from oracle.batch_svd import DATA, RESULTS, read_stream, residual_scores  # noqa: E402
from oracle.chost import TRACKER, CDefaults, require, run_c_tracker       # noqa: E402
from oracle.generate_data import make_rotating                            # noqa: E402
from oracle.incremental import (CHECK_EVERY, REORTH_TOL,                  # noqa: E402
                                principal_angles, run_stream)

WINDOWS = (None, 5000, 1000, 400, 200, 100, 50)     # None means lambda = 1
TAIL = 200


def lam_from_window(n_eff: float | None) -> float:
    """lambda = sqrt(1 - 1/N_eff); N_eff = None is 'remember everything'."""
    return 1.0 if n_eff is None else math.sqrt(1.0 - 1.0 / n_eff)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    stream = make_rotating()                     # same seed as the generator wrote out
    x = read_stream(DATA / "rotating.csv")
    truth = stream.basis_final                   # exact subspace at the last sample
    have_c = TRACKER.exists()
    if not have_c:
        print("note: build/tracker not found -- C columns omitted. Run `make host`.")

    rows = []
    for n_eff in WINDOWS:
        lam = lam_from_window(n_eff)
        res = run_stream(x, lam=lam, reorth=True, reorth_tol=REORTH_TOL,
                         check_every=CHECK_EVERY)
        angle = float(np.degrees(principal_angles(res.state.u, truth[:, :res.r]).max()))
        resid = float(np.mean(residual_scores(x[:, -TAIL:], res.state.u)))

        c_angle = ""
        c_resid = ""
        if have_c:
            c = run_c_tracker(DATA / "rotating.csv",
                              CDefaults(lam=lam, reorth=True, reorth_tol=REORTH_TOL,
                                        check_every=CHECK_EVERY),
                              want_basis=True)
            c_angle = round(float(np.degrees(
                principal_angles(c.basis, truth[:, :c.rank]).max())), 4)
            c_resid = round(float(np.mean(residual_scores(x[:, -TAIL:], c.basis))), 8)

        rows.append({
            "effective_window": "inf" if n_eff is None else n_eff,
            "lambda": round(lam, 8),
            "half_life_samples": ("inf" if lam >= 1.0 else
                                  round(math.log(0.5) / (2.0 * math.log(lam)), 1)),
            "rank": res.r,
            "angle_to_truth_deg_numpy": round(angle, 4),
            "mean_residual_tail_numpy": round(resid, 8),
            "angle_to_truth_deg_c": c_angle,
            "mean_residual_tail_c": c_resid,
        })
        print(f"N_eff={str(rows[-1]['effective_window']):>5s} lambda={lam:.6f}  "
              f"angle={angle:7.3f} deg  tail residual={resid:.6f}"
              + (f"   [C: {c_angle} deg, {c_resid}]" if have_c else ""))

    out = RESULTS / "forgetting.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
