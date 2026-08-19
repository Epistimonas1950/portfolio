#!/usr/bin/env python3
"""Does the C agree with the numpy oracle? Writes results/c_vs_python.csv.

The C in src/ is the deliverable and numpy is the reference. This script runs both over
the same streams with the same parameters and reports the disagreement, in the two
quantities that actually matter:

`score`     max |s_C - s_numpy| over every sample, absolute and relative to the largest
            score. The detector's output is the score, so this is the end-to-end check.
`subspace`  || U_C U_C^T - U_np U_np^T ||_F and the largest principal angle between the
            two tracked subspaces. NOT an elementwise comparison of U: the factors of an
            SVD are defined only up to an r x r rotation within each singular subspace
            and a sign per column, so an elementwise test would fail on a correct
            implementation and pass on a wrong one that happened to agree on signs. Both
            quantities used here are invariant under that freedom.

WHY THE AGREEMENT IS ~1e-7 AND NOT ~1e-15, stated up front because it looks like a bug
and is not. The two implementations draw their Gaussian sketch from different generators
-- PCG32 in the C, PCG64 in numpy -- so their initial subspaces are two different draws
of the same randomized algorithm, agreeing only to sketch accuracy. The rank-one updates
then converge from those two starting points to nearby but not identical subspaces. What
this comparison establishes is that the C implements the same algorithm to well within
the algorithm's own approximation error; it is not a bit-reproducibility test, and
claiming it as one would be the dishonest reading.

The float32 build is compared too. Its disagreement with the double oracle is a
measurement of what single precision costs, which is the number a person deciding what
to run on a 512 MB board would want.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from oracle.batch_svd import DATA, RESULTS, read_stream                  # noqa: E402
from oracle.chost import (TRACKER, TRACKER32, CDefaults, require,        # noqa: E402
                          run_c_rangefinder, run_c_tracker)
from oracle.incremental import (CHECK_EVERY, REORTH_TOL,                 # noqa: E402
                                principal_angles, projection_distance, run_stream)
from oracle.rangefinder import (projection_error,                        # noqa: E402
                                randomized_range_finder)

LAMBDA = float(np.sqrt(1.0 - 1.0 / 400.0))
STREAMS = ("normal.csv", "anomalous.csv", "multimode.csv", "manymode.csv",
           "rotating.csv")


def compare_stream(name: str, binary: pathlib.Path, tag: str) -> dict:
    path = DATA / name
    x = read_stream(path)
    opts = CDefaults(lam=LAMBDA, reorth=True, reorth_tol=REORTH_TOL,
                     check_every=CHECK_EVERY)
    c = run_c_tracker(path, opts, binary=binary, want_basis=True)
    npres = run_stream(x, lam=LAMBDA, reorth=True, reorth_tol=REORTH_TOL,
                       check_every=CHECK_EVERY)

    d_score = np.abs(c.scores - npres.scores)
    scale = float(np.max(npres.scores))
    angles = principal_angles(npres.state.u, c.basis)
    return {
        "stream": name,
        "build": tag,
        "rank_c": c.rank,
        "rank_numpy": npres.r,
        "max_abs_score_diff": f"{float(np.max(d_score)):.3e}",
        "max_rel_score_diff": f"{float(np.max(d_score) / scale):.3e}",
        "mean_abs_score_diff": f"{float(np.mean(d_score)):.3e}",
        "subspace_projection_distance": f"{projection_distance(npres.state.u, c.basis):.3e}",
        "max_principal_angle_deg": f"{float(np.degrees(angles.max())):.3e}",
        "threshold_c": f"{c.threshold:.8g}",
        "threshold_numpy": f"{npres.threshold:.8g}",
        "final_drift_c": f"{c.final_drift:.3e}",
        "final_drift_numpy": f"{float(npres.drift[-1]):.3e}",
        "us_per_sample_x86_host": f"{float(c.summary['us_per_sample']):.3f}",
    }


def compare_against_batch() -> list[dict]:
    """The correctness anchor, as a committed artifact rather than a number in prose.

    With lambda = 1, Brand's update is exact: streaming n samples must land on the same
    subspace AND the same singular values a full SVD of all n finds. Both are reported,
    for both C builds and for the numpy oracle, and both basis-independently.
    """
    rows = []
    for name in ("normal.csv", "anomalous.csv"):
        x = read_stream(DATA / name)
        u_batch, s_batch, _ = np.linalg.svd(x, full_matrices=False)
        impls: list[tuple[str, np.ndarray, np.ndarray | None]] = []
        npres = run_stream(x, lam=1.0, reorth=True, reorth_tol=REORTH_TOL)
        impls.append(("numpy double", npres.state.u, npres.state.sigma))
        for tag, binary, tol in (("C double", TRACKER, REORTH_TOL),
                                 ("C float32", TRACKER32,
                                  100.0 * float(np.finfo(np.float32).eps))):
            if not binary.exists():
                continue
            c = run_c_tracker(DATA / name, CDefaults(lam=1.0, reorth=True,
                                                     reorth_tol=tol),
                              binary=binary, want_basis=True)
            sig = np.array([float(c.summary[f"sigma{i}"]) for i in range(c.rank)])
            impls.append((tag, c.basis, sig))
        for tag, basis, sigma in impls:
            r = basis.shape[1]
            angle = float(np.degrees(principal_angles(basis, u_batch[:, :r]).max()))
            rel = float(np.max(np.abs(sigma - s_batch[:r]) / s_batch[:r]))
            rows.append({
                "stream": name, "implementation": tag, "rank": r,
                "max_principal_angle_deg_vs_batch": f"{angle:.4e}",
                "projection_distance_vs_batch":
                    f"{projection_distance(basis, u_batch[:, :r]):.4e}",
                "max_relative_sigma_error_vs_batch": f"{rel:.4e}",
            })
    return rows


def compare_rangefinder() -> list[dict]:
    """The C range finder against the numpy one on the same warm-up block.

    Both are randomized and seeded independently, so the comparison is between two
    draws: what must agree is the projection error, not the basis.
    """
    rows = []
    rng = np.random.default_rng(0)
    for name in ("normal.csv", "multimode.csv"):
        path = DATA / name
        x = read_stream(path)[:, :300]
        for p, q in ((6, 1), (6, 2), (0, 1)):
            c_err = run_c_rangefinder(path, rank=8, oversampling=p, power_iters=q,
                                      seed=0, n_cols=300)
            ell = min(8 + p, x.shape[0], x.shape[1])
            qmat = randomized_range_finder(x, 8, p, q, rng)
            np_err = projection_error(x, qmat)
            exact = float(np.sqrt(np.sum(
                np.linalg.svd(x, compute_uv=False)[ell:] ** 2)))
            rows.append({
                "stream": name, "sketch_width": ell, "oversampling": p,
                "power_iters": q,
                "proj_error_c": f"{c_err:.8g}",
                "proj_error_numpy": f"{np_err:.8g}",
                "optimal_rank_ell": f"{exact:.8g}",
                "relative_difference": f"{abs(c_err - np_err) / np_err:.3e}",
            })
    return rows


def main() -> None:
    require(TRACKER)
    RESULTS.mkdir(exist_ok=True)

    rows = []
    for name in STREAMS:
        rows.append(compare_stream(name, TRACKER, "double"))
        if TRACKER32.exists():
            rows.append(compare_stream(name, TRACKER32, "float32"))
        for r in rows[-2:]:
            print(f"{r['stream']:16s} {r['build']:8s} "
                  f"rel score diff={r['max_rel_score_diff']}  "
                  f"subspace dist={r['subspace_projection_distance']}")

    out = RESULTS / "c_vs_python.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")

    batch_rows = compare_against_batch()
    out = RESULTS / "batch_agreement.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(batch_rows[0]))
        writer.writeheader()
        writer.writerows(batch_rows)
    for r in batch_rows:
        print(f"{r['stream']:16s} {r['implementation']:12s} vs batch SVD: "
              f"angle={r['max_principal_angle_deg_vs_batch']} deg  "
              f"sigma rel={r['max_relative_sigma_error_vs_batch']}")
    print(f"wrote {out}")

    rf_rows = compare_rangefinder()
    out = RESULTS / "c_vs_python_rangefinder.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rf_rows[0]))
        writer.writeheader()
        writer.writerows(rf_rows)
    for r in rf_rows:
        print(f"{r['stream']:16s} p={r['oversampling']} q={r['power_iters']}  "
              f"C={r['proj_error_c']}  numpy={r['proj_error_numpy']}")
    print(f"wrote {out}")

    # The per-sample output of the C on the labelled stream, kept as a committed
    # artifact: it is the file the README's drift and score claims come from.
    run_c_tracker(DATA / "anomalous.csv",
                  CDefaults(lam=LAMBDA, reorth=True, reorth_tol=REORTH_TOL),
                  out_csv=RESULTS / "scores_anomalous_c.csv")
    print(f"wrote {RESULTS / 'scores_anomalous_c.csv'}")


if __name__ == "__main__":
    main()
