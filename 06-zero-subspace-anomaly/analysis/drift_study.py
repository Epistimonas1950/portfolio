#!/usr/bin/env python3
"""Orthogonality drift over a long stream, with and without periodic repair.

BRIEF.md section 3 asks for exactly this figure. It is the cheapest experiment in the
repo and the one that shows whether the person who wrote the tracker knows what floating
point does to a basis that is rebuilt twenty thousand times.

The stream is data/normal.csv replayed `REPEATS` = 15 times, for 22 200 rank-one updates
(the first 300 samples build the basis and are scored without updating it, so updates and
output rows are not the same count -- the C reports both). Drift is a function of the
NUMBER OF RANK-ONE UPDATES, not of how much distinct data there was, so replaying a
1 500-sample file fifteen times measures the same thing as a 22 500-sample file and keeps
data/ small. Forgetting is switched OFF (lambda = 1) on purpose: with lambda < 1 old
rounding error decays out of Sigma along with old data, the drift plateaus, and the
accumulation this experiment is about is masked by the very mechanism that would hide
it in production.

Both precisions are run. Single precision is a defensible choice on a 512 MB board, and
the same experiment at two precisions says more than either alone: the shape of the
curve is identical and the scale is set by eps.

Writes results/orthogonality_drift.csv (the traces) and prints the summary the README
quotes.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from oracle.batch_svd import DATA, RESULTS, read_stream                  # noqa: E402
from oracle.chost import (TRACKER, TRACKER32, CDefaults, require,        # noqa: E402
                          run_c_tracker)
from oracle.incremental import CHECK_EVERY, REORTH_TOL, run_stream       # noqa: E402

REPEATS = 15                 # 22 200 rank-one updates on the 1 500-sample stream
STREAM = "normal.csv"
SUBSAMPLE = 50               # thin the trace before writing; the monitor only runs
                             # every CHECK_EVERY samples anyway

# The C and the numpy oracle are the same algorithm and do NOT drift by the same amount.
# One stream would make that a curiosity; four make it a measurement, and the spread
# across them (2.7x to 23x) is itself the reason the README does not attribute it to a
# single step. Traces are written for STREAM only -- four full traces would be 450 kB of
# CSV to make one point that a summary row already makes.
CROSS_STREAMS = (("normal.csv", 15), ("anomalous.csv", 11),
                 ("rotating.csv", 11), ("multimode.csv", 11))


def run_c(binary: pathlib.Path, tol: float, reorth: bool) -> tuple[np.ndarray, dict]:
    opts = CDefaults(lam=1.0, reorth=reorth, reorth_tol=tol, check_every=CHECK_EVERY,
                     repeat=REPEATS)
    res = run_c_tracker(DATA / STREAM, opts, binary=binary)
    return res.drift, res.summary


def main() -> None:
    require(TRACKER)
    RESULTS.mkdir(exist_ok=True)
    rows: list[dict] = []
    summary: list[dict] = []

    builds = [("C double", TRACKER, REORTH_TOL)]
    if TRACKER32.exists():
        # 100 * FLT_EPSILON, the same rule the double build uses; see host/main.c.
        builds.append(("C float32", TRACKER32, 100.0 * float(np.finfo(np.float32).eps)))

    for label, binary, tol in builds:
        for reorth in (True, False):
            drift, info = run_c(binary, tol, reorth)
            tag = f"{label}, {'reorth' if reorth else 'no reorth'}"
            for i in range(0, drift.size, SUBSAMPLE):
                rows.append({"method": tag, "update": i,
                             "drift": f"{drift[i]:.6e}"})
            summary.append({
                "method": tag, "updates": int(info["updates"]),
                "reorth_tol": f"{tol:.3e}",
                "initial_drift": f"{drift[0]:.3e}",
                "final_drift": f"{float(info['final_drift']):.3e}",
                "max_drift": f"{float(info['max_drift']):.3e}",
                "n_reorth": int(info["n_reorth"]),
                "growth_final_over_initial": f"{drift[-1] / drift[0]:.1f}",
            })

    # The numpy oracle over the same stream, so the C's drift can be read as a property
    # of the algorithm rather than of this particular implementation of it.
    x = read_stream(DATA / STREAM)
    for reorth in (True, False):
        res = run_stream(x, lam=1.0, reorth=reorth, reorth_tol=REORTH_TOL,
                         check_every=CHECK_EVERY, repeat=REPEATS)
        tag = f"numpy double, {'reorth' if reorth else 'no reorth'}"
        for i in range(0, res.drift.size, SUBSAMPLE):
            rows.append({"method": tag, "update": i, "drift": f"{res.drift[i]:.6e}"})
        summary.append({
            "method": tag, "updates": int(res.drift.size),
            "reorth_tol": f"{REORTH_TOL:.3e}",
            "initial_drift": f"{res.drift[0]:.3e}",
            "final_drift": f"{res.drift[-1]:.3e}",
            "max_drift": f"{float(res.state.max_drift):.3e}",
            "n_reorth": res.state.n_reorth,
            "growth_final_over_initial": f"{res.drift[-1] / res.drift[0]:.1f}",
        })

    # C against numpy, no repair, on every stream: same algorithm, different arithmetic.
    cross = []
    for name, rep in CROSS_STREAMS:
        c = run_c_tracker(DATA / name,
                          CDefaults(lam=1.0, reorth=False, reorth_tol=REORTH_TOL,
                                    check_every=CHECK_EVERY, repeat=rep))
        npres = run_stream(read_stream(DATA / name), lam=1.0, reorth=False, repeat=rep)
        c_late = float(np.median(c.drift[-c.drift.size // 10:]))
        np_late = float(np.median(npres.drift[-npres.drift.size // 10:]))
        cross.append({"stream": name, "updates": int(c.summary["updates"]),
                      "drift_c_late_median": f"{c_late:.4e}",
                      "drift_numpy_late_median": f"{np_late:.4e}",
                      "numpy_over_c": round(np_late / c_late, 2)})
        print(f"{name:16s} no repair: C={c_late:.3e}  numpy={np_late:.3e}  "
              f"numpy/C={np_late / c_late:.1f}x")
    out3 = RESULTS / "drift_c_vs_numpy.csv"
    with out3.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(cross[0]))
        writer.writeheader()
        writer.writerows(cross)

    out = RESULTS / "orthogonality_drift.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["method", "update", "drift"])
        writer.writeheader()
        writer.writerows(rows)

    out2 = RESULTS / "orthogonality_drift_summary.csv"
    with out2.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    for s in summary:
        print(f"{s['method']:28s} final={s['final_drift']}  max={s['max_drift']}  "
              f"repairs={s['n_reorth']:5d}  growth={s['growth_final_over_initial']}x")
    print(f"\nwrote {out}\nwrote {out2}\nwrote {out3}")


if __name__ == "__main__":
    main()
