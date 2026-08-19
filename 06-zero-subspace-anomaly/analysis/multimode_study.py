#!/usr/bin/env python3
"""When several operating modes break the detector, and what it takes to fix it.

This is the experiment behind the README's "limitation I volunteer first", and it is
here because the first version of that limitation was wrong. BRIEF.md says a single
subspace blurs several operating modes and the detector degrades. It does -- but on the
two-mode stream, changing the rank criterion from energy(0.95) to the singular-value gap
takes the AUC from 0.65 back to 0.99, so the failure there is a rank-selection failure
wearing the costume of a modelling failure. Reporting only that scenario would have
overstated the limitation in one direction and understated the fix.

So the sweep below varies the two things the tracker actually controls, over both the
two-mode and the four-mode stream, and reports the whole surface:

`r`       the tracked rank. The escape route is to make the subspace big enough to
          contain the UNION of the modes. Its cost is that the detector measures energy
          in the orthogonal complement, whose dimension is m - r: raising r shrinks the
          space in which an anomaly can be seen at all, and at r = m every score is zero.
`N_eff`   the effective forgetting window. The other escape route is to make it SHORT
          relative to the mode dwell time, so the tracker locks onto whichever mode is
          currently running instead of straddling all of them. Its costs are that the
          subspace becomes noisier (results/forgetting.csv shows the resulting U-shape)
          and that it requires knowing the dwell time in advance -- which is the
          mode structure you were claiming not to know.

The batch full-SVD oracle is swept over r as well, which separates the two mechanisms:
where the batch oracle recovers at some r and the streaming tracker does not, the
failure is a tracking-timescale failure; where neither recovers, it is the geometry.

Writes results/multimode_sweep.csv.
"""

from __future__ import annotations

import csv
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from oracle.batch_svd import (CALIBRATION, DATA, RESULTS, WARMUP,        # noqa: E402
                              read_labels, read_stream, residual_scores)
from oracle.incremental import run_stream                                # noqa: E402
from oracle.roc import _restrict, auc                                    # noqa: E402

SCENARIOS = (
    ("unimodal", "anomalous.csv", "labels.csv"),
    ("multimode (2 modes, dwell 150)", "multimode.csv", "multimode_labels.csv"),
    ("manymode (4 modes, dwell 100)", "manymode.csv", "manymode_labels.csv"),
)
RANKS = (4, 5, 6, 8, 10, 13, 16)
WINDOWS = (None, 400, 100, 50)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for scenario, stream_name, label_name in SCENARIOS:
        x = read_stream(DATA / stream_name)
        _, kind = read_labels(DATA / label_name)
        if "spike" not in set(kind):
            continue
        block = x[:, :WARMUP + CALIBRATION]
        u_batch, _, _ = np.linalg.svd(block, full_matrices=False)

        for r in RANKS:
            # Batch oracle at a forced rank: no tracking, no forgetting, exact SVD.
            s, y = _restrict(residual_scores(x, u_batch[:, :r]), kind, "spike")
            batch_auc = auc(s, y)
            for n_eff in WINDOWS:
                lam = 1.0 if n_eff is None else math.sqrt(1.0 - 1.0 / n_eff)
                # r_max = r with the energy criterion at 1.0 forces exactly rank r.
                res = run_stream(x, lam=lam, r_max=r, oversampling=6, energy=1.0,
                                 rank_mode="energy")
                s, y = _restrict(res.scores, kind, "spike")
                rows.append({
                    "scenario": scenario,
                    "rank": r,
                    "rank_actual": res.r,
                    "effective_window": "inf" if n_eff is None else n_eff,
                    "auc_incremental": round(float(auc(s, y)), 5),
                    "auc_batch_same_rank": round(float(batch_auc), 5),
                    "complement_dimension": x.shape[0] - r,
                })

    out = RESULTS / "multimode_sweep.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for scenario, _, _ in SCENARIOS:
        sub = [r for r in rows if r["scenario"] == scenario]
        if not sub:
            continue
        best = max(sub, key=lambda r: r["auc_incremental"])
        best_batch = max(sub, key=lambda r: r["auc_batch_same_rank"])
        print(f"{scenario:32s} best incremental AUC={best['auc_incremental']:.4f} "
              f"at r={best['rank']}, N_eff={best['effective_window']}"
              f"   | best batch AUC={best_batch['auc_batch_same_rank']:.4f} "
              f"at r={best_batch['rank']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
