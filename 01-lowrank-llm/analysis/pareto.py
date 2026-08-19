#!/usr/bin/env python3
"""Compression against activation-weighted error: plain SVD versus whitened SVD.

Produces results/pareto.csv -- the table the README leads with. Three things are
swept, because three things are being claimed:

  regime   anisotropic activations (cond 1e5), where whitening is supposed to win,
           and the isotropic control (cond 1.0), where the advantage must collapse
           because M is already a multiple of the identity. The control is not a
           formality; it is the check that the gain comes from the advertised
           mathematics rather than from somewhere else.
  ridge    the dimensionless damping ratio. Whitened SVD at a large ridge must
           degrade continuously into plain truncated SVD -- that is what
           ||E S||^2 = ||E X||^2 + lambda ||E||^2 predicts, and the last rows of the
           table are the test of it.
  rank     only ranks below the break-even point r < mn/(m+n), so every row in the
           file is a genuine compression rather than an expansion.

Each factorization is scored four ways: on the calibration activations it was fitted
to, on a fresh draw from the same distribution, and on two shifted distributions that
keep the covariance spectrum but rotate its eigenbasis half way and all the way. The
gap between the first two is calibration overfitting. The last two columns price the
distribution-shift risk that whitening buys its accuracy with -- there is a mix at
which plain SVD, which never looked at any activations, wins.

Writes CSV unconditionally; plotting lives in analysis/plot_results.py so the numbers
exist whether or not matplotlib does.
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.factorize import (plain_truncated_svd, relative_activation_error,  # noqa: E402
                           weight_error, whitened_svd)
from src.rebuild import break_even_rank, compression_ratio, factored_params  # noqa: E402
from src.synth import make_layer, shifted_activations                       # noqa: E402
from src.whiten import whiten                                               # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

M_OUT, N_IN = 256, 256
N_CALIB = 512          # > n_in, so the undamped second moment is positive definite
N_EVAL = 768
W_COND = 10.0          # one decade of decay in the weight spectrum; see src/synth.py
SEEDS = range(5)
RANKS = (8, 16, 32, 48, 64, 96, 112)
REGIMES = (("anisotropic", 1e5), ("isotropic", 1.0))
RIDGES = (1e-6, 1e-4, 1e-2, 1.0)

#: (column name, covariance mixing fraction, rng tag). mix=0 is a fresh draw from the
#: calibration distribution, so holdout-vs-calib isolates calibration overfitting;
#: mix=0.5 and 1.0 rotate the covariance eigenbasis half way and all the way at
#: matched spectrum, which is what a domain change looks like to this method.
EVAL_SETS = (("holdout", 0.0, 9000), ("shift50", 0.5, 7000), ("shift100", 1.0, 5000))


def run() -> list[dict]:
    dense = M_OUT * N_IN
    cap = break_even_rank(M_OUT, N_IN)
    ranks = [r for r in RANKS if r <= cap]
    if len(ranks) != len(RANKS):
        print(f"dropped ranks above the break-even rank {cap}")

    rows: list[dict] = []
    for regime, cond in REGIMES:
        # One draw per seed, reused by every method and rank so the comparison is
        # paired rather than merely averaged.
        layers, evals, whitenings = [], [], []
        for seed in SEEDS:
            layer = make_layer(n_out=M_OUT, n_in=N_IN, n_samples=N_CALIB, cond=cond,
                               w_cond=W_COND, seed=seed)
            layers.append(layer)
            evals.append({name: shifted_activations(layer, seed=tag + seed, mix=mix,
                                                    n_samples=N_EVAL)
                          for name, mix, tag in EVAL_SETS})
            whitenings.append({r: whiten(layer.x, r) for r in RIDGES})

        methods: list[tuple[str, float]] = [("plain SVD", float("nan"))]
        methods += [("whitened SVD", r) for r in RIDGES]

        for name, ridge in methods:
            for rank in ranks:
                calib, wrel, conds = [], [], []
                shifted = {name: [] for name, _, _ in EVAL_SETS}
                for i, layer in enumerate(layers):
                    if name == "plain SVD":
                        fac = plain_truncated_svd(layer.w, rank)
                        conds.append(float("nan"))
                    else:
                        wh = whitenings[i][ridge]
                        fac = whitened_svd(layer.w, layer.x, rank, whitening=wh)
                        conds.append(wh.cond_raw)
                    calib.append(relative_activation_error(layer.w, fac.w_hat, layer.x))
                    for set_name, _, _ in EVAL_SETS:
                        shifted[set_name].append(
                            relative_activation_error(layer.w, fac.w_hat,
                                                      evals[i][set_name]))
                    wrel.append(weight_error(layer.w, fac.w_hat)
                                / float(np.linalg.norm(layer.w)))
                rows.append({
                    "regime": regime,
                    "cond_target": cond,
                    "method": name,
                    "ridge_ratio": "" if name == "plain SVD" else ridge,
                    "rank": rank,
                    "params_dense": dense,
                    "params_factored": factored_params(M_OUT, N_IN, rank),
                    "compression_ratio": round(compression_ratio(M_OUT, N_IN, rank), 4),
                    "rel_error_calib_mean": round(float(np.mean(calib)), 6),
                    "rel_error_calib_std": round(float(np.std(calib)), 6),
                    "rel_error_holdout_mean": round(float(np.mean(shifted["holdout"])), 6),
                    "rel_error_shift50_mean": round(float(np.mean(shifted["shift50"])), 6),
                    "rel_error_shift100_mean": round(float(np.mean(shifted["shift100"])), 6),
                    "rel_weight_error_mean": round(float(np.mean(wrel)), 6),
                    "cond_m_mean": "" if name == "plain SVD" else f"{np.mean(conds):.4e}",
                    "n_seeds": len(SEEDS),
                })
                tag = name if name == "plain SVD" else f"{name} ridge={ridge:g}"
                print(f"{regime:12s} {tag:28s} r={rank:3d} "
                      f"comp={rows[-1]['compression_ratio']:6.2f}x "
                      f"calib={rows[-1]['rel_error_calib_mean']:.4f} "
                      f"holdout={rows[-1]['rel_error_holdout_mean']:.4f} "
                      f"shift50={rows[-1]['rel_error_shift50_mean']:.4f} "
                      f"shift100={rows[-1]['rel_error_shift100_mean']:.4f}")
    return rows


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = run()
    out = RESULTS / "pareto.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
